"""Selection strategies for choosing one CF from a candidate pool.

Each selector receives a list of CandidateObjectives for a single query
and returns the index of the selected candidate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import torch

from src.robustness.score import (
    CandidateObjectives,
    normalize_objectives,
    pareto_front,
)


@dataclass
class SelectorResult:
    """Outcome of applying a selector to one query's candidate pool."""

    selector_name: str
    query_uuid: str
    selected_cf_index: int
    proximity: float
    geometric_instability: float
    intervention_instability: float


# ---------------------------------------------------------------------------
# Selector implementations
# ---------------------------------------------------------------------------


def select_min_proximity(
    candidates: List[CandidateObjectives],
) -> Optional[CandidateObjectives]:
    """Pick the candidate closest to the original query."""
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.proximity)


def select_min_geo(
    candidates: List[CandidateObjectives],
) -> Optional[CandidateObjectives]:
    """Pick the candidate with lowest geometric instability."""
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.geometric_instability)


def select_weighted_sum(
    candidates: List[CandidateObjectives],
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> Optional[CandidateObjectives]:
    """Pick the candidate minimising a weighted sum of *normalised* objectives.

    Args:
        candidates: pool scored with raw objectives.
        weights: (w_proximity, w_geo, w_int).  Will be applied after
                 min-max normalisation across the pool.
    """
    if not candidates:
        return None
    normed = normalize_objectives(candidates)
    w_p, w_g, w_i = weights
    return min(
        normed,
        key=lambda c: w_p * c.proximity
        + w_g * c.geometric_instability
        + w_i * c.intervention_instability,
    )


def select_pareto_knee(
    candidates: List[CandidateObjectives],
) -> Optional[CandidateObjectives]:
    """Pick the Pareto-optimal point closest to the ideal point.

    After min-max normalising across *all* candidates, extract the Pareto
    front and pick the member with the smallest L2 distance to the origin
    (the ideal point in normalised space).
    """
    if not candidates:
        return None
    front = pareto_front(candidates)
    if not front:
        return None
    # Normalise w.r.t. the *full* pool so all objectives are on [0, 1].
    normed_all = normalize_objectives(candidates)
    # Build a lookup from cf_index -> normalised objectives
    normed_map = {c.cf_index: c for c in normed_all}
    # Among front members, find the one nearest the ideal (0, 0, 0).
    best, best_dist = None, float("inf")
    for c in front:
        nc = normed_map[c.cf_index]
        dist = (nc.proximity ** 2
                + nc.geometric_instability ** 2
                + nc.intervention_instability ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best = c
    return best


def select_pareto_lex(
    candidates: List[CandidateObjectives],
    order: tuple[str, str, str] = (
        "geometric_instability",
        "intervention_instability",
        "proximity",
    ),
) -> Optional[CandidateObjectives]:
    """Lexicographic selection from the Pareto front.

    Default priority: robustness first (geo → int → proximity).
    """
    if not candidates:
        return None
    front = pareto_front(candidates)
    if not front:
        return None
    return min(front, key=lambda c: tuple(getattr(c, k) for k in order))


def select_random(
    candidates: List[CandidateObjectives],
    rng: Optional[random.Random] = None,
) -> Optional[CandidateObjectives]:
    """Uniformly sample one valid candidate (sanity baseline)."""
    if not candidates:
        return None
    r = rng or random.Random()
    return r.choice(candidates)


# ---------------------------------------------------------------------------
# Registry: name -> (function, extra kwargs)
# ---------------------------------------------------------------------------

SELECTOR_REGISTRY: dict[str, dict] = {
    "min_proximity": {"fn": select_min_proximity},
    "min_geo": {"fn": select_min_geo},
    "weighted_sum_equal": {
        "fn": select_weighted_sum,
        "kwargs": {"weights": (1 / 3, 1 / 3, 1 / 3)},
    },
    "weighted_sum_prox_heavy": {
        "fn": select_weighted_sum,
        "kwargs": {"weights": (0.6, 0.2, 0.2)},
    },
    "pareto_knee": {"fn": select_pareto_knee},
    "pareto_lex": {"fn": select_pareto_lex},
    "random": {"fn": select_random},
}


def apply_selector(
    name: str,
    candidates: List[CandidateObjectives],
    **extra_kwargs,
) -> Optional[SelectorResult]:
    """Apply a named selector to a candidate pool.

    Returns ``None`` when the pool is empty or the selector yields nothing.
    """
    entry = SELECTOR_REGISTRY[name]
    fn = entry["fn"]
    kwargs = {**entry.get("kwargs", {}), **extra_kwargs}
    chosen = fn(candidates, **kwargs)
    if chosen is None:
        return None

    # For weighted-sum selectors the *chosen* object holds normalised
    # values.  Look up raw values from the original candidates list.
    raw = next((c for c in candidates if c.cf_index == chosen.cf_index), chosen)

    return SelectorResult(
        selector_name=name,
        query_uuid=raw.query_uuid,
        selected_cf_index=raw.cf_index,
        proximity=raw.proximity,
        geometric_instability=raw.geometric_instability,
        intervention_instability=raw.intervention_instability,
    )


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def records_to_candidates(records: list[dict]) -> dict[tuple[str, float], List[CandidateObjectives]]:
    """Group flat record dicts into per-(query_uuid, sigma) candidate lists."""
    grouped: dict[tuple[str, float], List[CandidateObjectives]] = {}
    for r in records:
        key = (r["query_uuid"], r["sigma"])
        obj = CandidateObjectives(
            query_uuid=r["query_uuid"],
            cf_index=r["cf_index"],
            proximity=r["proximity"],
            geometric_instability=r["geometric_instability"],
            intervention_instability=r["intervention_instability"],
        )
        grouped.setdefault(key, []).append(obj)
    return grouped


def apply_all_selectors(
    records: list[dict],
    selector_names: Optional[List[str]] = None,
    seed: int = 42,
) -> list[dict]:
    """Apply every selector to every (query, sigma) group.

    Returns a flat list of dicts ready for ``pd.DataFrame``.
    """
    names = selector_names or list(SELECTOR_REGISTRY.keys())
    grouped = records_to_candidates(records)
    rng = random.Random(seed)

    results: list[dict] = []
    for (qid, sigma), candidates in grouped.items():
        for name in names:
            extra = {"rng": rng} if name == "random" else {}
            res = apply_selector(name, candidates, **extra)
            if res is None:
                continue
            results.append({
                "selector": res.selector_name,
                "query_uuid": res.query_uuid,
                "sigma": sigma,
                "selected_cf_index": res.selected_cf_index,
                "proximity": res.proximity,
                "geometric_instability": res.geometric_instability,
                "intervention_instability": res.intervention_instability,
            })
    return results
