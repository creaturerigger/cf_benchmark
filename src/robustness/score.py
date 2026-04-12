from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class CandidateObjectives:
    """r(c) = (p(c), s_E(c), s_I(c)) for one candidate CF."""
    query_uuid: str
    cf_index: int
    proximity: float
    geometric_instability: float
    intervention_instability: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.proximity,
            self.geometric_instability,
            self.intervention_instability,
        ])

    def dominates(self, other: CandidateObjectives) -> bool:
        """True if self Pareto-dominates other (all minimised)."""
        s = self.to_tensor()
        o = other.to_tensor()
        return bool((s <= o).all() and (s < o).any())


def pareto_front(
    candidates: List[CandidateObjectives],
) -> List[CandidateObjectives]:
    """Extract the Pareto-optimal set (all objectives minimised).

    Returns the subset of candidates that are not dominated by any other.
    """
    if not candidates:
        return []

    objectives = torch.stack([c.to_tensor() for c in candidates])
    n = objectives.shape[0]
    is_dominated = torch.zeros(n, dtype=torch.bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # j dominates i?
            if bool(
                (objectives[j] <= objectives[i]).all()
                and (objectives[j] < objectives[i]).any()
            ):
                is_dominated[i] = True
                break

    return [c for c, dom in zip(candidates, is_dominated) if not dom]


def normalize_objectives(
    candidates: List[CandidateObjectives],
) -> List[CandidateObjectives]:
    """Min-max normalise each objective to [0, 1] across candidates."""
    if len(candidates) <= 1:
        return candidates

    objectives = torch.stack([c.to_tensor() for c in candidates])
    mins = objectives.min(dim=0).values
    maxs = objectives.max(dim=0).values
    span = maxs - mins
    span[span == 0] = 1.0
    normed = (objectives - mins) / span

    return [
        CandidateObjectives(
            query_uuid=c.query_uuid,
            cf_index=c.cf_index,
            proximity=float(normed[i, 0]),
            geometric_instability=float(normed[i, 1]),
            intervention_instability=float(normed[i, 2]),
        )
        for i, c in enumerate(candidates)
    ]


def lexicographic_sort(
    candidates: list[CandidateObjectives],
) -> list[CandidateObjectives]:
    return sorted(
        candidates,
        key=lambda c: (
            c.proximity,
            c.geometric_instability,
            c.intervention_instability,
        ),
    )
