"""Unit tests for selection strategies."""

from __future__ import annotations

import random

import pytest

from src.robustness.score import CandidateObjectives, pareto_front
from src.evaluation.selectors import (
    select_min_proximity,
    select_min_geo,
    select_weighted_sum,
    select_pareto_knee,
    select_pareto_lex,
    select_random,
    apply_selector,
    apply_all_selectors,
    records_to_candidates,
    SELECTOR_REGISTRY,
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_candidates() -> list[CandidateObjectives]:
    """Five candidates with varied trade-offs."""
    return [
        CandidateObjectives("q1", 0, proximity=0.1, geometric_instability=0.9, intervention_instability=0.8),
        CandidateObjectives("q1", 1, proximity=0.5, geometric_instability=0.2, intervention_instability=0.3),
        CandidateObjectives("q1", 2, proximity=0.3, geometric_instability=0.4, intervention_instability=0.5),
        CandidateObjectives("q1", 3, proximity=0.9, geometric_instability=0.1, intervention_instability=0.1),
        CandidateObjectives("q1", 4, proximity=0.2, geometric_instability=0.6, intervention_instability=0.7),
    ]


@pytest.fixture
def sample_records() -> list[dict]:
    """Flat records as produced by the pipeline."""
    return [
        {"query_uuid": "q1", "cf_index": 0, "sigma": 0.05, "proximity": 0.1,
         "geometric_instability": 0.9, "intervention_instability": 0.8, "is_pareto_optimal": False},
        {"query_uuid": "q1", "cf_index": 1, "sigma": 0.05, "proximity": 0.5,
         "geometric_instability": 0.2, "intervention_instability": 0.3, "is_pareto_optimal": True},
        {"query_uuid": "q1", "cf_index": 2, "sigma": 0.05, "proximity": 0.3,
         "geometric_instability": 0.4, "intervention_instability": 0.5, "is_pareto_optimal": True},
        {"query_uuid": "q1", "cf_index": 3, "sigma": 0.05, "proximity": 0.9,
         "geometric_instability": 0.1, "intervention_instability": 0.1, "is_pareto_optimal": True},
        {"query_uuid": "q1", "cf_index": 4, "sigma": 0.05, "proximity": 0.2,
         "geometric_instability": 0.6, "intervention_instability": 0.7, "is_pareto_optimal": False},
    ]


# ── Min-Proximity ────────────────────────────────────────────

class TestMinProximity:
    def test_selects_closest(self, sample_candidates):
        chosen = select_min_proximity(sample_candidates)
        assert chosen.cf_index == 0
        assert chosen.proximity == 0.1

    def test_empty(self):
        assert select_min_proximity([]) is None


# ── Min-Geo ──────────────────────────────────────────────────

class TestMinGeo:
    def test_selects_most_stable(self, sample_candidates):
        chosen = select_min_geo(sample_candidates)
        assert chosen.cf_index == 3
        assert chosen.geometric_instability == 0.1

    def test_empty(self):
        assert select_min_geo([]) is None


# ── Weighted Sum ─────────────────────────────────────────────

class TestWeightedSum:
    def test_equal_weights(self, sample_candidates):
        chosen = select_weighted_sum(sample_candidates, weights=(1/3, 1/3, 1/3))
        assert chosen is not None

    def test_prox_heavy(self, sample_candidates):
        chosen = select_weighted_sum(sample_candidates, weights=(0.9, 0.05, 0.05))
        # With proximity heavily weighted, should pick the closest
        assert chosen is not None
        assert chosen.cf_index == 0

    def test_empty(self):
        assert select_weighted_sum([]) is None


# ── Pareto Knee ──────────────────────────────────────────────

class TestParetoKnee:
    def test_selects_from_front(self, sample_candidates):
        chosen = select_pareto_knee(sample_candidates)
        assert chosen is not None
        front = pareto_front(sample_candidates)
        front_indices = {c.cf_index for c in front}
        assert chosen.cf_index in front_indices

    def test_empty(self):
        assert select_pareto_knee([]) is None

    def test_single_candidate(self):
        c = CandidateObjectives("q1", 0, 0.5, 0.5, 0.5)
        chosen = select_pareto_knee([c])
        assert chosen.cf_index == 0


# ── Pareto Lex ───────────────────────────────────────────────

class TestParetoLex:
    def test_selects_from_front(self, sample_candidates):
        chosen = select_pareto_lex(sample_candidates)
        assert chosen is not None
        front = pareto_front(sample_candidates)
        front_indices = {c.cf_index for c in front}
        assert chosen.cf_index in front_indices

    def test_default_order_prioritises_geo(self, sample_candidates):
        # Default: geo first, so among Pareto members pick lowest geo
        chosen = select_pareto_lex(sample_candidates)
        front = pareto_front(sample_candidates)
        min_geo = min(c.geometric_instability for c in front)
        assert chosen.geometric_instability == min_geo

    def test_empty(self):
        assert select_pareto_lex([]) is None


# ── Random ───────────────────────────────────────────────────

class TestRandom:
    def test_returns_candidate(self, sample_candidates):
        chosen = select_random(sample_candidates, rng=random.Random(42))
        assert chosen is not None
        assert chosen.cf_index in range(5)

    def test_empty(self):
        assert select_random([]) is None


# ── Registry / apply ─────────────────────────────────────────

class TestApplySelector:
    def test_all_registered(self):
        expected = {"min_proximity", "min_geo", "weighted_sum_equal",
                    "weighted_sum_prox_heavy", "pareto_knee", "pareto_lex", "random"}
        assert expected == set(SELECTOR_REGISTRY.keys())

    def test_apply_returns_raw_values(self, sample_candidates):
        res = apply_selector("weighted_sum_equal", sample_candidates)
        # The result should have raw (un-normalised) metric values
        assert res is not None
        raw = next(c for c in sample_candidates if c.cf_index == res.selected_cf_index)
        assert res.proximity == raw.proximity


# ── Batch helpers ────────────────────────────────────────────

class TestBatchHelpers:
    def test_records_to_candidates(self, sample_records):
        grouped = records_to_candidates(sample_records)
        assert ("q1", 0.05) in grouped
        assert len(grouped[("q1", 0.05)]) == 5

    def test_apply_all_selectors(self, sample_records):
        results = apply_all_selectors(sample_records, seed=42)
        assert len(results) > 0
        selectors_seen = {r["selector"] for r in results}
        assert "pareto_knee" in selectors_seen
        assert "min_proximity" in selectors_seen

    def test_apply_all_selectors_empty(self):
        results = apply_all_selectors([], seed=42)
        assert results == []
