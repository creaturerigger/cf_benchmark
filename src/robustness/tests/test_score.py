import pytest
import torch

from src.robustness.score import (
    CandidateObjectives,
    pareto_front,
    normalize_objectives,
    lexicographic_sort,
)


def _cand(p: float, g: float, i: float, idx: int = 0) -> CandidateObjectives:
    return CandidateObjectives(
        query_uuid="q1", cf_index=idx,
        proximity=p, geometric_instability=g, intervention_instability=i,
    )


# ── CandidateObjectives ─────────────────────────────────────


class TestCandidateObjectives:
    def test_to_tensor(self):
        c = _cand(0.1, 0.2, 0.3)
        t = c.to_tensor()
        assert torch.allclose(t, torch.tensor([0.1, 0.2, 0.3]))

    def test_dominates_true(self):
        a = _cand(0.1, 0.1, 0.1)
        b = _cand(0.2, 0.2, 0.2)
        assert a.dominates(b)

    def test_dominates_false_equal(self):
        a = _cand(0.1, 0.1, 0.1)
        assert not a.dominates(a)

    def test_dominates_false_trade_off(self):
        a = _cand(0.1, 0.3, 0.1)
        b = _cand(0.2, 0.1, 0.2)
        assert not a.dominates(b)
        assert not b.dominates(a)


# ── pareto_front (tensor-based) ──────────────────────────────


class TestParetoFront:
    def test_empty(self):
        assert pareto_front([]) == []

    def test_single_candidate(self):
        c = _cand(0.5, 0.5, 0.5)
        assert pareto_front([c]) == [c]

    def test_one_dominates_another(self):
        a = _cand(0.1, 0.1, 0.1, idx=0)
        b = _cand(0.2, 0.2, 0.2, idx=1)
        front = pareto_front([a, b])
        assert len(front) == 1
        assert front[0].cf_index == 0

    def test_trade_off_both_on_front(self):
        a = _cand(0.1, 0.9, 0.5, idx=0)
        b = _cand(0.9, 0.1, 0.5, idx=1)
        front = pareto_front([a, b])
        indices = {c.cf_index for c in front}
        assert indices == {0, 1}

    def test_three_candidates_mixed(self):
        a = _cand(0.1, 0.1, 0.9, idx=0)
        b = _cand(0.1, 0.9, 0.1, idx=1)
        c = _cand(0.5, 0.5, 0.5, idx=2)  # dominated by neither a nor b
        d = _cand(0.9, 0.9, 0.9, idx=3)  # dominated by a, b, and c
        front = pareto_front([a, b, c, d])
        indices = {c_.cf_index for c_ in front}
        assert 3 not in indices
        assert 0 in indices
        assert 1 in indices


# ── dominates (scalar-based) ────────────────────────────────


# ── normalize_objectives ─────────────────────────────────────


class TestNormalizeObjectives:
    def test_single_returns_unchanged(self):
        c = _cand(0.5, 0.5, 0.5)
        result = normalize_objectives([c])
        assert result[0].proximity == 0.5

    def test_two_candidates_normalized(self):
        a = _cand(1.0, 2.0, 3.0, idx=0)
        b = _cand(3.0, 4.0, 5.0, idx=1)
        normed = normalize_objectives([a, b])
        # min-max: a should be all 0, b should be all 1
        assert normed[0].proximity == pytest.approx(0.0)
        assert normed[0].geometric_instability == pytest.approx(0.0)
        assert normed[1].proximity == pytest.approx(1.0)
        assert normed[1].intervention_instability == pytest.approx(1.0)

    def test_constant_objective_no_division_error(self):
        a = _cand(0.5, 0.5, 0.1, idx=0)
        b = _cand(0.5, 0.5, 0.9, idx=1)
        normed = normalize_objectives([a, b])
        # proximity and geometric are constant → span=0 → set to 1 → (0.5-0.5)/1=0
        assert normed[0].proximity == pytest.approx(0.0)
        assert normed[1].proximity == pytest.approx(0.0)
        # intervention should still be normalized
        assert normed[0].intervention_instability == pytest.approx(0.0)
        assert normed[1].intervention_instability == pytest.approx(1.0)

    def test_preserves_query_uuid_and_index(self):
        a = CandidateObjectives("q42", 7, 1.0, 2.0, 3.0)
        b = CandidateObjectives("q42", 8, 3.0, 4.0, 5.0)
        normed = normalize_objectives([a, b])
        assert normed[0].query_uuid == "q42"
        assert normed[0].cf_index == 7
        assert normed[1].cf_index == 8


# ── lexicographic_sort ───────────────────────────────────────


class TestLexicographicSort:
    def test_sorts_by_proximity_first(self):
        a = _cand(0.3, 0.0, 0.0, idx=0)
        b = _cand(0.1, 0.0, 0.0, idx=1)
        c = _cand(0.2, 0.0, 0.0, idx=2)
        result = lexicographic_sort([a, b, c])
        assert [c.cf_index for c in result] == [1, 2, 0]

    def test_tiebreak_geometric(self):
        a = _cand(0.1, 0.3, 0.0, idx=0)
        b = _cand(0.1, 0.1, 0.0, idx=1)
        result = lexicographic_sort([a, b])
        assert result[0].cf_index == 1

    def test_tiebreak_intervention(self):
        a = _cand(0.1, 0.1, 0.5, idx=0)
        b = _cand(0.1, 0.1, 0.1, idx=1)
        result = lexicographic_sort([a, b])
        assert result[0].cf_index == 1

    def test_empty(self):
        assert lexicographic_sort([]) == []
