import pytest
import torch

from src.robustness.matcher import NearestCFMatcher
from src.robustness.score import CandidateObjectives
from src.utils.constants import (
    GeometricDistanceType,
    InterventionDistanceType,
)
from src.evaluation.experiment import RobustnessExperiment, QueryResult


# ── helpers ──────────────────────────────────────────────────

def _make_query_data(n_pool, n_prime, n_perturb, dim=4):
    """Synthetic query data dict for one query_uuid."""
    torch.manual_seed(0)
    x = torch.rand(1, dim)
    pool = torch.rand(n_pool, dim)
    perturbed_queries = [x + 0.01 * torch.randn(1, dim)
                         for _ in range(n_perturb)]
    perturbed_pools = [torch.rand(n_prime, dim)
                       for _ in range(n_perturb)]
    return {
        "x": x,
        "pool": pool,
        "perturbed_queries": perturbed_queries,
        "perturbed_pools": perturbed_pools,
    }


@pytest.fixture
def experiment():
    return RobustnessExperiment(
        matcher=NearestCFMatcher(
            metric=GeometricDistanceType.L_2,
        ),
        geometric_metric=GeometricDistanceType.L_1,
        intervention_metric=InterventionDistanceType.JACCARD_INDEX,
        encoded_cont_feature_indices=[0, 1, 2, 3],
        encoded_cat_feature_indices=[],
    )


@pytest.fixture
def single_query():
    return _make_query_data(n_pool=10, n_prime=8, n_perturb=3)


# ── TestEvaluateQuery ────────────────────────────────────────

class TestEvaluateQuery:
    def test_returns_query_result(self, experiment, single_query):
        result = experiment.evaluate_query(
            query_uuid="u1", sigma=0.05,
            x=single_query["x"],
            pool=single_query["pool"],
            perturbed_queries=single_query["perturbed_queries"],
            perturbed_pools=single_query["perturbed_pools"],
        )
        assert isinstance(result, QueryResult)

    def test_candidates_count(self, experiment, single_query):
        result = experiment.evaluate_query(
            query_uuid="u1", sigma=0.05, **single_query,
        )
        assert len(result.candidates) == single_query["pool"].shape[0]

    def test_sigma_stored(self, experiment, single_query):
        result = experiment.evaluate_query(
            query_uuid="u1", sigma=0.07, **single_query,
        )
        assert result.sigma == 0.07

    def test_pareto_front_subset_of_candidates(
        self, experiment, single_query
    ):
        result = experiment.evaluate_query(
            query_uuid="u1", sigma=0.05, **single_query,
        )
        cand_indices = {c.cf_index for c in result.candidates}
        for p in result.pareto_front:
            assert p.cf_index in cand_indices

    def test_proximity_non_negative(self, experiment, single_query):
        result = experiment.evaluate_query(
            query_uuid="u1", sigma=0.05, **single_query,
        )
        for c in result.candidates:
            assert c.proximity >= 0.0

    def test_instability_non_negative(self, experiment, single_query):
        result = experiment.evaluate_query(
            query_uuid="u1", sigma=0.05, **single_query,
        )
        for c in result.candidates:
            assert c.geometric_instability >= 0.0
            assert c.intervention_instability >= 0.0

    def test_empty_perturbed_pool_handled(self, experiment):
        data = _make_query_data(n_pool=5, n_prime=0, n_perturb=2)
        # prime pools are empty tensors
        data["perturbed_pools"] = [
            torch.zeros(0, 4) for _ in range(2)
        ]
        result = experiment.evaluate_query(
            query_uuid="u1", sigma=0.05, **data,
        )
        # Should still return candidates with zero instability
        assert len(result.candidates) == 5
        for c in result.candidates:
            assert c.geometric_instability == 0.0
            assert c.intervention_instability == 0.0


# ── TestRun ──────────────────────────────────────────────────

class TestRun:
    def test_returns_list_of_query_results(self, experiment):
        queries = {
            "u1": _make_query_data(5, 4, 2),
            "u2": _make_query_data(8, 6, 2),
        }
        results = experiment.run(queries, sigma=0.05)
        assert len(results) == 2
        assert all(isinstance(r, QueryResult) for r in results)

    def test_each_result_has_correct_uuid(self, experiment):
        queries = {
            "u1": _make_query_data(5, 4, 2),
            "u2": _make_query_data(8, 6, 2),
        }
        results = experiment.run(queries, sigma=0.05)
        uuids = {r.query_uuid for r in results}
        assert uuids == {"u1", "u2"}


# ── TestToRecords ────────────────────────────────────────────

class TestToRecords:
    def test_record_keys(self, experiment, single_query):
        results = [experiment.evaluate_query(
            query_uuid="u1", sigma=0.05, **single_query,
        )]
        records = experiment.to_records(results)
        expected_keys = {
            "query_uuid", "cf_index", "sigma",
            "proximity", "geometric_instability",
            "intervention_instability", "is_pareto_optimal",
        }
        assert set(records[0].keys()) == expected_keys

    def test_record_count(self, experiment, single_query):
        results = [experiment.evaluate_query(
            query_uuid="u1", sigma=0.05, **single_query,
        )]
        records = experiment.to_records(results)
        assert len(records) == single_query["pool"].shape[0]

    def test_pareto_flag_present(self, experiment, single_query):
        results = [experiment.evaluate_query(
            query_uuid="u1", sigma=0.05, **single_query,
        )]
        records = experiment.to_records(results)
        pareto_count = sum(1 for r in records if r["is_pareto_optimal"])
        assert pareto_count >= 1

    def test_sigma_in_records(self, experiment, single_query):
        results = [experiment.evaluate_query(
            query_uuid="u1", sigma=0.1, **single_query,
        )]
        records = experiment.to_records(results)
        assert all(r["sigma"] == 0.1 for r in records)
