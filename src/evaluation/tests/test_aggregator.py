import pytest
import pandas as pd

from src.evaluation.aggregator import ResultsAggregator


def _make_records():
    """Minimal records simulating two queries, two sigma levels."""
    return [
        {"query_uuid": "u1", "cf_index": 0, "sigma": 0.01,
         "proximity": 0.2, "geometric_instability": 0.1,
         "intervention_instability": 0.3, "is_pareto_optimal": True},
        {"query_uuid": "u1", "cf_index": 1, "sigma": 0.01,
         "proximity": 0.5, "geometric_instability": 0.05,
         "intervention_instability": 0.1, "is_pareto_optimal": False},
        {"query_uuid": "u1", "cf_index": 0, "sigma": 0.05,
         "proximity": 0.2, "geometric_instability": 0.3,
         "intervention_instability": 0.5, "is_pareto_optimal": True},
        {"query_uuid": "u1", "cf_index": 1, "sigma": 0.05,
         "proximity": 0.5, "geometric_instability": 0.15,
         "intervention_instability": 0.2, "is_pareto_optimal": True},
        {"query_uuid": "u2", "cf_index": 0, "sigma": 0.01,
         "proximity": 0.3, "geometric_instability": 0.12,
         "intervention_instability": 0.25, "is_pareto_optimal": True},
        {"query_uuid": "u2", "cf_index": 0, "sigma": 0.05,
         "proximity": 0.3, "geometric_instability": 0.35,
         "intervention_instability": 0.45, "is_pareto_optimal": False},
    ]


@pytest.fixture
def agg():
    return ResultsAggregator()


@pytest.fixture
def records():
    return _make_records()


class TestToDataframe:
    def test_returns_dataframe(self, agg, records):
        df = agg.to_dataframe(records)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, agg, records):
        df = agg.to_dataframe(records)
        assert len(df) == len(records)


class TestAggregateQueryLevel:
    def test_groups_by_query_and_sigma(self, agg, records):
        df = agg.aggregate_query_level(records)
        groups = set(zip(df["query_uuid"], df["sigma"]))
        assert ("u1", 0.01) in groups
        assert ("u2", 0.05) in groups

    def test_columns_present(self, agg, records):
        df = agg.aggregate_query_level(records)
        for col in ResultsAggregator.METRIC_COLS:
            assert col in df.columns


class TestAggregateParetoOnly:
    def test_excludes_non_pareto(self, agg, records):
        df = agg.aggregate_pareto_only(records)
        # u1/sigma=0.01 has one pareto (cf0) and one non-pareto (cf1)
        row = df[
            (df["query_uuid"] == "u1") & (df["sigma"] == 0.01)
        ]
        assert len(row) == 1
        # proximity should be 0.2 (only the pareto candidate)
        assert row.iloc[0]["proximity"] == pytest.approx(0.2)

    def test_returns_dataframe(self, agg, records):
        df = agg.aggregate_pareto_only(records)
        assert isinstance(df, pd.DataFrame)


class TestAggregateBySigma:
    def test_one_row_per_sigma(self, agg, records):
        df = agg.aggregate_by_sigma(records)
        assert set(df["sigma"]) == {0.01, 0.05}

    def test_mean_is_correct(self, agg, records):
        df = agg.aggregate_by_sigma(records)
        row_01 = df[df["sigma"] == 0.01].iloc[0]
        # sigma=0.01 records: prox = [0.2, 0.5, 0.3], mean = 1/3
        assert row_01["proximity"] == pytest.approx(1.0 / 3.0, abs=1e-6)


class TestAggregateByDataset:
    def test_includes_dataset_column(self, agg, records):
        all_results = {
            "adult": records[:3],
            "compas": records[3:],
        }
        df = agg.aggregate_by_dataset(all_results)
        assert "dataset" in df.columns
        assert set(df["dataset"]) == {"adult", "compas"}

    def test_returns_dataframe(self, agg, records):
        all_results = {"adult": records}
        df = agg.aggregate_by_dataset(all_results)
        assert isinstance(df, pd.DataFrame)
