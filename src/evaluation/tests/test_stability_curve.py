import pytest
import numpy as np
import pandas as pd

from src.evaluation.stability_curve import (
    StabilityCurve,
    StabilityCurveBuilder,
)


@pytest.fixture
def builder():
    return StabilityCurveBuilder()


@pytest.fixture
def sigma_agg():
    """Simulated output of ResultsAggregator.aggregate_by_sigma."""
    return pd.DataFrame({
        "sigma": [0.05, 0.01, 0.1, 0.03],
        "proximity": [0.21, 0.10, 0.30, 0.15],
        "geometric_instability": [0.18, 0.05, 0.31, 0.09],
        "intervention_instability": [0.25, 0.04, 0.39, 0.11],
    })


@pytest.fixture
def dataset_agg():
    """Simulated output of ResultsAggregator.aggregate_by_dataset."""
    return pd.DataFrame({
        "dataset": ["adult", "adult", "compas", "compas"],
        "sigma": [0.01, 0.05, 0.01, 0.05],
        "proximity": [0.1, 0.2, 0.15, 0.25],
        "geometric_instability": [0.05, 0.18, 0.08, 0.22],
        "intervention_instability": [0.04, 0.25, 0.06, 0.30],
    })


# ── TestStabilityCurve ───────────────────────────────────────

class TestStabilityCurve:
    def test_to_dict_keys(self):
        curve = StabilityCurve(
            sigma=[0.01, 0.05],
            geometric_mean=[0.1, 0.2],
            intervention_mean=[0.15, 0.3],
        )
        d = curve.to_dict()
        assert set(d.keys()) == {"sigma", "geom_mean", "interv_mean"}

    def test_to_dataframe_shape(self):
        curve = StabilityCurve(
            sigma=[0.01, 0.05, 0.1],
            geometric_mean=[0.1, 0.2, 0.3],
            intervention_mean=[0.15, 0.3, 0.45],
        )
        df = curve.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3


# ── TestBuild ────────────────────────────────────────────────

class TestBuild:
    def test_returns_stability_curve(self, builder, sigma_agg):
        curve = builder.build(sigma_agg)
        assert isinstance(curve, StabilityCurve)

    def test_sigma_sorted(self, builder, sigma_agg):
        curve = builder.build(sigma_agg)
        assert curve.sigma == sorted(curve.sigma)

    def test_values_match_sorted_order(self, builder, sigma_agg):
        curve = builder.build(sigma_agg)
        # After sorting: sigma = [0.01, 0.03, 0.05, 0.1]
        assert curve.sigma == [0.01, 0.03, 0.05, 0.1]
        assert curve.geometric_mean == [0.05, 0.09, 0.18, 0.31]
        assert curve.intervention_mean == [0.04, 0.11, 0.25, 0.39]

    def test_length_matches_input(self, builder, sigma_agg):
        curve = builder.build(sigma_agg)
        assert len(curve.sigma) == len(sigma_agg)


# ── TestComputeAuc ───────────────────────────────────────────

class TestComputeAuc:
    def test_returns_dict_with_keys(self, builder):
        curve = StabilityCurve(
            sigma=[0.01, 0.05, 0.1],
            geometric_mean=[0.05, 0.18, 0.31],
            intervention_mean=[0.04, 0.25, 0.39],
        )
        auc = builder.compute_auc(curve)
        assert "geometric_auc" in auc
        assert "intervention_auc" in auc

    def test_auc_non_negative(self, builder):
        curve = StabilityCurve(
            sigma=[0.01, 0.05, 0.1],
            geometric_mean=[0.05, 0.18, 0.31],
            intervention_mean=[0.04, 0.25, 0.39],
        )
        auc = builder.compute_auc(curve)
        assert auc["geometric_auc"] >= 0.0
        assert auc["intervention_auc"] >= 0.0

    def test_auc_matches_manual(self, builder):
        # Simple linear case: f(x) = x on [0, 1]
        # AUC should be 0.5
        curve = StabilityCurve(
            sigma=[0.0, 1.0],
            geometric_mean=[0.0, 1.0],
            intervention_mean=[0.0, 1.0],
        )
        auc = builder.compute_auc(curve)
        assert auc["geometric_auc"] == pytest.approx(0.5)
        assert auc["intervention_auc"] == pytest.approx(0.5)

    def test_auc_constant_curve(self, builder):
        # Constant value 0.5 on [0, 1] => AUC = 0.5
        curve = StabilityCurve(
            sigma=[0.0, 0.5, 1.0],
            geometric_mean=[0.5, 0.5, 0.5],
            intervention_mean=[0.5, 0.5, 0.5],
        )
        auc = builder.compute_auc(curve)
        assert auc["geometric_auc"] == pytest.approx(0.5)
        assert auc["intervention_auc"] == pytest.approx(0.5)


# ── TestBuildPerDataset ──────────────────────────────────────

class TestBuildPerDataset:
    def test_returns_dict(self, builder, dataset_agg):
        curves = builder.build_per_dataset(dataset_agg)
        assert isinstance(curves, dict)

    def test_one_curve_per_dataset(self, builder, dataset_agg):
        curves = builder.build_per_dataset(dataset_agg)
        assert set(curves.keys()) == {"adult", "compas"}

    def test_each_curve_sorted(self, builder, dataset_agg):
        curves = builder.build_per_dataset(dataset_agg)
        for curve in curves.values():
            assert curve.sigma == sorted(curve.sigma)

    def test_curve_length(self, builder, dataset_agg):
        curves = builder.build_per_dataset(dataset_agg)
        assert len(curves["adult"].sigma) == 2
        assert len(curves["compas"].sigma) == 2
