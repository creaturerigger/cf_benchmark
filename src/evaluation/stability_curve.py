from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class StabilityCurve:
    """Holds a stability profile: metric values over sigma levels."""
    sigma: List[float]
    geometric_mean: List[float]
    intervention_mean: List[float]

    def to_dict(self) -> dict:
        return {
            "sigma": self.sigma,
            "geom_mean": self.geometric_mean,
            "interv_mean": self.intervention_mean,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict())


class StabilityCurveBuilder:
    """Builds stability profiles from sigma-aggregated results."""

    def build(self, sigma_agg: pd.DataFrame) -> StabilityCurve:
        """Build a combined stability curve.

        Args:
            sigma_agg: DataFrame with columns
                [sigma, proximity, geometric_instability,
                 intervention_instability],
                one row per sigma level (as returned by
                ResultsAggregator.aggregate_by_sigma).

        Returns:
            StabilityCurve with sorted sigma values.
        """
        df = sigma_agg.sort_values("sigma").reset_index(drop=True)
        return StabilityCurve(
            sigma=df["sigma"].tolist(),
            geometric_mean=df["geometric_instability"].tolist(),
            intervention_mean=df["intervention_instability"].tolist(),
        )

    def compute_auc(self, curve: StabilityCurve) -> dict[str, float]:
        """AUC over the perturbation spectrum via the trapezoidal rule.

        Returns:
            Dict with keys "geometric_auc" and "intervention_auc".
        """
        sigma = np.array(curve.sigma)
        geo = np.array(curve.geometric_mean)
        interv = np.array(curve.intervention_mean)

        return {
            "geometric_auc": float(np.trapezoid(geo, sigma)),
            "intervention_auc": float(np.trapezoid(interv, sigma)),
        }

    def build_per_dataset(
        self, dataset_agg: pd.DataFrame,
    ) -> dict[str, StabilityCurve]:
        """Build one curve per dataset.

        Args:
            dataset_agg: DataFrame with columns
                [dataset, sigma, geometric_instability,
                 intervention_instability] (as returned by
                ResultsAggregator.aggregate_by_dataset).

        Returns:
            Mapping of dataset_name -> StabilityCurve.
        """
        curves = {}
        for ds_name, group in dataset_agg.groupby("dataset"):
            df = group.sort_values("sigma").reset_index(drop=True)
            curves[ds_name] = StabilityCurve(
                sigma=df["sigma"].tolist(),
                geometric_mean=df["geometric_instability"].tolist(),
                intervention_mean=df["intervention_instability"].tolist(),
            )
        return curves
