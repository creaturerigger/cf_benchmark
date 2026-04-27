"""LORE (LOcal Rule-based Explanations) counterfactual method wrapper.

Uses the ``lore_sa`` library (kdd-lab/LORE_sa) which implements:
    Guidotti, R., Monreale, A., Ruggieri, S., Pedreschi, D., Turini, F.,
    & Giannotti, F. (2018). Local rule-based explanations of black box
    decision systems.  arXiv:1805.10820.

LORE generates a synthetic neighbourhood via a genetic algorithm, trains
a local decision-tree surrogate, and extracts counterfactual rules.  The
actual neighbourhood samples with a flipped prediction serve as the CFs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from lore_sa.bbox import AbstractBBox
from lore_sa.dataset import TabularDataset
from lore_sa.lore import TabularGeneticGeneratorLore, TabularRandomGeneratorLore

from .base_cf_method import BaseCounterfactualGenerationMethod
from .registry import register_method

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Result adapter
# ────────────────────────────────────────────────────────────
@dataclass
class LOREResult:
    """Adapter so pool builder can call ``result.to_dataframe()``."""

    cfs: pd.DataFrame | None
    feature_names: list[str]

    def to_dataframe(self) -> pd.DataFrame | None:
        return self.cfs


# ────────────────────────────────────────────────────────────
# BBox wrapper for PyTorch models
# ────────────────────────────────────────────────────────────
class _PytorchBBox(AbstractBBox):
    """Wraps a PyTorch binary classifier so LORE can call
    ``predict`` / ``predict_proba`` on *decoded* (original-space) data."""

    def __init__(self, model, scaler, ohe, feature_cols,
                 continuous_features, categorical_features):
        super().__init__(model)
        self._model = model
        self._scaler = scaler
        self._ohe = ohe
        self._feature_cols = feature_cols
        self._continuous = continuous_features
        self._categorical = categorical_features
        self._scaler_cols = list(scaler.feature_names_in_)
        self._ohe_cols = list(ohe.feature_names_in_) if ohe is not None else []

    def _to_tensor(self, X) -> torch.Tensor:
        """Decoded (original-space) array → model-ready float tensor."""
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self._feature_cols)
        elif isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(np.atleast_2d(X), columns=self._feature_cols)

        # Ensure numeric columns are float
        for col in self._continuous:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        parts = []
        if self._scaler_cols:
            parts.append(self._scaler.transform(df[self._scaler_cols]))
        if self._ohe_cols and self._ohe is not None:
            parts.append(self._ohe.transform(df[self._ohe_cols]))
        encoded = np.concatenate(parts, axis=1).astype(np.float32)
        return torch.tensor(encoded, dtype=torch.float32)

    def predict(self, X):
        t = self._to_tensor(X)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(t)
        probs = logits.squeeze(-1).numpy()
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        t = self._to_tensor(X)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(t)
        p1 = logits.squeeze(-1).numpy()
        return np.column_stack([1 - p1, p1])


# ────────────────────────────────────────────────────────────
# Public wrapper
# ────────────────────────────────────────────────────────────
@register_method(name="lore")
class LOREMethod(BaseCounterfactualGenerationMethod):
    """LORE wrapper using ``lore_sa.TabularGeneticGeneratorLore``."""

    def __init__(
        self,
        cfg: dict,
        model,
        dataframe: pd.DataFrame,
        target_column: str,
        continuous_features: list[str],
    ):
        super().__init__(cfg)

        self.target_column = target_column
        feature_cols = [c for c in dataframe.columns if c != target_column]
        self.feature_names = feature_cols
        self._continuous = continuous_features
        self._categorical = [
            c for c in feature_cols if c not in continuous_features
        ]

        # Build scaler / OHE from PYTDataset (same as other wrappers)
        from src.data.preprocessing.py_dataset import PYTDataset

        tmp_ds = PYTDataset(
            dataframe=dataframe,
            target_column=target_column,
            train=True,
        )
        self._scaler = tmp_ds.scaler
        self._ohe = tmp_ds.encoder

        # BBox adapter
        bbox = _PytorchBBox(
            model=model,
            scaler=self._scaler,
            ohe=self._ohe,
            feature_cols=feature_cols,
            continuous_features=self._continuous,
            categorical_features=self._categorical,
        )

        # TabularDataset — LORE needs the training DataFrame *with* target.
        # The target must be categorical (non-numeric) so TabularDataset
        # doesn't raise "target column cannot be continuous".
        # We keep the target as integers but pass it in categorial_columns so
        # the descriptor classifies it as categorical — avoiding the sklearn
        # OrdinalEncoder isnan bug that occurs when the target is cast to str.
        lore_df = dataframe.copy()
        lore_df[target_column] = lore_df[target_column].astype(int)
        lore_dataset = TabularDataset(
            lore_df,
            class_name=target_column,
            categorial_columns=self._categorical + [target_column],
        )

        # Explainer
        lore_cfg = cfg.get("lore", {})
        generator_type = lore_cfg.get("generator", "genetic")
        if generator_type == "random":
            self._explainer = TabularRandomGeneratorLore(bbox, lore_dataset)
        else:
            self._explainer = TabularGeneticGeneratorLore(bbox, lore_dataset)

        self._lore_cfg = lore_cfg
        self._model = model

    # ── generate ─────────────────────────────────────────────
    def generate(
        self,
        query_instance: pd.DataFrame,
        num_cfs: int,
        **kwargs,
    ) -> LOREResult:
        feat_cols = self.feature_names
        q = query_instance[feat_cols].iloc[0]
        x_arr = q.values

        num_instances = self._lore_cfg.get("num_instances", 1000)

        try:
            explanation = self._explainer.explain(x_arr, num_instances=num_instances)
        except Exception:
            logger.warning("LORE explain failed", exc_info=True)
            return LOREResult(cfs=None, feature_names=self.feature_names)

        cf_samples = explanation.get("counterfactual_samples", [])
        if not cf_samples:
            return LOREResult(cfs=None, feature_names=self.feature_names)

        cfs_df = pd.DataFrame(cf_samples, columns=self.feature_names)
        # Ensure proper dtypes
        for col in self._continuous:
            cfs_df[col] = pd.to_numeric(cfs_df[col], errors="coerce")

        # Rank by Euclidean distance (on numerics) to the query and pick top num_cfs
        q_num = q[self._continuous].values.astype(float)
        dists = np.sqrt(
            ((cfs_df[self._continuous].values.astype(float) - q_num) ** 2).sum(
                axis=1,
            ),
        )
        order = np.argsort(dists)
        cfs_df = cfs_df.iloc[order[:num_cfs]].reset_index(drop=True)

        return LOREResult(cfs=cfs_df, feature_names=self.feature_names)
