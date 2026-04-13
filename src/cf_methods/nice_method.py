"""NICE (Nearest Instance Counterfactual Explanations) wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from nice import NICE
from sklearn.preprocessing import LabelEncoder

from .base_cf_method import BaseCounterfactualGenerationMethod
from .registry import register_method

logger = logging.getLogger(__name__)


@dataclass
class NICEResult:
    """Adapter so pool builder can call ``result.to_dataframe()``."""

    cfs: pd.DataFrame | None
    feature_names: list[str]

    def to_dataframe(self) -> pd.DataFrame | None:
        return self.cfs


@register_method(name="nice")
class NICEMethod(BaseCounterfactualGenerationMethod):
    """Wraps the NICE library for counterfactual generation."""

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
        feature_cols = [
            c for c in dataframe.columns if c != target_column
        ]
        self.feature_names = feature_cols
        self._continuous = continuous_features
        self._categorical = [
            c for c in feature_cols if c not in continuous_features
        ]

        # Label-encode categoricals so NICE gets a numeric array.
        self._label_encoders: dict[str, LabelEncoder] = {}
        train_df = dataframe[feature_cols].copy()
        for col in self._categorical:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(
                train_df[col].astype(str)
            )
            self._label_encoders[col] = le

        X_train = train_df.values.astype(np.float64)
        y_train = dataframe[target_column].values.astype(np.float64)

        # Feature index lists (column order kept from feature_cols)
        cat_feat = [
            i for i, c in enumerate(feature_cols)
            if c in self._categorical
        ]
        num_feat = [
            i for i, c in enumerate(feature_cols)
            if c in continuous_features
        ]

        # Build predict_fn that bridges label-encoded arrays
        # to the OHE+scaled tensors our PYTModel expects.
        self._model = model
        self._build_predict_fn(
            model, dataframe, target_column, feature_cols,
        )

        nice_cfg = cfg.get("nice", {})
        self.explainer = NICE(
            predict_fn=self._predict_fn,
            X_train=X_train,
            y_train=y_train,
            cat_feat=cat_feat,
            num_feat=num_feat,
            optimization=nice_cfg.get(
                "optimization", "sparsity",
            ),
            justified_cf=nice_cfg.get("justified_cf", True),
            distance_metric=nice_cfg.get(
                "distance_metric", "HEOM",
            ),
            num_normalization=nice_cfg.get(
                "num_normalization", "minmax",
            ),
        )

    # ----------------------------------------------------------
    def _build_predict_fn(
        self,
        model,
        dataframe: pd.DataFrame,
        target_column: str,
        feature_cols: list[str],
    ) -> None:
        """Build ``predict_fn(X_label_encoded) -> proba``."""
        from src.data.preprocessing.py_dataset import PYTDataset

        tmp_ds = PYTDataset(
            dataframe=dataframe,
            target_column=target_column,
            train=True,
        )
        self._scaler = tmp_ds.scaler
        self._ohe = tmp_ds.encoder

        # Use the exact column orders the scaler/OHE were fitted on
        scaler_cols = list(self._scaler.feature_names_in_)
        ohe_cols = list(self._ohe.feature_names_in_) if self._ohe is not None else []
        categorical = self._categorical

        def predict_fn(X: np.ndarray) -> np.ndarray:
            """Label-encoded array → [p(0), p(1)]."""
            df = pd.DataFrame(X, columns=feature_cols)
            # Inverse label-encoding → original strings
            for col in categorical:
                le = self._label_encoders[col]
                df[col] = le.inverse_transform(
                    df[col].round().astype(int),
                )

            parts = []
            if scaler_cols:
                parts.append(
                    self._scaler.transform(df[scaler_cols])
                )
            if ohe_cols and self._ohe is not None:
                parts.append(
                    self._ohe.transform(df[ohe_cols])
                )
            encoded = np.concatenate(
                parts, axis=1,
            ).astype(np.float32)

            t = torch.tensor(encoded, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                logits = model(t)
            p = logits.squeeze(-1).numpy()
            return np.column_stack([1 - p, p])

        self._predict_fn = predict_fn

    # ----------------------------------------------------------
    def generate(
        self,
        query_instance: pd.DataFrame,
        num_cfs: int,
        **kwargs,
    ) -> NICEResult:
        """Generate counterfactuals for a single query instance.

        NICE is deterministic for a given query, so the single CF
        is replicated ``num_cfs`` times.  The pool builder's
        deduplicator collapses duplicates afterwards.
        """
        feat_cols = self.feature_names
        q = query_instance[feat_cols].copy()
        for col in self._categorical:
            le = self._label_encoders[col]
            q[col] = le.transform(q[col].astype(str))
        x = q.values.astype(np.float64)

        try:
            cf = self.explainer.explain(x)
        except Exception:
            logger.warning(
                "NICE failed to generate a CF", exc_info=True,
            )
            return NICEResult(
                cfs=None, feature_names=self.feature_names,
            )

        if cf is None:
            return NICEResult(
                cfs=None, feature_names=self.feature_names,
            )

        cf = np.atleast_2d(cf)
        cfs = np.repeat(cf, num_cfs, axis=0)

        # Decode back to original feature space
        cfs_df = pd.DataFrame(cfs, columns=self.feature_names)
        for col in self._categorical:
            le = self._label_encoders[col]
            cfs_df[col] = le.inverse_transform(
                cfs_df[col].round().astype(int),
            )

        return NICEResult(
            cfs=cfs_df, feature_names=self.feature_names,
        )
