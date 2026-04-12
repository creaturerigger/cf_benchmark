"""Growing Spheres (GS) counterfactual method wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from growingspheres import counterfactuals as gs_cf
from sklearn.preprocessing import LabelEncoder

from .base_cf_method import BaseCounterfactualGenerationMethod
from .registry import register_method

logger = logging.getLogger(__name__)


@dataclass
class GSResult:
    """Adapter so pool builder can call ``result.to_dataframe()``."""

    cfs: pd.DataFrame | None
    feature_names: list[str]

    def to_dataframe(self) -> pd.DataFrame | None:
        return self.cfs


@register_method(name="gs")
class GSMethod(BaseCounterfactualGenerationMethod):
    """Wraps the Growing Spheres library for counterfactual generation."""

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

        # Label-encode categoricals so GS gets a numeric array.
        self._label_encoders: dict[str, LabelEncoder] = {}
        train_df = dataframe[feature_cols].copy()
        for col in self._categorical:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(
                train_df[col].astype(str),
            )
            self._label_encoders[col] = le

        self._model = model
        self._build_predict_fn(
            model, dataframe, target_column, feature_cols,
        )

        self._gs_cfg = cfg.get("gs", {})

    # ----------------------------------------------------------
    def _build_predict_fn(
        self,
        model,
        dataframe: pd.DataFrame,
        target_column: str,
        feature_cols: list[str],
    ) -> None:
        """Build ``predict_fn(X_label_encoded) -> integer labels``."""
        from src.data.preprocessing.py_dataset import PYTDataset

        tmp_ds = PYTDataset(
            dataframe=dataframe,
            target_column=target_column,
            train=True,
        )
        self._scaler = tmp_ds.scaler
        self._ohe = tmp_ds.encoder

        scaler_cols = list(self._scaler.feature_names_in_)
        ohe_cols = list(self._ohe.feature_names_in_)
        categorical = self._categorical

        def predict_fn(X: np.ndarray) -> np.ndarray:
            """Label-encoded array → integer class labels."""
            X = np.atleast_2d(X)
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
                    self._scaler.transform(df[scaler_cols]),
                )
            if ohe_cols and self._ohe is not None:
                parts.append(
                    self._ohe.transform(df[ohe_cols]),
                )
            encoded = np.concatenate(
                parts, axis=1,
            ).astype(np.float32)

            t = torch.tensor(encoded, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                logits = model(t)
            p = logits.squeeze(-1).numpy()
            preds = (p >= 0.5).astype(int)
            # GS calls int() on single-instance result; NumPy 2.0+
            # rejects int() on 1-element arrays, so return a scalar.
            if preds.shape[0] == 1:
                return int(preds[0])
            return preds

        self._predict_fn = predict_fn

    # ----------------------------------------------------------
    def generate(
        self,
        query_instance: pd.DataFrame,
        num_cfs: int,
        **kwargs,
    ) -> GSResult:
        """Generate counterfactuals for a single query instance.

        GS is stochastic (random sphere sampling), so calling it
        ``num_cfs`` times may yield different CFs.
        """
        feat_cols = self.feature_names
        q = query_instance[feat_cols].copy()
        for col in self._categorical:
            le = self._label_encoders[col]
            q[col] = le.transform(q[col].astype(str))
        x = q.values.astype(np.float64).reshape(1, -1)

        gs_params = self._gs_cfg
        cfs_list: list[np.ndarray] = []

        for _ in range(num_cfs):
            try:
                explanation = gs_cf.CounterfactualExplanation(
                    obs_to_interprete=x,
                    prediction_fn=self._predict_fn,
                    method="GS",
                    target_class=None,
                )
                explanation.fit(
                    n_in_layer=gs_params.get("n_in_layer", 2000),
                    layer_shape=gs_params.get("layer_shape", "ball"),
                    first_radius=gs_params.get("first_radius", 0.1),
                    dicrease_radius=gs_params.get(
                        "dicrease_radius", 10,
                    ),
                    sparse=gs_params.get("sparse", True),
                    verbose=False,
                )
                cfs_list.append(explanation.enemy)
            except Exception:
                logger.warning(
                    "GS failed to generate a CF", exc_info=True,
                )

        if not cfs_list:
            return GSResult(
                cfs=None, feature_names=self.feature_names,
            )

        cfs = np.stack(cfs_list)

        # Decode back to original feature space
        cfs_df = pd.DataFrame(cfs, columns=self.feature_names)
        for col in self._categorical:
            le = self._label_encoders[col]
            cfs_df[col] = le.inverse_transform(
                cfs_df[col].round().astype(int),
            )

        return GSResult(
            cfs=cfs_df, feature_names=self.feature_names,
        )
