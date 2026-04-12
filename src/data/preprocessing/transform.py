from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


@dataclass
class CFMethodSpec:
    """Method-agnostic descriptor of what a CF generator needs."""
    name: str
    continuous_features: list[str]
    categorical_features: list[str]
    target_column: str
    requires_one_hot: bool = True
    requires_scaling: bool = True
    scale_range: tuple[float, float] = (0.0, 1.0)
    backend: str = "pytorch"


class Transformer:
    def __init__(self, spec: CFMethodSpec) -> None:
        self.spec = spec
        self.scaler_: Optional[MinMaxScaler] = None
        self.encoder_: Optional[OneHotEncoder] = None
        self._cont_indices: list[int] = []
        self._cat_indices: list[list[int]] = []

    def fit(self, df: pd.DataFrame) -> Transformer:
        if self.spec.requires_scaling and self.spec.continuous_features:
            self.scaler_ = MinMaxScaler(feature_range=self.spec.scale_range) # type: ignore
            self.scaler_.fit(df[self.spec.continuous_features])

        if self.spec.requires_one_hot and self.spec.categorical_features:
            self.encoder_ = OneHotEncoder(sparse_output=False,
                                          handle_unknown="ignore")
            self.encoder_.fit(df[self.spec.categorical_features])

        self._cont_indices = list(range(len(self.spec.continuous_features)))
        self._cat_indices = self._compute_cat_indices()
        return self

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        parts = []

        if self.spec.continuous_features:
            if self.scaler_ is not None:
                parts.append(self.scaler_.transform(df[self.spec.continuous_features]))
            else:
                parts.append(df[self.spec.continuous_features].to_numpy(dtype=np.float64))

        if self.spec.categorical_features:
            if self.encoder_ is not None:
                parts.append(self.encoder_.transform(df[self.spec.categorical_features]))
            else:
                parts.append(df[self.spec.categorical_features].to_numpy(dtype=np.float64))

        transformed = np.concatenate(parts, axis=1)
        return torch.tensor(transformed, dtype=torch.float32)

    def fit_transform(self, df: pd.DataFrame) -> torch.Tensor:
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, tensor: torch.Tensor) -> pd.DataFrame:
        arr = tensor.numpy()
        n_cont = len(self.spec.continuous_features)
        result_parts = {}

        # Continuous columns
        if self.spec.continuous_features:
            cont_arr = arr[:, :n_cont]
            if self.scaler_ is not None:
                cont_arr = self.scaler_.inverse_transform(cont_arr)
            for i, col in enumerate(self.spec.continuous_features):
                result_parts[col] = cont_arr[:, i]

        # Categorical columns
        if self.spec.categorical_features and self.encoder_ is not None:
            cat_arr = arr[:, n_cont:]
            decoded = self.encoder_.inverse_transform(cat_arr)
            for i, col in enumerate(self.spec.categorical_features):
                result_parts[col] = decoded[:, i]

        return pd.DataFrame(result_parts)

    def _compute_cat_indices(self) -> list[list[int]]:
        if self.encoder_ is None:
            return []
        offset = len(self.spec.continuous_features)
        cat_indices = []
        for cats_per_feature in self.encoder_.categories_:
            group = list(range(offset, offset + len(cats_per_feature)))
            cat_indices.append(group)
            offset += len(cats_per_feature)
        return cat_indices

    @property
    def encoded_continuous_feature_indices(self) -> list[int]:
        return self._cont_indices

    @property
    def encoded_categorical_feature_indices(self) -> list[list[int]]:
        return self._cat_indices
