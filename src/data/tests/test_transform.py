import pytest
import pandas as pd
import numpy as np
import torch

from src.data.preprocessing.transform import CFMethodSpec, Transformer


@pytest.fixture
def mixed_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.uniform(18, 65, size=50),
        "income": np.random.uniform(20000, 100000, size=50),
        "color": np.random.choice(["red", "blue", "green"], size=50),
        "size": np.random.choice(["S", "M", "L"], size=50),
        "target": np.random.choice([0, 1], size=50),
    })


@pytest.fixture
def spec():
    return CFMethodSpec(
        name="test",
        continuous_features=["age", "income"],
        categorical_features=["color", "size"],
        target_column="target",
    )


@pytest.fixture
def fitted_transformer(spec, mixed_df):
    t = Transformer(spec)
    t.fit(mixed_df)
    return t


class TestFit:
    def test_fit_returns_self(self, spec, mixed_df):
        t = Transformer(spec)
        assert t.fit(mixed_df) is t

    def test_fit_creates_scaler(self, fitted_transformer):
        assert fitted_transformer.scaler_ is not None

    def test_fit_creates_encoder(self, fitted_transformer):
        assert fitted_transformer.encoder_ is not None

    def test_fit_no_scaler_when_disabled(self, mixed_df):
        spec = CFMethodSpec(
            name="test",
            continuous_features=["age", "income"],
            categorical_features=["color", "size"],
            target_column="target",
            requires_scaling=False,
        )
        t = Transformer(spec)
        t.fit(mixed_df)
        assert t.scaler_ is None

    def test_fit_no_encoder_when_disabled(self, mixed_df):
        spec = CFMethodSpec(
            name="test",
            continuous_features=["age", "income"],
            categorical_features=["color", "size"],
            target_column="target",
            requires_one_hot=False,
        )
        t = Transformer(spec)
        t.fit(mixed_df)
        assert t.encoder_ is None


class TestTransform:
    def test_output_is_float32_tensor(self, fitted_transformer, mixed_df):
        out = fitted_transformer.transform(mixed_df)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32

    def test_output_rows_match_input(self, fitted_transformer, mixed_df):
        out = fitted_transformer.transform(mixed_df)
        assert out.shape[0] == len(mixed_df)

    def test_continuous_scaled_to_range(self, fitted_transformer, mixed_df):
        out = fitted_transformer.transform(mixed_df)
        n_cont = len(fitted_transformer.spec.continuous_features)
        cont = out[:, :n_cont]
        assert cont.min() >= -1e-6
        assert cont.max() <= 1.0 + 1e-6

    def test_categorical_one_hot_rows_sum_to_one(self, fitted_transformer, mixed_df):
        out = fitted_transformer.transform(mixed_df)
        for group in fitted_transformer.encoded_categorical_feature_indices:
            row_sums = out[:, group].sum(dim=1)
            assert torch.allclose(row_sums, torch.ones(len(mixed_df)))

    def test_output_column_count(self, fitted_transformer, mixed_df):
        out = fitted_transformer.transform(mixed_df)
        n_cont = len(fitted_transformer.spec.continuous_features)
        n_cat_cols = sum(
            len(g) for g in fitted_transformer.encoded_categorical_feature_indices
        )
        assert out.shape[1] == n_cont + n_cat_cols


class TestFitTransform:
    def test_equals_fit_then_transform(self, spec, mixed_df):
        t1 = Transformer(spec)
        t1.fit(mixed_df)
        out1 = t1.transform(mixed_df)

        t2 = Transformer(spec)
        out2 = t2.fit_transform(mixed_df)
        assert torch.allclose(out1, out2)


class TestInverseTransform:
    def test_roundtrip_continuous(self, fitted_transformer, mixed_df):
        tensor = fitted_transformer.transform(mixed_df)
        recovered = fitted_transformer.inverse_transform(tensor)
        for col in fitted_transformer.spec.continuous_features:
            np.testing.assert_allclose(
                recovered[col].to_numpy(), mixed_df[col].to_numpy(), atol=1e-4
            )

    def test_roundtrip_categorical(self, fitted_transformer, mixed_df):
        tensor = fitted_transformer.transform(mixed_df)
        recovered = fitted_transformer.inverse_transform(tensor)
        for col in fitted_transformer.spec.categorical_features:
            assert list(recovered[col]) == list(mixed_df[col])


class TestIndices:
    def test_continuous_indices(self, fitted_transformer):
        assert fitted_transformer.encoded_continuous_feature_indices == [0, 1]

    def test_categorical_indices_count(self, fitted_transformer):
        assert len(fitted_transformer.encoded_categorical_feature_indices) == 2

    def test_categorical_indices_start_after_continuous(self, fitted_transformer):
        n_cont = len(fitted_transformer.spec.continuous_features)
        for group in fitted_transformer.encoded_categorical_feature_indices:
            assert all(i >= n_cont for i in group)

    def test_categorical_indices_no_overlap(self, fitted_transformer):
        all_idx = [
            i for g in fitted_transformer.encoded_categorical_feature_indices for i in g
        ]
        assert len(all_idx) == len(set(all_idx))

    def test_indices_cover_all_columns(self, fitted_transformer, mixed_df):
        out = fitted_transformer.transform(mixed_df)
        cont = fitted_transformer.encoded_continuous_feature_indices
        cat = [i for g in fitted_transformer.encoded_categorical_feature_indices for i in g]
        assert len(cont) + len(cat) == out.shape[1]


class TestContinuousOnly:
    def test_no_categorical(self, mixed_df):
        spec = CFMethodSpec(
            name="test", continuous_features=["age", "income"],
            categorical_features=[], target_column="target",
        )
        t = Transformer(spec)
        out = t.fit_transform(mixed_df)
        assert out.shape[1] == 2
        assert t.encoded_categorical_feature_indices == []


class TestCategoricalOnly:
    def test_no_continuous(self, mixed_df):
        spec = CFMethodSpec(
            name="test", continuous_features=[],
            categorical_features=["color", "size"], target_column="target",
        )
        t = Transformer(spec)
        out = t.fit_transform(mixed_df)
        assert t.encoded_continuous_feature_indices == []
        assert out.shape[0] == len(mixed_df)