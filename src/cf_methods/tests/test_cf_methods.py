import pytest
import numpy as np
import pandas as pd

from src.cf_methods.registry import METHOD_REGISTRY, register_method, get_method_class, create_method
from src.cf_methods.base_cf_method import BaseCounterfactualGenerationMethod
from src.cf_methods.dice_method import DiCEMethod  # triggers registration
from src.cf_methods.nice_method import NICEMethod
from src.cf_methods.gs_method import GSMethod
from src.cf_methods.moc_method import MOCMethod
from src.cf_methods.lore_method import LOREMethod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adult_like_df():
    """Small synthetic dataframe that mimics Adult dataset structure (numeric-only)."""
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        'age':            np.random.randint(18, 70, n).astype(float),
        'hours_per_week': np.random.randint(1, 80, n).astype(float),
        'education_num':  np.random.randint(1, 16, n).astype(float),
        'income':         np.random.randint(0, 2, n),
    })


@pytest.fixture
def adult_mixed_df():
    """Synthetic dataframe with both numeric and categorical columns."""
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        'age':            np.random.randint(18, 70, n).astype(float),
        'hours_per_week': np.random.randint(1, 80, n).astype(float),
        'workclass':      np.random.choice(['Private', 'Self-emp', 'Gov'], n),
        'education':      np.random.choice(['HS-grad', 'Bachelors', 'Masters'], n),
        'income':         pd.Categorical(np.random.randint(0, 2, n)),
    })


@pytest.fixture
def continuous_features():
    return ['age', 'hours_per_week', 'education_num']


@pytest.fixture
def mixed_continuous():
    return ['age', 'hours_per_week']


@pytest.fixture
def dice_cfg():
    return {
        "method": {"name": "dice"},
        "generation": {
            "total_cfs": 3,
            "desired_class": "opposite",
            "proximity_weight": 0.5,
            "diversity_weight": 1.0,
        },
        "search": {"algorithm": "random"},
    }


@pytest.fixture
def nice_cfg():
    return {
        "method": {"name": "nice"},
        "generation": {"total_cfs": 3, "desired_class": "opposite"},
        "nice": {
            "optimization": "sparsity",
            "justified_cf": True,
            "distance_metric": "HEOM",
            "num_normalization": "minmax",
        },
    }


@pytest.fixture
def gs_cfg():
    return {
        "method": {"name": "gs"},
        "generation": {"total_cfs": 3, "desired_class": "opposite"},
        "gs": {
            "n_in_layer": 500,
            "layer_shape": "ball",
            "first_radius": 0.1,
            "dicrease_radius": 10,
            "sparse": True,
        },
    }


@pytest.fixture
def moc_cfg():
    return {
        "method": {"name": "moc"},
        "generation": {"total_cfs": 3, "desired_class": "opposite"},
        "moc": {
            "pop_size": 10,
            "n_gen": 10,
            "p_cross": 0.57,
            "p_mut": 0.79,
            "eta_cross": 15,
            "eta_mut": 20,
            "k": 1,
        },
    }


@pytest.fixture
def lore_cfg():
    return {
        "method": {"name": "lore"},
        "generation": {"total_cfs": 3, "desired_class": "opposite"},
        "lore": {
            "generator": "random",
            "num_instances": 200,
        },
    }


@pytest.fixture
def tiny_model(adult_like_df, continuous_features):
    """A trivial nn.Module that returns a probability for each row (numeric-only)."""
    import torch.nn as nn

    in_features = len(continuous_features)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(in_features, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.fc(x))

    return TinyModel()


@pytest.fixture
def mixed_model(adult_mixed_df, mixed_continuous):
    """A nn.Module sized for the OHE+scaled mixed dataframe.

    2 numeric + OHE(workclass=3, education=3) → 2 + 3 + 3 = 8 features.
    """
    import torch.nn as nn

    in_features = 8  # 2 scaled numerics + 3 + 3 OHE categoricals

    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(in_features, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.fc(x))

    return MixedModel()


@pytest.fixture
def dice_method(dice_cfg, tiny_model, adult_like_df, continuous_features):
    return DiCEMethod(
        cfg=dice_cfg,
        model=tiny_model,
        dataframe=adult_like_df,
        target_column='income',
        continuous_features=continuous_features,
    )


@pytest.fixture
def nice_method(nice_cfg, mixed_model, adult_mixed_df, mixed_continuous):
    return NICEMethod(
        cfg=nice_cfg,
        model=mixed_model,
        dataframe=adult_mixed_df,
        target_column='income',
        continuous_features=mixed_continuous,
    )


@pytest.fixture
def gs_method(gs_cfg, mixed_model, adult_mixed_df, mixed_continuous):
    return GSMethod(
        cfg=gs_cfg,
        model=mixed_model,
        dataframe=adult_mixed_df,
        target_column='income',
        continuous_features=mixed_continuous,
    )


@pytest.fixture
def moc_method(moc_cfg, mixed_model, adult_mixed_df, mixed_continuous):
    return MOCMethod(
        cfg=moc_cfg,
        model=mixed_model,
        dataframe=adult_mixed_df,
        target_column='income',
        continuous_features=mixed_continuous,
    )


@pytest.fixture
def lore_method(lore_cfg, mixed_model, adult_mixed_df, mixed_continuous):
    return LOREMethod(
        cfg=lore_cfg,
        model=mixed_model,
        dataframe=adult_mixed_df,
        target_column='income',
        continuous_features=mixed_continuous,
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_dice_is_registered(self):
        assert 'dice' in METHOD_REGISTRY

    def test_nice_is_registered(self):
        assert 'nice' in METHOD_REGISTRY

    def test_gs_is_registered(self):
        assert 'gs' in METHOD_REGISTRY

    def test_moc_is_registered(self):
        assert 'moc' in METHOD_REGISTRY

    def test_lore_is_registered(self):
        assert 'lore' in METHOD_REGISTRY

    def test_get_method_class_returns_dice(self):
        cls = get_method_class('dice')
        assert cls is DiCEMethod

    def test_get_method_class_returns_nice(self):
        assert get_method_class('nice') is NICEMethod

    def test_get_method_class_returns_gs(self):
        assert get_method_class('gs') is GSMethod

    def test_get_method_class_returns_moc(self):
        assert get_method_class('moc') is MOCMethod

    def test_get_method_class_returns_lore(self):
        assert get_method_class('lore') is LOREMethod

    def test_get_method_class_unknown_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_method_class('nonexistent_method')

    def test_register_method_decorator(self):
        @register_method(name='_test_dummy')
        class DummyMethod(BaseCounterfactualGenerationMethod):
            def generate(self, query_instance, num_cfs, *args, **kwargs):
                pass

        assert '_test_dummy' in METHOD_REGISTRY
        assert METHOD_REGISTRY['_test_dummy'] is DummyMethod
        del METHOD_REGISTRY['_test_dummy']

    def test_create_method_instantiates_correctly(self, dice_cfg, tiny_model,
                                                   adult_like_df, continuous_features):
        method = create_method(
            dice_cfg,
            model=tiny_model,
            dataframe=adult_like_df,
            target_column='income',
            continuous_features=continuous_features,
        )
        assert isinstance(method, DiCEMethod)

    def test_create_method_unknown_raises(self, tiny_model, adult_like_df, continuous_features):
        bad_cfg = {"method": {"name": "unknown_method"}}
        with pytest.raises(ValueError):
            create_method(bad_cfg, model=tiny_model, dataframe=adult_like_df,
                          target_column='income', continuous_features=continuous_features)


# ---------------------------------------------------------------------------
# DiCEMethod construction tests
# ---------------------------------------------------------------------------

class TestDiCEMethodConstruction:
    def test_instantiation(self, dice_method):
        assert isinstance(dice_method, DiCEMethod)

    def test_data_interface_is_set(self, dice_method):
        assert dice_method.data_interface is not None

    def test_model_interface_is_set(self, dice_method):
        assert dice_method.model_interface is not None

    def test_explainer_is_set(self, dice_method):
        assert dice_method.explainer is not None

    def test_cfg_stored(self, dice_method, dice_cfg):
        assert dice_method.cfg == dice_cfg


# ---------------------------------------------------------------------------
# DiCEMethod.generate tests
# ---------------------------------------------------------------------------

class TestDiCEMethodGenerate:
    def test_generate_returns_result(self, dice_method, adult_like_df, continuous_features):
        query = adult_like_df[continuous_features].iloc[[0]]
        result = dice_method.generate(query_instance=query, num_cfs=2)
        assert result is not None

    def test_generate_returns_dice_result(self, dice_method, adult_like_df, continuous_features):
        from src.cf_methods.dice_method import DiCEResult
        query = adult_like_df[continuous_features].iloc[[0]]
        result = dice_method.generate(query_instance=query, num_cfs=2)
        assert isinstance(result, DiCEResult)

    def test_to_dataframe_returns_df_or_none(self, dice_method, adult_like_df, continuous_features):
        query = adult_like_df[continuous_features].iloc[[0]]
        result = dice_method.generate(query_instance=query, num_cfs=2)
        df = result.to_dataframe()
        assert df is None or isinstance(df, pd.DataFrame)

    def test_generate_correct_number_of_cfs(self, dice_method, adult_like_df, continuous_features):
        query = adult_like_df[continuous_features].iloc[[0]]
        num_cfs = 3
        result = dice_method.generate(query_instance=query, num_cfs=num_cfs)
        cfs_df = result.to_dataframe()
        assert len(cfs_df) == num_cfs

    def test_generate_cfs_have_correct_columns(self, dice_method, adult_like_df, continuous_features):
        query = adult_like_df[continuous_features].iloc[[0]]
        result = dice_method.generate(query_instance=query, num_cfs=2)
        cfs_df = result.to_dataframe()
        for col in continuous_features:
            assert col in cfs_df.columns

    def test_generate_cfs_exclude_target(self, dice_method, adult_like_df, continuous_features):
        query = adult_like_df[continuous_features].iloc[[0]]
        result = dice_method.generate(query_instance=query, num_cfs=2)
        cfs_df = result.to_dataframe()
        assert "income" not in cfs_df.columns


# ---------------------------------------------------------------------------
# NICEMethod tests
# ---------------------------------------------------------------------------

class TestNICEMethodConstruction:
    def test_instantiation(self, nice_method):
        assert isinstance(nice_method, NICEMethod)

    def test_explainer_is_set(self, nice_method):
        assert nice_method.explainer is not None

    def test_feature_names(self, nice_method):
        assert nice_method.feature_names == ['age', 'hours_per_week', 'workclass', 'education']

    def test_cfg_stored(self, nice_method, nice_cfg):
        assert nice_method.cfg == nice_cfg


class TestNICEMethodGenerate:
    def test_generate_returns_result(self, nice_method, adult_mixed_df, mixed_continuous):
        feat_cols = nice_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = nice_method.generate(query_instance=query, num_cfs=2)
        assert result is not None

    def test_generate_returns_nice_result(self, nice_method, adult_mixed_df):
        from src.cf_methods.nice_method import NICEResult
        feat_cols = nice_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = nice_method.generate(query_instance=query, num_cfs=2)
        assert isinstance(result, NICEResult)

    def test_to_dataframe_returns_df_or_none(self, nice_method, adult_mixed_df):
        feat_cols = nice_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = nice_method.generate(query_instance=query, num_cfs=2)
        df = result.to_dataframe()
        assert df is None or isinstance(df, pd.DataFrame)

    def test_generate_cfs_have_correct_columns(self, nice_method, adult_mixed_df):
        feat_cols = nice_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = nice_method.generate(query_instance=query, num_cfs=2)
        cfs_df = result.to_dataframe()
        if cfs_df is not None:
            for col in feat_cols:
                assert col in cfs_df.columns


# ---------------------------------------------------------------------------
# GSMethod tests
# ---------------------------------------------------------------------------

class TestGSMethodConstruction:
    def test_instantiation(self, gs_method):
        assert isinstance(gs_method, GSMethod)

    def test_feature_names(self, gs_method):
        assert gs_method.feature_names == ['age', 'hours_per_week', 'workclass', 'education']

    def test_cfg_stored(self, gs_method, gs_cfg):
        assert gs_method.cfg == gs_cfg


class TestGSMethodGenerate:
    def test_generate_returns_result(self, gs_method, adult_mixed_df):
        from src.cf_methods.gs_method import GSResult
        feat_cols = gs_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = gs_method.generate(query_instance=query, num_cfs=1)
        assert isinstance(result, GSResult)

    def test_to_dataframe_returns_df_or_none(self, gs_method, adult_mixed_df):
        feat_cols = gs_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = gs_method.generate(query_instance=query, num_cfs=1)
        df = result.to_dataframe()
        assert df is None or isinstance(df, pd.DataFrame)

    def test_generate_cfs_have_correct_columns(self, gs_method, adult_mixed_df):
        feat_cols = gs_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = gs_method.generate(query_instance=query, num_cfs=1)
        cfs_df = result.to_dataframe()
        if cfs_df is not None:
            for col in feat_cols:
                assert col in cfs_df.columns


# ---------------------------------------------------------------------------
# MOCMethod tests
# ---------------------------------------------------------------------------

class TestMOCMethodConstruction:
    def test_instantiation(self, moc_method):
        assert isinstance(moc_method, MOCMethod)

    def test_feature_names(self, moc_method):
        assert moc_method.feature_names == ['age', 'hours_per_week', 'workclass', 'education']

    def test_cfg_stored(self, moc_method, moc_cfg):
        assert moc_method.cfg == moc_cfg


class TestMOCMethodGenerate:
    def test_generate_returns_result(self, moc_method, adult_mixed_df):
        from src.cf_methods.moc_method import MOCResult
        feat_cols = moc_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = moc_method.generate(query_instance=query, num_cfs=3)
        assert isinstance(result, MOCResult)

    def test_to_dataframe_returns_df_or_none(self, moc_method, adult_mixed_df):
        feat_cols = moc_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = moc_method.generate(query_instance=query, num_cfs=3)
        df = result.to_dataframe()
        assert df is None or isinstance(df, pd.DataFrame)

    def test_generate_respects_num_cfs(self, moc_method, adult_mixed_df):
        feat_cols = moc_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = moc_method.generate(query_instance=query, num_cfs=2)
        cfs_df = result.to_dataframe()
        if cfs_df is not None:
            assert len(cfs_df) <= 2

    def test_generate_cfs_have_correct_columns(self, moc_method, adult_mixed_df):
        feat_cols = moc_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = moc_method.generate(query_instance=query, num_cfs=3)
        cfs_df = result.to_dataframe()
        if cfs_df is not None:
            for col in feat_cols:
                assert col in cfs_df.columns


# ---------------------------------------------------------------------------
# LOREMethod tests
# ---------------------------------------------------------------------------

class TestLOREMethodConstruction:
    def test_instantiation(self, lore_method):
        assert isinstance(lore_method, LOREMethod)

    def test_feature_names(self, lore_method):
        assert lore_method.feature_names == ['age', 'hours_per_week', 'workclass', 'education']

    def test_cfg_stored(self, lore_method, lore_cfg):
        assert lore_method.cfg == lore_cfg


class TestLOREMethodGenerate:
    def test_generate_returns_result(self, lore_method, adult_mixed_df):
        from src.cf_methods.lore_method import LOREResult
        feat_cols = lore_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = lore_method.generate(query_instance=query, num_cfs=3)
        assert isinstance(result, LOREResult)

    def test_to_dataframe_returns_df_or_none(self, lore_method, adult_mixed_df):
        feat_cols = lore_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = lore_method.generate(query_instance=query, num_cfs=3)
        df = result.to_dataframe()
        assert df is None or isinstance(df, pd.DataFrame)

    def test_generate_respects_num_cfs(self, lore_method, adult_mixed_df):
        feat_cols = lore_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = lore_method.generate(query_instance=query, num_cfs=2)
        cfs_df = result.to_dataframe()
        if cfs_df is not None:
            assert len(cfs_df) <= 2

    def test_generate_cfs_have_correct_columns(self, lore_method, adult_mixed_df):
        feat_cols = lore_method.feature_names
        query = adult_mixed_df[feat_cols].iloc[[0]]
        result = lore_method.generate(query_instance=query, num_cfs=3)
        cfs_df = result.to_dataframe()
        if cfs_df is not None:
            for col in feat_cols:
                assert col in cfs_df.columns
