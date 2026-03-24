import pytest
import numpy as np
import pandas as pd

from src.cf_methods.registry import METHOD_REGISTRY, register_method, get_method_class, create_method
from src.cf_methods.base_cf_method import BaseCounterfactualGenerationMethod
from src.cf_methods.dice_method import DiCEMethod  # triggers registration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adult_like_df():
    """Small synthetic dataframe that mimics Adult dataset structure."""
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        'age':            np.random.randint(18, 70, n).astype(float),
        'hours_per_week': np.random.randint(1, 80, n).astype(float),
        'education_num':  np.random.randint(1, 16, n).astype(float),
        'income':         np.random.randint(0, 2, n),
    })


@pytest.fixture
def continuous_features():
    return ['age', 'hours_per_week', 'education_num']


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
def tiny_model(adult_like_df, continuous_features):
    """A trivial nn.Module that returns a probability for each row."""
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
def dice_method(dice_cfg, tiny_model, adult_like_df, continuous_features):
    return DiCEMethod(
        cfg=dice_cfg,
        model=tiny_model,
        dataframe=adult_like_df,
        target_column='income',
        continuous_features=continuous_features,
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_dice_is_registered(self):
        assert 'dice' in METHOD_REGISTRY

    def test_get_method_class_returns_dice(self):
        cls = get_method_class('dice')
        assert cls is DiCEMethod

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

    def test_generate_correct_number_of_cfs(self, dice_method, adult_like_df, continuous_features):
        query = adult_like_df[continuous_features].iloc[[0]]
        num_cfs = 3
        result = dice_method.generate(query_instance=query, num_cfs=num_cfs)
        cfs_df = result.cf_examples_list[0].final_cfs_df
        assert len(cfs_df) == num_cfs

    def test_generate_cfs_have_correct_columns(self, dice_method, adult_like_df, continuous_features):
        query = adult_like_df[continuous_features].iloc[[0]]
        result = dice_method.generate(query_instance=query, num_cfs=2)
        cfs_df = result.cf_examples_list[0].final_cfs_df
        for col in continuous_features:
            assert col in cfs_df.columns
