import pytest
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from src.data.preprocessing.py_dataset import PYTDataset


@pytest.fixture
def numeric_dataframe():
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.choice([0, 1], size=100),
    })


@pytest.fixture
def categorical_dataframe():
    np.random.seed(42)
    return pd.DataFrame({
        'color': np.random.choice(['red', 'blue', 'green'], size=100),
        'size': np.random.choice(['S', 'M', 'L'], size=100),
        'target': np.random.choice(['yes', 'no'], size=100),
    })


@pytest.fixture
def mixed_dataframe():
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'color': np.random.choice(['red', 'blue', 'green'], size=100),
        'size': np.random.choice(['S', 'M', 'L'], size=100),
        'target': np.random.choice([0, 1], size=100),
    })


class TestPYTDatasetInit:
    def test_init_stores_target_column(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert ds.target_column == 'target'

    def test_init_does_not_modify_original_dataframe(self, numeric_dataframe):
        original = numeric_dataframe.copy()
        PYTDataset(numeric_dataframe, target_column='target')
        pd.testing.assert_frame_equal(numeric_dataframe, original)

    def test_init_default_test_size(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert ds.test_size == 0.2

    def test_init_custom_test_size(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', test_size=0.3)
        assert ds.test_size == 0.3

    def test_init_default_train_mode(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert ds.train is True

    def test_init_test_mode(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=False)
        assert ds.train is False


class TestNumericData:
    def test_scaler_is_created(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert ds.scaler is not None
        assert isinstance(ds.scaler, StandardScaler)

    def test_encoder_is_none_for_numeric_only(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert ds.encoder is None

    def test_target_encoder_is_none_for_numeric_target(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert ds.target_encoder is None

    def test_train_test_split_sizes(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', test_size=0.2)
        assert len(ds.y_train_tensor) == 80
        assert len(ds.y_test_tensor) == 20

    def test_train_features_tensor_shape(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        # 3 numeric features, last column is target appended
        assert ds.train_features_tensor.shape[1] == 4  # 3 features + 1 target
        assert ds.train_features_tensor.shape[0] == 80

    def test_test_features_tensor_shape(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert ds.test_features_tensor.shape[1] == 4
        assert ds.test_features_tensor.shape[0] == 20

    def test_features_are_float32(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert ds.train_features_tensor.dtype == torch.float32

    def test_target_is_long(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert ds.y_train_tensor.dtype == torch.long
        assert ds.y_test_tensor.dtype == torch.long

    def test_custom_scaler_is_used(self, numeric_dataframe):
        features = numeric_dataframe.drop(columns=['target'])
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        custom_scaler = StandardScaler().fit(features[numerical_cols])
        ds = PYTDataset(numeric_dataframe, target_column='target', scaler=custom_scaler)
        assert ds.scaler is custom_scaler


class TestCategoricalData:
    def test_encoder_is_created(self, categorical_dataframe):
        ds = PYTDataset(categorical_dataframe, target_column='target')
        assert ds.encoder is not None
        assert isinstance(ds.encoder, OneHotEncoder)

    def test_target_encoder_is_created_for_string_target(self, categorical_dataframe):
        ds = PYTDataset(categorical_dataframe, target_column='target')
        assert ds.target_encoder is not None
        assert isinstance(ds.target_encoder, LabelEncoder)

    def test_target_values_are_encoded(self, categorical_dataframe):
        ds = PYTDataset(categorical_dataframe, target_column='target')
        unique_targets = ds.y_train_tensor.unique().tolist()
        for t in unique_targets:
            assert t in [0, 1]

    def test_one_hot_encoded_features_shape(self, categorical_dataframe):
        ds = PYTDataset(categorical_dataframe, target_column='target')
        # 'color' has 3 categories, 'size' has 3 categories -> 6 one-hot features
        # plus 1 target column appended
        assert ds.train_features_tensor.shape[1] == 7

    def test_custom_encoder_is_used(self, categorical_dataframe):
        features = categorical_dataframe.drop(columns=['target'])
        categorical_cols = features.columns
        custom_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(features[categorical_cols])
        ds = PYTDataset(categorical_dataframe, target_column='target', encoder=custom_encoder)
        assert ds.encoder is custom_encoder

    def test_custom_target_encoder_is_used(self, categorical_dataframe):
        custom_target_encoder = LabelEncoder()
        custom_target_encoder.fit(categorical_dataframe['target'])
        ds = PYTDataset(categorical_dataframe, target_column='target', target_encoder=custom_target_encoder)
        assert ds.target_encoder is custom_target_encoder


class TestCategoryDtypeTarget:
    def test_category_dtype_target_is_encoded(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'target': pd.Categorical(np.random.choice(['cat', 'dog'], size=100)),
        })
        ds = PYTDataset(df, target_column='target')
        assert ds.target_encoder is not None
        assert isinstance(ds.target_encoder, LabelEncoder)
        unique_targets = ds.y_train_tensor.unique().tolist()
        for t in unique_targets:
            assert t in [0, 1]


class TestMixedData:
    def test_scaler_and_encoder_both_created(self, mixed_dataframe):
        ds = PYTDataset(mixed_dataframe, target_column='target')
        assert ds.scaler is not None
        assert ds.encoder is not None

    def test_mixed_features_shape(self, mixed_dataframe):
        ds = PYTDataset(mixed_dataframe, target_column='target')
        # 2 numeric + 3 (color) + 3 (size) = 8 features + 1 target appended
        assert ds.train_features_tensor.shape[1] == 9

    def test_target_encoder_none_for_numeric_target_mixed(self, mixed_dataframe):
        ds = PYTDataset(mixed_dataframe, target_column='target')
        assert ds.target_encoder is None


class TestLen:
    def test_len_train_mode(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=True)
        assert len(ds) == 80

    def test_len_test_mode(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=False)
        assert len(ds) == 20

    def test_len_custom_test_size(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', test_size=0.3, train=True)
        assert len(ds) == 70

    def test_len_custom_test_size_test_mode(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', test_size=0.3, train=False)
        assert len(ds) == 30


class TestGetItem:
    def test_getitem_train_returns_tuple(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=True)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_getitem_test_returns_tuple(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=False)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_getitem_train_feature_shape(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=True)
        features, target = ds[0]
        assert features.shape == (3,)  # 3 numeric features

    def test_getitem_test_feature_shape(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=False)
        features, target = ds[0]
        assert features.shape == (3,)

    def test_getitem_train_target_is_scalar(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=True)
        _, target = ds[0]
        assert target.dim() == 0

    def test_getitem_test_target_is_scalar(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=False)
        _, target = ds[0]
        assert target.dim() == 0

    def test_getitem_features_are_float32(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=True)
        features, _ = ds[0]
        assert features.dtype == torch.float32

    def test_getitem_target_is_long(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=True)
        _, target = ds[0]
        assert target.dtype == torch.long

    def test_getitem_all_train_indices(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=True)
        for i in range(len(ds)):
            features, target = ds[i]
            assert features.shape == (3,)

    def test_getitem_all_test_indices(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=False)
        for i in range(len(ds)):
            features, target = ds[i]
            assert features.shape == (3,)

    def test_getitem_mixed_train_feature_shape(self, mixed_dataframe):
        ds = PYTDataset(mixed_dataframe, target_column='target', train=True)
        features, _ = ds[0]
        # 2 numeric + 3 (color) + 3 (size) = 8
        assert features.shape == (8,)

    def test_getitem_categorical_train_feature_shape(self, categorical_dataframe):
        ds = PYTDataset(categorical_dataframe, target_column='target', train=True)
        features, _ = ds[0]
        # 3 (color) + 3 (size) = 6
        assert features.shape == (6,)


class TestDataFrameSplits:
    def test_train_dataset_df_shape(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert ds.train_dataset_df.shape[0] == 80
        assert ds.train_dataset_df.shape[1] == numeric_dataframe.shape[1]

    def test_test_dataset_df_shape(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert ds.test_dataset_df.shape[0] == 20
        assert ds.test_dataset_df.shape[1] == numeric_dataframe.shape[1]

    def test_y_train_df_length(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert len(ds.y_train_df) == 80

    def test_y_test_df_length(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        assert len(ds.y_test_df) == 20

    def test_train_test_df_columns_preserved(self, mixed_dataframe):
        ds = PYTDataset(mixed_dataframe, target_column='target')
        assert list(ds.train_dataset_df.columns) == list(mixed_dataframe.columns)
        assert list(ds.test_dataset_df.columns) == list(mixed_dataframe.columns)


class TestStratification:
    def test_stratified_split_preserves_class_ratios(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target')
        original_ratio = numeric_dataframe['target'].value_counts(normalize=True).sort_index()
        train_ratio = ds.y_train_df.value_counts(normalize=True).sort_index()
        for cls in original_ratio.index:
            assert abs(original_ratio[cls] - train_ratio[cls]) < 0.1


class TestReproducibility:
    def test_same_random_state_produces_same_split(self, numeric_dataframe):
        ds1 = PYTDataset(numeric_dataframe, target_column='target')
        ds2 = PYTDataset(numeric_dataframe, target_column='target')
        assert torch.equal(ds1.train_features_tensor, ds2.train_features_tensor)
        assert torch.equal(ds1.test_features_tensor, ds2.test_features_tensor)
        assert torch.equal(ds1.y_train_tensor, ds2.y_train_tensor)
        assert torch.equal(ds1.y_test_tensor, ds2.y_test_tensor)


class TestDataLoader:
    def test_works_with_dataloader_train(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
        total = 0
        for features, targets in loader:
            assert features.dtype == torch.float32
            assert targets.dtype == torch.long
            total += features.shape[0]
        assert total == 80

    def test_works_with_dataloader_test(self, numeric_dataframe):
        ds = PYTDataset(numeric_dataframe, target_column='target', train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
        total = 0
        for features, targets in loader:
            total += features.shape[0]
        assert total == 20


class TestEdgeCases:
    def test_single_numeric_feature(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'target': np.random.choice([0, 1], size=50),
        })
        ds = PYTDataset(df, target_column='target')
        features, target = ds[0]
        assert features.shape == (1,)

    def test_many_categories(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'cat_col': [f'cat_{i % 10}' for i in range(100)],
            'target': np.random.choice([0, 1], size=100),
        })
        ds = PYTDataset(df, target_column='target')
        features, _ = ds[0]
        assert features.shape == (10,)  # 10 one-hot encoded categories

    def test_multiclass_target(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(150),
            'target': np.random.choice([0, 1, 2], size=150),
        })
        ds = PYTDataset(df, target_column='target')
        unique_targets = ds.y_train_tensor.unique()
        assert len(unique_targets) == 3

    def test_multiclass_string_target(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(150),
            'target': np.random.choice(['a', 'b', 'c'], size=150),
        })
        ds = PYTDataset(df, target_column='target')
        assert ds.target_encoder is not None
        unique_targets = ds.y_train_tensor.unique()
        assert len(unique_targets) == 3
