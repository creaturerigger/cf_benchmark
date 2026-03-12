import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


class PYTDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, target_column: str,
                 scaler=None, encoder=None, target_encoder=None,
                 test_size=0.2, train=True):
        self.dataframe = dataframe.copy()
        self.target_column = target_column
        self.scaler = scaler
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.test_size = test_size
        self.train = train
        self.train_features_tensor, self.test_features_tensor, \
            self.y_train_tensor, self.y_test_tensor = None, None, None, None
        self.train_dataset_df, self.test_dataset_df, \
            self.y_train_df, self.y_test_df = None, None, None, None
        self.preprocess_for_torch_training()

    def preprocess_for_torch_training(self):

        self.features = self.dataframe.drop(columns=[self.target_column])
        self.target = self.dataframe[self.target_column]

        numerical_cols = self.features.select_dtypes(include=[np.number]).columns
        categorical_cols = self.features.columns.difference(numerical_cols)
        if self.scaler is None and len(numerical_cols) > 0:
            self.scaler = StandardScaler().fit(self.features[numerical_cols])

        if self.encoder is None and len(categorical_cols) > 0:
            self.encoder = OneHotEncoder(sparse_output=False,
                                         handle_unknown='ignore') \
                                         .fit(self.features[categorical_cols])

        if len(numerical_cols) > 0:
            self.features[numerical_cols] = self.scaler \
                                            .transform(self.features[numerical_cols])

        if len(categorical_cols) > 0:
            encoded_cats = self.encoder.transform(self.features[categorical_cols])
            encoded_cat_df = pd.DataFrame(encoded_cats, columns=self.encoder.get_feature_names_out())
            encoded_cat_df.index = self.features.index
            self.features = self.features.drop(columns=list(categorical_cols))
            self.features = pd.concat([self.features, encoded_cat_df], axis=1)

        if not pd.api.types.is_numeric_dtype(self.target):
            if self.target_encoder is None:
                self.target_encoder = LabelEncoder()
                self.target_encoder.fit(self.target)
            self.target = self.target_encoder.transform(self.target)
        else:
            self.target_encoder = None

        self.features = torch.tensor(self.features.values, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.long)
        combined_features = torch.cat((self.features, self.target.unsqueeze(1)), dim=1)
        self.train_features_tensor, self.test_features_tensor, \
        self.y_train_tensor, self.y_test_tensor = train_test_split(combined_features,
                                                                    self.target, test_size=self.test_size,
                                                                    stratify=combined_features[:, -1],
                                                                    random_state=42)
        self.train_dataset_df, self.test_dataset_df, \
        self.y_train_df, self.y_test_df = train_test_split(self.dataframe, self.dataframe[self.target_column],
                                                            test_size=self.test_size,
                                                            stratify=self.dataframe[self.target_column],
                                                            random_state=42)

    def __len__(self):
        if self.train:
            return len(self.y_train_tensor)
        else:
            return len(self.y_test_tensor)

    def __getitem__(self, position):
        if self.train:
            train_features = self.train_features_tensor[:, :-1]
            return train_features[position], self.y_train_tensor[position]
        else:
            test_features = self.test_features_tensor[:, :-1]
            return test_features[position], self.y_test_tensor[position]