from pathlib import Path
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
from scipy.io import arff
from ..registry import register_dataset
from sklearn.model_selection import train_test_split


def preprocess_compas_dataset(df: pd.DataFrame) -> pd.DataFrame:
    age_cat_columns = ['age_cat_25-45', 'age_cat_Greaterthan45', 'age_cat_Lessthan25']
    df['age_cat'] = df[age_cat_columns].idxmax(axis=1).str.replace('age_cat_', '')
    race_columns = ['race_African-American', 'race_Caucasian']
    df['race'] = df[race_columns].idxmax(axis=1).str.replace('race_', '')
    charge_degree_columns = ['c_charge_degree_F', 'c_charge_degree_M']
    df['c_charge_degree'] = df[charge_degree_columns].idxmax(axis=1).str.replace('c_charge_degree_', '')
    df['sex'] = df["sex"].map({0: "Female", 1: "Male"})
    df = df.drop(columns=age_cat_columns + race_columns + charge_degree_columns)
    return df


@register_dataset(name='compas-recidivism')
def load_compas(dataset_cfg: dict) -> pd.DataFrame:

    train_size = dataset_cfg['specs']['train_size'] / 100.0
    test_size = 1.0 - train_size
    train_only = dataset_cfg['specs'].get('train_only', False)
    dataset_url = dataset_cfg['dataset']['url']
    root_download_dir = Path(dataset_cfg["download_location"])
    dataset_dir_name = dataset_cfg["dataset"]["name"]
    dataset_dir = root_download_dir / dataset_dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    arff_filepath = dataset_dir / f"{dataset_dir_name}.arff"
    if not arff_filepath.is_file():
        urlretrieve(dataset_url, arff_filepath)
    arff_data = arff.loadarff(arff_filepath)
    df = pd.DataFrame(arff_data[0])
    byte_string_cols = [col for col in df.columns if df[col].dtype == "object"]
    df[byte_string_cols] = df[byte_string_cols].map(lambda x: int(x.decode("utf-8")))
    df = preprocess_compas_dataset(df)
    cols_to_get = ["age", "sex", "race", "priors_count", "c_charge_degree",
                   "twoyearrecid"]
    drop_cols = df.columns.difference(cols_to_get)
    df.drop(columns=drop_cols, inplace=True)

    if train_only:
        train, _ = train_test_split(df, test_size=test_size,
                                    random_state=42)
        df = train.reset_index(drop=True)

    return df
