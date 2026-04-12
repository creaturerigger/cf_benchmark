import pandas as pd
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
from scipy.io import arff
from ..registry import register_dataset
from sklearn.model_selection import train_test_split


# Mapping from generic OpenML column names to the original dataset names.
_COLUMN_RENAME = {
    "x1": "limit_bal",
    "x2": "sex",
    "x3": "education",
    "x4": "marriage",
    "x5": "age",
    "x6": "pay_0",
    "x7": "pay_2",
    "x8": "pay_3",
    "x9": "pay_4",
    "x10": "pay_5",
    "x11": "pay_6",
    "x12": "bill_amt1",
    "x13": "bill_amt2",
    "x14": "bill_amt3",
    "x15": "bill_amt4",
    "x16": "bill_amt5",
    "x17": "bill_amt6",
    "x18": "pay_amt1",
    "x19": "pay_amt2",
    "x20": "pay_amt3",
    "x21": "pay_amt4",
    "x22": "pay_amt5",
    "x23": "pay_amt6",
    "y": "default",
}


def _preprocess_credit_default(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the Default of Credit Card Clients dataset.

    Reference: Yeh & Lien (2009), *Expert Systems with Applications* 36(2).
    30 000 rows, 23 features + binary target.
    """
    # Drop the row-ID column
    df = df.drop(columns=["id"])

    # Rename generic OpenML columns to meaningful names
    df = df.rename(columns=_COLUMN_RENAME)

    # --- Categorical features ---
    # sex: 1=male, 2=female
    df["sex"] = df["sex"].map({1.0: "Male", 2.0: "Female"})

    # education: 1=grad school, 2=university, 3=high school, 4+=others
    # Collapse undocumented codes (0, 5, 6) into "Other"
    edu_map = {1.0: "Graduate", 2.0: "University", 3.0: "HighSchool"}
    df["education"] = df["education"].map(edu_map).fillna("Other")

    # marriage: 1=married, 2=single, 3=others; 0 is undocumented → "Other"
    mar_map = {1.0: "Married", 2.0: "Single"}
    df["marriage"] = df["marriage"].map(mar_map).fillna("Other")

    # target: string "0"/"1" → int
    df["default"] = df["default"].astype(int)

    # Continuous columns to int (amounts and counts)
    int_cols = [
        "limit_bal", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "bill_amt1", "bill_amt2", "bill_amt3",
        "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3",
        "pay_amt4", "pay_amt5", "pay_amt6",
    ]
    df[int_cols] = df[int_cols].astype(int)

    return df


@register_dataset(name="credit-default")
def load_credit_default(dataset_cfg: dict) -> pd.DataFrame:
    """Load the Default of Credit Card Clients dataset (Yeh & Lien 2009).

    Source (OpenML mirror of UCI #350): https://www.openml.org/d/42477
    30 000 rows, 20 continuous + 3 categorical features, binary target.
    Task: predict whether a client will default on the next monthly payment.
    """
    train_size = dataset_cfg["specs"]["train_size"] / 100.0
    test_size = 1.0 - train_size
    train_only = dataset_cfg["specs"].get("train_only", False)
    dataset_url = dataset_cfg["dataset"]["url"]
    root_download_dir = Path(dataset_cfg["download_location"])
    dataset_dir_name = dataset_cfg["dataset"]["name"]
    dataset_dir = root_download_dir / dataset_dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    arff_filepath = dataset_dir / "default-of-credit-card-clients.arff"
    if not arff_filepath.is_file():
        urlretrieve(dataset_url, arff_filepath)

    arff_data, _ = arff.loadarff(arff_filepath)
    df = pd.DataFrame(arff_data)

    # Decode byte-string columns (target 'y' comes as bytes from ARFF)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.decode("utf-8")

    df = _preprocess_credit_default(df)

    if train_only:
        train, _ = train_test_split(df, test_size=test_size, random_state=42)
        df = train.reset_index(drop=True)

    return df
