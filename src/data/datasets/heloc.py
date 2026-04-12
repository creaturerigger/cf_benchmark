import pandas as pd
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
from scipy.io import arff
from ..registry import register_dataset
from sklearn.model_selection import train_test_split


def _preprocess_heloc(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the HELOC dataset.

    The original FICO dataset uses special sentinel values:
        -9  → no bureau record / no investigation
        -8  → no usable / valid trades or inquiries
        -7  → condition not met (e.g. no inquiries in last 6 months)

    All three are treated as missing and replaced with the column median.
    """
    feature_cols = [c for c in df.columns if c != "RiskPerformance"]

    # Replace sentinel values (-9, -8, -7) with NaN
    df[feature_cols] = df[feature_cols].replace([-9, -8, -7], np.nan)

    # Impute with median
    for col in feature_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Ensure integer-valued features stay int
    df[feature_cols] = df[feature_cols].astype(int)

    # Target: 0 = Bad, 1 = Good  (already encoded as 0/1 in OpenML version)
    df["RiskPerformance"] = df["RiskPerformance"].astype(int)

    return df


@register_dataset(name="heloc")
def load_heloc(dataset_cfg: dict) -> pd.DataFrame:
    """Load the FICO HELOC (Home Equity Line of Credit) dataset.

    Source (OpenML mirror): https://www.openml.org/d/45026
    10 000 rows, 22 integer features (all continuous), binary target.
    Task: predict whether a HELOC applicant will repay within 2 years.
    """
    train_size = dataset_cfg["specs"]["train_size"] / 100.0
    test_size = 1.0 - train_size
    train_only = dataset_cfg["specs"].get("train_only", False)
    dataset_url = dataset_cfg["dataset"]["url"]
    root_download_dir = Path(dataset_cfg["download_location"])
    dataset_dir_name = dataset_cfg["dataset"]["name"]
    dataset_dir = root_download_dir / dataset_dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    arff_filepath = dataset_dir / "heloc.arff"
    if not arff_filepath.is_file():
        urlretrieve(dataset_url, arff_filepath)

    arff_data, _ = arff.loadarff(arff_filepath)
    df = pd.DataFrame(arff_data)

    # Decode byte-string columns (RiskPerformance comes as bytes from ARFF)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.decode("utf-8")

    df = _preprocess_heloc(df)

    if train_only:
        train, _ = train_test_split(df, test_size=test_size, random_state=42)
        df = train.reset_index(drop=True)

    return df
