import pandas as pd
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
from ..registry import register_dataset
from sklearn.model_selection import train_test_split


# Admission types to exclude (death / hospice — no meaningful readmission).
_EXCLUDE_DISCHARGE = {11, 13, 14, 19, 20, 21}

# Features to keep — a curated subset suitable for counterfactual explanations.
# Avoids high-cardinality ICD codes, 24+ medication flags, and mostly-missing cols.
_KEEP_COLS = [
    "race", "gender", "age",
    "time_in_hospital",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses",
    "max_glu_serum", "A1Cresult",
    "change", "diabetesMed",
    "readmitted",
]

_AGE_MAP = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}


def _preprocess_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the Diabetes 130-US Hospitals dataset.

    Reference: Strack et al. (2014), *BioMed Research International*.
    Steps mirror the original paper's preprocessing:
      1. Remove duplicate patients (keep first encounter).
      2. Remove encounters where patient died or went to hospice.
      3. Map age ranges to midpoints (continuous).
      4. Binary target: readmitted <30 days → 1, else → 0.
    """
    # Replace "?" with NaN
    df = df.replace("?", np.nan)

    # Drop encounters where patient died / discharged to hospice
    df["discharge_disposition_id"] = df["discharge_disposition_id"].astype(int)
    df = df[~df["discharge_disposition_id"].isin(_EXCLUDE_DISCHARGE)]

    # Keep first encounter per patient
    df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")

    # Select features
    df = df[_KEEP_COLS].copy()

    # --- Age: ordinal bins → continuous midpoint ---
    df["age"] = df["age"].map(_AGE_MAP).astype(int)

    # --- Race: fill missing, collapse rare ---
    df["race"] = df["race"].fillna("Other")

    # --- Gender: drop unknown/invalid (very few rows) ---
    df = df[df["gender"] != "Unknown/Invalid"]

    # --- Glucose & A1C: "None" means not measured ---
    # Use "NotMeasured" to avoid pandas CSV round-trip interpreting "None" as NaN
    df["max_glu_serum"] = df["max_glu_serum"].fillna("NotMeasured")
    df["A1Cresult"] = df["A1Cresult"].fillna("NotMeasured")

    # --- Binary target: <30 → 1, else → 0 ---
    df["readmitted"] = (df["readmitted"] == "<30").astype(int)

    # Continuous columns to int
    int_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
    ]
    df[int_cols] = df[int_cols].astype(int)

    # Ensure categorical columns are object dtype (not pandas StringDtype)
    # to stay compatible with sklearn OneHotEncoder
    cat_cols = ["race", "gender", "max_glu_serum", "A1Cresult", "change", "diabetesMed"]
    df[cat_cols] = df[cat_cols].astype(object)

    df = df.reset_index(drop=True)
    return df


@register_dataset(name="diabetes")
def load_diabetes(dataset_cfg: dict) -> pd.DataFrame:
    """Load the Diabetes 130-US Hospitals dataset (Strack et al. 2014).

    Source (UCI ML Repository #296):
        https://archive.ics.uci.edu/dataset/296
    ~101k rows → ~70k after dedup + exclusions, 14 features, binary target.
    Task: predict whether a diabetic patient is readmitted within 30 days.
    """
    train_size = dataset_cfg["specs"]["train_size"] / 100.0
    test_size = 1.0 - train_size
    train_only = dataset_cfg["specs"].get("train_only", False)
    dataset_url = dataset_cfg["dataset"]["url"]
    root_download_dir = Path(dataset_cfg["download_location"])
    dataset_dir_name = dataset_cfg["dataset"]["name"]
    dataset_dir = root_download_dir / dataset_dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    csv_filepath = dataset_dir / "diabetic_data.csv"
    if not csv_filepath.is_file():
        zipfilepath = dataset_dir / "diabetes.zip"
        urlretrieve(dataset_url, zipfilepath)
        with zipfile.ZipFile(zipfilepath, "r") as zf:
            zf.extractall(dataset_dir)
        zipfilepath.unlink(missing_ok=True)

    df = pd.read_csv(csv_filepath, low_memory=False)
    df = _preprocess_diabetes(df)

    if train_only:
        train, _ = train_test_split(df, test_size=test_size, random_state=42)
        df = train.reset_index(drop=True)

    return df
