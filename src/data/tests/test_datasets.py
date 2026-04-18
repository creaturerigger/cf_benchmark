import yaml
import pytest
import numpy as np
from src.data.data_module import load_dataset

DATASET_SPECS = [
    (
        "configs/dataset/adult.yaml",
        {"age", "workclass", "education", "marital_status",
         "occupation", "race", "gender", "hours_per_week", "income"},
        "income",
    ),
    (
        "configs/dataset/compas.yaml",
        {"sex", "age", "priors_count", "twoyearrecid", "race", "c_charge_degree"},
        "twoyearrecid",
    ),
    (
        "configs/dataset/german.yaml",
        {"status_of_existing_checking_account", "duration_in_month",
         "credit_history", "purpose", "credit_amount",
         "savings_account_bonds", "present_employment_since",
         "installment_rate_in_percentage_of_disposable_income",
         "personal_status_and_sex", "other_debtors_guarantors",
         "present_residence_since", "property", "age_in_years",
         "other_installment_plans", "housing",
         "number_of_existing_credits_at_this_bank", "job",
         "number_of_people_being_liable_to_provide_maintenance_for",
         "telephone", "foreign_worker", "credit_risk"},
        "credit_risk",
    ),
    (
        "configs/dataset/lending.yaml",
        {"employment_years", "num_open_credit_acc", "annual_income",
         "loan_grade", "credit_history", "purpose", "home", "addr_state",
         "loan_status"},
        "loan_status",
    ),
    (
        "configs/dataset/heloc.yaml",
        {"ExternalRiskEstimate", "MSinceOldestTradeOpen",
         "MSinceMostRecentTradeOpen", "AverageMInFile",
         "NumSatisfactoryTrades", "NumTrades60Ever2DerogPubRec",
         "NumTrades90Ever2DerogPubRec", "PercentTradesNeverDelq",
         "MSinceMostRecentDelq", "MaxDelq2PublicRecLast12M",
         "NumTotalTrades", "NumTradesOpeninLast12M",
         "PercentInstallTrades", "MSinceMostRecentInqexcl7days",
         "NumInqLast6M", "NumInqLast6Mexcl7days",
         "NetFractionRevolvingBurden", "NetFractionInstallBurden",
         "NumRevolvingTradesWBalance", "NumInstallTradesWBalance",
         "NumBank2NatlTradesWHighUtilization", "PercentTradesWBalance",
         "RiskPerformance"},
        "RiskPerformance",
    ),
    (
        "configs/dataset/diabetes.yaml",
        {"race", "gender", "age",
         "time_in_hospital", "num_lab_procedures", "num_procedures",
         "num_medications", "number_outpatient", "number_emergency",
         "number_inpatient", "number_diagnoses",
         "max_glu_serum", "A1Cresult", "change", "diabetesMed",
         "readmitted"},
        "readmitted",
    ),
]


@pytest.fixture(params=DATASET_SPECS, ids=["adult", "compas",
                                           "german", "lending",
                                           "heloc", "diabetes"])
def dataset(request):
    """Dataset object that is created from the data in DATASET_SPECS."""
    config_path, expected_columns, target_col = request.param
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    df = load_dataset(cfg)
    return df, expected_columns, target_col


def test_shape(dataset):
    df, expected_columns, _ = dataset
    assert df.shape[1] == len(expected_columns)
    assert len(df) > 0


def test_columns(dataset):
    """Dataset must have the expected columns."""
    df, expected_columns, _ = dataset
    assert set(df.columns) == expected_columns


def test_target_binary(dataset):
    """Target column should be binary."""
    df, _, target_col = dataset
    assert set(df[target_col].unique()).issubset({0, 1})


def test_no_missing_values(dataset):
    """No dataset should have NaN after preprocessing."""
    df, _, _ = dataset
    assert df.isnull().sum().sum() == 0


# ── Adult-specific tests ─────────────────────────────────────

@pytest.fixture(scope="module")
def adult_df():
    with open("configs/dataset/adult.yaml") as f:
        cfg = yaml.safe_load(f)
    return load_dataset(cfg)


def test_adult_row_count(adult_df):
    """Raw adult dataset has ~32k rows; after loading all should remain."""
    assert len(adult_df) > 30_000


def test_adult_workclass_collapsed(adult_df):
    """Workclass should be collapsed to a small set of categories."""
    valid = {"Private", "Government", "Self-Employed", "Other/Unknown"}
    assert set(adult_df["workclass"].unique()).issubset(valid)


def test_adult_occupation_collapsed(adult_df):
    """Occupation should be collapsed into broad categories."""
    valid = {"White-Collar", "Blue-Collar", "Service", "Professional", "Sales", "Other/Unknown"}
    assert set(adult_df["occupation"].unique()).issubset(valid)


def test_adult_education_collapsed(adult_df):
    """Low-level education categories should be collapsed into 'School' and 'Assoc'."""
    edu_values = set(adult_df["education"].unique())
    # Raw categories like '11th', '10th', 'Assoc-voc' should not survive
    assert "11th" not in edu_values
    assert "Assoc-voc" not in edu_values
    assert "School" in edu_values


def test_adult_marital_status_collapsed(adult_df):
    """Marital status should be simplified."""
    values = set(adult_df["marital_status"].unique())
    assert "Married-civ-spouse" not in values
    assert "Married" in values


def test_adult_income_binary(adult_df):
    """Income target should be 0/1, not string labels."""
    assert set(adult_df["income"].unique()) == {0, 1}


def test_adult_columns_renamed(adult_df):
    """Hyphenated columns should be renamed to underscores."""
    assert "marital_status" in adult_df.columns
    assert "hours_per_week" in adult_df.columns
    assert "marital-status" not in adult_df.columns


# ── COMPAS-specific tests ────────────────────────────────────

@pytest.fixture(scope="module")
def compas_df():
    with open("configs/dataset/compas.yaml") as f:
        cfg = yaml.safe_load(f)
    return load_dataset(cfg)


def test_compas_row_count(compas_df):
    """COMPAS dataset should have a reasonable number of rows."""
    assert len(compas_df) > 4_000


def test_compas_sex_mapped(compas_df):
    """Sex should be mapped from 0/1 to Female/Male strings."""
    assert set(compas_df["sex"].unique()) == {"Female", "Male"}


def test_compas_race_from_onehot(compas_df):
    """Race should be reconstructed from one-hot columns."""
    valid = {"African-American", "Caucasian"}
    assert set(compas_df["race"].unique()).issubset(valid)


def test_compas_charge_degree_from_onehot(compas_df):
    """Charge degree should be reconstructed from one-hot columns."""
    assert set(compas_df["c_charge_degree"].unique()) == {"F", "M"}


def test_compas_no_onehot_cols_remain(compas_df):
    """One-hot source columns should be dropped."""
    for col in compas_df.columns:
        assert not col.startswith("age_cat_")
        assert not col.startswith("race_")
        assert not col.startswith("c_charge_degree_")


# ── German-specific tests ────────────────────────────────────

@pytest.fixture(scope="module")
def german_df():
    with open("configs/dataset/german.yaml") as f:
        cfg = yaml.safe_load(f)
    return load_dataset(cfg)


def test_german_row_count(german_df):
    """German credit dataset has exactly 1000 rows."""
    assert len(german_df) == 1000


def test_german_codes_mapped(german_df):
    """Raw category codes (A11, A12, …) should be replaced with human-readable labels."""
    # Check a few columns for residual codes
    for col in ("status_of_existing_checking_account", "credit_history", "purpose"):
        values = german_df[col].unique()
        for v in values:
            assert not (isinstance(v, str) and v.startswith("A")), \
                f"Unmapped code '{v}' in column '{col}'"


def test_german_credit_risk_mapped(german_df):
    """Credit risk should be mapped from {1, 2} to {0, 1}."""
    assert set(german_df["credit_risk"].unique()) == {0, 1}


def test_german_numeric_cols_int(german_df):
    """Numeric features should be cast to int."""
    int_cols = [
        "duration_in_month", "credit_amount",
        "installment_rate_in_percentage_of_disposable_income",
        "present_residence_since", "age_in_years",
        "number_of_existing_credits_at_this_bank",
        "number_of_people_being_liable_to_provide_maintenance_for",
    ]
    for col in int_cols:
        assert np.issubdtype(german_df[col].dtype, np.integer), \
            f"Column '{col}' has dtype {german_df[col].dtype}, expected integer"


# ── Lending-specific tests ───────────────────────────────────

@pytest.fixture(scope="module")
def lending_df():
    with open("configs/dataset/lending.yaml") as f:
        cfg = yaml.safe_load(f)
    return load_dataset(cfg)


def test_lending_row_count(lending_df):
    """Lending dataset should have a substantial number of rows."""
    assert len(lending_df) > 30_000


def test_lending_employment_years_int(lending_df):
    """employment_years should be parsed to int (raw is '< 1 year', '10+ years', etc.)."""
    assert np.issubdtype(lending_df["employment_years"].dtype, np.integer)


def test_lending_no_negative_credit_history(lending_df):
    """Rows with negative credit history should be dropped."""
    assert (lending_df["credit_history"] >= 0).all()


def test_lending_purpose_collapsed(lending_df):
    """Purpose should be collapsed into a small set."""
    valid = {"debt", "purchase", "small_business", "educational"}
    assert set(lending_df["purpose"].unique()).issubset(valid)


def test_lending_home_collapsed(lending_df):
    """Home ownership 'ANY'/'NONE' should be collapsed into 'OTHER'."""
    values = set(lending_df["home"].unique())
    assert "ANY" not in values
    assert "NONE" not in values


def test_lending_loan_status_binary(lending_df):
    """loan_status should be 0 (not paid) or 1 (fully paid)."""
    assert set(lending_df["loan_status"].unique()) == {0, 1}


# ── HELOC-specific tests ────────────────────────────────────

@pytest.fixture(scope="module")
def heloc_df():
    with open("configs/dataset/heloc.yaml") as f:
        cfg = yaml.safe_load(f)
    return load_dataset(cfg)


def test_heloc_row_count(heloc_df):
    """HELOC dataset has ~10k rows."""
    assert 9_000 <= len(heloc_df) <= 11_000


def test_heloc_all_numeric(heloc_df):
    """HELOC has no categorical features — all columns should be numeric."""
    for col in heloc_df.columns:
        assert np.issubdtype(heloc_df[col].dtype, np.number), \
            f"Column '{col}' has non-numeric dtype {heloc_df[col].dtype}"


def test_heloc_no_sentinel_values(heloc_df):
    """Sentinel values (-9, -8, -7) should be replaced during preprocessing."""
    feature_cols = [c for c in heloc_df.columns if c != "RiskPerformance"]
    for col in feature_cols:
        assert heloc_df[col].isin([-9, -8, -7]).sum() == 0, \
            f"Sentinel values remain in column '{col}'"


def test_heloc_all_int(heloc_df):
    """All HELOC features should be integer after imputation."""
    for col in heloc_df.columns:
        assert np.issubdtype(heloc_df[col].dtype, np.integer), \
            f"Column '{col}' has dtype {heloc_df[col].dtype}, expected integer"


# ── Diabetes-specific tests ──────────────────────────────────

@pytest.fixture(scope="module")
def diabetes_df():
    with open("configs/dataset/diabetes.yaml") as f:
        cfg = yaml.safe_load(f)
    return load_dataset(cfg)


def test_diabetes_no_duplicates(diabetes_df):
    """Each patient should appear only once (first encounter kept)."""
    assert len(diabetes_df) < 80_000


def test_diabetes_age_midpoints(diabetes_df):
    """Age should be mapped to integer midpoints (5, 15, …, 95)."""
    valid = set(range(5, 100, 10))
    assert set(diabetes_df["age"].unique()).issubset(valid)


def test_diabetes_no_none_string(diabetes_df):
    """'None' values should be replaced with 'NotMeasured' to survive CSV round-trips."""
    for col in ("max_glu_serum", "A1Cresult"):
        assert "None" not in diabetes_df[col].values


def test_diabetes_gender_clean(diabetes_df):
    """Unknown/Invalid gender rows should be removed."""
    assert "Unknown/Invalid" not in diabetes_df["gender"].values


def test_diabetes_categorical_dtypes(diabetes_df):
    """Categorical columns must be object dtype for sklearn compatibility."""
    for col in ("race", "gender", "max_glu_serum", "A1Cresult", "change", "diabetesMed"):
        assert diabetes_df[col].dtype == object


def test_diabetes_csv_roundtrip(diabetes_df):
    """Categorical values must survive a CSV write/read without becoming NaN."""
    import io
    buf = io.StringIO()
    diabetes_df.head(50).to_csv(buf, index=False)
    buf.seek(0)
    import pandas as pd
    reloaded = pd.read_csv(buf)
    for col in ("max_glu_serum", "A1Cresult"):
        assert reloaded[col].isnull().sum() == 0
