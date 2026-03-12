import yaml
import pytest
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
]


@pytest.fixture(params=DATASET_SPECS, ids=["adult", "compas",
                                           "german", "lending"])
def dataset(request):
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
    df, expected_columns, _ = dataset
    assert set(df.columns) == expected_columns


def test_target_binary(dataset):
    df, _, target_col = dataset
    assert set(df[target_col].unique()).issubset({0, 1})
