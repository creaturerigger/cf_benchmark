import pandas as pd
import numpy as np
import zipfile
from urllib.request import urlretrieve
from pathlib import Path
from ..registry import register_dataset
from sklearn.model_selection import train_test_split


def _preprocess_german_data(data: np.ndarray) -> pd.DataFrame:
    data_split = [row.split() for row in data]
    df = pd.DataFrame(data_split)
    column_names = ['status_of_existing_checking_account',
                    'duration_in_month',
                    'credit_history',
                    'purpose',
                    'credit_amount',
                    'savings_account_bonds',
                    'present_employment_since',
                    'installment_rate_in_percentage_of_disposable_income',
                    'personal_status_and_sex',
                    'other_debtors_guarantors',
                    'present_residence_since',
                    'property',
                    'age_in_years',
                    'other_installment_plans',
                    'housing',
                    'number_of_existing_credits_at_this_bank',
                    'job',
                    'number_of_people_being_liable_to_provide_maintenance_for',
                    'telephone',
                    'foreign_worker',
                    'credit_risk'
    ]
    df.columns = column_names
    categorical_value_mapping = {
       'status_of_existing_checking_account': {
            'A11': '... < 0 DM',
            'A12': '0 <= ... < 200 DM',
            'A13': '... >= 200 DM / salary assignments for at least 1 year',
            'A14': 'no checking account'
        },
        'credit_history': {
            'A30': 'no credits taken/ all credits paid back duly',
            'A31': 'all credits at this bank paid back duly',
            'A32': 'existing credits paid back duly till now',
            'A33': 'delay in paying off in the past',
            'A34': 'critical account/ other credits existing (not at this bank)'
        },
        'purpose': {
            'A40': 'car (new)',
            'A41': 'car (used)',
            'A42': 'furniture/equipment',
            'A43': 'radio/television',
            'A44': 'domestic appliances',
            'A45': 'repairs',
            'A46': 'education',
            'A47': '(vacation - does not exist?)',
            'A48': 'retraining',
            'A49': 'business',
            'A410': 'others'
        },
        'savings _account_bonds': {
            'A61': '... < 100 DM',
            'A62': '100 <= ... < 500 DM',
            'A63': '500 <= ... < 1000 DM',
            'A64': '.. >= 1000 DM',
            'A65': 'unknown/ no savings account'
        },
        'present_employment_since': {
            'A71': 'unemployed',
            'A72': '... < 1 year',
            'A73': '1 <= ... < 4 years',
            'A74': '4 <= ... < 7 years',
            'A75': '.. >= 7 years'
        },
        'personal_status_and_sex': {
            'A91': 'male : divorced/separated',
            'A92': 'female : divorced/separated/married',
            'A93': 'male : single',
            'A94': 'male : married/widowed',
            'A95': 'female : single'
        },
        'other_debtors_guarantors': {
            'A101': 'none',
            'A102': 'co-applicant',
            'A103': 'guarantor'
        },
        'property': {
            'A121': 'real estate',
            'A122': 'if not A121 : building society savings agreement/ life insurance',
            'A123': 'if not A121/A122 : car or other, not in attribute 6',
            'A124': 'unknown / no property'
        },
        'other_installment_plans': {
            'A141': 'bank',
            'A142': 'stores',
            'A143': 'none'
        },
        'housing': {
            'A151': 'rent',
            'A152': 'own',
            'A153': 'for free'
        },
        'job': {
            'A171': 'unemployed/ unskilled - non-resident',
            'A172': 'unskilled - resident',
            'A173': 'skilled employee / official',
            'A174': 'management/ self-employed/ highly qualified employee/ officer'
        },
        'telephone': {
            'A191': 'none',
            'A192': 'yes, registered under the customers name'
        },
        'foreign_worker': {
            'A201': 'yes',
            'A202': 'no'
        },
        'credit_risk': {
            '1': 0,
           '2': 1
        } 
    }
    for col, mapping in categorical_value_mapping.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    # duration_in_month
    # credit_amount
    # installment_rate_in_percentage_of_disposable_income
    # present_residence_since
    # age_in_years
    # number_of_existing_credits_at_this_bank
    # number_of_people_being_liable_to_provide_maintenance_for
    number_cols = ['duration_in_month', 'credit_amount', 'installment_rate_in_percentage_of_disposable_income',
                   'present_residence_since', 'age_in_years', 'number_of_existing_credits_at_this_bank',
                   'number_of_people_being_liable_to_provide_maintenance_for']
    df[number_cols] = df[number_cols].astype(int)
    return df


@register_dataset(name='german-credit')
def load_german(dataset_cfg: dict) -> pd.DataFrame:
    train_size = dataset_cfg['specs']['train_size'] / 100.0
    test_size = 1.0 - train_size
    train_only = dataset_cfg['specs'].get('train_only', False)
    dataset_url = dataset_cfg['dataset']['url']
    root_download_dir = Path(dataset_cfg["download_location"])
    dataset_dir_name = dataset_cfg["dataset"]["name"]
    dataset_dir = root_download_dir / dataset_dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    zipfilename = dataset_dir_name.split("-")[0] + '.zip'
    zipfilepath = dataset_dir / zipfilename
    if not (dataset_dir / 'adult.data').is_file():
        urlretrieve(dataset_url, zipfilepath)
        with zipfile.ZipFile(zipfilepath, 'r') as unzip:
            unzip.extractall(dataset_dir)
        zipfilepath.unlink(missing_ok=True)

    raw_data = np.genfromtxt(str(dataset_dir / 'german.data'), delimiter=", ",
                             dtype=str, invalid_raise=False)
    df = _preprocess_german_data(raw_data)

    if train_only:
        train, _ = train_test_split(df, test_size=test_size, random_state=42)
        df = train.reset_index(drop=True)
    return df
