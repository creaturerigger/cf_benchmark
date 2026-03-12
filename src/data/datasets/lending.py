import pandas as pd
import numpy as np
from pathlib import Path
from ..registry import register_dataset
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
import zipfile
import math


@register_dataset(name='lending-club')
def load_lending(dataset_cfg: dict) -> pd.DataFrame:
    """
    As described in the DiCE paper by the authors.
    """
    train_size = dataset_cfg['specs']['train_size'] / 100.0
    test_size = 1.0 - train_size
    train_only = dataset_cfg['specs'].get('train_only', False)
    dataset_url = dataset_cfg['dataset']['url']
    root_download_dir = Path(dataset_cfg["download_location"])
    dataset_dir_name = dataset_cfg["dataset"]["name"]
    dataset_dir = root_download_dir / dataset_dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    zipfilepath = dataset_dir / f'{dataset_dir_name}.zip'

    if not (dataset_dir / 'loan.csv').is_file():
        urlretrieve(dataset_url, zipfilepath)
        with zipfile.ZipFile(zipfilepath, 'r') as unzip:
            unzip.extractall(dataset_dir)
        zipfilepath.unlink(missing_ok=True)

    csv_filepath = dataset_dir / 'loan.csv'
    df = pd.read_csv(csv_filepath, low_memory=False)

    def parse_year(year_as_str):
        year = int(year_as_str)
        if year <= 99:
            return 1900 + year if year > 50 else 2000 + year
        return 2000 + year
    # raw features
    new_df = pd.DataFrame()
    new_df['employment_years'] = df['emp_length']
    new_df['num_open_credit_acc'] = df['open_acc']
    new_df['annual_income'] = df['annual_inc']
    new_df['loan_grade'] = df['grade']

    # credit history
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    lending_months = df['issue_d'].apply(lambda x: x.split('-')[0]).__deepcopy__().copy()
    issue_month = pd.DataFrame(pd.Index(months).get_indexer(lending_months))
    issue_year = pd.DataFrame(df['issue_d'].apply(lambda x: parse_year(x.split('-')[1])).__deepcopy__().copy())
    issue_yearmonth = pd.DataFrame(issue_year.values * 100 + issue_month.values)
    adjusted_last_ym = issue_yearmonth[0].apply(lambda x: (float(x) / 100.0) * 100 + ((float(x) - math.floor(float(x) / 100.0) * 100) - 1) / 12 * 100) / 100
    earliest_credit_line_year = pd.DataFrame(df['earliest_cr_line'].apply(lambda x: parse_year(x.split('-')[1])).__deepcopy__().copy())
    earliest_credit_line_months = df['earliest_cr_line'].apply(lambda x: x.split('-')[0]).__deepcopy__().copy()
    earliest_credit_line_months = pd.DataFrame(pd.Index(months).get_indexer(earliest_credit_line_months))
    adjusted_credit_line_ym = pd.DataFrame(earliest_credit_line_year.values * 100 + \
                            earliest_credit_line_months.apply(
                                lambda x: x - 1
                            ) / 12 * 100) / 100
    adjusted_credit_line_ym = adjusted_credit_line_ym.apply(lambda x: round(x, 2))
    adjusted_last_ym = adjusted_last_ym.apply(lambda x: round(x, 2)).to_frame()
    credit_ym = (adjusted_last_ym - adjusted_credit_line_ym).apply(lambda x: round(x, 1))
    new_df['credit_history'] = credit_ym.values

    # purpose
    conditions = [
        df['purpose'].isin(['credit_card', 'debt_consolidation']),  # Condition for "debt"
        df['purpose'].isin(['car', 'major_purchase', 'vacation', 'wedding', 'medical', 'other']),  # Condition for "purchase"
        df['purpose'].isin(['house', 'home_improvement', 'moving', 'renewable_energy'])  # Another condition for "purchase"
    ]

    outputs = ['debt', 'purchase', 'purchase']

    new_df['purpose'] = np.select(conditions, outputs, default=df['purpose'])

    # home ownership

    new_df['home'] = np.where(
        df['home_ownership'].isin(['ANY', 'NONE']), 
        'OTHER', 
        df['home_ownership']
    )

    # state

    new_df['addr_state'] = df['addr_state']
    new_df.replace('n/a', np.nan, inplace=True)
    new_df['employment_years'] = new_df['employment_years'].fillna('0')
    new_df['employment_years'] = new_df['employment_years'].str.replace(r'[^0-9]', '', regex=True)
    new_df['employment_years'] = new_df['employment_years'].replace('', '0')
    new_df['employment_years'] = new_df['employment_years'].astype(int)

    # target column (loan_status) 0 if never paid or not paid yet 1 if paid

    ls_conditions = [
        df['loan_status'].isin(['Charged Off', 'Current']),
        df['loan_status'] == 'Fully Paid'
    ]
    ls_outputs = [0, 1]

    new_df['loan_status'] = np.select(ls_conditions, ls_outputs, default=df['loan_status'])
    new_df['loan_status'] = new_df['loan_status'].astype(int)

    # Remove rows with negative credit history
    neg_credi_his_idx = new_df.loc[new_df['credit_history'] < 0].index
    new_df.drop(neg_credi_his_idx, inplace=True)
    if train_only:
        train, _ = train_test_split(new_df, test_size=test_size, random_state=42)
        new_df = train.reset_index(drop=True)
    return new_df
