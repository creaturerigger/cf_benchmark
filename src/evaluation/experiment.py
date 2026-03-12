import yaml
from src.data.data_module import load_dataset
from pathlib import Path
import pandas as pd

configs = [yaml.safe_load(p.read_text()) for p in Path('configs/dataset').glob('*.yaml')]
datasets = {cfg['dataset']['name']: load_dataset(cfg) for cfg in configs}

df_names = ['adult-income', 'lending-club',
            'german-credit', 'compas-recidivism']


def run_experiment(name: str, df: pd.DataFrame):
    pass


for name, df in datasets.items():
    run_experiment(name, df)
