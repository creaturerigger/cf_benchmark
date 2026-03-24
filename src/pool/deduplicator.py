import pandas as pd


class Deduplicator:
    def __call__(self, df: pd.DataFrame, ignore_cols=("query_id",)) -> pd.DataFrame:
        feature_cols = [c for c in df.columns if c not in ignore_cols]
        return df.drop_duplicates(subset=feature_cols).reset_index(drop=True)