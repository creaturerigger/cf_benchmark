import pandas as pd


class Deduplicator:
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().reset_index(drop=True)
