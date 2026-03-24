import uuid
from pathlib import Path
import pandas as pd
from src.utils.constants import DefaultPaths
from src.pool.deduplicator import Deduplicator


class CFPoolBuilder:

    def __init__(self, cf_method, runs: int = 200, per_run: int = 5,
                 ds_name: str=None, save_interval: int = 5,
                 perturbed: bool=False):
        self.runs = runs
        self.per_run = per_run
        self.cf_method = cf_method
        self.ds_name = ds_name
        self.perturbed = perturbed
        self.save_interval = save_interval
        self.ds_pool_path = DefaultPaths.poolPath / self.ds_name
        self.ds_pool_path.mkdir(parents=True, exist_ok=True)

        suffix = "perturbed" if perturbed else "original"
        self.cfs_filepath = self.ds_pool_path / f"{self.ds_name}_{suffix}_cfs.csv"
        self.queries_filepath = self.ds_pool_path / f"{self.ds_name}_{suffix}_queries.csv"

    def _append_to_csv(self, df: pd.DataFrame, filepath: Path) -> None:
        """Append df to a CSV, writing the header only when the file is new."""
        write_header = not filepath.is_file()
        df.to_csv(filepath, mode='a', header=write_header, index=False)

    def _save_deduplicated(self, new_batch: pd.DataFrame, filepath: Path) -> None:
        """Merge new_batch with the existing CSV, deduplicate, and overwrite."""
        if filepath.is_file():
            existing = pd.read_csv(filepath)
            combined = pd.concat([existing, new_batch], axis=0, ignore_index=True)
        else:
            combined = new_batch
        deduped = Deduplicator()(combined)
        deduped.to_csv(filepath, index=False)

    def build(self, x: pd.DataFrame, query_id: str=None) -> str:
        """Generate a pool of CFs for a single query instance.

        Args:
            x: single-row DataFrame of the query instance.
            query_id: UUID linking this query to an existing pool entry.
                      Pass the value returned by the original build when
                      building a perturbed pool. If None, a new UUID is
                      generated.

        Returns:
            The query_id assigned to this query.
        """
        if query_id is None:
            query_id = str(uuid.uuid4())

        query_row = x.copy()
        query_row["query_id"] = query_id
        self._append_to_csv(query_row, self.queries_filepath)

        batch = []
        for i in range(self.runs):
            result = self.cf_method.generate(x, self.per_run)
            cfs_df = result.to_dataframe()
            if cfs_df is not None and len(cfs_df) > 0:
                cfs_df = cfs_df.copy()
                cfs_df["query_id"] = query_id
                batch.append(cfs_df)

            if batch and (i + 1) % self.save_interval == 0:
                self._save_deduplicated(pd.concat(batch, axis=0, ignore_index=True),
                                        self.cfs_filepath)
                batch = []

        if batch:
            self._save_deduplicated(pd.concat(batch, axis=0, ignore_index=True),
                                    self.cfs_filepath)

        return query_id
