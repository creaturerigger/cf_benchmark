from __future__ import annotations

import pandas as pd


class ResultsAggregator:
    """Transforms raw experiment records into summary tables."""

    METRIC_COLS = [
        "proximity",
        "geometric_instability",
        "intervention_instability",
    ]

    def to_dataframe(self, records: list[dict]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(
                columns=["query_uuid", "cf_index", "sigma"] + self.METRIC_COLS + ["is_pareto_optimal"],
            )
        return pd.DataFrame(records)

    def aggregate_candidate_level(
        self, records: list[dict],
    ) -> pd.DataFrame:
        """Per-candidate stats grouped by (query_uuid, sigma)."""
        df = self.to_dataframe(records)
        if df.empty:
            return pd.DataFrame()
        return df.groupby(["query_uuid", "sigma"])[self.METRIC_COLS].describe()

    def aggregate_query_level(
        self, records: list[dict],
    ) -> pd.DataFrame:
        """Mean metrics per query_uuid across all candidates."""
        df = self.to_dataframe(records)
        if df.empty:
            return pd.DataFrame(columns=["query_uuid", "sigma"] + self.METRIC_COLS)
        return (
            df.groupby(["query_uuid", "sigma"])[self.METRIC_COLS]
            .mean()
            .reset_index()
        )

    def aggregate_pareto_only(
        self, records: list[dict],
    ) -> pd.DataFrame:
        """Mean metrics computed only over Pareto-optimal candidates."""
        df = self.to_dataframe(records)
        if df.empty:
            return pd.DataFrame(columns=["query_uuid", "sigma"] + self.METRIC_COLS)
        pareto = df[df["is_pareto_optimal"] == True]  # noqa: E712
        if pareto.empty:
            return pd.DataFrame(columns=["query_uuid", "sigma"] + self.METRIC_COLS)
        return (
            pareto.groupby(["query_uuid", "sigma"])[self.METRIC_COLS]
            .mean()
            .reset_index()
        )

    def aggregate_by_sigma(
        self, records: list[dict],
    ) -> pd.DataFrame:
        """Mean metrics across all queries per sigma level."""
        df = self.to_dataframe(records)
        if df.empty:
            return pd.DataFrame(columns=["sigma"] + self.METRIC_COLS)
        return (
            df.groupby("sigma")[self.METRIC_COLS]
            .mean()
            .reset_index()
        )

    def aggregate_by_dataset(
        self, all_results: dict[str, list[dict]],
    ) -> pd.DataFrame:
        """Cross-dataset summary: mean metrics per (dataset, sigma).

        Args:
            all_results: mapping of dataset_name -> list of record dicts.
        """
        frames = []
        for ds_name, records in all_results.items():
            df = self.to_dataframe(records)
            df["dataset"] = ds_name
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        if combined.empty:
            return pd.DataFrame(columns=["dataset", "sigma"] + self.METRIC_COLS)
        return (
            combined.groupby(["dataset", "sigma"])[self.METRIC_COLS]
            .mean()
            .reset_index()
        )
