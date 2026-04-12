"""Pipeline tasks — each function is one stage of the experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from prefect import task
from torch.utils.data import DataLoader

from src.data.data_module import load_dataset
from src.data.preprocessing.py_dataset import PYTDataset
from src.data.preprocessing.transform import Transformer
from src.models.pytorch_classifier import PYTModel
from src.models.trainer import Trainer
from src.pool.pool_builder import CFPoolBuilder
from src.perturbations.gaussian import GaussianPerturbation
from src.perturbations.uniform import UniformPerturbation
from src.utils.constants import DefaultPaths
from src.robustness.matcher import NearestCFMatcher
from src.evaluation.experiment import RobustnessExperiment
from src.evaluation.aggregator import ResultsAggregator
from src.evaluation.stability_curve import StabilityCurveBuilder
from src.utils.constants import GeometricDistanceType, InterventionDistanceType


# Tasks: Data Preparation

@task(name="load_and_prepare_data")
def load_and_prepare_data(
    dataset_cfg: dict,
    target_column: str,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Load a dataset, build PYTDataset for training, return metadata."""
    df = load_dataset(dataset_cfg)

    pyt_ds_train = PYTDataset(
        dataframe=df,
        target_column=target_column,
        test_size=test_size,
        train=True,
    )
    pyt_ds_test = PYTDataset(
        dataframe=df,
        target_column=target_column,
        scaler=pyt_ds_train.scaler,
        encoder=pyt_ds_train.encoder,
        target_encoder=pyt_ds_train.target_encoder,
        test_size=test_size,
        train=False,
    )

    numerical_cols = df.drop(columns=[target_column]).select_dtypes(
        include=["number"],
    ).columns.tolist()
    categorical_cols = df.drop(columns=[target_column]).columns.difference(
        numerical_cols,
    ).tolist()

    return {
        "dataframe": df,
        "train_dataset": pyt_ds_train,
        "test_dataset": pyt_ds_test,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "target_column": target_column,
    }


# Tasks: Model Training


@task(name="train_model")
def train_model(
    train_dataset: PYTDataset,
    test_dataset: PYTDataset,
    model_cfg: dict,
    training_cfg: dict,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Instantiate, train, and return a PYTModel with its trainer."""
    in_features = train_dataset.train_features_tensor.shape[1] - 1
    model = PYTModel(in_features=in_features, cfg=model_cfg)

    trainer = Trainer(model)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trainer.train(
        epochs=training_cfg.get("epochs", 50),
        learning_rate=training_cfg.get("learning_rate", 1e-3),
        optimizer=training_cfg.get("optimizer", "adam"),
        train_dataloader=train_loader,
        test_dataloader=test_loader,
    )

    return {"model": model, "trainer": trainer}


# Tasks: Pool Building


@task(name="build_cf_pool")
def build_cf_pool(
    cf_method,
    query_instances: pd.DataFrame,
    pool_cfg: dict,
    ds_name: str,
    pool_path: Path | None = None,
) -> dict[str, Any]:
    """Build the original CF pool for each query instance.

    Args:
        cf_method: an instantiated CF generation method.
        query_instances: DataFrame of query rows to explain.
        pool_cfg: dict with keys ``runs`` and ``per_run``.
        ds_name: dataset name for file paths.
        pool_path: override base pool directory.

    Returns:
        Dict with ``query_ids`` (list[str]) and ``pool_stats`` (list[dict]).
    """
    builder = CFPoolBuilder(
        cf_method=cf_method,
        runs=pool_cfg.get("runs", 15),
        per_run=pool_cfg.get("per_run", 5),
        ds_name=ds_name,
        perturbed=False,
        pool_path=pool_path,
    )

    query_ids = []
    pool_stats = []
    for idx in range(len(query_instances)):
        row = query_instances.iloc[[idx]]
        qid, gen_df = builder.build(row)
        query_ids.append(qid)
        stats = gen_df.attrs.get("pool_stats", {})
        pool_stats.append({"query_id": qid, **stats})

    return {"query_ids": query_ids, "pool_stats": pool_stats}


@task(name="build_perturbed_pools")
def build_perturbed_pools(
    cf_method,
    query_instances: pd.DataFrame,
    query_ids: list[str],
    perturbation_cfg: dict,
    pool_cfg: dict,
    ds_name: str,
    transformer: Transformer,
    pool_path: Path | None = None,
) -> dict[str, dict]:
    """Generate perturbed queries and their CF pools.

    For each query, generates M perturbed versions and builds
    CF pools for each. Returns a nested dict keyed by query_uuid.

    Args:
        cf_method: an instantiated CF generation method.
        query_instances: DataFrame of original query rows.
        query_ids: matching list of query UUIDs from original pool.
        perturbation_cfg: dict with ``type``, ``sigma``/``epsilon``, ``M``.
        pool_cfg: dict with ``runs`` and ``per_run``.
        ds_name: dataset name.
        transformer: fitted Transformer to convert between DataFrame/tensor.

    Returns:
        Dict ``{query_uuid: {"perturbed_queries": [...], "perturbed_pools": [...]}}``.
    """
    # Create perturbation strategy
    p_type = perturbation_cfg.get("type", "gaussian")
    m = perturbation_cfg.get("M", 20)
    cont_indices = transformer.encoded_continuous_feature_indices

    if p_type == "gaussian":
        sigma = perturbation_cfg.get("sigma", 0.05)
        perturbation = GaussianPerturbation(
            sigma=sigma, continuous_indices=cont_indices,
        )
    elif p_type == "uniform":
        epsilon = perturbation_cfg.get("epsilon", 0.05)
        perturbation = UniformPerturbation(
            epsilon=epsilon, continuous_indices=cont_indices,
        )
    else:
        raise ValueError(f"Unsupported perturbation type: {p_type}")

    builder = CFPoolBuilder(
        cf_method=cf_method,
        runs=pool_cfg.get("runs", 15),
        per_run=pool_cfg.get("per_run", 5),
        ds_name=ds_name,
        perturbed=True,
        pool_path=pool_path,
    )

    result: dict[str, dict] = {}

    for idx, qid in enumerate(query_ids):
        row = query_instances.iloc[[idx]]
        x_tensor = transformer.transform(row)

        perturbed_queries = []
        perturbed_pools = []

        for _ in range(m):
            x_prime_tensor = perturbation(x_tensor)
            x_prime_df = transformer.inverse_transform(x_prime_tensor)

            _pid, iter_cfs_df = builder.build(x_prime_df, query_id=qid)

            perturbed_queries.append(x_prime_tensor)

            # Use the CFs returned directly from build() instead of
            # reading from CSV, which would accumulate across iterations.
            if len(iter_cfs_df) > 0:
                feature_cols = [
                    c for c in iter_cfs_df.columns if c != "query_id"
                ]
                pool_tensor = transformer.transform(iter_cfs_df[feature_cols])
                perturbed_pools.append(pool_tensor)
            else:
                perturbed_pools.append(torch.empty(0, x_tensor.shape[-1]))

        result[qid] = {
            "perturbed_queries": perturbed_queries,
            "perturbed_pools": perturbed_pools,
        }

    return result


# Tasks: Evaluation


@task(name="run_robustness_evaluation")
def run_robustness_evaluation(
    query_ids: list[str],
    query_instances: pd.DataFrame,
    original_pools: dict[str, torch.Tensor],
    perturbed_data: dict[str, dict],
    transformer: Transformer,
    sigma: float,
    robustness_cfg: dict,
) -> list[dict]:
    """Run the robustness experiment and return flat records.

    Args:
        query_ids: list of query UUIDs.
        query_instances: original query DataFrame rows.
        original_pools: ``{query_uuid: (N, D) tensor}`` of original CFs.
        perturbed_data: output of ``build_perturbed_pools``.
        transformer: fitted Transformer.
        sigma: perturbation magnitude.
        robustness_cfg: dict with metric preferences.

    Returns:
        List of flat dicts for aggregation.
    """
    geo_metric = GeometricDistanceType(
        robustness_cfg.get("geometric_metric", "l1"),
    )
    int_metric = InterventionDistanceType(
        robustness_cfg.get("intervention_metric", "jaccard_index"),
    )

    experiment = RobustnessExperiment(
        matcher=NearestCFMatcher(metric=geo_metric),
        geometric_metric=geo_metric,
        intervention_metric=int_metric,
        encoded_cont_feature_indices=transformer.encoded_continuous_feature_indices,
        encoded_cat_feature_indices=transformer.encoded_categorical_feature_indices,
    )

    queries: dict[str, dict] = {}
    for idx, qid in enumerate(query_ids):
        row = query_instances.iloc[[idx]]
        x_tensor = transformer.transform(row)
        queries[qid] = {
            "x": x_tensor.squeeze(0),
            "pool": original_pools[qid],
            **perturbed_data[qid],
        }

    results = experiment.run(queries, sigma=sigma)
    return experiment.to_records(results)


# Tasks: Aggregation


@task(name="aggregate_results")
def aggregate_results(
    records: list[dict],
    dataset_name: str,
) -> dict[str, pd.DataFrame]:
    """Aggregate raw records into multiple summary tables."""
    agg = ResultsAggregator()
    return {
        "candidate_level": agg.aggregate_candidate_level(records),
        "query_level": agg.aggregate_query_level(records),
        "pareto_only": agg.aggregate_pareto_only(records),
        "by_sigma": agg.aggregate_by_sigma(records),
        "by_dataset": agg.aggregate_by_dataset({dataset_name: records}),
    }


@task(name="build_stability_curves")
def build_stability_curves(
    sigma_agg: pd.DataFrame,
    dataset_name: str,
) -> dict:
    """Build stability curves and compute AUC."""
    builder = StabilityCurveBuilder()
    curve = builder.build(sigma_agg)
    auc = builder.compute_auc(curve)

    return {
        "geometric_auc": auc["geometric_auc"],
        "intervention_auc": auc["intervention_auc"],
        "curve": curve,
    }
