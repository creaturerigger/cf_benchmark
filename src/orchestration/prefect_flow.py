"""End-to-end robustness experiment pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from prefect import flow

from src.data.preprocessing.transform import CFMethodSpec, Transformer
from src.cf_methods.registry import create_method
from src.utils.config_loader import load_config
from src.utils.constants import DefaultPaths
from src.utils.seed import set_seed
from src.utils.logger import ExperimentLogger
from src.evaluation.plotting import generate_all_figures, save_tables, save_raw_records, save_pareto_cfs
from src.evaluation.selectors import apply_all_selectors
from src.orchestration.tasks import (
    load_and_prepare_data,
    train_model,
    build_cf_pool,
    build_perturbed_pools,
    run_robustness_evaluation,
    aggregate_results,
    build_stability_curves,
)


logger = logging.getLogger(__name__)


class StageTimer:
    """Lightweight wall-clock timer for pipeline stages."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._start: float | None = None
        self._stage: str | None = None

    def start(self, stage: str) -> None:
        self._stage = stage
        self._start = time.perf_counter()

    def stop(self) -> float:
        elapsed = time.perf_counter() - self._start
        self._records.append({"stage": self._stage, "seconds": round(elapsed, 3)})
        self._start = None
        self._stage = None
        return elapsed

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)


def _try_reuse_original_pool(
    query_instances: pd.DataFrame,
    pool_dir: Path,
    dataset_name: str,
    min_pool_size: int,
) -> tuple[list[str], list[dict]] | None:
    """Check if existing original pools on disk can be reused.

    Matches current query instances to stored queries by feature values.
    If every query has a match with at least ``min_pool_size`` deduplicated
    CFs, returns ``(query_ids, pool_stats)``; otherwise returns ``None``.
    """
    if min_pool_size <= 0:
        return None

    cfs_path = pool_dir / f"{dataset_name}_original_cfs.csv"
    queries_path = pool_dir / f"{dataset_name}_original_queries.csv"

    if not cfs_path.is_file() or not queries_path.is_file():
        return None

    stored_queries = pd.read_csv(queries_path)
    stored_cfs = pd.read_csv(cfs_path)

    feature_cols = [c for c in stored_queries.columns if c != "query_id"]

    # Pre-compute per-query CF counts
    cf_counts = stored_cfs.groupby("query_id").size()

    query_ids: list[str] = []
    pool_stats: list[dict] = []

    for idx in range(len(query_instances)):
        row = query_instances.iloc[idx]

        # Find a stored query that matches by feature values
        match = stored_queries.copy()
        for col in feature_cols:
            if col in row.index:
                match = match[match[col] == row[col]]

        if match.empty:
            return None

        # Among matches, pick the one with the most CFs
        best_qid = None
        best_count = 0
        for candidate_qid in match["query_id"].unique():
            count = cf_counts.get(candidate_qid, 0)
            if count > best_count:
                best_count = count
                best_qid = candidate_qid

        if best_qid is None or best_count < min_pool_size:
            return None

        query_ids.append(best_qid)
        pool_stats.append({
            "query_id": best_qid,
            "generated": int(best_count),
            "duplicates": 0,
            "after_dedup": int(best_count),
        })

    return query_ids, pool_stats


@flow(name="robustness_pipeline")
def run_pipeline(
    dataset_name: str,
    model_name: str = "pytorch_classifier",
    cf_method_name: str = "dice",
    experiment_name: str = "robustness_experiment",
    seed: int = 42,
    n_queries: int = 50,
    sigmas: list[float] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the full robustness experiment for one dataset.

    Args:
        dataset_name: config name for the dataset (e.g. "adult").
        model_name: config name for the model.
        cf_method_name: config name for the CF method.
        experiment_name: config name for the experiment.
        seed: random seed for reproducibility.
        n_queries: number of test instances to evaluate.
        sigmas: perturbation magnitudes to sweep.  If ``None``,
                uses the single sigma from the experiment config.
        overrides: optional dict to override config values
                   (e.g. ``{"pool": {"runs": 3}, "perturbation": {"M": 3}}``).

    Returns:
        Dict with keys: ``records``, ``tables``, ``stability``, ``timing``.
    """
    timer = StageTimer()

    # ── 0. Setup ─────────────────────────────────────────────
    set_seed(seed)
    cfg = load_config(
        dataset=dataset_name,
        model=model_name,
        cf_method=cf_method_name,
        experiment=experiment_name,
    )
    if overrides:
        for key, val in overrides.items():
            if isinstance(val, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(val)
            else:
                cfg[key] = val

    # Method-scoped paths (models stay universal)
    paths = DefaultPaths.for_method(cf_method_name)

    exp_logger = ExperimentLogger(
        f"{dataset_name}_{cf_method_name}_{seed}",
        logs_dir=paths.logsPath,
    )
    exp_logger.log_config(cfg)
    exp_logger.log_seed(seed)

    target_column = cfg["specs"]["target"]
    perturbation_cfg = cfg.get("perturbation", {"type": "gaussian", "sigma": 0.05, "M": 20})
    pool_cfg = cfg.get("pool", {"runs": 15, "per_run": 5})
    robustness_cfg = cfg.get("robustness", {})

    if sigmas is None:
        sigmas = [perturbation_cfg.get("sigma", 0.05)]

    # ── 1. Data loading ──────────────────────────────────────
    timer.start("data_loading")
    logger.info("Loading dataset: %s", dataset_name)
    data = load_and_prepare_data(
        dataset_cfg=cfg,
        target_column=target_column,
    )
    exp_logger.log_dataset(
        name=dataset_name,
        n_rows=len(data["dataframe"]),
        n_features=len(data["numerical_cols"]) + len(data["categorical_cols"]),
        target=target_column,
        class_distribution=data["dataframe"][target_column].value_counts().to_dict(),
    )
    elapsed = timer.stop()
    logger.info("Data loading took %.2fs", elapsed)

    # ── 2. Model training (or loading) ─────────────────────────
    timer.start("model_training")
    model_dir = paths.modelsPath / dataset_name
    model_path = model_dir / "model.pt"

    if model_path.is_file():
        logger.info("Loading existing model from %s", model_path)
        from src.models.pytorch_classifier import PYTModel
        model = PYTModel.load(model_path, cfg)
        final_acc = None  # accuracy is in the checkpoint history if needed
        exp_logger.log_model(
            model_name=model_name,
            params=cfg.get("model", {}),
            accuracy=final_acc,
        )
    else:
        logger.info("Training model")
        training_cfg = cfg.get("training", {})
        model_result = train_model(
            train_dataset=data["train_dataset"],
            test_dataset=data["test_dataset"],
            model_cfg=cfg,
            training_cfg=training_cfg,
            batch_size=training_cfg.get("batch_size", 64),
        )
        model = model_result["model"]
        trainer = model_result["trainer"]

        # Persist trained model to results/models/
        trainer.save_model(model_dir)
        logger.info("Model saved to %s", model_dir)

        final_acc = trainer.history["test_acc"][-1] if trainer.history["test_acc"] else None
        exp_logger.log_model(
            model_name=model_name,
            params=cfg.get("model", {}),
            accuracy=final_acc,
        )
    elapsed = timer.stop()
    logger.info("Model training took %.2fs", elapsed)

    # ── 3. Data transformation ───────────────────────────────
    spec = CFMethodSpec(
        name=cf_method_name,
        continuous_features=data["numerical_cols"],
        categorical_features=data["categorical_cols"],
        target_column=target_column,
    )
    transformer = Transformer(spec)
    train_df = data["train_dataset"].train_dataset_df
    transformer.fit(train_df)

    # ── 4. CF method instantiation ───────────────────────────
    logger.info("Instantiating CF method: %s", cf_method_name)
    train_df_for_method = data["train_dataset"].train_dataset_df
    max_samples = cfg.get("method", {}).get("data_max_samples")
    if max_samples and len(train_df_for_method) > max_samples:
        logger.info(
            "Subsampling training data for CF method: %d → %d rows",
            len(train_df_for_method), max_samples,
        )
        train_df_for_method = train_df_for_method.sample(
            n=max_samples, random_state=seed,
        )
    cf_method = create_method(
        cfg,
        model,
        train_df_for_method,
        target_column,
        data["numerical_cols"],
    )

    # Select query instances
    test_df = data["test_dataset"].test_dataset_df
    query_instances = test_df.drop(columns=[target_column]).head(n_queries)
    logger.info("Evaluating %d query instances", len(query_instances))

    min_pool_size = pool_cfg.get("min_pool_size", 0)

    # ── 5a. Original pool building (shared across sigmas) ────
    timer.start("pool_building")
    pool_dir = paths.poolPath / dataset_name
    n_encoded = (
        len(transformer.encoded_continuous_feature_indices)
        + sum(len(g) for g in transformer.encoded_categorical_feature_indices)
    )

    reuse = _try_reuse_original_pool(
        query_instances=query_instances,
        pool_dir=pool_dir,
        dataset_name=dataset_name,
        min_pool_size=min_pool_size,
    )

    all_pool_stats: list[dict] = [].copy()

    if reuse is not None:
        query_ids, reuse_stats = reuse
        all_pool_stats.extend(reuse_stats)
        logger.info(
            "Reusing existing original pools (%d queries, min %d CFs each)",
            len(query_ids),
            min(s["after_dedup"] for s in reuse_stats),
        )
    else:
        pool_result = build_cf_pool(
            cf_method=cf_method,
            query_instances=query_instances,
            pool_cfg=pool_cfg,
            ds_name=dataset_name,
            pool_path=paths.poolPath,
        )
        query_ids = pool_result["query_ids"]
        all_pool_stats.extend(pool_result["pool_stats"])
        for qid in query_ids:
            exp_logger.log_pool(
                qid,
                pool_size=pool_cfg.get("runs", 15) * pool_cfg.get("per_run", 5),
                is_perturbed=False,
            )

    # Load original pools as tensors (once, reused across sigmas)
    original_pools: dict[str, torch.Tensor] = {}
    cfs_path = pool_dir / f"{dataset_name}_original_cfs.csv"
    if cfs_path.is_file():
        cfs_df = pd.read_csv(cfs_path)
        for qid in query_ids:
            qid_cfs = cfs_df[cfs_df["query_id"] == qid]
            feature_cols = [c for c in qid_cfs.columns if c != "query_id"]
            if len(qid_cfs) > 0:
                original_pools[qid] = transformer.transform(
                    qid_cfs[feature_cols],
                )
            else:
                original_pools[qid] = torch.empty(0, n_encoded)
    for qid in query_ids:
        if qid not in original_pools:
            original_pools[qid] = torch.empty(0, n_encoded)

    elapsed = timer.stop()
    logger.info("Original pool building took %.2fs", elapsed)

    # ── 5b. Per-sigma loop (perturbed pools + evaluation) ────
    all_records: list[dict] = []

    for sigma in sigmas:
        logger.info("Sigma = %.4f", sigma)
        current_perturbation_cfg = {**perturbation_cfg, "sigma": sigma}

        # Tag pool stats with this sigma
        for s in all_pool_stats:
            if "sigma" not in s:
                s["sigma"] = sigma

        # Build perturbed pools
        timer.start(f"perturbed_pools_sigma{sigma}")
        perturbed_data = build_perturbed_pools(
            cf_method=cf_method,
            query_instances=query_instances,
            query_ids=query_ids,
            perturbation_cfg=current_perturbation_cfg,
            pool_cfg=pool_cfg,
            ds_name=dataset_name,
            transformer=transformer,
            pool_path=paths.poolPath,
        )
        elapsed = timer.stop()
        logger.info("Perturbed pool building (sigma=%.4f) took %.2fs", sigma, elapsed)

        # Run evaluation
        timer.start(f"evaluation_sigma{sigma}")
        records = run_robustness_evaluation(
            query_ids=query_ids,
            query_instances=query_instances,
            original_pools=original_pools,
            perturbed_data=perturbed_data,
            transformer=transformer,
            sigma=sigma,
            robustness_cfg=robustness_cfg,
        )
        all_records.extend(records)
        elapsed = timer.stop()
        logger.info("Evaluation (sigma=%.4f) took %.2fs", sigma, elapsed)

        # Log per query summaries and enrich pool stats with Pareto counts
        for qid in query_ids:
            qid_records = [r for r in records if r["query_uuid"] == qid]
            if qid_records:
                n_cand = len(qid_records)
                n_pareto = sum(1 for r in qid_records if r.get("is_pareto_optimal"))
                mean_prox = sum(r["proximity"] for r in qid_records) / n_cand
                mean_geo = sum(r["geometric_instability"] for r in qid_records) / n_cand
                mean_int = sum(r["intervention_instability"] for r in qid_records) / n_cand
                exp_logger.log_query_result(
                    query_uuid=qid, sigma=sigma,
                    n_candidates=n_cand, n_pareto=n_pareto,
                    mean_proximity=mean_prox,
                    mean_geo_instability=mean_geo,
                    mean_int_instability=mean_int,
                )
                # Attach Pareto count to the matching pool stats entry
                for ps in all_pool_stats:
                    if ps["query_id"] == qid and ps["sigma"] == sigma:
                        ps["pareto"] = n_pareto
                        break

    # ── 6. Selection strategy comparison ─────────────────────
    timer.start("selection_analysis")
    logger.info("Running selection strategy comparison")
    selection_records = apply_all_selectors(all_records, seed=seed)
    selection_df = pd.DataFrame(selection_records)
    elapsed = timer.stop()
    logger.info("Selection analysis took %.2fs", elapsed)

    # ── 7. Aggregation ───────────────────────────────────────
    timer.start("aggregation")
    logger.info("Aggregating results")
    tables = aggregate_results(all_records, dataset_name)

    stability = {}
    sigma_agg = tables.get("by_sigma")
    if sigma_agg is not None and len(sigma_agg) > 1:
        stability = build_stability_curves(sigma_agg, dataset_name)
        exp_logger.log_stability_auc(
            dataset=dataset_name,
            geometric_auc=stability["geometric_auc"],
            intervention_auc=stability["intervention_auc"],
        )
    elapsed = timer.stop()
    logger.info("Aggregation took %.2fs", elapsed)

    # ── 8. Save artefacts ────────────────────────────────────
    timer.start("save_artefacts")
    logger.info("Saving tables and generating figures")
    save_raw_records(all_records, dataset_name, out_dir=paths.rawPath)
    save_pareto_cfs(all_records, dataset_name, pool_path=paths.poolPath, out_dir=paths.rawPath)
    save_tables(tables, dataset_name, out_dir=paths.tablesPath)

    # Save selection strategy comparison
    if not selection_df.empty:
        sel_path = paths.tablesPath / f"{dataset_name}_selection_comparison.csv"
        sel_path.parent.mkdir(parents=True, exist_ok=True)
        selection_df.to_csv(sel_path, index=False)
        logger.info("Selection comparison saved to %s", sel_path)

    generate_all_figures(
        records=all_records,
        tables=tables,
        stability=stability,
        dataset_name=dataset_name,
        out_dir=paths.figuresPath,
    )

    # Save pool statistics (per-query + aggregated totals)
    pool_stats_df = pd.DataFrame(all_pool_stats)
    totals = {
        "query_id": "TOTAL",
        "generated": pool_stats_df["generated"].sum(),
        "duplicates": pool_stats_df["duplicates"].sum(),
        "after_dedup": pool_stats_df["after_dedup"].sum(),
        "sigma": "",
        "pareto": pool_stats_df["pareto"].sum() if "pareto" in pool_stats_df.columns else 0,
    }
    pool_stats_with_totals = pd.concat(
        [pool_stats_df, pd.DataFrame([totals])], ignore_index=True,
    )
    pool_stats_path = paths.tablesPath / f"{dataset_name}_pool_stats.csv"
    pool_stats_path.parent.mkdir(parents=True, exist_ok=True)
    pool_stats_with_totals.to_csv(pool_stats_path, index=False)
    logger.info(
        "Pool stats:\n%s",
        pool_stats_with_totals.to_string(index=False),
    )

    # Save timing summary
    timing_df = timer.summary_df()
    timing_path = paths.tablesPath
    timing_path.mkdir(parents=True, exist_ok=True)
    timing_df.to_csv(timing_path / f"{dataset_name}_timing.csv", index=False)
    logger.info("Stage timing:\n%s", timing_df.to_string(index=False))

    elapsed = timer.stop()
    logger.info("Saving artefacts took %.2fs", elapsed)
    logger.info("Pipeline complete for %s", dataset_name)

    return {
        "records": all_records,
        "tables": tables,
        "stability": stability,
        "timing": timer.records,
        "pool_stats": all_pool_stats,
        "selection": selection_records,
    }
