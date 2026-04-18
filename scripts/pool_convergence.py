#!/usr/bin/env python
"""Pool-size convergence experiment.

Builds a large CF pool (N_MAX CFs) for a few fast (dataset, method) combos,
evaluates all three objectives (proximity, geometric instability, intervention
instability) for each candidate, then computes the hypervolume indicator of the
Pareto front at increasing pool-size subsets.

The result is a convergence plot: HV vs pool size.  If HV plateaus (< 5%
relative change), the smaller pool size is sufficient.

Usage:
    python -m scripts.pool_convergence
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymoo.indicators.hv import HV

from src.cf_methods.registry import create_method
from src.data.preprocessing.transform import CFMethodSpec, Transformer
from src.evaluation.experiment import RobustnessExperiment
from src.orchestration.tasks import (
    build_cf_pool,
    build_perturbed_pools,
    load_and_prepare_data,
    train_model,
)
from src.robustness.matcher import NearestCFMatcher
from src.robustness.score import CandidateObjectives, pareto_front
from src.utils.config_loader import load_config
from src.utils.constants import DefaultPaths, GeometricDistanceType, InterventionDistanceType
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
# Combos that produce diverse pools (NICE/GS are ~deterministic → 1 CF after dedup)
COMBOS = [
    ("compas", "dice"),
    ("adult", "dice"),
    ("compas", "moc"),
]

# Pool size sweep — build up to N_MAX, then subsample
N_MAX = 300
POOL_SIZES = [50, 100, 150, 200, 250, 300]

# Fixed experiment params
SEED = 42
N_QUERIES = 3        # a few queries to average over
SIGMA = 0.05         # single sigma for perturbation
M = 3                # perturbations per query

# HV reference point — worst-case (all objectives minimised, so ref > max)
HV_REF = np.array([1.1, 1.1, 1.1])


def _compute_objectives(
    query_instances: pd.DataFrame,
    query_ids: list[str],
    original_pools: dict[str, torch.Tensor],
    perturbed_data: dict[str, dict],
    transformer: Transformer,
    sigma: float,
) -> list[CandidateObjectives]:
    """Run the robustness evaluation and return raw CandidateObjectives."""
    experiment = RobustnessExperiment(
        matcher=NearestCFMatcher(metric=GeometricDistanceType.L_2),
        geometric_metric=GeometricDistanceType.L_2,
        intervention_metric=InterventionDistanceType.JACCARD_INDEX,
        encoded_cont_feature_indices=transformer.encoded_continuous_feature_indices,
        encoded_cat_feature_indices=transformer.encoded_categorical_feature_indices,
    )

    queries = {}
    for idx, qid in enumerate(query_ids):
        row = query_instances.iloc[[idx]]
        x_tensor = transformer.transform(row)
        queries[qid] = {
            "x": x_tensor.squeeze(0),
            "pool": original_pools[qid],
            **perturbed_data[qid],
        }

    results = experiment.run(queries, sigma=sigma)

    # Collect all candidates across queries
    all_candidates = []
    for qr in results:
        all_candidates.extend(qr.candidates)
    return all_candidates


def _hv_at_size(
    candidates: list[CandidateObjectives],
    size: int,
) -> tuple[float, int]:
    """Subsample `size` candidates, compute HV of their Pareto front.

    Returns (hypervolume, |Pareto front|).
    """
    subset = candidates[:size]

    # Normalise objectives to [0, 1] for fair HV comparison
    objs = np.array([[c.proximity, c.geometric_instability, c.intervention_instability]
                      for c in subset])
    if len(objs) == 0:
        return 0.0, 0

    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    span = maxs - mins
    span[span == 0] = 1.0
    normed = (objs - mins) / span

    # Pareto front on normalised objectives
    front = pareto_front([
        CandidateObjectives(
            query_uuid=c.query_uuid, cf_index=i,
            proximity=float(normed[i, 0]),
            geometric_instability=float(normed[i, 1]),
            intervention_instability=float(normed[i, 2]),
        )
        for i, c in enumerate(subset)
    ])

    if not front:
        return 0.0, 0

    front_objs = np.array([[c.proximity, c.geometric_instability, c.intervention_instability]
                            for c in front])

    hv_indicator = HV(ref_point=HV_REF)
    hv_value = float(hv_indicator(front_objs))

    return hv_value, len(front)


def run_combo(ds: str, method: str) -> list[dict]:
    """Run convergence experiment for one (dataset, method) combo."""
    set_seed(SEED)
    cfg = load_config(
        dataset=ds,
        model="pytorch_classifier",
        cf_method=method,
        experiment="robustness_experiment",
    )

    paths = DefaultPaths.for_method(method)
    target_column = cfg["specs"]["target"]

    # Pool config: build N_MAX CFs per query
    # per_run=5 is standard; runs = ceil(N_MAX / per_run)
    per_run = 5
    runs = (N_MAX + per_run - 1) // per_run  # 60 runs × 5 = 300
    pool_cfg = {"runs": runs, "per_run": per_run}
    perturbation_cfg = {"type": "gaussian", "sigma": SIGMA, "M": M}

    # Override in cfg
    cfg["pool"] = pool_cfg
    cfg["perturbation"] = perturbation_cfg

    # 1. Load data
    print(f"  Loading data...", flush=True)
    data = load_and_prepare_data(dataset_cfg=cfg, target_column=target_column)

    # 2. Train / load model
    print(f"  Training model...", flush=True)
    model_dir = paths.modelsPath / ds
    model_path = model_dir / "model.pt"
    if model_path.is_file():
        from src.models.pytorch_classifier import PYTModel
        model = PYTModel.load(model_path, cfg)
    else:
        training_cfg = cfg.get("training", {})
        model_result = train_model(
            train_dataset=data["train_dataset"],
            test_dataset=data["test_dataset"],
            model_cfg=cfg,
            training_cfg=training_cfg,
            batch_size=training_cfg.get("batch_size", 64),
        )
        model = model_result["model"]
        model_result["trainer"].save_model(model_dir)

    # 3. Transformer
    spec = CFMethodSpec(
        name=method,
        continuous_features=data["numerical_cols"],
        categorical_features=data["categorical_cols"],
        target_column=target_column,
    )
    transformer = Transformer(spec)
    train_df = data["train_dataset"].train_dataset_df
    transformer.fit(train_df)

    # 4. CF method
    print(f"  Instantiating CF method...", flush=True)
    train_df_for_method = data["train_dataset"].train_dataset_df
    max_samples = cfg.get("method", {}).get("data_max_samples")
    if max_samples and len(train_df_for_method) > max_samples:
        train_df_for_method = train_df_for_method.sample(n=max_samples, random_state=SEED)
    cf_method = create_method(cfg, model, train_df_for_method, target_column, data["numerical_cols"])

    # 5. Select queries
    test_df = data["test_dataset"].test_dataset_df
    query_instances = test_df.drop(columns=[target_column]).head(N_QUERIES)

    # 6. Build original pool (N_MAX CFs per query)
    print(f"  Building original pool ({runs}×{per_run} = {runs*per_run} per query)...", flush=True)
    t0 = time.perf_counter()
    pool_result = build_cf_pool(
        cf_method=cf_method,
        query_instances=query_instances,
        pool_cfg=pool_cfg,
        ds_name=f"_convergence_{ds}",  # separate dir to avoid polluting main pools
        pool_path=paths.poolPath,
    )
    query_ids = pool_result["query_ids"]
    pool_time = time.perf_counter() - t0
    print(f"  Original pool built in {pool_time:.1f}s", flush=True)

    # 7. Build perturbed pools
    print(f"  Building perturbed pools (M={M})...", flush=True)
    t0 = time.perf_counter()
    perturbed_data = build_perturbed_pools(
        cf_method=cf_method,
        query_instances=query_instances,
        query_ids=query_ids,
        perturbation_cfg=perturbation_cfg,
        pool_cfg=pool_cfg,
        ds_name=f"_convergence_{ds}",
        transformer=transformer,
        pool_path=paths.poolPath,
    )
    pert_time = time.perf_counter() - t0
    print(f"  Perturbed pools built in {pert_time:.1f}s", flush=True)

    # 8. Load original pools as tensors
    pool_dir = paths.poolPath / f"_convergence_{ds}"
    n_encoded = (
        len(transformer.encoded_continuous_feature_indices)
        + sum(len(g) for g in transformer.encoded_categorical_feature_indices)
    )
    original_pools: dict[str, torch.Tensor] = {}
    cfs_path = pool_dir / f"_convergence_{ds}_original_cfs.csv"
    if cfs_path.is_file():
        cfs_df = pd.read_csv(cfs_path)
        for qid in query_ids:
            qid_cfs = cfs_df[cfs_df["query_id"] == qid]
            feature_cols = [c for c in qid_cfs.columns if c != "query_id"]
            if len(qid_cfs) > 0:
                original_pools[qid] = transformer.transform(qid_cfs[feature_cols])
            else:
                original_pools[qid] = torch.empty(0, n_encoded)
    for qid in query_ids:
        if qid not in original_pools:
            original_pools[qid] = torch.empty(0, n_encoded)

    # Log actual pool sizes
    for qid in query_ids:
        actual = original_pools[qid].shape[0]
        print(f"    Query {qid[:8]}… pool size: {actual}", flush=True)

    # 9. Compute objectives for all candidates
    print(f"  Computing objectives...", flush=True)
    all_candidates = _compute_objectives(
        query_instances=query_instances,
        query_ids=query_ids,
        original_pools=original_pools,
        perturbed_data=perturbed_data,
        transformer=transformer,
        sigma=SIGMA,
    )
    print(f"  Total candidates: {len(all_candidates)}", flush=True)

    # 10. HV at each pool size
    # Group candidates by query, then subsample per query
    by_query: dict[str, list[CandidateObjectives]] = {}
    for c in all_candidates:
        by_query.setdefault(c.query_uuid, []).append(c)

    rows = []
    for size in POOL_SIZES:
        hvs = []
        pf_sizes = []
        for qid, candidates in by_query.items():
            actual_n = min(size, len(candidates))
            hv_val, pf_size = _hv_at_size(candidates, actual_n)
            hvs.append(hv_val)
            pf_sizes.append(pf_size)

        mean_hv = np.mean(hvs)
        mean_pf = np.mean(pf_sizes)
        rows.append({
            "dataset": ds,
            "method": method,
            "pool_size": size,
            "mean_hv": round(float(mean_hv), 6),
            "mean_pf_size": round(float(mean_pf), 1),
            "n_queries": len(by_query),
        })
        print(f"    pool_size={size:>3}  HV={mean_hv:.6f}  |PF|={mean_pf:.1f}", flush=True)

    return rows


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s  %(message)s",
        stream=sys.stdout,
    )
    for name in ("src.cf_methods.gs_method", "growingspheres",
                 "src.cf_methods.lore_method", "lore_sa",
                 "prefect", "prefect.flow_runs", "prefect.task_runs"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    print("=" * 60)
    print("  Pool-Size Convergence Experiment")
    print("=" * 60)
    print(f"  Combos:     {len(COMBOS)}")
    print(f"  N_MAX:      {N_MAX}")
    print(f"  Pool sizes: {POOL_SIZES}")
    print(f"  N_QUERIES:  {N_QUERIES}")
    print(f"  SIGMA:      {SIGMA}")
    print(f"  M:          {M}")
    print("=" * 60)

    all_rows = []
    total_start = time.perf_counter()

    for i, (ds, method) in enumerate(COMBOS, 1):
        combo = f"{ds} × {method}"
        print(f"\n[{i}/{len(COMBOS)}] {combo}", flush=True)
        t0 = time.perf_counter()
        try:
            rows = run_combo(ds, method)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s", flush=True)

    total_elapsed = time.perf_counter() - total_start

    if not all_rows:
        print("\nNo results to report.")
        return

    df = pd.DataFrame(all_rows)

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CONVERGENCE RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    # ── Relative change ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HV RELATIVE CHANGE (ΔHV %)")
    print("=" * 60)
    for (ds, method), group in df.groupby(["dataset", "method"]):
        print(f"\n  {ds} × {method}:")
        hvs = group.sort_values("pool_size")["mean_hv"].values
        sizes = group.sort_values("pool_size")["pool_size"].values
        for j in range(1, len(hvs)):
            if hvs[j - 1] > 0:
                delta = (hvs[j] - hvs[j - 1]) / hvs[j - 1] * 100
            else:
                delta = float("inf")
            print(f"    {sizes[j-1]:>3} → {sizes[j]:>3}: ΔHV = {delta:+.2f}%")

    # ── Save ───────────────────────────────────────────────────
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pool_convergence.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {total_elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
