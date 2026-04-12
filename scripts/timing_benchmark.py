#!/usr/bin/env python
"""Minimal timing benchmark: run every (dataset × method) with tiny parameters
to estimate how long the full-scale experiment will take.

Full-scale target: pool_size = 250 (e.g. runs=50, per_run=5),
                   n_queries = 50,  M = 20,  sigmas = [0.01, 0.03, 0.05, 0.07, 0.10]

This script runs with: pool = 3×2 = 6, n_queries = 2, M = 2, 1 sigma.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from itertools import product

import pandas as pd

from src.orchestration.prefect_flow import run_pipeline

DATASETS = ["adult", "compas", "german", "lending", "heloc", "credit_default"]
METHODS = ["dice", "nice", "gs", "moc", "lore"]

# Minimal overrides for fast timing
MINI_OVERRIDES = {
    "pool": {"runs": 3, "per_run": 2},
    "perturbation": {"M": 2},
}
MINI_N_QUERIES = 2
MINI_SIGMAS = [0.05]

# Per-combo wall-clock timeout (seconds). If a combo exceeds this it's
# recorded as a timeout so we can still estimate the rest.
COMBO_TIMEOUT = 300  # 5 min

# Full-scale parameters (for extrapolation)
FULL_N_QUERIES = 50
FULL_POOL_SIZE = 250          # runs × per_run
FULL_M = 20
FULL_N_SIGMAS = 5

MINI_POOL_SIZE = MINI_OVERRIDES["pool"]["runs"] * MINI_OVERRIDES["pool"]["per_run"]
MINI_M = MINI_OVERRIDES["perturbation"]["M"]
MINI_N_SIGMAS = len(MINI_SIGMAS)


def extrapolate(mini_seconds: float) -> float:
    """Estimate full-scale time from a mini run.

    The pipeline cost is dominated by CF generation which scales as:
        n_queries × n_sigmas × (pool_size + M × pool_size)
    """
    mini_work = MINI_N_QUERIES * MINI_N_SIGMAS * MINI_POOL_SIZE * (1 + MINI_M)
    full_work = FULL_N_QUERIES * FULL_N_SIGMAS * FULL_POOL_SIZE * (1 + FULL_M)
    factor = full_work / mini_work
    return mini_seconds * factor


def main():
    logging.basicConfig(
        level=logging.WARNING,   # reduce noise
        format="%(asctime)s %(levelname)s  %(message)s",
        stream=sys.stdout,
    )
    # Suppress extremely noisy loggers completely
    for name in ("src.cf_methods.gs_method", "growingspheres",
                 "src.cf_methods.lore_method", "lore_sa"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    class _Timeout(Exception):
        pass

    def _alarm_handler(signum, frame):
        raise _Timeout()

    rows: list[dict] = []
    total_start = time.perf_counter()

    for ds, method in product(DATASETS, METHODS):
        combo = f"{ds} × {method}"
        print(f"\n{'='*60}", flush=True)
        print(f"  {combo}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.perf_counter()
        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(COMBO_TIMEOUT)
            result = run_pipeline(
                dataset_name=ds,
                model_name="pytorch_classifier",
                cf_method_name=method,
                experiment_name="robustness_experiment",
                seed=42,
                n_queries=MINI_N_QUERIES,
                sigmas=MINI_SIGMAS,
                overrides=MINI_OVERRIDES,
            )
            signal.alarm(0)
            elapsed = time.perf_counter() - t0
            status = "OK"
            timing = {t["stage"]: t["seconds"] for t in result.get("timing", [])}
        except _Timeout:
            signal.alarm(0)
            elapsed = time.perf_counter() - t0
            status = f"TIMEOUT ({COMBO_TIMEOUT}s)"
            timing = {}
        except Exception as e:
            signal.alarm(0)
            elapsed = time.perf_counter() - t0
            status = f"FAIL: {type(e).__name__}: {e}"
            timing = {}

        estimated_full = extrapolate(elapsed)

        rows.append({
            "dataset": ds,
            "method": method,
            "status": status,
            "mini_secs": round(elapsed, 2),
            "est_full_secs": round(estimated_full, 1),
            "est_full_mins": round(estimated_full / 60, 1),
            "est_full_hours": round(estimated_full / 3600, 2),
            **{f"stage_{k}": v for k, v in timing.items()},
        })
        print(f"  ⏱  {elapsed:.2f}s  →  est. full: {estimated_full/60:.1f} min")

    total_elapsed = time.perf_counter() - total_start

    # Summary table
    df = pd.DataFrame(rows)
    core_cols = ["dataset", "method", "status", "mini_secs",
                 "est_full_secs", "est_full_mins", "est_full_hours"]
    print("\n" + "=" * 80)
    print("  TIMING SUMMARY")
    print("=" * 80)
    print(df[core_cols].to_string(index=False))

    total_est_hours = df["est_full_hours"].sum()
    print(f"\nMini benchmark total:  {total_elapsed/60:.1f} min")
    print(f"Estimated full total:  {total_est_hours:.1f} hours")
    print(f"  (sequential — can be parallelised across datasets/methods)")

    # Save to CSV
    import os
    out_path = "results/tables/timing_benchmark.csv"
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
