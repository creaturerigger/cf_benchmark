#!/usr/bin/env python
"""Timing benchmark: run every working (dataset × method) with small parameters
to estimate how long full-scale experiments will take.

Updated to reflect the refactored pipeline where:
  - Original pools are built ONCE (not per sigma)
  - Perturbed pools are built per sigma
  - Selection analysis adds negligible overhead

Mini params: pool=3×2=6, n_queries=2, M=2, 2 sigmas
This gives enough info to separate original-pool vs perturbed-pool cost.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from itertools import product

import numpy as np
import pandas as pd

from src.orchestration.prefect_flow import run_pipeline

DATASETS = ["adult", "compas", "german", "lending", "heloc", "diabetes"]
METHODS = ["dice", "nice", "gs", "moc", "lore"]

# Known broken combo — skip it
SKIP = set()

# Minimal overrides for fast timing
MINI_OVERRIDES = {
    "pool": {"runs": 3, "per_run": 2},
    "perturbation": {"M": 2},
}
MINI_N_QUERIES = 2
MINI_SIGMAS = [0.03, 0.05]  # 2 sigmas to measure per-sigma cost

MINI_POOL_SIZE = MINI_OVERRIDES["pool"]["runs"] * MINI_OVERRIDES["pool"]["per_run"]
MINI_M = MINI_OVERRIDES["perturbation"]["M"]
MINI_N_SIGMAS = len(MINI_SIGMAS)

# Per-combo wall-clock timeout (seconds)
COMBO_TIMEOUT = 600  # 10 min


# ── Scenario definitions ──────────────────────────────────────

SCENARIOS = {
    "Lean": {"n_queries": 10, "pool_size": 250, "n_sigmas": 3, "M": 5},
    "Moderate": {"n_queries": 15, "pool_size": 250, "n_sigmas": 3, "M": 5},
    "Practical": {"n_queries": 20, "pool_size": 250, "n_sigmas": 3, "M": 5},
    "Full": {"n_queries": 50, "pool_size": 250, "n_sigmas": 5, "M": 20},
}


def compute_work_units(n_queries, pool_size, n_sigmas, M):
    """CF-generation work: original pool (once) + perturbed pools (per sigma).

    work = n_queries × pool_size × (1 + n_sigmas × M)
    """
    return n_queries * pool_size * (1 + n_sigmas * M)


MINI_WORK = compute_work_units(MINI_N_QUERIES, MINI_POOL_SIZE, MINI_N_SIGMAS, MINI_M)


def extrapolate(mini_seconds: float, scenario: dict) -> float:
    """Estimate full-scale time from a mini run."""
    full_work = compute_work_units(**scenario)
    return mini_seconds * full_work / MINI_WORK


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

    class _Timeout(Exception):
        pass

    def _alarm_handler(signum, frame):
        raise _Timeout()

    rows: list[dict] = []
    total_start = time.perf_counter()

    combos = [(ds, m) for ds, m in product(DATASETS, METHODS) if (ds, m) not in SKIP]
    n_combos = len(combos)

    for i, (ds, method) in enumerate(combos, 1):
        combo = f"{ds} × {method}"
        print(f"\n[{i}/{n_combos}] {combo}", flush=True)

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

        row = {
            "dataset": ds,
            "method": method,
            "status": status,
            "mini_secs": round(elapsed, 2),
            **{f"stage_{k}": v for k, v in timing.items()},
        }
        # Add estimates for each scenario
        for name, scenario in SCENARIOS.items():
            est = extrapolate(elapsed, scenario) if status == "OK" else float("nan")
            row[f"est_{name}_hours"] = round(est / 3600, 2)

        rows.append(row)
        lean_est = row.get("est_Lean_hours", float("nan"))
        print(f"  {elapsed:.1f}s  →  Lean est: {lean_est:.1f}h", flush=True)

    total_elapsed = time.perf_counter() - total_start

    # ── Summary table ──────────────────────────────────────────
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "OK"]

    print("\n" + "=" * 80)
    print("  TIMING RESULTS")
    print("=" * 80)
    summary_cols = ["dataset", "method", "status", "mini_secs"]
    for name in SCENARIOS:
        summary_cols.append(f"est_{name}_hours")
    print(ok[summary_cols].to_string(index=False))

    # Failed combos
    failed = df[df["status"] != "OK"]
    if len(failed):
        print(f"\nFailed ({len(failed)}):")
        print(failed[["dataset", "method", "status"]].to_string(index=False))

    # ── Per-scenario totals ────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SCENARIO ESTIMATES (sequential total for all combos)")
    print("=" * 80)
    print(f"{'Scenario':<12} {'Params':<40} {'Total (h)':>10} {'4-core (h)':>11}")
    print("-" * 75)
    for name, scenario in SCENARIOS.items():
        col = f"est_{name}_hours"
        total_h = ok[col].sum()
        params = (f"q={scenario['n_queries']}, σ={scenario['n_sigmas']}, "
                  f"M={scenario['M']}, pool={scenario['pool_size']}")
        print(f"{name:<12} {params:<40} {total_h:>10.1f} {total_h/4:>11.1f}")

    # ── Per-method totals (Lean) ───────────────────────────────
    print("\n" + "=" * 80)
    print("  PER-METHOD TOTALS (Lean scenario)")
    print("=" * 80)
    method_totals = ok.groupby("method")["est_Lean_hours"].agg(["sum", "mean", "max"])
    method_totals.columns = ["total_h", "mean_h", "max_h"]
    print(method_totals.round(1).to_string())

    # ── Per-dataset totals (Lean) ──────────────────────────────
    print("\n" + "=" * 80)
    print("  PER-DATASET TOTALS (Lean scenario)")
    print("=" * 80)
    ds_totals = ok.groupby("dataset")["est_Lean_hours"].agg(["sum", "mean", "max"])
    ds_totals.columns = ["total_h", "mean_h", "max_h"]
    print(ds_totals.round(1).to_string())

    print(f"\nMini benchmark total: {total_elapsed/60:.1f} min")
    print(f"Working combos: {len(ok)}/{len(df)}")

    # Save
    import os
    out_path = "results/tables/timing_benchmark.csv"
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

