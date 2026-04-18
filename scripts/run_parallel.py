#!/usr/bin/env python
"""Run all (dataset × method) combos in parallel using multiprocessing.

Usage:
    # Use all available cores minus 1
    python -m scripts.run_parallel

    # Specify max workers
    python -m scripts.run_parallel --workers 12

    # Run a specific scenario
    python -m scripts.run_parallel --scenario lean

    # Custom params
    python -m scripts.run_parallel --n-queries 10 --sigmas 0.03 0.05 0.07 --M 5

    # Dry-run: just show what would run
    python -m scripts.run_parallel --dry-run

    # Run a single combo (useful for retries)
    python -m scripts.run_parallel --only adult,dice

    # Exclude specific combos
    python -m scripts.run_parallel --exclude heloc,gs german,gs

    # Per-combo timeout (seconds, 0 = no timeout)
    python -m scripts.run_parallel --timeout 7200
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from itertools import product
from pathlib import Path

# ── Combos ────────────────────────────────────────────────────
DATASETS = ["adult", "compas", "german", "lending", "heloc", "diabetes"]
METHODS = ["dice", "nice", "gs", "moc", "lore"]

# Known broken — always skip
ALWAYS_SKIP = set()

# Pre-defined scenarios
SCENARIOS = {
             "budget": {"n_queries": 10, "n_sigmas": 3, "M": 3,  "pool_runs": 50, "pool_per_run": 5},
             "lean": {"n_queries": 10, "n_sigmas": 3, "M": 5,  "pool_runs": 50, "pool_per_run": 5},
             "moderate": {"n_queries": 15, "n_sigmas": 3, "M": 5,  "pool_runs": 50, "pool_per_run": 5},
             "practical": {"n_queries": 20, "n_sigmas": 3, "M": 5,  "pool_runs": 50, "pool_per_run": 5},
             "full":     {"n_queries": 50, "n_sigmas": 5, "M": 20, "pool_runs": 50, "pool_per_run": 5},
            }

DEFAULT_SIGMAS = {
    3: [0.03, 0.05, 0.07],
    5: [0.01, 0.03, 0.05, 0.07, 0.10],
}


def _run_one_combo(args: dict) -> dict:
    """Worker function: runs a single (dataset, method) pipeline.

    Executed in a child process — each has its own memory space.
    """
    ds = args["dataset"]
    method = args["method"]
    n_queries = args["n_queries"]
    sigmas = args["sigmas"]
    M = args["M"]
    seed = args["seed"]
    timeout = args["timeout"]
    pool_runs = args.get("pool_runs", 50)
    pool_per_run = args.get("pool_per_run", 5)
    min_pool_size = args.get("min_pool_size", 0)

    combo_id = f"{ds}_{method}"
    pid = os.getpid()

    # Suppress noisy loggers in worker
    logging.basicConfig(
        level=logging.WARNING,
        format=f"%(asctime)s [{combo_id}] %(levelname)s  %(message)s",
    )
    for name in ("src.cf_methods.gs_method", "growingspheres",
                 "src.cf_methods.lore_method", "lore_sa",
                 "prefect", "prefect.flow_runs", "prefect.task_runs"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    print(f"[{combo_id}] START  (pid={pid})", flush=True)

    overrides = {
        "pool": {"runs": pool_runs, "per_run": pool_per_run},
        "perturbation": {"M": M},
    }

    t0 = time.perf_counter()
    try:
        # Timeout via SIGALRM (Unix only)
        if timeout > 0:
            class _Timeout(Exception):
                pass

            def _alarm_handler(signum, frame):
                raise _Timeout()

            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)

        from src.orchestration.prefect_flow import run_pipeline

        result = run_pipeline(
            dataset_name=ds,
            model_name="pytorch_classifier",
            cf_method_name=method,
            experiment_name="robustness_experiment",
            seed=seed,
            n_queries=n_queries,
            sigmas=sigmas,
            overrides=overrides,
        )
        if timeout > 0:
            signal.alarm(0)

        elapsed = time.perf_counter() - t0
        status = "OK"
        timing = {t["stage"]: t["seconds"] for t in result.get("timing", [])}
        n_records = len(result.get("records", []))

    except Exception as e:
        if timeout > 0:
            signal.alarm(0)
        elapsed = time.perf_counter() - t0
        etype = type(e).__name__
        if "Timeout" in etype or "alarm" in str(e).lower():
            status = f"TIMEOUT ({timeout}s)"
        else:
            status = f"FAIL: {etype}: {e}"
        timing = {}
        n_records = 0

    print(f"[{combo_id}] {status}  ({elapsed:.1f}s)", flush=True)

    return {
        "dataset": ds,
        "method": method,
        "status": status,
        "elapsed_secs": round(elapsed, 2),
        "n_records": n_records,
        **{f"stage_{k}": round(v, 2) for k, v in timing.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run CF benchmark combos in parallel across CPU cores."
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=0,
        help="Max parallel workers (0 = ncpus - 1)"
    )
    parser.add_argument(
        "--scenario", "-s", choices=list(SCENARIOS.keys()),
        help="Pre-defined scenario (overrides --n-queries, --sigmas, --M)"
    )
    parser.add_argument("--n-queries", type=int, default=10)
    parser.add_argument("--sigmas", type=float, nargs="+", default=None)
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-runs", type=int, default=50)
    parser.add_argument("--pool-per-run", type=int, default=5)
    parser.add_argument("--min-pool-size", type=int, default=1,
                        help="Reuse existing original pools with at least this many CFs (default: 1)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip combos already marked OK in results/tables/parallel_run_results.csv"
    )
    parser.add_argument(
        "--timeout", type=int, default=0,
        help="Per-combo timeout in seconds (0 = no limit)"
    )
    parser.add_argument(
        "--only", type=str, nargs="+", default=None,
        help="Run only these combos: 'adult,dice' 'compas,moc'"
    )
    parser.add_argument(
        "--exclude", type=str, nargs="+", default=None,
        help="Skip these combos: 'heloc,gs' 'german,gs'"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print combos and exit without running"
    )

    args = parser.parse_args()

    # ── Resolve scenario params ────────────────────────────────
    if args.scenario:
        sc = SCENARIOS[args.scenario]
        n_queries = sc["n_queries"]
        M = sc["M"]
        n_sigmas = sc["n_sigmas"]
        args.pool_runs = sc.get("pool_runs", args.pool_runs)
        args.pool_per_run = sc.get("pool_per_run", args.pool_per_run)
        sigmas = args.sigmas or DEFAULT_SIGMAS.get(n_sigmas, [0.03, 0.05, 0.07])
    else:
        n_queries = args.n_queries
        M = args.M
        sigmas = args.sigmas or [0.03, 0.05, 0.07]

    # ── Build combo list ───────────────────────────────────────
    if args.only:
        combos = [tuple(c.split(",")) for c in args.only]
    else:
        combos = [(ds, m) for ds, m in product(DATASETS, METHODS)
                   if (ds, m) not in ALWAYS_SKIP]

    skip_set = set()
    if args.exclude:
        skip_set = {tuple(c.split(",")) for c in args.exclude}
    combos = [c for c in combos if c not in skip_set]

    # ── Resume: skip already-OK combos ────────────────────────
    out_path = Path("results/tables/parallel_run_results.csv")
    if args.resume and out_path.is_file():
        import pandas as _pd
        prev = _pd.read_csv(out_path)
        done = set(
            zip(prev.loc[prev["status"] == "OK", "dataset"],
                prev.loc[prev["status"] == "OK", "method"])
        )
        skipped = [c for c in combos if c in done]
        combos = [c for c in combos if c not in done]
        if skipped:
            print(f"  Resuming: skipping {len(skipped)} already-OK combos")
            for ds, m in skipped:
                print(f"    {ds} × {m}")

    # ── Workers ────────────────────────────────────────────────
    ncpus = mp.cpu_count()
    workers = args.workers if args.workers > 0 else max(1, ncpus - 1)
    workers = min(workers, len(combos))

    # ── Summary ────────────────────────────────────────────────
    print("=" * 60)
    print(f"  CF Benchmark — Parallel Runner")
    print("=" * 60)
    print(f"  Combos:       {len(combos)}")
    print(f"  Workers:      {workers}  (of {ncpus} CPUs)")
    print(f"  n_queries:    {n_queries}")
    print(f"  sigmas:       {sigmas}")
    print(f"  M:            {M}")
    print(f"  pool:         {args.pool_runs} runs × {args.pool_per_run}/run")
    print(f"  seed:         {args.seed}")
    print(f"  timeout:      {args.timeout}s" if args.timeout else "  timeout:      none")
    print(f"  min_pool_size:{args.min_pool_size}")
    print("=" * 60)

    if args.dry_run:
        print("\nCombos that would run:")
        for ds, m in combos:
            print(f"  {ds} × {m}")
        print(f"\nTotal: {len(combos)} combos")
        return

    # ── Build work items ───────────────────────────────────────
    work_items = [
        {
            "dataset": ds,
            "method": m,
            "n_queries": n_queries,
            "sigmas": sigmas,
            "M": M,
            "seed": args.seed,
            "timeout": args.timeout,
            "pool_runs": args.pool_runs,
            "pool_per_run": args.pool_per_run,
            "min_pool_size": args.min_pool_size,
        }
        for ds, m in combos
    ]

    # ── Run ────────────────────────────────────────────────────
    wall_start = time.perf_counter()

    # Use spawn to avoid fork issues with PyTorch / Prefect
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(_run_one_combo, work_items)

    wall_elapsed = time.perf_counter() - wall_start

    # ── Results ────────────────────────────────────────────────
    import pandas as pd

    df = pd.DataFrame(results)

    # Merge with previous results when resuming
    if args.resume and out_path.is_file():
        prev = pd.read_csv(out_path)
        prev_ok = prev[prev["status"] == "OK"]
        df = pd.concat([prev_ok, df], ignore_index=True)
    ok = df[df["status"] == "OK"]
    failed = df[df["status"] != "OK"]

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(df[["dataset", "method", "status", "elapsed_secs", "n_records"]].to_string(index=False))

    if len(failed):
        print(f"\nFailed ({len(failed)}):")
        print(failed[["dataset", "method", "status"]].to_string(index=False))

    cpu_hours = ok["elapsed_secs"].sum() / 3600
    print(f"\nOK: {len(ok)}/{len(df)}")
    print(f"Wall time:  {wall_elapsed/3600:.2f}h ({wall_elapsed:.0f}s)")
    print(f"CPU time:   {cpu_hours:.2f}h")
    print(f"Speedup:    {cpu_hours / (wall_elapsed/3600):.1f}x over sequential")

    # Save
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parallel_run_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
