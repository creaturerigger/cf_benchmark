import logging
import sys

from src.orchestration.prefect_flow import run_pipeline


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        stream=sys.stdout,
    )

    result = run_pipeline(
        dataset_name="adult",
        model_name="pytorch_classifier",
        cf_method_name="dice",
        experiment_name="robustness_experiment",
        seed=42,
        n_queries=3,
        sigmas=[0.03, 0.05],
        overrides={
            "pool": {"runs": 3, "per_run": 3},
            "perturbation": {"M": 3},
        },
    )

    print(f"\nTotal records: {len(result['records'])}")
    for key, df in result["tables"].items():
        print(f"\n── {key} ──")
        print(df.to_string())

    if result["stability"]:
        print(f"\nGeometric AUC:    {result['stability']['geometric_auc']:.4f}")
        print(f"Intervention AUC: {result['stability']['intervention_auc']:.4f}")

    if result.get("timing"):
        print("\n── Stage timing ──")
        for t in result["timing"]:
            print(f"  {t['stage']:40s} {t['seconds']:>8.3f}s")

    if result.get("pool_stats"):
        import pandas as _pd
        ps = _pd.DataFrame(result["pool_stats"])
        totals = {
            "query_id": "TOTAL",
            "generated": ps["generated"].sum(),
            "duplicates": ps["duplicates"].sum(),
            "after_dedup": ps["after_dedup"].sum(),
            "sigma": "",
            "pareto": ps["pareto"].sum() if "pareto" in ps.columns else 0,
        }
        ps = _pd.concat([ps, _pd.DataFrame([totals])], ignore_index=True)
        print("\n── Pool statistics ──")
        print(ps.to_string(index=False))


if __name__ == "__main__":
    main()
