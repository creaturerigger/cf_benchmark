"""Publication-quality plotting for robustness evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.stability_curve import StabilityCurve
from src.utils.constants import DefaultPaths

# Default params for plotting

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (6, 4),
})


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_stability_curve(
    curve: StabilityCurve,
    dataset_name: str,
    geometric_auc: Optional[float] = None,
    intervention_auc: Optional[float] = None,
    save: bool = True,
    out_dir: Optional[Path] = None,
) -> plt.Figure:
    """Plot geometric and intervention instability as a function of σ.

    Args:
        curve: StabilityCurve with sigma, geometric_mean, intervention_mean.
        dataset_name: used for the title and filename.
        geometric_auc / intervention_auc: if provided, shown in the legend.
        save: persist to ``results/figures/``.
        out_dir: override output directory.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots()

    geo_label = "Geometric instability"
    int_label = "Intervention instability"
    if geometric_auc is not None:
        geo_label += f" (AUC={geometric_auc:.4f})"
    if intervention_auc is not None:
        int_label += f" (AUC={intervention_auc:.4f})"

    ax.plot(curve.sigma, curve.geometric_mean, "o-", label=geo_label, color="#2176AE")
    ax.fill_between(curve.sigma, 0, curve.geometric_mean, alpha=0.15, color="#2176AE")

    ax.plot(curve.sigma, curve.intervention_mean, "s--", label=int_label, color="#E04E39")
    ax.fill_between(curve.sigma, 0, curve.intervention_mean, alpha=0.15, color="#E04E39")

    ax.set_xlabel("Perturbation magnitude (σ)")
    ax.set_ylabel("Mean instability")
    ax.set_title(f"Stability curve — {dataset_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        dest = _ensure_dir(out_dir or DefaultPaths.figuresPath)
        fig.savefig(dest / f"{dataset_name}_stability_curve.pdf")
        fig.savefig(dest / f"{dataset_name}_stability_curve.png")

    return fig


def plot_pareto_front(
    records: list[dict],
    dataset_name: str,
    sigma: Optional[float] = None,
    save: bool = True,
    out_dir: Optional[Path] = None,
) -> plt.Figure:
    """Scatter plot of proximity vs geometric instability with Pareto front.

    Args:
        records: flat record dicts from the pipeline.
        dataset_name: used for title/filename.
        sigma: if given, filter records to this sigma level.
        save: persist to ``results/figures/``.
        out_dir: override output directory.

    Returns:
        The matplotlib Figure.
    """
    df = pd.DataFrame(records)
    if sigma is not None:
        df = df[df["sigma"] == sigma]

    pareto = df[df["is_pareto_optimal"]]
    non_pareto = df[~df["is_pareto_optimal"]]

    fig, ax = plt.subplots()

    if len(non_pareto) > 0:
        ax.scatter(
            non_pareto["proximity"],
            non_pareto["geometric_instability"],
            alpha=0.4, s=30, c="#888888", label="Dominated",
        )
    if len(pareto) > 0:
        ax.scatter(
            pareto["proximity"],
            pareto["geometric_instability"],
            alpha=0.9, s=60, c="#E04E39", edgecolors="black",
            linewidths=0.5, label="Pareto-optimal", zorder=5,
        )
        # Draw the Pareto step-front
        front = pareto.sort_values("proximity")
        ax.step(
            front["proximity"], front["geometric_instability"],
            where="post", color="#E04E39", linewidth=1.2, alpha=0.6,
        )

    ax.set_xlabel("Proximity (lower = closer)")
    ax.set_ylabel("Geometric instability (lower = stabler)")
    sigma_str = f" (σ={sigma})" if sigma is not None else ""
    ax.set_title(f"Pareto front — {dataset_name}{sigma_str}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        dest = _ensure_dir(out_dir or DefaultPaths.figuresPath)
        suffix = f"_sigma{sigma}" if sigma is not None else ""
        fig.savefig(dest / f"{dataset_name}_pareto_front{suffix}.pdf")
        fig.savefig(dest / f"{dataset_name}_pareto_front{suffix}.png")

    return fig


def plot_metric_distributions(
    records: list[dict],
    dataset_name: str,
    save: bool = True,
    out_dir: Optional[Path] = None,
) -> plt.Figure:
    """Boxplots of the three core metrics grouped by σ.

    Args:
        records: flat record dicts.
        dataset_name: for title/filename.
        save: persist to ``results/figures/``.
        out_dir: override output directory.

    Returns:
        The matplotlib Figure.
    """
    df = pd.DataFrame(records)
    metrics = ["proximity", "geometric_instability", "intervention_instability"]
    labels = ["Proximity", "Geometric\ninstability", "Intervention\ninstability"]
    sigmas = sorted(df["sigma"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, metric, label in zip(axes, metrics, labels):
        data = [df[df["sigma"] == s][metric].values for s in sigmas]
        bp = ax.boxplot(
            data, labels=[f"σ={s}" for s in sigmas],
            patch_artist=True, widths=0.5,
        )
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(sigmas)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Metric distributions — {dataset_name}", fontsize=13)
    fig.tight_layout()

    if save:
        dest = _ensure_dir(out_dir or DefaultPaths.figuresPath)
        fig.savefig(dest / f"{dataset_name}_metric_distributions.pdf")
        fig.savefig(dest / f"{dataset_name}_metric_distributions.png")

    return fig


def plot_cross_dataset_auc(
    auc_data: dict[str, dict[str, float]],
    save: bool = True,
    out_dir: Optional[Path] = None,
) -> plt.Figure:
    """Grouped bar chart comparing geometric and intervention AUC across datasets.

    Args:
        auc_data: ``{dataset_name: {"geometric_auc": ..., "intervention_auc": ...}}``.
        save: persist to ``results/figures/``.
        out_dir: override output directory.

    Returns:
        The matplotlib Figure.
    """
    datasets = list(auc_data.keys())
    geo_aucs = [auc_data[d]["geometric_auc"] for d in datasets]
    int_aucs = [auc_data[d]["intervention_auc"] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 2), 4.5))
    ax.bar(x - width / 2, geo_aucs, width, label="Geometric AUC", color="#2176AE")
    ax.bar(x + width / 2, int_aucs, width, label="Intervention AUC", color="#E04E39")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("AUC (lower = more robust)")
    ax.set_title("Cross-dataset robustness comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    if save:
        dest = _ensure_dir(out_dir or DefaultPaths.figuresPath)
        fig.savefig(dest / "cross_dataset_auc.pdf")
        fig.savefig(dest / "cross_dataset_auc.png")

    return fig


def save_tables(
    tables: dict[str, pd.DataFrame],
    dataset_name: str,
    out_dir: Optional[Path] = None,
) -> list[Path]:
    """Persist all aggregation tables to CSV.

    Args:
        tables: mapping of table_name -> DataFrame (from aggregate_results).
        dataset_name: prefix for filenames.
        out_dir: override output directory.

    Returns:
        List of written file paths.
    """
    dest = _ensure_dir(out_dir or DefaultPaths.tablesPath)
    written: list[Path] = []
    for name, df in tables.items():
        path = dest / f"{dataset_name}_{name}.csv"
        df.to_csv(path)
        written.append(path)
    return written


def save_raw_records(
    records: list[dict],
    dataset_name: str,
    out_dir: Optional[Path] = None,
) -> Path:
    """Persist raw experiment records to CSV for later re-plotting.

    Args:
        records: flat record dicts from the pipeline.
        dataset_name: prefix for filename.
        out_dir: override output directory.

    Returns:
        Path to the written file.
    """
    dest = _ensure_dir(out_dir or DefaultPaths.rawPath)
    path = dest / f"{dataset_name}_records.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def save_pareto_cfs(
    records: list[dict],
    dataset_name: str,
    pool_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """Save the Pareto-optimal counterfactual set with feature values + metrics.

    Joins the evaluation records (filtered to Pareto-optimal) back to the
    original pool CSV to produce a single file containing both the CF
    feature vectors and their robustness metrics per sigma level.

    Args:
        records: flat record dicts from the pipeline.
        dataset_name: dataset name used in file paths.
        pool_path: override base pool directory.
        out_dir: override output directory.

    Returns:
        Path to the written CSV.
    """
    rec_df = pd.DataFrame(records)
    if rec_df.empty or "is_pareto_optimal" not in rec_df.columns:
        dest = _ensure_dir(out_dir or DefaultPaths.rawPath)
        path = dest / f"{dataset_name}_pareto_cfs.csv"
        pd.DataFrame().to_csv(path, index=False)
        return path
    pareto = rec_df[rec_df["is_pareto_optimal"]].copy()

    # Load the original CFs pool
    pool_base = pool_path or DefaultPaths.poolPath
    cfs_csv = pool_base / dataset_name / f"{dataset_name}_original_cfs.csv"

    if cfs_csv.is_file():
        pool_df = pd.read_csv(cfs_csv)
        # Build a row-position index within each query_id group
        pool_df["cf_index"] = pool_df.groupby("query_id").cumcount()
        # Merge feature values onto metrics
        pareto = pareto.merge(
            pool_df,
            left_on=["query_uuid", "cf_index"],
            right_on=["query_id", "cf_index"],
            how="left",
        ).drop(columns=["query_id"], errors="ignore")

    dest = _ensure_dir(out_dir or DefaultPaths.rawPath)
    path = dest / f"{dataset_name}_pareto_cfs.csv"
    pareto.to_csv(path, index=False)
    return path


def generate_all_figures(
    records: list[dict],
    tables: dict[str, pd.DataFrame],
    stability: dict[str, Any],
    dataset_name: str,
    save: bool = True,
    out_dir: Optional[Path] = None,
) -> dict[str, plt.Figure]:
    """Generate all standard figures for a single pipeline run.

    Args:
        records: flat record dicts from the pipeline.
        tables: aggregation tables from aggregate_results.
        stability: dict with 'curve', 'geometric_auc', 'intervention_auc'.
        dataset_name: for titles/filenames.
        save: persist to disk.
        out_dir: override output directory.

    Returns:
        Dict mapping figure name to Figure object.
    """
    figures: dict[str, plt.Figure] = {}

    if not records:
        plt.close("all")
        return figures

    # Stability curve
    if stability and "curve" in stability:
        figures["stability_curve"] = plot_stability_curve(
            curve=stability["curve"],
            dataset_name=dataset_name,
            geometric_auc=stability.get("geometric_auc"),
            intervention_auc=stability.get("intervention_auc"),
            save=save,
            out_dir=out_dir,
        )

    df = pd.DataFrame(records)
    for sigma in sorted(df["sigma"].unique()):
        fig = plot_pareto_front(
            records, dataset_name, sigma=sigma,
            save=save, out_dir=out_dir,
        )
        figures[f"pareto_front_sigma{sigma}"] = fig

    figures["metric_distributions"] = plot_metric_distributions(
        records, dataset_name, save=save, out_dir=out_dir,
    )

    plt.close("all")
    return figures


# ── 7.  Replot from saved results ──────────────────────────────────────

def replot_from_disk(
    dataset_name: str,
    cf_method: str = "",
    raw_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> dict[str, plt.Figure]:
    """Regenerate all figures from previously saved CSV artefacts.

    Requires:
        - ``results/<cf_method>/raw/<dataset>_records.csv``  (raw records)
        - ``results/<cf_method>/tables/<dataset>_by_sigma.csv``  (sigma aggregation)

    Args:
        dataset_name: dataset prefix used when saving.
        cf_method: CF method name for path scoping (e.g. "dice").
        raw_dir: override raw records directory.
        tables_dir: override tables directory.
        out_dir: override output directory for figures.

    Returns:
        Dict mapping figure name to Figure object.
    """
    paths = DefaultPaths.for_method(cf_method) if cf_method else DefaultPaths()
    raw_path = (raw_dir or paths.rawPath) / f"{dataset_name}_records.csv"
    sigma_path = (tables_dir or paths.tablesPath) / f"{dataset_name}_by_sigma.csv"

    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Raw records not found: {raw_path}. "
            "Run the pipeline first, or check the dataset name."
        )

    records_df = pd.read_csv(raw_path)
    records = records_df.to_dict("records")

    # Rebuild tables from raw records
    tables: dict[str, pd.DataFrame] = {}
    if sigma_path.is_file():
        tables["by_sigma"] = pd.read_csv(sigma_path, index_col=0)

    # Rebuild stability curve from by_sigma table
    stability: dict[str, Any] = {}
    if "by_sigma" in tables and len(tables["by_sigma"]) > 1:
        from src.evaluation.stability_curve import StabilityCurveBuilder
        builder = StabilityCurveBuilder()
        curve = builder.build(tables["by_sigma"])
        auc = builder.compute_auc(curve)
        stability = {
            "curve": curve,
            "geometric_auc": auc["geometric_auc"],
            "intervention_auc": auc["intervention_auc"],
        }

    return generate_all_figures(
        records=records,
        tables=tables,
        stability=stability,
        dataset_name=dataset_name,
        save=True,
        out_dir=out_dir or paths.figuresPath,
    )
