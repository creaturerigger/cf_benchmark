"""Post-hoc analysis script: selection strategy comparison.

Loads raw records produced by the pipeline, applies all selectors,
computes AUC metrics, and generates the tables + figures described
in the experimentation document.

Usage:
    python -m scripts.selection_analysis                    # all methods + datasets found in results/
    python -m scripts.selection_analysis --methods dice nice --datasets adult compas
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.selectors import apply_all_selectors, SELECTOR_REGISTRY
from src.evaluation.stability_curve import StabilityCurveBuilder
from src.utils.constants import DefaultPaths

# ── Plot defaults ─────────────────────────────────────────────
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
})

METRICS = ["proximity", "geometric_instability", "intervention_instability"]
PRIMARY_SELECTORS = [
    "min_proximity",
    "weighted_sum_equal",
    "weighted_sum_prox_heavy",
    "pareto_knee",
    "pareto_lex",
]

OUT = Path("results/selection_analysis")


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def discover_raw_records(
    methods: list[str] | None = None,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Walk results/<method>/raw/ and load *_records.csv files."""
    base = Path("results")
    frames = []
    for method_dir in sorted(base.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        if method in ("models", "tables", "figures", "raw", "pools", "logs", "selection_analysis"):
            continue
        if methods and method not in methods:
            continue
        raw_dir = method_dir / "raw"
        if not raw_dir.is_dir():
            continue
        for csv in sorted(raw_dir.glob("*_records.csv")):
            ds_name = csv.stem.replace("_records", "")
            if datasets and ds_name not in datasets:
                continue
            df = pd.read_csv(csv)
            df["method"] = method
            df["dataset"] = ds_name
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No raw record CSVs found under results/")
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# Selector application
# ──────────────────────────────────────────────────────────────

def run_selection(all_records: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Apply selectors per (method, dataset, query, sigma)."""
    groups = all_records.groupby(["method", "dataset"])
    frames = []
    for (method, dataset), grp in groups:
        records = grp.to_dict("records")
        sel = apply_all_selectors(records, selector_names=PRIMARY_SELECTORS, seed=seed)
        df = pd.DataFrame(sel)
        df["method"] = method
        df["dataset"] = dataset
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# AUC computation per selector (across sigma levels)
# ──────────────────────────────────────────────────────────────

def compute_selector_auc(sel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Geo-AUC and Int-AUC for each (method, dataset, selector).

    For each group, averages metrics per sigma across queries, then
    applies the trapezoidal rule over sigma levels.
    """
    rows = []
    for (method, dataset, selector), grp in sel_df.groupby(["method", "dataset", "selector"]):
        sigma_means = grp.groupby("sigma")[METRICS].mean().sort_index()
        sigmas = sigma_means.index.values
        if len(sigmas) < 2:
            geo_auc = float(sigma_means["geometric_instability"].iloc[0])
            int_auc = float(sigma_means["intervention_instability"].iloc[0])
        else:
            geo_auc = float(np.trapz(sigma_means["geometric_instability"].values, sigmas))
            int_auc = float(np.trapz(sigma_means["intervention_instability"].values, sigmas))
        mean_prox = float(grp["proximity"].mean())
        rows.append({
            "method": method,
            "dataset": dataset,
            "selector": selector,
            "proximity": round(mean_prox, 6),
            "geo_auc": round(geo_auc, 6),
            "int_auc": round(int_auc, 6),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Win-rate computation
# ──────────────────────────────────────────────────────────────

def compute_win_rate(sel_df: pd.DataFrame) -> pd.DataFrame:
    """For each (method, dataset, query, sigma) determine which selector wins.

    A selector 'wins' if it achieves the lowest normalised combined score:
        score = prox_norm + geo_norm + int_norm
    """
    win_counts: dict[str, int] = {s: 0 for s in PRIMARY_SELECTORS}
    total = 0

    for (method, dataset, qid, sigma), grp in sel_df.groupby(
        ["method", "dataset", "query_uuid", "sigma"]
    ):
        if len(grp) < 2:
            continue
        # min-max normalise within this group
        for m in METRICS:
            mn, mx = grp[m].min(), grp[m].max()
            span = mx - mn if mx != mn else 1.0
            grp = grp.copy()
            grp[f"{m}_norm"] = (grp[m] - mn) / span
        grp["combined"] = (
            grp["proximity_norm"]
            + grp["geometric_instability_norm"]
            + grp["intervention_instability_norm"]
        )
        winner = grp.loc[grp["combined"].idxmin(), "selector"]
        win_counts[winner] = win_counts.get(winner, 0) + 1
        total += 1

    if total == 0:
        return pd.DataFrame(columns=["selector", "wins", "total", "win_rate"])
    rows = []
    for s in PRIMARY_SELECTORS:
        rows.append({
            "selector": s,
            "wins": win_counts.get(s, 0),
            "total": total,
            "win_rate": round(win_counts.get(s, 0) / total, 4),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Tables (as described in the document)
# ──────────────────────────────────────────────────────────────

def table1_aggregated(auc_df: pd.DataFrame, win_df: pd.DataFrame) -> pd.DataFrame:
    """Table 1 — aggregated across all datasets and methods."""
    agg = auc_df.groupby("selector")[["proximity", "geo_auc", "int_auc"]].mean().reset_index()
    if not win_df.empty:
        agg = agg.merge(win_df[["selector", "win_rate"]], on="selector", how="left")
    else:
        agg["win_rate"] = float("nan")
    return agg.round(4)


def table2_by_method(auc_df: pd.DataFrame) -> pd.DataFrame:
    """Table 2 — per-method breakdown."""
    return (
        auc_df.groupby(["method", "selector"])[["proximity", "geo_auc", "int_auc"]]
        .mean()
        .reset_index()
        .round(4)
    )


def table3_by_dataset(auc_df: pd.DataFrame) -> pd.DataFrame:
    """Table 3 — per-dataset breakdown."""
    return (
        auc_df.groupby(["dataset", "selector"])[["proximity", "geo_auc", "int_auc"]]
        .mean()
        .reset_index()
        .round(4)
    )


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────

SELECTOR_COLORS = {
    "min_proximity": "#1b9e77",
    "weighted_sum_equal": "#d95f02",
    "weighted_sum_prox_heavy": "#e7298a",
    "pareto_knee": "#2176AE",
    "pareto_lex": "#E04E39",
}
SELECTOR_LABELS = {
    "min_proximity": "Min-Proximity",
    "weighted_sum_equal": "WS (equal)",
    "weighted_sum_prox_heavy": "WS (prox-heavy)",
    "pareto_knee": "Pareto-Knee",
    "pareto_lex": "Pareto-Lex",
}


def fig1_stability_curves_grid(sel_df: pd.DataFrame, out: Path) -> None:
    """Figure 1 — panel grid: rows=methods, cols=selectors.

    Each panel: x=sigma, y=mean instability (geo + int lines).
    """
    methods = sorted(sel_df["method"].unique())
    selectors = [s for s in PRIMARY_SELECTORS if s in sel_df["selector"].unique()]
    if not methods or not selectors:
        return

    fig, axes = plt.subplots(
        len(methods), len(selectors),
        figsize=(4 * len(selectors), 3 * len(methods)),
        sharex=True, sharey=True, squeeze=False,
    )

    for i, method in enumerate(methods):
        for j, selector in enumerate(selectors):
            ax = axes[i, j]
            sub = sel_df[(sel_df["method"] == method) & (sel_df["selector"] == selector)]
            if sub.empty:
                ax.set_visible(False)
                continue
            sigma_means = sub.groupby("sigma")[METRICS].mean().sort_index()
            ax.plot(
                sigma_means.index, sigma_means["geometric_instability"],
                "o-", color="#2176AE", markersize=4, label="Geometric",
            )
            ax.plot(
                sigma_means.index, sigma_means["intervention_instability"],
                "s--", color="#E04E39", markersize=4, label="Intervention",
            )
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(SELECTOR_LABELS.get(selector, selector), fontsize=10)
            if j == 0:
                ax.set_ylabel(method, fontsize=10)
            if i == len(methods) - 1:
                ax.set_xlabel("σ")
            if i == 0 and j == len(selectors) - 1:
                ax.legend(fontsize=7)

    fig.suptitle("Stability Curves by Selector and Method", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "fig1_stability_curves_grid.pdf")
    fig.savefig(out / "fig1_stability_curves_grid.png")
    plt.close(fig)


def fig2_trade_off_scatter(
    all_records: pd.DataFrame,
    sel_df: pd.DataFrame,
    out: Path,
    max_queries: int = 3,
) -> None:
    """Figure 2 — trade-off scatter with highlighted selected points.

    Pick a few representative queries and show candidates + selector picks.
    """
    # Pick first sigma available
    sigma = sorted(all_records["sigma"].unique())[0]
    rec = all_records[all_records["sigma"] == sigma]
    qids = rec["query_uuid"].unique()[:max_queries]

    for qid in qids:
        q_rec = rec[rec["query_uuid"] == qid]
        q_sel = sel_df[
            (sel_df["query_uuid"] == qid) & (sel_df["sigma"] == sigma)
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, y_col, y_label in zip(
            axes,
            ["geometric_instability", "intervention_instability"],
            ["Geometric instability", "Intervention instability"],
        ):
            # All candidates
            is_pareto = q_rec.get("is_pareto_optimal", pd.Series(dtype=bool))
            pareto_mask = is_pareto.fillna(False).astype(bool)
            ax.scatter(
                q_rec["proximity"], q_rec[y_col],
                alpha=0.3, s=20, c="#999999", label="Candidates",
            )
            if pareto_mask.any():
                p = q_rec[pareto_mask]
                ax.scatter(p["proximity"], p[y_col], alpha=0.5, s=35,
                           c="none", edgecolors="black", linewidths=0.8, label="Pareto front")

            # Selector picks
            for _, row in q_sel.iterrows():
                sel_name = row["selector"]
                ax.scatter(
                    row["proximity"], row[y_col],
                    s=100, marker="*", zorder=10,
                    c=SELECTOR_COLORS.get(sel_name, "black"),
                    label=SELECTOR_LABELS.get(sel_name, sel_name),
                )

            ax.set_xlabel("Proximity")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            if ax == axes[0]:
                ax.legend(fontsize=7, loc="upper right")

        ds = q_rec["dataset"].iloc[0] if "dataset" in q_rec.columns else ""
        method = q_rec["method"].iloc[0] if "method" in q_rec.columns else ""
        fig.suptitle(f"Trade-off scatter — {ds} / {method} (σ={sigma})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        short_qid = qid[:8]
        fig.savefig(out / f"fig2_tradeoff_{short_qid}.pdf")
        fig.savefig(out / f"fig2_tradeoff_{short_qid}.png")
        plt.close(fig)


def fig3_win_rate_bar(win_df: pd.DataFrame, out: Path) -> None:
    """Figure 3 — win-rate bar chart."""
    if win_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [SELECTOR_LABELS.get(s, s) for s in win_df["selector"]]
    colors = [SELECTOR_COLORS.get(s, "#888888") for s in win_df["selector"]]
    bars = ax.bar(labels, win_df["win_rate"], color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, win_df["win_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Win rate")
    ax.set_title("Selector Win Rates (normalised combined score)")
    ax.set_ylim(0, min(1.0, win_df["win_rate"].max() + 0.15))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_win_rate.pdf")
    fig.savefig(out / "fig3_win_rate.png")
    plt.close(fig)


def fig4_proximity_vs_robustness(auc_df: pd.DataFrame, out: Path) -> None:
    """Figure 4 — proximity vs combined instability AUC per selector."""
    if auc_df.empty:
        return
    agg = auc_df.groupby("selector")[["proximity", "geo_auc", "int_auc"]].mean()
    agg["combined_auc"] = agg["geo_auc"] + agg["int_auc"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for sel in agg.index:
        ax.scatter(
            agg.loc[sel, "proximity"],
            agg.loc[sel, "combined_auc"],
            s=150, marker="o", zorder=5,
            color=SELECTOR_COLORS.get(sel, "#888"),
            edgecolors="black", linewidths=0.6,
            label=SELECTOR_LABELS.get(sel, sel),
        )
    ax.set_xlabel("Mean Proximity ↓")
    ax.set_ylabel("Combined Instability AUC (Geo + Int) ↓")
    ax.set_title("Robustness–Proximity Trade-off by Selector")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig4_proximity_vs_robustness.pdf")
    fig.savefig(out / "fig4_proximity_vs_robustness.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Selection strategy analysis")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = _ensure(OUT)

    print("Loading raw records ...")
    all_records = discover_raw_records(methods=args.methods, datasets=args.datasets)
    n_methods = all_records["method"].nunique()
    n_datasets = all_records["dataset"].nunique()
    n_records = len(all_records)
    print(f"  {n_records} records across {n_methods} methods × {n_datasets} datasets")

    print("Applying selectors ...")
    sel_df = run_selection(all_records, seed=args.seed)
    print(f"  {len(sel_df)} selector results")

    print("Computing AUC metrics ...")
    auc_df = compute_selector_auc(sel_df)

    print("Computing win rates ...")
    win_df = compute_win_rate(sel_df)

    # ── Tables ────────────────────────────────────────────────
    t1 = table1_aggregated(auc_df, win_df)
    t2 = table2_by_method(auc_df)
    t3 = table3_by_dataset(auc_df)

    t1.to_csv(out / "table1_aggregated.csv", index=False)
    t2.to_csv(out / "table2_by_method.csv", index=False)
    t3.to_csv(out / "table3_by_dataset.csv", index=False)
    win_df.to_csv(out / "win_rates.csv", index=False)
    auc_df.to_csv(out / "selector_auc_detail.csv", index=False)
    sel_df.to_csv(out / "selection_records.csv", index=False)

    print("\n── Table 1: Aggregated Selector Comparison ──")
    print(t1.to_string(index=False))
    print("\n── Win Rates ──")
    print(win_df.to_string(index=False))

    # ── Figures ───────────────────────────────────────────────
    print("\nGenerating figures ...")
    fig1_stability_curves_grid(sel_df, out)
    fig2_trade_off_scatter(all_records, sel_df, out)
    fig3_win_rate_bar(win_df, out)
    fig4_proximity_vs_robustness(auc_df, out)

    print(f"\nAll outputs saved to {out}/")


if __name__ == "__main__":
    main()
