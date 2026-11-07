"""Plots and tables written to reports/."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of every metric per (model, split) across seeds."""
    metrics = ["rmse", "mae", "r2", "spearman"]
    g = results.groupby(["model", "split"])[metrics].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    return g.reset_index()


def write_summary_markdown(summary: pd.DataFrame, path: Path, notes: str = "") -> None:
    lines = ["# Ablation summary (scaffold splits, mean ± std over seeds)\n"]
    for split in ("test", "val"):
        sub = summary[summary["split"] == split].sort_values("rmse_mean")
        lines += [f"\n## {split.capitalize()} set\n", "| model | RMSE | MAE | R² | Spearman |", "|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['model']} | {r['rmse_mean']:.3f} ± {np.nan_to_num(r['rmse_std']):.3f} "
                f"| {r['mae_mean']:.3f} ± {np.nan_to_num(r['mae_std']):.3f} "
                f"| {r['r2_mean']:.3f} ± {np.nan_to_num(r['r2_std']):.3f} "
                f"| {r['spearman_mean']:.3f} ± {np.nan_to_num(r['spearman_std']):.3f} |"
            )
    if notes:
        lines += ["\n" + notes]
    Path(path).write_text("\n".join(lines) + "\n")


def plot_ablation(summary: pd.DataFrame, path: Path) -> None:
    sub = summary[summary["split"] == "test"].sort_values("rmse_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(sub) + 1.5))
    ax.barh(sub["model"], sub["rmse_mean"], xerr=sub["rmse_std"].fillna(0), color="#4c78a8", capsize=3)
    ax.set_xlabel("Test RMSE (pIC50 units, lower is better)")
    ax.set_title("Feature-set ablation on scaffold-split test sets")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ad_calibration_table(tmax: np.ndarray, abs_err: np.ndarray, thresholds=None) -> pd.DataFrame:
    """For each candidate threshold: coverage and RMSE inside vs outside the domain."""
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.15, 0.76, 0.05), 2)
    rows = []
    for t in thresholds:
        inside = tmax >= t
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(inside.mean()),
                "n_inside": int(inside.sum()),
                "rmse_inside": float(np.sqrt(np.mean(abs_err[inside] ** 2))) if inside.sum() >= 5 else np.nan,
                "rmse_outside": float(np.sqrt(np.mean(abs_err[~inside] ** 2))) if (~inside).sum() >= 5 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ad_error_spearman(tmax: np.ndarray, abs_err: np.ndarray) -> Optional[float]:
    ok = ~np.isnan(tmax)
    if ok.sum() < 5:
        return None
    rho, _ = spearmanr(tmax[ok], abs_err[ok])
    return None if np.isnan(rho) else float(rho)


def plot_ad_error(tmax: np.ndarray, abs_err: np.ndarray, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(tmax, abs_err, s=14, alpha=0.6, color="#4c78a8")
    ax.axvline(threshold, color="crimson", ls="--", label=f"AD threshold = {threshold}")
    ax.set_xlabel("Max Tanimoto similarity to training set (Tmax)")
    ax.set_ylabel("|prediction error| (pIC50)")
    ax.set_title("Does the applicability domain predict error? (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
