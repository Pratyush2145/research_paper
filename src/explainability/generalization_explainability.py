"""Generalization-Aware SHAP Analysis module.

Compares SHAP explanations between:
A. LegitPhish internal held-out domain-disjoint test set.
B. PhiUSIIL strict entire-primary-dataset-disjoint external set.

Quantifies:
1. Feature ranking similarity (Spearman rho, Kendall tau).
2. Top-K feature set overlap (Jaccard similarity for Top-10 and Top-20).
3. Feature influence drift: which features become more or less important under domain shift.
4. Generates side-by-side comparative feature importance visualizations.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import shap

logger = logging.getLogger(__name__)


def compare_internal_vs_external_explanations(
    exp_internal: shap.Explanation,
    exp_external: shap.Explanation,
    model_name: str,
    output_fig_dir: str | Path = "reports/figures/shap",
    output_tab_dir: str | Path = "reports/tables/shap",
    top_k: int = 20,
) -> Dict[str, Any]:
    """Perform cross-dataset explainability comparison and rank correlation analysis.

    Args:
        exp_internal: Class-0 Explanation on internal test sample.
        exp_external: Class-0 Explanation on external disjoint sample.
        model_name: Model name.
        output_fig_dir: Figures directory.
        output_tab_dir: Tables directory.
        top_k: Top features to analyze.

    Returns:
        Dictionary containing correlation metrics, shift tables, and figure paths.
    """
    fig_dir = Path(output_fig_dir)
    tab_dir = Path(output_tab_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    fnames = exp_internal.feature_names
    prefix = f"{model_name.lower().replace(' ', '_')}_generalization"

    # Mean absolute SHAP values across all 68 features
    mean_abs_int = np.mean(np.abs(exp_internal.values), axis=0)
    mean_abs_ext = np.mean(np.abs(exp_external.values), axis=0)

    df_comp = pd.DataFrame({
        "feature": fnames,
        "mean_abs_internal": np.round(mean_abs_int, 6),
        "mean_abs_external": np.round(mean_abs_ext, 6),
    })

    df_comp["rank_internal"] = df_comp["mean_abs_internal"].rank(ascending=False, method="min").astype(int)
    df_comp["rank_external"] = df_comp["mean_abs_external"].rank(ascending=False, method="min").astype(int)
    df_comp["rank_shift"] = df_comp["rank_internal"] - df_comp["rank_external"]  # Positive means rose in external rank
    df_comp["abs_diff"] = np.round(np.abs(df_comp["mean_abs_internal"] - df_comp["mean_abs_external"]), 6)

    # Statistical Rank Correlation across all 68 features
    rho, p_spearman = spearmanr(df_comp["rank_internal"], df_comp["rank_external"])
    tau, p_kendall = kendalltau(df_comp["rank_internal"], df_comp["rank_external"])

    # Top-K Jaccard Overlap
    top10_int = set(df_comp.nsmallest(10, "rank_internal")["feature"])
    top10_ext = set(df_comp.nsmallest(10, "rank_external")["feature"])
    jaccard_top10 = len(top10_int.intersection(top10_ext)) / len(top10_int.union(top10_ext))

    top20_int = set(df_comp.nsmallest(20, "rank_internal")["feature"])
    top20_ext = set(df_comp.nsmallest(20, "rank_external")["feature"])
    jaccard_top20 = len(top20_int.intersection(top20_ext)) / len(top20_int.union(top20_ext))

    # Sort by external rank for export
    df_comp_sorted = df_comp.sort_values(by="rank_internal").reset_index(drop=True)
    comp_csv_path = tab_dir / f"{prefix}_feature_comparison.csv"
    df_comp_sorted.to_csv(comp_csv_path, index=False)
    logger.info(f"Saved generalization comparison table to {comp_csv_path}")

    # Generate Comparative Bar Chart: Internal vs. External Top 15 Features
    top_display_feats = df_comp.nsmallest(15, "rank_internal")["feature"].tolist()
    df_plot = df_comp[df_comp["feature"].isin(top_display_feats)].sort_values(by="mean_abs_internal", ascending=True)

    y_pos = np.arange(len(df_plot))
    height = 0.38

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y_pos - height/2, df_plot["mean_abs_internal"], height, label="Internal Test Set (LegitPhish)", color="#2b5c8f", alpha=0.9)
    ax.barh(y_pos + height/2, df_plot["mean_abs_external"], height, label="External Disjoint Set (PhiUSIIL)", color="#e67e22", alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["feature"], fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Phishing Classification)", fontsize=11)
    ax.set_title(
        f"Generalization-Aware Feature Attribution: {model_name}\nInternal vs. External Disjoint SHAP Importance (Top 15 Features)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    comp_plot_path = fig_dir / f"{prefix}_top15_internal_vs_external.png"
    plt.savefig(comp_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved generalization comparison plot to {comp_plot_path}")

    summary_metrics = {
        "model_name": model_name,
        "n_features": len(fnames),
        "spearman_rank_correlation": round(float(rho), 4),
        "spearman_p_value": float(p_spearman),
        "kendall_tau": round(float(tau), 4),
        "kendall_p_value": float(p_kendall),
        "top10_jaccard_overlap": round(float(jaccard_top10), 4),
        "top10_shared_features": sorted(list(top10_int.intersection(top10_ext))),
        "top20_jaccard_overlap": round(float(jaccard_top20), 4),
        "top20_shared_features": sorted(list(top20_int.intersection(top20_ext))),
        "table_csv": str(comp_csv_path),
        "comparison_plot": str(comp_plot_path),
    }

    return summary_metrics

