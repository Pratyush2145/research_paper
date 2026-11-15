"""Global SHAP explanation generator for phishing URL detection.

Produces:
1. SHAP summary beeswarm plots.
2. Global mean absolute SHAP bar charts.
3. Feature ranking tables with effect directions.
4. Comparison of SHAP importance vs. Model native split/impurity importance.
5. Class-specific SHAP distribution analysis.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure headless Agg backend and writable cache
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.explainability.explainer import PhishingTreeExplainer

logger = logging.getLogger(__name__)


def generate_global_plots_and_tables(
    explainer: PhishingTreeExplainer,
    explanation: shap.Explanation,
    dataset_name: str,
    output_fig_dir: str | Path = "reports/figures/shap",
    output_tab_dir: str | Path = "reports/tables/shap",
    top_k: int = 20,
) -> Dict[str, Any]:
    """Generate and save global beeswarm, bar plots, and ranking tables.

    Args:
        explainer: Initialized PhishingTreeExplainer.
        explanation: Standardized class-0 Explanation object.
        dataset_name: "legitphish_internal" or "phiusil_external".
        output_fig_dir: Directory to save PNG figures.
        output_tab_dir: Directory to save CSV/JSON tables.
        top_k: Number of top features to display.

    Returns:
        Dictionary containing file paths and summary tables.
    """
    fig_dir = Path(output_fig_dir)
    tab_dir = Path(output_tab_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{explainer.model_name.lower().replace(' ', '_')}_{dataset_name}"

    # 1. Feature Importance Table
    df_rank = explainer.get_feature_importance_table(explanation, top_k=top_k)
    rank_csv_path = tab_dir / f"{prefix}_top{top_k}_features.csv"
    df_rank.to_csv(rank_csv_path, index=False)
    logger.info(f"Saved feature ranking table to {rank_csv_path}")

    # 2. Beeswarm Summary Plot
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=top_k, show=False)
    plt.title(
        f"SHAP Summary (Beeswarm): {explainer.model_name}\nTarget: Phishing (Class 0) | Dataset: {dataset_name.replace('_', ' ').title()}",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("SHAP Value (Impact on Phishing Likelihood: Higher = More Phishing)", fontsize=11)
    beeswarm_path = fig_dir / f"{prefix}_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved beeswarm plot to {beeswarm_path}")

    # 3. Bar Plot (Mean |SHAP|)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(explanation, max_display=top_k, show=False)
    plt.title(
        f"Global Feature Importance (Mean |SHAP|): {explainer.model_name}\nTarget: Phishing (Class 0) | Dataset: {dataset_name.replace('_', ' ').title()}",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Mean |SHAP Value| (Average Impact Magnitude)", fontsize=11)
    bar_path = fig_dir / f"{prefix}_bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved bar plot to {bar_path}")

    # 4. Native Model Importance vs. SHAP Importance Comparison
    native_importances = None
    if hasattr(explainer.model, "feature_importances_"):
        native_importances = explainer.model.feature_importances_

    df_comp = None
    if native_importances is not None:
        all_features = explainer.feature_names
        all_mean_abs_shap = np.mean(np.abs(explanation.values), axis=0)
        df_comp = pd.DataFrame({
            "feature": all_features,
            "shap_mean_abs": np.round(all_mean_abs_shap, 6),
            "native_importance": np.round(native_importances, 6),
        })
        df_comp["shap_rank"] = df_comp["shap_mean_abs"].rank(ascending=False, method="min").astype(int)
        df_comp["native_rank"] = df_comp["native_importance"].rank(ascending=False, method="min").astype(int)
        df_comp["rank_diff"] = df_comp["native_rank"] - df_comp["shap_rank"]
        df_comp = df_comp.sort_values(by="shap_mean_abs", ascending=False).reset_index(drop=True)

        comp_csv_path = tab_dir / f"{prefix}_shap_vs_native_importance.csv"
        df_comp.head(top_k).to_csv(comp_csv_path, index=False)
        logger.info(f"Saved SHAP vs native comparison to {comp_csv_path}")

    return {
        "model_name": explainer.model_name,
        "dataset_name": dataset_name,
        "beeswarm_plot": str(beeswarm_path),
        "bar_plot": str(bar_path),
        "rank_table_csv": str(rank_csv_path),
        "top_features": df_rank.to_dict(orient="records"),
        "shap_vs_native_df": df_comp,
    }


def analyze_class_specific_contributions(
    explanation: shap.Explanation,
    y_true: np.ndarray,
    feature_names: List[str],
    top_k: int = 15,
) -> pd.DataFrame:
    """Analyze how feature contributions differ between actual Phishing and Legitimate URLs.

    Args:
        explanation: Standardized class-0 Explanation object.
        y_true: Ground truth binary labels (0 = Phishing, 1 = Legitimate).
        feature_names: List of feature names.
        top_k: Number of features to report.

    Returns:
        DataFrame summarizing mean signed and absolute SHAP values per class.
    """
    vals = explanation.values
    phish_mask = (y_true == 0)
    legit_mask = (y_true == 1)

    mean_shap_phish = np.mean(vals[phish_mask], axis=0)
    mean_shap_legit = np.mean(vals[legit_mask], axis=0)
    mean_abs_overall = np.mean(np.abs(vals), axis=0)

    df_class = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_overall": np.round(mean_abs_overall, 6),
        "mean_shap_phishing": np.round(mean_shap_phish, 6),
        "mean_shap_legitimate": np.round(mean_shap_legit, 6),
    }).sort_values(by="mean_abs_overall", ascending=False).reset_index(drop=True)

    df_class["rank"] = df_class.index + 1
    df_class["divergence"] = np.round(df_class["mean_shap_phishing"] - df_class["mean_shap_legitimate"], 6)

    cols = ["rank", "feature", "mean_abs_overall", "mean_shap_phishing", "mean_shap_legitimate", "divergence"]
    return df_class[cols].head(top_k)

