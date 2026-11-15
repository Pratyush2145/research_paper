"""Local SHAP explanation generator for phishing URL detection.

Extracts representative case studies across decision outcomes:
1. Correctly classified Phishing URL (True Positive).
2. Correctly classified Legitimate URL (True Negative).
3. False-Negative Phishing URL (Phishing missed by model).
4. False-Positive Legitimate URL (Legitimate falsely blocked by model).
5. High-confidence Phishing prediction (P(phish) > 0.95).
6. Low-confidence / Borderline Phishing prediction (0.40 <= P(phish) <= 0.60).

Produces:
- Individual SHAP waterfall plots.
- Case study metadata table with human-readable security interpretations.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def find_representative_case_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_phish: np.ndarray,
) -> Dict[str, Optional[int]]:
    """Identify sample indices representing key diagnostic decision cases.

    Args:
        y_true: Ground truth binary labels (0 = Phishing, 1 = Legitimate).
        y_pred: Predicted binary labels (0 = Phishing, 1 = Legitimate).
        y_prob_phish: Predicted probability of Phishing (Class 0).

    Returns:
        Dictionary mapping case names to sample indices in the evaluation set.
    """
    cases: Dict[str, Optional[int]] = {
        "true_positive": None,
        "true_negative": None,
        "false_negative": None,
        "false_positive": None,
        "high_confidence_phishing": None,
        "low_confidence_borderline": None,
    }

    # 1. True Positive: Actual Phish (0) and Pred Phish (0)
    tp_indices = np.where((y_true == 0) & (y_pred == 0))[0]
    if len(tp_indices) > 0:
        # Choose a representative one with high confidence
        cases["true_positive"] = int(tp_indices[np.argmax(y_prob_phish[tp_indices])])

    # 2. True Negative: Actual Legit (1) and Pred Legit (1)
    tn_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    if len(tn_indices) > 0:
        # Choose one with lowest phishing probability
        cases["true_negative"] = int(tn_indices[np.argmin(y_prob_phish[tn_indices])])

    # 3. False Negative: Actual Phish (0) but Pred Legit (1)
    fn_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    if len(fn_indices) > 0:
        cases["false_negative"] = int(fn_indices[0])

    # 4. False Positive: Actual Legit (1) but Pred Phish (0)
    fp_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    if len(fp_indices) > 0:
        cases["false_positive"] = int(fp_indices[0])

    # 5. High Confidence Phishing: P(phish) >= 0.95
    hc_indices = np.where((y_true == 0) & (y_prob_phish >= 0.95))[0]
    if len(hc_indices) > 0:
        cases["high_confidence_phishing"] = int(hc_indices[0])

    # 6. Low Confidence Borderline: |P(phish) - 0.50| is minimal
    borderline_diffs = np.abs(y_prob_phish - 0.50)
    cases["low_confidence_borderline"] = int(np.argmin(borderline_diffs))

    return cases


def generate_local_explanations(
    explanation: shap.Explanation,
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_phish: np.ndarray,
    model_name: str,
    dataset_name: str,
    output_fig_dir: str | Path = "reports/figures/shap/local",
    output_tab_dir: str | Path = "reports/tables/shap",
) -> List[Dict[str, Any]]:
    """Generate SHAP waterfall plots and human-readable case audit reports.

    Args:
        explanation: Standardized class-0 Explanation object.
        meta_df: Metadata dataframe with raw_url and registrable_domain.
        y_true: Ground truth binary labels (0 = Phishing, 1 = Legitimate).
        y_pred: Predicted labels.
        y_prob_phish: Probability of Phishing.
        model_name: Model name.
        dataset_name: Dataset name.
        output_fig_dir: Figures directory.
        output_tab_dir: Tables directory.

    Returns:
        List of case report dictionaries.
    """
    fig_dir = Path(output_fig_dir)
    tab_dir = Path(output_tab_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    cases = find_representative_case_indices(y_true, y_pred, y_prob_phish)
    case_reports = []

    prefix = f"{model_name.lower().replace(' ', '_')}_{dataset_name}"

    for case_type, idx in cases.items():
        if idx is None:
            continue

        raw_url = str(meta_df.iloc[idx]["raw_url"])
        domain = str(meta_df.iloc[idx]["registrable_domain"])
        true_lbl = int(y_true[idx])
        pred_lbl = int(y_pred[idx])
        prob_phish = float(y_prob_phish[idx])

        # Slice individual explanation row
        row_exp = explanation[idx]

        # Extract top positive contributors (pushing towards phishing)
        shap_vals = row_exp.values
        fnames = row_exp.feature_names
        feat_data = row_exp.data

        top_pos_idx = np.argsort(shap_vals)[::-1]
        top_pos_feats = []
        for fi in top_pos_idx:
            if shap_vals[fi] > 0.001 and len(top_pos_feats) < 3:
                top_pos_feats.append(f"{fnames[fi]}={feat_data[fi]} (+{shap_vals[fi]:.3f})")

        # Top negative contributors (pushing towards legitimate)
        top_neg_idx = np.argsort(shap_vals)
        top_neg_feats = []
        for fi in top_neg_idx:
            if shap_vals[fi] < -0.001 and len(top_neg_feats) < 3:
                top_neg_feats.append(f"{fnames[fi]}={feat_data[fi]} ({shap_vals[fi]:.3f})")

        # Human-readable interpretation
        status_str = "Correct" if true_lbl == pred_lbl else "Incorrect"
        label_map = {0: "Phishing", 1: "Legitimate"}
        interp = (
            f"URL was {status_str.lower()}ly predicted as {label_map[pred_lbl]} "
            f"(P(phish)={prob_phish:.3f}, True={label_map[true_lbl]}). "
            f"Key risk factors: {', '.join(top_pos_feats) if top_pos_feats else 'None'}. "
            f"Benign indicators: {', '.join(top_neg_feats) if top_neg_feats else 'None'}."
        )

        # Generate and save waterfall plot
        plt.close("all")
        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(row_exp, max_display=10, show=False)
        plt.title(
            f"SHAP Waterfall: {case_type.replace('_', ' ').title()}\nModel: {model_name} | Target: Phishing (Class 0)\nP(phish)={prob_phish:.3f} | True={label_map[true_lbl]}",
            fontsize=11,
            fontweight="bold",
            pad=15,
        )
        waterfall_path = fig_dir / f"{prefix}_{case_type}_waterfall.png"
        plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
        plt.close()

        case_reports.append({
            "model": model_name,
            "dataset": dataset_name,
            "case_type": case_type,
            "sample_index": idx,
            "raw_url": raw_url,
            "domain": domain,
            "true_label": label_map[true_lbl],
            "predicted_label": label_map[pred_lbl],
            "predicted_phishing_probability": round(prob_phish, 4),
            "prediction_status": status_str,
            "top_phishing_contributors": "; ".join(top_pos_feats),
            "top_legitimate_contributors": "; ".join(top_neg_feats),
            "human_interpretation": interp,
            "waterfall_plot": str(waterfall_path),
        })

    # Save summary table
    df_cases = pd.DataFrame(case_reports)
    csv_path = tab_dir / f"{prefix}_case_studies.csv"
    df_cases.to_csv(csv_path, index=False)
    logger.info(f"Saved local case studies to {csv_path}")

    return case_reports

