"""SHAP Feature Dependence and Interaction Analysis module.

Generates partial dependence / SHAP dependence scatter plots for key structural
and lexical URL features, analyzing how feature values influence model predictions
and identifying non-linear interactions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

logger = logging.getLogger(__name__)


def generate_dependence_plots(
    explanation: shap.Explanation,
    model_name: str,
    dataset_name: str,
    target_features: Optional[List[str]] = None,
    output_dir: str | Path = "reports/figures/shap/dependence",
) -> List[str]:
    """Generate and save SHAP dependence plots for designated key features.

    Args:
        explanation: Standardized class-0 Explanation object.
        model_name: Model name.
        dataset_name: Dataset name.
        target_features: Features to plot. Defaults to top structural/lexical features.
        output_dir: Directory to save dependence plot PNGs.

    Returns:
        List of generated plot file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if target_features is None:
        target_features = [
            "path_segment_count",
            "path_length",
            "protocol_type",
            "url_entropy",
            "subdomain_count",
            "digit_ratio",
            "special_char_count",
            "max_consecutive_repeated_char",
        ]

    saved_paths = []
    prefix = f"{model_name.lower().replace(' ', '_')}_{dataset_name}"
    fnames = explanation.feature_names

    for feat in target_features:
        if feat not in fnames:
            continue

        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 5.5))

        try:
            # Let SHAP automatically pick the strongest interaction feature
            shap.plots.scatter(explanation[:, feat], color=explanation, show=False)
            plt.title(
                f"SHAP Dependence: {feat}\nModel: {model_name} | Dataset: {dataset_name.replace('_', ' ').title()}",
                fontsize=11,
                fontweight="bold",
                pad=12,
            )
            plt.ylabel(f"SHAP Value for {feat} (Class 0: Phishing)", fontsize=10)
            plt.xlabel(f"Feature Value: {feat}", fontsize=10)

            plot_file = out_path / f"{prefix}_dependence_{feat}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            saved_paths.append(str(plot_file))
            logger.info(f"Saved dependence plot to {plot_file}")
        except Exception as e:
            logger.warning(f"Could not generate dependence plot for {feat}: {e}")
            plt.close()

    return saved_paths

