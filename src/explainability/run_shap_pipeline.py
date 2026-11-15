"""Master execution pipeline for Phase 5 Explainable AI using SHAP.

Orchestrates:
1. Reproducible stratified sampling of background and evaluation datasets.
2. Global SHAP analysis (beeswarm, mean |SHAP| bars, ranking tables) for Random Forest and XGBoost.
3. Native model importance vs. SHAP importance comparison.
4. Local explainability (waterfall plots, case audits for TP, TN, FN, FP, high/low confidence).
5. Non-linear dependence and interaction analysis on top structural/lexical features.
6. Generalization-aware explainability comparing internal vs. external feature attributions.
7. Model architecture attribution comparison (Random Forest bagging vs. XGBoost boosting).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.explainability.data_sampler import get_stratified_evaluation_samples
from src.explainability.dependence_analysis import generate_dependence_plots
from src.explainability.explainer import PhishingTreeExplainer
from src.explainability.generalization_explainability import compare_internal_vs_external_explanations
from src.explainability.global_explanations import (
    analyze_class_specific_contributions,
    generate_global_plots_and_tables,
)
from src.explainability.local_explanations import generate_local_explanations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_phase5_shap_pipeline(
    base_dir: str | Path = ".",
    n_samples: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end Phase 5 explainability pipeline."""
    base_path = Path(base_dir)
    fig_dir = base_path / "reports/figures/shap"
    tab_dir = base_path / "reports/tables/shap"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    t0_total = time.time()
    logger.info("Starting Phase 5 SHAP Explainability Pipeline...")

    # 1. Load Trained Models
    rf_path = base_path / "models/trained/random_forest.joblib"
    xgb_path = base_path / "models/trained/xgboost.joblib"

    logger.info("Loading pre-trained Random Forest and XGBoost models...")
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    # 2. Extract Stratified Samples
    logger.info(f"Extracting stratified samples (N={n_samples}, Seed={random_state})...")
    (
        (feat_bg, meta_bg),
        (feat_int, meta_int),
        (feat_ext, meta_ext),
        fnames,
        sampling_meta,
    ) = get_stratified_evaluation_samples(
        base_dir=base_path,
        n_background=n_samples,
        n_internal=n_samples,
        n_external=n_samples,
        random_state=random_state,
    )

    with open(tab_dir / "sampling_metadata.json", "w") as f:
        json.dump(sampling_meta, f, indent=2)

    X_int = feat_int[fnames].to_numpy(dtype=np.float32)
    y_int = feat_int["label"].to_numpy(dtype=np.int32)

    X_ext = feat_ext[fnames].to_numpy(dtype=np.float32)
    y_ext = feat_ext["label"].to_numpy(dtype=np.int32)

    pipeline_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": n_samples,
            "random_state": random_state,
            "sampling_metadata": sampling_meta,
        },
        "models": {},
        "cross_model_comparison": {},
    }

    models_to_analyze = [
        ("Random Forest", rf_model),
        ("XGBoost", xgb_model),
    ]

    explanations_cache = {}

    for model_name, model_obj in models_to_analyze:
        logger.info(f"\n==========================================")
        logger.info(f"Running SHAP Analysis for {model_name}")
        logger.info(f"==========================================")

        explainer = PhishingTreeExplainer(model_obj, model_name, feature_names=fnames)

        # A. Internal Test Explanations
        logger.info(f"[{model_name}] Computing SHAP values on Internal Test set...")
        t0_exp = time.time()
        exp_int = explainer.explain(X_int)
        logger.info(f"[{model_name}] Internal SHAP computed in {time.time() - t0_exp:.2f}s")

        # B. External Disjoint Explanations
        logger.info(f"[{model_name}] Computing SHAP values on External Disjoint set...")
        t0_exp = time.time()
        exp_ext = explainer.explain(X_ext)
        logger.info(f"[{model_name}] External SHAP computed in {time.time() - t0_exp:.2f}s")

        explanations_cache[f"{model_name}_internal"] = exp_int
        explanations_cache[f"{model_name}_external"] = exp_ext

        # C. Global Plots & Tables (Internal & External)
        logger.info(f"[{model_name}] Generating global beeswarm, bar plots, and ranking tables...")
        global_int = generate_global_plots_and_tables(
            explainer, exp_int, "legitphish_internal", fig_dir, tab_dir, top_k=20
        )
        global_ext = generate_global_plots_and_tables(
            explainer, exp_ext, "phiusil_external", fig_dir, tab_dir, top_k=20
        )

        # D. Class-Specific Contribution Analysis
        logger.info(f"[{model_name}] Analyzing class-specific feature divergence...")
        class_div_int = analyze_class_specific_contributions(exp_int, y_int, fnames, top_k=15)
        prefix = model_name.lower().replace(" ", "_")
        class_div_int.to_csv(tab_dir / f"{prefix}_class_divergence_internal.csv", index=False)

        class_div_ext = analyze_class_specific_contributions(exp_ext, y_ext, fnames, top_k=15)
        class_div_ext.to_csv(tab_dir / f"{prefix}_class_divergence_external.csv", index=False)

        # E. Local Case Explanations (Internal & External)
        logger.info(f"[{model_name}] Generating local case studies (waterfall plots)...")
        y_prob_int_phish = model_obj.predict_proba(X_int)[:, 0]
        y_pred_int = model_obj.predict(X_int)
        cases_int = generate_local_explanations(
            exp_int, meta_int, y_int, y_pred_int, y_prob_int_phish,
            model_name, "legitphish_internal", fig_dir / "local", tab_dir
        )

        y_prob_ext_phish = model_obj.predict_proba(X_ext)[:, 0]
        y_pred_ext = model_obj.predict(X_ext)
        cases_ext = generate_local_explanations(
            exp_ext, meta_ext, y_ext, y_pred_ext, y_prob_ext_phish,
            model_name, "phiusil_external", fig_dir / "local", tab_dir
        )

        # F. Feature Dependence Plots
        logger.info(f"[{model_name}] Generating feature dependence plots...")
        dep_plots_int = generate_dependence_plots(
            exp_int, model_name, "legitphish_internal", output_dir=fig_dir / "dependence"
        )
        dep_plots_ext = generate_dependence_plots(
            exp_ext, model_name, "phiusil_external", output_dir=fig_dir / "dependence"
        )

        # G. Generalization-Aware Comparison (Internal vs. External)
        logger.info(f"[{model_name}] Comparing internal vs. external feature attribution consistency...")
        gen_metrics = compare_internal_vs_external_explanations(
            exp_int, exp_ext, model_name, fig_dir, tab_dir, top_k=20
        )

        pipeline_results["models"][model_name] = {
            "global_internal": global_int,
            "global_external": global_ext,
            "generalization_metrics": gen_metrics,
            "local_cases_internal_count": len(cases_int),
            "local_cases_external_count": len(cases_ext),
        }

    # 3. Cross-Model Attribution Comparison (Random Forest vs. XGBoost)
    logger.info("\nEvaluating Cross-Model Attribution Consistency (Random Forest vs. XGBoost)...")
    exp_rf_int = explanations_cache["Random Forest_internal"]
    exp_xgb_int = explanations_cache["XGBoost_internal"]

    rf_abs_int = np.mean(np.abs(exp_rf_int.values), axis=0)
    xgb_abs_int = np.mean(np.abs(exp_xgb_int.values), axis=0)

    df_model_comp = pd.DataFrame({
        "feature": fnames,
        "rf_mean_abs_shap": np.round(rf_abs_int, 6),
        "xgb_mean_abs_shap": np.round(xgb_abs_int, 6),
    })
    df_model_comp["rf_rank"] = df_model_comp["rf_mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    df_model_comp["xgb_rank"] = df_model_comp["xgb_mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    df_model_comp["rank_diff"] = df_model_comp["xgb_rank"] - df_model_comp["rf_rank"]

    rho_models, p_models = spearmanr(df_model_comp["rf_rank"], df_model_comp["xgb_rank"])
    df_model_comp = df_model_comp.sort_values(by="rf_rank").reset_index(drop=True)
    df_model_comp.to_csv(tab_dir / "random_forest_vs_xgboost_internal_shap.csv", index=False)

    pipeline_results["cross_model_comparison"] = {
        "internal_spearman_rank_correlation": round(float(rho_models), 4),
        "internal_spearman_p_value": float(p_models),
        "table_csv": str(tab_dir / "random_forest_vs_xgboost_internal_shap.csv"),
    }

    # Save master execution summary
    summary_path = tab_dir / "shap_execution_summary.json"
    with open(summary_path, "w") as f:
        # Exclude large DataFrames
        clean_results = {k: v for k, v in pipeline_results.items() if k != "models"}
        clean_results["models"] = {}
        for m_name, m_data in pipeline_results["models"].items():
            clean_results["models"][m_name] = {
                "generalization_metrics": m_data["generalization_metrics"],
                "local_cases_internal_count": m_data["local_cases_internal_count"],
                "local_cases_external_count": m_data["local_cases_external_count"],
            }
        json.dump(clean_results, f, indent=2)

    logger.info(f"\nPhase 5 SHAP Pipeline completed in {time.time() - t0_total:.2f}s.")
    logger.info(f"Artifacts saved in {fig_dir} and {tab_dir}.")
    return pipeline_results


if __name__ == "__main__":
    run_phase5_shap_pipeline()

