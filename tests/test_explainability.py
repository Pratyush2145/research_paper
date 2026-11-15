"""Automated unit tests for Phase 5 Explainable AI using SHAP.

Verifies:
1. PhishingTreeExplainer initialization and orientation toward Class 0 (Phishing).
2. Random Forest 3D explanation slicing to Class 0 probability space.
3. XGBoost 2D raw margin log-odds inversion to Class 0 log-odds space.
4. Conformance of explanation feature names to the 68-feature schema.
5. Reproducible sampling integrity and zero domain leakage across splits.
6. Local case study extraction and classification status.
7. Generalization comparison metrics (Spearman rho, Jaccard overlap).
"""

from __future__ import annotations

import os
from pathlib import Path

# Ensure headless Agg backend and writable cache
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import joblib
import numpy as np
import pandas as pd
import pytest
import shap

from src.explainability.data_sampler import get_stratified_evaluation_samples
from src.explainability.explainer import PhishingTreeExplainer
from src.explainability.generalization_explainability import compare_internal_vs_external_explanations
from src.explainability.local_explanations import find_representative_case_indices

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def loaded_models():
    """Load trained models and feature schema."""
    rf = joblib.load(BASE_DIR / "models/trained/random_forest.joblib")
    xgb = joblib.load(BASE_DIR / "models/trained/xgboost.joblib")
    import json
    with open(BASE_DIR / "models/preprocessing/feature_names.json") as f:
        fnames = json.load(f)
    return rf, xgb, fnames


def test_stratified_data_sampler():
    """Test data sampler for zero domain leakage and exact sample counts."""
    (
        (feat_bg, meta_bg),
        (feat_int, meta_int),
        (feat_ext, meta_ext),
        fnames,
        meta,
    ) = get_stratified_evaluation_samples(BASE_DIR, n_background=100, n_internal=100, n_external=100, random_state=42)

    assert len(feat_bg) == 100
    assert len(feat_int) == 100
    assert len(feat_ext) == 100
    assert len(fnames) == 68

    # Zero domain leakage across the three sampled sets
    dom_bg = set(meta_bg["registrable_domain"].dropna())
    dom_int = set(meta_int["registrable_domain"].dropna())
    dom_ext = set(meta_ext["registrable_domain"].dropna())

    assert len(dom_bg.intersection(dom_int)) == 0
    assert len(dom_bg.intersection(dom_ext)) == 0
    assert len(dom_int.intersection(dom_ext)) == 0


def test_random_forest_tree_explainer_class0_orientation(loaded_models):
    """Test that Random Forest SHAP explanation targets Class 0 (Phishing) probability."""
    rf, _, fnames = loaded_models

    explainer = PhishingTreeExplainer(rf, "Random Forest", feature_names=fnames)

    # Synthetic test input
    X_test = np.zeros((3, 68), dtype=np.float32)
    exp = explainer.explain(X_test)

    assert isinstance(exp, shap.Explanation)
    assert exp.values.shape == (3, 68)
    assert exp.base_values.shape == (3,)
    assert exp.feature_names == fnames

    # Base values should match training prior for Phishing (Class 0: ~0.637)
    np.testing.assert_allclose(exp.base_values, 0.6373, atol=0.01)

    # Feature importance table
    df_rank = explainer.get_feature_importance_table(exp, top_k=10)
    assert len(df_rank) == 10
    assert "mean_abs_shap" in df_rank.columns
    assert "effect_direction" in df_rank.columns


def test_xgboost_tree_explainer_class0_orientation(loaded_models):
    """Test that XGBoost SHAP explanation inverts margin to target Class 0 (Phishing) log-odds."""
    _, xgb, fnames = loaded_models

    explainer = PhishingTreeExplainer(xgb, "XGBoost", feature_names=fnames)

    X_test = np.zeros((3, 68), dtype=np.float32)
    exp = explainer.explain(X_test)

    assert isinstance(exp, shap.Explanation)
    assert exp.values.shape == (3, 68)
    assert exp.base_values.shape == (3,)
    assert exp.feature_names == fnames

    # Check that positive SHAP values increase Phishing log-odds
    # For a synthetic vector of high path_length, SHAP value for path_length should be positive
    X_phish_like = np.zeros((1, 68), dtype=np.float32)
    path_len_idx = fnames.index("path_length")
    X_phish_like[0, path_len_idx] = 50.0  # high path length characteristic of phishing

    exp_phish = explainer.explain(X_phish_like)
    # path_length should positively contribute to phishing log-odds
    assert exp_phish.values[0, path_len_idx] > 0.0


def test_representative_case_indices():
    """Test finding representative case study indices."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0])  # TP(0), TN(1), FN(2), FP(3), TP(4)
    y_prob = np.array([0.98, 0.02, 0.35, 0.85, 0.52])

    cases = find_representative_case_indices(y_true, y_pred, y_prob)

    assert cases["true_positive"] in [0, 4]
    assert cases["true_negative"] == 1
    assert cases["false_negative"] == 2
    assert cases["false_positive"] == 3
    assert cases["high_confidence_phishing"] == 0
    assert cases["low_confidence_borderline"] == 4  # 0.52 is closest to 0.50


def test_generalization_comparison_metrics(loaded_models, tmp_path):
    """Test generalization comparison module between synthetic internal and external explanations."""
    rf, _, fnames = loaded_models

    explainer = PhishingTreeExplainer(rf, "Random Forest", feature_names=fnames)

    X1 = np.random.randn(20, 68).astype(np.float32)
    X2 = np.random.randn(20, 68).astype(np.float32)

    exp1 = explainer.explain(X1)
    exp2 = explainer.explain(X2)

    metrics = compare_internal_vs_external_explanations(
        exp1, exp2, "Random Forest",
        output_fig_dir=tmp_path / "figs",
        output_tab_dir=tmp_path / "tabs",
    )

    assert "spearman_rank_correlation" in metrics
    assert "kendall_tau" in metrics
    assert "top10_jaccard_overlap" in metrics
    assert 0.0 <= metrics["top10_jaccard_overlap"] <= 1.0
    assert (tmp_path / "tabs/random_forest_generalization_feature_comparison.csv").exists()
    assert (tmp_path / "figs/random_forest_generalization_top15_internal_vs_external.png").exists()


def test_shap_exact_additivity_and_probability_reconstruction(loaded_models):
    """Verify exact SHAP additivity and probability reconstruction for both RF and XGBoost."""
    rf, xgb, fnames = loaded_models

    rng = np.random.RandomState(42)
    X_sample = rng.randn(10, 68).astype(np.float32)

    # 1. Random Forest Additivity Check (Probability space)
    rf_explainer = PhishingTreeExplainer(rf, "Random Forest", feature_names=fnames)
    rf_exp = rf_explainer.explain(X_sample)
    rf_pred_proba_class0 = rf.predict_proba(X_sample)[:, 0]

    rf_recon = rf_exp.base_values + np.sum(rf_exp.values, axis=1)
    np.testing.assert_allclose(rf_recon, rf_pred_proba_class0, atol=1e-5, err_msg="RF SHAP additivity failed!")

    # 2. XGBoost Additivity Check (Log-odds space & Sigmoid link probability)
    xgb_explainer = PhishingTreeExplainer(xgb, "XGBoost", feature_names=fnames)
    xgb_exp = xgb_explainer.explain(X_sample)
    xgb_pred_proba_class0 = xgb.predict_proba(X_sample)[:, 0]

    # Inverted log-odds reconstruction
    xgb_recon_log_odds = xgb_exp.base_values + np.sum(xgb_exp.values, axis=1)
    sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
    xgb_recon_prob_class0 = sigmoid(xgb_recon_log_odds)

    np.testing.assert_allclose(xgb_recon_prob_class0, xgb_pred_proba_class0, atol=1e-5, err_msg="XGBoost SHAP probability reconstruction failed!")


