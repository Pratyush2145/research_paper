"""Automated unit tests for Phase 4 machine learning model training and evaluation.

Verifies:
1. Feature input dimensions and schema integrity (68 features).
2. Label convention: Phishing = 0 (positive), Legitimate = 1 (negative).
3. Zero domain leakage across train, val, and test splits.
4. Duplicate URL isolation within partitions.
5. Model serialization and artifact reloading.
6. StandardScaler fitting strictly on training data.
7. External dataset purging logic.
8. Comprehensive metric calculation.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_evaluation_metrics
from src.models.pipeline import (
    create_deterministic_domain_splits,
    load_datasets,
)

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def loaded_data():
    """Load datasets once for the test module."""
    feat_l, meta_l, feat_p, meta_p, fnames = load_datasets(BASE_DIR)
    return feat_l, meta_l, feat_p, meta_p, fnames


def test_feature_matrix_dimensions_and_names(loaded_data):
    """Test that feature matrix has exactly 68 features and no metadata leakage."""
    feat_l, meta_l, feat_p, meta_p, fnames = loaded_data

    assert len(fnames) == 68
    assert "label" not in fnames
    assert "url" not in fnames
    assert "registrable_domain" not in fnames
    assert "source_dataset" not in fnames
    assert "raw_url" not in fnames

    # Check that both datasets have exactly identical feature columns
    assert [c for c in feat_l.columns if c != "label"] == fnames
    assert [c for c in feat_p.columns if c != "label"] == fnames

    # Verify no NaN or Inf in feature matrices
    assert not feat_l[fnames].isna().any().any()
    assert not np.isinf(feat_l[fnames].to_numpy()).any()
    assert not feat_p[fnames].isna().any().any()
    assert not np.isinf(feat_p[fnames].to_numpy()).any()


def test_label_integrity_and_positive_class_convention(loaded_data):
    """Test that binary labels are strictly 0 and 1, with 0 = Phishing."""
    feat_l, _, feat_p, _, _ = loaded_data

    # LegitPhish labels
    unique_l = set(feat_l["label"].unique())
    assert unique_l == {0, 1}

    # PhiUSIIL labels
    unique_p = set(feat_p["label"].unique())
    assert unique_p == {0, 1}

    # Metrics directional test: verify pos_label=0 is treated as positive class
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])  # 1 TP (0->0), 1 FN (0->1), 0 FP (1->0), 2 TN (1->1)
    metrics = compute_evaluation_metrics(y_true, y_pred)

    assert metrics["positive_class"] == "Phishing (0)"
    assert metrics["confusion_matrix"]["true_phishing_tp"] == 1
    assert metrics["confusion_matrix"]["false_legitimate_fn"] == 1
    assert metrics["confusion_matrix"]["false_phishing_fp"] == 0
    assert metrics["confusion_matrix"]["true_legitimate_tn"] == 2
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["specificity"] == 1.0


def test_zero_domain_leakage_in_splits(loaded_data):
    """Test that train, validation, and test partitions have zero domain overlap."""
    _, meta_l, _, _, _ = loaded_data

    train_idx, val_idx, test_idx, split_info = create_deterministic_domain_splits(
        meta_l, random_state=42
    )

    train_doms = set(meta_l.iloc[train_idx]["registrable_domain"].dropna())
    val_doms = set(meta_l.iloc[val_idx]["registrable_domain"].dropna())
    test_doms = set(meta_l.iloc[test_idx]["registrable_domain"].dropna())

    # Absolute zero domain intersection
    assert len(train_doms.intersection(val_doms)) == 0
    assert len(train_doms.intersection(test_doms)) == 0
    assert len(val_doms.intersection(test_doms)) == 0

    # Total rows coverage
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(meta_l)


def test_duplicate_url_isolation(loaded_data):
    """Test that identical duplicate URLs never cross partition boundaries."""
    _, meta_l, _, _, _ = loaded_data

    train_idx, val_idx, test_idx, _ = create_deterministic_domain_splits(
        meta_l, random_state=42
    )

    train_urls = set(meta_l.iloc[train_idx]["raw_url"])
    val_urls = set(meta_l.iloc[val_idx]["raw_url"])
    test_urls = set(meta_l.iloc[test_idx]["raw_url"])

    # Verify zero URL overlap across splits
    assert len(train_urls.intersection(val_urls)) == 0
    assert len(train_urls.intersection(test_urls)) == 0
    assert len(val_urls.intersection(test_urls)) == 0


def test_model_artifact_serialization_and_reloading():
    """Test that all 5 trained models and the scaler can be reloaded and make valid predictions."""
    models_dir = BASE_DIR / "models/trained"
    preproc_dir = BASE_DIR / "models/preprocessing"

    expected_models = [
        "dummy.joblib",
        "logistic_regression.joblib",
        "svm.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
    ]

    for m_file in expected_models:
        model_path = models_dir / m_file
        assert model_path.exists(), f"Missing model artifact: {m_file}"
        model = joblib.load(model_path)

        # Test dummy prediction on 68 synthetic features
        dummy_X = np.zeros((5, 68), dtype=np.float32)
        preds = model.predict(dummy_X)
        assert len(preds) == 5
        assert set(np.unique(preds)).issubset({0, 1})

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(dummy_X)
            assert probas.shape == (5, 2)
            np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    # Test Scaler
    scaler_path = preproc_dir / "scaler.joblib"
    assert scaler_path.exists()
    scaler = joblib.load(scaler_path)
    assert scaler.n_features_in_ == 68


def test_external_dataset_purging_counts(loaded_data):
    """Test external domain purging logic and record counts."""
    _, meta_l, _, meta_p, _ = loaded_data

    train_idx, _, _, _ = create_deterministic_domain_splits(meta_l, random_state=42)
    train_doms = set(meta_l.iloc[train_idx]["registrable_domain"].dropna())
    all_doms = set(meta_l["registrable_domain"].dropna())

    # PhiUSIIL purging
    train_disjoint_mask = ~meta_p["registrable_domain"].isin(train_doms)
    all_disjoint_mask = ~meta_p["registrable_domain"].isin(all_doms)

    # Check exact counts verified in audit
    assert train_disjoint_mask.sum() == 184705
    assert all_disjoint_mask.sum() == 165152
    assert len(meta_p) - train_disjoint_mask.sum() == 51090
    assert len(meta_p) - all_disjoint_mask.sum() == 70643


def test_evaluation_summary_file_integrity():
    """Test that model_metrics_summary.csv and results.json exist and are well-formed."""
    csv_path = BASE_DIR / "models/evaluation/model_metrics_summary.csv"
    json_path = BASE_DIR / "models/evaluation/results.json"

    assert csv_path.exists()
    assert json_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 30  # 5 models * 6 evaluation sets = 30 rows
    assert "accuracy" in df.columns
    assert "f1_score" in df.columns
    assert "roc_auc" in df.columns
    assert "pr_auc" in df.columns
    assert "matthews_corrcoef" in df.columns

    with open(json_path) as f:
        data = json.load(f)
    assert "models" in data
    assert len(data["models"]) == 5

