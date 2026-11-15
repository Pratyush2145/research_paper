"""End-to-end model training, evaluation, and serialization pipeline.

Implements strict leakage-safe partitioning, feature scaling on training data only,
model training for 5 candidate algorithms, and comprehensive evaluation across
internal validation, internal test, deduplicated test, and external datasets.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import compute_evaluation_metrics
from src.models.dummy_model import build_dummy
from src.models.logistic_regression import build_logistic_regression
from src.models.random_forest import build_random_forest
from src.models.svm_model import build_svm
from src.models.xgboost_model import build_xgboost

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_deterministic_domain_splits(
    meta_df: pd.DataFrame,
    random_state: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Partition dataset by registrable domain with zero domain leakage.

    Args:
        meta_df: Metadata dataframe containing 'registrable_domain'.
        random_state: Random seed for deterministic shuffling.
        train_ratio: Fraction of unique domains assigned to training.
        val_ratio: Fraction of unique domains assigned to validation.

    Returns:
        Tuple of (train_indices, val_indices, test_indices, split_info_dict).
    """
    domains = meta_df["registrable_domain"].unique()
    rng = np.random.RandomState(random_state)
    shuffled_domains = rng.permutation(domains)

    n_doms = len(shuffled_domains)
    n_train_doms = int(train_ratio * n_doms)
    n_val_doms = int(val_ratio * n_doms)

    train_doms = set(shuffled_domains[:n_train_doms])
    val_doms = set(shuffled_domains[n_train_doms : n_train_doms + n_val_doms])
    test_doms = set(shuffled_domains[n_train_doms + n_val_doms :])

    # Ensure zero domain overlap
    assert len(train_doms.intersection(val_doms)) == 0, "Train-Val domain overlap detected!"
    assert len(train_doms.intersection(test_doms)) == 0, "Train-Test domain overlap detected!"
    assert len(val_doms.intersection(test_doms)) == 0, "Val-Test domain overlap detected!"

    train_mask = meta_df["registrable_domain"].isin(train_doms)
    val_mask = meta_df["registrable_domain"].isin(val_doms)
    test_mask = meta_df["registrable_domain"].isin(test_doms)

    train_idx = meta_df.index[train_mask].to_numpy()
    val_idx = meta_df.index[val_mask].to_numpy()
    test_idx = meta_df.index[test_mask].to_numpy()

    # Verify complete coverage
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(meta_df), "Split indices mismatch total rows!"

    split_info = {
        "random_state": random_state,
        "n_total_domains": int(n_doms),
        "train": {
            "n_domains": int(len(train_doms)),
            "n_rows": int(len(train_idx)),
            "row_percentage": round(len(train_idx) / len(meta_df) * 100, 2),
        },
        "val": {
            "n_domains": int(len(val_doms)),
            "n_rows": int(len(val_idx)),
            "row_percentage": round(len(val_idx) / len(meta_df) * 100, 2),
        },
        "test": {
            "n_domains": int(len(test_doms)),
            "n_rows": int(len(test_idx)),
            "row_percentage": round(len(test_idx) / len(meta_df) * 100, 2),
        },
        "train_domains_list": sorted([str(d) for d in train_doms if pd.notna(d)]),
        "all_legitphish_domains_list": sorted([str(d) for d in domains if pd.notna(d)]),
    }

    return train_idx, val_idx, test_idx, split_info


def load_datasets(
    base_dir: str | Path = ".",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Load processed feature matrices and metadata for LegitPhish and PhiUSIIL.

    Returns:
        Tuple of (feat_legit, meta_legit, feat_phi, meta_phi, feature_names).
    """
    base_path = Path(base_dir)
    legit_feat_path = base_path / "data/processed/legitphish/url_features.csv"
    legit_meta_path = base_path / "data/processed/legitphish/metadata.csv"
    phi_feat_path = base_path / "data/processed/phiusil/url_features.csv"
    phi_meta_path = base_path / "data/processed/phiusil/metadata.csv"

    logger.info("Loading LegitPhish features and metadata...")
    feat_legit = pd.read_csv(legit_feat_path)
    meta_legit = pd.read_csv(legit_meta_path)

    logger.info("Loading PhiUSIIL features and metadata...")
    feat_phi = pd.read_csv(phi_feat_path)
    meta_phi = pd.read_csv(phi_meta_path)

    feature_names = [col for col in feat_legit.columns if col != "label"]
    assert len(feature_names) == 68, f"Expected 68 features, got {len(feature_names)}"
    assert [c for c in feat_phi.columns if c != "label"] == feature_names, "Feature mismatch between datasets!"

    return feat_legit, meta_legit, feat_phi, meta_phi, feature_names


def measure_inference_latency(
    model: Any,
    X_sample: np.ndarray,
    n_iterations: int = 5,
) -> float:
    """Measure average inference latency in milliseconds per 1,000 samples.

    Args:
        model: Fitted estimator.
        X_sample: Input feature matrix sample.
        n_iterations: Number of warmup/timing repetitions.

    Returns:
        Latency in milliseconds per 1,000 samples.
    """
    n_samples = len(X_sample)
    if n_samples == 0:
        return 0.0

    # Warmup
    _ = model.predict(X_sample[: min(100, n_samples)])

    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = model.predict(X_sample)
        times.append(time.perf_counter() - t0)

    avg_seconds = np.mean(times)
    ms_per_1000 = (avg_seconds / n_samples) * 1000 * 1000
    return round(float(ms_per_1000), 2)


def run_pipeline(
    base_dir: str | Path = ".",
    output_dir: str | Path = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute complete Phase 4 model training, evaluation, and artifact serialization.

    Returns:
        Structured dictionary of all results and evaluation metrics.
    """
    base_path = Path(base_dir)
    out_path = Path(output_dir)
    trained_dir = out_path / "trained"
    preproc_dir = out_path / "preprocessing"
    eval_dir = out_path / "evaluation"

    trained_dir.mkdir(parents=True, exist_ok=True)
    preproc_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    feat_legit, meta_legit, feat_phi, meta_phi, feature_names = load_datasets(base_path)

    # 2. Deterministic domain-isolated split
    logger.info("Generating deterministic domain-level splits (70/15/15)...")
    train_idx, val_idx, test_idx, split_info = create_deterministic_domain_splits(
        meta_legit, random_state=random_state
    )

    # Save feature names and split metadata
    with open(preproc_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(preproc_dir / "splits.json", "w") as f:
        # Exclude large lists from high-level splits summary for readability
        summary_splits = {k: v for k, v in split_info.items() if not k.endswith("_list")}
        json.dump(summary_splits, f, indent=2)

    # Extract internal matrices
    X = feat_legit[feature_names].to_numpy(dtype=np.float32)
    y = feat_legit["label"].to_numpy(dtype=np.int32)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logger.info(
        f"Internal Split: Train={len(y_train)} (Phish: {(y_train == 0).sum()}, Legit: {(y_train == 1).sum()}), "
        f"Val={len(y_val)} (Phish: {(y_val == 0).sum()}, Legit: {(y_val == 1).sum()}), "
        f"Test={len(y_test)} (Phish: {(y_test == 0).sum()}, Legit: {(y_test == 1).sum()})"
    )

    # Deduplicated Test Set
    test_meta = meta_legit.iloc[test_idx]
    dedup_test_mask = ~test_meta.duplicated(subset=["raw_url"])
    dedup_test_idx_local = np.where(dedup_test_mask)[0]
    X_test_dedup = X_test[dedup_test_idx_local]
    y_test_dedup = y_test[dedup_test_idx_local]
    logger.info(f"Deduplicated Test Set: {len(y_test_dedup)} records ({len(y_test) - len(y_test_dedup)} duplicates dropped)")

    # 3. Fit StandardScaler strictly on training set
    logger.info("Fitting StandardScaler strictly on training set...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, preproc_dir / "scaler.joblib")

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_test_dedup_scaled = scaler.transform(X_test_dedup)

    # 4. External Datasets (PhiUSIIL)
    logger.info("Preparing external validation datasets...")
    X_phi_full = feat_phi[feature_names].to_numpy(dtype=np.float32)
    y_phi_full = feat_phi["label"].to_numpy(dtype=np.int32)
    X_phi_full_scaled = scaler.transform(X_phi_full)

    # Training-Domain-Disjoint External Set
    train_dom_set = set(split_info["train_domains_list"])
    phi_train_disjoint_mask = ~meta_phi["registrable_domain"].isin(train_dom_set).to_numpy()
    X_phi_train_disjoint = X_phi_full[phi_train_disjoint_mask]
    y_phi_train_disjoint = y_phi_full[phi_train_disjoint_mask]
    X_phi_train_disjoint_scaled = X_phi_full_scaled[phi_train_disjoint_mask]

    # Entire-LegitPhish-Disjoint External Set
    all_legit_dom_set = set(split_info["all_legitphish_domains_list"])
    phi_all_disjoint_mask = ~meta_phi["registrable_domain"].isin(all_legit_dom_set).to_numpy()
    X_phi_all_disjoint = X_phi_full[phi_all_disjoint_mask]
    y_phi_all_disjoint = y_phi_full[phi_all_disjoint_mask]
    X_phi_all_disjoint_scaled = X_phi_full_scaled[phi_all_disjoint_mask]

    logger.info(
        f"External Datasets: Standard={len(y_phi_full)}, "
        f"Train-Disjoint={len(y_phi_train_disjoint)} ({len(y_phi_full) - len(y_phi_train_disjoint)} purged), "
        f"Entire-Disjoint={len(y_phi_all_disjoint)} ({len(y_phi_full) - len(y_phi_all_disjoint)} purged)"
    )

    # 5. Define Evaluation Sets dictionary
    # Tuple of (unscaled_X, scaled_X, y, description)
    eval_sets = {
        "val": (X_val, X_val_scaled, y_val, "Internal Validation Set (15% domains)"),
        "test": (X_test, X_test_scaled, y_test, "Internal Test Set (15% domains)"),
        "test_dedup": (X_test_dedup, X_test_dedup_scaled, y_test_dedup, "Internal Test Deduplicated (unique URLs)"),
        "external_standard": (X_phi_full, X_phi_full_scaled, y_phi_full, "PhiUSIIL Standard External Set (all records)"),
        "external_train_disjoint": (
            X_phi_train_disjoint,
            X_phi_train_disjoint_scaled,
            y_phi_train_disjoint,
            "PhiUSIIL Training-Domain-Disjoint External Set",
        ),
        "external_entire_disjoint": (
            X_phi_all_disjoint,
            X_phi_all_disjoint_scaled,
            y_phi_all_disjoint,
            "PhiUSIIL Entire-LegitPhish-Disjoint External Set",
        ),
    }

    # 6. Initialize Models
    # Model definition: (name, builder_func, requires_scaling, filename)
    model_defs = [
        ("Dummy (Majority)", build_dummy, False, "dummy.joblib"),
        ("Logistic Regression", build_logistic_regression, True, "logistic_regression.joblib"),
        ("Support Vector Machine (Linear Calibrated)", build_svm, True, "svm.joblib"),
        ("Random Forest", build_random_forest, False, "random_forest.joblib"),
        ("XGBoost", build_xgboost, False, "xgboost.joblib"),
    ]

    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_features": len(feature_names),
            "split_info": summary_splits,
            "dataset_sizes": {
                "train": len(y_train),
                "val": len(y_val),
                "test": len(y_test),
                "test_dedup": len(y_test_dedup),
                "external_standard": len(y_phi_full),
                "external_train_disjoint": len(y_phi_train_disjoint),
                "external_entire_disjoint": len(y_phi_all_disjoint),
            },
        },
        "models": {},
    }

    summary_rows = []

    # 7. Train and Evaluate each model
    for model_name, builder_fn, use_scaled, filename in model_defs:
        logger.info(f"--- Training {model_name} ---")
        model = builder_fn()

        # Training input
        X_tr = X_train_scaled if use_scaled else X_train

        t0_train = time.perf_counter()
        model.fit(X_tr, y_train)
        train_time = round(time.perf_counter() - t0_train, 3)
        logger.info(f"{model_name} trained in {train_time:.3f}s")

        # Save model
        joblib.dump(model, trained_dir / filename)

        # Measure latency on test set
        X_test_eval = X_test_scaled if use_scaled else X_test
        latency_ms_per_1000 = measure_inference_latency(model, X_test_eval)

        model_results = {
            "name": model_name,
            "requires_scaling": use_scaled,
            "artifact": filename,
            "train_time_sec": train_time,
            "latency_ms_per_1000": latency_ms_per_1000,
            "evaluations": {},
        }

        # Evaluate across all partitions
        for set_key, (X_unscaled, X_scaled_set, y_target, desc) in eval_sets.items():
            X_eval = X_scaled_set if use_scaled else X_unscaled

            y_pred = model.predict(X_eval)

            # Probabilities for Phishing (class 0)
            y_prob_phish = None
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X_eval)
                y_prob_phish = probas[:, 0]

            metrics = compute_evaluation_metrics(y_target, y_pred, y_prob_phish)
            model_results["evaluations"][set_key] = {
                "description": desc,
                "metrics": metrics,
            }

            summary_rows.append({
                "model": model_name,
                "evaluation_set": set_key,
                "train_time_sec": train_time,
                "latency_ms_per_1000": latency_ms_per_1000,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "matthews_corrcoef": metrics["matthews_corrcoef"],
                "specificity": metrics["specificity"],
                "fpr": metrics["false_positive_rate"],
                "fnr": metrics["false_negative_rate"],
                "support_phishing": metrics["support"]["phishing_count"],
                "support_legitimate": metrics["support"]["legitimate_count"],
            })

            logger.info(
                f"[{model_name}] {set_key}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']}, Recall={metrics['recall']:.4f}, Spec={metrics['specificity']:.4f}"
            )

        results["models"][model_name] = model_results

    # 8. Save results and summary table
    with open(eval_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(eval_dir / "model_metrics_summary.csv", index=False)

    logger.info(f"Phase 4 pipeline completed successfully. Artifacts saved in {out_path}.")
    return results


if __name__ == "__main__":
    run_pipeline()

