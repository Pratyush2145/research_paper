"""Pipeline runner for Phase 6 Adversarial Robustness and Diagnostic Testing.

Executes:
- Stratified, domain-aware benchmark sampling (Internal Test N=2000, External Disjoint N=2000, Seed 42).
- Evaluation of all registered attack categories (Tier A, Tier B, Tier C) on Random Forest and XGBoost.
- Applicability tracking (N_cand, N_app, N_tx, Ar, Tr).
- Paired contingency evaluation and closed-form metrics (ASR_cond, ER_overall, FPSR, PFR, Delta Recall, Delta F1).
- 1,000-iteration paired bootstrap 95% confidence intervals.
- Edwards continuity-corrected McNemar tests with exact binomial fallbacks.
- Benjamini-Hochberg FDR and Bonferroni FWER multiple comparison corrections.
- Five specialized diagnostic robustness audits (Canonicalization, Unicode/IDN, Perturbation Scans,
  Calibration Drift, and OOD Structural Boundaries).
- Comprehensive tables, JSON summaries, and publication-ready visualization figures.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Matplotlib headless config
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.adversarial.attacks import ATTACK_REGISTRY
from src.adversarial.evaluator import (
    AdversarialEvaluator,
    apply_multiple_testing_correction,
    audit_canonicalization_sensitivity,
    audit_feature_perturbation_scans,
    audit_ood_structural_boundaries,
    audit_unicode_normalization_sensitivity,
    compute_attack_metrics,
    compute_calibration_drift,
    mcnemar_test,
    paired_bootstrap_ci,
)
from src.features.feature_extractor import extract_url_features
from src.models.pipeline import create_deterministic_domain_splits, load_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sample_stratified_domain_benchmark(
    df_meta: pd.DataFrame,
    df_feat: pd.DataFrame,
    n_per_class: int = 1000,
    random_state: int = 42,
) -> Tuple[List[str], List[int], List[str], pd.DataFrame]:
    """Sample stratified, domain-diverse benchmark URLs with balanced class support.

    Returns:
        (urls, labels, domains, sample_df)
    """
    rng = np.random.RandomState(random_state)

    phish_mask = (df_feat["label"] == 0)
    legit_mask = (df_feat["label"] == 1)

    phish_meta = df_meta[phish_mask].copy()
    legit_meta = df_meta[legit_mask].copy()

    # Prioritize 1 sample per unique domain to ensure domain diversity
    def _sample_from_meta(meta_subset: pd.DataFrame, n_target: int) -> pd.DataFrame:
        shuffled_meta = meta_subset.sample(frac=1.0, random_state=random_state)
        # Unique domain selection first
        dedup_dom = shuffled_meta.drop_duplicates(subset=["registrable_domain"])
        if len(dedup_dom) >= n_target:
            return dedup_dom.iloc[:n_target]
        # If fewer domains than target, sample remaining from rest
        remaining_needed = n_target - len(dedup_dom)
        rest = shuffled_meta[~shuffled_meta.index.isin(dedup_dom.index)]
        rest_sample = rest.sample(n=min(remaining_needed, len(rest)), random_state=random_state)
        return pd.concat([dedup_dom, rest_sample]).iloc[:n_target]

    sampled_phish = _sample_from_meta(phish_meta, n_per_class)
    sampled_legit = _sample_from_meta(legit_meta, n_per_class)

    combined = pd.concat([sampled_phish, sampled_legit]).sample(frac=1.0, random_state=random_state)

    urls = combined["raw_url"].tolist()
    labels = [int(df_feat.loc[idx, "label"]) for idx in combined.index]
    domains = combined["registrable_domain"].tolist()

    return urls, labels, domains, combined


def run_partition_adversarial_evaluation(
    evaluator: AdversarialEvaluator,
    urls: List[str],
    labels: List[int],
    domains: List[str],
    partition_name: str,
    categories: List[str],
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute complete adversarial evaluation across all attack categories for a partition."""
    logger.info(f"Starting adversarial evaluation for partition: '{partition_name}' (N={len(urls)})...")

    results_rf: Dict[str, Any] = {}
    results_xgb: Dict[str, Any] = {}
    applicability_table: List[Dict[str, Any]] = []
    all_rf_records = []
    all_xgb_records = []

    for cat_id in categories:
        logger.info(f"Evaluating {cat_id} on {partition_name}...")
        rf_recs, xgb_recs, app_stats = evaluator.evaluate_url_batch(
            urls, labels, domains, cat_id, partition_name
        )
        applicability_table.append(app_stats)
        all_rf_records.extend(rf_recs)
        all_xgb_records.extend(xgb_recs)

        # 1. Random Forest evaluation
        rf_metrics = compute_attack_metrics(rf_recs)
        rf_ci_asr = paired_bootstrap_ci(rf_recs, "asr_cond", n_resamples=n_bootstrap, random_state=random_state)
        rf_ci_er = paired_bootstrap_ci(rf_recs, "er_overall", n_resamples=n_bootstrap, random_state=random_state)
        rf_ci_fpsr = paired_bootstrap_ci(rf_recs, "fpsr", n_resamples=n_bootstrap, random_state=random_state)
        rf_ci_pfr = paired_bootstrap_ci(rf_recs, "pfr", n_resamples=n_bootstrap, random_state=random_state)
        rf_ci_drec = paired_bootstrap_ci(rf_recs, "delta_recall", n_resamples=n_bootstrap, random_state=random_state)
        rf_mcnemar = mcnemar_test(rf_recs)

        results_rf[cat_id] = {
            "metrics": rf_metrics,
            "ci_asr": rf_ci_asr,
            "ci_er": rf_ci_er,
            "ci_fpsr": rf_ci_fpsr,
            "ci_pfr": rf_ci_pfr,
            "ci_delta_recall": rf_ci_drec,
            "mcnemar": rf_mcnemar,
        }

        # 2. XGBoost evaluation
        xgb_metrics = compute_attack_metrics(xgb_recs)
        xgb_ci_asr = paired_bootstrap_ci(xgb_recs, "asr_cond", n_resamples=n_bootstrap, random_state=random_state)
        xgb_ci_er = paired_bootstrap_ci(xgb_recs, "er_overall", n_resamples=n_bootstrap, random_state=random_state)
        xgb_ci_fpsr = paired_bootstrap_ci(xgb_recs, "fpsr", n_resamples=n_bootstrap, random_state=random_state)
        xgb_ci_pfr = paired_bootstrap_ci(xgb_recs, "pfr", n_resamples=n_bootstrap, random_state=random_state)
        xgb_ci_drec = paired_bootstrap_ci(xgb_recs, "delta_recall", n_resamples=n_bootstrap, random_state=random_state)
        xgb_mcnemar = mcnemar_test(xgb_recs)

        results_xgb[cat_id] = {
            "metrics": xgb_metrics,
            "ci_asr": xgb_ci_asr,
            "ci_er": xgb_ci_er,
            "ci_fpsr": xgb_ci_fpsr,
            "ci_pfr": xgb_ci_pfr,
            "ci_delta_recall": xgb_ci_drec,
            "mcnemar": xgb_mcnemar,
        }

    # Apply multiple testing corrections across categories
    rf_pvals = [results_rf[cid]["mcnemar"]["p_value"] for cid in categories]
    rf_bonf_p, rf_bonf_sig, rf_fdr_p, rf_fdr_sig = apply_multiple_testing_correction(rf_pvals)

    xgb_pvals = [results_xgb[cid]["mcnemar"]["p_value"] for cid in categories]
    xgb_bonf_p, xgb_bonf_sig, xgb_fdr_p, xgb_fdr_sig = apply_multiple_testing_correction(xgb_pvals)

    for idx, cid in enumerate(categories):
        results_rf[cid]["mcnemar"]["p_bonferroni"] = rf_bonf_p[idx]
        results_rf[cid]["mcnemar"]["sig_bonferroni"] = rf_bonf_sig[idx]
        results_rf[cid]["mcnemar"]["p_fdr"] = rf_fdr_p[idx]
        results_rf[cid]["mcnemar"]["sig_fdr"] = rf_fdr_sig[idx]

        results_xgb[cid]["mcnemar"]["p_bonferroni"] = xgb_bonf_p[idx]
        results_xgb[cid]["mcnemar"]["sig_bonferroni"] = xgb_bonf_sig[idx]
        results_xgb[cid]["mcnemar"]["p_fdr"] = xgb_fdr_p[idx]
        results_xgb[cid]["mcnemar"]["sig_fdr"] = xgb_fdr_sig[idx]

    return {
        "partition": partition_name,
        "applicability": applicability_table,
        "rf": results_rf,
        "xgb": results_xgb,
        "rf_records": all_rf_records,
        "xgb_records": all_xgb_records,
    }


def generate_adversarial_plots(
    categories: List[str],
    eval_int: Dict[str, Any],
    eval_ext: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Generate publication-ready visualization figures for adversarial metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    # Plot 1: Recall Degradation across categories (RF vs XGB, Internal vs External)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    rf_rec_int = [eval_int["rf"][cid]["metrics"]["adv_recall"] or 0.0 for cid in categories]
    xgb_rec_int = [eval_int["xgb"][cid]["metrics"]["adv_recall"] or 0.0 for cid in categories]
    rf_rec_ext = [eval_ext["rf"][cid]["metrics"]["adv_recall"] or 0.0 for cid in categories]
    xgb_rec_ext = [eval_ext["xgb"][cid]["metrics"]["adv_recall"] or 0.0 for cid in categories]

    x = np.arange(len(categories))
    width = 0.35

    # Internal Test Subplot
    axes[0].bar(x - width / 2, rf_rec_int, width, label="Random Forest", color="#1f77b4")
    axes[0].bar(x + width / 2, xgb_rec_int, width, label="XGBoost", color="#ff7f0e")
    axes[0].set_title("LegitPhish Internal Test Partition (N=2000)")
    axes[0].set_ylabel("Adversarial Recall (Phishing Class 0)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower left")

    # External Disjoint Subplot
    axes[1].bar(x - width / 2, rf_rec_ext, width, label="Random Forest", color="#1f77b4")
    axes[1].bar(x + width / 2, xgb_rec_ext, width, label="XGBoost", color="#ff7f0e")
    axes[1].set_title("PhiUSIIL Strict External Disjoint Partition (N=2000)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    axes[1].axhline(eval_ext["rf"][categories[0]]["metrics"]["orig_recall"], color="#1f77b4", linestyle="--", alpha=0.5, label="RF Baseline")
    axes[1].axhline(eval_ext["xgb"][categories[0]]["metrics"]["orig_recall"], color="#ff7f0e", linestyle="--", alpha=0.5, label="XGB Baseline")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left")

    fig.suptitle("Adversarial Recall Across Attack Categories", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "recall_degradation.png", dpi=300)
    plt.close(fig)

    # Plot 2: Conditional Attack Success Rate (ASR_cond)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    rf_asr_int = [(eval_int["rf"][cid]["metrics"]["asr_cond"] or 0.0) * 100 for cid in categories]
    xgb_asr_int = [(eval_int["xgb"][cid]["metrics"]["asr_cond"] or 0.0) * 100 for cid in categories]
    rf_asr_ext = [(eval_ext["rf"][cid]["metrics"]["asr_cond"] or 0.0) * 100 for cid in categories]
    xgb_asr_ext = [(eval_ext["xgb"][cid]["metrics"]["asr_cond"] or 0.0) * 100 for cid in categories]

    axes[0].bar(x - width / 2, rf_asr_int, width, label="Random Forest", color="#2ca02c")
    axes[0].bar(x + width / 2, xgb_asr_int, width, label="XGBoost", color="#d62728")
    axes[0].set_title("Internal Test Partition")
    axes[0].set_ylabel("Conditional ASR (%) [Flips from Detected TP]")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left")

    axes[1].bar(x - width / 2, rf_asr_ext, width, label="Random Forest", color="#2ca02c")
    axes[1].bar(x + width / 2, xgb_asr_ext, width, label="XGBoost", color="#d62728")
    axes[1].set_title("External Disjoint Partition")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left")

    fig.suptitle("Conditional Attack Success Rate (ASR) Across Transformations", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "conditional_asr.png", dpi=300)
    plt.close(fig)

    # Plot 3: Prediction Flip Rates (PFR)
    fig, ax = plt.subplots(figsize=(12, 6))
    rf_pfr_ext = [eval_ext["rf"][cid]["metrics"]["pfr"] * 100 for cid in categories]
    xgb_pfr_ext = [eval_ext["xgb"][cid]["metrics"]["pfr"] * 100 for cid in categories]

    ax.bar(x - width / 2, rf_pfr_ext, width, label="Random Forest (External Disjoint)", color="#17becf")
    ax.bar(x + width / 2, xgb_pfr_ext, width, label="XGBoost (External Disjoint)", color="#bcbd22")
    ax.set_ylabel("Prediction Flip Rate (%) [Total Instability]")
    ax.set_title("Model Decision Boundary Fragility (Prediction Flip Rate)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    fig.savefig(out_dir / "prediction_flip_rates.png", dpi=300)
    plt.close(fig)

    logger.info("Adversarial visualization charts generated successfully.")


def run_pipeline() -> None:
    """Execute complete Phase 6 experimental evaluation."""
    logger.info("Initializing Phase 6 Adversarial Evaluation Pipeline...")

    base_dir = Path(".")
    models_dir = base_dir / "models/trained"
    preproc_dir = base_dir / "models/preprocessing"
    tables_dir = base_dir / "reports/tables/adversarial"
    figures_dir = base_dir / "reports/figures/adversarial"
    data_adv_dir = base_dir / "data/adversarial"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_adv_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen models and preprocessors
    logger.info("Loading frozen models and metadata...")
    rf_model = joblib.load(models_dir / "random_forest.joblib")
    xgb_model = joblib.load(models_dir / "xgboost.joblib")
    scaler = joblib.load(preproc_dir / "scaler.joblib")

    with open(preproc_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    with open(preproc_dir / "splits.json") as f:
        splits_meta = json.load(f)

    # 2. Load processed datasets
    feat_legit, meta_legit, feat_phi, meta_phi, _ = load_datasets(base_dir)

    # Recreate deterministic splits for LegitPhish
    train_idx, val_idx, test_idx, split_info = create_deterministic_domain_splits(
        meta_legit, random_state=42
    )

    # Filter to LegitPhish test partition
    test_meta_legit = meta_legit.iloc[test_idx].reset_index(drop=True)
    test_feat_legit = feat_legit.iloc[test_idx].reset_index(drop=True)

    # Filter to PhiUSIIL strict external entire-disjoint partition
    all_legit_dom_set = set(split_info["all_legitphish_domains_list"])
    phi_all_disjoint_mask = ~meta_phi["registrable_domain"].isin(all_legit_dom_set).to_numpy()
    disjoint_meta_phi = meta_phi[phi_all_disjoint_mask].reset_index(drop=True)
    disjoint_feat_phi = feat_phi[phi_all_disjoint_mask].reset_index(drop=True)

    logger.info(
        f"Available Evaluation Partitions: "
        f"LegitPhish Test={len(test_feat_legit)}, PhiUSIIL Disjoint={len(disjoint_feat_phi)}"
    )

    # 3. Create stratified, domain-diverse benchmark samples (N=2000 each)
    logger.info("Sampling stratified, domain-diverse benchmark subsets (N=2000 per partition, Seed 42)...")
    urls_int, labels_int, doms_int, sample_int_df = sample_stratified_domain_benchmark(
        test_meta_legit, test_feat_legit, n_per_class=1000, random_state=42
    )
    urls_ext, labels_ext, doms_ext, sample_ext_df = sample_stratified_domain_benchmark(
        disjoint_meta_phi, disjoint_feat_phi, n_per_class=1000, random_state=42
    )

    # Verify zero domain contamination
    train_doms = set(split_info["train_domains_list"])
    assert len(set(doms_int).intersection(train_doms)) == 0, "Domain contamination in internal test sample!"
    assert len(set(doms_ext).intersection(train_doms)) == 0, "Domain contamination in external disjoint sample!"
    logger.info("Verified zero domain contamination in sampled benchmarks.")

    # Save sampled benchmarks
    sample_int_df.to_csv(data_adv_dir / "sample_internal_test_benchmark.csv", index=False)
    sample_ext_df.to_csv(data_adv_dir / "sample_external_disjoint_benchmark.csv", index=False)

    # 4. Initialize Evaluator
    evaluator = AdversarialEvaluator(
        rf_model=rf_model,
        xgb_model=xgb_model,
        scaler=scaler,
        feature_names=feature_names,
    )

    categories = list(ATTACK_REGISTRY.keys())

    # 5. Execute Adversarial Evaluations on Internal Test and External Disjoint
    eval_int = run_partition_adversarial_evaluation(
        evaluator, urls_int, labels_int, doms_int, "LegitPhish_Internal_Test", categories
    )
    eval_ext = run_partition_adversarial_evaluation(
        evaluator, urls_ext, labels_ext, doms_ext, "PhiUSIIL_External_Disjoint", categories
    )

    # 6. Build and Serialize Result Tables
    logger.info("Serializing attack evaluation tables...")

    def _build_model_table(eval_res: Dict[str, Any], model_key: str) -> pd.DataFrame:
        rows = []
        for cat_id in categories:
            defn = ATTACK_REGISTRY[cat_id]
            m = eval_res[model_key][cat_id]["metrics"]
            ci_asr = eval_res[model_key][cat_id]["ci_asr"]
            ci_er = eval_res[model_key][cat_id]["ci_er"]
            ci_pfr = eval_res[model_key][cat_id]["ci_pfr"]
            mcn = eval_res[model_key][cat_id]["mcnemar"]

            asr_str = f"{m['asr_cond']:.4f} [{ci_asr[0]:.4f}, {ci_asr[1]:.4f}]" if m["asr_cond"] is not None and ci_asr[0] is not None else "N/A"
            er_str = f"{m['er_overall']:.4f} [{ci_er[0]:.4f}, {ci_er[1]:.4f}]" if m["er_overall"] is not None and ci_er[0] is not None else "N/A"
            pfr_str = f"{m['pfr']:.4f} [{ci_pfr[0]:.4f}, {ci_pfr[1]:.4f}]" if ci_pfr[0] is not None else f"{m['pfr']:.4f}"

            rows.append({
                "category_id": cat_id,
                "name": defn.name,
                "tier": defn.tier,
                "n_tx": m["n_tx"],
                "orig_recall": round(m["orig_recall"], 4) if m["orig_recall"] is not None else None,
                "adv_recall": round(m["adv_recall"], 4) if m["adv_recall"] is not None else None,
                "delta_recall": round(m["delta_recall"], 4),
                "asr_cond": asr_str,
                "er_overall": er_str,
                "fpsr": round(m["fpsr"], 4) if m["fpsr"] is not None else None,
                "pfr": pfr_str,
                "orig_f1": round(m["orig_f1"], 4),
                "adv_f1": round(m["adv_f1"], 4),
                "mcnemar_chi2": round(mcn["statistic"], 3),
                "p_raw": mcn["p_value"],
                "p_bonferroni": mcn.get("p_bonferroni"),
                "sig_fdr": mcn.get("sig_fdr"),
            })
        return pd.DataFrame(rows)

    df_rf_int = _build_model_table(eval_int, "rf")
    df_xgb_int = _build_model_table(eval_int, "xgb")
    df_rf_ext = _build_model_table(eval_ext, "rf")
    df_xgb_ext = _build_model_table(eval_ext, "xgb")

    df_rf_int.to_csv(tables_dir / "rf_internal_test_attacks.csv", index=False)
    df_xgb_int.to_csv(tables_dir / "xgb_internal_test_attacks.csv", index=False)
    df_rf_ext.to_csv(tables_dir / "rf_external_disjoint_attacks.csv", index=False)
    df_xgb_ext.to_csv(tables_dir / "xgb_external_disjoint_attacks.csv", index=False)

    # Applicability Table
    df_app = pd.DataFrame(eval_int["applicability"])
    df_app.to_csv(tables_dir / "attack_applicability_summary.csv", index=False)

    # 7. Execute the 5 Specialized Diagnostic Robustness Audits
    logger.info("Executing 5 Specialized Diagnostic Robustness Audits...")

    diag_urls = urls_ext[:500]
    diag_labels = labels_ext[:500]

    # Diagnostic 6.1: Canonicalization
    canon_rf = audit_canonicalization_sensitivity(diag_urls, diag_labels, rf_model)
    canon_xgb = audit_canonicalization_sensitivity(diag_urls, diag_labels, xgb_model)

    # Diagnostic 6.2: Unicode Normalization
    unicode_rf = audit_unicode_normalization_sensitivity(diag_urls, diag_labels, rf_model)
    unicode_xgb = audit_unicode_normalization_sensitivity(diag_urls, diag_labels, xgb_model)

    # Diagnostic 6.3: Continuous Feature Perturbation Scans
    f_sample = [list(extract_url_features(u).values()) for u in diag_urls]
    X_sample = np.array(f_sample, dtype=np.float32)
    pert_rf = audit_feature_perturbation_scans(X_sample, feature_names, rf_model)
    pert_xgb = audit_feature_perturbation_scans(X_sample, feature_names, xgb_model)

    # Diagnostic 6.4: Calibration Drift
    # Take CAT-01A (Path encoding) records on External Disjoint
    rf_cat01_recs = [r for r in eval_ext["rf_records"] if r.category_id == "CAT-01A"]
    xgb_cat01_recs = [r for r in eval_ext["xgb_records"] if r.category_id == "CAT-01A"]

    calib_rf = compute_calibration_drift(
        np.array([r.true_label for r in rf_cat01_recs]),
        np.array([r.orig_prob_phish for r in rf_cat01_recs]),
        np.array([r.adv_prob_phish for r in rf_cat01_recs]),
    )
    calib_xgb = compute_calibration_drift(
        np.array([r.true_label for r in xgb_cat01_recs]),
        np.array([r.orig_prob_phish for r in xgb_cat01_recs]),
        np.array([r.adv_prob_phish for r in xgb_cat01_recs]),
    )

    # Diagnostic 6.5: OOD Boundaries
    ood_rf = audit_ood_structural_boundaries(rf_model)
    ood_xgb = audit_ood_structural_boundaries(xgb_model)

    diagnostics_summary = {
        "6.1_canonicalization": {"rf": canon_rf, "xgb": canon_xgb},
        "6.2_unicode_normalization": {"rf": unicode_rf, "xgb": unicode_xgb},
        "6.3_feature_perturbation_scans": {"rf": pert_rf, "xgb": pert_xgb},
        "6.4_calibration_drift_cat01a": {"rf": calib_rf, "xgb": calib_xgb},
        "6.5_ood_structural_boundaries": {"rf": ood_rf, "xgb": ood_xgb},
    }

    with open(tables_dir / "diagnostics_summary.json", "w") as f:
        json.dump(diagnostics_summary, f, indent=2)

    # 8. Generate Visualization Plots
    logger.info("Generating adversarial visualization plots...")
    generate_adversarial_plots(categories, eval_int, eval_ext, figures_dir)

    # Additional Plot: Calibration comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(["RF (Orig)", "RF (Adv)", "XGB (Orig)", "XGB (Adv)"],
                [calib_rf["brier_orig"], calib_rf["brier_adv"], calib_xgb["brier_orig"], calib_xgb["brier_adv"]],
                color=["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"])
    axes[0].set_title("Brier Score Loss (Lower is Better)")
    axes[0].set_ylabel("Brier Score")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    axes[1].bar(["RF (Orig)", "RF (Adv)", "XGB (Orig)", "XGB (Adv)"],
                [calib_rf["ece_orig"], calib_rf["ece_adv"], calib_xgb["ece_orig"], calib_xgb["ece_adv"]],
                color=["#2ca02c", "#98df8a", "#d62728", "#ff9896"])
    axes[1].set_title("Expected Calibration Error (ECE)")
    axes[1].set_ylabel("ECE")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle("Probability Calibration Drift under Adversarial Attack (CAT-01A External Disjoint)", fontweight="bold")
    fig.savefig(figures_dir / "calibration_comparison.png", dpi=300)
    plt.close(fig)

    logger.info("Phase 6 Adversarial Evaluation Pipeline Completed Successfully.")


if __name__ == "__main__":
    run_pipeline()

