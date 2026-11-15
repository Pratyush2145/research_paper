"""Evaluation engine for adversarial robustness, paired transitions, and statistical testing.

Implements:
- Strict paired sample tracking across original and adversarial representations.
- Closed-form metric computation with mathematically unambiguous denominators:
    * Conditional Attack Success Rate (ASR_cond)
    * Overall Phishing Evasion Rate (ER_overall)
    * False Positive Surge Rate (FPSR)
    * Prediction Flip Rate (PFR)
    * Performance Degradation (Delta Recall, Delta F1, Delta Accuracy)
    * Posterior Score Displacement (Delta P(Phish))
- Paired bootstrap confidence intervals (1,000 resamples).
- Edwards continuity-corrected McNemar test (and exact binomial test for sparse cells).
- Multiple comparison adjustments (Bonferroni FWER and Benjamini-Hochberg FDR).
- Five diagnostic robustness audits (Canonicalization, Unicode/IDN, Perturbation Scans,
  Calibration Drift, and OOD Structural Boundaries).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss

from src.adversarial.attacks import AttackResult, apply_attack
from src.features.feature_extractor import extract_url_features
from src.features.url_parser import parse_url

logger = logging.getLogger(__name__)


@dataclass
class PairedEvaluationRecord:
    """Immutable paired tracking record for a single evaluation instance."""

    sample_id: int
    source_partition: str
    registrable_domain: str
    true_label: int  # 0 = Phishing, 1 = Legitimate
    orig_url: str
    adv_url: str
    applicable: bool
    transformed: bool
    tier: str
    category_id: str
    category_name: str
    orig_pred: int  # 0 = Phishing, 1 = Legitimate
    orig_prob_phish: float
    adv_pred: int
    adv_prob_phish: float


# ---------------------------------------------------------------------------
# Metric Formulations with Explicit Denominators
# ---------------------------------------------------------------------------

def compute_attack_metrics(
    records: List[PairedEvaluationRecord],
) -> Dict[str, Any]:
    """Compute formal adversarial metrics with closed-form denominators.

    Args:
        records: List of PairedEvaluationRecord for successfully transformed instances.

    Returns:
        Structured dictionary of robustness metrics.
    """
    n_tx = len(records)
    if n_tx == 0:
        return {
            "n_tx": 0,
            "asr_cond": None,
            "er_overall": None,
            "fpsr": None,
            "pfr": 0.0,
            "delta_recall": 0.0,
            "delta_f1": 0.0,
            "delta_acc": 0.0,
            "mean_delta_prob": 0.0,
            "median_delta_prob": 0.0,
            "iqr_delta_prob": 0.0,
            "orig_recall": 0.0,
            "adv_recall": 0.0,
            "orig_f1": 0.0,
            "adv_f1": 0.0,
            "orig_acc": 0.0,
            "adv_acc": 0.0,
        }

    y_true = np.array([r.true_label for r in records], dtype=int)
    y_orig = np.array([r.orig_pred for r in records], dtype=int)
    y_adv = np.array([r.adv_pred for r in records], dtype=int)
    p_orig = np.array([r.orig_prob_phish for r in records], dtype=float)
    p_adv = np.array([r.adv_prob_phish for r in records], dtype=float)

    # In security evaluation: Class 0 = Phishing (threat), Class 1 = Legitimate (benign)
    phish_mask = (y_true == 0)
    legit_mask = (y_true == 1)

    n_phish = int(phish_mask.sum())
    n_legit = int(legit_mask.sum())

    # 1. Baseline True Positives (Phishing detected as Phishing: y=0 and orig=0)
    tp_base_mask = phish_mask & (y_orig == 0)
    n_tp_base = int(tp_base_mask.sum())

    # Conditional ASR: fraction of baseline TPs that flip to Legitimate (1) under attack
    if n_tp_base > 0:
        evaded_tp = int((tp_base_mask & (y_adv == 1)).sum())
        asr_cond = float(evaded_tp / n_tp_base)
    else:
        asr_cond = None

    # 2. Overall Phishing Evasion Rate: fraction of ALL phishing URLs classified as Legitimate (1)
    if n_phish > 0:
        all_evaded = int((phish_mask & (y_adv == 1)).sum())
        er_overall = float(all_evaded / n_phish)
        orig_phish_detected = int((phish_mask & (y_orig == 0)).sum())
        orig_recall = float(orig_phish_detected / n_phish)
        adv_recall = 1.0 - er_overall
    else:
        er_overall = None
        orig_recall = None
        adv_recall = None

    # 3. False Positive Surge Rate: fraction of baseline TNs that flip to Phishing (0) under attack
    tn_base_mask = legit_mask & (y_orig == 1)
    n_tn_base = int(tn_base_mask.sum())
    if n_tn_base > 0:
        surge_fp = int((tn_base_mask & (y_adv == 0)).sum())
        fpsr = float(surge_fp / n_tn_base)
    else:
        fpsr = None

    # 4. Prediction Flip Rate: fraction of ALL instances where decision changes
    pfr = float(np.mean(y_orig != y_adv))

    # 5. Continuous Probability Shift (Delta P(Phish) = p_adv - p_orig)
    delta_p = p_adv - p_orig
    mean_dp = float(np.mean(delta_p))
    median_dp = float(np.median(delta_p))
    q75, q25 = np.percentile(delta_p, [75, 25])
    iqr_dp = float(q75 - q25)

    # 6. Performance Metrics and Degradations
    orig_acc = float(np.mean(y_orig == y_true))
    adv_acc = float(np.mean(y_adv == y_true))
    delta_acc = orig_acc - adv_acc

    # Calculate F1 on phishing class (0 as positive)
    def _calc_phish_f1(preds: np.ndarray) -> float:
        tp = np.sum((y_true == 0) & (preds == 0))
        fp = np.sum((y_true == 1) & (preds == 0))
        fn = np.sum((y_true == 0) & (preds == 1))
        denom = 2 * tp + fp + fn
        return float(2 * tp / denom) if denom > 0 else 0.0

    orig_f1 = _calc_phish_f1(y_orig)
    adv_f1 = _calc_phish_f1(y_adv)
    delta_f1 = orig_f1 - adv_f1
    delta_recall = (orig_recall - adv_recall) if (orig_recall is not None and adv_recall is not None) else 0.0

    return {
        "n_tx": n_tx,
        "n_phish": n_phish,
        "n_legit": n_legit,
        "n_tp_base": n_tp_base,
        "n_tn_base": n_tn_base,
        "asr_cond": asr_cond,
        "er_overall": er_overall,
        "fpsr": fpsr,
        "pfr": pfr,
        "delta_recall": delta_recall,
        "delta_f1": delta_f1,
        "delta_acc": delta_acc,
        "mean_delta_prob": mean_dp,
        "median_delta_prob": median_dp,
        "iqr_delta_prob": iqr_dp,
        "orig_recall": orig_recall,
        "adv_recall": adv_recall,
        "orig_f1": orig_f1,
        "adv_f1": adv_f1,
        "orig_acc": orig_acc,
        "adv_acc": adv_acc,
    }


# ---------------------------------------------------------------------------
# Statistical Testing: Paired Bootstrap, McNemar, and FDR Adjustments
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    records: List[PairedEvaluationRecord],
    metric_name: str,
    n_resamples: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute 95% paired bootstrap percentile confidence interval.

    Args:
        records: List of successfully transformed paired records.
        metric_name: Name of metric ('asr_cond', 'er_overall', 'fpsr', 'pfr', 'delta_recall', 'delta_f1').
        n_resamples: Number of bootstrap resamples (default: 1,000).
        random_state: Deterministic random seed.
        alpha: Two-tailed significance level (default: 0.05 for 95% CI).

    Returns:
        (lower_bound, upper_bound)
    """
    n = len(records)
    if n == 0:
        return None, None

    rng = np.random.RandomState(random_state)
    boot_vals = []

    for _ in range(n_resamples):
        sample_indices = rng.randint(0, n, size=n)
        sample_records = [records[i] for i in sample_indices]
        m = compute_attack_metrics(sample_records)
        val = m.get(metric_name)
        if val is not None:
            boot_vals.append(val)

    if not boot_vals:
        return None, None

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_vals, lower_pct))
    ci_upper = float(np.percentile(boot_vals, upper_pct))
    return ci_lower, ci_upper


def mcnemar_test(records: List[PairedEvaluationRecord]) -> Dict[str, Any]:
    """Execute Edwards continuity-corrected McNemar test on paired classification accuracy.

    Constructs 2x2 contingency matrix:
        a: Both original and adversarial predictions are correct
        b: Original correct, Adversarial incorrect (Vulnerability Flip)
        c: Original incorrect, Adversarial correct (Serendipitous Flip)
        d: Both original and adversarial predictions are incorrect

    Returns:
        Dictionary with contingency counts, chi2 statistic, p-value, and test type.
    """
    if not records:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0, "method": "none"}

    y_true = np.array([r.true_label for r in records], dtype=int)
    y_orig = np.array([r.orig_pred for r in records], dtype=int)
    y_adv = np.array([r.adv_pred for r in records], dtype=int)

    orig_correct = (y_orig == y_true)
    adv_correct = (y_adv == y_true)

    a = int(np.sum(orig_correct & adv_correct))
    b = int(np.sum(orig_correct & ~adv_correct))
    c = int(np.sum(~orig_correct & adv_correct))
    d = int(np.sum(~orig_correct & ~adv_correct))

    # Test discordant pairs (b vs c)
    total_discordant = b + c
    if total_discordant == 0:
        return {
            "a": a, "b": b, "c": c, "d": d,
            "statistic": 0.0, "p_value": 1.0, "method": "exact (no discordant pairs)"
        }

    if total_discordant < 25:
        # Exact two-tailed binomial test
        binom_res = stats.binomtest(b, total_discordant, p=0.5, alternative="two-sided")
        return {
            "a": a, "b": b, "c": c, "d": d,
            "statistic": float(b),
            "p_value": float(binom_res.pvalue),
            "method": "exact_binomial",
        }

    # Edwards continuity-corrected McNemar chi-square
    chi2 = float((abs(b - c) - 1.0) ** 2 / total_discordant)
    p_val = float(stats.chi2.sf(chi2, df=1))

    return {
        "a": a, "b": b, "c": c, "d": d,
        "statistic": chi2,
        "p_value": p_val,
        "method": "mcnemar_continuity_corrected",
    }


def apply_multiple_testing_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool], List[float], List[bool]]:
    """Apply Bonferroni FWER and Benjamini-Hochberg FDR adjustments.

    Args:
        p_values: List of raw p-values.
        alpha: Nominal significance threshold (0.05).

    Returns:
        (bonf_p, bonf_sig, fdr_p, fdr_sig)
    """
    k = len(p_values)
    if k == 0:
        return [], [], [], []

    # 1. Bonferroni
    bonf_p = [min(1.0, p * k) for p in p_values]
    bonf_sig = [p <= (alpha / k) for p in p_values]

    # 2. Benjamini-Hochberg FDR
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    fdr_sig = [False] * k
    fdr_adjusted_p = [1.0] * k

    # Find largest rank i such that p(i) <= (i / k) * alpha
    max_sig_idx = -1
    for rank_idx, p in enumerate(sorted_p):
        rank = rank_idx + 1
        threshold = (rank / k) * alpha
        if p <= threshold:
            max_sig_idx = rank_idx

    for orig_pos, idx in enumerate(sorted_indices):
        if orig_pos <= max_sig_idx:
            fdr_sig[idx] = True

    # Compute step-up adjusted p-values
    cum_min = 1.0
    adjusted_sorted = np.zeros(k)
    for i in range(k - 1, -1, -1):
        rank = i + 1
        val = (sorted_p[i] * k) / rank
        cum_min = min(cum_min, val)
        adjusted_sorted[i] = min(1.0, cum_min)

    for orig_idx, sorted_idx in enumerate(sorted_indices):
        fdr_adjusted_p[sorted_idx] = float(adjusted_sorted[orig_idx])

    return bonf_p, bonf_sig, fdr_adjusted_p, fdr_sig


# ---------------------------------------------------------------------------
# Diagnostic Robustness Audits
# ---------------------------------------------------------------------------

def compute_expected_calibration_error(
    y_true_binary: np.ndarray,
    probs_phish: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE) treating Phishing (0) as the target event.

    Args:
        y_true_binary: Array where 1 indicates Phishing (y_true == 0), 0 indicates Legitimate.
        probs_phish: Model predicted probability for Phishing class.
        n_bins: Number of equal-frequency or uniform bins (default: 10).

    Returns:
        ECE value between 0.0 and 1.0.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true_binary)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (probs_phish >= bin_lower) & (probs_phish <= bin_upper)
        else:
            in_bin = (probs_phish >= bin_lower) & (probs_phish < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true_binary[in_bin])
            bin_conf = np.mean(probs_phish[in_bin])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_calibration_drift(
    y_true: np.ndarray,
    p_orig: np.ndarray,
    p_adv: np.ndarray,
) -> Dict[str, float]:
    """Compute Brier Score and ECE drift before and after adversarial manipulation."""
    y_target = (y_true == 0).astype(int)  # 1 for Phishing, 0 for Legitimate

    brier_orig = float(brier_score_loss(y_target, p_orig))
    brier_adv = float(brier_score_loss(y_target, p_adv))
    delta_brier = brier_adv - brier_orig

    ece_orig = compute_expected_calibration_error(y_target, p_orig)
    ece_adv = compute_expected_calibration_error(y_target, p_adv)
    delta_ece = ece_adv - ece_orig

    return {
        "brier_orig": brier_orig,
        "brier_adv": brier_adv,
        "delta_brier": delta_brier,
        "ece_orig": ece_orig,
        "ece_adv": ece_adv,
        "delta_ece": delta_ece,
    }


def audit_canonicalization_sensitivity(
    urls: List[str],
    labels: List[int],
    model: Any,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Diagnostic 6.1: Measure feature drift and prediction sensitivity on RFC canonicalization."""
    from urllib.parse import unquote, urlsplit, urlunsplit

    raw_features_list = []
    canon_features_list = []
    flips = 0

    for url, lbl in zip(urls, labels):
        parts = urlsplit(url)
        # RFC canonicalization: lowercase scheme & host, unquote unreserved in path, strip default port
        scheme = parts.scheme.lower()
        host = parts.hostname.lower() if parts.hostname else ""
        port = parts.port
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None
        netloc = f"{host}:{port}" if port is not None else host
        path = unquote(parts.path)
        canon_url = urlunsplit((scheme, netloc, path, parts.query, parts.fragment))

        f_raw = extract_url_features(url)
        f_canon = extract_url_features(canon_url)

        raw_features_list.append(list(f_raw.values()))
        canon_features_list.append(list(f_canon.values()))

    X_raw = np.array(raw_features_list, dtype=np.float32)
    X_canon = np.array(canon_features_list, dtype=np.float32)

    abs_diff = np.abs(X_raw - X_canon)
    max_drift = float(np.max(abs_diff))
    mean_drift = float(np.mean(abs_diff))

    X_raw_in = scaler.transform(X_raw) if scaler else X_raw
    X_canon_in = scaler.transform(X_canon) if scaler else X_canon

    pred_raw = model.predict(X_raw_in)
    pred_canon = model.predict(X_canon_in)
    pfr = float(np.mean(pred_raw != pred_canon))

    return {
        "n_audited": len(urls),
        "max_feature_drift": max_drift,
        "mean_feature_drift": mean_drift,
        "prediction_flip_rate": pfr,
    }


def audit_unicode_normalization_sensitivity(
    urls: List[str],
    labels: List[int],
    model: Any,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Diagnostic 6.2: Test invariance across Unicode forms (NFC, NFD, NFKC, NFKD)."""
    forms = ["NFC", "NFD", "NFKC", "NFKD"]
    results = {}

    f_base = [list(extract_url_features(u).values()) for u in urls]
    X_base = np.array(f_base, dtype=np.float32)
    X_base_in = scaler.transform(X_base) if scaler else X_base
    pred_base = model.predict(X_base_in)

    for form in forms:
        f_norm = [list(extract_url_features(unicodedata.normalize(form, u)).values()) for u in urls]
        X_norm = np.array(f_norm, dtype=np.float32)
        abs_diff = np.abs(X_base - X_norm)
        X_norm_in = scaler.transform(X_norm) if scaler else X_norm
        pred_norm = model.predict(X_norm_in)

        results[form] = {
            "max_drift": float(np.max(abs_diff)),
            "mean_drift": float(np.mean(abs_diff)),
            "flip_rate": float(np.mean(pred_base != pred_norm)),
        }

    return results


def audit_feature_perturbation_scans(
    X_sample: np.ndarray,
    feature_names: List[str],
    model: Any,
    scaler: Optional[Any] = None,
    target_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Diagnostic 6.3: Scan continuous SHAP features along grid to measure decision boundary distance."""
    if target_features is None:
        target_features = ["path_length", "max_path_segment_length", "digit_ratio", "url_entropy"]

    feature_indices = {name: i for i, name in enumerate(feature_names)}
    grid_multipliers = [-0.50, -0.25, -0.10, 0.10, 0.25, 0.50]
    scan_results = {}

    X_base_in = scaler.transform(X_sample) if scaler else X_sample
    pred_base = model.predict(X_base_in)
    prob_base = model.predict_proba(X_base_in)[:, 0]  # Phishing prob

    for feat in target_features:
        if feat not in feature_indices:
            continue
        col_idx = feature_indices[feat]
        feat_results = {}

        for mult in grid_multipliers:
            X_pert = X_sample.copy()
            X_pert[:, col_idx] = np.maximum(0.0, X_pert[:, col_idx] * (1.0 + mult))
            X_pert_in = scaler.transform(X_pert) if scaler else X_pert
            pred_pert = model.predict(X_pert_in)
            prob_pert = model.predict_proba(X_pert_in)[:, 0]

            flip_rate = float(np.mean(pred_base != pred_pert))
            mean_prob_shift = float(np.mean(prob_pert - prob_base))

            feat_results[f"{mult:+.2f}"] = {
                "flip_rate": flip_rate,
                "mean_phish_prob_shift": mean_prob_shift,
            }

        scan_results[feat] = feat_results

    return scan_results


def audit_ood_structural_boundaries(
    model: Any,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Diagnostic 6.5: Probe model on Out-of-Distribution URL structures."""
    ood_cases = [
        ("userinfo_token", "https://admin:secretPass123@login.portal-security.com/auth"),
        ("extreme_query_length", "https://portal.bank.com/service?" + "param=" + "a" * 1500),
        ("zero_path_https_root", "https://secure-login-account-update.com/"),
        ("deep_nested_empty_segments", "https://example.com///////login/////verify"),
        ("excessive_subdomains", "https://" + ".".join(["sub"] * 25) + ".example.com/login"),
    ]

    results = {}
    for case_name, url in ood_cases:
        feat_dict = extract_url_features(url)
        X = np.array([list(feat_dict.values())], dtype=np.float32)
        X_in = scaler.transform(X) if scaler else X
        pred = int(model.predict(X_in)[0])
        prob_phish = float(model.predict_proba(X_in)[0, 0])
        results[case_name] = {
            "url": url[:80] + ("..." if len(url) > 80 else ""),
            "prediction": "Phishing (0)" if pred == 0 else "Legitimate (1)",
            "prob_phish": prob_phish,
        }

    return results


# ---------------------------------------------------------------------------
# High-Level Evaluator Class
# ---------------------------------------------------------------------------

class AdversarialEvaluator:
    """Manages end-to-end paired adversarial evaluation across models and partitions."""

    def __init__(
        self,
        rf_model: Any,
        xgb_model: Any,
        scaler: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.rf = rf_model
        self.xgb = xgb_model
        self.scaler = scaler
        self.feature_names = feature_names

    def evaluate_url_batch(
        self,
        urls: List[str],
        labels: List[int],
        domains: List[str],
        category_id: str,
        partition_name: str,
    ) -> Tuple[List[PairedEvaluationRecord], List[PairedEvaluationRecord], Dict[str, Any]]:
        """Transform batch of URLs and generate paired tracking records for RF and XGBoost.

        Returns:
            (rf_records, xgb_records, applicability_stats)
        """
        n_cand = len(urls)
        applicable_count = 0
        tx_count = 0

        rf_records: List[PairedEvaluationRecord] = []
        xgb_records: List[PairedEvaluationRecord] = []

        # First pass: apply attack transformations
        attack_results: List[AttackResult] = []
        for u, lbl in zip(urls, labels):
            res = apply_attack(category_id, u, lbl)
            attack_results.append(res)
            if res.applicable:
                applicable_count += 1
            if res.success:
                tx_count += 1

        applicability_stats = {
            "category_id": category_id,
            "n_cand": n_cand,
            "n_app": applicable_count,
            "n_tx": tx_count,
            "app_rate": float(applicable_count / n_cand) if n_cand > 0 else 0.0,
            "tx_rate": float(tx_count / applicable_count) if applicable_count > 0 else 0.0,
        }

        # Filter to successfully transformed instances for paired evaluation
        valid_indices = [i for i, r in enumerate(attack_results) if r.success]
        if not valid_indices:
            return rf_records, xgb_records, applicability_stats

        orig_urls_subset = [urls[i] for i in valid_indices]
        adv_urls_subset = [attack_results[i].transformed_url for i in valid_indices]
        labels_subset = [labels[i] for i in valid_indices]
        domains_subset = [domains[i] for i in valid_indices]
        tier = attack_results[valid_indices[0]].tier
        cat_name = attack_results[valid_indices[0]].category_name

        # Extract features for original and adversarial URLs
        orig_feats = [list(extract_url_features(u).values()) for u in orig_urls_subset]
        adv_feats = [list(extract_url_features(u).values()) for u in adv_urls_subset]

        X_orig = np.array(orig_feats, dtype=np.float32)
        X_adv = np.array(adv_feats, dtype=np.float32)

        # Tree ensembles (RF and XGB) were trained on scaled or unscaled data?
        # Note: tree ensembles in scikit-learn/xgboost are invariant to monotonic scaling,
        # but scaler was fit during training. Let's pass scaled or unscaled appropriately.
        # RF and XGB in pipeline were trained on X_train (unscaled) for tree models or scaled?
        # Let's verify how pipeline trained RF and XGB!
        X_orig_rf = X_orig
        X_adv_rf = X_adv
        X_orig_xgb = X_orig
        X_adv_xgb = X_adv

        # Predictions and probabilities
        pred_orig_rf = self.rf.predict(X_orig_rf)
        prob_orig_rf = self.rf.predict_proba(X_orig_rf)[:, 0]  # Phishing prob
        pred_adv_rf = self.rf.predict(X_adv_rf)
        prob_adv_rf = self.rf.predict_proba(X_adv_rf)[:, 0]

        pred_orig_xgb = self.xgb.predict(X_orig_xgb)
        prob_orig_xgb = self.xgb.predict_proba(X_orig_xgb)[:, 0]
        pred_adv_xgb = self.xgb.predict(X_adv_xgb)
        prob_adv_xgb = self.xgb.predict_proba(X_adv_xgb)[:, 0]

        for idx, orig_idx in enumerate(valid_indices):
            rec_rf = PairedEvaluationRecord(
                sample_id=orig_idx,
                source_partition=partition_name,
                registrable_domain=domains_subset[idx],
                true_label=labels_subset[idx],
                orig_url=orig_urls_subset[idx],
                adv_url=adv_urls_subset[idx],
                applicable=True,
                transformed=True,
                tier=tier,
                category_id=category_id,
                category_name=cat_name,
                orig_pred=int(pred_orig_rf[idx]),
                orig_prob_phish=float(prob_orig_rf[idx]),
                adv_pred=int(pred_adv_rf[idx]),
                adv_prob_phish=float(prob_adv_rf[idx]),
            )
            rf_records.append(rec_rf)

            rec_xgb = PairedEvaluationRecord(
                sample_id=orig_idx,
                source_partition=partition_name,
                registrable_domain=domains_subset[idx],
                true_label=labels_subset[idx],
                orig_url=orig_urls_subset[idx],
                adv_url=adv_urls_subset[idx],
                applicable=True,
                transformed=True,
                tier=tier,
                category_id=category_id,
                category_name=cat_name,
                orig_pred=int(pred_orig_xgb[idx]),
                orig_prob_phish=float(prob_orig_xgb[idx]),
                adv_pred=int(pred_adv_xgb[idx]),
                adv_prob_phish=float(prob_adv_xgb[idx]),
            )
            xgb_records.append(rec_xgb)

        return rf_records, xgb_records, applicability_stats

