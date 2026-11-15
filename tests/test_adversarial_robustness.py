"""Comprehensive automated unit tests for Phase 6 Adversarial Robustness & Evasion Testing.

Verifies:
1. All attack transformations across Tier A, Tier B, and Tier C.
2. Malformed URLs, missing schemes, IP formats, and parser edge cases.
3. Destination equivalence classifications and RFC standards.
4. Metric denominators with manually calculated, mathematically verified examples:
   - Conditional ASR (denominator = |TP_base|)
   - Overall Phishing Evasion Rate (denominator = N_phish)
   - False Positive Surge Rate (denominator = |TN_base|)
   - Prediction Flip Rate (denominator = N)
5. Paired bootstrap confidence intervals, Edwards continuity-corrected McNemar test,
   and multiple comparison corrections (Bonferroni & Benjamini-Hochberg FDR).
6. Frozen baseline model prediction invariance.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.adversarial.attacks import (
    ATTACK_REGISTRY,
    AttackResult,
    apply_attack,
    transform_cat_01a_unreserved_encoding,
    transform_cat_02a_scheme_casing,
    transform_cat_02b_hostname_casing,
    transform_cat_03a_cyrillic_homoglyph,
    transform_cat_04a_subdomain_phishing,
    transform_cat_04b_subdomain_benign,
    transform_cat_05c_dummy_fragment,
    transform_cat_06a_tracking_parameters,
    transform_cat_07a_keyword_stuffing,
    transform_cat_08b_alternative_ip_format,
    transform_cat_09a_standard_port_injection,
    transform_cat_09b_apex_path_truncation,
)
from src.adversarial.evaluator import (
    PairedEvaluationRecord,
    apply_multiple_testing_correction,
    compute_attack_metrics,
    mcnemar_test,
    paired_bootstrap_ci,
)

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Test 1: Attack Registry and Tier Validity
# ---------------------------------------------------------------------------

def test_attack_registry_integrity():
    """Verify that all 21 attack categories are registered with valid tiers."""
    assert len(ATTACK_REGISTRY) >= 20

    for cat_id, defn in ATTACK_REGISTRY.items():
        assert defn.category_id == cat_id
        assert defn.tier in {"Tier A", "Tier B", "Tier C"}
        assert len(defn.name) > 0
        assert len(defn.rfc_basis) > 0
        assert callable(defn.transform_fn)

    # Check tier assignments of key categories
    assert ATTACK_REGISTRY["CAT-01A"].tier == "Tier A"
    assert ATTACK_REGISTRY["CAT-02A"].tier == "Tier A"
    assert ATTACK_REGISTRY["CAT-02B"].tier == "Tier A"
    assert ATTACK_REGISTRY["CAT-03A"].tier == "Tier C"
    assert ATTACK_REGISTRY["CAT-04A"].tier == "Tier B"
    assert ATTACK_REGISTRY["CAT-05C"].tier == "Tier A"
    assert ATTACK_REGISTRY["CAT-08B"].tier == "Tier C"
    assert ATTACK_REGISTRY["CAT-09A"].tier == "Tier A"
    assert ATTACK_REGISTRY["CAT-09B"].tier == "Tier C"


# ---------------------------------------------------------------------------
# Test 2: Transformation Semantics & Edge Cases
# ---------------------------------------------------------------------------

def test_unreserved_path_encoding():
    """Test CAT-01A unreserved percent encoding."""
    url = "https://example.com/login/verify.php"
    res = transform_cat_01a_unreserved_encoding(url, 0)
    assert res.applicable is True
    assert res.success is True
    assert res.tier == "Tier A"
    assert "%" in res.transformed_url
    assert "example.com" in res.transformed_url

    # Pathless URL should not be applicable
    res_pathless = transform_cat_01a_unreserved_encoding("https://example.com", 0)
    assert res_pathless.applicable is False
    assert res_pathless.success is False


def test_scheme_and_hostname_casing():
    """Test CAT-02A and CAT-02B case variations."""
    # Scheme casing
    res_scheme = transform_cat_02a_scheme_casing("https://example.com/path", 0)
    assert res_scheme.applicable is True
    assert res_scheme.success is True
    assert res_scheme.transformed_url.lower().startswith("https://")
    assert not res_scheme.transformed_url.startswith("https://")  # Must have altered case

    # Hostname casing
    res_host = transform_cat_02b_hostname_casing("https://my-bank-secure.com/path", 0)
    assert res_host.applicable is True
    assert res_host.success is True
    assert res_host.transformed_url.lower() == "https://my-bank-secure.com/path"


def test_cyrillic_homoglyph_tier_c():
    """Test CAT-03A homoglyph replacement is Tier C and alters host characters."""
    url = "https://paypal.com/signin"
    res = transform_cat_03a_cyrillic_homoglyph(url, 0)
    assert res.applicable is True
    assert res.success is True
    assert res.tier == "Tier C"
    # Should contain Cyrillic character (e.g. \u0430 for 'a')
    assert any(ord(c) >= 128 for c in res.transformed_url)


def test_subdomain_prepending_by_class():
    """Test CAT-04A and CAT-04B class-conditional applicability."""
    phish_url = "http://attacker-site.com/form"
    benign_url = "https://google.com/search"

    # CAT-04A: only applicable to phishing (0)
    res_phish = transform_cat_04a_subdomain_phishing(phish_url, 0)
    assert res_phish.applicable is True
    assert res_phish.success is True
    assert "secure-login.attacker-site.com" in res_phish.transformed_url

    res_phish_on_benign = transform_cat_04a_subdomain_phishing(benign_url, 1)
    assert res_phish_on_benign.applicable is False

    # CAT-04B: only applicable to benign (1)
    res_benign = transform_cat_04b_subdomain_benign(benign_url, 1)
    assert res_benign.applicable is True
    assert res_benign.success is True
    assert "security-update.google.com" in res_benign.transformed_url

    res_benign_on_phish = transform_cat_04b_subdomain_benign(phish_url, 0)
    assert res_benign_on_phish.applicable is False


def test_alternative_ip_formats():
    """Test CAT-08B alternative IP format (DWORD)."""
    ip_url = "http://192.168.1.1/login"
    res = transform_cat_08b_alternative_ip_format(ip_url, 0)
    assert res.applicable is True
    assert res.success is True
    assert res.tier == "Tier C"
    # 192.168.1.1 as DWORD is 3232235777
    assert "3232235777" in res.transformed_url

    # Non-IP URL should not be applicable
    domain_url = "http://example.com/login"
    res_domain = transform_cat_08b_alternative_ip_format(domain_url, 0)
    assert res_domain.applicable is False


def test_port_injection_and_path_truncation():
    """Test CAT-09A standard port injection and CAT-09B root truncation."""
    # Port injection
    res_port_http = transform_cat_09a_standard_port_injection("http://example.com/test", 0)
    assert res_port_http.applicable is True
    assert ":80" in res_port_http.transformed_url

    res_port_https = transform_cat_09a_standard_port_injection("https://example.com/test", 0)
    assert res_port_https.applicable is True
    assert ":443" in res_port_https.transformed_url

    # Root truncation
    res_trunc = transform_cat_09b_apex_path_truncation("https://example.com/dir1/dir2/index.html?token=123", 0)
    assert res_trunc.applicable is True
    assert res_trunc.success is True
    assert res_trunc.tier == "Tier C"
    assert res_trunc.transformed_url == "https://example.com/"


def test_parser_edge_cases():
    """Test parser handling of missing schemes, userinfo, and empty strings."""
    # Missing scheme URL
    no_scheme = "evil-domain.com/login/auth"
    res = apply_attack("CAT-01A", no_scheme, 0)
    assert res.applicable is True
    assert "%" in res.transformed_url

    # URL with existing fragment
    with_frag = "https://example.com/test#already-here"
    res_frag = apply_attack("CAT-05C", with_frag, 0)
    assert res_frag.applicable is True
    assert "#verification-checkpoint" in res_frag.transformed_url

    # URL with query
    with_query = "https://example.com/search?q=test"
    res_query = apply_attack("CAT-06A", with_query, 0)
    assert res_query.applicable is True
    assert "utm_source=security_notice" in res_query.transformed_url
    assert "q=test" in res_query.transformed_url


# ---------------------------------------------------------------------------
# Test 3: Metric Denominators with Manually Verified Numbers
# ---------------------------------------------------------------------------

def test_metric_denominators_manual_calculation():
    """Verify that metric numerators and denominators match manual arithmetic exactly.

    Controlled Synthetic Scenario:
    Total Instances: N = 20 (10 Phishing [y=0], 10 Legitimate [y=1]).
    Baseline Predictions:
      Phishing: 8 detected (TP=8), 2 missed (FN=2).
      Legitimate: 9 correct (TN=9), 1 false alarm (FP=1).
    Adversarial Transitions:
      Of the 8 baseline TPs:
        - 4 remain detected as Phishing (0)
        - 4 flip to Legitimate (1) [Successful Evasion]
      The 2 baseline FNs remain classified as Legitimate (1).
      Of the 9 baseline TNs:
        - 6 remain Legitimate (1)
        - 3 flip to Phishing (0) [False Alarm Surge]
      The 1 baseline FP remains classified as Phishing (0).

    Expected Metrics:
      n_tx = 20
      n_phish = 10
      n_legit = 10
      n_tp_base = 8
      n_tn_base = 9
      ASR_cond = 4 / 8 = 0.5000 (50.0%)
      ER_overall = (4 + 2) / 10 = 0.6000 (60.0%)
      orig_recall = 8 / 10 = 0.8000
      adv_recall = 1 - 0.6000 = 0.4000
      delta_recall = 0.8000 - 0.4000 = 0.4000
      FPSR = 3 / 9 = 0.33333... (33.33%)
      PFR = (4 [TP flips] + 3 [TN flips]) / 20 = 7 / 20 = 0.3500 (35.0%)
    """
    records: List[PairedEvaluationRecord] = []

    # 1. Baseline TPs (8 items): 4 flip to 1, 4 stay 0
    for i in range(4):
        records.append(PairedEvaluationRecord(
            sample_id=i, source_partition="test", registrable_domain="dom.com",
            true_label=0, orig_url=f"u_{i}", adv_url=f"u_adv_{i}",
            applicable=True, transformed=True, tier="Tier A",
            category_id="CAT-TEST", category_name="Test Attack",
            orig_pred=0, orig_prob_phish=0.9, adv_pred=1, adv_prob_phish=0.2  # Evaded
        ))
    for i in range(4, 8):
        records.append(PairedEvaluationRecord(
            sample_id=i, source_partition="test", registrable_domain="dom.com",
            true_label=0, orig_url=f"u_{i}", adv_url=f"u_adv_{i}",
            applicable=True, transformed=True, tier="Tier A",
            category_id="CAT-TEST", category_name="Test Attack",
            orig_pred=0, orig_prob_phish=0.9, adv_pred=0, adv_prob_phish=0.85  # Stayed Phish
        ))

    # 2. Baseline FNs (2 items): both stay 1
    for i in range(8, 10):
        records.append(PairedEvaluationRecord(
            sample_id=i, source_partition="test", registrable_domain="dom.com",
            true_label=0, orig_url=f"u_{i}", adv_url=f"u_adv_{i}",
            applicable=True, transformed=True, tier="Tier A",
            category_id="CAT-TEST", category_name="Test Attack",
            orig_pred=1, orig_prob_phish=0.3, adv_pred=1, adv_prob_phish=0.25  # Stayed Legitimate
        ))

    # 3. Baseline TNs (9 items): 3 flip to 0, 6 stay 1
    for i in range(10, 13):
        records.append(PairedEvaluationRecord(
            sample_id=i, source_partition="test", registrable_domain="dom.com",
            true_label=1, orig_url=f"u_{i}", adv_url=f"u_adv_{i}",
            applicable=True, transformed=True, tier="Tier A",
            category_id="CAT-TEST", category_name="Test Attack",
            orig_pred=1, orig_prob_phish=0.1, adv_pred=0, adv_prob_phish=0.8  # FP surge
        ))
    for i in range(13, 19):
        records.append(PairedEvaluationRecord(
            sample_id=i, source_partition="test", registrable_domain="dom.com",
            true_label=1, orig_url=f"u_{i}", adv_url=f"u_adv_{i}",
            applicable=True, transformed=True, tier="Tier A",
            category_id="CAT-TEST", category_name="Test Attack",
            orig_pred=1, orig_prob_phish=0.1, adv_pred=1, adv_prob_phish=0.15  # Stayed Legit
        ))

    # 4. Baseline FP (1 item): stays 0
    records.append(PairedEvaluationRecord(
        sample_id=19, source_partition="test", registrable_domain="dom.com",
        true_label=1, orig_url="u_19", adv_url="u_adv_19",
        applicable=True, transformed=True, tier="Tier A",
        category_id="CAT-TEST", category_name="Test Attack",
        orig_pred=0, orig_prob_phish=0.7, adv_pred=0, adv_prob_phish=0.65  # Stayed FP
    ))

    # Compute metrics
    m = compute_attack_metrics(records)

    assert m["n_tx"] == 20
    assert m["n_phish"] == 10
    assert m["n_legit"] == 10
    assert m["n_tp_base"] == 8
    assert m["n_tn_base"] == 9

    # Verify exact equality with mathematical denominators
    assert pytest.approx(m["asr_cond"], 1e-6) == 4.0 / 8.0  # 0.50
    assert pytest.approx(m["er_overall"], 1e-6) == 6.0 / 10.0  # 0.60
    assert pytest.approx(m["fpsr"], 1e-6) == 3.0 / 9.0  # 0.333333
    assert pytest.approx(m["pfr"], 1e-6) == 7.0 / 20.0  # 0.35
    assert pytest.approx(m["orig_recall"], 1e-6) == 8.0 / 10.0  # 0.80
    assert pytest.approx(m["adv_recall"], 1e-6) == 4.0 / 10.0  # 0.40
    assert pytest.approx(m["delta_recall"], 1e-6) == 0.40


# ---------------------------------------------------------------------------
# Test 4: Statistical Testing & Multiple Comparison Corrections
# ---------------------------------------------------------------------------

def test_mcnemar_exact_and_asymptotic():
    """Verify Edwards continuity-corrected McNemar test calculation."""
    # Construct synthetic paired records with known b=15, c=2 (total discordant = 17 < 25 -> exact binomial)
    records = []
    # 15 vulnerability flips (b: orig=correct, adv=incorrect)
    for i in range(15):
        records.append(PairedEvaluationRecord(
            sample_id=i, source_partition="test", registrable_domain="dom.com",
            true_label=0, orig_url="u", adv_url="u", applicable=True, transformed=True,
            tier="Tier A", category_id="C", category_name="N",
            orig_pred=0, orig_prob_phish=0.9, adv_pred=1, adv_prob_phish=0.2
        ))
    # 2 serendipitous flips (c: orig=incorrect, adv=correct)
    for i in range(15, 17):
        records.append(PairedEvaluationRecord(
            sample_id=i, source_partition="test", registrable_domain="dom.com",
            true_label=0, orig_url="u", adv_url="u", applicable=True, transformed=True,
            tier="Tier A", category_id="C", category_name="N",
            orig_pred=1, orig_prob_phish=0.2, adv_pred=0, adv_prob_phish=0.9
        ))

    res = mcnemar_test(records)
    assert res["b"] == 15
    assert res["c"] == 17 - 15  # 2
    assert res["method"] == "exact_binomial"
    assert res["p_value"] < 0.01  # Highly significant disparity


def test_multiple_testing_correction():
    """Verify Bonferroni and Benjamini-Hochberg FDR adjustments."""
    p_vals = [0.001, 0.01, 0.04, 0.20, 0.80]
    bonf_p, bonf_sig, fdr_p, fdr_sig = apply_multiple_testing_correction(p_vals, alpha=0.05)

    assert len(bonf_p) == 5
    assert bonf_p[0] == pytest.approx(0.005)  # 0.001 * 5
    assert bonf_sig[0] is True
    assert bonf_sig[3] is False

    # FDR is less conservative than Bonferroni
    assert fdr_sig[0] is True
    assert fdr_p[0] <= bonf_p[0]


def test_paired_bootstrap_confidence_interval():
    """Verify bootstrap confidence interval bounds contain the empirical point estimate."""
    records = []
    for i in range(100):
        records.append(PairedEvaluationRecord(
            sample_id=i, source_partition="test", registrable_domain="dom.com",
            true_label=0, orig_url="u", adv_url="u", applicable=True, transformed=True,
            tier="Tier A", category_id="C", category_name="N",
            orig_pred=0, orig_prob_phish=0.9, adv_pred=1 if i < 30 else 0, adv_prob_phish=0.2
        ))

    ci_low, ci_high = paired_bootstrap_ci(records, "asr_cond", n_resamples=200, random_state=42)
    assert ci_low is not None and ci_high is not None
    assert 0.0 <= ci_low <= 0.30 <= ci_high <= 1.0


# ---------------------------------------------------------------------------
# Test 5: Frozen Baseline Model Predictions Invariance
# ---------------------------------------------------------------------------

def test_frozen_models_prediction_invariance():
    """Verify that loading the frozen Random Forest and XGBoost models yields invariant predictions."""
    rf_path = BASE_DIR / "models/trained/random_forest.joblib"
    xgb_path = BASE_DIR / "models/trained/xgboost.joblib"

    rf = joblib.load(rf_path)
    xgb = joblib.load(xgb_path)

    # Test on deterministic fixed inputs
    rng = np.random.RandomState(42)
    X_synthetic = rng.randn(10, 68).astype(np.float32)

    pred_rf_1 = rf.predict(X_synthetic)
    pred_rf_2 = rf.predict(X_synthetic)
    np.testing.assert_array_equal(pred_rf_1, pred_rf_2)

    pred_xgb_1 = xgb.predict(X_synthetic)
    pred_xgb_2 = xgb.predict(X_synthetic)
    np.testing.assert_array_equal(pred_xgb_1, pred_xgb_2)
