"""Comprehensive unit tests for URL-only feature engineering and dataset preparation.

Tests URL normalization, lexical feature extraction, structural decomposition,
IP detection, punycode handling, schema consistency, deterministic output,
domain-isolated splitting, and overlap purging using synthetic fixtures only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.feature_config import (
    FEATURE_DTYPES,
    FEATURE_NAMES,
    SUSPICIOUS_KEYWORDS,
    TOTAL_FEATURE_COUNT,
)
from src.features.feature_extractor import (
    extract_features_dataframe,
    extract_url_features,
    extract_url_metadata,
)
from src.features.lexical_features import (
    compute_max_consecutive_repeated_char,
    compute_shannon_entropy,
    extract_lexical_features,
)
from src.features.structural_features import extract_structural_features
from src.features.url_parser import (
    check_ip_representation,
    normalize_url,
    parse_url,
)


@pytest.fixture
def sample_synthetic_urls() -> list[str]:
    """Collection of diverse synthetic URLs covering varied structural types."""
    return [
        "https://login.example.com/auth/verify?session=123&token=abc#step1",
        "http://192.168.1.1:8080/admin/index.php",
        "http://http://sub.double-prefix.co.uk/path/file.exe",
        "ttp://malformed-scheme.org/download",
        "https://xn--e1afmkfd.xn--p1ai/test",  # Punycode IDN
        "https://0x7f000001/hex-ip",           # Hex IP
        "https://portal.bank.com/secure/banking/payment/login",
        "",                                    # Empty URL
        "just-a-plain-string",                 # Missing protocol
        "https://user:password@spoofed.com@actual.com/path",  # Userinfo
    ]


def test_url_normalization():
    """Verify deterministic scheme correction and prefix collapse."""
    u1, m1 = normalize_url("https://example.com")
    assert u1 == "https://example.com"
    assert m1 is False

    # Duplicated protocol
    u2, m2 = normalize_url("http://http://example.com")
    assert u2 == "http://example.com"
    assert m2 is False

    # Typo schemes
    u3, m3 = normalize_url("ttp://example.com")
    assert u3 == "http://example.com"
    assert m3 is False

    u4, m4 = normalize_url("ttps://example.com")
    assert u4 == "https://example.com"
    assert m4 is False

    # Missing protocol
    u5, m5 = normalize_url("example.com/login")
    assert u5 == "http://example.com/login"
    assert m5 is True

    # Empty string
    u6, m6 = normalize_url("")
    assert u6 == ""
    assert m6 is True


def test_ip_address_detection():
    """Verify standard and obfuscated IP address detection."""
    # Standard IPv4 and IPv6
    is_ip, is_unusual = check_ip_representation("192.168.1.1")
    assert is_ip is True and is_unusual is False

    is_ip_v6, _ = check_ip_representation("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
    assert is_ip_v6 is True

    # Hex IP
    is_hex, is_hex_unusual = check_ip_representation("0x7f000001")
    assert is_hex is True and is_hex_unusual is True

    # DWORD integer IP
    is_dword, is_dword_unusual = check_ip_representation("2130706433")
    assert is_dword is True and is_dword_unusual is True

    # Octal IP
    is_oct, is_oct_unusual = check_ip_representation("0177.0.0.1")
    assert is_oct is True and is_oct_unusual is True

    # Non-IP host
    is_dom, is_dom_unusual = check_ip_representation("www.google.com")
    assert is_dom is False and is_dom_unusual is False


def test_shannon_entropy_calculation():
    """Verify Shannon entropy computation."""
    assert compute_shannon_entropy("") == 0.0
    assert compute_shannon_entropy("aaaa") == 0.0  # Zero uncertainty
    entropy_mixed = compute_shannon_entropy("abcdefgh")
    assert entropy_mixed > 2.5


def test_max_consecutive_repeated_characters():
    """Verify calculation of max run of consecutive identical characters."""
    assert compute_max_consecutive_repeated_char("") == 0
    assert compute_max_consecutive_repeated_char("a") == 1
    assert compute_max_consecutive_repeated_char("https://example.com/aaaaa/bb") == 5


def test_punycode_and_userinfo_detection():
    """Verify detection of punycode domains and authority userinfo."""
    p1 = parse_url("https://xn--e1afmkfd.xn--p1ai/home")
    assert p1.has_punycode is True

    p2 = parse_url("https://user:pass@legit.com/auth")
    assert p2.userinfo == "user:pass"
    assert extract_structural_features(p2)["has_userinfo"] == 1


def test_feature_schema_and_types(sample_synthetic_urls):
    """Verify that feature extraction produces the exact 68 features with target dtypes."""
    for u in sample_synthetic_urls:
        feats = extract_url_features(u)
        assert len(feats) == TOTAL_FEATURE_COUNT
        assert list(feats.keys()) == FEATURE_NAMES

        for name, expected_dtype in FEATURE_DTYPES.items():
            val = feats[name]
            if expected_dtype == "int64":
                assert isinstance(val, int), f"{name} should be int, got {type(val)}"
            elif expected_dtype == "float64":
                assert isinstance(val, (float, int)), f"{name} should be float, got {type(val)}"


def test_deterministic_extraction():
    """Verify that repeated extraction on the same URL produces identical results."""
    test_url = "https://secure-banking-login.example.co.uk:443/verify?user=alice&id=99#frag"
    f1 = extract_url_features(test_url)
    f2 = extract_url_features(test_url)
    assert f1 == f2


def test_suspicious_keyword_features():
    """Verify detection and aggregation of targeted security keywords."""
    url = "https://login-verify-account-update.com/signin/secure/banking/payment/password/confirm/authentication"
    feats = extract_url_features(url)

    for kw in SUSPICIOUS_KEYWORDS:
        assert feats[f"has_kw_{kw}"] == 1

    assert feats["suspicious_keyword_count"] == len(SUSPICIOUS_KEYWORDS)


def test_path_file_extension_features():
    """Verify path extension count and binary indicator."""
    u_exe = "https://example.com/downloads/setup.exe"
    f_exe = extract_url_features(u_exe)
    assert f_exe["path_has_file_extension"] == 1
    assert f_exe["path_file_extension_count"] == 1

    u_double = "https://example.com/invoice.pdf.exe"
    f_double = extract_url_features(u_double)
    assert f_double["path_has_file_extension"] == 1
    assert f_double["path_file_extension_count"] == 2

    u_clean = "https://example.com/about/team"
    f_clean = extract_url_features(u_clean)
    assert f_clean["path_has_file_extension"] == 0
    assert f_clean["path_file_extension_count"] == 0


def test_extract_features_dataframe(sample_synthetic_urls):
    """Verify batch extraction into DataFrame."""
    df = extract_features_dataframe(sample_synthetic_urls)
    assert len(df) == len(sample_synthetic_urls)
    assert df.shape[1] == TOTAL_FEATURE_COUNT
    assert df.isnull().sum().sum() == 0


def test_extract_url_metadata():
    """Verify metadata separation and duplicate group tagging."""
    synthetic_raw_df = pd.DataFrame({
        "URL": [
            "https://unique.com/1",
            "https://duplicate.com/page",
            "https://duplicate.com/page",  # duplicate URL
            "https://no-label.com",
        ],
        "label": [1, 0, 0, None],  # One missing label
    })

    feat_df, meta_df = extract_url_metadata(
        synthetic_raw_df, url_col="URL", label_col="label", dataset_source="TestDataset"
    )

    # Missing label row dropped
    assert len(feat_df) == 3
    assert len(meta_df) == 3

    # Metadata columns check
    expected_meta_cols = [
        "original_row_id", "raw_url", "normalized_url", "registrable_domain",
        "dataset_source", "original_label", "is_valid_url", "duplicate_group_id",
        "is_duplicate_url", "duplicate_occurrence_count"
    ]
    for col in expected_meta_cols:
        assert col in meta_df.columns

    # Duplicate detection in metadata
    dup_row = meta_df[meta_df["raw_url"] == "https://duplicate.com/page"]
    assert len(dup_row) == 2
    assert (dup_row["is_duplicate_url"] == 1).all()
    assert (dup_row["duplicate_occurrence_count"] == 2).all()
    assert dup_row["duplicate_group_id"].iloc[0] == dup_row["duplicate_group_id"].iloc[1]


def test_domain_isolated_split_simulation():
    """Verify that domain-grouped splitting assigns all URLs from a domain to one split."""
    meta = pd.DataFrame({
        "registrable_domain": ["a.com", "a.com", "b.org", "c.net", "d.io", "d.io"],
        "raw_url": ["u1", "u2", "u3", "u4", "u5", "u6"],
    })

    unique_doms = meta["registrable_domain"].unique()
    train_doms = set(unique_doms[:2])
    test_doms = set(unique_doms[2:])

    train_meta = meta[meta["registrable_domain"].isin(train_doms)]
    test_meta = meta[meta["registrable_domain"].isin(test_doms)]

    # Disjoint domain verification
    overlap = set(train_meta["registrable_domain"]).intersection(set(test_meta["registrable_domain"]))
    assert len(overlap) == 0


def test_external_overlap_purging_simulation():
    """Verify external validation purge simulation excludes overlapping domains."""
    train_domains = {"example.com", "bank.co.uk", "portal.org"}
    external_meta = pd.DataFrame({
        "raw_url": ["u1", "u2", "u3", "u4"],
        "registrable_domain": ["example.com", "unseen-site.de", "new-phish.net", "portal.org"],
        "label": [1, 1, 0, 0],
    })

    purged_mask = ~external_meta["registrable_domain"].isin(train_domains)
    purged_external = external_meta[purged_mask]

    assert len(purged_external) == 2
    assert "example.com" not in purged_external["registrable_domain"].values
    assert "portal.org" not in purged_external["registrable_domain"].values
    assert set(purged_external["registrable_domain"]) == {"unseen-site.de", "new-phish.net"}

