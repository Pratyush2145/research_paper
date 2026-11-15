"""Unit tests for dataset auditing, loading, validation, and domain parsing.

All tests utilize small synthetic test data only to ensure tests remain
fast, reproducible, and isolated without modifying actual research data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.dataset_audit import audit_cross_dataset, audit_single_dataset
from src.data.load_data import (
    compute_file_sha256,
    load_legitphish,
    load_phiusiil,
    verify_dataset_integrity,
)
from src.data.preprocess import (
    extract_hostname,
    extract_registrable_domain,
    identify_conflicting_label_domains,
    identify_conflicting_label_urls,
    identify_duplicate_urls,
    normalize_url_scheme,
    parse_url_components,
)


@pytest.fixture
def synthetic_url_dataframe() -> pd.DataFrame:
    """Fixture providing a controlled synthetic dataset."""
    return pd.DataFrame({
        "URL": [
            "https://login.example.com/auth?token=123#sec",
            "http://example.com/login",
            "https://portal.bank.co.uk/verify",
            "http://192.168.1.50/update.exe",
            "https://phish.example.com/steal",
            "https://login.example.com/auth?token=123#sec",  # Duplicate URL and row
            "https://singletoken",                           # Invalid URL
            "",                                              # Empty URL
            "https://legit-site.org/home",
            "https://conflict-test.com/a",
            "https://conflict-test.com/b",
        ],
        "label": [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    })


def test_normalize_url_scheme():
    """Verify offline scheme normalization and prefix collapse."""
    assert normalize_url_scheme("http://example.com") == "http://example.com"
    assert normalize_url_scheme("https://example.com") == "https://example.com"
    assert normalize_url_scheme("http://http://example.com") == "http://example.com"
    assert normalize_url_scheme("ttp://example.com") == "http://example.com"
    assert normalize_url_scheme("ttps://example.com") == "https://example.com"
    assert normalize_url_scheme("example.com/page") == "http://example.com/page"
    assert normalize_url_scheme("") == ""


def test_extract_hostname_and_registrable_domain():
    """Verify accurate extraction of hosts, registrable domains, ccTLDs, and IPs."""
    # Standard domains
    assert extract_hostname("https://www.example.com/path") == "www.example.com"
    assert extract_registrable_domain("https://www.example.com/path") == "example.com"

    # Multi-part ccTLDs (.co.uk, .gov.in)
    assert extract_registrable_domain("https://sub.portal.bank.co.uk") == "bank.co.uk"
    assert extract_registrable_domain("http://nic.gov.in/portal") == "nic.gov.in"

    # IP address literals
    assert extract_registrable_domain("http://192.168.1.1:8080/index") == "192.168.1.1"

    # Malformed or unparseable strings
    assert extract_registrable_domain("Offline") == ""
    assert extract_registrable_domain("") == ""
    assert extract_registrable_domain("https://") == ""


def test_parse_url_components():
    """Verify structural URL decomposition without network access."""
    res = parse_url_components("https://sub.bank.co.uk:443/login/auth?user=test#header")
    assert res["scheme"] == "https"
    assert res["hostname"] == "sub.bank.co.uk"
    assert res["registrable_domain"] == "bank.co.uk"
    assert res["path"] == "/login/auth"
    assert res["query"] == "user=test"
    assert res["fragment"] == "header"
    assert res["has_subdomain"] is True
    assert res["has_path"] is True
    assert res["has_query"] is True
    assert res["has_fragment"] is True
    assert res["is_valid"] is True


def test_missing_and_empty_url_detection(synthetic_url_dataframe):
    """Verify detection of missing, whitespace, and empty URL records."""
    df = synthetic_url_dataframe.copy()
    urls = df["URL"].fillna("").astype(str).str.strip()
    empty_count = (urls == "").sum()
    assert empty_count == 1


def test_duplicate_url_detection(synthetic_url_dataframe):
    """Verify identification of duplicate URL records."""
    dup_df = identify_duplicate_urls(synthetic_url_dataframe, url_col="URL")
    assert len(dup_df) == 2
    assert "https://login.example.com/auth?token=123#sec" in dup_df["URL"].values


def test_conflicting_label_detection():
    """Verify identification of URLs and domains with conflicting labels."""
    df = pd.DataFrame({
        "URL": [
            "https://conflict.com/page1",
            "https://conflict.com/page1",  # Same URL, different label
            "https://test.org/good",
            "https://sub.domain-conflict.com/1",
            "https://domain-conflict.com/2",
        ],
        "registrable_domain": [
            "conflict.com",
            "conflict.com",
            "test.org",
            "domain-conflict.com",
            "domain-conflict.com",
        ],
        "label": [1, 0, 1, 0, 1],
    })

    conflicting_urls = identify_conflicting_label_urls(df, url_col="URL", label_col="label")
    assert len(conflicting_urls) == 2
    assert conflicting_urls["URL"].iloc[0] == "https://conflict.com/page1"

    conflicting_doms = identify_conflicting_label_domains(
        df, domain_col="registrable_domain", label_col="label"
    )
    assert "conflict.com" in conflicting_doms
    assert "domain-conflict.com" in conflicting_doms
    assert conflicting_doms["conflict.com"] == {1: 1, 0: 1}


def test_dataset_audit_report_generation(synthetic_url_dataframe):
    """Verify that audit_single_dataset produces complete and valid metrics."""
    audit = audit_single_dataset(
        df=synthetic_url_dataframe,
        dataset_name="SyntheticTest",
        url_col="URL",
        label_col="label",
    )

    assert audit["dataset_name"] == "SyntheticTest"
    assert audit["total_rows"] == 11
    assert audit["total_cols"] == 2
    assert audit["empty_url_count"] == 1
    assert audit["label_distribution"]["phishing_0"] == 5
    assert audit["label_distribution"]["legitimate_1"] == 6
    assert audit["domain_analysis"]["unique_registrable_domains"] > 0
    assert "example.com" in audit["domain_analysis"]["top_repeated_domains"]


def test_cross_dataset_audit():
    """Verify cross-dataset domain and URL overlap computation."""
    df_a = pd.DataFrame({
        "URL": ["https://site1.com/a", "https://shared.com/page", "https://only-a.com"],
        "label": [1, 1, 0],
    })
    df_b = pd.DataFrame({
        "URL": ["https://site2.com/b", "https://shared.com/page", "https://sub.shared.com/diff"],
        "label": [0, 1, 0],
    })

    audit_a = audit_single_dataset(df_a, "A", "URL", "label")
    audit_b = audit_single_dataset(df_b, "B", "URL", "label")

    cross = audit_cross_dataset(
        audit_a, audit_b, df_a, df_b,
        url_col_a="URL", label_col_a="label",
        url_col_b="URL", label_col_b="label"
    )

    assert cross["exact_url_overlap_count"] == 1
    assert "https://shared.com/page" in cross["exact_url_overlap_sample"]
    assert cross["domain_overlap_count"] == 1  # shared.com
    assert cross["cross_dataset_label_divergence_count"] == 1  # shared.com is 1 in A, but [1, 0] in B


def test_file_integrity_and_sha256_computation():
    """Verify SHA-256 computation and integrity check on a temporary file."""
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("URL,ClassLabel\nhttps://example.com,1\n")
        temp_path = f.name

    try:
        sha = compute_file_sha256(temp_path)
        assert len(sha) == 64
        valid, computed = verify_dataset_integrity(temp_path, sha)
        assert valid is True
        assert computed == sha

        invalid, _ = verify_dataset_integrity(temp_path, "0" * 64)
        assert invalid is False
    finally:
        Path(temp_path).unlink()


def test_dataset_loaders_url_only_structure():
    """Verify load_legitphish and load_phiusiil correctly return URL-only subsets."""
    # Test LegitPhish loader
    df_l = load_legitphish(url_only=True)
    assert list(df_l.columns) == ["URL", "ClassLabel"]
    assert len(df_l) == 101219

    # Test PhiUSIIL loader
    df_p = load_phiusiil(url_only=True)
    assert list(df_p.columns) == ["URL", "label"]
    assert len(df_p) == 235795

