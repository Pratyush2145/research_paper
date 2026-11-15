"""Dataset audit and validation module for phishing detection research.

Performs offline dataset inspection, quality assessment, label distribution analysis,
URL component profiling, domain extraction, and cross-dataset leakage analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.data.load_data import (
    LEGITPHISH_CSV,
    LEGITPHISH_EXPECTED_SHA256,
    PHIUSIIL_CSV,
    PHIUSIIL_CSV_EXPECTED_SHA256,
    compute_file_sha256,
    load_legitphish,
    load_phiusiil,
)
from src.data.preprocess import (
    extract_hostname,
    extract_registrable_domain,
    identify_conflicting_label_domains,
    identify_conflicting_label_urls,
    parse_url_components,
)


def audit_single_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    url_col: str,
    label_col: str,
    file_path: Optional[Path | str] = None,
    expected_sha256: Optional[str] = None,
    reported_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Perform a comprehensive audit of an individual dataset.

    Inspects basic info, label distributions, data quality, URL structure,
    and domain-level representations strictly offline.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    column_names = list(df.columns)
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # File information
    file_size_bytes = 0
    actual_sha256 = ""
    sha256_match = None
    if file_path and Path(file_path).exists():
        path = Path(file_path)
        file_size_bytes = path.stat().st_size
        actual_sha256 = compute_file_sha256(path)
        if expected_sha256:
            sha256_match = actual_sha256.lower() == expected_sha256.lower()

    # Missing values
    missing_by_col = df.isnull().sum().to_dict()
    total_missing = sum(missing_by_col.values())
    cols_with_missing = {k: v for k, v in missing_by_col.items() if v > 0}

    # URL presence and basic validation
    urls = df[url_col].fillna("").astype(str).str.strip()
    empty_url_count = int((urls == "").sum())

    # Label distribution
    label_series = df[label_col]
    label_val_counts = label_series.value_counts(dropna=False).to_dict()
    label_pcts = (label_series.value_counts(normalize=True, dropna=False) * 100).to_dict()

    # Normalize counts of standard labels 0 and 1
    phishing_count = int(label_val_counts.get(0, label_val_counts.get(0.0, 0)))
    legit_count = int(label_val_counts.get(1, label_val_counts.get(1.0, 0)))
    nan_label_count = int(label_series.isna().sum())

    # Unexpected labels
    unexpected_labels = {}
    for val, count in label_val_counts.items():
        if pd.isna(val):
            continue
        if val not in (0, 1, 0.0, 1.0):
            unexpected_labels[str(val)] = int(count)

    # Data Quality: Duplicates
    exact_duplicates = int(df.duplicated().sum())
    duplicate_urls_count = int(df[url_col].duplicated().sum())

    # Conflicting URLs (same URL string with multiple labels)
    conflicting_urls_df = identify_conflicting_label_urls(df, url_col=url_col, label_col=label_col)
    conflicting_urls_count = int(conflicting_urls_df[url_col].nunique())

    # URL Component Profiling & Domain Extraction
    parsed_components = [parse_url_components(u) for u in urls]
    comp_df = pd.DataFrame(parsed_components)

    schemes_dist = comp_df["scheme"].value_counts().to_dict()
    subdomains_count = int(comp_df["has_subdomain"].sum())
    paths_count = int(comp_df["has_path"].sum())
    queries_count = int(comp_df["has_query"].sum())
    fragments_count = int(comp_df["has_fragment"].sum())
    invalid_url_count = int((~comp_df["is_valid"]).sum())

    # Sample URLs
    sample_urls = df[url_col].head(5).tolist()

    # Domain Analysis
    registrable_domains = comp_df["registrable_domain"]
    non_empty_domains = registrable_domains[registrable_domains != ""]
    unique_domains_count = int(non_empty_domains.nunique())

    domain_counts = non_empty_domains.value_counts()
    repeated_domains_count = int((domain_counts > 1).sum())
    top_repeated_domains = {k: int(v) for k, v in domain_counts.head(10).items()}

    # Conflicting domains (same domain with both phishing and legitimate records)
    temp_df = pd.DataFrame({
        "registrable_domain": registrable_domains,
        "label": label_series,
    })
    conflicting_domains = identify_conflicting_label_domains(
        temp_df, domain_col="registrable_domain", label_col="label"
    )

    # Discrepancies with reported
    discrepancies: List[str] = []
    if reported_stats:
        if "rows" in reported_stats and reported_stats["rows"] != total_rows:
            discrepancies.append(
                f"Row count mismatch: reported {reported_stats['rows']}, found {total_rows}."
            )
        if "phishing" in reported_stats and reported_stats["phishing"] != phishing_count:
            discrepancies.append(
                f"Phishing count mismatch: reported {reported_stats['phishing']}, found {phishing_count}."
            )
        if "legitimate" in reported_stats and reported_stats["legitimate"] != legit_count:
            discrepancies.append(
                f"Legitimate count mismatch: reported {reported_stats['legitimate']}, found {legit_count}."
            )
        if "cols" in reported_stats and reported_stats["cols"] != total_cols:
            discrepancies.append(
                f"Column count mismatch: reported {reported_stats['cols']}, found {total_cols}."
            )

    return {
        "dataset_name": dataset_name,
        "file_path": str(file_path) if file_path else "",
        "file_size_bytes": file_size_bytes,
        "sha256": actual_sha256,
        "sha256_match": sha256_match,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "columns": column_names,
        "dtypes": dtypes_dict,
        "missing_values": cols_with_missing,
        "total_missing": total_missing,
        "empty_url_count": empty_url_count,
        "invalid_url_count": invalid_url_count,
        "label_distribution": {
            "phishing_0": phishing_count,
            "phishing_pct": (phishing_count / total_rows) * 100 if total_rows else 0.0,
            "legitimate_1": legit_count,
            "legitimate_pct": (legit_count / total_rows) * 100 if total_rows else 0.0,
            "nan_labels": nan_label_count,
            "unexpected_labels": unexpected_labels,
        },
        "data_quality": {
            "exact_duplicate_rows": exact_duplicates,
            "duplicate_urls": duplicate_urls_count,
            "conflicting_label_urls": conflicting_urls_count,
        },
        "url_profiling": {
            "sample_urls": sample_urls,
            "schemes": schemes_dist,
            "has_subdomain_count": subdomains_count,
            "has_subdomain_pct": (subdomains_count / total_rows) * 100 if total_rows else 0.0,
            "has_path_count": paths_count,
            "has_path_pct": (paths_count / total_rows) * 100 if total_rows else 0.0,
            "has_query_count": queries_count,
            "has_query_pct": (queries_count / total_rows) * 100 if total_rows else 0.0,
            "has_fragment_count": fragments_count,
            "has_fragment_pct": (fragments_count / total_rows) * 100 if total_rows else 0.0,
        },
        "domain_analysis": {
            "unique_registrable_domains": unique_domains_count,
            "repeated_domains_count": repeated_domains_count,
            "top_repeated_domains": top_repeated_domains,
            "conflicting_label_domains_count": len(conflicting_domains),
            "conflicting_domains_sample": {
                k: conflicting_domains[k] for k in list(conflicting_domains.keys())[:10]
            },
        },
        "discrepancies": discrepancies,
    }


def audit_cross_dataset(
    audit_a: Dict[str, Any],
    audit_b: Dict[str, Any],
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    url_col_a: str = "URL",
    label_col_a: str = "ClassLabel",
    url_col_b: str = "URL",
    label_col_b: str = "label",
) -> Dict[str, Any]:
    """Perform cross-dataset leakage and overlap audit."""
    urls_a = set(df_a[url_col_a].dropna().astype(str).str.strip().unique())
    urls_b = set(df_b[url_col_b].dropna().astype(str).str.strip().unique())
    exact_url_overlap = urls_a.intersection(urls_b)

    # Domain extraction
    doms_a = df_a[url_col_a].dropna().apply(extract_registrable_domain)
    doms_b = df_b[url_col_b].dropna().apply(extract_registrable_domain)

    set_doms_a = set(doms_a[doms_a != ""].unique())
    set_doms_b = set(doms_b[doms_b != ""].unique())
    domain_overlap = set_doms_a.intersection(set_doms_b)

    # Check cross-dataset label divergence
    # Map domain to set of labels in dataset A
    temp_a = pd.DataFrame({"domain": doms_a, "label": df_a[label_col_a]}).dropna()
    temp_a = temp_a[temp_a["domain"].isin(domain_overlap)]
    dom_labels_a = temp_a.groupby("domain")["label"].unique().to_dict()

    temp_b = pd.DataFrame({"domain": doms_b, "label": df_b[label_col_b]}).dropna()
    temp_b = temp_b[temp_b["domain"].isin(domain_overlap)]
    dom_labels_b = temp_b.groupby("domain")["label"].unique().to_dict()

    divergent_domains = {}
    for d in domain_overlap:
        labels_in_a = set(dom_labels_a.get(d, []))
        labels_in_b = set(dom_labels_b.get(d, []))
        if labels_in_a and labels_in_b and labels_in_a != labels_in_b:
            divergent_domains[d] = {
                f"{audit_a['dataset_name']}_labels": [float(x) for x in labels_in_a],
                f"{audit_b['dataset_name']}_labels": [float(x) for x in labels_in_b],
            }

    return {
        "dataset_a": audit_a["dataset_name"],
        "dataset_b": audit_b["dataset_name"],
        "exact_url_overlap_count": len(exact_url_overlap),
        "exact_url_overlap_sample": list(exact_url_overlap)[:10],
        "domain_overlap_count": len(domain_overlap),
        "domain_overlap_pct_of_a": (len(domain_overlap) / len(set_doms_a)) * 100 if set_doms_a else 0.0,
        "domain_overlap_pct_of_b": (len(domain_overlap) / len(set_doms_b)) * 100 if set_doms_b else 0.0,
        "cross_dataset_label_divergence_count": len(divergent_domains),
        "cross_dataset_label_divergence_sample": {
            k: divergent_domains[k] for k in list(divergent_domains.keys())[:10]
        },
    }


def run_full_audit() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Execute full audit of both research datasets."""
    print("Loading LegitPhish...")
    df_legit = load_legitphish(url_only=False)
    audit_legit = audit_single_dataset(
        df=df_legit,
        dataset_name="LegitPhish",
        url_col="URL",
        label_col="ClassLabel",
        file_path=LEGITPHISH_CSV,
        expected_sha256=LEGITPHISH_EXPECTED_SHA256,
        reported_stats={"rows": 101219, "cols": 18, "phishing": 63678, "legitimate": 37541},
    )

    print("Loading PhiUSIIL...")
    df_phi = load_phiusiil(url_only=False)
    audit_phi = audit_single_dataset(
        df=df_phi,
        dataset_name="PhiUSIIL",
        url_col="URL",
        label_col="label",
        file_path=PHIUSIIL_CSV,
        expected_sha256=PHIUSIIL_CSV_EXPECTED_SHA256,
        reported_stats={"rows": 235795, "cols": 56, "phishing": 100945, "legitimate": 134850},
    )

    print("Performing cross-dataset audit...")
    cross_audit = audit_cross_dataset(
        audit_a=audit_legit,
        audit_b=audit_phi,
        df_a=df_legit,
        df_b=df_phi,
        url_col_a="URL",
        label_col_a="ClassLabel",
        url_col_b="URL",
        label_col_b="label",
    )

    return audit_legit, audit_phi, cross_audit


if __name__ == "__main__":
    a_l, a_p, c_aud = run_full_audit()
    print("=== AUDIT SUMMARY ===")
    print(f"LegitPhish: {a_l['total_rows']} rows, {a_l['total_cols']} cols, missing: {a_l['total_missing']}")
    print(f"PhiUSIIL: {a_p['total_rows']} rows, {a_p['total_cols']} cols, missing: {a_p['total_missing']}")
    print(f"Domain overlap: {c_aud['domain_overlap_count']}")

