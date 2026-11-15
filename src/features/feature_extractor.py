"""High-level URL feature extraction and dataset preparation pipeline.

Coordinates offline URL parsing, lexical feature calculation, structural
feature derivation, schema enforcement, and metadata separation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.data.load_data import compute_file_sha256
from src.features.feature_config import (
    FEATURE_DTYPES,
    FEATURE_NAMES,
    TOTAL_FEATURE_COUNT,
)
from src.features.lexical_features import extract_lexical_features
from src.features.structural_features import extract_structural_features
from src.features.url_parser import ParsedURL, parse_url


def extract_url_features(raw_url: str) -> Dict[str, Any]:
    """Extract complete 68-feature dictionary for a single URL string.

    Args:
        raw_url: The uncleaned URL string.

    Returns:
        Ordered dictionary with all 68 features cast to target dtypes.
    """
    parsed = parse_url(raw_url)
    lex = extract_lexical_features(parsed)
    struct = extract_structural_features(parsed)

    # Merge dictionaries
    merged = {**lex, **struct}

    # Enforce strict ordering and type casting
    result = {}
    for name in FEATURE_NAMES:
        val = merged[name]
        dtype = FEATURE_DTYPES[name]
        if dtype == "int64":
            result[name] = int(val)
        elif dtype == "float64":
            result[name] = float(val)
        else:
            result[name] = val

    return result


def extract_features_dataframe(urls: Iterable[str]) -> pd.DataFrame:
    """Extract features for an iterable of URLs into an ordered, typed DataFrame.

    Args:
        urls: Iterable of URL strings.

    Returns:
        pd.DataFrame with exact FEATURE_NAMES columns and FEATURE_DTYPES types.
    """
    records = [extract_url_features(u) for u in urls]
    df = pd.DataFrame(records, columns=FEATURE_NAMES)
    return df.astype(FEATURE_DTYPES)


def extract_url_metadata(
    df: pd.DataFrame,
    url_col: str,
    label_col: str,
    dataset_source: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separate raw DataFrame into a predictive feature DataFrame and an isolated metadata DataFrame.

    Handles:
    - Exclusion of rows with missing labels.
    - Identification of duplicate URL groups.
    - Offline registrable domain extraction for metadata tracking.

    Args:
        df: Input raw DataFrame.
        url_col: Column name containing raw URLs.
        label_col: Column name containing target labels.
        dataset_source: Name of the dataset (e.g. 'LegitPhish', 'PhiUSIIL').

    Returns:
        Tuple of (feature_df_with_label, metadata_df).
    """
    total_raw = len(df)

    # 1. Check for missing labels
    valid_mask = df[label_col].notna()
    clean_df = df[valid_mask].copy().reset_index()
    # clean_df['original_row_id'] holds the original row index in raw file
    clean_df = clean_df.rename(columns={"index": "original_row_id"})

    urls = clean_df[url_col].astype(str).tolist()
    labels = clean_df[label_col].astype(int).tolist()

    # 2. Parse URLs for metadata
    parsed_list: List[ParsedURL] = [parse_url(u) for u in urls]

    # 3. Build metadata table
    meta_records = []
    for row_id, parsed, lbl in zip(clean_df["original_row_id"], parsed_list, labels):
        meta_records.append({
            "original_row_id": int(row_id),
            "raw_url": parsed.raw_url,
            "normalized_url": parsed.normalized_url,
            "registrable_domain": parsed.registrable_domain,
            "dataset_source": dataset_source,
            "original_label": int(lbl),
            "is_valid_url": int(parsed.parse_success and bool(parsed.hostname)),
        })
    metadata_df = pd.DataFrame(meta_records)

    # Compute duplicate group identifiers in metadata
    # A duplicate group ID assigns the same integer to all instances of the identical raw URL string
    url_to_dup_id = {u: idx for idx, u in enumerate(metadata_df["raw_url"].unique())}
    metadata_df["duplicate_group_id"] = metadata_df["raw_url"].map(url_to_dup_id)

    # Count occurrences in duplicate group
    dup_counts = metadata_df["raw_url"].map(metadata_df["raw_url"].value_counts())
    metadata_df["is_duplicate_url"] = (dup_counts > 1).astype(int)
    metadata_df["duplicate_occurrence_count"] = dup_counts

    # 4. Build feature matrix
    print(f"Extracting 68 URL-only features for {len(urls):,} records ({dataset_source})...")
    feat_records = []
    for parsed in parsed_list:
        lex = extract_lexical_features(parsed)
        struct = extract_structural_features(parsed)
        merged = {**lex, **struct}
        feat_records.append(merged)

    features_df = pd.DataFrame(feat_records, columns=FEATURE_NAMES).astype(FEATURE_DTYPES)
    features_df["label"] = labels

    return features_df, metadata_df


def process_and_save_dataset(
    raw_df: pd.DataFrame,
    url_col: str,
    label_col: str,
    dataset_name: str,
    output_dir: Path | str,
    manifest_path: Path | str,
) -> Dict[str, Any]:
    """Process a raw dataset, separate features from metadata, and save CSVs and manifest.

    Args:
        raw_df: Input raw DataFrame.
        url_col: Column name of raw URL.
        label_col: Column name of original label.
        dataset_name: Dataset identifier.
        output_dir: Directory where processed CSVs will be saved.
        manifest_path: Path where JSON manifest will be written.

    Returns:
        Manifest summary dictionary.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = Path(manifest_path)
    manifest_p.parent.mkdir(parents=True, exist_ok=True)

    features_df, metadata_df = extract_url_metadata(
        df=raw_df,
        url_col=url_col,
        label_col=label_col,
        dataset_source=dataset_name,
    )

    features_csv_path = out_dir / "url_features.csv"
    metadata_csv_path = out_dir / "metadata.csv"

    print(f"Saving features to {features_csv_path}...")
    features_df.to_csv(features_csv_path, index=False)

    print(f"Saving metadata to {metadata_csv_path}...")
    metadata_df.to_csv(metadata_csv_path, index=False)

    # Compute checksums
    feat_sha = compute_file_sha256(features_csv_path)
    meta_sha = compute_file_sha256(metadata_csv_path)

    total_raw = len(raw_df)
    total_processed = len(features_df)
    dropped_missing_labels = total_raw - total_processed
    dup_url_strings = int((metadata_df["is_duplicate_url"] == 1).sum())
    unique_urls = int(metadata_df["raw_url"].nunique())
    unique_domains = int(metadata_df["registrable_domain"][metadata_df["registrable_domain"] != ""].nunique())

    manifest = {
        "dataset_name": dataset_name,
        "total_raw_records": total_raw,
        "total_processed_records": total_processed,
        "records_excluded_missing_labels": dropped_missing_labels,
        "total_features": TOTAL_FEATURE_COUNT,
        "feature_names": FEATURE_NAMES,
        "unique_url_strings": unique_urls,
        "duplicate_url_records": dup_url_strings,
        "unique_registrable_domains": unique_domains,
        "class_distribution": {
            "phishing_0": int((features_df["label"] == 0).sum()),
            "legitimate_1": int((features_df["label"] == 1).sum()),
        },
        "output_files": {
            "features_csv": {
                "path": str(features_csv_path),
                "size_bytes": features_csv_path.stat().st_size,
                "sha256": feat_sha,
                "rows": len(features_df),
                "columns": len(features_df.columns),
            },
            "metadata_csv": {
                "path": str(metadata_csv_path),
                "size_bytes": metadata_csv_path.stat().st_size,
                "sha256": meta_sha,
                "rows": len(metadata_df),
                "columns": len(metadata_df.columns),
            },
        },
    }

    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {manifest_p}.")

    return manifest
