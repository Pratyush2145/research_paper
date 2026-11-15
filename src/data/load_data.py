"""Data loading module for phishing detection research datasets.

Provides reproducible and verified loaders for:
1. LegitPhish (Primary Dataset from Mendeley Data)
2. PhiUSIIL (External Validation Dataset from UCI Machine Learning Repository)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

LEGITPHISH_DIR = RAW_DATA_DIR / "legitphish"
LEGITPHISH_CSV = LEGITPHISH_DIR / "url_features_extracted1.csv"
LEGITPHISH_EXPECTED_SHA256 = (
    "8685e7901c3b4e73eb43c106cb166cc932dd14f76960a032c713c2c32f09dbcd"
)

PHIUSIIL_DIR = RAW_DATA_DIR / "phiusil"
PHIUSIIL_CSV = PHIUSIIL_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
PHIUSIIL_ZIP = PHIUSIIL_DIR / "phiusiil+phishing+url+dataset.zip"
PHIUSIIL_CSV_EXPECTED_SHA256 = (
    "a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6"
)
PHIUSIIL_ZIP_EXPECTED_SHA256 = (
    "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a"
)


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute the SHA-256 hash of a file.

    Args:
        filepath: Path to the target file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_dataset_integrity(filepath: Path | str, expected_sha256: str) -> Tuple[bool, str]:
    """Verify that a dataset file matches its expected SHA-256 checksum.

    Args:
        filepath: Path to the target file.
        expected_sha256: Expected hexadecimal SHA-256 string.

    Returns:
        Tuple of (matches_bool, actual_sha256_string).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    actual_sha = compute_file_sha256(path)
    return actual_sha.lower() == expected_sha256.lower(), actual_sha


def load_legitphish(
    filepath: Optional[Path | str] = None,
    url_only: bool = False,
    verify_checksum: bool = False,
) -> pd.DataFrame:
    """Load the raw LegitPhish dataset.

    Reported schema:
        101,219 records, 18 columns.
        ClassLabel: 0 = Phishing, 1 = Legitimate.

    Args:
        filepath: Optional custom path to the CSV file. Defaults to RAW_LEGITPHISH_CSV.
        url_only: If True, returns only ['URL', 'ClassLabel'].
        verify_checksum: If True, checks file against LEGITPHISH_EXPECTED_SHA256.

    Returns:
        pd.DataFrame containing the requested dataset columns.
    """
    target_path = Path(filepath) if filepath else LEGITPHISH_CSV
    if not target_path.exists():
        raise FileNotFoundError(
            f"LegitPhish raw file not found at {target_path}. "
            "Please ensure dataset acquisition is complete."
        )

    if verify_checksum:
        is_valid, actual_sha = verify_dataset_integrity(target_path, LEGITPHISH_EXPECTED_SHA256)
        if not is_valid:
            raise ValueError(
                f"LegitPhish SHA-256 mismatch! Expected {LEGITPHISH_EXPECTED_SHA256}, "
                f"got {actual_sha}."
            )

    usecols = ["URL", "ClassLabel"] if url_only else None
    return pd.read_csv(target_path, usecols=usecols)


def load_phiusiil(
    filepath: Optional[Path | str] = None,
    url_only: bool = True,
    verify_checksum: bool = False,
) -> pd.DataFrame:
    """Load the raw PhiUSIIL dataset.

    Reported schema:
        235,795 records, 56 columns (including FILENAME, URL, 53 features, label).
        label: 0 = Phishing, 1 = Legitimate.

    RESEARCH REQUIREMENT:
        Our research focuses strictly on URL-only detection. By default,
        url_only=True returns only ['URL', 'label'], ignoring all HTML,
        webpage content, DNS, and precomputed features.

    Args:
        filepath: Optional custom path to the CSV file. Defaults to PHIUSIIL_CSV.
        url_only: If True (default), extracts strictly ['URL', 'label'].
        verify_checksum: If True, checks file against PHIUSIIL_CSV_EXPECTED_SHA256.

    Returns:
        pd.DataFrame containing the requested dataset columns.
    """
    target_path = Path(filepath) if filepath else PHIUSIIL_CSV
    if not target_path.exists():
        raise FileNotFoundError(
            f"PhiUSIIL raw file not found at {target_path}. "
            "Please ensure dataset acquisition is complete."
        )

    if verify_checksum:
        is_valid, actual_sha = verify_dataset_integrity(target_path, PHIUSIIL_CSV_EXPECTED_SHA256)
        if not is_valid:
            raise ValueError(
                f"PhiUSIIL SHA-256 mismatch! Expected {PHIUSIIL_CSV_EXPECTED_SHA256}, "
                f"got {actual_sha}."
            )

    usecols = ["URL", "label"] if url_only else None
    return pd.read_csv(target_path, usecols=usecols)
