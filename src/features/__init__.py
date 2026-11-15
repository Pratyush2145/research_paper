"""URL feature extraction package for phishing detection research."""

from src.features.feature_config import (
    FEATURE_CATEGORIES,
    FEATURE_DESCRIPTIONS,
    FEATURE_DTYPES,
    FEATURE_NAMES,
    SUSPICIOUS_KEYWORDS,
    TOTAL_FEATURE_COUNT,
)
from src.features.feature_extractor import (
    extract_features_dataframe,
    extract_url_features,
    extract_url_metadata,
    process_and_save_dataset,
)
from src.features.url_parser import ParsedURL, parse_url

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_DTYPES",
    "FEATURE_CATEGORIES",
    "FEATURE_DESCRIPTIONS",
    "SUSPICIOUS_KEYWORDS",
    "TOTAL_FEATURE_COUNT",
    "ParsedURL",
    "parse_url",
    "extract_url_features",
    "extract_features_dataframe",
    "extract_url_metadata",
    "process_and_save_dataset",
]
