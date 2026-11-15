"""Lexical and character-level URL feature extraction module.

Extracts length metrics, character counts, ratios, character repetitions,
Shannon entropy, and targeted heuristic keyword indicators directly from
the URL string.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict

from src.features.feature_config import SUSPICIOUS_KEYWORDS
from src.features.url_parser import ParsedURL


def compute_shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of the character distribution in a string.

    H = - sum(p_i * log2(p_i))
    """
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return float(round(entropy, 4))


def compute_max_consecutive_repeated_char(text: str) -> int:
    """Find the length of the longest run of consecutive identical characters."""
    if not text:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    return max_run


def extract_lexical_features(parsed: ParsedURL) -> Dict[str, Any]:
    """Extract all lexical, length, and character-based features from a ParsedURL.

    Args:
        parsed: ParsedURL object.

    Returns:
        Dictionary mapping feature names to extracted numeric values.
    """
    url = parsed.raw_url
    url_len = len(url)
    lower_url = url.lower()

    # Length features
    hostname_len = len(parsed.hostname)
    path_len = len(parsed.path)
    query_len = len(parsed.query)
    fragment_len = len(parsed.fragment)

    # Character counts
    dot_count = url.count(".")
    hyphen_count = url.count("-")
    underscore_count = url.count("_")
    slash_count = url.count("/")
    backslash_count = url.count("\\")
    at_count = url.count("@")
    question_count = url.count("?")
    equal_count = url.count("=")
    ampersand_count = url.count("&")
    percent_count = url.count("%")

    # Character classification counts
    digit_count = 0
    alpha_count = 0
    special_char_count = 0

    for char in url:
        if char.isdigit():
            digit_count += 1
        elif char.isalpha():
            alpha_count += 1
        elif char not in (":", "/"):  # standard scheme/path separators
            special_char_count += 1

    # Ratios (relative to total URL length)
    digit_ratio = (digit_count / url_len) if url_len > 0 else 0.0
    alpha_ratio = (alpha_count / url_len) if url_len > 0 else 0.0
    special_char_ratio = (special_char_count / url_len) if url_len > 0 else 0.0

    # Consecutive repetitions
    max_consec = compute_max_consecutive_repeated_char(url)
    consec_dots = url.count("..")
    consec_hyphens = url.count("--")

    # Shannon Entropy
    entropy = compute_shannon_entropy(url)

    # Suspicious keywords
    # Check keyword presence in lowercased URL
    kw_features: Dict[str, Any] = {}
    total_kw_count = 0
    for kw in SUSPICIOUS_KEYWORDS:
        feat_name = f"has_kw_{kw}"
        has_kw = int(kw in lower_url)
        kw_features[feat_name] = has_kw
        total_kw_count += has_kw
    kw_features["suspicious_keyword_count"] = total_kw_count

    return {
        "url_length": url_len,
        "hostname_length": hostname_len,
        "path_length": path_len,
        "query_length": query_len,
        "fragment_length": fragment_len,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "underscore_count": underscore_count,
        "slash_count": slash_count,
        "backslash_count": backslash_count,
        "at_symbol_count": at_count,
        "question_mark_count": question_count,
        "equal_count": equal_count,
        "ampersand_count": ampersand_count,
        "percent_count": percent_count,
        "digit_count": digit_count,
        "alpha_count": alpha_count,
        "special_char_count": special_char_count,
        "digit_ratio": round(digit_ratio, 4),
        "alpha_ratio": round(alpha_ratio, 4),
        "special_char_ratio": round(special_char_ratio, 4),
        "max_consecutive_repeated_char": max_consec,
        "consecutive_dots_count": consec_dots,
        "consecutive_hyphens_count": consec_hyphens,
        "url_entropy": entropy,
        **kw_features,
    }

