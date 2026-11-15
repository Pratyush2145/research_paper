"""Structural, component-level URL feature extraction module.

Extracts features describing hostname composition, path segments,
query parameter structures, protocol types, and parsing indicators offline.
"""

from __future__ import annotations

import re
from typing import Any, Dict
from urllib.parse import parse_qsl

from src.features.feature_config import COMMON_FILE_EXTENSIONS
from src.features.url_parser import ParsedURL


def extract_structural_features(parsed: ParsedURL) -> Dict[str, Any]:
    """Extract structural, hostname, path, query, and protocol features from a ParsedURL.

    Args:
        parsed: ParsedURL object.

    Returns:
        Dictionary mapping structural feature names to extracted numeric values.
    """
    # -------------------------------------------------------------
    # C. Hostname Features
    # -------------------------------------------------------------
    host = parsed.hostname
    if host:
        labels = host.split(".")
        label_count = len(labels)
        label_lens = [len(lbl) for lbl in labels]
        max_label_len = max(label_lens)
        avg_label_len = sum(label_lens) / label_count

        # Subdomain count: labels preceding the registrable domain
        reg_domain = parsed.registrable_domain
        if reg_domain and not parsed.is_ip:
            reg_labels_count = len(reg_domain.split("."))
            subdomain_count = max(0, label_count - reg_labels_count)
        else:
            subdomain_count = 0

        has_digits = int(any(char.isdigit() for char in host))
        has_hyphens = int("-" in host)
    else:
        label_count = 0
        max_label_len = 0
        avg_label_len = 0.0
        subdomain_count = 0
        has_digits = 0
        has_hyphens = 0

    is_ip = int(parsed.is_ip)
    has_unusual_ip = int(parsed.is_unusual_ip)
    has_puny = int(parsed.has_punycode)

    # -------------------------------------------------------------
    # D. Path Features
    # -------------------------------------------------------------
    path = parsed.path
    # Non-empty path segments
    raw_segments = [s for s in path.split("/") if s]
    segment_count = len(raw_segments)

    if segment_count > 0:
        segment_lens = [len(s) for s in raw_segments]
        avg_seg_len = sum(segment_lens) / segment_count
        max_seg_len = max(segment_lens)
    else:
        avg_seg_len = 0.0
        max_seg_len = 0

    path_encoded_count = path.count("%")
    path_digit_count = sum(1 for c in path if c.isdigit())
    path_special_count = sum(1 for c in path if not c.isalnum() and c not in ("/", "."))
    is_path_empty = int(not path or path == "/")

    # File extension analysis
    lower_path = path.lower()
    ext_count = 0
    for ext in COMMON_FILE_EXTENSIONS:
        if ext in lower_path:
            ext_count += lower_path.count(ext)
    has_ext = int(ext_count > 0)

    # -------------------------------------------------------------
    # E. Query Parameter Features
    # -------------------------------------------------------------
    query = parsed.query
    if query:
        # Delimiters count
        query_sep_count = query.count("&") + query.count(";")
        query_encoded_count = query.count("%")

        # Key-value parsing
        try:
            params = parse_qsl(query, keep_blank_values=True)
            param_count = len(params)
            has_long_val = int(any(len(k) > 50 or len(v) > 50 for k, v in params))
            numeric_val_count = sum(1 for _, v in params if v.isdigit())
        except Exception:
            param_count = query_sep_count + 1
            has_long_val = int(len(query) > 50)
            numeric_val_count = 0
    else:
        param_count = 0
        query_sep_count = 0
        query_encoded_count = 0
        has_long_val = 0
        numeric_val_count = 0

    # -------------------------------------------------------------
    # G. Protocol and Parsing Features
    # -------------------------------------------------------------
    scheme = parsed.scheme
    is_https = int(scheme == "https")
    is_http = int(scheme == "http")
    is_missing_protocol = int(parsed.is_missing_protocol)

    if is_https:
        protocol_type = 2
    elif is_http:
        protocol_type = 1
    elif scheme:
        protocol_type = 3  # other (e.g. ftp, sftp)
    else:
        protocol_type = 0  # none / unknown

    has_userinfo = int(bool(parsed.userinfo))
    has_port = int(parsed.port is not None)
    has_fragment = int(bool(parsed.fragment))
    url_parse_success = int(parsed.parse_success)

    return {
        "subdomain_count": subdomain_count,
        "hostname_label_count": label_count,
        "max_hostname_label_length": max_label_len,
        "avg_hostname_label_length": round(avg_label_len, 4),
        "is_ip_address": is_ip,
        "hostname_has_digits": has_digits,
        "hostname_has_hyphens": has_hyphens,
        "hostname_has_punycode": has_puny,
        "hostname_has_unusual_ip": has_unusual_ip,
        "path_segment_count": segment_count,
        "avg_path_segment_length": round(avg_seg_len, 4),
        "max_path_segment_length": max_seg_len,
        "path_encoded_char_count": path_encoded_count,
        "path_digit_count": path_digit_count,
        "path_special_char_count": path_special_count,
        "is_path_empty": is_path_empty,
        "path_has_file_extension": has_ext,
        "path_file_extension_count": ext_count,
        "query_param_count": param_count,
        "query_param_separator_count": query_sep_count,
        "query_encoded_char_count": query_encoded_count,
        "query_has_long_value": has_long_val,
        "query_numeric_value_count": numeric_val_count,
        "is_https": is_https,
        "is_http": is_http,
        "is_missing_protocol": is_missing_protocol,
        "protocol_type": protocol_type,
        "has_userinfo": has_userinfo,
        "has_port": has_port,
        "has_fragment": has_fragment,
        "url_parse_success": url_parse_success,
    }

