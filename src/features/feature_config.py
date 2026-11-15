"""Feature configuration, vocabulary definitions, and schema specifications.

Defines the exact, fixed schema for all URL-only features extracted
across all research datasets.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Targeted suspicious keyword vocabulary for phishing heuristic analysis
SUSPICIOUS_KEYWORDS: Tuple[str, ...] = (
    "login",
    "signin",
    "verify",
    "account",
    "update",
    "secure",
    "banking",
    "payment",
    "password",
    "confirm",
    "authentication",
)

# Common file extensions inspected in URL paths
COMMON_FILE_EXTENSIONS: Tuple[str, ...] = (
    ".exe", ".zip", ".rar", ".tar", ".gz", ".7z", ".iso", ".dmg", ".bin",
    ".sh", ".bat", ".cmd", ".apk", ".php", ".asp", ".aspx", ".jsp", ".cgi",
    ".js", ".html", ".htm", ".pdf", ".doc", ".docx", ".xls", ".xlsx",
)

# Ordered feature definitions: (feature_name, data_type, category, description)
FEATURE_SPECIFICATIONS: List[Tuple[str, str, str, str]] = [
    # A. Length Features (5)
    ("url_length", "int64", "length", "Total character count of the URL string"),
    ("hostname_length", "int64", "length", "Character count of the parsed hostname"),
    ("path_length", "int64", "length", "Character count of the path component"),
    ("query_length", "int64", "length", "Character count of the query component"),
    ("fragment_length", "int64", "length", "Character count of the fragment component"),

    # B. Character-Based Features (20)
    ("dot_count", "int64", "character", "Count of period (.) characters"),
    ("hyphen_count", "int64", "character", "Count of hyphen (-) characters"),
    ("underscore_count", "int64", "character", "Count of underscore (_) characters"),
    ("slash_count", "int64", "character", "Count of forward slash (/) characters"),
    ("backslash_count", "int64", "character", "Count of backslash (\\) characters"),
    ("at_symbol_count", "int64", "character", "Count of at (@) symbols"),
    ("question_mark_count", "int64", "character", "Count of question mark (?) characters"),
    ("equal_count", "int64", "character", "Count of equal (=) characters"),
    ("ampersand_count", "int64", "character", "Count of ampersand (&) characters"),
    ("percent_count", "int64", "character", "Count of percent (%) characters indicating encoding"),
    ("digit_count", "int64", "character", "Count of numeric digits (0-9)"),
    ("alpha_count", "int64", "character", "Count of alphabetic characters (a-zA-Z)"),
    ("special_char_count", "int64", "character", "Count of special characters (non-alphanumeric, non-slash/colon)"),
    ("digit_ratio", "float64", "character", "Ratio of numeric digits to total URL length"),
    ("alpha_ratio", "float64", "character", "Ratio of alphabetic characters to total URL length"),
    ("special_char_ratio", "float64", "character", "Ratio of special characters to total URL length"),
    ("max_consecutive_repeated_char", "int64", "character", "Length of the longest run of consecutive identical characters"),
    ("consecutive_dots_count", "int64", "character", "Count of consecutive dot sequences (..) in URL"),
    ("consecutive_hyphens_count", "int64", "character", "Count of consecutive hyphen sequences (--) in URL"),
    ("url_entropy", "float64", "character", "Shannon entropy of characters across the entire URL string"),

    # C. Hostname Features (9)
    ("subdomain_count", "int64", "hostname", "Count of subdomain levels preceding the domain"),
    ("hostname_label_count", "int64", "hostname", "Total dot-separated labels in the hostname"),
    ("max_hostname_label_length", "int64", "hostname", "Length of the longest dot-separated hostname label"),
    ("avg_hostname_label_length", "float64", "hostname", "Average length of dot-separated hostname labels"),
    ("is_ip_address", "int64", "hostname", "Binary indicator (1 if hostname is an IPv4 or IPv6 address literal, 0 otherwise)"),
    ("hostname_has_digits", "int64", "hostname", "Binary indicator (1 if hostname contains numeric digits, 0 otherwise)"),
    ("hostname_has_hyphens", "int64", "hostname", "Binary indicator (1 if hostname contains hyphens, 0 otherwise)"),
    ("hostname_has_punycode", "int64", "hostname", "Binary indicator (1 if hostname contains IDN punycode 'xn--', 0 otherwise)"),
    ("hostname_has_unusual_ip", "int64", "hostname", "Binary indicator (1 if hostname contains obfuscated or non-standard IP format, 0 otherwise)"),

    # D. Path Features (9)
    ("path_segment_count", "int64", "path", "Count of non-empty path segments separated by '/'"),
    ("avg_path_segment_length", "float64", "path", "Average character length of path segments"),
    ("max_path_segment_length", "int64", "path", "Maximum character length of any path segment"),
    ("path_encoded_char_count", "int64", "path", "Count of percent-encoded characters in path"),
    ("path_digit_count", "int64", "path", "Count of numeric digits in path"),
    ("path_special_char_count", "int64", "path", "Count of special characters in path"),
    ("is_path_empty", "int64", "path", "Binary indicator (1 if path is empty or root '/', 0 otherwise)"),
    ("path_has_file_extension", "int64", "path", "Binary indicator (1 if path contains a file extension, 0 otherwise)"),
    ("path_file_extension_count", "int64", "path", "Total occurrences of recognized file extensions in path"),

    # E. Query Parameter Features (5)
    ("query_param_count", "int64", "query", "Count of query parameter key-value pairs"),
    ("query_param_separator_count", "int64", "query", "Count of query parameter delimiters (& and ;)"),
    ("query_encoded_char_count", "int64", "query", "Count of percent-encoded characters in query string"),
    ("query_has_long_value", "int64", "query", "Binary indicator (1 if any parameter key/value exceeds 50 chars, 0 otherwise)"),
    ("query_numeric_value_count", "int64", "query", "Count of query parameter values that are purely numeric"),

    # F. Suspicious Lexical Pattern Features (12)
    ("has_kw_login", "int64", "lexical_pattern", "Binary indicator for presence of 'login' in URL"),
    ("has_kw_signin", "int64", "lexical_pattern", "Binary indicator for presence of 'signin' in URL"),
    ("has_kw_verify", "int64", "lexical_pattern", "Binary indicator for presence of 'verify' in URL"),
    ("has_kw_account", "int64", "lexical_pattern", "Binary indicator for presence of 'account' in URL"),
    ("has_kw_update", "int64", "lexical_pattern", "Binary indicator for presence of 'update' in URL"),
    ("has_kw_secure", "int64", "lexical_pattern", "Binary indicator for presence of 'secure' in URL"),
    ("has_kw_banking", "int64", "lexical_pattern", "Binary indicator for presence of 'banking' in URL"),
    ("has_kw_payment", "int64", "lexical_pattern", "Binary indicator for presence of 'payment' in URL"),
    ("has_kw_password", "int64", "lexical_pattern", "Binary indicator for presence of 'password' in URL"),
    ("has_kw_confirm", "int64", "lexical_pattern", "Binary indicator for presence of 'confirm' in URL"),
    ("has_kw_authentication", "int64", "lexical_pattern", "Binary indicator for presence of 'authentication' in URL"),
    ("suspicious_keyword_count", "int64", "lexical_pattern", "Total count of suspicious keywords present across URL tokens"),

    # G. Protocol and Parsing Features (8)
    ("is_https", "int64", "protocol", "Binary indicator (1 if URL scheme is HTTPS, 0 otherwise)"),
    ("is_http", "int64", "protocol", "Binary indicator (1 if URL scheme is HTTP, 0 otherwise)"),
    ("is_missing_protocol", "int64", "protocol", "Binary indicator (1 if original raw URL lacked explicit scheme, 0 otherwise)"),
    ("protocol_type", "int64", "protocol", "Categorical code for scheme: 0=none/unknown, 1=http, 2=https, 3=other"),
    ("has_userinfo", "int64", "protocol", "Binary indicator (1 if userinfo 'user:pass@' is present in URL authority, 0 otherwise)"),
    ("has_port", "int64", "protocol", "Binary indicator (1 if non-default or explicit port ':PORT' is specified, 0 otherwise)"),
    ("has_fragment", "int64", "protocol", "Binary indicator (1 if fragment identifier '#...' is present, 0 otherwise)"),
    ("url_parse_success", "int64", "protocol", "Binary indicator (1 if URL was parsed without fatal syntax exceptions, 0 otherwise)"),
]

# Quick access lists and lookup mappings
FEATURE_NAMES: List[str] = [spec[0] for spec in FEATURE_SPECIFICATIONS]
FEATURE_DTYPES: Dict[str, str] = {spec[0]: spec[1] for spec in FEATURE_SPECIFICATIONS}
FEATURE_CATEGORIES: Dict[str, str] = {spec[0]: spec[2] for spec in FEATURE_SPECIFICATIONS}
FEATURE_DESCRIPTIONS: Dict[str, str] = {spec[0]: spec[3] for spec in FEATURE_SPECIFICATIONS}

TOTAL_FEATURE_COUNT = len(FEATURE_NAMES)

