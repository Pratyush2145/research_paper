"""Data preprocessing and offline URL parsing module for phishing research.

Provides offline URL validation, component extraction, registrable domain
derivation, and duplicate / conflicting label identification without making
any external network calls or web requests.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import pandas as pd

# Multi-part public suffixes (ccTLDs and common delegated second-level domains)
TWO_PART_TLDS = frozenset(
    {
        "co.uk", "org.uk", "me.uk", "ltd.uk", "plc.uk", "net.uk", "sch.uk", "ac.uk", "gov.uk",
        "com.au", "net.au", "org.au", "edu.au", "gov.au", "asn.au", "id.au",
        "co.nz", "net.nz", "org.nz", "govt.nz", "ac.nz",
        "co.jp", "ne.jp", "or.jp", "go.jp", "ac.jp", "ed.jp",
        "co.in", "net.in", "org.in", "gen.in", "firm.in", "ind.in", "ac.in", "edu.in", "res.in", "gov.in", "mil.in",
        "com.br", "net.br", "org.br", "gov.br", "edu.br",
        "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
        "com.tw", "net.tw", "org.tw", "gov.tw", "edu.tw",
        "com.hk", "net.hk", "org.hk", "gov.hk", "edu.hk",
        "co.za", "net.za", "org.za", "gov.za", "ac.za",
        "com.sg", "net.sg", "org.sg", "gov.sg", "edu.sg",
        "com.my", "net.my", "org.my", "gov.my", "edu.my",
        "com.mx", "net.mx", "org.mx", "gob.mx", "edu.mx",
        "com.ar", "net.ar", "org.ar", "gob.ar", "edu.ar",
        "co.kr", "ne.kr", "or.kr", "re.kr", "pe.kr", "go.kr", "mil.kr", "ac.kr",
        "com.ru", "net.ru", "org.ru", "pp.ru",
        "com.tr", "net.tr", "org.tr", "gov.tr", "edu.tr",
        "com.pk", "net.pk", "org.pk", "gov.pk", "edu.pk",
        "com.ng", "net.ng", "org.ng", "gov.ng", "edu.ng",
        "com.ph", "net.ph", "org.ph", "gov.ph", "edu.ph",
        "com.co", "net.co", "nom.co", "gov.co", "edu.co",
    }
)


def normalize_url_scheme(url: str) -> str:
    """Normalize and clean URL scheme offline.

    Handles duplicated prefixes (e.g. 'http://http://') and common typos (e.g. 'ttp://').
    """
    if not isinstance(url, str):
        return ""
    cleaned = url.strip()
    if not cleaned:
        return ""

    # Normalize typo schemes
    if cleaned.startswith("ttp://"):
        cleaned = "http://" + cleaned[6:]
    elif cleaned.startswith("ttps://"):
        cleaned = "https://" + cleaned[7:]

    # Collapse duplicated scheme prefixes (e.g., http://http:// or http://https://)
    while True:
        m = re.match(r"^(https?://)(https?://.*)$", cleaned, re.IGNORECASE)
        if m:
            cleaned = m.group(2)
        else:
            break

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        cleaned = "http://" + cleaned

    return cleaned


def extract_hostname(url: str) -> str:
    """Extract lowercased hostname from a URL offline without network requests."""
    normalized = normalize_url_scheme(url)
    if not normalized:
        return ""
    try:
        parsed = urlsplit(normalized)
        return (parsed.hostname or "").lower().strip(".")
    except Exception:
        return ""


def extract_registrable_domain(url: str) -> str:
    """Extract the registrable domain (eTLD+1) or IP address offline.

    Uses offline public suffix heuristics without DNS or network resolution.

    Args:
        url: Raw URL string.

    Returns:
        Registrable domain (e.g. 'example.com', 'site.co.uk') or IP address,
        or empty string if unparseable/invalid.
    """
    host = extract_hostname(url)
    if not host:
        return ""

    # Check for IPv4 / IPv6 literal
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass

    parts = host.split(".")
    if len(parts) <= 1:
        # Single token hostname without TLD (e.g. local hostname or malformed string)
        return ""

    # Multi-part public suffix check (e.g. .co.uk, .ac.in)
    if len(parts) >= 3:
        two_part = ".".join(parts[-2:])
        if two_part in TWO_PART_TLDS:
            return ".".join(parts[-3:])

    return ".".join(parts[-2:])


def parse_url_components(url: str) -> Dict[str, Any]:
    """Parse a URL into its structural components offline.

    DO NOT visit or request the target URL.
    """
    if not isinstance(url, str) or not url.strip():
        return {
            "scheme": "",
            "hostname": "",
            "registrable_domain": "",
            "path": "",
            "query": "",
            "fragment": "",
            "has_subdomain": False,
            "has_path": False,
            "has_query": False,
            "has_fragment": False,
            "is_valid": False,
        }

    raw = url.strip()
    raw_scheme = ""
    if "://" in raw:
        raw_scheme = raw.split("://")[0].lower()

    normalized = normalize_url_scheme(raw)
    try:
        parsed = urlsplit(normalized)
        host = (parsed.hostname or "").lower().strip(".")
        reg_domain = extract_registrable_domain(raw)
        has_sub = len(host.split(".")) > 2 if host else False
        path = parsed.path or ""
        query = parsed.query or ""
        fragment = parsed.fragment or ""

        is_valid = bool(host and ("." in host or reg_domain))

        return {
            "scheme": raw_scheme or parsed.scheme,
            "hostname": host,
            "registrable_domain": reg_domain,
            "path": path,
            "query": query,
            "fragment": fragment,
            "has_subdomain": has_sub,
            "has_path": bool(path and path != "/"),
            "has_query": bool(query),
            "has_fragment": bool(fragment),
            "is_valid": is_valid,
        }
    except Exception:
        return {
            "scheme": raw_scheme,
            "hostname": "",
            "registrable_domain": "",
            "path": "",
            "query": "",
            "fragment": "",
            "has_subdomain": False,
            "has_path": False,
            "has_query": False,
            "has_fragment": False,
            "is_valid": False,
        }


def identify_duplicate_urls(df: pd.DataFrame, url_col: str = "URL") -> pd.DataFrame:
    """Identify duplicate URL records in a dataframe."""
    return df[df[url_col].duplicated(keep=False)].sort_values(url_col)


def identify_conflicting_label_urls(
    df: pd.DataFrame, url_col: str = "URL", label_col: str = "label"
) -> pd.DataFrame:
    """Identify URLs associated with more than one distinct label."""
    clean_df = df.dropna(subset=[url_col, label_col])
    conflicts = clean_df.groupby(url_col)[label_col].nunique()
    conflicting_urls = conflicts[conflicts > 1].index
    return df[df[url_col].isin(conflicting_urls)].sort_values(url_col)


def identify_conflicting_label_domains(
    df: pd.DataFrame, domain_col: str = "registrable_domain", label_col: str = "label"
) -> Dict[str, Dict[Any, int]]:
    """Identify domains associated with both phishing and legitimate labels."""
    clean_df = df.dropna(subset=[domain_col, label_col])
    clean_df = clean_df[clean_df[domain_col] != ""]
    conflicts = clean_df.groupby(domain_col)[label_col].nunique()
    conflicting_domains = conflicts[conflicts > 1].index

    results = {}
    for dom in conflicting_domains:
        counts = clean_df[clean_df[domain_col] == dom][label_col].value_counts().to_dict()
        results[dom] = counts
    return results
