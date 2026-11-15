"""URL parsing and offline normalization module for feature extraction.

Decomposes URL strings into structured components offline without making
any DNS, WHOIS, or network requests.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

from src.data.preprocess import TWO_PART_TLDS, extract_registrable_domain


@dataclass(slots=True, frozen=True)
class ParsedURL:
    """Structured representation of a parsed URL string."""

    raw_url: str
    normalized_url: str
    scheme: str
    netloc: str
    hostname: str
    port: Optional[int]
    userinfo: str
    path: str
    query: str
    fragment: str
    is_ip: bool
    is_unusual_ip: bool
    has_punycode: bool
    is_missing_protocol: bool
    parse_success: bool
    registrable_domain: str


def check_ip_representation(host: str) -> tuple[bool, bool]:
    """Check if a hostname string represents a standard or unusual IP address.

    Returns:
        (is_ip, is_unusual_ip)
    """
    if not host:
        return False, False

    # Standard IPv4 or IPv6 literal
    try:
        ipaddress.ip_address(host)
        return True, False
    except ValueError:
        pass

    # Hexadecimal IP representation (e.g., 0x7f000001 or 0x7f.0x00.0x00.0x01)
    if re.match(r"^0x[0-9a-fA-F]+(\.0x[0-9a-fA-F]+)*$", host):
        return True, True

    # 32-bit DWORD integer IP (e.g., 2130706433)
    if host.isdigit():
        val = int(host)
        if 0 <= val <= 0xFFFFFFFF:
            return True, True

    # Octal IP parts (e.g., 0177.0.0.1)
    parts = host.split(".")
    if len(parts) == 4 and all(re.match(r"^0[0-7]+$", p) or p.isdigit() for p in parts):
        if any(p.startswith("0") and len(p) > 1 for p in parts):
            return True, True

    return False, False


def normalize_url(raw_url: str) -> tuple[str, bool]:
    """Perform deterministic offline URL normalization.

    Returns:
        (normalized_url, was_protocol_missing)
    """
    if not isinstance(raw_url, str):
        return "", True

    cleaned = raw_url.strip()
    if not cleaned:
        return "", True

    is_missing_protocol = False

    # Fix protocol typos
    if cleaned.startswith("ttp://"):
        cleaned = "http://" + cleaned[6:]
    elif cleaned.startswith("ttps://"):
        cleaned = "https://" + cleaned[7:]

    # Collapse duplicated scheme prefixes (e.g. http://http:// or http://https://)
    while True:
        m = re.match(r"^(https?://)(https?://.*)$", cleaned, re.IGNORECASE)
        if m:
            cleaned = m.group(2)
        else:
            break

    # Add default scheme if missing
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        is_missing_protocol = True
        cleaned = "http://" + cleaned

    return cleaned, is_missing_protocol


def parse_url(raw_url: str) -> ParsedURL:
    """Parse a URL string into structured components offline.

    Args:
        raw_url: The uncleaned URL string.

    Returns:
        ParsedURL dataclass instance.
    """
    if not isinstance(raw_url, str) or not raw_url.strip():
        return ParsedURL(
            raw_url="" if raw_url is None else str(raw_url),
            normalized_url="",
            scheme="",
            netloc="",
            hostname="",
            port=None,
            userinfo="",
            path="",
            query="",
            fragment="",
            is_ip=False,
            is_unusual_ip=False,
            has_punycode=False,
            is_missing_protocol=True,
            parse_success=False,
            registrable_domain="",
        )

    normalized, was_missing = normalize_url(raw_url)

    try:
        split_res = urlsplit(normalized)
        scheme = split_res.scheme.lower()
        netloc = split_res.netloc
        hostname = (split_res.hostname or "").lower().strip(".")
        port = split_res.port

        # Extract userinfo if present in netloc (e.g. user:pass@host)
        userinfo = ""
        if "@" in netloc:
            userinfo = netloc.split("@")[0]

        path = split_res.path or ""
        query = split_res.query or ""
        fragment = split_res.fragment or ""

        is_ip, is_unusual_ip = check_ip_representation(hostname)
        has_puny = "xn--" in hostname

        reg_domain = extract_registrable_domain(raw_url)

        return ParsedURL(
            raw_url=raw_url,
            normalized_url=normalized,
            scheme=scheme,
            netloc=netloc,
            hostname=hostname,
            port=port,
            userinfo=userinfo,
            path=path,
            query=query,
            fragment=fragment,
            is_ip=is_ip,
            is_unusual_ip=is_unusual_ip,
            has_punycode=has_puny,
            is_missing_protocol=was_missing,
            parse_success=True,
            registrable_domain=reg_domain,
        )
    except Exception:
        return ParsedURL(
            raw_url=raw_url,
            normalized_url=normalized,
            scheme="",
            netloc="",
            hostname="",
            port=None,
            userinfo="",
            path="",
            query="",
            fragment="",
            is_ip=False,
            is_unusual_ip=False,
            has_punycode=False,
            is_missing_protocol=was_missing,
            parse_success=False,
            registrable_domain="",
        )

