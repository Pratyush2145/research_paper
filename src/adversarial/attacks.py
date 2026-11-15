"""Adversarial URL transformation generators with formal validity tiers.

Implements the Three-Tier Validity Framework:
- Tier A: Strictly Semantics-Preserving (Destination & Label Equivalence Guaranteed)
- Tier B: Conditionally Semantics-Preserving (Requires Explicit Pre-Condition Validation)
- Tier C: Exploratory & Out-of-Distribution (Label Invariance Not Established)

Every transformation performs strict applicability checking, handles URL parsing
edge cases cleanly, and logs transformation rationale without breaking syntax.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urlsplit, urlunsplit


@dataclass(frozen=True)
class AttackResult:
    """Outcome of an adversarial transformation attempt on a URL."""

    original_url: str
    transformed_url: str
    applicable: bool
    success: bool
    tier: str
    category_id: str
    category_name: str
    reason: Optional[str] = None


@dataclass(frozen=True)
class AttackDefinition:
    """Metadata and transformation callable for an attack category."""

    category_id: str
    name: str
    tier: str
    rfc_basis: str
    transform_fn: Callable[[str, int], AttackResult]


# ---------------------------------------------------------------------------
# Helper Parsing Utilities
# ---------------------------------------------------------------------------

def _decompose_url(raw_url: str) -> Tuple[str, str, str, str, str, bool]:
    """Safely split URL, handling missing schemes gracefully."""
    url = raw_url.strip()
    has_explicit_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url))
    if not has_explicit_scheme:
        # Prepend dummy scheme for splitting if missing
        parts = urlsplit("http://" + url)
        return "", parts.netloc, parts.path, parts.query, parts.fragment, False
    parts = urlsplit(url)
    return parts.scheme, parts.netloc, parts.path, parts.query, parts.fragment, True


def _recompose_url(scheme: str, netloc: str, path: str, query: str, fragment: str, had_scheme: bool) -> str:
    """Rebuild URL string adhering to original scheme presence."""
    if not had_scheme:
        # Rebuild without scheme
        built = urlunsplit(("", netloc, path, query, fragment))
        if built.startswith("//"):
            built = built[2:]
        return built
    return urlunsplit((scheme, netloc, path, query, fragment))


def _extract_host_and_port(netloc: str) -> Tuple[str, Optional[int]]:
    """Separate host and port from netloc."""
    if not netloc:
        return "", None
    if "@" in netloc:
        netloc = netloc.split("@", 1)[1]
    if netloc.startswith("["):
        # IPv6
        idx = netloc.find("]")
        if idx != -1:
            host = netloc[: idx + 1]
            rest = netloc[idx + 1 :]
            if rest.startswith(":"):
                try:
                    return host, int(rest[1:])
                except ValueError:
                    return host, None
            return host, None
    parts = netloc.split(":")
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            return parts[0], None
    return parts[0], None


def _is_ipv4(host: str) -> bool:
    """Check if host is a valid IPv4 dotted-decimal address."""
    try:
        ip = ipaddress.ip_address(host)
        return isinstance(ip, ipaddress.IPv4Address)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Category 1: URL Percent-Encoding Variations
# ---------------------------------------------------------------------------

def transform_cat_01a_unreserved_encoding(url: str, label: int) -> AttackResult:
    """CAT-01A (Tier A): Path Unreserved Percent-Encoding (RFC 3986 §2.3)."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    if not path or path == "/":
        return AttackResult(url, url, False, False, "Tier A", "CAT-01A", "Path Unreserved Encoding", "No path to encode")

    # Encode ASCII alphanumeric characters in the path
    encoded_chars = []
    modified = False
    for char in path:
        if char.isalnum() and ord(char) < 128:
            encoded_chars.append(f"%{ord(char):02X}")
            modified = True
        else:
            encoded_chars.append(char)

    if not modified:
        return AttackResult(url, url, False, False, "Tier A", "CAT-01A", "Path Unreserved Encoding", "No unreserved chars in path")

    new_path = "".join(encoded_chars)
    new_url = _recompose_url(scheme, netloc, new_path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier A", "CAT-01A", "Path Unreserved Encoding")


def transform_cat_01b_reserved_encoding(url: str, label: int) -> AttackResult:
    """CAT-01B (Tier B): Path Reserved Character Encoding."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    if not path or path == "/":
        return AttackResult(url, url, False, False, "Tier B", "CAT-01B", "Path Reserved Char Encoding", "No path to encode")

    # Encode hyphen, underscore, tilde, dot if present
    target_chars = {"-": "%2D", "_": "%5F", ".": "%2E", "~": "%7E"}
    modified = False
    new_path_chars = []
    for char in path:
        if char in target_chars:
            new_path_chars.append(target_chars[char])
            modified = True
        else:
            new_path_chars.append(char)

    if not modified:
        return AttackResult(url, url, False, False, "Tier B", "CAT-01B", "Path Reserved Char Encoding", "No target delimiter chars in path")

    new_path = "".join(new_path_chars)
    new_url = _recompose_url(scheme, netloc, new_path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier B", "CAT-01B", "Path Reserved Char Encoding")


def transform_cat_01c_double_encoding(url: str, label: int) -> AttackResult:
    """CAT-01C (Tier C): Double Percent-Encoding (% -> %25)."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    if not path or path == "/":
        return AttackResult(url, url, False, False, "Tier C", "CAT-01C", "Double Percent-Encoding", "No path to encode")

    # If path already contains percent-encoded sequence, replace % with %25
    if "%" in path:
        new_path = path.replace("%", "%25")
    else:
        # First encode one or more characters, then double encode
        chars = []
        encoded = False
        for c in path:
            if c.isalnum() and not encoded:
                chars.append(f"%25{ord(c):02X}")
                encoded = True
            else:
                chars.append(c)
        if not encoded:
            return AttackResult(url, url, False, False, "Tier C", "CAT-01C", "Double Percent-Encoding", "No eligible characters in path")
        new_path = "".join(chars)

    new_url = _recompose_url(scheme, netloc, new_path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier C", "CAT-01C", "Double Percent-Encoding")


# ---------------------------------------------------------------------------
# Category 2: Case Variation & Delimiter Manipulation
# ---------------------------------------------------------------------------

def transform_cat_02a_scheme_casing(url: str, label: int) -> AttackResult:
    """CAT-02A (Tier A): Scheme Case Variation (RFC 3986 §3.1)."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    if not had_scheme or not scheme:
        return AttackResult(url, url, False, False, "Tier A", "CAT-02A", "Scheme Case Variation", "No explicit scheme present")

    # Alternate case in scheme (e.g., https -> hTtPs)
    new_scheme = "".join(c.upper() if i % 2 == 1 else c.lower() for i, c in enumerate(scheme))
    if new_scheme == scheme:
        new_scheme = scheme.upper()

    new_url = _recompose_url(new_scheme, netloc, path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier A", "CAT-02A", "Scheme Case Variation")


def transform_cat_02b_hostname_casing(url: str, label: int) -> AttackResult:
    """CAT-02B (Tier A): Hostname Case Variation (RFC 4343 §2)."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    host, port = _extract_host_and_port(netloc)
    if not host or _is_ipv4(host) or not any(c.isalpha() for c in host):
        return AttackResult(url, url, False, False, "Tier A", "CAT-02B", "Hostname Case Variation", "No alphabetic hostname")

    # Alternate case in hostname
    new_host_chars = []
    alpha_idx = 0
    for c in host:
        if c.isalpha():
            new_host_chars.append(c.upper() if alpha_idx % 2 == 1 else c.lower())
            alpha_idx += 1
        else:
            new_host_chars.append(c)

    new_host = "".join(new_host_chars)
    if new_host == host:
        new_host = host.upper()

    new_netloc = f"{new_host}:{port}" if port is not None else new_host
    new_url = _recompose_url(scheme, new_netloc, path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier A", "CAT-02B", "Hostname Case Variation")


def transform_cat_02c_redundant_slashes(url: str, label: int) -> AttackResult:
    """CAT-02C (Tier B): Redundant Consecutive Slashes."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    if not path or path == "/":
        return AttackResult(url, url, False, False, "Tier B", "CAT-02C", "Redundant Consecutive Slashes", "No path to inject slashes")

    # Prepend redundant slashes to path
    new_path = "//" + path.lstrip("/")
    new_url = _recompose_url(scheme, netloc, new_path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier B", "CAT-02C", "Redundant Consecutive Slashes")


# ---------------------------------------------------------------------------
# Category 3: Homoglyphs & Internationalized Domain Names (IDN)
# ---------------------------------------------------------------------------

CYRILLIC_LOOKALIKES = {
    "a": "\u0430",  # Cyrillic small letter a
    "e": "\u0435",  # Cyrillic small letter ie
    "o": "\u043e",  # Cyrillic small letter o
    "p": "\u0440",  # Cyrillic small letter er
    "c": "\u0441",  # Cyrillic small letter es
    "y": "\u0443",  # Cyrillic small letter u
    "x": "\u0445",  # Cyrillic small letter ha
    "i": "\u0456",  # Cyrillic small letter byelorussian-ukrainian i
}


def transform_cat_03a_cyrillic_homoglyph(url: str, label: int) -> AttackResult:
    """CAT-03A (Tier C): Cyrillic Lookalike Homoglyph Replacement."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    host, port = _extract_host_and_port(netloc)
    if not host or _is_ipv4(host):
        return AttackResult(url, url, False, False, "Tier C", "CAT-03A", "Cyrillic Lookalike Homoglyph", "No eligible domain host")

    # Replace the first substitutable Latin character in the second-level domain
    parts = host.split(".")
    if len(parts) < 2:
        return AttackResult(url, url, False, False, "Tier C", "CAT-03A", "Cyrillic Lookalike Homoglyph", "Single-label host")

    sld = parts[-2]
    modified_sld = None
    for i, char in enumerate(sld):
        if char.lower() in CYRILLIC_LOOKALIKES:
            sub = CYRILLIC_LOOKALIKES[char.lower()]
            modified_sld = sld[:i] + sub + sld[i + 1 :]
            break

    if modified_sld is None:
        # Fallback to any label in host
        host_chars = list(host)
        for i, char in enumerate(host_chars):
            if char.lower() in CYRILLIC_LOOKALIKES and host_chars[i] != ".":
                host_chars[i] = CYRILLIC_LOOKALIKES[char.lower()]
                modified_sld = "".join(host_chars)
                break

    if modified_sld is None:
        return AttackResult(url, url, False, False, "Tier C", "CAT-03A", "Cyrillic Lookalike Homoglyph", "No substitutable Latin characters in host")

    parts[-2] = modified_sld if "." in host else modified_sld
    new_host = ".".join(parts) if "." in host else modified_sld
    new_netloc = f"{new_host}:{port}" if port is not None else new_host
    new_url = _recompose_url(scheme, new_netloc, path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier C", "CAT-03A", "Cyrillic Lookalike Homoglyph")


def transform_cat_03b_punycode_equivalence(url: str, label: int) -> AttackResult:
    """CAT-03B (Tier A): Punycode Round-Trip Equivalence (RFC 3492)."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    host, port = _extract_host_and_port(netloc)
    if not host or _is_ipv4(host):
        return AttackResult(url, url, False, False, "Tier A", "CAT-03B", "Punycode Round-Trip Equivalence", "No domain host")

    # If host already contains non-ASCII or punycode, transcode it
    try:
        if "xn--" in host.lower():
            # Decode to unicode
            new_host = host.encode("ascii").decode("idna")
        elif any(ord(c) >= 128 for c in host):
            # Encode to punycode
            new_host = host.encode("idna").decode("ascii")
        else:
            # Pure ASCII host: strictly speaking, no international characters to transcode.
            # Mark not applicable to avoid fabricating unrepresentative mutations.
            return AttackResult(url, url, False, False, "Tier A", "CAT-03B", "Punycode Round-Trip Equivalence", "Host is standard ASCII (no IDN characters)")
    except Exception as e:
        return AttackResult(url, url, False, False, "Tier A", "CAT-03B", "Punycode Round-Trip Equivalence", f"IDNA encoding error: {e}")

    new_netloc = f"{new_host}:{port}" if port is not None else new_host
    new_url = _recompose_url(scheme, new_netloc, path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier A", "CAT-03B", "Punycode Round-Trip Equivalence")


# ---------------------------------------------------------------------------
# Category 4: Subdomain Manipulation & Brand Prepending
# ---------------------------------------------------------------------------

def transform_cat_04a_subdomain_phishing(url: str, label: int) -> AttackResult:
    """CAT-04A (Tier B): Subdomain Prepending on Attacker Domain (Phishing Only)."""
    if label != 0:
        return AttackResult(url, url, False, False, "Tier B", "CAT-04A", "Subdomain Prepending (Phishing)", "Only applicable to Phishing URLs (label=0)")

    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    host, port = _extract_host_and_port(netloc)
    if not host or _is_ipv4(host):
        return AttackResult(url, url, False, False, "Tier B", "CAT-04A", "Subdomain Prepending (Phishing)", "Host is IP or empty")

    new_host = "secure-login." + host
    new_netloc = f"{new_host}:{port}" if port is not None else new_host
    new_url = _recompose_url(scheme, new_netloc, path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier B", "CAT-04A", "Subdomain Prepending (Phishing)")


def transform_cat_04b_subdomain_benign(url: str, label: int) -> AttackResult:
    """CAT-04B (Tier C): Subdomain Prepending on Benign Domain (Exploratory / OOD)."""
    if label != 1:
        return AttackResult(url, url, False, False, "Tier C", "CAT-04B", "Subdomain Prepending (Benign)", "Only applicable to Legitimate URLs (label=1)")

    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    host, port = _extract_host_and_port(netloc)
    if not host or _is_ipv4(host):
        return AttackResult(url, url, False, False, "Tier C", "CAT-04B", "Subdomain Prepending (Benign)", "Host is IP or empty")

    new_host = "security-update." + host
    new_netloc = f"{new_host}:{port}" if port is not None else new_host
    new_url = _recompose_url(scheme, new_netloc, path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier C", "CAT-04B", "Subdomain Prepending (Benign)")


# ---------------------------------------------------------------------------
# Category 5: Path Manipulation & Structure Padding
# ---------------------------------------------------------------------------

def transform_cat_05a_dot_slash_path(url: str, label: int) -> AttackResult:
    """CAT-05A (Tier B): Dot-Slash Path Redundancy (RFC 3986 §5.2.4)."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    if not path or path == "/" or "/" not in path.lstrip("/"):
        return AttackResult(url, url, False, False, "Tier B", "CAT-05A", "Dot-Slash Path Redundancy", "No subdirectories in path")

    # Insert /./ after the first directory segment
    segments = path.split("/")
    # Find first non-empty segment
    for i in range(1, len(segments)):
        if segments[i]:
            segments.insert(i + 1, ".")
            break

    new_path = "/".join(segments)
    new_url = _recompose_url(scheme, netloc, new_path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier B", "CAT-05A", "Dot-Slash Path Redundancy")


def transform_cat_05b_directory_padding(url: str, label: int) -> AttackResult:
    """CAT-05B (Tier B): Path Directory Padding."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    if not path or path == "/":
        new_path = "/portal/account/service/"
    else:
        new_path = path.rstrip("/") + "/portal/service/"

    new_url = _recompose_url(scheme, netloc, new_path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier B", "CAT-05B", "Path Directory Padding")


def transform_cat_05c_dummy_fragment(url: str, label: int) -> AttackResult:
    """CAT-05C (Tier A): Dummy Anchor Fragment Appendage (RFC 3986 §3.5)."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    if fragment == "verification-checkpoint":
        return AttackResult(url, url, False, False, "Tier A", "CAT-05C", "Dummy Anchor Fragment", "Fragment already present")

    new_fragment = "verification-checkpoint"
    new_url = _recompose_url(scheme, netloc, path, query, new_fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier A", "CAT-05C", "Dummy Anchor Fragment")


# ---------------------------------------------------------------------------
# Category 6: Query Parameter Manipulation
# ---------------------------------------------------------------------------

def transform_cat_06a_tracking_parameters(url: str, label: int) -> AttackResult:
    """CAT-06A (Tier B): Benign Tracking Parameter Injection."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    tracking = "utm_source=security_notice&utm_medium=email&utm_campaign=verification_2026"
    new_query = f"{query}&{tracking}" if query else tracking
    new_url = _recompose_url(scheme, netloc, path, new_query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier B", "CAT-06A", "Tracking Parameter Injection")


def transform_cat_06b_entropy_noise_query(url: str, label: int) -> AttackResult:
    """CAT-06B (Tier B): High-Entropy Noise Query Padding."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    # 64-character pseudo-random alphanumeric token
    noise_param = "session_token=8f9a2b4c6e1d3f5a7b9c2d4e6f8a1b3c5e7d9f2a4b6c8e1d3f5a7b9c2d4e6f8a"
    new_query = f"{query}&{noise_param}" if query else noise_param
    new_url = _recompose_url(scheme, netloc, path, new_query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier B", "CAT-06B", "High-Entropy Noise Query")


# ---------------------------------------------------------------------------
# Category 7: Keyword Manipulation
# ---------------------------------------------------------------------------

def transform_cat_07a_keyword_stuffing(url: str, label: int) -> AttackResult:
    """CAT-07A (Tier B): Benign Security Keyword Stuffing."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    stuffing = "secure-portal-official-verification"
    if not path or path == "/":
        new_path = f"/{stuffing}/"
    else:
        new_path = f"/{stuffing}{path}" if path.startswith("/") else f"/{stuffing}/{path}"

    new_url = _recompose_url(scheme, netloc, new_path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier B", "CAT-07A", "Security Keyword Stuffing")


SUSPICIOUS_KEYWORDS = ["login", "signin", "verify", "account", "banking", "update", "security", "admin"]


def transform_cat_07b_keyword_stripping(url: str, label: int) -> AttackResult:
    """CAT-07B (Tier B): Suspicious Keyword Stripping / Obfuscation."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    modified = False
    new_path = path
    new_query = query

    for kw in SUSPICIOUS_KEYWORDS:
        if kw in new_path.lower():
            # Replace case-insensitively with neutral token
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            new_path = pattern.sub("endpoint", new_path)
            modified = True
        if kw in new_query.lower():
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            new_query = pattern.sub("endpoint", new_query)
            modified = True

    if not modified:
        return AttackResult(url, url, False, False, "Tier B", "CAT-07B", "Suspicious Keyword Stripping", "No suspicious keywords present")

    new_url = _recompose_url(scheme, netloc, new_path, new_query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier B", "CAT-07B", "Suspicious Keyword Stripping")


# ---------------------------------------------------------------------------
# Category 8: IP Address Representation
# ---------------------------------------------------------------------------

def transform_cat_08a_standard_ipv4(url: str, label: int) -> AttackResult:
    """CAT-08A (Tier A): Standard IPv4 Dotted-Decimal Representation."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    host, port = _extract_host_and_port(netloc)
    if not _is_ipv4(host):
        return AttackResult(url, url, False, False, "Tier A", "CAT-08A", "Standard IPv4 Baseline", "Host is not IPv4")

    # Host is already standard IPv4
    return AttackResult(url, url, True, True, "Tier A", "CAT-08A", "Standard IPv4 Baseline")


def transform_cat_08b_alternative_ip_format(url: str, label: int) -> AttackResult:
    """CAT-08B (Tier C): Alternative IP Format (Hexadecimal / DWORD)."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    host, port = _extract_host_and_port(netloc)
    if not _is_ipv4(host):
        return AttackResult(url, url, False, False, "Tier C", "CAT-08B", "Alternative IP Format", "Host is not an IPv4 address")

    # Convert IPv4 to 32-bit DWORD integer
    ip_int = int(ipaddress.IPv4Address(host))
    new_host = str(ip_int)
    new_netloc = f"{new_host}:{port}" if port is not None else new_host
    new_url = _recompose_url(scheme, new_netloc, path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier C", "CAT-08B", "Alternative IP Format")


# ---------------------------------------------------------------------------
# Category 9: Transport & Structural Truncation
# ---------------------------------------------------------------------------

def transform_cat_09a_standard_port_injection(url: str, label: int) -> AttackResult:
    """CAT-09A (Tier A): Standard Port Injection (RFC 3986 §3.2.3)."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    host, port = _extract_host_and_port(netloc)
    if port is not None:
        return AttackResult(url, url, False, False, "Tier A", "CAT-09A", "Standard Port Injection", "Port already explicitly present")

    scheme_lower = scheme.lower() if scheme else "http"
    if scheme_lower == "http":
        default_port = 80
    elif scheme_lower == "https":
        default_port = 443
    else:
        return AttackResult(url, url, False, False, "Tier A", "CAT-09A", "Standard Port Injection", f"Non-standard scheme '{scheme}'")

    new_netloc = f"{host}:{default_port}"
    new_url = _recompose_url(scheme, new_netloc, path, query, fragment, had_scheme)
    return AttackResult(url, new_url, True, True, "Tier A", "CAT-09A", "Standard Port Injection")


def transform_cat_09b_apex_path_truncation(url: str, label: int) -> AttackResult:
    """CAT-09B (Tier C): Apex Path Truncation (Root Stripping)."""
    scheme, netloc, path, query, fragment, had_scheme = _decompose_url(url)
    if not path or path == "/" and not query and not fragment:
        return AttackResult(url, url, False, False, "Tier C", "CAT-09B", "Apex Path Truncation", "URL is already at apex root")

    new_path = "/"
    new_url = _recompose_url(scheme, netloc, new_path, "", "", had_scheme)
    return AttackResult(url, new_url, True, True, "Tier C", "CAT-09B", "Apex Path Truncation")


# ---------------------------------------------------------------------------
# Central Attack Registry
# ---------------------------------------------------------------------------

ATTACK_REGISTRY: Dict[str, AttackDefinition] = {
    "CAT-01A": AttackDefinition("CAT-01A", "Path Unreserved Encoding", "Tier A", "RFC 3986 §2.3", transform_cat_01a_unreserved_encoding),
    "CAT-01B": AttackDefinition("CAT-01B", "Path Reserved Char Encoding", "Tier B", "RFC 3986 §2.2", transform_cat_01b_reserved_encoding),
    "CAT-01C": AttackDefinition("CAT-01C", "Double Percent-Encoding", "Tier C", "Non-standard / double-decode", transform_cat_01c_double_encoding),
    "CAT-02A": AttackDefinition("CAT-02A", "Scheme Case Variation", "Tier A", "RFC 3986 §3.1", transform_cat_02a_scheme_casing),
    "CAT-02B": AttackDefinition("CAT-02B", "Hostname Case Variation", "Tier A", "RFC 4343 §2", transform_cat_02b_hostname_casing),
    "CAT-02C": AttackDefinition("CAT-02C", "Redundant Slashes", "Tier B", "RFC 3986 §3.3 normalization", transform_cat_02c_redundant_slashes),
    "CAT-03A": AttackDefinition("CAT-03A", "Cyrillic Lookalike Homoglyph", "Tier C", "Unicode / IDN lookalike", transform_cat_03a_cyrillic_homoglyph),
    "CAT-03B": AttackDefinition("CAT-03B", "Punycode Equivalence", "Tier A", "RFC 3492 / RFC 5891", transform_cat_03b_punycode_equivalence),
    "CAT-04A": AttackDefinition("CAT-04A", "Subdomain Prepending (Phishing)", "Tier B", "DNS attacker control", transform_cat_04a_subdomain_phishing),
    "CAT-04B": AttackDefinition("CAT-04B", "Subdomain Prepending (Benign)", "Tier C", "DNS invalidity / OOD", transform_cat_04b_subdomain_benign),
    "CAT-05A": AttackDefinition("CAT-05A", "Dot-Slash Redundancy", "Tier B", "RFC 3986 §5.2.4 dot-segments", transform_cat_05a_dot_slash_path),
    "CAT-05B": AttackDefinition("CAT-05B", "Path Directory Padding", "Tier B", "Hierarchical path padding", transform_cat_05b_directory_padding),
    "CAT-05C": AttackDefinition("CAT-05C", "Dummy Anchor Fragment", "Tier A", "RFC 3986 §3.5 client-side", transform_cat_05c_dummy_fragment),
    "CAT-06A": AttackDefinition("CAT-06A", "Tracking Parameter Injection", "Tier B", "Marketing query semantics", transform_cat_06a_tracking_parameters),
    "CAT-06B": AttackDefinition("CAT-06B", "Entropy Noise Query Padding", "Tier B", "Dynamic token padding", transform_cat_06b_entropy_noise_query),
    "CAT-07A": AttackDefinition("CAT-07A", "Security Keyword Stuffing", "Tier B", "Lexical token injection", transform_cat_07a_keyword_stuffing),
    "CAT-07B": AttackDefinition("CAT-07B", "Suspicious Keyword Stripping", "Tier B", "Lexical token obfuscation", transform_cat_07b_keyword_stripping),
    "CAT-08A": AttackDefinition("CAT-08A", "Standard IPv4 Baseline", "Tier A", "RFC 791 IPv4 dotted-decimal", transform_cat_08a_standard_ipv4),
    "CAT-08B": AttackDefinition("CAT-08B", "Alternative IP Format (DWORD)", "Tier C", "OS socket legacy format", transform_cat_08b_alternative_ip_format),
    "CAT-09A": AttackDefinition("CAT-09A", "Standard Port Injection", "Tier A", "RFC 3986 §3.2.3 default port", transform_cat_09a_standard_port_injection),
    "CAT-09B": AttackDefinition("CAT-09B", "Apex Path Truncation", "Tier C", "Root directory stripping", transform_cat_09b_apex_path_truncation),
}


def apply_attack(category_id: str, url: str, label: int) -> AttackResult:
    """Dispatch URL to the registered attack transformation by Category ID."""
    if category_id not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack category ID: '{category_id}'. Valid IDs: {list(ATTACK_REGISTRY.keys())}")
    return ATTACK_REGISTRY[category_id].transform_fn(url, label)

