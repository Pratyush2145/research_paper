# Comprehensive Feature Dictionary: URL-Only Phishing Detection

**Project:** Explainable and Generalization-Aware Phishing URL Detection Using Machine Learning and SHAP  
**Phase:** 3 — URL-Only Feature Engineering & Leakage-Safe Dataset Preparation  
**Total Features:** 68 Deterministic URL-Only Features  
**Operational Constraint:** Zero network requests, zero DNS/WHOIS queries, zero HTML/DOM dependence, zero domain identity memorization.

---

## 1. Feature Design Principles and Integrity Rules

Every feature in this dictionary adheres to four strict research criteria:
1. **Strictly URL-Only:** Computed purely from the input character string before any network connection is attempted.
2. **Deterministic and Stateless:** For any given URL string $u$, the feature extraction function $f(u)$ always returns the exact same vector without random seed dependence or temporal drift.
3. **Domain-Identity Agnostic:** No feature represents, encodes, or memorizes the specific registrable domain name (e.g. `google.com` or `paypal.com`). Features capture structural, statistical, and lexical properties rather than identity tokens.
4. **Leakage-Safe:** No feature depends on target labels, dataset provenance, or row identifiers.

---

## 2. Feature Summary by Category

| Category | Count | Data Types | Representation |
| :--- | :---: | :--- | :--- |
| **A. URL Length Features** | 5 | Integer (`int64`) | Component character lengths |
| **B. Character-Based Features** | 20 | Integer (`int64`), Float (`float64`) | Character frequencies, ratios, repetitions, entropy |
| **C. Hostname Features** | 9 | Integer (`int64`), Float (`float64`) | Subdomain depth, label statistics, IP indicators |
| **D. Path Features** | 9 | Integer (`int64`), Float (`float64`) | Segment depth, lengths, extensions, special chars |
| **E. Query Parameter Features** | 5 | Integer (`int64`) | Parameter counts, separators, value heuristics |
| **F. Suspicious Lexical Patterns** | 12 | Integer (`int64`) | Targeted security keyword flags and aggregate count |
| **G. Protocol and Parsing Features** | 8 | Integer (`int64`) | Scheme flags, categorical code, authority markers |
| **Total Features** | **68** | — | **55 Integer / Binary, 13 Continuous Float** |

---

## 3. Detailed Feature Specifications

### Category A: URL Length Features

#### 1. `url_length`
- **Description:** Total character count of the input URL string.
- **Definition:** $|u| = \text{len}(u)$
- **Type:** Numerical (`int64`)
- **Expected Range:** $[1, \infty)$, typical range $[10, 250]$
- **Relevance:** Phishing URLs frequently exhibit excessive length due to deep redirection chains, embedded tokens, obfuscated paths, or embedded target domains.
- **Limitations:** Legitimate search queries, tracking links, and deep API URLs can also be exceptionally long.

#### 2. `hostname_length`
- **Description:** Character count of the parsed host component.
- **Definition:** $|h| = \text{len}(\text{hostname})$
- **Type:** Numerical (`int64`)
- **Expected Range:** $[0, 253]$ (RFC 1035 limit)
- **Relevance:** Phishers often chain long subdomains to mimic legitimate brand names or bypass domain reputation filters.
- **Limitations:** Legitimate Content Delivery Networks (CDNs) and cloud endpoints often have long hostnames.

#### 3. `path_length`
- **Description:** Character count of the URL path component.
- **Definition:** $|p| = \text{len}(\text{path})$
- **Type:** Numerical (`int64`)
- **Expected Range:** $[0, \infty)$, typical range $[0, 150]$
- **Relevance:** Phishing kits are often deployed inside multi-tiered nested directories on compromised servers.
- **Limitations:** Legitimate REST APIs and content-heavy sites (e.g., news archives, e-commerce) have long path segments.

#### 4. `query_length`
- **Description:** Character count of the query component (excluding leading `?`).
- **Definition:** $|q| = \text{len}(\text{query})$
- **Type:** Numerical (`int64`)
- **Expected Range:** $[0, \infty)$, typical range $[0, 500]$
- **Relevance:** Phishers pass encoded base64 victim emails or tracking credentials inside query parameters.
- **Limitations:** Legitimate analytics tags (UTM parameters) frequently produce long query strings.

#### 5. `fragment_length`
- **Description:** Character count of the fragment identifier (excluding leading `#`).
- **Definition:** $|fr| = \text{len}(\text{fragment})$
- **Type:** Numerical (`int64`)
- **Expected Range:** $[0, \infty)$, typical range $[0, 100]$
- **Relevance:** Client-side routing phishing kits use fragments to hide victim identifiers from server access logs.
- **Limitations:** Modern single-page web applications (SPAs) use fragments for routing.

---

### Category B: Character-Based Features

#### 6. `dot_count`
- **Description:** Count of period characters (`.`) in the entire URL.
- **Definition:** $\sum \mathbb{I}(c_i = '.')$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Multiple dots signal excessive subdomains, IP hosts, or nested file extensions.
- **Limitations:** File repositories and mirrors naturally contain multiple dots.

#### 7. `hyphen_count`
- **Description:** Count of hyphen characters (`-`) in the URL.
- **Definition:** $\sum \mathbb{I}(c_i = '-')$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Hyphens are heavily favored in cybersquatting and typosquatting (e.g. `paypal-security-update.com`).
- **Limitations:** Legitimate domain names commonly use hyphens for multi-word brands.

#### 8. `underscore_count`
- **Description:** Count of underscore characters (`_`) in the URL.
- **Definition:** $\sum \mathbb{I}(c_i = '\_')$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Underscores are rare in legitimate hostnames (invalid by RFC 952/1123) and more common in obfuscated paths.
- **Limitations:** Legitimate query parameter keys frequently use underscores (e.g. `user_id`).

#### 9. `slash_count`
- **Description:** Count of forward slash characters (`/`) in the URL.
- **Definition:** $\sum \mathbb{I}(c_i = '/')$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Measures directory depth and path nesting.
- **Limitations:** Complex legitimate websites use deeply nested directories.

#### 10. `backslash_count`
- **Description:** Count of backslash characters (`\`) in the URL.
- **Definition:** $\sum \mathbb{I}(c_i = '\setminus')$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Windows path traversal attempts, open redirects, or parser evasion tactics.
- **Limitations:** Rare in modern web links.

#### 11. `at_symbol_count`
- **Description:** Count of at symbols (`@`) in the URL.
- **Definition:** $\sum \mathbb{I}(c_i = '@')$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** In HTTP URLs, the `@` symbol precedes the host, causing browsers to ignore the preceding text (a classic spoofing tactic).
- **Limitations:** Legitimate mailto links or specific API tokens.

#### 12. `question_mark_count`
- **Description:** Count of question mark characters (`?`).
- **Definition:** $\sum \mathbb{I}(c_i = '?')$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** More than one question mark indicates parameter confusion or malformed obfuscation.
- **Limitations:** Some dynamic web apps pass nested URLs in queries.

#### 13. `equal_count`
- **Description:** Count of equal sign characters (`=`).
- **Definition:** $\sum \mathbb{I}(c_i = '=')$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Indicates the density of key-value pairs in query strings.
- **Limitations:** Tracking URLs (e.g. affiliate marketing) have high equal sign counts.

#### 14. `ampersand_count`
- **Description:** Count of ampersand characters (`&`).
- **Definition:** $\sum \mathbb{I}(c_i = '\&')$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Parameter delimiters; reflects parameter volume.
- **Limitations:** Directly correlates with query length.

#### 15. `percent_count`
- **Description:** Count of percent sign characters (`%`).
- **Definition:** $\sum \mathbb{I}(c_i = '\%')$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Measures percent-encoding density. Phishers use URL encoding to disguise IP addresses or suspicious keywords.
- **Limitations:** Non-ASCII alphabets (Cyrillic, Arabic, Chinese) are legitimately percent-encoded.

#### 16. `digit_count`
- **Description:** Total numeric digits (`0-9`) in the URL.
- **Definition:** $\sum \mathbb{I}(c_i \in [0-9])$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Phishing URLs frequently contain IP addresses, timestamp IDs, or random digit hashes.
- **Limitations:** E-commerce product IDs and dates naturally contain digits.

#### 17. `alpha_count`
- **Description:** Total alphabetic characters (`a-zA-Z`) in the URL.
- **Definition:** $\sum \mathbb{I}(c_i \in [a-zA-Z])$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Baseline text length measure for ratio computations.
- **Limitations:** Correlates strongly with total length.

#### 18. `special_char_count`
- **Description:** Total non-alphanumeric characters excluding standard scheme/path delimiters (`:`, `/`).
- **Definition:** $|u| - (\text{alpha\_count} + \text{digit\_count} + \text{count(':')} + \text{count('/')})$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Dense special characters often signal evasion techniques.
- **Limitations:** Legitimate complex queries contain symbols.

#### 19. `digit_ratio`
- **Description:** Ratio of digits to total URL length.
- **Definition:** $\text{digit\_count} / |u|$ (0 if $|u| = 0$)
- **Type:** Continuous (`float64`) | **Range:** $[0.0, 1.0]$
- **Relevance:** Normalizes digit presence across URLs of varying lengths.
- **Limitations:** Short URLs with an IP host will have high digit ratios.

#### 20. `alpha_ratio`
- **Description:** Ratio of alphabetic characters to total URL length.
- **Definition:** $\text{alpha\_count} / |u|$ (0 if $|u| = 0$)
- **Type:** Continuous (`float64`) | **Range:** $[0.0, 1.0]$
- **Relevance:** Human-readable URLs have high alpha ratios; obfuscated or IP links have low ratios.
- **Limitations:** Natural variations across languages.

#### 21. `special_char_ratio`
- **Description:** Ratio of special characters to total URL length.
- **Definition:** $\text{special\_char\_count} / |u|$ (0 if $|u| = 0$)
- **Type:** Continuous (`float64`) | **Range:** $[0.0, 1.0]$
- **Relevance:** High values indicate structural complexity or heavy encoding.
- **Limitations:** Highly parameterized legitimate URLs.

#### 22. `max_consecutive_repeated_char`
- **Description:** Length of the longest run of consecutive identical characters.
- **Definition:** $\max \{k : c_i = c_{i+1} = \dots = c_{i+k-1}\}$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Repeated characters (e.g. `googleeeee.com`, `////`, `-----`) indicate typosquatting or path padding.
- **Limitations:** Valid paths may contain double letters (e.g. `bookkeeper`).

#### 23. `consecutive_dots_count`
- **Description:** Occurrences of consecutive dots (`..`) in the URL.
- **Definition:** Count of substring `".."` in URL
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Path traversal attacks (`../..`) or DNS label evasion.
- **Limitations:** Rare in standard URLs.

#### 24. `consecutive_hyphens_count`
- **Description:** Occurrences of consecutive hyphens (`--`) in the URL.
- **Definition:** Count of substring `"--"` in URL
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Punycode indicators or deliberate brand spoofing (`paypal--login.com`).
- **Limitations:** Legitimate internationalized domain names (IDNs) legitimately contain `xn--`.

#### 25. `url_entropy`
- **Description:** Shannon entropy of the character distribution across the entire URL.
- **Definition:** $H(u) = -\sum_{c \in \Sigma} p(c) \log_2 p(c)$, where $p(c) = \text{count}(c) / |u|$.
- **Type:** Continuous (`float64`) | **Range:** $[0.0, 8.0]$
- **Relevance:** High entropy indicates algorithmic domain generation (DGAs), random tokens, or encrypted query strings.
- **Limitations:** Long English text in paths also produces moderately high entropy.

---

### Category C: Hostname Features

#### 26. `subdomain_count`
- **Description:** Count of subdomain labels preceding the registrable domain.
- **Definition:** $\max(0, \text{total\_host\_labels} - \text{registrable\_domain\_labels})$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Phishing sites frequently employ deep subdomains on free hosting providers (e.g. `paypal.verification.portal.example.com`).
- **Limitations:** Large enterprise architectures use multi-level subdomains.

#### 27. `hostname_label_count`
- **Description:** Total dot-separated segments in the hostname.
- **Definition:** Length of `hostname.split('.')`
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Reflects total DNS hierarchy depth.
- **Limitations:** Multi-part TLDs (e.g. `.co.uk`) increase count without indicating phishing.

#### 28. `max_hostname_label_length`
- **Description:** Character length of the longest dot-separated label in the hostname.
- **Definition:** $\max_{l \in \text{labels}} \text{len}(l)$
- **Type:** Numerical (`int64`) | **Range:** $[0, 63]$ (RFC 1035 label limit)
- **Relevance:** DGA domains or brand-stuffed labels exhibit unusually high lengths.
- **Limitations:** Single-word descriptive brand names can also be long.

#### 29. `avg_hostname_label_length`
- **Description:** Average character length of labels in the hostname.
- **Definition:** $(\sum_{l \in \text{labels}} \text{len}(l)) / \text{label\_count}$
- **Type:** Continuous (`float64`) | **Range:** $[0.0, 63.0]$
- **Relevance:** Balances label count against total host length.
- **Limitations:** Sensitive to short 2-letter TLDs.

#### 30. `is_ip_address`
- **Description:** Binary indicator for whether the hostname is an IPv4 or IPv6 address literal.
- **Definition:** $\mathbb{I}(\text{hostname is valid IP literal})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Legitimate web services rarely present bare IP addresses to consumers. Phishing campaigns frequently use direct IP hosting to evade domain reputation blocks.
- **Limitations:** Local development servers, router interfaces, and specialized IoT endpoints.

#### 31. `hostname_has_digits`
- **Description:** Binary indicator for whether the hostname contains numeric digits.
- **Definition:** $\mathbb{I}(\exists c \in \text{hostname} : c \in [0-9])$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Cybersquatters often append digits to brand names (e.g. `chase24-online.com`).
- **Limitations:** Legitimate companies with numbers in their names (e.g. `3m.com`, `7-eleven.com`).

#### 32. `hostname_has_hyphens`
- **Description:** Binary indicator for whether the hostname contains hyphens.
- **Definition:** $\mathbb{I}('-' \in \text{hostname})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Common indicator of compound typosquatting.
- **Limitations:** Commonly used in legitimate multi-word domain names.

#### 33. `hostname_has_punycode`
- **Description:** Binary indicator for presence of IDN punycode prefix (`xn--`) in hostname.
- **Definition:** $\mathbb{I}(\text{'xn--'} \in \text{hostname})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Homograph attacks use punycode to visually mimic legitimate Latin domains with Cyrillic or Greek characters (e.g. `xn--pypal-4ve.com`).
- **Limitations:** Legitimate non-Latin international domain names.

#### 34. `hostname_has_unusual_ip`
- **Description:** Binary indicator for non-standard IP formats (hexadecimal, octal, or 32-bit integer).
- **Definition:** $\mathbb{I}(\text{host matches hex, dword, or octal IP pattern})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Obfuscation technique designed to bypass basic IP regex filters while remaining resolvable by web browsers.
- **Limitations:** Extremely rare in benign traffic.

---

### Category D: Path Features

#### 35. `path_segment_count`
- **Description:** Count of non-empty `/`-separated directories in the URL path.
- **Definition:** Count of non-empty elements in `path.split('/')`
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Measures hierarchical directory depth.
- **Limitations:** Legitimate CMS platforms (e.g. WordPress) produce deep paths.

#### 36. `avg_path_segment_length`
- **Description:** Average character length of non-empty path segments.
- **Definition:** $(\sum_{s \in \text{segments}} \text{len}(s)) / \text{segment\_count}$ (0.0 if count is 0)
- **Type:** Continuous (`float64`) | **Range:** $[0.0, \infty)$
- **Relevance:** Phishing paths often feature long random directory names.
- **Limitations:** High on long RESTful resource names.

#### 37. `max_path_segment_length`
- **Description:** Length of the longest individual path segment.
- **Definition:** $\max_{s \in \text{segments}} \text{len}(s)$ (0 if count is 0)
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Identifies path tokens that embed encoded payloads or brand names.
- **Limitations:** Long article slugs in news publications.

#### 38. `path_encoded_char_count`
- **Description:** Count of `%` characters in the path component.
- **Definition:** Count of `"%"` in path
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Percent-encoding in paths is used to obscure malicious filenames.
- **Limitations:** Spaces and UTF-8 characters in filenames are percent-encoded.

#### 39. `path_digit_count`
- **Description:** Total numeric digits inside the path component.
- **Definition:** $\sum_{c \in \text{path}} \mathbb{I}(c \in [0-9])$
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Phishing kits often use numeric session IDs or random hashes in paths.
- **Limitations:** Database IDs, article dates, and version numbers.

#### 40. `path_special_char_count`
- **Description:** Total non-alphanumeric characters in the path excluding `/` and `.`.
- **Definition:** Count of special characters in path
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Obfuscated path strings with underscores, tildes, or dollar signs.
- **Limitations:** Hyphenated slugs.

#### 41. `is_path_empty`
- **Description:** Binary indicator for whether the URL lacks a path or targets the root (`/`).
- **Definition:** $\mathbb{I}(\text{path} \in \{\text{""}, \text{"/"}\})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Crucial structural feature. As observed in Phase 2, 88.36% of PhiUSIIL legitimate URLs have empty paths.
- **Limitations:** Must be evaluated in conjunction with domain reputation to avoid shortcut learning.

#### 42. `path_has_file_extension`
- **Description:** Binary indicator for presence of a recognized file extension in the path.
- **Definition:** $\mathbb{I}(\exists e \in \text{COMMON\_EXTENSIONS} : e \in \text{path.lower()})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Phishing URLs frequently point directly to executable files (`.exe`), scripts (`.php`), or archived packages (`.zip`).
- **Limitations:** Benign links frequently point to `.html`, `.pdf`, or `.php`.

#### 43. `path_file_extension_count`
- **Description:** Total count of recognized file extensions found in the path string.
- **Definition:** Total matches of known extensions in path
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Double extensions (e.g. `document.pdf.exe`) are a classic malware and phishing deception technique.
- **Limitations:** Rare in ordinary URLs.

---

### Category E: Query Parameter Features

#### 44. `query_param_count`
- **Description:** Count of key-value parameter pairs in the query string.
- **Definition:** Length of parsed parameter list (split on `&` and `;`)
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Phishing attacks forwarding stolen credentials or tokens often feature specific parameter counts.
- **Limitations:** Search engines and marketing trackers generate high parameter counts.

#### 45. `query_param_separator_count`
- **Description:** Count of parameter delimiters (`&` and `;`) in the query string.
- **Definition:** Count of `&` + Count of `;` in query
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Reflects query complexity and parameter volume.
- **Limitations:** Correlates with parameter count.

#### 46. `query_encoded_char_count`
- **Description:** Count of percent sign characters (`%`) in the query component.
- **Definition:** Count of `"%"` in query
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** High values suggest encoded redirect URLs or obfuscated scripts passed via parameters.
- **Limitations:** Complex search queries with spaces and symbols.

#### 47. `query_has_long_value`
- **Description:** Binary indicator for whether any parameter key or value exceeds 50 characters.
- **Definition:** $\mathbb{I}(\exists (k, v) : \text{len}(k) > 50 \lor \text{len}(v) > 50)$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Long query values often contain base64 payloads, encrypted tokens, or whole nested URLs.
- **Limitations:** OAuth state tokens and session JWTs are legitimately long.

#### 48. `query_numeric_value_count`
- **Description:** Count of query parameters whose values are purely numeric.
- **Definition:** Count of parameters where `value.isdigit() == True`
- **Type:** Numerical (`int64`) | **Range:** $[0, \infty)$
- **Relevance:** Distinguishes simple pagination/ID queries from complex tokenized payloads.
- **Limitations:** Common in database-driven websites.

---

### Category F: Suspicious Lexical Pattern Features

These features capture heuristic keyword indicators. They do not constitute definitive proof of phishing, but provide probabilistic signals when combined with structural features.

- **49. `has_kw_login`**: $\mathbb{I}(\text{'login'} \in \text{URL})$
- **50. `has_kw_signin`**: $\mathbb{I}(\text{'signin'} \in \text{URL})$
- **51. `has_kw_verify`**: $\mathbb{I}(\text{'verify'} \in \text{URL})$
- **52. `has_kw_account`**: $\mathbb{I}(\text{'account'} \in \text{URL})$
- **53. `has_kw_update`**: $\mathbb{I}(\text{'update'} \in \text{URL})$
- **54. `has_kw_secure`**: $\mathbb{I}(\text{'secure'} \in \text{URL})$
- **55. `has_kw_banking`**: $\mathbb{I}(\text{'banking'} \in \text{URL})$
- **56. `has_kw_payment`**: $\mathbb{I}(\text{'payment'} \in \text{URL})$
- **57. `has_kw_password`**: $\mathbb{I}(\text{'password'} \in \text{URL})$
- **58. `has_kw_confirm`**: $\mathbb{I}(\text{'confirm'} \in \text{URL})$
- **59. `has_kw_authentication`**: $\mathbb{I}(\text{'authentication'} \in \text{URL})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Phishing attacks target authentication, credential harvesting, and financial verification. Attackers embed these target words in subdomains or paths to build false credibility.
- **Limitations:** Legitimate banks and web services also host genuine `/login` and `/account` endpoints.

#### 60. `suspicious_keyword_count`
- **Description:** Total count of suspicious security keywords appearing across the URL string.
- **Definition:** $\sum_{kw \in \text{KEYWORDS}} \mathbb{I}(kw \in \text{lower\_url})$
- **Type:** Numerical (`int64`) | **Range:** $[0, 11]$
- **Relevance:** Phishing URLs frequently stack multiple security terms (e.g. `secure-banking-login-verify.com`).
- **Limitations:** Financial portal landing pages may legitimately mention multiple terms.

---

### Category G: Protocol and Parsing Features

#### 61. `is_https`
- **Description:** Binary indicator for whether the URL uses the `https://` scheme.
- **Definition:** $\mathbb{I}(\text{scheme} = \text{'https'})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Historical phishing heavily used unencrypted HTTP. However, modern phishing predominantly uses free HTTPS certificates (Let's Encrypt).
- **Limitations:** HTTPS does NOT guarantee legitimacy; it only guarantees encrypted transport.

#### 62. `is_http`
- **Description:** Binary indicator for whether the URL uses unencrypted `http://`.
- **Definition:** $\mathbb{I}(\text{scheme} = \text{'http'})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Major legitimate web brands enforce HTTPS redirects. Plain HTTP links for credential submission are disproportionately suspicious.
- **Limitations:** Older legacy websites.

#### 63. `is_missing_protocol`
- **Description:** Binary indicator for whether the original raw URL lacked an explicit protocol prefix.
- **Definition:** $\mathbb{I}(\text{raw URL had no scheme prefix})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Links shared via SMS or messaging apps often omit the scheme.
- **Limitations:** Common formatting artifact in datasets.

#### 64. `protocol_type`
- **Description:** Categorical integer code representing the protocol scheme.
- **Definition:** $0 = \text{none/unknown}, 1 = \text{http}, 2 = \text{https}, 3 = \text{other}$
- **Type:** Categorical / Integer (`int64`) | **Range:** $\{0, 1, 2, 3\}$
- **Relevance:** Provides a single unified ordinal signal for tree-based models.
- **Limitations:** Non-linear models handle this well; linear models should use one-hot indicators.

#### 65. `has_userinfo`
- **Description:** Binary indicator for presence of user authentication credentials in the authority component.
- **Definition:** $\mathbb{I}(\text{'@'} \in \text{authority})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** RFC 3986 allows `user:pass@host`. Phishers exploit this to display `https://paypal.com@attacker.com`, tricking victims who only read the left side.
- **Limitations:** Rare in legitimate modern web traffic.

#### 66. `has_port`
- **Description:** Binary indicator for presence of an explicit port specification (e.g. `:8080`, `:8443`).
- **Definition:** $\mathbb{I}(\text{parsed.port is not None})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Phishing servers frequently run on non-standard ports (e.g. 8080, 8888, 5000) to avoid standard web server detection.
- **Limitations:** Development endpoints.

#### 67. `has_fragment`
- **Description:** Binary indicator for presence of a fragment identifier (`#`).
- **Definition:** $\mathbb{I}(\text{len(fragment)} > 0)$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Used in single-page phishing apps to conceal parameters from server logs.
- **Limitations:** Standard on documentation and modern web applications.

#### 68. `url_parse_success`
- **Description:** Binary indicator for whether the URL was successfully parsed into valid RFC components.
- **Definition:** $\mathbb{I}(\text{urlsplit completed without fatal syntax failure and valid host was identified})$
- **Type:** Binary (`int64`) | **Range:** $\{0, 1\}$
- **Relevance:** Severe obfuscation, illegal characters, or binary injection can cause URL parser failures.
- **Limitations:** Most strings can be weakly parsed by `urlsplit`.

