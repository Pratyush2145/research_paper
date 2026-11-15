# Comprehensive Dataset Audit and Empirical Validation Report

**Project:** Explainable and Generalization-Aware Phishing URL Detection Using Machine Learning and SHAP  
**Phase:** 2 — Dataset Acquisition, Inspection, and Validation  
**Date of Audit:** 2026-11-14  
**Audit Protocol:** Offline inspection strictly; no URL visitation, DNS lookup, or external requests.

---

## Executive Summary of Audit Findings

| Audit Metric | Dataset A: LegitPhish (Primary) | Dataset B: PhiUSIIL (External Validation) | Cross-Dataset Comparison / Leakage |
| :--- | :--- | :--- | :--- |
| **Official Source** | Mendeley Data (`hx4m73v2sf/2`) | UCI ML Repository (`967`) | Multi-repository evaluation |
| **Cryptographic Status** | Verified SHA-256 match | Verified SHA-256 match | Bit-for-bit authenticity confirmed |
| **Raw File Size** | 8,963,838 bytes (~8.55 MiB) | 56,854,345 bytes (~54.22 MiB) | Uncompressed CSV sizes |
| **Actual Row Count** | 101,219 records | 235,795 records | Matches reported counts |
| **Actual Column Count** | 18 columns | 56 columns | PhiUSIIL includes 54 precomputed features |
| **URL-Only Feature Scope** | `URL` + `ClassLabel` | `URL` + `label` | Strict exclusion of webpage features |
| **Target Label Convention** | 0 = Phishing, 1 = Legitimate | 0 = Phishing, 1 = Legitimate | Identical class definition |
| **Class Distribution** | Phishing: 63,678 (62.91%)<br>Legitimate: 37,540 (37.09%)<br>Missing/NaN: 1 (0.001%) | Phishing: 100,945 (42.81%)<br>Legitimate: 134,850 (57.19%)<br>Missing: 0 (0.00%) | LegitPhish is phishing-majority;<br>PhiUSIIL is legitimate-majority |
| **Missing Values** | Exactly 1 (`ClassLabel` at row 65118) | 0 missing across all 56 columns | Single missing record documented |
| **Exact Duplicate Rows** | 346 rows | 0 rows | LegitPhish has 346 redundant entries |
| **Duplicate URL Strings** | 346 URLs | 425 URLs (850 total records) | PhiUSIIL has duplicate URLs with varied metadata |
| **Conflicting-Label URLs** | 0 URLs (0.00%) | 0 URLs (0.00%) | Zero intra-dataset URL ambiguity |
| **Unique Registrable Domains** | 62,156 domains | 172,296 domains | 35,925 domains overlap between datasets |
| **Repeated Domains (Count > 1)** | 13,015 domains | 6,210 domains | High sub-page representation |
| **Conflicting-Label Domains** | 30 domains | 223 domains | Domains hosting both benign & malicious URLs |
| **Cross-Dataset URL Overlap** | — | — | **36,936 identical URLs** (all Legitimate) |
| **Domain Overlap Ratio** | **57.80%** of LegitPhish domains | **20.85%** of PhiUSIIL domains | **Critical data leakage vulnerability** |

---

## 1. Audit Methodology and Rules

All inspection scripts were executed in the project's Python virtual environment (`.venv`) using `pandas`, `numpy`, and standard library offline URL parsing (`urllib.parse`, `ipaddress`).
- **No Network Requests:** No URL found in either dataset was resolved, pinged, or retrieved.
- **Label Preservation:** Label semantics (0 = Phishing, 1 = Legitimate) were strictly preserved without inversion.
- **Offline Domain Extraction:** Registrable domains (eTLD+1) were extracted offline using a multi-part ccTLD dictionary (`.co.uk`, `.gov.in`, etc.) and IP parsing.

---

## 2. Dataset A — LegitPhish (Primary Dataset) Audit

### 2.1 Basic File Information
- **Raw File:** `data/raw/legitphish/url_features_extracted1.csv`
- **File Size:** 8,963,838 bytes (~8.96 MB)
- **SHA-256 Digest:** `8685e7901c3b4e73eb43c106cb166cc932dd14f76960a032c713c2c32f09dbcd`
- **Reported vs. Actual:**
  - *Reported:* ~101,219 URL records, 18 columns.
  - *Actual:* Exactly 101,219 rows and 18 columns.

### 2.2 Column Schema and Data Types
1. `URL` (*object/string*): Full URL string.
2. `url_length` (*int64*): Character count of URL.
3. `has_ip_address` (*int64*): Binary flag for IP presence.
4. `dot_count` (*int64*): Count of period characters.
5. `https_flag` (*int64*): Binary flag for HTTPS scheme.
6. `url_entropy` (*float64*): Shannon entropy of URL string.
7. `token_count` (*int64*): Token/word count in URL.
8. `subdomain_count` (*int64*): Count of subdomains.
9. `query_param_count` (*int64*): Count of parameters in query string.
10. `tld_length` (*int64*): Length of top-level domain.
11. `path_length` (*int64*): Length of URL path.
12. `has_hyphen_in_domain` (*int64*): Binary flag for domain hyphens.
13. `number_of_digits` (*int64*): Count of numeric digits.
14. `tld_popularity` (*int64*): Binary flag for popular TLDs.
15. `suspicious_file_extension` (*int64*): Binary flag for risky extensions.
16. `domain_name_length` (*int64*): Character length of domain.
17. `percentage_numeric_chars` (*float64*): Numeric character ratio.
18. `ClassLabel` (*float64*): Target class label.

### 2.3 Label Distribution and Discrepancies
- **Phishing (`0.0`):** 63,678 records (**62.91%**)
- **Legitimate (`1.0`):** 37,540 records (**37.09%**)
- **Missing (`NaN`):** 1 record (**0.001%**)
  - *Reported Stat:* The Mendeley repository description states: *"Phishing (0): 63,678 URLs"*. 
  - *Discrepancy Documented:* The reported total was 101,219. Subtracting 63,678 leaves 37,541. However, exactly one record has a `NaN` label.
  - *Identified Row:* Row index `65118`: `URL = "https://www.chocolatleroux.com"`, `ClassLabel = NaN`. All other 17 numeric features are populated, but the label is null. During preprocessing, this row must be safely dropped.

### 2.4 Data Quality and Structural Formatting
- **Exact Duplicate Rows:** 346 rows are 100% duplicate across all 18 columns.
- **Duplicate URL Strings:** Exactly 346 duplicate URLs (identical to the exact duplicate rows).
- **Conflicting-Label URLs:** 0 URLs (no single URL string is assigned both 0 and 1).
- **Malformed / Unusual URLs:** 11 records possess unusual formatting:
  - Row 599: `URL = 'Offline'` (labeled 0.0) — non-URL string.
  - Row 1313: `URL = '772050312/www1.royalbank.com/cgi-bin/rbaccess/rbunxcgi/'` (labeled 0.0) — prepended identifier.
  - Row 2069: `URL = 'http://http://comodozeropoint.com/updates/96313727/aqq/awq.exe'` (labeled 0.0) — doubled protocol.
  - Row 2092: `URL = 'http://http:/ronnystein.de/kali/something32.exe'` (labeled 0.0) — malformed protocol.
  - Rows 2226, 2229, 2230, 2231, 2288, 2291: `http://http://69.165.65.90/bins/...` (labeled 0.0) — doubled protocol with IP.
  - Row 4386: `URL = 'ttp://103.149.87.69/la.bot.arc'` (labeled 0.0) — truncated scheme (`ttp://`).
  - *Observation:* All 11 malformed records are phishing samples.

### 2.5 URL Component Profiling
- **Schemes:** `http` (58,859; 58.15%), `https` (41,059; 40.56%), `none/missing` (1,300; 1.28%), `ttp` (1).
- **Subdomains Present:** 98,265 URLs (97.08%).
- **Non-root Paths Present:** 63,813 URLs (63.04%).
- **Query Strings Present:** 800 URLs (0.79%).
- **Fragments Present:** 851 URLs (0.84%).

### 2.6 Domain-Level Analysis
- **Unique Registrable Domains:** 62,156 domains.
- **Repeated Domains:** 13,015 domains appear multiple times.
  - Top repeated domains: `cumpar-auto-orice-tip.ro` (965 URLs), `healthnlife.pk` (924 URLs), `powerforxes.shop` (770 URLs), `uniformeslaamistad.com` (768 URLs), `mugutu.com` (766 URLs).
- **Conflicting-Label Domains:** 30 domains appear with both phishing and legitimate labels (e.g., `altervista.org`, `archive.org`, `co.il`, `co.id`, `ac.th`).
  - *Cause:* Public hosting providers, web archives, and delegated ccTLD zones.

---

## 3. Dataset B — PhiUSIIL (External Validation Dataset) Audit

### 3.1 Basic File Information
- **Raw File:** `data/raw/phiusil/PhiUSIIL_Phishing_URL_Dataset.csv` (extracted from `phiusiil+phishing+url+dataset.zip`)
- **File Size:** 56,854,345 bytes (~54.22 MB)
- **SHA-256 Digest:** `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`
- **Reported vs. Actual:**
  - *Reported:* 235,795 records, 54 features.
  - *Actual:* Exactly 235,795 rows and 56 columns (1 identifier `FILENAME`, 1 raw string `URL`, 53 features, 1 `label`).

### 3.2 URL-Only Enforcement & Column Separation
PhiUSIIL includes 54 precomputed features extracted from webpage source code, DOM structure, and external network responses. In strict accordance with Section 5 of the research specification:
- **Retained Columns:** `URL` and `label`.
- **Excluded Columns (54 features):** `FILENAME`, `URLLength`, `Domain`, `DomainLength`, `IsDomainIP`, `TLD`, `URLSimilarityIndex`, `CharContinuationRate`, `TLDLegitimateProb`, `URLCharProb`, `TLDLength`, `NoOfSubDomain`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `IsHTTPS`, `LineOfCode`, `LargestLineLength`, `HasTitle`, `Title`, `DomainTitleMatchScore`, `URLTitleMatchScore`, `HasFavicon`, `Robots`, `IsResponsive`, `NoOfURLRedirect`, `NoOfSelfRedirect`, `HasDescription`, `NoOfPopup`, `NoOfiFrame`, `HasExternalFormSubmit`, `HasSocialNet`, `HasSubmitButton`, `HasHiddenFields`, `HasPasswordField`, `Bank`, `Pay`, `Crypto`, `HasCopyrightInfo`, `NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfSelfRef`, `NoOfEmptyRef`, `NoOfExternalRef`.
- **Reason:** Webpage source code, title matching, and dynamic DOM features violate the operational requirements of pre-click, zero-footprint URL detection.

### 3.3 Label Distribution
- **Phishing (`0`):** 100,945 records (**42.81%**)
- **Legitimate (`1`):** 134,850 records (**57.19%**)
- **Missing Values:** 0 across all columns (**100% complete**)
- **Reported vs Actual Match:** Exact match with reported figures (100,945 phishing and 134,850 legitimate).

### 3.4 Data Quality and Duplicates
- **Exact Duplicate Rows:** 0 rows (when considering all 56 columns).
- **Duplicate URL Strings:** 425 URL strings appear more than once (representing 850 total rows).
  - *Cause:* Identical URLs were re-crawled at different timestamps, resulting in different `FILENAME` and slight variations in webpage DOM features (e.g. `NoOfJS`, `NoOfCSS`).
- **Conflicting-Label URLs:** 0 URLs (no URL in PhiUSIIL is labeled both 0 and 1).
- **Invalid Domain Values:** 0 unparseable domains.

### 3.5 URL Component Profiling
- **Schemes:** `https` (184,046; 78.05%), `http` (51,749; 21.95%).
- **Subdomains Present:** 221,808 URLs (94.07%).
- **Non-root Paths Present:** 27,456 URLs (11.64%).
  - *Critical Observation:* 88.36% of URLs in PhiUSIIL are bare domain homepages without deep paths. In contrast, 63.04% of LegitPhish URLs contain deep paths.
- **Query Strings Present:** 6,079 URLs (2.58%).
- **Fragments Present:** 832 URLs (0.35%).

### 3.6 Domain-Level Analysis
- **Unique Registrable Domains:** 172,296 domains.
- **Repeated Domains:** 6,210 domains appear multiple times.
  - Top repeated domains are predominantly free web app and cloud hosting platforms:
    - `web.app`: 5,754 URLs
    - `firebaseapp.com`: 5,594 URLs
    - `repl.co`: 3,754 URLs
    - `weeblysite.com`: 3,097 URLs
    - `ipfs.io`: 1,559 URLs
- **Conflicting-Label Domains:** 223 domains appear with both phishing and legitimate labels (e.g., `web.app`, `firebaseapp.com`, `repl.co`, `ac.id`).
  - *Significance:* Cloud app platforms and shared hosting host both benign applications and phishing pages. A model that relies heavily on domain identity without path analysis will suffer false positives on these domains.

---

## 4. Cross-Dataset Overlap and Data Leakage Findings

Cross-dataset audit between LegitPhish and PhiUSIIL uncovered major data overlap that directly threatens external generalization validity if left unmanaged:

### 4.1 Exact URL Overlap
- **Exact URL String Overlap:** Exactly **36,936 URLs** are present in both datasets.
- **Class Breakdown of Overlapping URLs:**
  - `label == 1` (Legitimate): **36,936 URLs (100.0%)**
  - `label == 0` (Phishing): **0 URLs (0.0%)**
- **Provenance Impact:**
  - LegitPhish contains 37,540 legitimate URLs.
  - **98.39%** of all legitimate URLs in LegitPhish are identical to legitimate URLs in PhiUSIIL!
  - Both research groups acquired their benign URL corpora from identical top-domain listings (Alexa/Tranco/Majestic).

### 4.2 Registrable Domain Overlap
- **Overlapping Registrable Domains:** **35,925 domains**.
- **Overlap Percentage:**
  - **57.80%** of all unique domains in LegitPhish exist in PhiUSIIL.
  - **20.85%** of all unique domains in PhiUSIIL exist in LegitPhish.
- **Cross-Dataset Conflicting Domains:** 111 domains have divergent labels across the two datasets (e.g., legitimate in LegitPhish, but phishing in PhiUSIIL).

### 4.3 Research Implication: The Leakage Trap
If a machine learning model is trained on LegitPhish and then naively tested on PhiUSIIL without cross-dataset filtering:
1. 36,936 legitimate test instances would be direct duplicates of training samples, yielding artificially perfect legitimate recall.
2. The model would appear to "generalize" externally, when in reality it is memorizing the benign web distribution.
3. **Mandatory Protocol Adjustment:** When evaluating on PhiUSIIL in Phase 4/5, all training URLs and training registrable domains must be subtracted, forming a strict **zero-leakage out-of-distribution external test split**.

---

## 5. Summary of Preprocessing Directives for Future Phases

1. **Drop Null Label:** Remove row 65118 (`https://www.chocolatleroux.com`) from LegitPhish.
2. **Deduplicate LegitPhish:** Remove 346 redundant duplicate rows.
3. **Deduplicate PhiUSIIL:** Retain one representative record per duplicate URL string.
4. **Isolate URL-Only Subset:** Discard all 54 HTML/content columns from PhiUSIIL.
5. **Handle Shared Hosting:** Treat platforms (`web.app`, `firebaseapp.com`, `godaddysites.com`) as multi-tenant hosts where path and token features must dominate over domain identity.
6. **Domain-Isolated Validation:** Enforce strict Group-K-Fold or domain-separated partitioning so that URLs from overlapping domains never appear in both training and test sets.

