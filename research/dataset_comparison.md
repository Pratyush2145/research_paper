# Comparative Dataset Analysis and Generalization Assessment

**Project:** Explainable and Generalization-Aware Phishing URL Detection Using Machine Learning and SHAP  
**Document:** Research Dataset Comparison  
**Phase:** 2 — Dataset Acquisition, Inspection, and Validation  

---

## 1. High-Level Dataset Comparison Matrix

| Property | Dataset A: LegitPhish (Primary) | Dataset B: PhiUSIIL (External Validation) | Comparative Impact & Risk Assessment |
| :--- | :--- | :--- | :--- |
| **Official Repository** | Mendeley Data | UCI Machine Learning Repository | High institutional credibility for both |
| **Release / Donation Date** | May 22, 2025 (Version 2) | March 4, 2024 | Contemporaneous datasets (2024–2025) |
| **Primary Paper / Citation** | Potpelwar, Kulkarni, Waghmare (2025) | Prasad & Chandra (2024), *Computers & Security* | Peer-reviewed benchmarks |
| **Official License** | CC BY 4.0 | CC BY 4.0 | Fully open for research and adaptation |
| **Verified Cryptographic Hash** | `8685e79...` (SHA-256 match) | `a236549...` (SHA-256 match) | Verified byte-level integrity |
| **Dataset Size on Disk** | 8.96 MB (CSV) | 54.22 MB (CSV, 14.66 MB zip) | Manageable for local in-memory training |
| **Total Record Count** | 101,219 rows | 235,795 rows | PhiUSIIL is 2.33x larger |
| **Total Features in Raw File** | 18 columns | 56 columns | PhiUSIIL contains 54 precomputed features |
| **Usable URL-Only Columns** | `URL`, `ClassLabel` | `URL`, `label` | Precomputed webpage features discarded |
| **Target Label Definition** | `0` = Phishing, `1` = Legitimate | `0` = Phishing, `1` = Legitimate | Harmonized label definitions |
| **Class Distribution (Phishing)** | 63,678 (**62.91%**) | 100,945 (**42.81%**) | LegitPhish is phishing-majority |
| **Class Distribution (Legitimate)**| 37,540 (**37.09%**) | 134,850 (**57.19%**) | PhiUSIIL is legitimate-majority |
| **Missing Values** | 1 missing (`ClassLabel` at row 65118) | 0 missing across all 56 columns | Trivial data loss (<0.001%) after drop |
| **Exact Duplicate Rows** | 346 rows | 0 rows | Redundancies present in LegitPhish |
| **Duplicate URL Strings** | 346 duplicate URLs | 425 duplicate URLs (850 total rows) | Duplicate re-crawls in PhiUSIIL |
| **Unique Registrable Domains** | 62,156 domains | 172,296 domains | Broad domain diversity |
| **Repeated Domains (Count > 1)** | 13,015 domains | 6,210 domains | Multi-page sampling per domain |
| **Conflicting Domains in Dataset** | 30 domains | 223 domains | Multi-tenant cloud & hosting platforms |
| **Percentage of URLs with Paths** | **63.04%** | **11.64%** | **Severe structural asymmetry** |
| **HTTPS Adoption Rate** | 40.56% HTTPS / 58.15% HTTP | 78.05% HTTPS / 21.95% HTTP | PhiUSIIL reflects higher HTTPS adoption |

---

## 2. Collection Methodology and Provenance Analysis

### 2.1 LegitPhish (Potpelwar et al., 2025)
- **Phishing Sources:** Assembled from active feed captures (predominantly OpenPhish and PhishTank) during early 2025. Phishing URLs in LegitPhish feature diverse path structures, IP hosts, URL parameters, and varied TLDs.
- **Legitimate Sources:** Curated by scraping top-ranked domains from web authority lists (Alexa / Tranco) and popular community web listings.
- **Structural Profile:** 63.04% of LegitPhish URLs contain path segments beyond the root hostname, and 97.08% contain subdomains or multi-part hosts.

### 2.2 PhiUSIIL (Prasad & Chandra, 2024)
- **Phishing Sources:** Collected from active feeds with live webpage confirmation. The authors ran automated crawlers to capture HTML source code, DOM structure, and network redirections.
- **Legitimate Sources:** Sourced from Alexa/Tranco top sites. Critically, the crawlers recorded predominantly the root entrypoint (`https://www.domain.com/`) for legitimate sites.
- **Structural Profile:** Only 11.64% of PhiUSIIL URLs contain path segments. 88.36% are root domain landings.

---

## 3. The Cross-Dataset Provenance Overlap & Data Leakage Discovery

Our empirical cross-audit revealed an extraordinary structural overlap between LegitPhish and PhiUSIIL:

```text
Cross-Dataset Overlap Metrics:
├── Exact URL Overlap:         36,936 URLs
│   ├── Phishing Overlap:           0 URLs (0.00%)
│   └── Legitimate Overlap:    36,936 URLs (100.00%)
│
└── Registrable Domain Overlap: 35,925 domains
    ├── Overlap as % of LegitPhish Domains: 57.80%
    └── Overlap as % of PhiUSIIL Domains:   20.85%
```

### Critical Scientific Insights:
1. **Identical Benign Source Distribution:** LegitPhish contains 37,540 legitimate URLs. Exactly 36,936 of them (**98.39%**) are identical to URLs in PhiUSIIL. Both research groups drew their benign samples from the identical Tranco/Alexa top-domain crawl snapshot.
2. **Disjoint Phishing Distribution:** Exactly 0 phishing URLs overlap between the two datasets. This confirms that phishing feeds are temporal, volatile, and highly non-overlapping across different collection windows.
3. **The Data Leakage Vulnerability:** In conventional literature, researchers train on Dataset A and evaluate on Dataset B, claiming "strong external generalization." However, if a model trained on LegitPhish is evaluated naively on PhiUSIIL:
   - 98.4% of the benign training set is re-tested as "unseen" external validation data.
   - The model achieves artificially inflated specificity, precision, and AUC by memorizing benign domain patterns.
   - **Methodological Fix:** In Phase 4/5, all training URLs and training registrable domains MUST be purged from the external test set to create a genuine **zero-leakage out-of-distribution benchmark**.

---

## 4. URL-Only Suitability and Strict Feature Filtering

### 4.1 Suitability of LegitPhish for URL-Only Research
- **Verdict: High Suitability.**
- The dataset provides full URL strings with natural diversity across protocols, subdomains, path depths, and query tokens.
- While LegitPhish provides 17 pre-extracted features, our research framework will validate their correctness and engineer comprehensive lexical/structural predictors independently in Phase 3.

### 4.2 Suitability of PhiUSIIL for URL-Only Research
- **Verdict: High Suitability (under strict column isolation).**
- PhiUSIIL contains 54 precomputed features that rely on:
  - HTML source code lines (`LineOfCode`, `LargestLineLength`).
  - Webpage DOM tags (`NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfiFrame`, `NoOfPopup`).
  - Page content semantics (`HasTitle`, `Title`, `DomainTitleMatchScore`, `URLTitleMatchScore`).
  - Network and redirection behavior (`NoOfURLRedirect`, `NoOfSelfRedirect`).
- **Research Exclusion:** Real-time client-side detection cannot afford the network latency, server footprint, or active risk of downloading malicious HTML payloads. Therefore, all 54 webpage features are discarded, and only `URL` and `label` are ingested.

---

## 5. Potential Biases and Research Limitations

### 5.1 Root vs. Path Asymmetry (Shortcut Learning)
- In PhiUSIIL, 88.4% of URLs have no path, while phishing URLs frequently contain subdirectories or script names. A classifier exposed to this distribution can learn a trivial shortcut (*"if URL has path -> phishing"*).
- Our feature engineering and model evaluation must explicitly evaluate models on path-matched subsets to prevent shortcut learning.

### 5.2 Multi-Tenant Cloud and Web App Hosting Platforms
- Domains such as `web.app` (5,754 URLs), `firebaseapp.com` (5,594 URLs), `repl.co` (3,754 URLs), and `weeblysite.com` (3,097 URLs) appear with both legitimate and phishing labels in PhiUSIIL.
- A model relying purely on domain popularity will misclassify legitimate apps on Firebase or flag all Google cloud apps as malicious. Features must capture subdomain randomness, token entropy, and lexical keywords within the path.

### 5.3 Class Imbalance Disparity
- LegitPhish: 62.9% Phishing / 37.1% Legitimate.
- PhiUSIIL: 42.8% Phishing / 57.2% Legitimate.
- Evaluation metrics must report PR-AUC, False Positive Rate (FPR) at fixed True Positive Rate (TPR), Matthews Correlation Coefficient (MCC), and confusion matrices rather than relying on raw accuracy.

