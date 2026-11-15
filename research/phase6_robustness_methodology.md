# Phase 6 — Robustness & Adversarial Testing: Comprehensive Experimental Methodology

**Project:** Explainable and Generalization-Aware Phishing URL Detection Using Machine Learning and SHAP  
**Document:** Research Methodology — Phase 6: Adversarial Robustness, Evasion Testing & Distributional Sensitivity  
**Date:** November 2026  
**Status:** **METHODOLOGY FINALIZED, IMPLEMENTED & SCIENTIFICALLY EXECUTED**  
**Target Architectures:** Frozen Random Forest (Champion) & Frozen XGBoost (Benchmark)  

---

## 1. Primary Research Objective & Experimental Scope

The primary objective of Phase 6 is to evaluate whether our trained, leakage-safe URL-only machine learning models—primarily the champion **Random Forest** ensemble and secondarily the benchmark **XGBoost** model—remain reliable when URLs undergo realistic adversarial manipulations, syntax transformations, and distributional shifts.

Our evaluation specifically tests hypotheses established during the Phase 5 SHAP explainability audit:
1. **Path Reliance Hypothesis:** Because both models concentrate $\approx 50\%$ of their decision attribution on path structure (`path_length`, `max_path_segment_length`, `path_segment_count`), adversaries manipulating path depth, injecting redundant segments, or stripping paths are expected to induce significant evasion.
2. **Transport Security Hypothesis:** Because cleartext HTTP provides a strong positive phishing attribution while HTTPS provides a substantial negative (legitimate) attribution ($\approx 20-26\%$), scheme and transport manipulation are primary attack vectors.
3. **Keyword Inefficacy Hypothesis:** Because explicit keyword flags contribute $<0.1\%$ to model decisions, keyword stuffing or stripping is hypothesized to have near-zero evasion impact, demonstrating model resilience against superficial lexical token attacks.

---

## 2. Transformation Validity & Label Invariance Framework

A major pitfall in adversarial machine learning for cybersecurity is the naive assumption that perturbing an input preserves its ground-truth label. If an adversarial transformation changes the destination host or breaks application routing, the resulting URL is not a valid phishing lure.

We establish a rigorous, three-tier taxonomy governing every transformation:

```text
Adversarial Transformation Validity Framework:
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Tier A: Strictly Semantics-Preserving (Demonstrable Destination & Label Equivalence)           │
│ • Defined by RFC standards (RFC 3986, RFC 4343, RFC 3492).                                      │
│ • Web clients and DNS resolvers resolve to the EXACT same origin, host, and resource.          │
│ • Ground truth label invariance is mathematically and operationally guaranteed.               │
│ • Valid for computing: Attack Success Rate (ASR), Evasion Rate (ER), and Recall Degradation.    │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Tier B: Conditionally Semantics-Preserving (Requires Explicit Pre-Condition Enforcement)        │
│ • Semantics and routing equivalence hold ONLY under specific server/attacker preconditions.     │
│ • Must be constrained strictly to valid subsets (e.g. attacker-controlled phishing domains).    │
│ • Inapplicable instances must be logged rather than forced into invalid mutations.             │
│ • Valid for computing: Conditional ASR on verified applicable sub-populations.                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Tier C: Exploratory & Out-of-Distribution Perturbations (Label Invariance Not Established)       │
│ • Alters destination host, breaks DNS resolution, or strips resource context.                  │
│ • Ground truth labels CANNOT be assumed invariant.                                              │
│ • Disallowed for ASR/Recall claims; evaluated strictly for Prediction Flip & OOD Sensitivity.   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Table 1: Comprehensive Classification of Attack Transformations

| Category ID | Transformation Name | Validity Tier | Transformation Rule & RFC Basis | Ground Truth Invariance Justification | Permissible Quantitative Metrics |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **CAT-01A** | Path Unreserved Percent-Encoding | **Tier A** | Encode `[a-zA-Z0-9_.~-]` in path to `%XX` (RFC 3986 §2.3). | RFC 3986 §2.3 mandates receiving applications decode unreserved octets identically. Destination and path resource remain identical. | ASR, Evasion Rate, $\Delta \text{Recall}$, $\Delta F_1$ |
| **CAT-01B** | Path Reserved Char Encoding | **Tier B** | Percent-encode sub-delimiters (`/` $\to$ `%2F`, `-` $\to$ `%2D`). | Valid only if web server / web framework decodes before routing. Slashes encoded as `%2F` may return 404 on strict servers (Apache `AllowEncodedSlashes`). | Conditional ASR on applicable subset |
| **CAT-01C** | Double Percent-Encoding | **Tier C** | Encode `%` to `%25` (`%256C`). | Browser sends `%25`; destination server must explicitly double-decode, or routing breaks. | Prediction Flip Rate, Sensitivity Drift |
| **CAT-02A** | Scheme Case Variation | **Tier A** | Convert scheme to uppercase / mixed (`hTtPs://`). | RFC 3986 §3.1: Schemes are strictly case-insensitive. Client normalizes to lowercase. | ASR, Evasion Rate, $\Delta \text{Recall}$, $\Delta F_1$ |
| **CAT-02B** | Hostname Case Variation | **Tier A** | Randomize host casing (`wWw.ExAmPlE.cOm`). | RFC 4343 §2: Domain Name System is strictly case-insensitive. Exact same IP resolved. | ASR, Evasion Rate, $\Delta \text{Recall}$, $\Delta F_1$ |
| **CAT-02C** | Redundant Consecutive Slashes | **Tier B** | Collapse/expand slashes in path (`///path`). | RFC 3986 §3.3 normalization collapses empty segments, but some WSGI/ASGI muxes reject `//`. | Conditional ASR, Flip Rate |
| **CAT-03A** | Cyrillic Lookalike Homoglyph | **Tier C** | Replace Latin chars with Cyrillic lookalikes (`а` for `a`). | Changes registrable domain to a different IDN/Punycode. Belongs to a different registrant or unregistered. | Prediction Flip Rate, OOD Sensitivity |
| **CAT-03B** | Punycode Round-Trip Equivalence | **Tier A** | Transcode valid IDN to Punycode (`xn--...`). | RFC 3492 / RFC 5891: Punycode and Unicode representations are semantically identical in DNS. | ASR, Evasion Rate, Feature Stability |
| **CAT-04A** | Subdomain Prepending on Phishing | **Tier B** | Prepend brand token as subdomain on attacker host. | Attacker controls DNS for their own domain; can provision arbitrary subdomains (`paypal.attacker.com`). Resolves to attacker lure. | Conditional ASR (Phishing only) |
| **CAT-04B** | Subdomain Prepending on Benign | **Tier C** | Prepend brand token to legitimate domain (`paypal.google.com`). | Unmapped DNS record; fails resolution in reality. Cannot assume legitimate traffic. | Prediction Flip Rate (Benign robustness) |
| **CAT-05A** | Dot-Slash Path Redundancy | **Tier B** | Insert `/./` into path (`/login/./verify`). | RFC 3986 §5.2.4 dot-segment algorithm normalizes `/./` to `/`. Equivalent on compliant servers. | Conditional ASR, Evasion Rate |
| **CAT-05B** | Path Directory Padding | **Tier B** | Append benign dummy subdirectories (`/.../index.html`). | Preserves phishing lure if server ignores extra path or uses URL rewrite rules. | Conditional ASR, Evasion Rate |
| **CAT-05C** | Dummy Fragment Appendage | **Tier A** | Append anchor fragment (`#section`). | RFC 3986 §3.5: Fragment identifiers are strictly client-side; never sent to server. Destination resource is identical. | ASR, Evasion Rate, $\Delta \text{Recall}$, $\Delta F_1$ |
| **CAT-06A** | Benign Tracking Parameter Injection | **Tier B** | Append marketing query keys (`?utm_source=email`). | Most web servers ignore unmapped query parameters and deliver the target phishing page. | Conditional ASR, Evasion Rate |
| **CAT-06B** | High-Entropy Noise Query Padding | **Tier B** | Append random 100-character alphanumeric query string. | Preserves destination if application ignores unrecognized parameters. | Conditional ASR, Evasion Rate |
| **CAT-07A** | Benign Security Keyword Stuffing | **Tier B** | Inject words like `secure`, `official`, `portal` into path. | Path alteration. If server routes via rewrite to target page, lure remains active. | Conditional ASR, Evasion Rate |
| **CAT-07B** | Suspicious Keyword Stripping | **Tier B** | Replace `login`, `verify` with random tokens. | Phishing kits often operate with randomized PHP script names (`a7x.php`). Phishing intent preserved. | Conditional ASR, Evasion Rate |
| **CAT-08A** | Standard IPv4 Dotted-Decimal | **Tier A** | Normal IPv4 host (`192.168.1.1`). | Baseline standard IP routing. | Baseline Reference |
| **CAT-08B** | Alternative IP Formats (Hex/Dword) | **Tier C** | Convert IP to DWORD or Hexadecimal (`0xC0A80101`). | Supported by Windows OS socket APIs, but rejected by modern Chrome/Linux network stacks. Destination equivalence is client-dependent. | Prediction Flip Rate, OOD Sensitivity |
| **CAT-09A** | Standard Port Injection | **Tier A** | Explicitly append `:80` (HTTP) or `:443` (HTTPS). | RFC 3986 §3.2.3: Port matching default scheme port is semantically and operationally equivalent. | ASR, Evasion Rate, $\Delta \text{Recall}$, $\Delta F_1$ |
| **CAT-09B** | Apex Path Truncation (Root Stripping) | **Tier C** | Strip path components, leaving bare domain. | Often leaves parked page or benign compromised root domain; does NOT preserve credential harvesting lure. | Prediction Flip Rate, Feature Sensitivity |

---

## 3. Strict Separation of Evaluation Partitions

To prevent cross-dataset contamination and maintain domain-level integrity:
1. **Never Pool Partitions:** Internal held-out test data and external disjoint data must **never** be averaged or combined into a single aggregated headline result. Each dataset exhibits distinct baseline properties and collection characteristics:
   - **LegitPhish Internal Test Set:** $N = 14,778$ rows ($9,195$ Phishing, $5,583$ Legitimate; $9,324$ domains). Baseline RF Recall: $99.99\%$; Baseline XGB Recall: $99.97\%$.
   - **PhiUSIIL Strict External Disjoint Set:** $N = 165,152$ rows ($71,069$ Phishing, $94,083$ Legitimate; $136,372$ domains). Baseline RF Recall: $95.42\%$; Baseline XGB Recall: $49.31\%$.
2. **Paired Sample Identifiers:** Every evaluated URL maintains an immutable tuple:
   $$\langle \text{Sample ID}, \text{Source Partition}, \text{Registrable Domain}, y_i, x_{i,\text{orig}}, \hat{y}_{i,\text{orig}}, \hat{p}_{i,\text{orig}}, x'_{i,\text{adv}}, \hat{y}_{i,\text{adv}}, \hat{p}_{i,\text{adv}} \rangle$$
   This enables exact paired contingency tracking and eliminates sample misalignment.

---

## 4. Formal Definition of Attack & Robustness Metrics

Every metric must have an explicitly documented mathematical definition and unambiguous denominator:

### 4.1 Conditional Attack Success Rate ($\text{ASR}_{\text{cond}}$)
Measures the fraction of **baseline-detected phishing URLs** that successfully evade detection after adversarial transformation:
$$\text{ASR}_{\text{cond}} = \frac{\sum_{i \in \text{TP}_{\text{base}}} \mathbb{I}(\hat{y}_{\text{adv}}(x'_i) = 1)}{|\text{TP}_{\text{base}}|}$$
- **Numerator:** Baseline true positives that flip to "Legitimate" ($1$) under attack.
- **Denominator:** $|\text{TP}_{\text{base}}| = \sum_{i: y_i=0} \mathbb{I}(\hat{y}_{\text{orig}}(x_i) = 0)$.
- **Scope:** Computed exclusively for Tier A and verified Tier B transformations where ground-truth phishing status is guaranteed.

### 4.2 Overall Phishing Evasion Rate ($\text{ER}_{\text{overall}}$)
Measures the proportion of **all phishing URLs** in the evaluation set that escape detection under attack (including baseline misses):
$$\text{ER}_{\text{overall}} = \frac{\sum_{i: y_i=0} \mathbb{I}(\hat{y}_{\text{adv}}(x'_i) = 1)}{N_{\text{phish}}}$$
- **Numerator:** Total phishing URLs classified as "Legitimate" ($1$) under attack.
- **Denominator:** $N_{\text{phish}} = \sum_{i} \mathbb{I}(y_i=0)$.
- **Relation to Recall:** $\text{ER}_{\text{overall}} = 1 - \text{Recall}_{\text{adv}}$.

### 4.3 False Positive Surge Rate ($\text{FPSR}$)
Measures the fraction of **baseline-correct legitimate URLs** that become false alarms under adversarial formatting:
$$\text{FPSR} = \frac{\sum_{i \in \text{TN}_{\text{base}}} \mathbb{I}(\hat{y}_{\text{adv}}(x'_i) = 0)}{|\text{TN}_{\text{base}}|}$$
- **Numerator:** Baseline true negatives that flip to "Phishing" ($0$) under attack.
- **Denominator:** $|\text{TN}_{\text{base}}| = \sum_{i: y_i=1} \mathbb{I}(\hat{y}_{\text{orig}}(x_i) = 1)$.
- **Operational Meaning:** Quantifies operational disruption and alert fatigue inflicted on SOC analysts by non-standard benign URL formatting.

### 4.4 Prediction Flip Rate ($\text{PFR}$)
Measures the total instability of the model's decision boundary across all inputs, regardless of true label:
$$\text{PFR} = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_{\text{orig}}(x_i) \ne \hat{y}_{\text{adv}}(x'_i))}{N}$$
- **Denominator:** Total evaluated URLs ($N$).
- **Scope:** Applicable across all Tiers (A, B, and C) to quantify model fragility.

### 4.5 Phishing Probability Shift ($\Delta P(\text{Phish})$)
Quantifies the continuous directional score change induced by the transformation:
$$\Delta p_i = \hat{p}_{\text{adv}}(x'_i) - \hat{p}_{\text{orig}}(x_i)$$
We report:
- **Mean $\Delta p$:** $\frac{1}{N} \sum_{i=1}^N \Delta p_i$ (Parametric mean shift).
- **Median $\Delta p$:** $\text{Median}(\{\Delta p_i\})$ (Non-parametric robust shift).
- **Interquartile Range (IQR):** $\text{IQR}(\{\Delta p_i\}) = Q_3 - Q_1$.

### 4.6 Performance Degradation Metrics
$$\Delta \text{Recall} = \text{Recall}_{\text{orig}} - \text{Recall}_{\text{adv}}$$
$$\Delta F_1 = F_{1,\text{orig}} - F_{1,\text{adv}}$$

---

## 5. Statistical Rigor & Hypothesis Testing

To avoid reporting cosmetic or unverified differences, all robustness evaluations must implement formal statistical tests:

### 5.1 Paired Bootstrap Confidence Intervals (1,000 Resamples)
For every metric $M$ ($\text{ASR}$, $\text{ER}$, $\text{FPSR}$, $\Delta \text{Recall}$, $\Delta F_1$):
1. Sample $N$ paired tuples $(x_i, x'_i, y_i)$ with replacement from the evaluated partition.
2. Calculate metric $M^{(b)}$ for $b = 1, \dots, 1000$.
3. Compute the 95% percentile confidence interval:
   $$\text{CI}_{95\%} = [M_{(0.025)}^{*}, M_{(0.975)}^{*}]$$

### 5.2 McNemar's Test for Paired Binary Transitions
To determine whether an adversarial transformation causes a statistically significant change in classification error:
- Construct the $2 \times 2$ paired contingency table:
  - $a$: Both Original and Adversarial are Correct.
  - $b$: Original is Correct ($\hat{y}_{\text{orig}} = y$) but Adversarial is Incorrect ($\hat{y}_{\text{adv}} \ne y$) [Vulnerability Flip].
  - $c$: Original is Incorrect ($\hat{y}_{\text{orig}} \ne y$) but Adversarial is Correct ($\hat{y}_{\text{adv}} = y$) [Serendipitous Flip].
  - $d$: Both Original and Adversarial are Incorrect.
- Compute the Edwards continuity-corrected McNemar test statistic:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{df} = 1$$
- Compute the exact two-tailed p-value. If $b + c < 25$, use the exact binomial test ($p = 2 \sum_{k=b}^{b+c} \binom{b+c}{k} 0.5^{b+c}$).

### 5.3 Multiple Comparison Correction
When evaluating across $K$ attack categories simultaneously:
- **Family-Wise Error Rate (FWER):** Apply the **Bonferroni correction**:
  $$\alpha_{\text{adjusted}} = \frac{\alpha}{K} = \frac{0.05}{K}$$
- **False Discovery Rate (FDR):** Apply the **Benjamini-Hochberg (BH) procedure**:
  1. Rank raw p-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(K)}$.
  2. Find the largest rank $i$ such that $p_{(i)} \le \frac{i}{K} Q$ (with $Q = 0.05$).
  3. Reject all null hypotheses for $k \le i$.

---

## 6. Additional Diagnostic Robustness Tests

In addition to standard evasion testing, Phase 6 incorporates five specialized diagnostic analyses:

### 6.1 Raw vs. Canonicalized Feature Sensitivity
- **Audit Process:** Extract 68 features from raw unparsed URL strings versus fully RFC-canonicalized URLs (lowercase host, decoded unreserved percent-octets, collapsed slashes, normalized ports).
- **Metric:** Maximum Absolute Feature Drift $\max_j |x_{\text{raw}, j} - x_{\text{canon}, j}|$.
- **Objective:** Reveal whether the feature extraction pipeline inadvertently contains lexical bugs where semantically identical URLs yield divergent numerical feature vectors.

### 6.2 Unicode Normalization & Punycode Equivalence
- **Audit Process:** Process internationalized domain names across four Unicode normalization forms (NFC, NFD, NFKC, NFKD) and explicit ASCII Punycode (`xn--...`).
- **Objective:** Verify whether Unicode normalization changes character counts, entropy, or Punycode flags, and whether model predictions remain invariant under equivalent Unicode forms.

### 6.3 Feature-Level Perturbation Sensitivity (Continuous Perturbation Scans)
- **Audit Process:** For the top continuous features identified in Phase 5 (`path_length`, `max_path_segment_length`, `digit_ratio`, `url_entropy`), perturb feature values along a linear grid ($\pm 10\%, \pm 25\%, \pm 50\%$) while holding all other features fixed.
- **Metric:** Empirical decision boundary distance $\Delta x_j^*$ required to induce a prediction flip ($\hat{p} = 0.50$).
- **Objective:** Measure the mathematical steepness and margin stability of the Random Forest and XGBoost decision surfaces.

### 6.4 Probability Calibration Drift (Brier Score & Expected Calibration Error)
- **Audit Process:** Evaluate the reliability of posterior probabilities $\hat{p}_i$ under adversarial distribution shift:
  - **Brier Score:** $\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$.
  - **Expected Calibration Error (ECE):** Partition probabilities into $M=10$ equal-frequency bins:
    $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
- **Objective:** Determine whether adversarial perturbations degrade probability confidence and whether models become overconfident in erroneous predictions.

### 6.5 Out-of-Distribution (OOD) Structural Failure Boundaries
- **Audit Process:** Feed non-standard URL structures into the models:
  - URLs with userinfo authentication tokens (`https://user:pass@host/`).
  - URLs with extreme query string lengths ($>1,000$ characters).
  - URLs with zero-path root architecture over HTTPS.
- **Objective:** Map the architectural blind spots where feature extractors saturate or where decision trees default to prior leaf weights.

---

## 7. Sampling Methodology & Applicability Tracking

### 7.1 Stratified, Domain-Aware Evaluation Samples
To ensure reproducibility and manage computational complexity without introducing selection bias:
1. **Sample Construction:**
   - **Internal Test Benchmark:** $N = 2,000$ paired URLs ($1,000$ Phishing, $1,000$ Legitimate) sampled with stratified class balance across unique registrable domains from the held-out LegitPhish test partition (Random Seed $42$).
   - **External Disjoint Benchmark:** $N = 2,000$ paired URLs ($1,000$ Phishing, $1,000$ Legitimate) sampled with stratified class balance across unique registrable domains from the strict external PhiUSIIL disjoint partition (Random Seed $42$).
2. **Domain Contamination Guarantee:**
   $$\text{Domain}(S_{\text{int}}) \cap \text{Domain}(\text{Train}) = \emptyset, \quad \text{Domain}(S_{\text{ext}}) \cap \text{Domain}(\text{Train}) = \emptyset$$

### 7.2 Explicit Transformation Applicability Accounting
Adversarial evaluations often commit survivorship bias by reporting metrics only on "easy-to-transform" URLs. To eliminate this bias, we enforce strict accounting for every candidate URL:
- **Total Candidate URLs ($N_{\text{cand}}$):** Total URLs submitted for transformation.
- **Applicable URLs ($N_{\text{app}}$):** URLs containing the prerequisite structure (e.g., must contain a path to undergo path percent-encoding; must be an IP address to undergo hex-IP encoding).
- **Successfully Transformed URLs ($N_{\text{tx}}$):** URLs successfully transformed and verified against Tier validity rules.
- **Applicability Rate ($A_r$):** $\frac{N_{\text{app}}}{N_{\text{cand}}}$.
- **Transformation Success Rate ($T_r$):** $\frac{N_{\text{tx}}}{N_{\text{app}}}$.
All metrics report the effective denominator ($N_{\text{tx}}$) alongside the applicability rate ($A_r$).

---

## 8. Artifact Protection & Scientific Integrity Protocol

1. **Artifact Freeze:** All Phase 1 through Phase 5 files, models, datasets, and reports are frozen and strictly read-only.
2. **No Adversarial Training:** Phase 6 is exclusively a diagnostic vulnerability audit. No adversarial retraining or data augmentation is permitted.
3. **Dedicated Output Architecture:**
   - Source code: `src/adversarial/`
   - Generated adversarial datasets: `data/adversarial/`
   - Evaluation tables: `reports/tables/adversarial/`
   - Visualizations: `reports/figures/adversarial/`
   - Research documentation: `research/phase6_...`
4. **Tool & Environment Restrictions:** No Git or GitHub commands. All code executed inside the sandbox with deterministic random seeds (`Seed 42`).

