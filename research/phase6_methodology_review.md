# Phase 6 — Adversarial Robustness & Evasion Testing: Methodology Review & Scientific Validity Audit

**Project:** Explainable and Generalization-Aware Phishing URL Detection Using Machine Learning and SHAP  
**Document:** Methodology Review & Scientific Pre-Implementation Audit  
**Phase:** Phase 6 — Robustness & Adversarial Testing  
**Date:** November 2026  
**Status:** **AUDITED, VERIFIED & FORMALIZED (PRE-IMPLEMENTATION GATEWAY)**  
**Target Architectures:** Frozen Random Forest (Champion) & Frozen XGBoost (Benchmark)  

---

## 1. Executive Summary & Purpose of the Review

This document establishes the formal **Methodology Review and Scientific Validity Audit** for Phase 6 of our research project. Following the completion of Phase 4 (Model Development & Generalization Audit) and Phase 5 (Explainable AI using SHAP), Phase 6 investigates whether our trained URL-only detection models maintain structural integrity and classification reliability when subjected to adversarial manipulation, lexical evasion techniques, and distributional shifts.

Adversarial machine learning evaluations in cybersecurity frequently suffer from critical methodological vulnerabilities, including:
1. **Unsubstantiated label invariance** (assuming any string mutation preserves the original phishing or legitimate intent);
2. **Conflated metric denominators** (reporting attack success without distinguishing detected baselines from total populations);
3. **Partition contamination** (pooling internal test data with zero-shot external disjoint data);
4. **Survivorship/applicability bias** (failing to report how many URLs were eligible for a specific syntactic mutation); and
5. **Statistical informality** (reporting unverified percentage drops without paired confidence intervals or hypothesis testing).

This review comprehensively audits the refined Phase 6 experimental design documented in [`research/phase6_robustness_methodology.md`](file:///Users/pratyushsharma/Desktop/phishing-detection-research/research/phase6_robustness_methodology.md). It confirms that all methodological deficiencies identified in preliminary proposals have been systematically eliminated prior to writing attack implementation code or executing experiments.

---

## 2. Audit of Preliminary Methodology & Formal Corrections

A rigorous review of the initial Phase 6 proposal identified five significant methodological risks. Table 1 summarizes these identified issues and their formal scientific resolutions.

### Table 1: Summary of Methodological Deficiencies and Enforced Corrections

| # | Preliminary Proposal Deficiency | Scientific Risk | Enforced Methodological Correction | Verification Standard |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Naive Label Invariance Assumption:** Assumed all URL string perturbations preserve ground-truth class. | Methodological invalidity; mutations like Cyrillic homoglyphs or apex path stripping change destination servers or break routing, invalidating true phishing intent. | Established a **Three-Tier Validity Framework** (Tier A: Strictly Semantics-Preserving; Tier B: Conditionally Preserving; Tier C: Exploratory / OOD). | Grounded in RFC 3986, RFC 4343, and RFC 3492/5891. Tier C is barred from ASR/Recall claims. |
| **2** | **Ambiguous Metric Denominators:** Conflated "evasion rate", "attack success rate", and "flip rate" without explicit denominators. | Incomparable metrics; artificially inflated evasion figures by counting baseline false negatives as successful attacks. | Formulated four distinct, mathematically unambiguous metrics with explicit numerators and denominators: $\text{ASR}_{\text{cond}}$, $\text{ER}_{\text{overall}}$, $\text{FPSR}$, and $\text{PFR}$. | Formally defined in Section 4 with closed-form mathematical equations and contingency tracking. |
| **3** | **Risk of Cross-Partition Metric Pooling:** Potential aggregation of internal test and external disjoint evaluation metrics. | Masks the generalization gap; confounds memorized domain distributions with true zero-shot external robustness. | Enforced **absolute partition separation**. Internal held-out test and external disjoint benchmarks are evaluated and reported in separate tables. | Never average metrics into single headline results. Maintain immutable paired tracking tuples. |
| **4** | **Unaccounted Transformation Applicability:** Ignoring candidate URLs that lack the syntactic feature being perturbed. | Survivorship bias; reporting high evasion rates over an unrepresentative, cherry-picked subset of URLs. | Implemented formal **Transformation Applicability Accounting** tracking $N_{\text{cand}}$, $N_{\text{app}}$, $N_{\text{tx}}$, $A_r$, and $T_r$. | Every table must report effective sample sizes ($N_{\text{tx}}$) alongside applicability rates ($A_r$). |
| **5** | **Lack of Paired Statistical Testing:** Reporting point estimates of $\Delta \text{Accuracy}$ or $\Delta \text{Recall}$ without hypothesis testing or error control. | Type I errors; reporting random sampling noise or insignificant shifts as true architectural vulnerabilities. | Mandated **Paired Bootstrap CIs (1,000 resamples)**, **continuity-corrected McNemar tests**, and **Benjamini-Hochberg (FDR) / Bonferroni corrections**. | Calculate exact two-tailed $p$-values and 95% confidence intervals for every evaluated attack vector. |

---

## 3. The Three-Tier Transformation Validity Framework

To ensure that reported evasion rates reflect genuine detection failure rather than malformed, unresolvable strings, every adversarial transformation is classified into one of three tiers based on RFC standards and operational web semantics:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER A: STRICTLY SEMANTICS-PRESERVING (Demonstrable Destination & Label Equivalence)           │
│ RFC Standards: RFC 3986 (URI Syntax), RFC 4343 (DNS Case Insensitivity), RFC 3492 (Punycode).  │
│ Routing Guarantees: Browsers and DNS resolvers resolve to the exact same host and resource.     │
│ Ground Truth Status: INVARIANT. A phishing lure remains an active phishing lure.                │
│ Permissible Metrics: Conditional ASR, Overall Evasion Rate, ΔRecall, ΔF1, ΔMCC.                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER B: CONDITIONALLY SEMANTICS-PRESERVING (Requires Explicit Pre-Condition Validation)         │
│ Context: Query injection, redundant slashes, dot-segment traversal, directory padding.          │
│ Routing Guarantees: Preserved IF web servers / frameworks normalize paths or ignore queries.   │
│ Ground Truth Status: CONDITIONALLY INVARIANT on verified applicable subsets.                    │
│ Permissible Metrics: Conditional ASR on applicable subsets; Applicability Rate (Ar) reported.   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER C: EXPLORATORY & OUT-OF-DISTRIBUTION PERTURBATIONS (Label Invariance Not Established)       │
│ Context: Cyrillic homoglyphs, non-standard IP formats (Hex/DWORD), apex path stripping.         │
│ Routing Guarantees: Changes registrable domain, breaks DNS resolution, or strips lure context.  │
│ Ground Truth Status: NOT ESTABLISHED. Destination equivalence cannot be assumed.                │
│ Permissible Metrics: Prediction Flip Rate (PFR), Posterior Score Drift, OOD Sensitivity.        │
│ PROHIBITED METRICS: Attack Success Rate (ASR) and Recall Degradation claims.                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Table 2: Complete Transformation Matrix and Operational Justifications

| ID | Attack Transformation | Validity Tier | Mechanism & RFC Basis | Ground Truth Invariance Justification | Permissible Evaluative Metrics |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **CAT-01A** | Path Unreserved Percent-Encoding | **Tier A** | Encodes ASCII unreserved chars `[a-zA-Z0-9_.~-]` to `%XX` (RFC 3986 §2.3). | RFC 3986 §2.3 mandates receiving servers decode unreserved characters identically. Target page and harvested credentials remain identical. | $\text{ASR}_{\text{cond}}$, $\text{ER}_{\text{overall}}$, $\Delta \text{Recall}$, $\Delta F_1$ |
| **CAT-01B** | Path Reserved Char Encoding | **Tier B** | Percent-encodes delimiters (`/` $\to$ `%2F`, `-` $\to$ `%2D`). | Server-dependent; strict web servers (e.g. Apache with default `AllowEncodedSlashes Off`) return 404. Preserved only on frameworks with pre-route decoding. | Conditional ASR on applicable subset |
| **CAT-01C** | Double Percent-Encoding | **Tier C** | Encodes `%` to `%25` (`%256C`). | Browser transmits `%25`; destination web application must explicitly execute a double-decoding pass, or routing fails. | $\text{PFR}$, Posterior Score Drift, OOD Sensitivity |
| **CAT-02A** | Scheme Case Variation | **Tier A** | Randomizes scheme casing (`hTtPs://`). | RFC 3986 §3.1: URI schemes are explicitly case-insensitive. User agents canonicalize to lowercase before socket creation. | $\text{ASR}_{\text{cond}}$, $\text{ER}_{\text{overall}}$, $\Delta \text{Recall}$, $\Delta F_1$ |
| **CAT-02B** | Hostname Case Variation | **Tier A** | Randomizes hostname casing (`wWw.PayPaL.cOm`). | RFC 4343 §2: Domain Name System is strictly case-insensitive. DNS resolution yields the identical IP address. | $\text{ASR}_{\text{cond}}$, $\text{ER}_{\text{overall}}$, $\Delta \text{Recall}$, $\Delta F_1$ |
| **CAT-02C** | Redundant Consecutive Slashes | **Tier B** | Injects redundant slashes into path (`///login`). | RFC 3986 §3.3 normalization collapses empty segments, but some WSGI/ASGI application routers reject `//` with 400/404 errors. | Conditional ASR, $\text{PFR}$ |
| **CAT-03A** | Cyrillic Lookalike Homoglyph | **Tier C** | Replaces Latin characters with visually identical Cyrillic characters (`а` for `a`). | Changes the registrable domain to a completely different IDN/Punycode string (`xn--...`). Resolves to an unowned domain or third-party server. | $\text{PFR}$, Continuous Score Drift, OOD Sensitivity |
| **CAT-03B** | Punycode Round-Trip Equivalence | **Tier A** | Transcodes valid internationalized domain to ASCII Punycode (`xn--...`). | RFC 3492 & RFC 5891: Punycode and Unicode representations are semantically and operationally equivalent in DNS resolution. | $\text{ASR}_{\text{cond}}$, $\text{ER}_{\text{overall}}$, Feature Vector Stability |
| **CAT-04A** | Subdomain Prepending on Phishing | **Tier B** | Prepends trusted brand tokens as subdomains on attacker domain (`paypal.attacker.com`). | The attacker owns and controls DNS for `attacker.com`; can provision arbitrary A/CNAME records routing to the phishing kit. | Conditional ASR (Phishing URLs only) |
| **CAT-04B** | Subdomain Prepending on Benign | **Tier C** | Prepends brand tokens to legitimate domains (`paypal.google.com`). | Unmapped DNS record; fails resolution in the real world. Cannot be treated as valid operational benign traffic. | $\text{PFR}$ (Benign Model Brittleness), $\text{FPSR}$ |
| **CAT-05A** | Dot-Slash Path Redundancy | **Tier B** | Injects `/./` into path hierarchy (`/login/./verify`). | RFC 3986 §5.2.4 dot-segment algorithm normalizes `/./` to `/`. Equivalent on compliant HTTP servers. | Conditional ASR, $\text{ER}_{\text{overall}}$ |
| **CAT-05B** | Path Directory Padding | **Tier B** | Appends benign-sounding dummy directories (`/.../index.html`). | Preserves phishing lure if web application routes via catch-all script or URL rewrite rules. | Conditional ASR, $\text{ER}_{\text{overall}}$ |
| **CAT-05C** | Dummy Fragment Appendage | **Tier A** | Appends client-side anchor fragment (`#account-verification`). | RFC 3986 §3.5: Fragment identifiers are processed strictly client-side by user agents and never transmitted to the server. Destination resource is 100% identical. | $\text{ASR}_{\text{cond}}$, $\text{ER}_{\text{overall}}$, $\Delta \text{Recall}$, $\Delta F_1$ |
| **CAT-06A** | Benign Tracking Parameter Injection | **Tier B** | Appends standard marketing query keys (`?utm_source=email&utm_campaign=security`). | Standard web servers and web applications ignore unhandled GET query parameters and deliver the target credential harvest page. | Conditional ASR, $\text{ER}_{\text{overall}}$ |
| **CAT-06B** | High-Entropy Noise Query Padding | **Tier B** | Appends high-entropy alphanumeric query string (`?token=a8f9b...`). | Phishing kits frequently use dynamic session tokens. Destination page remains active. | Conditional ASR, $\text{ER}_{\text{overall}}$ |
| **CAT-07A** | Benign Security Keyword Stuffing | **Tier B** | Injects tokens like `secure`, `official`, `portal` into path or query. | Preserves lure if routing rewrite directs traffic to login script. | Conditional ASR, $\text{ER}_{\text{overall}}$ |
| **CAT-07B** | Suspicious Keyword Stripping | **Tier B** | Replaces tokens like `login`, `verify` with randomized alphanumeric strings. | Modern phishing kits routinely employ randomized script endpoints (`a7b.php`). Harvesting functionality is preserved. | Conditional ASR, $\text{ER}_{\text{overall}}$ |
| **CAT-08A** | Standard IPv4 Dotted-Decimal | **Tier A** | Formats standard IPv4 address (`192.168.1.1`). | Baseline standard IP routing. | Baseline Reference |
| **CAT-08B** | Alternative IP Formats (Hex/DWORD) | **Tier C** | Converts IP address to DWORD or Hexadecimal notation (`0xC0A80101`). | Historically supported by Windows WinSock APIs, but rejected by modern Chrome/Linux network stacks. Destination equivalence is client-dependent. | $\text{PFR}$, OOD Sensitivity |
| **CAT-09A** | Standard Port Injection | **Tier A** | Explicitly appends default scheme port (`:80` for HTTP, `:443` for HTTPS). | RFC 3986 §3.2.3: Port matching default scheme port is semantically and operationally equivalent. | $\text{ASR}_{\text{cond}}$, $\text{ER}_{\text{overall}}$, $\Delta \text{Recall}$, $\Delta F_1$ |
| **CAT-09B** | Apex Path Truncation (Root Stripping) | **Tier C** | Strips path hierarchy, leaving bare domain root. | Frequently points to a parked domain or compromised benign root; credential harvesting form is stripped. Does not preserve phishing lure. | $\text{PFR}$, Feature Sensitivity Drift |

---

## 4. Formal Definition of Robustness Metrics & Mathematical Rigor

To eliminate ambiguity, all evaluative metrics are defined with explicit mathematical formulas, numerators, and denominators:

### 4.1 Conditional Attack Success Rate ($\text{ASR}_{\text{cond}}$)
Measures the proportion of **originally detected phishing URLs** that successfully evade detection following adversarial perturbation:
$$\text{ASR}_{\text{cond}} = \frac{\sum_{i \in \text{TP}_{\text{base}}} \mathbb{I}(\hat{y}_{\text{adv}}(x'_i) = 1)}{|\text{TP}_{\text{base}}|}$$
- **Numerator:** Phishing URLs correctly identified at baseline ($\hat{y}_{\text{orig}} = 0$) that flip to legitimate ($\hat{y}_{\text{adv}} = 1$) under attack.
- **Denominator:** $|\text{TP}_{\text{base}}| = \sum_{i: y_i=0} \mathbb{I}(\hat{y}_{\text{orig}}(x_i) = 0)$.
- **Scope:** Computed exclusively for Tier A and verified Tier B transformations where ground-truth phishing status is guaranteed.

### 4.2 Overall Phishing Evasion Rate ($\text{ER}_{\text{overall}}$)
Measures the proportion of **all phishing URLs** in the evaluation partition that escape detection under attack (accounting for both baseline false negatives and induced evasions):
$$\text{ER}_{\text{overall}} = \frac{\sum_{i: y_i=0} \mathbb{I}(\hat{y}_{\text{adv}}(x'_i) = 1)}{N_{\text{phish}}}$$
- **Numerator:** Total phishing URLs classified as legitimate ($\hat{y}_{\text{adv}} = 1$) in the adversarial state.
- **Denominator:** $N_{\text{phish}} = \sum_{i} \mathbb{I}(y_i=0)$.
- **Mathematical Relationship:** $\text{ER}_{\text{overall}} = 1 - \text{Recall}_{\text{adv}}$.

### 4.3 False Positive Surge Rate ($\text{FPSR}$)
Measures the proportion of **originally correct legitimate URLs** that are erroneously flagged as phishing due to adversarial or non-standard formatting:
$$\text{FPSR} = \frac{\sum_{i \in \text{TN}_{\text{base}}} \mathbb{I}(\hat{y}_{\text{adv}}(x'_i) = 0)}{|\text{TN}_{\text{base}}|}$$
- **Numerator:** Legitimate URLs correctly identified at baseline ($\hat{y}_{\text{orig}} = 1$) that flip to phishing ($\hat{y}_{\text{adv}} = 0$) under perturbation.
- **Denominator:** $|\text{TN}_{\text{base}}| = \sum_{i: y_i=1} \mathbb{I}(\hat{y}_{\text{orig}}(x_i) = 1)$.
- **Operational Meaning:** Quantifies analyst alert fatigue and false alarm burden induced by benign formatting variations.

### 4.4 Prediction Flip Rate ($\text{PFR}$)
Measures the overall brittleness of the model's decision boundary across all evaluated instances, regardless of ground truth label:
$$\text{PFR} = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_{\text{orig}}(x_i) \ne \hat{y}_{\text{adv}}(x'_i))}{N}$$
- **Denominator:** Total evaluated instances ($N$).
- **Scope:** Permissible across all tiers (Tier A, Tier B, Tier C) as a measure of boundary sensitivity and numerical stability.

### 4.5 Continuous Probability Shift ($\Delta P(\text{Phish})$)
Measures the continuous displacement in model posterior probabilities:
$$\Delta p_i = \hat{p}_{\text{adv}}(x'_i) - \hat{p}_{\text{orig}}(x_i)$$
- **Reported Statistics:** Mean $\Delta p$, Median $\Delta p$, Interquartile Range (IQR), and empirical score distribution shifts.

---

## 5. Statistical Framework & Hypothesis Testing

To ensure all reported performance degradations represent statistically significant phenomena rather than random sampling fluctuations, we enforce the following statistical testing protocol:

### 5.1 Paired Bootstrap Confidence Intervals (1,000 Resamples)
For each evaluated attack category and metric $M \in \{\text{ASR}_{\text{cond}}, \text{ER}_{\text{overall}}, \text{FPSR}, \Delta \text{Recall}, \Delta F_1\}$:
1. Resample $N$ paired tuples $(x_i, x'_i, y_i)$ with replacement across 1,000 bootstrap iterations ($B = 1,000$).
2. Compute the metric realization $M^{(b)}$ for each iteration $b \in \{1, \dots, B\}$.
3. Establish the 95% percentile confidence interval:
   $$\text{CI}_{95\%} = [M_{(0.025)}^{*}, M_{(0.975)}^{*}]$$

### 5.2 McNemar's Test for Paired Transitions
To determine whether an adversarial transformation induces a statistically significant increase in classification error relative to baseline, we construct the $2 \times 2$ paired transition contingency table:

```text
                        Adversarial Prediction
                     Correct (adv)    Incorrect (adv)
Baseline  Correct      Cell a              Cell b (Vulnerability Flip)
Baseline  Incorrect    Cell c              Cell d
```

- **Vulnerability Flips ($b$):** Sample was correctly classified at baseline ($\hat{y}_{\text{orig}} = y$) but misclassified under attack ($\hat{y}_{\text{adv}} \ne y$).
- **Serendipitous Flips ($c$):** Sample was misclassified at baseline ($\hat{y}_{\text{orig}} \ne y$) but correctly classified under attack ($\hat{y}_{\text{adv}} = y$).
- **Edwards Continuity-Corrected Statistic:**
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{df} = 1$$
- **Exact Binomial Test:** Executed when discordant transitions are sparse ($b + c < 25$):
  $$p = 2 \cdot \sum_{k=b}^{b+c} \binom{b+c}{k} 0.5^{b+c}$$

### 5.3 Multiple Comparison Error Control
Because evaluations span multiple attack categories simultaneously, uncorrected hypothesis testing leads to severe family-wise error inflation:
- **Family-Wise Error Rate (FWER):** Control via Bonferroni correction:
  $$\alpha_{\text{Bonf}} = \frac{0.05}{K}$$
- **False Discovery Rate (FDR):** Control via Benjamini-Hochberg (BH) procedure at $Q = 0.05$:
  $$p_{(i)} \le \frac{i}{K} \cdot 0.05$$
  All null hypotheses up to rank $i^*$ are rejected, ensuring robust statistical control against false discoveries.

---

## 6. Audit of the Five Diagnostic Robustness Analyses

In addition to discrete attack evaluations, Phase 6 incorporates five specialized diagnostic analyses designed to probe internal model mechanics:

1. **Raw vs. Canonicalized Feature Sensitivity:**
   - Evaluates whether the feature extraction pipeline extracts divergent numerical feature vectors from semantically identical raw vs. RFC-canonicalized URLs (e.g. unreserved percent-decoding, lowercase host normalization, default port stripping).
   - Computes Maximum Absolute Feature Drift $\max_j |x_{\text{raw}, j} - x_{\text{canon}, j}|$.

2. **Unicode Normalization & Punycode Equivalence:**
   - Evaluates whether Unicode normalization forms (NFC, NFD, NFKC, NFKD) or conversion to ASCII Punycode (`xn--...`) induce spurious feature shifts or classification flips.

3. **Continuous Feature Perturbation Scans:**
   - Systematically scans the top SHAP-identified continuous features (`path_length`, `max_path_segment_length`, `digit_ratio`, `url_entropy`) along a fixed grid ($\pm 10\%, \pm 25\%, \pm 50\%$) while holding remaining features constant.
   - Measures the empirical margin distance $\Delta x_j^*$ required to force a decision boundary transition ($\hat{p} = 0.50$).

4. **Probability Calibration Drift (Brier Score & ECE):**
   - Assesses the degradation of posterior probability calibration under distribution shift.
   - Computes Brier Score and Expected Calibration Error (ECE across $M=10$ equal-frequency bins) before and after adversarial perturbation.

5. **Out-of-Distribution (OOD) Structural Failure Boundaries:**
   - Probes model behavior when presented with structural anomalies not encountered during training:
     - Userinfo authentication tokens (`https://user:pass@host/`);
     - Massive query strings ($>1,000$ characters);
     - Zero-path root architectures over HTTPS.

---

## 7. Sampling Protocol & Applicability Accounting

### 7.1 Stratified, Domain-Aware Sampling Protocol
To ensure computational tractability while maintaining strict domain-level isolation and statistical power:
- **Internal Held-Out Benchmark Sample ($S_{\text{int}}$):** $N = 2,000$ URLs ($1,000$ Phishing, $1,000$ Legitimate) sampled via stratified selection across unique registrable domains from the LegitPhish test partition.
- **External Disjoint Benchmark Sample ($S_{\text{ext}}$):** $N = 2,000$ URLs ($1,000$ Phishing, $1,000$ Legitimate) sampled via stratified selection across unique registrable domains from the strict external PhiUSIIL disjoint partition.
- **Random Seed:** Deterministically fixed to Seed 42.
- **Domain Independence Guarantee:**
  $$\text{Domain}(S_{\text{int}}) \cap \text{Domain}(\text{Train}) = \emptyset, \quad \text{Domain}(S_{\text{ext}}) \cap \text{Domain}(\text{Train}) = \emptyset$$

### 7.2 Explicit Transformation Applicability Accounting
To prevent survivorship bias, the evaluation harness tracks and reports five parameters for every attack category:
1. $N_{\text{cand}}$: Total candidate URLs evaluated.
2. $N_{\text{app}}$: Total URLs possessing the prerequisite syntactic structure (e.g. non-empty path for path encoding; domain name for subdomain prepending).
3. $N_{\text{tx}}$: Total URLs successfully perturbed and validated against Tier constraints.
4. $A_r = \frac{N_{\text{app}}}{N_{\text{cand}}}$: Applicability Rate.
5. $T_r = \frac{N_{\text{tx}}}{N_{\text{app}}}$: Transformation Success Rate.

---

## 8. Artifact Protection & Scientific Integrity Protocol

The evaluation adheres strictly to the following scientific integrity constraints:
1. **Model & Dataset Freeze:** All artifacts from Phases 1 through 5 (`models/trained/`, `data/processed/`, `reports/tables/`, `research/shap_analysis.md`) remain strictly immutable. No retraining or fine-tuning of models is permitted.
2. **Dedicated Workspace Layout:**
   - Adversarial attack generators: `src/adversarial/attacks.py`
   - Evaluation engine and statistical harness: `src/adversarial/evaluator.py`
   - Pipeline orchestrator: `src/adversarial/run_adversarial_pipeline.py`
   - Perturbed dataset artifacts: `data/adversarial/`
   - Formatted results tables: `reports/tables/adversarial/`
   - Visualization charts: `reports/figures/adversarial/`
   - Research synthesis: `research/phase6_...`
3. **Execution Constraints:** No Git or GitHub operations. All code runs sandboxed with deterministic seeds.

---

## 9. Pre-Implementation Audit Sign-Off

The methodology specified in [`research/phase6_robustness_methodology.md`](file:///Users/pratyushsharma/Desktop/phishing-detection-research/research/phase6_robustness_methodology.md) and reviewed herein satisfies all scientific, statistical, and operational requirements.

- [x] **Transformation Validity:** Three-tier classification grounded in RFC standards enforced.
- [x] **Partition Separation:** Internal test and external disjoint benchmarks strictly segregated.
- [x] **Metric Definitions:** Unambiguous numerators and denominators formalized.
- [x] **Statistical Rigor:** Paired bootstrap, McNemar testing, and Benjamini-Hochberg FDR control integrated.
- [x] **Applicability Accounting:** Survivorship bias tracking formalized.
- [x] **Artifact Safety:** Strict freeze on Phase 1–5 assets verified.

**Implementation Status:** Pre-implementation methodology review is **100% COMPLETE**. As commanded, all implementation and attack executions are paused pending user direction.

