# Explainable AI Using SHAP: Global, Local, and Generalization-Aware Feature Attribution

**Research Project:** Explainable and Generalization-Aware Phishing URL Detection Using Machine Learning and SHAP  
**Document:** Research Analysis — Phase 5: SHAP Explainability & Attribution Validation  
**Date:** November 2026  
**Phase:** Phase 5 — Model Interpretability & Generalization-Aware Attribution  
**Status:** Scientifically Finalized and Validated  

---

## Executive Summary

Phase 5 implements a rigorous post-hoc explainability framework using **SHAP (SHapley Additive exPlanations)** based on cooperative game theory. We evaluate how our pre-trained, leakage-safe URL-only models—primarily the champion **Random Forest** ensemble and secondarily the benchmark **XGBoost** model—distinguish phishing threats from benign websites across both internal domain-disjoint test data and strict out-of-distribution external data.

All explanations strictly target the **Phishing Threat Class (`Class 0`)**:
- **Positive SHAP Value ($\phi_j > 0$):** Increases the likelihood of classification as **Phishing**.
- **Negative SHAP Value ($\phi_j < 0$):** Decreases the likelihood of Phishing (acts as evidence for **Legitimate/Benign**).
- **Random Forest Base Value:** $\approx 0.6373$ in **probability units** ($P(\text{Class 0})$), matching the training prior.
- **XGBoost Base Value:** $+0.6661$ in **margin log-odds units** ($\log \frac{P(0)}{P(1)}$), corresponding to $\sigma(+0.6661) = 0.6607$ prior probability of phishing across the tree leaf weights.

```text
SHAP Explainability Core Research Findings Matrix:
┌──────────────────────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│ Evaluation Dimension                         │ Primary Model: Random Forest  │ Secondary Model: XGBoost      │
├──────────────────────────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Explanatory Target Metric                    │ Probability P(Class 0)        │ Margin Log-Odds (Class 0)     │
│ Explanatory Base Value                       │ 0.6373 (Probability units)    │ +0.6661 (Log-Odds units)      │
│ Additivity Check Status                      │ Exact (Max Error = 2.77e-16)  │ Exact (Max Error = 3.81e-06)  │
│ Top Feature (Internal Test)                  │ path_length                   │ path_length                   │
│ Top Feature (External Disjoint)              │ protocol_type                 │ path_length                   │
│ Internal vs. External Spearman Rank Corr (ρ) │ ρ = 0.9906 (Descriptive)      │ ρ = 0.9910 (Descriptive)      │
│ Internal vs. External Kendall Tau (τ)        │ τ = 0.9387 (Descriptive)      │ τ = 0.9448 (Descriptive)      │
│ Top-10 Feature Jaccard Set Overlap           │ 1.0000 (10/10 features shared)│ 0.8182 (9/11 features shared) │
│ Top-20 Feature Jaccard Set Overlap           │ 0.8182 (18/22 features shared)│ 0.8182 (18/22 features shared)│
│ Cross-Model Spearman Rank Correlation (ρ)    │               ρ = 0.8271 (Descriptive Concordance)           │
│ Strict External Phishing Recall (Full Set)   │ 95.42% (67,816 / 71,069)      │ 49.31% (35,046 / 71,069)      │
│ Strict External Recall (Bootstrap Mean)      │ 95.42% [95% CI: 94.23-96.22%] │ 49.27% [95% CI: 45.89-53.58%] │
│ 1,000-Sample External Explanation Recall     │ 94.88% (408 / 430)            │ 47.44% (204 / 430)            │
└──────────────────────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

### Direct Answers to the Six Central Research Questions:
1. **Which URL features contribute most to phishing detection?**  
   Across both architectures and datasets, URL path structure (`path_length`, `max_path_segment_length`, `path_segment_count`), transport protocol (`protocol_type`, `is_https`, `is_http`), character repetition (`max_consecutive_repeated_char`), and structural complexity (`slash_count`, `dot_count`, `digit_ratio`) represent the primary drivers of model predictions.
2. **How do feature contributions differ between phishing and legitimate URLs?**  
   Legitimate URLs are characterized by strong negative SHAP contributions driven by the complete absence of path segments (`is_path_empty = 1`, `path_segment_count = 0`), zero path length, and valid HTTPS protocol (`protocol_type = 2`). Phishing URLs exhibit strong positive SHAP contributions driven by deep multi-segment paths, cleartext HTTP protocol, character repetitions, and high digit ratios.
3. **Are explanations consistent across internal and external datasets (Domain Shift Invariance)?**  
   **Yes, descriptively highly invariant.** The Spearman rank correlation between internal held-out and external feature importance exceeds **$\rho = 0.990$** for both Random Forest and XGBoost. The Top-10 features for Random Forest are **$100\%$ identical** (Jaccard = $1.000$) between internal held-out test data and external out-of-distribution data. *(Note: This ranking concordance demonstrates behavioral stability across the evaluated datasets; it does not constitute mathematical proof of causal invariance or universal immunity to adversarial shift).*
4. **Why do Random Forest and XGBoost produce different predictions on external data?**  
   SHAP feature attributions clearly isolate the mechanism: XGBoost's sequential boosting concentrated **$>90\%$** of its split gain on `path_length` and `path_segment_count`. In the external PhiUSIIL dataset, where $72.8\%$ of phishing URLs have no path, XGBoost heavily penalized root phishing URLs with negative logits, dropping posterior probabilities to $0.40 - 0.48$ (median $0.4596$), missing the $0.50$ threshold. In contrast, Random Forest's random feature subspace sampling ($\approx 8$ features per split) forced many trees to train on path-free subsets (relying on host entropy, digit ratio, and transport protocol), allowing non-path trees to outvote path-dependent trees and sustain $95.42\%$ external recall.
5. **Do the models rely on robust URL characteristics or benchmark-specific patterns?**  
   Models utilize both: path presence is a genuine operational characteristic of credential-harvesting kits (which deploy PHP/HTML login scripts into subdirectories), but its extreme correlation in LegitPhish ($99.4\%$ in phishing vs. $1.4\%$ in legitimate) is a collection artifact. Under external domain shift, Random Forest dynamically shifts its primary attribution weight to transport protocol (`protocol_type` surges to Rank 1), demonstrating ensemble resilience against single-feature reliance.
6. **Can individual phishing predictions be explained clearly to human analysts?**  
   **Yes.** Local SHAP waterfall plots decompose individual URL predictions into an additive balance sheet of risk factors (positive red attributions) and benign indicators (negative blue attributions), enabling security analysts to audit decision rationale in sub-millisecond execution times.

---

## 1. Background Dataset & Evaluation Sampling Methodology

To ensure strict reproducibility, prevent data snooping, and eliminate information leakage:
1. **Background Reference Dataset:** Formed strictly from the **LegitPhish primary training partition** ($N = 1,000$ records, stratified by class: $637$ Phishing, $363$ Legitimate; $871$ unique domains; Random Seed $42$). External evaluation data was never used as background reference.
2. **Internal Explanation Dataset:** Formed strictly from the held-out **LegitPhish domain-disjoint test partition** ($N = 1,000$ records, stratified by class: $622$ Phishing, $378$ Legitimate; $861$ unique domains; Random Seed $42$).
3. **External Explanation Dataset:** Formed strictly from the **PhiUSIIL entire-primary-dataset-disjoint partition** ($N = 1,000$ records, stratified by class: $430$ Phishing, $570$ Legitimate; $964$ unique domains; Random Seed $42$).
4. **Zero Domain Leakage Verification:**  
   $$\text{Domain}(BG) \cap \text{Domain}(Int) = \emptyset, \quad \text{Domain}(BG) \cap \text{Domain}(Ext) = \emptyset, \quad \text{Domain}(Int) \cap \text{Domain}(Ext) = \emptyset$$  
   All three partitions are completely disjoint with zero shared registrable domains.

---

## 2. Global Feature Importance & Contribution Analysis

### Table 1: Random Forest — Top 20 Global Features (Internal vs. External)
*Units: Mean Absolute SHAP Impact on Class 0 Phishing Probability ($P(\text{Phish}) \in [0, 1]$)*

| Rank | Feature Name | Internal Mean \|SHAP\| | Internal Effect Direction | External Mean \|SHAP\| | External Effect Direction | Domain Rank Shift |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: |
| **1** | `path_length` | 0.064128 | Higher $\to$ Phishing | 0.067637 | Higher $\to$ Phishing | -2 (1 $\to$ 3) |
| **2** | `max_path_segment_length` | 0.047351 | Higher $\to$ Phishing | 0.067476 | Higher $\to$ Phishing | -2 (2 $\to$ 4) |
| **3** | `path_segment_count` | 0.040842 | Higher $\to$ Phishing | 0.057102 | Higher $\to$ Phishing | -2 (3 $\to$ 5) |
| **4** | `protocol_type` | 0.040147 | Higher $\to$ Legitimate | 0.071657 | Higher $\to$ Legitimate | **+3 (4 $\to$ 1)** |
| **5** | `is_https` | 0.038846 | Higher $\to$ Legitimate | 0.070307 | Higher $\to$ Legitimate | **+3 (5 $\to$ 2)** |
| **6** | `slash_count` | 0.036251 | Higher $\to$ Phishing | 0.046573 | Higher $\to$ Phishing | -1 (6 $\to$ 7) |
| **7** | `max_consecutive_repeated_char` | 0.035299 | Higher $\to$ Legitimate | 0.042735 | Higher $\to$ Legitimate | -1 (7 $\to$ 8) |
| **8** | `avg_path_segment_length` | 0.034189 | Higher $\to$ Phishing | 0.051278 | Higher $\to$ Phishing | +2 (8 $\to$ 6) |
| **9** | `is_path_empty` | 0.023032 | Higher $\to$ Legitimate | 0.036279 | Higher $\to$ Legitimate | -1 (9 $\to$ 10) |
| **10** | `is_http` | 0.020212 | Higher $\to$ Phishing | 0.037592 | Higher $\to$ Phishing | **+1 (10 $\to$ 9)** |
| **11** | `dot_count` | 0.018679 | Higher $\to$ Phishing | 0.016333 | Higher $\to$ Phishing | 0 (11 $\to$ 11) |
| **12** | `digit_ratio` | 0.018113 | Higher $\to$ Phishing | 0.012528 | Higher $\to$ Phishing | -1 (12 $\to$ 13) |
| **13** | `digit_count` | 0.016003 | Higher $\to$ Phishing | 0.006932 | Higher $\to$ Phishing | -4 (13 $\to$ 17) |
| **14** | `subdomain_count` | 0.011855 | Non-linear / Mixed | 0.010196 | Non-linear / Mixed | 0 (14 $\to$ 14) |
| **15** | `special_char_count` | 0.008409 | Higher $\to$ Phishing | 0.009949 | Higher $\to$ Phishing | 0 (15 $\to$ 15) |
| **16** | `hostname_has_digits` | 0.003854 | Higher $\to$ Phishing | 0.007693 | Higher $\to$ Phishing | 0 (16 $\to$ 16) |
| **17** | `path_has_file_extension` | 0.003628 | Higher $\to$ Phishing | 0.014294 | Higher $\to$ Phishing | **+5 (17 $\to$ 12)** |
| **18** | `path_file_extension_count` | 0.003173 | Higher $\to$ Phishing | 0.006323 | Higher $\to$ Phishing | 0 (18 $\to$ 18) |
| **19** | `alpha_ratio` | 0.003153 | Higher $\to$ Legitimate | 0.004928 | Higher $\to$ Legitimate | 0 (19 $\to$ 19) |
| **20** | `is_ip_address` | 0.003084 | Higher $\to$ Phishing | 0.002996 | Higher $\to$ Phishing | 0 (20 $\to$ 20) |

---

### Table 2: XGBoost — Top 20 Global Features (Internal vs. External)
*Units: Mean Absolute SHAP Impact on Class 0 Phishing Margin Log-Odds ($\log \frac{P(0)}{P(1)} \in (-\infty, +\infty)$)*

| Rank | Feature Name | Internal Mean \|SHAP\| | Internal Effect Direction | External Mean \|SHAP\| | External Effect Direction | Domain Rank Shift |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: |
| **1** | `path_length` | 4.590352 | Higher $\to$ Phishing | 5.462294 | Higher $\to$ Phishing | 0 (1 $\to$ 1) |
| **2** | `is_https` | 1.707069 | Higher $\to$ Legitimate | 2.000826 | Higher $\to$ Legitimate | 0 (2 $\to$ 2) |
| **3** | `max_consecutive_repeated_char` | 1.129532 | Higher $\to$ Legitimate | 1.303542 | Higher $\to$ Legitimate | 0 (3 $\to$ 3) |
| **4** | `alpha_ratio` | 0.762470 | Higher $\to$ Legitimate | 0.282924 | Higher $\to$ Legitimate | -5 (4 $\to$ 9) |
| **5** | `dot_count` | 0.672529 | Higher $\to$ Phishing | 0.522349 | Higher $\to$ Phishing | **+1 (5 $\to$ 4)** |
| **6** | `is_http` | 0.508056 | Higher $\to$ Phishing | 0.496018 | Higher $\to$ Phishing | 0 (6 $\to$ 6) |
| **7** | `path_segment_count` | 0.407865 | Higher $\to$ Phishing | 0.511790 | Higher $\to$ Phishing | **+2 (7 $\to$ 5)** |
| **8** | `subdomain_count` | 0.326471 | Higher $\to$ Legitimate | 0.304930 | Higher $\to$ Legitimate | 0 (8 $\to$ 8) |
| **9** | `digit_ratio` | 0.296704 | Higher $\to$ Phishing | 0.220166 | Higher $\to$ Phishing | -2 (9 $\to$ 11) |
| **10** | `slash_count` | 0.216228 | Higher $\to$ Phishing | 0.306339 | Higher $\to$ Phishing | **+3 (10 $\to$ 7)** |
| **11** | `hostname_has_digits` | 0.213993 | Higher $\to$ Phishing | 0.161042 | Higher $\to$ Phishing | -1 (11 $\to$ 12) |
| **12** | `max_path_segment_length` | 0.204206 | Higher $\to$ Phishing | 0.150493 | Higher $\to$ Phishing | -1 (12 $\to$ 13) |
| **13** | `avg_hostname_label_length` | 0.197127 | Higher $\to$ Phishing | 0.263211 | Higher $\to$ Phishing | **+3 (13 $\to$ 10)** |
| **14** | `path_has_file_extension` | 0.178491 | Higher $\to$ Phishing | 0.134810 | Higher $\to$ Phishing | 0 (14 $\to$ 14) |
| **15** | `alpha_count` | 0.147719 | Higher $\to$ Legitimate | 0.119280 | Higher $\to$ Legitimate | 0 (15 $\to$ 15) |
| **16** | `hostname_length` | 0.136890 | Higher $\to$ Phishing | 0.108422 | Higher $\to$ Phishing | 0 (16 $\to$ 16) |
| **17** | `url_entropy` | 0.104850 | Higher $\to$ Phishing | 0.089201 | Higher $\to$ Phishing | 0 (17 $\to$ 17) |
| **18** | `url_length` | 0.098205 | Higher $\to$ Phishing | 0.081220 | Higher $\to$ Phishing | 0 (18 $\to$ 18) |
| **19** | `avg_path_segment_length` | 0.091194 | Higher $\to$ Phishing | 0.065110 | Higher $\to$ Phishing | 0 (19 $\to$ 19) |
| **20** | `path_has_file_extension` | 0.071941 | Higher $\to$ Phishing | 0.059410 | Higher $\to$ Phishing | 0 (20 $\to$ 20) |

---

## 3. Methodological Audit: Tree Importance, Feature Correlation & Attribution Mechanics

### 3.1 Cross-Model Scale Comparability
A fundamental methodological principle verified in our audit is that **Random Forest and XGBoost SHAP values cannot be compared in absolute magnitudes**:
- **Random Forest:** SHAP values represent linear perturbations in probability space $[0, 1]$. An attribution of $+0.064$ directly shifts predicted probability $P(\text{Phish})$ by $+6.4$ percentage points.
- **XGBoost:** SHAP values represent linear perturbations in margin log-odds space $(-\infty, +\infty)$. An attribution of $+4.59$ shifts the logit $\log \frac{P(0)}{P(1)}$ by $+4.59$, which has a non-linear effect on probability depending on the baseline operating point via $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- Cross-model comparisons are therefore conducted strictly using non-parametric ordinal statistics (Spearman rank correlation $\rho = 0.8271$) and Top-$K$ set overlap.

### 3.2 Impact of Feature Collinearity on Attribution
URL syntax naturally produces strongly correlated feature pairs:
- `protocol_type` vs `is_https`: $r = +1.0000$ (perfect collinearity).
- `path_length` vs `max_path_segment_length`: $r = +0.9375$.
- `path_segment_count` vs `slash_count`: $r = +0.9570$.
- `path_length` vs `path_segment_count`: $r = +0.7975$.

Under `TreeExplainer(model)` with `feature_perturbation="tree_path_dependent"`, the explainer conditions on observed tree paths. When two features are collinear, the tree splits arbitrarily between them across different nodes. Consequently, **SHAP distributes the attribution across both correlated features**:
- The total transport protocol impact ($|SHAP| \approx 0.079$ in RF) is divided between `protocol_type` ($0.040$) and `is_https` ($0.039$).
- Analysts must **not** interpret `protocol_type` as having an independent causal effect distinct from `is_https`.

### 3.3 Grouped Feature Family Analysis
To eliminate collinearity splitting artifacts, we grouped all 68 features into five semantic families:

| Feature Family | Primary Features Included | RF Internal Attribution (%) | RF External Attribution (%) | XGB Internal Attribution (%) | XGB External Attribution (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Path Structure & Depth** | `path_length`, `max_path_segment_length`, `path_segment_count`, `slash_count`, `is_path_empty`, extensions | **52.1%** | **48.3%** | **45.3%** | **49.9%** |
| **Transport Protocol Security** | `protocol_type`, `is_https`, `is_http` | **20.4%** | **25.9%** (Surge) | **17.8%** | **18.6%** |
| **Lexical & Syntactic Complexity** | `max_consecutive_repeated_char`, `dot_count`, `digit_ratio`, `digit_count`, `url_entropy` | **22.4%** | **20.3%** | **28.4%** | **22.7%** |
| **Host & Domain Morphology** | `subdomain_count`, `hostname_length`, `hostname_has_digits`, `avg_hostname_label_length`, `is_ip_address` | **5.2%** | **5.5%** | **8.5%** | **8.8%** |
| **Suspicious Security Keywords** | `has_kw_login`, `has_kw_verify`, `has_kw_account`, `has_kw_update`, etc. (16 keyword flags) | **<0.1%** | **<0.1%** | **<0.1%** | **<0.1%** |

*Key Structural Finding:* URL-only phishing detection relies overwhelmingly on **Path Structure** ($\approx 50\%$), **Transport Security** ($\approx 20-26\%$), and **Lexical Complexity** ($\approx 21-28\%$). Lexical keyword flags contribute $<0.1\%$, showing that modern machine learning models bypass superficial keyword matching in favor of robust structural syntax.

---

## 4. Feature Dependence vs. Partial Dependence Plots

In our dependence analysis:
- **Clarification:** Figures in `reports/figures/shap/dependence/` are **SHAP Dependence Scatter Plots**, NOT classical **Partial Dependence Plots (PDPs)**:
  - **Classical PDPs (Friedman, 2001):** Compute the average marginal model prediction across the population: $\bar{f}(x_j) = \frac{1}{N} \sum_{i=1}^N f(x_j, x_{i,-j})$. They display a single deterministic curve and assume feature independence.
  - **SHAP Dependence Plots:** Plot the actual instance-level Shapley value $\phi_{i,j}$ against feature value $x_{i,j}$ for each evaluation record, color-coded by the most interacting feature. Vertical scatter at a single $x_j$ value explicitly reveals interaction effects with other features.
- **Threshold Inflections Identified:**
  - `path_length`: Values below $22$ characters receive negative/neutral SHAP attribution; values exceeding $22$ characters exhibit a sharp, non-linear inflection to positive phishing attribution ($\phi > +0.05$).
  - `protocol_type`: Clean step function where HTTP ($1$) adds $+0.038$ to phishing probability, while HTTPS ($2$) subtracts $-0.041$.

---

## 5. Local Explainability: Representative Diagnostic Case Studies

Using Random Forest on verified dataset records:

### Case 1: True Positive (Phishing Intercepted — Deep Path Harvester)
- **URL:** `http://phoenixfunctions.com.au/mygov/otp2.html` (External PhiUSIIL Record #136)
- **Actual:** Phishing | **Predicted:** Phishing | **$P(\text{phish})$:** $1.000$ (Base = $0.6373$)
- **Top Risk Indicators (Positive SHAP):**
  - `path_length = 16` ($\phi = +0.078$): Explicit directory path characteristic of login lures.
  - `protocol_type = 1 (HTTP)` ($\phi = +0.037$): Cleartext unencrypted transport.
  - `is_https = 0` ($\phi = +0.036$): Absence of TLS encryption.
- **SOC Assessment:** High-confidence threat verdict explained by insecure transport combined with credential-harvesting endpoint syntax.

### Case 2: True Negative (Legitimate Domain Correctly Permitted)
- **URL:** `https://www.artra-group.co.jp` (External PhiUSIIL Record)
- **Actual:** Legitimate | **Predicted:** Legitimate | **$P(\text{phish})$:** $0.000$ (Base = $0.6373$)
- **Top Benign Indicators (Negative SHAP):**
  - `max_path_segment_length = 0` ($\phi = -0.090$): Complete absence of sub-paths or scripts.
  - `path_segment_count = 0` ($\phi = -0.075$): Clean apex landing page.
  - `protocol_type = 2 (HTTPS)` ($\phi = -0.041$): Standard encrypted transport.
- **SOC Assessment:** Permitted with zero friction due to authoritative apex domain morphology.

### Case 3: False Negative (Apex Domain Structure in Phishing-Labelled Benchmark Record)
- **URL:** `https://www.faleballe.com` (PhiUSIIL Row 99076, Phishing-Labelled Benchmark Record, Label 0)
- **Benchmark Ground Truth:** Phishing (Class 0) | **Model Prediction:** Legitimate (Class 1) | **$P(\text{phish})$:** $0.000$
- **Feature Attribution Analysis:**
  - The record is an apex domain URL lacking path components (`path_length = 0`, `is_path_empty = 1`).
  - The URL uses HTTPS transport (`is_https = 1`, `protocol_type = 2`).
  - Because the training distribution contained almost zero root-level HTTPS phishing URLs, the model assigned large negative attributions (`is_path_empty = 1` $\to \phi = -0.165$; `is_https = 1` $\to \phi = -0.112$), driving predicted phishing probability to $0.000$.
  - *Provenance Note:* This URL is analyzed strictly as a phishing-labelled record from the PhiUSIIL benchmark dataset; we do not claim independent verification of an actual live attack or confirmed adversary evasion in the wild.
- **Theoretical Boundary:** This case demonstrates an intrinsic limitation of URL-only models: when a phishing-labelled benchmark URL possesses a clean apex structure over HTTPS, purely lexical features cannot distinguish it from benign root portals without external data (such as WHOIS registration age, DNS telemetry, or webpage content).

---

## 6. Generalization-Aware Explainability & Rank Stability Audit

### 6.1 Epistemological Qualification of Ranking Concordance
- **Descriptive Concordance vs. Statistical Significance:**  
  The asymptotic p-values reported for Spearman rank correlation ($p = 1.03 \times 10^{-58}$) and Kendall's tau ($p = 6.23 \times 10^{-28}$) assume $N$ independent bivariate samples. In our study, the $N=68$ entities are fixed, mutually correlated engineered features from a closed dictionary. Therefore, degrees of freedom are inflated, and the formal p-values must not be cited as rigorous hypothesis test confirmations. Instead, $\rho = 0.9906$ and $\tau = 0.9387$ must be interpreted strictly as **high descriptive concordance metrics** between evaluation sets.
- **Attribution Stability $\ne$ Proof of Generalization:**  
  A stable feature ranking across datasets proves that the *model applies consistent decision heuristics* across both benchmarks. It does **not** prove that the learned heuristics are causally optimal, free from dataset collection bias, or universally generalizable to future zero-day phishing strategies.

### 6.2 Domain-Cluster Bootstrap Rank Uncertainty
To assess the empirical variance of feature rankings under domain resampling, we executed a 200-iteration domain-level cluster bootstrap across all 964 unique domains in the external explanation dataset:

| Feature Name | Empirical Overall Rank | Bootstrap Mean Rank | 95% Bootstrap CI | Rank Stability Assessment |
| :--- | :---: | :---: | :---: | :--- |
| `protocol_type` | **1** | 1.0 | **[1.0, 2.0]** | Exceptionally Stable (Always Top 2) |
| `is_https` | **2** | 2.1 | **[2.0, 4.0]** | Exceptionally Stable (Always Top 4) |
| `path_length` | **3** | 3.4 | **[1.0, 4.0]** | Exceptionally Stable (Always Top 4) |
| `max_path_segment_length` | **4** | 3.5 | **[2.0, 4.0]** | Exceptionally Stable (Always Top 4) |
| `path_segment_count` | **5** | 5.0 | **[5.0, 5.0]** | Perfectly Stable (Zero Variance) |
| `avg_path_segment_length` | **6** | 6.0 | **[6.0, 6.0]** | Perfectly Stable (Zero Variance) |
| `slash_count` | **7** | 7.0 | **[7.0, 7.0]** | Perfectly Stable (Zero Variance) |
| `max_consecutive_repeated_char` | **8** | 8.0 | **[8.0, 8.0]** | Perfectly Stable (Zero Variance) |
| `is_http` | **9** | 9.1 | **[9.0, 10.0]** | Exceptionally Stable (Ranks 9-10) |
| `is_path_empty` | **10** | 9.9 | **[9.0, 10.0]** | Exceptionally Stable (Ranks 9-10) |

The 200-iteration domain-cluster bootstrap quantifies the variance of individual feature ranks under domain resampling, demonstrating narrow confidence intervals for these top attributes. However, individual feature rank stability must not be conflated with complete Top-10 set invariance across arbitrary resampling subsets; set-level selection frequencies and bootstrap Jaccard distributions across resamples were not evaluated. The reported 100% Top-10 Jaccard overlap (10/10 shared features) applies strictly to the point-estimate comparison between the full internal held-out test set and the external disjoint evaluation set.

---

## 7. Operational Recommendations & Threat Boundaries

1. **Primary URL Filter:** Random Forest provides the most balanced, generalizable URL-only detection due to its subspace bagging diversity.
2. **Adversarial Blind Spot Mitigation:** URL-only detectors must be paired with downstream defenses. When a URL is an apex domain (`path_length = 0`) over HTTPS, but is newly observed in network traffic, security gateways should automatically trigger full webpage inspection rather than trusting the clean lexical URL score.
3. **Benchmarking Realism:** In the evaluated external benchmark partition (`external_entire_disjoint`), zero false positives were observed among 94,083 legitimate URLs ($FP = 0$). Under the standard rule-of-three for zero-event binomial outcomes, this establishes a 95% one-sided statistical confidence upper bound of approximately $3 / 94,083 \approx 0.0032\%$ ($3.19 \times 10^{-5}$) on this specific benchmark distribution. This is a statistical sample confidence bound, NOT an operational guarantee of zero false alarms in production network traffic, where diverse, complex benign URLs will inevitably generate false positives.

---

## Phase 5 Scientific Conclusions

1. **Attribution Reliability:** URL-only phishing detection is driven primarily by path structure ($\approx 50\%$), transport protocol security ($\approx 20-26\%$), and character complexity ($\approx 20-23\%$).
2. **Mathematical Correctness:** Both explainers satisfy exact mathematical additivity checks with numerical error $< 10^{-5}$.
3. **Bagging vs. Boosting Disparity Explained:** XGBoost concentrated $>90\%$ split importance on path features, causing it to assign negative logits to root phishing URLs in PhiUSIIL ($49.31\%$ recall). Random Forest's subspace bagging preserved alternate non-path trees that maintained $95.42\%$ recall.
4. **Generalization Concordance:** Top-10 features exhibit 100% Jaccard overlap between the internal and external point estimates, and individual top feature ranks display narrow domain-bootstrap confidence intervals, confirming strong descriptive attribution consistency across evaluated benchmarks.
