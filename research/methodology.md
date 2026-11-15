# Research Methodology Specification

**Status:** Updated following Phase 2 Empirical Dataset Acquisition and Audit.

---

## 1. Problem Definition and Operational Scope

The research system predicts phishing risk exclusively from an input URL string. The prediction pipeline outputs a binary classification (`phishing` = 0 or `legitimate` = 1, following the benchmark convention) along with a model-estimated risk score.

**Strict Operational Constraints:**
1. **Zero-Network Client-Side Detection:** All inference is computed purely offline from lexical, structural, and informational properties of the URL string. The system will not open, connect to, resolve via DNS, or download content from the submitted URL.
2. **Score Interpretation:** Output scores represent model confidence rankings and will not be reported as calibrated probabilities unless explicit isotonic or Platt calibration is performed and validated.

---

## 2. Dataset Selection and Empirical Provenance

Phase 2 completed the acquisition and empirical audit of two benchmark datasets:

### 2.1 Primary Development Dataset — LegitPhish
- **Source:** Mendeley Data (DOI: [10.17632/hx4m73v2sf.2](https://doi.org/10.17632/hx4m73v2sf.2))
- **File Integrity:** Verified SHA-256 (`8685e7901c3b4e73eb43c106cb166cc932dd14f76960a032c713c2c32f09dbcd`)
- **Verified Record Count:** 101,219 rows, 18 columns.
- **Class Breakdown:** 63,678 Phishing (62.91%), 37,540 Legitimate (37.09%), 1 Missing label (0.001%).
- **Role:** Used for feature engineering validation, model training, cross-validation, and intra-dataset domain-isolated evaluation.

### 2.2 External Validation Dataset — PhiUSIIL
- **Source:** UCI Machine Learning Repository (DOI: [10.1016/j.cose.2023.103545](https://doi.org/10.1016/j.cose.2023.103545))
- **File Integrity:** Verified SHA-256 (`a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`)
- **Verified Record Count:** 235,795 rows, 56 columns.
- **Class Breakdown:** 100,945 Phishing (42.81%), 134,850 Legitimate (57.19%).
- **Role:** Reserved strictly for external generalization testing; never used for model selection, threshold tuning, or feature selection.

---

## 3. Data Cleaning and Preprocessing Protocol

Empirical findings from Phase 2 establish the following mandatory preprocessing pipeline before feature engineering:

1. **Missing Label Removal:** Drop row index `65118` (`https://www.chocolatleroux.com`) from LegitPhish due to null `ClassLabel`.
2. **Deduplication:**
   - Remove 346 exact duplicate rows from LegitPhish (leaving 100,872 unique records).
   - In PhiUSIIL, deduplicate the 425 duplicate URL strings (collapsing 850 records to 425 unique URL instances).
3. **Strict Column Filtering for PhiUSIIL:**
   - Retain only `URL` and `label`.
   - Discard all 54 webpage-derived, DOM, title, and network features (`LineOfCode`, `NoOfJS`, `URLTitleMatchScore`, etc.) to enforce strict URL-only detection.
4. **URL Normalization (Offline Only):**
   - Strip leading and trailing whitespace.
   - Correct protocol typos (e.g., `ttp://` -> `http://`) and collapse duplicate protocol prefixes (e.g., `http://http://` -> `http://`) identified in the audit.
   - Do not perform DNS canonicalization or live URL expansion.

---

## 4. Generalization Protocol and Data Leakage Mitigation

### 4.1 The Cross-Dataset Leakage Discovery
Our empirical audit revealed that **36,936 URLs** (all legitimate, representing **98.39%** of LegitPhish's legitimate set) and **35,925 registrable domains** (**57.80%** of LegitPhish domains) are identical between LegitPhish and PhiUSIIL.

Because both datasets acquired their benign URLs from standard top-domain lists (Alexa/Tranco), naive external evaluation on PhiUSIIL would evaluate a model on memorized legitimate domains, creating an illusion of high external generalization.

### 4.2 Dual-Tier External Evaluation Protocol
To guarantee scientifically sound external validation, external testing on PhiUSIIL will be conducted under two separate regimes:
1. **Unfiltered External Benchmark:** Standard full PhiUSIIL test set, reported alongside transparent documentation of the 36,936 overlapping training URLs.
2. **Purged Zero-Leakage External Benchmark:** A filtered PhiUSIIL subset where all exact URLs and all registrable domains present in the training set are strictly removed. This tests whether the model can detect phishing on truly unseen domains.

### 4.3 Internal Validation: Domain-Isolated Group-K-Fold
For development on LegitPhish:
- **Random Split Baseline:** Standard stratified 5-fold cross-validation.
- **Domain-Separated Split (Primary):** GroupKFold grouped by registrable domain. URLs from the same domain (e.g., multi-page paths on the same host) are strictly confined to either the training fold or the validation fold, preventing intra-domain memorization.

---

## 5. Planned Model Comparison and Explainability Framework

### 5.1 Models for Comparison
1. **Logistic Regression (L2 / ElasticNet):** Linear baseline for interpretability and linear separability checks.
2. **Random Forest:** Ensemble bagging baseline robust to feature scale and non-linear interactions.
3. **XGBoost:** Gradient boosted decision trees (subject to environment compatibility check before Phase 4).
4. **Support Vector Machine (Linear & RBF):** Margin-maximizing classifier baseline.

### 5.2 Explainability Architecture
- **SHAP (SHapley Additive exPlanations):** TreeExplainer for tree ensembles and LinearExplainer/KernelExplainer for linear/kernel models.
- **Analysis Scope:**
  - *Global Feature Importance:* Mean absolute SHAP values across the validation set.
  - *Local Instance Explanations:* Waterfall / force plots for individual URL risk predictions.
  - *Generalization Interaction:* Comparing SHAP feature attributions on known domains vs. out-of-distribution purged domains to verify whether models rely on robust structural features or overfit to domain tokens.

---

## 6. Evaluation Metrics

Metrics will be computed with `phishing` (0) as the target of interest for threat detection, but also reported symmetrically:
- **Primary Detection Metrics:** Precision, Recall, F1-score, PR-AUC (Precision-Recall Area Under Curve), ROC-AUC.
- **Robustness & Error Metrics:** False Positive Rate (FPR) at 95% and 99% True Positive Rate (TPR), Matthews Correlation Coefficient (MCC).
- **Confusion Matrices:** Complete counts of TP, FP, TN, FN.
