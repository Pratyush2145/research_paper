# Proposed Methodology

**Status: proposal only. This methodology has not been implemented or evaluated.**

## Problem Definition

The proposed input is a URL string. The output is a binary classification (`phishing` or `legitimate`) accompanied by a model-estimated risk score. A score is a model output, not a calibrated probability or a guarantee of safety unless calibration and its limitations are explicitly evaluated.

The initial detector will analyze the supplied string only. It will not open, visit, resolve, or download content from the submitted URL.

## Features and Models

The planned predictors are lexical and structural properties derived from the URL string. Page content, network responses, DNS lookups, and other external signals are outside the initial feature scope.

Models planned for comparison:

- Logistic Regression as a baseline.
- Random Forest.
- XGBoost, subject to a later compatibility check before installation.
- Support Vector Machine (SVM).

Feature extraction, preprocessing, hyperparameter selection, and decision thresholds will be documented before interpreting evaluation results.

## Explainability

SHAP is the proposed explanation approach. The analysis will distinguish global summaries from individual prediction explanations and will describe limitations, including dependence on the fitted model and input representation. SHAP has not been installed or run.

## Evaluation

The planned metrics are accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, Matthews correlation coefficient (MCC), false positive rate, and confusion matrix. Metric averaging, positive-class convention, threshold selection, and handling of class imbalance will be specified in the experiment protocol.

## Generalization and Data Splits

Evaluation will compare a random split with a domain-separated split and external-dataset validation. The domain-separated protocol will group URLs by registrable domain so that URLs from the same registrable domain do not occur in both training and test partitions. Dataset provenance, labels, preprocessing, deduplication, and split decisions will be recorded to reduce leakage and support reproducibility. External data will not be used for model selection if it is reserved as a final validation set.

No dataset has been chosen, downloaded, or processed, and no experimental results exist yet.
