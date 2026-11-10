# Explainable and Generalization-Aware Phishing URL Detection Using Machine Learning and SHAP

A research project to study phishing-versus-legitimate classification from URL strings using lexical and structural signals, with model explanations and evaluation designed to measure generalization beyond randomly held-out URLs. This repository is in project initialization; no datasets have been added and no experiments have been run.

## Research Objectives

- Build a URL-string-only binary classifier for phishing and legitimate URLs.
- Compare conventional machine-learning algorithms under consistent preprocessing and splits.
- Explain model predictions with SHAP after model behavior is established.
- Measure how performance changes from random splits to domain-separated and external-dataset evaluation.
- Preserve reproducibility and report only results produced by documented experiments.

## Planned Algorithms

Logistic Regression is the baseline. Random Forest, XGBoost, and Support Vector Machine (SVM) are planned comparison models. Model training and dependency additions beyond the initial environment are deferred to later phases.

## Planned Explainability

SHAP is planned for global and per-example explanations. Explanations will be interpreted as model behavior, not as proof of causality or a guarantee that a URL is safe.

## Dataset Strategy

Datasets have not been selected or downloaded. Candidate sources will be reviewed and documented before use. Raw datasets will remain unchanged under `data/raw/`; derived datasets will be written separately under `data/processed/`, with independent external validation data kept under `data/external/`. Provenance, label interpretation, preprocessing, and split decisions will be recorded. URLs sharing a registrable domain must not leak across train and test partitions in domain-separated evaluation.

The planned detector will inspect URL strings only. It must not open, visit, resolve, or otherwise fetch submitted URLs.

## Project Structure

```text
data/                 Raw, processed, and external dataset locations
notebooks/            Staged research notebooks
src/                  Future data, feature, model, evaluation, and explanation modules
artifacts/            Generated model artifacts (not committed)
app/                  Reserved for a later application phase
research/             Methodology, literature review, and results records
tests/                Project setup and future behavior tests
requirements.txt      Pinned initial direct dependencies
pyproject.toml        Project metadata and pytest configuration
```

## Environment Setup

This project was initialized and tested with Python 3.14.3. Create a local environment and install the pinned initial dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select the interpreter at `.venv/bin/python` (on Windows, `.venv\\Scripts\\python.exe`). XGBoost, SHAP, Streamlit, and other later-phase dependencies are intentionally not included yet.

## Tests

With the environment active, run:

```bash
python -m pytest
```

## Current Development Phase

**Phase 1: Project initialization.** The environment, structure, and initial documentation are being established. Dataset analysis, feature engineering, model training, SHAP analysis, and application development have not started.

## Research Integrity and Security

Do not invent metrics, citations, dataset contents, labels, or findings; do not claim training or novelty without evidence. Keep raw source data unchanged, document transformations and split logic, and investigate domain leakage. Never alter results to improve reported performance. Prediction must remain offline with respect to the submitted URL string.
