"""Reproducible sampling module for SHAP background and evaluation datasets.

Extracts stratified, leakage-safe subsets of:
1. Primary Training data (Background reference for SHAP TreeExplainer).
2. Internal Held-out Test data (LegitPhish domain-disjoint test partition).
3. External Evaluation data (PhiUSIIL strict entire-primary-dataset-disjoint partition).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models.pipeline import create_deterministic_domain_splits, load_datasets


def get_stratified_evaluation_samples(
    base_dir: str | Path = ".",
    n_background: int = 1000,
    n_internal: int = 1000,
    n_external: int = 1000,
    random_state: int = 42,
) -> Tuple[
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame],
    List[str],
    Dict[str, Any],
]:
    """Extract stratified background and evaluation samples with verified metadata.

    Args:
        base_dir: Project root directory.
        n_background: Number of training records to sample for SHAP background.
        n_internal: Number of internal test records to sample for internal explanation.
        n_external: Number of external disjoint records to sample for external explanation.
        random_state: Deterministic random seed.

    Returns:
        Tuple of:
            (feat_bg, meta_bg),
            (feat_internal, meta_internal),
            (feat_external, meta_external),
            feature_names,
            sampling_metadata_dict
    """
    base_path = Path(base_dir)
    feat_l, meta_l, feat_p, meta_p, fnames = load_datasets(base_path)

    # Recreate deterministic domain splits for LegitPhish (70/15/15)
    train_idx, val_idx, test_idx, split_info = create_deterministic_domain_splits(
        meta_l, random_state=random_state
    )

    # 1. Background Sample: drawn strictly from LegitPhish Training Set
    feat_train = feat_l.iloc[train_idx].copy()
    meta_train = meta_l.iloc[train_idx].copy()

    # Stratified sample by label
    rng = np.random.RandomState(random_state)
    bg_sub_0 = feat_train[feat_train["label"] == 0].sample(
        n=int(n_background * (feat_train["label"] == 0).mean()),
        random_state=random_state,
    )
    bg_sub_1 = feat_train[feat_train["label"] == 1].sample(
        n=n_background - len(bg_sub_0),
        random_state=random_state,
    )
    bg_idx = pd.concat([bg_sub_0, bg_sub_1]).index
    # Shuffle
    bg_idx = rng.permutation(bg_idx)

    feat_bg = feat_train.loc[bg_idx].reset_index(drop=True)
    meta_bg = meta_train.loc[bg_idx].reset_index(drop=True)

    # 2. Internal Explanation Sample: drawn strictly from LegitPhish Test Set
    feat_test = feat_l.iloc[test_idx].copy()
    meta_test = meta_l.iloc[test_idx].copy()

    test_sub_0 = feat_test[feat_test["label"] == 0].sample(
        n=int(n_internal * (feat_test["label"] == 0).mean()),
        random_state=random_state,
    )
    test_sub_1 = feat_test[feat_test["label"] == 1].sample(
        n=n_internal - len(test_sub_0),
        random_state=random_state,
    )
    test_sample_idx = pd.concat([test_sub_0, test_sub_1]).index
    test_sample_idx = rng.permutation(test_sample_idx)

    feat_internal = feat_test.loc[test_sample_idx].reset_index(drop=True)
    meta_internal = meta_test.loc[test_sample_idx].reset_index(drop=True)

    # 3. External Explanation Sample: drawn strictly from PhiUSIIL Entire-Disjoint Set
    all_legit_doms = set(meta_l["registrable_domain"].dropna())
    ext_disjoint_mask = ~meta_p["registrable_domain"].isin(all_legit_doms)

    feat_ext_disjoint = feat_p[ext_disjoint_mask].copy()
    meta_ext_disjoint = meta_p[ext_disjoint_mask].copy()

    ext_sub_0 = feat_ext_disjoint[feat_ext_disjoint["label"] == 0].sample(
        n=int(n_external * (feat_ext_disjoint["label"] == 0).mean()),
        random_state=random_state,
    )
    ext_sub_1 = feat_ext_disjoint[feat_ext_disjoint["label"] == 1].sample(
        n=n_external - len(ext_sub_0),
        random_state=random_state,
    )
    ext_sample_idx = pd.concat([ext_sub_0, ext_sub_1]).index
    ext_sample_idx = rng.permutation(ext_sample_idx)

    feat_external = feat_ext_disjoint.loc[ext_sample_idx].reset_index(drop=True)
    meta_external = meta_ext_disjoint.loc[ext_sample_idx].reset_index(drop=True)

    # Verify zero domain overlap between background and external sample
    bg_domains = set(meta_bg["registrable_domain"].dropna())
    int_domains = set(meta_internal["registrable_domain"].dropna())
    ext_domains = set(meta_external["registrable_domain"].dropna())

    assert len(bg_domains.intersection(int_domains)) == 0, "Background-Internal domain leakage!"
    assert len(bg_domains.intersection(ext_domains)) == 0, "Background-External domain leakage!"
    assert len(int_domains.intersection(ext_domains)) == 0, "Internal-External domain leakage!"

    metadata = {
        "random_state": random_state,
        "n_features": len(fnames),
        "background_sample": {
            "source": "LegitPhish Training Split",
            "n_records": int(len(feat_bg)),
            "n_unique_domains": int(meta_bg["registrable_domain"].nunique()),
            "phishing_count": int((feat_bg["label"] == 0).sum()),
            "legitimate_count": int((feat_bg["label"] == 1).sum()),
            "phishing_percentage": round((feat_bg["label"] == 0).mean() * 100, 2),
        },
        "internal_explanation_sample": {
            "source": "LegitPhish Test Split (Domain-Disjoint)",
            "n_records": int(len(feat_internal)),
            "n_unique_domains": int(meta_internal["registrable_domain"].nunique()),
            "phishing_count": int((feat_internal["label"] == 0).sum()),
            "legitimate_count": int((feat_internal["label"] == 1).sum()),
            "phishing_percentage": round((feat_internal["label"] == 0).mean() * 100, 2),
        },
        "external_explanation_sample": {
            "source": "PhiUSIIL Entire-Primary-Dataset-Disjoint",
            "n_records": int(len(feat_external)),
            "n_unique_domains": int(meta_external["registrable_domain"].nunique()),
            "phishing_count": int((feat_external["label"] == 0).sum()),
            "legitimate_count": int((feat_external["label"] == 1).sum()),
            "phishing_percentage": round((feat_external["label"] == 0).mean() * 100, 2),
        },
    }

    return (
        (feat_bg, meta_bg),
        (feat_internal, meta_internal),
        (feat_external, meta_external),
        fnames,
        metadata,
    )

