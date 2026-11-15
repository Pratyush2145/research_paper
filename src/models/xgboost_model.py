"""XGBoost gradient-boosted decision tree builder for URL phishing detection.

Provides configured XGBoost classifier for gradient boosted tree evaluation.
"""

from __future__ import annotations

from typing import Any, Optional
import xgboost as xgb


def build_xgboost(
    n_estimators: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: Optional[float] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> xgb.XGBClassifier:
    """Instantiate a configured XGBClassifier.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage (eta).
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns per tree.
        scale_pos_weight: Balancing of positive and negative weights.
        random_state: Deterministic random seed.
        n_jobs: Number of parallel threads.

    Returns:
        XGBClassifier instance.
    """
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
    )
