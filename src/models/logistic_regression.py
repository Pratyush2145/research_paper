"""Logistic Regression classifier builder for URL phishing detection.

Provides linear probability baseline model with regularized optimization.
"""

from __future__ import annotations

from typing import Any, Optional
from sklearn.linear_model import LogisticRegression


def build_logistic_regression(
    C: float = 1.0,
    class_weight: Optional[str | dict] = None,
    random_state: int = 42,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Instantiate a configured LogisticRegression model.

    Args:
        C: Inverse of regularization strength.
        class_weight: Optional class weights ('balanced' or dict).
        random_state: Deterministic random seed.
        max_iter: Maximum solver iterations.

    Returns:
        LogisticRegression instance.
    """
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        random_state=random_state,
        max_iter=max_iter,
    )
