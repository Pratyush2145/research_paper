"""Random Forest classifier builder for URL phishing detection.

Provides configured Random Forest estimators for nonlinear and feature
interaction evaluation.
"""

from __future__ import annotations

from typing import Any, Optional
from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = 20,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    class_weight: Optional[str | dict] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Instantiate a configured RandomForestClassifier.

    Args:
        n_estimators: Number of decision trees.
        max_depth: Maximum tree depth to constrain overfitting.
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf: Minimum samples required at a leaf node.
        class_weight: Optional class weights ('balanced' or dict).
        random_state: Deterministic random seed.
        n_jobs: Thread count for parallel execution.

    Returns:
        RandomForestClassifier instance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
