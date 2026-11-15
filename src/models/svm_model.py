"""Support Vector Machine (SVM) classifier builder for URL phishing detection.

Provides computationally practical linear margin-based classification with
calibrated probability estimates suitable for large tabular datasets.
"""

from __future__ import annotations

from typing import Any, Optional
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


def build_svm(
    C: float = 1.0,
    class_weight: Optional[str | dict] = None,
    random_state: int = 42,
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Instantiate a calibrated Linear Support Vector Classifier.

    Standard non-linear RBF kernel SVM exhibits O(N^2 * D) to O(N^3 * D)
    complexity, making it computationally impractical for datasets with >70,000
    records. LinearSVC with CalibratedClassifierCV provides efficient margin
    optimization while generating well-calibrated posterior probability estimates
    for ROC-AUC and PR-AUC ranking.

    Args:
        C: Regularization parameter.
        class_weight: Optional class weights ('balanced' or dict).
        random_state: Deterministic random seed.
        cv: Cross-validation folds for Platt probability calibration.

    Returns:
        CalibratedClassifierCV wrapping LinearSVC.
    """
    base_svc = LinearSVC(
        C=C,
        dual=False,
        class_weight=class_weight,
        random_state=random_state,
        max_iter=3000,
    )
    return CalibratedClassifierCV(estimator=base_svc, cv=cv)
