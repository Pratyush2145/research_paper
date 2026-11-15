"""Comprehensive evaluation metrics module for phishing URL detection.

Target Convention:
    Class 0 = Phishing (the positive threat class of interest in security evaluation)
    Class 1 = Legitimate (the negative/benign class)

All directional metrics (Precision, Recall, F1, PR-AUC, FPR, FNR) treat Phishing (0)
as the positive class to measure threat detection capability and benign disruption.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_evaluation_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_prob_phish: Optional[np.ndarray | list[float]] = None,
) -> Dict[str, Any]:
    """Calculate all 12 mandatory evaluation metrics treating Phishing (class 0) as positive.

    Args:
        y_true: Ground truth binary labels (0 = Phishing, 1 = Legitimate).
        y_pred: Predicted binary labels (0 = Phishing, 1 = Legitimate).
        y_prob_phish: Predicted probability or decision score for Phishing (class 0).

    Returns:
        Dictionary containing all 12 metrics, confusion matrix, and class documentation.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    # Confusion matrix with labels=[0, 1]
    # cm[0, 0] = True Phishing (TP)
    # cm[0, 1] = False Legitimate (FN) - phishing missed
    # cm[1, 0] = False Phishing (FP)   - legitimate blocked
    # cm[1, 1] = True Legitimate (TN)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    # 1. Accuracy
    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    # 2. Precision (Phishing = 0)
    prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))

    # 3. Recall / True Positive Rate (Phishing = 0)
    rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))

    # 4. F1-Score (Phishing = 0)
    f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))

    # 5. Macro-F1 (unweighted mean across both classes)
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    # 6. Balanced Accuracy
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    # 9. Matthews Correlation Coefficient
    mcc = float(matthews_corrcoef(y_true_arr, y_pred_arr))

    # 10. Specificity (True Negative Rate for Phishing: TN / (TN + FP))
    denom_spec = tn + fp
    spec = float(tn / denom_spec) if denom_spec > 0 else 0.0

    # 11. False Positive Rate for Phishing (FP / (FP + TN))
    fpr = float(fp / denom_spec) if denom_spec > 0 else 0.0

    # 12. False Negative Rate for Phishing (FN / (FN + TP))
    denom_fnr = fn + tp
    fnr = float(fn / denom_fnr) if denom_fnr > 0 else 0.0

    # 7 & 8: Probability-based ranking metrics (ROC-AUC and PR-AUC)
    roc_auc = None
    pr_auc = None

    if y_prob_phish is not None:
        y_prob_arr = np.asarray(y_prob_phish, dtype=float)
        # Ground truth indicator for phishing: 1 where y_true is 0, 0 where y_true is 1
        y_true_is_phish = (y_true_arr == 0).astype(int)

        # Check if both classes exist in y_true
        if len(np.unique(y_true_arr)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true_is_phish, y_prob_arr))
                pr_auc = float(average_precision_score(y_true_is_phish, y_prob_arr))
            except Exception:
                roc_auc = None
                pr_auc = None

    return {
        "positive_class": "Phishing (0)",
        "negative_class": "Legitimate (1)",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "matthews_corrcoef": round(mcc, 4),
        "specificity": round(spec, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {
            "true_phishing_tp": tp,
            "false_legitimate_fn": fn,
            "false_phishing_fp": fp,
            "true_legitimate_tn": tn,
        },
        "support": {
            "total": int(len(y_true_arr)),
            "phishing_count": int((y_true_arr == 0).sum()),
            "legitimate_count": int((y_true_arr == 1).sum()),
        },
    }
