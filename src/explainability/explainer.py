"""SHAP TreeExplainer wrapper module for Random Forest and XGBoost.

Guarantees:
1. All explanations explicitly target Class 0 (Phishing) as the positive threat class.
2. Slices 3D Random Forest explanations to Class 0 probability space.
3. Inverts 2D XGBoost raw margin log-odds so positive values increase Phishing likelihood.
4. Generates standardized shap.Explanation objects and feature ranking tables.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from src.features.feature_config import FEATURE_NAMES

logger = logging.getLogger(__name__)


class PhishingTreeExplainer:
    """Standardized SHAP explainer wrapper targeting the Phishing class (Class 0)."""

    def __init__(
        self,
        model: Any,
        model_name: str,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray | pd.DataFrame] = None,
    ) -> None:
        """Initialize TreeExplainer with class-0 orientation.

        Args:
            model: Fitted RandomForestClassifier or XGBClassifier.
            model_name: "Random Forest" or "XGBoost".
            feature_names: List of 68 feature names.
            background_data: Optional background matrix for interventional explanation.
        """
        self.model = model
        self.model_name = model_name
        self.feature_names = feature_names or FEATURE_NAMES

        # Initialize underlying SHAP TreeExplainer
        if background_data is not None:
            bg_arr = np.asarray(background_data, dtype=np.float32)
            self.explainer = shap.TreeExplainer(model, data=bg_arr)
        else:
            self.explainer = shap.TreeExplainer(model)

    def explain(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> shap.Explanation:
        """Compute SHAP explanation object strictly oriented toward Phishing (Class 0).

        Args:
            X: Input feature matrix (N, 68).

        Returns:
            shap.Explanation object where positive values increase Phishing prediction.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        raw_exp = self.explainer(X_arr)

        if "Random Forest" in self.model_name:
            # Random Forest returns (N, D, 2)
            # Class 0 = Phishing, Class 1 = Legitimate
            values_phish = raw_exp.values[:, :, 0]
            base_phish = raw_exp.base_values[:, 0]
            data_arr = raw_exp.data

            phish_exp = shap.Explanation(
                values=values_phish,
                base_values=base_phish,
                data=data_arr,
                feature_names=self.feature_names,
            )
        elif "XGBoost" in self.model_name:
            # XGBoost binary classification returns margin for Class 1 (Legitimate).
            # To explain Phishing (Class 0), negate margin: log(P(0)/P(1)) = -log(P(1)/P(0)).
            # Positive SHAP value then indicates increased log-odds of Phishing.
            values_phish = -1.0 * raw_exp.values
            base_phish = -1.0 * raw_exp.base_values
            data_arr = raw_exp.data

            phish_exp = shap.Explanation(
                values=values_phish,
                base_values=base_phish,
                data=data_arr,
                feature_names=self.feature_names,
            )
        else:
            raise ValueError(f"Unsupported model type for PhishingTreeExplainer: {self.model_name}")

        return phish_exp

    def get_feature_importance_table(
        self,
        explanation: shap.Explanation,
        top_k: int = 20,
    ) -> pd.DataFrame:
        """Calculate global feature importance rankings from SHAP values.

        Args:
            explanation: Standardized class-0 Explanation object.
            top_k: Number of top features to return.

        Returns:
            DataFrame with columns: rank, feature, mean_abs_shap, mean_shap, direction.
        """
        vals = explanation.values
        data = explanation.data
        mean_abs = np.mean(np.abs(vals), axis=0)
        mean_signed = np.mean(vals, axis=0)

        # Correlation between feature value and SHAP value to capture general effect direction
        corrs = []
        for j in range(len(self.feature_names)):
            feat_col = data[:, j]
            shap_col = vals[:, j]
            # Avoid divide-by-zero on invariant features
            if np.std(feat_col) > 1e-6 and np.std(shap_col) > 1e-6:
                c = np.corrcoef(feat_col, shap_col)[0, 1]
            else:
                c = 0.0
            corrs.append(round(float(c), 3))

        df_rank = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.round(mean_abs, 6),
            "mean_shap": np.round(mean_signed, 6),
            "correlation_with_shap": corrs,
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        df_rank["rank"] = df_rank.index + 1

        # Classify general direction
        def get_direction(row):
            c = row["correlation_with_shap"]
            if c > 0.15:
                return "Higher -> Phishing"
            elif c < -0.15:
                return "Higher -> Legitimate"
            else:
                return "Non-linear / Complex"

        df_rank["effect_direction"] = df_rank.apply(get_direction, axis=1)

        cols = ["rank", "feature", "mean_abs_shap", "mean_shap", "correlation_with_shap", "effect_direction"]
        return df_rank[cols].head(top_k)

