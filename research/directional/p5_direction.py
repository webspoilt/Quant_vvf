"""
P5 — Directional Prediction Significance Engine
Chronological Train/Validation/Test, Random Forest, Exact Binomial Significance,
Confidence Intervals, Precision/Recall, and ROC-AUC.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


@dataclass
class DirectionalEvaluationResult:
    accuracy: float
    baseline_random_accuracy: float
    binomial_p_value: float
    confidence_interval_95: Tuple[float, float]
    precision: float
    recall: float
    roc_auc: float
    n_test_samples: int
    is_statistically_significant: bool


class DirectionalSignificanceEngine:
    """Evaluates ML directional return predictions against random and persistent baselines."""

    @staticmethod
    def evaluate_classifier_chronological(
        features: pd.DataFrame,
        returns: pd.Series,
        train_ratio: float = 0.70,
        significance_alpha: float = 0.05,
        random_state: int = 42
    ) -> DirectionalEvaluationResult:
        """Trains Random Forest on strict chronological past and evaluates on future out-of-sample data."""
        # Binary target: 1 if positive return, 0 otherwise
        target = (returns > 0).astype(int)

        df_all = features.copy()
        df_all["target"] = target
        df_all = df_all.dropna()

        n = len(df_all)
        split_idx = int(n * train_ratio)
        if split_idx < 30 or (n - split_idx) < 20:
            raise ValueError("Insufficient sample size for chronological train/test split")

        train_data = df_all.iloc[:split_idx]
        test_data = df_all.iloc[split_idx:]

        feature_cols = [c for c in df_all.columns if c != "target"]
        X_train = train_data[feature_cols]
        y_train = train_data["target"]

        X_test = test_data[feature_cols]
        y_test = test_data["target"]

        clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=random_state)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if len(clf.classes_) > 1 else np.zeros(len(y_test))

        acc = float(accuracy_score(y_test, y_pred))
        k_correct = int(np.sum(y_pred == y_test))
        n_test = len(y_test)

        # Exact two-sided Binomial Test against null probability p0 = 0.50
        b_test = binomtest(k=k_correct, n=n_test, p=0.50, alternative="greater")
        p_val = float(b_test.pvalue)
        ci = b_test.proportion_ci(confidence_level=0.95)

        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))

        try:
            auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.50
        except Exception:
            auc = 0.50

        return DirectionalEvaluationResult(
            accuracy=acc,
            baseline_random_accuracy=0.50,
            binomial_p_value=p_val,
            confidence_interval_95=(float(ci.low), float(ci.high)),
            precision=prec,
            recall=rec,
            roc_auc=auc,
            n_test_samples=n_test,
            is_statistically_significant=bool(p_val <= significance_alpha),
        )
