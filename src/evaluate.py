"""Evaluation harness: leave-one-group-out scoring and the fixed-threshold baseline.

The split is by driver (or trip, where only one driver exists). Splitting windows at
random would place overlapping windows from the same trip on both sides of the split,
so the score would measure memorisation of a trip rather than transfer to a new driver.
Saeb et al. (2017) measured that gap directly in a comparable setting: record-wise
cross-validation reported roughly half the error of subject-wise validation on the same
data.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut

FEATURE_EXCLUDE = {"label", "group", "t_start"}


def feature_columns(table: pd.DataFrame) -> List[str]:
    return [c for c in table.columns if c not in FEATURE_EXCLUDE]


def default_model() -> HistGradientBoostingClassifier:
    """Gradient-boosted trees: handles heterogeneous feature scales without tuning and
    stays small enough to meet the on-device size budget."""
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42)


class ThresholdBaseline:
    """The incumbent this project has to beat: flag a window as aggressive when its peak
    magnitude jerk exceeds a fixed limit, otherwise call it normal.

    This is the logic a threshold-based telematics dongle applies. The threshold is
    chosen on the training folds only, by sweeping candidate values and keeping the one
    with the best macro-F1, so the baseline is given every fair advantage.
    """

    def __init__(self, feature: str = "mag_jerk_rms", positive_label: str = "aggressive"):
        self.feature = feature
        self.positive_label = positive_label
        self.threshold_: Optional[float] = None
        self.negative_label_: Optional[str] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ThresholdBaseline":
        labels, counts = np.unique(y, return_counts=True)
        others = [l for l in labels if l != self.positive_label]
        self.negative_label_ = (
            others[int(np.argmax([counts[list(labels).index(o)] for o in others]))]
            if others
            else self.positive_label
        )
        values = X[self.feature].to_numpy(dtype=float)
        candidates = np.percentile(values, np.arange(50, 100, 2))
        best, best_score = candidates[0], -1.0
        for c in candidates:
            pred = np.where(values >= c, self.positive_label, self.negative_label_)
            score = f1_score(y, pred, average="macro", zero_division=0)
            if score > best_score:
                best, best_score = c, score
        self.threshold_ = float(best)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.feature].to_numpy(dtype=float)
        return np.where(values >= self.threshold_, self.positive_label, self.negative_label_)


def leave_one_group_out(
    table: pd.DataFrame,
    model,
    label_column: str = "label",
    group_column: str = "group",
) -> Dict[str, object]:
    """Run leave-one-group-out and return per-fold and pooled metrics."""
    features = feature_columns(table)
    X = table[features]
    y = table[label_column].to_numpy()
    groups = table[group_column].to_numpy()

    folds = []
    all_true, all_pred = [], []
    splitter = LeaveOneGroupOut()
    for train_idx, test_idx in splitter.split(X, y, groups):
        held_out = groups[test_idx][0]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2:
            continue
        est = clone(model) if hasattr(model, "get_params") else model.__class__()
        est.fit(X_train, y_train)
        pred = est.predict(X_test)
        folds.append(
            {
                "held_out": held_out,
                "n_test": len(test_idx),
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
            }
        )
        all_true.append(y_test)
        all_pred.append(pred)

    if not folds:
        return {"folds": pd.DataFrame(), "pooled": {}, "confusion": None}

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "folds": pd.DataFrame(folds),
        "pooled": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        },
        "confusion": pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels
        ),
    }


def compare_models(table: pd.DataFrame) -> pd.DataFrame:
    """Model, threshold baseline and majority-class floor under the same splits."""
    rows = []
    for name, model in [
        ("Gradient-boosted trees", default_model()),
        ("Fixed-threshold baseline", ThresholdBaseline()),
        ("Majority class", DummyClassifier(strategy="most_frequent")),
    ]:
        result = leave_one_group_out(table, model)
        if not result["pooled"]:
            continue
        folds = result["folds"]
        rows.append(
            {
                "Model": name,
                "Pooled accuracy": result["pooled"]["accuracy"],
                "Pooled macro-F1": result["pooled"]["macro_f1"],
                "Mean fold macro-F1": folds["macro_f1"].mean(),
                "Fold macro-F1 std": folds["macro_f1"].std(),
                "Worst fold": folds["macro_f1"].min(),
            }
        )
    return pd.DataFrame(rows)
