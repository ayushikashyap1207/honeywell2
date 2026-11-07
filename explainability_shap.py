"""Global + local feature attribution for the attack classifier.

If SHAP is installed, uses TreeExplainer. Otherwise falls back to the
classifier's built-in impurity importance and the z-score attributions already
stored in the alert output, so the report is always generated.

Outputs:
    data/shap_global.json
    data/shap_local.json
    dashboard-app/src/data/shap.json

Run:
    python explainability_shap.py
"""

from __future__ import annotations

import ast
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from serving.explain import FEATURE_COLS

CLASSIFIER_PATH = "models/anomaly_classifier.pkl"
FEATURES_CSV = "data/features.csv"
LABELS_CSV = "data/ground_truth_labels.csv"
ALERTS_CSV = "data/alerts_explained.csv"
GLOBAL_OUTPUT = "data/shap_global.json"
LOCAL_OUTPUT = "data/shap_local.json"
DASHBOARD_OUTPUT = "dashboard-app/src/data/shap.json"
SAMPLE_SIZE = 1000


def _load_classifier():
    with open(CLASSIFIER_PATH, "rb") as handle:
        return pickle.load(handle)


def _global_importance_shap(classifier, sample):
    import shap  # imported lazily so the fallback works without the dependency

    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        importance = np.mean([np.abs(value).mean(axis=0) for value in values], axis=0)
    else:
        array = np.asarray(values)
        if array.ndim == 3:
            array = np.abs(array).mean(axis=2)
        importance = np.abs(array).mean(axis=0)
    return importance, "shap.TreeExplainer (mean |SHAP| across classes)"


def _global_importance_fallback(classifier):
    return np.asarray(classifier.feature_importances_, dtype=float), (
        "RandomForest impurity importance (SHAP not installed)"
    )


def _parse_contributions(value):
    if isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def main():
    features = pd.read_csv(FEATURES_CSV)
    labels = pd.read_csv(LABELS_CSV)
    frame = features.merge(labels, on="session_id", how="left")
    anomaly_frame = frame[frame["label"] == "anomaly"]
    sample = anomaly_frame[FEATURE_COLS].fillna(0)
    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(SAMPLE_SIZE, random_state=42)

    classifier = _load_classifier()
    try:
        importance, method = _global_importance_shap(classifier, sample.values)
    except Exception as exc:  # noqa: BLE001
        importance, method = _global_importance_fallback(classifier)
        print(f"SHAP unavailable ({exc}); using the fallback importance.")

    feature_rows = [
        {"feature": feature, "importance": round(float(value), 6)}
        for feature, value in sorted(
            zip(FEATURE_COLS, importance), key=lambda item: -item[1]
        )
    ]

    global_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "sample_rows": int(len(sample)),
        "features": feature_rows,
    }
    with open(GLOBAL_OUTPUT, "w") as handle:
        json.dump(global_report, handle, indent=2)

    local_rows = []
    try:
        alerts = pd.read_csv(ALERTS_CSV).head(10)
        for _, row in alerts.iterrows():
            contributions = _parse_contributions(row.get("top_contributing_features"))
            local_rows.append(
                {
                    "session_id": row.get("session_id"),
                    "entity_id": row.get("entity_id"),
                    "predicted_type": row.get("predicted_type"),
                    "risk_score": float(row.get("risk_score", 0) or 0),
                    "contributions": contributions,
                    "method": "z-score deviation (batch explainability)",
                }
            )
    except FileNotFoundError:
        pass

    local_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": local_rows,
    }
    with open(LOCAL_OUTPUT, "w") as handle:
        json.dump(local_report, handle, indent=2)
    with open(DASHBOARD_OUTPUT, "w") as handle:
        json.dump(global_report, handle, indent=2)

    print(f"Saved {GLOBAL_OUTPUT}, {LOCAL_OUTPUT}, {DASHBOARD_OUTPUT}")
    print(f"Attribution method: {method}")
    for item in feature_rows[:5]:
        print(f"  {item['feature']:<28} {item['importance']}")


if __name__ == "__main__":
    main()
