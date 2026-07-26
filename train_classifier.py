"""
Phase 4: Anomaly Classification
----------------------------------
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLS = [
    "hour_z", "resource_novelty", "n_resources_this_session", "device_novel",
    "geo_km_per_hr", "session_duration_sec", "cmd_seq_len", "failed_auth_attempts",
    "is_cold_start", "ip_failed_5min", "ip_unique_entities_5min",
]


def load_data():
    feat = pd.read_csv("data/features.csv")
    labels = pd.read_csv("data/ground_truth_labels.csv")
    return feat.merge(labels, on="session_id", how="left")


def train_classifier(df: pd.DataFrame):
    train_df = df[df["label"] == "anomaly"].copy()
    X = train_df[FEATURE_COLS].fillna(0)
    y = train_df["anomaly_type"]

    class_counts = y.value_counts()
    can_stratify = (class_counts >= 2).all()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if can_stratify else None
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",   # attack types are imbalanced among themselves too
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification report (held-out anomaly rows):")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted))

 
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nGlobal feature importances:")
    print(importances.to_string())

    return clf


def classify_all(df: pd.DataFrame, clf: RandomForestClassifier):
    """Score every session (not just training anomalies) so this can plug into the
    full pipeline: baseline model flags -> this classifier labels the flagged ones."""
    X_all = df[FEATURE_COLS].fillna(0)
    pred_type = clf.predict(X_all)
    pred_proba = clf.predict_proba(X_all).max(axis=1)

    out = df[["session_id", "entity_id", "entity_type", "label", "anomaly_type"]].copy()
    out["predicted_type"] = pred_type
    out["predicted_confidence"] = pred_proba
    return out


if __name__ == "__main__":
    import os
    import pickle

    df = load_data()
    clf = train_classifier(df)

    # Save RandomForest classifier model
    os.makedirs("models", exist_ok=True)
    with open("models/anomaly_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Saved anomaly classification model to models/")

    classified = classify_all(df, clf)
    classified.to_csv("data/classified_alerts.csv", index=False)
    print("\nSaved data/classified_alerts.csv")

    # Sanity check: how does the model treat the ambiguous insider_drift edge cases?
    edge = classified[classified["label"] == "edge_case"]
    if len(edge) > 0:
        print("\ninsider_drift (edge_case) rows -- what the classifier guesses for them:")
        print(edge[["entity_id", "predicted_type", "predicted_confidence"]].to_string(index=False))