"""Evaluation lab: compare the hybrid detector against baselines.

Produces a report consumed by the React dashboard and MLflow:
    data/evaluation_report.json
    dashboard-app/src/data/evaluation.json

Baselines are trained only on normal training rows (except the supervised
upper-bound models, which are explicitly labelled as such). Metrics focus on
the top-1% analyst alert budget, plus PR/ROC curves, per-attack recall and
cold-start vs warm-entity slices.

Run:
    python evaluation.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from serving.explain import FEATURE_COLS

RANDOM_STATE = 42
TEST_SIZE = 0.3
ALERT_BUDGET = 0.01
OUTPUT = "data/evaluation_report.json"
DASHBOARD_OUTPUT = "dashboard-app/src/data/evaluation.json"


def _downsample_curve(x_values, y_values, max_points=120):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if len(x_values) <= max_points:
        indices = np.arange(len(x_values))
    else:
        indices = np.linspace(0, len(x_values) - 1, max_points).astype(int)
    return [
        [round(float(x_values[index]), 5), round(float(y_values[index]), 5)]
        for index in indices
    ]


def _top_k_metrics(y_true, scores, fraction=ALERT_BUDGET):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    k = max(1, int(len(y_true) * fraction))
    top = np.argsort(-scores)[:k]

    true_positives = int(y_true[top].sum())
    false_positives = k - true_positives
    negatives = int((y_true == 0).sum())
    positives = int(y_true.sum())

    return {
        "k": k,
        "precision_at_1": round(float(true_positives / max(k, 1)), 4),
        "recall_at_1": round(float(true_positives / max(positives, 1)), 4),
        "fpr_at_1": round(float(false_positives / max(negatives, 1)), 5),
        "true_positives": true_positives,
    }


def _per_attack_recall(frame, y_true, scores, fraction=ALERT_BUDGET):
    k = max(1, int(len(y_true) * fraction))
    top_indices = set(np.argsort(-np.asarray(scores))[:k].tolist())
    output = {}
    for attack_type in sorted(frame["anomaly_type"].dropna().unique()):
        if attack_type == "none":
            continue
        mask = frame["anomaly_type"].values == attack_type
        total = int(mask.sum())
        if total == 0:
            continue
        hits = sum(1 for index in np.where(mask)[0] if index in top_indices)
        output[str(attack_type)] = round(float(hits / total), 4)
    return output


def _slice_metrics(frame, y_true, scores, mask):
    subset_y = np.asarray(y_true)[mask]
    subset_scores = np.asarray(scores)[mask]
    if len(subset_y) == 0 or subset_y.sum() == 0:
        return {"rows": int(len(subset_y)), "anomalies": int(subset_y.sum())}
    metrics = _top_k_metrics(subset_y, subset_scores)
    metrics["rows"] = int(len(subset_y))
    metrics["anomalies"] = int(subset_y.sum())
    metrics["pr_auc"] = round(float(average_precision_score(subset_y, subset_scores)), 4)
    if 0 < subset_y.sum() < len(subset_y):
        metrics["roc_auc"] = round(float(roc_auc_score(subset_y, subset_scores)), 4)
    return metrics


def _calibration_bins(y_true, probabilities, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        mask = (probabilities >= low) & (
            probabilities <= high if index == n_bins - 1 else probabilities < high
        )
        count = int(mask.sum())
        if count == 0:
            continue
        mean_prediction = float(probabilities[mask].mean())
        fraction_positive = float(y_true[mask].mean())
        ece += (count / max(len(probabilities), 1)) * abs(
            mean_prediction - fraction_positive
        )
        bins.append(
            {
                "bin_low": round(float(low), 3),
                "bin_high": round(float(high), 3),
                "count": count,
                "mean_predicted": round(mean_prediction, 4),
                "fraction_positive": round(fraction_positive, 4),
            }
        )
    return round(float(ece), 4), bins


def _fit_unsupervised(train_normal, test_features):
    scores = {}

    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    pca.fit(train_normal)
    reconstructed = pca.inverse_transform(pca.transform(test_features))
    scores["PCA reconstruction"] = np.mean((test_features - reconstructed) ** 2, axis=1)

    sample_size = min(5000, len(train_normal))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = rng.choice(len(train_normal), size=sample_size, replace=False)
    ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    ocsvm.fit(train_normal[sample_indices])
    scores["One-Class SVM"] = -ocsvm.decision_function(test_features)

    isolation = IsolationForest(
        n_estimators=200,
        contamination="auto",
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    isolation.fit(train_normal)
    scores["Isolation Forest"] = -isolation.decision_function(test_features)

    return scores


def _fit_supervised(train_features, train_labels, test_features):
    scores = {}

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    logistic = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    logistic.fit(train_scaled, train_labels)
    scores["Logistic regression (supervised)"] = logistic.predict_proba(test_scaled)[:, 1]

    forest = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    forest.fit(train_features, train_labels)
    scores["Random forest (supervised upper bound)"] = forest.predict_proba(test_features)[:, 1]

    return scores, forest


def _load_pipeline_scores(test_frame):
    """Attach the existing hybrid detector and calibrated scores to test rows."""
    output = {}
    try:
        final = pd.read_csv("data/final_scores.csv")
        merged = test_frame[["session_id"]].merge(
            final[["session_id", "risk_score"]], on="session_id", how="left"
        )
        if merged["risk_score"].notna().all():
            output["Hybrid IF+LSTM (pipeline)"] = merged["risk_score"].fillna(0).values
    except FileNotFoundError:
        pass

    try:
        calibrated = pd.read_csv("data/calibrated_scores.csv")
        merged = test_frame[["session_id"]].merge(
            calibrated[["session_id", "calibrated_risk", "conformal_p_value"]],
            on="session_id",
            how="left",
        )
        if merged["calibrated_risk"].notna().any():
            output["Hybrid + isotonic calibration"] = (
                merged["calibrated_risk"].fillna(0).values
            )
    except FileNotFoundError:
        pass

    return output


def main():
    features = pd.read_csv("data/features.csv")
    labels = pd.read_csv("data/ground_truth_labels.csv")
    frame = features.merge(labels, on="session_id", how="left")
    frame = frame[frame["label"].isin(["normal", "anomaly"])].reset_index(drop=True)

    train_frame, test_frame = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame["label"],
    )

    train_labels = (train_frame["label"] == "anomaly").astype(int).values
    test_labels = (test_frame["label"] == "anomaly").astype(int).values

    train_features = train_frame[FEATURE_COLS].fillna(0).values
    test_features = test_frame[FEATURE_COLS].fillna(0).values
    train_normal = train_features[train_labels == 0]

    scores = _fit_unsupervised(train_normal, test_features)
    supervised_scores, forest = _fit_supervised(
        train_features, train_labels, test_features
    )
    scores.update(supervised_scores)
    scores.update(_load_pipeline_scores(test_frame))

    models = []
    curves = {}
    per_attack = {}
    cold_start = {}

    is_cold_start = test_frame["is_cold_start"].fillna(0).astype(int).values == 1

    for name, score_values in scores.items():
        score_values = np.nan_to_num(
            np.asarray(score_values, dtype=float), copy=True
        )

        pr_auc = float(average_precision_score(test_labels, score_values))
        roc_auc = float(roc_auc_score(test_labels, score_values))
        top_metrics = _top_k_metrics(test_labels, score_values)

        precision, recall, _ = precision_recall_curve(test_labels, score_values)
        fpr, tpr, _ = roc_curve(test_labels, score_values)
        curves[name] = {
            "pr": _downsample_curve(recall[::-1], precision[::-1]),
            "roc": _downsample_curve(fpr, tpr),
        }

        models.append(
            {
                "name": name,
                "kind": (
                    "supervised"
                    if "supervised" in name
                    else "pipeline"
                    if "Hybrid" in name
                    else "unsupervised"
                ),
                "pr_auc": round(pr_auc, 4),
                "roc_auc": round(roc_auc, 4),
                **top_metrics,
            }
        )
        per_attack[name] = _per_attack_recall(
            test_frame, test_labels, score_values
        )
        cold_start[name] = {
            "cold_start": _slice_metrics(
                test_frame, test_labels, score_values, is_cold_start
            ),
            "warm": _slice_metrics(
                test_frame, test_labels, score_values, ~is_cold_start
            ),
        }

    models.sort(key=lambda item: item["pr_auc"], reverse=True)

    calibration_summary = None
    calibration_model = scores.get("Hybrid + isotonic calibration")
    if calibration_model is not None:
        probabilities = np.clip(np.asarray(calibration_model, dtype=float), 0, 1)
        ece, bins = _calibration_bins(test_labels, probabilities)
        calibration_summary = {
            "ece": ece,
            "brier": round(float(brier_score_loss(test_labels, probabilities)), 4),
            "bins": bins,
        }

    feature_importance = [
        {"feature": feature, "importance": round(float(importance), 5)}
        for feature, importance in sorted(
            zip(FEATURE_COLS, forest.feature_importances_),
            key=lambda item: -item[1],
        )
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "rows_scored": int(len(frame)),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "test_anomalies": int(test_labels.sum()),
            "alert_budget_fraction": ALERT_BUDGET,
            "cold_start_test_rows": int(is_cold_start.sum()),
        },
        "models": models,
        "curves": curves,
        "per_attack_recall": per_attack,
        "cold_start": cold_start,
        "calibration": calibration_summary,
        "feature_importance": feature_importance,
    }

    with open(OUTPUT, "w") as handle:
        json.dump(report, handle, indent=2)
    with open(DASHBOARD_OUTPUT, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved {OUTPUT} and {DASHBOARD_OUTPUT}")
    print("\nModel comparison (test set, top-1% alert budget):")
    table = pd.DataFrame(models)[
        ["name", "pr_auc", "roc_auc", "precision_at_1", "recall_at_1", "fpr_at_1"]
    ]
    print(table.to_string(index=False))
    if calibration_summary:
        print(
            f"\nCalibration: ECE={calibration_summary['ece']} "
            f"Brier={calibration_summary['brier']}"
        )


if __name__ == "__main__":
    main()
