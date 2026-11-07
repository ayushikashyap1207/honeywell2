"""Fit probability calibration + conformal uncertainty for both detectors.

Run after ``combine_scores.py`` and before ``train_classifier.py`` /
``explainability.py``:

    python -m calibration.calibrate

Outputs:
    models/calibrators.pkl          # streaming scorer artifact
    data/calibrated_scores.csv      # session_id, calibrated_risk, conformal_p_value
    data/calibration_report.json    # Brier / ECE / reliability curves
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

FINAL_SCORES = "data/final_scores.csv"
OUTPUT_ARTIFACT = "models/calibrators.pkl"
OUTPUT_SCORES = "data/calibrated_scores.csv"
OUTPUT_REPORT = "data/calibration_report.json"

DETECTORS = ("sequence_lstm", "cold_start_isolation_forest")
CONFORMAL_ALPHA = 0.05


def _reliability_curve(y_true, probabilities, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    total = max(len(probabilities), 1)

    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_prediction = float(probabilities[mask].mean())
        fraction_positive = float(y_true[mask].mean())
        ece += (count / total) * abs(mean_prediction - fraction_positive)
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


def _fit_isotonic(risk_scores, labels):
    if len(set(labels)) < 2:
        return None
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(risk_scores, labels)
    return model


def _predict_isotonic(model, risk_scores):
    if model is None:
        return np.clip(np.asarray(risk_scores, dtype=float), 0.0, 1.0)
    return np.clip(model.predict(risk_scores), 0.0, 1.0)


def _conformal_p_values(normal_scores, risk_scores, chunk_size=2048):
    normal_scores = np.asarray(normal_scores, dtype=float)
    risk_scores = np.asarray(risk_scores, dtype=float)
    if len(normal_scores) == 0:
        return np.full(len(risk_scores), np.nan)

    output = np.empty(len(risk_scores), dtype=float)
    denominator = len(normal_scores) + 1.0
    for start in range(0, len(risk_scores), chunk_size):
        block = risk_scores[start : start + chunk_size]
        counts = (normal_scores[None, :] >= block[:, None]).sum(axis=1)
        output[start : start + chunk_size] = (1.0 + counts) / denominator
    return output


def calibrate_detector(frame, detector):
    subset = frame[
        (frame["detection_method"] == detector)
        & (frame["label"].isin(["normal", "anomaly"]))
    ].dropna(subset=["risk_score"])

    if subset.empty or subset["label"].nunique() < 2:
        return None, {
            "detector": detector,
            "status": "skipped",
            "reason": "not enough labeled rows or one class only",
            "rows": int(len(subset)),
        }

    labels = (subset["label"] == "anomaly").astype(int).values
    risks = subset["risk_score"].astype(float).values

    stratify = labels if np.bincount(labels).min() >= 2 else None
    train_risk, test_risk, train_y, test_y = train_test_split(
        risks, labels, test_size=0.3, random_state=42, stratify=stratify
    )

    train_isotonic = _fit_isotonic(train_risk, train_y)
    train_normal_scores = train_risk[train_y == 0]
    test_probabilities = _predict_isotonic(train_isotonic, test_risk)

    raw_brier = float(brier_score_loss(test_y, np.clip(test_risk, 0, 1)))
    calibrated_brier = float(brier_score_loss(test_y, test_probabilities))
    raw_ece, raw_curve = _reliability_curve(test_y, np.clip(test_risk, 0, 1))
    calibrated_ece, calibrated_curve = _reliability_curve(test_y, test_probabilities)

    test_p_values = _conformal_p_values(train_normal_scores, test_risk)
    flagged = test_p_values <= CONFORMAL_ALPHA
    anomaly_mask = test_y == 1
    normal_mask = ~anomaly_mask
    conformal_precision = (
        float((flagged & anomaly_mask).sum() / max(flagged.sum(), 1))
        if flagged.any()
        else 0.0
    )
    conformal_recall = (
        float((flagged & anomaly_mask).sum() / max(anomaly_mask.sum(), 1))
        if anomaly_mask.any()
        else 0.0
    )
    normal_coverage = (
        float((~flagged[normal_mask]).mean()) if normal_mask.any() else None
    )

    # Refit on all labeled rows for the production artifact.
    final_isotonic = _fit_isotonic(risks, labels)
    final_normal_scores = risks[labels == 0]
    final_probabilities = _predict_isotonic(final_isotonic, risks)

    report = {
        "detector": detector,
        "status": "ok",
        "rows": int(len(subset)),
        "anomalies": int(labels.sum()),
        "raw_brier": round(raw_brier, 4),
        "calibrated_brier": round(calibrated_brier, 4),
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "conformal_alpha": CONFORMAL_ALPHA,
        "conformal_precision": round(conformal_precision, 4),
        "conformal_recall": round(conformal_recall, 4),
        "normal_coverage": round(normal_coverage, 4)
        if normal_coverage is not None
        else None,
        "reliability_raw": raw_curve,
        "reliability_calibrated": calibrated_curve,
    }

    artifact = {
        "isotonic": final_isotonic,
        "normal_scores": final_normal_scores,
    }
    return artifact, report, final_probabilities


def main():
    frame = pd.read_csv(FINAL_SCORES)
    calibrators = {}
    reports = []

    calibrated_rows = []
    for detector in DETECTORS:
        result = calibrate_detector(frame, detector)
        if result[0] is None:
            _, report = result
            reports.append(report)
            continue

        artifact, report, probabilities = result
        calibrators[detector] = artifact
        reports.append(report)

        subset = frame[
            (frame["detection_method"] == detector)
            & (frame["risk_score"].notna())
        ].copy()
        # Recompute on the exact subset order used for final fitting.
        subset = subset[
            subset["label"].isin(["normal", "anomaly"])
        ].dropna(subset=["risk_score"])
        risks = subset["risk_score"].astype(float).values
        p_values = _conformal_p_values(artifact["normal_scores"], risks)
        subset["calibrated_risk"] = _predict_isotonic(artifact["isotonic"], risks)
        subset["conformal_p_value"] = p_values
        calibrated_rows.append(subset[["session_id", "calibrated_risk", "conformal_p_value"]])

    if calibrated_rows:
        output = pd.concat(calibrated_rows, ignore_index=True)
    else:
        output = pd.DataFrame(
            columns=["session_id", "calibrated_risk", "conformal_p_value"]
        )

    Path("models").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ARTIFACT, "wb") as handle:
        pickle.dump(calibrators, handle)
    output.to_csv(OUTPUT_SCORES, index=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conformal_alpha": CONFORMAL_ALPHA,
        "detectors": reports,
        "rows_calibrated": int(len(output)),
    }
    with open(OUTPUT_REPORT, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved {OUTPUT_ARTIFACT}")
    print(f"Saved {OUTPUT_SCORES} ({len(output)} rows)")
    print(f"Saved {OUTPUT_REPORT}")
    for detector_report in reports:
        if detector_report["status"] == "ok":
            print(
                f"  {detector_report['detector']}: "
                f"Brier {detector_report['raw_brier']} -> {detector_report['calibrated_brier']}, "
                f"ECE {detector_report['raw_ece']} -> {detector_report['calibrated_ece']}"
            )
        else:
            print(f"  {detector_report['detector']}: skipped")


if __name__ == "__main__":
    main()
