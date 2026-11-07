"""Lightweight model/data drift monitoring.

Compares an older reference window with the most recent window using:
  * PSI  — population stability index (0.1 warning, 0.25 significant)
  * KS   — two-sample Kolmogorov-Smirnov statistic + p-value

This mirrors what Evidently/WhyLabs report, without adding a heavy dependency.
The output is written for both the API and the dashboard.

Run:
    python -m monitoring.drift
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from serving.explain import FEATURE_COLS

FEATURES_CSV = "data/features.csv"
LOGS_CSV = "data/access_logs_unlabeled.csv"
FINAL_SCORES = "data/final_scores.csv"
OUTPUT = "data/drift_report.json"
DASHBOARD_OUTPUT = "dashboard-app/src/data/drift.json"

PSI_WARNING = 0.1
PSI_SIGNIFICANT = 0.25


def population_stability_index(expected, actual, bins=10):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    breakpoints = np.quantile(expected, np.linspace(0.0, 1.0, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _status(psi):
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_WARNING:
        return "warning"
    return "stable"


def _feature_drift(expected, actual, feature):
    psi = population_stability_index(expected[feature].values, actual[feature].values)
    try:
        ks_stat, ks_p = ks_2samp(
            expected[feature].dropna().values,
            actual[feature].dropna().values,
        )
    except Exception:  # noqa: BLE001
        ks_stat, ks_p = 0.0, 1.0
    return {
        "feature": feature,
        "psi": round(float(psi), 4),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_p_value": round(float(ks_p), 6),
        "status": _status(psi),
    }


def _split_by_time(frame, reference_fraction=0.7):
    if "timestamp" in frame.columns and frame["timestamp"].notna().any():
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        split = max(1, int(len(frame) * reference_fraction))
        return frame.iloc[:split], frame.iloc[split:]
    split = max(1, int(len(frame) * reference_fraction))
    return frame.iloc[:split], frame.iloc[split:]


def main():
    parser = argparse.ArgumentParser(description="Generate a data/score drift report.")
    parser.add_argument("--reference-fraction", type=float, default=0.7)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()

    features = pd.read_csv(FEATURES_CSV)
    try:
        logs = pd.read_csv(LOGS_CSV, usecols=["session_id", "timestamp"])
        logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")
        frame = features.merge(logs, on="session_id", how="left")
    except Exception:  # noqa: BLE001
        frame = features.copy()
        frame["timestamp"] = pd.NaT

    reference, current = _split_by_time(frame, args.reference_fraction)

    feature_reports = [
        _feature_drift(reference, current, feature) for feature in FEATURE_COLS
    ]

    score_report = None
    try:
        final = pd.read_csv(FINAL_SCORES)
        score_frame = final.merge(
            frame[["session_id", "timestamp"]], on="session_id", how="left"
        )
        score_reference, score_current = _split_by_time(
            score_frame, args.reference_fraction
        )
        score_report = _feature_drift(
            score_reference, score_current, "risk_score"
        )
        score_report["feature"] = "risk_score"
    except FileNotFoundError:
        pass

    significant = [item for item in feature_reports if item["status"] == "significant"]
    warnings = [item for item in feature_reports if item["status"] == "warning"]
    max_psi = max((item["psi"] for item in feature_reports), default=0.0)
    overall = "significant" if significant else "warning" if warnings else "stable"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "reference_fraction": args.reference_fraction,
        "overall_status": overall,
        "max_feature_psi": round(float(max_psi), 4),
        "drifted_features": len(significant),
        "warning_features": len(warnings),
        "features": sorted(feature_reports, key=lambda item: -item["psi"]),
        "score": score_report,
        "thresholds": {
            "warning": PSI_WARNING,
            "significant": PSI_SIGNIFICANT,
        },
    }

    with open(OUTPUT, "w") as handle:
        json.dump(report, handle, indent=2)
    with open(DASHBOARD_OUTPUT, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved {OUTPUT} and {DASHBOARD_OUTPUT}")
    print(
        f"Overall: {overall} | max PSI: {max_psi:.4f} | "
        f"{len(significant)} significant, {len(warnings)} warning"
    )
    for item in report["features"][:5]:
        print(
            f"  {item['feature']:<28} PSI={item['psi']:.4f} "
            f"KS={item['ks_statistic']:.4f} [{item['status']}]"
        )

    if args.fail_on_drift and overall == "significant":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
