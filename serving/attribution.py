"""Counterfactual explanations for live alerts.

For the top z-score contributors, replace a feature with its learned normal
value and re-score the session. The change that lowers the detector score the
most becomes the counterfactual returned to the analyst.

This is a real model interrogation (not an LLM guess), and works for both the
cold-start Isolation Forest and the LSTM window model.
"""

from __future__ import annotations

from serving.explain import BINARY_FEATURES, FEATURE_EXPLANATIONS, explain_row
from serving.models import score_isolation_forest, score_lstm


def _normal_value(stats, entity_type, feature):
    entity_stats = stats.get(entity_type) or stats.get("_global") or {}
    mean, _ = entity_stats.get(feature, (0.0, 1.0))
    return float(mean)


def _candidate_features(contributions):
    return [feature for feature in contributions if feature]


def _counterfactual_cold_start(bundle, feature_row, baseline_stats):
    current_raw = score_isolation_forest(bundle, feature_row)
    _, contributions = explain_row(feature_row, baseline_stats)
    if not contributions:
        return None

    entity_type = feature_row.get("entity_type")
    best = None
    for feature in _candidate_features(contributions):
        replacement = (
            0.0
            if feature in BINARY_FEATURES
            else _normal_value(baseline_stats, entity_type, feature)
        )
        candidate = dict(feature_row)
        candidate[feature] = replacement
        candidate_raw = score_isolation_forest(bundle, candidate)
        if candidate_raw < current_raw:
            if best is None or candidate_raw < best["counterfactual_raw_score"]:
                best = {
                    "feature": feature,
                    "baseline_value": round(float(replacement), 4),
                    "current_raw_score": round(float(current_raw), 6),
                    "counterfactual_raw_score": round(float(candidate_raw), 6),
                }

    if best is None:
        return None
    best["text"] = (
        "If "
        + FEATURE_EXPLANATIONS[best["feature"]]
        + f" were closer to normal (≈{best['baseline_value']}), the detector's "
        + f"raw anomaly score would fall from {best['current_raw_score']:.3f} "
        + f"to {best['counterfactual_raw_score']:.3f}."
    )
    return best


def _counterfactual_sequence(bundle, feature_row, baseline_stats):
    window = feature_row.get("window") or []
    if len(window) < 5:
        return _counterfactual_cold_start(bundle, feature_row, baseline_stats)

    current_raw = score_lstm(bundle, window)
    _, contributions = explain_row(feature_row, baseline_stats)
    if not contributions:
        return None

    entity_type = feature_row.get("entity_type")
    best = None
    for feature in _candidate_features(contributions):
        if feature == "is_cold_start":
            continue
        replacement = _normal_value(baseline_stats, entity_type, feature)
        candidate_window = [dict(session) for session in window]
        for session in candidate_window:
            session[feature] = replacement
        try:
            candidate_raw = score_lstm(bundle, candidate_window)
        except Exception:  # noqa: BLE001 - a single bad counterfactual must not break scoring
            continue
        if candidate_raw < current_raw:
            if best is None or candidate_raw < best["counterfactual_raw_score"]:
                best = {
                    "feature": feature,
                    "baseline_value": round(float(replacement), 4),
                    "current_raw_score": round(float(current_raw), 6),
                    "counterfactual_raw_score": round(float(candidate_raw), 6),
                }

    if best is None:
        return None
    best["text"] = (
        "If "
        + FEATURE_EXPLANATIONS[best["feature"]]
        + f" were closer to normal (≈{best['baseline_value']}) across the recent "
        + f"window, the reconstruction error would fall from "
        + f"{best['current_raw_score']:.3f} to {best['counterfactual_raw_score']:.3f}."
    )
    return best


def build_counterfactual(bundle, feature_row, detection_method, baseline_stats):
    """Return a counterfactual dict, or ``None`` when the model has no evidence."""
    if not baseline_stats:
        return None
    if detection_method == "sequence_lstm":
        return _counterfactual_sequence(bundle, feature_row, baseline_stats)
    return _counterfactual_cold_start(bundle, feature_row, baseline_stats)
