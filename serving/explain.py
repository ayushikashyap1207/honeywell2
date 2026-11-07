"""Model-agnostic, z-score based explanations.

This module intentionally has no heavy dependencies so the FastAPI container
can import it cheaply, and so the exact same explanation logic is used by the
batch pipeline (``explainability.py``) and the streaming scorer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

FEATURE_COLS = [
    "hour_z",
    "resource_novelty",
    "n_resources_this_session",
    "device_novel",
    "geo_km_per_hr",
    "session_duration_sec",
    "cmd_seq_len",
    "failed_auth_attempts",
    "is_cold_start",
    "ip_failed_5min",
    "ip_unique_entities_5min",
]

# Binary flags: a plain z-score explodes when the normal population has almost
# no variance. Treat them as a fixed, interpretable contribution instead.
BINARY_FEATURES = {"device_novel", "is_cold_start"}
BINARY_CONTRIBUTION = 3.0

# Several "normal" features have near-zero variance in the training data
# (e.g. failed_auth_attempts is almost always 0). Dividing by that variance
# produces absurd z-scores like 2,000,000; these floors keep the attribution
# interpretable while still ranking deviations correctly.
MIN_STD = {
    "hour_z": 1.0,
    "resource_novelty": 0.1,
    "n_resources_this_session": 1.0,
    "geo_km_per_hr": 50.0,
    "session_duration_sec": 60.0,
    "cmd_seq_len": 1.0,
    "failed_auth_attempts": 1.0,
    "ip_failed_5min": 1.0,
    "ip_unique_entities_5min": 1.0,
}
MAX_Z = 99.0

# Human-readable phrasing per feature, used when it is a top contributor.
FEATURE_EXPLANATIONS = {
    "hour_z": "login at an unusual hour for this entity",
    "resource_novelty": "accessed resources never touched before",
    "n_resources_this_session": "unusually high number of resources touched in one session",
    "device_novel": "new/unrecognized device fingerprint",
    "geo_km_per_hr": "implausible geographic velocity (impossible travel)",
    "session_duration_sec": "session duration far from this entity's norm",
    "cmd_seq_len": "unusually long command sequence",
    "failed_auth_attempts": "failed authentication attempt(s)",
    "is_cold_start": "new entity with little to no history (cold start)",
    "ip_failed_5min": "many failed logins from this source IP in a short window",
    "ip_unique_entities_5min": "one source IP touching many different accounts (credential stuffing pattern)",
}


def _as_float(value):
    """Return a finite float, or ``None`` for missing/non-numeric values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _row_get(row, key):
    if hasattr(row, "get"):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def explain_row(row, stats):
    """Explain one session against a baseline distribution.

    ``stats`` is ``{entity_type: {feature: (mean, std)}}`` with a mandatory
    ``"_global"`` fallback. Returns ``(plain_language_reason, contributions)``
    where contributions maps feature name -> absolute z-score.
    """
    entity_type = _row_get(row, "entity_type")
    entity_stats = stats.get(entity_type) or stats.get("_global") or {}

    z_scores = {}
    for col in FEATURE_COLS:
        value = _as_float(_row_get(row, col))
        if value is None:
            continue

        if col in BINARY_FEATURES:
            z = BINARY_CONTRIBUTION if value >= 1 else 0.0
        elif col == "session_duration_sec" and value <= 0:
            # Missing duration in real event logs (Okta has no session field)
            # is not itself suspicious -- skip instead of flagging 0 seconds.
            z = 0.0
        else:
            mean, std = entity_stats.get(col, (0.0, 1.0))
            safe_std = max(std, MIN_STD.get(col, 1e-6))
            z = min(abs(value - mean) / safe_std, MAX_Z)

        z_scores[col] = z

    top_features = sorted(z_scores.items(), key=lambda item: -item[1])[:3]
    top_features = [(feature, z) for feature, z in top_features if z > 1.0]

    if not top_features:
        return "No strong deviation from baseline; low-confidence flag.", {}

    reason_parts = [FEATURE_EXPLANATIONS[feature] for feature, _ in top_features]
    explanation = "Flagged due to: " + " + ".join(reason_parts) + "."
    contributions = {feature: round(float(z), 2) for feature, z in top_features}
    return explanation, contributions


def baseline_to_json(stats):
    """Convert ``{etype: {feature: (mean, std)}}`` to a JSON-safe dict."""
    return {
        entity_type: {
            feature: [float(mean), float(std)]
            for feature, (mean, std) in feature_stats.items()
        }
        for entity_type, feature_stats in stats.items()
    }


def baseline_from_json(payload):
    """Inverse of :func:`baseline_to_json`."""
    return {
        entity_type: {
            feature: (float(values[0]), float(values[1]))
            for feature, values in feature_stats.items()
        }
        for entity_type, feature_stats in payload.items()
    }


def save_baseline(stats, path="models/explanation_baseline.json"):
    """Persist the baseline so streaming workers can explain live sessions."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(baseline_to_json(stats), handle, indent=2)
    return str(output)


def load_baseline(path="models/explanation_baseline.json"):
    """Load a baseline artifact, returning ``None`` when it does not exist."""
    input_path = Path(path)
    if not input_path.exists():
        return None
    with input_path.open() as handle:
        return baseline_from_json(json.load(handle))
