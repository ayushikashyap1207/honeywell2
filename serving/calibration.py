"""Calibration + conformal uncertainty for detector risk scores.

The two detectors produce percentile ranks within their own populations. Those
ranks are not probabilities, and a 0.99 from the cold-start detector is not
equivalent to a 0.99 from the LSTM. This module maps each detector's rank to a
calibrated probability using isotonic regression fitted on labeled validation
rows, and computes a conformal p-value from the normal calibration scores.

Drop-in behaviour: if no calibrator artifact exists, scores pass through
unchanged and the p-value is ``None``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

DEFAULT_CALIBRATOR_PATH = "models/calibrators.pkl"
CONFORMAL_ALPHA = 0.05


def apply_calibration(calibrators, detector_name, risk_score):
    """Return ``(calibrated_risk, conformal_p_value)`` for one score."""
    risk_score = float(risk_score)
    entry = (calibrators or {}).get(detector_name)
    if not entry:
        return risk_score, None

    isotonic = entry.get("isotonic")
    if isotonic is not None:
        calibrated = float(isotonic.predict([risk_score])[0])
    else:
        calibrated = risk_score
    calibrated = min(max(calibrated, 0.0), 1.0)

    normal_scores = entry.get("normal_scores")
    if normal_scores is None or len(normal_scores) == 0:
        return calibrated, None

    scores = np.asarray(normal_scores, dtype=float)
    # Conformal p-value for the null hypothesis "this session is normal".
    # Lower p = more anomalous.
    p_value = (1.0 + float(np.sum(scores >= risk_score))) / (len(scores) + 1.0)
    return calibrated, p_value


def load_calibrators(path=DEFAULT_CALIBRATOR_PATH):
    artifact = Path(path)
    if not artifact.exists():
        return {}
    try:
        with artifact.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - never crash the scorer on a bad artifact
        print(f"WARNING: could not load calibrators ({exc}); using raw risk scores")
        return {}
