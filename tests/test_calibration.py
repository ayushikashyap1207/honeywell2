import numpy as np
from sklearn.isotonic import IsotonicRegression

from serving.calibration import apply_calibration


def test_calibration_applies_isotonic_and_conformal_p():
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(
        [0.0, 0.5, 1.0], [0.0, 0.0, 1.0]
    )
    calibrators = {
        "sequence_lstm": {
            "isotonic": isotonic,
            "normal_scores": np.array([0.1, 0.2, 0.3]),
        }
    }

    calibrated, p_value = apply_calibration(calibrators, "sequence_lstm", 0.55)

    assert 0.0 <= calibrated <= 1.0
    assert p_value is not None
    assert 0.0 <= p_value <= 1.0


def test_calibration_passthrough_without_artifact():
    calibrated, p_value = apply_calibration({}, "sequence_lstm", 0.93)

    assert calibrated == 0.93
    assert p_value is None
