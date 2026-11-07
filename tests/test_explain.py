import json

from serving.explain import (
    FEATURE_COLS,
    baseline_to_json,
    explain_row,
    load_baseline,
    save_baseline,
)


def _stats():
    return {
        "_global": {
            feature: (0.0, 1.0) for feature in FEATURE_COLS
        }
    }


def _row(**overrides):
    row = {feature: 0.0 for feature in FEATURE_COLS}
    row.update({"entity_type": "user", "session_duration_sec": 300.0})
    row.update(overrides)
    return row


def _stats_with_duration():
    stats = _stats()
    stats["_global"]["session_duration_sec"] = (300.0, 50.0)
    return stats


def test_impossible_travel_is_explained():
    reason, contributions = explain_row(
        _row(geo_km_per_hr=900.0), _stats_with_duration()
    )

    assert "impossible travel" in reason
    assert contributions["geo_km_per_hr"] == 18.0  # 900 / minimum std of 50


def test_cold_start_uses_interpretable_contribution():
    reason, contributions = explain_row(
        _row(is_cold_start=1), _stats_with_duration()
    )

    assert "cold start" in reason
    assert contributions["is_cold_start"] == 3.0


def test_missing_duration_is_not_flagged():
    reason, contributions = explain_row(
        _row(session_duration_sec=None), _stats_with_duration()
    )

    assert "No strong deviation" in reason
    assert "session_duration_sec" not in contributions


def test_unknown_entity_type_falls_back_to_global():
    reason, contributions = explain_row(
        _row(entity_type="unknown", geo_km_per_hr=500.0), _stats_with_duration()
    )

    assert "impossible travel" in reason
    assert contributions["geo_km_per_hr"] == 10.0  # 500 / minimum std of 50


def test_baseline_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"
    stats = _stats_with_duration()

    save_baseline(stats, path)
    loaded = load_baseline(path)

    assert loaded is not None
    assert loaded["_global"]["session_duration_sec"] == (300.0, 50.0)
    assert json.loads(path.read_text())["_global"]["hour_z"] == [0.0, 1.0]


def test_missing_baseline_returns_none(tmp_path):
    assert load_baseline(tmp_path / "does-not-exist.json") is None
