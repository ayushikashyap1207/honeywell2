"""Read-only tools the copilot can call.

Every tool returns JSON-serializable data and falls back to the exported CSV
snapshot when Postgres is unavailable, so the demo still works offline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sqlalchemy import text

from api.db import engine
from copilot.mitre import get_mitre as mitre_lookup
from copilot.retrieval import similar

ALERTS_CSV = "data/alerts_explained.csv"
LOGS_CSV = "data/access_logs_unlabeled.csv"


def _json_safe(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _row_dict(row):
    return {key: _json_safe(value) for key, value in dict(row).items()}


def _csv_rows(path, limit=5000):
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit]


def _find_csv_alert(session_id):
    for row in _csv_rows(ALERTS_CSV, limit=50000):
        if row.get("session_id") == session_id:
            return row
    return None


def get_alert(session_id):
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT s.session_id, s.entity_id, s.entity_type,
                           s.risk_score, s.calibrated_risk, s.conformal_p_value,
                           s.detection_method, s.predicted_type,
                           s.predicted_confidence, s.explanation,
                           s.explanation_details, s.top_contributing_features,
                           s.label, s.anomaly_type, s.scored_at,
                           e.ts, e.source_ip, e.geo_location,
                           e.resource_accessed, e.auth_method,
                           e.session_duration_sec
                    FROM scores s
                    LEFT JOIN sessions e ON e.session_id = s.session_id
                    WHERE s.session_id = :session_id
                    """
                ),
                {"session_id": session_id},
            ).mappings().first()
        if row:
            return _row_dict(row)
    except Exception:  # noqa: BLE001
        pass

    fallback = _find_csv_alert(session_id)
    return fallback or {"error": f"No alert found for session {session_id}"}


def get_entity_history(entity_id, limit=8):
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT session_id, ts, source_ip, geo_location,
                           resource_accessed, auth_method, session_duration_sec
                    FROM sessions
                    WHERE entity_id = :entity_id
                    ORDER BY ts DESC
                    LIMIT :limit
                    """
                ),
                {"entity_id": entity_id, "limit": limit},
            ).mappings().all()
        if rows:
            return [_row_dict(row) for row in rows]
    except Exception:  # noqa: BLE001
        pass

    rows = [
        row
        for row in _csv_rows(LOGS_CSV, limit=50000)
        if row.get("entity_id") == entity_id
    ]
    rows.sort(key=lambda item: item.get("timestamp") or "", reverse=True)
    return rows[:limit]


def search_alerts(entity_id=None, attack_type=None, min_risk=0.9, limit=10):
    try:
        clauses = ["risk_score >= :min_risk"]
        params = {"min_risk": min_risk, "limit": limit}
        if entity_id:
            clauses.append("entity_id = :entity_id")
            params["entity_id"] = entity_id
        if attack_type:
            clauses.append("predicted_type = :attack_type")
            params["attack_type"] = attack_type

        query = (
            "SELECT session_id, entity_id, entity_type, risk_score, "
            "calibrated_risk, predicted_type, explanation, label, anomaly_type "
            "FROM scores WHERE " + " AND ".join(clauses) + " "
            "ORDER BY risk_score DESC LIMIT :limit"
        )
        with engine.connect() as connection:
            rows = connection.execute(text(query), params).mappings().all()
        if rows:
            return [_row_dict(row) for row in rows]
    except Exception:  # noqa: BLE001
        pass

    rows = _csv_rows(ALERTS_CSV)
    output = []
    for row in rows:
        try:
            risk = float(row.get("risk_score") or 0)
        except ValueError:
            continue
        if risk < min_risk:
            continue
        if entity_id and row.get("entity_id") != entity_id:
            continue
        if attack_type and row.get("predicted_type") != attack_type:
            continue
        output.append(row)
    output.sort(key=lambda item: float(item.get("risk_score") or 0), reverse=True)
    return output[:limit]


def get_similar_incidents(session_id=None, query=None, k=5):
    return similar(query=query, session_id=session_id, k=k)


def get_evaluation_summary():
    path = Path("data/evaluation_report.json")
    if not path.exists():
        return {"available": False}
    with path.open() as handle:
        report = json.load(handle)
    models = report.get("models", [])
    pipeline = next(
        (item for item in models if item["name"] == "Hybrid IF+LSTM (pipeline)"), {}
    )
    best = models[0] if models else {}
    return {
        "available": True,
        "generated_at": report.get("generated_at"),
        "pipeline": pipeline,
        "best_model": best,
        "calibration": report.get("calibration"),
        "dataset": report.get("dataset"),
    }


def get_drift_report():
    path = Path("data/drift_report.json")
    if not path.exists():
        return {"available": False}
    with path.open() as handle:
        report = json.load(handle)
    return {
        "available": True,
        "overall_status": report.get("overall_status"),
        "max_feature_psi": report.get("max_feature_psi"),
        "drifted_features": report.get("drifted_features"),
        "top_features": report.get("features", [])[:5],
    }


def get_mitre(attack_type):
    return mitre_lookup(attack_type)


def get_feedback(session_id):
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT verdict, note, analyst, created_at FROM feedback "
                    "WHERE session_id = :session_id ORDER BY created_at DESC"
                ),
                {"session_id": session_id},
            ).mappings().all()
        return [_row_dict(row) for row in rows]
    except Exception:  # noqa: BLE001
        return []
