"""FastAPI backend for the analyst dashboard.

Run locally (from the project root):
    uvicorn api.main:app --reload --port 8000

Environment variables (see .env.example):
    DATABASE_URL, REDIS_URL, CORS_ORIGINS
"""

import json
import math
import os

from fastapi import FastAPI, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from api.analytics_routes import router as analytics_router
from api.copilot_routes import router as copilot_router
from api.db import REDIS_URL, engine

CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:8080"
    ).split(",")
    if origin.strip()
]
ALERT_BUDGET_PCT = float(os.environ.get("ALERT_BUDGET_PCT", "0.99"))

import redis.asyncio as aioredis

app = FastAPI(title="Behavioural Anomaly Detection API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(analytics_router)
app.include_router(copilot_router)


def _finite(value):
    """Return a JSON-safe float or ``None`` (Postgres can hold NaN)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _parse_features(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


@app.get("/api/health")
def health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok", "database": True}
    except Exception as exc:  # noqa: BLE001
        return {"status": "degraded", "database": False, "error": str(exc)}


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("live_scores")
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            await websocket.send_text(message["data"])
    except Exception:  # noqa: BLE001 - client disconnect / Redis restart
        pass
    finally:
        await pubsub.unsubscribe("live_scores")
        await redis_client.aclose()


@app.get("/api/summary")
def summary():
    with engine.connect() as connection:
        total = connection.execute(text("SELECT count(*) FROM scores")).scalar() or 0
        if total == 0:
            return {
                "total_sessions": 0,
                "alert_budget": 0,
                "precision_in_queue": 0,
                "true_anomalies_total": 0,
                "entities_monitored": 0,
                "feedback_pending": 0,
            }

        entities = connection.execute(
            text("SELECT count(DISTINCT entity_id) FROM scores")
        ).scalar() or 0
        threshold = connection.execute(
            text(
                """
                SELECT percentile_cont(:pct) WITHIN GROUP (ORDER BY risk_score)
                FROM scores
                """
            ),
            {"pct": ALERT_BUDGET_PCT},
        ).scalar()

        if threshold is None:
            threshold = 1.0

        budget = connection.execute(
            text("SELECT count(*) FROM scores WHERE risk_score >= :threshold"),
            {"threshold": threshold},
        ).scalar() or 0
        # Ignore unlabeled live rows instead of counting them as false positives.
        precision = connection.execute(
            text(
                """
                SELECT avg(CASE WHEN label = 'anomaly' THEN 1.0
                                WHEN label = 'normal' THEN 0.0 END)
                FROM scores
                WHERE risk_score >= :threshold
                """
            ),
            {"threshold": threshold},
        ).scalar()
        true_anomalies = connection.execute(
            text("SELECT count(*) FROM scores WHERE label = 'anomaly'")
        ).scalar() or 0
        try:
            feedback_pending = connection.execute(
                text("SELECT count(*) FROM feedback WHERE used_in_training = FALSE")
            ).scalar() or 0
        except Exception:  # noqa: BLE001 - table may not exist on an old volume yet
            feedback_pending = 0

    return {
        "total_sessions": total,
        "alert_budget": budget,
        "precision_in_queue": round(float(precision), 4) if precision is not None else 0,
        "true_anomalies_total": true_anomalies,
        "entities_monitored": entities,
        "feedback_pending": feedback_pending,
    }


@app.get("/api/alerts")
def get_alerts(
    limit: int = Query(default=200, ge=1, le=1000),
    search: str | None = None,
    predicted_type: str | None = None,
):
    query = """
        WITH ranked AS (
            SELECT *, row_number() OVER (ORDER BY risk_score DESC) AS rank
            FROM scores
        )
        SELECT ranked.rank, ranked.session_id, ranked.entity_id,
               ranked.entity_type, ranked.risk_score, ranked.calibrated_risk,
               ranked.conformal_p_value, ranked.raw_score,
               ranked.detection_method, ranked.predicted_type,
               ranked.predicted_confidence, ranked.explanation,
               ranked.explanation_details, ranked.top_contributing_features,
               ranked.label, ranked.scored_at,
               feedback.verdict AS feedback_verdict
        FROM ranked
        LEFT JOIN LATERAL (
            SELECT verdict
            FROM feedback
            WHERE session_id = ranked.session_id
            ORDER BY created_at DESC
            LIMIT 1
        ) feedback ON TRUE
        WHERE ranked.risk_score >= (
            SELECT percentile_cont(:pct) WITHIN GROUP (ORDER BY risk_score) FROM scores
        )
          AND (:search IS NULL OR ranked.entity_id ILIKE '%' || :search || '%')
          AND (:ptype IS NULL OR ranked.predicted_type = :ptype)
        ORDER BY ranked.risk_score DESC
        LIMIT :limit
    """
    with engine.connect() as connection:
        rows = (
            connection.execute(
                text(query),
                {
                    "pct": ALERT_BUDGET_PCT,
                    "search": search,
                    "ptype": predicted_type,
                    "limit": limit,
                },
            )
            .mappings()
            .all()
        )

    return [
        {
            "rank": row["rank"],
            "session_id": row["session_id"],
            "entity_id": row["entity_id"],
            "entity_type": row["entity_type"],
            "risk_score": _finite(row["risk_score"]),
            "calibrated_risk": _finite(row["calibrated_risk"]),
            "conformal_p_value": _finite(row["conformal_p_value"]),
            "raw_score": _finite(row["raw_score"]),
            "detection_method": row["detection_method"],
            "predicted_attack": row["predicted_type"],
            "classifier_confidence": _finite(row["predicted_confidence"]),
            "reason_string": row["explanation"],
            "explanation_details": _parse_features(row["explanation_details"]),
            "top_features": [
                {"feature": feature, "z_score": _finite(z_score)}
                for feature, z_score in _parse_features(
                    row["top_contributing_features"] or {}
                ).items()
            ],
            "ground_truth": row["label"],
            "feedback_verdict": row["feedback_verdict"],
            "scored_at": (
                row["scored_at"].isoformat()
                if hasattr(row["scored_at"], "isoformat")
                else row["scored_at"]
            ),
        }
        for row in rows
    ]


@app.get("/api/entity/{entity_id}/history")
def entity_history(entity_id: str, limit: int = Query(default=8, ge=1, le=100)):
    query = """
        SELECT session_id, ts, source_ip, geo_location, resource_accessed,
               auth_method, session_duration_sec
        FROM sessions
        WHERE entity_id = :entity_id
        ORDER BY ts DESC
        LIMIT :limit
    """
    with engine.connect() as connection:
        rows = (
            connection.execute(
                text(query), {"entity_id": entity_id, "limit": limit}
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]
