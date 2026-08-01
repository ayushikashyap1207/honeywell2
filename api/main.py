"""
FastAPI backend for the analyst dashboard.

Run (from honeywell2/, after docker compose up -d and load_to_postgres.py):
    pip install fastapi "uvicorn[standard]" sqlalchemy psycopg2-binary --break-system-packages
    uvicorn api.main:app --reload --port 8000

Then in dashboard-app/.env (create it if missing):
    VITE_API_URL=http://localhost:8000
and restart `npm run dev` — the dashboard will automatically switch from
"Showing exported pipeline data" to "Live feed connected".
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
import json


DB_URL = "postgresql://anomaly:anomaly_pw@localhost:5432/anomaly_detection"
ALERT_BUDGET_PCT = 0.99  # top 1%, matches ALERT_BUDGET_PCT in generate_dashboard.py

engine = create_engine(DB_URL)
def _parse_features(val):
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return {}

app = FastAPI(title="Behavioural Anomaly Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import WebSocket
import asyncio
import redis.asyncio as aioredis

@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()
    redis_client = aioredis.Redis(host="localhost", port=6379, decode_responses=True)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("live_scores")
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            await websocket.send_text(message["data"])
    except Exception:
        pass
    finally:
        await pubsub.unsubscribe("live_scores")
        await redis_client.close()
    
@app.get("/api/summary")
def summary():
    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM scores")).scalar() or 0
        entities = conn.execute(text("SELECT count(DISTINCT entity_id) FROM scores")).scalar() or 0

        threshold = conn.execute(text("""
            SELECT percentile_cont(:pct) WITHIN GROUP (ORDER BY risk_score) FROM scores
        """), {"pct": ALERT_BUDGET_PCT}).scalar()
        threshold = threshold if threshold is not None else 1.0

        budget = conn.execute(text("SELECT count(*) FROM scores WHERE risk_score >= :t"),
                               {"t": threshold}).scalar() or 0
        precision = conn.execute(text("""
            SELECT avg(CASE WHEN label = 'anomaly' THEN 1.0 ELSE 0.0 END)
            FROM scores WHERE risk_score >= :t
        """), {"t": threshold}).scalar()
        true_anomalies = conn.execute(text("SELECT count(*) FROM scores WHERE label = 'anomaly'")).scalar() or 0

    return {
        "total_sessions": total,
        "alert_budget": budget,
        "precision_in_queue": round(precision, 4) if precision is not None else 0,
        "true_anomalies_total": true_anomalies,
        "entities_monitored": entities,
    }


@app.get("/api/alerts")
def get_alerts(limit: int = 200, search: str | None = None, predicted_type: str | None = None):
    query = """
        WITH ranked AS (
            SELECT *, row_number() OVER (ORDER BY risk_score DESC) AS rank
            FROM scores
        )
        SELECT rank, session_id, entity_id, entity_type, risk_score, detection_method,
               predicted_type, predicted_confidence, explanation,
               top_contributing_features, label
        FROM ranked
        WHERE risk_score >= (SELECT percentile_cont(:pct) WITHIN GROUP (ORDER BY risk_score) FROM scores)
          AND (:search IS NULL OR entity_id ILIKE '%' || :search || '%')
          AND (:ptype IS NULL OR predicted_type = :ptype)
        ORDER BY risk_score DESC
        LIMIT :limit
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query), {
            "pct": ALERT_BUDGET_PCT, "search": search, "ptype": predicted_type, "limit": limit,
        }).mappings().all()

    return [
        {
            "rank": r["rank"],
            "session_id": r["session_id"],
            "entity_id": r["entity_id"],
            "entity_type": r["entity_type"],
            "risk_score": r["risk_score"],
            "detection_method": r["detection_method"],
            "predicted_attack": r["predicted_type"],
            "classifier_confidence": r["predicted_confidence"],
            "reason_string": r["explanation"],
            "top_features": [{"feature": k, "z_score": v} for k, v in _parse_features (r["top_contributing_features"] or {}).items()],
            "ground_truth": r["label"],
        }
        for r in rows
    ]


@app.get("/api/entity/{entity_id}/history")
def entity_history(entity_id: str, limit: int = 8):
    query = """
        SELECT session_id, ts, geo_location, resource_accessed, auth_method, session_duration_sec
        FROM sessions WHERE entity_id = :eid ORDER BY ts DESC LIMIT :limit
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query), {"eid": entity_id, "limit": limit}).mappings().all()
    return [dict(r) for r in rows]
