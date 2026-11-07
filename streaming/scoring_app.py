"""Streaming scoring and live persistence.

Reads computed feature rows from ``session_features``, scores each one with
whichever detector applies (Isolation Forest for cold-start entities, LSTM for
entities with a full 5-session window), fuses the raw scores into a 0-1 risk
rank, attaches the attack-type classifier and explanation, then:

* writes the full row to TimescaleDB (``sessions`` + ``scores``)
* publishes it to the ``scored_sessions`` Kafka topic
* publishes it to Redis pub/sub ``live_scores`` for the WebSocket dashboard

Run:
    python -m faust -A streaming.scoring_app worker -l info --without-web
"""

import json
import os
import time
from datetime import datetime, timezone

import asyncpg
import faust
import redis
from dotenv import load_dotenv

from serving.attribution import build_counterfactual
from serving.calibration import CONFORMAL_ALPHA, apply_calibration, load_calibrators
from serving.models import (
    classify_row,
    explain,
    load_models,
    score_isolation_forest,
    score_lstm,
)

load_dotenv()

KAFKA_BROKER = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://anomaly:anomaly_pw@localhost:5432/anomaly_detection",
)


def _kafka_url(broker: str) -> str:
    return broker if "://" in broker else f"kafka://{broker}"


app = faust.App(
    "anomaly-scoring",
    broker=_kafka_url(KAFKA_BROKER),
    value_serializer="json",
    topic_replication_factor=1,
)

features_topic = app.topic("session_features")
scored_topic = app.topic("scored_sessions", partitions=6)

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

REFERENCE_SAMPLE_SIZE = 2000

BUNDLE = load_models()
CALIBRATORS = load_calibrators()

_pool: asyncpg.Pool | None = None
_pool_error_logged = False
_last_pool_attempt = 0.0


async def get_pool():
    """Lazily connect to Postgres; keep streaming even if the DB is down."""
    global _pool, _pool_error_logged, _last_pool_attempt
    if _pool is not None:
        return _pool

    now = time.monotonic()
    if now - _last_pool_attempt < 30:
        return None
    _last_pool_attempt = now

    try:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=5,
            command_timeout=10,
        )
        _pool_error_logged = False
        print("Connected to Postgres for live score persistence")
    except Exception as exc:  # noqa: BLE001
        if not _pool_error_logged:
            print(f"WARNING: Postgres unavailable ({exc}); live scores will not be persisted")
            _pool_error_logged = True
        return None
    return _pool


def rank_against_reference(raw_score: float, detector_name: str) -> float:
    """Percentile rank against a rolling reference sample.

    Uses a bounded Redis list rather than a sorted set: trimming a sorted set
    by rank keeps only the highest scores, which biases every percentile
    downward. A rolling window matches the streaming/concept-drift setting.
    """
    key = f"ref_scores:{detector_name}"
    r.rpush(key, raw_score)
    r.ltrim(key, -REFERENCE_SAMPLE_SIZE, -1)

    scores = [float(score) for score in r.lrange(key, 0, -1)]
    if len(scores) < 10:
        return 0.5  # not enough reference data yet
    return sum(1 for score in scores if score <= raw_score) / len(scores)


def _parse_ts(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


async def persist_result(result: dict):
    """Upsert the session and its score into TimescaleDB."""
    pool = await get_pool()
    if pool is None:
        return

    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO sessions (
                        session_id, entity_id, entity_type, ts, source_ip,
                        geo_location, resource_accessed, auth_method,
                        session_duration_sec
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (session_id) DO UPDATE SET
                        entity_id = EXCLUDED.entity_id,
                        entity_type = EXCLUDED.entity_type,
                        ts = EXCLUDED.ts,
                        source_ip = EXCLUDED.source_ip,
                        geo_location = EXCLUDED.geo_location,
                        resource_accessed = EXCLUDED.resource_accessed,
                        auth_method = EXCLUDED.auth_method,
                        session_duration_sec = EXCLUDED.session_duration_sec
                    """,
                    result["session_id"],
                    result["entity_id"],
                    result.get("entity_type"),
                    _parse_ts(result.get("ts")),
                    result.get("source_ip"),
                    result.get("geo_location"),
                    result.get("resource_accessed"),
                    result.get("auth_method"),
                    result.get("session_duration_sec"),
                )
                await connection.execute(
                    """
                    INSERT INTO scores (
                        session_id, entity_id, entity_type, risk_score,
                        calibrated_risk, conformal_p_value, detection_method,
                        predicted_type, predicted_confidence, explanation,
                        explanation_details, top_contributing_features, label,
                        anomaly_type, raw_score
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                        $13, $14, $15
                    )
                    ON CONFLICT (session_id) DO UPDATE SET
                        risk_score = EXCLUDED.risk_score,
                        calibrated_risk = EXCLUDED.calibrated_risk,
                        conformal_p_value = EXCLUDED.conformal_p_value,
                        detection_method = EXCLUDED.detection_method,
                        predicted_type = EXCLUDED.predicted_type,
                        predicted_confidence = EXCLUDED.predicted_confidence,
                        explanation = EXCLUDED.explanation,
                        explanation_details = EXCLUDED.explanation_details,
                        top_contributing_features = EXCLUDED.top_contributing_features,
                        label = EXCLUDED.label,
                        anomaly_type = EXCLUDED.anomaly_type,
                        raw_score = EXCLUDED.raw_score,
                        scored_at = now()
                    """,
                    result["session_id"],
                    result["entity_id"],
                    result.get("entity_type"),
                    result["risk_score"],
                    result.get("calibrated_risk"),
                    result.get("conformal_p_value"),
                    result["detection_method"],
                    result.get("predicted_type"),
                    result.get("predicted_confidence"),
                    result.get("explanation"),
                    json.dumps(result.get("explanation_details") or {}),
                    json.dumps(result.get("top_contributing_features") or {}),
                    result.get("label"),
                    result.get("anomaly_type"),
                    result.get("raw_score"),
                )
    except Exception as exc:  # noqa: BLE001 - a DB hiccup must not kill the consumer
        print(f"WARNING: could not persist {result.get('session_id')} to Postgres: {exc}")


def build_result(feature_row: dict):
    # Prefer the sequence model whenever a full window exists; the cold-start
    # Isolation Forest is the fallback, matching combine_scores.py.
    if feature_row.get("window_ready") and BUNDLE.sequence_model is not None:
        raw_score = score_lstm(BUNDLE, feature_row["window"])
        risk_score = rank_against_reference(raw_score, "lstm")
        detection_method = "sequence_lstm"
    elif feature_row.get("is_cold_start") and BUNDLE.iso_forest is not None:
        raw_score = score_isolation_forest(BUNDLE, feature_row)
        risk_score = rank_against_reference(raw_score, "isolation_forest")
        detection_method = "cold_start_isolation_forest"
    else:
        return None

    calibrated_risk, conformal_p_value = apply_calibration(
        CALIBRATORS, detection_method, risk_score
    )
    predicted_type, predicted_confidence = classify_row(BUNDLE, feature_row)
    explanation, contributions = explain(BUNDLE, feature_row)

    # Counterfactual re-scoring is relatively expensive; only do it for alerts
    # that would actually reach the analyst queue.
    counterfactual = None
    if risk_score >= 0.9:
        counterfactual = build_counterfactual(
            BUNDLE, feature_row, detection_method, BUNDLE.baseline_stats
        )

    explanation_details = {
        "method": "zscore+counterfactual",
        "z_scores": contributions,
        "counterfactual": counterfactual,
        "calibration": {
            "calibrated_risk": round(float(calibrated_risk), 6),
            "conformal_p_value": (
                round(float(conformal_p_value), 6)
                if conformal_p_value is not None
                else None
            ),
            "alpha": CONFORMAL_ALPHA,
        },
    }

    return {
        "session_id": feature_row["session_id"],
        "entity_id": feature_row["entity_id"],
        "entity_type": feature_row.get("entity_type") or "unknown",
        "risk_score": round(float(risk_score), 6),
        "calibrated_risk": round(float(calibrated_risk), 6),
        "conformal_p_value": (
            round(float(conformal_p_value), 6)
            if conformal_p_value is not None
            else None
        ),
        "raw_score": float(raw_score),
        "detection_method": detection_method,
        "predicted_type": predicted_type,
        "predicted_confidence": predicted_confidence,
        "explanation": explanation,
        "explanation_details": explanation_details,
        "top_contributing_features": contributions,
        "label": feature_row.get("label"),
        "anomaly_type": feature_row.get("anomaly_type"),
        "ts": feature_row.get("timestamp"),
        "source_ip": feature_row.get("source_ip"),
        "geo_location": feature_row.get("geo_location"),
        "resource_accessed": feature_row.get("resource_accessed"),
        "auth_method": feature_row.get("auth_method"),
        "session_duration_sec": feature_row.get("session_duration_sec"),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


@app.agent(features_topic)
async def score_sessions(features):
    async for raw in features:
        feature_row = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        result = build_result(feature_row)
        if result is None:
            continue  # not enough window yet, and not flagged cold-start

        print(
            f"Scored {result['session_id']}: {result['risk_score']:.3f} "
            f"via {result['detection_method']}"
        )

        await scored_topic.send(key=result["entity_id"], value=result)
        try:
            r.publish("live_scores", json.dumps(result))
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: Redis publish failed: {exc}")

        await persist_result(result)


if __name__ == "__main__":
    app.main()
