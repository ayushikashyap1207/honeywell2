"""Streaming feature engineering — the live equivalent of ``features.py``.

Reads raw sessions from Kafka, maintains per-entity state (EWMA baseline,
sliding window, resource/device history) in Redis so it survives restarts,
and writes computed feature rows to the ``session_features`` topic.

Run (from the project root, with Kafka/Redis up):
    python -m faust -A streaming.feature_app worker -l info --without-web

The consumer group name is ``anomaly-features``.
"""

import json
import math
import os
import time
from datetime import datetime, timezone

import faust
import redis
from dotenv import load_dotenv

from serving.geo import coords_for, haversine_coords

load_dotenv()

KAFKA_BROKER = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def _kafka_url(broker: str) -> str:
    return broker if "://" in broker else f"kafka://{broker}"


app = faust.App(
    "anomaly-features",
    broker=_kafka_url(KAFKA_BROKER),
    value_serializer="json",
    topic_replication_factor=1,
    topic_partitions=6,
)

raw_topic = app.topic("raw_sessions")
features_topic = app.topic("session_features", partitions=6)

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

EWMA_HALF_LIFE = 30
ALPHA = 1 - 0.5 ** (1 / EWMA_HALF_LIFE)
COLD_START_THRESHOLD = 5


def _event_epoch(session: dict) -> float:
    """Accept ``timestamp_epoch`` or an ISO-8601 ``timestamp`` string."""
    epoch = session.get("timestamp_epoch")
    if epoch is not None:
        try:
            return float(epoch)
        except (TypeError, ValueError):
            pass

    raw = session.get("timestamp")
    if raw:
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time()


def get_entity_state(entity_id: str) -> dict:
    raw = r.get(f"entity_state:{entity_id}")
    if raw:
        return json.loads(raw)
    return {
        "session_count": 0,
        "hour_mean": None,
        "hour_var": 0.0,
        "known_resources": [],
        "known_devices": [],
        "last_lat": None,
        "last_lon": None,
        "last_ts": None,
        "window": [],
    }


def save_entity_state(entity_id: str, state: dict):
    r.set(f"entity_state:{entity_id}", json.dumps(state), ex=86400)  # 24h TTL


@app.agent(raw_topic)
async def compute_features(sessions):
    async for raw in sessions:
        session = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        entity_id = session["entity_id"]
        state = get_entity_state(entity_id)

        ts = _event_epoch(session)
        hour = time.gmtime(int(ts)).tm_hour

        if state["session_count"] < 3:
            hour_z = 0.0
        else:
            deviation = hour - state["hour_mean"]
            std = math.sqrt(state["hour_var"]) or 1.0
            hour_z = abs(deviation) / std

        if state["hour_mean"] is None:
            state["hour_mean"] = hour
        else:
            delta = hour - state["hour_mean"]
            state["hour_mean"] += ALPHA * delta
            state["hour_var"] = (1 - ALPHA) * (state["hour_var"] + ALPHA * delta ** 2)

        resources_this_session = [
            resource.strip()
            for resource in str(session.get("resource_accessed") or "").split(",")
            if resource.strip()
        ]
        novel = [res for res in resources_this_session if res not in state["known_resources"]]
        resource_novelty = len(novel) / max(len(resources_this_session), 1)
        state["known_resources"] = list(
            set(state["known_resources"] + resources_this_session)
        )

        device = session.get("device_fingerprint")
        known_devices = state["known_devices"]
        device_novel = 1 if (state["session_count"] > 0 and device and device not in known_devices) else 0
        if device and device not in known_devices:
            state["known_devices"] = list(set(known_devices + [device]))

        # Geo velocity: synthetic data supplies location names; Okta supplies
        # free text ("City, State, Country"). Unknown/missing geo is skipped,
        # not treated as suspicious.
        geo_km_per_hr = 0.0
        lat, lon = session.get("lat"), session.get("lon")
        if lat is None or lon is None:
            lat, lon = coords_for(session.get("geo_location"))
        if (
            state["last_lat"] is not None
            and state["last_lon"] is not None
            and state["last_ts"] is not None
            and lat is not None
            and lon is not None
        ):
            distance = haversine_coords(
                state["last_lat"], state["last_lon"], lat, lon
            )
            elapsed_hr = max((ts - state["last_ts"]) / 3600, 1 / 3600)
            geo_km_per_hr = distance / elapsed_hr
        if lat is not None and lon is not None:
            state["last_lat"], state["last_lon"], state["last_ts"] = lat, lon, ts

        is_cold_start = 1 if state["session_count"] < COLD_START_THRESHOLD else 0

        # Cross-entity signals — rolling 5-minute window keyed by source IP.
        source_ip = session.get("source_ip", "unknown")
        failed_attempts = int(session.get("failed_auth_attempts") or 0)
        ip_failed_key = f"ip_failed:{source_ip}"
        ip_entities_key = f"ip_entities:{source_ip}"
        if failed_attempts > 0:
            r.zadd(ip_failed_key, {session["session_id"]: ts})
        r.zadd(ip_entities_key, {entity_id: ts})
        r.zremrangebyscore(ip_failed_key, 0, ts - 300)
        r.zremrangebyscore(ip_entities_key, 0, ts - 300)
        r.expire(ip_failed_key, 600)
        r.expire(ip_entities_key, 600)

        command_sequence = str(session.get("command_sequence") or "")
        cmd_seq_len = len([c for c in command_sequence.split(",") if c.strip()])

        feature_row = {
            "session_id": session["session_id"],
            "entity_id": entity_id,
            "entity_type": session.get("entity_type"),
            "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "ts_epoch": ts,
            "geo_location": session.get("geo_location"),
            "resource_accessed": session.get("resource_accessed"),
            "auth_method": session.get("auth_method"),
            "source_ip": source_ip,
            "device_fingerprint": device,
            "command_sequence": command_sequence,
            "label": session.get("label"),
            "anomaly_type": session.get("anomaly_type"),
            "hour_z": hour_z,
            "resource_novelty": resource_novelty,
            "n_resources_this_session": len(resources_this_session),
            "device_novel": device_novel,
            "geo_km_per_hr": geo_km_per_hr,
            "session_duration_sec": session.get("session_duration_sec") or 0,
            "cmd_seq_len": cmd_seq_len,
            "failed_auth_attempts": failed_attempts,
            "is_cold_start": is_cold_start,
            "ip_failed_5min": r.zcard(ip_failed_key),
            "ip_unique_entities_5min": r.zcard(ip_entities_key),
        }

        state["window"] = (state["window"] + [dict(feature_row)])[-5:]
        state["session_count"] += 1
        save_entity_state(entity_id, state)

        feature_row["window_ready"] = len(state["window"]) == 5
        feature_row["window"] = state["window"] if feature_row["window_ready"] else None

        await features_topic.send(key=entity_id, value=feature_row)


if __name__ == "__main__":
    app.main()
