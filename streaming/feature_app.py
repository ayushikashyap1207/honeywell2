"""
Streaming feature engineering — the live equivalent of features.py.

Reads raw sessions from Kafka, maintains per-entity state (EWMA baseline,
sliding window, resource/device history) in Redis so it survives restarts,
and writes computed feature rows to the session_features topic.

Run (from honeywell2/, with docker compose up -d already running):
    pip install faust-streaming redis --break-system-packages
    python -m faust -A streaming.feature_app worker -l info

The consumer group name below ("anomaly-features") is what you were
checking earlier with kafka-consumer-groups.sh --describe --group
anomaly-features — it'll show up as ACTIVE once this worker is running.
"""
import math
import time
import json
import faust
import redis

# Your docker-compose Kafka only exposes the single PLAINTEXT://localhost:9092
# listener (no separate 29092 host port), so that's what we connect to here.
app = faust.App(
    "anomaly-features",
    broker="kafka://localhost:9092",
    value_serializer="json",
    topic_replication_factor=1,     # matches your single-broker setup
    topic_partitions=6,
)

raw_topic = app.topic("raw_sessions")
features_topic = app.topic("session_features", partitions=6)

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

EWMA_HALF_LIFE = 30
ALPHA = 1 - 0.5 ** (1 / EWMA_HALF_LIFE)
COLD_START_THRESHOLD = 5


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


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

        ts = session.get("timestamp_epoch") or time.time()
        hour = time.gmtime(int(ts)).tm_hour

        if state["session_count"] < 3:
            hour_z = 0.0
        else:
            deviation = hour - state["hour_mean"]
            std = math.sqrt(state["hour_var"]) or 1.0
            hour_z = deviation / std

        if state["hour_mean"] is None:
            state["hour_mean"] = hour
        else:
            delta = hour - state["hour_mean"]
            state["hour_mean"] += ALPHA * delta
            state["hour_var"] = (1 - ALPHA) * (state["hour_var"] + ALPHA * delta ** 2)

        resources_this_session = str(session.get("resource_accessed", "")).split(",")
        novel = [res for res in resources_this_session if res not in state["known_resources"]]
        resource_novelty = len(novel) / max(len(resources_this_session), 1)
        state["known_resources"] = list(set(state["known_resources"] + resources_this_session))

        device_novel = 1 if session.get("device_fingerprint") not in state["known_devices"] else 0
        state["known_devices"] = list(set(state["known_devices"] + [session.get("device_fingerprint")]))

        geo_km_per_hr = 0.0
        lat, lon = session.get("lat"), session.get("lon")
        if state["last_lat"] is not None and lat is not None:
            dist = haversine_km(state["last_lat"], state["last_lon"], lat, lon)
            elapsed_hr = max((ts - state["last_ts"]) / 3600, 1 / 3600)
            geo_km_per_hr = dist / elapsed_hr
        state["last_lat"], state["last_lon"], state["last_ts"] = lat, lon, ts

        is_cold_start = 1 if state["session_count"] < COLD_START_THRESHOLD else 0

        # cross-entity signals — rolling 5-minute window keyed by source IP
        source_ip = session.get("source_ip", "unknown")
        ip_failed_key = f"ip_failed:{source_ip}"
        ip_entities_key = f"ip_entities:{source_ip}"
        if session.get("failed_auth_attempts", 0) > 0:
            r.zadd(ip_failed_key, {session["session_id"]: ts})
        r.zadd(ip_entities_key, {entity_id: ts})
        r.zremrangebyscore(ip_failed_key, 0, ts - 300)
        r.zremrangebyscore(ip_entities_key, 0, ts - 300)
        r.expire(ip_failed_key, 600)
        r.expire(ip_entities_key, 600)

        feature_row = {
            "session_id": session["session_id"],
            "entity_id": entity_id,
            "entity_type": session.get("entity_type"),
            "hour_z": hour_z,
            "resource_novelty": resource_novelty,
            "n_resources_this_session": len(resources_this_session),
            "device_novel": device_novel,
            "geo_km_per_hr": geo_km_per_hr,
            "session_duration_sec": session.get("session_duration_sec") or 0,
            "cmd_seq_len": len(str(session.get("command_sequence", "")).split(",")),
            "failed_auth_attempts": session.get("failed_auth_attempts", 0),
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
