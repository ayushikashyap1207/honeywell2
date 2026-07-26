
#Synthetic Access-Log Generator


import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import argparse
import math

random.seed(42)
np.random.seed(42)


DEFAULT_TARGET_SESSIONS = 20000
MIN_TARGET_SESSIONS = 50
MAX_TARGET_SESSIONS = 100000

ANOMALY_RATE = 0.02                # ~2% of sessions are attacks
# entity type mix stays proportional at any scale: 60% users, 15% service accts, 25% devices
USER_FRAC, SVC_FRAC, DEV_FRAC = 0.60, 0.15, 0.25


_USER_SESSIONS_MID = (2 + 6) / 2          # 4
_SVC_SESSIONS_MID = (10 + 40) / 2         # 25
_DEV_SESSIONS_MID = (20 + 100) / 2        # 60
AVG_SESSIONS_PER_ENTITY_PER_DAY = (
    USER_FRAC * _USER_SESSIONS_MID
    + SVC_FRAC * _SVC_SESSIONS_MID
    + DEV_FRAC * _DEV_SESSIONS_MID
)  # ~21.15, vs. the old hardcoded 8 -- this was the root cause of the rate bug


def compute_scale(target_sessions: int):
    """Scale entity counts + simulated days to approximately hit target_sessions total rows,
    while keeping enough entities/days for cold-start + multi-day patterns to make sense."""
    target_sessions = max(MIN_TARGET_SESSIONS, min(MAX_TARGET_SESSIONS, target_sessions))


    min_days = 3
    min_total_entities = 6

    total_entities = max(min_total_entities, round(math.sqrt(target_sessions / AVG_SESSIONS_PER_ENTITY_PER_DAY)))
    n_days = max(min_days, round(target_sessions / (total_entities * AVG_SESSIONS_PER_ENTITY_PER_DAY)))

    n_users = max(3, round(total_entities * USER_FRAC))
    n_svc = max(1, round(total_entities * SVC_FRAC))
    n_dev = max(1, round(total_entities * DEV_FRAC))

    cold_start = max(1, min(n_users + n_svc + n_dev - 1, round((n_users + n_svc + n_dev) * 0.08)))

    return n_users, n_svc, n_dev, n_days, cold_start


N_USERS, N_SERVICE_ACCOUNTS, N_DEVICES, N_DAYS, COLD_START_ENTITY_COUNT = compute_scale(DEFAULT_TARGET_SESSIONS)


INJECT_SCALE = 1.0


def scaled_range(lo, hi):
    lo_s = max(2, int(round(lo * INJECT_SCALE)))
    hi_s = max(lo_s + 1, int(round(hi * INJECT_SCALE)))
    return lo_s, hi_s

RESOURCE_POOL = [f"resource_{i:03d}" for i in range(1, 121)]
AUTH_METHODS = ["password", "token", "certificate", "biometric"]
GEO_LOCATIONS = {
    "Bangalore_IN": (12.97, 77.59), "Mumbai_IN": (19.07, 72.87),
    "Delhi_IN": (28.61, 77.20), "London_UK": (51.50, -0.12),
    "NewYork_US": (40.71, -74.00), "Frankfurt_DE": (50.11, 8.68),
    "Singapore_SG": (1.35, 103.82), "SaoPaulo_BR": (-23.55, -46.63),
    "Sydney_AU": (-33.87, 151.21), "Moscow_RU": (55.75, 37.61),
}
GEO_NAMES = list(GEO_LOCATIONS.keys())

COMMANDS = ["read", "write", "list", "delete", "exec", "download", "upload", "config_change"]


def random_ip():
    return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def haversine_km(geo1, geo2):
    from math import radians, sin, cos, sqrt, atan2
    lat1, lon1 = GEO_LOCATIONS[geo1]
    lat2, lon2 = GEO_LOCATIONS[geo2]
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


# ---------------------------------------------------------------------------
# Step 1: Build entity behavioural profiles
# ---------------------------------------------------------------------------
def build_entities():
    entities = []
    for i in range(N_USERS):
        entities.append({
            "entity_id": f"user_{uuid.uuid4().hex[:8]}",
            "entity_type": "user",
            "home_geo": random.choice(GEO_NAMES),
            "login_hour_mean": random.uniform(8, 11),
            "login_hour_std": random.uniform(0.5, 1.5),
            "usual_resources": random.sample(RESOURCE_POOL, k=random.randint(5, 15)),
            "usual_auth": random.choice(AUTH_METHODS),
            "device_fingerprint": f"{random.choice(['Win11','macOS14','Ubuntu22'])}-{uuid.uuid4().hex[:6]}",
            "sessions_per_day": random.uniform(2, 6),
        })
    for i in range(N_SERVICE_ACCOUNTS):
        entities.append({
            "entity_id": f"svc_{uuid.uuid4().hex[:8]}",
            "entity_type": "service_account",
            "home_geo": random.choice(GEO_NAMES),
            "login_hour_mean": random.uniform(0, 23),  # service accounts run all hours
            "login_hour_std": 4.0,
            "usual_resources": random.sample(RESOURCE_POOL, k=random.randint(2, 6)),
            "usual_auth": "certificate",
            "device_fingerprint": f"svc-node-{uuid.uuid4().hex[:6]}",
            "sessions_per_day": random.uniform(10, 40),
        })
    for i in range(N_DEVICES):
        entities.append({
            "entity_id": f"dev_{uuid.uuid4().hex[:8]}",
            "entity_type": "edge_device",
            "home_geo": random.choice(GEO_NAMES),
            "login_hour_mean": random.uniform(0, 23),
            "login_hour_std": 6.0,
            "usual_resources": random.sample(RESOURCE_POOL, k=random.randint(1, 3)),
            "usual_auth": random.choice(["token", "certificate"]),
            "device_fingerprint": f"{random.choice(['fw2.1','fw3.0','fw1.8'])}-{uuid.uuid4().hex[:6]}",
            "sessions_per_day": random.uniform(20, 100),
        })
    return entities


# ---------------------------------------------------------------------------
# Step 2: Simulate normal sessions for an entity
# ---------------------------------------------------------------------------
def gen_normal_session(entity, day_offset, base_date):
    hour = np.clip(np.random.normal(entity["login_hour_mean"], entity["login_hour_std"]), 0, 23.99)
    ts = base_date + timedelta(days=day_offset, hours=hour)
    n_resources = random.randint(1, min(4, len(entity["usual_resources"])))
    resources = random.sample(entity["usual_resources"], k=n_resources)
    session_id = uuid.uuid4().hex[:12]
    return {
        "session_id": session_id,
        "entity_id": entity["entity_id"],
        "entity_type": entity["entity_type"],
        "timestamp": ts,
        "source_ip": random_ip(),
        "geo_location": entity["home_geo"],
        "resource_accessed": ",".join(resources),
        "auth_method": entity["usual_auth"],
        "session_duration_sec": max(5, int(np.random.normal(300, 120))),
        "command_sequence": ",".join(random.choices(COMMANDS, k=random.randint(2, 6))),
        "device_fingerprint": entity["device_fingerprint"],
        "failed_auth_attempts": 0,
        "label": "normal",
        "anomaly_type": "none",
    }


# ---------------------------------------------------------------------------
# Step 3: Anomaly injectors -- each returns a list of session dicts
# ---------------------------------------------------------------------------
def inject_brute_force(entity, base_date, day_offset):
    ts0 = base_date + timedelta(days=day_offset, hours=random.uniform(0, 23))
    sessions = []
    attacker_ip = random_ip()
    lo, hi = scaled_range(8, 25)
    n_attempts = random.randint(lo, hi)
    for i in range(n_attempts):
        sessions.append({
            "session_id": uuid.uuid4().hex[:12], "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"], "timestamp": ts0 + timedelta(seconds=i * random.uniform(1, 3)),
            "source_ip": attacker_ip, "geo_location": random.choice(GEO_NAMES),
            "resource_accessed": "login_endpoint", "auth_method": entity["usual_auth"],
            "session_duration_sec": random.randint(1, 5),
            "command_sequence": "auth_attempt", "device_fingerprint": "unknown",
            "failed_auth_attempts": 1, "label": "anomaly", "anomaly_type": "brute_force",
        })
    return sessions


def inject_impossible_travel(entity, base_date, day_offset):
    hour = random.uniform(0, 20)
    ts1 = base_date + timedelta(days=day_offset, hours=hour)
    far_geo = random.choice([g for g in GEO_NAMES if g != entity["home_geo"]])
    gap_minutes = random.uniform(2, 20)
    ts2 = ts1 + timedelta(minutes=gap_minutes)
    dist = haversine_km(entity["home_geo"], far_geo)
    s1 = gen_normal_session(entity, day_offset, base_date)
    s1["timestamp"] = ts1
    s2 = dict(s1)
    s2["session_id"] = uuid.uuid4().hex[:12]
    s2["timestamp"] = ts2
    s2["geo_location"] = far_geo
    s2["source_ip"] = random_ip()
    s2["label"], s2["anomaly_type"] = "anomaly", "impossible_travel"
    s2["_geo_km_per_hr"] = dist / (gap_minutes / 60)
    return [s1, s2]


def inject_credential_stuffing(entities, base_date, day_offset):
    """Many entity_ids, few source_ips, high failure rate."""
    ts0 = base_date + timedelta(days=day_offset, hours=random.uniform(0, 23))
    attacker_ips = [random_ip() for _ in range(2)]
    lo, hi = scaled_range(20, 20)
    targets = random.sample(entities, k=min(hi, len(entities)))
    sessions = []
    for i, ent in enumerate(targets):
        sessions.append({
            "session_id": uuid.uuid4().hex[:12], "entity_id": ent["entity_id"],
            "entity_type": ent["entity_type"], "timestamp": ts0 + timedelta(seconds=i * random.uniform(1, 4)),
            "source_ip": random.choice(attacker_ips), "geo_location": random.choice(GEO_NAMES),
            "resource_accessed": "login_endpoint", "auth_method": ent["usual_auth"],
            "session_duration_sec": random.randint(1, 4),
            "command_sequence": "auth_attempt", "device_fingerprint": "unknown",
            "failed_auth_attempts": 1, "label": "anomaly", "anomaly_type": "credential_stuffing",
        })
    return sessions


def inject_lateral_movement(entity, base_date, day_offset, all_resources):
    ts0 = base_date + timedelta(days=day_offset, hours=random.uniform(9, 18))
    unseen = [r for r in all_resources if r not in entity["usual_resources"]]
    lo, hi = scaled_range(6, 12)
    breadth = random.sample(unseen, k=min(len(unseen), random.randint(lo, hi)))
    sessions = []
    for i, res in enumerate(breadth):
        sessions.append({
            "session_id": uuid.uuid4().hex[:12], "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"], "timestamp": ts0 + timedelta(minutes=i * 2),
            "source_ip": random_ip(), "geo_location": entity["home_geo"],
            "resource_accessed": res, "auth_method": entity["usual_auth"],
            "session_duration_sec": random.randint(30, 90),
            "command_sequence": "list,read,exec", "device_fingerprint": entity["device_fingerprint"],
            "failed_auth_attempts": 0, "label": "anomaly", "anomaly_type": "lateral_movement",
        })
    return sessions


def inject_device_spoofing(entity, base_date, day_offset):
    s = gen_normal_session(entity, day_offset, base_date)
    s["device_fingerprint"] = f"{random.choice(['Win7','AndroidOld','LinuxUnknown'])}-{uuid.uuid4().hex[:6]}"
    s["label"], s["anomaly_type"] = "anomaly", "device_spoofing"
    return [s]


def inject_low_and_slow(entity, base_date, day_offset, all_resources):
    """Gradual off-hours resource access building up over several days -- call once per day in the window."""
    unseen = [r for r in all_resources if r not in entity["usual_resources"]]
    n = max(1, int(round(random.randint(1, 3) * max(INJECT_SCALE, 0.34))))
    sessions = []
    for i in range(n):
        hour = random.uniform(0, 5)  # off-hours
        ts = base_date + timedelta(days=day_offset, hours=hour)
        sessions.append({
            "session_id": uuid.uuid4().hex[:12], "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"], "timestamp": ts,
            "source_ip": random_ip(), "geo_location": entity["home_geo"],
            "resource_accessed": random.choice(unseen), "auth_method": entity["usual_auth"],
            "session_duration_sec": random.randint(60, 200),
            "command_sequence": "read,download", "device_fingerprint": entity["device_fingerprint"],
            "failed_auth_attempts": 0, "label": "anomaly", "anomaly_type": "low_and_slow_exfiltration",
        })
    return sessions


def inject_insider_drift(entity, base_date, day_offset, all_resources):
    """Legitimate-looking slow privilege/resource expansion -- edge case, ambiguous label."""
    unseen = [r for r in all_resources if r not in entity["usual_resources"]]
    res = random.choice(unseen) if unseen else random.choice(all_resources)
    s = gen_normal_session(entity, day_offset, base_date)
    s["resource_accessed"] = res
    s["label"], s["anomaly_type"] = "edge_case", "insider_drift"
    return [s]


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def generate(target_sessions: int = DEFAULT_TARGET_SESSIONS):
    global N_USERS, N_SERVICE_ACCOUNTS, N_DEVICES, N_DAYS, COLD_START_ENTITY_COUNT, INJECT_SCALE
    N_USERS, N_SERVICE_ACCOUNTS, N_DEVICES, N_DAYS, COLD_START_ENTITY_COUNT = compute_scale(target_sessions)
    # scale=1.0 at 5000+ rows, shrinks down to ~0.15 at the 50-row floor
    INJECT_SCALE = max(0.15, min(1.0, target_sessions / 5000))

    entities = build_entities()
    base_date = datetime(2026, 6, 1)
    all_rows = []

    cold_start_ids = set(e["entity_id"] for e in random.sample(entities, COLD_START_ENTITY_COUNT))

    for entity in entities:
        n_days = N_DAYS
        start_day = 0
        if entity["entity_id"] in cold_start_ids:
            start_day = N_DAYS - random.randint(1, 2)  # only last 1-2 days of history

        for day in range(start_day, n_days):
            n_sessions = max(1, int(np.random.poisson(entity["sessions_per_day"])))
            for _ in range(n_sessions):
                all_rows.append(gen_normal_session(entity, day, base_date))

    # --- Inject anomalies ---
    all_resources = RESOURCE_POOL

    target_anomaly_sessions = int(target_sessions * ANOMALY_RATE / (1 - ANOMALY_RATE))

    injected = 0
    while injected < target_anomaly_sessions:
        pattern = random.choices(
            ["brute_force", "impossible_travel", "credential_stuffing",
             "lateral_movement", "device_spoofing", "low_and_slow", "insider_drift"],
            weights=[15, 15, 10, 15, 15, 20, 10]
        )[0]
        entity = random.choice(entities)
        day = random.randint(max(0, N_DAYS - 10), N_DAYS - 1)  # attacks concentrated in recent window (test period)

        if pattern == "brute_force":
            rows = inject_brute_force(entity, base_date, day)
        elif pattern == "impossible_travel":
            rows = inject_impossible_travel(entity, base_date, day)
        elif pattern == "credential_stuffing":
            rows = inject_credential_stuffing(entities, base_date, day)
        elif pattern == "lateral_movement":
            rows = inject_lateral_movement(entity, base_date, day, all_resources)
        elif pattern == "device_spoofing":
            rows = inject_device_spoofing(entity, base_date, day)
        elif pattern == "low_and_slow":
            rows = inject_low_and_slow(entity, base_date, day, all_resources)
        else:
            rows = inject_insider_drift(entity, base_date, day, all_resources)

        all_rows.extend(rows)
        injected += len(rows)

    df = pd.DataFrame(all_rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "_geo_km_per_hr" not in df.columns:
        df["_geo_km_per_hr"] = np.nan



    n_anomalous = (df["label"] != "normal").sum()
    n_normal_target = max(0, target_sessions - n_anomalous)
    normal_df = df[df["label"] == "normal"]
    other_df = df[df["label"] != "normal"]
    if len(normal_df) > n_normal_target:
        normal_df = normal_df.sample(n=n_normal_target, random_state=42)
    df = pd.concat([normal_df, other_df]).sort_values("timestamp").reset_index(drop=True)

    entities_df = pd.DataFrame(entities)
    return df, entities_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic access-log data.")
    parser.add_argument(
        "--target_sessions", type=int, default=DEFAULT_TARGET_SESSIONS,
        help=f"Approximate total rows to generate ({MIN_TARGET_SESSIONS}-{MAX_TARGET_SESSIONS}). Default {DEFAULT_TARGET_SESSIONS}."
    )
    args = parser.parse_args()

    df, entities_df = generate(target_sessions=args.target_sessions)

  
    df.to_csv("data/access_logs_full_with_labels.csv", index=False)

    ground_truth = df[["session_id", "label", "anomaly_type"]].copy()
    ground_truth.to_csv("data/ground_truth_labels.csv", index=False)

    inference_df = df.drop(columns=["label", "anomaly_type", "_geo_km_per_hr"])
    inference_df.to_csv("data/access_logs_unlabeled.csv", index=False)

   
    entities_out = entities_df.copy()
    entities_out["usual_resources"] = entities_out["usual_resources"].apply(lambda r: "|".join(r))
    entities_out.to_csv("data/entity_profiles.csv", index=False)

    print(f"Total sessions: {len(df)}")
    print(f"Anomalies: {(df['label']=='anomaly').sum()}  "
          f"Edge cases: {(df['label']=='edge_case').sum()}  "
          f"Normal: {(df['label']=='normal').sum()}")
    print(f"Actual anomaly rate: {(df['label']=='anomaly').sum() / len(df) * 100:.2f}% "
          f"(target: {ANOMALY_RATE*100:.1f}%)")
    print(df["anomaly_type"].value_counts())