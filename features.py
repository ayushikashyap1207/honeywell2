
#Feature Engineering


import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

DRIFT_HALF_LIFE_SESSIONS = 30
_EWMA_ALPHA = 1 - 0.5 ** (1 / DRIFT_HALF_LIFE_SESSIONS)


def ewma_update(prev_mean, prev_var, x, alpha=_EWMA_ALPHA):
    """Incremental exponentially-weighted mean/variance update (one new
    observation x). Returns (new_mean, new_var). Standard EWMA variance
    recurrence: var_t = (1-a)*(var_{t-1} + a*(x - mean_{t-1})^2)."""
    diff = x - prev_mean
    new_mean = prev_mean + alpha * diff
    new_var = (1 - alpha) * (prev_var + alpha * diff * diff)
    return new_mean, new_var

GEO_LOCATIONS = {
    "Bangalore_IN": (12.97, 77.59), "Mumbai_IN": (19.07, 72.87),
    "Delhi_IN": (28.61, 77.20), "London_UK": (51.50, -0.12),
    "NewYork_US": (40.71, -74.00), "Frankfurt_DE": (50.11, 8.68),
    "Singapore_SG": (1.35, 103.82), "SaoPaulo_BR": (-23.55, -46.63),
    "Sydney_AU": (-33.87, 151.21), "Moscow_RU": (55.75, 37.61),
}


def haversine_km(g1, g2):
    if g1 not in GEO_LOCATIONS or g2 not in GEO_LOCATIONS:
        return 0.0
    lat1, lon1 = GEO_LOCATIONS[g1]
    lat2, lon2 = GEO_LOCATIONS[g2]
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def build_features(df: pd.DataFrame):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    df["hour_of_day"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["resources_list"] = df["resource_accessed"].apply(lambda x: str(x).split(","))
    df["n_resources_this_session"] = df["resources_list"].apply(len)
    df["cmd_list"] = df["command_sequence"].apply(lambda x: str(x).split(","))
    df["cmd_seq_len"] = df["cmd_list"].apply(len)

    # Precompute peer-group login hour stats per entity_type for cold-start scoring
    peer_stats = {}
    for etype, grp in df.groupby("entity_type"):
        peer_stats[etype] = {
            "mean": grp["hour_of_day"].mean(),
            "std": max(grp["hour_of_day"].std(), 0.5)
        }

    feature_rows = []
    profile_rows = []   # <-- NEW: one row per entity, the learned baseline profile

    for entity_id, group in df.groupby("entity_id", sort=False):
        seen_resources = set()
        seen_devices = set()
        hours_seen = []
        durations_seen = []
        cmd_lens_seen = []
        prev_ts, prev_geo = None, None
        entity_type = group["entity_type"].iloc[0]

        # --- adaptive (EWMA) login-hour baseline state, for concept drift ---
        ewma_mean_h, ewma_var_h = None, None

        for idx, row in group.iterrows():
            n_hist = len(hours_seen)

            # --- time-of-day deviation (adaptive baseline, see ewma_update) ---
            if n_hist >= 3:
                std_h = max(np.sqrt(ewma_var_h), 0.5)
                hour_z = abs(row["hour_of_day"] - ewma_mean_h) / std_h
            else:
                # Cold start: fallback to peer-group baseline instead of flat 0.0
                p_stats = peer_stats.get(row["entity_type"], {"mean": 12.0, "std": 4.0})
                hour_z = abs(row["hour_of_day"] - p_stats["mean"]) / p_stats["std"]

            # --- resource novelty ---
            new_res = [r for r in row["resources_list"] if r not in seen_resources]
            resource_novelty = len(new_res) / max(len(row["resources_list"]), 1)

            # --- device fingerprint change ---
            device_novel = 1 if (n_hist > 0 and row["device_fingerprint"] not in seen_devices) else 0

            # --- geo velocity (impossible travel signal) ---
            geo_km_per_hr = 0.0
            if prev_ts is not None and prev_geo is not None:
                dt_hr = max((row["timestamp"] - prev_ts).total_seconds() / 3600, 1/3600)
                dist = haversine_km(prev_geo, row["geo_location"])
                geo_km_per_hr = dist / dt_hr

            # --- cold start flag ---
            is_cold_start = 1 if n_hist < 5 else 0

            feature_rows.append({
                "session_id": row["session_id"],
                "entity_id": entity_id,
                "entity_type": row["entity_type"],
                "hour_z": hour_z,
                "resource_novelty": resource_novelty,
                "n_resources_this_session": row["n_resources_this_session"],
                "device_novel": device_novel,
                "geo_km_per_hr": geo_km_per_hr,
                "session_duration_sec": row["session_duration_sec"],
                "cmd_seq_len": row["cmd_seq_len"],
                "failed_auth_attempts": row["failed_auth_attempts"],
                "is_cold_start": is_cold_start,
                "history_size": n_hist,
            })

            seen_resources.update(row["resources_list"])
            seen_devices.add(row["device_fingerprint"])
            hours_seen.append(row["hour_of_day"])
            durations_seen.append(row["session_duration_sec"])
            cmd_lens_seen.append(row["cmd_seq_len"])
            prev_ts, prev_geo = row["timestamp"], row["geo_location"]


            # of a lifetime average.
            if len(hours_seen) == 3:
                ewma_mean_h = float(np.mean(hours_seen))
                ewma_var_h = float(np.var(hours_seen))
            elif len(hours_seen) > 3:
                ewma_mean_h, ewma_var_h = ewma_update(ewma_mean_h, ewma_var_h, row["hour_of_day"])

        profile_rows.append({
            "entity_id": entity_id,
            "entity_type": entity_type,
            "n_sessions_observed": len(hours_seen),
            "mean_login_hour": round(float(np.mean(hours_seen)), 2) if hours_seen else None,
            "std_login_hour": round(float(np.std(hours_seen)), 2) if hours_seen else None,
            "adaptive_mean_login_hour": round(ewma_mean_h, 2) if ewma_mean_h is not None else None,
            "adaptive_std_login_hour": round(float(np.sqrt(ewma_var_h)), 2) if ewma_var_h is not None else None,
            "n_distinct_resources_seen": len(seen_resources),
            "resources_seen": "|".join(sorted(seen_resources)),
            "n_distinct_devices_seen": len(seen_devices),
            "mean_session_duration_sec": round(float(np.mean(durations_seen)), 1) if durations_seen else None,
            "mean_cmd_seq_len": round(float(np.mean(cmd_lens_seen)), 2) if cmd_lens_seen else None,
            "is_cold_start_entity": 1 if len(hours_seen) < 5 else 0,
        })

    feat_df = pd.DataFrame(feature_rows)
    profile_df = pd.DataFrame(profile_rows)

    # --- Global (cross-entity) features: brute force / credential stuffing signals ---
    # Rolling 5-minute window failed-attempt count per source_ip, and unique entities per source_ip.
    # Vectorized via pandas time-indexed rolling, grouped by source_ip (fast even on 40k+ rows).
    df_sorted = df.sort_values("timestamp").set_index("timestamp")

    def rolling_per_ip(g):
        g = g.sort_index()
        fail_roll = g["failed_auth_attempts"].rolling("5min").sum()
        ent_codes = g["entity_id"].astype("category").cat.codes
        ent_roll = ent_codes.rolling("5min").apply(lambda x: len(set(x)), raw=True)
        return pd.DataFrame({
            "session_id": g["session_id"].values,
            "ip_failed_5min": fail_roll.values,
            "ip_unique_entities_5min": ent_roll.values,
        }, index=g.index)

    rolled = df_sorted.groupby("source_ip", group_keys=False)[["session_id", "failed_auth_attempts", "entity_id"]].apply(
        rolling_per_ip, include_groups=False
    )
    rolled = rolled.reset_index(drop=True)

    feat_df = feat_df.merge(
        rolled[["session_id", "ip_failed_5min", "ip_unique_entities_5min"]],
        on="session_id", how="left"
    )

    return feat_df, profile_df


if __name__ == "__main__":
    df = pd.read_csv("data/access_logs_unlabeled.csv")
    feat_df, profile_df = build_features(df)
    feat_df.to_csv("data/features.csv", index=False)
    profile_df.to_csv("data/entity_baseline_profiles.csv", index=False)
    print(feat_df.shape)
    print(feat_df.head())
    print(f"\nSaved data/entity_baseline_profiles.csv ({len(profile_df)} entities)")
    print(profile_df.head())