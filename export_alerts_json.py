"""
Run this AFTER your normal pipeline (generate_data.py ... generate_dashboard.py)
has produced data/alerts_explained.csv. It exports the same alert-budget slice
your static dashboard/index.html already shows, as a JSON file the React app
can import directly — no backend, no Docker, no Kafka needed for this step.

Usage (from your project root, honeywell2/):
    python export_alerts_json.py

Output:
    dashboard-app/src/data/alerts.json
"""
import json
import os
import pandas as pd

ALERTS_CSV = "data/alerts_explained.csv"
ACCESS_LOGS_CSV = "data/access_logs_unlabeled.csv"  # optional, for the "recent entity history" panel
OUTPUT_PATH = "dashboard-app/src/data/alerts.json"
ALERT_BUDGET_PCT = 0.99  # top 1% — matches ALERT_BUDGET_PCT in generate_dashboard.py

df = pd.read_csv(ALERTS_CSV)

# keep only the alert-budget slice, ranked by risk_score, same as the static dashboard
threshold = df["risk_score"].quantile(ALERT_BUDGET_PCT)
queue = df[df["risk_score"] >= threshold].sort_values("risk_score", ascending=False).reset_index(drop=True)
queue.insert(0, "rank", queue.index + 1)

import ast

# top_contributing_features is stored as a Python-dict-literal string, e.g.
# "{'device_novel': 1000000.0, 'resource_novelty': 5.71, 'is_cold_start': 4.84}"
# — not JSON — so parse it with ast.literal_eval and turn it into the
# [{feature, z_score}, ...] shape the dashboard expects.
def parse_top_features(val):
    if pd.isna(val):
        return []
    if isinstance(val, dict):
        d = val
    elif isinstance(val, list):
        return val
    else:
        try:
            d = ast.literal_eval(str(val))
        except Exception:
            return [{"feature": "unparsed", "z_score": None, "raw": str(val)}]
    if isinstance(d, dict):
        items = sorted(d.items(), key=lambda kv: abs(kv[1]) if isinstance(kv[1], (int, float)) else 0, reverse=True)
        return [{"feature": k, "z_score": v} for k, v in items]
    return []

records = []
for _, row in queue.iterrows():
    records.append({
        "rank": int(row["rank"]),
        "session_id": row.get("session_id"),
        "entity_id": row.get("entity_id"),
        "entity_type": row.get("entity_type"),
        "risk_score": float(row.get("risk_score", 0)),
        "detection_method": row.get("detection_method"),
        "predicted_attack": row.get("predicted_type"),
        "classifier_confidence": row.get("predicted_confidence"),
        "ground_truth": row.get("label") or row.get("ground_truth") or row.get("anomaly_type"),
        "reason_string": row.get("explanation"),
        "top_features": parse_top_features(row.get("top_contributing_features")),
    })

summary = {
    "total_sessions": int(len(df)),
    "alert_budget": int(len(queue)),
    "precision_in_queue": float((queue["label"] == "anomaly").mean()) if len(queue) else 0,
    "true_anomalies_total": int((df["label"] == "anomaly").sum()),
    "entities_monitored": int(df["entity_id"].nunique()),
}

# optional: last 8 raw sessions per flagged entity, for the detail panel's history list
entity_history = {}
try:
    logs = pd.read_csv(ACCESS_LOGS_CSV, parse_dates=["timestamp"])
    flagged_entities = queue["entity_id"].unique()
    for eid in flagged_entities:
        recent = (
            logs[logs["entity_id"] == eid]
            .sort_values("timestamp", ascending=False)
            .head(8)
        )
        entity_history[str(eid)] = [
            {
                "session_id": r.get("session_id"),
                "ts": str(r.get("timestamp")),
                "geo_location": r.get("geo_location"),
                "resource_accessed": r.get("resource_accessed"),
                "auth_method": r.get("auth_method"),
                "session_duration_sec": r.get("session_duration_sec"),
            }
            for _, r in recent.iterrows()
        ]
except FileNotFoundError:
    print(f"Note: {ACCESS_LOGS_CSV} not found — history panel will be empty until you add it.")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump({"summary": summary, "alerts": records, "entity_history": entity_history}, f, indent=2, default=str)

print(f"Wrote {len(records)} alerts + summary + history for {len(entity_history)} entities to {OUTPUT_PATH}")