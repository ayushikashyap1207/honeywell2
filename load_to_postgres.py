"""
Loads your existing pipeline output (data/alerts_explained.csv and
data/access_logs_unlabeled.csv) into the Postgres/TimescaleDB tables created
by db/schema.sql. Run this any time you regenerate the pipeline to refresh
the backend's data.

Usage (from honeywell2/, with docker compose up -d already running):
    pip install psycopg2-binary sqlalchemy pandas --break-system-packages
    python load_to_postgres.py
"""
import json
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import JSONB

DB_URL = "postgresql://anomaly:anomaly_pw@localhost:5432/anomaly_detection"
ALERTS_CSV = "data/alerts_explained.csv"
ACCESS_LOGS_CSV = "data/access_logs_unlabeled.csv"

engine = create_engine(DB_URL)


def load_sessions():
    try:
        logs = pd.read_csv(ACCESS_LOGS_CSV)
    except FileNotFoundError:
        print(f"Skipping sessions table — {ACCESS_LOGS_CSV} not found.")
        return
    cols = ["session_id", "entity_id", "entity_type", "timestamp", "geo_location",
            "resource_accessed", "auth_method", "session_duration_sec"]
    logs = logs[[c for c in cols if c in logs.columns]].rename(columns={"timestamp": "ts"})
    logs = logs.drop_duplicates(subset="session_id")
    with engine.begin() as conn:
        logs.to_sql("sessions", conn, if_exists="append", index=False,
                     method="multi", chunksize=1000)
    print(f"Loaded {len(logs)} rows into sessions")


def load_scores():
    df = pd.read_csv(ALERTS_CSV)

    def to_jsonb(val):
        if pd.isna(val):
            return json.dumps({})
        if isinstance(val, str):
            try:
                import ast
                return json.dumps(ast.literal_eval(val))
            except Exception:
                return json.dumps({})
        return json.dumps(val)

    df["top_contributing_features"] = df["top_contributing_features"].apply(to_jsonb) \
        if "top_contributing_features" in df.columns else json.dumps({})

    cols = ["session_id", "entity_id", "entity_type", "risk_score", "detection_method",
            "predicted_type", "predicted_confidence", "explanation",
            "top_contributing_features", "label", "anomaly_type"]
    df = df[[c for c in cols if c in df.columns]]

    with engine.begin() as conn:
        df.to_sql("scores", conn, if_exists="append", index=False,
                   method="multi", chunksize=500,
                   dtype={"top_contributing_features": JSONB})
    print(f"Loaded {len(df)} rows into scores")


if __name__ == "__main__":
    with engine.begin() as conn:
        # clear previous run's data so re-running this script doesn't duplicate/conflict
        # on the session_id primary keys — order matters because of the FK
        conn.execute(text("TRUNCATE scores"))
        conn.execute(text("TRUNCATE sessions CASCADE"))
    load_sessions()
    load_scores()
    print("Done. Verify with: psql ... -c 'SELECT count(*) FROM scores;'")
