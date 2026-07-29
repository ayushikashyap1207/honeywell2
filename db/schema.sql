-- Run this once against your running TimescaleDB container:
--   psql "postgresql://anomaly:anomaly_pw@localhost:5432/anomaly_detection" -f db/schema.sql

CREATE TABLE IF NOT EXISTS sessions (
    session_id              TEXT PRIMARY KEY,
    entity_id               TEXT NOT NULL,
    entity_type              TEXT NOT NULL,
    ts                        TIMESTAMPTZ,
    geo_location              TEXT,
    resource_accessed         TEXT,
    auth_method               TEXT,
    session_duration_sec      DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS scores (
    session_id                TEXT PRIMARY KEY REFERENCES sessions(session_id),
    entity_id                 TEXT NOT NULL,
    entity_type                TEXT,
    risk_score                 DOUBLE PRECISION NOT NULL,
    detection_method           TEXT NOT NULL,       -- cold_start_isolation_forest | sequence_lstm
    predicted_type              TEXT,                 -- matches your explainability.py column name
    predicted_confidence         DOUBLE PRECISION,
    explanation                  TEXT,                 -- the plain-language reason string
    top_contributing_features    JSONB,                -- {"feature": z_score, ...}
    label                         TEXT,                 -- normal | anomaly | edge_case (ground truth)
    anomaly_type                  TEXT,
    scored_at                     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_scores_risk ON scores (risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_entity ON sessions (entity_id, ts DESC);

-- convert to a hypertable if the timescaledb extension is available
-- (safe to ignore the notice if it's already a hypertable or ts has nulls)
SELECT create_hypertable('sessions', 'ts', if_not_exists => TRUE, migrate_data => TRUE);
