-- Run this once against your running TimescaleDB container:
--   psql "postgresql://anomaly:anomaly_pw@localhost:5432/anomaly_detection" -f db/schema.sql
--
-- The Docker Compose demo mounts this file into the Postgres init directory,
-- so a fresh `docker compose up` creates the tables automatically.

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS sessions (
    session_id              TEXT PRIMARY KEY,
    entity_id               TEXT NOT NULL,
    entity_type             TEXT NOT NULL,
    ts                      TIMESTAMPTZ,
    source_ip               TEXT,
    geo_location            TEXT,
    resource_accessed       TEXT,
    auth_method             TEXT,
    session_duration_sec    DOUBLE PRECISION
);

-- Keep existing databases compatible without a manual migration.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS source_ip TEXT;

CREATE TABLE IF NOT EXISTS scores (
    session_id                TEXT PRIMARY KEY REFERENCES sessions(session_id),
    entity_id                 TEXT NOT NULL,
    entity_type               TEXT,
    risk_score                DOUBLE PRECISION NOT NULL,
    calibrated_risk           DOUBLE PRECISION,
    conformal_p_value         DOUBLE PRECISION,
    raw_score                 DOUBLE PRECISION,
    detection_method          TEXT NOT NULL,       -- cold_start_isolation_forest | sequence_lstm
    predicted_type            TEXT,                 -- attack type from the classifier
    predicted_confidence      DOUBLE PRECISION,
    explanation               TEXT,                 -- plain-language reason string
    explanation_details       JSONB,                -- z-scores, SHAP, counterfactual
    top_contributing_features JSONB,                -- {"feature": z_score, ...}
    label                     TEXT,                 -- normal | anomaly | edge_case (ground truth)
    anomaly_type              TEXT,
    scored_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE scores ADD COLUMN IF NOT EXISTS raw_score DOUBLE PRECISION;
ALTER TABLE scores ADD COLUMN IF NOT EXISTS calibrated_risk DOUBLE PRECISION;
ALTER TABLE scores ADD COLUMN IF NOT EXISTS conformal_p_value DOUBLE PRECISION;
ALTER TABLE scores ADD COLUMN IF NOT EXISTS explanation_details JSONB;

CREATE TABLE IF NOT EXISTS feedback (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES scores(session_id) ON DELETE CASCADE,
    verdict         TEXT NOT NULL CHECK (verdict IN ('confirm', 'false_positive', 'escalate', 'acknowledge')),
    note            TEXT,
    analyst         TEXT,
    used_in_training BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_pending ON feedback (used_in_training) WHERE used_in_training = FALSE;

CREATE TABLE IF NOT EXISTS alert_embeddings (
    session_id      TEXT PRIMARY KEY REFERENCES scores(session_id) ON DELETE CASCADE,
    content         TEXT NOT NULL,
    embedding       JSONB NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_scores_risk ON scores (risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_scores_calibrated ON scores (calibrated_risk DESC);
CREATE INDEX IF NOT EXISTS idx_scores_scored_at ON scores (scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_entity ON sessions (entity_id, ts DESC);

-- Convert sessions to a hypertable when the extension is available. A plain
-- primary key on session_id is not compatible with time partitioning on some
-- TimescaleDB versions, so downgrade gracefully instead of failing init.
DO $$
BEGIN
    PERFORM create_hypertable('sessions', 'ts', if_not_exists => TRUE, migrate_data => TRUE);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'sessions left as a regular table: %', SQLERRM;
END $$;
