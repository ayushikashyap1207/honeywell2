# AI-Powered Behavioural Anomaly Detection for Cybersecurity

Learns "normal" access patterns for users, service accounts, and edge devices, flags deviations, classifies the likely attack type, and explains every alert in plain language — instead of relying on fixed rules or a black-box score.

---

## What it does

- Learns behavior **per entity type** (user / service account / edge device), not one global rule
- Flags sessions that deviate from an entity's learned normal
- Classifies flagged sessions into 6 attack types (brute force, impossible travel, credential stuffing, lateral movement, device spoofing, low-and-slow exfiltration)
- Explains every alert with plain-language reasons + feature-level z-scores
- Handles **cold-start** entities (no history yet) and **concept drift** (normal behavior changing over time)

## Two ways to view results

| | Static dashboard | Live dashboard |
|---|---|---|
| Stack | Plain HTML/CSS/JS | React + FastAPI + Postgres |
| Setup needed | None | Docker + Node.js |
| Best for | Quick check, sharing, submission | Day-to-day live use |
| Location | `dashboard/index.html` | `dashboard-app/` |

Both read the same underlying pipeline output — pick whichever fits what you're doing.

---

## Quick Setup

### 1. Python environment
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the ML pipeline (produces the data everything else needs)
```bash
python generate_data.py --target_sessions 20000
python features.py
python train_baseline_model.py
python train_sequence_model.py
python combine_scores.py
python train_classifier.py
python explainability.py
python generate_dashboard.py
```
Open `dashboard/index.html` in any browser — done, no server needed.

### 3. (Optional) Live dashboard setup
```bash
# Start Postgres/TimescaleDB + Kafka + Redis + MLflow
docker compose up -d

# Create the DB tables (first time only)
psql "postgresql://anomaly:anomaly_pw@localhost:5432/anomaly_detection" -f db/schema.sql

# Load pipeline output into Postgres
python load_to_postgres.py

# Start the API
uvicorn api.main:app --reload --port 8000

# In a separate terminal — start the React dashboard
cd dashboard-app
npm install
npm run dev
```
Open the printed URL (usually `http://localhost:5173`). Header shows:
- **Live feed connected** → reading from the FastAPI/Postgres backend
- **Showing exported pipeline data** → backend not running, using last snapshot
- **Empty state** → run `python export_alerts_json.py` first

### 4. (Optional) Real-time streaming scoring
With Kafka/Redis running (from step 3) and the API server up:
```bash
# Terminal A — stateful feature computation (EWMA baseline, sliding window, per-entity Redis state)
python -m faust -A streaming.feature_app worker -l info --without-web

# Terminal B — live scoring (Isolation Forest cold-start + LSTM sequence, percentile-rank fusion)
python -m faust -A streaming.scoring_app worker -l info --without-web
```
Use `--without-web` — Faust's built-in web dashboard competes for a port with the FastAPI backend on this single-node setup. Scored sessions publish to Kafka (`scored_sessions`) and to Redis pub/sub (`live_scores`), which `/ws/alerts` forwards to the React dashboard in real time.

---

## Architecture

```
generate_data.py → features.py → train_baseline_model.py ─┐
                                → train_sequence_model.py ─┴→ combine_scores.py
                                                                    ↓
                                                          train_classifier.py
                                                                    ↓
                                                          explainability.py
                                                                    ↓
                                          ┌─────────────────────────┴───┐
                                          ▼                             ▼
                              generate_dashboard.py         export_alerts_json.py
                              (static HTML)                          ↓
                                                              load_to_postgres.py
                                                                       ↓
                                                          api/main.py (FastAPI)
                                                                       ↓
                                                          dashboard-app/ (React)
```

**Two detectors run in parallel:**
- **Isolation Forest** — no history needed, handles new/cold-start entities
- **LSTM Autoencoder** — needs session history, primary detector once available

Their scores aren't on the same scale, so they're merged as **percentile ranks**, not raw numbers.

## Tech Stack

| Layer | Choice |
|---|---|
| ML | scikit-learn (Isolation Forest, RandomForest), PyTorch (LSTM) |
| Database | PostgreSQL + TimescaleDB (Docker) |
| Backend API | FastAPI + SQLAlchemy |
| Live frontend | React + Vite |
| Static frontend | Plain HTML/CSS/JS |
| Streaming | Kafka, Redis, Faust (live) — MLflow, Prefect (planned, Phase 4) |
| Real data source | Okta System Log API |

## Why these design choices (short version)

- **Two detectors, not one** — a single model can't serve "entity with tons of history" and "brand-new entity" equally well.
- **Percentile-rank fusion, not raw score blending** — the two detectors' raw scores aren't comparable; ranking within each one's own population and merging *those* works better.
- **Classifier trained only on confirmed attacks** — ambiguous "edge case" rows are held out so they don't blur the model's decision boundary.
- **Z-scores for explainability, not a black-box tool** — "4 standard deviations from normal" is something an analyst can act on immediately.
- **EWMA baseline, not a frozen one** — lets normal behavior legitimately drift over time without triggering false alerts.
- **Metrics measured at the top-1% alert budget** — an analyst can only investigate so many alerts a day; that's what actually matters, not aggregate AUC alone.

## Project Structure

```
.
├── generate_data.py, features.py, train_*.py, combine_scores.py,
│   explainability.py, generate_dashboard.py, export_alerts_json.py   # ML pipeline
├── load_to_postgres.py, docker-compose.yml, db/schema.sql            # Live backend setup
├── api/main.py                                                       # FastAPI backend
├── dashboard-app/                                                    # React live dashboard
├── dashboard/index.html                                              # Static dashboard (generated)
├── okta_puller.py, kafka_producer.py                                 # Real data ingestion (Phase 2)
├── mlflow/Dockerfile                                                 # MLflow container build
├── data/, models/                                                    # Generated at runtime
└── requirements.txt
```

## Dataset

Synthetic access logs (`generate_data.py`), ground truth kept in a separate file from the data the models actually see — mirrors how this would work on real unlabeled traffic.

- 3 entity types with distinct behavior profiles: `user`, `service_account`, `edge_device`
- ~2% of sessions carry an injected attack pattern (7 types + 1 ambiguous edge case)
- ~8% of entities are deliberately cold-start (only 1–2 days of prior history)

## Models

| Model | Role | Saved as |
|---|---|---|
| Isolation Forest | Cold-start fallback | `models/baseline_isolation_forest.pkl` |
| LSTM Autoencoder | Primary sequence-aware detector | `models/sequence_lstm.pth` |
| RandomForest | Attack-type classifier | `models/anomaly_classifier.pkl` |

Both detectors train **only on normal sessions** — never on labeled attacks — mirroring real production use.

## Backend API

| Endpoint | Returns |
|---|---|
| `GET /api/summary` | Summary counts for the dashboard cards |
| `GET /api/alerts` | Ranked, filterable alert list |
| `GET /api/entity/{id}/history` | An entity's recent sessions |
| `WS /ws/alerts` | Pushes live scored sessions in real time via Redis pub/sub |

## Evaluation

Reported by the pipeline scripts, focused on the top-1% alert budget (what an analyst would actually see):

- **Precision@top-1%**, **Recall@top-1%**, **FPR@top-1%**
- PR-AUC (primary aggregate metric — more informative than ROC-AUC given the class imbalance)

Sample run at default settings: ~18,600 sessions scored, ~186 in the top-1% queue, ~89% precision within that queue. Exact numbers vary by run — re-run the pipeline for your own numbers.

---

## Project Status & Roadmap

**✅ Done — Layer 1: Batch ML pipeline** (everything above)

**✅ Done — Layer 2: Live dashboard** (Postgres + FastAPI + React on top of the pipeline)

**✅ Done — Layer 3, Phase 1: Local infra** — Kafka, Redis, MLflow, TimescaleDB all running via `docker compose up -d`. Used the official `apache/kafka` image (Bitnami's free tags were deprecated) and a custom `mlflow/Dockerfile` (no simple all-in-one MLflow image exists).

**✅ Done — Layer 3, Phase 2: Real data ingestion** — `okta_puller.py` pulls real events from a free Okta Developer org; `kafka_producer.py` publishes them to Kafka, keyed by entity for ordered per-entity processing.

What real data broke vs. synthetic assumptions:

| Assumed (synthetic) | Actually true (real Okta data) |
|---|---|
| Clean `user_XXXXXXXX` IDs | Opaque real Okta IDs — treat as opaque strings |
| One clean session duration | Okta logs events, not sessions — no duration field exists |
| Every session has geo data | ~78% missing geo (VPNs, proxies, system events) — not itself suspicious |
| One stable device fingerprint | Real user-agents drift (browser updates) |
| A single failure-count field | Each failed attempt is its own separate event |
| Clean 3-way entity type split | Real actor types don't map 1:1 — needs an "unknown" fallback |

**✅ Done — Layer 3, Phase 3: Streaming features + scoring** — the Kafka consumer-group coordination issue is resolved (single-node KRaft works cleanly once Faust's workers run with `--without-web`, avoiding a port conflict with the FastAPI backend). `streaming/feature_app.py` and `streaming/scoring_app.py` are live:
- Per-entity state (EWMA hour stats, haversine geo velocity, 5-session sliding window, cross-entity IP signals) now lives in Redis with a 24h TTL, replacing the batch CSV recompute
- Both detectors score live: Isolation Forest for cold-start entities, LSTM autoencoder once an entity's window fills, fused via percentile rank against a rolling Redis reference sample
- Scored sessions publish to Kafka (`scored_sessions`) and Redis pub/sub (`live_scores`)

**🔜 Remaining for Phase 3:** TimescaleDB live writer — streaming scores aren't yet landing in Postgres, so `/api/alerts` and `/api/summary` still only reflect the last batch load. Next step: have `scoring_app.py` write each scored session directly into the `scores` table alongside its existing Kafka/Redis outputs.

**✅ Done — Layer 3, Phase 5: Final wiring** — `/ws/alerts` now subscribes to the `live_scores` Redis channel and pushes real scored sessions to the dashboard as they happen, instead of being a static "connected" stub. Done ahead of Phase 4 since it only depended on Phase 3's scoring output existing.

**🔜 Phase 4: Self-improving system**
- MLflow model registry (proper versioning)
- Drift detection → automatic Prefect retraining
- Slack alerts for top-ranked live alerts


**🔜 Phase 5: Final wiring**
- Make `/ws/alerts` push real events instead of just a static "connected" status

## References

- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. ICDM.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Sakurada, M., & Yairi, T. (2014). *Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction*.
- scikit-learn: https://scikit-learn.org/stable/
- PyTorch: https://pytorch.org/docs/stable/
- FastAPI: https://fastapi.tiangolo.com/
- TimescaleDB: https://docs.timescale.com/