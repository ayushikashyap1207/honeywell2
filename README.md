# AI-Powered Behavioural Anomaly Detection for Cybersecurity

Learns "normal" access patterns for users, service accounts, and edge devices, flags deviations, classifies the likely attack type, and explains every alert in plain language — instead of relying on fixed rules or a black-box score.

---

## What it does

- Learns behavior **per entity type** (user / service account / edge device), not one global rule
- Flags sessions that deviate from an entity's learned normal
- Classifies flagged sessions into 6 attack types (brute force, impossible travel, credential stuffing, lateral movement, device spoofing, low-and-slow exfiltration)
- Explains every alert with plain-language reasons + feature-level z-scores
- Handles **cold-start** entities (no history yet) and **concept drift** (normal behavior changing over time)
- **Calibrated risk**: isotonic regression per detector + conformal p-values, so "0.93" means a probability, not just a percentile
- **Counterfactual explanations**: "if this feature were normal, the anomaly score would fall from X to Y"
- **Evaluation lab**: PR/ROC curves, per-attack recall, cold-start vs warm slices, and comparisons against PCA, One-Class SVM, Isolation Forest and supervised upper bounds
- **Drift monitoring**: PSI + KS reports with a dashboard tab and a retrain trigger
- **Analyst feedback loop**: confirm / false-positive / escalate dispositions feed the next classifier retrain
- **Grounded SOC copilot**: retrieval over similar incidents, entity history, MITRE ATT&CK and model reports; deterministic answers work without an LLM, Ollama/OpenAI optional
- **MLflow + Prefect**: experiment/model registry tracking and an orchestrated retrain flow

## Two ways to view results

| | Static dashboard | Live workspace |
|---|---|---|
| Stack | Plain HTML/CSS/JS | React + FastAPI + Postgres + copilot |
| Setup needed | None | Docker (`make demo`) |
| Best for | Quick check, sharing, submission | Alerts, evaluation, drift, copilot |
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
python -m calibration.calibrate          # isotonic risk calibration + conformal p-values
python train_classifier.py
python explainability.py
python evaluation.py                     # baseline comparison + dashboard report
python -m monitoring.drift               # PSI/KS drift report
python explainability_shap.py            # SHAP when installed, RF-importance fallback
python track_experiment.py               # MLflow (or local data/mlflow_run.json)
python generate_dashboard.py
```
Open `dashboard/index.html` in any browser — done, no server needed.

`make reports` runs the calibration/evaluation/drift/attribution refresh in the
right order.

### 3. One-command live demo (recommended)

Docker is the fastest way to see the whole system running end to end:

```bash
make demo
```

This starts TimescaleDB, Redis, Kafka, the FastAPI backend, both Faust workers, and the React dashboard, seeds the database with the batch pipeline output, and replays a slice of synthetic events through the live path. The first run builds a CPU-only PyTorch image, so it can take a few minutes.

- Dashboard: `http://localhost:8080`
- API docs: `http://localhost:8000/docs`
- `make logs` — follow worker/API logs
- `make migrate` — apply schema updates to an existing database volume
- `make down` — stop everything
- `make clean` — stop and delete volumes (fresh database next time)

Every scored live session is written to both Postgres (`sessions` + `scores`) and Redis pub/sub, so the REST endpoints and the WebSocket feed reflect real-time traffic — not just the last batch load.

The dashboard has four tabs:

- **Alert queue** — ranked live alerts, calibrated risk, counterfactual, feedback buttons
- **Model evaluation** — PR/ROC curves, baseline comparison, per-attack and cold-start metrics
- **Drift** — PSI/KS feature and score drift
- **Copilot** — evidence-grounded chat over alerts, history, MITRE and model reports

Analyst feedback is stored in Postgres. If you are upgrading an existing volume, run `make migrate` first; a fresh `make demo` applies the schema automatically.

### 4. (Optional) Manual live dashboard setup
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

### 5. (Optional) Real-time streaming scoring without Docker
With Kafka/Redis running (from step 4) and the API server up:
```bash
# Terminal A — stateful feature computation (EWMA baseline, sliding window, per-entity Redis state)
python -m faust -A streaming.feature_app worker -l info --without-web

# Terminal B — live scoring (Isolation Forest cold-start + LSTM sequence, percentile-rank fusion, Postgres writes)
python -m faust -A streaming.scoring_app worker -l info --without-web

# Terminal C — replay synthetic events so the live path has traffic
python streaming/replay.py --limit 5000 --delay-ms 25
```
Use `--without-web` — Faust's built-in web dashboard competes for a port with the FastAPI backend on this single-node setup. Scored sessions publish to Kafka (`scored_sessions`), Redis pub/sub (`live_scores`), and Postgres (`sessions` + `scores`), which `/ws/alerts` forwards to the React dashboard in real time.

### 6. Copilot, feedback and retraining

The copilot works with **no LLM configured**: it answers deterministically from the same evidence packet and still cites sources. For generated prose:

```bash
# Option A — local Ollama (no API key)
ollama pull llama3.2
export COPILOT_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434   # use host.docker.internal inside Compose
export OLLAMA_MODEL=llama3.2

# Option B — OpenAI-compatible API
export COPILOT_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
```

Restart the API after changing provider env vars.

Feedback and retraining:

```bash
# Analyst confirms/false-positives are stored in Postgres by the dashboard.
python apply_feedback.py            # export labels for the next classifier run
python flows/retrain.py --force     # Prefect flow: drift check -> retrain -> recalibrate -> reports -> MLflow
make index                          # refresh the alert_embeddings vector index
```

MLflow defaults to a local SQLite store (`sqlite:///mlflow.db`). With Compose running, use `MLFLOW_TRACKING_URI=http://localhost:5000 python track_experiment.py`.

---

## Architecture

```
Batch:
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

Live:
replay.py / okta_puller.py / kafka_producer.py
        ↓
Kafka: raw_sessions
        ↓
streaming/feature_app.py  (Redis per-entity state)
        ↓
Kafka: session_features
        ↓
streaming/scoring_app.py  (Isolation Forest / LSTM + classifier + explanation)
        ↓                             ↓
Kafka: scored_sessions      Postgres/TimescaleDB: sessions + scores
                                      ↓
                          api/main.py (FastAPI REST + WebSocket)
                                      ↓
                          dashboard-app/ (React live dashboard)

Model / ops loop:
combine_scores.py → calibration/calibrate.py → train_classifier.py
        → explainability.py → evaluation.py → monitoring/drift.py
        → explainability_shap.py → track_experiment.py (MLflow)

analyst feedback (API) → apply_feedback.py → flows/retrain.py (Prefect)
        → retrain → recalibrate → reevaluate → register model

Copilot:
api/copilot_routes.py → copilot/agent.py
        → tools (alert, entity history, search, similar, evaluation, drift, MITRE)
        → vector index (alert_embeddings) + Ollama/OpenAI or deterministic fallback
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
| Streaming | Kafka, Redis, Faust (live) |
| Model ops | MLflow (tracking + registry), Prefect (retrain flow), PSI/KS drift |
| Copilot | OpenAI-compatible chat (Ollama/OpenAI), hashed local embeddings, JSONB vector index (pgvector-ready) |
| Packaging / CI | Docker, Docker Compose, GitHub Actions, pytest |
| Real data source | Okta System Log API |

## Why these design choices (short version)

- **Two detectors, not one** — a single model can't serve "entity with tons of history" and "brand-new entity" equally well.
- **Percentile-rank fusion, then calibration** — the two detectors' raw scores aren't comparable; ranks merge them, then isotonic regression maps each detector's rank to a probability. A conformal p-value adds uncertainty without assuming a distribution.
- **Classifier trained only on confirmed attacks** — ambiguous "edge case" rows are held out so they don't blur the model's decision boundary. Analyst-confirmed alerts can be added later through the feedback loop.
- **Z-scores + counterfactuals for explainability** — "4 standard deviations from normal" is actionable, and "if this feature were normal, the score would drop" is even more so. SHAP is used offline when installed.
- **EWMA baseline, not a frozen one** — lets normal behavior legitimately drift over time without triggering false alerts.
- **Metrics measured at the top-1% alert budget** — an analyst can only investigate so many alerts a day; that's what actually matters, not aggregate AUC alone.
- **Copilot with citations, not free-form chat** — the LLM only sees a bounded evidence packet, must cite evidence IDs, and falls back to deterministic answers when no provider is configured. Prompt-injection phrases are ignored.

## Project Structure

```
.
├── generate_data.py, features.py, train_*.py, combine_scores.py,
│   explainability.py, generate_dashboard.py, export_alerts_json.py   # ML pipeline
├── serving/                                                          # Shared inference + explanation code
│   ├── models.py                                                     # Model loading / scoring / classifier
│   ├── explain.py                                                    # One explanation implementation for batch + live
│   ├── calibration.py                                                # Isotonic + conformal calibration
│   ├── attribution.py                                                # Counterfactual re-scoring
│   └── geo.py                                                        # Shared geo-velocity helpers
├── calibration/calibrate.py                                          # Fit + evaluate calibrators
├── evaluation.py                                                     # Baseline comparison + chart JSON
├── monitoring/drift.py                                               # PSI/KS drift report
├── explainability_shap.py                                            # SHAP or RF-importance attribution
├── track_experiment.py                                               # MLflow tracking + model registry
├── apply_feedback.py                                                 # Feedback → training labels
├── flows/retrain.py                                                  # Prefect retrain flow
├── copilot/                                                          # Grounded SOC copilot
│   ├── agent.py                                                      # Tool routing + LLM/fallback
│   ├── retrieval.py                                                  # Vector index + similar incidents
│   ├── embeddings.py                                                 # Local hashed embeddings
│   ├── tools.py                                                      # Read-only evidence tools
│   └── mitre.py                                                      # MITRE ATT&CK + playbooks
├── streaming/                                                        # Faust feature + scoring workers
│   ├── feature_app.py
│   ├── scoring_app.py                                                # Scores + writes to Postgres
│   └── replay.py                                                     # Demo event replay into Kafka
├── api/                                                              # FastAPI backend + analytics/copilot routes
├── dashboard-app/                                                    # React workspace (alerts/eval/drift/copilot)
├── dashboard/index.html                                              # Static dashboard (generated)
├── docker/                                                           # API/streaming/dashboard/seed images + nginx
├── tests/                                                            # Unit tests (pytest)
├── load_to_postgres.py, docker-compose.yml, db/schema.sql            # Live backend setup
├── okta_puller.py, kafka_producer.py                                 # Real data ingestion
├── Makefile                                                          # make demo / reports / retrain / test
├── .env.example                                                      # Configuration template
├── mlflow/Dockerfile                                                 # MLflow container build
├── data/, models/                                                    # Generated at runtime
└── requirements*.txt
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
| Explanation baseline | Per-entity-type mean/std used by batch + live z-scores | `models/explanation_baseline.json` |
| Calibrators | Per-detector isotonic model + normal scores for conformal p-values | `models/calibrators.pkl` |

Both detectors train **only on normal sessions** — never on labeled attacks — mirroring real production use.

## Backend API

| Endpoint | Returns |
|---|---|
| `GET /api/summary` | Summary counts for the dashboard cards |
| `GET /api/alerts` | Ranked, filterable alert list |
| `GET /api/entity/{id}/history` | An entity's recent sessions |
| `GET /api/health` | Liveness/readiness probe used by Docker and CI |
| `GET /api/evaluation` | Model-comparison, curves, per-attack and cold-start report |
| `GET /api/drift` | Latest PSI/KS drift report |
| `GET /api/shap` | Global feature-attribution report |
| `GET /api/mlflow` | Last MLflow run summary |
| `POST /api/feedback` | Analyst disposition (`confirm`, `false_positive`, `escalate`, `acknowledge`) |
| `GET /api/feedback/summary` | Feedback counts + retrain queue |
| `POST /api/copilot/chat` | Evidence-grounded copilot answer with citations |
| `GET /api/copilot/status` | Provider/model/deterministic mode |
| `GET /api/copilot/similar` | Similar incidents from the vector index |
| `GET /api/copilot/mitre/{type}` | MITRE mapping + response playbook |
| `WS /ws/alerts` | Pushes live scored sessions in real time via Redis pub/sub |

## Evaluation

`evaluation.py` holds out 30% of the labeled data, trains unsupervised baselines on normal training rows only, and reports everything at the **top-1% analyst alert budget**. `calibration/calibrate.py` separately reports Brier score / ECE and conformal coverage.

Sample run at default settings (~18,600 sessions, 18,639 rows scored in the batch):

| Model | PR-AUC | ROC-AUC | Precision@1% | Recall@1% | FPR@1% |
|---|---|---|---|---|---|
| Random forest (supervised upper bound) | 0.995 | 0.9999 | 1.000 | 0.455 | 0.000 |
| Logistic regression (supervised) | 0.952 | 0.986 | 1.000 | 0.455 | 0.000 |
| **Hybrid IF+LSTM (pipeline)** | **0.728** | **0.983** | **0.873** | **0.397** | **0.0013** |
| One-Class SVM | 0.710 | 0.968 | 0.945 | 0.430 | 0.0006 |
| Hybrid + isotonic calibration | 0.704 | 0.989 | 0.873 | 0.397 | 0.0013 |
| PCA reconstruction | 0.616 | 0.953 | 0.745 | 0.339 | 0.0026 |
| Isolation Forest | 0.494 | 0.973 | 0.582 | 0.265 | 0.0042 |

Calibration reduces expected calibration error to **0.001** and Brier score to **0.0096** on the held-out set. The supervised models are an upper bound, not deployed detectors — the deployed pipeline never trains on labeled attacks.

Per-attack recall, cold-start vs warm slices, reliability curves and feature attribution are all available in the **Model evaluation** dashboard tab and in `data/evaluation_report.json`. Exact numbers vary by run — re-run `make reports` for your own.

---

## References

- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. ICDM.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Sakurada, M., & Yairi, T. (2014). *Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction*.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- scikit-learn: https://scikit-learn.org/stable/
- PyTorch: https://pytorch.org/docs/stable/
- FastAPI: https://fastapi.tiangolo.com/
- TimescaleDB: https://docs.timescale.com/
