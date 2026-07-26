# AI-Powered Behavioural Anomaly Detection for Cybersecurity

A behavioural anomaly detection system that learns "normal" access patterns for users, service accounts, and edge devices from access-log data, flags deviations in near real time, classifies the likely attack type, and explains every alert in plain language for a SOC analyst — with explicit handling of cold-start entities and concept drift.

---

## Overview

This project develops an AI-powered behavioural anomaly detection system for cybersecurity by learning the normal access patterns of users, service accounts, and edge devices from historical access logs. Instead of relying on predefined rules or signature-based detection, the system identifies deviations from learned behavioural profiles to detect suspicious activities such as brute-force attacks, credential stuffing, impossible travel, lateral movement, device spoofing, and low-and-slow data exfiltration.

The solution combines a hybrid machine learning architecture consisting of an Isolation Forest for cold-start and single-session anomaly detection, an LSTM Autoencoder for sequence-aware behavioural analysis, and a Random Forest classifier for attack categorization. An explainability module provides feature-level reasoning for every alert, allowing security analysts to understand why a session was flagged instead of receiving only an anomaly score.

The entire pipeline is domain-agnostic and can be applied to enterprise users, service accounts, IoT devices, or industrial edge systems. It is implemented as a modular seven-stage workflow, where each stage is an independent Python script that reads and writes intermediate CSV artifacts. This modular design makes every processing step transparent, reproducible, and easy to debug while allowing individual components to be replaced or extended independently.

## Features

- **Synthetic data generator** — configurable target size, 3 entity types (user / service_account / edge_device), 7 injected attack patterns plus one ambiguous edge case, with ground-truth labels kept separate from the inference-time data.
- **Per-entity baseline profiling** — a standalone, inspectable "what does normal look like for this entity" artifact, built only from observed sessions.
- **Two-path detection** — a sequence-aware LSTM autoencoder for entities with enough history, and an Isolation Forest fallback for cold-start entities, merged into one comparable risk score via percentile-rank normalization.
- **Attack-type classification** — a supervised RandomForest predicts which of 6 attack categories a flagged session resembles.
- **Explainability layer** — every alert comes with a human-readable reason string and per-feature deviation scores, no black-box attribution needed.
- **Concept-drift-aware baseline** — an entity's "normal" login-hour profile adapts over time via an exponentially-weighted moving baseline, instead of anchoring forever to an entity's earliest sessions.
- **Cold-start handling** — entities with fewer than 5 sessions of history are routed to the Isolation Forest path automatically; every alert is traceable to which detection path produced it.
- **Self-contained analyst dashboard** — a single static HTML file: ranked, searchable/filterable alert queue with click-to-expand explanations and entity history, no server required.
- **Persisted models** — the Isolation Forest, LSTM autoencoder, and RandomForest classifier are all saved to disk after training, so scoring doesn't require re-running the full training pass every time.

## Demo

The dashboard (`dashboard/index.html`) opens directly in any browser once the pipeline has run:

- **Summary cards** — total sessions scored, alert budget (top 1%), precision within the queue, total true anomalies, entities monitored.
- **Ranked alert table** — rank, entity, entity type, risk score (with a colour-coded bar), detection path (Sequence LSTM vs. Cold-start Isolation Forest), predicted attack type, classifier confidence, and ground truth (for validation).
- **Search and filter** — by entity ID or predicted attack type.
- **Click-to-expand row** — shows the plain-language explanation, the top contributing features with their deviation (z-)scores, and the entity's last 8 raw sessions for context.




## Architecture

```
                ┌─────────────────────┐
                │  generate_data.py   │   synthetic access logs + ground truth
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │     features.py     │   per-entity deviation features + baseline profile
                └──────────┬──────────┘
                           │
              ┌────────────┴─────────────┐
              ▼                          ▼
  ┌───────────────────────┐  ┌────────────────────────────┐
  │ train_baseline_model  │  │  train_sequence_model.py    │
  │  Isolation Forest      │  │  LSTM Autoencoder            │
  │  (cold-start fallback) │  │  (sequence-aware, primary)   │
  └───────────┬───────────┘  └──────────────┬───────────────┘
              └─────────────┬────────────────┘
                           ▼
                ┌─────────────────────┐
                │  combine_scores.py   │   rank-normalized merge -> one risk score
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │ train_classifier.py  │   RandomForest: which attack type?
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │  explainability.py   │   z-score attribution -> reason strings
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │ generate_dashboard.py│   ranked, searchable analyst view
                └─────────────────────┘
```

Two detectors run in parallel rather than one model trying to do both jobs: the Isolation Forest is the cold-start fallback for entities without enough session history to form a sequence window; the LSTM autoencoder is the primary detector once that history exists. Their raw scores are **not** on a comparable scale, so `combine_scores.py` converts each to a percentile rank within its own scoring population before merging.


## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.9+ |
| Data handling | pandas, numpy |
| Classical ML | scikit-learn (Isolation Forest, RandomForest, StandardScaler, metrics) |
| Deep learning | PyTorch (LSTM autoencoder) |
| Model persistence | pickle (sklearn models/scalers), native PyTorch `state_dict` |
| Dashboard | Static HTML/CSS/vanilla JS, data embedded as JSON — no server or build step |
| Synthetic data | Python standard library (`random`, `uuid`, `argparse`, `datetime`, `math`) |

No external services, databases, or network access are required — everything runs locally against CSV files.

## Project Structure

```
.
├── generate_data.py          # Phase 1: synthetic access-log generator
├── features.py                # Phase 2: feature engineering + entity baseline profiles
├── train_baseline_model.py    # Phase 3a: Isolation Forest (cold-start fallback)
├── train_sequence_model.py    # Phase 3b: LSTM autoencoder (sequence-aware primary model)
├── combine_scores.py          # Phase 3c: rank-normalized score combination
├── train_classifier.py        # Phase 4: RandomForest attack-type classifier
├── explainability.py          # Phase 5: per-alert z-score explanations
├── generate_dashboard.py      # Phase 6: analyst dashboard (static HTML)
├── data/                       # generated at runtime — all intermediate CSVs
│   ├── access_logs_full_with_labels.csv
│   ├── access_logs_unlabeled.csv
│   ├── ground_truth_labels.csv
│   ├── entity_profiles.csv
│   ├── features.csv
│   ├── entity_baseline_profiles.csv
│   ├── baseline_scores.csv
│   ├── sequence_scores.csv
│   ├── final_scores.csv
│   ├── classified_alerts.csv
│   └── alerts_explained.csv
├── models/                     # generated at runtime — persisted trained models
│   ├── baseline_isolation_forest.pkl
│   ├── baseline_scaler.pkl
│   ├── anomaly_classifier.pkl
│   ├── sequence_lstm.pth
│   └── sequence_scaler.pkl
├── dashboard/
│   └── index.html              # generated at runtime — the analyst dashboard
└── README.md
```

## Requirements

- Python 3.9 or later
- pip
- ~500 MB free disk (synthetic data + models at default scale)
- No GPU required (PyTorch will use CUDA automatically if available, otherwise CPU)

**`requirements.txt`**
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
torch>=2.0
```

Install everything with:
```bash
pip install -r requirements.txt
```

## Installation

```bash
# 1. Clone or copy the project files into one folder
cd behavioural-anomaly-detection

# 2. Create the requirements file (see Requirements section above), then install
pip install -r requirements.txt
```

## Environment Setup

It's recommended to isolate dependencies in a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create the runtime folders (the scripts also create these automatically if missing):
```bash
mkdir -p data models dashboard
```

No API keys, credentials, or network access are needed — the entire pipeline runs offline against locally generated CSV files.

## Running the Project

Run each phase in order from the project root:

```bash
# Phase 1 — generate synthetic access logs (default ~20,000 sessions)
python generate_data.py --target_sessions 20000

# Phase 2 — feature engineering + entity baseline profiles
python features.py

# Phase 3a — cold-start fallback model (Isolation Forest)
python train_baseline_model.py

# Phase 3b — sequence-aware model (LSTM autoencoder)
python train_sequence_model.py

# Phase 3c — combine both detectors into one comparable risk score
python combine_scores.py

# Phase 4 — classify each flagged session's likely attack type
python train_classifier.py

# Phase 5 — attach a plain-language explanation to every alert
python explainability.py

# Phase 6 — build the analyst dashboard
python generate_dashboard.py
```

Then open `dashboard/index.html` directly in any browser — no server needed.

`--target_sessions` accepts anything between 50 and 100,000; the generator automatically scales the number of entities, simulated days, and injected-attack volume to approximately hit that total.

## Dataset

All data is synthetic, generated by `generate_data.py`, with ground-truth labels kept in a separate file from the inference-time data (mirroring how this would work against real unlabeled traffic).

**Schema** (`access_logs_unlabeled.csv`):

| Field | Description |
|---|---|
| `session_id` | Unique session identifier |
| `entity_id` | The user/service account/device this session belongs to |
| `entity_type` | `user`, `service_account`, or `edge_device` |
| `timestamp` | Access/connection time |
| `source_ip` | Origin IP of the access |
| `geo_location` | Named location (mapped internally to lat/long for distance calculations) |
| `resource_accessed` | Comma-separated resource(s) touched this session |
| `auth_method` | `password`, `token`, `certificate`, or `biometric` |
| `session_duration_sec` | Length of the session |
| `command_sequence` | Comma-separated ordered actions taken |
| `device_fingerprint` | OS/firmware + identifier |
| `failed_auth_attempts` | Count of failed auth attempts this session |

`ground_truth_labels.csv` holds `session_id`, `label` (`normal` / `anomaly` / `edge_case`), and `anomaly_type` — used only for training-slice selection and evaluation, never as a model input.

**Entity types** are given deliberately different behavioural profiles (e.g. service accounts run 10–40 sessions/day at all hours on certificate auth; edge devices touch only 1–3 usual resources; human users cluster around a personal login-hour mean with 2–6 sessions/day) — this is what makes per-entity, per-type baselining meaningful rather than trivial.

**Injected attack taxonomy** (~2% of sessions by default, concentrated in the most recent ~10 simulated days):

| Pattern | Simulation approach | Label |
|---|---|---|
| Brute force | Rapid repeated failed-auth attempts from one source IP in a short window | `anomaly` |
| Impossible travel | Same entity logging in from geographically distant locations within an implausible time gap | `anomaly` |
| Credential stuffing | Many entities, few source IPs, high failure rate | `anomaly` |
| Lateral movement | An entity accessing an unusual breadth of resources it never touched before | `anomaly` |
| Device spoofing | A device fingerprint reappearing with a mismatched OS/identifier | `anomaly` |
| Low-and-slow exfiltration | Gradual, small, off-hours resource access building up over days | `anomaly` |
| Insider drift | Legitimate-looking, slowly expanding resource footprint — ambiguous by design | `edge_case` |

A configurable slice of entities (~8%) are designated cold-start: they only have 1–2 days of prior "normal" history before the evaluation window, to exercise the cold-start path deliberately.

## Models

| Model | Role | File | Persisted as |
|---|---|---|---|
| Isolation Forest | Cold-start fallback — single-session risk score for entities without enough history for a sequence window | `train_baseline_model.py` | `models/baseline_isolation_forest.pkl`, `models/baseline_scaler.pkl` |
| LSTM Autoencoder | Primary sequence-aware detector — reconstructs 5-session windows, reconstruction error = anomaly score | `train_sequence_model.py` | `models/sequence_lstm.pth`, `models/sequence_scaler.pkl` |
| RandomForest Classifier | Predicts which of 6 attack types a session most resembles | `train_classifier.py` | `models/anomaly_classifier.pkl` |

Both the Isolation Forest and the LSTM autoencoder are trained **only on rows labelled `normal`** — never on labelled attacks — mirroring how this would work against real unlabeled live traffic where labels exist only for evaluation. The classifier is the exception: it's supervised, trained only on confirmed `anomaly` rows (the `edge_case`/`insider_drift` rows are deliberately held out as an ambiguous sanity check rather than used for training).

`combine_scores.py` merges the two detectors by converting each one's raw score to a **percentile rank within its own scoring population** before merging — mixing the two raw scores directly was found to make the combined ranking worse than either detector alone, since the two models' raw scores aren't on a comparable scale.

## Dashboard

`generate_dashboard.py` builds a single self-contained HTML file — chosen over a framework like Streamlit so it needs no server, build step, or network access, and packages easily for submission. It shows:

- The top-1%-by-risk alert queue (configurable via `ALERT_BUDGET_PCT`)
- Rank, entity, entity type, risk score, detection path, predicted attack type, classifier confidence, and ground truth
- Search-by-entity and filter-by-predicted-type
- A click-to-expand detail view per alert: the explanation, top contributing (z-scored) features, and the entity's last 8 raw sessions

## Evaluation Metrics

Reported at every scoring stage (`train_baseline_model.py`, `train_sequence_model.py`, `combine_scores.py`), evaluated strictly on `normal` vs. `anomaly` rows (`edge_case` rows are set aside as ambiguous by design):

- **PR-AUC** — the primary metric given the extreme class imbalance (ROC-AUC alone is optimistic under imbalance)
- **ROC-AUC**
- **Precision@top-1%** — of the top 1% highest-risk sessions, what fraction are true anomalies
- **Recall@top-1%** — of all true anomalies, what fraction are captured in that top-1% budget
- **FPR@top-1%** — false positive rate within that same alert budget, named directly in the assessment's own evaluation criteria

For attack-type classification (`train_classifier.py`): a full classification report (precision/recall/F1 per class) and a confusion matrix on held-out anomaly rows, plus global RandomForest feature importances as a cheap cross-check against the per-alert explainability layer.

## Results

Illustrative figures from a sample run at the default `--target_sessions 20000` — exact numbers will vary with dataset size, random seed, and any code changes since:

| Metric | Value |
|---|---|
| Total sessions scored | ~18,600 |
| Entities monitored | ~32 |
| True anomalies (all data) | ~400 |
| Alert budget (top 1%) | ~186 sessions |
| Precision within the top-1% queue | ~89% |

Re-run the pipeline and read the printed metrics from each script (`train_baseline_model.py`, `train_sequence_model.py`, `combine_scores.py`, `train_classifier.py`) for the authoritative numbers on your own generated dataset.

## Future Improvements

- **Model-level drift retraining** — the Isolation Forest and LSTM are each a one-shot static fit; production use should refit both periodically (e.g. weekly on a trailing window of confirmed-normal traffic) so population-level drift doesn't slowly stale them out.
- **Real-time/streaming scoring** — replace the current batch CSV-in/CSV-out flow with a stateful stream processor keyed by `entity_id`, scoring each session as it arrives using the already-persisted models, and approximating `combine_scores.py`'s percentile-rank merge against a periodically-refreshed reference sample instead of the full batch.
- **Entity-level alert diversity** — the current top-1% queue can be dominated by one entity's sustained attack burst; an entity-level cap (or grouping repeated alerts from the same entity into one row with a count) would surface a more diverse set of at-risk entities to the analyst.
- **Per-entity-type feature scaling** — some raw features (e.g. session duration, command-sequence length) are fed to the models as absolute values even though entity types are deliberately different in scale by design; normalizing these per entity (the way `hour_z` already is) or explicitly encoding entity type as a model feature would reduce the risk of the models learning "entity type" instead of genuine deviation.
- **Presentation-ready explainability export** — a PDF/slide export of the top alerts and their explanations for handoff to non-technical stakeholders.
- **Alerting integration** — push top-ranked alerts to a real notification channel (email/Slack/webhook) instead of requiring an analyst to open the dashboard.

## References

- Isolation Forest: Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. ICDM.
- LSTM: Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Autoencoders for anomaly detection: Sakurada, M., & Yairi, T. (2014). *Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction*.
- scikit-learn documentation: https://scikit-learn.org/stable/
- PyTorch documentation: https://pytorch.org/docs/stable/
