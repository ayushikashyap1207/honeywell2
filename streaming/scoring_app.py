"""
Streaming scoring — the live equivalent of train_baseline_model.py /
train_sequence_model.py / combine_scores.py's *inference* side (not
training — your models are already trained offline, this just applies
them live).

Reads computed feature rows from session_features, scores each one with
whichever detector applies (Isolation Forest for cold-start, LSTM for
entities with a full 5-session window), merges the two onto one comparable
0-1 risk score via a rolling Redis reference sample (the streaming
equivalent of combine_scores.py's batch percentile rank), and writes the
result to scored_sessions.

Run (separate terminal from feature_app.py — both run at once):
    python -m faust -A streaming.scoring_app worker -l info

Model format (confirmed from your actual train_baseline_model.py /
train_sequence_model.py):
- baseline_isolation_forest.pkl: the raw IsolationForest object, scaler
  saved separately as baseline_scaler.pkl
- sequence_lstm.pth: a state_dict (not the full model), reconstructed here
  using the exact LSTMAutoencoder class from train_sequence_model.py, with
  its scaler saved separately as sequence_scaler.pkl
"""
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import faust
import redis

app = faust.App(
    "anomaly-scoring",
    broker="kafka://localhost:9092",
    value_serializer="json",
    topic_replication_factor=1,
)

features_topic = app.topic("session_features")
scored_topic = app.topic("scored_sessions", partitions=6)

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Must match the feature order features.py / feature_app.py used for training.
# is_cold_start is included for the Isolation Forest but excluded for the
# LSTM (per your docs, section 6.2 — it's a history-depth flag, not behaviour).
FEATURE_ORDER = [
    "hour_z", "resource_novelty", "n_resources_this_session", "device_novel",
    "geo_km_per_hr", "session_duration_sec", "cmd_seq_len",
    "failed_auth_attempts", "ip_failed_5min", "ip_unique_entities_5min",
]

REFERENCE_SAMPLE_SIZE = 2000


def load_isolation_forest():
    with open("models/baseline_isolation_forest.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/baseline_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# Exact architecture from train_sequence_model.py — must match precisely for
# load_state_dict() to work, since PyTorch state_dicts have no shape info
# of their own to reconstruct the model from.
SEQ_LEN = 5
HIDDEN_DIM = 16
LATENT_DIM = 8


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(h_n[-1])
        dec_input = self.decoder_fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_input)
        recon = self.output_fc(dec_out)
        return recon


def load_lstm():
    model = LSTMAutoencoder(n_features=len(FEATURE_ORDER))
    state_dict = torch.load("models/sequence_lstm.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with open("models/sequence_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


try:
    iso_forest_model, iso_forest_scaler = load_isolation_forest()
    print("Loaded Isolation Forest model + scaler")
except Exception as e:
    iso_forest_model, iso_forest_scaler = None, None
    print(f"WARNING: could not load Isolation Forest — {e}")

try:
    sequence_model, sequence_scaler = load_lstm()
    print("Loaded LSTM Autoencoder + scaler")
except Exception as e:
    sequence_model, sequence_scaler = None, None
    print(f"WARNING: could not load LSTM model — {e}")


def score_isolation_forest(feature_row: dict) -> float:
    vec = [feature_row[f] for f in FEATURE_ORDER] + [feature_row["is_cold_start"]]
    X = iso_forest_scaler.transform([vec])
    raw = iso_forest_model.decision_function(X)[0]
    return -raw  # higher = more anomalous, matches your batch convention


def score_lstm(window: list[dict]) -> float:
    vecs = [[w[f] for f in FEATURE_ORDER] for w in window]
    vecs_scaled = sequence_scaler.transform(vecs)  # same StandardScaler used at training time
    x = torch.tensor(np.array([vecs_scaled]), dtype=torch.float32)
    with torch.no_grad():
        reconstruction = sequence_model(x)
        mse = torch.mean((reconstruction - x) ** 2).item()
    return mse


def rank_against_reference(raw_score: float, detector_name: str) -> float:
    key = f"ref_scores:{detector_name}"
    member = f"{raw_score}:{r.incr('score_counter')}"
    r.zadd(key, {member: raw_score})
    r.zremrangebyrank(key, 0, -REFERENCE_SAMPLE_SIZE - 1)

    all_scores = [score for _, score in r.zrange(key, 0, -1, withscores=True)]
    if len(all_scores) < 10:
        return 0.5  # not enough reference data yet
    rank = sum(1 for s in all_scores if s <= raw_score) / len(all_scores)
    return rank


@app.agent(features_topic)
async def score_sessions(features):
    async for raw in features:
        feature_row = json.loads(raw) if isinstance(raw, (str, bytes)) else raw

        if feature_row.get("is_cold_start") and iso_forest_model is not None:
            raw_score = score_isolation_forest(feature_row)
            pct_rank = rank_against_reference(raw_score, "isolation_forest")
            detection_method = "cold_start_isolation_forest"
        elif feature_row.get("window_ready") and sequence_model is not None:
            raw_score = score_lstm(feature_row["window"])
            pct_rank = rank_against_reference(raw_score, "lstm")
            detection_method = "sequence_lstm"
        else:
            continue  # not enough window yet, and not flagged cold-start

        result = {
            "session_id": feature_row["session_id"],
            "entity_id": feature_row["entity_id"],
            "entity_type": feature_row.get("entity_type"),
            "risk_score": pct_rank,
            "detection_method": detection_method,
        }
        print(f"Scored {result['session_id']}: {pct_rank:.3f} via {detection_method}")
        await scored_topic.send(key=feature_row["entity_id"], value=result)
        r.publish("live_scores", json.dumps(result))


if __name__ == "__main__":
    app.main()
