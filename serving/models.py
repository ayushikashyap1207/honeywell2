"""Model loading and inference shared by the streaming worker.

Models are trained offline by ``train_baseline_model.py``,
``train_sequence_model.py`` and ``train_classifier.py``; this module only
applies them. Keeping the architecture and feature order in one place avoids
the batch/streaming drift that duplicated copies caused before.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from serving.explain import FEATURE_COLS, explain_row, load_baseline

# Isolation Forest sees the cold-start flag; the LSTM deliberately does not
# (it is a history-depth flag, not behaviour, and windows already require
# enough history to exist).
FEATURE_ORDER_IF = list(FEATURE_COLS)
FEATURE_ORDER_SEQ = [feature for feature in FEATURE_COLS if feature != "is_cold_start"]

SEQ_LEN = 5
HIDDEN_DIM = 16
LATENT_DIM = 8


class LSTMAutoencoder(nn.Module):
    """Must match ``train_sequence_model.py`` exactly for state_dict loading."""

    def __init__(
        self,
        n_features,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        seq_len=SEQ_LEN,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(hidden[-1])
        dec_input = self.decoder_fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_input)
        return self.output_fc(dec_out)


@dataclass
class ModelBundle:
    iso_forest: object | None = None
    iso_scaler: object | None = None
    sequence_model: LSTMAutoencoder | None = None
    sequence_scaler: object | None = None
    classifier: object | None = None
    baseline_stats: dict | None = None


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_models(models_dir="models", baseline_path="models/explanation_baseline.json"):
    """Load every artifact that is present. Missing models degrade gracefully."""
    models_path = Path(models_dir)
    bundle = ModelBundle()

    try:
        bundle.iso_forest = _load_pickle(models_path / "baseline_isolation_forest.pkl")
        bundle.iso_scaler = _load_pickle(models_path / "baseline_scaler.pkl")
        print("Loaded Isolation Forest model + scaler")
    except Exception as exc:  # noqa: BLE001 - startup should not crash on a missing model
        print(f"WARNING: could not load Isolation Forest -- {exc}")

    try:
        sequence_model = LSTMAutoencoder(n_features=len(FEATURE_ORDER_SEQ))
        state_dict = torch.load(
            models_path / "sequence_lstm.pth",
            map_location="cpu",
        )
        sequence_model.load_state_dict(state_dict)
        sequence_model.eval()
        bundle.sequence_model = sequence_model
        bundle.sequence_scaler = _load_pickle(models_path / "sequence_scaler.pkl")
        print("Loaded LSTM Autoencoder + scaler")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not load LSTM Autoencoder -- {exc}")

    try:
        bundle.classifier = _load_pickle(models_path / "anomaly_classifier.pkl")
        print("Loaded attack-type classifier")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not load attack-type classifier -- {exc}")

    try:
        bundle.baseline_stats = load_baseline(baseline_path)
        if bundle.baseline_stats:
            print("Loaded explanation baseline")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not load explanation baseline -- {exc}")

    return bundle


def _number(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(number) or np.isinf(number):
        return default
    return number


def score_isolation_forest(bundle: ModelBundle, feature_row) -> float:
    """Higher = more anomalous, matching the batch convention."""
    vector = [_number(feature_row.get(feature)) for feature in FEATURE_ORDER_IF]
    frame = pd.DataFrame([vector], columns=FEATURE_ORDER_IF)
    scaled = bundle.iso_scaler.transform(frame)
    raw = bundle.iso_forest.decision_function(scaled)[0]
    return float(-raw)


def score_lstm(bundle: ModelBundle, window) -> float:
    """Reconstruction MSE for one entity's 5-session window."""
    vectors = [
        [_number(session.get(feature)) for feature in FEATURE_ORDER_SEQ]
        for session in window
    ]
    frame = pd.DataFrame(vectors, columns=FEATURE_ORDER_SEQ)
    scaled = bundle.sequence_scaler.transform(frame)
    tensor = torch.tensor(np.array([scaled]), dtype=torch.float32)
    with torch.no_grad():
        reconstruction = bundle.sequence_model(tensor)
        mse = torch.mean((reconstruction - tensor) ** 2).item()
    return float(mse)


def classify_row(bundle: ModelBundle, feature_row):
    """Return ``(attack_type, confidence)`` or ``(None, None)``."""
    if bundle.classifier is None:
        return None, None
    vector = [_number(feature_row.get(feature)) for feature in FEATURE_ORDER_IF]
    frame = pd.DataFrame([vector], columns=FEATURE_ORDER_IF)
    predicted = bundle.classifier.predict(frame)[0]
    confidence = float(bundle.classifier.predict_proba(frame)[0].max())
    return str(predicted), confidence


def explain(bundle: ModelBundle, feature_row):
    """Explain one live session, or degrade to a neutral message."""
    if not bundle.baseline_stats:
        return "Flagged by the model; explanation baseline unavailable.", {}
    return explain_row(feature_row, bundle.baseline_stats)
