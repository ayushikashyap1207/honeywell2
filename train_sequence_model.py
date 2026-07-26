"""
Phase 3b: Sequence-Aware Detection Model (LSTM autoencoder)
-----------------------------------------------------------------
This is a sequence-aware model that learns the "normal trajectory" of an entity's sessions over time, and flags windows of sessions that deviate from that learned trajectory. It is trained on a sliding window of consecutive sessions for each entity, using only windows where all sessions are labeled normal. The model is an LSTM autoencoder: the encoder compresses a window of session feature vectors into a latent vector, and the decoder reconstructs the window from that latent vector. The reconstruction error (MSE) is used as the anomaly score for each window.
The model is trained on normal-only windows, and at inference, windows that do not resemble the learned "normal trajectory" reconstruct poorly, resulting in a high reconstruction error, which is used as the anomaly score. The anomaly scores are then rank-normalized to a 0-1 scale, where higher = more anomalous. This score is used in conjunction with the Phase 3a Isolation Forest baseline model, which serves as a fallback for cold-start sessions with no prior history. The final risk score for each session is a combination of the sequence-aware model's score and the baseline model's score  
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURE_COLS = [
    "hour_z", "resource_novelty", "n_resources_this_session", "device_novel",
    "geo_km_per_hr", "session_duration_sec", "cmd_seq_len", "failed_auth_attempts",
    "ip_failed_5min", "ip_unique_entities_5min",
]  # is_cold_start deliberately excluded: it's a flag about history depth, not
   # behaviour, and windows already require enough history to exist.

SEQ_LEN = 5          # sessions per window
HIDDEN_DIM = 16
LATENT_DIM = 8
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)


class LSTMAutoencoder(nn.Module):
    """Encoder LSTM compresses a window of session vectors to a latent vector;
    decoder LSTM reconstructs the window from that latent vector. Trained to
    minimize reconstruction MSE on normal-only windows, so at inference,
    windows that don't look like the learned 'normal trajectory' shape
    reconstruct poorly -- that reconstruction error IS the anomaly score."""

    def __init__(self, n_features, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(h_n[-1])                       # (batch, latent_dim)
        dec_input = self.decoder_fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_input)
        recon = self.output_fc(dec_out)                         # (batch, seq_len, n_features)
        return recon


def build_windows(df: pd.DataFrame, seq_len=SEQ_LEN):
    """For each entity, build every sliding window of seq_len consecutive
    sessions (chronological order already guaranteed by features.py sort).
    Returns window feature arrays, the session_id each window "belongs to"
    (its last/most-recent session), and whether every session in the window
    is labelled normal (used to build the training set)."""
    windows, window_session_ids, window_all_normal = [], [], []

    for entity_id, group in df.groupby("entity_id", sort=False):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < seq_len:
            continue  # cold-start entity: no window possible, handled by IF fallback
        feats = group[FEATURE_COLS].fillna(0).values
        labels = group["label"].values
        session_ids = group["session_id"].values

        for i in range(seq_len - 1, n):
            window = feats[i - seq_len + 1: i + 1]
            windows.append(window)
            window_session_ids.append(session_ids[i])
            window_all_normal.append(bool(np.all(labels[i - seq_len + 1: i + 1] == "normal")))

    return np.array(windows), np.array(window_session_ids), np.array(window_all_normal)


def train_autoencoder(train_windows: np.ndarray):
    model = LSTMAutoencoder(n_features=train_windows.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    X = torch.tensor(train_windows, dtype=torch.float32)
    n = len(X)

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            batch = X[idx].to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}  train_recon_mse={total_loss/n:.5f}")

    return model


def score_windows(model, windows: np.ndarray) -> np.ndarray:
    model.eval()
    X = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        recon = model(X)
        # per-window MSE, mean over timesteps and features -> one score per window
        mse = ((recon - X) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return mse


def run():
    feat = pd.read_csv("data/features.csv")
    labels = pd.read_csv("data/ground_truth_labels.csv")
    df = feat.merge(labels, on="session_id", how="left")
    df["timestamp_order"] = df.groupby("entity_id").cumcount()  # already sorted by features.py

    # Scale features using stats from normal rows only, same principle as the IF baseline.
    scaler = StandardScaler()
    normal_mask = df["label"] == "normal"
    scaler.fit(df.loc[normal_mask, FEATURE_COLS].fillna(0))
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS].fillna(0))

    windows, window_session_ids, window_all_normal = build_windows(df)
    print(f"Built {len(windows)} windows (seq_len={SEQ_LEN}) across "
          f"{df['entity_id'].nunique()} entities; "
          f"{window_all_normal.sum()} are all-normal (used for training).")

    if window_all_normal.sum() < 50:
        print("WARNING: very few all-normal windows available -- sequence model "
              "may be under-trained on this data size. Consider a larger --target_sessions.")

    train_windows = windows[window_all_normal]
    model = train_autoencoder(train_windows)

    # Save PyTorch model and scaler
    import os
    import pickle
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/sequence_lstm.pth")
    with open("models/sequence_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Saved LSTM autoencoder model and scaler to models/")

    all_scores = score_windows(model, windows)
    # min-max scale to 0-1, same convention as the Isolation Forest risk_score
    risk = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min() + 1e-9)

    seq_scores_df = pd.DataFrame({
        "session_id": window_session_ids,
        "seq_risk_score": risk,
        "has_sequence_score": 1,
    })

    # Sessions that never appeared as the last element of any window (cold-start
    # entities, or an entity's first SEQ_LEN-1 sessions) get no sequence score.
    all_session_ids = df["session_id"].values
    missing = pd.DataFrame({
        "session_id": [sid for sid in all_session_ids if sid not in set(window_session_ids)],
    })
    missing["seq_risk_score"] = np.nan
    missing["has_sequence_score"] = 0

    out = pd.concat([seq_scores_df, missing], ignore_index=True)
    out.to_csv("data/sequence_scores.csv", index=False)
    print(f"\nSaved data/sequence_scores.csv "
          f"({int(out['has_sequence_score'].sum())} sessions scored by LSTM, "
          f"{int((out['has_sequence_score']==0).sum())} fell back to cold-start path)")

    # --- Evaluate the sequence model on its own, on the sessions it actually covers ---
    eval_df = df[["session_id", "label"]].merge(seq_scores_df, on="session_id", how="inner")
    eval_df = eval_df[eval_df["label"].isin(["normal", "anomaly"])]
    y_true = (eval_df["label"] == "anomaly").astype(int).values
    y_score = eval_df["seq_risk_score"].values

    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        pr_auc = average_precision_score(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        k = max(1, int(len(y_true) * 0.01))
        top_k_idx = np.argsort(-y_score)[:k]
        precision_at_k = y_true[top_k_idx].mean()
        recall_at_k = y_true[top_k_idx].sum() / max(y_true.sum(), 1)
        fp_at_k = int(k - y_true[top_k_idx].sum())
        n_neg = int((y_true == 0).sum())
        fpr_at_k = fp_at_k / max(n_neg, 1)

        print(f"\nSequence model standalone evaluation (sessions with a window only):")
        print(f"  Rows evaluated: {len(eval_df)}  (anomalies: {int(y_true.sum())})")
        print(f"  PR-AUC:  {pr_auc:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Precision@top1%: {precision_at_k:.4f}")
        print(f"  Recall@top1%:    {recall_at_k:.4f}")
        print(f"  FPR@top1%:       {fpr_at_k:.5f}")


if __name__ == "__main__":
    run()