"""
Phase 3a: Baseline Detection Model (cold-start fallback)
This is a simple Isolation Forest model trained on a "clean" slice of the data (rows labeled normal).
The model is trained on the features engineered in Phase 2, and produces a risk score for each session.
The risk score is then rank-normalized to a 0-1 scale, where higher = more anomalous. This score is used as a fallback when the sequence-aware model cannot be applied (e.g. cold-start sessions with no prior history).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURE_COLS = [
    "hour_z", "resource_novelty", "n_resources_this_session", "device_novel",
    "geo_km_per_hr", "session_duration_sec", "cmd_seq_len", "failed_auth_attempts",
    "is_cold_start", "ip_failed_5min", "ip_unique_entities_5min",
]


def load_data():
    feat = pd.read_csv("data/features.csv")
    labels = pd.read_csv("data/ground_truth_labels.csv")
    df = feat.merge(labels, on="session_id", how="left")
    df["y_true"] = (df["label"] == "anomaly").astype(int)  # edge_case rows excluded from strict eval below
    return df


def train_and_score(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train on a "clean" slice: rows labeled normal. In production this would be
    # "traffic older than N days with no confirmed incidents" instead of a ground-truth label.
    train_mask = df["label"] == "normal"

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled[train_mask.values])

    # decision_function: higher = more normal. Flip + min-max scale to a 0-1 risk score.
    raw_scores = -model.decision_function(X_scaled)
    risk_score = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

    df = df.copy()
    df["risk_score"] = risk_score
    return df, model, scaler


def evaluate(df: pd.DataFrame):
    # Evaluate strictly on anomaly vs normal (edge_case set aside as ambiguous by design).
    eval_df = df[df["label"].isin(["normal", "anomaly"])]
    y_true = eval_df["y_true"].values
    y_score = eval_df["risk_score"].values

    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    # Precision / recall / FPR at a realistic analyst alert budget: top 1% of events by risk score
    k = max(1, int(len(eval_df) * 0.01))
    top_k_idx = np.argsort(-y_score)[:k]

    tp_at_k = int(y_true[top_k_idx].sum())
    fp_at_k = int(k - tp_at_k)
    n_negatives_total = int((y_true == 0).sum())

    precision_at_k = y_true[top_k_idx].mean()
    recall_at_k = y_true[top_k_idx].sum() / max(y_true.sum(), 1)
    # FIX: explicit FPR, named directly in the evaluation criteria.
    fpr_at_k = fp_at_k / max(n_negatives_total, 1)

    print(f"Rows evaluated: {len(eval_df)}  (anomalies: {int(y_true.sum())})")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Alert budget = top {k} events (1%)")
    print(f"  Precision@top1%: {precision_at_k:.4f}")
    print(f"  Recall@top1%:    {recall_at_k:.4f}")
    print(f"  FPR@top1%:       {fpr_at_k:.5f}  ({fp_at_k} false positives / {n_negatives_total} true negatives)")

    return {
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "precision_at_top1pct": precision_at_k, "recall_at_top1pct": recall_at_k,
        "fpr_at_top1pct": fpr_at_k, "alert_budget_k": k,
    }


if __name__ == "__main__":
    import os
    import pickle

    df = load_data()
    scored_df, model, scaler = train_and_score(df)
    metrics = evaluate(scored_df)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    with open("models/baseline_isolation_forest.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/baseline_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Saved baseline model and scaler to models/")

    out_cols = ["session_id", "entity_id", "entity_type", "risk_score", "label", "anomaly_type", "is_cold_start"]
    scored_df.sort_values("risk_score", ascending=False)[out_cols].rename(
        columns={"risk_score": "if_risk_score"}
    ).to_csv("data/baseline_scores.csv", index=False)
    print("\nSaved data/baseline_scores.csv (sorted by risk_score, highest first)")

    # top-10 preview
    print("\nTop 10 highest-risk sessions:")
    print(scored_df.sort_values("risk_score", ascending=False)[
        ["entity_id", "risk_score", "label", "anomaly_type"]
    ].head(10).to_string(index=False))