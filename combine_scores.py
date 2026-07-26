
#Phase 3: Combine sequence-aware and cold-start-fallback scores


import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def combine():
    seq = pd.read_csv("data/sequence_scores.csv")
    base = pd.read_csv("data/baseline_scores.csv")  # entity_id, entity_type, if_risk_score, label, anomaly_type, is_cold_start
    labels = pd.read_csv("data/ground_truth_labels.csv")

    merged = base.merge(seq, on="session_id", how="left")

    merged["detection_method"] = np.where(
        merged["has_sequence_score"] == 1,
        "sequence_lstm",
        "cold_start_isolation_forest",
    )

    # --- Rank-normalize within each model's own population ---
    is_seq = merged["detection_method"] == "sequence_lstm"
    is_cold = ~is_seq

    merged["risk_score"] = 0.0
    merged.loc[is_seq, "risk_score"] = merged.loc[is_seq, "seq_risk_score"].rank(pct=True)
    merged.loc[is_cold, "risk_score"] = merged.loc[is_cold, "if_risk_score"].rank(pct=True)

    merged = merged.sort_values("risk_score", ascending=False)
    out_cols = [
        "session_id", "entity_id", "entity_type", "risk_score", "detection_method",
        "if_risk_score", "seq_risk_score", "label", "anomaly_type", "is_cold_start",
    ]
    merged[out_cols].to_csv("data/final_scores.csv", index=False)
    print(f"Saved data/final_scores.csv ({len(merged)} rows)")
    print(merged["detection_method"].value_counts().to_string())

    # --- Evaluate the combined score end to end ---
    eval_df = merged[merged["label"].isin(["normal", "anomaly"])].copy()
    y_true = (eval_df["label"] == "anomaly").astype(int).values
    y_score = eval_df["risk_score"].values

    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    k = max(1, int(len(eval_df) * 0.01))
    top_k_idx = np.argsort(-y_score)[:k]
    precision_at_k = y_true[top_k_idx].mean()
    recall_at_k = y_true[top_k_idx].sum() / max(y_true.sum(), 1)
    fp_at_k = int(k - y_true[top_k_idx].sum())
    n_neg = int((y_true == 0).sum())
    fpr_at_k = fp_at_k / max(n_neg, 1)

    print("\nCombined (sequence + cold-start fallback) evaluation:")
    print(f"  Rows evaluated: {len(eval_df)}  (anomalies: {int(y_true.sum())})")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Alert budget = top {k} events (1%)")
    print(f"  Precision@top1%: {precision_at_k:.4f}")
    print(f"  Recall@top1%:    {recall_at_k:.4f}")
    print(f"  FPR@top1%:       {fpr_at_k:.5f}  ({fp_at_k} false positives / {n_neg} true negatives)")

    return {
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "precision_at_top1pct": precision_at_k, "recall_at_top1pct": recall_at_k,
        "fpr_at_top1pct": fpr_at_k, "alert_budget_k": k,
    }


if __name__ == "__main__":
    combine()