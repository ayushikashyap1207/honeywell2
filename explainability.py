
#Phase 5: Explainability Layer



import numpy as np
import pandas as pd

FEATURE_COLS = [
    "hour_z", "resource_novelty", "n_resources_this_session", "device_novel",
    "geo_km_per_hr", "session_duration_sec", "cmd_seq_len", "failed_auth_attempts",
    "is_cold_start", "ip_failed_5min", "ip_unique_entities_5min",
]

# Human-readable phrasing per feature, used when it's a top contributor to a flag.
FEATURE_EXPLANATIONS = {
    "hour_z": "login at an unusual hour for this entity",
    "resource_novelty": "accessed resources never touched before",
    "n_resources_this_session": "unusually high number of resources touched in one session",
    "device_novel": "new/unrecognized device fingerprint",
    "geo_km_per_hr": "implausible geographic velocity (impossible travel)",
    "session_duration_sec": "session duration far from this entity's norm",
    "cmd_seq_len": "unusually long command sequence",
    "failed_auth_attempts": "failed authentication attempt(s)",
    "is_cold_start": "new entity with little to no history (cold start)",
    "ip_failed_5min": "many failed logins from this source IP in a short window",
    "ip_unique_entities_5min": "one source IP touching many different accounts (credential stuffing pattern)",
}


def compute_normal_baseline(features_df: pd.DataFrame, labels_df: pd.DataFrame):
    """Population-level mean/std per feature, computed only from confirmed-normal rows.
    Used as the reference distribution for z-score attribution -- separate per entity_type
    since a service account's 'normal' session_duration looks nothing like a human user's."""
    df = features_df.merge(labels_df, on="session_id", how="left")
    normal = df[df["label"] == "normal"]

    stats = {}
    for etype, group in normal.groupby("entity_type"):
        stats[etype] = {
            col: (group[col].mean(), max(group[col].std(), 1e-6))
            for col in FEATURE_COLS
        }
    # global fallback for any entity_type not seen in normal training data
    stats["_global"] = {
        col: (normal[col].mean(), max(normal[col].std(), 1e-6))
        for col in FEATURE_COLS
    }
    return stats


def explain_row(row, stats):
    etype_stats = stats.get(row["entity_type"], stats["_global"])
    z_scores = {}
    for col in FEATURE_COLS:
        mean, std = etype_stats[col]
        z_scores[col] = abs((row[col] - mean) / std)

    top_features = sorted(z_scores.items(), key=lambda x: -x[1])[:3]
    top_features = [(f, z) for f, z in top_features if z > 1.0]  # only meaningfully deviant ones

    if not top_features:
        return "No strong deviation from baseline; low-confidence flag.", {}

    reason_parts = [FEATURE_EXPLANATIONS[f] for f, z in top_features]
    explanation = "Flagged due to: " + " + ".join(reason_parts) + "."
    contribution = {f: round(float(z), 2) for f, z in top_features}
    return explanation, contribution


def build_explained_alerts():
    features_df = pd.read_csv("data/features.csv")
    labels_df = pd.read_csv("data/ground_truth_labels.csv")
    # FIX: read the combined sequence-LSTM + cold-start-fallback score instead
    # of the old Isolation-Forest-only baseline_scores.csv, so explanations
    # attach to the system's actual final risk ranking.
    final_scores_df = pd.read_csv("data/final_scores.csv")
    classified_df = pd.read_csv("data/classified_alerts.csv")

    stats = compute_normal_baseline(features_df, labels_df)

    full = features_df.merge(labels_df, on="session_id", how="left")

    explanations, contributions = [], []
    for _, row in full.iterrows():
        exp, contrib = explain_row(row, stats)
        explanations.append(exp)
        contributions.append(contrib)

    full["explanation"] = explanations
    full["top_contributing_features"] = contributions

    result = full.merge(
        final_scores_df[["session_id", "risk_score", "detection_method"]],
        on="session_id", how="left"
    ).merge(
        classified_df[["session_id", "predicted_type", "predicted_confidence"]],
        on="session_id", how="left"
    )

    result = result.sort_values("risk_score", ascending=False)
    return result


if __name__ == "__main__":
    result = build_explained_alerts()

    out_cols = [
        "session_id", "entity_id", "entity_type", "risk_score", "detection_method",
        "predicted_type", "predicted_confidence", "explanation",
        "top_contributing_features", "label", "anomaly_type",
    ]
    result[out_cols].to_csv("data/alerts_explained.csv", index=False)
    print("Saved data/alerts_explained.csv")

    print("\nSample explanations for top 5 highest-risk alerts:\n")
    for _, row in result.head(5).iterrows():
        print(f"[{row['entity_id']}] risk={row['risk_score']:.3f} "
              f"predicted={row['predicted_type']} ({row['predicted_confidence']:.2f}) "
              f"true={row['anomaly_type']}")
        print(f"  {row['explanation']}\n")