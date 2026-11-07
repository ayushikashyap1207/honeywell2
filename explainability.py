
# Phase 5: Explainability Layer
#
# The actual explanation logic lives in serving/explain.py so the batch
# pipeline and the live streaming worker produce identical explanations.

from pathlib import Path

import pandas as pd

from serving.explain import FEATURE_COLS, explain_row, save_baseline


def _mean_std(series: pd.Series):
    mean = series.mean()
    std = series.std()
    if pd.isna(mean):
        mean = 0.0
    if pd.isna(std):
        std = 1.0
    return float(mean), float(max(std, 1e-6))


def compute_normal_baseline(features_df: pd.DataFrame, labels_df: pd.DataFrame):
    """Population-level mean/std per feature, computed only from confirmed-normal rows.
    Used as the reference distribution for z-score attribution -- separate per entity_type
    since a service account's 'normal' session_duration looks nothing like a human user's."""
    df = features_df.merge(labels_df, on="session_id", how="left")
    normal = df[df["label"] == "normal"]

    stats = {}
    for etype, group in normal.groupby("entity_type"):
        stats[etype] = {
            col: _mean_std(group[col]) for col in FEATURE_COLS
        }
    # global fallback for any entity_type not seen in normal training data
    stats["_global"] = {
        col: _mean_std(normal[col]) for col in FEATURE_COLS
    }
    return stats


def build_explained_alerts():
    features_df = pd.read_csv("data/features.csv")
    labels_df = pd.read_csv("data/ground_truth_labels.csv")
    # FIX: read the combined sequence-LSTM + cold-start-fallback score instead
    # of the old Isolation-Forest-only baseline_scores.csv, so explanations
    # attach to the system's actual final risk ranking.
    final_scores_df = pd.read_csv("data/final_scores.csv")
    classified_df = pd.read_csv("data/classified_alerts.csv")

    stats = compute_normal_baseline(features_df, labels_df)
    save_baseline(stats, "models/explanation_baseline.json")
    print("Saved models/explanation_baseline.json (used by the live scorer)")

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

    calibrated_path = Path("data/calibrated_scores.csv")
    if calibrated_path.exists():
        result = result.merge(pd.read_csv(calibrated_path), on="session_id", how="left")
    else:
        result["calibrated_risk"] = float("nan")
        result["conformal_p_value"] = float("nan")

    result = result.sort_values("risk_score", ascending=False)
    return result


if __name__ == "__main__":
    result = build_explained_alerts()

    out_cols = [
        "session_id", "entity_id", "entity_type", "risk_score", "calibrated_risk",
        "conformal_p_value", "detection_method", "predicted_type",
        "predicted_confidence", "explanation", "top_contributing_features",
        "label", "anomaly_type",
    ]
    result[out_cols].to_csv("data/alerts_explained.csv", index=False)
    print("Saved data/alerts_explained.csv")

    print("\nSample explanations for top 5 highest-risk alerts:\n")
    for _, row in result.head(5).iterrows():
        print(f"[{row['entity_id']}] risk={row['risk_score']:.3f} "
              f"predicted={row['predicted_type']} ({row['predicted_confidence']:.2f}) "
              f"true={row['anomaly_type']}")
        print(f"  {row['explanation']}\n")