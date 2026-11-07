"""Log the pipeline's metrics, reports and models to MLflow.

Works offline too: if MLflow or the tracking server is unavailable, the same
summary is written to ``data/mlflow_run.json`` so the dashboard/CI still has a
record.

Run:
    python track_experiment.py
    MLFLOW_TRACKING_URI=http://localhost:5000 python track_experiment.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path


def _read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def _flatten_metrics(evaluation, calibration, drift):
    metrics = {}
    if evaluation:
        models = {item["name"]: item for item in evaluation.get("models", [])}
        pipeline = models.get("Hybrid IF+LSTM (pipeline)", {})
        calibrated = models.get("Hybrid + isotonic calibration", {})
        for prefix, model in (("pipeline", pipeline), ("calibrated", calibrated)):
            for key in ("pr_auc", "roc_auc", "precision_at_1", "recall_at_1", "fpr_at_1"):
                if key in model:
                    metrics[f"{prefix}_{key}"] = model[key]
        calibration_summary = evaluation.get("calibration") or {}
        if calibration_summary:
            metrics["calibration_ece"] = calibration_summary.get("ece")
            metrics["calibration_brier"] = calibration_summary.get("brier")

    if calibration:
        for item in calibration.get("detectors", []):
            if item.get("status") != "ok":
                continue
            detector = item["detector"]
            metrics[f"{detector}_brier"] = item["calibrated_brier"]
            metrics[f"{detector}_ece"] = item["calibrated_ece"]
            metrics[f"{detector}_conformal_recall"] = item["conformal_recall"]

    if drift:
        metrics["drift_max_psi"] = drift.get("max_feature_psi")
        metrics["drift_significant_features"] = drift.get("drifted_features")

    return {key: value for key, value in metrics.items() if value is not None}


def _log_to_mlflow(tracking_uri, evaluation, calibration, drift, shap_report):
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("behavioural-anomaly-detection")

    metrics = _flatten_metrics(evaluation, calibration, drift)
    params = {
        "alert_budget_fraction": evaluation.get("dataset", {}).get(
            "alert_budget_fraction", 0.01
        ),
        "test_rows": evaluation.get("dataset", {}).get("test_rows"),
        "calibration_alpha": calibration.get("conformal_alpha"),
        "attribution_method": shap_report.get("method") if shap_report else None,
    }
    params = {key: value for key, value in params.items() if value is not None}
    with mlflow.start_run(run_name="pipeline-report") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        for path in (
            "data/evaluation_report.json",
            "data/calibration_report.json",
            "data/drift_report.json",
            "data/shap_global.json",
            "data/shap_local.json",
        ):
            if Path(path).exists():
                mlflow.log_artifact(path, artifact_path="reports")

        registry = {}
        for model_path in (
            "models/anomaly_classifier.pkl",
            "models/baseline_isolation_forest.pkl",
            "models/baseline_scaler.pkl",
            "models/sequence_lstm.pth",
            "models/sequence_scaler.pkl",
            "models/calibrators.pkl",
        ):
            if Path(model_path).exists():
                mlflow.log_artifact(model_path, artifact_path="models")

        # Register the attack classifier when the server supports a registry.
        try:
            with open("models/anomaly_classifier.pkl", "rb") as handle:
                classifier = pickle.load(handle)
            for kwargs in (
                {"name": "attack_classifier", "registered_model_name": "anomaly-attack-classifier"},
                {"artifact_path": "attack_classifier", "registered_model_name": "anomaly-attack-classifier"},
            ):
                try:
                    mlflow.sklearn.log_model(classifier, **kwargs)
                    registry["classifier"] = "registered as anomaly-attack-classifier"
                    break
                except TypeError:
                    continue
        except Exception as exc:  # noqa: BLE001
            registry["classifier"] = f"artifact only ({exc})"

        return run.info.run_id, metrics, registry


def main():
    parser = argparse.ArgumentParser(description="Log pipeline results to MLflow.")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    )
    args = parser.parse_args()

    evaluation = _read_json("data/evaluation_report.json")
    calibration = _read_json("data/calibration_report.json")
    drift = _read_json("data/drift_report.json")
    shap_report = _read_json("data/shap_global.json")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracking_uri": args.tracking_uri,
        "metrics": _flatten_metrics(evaluation, calibration, drift),
    }

    try:
        run_id, metrics, registry = _log_to_mlflow(
            args.tracking_uri, evaluation or {}, calibration or {}, drift or {}, shap_report
        )
        summary.update(
            {
                "status": "logged",
                "run_id": run_id,
                "metrics": metrics,
                "registry": registry,
            }
        )
        print(f"Logged MLflow run {run_id} to {args.tracking_uri}")
    except Exception as exc:  # noqa: BLE001
        summary.update({"status": "offline", "error": str(exc)})
        print(f"MLflow unavailable ({exc}); writing local summary only")

    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/mlflow_run.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print("Saved data/mlflow_run.json")


if __name__ == "__main__":
    main()
