"""Orchestrated retraining flow.

Checks drift, consumes analyst feedback, reruns the batch pipeline, refits
calibration, regenerates evaluation/drift/SHAP reports, and logs the run to
MLflow. Prefect is used when installed; the same functions can be run directly.

Run:
    python flows/retrain.py --force
    make retrain
"""

from __future__ import annotations

import argparse
import subprocess
import sys

try:
    from prefect import flow, task
except Exception:  # noqa: BLE001 - keep the flow runnable without Prefect
    def task(function=None, **_kwargs):
        if function is None:
            return lambda inner: inner
        return function

    def flow(function=None, **_kwargs):
        if function is None:
            return lambda inner: inner
        return function


@task
def run_command(command):
    print(f"$ {' '.join(command)}")
    subprocess.run(command, check=True)


@task
def check_drift():
    result = subprocess.run(
        [sys.executable, "-m", "monitoring.drift", "--fail-on-drift"],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    if result.returncode == 0:
        return "stable"
    if result.returncode == 1:
        return "drift"
    raise RuntimeError(result.stderr.strip() or "drift check failed")


@flow(name="retrain-anomaly-pipeline")
def retrain_flow(force: bool = False, skip_training: bool = False):
    run_command([sys.executable, "apply_feedback.py"])

    drift_status = check_drift()
    print(f"Drift status: {drift_status}")

    if not force and drift_status != "drift":
        print("No significant drift. Skipping retrain (use --force to override).")
        return {"status": "skipped", "reason": "no significant drift"}

    if skip_training:
        print("skip_training=True: reports only.")
    else:
        for command in (
            [sys.executable, "features.py"],
            [sys.executable, "train_baseline_model.py"],
            [sys.executable, "train_sequence_model.py"],
            [sys.executable, "combine_scores.py"],
            [sys.executable, "-m", "calibration.calibrate"],
            [sys.executable, "train_classifier.py"],
            [sys.executable, "explainability.py"],
        ):
            run_command(command)

    for command in (
        [sys.executable, "evaluation.py"],
        [sys.executable, "-m", "monitoring.drift"],
        [sys.executable, "explainability_shap.py"],
        [sys.executable, "apply_feedback.py", "--mark-used"],
        [sys.executable, "export_alerts_json.py"],
        [sys.executable, "track_experiment.py"],
    ):
        run_command(command)

    print("Retrain flow complete.")
    return {"status": "completed", "drift": drift_status}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain the anomaly pipeline.")
    parser.add_argument("--force", action="store_true", help="Retrain even without drift")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Refresh reports only, without retraining models",
    )
    args = parser.parse_args()
    retrain_flow(force=args.force, skip_training=args.skip_training)
