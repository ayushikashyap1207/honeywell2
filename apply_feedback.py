"""Turn analyst feedback into training labels for the next retrain.

Confirmed attacks become ``anomaly`` rows; false positives become ``normal``
rows. Rows are written to ``data/analyst_feedback.csv`` and can be marked as
consumed once a retrain completes.

Run:
    python apply_feedback.py
    python apply_feedback.py --mark-used
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://anomaly:anomaly_pw@localhost:5432/anomaly_detection",
)
OUTPUT = "data/analyst_feedback.csv"


def main():
    parser = argparse.ArgumentParser(description="Export analyst feedback as labels.")
    parser.add_argument("--mark-used", action="store_true")
    args = parser.parse_args()

    try:
        engine = create_engine(DB_URL, pool_pre_ping=True)
        with engine.connect() as connection:
            frame = pd.read_sql(
                text(
                    """
                    SELECT f.id, f.session_id, f.verdict, f.note, f.created_at,
                           s.predicted_type, s.entity_id, s.risk_score
                    FROM feedback f
                    LEFT JOIN scores s ON s.session_id = f.session_id
                    ORDER BY f.created_at
                    """
                ),
                connection,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read feedback ({exc}); nothing exported.")
        return

    if frame.empty:
        print("No analyst feedback found.")
        Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=["session_id", "label", "anomaly_type", "source"]
        ).to_csv(OUTPUT, index=False)
        return

    rows = []
    for _, row in frame.iterrows():
        if row["verdict"] == "confirm":
            rows.append(
                {
                    "session_id": row["session_id"],
                    "label": "anomaly",
                    "anomaly_type": row["predicted_type"] or "unknown",
                    "source": "analyst",
                }
            )
        elif row["verdict"] == "false_positive":
            rows.append(
                {
                    "session_id": row["session_id"],
                    "label": "normal",
                    "anomaly_type": "none",
                    "source": "analyst",
                }
            )

    output = pd.DataFrame(rows).drop_duplicates(subset="session_id", keep="last")
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(output)} analyst labels to {OUTPUT}")

    if args.mark_used and not frame.empty:
        try:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE feedback SET used_in_training = TRUE "
                        "WHERE id = ANY(:ids)"
                    ),
                    {"ids": frame["id"].tolist()},
                )
            print(f"Marked {len(frame)} feedback rows as used_in_training")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not mark feedback as used: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
