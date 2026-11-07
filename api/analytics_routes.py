"""Analytics, model-report and analyst-feedback endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from api.db import engine, read_json

router = APIRouter(prefix="/api", tags=["analytics"])


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1)
    verdict: Literal["confirm", "false_positive", "escalate", "acknowledge"]
    note: str | None = Field(default=None, max_length=2000)
    analyst: str | None = Field(default=None, max_length=120)


@router.get("/evaluation")
def evaluation():
    return read_json("data/evaluation_report.json")


@router.get("/drift")
def drift():
    return read_json("data/drift_report.json")


@router.get("/shap")
def shap_report():
    return read_json("data/shap_global.json")


@router.get("/mlflow")
def mlflow_summary():
    return read_json("data/mlflow_run.json")


@router.post("/feedback")
def create_feedback(payload: FeedbackRequest):
    try:
        with engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    INSERT INTO feedback (session_id, verdict, note, analyst)
                    VALUES (:session_id, :verdict, :note, :analyst)
                    RETURNING id, session_id, verdict, note, analyst, created_at
                    """
                ),
                {
                    "session_id": payload.session_id,
                    "verdict": payload.verdict,
                    "note": payload.note,
                    "analyst": payload.analyst,
                },
            ).mappings().one()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown session_id: {payload.session_id}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Feedback store unavailable: {exc}") from exc

    return {
        **dict(row),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


@router.get("/feedback/summary")
def feedback_summary():
    try:
        with engine.connect() as connection:
            counts = connection.execute(
                text("SELECT verdict, count(*) AS count FROM feedback GROUP BY verdict")
            ).mappings().all()
            pending = connection.execute(
                text(
                    "SELECT count(*) FROM feedback WHERE used_in_training = FALSE"
                )
            ).scalar() or 0
            total = connection.execute(text("SELECT count(*) FROM feedback")).scalar() or 0
    except Exception:  # noqa: BLE001
        return {"available": False, "total": 0, "pending": 0, "by_verdict": {}}

    return {
        "available": True,
        "total": int(total),
        "pending": int(pending),
        "by_verdict": {row["verdict"]: int(row["count"]) for row in counts},
    }


@router.get("/feedback/{session_id}")
def feedback_for_session(session_id: str):
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT id, session_id, verdict, note, analyst,
                           used_in_training, created_at
                    FROM feedback
                    WHERE session_id = :session_id
                    ORDER BY created_at DESC
                    """
                ),
                {"session_id": session_id},
            ).mappings().all()
    except Exception:  # noqa: BLE001
        return []

    return [
        {
            **dict(row),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
    ]
