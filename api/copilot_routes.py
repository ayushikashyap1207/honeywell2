"""Copilot API routes."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from copilot.agent import agent
from copilot.mitre import get_mitre
from copilot.retrieval import build_index, similar

router = APIRouter(prefix="/api/copilot", tags=["copilot"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)
    session_id: str | None = Field(default=None, max_length=200)
    entity_id: str | None = Field(default=None, max_length=200)


@router.get("/status")
def status():
    return agent.status()


@router.post("/chat")
def chat(payload: ChatRequest):
    return agent.chat(
        payload.message, session_id=payload.session_id, entity_id=payload.entity_id
    )


@router.get("/similar")
def similar_incidents(
    session_id: str | None = None,
    query: str | None = None,
    k: int = Query(default=5, ge=1, le=20),
):
    return similar(session_id=session_id, query=query, k=k)


@router.get("/mitre/{attack_type}")
def mitre(attack_type: str):
    return get_mitre(attack_type)


@router.post("/refresh-index")
def refresh_index(limit: int = Query(default=5000, ge=1, le=20000)):
    try:
        count = build_index(limit=limit)
        return {"status": "ok", "indexed": count}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc)}
