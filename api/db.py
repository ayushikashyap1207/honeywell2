"""Shared database engine + small file helpers for the API."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://anomaly:anomaly_pw@localhost:5432/anomaly_detection",
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

engine = create_engine(DB_URL, pool_pre_ping=True)


def read_json(path, fallback=None):
    data_path = Path(path)
    if not data_path.exists():
        return {"available": False, **(fallback or {})}
    try:
        with data_path.open() as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload.setdefault("available", True)
        return payload
    except (json.JSONDecodeError, OSError):
        return {"available": False, **(fallback or {})}
