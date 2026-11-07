"""Incident retrieval index + similar-incident search.

Primary store: the ``alert_embeddings`` table (JSONB vectors, ready to swap for
pgvector). Fallback: ``data/alerts_explained.csv`` so the copilot works before
the index has been built or when Postgres is unavailable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sqlalchemy import text

from api.db import engine
from copilot.embeddings import alert_content, cosine, embed

ALERTS_CSV = "data/alerts_explained.csv"
MAX_FALLBACK_ROWS = 2000


def _parse_json(value):
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


def _fetch_db_alerts(limit=5000):
    query = """
        SELECT s.session_id, s.entity_id, s.entity_type, s.risk_score,
               s.predicted_type, s.explanation, s.anomaly_type,
               s.top_contributing_features, e.geo_location, e.resource_accessed
        FROM scores s
        LEFT JOIN sessions e ON e.session_id = s.session_id
        WHERE s.explanation IS NOT NULL
        ORDER BY s.scored_at DESC NULLS LAST
        LIMIT :limit
    """
    with engine.connect() as connection:
        rows = connection.execute(text(query), {"limit": limit}).mappings().all()
    output = []
    for row in rows:
        item = dict(row)
        item["top_contributing_features"] = _parse_json(
            item.get("top_contributing_features")
        )
        output.append(item)
    return output


def _csv_alerts(limit=MAX_FALLBACK_ROWS):
    path = Path(ALERTS_CSV)
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                row["risk_score"] = float(row.get("risk_score") or 0)
            except ValueError:
                row["risk_score"] = 0.0
            row["top_contributing_features"] = _parse_json(
                row.get("top_contributing_features")
            )
            rows.append(row)
    rows.sort(key=lambda item: item["risk_score"], reverse=True)
    return rows[:limit]


def build_index(limit=5000, force=False):
    """Embed recent scored sessions and upsert them into the vector table."""
    try:
        rows = _fetch_db_alerts(limit=limit)
    except Exception as exc:  # noqa: BLE001
        print(f"Database unavailable ({exc}); using CSV fallback index")
        return 0

    if not rows:
        return 0

    count = 0
    with engine.begin() as connection:
        for row in rows:
            content = alert_content(row)
            vector = json.dumps(embed(content))
            connection.execute(
                text(
                    """
                    INSERT INTO alert_embeddings (session_id, content, embedding)
                    VALUES (:session_id, :content, CAST(:embedding AS JSONB))
                    ON CONFLICT (session_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """
                ),
                {
                    "session_id": row["session_id"],
                    "content": content,
                    "embedding": vector,
                },
            )
            count += 1
    print(f"Indexed {count} alerts into alert_embeddings")
    return count


def _load_index():
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text("SELECT session_id, content, embedding FROM alert_embeddings")
            ).mappings().all()
        if rows:
            return [
                {
                    "session_id": row["session_id"],
                    "content": row["content"],
                    "embedding": _parse_json(row["embedding"]),
                }
                for row in rows
            ]
    except Exception:  # noqa: BLE001
        pass

    fallback = []
    for row in _csv_alerts():
        content = alert_content(row)
        fallback.append(
            {
                "session_id": row.get("session_id"),
                "content": content,
                "embedding": embed(content),
            }
        )
    return fallback


def similar(query=None, session_id=None, k=5, exclude_session_id=None):
    index = _load_index()
    if not index:
        return []

    if query:
        query_vector = embed(query)
    elif session_id:
        match = next(
            (item for item in index if item["session_id"] == session_id), None
        )
        if match is None:
            return []
        query_vector = match["embedding"] or embed(match["content"])
    else:
        return []

    scored = []
    for item in index:
        if item["session_id"] == exclude_session_id:
            continue
        if query_vector is item["embedding"]:
            continue
        scored.append(
            {
                "session_id": item["session_id"],
                "content": item["content"][:600],
                "similarity": round(cosine(query_vector, item["embedding"]), 4),
            }
        )

    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored[:k]


if __name__ == "__main__":
    build_index()
