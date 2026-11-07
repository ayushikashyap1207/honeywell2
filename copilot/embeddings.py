"""Deterministic local text embeddings for incident retrieval.

A signed hashing trick over word unigrams + bigrams. This keeps the copilot
self-contained (no model download, no API key) while still providing real
vector similarity search. The stored JSON vectors are pgvector-ready: swap
``embedding JSONB`` for ``embedding vector(256)`` and use ``<=>`` when the
extension is available.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

DIMENSION = 256
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text):
    words = _TOKEN_RE.findall(str(text).lower())
    unigrams = words
    bigrams = [f"{left}_{right}" for left, right in zip(words, words[1:])]
    return unigrams + bigrams


def embed(text, dimension=DIMENSION):
    vector = [0.0] * dimension
    for token, count in Counter(_tokens(text)).items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * (1.0 + math.log(count))

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 6) for value in vector]


def cosine(left, right):
    if not left or not right:
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


def alert_content(row):
    """Build the text that gets embedded for an alert."""
    features = row.get("top_contributing_features") or {}
    return (
        f"entity {row.get('entity_id')} type {row.get('entity_type')} "
        f"attack {row.get('predicted_type') or row.get('anomaly_type') or 'unknown'} "
        f"risk {row.get('risk_score')} "
        f"explanation {row.get('explanation') or ''} "
        f"features {features} "
        f"resource {row.get('resource_accessed') or ''} "
        f"geo {row.get('geo_location') or ''}"
    )
