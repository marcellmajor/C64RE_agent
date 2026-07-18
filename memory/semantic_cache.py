"""Persistent, content-addressed semantic embedding cache (tracker 2.5)."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

from memory.semantic_config import KBSemanticConfig
from memory.semantic_embed import embed_texts_batched


_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    cache_key  TEXT PRIMARY KEY,
    model_key  TEXT NOT NULL,
    text_hash  TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vectors_model ON vectors(model_key);
"""


def _model_key(cfg: KBSemanticConfig) -> str:
    dimensions = cfg.embedding_dimensions if cfg.embedding_dimensions else "native"
    return f"{cfg.use_llm_provider}|{cfg.embedding_model}|{dimensions}"


def _cache_key(cfg: KBSemanticConfig, text: str) -> tuple[str, str, str]:
    model_key = _model_key(cfg)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cache_key = hashlib.sha256(
        f"{model_key}\0{text_hash}".encode("utf-8"),
    ).hexdigest()
    return cache_key, model_key, text_hash


class SemanticVectorCache:
    """Small SQLite cache keyed by model configuration and exact text."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.path))
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(_SCHEMA)
        return db

    def get_many(self, keys: Iterable[str]) -> dict[str, list[float]]:
        unique = list(dict.fromkeys(str(key) for key in keys))
        if not unique or not self.path.exists():
            return {}
        out: dict[str, list[float]] = {}
        with self._connect() as db:
            # Stay comfortably below SQLite's common 999-parameter limit.
            for start in range(0, len(unique), 400):
                batch = unique[start : start + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = db.execute(
                    "SELECT cache_key, dimensions, vector FROM vectors "
                    f"WHERE cache_key IN ({placeholders})",
                    tuple(batch),
                ).fetchall()
                for row in rows:
                    dim = int(row["dimensions"])
                    vector = np.frombuffer(row["vector"], dtype=np.float32)
                    if dim <= 0 or len(vector) != dim:
                        # A partial/corrupt row is a cache miss, never evidence.
                        continue
                    out[str(row["cache_key"])] = vector.astype(float).tolist()
        return out

    def put_many(
        self,
        rows: Iterable[tuple[str, str, str, list[float]]],
    ) -> None:
        encoded = []
        for cache_key, model_key, text_hash, raw_vector in rows:
            vector = np.asarray(raw_vector, dtype=np.float32)
            if vector.ndim != 1 or vector.size <= 0:
                raise ValueError("semantic vector must be a non-empty 1-D row")
            encoded.append((
                cache_key,
                model_key,
                text_hash,
                int(vector.size),
                vector.tobytes(),
            ))
        if not encoded:
            return
        with self._connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO vectors"
                " (cache_key, model_key, text_hash, dimensions, vector)"
                " VALUES (?, ?, ?, ?, ?)",
                encoded,
            )


def embed_texts_cached(
    cfg: KBSemanticConfig,
    texts: list[str],
    cache_path: str | Path,
) -> list[list[float]]:
    """Embed only cache misses and return vectors in original text order."""
    if not texts:
        return []

    identities = [_cache_key(cfg, text) for text in texts]
    cache = SemanticVectorCache(cache_path)
    cached = cache.get_many(cache_key for cache_key, _, _ in identities)

    # Deduplicate equal missing texts within the same batch as well as across
    # process restarts. Dict insertion order makes provider input deterministic.
    missing: dict[str, tuple[str, str, str]] = {}
    for text, identity in zip(texts, identities):
        if identity[0] not in cached:
            missing.setdefault(identity[0], (text, identity[1], identity[2]))

    if missing:
        missing_rows = list(missing.items())
        vectors = embed_texts_batched(
            cfg, [row[1][0] for row in missing_rows],
        )
        if len(vectors) != len(missing_rows):
            raise ValueError(
                f"embedding cache fill mismatch: {len(vectors)} "
                f"vs {len(missing_rows)}",
            )
        to_store = []
        for ((cache_key, (_text, model_key, text_hash)), vector) in zip(
            missing_rows, vectors,
        ):
            cached[cache_key] = vector
            to_store.append((cache_key, model_key, text_hash, vector))
        cache.put_many(to_store)

    return [cached[cache_key] for cache_key, _, _ in identities]
