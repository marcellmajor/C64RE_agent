"""Load semantic / embedding settings for KB in-memory retrieval."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "kb_semantic.json"


@dataclass(frozen=True)
class KBSemanticConfig:
    enabled: bool
    use_llm_provider: str
    embedding_model: str
    embedding_dimensions: int | None
    chunk_chars: int
    chunk_overlap: int
    embed_batch_size: int
    embed_timeout_s: int
    digest_semantic_limit: int
    semantic_search_limit_default: int
    index_event_kinds: frozenset[str]


def _as_bool(v: object, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def load_semantic_config() -> KBSemanticConfig:
    if not CONFIG_PATH.exists():
        return KBSemanticConfig(
            enabled=False,
            use_llm_provider="executor",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=None,
            chunk_chars=1200,
            chunk_overlap=200,
            embed_batch_size=48,
            embed_timeout_s=120,
            digest_semantic_limit=6,
            semantic_search_limit_default=10,
            index_event_kinds=frozenset(),
        )

    raw = json.loads(CONFIG_PATH.read_text())
    kinds = raw.get("index_event_kinds") or []
    if not isinstance(kinds, list):
        kinds = []
    dims = raw.get("embedding_dimensions")
    try:
        dim_i = int(dims) if dims is not None else None
    except (TypeError, ValueError):
        dim_i = None

    return KBSemanticConfig(
        enabled=_as_bool(raw.get("enabled"), False),
        use_llm_provider=str(raw.get("use_llm_provider") or "executor"),
        embedding_model=str(raw.get("embedding_model") or "text-embedding-3-small"),
        embedding_dimensions=dim_i,
        chunk_chars=max(128, int(raw.get("chunk_chars") or 1200)),
        chunk_overlap=max(0, int(raw.get("chunk_overlap") or 200)),
        embed_batch_size=max(1, int(raw.get("embed_batch_size") or 48)),
        embed_timeout_s=max(10, int(raw.get("embed_timeout_s") or 120)),
        digest_semantic_limit=max(1, int(raw.get("digest_semantic_limit") or 6)),
        semantic_search_limit_default=max(1, int(raw.get("semantic_search_limit_default") or 10)),
        index_event_kinds=frozenset(str(k) for k in kinds),
    )


def invalidate_semantic_config_cache() -> None:
    load_semantic_config.cache_clear()


def semantic_config_reload_warn() -> None:
    msg = "[kb_semantic] Config changed on disk — restart the process to reload."
    print(msg, file=sys.stderr)
