"""Batch embedding HTTP calls configured via `kb_semantic` + LLM providers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from memory.embeddings_openai_compat import embeddings_create
from memory.semantic_config import KBSemanticConfig


class EmbeddingServiceError(RuntimeError):
    """Raised when the embedding endpoint cannot authenticate or responds with an error."""


@lru_cache(maxsize=1)
def _load_llm_providers() -> dict[str, Any]:
    from graph.llm import load_config

    cfg = load_config()
    agents = cfg.get("agents") or {}
    provs = cfg.get("providers") or {}
    if not isinstance(agents, dict) or not isinstance(provs, dict):
        raise EmbeddingServiceError(
            "config/llm.json is malformed (missing agents/providers).",
        )
    return {"agents": agents, "providers": provs}


def resolve_embedding_http(cfg: KBSemanticConfig) -> tuple[str, str]:
    """Return ``(base_url, api_key)`` for an OpenAI-compatible `/v1/embeddings` call."""

    bundle = _load_llm_providers()
    agents: dict[str, Any] = bundle["agents"]
    providers: dict[str, Any] = bundle["providers"]

    role = cfg.use_llm_provider
    if role not in agents:
        known = ", ".join(sorted(agents.keys()))
        raise EmbeddingServiceError(
            f"kb_semantic.use_llm_provider={role!r} is not a role in llm.json. "
            f"Known: {known}",
        )

    pname = agents[role].get("provider")
    if pname not in providers:
        raise EmbeddingServiceError(
            f"Provider {pname!r} (for role {role!r}) missing from llm.json.",
        )

    prow = providers[pname]
    base_url = str(prow.get("base_url") or "").strip().rstrip("/")
    api_key = str(prow.get("api_key") or "").strip()
    if not base_url:
        raise EmbeddingServiceError(
            f"Provider {pname!r} has no base_url for embeddings.",
        )
    if not api_key or api_key == "missing":
        raise EmbeddingServiceError(
            f"Provider {pname!r} has no usable api_key for embeddings "
            f"(used by kb_semantic.use_llm_provider={role!r}).",
        )
    return base_url, api_key


def embed_texts_batched(
    cfg: KBSemanticConfig,
    texts: list[str],
) -> list[list[float]]:
    """Return L2-normalisable raw vectors (client normalises in the index)."""

    if not texts:
        return []

    base_url, api_key = resolve_embedding_http(cfg)
    out: list[list[float]] = []
    bs = max(1, int(cfg.embed_batch_size))

    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        vecs = embeddings_create(
            base_url=base_url,
            api_key=api_key,
            model=cfg.embedding_model,
            inputs=batch,
            dimensions=cfg.embedding_dimensions,
            timeout_s=float(cfg.embed_timeout_s),
        )
        out.extend(vecs)

    if len(out) != len(texts):
        raise EmbeddingServiceError(
            f"Embedding batching mismatch: {len(out)} vs {len(texts)} inputs.",
        )
    return out
