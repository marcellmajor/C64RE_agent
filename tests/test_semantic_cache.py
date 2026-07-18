"""Persistent semantic-vector cache regressions (tracker 2.5)."""

from __future__ import annotations

from dataclasses import replace

import pytest

import memory.semantic_cache as semantic_cache
import memory.store as store_module
from memory.semantic_cache import embed_texts_cached
from memory.semantic_config import KBSemanticConfig


def _config(**overrides) -> KBSemanticConfig:
    base = KBSemanticConfig(
        enabled=True,
        use_llm_provider="executor",
        embedding_model="test-embedding-v1",
        embedding_dimensions=3,
        chunk_chars=700,
        chunk_overlap=120,
        embed_batch_size=48,
        embed_timeout_s=120,
        digest_semantic_limit=8,
        semantic_search_limit_default=12,
        index_event_kinds=frozenset({"ingest_text"}),
    )
    return replace(base, **overrides)


def test_cache_embeds_only_unique_content_misses(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_embed(_cfg, texts):
        calls.append(list(texts))
        return [[float(len(text)), 1.0, 2.0] for text in texts]

    monkeypatch.setattr(semantic_cache, "embed_texts_batched", fake_embed)
    cache_path = tmp_path / "vectors.sqlite"
    cfg = _config()

    first = embed_texts_cached(cfg, ["alpha", "beta", "alpha"], cache_path)
    second = embed_texts_cached(cfg, ["beta", "alpha"], cache_path)
    third = embed_texts_cached(cfg, ["alpha", "gamma"], cache_path)

    assert calls == [["alpha", "beta"], ["gamma"]]
    assert first[0] == first[2]
    assert second == [first[1], first[0]]
    assert third[0] == first[0]


def test_embedding_model_or_dimension_change_is_a_cache_miss(
    tmp_path, monkeypatch,
):
    calls = 0

    def fake_embed(_cfg, texts):
        nonlocal calls
        calls += 1
        return [[1.0] * int(_cfg.embedding_dimensions or 2) for _ in texts]

    monkeypatch.setattr(semantic_cache, "embed_texts_batched", fake_embed)
    path = tmp_path / "vectors.sqlite"
    embed_texts_cached(_config(), ["same text"], path)
    embed_texts_cached(_config(), ["same text"], path)
    embed_texts_cached(
        _config(embedding_model="test-embedding-v2"), ["same text"], path,
    )
    embed_texts_cached(
        _config(embedding_dimensions=4), ["same text"], path,
    )

    assert calls == 3


def test_store_reopen_hydrates_vectors_without_reembedding(
    tmp_path, monkeypatch,
):
    cfg = _config(chunk_chars=128, chunk_overlap=0)
    calls: list[list[str]] = []

    def fake_embed(_cfg, texts):
        calls.append(list(texts))
        return [[float(index + 1), 0.5, 0.25] for index, _ in enumerate(texts)]

    monkeypatch.setattr(store_module, "load_semantic_config", lambda: cfg)
    monkeypatch.setattr(semantic_cache, "embed_texts_batched", fake_embed)

    root = tmp_path / "kb"
    note = tmp_path / "notes.txt"
    note.write_text("player animation state machine and sprite update")
    first = store_module.KnowledgeStore.load_or_init(root)
    first.ingest_text_file(note)
    call_count = len(calls)
    assert call_count == 1
    assert first.semantic_search_enabled()
    assert (root / "vectors.sqlite").is_file()
    assert first._db is not None
    first._db.close()

    reopened = store_module.KnowledgeStore.load_or_init(root)
    assert len(calls) == call_count
    assert reopened.semantic_search_enabled()
    assert reopened._semantic_index is not None
    assert len(reopened._semantic_index) >= 1


def test_float32_cache_round_trip_is_numerically_stable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        semantic_cache,
        "embed_texts_batched",
        lambda _cfg, _texts: [[0.123456789, -0.25, 3.5]],
    )
    path = tmp_path / "vectors.sqlite"
    first = embed_texts_cached(_config(), ["precision"], path)[0]
    second = embed_texts_cached(_config(), ["precision"], path)[0]

    assert second == pytest.approx(first, rel=1e-6, abs=1e-7)
