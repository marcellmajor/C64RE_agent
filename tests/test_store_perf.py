"""KnowledgeStore persistence/scan improvements (tracker 1.5 / 1.6).

Uses `KnowledgeStore.load_or_init` directly (not `get_store`) so several
instances can be opened against the same on-disk root, simulating
process restarts.
"""

import json

import pytest

from memory.schema import EVT_LABEL, EVT_TOOL_RESULT, SCHEMA_VERSION
from memory.store import KnowledgeStore


@pytest.fixture()
def rebuild_counter(monkeypatch):
    """Count _full_rebuild invocations while keeping its behaviour."""
    calls: list[int] = []
    orig = KnowledgeStore._full_rebuild

    def _counted(self):
        calls.append(1)
        orig(self)

    monkeypatch.setattr(KnowledgeStore, "_full_rebuild", _counted)
    return calls


def _label(addr, name="x"):
    return {"addr": addr, "name": name, "confidence": 0.5}


# ---------------------------------------------------------------------------
# incremental replay (1.6)
# ---------------------------------------------------------------------------

def test_reopen_is_incremental_not_full_rebuild(tmp_path, rebuild_counter):
    root = tmp_path / "kb"
    s1 = KnowledgeStore.load_or_init(root)   # empty log → one initial rebuild
    initial_rebuilds = len(rebuild_counter)
    s1.append_event(EVT_LABEL, "t", _label(1, "a"))
    s1.append_event(EVT_LABEL, "t", _label(2, "b"))

    s2 = KnowledgeStore.load_or_init(root)   # "process restart"
    assert len(rebuild_counter) == initial_rebuilds  # no second full replay
    assert s2.stats()["events_total"] == 2
    assert {r["name"] for r in s2.query("SELECT name FROM labels")} == {"a", "b"}


def test_reopen_replays_only_the_foreign_tail(tmp_path, rebuild_counter):
    """Events appended by another process (raw JSONL line) are picked up
    incrementally from the recorded offset."""
    root = tmp_path / "kb"
    s1 = KnowledgeStore.load_or_init(root)
    s1.append_event(EVT_LABEL, "t", _label(1, "a"))
    before = len(rebuild_counter)

    foreign = {
        "id": "feedc0ffee11", "ts": "2026-07-17T00:00:00+00:00",
        "kind": EVT_LABEL, "source": "other-process",
        "payload": _label(2, "b"),
    }
    with (root / "kb.json").open("a") as f:
        f.write(json.dumps(foreign) + "\n")

    s2 = KnowledgeStore.load_or_init(root)
    assert len(rebuild_counter) == before          # incremental, not full
    assert s2.stats()["events_total"] == 2
    assert {r["name"] for r in s2.query("SELECT name FROM labels")} == {"a", "b"}


def test_truncated_log_triggers_full_rebuild(tmp_path, rebuild_counter):
    root = tmp_path / "kb"
    s1 = KnowledgeStore.load_or_init(root)
    s1.append_event(EVT_LABEL, "t", _label(1, "a"))
    s1.append_event(EVT_LABEL, "t", _label(2, "b"))
    before = len(rebuild_counter)

    # Replace the log with a shorter one (offset now beyond EOF).
    lines = (root / "kb.json").read_text().splitlines()
    (root / "kb.json").write_text(lines[0] + "\n")

    s2 = KnowledgeStore.load_or_init(root)
    assert len(rebuild_counter) == before + 1
    assert s2.stats()["events_total"] == 1


def test_same_size_replacement_triggers_rebuild(tmp_path, rebuild_counter):
    """Review finding 6: `offset == size` alone is not proof of currency —
    a replaced log with identical byte length must rebuild."""
    root = tmp_path / "kb"
    s1 = KnowledgeStore.load_or_init(root)
    s1.append_event(EVT_LABEL, "t", _label(1, "aa"))
    s1.append_event(EVT_LABEL, "t", _label(2, "cc"))
    before = len(rebuild_counter)

    log = root / "kb.json"
    original = log.read_text()
    replaced = original.replace('"name": "aa"', '"name": "bb"')
    assert len(replaced) == len(original)  # same byte size, new content
    log.write_text(replaced)

    s2 = KnowledgeStore.load_or_init(root)
    assert len(rebuild_counter) == before + 1
    names = {r["name"] for r in s2.query("SELECT name FROM labels")}
    assert names == {"bb", "cc"}  # SQLite matches the NEW kb.json


def test_prefix_edit_triggers_rebuild(tmp_path, rebuild_counter):
    """Editing the already-replayed prefix (size unchanged) must rebuild."""
    root = tmp_path / "kb"
    s1 = KnowledgeStore.load_or_init(root)
    s1.append_event(EVT_LABEL, "t", _label(1, "first"))
    s1.append_event(EVT_LABEL, "t", _label(2, "second"))
    before = len(rebuild_counter)

    log = root / "kb.json"
    lines = log.read_text().splitlines(keepends=True)
    lines[0] = lines[0].replace('"name": "first"', '"name": "FIRST"')
    log.write_text("".join(lines))

    s2 = KnowledgeStore.load_or_init(root)
    assert len(rebuild_counter) == before + 1
    names = {r["name"] for r in s2.query("SELECT name FROM labels")}
    assert names == {"FIRST", "second"}


def test_schema_version_mismatch_triggers_full_rebuild(tmp_path,
                                                       rebuild_counter):
    root = tmp_path / "kb"
    s1 = KnowledgeStore.load_or_init(root)
    s1.append_event(EVT_LABEL, "t", _label(1, "a"))
    s1._meta_set("schema_version", "stale-version")
    s1._db.commit()
    before = len(rebuild_counter)

    s2 = KnowledgeStore.load_or_init(root)
    assert len(rebuild_counter) == before + 1
    assert s2._meta_get("schema_version") == SCHEMA_VERSION
    assert s2.stats()["events_total"] == 1


# ---------------------------------------------------------------------------
# indexed tool-result dedup (1.6)
# ---------------------------------------------------------------------------

def test_has_tool_result_full_content_identity(tmp_path):
    store = KnowledgeStore.load_or_init(tmp_path / "kb")
    prefix = "$C000: NOP\n" * 30  # > 256 chars of shared opening lines
    payload = {"step_id": "i1_s1", "tool": "capstone", "ok": True,
               "data": prefix + "$C100: RTS"}
    assert store.has_tool_result(payload) is False
    store.append_event(EVT_TOOL_RESULT, "synthesizer", payload)
    assert store.has_tool_result(payload) is True
    # Same step/tool, evidence differs past byte 256 → distinct result.
    other = {**payload, "data": prefix + "$C100: DEC $0780"}
    assert store.has_tool_result(other) is False


# ---------------------------------------------------------------------------
# ingest idempotency without full-log scans (1.6)
# ---------------------------------------------------------------------------

def test_ingest_dump_and_text_idempotency(tmp_path):
    store = KnowledgeStore.load_or_init(tmp_path / "kb")
    dump = tmp_path / "game.bin"
    dump.write_bytes(b"\x00" * 64)
    assert store.ingest_dump(dump) is not None
    assert store.ingest_dump(dump) is None          # path already recorded

    note = tmp_path / "note.txt"
    note.write_text("lives at $0780")
    assert store.ingest_text_file(note) is not None
    assert store.ingest_text_file(note) is None     # mtime unchanged
    import os
    os.utime(note, (note.stat().st_atime, note.stat().st_mtime + 5))
    assert store.ingest_text_file(note) is not None  # newer mtime → re-ingest


# ---------------------------------------------------------------------------
# memoization (1.5)
# ---------------------------------------------------------------------------

def test_digest_memoized_until_kb_changes(tmp_path):
    store = KnowledgeStore.load_or_init(tmp_path / "kb")
    store.append_event(EVT_LABEL, "t", _label(0x0780, "lives"))
    d1 = store.digest_for_question("where is the lives counter?")
    d2 = store.digest_for_question("where is the lives counter?")
    assert d1 is d2                                   # cache hit
    assert store.digest_for_question("other question?") is not d2
    store.append_event(EVT_LABEL, "t", _label(0x0781, "score"))
    d3 = store.digest_for_question("where is the lives counter?")
    # Invalidated by KB growth: fresh build reflecting the new event count
    # ("score" itself is correctly filtered out by question relevance).
    assert d3 is not d1 and "events_total: 2" in d3


def test_partial_asm_excerpt_cached(tmp_path):
    store = KnowledgeStore.load_or_init(tmp_path / "kb")
    asm = tmp_path / "game.asm"
    asm.write_text("$C000 game_entry\n$C010 update_lives\n")
    store.ingest_partial_asm(asm)
    e1 = store.partial_asm_excerpt(max_lines=80)
    assert "game_entry" in e1
    assert store.partial_asm_excerpt(max_lines=80) is e1  # cache hit
