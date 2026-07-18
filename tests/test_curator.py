"""Curator honesty (tracker 2.1): uncompacted gate, high-water mark,
consolidated digest section — against a real KnowledgeStore."""

import json

import pytest

import graph.nodes as nodes
from graph.routers import UNCOMPACTED_TOKEN_THRESHOLD, post_synth_router
from memory import get_store
from memory.schema import EVT_CONSOLIDATED, EVT_TOOL_RESULT


@pytest.fixture()
def kb(tmp_path):
    handle = str(tmp_path / "kb")
    return handle, get_store(handle)


def _seed_result(store, step_id, data):
    return store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        {"step_id": step_id, "tool": "capstone", "ok": True, "data": data},
    )


def _curator_reply(summary="S1"):
    return {
        "summary": summary,
        "addresses_kept": ["$C000"],
        # Deliberately wrong: the node must use ITS mechanically-derived
        # event ids / through_ts, never the LLM's claims.
        "events_compacted": ["llm-invented-id"],
    }


# ---------------------------------------------------------------------------
# gate: uncompacted volume, not total log size
# ---------------------------------------------------------------------------

def test_gate_opens_on_uncompacted_volume_and_closes_after_compaction(
    kb, monkeypatch,
):
    handle, store = kb
    state = {"kb_handle": handle, "plan": [], "tool_results": []}

    # Small backlog → no curation (plan drained → analyst).
    _seed_result(store, "s1", "small")
    assert post_synth_router(state) == "analyst"

    # Big uncompacted backlog → curate.
    big = "x" * (UNCOMPACTED_TOKEN_THRESHOLD * 4 + 10_000)
    _seed_result(store, "s2", big)
    assert store.uncompacted_tool_result_tokens() > UNCOMPACTED_TOKEN_THRESHOLD
    assert post_synth_router(state) == "curate"

    # Curator pass advances the high-water → the gate CLOSES (the old
    # total-size gate could never close: kb.json only grows).
    monkeypatch.setattr(nodes, "_safe_invoke",
                        lambda *a, **kw: _curator_reply())
    nodes.curator_node(state)
    assert store.uncompacted_tool_result_tokens() == 0
    assert post_synth_router(state) == "analyst"


# ---------------------------------------------------------------------------
# high-water: oldest-first, never re-summarises, exact bookkeeping
# ---------------------------------------------------------------------------

def test_curator_compacts_oldest_first_and_never_twice(kb, monkeypatch):
    handle, store = kb
    ids = [_seed_result(store, f"s{i}", f"data-{i}") for i in range(3)]
    state = {"kb_handle": handle, "question": "q"}

    monkeypatch.setattr(nodes, "_safe_invoke",
                        lambda *a, **kw: _curator_reply("S1"))
    nodes.curator_node(state)

    rows = store.query(
        "SELECT payload_json FROM events WHERE kind = ?", (EVT_CONSOLIDATED,),
    )
    assert len(rows) == 1
    payload = json.loads(rows[0]["payload_json"])
    # Mechanical bookkeeping, not the LLM's invented list:
    assert payload["events_compacted"] == ids
    assert payload["through_ts"]

    # Nothing new → nothing to compact (the old code re-summarised the
    # newest 20 every time).
    out = nodes.curator_node(state)
    assert "nothing to compact" in out["messages"][0].content

    # One new event → only that one is compacted next.
    new_id = _seed_result(store, "s9", "fresh-data")
    monkeypatch.setattr(nodes, "_safe_invoke",
                        lambda *a, **kw: _curator_reply("S2"))
    nodes.curator_node(state)
    rows = store.query(
        "SELECT payload_json FROM events WHERE kind = ?"
        " ORDER BY ts ASC", (EVT_CONSOLIDATED,),
    )
    assert len(rows) == 2
    assert json.loads(rows[1]["payload_json"])["events_compacted"] == [new_id]


def test_failed_curator_llm_does_not_mark_anything_compacted(kb, monkeypatch):
    handle, store = kb
    _seed_result(store, "s1", "data")
    monkeypatch.setattr(
        nodes, "_safe_invoke",
        lambda *a, **kw: {"summary": "", "addresses_kept": [],
                          "events_compacted": [], "_error": "all failed"},
    )
    nodes.curator_node({"kb_handle": handle, "question": "q"})
    # No consolidated event → the events stay uncompacted for retry.
    rows = store.query("SELECT COUNT(*) AS n FROM compacted_tool_results")
    assert rows[0]["n"] == 0
    assert len(store.uncompacted_tool_results()) == 1


# ---------------------------------------------------------------------------
# Hardening: exact event-ID bookkeeping, never timestamp cursors
# ---------------------------------------------------------------------------

def test_legacy_consolidated_event_hides_only_what_it_listed(kb):
    """A legacy consolidated event summarised only its newest 20 of 25
    results. The other 5 must stay uncompacted and visible — the old
    timestamp fallback wrongly marked EVERYTHING before it compacted."""
    handle, store = kb
    ids = [_seed_result(store, f"s{i}", f"payload-{i}") for i in range(25)]

    # Legacy shape: no `through_ts`, lists only the newest 20 ids.
    store.append_event(EVT_CONSOLIDATED, "curator", {
        "summary": "legacy summary of the newest 20",
        "addresses_kept": [],
        "events_compacted": ids[5:],
    })

    remaining = store.uncompacted_tool_results(limit=50)
    assert [r["id"] for r in remaining] == ids[:5]   # oldest 5 survive
    assert store.uncompacted_tool_result_tokens() > 0

    # And they stay visible in the digest's raw excerpts.
    excerpts = store.recent_tool_excerpts(limit=50, exclude_compacted=True)
    assert {e["event_id"] for e in excerpts} == set(ids[:5])


def test_equal_timestamps_cannot_skip_events(kb, monkeypatch):
    """Two results sharing one timestamp, batch limit=1: compacting the
    first must leave the second uncompacted (a `ts > boundary` cursor
    lost it)."""
    import memory.store as store_mod
    handle, store = kb

    class _FrozenDatetime:
        @staticmethod
        def now(tz=None):
            from datetime import datetime as _dt
            return _dt(2026, 7, 18, 12, 0, 0)

    monkeypatch.setattr(store_mod, "datetime", _FrozenDatetime)
    id_a = _seed_result(store, "sA", "data-A")
    id_b = _seed_result(store, "sB", "data-B")
    row_a = store.query("SELECT ts FROM events WHERE id=?", (id_a,))
    row_b = store.query("SELECT ts FROM events WHERE id=?", (id_b,))
    assert row_a[0]["ts"] == row_b[0]["ts"]          # genuinely equal

    batch = store.uncompacted_tool_results(limit=1)
    assert len(batch) == 1
    store.append_event(EVT_CONSOLIDATED, "curator", {
        "summary": "S", "addresses_kept": [],
        "events_compacted": [batch[0]["id"]],
    })

    remaining = store.uncompacted_tool_results(limit=10)
    assert [r["id"] for r in remaining] == (
        [id_b] if batch[0]["id"] == id_a else [id_a]
    )


def test_compaction_bookkeeping_survives_full_rebuild(kb, monkeypatch):
    handle, store = kb
    ids = [_seed_result(store, f"s{i}", f"payload-{i}") for i in range(3)]
    monkeypatch.setattr(nodes, "_safe_invoke",
                        lambda *a, **kw: _curator_reply("S1"))
    nodes.curator_node({"kb_handle": handle, "question": "q"})
    assert store.uncompacted_tool_results() == []

    store._full_rebuild()                           # from-scratch replay
    assert store.uncompacted_tool_results() == []
    rows = store.query(
        "SELECT event_id FROM compacted_tool_results ORDER BY event_id",
    )
    assert {r["event_id"] for r in rows} == set(ids)


# ---------------------------------------------------------------------------
# digest: consolidated section restores long-term memory
# ---------------------------------------------------------------------------

def test_digest_shows_consolidated_and_hides_compacted_raw(kb, monkeypatch):
    handle, store = kb
    _seed_result(store, "s1", "MARKER_OLD_EVIDENCE $C05D")
    monkeypatch.setattr(
        nodes, "_safe_invoke",
        lambda *a, **kw: _curator_reply("Consolidated: loop at $C05D"),
    )
    nodes.curator_node({"kb_handle": handle, "question": "q"})
    _seed_result(store, "s2", "MARKER_NEW_EVIDENCE $D020")

    digest = store.digest_for_question("where is the loop?")
    assert "## Consolidated observations" in digest
    assert "Consolidated: loop at $C05D" in digest
    # Compacted raw payload is no longer duplicated in the excerpts …
    assert "MARKER_OLD_EVIDENCE" not in digest
    # … while fresh, uncompacted evidence still is.
    assert "MARKER_NEW_EVIDENCE" in digest
