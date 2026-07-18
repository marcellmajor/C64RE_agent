"""Durable per-run outcome telemetry (tracker 5.4)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import graph.nodes as nodes
from memory import get_store
from memory.schema import EVT_RUN_SUMMARY


def _completed_state(handle: str) -> dict:
    return {
        "run_id": "run-phase5-001",
        "run_started_at": (
            datetime.now(timezone.utc) - timedelta(seconds=2)
        ).isoformat(),
        "game": "Test Game",
        "question": "Where is the lives counter?",
        "kb_handle": handle,
        "candidate_answer": {
            "answer": "Lives are stored at $008B.",
            "confidence": 0.85,
            "evidence": [],
            "open_questions": [],
        },
        "verdict": {"decision": "accept", "critique": "supported"},
        "iteration": 2,
        "plan": [],
        "tool_results": [
            {"step_id": "s1", "tool": "capstone", "ok": True, "data": "x"},
            {"step_id": "s2", "tool": "kb", "ok": True, "data": "y"},
        ],
        "llm_usage": [
            {"role": "planner", "input_tokens": 100, "output_tokens": 40},
            {"role": "analyst", "input_tokens": 80, "output_tokens": 30},
        ],
        "budget_used": 0.125,
        "tokens_used": 250,
    }


def test_write_report_records_one_idempotent_run_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "SESSIONS_DIR", tmp_path)
    handle = str(tmp_path / "test_game" / "kb")
    state = _completed_state(handle)

    nodes.write_report(state)
    nodes.write_report(state)  # report retry must not duplicate the run

    store = get_store(handle)
    rows = store.query("SELECT * FROM run_summaries")
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "run-phase5-001"
    assert row["question"] == "Where is the lives counter?"
    assert row["verdict"] == "accept"
    assert row["confidence"] == 0.85
    assert row["iterations"] == 2
    assert row["cost_usd"] == 0.125
    assert row["tokens"] == 250
    assert row["llm_calls"] == 2
    assert row["tool_calls"] == 2
    assert row["elapsed_s"] >= 1.0
    assert row["report_path"].endswith("test_game/report.md")
    assert store.stats()["run_summaries"] == 1

    events = store.query(
        "SELECT payload_json FROM events WHERE kind = ?", (EVT_RUN_SUMMARY,),
    )
    assert len(events) == 1
    assert json.loads(events[0]["payload_json"])["run_id"] == "run-phase5-001"


def test_run_summary_replays_and_appears_in_question_digest(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "SESSIONS_DIR", tmp_path)
    handle = str(tmp_path / "test_game" / "kb")
    state = _completed_state(handle)
    nodes.write_report(state)

    store = get_store(handle)
    store._full_rebuild()
    rows = store.recent_run_summaries("Where is the lives counter?")
    assert len(rows) == 1 and rows[0]["run_id"] == "run-phase5-001"

    digest = store.digest_for_question("Where is the lives counter?")
    assert "## Prior completed runs" in digest
    assert "run-phase5-001" in digest
    assert "verdict=accept confidence=0.85" in digest
    assert "Lives are stored at $008B" in digest


def test_run_summary_is_bookkeeping_not_substantive_growth(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "SESSIONS_DIR", tmp_path)
    handle = str(tmp_path / "test_game" / "kb")
    state = _completed_state(handle)
    store = get_store(handle)
    before = nodes._substantive_event_count(store)
    nodes.write_report(state)
    assert nodes._substantive_event_count(store) == before


def test_load_inputs_rotates_run_identity_after_completed_checkpoint(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(nodes, "SESSIONS_DIR", tmp_path)
    completed = {
        "game": "Test Game",
        "question": "Where is the lives counter?",
        "run_id": "completed-run",
        "run_started_at": "2026-01-01T00:00:00+00:00",
        "run_completed_at": "2026-01-01T00:00:05+00:00",
    }

    loaded = nodes.load_inputs(completed)

    assert loaded["run_id"] != "completed-run"
    assert loaded["run_started_at"] != completed["run_started_at"]
    assert loaded["run_completed_at"] is None
