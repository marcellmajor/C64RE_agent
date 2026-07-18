"""Recursion-exhaustion recovery in BOTH runners (tracker 0.1 review).

A fake compiled graph raises GraphRecursionError mid-stream; each runner
must normalize the checkpoint state (explicit termination reason, capped
confidence, truncation open-question) and feed THAT state to both
write_report and its own output (CLI print / TurnResult).
"""

from types import SimpleNamespace

import pytest
from langgraph.errors import GraphRecursionError

from graph.plan_utils import TRUNCATED_CONFIDENCE_CAP


_CHECKPOINT_STATE = {
    "game": "Wizard of Wor",
    "question": "where is the score stored?",
    "candidate_answer": {
        "answer": "probably at $0790",
        "confidence": 0.92,          # never critic-accepted → must be capped
        "evidence": [],
        "open_questions": [],
    },
    "verdict": {"decision": "replan", "critique": "keep digging"},
    "iteration": 4,
    "messages": [],
}


class FakeGraph:
    """Streams one event then raises GraphRecursionError."""

    def __init__(self, state):
        self._state = dict(state)

    def stream(self, initial_state, config, stream_mode="values"):
        yield dict(self._state)
        raise GraphRecursionError("recursion limit reached")

    def get_state(self, config):
        return SimpleNamespace(values=dict(self._state))

    # main.py compiles with a checkpointer; the stub must accept it.
    def compile(self, checkpointer=None):
        return self


@pytest.fixture()
def report_capture(monkeypatch):
    """Capture the state handed to write_report without touching disk."""
    import graph.nodes as nodes

    captured: list[dict] = []

    def _fake_write_report(state):
        captured.append(dict(state))
        return {"messages": []}

    monkeypatch.setattr(nodes, "write_report", _fake_write_report)
    return captured


def _assert_normalized(state):
    assert state["termination_reason"] == "recursion_exhausted"
    cand = state["candidate_answer"]
    assert cand["confidence"] == TRUNCATED_CONFIDENCE_CAP
    assert any("Research truncated" in q for q in cand["open_questions"])
    assert state["verdict"]["decision"] == "recursion_exhausted"
    assert state["verdict"]["prior_decision"] == "replan"


def test_agent_runner_recovers_with_normalized_turnresult(
    monkeypatch, report_capture,
):
    from tools import agent_runner

    monkeypatch.setattr(
        agent_runner, "_compiled_graph",
        lambda: FakeGraph(_CHECKPOINT_STATE),
    )

    events = list(agent_runner.run_question(
        game="Wizard of Wor",
        question="where is the score stored?",
        dump_path="memdump_dir/does_not_matter.bin",
    ))

    kinds = [k for k, _ in events]
    assert "error" not in kinds, "recursion must degrade, not error out"
    done = [payload for k, payload in events if k == "done"]
    assert len(done) == 1
    result = done[0]

    assert result.verdict == "recursion_exhausted"
    assert result.confidence == TRUNCATED_CONFIDENCE_CAP
    assert any("Research truncated" in q for q in result.open_questions)
    _assert_normalized(result.raw_state)

    # write_report received the SAME normalized state.
    assert len(report_capture) == 1
    _assert_normalized(report_capture[0])


def test_main_cli_recovers_with_normalized_report(
    monkeypatch, capsys, report_capture, tmp_path,
):
    import main as main_mod

    monkeypatch.setattr(
        main_mod, "build_graph", lambda: FakeGraph(_CHECKPOINT_STATE),
    )
    # Keep the durable checkpointer (tracker 2.4) out of the repo's real
    # sessions/ tree.
    monkeypatch.setattr(main_mod, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", [
        "main.py",
        "--game", "Wizard of Wor",
        "--question", "where is the score stored?",
        "--dump", "memdump_dir/does_not_matter.bin",
        "--no-trace",
    ])

    main_mod.main()
    out = capsys.readouterr().out

    # CLI slug (0.7) + per-question durable thread (2.4).
    assert "thread_id=wizard_of_wor-" in out
    # Durable checkpoint file created under the (redirected) session dir.
    assert (tmp_path / "wizard_of_wor" / "checkpoint.sqlite").exists()
    # The printed verdict reflects the truncation, not the stale checkpoint.
    assert "recursion_exhausted" in out
    assert "Research truncated" in out

    assert len(report_capture) == 1
    _assert_normalized(report_capture[0])
