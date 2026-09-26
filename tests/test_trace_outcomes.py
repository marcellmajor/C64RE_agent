"""Watchpoint observation outcomes, durable status, and visible diagnostics."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import graph.nodes as nodes
from graph.plan_utils import failed_step_notes, runnable_steps, step_status
from graph.state import TOOL_RESULTS_STATE_LIMIT, compact_tool_results_update, reduce_tool_results


@pytest.fixture
def trace(monkeypatch):
    # Advance only a fake wall clock; no emulator, API calls, or real sleeps.
    clock = [0.0]
    def sleep(seconds):
        clock[0] += seconds
    monkeypatch.setattr(nodes, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(nodes, "_usage_update", lambda: {})

    def run(*, polls=None, run_error=None, checkpoint_id=1):
        calls = []
        responses = iter(polls) if polls else None
        def call(method, args):
            calls.append(method)
            if method == "vice.ping":
                return {"data": {"execution": "paused"}}
            if method == "vice.checkpoint.add":
                return {"data": {"checkpoint_num": checkpoint_id}}
            if method == "vice.execution.run" and run_error:
                raise run_error
            if method == "vice.checkpoint.list":
                result = next(responses) if responses else {
                    "checkpoints": [{"checkpoint_num": 1, "hit_count": 0}],
                }
                if isinstance(result, Exception):
                    raise result
                return {"data": result}
            if method == "vice.registers.get":
                return {"data": {"PC": 0x73AF}}
            if method == "vice.memory.read":
                return {"data": {"data_hex": "eaeaea"}}
            return {"data": "ok"}
        monkeypatch.setattr(nodes, "_vice_call", call)
        result = nodes._vice_trace({"address": "$D000", "timeout_s": 0.1}, "i2_s2")
        return result, calls
    return run


def test_timeout_is_visible_inconclusive_and_not_automatically_retried(trace):
    out, calls = trace()
    result = out["tool_results"][0]
    assert result["outcome"] == "inconclusive"
    assert result["ok"] is False and result["hit_confirmed"] is False
    assert result["retryable"] is False
    assert result["writer_pc"] is None and result["writer_proven"] is False
    assert out["tool_call_stats"]["vice"] == {"calls": 1, "failures": 0}
    message = out["messages"][0].content
    assert message.splitlines()[0] == (
        "[tool:vice] ⚠ step=i2_s2 — No confirmed write to $D000 within 0.10s — inconclusive."
    )
    assert result["data"] in message
    assert "vice.checkpoint.delete" in calls

    state = {"plan": [
        {"id": "i2_s2", "tool": "vice", "depends_on": []},
        {"id": "dependent", "tool": "kb", "depends_on": ["i2_s2"]},
    ], "tool_results": [result]}
    status = step_status([result])
    assert status.inconclusive == {"i2_s2"}
    assert status.done == {"i2_s2"}
    assert not status.succeeded and not status.fail_counts and not status.exhausted
    assert runnable_steps(state) == []
    assert "INCONCLUSIVE" in failed_step_notes(state)[0]
    blocked = nodes.executor_node(state)
    assert blocked["plan_blocked"] is True
    assert "was inconclusive" in blocked["tool_results"][0]["data"]


@pytest.mark.parametrize("response,expected", [
    (RuntimeError("monitor disconnected"), "monitor disconnected"),
    ({"checkpoints": [{"checkpoint_num": 2, "hit_count": 12}]}, "valid hit count"),
    ({"checkpoints": [{"checkpoint_num": 1}]}, "valid hit count"),
    ({"checkpoints": [{"checkpoint_num": 1, "hit_count": "unknown"}]}, "valid hit count"),
])
def test_polling_errors_are_not_reported_as_no_write(trace, response, expected):
    out, calls = trace(polls=[response, response])
    result = out["tool_results"][0]
    assert result["outcome"] == "error"
    assert result["ok"] is False and result["hit_confirmed"] is False
    assert result["poll_error_count"] == 2
    assert expected in result["poll_error"] and expected in result["data"]
    assert "no confirmed write within" not in result["data"].lower()
    assert "checkpoint polling failed" in out["messages"][0].content.splitlines()[0]
    assert "[tool:vice] ✗" in out["messages"][0].content
    assert out["tool_call_stats"]["vice"]["failures"] == 1
    assert "vice.checkpoint.delete" in calls


def test_transient_poll_error_remains_visible_after_a_confirmed_hit(trace):
    out, _ = trace(polls=[RuntimeError("temporary outage"), {
        "checkpoints": [{"checkpoint_num": 1, "hit_count": "1"}],
    }])
    result = out["tool_results"][0]
    assert result["outcome"] == "confirmed" and result["ok"] is True
    assert result["hit_confirmed"] is True and result["writer_proven"] is False
    assert result["poll_error_count"] == 1
    assert "temporary outage" in result["data"]


def test_run_failure_remains_an_error_even_with_valid_zero_hit_polls(trace):
    out, calls = trace(run_error=RuntimeError("emulator wedged"))
    result = out["tool_results"][0]
    assert result["outcome"] == "error"
    assert "could not resume" in result["summary"]
    assert "emulator wedged" in result["data"]
    assert result["poll_error_count"] == 0
    assert "vice.checkpoint.delete" in calls


def test_missing_checkpoint_id_is_an_error_and_not_automatically_retried(trace):
    out, _ = trace(checkpoint_id=None)
    result = out["tool_results"][0]
    assert result["outcome"] == "error" and result["retryable"] is False
    assert "no checkpoint ID" in result["summary"]
    assert "may still be armed" in result["data"]


def test_inconclusive_survives_compaction_and_eviction(trace):
    out, _ = trace()
    rows = reduce_tool_results([], out["tool_results"])
    rows = reduce_tool_results(rows, compact_tool_results_update({
        rows[0]["_state_result_id"]: "event-trace",
    }))
    assert rows[0]["outcome"] == "inconclusive"
    rows = reduce_tool_results(rows, [
        {"step_id": f"noise-{i}", "tool": "kb", "ok": True, "data": "ok"}
        for i in range(TOOL_RESULTS_STATE_LIMIT + 1)
    ])
    assert rows[0]["_status_only"] is True
    assert "data" not in rows[0]
    assert rows[0]["outcome"] == "inconclusive"
    assert "inconclusive" in rows[0]["summary"]
    assert step_status(rows).inconclusive == {"i2_s2"}
    assert "i2_s2" not in step_status(rows).fail_counts


def test_inconclusive_report_uses_warning_and_does_not_count_failure(trace, monkeypatch, tmp_path):
    out, _ = trace()
    monkeypatch.setattr(nodes, "SESSIONS_DIR", tmp_path)
    nodes.write_report({"game": "Trace Test", "tool_results": out["tool_results"]})
    report = (tmp_path / "trace_test" / "report.md").read_text()
    assert "- ⚠ **vice** `i2_s2`" in report
    assert "No confirmed write to $D000 within 0.10s — inconclusive." in report
    assert "1 call(s), 0 failure(s)" in report


@pytest.mark.parametrize("poll_error,icon", [(False, "⚠️"), (True, "❌")])
def test_streamlit_shows_reason_and_expandable_full_diagnostics(trace, poll_error, icon):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    out, _ = trace(polls=[RuntimeError("monitor disconnected")] * 2 if poll_error else None)
    message = out["messages"][0].content
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    script = f"from runpy import run_path\nns = run_path({str(app_path)!r})\nns['_render_agent_msg']({message!r})"
    at = AppTest.from_string(script, default_timeout=20).run()
    assert not at.exception, [e.value for e in at.exception]
    expected = "checkpoint polling failed" if poll_error else "inconclusive"
    assert any(icon in m.value and expected in m.value for m in at.markdown)
    assert any(e.label == "Diagnostics · vice · i2_s2" for e in at.expander)
    assert any(c.value == out["tool_results"][0]["data"] for c in at.code)


def test_later_confirmed_hit_satisfies_previously_inconclusive_step():
    result = {"step_id": "s1", "ok": False, "outcome": "inconclusive"}
    status = step_status([result, {"step_id": "s1", "ok": True}])
    assert status.succeeded == {"s1"} and not status.inconclusive
