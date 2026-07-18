"""Bounded checkpoint tool-result state (tracker 4.8)."""

from __future__ import annotations

import graph.nodes as nodes
from graph.plan_utils import pending_steps, runnable_steps, step_status
from graph.state import (
    TOOL_RESULTS_STATE_LIMIT,
    compact_tool_results_update,
    merge_tool_call_stats,
    reduce_tool_results,
)
from memory import get_store


def _result(step: str, *, ok: bool = True, data: str = "evidence") -> dict:
    return {
        "step_id": step,
        "tool": "capstone",
        "ok": ok,
        "mode": "linear",
        "data": data,
    }


def test_tool_result_reducer_bounds_payloads_but_keeps_status_ledger():
    rows = []
    for index in range(TOOL_RESULTS_STATE_LIMIT + 40):
        rows = reduce_tool_results(rows, [_result(f"s{index}")])

    assert len(rows) == TOOL_RESULTS_STATE_LIMIT + 40
    assert rows[0]["step_id"] == "s0"
    assert rows[0]["_status_only"] is True
    assert "data" not in rows[0]
    assert sum("data" in row for row in rows) == TOOL_RESULTS_STATE_LIMIT
    assert rows[-1]["step_id"] == f"s{TOOL_RESULTS_STATE_LIMIT + 39}"
    assert len({row["_state_result_id"] for row in rows}) == len(rows)


def test_evicted_success_stays_done_and_satisfies_dependency():
    rows = reduce_tool_results([], [_result("old_success")])
    for index in range(TOOL_RESULTS_STATE_LIMIT + 8):
        rows = reduce_tool_results(rows, [_result(f"noise_{index}")])

    state = {
        "plan": [
            {"id": "old_success", "tool": "kb", "depends_on": []},
            {
                "id": "dependent", "tool": "kb",
                "args": {"mode": "stats"},
                "depends_on": ["old_success"],
            },
        ],
        "tool_results": rows,
    }

    old = next(row for row in rows if row["step_id"] == "old_success")
    assert old["_status_only"] is True
    assert "data" not in old
    assert [step["id"] for step in pending_steps(state)] == ["dependent"]
    assert [step["id"] for step in runnable_steps(state)] == ["dependent"]


def test_identical_retries_remain_distinct_status_attempts():
    failure = _result("s1", ok=False, data="temporary failure")
    rows = reduce_tool_results([], [failure, failure])

    assert len(rows) == 2
    assert rows[0]["_state_result_id"] != rows[1]["_state_result_id"]
    assert step_status(rows).fail_counts == {"s1": 2}


def test_persisted_results_compact_without_losing_execution_ledger():
    rows = reduce_tool_results([], [
        _result("s1", data="X" * 20_000),
        _result("s2", ok=False, data="repair detail " + "Y" * 2_000),
    ])
    event_ids = {
        rows[0]["_state_result_id"]: "evt_success",
        rows[1]["_state_result_id"]: "evt_failure",
    }

    compacted = reduce_tool_results(
        rows, compact_tool_results_update(event_ids),
    )

    assert "data" not in compacted[0]
    assert compacted[0]["_event_id"] == "evt_success"
    assert compacted[1]["_event_id"] == "evt_failure"
    assert compacted[1]["data"].startswith("repair detail")
    assert len(compacted[1]["data"]) <= 1_001
    status = step_status(compacted)
    assert status.succeeded == {"s1"}
    assert status.fail_counts == {"s2": 1}


def test_tool_call_stats_survive_result_eviction():
    stats = {}
    stats = merge_tool_call_stats(stats, {
        "capstone": {"calls": 100, "failures": 3},
    })
    stats = merge_tool_call_stats(stats, {
        "capstone": {"calls": 2, "failures": 1},
        "kb": {"calls": 4, "failures": 0},
    })

    assert stats == {
        "capstone": {"calls": 102, "failures": 4},
        "kb": {"calls": 4, "failures": 0},
    }


def test_synthesizer_identity_cursor_handles_compacted_prefix(
    tmp_path, monkeypatch,
):
    handle = str(tmp_path / "kb")
    get_store(handle)
    prompts: list[str] = []

    def fake_invoke(_role, prompt, fallback, **_kwargs):
        prompts.append(prompt)
        return dict(fallback)

    monkeypatch.setattr(nodes, "_safe_invoke", fake_invoke)
    first = _result("s1", data="$C000: LDA #$01")
    state = {
        "kb_handle": handle,
        "question": "what does this do?",
        "plan": [{"id": "s1", "tool": "capstone", "depends_on": []}],
        "tool_results": reduce_tool_results([], [first]),
        "synth_processed_count": 0,
    }

    first_out = nodes.synthesizer_node(state)
    state["tool_results"] = reduce_tool_results(
        state["tool_results"], first_out["tool_results"],
    )
    state["synth_processed_result_ids"] = first_out[
        "synth_processed_result_ids"
    ]
    state["synth_processed_count"] = first_out["synth_processed_count"]
    assert "data" not in state["tool_results"][0]

    second = _result("s2", data="$C010: STA $D020")
    state["plan"] = [
        *state["plan"],
        {"id": "s2", "tool": "capstone", "depends_on": []},
    ]
    state["tool_results"] = reduce_tool_results(
        state["tool_results"], [second],
    )
    second_out = nodes.synthesizer_node(state)

    assert len(prompts) == 2
    assert "$C010: STA $D020" in prompts[1]
    assert "$C000: LDA #$01" not in prompts[1]
    assert second_out["synth_processed_count"] == 2
