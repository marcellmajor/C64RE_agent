"""Native LangGraph fan-out for independent read-only steps (tracker 1.7)."""

from __future__ import annotations

import importlib
import threading
from concurrent.futures import ThreadPoolExecutor

from langgraph.types import Send

import graph.nodes as nodes
from graph.nodes import executor_node
from graph.plan_utils import (
    MAX_PARALLEL_TOOL_STEPS,
    is_parallel_read_only_step,
    parallel_runnable_steps,
)
from graph.routers import dispatch_tools
from memory import get_store


def _step(step_id: str, tool: str, args: dict, depends_on=None) -> dict:
    return {
        "id": step_id,
        "goal": step_id,
        "tool": tool,
        "args": args,
        "depends_on": list(depends_on or []),
        "hypothesis": None,
    }


def test_read_only_policy_excludes_external_state_and_code_kb_writes():
    assert is_parallel_read_only_step(_step("c", "capstone", {"mode": "vectors"}))
    assert is_parallel_read_only_step(_step("k", "kb", {"mode": "stats"}))
    assert is_parallel_read_only_step(_step("t", "tavily", {"q": "C64 IRQ"}))
    assert is_parallel_read_only_step(_step("r", "code_kb", {"mode": "routine"}))

    assert not is_parallel_read_only_step(
        _step("v", "vice", {"method": "vice.memory.read", "address": "$C000"}),
    )
    for mode in ("disasm", "annotate", "export"):
        assert not is_parallel_read_only_step(
            _step(mode, "code_kb", {"mode": mode}),
        )


def test_parallel_batch_is_bounded_fresh_concrete_and_dependency_ready():
    plan = [
        _step(f"s{i}", "kb", {"mode": "stats"})
        for i in range(MAX_PARALLEL_TOOL_STEPS + 2)
    ]
    state = {"plan": plan, "tool_results": []}

    assert [step["id"] for step in parallel_runnable_steps(state)] == [
        f"s{i}" for i in range(MAX_PARALLEL_TOOL_STEPS)
    ]

    # A retry must go through serial executor enrichment, and an unresolved
    # argument or dependency cannot join the batch.
    state = {
        "plan": [
            _step("retry", "kb", {"mode": "stats"}),
            _step("null", "kb", {"mode": "text", "q": None}),
            _step("blocked", "kb", {"mode": "stats"}, ["missing"]),
            _step("fresh1", "kb", {"mode": "stats"}),
            _step("fresh2", "capstone", {"mode": "vectors"}),
        ],
        "tool_results": [{
            "step_id": "retry", "tool": "kb", "ok": False,
            "data": "temporary",
        }],
    }
    assert [step["id"] for step in parallel_runnable_steps(state)] == [
        "fresh1", "fresh2",
    ]


def test_executor_and_dispatcher_build_one_send_per_selected_step():
    state = {
        "question": "q",
        "plan": [
            _step("s1", "capstone", {"mode": "vectors"}),
            _step("s2", "kb", {"mode": "stats"}),
        ],
        "tool_results": [],
    }
    selected = executor_node(state)

    assert selected["current_step_id"] is None
    assert selected["current_step_ids"] == ["s1", "s2"]
    merged = {**state, **selected}
    sends = dispatch_tools(merged)
    assert isinstance(sends, list)
    assert all(isinstance(item, Send) for item in sends)
    assert [(item.node, item.arg["current_step_id"]) for item in sends] == [
        ("capstone", "s1"), ("kb", "s2"),
    ]


def test_compiled_graph_runs_branches_concurrently_and_synthesizes_once(
    monkeypatch,
):
    build_module = importlib.import_module("graph.build")
    barrier = threading.Barrier(2)
    visited: list[str] = []
    synth_snapshots: list[list[str]] = []

    monkeypatch.setattr(build_module, "load_inputs", lambda _state: {})
    monkeypatch.setattr(build_module, "planner_node", lambda _state: {
        "plan": [
            _step("s1", "capstone", {"mode": "vectors"}),
            _step("s2", "kb", {"mode": "stats"}),
        ],
        "iteration": 1,
        "plan_blocked": False,
    })
    monkeypatch.setattr(build_module, "executor_node", executor_node)

    def fake_tool(state, tool):
        step_id = str(state["current_step_id"])
        visited.append(step_id)
        # This succeeds only if LangGraph schedules both Send branches at the
        # same time; a serial implementation breaks the barrier.
        barrier.wait(timeout=3)
        return {
            "tool_results": [{
                "step_id": step_id, "tool": tool, "ok": True,
                "data": f"result {step_id}",
            }],
            "tool_call_stats": {tool: {"calls": 1, "failures": 0}},
        }

    monkeypatch.setattr(
        build_module, "capstone_node", lambda state: fake_tool(state, "capstone"),
    )
    monkeypatch.setattr(
        build_module, "kb_query_node", lambda state: fake_tool(state, "kb"),
    )

    def fake_synth(state):
        synth_snapshots.append(sorted(
            str(result["step_id"]) for result in state.get("tool_results", [])
        ))
        return {}

    monkeypatch.setattr(build_module, "synthesizer_node", fake_synth)
    monkeypatch.setattr(build_module, "post_synth_router", lambda _state: "analyst")
    monkeypatch.setattr(build_module, "analyst_node", lambda _state: {
        "candidate_answer": {"answer": "done", "confidence": 1.0},
    })
    monkeypatch.setattr(build_module, "critic_node", lambda _state: {
        "verdict": {"decision": "accept"},
    })
    monkeypatch.setattr(build_module, "verdict_router", lambda _state: "accept")
    monkeypatch.setattr(build_module, "write_report", lambda _state: {})

    graph = build_module.build_graph().compile()
    final = graph.invoke({"question": "q", "tool_results": []})

    assert sorted(visited) == ["s1", "s2"]
    assert synth_snapshots == [["s1", "s2"]]
    assert final["tool_call_stats"] == {
        "capstone": {"calls": 1, "failures": 0},
        "kb": {"calls": 1, "failures": 0},
    }


def test_real_capstone_and_parent_kb_reads_share_store_safely(tmp_path):
    dump_path = tmp_path / "parallel.dump"
    dump_path.write_bytes(bytes(0x10000))
    handle = str(tmp_path / "kb")
    store = get_store(handle)
    store.ingest_dump(dump_path)
    plan = [
        _step("s1", "capstone", {"mode": "vectors"}),
        _step("s2", "kb", {"mode": "stats"}),
    ]
    base = {"kb_handle": handle, "plan": plan}

    with ThreadPoolExecutor(max_workers=2) as pool:
        cap_future = pool.submit(
            nodes.capstone_node, {**base, "current_step_id": "s1"},
        )
        kb_future = pool.submit(
            nodes.kb_query_node, {**base, "current_step_id": "s2"},
        )
        results = [cap_future.result(), kb_future.result()]

    assert [result["tool_results"][0]["ok"] for result in results] == [
        True, True,
    ]
