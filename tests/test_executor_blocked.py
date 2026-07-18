"""Executor blocked-plan handling + routing (tracker 0.6) — no LLM.

Integration-level: executor output is merged into state the way the
LangGraph reducers would (`tool_results` appends, scalars replace) and
then fed to `route_tool`, proving blocked plans route deterministically
to a replan bounded by MAX_ITERS.
"""

import pytest

import graph.nodes as nodes
from graph.plan_utils import MAX_PLAN_STEPS, MAX_STEP_ATTEMPTS
from graph.routers import MAX_ITERS, route_tool


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """The blocked/complete paths must return before any LLM call."""
    def _boom(*a, **kw):  # pragma: no cover
        raise AssertionError("executor_node called the LLM on a blocked plan")
    monkeypatch.setattr(nodes, "_safe_invoke", _boom)
    monkeypatch.setattr(nodes, "_kb_digest_for_state", lambda s: "(digest)")


def _merge(state, update):
    """Apply a node's state update like the graph reducers would."""
    merged = dict(state)
    for k, v in (update or {}).items():
        if k == "tool_results":
            merged[k] = list(merged.get(k) or []) + list(v)
        elif k == "messages":
            continue  # not needed for routing assertions
        else:
            merged[k] = v
    return merged


def _fail(sid, **kw):
    return {"step_id": sid, "tool": "kb", "ok": False, "data": "boom", **kw}


def test_completed_plan_returns_no_step_and_not_blocked():
    state = {
        "plan": [{"id": "s1", "tool": "kb", "depends_on": []}],
        "tool_results": [{"step_id": "s1", "tool": "kb", "ok": True,
                          "data": "x"}],
    }
    out = nodes.executor_node(state)
    assert out["current_step_id"] is None
    assert out["plan_blocked"] is False
    assert "tool_results" not in out


def test_cycle_marks_all_blocked_steps_failed_permanent():
    state = {
        "plan": [
            {"id": "s1", "tool": "kb", "depends_on": ["s2"]},
            {"id": "s2", "tool": "capstone", "depends_on": ["s1"]},
        ],
        "tool_results": [],
    }
    out = nodes.executor_node(state)

    assert out["current_step_id"] is None
    assert out["plan_blocked"] is True
    blocked = out["tool_results"]
    assert {r["step_id"] for r in blocked} == {"s1", "s2"}
    for r in blocked:
        assert r["ok"] is False
        assert r["retryable"] is False
        assert r["rejection"] == "dependency_unsatisfied"
        assert "dependency cycle" in r["data"]


def test_missing_dependency_id_blocks_step():
    state = {
        "plan": [{"id": "s1", "tool": "kb", "depends_on": ["ghost"]}],
        "tool_results": [],
    }
    out = nodes.executor_node(state)
    assert out["plan_blocked"] is True
    assert out["tool_results"][0]["rejection"] == "dependency_unsatisfied"
    assert "missing from the plan" in out["tool_results"][0]["data"]


def test_failed_prerequisite_blocks_dependent_and_reports_reason():
    """Retry-exhausted prerequisite → dependent blocked → plan blocked."""
    state = {
        "plan": [
            {"id": "s1", "tool": "tavily", "depends_on": []},
            {"id": "s2", "tool": "kb", "depends_on": ["s1"]},
        ],
        "tool_results": [_fail("s1")] * MAX_STEP_ATTEMPTS,
    }
    out = nodes.executor_node(state)
    assert out["plan_blocked"] is True
    blocked = out["tool_results"]
    assert [r["step_id"] for r in blocked] == ["s2"]
    assert "failed permanently" in blocked[0]["data"]


# ---------------------------------------------------------------------------
# Integration: executor output → reducers → route_tool
# ---------------------------------------------------------------------------

def test_blocked_plan_routes_to_planner():
    state = {
        "plan": [
            {"id": "s1", "tool": "kb", "depends_on": ["s2"]},
            {"id": "s2", "tool": "capstone", "depends_on": ["s1"]},
        ],
        "tool_results": [],
        "iteration": 2,
    }
    merged = _merge(state, nodes.executor_node(state))
    assert route_tool(merged) == "planner"


def test_failed_prerequisite_routes_to_planner():
    state = {
        "plan": [
            {"id": "s1", "tool": "tavily", "depends_on": []},
            {"id": "s2", "tool": "kb", "depends_on": ["s1"]},
        ],
        "tool_results": [_fail("s1", retryable=False)],
        "iteration": 3,
    }
    merged = _merge(state, nodes.executor_node(state))
    assert route_tool(merged) == "planner"


def test_blocked_plan_at_iteration_cap_routes_to_synthesizer():
    """Blocked-plan replanning is bounded by MAX_ITERS: past the cap the
    run flows to the analyst/critic end path instead of replanning."""
    state = {
        "plan": [{"id": "s1", "tool": "kb", "depends_on": ["ghost"]}],
        "tool_results": [],
        "iteration": MAX_ITERS,
    }
    merged = _merge(state, nodes.executor_node(state))
    assert route_tool(merged) == "synthesizer"


def test_completed_plan_routes_to_synthesizer():
    state = {
        "plan": [{"id": "s1", "tool": "kb", "depends_on": []}],
        "tool_results": [{"step_id": "s1", "tool": "kb", "ok": True,
                          "data": "x"}],
        "iteration": 1,
    }
    merged = _merge(state, nodes.executor_node(state))
    assert route_tool(merged) == "synthesizer"


def test_plan_cap_preserves_dependency_on_dropped_step(monkeypatch):
    """Capping must not make a malformed forward dependency runnable.

    Preserve the reference to dropped s13; the executor then reports the
    missing dependency and deterministically replans.
    """
    raw_plan = [
        {
            "id": f"s{i}",
            "goal": f"step {i}",
            "tool": "kb",
            "args": {"mode": "stats"},
            "depends_on": ["s13"] if i == 1 else [],
            "hypothesis": None,
        }
        for i in range(1, MAX_PLAN_STEPS + 2)
    ]
    monkeypatch.setattr(
        nodes, "_safe_invoke", lambda *a, **kw: {"plan": raw_plan},
    )

    planned = nodes.planner_node({
        "game": "test", "question": "q", "iteration": 0,
        "replan_count": 0,
    })
    assert len(planned["plan"]) == MAX_PLAN_STEPS
    assert planned["plan"][0]["depends_on"] == ["i1_s13"]

    blocked_state = {
        "plan": [planned["plan"][0]],
        "tool_results": [],
        "iteration": planned["iteration"],
    }
    merged = _merge(blocked_state, nodes.executor_node(blocked_state))
    assert merged["plan_blocked"] is True
    assert "missing from the plan" in merged["tool_results"][0]["data"]
    assert route_tool(merged) == "planner"


def test_planner_budget_exhaustion_does_not_emit_fallback_work(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_safe_invoke",
        lambda *a, **kw: {
            "_budget_exhausted": True,
            "_error": "cost reservation exceeds remaining allowance",
        },
    )

    planned = nodes.planner_node({
        "game": "test", "question": "Where is the routine?", "iteration": 2,
        "replan_count": 1,
    })

    assert planned["plan"] == []
    assert planned["iteration"] == 3
    assert "0 steps" in planned["messages"][0].content
