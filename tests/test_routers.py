"""Unit tests for graph.routers (tracker items 0.2 / 0.3 / 0.6)."""

from graph.routers import (
    MAX_ITERS,
    _dead_end_detected,
    route_tool,
    verdict_router,
)


def _v(decision, grew=None):
    out = {"decision": decision}
    if grew is not None:
        out["kb_grew_since_last_verdict"] = grew
    return out


# ---------------------------------------------------------------------------
# dead-end detector (0.2)
# ---------------------------------------------------------------------------

def test_no_dead_end_with_short_history():
    assert not _dead_end_detected({"history": []})
    assert not _dead_end_detected({"history": [_v("replan", grew=False)]})


def test_two_replans_without_growth_is_dead_end():
    state = {"history": [_v("replan", grew=True), _v("replan", grew=False)]}
    assert _dead_end_detected(state)


def test_two_replans_with_growth_is_not_dead_end():
    """The core 0.2 fix: a productive replan must survive."""
    state = {"history": [_v("replan", grew=True), _v("replan", grew=True)]}
    assert not _dead_end_detected(state)


def test_mixed_verdicts_are_not_dead_end():
    state = {"history": [_v("revise"), _v("replan", grew=False)]}
    assert not _dead_end_detected(state)


def test_missing_growth_flag_defaults_to_alive():
    """Old histories without the flag must not terminate the run."""
    state = {"history": [_v("replan"), _v("replan")]}
    assert not _dead_end_detected(state)


# ---------------------------------------------------------------------------
# route_tool (0.6)
# ---------------------------------------------------------------------------

def test_route_tool_none_step_goes_to_synthesizer():
    assert route_tool({"current_step_id": None, "plan": []}) == "synthesizer"
    assert route_tool({"plan": []}) == "synthesizer"


def test_route_tool_blocked_plan_goes_to_planner():
    state = {"current_step_id": None, "plan_blocked": True, "iteration": 1}
    assert route_tool(state) == "planner"


def test_route_tool_blocked_plan_at_iteration_cap_goes_to_synthesizer():
    state = {
        "current_step_id": None,
        "plan_blocked": True,
        "iteration": MAX_ITERS,
    }
    assert route_tool(state) == "synthesizer"


def test_route_tool_normal_dispatch():
    state = {
        "current_step_id": "s1",
        "plan": [{"id": "s1", "tool": "capstone"}],
    }
    assert route_tool(state) == "capstone"


def test_route_tool_unknown_tool_falls_back_to_kb():
    state = {
        "current_step_id": "s1",
        "plan": [{"id": "s1", "tool": "wat"}],
    }
    assert route_tool(state) == "kb"


# ---------------------------------------------------------------------------
# verdict_router caps
# ---------------------------------------------------------------------------

def test_verdict_router_passthrough():
    state = {"verdict": {"decision": "replan"}, "iteration": 1, "history": []}
    assert verdict_router(state) == "replan"


def test_verdict_router_iteration_cap():
    state = {"verdict": {"decision": "replan"}, "iteration": MAX_ITERS,
             "history": []}
    assert verdict_router(state) == "budget_exceeded"


def test_verdict_router_dead_end_terminates():
    state = {
        "verdict": {"decision": "replan"},
        "iteration": 3,
        "history": [_v("replan", grew=False), _v("replan", grew=False)],
    }
    assert verdict_router(state) == "budget_exceeded"


# ---------------------------------------------------------------------------
# Token-budget fallback (review finding 3): protection without pricing
# ---------------------------------------------------------------------------

def test_token_budget_terminates_unpriced_run(monkeypatch):
    from graph.routers import token_budget

    monkeypatch.setenv("C64RE_TOKEN_BUDGET", "1000")
    assert token_budget() == 1000

    over = {
        "verdict": {"decision": "replan"},
        "iteration": 1,
        "history": [],
        "budget_used": 0.0,        # no pricing table → USD never accrues
        "tokens_used": 1_500,
    }
    assert verdict_router(over) == "budget_exceeded"

    under = dict(over, tokens_used=999)
    assert verdict_router(under) == "replan"


def test_token_budget_env_invalid_falls_back(monkeypatch):
    from graph.routers import token_budget

    monkeypatch.setenv("C64RE_TOKEN_BUDGET", "not-a-number")
    fallback = token_budget()
    monkeypatch.delenv("C64RE_TOKEN_BUDGET")
    # Invalid env is ignored — same resolution as no env at all
    # (config `defaults.token_budget` or DEFAULT_TOKEN_BUDGET).
    assert fallback == token_budget()
    assert isinstance(fallback, int) and fallback > 0


def test_usd_budget_env_controls_verdict_cap(monkeypatch):
    from graph.routers import budget_cap

    monkeypatch.setenv("C64RE_USD_BUDGET", "0.25")
    assert budget_cap() == 0.25
    state = {
        "verdict": {"decision": "replan"},
        "iteration": 1,
        "history": [],
        "budget_used": 0.25,
        "tokens_used": 1,
    }
    assert verdict_router(state) == "budget_exceeded"

    monkeypatch.setenv("C64RE_USD_BUDGET", "invalid")
    assert budget_cap() > 0.0


def test_pre_call_reservation_exhaustion_terminates_at_verdict():
    state = {
        "verdict": {"decision": "replan"},
        "iteration": 1,
        "history": [],
        "budget_used": 0.1,
        "tokens_used": 10,
        "budget_reservation_exhausted": True,
    }
    assert verdict_router(state) == "budget_exceeded"
