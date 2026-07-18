"""Conditional-edge router functions.

These are pure functions over `C64State` — no LLM calls, very limited I/O
(read-only KB stat lookup is allowed because routers run before/between
node invocations and need real KB-size to decide when to compact).

Per `CLAUDE_graph.md` §7's footnote, the curator + more-steps gates are
merged into a single post-synthesizer router so the graph stays a clean
DAG-with-loops.
"""

from __future__ import annotations

from typing import Literal

import os
from langgraph.types import Send

from graph.plan_utils import MAX_ITERS, pending_steps
from graph.state import C64State
from memory import get_store

# Token volume of UNCOMPACTED tool_result events over which the curator
# runs (tracker 2.1). The old gate keyed on total kb.json size, which is
# append-only and never shrinks — once a game's KB crossed the threshold,
# every synthesizer step in every future session paid a curator LLM call
# forever. Compaction advances a high-water mark, so this gate closes.
UNCOMPACTED_TOKEN_THRESHOLD = 40_000

# Hard caps for the outer critic loop. MAX_ITERS lives in
# graph.plan_utils (RECURSION_LIMIT is derived from it) and is
# re-exported here for existing importers.
#
# Two budget limits, deliberately in distinct units/fields (review
# finding 3): `budget_used` is estimated USD and only accrues when
# config/llm.json has a "pricing" section; `tokens_used` always accrues,
# so unpriced/unknown models still hit a runtime cap.
DEFAULT_BUDGET_CAP = 5.0  # USD

# Fallback token cap: ~MAX_ITERS iterations of a busy plan at ~20k
# tokens per LLM call stay well under this; a runaway loop does not.
DEFAULT_TOKEN_BUDGET = 3_000_000


def budget_cap() -> float:
    """Resolve the estimated-USD cap: env → config default → constant."""
    env = os.getenv("C64RE_USD_BUDGET", "").strip()
    if env:
        try:
            value = float(env)
            if value > 0:
                return value
        except ValueError:
            pass
    try:
        from graph.llm import load_config
        value = (load_config().get("defaults") or {}).get("usd_budget")
        if value is not None and float(value) > 0:
            return float(value)
    except (TypeError, ValueError, Exception):  # noqa: BLE001
        pass
    return DEFAULT_BUDGET_CAP


def token_budget() -> int:
    """Resolve the total-token cap: env → config default → constant.

    Env: ``C64RE_TOKEN_BUDGET``; config: ``defaults.token_budget`` in
    `config/llm.json`.
    """
    env = os.getenv("C64RE_TOKEN_BUDGET", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    try:
        from graph.llm import load_config
        v = (load_config().get("defaults") or {}).get("token_budget")
        if v:
            return int(v)
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_TOKEN_BUDGET


def _pending_steps(state: C64State) -> list[dict]:
    """Steps still needing a (re)run.

    Delegates to `plan_utils.pending_steps`: a step only counts as done
    when it succeeded or its retries are exhausted — previously any
    attempted step (even `ok=False` from a missing API key) was
    permanently burned (tracker 0.3).
    """
    return pending_steps(state)


def _uncompacted_tokens(state: C64State) -> int:
    """Token volume of tool_result events past the compaction high-water."""
    handle = state.get("kb_handle")
    if handle:
        try:
            return get_store(handle).uncompacted_tool_result_tokens()
        except Exception:  # noqa: BLE001
            pass
    return 1_000 * len(state.get("tool_results", []))


def _dead_end_detected(state: C64State) -> bool:
    """Two consecutive `replan` verdicts AND no new substantive KB facts.

    The growth signal (`kb_grew_since_last_verdict`) is computed inside
    `critic_node`, which compares the substantive-event count against the
    snapshot taken at the *previous* verdict before overwriting it. The
    old implementation compared against a snapshot written in the same
    state update the router then read — always equal, so every second
    consecutive replan terminated the run regardless of progress
    (tracker 0.2).
    """
    history = state.get("history", []) or []
    if len(history) < 2:
        return False
    last, prev = history[-1], history[-2]
    if last.get("decision") != "replan" or prev.get("decision") != "replan":
        return False
    # Default True: when the flag is missing (old histories, no KB
    # handle) we must not terminate a possibly-productive replan.
    return not last.get("kb_grew_since_last_verdict", True)


def route_tool(
    state: C64State,
) -> Literal[
    "vice", "capstone", "tavily", "kb", "code_kb", "synthesizer", "planner",
]:
    step_id = state.get("current_step_id")
    if not step_id:
        # Nothing dispatchable. Two cases (tracker 0.6):
        # * blocked plan (executor set `plan_blocked` after recording a
        #   structured failure per blocked step) → deterministically
        #   replan, bounded by MAX_ITERS since the planner increments
        #   `iteration` on every pass;
        # * plan complete (or iteration budget spent) → synthesizer,
        #   which persists any recorded failures and lets post_synth
        #   route on to the analyst/critic end path.
        if state.get("plan_blocked") and state.get("iteration", 0) < MAX_ITERS:
            return "planner"
        return "synthesizer"
    for s in state.get("plan", []):
        if s.get("id") == step_id:
            tool = str(s.get("tool", "kb"))
            if tool not in {"vice", "capstone", "tavily", "kb", "code_kb"}:
                return "kb"
            return tool  # type: ignore[return-value]
    return "kb"


def dispatch_tools(
    state: C64State,
) -> list[Send] | Literal[
    "vice", "capstone", "tavily", "kb", "code_kb", "synthesizer", "planner",
]:
    """Route one serial step or fan out a selected read-only batch.

    The executor is the policy owner: it populates ``current_step_ids`` only
    for fresh, concrete, dependency-ready, read-only work. Each Send branch
    receives the full state with one scalar step id, so existing tool nodes do
    not need a parallel-only contract. Their reducer-backed deltas merge before
    the shared synthesizer runs once.
    """
    step_ids = [str(step_id) for step_id in state.get("current_step_ids") or []]
    if not step_ids:
        return route_tool(state)

    plan_by_id = {
        str(step.get("id")): step for step in (state.get("plan") or [])
    }
    sends: list[Send] = []
    for step_id in step_ids:
        step = plan_by_id.get(step_id)
        if step is None:
            continue
        tool = str(step.get("tool") or "kb")
        if tool not in {"vice", "capstone", "tavily", "kb", "code_kb"}:
            tool = "kb"
        branch_state = dict(state)
        branch_state["current_step_id"] = step_id
        branch_state["current_step_ids"] = []
        sends.append(Send(tool, branch_state))

    # Defensive fallback for a stale/malformed batch state. Normal executor
    # output always resolves at least two ids.
    return sends or route_tool({**state, "current_step_ids": []})


def post_synth_router(
    state: C64State,
) -> Literal["curate", "executor", "analyst"]:
    """Merged curator-gate + more-steps decision (see graph doc §7)."""
    if _uncompacted_tokens(state) > UNCOMPACTED_TOKEN_THRESHOLD:
        return "curate"
    if _pending_steps(state):
        return "executor"
    return "analyst"


def post_curator_router(state: C64State) -> Literal["executor", "analyst"]:
    return "executor" if _pending_steps(state) else "analyst"


def verdict_router(
    state: C64State,
) -> Literal["accept", "revise", "replan", "budget_exceeded"]:
    if state.get("budget_reservation_exhausted"):
        return "budget_exceeded"
    if state.get("budget_used", 0.0) >= budget_cap():
        return "budget_exceeded"
    if int(state.get("tokens_used", 0) or 0) >= token_budget():
        return "budget_exceeded"
    if state.get("iteration", 0) >= MAX_ITERS:
        return "budget_exceeded"
    if _dead_end_detected(state):
        return "budget_exceeded"

    decision = (state.get("verdict") or {}).get("decision", "accept")
    if decision in {"accept", "revise", "replan"}:
        return decision  # type: ignore[return-value]
    return "accept"
