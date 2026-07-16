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

from graph.plan_utils import MAX_ITERS, pending_steps
from graph.state import C64State
from memory import get_store

# Token budget over which the curator should compact tool_result events.
KB_TOKEN_THRESHOLD = 80_000

# Hard caps for the outer critic loop. MAX_ITERS lives in
# graph.plan_utils (RECURSION_LIMIT is derived from it) and is
# re-exported here for existing importers.
BUDGET_CAP = 5.0  # USD or tokens (depending on what `budget_used` tracks)


def _pending_steps(state: C64State) -> list[dict]:
    """Steps still needing a (re)run.

    Delegates to `plan_utils.pending_steps`: a step only counts as done
    when it succeeded or its retries are exhausted — previously any
    attempted step (even `ok=False` from a missing API key) was
    permanently burned (tracker 0.3).
    """
    return pending_steps(state)


def _kb_size_tokens(state: C64State) -> int:
    """Use the real KB size when a handle is available."""
    handle = state.get("kb_handle")
    if handle:
        try:
            return get_store(handle).kb_size_tokens()
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


def post_synth_router(
    state: C64State,
) -> Literal["curate", "executor", "analyst"]:
    """Merged curator-gate + more-steps decision (see graph doc §7)."""
    if _kb_size_tokens(state) > KB_TOKEN_THRESHOLD:
        return "curate"
    if _pending_steps(state):
        return "executor"
    return "analyst"


def post_curator_router(state: C64State) -> Literal["executor", "analyst"]:
    return "executor" if _pending_steps(state) else "analyst"


def verdict_router(
    state: C64State,
) -> Literal["accept", "revise", "replan", "budget_exceeded"]:
    if state.get("budget_used", 0.0) >= BUDGET_CAP:
        return "budget_exceeded"
    if state.get("iteration", 0) >= MAX_ITERS:
        return "budget_exceeded"
    if _dead_end_detected(state):
        return "budget_exceeded"

    decision = (state.get("verdict") or {}).get("decision", "accept")
    if decision in {"accept", "revise", "replan"}:
        return decision  # type: ignore[return-value]
    return "accept"
