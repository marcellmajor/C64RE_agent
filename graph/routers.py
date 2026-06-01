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

from graph.state import C64State
from memory import get_store

# Token budget over which the curator should compact tool_result events.
KB_TOKEN_THRESHOLD = 80_000

# Hard caps for the outer critic loop.
MAX_ITERS = 12
BUDGET_CAP = 5.0  # USD or tokens (depending on what `budget_used` tracks)


def _pending_steps(state: C64State) -> list[dict]:
    done = {r.get("step_id") for r in state.get("tool_results", [])}
    return [s for s in state.get("plan", []) if s.get("id") not in done]


def _kb_size_tokens(state: C64State) -> int:
    """Use the real KB size when a handle is available."""
    handle = state.get("kb_handle")
    if handle:
        try:
            return get_store(handle).kb_size_tokens()
        except Exception:  # noqa: BLE001
            pass
    return 1_000 * len(state.get("tool_results", []))


def _kb_event_count(state: C64State) -> int:
    handle = state.get("kb_handle")
    if not handle:
        return 0
    try:
        return int(get_store(handle).stats().get("events_total", 0))
    except Exception:  # noqa: BLE001
        return 0


def _dead_end_detected(state: C64State) -> bool:
    """Two consecutive `replan` verdicts AND no new KB facts.

    The original implementation only checked for two replans in a row,
    which would terminate even if the second replan produced new
    Capstone evidence. We also require the KB to have stopped growing.
    """
    history = state.get("history", []) or []
    last_two = [h for h in history[-2:] if h.get("decision") == "replan"]
    if len(last_two) < 2:
        return False
    return _kb_event_count(state) <= int(state.get("last_kb_event_count", 0))


def route_tool(
    state: C64State,
) -> Literal["vice", "capstone", "tavily", "kb", "code_kb"]:
    step_id = state.get("current_step_id")
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
