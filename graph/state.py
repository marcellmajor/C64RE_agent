"""Shared LangGraph state for the C64-RE Agent.

Mirrors `CLAUDE_graph.md` §2. Reducers are declared via `Annotated` so that
LangGraph merges parallel/looped updates without clobbering history.
"""

from __future__ import annotations

import hashlib
import json
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


# Tool outputs can be tens of kilobytes each. They are durably persisted in
# the evidence KB by the synthesizer, so retaining an unlimited second copy in
# every LangGraph checkpoint is both wasteful and eventually dangerous. Keep
# enough BULKY state rows for the largest possible current plan (12 steps × two
# attempts) plus a generous recent-history margin. Older rows remain as a
# compact, unbounded status ledger so success/failure identity cannot roll out
# from under dependency and retry decisions; persisted evidence stays
# queryable from the KB by event id.
TOOL_RESULTS_STATE_LIMIT = 96
_TOOL_RESULTS_OP = "__c64re_tool_results_op__"
_RESULT_ID_KEY = "_state_result_id"
_EVENT_ID_KEY = "_event_id"
_STATUS_ONLY_KEY = "_status_only"
_STATUS_LEDGER_KEYS = frozenset({
    _RESULT_ID_KEY,
    _EVENT_ID_KEY,
    _STATUS_ONLY_KEY,
    "step_id",
    "tool",
    "ok",
    "retryable",
    "rejection",
    "mode",
    "method",
})


def _result_id_base(result: dict[str, Any]) -> str:
    stable = {
        key: value for key, value in result.items()
        if key not in {_RESULT_ID_KEY, _EVENT_ID_KEY}
    }
    try:
        raw = json.dumps(
            stable, sort_keys=True, separators=(",", ":"), default=str,
        )
    except Exception:  # noqa: BLE001
        raw = repr(stable)
    return "tr_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def identify_tool_results(
    results: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return copies with deterministic, occurrence-safe state identities.

    Tool retries can legitimately produce byte-for-byte identical results;
    suffixing later occurrences keeps both attempts in the status ledger.
    Existing ids survive checkpoint replay unchanged.
    """
    identified: list[dict[str, Any]] = []
    used: set[str] = set()
    for original in results or []:
        result = dict(original)
        candidate = str(result.get(_RESULT_ID_KEY) or _result_id_base(result))
        result_id = candidate
        suffix = 2
        while result_id in used:
            result_id = f"{candidate}_{suffix}"
            suffix += 1
        result[_RESULT_ID_KEY] = result_id
        used.add(result_id)
        identified.append(result)
    return identified


def compact_tool_results_update(
    event_ids: dict[str, str],
) -> dict[str, Any]:
    """Build the reducer command used after results reach the evidence KB."""
    return {_TOOL_RESULTS_OP: "compact", "event_ids": dict(event_ids)}


def _compact_result(result: dict[str, Any], event_id: str) -> dict[str, Any]:
    """Keep the execution ledger while dropping persisted bulky payloads."""
    compact: dict[str, Any] = {
        _RESULT_ID_KEY: result[_RESULT_ID_KEY],
        _EVENT_ID_KEY: event_id,
        "step_id": result.get("step_id"),
        "tool": result.get("tool"),
        "ok": bool(result.get("ok")),
    }
    for key in ("retryable", "rejection", "mode", "method"):
        if key in result:
            compact[key] = result[key]
    if not result.get("ok"):
        # The executor needs the last error to repair a retry. A bounded
        # preview is sufficient; the complete result remains in the KB.
        data = str(result.get("data", ""))
        compact["data"] = data[:1_000] + ("…" if len(data) > 1_000 else "")
    return compact


def _status_only_result(result: dict[str, Any]) -> dict[str, Any]:
    """Drop payload fields but retain the durable execution identity."""
    compact: dict[str, Any] = {
        _RESULT_ID_KEY: result[_RESULT_ID_KEY],
        _EVENT_ID_KEY: result.get(_EVENT_ID_KEY),
        _STATUS_ONLY_KEY: True,
        "step_id": result.get("step_id"),
        "tool": result.get("tool"),
        "ok": bool(result.get("ok")),
        "retryable": result.get("retryable"),
        "rejection": result.get("rejection"),
    }
    for key in ("mode", "method"):
        if key in result:
            compact[key] = result[key]
    return compact


def _has_bulky_payload(result: dict[str, Any]) -> bool:
    return any(key not in _STATUS_LEDGER_KEYS for key in result)


def reduce_tool_results(
    current: list[dict[str, Any]] | None,
    update: list[dict[str, Any]] | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Append results while hard-bounding payloads, never status identity."""
    rows = identify_tool_results(current)
    if isinstance(update, dict) and update.get(_TOOL_RESULTS_OP) == "compact":
        event_ids = {
            str(key): str(value)
            for key, value in (update.get("event_ids") or {}).items()
            if value
        }
        rows = [
            _compact_result(row, event_ids[row[_RESULT_ID_KEY]])
            if row[_RESULT_ID_KEY] in event_ids
            else row
            for row in rows
        ]
    elif update:
        incoming = update if isinstance(update, list) else [update]
        rows = identify_tool_results([*rows, *incoming])
    bulky = [index for index, row in enumerate(rows) if _has_bulky_payload(row)]
    evict_payloads = set(bulky[:-TOOL_RESULTS_STATE_LIMIT])
    return [
        _status_only_result(row) if index in evict_payloads else row
        for index, row in enumerate(rows)
    ]


def merge_tool_call_stats(
    current: dict[str, dict[str, int]] | None,
    update: dict[str, dict[str, int]] | None,
) -> dict[str, dict[str, int]]:
    """Add per-tool call/failure deltas without retaining their payloads."""
    merged = {
        str(tool): {
            "calls": int(values.get("calls", 0)),
            "failures": int(values.get("failures", 0)),
        }
        for tool, values in (current or {}).items()
    }
    for tool, values in (update or {}).items():
        row = merged.setdefault(str(tool), {"calls": 0, "failures": 0})
        row["calls"] += int(values.get("calls", 0))
        row["failures"] += int(values.get("failures", 0))
    return merged


def tool_call_total(state: dict[str, Any]) -> int:
    """Return the durable total, with legacy-state fallback."""
    stats = state.get("tool_call_stats") or {}
    if stats:
        return sum(int(row.get("calls", 0)) for row in stats.values())
    return len(state.get("tool_results") or [])


class C64State(TypedDict, total=False):
    # --- inputs (set once at load_inputs) ---
    game: str
    question: str
    dump_path: str
    partial_asm_path: str | None
    text_dir: str | None
    # Directory of partial-asm files fed to the layered code-comprehension
    # sub-agent (`code_kb`). Independent from `partial_asm_path` so a run
    # can mix one curated partial asm with a directory of community
    # disassemblies, and from `text_dir` because the asm pipeline owns
    # its own parser/store.
    asm_dir: str | None
    # Explicit per-game asm-file selection. When set, this overrides the
    # auto-scoping done by `code_kb.scoping.select_asm_files`. Each entry
    # may be absolute or relative to `asm_dir`. Used when `asm_dir` holds
    # files belonging to several games and the game's slug-token heuristic
    # would otherwise pick up the wrong subset.
    asm_files: list[str] | None
    # Stable identity/timing for one runner invocation. `write_report` uses
    # these to persist exactly one queryable run_summary event (tracker 5.4).
    run_id: str
    run_started_at: str
    run_completed_at: str | None
    # Human approval boundary for emulator-mutating VICE steps (Phase 6.1).
    require_vice_approval: bool
    approved_mutation_steps: list[str]
    approve_all_vice_mutations: bool

    # --- working memory ---
    kb_handle: str
    # Handle to the *separate* CodeKnowledgeStore that owns Layer-0
    # ground truth and Layer-1+ LLM annotations of the disassembly. Set
    # by `load_inputs` when an asm dir / partial asm is supplied.
    code_kb_handle: str | None
    plan: list[dict[str, Any]]
    current_step_id: str | None
    # Two or more fresh read-only steps selected for native LangGraph Send
    # fan-out. Each branch receives one `current_step_id`; VICE/mutating and
    # enrichment/retry paths continue to use the scalar serial field only.
    current_step_ids: list[str]
    tool_results: Annotated[list[dict[str, Any]], reduce_tool_results]
    # Exact counters survive raw-result eviction and keep reports/evaluations
    # truthful even after the bounded state window rolls over.
    tool_call_stats: Annotated[
        dict[str, dict[str, int]], merge_tool_call_stats,
    ]

    # --- analysis ---
    candidate_answer: dict[str, Any] | None
    verdict: dict[str, Any] | None
    history: Annotated[list[dict[str, Any]], add]

    # Question-relevant KB digest, refreshed by the synthesizer / curator
    # so the analyst and critic see the same evidence sheet.
    kb_digest: str
    # Deterministic excerpts from the explicitly selected partial-assembly
    # documents. Kept separately so parent-KB digest refreshes cannot erase
    # the strongest static evidence before the analyst/critic run.
    code_kb_digest: str

    # Snapshot of the *distinct substantive fact* count (bookkeeping
    # kinds excluded; failed tool attempts excluded; identity is stable
    # across replanned step IDs) at the last critic verdict.
    # `critic_node` compares the fresh count against this snapshot BEFORE
    # overwriting it and ships the result on the verdict as
    # `kb_grew_since_last_verdict`, which the dead-end router reads.
    last_kb_event_count: int

    # Set by the executor when every pending step has an unsatisfiable
    # dependency (failed prerequisite, missing id, or cycle); routes
    # deterministically to a replan. Cleared by the planner.
    plan_blocked: bool

    # Stamped post-hoc by the runners when a run ends without a critic
    # accept (e.g. "recursion_exhausted") — see
    # `plan_utils.normalize_truncated_state`.
    termination_reason: str | None

    # Stable identities of the bounded result window already considered for
    # LLM extraction. Unlike the legacy numeric cursor, ids remain correct
    # when the reducer evicts old rows from the front (tracker 4.8).
    synth_processed_result_ids: list[str]
    # Legacy migration/debug cursor. New code uses ids above; retaining this
    # scalar lets old durable checkpoints migrate their already-seen prefix.
    synth_processed_count: int

    # --- control ---
    iteration: int
    replan_count: int
    # Consecutive `revise` verdicts; reset on accept/replan. The critic
    # forces `accept` once this reaches MAX_CONSECUTIVE_REVISES so the
    # analyst↔critic ping-pong terminates (tracker 0.4).
    revise_count: int
    # Accumulated estimated USD across all LLM calls (tracker 1.4).
    # Nodes return per-drain deltas; stays 0.0 when config/llm.json has
    # no "pricing" section (tokens are still tracked in `llm_usage`).
    budget_used: Annotated[float, add]
    # Set when no role in an invocation chain can be called safely within
    # the remaining worst-case reservation. The critic router terminates the
    # run even though actual spend is still below the nominal USD cap.
    budget_reservation_exhausted: bool
    # Accumulated total tokens (input+output) across all LLM calls —
    # the always-available budget fallback (`routers.token_budget()`)
    # so unpriced models still hit a runtime cap. Deliberately a
    # separate field from `budget_used`: no unit mixing.
    tokens_used: Annotated[int, add]
    # One entry per LLM call: role, model, tokens, cost, ok (usable
    # contract response) / transport_ok / error. Aggregated per-role in
    # the report (tracker 1.4).
    llm_usage: Annotated[list[dict[str, Any]], add]

    # --- transcript (LangChain message log; viewable in Studio/LangSmith) ---
    messages: Annotated[list, add_messages]
