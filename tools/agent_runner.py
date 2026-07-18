"""Headless wrapper around the LangGraph agent for the Streamlit UI.

`run_question()` is a generator: each yielded item is a tuple
`(kind, payload)` so the UI can stream progress into a chat bubble
without re-implementing the orchestration loop in `main.py`.

The wrapper persists the *underlying knowledge bases* across turns by
keying their on-disk locations off the game slug (this is already how
`load_inputs` resolves them). Each turn uses a fresh thread_id so the
planner / executor cleanly start a new investigation while still
benefitting from everything the previous turn wrote into the KB.
"""

from __future__ import annotations

import logging
import re
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from c64re_agent.paths import sessions_dir
from graph.build import build_graph
from graph.plan_utils import (
    RECURSION_LIMIT,
    normalize_truncated_state,
    resolve_session_slug,
)

log = logging.getLogger(__name__)


_GRAPH_CACHE: dict[str, Any] = {}


def _compiled_graph() -> Any:
    """Return a process-wide compiled graph (Streamlit reruns the script
    on every interaction; we don't want to re-compile each time)."""
    g = _GRAPH_CACHE.get("g")
    if g is None:
        g = build_graph().compile(checkpointer=MemorySaver())
        _GRAPH_CACHE["g"] = g
    return g


def _compiled_hitl_graph() -> Any:
    """Graph paused after every planner pass for human plan review."""
    graph = _GRAPH_CACHE.get("hitl")
    if graph is None:
        graph = build_graph().compile(
            checkpointer=MemorySaver(), interrupt_after=["planner"],
        )
        _GRAPH_CACHE["hitl"] = graph
    return graph


@dataclass
class TurnResult:
    """Everything one chat turn produced — handed back to the UI."""
    question: str
    answer: str
    confidence: float | None
    verdict: str
    critique: str | None
    evidence: list[str]
    open_questions: list[str]
    addresses: list[int]            # parsed out of answer + evidence
    resolved_evidence: list[str] = field(default_factory=list)
    raw_state: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    thread_id: str = ""
    run_id: str = ""


@dataclass
class PlanReview:
    """Checkpointed run waiting for plan edits/approval."""

    question: str
    thread_id: str
    plan: list[dict[str, Any]]
    mutating_step_ids: list[str]
    seen_messages: int
    started_monotonic: float


# --------------------------------------------------------------------------- #
# Address extraction
# --------------------------------------------------------------------------- #

_ADDR_RE = re.compile(r"\$([0-9A-Fa-f]{2,4})\b")
_ADDR_RANGE_RE = re.compile(
    r"\$([0-9A-Fa-f]{2,4})\s*[-–]\s*\$([0-9A-Fa-f]{2,4})"
)


def parse_addresses(text: str) -> list[int]:
    """Return a stable, de-duplicated list of $XXXX addresses found in text.

    Order-preserving: first occurrence wins. Ranges like `$0780-$079F`
    contribute the start of the range only — the UI uses each address
    as a query into `code_routines`, which then resolves the enclosing
    routine itself.
    """
    seen: dict[int, None] = {}

    for m in _ADDR_RANGE_RE.finditer(text or ""):
        try:
            seen.setdefault(int(m.group(1), 16) & 0xFFFF, None)
        except ValueError:
            pass

    text_no_ranges = _ADDR_RANGE_RE.sub(" ", text or "")
    for m in _ADDR_RE.finditer(text_no_ranges):
        try:
            v = int(m.group(1), 16) & 0xFFFF
        except ValueError:
            continue
        seen.setdefault(v, None)
    return list(seen.keys())


# --------------------------------------------------------------------------- #
# Streaming runner
# --------------------------------------------------------------------------- #


def _initial_state(
    *, game: str, question: str, dump_path: str | Path,
    asm_dir: str | Path | None = None,
    asm_files: list[str | Path] | None = None,
    partial_asm: str | Path | None = None,
    text_dir: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "run_id": uuid.uuid4().hex,
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "game": game,
        "question": question,
        "dump_path": str(dump_path),
        "partial_asm_path": str(partial_asm) if partial_asm else None,
        "text_dir": str(text_dir) if text_dir else None,
        "asm_dir": str(asm_dir) if asm_dir else None,
        "asm_files": [str(path) for path in asm_files] if asm_files else None,
        "plan": [],
        "current_step_ids": [],
        "tool_results": [],
        "tool_call_stats": {},
        "history": [],
        "messages": [],
        "require_vice_approval": True,
        "approved_mutation_steps": [],
        "approve_all_vice_mutations": False,
    }


def _turn_result_from_state(
    final: dict[str, Any], *, question: str, thread_id: str,
    started: float,
) -> TurnResult:
    candidate = final.get("candidate_answer") or {}
    verdict_obj = final.get("verdict") or {}
    answer = str(candidate.get("answer") or "(no answer produced)")
    evidence = [str(item) for item in (candidate.get("evidence") or [])]
    addresses = parse_addresses(answer + "\n" + "\n".join(evidence))
    try:
        from memory import get_store
        from memory.evidence import resolve_evidence

        evidence_store = (
            get_store(final["kb_handle"]) if final.get("kb_handle") else None
        )
        resolved_evidence = resolve_evidence(evidence, evidence_store)
    except Exception:  # noqa: BLE001
        resolved_evidence = evidence
    return TurnResult(
        question=question,
        answer=answer,
        confidence=(
            float(candidate["confidence"])
            if isinstance(candidate.get("confidence"), (int, float))
            else None
        ),
        verdict=str(verdict_obj.get("decision") or "n/a"),
        critique=verdict_obj.get("critique"),
        evidence=evidence,
        open_questions=[
            str(item) for item in (candidate.get("open_questions") or [])
        ],
        addresses=addresses,
        resolved_evidence=resolved_evidence,
        raw_state=final,
        elapsed_s=time.monotonic() - started,
        thread_id=thread_id,
        run_id=str(final.get("run_id") or ""),
    )


def run_question(
    *,
    game: str,
    question: str,
    dump_path: str | Path,
    asm_dir: str | Path | None = None,
    asm_files: list[str | Path] | None = None,
    partial_asm: str | Path | None = None,
    text_dir: str | Path | None = None,
    thread_id: str | None = None,
) -> Iterator[tuple[str, Any]]:
    """Stream a question through the graph.

    Yields tuples:

        ("status", "load_inputs running...")
        ("step", {"node": "planner", "messages": [...]})
        ("done",  TurnResult)

    The UI displays "step" entries as transient progress, then renders
    the final `TurnResult` into the chat history + code lens.
    """
    started = time.monotonic()
    graph = _compiled_graph()

    initial_state = _initial_state(
        game=game,
        question=question,
        dump_path=dump_path,
        asm_dir=asm_dir,
        asm_files=asm_files,
        partial_asm=partial_asm,
        text_dir=text_dir,
    )

    tid = thread_id or f"webui-{uuid.uuid4().hex[:8]}"
    # LangGraph defaults to 25 super-steps — one 7-step plan iteration.
    # RECURSION_LIMIT budgets the full MAX_ITERS loop (tracker 0.1).
    config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": tid},
    }

    yield ("status", f"thread_id={tid}")
    seen_msgs = 0
    last_state: dict[str, Any] = {}
    truncated: dict[str, Any] | None = None
    try:
        for event in graph.stream(initial_state, config, stream_mode="values"):
            last_state = event
            msgs = event.get("messages") or []
            new_msgs = msgs[seen_msgs:]
            seen_msgs = len(msgs)
            if new_msgs:
                yield ("step", {
                    "messages": [getattr(m, "content", str(m)) for m in new_msgs],
                    "iteration": event.get("iteration", 0),
                })
    except GraphRecursionError:
        # Degrade to an honest best-effort answer instead of an error
        # bubble (tracker 0.1): normalize the last completed state —
        # explicit termination reason, capped confidence, truncation
        # note in open_questions — and use THAT for both the report
        # (the write_report node never ran) and the TurnResult below.
        yield ("status", (
            f"recursion limit ({RECURSION_LIMIT} super-steps) exhausted — "
            "returning best-effort truncated state"
        ))
        try:
            raw = graph.get_state(config).values or last_state
        except Exception:  # noqa: BLE001
            raw = last_state
        truncated = normalize_truncated_state(
            raw,
            reason="recursion_exhausted",
            detail=(
                f"the {RECURSION_LIMIT}-super-step recursion limit was "
                "reached before the critic accepted an answer; findings "
                "reflect the last completed iteration only."
            ),
        )
        try:
            from graph.nodes import write_report
            write_report(truncated)
        except Exception:  # noqa: BLE001
            log.warning("best-effort write_report failed", exc_info=True)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log.error("run_question failed:\n%s", tb)
        yield ("error", f"{type(e).__name__}: {e}\n\n```\n{tb}\n```")
        return

    if truncated is not None:
        final = truncated
    else:
        final = graph.get_state(config).values or last_state
    yield ("done", _turn_result_from_state(
        final, question=question, thread_id=tid, started=started,
    ))


def mutating_step_ids(plan: list[dict[str, Any]]) -> list[str]:
    from graph.nodes import vice_step_requires_approval

    return [
        str(step.get("id") or "?") for step in plan
        if str(step.get("tool") or "").lower() == "vice"
        and vice_step_requires_approval(step)
    ]


def validate_review_plan(plan: Any) -> list[dict[str, Any]]:
    """Validate a user-edited planner JSON array before checkpoint resume."""
    if not isinstance(plan, list) or not plan:
        raise ValueError("reviewed plan must be a non-empty JSON array")
    allowed_tools = {"vice", "capstone", "tavily", "kb", "code_kb"}
    out: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw in enumerate(plan, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"plan row {index} must be an object")
        step = dict(raw)
        step_id = str(step.get("id") or "").strip()
        tool = str(step.get("tool") or "").strip().lower()
        if not step_id or not tool:
            raise ValueError(f"plan row {index} requires id and tool")
        if tool not in allowed_tools:
            raise ValueError(
                f"plan row {step_id} has unsupported tool {tool!r}",
            )
        if step_id in ids:
            raise ValueError(f"duplicate reviewed plan id: {step_id}")
        ids.add(step_id)
        if not isinstance(step.get("args") or {}, dict):
            raise ValueError(f"plan row {step_id} args must be an object")
        dependencies = step.get("depends_on") or []
        if not isinstance(dependencies, list):
            raise ValueError(f"plan row {step_id} depends_on must be an array")
        step["args"] = dict(step.get("args") or {})
        step["depends_on"] = [str(item) for item in dependencies]
        step["tool"] = tool
        out.append(step)
    unknown = sorted({
        dependency for step in out for dependency in step["depends_on"]
        if dependency not in ids
    })
    if unknown:
        raise ValueError("unknown reviewed plan dependencies: " + ", ".join(unknown))

    dependencies_by_id = {
        str(step["id"]): set(step["depends_on"]) for step in out
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visiting:
            raise ValueError("reviewed plan dependencies contain a cycle")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in dependencies_by_id[step_id]:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in dependencies_by_id:
        visit(step_id)
    return out


def _plan_review_from_state(
    state: dict[str, Any], *, thread_id: str, seen_messages: int,
    started: float,
) -> PlanReview:
    plan = validate_review_plan(state.get("plan") or [])
    return PlanReview(
        question=str(state.get("question") or ""),
        thread_id=thread_id,
        plan=plan,
        mutating_step_ids=mutating_step_ids(plan),
        seen_messages=seen_messages,
        started_monotonic=started,
    )


def begin_question(
    *, game: str, question: str, dump_path: str | Path,
    asm_dir: str | Path | None = None,
    asm_files: list[str | Path] | None = None,
    partial_asm: str | Path | None = None,
    text_dir: str | Path | None = None,
    thread_id: str | None = None,
) -> Iterator[tuple[str, Any]]:
    """Run through planning, then pause at a durable human-review point."""
    graph = _compiled_hitl_graph()
    started = time.monotonic()
    tid = thread_id or f"webui-hitl-{uuid.uuid4().hex[:8]}"
    config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": tid},
    }
    yield ("status", f"thread_id={tid} · planning")
    seen_messages = 0
    try:
        for event in graph.stream(
            _initial_state(
                game=game,
                question=question,
                dump_path=dump_path,
                asm_dir=asm_dir,
                asm_files=asm_files,
                partial_asm=partial_asm,
                text_dir=text_dir,
            ),
            config,
            stream_mode="values",
        ):
            messages = event.get("messages") or []
            new_messages = messages[seen_messages:]
            seen_messages = len(messages)
            if new_messages:
                yield ("step", {
                    "messages": [
                        getattr(message, "content", str(message))
                        for message in new_messages
                    ],
                    "iteration": event.get("iteration", 0),
                })
    except Exception as exc:  # noqa: BLE001
        yield ("error", f"{type(exc).__name__}: {exc}")
        return
    state = graph.get_state(config).values or {}
    try:
        review = _plan_review_from_state(
            state,
            thread_id=tid,
            seen_messages=seen_messages,
            started=started,
        )
    except ValueError as exc:
        yield ("error", f"planner produced an unreviewable plan: {exc}")
        return
    yield ("plan_review", review)


def resume_question(
    review: PlanReview, *, plan: Any,
    approved_mutation_steps: list[str] | None = None,
) -> Iterator[tuple[str, Any]]:
    """Apply a reviewed plan and resume until completion or the next replan."""
    graph = _compiled_hitl_graph()
    reviewed_plan = validate_review_plan(plan)
    config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": review.thread_id},
    }
    approved = [str(item) for item in approved_mutation_steps or []]
    graph.update_state(config, {
        "plan": reviewed_plan,
        "approved_mutation_steps": approved,
        "approve_all_vice_mutations": False,
    }, as_node="planner")
    yield ("status", f"thread_id={review.thread_id} · approved plan resumed")
    seen_messages = review.seen_messages
    last_state: dict[str, Any] = {}
    truncated: dict[str, Any] | None = None
    try:
        for event in graph.stream(None, config, stream_mode="values"):
            last_state = event
            messages = event.get("messages") or []
            new_messages = messages[seen_messages:]
            seen_messages = len(messages)
            if new_messages:
                yield ("step", {
                    "messages": [
                        getattr(message, "content", str(message))
                        for message in new_messages
                    ],
                    "iteration": event.get("iteration", 0),
                })
    except GraphRecursionError:
        raw = graph.get_state(config).values or last_state
        truncated = normalize_truncated_state(
            raw,
            reason="recursion_exhausted",
            detail="the reviewed run exhausted its super-step budget",
        )
        try:
            from graph.nodes import write_report

            write_report(truncated)
        except Exception:  # noqa: BLE001
            log.warning("best-effort HITL write_report failed", exc_info=True)
    except Exception as exc:  # noqa: BLE001
        yield ("error", f"{type(exc).__name__}: {exc}")
        return

    snapshot = graph.get_state(config)
    final = truncated or snapshot.values or last_state
    if final.get("run_completed_at") or truncated is not None:
        yield ("done", _turn_result_from_state(
            final,
            question=review.question,
            thread_id=review.thread_id,
            started=review.started_monotonic,
        ))
        return
    try:
        next_review = _plan_review_from_state(
            final,
            thread_id=review.thread_id,
            seen_messages=seen_messages,
            started=review.started_monotonic,
        )
    except ValueError as exc:
        yield ("error", f"replanner produced an unreviewable plan: {exc}")
        return
    yield ("plan_review", next_review)


# --------------------------------------------------------------------------- #
# Code-lens helpers — used by the UI right after a turn completes.
# --------------------------------------------------------------------------- #


def _resolve_code_store(game: str):
    """Locate the persistent CodeKnowledgeStore for a game, if it exists."""
    from code_kb import get_code_store

    sessions = sessions_dir()
    root = sessions / resolve_session_slug(game, sessions) / "code_kb"
    if not root.exists():
        return None
    return get_code_store(root)


def disasm_for_addresses(
    game: str, addresses: list[int], *, max_routines: int = 6,
) -> list[dict[str, Any]]:
    """For each address, find the enclosing routine and pull a snippet.

    Each snippet is plain text formatted like a real disassembly listing
    so the UI can drop it into a `st.code(... , language="asm6502")`
    block. Returns at most `max_routines` distinct routines, even if
    several addresses fall inside the same one.
    """
    store = _resolve_code_store(game)
    if store is None or not addresses:
        return []

    snippets: list[dict[str, Any]] = []
    seen_starts: set[int] = set()

    for addr in addresses:
        rows = store.query(
            "SELECT start_addr, end_addr, name, source_file"
            "  FROM code_routines"
            " WHERE ? BETWEEN start_addr AND end_addr"
            " ORDER BY (end_addr - start_addr) ASC LIMIT 1",
            (int(addr) & 0xFFFF,),
        )
        if not rows:
            continue
        r = rows[0]
        start = int(r["start_addr"])
        if start in seen_starts:
            continue
        seen_starts.add(start)

        insns = store.query(
            "SELECT addr, bytes_hex, mnemonic, operand FROM instructions"
            " WHERE addr BETWEEN ? AND ? ORDER BY addr",
            (start, int(r["end_addr"])),
        )
        listing = _format_listing(insns)

        layer1 = store.query(
            "SELECT text, name_suggestion, idiom_match, hardware_touched_json, confidence"
            "  FROM hypotheses"
            " WHERE start_addr = ? AND layer = 1"
            " ORDER BY confidence DESC LIMIT 1",
            (start,),
        )
        l1 = layer1[0] if layer1 else None
        display_name = r.get("name")
        if (
            l1
            and isinstance(l1.get("name_suggestion"), str)
            and isinstance(l1.get("confidence"), (int, float))
            and float(l1["confidence"]) >= 0.60
            and (
                not display_name
                or str(display_name).lower().startswith("sub_")
            )
        ):
            display_name = l1.get("name_suggestion")
        snippets.append({
            "addr_query": addr,
            "start_addr": start,
            "end_addr": int(r["end_addr"]),
            "name": display_name,
            "source_file": r.get("source_file"),
            "listing": listing,
            "instruction_count": len(insns),
            "layer1": l1,
        })
        if len(snippets) >= max_routines:
            break

    return snippets


def _format_listing(insns: list[dict[str, Any]]) -> str:
    """Render `instructions` rows as a 6502 listing the UI can show."""
    if not insns:
        return "(no instructions stored — try `code_kb mode='disasm' ...`)"
    out: list[str] = []
    for ins in insns:
        addr = int(ins.get("addr") or 0) & 0xFFFF
        bytes_hex = (ins.get("bytes_hex") or "").upper()
        mnem = (ins.get("mnemonic") or "?").lower()
        operand = ins.get("operand") or ""
        operand = operand if isinstance(operand, str) else str(operand)
        out.append(
            f"${addr:04X}  {bytes_hex:<10}  {mnem:<4} {operand}".rstrip()
        )
    return "\n".join(out)


def call_graph_dot(
    game: str, *, start: int, hops: int = 1, max_nodes: int = 60,
) -> dict[str, Any] | None:
    """Build a local DOT call graph centred on a routine start address."""
    from code_kb.call_graph import local_dot

    store = _resolve_code_store(game)
    if store is None:
        return None
    return local_dot(
        store, start=int(start) & 0xFFFF, hops=int(hops),
        max_nodes=int(max_nodes),
    )


def code_kb_summary(game: str) -> dict[str, Any] | None:
    """Quick stats card the UI shows in the sidebar after a session loads."""
    store = _resolve_code_store(game)
    if store is None:
        return None
    return store.stats()


def symbol_map_for_game(
    game: str, *, min_confidence: float = 0.75,
) -> str | None:
    """Render a VICE monitor label map from trustworthy Code-KB names."""
    store = _resolve_code_store(game)
    if store is None:
        return None
    from code_kb import export_vice_symbols

    return export_vice_symbols(store, min_confidence=min_confidence)


def reset_code_kb(game: str) -> bool:
    """Delete `sessions/<slug>/code_kb/` so the next run rebuilds it.

    Used when a previous run mixed files from several games and the
    user wants a clean slate. Returns True iff something was removed.
    """
    import shutil as _sh
    sessions = sessions_dir()
    target = sessions / resolve_session_slug(game, sessions) / "code_kb"
    if not target.exists():
        return False
    _sh.rmtree(target)
    # Drop any cached handle pointing at the now-deleted store, otherwise
    # subsequent helper calls would return a stale (in-memory) view.
    from code_kb.store import _CODE_STORE_CACHE
    _CODE_STORE_CACHE.pop(str(target), None)
    return True


def preview_asm_scoping(
    *,
    game: str,
    asm_dir: str | Path | None,
    override: list[str | Path] | None = None,
    extra_files: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Run `select_asm_files` without ingesting — used to render the UI's
    "what will get loaded for this game" preview before the user submits.
    """
    from code_kb import select_asm_files

    s = select_asm_files(
        game=game, asm_dir=asm_dir,
        override=override, extra_files=extra_files,
    )
    return {
        "strategy": s.strategy,
        "tokens": s.tokens,
        "suggestion": s.suggestion,
        "selected": [str(p) for p in s.selected],
        "skipped": [
            {"path": str(p), "reason": r} for p, r in s.skipped
        ],
    }


def list_dump_candidates(memdump_dir: str | Path | None, *, game: str):
    """Return matched/other dump files under `memdump_dir`."""
    from code_kb import candidate_dumps

    matched, others = candidate_dumps(memdump_dir, game=game)
    return {
        "matched": [str(p) for p in matched],
        "others": [str(p) for p in others],
    }


def top_routines_by_xrefs(
    game: str, *, limit: int = 12,
) -> list[dict[str, Any]]:
    """Return the most-called routines (handy starting points for the UI)."""
    store = _resolve_code_store(game)
    if store is None:
        return []
    return store.query(
        "WITH l1 AS ("
        "  SELECT h.annotation_id, h.start_addr, h.text,"
        "         h.name_suggestion, h.idiom_match, h.confidence,"
        "         ROW_NUMBER() OVER ("
        "           PARTITION BY h.start_addr"
        "           ORDER BY h.confidence DESC, h.annotation_id DESC"
        "         ) AS rn"
        "    FROM hypotheses h"
        "   WHERE h.layer = 1"
        ")"
        "SELECT cr.start_addr, cr.end_addr,"
        "       CASE"
        "         WHEN (cr.name IS NULL OR lower(cr.name) LIKE 'sub_%')"
        "              AND l1.name_suggestion IS NOT NULL"
        "              AND l1.confidence >= 0.60"
        "         THEN l1.name_suggestion"
        "         ELSE cr.name"
        "       END AS name,"
        "       CASE WHEN l1.annotation_id IS NULL THEN 0 ELSE 1 END AS has_layer1,"
        "       COUNT(DISTINCT cx.src_addr) AS callers,"
        "       l1.text AS hypothesis_text,"
        "       l1.name_suggestion AS name_suggestion,"
        "       l1.idiom_match AS idiom_match,"
        "       l1.confidence AS layer1_confidence"
        "  FROM code_routines cr"
        "  LEFT JOIN code_xrefs cx"
        "    ON cx.dst_addr BETWEEN cr.start_addr AND cr.end_addr"
        "   AND cx.kind IN ('jsr','jmp','jmp_indirect')"
        "  LEFT JOIN l1"
        "    ON l1.start_addr = cr.start_addr AND l1.rn = 1"
        " GROUP BY cr.start_addr"
        " ORDER BY has_layer1 DESC, callers DESC, cr.start_addr"
        " LIMIT ?",
        (int(limit),),
    )


def annotate_routine(
    game: str,
    *,
    start: int,
    role: str = "analyst",
    backup_roles: list[str] | None = None,
    disasm_engine: str = "capstone",
) -> dict[str, Any]:
    """Run one on-demand Layer-1 annotation for a routine start address.

    Intended for the Streamlit Routines tab so users can enrich semantic
    fields (`idiom`, `confidence`, `hypothesis`) without launching a full
    planner/executor run.
    """
    store = _resolve_code_store(game)
    if store is None:
        return {
            "ok": False,
            "error": "No code_kb store found for this game.",
        }

    try:
        from graph.code_kb_node import _mode_annotate
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"Annotate handler import failed: {type(e).__name__}: {e}",
        }

    args = {
        "start": f"${int(start) & 0xFFFF:04X}",
        "role": role,
        "backup_roles": backup_roles or ["critic", "synthesizer"],
        "auto_disasm_if_missing": True,
        "disasm_engine": disasm_engine,
    }
    state = {"question": ""}
    step_id = f"ui_annotate_{int(start) & 0xFFFF:04X}"

    try:
        out = _mode_annotate(state, store, args, step_id)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"Annotate failed: {type(e).__name__}: {e}",
        }

    results = out.get("tool_results") or []
    if not results:
        return {
            "ok": False,
            "error": "Annotate returned no tool_results payload.",
            "raw": out,
        }
    return dict(results[0])
