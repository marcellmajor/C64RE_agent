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
from pathlib import Path
from typing import Any, Iterator

from langgraph.checkpoint.memory import MemorySaver

from graph.build import build_graph

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
    raw_state: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    thread_id: str = ""


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

    initial_state = {
        "game": game,
        "question": question,
        "dump_path": str(dump_path),
        "partial_asm_path": str(partial_asm) if partial_asm else None,
        "text_dir": str(text_dir) if text_dir else None,
        "asm_dir": str(asm_dir) if asm_dir else None,
        "asm_files": (
            [str(p) for p in asm_files] if asm_files else None
        ),
        "plan": [],
        "tool_results": [],
        "history": [],
        "messages": [],
    }

    tid = thread_id or f"webui-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": tid}}

    yield ("status", f"thread_id={tid}")
    seen_msgs = 0
    last_state: dict[str, Any] = {}
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
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log.error("run_question failed:\n%s", tb)
        yield ("error", f"{type(e).__name__}: {e}\n\n```\n{tb}\n```")
        return

    final = graph.get_state(config).values or last_state
    candidate = final.get("candidate_answer") or {}
    verdict_obj = final.get("verdict") or {}
    answer = str(candidate.get("answer") or "(no answer produced)")
    evidence = [str(e) for e in (candidate.get("evidence") or [])]
    blob = answer + "\n" + "\n".join(evidence)
    addrs = parse_addresses(blob)

    result = TurnResult(
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
            str(q) for q in (candidate.get("open_questions") or [])
        ],
        addresses=addrs,
        raw_state=final,
        elapsed_s=time.monotonic() - started,
        thread_id=tid,
    )
    yield ("done", result)


# --------------------------------------------------------------------------- #
# Code-lens helpers — used by the UI right after a turn completes.
# --------------------------------------------------------------------------- #


def _resolve_code_store(game: str):
    """Locate the persistent CodeKnowledgeStore for a game, if it exists."""
    from code_kb import get_code_store

    slug = (game or "unknown").strip().lower().replace(" ", "_")
    root = Path("sessions") / slug / "code_kb"
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


def reset_code_kb(game: str) -> bool:
    """Delete `sessions/<slug>/code_kb/` so the next run rebuilds it.

    Used when a previous run mixed files from several games and the
    user wants a clean slate. Returns True iff something was removed.
    """
    import shutil as _sh
    slug = (game or "unknown").strip().lower().replace(" ", "_")
    target = Path("sessions") / slug / "code_kb"
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
        "       COUNT(cx.src_addr) AS callers,"
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
