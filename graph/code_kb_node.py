"""LangGraph node implementing the `code_kb` planner tool.

Modes the planner can request via ``args["mode"]``:

    - **stats**       : counts (routines, xrefs, SMC, instructions, …)
    - **routines**    : list code_routines (optional `like` for name filter)
    - **routine**     : full window for a single routine (by `start` addr)
    - **pseudocode**  : conservative address-preserving Layer-0 transliteration
    - **xrefs_to**    : who calls/jumps to `addr`?
    - **xrefs_from**  : where does `addr` (or routine `start..end`) jump?
    - **smc**         : list SMC suspects
    - **search**      : substring search over the ingested partial-asm
                       documents (whole-file, not just labels)
    - **disasm**      : disassemble a fresh range via capstone (default)
                       or vice (`engine: vice`); results feed Layer 0
    - **annotate**    : run Layer-1 LLM annotation on one routine
    - **layer2**      : group verified routines into global behaviours (LLM)
    - **groups**      : list stored Layer-2 behaviour groups
    - **layer3**      : adversarially review Layer-1/2 annotations (LLM)
    - **critiques**   : list stored Layer-3 critique records
    - **export**      : commented .asm dump of the entire code KB
    - **hardware**    : return the curated hardware pack as text
    - **schema**      : print the SQLite schema cheat-sheet
    - **sql**         : raw read-only SQL (same safety rules as parent KB)

The node always answers with a `_record_result` payload so the planner
loop can route the response just like any other tool.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from typing import Any
from tools.arguments import address_arg, boolean

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from code_kb import (
    Annotation,
    DEEP_RETRO_RE_PREAMBLE,
    annotation_from_layer1_json,
    build_window,
    disasm_capstone,
    disasm_vice,
    export_commented_asm,
    export_vice_symbols,
    fetch_routines,
    get_code_store,
    render_mechanical_pseudocode,
    render_user_prompt,
)
from code_kb import hardware_pack
from code_kb.schema import ANN_CRITIQUE, ANN_GROUP, EVT_LAYER_RUN
from graph.state import C64State
from memory.redaction import redact_text, redact_value

# We deliberately reach into nodes for `_record_result` / `_step_for` /
# `_hex_to_int` — re-implementing them here would drift from the contract
# the rest of the graph relies on.
from graph.nodes import (
    _hex_to_int,
    _record_result,
    _remaining_llm_budget,
    _step_for,
    _try_parse_json,
)


# --------------------------------------------------------------------------- #
# SQL safety + rewrites — mirror the parent KB so the planner sees a
# consistent contract across both stores.
# --------------------------------------------------------------------------- #


CODE_KB_SCHEMA_HINT = (
    "Tables (SQLite — separate from the parent KB). Read-only SELECT only.\n"
    "  events(id TEXT PK, ts TEXT, kind TEXT, source TEXT, payload_json TEXT)\n"
    "    kind ∈ {ingest_asm, ingest_dump, annotation, disasm_window, layer_run}\n"
    "  annotations(id TEXT PK, layer INT, kind TEXT, start_addr INT,\n"
    "              end_addr INT, producer TEXT, confidence REAL,\n"
    "              payload_json TEXT, evidence_json TEXT,\n"
    "              supersedes_json TEXT, flags_json TEXT, ts TEXT)\n"
    "    annotation kind ∈ {asm_doc, label, routine, xref, smc, classify,\n"
    "                       indirect_jump, disasm_window, hypothesis,\n"
    "                       group, critique}\n"
    "  code_routines(start_addr PK, end_addr, name, summary, source_file,"
    "                entries_json, exits_json, size_bytes, annotation_id,"
    "                confidence)\n"
    "  code_xrefs(src_addr, dst_addr, via_vector, kind, source_file,\n"
    "             annotation_id)  -- one row per (edge, source)\n"
    "    src_addr is the instruction address; dst_addr is the target.\n"
    "    Use these exact names, not source_addr / target_addr.\n"
    "    kind ∈ {jsr, jmp, branch, jmp_indirect, fallthrough}\n"
    "  code_smc_sites(src_addr, dst_addr, mnemonic, operand, smc_kind)\n"
    "  code_data_refs(src_addr, dst_addr, access, index_reg, indirect,\n"
    "                 source_file, annotation_id)  -- data-flow index\n"
    "    access ∈ {r, w, rmw}; one row per (ref, source) — SELECT DISTINCT.\n"
    "    Prefer the modes writes_to / refs_to / hardware_refs over raw SQL.\n"
    "  code_class(addr PK, classification, confidence, evidence)\n"
    "    classification ∈ {code, data, ambiguous}\n"
    "  code_labels(addr, name, source_file, annotation_id)\n"
    "  asm_docs(path PK, size, mtime, content, instructions, routines)\n"
    "  instructions(addr PK, bytes_hex, mnemonic, operand, size_bytes,\n"
    "               source_file, annotation_id)\n"
    "  hypotheses(annotation_id PK, layer, start_addr, end_addr, text,\n"
    "             name_suggestion, idiom_match,\n"
    "             hardware_touched_json, routine_id, source_file,\n"
    "             confidence, flags_json, producer)\n"
    "Tip: for layer-1 hypotheses, query `hypotheses` directly — the\n"
    "JSON payload is already projected.\n"
)


_CODE_KB_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|replace|"
    r"vacuum|pragma|reindex|truncate)\b",
    re.IGNORECASE,
)


_CODE_KB_SQL_REWRITES: tuple[tuple[str, str], ...] = (
    # Reference tables use src/dst, while planners often spell out source/target.
    (r"\bsource_addr\b", "src_addr"),
    (r"\btarget_addr\b", "dst_addr"),
    (r"\baddress\b",     "addr"),
    (r"\baddr_hex\b",    "addr"),
    (r"\bstart\b(?!_addr)",  "start_addr"),
    (r"\bend\b(?!_addr)",    "end_addr"),
    (r"\bevent_id\b",    "id"),
    # LLMs often use bare table names (from the parent KB schema) instead of
    # the code_-prefixed names used in the code KB.  \b ensures we don't
    # double-prefix already-correct names like `code_routines`.
    (r"\broutines\b",    "code_routines"),
    (r"\bxrefs\b",       "code_xrefs"),
    (r"\bsmc_sites\b",   "code_smc_sites"),
    (r"\blabels\b(?!\s*\()",  "code_labels"),  # not "labels(" (function call)
)


def _rewrite_sql(sql: str) -> tuple[str, list[str]]:
    if not sql:
        return sql, []
    parts = re.split(r"('(?:[^']|'')*')", sql)
    applied: list[str] = []
    for i in range(0, len(parts), 2):
        seg = parts[i]
        for pat, repl in _CODE_KB_SQL_REWRITES:
            new_seg, n = re.subn(pat, repl, seg, flags=re.IGNORECASE)
            if n:
                applied.append(f"{pat} -> {repl} (×{n})")
                seg = new_seg
        parts[i] = seg
    return "".join(parts), applied


# --------------------------------------------------------------------------- #
# Mode handlers
# --------------------------------------------------------------------------- #


def _mode_stats(store, args, step_id):
    s = store.stats()
    asm_paths = store.query("SELECT path, instructions, routines, size FROM asm_docs ORDER BY path", ())
    text = json.dumps(
        {"stats": s, "asm_docs": asm_paths, "schema_hint": "use mode='schema'"},
        indent=2, default=str,
    )
    return _record_result("code_kb", step_id, True, text, extra={"mode": "stats"})


def _mode_schema(store, args, step_id):
    return _record_result(
        "code_kb", step_id, True, CODE_KB_SCHEMA_HINT,
        extra={"mode": "schema"},
    )


def _mode_hardware(store, args, step_id):
    return _record_result(
        "code_kb", step_id, True, hardware_pack.pack_text(),
        extra={"mode": "hardware"},
    )


def _mode_routines(store, args, step_id):
    limit = int(args.get("limit", 50))
    offset = int(args.get("offset", 0))
    name_like = args.get("like") or args.get("name_like")
    rows = fetch_routines(store, limit=limit, offset=offset, name_like=name_like)
    for r in rows:
        if isinstance(r.get("start_addr"), int):
            r["start_hex"] = f"${r['start_addr']:04X}"
        if isinstance(r.get("end_addr"), int):
            r["end_hex"] = f"${r['end_addr']:04X}"
    text = json.dumps(rows, indent=2, default=str)
    return _record_result(
        "code_kb", step_id, True, text,
        extra={"mode": "routines", "row_count": len(rows)},
    )


def _containing_routine(store, addr: int):
    """Smallest Layer-0 routine whose span covers *addr*, if any."""
    rows = store.query(
        "SELECT start_addr, end_addr, name, source_file, entries_json,"
        " exits_json, annotation_id FROM code_routines"
        " WHERE start_addr <= ? AND end_addr >= ?"
        " ORDER BY (start_addr = ?) DESC, (end_addr - start_addr) ASC,"
        " start_addr ASC LIMIT 1",
        (addr, addr, addr),
    )
    return rows[0] if rows else None


def _routine_resolution_note(requested: int, routine: dict) -> str:
    start, end = int(routine["start_addr"]), int(routine["end_addr"])
    if start == requested:
        return ""
    return (
        f"Requested ${requested:04X} lies inside stored routine "
        f"${start:04X}-${end:04X}; using its entry ${start:04X}.\n\n"
    )


def _no_routine_message(store, addr: int, mode: str) -> str:
    """Explain a start-address miss, naming the enclosing routine if there is one.

    An address taken from a disassembly listing usually lands *inside* a
    routine rather than on its entry point. Saying which routine that is
    turns a dead end into a one-step retry, without this mode silently
    answering about a routine the caller did not ask for.
    """
    base = f"no routine found with start_addr=${addr:04X}."
    row = _containing_routine(store, addr)
    if row is None:
        return base + " Try mode='routines' first."
    start = int(row["start_addr"])
    end = int(row["end_addr"])
    name = (row["name"] or "").strip()
    named = f" ({name})" if name else ""
    return (
        f"{base} ${addr:04X} lies inside routine "
        f"${start:04X}-${end:04X}{named}; re-run mode='{mode}' with "
        f"start=\"${start:04X}\" for that routine, or mode='routines' "
        "to list entry points."
    )


def _mode_routine(store, args, step_id):
    start_raw = next((args[k] for k in ("start", "addr", "address") if k in args), None)
    if start_raw is None:
        return _record_result(
            "code_kb", step_id, False,
            "code_kb mode='routine' requires args.start (e.g. \"$0810\").",
            extra={"mode": "routine"},
        )
    start = _hex_to_int(start_raw, -1)
    if start < 0:
        return _record_result(
            "code_kb", step_id, False,
            f"could not parse start={start_raw!r} as address",
            extra={"mode": "routine"},
        )
    routine = _containing_routine(store, start)
    if not routine:
        return _record_result(
            "code_kb", step_id, False,
            _no_routine_message(store, start, "routine"),
            extra={"mode": "routine"},
        )
    win = build_window(store, routine_row=routine, focus_addr=start)
    text = _routine_resolution_note(start, routine) + render_user_prompt(win)
    return _record_result(
        "code_kb", step_id, True, text,
        extra={
            "mode": "routine",
            "requested_start_addr": start,
            "start_addr": win.start_addr, "end_addr": win.end_addr,
            "name": win.name, "callers": len(win.callers),
            "callees": len(win.callees), "smc_sites": len(win.smc_sites),
        },
    )


def _mode_pseudocode(store, args, step_id):
    """Address-preserving Layer-0 transliteration for one known routine."""
    start_raw = next((args[k] for k in ("start", "addr", "address") if k in args), None)
    if start_raw is None:
        return _record_result(
            "code_kb", step_id, False,
            "code_kb mode='pseudocode' requires args.start (e.g. \"$0810\").",
            extra={"mode": "pseudocode"},
        )
    start = _hex_to_int(start_raw, -1)
    if start < 0:
        return _record_result(
            "code_kb", step_id, False,
            f"could not parse start={start_raw!r} as address",
            extra={"mode": "pseudocode"},
        )
    routine = _containing_routine(store, start)
    if not routine:
        return _record_result(
            "code_kb", step_id, False,
            _no_routine_message(store, start, "pseudocode"),
            extra={"mode": "pseudocode"},
        )
    requested = start
    start = int(routine["start_addr"])
    end = int(routine["end_addr"])
    instructions = store.query(
        "SELECT addr, bytes_hex, mnemonic, operand, size_bytes"
        " FROM instructions WHERE addr BETWEEN ? AND ? ORDER BY addr",
        (start, end),
    )
    if not instructions:
        return _record_result(
            "code_kb", step_id, False,
            f"routine ${start:04X}-${end:04X} has no Layer-0 instructions",
            extra={"mode": "pseudocode", "start_addr": start,
                   "end_addr": end},
        )
    text = _routine_resolution_note(requested, routine) + render_mechanical_pseudocode(
        instructions, routine_name=routine.get("name"),
    )
    return _record_result(
        "code_kb", step_id, True, text,
        extra={
            "mode": "pseudocode",
            "requested_start_addr": requested,
            "start_addr": start,
            "end_addr": end,
            "instruction_count": len(instructions),
            "provenance": "mechanical_layer0",
        },
    )


def _mode_xrefs_to(store, args, step_id):
    addr = address_arg(args, "addr", "dst")
    # DISTINCT: xref rows are per-source (Phase 2 hardening) — the same
    # edge asserted by several files must show once here.
    rows = store.query(
        "SELECT DISTINCT src_addr, kind FROM code_xrefs"
        " WHERE dst_addr = ? ORDER BY src_addr LIMIT ?",
        (addr & 0xFFFF, int(args.get("limit", 100))),
    )
    for r in rows:
        if isinstance(r.get("src_addr"), int):
            r["src_hex"] = f"${r['src_addr']:04X}"
    text = json.dumps(rows, indent=2, default=str)
    return _record_result(
        "code_kb", step_id, True, text,
        extra={"mode": "xrefs_to", "addr": f"${addr:04X}", "row_count": len(rows)},
    )


def _mode_xrefs_from(store, args, step_id):
    src = address_arg(args, "addr", "src")
    end = address_arg(args, "end", default=src)
    rows = store.query(
        "SELECT DISTINCT src_addr, dst_addr, kind, via_vector"
        "  FROM code_xrefs WHERE src_addr BETWEEN ? AND ?"
        "  ORDER BY src_addr LIMIT ?",
        (src, end, int(args.get("limit", 200))),
    )
    for r in rows:
        if isinstance(r.get("src_addr"), int):
            r["src_hex"] = f"${r['src_addr']:04X}"
        if isinstance(r.get("dst_addr"), int):
            r["dst_hex"] = f"${r['dst_addr']:04X}"
        if isinstance(r.get("via_vector"), int):
            r["via_hex"] = f"${r['via_vector']:04X}"
    text = json.dumps(rows, indent=2, default=str)
    return _record_result(
        "code_kb", step_id, True, text,
        extra={
            "mode": "xrefs_from",
            "from": f"${src:04X}", "to": f"${end:04X}", "row_count": len(rows),
        },
    )


def _mode_smc(store, args, step_id):
    limit = int(args.get("limit", 100))
    rows = store.query(
        "SELECT src_addr, dst_addr, mnemonic, operand, smc_kind"
        "  FROM code_smc_sites ORDER BY src_addr LIMIT ?",
        (limit,),
    )
    for r in rows:
        if isinstance(r.get("src_addr"), int):
            r["src_hex"] = f"${r['src_addr']:04X}"
        if isinstance(r.get("dst_addr"), int):
            r["dst_hex"] = f"${r['dst_addr']:04X}"
    text = json.dumps(rows, indent=2, default=str)
    return _record_result(
        "code_kb", step_id, True, text,
        extra={"mode": "smc", "row_count": len(rows)},
    )


def _mode_writes_to(store, args, step_id):
    """Instructions that WRITE (or RMW) a target address (tracker 3.4)."""
    addr = address_arg(args, "addr", "dst")
    limit = int(args.get("limit", 100))
    rows = store.query(
        "SELECT DISTINCT src_addr, access, index_reg, indirect"
        "  FROM code_data_refs WHERE dst_addr = ?"
        "   AND access IN ('w', 'rmw') ORDER BY src_addr LIMIT ?",
        (addr & 0xFFFF, limit),
    )
    for r in rows:
        r["src_hex"] = f"${r['src_addr']:04X}"
    text = json.dumps(rows, indent=2, default=str)
    return _record_result(
        "code_kb", step_id, True, text,
        extra={"mode": "writes_to", "addr": f"${addr:04X}",
               "row_count": len(rows)},
    )


def _mode_refs_to(store, args, step_id):
    """All references (read/write/rmw) to a target address (tracker 3.4)."""
    addr = address_arg(args, "addr", "dst")
    limit = int(args.get("limit", 200))
    rows = store.query(
        "SELECT DISTINCT src_addr, access, index_reg, indirect"
        "  FROM code_data_refs WHERE dst_addr = ?"
        " ORDER BY src_addr LIMIT ?",
        (addr & 0xFFFF, limit),
    )
    for r in rows:
        r["src_hex"] = f"${r['src_addr']:04X}"
    text = json.dumps(rows, indent=2, default=str)
    return _record_result(
        "code_kb", step_id, True, text,
        extra={"mode": "refs_to", "addr": f"${addr:04X}",
               "row_count": len(rows)},
    )


def _mode_hardware_refs(store, args, step_id):
    """References into a hardware chip's register range (tracker 3.4).

    `chip` ∈ {vic|vic-ii, sid, cia|cia1, cia2}; or pass an explicit
    `lo`/`hi` range.
    """
    chip = str(args.get("chip") or "").strip().lower()
    ranges = {
        "vic": (0xD000, 0xD3FF), "vic-ii": (0xD000, 0xD3FF),
        "vicii": (0xD000, 0xD3FF),
        "sid": (0xD400, 0xD7FF),
        "cia": (0xDC00, 0xDCFF), "cia1": (0xDC00, 0xDCFF),
        "cia2": (0xDD00, 0xDDFF),
    }
    if chip in ranges:
        lo, hi = ranges[chip]
    else:
        lo = address_arg(args, "lo", default=0xD000)
        hi = address_arg(args, "hi", default=0xDFFF)
    limit = int(args.get("limit", 300))
    rows = store.query(
        "SELECT DISTINCT src_addr, dst_addr, access, index_reg"
        "  FROM code_data_refs WHERE dst_addr BETWEEN ? AND ?"
        " ORDER BY dst_addr, src_addr LIMIT ?",
        (lo, hi, limit),
    )
    for r in rows:
        r["src_hex"] = f"${r['src_addr']:04X}"
        r["dst_hex"] = f"${r['dst_addr']:04X}"
    text = json.dumps(rows, indent=2, default=str)
    return _record_result(
        "code_kb", step_id, True, text,
        extra={"mode": "hardware_refs", "chip": chip or f"${lo:04X}-${hi:04X}",
               "row_count": len(rows)},
    )


def _mode_search(store, args, step_id):
    q = (args.get("q") or args.get("query") or "").strip()
    if not q:
        return _record_result(
            "code_kb", step_id, False,
            "code_kb mode='search' requires args.q (substring/keyword).",
            extra={"mode": "search"},
        )
    limit = int(args.get("limit", 10))
    rows = store.query(
        "SELECT path, content FROM asm_docs"
        " WHERE lower(content) LIKE lower(?) LIMIT ?",
        (f"%{q}%", limit),
    )
    out = []
    for r in rows[:limit]:
        content = r.get("content") or ""
        idx = content.lower().find(q.lower())
        if idx < 0:
            snippet = content[:800]
        else:
            start = max(0, idx - 200)
            end = min(len(content), idx + 800)
            snippet = (
                ("..." if start > 0 else "")
                + content[start:end]
                + ("..." if end < len(content) else "")
            )
        out.append({
            "path": r["path"],
            "match_at": idx,
            "snippet": redact_text(snippet),
        })
    if not out:
        text = f"No asm_doc hits for {q!r}."
    else:
        lines = [f"{len(out)} hit(s) for {q!r}:"]
        for h in out:
            lines.append(f"- {h['path']} (offset={h['match_at']})")
            lines.append(f"  …{h['snippet']}…")
        text = "\n".join(lines)
    text = redact_text(text)
    return _record_result(
        "code_kb", step_id, True, text,
        extra={
            "mode": "search", "q": redact_text(q),
            "hits": redact_value(out),
        },
    )


def _mode_disasm(store, args, step_id):
    engine = str(args.get("engine") or "capstone").strip().lower()
    if engine not in {"capstone", "vice"}:
        raise ValueError("engine must be capstone or vice")
    if engine == "vice":
        addr = address_arg(args, "address", "addr")
        count = int(args.get("count", 32))
        out = disasm_vice(store, address=addr, count=count, note=args.get("note"))
    else:
        start = address_arg(args, "start", "addr")
        length = int(args.get("length", 256))
        recursive = boolean(args.get("recursive", False), "recursive")
        max_insns = int(args.get("max_insns", 600))
        out = disasm_capstone(
            store,
            start=start, length=length,
            recursive=recursive, max_insns=max_insns,
            note=args.get("note"),
        )
    if not out.get("ok"):
        return _record_result(
            "code_kb", step_id, False,
            str(out.get("error") or "disasm failed"),
            extra={"mode": "disasm", "engine": engine, **{k: v for k, v in out.items() if k != "rows"}},
        )
    listing = out.get("listing") or ""
    head_lines = listing.splitlines()
    if len(head_lines) > 300:
        listing = "\n".join(head_lines[:300]) + f"\n... ({len(head_lines) - 300} more lines)"
    text = (
        f"Disassembled via {engine}; new layer-0 facts:\n"
        f"  +{out['stats'].get('instructions', 0)} instructions, "
        f"+{out['stats'].get('xrefs', 0)} xrefs, "
        f"+{out['stats'].get('smc_sites', 0)} SMC, "
        f"+{out['stats'].get('indirect_sites', 0)} indirect.\n\n"
        f"{listing}"
    )
    return _record_result(
        "code_kb", step_id, True, text,
        extra={
            "mode": "disasm", "engine": engine,
            "stats": out.get("stats"),
            "row_count": len(out.get("rows") or []),
        },
    )


def _mode_export(store, args, step_id):
    # Export everything to disk under the session directory, and return the
    # path + a head excerpt.
    from pathlib import Path
    out_path_raw = args.get("path")
    export_format = str(args.get("format") or "asm").strip().lower()
    if out_path_raw:
        out_path = Path(out_path_raw)
    elif export_format in {"symbols", "labels", "vice"}:
        out_path = store.root.parent / "code_kb.labels"
    else:
        out_path = store.root.parent / "code_kb_annotated.asm"
    if export_format in {"symbols", "labels", "vice"}:
        text = export_vice_symbols(
            store,
            min_confidence=float(args.get("min_confidence") or 0.75),
        )
    else:
        text = export_commented_asm(
            store,
            game_name=args.get("game"),
            min_confidence=float(args.get("min_confidence") or 0.0),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    head = "\n".join(text.splitlines()[:40])
    return _record_result(
        "code_kb", step_id, True,
        f"Wrote {len(text)} bytes to {out_path}\n\n--- head ---\n{head}",
        extra={
            "mode": "export", "format": export_format,
            "path": str(out_path),
            "bytes": len(text),
        },
    )


def _mode_sql(store, args, step_id):
    sql = args.get("sql")
    if not sql:
        return _record_result(
            "code_kb", step_id, False,
            "code_kb mode='sql' requires args.sql.\n\n" + CODE_KB_SCHEMA_HINT,
            extra={"mode": "sql", "schema": CODE_KB_SCHEMA_HINT},
        )
    sql = str(sql).strip().rstrip(";")
    if _CODE_KB_SQL_FORBIDDEN.search(sql):
        return _record_result(
            "code_kb", step_id, False,
            "code_kb mode='sql' is read-only — INSERT/UPDATE/DELETE/DDL "
            "rejected.",
            extra={
                "mode": "sql", "sql": redact_text(sql),
                "rejection": "forbidden_statement",
            },
        )
    rewritten, applied = _rewrite_sql(sql)
    last_err = None
    for attempt_sql, label in (
        (sql, "original"),
        *([(rewritten, "auto-rewrite")]
          if applied and rewritten != sql else []),
    ):
        try:
            rows = store.query(attempt_sql)
            safe_rows = redact_value(rows[: int(args.get("limit", 100))])
            text = json.dumps(safe_rows,
                              indent=2, default=str)
            extra = {
                "mode": "sql", "sql": redact_text(attempt_sql),
                "row_count": len(rows),
            }
            if label != "original":
                extra["sql_rewrites_applied"] = applied
            return _record_result(
                "code_kb", step_id, True, text, extra=extra,
            )
        except Exception as e:  # noqa: BLE001
            last_err = e

    return _record_result(
        "code_kb", step_id, False,
        redact_text(
            f"{type(last_err).__name__ if last_err else 'Error'}: {last_err}\n\n"
            + CODE_KB_SCHEMA_HINT,
        ),
        extra={
            "mode": "sql", "sql": redact_text(sql),
            "schema": CODE_KB_SCHEMA_HINT,
        },
    )


# --------------------------------------------------------------------------- #
# Layer-1 annotate mode (LLM call)
# --------------------------------------------------------------------------- #


_LAYER_LLM_ROLES = frozenset({
    "analyst", "critic", "curator", "executor", "planner", "synthesizer",
})


def _normalize_layer_roles(
    raw_role: Any,
    raw_backup_roles: Any,
    *,
    default_role: str,
    default_backup_roles: list[str],
) -> tuple[str, list[str]]:
    """Keep planner prose out of the configured-agent role argument."""
    requested = str(raw_role or "").strip().lower()
    role = requested if requested in _LAYER_LLM_ROLES else default_role
    if raw_backup_roles is None:
        raw_backups = list(default_backup_roles)
    elif isinstance(raw_backup_roles, str):
        raw_backups = [raw_backup_roles]
    elif isinstance(raw_backup_roles, (list, tuple)):
        raw_backups = list(raw_backup_roles)
    else:
        raw_backups = []
    backups: list[str] = []
    for raw in raw_backups:
        candidate = str(raw or "").strip().lower()
        if (
            candidate in _LAYER_LLM_ROLES
            and candidate != role
            and candidate not in backups
        ):
            backups.append(candidate)
    if not backups:
        backups = [
            candidate for candidate in default_backup_roles
            if candidate != role
        ]
    return role, backups


def _disasm_window_context(rows, start: int, source: str):
    """Bound a provisional context to contiguous decoded instructions.

    This is an annotation context, not a new Layer-0 routine assertion.
    Stop at a gap or unconditional control transfer instead of merging
    unrelated callees from a recursive disassembly into one routine.
    """
    cursor = start
    for row in sorted(rows, key=lambda row: row["addr"]):
        addr = int(row["addr"])
        if addr < start:
            continue
        size = len(row.get("bytes_hex") or "") // 2
        if addr != cursor or size < 1:
            break
        cursor += size
        if str(row.get("mnemonic", "")).lower() in {"rts", "rti", "jmp", "brk"}:
            break
    if cursor == start:
        return None
    return {
        "start_addr": start, "end_addr": cursor - 1,
        "name": "unverified disassembly window", "source_file": source,
        "entries_json": "[]", "exits_json": "[]",
    }


def _mode_annotate(state, store, args, step_id):
    """Annotate a known routine or an explicitly provisional disassembly window."""
    start_raw = next((args[k] for k in ("start", "addr", "address") if k in args), None)
    if start_raw is None:
        return _record_result(
            "code_kb", step_id, False,
            "code_kb mode='annotate' requires args.start (routine address).",
            extra={"mode": "annotate"},
        )
    requested = _hex_to_int(start_raw, -1)
    if requested < 0:
        return _record_result(
            "code_kb", step_id, False,
            f"could not parse start={start_raw!r}",
            extra={"mode": "annotate"},
        )
    routine = _containing_routine(store, requested)
    start = int(routine["start_addr"]) if routine else requested
    context_kind = "routine"
    resolution_note = _routine_resolution_note(requested, routine) if routine else ""
    auto_disasm = bool(args.get("auto_disasm_if_missing", True))
    auto_disasm_info: dict[str, Any] = {"attempted": False}
    window = build_window(store, routine_row=routine, focus_addr=requested) if routine else None
    if window is None or not window.listing:
        if not auto_disasm:
            return _record_result(
                "code_kb", step_id, False,
                (f"routine ${start:04X} has no Layer-0 instructions" if routine
                 else _no_routine_message(store, requested, "annotate")),
                extra={"mode": "annotate", "requested_start_addr": requested,
                       "auto_disasm": auto_disasm_info},
            )
        engine = str(args.get("disasm_engine") or "capstone").lower()
        source = f"annotate-autodisasm:${start:04X}"
        auto_disasm_info.update(
            attempted=True, engine=engine,
            reason="empty_listing" if routine else "missing_routine",
        )
        if engine == "vice":
            d_out = disasm_vice(
                store, address=start,
                count=int(args.get("disasm_count") or 64), note=source,
            )
        else:
            length = int(routine["end_addr"]) - start + 1 if routine else 768
            d_out = disasm_capstone(
                store, start=start,
                length=int(args.get("disasm_length") or length),
                recursive=bool(args.get("disasm_recursive", False)),
                max_insns=int(args.get("disasm_max_insns") or 800), note=source,
            )
        auto_disasm_info["ok"] = bool(d_out.get("ok"))
        auto_disasm_info["stats"] = d_out.get("stats")
        if not d_out.get("ok"):
            return _record_result(
                "code_kb", step_id, False,
                resolution_note + f"Cannot annotate ${start:04X}: {d_out.get('error')}",
                extra={"mode": "annotate", "requested_start_addr": requested,
                       "start_addr": start, "auto_disasm": auto_disasm_info,
                       "error_code": d_out.get("error_code")},
            )
        if routine is None:
            routine = _disasm_window_context(d_out.get("rows") or [], start, source)
            context_kind = "disassembly_window"
            resolution_note = (
                "No indexed routine covers this address. Analyzing a bounded "
                "disassembly window; routine boundaries are unverified. "
                "Do not claim that this window is a complete routine.\n\n"
            )
        if routine is None:
            return _record_result(
                "code_kb", step_id, False,
                f"No decoded instruction starts at ${start:04X}; "
                "choose a known instruction boundary or verify the bytes in VICE.",
                extra={"mode": "annotate", "requested_start_addr": requested,
                       "auto_disasm": auto_disasm_info},
            )
        window = build_window(store, routine_row=routine, focus_addr=requested)
    if not window.listing:
        return _record_result(
            "code_kb", step_id, False,
            f"No instruction evidence available for annotation at ${start:04X}.",
            extra={"mode": "annotate", "auto_disasm": auto_disasm_info},
        )
    role, backup_roles = _normalize_layer_roles(
        args.get("role"), args.get("backup_roles"),
        default_role="analyst",
        default_backup_roles=["critic", "synthesizer"],
    )

    user_prompt = resolution_note + render_user_prompt(window, question=state.get("question"))
    system_prompt = (
        DEEP_RETRO_RE_PREAMBLE
        + "\n---\n## Hardware reference (keep nearby — do not paraphrase)\n"
        + hardware_pack.pack_text(max_chars=4_000)
    )

    parsed, used_role, last_err = _invoke_layer1(
        role, backup_roles, system_prompt, user_prompt,
        remaining_budget_usd=_remaining_llm_budget(state),
    )
    if parsed is None:
        return _record_result(
            "code_kb", step_id, False,
            f"Layer-1 LLM call failed for ${start:04X}: {last_err}",
            extra={
                "mode": "annotate",
                "start_addr": start,
                "tried_roles": [role, *backup_roles],
            },
        )

    ann = annotation_from_layer1_json(parsed, window, producer=f"layer1:{used_role}")
    ann.payload["context_kind"] = context_kind
    if context_kind == "disassembly_window":
        ann.flags.append("unverified_routine_boundaries")
        ann.payload["routine_id"] = None
    ann_id = store.append_annotation(ann, source="code_kb_node.layer1")
    store.append_event(
        EVT_LAYER_RUN, "code_kb_node",
        {"layer": 1, "start_addr": start, "role": used_role, "annotation_id": ann_id},
    )

    text = (
        resolution_note + f"Layer-1 annotation for ${window.start_addr:04X}-${window.end_addr:04X} "
        f"(producer={used_role}, conf={ann.confidence:.2f}):\n\n"
        f"{ann.payload.get('text', '')}\n\n"
        f"_idiom: {ann.payload.get('idiom_match')}_\n"
        f"_name suggestion: {ann.payload.get('name_suggestion')}_"
    )
    return _record_result(
        "code_kb", step_id, True, text,
        extra={
            "mode": "annotate",
            "start_addr": window.start_addr, "end_addr": window.end_addr,
            "requested_start_addr": requested,
            "context_kind": context_kind,
            "annotation_id": ann_id,
            "producer": ann.producer, "confidence": ann.confidence,
            "idiom_match": ann.payload.get("idiom_match"),
            "name_suggestion": ann.payload.get("name_suggestion"),
            "auto_disasm": auto_disasm_info,
        },
    )


def _bounded_confidence(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _mode_layer2(state, store, args, step_id):
    """Group verified routines into bounded cross-routine behaviours."""
    limit = max(2, min(int(args.get("limit") or 40), 80))
    routines = store.query(
        "SELECT start_addr, end_addr, name, summary, source_file, confidence"
        " FROM code_routines ORDER BY confidence DESC, start_addr LIMIT ?",
        (limit,),
    )
    if len(routines) < 2:
        return _record_result(
            "code_kb", step_id, False,
            "Layer 2 needs at least two verified Layer-0 routines.",
            extra={"mode": "layer2", "routine_count": len(routines)},
        )
    layer1 = store.query(
        "SELECT annotation_id, start_addr, end_addr, text, name_suggestion,"
        " idiom_match, confidence FROM hypotheses"
        " WHERE layer = 1 ORDER BY confidence DESC LIMIT ?",
        (limit,),
    )
    known = {int(row["start_addr"]): row for row in routines}
    prompt = (
        "Group the verified C64 routines below into a small set of global "
        "behaviours (rendering, input, score, audio, loader, IRQ, etc.).\n"
        "Never add a routine address not present in VERIFIED ROUTINES. A "
        "routine may appear in more than one group only when evidence warrants it.\n\n"
        f"VERIFIED ROUTINES:\n{json.dumps(routines, indent=2, default=str)}\n\n"
        f"LAYER-1 HYPOTHESES (fallible):\n"
        f"{json.dumps(layer1, indent=2, default=str)}\n\n"
        "Reply JSON only:\n"
        '{"groups":[{"name":"snake_case","summary":"...",'
        '"routine_starts":["$XXXX"],"confidence":0.0,'
        '"evidence":["$XXXX: reason"]}]}'
    )
    role, backup_roles = _normalize_layer_roles(
        args.get("role"), args.get("backup_roles"),
        default_role="analyst",
        default_backup_roles=["synthesizer"],
    )
    parsed, used_role, last_err = _invoke_layer1(
        role,
        backup_roles,
        DEEP_RETRO_RE_PREAMBLE
        + "\nYou are the Layer-2 global behaviour grouper. Layer 0 wins.",
        prompt,
        remaining_budget_usd=_remaining_llm_budget(state),
    )
    if parsed is None:
        return _record_result(
            "code_kb", step_id, False,
            f"Layer-2 LLM call failed: {last_err}",
            extra={"mode": "layer2", "tried_roles": [role, *backup_roles]},
        )

    raw_groups = parsed.get("groups") or []
    if not isinstance(raw_groups, list):
        raw_groups = []
    stored: list[dict[str, Any]] = []
    rejected_addresses: list[str] = []
    for raw_group in raw_groups[:12]:
        if not isinstance(raw_group, dict):
            continue
        starts: list[int] = []
        for raw_start in raw_group.get("routine_starts") or []:
            start = _hex_to_int(raw_start, -1)
            if start not in known:
                rejected_addresses.append(str(raw_start))
                continue
            if start not in starts:
                starts.append(start)
        if not starts:
            continue
        name = re.sub(
            r"[^a-z0-9_]+", "_",
            str(raw_group.get("name") or "behaviour_group").strip().lower(),
        ).strip("_") or "behaviour_group"
        summary = str(raw_group.get("summary") or "").strip()
        confidence = _bounded_confidence(raw_group.get("confidence"))
        start_addr = min(starts)
        end_addr = max(int(known[start]["end_addr"]) for start in starts)
        stable = json.dumps(
            {"name": name, "starts": starts}, separators=(",", ":"),
        )
        ann = Annotation(
            id="l2_" + hashlib.sha256(stable.encode()).hexdigest()[:16],
            layer=2,
            kind=ANN_GROUP,
            start_addr=start_addr,
            end_addr=end_addr,
            producer=f"layer2:{used_role}",
            confidence=confidence,
            payload={
                "text": summary or f"Cross-routine behaviour group {name}.",
                "name_suggestion": name,
                "routine_starts": starts,
                "routine_id": None,
                "source_file": None,
            },
            evidence=[str(item) for item in (raw_group.get("evidence") or [])],
            flags=["global_group"],
        )
        annotation_id = store.append_annotation(
            ann, source="code_kb_node.layer2",
        )
        stored.append({
            "annotation_id": annotation_id,
            "name": name,
            "routine_starts": [f"${start:04X}" for start in starts],
            "confidence": confidence,
        })
    store.append_event(EVT_LAYER_RUN, "code_kb_node", {
        "layer": 2,
        "role": used_role,
        "groups_written": len(stored),
        "rejected_addresses": rejected_addresses,
    })
    ok = bool(stored)
    return _record_result(
        "code_kb", step_id, ok,
        json.dumps({
            "groups": stored,
            "rejected_unverified_addresses": rejected_addresses,
        }, indent=2),
        extra={
            "mode": "layer2",
            "producer": f"layer2:{used_role}",
            "groups_written": len(stored),
            "rejected_addresses": rejected_addresses,
        },
    )


def _mode_layer3(state, store, args, step_id):
    """Adversarially review existing semantic annotations without deleting them."""
    limit = max(1, min(int(args.get("limit") or 40), 80))
    targets = store.query(
        "SELECT id, layer, kind, start_addr, end_addr, producer, confidence,"
        " payload_json, evidence_json, flags_json, source_file"
        " FROM annotations WHERE layer IN (1, 2)"
        " ORDER BY layer DESC, confidence DESC, seq DESC LIMIT ?",
        (limit,),
    )
    if not targets:
        return _record_result(
            "code_kb", step_id, False,
            "Layer 3 needs at least one Layer-1 or Layer-2 annotation.",
            extra={"mode": "layer3"},
        )
    by_id = {str(row["id"]): row for row in targets}
    prompt = (
        "Adversarially review the semantic annotations below against their "
        "address ranges and evidence. Do not critique deterministic Layer 0. "
        "Use only annotation_id values shown here.\n\n"
        f"ANNOTATIONS:\n{json.dumps(targets, indent=2, default=str)}\n\n"
        "Reply JSON only:\n"
        '{"critiques":[{"annotation_id":"...",'
        '"decision":"support|question|reject","critique":"...",'
        '"confidence":0.0,"evidence":["$XXXX: reason"]}]}'
    )
    role, backup_roles = _normalize_layer_roles(
        args.get("role"), args.get("backup_roles"),
        default_role="critic",
        default_backup_roles=["analyst"],
    )
    parsed, used_role, last_err = _invoke_layer1(
        role,
        backup_roles,
        DEEP_RETRO_RE_PREAMBLE
        + "\nYou are the Layer-3 adversarial annotation critic. Layer 0 wins.",
        prompt,
        remaining_budget_usd=_remaining_llm_budget(state),
    )
    if parsed is None:
        return _record_result(
            "code_kb", step_id, False,
            f"Layer-3 LLM call failed: {last_err}",
            extra={"mode": "layer3", "tried_roles": [role, *backup_roles]},
        )
    raw_critiques = parsed.get("critiques") or []
    if not isinstance(raw_critiques, list):
        raw_critiques = []
    stored: list[dict[str, Any]] = []
    rejected_targets: list[str] = []
    for raw_critique in raw_critiques[:limit]:
        if not isinstance(raw_critique, dict):
            continue
        target_id = str(raw_critique.get("annotation_id") or "").strip()
        target = by_id.get(target_id)
        if target is None:
            rejected_targets.append(target_id or "(missing)")
            continue
        decision = str(raw_critique.get("decision") or "question").lower()
        if decision not in {"support", "question", "reject"}:
            decision = "question"
        critique = str(raw_critique.get("critique") or "").strip()
        confidence = _bounded_confidence(raw_critique.get("confidence"))
        ann = Annotation(
            id="l3_" + hashlib.sha256(target_id.encode()).hexdigest()[:16],
            layer=3,
            kind=ANN_CRITIQUE,
            start_addr=int(target["start_addr"]),
            end_addr=int(target["end_addr"]),
            producer=f"layer3:{used_role}",
            confidence=confidence,
            payload={
                "target_annotation_id": target_id,
                "decision": decision,
                "text": critique,
                "source_file": target.get("source_file"),
            },
            evidence=[str(item) for item in (raw_critique.get("evidence") or [])],
            flags=[f"layer3_{decision}"],
        )
        annotation_id = store.append_annotation(
            ann, source="code_kb_node.layer3",
        )
        stored.append({
            "annotation_id": annotation_id,
            "target_annotation_id": target_id,
            "decision": decision,
            "confidence": confidence,
        })
    store.append_event(EVT_LAYER_RUN, "code_kb_node", {
        "layer": 3,
        "role": used_role,
        "critiques_written": len(stored),
        "rejected_targets": rejected_targets,
    })
    ok = bool(stored)
    return _record_result(
        "code_kb", step_id, ok,
        json.dumps({
            "critiques": stored,
            "rejected_unknown_targets": rejected_targets,
        }, indent=2),
        extra={
            "mode": "layer3",
            "producer": f"layer3:{used_role}",
            "critiques_written": len(stored),
            "rejected_targets": rejected_targets,
        },
    )


def _mode_groups(store, args, step_id):
    limit = max(1, min(int(args.get("limit") or 50), 200))
    rows = store.query(
        "SELECT annotation_id, start_addr, end_addr, text, name_suggestion,"
        " confidence, producer, flags_json FROM hypotheses"
        " WHERE layer = 2 ORDER BY confidence DESC, start_addr LIMIT ?",
        (limit,),
    )
    return _record_result(
        "code_kb", step_id, True,
        json.dumps(rows, indent=2, default=str),
        extra={"mode": "groups", "row_count": len(rows)},
    )


def _mode_critiques(store, args, step_id):
    limit = max(1, min(int(args.get("limit") or 50), 200))
    rows = store.query(
        "SELECT id, start_addr, end_addr, producer, confidence, payload_json,"
        " evidence_json, flags_json FROM annotations"
        " WHERE layer = 3 AND kind = ? ORDER BY seq DESC LIMIT ?",
        (ANN_CRITIQUE, limit),
    )
    decoded = []
    for row in rows:
        try:
            payload = json.loads(row.pop("payload_json") or "{}")
        except json.JSONDecodeError:
            payload = {}
        decoded.append({**row, **payload})
    return _record_result(
        "code_kb", step_id, True,
        json.dumps(decoded, indent=2, default=str),
        extra={"mode": "critiques", "row_count": len(decoded)},
    )


def _invoke_layer1(
    role: str, backup_roles: list[str],
    system_prompt: str, user_prompt: str,
    *,
    remaining_budget_usd: float | None = None,
) -> tuple[dict | None, str | None, str | None]:
    """Try the LLM once per role; return (parsed_json, role_used, last_err)."""
    from graph import usage as llm_usage
    from graph.llm import get_llm
    from graph.nodes import (
        _call_cost_reservation_usd,
        _flatten_lc_ai_message_content,
    )

    last_err = None
    remaining = remaining_budget_usd
    for r in [role, *backup_roles]:
        model_name = None
        try:
            llm = get_llm(r)
            model_name = (
                getattr(llm, "model_name", None) or getattr(llm, "model", None)
            )
            reservation = _call_cost_reservation_usd(
                llm,
                model_name,
                user_prompt,
                system_text=system_prompt,
            )
            if remaining is not None and reservation > remaining:
                last_err = (
                    f"{r}: cost reservation ${reservation:.4f} exceeds "
                    f"remaining run allowance ${remaining:.4f}"
                )
                llm_usage.record({
                    "role": f"layer1:{r}",
                    "model": model_name,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "reserved_cost_usd": reservation,
                    "ok": False,
                    "transport_ok": False,
                    "rejection": "cost_reservation",
                    "error": last_err[:200],
                })
                continue
            msg = llm.invoke([
                SystemMessage(system_prompt),
                HumanMessage(user_prompt),
            ])
        except Exception as e:  # noqa: BLE001
            llm_usage.record({
                "role": f"layer1:{r}", "model": model_name,
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "ok": False, "error": f"{type(e).__name__}: {e}"[:200],
            })
            last_err = f"{r}: {type(e).__name__}: {e}"
            print(f"[code_kb:layer1:{r}] FAILED — {last_err}",
                  file=sys.stderr, flush=True)
            continue

        # Layer-1 annotation calls count against the run budget too
        # (tracker 1.3/1.4) — they used to be entirely invisible. `ok`
        # means "usable parsed contract", not transport success (review
        # finding 5): empty/non-JSON replies are demoted below.
        in_tok, out_tok = llm_usage.extract_usage(msg)
        entry = llm_usage.record({
            "role": f"layer1:{r}", "model": model_name,
            "input_tokens": in_tok, "output_tokens": out_tok,
            "cost_usd": llm_usage.estimate_cost_usd(model_name, in_tok, out_tok),
            "ok": True, "transport_ok": True,
        })
        if remaining is not None:
            remaining = max(
                0.0, remaining - float(entry.get("cost_usd") or 0.0),
            )

        content = _flatten_lc_ai_message_content(msg)
        if not content.strip():
            last_err = f"{r}: empty response"
            llm_usage.mark_failed(entry, "empty response")
            print(f"[code_kb:layer1:{r}] empty response", file=sys.stderr, flush=True)
            continue
        parsed = _try_parse_json(content)
        if isinstance(parsed, dict):
            return parsed, r, None
        last_err = f"{r}: non-JSON response ({len(content)} chars)"
        llm_usage.mark_failed(entry, f"non-JSON response ({len(content)} chars)")
        print(f"[code_kb:layer1:{r}] non-JSON response", file=sys.stderr, flush=True)
    return None, None, last_err


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


_MODE_HANDLERS = {
    "stats":         _mode_stats,
    "schema":        _mode_schema,
    "hardware":      _mode_hardware,
    "routines":      _mode_routines,
    "routine":       _mode_routine,
    "pseudocode":    _mode_pseudocode,
    "xrefs_to":      _mode_xrefs_to,
    "xrefs_from":    _mode_xrefs_from,
    "smc":           _mode_smc,
    "writes_to":     _mode_writes_to,
    "refs_to":       _mode_refs_to,
    "hardware_refs": _mode_hardware_refs,
    "groups":        _mode_groups,
    "critiques":     _mode_critiques,
    "search":        _mode_search,
    "disasm":        _mode_disasm,
    "export":        _mode_export,
    "sql":           _mode_sql,
}


def code_kb_node(state: C64State) -> dict[str, Any]:
    """Dispatch the planner's `code_kb` step into a mode handler."""
    step = _step_for(state, state.get("current_step_id"))
    args = step.get("args", {}) or {}
    step_id = step.get("id", "?")

    handle = state.get("code_kb_handle")
    if not handle:
        return _record_result(
            "code_kb", step_id, False,
            "code_kb is unavailable: no `code_kb_handle` in state. "
            "Pass `--asm-dir <dir>` (or `--partial-asm <file>`) on launch "
            "to populate the code KB.",
            extra={"mode": args.get("mode")},
        )
    store = get_code_store(handle)
    mode = str(args.get("mode") or "stats").lower().strip()

    layered_modes = {
        "annotate": _mode_annotate,
        "layer2": _mode_layer2,
        "layer3": _mode_layer3,
    }
    if mode in layered_modes:
        try:
            return layered_modes[mode](state, store, args, step_id)
        except Exception as e:  # noqa: BLE001
            return _record_result(
                "code_kb", step_id, False,
                f"{type(e).__name__}: {e}",
                extra={"mode": mode, "args": args},
            )

    handler = _MODE_HANDLERS.get(mode)
    if handler is None:
        return _record_result(
            "code_kb", step_id, False,
            f"unknown code_kb mode {mode!r}. "
            f"Supported: {', '.join(sorted(set(_MODE_HANDLERS)) | set(layered_modes))}.",
            extra={"mode": mode, "args": args},
        )
    try:
        return handler(store, args, step_id)
    except Exception as e:  # noqa: BLE001
        return _record_result(
            "code_kb", step_id, False,
            f"{type(e).__name__}: {e}\n\n{CODE_KB_SCHEMA_HINT}",
            extra={"mode": mode, "args": args},
        )
