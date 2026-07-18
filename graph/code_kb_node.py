"""LangGraph node implementing the `code_kb` planner tool.

Modes the planner can request via ``args["mode"]``:

    - **stats**       : counts (routines, xrefs, SMC, instructions, …)
    - **routines**    : list code_routines (optional `like` for name filter)
    - **routine**     : full window for a single routine (by `start` addr)
    - **xrefs_to**    : who calls/jumps to `addr`?
    - **xrefs_from**  : where does `addr` (or routine `start..end`) jump?
    - **smc**         : list SMC suspects
    - **search**      : substring search over the ingested partial-asm
                       documents (whole-file, not just labels)
    - **disasm**      : disassemble a fresh range via capstone (default)
                       or vice (`engine: vice`); results feed Layer 0
    - **annotate**    : run Layer-1 LLM annotation on one routine
    - **export**      : commented .asm dump of the entire code KB
    - **hardware**    : return the curated hardware pack as text
    - **schema**      : print the SQLite schema cheat-sheet
    - **sql**         : raw read-only SQL (same safety rules as parent KB)

The node always answers with a `_record_result` payload so the planner
loop can route the response just like any other tool.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from code_kb import (
    DEEP_RETRO_RE_PREAMBLE,
    annotation_from_layer1_json,
    build_window,
    disasm_capstone,
    disasm_vice,
    export_commented_asm,
    export_vice_symbols,
    fetch_routines,
    get_code_store,
    render_user_prompt,
)
from code_kb import hardware_pack
from code_kb.schema import EVT_LAYER_RUN
from graph.state import C64State

# We deliberately reach into nodes for `_record_result` / `_step_for` /
# `_hex_to_int` — re-implementing them here would drift from the contract
# the rest of the graph relies on.
from graph.nodes import _hex_to_int, _record_result, _step_for, _try_parse_json


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


def _mode_routine(store, args, step_id):
    start_raw = args.get("start") or args.get("addr") or args.get("address")
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
    rows = store.query(
        "SELECT start_addr, end_addr, name, source_file, entries_json,"
        "       exits_json, annotation_id"
        "  FROM code_routines WHERE start_addr = ? LIMIT 1",
        (start,),
    )
    if not rows:
        return _record_result(
            "code_kb", step_id, False,
            f"no routine found with start_addr=${start:04X}. Try mode='routines' first.",
            extra={"mode": "routine"},
        )
    win = build_window(store, routine_row=rows[0])
    text = render_user_prompt(win)
    return _record_result(
        "code_kb", step_id, True, text,
        extra={
            "mode": "routine",
            "start_addr": win.start_addr, "end_addr": win.end_addr,
            "name": win.name, "callers": len(win.callers),
            "callees": len(win.callees), "smc_sites": len(win.smc_sites),
        },
    )


def _mode_xrefs_to(store, args, step_id):
    addr = _hex_to_int(args.get("addr") or args.get("dst") or 0, 0)
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
    src = _hex_to_int(args.get("addr") or args.get("src") or 0, 0)
    end = _hex_to_int(args.get("end"), src)
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
    addr = _hex_to_int(args.get("addr") or args.get("dst") or 0, 0)
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
    addr = _hex_to_int(args.get("addr") or args.get("dst") or 0, 0)
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
        lo = _hex_to_int(args.get("lo"), 0xD000)
        hi = _hex_to_int(args.get("hi"), 0xDFFF)
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
            "snippet": snippet,
        })
    if not out:
        text = f"No asm_doc hits for {q!r}."
    else:
        lines = [f"{len(out)} hit(s) for {q!r}:"]
        for h in out:
            lines.append(f"- {h['path']} (offset={h['match_at']})")
            lines.append(f"  …{h['snippet']}…")
        text = "\n".join(lines)
    return _record_result(
        "code_kb", step_id, True, text,
        extra={"mode": "search", "q": q, "hits": out},
    )


def _mode_disasm(store, args, step_id):
    engine = str(args.get("engine") or "capstone").lower()
    if engine == "vice":
        addr = _hex_to_int(args.get("address") or args.get("addr") or 0, 0)
        count = int(args.get("count", 32))
        out = disasm_vice(store, address=addr, count=count, note=args.get("note"))
    else:
        start = _hex_to_int(args.get("start") or args.get("addr") or 0, 0)
        length = int(args.get("length", 256))
        recursive = bool(args.get("recursive", False))
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
            extra={"mode": "sql", "sql": sql, "rejection": "forbidden_statement"},
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
            text = json.dumps(rows[: int(args.get("limit", 100))],
                              indent=2, default=str)
            extra = {"mode": "sql", "sql": attempt_sql, "row_count": len(rows)}
            if label != "original":
                extra["sql_rewrites_applied"] = applied
            return _record_result(
                "code_kb", step_id, True, text, extra=extra,
            )
        except Exception as e:  # noqa: BLE001
            last_err = e

    return _record_result(
        "code_kb", step_id, False,
        f"{type(last_err).__name__ if last_err else 'Error'}: {last_err}\n\n"
        + CODE_KB_SCHEMA_HINT,
        extra={"mode": "sql", "sql": sql, "schema": CODE_KB_SCHEMA_HINT},
    )


# --------------------------------------------------------------------------- #
# Layer-1 annotate mode (LLM call)
# --------------------------------------------------------------------------- #


def _mode_annotate(state, store, args, step_id):
    """Run Layer-1 LLM annotation on a routine.

    Args (planner):
      start: "$XXXX"          (required) — routine start address
      role:  "analyst"        (optional) — config/llm.json agent role
      backup_roles: ["critic"]  (optional)
    """
    start_raw = args.get("start") or args.get("addr") or args.get("address")
    if start_raw is None:
        return _record_result(
            "code_kb", step_id, False,
            "code_kb mode='annotate' requires args.start (routine address).",
            extra={"mode": "annotate"},
        )
    start = _hex_to_int(start_raw, -1)
    if start < 0:
        return _record_result(
            "code_kb", step_id, False,
            f"could not parse start={start_raw!r}",
            extra={"mode": "annotate"},
        )
    rows = store.query(
        "SELECT start_addr, end_addr, name, source_file, entries_json,"
        "       exits_json, annotation_id"
        "  FROM code_routines WHERE start_addr = ? LIMIT 1",
        (start,),
    )
    auto_disasm = bool(args.get("auto_disasm_if_missing", True))
    auto_disasm_info: dict[str, Any] = {"attempted": False}
    if not rows:
        if not auto_disasm:
            return _record_result(
                "code_kb", step_id, False,
                f"no routine at ${start:04X}",
                extra={"mode": "annotate", "auto_disasm": auto_disasm_info},
            )

        engine = str(args.get("disasm_engine") or "capstone").lower()
        auto_disasm_info["attempted"] = True
        auto_disasm_info["engine"] = engine
        if engine == "vice":
            d_out = disasm_vice(
                store,
                address=start,
                count=int(args.get("disasm_count") or 64),
                note=f"annotate-autodisasm:${start:04X}",
            )
        else:
            d_out = disasm_capstone(
                store,
                start=start,
                length=int(args.get("disasm_length") or 768),
                recursive=bool(args.get("disasm_recursive", False)),
                max_insns=int(args.get("disasm_max_insns") or 800),
                note=f"annotate-autodisasm:${start:04X}",
            )
        auto_disasm_info["ok"] = bool(d_out.get("ok"))
        auto_disasm_info["stats"] = d_out.get("stats")
        if not d_out.get("ok"):
            return _record_result(
                "code_kb", step_id, False,
                f"no routine at ${start:04X}; auto-disasm failed: {d_out.get('error')}",
                extra={"mode": "annotate", "auto_disasm": auto_disasm_info},
            )

        rows = store.query(
            "SELECT start_addr, end_addr, name, source_file, entries_json,"
            "       exits_json, annotation_id"
            "  FROM code_routines WHERE start_addr = ? LIMIT 1",
            (start,),
        )
        if not rows:
            return _record_result(
                "code_kb", step_id, False,
                f"no routine at ${start:04X} even after auto-disasm",
                extra={"mode": "annotate", "auto_disasm": auto_disasm_info},
            )
    role = str(args.get("role") or "analyst")
    backup_roles = list(args.get("backup_roles") or ["critic", "synthesizer"])
    window = build_window(store, routine_row=rows[0])

    # If the routine exists in code_routines but has NO instructions indexed
    # (e.g. it was mirrored from the LLM synthesizer without a Capstone scan),
    # auto-disasm the address range now so the LLM gets a real listing.
    if not window.listing and auto_disasm:
        engine = str(args.get("disasm_engine") or "capstone").lower()
        routine_len = max(1, window.end_addr - window.start_addr + 1)
        auto_disasm_info["attempted"] = True
        auto_disasm_info["engine"] = engine
        auto_disasm_info["reason"] = "empty_listing"
        if engine == "vice":
            d_out = disasm_vice(
                store,
                address=window.start_addr,
                count=int(args.get("disasm_count") or 64),
                note=f"annotate-autodisasm:${window.start_addr:04X}",
            )
        else:
            d_out = disasm_capstone(
                store,
                start=window.start_addr,
                length=int(args.get("disasm_length") or routine_len),
                recursive=bool(args.get("disasm_recursive", False)),
                max_insns=int(args.get("disasm_max_insns") or 800),
                note=f"annotate-autodisasm:${window.start_addr:04X}",
            )
        auto_disasm_info["ok"] = bool(d_out.get("ok"))
        auto_disasm_info["stats"] = d_out.get("stats")
        if d_out.get("ok"):
            # Re-fetch the routine row (auto-disasm may have updated it)
            refreshed = store.query(
                "SELECT start_addr, end_addr, name, source_file, entries_json,"
                "       exits_json, annotation_id"
                "  FROM code_routines WHERE start_addr = ? LIMIT 1",
                (window.start_addr,),
            )
            window = build_window(store, routine_row=(refreshed[0] if refreshed else rows[0]))

    user_prompt = render_user_prompt(window, question=state.get("question"))
    system_prompt = (
        DEEP_RETRO_RE_PREAMBLE
        + "\n---\n## Hardware reference (keep nearby — do not paraphrase)\n"
        + hardware_pack.pack_text(max_chars=4_000)
    )

    parsed, used_role, last_err = _invoke_layer1(
        role, backup_roles, system_prompt, user_prompt,
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
    ann_id = store.append_annotation(ann, source="code_kb_node.layer1")
    store.append_event(
        EVT_LAYER_RUN, "code_kb_node",
        {"layer": 1, "start_addr": start, "role": used_role, "annotation_id": ann_id},
    )

    text = (
        f"Layer-1 annotation for ${window.start_addr:04X}-${window.end_addr:04X} "
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
            "annotation_id": ann_id,
            "producer": ann.producer, "confidence": ann.confidence,
            "idiom_match": ann.payload.get("idiom_match"),
            "name_suggestion": ann.payload.get("name_suggestion"),
            "auto_disasm": auto_disasm_info,
        },
    )


def _invoke_layer1(
    role: str, backup_roles: list[str],
    system_prompt: str, user_prompt: str,
) -> tuple[dict | None, str | None, str | None]:
    """Try the LLM once per role; return (parsed_json, role_used, last_err)."""
    from graph import usage as llm_usage
    from graph.llm import get_llm
    from graph.nodes import _flatten_lc_ai_message_content

    last_err = None
    for r in [role, *backup_roles]:
        model_name = None
        try:
            llm = get_llm(r)
            model_name = (
                getattr(llm, "model_name", None) or getattr(llm, "model", None)
            )
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
    "xrefs_to":      _mode_xrefs_to,
    "xrefs_from":    _mode_xrefs_from,
    "smc":           _mode_smc,
    "writes_to":     _mode_writes_to,
    "refs_to":       _mode_refs_to,
    "hardware_refs": _mode_hardware_refs,
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

    if mode == "annotate":
        try:
            return _mode_annotate(state, store, args, step_id)
        except Exception as e:  # noqa: BLE001
            return _record_result(
                "code_kb", step_id, False,
                f"{type(e).__name__}: {e}",
                extra={"mode": "annotate", "args": args},
            )

    handler = _MODE_HANDLERS.get(mode)
    if handler is None:
        return _record_result(
            "code_kb", step_id, False,
            f"unknown code_kb mode {mode!r}. "
            f"Supported: {', '.join(sorted(set(_MODE_HANDLERS)) | {'annotate'})}.",
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
