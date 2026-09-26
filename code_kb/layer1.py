"""Layer 1 — per-routine semantic annotation (LLM, narrow window).

The Layer-1 pipeline takes one routine at a time, builds a tight,
evidence-rich context window (instructions + xrefs in/out + the
hardware pack), and asks the configured analyst-style LLM to propose:

    {
        "address_range":      "$XXXX-$YYYY",
        "name_suggestion":    "snake_case_id",
        "hypothesis":         "<what this block appears to do>",
        "idiom_match":        "<delay loop|copy loop|sprite multiplexer|"
                              "decompressor|raster wait|joystick read|"
                              "music player tick|null>",
        "hardware_touched":   ["VIC-II $D020 EXTCOL", ...],
        "confidence":         0..1,
        "evidence":           ["<addr>", "<addr>", ...]
    }

The actual LLM invocation lives in `graph/code_kb_node.py` so we reuse
`graph.llm.get_llm("analyst")` / its backups; this file only assembles
the prompt and parses the response. That keeps `code_kb` standalone and
testable without LangChain in the import path.

The DeepRetroRE persona text from the second build prompt is bolted on
as the system preamble so the LLM operates under the right framing
without contaminating the parent agent's master preamble.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from code_kb import hardware_pack
from code_kb.schema import ANN_HYPOTHESIS
from code_kb.store import Annotation, CodeKnowledgeStore


# --------------------------------------------------------------------------- #
# DeepRetroRE persona — used only by Layer-1/Layer-2 calls.
# --------------------------------------------------------------------------- #


DEEP_RETRO_RE_PREAMBLE = """\
You are the **DeepRetroRE Agent** — a senior reverse engineer of 8-bit
games with 15+ years of Commodore 64 experience.

Operate under these *non-negotiable* principles for every reply:

1. Respect 8-bit reality. The code was hand-optimised under 64 KB,
   raster-cycle, and space constraints. Self-modifying code,
   multiple-entry-point routines, and indirect dispatch are *normal*,
   not bugs.
2. The 5-pass deep-analysis methodology applies even when only a single
   routine is in scope:
     Pass 1 – Structural   : entry points, call graph, fall-through
     Pass 2 – Semantic     : what does this block actually do
     Pass 3 – Hardware     : zero-page roles, VIC-II/SID/CIA touches
     Pass 4 – Motivation   : *why* it was written this way (e.g.
                              "self-modifying LDA to save 6 cycles in
                              raster time")
     Pass 5 – Verification : flag uncertainties, ask for breakpoints
3. Cite every address you mention in `$XXXX` hex form, exactly as it
   appears in the listing.
4. Never invent disassembly. If a fact requires bytes you have not
   been shown, say so explicitly under `evidence` and lower confidence.
5. Tag *motivation* claims as such — they are guesses unless the
   evidence chain proves them.
6. When you spot a clever 8-bit trick, name it (delay loop, sprite
   multiplexer, raster IRQ, table-driven dispatch, …).
7. Output JSON only when a JSON schema is requested. No fences, no
   commentary outside the schema.

You are part of a larger multi-agent system. Your job here is *one
window of ~1-2 KB of code*. Do not invent global hypotheses; another
layer (Layer 2) handles cross-routine groupings.
"""


# --------------------------------------------------------------------------- #
# Window builder
# --------------------------------------------------------------------------- #


@dataclass
class RoutineWindow:
    routine_id: str          # annotation id of the ANN_ROUTINE row
    start_addr: int
    end_addr: int
    name: str | None
    source_file: str | None
    listing: str             # rendered $XXXX  bytes  mnemonic  operand
    callers: list[dict[str, Any]]
    callees: list[dict[str, Any]]
    smc_sites: list[dict[str, Any]]
    hardware_hits: list[str]
    entries: list[int]
    exits: list[int]


def fetch_routines(
    store: CodeKnowledgeStore, *, limit: int = 25, offset: int = 0,
    name_like: str | None = None,
) -> list[dict[str, Any]]:
    if name_like:
        return store.query(
            "SELECT start_addr, end_addr, name, source_file, entries_json,"
            "       exits_json, size_bytes, annotation_id"
            "  FROM code_routines"
            " WHERE name LIKE ?"
            " ORDER BY start_addr LIMIT ? OFFSET ?",
            (f"%{name_like}%", limit, offset),
        )
    return store.query(
        "SELECT start_addr, end_addr, name, source_file, entries_json,"
        "       exits_json, size_bytes, annotation_id"
        "  FROM code_routines ORDER BY start_addr LIMIT ? OFFSET ?",
        (limit, offset),
    )


def build_window(
    store: CodeKnowledgeStore, *, routine_row: dict[str, Any],
    max_listing_lines: int = 240,
    focus_addr: int | None = None,
) -> RoutineWindow:
    """Assemble a Layer-1 context window for one routine."""
    start = int(routine_row["start_addr"])
    end = int(routine_row["end_addr"])
    rid = str(routine_row.get("annotation_id") or "?")

    # Fetch instructions for the routine.
    rows = store.query(
        "SELECT addr, bytes_hex, mnemonic, operand, size_bytes"
        "  FROM instructions"
        " WHERE addr BETWEEN ? AND ?"
        " ORDER BY addr",
        (start, end),
    )
    listing_lines: list[str] = []
    first = 0
    if focus_addr is not None and len(rows) > max_listing_lines:
        focus_index = max(
            (i for i, row in enumerate(rows) if row["addr"] <= focus_addr),
            default=0,
        )
        first = min(max(0, focus_index - max_listing_lines // 2),
                    len(rows) - max_listing_lines)
    if first:
        listing_lines.append(f"; ... {first} earlier instructions omitted")
    hw_hits: list[str] = []
    for r in rows[first:first + max_listing_lines]:
        addr = int(r["addr"])
        bytes_hex = (r.get("bytes_hex") or "").lower()
        bytes_col = " ".join(
            bytes_hex[i : i + 2] for i in range(0, len(bytes_hex), 2)
        )
        mnem = (r.get("mnemonic") or "?").upper()
        op = (r.get("operand") or "").strip()
        listing_lines.append(
            f"${addr:04X}  {bytes_col:<10}  {mnem:<4} {op}".rstrip()
        )
        # Cheap hardware-pack flagging on operand text.
        for tok in re.findall(r"\$([0-9A-Fa-f]{4})", op):
            hw = hardware_pack.lookup_range(int(tok, 16))
            if hw and hw.chip in {"VIC-II", "SID", "CIA1", "CIA2", "KERNAL"}:
                hw_hits.append(
                    f"${int(tok, 16):04X} {hw.name} ({hw.chip}) — {hw.role}"
                )
    if len(rows) > first + max_listing_lines:
        listing_lines.append(
            f"; ... {len(rows) - first - max_listing_lines} more instructions truncated"
        )

    callers = store.query(
        "SELECT src_addr, kind FROM code_xrefs"
        " WHERE dst_addr = ?"
        " ORDER BY src_addr",
        (start,),
    )
    callees_rows = store.query(
        "SELECT dst_addr, kind FROM code_xrefs"
        " WHERE src_addr BETWEEN ? AND ? AND dst_addr IS NOT NULL"
        " ORDER BY src_addr",
        (start, end),
    )
    smc = store.query(
        "SELECT src_addr, dst_addr, mnemonic, operand, smc_kind"
        "  FROM code_smc_sites"
        " WHERE src_addr BETWEEN ? AND ?"
        " ORDER BY src_addr",
        (start, end),
    )

    try:
        entries = json.loads(routine_row.get("entries_json") or "[]") or [start]
    except (TypeError, json.JSONDecodeError):
        entries = [start]
    try:
        exits = json.loads(routine_row.get("exits_json") or "[]") or []
    except (TypeError, json.JSONDecodeError):
        exits = []

    # Dedup hardware hits by (addr,name) while preserving order.
    seen_hw = set()
    hw_dedup: list[str] = []
    for h in hw_hits:
        if h not in seen_hw:
            seen_hw.add(h)
            hw_dedup.append(h)

    return RoutineWindow(
        routine_id=rid,
        start_addr=start, end_addr=end,
        name=routine_row.get("name"),
        source_file=routine_row.get("source_file"),
        listing="\n".join(listing_lines),
        callers=[{"src": int(c["src_addr"]), "kind": c["kind"]} for c in callers],
        callees=[
            {"dst": int(c["dst_addr"]), "kind": c["kind"]}
            for c in callees_rows
            if c.get("dst_addr") is not None
        ],
        smc_sites=[
            {
                "src": int(s["src_addr"]),
                "dst": int(s["dst_addr"]) if s.get("dst_addr") is not None else None,
                "mnemonic": s.get("mnemonic"),
                "operand": s.get("operand"),
                "kind": s.get("smc_kind"),
            }
            for s in smc
        ],
        hardware_hits=hw_dedup,
        entries=[int(a) for a in entries],
        exits=[int(a) for a in exits],
    )


def render_user_prompt(window: RoutineWindow, *, question: str | None = None) -> str:
    """Build the user-prompt text the analyst-style LLM should consume."""
    lines: list[str] = []

    if question:
        lines.append(f"Parent agent question (background only): {question}\n")

    lines.append(
        f"## Routine {window.name or '(unnamed)'}  "
        f"(${window.start_addr:04X}-${window.end_addr:04X})"
    )
    if window.source_file:
        lines.append(f"_source: {window.source_file}_")
    lines.append("")
    lines.append("### Entry points")
    lines.extend(f"- ${a:04X}" for a in window.entries) or lines.append("- (none)")
    lines.append("")
    lines.append("### Exits")
    if window.exits:
        lines.extend(f"- ${a:04X}" for a in window.exits)
    else:
        lines.append("- (none observed — fall-through?)")
    lines.append("")

    lines.append("### Listing")
    lines.append("```")
    lines.append(window.listing or "(empty)")
    lines.append("```")
    lines.append("")

    lines.append("### Cross-references — callers (whose JSR/JMP/branch lands here)")
    if window.callers:
        for c in window.callers[:32]:
            lines.append(f"- ${c['src']:04X}  ({c['kind']})")
        if len(window.callers) > 32:
            lines.append(f"- ... {len(window.callers) - 32} more")
    else:
        lines.append("- (none observed in KB)")

    lines.append("")
    lines.append("### Cross-references — callees (calls/jumps leaving this routine)")
    if window.callees:
        for c in window.callees[:32]:
            lines.append(f"- → ${c['dst']:04X}  ({c['kind']})")
        if len(window.callees) > 32:
            lines.append(f"- ... {len(window.callees) - 32} more")
    else:
        lines.append("- (none observed)")

    if window.smc_sites:
        lines.append("")
        lines.append("### Self-modifying-code suspects in this routine")
        for s in window.smc_sites[:16]:
            dst = f"${s['dst']:04X}" if s.get("dst") is not None else "(unknown)"
            lines.append(
                f"- ${s['src']:04X}  {s.get('mnemonic') or '?'} "
                f"{s.get('operand') or ''} → {dst}  ({s.get('kind')})"
            )

    if window.hardware_hits:
        lines.append("")
        lines.append("### Hardware addresses referenced in this listing")
        for h in window.hardware_hits[:32]:
            lines.append(f"- {h}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Reply with JSON ONLY, exactly:")
    lines.append("")
    lines.append(_LAYER1_SCHEMA)
    return "\n".join(lines)


_LAYER1_SCHEMA = """\
{
  "address_range":    "$XXXX-$YYYY",
  "name_suggestion":  "snake_case_identifier",
  "hypothesis":       "<one paragraph: what this routine appears to do>",
  "idiom_match":      "delay_loop | copy_loop | sprite_multiplexer | raster_wait | joystick_read | music_player_tick | decompressor | dispatch_table | bcd_score | string_print | null",
  "hardware_touched": ["$D020 EXTCOL (VIC-II) — border colour", ...],
  "motivation":       "<why a 1985 programmer would write it this way — TAGGED AS A GUESS unless the evidence proves it>",
  "confidence":       0.0,
  "evidence":         ["$XXXX", "$YYYY", "<short reason>"]
}
"""


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def annotation_from_layer1_json(
    parsed: dict[str, Any],
    window: RoutineWindow,
    *,
    producer: str,
) -> Annotation:
    """Validate + normalise a Layer-1 JSON response into an Annotation row."""
    confidence = parsed.get("confidence")
    try:
        c = float(confidence) if confidence is not None else 0.5
    except (TypeError, ValueError):
        c = 0.5
    c = max(0.0, min(1.0, c))

    text = (parsed.get("hypothesis") or "").strip()
    if not text:
        text = "(no hypothesis emitted)"

    motiv = parsed.get("motivation")
    if motiv:
        text = f"{text}\n\n_motivation:_ {motiv}"

    name_suggestion = parsed.get("name_suggestion")
    idiom = parsed.get("idiom_match")
    if isinstance(idiom, str) and idiom.strip().lower() in {"null", "none", ""}:
        idiom = None

    hardware_touched = parsed.get("hardware_touched") or []
    if not isinstance(hardware_touched, list):
        hardware_touched = [str(hardware_touched)]

    evidence = parsed.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]

    return Annotation(
        layer=1,
        kind=ANN_HYPOTHESIS,
        start_addr=window.start_addr,
        end_addr=window.end_addr,
        producer=producer,
        confidence=c,
        payload={
            "text": text,
            "name_suggestion": name_suggestion,
            "idiom_match": idiom,
            "hardware_touched": [str(h) for h in hardware_touched],
            "routine_id": window.routine_id,
            "source_file": window.source_file,
        },
        evidence=[str(e) for e in evidence],
        flags=[],
    )
