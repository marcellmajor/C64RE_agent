"""Export the code KB to assembler-source form.

v1 ships a single exporter — **commented .asm** — compatible with the
common cross-assemblers in the C64 scene (acme / ca65 / 64tass). Output
shape:

    ; ============================================================
    ;  C64-RE annotated disassembly — <game>
    ;  (rebuilt from code_kb at <iso ts>)
    ; ============================================================

    !cpu 6510                     ; (acme/64tass — harmless to others)

    sprite_multiplex_irq:        ; [conf 0.78] sprite multiplexer (from Layer 1)
    $0810  A9 1B   LDA #$1B      ; setup VIC-II RSEL, raster IRQ
    ...

Labels are derived from:
    1. Layer-1 ``name_suggestion`` (when confidence ≥ 0.6)
    2. Layer-0 routine names from the original asm
    3. Auto-generated `sub_XXXX` / `loc_XXXX` fallbacks

Comments contain the layer-1 hypothesis prefix on the routine line, plus
inline hardware-pack hints on every register-mapped operand. Anything
with confidence < 0.5 is gated behind an explicit ``[low conf]`` tag.

The Ghidra annotation script exporter is a follow-up — leaving it for
v2 keeps the v1 surface area testable.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from code_kb import hardware_pack
from code_kb.store import CodeKnowledgeStore


_HEX_OPERAND_RE = re.compile(r"\$([0-9A-Fa-f]{2,4})")


def _comment_for_operand(op: str | None) -> str:
    """Annotate a hex operand with its hardware-pack role, if any."""
    if not op:
        return ""
    hits: list[str] = []
    for tok in _HEX_OPERAND_RE.findall(op):
        addr = int(tok, 16)
        e = hardware_pack.lookup_range(addr)
        if e and e.chip in {"VIC-II", "SID", "CIA1", "CIA2", "KERNAL", "BASIC"}:
            hits.append(f"{e.name} ({e.chip})")
    return "; " + " | ".join(hits) if hits else ""


def _resolve_label_for(
    addr: int,
    layer1_by_routine: dict[str, dict[str, Any]],
    routine_meta: dict[int, dict[str, Any]],
    code_labels: dict[int, list[str]],
) -> str | None:
    """Pick the best display label for a given address."""
    rt = routine_meta.get(addr)
    if rt:
        rid = rt.get("annotation_id") or ""
        l1 = layer1_by_routine.get(str(rid))
        if l1:
            sugg = l1.get("name_suggestion")
            conf = l1.get("confidence") or 0.0
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.0
            if sugg and conf >= 0.6:
                return str(sugg)
        if rt.get("name"):
            return str(rt["name"])
    names = code_labels.get(addr) or []
    if names:
        return names[0]
    return None


def export_commented_asm(
    store: CodeKnowledgeStore, *, game_name: str | None = None,
    min_confidence: float = 0.0,
) -> str:
    """Render the entire code KB as a single commented .asm document."""
    rows = store.query(
        "SELECT addr, bytes_hex, mnemonic, operand, source_file"
        "  FROM instructions ORDER BY addr",
        (),
    )
    if not rows:
        return "; (code KB has no instructions ingested)\n"

    routine_rows = store.query(
        "SELECT start_addr, end_addr, name, source_file, annotation_id"
        "  FROM code_routines ORDER BY start_addr",
        (),
    )
    routine_meta = {int(r["start_addr"]): r for r in routine_rows}

    label_rows = store.query(
        "SELECT addr, name FROM code_labels ORDER BY addr",
        (),
    )
    code_labels: dict[int, list[str]] = {}
    for r in label_rows:
        code_labels.setdefault(int(r["addr"]), []).append(str(r["name"]))

    layer1_rows = store.query(
        "SELECT id AS annotation_id, payload_json, confidence, flags_json"
        "  FROM annotations WHERE layer = 1 AND kind = ?"
        "  ORDER BY ts DESC",
        ("hypothesis",),
    )
    layer1_by_routine: dict[str, dict[str, Any]] = {}
    for r in layer1_rows:
        try:
            payload = json.loads(r.get("payload_json") or "{}")
        except json.JSONDecodeError:
            payload = {}
        rid = str(payload.get("routine_id") or "")
        if rid and rid not in layer1_by_routine:
            layer1_by_routine[rid] = {
                **payload,
                "confidence": r.get("confidence"),
                "flags": json.loads(r.get("flags_json") or "[]"),
            }

    out: list[str] = []
    out.append(
        f"; ============================================================"
    )
    out.append(
        f";  C64-RE annotated disassembly"
        + (f" — {game_name}" if game_name else "")
    )
    out.append(
        f";  rebuilt from code_kb at "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    out.append(
        f";  layers: deterministic Layer-0 + LLM Layer-1 hypotheses"
    )
    out.append(
        f"; ============================================================"
    )
    out.append("")
    out.append("    !cpu 6510")
    out.append("")

    prev_addr_end: int | None = None
    for r in rows:
        addr = int(r["addr"])
        bytes_hex = (r.get("bytes_hex") or "").lower()
        bytes_col = " ".join(
            bytes_hex[i : i + 2] for i in range(0, len(bytes_hex), 2)
        )
        mnem = (r.get("mnemonic") or "?").upper()
        op = (r.get("operand") or "").strip()
        size = max(1, len(bytes_hex) // 2)

        if prev_addr_end is not None and addr != prev_addr_end:
            out.append(
                f"    ; gap ${prev_addr_end:04X}-${addr - 1:04X} "
                f"({addr - prev_addr_end} bytes — likely data)"
            )

        # Routine header (label + hypothesis comment).
        rt = routine_meta.get(addr)
        if rt:
            label = _resolve_label_for(
                addr, layer1_by_routine, routine_meta, code_labels,
            )
            rid = str(rt.get("annotation_id") or "")
            l1 = layer1_by_routine.get(rid)
            if l1:
                conf = float(l1.get("confidence") or 0.0)
                hyp = (l1.get("text") or "").splitlines()[0][:120]
                tag = (
                    "low conf" if conf < 0.5
                    else "med conf" if conf < 0.75
                    else "high conf"
                )
                out.append(
                    f"\n; --- routine ${addr:04X}-${int(rt['end_addr']):04X} "
                    f"[{tag}={conf:.2f}] {hyp}"
                )
                idiom = l1.get("idiom_match")
                if idiom:
                    out.append(f"; idiom: {idiom}")
                hw = l1.get("hardware_touched") or []
                for line in hw[:6]:
                    out.append(f"; hw: {line}")
            else:
                out.append(
                    f"\n; --- routine ${addr:04X}-${int(rt['end_addr']):04X} "
                    f"(deterministic Layer-0 only)"
                )
            out.append(f"{label or f'sub_{addr:04x}'}:")
        else:
            # Local label?
            if addr in code_labels:
                for n in code_labels[addr]:
                    out.append(f"{n}:")

        comment = _comment_for_operand(op)
        line = f"    ${addr:04X}  {bytes_col:<10}  {mnem:<4} {op}"
        if comment:
            line = f"{line:<48} {comment}"
        out.append(line)

        prev_addr_end = addr + size

    out.append("")
    out.append(f"; ============================================================")
    out.append(f";  end — {len(rows)} instruction lines")
    out.append(f"; ============================================================")
    return "\n".join(out) + "\n"
