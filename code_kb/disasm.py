"""On-demand disassembly for newly-discovered code areas.

Two entry points:

    * `disasm_capstone(store, start, length=..., recursive=...)` — uses the
      existing `tools.c64_disasm` helpers against the dump bytes already
      cached in the `CodeKnowledgeStore`.

    * `disasm_vice(store, address, count)` — calls the running VICE
      monitor through `tools.vice_mcp`. Useful when:
        - the dump doesn't have the area mapped (banked memory, runtime
          decompression, copy-protection thunks)
        - you want to confirm static disassembly against live RAM
        - the planner has explicitly opted into dynamic verification.

Both helpers funnel their results through `code_kb.layer0
.build_from_disasm_window` so the new bytes immediately become
deterministic Layer-0 ground truth: instructions, xrefs, indirect-jump
sites, SMC suspects.
"""

from __future__ import annotations

import re
from typing import Any

from code_kb.layer0 import build_from_disasm_window, Layer0Stats
from code_kb.schema import EVT_DISASM_WINDOW
from code_kb.store import CodeKnowledgeStore


_VICE_LINE_RE = re.compile(
    r"""^\s*\.?C?:?\s*\$?(?P<addr>[0-9a-fA-F]{4})
        \s+(?P<bytes>(?:[0-9a-fA-F]{2}\s){0,3}[0-9a-fA-F]{2})?
        \s*(?P<mnem>[A-Za-z][A-Za-z]{1,3})
        (?:\s+(?P<op>.*))?
        \s*$
    """,
    re.VERBOSE,
)


def _normalize_capstone_insns(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert `tools.c64_disasm.recursive_disasm` output into our row shape."""
    out: list[dict[str, Any]] = []
    insns = rec.get("insns") or {}
    for k, info in insns.items():
        try:
            addr = int(str(k).lstrip("$"), 16) if not isinstance(k, int) else int(k)
        except ValueError:
            continue
        out.append({
            "addr": addr,
            "bytes": bytes(info.get("bytes") or b""),
            "mnemonic": str(info.get("mnemonic") or "").lower(),
            "op_str": info.get("op_str"),
        })
    return out


def disasm_capstone(
    store: CodeKnowledgeStore,
    *,
    start: int,
    length: int = 256,
    recursive: bool = False,
    max_insns: int = 600,
    note: str | None = None,
) -> dict[str, Any]:
    """Disassemble a slice of the dump and feed it back into Layer 0.

    Returns a dict suitable for inclusion in a tool_result data field
    (`{"text": "<listing>", "stats": {...}, "rows": [...]}` ).
    """
    if not store.has_dump():
        return {
            "ok": False,
            "error": (
                "code_kb has no dump bytes ingested — pass a `--dump` to "
                "the agent or call `code_kb.store.ingest_dump(...)` first."
            ),
            "start": start, "length": length,
        }
    from tools import c64_disasm  # lazy: avoids capstone import cost at module load

    mem = store.full_dump()
    if recursive:
        rec = c64_disasm.recursive_disasm(
            mem, start, max_insns=max_insns, seed_vectors=False,
        )
        rows = _normalize_capstone_insns(rec)
        listing = rec.get("listing", "")
        stats_dict = rec.get("stats", {})
    else:
        text = c64_disasm.linear_disasm(
            mem, start, length=length, max_lines=length // 2,
        )
        # Re-parse the textual listing to extract structured rows.
        rows = []
        for line in text.splitlines():
            m = _VICE_LINE_RE.match(line)
            if not m:
                continue
            try:
                addr = int(m.group("addr"), 16)
            except ValueError:
                continue
            b_field = (m.group("bytes") or "").strip().replace(" ", "")
            try:
                b = bytes.fromhex(b_field) if b_field else b""
            except ValueError:
                b = b""
            rows.append({
                "addr": addr,
                "bytes": b,
                "mnemonic": (m.group("mnem") or "").lower(),
                "op_str": (m.group("op") or "").strip(),
            })
        listing = text
        stats_dict = {"instructions": len(rows)}

    l0_stats: Layer0Stats = build_from_disasm_window(
        rows, store, source_file=note or f"capstone:${start:04X}+{length}",
    )

    store.append_event(
        EVT_DISASM_WINDOW, "code_kb.disasm",
        {
            "tool": "capstone",
            "start": int(start) & 0xFFFF,
            "length": int(length),
            "recursive": bool(recursive),
            "instructions": l0_stats.instructions,
            "xrefs": l0_stats.xrefs,
            "smc_sites": l0_stats.smc_sites,
            "note": note,
        },
    )
    return {
        "ok": True,
        "start": int(start) & 0xFFFF,
        "length": int(length),
        "recursive": bool(recursive),
        "rows": [
            {
                "addr": r["addr"],
                "bytes_hex": (r["bytes"] or b"").hex() if isinstance(r.get("bytes"), (bytes, bytearray)) else "",
                "mnemonic": r["mnemonic"],
                "operand": r.get("op_str") or r.get("operand"),
            }
            for r in rows
        ],
        "listing": listing,
        "stats": {
            "instructions": l0_stats.instructions,
            "xrefs": l0_stats.xrefs,
            "indirect_sites": l0_stats.indirect_sites,
            "smc_sites": l0_stats.smc_sites,
            "capstone_stats": stats_dict,
        },
    }


def _parse_vice_text(raw: str) -> list[dict[str, Any]]:
    """Parse a vice-mcp `vice.disassemble` text response into row dicts."""
    rows: list[dict[str, Any]] = []
    if not raw:
        return rows
    for line in raw.splitlines():
        m = _VICE_LINE_RE.match(line)
        if not m:
            continue
        try:
            addr = int(m.group("addr"), 16)
        except ValueError:
            continue
        b_field = (m.group("bytes") or "").strip().replace(" ", "")
        try:
            b = bytes.fromhex(b_field) if b_field else b""
        except ValueError:
            b = b""
        rows.append({
            "addr": addr,
            "bytes": b,
            "mnemonic": (m.group("mnem") or "").lower(),
            "op_str": (m.group("op") or "").strip(),
        })
    return rows


def disasm_vice(
    store: CodeKnowledgeStore,
    *,
    address: int,
    count: int = 32,
    note: str | None = None,
) -> dict[str, Any]:
    """Disassemble live RAM via vice-mcp; results feed Layer 0.

    Returns the same `{ok, rows, listing, stats}` shape as the capstone
    helper. If `VICE_MCP_URL` isn't configured or the tool call fails
    we surface a structured error rather than raising.
    """
    import os
    if not os.getenv("VICE_MCP_URL", "").strip():
        return {
            "ok": False,
            "error": "VICE_MCP_URL not set — vice disasm unavailable",
            "address": int(address) & 0xFFFF, "count": int(count),
        }
    try:
        from tools import vice_mcp
    except ModuleNotFoundError as e:
        return {
            "ok": False,
            "error": f"vice_mcp module unavailable: {e}",
            "address": int(address) & 0xFFFF, "count": int(count),
        }

    addr = int(address) & 0xFFFF
    cnt = max(1, min(100, int(count)))
    try:
        out = vice_mcp.call_tool(
            "vice.disassemble",
            {"address": f"${addr:04X}", "count": cnt, "show_symbols": True},
        )
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "address": addr, "count": cnt,
        }

    if out.get("is_error"):
        return {
            "ok": False,
            "error": str(out.get("data") or out.get("raw_content")),
            "address": addr, "count": cnt,
        }

    raw = out.get("data")
    if isinstance(raw, dict):
        text = raw.get("text") or "\n".join(
            (
                ln.get("text") or ln.get("instruction") or ""
                if isinstance(ln, dict) else str(ln)
            )
            for ln in (raw.get("lines") or [])
        )
    elif isinstance(raw, list):
        text = "\n".join(str(x) for x in raw)
    else:
        text = str(raw)

    rows = _parse_vice_text(text)
    l0_stats = build_from_disasm_window(
        rows, store, source_file=note or f"vice:${addr:04X}+{cnt}",
    )

    store.append_event(
        EVT_DISASM_WINDOW, "code_kb.disasm",
        {
            "tool": "vice",
            "address": addr, "count": cnt,
            "instructions": l0_stats.instructions,
            "xrefs": l0_stats.xrefs,
            "smc_sites": l0_stats.smc_sites,
            "note": note,
        },
    )
    return {
        "ok": True,
        "address": addr, "count": cnt,
        "listing": text,
        "rows": [
            {
                "addr": r["addr"],
                "bytes_hex": (r["bytes"] or b"").hex(),
                "mnemonic": r["mnemonic"],
                "operand": r.get("op_str") or r.get("operand"),
            }
            for r in rows
        ],
        "stats": {
            "instructions": l0_stats.instructions,
            "xrefs": l0_stats.xrefs,
            "indirect_sites": l0_stats.indirect_sites,
            "smc_sites": l0_stats.smc_sites,
        },
    }
