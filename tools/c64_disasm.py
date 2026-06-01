"""C64 6502 static-analysis helpers.

Distilled from a recursive-disassembler reference (see `capstone_sample.py`)
and exposed as composable helpers the agent can pick from per planner step:

  - linear_disasm        : block disassembly with bytes column + label hints
  - recursive_disasm     : control-flow following from a seed entry point
  - find_loops           : heuristic main-game-loop candidate scoring
  - find_entry           : BASIC SYS bootstrap detection / entry validation
  - list_vectors         : HW + RAM vectors
  - detect_polymorphic   : self-modifying-code suspect scan

All helpers operate on a 64 KB (or smaller, zero-padded) memory image.
None of them mutate state; they return JSON-friendly dicts/strings.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Optional

import capstone


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

HW_VECTORS = {
    0xFFFA: "nmi_vec",
    0xFFFC: "reset_vec",
    0xFFFE: "irq_vec",
}

C64_RAM_VECTORS = {
    0x0314: "c64_irq_vec",
    0x0316: "c64_brk_vec",
    0x0318: "c64_nmi_vec",
}

BRANCH_MNEMONICS = frozenset({"bpl", "bmi", "bvc", "bvs",
                              "bcc", "bcs", "bne", "beq", "bra"})
JUMP_MNEMONICS = frozenset({"jmp"})
CALL_MNEMONICS = frozenset({"jsr"})
RETURN_MNEMONICS = frozenset({"rts", "rti"})
STORE_MNEMONICS = frozenset({"sta", "stx", "sty", "stz"})
HALT_MNEMONICS = frozenset({"brk"})

_BARE_ADDR_RE = re.compile(r"^(0x[0-9a-fA-F]{1,4})$")
_INDIR_ADDR_RE = re.compile(r"^\(0x([0-9a-fA-F]{1,4})\)$")
_STORE_ADDR_RE = re.compile(r"^(0x[0-9a-fA-F]{1,4})(?:,\s*[xy])?$")
_HEX_RE = re.compile(r"0x([0-9a-fA-F]+)")

# Common I/O ranges (used by loop heuristics)
_VIC_RANGE = range(0xD000, 0xD400)
_SID_RANGE = range(0xD400, 0xD800)
_RASTER_ADDRS = frozenset({0xD011, 0xD012})
_JOY_ADDRS = frozenset({0xDC00, 0xDC01, 0xDD00, 0xDD01})


# --------------------------------------------------------------------------- #
# Operand parsing helpers (Capstone emits 0xNN syntax)
# --------------------------------------------------------------------------- #

def parse_direct_target(op_str: str) -> Optional[int]:
    m = _BARE_ADDR_RE.match(op_str.strip())
    return int(m.group(1), 16) if m else None


def parse_indirect_vector(op_str: str) -> Optional[int]:
    m = _INDIR_ADDR_RE.match(op_str.strip())
    return int(m.group(1), 16) if m else None


def parse_store_target(op_str: str) -> Optional[int]:
    m = _STORE_ADDR_RE.match(op_str.strip())
    if m:
        val = int(m.group(1), 16)
        return val if val > 0xFF else None
    return None


def to_c64(text: str) -> str:
    """Rewrite Capstone's `0xNN` to canonical C64 `$NN`."""
    return _HEX_RE.sub(lambda m: "$" + m.group(1).upper(), text)


# --------------------------------------------------------------------------- #
# Memory helpers
# --------------------------------------------------------------------------- #

def read_word(mem: bytes, addr: int) -> Optional[int]:
    if 0 <= addr <= 0xFFFE and addr + 1 < len(mem):
        return mem[addr] | (mem[addr + 1] << 8)
    return None


def in_bounds(mem: bytes, addr: int) -> bool:
    return 0 <= addr < len(mem)


def _make_cs() -> "capstone.Cs":
    cs = capstone.Cs(capstone.CS_ARCH_MOS65XX, capstone.CS_MODE_MOS65XX_6502)
    cs.detail = False
    return cs


# --------------------------------------------------------------------------- #
# Linear disassembly (block-mode)
# --------------------------------------------------------------------------- #

def linear_disasm(
    mem: bytes,
    start: int,
    length: int = 256,
    max_lines: int = 64,
    show_bytes: bool = True,
) -> str:
    """Pretty linear disassembly from `start`, formatted in C64 style."""
    if not in_bounds(mem, start):
        raise ValueError(f"start ${start:04X} out of range (mem={len(mem)} bytes)")

    end = min(start + length, len(mem))
    raw = mem[start:end]
    if not raw:
        raise ValueError(f"empty byte slice at ${start:04X}+{length}")

    cs = _make_cs()
    lines: list[str] = []
    for ins in cs.disasm(raw, start):
        bytes_col = " ".join(f"{b:02X}" for b in ins.bytes)
        op = to_c64(ins.op_str)
        if show_bytes:
            lines.append(f"${ins.address:04X}  {bytes_col:<10}  "
                         f"{ins.mnemonic:<4} {op}".rstrip())
        else:
            lines.append(f"${ins.address:04X}  {ins.mnemonic:<4} {op}".rstrip())

    if not lines:
        raise RuntimeError(
            f"capstone produced 0 instructions for {len(raw)} bytes at ${start:04X}"
        )

    text = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        text += f"\n... ({len(lines) - max_lines} more)"
    return text


# --------------------------------------------------------------------------- #
# Recursive control-flow disassembly
# --------------------------------------------------------------------------- #

def recursive_disasm(
    mem: bytes,
    entry: int,
    max_insns: int = 600,
    extra_seeds: Optional[list[tuple[int, str]]] = None,
    seed_vectors: bool = True,
) -> dict[str, Any]:
    """Recursive disassembly following JSR/JMP/branches from `entry`.

    Returns dict with: insns, labels, subroutines, remarks, paths_explored,
    listing (already C64-formatted text).
    """
    if not in_bounds(mem, entry):
        raise ValueError(f"entry ${entry:04X} out of range")

    cs = _make_cs()

    insns: dict[int, dict[str, Any]] = {}
    labels: dict[int, str] = {}
    subroutines: set[int] = set()
    remarks: defaultdict[int, list[str]] = defaultdict(list)
    queue: list[tuple[int, str]] = []
    queued: set[int] = set()
    paths_explored = 0

    def ensure_label(addr: int, prefer: str | None = None) -> None:
        if addr in labels:
            return
        if prefer:
            labels[addr] = prefer
        elif addr in subroutines:
            labels[addr] = f"sub_{addr:04x}"
        else:
            labels[addr] = f"loc_{addr:04x}"

    def enqueue(addr: int, origin: str, is_sub: bool = False) -> None:
        if not in_bounds(mem, addr) or addr in queued:
            return
        queued.add(addr)
        queue.append((addr, origin))
        if is_sub:
            subroutines.add(addr)
            ensure_label(addr, f"sub_{addr:04x}")
        else:
            ensure_label(addr)

    # Seed entry + extras + vectors
    ensure_label(entry, "entry")
    enqueue(entry, "entry point")

    if extra_seeds:
        for addr, label in extra_seeds:
            if in_bounds(mem, addr):
                ensure_label(addr, label)
                enqueue(addr, label)

    if seed_vectors:
        for vec_addr, name in HW_VECTORS.items():
            tgt = read_word(mem, vec_addr)
            if tgt and in_bounds(mem, tgt):
                ensure_label(tgt, name)
                enqueue(tgt, f"HW {name} @ ${vec_addr:04X}")
        for vec_addr, name in C64_RAM_VECTORS.items():
            tgt = read_word(mem, vec_addr)
            if tgt and in_bounds(mem, tgt) and tgt > 0x0400:
                ensure_label(tgt, name)
                enqueue(tgt, f"RAM {name} @ ${vec_addr:04X}")

    def follow_stream(start_addr: int, origin: str) -> None:
        nonlocal paths_explored
        addr = start_addr
        paths_explored += 1
        while True:
            if addr in insns or len(insns) >= max_insns:
                break
            if not in_bounds(mem, addr):
                remarks[addr].append(
                    f"WARNING: fell outside memory range (origin: {origin})"
                )
                break

            chunk = mem[addr : addr + 3]
            decoded = list(cs.disasm(chunk, addr))
            if not decoded:
                remarks[addr].append(
                    f"WARNING: failed to decode opcode ${mem[addr]:02X} "
                    f"(origin: {origin})"
                )
                break
            ins = decoded[0]

            insns[addr] = {
                "bytes": bytes(ins.bytes),
                "mnemonic": ins.mnemonic,
                "op_str": ins.op_str,
            }
            mnem = ins.mnemonic.lower()

            if mnem in RETURN_MNEMONICS:
                break
            if mnem in HALT_MNEMONICS:
                remarks[addr].append("BRK — software interrupt / end marker")
                break
            if mnem in BRANCH_MNEMONICS:
                tgt = parse_direct_target(ins.op_str)
                if tgt is not None:
                    ensure_label(tgt)
                    enqueue(tgt, f"branch from ${addr:04x}")
                addr += len(ins.bytes)
                continue
            if mnem in CALL_MNEMONICS:
                tgt = parse_direct_target(ins.op_str)
                if tgt is not None:
                    enqueue(tgt, f"JSR from ${addr:04x}", is_sub=True)
                addr += len(ins.bytes)
                continue
            if mnem in JUMP_MNEMONICS:
                vec = parse_indirect_vector(ins.op_str)
                if vec is not None:
                    tgt = read_word(mem, vec)
                    if tgt and in_bounds(mem, tgt):
                        remarks[addr].append(
                            f"Indirect JMP via ${vec:04X} -> ${tgt:04X}"
                        )
                        ensure_label(tgt)
                        enqueue(tgt, f"indirect JMP from ${addr:04x}")
                    else:
                        remarks[addr].append(
                            f"Indirect JMP via ${vec:04X} — runtime target"
                        )
                else:
                    tgt = parse_direct_target(ins.op_str)
                    if tgt is not None:
                        ensure_label(tgt)
                        enqueue(tgt, f"JMP from ${addr:04x}")
                    else:
                        remarks[addr].append(
                            "JMP with computed/indexed target — cannot trace statically"
                        )
                break

            addr += len(ins.bytes)

    while queue:
        a, origin = queue.pop(0)
        if a in insns:
            continue
        if len(insns) >= max_insns:
            break
        follow_stream(a, origin)

    listing = render_listing(insns, labels, remarks)

    return {
        "entry": entry,
        "insns": {f"${a:04X}": v for a, v in insns.items()},
        "labels": {f"${a:04X}": n for a, n in labels.items()},
        "subroutines": [f"${a:04X}" for a in sorted(subroutines)],
        "remarks": {f"${a:04X}": rs for a, rs in remarks.items()},
        "paths_explored": paths_explored,
        "listing": listing,
        "stats": {
            "instructions": len(insns),
            "labels": len(labels),
            "subroutines": len(subroutines),
            "max_insns": max_insns,
            "truncated": len(insns) >= max_insns,
        },
    }


def render_listing(
    insns: dict[int, dict[str, Any]],
    labels: dict[int, str],
    remarks: dict[int, list[str]],
    max_lines: int = 200,
) -> str:
    out: list[str] = []
    sorted_addrs = sorted(insns.keys())
    prev_end: Optional[int] = None
    line_count = 0
    truncated = False

    for addr in sorted_addrs:
        if line_count >= max_lines:
            truncated = True
            break

        info = insns[addr]
        if prev_end is not None and addr != prev_end:
            out.append(f";  gap ${prev_end:04X}-${addr - 1:04X} "
                       f"({addr - prev_end} bytes skipped)")
            line_count += 1

        for r in remarks.get(addr, []):
            out.append(f"; *** {r} ***")
            line_count += 1
            if line_count >= max_lines:
                truncated = True
                break
        if truncated:
            break

        if addr in labels:
            out.append(f"{labels[addr]}:")
            line_count += 1

        bytes_col = " ".join(f"{b:02X}" for b in info["bytes"])
        op = to_c64(info["op_str"])
        # Replace direct targets with label names where known.
        tgt = parse_direct_target(info["op_str"])
        if tgt is not None and tgt in labels:
            op = labels[tgt]
        else:
            vec = parse_indirect_vector(info["op_str"])
            if vec is not None and vec in labels:
                op = f"({labels[vec]})"

        out.append(
            f"  ${addr:04X}:  {bytes_col:<10}  {info['mnemonic']:<4} {op}".rstrip()
        )
        line_count += 1
        prev_end = addr + len(info["bytes"])

    if truncated:
        out.append(f"; ... listing truncated at {max_lines} lines ...")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Game-loop heuristics
# --------------------------------------------------------------------------- #

def _quick_disasm_stream(mem: bytes, start: int,
                         max_insns: int = 512) -> list[dict]:
    cs = _make_cs()
    out: list[dict] = []
    addr = start
    while len(out) < max_insns and 0 <= addr < len(mem):
        chunk = mem[addr : addr + 3]
        decoded = list(cs.disasm(chunk, addr))
        if not decoded:
            break
        ins = decoded[0]
        mnem = ins.mnemonic.lower()
        out.append({
            "addr": addr,
            "mnemonic": mnem,
            "op_str": ins.op_str,
            "size": len(ins.bytes),
            "target": parse_direct_target(ins.op_str),
        })
        if mnem in RETURN_MNEMONICS:
            break
        addr += len(ins.bytes)
    return out


def _score_loop(mem: bytes, top: int, end: int, insns: list[dict]
                ) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    inside = [i for i in insns if top <= i["addr"] <= end]
    if not inside:
        return 0.0, []

    span = end - top
    if span < 8:
        return 0.0, []
    if span > 0x2000:
        score -= 20.0

    jsr_n = sum(1 for i in inside if i["mnemonic"] == "jsr")
    if jsr_n >= 8:
        score += 40.0; reasons.append(f"{jsr_n} JSR calls (game-loop signal)")
    elif jsr_n >= 4:
        score += 20.0; reasons.append(f"{jsr_n} JSR calls")
    elif jsr_n >= 2:
        score += 8.0; reasons.append(f"{jsr_n} JSR calls")

    raster = joy = vic = sid = False
    for i in inside:
        t = i["target"]
        if t is None:
            m = re.match(r"^(0x[0-9a-fA-F]{1,4})", i["op_str"])
            if m:
                t = int(m.group(1), 16)
        if t is None:
            continue
        if t in _RASTER_ADDRS: raster = True
        if t in _JOY_ADDRS: joy = True
        if t in _VIC_RANGE: vic = True
        if t in _SID_RANGE: sid = True

    if raster: score += 30.0; reasons.append("raster wait ($D011/$D012)")
    if joy:    score += 20.0; reasons.append("joystick/keyboard read")
    if vic:    score += 10.0; reasons.append("VIC-II access")
    if sid:    score += 5.0;  reasons.append("SID access")

    for i in inside:
        if i["mnemonic"] == "jmp" and i["target"] is not None:
            if i["target"] == top:
                score += 25.0
                reasons.append(f"closes loop with JMP ${top:04X}")
                break
            if top <= i["target"] < i["addr"]:
                score += 10.0
                reasons.append(f"backward JMP to ${i['target']:04X}")

    if 0x0800 <= top <= 0xCFFF:
        score += 10.0
    elif top >= 0xE000:
        score -= 30.0

    return score, reasons


def find_loops(mem: bytes, top_n: int = 8) -> list[dict[str, Any]]:
    """Return up to top_n game-loop candidates, ranked by score desc."""
    candidates: dict[int, tuple[float, list[str]]] = {}

    def add(addr: int, score: float, reasons: list[str]) -> None:
        prev = candidates.get(addr)
        if prev is None or score > prev[0]:
            candidates[addr] = (score, reasons)

    cs = _make_cs()

    # Heuristic 1: $0314 RAM IRQ vector
    irq = read_word(mem, 0x0314)
    if irq and 0x0800 <= irq <= 0xCFFF:
        ins = _quick_disasm_stream(mem, irq, 256)
        s, r = _score_loop(mem, irq, ins[-1]["addr"] if ins else irq, ins)
        s += 15.0
        r.insert(0, "RAM IRQ vector $0314")
        add(irq, s, r)

    # Heuristic 2: backward JMP scan
    seen: set[int] = set()
    addr = 0x0800
    while addr < 0xCFFF:
        chunk = mem[addr : addr + 3]
        decoded = list(cs.disasm(chunk, addr))
        if not decoded:
            addr += 1
            continue
        ins = decoded[0]
        if ins.mnemonic == "jmp":
            tgt = parse_direct_target(ins.op_str)
            if tgt and 0x0800 <= tgt < addr and tgt not in seen:
                seen.add(tgt)
                if 8 <= addr - tgt <= 0x2000:
                    block = _quick_disasm_stream(mem, tgt, 512)
                    s, r = _score_loop(mem, tgt, addr, block)
                    if s > 0:
                        add(tgt, s, r)
        addr += len(decoded[0].bytes)

    # Heuristic 3: BASIC SYS bootstrap
    sys_addr = detect_basic_sys(mem)
    if sys_addr and 0x0800 <= sys_addr <= 0xCFFF:
        chain = sys_addr
        visited: set[int] = set()
        for _ in range(2048):
            if chain in visited or not (0x0800 <= chain <= 0xCFFF):
                break
            visited.add(chain)
            chunk = mem[chain : chain + 3]
            decoded = list(cs.disasm(chunk, chain))
            if not decoded:
                break
            ins = decoded[0]
            mnem = ins.mnemonic.lower()
            tgt = parse_direct_target(ins.op_str)
            if mnem == "jmp" and tgt and tgt < chain and tgt not in seen:
                block = _quick_disasm_stream(mem, tgt, 512)
                s, r = _score_loop(mem, tgt, chain, block)
                s += 20.0
                r.insert(0, "first backward JMP in BASIC SYS init chain")
                add(tgt, s, r)
                break
            chain += len(ins.bytes)

    ranked = sorted(
        ({"address": f"${a:04X}", "score": round(s, 1), "reasons": rs}
         for a, (s, rs) in candidates.items()),
        key=lambda c: c["score"],
        reverse=True,
    )
    return ranked[:top_n]


# --------------------------------------------------------------------------- #
# Entry-point detection
# --------------------------------------------------------------------------- #

def detect_basic_sys(mem: bytes) -> Optional[int]:
    """Return the SYS target address from a BASIC stub at $0801, if present."""
    try:
        region = mem[0x0801:0x0830]
        idx = region.index(0x9E)
        digits = bytearray()
        for b in region[idx + 1 :]:
            if 0x30 <= b <= 0x39:
                digits.append(b)
            elif b == 0x00:
                break
            else:
                break
        if digits:
            return int(digits.decode())
    except (ValueError, IndexError):
        pass
    return None


def find_entry(mem: bytes, hint: Optional[int] = None) -> dict[str, Any]:
    """Validate a hinted entry / suggest BASIC SYS entry / vector candidates."""
    out: dict[str, Any] = {
        "hint": f"${hint:04X}" if hint is not None else None,
        "basic_sys": None,
        "valid_at_hint": False,
        "byte_at_hint": None,
        "candidates": [],
    }

    sys_addr = detect_basic_sys(mem)
    if sys_addr is not None:
        out["basic_sys"] = f"${sys_addr:04X}"

    cs = _make_cs()
    if hint is not None and in_bounds(mem, hint):
        out["byte_at_hint"] = f"${mem[hint]:02X}"
        decoded = list(cs.disasm(mem[hint : hint + 3], hint))
        out["valid_at_hint"] = bool(decoded)

    cand: list[dict[str, Any]] = []
    if sys_addr is not None:
        cand.append({"address": f"${sys_addr:04X}", "source": "BASIC SYS"})
    for vec, name in C64_RAM_VECTORS.items():
        t = read_word(mem, vec)
        if t and in_bounds(mem, t) and t > 0x0400:
            cand.append({"address": f"${t:04X}",
                         "source": f"{name} (${vec:04X})"})
    for vec, name in HW_VECTORS.items():
        t = read_word(mem, vec)
        if t and in_bounds(mem, t):
            cand.append({"address": f"${t:04X}",
                         "source": f"{name} (${vec:04X})"})
    out["candidates"] = cand
    return out


# --------------------------------------------------------------------------- #
# Vector listing
# --------------------------------------------------------------------------- #

def list_vectors(mem: bytes) -> dict[str, Any]:
    def resolve(table: dict[int, str]) -> list[dict[str, Any]]:
        rows = []
        for vec, name in table.items():
            tgt = read_word(mem, vec)
            rows.append({
                "vector": f"${vec:04X}",
                "name": name,
                "target": f"${tgt:04X}" if tgt is not None else None,
            })
        return rows

    return {
        "hw_vectors": resolve(HW_VECTORS),
        "ram_vectors": resolve(C64_RAM_VECTORS),
    }


# --------------------------------------------------------------------------- #
# Polymorphic / self-modifying-code scan
# --------------------------------------------------------------------------- #

def detect_polymorphic(insns: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return suspect sites: stores into addresses we know contain code."""
    code_addrs = set(insns.keys())
    findings: list[dict[str, Any]] = []

    for addr, info in insns.items():
        mnem = info["mnemonic"].lower()
        if mnem not in STORE_MNEMONICS:
            continue
        op = info["op_str"]
        dest = parse_store_target(op)
        if dest is not None and dest in code_addrs:
            findings.append({
                "site": f"${addr:04X}",
                "mnemonic": mnem,
                "operand": to_c64(op),
                "writes_to": f"${dest:04X}",
                "kind": "absolute_into_code",
                "note": "store into known code address — possible self-modification",
            })
            continue
        m = re.match(r"^0x([0-9a-fA-F]{3,4})\s*,\s*([xy])$", op)
        if m and 0x0800 <= int(m.group(1), 16) <= 0xCFFF:
            base = int(m.group(1), 16)
            findings.append({
                "site": f"${addr:04X}",
                "mnemonic": mnem,
                "operand": to_c64(op),
                "writes_to": f"${base:04X},{m.group(2)}",
                "kind": "indexed_into_program_ram",
                "note": "indexed store into program RAM — possible decrypt loop",
            })
    return findings
