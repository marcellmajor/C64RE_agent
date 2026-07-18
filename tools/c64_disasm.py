"""C64 6502 static-analysis helpers.

Distilled from a recursive-disassembler reference (see `capstone_sample.py`)
and exposed as composable helpers the agent can pick from per planner step:

  - linear_disasm        : block disassembly with bytes column + label hints
  - recursive_disasm     : control-flow following from a seed entry point
  - find_loops           : heuristic main-game-loop candidate scoring
  - find_entry           : BASIC SYS bootstrap detection / entry validation
  - list_vectors         : HW + RAM vectors
  - detect_polymorphic   : self-modifying-code suspect scan
  - find_counters        : game-state variable heuristics (tracker 3.3)
  - find_idioms          : deterministic 6502 idiom pre-tagging (tracker 3.5)
  - extract_data_refs    : LDA/STA/CMP memory-operand index (tracker 3.4)
  - decode_screen_ram    : screen+colour RAM → PETSCII strings (tracker 3.7)

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
# Bank awareness — tracker 3.6
# --------------------------------------------------------------------------- #

def banking_state(mem: bytes) -> dict[str, Any]:
    """Interpret the $0001 processor port captured in a RAM dump.

    The input is a RAM dump, and many games run code in RAM *under* the
    KERNAL/BASIC ROMs. The dumped $0001 tells us which banks were mapped
    at dump time (bit0 LORAM → BASIC, bit1 HIRAM → KERNAL, bit2 CHAREN →
    I/O vs char ROM). When a ROM is banked OUT, the corresponding address
    range is live RAM code — so it must NOT be penalised as "ROM" by the
    loop heuristic, and its HW vectors point at real game code.
    """
    port = mem[0x0001] if len(mem) > 1 else 0x37
    loram = bool(port & 0x01)   # 1 = BASIC ROM visible at $A000-$BFFF
    hiram = bool(port & 0x02)   # 1 = KERNAL ROM visible at $E000-$FFFF
    charen = bool(port & 0x04)  # 1 = I/O visible at $D000-$DFFF
    return {
        "port": port,
        "port_hex": f"${port:02X}",
        "basic_rom_visible": loram,
        "kernal_rom_visible": hiram,
        "io_visible": charen,
        # RAM-under-ROM: the ROM is banked out, so this range is game RAM.
        "ram_under_kernal": not hiram,
        "ram_under_basic": not loram and hiram,  # BASIC out only meaningful w/ kernal in
        "char_rom_visible": not charen,
    }


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
        # Bank awareness (tracker 3.6): when the KERNAL ROM is banked in,
        # HW-vector targets in $E000+ are KERNAL routines — seeding them
        # burns the limited max_insns budget disassembling ROM. Skip them
        # unless the dump's $0001 shows RAM under the KERNAL.
        kernal_ram = banking_state(mem)["ram_under_kernal"]
        for vec_addr, name in HW_VECTORS.items():
            tgt = read_word(mem, vec_addr)
            if not tgt or not in_bounds(mem, tgt):
                continue
            if tgt >= 0xE000 and not kernal_ram:
                remarks[tgt].append(
                    f"HW {name} @ ${vec_addr:04X} → ${tgt:04X} in banked-in "
                    "KERNAL ROM; not seeded (bank-aware)"
                )
                continue
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


def _score_loop(mem: bytes, top: int, end: int, insns: list[dict],
                ram_under_kernal: bool = False
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
        # Only penalise the KERNAL range when the KERNAL ROM was actually
        # banked in at dump time; under RAM-under-ROM the range is real
        # game code and the old flat −30 systematically hid it (3.6).
        if ram_under_kernal:
            score += 5.0
            reasons.append("code in RAM under KERNAL ($E000+)")
        else:
            score -= 30.0

    return score, reasons


def find_loops(mem: bytes, top_n: int = 8) -> list[dict[str, Any]]:
    """Return up to top_n game-loop candidates, ranked by score desc.

    Bank-aware (tracker 3.6): when $0001 shows the KERNAL ROM banked out,
    the $E000-$FFFF range is scanned as game RAM instead of being
    penalised as ROM.
    """
    bank = banking_state(mem)
    ram_under_kernal = bank["ram_under_kernal"]
    scan_hi = 0xFFF0 if ram_under_kernal else 0xCFFF
    candidates: dict[int, tuple[float, list[str]]] = {}

    def add(addr: int, score: float, reasons: list[str]) -> None:
        prev = candidates.get(addr)
        if prev is None or score > prev[0]:
            candidates[addr] = (score, reasons)

    cs = _make_cs()

    # Heuristic 1: $0314 RAM IRQ vector
    irq = read_word(mem, 0x0314)
    if irq and 0x0800 <= irq <= scan_hi:
        ins = _quick_disasm_stream(mem, irq, 256)
        s, r = _score_loop(mem, irq, ins[-1]["addr"] if ins else irq, ins,
                           ram_under_kernal)
        s += 15.0
        r.insert(0, "RAM IRQ vector $0314")
        add(irq, s, r)

    # Heuristic 2: backward JMP scan
    seen: set[int] = set()
    addr = 0x0800
    while addr < scan_hi:
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
                    s, r = _score_loop(mem, tgt, addr, block, ram_under_kernal)
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
                s, r = _score_loop(mem, tgt, chain, block, ram_under_kernal)
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
    """Return the decimal ``SYS`` target from a tokenized BASIC stub.

    Commodore BASIC stores ``SYS`` as token ``$9E`` followed by PETSCII
    text.  Real loaders commonly spell it ``SYS 2064``, ``SYS(2064)``,
    ``SYS:2064`` or ``SYS +2064``; the old parser required the first byte
    after ``$9E`` to be a digit and silently missed all of those forms.

    Follow the BASIC line-link chain when it is intact, with a small fixed
    window fallback for damaged/snapshot-in-progress stubs.  Only harmless
    leading separators are skipped; once decimal digits start, the first
    non-digit terminates the address.
    """

    def _target_after_token(body: bytes) -> Optional[int]:
        separators = {0x20, 0x28, 0x2B, 0x3A, 0xA0}  # space,(,+,:,shift-space
        start = 0
        while True:
            try:
                idx = body.index(0x9E, start)
            except ValueError:
                return None
            pos = idx + 1
            while pos < len(body) and body[pos] in separators:
                pos += 1
            digits = bytearray()
            while pos < len(body) and 0x30 <= body[pos] <= 0x39:
                digits.append(body[pos])
                pos += 1
            if digits:
                target = int(digits.decode("ascii"))
                return target if 0 <= target <= 0xFFFF else None
            start = idx + 1

    line = 0x0801
    visited: set[int] = set()
    for _ in range(256):
        if line in visited or line < 0 or line + 4 > len(mem):
            break
        visited.add(line)
        next_line = read_word(mem, line)
        try:
            end = mem.index(0x00, line + 4, min(len(mem), line + 260))
        except ValueError:
            break
        target = _target_after_token(mem[line + 4 : end])
        if target is not None:
            return target
        if next_line == 0:
            return None
        if next_line <= line or next_line >= len(mem):
            break
        line = next_line

    # Corrupt line links are common in live snapshots captured while a
    # loader is modifying the BASIC area. Preserve the old bounded scan,
    # now with the separator-tolerant parser.
    if len(mem) > 0x0801:
        return _target_after_token(mem[0x0801 : min(len(mem), 0x0830)])
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


# --------------------------------------------------------------------------- #
# Operand classification (shared by data-refs, counters, idioms) — tracker 3.4
# --------------------------------------------------------------------------- #

# Mnemonic → memory-access class. RMW instructions read-modify-write their
# operand; loads/compares read; stores write.
_LOAD_MNEMONICS = frozenset({
    "lda", "ldx", "ldy", "cmp", "cpx", "cpy", "bit",
    "and", "ora", "eor", "adc", "sbc",
})
_RMW_MNEMONICS = frozenset({"inc", "dec", "asl", "lsr", "rol", "ror"})
# Reuse STORE_MNEMONICS for writes.

_OPERAND_RE = re.compile(
    r"^(?P<indir>\()?"
    r"(?:#)?(?P<imm>#)?"          # placeholder — immediates handled below
    r"0x(?P<addr>[0-9a-fA-F]{1,4})"
    r"(?P<indir_close>\))?"
    r"(?:\s*,\s*(?P<index>[xyXY]))?"
    r"(?P<post_close>\))?"
    r"(?:\s*,\s*(?P<post_index>[xyXY]))?"
    r"\s*$"
)
_IMMEDIATE_RE = re.compile(r"^#0x[0-9a-fA-F]{1,2}\s*$")


def classify_operand(op_str: str) -> dict[str, Any] | None:
    """Parse a Capstone MOS65XX operand into memory-reference facts.

    Returns ``{addr, index, indirect}`` for memory operands, or ``None``
    for immediates / accumulator / unparseable operands. ``index`` is
    ``"x"``/``"y"``/``None``; ``indirect`` is True for ``(zp),y`` /
    ``(zp,x)`` / ``(abs)`` forms.
    """
    s = (op_str or "").strip()
    if not s or s.lower() == "a":
        return None
    if _IMMEDIATE_RE.match(s):
        return None

    indirect = s.startswith("(")
    # Normalise: strip one layer of parens for the address extraction.
    core = s
    m = re.match(
        r"^\(?\s*0x(?P<addr>[0-9a-fA-F]{1,4})\s*"
        r"(?:,\s*(?P<xin>[xyXY]))?\s*\)?"
        r"(?:\s*,\s*(?P<yout>[xyXY]))?\s*$",
        core,
    )
    if not m:
        return None
    addr = int(m.group("addr"), 16)
    index = (m.group("xin") or m.group("yout") or "").lower() or None
    return {"addr": addr, "index": index, "indirect": indirect}


def _access_kind(mnem: str) -> str | None:
    m = mnem.lower()
    if m in STORE_MNEMONICS:
        return "w"
    if m in _RMW_MNEMONICS:
        return "rmw"
    if m in _LOAD_MNEMONICS:
        return "r"
    return None


def _insn_stream(insns: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise a recursive_disasm `insns` dict into a sorted list.

    Accepts int-keyed or ``$XXXX``-keyed dicts; each value carries
    ``mnemonic`` and ``op_str``.
    """
    out: list[dict[str, Any]] = []
    for k, v in insns.items():
        addr = k if isinstance(k, int) else int(str(k).lstrip("$"), 16)
        out.append({
            "addr": addr,
            "mnemonic": str(v.get("mnemonic") or "").lower(),
            "op_str": str(v.get("op_str") or ""),
        })
    out.sort(key=lambda r: r["addr"])
    return out


def extract_data_refs(
    insns: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Every memory-operand reference in a disassembled stream (tracker 3.4).

    Returns rows ``{src, dst, access, index, indirect}`` — the data-flow
    counterpart to the control-flow xref graph. ``access`` ∈ {r, w, rmw}.
    """
    refs: list[dict[str, Any]] = []
    for ins in _insn_stream(insns):
        access = _access_kind(ins["mnemonic"])
        if access is None:
            continue
        parsed = classify_operand(ins["op_str"])
        if parsed is None:
            continue
        refs.append({
            "src": ins["addr"],
            "dst": parsed["addr"],
            "access": access,
            "index": parsed["index"],
            "indirect": parsed["indirect"],
        })
    return refs


# --------------------------------------------------------------------------- #
# Game-state variable heuristics — tracker 3.3
# --------------------------------------------------------------------------- #

_SCREEN_RAM = range(0x0400, 0x07E8)


def find_counters(
    insns: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Heuristically locate game-state variables in a disassembled stream.

    `find_loops` finds control flow; nothing found *state*. 6502 game-state
    idioms are regular and cheap to scan (tracker 3.3):

    - ``DEC abs`` into RAM (< $D000, >= $0002) → lives/timer (DEC weighted
      higher for "lives"); ``INC abs`` → counter/timer.
    - ``SED … ADC/SBC … CLD`` clusters → BCD score arithmetic; the
      addresses stored between SED and CLD are score bytes.
    - stores into screen RAM ($0400–$07E7) preceded by ``ORA/ADC #$30`` →
      HUD digit rendering.
    - ``CMP #$0A`` shortly after ``INC abs`` → decimal rollover, marks
      multi-byte counters.

    Returns ONE candidate per address: ``{addr, address, kind, score,
    alias_kinds, evidence, sites}`` where ``kind`` is the highest-scoring
    interpretation for that address and ``alias_kinds`` lists the other
    interpretations the same site matched (so a single ``DEC`` no longer
    produces two competing first-class candidates — tracker 3.3
    hardening). ``kind`` ∈ {lives, timer, counter, score, hud_digit,
    multibyte_counter}.
    """
    stream = _insn_stream(insns)
    by_addr: dict[tuple[int, str], dict[str, Any]] = {}

    def bump(addr: int, kind: str, score: float, evidence: str,
             site: int) -> None:
        key = (addr, kind)
        row = by_addr.get(key)
        if row is None:
            row = {"addr": addr, "kind": kind, "score": 0.0,
                   "evidence": [], "sites": []}
            by_addr[key] = row
        row["score"] += score
        if evidence not in row["evidence"]:
            row["evidence"].append(evidence)
        if site not in row["sites"]:
            row["sites"].append(site)

    # --- DEC/INC absolute RAM counters ---
    for ins in stream:
        mnem = ins["mnemonic"]
        if mnem not in ("dec", "inc"):
            continue
        parsed = classify_operand(ins["op_str"])
        if parsed is None or parsed["indirect"]:
            continue
        addr = parsed["addr"]
        if not (0x0002 <= addr < 0xD000):
            continue
        if mnem == "dec":
            bump(addr, "lives", 12.0,
                 f"DEC ${addr:04X} at ${ins['addr']:04X}", ins["addr"])
            bump(addr, "timer", 4.0,
                 f"DEC ${addr:04X} at ${ins['addr']:04X}", ins["addr"])
        else:
            bump(addr, "counter", 8.0,
                 f"INC ${addr:04X} at ${ins['addr']:04X}", ins["addr"])
            bump(addr, "timer", 4.0,
                 f"INC ${addr:04X} at ${ins['addr']:04X}", ins["addr"])

    # --- SED … CLD BCD-score clusters ---
    in_bcd = False
    bcd_start = 0
    for ins in stream:
        if ins["mnemonic"] == "sed":
            in_bcd = True
            bcd_start = ins["addr"]
            continue
        if ins["mnemonic"] == "cld":
            in_bcd = False
            continue
        if in_bcd and ins["mnemonic"] in STORE_MNEMONICS:
            parsed = classify_operand(ins["op_str"])
            if parsed and not parsed["indirect"] and 0x0002 <= parsed["addr"] < 0xD000:
                bump(parsed["addr"], "score", 18.0,
                     f"store inside SED…CLD (from ${bcd_start:04X})",
                     ins["addr"])

    # --- HUD digit rendering: ORA/ADC #$30 then STA screen RAM ---
    for i, ins in enumerate(stream):
        if ins["mnemonic"] not in ("ora", "adc"):
            continue
        if not _IMMEDIATE_RE.match(ins["op_str"].strip()):
            continue
        if "0x30" not in ins["op_str"]:
            continue
        # Look ahead a few instructions for a store into screen RAM.
        for nxt in stream[i + 1:i + 6]:
            if nxt["mnemonic"] in STORE_MNEMONICS:
                parsed = classify_operand(nxt["op_str"])
                if parsed and parsed["addr"] in _SCREEN_RAM:
                    bump(parsed["addr"], "hud_digit", 10.0,
                         f"#$30 digit at ${ins['addr']:04X} → "
                         f"STA ${parsed['addr']:04X}", nxt["addr"])
                    break

    # --- CMP #$0A soon after INC abs: decimal-rollover multibyte counter ---
    last_inc: tuple[int, int] | None = None  # (addr_var, site)
    for ins in stream:
        if ins["mnemonic"] == "inc":
            parsed = classify_operand(ins["op_str"])
            if parsed and not parsed["indirect"]:
                last_inc = (parsed["addr"], ins["addr"])
            continue
        if (
            ins["mnemonic"] == "cmp"
            and last_inc is not None
            and "0x0a" in ins["op_str"].lower()
            and ins["addr"] - last_inc[1] <= 8
        ):
            # Weighted above a bare INC counter (8.0) so the more
            # specific rollover interpretation wins the primary kind.
            bump(last_inc[0], "multibyte_counter", 14.0,
                 f"INC ${last_inc[0]:04X} then CMP #$0A at ${ins['addr']:04X}",
                 ins["addr"])
            last_inc = None

    # Collapse to ONE candidate per address: keep the highest-scoring
    # kind as primary, fold the rest into `alias_kinds` (their evidence
    # merged in). Prevents e.g. a single DEC emitting both a lives and a
    # timer candidate as equal first-class rows (tracker 3.3 hardening).
    per_addr: dict[int, dict[str, Any]] = {}
    for row in by_addr.values():
        addr = row["addr"]
        cur = per_addr.get(addr)
        if cur is None or row["score"] > cur["score"]:
            if cur is not None:
                row.setdefault("alias_kinds", [])
                row["alias_kinds"] = sorted(
                    set(row.get("alias_kinds", []))
                    | {cur["kind"], *cur.get("alias_kinds", [])}
                )
                for ev in cur["evidence"]:
                    if ev not in row["evidence"]:
                        row["evidence"].append(ev)
            per_addr[addr] = row
        else:
            cur.setdefault("alias_kinds", [])
            if row["kind"] not in cur["alias_kinds"]:
                cur["alias_kinds"].append(row["kind"])
                cur["alias_kinds"].sort()
            for ev in row["evidence"]:
                if ev not in cur["evidence"]:
                    cur["evidence"].append(ev)

    out = sorted(per_addr.values(), key=lambda r: -r["score"])
    for r in out:
        r["score"] = round(r["score"], 1)
        r["address"] = f"${r['addr']:04X}"
        r.setdefault("alias_kinds", [])
    return out


# --------------------------------------------------------------------------- #
# Deterministic 6502 idiom pre-tagging — tracker 3.5
# --------------------------------------------------------------------------- #

def find_idioms(
    insns: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Table-driven recognition of rigid 6502 idioms (tracker 3.5).

    Most idioms have byte/structure signatures a matcher can pre-tag with
    high precision, so Layer-1 confirms rather than discovers. Recognises:
    raster wait, busy-wait delay, memcpy, jump-table dispatch, KERNAL
    trampoline, SID music tick. Returns ``{addr, idiom, evidence}`` rows.
    """
    stream = _insn_stream(insns)
    n = len(stream)
    found: list[dict[str, Any]] = []

    def at(i: int) -> dict[str, Any] | None:
        return stream[i] if 0 <= i < n else None

    for i, ins in enumerate(stream):
        mnem = ins["mnemonic"]
        op = ins["op_str"].lower()

        # KERNAL trampoline: JMP $FFxx (or JSR into KERNAL jump table).
        if mnem in ("jmp", "jsr"):
            tgt = parse_direct_target(ins["op_str"])
            if tgt is not None and 0xFF00 <= tgt <= 0xFFFF:
                found.append({
                    "addr": ins["addr"], "idiom": "kernal_trampoline",
                    "evidence": f"{mnem.upper()} ${tgt:04X}",
                })

        # Raster wait: LDA $D011/$D012 ; CMP #imm ; BNE/BCC back.
        if mnem == "lda":
            parsed = classify_operand(ins["op_str"])
            if parsed and parsed["addr"] in _RASTER_ADDRS:
                nxt, nxt2 = at(i + 1), at(i + 2)
                if (nxt and nxt["mnemonic"] in ("cmp", "and", "bit")
                        and nxt2 and nxt2["mnemonic"] in
                        ("bne", "bcc", "bcs", "beq")):
                    found.append({
                        "addr": ins["addr"], "idiom": "raster_wait",
                        "evidence": (
                            f"LDA ${parsed['addr']:04X} ; "
                            f"{nxt['mnemonic'].upper()} ; "
                            f"{nxt2['mnemonic'].upper()}"
                        ),
                    })

        # Busy-wait delay: DEX/DEY ; BNE (self-loop).
        if mnem in ("dex", "dey"):
            nxt = at(i + 1)
            if nxt and nxt["mnemonic"] == "bne":
                tgt = parse_direct_target(nxt["op_str"])
                if tgt is not None and tgt <= ins["addr"]:
                    found.append({
                        "addr": ins["addr"], "idiom": "delay_loop",
                        "evidence": f"{mnem.upper()} ; BNE ${tgt:04X}",
                    })

        # memcpy: LDA abs,X ; STA abs,X ; (INX/DEX) ; BNE.
        if mnem == "lda":
            p0 = classify_operand(ins["op_str"])
            nxt = at(i + 1)
            if p0 and p0["index"] == "x" and nxt and nxt["mnemonic"] == "sta":
                p1 = classify_operand(nxt["op_str"])
                nxt2, nxt3 = at(i + 2), at(i + 3)
                if (p1 and p1["index"] == "x"
                        and nxt2 and nxt2["mnemonic"] in ("inx", "dex")
                        and nxt3 and nxt3["mnemonic"] == "bne"):
                    found.append({
                        "addr": ins["addr"], "idiom": "memcpy",
                        "evidence": (
                            f"LDA ${p0['addr']:04X},X ; "
                            f"STA ${p1['addr']:04X},X ; "
                            f"{nxt2['mnemonic'].upper()} ; BNE"
                        ),
                    })

        # Jump-table dispatch: ASL ; TAX ; ... ; JMP ($XXXX).
        if mnem == "asl":
            nxt = at(i + 1)
            if nxt and nxt["mnemonic"] in ("tax", "tay"):
                for j in range(i + 2, min(i + 8, n)):
                    cand = stream[j]
                    if cand["mnemonic"] == "jmp" and cand["op_str"].strip().startswith("("):
                        found.append({
                            "addr": ins["addr"], "idiom": "jump_table",
                            "evidence": (
                                f"ASL ; {nxt['mnemonic'].upper()} ; "
                                f"JMP {to_c64(cand['op_str'])}"
                            ),
                        })
                        break

    # SID music tick: >=3 stores into $D400-$D418 within a short window.
    sid_stores = [
        ins for ins in stream
        if ins["mnemonic"] in STORE_MNEMONICS
        and (lambda p: p and 0xD400 <= p["addr"] <= 0xD418)(
            classify_operand(ins["op_str"]))
    ]
    if len(sid_stores) >= 3:
        # Cluster by proximity (<= 64 bytes apart).
        cluster: list[dict[str, Any]] = []
        for ins in sid_stores:
            if cluster and ins["addr"] - cluster[-1]["addr"] > 64:
                if len(cluster) >= 3:
                    found.append({
                        "addr": cluster[0]["addr"], "idiom": "sid_tick",
                        "evidence": f"{len(cluster)} SID register stores",
                    })
                cluster = []
            cluster.append(ins)
        if len(cluster) >= 3:
            found.append({
                "addr": cluster[0]["addr"], "idiom": "sid_tick",
                "evidence": f"{len(cluster)} SID register stores",
            })

    found.sort(key=lambda r: r["addr"])
    for r in found:
        r["address"] = f"${r['addr']:04X}"
    return found


# --------------------------------------------------------------------------- #
# Screen / colour RAM PETSCII decode — tracker 3.7
# --------------------------------------------------------------------------- #

def _screen_code_to_ascii(code: int) -> str | None:
    """C64 screen code → ASCII (printable subset), or None.

    Screen code $00 is the ``@`` glyph, but it is also the value of
    uninitialised/zeroed RAM, so a raw dump's blank screen would decode
    to ``@@@@…`` noise. We treat $00 as padding (None); the rare literal
    ``@`` in HUD text is an acceptable loss versus false-positive runs.
    """
    c = code & 0x7F  # ignore the reverse-video bit
    if c == 0x00:
        return None                   # padding (see docstring)
    if 0x01 <= c <= 0x1A:
        return chr(c + 0x40)          # A–Z
    if 0x1B <= c <= 0x1F:
        return chr(c + 0x40)          # [ \ ] ^ _
    if 0x20 <= c <= 0x3F:
        return chr(c)                 # space ! " … 0–9 : ; < = > ?
    return None                       # graphics / non-printable


def decode_screen_ram(
    mem: bytes,
    screen_base: int = 0x0400,
    color_base: int = 0xD800,
    *,
    cols: int = 40,
    rows: int = 25,
    min_run: int = 3,
) -> dict[str, Any]:
    """Decode screen RAM to PETSCII strings (tracker 3.7).

    Often answers "what does the HUD say / where is the score text?"
    without vision or full RE. Returns per-row decoded text plus a list
    of printable runs ``{addr, row, col, text}`` at least ``min_run``
    chars long. ``color_base`` is accepted for API symmetry (colour RAM
    doesn't change the glyphs, only their colour).
    """
    size = cols * rows
    if screen_base + size > len(mem):
        screen_data = bytes(mem[screen_base:len(mem)])
    else:
        screen_data = bytes(mem[screen_base:screen_base + size])

    row_texts: list[str] = []
    runs: list[dict[str, Any]] = []
    for r in range(rows):
        line_chars: list[str] = []
        run_start_col: int | None = None
        run_chars: list[str] = []

        def flush_run(end_col: int) -> None:
            nonlocal run_start_col, run_chars
            if run_start_col is not None and len(run_chars) >= min_run:
                text = "".join(run_chars).rstrip()
                if len(text) >= min_run and text.strip():
                    addr = screen_base + r * cols + run_start_col
                    runs.append({
                        "addr": addr, "addr_hex": f"${addr:04X}",
                        "row": r, "col": run_start_col, "text": text,
                    })
            run_start_col = None
            run_chars = []

        for c in range(cols):
            idx = r * cols + c
            if idx >= len(screen_data):
                break
            ch = _screen_code_to_ascii(screen_data[idx])
            if ch is None or ch == " ":
                line_chars.append(" ")
                # A single space breaks a run only if it's trailing; keep
                # internal spaces so "HI SCORE" stays one run.
                if ch == " " and run_start_col is not None:
                    run_chars.append(" ")
                else:
                    flush_run(c)
            else:
                line_chars.append(ch)
                if run_start_col is None:
                    run_start_col = c
                run_chars.append(ch)
        flush_run(cols)
        row_texts.append("".join(line_chars).rstrip())

    return {
        "screen_base": f"${screen_base:04X}",
        "color_base": f"${color_base:04X}",
        "rows": row_texts,
        "runs": runs,
    }
