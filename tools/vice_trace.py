"""Conservative reconstruction of a 6502 writer at a confirmed watchpoint stop."""

from __future__ import annotations

import re
from typing import Callable


# Official NMOS 6502 memory-writing instructions. Accumulator shifts are excluded.
_WRITES = {
    0x85: ("STA", "zp"), 0x95: ("STA", "zpx"), 0x8D: ("STA", "abs"),
    0x9D: ("STA", "absx"), 0x99: ("STA", "absy"),
    0x81: ("STA", "indx"), 0x91: ("STA", "indy"),
    0x86: ("STX", "zp"), 0x96: ("STX", "zpy"), 0x8E: ("STX", "abs"),
    0x84: ("STY", "zp"), 0x94: ("STY", "zpx"), 0x8C: ("STY", "abs"),
}
for mnemonic, opcodes in {
    "INC": (0xE6, 0xF6, 0xEE, 0xFE), "DEC": (0xC6, 0xD6, 0xCE, 0xDE),
    "ASL": (0x06, 0x16, 0x0E, 0x1E), "LSR": (0x46, 0x56, 0x4E, 0x5E),
    "ROL": (0x26, 0x36, 0x2E, 0x3E), "ROR": (0x66, 0x76, 0x6E, 0x7E),
}.items():
    _WRITES.update(zip(opcodes, ((mnemonic, mode) for mode in ("zp", "zpx", "abs", "absx"))))


def register_byte(registers: dict, key: str) -> int | None:
    value = registers.get(key, registers.get(key.lower()))
    if isinstance(value, bool):
        return None
    try:
        value = int(value.strip().removeprefix("$").removeprefix("0x"), 16) if isinstance(value, str) else value
    except ValueError:
        return None
    return value if isinstance(value, int) and 0 <= value <= 255 else None


def writer_candidates(pc: int, watched: int, registers: dict,
                      read: Callable[[int, int], bytes]) -> list[dict]:
    """Candidates ending at the stopped PC, verified against effective addresses.

    A stopped PC and memory bytes do not prove instruction boundaries or execution.
    Return candidates only: the checkpoint hit establishes that a write occurred.
    """
    if pc < 3:
        return []  # Wrapping/self-modifying code needs a richer execution trace.
    window = read(pc - 3, 3)
    if len(window) != 3:
        return []
    candidates = []
    for size in (2, 3):
        raw = window[-size:]
        spec = _WRITES.get(raw[0])
        if spec is None:
            continue
        mnemonic, mode = spec
        if (3 if mode.startswith("abs") else 2) != size:
            continue
        base = int.from_bytes(raw[1:], "little")
        effective = base
        if mode in {"zpx", "zpy", "absx", "absy", "indx", "indy"}:
            reg = register_byte(registers, "X" if mode.endswith("x") or mode == "indx" else "Y")
            if reg is None:
                continue
            if mode in {"indx", "indy"}:
                pointer = (base + reg) & 255 if mode == "indx" else base
                lo, hi = read(pointer, 1), read((pointer + 1) & 255, 1)
                if len(lo) != 1 or len(hi) != 1:
                    continue
                effective = (lo[0] | hi[0] << 8) + (reg if mode == "indy" else 0)
                # A write into its own pointer destroys the pre-write evidence.
                if watched in {pointer, (pointer + 1) & 255}:
                    continue
            else:
                effective += reg
            effective &= 255 if mode.startswith("zp") else 65535
        if effective == watched and not pc - size <= watched < pc:
            candidates.append({"address": f"${pc-size:04X}", "mnemonic": mnemonic,
                               "mode": mode, "bytes": raw.hex(),
                               "effective_address": f"${effective:04X}"})
    return candidates


def literal_write_in_listing(text: str, watched: int) -> bool:
    """Static direct-address match only; never evidence that an instruction ran."""
    for line in text.splitlines():
        match = re.fullmatch(
            r"\s*\$?[0-9a-f]{4}:?\s+(?:[0-9a-f]{2}\s+){1,3}"
            r"(?:sta|stx|sty|inc|dec|asl|lsr|rol|ror)\s+\$([0-9a-f]{2,4})\s*",
            line, re.I,
        )
        if match and int(match[1], 16) == watched:
            return True
    return False
