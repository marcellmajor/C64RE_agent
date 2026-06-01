"""Tolerant parser for partial-asm files in `asm_dir/`.

The files in our corpus come from at least two distinct disassemblers and
also include hand-edited listings, so the parser intentionally accepts a
union of common formats:

    Format A (capital hex, no `$`, no colon):
        ``0073  E6 7A       INC $7A``

    Format B (lower hex, leading ``$XXXX:``, capstone-style ``0x`` operands):
        ``  $0054:  4c 00 00    jmp    0x0000``

    Format C (Format A but with symbolic operands):
        ``1096  B9 B4 00   LDA  sprite_2_type,Y``

What we extract per file:
    - `instructions` :  ParsedInstruction(addr, bytes_, mnemonic, operand,
                                          raw_operand)
    - `labels`       :  ParsedLabel(addr_of_next_insn, name)
    - `gaps`         :  ParsedGap(start, end)  (inclusive)

Operand canonicalisation is intentionally light: any `0xNNNN` is rewritten
to `$NNNN`, hex digits go uppercase, trailing whitespace and inline
``;`` comments are stripped. Symbolic operands stay symbolic — the
Layer-0 builder is responsible for resolving them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# --------------------------------------------------------------------------- #
# Regex grammar
# --------------------------------------------------------------------------- #

# Instruction line — leading whitespace allowed; addr optional `$`/`:`; 1-3
# bytes; 2-4 letter mnemonic; everything after = operand (we strip comments).
_INSN_RE = re.compile(
    r"""^\s*
        \$?(?P<addr>[0-9a-fA-F]{4})\s*:?\s+        # address
        (?P<bytes>(?:[0-9a-fA-F]{2}\s){0,3}        # 0-3 hex bytes (some lines
         [0-9a-fA-F]{2})\s+                        # have all three on the line)
        (?P<mnem>[A-Za-z][A-Za-z]{1,3})            # mnemonic
        (?:\s+(?P<op>.*?))?                        # operand (optional)
        \s*$
    """,
    re.VERBOSE,
)

# Implied/accumulator instructions can lack an operand (e.g. `0084  38  SEC`).
_INSN_NO_OP_RE = re.compile(
    r"""^\s*
        \$?(?P<addr>[0-9a-fA-F]{4})\s*:?\s+
        (?P<bytes>(?:[0-9a-fA-F]{2}\s){0,3}[0-9a-fA-F]{2})\s+
        (?P<mnem>[A-Za-z][A-Za-z]{1,3})
        \s*$
    """,
    re.VERBOSE,
)

_LABEL_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?:;.*)?$"
)

_GAP_RE = re.compile(
    r"gap[: \t]*\$?([0-9a-fA-F]{4})\s*[-–—]\s*\$?([0-9a-fA-F]{4})",
    re.IGNORECASE,
)

_HEX_0X_RE = re.compile(r"0x([0-9a-fA-F]+)")
_TRAILING_COMMENT_RE = re.compile(r"\s*;.*$")


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParsedInstruction:
    """A single decoded instruction line."""

    addr: int
    bytes_: bytes
    mnemonic: str           # lowercase
    operand: str            # canonical ($XXXX hex, uppercase digits)
    raw_operand: str        # original after comment-stripping

    @property
    def size(self) -> int:
        return len(self.bytes_) or 1


@dataclass(frozen=True)
class ParsedLabel:
    """A name attached to the *next* instruction's address.

    During parsing we don't know that address yet, so labels are emitted as
    `(name, line_index)` first and resolved afterward.
    """

    addr: int
    name: str


@dataclass(frozen=True)
class ParsedGap:
    start: int   # inclusive
    end: int     # inclusive


@dataclass
class ParsedAsmFile:
    path: str
    instructions: list[ParsedInstruction] = field(default_factory=list)
    labels: list[ParsedLabel] = field(default_factory=list)
    gaps: list[ParsedGap] = field(default_factory=list)
    skipped_lines: int = 0
    total_lines: int = 0

    def stats(self) -> dict[str, int]:
        return {
            "instructions": len(self.instructions),
            "labels": len(self.labels),
            "gaps": len(self.gaps),
            "skipped_lines": self.skipped_lines,
            "total_lines": self.total_lines,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _strip_comment(s: str) -> str:
    return _TRAILING_COMMENT_RE.sub("", s).rstrip()


def _canon_operand(op: str) -> str:
    """Normalise a raw operand: strip comments, rewrite 0xNN -> $NN, uppercase hex."""
    if not op:
        return ""
    op = _strip_comment(op).strip()
    if not op:
        return ""
    op = _HEX_0X_RE.sub(lambda m: "$" + m.group(1).upper(), op)

    # Uppercase any leftover lowercase hex digits inside `$..` tokens.
    def _upper_hex(m: re.Match[str]) -> str:
        return "$" + m.group(1).upper()

    op = re.sub(r"\$([0-9a-fA-F]+)", _upper_hex, op)
    return op


def _parse_bytes(bytes_field: str) -> bytes:
    parts = bytes_field.split()
    out = bytearray()
    for p in parts:
        try:
            out.append(int(p, 16))
        except ValueError:
            return bytes(out)
    return bytes(out)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def parse_lines(
    lines: Iterable[str],
    *,
    path: str = "<memory>",
) -> ParsedAsmFile:
    """Parse an iterable of source lines into a `ParsedAsmFile`."""
    pf = ParsedAsmFile(path=path)
    pending_labels: list[str] = []
    last_addr_for_warning = -1

    for raw in lines:
        pf.total_lines += 1
        line = raw.rstrip()

        # 1. Skip blank lines / dividers / pure-comment lines (but probe gaps).
        bare = line.strip()
        if not bare:
            continue

        if bare.startswith((";", "#", "//")):
            gap = _GAP_RE.search(bare)
            if gap:
                start = int(gap.group(1), 16)
                end = int(gap.group(2), 16)
                if end >= start:
                    pf.gaps.append(ParsedGap(start=start, end=end))
            continue

        # 2. Try to read as an instruction first (most lines are insns).
        m = _INSN_RE.match(line)
        if not m:
            m = _INSN_NO_OP_RE.match(line)

        if m:
            addr = int(m.group("addr"), 16)
            bytes_ = _parse_bytes(m.group("bytes"))
            mnem = m.group("mnem").lower()
            try:
                op_field = m.group("op") or ""
            except IndexError:
                op_field = ""
            raw_op = _strip_comment(op_field).strip()
            canon_op = _canon_operand(raw_op)
            insn = ParsedInstruction(
                addr=addr,
                bytes_=bytes_,
                mnemonic=mnem,
                operand=canon_op,
                raw_operand=raw_op,
            )
            pf.instructions.append(insn)

            for n in pending_labels:
                pf.labels.append(ParsedLabel(addr=addr, name=n))
            pending_labels.clear()
            last_addr_for_warning = addr
            continue

        # 3. Label-only line (`storeBubType:`).
        m = _LABEL_RE.match(line)
        if m:
            pending_labels.append(m.group("name"))
            continue

        # 4. Anything else: count and move on.
        pf.skipped_lines += 1

    # Trailing labels with no following instruction get attached to the
    # last seen address (so the export still shows them). They're rare.
    if pending_labels and last_addr_for_warning >= 0:
        for n in pending_labels:
            pf.labels.append(ParsedLabel(addr=last_addr_for_warning, name=n))
    return pf


def parse_path(path: Path | str) -> ParsedAsmFile:
    """Parse a file on disk."""
    p = Path(path)
    text = p.read_text(errors="replace")
    pf = parse_lines(text.splitlines(), path=str(p))
    return pf


# --------------------------------------------------------------------------- #
# Operand introspection helpers (used by the Layer-0 builder)
# --------------------------------------------------------------------------- #


_OP_DIRECT_RE   = re.compile(r"^\$([0-9A-F]{1,4})(?:\s*,\s*[XYxy])?$")
_OP_INDIRECT_RE = re.compile(r"^\(\s*\$([0-9A-F]{1,4})\s*\)$")
_OP_IND_X_RE    = re.compile(r"^\(\s*\$([0-9A-F]{1,4})\s*,\s*[Xx]\s*\)$")
_OP_IND_Y_RE    = re.compile(r"^\(\s*\$([0-9A-F]{1,4})\s*\)\s*,\s*[Yy]$")
_OP_IMM_RE      = re.compile(r"^#\$?[0-9A-Fa-f]+$|^#%[01]+$|^#\d+$")
_BARE_LABEL_RE  = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[XYxy])?$")


def operand_is_immediate(op: str) -> bool:
    return bool(_OP_IMM_RE.match(op.strip()))


def operand_direct_target(op: str) -> int | None:
    """Return the absolute target of an absolute / zp operand (`$XXXX`).

    Skips immediate (`#$NN`) and indirect (`($XXXX)`) operands. Indexed
    operands (`$XXXX,X` / `$XXXX,Y`) are returned as their *base*.
    """
    if not op:
        return None
    s = op.strip()
    if operand_is_immediate(s):
        return None
    m = _OP_DIRECT_RE.match(s)
    if not m:
        return None
    try:
        return int(m.group(1), 16)
    except ValueError:
        return None


def operand_indirect_vector(op: str) -> int | None:
    """Return the vector address for `($XXXX)` operands; ``None`` otherwise."""
    if not op:
        return None
    m = _OP_INDIRECT_RE.match(op.strip())
    if not m:
        return None
    try:
        return int(m.group(1), 16)
    except ValueError:
        return None


def operand_label_ref(op: str) -> str | None:
    """Return the bare label name if the operand looks like a symbolic ref."""
    if not op:
        return None
    s = op.strip()
    if not _BARE_LABEL_RE.match(s):
        return None
    return s.split(",")[0].strip()


def operand_store_target(op: str) -> int | None:
    """Same as `operand_direct_target` but only matches non-zero-page writes."""
    target = operand_direct_target(op)
    if target is None:
        return None
    return target if target > 0xFF else None
