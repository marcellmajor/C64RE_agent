"""Conservative, address-preserving 6502 mechanical pseudocode.

This is intentionally a transliterator, not a decompiler: every output line
retains its source address and unknown instructions remain explicit assembly.
That makes it useful as an LLM/human reading aid without inventing recovered
types, loops, or control structures that Layer 0 cannot prove.
"""

from __future__ import annotations

import re
from typing import Any


_BRANCH_CONDITIONS = {
    "bcc": "not C",
    "bcs": "C",
    "beq": "Z",
    "bmi": "N",
    "bne": "not Z",
    "bpl": "not N",
    "bvc": "not V",
    "bvs": "V",
}
_LOAD_REG = {"lda": "A", "ldx": "X", "ldy": "Y"}
_STORE_REG = {"sta": "A", "stx": "X", "sty": "Y"}
_COMPARE_REG = {"cmp": "A", "cpx": "X", "cpy": "Y"}
_INC_DEC_REG = {
    "inx": "X += 1",
    "iny": "Y += 1",
    "dex": "X -= 1",
    "dey": "Y -= 1",
}
_FLAG_OPS = {
    "clc": "C = 0",
    "sec": "C = 1",
    "cli": "I = 0",
    "sei": "I = 1",
    "cld": "D = 0",
    "sed": "D = 1",
    "clv": "V = 0",
}
_TRANSFERS = {
    "tax": "X = A",
    "tay": "Y = A",
    "txa": "A = X",
    "tya": "A = Y",
    "tsx": "X = SP",
    "txs": "SP = X",
}


def _operand_text(raw: Any) -> str:
    text = str(raw or "").strip()
    text = re.sub(
        r"0x([0-9a-fA-F]{1,4})",
        lambda match: "$" + match.group(1).upper().zfill(4),
        text,
    )
    return text


def _value(operand: str) -> str:
    if not operand:
        return "?"
    if operand.upper() == "A":
        return "A"
    if operand.startswith("#"):
        return operand[1:]
    return f"mem[{operand}]"


def _target(operand: str) -> str:
    match = re.search(r"\$([0-9A-Fa-f]{2,4})", operand)
    if match:
        return f"L_{int(match.group(1), 16):04X}"
    return operand or "unknown"


def translate_instruction(mnemonic: str, operand: Any) -> str:
    """Translate one 6502 instruction without hiding unsupported cases."""
    mnem = str(mnemonic or "?").strip().lower()
    op = _operand_text(operand)
    value = _value(op)

    if mnem in _LOAD_REG:
        return f"{_LOAD_REG[mnem]} = {value}"
    if mnem in _STORE_REG:
        return f"mem[{op or '?'}] = {_STORE_REG[mnem]}"
    if mnem in _COMPARE_REG:
        return f"flags = {_COMPARE_REG[mnem]} - {value}"
    if mnem == "adc":
        return f"A = A + {value} + C"
    if mnem == "sbc":
        return f"A = A - {value} - (1 - C)"
    if mnem in {"and", "ora", "eor"}:
        symbol = {"and": "&", "ora": "|", "eor": "^"}[mnem]
        return f"A = A {symbol} {value}"
    if mnem in {"inc", "dec"}:
        delta = "+= 1" if mnem == "inc" else "-= 1"
        return f"mem[{op or '?'}] {delta}"
    if mnem in _INC_DEC_REG:
        return _INC_DEC_REG[mnem]
    if mnem in {"asl", "lsr", "rol", "ror"}:
        target = "A" if not op or op.upper() == "A" else f"mem[{op}]"
        operation = {
            "asl": f"{target} <<= 1",
            "lsr": f"{target} >>= 1",
            "rol": f"{target} = ({target} << 1) | C",
            "ror": f"{target} = (C << 7) | ({target} >> 1)",
        }[mnem]
        return operation
    if mnem in _BRANCH_CONDITIONS:
        return f"if ({_BRANCH_CONDITIONS[mnem]}) goto {_target(op)}"
    if mnem == "jmp":
        return f"goto {_target(op)}"
    if mnem == "jsr":
        return f"call {_target(op)}"
    if mnem == "rts":
        return "return"
    if mnem == "rti":
        return "return_from_interrupt"
    if mnem == "brk":
        return "software_interrupt()"
    if mnem in _FLAG_OPS:
        return _FLAG_OPS[mnem]
    if mnem in _TRANSFERS:
        return _TRANSFERS[mnem]
    if mnem == "pha":
        return "push(A)"
    if mnem == "pla":
        return "A = pop()"
    if mnem == "php":
        return "push(P)"
    if mnem == "plp":
        return "P = pop()"
    if mnem == "nop":
        return "no_op"
    assembly = f"{mnem.upper()} {op}".rstrip()
    return f'asm("{assembly}")'


def render_mechanical_pseudocode(
    instructions: list[dict[str, Any]],
    *,
    routine_name: str | None = None,
) -> str:
    """Render an honest one-source-line-to-one-pseudocode-line view."""
    title = routine_name or "routine"
    lines = [
        f"// MECHANICAL 6502 PSEUDOCODE: {title}",
        "// Reading aid only: no inferred types/control structures; addresses are authoritative.",
    ]
    for instruction in instructions:
        addr = int(instruction.get("addr") or 0) & 0xFFFF
        mnemonic = str(instruction.get("mnemonic") or "?").lower()
        operand = instruction.get("operand")
        statement = translate_instruction(mnemonic, operand)
        asm = f"{mnemonic.upper()} {_operand_text(operand)}".rstrip()
        lines.append(f"L_{addr:04X}: {statement};  // ${addr:04X} {asm}")
    return "\n".join(lines)
