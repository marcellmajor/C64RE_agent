"""Ensure the project root is importable regardless of pytest invocation dir."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def synthetic_c64_dump(tmp_path) -> Path:
    """Sparse synthetic 64 KiB dump shared by deterministic-layer tests.

    It contains a BASIC ``SYS 49152`` stub and a tiny two-routine program
    whose ``STA $C000`` is intentionally self-modifying.
    """
    mem = bytearray(0x10000)
    mem[0x0001] = 0x37

    # BASIC line 10: SYS 49152 (token $9E), final line link = $0000.
    mem[0x0801:0x0805] = b"\x00\x00\x0a\x00"
    body = b"\x9e 49152\x00"
    mem[0x0805:0x0805 + len(body)] = body

    # $C000: LDA #1 / STA $C000 / JSR $C010 / RTS
    mem[0xC000:0xC00A] = bytes([
        0xA9, 0x01,
        0x8D, 0x00, 0xC0,
        0x20, 0x10, 0xC0,
        0x60, 0xEA,
    ])
    mem[0xC010:0xC012] = bytes([0xE6, 0xC0])  # INC $C0
    mem[0xC012] = 0x60                         # RTS

    path = tmp_path / "synthetic_64k.dump"
    path.write_bytes(bytes(mem))
    return path
