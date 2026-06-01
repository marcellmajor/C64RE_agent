"""Curated C64 hardware knowledge pack.

This pack is preloaded into Layer 1 prompts so the LLM never has to guess
what `STA $D020` or `JSR $FFD2` means. Coverage is intentionally
*limited but accurate*: only chip registers and KERNAL entries we are
confident about, sourced from the Pickering/Bauer KERNAL ROM
disassembly, the "Mapping the C64" book, and codebase64 references.

Two consumers:
    1. `lookup(addr)` — used by the deterministic Layer-0 classifier and
       by the disasm exporters.
    2. `pack_text()` — emits a compact markdown pack the Layer-1 LLM can
       paste into its system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HwEntry:
    addr: int
    name: str
    role: str
    chip: str  # "VIC-II" | "SID" | "CIA1" | "CIA2" | "KERNAL" | "BASIC" | "ZP" | "RAM"


# ---- VIC-II ($D000-$D02E) -------------------------------------------------- #
# Register names from the Commodore 64 Programmer's Reference Guide.

VIC_II: tuple[HwEntry, ...] = tuple(
    HwEntry(addr, name, role, "VIC-II")
    for addr, name, role in [
        (0xD000, "SP0X",  "Sprite 0 X-coordinate (low 8 bits)"),
        (0xD001, "SP0Y",  "Sprite 0 Y-coordinate"),
        (0xD002, "SP1X",  "Sprite 1 X-coordinate"),
        (0xD003, "SP1Y",  "Sprite 1 Y-coordinate"),
        (0xD004, "SP2X",  "Sprite 2 X-coordinate"),
        (0xD005, "SP2Y",  "Sprite 2 Y-coordinate"),
        (0xD006, "SP3X",  "Sprite 3 X-coordinate"),
        (0xD007, "SP3Y",  "Sprite 3 Y-coordinate"),
        (0xD008, "SP4X",  "Sprite 4 X-coordinate"),
        (0xD009, "SP4Y",  "Sprite 4 Y-coordinate"),
        (0xD00A, "SP5X",  "Sprite 5 X-coordinate"),
        (0xD00B, "SP5Y",  "Sprite 5 Y-coordinate"),
        (0xD00C, "SP6X",  "Sprite 6 X-coordinate"),
        (0xD00D, "SP6Y",  "Sprite 6 Y-coordinate"),
        (0xD00E, "SP7X",  "Sprite 7 X-coordinate"),
        (0xD00F, "SP7Y",  "Sprite 7 Y-coordinate"),
        (0xD010, "MSIGX", "Sprite X high-bit (bit n = sprite n)"),
        (0xD011, "SCROLY", "Vert. scroll / display ctrl (bit7=raster MSB, bit5=BMM, bit4=DEN, bit3=RSEL, bit2-0=YSCROLL)"),
        (0xD012, "RASTER", "Raster line counter (read) / IRQ trigger line (write, low 8 bits)"),
        (0xD013, "LPENX", "Light-pen X (bits 8..1)"),
        (0xD014, "LPENY", "Light-pen Y"),
        (0xD015, "SPENA", "Sprite enable mask"),
        (0xD016, "SCROLX", "Horiz. scroll / display ctrl (bit4=MCM, bit3=CSEL, bit2-0=XSCROLL)"),
        (0xD017, "YXPAND", "Sprite Y double-height mask"),
        (0xD018, "VMCSB", "Video matrix base + char-set base (bits 7-4 = screen page, bits 3-1 = charset)"),
        (0xD019, "VICIRQ", "VIC IRQ status (read clears) — bits: RST,SBC,SSC,LP"),
        (0xD01A, "IRQMSK", "VIC IRQ enable mask"),
        (0xD01B, "SPBGPR", "Sprite-to-background priority (1 = sprite behind chars)"),
        (0xD01C, "SPMC",   "Sprite multicolor enable mask"),
        (0xD01D, "XXPAND", "Sprite X double-width mask"),
        (0xD01E, "SPSPCL", "Sprite-sprite collision latch"),
        (0xD01F, "SPBGCL", "Sprite-background collision latch"),
        (0xD020, "EXTCOL", "Border colour"),
        (0xD021, "BGCOL0", "Background colour 0"),
        (0xD022, "BGCOL1", "Background colour 1 (multicolor / EBCM)"),
        (0xD023, "BGCOL2", "Background colour 2 (multicolor / EBCM)"),
        (0xD024, "BGCOL3", "Background colour 3 (EBCM only)"),
        (0xD025, "SPMC0",  "Sprite multicolor 0"),
        (0xD026, "SPMC1",  "Sprite multicolor 1"),
        (0xD027, "SP0COL", "Sprite 0 individual colour"),
        (0xD028, "SP1COL", "Sprite 1 individual colour"),
        (0xD029, "SP2COL", "Sprite 2 individual colour"),
        (0xD02A, "SP3COL", "Sprite 3 individual colour"),
        (0xD02B, "SP4COL", "Sprite 4 individual colour"),
        (0xD02C, "SP5COL", "Sprite 5 individual colour"),
        (0xD02D, "SP6COL", "Sprite 6 individual colour"),
        (0xD02E, "SP7COL", "Sprite 7 individual colour"),
    ]
)

# ---- SID ($D400-$D41C) ---------------------------------------------------- #

SID: tuple[HwEntry, ...] = tuple(
    HwEntry(addr, name, role, "SID")
    for addr, name, role in [
        (0xD400, "FRELO1", "Voice 1 frequency low"),
        (0xD401, "FREHI1", "Voice 1 frequency high"),
        (0xD402, "PWLO1",  "Voice 1 pulse-width low"),
        (0xD403, "PWHI1",  "Voice 1 pulse-width high (bits 3-0)"),
        (0xD404, "VCREG1", "Voice 1 control: GATE,SYNC,RING,TEST,TRI,SAW,PULSE,NOISE"),
        (0xD405, "ATDCY1", "Voice 1 attack/decay"),
        (0xD406, "SUREL1", "Voice 1 sustain/release"),
        (0xD407, "FRELO2", "Voice 2 frequency low"),
        (0xD408, "FREHI2", "Voice 2 frequency high"),
        (0xD409, "PWLO2",  "Voice 2 pulse-width low"),
        (0xD40A, "PWHI2",  "Voice 2 pulse-width high (bits 3-0)"),
        (0xD40B, "VCREG2", "Voice 2 control"),
        (0xD40C, "ATDCY2", "Voice 2 attack/decay"),
        (0xD40D, "SUREL2", "Voice 2 sustain/release"),
        (0xD40E, "FRELO3", "Voice 3 frequency low"),
        (0xD40F, "FREHI3", "Voice 3 frequency high"),
        (0xD410, "PWLO3",  "Voice 3 pulse-width low"),
        (0xD411, "PWHI3",  "Voice 3 pulse-width high"),
        (0xD412, "VCREG3", "Voice 3 control"),
        (0xD413, "ATDCY3", "Voice 3 attack/decay"),
        (0xD414, "SUREL3", "Voice 3 sustain/release"),
        (0xD415, "CUTLO",  "Filter cutoff frequency low (bits 2-0)"),
        (0xD416, "CUTHI",  "Filter cutoff frequency high"),
        (0xD417, "RESON",  "Filter resonance/voice routing"),
        (0xD418, "SIGVOL", "Volume + filter mode (bits 3-0 = master volume)"),
        (0xD419, "POTX",   "Paddle X (read)"),
        (0xD41A, "POTY",   "Paddle Y (read)"),
        (0xD41B, "RANDOM", "Voice 3 oscillator (read) — common pseudo-random source"),
        (0xD41C, "ENV3",   "Voice 3 envelope generator (read)"),
    ]
)

# ---- CIA1 ($DC00-$DC0F) and CIA2 ($DD00-$DD0F) ---------------------------- #

def _cia_block(base: int, chip: str) -> tuple[HwEntry, ...]:
    return tuple(
        HwEntry(base + off, name, role, chip)
        for off, name, role in [
            (0x00, f"{chip}_PRA",  "Port A (joystick / kbd / VIC bank ctrl on CIA2)"),
            (0x01, f"{chip}_PRB",  "Port B (joystick / kbd / RS-232 on CIA2)"),
            (0x02, f"{chip}_DDRA", "Port A direction"),
            (0x03, f"{chip}_DDRB", "Port B direction"),
            (0x04, f"{chip}_TALO", "Timer A low"),
            (0x05, f"{chip}_TAHI", "Timer A high"),
            (0x06, f"{chip}_TBLO", "Timer B low"),
            (0x07, f"{chip}_TBHI", "Timer B high"),
            (0x08, f"{chip}_TOD10","TOD 1/10 sec"),
            (0x09, f"{chip}_TODSE","TOD seconds"),
            (0x0A, f"{chip}_TODMI","TOD minutes"),
            (0x0B, f"{chip}_TODHR","TOD hours"),
            (0x0C, f"{chip}_SDR",  "Serial shift register"),
            (0x0D, f"{chip}_ICR",  "Interrupt control / status"),
            (0x0E, f"{chip}_CRA",  "Timer A control"),
            (0x0F, f"{chip}_CRB",  "Timer B control"),
        ]
    )


CIA1 = _cia_block(0xDC00, "CIA1")
CIA2 = _cia_block(0xDD00, "CIA2")


# ---- KERNAL jump table ($FF81-$FFF6) -------------------------------------- #
# Documented entry points only.

KERNAL: tuple[HwEntry, ...] = tuple(
    HwEntry(addr, name, role, "KERNAL")
    for addr, name, role in [
        (0xFF81, "CINT",    "Initialize VIC and screen editor"),
        (0xFF84, "IOINIT",  "Initialize CIAs, IRQ, timers"),
        (0xFF87, "RAMTAS",  "RAM test, set pointers"),
        (0xFF8A, "RESTOR",  "Restore default I/O vectors"),
        (0xFF8D, "VECTOR",  "Read/set I/O vectors"),
        (0xFF90, "SETMSG",  "Set kernal message control"),
        (0xFF93, "SECOND",  "Send secondary address after LISTEN"),
        (0xFF96, "TKSA",    "Send secondary address after TALK"),
        (0xFF99, "MEMTOP",  "Read/set top of memory"),
        (0xFF9C, "MEMBOT",  "Read/set bottom of memory"),
        (0xFF9F, "SCNKEY",  "Scan keyboard"),
        (0xFFA2, "SETTMO",  "Set IEEE timeout"),
        (0xFFA5, "ACPTR",   "Receive byte from serial bus"),
        (0xFFA8, "CIOUT",   "Send byte to serial bus"),
        (0xFFAB, "UNTLK",   "Send UNTALK"),
        (0xFFAE, "UNLSN",   "Send UNLISTEN"),
        (0xFFB1, "LISTEN",  "Send LISTEN"),
        (0xFFB4, "TALK",    "Send TALK"),
        (0xFFB7, "READST",  "Read I/O status word"),
        (0xFFBA, "SETLFS",  "Set logical, first, and second addresses"),
        (0xFFBD, "SETNAM",  "Set filename"),
        (0xFFC0, "OPEN",    "Open a logical file"),
        (0xFFC3, "CLOSE",   "Close a logical file"),
        (0xFFC6, "CHKIN",   "Open channel for input"),
        (0xFFC9, "CHKOUT",  "Open channel for output"),
        (0xFFCC, "CLRCHN",  "Clear I/O channels"),
        (0xFFCF, "CHRIN",   "Get character from input channel"),
        (0xFFD2, "CHROUT",  "Print char to current output channel"),
        (0xFFD5, "LOAD",    "Load RAM from device"),
        (0xFFD8, "SAVE",    "Save RAM to device"),
        (0xFFDB, "SETTIM",  "Set TIME-OF-DAY clock"),
        (0xFFDE, "RDTIM",   "Read TIME-OF-DAY clock"),
        (0xFFE1, "STOP",    "Test STOP key"),
        (0xFFE4, "GETIN",   "Get character from queue (non-blocking CHRIN)"),
        (0xFFE7, "CLALL",   "Close all I/O channels"),
        (0xFFEA, "UDTIM",   "Update TIME-OF-DAY clock"),
        (0xFFED, "SCREEN",  "Get screen size"),
        (0xFFF0, "PLOT",    "Read/set cursor X/Y"),
        (0xFFF3, "IOBASE",  "Return I/O base address"),
    ]
)

# Common BASIC ROM internal entry points called from games.
BASIC: tuple[HwEntry, ...] = tuple(
    HwEntry(addr, name, role, "BASIC")
    for addr, name, role in [
        (0xA000, "BASIC_COLDSTART", "BASIC cold-start vector"),
        (0xA002, "BASIC_WARMSTART", "BASIC warm-start vector"),
        (0xA483, "BASIC_MAINLOOP",  "BASIC immediate-mode loop"),
        (0xAB1E, "STROUT",          "Print null-terminated string at A/Y"),
        (0xBDCD, "LINPRT",          "Print 16-bit unsigned integer in A/X"),
        (0xE544, "CLRSCN",          "Clear screen"),
        (0xE566, "HOME",            "Cursor home"),
    ]
)

# RAM vectors / zero-page constants the agent must know.
RAM_VECTORS: tuple[HwEntry, ...] = (
    HwEntry(0x0001, "PROC_PORT",   "Processor port — bank-switching ($A000+/$D000+/$E000+)", "ZP"),
    HwEntry(0x0314, "CINV",        "IRQ vector (low/high) — main-loop trampoline in many games", "RAM"),
    HwEntry(0x0316, "CBINV",       "BRK vector",                                                "RAM"),
    HwEntry(0x0318, "NMINV",       "NMI vector — RESTORE key / Datasette",                     "RAM"),
    HwEntry(0x0090, "STATUS",      "I/O status byte",                                          "ZP"),
    HwEntry(0x00C5, "LSTX",        "Last keyboard scan code",                                  "ZP"),
    HwEntry(0x00C6, "NDX",         "Number of chars in keyboard buffer",                       "ZP"),
    HwEntry(0x00CB, "SFDX",        "Current key pressed",                                      "ZP"),
    HwEntry(0x00FB, "ZP_FB",       "Conventional pointer pair (low byte)",                     "ZP"),
    HwEntry(0x00FC, "ZP_FC",       "Conventional pointer pair (high byte)",                    "ZP"),
    HwEntry(0x00FD, "ZP_FD",       "Conventional pointer pair (low byte)",                     "ZP"),
    HwEntry(0x00FE, "ZP_FE",       "Conventional pointer pair (high byte)",                    "ZP"),
    HwEntry(0xFFFA, "NMI_VEC",     "HW NMI vector (after KERNAL banked out)",                  "RAM"),
    HwEntry(0xFFFC, "RESET_VEC",   "HW reset vector",                                          "RAM"),
    HwEntry(0xFFFE, "IRQ_VEC",     "HW IRQ/BRK vector",                                        "RAM"),
)


_ALL: tuple[HwEntry, ...] = VIC_II + SID + CIA1 + CIA2 + KERNAL + BASIC + RAM_VECTORS
_BY_ADDR: dict[int, HwEntry] = {e.addr: e for e in _ALL}


def lookup(addr: int) -> HwEntry | None:
    """Return the curated hardware entry for `addr`, or None."""
    return _BY_ADDR.get(int(addr) & 0xFFFF)


def lookup_range(addr: int) -> HwEntry | None:
    """Like `lookup` but returns the *enclosing* chip range entry as a fallback.

    Useful when a write hits e.g. $D027 (sprite 0 colour) but you only have
    a curated label for the chip start; the caller can still tag the
    annotation with `chip='VIC-II'`.
    """
    e = lookup(addr)
    if e:
        return e
    a = int(addr) & 0xFFFF
    if 0xD000 <= a < 0xD400:
        return HwEntry(a, "VIC?",  "VIC-II register window", "VIC-II")
    if 0xD400 <= a < 0xD800:
        return HwEntry(a, "SID?",  "SID register window",    "SID")
    if 0xDC00 <= a < 0xDD00:
        return HwEntry(a, "CIA1?", "CIA1 register window",   "CIA1")
    if 0xDD00 <= a < 0xDE00:
        return HwEntry(a, "CIA2?", "CIA2 register window",   "CIA2")
    if 0xE000 <= a <= 0xFFFF:
        return HwEntry(a, "KERNAL?", "KERNAL ROM region",    "KERNAL")
    if 0xA000 <= a <= 0xBFFF:
        return HwEntry(a, "BASIC?", "BASIC ROM region",      "BASIC")
    return None


def pack_text(max_chars: int = 6_500) -> str:
    """Return a compact markdown pack suitable for a Layer-1 system prompt."""

    def _table(title: str, entries: tuple[HwEntry, ...]) -> str:
        lines = [f"### {title}"]
        for e in entries:
            lines.append(f"- ${e.addr:04X}  {e.name:<10}  {e.role}")
        return "\n".join(lines)

    blocks = [
        _table("VIC-II ($D000-$D02E)", VIC_II),
        _table("SID ($D400-$D41C)", SID),
        _table("CIA1 ($DC00-$DC0F)", CIA1),
        _table("CIA2 ($DD00-$DD0F)", CIA2),
        _table("KERNAL jump table ($FF81-$FFF3)", KERNAL),
        _table("BASIC ROM (selected)", BASIC),
        _table("RAM vectors / zero-page conventions", RAM_VECTORS),
    ]
    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        text = text[: max_chars - 32] + "\n... [hardware pack truncated]"
    return text


def chip_for(addr: int) -> str | None:
    e = lookup_range(addr)
    return e.chip if e else None
