"""Differential memory analysis — the "cheat-engine" workflow (tracker 3.1).

For the archetypal questions this agent exists for ("where is the lives /
score / level counter stored?"), the human workflow is not disassembly —
it is differential memory analysis: snapshot RAM, change the quantity
in-game, snapshot again, diff, repeat to intersect candidate sets.

Everything here is pure and offline: it operates on byte snapshots (the
ingested dump is itself one snapshot; a live VICE read is another). The
`graph.nodes` VICE composite sources the snapshots; these functions do
the analysis and are fully unit-testable without an emulator.
"""

from __future__ import annotations

from typing import Any

# C64 memory regions, most-specific first. Classification helps rank
# candidates: game state usually lives in zero page / low RAM, never in
# I/O or ROM shadow.
_REGIONS: tuple[tuple[int, int, str], ...] = (
    (0x0000, 0x0001, "processor_port"),
    (0x0002, 0x00FF, "zero_page"),
    (0x0100, 0x01FF, "stack"),
    (0x0200, 0x03FF, "os_workspace"),
    (0x0400, 0x07E7, "screen_ram"),
    (0x07E8, 0x07FF, "sprite_pointers"),
    (0x0800, 0x9FFF, "low_ram"),
    (0xA000, 0xBFFF, "basic_rom_or_ram"),
    (0xC000, 0xCFFF, "upper_ram"),
    (0xD000, 0xD3FF, "io_vic"),
    (0xD400, 0xD7FF, "io_sid"),
    (0xD800, 0xDBFF, "color_ram"),
    (0xDC00, 0xDCFF, "io_cia1"),
    (0xDD00, 0xDDFF, "io_cia2"),
    (0xDE00, 0xDFFF, "io_expansion"),
    (0xE000, 0xFFFF, "kernal_rom_or_ram"),
)

# Regions where a mutating game-state variable most plausibly lives.
_STATE_REGIONS = frozenset({
    "zero_page", "os_workspace", "low_ram", "upper_ram", "screen_ram",
})


def classify_region(addr: int) -> str:
    for lo, hi, name in _REGIONS:
        if lo <= addr <= hi:
            return name
    return "unknown"


def diff_snapshots(
    a: bytes,
    b: bytes,
    *,
    regions: frozenset[str] | None = None,
    exclude_io: bool = True,
    max_results: int = 4096,
) -> list[dict[str, Any]]:
    """Report addresses whose byte changed from *a* to *b*.

    Each row: ``{addr, addr_hex, old, new, delta, region}``. ``delta`` is
    the signed 8-bit difference (new − old, wrapped to −128..127) — the
    signal `monotonic_scan` keys on. ``exclude_io`` drops volatile I/O
    registers (raster line, timers) that change every frame regardless of
    game state; pass ``regions`` to restrict to specific region names.
    """
    n = min(len(a), len(b))
    out: list[dict[str, Any]] = []
    for addr in range(n):
        old, new = a[addr], b[addr]
        if old == new:
            continue
        region = classify_region(addr)
        if regions is not None and region not in regions:
            continue
        if exclude_io and region.startswith("io_"):
            continue
        delta = ((new - old + 128) % 256) - 128
        out.append({
            "addr": addr, "addr_hex": f"${addr:04X}",
            "old": old, "new": new, "delta": delta, "region": region,
        })
        if len(out) >= max_results:
            break
    return out


def monotonic_scan(
    snapshots: list[bytes],
    *,
    delta: int = -1,
    regions: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Addresses that changed by exactly *delta* across EVERY consecutive
    snapshot pair (tracker 3.1).

    The classic lives-counter finder: take a snapshot each time you lose a
    life; the lives byte is the one that decreased by 1 every single time
    while thousands of noise bytes changed inconsistently. Requires ≥2
    snapshots. ``regions`` defaults to plausible game-state regions.
    """
    if len(snapshots) < 2:
        return []
    region_filter = regions if regions is not None else _STATE_REGIONS

    n = min(len(s) for s in snapshots)
    # Candidate set = addresses matching `delta` on the FIRST transition,
    # then intersected across the rest.
    survivors: dict[int, list[int]] = {}
    first_a, first_b = snapshots[0], snapshots[1]
    for addr in range(n):
        region = classify_region(addr)
        if region not in region_filter:
            continue
        d = ((first_b[addr] - first_a[addr] + 128) % 256) - 128
        if d == delta:
            survivors[addr] = [first_a[addr], first_b[addr]]

    for i in range(1, len(snapshots) - 1):
        prev, cur = snapshots[i], snapshots[i + 1]
        still: dict[int, list[int]] = {}
        for addr, seq in survivors.items():
            d = ((cur[addr] - prev[addr] + 128) % 256) - 128
            if d == delta:
                still[addr] = seq + [cur[addr]]
        survivors = still
        if not survivors:
            break

    return [
        {
            "addr": addr, "addr_hex": f"${addr:04X}",
            "region": classify_region(addr),
            "delta": delta, "values": seq,
        }
        for addr, seq in sorted(survivors.items())
    ]


def summarize_diff(rows: list[dict[str, Any]], *, top: int = 40) -> str:
    """Human-readable diff summary, grouped by region."""
    if not rows:
        return "No differing bytes (in the scanned regions)."
    by_region: dict[str, int] = {}
    for r in rows:
        by_region[r["region"]] = by_region.get(r["region"], 0) + 1
    header = (
        f"{len(rows)} changed byte(s); by region: "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_region.items()))
    )
    lines = [
        f"  {r['addr_hex']} ({r['region']}): {r['old']} → {r['new']} "
        f"(Δ{r['delta']:+d})"
        for r in rows[:top]
    ]
    if len(rows) > top:
        lines.append(f"  … {len(rows) - top} more")
    return header + "\n" + "\n".join(lines)
