"""Verify game-scoped asm-file selection against the real asm_dir/.

The current asm_dir holds traces from three different games:

    blood_loaded_0803.asm.txt   → Captain Blood
    bubbob_fullmem_f0e8.asm.txt → Bubble Bobble
    bubbob_manual.asm           → Bubble Bobble
    vultures_disassembly.txt    → Vultures

Each game name should pull only its own files via the slug-token
heuristic; an unknown game should pull nothing (so a fresh session can
never be silently contaminated).

Run with:
    .venv/bin/python -m scripts.smoke_scoping
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code_kb import select_asm_files, candidate_dumps, game_tokens


def show(game: str) -> None:
    s = select_asm_files(game=game, asm_dir=ROOT / "asm_dir")
    print(f"\n=== {game!r}  (slug-tokens={s.tokens}) ===")
    print(f"  strategy: {s.strategy}")
    print(f"  suggestion: {s.suggestion}")
    print("  selected:")
    for p in s.selected:
        print(f"    + {p.name}")
    print("  skipped:")
    for p, reason in s.skipped[:6]:
        print(f"    - {p.name} ({reason})")
    matched, others = candidate_dumps(ROOT / "memdump_dir", game=game)
    print("  dump matches:")
    for p in matched:
        print(f"    ★ {p.name}")
    for p in others:
        print(f"      {p.name}")


def main() -> None:
    print("Asm dir:", ROOT / "asm_dir")
    print("Memdump dir:", ROOT / "memdump_dir")
    show("Captain Blood")
    show("Bubble Bobble")    # tokens won't match `bubbob_*`; expect empty
    show("Bubbob")           # nickname → matches
    show("Vultures")
    show("Boulder Dash")     # absent → empty + actionable hint

    # Override path: explicit list scopes correctly even when tokens miss.
    print("\n=== override (Bubble Bobble + bubbob_*.asm) ===")
    s = select_asm_files(
        game="Bubble Bobble",
        asm_dir=ROOT / "asm_dir",
        override=["bubbob_manual.asm", "bubbob_fullmem_f0e8.asm.txt"],
    )
    print(f"  strategy={s.strategy}")
    print("  selected:", [p.name for p in s.selected])
    assert s.strategy == "override"
    assert len(s.selected) == 2

    # Token assertion — Captain Blood should pick only the blood file.
    s = select_asm_files(game="Captain Blood", asm_dir=ROOT / "asm_dir")
    assert s.strategy == "heuristic"
    names = sorted(p.name for p in s.selected)
    assert names == ["blood_loaded_0803.asm.txt"], names

    # Vultures.
    s = select_asm_files(game="Vultures", asm_dir=ROOT / "asm_dir")
    assert s.strategy == "heuristic"
    assert sorted(p.name for p in s.selected) == ["vultures_disassembly.txt"]

    # Empty selection for an unknown game (instead of silently slurping all).
    s = select_asm_files(game="Boulder Dash", asm_dir=ROOT / "asm_dir")
    assert s.strategy == "empty"
    assert s.selected == []

    print("\nAll scoping assertions passed.")


if __name__ == "__main__":
    main()
