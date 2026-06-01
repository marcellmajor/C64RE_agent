"""Purge persistent C64-RE data.

Each game has TWO knowledge bases under sessions/<slug>/:
  * kb/        — the parent agent's general KB (events, labels, hypotheses)
  * code_kb/   — the layered code-comprehension store (routines, xrefs, …)
Plus an optional `report.md`. `--game` and `--all` always wipe both KBs
because they live inside the game's session directory.

Other persistence:
  * sessions/checkpoints.sqlite      — LangGraph SQLite checkpointer
  * .langgraph_api/store*.pckl       — LangGraph Studio (`langgraph dev`)
                                       in-memory store snapshots

Examples:
    python scripts/purge_persistence.py --game "Bubble Bobble" --dry-run
    python scripts/purge_persistence.py --game "Bubble Bobble" --yes
    python scripts/purge_persistence.py --all --yes
    # full reset — every game, every checkpoint, every studio cache:
    python scripts/purge_persistence.py --everything --yes
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SESSIONS_DIR = ROOT_DIR / "sessions"
LANGGRAPH_API_DIR = ROOT_DIR / ".langgraph_api"


def _slug(game: str) -> str:
    return game.strip().lower().replace(" ", "_")


def _collect_targets(
    game: str | None,
    purge_all: bool,
    include_checkpoints: bool,
    include_langgraph_api: bool,
) -> list[Path]:
    targets: list[Path] = []
    if purge_all:
        if SESSIONS_DIR.exists():
            for child in sorted(SESSIONS_DIR.iterdir()):
                if child.name == "checkpoints.sqlite":
                    continue
                targets.append(child)
    elif game:
        # Whole game dir → wipes both kb/ and code_kb/ at once.
        targets.append(SESSIONS_DIR / _slug(game))

    if include_checkpoints:
        targets.append(SESSIONS_DIR / "checkpoints.sqlite")

    if include_langgraph_api and LANGGRAPH_API_DIR.exists():
        # LangGraph Studio writes pickled in-memory stores here every
        # `langgraph dev` invocation. Nuking this dir resets Studio
        # without touching agent code.
        targets.append(LANGGRAPH_API_DIR)

    # Keep stable order and remove dupes.
    unique: list[Path] = []
    seen = set()
    for p in targets:
        k = str(p)
        if k not in seen:
            seen.add(k)
            unique.append(p)
    return unique


def _delete_path(path: Path, dry_run: bool) -> tuple[bool, str]:
    if not path.exists():
        return False, f"skip (missing): {path}"

    if dry_run:
        return True, f"would delete: {path}"

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True, f"deleted: {path}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Purge persistent C64-RE sessions/KB artifacts."
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--game",
        help='Game name to purge (e.g. "Bubble Bobble"; slug auto-derived). '
             "Wipes the entire sessions/<slug>/ tree, which includes both "
             "the parent KB (kb/) and the code-comprehension KB (code_kb/).",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="Purge every game-session directory under sessions/. "
             "Does NOT remove checkpoints.sqlite or .langgraph_api/ unless "
             "--include-checkpoints / --include-langgraph-api are passed.",
    )
    scope.add_argument(
        "--everything",
        action="store_true",
        help="Total reset: --all + --include-checkpoints + "
             "--include-langgraph-api. Use this when you want to start "
             "completely from scratch.",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Also remove sessions/checkpoints.sqlite (LangGraph "
             "SQLite-backed checkpointer state).",
    )
    parser.add_argument(
        "--include-langgraph-api",
        action="store_true",
        help="Also remove .langgraph_api/ (LangGraph Studio in-memory "
             "store snapshots written by `langgraph dev`).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    if args.everything:
        args.all = True
        args.include_checkpoints = True
        args.include_langgraph_api = True

    targets = _collect_targets(
        args.game,
        args.all,
        args.include_checkpoints,
        args.include_langgraph_api,
    )
    if not targets:
        print("Nothing to purge.")
        return

    print("Targets:")
    for t in targets:
        print(f"  - {t}")

    if not args.dry_run and not args.yes:
        answer = input("Proceed with deletion? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return

    removed = 0
    skipped = 0
    for t in targets:
        changed, msg = _delete_path(t, args.dry_run)
        print(msg)
        if changed:
            removed += 1
        else:
            skipped += 1

    mode = "dry-run" if args.dry_run else "purge"
    print(f"Done ({mode}): changed={removed}, skipped={skipped}")


if __name__ == "__main__":
    main()

