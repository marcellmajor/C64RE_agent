"""Pick which partial-asm files belong to which game.

Goal: keep one game's routines from polluting another game's code-KB.
The agent's `_hydrate_code_kb` previously slurped *every* asm file in
`asm_dir/` into a single store keyed off the game slug — wrong when
that directory holds traces from several different games.

Resolution order (most-explicit wins):

  1. Caller-supplied list (`override`).
     Each entry can be absolute, or a basename relative to `asm_dir`.
  2. Per-game subdirectory: `asm_dir/<game_slug>/` exists → take its
     `*.asm` / `*.txt` / `*.s` files.
  3. Filename-token heuristic: pick files whose basename (lower-cased)
     contains *any* token derived from the game name. Tokens are the
     game's words split on whitespace / `_` / `-`, lower-cased, with
     tokens shorter than 3 characters dropped.
  4. Nothing matched → return an empty selection plus a clear hint.
     We deliberately do NOT fall back to "all files" so a fresh game
     can't be silently contaminated.

Returned `Scoping` carries:

    selected   — paths the caller should ingest.
    skipped    — files in asm_dir that were NOT selected (with reason).
    strategy   — which tier matched ("override" | "subdir" | "heuristic"
                 | "empty").
    tokens     — the matching tokens used for the heuristic (if any).
    suggestion — short human-readable hint shown in the UI / log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ASM_SUFFIXES: frozenset[str] = frozenset({".asm", ".txt", ".s"})


@dataclass
class Scoping:
    selected:   list[Path]
    skipped:    list[tuple[Path, str]] = field(default_factory=list)
    strategy:   str = "empty"
    tokens:     list[str] = field(default_factory=list)
    suggestion: str = ""


def _slug(name: str) -> str:
    """Same slugifier the rest of the agent uses (graph.plan_utils).

    Imported lazily: `code_kb` must not import the `graph` package at
    module level — `graph/__init__.py` compiles the full LangGraph,
    whose nodes import `code_kb`, so a top-level import here would be
    circular. At call time the graph is either fully imported already
    (agent runtime) or gets imported once (standalone use).
    """
    from graph.plan_utils import slugify

    return slugify(name)


def game_tokens(game: str) -> list[str]:
    """Token list used by the filename-substring heuristic."""
    s = (game or "").strip().lower()
    raw = re.split(r"[\s_\-]+", s)
    return [t for t in raw if len(t) >= 3]


def _list_asm_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in ASM_SUFFIXES
    )


def _resolve_override(
    asm_dir: Path | None, override: list[str | Path],
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Resolve user-supplied paths against `asm_dir` if relative."""
    selected: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for raw in override:
        p = Path(raw)
        candidates: list[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            if asm_dir is not None:
                candidates.append(asm_dir / p)
            candidates.append(Path.cwd() / p)
        chosen = next((c for c in candidates if c.exists()), None)
        if chosen is None:
            skipped.append((p, "not found"))
            continue
        if chosen.suffix.lower() not in ASM_SUFFIXES:
            skipped.append((chosen, f"unsupported suffix {chosen.suffix!r}"))
            continue
        selected.append(chosen.resolve())
    return selected, skipped


def select_asm_files(
    *,
    game: str,
    asm_dir: Path | str | None,
    override: list[str | Path] | None = None,
    extra_files: list[str | Path] | None = None,
) -> Scoping:
    """Decide which asm files should be ingested for one game.

    `extra_files` (e.g. ``--partial-asm``) is unioned in regardless of
    the strategy — it's an explicit "always include this".
    """
    ad = Path(asm_dir).expanduser() if asm_dir else None

    if override:
        sel, skipped = _resolve_override(ad, override)
        scoping = Scoping(
            selected=sel, skipped=skipped, strategy="override",
            tokens=game_tokens(game),
            suggestion=f"using {len(sel)} explicit asm file(s)",
        )
        _add_extra(scoping, extra_files)
        return scoping

    slug = _slug(game)
    tokens = game_tokens(game)

    # ---- tier 2: per-game subdirectory ----
    if ad is not None:
        sub = ad / slug
        if sub.is_dir():
            sub_files = _list_asm_files(sub)
            others = [p for p in _list_asm_files(ad) if sub not in p.parents]
            scoping = Scoping(
                selected=sub_files,
                skipped=[(p, "outside per-game subdir") for p in others],
                strategy="subdir",
                tokens=tokens,
                suggestion=(
                    f"loaded {len(sub_files)} file(s) from "
                    f"{sub.name}/ (per-game subdir convention)"
                ),
            )
            _add_extra(scoping, extra_files)
            return scoping

    # ---- tier 3: filename-token heuristic ----
    if ad is not None and tokens:
        all_files = _list_asm_files(ad)
        sel: list[Path] = []
        skipped: list[tuple[Path, str]] = []
        for p in all_files:
            base = p.stem.lower() + p.suffix.lower()
            if any(tok in base for tok in tokens):
                sel.append(p)
            else:
                skipped.append((
                    p, f"no token of {tokens!r} in {p.name!r}",
                ))
        if sel:
            scoping = Scoping(
                selected=sel, skipped=skipped,
                strategy="heuristic", tokens=tokens,
                suggestion=(
                    f"matched {len(sel)} file(s) by tokens "
                    f"{tokens!r}; skipped {len(skipped)} as belonging "
                    "to other games. If wrong, pass --asm-files or "
                    f"move files into {ad.name}/{slug}/."
                ),
            )
            _add_extra(scoping, extra_files)
            return scoping

    # ---- tier 4: empty + actionable hint ----
    scoping = Scoping(
        selected=[],
        skipped=[
            (p, "no game-scope match") for p in _list_asm_files(ad)
        ] if ad is not None else [],
        strategy="empty", tokens=tokens,
        suggestion=(
            "no asm files were auto-selected for this game. Either:\n"
            f"  • move files into {ad}/{slug}/, or\n"
            "  • pass --asm-files <paths> on the CLI, or\n"
            "  • pick them in the Streamlit sidebar's multi-select."
            if ad is not None else
            "no --asm-dir given and no --asm-files override; "
            "code_kb will only contain dump-derived disassembly."
        ),
    )
    _add_extra(scoping, extra_files)
    return scoping


def _add_extra(scoping: Scoping, extra_files: list[str | Path] | None) -> None:
    if not extra_files:
        return
    seen = {p.resolve() for p in scoping.selected}
    for raw in extra_files:
        p = Path(raw).expanduser()
        if not p.exists():
            scoping.skipped.append((p, "extra file not found"))
            continue
        if p.suffix.lower() not in ASM_SUFFIXES:
            scoping.skipped.append((p, "extra file unsupported suffix"))
            continue
        rp = p.resolve()
        if rp in seen:
            continue
        scoping.selected.append(rp)
        seen.add(rp)


def candidate_dumps(
    memdump_dir: Path | str | None, *, game: str,
) -> tuple[list[Path], list[Path]]:
    """Return ``(matched, others)`` for the dumps under `memdump_dir`.

    Same token logic as for asm files. The UI uses this to default the
    dump-file picker to a name-matched candidate.
    """
    if not memdump_dir:
        return [], []
    md = Path(memdump_dir).expanduser()
    if not md.is_dir():
        return [], []
    tokens = game_tokens(game)
    matched: list[Path] = []
    others: list[Path] = []
    for p in sorted(md.rglob("*")):
        if not p.is_file():
            continue
        base = p.stem.lower() + p.suffix.lower()
        if tokens and any(t in base for t in tokens):
            matched.append(p)
        else:
            others.append(p)
    return matched, others
