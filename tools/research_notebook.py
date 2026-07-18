"""Durable multi-turn notebook and frozen dump catalog services."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TURN_ARCHIVE = "turns.jsonl"
CATALOG_DIR = "dumps"
CATALOG_FILE = "catalog.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_dump_state_name(value: str) -> str:
    """Return the canonical, path-safe key used for frozen dump states."""
    name = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    if not name:
        raise ValueError("state name must contain a letter or number")
    return name[:64]


def versioned_report_path(
    session_dir: str | Path, *, started_at: str | None, run_id: str,
) -> Path:
    """Return a stable report_<timestamp>.md path for one run."""
    try:
        dt = datetime.fromisoformat(str(started_at or "").replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        dt = datetime.now(timezone.utc)
    stamp = dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = re.sub(r"[^a-zA-Z0-9]", "", run_id)[:8] or "run"
    return Path(session_dir) / f"report_{stamp}_{suffix}.md"


def load_turns(session_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(session_dir) / TURN_ARCHIVE
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def append_turn(session_dir: str | Path, record: dict[str, Any]) -> bool:
    """Append one run idempotently; return True only for a new row."""
    root = Path(session_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = str(record.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("turn archive record requires run_id")
    if any(str(row.get("run_id")) == run_id for row in load_turns(root)):
        return False
    row = {**record, "archived_at": record.get("archived_at") or _utc_now()}
    with (root / TURN_ARCHIVE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return True


def prior_answers_digest(
    session_dir: str | Path, *, current_question: str = "", limit: int = 6,
    max_chars: int = 8_000,
) -> str:
    """Render prior accepted answers and their unresolved questions."""
    turns = [
        row for row in load_turns(session_dir)
        if str(row.get("verdict") or "").lower() == "accept"
        and str(row.get("question") or "") != current_question
    ][-max(1, int(limit)):]
    if not turns:
        return ""
    lines = ["## Prior accepted answers and open questions"]
    for row in reversed(turns):
        answer = re.sub(r"\s+", " ", str(row.get("answer") or "")).strip()[:700]
        lines.append(
            f"- **{row.get('question') or '(unknown question)'}** "
            f"(confidence={float(row.get('confidence') or 0.0):.2f}, "
            f"run={row.get('run_id')})\n"
            f"  answer: {answer or '(no answer)'}"
        )
        for question in (row.get("open_questions") or [])[:5]:
            lines.append(f"  open: {question}")
    return "\n".join(lines)[:max_chars]


def _catalog_paths(session_dir: str | Path) -> tuple[Path, Path]:
    root = Path(session_dir) / CATALOG_DIR
    return root, root / CATALOG_FILE


def load_dump_catalog(session_dir: str | Path) -> list[dict[str, Any]]:
    _root, path = _catalog_paths(session_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("states") if isinstance(payload, dict) else None
    return [dict(row) for row in rows or [] if isinstance(row, dict)]


def _write_catalog(session_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    root, path = _catalog_paths(session_dir)
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": 1, "states": rows}, indent=2, sort_keys=True))
    tmp.replace(path)


def dump_state_path(
    session_dir: str | Path, row: dict[str, Any],
) -> Path:
    """Resolve new catalog-relative and legacy path entries portably."""
    root = Path(session_dir)
    raw_text = str(row.get("path") or "").strip()
    if not raw_text:
        return root / CATALOG_DIR / f"{row.get('name') or 'unknown'}.bin"
    raw = Path(raw_text)
    if raw.is_absolute():
        return raw
    portable = root / raw
    if portable.is_file():
        return portable
    if raw.is_file():  # legacy path already rooted at sessions/<game>/
        return raw
    return root / CATALOG_DIR / raw.name


def freeze_dump_state(
    session_dir: str | Path, *, name: str, source_path: str | Path,
    description: str = "",
) -> dict[str, Any]:
    """Copy one immutable 64 KiB dump into the named state catalog."""
    source = Path(source_path)
    data = source.read_bytes()
    if len(data) != 0x10000:
        raise ValueError(
            f"named dump states must be exactly 65536 bytes; got {len(data)}",
        )
    state_name = normalize_dump_state_name(name)
    digest = hashlib.sha256(data).hexdigest()
    root, _path = _catalog_paths(session_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = load_dump_catalog(session_dir)
    existing = next((row for row in rows if row.get("name") == state_name), None)
    if existing:
        if existing.get("sha256") != digest:
            raise ValueError(
                f"dump state {state_name!r} is frozen with different bytes",
            )
        return existing
    destination = root / f"{state_name}.bin"
    shutil.copyfile(source, destination)
    row = {
        "name": state_name,
        "description": str(description or ""),
        "path": str(Path(CATALOG_DIR) / destination.name),
        "source_path": str(source),
        "size": len(data),
        "sha256": digest,
        "captured_at": _utc_now(),
    }
    rows.append(row)
    _write_catalog(session_dir, rows)
    return row


def diff_dump_states(
    session_dir: str | Path, *, before: str, after: str,
    exclude_io: bool = True,
) -> list[dict[str, Any]]:
    from tools.mem_diff import diff_snapshots

    rows = {str(row.get("name")): row for row in load_dump_catalog(session_dir)}
    before_name = normalize_dump_state_name(before)
    after_name = normalize_dump_state_name(after)
    if before_name not in rows or after_name not in rows:
        missing = [name for name in (before_name, after_name) if name not in rows]
        raise KeyError("unknown dump state(s): " + ", ".join(missing))
    a = dump_state_path(session_dir, rows[before_name]).read_bytes()
    b = dump_state_path(session_dir, rows[after_name]).read_bytes()
    return diff_snapshots(a, b, exclude_io=exclude_io)
