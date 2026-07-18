"""Shared evidence rendering for CLI, reports, and the Streamlit UI."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


_ADDRESS_RE = re.compile(
    r"^\$([0-9A-Fa-f]{1,4})(?:-\$[0-9A-Fa-f]{1,4})?$",
)
_HYPOTHESIS_RE = re.compile(r"^(h[0-9A-Za-z_]+)(?:/\w+)?$")
_EVENT_RE = re.compile(
    r"^[0-9a-f]{8,12}$|^[0-9a-f]{8}-[0-9a-f]{4}-",
    re.IGNORECASE,
)


def _event_description(kind: str, source: str, payload: dict[str, Any]) -> str:
    if kind == "label":
        addr = payload.get("addr")
        name = payload.get("name", "?")
        return (
            f"label `{name}` @ ${addr:04X}"
            if isinstance(addr, int) else f"label `{name}`"
        )
    if kind == "routine":
        start = payload.get("start")
        name = payload.get("name") or "?"
        summary = str(payload.get("summary") or "")[:120]
        text = (
            f"routine `{name}` @ ${start:04X}"
            if isinstance(start, int) else f"routine `{name}`"
        )
        return text + (f" — {summary}" if summary else "")
    if kind == "tool_result":
        tool = payload.get("tool", "?")
        step = payload.get("step_id", "?")
        data = str(payload.get("data", ""))[:120].replace("\n", " ")
        return f"tool result `{tool}/{step}`: {data}"
    if kind == "hypothesis":
        text = str(payload.get("text") or "")[:160]
        return f"hypothesis ({payload.get('status', '?')}): {text}"
    preview = json.dumps(payload, default=str)[:160]
    return f"{kind} ({source}): {preview}"


def resolve_evidence_ref(ref: Any, store: Any | None) -> str:
    """Resolve one evidence token using the parent KnowledgeStore."""
    text = str(ref).strip()
    if not text or store is None:
        return text
    if Path(text).is_absolute() or "/" in text or "\\" in text:
        return text

    address = _ADDRESS_RE.match(text)
    if address:
        addr = int(address.group(1), 16)
        rows = store.query(
            "SELECT name, kind, confidence FROM labels"
            " WHERE addr = ? ORDER BY confidence DESC LIMIT 3",
            (addr,),
        )
        if rows:
            labels = ", ".join(
                f"{row['name']} ({row.get('kind') or '?'}, "
                f"conf={float(row.get('confidence') or 0.0):.1f})"
                for row in rows
            )
            return f"{text}  →  {labels}"
        return text

    hypothesis = _HYPOTHESIS_RE.match(text)
    if hypothesis:
        hid = hypothesis.group(1)
        rows = store.query(
            "SELECT id, text, status FROM hypotheses"
            " WHERE id = ? LIMIT 1",
            (hid,),
        )
        if rows:
            row = rows[0]
            return (
                f"{text}  →  [{row.get('status')}] "
                f"{str(row.get('text') or '')[:160]}"
            )

    if _EVENT_RE.match(text):
        rows = store.query(
            "SELECT id, kind, source, payload_json FROM events"
            " WHERE id = ? LIMIT 1",
            (text,),
        )
        if not rows:
            prefix_rows = store.query(
                "SELECT id, kind, source, payload_json FROM events"
                " WHERE id LIKE ? ORDER BY id LIMIT 2",
                (f"{text}%",),
            )
            rows = prefix_rows if len(prefix_rows) == 1 else []
        if rows:
            row = rows[0]
            try:
                payload = json.loads(row.get("payload_json") or "{}")
            except json.JSONDecodeError:
                payload = {}
            return (
                f"{text}  →  "
                f"{_event_description(row.get('kind', '?'), row.get('source', '?'), payload)}"
            )
    return text


def resolve_evidence(evidence: Iterable[Any], store: Any | None) -> list[str]:
    return [
        resolved for item in evidence
        if (resolved := resolve_evidence_ref(item, store))
    ]


def format_evidence_markdown(
    evidence: Iterable[Any], store: Any | None,
) -> str:
    rows = resolve_evidence(evidence, store)
    return "\n".join(f"- {row}" for row in rows) if rows else "_none recorded_"
