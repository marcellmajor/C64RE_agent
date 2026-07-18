"""Best-effort secret redaction at prompt and preview boundaries.

The append-only stores retain their original evidence. Callers use these
helpers only when rendering content into LLM context, tool results, reports,
or UI previews, so redaction never destroys reverse-engineering source data.
"""

from __future__ import annotations

import re
from typing import Any


REDACTED = "[REDACTED]"

_SECRET_KEY_RE = re.compile(
    r"(?:^|[_.-])(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"refresh[_-]?token|client[_-]?secret|secret|password|passwd|"
    r"authorization)$",
    re.IGNORECASE,
)

_ASSIGNMENT_RE = re.compile(
    r"(?P<prefix>[\"']?[A-Za-z0-9_.-]*(?:api[_-]?key|access[_-]?token|"
    r"auth[_-]?token|refresh[_-]?token|client[_-]?secret|secret|password|"
    r"passwd|authorization)[\"']?\s*[:=]\s*)"
    r"(?P<quote>[\"']?)(?P<value>\[REDACTED\]|[^\s\"'`,;\]\}]+)(?P=quote)",
    re.IGNORECASE,
)

_PREFIXED_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:sk-(?:proj-|ant-)?|xai-|tvly-|"
    r"gh[opusr]_|AIza)[A-Za-z0-9._-]{12,}",
)
_BEARER_RE = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE,
)
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}\."
    r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])",
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
    r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_URL_PASSWORD_RE = re.compile(
    r"(?P<prefix>\b[a-z][a-z0-9+.-]*://[^\s/@:]+:)"
    r"(?P<password>[^\s/@]+)(?P<suffix>@)",
    re.IGNORECASE,
)

_C64_ADDRESS_RE = re.compile(r"\$(?:[0-9A-Fa-f]{2}|[0-9A-Fa-f]{4})")


def _looks_secret_assignment_value(value: str) -> bool:
    """Require credential-shaped assignment values, not only secret keys.

    Reverse-engineering notes commonly use names such as ``secret`` for game
    state and assign them C64 addresses. The name alone is not enough evidence
    to destroy that address at a text-rendering boundary.
    """
    candidate = str(value or "").strip()
    if not candidate or candidate == REDACTED:
        return False
    if _C64_ADDRESS_RE.fullmatch(candidate):
        return False
    return bool(
        _PREFIXED_TOKEN_RE.fullmatch(candidate)
        or _JWT_RE.fullmatch(candidate)
        or (
            len(candidate) >= 24
            and re.fullmatch(r"[A-Za-z0-9._~+/=-]+", candidate)
            and re.search(r"[A-Za-z]", candidate)
            and re.search(r"[0-9]", candidate)
        )
    )


def is_secret_key(key: Any) -> bool:
    """Return whether a mapping key conventionally contains a secret."""
    return bool(_SECRET_KEY_RE.search(str(key or "")))


def redact_text(value: Any) -> str:
    """Redact common credentials while leaving ordinary C64 text intact."""
    text = str(value or "")
    text = _PRIVATE_KEY_RE.sub(REDACTED, text)
    text = _URL_PASSWORD_RE.sub(
        lambda match: match.group("prefix") + REDACTED + match.group("suffix"),
        text,
    )
    text = _BEARER_RE.sub("Bearer " + REDACTED, text)
    text = _JWT_RE.sub(REDACTED, text)
    text = _ASSIGNMENT_RE.sub(
        lambda match: (
            match.group("prefix")
            + match.group("quote")
            + REDACTED
            + match.group("quote")
            if _looks_secret_assignment_value(match.group("value"))
            else match.group(0)
        ),
        text,
    )
    text = _PREFIXED_TOKEN_RE.sub(REDACTED, text)
    return text


def redact_value(value: Any, *, key: Any | None = None) -> Any:
    """Recursively redact strings and values stored under secret-like keys."""
    if key is not None and is_secret_key(key) and value is not None:
        if isinstance(value, str) and (
            value == REDACTED or _C64_ADDRESS_RE.fullmatch(value.strip())
        ):
            return value
        return REDACTED
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            item_key: redact_value(item_value, key=item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    return value
