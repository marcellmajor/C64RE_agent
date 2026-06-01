"""Fixed-size overlapping character windows for semantic indexing."""

from __future__ import annotations

import re


_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def _split_blocks(text: str) -> list[str]:
    """Split text into semantically meaningful blocks when possible."""
    text = text or ""
    paras = [p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()]
    if len(paras) >= 2:
        return paras
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines if lines else ([text.strip()] if text.strip() else [])


def chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    """Return coherent chunks with overlap, preferring natural boundaries."""
    text = text or ""
    if not text.strip():
        return []
    size = max(64, chunk_chars)
    ov = max(0, min(overlap, size // 2))

    blocks = _split_blocks(text)
    out: list[str] = []
    cur = ""

    def _flush_cur() -> None:
        nonlocal cur
        piece = cur.strip()
        if piece:
            out.append(piece)
        cur = ""

    for block in blocks:
        b = block.strip()
        if not b:
            continue

        if len(b) > size:
            _flush_cur()
            step = max(1, size - ov)
            i = 0
            while i < len(b):
                piece = b[i : i + size].strip()
                if piece:
                    out.append(piece)
                i += step
            continue

        candidate = b if not cur else f"{cur}\n\n{b}"
        if len(candidate) <= size:
            cur = candidate
            continue

        _flush_cur()
        cur = b

    _flush_cur()

    if ov <= 0 or len(out) <= 1:
        return out

    with_overlap: list[str] = []
    for i, chunk in enumerate(out):
        if i == 0:
            with_overlap.append(chunk)
            continue
        prev_tail = out[i - 1][-ov:].strip()
        combined = (prev_tail + "\n" + chunk).strip() if prev_tail else chunk
        if len(combined) > size:
            combined = combined[-size:]
        with_overlap.append(combined)

    return with_overlap


def clip_for_embedding(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 21] + "\n... [clipped_for_embedding]"
