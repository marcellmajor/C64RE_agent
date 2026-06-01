"""OpenAI-compatible embedding HTTP client (any endpoint with `/v1/embeddings`)."""

from __future__ import annotations

from typing import Any

import httpx


def embeddings_create(
    *,
    base_url: str,
    api_key: str,
    model: str,
    inputs: list[str],
    dimensions: int | None,
    timeout_s: float,
) -> list[list[float]]:
    """Return one embedding vector per input string."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        url = f"{root}/embeddings"
    else:
        url = f"{root}/v1/embeddings"

    bodies: dict[str, Any] = {"model": model, "input": inputs}
    if dimensions is not None:
        bodies["dimensions"] = int(dimensions)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=bodies, headers=headers)

    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        snippet = ""
        try:
            snippet = (r.text or "")[:400]
        except Exception:
            snippet = ""
        raise RuntimeError(
            f"Embedding HTTP {r.status_code} at {url}: {snippet}",
        ) from e

    data = r.json()
    raw = data.get("data") or []
    if len(raw) != len(inputs):
        raise RuntimeError(
            f"Embedding API returned {len(raw)} rows for {len(inputs)} inputs.",
        )

    keyed = [(item.get("index", i), item.get("embedding")) for i, item in enumerate(raw)]
    keyed.sort(key=lambda x: x[0])
    out: list[list[float]] = []
    for _, emb in keyed:
        if emb is None or not isinstance(emb, list):
            raise RuntimeError("Embedding response missing 'embedding' floats.")
        out.append([float(x) for x in emb])

    for v in out:
        if dimensions is not None and len(v) != int(dimensions):
            raise RuntimeError(
                f"Expected dimension {dimensions}, got {len(v)} for model {model!r}.",
            )
    return out
