"""Normalized cosine retrieval over in-memory embedding rows."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(x)
    return x / max(float(n), 1e-12)


class SemanticVectorIndex:
    """Row store + matrix multiply search (fine for KB-sized corpora)."""

    __slots__ = ("ids", "metas", "vecs", "_matrix")

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.metas: list[dict[str, Any]] = []
        self.vecs: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def clear(self) -> None:
        self.ids.clear()
        self.metas.clear()
        self.vecs.clear()
        self._matrix = None

    def invalidate_matrix(self) -> None:
        self._matrix = None

    def _stack(self) -> np.ndarray | None:
        if not self.vecs:
            self._matrix = None
            return None
        if self._matrix is None:
            self._matrix = np.stack(self.vecs, axis=0)
        return self._matrix

    def remove_predicate(self, pred: Callable[[dict[str, Any]], bool]) -> int:
        """Drop rows whose metadata satisfies `pred`. Returns removed count."""

        kept_ids: list[str] = []
        kept_metas: list[dict[str, Any]] = []
        kept_vecs: list[np.ndarray] = []
        rm = 0
        for i, m in enumerate(self.metas):
            if pred(m):
                rm += 1
                continue
            kept_ids.append(self.ids[i])
            kept_metas.append(m)
            kept_vecs.append(self.vecs[i])
        if rm:
            self.ids = kept_ids
            self.metas = kept_metas
            self.vecs = kept_vecs
            self.invalidate_matrix()
        return rm

    def add_rows(
        self,
        ids: list[str],
        metas: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> None:
        if not ids:
            return
        if len(ids) != len(metas) or len(ids) != len(vectors):
            raise ValueError("ids / metas / vectors length mismatch.")

        dim = len(vectors[0])
        acc: list[np.ndarray] = []

        for i, raw in enumerate(vectors):
            if len(raw) != dim:
                raise ValueError(
                    f"Inconsistent embedding dim: chunk {ids[i]} has {len(raw)} vs {dim}.",
                )
            acc.append(_normalize(np.asarray(raw, dtype=np.float64)))

        self.ids.extend(ids)
        self.metas.extend(metas)
        self.vecs.extend(acc)
        self.invalidate_matrix()

    def search(
        self, query_embedding: list[float], top_k: int,
    ) -> list[dict[str, Any]]:
        if not query_embedding or not self.vecs:
            return []
        k = max(1, int(top_k))
        M = self._stack()
        if M is None:
            return []

        q = _normalize(np.asarray(query_embedding, dtype=np.float64))
        if q.shape[0] != M.shape[1]:
            raise ValueError(
                f"Query dim {q.shape[0]} != index dim {M.shape[1]}.",
            )

        sims = M @ q
        kk = min(k, len(sims))
        if kk <= 0:
            return []

        part = np.argpartition(-sims, kk - 1)[:kk]
        ranked = sorted(part.tolist(), key=lambda i_: float(-sims[i_]))
        out: list[dict[str, Any]] = []
        for i_ in ranked:
            out.append({
                "id": self.ids[i_],
                "score": float(sims[i_]),
                "meta": dict(self.metas[i_]),
            })
        out.sort(key=lambda r: -r["score"])
        return out
