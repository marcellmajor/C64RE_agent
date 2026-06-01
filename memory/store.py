"""Two-tier knowledge store.

`kb.json` is the source-of-truth append-only event log. `kb.sqlite` is a
derived, queryable view rebuilt from the event log on every
`load_or_init` so it can never drift out of sync.

Per `CLAUDE.md` §2.5.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import threading
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from memory.schema import (
    CLEAR_DDL,
    EVT_DATA_STRUCTURE,
    EVT_HYPOTHESIS,
    EVT_INGEST_DUMP,
    EVT_INGEST_PARTIAL_ASM,
    EVT_INGEST_TEXT,
    EVT_LABEL,
    EVT_ROUTINE,
    EVT_TOOL_RESULT,
    SCHEMA_DDL,
)
from memory.semantic_config import load_semantic_config
from memory.semantic_documents import (
    collect_chunks_from_derived_kb,
    incremental_chunks_from_event,
    semantic_hit_preview_text,
    snippet_for_semantic,
)
from memory.semantic_embed import EmbeddingServiceError, embed_texts_batched
from memory.semantic_index import SemanticVectorIndex

# Files we accept as "user knowledge" inside `--text-dir`.
TEXT_FILE_SUFFIXES = {".txt", ".md", ".markdown", ".text", ".rst", ".asc"}

# Per-file content cap when storing into the KB (1 MiB). Large notes still
# get ingested but get truncated with a marker, keeping kb.json sane.
TEXT_FILE_MAX_BYTES = 1 * 1024 * 1024

# Module-level cache so multiple nodes in the same process share a handle.
_STORE_CACHE: dict[str, "KnowledgeStore"] = {}
_CACHE_LOCK = threading.Lock()


def get_store(handle: str | Path) -> "KnowledgeStore":
    """Return a cached `KnowledgeStore` for the given root directory."""
    key = str(Path(handle))
    with _CACHE_LOCK:
        store = _STORE_CACHE.get(key)
        if store is None:
            store = KnowledgeStore.load_or_init(Path(key))
            _STORE_CACHE[key] = store
        return store


# ----- partial-asm parsing helpers ----------------------------------------- #

# Matches lines like:  "$F0E8  LDA #$00     ; comment"   or   "F0E8: name = ..."
_LABEL_RE = re.compile(
    r"^\s*\$?(?P<addr>[0-9a-fA-F]{4})\s*[:\-]?\s*(?P<name>[A-Za-z_][\w]*)?",
)


def _parse_partial_asm(text: str) -> list[tuple[int, str]]:
    """Best-effort: return `(addr, name)` pairs from a partial-asm file."""
    found: list[tuple[int, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith((";", "#", "//")):
            continue
        m = _LABEL_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if not name:
            continue
        try:
            addr = int(m.group("addr"), 16)
        except ValueError:
            continue
        found.append((addr, name))
    return found


# --------------------------------------------------------------------------- #


class KnowledgeStore:
    """Persistent KB for one C64 game session."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.kb_json: Path = root / "kb.json"
        self.kb_sqlite: Path = root / "kb.sqlite"
        self._db: sqlite3.Connection | None = None
        self._dump_bytes: bytes | None = None
        self._lock = threading.Lock()
        # In-memory embedding index — rebuilt after SQLite replay when enabled.
        self._semantic_index: SemanticVectorIndex | None = None
        self._semantic_embed_ok: bool = False

    # ---- lifecycle ------------------------------------------------------- #

    @classmethod
    def load_or_init(cls, root: Path) -> "KnowledgeStore":
        root.mkdir(parents=True, exist_ok=True)
        store = cls(root)
        store._open_sqlite()
        store._replay_events()
        store._semantic_build_from_derived_kb()
        store._warm_dump_cache()
        return store

    def _open_sqlite(self) -> None:
        # Always rebuild the derived view so it can never drift from kb.json.
        self._db = sqlite3.connect(str(self.kb_sqlite), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        # Drop + recreate all tables so this process gets a clean derived view
        # that exactly mirrors kb.json — without unlinking the file, which
        # would orphan any connection held by a concurrently-running process
        # (e.g. `langgraph dev` alongside `streamlit run app.py`).
        self._db.executescript(CLEAR_DDL)
        self._db.executescript(SCHEMA_DDL)
        self._db.commit()

    def _iter_events(self) -> Iterable[dict[str, Any]]:
        if not self.kb_json.exists():
            return []
        with self.kb_json.open("r") as f:
            for raw in f:
                raw = raw.strip()
                if raw:
                    yield json.loads(raw)

    def _replay_events(self) -> None:
        for evt in self._iter_events():
            self._apply_event_to_sqlite(evt)
        assert self._db is not None
        self._db.commit()

    def _warm_dump_cache(self) -> None:
        """Load the most recent ingested dump into memory if present."""
        for evt in reversed(list(self._iter_events())):
            if evt["kind"] == EVT_INGEST_DUMP:
                path = Path(evt["payload"]["path"])
                if path.exists():
                    self._dump_bytes = path.read_bytes()
                break

    # ---- writes ---------------------------------------------------------- #

    def append_event(
        self,
        kind: str,
        source: str,
        payload: dict[str, Any],
    ) -> str:
        evt = {
            "id": uuid.uuid4().hex[:12],
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "source": source,
            "payload": payload,
        }
        with self._lock:
            with self.kb_json.open("a") as f:
                f.write(json.dumps(evt) + "\n")
            self._apply_event_to_sqlite(evt)
            assert self._db is not None
            self._db.commit()
            self._semantic_on_append_locked(evt)
        return evt["id"]

    def _semantic_build_from_derived_kb(self) -> None:
        """Hydrate cosine index from SQLite after full event replay."""

        cfg = load_semantic_config()
        self._semantic_index = None
        self._semantic_embed_ok = False
        if not cfg.enabled:
            return

        self._semantic_index = SemanticVectorIndex()
        ids, metas, texts = collect_chunks_from_derived_kb(self.query, cfg)
        if not texts:
            self._semantic_embed_ok = True
            return

        try:
            vecs = embed_texts_batched(cfg, texts)
            self._semantic_index.add_rows(ids, metas, vecs)
            self._semantic_embed_ok = True
        except EmbeddingServiceError as exc:
            print(
                "[kb_semantic] Failed to bootstrap embedding index:",
                exc,
                file=sys.stderr,
            )
            self._semantic_index = None
            self._semantic_embed_ok = False

    def _semantic_on_append_locked(self, evt: dict[str, Any]) -> None:
        """Incremental vector update mirrors `append_event` writes."""

        cfg = load_semantic_config()
        if not cfg.enabled:
            return
        if (
            self._semantic_index is None or not self._semantic_embed_ok
        ) or not cfg.index_event_kinds:
            return

        pred, ids, metas, texts = incremental_chunks_from_event(evt, cfg)
        if not ids:
            return

        try:
            n_rm = self._semantic_index.remove_predicate(pred)
            if not texts:
                return
            vecs = embed_texts_batched(cfg, texts)
            self._semantic_index.add_rows(ids, metas, vecs)
            _ = n_rm  # reserved for diagnostics
        except EmbeddingServiceError as exc:
            print(
                "[kb_semantic] Incremental embedding failed;",
                evt.get("kind"),
                evt.get("id"),
                ":",
                exc,
                file=sys.stderr,
            )
            traceback.print_exc(limit=8, file=sys.stderr)
            self._semantic_build_from_derived_kb()

    def semantic_search_enabled(self) -> bool:
        cfg = load_semantic_config()
        return bool(
            cfg.enabled
            and self._semantic_embed_ok
            and self._semantic_index is not None,
        )

    def search_semantic(
        self,
        query: str,
        limit: int | None = None,
        *,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return cosine-ranked KB chunks (metadata + reconstructed text)."""

        cfg = load_semantic_config()
        if (
            not query.strip()
            or not cfg.enabled
            or self._semantic_index is None
            or not self._semantic_embed_ok
        ):
            return []

        k = (
            cfg.semantic_search_limit_default
            if limit is None
            else max(1, int(limit))
        )

        q_clean = query.strip()
        if not q_clean:
            return []

        q_tokens = set(re.findall(r"\$?[a-z0-9_]{2,}", q_clean.lower()))

        def _source_key(meta: dict[str, Any]) -> str:
            facet = str(meta.get("facet") or "?")
            if facet == "text_doc":
                return f"{facet}:{meta.get('path') or ''}"
            if meta.get("start") is not None:
                return f"{facet}:start:{int(meta.get('start') or 0)}"
            if meta.get("addr") is not None and meta.get("name"):
                return (
                    f"{facet}:addr:{int(meta.get('addr') or 0)}:"
                    f"{str(meta.get('name') or '')}"
                )
            if meta.get("event_id"):
                return f"{facet}:evt:{meta.get('event_id')}"
            if meta.get("source_event_id"):
                return f"{facet}:src:{meta.get('source_event_id')}"
            if meta.get("hypothesis_id"):
                return f"{facet}:hyp:{meta.get('hypothesis_id')}"
            return f"{facet}:id:{meta.get('id') or ''}"

        def _lexical_overlap_score(query_tokens: set[str], text: str) -> float:
            if not query_tokens:
                return 0.0
            t_tokens = set(re.findall(r"\$?[a-z0-9_]{2,}", (text or "").lower()))
            if not t_tokens:
                return 0.0
            overlap = len(query_tokens & t_tokens)
            return min(1.0, overlap / max(1, len(query_tokens)))

        try:
            q_vecs = embed_texts_batched(cfg, [q_clean])
        except EmbeddingServiceError as exc:
            print("[kb_semantic] query embedding failed:", exc, file=sys.stderr)
            return []

        qv = q_vecs[0]
        ranked = self._semantic_index.search(qv, top_k=max(k * 8, k + 8))
        candidates: list[dict[str, Any]] = []
        for row in ranked:
            if score_threshold is not None and row["score"] < float(score_threshold):
                continue
            meta = row["meta"]
            preview = semantic_hit_preview_text(
                meta,
                self.query,
                chunk_chars=cfg.chunk_chars,
                chunk_overlap=cfg.chunk_overlap,
            )
            lex = _lexical_overlap_score(q_tokens, preview)
            mixed_score = (0.8 * float(row["score"])) + (0.2 * lex)
            candidates.append({
                "id": row["id"],
                "score": round(row["score"], 6),
                "rank_score": round(mixed_score, 6),
                "lexical_overlap": round(lex, 6),
                "meta": meta,
                "text": preview,
                "snippet_line": snippet_for_semantic(meta, preview),
            })

        candidates.sort(
            key=lambda h: (
                -float(h.get("rank_score") or 0.0),
                -float(h.get("score") or 0.0),
            ),
        )

        hits: list[dict[str, Any]] = []
        per_source: dict[str, int] = {}
        for h in candidates:
            meta = h.get("meta") or {}
            sk = _source_key(meta)
            max_per = 2 if (meta.get("facet") == "text_doc") else 1
            if per_source.get(sk, 0) >= max_per:
                continue
            per_source[sk] = per_source.get(sk, 0) + 1
            hits.append(h)
            if len(hits) >= k:
                break

        return hits

    def _apply_event_to_sqlite(self, evt: dict[str, Any]) -> None:
        assert self._db is not None
        self._db.execute(
            "INSERT OR REPLACE INTO events (id, ts, kind, source, payload_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (evt["id"], evt["ts"], evt["kind"], evt["source"],
             json.dumps(evt["payload"])),
        )

        if evt["kind"] == EVT_LABEL:
            p = evt["payload"]
            self._db.execute(
                "INSERT OR REPLACE INTO labels"
                " (addr, name, kind, confidence, source_event_id)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    int(p["addr"]),
                    p["name"],
                    p.get("kind"),
                    float(p.get("confidence", 0.5)),
                    evt["id"],
                ),
            )

        if evt["kind"] == EVT_INGEST_TEXT:
            p = evt["payload"]
            self._db.execute(
                "INSERT OR REPLACE INTO text_docs"
                " (path, ts, size, mtime, content, source_event_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    p["path"],
                    evt["ts"],
                    int(p.get("size", 0)),
                    float(p.get("mtime") or 0.0),
                    p.get("content", ""),
                    evt["id"],
                ),
            )

        if evt["kind"] == EVT_ROUTINE:
            p = evt["payload"]
            self._db.execute(
                "INSERT OR REPLACE INTO routines"
                " (start, end, name, summary, calls_to_json, called_by_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    int(p["start"]),
                    int(p.get("end", p["start"])),
                    p.get("name"),
                    p.get("summary"),
                    json.dumps(p.get("calls_to") or []),
                    json.dumps(p.get("called_by") or []),
                ),
            )

        if evt["kind"] == EVT_DATA_STRUCTURE:
            p = evt["payload"]
            self._db.execute(
                "INSERT OR REPLACE INTO data_structures"
                " (start, end, kind, fields_json) VALUES (?, ?, ?, ?)",
                (
                    int(p["start"]),
                    int(p.get("end", p["start"])),
                    p.get("kind"),
                    json.dumps(p.get("fields") or {}),
                ),
            )

        if evt["kind"] == EVT_HYPOTHESIS:
            p = evt["payload"]
            self._db.execute(
                "INSERT OR REPLACE INTO hypotheses"
                " (id, text, status, evidence_json) VALUES (?, ?, ?, ?)",
                (
                    str(p.get("id") or evt["id"]),
                    str(p.get("text", "")),
                    str(p.get("status", "open")),
                    json.dumps(p.get("evidence") or []),
                ),
            )

    # ---- ingestion ------------------------------------------------------- #

    def ingest_dump(self, path: Path) -> str | None:
        """Idempotent: skips if a dump with this path is already recorded."""
        path = Path(path)
        for evt in self._iter_events():
            if evt["kind"] == EVT_INGEST_DUMP and evt["payload"]["path"] == str(path):
                self._dump_bytes = path.read_bytes() if path.exists() else None
                return None
        if not path.exists():
            return None
        self._dump_bytes = path.read_bytes()
        return self.append_event(
            EVT_INGEST_DUMP,
            "load_inputs",
            {"path": str(path), "size": len(self._dump_bytes)},
        )

    def ingest_text_file(self, path: Path) -> str | None:
        """Idempotent ingestion of a single text file.

        Re-ingests when the file's `mtime` is newer than the last record.
        """
        path = Path(path)
        if not path.exists() or not path.is_file():
            return None

        size = path.stat().st_size
        mtime = path.stat().st_mtime

        for evt in self._iter_events():
            if evt["kind"] != EVT_INGEST_TEXT:
                continue
            p = evt.get("payload") or {}
            if p.get("path") == str(path) and float(p.get("mtime") or 0.0) >= mtime:
                return None  # already current

        try:
            raw_bytes = path.read_bytes()
        except OSError:
            return None

        truncated = False
        if len(raw_bytes) > TEXT_FILE_MAX_BYTES:
            raw_bytes = raw_bytes[:TEXT_FILE_MAX_BYTES]
            truncated = True
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = raw_bytes.decode("utf-8", errors="replace")
        if truncated:
            content += (
                f"\n\n... [truncated to {TEXT_FILE_MAX_BYTES} bytes by KB]"
            )

        return self.append_event(
            EVT_INGEST_TEXT,
            "load_inputs",
            {
                "path": str(path),
                "size": size,
                "mtime": mtime,
                "truncated": truncated,
                "content": content,
            },
        )

    def ingest_text_dir(self, root: Path) -> dict[str, Any]:
        """Walk `root` recursively and ingest every recognised text file."""
        root = Path(root)
        result = {
            "root": str(root),
            "files_scanned": 0,
            "files_ingested": 0,
            "files_skipped": 0,
            "ingested_paths": [],
            "skipped_paths": [],
            "exists": root.exists() and root.is_dir(),
        }
        if not result["exists"]:
            return result

        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in TEXT_FILE_SUFFIXES:
                continue
            result["files_scanned"] += 1
            evt_id = self.ingest_text_file(p)
            if evt_id:
                result["files_ingested"] += 1
                result["ingested_paths"].append(str(p))
            else:
                result["files_skipped"] += 1
                result["skipped_paths"].append(str(p))
        return result

    # Words that carry no signal for text search.
    _TEXT_STOP_WORDS = frozenset(
        "a an the and or of in is at to for with from on by "
        "are was were be been have has had do does did will "
        "this that it its they them their".split()
    )

    def search_text(
        self, query: str, limit: int = 10, snippet_chars: int = 240,
    ) -> list[dict[str, Any]]:
        """Substring search across ingested text docs (case-insensitive).

        Phase 1: exact phrase match (LIKE '%query%').
        Phase 2: if no hits, split into significant words and run per-word
                 OR matches, then rank rows by how many words they contain.
        Each result includes a snippet around the best matching word.
        """
        if not query:
            return []

        # --- Phase 1: exact phrase ---
        rows = self.query(
            "SELECT path, content, size FROM text_docs"
            " WHERE lower(content) LIKE lower(?) LIMIT ?",
            (f"%{query}%", int(limit)),
        )
        matched_term = query if rows else None

        # --- Phase 2: per-word OR if no exact hit ---
        if not rows:
            words = [
                w for w in re.split(r"\W+", query.lower())
                if len(w) >= 3 and w not in self._TEXT_STOP_WORDS
            ]
            if words:
                placeholders = " OR ".join(
                    ["lower(content) LIKE ?"] * len(words)
                )
                params: list[Any] = [f"%{w}%" for w in words] + [int(limit)]
                rows = self.query(
                    f"SELECT path, content, size FROM text_docs"
                    f" WHERE {placeholders} LIMIT ?",
                    params,
                )
                # Re-rank: most words present first
                if rows:
                    def _word_hits(content: str) -> int:
                        cl = content.lower()
                        return sum(1 for w in words if w in cl)
                    rows = sorted(rows, key=lambda r: _word_hits(r.get("content") or ""), reverse=True)
                    matched_term = words[0]  # best single anchor for snippet

        out: list[dict[str, Any]] = []
        radius = max(40, snippet_chars // 3)
        for r in rows:
            content = r["content"] or ""
            anchor = (matched_term or query).lower()
            idx = content.lower().find(anchor)
            if idx == -1:
                snippet = content[:snippet_chars]
            else:
                start = max(0, idx - radius)
                end = min(len(content), idx + radius + len(anchor))
                snippet = (
                    ("..." if start > 0 else "")
                    + content[start:end]
                    + ("..." if end < len(content) else "")
                )
            out.append({
                "path": r["path"],
                "size": r.get("size"),
                "match_at": idx,
                "snippet": snippet,
            })
        return out

    def text_doc_count(self) -> int:
        rows = self.query("SELECT COUNT(*) AS n FROM text_docs")
        return int(rows[0]["n"]) if rows else 0

    def ingest_partial_asm(self, path: Path) -> str | None:
        path = Path(path)
        for evt in self._iter_events():
            if (
                evt["kind"] == EVT_INGEST_PARTIAL_ASM
                and evt["payload"]["path"] == str(path)
            ):
                return None
        if not path.exists():
            return None
        text = path.read_text(errors="replace")
        labels = _parse_partial_asm(text)
        evt_id = self.append_event(
            EVT_INGEST_PARTIAL_ASM,
            "load_inputs",
            {"path": str(path), "size": len(text), "labels_found": len(labels)},
        )
        for addr, name in labels:
            self.append_event(
                EVT_LABEL,
                "load_inputs",
                {"addr": addr, "name": name, "kind": "partial_asm",
                 "confidence": 0.8},
            )
        return evt_id

    # ---- reads ----------------------------------------------------------- #

    def read_bytes(self, addr: int, length: int) -> bytes:
        if self._dump_bytes is None:
            return b""
        end = min(addr + length, len(self._dump_bytes))
        return self._dump_bytes[addr:end]

    def has_dump(self) -> bool:
        return self._dump_bytes is not None and len(self._dump_bytes) > 0

    def full_dump(self) -> bytes:
        """Return the cached dump, padded to 64KB for full-memory analysis."""
        if not self._dump_bytes:
            return b""
        if len(self._dump_bytes) >= 0x10000:
            return self._dump_bytes[:0x10000]
        buf = bytearray(0x10000)
        buf[: len(self._dump_bytes)] = self._dump_bytes
        return bytes(buf)

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        assert self._db is not None
        rows = self._db.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        assert self._db is not None
        events = self._db.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        by_kind = {
            r["kind"]: r["n"]
            for r in self._db.execute(
                "SELECT kind, COUNT(*) AS n FROM events GROUP BY kind"
            ).fetchall()
        }
        labels = self._db.execute("SELECT COUNT(*) AS n FROM labels").fetchone()["n"]
        text_docs = self._db.execute(
            "SELECT COUNT(*) AS n FROM text_docs"
        ).fetchone()["n"]
        return {
            "events_total": events,
            "events_by_kind": by_kind,
            "labels": labels,
            "text_docs": text_docs,
            "dump_bytes": len(self._dump_bytes) if self._dump_bytes else 0,
            "semantic_kb_enabled": load_semantic_config().enabled,
            "semantic_kb_ready": bool(
                load_semantic_config().enabled
                and self._semantic_embed_ok
                and self._semantic_index is not None,
            ),
            "semantic_kb_chunks": (
                len(self._semantic_index) if self._semantic_index else 0
            ),
        }

    def kb_size_tokens(self) -> int:
        """Rough token estimate for the curator gate (~4 chars/token)."""
        size = self.kb_json.stat().st_size if self.kb_json.exists() else 0
        return size // 4

    @contextmanager
    def cursor(self):
        assert self._db is not None
        cur = self._db.cursor()
        try:
            yield cur
        finally:
            cur.close()

    # ---- digest helpers for the Analyst/Critic --------------------------- #
    #
    # These are the methods the analyst pipeline relies on to get a
    # *question-focused* view of the KB. The previous design dumped a few
    # truncated tool results at the model and called it a day; the result
    # was answers that systematically lagged what a single-prompt model
    # could produce. The helpers below let us assemble a structured
    # evidence sheet keyed off the user's question.
    # --------------------------------------------------------------------- #

    _STOPWORDS = frozenset({
        "the", "a", "an", "of", "to", "and", "or", "is", "are", "was",
        "were", "be", "been", "being", "this", "that", "those", "these",
        "where", "what", "how", "why", "when", "in", "on", "at", "by",
        "for", "with", "from", "as", "it", "its", "than", "then", "into",
        "do", "does", "did", "done", "i", "we", "you", "they", "he", "she",
        "his", "her", "their", "any", "all", "some", "most", "more", "less",
        "value", "values", "data", "code", "memory", "address", "addresses",
        "byte", "bytes",
    })

    @classmethod
    def _question_terms(cls, question: str, max_terms: int = 8) -> list[str]:
        """Pull out alphabetic search terms from the question."""
        if not question:
            return []
        raw = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question)
        seen: list[str] = []
        for w in raw:
            wl = w.lower()
            if wl in cls._STOPWORDS:
                continue
            if wl not in seen:
                seen.append(wl)
            if len(seen) >= max_terms:
                break
        return seen

    def relevant_labels(
        self, question: str, limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Labels whose name or partial-asm context relates to the question."""
        terms = self._question_terms(question)
        if not terms:
            return self.query(
                "SELECT addr, name, kind, confidence FROM labels"
                " ORDER BY confidence DESC, addr LIMIT ?",
                (limit,),
            )

        clauses = " OR ".join(["lower(name) LIKE ?"] * len(terms))
        params = [f"%{t}%" for t in terms] + [limit]
        rows = self.query(
            f"SELECT addr, name, kind, confidence FROM labels"
            f" WHERE {clauses}"
            f" ORDER BY confidence DESC, addr LIMIT ?",
            tuple(params),
        )
        if rows:
            return rows
        # Fallback to top-confidence labels so the analyst still sees something.
        return self.query(
            "SELECT addr, name, kind, confidence FROM labels"
            " ORDER BY confidence DESC, addr LIMIT ?",
            (min(limit, 20),),
        )

    def routines_summary(self, limit: int = 30) -> list[dict[str, Any]]:
        return self.query(
            "SELECT start, end, name, summary, calls_to_json, called_by_json"
            " FROM routines ORDER BY start LIMIT ?",
            (limit,),
        )

    def open_hypotheses(self, limit: int = 30) -> list[dict[str, Any]]:
        return self.query(
            "SELECT id, text, status, evidence_json FROM hypotheses"
            " WHERE status != 'refuted' ORDER BY id LIMIT ?",
            (limit,),
        )

    def data_structures_list(self, limit: int = 30) -> list[dict[str, Any]]:
        return self.query(
            "SELECT start, end, kind, fields_json FROM data_structures"
            " ORDER BY start LIMIT ?",
            (limit,),
        )

    def recent_tool_excerpts(
        self, limit: int = 12, max_chars_per: int = 4_000,
    ) -> list[dict[str, Any]]:
        """Most recent tool_result events, each lightly truncated."""
        rows = self.query(
            "SELECT id, ts, source, payload_json FROM events"
            " WHERE kind = ? ORDER BY ts DESC LIMIT ?",
            (EVT_TOOL_RESULT, limit),
        )
        out = []
        for r in rows:
            try:
                payload = json.loads(r["payload_json"])
            except Exception:  # noqa: BLE001
                payload = {}
            data_str = str(payload.get("data", ""))
            if len(data_str) > max_chars_per:
                data_str = data_str[:max_chars_per] + "\n... [truncated]"
            out.append({
                "event_id": r["id"],
                "ts":       r["ts"],
                "tool":     payload.get("tool"),
                "step_id":  payload.get("step_id"),
                "ok":       payload.get("ok"),
                "extra":    {
                    k: v for k, v in payload.items()
                    if k not in {"data", "tool", "step_id", "ok"}
                },
                "data":     data_str,
            })
        return out

    def text_search_for(
        self, question: str, hits_per_term: int = 2, snippet_chars: int = 500,
    ) -> list[dict[str, Any]]:
        """Aggregate `search_text` calls across all question terms."""
        terms = self._question_terms(question)
        seen: set[tuple[str, int]] = set()
        out: list[dict[str, Any]] = []
        for t in terms:
            for h in self.search_text(t, limit=hits_per_term,
                                      snippet_chars=snippet_chars):
                key = (h["path"], h["match_at"])
                if key in seen:
                    continue
                seen.add(key)
                out.append({**h, "matched_term": t})
        return out

    def partial_asm_excerpt(self, max_lines: int = 120) -> str:
        """Return the raw partial-asm content (head) if one was ingested."""
        for evt in self._iter_events():
            if evt["kind"] != EVT_INGEST_PARTIAL_ASM:
                continue
            path = Path(evt["payload"].get("path", ""))
            if not path.exists():
                continue
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                return ""
            head = "\n".join(lines[:max_lines])
            if len(lines) > max_lines:
                head += f"\n... [{len(lines) - max_lines} more lines]"
            return head
        return ""

    def digest_for_question(
        self, question: str, *, max_chars: int = 32_000,
    ) -> str:
        """Build a question-focused KB evidence sheet (markdown).

        This is the *single source of truth* for the Analyst and Critic
        contexts. Keeping it here (instead of inline in `analyst_node`)
        means both agents see exactly the same evidence — which is
        important for the critic to actually catch unsupported claims.
        """
        s = self.stats()
        terms = self._question_terms(question)

        sections: list[str] = []

        sections.append(
            f"## KB stats\n"
            f"- events_total: {s['events_total']}\n"
            f"- labels: {s['labels']}\n"
            f"- text_docs: {s.get('text_docs', 0)}\n"
            f"- dump_bytes: {s['dump_bytes']}\n"
            f"- semantic_kb_enabled: {s.get('semantic_kb_enabled')}\n"
            f"- semantic_kb_ready (embeddings hydrated): "
            f"{s.get('semantic_kb_ready')} "
            f"(chunks≈{s.get('semantic_kb_chunks')})\n"
            f"- events_by_kind: {s.get('events_by_kind', {})}\n"
            f"- question_terms: {terms or '(none extracted)'}"
        )

        labels = self.relevant_labels(question, limit=40)
        if labels:
            label_lines = [
                f"- ${r['addr']:04X}  {r['name']}  "
                f"({r.get('kind') or '?'}, conf={r['confidence']:.2f})"
                for r in labels
            ]
            sections.append(
                "## Relevant labels\n" + "\n".join(label_lines)
            )

        routines = self.routines_summary(limit=30)
        if routines:
            r_lines = []
            for r in routines:
                try:
                    calls = json.loads(r.get("calls_to_json") or "[]")
                except Exception:  # noqa: BLE001
                    calls = []
                calls_hex = [
                    f"${int(c):04X}" if isinstance(c, int) else str(c)
                    for c in calls[:6]
                ]
                calls_str = f" calls→ {', '.join(calls_hex)}" if calls_hex else ""
                r_lines.append(
                    f"- ${r['start']:04X}-${r['end']:04X}  "
                    f"{r.get('name') or '(unnamed)'} — "
                    f"{r.get('summary') or '(no summary)'}{calls_str}"
                )
            sections.append("## Identified routines\n" + "\n".join(r_lines))

        structs = self.data_structures_list(limit=20)
        if structs:
            s_lines = [
                f"- ${r['start']:04X}-${r['end']:04X}  "
                f"{r.get('kind') or '?'}  fields={r.get('fields_json') or '{}'}"
                for r in structs
            ]
            sections.append("## Data structures\n" + "\n".join(s_lines))

        hyps = self.open_hypotheses(limit=20)
        if hyps:
            h_lines = []
            for h in hyps:
                try:
                    ev = json.loads(h.get("evidence_json") or "[]")
                except Exception:  # noqa: BLE001
                    ev = []
                h_lines.append(
                    f"- [{h['id']}/{h['status']}] {h['text']}"
                    + (f"  (evidence: {', '.join(map(str, ev[:4]))})" if ev else "")
                )
            sections.append("## Hypotheses\n" + "\n".join(h_lines))

        notes = self.text_search_for(question, hits_per_term=2,
                                     snippet_chars=600)
        if notes:
            note_lines: list[str] = []
            for h in notes:
                note_lines.append(
                    f"- {h['path']}  (matched {h['matched_term']!r}, "
                    f"offset={h['match_at']})\n"
                    f"  …{h['snippet']}…"
                )
            sections.append(
                "## User-provided notes (authoritative)\n"
                + "\n".join(note_lines)
            )

        kb_sem_cfg = load_semantic_config()
        if kb_sem_cfg.enabled and question.strip():
            sem_hits = self.search_semantic(
                question, limit=kb_sem_cfg.digest_semantic_limit,
            )
            if sem_hits:
                semi_lines = []
                for h in sem_hits:
                    body = (
                        (h.get("text") or "")[:880]
                        or (h.get("snippet_line") or "")
                    )
                    semi_lines.append(
                        f"- score={h.get('score')}  id={h.get('id')}\n"
                        f"  {h.get('snippet_line')}\n"
                        f"  …{body}…",
                    )
                sections.append(
                    "## Semantic retrieval (embedding similarity)\n"
                    + "\n".join(semi_lines)
                )

        partial = self.partial_asm_excerpt(max_lines=80)
        if partial:
            sections.append("## Partial-asm head\n```\n" + partial + "\n```")

        # Build the base digest (without raw tool dumps) first so we know
        # how much budget remains for the verbose tool excerpts.
        base_digest = "\n\n".join(sections)
        budget_for_recent = max_chars - len(base_digest) - 100

        recent = self.recent_tool_excerpts(limit=12, max_chars_per=4_000)
        if recent and budget_for_recent > 200:
            ex_lines: list[str] = []
            used = 0
            for r in recent:
                entry = (
                    f"### Tool result `{r['event_id']}` "
                    f"({r['tool']}/{r['step_id']}, ok={r['ok']})\n"
                    f"{r['data']}"
                )
                if used + len(entry) > budget_for_recent - 100:
                    ex_lines.append(
                        f"... [{len(recent) - len(ex_lines)} older result(s) "
                        "omitted — budget exhausted]"
                    )
                    break
                ex_lines.append(entry)
                used += len(entry)
            if ex_lines:
                sections.append("## Recent tool results\n" + "\n\n".join(ex_lines))

        digest = "\n\n".join(sections).strip()

        # Safety cap — should rarely trigger now that recent results are
        # budget-gated, but keeps behaviour deterministic.
        if len(digest) > max_chars:
            digest = digest[: max_chars - 200] + "\n\n... [digest truncated to fit context budget]"
        return digest
