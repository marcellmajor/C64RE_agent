"""Two-tier knowledge store.

`kb.json` is the source-of-truth append-only event log. `kb.sqlite` is a
derived, queryable view. The view persists between processes: on load,
only the event-log tail beyond a recorded high-water byte offset is
replayed (tracker 1.6); a schema-version or offset mismatch triggers a
full rebuild from `kb.json`, so the view still can never drift.

Per `CLAUDE.md` §2.5.
"""

from __future__ import annotations

import hashlib
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
    EVT_CONSOLIDATED,
    EVT_DATA_STRUCTURE,
    EVT_HYPOTHESIS,
    EVT_INGEST_DUMP,
    EVT_INGEST_PARTIAL_ASM,
    EVT_INGEST_TEXT,
    EVT_LABEL,
    EVT_RUN_SUMMARY,
    EVT_ROUTINE,
    EVT_TOOL_RESULT,
    SCHEMA_DDL,
    SCHEMA_VERSION,
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

_HEX_ADDRESS_RE = re.compile(
    r"(?<![0-9A-Fa-f])(?:\$|0[xX])([0-9A-Fa-f]{2,4})(?![0-9A-Fa-f])",
)


def extract_hex_addresses(text: str, *, limit: int = 16) -> list[int]:
    """Extract unique ``$XXXX``/``0xXXXX`` addresses in encounter order."""
    out: list[int] = []
    for match in _HEX_ADDRESS_RE.finditer(str(text or "")):
        addr = int(match.group(1), 16)
        if addr not in out:
            out.append(addr)
        if len(out) >= limit:
            break
    return out

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


def _tool_result_key(payload: dict[str, Any]) -> str:
    """Content-hash identity of a tool result (tracker 1.6).

    Full-data hash — not a text prefix — so an expanded window whose new
    evidence appears after the opening lines is a distinct result.
    """
    data = payload.get("data", "")
    if not isinstance(data, str):
        data = json.dumps(data, sort_keys=True, default=str)
    raw = f"{payload.get('tool')}|{payload.get('step_id')}|{data}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
        # Memoization (tracker 1.5): digest keyed on
        # (question, events_total, max_chars); partial-asm head keyed on
        # max_lines; question-embedding vectors keyed on query text.
        self._digest_cache: tuple[tuple, str] | None = None
        self._partial_asm_cache: tuple[int, str] | None = None
        self._q_embed_cache: dict[str, Any] = {}

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
        self._db = sqlite3.connect(str(self.kb_sqlite), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        # The derived view PERSISTS between processes now (tracker 1.6):
        # only ensure missing tables exist here; `_replay_events` decides
        # between an incremental tail replay and a full rebuild. Tables
        # are never unlinked, which would orphan connections held by a
        # concurrently-running process (`langgraph dev` + Streamlit).
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

    # Meta keys for the persistent derived view.
    _META_SCHEMA_VERSION = "schema_version"
    _META_REPLAY_OFFSET = "replay_offset"  # byte offset into kb.json
    # Identity of the already-replayed prefix (review finding 6): a
    # replaced log with the same byte size — or an edited prefix — must
    # trigger a rebuild, not be silently treated as current.
    _META_REPLAY_FINGERPRINT = "replay_fingerprint"

    # Bytes probed at each end of the replayed prefix for the fingerprint.
    _FINGERPRINT_PROBE = 4096

    def _log_fingerprint(self, offset: int) -> str:
        """Cheap identity of kb.json's first `offset` bytes.

        Hashes the first and last `_FINGERPRINT_PROBE` bytes of the
        replayed prefix (plus the offset itself), so same-size
        replacement, head edits, and edits near the replay boundary are
        all detected in O(8KB) instead of re-reading the whole log.
        A modification strictly inside the un-probed middle of a >8KB
        prefix is the accepted blind spot (documented limitation).
        """
        if offset <= 0 or not self.kb_json.exists():
            return f"empty:{offset}"
        h = hashlib.sha256()
        h.update(str(offset).encode())
        with self.kb_json.open("rb") as f:
            head_len = min(self._FINGERPRINT_PROBE, offset)
            h.update(f.read(head_len))
            tail_start = max(head_len, offset - self._FINGERPRINT_PROBE)
            if tail_start < offset:
                f.seek(tail_start)
                h.update(f.read(offset - tail_start))
        return h.hexdigest()

    def _meta_get(self, key: str) -> str | None:
        assert self._db is not None
        try:
            row = self._db.execute(
                "SELECT value FROM meta WHERE key = ?", (key,),
            ).fetchone()
        except sqlite3.Error:
            return None
        return row["value"] if row else None

    def _meta_set(self, key: str, value: str) -> None:
        assert self._db is not None
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )

    def _full_rebuild(self) -> None:
        """Drop + replay the complete event log (the pre-1.6 behaviour)."""
        assert self._db is not None
        size = self.kb_json.stat().st_size if self.kb_json.exists() else 0
        self._db.executescript(CLEAR_DDL)
        self._db.executescript(SCHEMA_DDL)
        for evt in self._iter_events():
            self._apply_event_to_sqlite(evt)
        self._meta_set(self._META_SCHEMA_VERSION, SCHEMA_VERSION)
        self._meta_set(self._META_REPLAY_OFFSET, str(size))
        self._meta_set(self._META_REPLAY_FINGERPRINT, self._log_fingerprint(size))
        self._db.commit()

    def _replay_events(self) -> None:
        """Bring the derived view up to date with kb.json.

        Incremental: replay only the JSONL tail past the recorded
        high-water byte offset (tracker 1.6 — startup used to be
        O(all events) on every process start). Full rebuild when the
        schema version changed, the offset is missing/ahead of the file
        (truncated log), the already-replayed prefix's fingerprint no
        longer matches (replaced/edited log, including same-size
        replacement — review finding 6), or the tail fails to parse.
        """
        assert self._db is not None
        size = self.kb_json.stat().st_size if self.kb_json.exists() else 0

        version = self._meta_get(self._META_SCHEMA_VERSION)
        offset_raw = self._meta_get(self._META_REPLAY_OFFSET)
        try:
            offset = int(offset_raw) if offset_raw is not None else None
        except ValueError:
            offset = None

        if version != SCHEMA_VERSION or offset is None or offset > size:
            self._full_rebuild()
            return

        # Identity check on the replayed prefix: `offset == size` alone
        # is not proof of currency — the log may have been replaced with
        # different content of the same length.
        stored_fp = self._meta_get(self._META_REPLAY_FINGERPRINT)
        if stored_fp is None or stored_fp != self._log_fingerprint(offset):
            self._full_rebuild()
            return

        if offset == size:
            return  # already current

        try:
            # Binary mode: the recorded offset is a byte position at a
            # line boundary (text-mode seek only accepts tell() values).
            with self.kb_json.open("rb") as f:
                f.seek(offset)
                for raw in f:
                    line = raw.decode("utf-8").strip()
                    if line:
                        self._apply_event_to_sqlite(json.loads(line))
        except Exception:  # noqa: BLE001 — corrupt tail: rebuild from scratch
            self._full_rebuild()
            return
        self._meta_set(self._META_REPLAY_OFFSET, str(size))
        self._meta_set(self._META_REPLAY_FINGERPRINT, self._log_fingerprint(size))
        self._db.commit()

    def _warm_dump_cache(self) -> None:
        """Load the most recent ingested dump into memory if present."""
        rows = self.query(
            "SELECT payload_json FROM events WHERE kind = ?"
            " ORDER BY ts DESC LIMIT 1",
            (EVT_INGEST_DUMP,),
        )
        for r in rows:
            try:
                payload = json.loads(r["payload_json"])
            except Exception:  # noqa: BLE001
                continue
            path = Path(payload.get("path", ""))
            if path.exists():
                self._dump_bytes = path.read_bytes()

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
            # Advance the replay high-water mark: we just appended the
            # final line, so the derived view is current through EOF.
            new_size = self.kb_json.stat().st_size
            self._meta_set(self._META_REPLAY_OFFSET, str(new_size))
            self._meta_set(
                self._META_REPLAY_FINGERPRINT,
                self._log_fingerprint(new_size),
            )
            assert self._db is not None
            self._db.commit()
            self._semantic_on_append_locked(evt)
        return evt["id"]

    def record_run_summary(self, payload: dict[str, Any]) -> str | None:
        """Append one idempotent ``run_summary`` event.

        Report generation can be retried after a checkpoint/resume. The
        stable ``run_id`` makes that retry a no-op instead of inflating
        cross-run metrics with duplicate completions.
        """
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            raise ValueError("run_summary requires a non-empty run_id")
        rows = self.query(
            "SELECT event_id FROM run_summaries WHERE run_id = ? LIMIT 1",
            (run_id,),
        )
        if rows:
            return None
        return self.append_event(EVT_RUN_SUMMARY, "write_report", payload)

    def transition_hypotheses(
        self, references: Iterable[Any], *, status: str, source: str,
        evidence: Iterable[Any] | None = None,
    ) -> list[str]:
        """Append lifecycle updates for explicitly referenced hypotheses."""
        target = str(status).strip().lower()
        if target not in {"open", "supported", "refuted"}:
            raise ValueError(f"invalid hypothesis status: {status!r}")
        changed: list[str] = []
        seen: set[str] = set()
        for reference in references:
            hid = str(reference or "").strip().split("/", 1)[0]
            if not hid or hid in seen:
                continue
            seen.add(hid)
            rows = self.query(
                "SELECT id, text, status, evidence_json FROM hypotheses"
                " WHERE id = ? LIMIT 1",
                (hid,),
            )
            if not rows:
                # Analyst evidence may cite the append-only event id rather
                # than the semantic hypothesis id. Resolve that provenance
                # token on events.id only. A short prefix must be unique;
                # never let it degrade into substring matching against the
                # semantic hypothesis ids.
                events = self.query(
                    "SELECT payload_json FROM events"
                    " WHERE kind = ? AND id = ? LIMIT 1",
                    (EVT_HYPOTHESIS, hid),
                )
                if not events:
                    prefix_events = self.query(
                        "SELECT payload_json FROM events"
                        " WHERE kind = ? AND id LIKE ? ORDER BY id LIMIT 2",
                        (EVT_HYPOTHESIS, f"{hid}%"),
                    )
                    events = prefix_events if len(prefix_events) == 1 else []
                if events:
                    try:
                        event_payload = json.loads(
                            events[0].get("payload_json") or "{}",
                        )
                    except json.JSONDecodeError:
                        event_payload = {}
                    event_hid = str(event_payload.get("id") or "").strip()
                    if event_hid:
                        rows = self.query(
                            "SELECT id, text, status, evidence_json"
                            " FROM hypotheses WHERE id = ? LIMIT 1",
                            (event_hid,),
                        )
            if not rows or str(rows[0].get("status")) == target:
                continue
            row = rows[0]
            try:
                prior_evidence = json.loads(row.get("evidence_json") or "[]")
            except json.JSONDecodeError:
                prior_evidence = []
            combined = [*prior_evidence, *(str(item) for item in evidence or [])]
            payload = {
                "id": str(row["id"]),
                "text": str(row.get("text") or ""),
                "status": target,
                "evidence": list(dict.fromkeys(combined)),
                "transition_source": source,
            }
            self.append_event(EVT_HYPOTHESIS, source, payload)
            changed.append(str(row["id"]))
        return changed

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

        # Memoize the question vector — every digest build used to
        # re-embed the same question, one HTTP call per node per step
        # (tracker 1.5).
        qv = self._q_embed_cache.get(q_clean)
        if qv is None:
            try:
                q_vecs = embed_texts_batched(cfg, [q_clean])
            except EmbeddingServiceError as exc:
                print("[kb_semantic] query embedding failed:", exc, file=sys.stderr)
                return []
            qv = q_vecs[0]
            if len(self._q_embed_cache) > 32:
                self._q_embed_cache.clear()
            self._q_embed_cache[q_clean] = qv
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

        if evt["kind"] == EVT_TOOL_RESULT:
            # Indexed dedup identity for the synthesizer (tracker 1.6).
            self._db.execute(
                "INSERT OR REPLACE INTO tool_result_keys (key, event_id)"
                " VALUES (?, ?)",
                (_tool_result_key(evt["payload"]), evt["id"]),
            )

        if evt["kind"] == EVT_CONSOLIDATED:
            # Register the EXACT tool_result ids this summary covered
            # (tracker 2.1 hardening): membership in this table — never a
            # timestamp comparison — is what marks an event compacted.
            # Legacy consolidated events register only the ids they list;
            # ids that match nothing (e.g. LLM-invented ones in very old
            # events) are harmless.
            for eid in evt["payload"].get("events_compacted") or []:
                self._db.execute(
                    "INSERT OR REPLACE INTO compacted_tool_results"
                    " (event_id, consolidated_id) VALUES (?, ?)",
                    (str(eid), evt["id"]),
                )

        if evt["kind"] == EVT_RUN_SUMMARY:
            p = evt["payload"]
            self._db.execute(
                "INSERT OR IGNORE INTO run_summaries"
                " (run_id, event_id, completed_at, question, verdict,"
                "  confidence, iterations, cost_usd, tokens, llm_calls,"
                "  tool_calls, elapsed_s, termination_reason, answer_excerpt,"
                "  report_path)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(p["run_id"]), evt["id"],
                    str(p.get("completed_at") or evt["ts"]),
                    str(p.get("question") or ""),
                    str(p.get("verdict") or "n/a"),
                    float(p.get("confidence") or 0.0),
                    int(p.get("iterations") or 0),
                    float(p.get("cost_usd") or 0.0),
                    int(p.get("tokens") or 0),
                    int(p.get("llm_calls") or 0),
                    int(p.get("tool_calls") or 0),
                    (
                        float(p["elapsed_s"])
                        if p.get("elapsed_s") is not None else None
                    ),
                    p.get("termination_reason"),
                    str(p.get("answer_excerpt") or ""),
                    p.get("report_path"),
                ),
            )

        if evt["kind"] == EVT_LABEL:
            # Confidence-aware compare-and-swap (tracker 2.3): a later,
            # LOWER-confidence re-emit of the same (addr, name) must not
            # clobber a stronger fact. The old INSERT OR REPLACE let a
            # conservative re-observation downgrade good labels. Pure
            # derived-view change — the event log keeps every emission.
            p = evt["payload"]
            self._db.execute(
                "INSERT INTO labels"
                " (addr, name, kind, confidence, source_event_id)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(addr, name) DO UPDATE SET"
                "   kind = excluded.kind,"
                "   confidence = excluded.confidence,"
                "   source_event_id = excluded.source_event_id"
                " WHERE excluded.confidence >= labels.confidence",
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
            # Same confidence CAS as labels (tracker 2.3) — the routines
            # table previously had NO confidence column at all, so the
            # synthesizer prompt's "the KB will overwrite the old entry
            # automatically when confidence improves" promise was false.
            p = evt["payload"]
            self._db.execute(
                "INSERT INTO routines"
                " (start, end, name, summary, calls_to_json,"
                "  called_by_json, confidence)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(start) DO UPDATE SET"
                "   end = excluded.end,"
                "   name = excluded.name,"
                "   summary = excluded.summary,"
                "   calls_to_json = excluded.calls_to_json,"
                "   called_by_json = excluded.called_by_json,"
                "   confidence = excluded.confidence"
                " WHERE excluded.confidence >= routines.confidence",
                (
                    int(p["start"]),
                    int(p.get("end", p["start"])),
                    p.get("name"),
                    p.get("summary"),
                    json.dumps(p.get("calls_to") or []),
                    json.dumps(p.get("called_by") or []),
                    float(p.get("confidence", 0.5)),
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

    def has_tool_result(self, payload: dict[str, Any]) -> bool:
        """Indexed identity check for tool-result dedup (tracker 1.6).

        Identity = sha256(tool | step_id | full data). Replaces the
        synthesizer's full event-log scan (which also compared only a
        256-char prefix, losing evidence added past it).
        """
        assert self._db is not None
        row = self._db.execute(
            "SELECT 1 FROM tool_result_keys WHERE key = ?",
            (_tool_result_key(payload),),
        ).fetchone()
        return row is not None

    def ingest_dump(self, path: Path) -> str | None:
        """Idempotent on CONTENT, not just path (tracker 2.2).

        A dump overwritten in place under the same filename used to be
        silently ignored — stale static analysis then poisoned later
        turns. The ingest event now records the content sha256; when the
        file changed, a fresh ingest event is appended with
        `refreshed_from` so the digest can flag that earlier
        dump-derived analysis may be stale.
        """
        path = Path(path)
        if not path.exists():
            return None
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()

        prev_sha: str | None = None
        seen_path = False
        for r in self.query(
            "SELECT payload_json FROM events WHERE kind = ?"
            " ORDER BY ts DESC",
            (EVT_INGEST_DUMP,),
        ):
            try:
                payload = json.loads(r["payload_json"])
            except Exception:  # noqa: BLE001
                continue
            if payload.get("path") == str(path):
                seen_path = True
                prev_sha = payload.get("sha256")
                break

        self._dump_bytes = data
        if seen_path and prev_sha == sha:
            return None  # already current

        evt_payload: dict[str, Any] = {
            "path": str(path), "size": len(data), "sha256": sha,
        }
        if seen_path:
            if prev_sha is not None:
                evt_payload["refreshed_from"] = prev_sha  # genuine change
            else:
                evt_payload["sha_backfill"] = True  # legacy event lacked a hash
        return self.append_event(EVT_INGEST_DUMP, "load_inputs", evt_payload)

    def ingest_text_file(self, path: Path) -> str | None:
        """Idempotent ingestion of a single text file.

        Re-ingests when the file's `mtime` is newer than the last record.
        """
        path = Path(path)
        if not path.exists() or not path.is_file():
            return None

        size = path.stat().st_size
        mtime = path.stat().st_mtime

        # The text_docs table already tracks path+mtime — query it instead
        # of scanning the whole event log per file (tracker 1.6; the old
        # loop made ingest_text_dir O(files × events)).
        rows = self.query(
            "SELECT mtime FROM text_docs WHERE path = ?", (str(path),),
        )
        if rows and float(rows[0].get("mtime") or 0.0) >= mtime:
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
        for r in self.query(
            "SELECT payload_json FROM events WHERE kind = ?",
            (EVT_INGEST_PARTIAL_ASM,),
        ):
            try:
                payload = json.loads(r["payload_json"])
            except Exception:  # noqa: BLE001
                continue
            if payload.get("path") == str(path):
                return None
        if not path.exists():
            return None
        self._partial_asm_cache = None  # new source invalidates the head cache
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
        run_summaries = self._db.execute(
            "SELECT COUNT(*) AS n FROM run_summaries"
        ).fetchone()["n"]
        return {
            "events_total": events,
            "events_by_kind": by_kind,
            "labels": labels,
            "text_docs": text_docs,
            "run_summaries": run_summaries,
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
        """Labels whose name or address relates to the question.

        Address literals are first-class retrieval keys: a question such
        as ``"what writes $C145?"`` must surface a label at $C145 even when
        its name shares no words with the question (tracker 4.3).
        """
        terms = self._question_terms(question)
        addresses = extract_hex_addresses(question)
        if not terms and not addresses:
            return self.query(
                "SELECT addr, name, kind, confidence FROM labels"
                " ORDER BY confidence DESC, addr LIMIT ?",
                (limit,),
            )

        where: list[str] = []
        params: list[Any] = []
        if addresses:
            where.append("addr IN (" + ",".join("?" for _ in addresses) + ")")
            params.extend(addresses)
        if terms:
            where.extend(["lower(name) LIKE ?"] * len(terms))
            params.extend(f"%{t}%" for t in terms)
        order_sql = "confidence DESC, addr"
        if addresses:
            # Exact address evidence must not be pushed past `limit` by a
            # large number of higher-confidence name matches.
            order_sql = (
                "CASE WHEN addr IN ("
                + ",".join("?" for _ in addresses)
                + ") THEN 0 ELSE 1 END, confidence DESC, addr"
            )
            params.extend(addresses)
        params.append(limit)
        rows = self.query(
            f"SELECT addr, name, kind, confidence FROM labels"
            f" WHERE {' OR '.join(where)}"
            f" ORDER BY {order_sql} LIMIT ?",
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
            "SELECT start, end, name, summary, calls_to_json,"
            " called_by_json, confidence"
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

    # ---- curator bookkeeping helpers (tracker 2.1, hardened) ------------- #
    #
    # Compaction membership is EXACT: a tool_result counts as compacted
    # iff its event id is registered in `compacted_tool_results` (derived
    # from consolidated events' `events_compacted` lists on replay).
    # Timestamps are never the cursor — equal-timestamp events can't be
    # skipped, and a legacy consolidated event can only hide the ids it
    # actually listed (worst case: harmless re-compaction, never hidden
    # evidence).

    def uncompacted_tool_result_tokens(self) -> int:
        """Rough token volume of tool_result events NOT yet compacted.

        The curator gate keys on this (tracker 2.1). The old gate used
        total kb.json size, which only ever grows — once a game crossed
        the threshold, every synthesizer step paid a curator LLM call
        forever.
        """
        rows = self.query(
            "SELECT COALESCE(SUM(LENGTH(payload_json)), 0) AS n"
            " FROM events WHERE kind = ? AND id NOT IN"
            " (SELECT event_id FROM compacted_tool_results)",
            (EVT_TOOL_RESULT,),
        )
        return int(rows[0]["n"]) // 4 if rows else 0

    def uncompacted_tool_results(
        self, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Oldest-first tool_result events not yet compacted.

        Oldest-first (ties broken by id for replay-stable determinism) so
        consecutive curator passes drain the backlog contiguously instead
        of re-summarising the same recent window.
        """
        return self.query(
            "SELECT id, ts, payload_json FROM events"
            " WHERE kind = ? AND id NOT IN"
            " (SELECT event_id FROM compacted_tool_results)"
            " ORDER BY ts ASC, id ASC LIMIT ?",
            (EVT_TOOL_RESULT, limit),
        )

    def consolidated_observations(
        self, limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Latest curator summaries, newest first (for the digest)."""
        rows = self.query(
            "SELECT id, ts, payload_json FROM events WHERE kind = ?"
            " ORDER BY ts DESC LIMIT ?",
            (EVT_CONSOLIDATED, limit),
        )
        out = []
        for r in rows:
            try:
                payload = json.loads(r["payload_json"])
            except Exception:  # noqa: BLE001
                continue
            out.append({
                "event_id": r["id"],
                "ts": r["ts"],
                "summary": str(payload.get("summary") or ""),
                "addresses_kept": payload.get("addresses_kept") or [],
                "events_compacted": payload.get("events_compacted") or [],
            })
        return out

    def recent_run_summaries(
        self, question: str | None = None, limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Recent completed runs, prioritising the same exact question."""
        limit = max(1, min(50, int(limit)))
        if question:
            return self.query(
                "SELECT * FROM run_summaries"
                " ORDER BY CASE WHEN question = ? THEN 0 ELSE 1 END,"
                " completed_at DESC LIMIT ?",
                (question, limit),
            )
        return self.query(
            "SELECT * FROM run_summaries"
            " ORDER BY completed_at DESC LIMIT ?",
            (limit,),
        )

    def recent_tool_excerpts(
        self, limit: int = 12, max_chars_per: int = 4_000,
        exclude_compacted: bool = False,
    ) -> list[dict[str, Any]]:
        """Most recent tool_result events, each lightly truncated.

        `exclude_compacted` hides results a consolidated summary has
        GENUINELY covered (exact id membership — tracker 2.1); evidence a
        legacy summary did not list stays visible.
        """
        if exclude_compacted:
            rows = self.query(
                "SELECT id, ts, source, payload_json FROM events"
                " WHERE kind = ? AND id NOT IN"
                " (SELECT event_id FROM compacted_tool_results)"
                " ORDER BY ts DESC LIMIT ?",
                (EVT_TOOL_RESULT, limit),
            )
        else:
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
        """Return the raw partial-asm content (head) if one was ingested.

        Resolves the source path via the derived events table and caches
        the built head — the old implementation re-read and JSON-parsed
        the entire kb.json per digest build (tracker 1.5/1.6).
        """
        if self._partial_asm_cache and self._partial_asm_cache[0] == max_lines:
            return self._partial_asm_cache[1]

        head = ""
        for r in self.query(
            "SELECT payload_json FROM events WHERE kind = ?",
            (EVT_INGEST_PARTIAL_ASM,),
        ):
            try:
                payload = json.loads(r["payload_json"])
            except Exception:  # noqa: BLE001
                continue
            path = Path(payload.get("path", ""))
            if not path.exists():
                continue
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                break
            head = "\n".join(lines[:max_lines])
            if len(lines) > max_lines:
                head += f"\n... [{len(lines) - max_lines} more lines]"
            break

        self._partial_asm_cache = (max_lines, head)
        return head

    def digest_for_question(
        self, question: str, *, max_chars: int = 32_000,
    ) -> str:
        """Build a question-focused KB evidence sheet (markdown).

        This is the *single source of truth* for the Analyst and Critic
        contexts. Keeping it here (instead of inline in `analyst_node`)
        means both agents see exactly the same evidence — which is
        important for the critic to actually catch unsupported claims.

        Memoized on (question, events_total, max_chars): all digest
        inputs derive from events, so an unchanged event count means an
        identical digest (tracker 1.5 — planner/executor/analyst used to
        trigger a full rebuild each, per step).
        """
        s = self.stats()
        cache_key = (question, s["events_total"], max_chars)
        if self._digest_cache and self._digest_cache[0] == cache_key:
            return self._digest_cache[1]
        terms = self._question_terms(question)

        sections: list[str] = []

        # Freshness flag (tracker 2.2): warn when the latest dump ingest
        # replaced earlier content — older dump-derived analysis in this
        # KB may describe bytes that no longer exist.
        dump_refreshed_note = ""
        latest_dump = self.query(
            "SELECT payload_json FROM events WHERE kind = ?"
            " ORDER BY ts DESC LIMIT 1",
            (EVT_INGEST_DUMP,),
        )
        if latest_dump:
            try:
                _ld = json.loads(latest_dump[0]["payload_json"])
            except Exception:  # noqa: BLE001
                _ld = {}
            if _ld.get("refreshed_from"):
                dump_refreshed_note = (
                    "\n- dump_refreshed: True — the dump file's content "
                    "changed since an earlier ingest; dump-derived facts "
                    "recorded before the refresh may be stale."
                )

        sections.append(
            f"## KB stats\n"
            f"- events_total: {s['events_total']}\n"
            f"- labels: {s['labels']}\n"
            f"- text_docs: {s.get('text_docs', 0)}\n"
            f"- dump_bytes: {s['dump_bytes']}{dump_refreshed_note}\n"
            f"- semantic_kb_enabled: {s.get('semantic_kb_enabled')}\n"
            f"- semantic_kb_ready (embeddings hydrated): "
            f"{s.get('semantic_kb_ready')} "
            f"(chunks≈{s.get('semantic_kb_chunks')})\n"
            f"- events_by_kind: {s.get('events_by_kind', {})}\n"
            f"- question_terms: {terms or '(none extracted)'}"
        )

        # Completed-run outcomes (tracker 5.4): the planner receives these
        # through the same digest as all other durable evidence, so it can
        # avoid repeating a prior accepted investigation and can target the
        # open gap from a low-confidence/replanned one.
        prior_runs = self.recent_run_summaries(question, limit=5)
        if prior_runs:
            run_lines: list[str] = []
            for run in prior_runs:
                same = "same question" if run.get("question") == question else "prior question"
                excerpt = re.sub(
                    r"\s+", " ", str(run.get("answer_excerpt") or ""),
                ).strip()[:500]
                run_lines.append(
                    f"- [{same}] `{run.get('run_id')}` verdict="
                    f"{run.get('verdict')} confidence="
                    f"{float(run.get('confidence') or 0.0):.2f} "
                    f"iterations={int(run.get('iterations') or 0)} "
                    f"cost=${float(run.get('cost_usd') or 0.0):.4f}\n"
                    f"  question: {run.get('question') or ''}\n"
                    f"  answer: {excerpt or '(no answer excerpt)'}"
                )
            sections.append("## Prior completed runs\n" + "\n".join(run_lines))

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
                conf = r.get("confidence")
                conf_str = (
                    f" (conf={float(conf):.2f})"
                    if isinstance(conf, (int, float)) else ""
                )
                r_lines.append(
                    f"- ${r['start']:04X}-${r['end']:04X}  "
                    f"{r.get('name') or '(unnamed)'}{conf_str} — "
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

        # Curator output (tracker 2.1): before this section existed, the
        # consolidated summaries were written but NEVER read by anyone in
        # the default configuration — old evidence survived only as
        # extracted labels/routines. This restores long-term memory.
        consolidated = self.consolidated_observations(limit=3)
        if consolidated:
            c_lines = []
            for c in consolidated:
                body = c["summary"][:2_000]
                addrs = ", ".join(str(a) for a in c["addresses_kept"][:12])
                c_lines.append(
                    f"### consolidated `{c['event_id']}`"
                    + (f"  (addresses: {addrs})" if addrs else "")
                    + f"\n{body}"
                )
            sections.append(
                "## Consolidated observations (older evidence, compacted)\n"
                + "\n\n".join(c_lines)
            )

        # Build the base digest (without raw tool dumps) first so we know
        # how much budget remains for the verbose tool excerpts.
        base_digest = "\n\n".join(sections)
        budget_for_recent = max_chars - len(base_digest) - 100

        # Exclude genuinely-compacted results — the consolidated section
        # above carries them; raw excerpts show only fresh evidence.
        recent = self.recent_tool_excerpts(
            limit=12, max_chars_per=4_000, exclude_compacted=True,
        )
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
        self._digest_cache = (cache_key, digest)
        return digest
