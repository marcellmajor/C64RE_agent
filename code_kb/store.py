"""`CodeKnowledgeStore` — the separate, layered code-comprehension KB.

Mirrors the structure of `memory.KnowledgeStore` (append-only JSONL event
log + derived SQLite view that's rebuilt on every `load_or_init`) but is
purpose-built for the layered code-comprehension pipeline:

    Layer 0  – deterministic ground truth (no LLM, immutable)
    Layer 1  – per-window semantic annotations (LLM)
    Layer 2  – global behavioural groupings (LLM)
    Layer 3  – critique flags (LLM)

The store explicitly enforces "Layer 0 is immutable": once a Layer-0
annotation has been written, higher layers can `supersede` (logically
shadow) it but never remove it. The SQLite view always reflects the
*latest* layer per (kind, start_addr, end_addr) tuple.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from code_kb.schema import (
    ANN_ASM_DOC,
    ANN_CLASSIFY,
    ANN_DATAREF,
    ANN_DISASM,
    ANN_GROUP,
    ANN_HYPOTHESIS,
    ANN_INDIRECT,
    ANN_LABEL,
    ANN_ROUTINE,
    ANN_SMC,
    ANN_XREF,
    CLEAR_DDL,
    EVT_ANNOTATION,
    EVT_DISASM_WINDOW,
    EVT_INGEST_ASM,
    EVT_INGEST_DUMP,
    EVT_INVALIDATE_SOURCE,
    EVT_LAYER_RUN,
    SCHEMA_DDL,
)


# Module-level cache so multiple nodes in the same process share a handle.
_CODE_STORE_CACHE: dict[str, "CodeKnowledgeStore"] = {}
_CACHE_LOCK = threading.Lock()


# Pre-hardening layer-0 classification payloads lacked `source_file`, but
# their evidence templates embedded the source path verbatim. Recovering
# it at replay time lets path-based invalidation reach legacy rows
# without touching the append-only event log (Phase 2 hardening).
_LEGACY_CLASSIFY_PATTERNS = (
    re.compile(r"consecutive parsed instructions in (?P<path>.+)$"),
    re.compile(r"gap marker in (?P<path>.+?) \(\d+ bytes\)$"),
)


def _recover_classify_source(payload_inner: dict[str, Any]) -> str | None:
    """Best-effort provenance for legacy `classify` payloads."""
    evidence = str(payload_inner.get("evidence_text") or "")
    for pat in _LEGACY_CLASSIFY_PATTERNS:
        m = pat.search(evidence)
        if m:
            return m.group("path")
    return None


def get_code_store(handle: str | Path) -> "CodeKnowledgeStore":
    """Return a cached `CodeKnowledgeStore` for the given root directory."""
    key = str(Path(handle))
    with _CACHE_LOCK:
        store = _CODE_STORE_CACHE.get(key)
        if store is None:
            store = CodeKnowledgeStore.load_or_init(Path(key))
            _CODE_STORE_CACHE[key] = store
        return store


class _BulkWriter:
    """Context manager: defer sqlite commits + JSONL flushes during ingest."""

    def __init__(self, store: "CodeKnowledgeStore") -> None:
        self._store = store

    def __enter__(self) -> "CodeKnowledgeStore":
        self._store._bulk_depth += 1
        return self._store

    def __exit__(self, exc_type, exc, tb) -> None:
        s = self._store
        s._bulk_depth -= 1
        if s._bulk_depth > 0:
            return
        # Flush JSONL handle (if any) and commit the sqlite transaction.
        if s._bulk_jsonl_handle is not None:
            try:
                s._bulk_jsonl_handle.flush()
                s._bulk_jsonl_handle.close()
            finally:
                s._bulk_jsonl_handle = None
        if s._db is not None:
            s._db.commit()


# --------------------------------------------------------------------------- #
# Annotation dataclass — carried around by the layer builders / node.
# --------------------------------------------------------------------------- #


@dataclass
class Annotation:
    layer: int
    kind: str
    start_addr: int
    end_addr: int
    producer: str
    payload: dict[str, Any]
    confidence: float = 0.5
    evidence: list[str] | None = None
    supersedes: list[str] | None = None
    flags: list[str] | None = None
    id: str | None = None
    ts: str | None = None

    def to_event(self, source: str) -> tuple[str, str, dict[str, Any]]:
        rec = {
            "id": self.id,
            "layer": int(self.layer),
            "kind": str(self.kind),
            "start_addr": int(self.start_addr) & 0xFFFF,
            "end_addr": int(self.end_addr) & 0xFFFF,
            "producer": str(self.producer),
            "confidence": float(self.confidence),
            "payload": dict(self.payload or {}),
            "evidence": list(self.evidence or []),
            "supersedes": list(self.supersedes or []),
            "flags": list(self.flags or []),
        }
        return EVT_ANNOTATION, source, rec


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class CodeKnowledgeStore:
    """Layered code-comprehension knowledge store for one game session."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.kb_json = root / "code_kb.json"
        self.kb_sqlite = root / "code_kb.sqlite"
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._dump_bytes: bytes | None = None
        self._dump_path: Path | None = None
        # Batch-mode bookkeeping. While `_bulk_depth > 0`, append_event
        # keeps the JSONL file open and skips per-row commits — a
        # ~40-100x speedup for big partial-asm ingestion.
        self._bulk_depth: int = 0
        self._bulk_jsonl_handle = None

    # ---- lifecycle ------------------------------------------------------- #

    @classmethod
    def load_or_init(cls, root: Path) -> "CodeKnowledgeStore":
        root.mkdir(parents=True, exist_ok=True)
        s = cls(root)
        s._open_sqlite()
        s._replay_events()
        s._warm_dump_cache()
        return s

    def _open_sqlite(self) -> None:
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
        for evt in reversed(list(self._iter_events())):
            if evt["kind"] == EVT_INGEST_DUMP:
                p = Path(evt["payload"].get("path") or "")
                if p.exists():
                    self._dump_bytes = p.read_bytes()
                    self._dump_path = p
                break

    # ---- writes ---------------------------------------------------------- #

    def append_event(
        self, kind: str, source: str, payload: dict[str, Any],
    ) -> str:
        evt = {
            "id": uuid.uuid4().hex[:12],
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "source": source,
            "payload": payload,
        }
        with self._lock:
            self._write_jsonl_locked(evt)
            self._apply_event_to_sqlite(evt)
            assert self._db is not None
            if self._bulk_depth == 0:
                self._db.commit()
        return evt["id"]

    def append_annotation(
        self, ann: Annotation, source: str,
    ) -> str:
        if ann.id is None:
            ann.id = uuid.uuid4().hex[:12]
        if ann.ts is None:
            ann.ts = datetime.now(timezone.utc).isoformat()
        kind, src, payload = ann.to_event(source)
        evt = {
            "id": ann.id,
            "ts": ann.ts,
            "kind": kind,
            "source": src,
            "payload": payload,
        }
        with self._lock:
            self._write_jsonl_locked(evt)
            self._apply_event_to_sqlite(evt)
            assert self._db is not None
            if self._bulk_depth == 0:
                self._db.commit()
        return ann.id

    def _write_jsonl_locked(self, evt: dict[str, Any]) -> None:
        line = json.dumps(evt) + "\n"
        if self._bulk_depth > 0:
            if self._bulk_jsonl_handle is None:
                self._bulk_jsonl_handle = self.kb_json.open("a")
            self._bulk_jsonl_handle.write(line)
        else:
            with self.kb_json.open("a") as f:
                f.write(line)

    def bulk_writes(self) -> "_BulkWriter":
        """Batch many appends into a single sqlite commit + jsonl flush.

        Use as `with store.bulk_writes(): ...` around large ingestion
        loops. Nested usage is supported and only the outermost block
        commits.
        """
        return _BulkWriter(self)

    # ---- ingestion convenience ------------------------------------------ #

    def ingest_dump(self, path: Path) -> tuple[str | None, bool]:
        """Content-aware idempotent dump ingest (tracker 2.2).

        Returns ``(event_id_or_None, changed)``: `event_id` is None when
        the dump is already current; `changed` is True when the file's
        content differs from a previously ingested version under the
        same path (the caller should invalidate dump-derived rows).
        """
        path = Path(path)
        if not path.exists():
            return None, False
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()

        prev_sha: str | None = None
        seen_path = False
        for evt in self._iter_events():
            if (
                evt["kind"] == EVT_INGEST_DUMP
                and evt["payload"].get("path") == str(path)
            ):
                seen_path = True
                prev_sha = evt["payload"].get("sha256")

        self._dump_bytes = data
        self._dump_path = path
        if seen_path and prev_sha == sha:
            return None, False

        payload: dict[str, Any] = {
            "path": str(path), "size": len(data), "sha256": sha,
        }
        changed = False
        if seen_path:
            if prev_sha is not None:
                payload["refreshed_from"] = prev_sha
                changed = True
            else:
                payload["sha_backfill"] = True  # legacy event lacked a hash
        return self.append_event(EVT_INGEST_DUMP, "load_inputs", payload), changed

    def ingest_asm_doc(
        self, path: Path, *, content: str, instructions: int, routines: int,
    ) -> str:
        """Record a partial-asm file as a row in `asm_docs` plus an event."""
        size = len(content)
        try:
            mtime = Path(path).stat().st_mtime
        except OSError:
            mtime = 0.0
        return self.append_event(
            EVT_INGEST_ASM, "load_inputs",
            {
                "path": str(path),
                "size": size,
                "mtime": mtime,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "content": content,
                "instructions": int(instructions),
                "routines": int(routines),
            },
        )

    def already_ingested_asm(self, path: Path) -> bool:
        """Path-only check (legacy). Prefer `asm_freshness` (tracker 2.2)."""
        path = str(Path(path))
        for evt in self._iter_events():
            if (
                evt["kind"] == EVT_INGEST_ASM
                and evt["payload"].get("path") == path
            ):
                return True
        return False

    def asm_freshness(self, path: Path, content: str) -> str:
        """"new" | "current" | "changed" for an asm file's content.

        `already_ingested_asm` was path-only — an asm file edited under
        the same filename was silently skipped, leaving stale Layer-0
        facts (tracker 2.2). Compares the content sha256 against the
        latest ingest event for the path; legacy events without a hash
        fall back to size+mtime.
        """
        latest: dict[str, Any] | None = None
        for evt in self._iter_events():
            if (
                evt["kind"] == EVT_INGEST_ASM
                and evt["payload"].get("path") == str(Path(path))
            ):
                latest = evt["payload"]
        if latest is None:
            return "new"

        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        prev_sha = latest.get("sha256")
        if prev_sha is not None:
            return "current" if prev_sha == sha else "changed"

        # Legacy event without a hash: size + mtime approximation.
        try:
            mtime = Path(path).stat().st_mtime
        except OSError:
            mtime = 0.0
        if (
            int(latest.get("size") or -1) == len(content)
            and float(latest.get("mtime") or 0.0) >= mtime
        ):
            return "current"
        return "changed"

    def invalidate_source(self, source_file: str, *, like: bool = False) -> str:
        """Append an event that removes everything *source_file* produced.

        Event-based so a full rebuild replays the deletion BEFORE the
        re-ingest events that follow it, reproducing the same state
        (tracker 2.2). `like=True` treats *source_file* as a SQL LIKE
        pattern (used to drop `capstone:%`/`vice:%` disasm windows when
        the dump content changes).

        Replay semantics (Phase 2 hardening, final): the source's
        CANONICAL annotation rows are deleted (all kinds — routines,
        labels, instructions, hypotheses, xrefs, SMC sites,
        classifications), then every typed projection is re-materialized
        from the surviving annotations in stable (ts, id) order. That
        restores a surviving source's fact on singleton-keyed views
        where the invalidated source had overwritten it, and preserves
        per-source xref/SMC provenance rows exactly.
        """
        return self.append_event(
            EVT_INVALIDATE_SOURCE, "load_inputs",
            {"source_file": str(source_file), "like": bool(like)},
        )

    # ---- reads ----------------------------------------------------------- #

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        assert self._db is not None
        rows = self._db.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def has_dump(self) -> bool:
        return self._dump_bytes is not None and len(self._dump_bytes) > 0

    def read_bytes(self, addr: int, length: int) -> bytes:
        if self._dump_bytes is None:
            return b""
        end = min(addr + length, len(self._dump_bytes))
        return self._dump_bytes[addr:end]

    def full_dump(self) -> bytes:
        if not self._dump_bytes:
            return b""
        if len(self._dump_bytes) >= 0x10000:
            return self._dump_bytes[:0x10000]
        buf = bytearray(0x10000)
        buf[: len(self._dump_bytes)] = self._dump_bytes
        return bytes(buf)

    def stats(self) -> dict[str, Any]:
        assert self._db is not None
        events_total = self._db.execute(
            "SELECT COUNT(*) AS n FROM events"
        ).fetchone()["n"]
        by_layer = {
            r["layer"]: r["n"]
            for r in self._db.execute(
                "SELECT layer, COUNT(*) AS n FROM annotations GROUP BY layer"
            ).fetchall()
        }
        by_kind = {
            r["kind"]: r["n"]
            for r in self._db.execute(
                "SELECT kind, COUNT(*) AS n FROM annotations GROUP BY kind"
            ).fetchall()
        }
        routines = self._db.execute(
            "SELECT COUNT(*) AS n FROM code_routines"
        ).fetchone()["n"]
        xrefs = self._db.execute(
            "SELECT COUNT(*) AS n FROM code_xrefs"
        ).fetchone()["n"]
        smcs = self._db.execute(
            "SELECT COUNT(*) AS n FROM code_smc_sites"
        ).fetchone()["n"]
        insns = self._db.execute(
            "SELECT COUNT(*) AS n FROM instructions"
        ).fetchone()["n"]
        asm_docs_total = self._db.execute(
            "SELECT COUNT(*) AS n FROM asm_docs"
        ).fetchone()["n"]
        labels = self._db.execute(
            "SELECT COUNT(*) AS n FROM code_labels"
        ).fetchone()["n"]
        data_refs = self._db.execute(
            "SELECT COUNT(*) AS n FROM code_data_refs"
        ).fetchone()["n"]
        return {
            "events_total": int(events_total),
            "annotations_by_layer": {int(k): int(v) for k, v in by_layer.items()},
            "annotations_by_kind": {str(k): int(v) for k, v in by_kind.items()},
            "routines": int(routines),
            "xrefs": int(xrefs),
            "smc_sites": int(smcs),
            "data_refs": int(data_refs),
            "instructions": int(insns),
            "asm_docs": int(asm_docs_total),
            "labels": int(labels),
            "dump_bytes": len(self._dump_bytes) if self._dump_bytes else 0,
        }

    # ---- replay → SQLite (rebuilds derived view) ------------------------- #

    def _apply_event_to_sqlite(self, evt: dict[str, Any]) -> None:
        assert self._db is not None
        self._db.execute(
            "INSERT OR REPLACE INTO events"
            " (id, ts, kind, source, payload_json) VALUES (?, ?, ?, ?, ?)",
            (
                evt["id"], evt["ts"], evt["kind"], evt["source"],
                json.dumps(evt["payload"]),
            ),
        )

        if evt["kind"] == EVT_INGEST_ASM:
            p = evt["payload"]
            self._db.execute(
                "INSERT OR REPLACE INTO asm_docs"
                " (path, size, mtime, content, instructions, routines,"
                "  annotation_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    p.get("path"),
                    int(p.get("size", 0)),
                    float(p.get("mtime") or 0.0),
                    p.get("content", ""),
                    int(p.get("instructions") or 0),
                    int(p.get("routines") or 0),
                    None,
                ),
            )

        if evt["kind"] == EVT_ANNOTATION:
            self._apply_annotation_to_sqlite(evt)

        if evt["kind"] == EVT_INVALIDATE_SOURCE:
            # Source file changed on disk (tracker 2.2, hardened): drop
            # the source's CANONICAL annotations, then re-materialize
            # every typed projection from the surviving annotations. The
            # earlier per-table DELETE could not restore a surviving
            # source's fact after the overwriting source was invalidated
            # (singleton keys: routines/labels/instructions/class).
            # Runs identically live, after reopen, and under full replay.
            p = evt["payload"]
            pattern = str(p.get("source_file") or "")
            if pattern:
                op = "LIKE" if p.get("like") else "="
                self._db.execute(
                    f"DELETE FROM annotations WHERE source_file {op} ?",  # noqa: S608
                    (pattern,),
                )
                self._db.execute(
                    f"DELETE FROM asm_docs WHERE path {op} ?",
                    (pattern,),
                )
                # Conservative sweep (documented): deterministic layer-0
                # classification annotations whose provenance could NOT
                # be recovered (hand-altered legacy payloads) cannot be
                # judged current after ANY source changed — drop them so
                # stale facts are never presented as current; a re-ingest
                # re-creates them with full provenance.
                self._db.execute(
                    "DELETE FROM annotations WHERE kind = ? AND layer = 0"
                    " AND producer = 'deterministic'"
                    " AND source_file IS NULL",
                    (ANN_CLASSIFY,),
                )
                self._rematerialize_projections()

        if evt["kind"] in (EVT_LAYER_RUN, EVT_DISASM_WINDOW, EVT_INGEST_DUMP):
            return

    def _apply_annotation_to_sqlite(self, evt: dict[str, Any]) -> None:
        assert self._db is not None
        p = evt["payload"]
        ann_id = evt["id"]
        layer = int(p.get("layer", 0))
        kind = str(p.get("kind", "?"))
        start_addr = int(p.get("start_addr", 0)) & 0xFFFF
        end_addr = int(p.get("end_addr", start_addr)) & 0xFFFF
        producer = str(p.get("producer", "unknown"))
        confidence = float(p.get("confidence", 0.5))
        payload_inner = p.get("payload") or {}
        flags = p.get("flags") or []
        # Provenance for source invalidation (Phase 2 hardening) — every
        # annotation row carries the source_file its payload declares.
        # Legacy pre-hardening `classify` payloads lacked it: recover it
        # from their evidence template so path-based invalidation reaches
        # them too (the append-only event log itself is never modified).
        source_file = payload_inner.get("source_file")
        if source_file is None and kind == ANN_CLASSIFY:
            source_file = _recover_classify_source(payload_inner)

        # Append-order sequence (Phase 2 final hardening): replay assigns
        # 1..N in event-log order; live appends continue with MAX+1
        # (survivors keep their seq after invalidation deletes, so the
        # relative order of surviving annotations never changes — the
        # exact invariant rematerialization sorts by). A re-apply of an
        # already-known annotation id keeps its original seq.
        prev = self._db.execute(
            "SELECT seq FROM annotations WHERE id = ?", (ann_id,),
        ).fetchone()
        if prev is not None and prev["seq"] is not None:
            seq = int(prev["seq"])
        else:
            row = self._db.execute(
                "SELECT COALESCE(MAX(seq), 0) AS m FROM annotations",
            ).fetchone()
            seq = int(row["m"]) + 1

        self._db.execute(
            "INSERT OR REPLACE INTO annotations"
            " (id, layer, kind, start_addr, end_addr, producer, confidence,"
            "  payload_json, evidence_json, supersedes_json, flags_json,"
            "  source_file, seq, ts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ann_id, layer, kind, start_addr, end_addr, producer,
                confidence,
                json.dumps(payload_inner),
                json.dumps(p.get("evidence") or []),
                json.dumps(p.get("supersedes") or []),
                json.dumps(flags),
                source_file,
                seq,
                evt["ts"],
            ),
        )

        self._project_annotation(
            ann_id=ann_id, layer=layer, kind=kind, start_addr=start_addr,
            end_addr=end_addr, producer=producer, confidence=confidence,
            payload_inner=payload_inner, flags=flags,
            source_file=source_file,
        )

    def _rematerialize_projections(self) -> None:
        """Rebuild every typed projection from the surviving canonical
        annotations, in TRUE append order (`seq`).

        Called after a source invalidation deleted canonical rows: the
        singleton-keyed views (`code_routines`, `code_labels`,
        `instructions`, `code_class`) may have been last written by the
        invalidated source even though another source still asserts the
        same fact — replaying the survivors restores that provenance
        instead of leaving the key empty (Phase 2 hardening).

        Ordering is the explicit `seq` column, never `(ts, id)`:
        timestamps tie within a bulk ingest and event ids are random
        UUID fragments, so sorting by them could reverse the append
        order of equal-timestamp annotations and flip which fact wins a
        singleton key (reproduced defect). `seq` survives deletions
        unchanged, so later annotations keep winning exactly as they did
        before the invalidation — live, after reopen, and under full
        event-log replay alike.
        """
        assert self._db is not None
        for table in ("code_routines", "code_labels", "instructions",
                      "hypotheses", "code_xrefs", "code_smc_sites",
                      "code_class", "code_data_refs"):
            self._db.execute(f"DELETE FROM {table}")  # noqa: S608 — fixed tuple
        rows = self._db.execute(
            "SELECT id, layer, kind, start_addr, end_addr, producer,"
            " confidence, payload_json, flags_json, source_file"
            " FROM annotations ORDER BY seq ASC",
        ).fetchall()
        for r in rows:
            try:
                payload_inner = json.loads(r["payload_json"] or "{}")
            except Exception:  # noqa: BLE001
                payload_inner = {}
            try:
                row_flags = json.loads(r["flags_json"] or "[]")
            except Exception:  # noqa: BLE001
                row_flags = []
            self._project_annotation(
                ann_id=r["id"], layer=int(r["layer"]), kind=str(r["kind"]),
                start_addr=int(r["start_addr"]), end_addr=int(r["end_addr"]),
                producer=str(r["producer"]),
                confidence=float(r["confidence"] or 0.5),
                payload_inner=payload_inner, flags=row_flags,
                source_file=r["source_file"],
            )

    def _project_annotation(
        self,
        *,
        ann_id: str,
        layer: int,
        kind: str,
        start_addr: int,
        end_addr: int,
        producer: str,
        confidence: float,
        payload_inner: dict[str, Any],
        flags: list[Any],
        source_file: str | None,
    ) -> None:
        """Project one canonical annotation into the typed views.

        Factored out of `_apply_annotation_to_sqlite` so
        `_rematerialize_projections` can replay projections from the
        surviving canonical rows after a source invalidation.
        """
        assert self._db is not None
        # ---- project into typed views ----
        if kind == ANN_ROUTINE:
            self._db.execute(
                "INSERT OR REPLACE INTO code_routines"
                " (start_addr, end_addr, name, summary, source_file, entries_json,"
                "  exits_json, size_bytes, annotation_id, confidence)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    start_addr, end_addr,
                    payload_inner.get("name"),
                    payload_inner.get("summary"),
                    source_file,
                    json.dumps(payload_inner.get("entries") or []),
                    json.dumps(payload_inner.get("exits") or []),
                    int(payload_inner.get("size_bytes") or (end_addr - start_addr + 1)),
                    ann_id,
                    confidence,
                ),
            )

        elif kind == ANN_XREF:
            src = int(payload_inner.get("src_addr") or start_addr) & 0xFFFF
            dst = payload_inner.get("dst_addr")
            via = payload_inner.get("via_vector")
            xkind = str(payload_inner.get("xref_kind") or "jmp")
            self._upsert_xref(
                src=src,
                dst=int(dst) & 0xFFFF if isinstance(dst, int) else None,
                via=int(via) & 0xFFFF if isinstance(via, int) else None,
                kind=xkind,
                ann_id=ann_id,
                source_file=source_file,
            )

        elif kind == ANN_SMC:
            # NULL-safe per-(site, source) dedup, mirroring xrefs: the
            # same site from two sources keeps two provenance rows so
            # invalidating one source preserves the other's.
            smc_vals = (
                int(payload_inner.get("src_addr") or start_addr) & 0xFFFF,
                (
                    int(payload_inner["dst_addr"]) & 0xFFFF
                    if isinstance(payload_inner.get("dst_addr"), int)
                    else None
                ),
                payload_inner.get("mnemonic"),
                payload_inner.get("operand"),
                payload_inner.get("smc_kind", "unknown"),
                source_file,
                ann_id,
            )
            self._db.execute(
                "INSERT INTO code_smc_sites"
                " (src_addr, dst_addr, mnemonic, operand, smc_kind,"
                "  source_file, annotation_id)"
                " SELECT ?, ?, ?, ?, ?, ?, ?"
                " WHERE NOT EXISTS ("
                "   SELECT 1 FROM code_smc_sites"
                "   WHERE src_addr = ? AND dst_addr IS ? AND smc_kind IS ?"
                "     AND source_file IS ?"
                " )",
                smc_vals + (smc_vals[0], smc_vals[1], smc_vals[4],
                            source_file),
            )

        elif kind == ANN_CLASSIFY:
            for off in range(start_addr, end_addr + 1):
                self._db.execute(
                    "INSERT OR REPLACE INTO code_class"
                    " (addr, classification, confidence, evidence,"
                    "  source_file, annotation_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        off & 0xFFFF,
                        payload_inner.get("classification", "ambiguous"),
                        confidence,
                        payload_inner.get("evidence_text"),
                        source_file,
                        ann_id,
                    ),
                )

        elif kind == ANN_DATAREF:
            # Per-(ref, source) dedup, NULL-safe — mirrors xref/SMC so
            # invalidating one source keeps another's data refs (3.4).
            dst = payload_inner.get("dst_addr")
            if isinstance(dst, int):
                dr = (
                    int(payload_inner.get("src_addr") or start_addr) & 0xFFFF,
                    int(dst) & 0xFFFF,
                    str(payload_inner.get("access") or "r"),
                    payload_inner.get("index"),
                    1 if payload_inner.get("indirect") else 0,
                    source_file,
                    ann_id,
                )
                self._db.execute(
                    "INSERT INTO code_data_refs"
                    " (src_addr, dst_addr, access, index_reg, indirect,"
                    "  source_file, annotation_id)"
                    " SELECT ?, ?, ?, ?, ?, ?, ?"
                    " WHERE NOT EXISTS ("
                    "   SELECT 1 FROM code_data_refs"
                    "   WHERE src_addr = ? AND dst_addr = ? AND access = ?"
                    "     AND index_reg IS ? AND source_file IS ?"
                    " )",
                    dr + (dr[0], dr[1], dr[2], dr[3], source_file),
                )

        elif kind == ANN_LABEL:
            self._db.execute(
                "INSERT OR REPLACE INTO code_labels"
                " (addr, name, source_file, annotation_id)"
                " VALUES (?, ?, ?, ?)",
                (
                    start_addr,
                    str(payload_inner.get("name") or ""),
                    source_file,
                    ann_id,
                ),
            )

        elif kind == ANN_DISASM:
            for ins in payload_inner.get("instructions") or []:
                try:
                    addr = int(ins["addr"]) & 0xFFFF
                except (KeyError, TypeError, ValueError):
                    continue
                self._db.execute(
                    "INSERT OR REPLACE INTO instructions"
                    " (addr, bytes_hex, mnemonic, operand, size_bytes,"
                    "  source_file, annotation_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        addr,
                        str(ins.get("bytes_hex") or ""),
                        str(ins.get("mnemonic") or "?").lower(),
                        ins.get("operand"),
                        int(ins.get("size_bytes") or 1),
                        source_file,
                        ann_id,
                    ),
                )

        elif kind in (ANN_HYPOTHESIS, ANN_GROUP):
            self._db.execute(
                "INSERT OR REPLACE INTO hypotheses"
                " (annotation_id, layer, start_addr, end_addr, text,"
                "  name_suggestion, idiom_match, hardware_touched_json,"
                "  routine_id, source_file, confidence, flags_json, producer)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ann_id, layer, start_addr, end_addr,
                    str(payload_inner.get("text") or ""),
                    payload_inner.get("name_suggestion"),
                    payload_inner.get("idiom_match"),
                    json.dumps(payload_inner.get("hardware_touched") or []),
                    payload_inner.get("routine_id"),
                    source_file,
                    confidence,
                    json.dumps(flags or []),
                    producer,
                ),
            )
            if kind == ANN_HYPOTHESIS and int(layer) == 1:
                self._promote_layer1_name(
                    start_addr=start_addr,
                    ann_id=ann_id,
                    confidence=confidence,
                    suggested_name=payload_inner.get("name_suggestion"),
                    source_file=source_file,
                )

        elif kind == ANN_INDIRECT:
            src = int(payload_inner.get("src_addr") or start_addr) & 0xFFFF
            via = payload_inner.get("via_vector")
            tgt = payload_inner.get("resolved_target")
            # source_file passed through (final hardening: indirect-jump
            # xrefs projected with NULL provenance survived invalidation).
            self._upsert_xref(
                src=src,
                dst=int(tgt) & 0xFFFF if isinstance(tgt, int) else None,
                via=int(via) & 0xFFFF if isinstance(via, int) else None,
                kind="jmp_indirect",
                ann_id=ann_id,
                source_file=source_file,
            )

    def _upsert_xref(
        self,
        *,
        src: int,
        dst: int | None,
        via: int | None,
        kind: str,
        ann_id: str,
        source_file: str | None = None,
    ) -> None:
        """Insert a cross-reference row, skipping silently if an identical
        row from the SAME source already exists.

        SQLite treats NULL != NULL in uniqueness checks, so an explicit
        ``NOT EXISTS`` with the ``IS`` operator gives NULL-safe equality.
        Dedup is per (edge, source_file): the same edge asserted by two
        sources keeps two rows, so invalidating one source never deletes
        the other's provenance (Phase 2 hardening).
        """
        assert self._db is not None
        self._db.execute(
            "INSERT INTO code_xrefs"
            " (src_addr, dst_addr, via_vector, kind, source_file,"
            "  annotation_id)"
            " SELECT ?, ?, ?, ?, ?, ?"
            " WHERE NOT EXISTS ("
            "   SELECT 1 FROM code_xrefs"
            "   WHERE src_addr = ? AND dst_addr IS ? AND via_vector IS ?"
            "     AND kind = ? AND source_file IS ?"
            " )",
            (src, dst, via, kind, source_file, ann_id,
             src, dst, via, kind, source_file),
        )

    def _promote_layer1_name(
        self,
        *,
        start_addr: int,
        ann_id: str,
        confidence: float,
        suggested_name: Any,
        source_file: Any,
    ) -> None:
        """Promote strong Layer-1 name suggestions into display-level naming.

        Keeps deterministic labels/routine names unless they are generic
        `sub_xxxx` placeholders.
        """
        assert self._db is not None

        if confidence < 0.60:
            return
        if not isinstance(suggested_name, str):
            return
        name = suggested_name.strip()
        if not name:
            return
        # Conservative identifier gate; avoid storing prose as labels.
        import re as _re

        if not _re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{2,63}", name):
            return

        # Add a promoted label at the routine start for tools that resolve
        # labels separately from routine names.
        self._db.execute(
            "INSERT OR REPLACE INTO code_labels"
            " (addr, name, source_file, annotation_id) VALUES (?, ?, ?, ?)",
            (
                int(start_addr) & 0xFFFF,
                name,
                str(source_file) if source_file is not None else None,
                ann_id,
            ),
        )

        rows = self._db.execute(
            "SELECT name, confidence FROM code_routines WHERE start_addr = ? LIMIT 1",
            (int(start_addr) & 0xFFFF,),
        ).fetchall()
        if not rows:
            return
        current = rows[0]["name"]
        existing_conf = float(rows[0]["confidence"] or 0.0)
        cur = str(current or "").strip().lower()
        # Allow overwrite only if: placeholder name, or new confidence is higher.
        if cur and not cur.startswith("sub_") and confidence <= existing_conf:
            return

        self._db.execute(
            "UPDATE code_routines"
            "   SET name = ?, annotation_id = ?, confidence = ?"
            " WHERE start_addr = ?",
            (name, ann_id, confidence, int(start_addr) & 0xFFFF),
        )
        # Prune superseded lower-confidence hypotheses for this address.
        self._db.execute(
            "DELETE FROM hypotheses"
            " WHERE start_addr = ? AND layer = 1"
            "   AND annotation_id != ?"
            "   AND (confidence IS NULL OR confidence < ?)",
            (int(start_addr) & 0xFFFF, ann_id, confidence),
        )
