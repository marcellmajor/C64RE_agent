"""Turn KB SQLite rows / events into embeddable text chunks."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from memory.schema import (
    EVT_CONSOLIDATED,
    EVT_DATA_STRUCTURE,
    EVT_HYPOTHESIS,
    EVT_INGEST_TEXT,
    EVT_LABEL,
    EVT_ROUTINE,
    EVT_TOOL_RESULT,
)
from memory.semantic_config import KBSemanticConfig
from memory.text_chunking import chunk_text, clip_for_embedding


def _path_digest(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8", errors="replace")).hexdigest()[:16]


TEXT_CLIP = 12_000


def collect_chunks_from_derived_kb(
    query_fn: Any,
    cfg: KBSemanticConfig,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Return parallel lists: ids, metadata, embedding texts."""

    kinds = cfg.index_event_kinds
    ids: list[str] = []
    metas: list[dict[str, Any]] = []
    texts: list[str] = []

    if EVT_INGEST_TEXT in kinds:
        rows = query_fn(
            "SELECT path, content, size FROM text_docs ORDER BY path",
            (),
        )
        for row in rows:
            path_s = row["path"] or ""
            did = _path_digest(path_s)
            content = clip_for_embedding(row["content"] or "", TEXT_CLIP)
            for ci, chunk in enumerate(
                chunk_text(content, cfg.chunk_chars, cfg.chunk_overlap),
            ):
                ids.append(f"txt:{did}:{ci}")
                metas.append({
                    "facet": "text_doc",
                    "_path_digest": did,
                    "path": path_s,
                    "chunk_index": ci,
                    "size": row.get("size"),
                })
                texts.append(chunk)

    if EVT_LABEL in kinds:
        rows = query_fn(
            "SELECT addr, name, kind, confidence, source_event_id FROM labels "
            "ORDER BY addr",
            (),
        )
        for row in rows:
            blob = clip_for_embedding(
                _format_label_row(row),
                TEXT_CLIP,
            )
            eeid = row.get("source_event_id") or ""
            lk = eeid if eeid else f"{int(row['addr']):04X}-{row['name']}"
            ids.append(f"lbl:{lk}")
            metas.append({
                "facet": "label_table",
                "addr": int(row["addr"]),
                "name": row.get("name"),
                "kind": row.get("kind"),
                "confidence": row.get("confidence"),
                "source_event_id": eeid or None,
            })
            texts.append(blob)

    if EVT_ROUTINE in kinds:
        rows = query_fn(
            "SELECT start, end, name, summary, calls_to_json, called_by_json "
            "FROM routines ORDER BY start",
            (),
        )
        for row in rows:
            blob = clip_for_embedding(
                _format_routine_row(row),
                TEXT_CLIP,
            )
            st = int(row["start"])
            ids.append(f"rt:${st:04X}")
            metas.append({
                "facet": "routine_table",
                "start": st,
                "end": int(row.get("end") or row["start"]),
                "name": row.get("name"),
            })
            texts.append(blob)

    if EVT_DATA_STRUCTURE in kinds:
        rows = query_fn(
            "SELECT start, end, kind, fields_json FROM data_structures "
            "ORDER BY start",
            (),
        )
        for row in rows:
            blob = clip_for_embedding(
                _format_ds_row(row),
                TEXT_CLIP,
            )
            st = int(row["start"])
            ids.append(f"ds:${st:04X}")
            metas.append({
                "facet": "data_structure_table",
                "start": st,
                "end": int(row.get("end") or row["start"]),
                "kind": row.get("kind"),
            })
            texts.append(blob)

    if EVT_HYPOTHESIS in kinds:
        rows = query_fn(
            "SELECT id, text, status, evidence_json FROM hypotheses ORDER BY id",
            (),
        )
        for row in rows:
            blob = clip_for_embedding(
                _format_hypothesis_row(row),
                TEXT_CLIP,
            )
            hid = str(row["id"])
            ids.append(f"hyp:{hid}")
            metas.append({
                "facet": "hypothesis_table",
                "hypothesis_id": hid,
                "status": row.get("status"),
            })
            texts.append(blob)

    if EVT_CONSOLIDATED in kinds:
        rows = query_fn(
            "SELECT id, payload_json FROM events WHERE kind = ? ORDER BY ts",
            (EVT_CONSOLIDATED,),
        )
        for row in rows:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            blob = clip_for_embedding(
                _format_consolidated_payload(payload),
                TEXT_CLIP,
            )
            evt_id = str(row["id"])
            ids.append(f"cn:{evt_id}")
            metas.append({
                "facet": "consolidated_event",
                "event_id": evt_id,
            })
            texts.append(blob)

    if EVT_TOOL_RESULT in kinds:
        rows = query_fn(
            "SELECT id, payload_json FROM events WHERE kind = ? ORDER BY ts",
            (EVT_TOOL_RESULT,),
        )
        for row in rows:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            blob = clip_for_embedding(
                _format_tool_result_payload(payload),
                min(4096, TEXT_CLIP),
            )
            evt_id = str(row["id"])
            ids.append(f"tr:{evt_id}")
            metas.append({
                "facet": "tool_result_event",
                "event_id": evt_id,
                "tool": payload.get("tool"),
                "step_id": payload.get("step_id"),
                "ok": payload.get("ok"),
            })
            texts.append(blob)

    return ids, metas, texts


def incremental_chunks_from_event(
    evt: dict[str, Any],
    cfg: KBSemanticConfig,
) -> tuple[
    Callable[[dict[str, Any]], bool],
    list[str],
    list[dict[str, Any]],
    list[str],
]:
    """Return predicate to remove superseded semantic rows plus new embedding rows."""

    kind = evt.get("kind")
    ek = evt.get("id") or ""
    pid = evt.get("payload") or {}

    kinds = cfg.index_event_kinds
    if kind not in kinds:
        return (lambda _m: False, [], [], [])

    if kind == EVT_INGEST_TEXT:
        path_s = str(pid.get("path") or "")
        did = _path_digest(path_s)
        content = clip_for_embedding(pid.get("content") or "", TEXT_CLIP)
        preds: Callable[[dict[str, Any]], bool] = lambda m, ds=did: (
            m.get("facet") == "text_doc"
            and m.get("_path_digest") == ds
        )
        cid_list: list[str] = []
        meta_list: list[dict[str, Any]] = []
        text_list: list[str] = []
        for ci, chunk in enumerate(
            chunk_text(content, cfg.chunk_chars, cfg.chunk_overlap),
        ):
            cid_list.append(f"txt:{did}:{ci}")
            meta_list.append({
                "facet": "text_doc",
                "_path_digest": did,
                "path": path_s,
                "chunk_index": ci,
                "source_event_id": ek,
                "size": pid.get("size"),
            })
            text_list.append(chunk)
        return (preds, cid_list, meta_list, text_list)

    if kind == EVT_LABEL:
        addr = int(pid["addr"])
        name_s = str(pid.get("name") or "")
        pred = lambda m, a_=addr, n_=name_s: (
            m.get("facet") in {"label_live", "label_table"}
            and int(m.get("addr") or -1) == a_
            and str(m.get("name") or "") == n_
        )
        blob = clip_for_embedding(
            _format_label_payload(pid, ek),
            TEXT_CLIP,
        )
        cid = f"lbl_evt:{ek}"
        return (
            pred,
            [cid],
            [{
                "facet": "label_live",
                "addr": addr,
                "name": name_s,
                "kind": pid.get("kind"),
                "confidence": pid.get("confidence"),
                "source_event_id": ek,
                "kb_event_id": ek,
            }],
            [blob],
        )

    if kind == EVT_ROUTINE:
        st = int(pid["start"])
        pred = lambda m, st_=st: (
            m.get("facet") in {"routine_live", "routine_table"}
            and int(m.get("start") or -1) == st_
        )
        blob = clip_for_embedding(
            _format_routine_payload(pid, ek),
            TEXT_CLIP,
        )
        return (
            pred,
            [f"rt_evt:{ek}"],
            [{
                "facet": "routine_live",
                "start": st,
                "end": int(pid.get("end") or pid["start"]),
                "name": pid.get("name"),
                "source_event_id": ek,
                "kb_event_id": ek,
            }],
            [blob],
        )

    if kind == EVT_DATA_STRUCTURE:
        st = int(pid["start"])
        pred = lambda m, st_=st: (
            m.get("facet") in {"data_structure_live", "data_structure_table"}
            and int(m.get("start") or -1) == st_
        )
        blob = clip_for_embedding(
            _format_ds_payload(pid, ek),
            TEXT_CLIP,
        )
        return (
            pred,
            [f"ds_evt:{ek}"],
            [{
                "facet": "data_structure_live",
                "start": st,
                "end": int(pid.get("end") or pid["start"]),
                "kind": pid.get("kind"),
                "kb_event_id": ek,
            }],
            [blob],
        )

    if kind == EVT_HYPOTHESIS:
        hid = str(pid.get("id") or ek)
        pred = lambda m, h_=hid: (
            m.get("facet") in {"hypothesis_live", "hypothesis_table"}
            and str(m.get("hypothesis_id") or "") == h_
        )
        blob = clip_for_embedding(
            _format_hypothesis_payload(pid, ek),
            TEXT_CLIP,
        )
        return (
            pred,
            [f"hyp_evt:{ek}"],
            [{
                "facet": "hypothesis_live",
                "hypothesis_id": hid,
                "status": pid.get("status"),
                "kb_event_id": ek,
            }],
            [blob],
        )

    if kind == EVT_CONSOLIDATED:
        blob = clip_for_embedding(
            _format_consolidated_payload(pid),
            TEXT_CLIP,
        )
        meta = {"facet": "consolidated_event", "event_id": ek, "kb_event_id": ek}
        pred = lambda m, e_=ek: (
            m.get("facet") == "consolidated_event"
            and str(m.get("event_id")) == e_
        )
        return (pred, [f"cn_evt:{ek}"], [meta], [blob])

    if kind == EVT_TOOL_RESULT:
        blob = clip_for_embedding(
            _format_tool_result_payload_live(pid),
            min(4096, TEXT_CLIP),
        )
        pred = lambda m, e_=ek: (
            m.get("facet") == "tool_result_event"
            and str(m.get("event_id")) == e_
        )
        meta = {
            "facet": "tool_result_event",
            "event_id": ek,
            "kb_event_id": ek,
            "tool": pid.get("tool"),
            "step_id": pid.get("step_id"),
            "ok": pid.get("ok"),
        }
        return (pred, [f"tr_evt:{ek}"], [meta], [blob])

    return (lambda _m: False, [], [], [])


def snippet_for_semantic(meta: dict[str, Any], text_preview: str) -> str:
    """Compact line for synthesizer/report output."""
    facet = meta.get("facet") or "?"
    if meta.get("path"):
        return f"[{facet}] {meta['path']} :: {text_preview[:180]}"
    if meta.get("addr") is not None and meta.get("name"):
        return f"[{facet}] ${int(meta['addr']):04X} {meta['name']} :: {text_preview[:180]}"
    if meta.get("start") is not None:
        return f"[{facet}] ${int(meta['start']):04X} :: {text_preview[:180]}"
    ev = meta.get("event_id")
    return f"[{facet}] evt={ev} :: {text_preview[:180]}"


# ---- serialization helpers ---------------------------------------------- #


def _format_label_row(row: dict[str, Any]) -> str:
    return (
        f"KB label: ${int(row['addr']):04X} name={row.get('name')!s} "
        f"kind={row.get('kind')!s} confidence={row.get('confidence')}"
    )


def _format_label_payload(p: dict[str, Any], ev_id: str) -> str:
    return (
        f"KB label (event {ev_id}): ${int(p['addr']):04X} "
        f"name={p.get('name')!s} kind={p.get('kind')!s} "
        f"confidence={p.get('confidence')} "
        f"evidence={p.get('evidence')!s}"
    )


def _format_routine_row(row: dict[str, Any]) -> str:
    ct = row.get("calls_to_json") or "[]"
    cb = row.get("called_by_json") or "[]"
    return (
        f"KB routine: ${int(row['start']):04X}-${int(row['end']):04X} "
        f"name={row.get('name')!s}\nsummary={row.get('summary')!s}"
        f"\ncalls_to={ct}\ncalled_by={cb}"
    )


def _format_routine_payload(p: dict[str, Any], ev_id: str) -> str:
    return (
        f"KB routine (event {ev_id}): ${int(p['start']):04X}-"
        f"${int(p.get('end', p['start'])):04X} name={p.get('name')!s}\n"
        f"summary={p.get('summary')!s}\n"
        f"calls_to={json.dumps(p.get('calls_to') or [])}"
        f"\ncalled_by={json.dumps(p.get('called_by') or [])}"
    )


def _format_ds_row(row: dict[str, Any]) -> str:
    return (
        f"KB data_structure: ${int(row['start']):04X}-"
        f"${int(row['end']):04X} kind={row.get('kind')!s}\n"
        f"fields={row.get('fields_json')}"
    )


def _format_ds_payload(p: dict[str, Any], ev_id: str) -> str:
    return (
        f"KB data_structure (event {ev_id}): "
        f"${int(p['start']):04X}-${int(p.get('end', p['start'])):04X} "
        f"kind={p.get('kind')!s}\n"
        f"fields={json.dumps(p.get('fields') or {})}"
    )


def _format_hypothesis_row(row: dict[str, Any]) -> str:
    return (
        f"KB hypothesis [{row['id']}] ({row.get('status')}): "
        f"{row.get('text')}\n(evidence_json={row.get('evidence_json')})"
    )


def _format_hypothesis_payload(p: dict[str, Any], ev_id: str) -> str:
    hid = str(p.get("id") or ev_id)
    return (
        f"KB hypothesis [{hid}] ({p.get('status')}): "
        f"{p.get('text')}\nevidence={p.get('evidence')}"
    )


def _format_consolidated_payload(p: dict[str, Any]) -> str:
    return (
        f"KB consolidated_observation summary:\n{p.get('summary') or ''}"
        f"\naddresses_kept={json.dumps(p.get('addresses_kept') or [])}"
    )


def _format_tool_result_payload(p: dict[str, Any]) -> str:
    return (
        f"KB tool_result {p.get('tool')}/{p.get('step_id')} "
        f"ok={p.get('ok')}\ndata:\n{str(p.get('data', ''))}"
    )


def _format_tool_result_payload_live(p: dict[str, Any]) -> str:
    return _format_tool_result_payload(p)


def semantic_hit_preview_text(
    meta: dict[str, Any],
    query_fn: Any,
    *,
    chunk_chars: int,
    chunk_overlap: int,
) -> str:
    """Reconstruct display text aligned with deterministic chunk boundaries."""

    facet = meta.get("facet")
    try:
        if facet == "text_doc" and meta.get("path"):
            hits = query_fn(
                "SELECT content FROM text_docs WHERE path = ? LIMIT 1",
                (meta["path"],),
            )
            if not hits:
                return ""
            body = hits[0].get("content") or ""

            ci = int(meta.get("chunk_index") or 0)
            chunks = chunk_text(
                clip_for_embedding(body, TEXT_CLIP),
                chunk_chars,
                chunk_overlap,
            )
            return chunks[ci] if 0 <= ci < len(chunks) else ""

        if facet == "label_table":
            hits = query_fn(
                "SELECT addr, name, kind, confidence FROM labels "
                "WHERE addr = ? AND name = ? LIMIT 1",
                (int(meta["addr"]), str(meta["name"])),
            )
            return _format_label_row(hits[0]) if hits else ""

        ev_id = meta.get("kb_event_id") or meta.get("source_event_id")

        def _evt_payload(ev: str | None) -> dict[str, Any]:
            if not ev:
                return {}
            rows = query_fn(
                "SELECT payload_json FROM events WHERE id = ? LIMIT 1",
                (str(ev),),
            )
            if not rows:
                return {}
            try:
                return json.loads(rows[0].get("payload_json") or "{}")
            except json.JSONDecodeError:
                return {}

        if facet == "label_live" and ev_id:
            pay = _evt_payload(str(ev_id))
            return (
                clip_for_embedding(_format_label_payload(pay, str(ev_id)), TEXT_CLIP)
                if pay
                else ""
            )

        if facet == "routine_table":
            hits = query_fn(
                "SELECT start, end, name, summary, calls_to_json, "
                "called_by_json FROM routines WHERE start = ? LIMIT 1",
                (int(meta["start"]),),
            )
            return _format_routine_row(hits[0]) if hits else ""

        if facet == "routine_live" and ev_id:
            pay = _evt_payload(str(ev_id))
            return clip_for_embedding(
                _format_routine_payload(pay, str(ev_id)), TEXT_CLIP,
            )

        if facet == "data_structure_table":
            hits = query_fn(
                "SELECT start, end, kind, fields_json FROM data_structures "
                "WHERE start = ? LIMIT 1",
                (int(meta["start"]),),
            )
            return _format_ds_row(hits[0]) if hits else ""

        if facet == "data_structure_live" and ev_id:
            pay = _evt_payload(str(ev_id))
            return clip_for_embedding(
                _format_ds_payload(pay, str(ev_id)), TEXT_CLIP,
            )

        if facet == "hypothesis_table":
            hits = query_fn(
                "SELECT id, text, status, evidence_json FROM hypotheses "
                "WHERE id = ? LIMIT 1",
                (str(meta["hypothesis_id"]),),
            )
            return _format_hypothesis_row(hits[0]) if hits else ""

        if facet == "hypothesis_live" and ev_id:
            pay = _evt_payload(str(ev_id))
            return clip_for_embedding(
                _format_hypothesis_payload(pay, str(ev_id)), TEXT_CLIP,
            )

        if facet == "consolidated_event":
            evid = meta.get("event_id")
            rows = query_fn(
                "SELECT payload_json FROM events WHERE id = ? LIMIT 1",
                (str(evid),),
            )
            if not rows:
                return ""
            try:
                p = json.loads(rows[0]["payload_json"] or "{}")
            except json.JSONDecodeError:
                p = {}
            return clip_for_embedding(
                _format_consolidated_payload(p), TEXT_CLIP,
            )

        if facet == "tool_result_event":
            evid = meta.get("event_id")
            rows = query_fn(
                "SELECT payload_json FROM events WHERE id = ? LIMIT 1",
                (str(evid),),
            )
            if not rows:
                return ""
            try:
                p = json.loads(rows[0]["payload_json"] or "{}")
            except json.JSONDecodeError:
                p = {}
            return clip_for_embedding(_format_tool_result_payload(p), TEXT_CLIP)

    except (KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError):
        pass
    return ""