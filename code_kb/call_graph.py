"""Build call-graph visualisations from the code knowledge base.

The output is graphviz DOT source, which Streamlit (and any DOT viewer)
can render directly without needing a system graphviz binary.

Two main shapes:

    * `local_dot(start, hops=...)` — center a routine, walk N hops in
      both directions through the xref graph, and emit a tight DOT
      snippet suitable for embedding next to a chat answer.

    * `routines_dot(routine_starts)` — emit a graph spanning a
      user-selected set of routines and the edges between them only.

Edge styling follows the xref kind: `jsr` is a solid arrow (logical
"call"); `jmp` is dashed (control transfer without return); `branch`
is dotted (intra-routine — usually filtered); `jmp_indirect` is bold
red (runtime / table-dispatch — worth flagging visually).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

from code_kb.store import CodeKnowledgeStore


_EDGE_STYLE = {
    "jsr":          'color="#2563eb", arrowhead="normal"',
    "jmp":          'color="#9333ea", style="dashed", arrowhead="normal"',
    "branch":       'color="#6b7280", style="dotted", arrowhead="open"',
    "jmp_indirect": 'color="#dc2626", style="bold", arrowhead="diamond"',
    "fallthrough":  'color="#94a3b8", style="dashed", arrowhead="none"',
}


@dataclass(frozen=True)
class _RoutineLite:
    start: int
    end: int
    name: str | None
    source_file: str | None


@dataclass(frozen=True)
class _RoutineSem:
    text: str | None
    name_suggestion: str | None
    idiom: str | None
    confidence: float | None
    hardware_touched: list[str]


def _best_layer1_semantics(
    store: CodeKnowledgeStore, starts: Iterable[int],
) -> dict[int, _RoutineSem]:
    """Return the highest-confidence Layer-1 hypothesis per routine start."""
    starts = sorted(set(int(s) & 0xFFFF for s in starts if int(s) >= 0))
    if not starts:
        return {}

    out: dict[int, _RoutineSem] = {}
    for s in starts:
        rows = store.query(
            "SELECT text, name_suggestion, idiom_match, hardware_touched_json, confidence"
            "  FROM hypotheses"
            " WHERE layer = 1 AND start_addr = ?"
            " ORDER BY confidence DESC LIMIT 1",
            (s,),
        )
        if not rows:
            continue
        r = rows[0]
        try:
            conf = float(r.get("confidence"))
        except (TypeError, ValueError):
            conf = None
        hw_raw = r.get("hardware_touched_json")
        try:
            hw = json.loads(hw_raw) if isinstance(hw_raw, str) else []
            if not isinstance(hw, list):
                hw = []
        except json.JSONDecodeError:
            hw = []
        out[s] = _RoutineSem(
            text=(r.get("text") or None),
            name_suggestion=(r.get("name_suggestion") or None),
            idiom=(r.get("idiom_match") or None),
            confidence=conf,
            hardware_touched=[str(x) for x in hw[:8]],
        )
    return out


def _routine_for_addr(
    store: CodeKnowledgeStore, addr: int,
) -> _RoutineLite | None:
    """Return the routine row enclosing `addr`, or None."""
    rows = store.query(
        "SELECT start_addr, end_addr, name, source_file"
        "  FROM code_routines"
        " WHERE ? BETWEEN start_addr AND end_addr"
        " ORDER BY (end_addr - start_addr) ASC LIMIT 1",
        (int(addr) & 0xFFFF,),
    )
    if not rows:
        return None
    r = rows[0]
    return _RoutineLite(
        start=int(r["start_addr"]),
        end=int(r["end_addr"]),
        name=r.get("name"),
        source_file=r.get("source_file"),
    )


def _node_label(r: _RoutineLite, sem: _RoutineSem | None = None) -> str:
    title = r.name or f"sub_{r.start:04x}"
    if (
        sem
        and sem.name_suggestion
        and sem.confidence is not None
        and sem.confidence >= 0.60
        and (not r.name or str(r.name).lower().startswith("sub_"))
    ):
        title = sem.name_suggestion
    lines = [f"{title}", f"${r.start:04X}-${r.end:04X}"]
    if sem:
        if sem.idiom:
            lines.append(f"[{sem.idiom}]")
        if sem.confidence is not None:
            lines.append(f"conf={sem.confidence:.2f}")
        if sem.text:
            summary = sem.text.strip().replace('"', "'")
            if len(summary) > 78:
                summary = summary[:75].rstrip() + "..."
            lines.append(summary)
    return "\\n".join(lines)


def _node_style(sem: _RoutineSem | None) -> str:
    if sem is None:
        return 'fillcolor="#f8fafc", color="#94a3b8"'
    if sem.confidence is not None and sem.confidence >= 0.75:
        return 'fillcolor="#dcfce7", color="#166534"'
    if sem.confidence is not None and sem.confidence < 0.5:
        return 'fillcolor="#fef3c7", color="#92400e"'
    return 'fillcolor="#e0f2fe", color="#075985"'


def _edge_label(kind: str, dst_sem: _RoutineSem | None) -> str:
    if kind != "jsr" or not dst_sem or not dst_sem.idiom:
        return kind
    return f"{kind}:{dst_sem.idiom}"


def _node_id(start: int) -> str:
    return f"r_{start:04X}"


def local_dot(
    store: CodeKnowledgeStore,
    *,
    start: int,
    hops: int = 1,
    include_branches: bool = False,
    include_indirect: bool = True,
    max_nodes: int = 60,
) -> dict:
    """Center a routine and walk N hops of the xref graph.

    Returns ``{"dot": "<digraph...>", "nodes": [...], "edges": [...]}``.
    The center routine is rendered with a heavier outline and the
    nodes the user "came from" (callers) sit above; callees below.
    """
    center = _routine_for_addr(store, start)
    if center is None:
        return {
            "dot": (
                'digraph G {\n'
                '  rankdir=LR;\n'
                f'  empty [label="no routine found enclosing ${int(start) & 0xFFFF:04X}",'
                ' shape=note, color="#dc2626"];\n'
                '}\n'
            ),
            "nodes": [], "edges": [], "center": None,
        }

    visited: dict[int, _RoutineLite] = {center.start: center}
    stubs: set[int] = set()  # addresses with no code_routines entry yet
    edges: list[tuple[int, int, str, int | None]] = []
    queue: list[tuple[int, int]] = [(center.start, 0)]

    while queue:
        cur_start, depth = queue.pop(0)
        if depth >= hops:
            continue
        if cur_start in stubs:
            # Stub nodes have no known address range — can't query xrefs.
            continue
        cur = visited[cur_start]

        # Outgoing: callees from anywhere in the routine range.
        out_rows = store.query(
            "SELECT src_addr, dst_addr, kind, via_vector"
            "  FROM code_xrefs"
            " WHERE src_addr BETWEEN ? AND ?"
            "   AND kind IN ('jsr', 'jmp', 'jmp_indirect')",
            (cur.start, cur.end),
        )
        for x in out_rows:
            kind = str(x["kind"])
            if not include_indirect and kind == "jmp_indirect":
                continue
            dst = x.get("dst_addr")
            if dst is None:
                # Indirect site without resolved target — render as
                # vector node so the planner can still see something.
                vec = x.get("via_vector")
                if vec is None:
                    continue
                edges.append((cur.start, -int(vec), kind, int(vec)))
                if len(visited) + 1 > max_nodes:
                    continue
                visited.setdefault(-int(vec), _RoutineLite(
                    start=-int(vec), end=-int(vec),
                    name=f"via (${int(vec):04X})",
                    source_file=None,
                ))
                continue
            target = _routine_for_addr(store, int(dst))
            if target is None:
                # No routine boundary recorded yet — show as a stub node
                # so the call-graph edge is still visible.
                stub_addr = int(dst) & 0xFFFF
                edges.append((cur.start, stub_addr, kind, None))
                if stub_addr not in visited and len(visited) < max_nodes:
                    visited[stub_addr] = _RoutineLite(
                        start=stub_addr, end=stub_addr,
                        name=f"sub_{stub_addr:04x}",
                        source_file=None,
                    )
                    stubs.add(stub_addr)
                    # Don't enqueue stubs — we can't walk their xrefs.
                continue
            edges.append((cur.start, target.start, kind, None))
            if target.start not in visited:
                if len(visited) >= max_nodes:
                    continue
                visited[target.start] = target
                queue.append((target.start, depth + 1))

        # Incoming: callers landing inside this routine range.
        in_rows = store.query(
            "SELECT src_addr, dst_addr, kind FROM code_xrefs"
            " WHERE dst_addr BETWEEN ? AND ?"
            "   AND kind IN ('jsr', 'jmp', 'jmp_indirect')",
            (cur.start, cur.end),
        )
        for x in in_rows:
            kind = str(x["kind"])
            src = int(x["src_addr"])
            caller = _routine_for_addr(store, src)
            if caller is None:
                # Caller not in code_routines — stub it.
                stub_addr = src & 0xFFFF
                edges.append((stub_addr, cur.start, kind, None))
                if stub_addr not in visited and len(visited) < max_nodes:
                    visited[stub_addr] = _RoutineLite(
                        start=stub_addr, end=stub_addr,
                        name=f"sub_{stub_addr:04x}",
                        source_file=None,
                    )
                    stubs.add(stub_addr)
                continue
            edges.append((caller.start, cur.start, kind, None))
            if caller.start not in visited:
                if len(visited) >= max_nodes:
                    continue
                visited[caller.start] = caller
                queue.append((caller.start, depth + 1))

    sem_by_start = _best_layer1_semantics(store, [k for k in visited if k >= 0])

    # Render DOT.
    dot_lines: list[str] = [
        "digraph G {",
        '  rankdir=LR;',
        '  node [shape=box, fontname="Helvetica", fontsize=10, '
        'style="rounded,filled", fillcolor="#f8fafc", color="#94a3b8"];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]
    for r in visited.values():
        if r.start == center.start:
            dot_lines.append(
                f'  {_node_id(r.start)} [label="{_node_label(r, sem_by_start.get(r.start))}", '
                'fillcolor="#fde68a", color="#92400e", penwidth=2];'
            )
        elif r.start < 0:  # synthetic indirect-vector node
            via = -r.start
            dot_lines.append(
                f'  v_{via:04X} [label="indirect via\\n${via:04X}", '
                'shape=hexagon, fillcolor="#fee2e2", color="#dc2626"];'
            )
        elif r.start in stubs:  # unanalyzed callee — no routine boundary yet
            dot_lines.append(
                f'  {_node_id(r.start)} [label="{_node_label(r)}", '
                'fillcolor="#f1f5f9", color="#94a3b8", style="rounded,filled,dashed"];'
            )
        else:
            dot_lines.append(
                f'  {_node_id(r.start)} [label="{_node_label(r, sem_by_start.get(r.start))}", '
                f'{_node_style(sem_by_start.get(r.start))}];'
            )

    seen_edges: set[tuple[int, int, str]] = set()
    edges_out: list[dict] = []
    for src, dst, kind, via in edges:
        if (src, dst, kind) in seen_edges:
            continue
        seen_edges.add((src, dst, kind))
        style = _EDGE_STYLE.get(kind, "color=\"#475569\"")
        if dst < 0:
            tgt = f"v_{(-dst):04X}"
            dst_sem = None
        else:
            tgt = _node_id(dst)
            dst_sem = sem_by_start.get(dst)
        dot_lines.append(
            f'  {_node_id(src)} -> {tgt} [label="{_edge_label(kind, dst_sem)}", {style}];'
        )
        edges_out.append({
            "src": src, "dst": dst if dst >= 0 else None,
            "kind": kind, "via_vector": via,
            "dst_idiom": dst_sem.idiom if dst_sem else None,
            "dst_confidence": dst_sem.confidence if dst_sem else None,
        })

    if not include_branches:
        # Branches were not added above; nothing to do. Kept as a hook
        # for callers that want to switch on intra-routine branches.
        pass

    dot_lines.append("}")
    return {
        "dot": "\n".join(dot_lines),
        "nodes": [
            {
                "start": r.start, "end": r.end,
                "name": r.name, "source_file": r.source_file,
                "layer1": (
                    {
                        "text": sem_by_start[r.start].text,
                        "name_suggestion": sem_by_start[r.start].name_suggestion,
                        "idiom_match": sem_by_start[r.start].idiom,
                        "confidence": sem_by_start[r.start].confidence,
                        "hardware_touched": sem_by_start[r.start].hardware_touched,
                    }
                    if r.start in sem_by_start else None
                ),
            }
            for r in visited.values() if r.start >= 0
        ],
        "edges": edges_out,
        "center": {
            "start": center.start, "end": center.end, "name": center.name,
        },
    }


def routines_dot(
    store: CodeKnowledgeStore, *, routine_starts: Iterable[int],
) -> dict:
    """Render a graph including only the listed routines + edges between them."""
    starts = sorted(set(int(s) & 0xFFFF for s in routine_starts))
    routines: list[_RoutineLite] = []
    for s in starts:
        r = _routine_for_addr(store, s)
        if r and r.start == s:
            routines.append(r)
    if not routines:
        return {
            "dot": (
                'digraph G { empty [label="no routines selected", '
                'shape=note, color="#dc2626"]; }'
            ),
            "nodes": [], "edges": [],
        }

    by_start = {r.start: r for r in routines}
    sem_by_start = _best_layer1_semantics(store, by_start.keys())

    dot_lines: list[str] = [
        "digraph G {",
        '  rankdir=LR;',
        '  node [shape=box, fontname="Helvetica", fontsize=10, '
        'style="rounded,filled", fillcolor="#f8fafc", color="#94a3b8"];',
    ]
    for r in routines:
        dot_lines.append(
            f'  {_node_id(r.start)} [label="{_node_label(r, sem_by_start.get(r.start))}", '
            f'{_node_style(sem_by_start.get(r.start))}];'
        )

    # All xrefs whose src is inside one selected routine and dst inside another.
    rows = store.query(
        "SELECT src_addr, dst_addr, kind FROM code_xrefs"
        " WHERE dst_addr IS NOT NULL"
        "   AND kind IN ('jsr', 'jmp', 'jmp_indirect')",
        (),
    )
    edges_out: list[dict] = []
    seen_edges: set[tuple[int, int, str]] = set()
    for x in rows:
        src = int(x["src_addr"])
        dst = int(x["dst_addr"])
        # find enclosing routines among the selected set
        src_r = next(
            (rr for rr in routines if rr.start <= src <= rr.end), None,
        )
        dst_r = by_start.get(dst) or next(
            (rr for rr in routines if rr.start <= dst <= rr.end), None,
        )
        if src_r is None or dst_r is None or src_r.start == dst_r.start:
            continue
        kind = str(x["kind"])
        if (src_r.start, dst_r.start, kind) in seen_edges:
            continue
        seen_edges.add((src_r.start, dst_r.start, kind))
        style = _EDGE_STYLE.get(kind, "")
        dst_sem = sem_by_start.get(dst_r.start)
        dot_lines.append(
            f'  {_node_id(src_r.start)} -> {_node_id(dst_r.start)} '
            f'[label="{_edge_label(kind, dst_sem)}", {style}];'
        )
        edges_out.append({
            "src": src_r.start, "dst": dst_r.start, "kind": kind,
            "dst_idiom": dst_sem.idiom if dst_sem else None,
            "dst_confidence": dst_sem.confidence if dst_sem else None,
        })
    dot_lines.append("}")
    return {
        "dot": "\n".join(dot_lines),
        "nodes": [
            {
                "start": r.start,
                "end": r.end,
                "name": r.name,
                "layer1": (
                    {
                        "text": sem_by_start[r.start].text,
                        "name_suggestion": sem_by_start[r.start].name_suggestion,
                        "idiom_match": sem_by_start[r.start].idiom,
                        "confidence": sem_by_start[r.start].confidence,
                        "hardware_touched": sem_by_start[r.start].hardware_touched,
                    }
                    if r.start in sem_by_start else None
                ),
            }
            for r in routines
        ],
        "edges": edges_out,
    }
