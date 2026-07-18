"""code_kb data-reference index end-to-end (tracker 3.4)."""

import json
import tempfile
from pathlib import Path

import pytest

from code_kb.asm_parser import parse_path
from code_kb.layer0 import build_from_parsed_asm
from code_kb.schema import ANN_DATAREF
from code_kb.store import Annotation, CodeKnowledgeStore


_ASM = """\
C000  A9 05     LDA #$05
C002  8D 20 D0  STA $D020
C005  AD 12 D0  LDA $D012
C008  CE 80 07  DEC $0780
C00B  8D 00 D4  STA $D400
C00E  8D 01 D4  STA $D401
"""


@pytest.fixture()
def store_with_asm(tmp_path):
    asm = tmp_path / "game.asm"
    asm.write_text(_ASM)
    store = CodeKnowledgeStore.load_or_init(tmp_path / "code_kb")
    build_from_parsed_asm(parse_path(asm), store)
    return store, tmp_path


def test_data_refs_projected_with_access(store_with_asm):
    store, _ = store_with_asm
    rows = store.query(
        "SELECT src_addr, dst_addr, access FROM code_data_refs"
        " ORDER BY src_addr",
    )
    got = {(r["src_addr"], r["dst_addr"], r["access"]) for r in rows}
    assert (0xC002, 0xD020, "w") in got
    assert (0xC005, 0xD012, "r") in got
    assert (0xC008, 0x0780, "rmw") in got
    # LDA #$05 is immediate — no data-ref row.
    assert not any(r["src_addr"] == 0xC000 for r in rows)


def test_writes_to_query(store_with_asm):
    store, _ = store_with_asm
    w = store.query(
        "SELECT src_addr FROM code_data_refs"
        " WHERE dst_addr = 0xD020 AND access IN ('w','rmw')",
    )
    assert [r["src_addr"] for r in w] == [0xC002]
    # $D012 is read-only here → not a writer.
    w2 = store.query(
        "SELECT src_addr FROM code_data_refs"
        " WHERE dst_addr = 0xD012 AND access IN ('w','rmw')",
    )
    assert w2 == []


def test_hardware_refs_sid_range(store_with_asm):
    store, _ = store_with_asm
    sid = store.query(
        "SELECT DISTINCT dst_addr FROM code_data_refs"
        " WHERE dst_addr BETWEEN 0xD400 AND 0xD7FF ORDER BY dst_addr",
    )
    assert [r["dst_addr"] for r in sid] == [0xD400, 0xD401]


def test_stats_counts_data_refs(store_with_asm):
    store, _ = store_with_asm
    assert store.stats()["data_refs"] >= 5


# ---------------------------------------------------------------------------
# Provenance: data refs invalidate per source and survive replay (2.2/3.4)
# ---------------------------------------------------------------------------

def _dataref_ann(src, dst, source_file, access="r"):
    return Annotation(
        layer=0, kind=ANN_DATAREF, start_addr=src, end_addr=src,
        producer="deterministic", confidence=1.0,
        payload={"src_addr": src, "dst_addr": dst, "access": access,
                 "index": None, "indirect": False, "source_file": source_file},
    )


# ---------------------------------------------------------------------------
# code_kb_node integration (dispatched through the node, not raw SQL)
# ---------------------------------------------------------------------------

def _run_mode(handle, mode, extra=None):
    import graph.code_kb_node as ckn
    args = {"mode": mode, **(extra or {})}
    state = {
        "code_kb_handle": handle,
        "plan": [{"id": "s1", "tool": "code_kb", "args": args}],
        "current_step_id": "s1",
    }
    out = ckn.code_kb_node(state)
    return out["tool_results"][0]


def test_node_writes_to(store_with_asm):
    store, tmp = store_with_asm
    handle = str(tmp / "code_kb")
    r = _run_mode(handle, "writes_to", {"addr": "$D020"})
    assert r["ok"] and r["row_count"] == 1
    assert any(row["src_addr"] == 0xC002 for row in json.loads(r["data"]))


def test_node_refs_to(store_with_asm):
    store, tmp = store_with_asm
    handle = str(tmp / "code_kb")
    r = _run_mode(handle, "refs_to", {"addr": "$0780"})
    assert r["ok"]
    accesses = {row["access"] for row in json.loads(r["data"])}
    assert "rmw" in accesses            # DEC $0780


def test_node_hardware_refs_sid(store_with_asm):
    store, tmp = store_with_asm
    handle = str(tmp / "code_kb")
    r = _run_mode(handle, "hardware_refs", {"chip": "sid"})
    assert r["ok"]
    dsts = {row["dst_addr"] for row in json.loads(r["data"])}
    assert dsts == {0xD400, 0xD401}
    # VIC write ($D020) must NOT appear in the SID range result.
    assert 0xD020 not in dsts


def test_node_writes_to_deduplicates_multi_source(tmp_path):
    """A write asserted by two sources shows once (SELECT DISTINCT)."""
    import graph.code_kb_node as ckn
    store = CodeKnowledgeStore.load_or_init(tmp_path / "code_kb")
    for src in ("a.asm", "b.asm"):
        store.append_annotation(
            Annotation(
                layer=0, kind=ANN_DATAREF, start_addr=0xC000, end_addr=0xC000,
                producer="deterministic", confidence=1.0,
                payload={"src_addr": 0xC000, "dst_addr": 0xD020, "access": "w",
                         "index": None, "indirect": False, "source_file": src},
            ),
            source="t",
        )
    state = {
        "code_kb_handle": str(tmp_path / "code_kb"),
        "plan": [{"id": "s1", "tool": "code_kb",
                  "args": {"mode": "writes_to", "addr": "$D020"}}],
        "current_step_id": "s1",
    }
    r = ckn.code_kb_node(state)["tool_results"][0]
    assert r["ok"] and r["row_count"] == 1     # deduped across sources


def test_data_refs_invalidate_per_source_and_survive_replay(tmp_path):
    store = CodeKnowledgeStore.load_or_init(tmp_path / "code_kb")
    store.append_annotation(_dataref_ann(0xC000, 0xD020, "old.asm"), source="t")
    store.append_annotation(_dataref_ann(0xC010, 0xD021, "keep.asm"), source="t")
    # Same edge from two sources → two provenance rows.
    store.append_annotation(_dataref_ann(0xC020, 0xD030, "old.asm"), source="t")
    store.append_annotation(_dataref_ann(0xC020, 0xD030, "keep.asm"), source="t")

    store.invalidate_source("old.asm")

    def _check(s):
        rows = s.query(
            "SELECT src_addr, source_file FROM code_data_refs"
            " ORDER BY src_addr",
        )
        got = {(r["src_addr"], r["source_file"]) for r in rows}
        assert got == {(0xC010, "keep.asm"), (0xC020, "keep.asm")}

    _check(store)                                             # live
    _check(CodeKnowledgeStore.load_or_init(tmp_path / "code_kb"))  # replay
