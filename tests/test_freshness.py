"""Content-freshness of dumps and asm files (tracker 2.2) and COMPLETE
source invalidation (Phase 2 hardening: stale xrefs/SMC/classification/
annotations rows were left behind and consumed by call-graph and
Layer-1 queries)."""

import json

import pytest

from code_kb.schema import (
    ANN_CLASSIFY,
    ANN_LABEL,
    ANN_ROUTINE,
    ANN_SMC,
    ANN_XREF,
)
from code_kb.store import Annotation, CodeKnowledgeStore
from memory.schema import EVT_INGEST_DUMP
from memory.store import KnowledgeStore


# ---------------------------------------------------------------------------
# General KB: dump content changes are detected and flagged
# ---------------------------------------------------------------------------

@pytest.fixture()
def kb(tmp_path):
    return KnowledgeStore.load_or_init(tmp_path / "kb"), tmp_path


def test_dump_reingest_only_when_content_changes(kb):
    store, tmp = kb
    dump = tmp / "game.bin"
    dump.write_bytes(b"\x01" * 64)

    assert store.ingest_dump(dump) is not None      # first ingest
    assert store.ingest_dump(dump) is None          # same content → skip

    dump.write_bytes(b"\x02" * 64)                  # overwritten in place
    evt_id = store.ingest_dump(dump)
    assert evt_id is not None                       # change detected

    rows = store.query(
        "SELECT payload_json FROM events WHERE kind = ? ORDER BY ts",
        (EVT_INGEST_DUMP,),
    )
    assert len(rows) == 2
    latest = json.loads(rows[-1]["payload_json"])
    assert latest["refreshed_from"]                 # provenance of the change
    assert store.read_bytes(0, 4) == b"\x02" * 4    # cache refreshed


def test_digest_flags_refreshed_dump(kb):
    store, tmp = kb
    dump = tmp / "game.bin"
    dump.write_bytes(b"\x01" * 64)
    store.ingest_dump(dump)
    assert "dump_refreshed" not in store.digest_for_question("q")

    dump.write_bytes(b"\x02" * 64)
    store.ingest_dump(dump)
    digest = store.digest_for_question("q")
    assert "dump_refreshed: True" in digest


# ---------------------------------------------------------------------------
# Code KB: asm freshness + event-based invalidation
# ---------------------------------------------------------------------------

@pytest.fixture()
def code_store(tmp_path):
    return CodeKnowledgeStore.load_or_init(tmp_path / "code_kb"), tmp_path


def _routine_ann(start, source_file):
    return Annotation(
        layer=0, kind=ANN_ROUTINE, start_addr=start, end_addr=start + 0x20,
        producer="layer0", confidence=0.9,
        payload={"name": f"sub_{start:04x}", "summary": "s",
                 "source_file": source_file, "entries": [start], "exits": [],
                 "size_bytes": 0x21},
    )


def test_asm_freshness_states(code_store):
    store, tmp = code_store
    asm = tmp / "game.asm"
    asm.write_text("$C000 LDA #$01\n")
    content = asm.read_text()

    assert store.asm_freshness(asm, content) == "new"
    store.ingest_asm_doc(asm, content=content, instructions=1, routines=0)
    assert store.asm_freshness(asm, content) == "current"

    new_content = "$C000 LDA #$02\n"
    assert store.asm_freshness(asm, new_content) == "changed"


def test_invalidate_source_survives_full_replay(code_store):
    """Invalidation is an EVENT: a from-scratch rebuild replays the
    deletion before the re-ingest, reproducing the same state."""
    store, tmp = code_store
    store.append_annotation(_routine_ann(0xC000, "old.asm"), source="t")
    store.append_annotation(_routine_ann(0xD000, "keep.asm"), source="t")
    assert len(store.query("SELECT * FROM code_routines")) == 2

    store.invalidate_source("old.asm")
    starts = {r["start_addr"]
              for r in store.query("SELECT start_addr FROM code_routines")}
    assert starts == {0xD000}

    # Fresh instance → full event replay → same result.
    reloaded = CodeKnowledgeStore.load_or_init(tmp / "code_kb")
    starts = {r["start_addr"]
              for r in reloaded.query("SELECT start_addr FROM code_routines")}
    assert starts == {0xD000}


def test_invalidate_source_like_pattern(code_store):
    """Dump changes drop `capstone:%`/`vice:%` disassembly-window rows."""
    store, _tmp = code_store
    store.append_annotation(Annotation(
        layer=0, kind=ANN_LABEL, start_addr=0xC000, end_addr=0xC000,
        producer="layer0", confidence=0.9,
        payload={"name": "from_disasm", "source_file": "capstone:$C000+256"},
    ), source="t")
    store.append_annotation(Annotation(
        layer=0, kind=ANN_LABEL, start_addr=0xD000, end_addr=0xD000,
        producer="layer0", confidence=0.9,
        payload={"name": "from_asm", "source_file": "game.asm"},
    ), source="t")

    store.invalidate_source("capstone:%", like=True)
    names = {r["name"] for r in store.query("SELECT name FROM code_labels")}
    assert names == {"from_asm"}


def _xref_ann(src, dst, source_file):
    return Annotation(
        layer=0, kind=ANN_XREF, start_addr=src, end_addr=src,
        producer="layer0", confidence=1.0,
        payload={"src_addr": src, "dst_addr": dst, "xref_kind": "jsr",
                 "via_vector": None, "source_file": source_file},
    )


def _smc_ann(src, dst, source_file):
    return Annotation(
        layer=0, kind=ANN_SMC, start_addr=src, end_addr=src,
        producer="layer0", confidence=0.7,
        payload={"src_addr": src, "dst_addr": dst, "mnemonic": "sta",
                 "operand": f"${dst:04X}", "smc_kind": "absolute_into_code",
                 "source_file": source_file},
    )


def _classify_ann(start, end, source_file):
    return Annotation(
        layer=0, kind=ANN_CLASSIFY, start_addr=start, end_addr=end,
        producer="layer0", confidence=1.0,
        payload={"classification": "code", "source_file": source_file},
    )


def _seed_all_projections(store):
    """Two sources contributing every projection kind; `old.asm` also
    shares one identical xref edge with `keep.asm`."""
    store.append_annotation(_routine_ann(0xC000, "old.asm"), source="t")
    store.append_annotation(_routine_ann(0xD000, "keep.asm"), source="t")
    store.append_annotation(_xref_ann(0xC010, 0xC100, "old.asm"), source="t")
    store.append_annotation(_xref_ann(0xD010, 0xD100, "keep.asm"), source="t")
    # Identical edge asserted by BOTH sources:
    store.append_annotation(_xref_ann(0xE000, 0xE100, "old.asm"), source="t")
    store.append_annotation(_xref_ann(0xE000, 0xE100, "keep.asm"), source="t")
    store.append_annotation(_smc_ann(0xC020, 0xC030, "old.asm"), source="t")
    store.append_annotation(_smc_ann(0xD020, 0xD030, "keep.asm"), source="t")
    store.append_annotation(_classify_ann(0xC000, 0xC00F, "old.asm"),
                            source="t")
    store.append_annotation(_classify_ann(0xD000, 0xD00F, "keep.asm"),
                            source="t")


def _assert_only_keep_remains(store):
    """Post-invalidation assertions shared by live/reopen checks."""
    # xrefs: old.asm's own edge gone; keep.asm's edges — including its
    # provenance row for the SHARED edge — survive.
    xrefs = store.query(
        "SELECT src_addr, source_file FROM code_xrefs ORDER BY src_addr",
    )
    assert {(r["src_addr"], r["source_file"]) for r in xrefs} == {
        (0xD010, "keep.asm"), (0xE000, "keep.asm"),
    }
    # SMC sites:
    smc = store.query("SELECT src_addr FROM code_smc_sites")
    assert {r["src_addr"] for r in smc} == {0xD020}
    # Classification bytes:
    cls = store.query("SELECT DISTINCT source_file FROM code_class")
    assert {r["source_file"] for r in cls} == {"keep.asm"}
    # Canonical annotations:
    anns = store.query(
        "SELECT COUNT(*) AS n FROM annotations WHERE source_file = 'old.asm'",
    )
    assert anns[0]["n"] == 0
    anns_keep = store.query(
        "SELECT COUNT(*) AS n FROM annotations WHERE source_file = 'keep.asm'",
    )
    assert anns_keep[0]["n"] > 0
    # Routines (pre-hardening behavior, still holds):
    starts = {r["start_addr"]
              for r in store.query("SELECT start_addr FROM code_routines")}
    assert starts == {0xD000}


def test_invalidation_covers_all_projections_and_preserves_others(
    code_store,
):
    store, tmp = code_store
    _seed_all_projections(store)
    store.invalidate_source("old.asm")
    _assert_only_keep_remains(store)

    # Same result after close/reopen (code_kb fully replays its event
    # log on every open — the invalidation event replays in order).
    reloaded = CodeKnowledgeStore.load_or_init(tmp / "code_kb")
    _assert_only_keep_remains(reloaded)


def test_dump_change_invalidates_disasm_projections(code_store):
    """`capstone:%` / `vice:%` LIKE invalidation reaches xrefs and SMC
    rows derived from stale dump disassembly, not just labels."""
    store, _tmp = code_store
    store.append_annotation(
        _xref_ann(0xC010, 0xC100, "capstone:$C000+256"), source="t")
    store.append_annotation(
        _smc_ann(0xC020, 0xC030, "vice:$C000+32"), source="t")
    store.append_annotation(_xref_ann(0xD010, 0xD100, "game.asm"), source="t")

    store.invalidate_source("capstone:%", like=True)
    store.invalidate_source("vice:%", like=True)

    xrefs = store.query("SELECT source_file FROM code_xrefs")
    assert {r["source_file"] for r in xrefs} == {"game.asm"}
    assert store.query("SELECT * FROM code_smc_sites") == []


# ---------------------------------------------------------------------------
# Final hardening: indirect xrefs, legacy classify, singleton restoration
# ---------------------------------------------------------------------------

def _indirect_ann(src, via, source_file):
    from code_kb.schema import ANN_INDIRECT
    return Annotation(
        layer=0, kind=ANN_INDIRECT, start_addr=src, end_addr=src,
        producer="layer0", confidence=1.0,
        payload={"src_addr": src, "via_vector": via, "resolved_target": None,
                 "source_file": source_file},
    )


def test_indirect_xrefs_carry_provenance_and_invalidate(code_store):
    """Reproduced defect: ANN_INDIRECT projected code_xrefs rows with
    NULL source_file, so path invalidation left them stale."""
    store, tmp = code_store
    store.append_annotation(_indirect_ann(0xC050, 0x0314, "old.asm"),
                            source="t")
    store.append_annotation(_indirect_ann(0xD050, 0x0316, "keep.asm"),
                            source="t")
    rows = store.query(
        "SELECT src_addr, source_file FROM code_xrefs"
        " WHERE kind = 'jmp_indirect'",
    )
    assert {(r["src_addr"], r["source_file"]) for r in rows} == {
        (0xC050, "old.asm"), (0xD050, "keep.asm"),
    }

    store.invalidate_source("old.asm")

    def _check(s):
        rows = s.query(
            "SELECT src_addr, source_file FROM code_xrefs"
            " WHERE kind = 'jmp_indirect'",
        )
        assert {(r["src_addr"], r["source_file"]) for r in rows} == {
            (0xD050, "keep.asm"),
        }

    _check(store)                                            # live
    _check(CodeKnowledgeStore.load_or_init(tmp / "code_kb"))  # full replay


def _legacy_classify_ann(start, end, classification, evidence_text):
    """EXACT pre-hardening payload shape: no source_file key."""
    return Annotation(
        layer=0, kind=ANN_CLASSIFY, start_addr=start, end_addr=end,
        producer="deterministic", confidence=1.0,
        payload={"classification": classification,
                 "evidence_text": evidence_text},
    )


def test_legacy_classify_provenance_recovered_and_invalidated(code_store):
    """Legacy classify events (pre-hardening payloads) must be reachable
    by path invalidation via evidence-template recovery — and removed
    ranges must NOT survive when the new file emits no replacement."""
    store, tmp = code_store
    store.append_annotation(_legacy_classify_ann(
        0xC000, 0xC00F, "code",
        "16 consecutive parsed instructions in old.asm",
    ), source="layer0.asm")
    store.append_annotation(_legacy_classify_ann(
        0xC100, 0xC10F, "data",
        "gap marker in old.asm (16 bytes)",
    ), source="layer0.asm")
    store.append_annotation(_legacy_classify_ann(
        0xD000, 0xD00F, "code",
        "16 consecutive parsed instructions in keep.asm",
    ), source="layer0.asm")

    # Recovery attributed the rows at apply time.
    srcs = store.query("SELECT DISTINCT source_file FROM code_class")
    assert {r["source_file"] for r in srcs} == {"old.asm", "keep.asm"}

    store.invalidate_source("old.asm")   # no re-ingest follows

    def _check(s):
        rows = s.query("SELECT addr, source_file FROM code_class")
        assert all(r["source_file"] == "keep.asm" for r in rows)
        assert all(0xD000 <= r["addr"] <= 0xD00F for r in rows)
        anns = s.query(
            "SELECT COUNT(*) AS n FROM annotations"
            " WHERE source_file = 'old.asm'",
        )
        assert anns[0]["n"] == 0

    _check(store)
    _check(CodeKnowledgeStore.load_or_init(tmp / "code_kb"))


def test_unattributable_legacy_classify_swept_conservatively(code_store):
    """A deterministic layer-0 classify whose provenance cannot be
    recovered is dropped on ANY invalidation (documented conservative
    policy: never present possibly-stale facts as current)."""
    store, _tmp = code_store
    store.append_annotation(_legacy_classify_ann(
        0xE000, 0xE00F, "code", "manually curated — no path here",
    ), source="layer0.asm")
    assert store.query("SELECT * FROM code_class")  # present before

    store.invalidate_source("unrelated.asm")

    assert store.query("SELECT * FROM code_class") == []
    anns = store.query(
        "SELECT COUNT(*) AS n FROM annotations WHERE kind = 'classify'",
    )
    assert anns[0]["n"] == 0


def _named_routine_ann(start, name, source_file):
    return Annotation(
        layer=0, kind=ANN_ROUTINE, start_addr=start, end_addr=start + 0x20,
        producer="layer0", confidence=0.9,
        payload={"name": name, "summary": "s", "source_file": source_file,
                 "entries": [start], "exits": [], "size_bytes": 0x21},
    )


def _label_ann(addr, name, source_file):
    return Annotation(
        layer=0, kind=ANN_LABEL, start_addr=addr, end_addr=addr,
        producer="layer0", confidence=0.9,
        payload={"name": name, "source_file": source_file},
    )


def _disasm_ann(addr, bytes_hex, source_file):
    from code_kb.schema import ANN_DISASM
    return Annotation(
        layer=0, kind=ANN_DISASM, start_addr=addr, end_addr=addr,
        producer="deterministic", confidence=1.0,
        payload={"source_file": source_file, "instructions": [
            {"addr": addr, "bytes_hex": bytes_hex, "mnemonic": "lda",
             "operand": "#$01", "size_bytes": 2},
        ]},
    )


def test_singleton_projections_restore_surviving_source(code_store):
    """Reproduced defect: A contributes a fact, B overwrites the same
    singleton key, B is invalidated → A's still-valid fact must be
    re-materialized, not left deleted. Covers routines, labels,
    instructions, classifications."""
    store, tmp = code_store
    # A first, B overwrites each singleton key.
    store.append_annotation(_named_routine_ann(0xC000, "from_a", "a.asm"),
                            source="t")
    store.append_annotation(_label_ann(0xC000, "shared_lbl", "a.asm"),
                            source="t")
    store.append_annotation(_disasm_ann(0xC000, "a901", "a.asm"), source="t")
    store.append_annotation(_classify_ann(0xC000, 0xC00F, "a.asm"),
                            source="t")

    store.append_annotation(_named_routine_ann(0xC000, "from_b", "b.asm"),
                            source="t")
    store.append_annotation(_label_ann(0xC000, "shared_lbl", "b.asm"),
                            source="t")
    store.append_annotation(_disasm_ann(0xC000, "a902", "b.asm"), source="t")
    store.append_annotation(_classify_ann(0xC000, 0xC00F, "b.asm"),
                            source="t")

    # B won every singleton key.
    assert store.query("SELECT name FROM code_routines")[0]["name"] == "from_b"
    assert store.query(
        "SELECT source_file FROM code_labels",
    )[0]["source_file"] == "b.asm"
    assert store.query(
        "SELECT bytes_hex FROM instructions",
    )[0]["bytes_hex"] == "a902"
    assert store.query(
        "SELECT DISTINCT source_file FROM code_class",
    )[0]["source_file"] == "b.asm"

    store.invalidate_source("b.asm")

    def _check_a_restored(s):
        assert s.query(
            "SELECT name, source_file FROM code_routines",
        )[0]["name"] == "from_a"
        assert s.query(
            "SELECT source_file FROM code_labels",
        )[0]["source_file"] == "a.asm"
        assert s.query(
            "SELECT bytes_hex FROM instructions",
        )[0]["bytes_hex"] == "a901"
        cls = s.query("SELECT DISTINCT source_file FROM code_class")
        assert {r["source_file"] for r in cls} == {"a.asm"}

    _check_a_restored(store)                                  # live
    _check_a_restored(CodeKnowledgeStore.load_or_init(tmp / "code_kb"))


def test_invalidating_older_source_preserves_current_one(code_store):
    store, tmp = code_store
    store.append_annotation(_named_routine_ann(0xC000, "from_a", "a.asm"),
                            source="t")
    store.append_annotation(_named_routine_ann(0xC000, "from_b", "b.asm"),
                            source="t")

    store.invalidate_source("a.asm")

    def _check(s):
        rows = s.query("SELECT name, source_file FROM code_routines")
        assert [(r["name"], r["source_file"]) for r in rows] == [
            ("from_b", "b.asm"),
        ]

    _check(store)
    _check(CodeKnowledgeStore.load_or_init(tmp / "code_kb"))


# ---------------------------------------------------------------------------
# Final hardening: rematerialization must use TRUE append order, never
# (ts, id) — event ids are random, so equal-timestamp annotations could
# have their order reversed and the singleton winner flipped.
# ---------------------------------------------------------------------------

_FROZEN_TS = "2026-07-18T12:00:00+00:00"


def _ann_for_kind(kind_name, name, source_file, *, ann_id, ts):
    """One singleton-key annotation per projection kind, with explicit
    event id and timestamp (the reproducer needs equal timestamps and
    deliberately reversed lexical ids)."""
    if kind_name == "routine":
        ann = _named_routine_ann(0xC000, name, source_file)
    elif kind_name == "label":
        ann = _label_ann(0xC000, "shared_lbl", source_file)
    elif kind_name == "instruction":
        ann = _disasm_ann(0xC000, "a9" + ("01" if name == "from_a" else "02"),
                          source_file)
    elif kind_name == "classify":
        ann = _classify_ann(0xC000, 0xC00F, source_file)
    else:  # pragma: no cover
        raise AssertionError(kind_name)
    ann.id = ann_id
    ann.ts = ts
    return ann


def _singleton_winner(store, kind_name):
    """The (marker, source_file) currently winning the singleton key."""
    if kind_name == "routine":
        r = store.query("SELECT name, source_file FROM code_routines")[0]
        return r["name"], r["source_file"]
    if kind_name == "label":
        r = store.query("SELECT source_file FROM code_labels")[0]
        return None, r["source_file"]
    if kind_name == "instruction":
        r = store.query("SELECT bytes_hex, source_file FROM instructions")[0]
        return r["bytes_hex"], r["source_file"]
    r = store.query("SELECT DISTINCT source_file FROM code_class")[0]
    return None, r["source_file"]


@pytest.mark.parametrize("kind_name",
                         ["routine", "label", "instruction", "classify"])
def test_rematerialize_uses_append_order_not_ts_id(code_store, kind_name):
    """Exact reproducer: A appended FIRST with a lexically LATER id, B
    appended second with a lexically EARLIER id, identical timestamps.
    B wins before rematerialization; invalidating an UNRELATED source
    must not flip the winner to A ((ts, id) ordering did)."""
    store, tmp = code_store
    store.append_annotation(
        _ann_for_kind(kind_name, "from_a", "a.asm",
                      ann_id="zzzzzzzzzzzz", ts=_FROZEN_TS),
        source="t",
    )
    store.append_annotation(
        _ann_for_kind(kind_name, "from_b", "b.asm",
                      ann_id="aaaaaaaaaaaa", ts=_FROZEN_TS),
        source="t",
    )
    assert _singleton_winner(store, kind_name)[1] == "b.asm"  # append order

    # Unrelated source appended + invalidated → rematerialization runs.
    store.append_annotation(_label_ann(0xF000, "unrelated", "c.asm"),
                            source="t")
    store.invalidate_source("c.asm")

    assert _singleton_winner(store, kind_name)[1] == "b.asm"  # B still wins

    # Reopen = complete event-log replay: B must still win.
    reloaded = CodeKnowledgeStore.load_or_init(tmp / "code_kb")
    assert _singleton_winner(reloaded, kind_name)[1] == "b.asm"

    # Invalidating B (the overlapping source) restores A.
    store.invalidate_source("b.asm")
    assert _singleton_winner(store, kind_name)[1] == "a.asm"
    reloaded2 = CodeKnowledgeStore.load_or_init(tmp / "code_kb")
    assert _singleton_winner(reloaded2, kind_name)[1] == "a.asm"


def test_per_source_xref_smc_rows_unchanged_by_seq_ordering(code_store):
    """Per-source xref/SMC projection behavior is order-insensitive and
    must be byte-identical after a seq-ordered rematerialization."""
    store, _tmp = code_store
    store.append_annotation(_xref_ann(0xE000, 0xE100, "a.asm"), source="t")
    store.append_annotation(_xref_ann(0xE000, 0xE100, "b.asm"), source="t")
    store.append_annotation(_smc_ann(0xE010, 0xE020, "a.asm"), source="t")
    store.append_annotation(_smc_ann(0xE010, 0xE020, "b.asm"), source="t")

    store.append_annotation(_label_ann(0xF000, "unrelated", "c.asm"),
                            source="t")
    store.invalidate_source("c.asm")          # triggers rematerialization

    xrefs = store.query("SELECT src_addr, source_file FROM code_xrefs")
    assert {(r["src_addr"], r["source_file"]) for r in xrefs} == {
        (0xE000, "a.asm"), (0xE000, "b.asm"),
    }
    smc = store.query("SELECT src_addr, source_file FROM code_smc_sites")
    assert {(r["src_addr"], r["source_file"]) for r in smc} == {
        (0xE010, "a.asm"), (0xE010, "b.asm"),
    }


def test_code_kb_dump_change_detection(code_store):
    store, tmp = code_store
    dump = tmp / "game.bin"
    dump.write_bytes(b"\x01" * 64)

    evt, changed = store.ingest_dump(dump)
    assert evt is not None and changed is False     # first ingest

    evt, changed = store.ingest_dump(dump)
    assert evt is None and changed is False         # same content

    dump.write_bytes(b"\x02" * 64)
    evt, changed = store.ingest_dump(dump)
    assert evt is not None and changed is True      # content changed
    assert store.read_bytes(0, 2) == b"\x02\x02"
