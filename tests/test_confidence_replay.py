"""Confidence-aware replay (tracker 2.3): lower-confidence re-emissions
must never overwrite stronger facts in the derived view."""

import pytest

from memory.schema import EVT_LABEL, EVT_ROUTINE
from memory.store import KnowledgeStore


@pytest.fixture()
def store(tmp_path):
    return KnowledgeStore.load_or_init(tmp_path / "kb")


def _label(addr, name, conf, kind="code"):
    return {"addr": addr, "name": name, "kind": kind, "confidence": conf}


def _routine(start, name, conf, summary="does things"):
    return {"start": start, "end": start + 0x40, "name": name,
            "summary": summary, "calls_to": [], "called_by": [],
            "confidence": conf}


def test_lower_confidence_label_does_not_downgrade(store):
    store.append_event(EVT_LABEL, "t", _label(0x0780, "lives", 0.9))
    store.append_event(EVT_LABEL, "t", _label(0x0780, "lives", 0.4,
                                              kind="ram_var"))
    rows = store.query("SELECT kind, confidence FROM labels WHERE addr=0x780")
    assert rows[0]["confidence"] == 0.9
    assert rows[0]["kind"] == "code"     # weaker re-emit changed nothing


def test_higher_or_equal_confidence_label_updates(store):
    store.append_event(EVT_LABEL, "t", _label(0x0780, "lives", 0.5))
    store.append_event(EVT_LABEL, "t", _label(0x0780, "lives", 0.5,
                                              kind="ram_var"))
    rows = store.query("SELECT kind FROM labels WHERE addr=0x780")
    assert rows[0]["kind"] == "ram_var"  # equal confidence may refresh
    store.append_event(EVT_LABEL, "t", _label(0x0780, "lives", 0.95))
    rows = store.query("SELECT confidence FROM labels WHERE addr=0x780")
    assert rows[0]["confidence"] == 0.95


def test_lower_confidence_routine_does_not_downgrade(store):
    store.append_event(EVT_ROUTINE, "t",
                       _routine(0xC000, "update_lives", 0.85))
    store.append_event(EVT_ROUTINE, "t",
                       _routine(0xC000, "sub_c000", 0.3, summary="unknown"))
    rows = store.query(
        "SELECT name, summary, confidence FROM routines WHERE start=0xC000",
    )
    assert rows[0]["name"] == "update_lives"
    assert rows[0]["confidence"] == 0.85

    store.append_event(EVT_ROUTINE, "t",
                       _routine(0xC000, "update_lives_bcd", 0.9))
    rows = store.query("SELECT name FROM routines WHERE start=0xC000")
    assert rows[0]["name"] == "update_lives_bcd"


def test_full_replay_preserves_highest_confidence(store):
    """The CAS must hold under a from-scratch rebuild of the derived view
    (events replayed oldest-first, weaker one last)."""
    store.append_event(EVT_LABEL, "t", _label(0x0780, "lives", 0.9))
    store.append_event(EVT_LABEL, "t", _label(0x0780, "lives", 0.2))
    store.append_event(EVT_ROUTINE, "t", _routine(0xC000, "strong", 0.8))
    store.append_event(EVT_ROUTINE, "t", _routine(0xC000, "weak", 0.1))

    store._full_rebuild()

    labels = store.query("SELECT confidence FROM labels WHERE addr=0x780")
    assert labels[0]["confidence"] == 0.9
    routines = store.query("SELECT name FROM routines WHERE start=0xC000")
    assert routines[0]["name"] == "strong"
