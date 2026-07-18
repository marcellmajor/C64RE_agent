"""Mechanical KB extraction for Phase 3 structured outputs (hardening 3.1/3.3/3.7).

Exercises `_mechanical_extract` directly against a real KnowledgeStore so
screen_text / find_counters / vice.memory.diff / vice.memory.monotonic_scan
persist deterministically without the LLM, and re-runs stay idempotent.
"""

import pytest

import graph.nodes as nodes
from memory import get_store
from memory.schema import EVT_LABEL


@pytest.fixture()
def store(tmp_path):
    return get_store(str(tmp_path / "kb"))


def _labels(store):
    return store.query("SELECT addr, name, kind FROM labels ORDER BY addr")


# ---------------------------------------------------------------------------
# 3.3 find_counters: one label per DEC site, idempotent on re-run
# ---------------------------------------------------------------------------

def test_find_counters_one_label_per_site(store):
    result = {
        "tool": "capstone", "ok": True, "mode": "find_counters",
        "candidates": [
            {"address": "$0780", "addr": 0x0780, "kind": "lives",
             "score": 12.0, "alias_kinds": ["timer"],
             "evidence": ["DEC $0780 at $C000"]},
        ],
    }
    nodes._mechanical_extract(store, result, "evt1")
    labels = _labels(store)
    # Exactly ONE ram_var label for the DEC site (not lives+timer).
    assert len(labels) == 1
    assert labels[0]["addr"] == 0x0780
    assert labels[0]["kind"] == "ram_var"
    assert labels[0]["name"] == "candidate_lives_0780"

    # Re-running the same result must not spam a second label.
    nodes._mechanical_extract(store, result, "evt2")
    assert len(_labels(store)) == 1


# ---------------------------------------------------------------------------
# 3.7 screen_text → text labels
# ---------------------------------------------------------------------------

def test_screen_text_labels(store):
    result = {
        "tool": "capstone", "ok": True, "mode": "screen_text",
        "screen_base": "$0400",
        "runs": [
            {"addr": 0x0400, "addr_hex": "$0400", "row": 0, "col": 0,
             "text": "SCORE"},
            {"addr": 0x0428, "addr_hex": "$0428", "row": 1, "col": 0,
             "text": "HI 0000"},
        ],
    }
    nodes._mechanical_extract(store, result, "evt1")
    labels = _labels(store)
    assert {row["name"] for row in labels} == {
        "screen_text_0400", "screen_text_0428"}
    assert all(row["kind"] == "text" for row in labels)
    # Idempotent.
    nodes._mechanical_extract(store, result, "evt2")
    assert len(_labels(store)) == 2


# ---------------------------------------------------------------------------
# 3.1 monotonic_scan → ram_var candidates (kind from delta)
# ---------------------------------------------------------------------------

def test_monotonic_scan_labels_lives(store):
    result = {
        "tool": "vice", "ok": True, "method": "vice.memory.monotonic_scan",
        "delta": -1,
        "candidates": [
            {"addr": 0x00C0, "addr_hex": "$00C0", "region": "zero_page",
             "delta": -1, "values": [5, 4, 3]},
        ],
    }
    nodes._mechanical_extract(store, result, "evt1")
    labels = _labels(store)
    assert len(labels) == 1
    assert labels[0]["name"] == "candidate_lives_00c0"
    assert labels[0]["kind"] == "ram_var"
    # confidence capped
    row = store.query("SELECT confidence FROM labels WHERE addr=0x00C0")[0]
    assert row["confidence"] <= 0.7
    nodes._mechanical_extract(store, result, "evt2")
    assert len(_labels(store)) == 1


# ---------------------------------------------------------------------------
# 3.1 diff → short-listed state-region candidates (I/O excluded, capped)
# ---------------------------------------------------------------------------

def test_diff_shortlist_excludes_io_and_caps(store):
    changed = [
        {"addr": 0x00C0, "addr_hex": "$00C0", "old": 5, "new": 4,
         "delta": -1, "region": "zero_page"},
        {"addr": 0xD012, "addr_hex": "$D012", "old": 1, "new": 2,
         "delta": 1, "region": "io_vic"},   # should never appear
    ] + [
        {"addr": 0x0900 + i, "addr_hex": f"${0x0900 + i:04X}", "old": 0,
         "new": 1, "delta": 1, "region": "low_ram"}
        for i in range(30)                   # 30 low-ram changes → capped to 12
    ]
    result = {
        "tool": "vice", "ok": True, "method": "vice.memory.diff",
        "a": "dump", "b": "after_death", "changed": changed,
    }
    nodes._mechanical_extract(store, result, "evt1")
    labels = _labels(store)
    addrs = {row["addr"] for row in labels}
    assert 0xD012 not in addrs               # I/O excluded
    assert 0x00C0 in addrs                    # zero_page prioritised
    assert len(labels) <= 12                  # short-list cap


# ---------------------------------------------------------------------------
# routing: these results are NOT sent to the extraction LLM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("result", [
    {"tool": "capstone", "ok": True, "mode": "screen_text", "runs": []},
    {"tool": "vice", "ok": True, "method": "vice.memory.diff", "changed": []},
    {"tool": "vice", "ok": True, "method": "vice.memory.monotonic_scan",
     "candidates": []},
    {"tool": "vice", "ok": True, "method": "vice.memory.snapshot"},
])
def test_structured_results_not_llm_worthy(result):
    assert nodes._llm_worthy_result(result) is False


def test_vice_trace_is_llm_worthy():
    # trace output is prose disasm the analyst reasons over.
    assert nodes._llm_worthy_result(
        {"tool": "vice", "ok": True, "method": "vice.trace", "data": "..."}
    ) is True
