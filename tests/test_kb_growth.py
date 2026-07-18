"""KB-growth signal against a REAL KnowledgeStore (tracker 0.2 review).

`_substantive_event_count` counts DISTINCT substantive facts:
bookkeeping never counts, failed tool attempts never count, identical
evidence re-recorded under a replan's fresh step id never counts twice,
and genuinely new evidence/facts do count.
"""

import pytest

from graph.nodes import _substantive_event_count
from memory import get_store
from memory.schema import (
    EVT_ANALYSIS,
    EVT_CONSOLIDATED,
    EVT_HYPOTHESIS,
    EVT_LABEL,
    EVT_TOOL_RESULT,
    EVT_VERDICT,
)


@pytest.fixture()
def store(tmp_path):
    return get_store(str(tmp_path / "kb"))


def _tool_result(step_id, *, ok, data, tool="capstone", **extra):
    return {
        "step_id": step_id,
        "tool": tool,
        "ok": ok,
        "data": data,
        **extra,
    }


def test_bookkeeping_events_do_not_count(store):
    store.append_event(EVT_ANALYSIS, "analyst", {"answer": "x"})
    store.append_event(EVT_VERDICT, "critic", {"decision": "replan"})
    store.append_event(EVT_CONSOLIDATED, "curator", {"summary": "s"})
    assert _substantive_event_count(store) == 0


def test_repeated_identical_failures_do_not_count(store):
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result("i1_s3", ok=False, data="TAVILY_API_KEY not set"),
    )
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result("i2_s1", ok=False, data="TAVILY_API_KEY not set"),
    )
    assert _substantive_event_count(store) == 0


def test_identical_success_under_new_step_id_counts_once(store):
    """A replan re-running the same call must not read as progress."""
    disasm = "$C000: LDA $0780\n$C003: STA $D020"
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result("i1_s2", ok=True, data=disasm),
    )
    assert _substantive_event_count(store) == 1
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result("i2_s5", ok=True, data=disasm),
    )
    assert _substantive_event_count(store) == 1


def test_same_prefix_with_new_evidence_tail_counts_twice(store):
    """Expanded windows often share their opening lines; identity must use
    the complete result or decisive evidence added after byte 256 is lost."""
    prefix = "$C000: NOP\n" * 30  # deliberately longer than 256 characters
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result("i1_s2", ok=True, data=prefix + "$C100: RTS"),
    )
    assert _substantive_event_count(store) == 1
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result("i2_s5", ok=True, data=prefix + "$C100: DEC $0780"),
    )
    assert _substantive_event_count(store) == 2


def test_parent_kb_queries_do_not_manufacture_growth(store):
    """The `kb` tool only reads facts/events already in this store. Its
    volatile stats output must not keep the dead-end detector alive."""
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result(
            "i1_s7", ok=True, tool="kb", mode="stats",
            data='{"stats":{"events_total":10}}',
        ),
    )
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result(
            "i2_s7", ok=True, tool="kb", mode="stats",
            data='{"stats":{"events_total":20}}',
        ),
    )
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result(
            "i2_s8", ok=True, tool="kb", mode="labels",
            data='[{"addr":1920,"name":"lives"}]',
        ),
    )
    assert _substantive_event_count(store) == 0


def test_code_kb_metadata_modes_do_not_count(store):
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result(
            "i1_s1", ok=True, tool="code_kb", mode="stats", data="counts=1",
        ),
    )
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result(
            "i2_s1", ok=True, tool="code_kb", mode="schema", data="DDL",
        ),
    )
    assert _substantive_event_count(store) == 0


def test_new_evidence_and_facts_count(store):
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result("i1_s1", ok=True, data="$C000: LDA $0780"),
    )
    assert _substantive_event_count(store) == 1

    # Different tool output → +1.
    store.append_event(
        EVT_TOOL_RESULT, "synthesizer",
        _tool_result("i1_s2", ok=True, data="$C100: DEC $0781"),
    )
    assert _substantive_event_count(store) == 2

    # New label → +1; re-emitting it with different confidence → +0.
    store.append_event(EVT_LABEL, "synthesizer", {
        "addr": 0x0780, "name": "lives", "confidence": 0.5,
    })
    assert _substantive_event_count(store) == 3
    store.append_event(EVT_LABEL, "synthesizer", {
        "addr": 0x0780, "name": "lives", "confidence": 0.9,
    })
    assert _substantive_event_count(store) == 3

    # New hypothesis text → +1; identical text re-emitted → +0.
    store.append_event(EVT_HYPOTHESIS, "synthesizer", {
        "id": "h1", "text": "$0780 is the lives counter", "status": "open",
    })
    store.append_event(EVT_HYPOTHESIS, "synthesizer", {
        "id": "h9", "text": "$0780 is the lives counter", "status": "open",
    })
    assert _substantive_event_count(store) == 4
