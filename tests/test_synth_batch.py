"""Deterministic + batched synthesis (tracker 1.2) and the Layer-1
auto-annotate cap (tracker 1.3), against a real KnowledgeStore."""

import pytest

import graph.nodes as nodes
from graph.nodes import MAX_AUTO_ANNOTATE_PER_SYNTH, SYNTH_BATCH_MAX_RESULTS
from memory import get_store
from memory.schema import EVT_HYPOTHESIS, EVT_LABEL, EVT_TOOL_RESULT


@pytest.fixture()
def kb(tmp_path):
    handle = str(tmp_path / "kb")
    return handle, get_store(handle)


def _linear_result(step_id, data="$C000: LDA $0780\n$C003: STA $D020"):
    return {"step_id": step_id, "tool": "capstone", "ok": True,
            "mode": "linear", "data": data}


def _empty_llm_reply():
    return {"labels": [], "routines": [], "data_structures": [],
            "hypotheses": [], "notes": ""}


def _seed_layer0_routines(code_store, routines):
    from code_kb import Annotation
    from code_kb.schema import ANN_ROUTINE

    for routine in routines:
        start = int(routine["start"].lstrip("$"), 16)
        end = int(routine["end"].lstrip("$"), 16)
        code_store.append_annotation(Annotation(
            layer=0,
            kind=ANN_ROUTINE,
            start_addr=start,
            end_addr=end,
            producer="test_layer0",
            confidence=1.0,
            payload={
                "name": routine["name"],
                "entries": [start],
                "exits": [],
                "size_bytes": end - start + 1,
                "source_file": "test.asm",
            },
        ), source="test")


def _state(kb_handle, *, plan, results, processed=0):
    return {
        "kb_handle": kb_handle,
        "question": "where is the lives counter?",
        "plan": plan,
        "tool_results": results,
        "synth_processed_count": processed,
    }


def test_defer_while_steps_pending(kb, monkeypatch):
    """One worthy result + pending steps → record only, no LLM call."""
    handle, store = kb

    def _boom(*a, **kw):  # pragma: no cover
        raise AssertionError("synthesizer called the LLM on a deferred batch")
    monkeypatch.setattr(nodes, "_safe_invoke", _boom)

    plan = [
        {"id": "s1", "tool": "capstone", "depends_on": []},
        {"id": "s2", "tool": "kb", "depends_on": []},   # still pending
    ]
    out = nodes.synthesizer_node(
        _state(handle, plan=plan, results=[_linear_result("s1")]),
    )
    # Raw result recorded exactly once, extraction deferred.
    rows = store.query(
        "SELECT COUNT(*) AS n FROM events WHERE kind = ?", (EVT_TOOL_RESULT,),
    )
    assert rows[0]["n"] == 1
    assert "synth_processed_count" not in out
    assert "deferred" in out["messages"][0].content


def test_flush_when_plan_drained(kb, monkeypatch):
    handle, store = kb
    calls = []

    def _fake(role, prompt, fallback, **kw):
        calls.append(prompt)
        return {**_empty_llm_reply(),
                "labels": [{"addr": "$0780", "name": "lives",
                            "confidence": 0.9}]}
    monkeypatch.setattr(nodes, "_safe_invoke", _fake)

    plan = [{"id": "s1", "tool": "capstone", "depends_on": []}]
    results = [_linear_result("s1")]
    out = nodes.synthesizer_node(_state(handle, plan=plan, results=results))

    assert len(calls) == 1
    assert out["synth_processed_count"] == 1
    labels = store.query("SELECT payload_json FROM events WHERE kind = ?",
                         (EVT_LABEL,))
    assert len(labels) == 1
    assert '"provenance": "llm"' in labels[0]["payload_json"]
    assert "LLM batch of 1" in out["messages"][0].content


def test_flush_when_batch_fills_despite_pending_steps(kb, monkeypatch):
    handle, _store = kb
    calls = []

    def _fake(role, prompt, fallback, **kw):
        calls.append(prompt)
        return _empty_llm_reply()
    monkeypatch.setattr(nodes, "_safe_invoke", _fake)

    plan = [
        *({"id": f"s{i}", "tool": "capstone", "depends_on": []}
          for i in range(1, SYNTH_BATCH_MAX_RESULTS + 1)),
        {"id": "s_last", "tool": "kb", "depends_on": []},  # keeps plan pending
    ]
    results = [
        _linear_result(f"s{i}", data=f"$C{i:03X}: DEC $078{i}")
        for i in range(1, SYNTH_BATCH_MAX_RESULTS + 1)
    ]
    out = nodes.synthesizer_node(_state(handle, plan=plan, results=results))
    assert len(calls) == 1
    assert f"LLM batch of {SYNTH_BATCH_MAX_RESULTS}" in out["messages"][0].content


def test_failed_and_kb_results_never_reach_the_llm(kb, monkeypatch):
    handle, store = kb

    def _boom(*a, **kw):  # pragma: no cover
        raise AssertionError("unworthy results reached the extraction LLM")
    monkeypatch.setattr(nodes, "_safe_invoke", _boom)

    plan = [
        {"id": "s1", "tool": "capstone", "depends_on": []},
        {"id": "s2", "tool": "kb", "depends_on": []},
    ]
    results = [
        {"step_id": "s1", "tool": "capstone", "ok": False, "mode": "linear",
         "data": "no KB handle", "retryable": False},
        {"step_id": "s2", "tool": "kb", "ok": True, "mode": "stats",
         "data": '{"stats": {"events_total": 3}}'},
    ]
    out = nodes.synthesizer_node(_state(handle, plan=plan, results=results))
    # Both recorded as raw events, tail marked considered, no LLM.
    rows = store.query("SELECT COUNT(*) AS n FROM events WHERE kind = ?",
                       (EVT_TOOL_RESULT,))
    assert rows[0]["n"] == 2
    assert out["synth_processed_count"] == 2


def test_vectors_extracted_mechanically_without_llm(kb, monkeypatch):
    """`vectors` output is pure structure — labels appear, LLM untouched."""
    handle, store = kb

    def _boom(*a, **kw):  # pragma: no cover
        raise AssertionError("mechanical-only modes must not use the LLM")
    monkeypatch.setattr(nodes, "_safe_invoke", _boom)

    plan = [{"id": "s1", "tool": "capstone", "depends_on": []}]
    results = [
        {"step_id": "s1", "tool": "capstone", "ok": True, "mode": "vectors",
         "data": "{...}",
         "info": {
             "hw_vectors": [
                 {"vector": "$FFFE", "name": "IRQ/BRK", "target": "$C05D"},
                 {"vector": "$FFFC", "name": "RESET", "target": None},
             ],
             "ram_vectors": [
                 {"vector": "$0314", "name": "CINV (IRQ)", "target": "$C0F0"},
             ],
         }},
    ]
    out = nodes.synthesizer_node(_state(handle, plan=plan, results=results))

    labels = store.query("SELECT addr, name, kind FROM labels ORDER BY addr")
    assert {(r["addr"], r["kind"]) for r in labels} == {
        (0xC05D, "vector"), (0xC0F0, "vector"),
    }
    assert "mechanical" in out["messages"][0].content


def test_find_loops_is_hybrid_mechanical_plus_llm(kb, monkeypatch):
    """`find_loops` candidates become hypotheses mechanically, while its
    embedded auto-disassembly listing still goes to the (batched) LLM."""
    handle, store = kb
    calls = []

    def _fake(role, prompt, fallback, **kw):
        calls.append(prompt)
        return _empty_llm_reply()
    monkeypatch.setattr(nodes, "_safe_invoke", _fake)

    plan = [{"id": "s2", "tool": "capstone", "depends_on": []}]
    results = [
        {"step_id": "s2", "tool": "capstone", "ok": True, "mode": "find_loops",
         "data": "#1 $C05D score=42\n\nAuto-disassembly of top candidate…",
         "candidates": [
             {"address": "$C05D", "score": 42.0, "reasons": ["IRQ vector"]},
         ]},
    ]
    out = nodes.synthesizer_node(_state(handle, plan=plan, results=results))

    hyps = store.query("SELECT id, text FROM hypotheses")
    assert len(hyps) == 1
    assert hyps[0]["id"] == "h_loop_c05d"
    assert "$C05D" in hyps[0]["text"]
    assert len(calls) == 1  # the prose listing is still LLM-worthy
    assert "mechanical" in out["messages"][0].content


def test_rerecorded_results_do_not_duplicate_mechanical_facts(kb, monkeypatch):
    handle, store = kb
    monkeypatch.setattr(
        nodes, "_safe_invoke", lambda *a, **kw: _empty_llm_reply(),
    )
    plan = [{"id": "s1", "tool": "capstone", "depends_on": []}]
    results = [
        {"step_id": "s1", "tool": "capstone", "ok": True, "mode": "find_loops",
         "data": "#1 $C05D score=42",
         "candidates": [{"address": "$C05D", "score": 42.0, "reasons": []}]},
    ]
    nodes.synthesizer_node(_state(handle, plan=plan, results=results))
    nodes.synthesizer_node(_state(handle, plan=plan, results=results,
                                  processed=1))
    hyps = store.query("SELECT COUNT(*) AS n FROM events WHERE kind = ?",
                       (EVT_HYPOTHESIS,))
    assert hyps[0]["n"] == 1  # stage-1 dedup gated the second pass


def test_auto_annotate_capped_and_prioritized(kb, tmp_path, monkeypatch):
    """N extracted routines fire at most MAX_AUTO_ANNOTATE_PER_SYNTH
    Layer-1 calls (tracker 1.3), highest (confidence×relevance) first."""
    handle, _store = kb
    from code_kb import get_code_store
    code_handle = tmp_path / "code_kb"
    code_handle.mkdir()
    code_store = get_code_store(str(code_handle))

    routines = [
        {"start": f"$C{i}00", "end": f"$C{i}40", "name": f"sub_c{i}00",
         "summary": "updates the lives counter" if i == 3 else "misc helper",
         "confidence": 0.9, "calls_to": [], "called_by": []}
        for i in range(1, 5)
    ]
    _seed_layer0_routines(code_store, routines)
    monkeypatch.setattr(
        nodes, "_safe_invoke",
        lambda *a, **kw: {**_empty_llm_reply(), "routines": routines},
    )

    annotated: list[str] = []

    def _fake_annotate(state, store, args, step_id):
        annotated.append(args["start"])
        return {"tool_results": [{"step_id": step_id, "tool": "code_kb",
                                  "ok": True, "data": "annotated"}]}

    import graph.code_kb_node as ckn
    monkeypatch.setattr(ckn, "_mode_annotate", _fake_annotate)

    plan = [{"id": "s1", "tool": "capstone", "depends_on": []}]
    state = _state(handle, plan=plan, results=[_linear_result("s1")])
    state["code_kb_handle"] = str(code_handle)
    out = nodes.synthesizer_node(state)

    assert len(annotated) == MAX_AUTO_ANNOTATE_PER_SYNTH
    # The question-relevant routine ("lives counter") must be in the batch.
    assert "$C300" in annotated
    assert "deferred" in out["messages"][0].content  # skipped candidates noted


def _annotate_test_state(kb_handle, tmp_path, monkeypatch, fake_annotate,
                         n_routines=1):
    """Shared scaffold: flush path with N extracted routines conf 0.9."""
    from code_kb import get_code_store
    code_handle = tmp_path / "code_kb"
    code_handle.mkdir(exist_ok=True)
    code_store = get_code_store(str(code_handle))

    routines = [
        {"start": f"$C{i}00", "end": f"$C{i}40", "name": f"sub_c{i}00",
         "summary": "misc helper", "confidence": 0.9,
         "calls_to": [], "called_by": []}
        for i in range(1, n_routines + 1)
    ]
    _seed_layer0_routines(code_store, routines)
    monkeypatch.setattr(
        nodes, "_safe_invoke",
        lambda *a, **kw: {**_empty_llm_reply(), "routines": routines},
    )
    import graph.code_kb_node as ckn
    monkeypatch.setattr(ckn, "_mode_annotate", fake_annotate)

    plan = [{"id": "s1", "tool": "capstone", "depends_on": []}]
    state = _state(kb_handle, plan=plan, results=[_linear_result("s1")])
    state["code_kb_handle"] = str(code_handle)
    return state


def test_auto_annotation_usage_reaches_state(kb, tmp_path, monkeypatch):
    """Review finding 1: `_mode_annotate` returns through `_record_result`,
    whose drain moved the Layer-1 usage into the (previously discarded)
    return value — the synthesizer must re-record it so tokens/cost reach
    the node's state update."""
    handle, _store = kb
    layer1_entry = {"role": "layer1:analyst", "model": "m",
                    "input_tokens": 500, "output_tokens": 40,
                    "cost_usd": 0.02, "ok": True}

    def _fake_annotate(state, store, args, step_id):
        return {
            "tool_results": [{"step_id": step_id, "tool": "code_kb",
                              "ok": True, "data": "annotated"}],
            "llm_usage": [dict(layer1_entry)],
            "budget_used": 0.02,
            "tokens_used": 540,
        }

    state = _annotate_test_state(handle, tmp_path, monkeypatch, _fake_annotate)
    out = nodes.synthesizer_node(state)

    roles = [e["role"] for e in out["llm_usage"]]
    assert "layer1:analyst" in roles
    assert out["tokens_used"] >= 540
    assert out["budget_used"] >= 0.02
    assert "auto-annotated $C100" in out["messages"][0].content


def test_failed_auto_annotation_not_logged_as_success(kb, tmp_path,
                                                      monkeypatch):
    handle, _store = kb

    def _fake_annotate(state, store, args, step_id):
        return {
            "tool_results": [{"step_id": step_id, "tool": "code_kb",
                              "ok": False,
                              "data": "no Layer-0 window at this address"}],
            "llm_usage": [{"role": "layer1:analyst", "model": "m",
                           "input_tokens": 100, "output_tokens": 0,
                           "cost_usd": 0.0, "ok": False,
                           "error": "empty response"}],
        }

    state = _annotate_test_state(handle, tmp_path, monkeypatch, _fake_annotate)
    out = nodes.synthesizer_node(state)

    msg = out["messages"][0].content
    assert "auto-annotation of $C100 FAILED" in msg
    assert "auto-annotated $C100" not in msg
    # The failed call's usage is still accounted for.
    assert any(e["role"] == "layer1:analyst" and not e["ok"]
               for e in out["llm_usage"])
