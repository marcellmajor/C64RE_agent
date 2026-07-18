"""Offline regression coverage for the Phase 6 product/UX services."""

from __future__ import annotations

import json

import pytest

import graph.nodes as nodes
from code_kb import Annotation, export_vice_symbols, get_code_store
from code_kb.schema import ANN_HYPOTHESIS, ANN_LABEL, ANN_ROUTINE
from memory import get_store
from memory.evidence import resolve_evidence_ref
from memory.schema import EVT_HYPOTHESIS, EVT_LABEL, EVT_ROUTINE
from tools.agent_runner import mutating_step_ids, validate_review_plan
from tools.research_notebook import (
    append_turn,
    diff_dump_states,
    freeze_dump_state,
    load_dump_catalog,
    prior_answers_digest,
    versioned_report_path,
)


def _vice_gate_state(method, *, step_id="poke1", approved=None):
    return {
        "plan": [{
            "id": step_id,
            "tool": "vice",
            "args": {
                "method": method,
                "address": "$C000",
                "size": 1,
                "value": 1,
            },
        }],
        "current_step_id": step_id,
        "require_vice_approval": True,
        "approved_mutation_steps": list(approved or []),
    }


@pytest.mark.parametrize(("method", "canonical"), [
    ("poke", "vice.memory.write"),
    ("memory.write", "vice.memory.write"),
    ("vice.machine.reset", "vice.machine.reset"),
    ("reset", "vice.machine.reset"),
    ("vice.autostart", "vice.autostart"),
    ("vice.disk.attach", "vice.disk.attach"),
    ("attach_disk", "vice.disk.attach"),
    ("tape.attach", "vice.tape.attach"),
    ("cartridge.attach", "vice.cartridge.attach"),
    ("snapshot.load", "vice.snapshot.load"),
    ("resources.set", "vice.resources.set"),
    ("checkpoint_delete", "vice.checkpoint.delete"),
    ("keyboard.type", "vice.keyboard.type"),
])
def test_mutating_vice_methods_require_then_honor_step_approval(
    method, canonical, monkeypatch,
):
    calls = []

    def fake_call(called_method, args):
        calls.append((called_method, args))
        return {"tool": called_method, "args": args, "data": "ok"}

    monkeypatch.setattr(nodes, "_vice_call", fake_call)
    monkeypatch.setenv("VICE_MCP_URL", "http://offline-test.invalid")

    rejected = nodes.vice_mcp_node(_vice_gate_state(method))
    rejection = rejected["tool_results"][0]
    assert rejection["ok"] is False
    assert rejection["method"] == canonical
    assert rejection["rejection"] == "human_approval_required"
    assert calls == []

    allowed = nodes.vice_mcp_node(
        _vice_gate_state(method, approved=["poke1"]),
    )
    assert allowed["tool_results"][0]["ok"] is True
    assert calls[0][0] == canonical


def test_read_only_vice_method_does_not_require_approval(monkeypatch):
    calls = []

    def fake_call(method, args):
        calls.append((method, args))
        return {"tool": method, "args": args, "data": [0]}

    monkeypatch.setattr(nodes, "_vice_call", fake_call)
    monkeypatch.setenv("VICE_MCP_URL", "http://offline-test.invalid")
    out = nodes.vice_mcp_node(_vice_gate_state("memory.read"))

    assert out["tool_results"][0]["ok"] is True
    assert calls[0][0] == "vice.memory.read"


def test_poke_verify_requires_approval_before_composite_dispatch(monkeypatch):
    calls = []

    def fake_composite(state, method, args, step_id):
        calls.append((method, args, step_id))
        return nodes._record_result(
            "vice", step_id, True, "mock experiment complete",
            extra={"method": method},
        )

    monkeypatch.setattr(nodes, "_vice_composite", fake_composite)
    monkeypatch.setenv("VICE_MCP_URL", "http://offline-test.invalid")
    state = {
        "plan": [{
            "id": "experiment",
            "tool": "vice",
            "args": {
                "method": "poke_and_peek",
                "address": "$00C0",
                "value": 7,
                "expect": "lives digit changes",
            },
        }],
        "current_step_id": "experiment",
        "require_vice_approval": True,
        "approved_mutation_steps": [],
    }
    rejected = nodes.vice_mcp_node(state)["tool_results"][0]
    assert rejected["rejection"] == "human_approval_required"
    assert rejected["method"] == "vice.poke_verify"
    assert calls == []

    state["approved_mutation_steps"] = ["experiment"]
    allowed = nodes.vice_mcp_node(state)["tool_results"][0]
    assert allowed["ok"] is True
    assert calls[0][0] == "vice.poke_verify"


def test_review_plan_identifies_mutations_and_rejects_unknown_dependencies():
    plan = validate_review_plan([
        {"id": "read", "tool": "vice", "args": {
            "method": "memory.read", "address": "$C000", "size": 8,
        }},
        {"id": "trace", "tool": "vice", "args": {
            "method": "trace", "address": "$C000",
        }, "depends_on": ["read"]},
    ])
    assert mutating_step_ids(plan) == ["trace"]

    with pytest.raises(ValueError, match="unknown reviewed plan dependencies"):
        validate_review_plan([
            {"id": "s1", "tool": "kb", "depends_on": ["missing"]},
        ])
    with pytest.raises(ValueError, match="unsupported tool"):
        validate_review_plan([
            {"id": "s1", "tool": "shell", "depends_on": []},
        ])
    with pytest.raises(ValueError, match="contain a cycle"):
        validate_review_plan([
            {"id": "s1", "tool": "kb", "depends_on": ["s2"]},
            {"id": "s2", "tool": "kb", "depends_on": ["s1"]},
        ])


def test_turn_archive_digest_is_idempotent_and_reports_are_versioned(tmp_path):
    record = {
        "run_id": "run-one",
        "question": "Where are lives stored?",
        "answer": "The lives counter is at $0780.",
        "confidence": 0.9,
        "verdict": "accept",
        "open_questions": ["Who decrements it?"],
    }
    assert append_turn(tmp_path, record) is True
    assert append_turn(tmp_path, record) is False

    digest = prior_answers_digest(
        tmp_path, current_question="Who decrements it?",
    )
    assert "Prior accepted answers" in digest
    assert "The lives counter is at $0780" in digest
    assert "Who decrements it?" in digest

    report = versioned_report_path(
        tmp_path,
        started_at="2026-07-18T12:34:56+00:00",
        run_id="abc-123",
    )
    assert report.name == "report_20260718T123456000000Z_abc123.md"


def test_named_dump_catalog_freezes_bytes_and_diffs_offline(tmp_path):
    before = tmp_path / "before.dump"
    after = tmp_path / "after.dump"
    before_bytes = bytearray(0x10000)
    after_bytes = bytearray(before_bytes)
    after_bytes[0x0780] = 2
    before.write_bytes(before_bytes)
    after.write_bytes(after_bytes)

    title = freeze_dump_state(
        tmp_path, name="Title Screen", source_path=before,
    )
    ingame = freeze_dump_state(
        tmp_path, name="In Game", source_path=after,
    )
    assert title["name"] == "title_screen"
    assert ingame["name"] == "in_game"
    assert title["path"] == "dumps/title_screen.bin"
    assert [row["name"] for row in load_dump_catalog(tmp_path)] == [
        "title_screen", "in_game",
    ]
    assert diff_dump_states(
        tmp_path, before="title_screen", after="in_game",
    ) == [{
        "addr": 0x0780,
        "addr_hex": "$0780",
        "old": 0,
        "new": 2,
        "delta": 2,
        "region": "screen_ram",
    }]
    assert nodes._load_named_snapshot(
        {"kb_handle": str(tmp_path / "kb")}, "Title--Screen",
    ) == bytes(before_bytes)

    changed = bytearray(before_bytes)
    changed[0x2000] = 1
    before.write_bytes(changed)
    with pytest.raises(ValueError, match="frozen with different bytes"):
        freeze_dump_state(
            tmp_path, name="Title Screen", source_path=before,
        )


def test_shared_evidence_resolver_handles_addresses_hypotheses_and_events(tmp_path):
    store = get_store(tmp_path / "kb")
    label_event = store.append_event(EVT_LABEL, "test", {
        "addr": 0x0780,
        "name": "lives_counter",
        "kind": "variable",
        "confidence": 0.9,
    })
    store.append_event(EVT_HYPOTHESIS, "test", {
        "id": "h_lives",
        "text": "Lives are decremented once per death.",
        "status": "open",
        "evidence": [label_event],
    })

    assert "lives_counter" in resolve_evidence_ref("$0780", store)
    assert "Lives are decremented" in resolve_evidence_ref("h_lives", store)
    assert "label `lives_counter`" in resolve_evidence_ref(label_event, store)


def test_hypothesis_lifecycle_is_append_only(tmp_path):
    store = get_store(tmp_path / "kb")
    hypothesis_event = store.append_event(EVT_HYPOTHESIS, "analyst", {
        "id": "h_irq",
        "text": "The IRQ decrements the timer.",
        "status": "open",
        "evidence": ["$C100"],
    })
    changed = store.transition_hypotheses(
        [hypothesis_event], status="supported", source="critic_accept",
        evidence=["$C120"],
    )
    assert changed == ["h_irq"]
    row = store.query(
        "SELECT status, evidence_json FROM hypotheses WHERE id = ?",
        ("h_irq",),
    )[0]
    assert row["status"] == "supported"
    assert json.loads(row["evidence_json"]) == ["$C100", "$C120"]
    events = store.query(
        "SELECT COUNT(*) AS n FROM events WHERE kind = ?",
        (EVT_HYPOTHESIS,),
    )
    assert events[0]["n"] == 2


def test_hypothesis_lookup_and_transition_prefer_exact_id(tmp_path):
    store = get_store(tmp_path / "kb")
    # Insert the colliding longer id first to reproduce the old unordered
    # `OR id LIKE '%h_irq%' LIMIT 1` failure deterministically.
    store.append_event(EVT_HYPOTHESIS, "analyst", {
        "id": "h_irq_extra",
        "text": "Wrong longer hypothesis.",
        "status": "open",
        "evidence": [],
    })
    store.append_event(EVT_HYPOTHESIS, "analyst", {
        "id": "h_irq",
        "text": "Exact IRQ hypothesis.",
        "status": "open",
        "evidence": [],
    })

    assert store.transition_hypotheses(
        ["h_irq"], status="supported", source="critic_accept",
    ) == ["h_irq"]
    statuses = {
        row["id"]: row["status"]
        for row in store.query("SELECT id, status FROM hypotheses")
    }
    assert statuses == {"h_irq": "supported", "h_irq_extra": "open"}

    resolved = resolve_evidence_ref("h_irq", store)
    assert "Exact IRQ hypothesis." in resolved
    assert "Wrong longer hypothesis." not in resolved


def test_symbol_export_keeps_only_verified_high_confidence_names(tmp_path):
    store = get_code_store(tmp_path / "code_kb")
    store.append_annotation(Annotation(
        layer=0,
        kind=ANN_ROUTINE,
        start_addr=0xC000,
        end_addr=0xC00F,
        producer="asm",
        confidence=0.95,
        payload={"name": "main_loop", "entries": [0xC000], "exits": []},
    ), source="test")
    store.append_annotation(Annotation(
        layer=1,
        kind=ANN_LABEL,
        start_addr=0x0780,
        end_addr=0x0780,
        producer="reviewer",
        confidence=0.9,
        payload={"name": "lives_counter"},
    ), source="test")
    store.append_annotation(Annotation(
        layer=1,
        kind=ANN_LABEL,
        start_addr=0x0781,
        end_addr=0x0781,
        producer="reviewer",
        confidence=0.6,
        payload={"name": "maybe_score"},
    ), source="test")
    store.append_annotation(Annotation(
        layer=1,
        kind=ANN_HYPOTHESIS,
        start_addr=0xC100,
        end_addr=0xC10F,
        producer="synthesizer",
        confidence=0.99,
        payload={"name_suggestion": "unverified_irq"},
        flags=["unverified_llm"],
    ), source="test")

    symbols = export_vice_symbols(store, min_confidence=0.75)
    assert "al $C000 .main_loop" in symbols
    assert "al $0780 .lives_counter" in symbols
    assert "maybe_score" not in symbols
    assert "unverified_irq" not in symbols


def test_synthesizer_tags_routine_without_layer0_as_unverified(tmp_path, monkeypatch):
    parent_handle = str(tmp_path / "kb")
    parent_store = get_store(parent_handle)
    code_handle = str(tmp_path / "code_kb")
    code_store = get_code_store(code_handle)

    monkeypatch.setattr(nodes, "_safe_invoke", lambda *_args, **_kwargs: {
        "labels": [],
        "routines": [{
            "start": "$C000",
            "end": "$C00F",
            "name": "maybe_main",
            "summary": "Possible main loop.",
            "confidence": 0.91,
        }],
        "data_structures": [],
        "hypotheses": [],
        "notes": "",
    })
    out = nodes.synthesizer_node({
        "kb_handle": parent_handle,
        "code_kb_handle": code_handle,
        "question": "Where is the main loop?",
        "plan": [{"id": "s1", "tool": "capstone", "depends_on": []}],
        "tool_results": [{
            "step_id": "s1", "tool": "capstone", "ok": True,
            "mode": "linear", "data": "$C000: NOP\n$C001: RTS",
        }],
        "synth_processed_count": 0,
    })

    assert out.get("auto_annotated_count", 0) == 0
    assert code_store.query("SELECT * FROM code_routines") == []
    assert parent_store.query("SELECT * FROM routines") == []
    parent_event = parent_store.query(
        "SELECT payload_json FROM events WHERE kind = ?",
        (EVT_HYPOTHESIS,),
    )[0]
    assert json.loads(parent_event["payload_json"])["provenance"] == (
        "unverified_llm"
    )
    assert parent_store.query(
        "SELECT COUNT(*) AS n FROM events WHERE kind = ?",
        (EVT_ROUTINE,),
    )[0]["n"] == 0
    row = code_store.query(
        "SELECT kind, flags_json, payload_json FROM annotations"
        " WHERE layer = 1",
    )[0]
    assert row["kind"] == ANN_HYPOTHESIS
    assert "unverified_llm" in json.loads(row["flags_json"])
    assert json.loads(row["payload_json"])["name_suggestion"] == "maybe_main"
