"""Bounded Code-KB Layer-2 grouping and Layer-3 critique (tracker 6.8)."""

from __future__ import annotations

import json

import graph.code_kb_node as code_node
from code_kb import Annotation, get_code_store
from code_kb.schema import ANN_CRITIQUE, ANN_HYPOTHESIS, ANN_ROUTINE
from graph.plan_utils import is_parallel_read_only_step, step_is_concrete


def _routine(start, end, name):
    return Annotation(
        layer=0,
        kind=ANN_ROUTINE,
        start_addr=start,
        end_addr=end,
        producer="layer0",
        confidence=1.0,
        payload={
            "name": name,
            "summary": f"verified {name}",
            "source_file": "test.asm",
            "entries": [start],
            "exits": [],
            "size_bytes": end - start + 1,
        },
    )


def _state(handle, mode, **args):
    return {
        "code_kb_handle": handle,
        "question": "How does input reach rendering?",
        "current_step_id": "s1",
        "plan": [{
            "id": "s1",
            "tool": "code_kb",
            "args": {"mode": mode, **args},
        }],
    }


def test_layer_role_normalization_replaces_planner_prose_and_string_backups():
    role, backups = code_node._normalize_layer_roles(
        "animation flag initializer",
        "critic",
        default_role="analyst",
        default_backup_roles=["critic", "synthesizer"],
    )

    assert role == "analyst"
    assert backups == ["critic"]


def test_layer2_rejects_hallucinated_members_and_persists_valid_group(
    tmp_path, monkeypatch,
):
    handle = str(tmp_path / "code_kb")
    store = get_code_store(handle)
    store.append_annotation(_routine(0xC000, 0xC00F, "read_input"), source="test")
    store.append_annotation(_routine(0xC100, 0xC11F, "move_sprite"), source="test")
    store.append_annotation(Annotation(
        layer=1,
        kind=ANN_HYPOTHESIS,
        start_addr=0xC000,
        end_addr=0xC00F,
        producer="layer1:test",
        confidence=0.8,
        payload={
            "text": "Reads joystick input.",
            "name_suggestion": "read_input",
            "routine_id": "r1",
            "source_file": "test.asm",
        },
    ), source="test")

    monkeypatch.setattr(code_node, "_invoke_layer1", lambda *_args, **_kwargs: ({
        "groups": [{
            "name": "player_control",
            "summary": "Input feeds sprite movement.",
            "routine_starts": ["$C000", "$C100", "$DEAD"],
            "confidence": 0.86,
            "evidence": ["$C000 reads input", "$C100 moves sprite"],
        }],
    }, "analyst", None))

    result = code_node.code_kb_node(_state(handle, "layer2"))["tool_results"][0]

    assert result["ok"] is True
    assert result["groups_written"] == 1
    assert result["rejected_addresses"] == ["$DEAD"]
    payload = json.loads(result["data"])
    annotation_id = payload["groups"][0]["annotation_id"]
    rows = store.query(
        "SELECT layer, kind, payload_json FROM annotations WHERE id = ?",
        (annotation_id,),
    )
    assert rows[0]["layer"] == 2
    assert rows[0]["kind"] == "group"
    assert json.loads(rows[0]["payload_json"])["routine_starts"] == [
        0xC000, 0xC100,
    ]

    listed = code_node.code_kb_node(
        _state(handle, "groups"),
    )["tool_results"][0]
    assert listed["ok"] and "player_control" in listed["data"]


def test_layer3_targets_exact_known_annotation_and_keeps_layer0_immutable(
    tmp_path, monkeypatch,
):
    handle = str(tmp_path / "code_kb")
    store = get_code_store(handle)
    store.append_annotation(_routine(0xC000, 0xC00F, "main_loop"), source="test")
    target_id = store.append_annotation(Annotation(
        layer=1,
        kind=ANN_HYPOTHESIS,
        start_addr=0xC000,
        end_addr=0xC00F,
        producer="layer1:test",
        confidence=0.7,
        payload={
            "text": "Possibly a score routine.",
            "name_suggestion": "maybe_score",
            "routine_id": "r1",
            "source_file": "test.asm",
        },
    ), source="test")
    monkeypatch.setattr(code_node, "_invoke_layer1", lambda *_args, **_kwargs: ({
        "critiques": [
            {
                "annotation_id": target_id,
                "decision": "reject",
                "critique": "No score writes appear in the Layer-0 range.",
                "confidence": 0.91,
                "evidence": ["$C000-$C00F"],
            },
            {
                "annotation_id": "h_prefix_collision",
                "decision": "support",
                "critique": "must be rejected",
                "confidence": 1.0,
            },
        ],
    }, "critic", None))

    result = code_node.code_kb_node(_state(handle, "layer3"))["tool_results"][0]

    assert result["ok"] is True
    assert result["critiques_written"] == 1
    assert result["rejected_targets"] == ["h_prefix_collision"]
    critiques = store.query(
        "SELECT layer, kind, payload_json FROM annotations WHERE kind = ?",
        (ANN_CRITIQUE,),
    )
    assert len(critiques) == 1
    payload = json.loads(critiques[0]["payload_json"])
    assert payload["target_annotation_id"] == target_id
    assert payload["decision"] == "reject"
    # Layer 3 records contrary evidence; it cannot overwrite deterministic
    # routine identity or delete the challenged annotation.
    assert store.query(
        "SELECT name FROM code_routines WHERE start_addr = ?", (0xC000,),
    )[0]["name"] == "main_loop"
    assert store.query(
        "SELECT COUNT(*) AS n FROM annotations WHERE id = ?", (target_id,),
    )[0]["n"] == 1

    listed = code_node.code_kb_node(
        _state(handle, "critiques"),
    )["tool_results"][0]
    assert listed["ok"] and target_id in listed["data"]


def test_deeper_layer_execution_modes_stay_serial_but_queries_can_fan_out():
    for mode in ("layer2", "layer3"):
        step = {"tool": "code_kb", "args": {"mode": mode}}
        assert step_is_concrete(step)
        assert not is_parallel_read_only_step(step)
    for mode in ("groups", "critiques"):
        step = {"tool": "code_kb", "args": {"mode": mode}}
        assert step_is_concrete(step)
        assert is_parallel_read_only_step(step)


def test_layer2_without_verified_routines_does_not_call_llm(tmp_path, monkeypatch):
    handle = str(tmp_path / "code_kb")
    get_code_store(handle)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Layer 2 must validate its Layer-0 input first")

    monkeypatch.setattr(code_node, "_invoke_layer1", unexpected)
    result = code_node.code_kb_node(
        _state(handle, "layer2"),
    )["tool_results"][0]
    assert result["ok"] is False
    assert "at least two verified" in result["data"]


# ---------------------------------------------------------------------------
# Interior addresses resolve to the enclosing routine and report the mapping.
# ---------------------------------------------------------------------------

def test_routine_interior_address_resolves_to_known_routine(tmp_path):
    handle = str(tmp_path / "code_kb")
    store = get_code_store(handle)
    store.append_annotation(_routine(0xC380, 0xC3F2, "score_update"), source="test")

    res = code_node._mode_routine(store, {"start": "$C394"}, "s1")
    result = res["tool_results"][0]

    assert result["ok"] is True
    assert result["requested_start_addr"] == 0xC394
    assert result["start_addr"] == 0xC380
    assert "$C380-$C3F2" in result["data"]
    assert "score_update" in result["data"]
    assert "using its entry $C380" in result["data"]


def test_routine_miss_outside_every_routine_keeps_the_generic_hint(tmp_path):
    handle = str(tmp_path / "code_kb")
    store = get_code_store(handle)
    store.append_annotation(_routine(0xC380, 0xC3F2, "score_update"), source="test")

    res = code_node._mode_routine(store, {"start": "$8000"}, "s1")
    result = res["tool_results"][0]

    assert result["ok"] is False
    assert "Try mode='routines' first." in result["data"]
    assert "lies inside" not in result["data"]


def test_pseudocode_interior_address_resolves_to_known_routine(tmp_path):
    from code_kb.layer0 import build_from_disasm_window
    handle = str(tmp_path / "code_kb")
    store = get_code_store(handle)
    store.append_annotation(_routine(0xC380, 0xC3F2, "score_update"), source="test")
    build_from_disasm_window([
        {"addr": 0xC380, "bytes": b"\xea", "mnemonic": "nop"},
        {"addr": 0xC3A0, "bytes": b"\x60", "mnemonic": "rts"},
    ], store)

    res = code_node._mode_pseudocode(store, {"start": "$C3A0"}, "s1")
    result = res["tool_results"][0]

    assert result["ok"] is True
    assert result["requested_start_addr"] == 0xC3A0
    assert result["start_addr"] == 0xC380
    assert "$C380-$C3F2" in result["data"]
    assert "$C380" in result["data"] and "$C3A0" in result["data"]
