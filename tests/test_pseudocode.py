"""Address-preserving mechanical pseudocode (tracker 3.8 offline slice)."""

from __future__ import annotations

from code_kb import Annotation, get_code_store
from code_kb.pseudocode import (
    render_mechanical_pseudocode,
    translate_instruction,
)
from code_kb.schema import ANN_DISASM, ANN_ROUTINE
from graph.code_kb_node import code_kb_node
from graph.plan_utils import step_is_concrete


def test_core_6502_translations_are_explicit_and_flag_aware():
    assert translate_instruction("lda", "#$01") == "A = $01"
    assert translate_instruction("sta", "$D020") == "mem[$D020] = A"
    assert translate_instruction("adc", "$10") == "A = A + mem[$10] + C"
    assert translate_instruction("sbc", "#$01") == "A = A - $01 - (1 - C)"
    assert translate_instruction("bne", "$C000") == "if (not Z) goto L_C000"
    assert translate_instruction("jsr", "0xc100") == "call L_C100"
    assert translate_instruction("rti", "") == "return_from_interrupt"
    assert translate_instruction("lax", "$20") == 'asm("LAX $20")'


def test_render_keeps_one_authoritative_address_per_instruction():
    text = render_mechanical_pseudocode([
        {"addr": 0xC000, "mnemonic": "lda", "operand": "#$01"},
        {"addr": 0xC002, "mnemonic": "sta", "operand": "$D020"},
        {"addr": 0xC005, "mnemonic": "rts", "operand": ""},
    ], routine_name="set_border")

    assert "MECHANICAL 6502 PSEUDOCODE" in text
    assert "no inferred types/control structures" in text
    for addr in ("$C000", "$C002", "$C005"):
        assert text.count(addr) == 1
    assert "L_C000: A = $01" in text
    assert "L_C005: return" in text


def test_code_kb_pseudocode_mode_uses_only_layer0_rows(tmp_path):
    handle = str(tmp_path / "code_kb")
    store = get_code_store(handle)
    store.append_annotation(Annotation(
        layer=0,
        kind=ANN_ROUTINE,
        start_addr=0xC000,
        end_addr=0xC005,
        producer="test",
        confidence=1.0,
        payload={
            "name": "set_border",
            "summary": "test routine",
            "source_file": "test.asm",
            "entries": [0xC000],
            "exits": [0xC005],
            "size_bytes": 6,
        },
    ), source="test")
    store.append_annotation(Annotation(
        layer=0,
        kind=ANN_DISASM,
        start_addr=0xC000,
        end_addr=0xC005,
        producer="test",
        confidence=1.0,
        payload={
            "source_file": "test.asm",
            "instructions": [
                {"addr": 0xC000, "bytes_hex": "a901", "mnemonic": "lda",
                 "operand": "#$01", "size_bytes": 2},
                {"addr": 0xC002, "bytes_hex": "8d20d0", "mnemonic": "sta",
                 "operand": "$D020", "size_bytes": 3},
                {"addr": 0xC005, "bytes_hex": "60", "mnemonic": "rts",
                 "operand": "", "size_bytes": 1},
            ],
        },
    ), source="test")
    state = {
        "code_kb_handle": handle,
        "current_step_id": "s1",
        "plan": [{
            "id": "s1",
            "tool": "code_kb",
            "args": {"mode": "pseudocode", "start": "$C000"},
        }],
    }

    result = code_kb_node(state)["tool_results"][0]

    assert result["ok"] is True
    assert result["provenance"] == "mechanical_layer0"
    assert result["instruction_count"] == 3
    assert "mem[$D020] = A" in result["data"]
    assert step_is_concrete(state["plan"][0])


def test_pseudocode_mode_refuses_an_unknown_routine(tmp_path):
    handle = str(tmp_path / "code_kb")
    get_code_store(handle)
    result = code_kb_node({
        "code_kb_handle": handle,
        "current_step_id": "s1",
        "plan": [{
            "id": "s1", "tool": "code_kb",
            "args": {"mode": "pseudocode", "start": "$C000"},
        }],
    })["tool_results"][0]

    assert result["ok"] is False
    assert "no routine found" in result["data"]
