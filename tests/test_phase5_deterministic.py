"""Deterministic-layer regression coverage and synthetic dump fixtures (5.2)."""

from __future__ import annotations

from code_kb.asm_parser import ParsedInstruction
from code_kb.layer0 import _detect_routines
from code_kb.scoping import select_asm_files
from graph.nodes import _normalize_plan_ids, _rewrite_kb_sql
from tools import c64_disasm


def test_synthetic_64k_fixture_drives_basic_recursive_and_smc_analysis(
    synthetic_c64_dump,
):
    mem = synthetic_c64_dump.read_bytes()
    assert len(mem) == 0x10000
    assert c64_disasm.detect_basic_sys(mem) == 0xC000

    recursive = c64_disasm.recursive_disasm(
        mem, 0xC000, max_insns=32, seed_vectors=False,
    )
    assert {"$C000", "$C002", "$C005", "$C008", "$C010", "$C012"} <= set(
        recursive["insns"],
    )
    assert recursive["subroutines"] == ["$C010"]

    int_keyed = {
        int(address[1:], 16): info
        for address, info in recursive["insns"].items()
    }
    smc = c64_disasm.detect_polymorphic(int_keyed)
    assert any(
        finding["site"] == "$C002" and finding["writes_to"] == "$C000"
        for finding in smc
    )


def test_rewrite_kb_sql_fixes_columns_but_not_string_literals():
    sql = (
        "SELECT address_hex, start_addr, payload, event_id FROM events "
        "WHERE payload = 'address start_addr payload event_id'"
    )
    rewritten, applied = _rewrite_kb_sql(sql)
    assert rewritten == (
        "SELECT addr, start, payload_json, id FROM events "
        "WHERE payload_json = 'address start_addr payload event_id'"
    )
    assert len(applied) == 4
    assert any("payload_json (×2)" in rule for rule in applied)


def test_normalize_plan_ids_rewrites_dependencies_and_deduplicates():
    plan = [
        {"id": "s1", "tool": "kb", "depends_on": []},
        {"id": "s1", "tool": "capstone", "depends_on": []},
        {"id": "s3", "tool": "kb", "depends_on": ["s1"]},
    ]
    normalized = _normalize_plan_ids(plan, {"iteration": 2})
    assert [step["id"] for step in normalized] == [
        "i3_s1", "i3_s1_2", "i3_s3",
    ]
    assert normalized[2]["depends_on"] == ["i3_s1"]


def test_normalize_plan_ids_defaults_missing_tool_and_invalid_args():
    normalized = _normalize_plan_ids(
        [{"id": "s1", "goal": "fallback step", "args": None}],
        {"iteration": 0},
    )
    assert normalized == [{
        "id": "i1_s1",
        "goal": "fallback step",
        "args": {},
        "tool": "kb",
        "depends_on": [],
    }]


def test_select_asm_files_scopes_by_game_and_honors_override(tmp_path):
    bubble = tmp_path / "bubble_game.asm"
    bobble = tmp_path / "bobble_helpers.txt"
    other = tmp_path / "vultures.asm"
    for path in (bubble, bobble, other):
        path.write_text("C000  60  RTS\n")

    scoped = select_asm_files(game="Bubble Bobble", asm_dir=tmp_path)
    assert scoped.strategy == "heuristic"
    assert {path.name for path in scoped.selected} == {
        "bubble_game.asm", "bobble_helpers.txt",
    }
    assert [path.name for path, _reason in scoped.skipped] == ["vultures.asm"]

    overridden = select_asm_files(
        game="Bubble Bobble", asm_dir=tmp_path,
        override=["vultures.asm"],
    )
    assert overridden.strategy == "override"
    assert [path.name for path in overridden.selected] == ["vultures.asm"]


def _insn(address: int, raw: bytes, mnemonic: str, operand: str = ""):
    return ParsedInstruction(
        addr=address, bytes_=raw, mnemonic=mnemonic,
        operand=operand, raw_operand=operand,
    )


def test_layer0_routine_boundaries_include_jsr_targets_and_gap_heads():
    insns = [
        _insn(0xC000, b"\x20\x10\xC0", "jsr", "$C010"),
        _insn(0xC003, b"\x60", "rts"),
        _insn(0xC010, b"\xEA", "nop"),
        _insn(0xC011, b"\x60", "rts"),
        _insn(0xC020, b"\x60", "rts"),
    ]
    routines = _detect_routines(
        insns,
        labels_by_addr={0xC000: ["entry_main"]},
        gaps=[(0xC012, 0xC01F)],
        jsr_targets={0xC010},
    )
    assert [routine["start_addr"] for routine in routines] == [
        0xC000, 0xC010, 0xC020,
    ]
    assert routines[0]["end_addr"] == 0xC003
    assert routines[1]["end_addr"] == 0xC011
    assert routines[2]["name"] == "sub_c020"
