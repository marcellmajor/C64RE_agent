"""Golden manifest contracts plus opt-in end-to-end graph evaluation."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from evals.runner import (
    GoldenCase,
    ROOT,
    evaluate_turn,
    execute_case,
    inspect_dump_anchors,
    load_cases,
)


CASES = load_cases()
RUN_GOLDEN = os.getenv("C64RE_RUN_GOLDEN", "").strip().lower() in {
    "1", "true", "yes", "on",
}


def test_golden_manifest_is_valid_and_covers_the_five_corpus_dumps():
    assert 10 <= len(CASES) <= 20
    assert len({case.id for case in CASES}) == len(CASES)
    assert {case.dump_path for case in CASES} == {
        "memdump_dir/bubbob_fullmem.bin",
        "memdump_dir/landed_blood.bin",
        "memdump_dir/petch_ingame.bin",
        "memdump_dir/vultures_ingame.bin",
        "memdump_dir/wizofwor.bin",
    }
    for case in CASES:
        assert case.dump.is_file() and case.dump.stat().st_size == 0x10000
        assert case.asm_paths
        assert all(path.is_file() for path in case.asm_paths)
        assert 0.0 <= case.min_confidence <= 1.0


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_golden_addresses_have_offline_dump_evidence_and_consistent_asm(case):
    anchors = inspect_dump_anchors(case)
    assert [anchor.address for anchor in anchors] == list(case.expected_addresses)
    for anchor in anchors:
        assert anchor.supported, (
            f"{case.id} {anchor.address}: {'; '.join(anchor.failures)}"
        )
        assert anchor.bytes_hex
        assert anchor.disassembly or anchor.reference_sites
        assert anchor.evidence


def test_dump_anchor_rejects_exact_address_asm_byte_conflict(tmp_path):
    dump = tmp_path / "dump.bin"
    memory = bytearray(0x10000)
    memory[0x1000:0x1002] = b"\xA9\x01"
    dump.write_bytes(memory)
    asm = tmp_path / "conflicting.asm"
    asm.write_text("1000  A9 02  LDA #$02\n")
    case = GoldenCase.from_dict({
        "id": "asm_conflict",
        "game": "Synthetic",
        "question": "Where is the value loaded?",
        "dump_path": str(dump),
        "asm_files": [str(asm)],
        "expected_addresses": ["$1000"],
        "expected_keywords": ["value"],
        "min_confidence": 0.5,
    })

    anchor = inspect_dump_anchors(case)[0]

    assert anchor.supported is False
    assert any("but dump has" in failure for failure in anchor.failures)


def test_dump_anchor_rejects_empty_unreferenced_ram(tmp_path):
    dump = tmp_path / "empty.bin"
    dump.write_bytes(bytes(0x10000))
    case = GoldenCase.from_dict({
        "id": "empty_ram",
        "game": "Synthetic",
        "question": "What is at the empty address?",
        "dump_path": str(dump),
        "asm_files": [],
        "expected_addresses": ["$1000"],
        "expected_keywords": ["empty"],
        "min_confidence": 0.5,
    })

    anchor = inspect_dump_anchors(case)[0]

    assert anchor.supported is False
    assert any("empty RAM" in failure for failure in anchor.failures)


def test_golden_evaluator_records_quality_cost_and_runtime_metrics():
    case = CASES[0]
    turn = SimpleNamespace(
        answer="Player lives are stored at $045A.",
        evidence=["$04D8 decrements the lives value"],
        critique="supported",
        open_questions=[],
        addresses=[0x045A, 0x04D8],
        confidence=0.8,
        verdict="accept",
        raw_state={
            "run_id": "golden-test-run",
            "tokens_used": 180,
            "budget_used": 0.01,
            "tool_results": [{"ok": True}, {"ok": True}],
            "llm_usage": [
                {"input_tokens": 100, "output_tokens": 30},
                {"input_tokens": 40, "output_tokens": 10},
            ],
        },
    )
    result = evaluate_turn(case, turn, wall_time_s=1.25)
    assert result.passed is True
    assert result.llm_calls == 2
    assert result.input_tokens == 140
    assert result.output_tokens == 40
    assert result.total_tokens == 180
    assert result.cost_usd == 0.01
    assert result.tool_calls == 2
    assert result.wall_time_s == 1.25
    assert result.run_id == "golden-test-run"


def _offline_turn(
    *, answer: str, addresses: list[int], confidence: float,
    evidence: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        answer=answer,
        evidence=evidence or [],
        addresses=addresses,
        confidence=confidence,
        verdict="accept",
        raw_state={},
    )


def test_golden_evaluator_fails_when_an_address_is_missing():
    case = CASES[0]
    turn = _offline_turn(
        answer="The lives counter is identified, but no address was recovered.",
        addresses=[],
        confidence=0.8,
    )
    result = evaluate_turn(case, turn, wall_time_s=0.0)
    assert result.passed is False
    assert result.missing_addresses == ["$045A"]


def test_golden_evaluator_fails_when_a_keyword_is_missing():
    case = CASES[0]
    turn = _offline_turn(
        answer="The counter is stored at $045A.",
        addresses=[0x045A],
        confidence=0.8,
    )
    result = evaluate_turn(case, turn, wall_time_s=0.0)
    assert result.passed is False
    assert result.missing_keywords == ["lives"]


def test_golden_keyword_matching_rejects_substrings():
    case = CASES[0]
    turn = _offline_turn(
        answer="Olives are stored at $045A.",
        addresses=[0x045A],
        confidence=0.8,
    )
    result = evaluate_turn(case, turn, wall_time_s=0.0)
    assert result.passed is False
    assert result.missing_keywords == ["lives"]


def test_golden_evaluator_fails_below_the_confidence_floor():
    case = CASES[0]
    turn = _offline_turn(
        answer="Player lives are stored at $045A.",
        addresses=[0x045A],
        confidence=case.min_confidence - 0.01,
    )
    result = evaluate_turn(case, turn, wall_time_s=0.0)
    assert result.passed is False
    assert any("confidence" in reason for reason in result.failure_reasons)


def test_golden_evaluator_accepts_short_address_format_after_integer_parse():
    case = CASES[0]
    turn = _offline_turn(
        answer="Player lives are stored at $45A.",
        addresses=[int("45A", 16)],
        confidence=0.8,
    )
    result = evaluate_turn(case, turn, wall_time_s=0.0)
    assert result.passed is True
    assert result.found_addresses == ["$045A"]


def test_state_keyword_does_not_accept_an_unrelated_verb_use():
    case = next(case for case in CASES if case.id == "petch_entity_state")
    turn = _offline_turn(
        answer="I will state that nothing is known about $1F6C and $0D25.",
        addresses=[0x1F6C, 0x0D25],
        confidence=0.8,
    )
    result = evaluate_turn(case, turn, wall_time_s=0.0)
    assert result.passed is False
    assert result.missing_keywords == ["state"]


@pytest.mark.parametrize("phrase", ["entity state", "animation state"])
def test_state_keyword_accepts_relevant_noun_phrases(phrase):
    case = next(case for case in CASES if case.id == "petch_entity_state")
    turn = _offline_turn(
        answer=f"The {phrase} is written at $1F6C into $0D25.",
        addresses=[0x1F6C, 0x0D25],
        confidence=0.8,
    )
    result = evaluate_turn(case, turn, wall_time_s=0.0)
    assert result.passed is True


def test_state_keyword_accepts_natural_intervening_qualifiers():
    case = next(case for case in CASES if case.id == "petch_entity_state")
    turn = _offline_turn(
        answer=(
            "The animation and setup state is written at $1F6C into $0D25."
        ),
        addresses=[0x1F6C, 0x0D25],
        confidence=0.8,
    )
    result = evaluate_turn(case, turn, wall_time_s=0.0)
    assert result.passed is True


def test_flag_keyword_accepts_natural_plural_form():
    case = next(case for case in CASES if case.id == "petch_animation_flags")
    turn = _offline_turn(
        answer="The helper at $5A61 initializes animation flags at $7900.",
        addresses=[0x5A61, 0x7900],
        confidence=0.9,
    )
    result = evaluate_turn(case, turn, wall_time_s=0.0)
    assert result.passed is True


@pytest.mark.parametrize(
    "phrase",
    ["is limited", "limits the value", "is bounded", "checks the bound"],
)
def test_limit_keyword_accepts_natural_forms(phrase):
    case = next(case for case in CASES if case.id == "blood_prng_bounded")
    turn = _offline_turn(
        answer=f"The routine at $1D3A {phrase} using $00BE.",
        addresses=[0x1D3A, 0x00BE],
        confidence=0.8,
    )
    result = evaluate_turn(case, turn, wall_time_s=0.0)
    assert result.passed is True


def test_lives_keyword_accepts_natural_singular_form():
    case = next(case for case in CASES if case.id == "bubble_lives_decrement")
    turn = _offline_turn(
        answer="A player life is decremented at $04D8 using $045A.",
        addresses=[0x04D8, 0x045A],
        confidence=0.8,
    )
    result = evaluate_turn(case, turn, wall_time_s=0.0)
    assert result.passed is True


@pytest.mark.slow
@pytest.mark.golden
@pytest.mark.skipif(
    not RUN_GOLDEN,
    reason="set C64RE_RUN_GOLDEN=1 to run paid/live graph evaluations",
)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_golden_question_through_real_graph(case, tmp_path):
    result = execute_case(case, session_root=tmp_path / case.id)
    assert result.passed, "; ".join(result.failure_reasons)
