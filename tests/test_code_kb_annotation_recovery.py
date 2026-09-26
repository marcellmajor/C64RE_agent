"""Routine resolution and annotation recovery using real stores and decoding."""

import json

import pytest

import graph.code_kb_node as nodes
from code_kb import Annotation, get_code_store
from code_kb.disasm import disasm_capstone, disasm_vice
from code_kb.layer0 import build_from_disasm_window
from code_kb.layer1 import build_window
from code_kb.schema import ANN_ROUTINE


def add_routine(store, start, end):
    store.append_annotation(Annotation(
        layer=0, kind=ANN_ROUTINE, start_addr=start, end_addr=end,
        producer="test", confidence=0.9,
        payload={"name": f"sub_{start:04x}", "source_file": "test.asm",
                 "entries": [start], "exits": [], "size_bytes": end - start + 1},
    ), source="test")


def add_instructions(store, rows):
    build_from_disasm_window(rows, store, source_file="test.asm")


def seed_dump(store, tmp_path, start, data):
    mem = bytearray([0x02]) * 65536  # unsupported opcode stops linear decoding
    mem[start:start + len(data)] = data
    path = tmp_path / "dump.bin"
    path.write_bytes(mem)
    store.ingest_dump(path)


@pytest.fixture
def store(tmp_path):
    return get_code_store(str(tmp_path / "code_kb"))


@pytest.fixture
def llm(monkeypatch):
    prompts = []

    def invoke(role, backups, system, prompt, **kwargs):
        prompts.append(prompt)
        return {"hypothesis": "Test hypothesis", "confidence": 0.6}, role, None

    monkeypatch.setattr(nodes, "_invoke_layer1", invoke)
    return prompts


def annotate(store, addr, **kwargs):
    return nodes._mode_annotate(
        {}, store, {"start": f"${addr:04X}", **kwargs}, "s1",
    )["tool_results"][0]


@pytest.mark.parametrize("start,end,requested", [
    (0x8171, 0x81D5, 0x81C0),
    (0x8518, 0x8587, 0x851C),
    (0x8C92, 0xFF5A, 0x8F10),
    (0x8C92, 0xFF5A, 0x8F14),
])
def test_annotation_resolves_reported_interior_addresses(
    store, llm, monkeypatch, start, end, requested,
):
    add_routine(store, start, end)
    add_instructions(store, [
        {"addr": start, "bytes": b"\xea", "mnemonic": "nop"},
        {"addr": requested, "bytes": b"\x60", "mnemonic": "rts"},
    ])

    def no_disasm(*args, **kwargs):
        pytest.fail("Existing routine evidence must not trigger auto-disassembly")

    monkeypatch.setattr(nodes, "disasm_capstone", no_disasm)
    result = annotate(store, requested)
    assert result["ok"]
    assert result["requested_start_addr"] == requested
    assert result["start_addr"] == start
    assert result["end_addr"] == end
    assert result["context_kind"] == "routine"
    assert not result["auto_disasm"]["attempted"]
    assert f"using its entry ${start:04X}" in llm[0]
    assert f"${requested:04X}" in llm[0]


def test_exact_start_wins_over_smaller_overlapping_routine(store):
    add_routine(store, 0xC000, 0xC100)
    add_routine(store, 0xC010, 0xC015)
    add_routine(store, 0xC012, 0xC030)
    assert nodes._containing_routine(store, 0xC012)["start_addr"] == 0xC012
    assert nodes._containing_routine(store, 0xC014)["start_addr"] == 0xC010
    assert nodes._containing_routine(store, 0xC101) is None


def test_long_routine_listing_keeps_the_requested_instruction(store):
    add_routine(store, 0xC000, 0xC1FF)
    add_instructions(store, [
        {"addr": addr, "bytes": b"\xea", "mnemonic": "nop"}
        for addr in range(0xC000, 0xC200)
    ])
    window = build_window(
        store, routine_row=nodes._containing_routine(store, 0xC1E0),
        focus_addr=0xC1E0,
    )
    assert "$C1E0" in window.listing
    assert "earlier instructions omitted" in window.listing
    assert len([line for line in window.listing.splitlines() if line.startswith("$")]) == 240


@pytest.mark.parametrize("recursive", [False, True])
def test_missing_routine_annotates_provisional_window(store, tmp_path, llm, recursive):
    seed_dump(store, tmp_path, 0x8F14, bytes.fromhex("a911 8d7c98 60"))
    result = annotate(store, 0x8F14, disasm_recursive=recursive)
    assert result["ok"]
    assert result["auto_disasm"]["ok"]
    assert result["start_addr"] == 0x8F14
    assert result["end_addr"] == 0x8F19
    assert result["context_kind"] == "disassembly_window"
    assert "routine boundaries are unverified" in llm[0]
    assert "$8F14" in llm[0] and "LDA" in llm[0]
    assert store.query("SELECT * FROM code_routines") == []
    saved = store.query("SELECT payload_json, flags_json FROM annotations WHERE id = ?",
                        (result["annotation_id"],))[0]
    assert "unverified_routine_boundaries" in json.loads(saved["flags_json"])
    assert json.loads(saved["payload_json"])["routine_id"] is None


def test_recursive_window_excludes_disconnected_callee(store, tmp_path, llm):
    # Call another routine, then return; its body must not extend our window.
    seed_dump(store, tmp_path, 0xC000, bytes.fromhex("2010c0 60"))
    mem = bytearray(store.full_dump())
    mem[0xC010:0xC013] = bytes.fromhex("a90160")
    path = tmp_path / "dump.bin"
    path.write_bytes(mem)
    store.ingest_dump(path)
    result = annotate(store, 0xC000, disasm_recursive=True)
    assert result["ok"] and result["end_addr"] == 0xC003
    listing = llm[0].split("### Listing", 1)[1].split("### Cross-references", 1)[0]
    assert "$C010  " not in listing


def test_empty_known_routine_is_hydrated_before_annotation(store, tmp_path, llm):
    add_routine(store, 0xC000, 0xC002)
    seed_dump(store, tmp_path, 0xC000, bytes.fromhex("a90160"))
    result = annotate(store, 0xC001)
    assert result["ok"] and result["context_kind"] == "routine"
    assert result["auto_disasm"]["reason"] == "empty_listing"
    assert "$C000" in llm[0] and "LDA" in llm[0]


@pytest.mark.parametrize("known_routine", [False, True])
def test_decode_failure_never_calls_llm_or_creates_routine(store, tmp_path, llm, known_routine):
    seed_dump(store, tmp_path, 0x8F0E, bytes.fromhex("4c318f20b98aa91160"))
    add_instructions(store, [
        {"addr": 0x8F0E, "bytes": bytes.fromhex("4c318f"),
         "mnemonic": "jmp", "op_str": "$8F31"},
    ])
    if known_routine:
        add_routine(store, 0x8F10, 0x8F13)  # indexed boundary can itself be wrong
    before = store.stats()
    result = annotate(store, 0x8F10)
    assert not result["ok"]
    assert result["error_code"] == "disassembly_failed"
    assert "inside the indexed instruction at $8F0E" in result["data"]
    assert "known routine/instruction boundary" in result["data"]
    assert result["auto_disasm"]["ok"] is False
    assert llm == []
    assert store.stats() == before


def test_recursive_zero_instructions_is_failure(store, tmp_path):
    seed_dump(store, tmp_path, 0xC000, b"\x02")
    result = disasm_capstone(store, start=0xC000, recursive=True)
    assert not result["ok"] and result["error_code"] == "disassembly_failed"
    assert store.query("SELECT * FROM instructions") == []


def test_auto_disassembly_disabled_does_not_invoke_llm(store, llm):
    result = annotate(store, 0xC000, auto_disasm_if_missing=False)
    assert not result["ok"]
    assert not result["auto_disasm"]["attempted"]
    assert llm == []


def test_empty_vice_response_is_failure(store, monkeypatch):
    from tools import vice_mcp
    monkeypatch.setenv("VICE_MCP_URL", "http://unused.invalid")
    monkeypatch.setattr(vice_mcp, "call_tool", lambda *a, **kw: {"data": ""})
    result = disasm_vice(store, address=0xC000)
    assert not result["ok"]
    assert store.query("SELECT * FROM instructions") == []
