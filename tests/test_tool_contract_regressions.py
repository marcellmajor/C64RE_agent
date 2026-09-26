"""Regression coverage for malformed plans and transport/trace contract drift."""

import pytest

import graph.nodes as nodes
from graph.plan_utils import step_is_concrete
from tools.agent_runner import validate_review_plan, _plan_review_from_state
from tools.arguments import ToolArgsError
from tools.vice_trace import writer_candidates


def step(tool, **args):
    return {"id": "s1", "tool": tool, "args": args, "depends_on": []}


@pytest.mark.parametrize("encoding", ["base64", "HEX", "binary", 7])
def test_invalid_encoding_never_reaches_transport(monkeypatch, encoding):
    from tools import vice_mcp
    monkeypatch.setattr(vice_mcp, "call_tool", lambda *a, **k: pytest.fail("transport reached"))
    args = {"method": "vice.memory.read", "address": "$0800", "size": 1024, "encoding": encoding}
    assert not step_is_concrete(step("vice", **args))
    with pytest.raises(ToolArgsError, match="encoding"):
        nodes._vice_call(args.pop("method"), args)
    with pytest.raises(ToolArgsError, match="encoding"):
        validate_review_plan([step("vice", method="vice.memory.read", **args)])


def test_bad_planner_options_remain_editable_before_approval():
    plan = [step("vice", method="vice.memory.read", address="$0800", size=8, encoding="base64")]
    review = _plan_review_from_state({"plan": plan}, thread_id="test", seen_messages=0, started=0)
    assert review.plan == plan
    with pytest.raises(ToolArgsError, match="encoding"):
        validate_review_plan(review.plan)


@pytest.mark.parametrize("method", [
    "vice.registers.set", "vice.machine.config.set", "vice.checkpoint.toggle",
    "vice.vicii.set_state", "vice.sid.set_state", "vice.cia.set_state",
    "vice.sprite.set", "vice.run_until", "vice_run_until", "vice.future_mutation",
])
def test_mutations_cannot_bypass_approval(monkeypatch, method):
    monkeypatch.setenv("VICE_MCP_URL", "http://offline.invalid")
    monkeypatch.setattr(nodes, "_vice_call", lambda *a, **k: pytest.fail("unapproved call"))
    state = {"plan": [step("vice", method=method)], "current_step_id": "s1",
             "require_vice_approval": True, "approved_mutation_steps": []}
    result = nodes.vice_mcp_node(state)["tool_results"][0]
    assert result["rejection"] == "human_approval_required"


def test_stopwatch_read_is_read_only_but_reset_requires_approval():
    assert not nodes.vice_step_requires_approval(step("vice", method="vice.cycles.stopwatch", action="read"))
    assert nodes.vice_step_requires_approval(step("vice", method="vice.cycles.stopwatch", action="reset"))


@pytest.mark.parametrize(("raw", "expected"), [("81C0", 0x81C0), ("8171", 0x8171), (0, 0)])
def test_code_kb_uses_same_addresses_as_vice(raw, expected):
    from graph.code_kb_node import _mode_refs_to
    class Store:
        def query(self, sql, params):
            assert params[0] == expected
            return []
    result = _mode_refs_to(Store(), {"addr": raw}, "s1")["tool_results"][0]
    assert result["ok"]
    assert result["addr"] == nodes._coerce_vice_address(raw)


@pytest.mark.parametrize("raw", ["$FROM_s1", None, -1, 65536, "49556", True])
def test_invalid_reference_addresses_never_query_zero(raw):
    from graph.code_kb_node import _mode_refs_to
    class Store:
        def query(self, *args):
            pytest.fail("invalid address queried")
    with pytest.raises(ToolArgsError):
        _mode_refs_to(Store(), {"addr": raw}, "s1")
    assert nodes._coerce_vice_address(raw) is None


@pytest.mark.parametrize("handler", [nodes._capstone_recursive, nodes._capstone_polymorphic, nodes._recursive_insns_int_keyed])
def test_unresolved_recursive_entry_never_defaults(monkeypatch, handler):
    monkeypatch.setattr(nodes.c64_disasm, "recursive_disasm", lambda *a, **k: pytest.fail("disassembled fallback"))
    with pytest.raises(ToolArgsError):
        if handler is nodes._recursive_insns_int_keyed:
            handler(bytes(65536), {"entry": "$ADDRESS_FROM_s1"})
        else:
            handler(bytes(65536), {"entry": "$ADDRESS_FROM_s1"}, "s1")


@pytest.mark.parametrize("size", [0, 65536, 2.9, True, "all"])
def test_invalid_sizes_are_not_silently_clamped(size):
    with pytest.raises(ToolArgsError):
        nodes._vice_normalize_args("vice.memory.read", {"address": "$0800", "size": size})


def test_memory_search_and_checkpoint_delete_contracts():
    valid = step("vice", method="vice.memory.search", start="$0800", end="$0BFF", pattern=[1, 2])
    assert step_is_concrete(valid)
    assert not step_is_concrete(step("vice", method="vice.memory.search", address="$0800", size=1024))
    assert step_is_concrete(step("vice", method="vice.checkpoint.delete", checkpoint_num=1))


def test_false_strings_are_not_true_and_junk_is_rejected():
    normalized = nodes._vice_normalize_args("vice.checkpoint.add", {"start": "$0044", "stop": "false", "store": "false"})
    assert normalized["stop"] is False and normalized["store"] is False
    with pytest.raises(ToolArgsError):
        nodes._vice_normalize_args("vice.checkpoint.add", {"start": "$0044", "store": "maybe"})


@pytest.mark.parametrize(("method", "args", "expected"), [
    ("vice.execution.reset", {}, "vice_machine_reset"),
    ("resources.set", {"resources": {"Speed": 100}}, "vice_machine_config_set"),
    ("vice_memory_read", {"address": "$0800", "size": 8}, "vice_memory_read"),
])
def test_aliases_reach_registered_names(monkeypatch, method, args, expected):
    from tools import vice_mcp
    calls = []
    def call(name, arguments):
        calls.append((name, arguments))
        return {"data": "ok"}
    monkeypatch.setattr(vice_mcp, "call_tool", call)
    nodes._vice_call(method, args)
    assert calls[0][0] == expected


def test_code_kb_disassembly_uses_native_transport_name(monkeypatch):
    from code_kb.disasm import disasm_vice
    from tools import vice_mcp
    monkeypatch.setenv("VICE_MCP_URL", "http://offline.invalid")
    calls = []
    def call(name, args):
        calls.append(name)
        return {"is_error": True, "data": "intentional offline stop"}
    monkeypatch.setattr(vice_mcp, "call_tool", call)
    assert not disasm_vice(object(), address=0x81C0)["ok"]
    assert calls == ["vice_disassemble"]


@pytest.mark.parametrize("limit", [None, "all", 0, -1, 1.5])
def test_bad_kb_limit_is_a_step_error_not_a_graph_exception(monkeypatch, limit):
    monkeypatch.setattr(nodes, "get_store", lambda _: object())
    result = nodes.kb_query_node({"kb_handle": "fake", "plan": [step("kb", mode="stats", limit=limit)],
                                  "current_step_id": "s1"})["tool_results"][0]
    assert not result["ok"] and "limit" in result["data"]


def test_trace_literal_match_does_not_confuse_pc_or_indexed_operand():
    assert not nodes._disasm_writes_addr("$0044: 8D 20 D0 STA $D020", 0x44)
    assert not nodes._disasm_writes_addr("$8000: 95 44 STA $44,X", 0x44)
    assert nodes._disasm_writes_addr("$8000: 85 44 STA $44", 0x44)


@pytest.mark.parametrize(("opcode", "operands", "registers", "target", "pointer"), [
    (0x85, [0x44], {}, 0x44, {}),
    (0x95, [0x44], {"X": 2}, 0x46, {}),
    (0x9D, [0xFF, 0x0A], {"X": 1}, 0x0B00, {}),
    (0x91, [0x32], {"Y": 0x29}, 0x0A80, {0x32: 0x57, 0x33: 0x0A}),
    (0x81, [0xFF], {"X": 0}, 0x0A80, {0xFF: 0x80, 0: 0x0A}),
])
def test_trace_reconstructs_effective_destination(opcode, operands, registers, target, pointer):
    mem = bytearray(65536)
    raw = bytes([opcode, *operands])
    pc = 0x8000 + len(raw)
    mem[0x8000:pc] = raw
    for addr, value in pointer.items():
        mem[addr] = value
    candidates = writer_candidates(pc, target, registers, lambda a, n: bytes(mem[a:a+n]))
    assert any(c["address"] == "$8000" and c["effective_address"] == f"${target:04X}" for c in candidates)
    assert not writer_candidates(pc, (target + 1) & 65535, registers, lambda a, n: bytes(mem[a:a+n]))
