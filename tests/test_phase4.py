"""Regression coverage for Phase 4 correctness/robustness fixes."""

from __future__ import annotations

import base64
import sys
import types

import pytest
from langchain_core.messages import AIMessage, SystemMessage

import graph.llm as llm_factory
import graph.nodes as nodes
from memory import extract_hex_addresses, get_store
from memory.schema import EVT_LABEL
from tools.c64_disasm import detect_basic_sys


def _basic_line(mem: bytearray, address: int, body: bytes, *, next_line: int = 0) -> None:
    mem[address] = next_line & 0xFF
    mem[address + 1] = next_line >> 8
    mem[address + 2 : address + 4] = b"\x0a\x00"  # line number 10
    mem[address + 4 : address + 4 + len(body)] = body
    mem[address + 4 + len(body)] = 0


@pytest.mark.parametrize("prefix", [b" ", b"(", b":", b" +", b"\xa0"])
def test_detect_basic_sys_accepts_common_separators(prefix):
    mem = bytearray(0x10000)
    _basic_line(mem, 0x0801, b"\x9e" + prefix + b"2064")
    assert detect_basic_sys(bytes(mem)) == 2064


def test_detect_basic_sys_follows_line_links_beyond_old_window():
    mem = bytearray(0x10000)
    _basic_line(mem, 0x0801, b"REM", next_line=0x0840)
    _basic_line(mem, 0x0840, b"\x9e 49152")
    assert detect_basic_sys(bytes(mem)) == 49152


@pytest.mark.parametrize("with_dump", [False, True])
def test_capstone_linear_requires_explicit_start(tmp_path, monkeypatch, with_dump):
    handle = str(tmp_path / ("with_dump" if with_dump else "without_dump"))
    if with_dump:
        dump = tmp_path / "game.bin"
        dump.write_bytes(bytes(0x10000))
        get_store(handle).ingest_dump(dump)

    def unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("missing start must not fall back to VICE/$0801")

    monkeypatch.setattr(nodes, "_vice_disassemble_fallback", unexpected_fallback)
    state = {
        "kb_handle": handle,
        "plan": [{"id": "s1", "tool": "capstone", "args": {"mode": "linear"}}],
        "current_step_id": "s1",
    }
    result = nodes.capstone_node(state)["tool_results"][0]
    assert result["ok"] is False
    assert result["rejection"] == "missing_arg"
    assert "explicit `start`" in result["data"]


@pytest.mark.parametrize("with_dump", [False, True])
def test_capstone_linear_rejects_invalid_start(tmp_path, monkeypatch, with_dump):
    handle = str(tmp_path / ("invalid_with_dump" if with_dump else "invalid_no_dump"))
    if with_dump:
        dump = tmp_path / "invalid_game.bin"
        dump.write_bytes(bytes(0x10000))
        get_store(handle).ingest_dump(dump)

    def unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("invalid start must not attempt VICE fallback")

    monkeypatch.setattr(nodes, "_vice_disassemble_fallback", unexpected_fallback)
    state = {
        "kb_handle": handle,
        "plan": [{
            "id": "s1", "tool": "capstone",
            "args": {"mode": "linear", "start": "nope"},
        }],
        "current_step_id": "s1",
    }
    result = nodes.capstone_node(state)["tool_results"][0]
    assert result["ok"] is False
    assert result["rejection"] == "invalid_arg"


def test_relevant_labels_and_kb_label_mode_match_hex_address(tmp_path):
    handle = str(tmp_path / "kb")
    store = get_store(handle)
    store.append_event(EVT_LABEL, "test", {
        "addr": 0xC145, "name": "opaque_dispatch", "kind": "code",
        "confidence": 0.4,
    })
    store.append_event(EVT_LABEL, "test", {
        "addr": 0x0780, "name": "unrelated_lives", "kind": "ram_var",
        "confidence": 0.99,
    })
    for offset in range(10):
        store.append_event(EVT_LABEL, "test", {
            "addr": 0x0800 + offset, "name": f"lives_alias_{offset}",
            "kind": "ram_var", "confidence": 0.95,
        })

    assert extract_hex_addresses("drop $0 and 0xF") == []
    assert extract_hex_addresses("keep $C0") == [0xC0]
    assert extract_hex_addresses("keep $00C0 and 0xD020") == [0x00C0, 0xD020]
    assert extract_hex_addresses("what writes $C145 or 0xD020?") == [0xC145, 0xD020]
    rows = store.relevant_labels("what writes $C145?")
    assert [row["addr"] for row in rows] == [0xC145]
    prioritized = store.relevant_labels("where are lives at $C145?", limit=5)
    assert prioritized[0]["addr"] == 0xC145
    assert "$C145  opaque_dispatch" in store.digest_for_question("what writes $C145?")

    state = {
        "kb_handle": handle,
        "plan": [{
            "id": "s1", "tool": "kb",
            "args": {"mode": "labels", "like": "$C145"},
        }],
        "current_step_id": "s1",
    }
    result = nodes.kb_query_node(state)["tool_results"][0]
    assert result["ok"] is True
    assert result["row_count"] == 1
    assert '"addr_hex": "$C145"' in result["data"]


def _vice_state(tmp_path, method: str, **args):
    return {
        "game": "phase4",
        "kb_handle": str(tmp_path / "kb"),
        "plan": [{
            "id": "s1", "tool": "vice",
            "args": {"method": method, **args},
        }],
        "current_step_id": "s1",
    }


def test_screenshot_uses_one_dedicated_vision_call_and_persists_path(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("VICE_MCP_URL", "http://localhost:3000")
    image_bytes = b"\x89PNG\r\n\x1a\nphase-4-test"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_uri = f"data:image/png;base64,{encoded}"
    monkeypatch.setattr(nodes, "_vice_call", lambda method, args: {
        "tool": "vice.display.screenshot", "args": args,
        "url": "http://localhost:3000/mcp",
        "data": {"format": "PNG", "base64": encoded},
    })

    calls = []

    class FakeVision:
        model_name = "vision-test-model"

        def invoke(self, messages):
            calls.append(messages)
            return AIMessage(content="Gameplay screen with SCORE 001200 at the top.")

    monkeypatch.setattr(nodes, "get_llm", lambda role: FakeVision())
    out = nodes.vice_mcp_node(
        _vice_state(tmp_path, "vice.display.screenshot"),
    )
    result = out["tool_results"][0]

    assert result["ok"] is True
    assert len(calls) == 1
    assert isinstance(calls[0][0], SystemMessage)
    assert "neutral visual inspection" in calls[0][0].content
    assert calls[0][1].content[0]["image_url"]["url"] == data_uri
    assert result["vision_role"] == "vision"
    assert result["vision_status"] == "ok"
    assert result["vision_model"] == "vision-test-model"
    assert result["screenshot_path"] == result["thumbnail_path"]
    assert len(out["llm_usage"]) == 1
    assert out["llm_usage"][0]["role"] == "vision"
    assert out["llm_usage"][0]["ok"] is True
    with open(result["screenshot_path"], "rb") as saved:
        assert saved.read() == image_bytes
    assert "SCORE 001200" in result["data"]


def test_screenshot_vision_reservation_denial_skips_provider(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("VICE_MCP_URL", "http://localhost:3000")
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\nbudget-test").decode("ascii")
    monkeypatch.setattr(nodes, "_vice_call", lambda method, args: {
        "tool": "vice.display.screenshot", "args": args,
        "data": {"format": "PNG", "base64": encoded},
    })
    monkeypatch.setattr(
        nodes, "_call_cost_reservation_usd", lambda *_a, **_k: 0.50,
    )
    calls = 0

    class FakeVision:
        model_name = "vision-test-model"
        max_tokens = 2048

        def invoke(self, _messages):
            nonlocal calls
            calls += 1
            raise AssertionError("reservation denial must skip provider")

    monkeypatch.setattr(nodes, "get_llm", lambda _role: FakeVision())
    state = _vice_state(tmp_path, "vice.display.screenshot")
    state["budget_used"] = 2.99
    monkeypatch.setenv("C64RE_USD_BUDGET", "3.0")

    out = nodes.vice_mcp_node(state)
    result = out["tool_results"][0]

    assert calls == 0
    assert result["vision_status"] == "cost_reservation"
    assert "vision description unavailable" in result["data"]
    assert out["budget_reservation_exhausted"] is True
    assert out["llm_usage"][0]["rejection"] == "cost_reservation"


def test_generic_vice_image_flattening_never_calls_vision_without_budget(
    monkeypatch,
):
    calls = 0

    def unexpected_llm(_role):
        nonlocal calls
        calls += 1
        raise AssertionError("generic flattening has no run budget context")

    monkeypatch.setattr(nodes, "get_llm", unexpected_llm)
    nodes.llm_usage.reset()

    text = nodes._vice_text("data:image/png;base64,YQ==")

    assert calls == 0
    assert "vision description unavailable" in text
    assert nodes.llm_usage.drain() == []


def test_screenshot_keeps_artifact_and_failure_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("VICE_MCP_URL", "http://localhost:3000")
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\nfailed-vision").decode("ascii")
    monkeypatch.setattr(nodes, "_vice_call", lambda method, args: {
        "tool": "vice.display.screenshot", "args": args,
        "data": {"format": "PNG", "base64": encoded},
    })

    calls = 0

    class FailedVision:
        model_name = "vision-test-model"

        def invoke(self, _messages):
            nonlocal calls
            calls += 1
            raise RuntimeError("vision endpoint unavailable")

    monkeypatch.setattr(nodes, "get_llm", lambda role: FailedVision())
    result = nodes.vice_mcp_node(
        _vice_state(tmp_path, "vice.display.screenshot"),
    )["tool_results"][0]

    assert calls == 1
    assert result["vision_status"] == "error"
    assert "vision endpoint unavailable" in result["vision_error"]
    assert result["screenshot_path"]
    assert "vision description unavailable" in result["data"]


def test_vice_memory_read_is_hex_dumped_before_truncation(tmp_path, monkeypatch):
    monkeypatch.setenv("VICE_MCP_URL", "http://localhost:3000")
    values = [i & 0xFF for i in range(4096)]
    monkeypatch.setattr(nodes, "_vice_call", lambda method, args: {
        "tool": "vice.memory.read", "args": args,
        "url": "http://localhost:3000/mcp",
        "data": {"address": "$2000", "data": values},
    })
    result = nodes.vice_mcp_node(
        _vice_state(tmp_path, "vice.memory.read", address="$2000", size=4096),
    )["tool_results"][0]

    assert result["memory_format"] == "hex16"
    assert result["memory_bytes_returned"] == 4096
    assert result["memory_bytes_omitted"] > 0
    assert "$2000: 00 01 02 03" in result["data"]
    assert "$2FF0: F0 F1 F2 F3" in result["data"]
    assert "middle byte(s) omitted" in result["data"]
    assert '"data": [' not in result["data"]
    assert len(result["data"]) < 4_500  # deterministic header + ≤4k body


def test_vice_transport_uses_registered_names_and_live_write_schema(monkeypatch):
    from tools import vice_mcp

    calls = []

    def fake_call(tool, args):
        calls.append((tool, args))
        return {"data": "ok", "is_error": False}

    monkeypatch.setattr(vice_mcp, "call_tool", fake_call)
    write = nodes._vice_call(
        "vice.memory.write", {"address": "$00C0", "value": "$07"},
    )
    read = nodes._vice_call(
        "vice.memory.read", {"address": "$00C0", "size": 1},
    )

    assert calls == [
        ("vice_memory_write", {"address": "$00C0", "data": [7]}),
        ("vice_memory_read", {
            "address": "$00C0", "size": 1, "encoding": "array",
        }),
    ]
    # Internal policy/provenance stays canonical even though transport names
    # match the server registration.
    assert write["tool"] == "vice.memory.write"
    assert read["tool"] == "vice.memory.read"


def test_live_shaped_memory_array_uses_hex_strings_not_decimal():
    values, address = nodes._extract_vice_bytes({
        "address": 0x04E9,
        "encoding": "array",
        "data": ["09", "10", "4F", "80", "FF"],
    })
    assert address == 0x04E9
    assert values == [0x09, 0x10, 0x4F, 0x80, 0xFF]


def test_tavily_uses_configurable_domains_and_advanced_depth(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    captured = {}

    class FakeTavilyClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key

        def search(self, **kwargs):
            captured.update(kwargs)
            return {"results": []}

    fake_module = types.ModuleType("tavily")
    fake_module.TavilyClient = FakeTavilyClient
    monkeypatch.setitem(sys.modules, "tavily", fake_module)
    monkeypatch.setattr(nodes, "load_config", lambda: {"tools": {"tavily": {
        "include_domains": ["archive.org", "forum64.de", "c64-wiki.com"],
        "search_depth": "advanced", "max_results": 7,
    }}})
    state = {
        "question": "C64 loader format",
        "plan": [{"id": "s1", "tool": "tavily", "args": {"q": "C64 loader"}}],
        "current_step_id": "s1",
    }
    result = nodes.tavily_node(state)["tool_results"][0]
    assert result["ok"] is True
    assert captured["include_domains"] == [
        "archive.org", "forum64.de", "c64-wiki.com",
    ]
    assert captured["search_depth"] == "advanced"
    assert captured["max_results"] == 7


def test_get_llm_honors_per_role_max_tokens(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = {
        "providers": {"gemini": {"base_url": "http://example", "api_key": "k"}},
        "agents": {
            "analyst": {
                "provider": "gemini", "model": "test-model",
                "temperature": 0.2, "max_tokens": 8192,
            },
        },
        "defaults": {"max_tokens": 4096, "timeout_s": 5, "retries": 0},
    }
    monkeypatch.setattr(llm_factory, "load_config", lambda: config)
    monkeypatch.setattr(llm_factory, "ChatOpenAI", FakeChatOpenAI)
    llm_factory.get_llm.cache_clear()
    try:
        llm_factory.get_llm("analyst")
        assert captured["max_tokens"] == 8192
    finally:
        llm_factory.get_llm.cache_clear()


def test_get_llm_honors_live_output_cap_after_model_floor(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = {
        "providers": {"openai": {"base_url": "http://example", "api_key": "k"}},
        "agents": {
            "executor": {
                "provider": "openai", "model": "gpt-5.5",
                "temperature": 0.0, "max_tokens": 4096,
            },
        },
        "defaults": {"max_tokens": 4096, "timeout_s": 5, "retries": 0},
    }
    monkeypatch.setattr(llm_factory, "load_config", lambda: config)
    monkeypatch.setattr(llm_factory, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("C64RE_MAX_OUTPUT_TOKENS", "1024")
    llm_factory.get_llm.cache_clear()
    try:
        llm_factory.get_llm("executor")
        assert captured["max_tokens"] == 1024
    finally:
        llm_factory.get_llm.cache_clear()


def test_grok45_rejects_none_reasoning_effort_before_transport():
    assert llm_factory._reasoning_effort_grok("grok-4.5", "none") == "low"
    assert llm_factory._reasoning_effort_grok("grok-4.5", "high") == "high"
    assert llm_factory._reasoning_effort_grok("grok-4.3", "none") == "none"


def test_real_llm_config_exposes_phase4_role_budgets(monkeypatch):
    """Smoke the checked-in JSON through the real load_config/get_llm path."""

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(llm_factory, "ChatOpenAI", FakeChatOpenAI)
    llm_factory.load_config.cache_clear()
    llm_factory.get_llm.cache_clear()
    try:
        clients = {
            role: llm_factory.get_llm(role)
            for role in ("planner", "analyst", "critic", "vision")
        }
        assert {
            role: client.kwargs["max_tokens"]
            for role, client in clients.items()
        } == {
            "planner": 8192,
            "analyst": 8192,
            "critic": 8192,
            "vision": 2048,
        }
        assert {
            role: clients[role].kwargs["reasoning_effort"]
            for role in ("analyst", "critic", "vision")
        } == {"analyst": "low", "critic": "low", "vision": "low"}
    finally:
        llm_factory.get_llm.cache_clear()
        llm_factory.load_config.cache_clear()
