"""VICE composites (tracker 3.1 / 3.2 / 3.8) — snapshot/diff/scan/trace/
poke verification and the registers-after-checkpoint un-ban. `_vice_call` is mocked,
so no emulator is needed; snapshots live under tmp_path."""

import pytest
from langchain_core.messages import AIMessage

import graph.nodes as nodes
from memory import get_store


@pytest.fixture()
def kb_state(tmp_path, monkeypatch):
    monkeypatch.setenv("VICE_MCP_URL", "http://localhost:3000")
    handle = str(tmp_path / "kb")
    store = get_store(handle)
    # Seed the ingested dump so the reserved 'dump' snapshot works.
    dump = bytearray(0x10000)
    dump[0x00C0] = 5
    dump_path = tmp_path / "game.bin"
    dump_path.write_bytes(bytes(dump))
    store.ingest_dump(dump_path)
    return {"kb_handle": handle}, tmp_path


def _step(state, args):
    return {**state, "plan": [{"id": "s1", "tool": "vice", "args": args}],
            "current_step_id": "s1"}


def _run(state, args):
    return nodes.vice_mcp_node(_step(state, args))


# ---------------------------------------------------------------------------
# 3.1 snapshot / diff / monotonic_scan
# ---------------------------------------------------------------------------

def test_snapshot_then_diff_against_dump(kb_state, monkeypatch):
    state, tmp = kb_state
    live = bytearray(0x10000)
    live[0x00C0] = 4                       # lives dropped 5 → 4 vs dump
    live[0xD012] = 99                       # I/O noise
    monkeypatch.setattr(
        nodes, "_vice_read_full_ram", lambda *a, **k: bytes(live),
    )

    snap = _run(state, {"method": "vice.memory.snapshot", "name": "after_death"})
    assert snap["tool_results"][0]["ok"]
    # Assert the .bin actually exists at the path _snapshot_dir derives
    # from kb_handle (sessions/<slug>/snapshots/), not a truthy Path.
    snap_path = nodes._snapshot_dir(state) / "after_death.bin"
    assert snap_path.exists(), f"snapshot .bin missing at {snap_path}"
    assert snap_path.stat().st_size == 0x10000

    out = _run(state, {"method": "vice.memory.diff",
                       "a": "dump", "b": "after_death"})
    r = out["tool_results"][0]
    assert r["ok"]
    changed = {c["addr"] for c in r["changed"]}
    assert 0x00C0 in changed               # state change surfaced
    assert 0xD012 not in changed           # I/O excluded


def test_snapshot_writes_bin_to_derived_path(kb_state, monkeypatch):
    """Regression: the snapshot must actually land on disk at the path
    derived from kb_handle (a truthy-Path assertion missed this)."""
    state, _ = kb_state
    monkeypatch.setattr(
        nodes, "_vice_read_full_ram", lambda *a, **k: bytes(0x10000),
    )
    snap_dir = nodes._snapshot_dir(state)
    target = snap_dir / "probe.bin"
    assert not target.exists()
    r = _run(state, {"method": "vice.memory.snapshot", "name": "probe"})
    assert r["tool_results"][0]["ok"]
    assert target.exists() and target.stat().st_size == 0x10000


def test_diff_against_live_when_b_omitted(kb_state, monkeypatch):
    state, _ = kb_state
    live = bytearray(0x10000)
    live[0x00C0] = 4
    monkeypatch.setattr(
        nodes, "_vice_read_full_ram", lambda *a, **k: bytes(live),
    )
    out = _run(state, {"method": "vice.memory.diff", "a": "dump"})
    r = out["tool_results"][0]
    assert r["ok"] and any(c["addr"] == 0x00C0 for c in r["changed"])


def test_monotonic_scan_over_saved_snapshots(kb_state, monkeypatch):
    state, _ = kb_state
    for i in range(3):
        buf = bytearray(0x10000)
        buf[0x00C0] = 5 - i                # 5,4,3
        monkeypatch.setattr(
            nodes, "_vice_read_full_ram", lambda *a, b=bytes(buf), **k: b,
        )
        _run(state, {"method": "vice.memory.snapshot", "name": f"s{i}"})

    out = _run(state, {"method": "vice.memory.monotonic_scan",
                       "snapshots": ["s0", "s1", "s2"], "delta": -1})
    r = out["tool_results"][0]
    assert r["ok"]
    assert any(c["addr"] == 0x00C0 for c in r["candidates"])


def test_diff_missing_snapshot_is_clear_error(kb_state, monkeypatch):
    state, _ = kb_state
    # A named 'b' that doesn't exist must be reported clearly (no live
    # fallback, since 'b' was explicitly requested).
    out = _run(state, {"method": "vice.memory.diff", "a": "dump", "b": "nope"})
    r = out["tool_results"][0]
    assert not r["ok"] and "nope" in r["data"]


# ---------------------------------------------------------------------------
# 3.2 vice.trace
# ---------------------------------------------------------------------------

def test_trace_resolves_writer_and_cleans_up(kb_state, monkeypatch):
    state, _ = kb_state
    calls = []

    def fake_call(method, args):
        calls.append(method)
        if method == "vice.checkpoint.add":
            return {"data": {"number": 7}}
        if method == "vice.registers.get":
            return {"data": "A:00 X:01 Y:02 SP:F8 PC:$C143 NV-BDIZC"}
        if method == "vice.disassemble":
            return {"data": "$C140  EE C0 00  INC $00C0"}
        return {"data": "ok"}

    monkeypatch.setattr(nodes, "_vice_call", fake_call)
    out = _run(state, {"method": "vice.trace", "address": "$00C0", "frames": 3})
    r = out["tool_results"][0]
    assert r["ok"]                                   # writer proven
    assert r["writer_pc"] == "$C143"
    assert r["writer_proven"] is True
    assert "vice.checkpoint.add" in calls
    assert "vice.checkpoint.delete" in calls         # cleanup ran
    assert "INC $00C0" in r["data"]


def test_trace_confirms_hit_via_checkpoint_list(kb_state, monkeypatch):
    """A hit reported by checkpoint.list is sufficient even if the PC's
    disasm doesn't literally name the address (e.g. indexed write)."""
    state, _ = kb_state

    def fake_call(method, args):
        if method == "vice.checkpoint.add":
            return {"data": {"number": 3}}
        if method == "vice.checkpoint.list":
            return {"data": {"checkpoints": [{"number": 3, "hit_count": 1}]}}
        if method == "vice.registers.get":
            return {"data": "PC:$C200"}
        if method == "vice.disassemble":
            return {"data": "$C1F8  9D 00 C0  STA $C000,X"}   # indexed
        return {"data": "ok"}

    monkeypatch.setattr(nodes, "_vice_call", fake_call)
    out = _run(state, {"method": "vice.trace", "address": "$00C0"})
    r = out["tool_results"][0]
    assert r["ok"] and r["hit_confirmed"] is True


def test_trace_no_checkpoint_id_ignores_unrelated_hit(kb_state, monkeypatch):
    """If checkpoint.add returns no id, a hit_count on some OTHER
    breakpoint must NOT be mis-claimed as our watchpoint's hit — the run
    must fall back to PC/disasm proof (review nit)."""
    state, _ = kb_state
    calls = []

    def fake_call(method, args):
        calls.append(method)
        if method == "vice.checkpoint.add":
            return {"data": "watchpoint set"}          # no id parseable
        if method == "vice.checkpoint.list":
            # An unrelated breakpoint (#5) has hits; ours is unknown.
            return {"data": {"checkpoints": [{"number": 5, "hit_count": 3}]}}
        if method == "vice.registers.get":
            return {"data": "PC:$0800"}                 # unrelated PC
        if method == "vice.disassemble":
            return {"data": "$07F0  EA  NOP"}           # no write to $00C0
        return {"data": "ok"}

    monkeypatch.setattr(nodes, "_vice_call", fake_call)
    out = _run(state, {"method": "vice.trace", "address": "$00C0"})
    r = out["tool_results"][0]
    assert not r["ok"]                                   # not a false hit
    assert r["hit_confirmed"] is False
    assert "vice.checkpoint.delete" in calls            # cleanup by address


def test_trace_no_hit_is_failure_but_cleans_up(kb_state, monkeypatch):
    state, _ = kb_state
    calls = []

    def fake_call(method, args):
        calls.append(method)
        if method == "vice.checkpoint.add":
            return {"data": {"number": 9}}
        if method == "vice.checkpoint.list":
            return {"data": {"checkpoints": [{"number": 9, "hit_count": 0}]}}
        if method == "vice.registers.get":
            return {"data": "A:00 X:00 PC:$0800"}       # unrelated PC
        if method == "vice.disassemble":
            return {"data": "$07F0  EA  NOP"}           # no write to $00C0
        return {"data": "ok"}

    monkeypatch.setattr(nodes, "_vice_call", fake_call)
    out = _run(state, {"method": "vice.trace", "address": "$00C0"})
    r = out["tool_results"][0]
    assert not r["ok"]                                # no confirmed write
    assert r["writer_pc"] is None
    assert "No confirmed write" in r["data"]
    assert "vice.checkpoint.delete" in calls          # cleanup still ran


def test_trace_cleans_up_even_when_run_raises(kb_state, monkeypatch):
    state, _ = kb_state
    calls = []

    def fake_call(method, args):
        calls.append(method)
        if method == "vice.checkpoint.add":
            return {"data": {"number": 1}}
        if method == "vice.execution.run":
            raise RuntimeError("emulator wedged")
        if method == "vice.registers.get":
            return {"data": "PC:$C143"}
        if method == "vice.disassemble":
            return {"data": "$C140  EE C0 00  INC $00C0"}
        return {"data": "ok"}

    monkeypatch.setattr(nodes, "_vice_call", fake_call)
    out = _run(state, {"method": "vice.trace", "address": "$00C0"})
    # Run raised but the writer is still proven from PC/disasm.
    assert out["tool_results"][0]["ok"]
    assert "vice.checkpoint.delete" in calls          # try/finally cleanup


def test_trace_requires_address(kb_state):
    state, _ = kb_state
    out = _run(state, {"method": "vice.trace"})
    assert not out["tool_results"][0]["ok"]


# ---------------------------------------------------------------------------
# registers.get ban — option (c): always banned standalone (tracker 3.2)
# ---------------------------------------------------------------------------

def test_registers_get_banned_standalone(kb_state):
    state, _ = kb_state
    out = _run(state, {"method": "vice.registers.get"})
    r = out["tool_results"][0]
    assert not r["ok"] and r["rejection"] == "banned_method"


@pytest.mark.parametrize(
    "alias",
    ["ping", "registers", "registers_get", "registers.get",
     "get_registers", "vice.registers"],
)
def test_banned_vice_aliases_are_normalized_before_policy_check(
    kb_state, monkeypatch, alias,
):
    """Raw aliases must not bypass the canonical standalone-call ban."""
    state, _ = kb_state

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("banned aliases must not invoke _vice_call")

    monkeypatch.setattr(nodes, "_vice_call", unexpected_call)
    r = _run(state, {"method": alias})["tool_results"][0]
    assert r["ok"] is False
    assert r["rejection"] == "banned_method"
    assert r["method"] in {"vice.ping", "vice.registers.get"}


def test_registers_get_armed_flag_does_not_bypass_ban(kb_state):
    """A planner-supplied `armed: true` is not trustworthy proof of a
    hit and must NOT bypass the ban (hardening — option c)."""
    state, _ = kb_state
    out = _run(state, {"method": "vice.registers.get", "armed": True})
    r = out["tool_results"][0]
    assert not r["ok"] and r["rejection"] == "banned_method"
    assert "vice.trace" in r["data"]                  # points to the right verb


# ---------------------------------------------------------------------------
# 3.8 restore-on-exit multimodal poke verification
# ---------------------------------------------------------------------------

def test_poke_verify_captures_pair_and_restores_original(kb_state, monkeypatch):
    state, _ = kb_state
    image = "data:image/png;base64,YQ=="
    writes = []
    screenshots = 0
    current = 5

    def fake_call(method, args):
        nonlocal screenshots, current
        if method == "vice.memory.read":
            return {"data": {"address": "$00C0", "data": [current]}}
        if method == "vice.memory.write":
            writes.append(dict(args))
            current = args["value"]
            return {"data": "ok"}
        if method == "vice.display.screenshot":
            screenshots += 1
            return {"data": image}
        if method == "vice.execution.run":
            return {"data": "ok"}
        raise AssertionError(method)

    monkeypatch.setattr(nodes, "_vice_call", fake_call)
    monkeypatch.setattr(
        nodes,
        "_compare_screenshot_pair",
        lambda *_a, **_k: (
            "[comparison]\nvisible score changed",
            {"vision_ok": True},
        ),
    )
    result = _run(state, {
        "method": "poke_and_peek",
        "address": "$00C0",
        "value": 9,
        "expect": "score digit changes",
        "frames": 1,
    })["tool_results"][0]

    assert result["ok"] is True
    assert result["restored"] is True
    assert result["original_value"] == 5
    assert writes == [
        {"address": "$00C0", "value": 9},
        {"address": "$00C0", "value": 5},
    ]
    assert screenshots == 2
    assert result["before_screenshot_path"]
    assert result["after_screenshot_path"]


def test_poke_verify_restores_when_after_capture_fails(kb_state, monkeypatch):
    state, _ = kb_state
    writes = []
    screenshot_count = 0
    current = 4

    def fake_call(method, args):
        nonlocal screenshot_count, current
        if method == "vice.memory.read":
            return {"data": [current]}
        if method == "vice.memory.write":
            writes.append(args["value"])
            current = args["value"]
            return {"data": "ok"}
        if method == "vice.display.screenshot":
            screenshot_count += 1
            if screenshot_count == 2:
                raise RuntimeError("capture failed")
            return {"data": "data:image/png;base64,YQ=="}
        raise AssertionError(method)

    monkeypatch.setattr(nodes, "_vice_call", fake_call)
    result = _run(state, {
        "method": "vice.poke_verify",
        "address": "$00C0",
        "value": 7,
        "expect": "lives digit changes",
    })["tool_results"][0]

    assert result["ok"] is False
    assert result["restored"] is True
    assert writes == [7, 4]
    assert "original byte was restored" in result["data"]


def test_poke_verify_uses_selected_bank_for_read_write_and_restore(
    kb_state, monkeypatch,
):
    state, _ = kb_state
    calls = []
    current = 4

    def fake_call(method, args):
        nonlocal current
        calls.append((method, dict(args)))
        if method == "vice.snapshot.save":
            raise RuntimeError("snapshot unavailable")
        if method == "vice.memory.read":
            return {"data": [current]}
        if method == "vice.memory.write":
            current = args["value"]
            return {"data": "ok"}
        if method == "vice.display.screenshot":
            return {"data": "data:image/png;base64,YQ=="}
        raise AssertionError(method)

    monkeypatch.setattr(nodes, "_vice_call", fake_call)
    monkeypatch.setattr(
        nodes,
        "_compare_screenshot_pair",
        lambda *_a, **_k: ("supported", {"vision_ok": True}),
    )

    result = _run(state, {
        "method": "vice.poke_verify",
        "address": "$00C0",
        "bank": "ram",
        "value": 7,
        "expect": "lives digit changes",
    })["tool_results"][0]

    assert result["ok"] is True
    memory_calls = [
        (method, args) for method, args in calls
        if method in {"vice.memory.read", "vice.memory.write"}
    ]
    assert memory_calls == [
        ("vice.memory.read", {"address": "$00C0", "size": 1, "bank": "ram"}),
        ("vice.memory.write", {"address": "$00C0", "value": 7, "bank": "ram"}),
        ("vice.memory.read", {"address": "$00C0", "size": 1, "bank": "ram"}),
        ("vice.memory.write", {"address": "$00C0", "value": 4, "bank": "ram"}),
    ]


def test_poke_verify_prefers_full_snapshot_restore(kb_state, monkeypatch):
    state, _ = kb_state
    calls = []
    current = 4
    snap_value = None

    def fake_call(method, args):
        nonlocal current, snap_value
        calls.append((method, dict(args)))
        if method == "vice.memory.read":
            return {"data": [current]}
        if method == "vice.display.screenshot":
            return {"data": "data:image/png;base64,YQ=="}
        if method == "vice.memory.write":
            current = args["value"]
            return {"data": "ok"}
        if method == "vice.snapshot.save":
            snap_value = current
            return {"data": "ok"}
        if method == "vice.snapshot.load":
            current = snap_value
            return {"data": "ok"}
        raise AssertionError(method)

    monkeypatch.setattr(nodes, "_vice_call", fake_call)
    monkeypatch.setattr(
        nodes,
        "_compare_screenshot_pair",
        lambda *_a, **_k: ("supported", {"vision_ok": True}),
    )
    result = _run(state, {
        "method": "vice.poke_verify",
        "address": "$00C0",
        "value": 7,
        "expect": "lives digit changes",
    })["tool_results"][0]

    assert result["ok"] is True
    assert result["restore_method"] == "full_snapshot"
    assert result["safety_snapshot"].startswith("c64re_poke_")
    writes = [args for method, args in calls if method == "vice.memory.write"]
    assert writes == [{"address": "$00C0", "value": 7}]
    methods = [method for method, _ in calls]
    assert methods.index("vice.snapshot.save") < methods.index(
        "vice.memory.write",
    )
    assert methods[-1] == "vice.snapshot.load"


def test_poke_verify_fails_if_live_write_does_not_read_back(
    kb_state, monkeypatch,
):
    state, _ = kb_state
    screenshots = 0

    def fake_call(method, args):
        nonlocal screenshots
        if method == "vice.memory.read":
            return {"data": [4]}  # write is silently ignored
        if method == "vice.display.screenshot":
            screenshots += 1
            return {"data": "data:image/png;base64,YQ=="}
        if method in {
            "vice.memory.write", "vice.snapshot.save", "vice.snapshot.load",
        }:
            return {"data": "ok"}
        raise AssertionError(method)

    monkeypatch.setattr(nodes, "_vice_call", fake_call)
    result = _run(state, {
        "method": "vice.poke_verify",
        "address": "$00C0",
        "value": 7,
        "expect": "lives digit changes",
    })["tool_results"][0]

    assert result["ok"] is False
    assert result["write_verified"] is False
    assert result["restored"] is True
    assert result["restore_method"] == "full_snapshot"
    assert "read-back mismatch" in result["data"]
    assert screenshots == 1  # no misleading after image was captured


def test_poke_verify_requires_complete_experiment_args(kb_state, monkeypatch):
    state, _ = kb_state
    monkeypatch.setattr(
        nodes,
        "_vice_call",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("invalid experiment must not touch VICE"),
        ),
    )
    result = _run(state, {
        "method": "vice.poke_verify", "address": "$00C0", "value": 1,
    })["tool_results"][0]
    assert result["ok"] is False
    assert result["rejection"] == "missing_arg"


def test_poke_verify_vision_compares_two_labelled_images(monkeypatch):
    calls = []

    class FakeVision:
        model_name = "fake-vision"

        def invoke(self, messages):
            calls.append(messages)
            return AIMessage(
                content="supported: the score changed from 10 to 20",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 8,
                    "total_tokens": 18,
                },
            )

    monkeypatch.setattr(nodes, "get_llm", lambda role: FakeVision())
    nodes.llm_usage.reset()
    before = "data:image/png;base64,YmVmb3Jl"
    after = "data:image/png;base64,YWZ0ZXI="
    text, meta = nodes._compare_screenshot_pair(
        before, after, expectation="score changes",
    )
    usage = nodes.llm_usage.drain()

    content = calls[0][1].content
    assert [item["type"] for item in content] == [
        "text", "image_url", "text", "image_url", "text",
    ]
    assert content[1]["image_url"]["url"] == before
    assert content[3]["image_url"]["url"] == after
    assert "supported" in text
    assert meta["vision_ok"] is True
    assert len(calls) == 1
    assert usage[0]["input_tokens"] == 10


def test_poke_verify_vision_reservation_denial_skips_provider(monkeypatch):
    calls = 0

    class FakeVision:
        model_name = "priced-vision"
        max_tokens = 2048

        def invoke(self, _messages):
            nonlocal calls
            calls += 1
            raise AssertionError("reservation denial must skip provider")

    monkeypatch.setattr(nodes, "get_llm", lambda _role: FakeVision())
    monkeypatch.setattr(
        nodes, "_call_cost_reservation_usd", lambda *_a, **_k: 0.25,
    )
    nodes.llm_usage.reset()

    text, meta = nodes._compare_screenshot_pair(
        "data:image/png;base64,YmVmb3Jl",
        "data:image/png;base64,YWZ0ZXI=",
        expectation="score changes",
        remaining_budget_usd=0.01,
    )
    usage = nodes.llm_usage.drain()

    assert calls == 0
    assert text == "[before/after vision comparison unavailable]"
    assert meta["vision_status"] == "cost_reservation"
    assert usage[0]["rejection"] == "cost_reservation"
