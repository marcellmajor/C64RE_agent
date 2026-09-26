"""Frame-equivalent advancement must measure emulation and gate snapshots."""

import pytest

import graph.nodes as nodes
from graph.plan_utils import runnable_steps, step_is_concrete
from tools import vice_execution


class FakeVice:
    def __init__(self, standard="PAL"):
        self.standard = standard
        self.cycles = 1000
        self.calls = []
        self.step_error = False
        self.stalled = False

    def __call__(self, method, args):
        self.calls.append((method, args))
        if method == "vice.machine.config.get":
            return {"data": {"video_standard": self.standard}}
        if method == "vice.cycles.stopwatch":
            assert args == {"action": "read"}
            return {"data": {"cycles": self.cycles}}
        if method == "vice.execution.step":
            assert args["stepOver"] is False
            if self.step_error:
                raise RuntimeError("step interrupted")
            if not self.stalled:
                self.cycles += 3 * args["count"]
        elif method == "vice.memory.read":
            value = 1 if self.cycles > 1000 else 0
            return {"data": {"data": [value] * args["size"]}}
        else:
            assert method == "vice.execution.pause", method
        return {"data": "ok"}


@pytest.mark.parametrize("method", ["vice.execute", "execute", "vice.execution.advance", "vice.execution.run"])
@pytest.mark.parametrize("standard,per_frame", [("PAL", 19656), ("NTSC", 17095)])
def test_100_frame_request_measures_cycles_and_finishes_paused(monkeypatch, method, standard, per_frame):
    fake = FakeVice(standard)
    monkeypatch.setattr(nodes, "_vice_call", fake)
    monkeypatch.setenv("VICE_MCP_URL", "http://unused.invalid")
    state = {"plan": [{"id": "s1", "tool": "vice", "args": {
        "method": method, "frames": 100,
    }}], "current_step_id": "s1"}
    assert step_is_concrete(state["plan"][0])
    result = nodes.vice_mcp_node(state)["tool_results"][0]
    assert result["ok"]
    assert result["method"] == "vice.execution.advance"
    assert result["requested_frames"] == 100
    assert 100 * per_frame <= result["cycles_advanced"] < 100 * per_frame + 3
    assert result["paused"] is True
    assert fake.calls[-1] == ("vice.execution.pause", {})
    assert not any(name in ("vice.execute", "vice.execution.run") for name, _ in fake.calls)
    assert "does not prove" in result["data"]


@pytest.mark.parametrize("frames", [None, False, True, 0, -1, 301, 2.5, "", "NaN"])
def test_invalid_frames_do_not_touch_emulator(frames):
    fake = FakeVice()
    result = vice_execution.advance_frames(fake, frames)
    assert not result["ok"] and result["rejection"] == "invalid_frames"
    assert not step_is_concrete({"tool": "vice", "args": {"method": "vice.execute", "frames": frames}})
    assert fake.calls == []


@pytest.mark.parametrize("method", ["vice.execute", "execute", "vice.execution.advance", "vice.execution.run"])
def test_advance_requires_mutation_approval(monkeypatch, method):
    monkeypatch.setattr(nodes, "_vice_call", lambda *a, **kw: pytest.fail("unapproved mutation"))
    state = {"require_vice_approval": True, "current_step_id": "s1", "plan": [
        {"id": "s1", "tool": "vice", "args": {"method": method, "frames": 100}},
    ]}
    result = nodes.vice_mcp_node(state)["tool_results"][0]
    assert result["rejection"] == "human_approval_required"


@pytest.mark.parametrize("failure", ["stall", "step_error", "unknown_standard", "missing_counter", "timeout", "pause_error"])
def test_failures_do_not_claim_advancement_or_automatically_retry(monkeypatch, failure):
    fake = FakeVice("UNKNOWN" if failure == "unknown_standard" else "PAL")
    fake.stalled = failure == "stall"
    fake.step_error = failure == "step_error"
    if failure == "timeout":
        monkeypatch.setattr(vice_execution, "_ADVANCE_TIMEOUT_S", -1)

    def call(method, args):
        if failure == "missing_counter" and method == "vice.cycles.stopwatch":
            return {"data": {}}
        if failure == "pause_error" and method == "vice.execution.pause" and fake.cycles > 1000:
            raise RuntimeError("pause failed")
        return fake(method, args)

    result = vice_execution.advance_frames(call, 1)
    assert not result["ok"] and result["retryable"] is False
    assert result["rejection"] == "advance_failed"
    assert not any(name == "vice.execution.run" for name, _ in fake.calls)
    if failure not in {"unknown_standard", "pause_error"}:
        assert fake.calls[-1] == ("vice.execution.pause", {})
    if failure == "pause_error":
        assert result["paused"] is False


def test_partial_progress_failure_blocks_dependent_snapshot():
    fake = FakeVice()
    def call(method, args):
        if method == "vice.execution.step" and fake.cycles > 1000:
            raise RuntimeError("connection lost")
        return fake(method, args)
    result = vice_execution.advance_frames(call, 100)
    assert not result["ok"] and result["cycles_advanced"] > 0
    state = {"plan": [
        {"id": "i2_s1", "tool": "vice"},
        {"id": "i2_s2", "tool": "vice", "depends_on": ["i2_s1"]},
    ], "tool_results": [{"step_id": "i2_s1", **result}]}
    assert runnable_steps(state) == []


def test_snapshot_dependency_observes_completed_advance(tmp_path, monkeypatch):
    fake = FakeVice()
    monkeypatch.setattr(nodes, "_vice_call", fake)
    monkeypatch.setenv("VICE_MCP_URL", "http://unused.invalid")
    state = {"kb_handle": str(tmp_path / "kb"), "tool_results": [], "plan": [
        {"id": "i2_s1", "tool": "vice", "args": {"method": "vice.execute", "frames": 100}},
        {"id": "i2_s2", "tool": "vice", "depends_on": ["i2_s1"],
         "args": {"method": "vice.memory.snapshot", "name": "amoeba_t2"}},
    ]}
    assert [step["id"] for step in runnable_steps(state)] == ["i2_s1"]
    advance = nodes.vice_mcp_node({**state, "current_step_id": "i2_s1"})
    state["tool_results"] = advance["tool_results"]
    assert [step["id"] for step in runnable_steps(state)] == ["i2_s2"]
    snapshot = nodes.vice_mcp_node({**state, "current_step_id": "i2_s2"})["tool_results"][0]
    assert snapshot["ok"]
    assert (tmp_path / "snapshots" / "amoeba_t2.bin").read_bytes() == bytes([1]) * 65536
    methods = [name for name, _ in fake.calls]
    last_step = len(methods) - 1 - methods[::-1].index("vice.execution.step")
    assert methods.index("vice.memory.read") > last_step
