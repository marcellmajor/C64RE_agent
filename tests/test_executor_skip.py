"""Executor LLM bypass for concrete steps (tracker 1.1)."""

import pytest

import graph.nodes as nodes
from graph.plan_utils import step_is_concrete


# ---------------------------------------------------------------------------
# step_is_concrete — unit table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step,expected", [
    # kb: metadata modes need nothing; search modes need a query.
    ({"tool": "kb", "args": {"mode": "stats"}}, True),
    ({"tool": "kb", "args": {"mode": "labels", "like": "lives"}}, True),
    ({"tool": "kb", "args": {"mode": "text", "q": "score"}}, True),
    ({"tool": "kb", "args": {"mode": "text"}}, False),
    ({"tool": "kb", "args": {"mode": "sql", "sql": "SELECT 1"}}, True),
    ({"tool": "kb", "args": {"mode": "sql"}}, False),
    # capstone: linear needs a start; the rest self-default.
    ({"tool": "capstone", "args": {"mode": "linear", "start": "$C000",
                                   "length": 256}}, True),
    ({"tool": "capstone", "args": {"mode": "linear"}}, False),
    ({"tool": "capstone", "args": {"mode": "find_loops", "top_n": 8}}, True),
    ({"tool": "capstone", "args": {"mode": "recursive",
                                   "entry": "0x0801"}}, True),
    # tavily needs a query.
    ({"tool": "tavily", "args": {"q": "boulder dash memory map"}}, True),
    ({"tool": "tavily", "args": {}}, False),
    # vice: address-bearing methods need an address (any alias).
    ({"tool": "vice", "args": {"method": "vice.disassemble",
                               "address": "$1135", "count": 16}}, True),
    ({"tool": "vice", "args": {"method": "vice.disassemble"}}, False),
    ({"tool": "vice", "args": {"method": "memory.read", "addr": "$03F0",
                               "size": 16}}, True),
    ({"tool": "vice", "args": {"method": "memory.read",
                               "addr": "$03F0"}}, False),
    ({"tool": "vice", "args": {"method": "vice.display.screenshot"}}, True),
    ({"tool": "vice", "args": {}}, False),
    # Null-valued args always mean "executor, fill this in".
    ({"tool": "capstone", "args": {"mode": "linear", "start": None,
                                   "length": 256}}, False),
    ({"tool": "kb", "args": {"mode": "text", "q": "  "}}, False),
    # vice checkpoint/breakpoint methods are address-bearing (review
    # finding 4 — used to be blanket-True); execution methods are not.
    ({"tool": "vice", "args": {"method": "checkpoint_add",
                               "address": "$C05D"}}, True),
    ({"tool": "vice", "args": {"method": "checkpoint_add"}}, False),
    ({"tool": "vice", "args": {"method": "vice.checkpoint.add"}}, False),
    ({"tool": "vice", "args": {"method": "vice.execution.step"}}, True),
    ({"tool": "vice", "args": {"method": "vice.execution.run"}}, True),
    ({"tool": "vice", "args": {"method": "vice.mystery.mode"}}, False),
    # code_kb per-mode requirements (review finding 4 — "has a mode"
    # was not enough).
    ({"tool": "code_kb", "args": {"mode": "stats"}}, True),
    ({"tool": "code_kb", "args": {"mode": "schema"}}, True),
    ({"tool": "code_kb", "args": {"mode": "routines"}}, True),
    ({"tool": "code_kb", "args": {"mode": "smc"}}, True),
    # routine/annotate handlers read start|addr|address ONLY (final
    # review: `dst`/`src` were accepted globally but ignored by handlers).
    ({"tool": "code_kb", "args": {"mode": "routine",
                                  "start": "$C000"}}, True),
    ({"tool": "code_kb", "args": {"mode": "routine",
                                  "address": "$C000"}}, True),
    ({"tool": "code_kb", "args": {"mode": "routine", "dst": "$C000"}}, False),
    ({"tool": "code_kb", "args": {"mode": "routine", "src": "$C000"}}, False),
    ({"tool": "code_kb", "args": {"mode": "routine"}}, False),
    ({"tool": "code_kb", "args": {"mode": "annotate",
                                  "start": "$C000"}}, True),
    ({"tool": "code_kb", "args": {"mode": "annotate",
                                  "addr": "$C000"}}, True),
    ({"tool": "code_kb", "args": {"mode": "annotate", "dst": "$C000"}}, False),
    ({"tool": "code_kb", "args": {"mode": "annotate"}}, False),
    # disasm: capstone/default engine reads start|addr; vice engine
    # reads address|addr.
    ({"tool": "code_kb", "args": {"mode": "disasm", "addr": "$C000"}}, True),
    ({"tool": "code_kb", "args": {"mode": "disasm", "start": "$C000"}}, True),
    ({"tool": "code_kb", "args": {"mode": "disasm",
                                  "address": "$C000"}}, False),
    ({"tool": "code_kb", "args": {"mode": "disasm", "engine": "vice",
                                  "address": "$C000"}}, True),
    ({"tool": "code_kb", "args": {"mode": "disasm", "engine": "vice",
                                  "start": "$C000"}}, False),
    ({"tool": "code_kb", "args": {"mode": "disasm"}}, False),
    # xrefs_to reads addr|dst; xrefs_from reads addr|src — the exact
    # false positives the final review reproduced.
    ({"tool": "code_kb", "args": {"mode": "xrefs_to", "addr": "$D012"}}, True),
    ({"tool": "code_kb", "args": {"mode": "xrefs_to", "dst": "$D012"}}, True),
    ({"tool": "code_kb", "args": {"mode": "xrefs_to", "src": "$D012"}}, False),
    ({"tool": "code_kb", "args": {"mode": "xrefs_to"}}, False),
    ({"tool": "code_kb", "args": {"mode": "xrefs_from", "src": "$C000"}}, True),
    ({"tool": "code_kb", "args": {"mode": "xrefs_from",
                                  "addr": "$C000"}}, True),
    ({"tool": "code_kb", "args": {"mode": "xrefs_from",
                                  "dst": "$C000"}}, False),
    ({"tool": "code_kb", "args": {"mode": "xrefs_from"}}, False),
    ({"tool": "code_kb", "args": {"mode": "search", "q": "raster"}}, True),
    ({"tool": "code_kb", "args": {"mode": "search"}}, False),
    ({"tool": "code_kb", "args": {"mode": "sql", "sql": "SELECT 1"}}, True),
    ({"tool": "code_kb", "args": {"mode": "sql"}}, False),
    ({"tool": "code_kb", "args": {"mode": "made_up_mode"}}, False),
    ({"tool": "code_kb", "args": {}}, False),
    # Phase 3 code_kb data-ref modes (tracker 3.4).
    ({"tool": "code_kb", "args": {"mode": "writes_to", "addr": "$D012"}}, True),
    ({"tool": "code_kb", "args": {"mode": "writes_to", "dst": "$D012"}}, True),
    ({"tool": "code_kb", "args": {"mode": "writes_to"}}, False),
    ({"tool": "code_kb", "args": {"mode": "refs_to", "addr": "$0780"}}, True),
    ({"tool": "code_kb", "args": {"mode": "refs_to"}}, False),
    ({"tool": "code_kb", "args": {"mode": "hardware_refs"}}, True),  # self-defaults
    ({"tool": "code_kb", "args": {"mode": "hardware_refs", "chip": "sid"}}, True),
    # Phase 3 capstone modes — all self-default (recursive-disasm inside).
    ({"tool": "capstone", "args": {"mode": "find_counters"}}, True),
    ({"tool": "capstone", "args": {"mode": "find_counters",
                                   "kind": "lives"}}, True),
    ({"tool": "capstone", "args": {"mode": "idioms"}}, True),
    ({"tool": "capstone", "args": {"mode": "screen_text"}}, True),
    ({"tool": "capstone", "args": {"mode": "bank"}}, True),
    # Phase 3 VICE composites (tracker 3.1 / 3.2).
    ({"tool": "vice", "args": {"method": "vice.memory.snapshot",
                               "name": "s0"}}, True),
    ({"tool": "vice", "args": {"method": "vice.memory.snapshot"}}, False),
    ({"tool": "vice", "args": {"method": "vice.memory.diff", "a": "dump"}}, True),
    ({"tool": "vice", "args": {"method": "vice.memory.diff"}}, True),  # a defaults
    ({"tool": "vice", "args": {"method": "vice.memory.monotonic_scan",
                               "snapshots": ["s0", "s1"]}}, True),
    ({"tool": "vice", "args": {"method": "vice.memory.monotonic_scan"}}, False),
    ({"tool": "vice", "args": {"method": "vice.trace", "address": "$00C0"}}, True),
    ({"tool": "vice", "args": {"method": "vice.trace"}}, False),
    ({"tool": "vice", "args": {"method": "vice.poke_verify",
                                  "address": "$00C0", "value": 7,
                                  "expect": "lives digit changes"}}, True),
    ({"tool": "vice", "args": {"method": "vice.poke_verify",
                                  "address": "$00C0", "value": 7}}, False),
    ({"tool": "wat", "args": {"x": 1}}, False),
])
def test_step_is_concrete(step, expected):
    assert step_is_concrete(step) is expected


# ---------------------------------------------------------------------------
# executor_node integration
# ---------------------------------------------------------------------------

def test_concrete_step_skips_the_llm(monkeypatch):
    def _boom(*a, **kw):  # pragma: no cover
        raise AssertionError("executor called the LLM for a concrete step")
    monkeypatch.setattr(nodes, "_safe_invoke", _boom)
    monkeypatch.setattr(nodes, "_kb_digest_for_state", _boom)

    state = {
        "plan": [{"id": "s1", "tool": "kb",
                  "args": {"mode": "stats"}, "depends_on": []}],
        "tool_results": [],
    }
    out = nodes.executor_node(state)
    assert out["current_step_id"] == "s1"
    assert "LLM skipped" in out["messages"][0].content


def test_null_arg_step_uses_the_llm(monkeypatch):
    calls = []

    def _fake(role, prompt, fallback, **kw):
        calls.append(role)
        return {"step_id": "s1", "tool": "capstone",
                "args": {"mode": "linear", "start": "$C000", "length": 128},
                "rationale": "filled from KB"}

    monkeypatch.setattr(nodes, "_safe_invoke", _fake)
    state = {
        "plan": [{"id": "s1", "tool": "capstone",
                  "args": {"mode": "linear", "start": None}, "depends_on": []}],
        "tool_results": [],
        "kb_digest": "(digest)",
    }
    out = nodes.executor_node(state)
    assert calls == ["executor"]
    assert out["current_step_id"] == "s1"
    merged = next(s for s in out["plan"] if s["id"] == "s1")["args"]
    assert merged["start"] == "$C000"


def test_retry_always_uses_the_llm(monkeypatch):
    """A previously-failed step goes through enrichment so the LLM can
    repair the args using the recorded error — even if args look complete."""
    calls = []

    def _fake(role, prompt, fallback, **kw):
        calls.append(prompt)
        return dict(fallback)

    monkeypatch.setattr(nodes, "_safe_invoke", _fake)
    state = {
        "plan": [{"id": "s1", "tool": "kb",
                  "args": {"mode": "stats"}, "depends_on": []}],
        "tool_results": [{"step_id": "s1", "tool": "kb", "ok": False,
                          "data": "transient sqlite lock"}],
        "kb_digest": "(digest)",
    }
    out = nodes.executor_node(state)
    assert len(calls) == 1
    assert "already FAILED 1 time(s)" in calls[0]
    assert out["current_step_id"] == "s1"


@pytest.mark.parametrize("address_key", ["address", "addr"])
@pytest.mark.parametrize("replacement,expected", [
    ("", "$8700"),
    ("   ", "$8700"),
    ("unknown", "$8700"),
    (False, "$8700"),
    ({"address": "$8800"}, "$8700"),
    (None, "$8700"),
    ("$8800", "$8800"),
    (0, "$0000"),
])
def test_trace_retry_preserves_usable_address(
    monkeypatch, address_key, replacement, expected,
):
    """An inconclusive trace must not become a malformed call on retry."""
    monkeypatch.setenv("VICE_MCP_URL", "http://unused.invalid")
    monkeypatch.setattr(nodes, "_safe_invoke", lambda *a, **kw: {
        "args": {address_key: replacement, "frames": 40},
        "rationale": "Retry the trace for longer",
    })
    calls = []

    def fake_vice_call(method, args):
        calls.append((method, args))
        if method == "vice.ping":
            return {"data": {"execution": "paused"}}
        if method == "vice.checkpoint.add":
            return {"data": {"checkpoint_num": 1}}
        if method == "vice.checkpoint.list":
            return {"data": {"checkpoints": [
                {"checkpoint_num": 1, "hit_count": 1},
            ]}}
        return {"data": "ok"}

    monkeypatch.setattr(nodes, "_vice_call", fake_vice_call)
    state = {
        "plan": [{"id": "i1_s11", "tool": "vice", "args": {
            "method": "vice.trace", address_key: "$8700", "frames": 2,
        }}],
        "tool_results": [{
            "step_id": "i1_s11", "tool": "vice", "ok": False,
            "data": "No confirmed write to $8700 within 1.00s",
        }],
        "kb_digest": "(digest)",
    }
    update = nodes.executor_node(state)
    assert update["plan"][0]["args"]["frames"] == 40
    assert state["plan"][0]["args"][address_key] == "$8700"
    result = nodes.vice_mcp_node({**state, **update})["tool_results"][0]
    assert result["ok"] is True
    assert result["address"] == expected
    arm_args = next(args for method, args in calls
                    if method == "vice.checkpoint.add")
    assert arm_args["start"] == expected
    assert ("vice.checkpoint.delete", {"checkpoint_num": 1}) in calls
