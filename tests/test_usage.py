"""LLM usage / budget accounting (tracker 1.4)."""

from types import SimpleNamespace

import pytest

import graph.nodes as nodes
from graph import usage


@pytest.fixture(autouse=True)
def _clean_collector():
    usage.drain()
    yield
    usage.drain()


# ---------------------------------------------------------------------------
# extraction + cost estimation
# ---------------------------------------------------------------------------

def test_extract_usage_prefers_usage_metadata():
    msg = SimpleNamespace(
        usage_metadata={"input_tokens": 120, "output_tokens": 30},
        response_metadata={"token_usage": {"prompt_tokens": 999}},
    )
    assert usage.extract_usage(msg) == (120, 30)


def test_extract_usage_falls_back_to_token_usage():
    msg = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"token_usage": {"prompt_tokens": 11,
                                           "completion_tokens": 7}},
    )
    assert usage.extract_usage(msg) == (11, 7)
    assert usage.extract_usage(SimpleNamespace()) == (0, 0)


def test_estimate_cost_longest_prefix_wins():
    pricing = {
        "gpt-5": {"input_per_mtok": 2.0, "output_per_mtok": 8.0},
        "gpt-5.3": {"input_per_mtok": 1.0, "output_per_mtok": 4.0},
    }
    cost = usage.estimate_cost_usd(
        "gpt-5.3-codex", 1_000_000, 500_000, pricing=pricing,
    )
    assert cost == pytest.approx(1.0 + 2.0)
    # Unknown model → honest zero, never a fabricated rate.
    assert usage.estimate_cost_usd("mystery-9", 10_000, 10_000,
                                   pricing=pricing) == 0.0
    assert usage.estimate_cost_usd("gpt-5.3", 0, 0, pricing={}) == 0.0


def test_summarize_by_role_aggregates():
    entries = [
        {"role": "planner", "model": "m1", "input_tokens": 10,
         "output_tokens": 5, "cost_usd": 0.01, "ok": True},
        {"role": "planner", "model": "m1", "input_tokens": 20,
         "output_tokens": 10, "cost_usd": 0.02, "ok": False},
        {"role": "analyst", "model": "m2", "as_role": "planner",
         "input_tokens": 7, "output_tokens": 3, "cost_usd": 0.0, "ok": True},
    ]
    rows = usage.summarize_by_role(entries)
    planner = next(r for r in rows if r["role"] == "planner")
    assert planner["calls"] == 2
    assert planner["failures"] == 1
    assert planner["input_tokens"] == 30
    assert planner["cost_usd"] == pytest.approx(0.03)
    analyst = next(r for r in rows if r["role"] == "analyst")
    assert analyst["backup_activations"] == 1


# ---------------------------------------------------------------------------
# _invoke_one records; _usage_update drains into a state delta
# ---------------------------------------------------------------------------

def _fake_llm(reply='{"ok": true}'):
    return SimpleNamespace(
        model_name="fake-model",
        invoke=lambda msgs: SimpleNamespace(
            content=reply,
            usage_metadata={"input_tokens": 42, "output_tokens": 13},
            additional_kwargs={},
            response_metadata={},
        ),
    )


def test_invoke_one_records_success(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda role: _fake_llm())
    content, err, entry = nodes._invoke_one("planner", "hi")
    assert err is None and content
    update = nodes._usage_update()
    assert len(update["llm_usage"]) == 1
    rec = update["llm_usage"][0]
    assert rec is entry  # live entry — amendable until drained
    assert rec["role"] == "planner" and rec["ok"] is True
    assert rec["transport_ok"] is True
    assert (rec["input_tokens"], rec["output_tokens"]) == (42, 13)
    assert update["budget_used"] == rec["cost_usd"]
    assert update["tokens_used"] == 55
    # Drained — a second call has nothing pending.
    assert nodes._usage_update() == {}


def test_invoke_one_records_failure(monkeypatch):
    def _raise(role):
        raise RuntimeError("provider down")
    monkeypatch.setattr(nodes, "get_llm", _raise)
    content, err, _entry = nodes._invoke_one("critic", "hi")
    assert err and not content
    recs = usage.drain()
    assert len(recs) == 1
    assert recs[0]["ok"] is False
    assert recs[0]["transport_ok"] is False
    assert "provider down" in recs[0]["error"]


def test_pre_call_cost_reservation_rejects_without_provider_call(monkeypatch):
    called = False

    class FakeLLM:
        model_name = "priced-model"
        max_tokens = 1000

        def invoke(self, _messages):
            nonlocal called
            called = True
            raise AssertionError("provider must not be called")

    monkeypatch.setattr(nodes, "get_llm", lambda _role: FakeLLM())
    monkeypatch.setattr(
        nodes.llm_usage,
        "estimate_cost_usd",
        lambda _model, _in_tok, _out_tok: 0.75,
    )

    content, err, entry = nodes._invoke_one(
        "planner", "prompt", max_cost_usd=0.50,
    )

    assert content == ""
    assert "cost reservation" in err
    assert called is False
    assert entry["rejection"] == "cost_reservation"
    assert entry["cost_usd"] == 0.0


def test_safe_invoke_charges_primary_before_reserving_backup(monkeypatch):
    calls: list[tuple[str, float | None]] = []

    def fake_invoke(role, _prompt, *, system_role=None, max_cost_usd=None):
        calls.append((role, max_cost_usd))
        if role == "analyst":
            entry = usage.record({
                "role": role, "model": "m", "input_tokens": 1,
                "output_tokens": 1, "cost_usd": 0.20,
                "ok": True, "transport_ok": True,
            })
            return "not json", None, entry
        entry = usage.record({
            "role": role, "model": "m", "input_tokens": 0,
            "output_tokens": 0, "cost_usd": 0.0,
            "ok": False, "transport_ok": False,
            "rejection": "cost_reservation",
        })
        return "", "cost reservation denied", entry

    monkeypatch.setattr(nodes, "_invoke_one", fake_invoke)
    out = nodes._safe_invoke(
        "analyst", "q", fallback={"answer": "fallback"},
        backup_roles=["critic"], remaining_budget_usd=0.30,
    )

    assert calls == [("analyst", 0.30), ("critic", pytest.approx(0.10))]
    assert out["answer"] == "fallback"
    entries = usage.drain()
    assert entries[-1]["budget_reservation_exhausted"] is True
    for entry in entries:
        usage.record(entry)
    assert nodes._usage_update()["budget_reservation_exhausted"] is True


def test_backup_call_marks_as_role(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda role: _fake_llm())
    nodes._invoke_one("critic", "hi", system_role="analyst")
    recs = usage.drain()
    assert recs[0]["role"] == "critic"
    assert recs[0]["as_role"] == "analyst"


# ---------------------------------------------------------------------------
# Review finding 5: unusable output counts as failure
# ---------------------------------------------------------------------------

def test_empty_response_is_a_failure(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda role: _fake_llm(reply=""))
    content, err, entry = nodes._invoke_one("planner", "hi")
    assert not content and "empty response" in err
    assert entry["ok"] is False
    assert entry["transport_ok"] is True   # HTTP worked; output didn't


def test_non_json_response_is_a_failure_even_when_salvaged(monkeypatch):
    monkeypatch.setattr(
        nodes, "get_llm",
        lambda role: _fake_llm(reply="The lives counter is at $0780."),
    )
    out = nodes._safe_invoke("analyst", "q", fallback={"answer": ""})
    # The caller still gets the salvaged text …
    assert out["_text"].startswith("The lives counter")
    # … but the call is honestly recorded as a contract failure.
    recs = usage.drain()
    assert len(recs) == 1
    assert recs[0]["ok"] is False
    assert "non-contract" in recs[0]["error"]
    assert recs[0]["transport_ok"] is True


def test_invalid_contract_array_is_a_failure(monkeypatch):
    monkeypatch.setattr(
        nodes, "get_llm", lambda role: _fake_llm(reply='["a", "b"]'),
    )
    nodes._safe_invoke("analyst", "q", fallback={"answer": ""})
    recs = usage.drain()
    assert recs and recs[0]["ok"] is False


def test_backup_success_after_transport_failure(monkeypatch):
    def _get(role):
        if role == "analyst":
            raise RuntimeError("analyst provider down")
        return _fake_llm(reply='{"answer": "at $0780"}')
    monkeypatch.setattr(nodes, "get_llm", _get)

    out = nodes._safe_invoke(
        "analyst", "q", fallback={"answer": ""}, backup_roles=["critic"],
    )
    assert out["answer"] == "at $0780"
    assert out["_role_used"] == "critic"
    recs = usage.drain()
    by_role = {r["role"]: r for r in recs}
    assert by_role["analyst"]["ok"] is False
    assert by_role["critic"]["ok"] is True
    assert by_role["critic"]["as_role"] == "analyst"  # backup activation
    rows = usage.summarize_by_role(recs)
    assert sum(r["failures"] for r in rows) == 1
    assert sum(r["backup_activations"] for r in rows) == 1


# ---------------------------------------------------------------------------
# Review finding 2: invocation-local collection (threads AND asyncio)
# ---------------------------------------------------------------------------

def test_concurrent_contexts_cannot_cross_drain():
    import threading

    results: dict[str, list] = {}
    barrier = threading.Barrier(2)

    def worker(tag: str):
        for i in range(5):
            usage.record({"role": tag, "ok": True, "input_tokens": i,
                          "output_tokens": 0, "cost_usd": 0.0})
        barrier.wait()          # both threads recorded before either drains
        results[tag] = usage.drain()

    threads = [threading.Thread(target=worker, args=(t,))
               for t in ("run_a", "run_b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results["run_a"]) == 5
    assert len(results["run_b"]) == 5
    assert {e["role"] for e in results["run_a"]} == {"run_a"}
    assert {e["role"] for e in results["run_b"]} == {"run_b"}
    assert usage.drain() == []  # main context untouched


def test_asyncio_tasks_cannot_cross_drain():
    """Final-review repro: asyncio context copies are shallow, so a
    mutable list stored in a ContextVar was SHARED between tasks — task A
    drained task B's entries (observed A=['a','b'], B=[]). Copy-on-write
    tuples must isolate them."""
    import asyncio

    # Parent context initialized/drained first — this is what used to
    # create the shared list the tasks then inherited.
    usage.record({"role": "parent-warmup", "ok": True})
    usage.drain()

    async def worker(tag: str, gate: "asyncio.Event"):
        usage.record({"role": tag, "ok": True})
        usage.record({"role": tag, "ok": True})
        await gate.wait()       # both tasks record before either drains
        return usage.drain()

    async def main():
        gate = asyncio.Event()
        t_a = asyncio.create_task(worker("a", gate))
        t_b = asyncio.create_task(worker("b", gate))
        await asyncio.sleep(0.01)
        gate.set()
        return await asyncio.gather(t_a, t_b)

    res_a, res_b = asyncio.run(main())
    assert [e["role"] for e in res_a] == ["a", "a"]
    assert [e["role"] for e in res_b] == ["b", "b"]
    assert usage.drain() == []  # parent context remains empty


def test_reset_clears_inherited_stragglers():
    usage.record({"role": "stale-from-crashed-run", "ok": True})
    usage.reset()
    assert usage.drain() == []


def test_mark_failed_still_amends_after_copy_on_write():
    """record() must keep returning the LIVE stored dict even though the
    container is now an immutable tuple."""
    entry = usage.record({"role": "planner", "ok": True})
    usage.mark_failed(entry, "late contract failure")
    drained = usage.drain()
    assert drained[0]["ok"] is False
    assert drained[0]["error"] == "late contract failure"


# ---------------------------------------------------------------------------
# Final review: role-contract dict validation + fallback telemetry
# ---------------------------------------------------------------------------

def test_wrong_shaped_dict_is_marked_failed_with_fallback(monkeypatch):
    """`{"unexpected": 1}` used to be returned as analyst success."""
    monkeypatch.setattr(
        nodes, "get_llm", lambda role: _fake_llm(reply='{"unexpected": 1}'),
    )
    out = nodes._safe_invoke("analyst", "q", fallback={"answer": "heuristic"})
    assert "_error" in out                      # heuristic fallback used
    assert out["answer"] == "heuristic"
    recs = usage.drain()
    assert len(recs) == 1
    assert recs[0]["ok"] is False
    assert "contract violated" in recs[0]["error"]
    assert recs[0]["fallback_activated"] is True


def test_wrong_shaped_primary_dict_triggers_backup(monkeypatch):
    def _get(role):
        if role == "analyst":
            return _fake_llm(reply='{"unexpected": 1}')
        return _fake_llm(reply='{"answer": "at $0780", "confidence": 0.7}')
    monkeypatch.setattr(nodes, "get_llm", _get)

    out = nodes._safe_invoke(
        "analyst", "q", fallback={"answer": ""}, backup_roles=["critic"],
    )
    assert out["answer"] == "at $0780"
    assert out["_role_used"] == "critic"
    recs = usage.drain()
    by_role = {r["role"]: r for r in recs}
    assert by_role["analyst"]["ok"] is False
    assert by_role["critic"]["ok"] is True
    rows = usage.summarize_by_role(recs)
    assert sum(r["backup_activations"] for r in rows) == 1
    assert sum(r["fallback_activations"] for r in rows) == 0  # distinct


@pytest.mark.parametrize("role,fallback", [
    ("planner", {"plan": [{"id": "s1", "tool": "kb"}]}),
    ("executor", {"step_id": "s1", "tool": "kb", "args": {}}),
    ("synthesizer", {"labels": [], "routines": [], "data_structures": [],
                     "hypotheses": [], "notes": ""}),
    ("curator", {"summary": "", "addresses_kept": [],
                 "events_compacted": []}),
    ("analyst", {"answer": "h", "confidence": 0.1}),
    ("critic", {"decision": "accept", "critique": ""}),
])
def test_wrong_shaped_dict_rejected_for_every_role(monkeypatch, role,
                                                   fallback):
    monkeypatch.setattr(
        nodes, "get_llm", lambda r: _fake_llm(reply='{"unexpected": 1}'),
    )
    out = nodes._safe_invoke(role, "q", fallback=fallback)
    assert "_error" in out
    recs = usage.drain()
    assert recs and recs[0]["ok"] is False
    assert "contract violated" in recs[0]["error"]


def test_valid_contracts_accepted_per_role(monkeypatch):
    """The validator must not reject legitimate minimal replies."""
    cases = {
        "planner": '{"plan": [{"id": "s1", "tool": "kb", "args": {}}]}',
        "executor": '{"step_id": "s1", "tool": "kb", "args": {"mode": "stats"}}',
        "synthesizer": '{"labels": [], "routines": []}',
        "curator": '{"summary": "compact"}',
        "analyst": '{"answer": "at $0780"}',
        "critic": '{"decision": "revise", "critique": "x"}',
    }
    for role, reply in cases.items():
        monkeypatch.setattr(
            nodes, "get_llm", lambda r, _reply=reply: _fake_llm(reply=_reply),
        )
        out = nodes._safe_invoke(role, "q", fallback={})
        assert "_error" not in out, f"{role} reply wrongly rejected"
        recs = usage.drain()
        assert recs[0]["ok"] is True, f"{role} usage wrongly demoted"


def test_envelope_wrapped_contract_is_unwrapped(monkeypatch):
    monkeypatch.setattr(
        nodes, "get_llm",
        lambda r: _fake_llm(
            reply='{"candidate_answer": {"answer": "at $0780"}}',
        ),
    )
    out = nodes._safe_invoke("analyst", "q", fallback={"answer": ""})
    assert out["answer"] == "at $0780"
    assert usage.drain()[0]["ok"] is True


def test_invalid_critic_decision_is_rejected(monkeypatch):
    monkeypatch.setattr(
        nodes, "get_llm",
        lambda r: _fake_llm(reply='{"decision": "maybe", "critique": "?"}'),
    )
    out = nodes._safe_invoke("critic", "q", fallback={"decision": "accept"})
    assert "_error" in out
    assert usage.drain()[0]["ok"] is False


def test_raw_text_salvage_counts_as_fallback(monkeypatch):
    monkeypatch.setattr(
        nodes, "get_llm",
        lambda role: _fake_llm(reply="Plain prose, no JSON at all."),
    )
    nodes._safe_invoke("analyst", "q", fallback={"answer": ""})
    recs = usage.drain()
    assert recs[0]["fallback_activated"] is True
    rows = usage.summarize_by_role(recs)
    assert sum(r["fallback_activations"] for r in rows) == 1


# ---------------------------------------------------------------------------
# report table
# ---------------------------------------------------------------------------

def test_write_report_includes_usage_table(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "SESSIONS_DIR", tmp_path)
    state = {
        "game": "Test Game",
        "question": "q?",
        "candidate_answer": {"answer": "a", "confidence": 0.5,
                             "evidence": [], "open_questions": []},
        "verdict": {"decision": "accept", "critique": ""},
        "plan": [],
        "tool_results": [
            {"step_id": "s1", "tool": "capstone", "ok": True, "data": "x"},
            {"step_id": "s2", "tool": "tavily", "ok": False, "data": "no key"},
        ],
        "llm_usage": [
            {"role": "planner", "model": "fake-model", "input_tokens": 100,
             "output_tokens": 50, "cost_usd": 0.0, "ok": True},
        ],
        "budget_used": 0.0,
        "iteration": 1,
    }
    nodes.write_report(state)
    report = (tmp_path / "test_game" / "report.md").read_text()
    assert "## LLM usage (per role)" in report
    assert "| planner | fake-model | 1 |" in report
    assert "No `pricing` section" in report        # tokens>0, cost==0
    assert "total tokens: 150" in report           # fallback from entries
    assert "calls priced/unpriced: 0/1" in report
    assert "budget limit enforced: none" in report
    assert "- capstone: 1 call(s), 0 failure(s)" in report
    assert "- tavily: 1 call(s), 1 failure(s)" in report
