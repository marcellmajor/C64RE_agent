"""Opt-in native structured output with provider-safe fallback (tracker 6.8)."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

import graph.llm as llm_factory
import graph.nodes as nodes
from graph import usage


_ANALYST = {
    "answer": "Counter is at $00C0.",
    "confidence": 0.9,
    "evidence": ["evt_1"],
    "open_questions": [],
}


def _message(content):
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
        },
    )


class _StructuredRunnable:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def invoke(self, _messages):
        if self.error:
            raise self.error
        return self.result


def test_native_schema_success_skips_plain_json_path(monkeypatch):
    captured = {}

    class FakeLLM:
        model_name = "fake-model"

        def with_structured_output(self, schema, **kwargs):
            captured["schema"] = schema
            captured["kwargs"] = kwargs
            return _StructuredRunnable({
                "raw": _message(""),
                "parsed": dict(_ANALYST),
                "parsing_error": None,
            })

        def invoke(self, _messages):
            raise AssertionError("plain invoke must not run after schema success")

    usage.reset()
    monkeypatch.setattr(nodes, "get_llm", lambda _role: FakeLLM())
    monkeypatch.setattr(nodes, "structured_output_enabled", lambda _role: True)

    result = nodes._safe_invoke("analyst", "prompt", fallback={
        "answer": "", "confidence": 0.0, "evidence": [], "open_questions": [],
    })
    entries = usage.drain()

    assert result["answer"] == _ANALYST["answer"]
    assert result["_role_used"] == "analyst"
    assert captured["schema"]["title"] == "C64REAnalyst"
    assert captured["kwargs"] == {
        "method": "json_schema", "include_raw": True, "strict": False,
    }
    assert len(entries) == 1
    assert entries[0]["structured_output"] is True
    assert entries[0]["ok"] is True
    assert entries[0]["input_tokens"] == 12


def test_provider_schema_rejection_falls_back_to_same_role_plain_json(
    monkeypatch,
):
    plain_calls = 0

    class FakeLLM:
        model_name = "compat-model"

        def with_structured_output(self, _schema, **_kwargs):
            return _StructuredRunnable(error=RuntimeError("response_format rejected"))

        def invoke(self, _messages):
            nonlocal plain_calls
            plain_calls += 1
            return _message(json.dumps(_ANALYST))

    usage.reset()
    monkeypatch.setattr(nodes, "get_llm", lambda _role: FakeLLM())
    monkeypatch.setattr(nodes, "structured_output_enabled", lambda _role: True)

    result = nodes._safe_invoke("analyst", "prompt", fallback={
        "answer": "", "confidence": 0.0, "evidence": [], "open_questions": [],
    })
    entries = usage.drain()

    assert result["answer"] == _ANALYST["answer"]
    assert plain_calls == 1
    assert len(entries) == 2
    assert entries[0]["structured_output"] is True
    assert entries[0]["transport_ok"] is False
    assert entries[1]["ok"] is True
    assert not entries[1].get("structured_output")


def test_structured_failure_cannot_reuse_reserved_budget_for_plain_retry(
    monkeypatch,
):
    plain_calls = 0

    class FakeLLM:
        model_name = "priced-model"
        max_tokens = 100

        def with_structured_output(self, _schema, **_kwargs):
            return _StructuredRunnable(error=RuntimeError("connection reset"))

        def invoke(self, _messages):
            nonlocal plain_calls
            plain_calls += 1
            raise AssertionError("plain provider call must be reservation-denied")

    usage.reset()
    monkeypatch.setattr(nodes, "get_llm", lambda _role: FakeLLM())
    monkeypatch.setattr(nodes, "structured_output_enabled", lambda _role: True)
    monkeypatch.setattr(
        nodes, "_call_cost_reservation_usd", lambda *_a, **_k: 0.04,
    )

    result = nodes._safe_invoke(
        "analyst",
        "prompt",
        fallback={
            "answer": "fallback", "confidence": 0.0,
            "evidence": [], "open_questions": [],
        },
        remaining_budget_usd=0.06,
    )
    entries = usage.drain()

    assert result["answer"] == "fallback"
    assert result["_budget_exhausted"] is True
    assert plain_calls == 0
    assert entries[0]["structured_output"] is True
    assert entries[0]["reserved_cost_usd"] > 0
    assert entries[1]["rejection"] == "cost_reservation"


def test_disabled_native_output_never_wraps_client(monkeypatch):
    class FakeLLM:
        model_name = "plain-model"

        def with_structured_output(self, *_args, **_kwargs):
            raise AssertionError("wrapper must stay disabled")

        def invoke(self, _messages):
            return _message(json.dumps(_ANALYST))

    usage.reset()
    monkeypatch.setattr(nodes, "get_llm", lambda _role: FakeLLM())
    monkeypatch.setattr(nodes, "structured_output_enabled", lambda _role: False)

    result = nodes._safe_invoke("analyst", "prompt", fallback={
        "answer": "", "confidence": 0.0, "evidence": [], "open_questions": [],
    })
    entries = usage.drain()

    assert result["answer"] == _ANALYST["answer"]
    assert len(entries) == 1 and entries[0]["ok"]


def test_structured_output_config_resolves_env_then_role_then_default(
    monkeypatch,
):
    monkeypatch.delenv("C64RE_STRUCTURED_OUTPUT", raising=False)
    monkeypatch.setattr(llm_factory, "load_config", lambda: {
        "defaults": {"structured_output": False},
        "agents": {
            "analyst": {"structured_output": True},
            "critic": {},
        },
    })
    assert llm_factory.structured_output_enabled("analyst") is True
    assert llm_factory.structured_output_enabled("critic") is False

    monkeypatch.setenv("C64RE_STRUCTURED_OUTPUT", "1")
    assert llm_factory.structured_output_enabled("critic") is True
    monkeypatch.setenv("C64RE_STRUCTURED_OUTPUT", "0")
    assert llm_factory.structured_output_enabled("analyst") is False


def test_structured_output_roles_env_enables_only_canary_roles(monkeypatch):
    monkeypatch.delenv("C64RE_STRUCTURED_OUTPUT", raising=False)
    monkeypatch.setenv("C64RE_STRUCTURED_OUTPUT_ROLES", "analyst, critic")
    monkeypatch.setattr(llm_factory, "load_config", lambda: {
        "defaults": {"structured_output": False}, "agents": {},
    })

    assert llm_factory.structured_output_enabled("analyst") is True
    assert llm_factory.structured_output_enabled("critic") is True
    assert llm_factory.structured_output_enabled("planner") is False
