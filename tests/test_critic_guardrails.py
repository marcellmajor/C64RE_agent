"""Critic guardrails (tracker 0.2 / 0.4 / 0.5) with the LLM mocked out.

States deliberately omit ``kb_handle`` so no KnowledgeStore is touched;
the growth flag then defaults to True (never terminate on unknown).
"""

import pytest

import graph.nodes as nodes
from graph.plan_utils import MAX_CONSECUTIVE_REVISES


def _patch_critic(monkeypatch, reply: dict):
    def _fake(role, prompt, fallback, backup_roles=None, **kw):
        return dict(reply)
    monkeypatch.setattr(nodes, "_safe_invoke", _fake)


def _state(**over):
    base = {
        "question": "where is the lives counter?",
        "candidate_answer": {
            "answer": "at $0780",
            "evidence": ["$0780"],
            "confidence": 0.7,
            "open_questions": [],
        },
        "kb_digest": "(digest)",
        "plan": [],
        "iteration": 1,
        "replan_count": 0,
        "revise_count": 0,
        "history": [],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# 0.5 — suggested_steps are BLOCKING; optional_followups are not
# ---------------------------------------------------------------------------

def test_accept_with_suggested_steps_escalates_to_replan(monkeypatch):
    """suggested_steps mean blocking tool work — an accept carrying them
    must replan (review fix), never silently drop the required work."""
    _patch_critic(monkeypatch, {
        "decision": "accept",
        "critique": "solid",
        "suggested_steps": [{"id": "c1", "goal": "trace the writer",
                             "tool": "vice", "args": {}}],
    })
    out = nodes.critic_node(_state())
    verdict = out["verdict"]
    assert verdict["decision"] == "replan"
    assert verdict["suggested_steps"]  # preserved for the planner
    assert "candidate_answer" not in out  # nothing merged on non-accept


def test_accept_with_optional_followups_stays_accept(monkeypatch):
    _patch_critic(monkeypatch, {
        "decision": "accept",
        "critique": "solid",
        "suggested_steps": [],
        "optional_followups": [{"goal": "trace the writer someday"}],
    })
    out = nodes.critic_node(_state())
    verdict = out["verdict"]
    assert verdict["decision"] == "accept"
    assert verdict["suggested_steps"] == []
    # Follow-ups reach the report via the candidate's open questions.
    cand = out["candidate_answer"]
    assert any("trace the writer someday" in q for q in cand["open_questions"])


def test_revise_with_suggested_steps_escalates_to_replan(monkeypatch):
    _patch_critic(monkeypatch, {
        "decision": "revise",
        "critique": "needs a live check",
        "suggested_steps": [{"id": "c1", "goal": "breakpoint $C145",
                             "tool": "vice", "args": {}}],
    })
    out = nodes.critic_node(_state())
    assert out["verdict"]["decision"] == "replan"
    assert out["verdict"]["suggested_steps"]  # preserved for the planner


# ---------------------------------------------------------------------------
# 0.4 — revise cap
# ---------------------------------------------------------------------------

def test_revise_increments_counter(monkeypatch):
    _patch_critic(monkeypatch, {
        "decision": "revise", "critique": "reword", "suggested_steps": [],
    })
    out = nodes.critic_node(_state(revise_count=0))
    assert out["verdict"]["decision"] == "revise"
    assert out["revise_count"] == 1


def test_revise_cap_forces_accept(monkeypatch):
    _patch_critic(monkeypatch, {
        "decision": "revise", "critique": "still unhappy",
        "suggested_steps": [],
    })
    out = nodes.critic_node(_state(revise_count=MAX_CONSECUTIVE_REVISES))
    assert out["verdict"]["decision"] == "accept"
    assert "forcing `accept`" in out["verdict"]["critique"]
    assert out["revise_count"] == 0


def test_accept_resets_revise_counter(monkeypatch):
    _patch_critic(monkeypatch, {
        "decision": "accept", "critique": "fine", "suggested_steps": [],
    })
    out = nodes.critic_node(_state(revise_count=1))
    assert out["revise_count"] == 0


# ---------------------------------------------------------------------------
# evidence guardrail still active, and capped
# ---------------------------------------------------------------------------

def test_evidence_guardrail_then_cap_terminates(monkeypatch):
    """confidence≥0.85 with no evidence demotes to revise — but the cap
    must still terminate the loop once revises are exhausted."""
    _patch_critic(monkeypatch, {
        "decision": "accept", "critique": "looks great",
        "suggested_steps": [],
    })
    st = _state(revise_count=MAX_CONSECUTIVE_REVISES)
    st["candidate_answer"] = {"answer": "x", "evidence": [],
                              "confidence": 0.9, "open_questions": []}
    out = nodes.critic_node(st)
    assert out["verdict"]["decision"] == "accept"          # cap wins last
    assert "auto-guardrail" in out["verdict"]["critique"]  # both noted


# ---------------------------------------------------------------------------
# 0.2 — growth flag default without a KB handle
# ---------------------------------------------------------------------------

def test_growth_flag_defaults_true_without_kb(monkeypatch):
    _patch_critic(monkeypatch, {
        "decision": "replan", "critique": "dig more",
        "suggested_steps": [{"id": "c1", "goal": "g", "tool": "kb",
                             "args": {}}],
    })
    out = nodes.critic_node(_state())
    assert out["verdict"]["kb_grew_since_last_verdict"] is True
