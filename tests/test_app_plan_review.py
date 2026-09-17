"""Regression tests for the Streamlit plan-review checkpoint.

`pending_plan_json` is the key of a `st.text_area`, and Streamlit refuses
writes to a widget-backed key once that widget has been instantiated in the
same script run. Every handler inside the review block therefore stages the
next textarea contents under `_pending_plan_json`, which is applied on the
following run. Cancelling a review used to assign the widget key directly and
took the whole app down with a `StreamlitAPIException`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.agent_runner import PlanReview

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")

PLAN_A = [{"id": "aaa", "goal": "PLAN-A", "tool": "capstone",
           "args": {"mode": "find_entry"}, "depends_on": [], "hypothesis": None}]
PLAN_B = [{"id": "bbb", "goal": "PLAN-B", "tool": "kb",
           "args": {"mode": "stats"}, "depends_on": [], "hypothesis": None}]
PLAN_MUT = [{"id": "m1", "goal": "poke a colour register", "tool": "vice",
             "args": {"method": "vice.memory.write", "address": "$D020", "value": 1},
             "depends_on": [], "hypothesis": None}]


def _review(plan, mutating=()):
    return PlanReview(
        question="Where is the score stored?",
        thread_id="test-thread",
        plan=plan,
        mutating_step_ids=list(mutating),
        seen_messages=0,
        started_monotonic=0.0,
    )


def _app_awaiting_review(plan, mutating=()):
    """Run the app up to a rendered plan-review checkpoint (no agent calls)."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.session_state["pending_plan_review"] = _review(plan, mutating)
    at.session_state["_pending_plan_json"] = json.dumps(plan, indent=2)
    at.run()
    return at


def test_staged_plan_reaches_the_textarea():
    at = _app_awaiting_review(PLAN_A)
    assert not at.exception
    assert "PLAN-A" in at.text_area(key="pending_plan_json").value


def test_cancel_after_textarea_is_rendered_does_not_raise():
    at = _app_awaiting_review(PLAN_A)
    at.button(key="cancel_reviewed_plan").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["pending_plan_review"] is None


def test_next_review_replaces_edited_json_from_a_cancelled_one():
    at = _app_awaiting_review(PLAN_A)
    at.text_area(key="pending_plan_json").set_value('["EDITED"]').run()
    at.button(key="cancel_reviewed_plan").click().run()

    # A later question checkpoints again; the textarea must show the new plan.
    at.session_state["pending_plan_review"] = _review(PLAN_B)
    at.session_state["_pending_plan_json"] = json.dumps(PLAN_B, indent=2)
    at.run()

    shown = at.text_area(key="pending_plan_json").value
    assert "PLAN-B" in shown
    assert "PLAN-A" not in shown and "EDITED" not in shown
    assert not at.exception


def test_unapproved_mutation_is_gated_without_crashing():
    at = _app_awaiting_review(PLAN_MUT, mutating=["m1"])
    at.button(key="run_reviewed_plan").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Explicit approval is still required" in e.value for e in at.error)


def test_malformed_plan_json_is_reported_without_crashing():
    at = _app_awaiting_review(PLAN_A)
    at.text_area(key="pending_plan_json").set_value("{not json").run()
    at.button(key="run_reviewed_plan").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Plan is not valid" in e.value for e in at.error)


# --------------------------------------------------------------------------- #
# Pre-flight guard: a run must not start with an unusable session.
#
# These tests deliberately leave `game` empty. That is the one blocked path
# that can never reach `begin_question`, so no agent turn (and no LLM spend or
# session-directory write) can be triggered from the test suite. `dump_path`
# is auto-populated by the sidebar widget, so it cannot be used for this.
# --------------------------------------------------------------------------- #


def _app_with_question(game: str, question: str = "Where is the score stored?"):
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.session_state["game"] = game
    at.session_state["queued_question"] = question
    at.run()
    return at


def test_empty_game_blocks_the_run_with_a_modal():
    at = _app_with_question("")
    assert not at.exception, [e.value for e in at.exception]
    assert any("sidebar is incomplete" in w.value for w in at.warning)
    assert any("**Game** is empty" in m.value for m in at.markdown)
    assert any(b.key == "dismiss_missing_inputs" for b in at.button)


def test_blocked_run_renders_no_chat_turn_and_keeps_the_question():
    at = _app_with_question("", question="How does the maze scroll?")
    assert len(at.chat_message) == 0
    assert at.session_state["pending_question"] == "How does the maze scroll?"
    assert at.session_state["pending_plan_review"] is None


def test_whitespace_only_game_is_still_treated_as_missing():
    at = _app_with_question("   ")
    assert any("**Game** is empty" in m.value for m in at.markdown)
    assert len(at.chat_message) == 0
