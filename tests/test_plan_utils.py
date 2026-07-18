"""Unit tests for graph.plan_utils (tracker items 0.1 / 0.3 / 0.6 / 0.7)."""

from graph.plan_utils import (
    MAX_CONSECUTIVE_REVISES,
    MAX_ITERS,
    MAX_PLAN_STEPS,
    MAX_STEP_ATTEMPTS,
    RECURSION_LIMIT,
    TRUNCATED_CONFIDENCE_CAP,
    failed_step_notes,
    is_permanent_failure,
    normalize_truncated_state,
    pending_steps,
    resolve_session_slug,
    runnable_steps,
    slugify,
    step_status,
)


def _ok(sid, **kw):
    return {"step_id": sid, "tool": "kb", "ok": True, "data": "fine", **kw}


def _fail(sid, **kw):
    return {"step_id": sid, "tool": "kb", "ok": False, "data": "boom", **kw}


# ---------------------------------------------------------------------------
# slugify (0.7)
# ---------------------------------------------------------------------------

def test_slugify_canonical():
    assert slugify("Wizard of Wor") == "wizard_of_wor"
    assert slugify("  Boulder Dash ") == "boulder_dash"
    assert slugify("") == "unknown"
    assert slugify(None) == "unknown"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# step_status (0.3)
# ---------------------------------------------------------------------------

def test_success_is_done():
    st = step_status([_ok("s1")])
    assert st.done == {"s1"}
    assert st.exhausted == set()


def test_single_failure_is_not_done():
    """The core 0.3 fix: one ok=False result must NOT burn the step."""
    st = step_status([_fail("s1")])
    assert "s1" not in st.done
    assert st.fail_counts["s1"] == 1


def test_failures_exhaust_after_max_attempts():
    st = step_status([_fail("s1")] * MAX_STEP_ATTEMPTS)
    assert "s1" in st.exhausted
    assert "s1" in st.done


def test_failure_then_success_counts_as_succeeded():
    st = step_status([_fail("s1"), _ok("s1")])
    assert "s1" in st.succeeded
    assert "s1" not in st.exhausted
    assert "s1" in st.done


def test_permanent_failure_exhausts_immediately():
    st = step_status([_fail("s1", retryable=False)])
    assert "s1" in st.exhausted
    st2 = step_status([_fail("s2", rejection="banned_method")])
    assert "s2" in st2.exhausted
    st3 = step_status([_fail("s3", rejection="dependency_unsatisfied")])
    assert "s3" in st3.exhausted


def test_is_permanent_failure_ignores_successes():
    assert not is_permanent_failure(_ok("s1", retryable=False))


# ---------------------------------------------------------------------------
# pending / runnable (0.3 / 0.6)
# ---------------------------------------------------------------------------

def _plan_state(plan, tool_results):
    return {"plan": plan, "tool_results": tool_results}


def test_failed_step_stays_pending_until_exhausted():
    plan = [{"id": "s1", "tool": "kb", "depends_on": []}]
    state = _plan_state(plan, [_fail("s1")])
    assert [s["id"] for s in pending_steps(state)] == ["s1"]

    state = _plan_state(plan, [_fail("s1")] * MAX_STEP_ATTEMPTS)
    assert pending_steps(state) == []


def test_runnable_respects_dependencies():
    plan = [
        {"id": "s1", "tool": "kb", "depends_on": []},
        {"id": "s2", "tool": "kb", "depends_on": ["s1"]},
    ]
    state = _plan_state(plan, [])
    assert [s["id"] for s in runnable_steps(state)] == ["s1"]

    state = _plan_state(plan, [_ok("s1")])
    assert [s["id"] for s in runnable_steps(state)] == ["s2"]


def test_exhausted_dependency_blocks_dependent():
    """Only a SUCCEEDED prerequisite satisfies `depends_on` (review fix):
    a permanently-failed dep leaves its dependent pending-but-unrunnable,
    so the executor declares the plan blocked and the router replans."""
    plan = [
        {"id": "s1", "tool": "tavily", "depends_on": []},
        {"id": "s2", "tool": "kb", "depends_on": ["s1"]},
    ]
    state = _plan_state(plan, [_fail("s1", retryable=False)])
    assert runnable_steps(state) == []
    assert [s["id"] for s in pending_steps(state)] == ["s2"]


def test_retry_exhausted_dependency_blocks_dependent():
    plan = [
        {"id": "s1", "tool": "kb", "depends_on": []},
        {"id": "s2", "tool": "kb", "depends_on": ["s1"]},
    ]
    state = _plan_state(plan, [_fail("s1")] * MAX_STEP_ATTEMPTS)
    assert runnable_steps(state) == []
    assert [s["id"] for s in pending_steps(state)] == ["s2"]


def test_retryable_failed_dependency_defers_dependent():
    """While a dep still has retries left, only the dep itself runs."""
    plan = [
        {"id": "s1", "tool": "kb", "depends_on": []},
        {"id": "s2", "tool": "kb", "depends_on": ["s1"]},
    ]
    state = _plan_state(plan, [_fail("s1")])
    assert [s["id"] for s in runnable_steps(state)] == ["s1"]


def test_untried_steps_run_before_retries():
    plan = [
        {"id": "s1", "tool": "kb", "depends_on": []},
        {"id": "s2", "tool": "kb", "depends_on": []},
    ]
    state = _plan_state(plan, [_fail("s1")])
    assert [s["id"] for s in runnable_steps(state)] == ["s2", "s1"]


def test_cycle_produces_no_runnable_but_keeps_pending():
    plan = [
        {"id": "s1", "tool": "kb", "depends_on": ["s2"]},
        {"id": "s2", "tool": "kb", "depends_on": ["s1"]},
    ]
    state = _plan_state(plan, [])
    assert runnable_steps(state) == []
    assert len(pending_steps(state)) == 2


# ---------------------------------------------------------------------------
# failed_step_notes (0.3)
# ---------------------------------------------------------------------------

def test_failed_step_notes_only_lists_current_plan_exhausted():
    plan = [
        {"id": "s1", "tool": "tavily", "goal": "search docs", "depends_on": []},
        {"id": "s2", "tool": "kb", "goal": "lookup", "depends_on": []},
    ]
    results = [
        _fail("s1", retryable=False),
        _fail("s2"),                       # still retryable — not listed
        _fail("old_s9", retryable=False),  # not in this plan — not listed
    ]
    notes = failed_step_notes(_plan_state(plan, results))
    assert len(notes) == 1
    assert "s1" in notes[0] and "tavily" in notes[0] and "FAILED" in notes[0]


# ---------------------------------------------------------------------------
# RECURSION_LIMIT derivation (0.1 review)
# ---------------------------------------------------------------------------

def test_recursion_limit_covers_worst_case_graph_path():
    """The limit must let MAX_ITERS complete even when every step of a
    maximal plan is retried to exhaustion via the curator path
    (executor → tool → synthesizer → curator = 4 super-steps), plus the
    capped revise loop and load_inputs/write_report overhead."""
    worst_case = (
        MAX_ITERS
        * (MAX_PLAN_STEPS * MAX_STEP_ATTEMPTS * 4
           + 3                            # planner + analyst + critic
           + MAX_CONSECUTIVE_REVISES * 2  # capped analyst↔critic revises
           )
        + 2                               # load_inputs + write_report
    )
    assert RECURSION_LIMIT >= worst_case


# ---------------------------------------------------------------------------
# resolve_session_slug — legacy-dir compatibility (0.7 review)
# ---------------------------------------------------------------------------

def test_resolve_prefers_canonical_when_it_exists(tmp_path):
    (tmp_path / "wizard_of_wor").mkdir()
    (tmp_path / "wizard-of-wor").mkdir()
    assert resolve_session_slug("Wizard of Wor", tmp_path) == "wizard_of_wor"


def test_resolve_falls_back_to_legacy_hyphen_dir(tmp_path):
    (tmp_path / "wizard-of-wor").mkdir()
    assert resolve_session_slug("Wizard of Wor", tmp_path) == "wizard-of-wor"


def test_resolve_maps_hyphenated_name_to_existing_underscore_dir(tmp_path):
    (tmp_path / "wizard_of_wor").mkdir()
    assert resolve_session_slug("Wizard-of-Wor", tmp_path) == "wizard_of_wor"


def test_resolve_defaults_to_canonical_when_nothing_exists(tmp_path):
    assert resolve_session_slug("Wizard of Wor", tmp_path) == "wizard_of_wor"


# ---------------------------------------------------------------------------
# normalize_truncated_state (0.1 review)
# ---------------------------------------------------------------------------

def test_normalize_caps_confidence_and_explains():
    state = {
        "candidate_answer": {
            "answer": "at $0780",
            "confidence": 0.95,
            "evidence": ["$0780"],
            "open_questions": [],
        },
        "verdict": {"decision": "replan", "critique": "needs more"},
    }
    out = normalize_truncated_state(
        state, reason="recursion_exhausted", detail="limit hit",
    )
    assert out["termination_reason"] == "recursion_exhausted"
    cand = out["candidate_answer"]
    assert cand["confidence"] == TRUNCATED_CONFIDENCE_CAP
    assert any("Research truncated" in q for q in cand["open_questions"])
    assert out["verdict"]["decision"] == "recursion_exhausted"
    assert out["verdict"]["prior_decision"] == "replan"
    # Original state untouched (runners keep the raw checkpoint).
    assert state["candidate_answer"]["confidence"] == 0.95


def test_normalize_keeps_lower_confidence_and_handles_empty_state():
    out = normalize_truncated_state(
        {"candidate_answer": {"answer": "x", "confidence": 0.2}},
        reason="budget_exceeded",
    )
    assert out["candidate_answer"]["confidence"] == 0.2

    out2 = normalize_truncated_state(None, reason="recursion_exhausted")
    assert out2["termination_reason"] == "recursion_exhausted"
    assert out2["candidate_answer"]["confidence"] == 0.0
    assert out2["candidate_answer"]["answer"]
