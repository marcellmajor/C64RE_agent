"""Deterministic plan/loop helpers shared by nodes, routers and runners.

Pure functions over ``C64State`` fragments — no LLM calls, no KB I/O —
so both `graph.nodes` and `graph.routers` can import them without
creating an import cycle, and so they are trivially unit-testable.

NOTE for consumers outside the `graph` package (`code_kb`, `scripts`):
importing this module executes ``graph/__init__.py``, which compiles the
full LangGraph. Import lazily (inside the function that needs it) to
avoid a circular import through ``graph.build`` → ``code_kb``.

Introduced by tracker items 0.1/0.3/0.4/0.6/0.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Loop budgets (tracker 0.1 / 0.3 / 0.4)
# ---------------------------------------------------------------------------

# Hard cap for the outer critic loop (planner increments `iteration`).
# Lives here (not graph.routers) so RECURSION_LIMIT below can be derived
# from it without an import cycle; graph.routers re-exports it.
MAX_ITERS = 12

# The planner's output is truncated to this many steps per iteration
# (dangling `depends_on` refs to truncated steps are dropped with it),
# which makes the recursion-limit derivation below an actual bound.
MAX_PLAN_STEPS = 12

# A step is abandoned (treated as done-but-failed) after this many failed
# attempts, or immediately when the failure is marked non-retryable.
MAX_STEP_ATTEMPTS = 2

# Consecutive `revise` verdicts before the critic is forced to `accept`
# (with its remaining concerns attached to the critique), so the
# analyst↔critic ping-pong cannot run until the recursion limit kills it.
MAX_CONSECUTIVE_REVISES = 2

# Worst-case super-steps per executed plan step:
# executor → tool → synthesizer → curator (the curator gate can fire
# after every synthesizer pass once the KB crosses its size threshold).
_STEP_SUPERSTEPS = 4

# Per-iteration overhead: planner + analyst + critic …
_ITER_OVERHEAD = 3
# … plus up to MAX_CONSECUTIVE_REVISES analyst→critic revise round-trips.
_REVISE_SUPERSTEPS = MAX_CONSECUTIVE_REVISES * 2

# Whole-run overhead: load_inputs + write_report.
_RUN_OVERHEAD = 2

# LangGraph's default recursion limit is 25 super-steps — a single 7-step
# plan iteration — which made `MAX_ITERS` unreachable (tracker 0.1).
# Derived worst case: every iteration runs a maximal plan where every
# step is retried to exhaustion through the curator path, plus the
# capped revise loop: 12 × (12×2×4 + 3 + 4) + 2 = 1238.
RECURSION_LIMIT = (
    MAX_ITERS
    * (MAX_PLAN_STEPS * MAX_STEP_ATTEMPTS * _STEP_SUPERSTEPS
       + _ITER_OVERHEAD + _REVISE_SUPERSTEPS)
    + _RUN_OVERHEAD
)

# Ceiling applied to `candidate_answer.confidence` when a run is
# truncated (recursion/budget) instead of critic-accepted: the answer
# was never validated, so it must not report a validated-looking score.
TRUNCATED_CONFIDENCE_CAP = 0.4

# Failure classes that cannot succeed on a retry within the same run.
# Tool nodes also set an explicit ``retryable: False`` on results whose
# failure is a missing runtime dependency (API key, server, dump).
_PERMANENT_REJECTIONS = {"banned_method", "dependency_unsatisfied"}


# ---------------------------------------------------------------------------
# Game slug (tracker 0.7)
# ---------------------------------------------------------------------------

def slugify(game: str) -> str:
    """Canonical game slug — the single source of truth for
    ``sessions/<slug>`` directories, thread ids, report paths and UI
    preference keys.

    Previously `main.py` derived thread ids with hyphens while the
    session directories used underscores; underscores win because the
    existing ``sessions/`` trees already use them.
    """
    return (game or "").strip().lower().replace(" ", "_") or "unknown"


def resolve_session_slug(game: str, sessions_root: Path | str) -> str:
    """Slug to use under ``sessions_root`` for *game*, honouring legacy dirs.

    Prefers the canonical `slugify()` result. If no directory exists for
    it but one exists for the legacy hyphen/underscore-swapped variant
    (e.g. a session created while thread ids used hyphens, or a game name
    typed with hyphens instead of spaces), the existing directory wins so
    accumulated KBs keep being found and extended.
    """
    canonical = slugify(game)
    root = Path(sessions_root)
    if (root / canonical).exists():
        return canonical
    for legacy in (
        canonical.replace("_", "-"),
        canonical.replace("-", "_"),
    ):
        if legacy != canonical and (root / legacy).exists():
            return legacy
    return canonical


# ---------------------------------------------------------------------------
# Step status (tracker 0.3 / 0.6)
# ---------------------------------------------------------------------------

def is_permanent_failure(result: dict[str, Any]) -> bool:
    """True when a failed tool result can never succeed on retry."""
    if result.get("ok"):
        return False
    if result.get("retryable") is False:
        return True
    return str(result.get("rejection") or "") in _PERMANENT_REJECTIONS


@dataclass
class StepStatus:
    """Per-step outcome ledger derived from ``state["tool_results"]``.

    Two distinct notions (tracker 0.6 review):

    * **terminal** (`done`) — the step will not be dispatched again:
      it succeeded, or its retries are exhausted;
    * **satisfied dependency** (`succeeded`) — only a step that actually
      SUCCEEDED satisfies a `depends_on` reference. A permanently-failed
      prerequisite blocks its dependents; the executor then reports the
      plan as blocked and the router replans.
    """

    succeeded: set[str] = field(default_factory=set)
    fail_counts: dict[str, int] = field(default_factory=dict)
    permanent_failures: set[str] = field(default_factory=set)

    @property
    def exhausted(self) -> set[str]:
        """Steps that failed permanently or ran out of attempts
        (and never succeeded)."""
        out = set(self.permanent_failures)
        out.update(
            sid for sid, n in self.fail_counts.items()
            if n >= MAX_STEP_ATTEMPTS
        )
        return out - self.succeeded

    @property
    def done(self) -> set[str]:
        """Terminal steps — never dispatched again (see class docstring:
        terminal ≠ satisfied-dependency)."""
        return self.succeeded | self.exhausted


def step_status(tool_results: list[dict[str, Any]] | None) -> StepStatus:
    st = StepStatus()
    for r in tool_results or []:
        sid = r.get("step_id")
        if not sid:
            continue
        sid = str(sid)
        if r.get("ok"):
            st.succeeded.add(sid)
        else:
            st.fail_counts[sid] = st.fail_counts.get(sid, 0) + 1
            if is_permanent_failure(r):
                st.permanent_failures.add(sid)
    return st


def pending_steps(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Plan steps that still need a (re)run: not succeeded, not exhausted."""
    done = step_status(state.get("tool_results")).done
    return [
        s for s in state.get("plan", []) or []
        if str(s.get("id")) not in done
    ]


def runnable_steps(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Pending steps whose dependencies all SUCCEEDED.

    A permanently-failed or retry-exhausted prerequisite does NOT
    satisfy its dependents — running them anyway would use args that
    were never resolved. Such dependents stay pending-but-unrunnable;
    when nothing is runnable, the executor declares the plan blocked
    and the router replans (tracker 0.6).

    Ordered untried-first (stable within equal fail counts) so a flaky
    step being retried never starves fresh work.
    """
    st = step_status(state.get("tool_results"))
    done = st.done
    pending = [
        s for s in state.get("plan", []) or []
        if str(s.get("id")) not in done
    ]
    runnable = [
        s for s in pending
        if all(str(d) in st.succeeded for d in (s.get("depends_on") or []))
    ]
    runnable.sort(key=lambda s: st.fail_counts.get(str(s.get("id")), 0))
    return runnable


def failed_step_notes(state: dict[str, Any]) -> list[str]:
    """One line per permanently-failed step of the *current* plan.

    Surfaced to the analyst so the absence of that evidence is never
    read as negative evidence (tracker 0.3).
    """
    st = step_status(state.get("tool_results"))
    by_id = {str(s.get("id")): s for s in state.get("plan", []) or []}
    dead = st.exhausted & set(by_id)
    if not dead:
        return []

    last_fail: dict[str, str] = {}
    for r in state.get("tool_results") or []:
        sid = str(r.get("step_id"))
        if sid in dead and not r.get("ok"):
            last_fail[sid] = str(r.get("data", ""))[:200].replace("\n", " ")

    notes = []
    for sid in sorted(dead):
        s = by_id[sid]
        goal = str(s.get("goal") or "").strip()
        notes.append(
            f"- {sid} ({s.get('tool', '?')}): {goal} — "
            f"FAILED: {last_fail.get(sid, 'unknown error')}"
        )
    return notes


# ---------------------------------------------------------------------------
# Truncated-run normalization (tracker 0.1 review)
# ---------------------------------------------------------------------------

def normalize_truncated_state(
    state: dict[str, Any] | None,
    *,
    reason: str,
    detail: str = "",
) -> dict[str, Any]:
    """Normalize the final state of a run that was cut short.

    Used by both runners after ``GraphRecursionError``: the checkpoint
    state still carries whatever the last analyst/critic pass produced,
    which would otherwise be reported as if the critic had accepted it.
    This stamps an explicit ``termination_reason``, caps the candidate
    confidence at ``TRUNCATED_CONFIDENCE_CAP``, appends an open question
    explaining the truncation, and rewrites the verdict decision so
    `write_report` and the CLI/UI surface the truth.
    """
    out = dict(state or {})
    out["termination_reason"] = reason
    note = f"Research truncated ({reason})" + (f": {detail}" if detail else ".")

    cand = dict(out.get("candidate_answer") or {})
    try:
        conf = float(cand.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    cand["confidence"] = min(conf, TRUNCATED_CONFIDENCE_CAP)
    cand.setdefault(
        "answer", "(run truncated before an answer was drafted)",
    )
    open_qs = list(cand.get("open_questions") or [])
    if note not in open_qs:
        open_qs.append(note)
    cand["open_questions"] = open_qs
    out["candidate_answer"] = cand

    verdict = dict(out.get("verdict") or {})
    prior = verdict.get("decision")
    if prior and prior != reason:
        verdict["prior_decision"] = prior
    verdict["decision"] = reason
    verdict["critique"] = (
        str(verdict.get("critique") or "") + f"\n\n_{note}_"
    ).strip()
    out["verdict"] = verdict
    return out
