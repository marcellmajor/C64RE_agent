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

# Bound native LangGraph fan-out so one over-eager plan cannot create an
# unbounded thread/API burst. Only fresh, concrete, read-only steps qualify;
# VICE and result-producing Code-KB modes stay serial.
MAX_PARALLEL_TOOL_STEPS = 4
_PARALLEL_CODE_KB_MODES = frozenset({
    "stats",
    "schema",
    "hardware",
    "routines",
    "routine",
    "pseudocode",
    "groups",
    "critiques",
    "xrefs_to",
    "xrefs_from",
    "smc",
    "writes_to",
    "refs_to",
    "hardware_refs",
    "search",
    "sql",
})

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

# Address-key aliases planner LLMs keep inventing for vice calls. Shared
# with graph.nodes (arg normalization) and `step_is_concrete` below.
VICE_ADDRESS_ALIASES = (
    "address", "addr", "start", "pc", "at", "from", "loc", "location",
    "entry", "target", "where",
)


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


def is_parallel_read_only_step(step: dict[str, Any]) -> bool:
    """Whether a plan step is safe for native concurrent dispatch.

    Planner order is not a dependency contract; `depends_on` is. Static dump
    analysis, parent-KB queries, Tavily searches, and explicitly read-only
    Code-KB modes may overlap once their dependencies are satisfied. Emulator
    calls and Code-KB modes that write evidence/files or invoke Layer-1 stay
    serial to preserve external-state and usage-accounting semantics.
    """
    tool = str(step.get("tool") or "").strip().lower()
    if tool in {"capstone", "kb", "tavily"}:
        return True
    if tool != "code_kb":
        return False
    args = step.get("args") or {}
    mode = str(args.get("mode") or "stats").strip().lower()
    return mode in _PARALLEL_CODE_KB_MODES


def parallel_runnable_steps(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return a bounded batch of fresh, concrete, read-only runnable steps."""
    status = step_status(state.get("tool_results"))
    batch = [
        step for step in runnable_steps(state)
        if str(step.get("id")) not in status.fail_counts
        and step_is_concrete(step)
        and is_parallel_read_only_step(step)
    ][:MAX_PARALLEL_TOOL_STEPS]
    return batch if len(batch) >= 2 else []


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
# Executor LLM bypass (tracker 1.1)
# ---------------------------------------------------------------------------

def _has_value(args: dict[str, Any], *keys: str) -> bool:
    return any(
        args.get(k) is not None
        and not (isinstance(args.get(k), str) and not args[k].strip())
        for k in keys
    )


def is_resolved_address(value: Any) -> bool:
    """True when *value* is a usable ``$0000``-``$FFFF`` address.

    Planner and executor LLMs sometimes write a symbolic forward
    reference where a concrete address belongs — ``"$DC00_REF_FROM_s5"``,
    ``"$C000_ROUTINE_START_FROM_s4"``. Those are non-empty strings, so
    the null check in `step_is_concrete` reads them as answered, the step
    skips enrichment, and the tool rejects the arg it can't parse.
    Requiring the value to actually parse keeps such steps on the
    enrichment path, where the KB can resolve what the reference meant.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0 <= value <= 0xFFFF
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if text.startswith("$"):
        candidates = ((text[1:], 16),)
    elif text.lower().startswith("0x"):
        candidates = ((text[2:], 16),)
    else:
        # `graph.nodes._hex_to_int` reads a bare token as decimal while
        # `_coerce_vice_address` reads it as hex. Accept either, so this
        # never rejects a value the tool itself would have parsed.
        candidates = ((text, 10), (text, 16))
    for body, base in candidates:
        try:
            parsed = int(body, base)
        except ValueError:
            continue
        if 0 <= parsed <= 0xFFFF:
            return True
    return False


def _has_address(args: dict[str, Any], *keys: str) -> bool:
    """`_has_value`, but the value must be a resolved address."""
    return any(is_resolved_address(args.get(k)) for k in keys)


def step_is_concrete(step: dict[str, Any]) -> bool:
    """True when a plan step can be dispatched without LLM enrichment.

    The executor LLM's only job is filling args the planner deferred
    (nulls, or references to prior discoveries). A step whose args are
    all present, non-null, and cover the tool/mode's required keys can
    go straight to the tool node — most planner steps arrive complete,
    so this skips roughly half the per-step LLM calls (tracker 1.1).

    Conservative by design: unknown tools and any null/empty arg values
    return False so the LLM enrichment path still runs for them.
    """
    tool = str(step.get("tool") or "").strip().lower()
    args = step.get("args")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return False

    # A null/empty value is the planner's "executor, fill this in".
    for v in args.values():
        if v is None or (isinstance(v, str) and not v.strip()):
            return False

    if tool == "kb":
        mode = str(
            args.get("mode") or ("sql" if args.get("sql") else "stats")
        ).strip().lower()
        required = {
            "sql": ("sql",),
            "text": ("q", "query"),           # either satisfies
            "text_semantic": ("q", "query"),
        }
        if mode in ("text", "text_semantic"):
            return _has_value(args, *required[mode])
        if mode == "sql":
            return _has_value(args, "sql")
        return True  # stats / schema / labels / events need nothing more

    if tool == "capstone":
        mode = str(args.get("mode") or "linear").strip().lower()
        if mode == "linear":
            return _has_address(args, "start")
        return True  # recursive/find_* / vectors / polymorphic self-default

    if tool == "tavily":
        return _has_value(args, "q", "query")

    if tool == "vice":
        method = str(
            args.get("method") or step.get("method") or "",
        ).strip().lower()
        if not method:
            return False
        if "screenshot" in method:
            return True
        # VICE composites (tracker 3.1 / 3.2).
        if "snapshot" in method:
            return _has_value(args, "name")
        if "monotonic_scan" in method:
            return _has_value(args, "snapshots")
        if method.endswith("diff") or method.endswith("memory.diff"):
            # `a` defaults to "dump"; `b` may be a live read — arg-free ok.
            return True
        if method.endswith("trace"):
            return _has_address(args, *VICE_ADDRESS_ALIASES)
        if method.endswith("poke_verify") or method.endswith("poke_and_peek"):
            return (
                _has_address(args, *VICE_ADDRESS_ALIASES)
                and _has_value(args, "value", "candidate")
                and _has_value(args, "expect", "expectation")
            )
        if "disassemble" in method:
            return _has_address(args, *VICE_ADDRESS_ALIASES)
        if "memory" in method and ("read" in method or "search" in method):
            return (
                _has_address(args, *VICE_ADDRESS_ALIASES)
                and _has_value(args, "size", "length")
            )
        # Breakpoints/watchpoints are address-bearing too (review
        # finding 4 — the old blanket True let a checkpoint step with no
        # address bypass enrichment straight into a tool error).
        if "checkpoint" in method or "breakpoint" in method:
            return _has_address(args, *VICE_ADDRESS_ALIASES)
        if "execution" in method or method in (
            "vice.ping", "ping", "run", "step", "pause",
        ):
            return True  # genuinely argument-free
        return False  # unknown vice methods → enrich conservatively

    if tool == "code_kb":
        mode = str(args.get("mode") or "").strip().lower()
        # Requirements mirror graph.code_kb_node's mode handlers EXACTLY
        # (review: a shared broad alias list accepted keys individual
        # handlers ignore — e.g. `xrefs_to` with only `src` bypassed
        # enrichment and then read a defaulted addr of 0).
        if mode in ("stats", "schema", "hardware", "routines", "smc",
                    "export", "hardware_refs", "groups", "critiques",
                    "layer2", "layer3"):
            # hardware_refs self-defaults to the full I/O range (3.4).
            return True
        if mode in ("routine", "pseudocode", "annotate"):
            # handlers read: start | addr | address
            return _has_address(args, "start", "addr", "address")
        if mode in ("xrefs_to", "writes_to", "refs_to"):
            # handlers read: addr | dst  (data-ref modes, tracker 3.4)
            return _has_address(args, "addr", "dst")
        if mode == "xrefs_from":
            # handler reads: addr | src
            return _has_address(args, "addr", "src")
        if mode == "disasm":
            # handler reads (vice engine): address | addr;
            # (capstone/default engine): start | addr
            engine = str(args.get("engine") or "capstone").strip().lower()
            if engine == "vice":
                return _has_address(args, "address", "addr")
            return _has_address(args, "start", "addr")
        if mode == "search":
            return _has_value(args, "q", "query")
        if mode == "sql":
            return _has_value(args, "sql")
        return False  # unknown/missing mode → enrich conservatively

    return False


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
