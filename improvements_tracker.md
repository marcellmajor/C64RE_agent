# C64-RE Agent — Improvements Tracker

**Purpose:** Living implementation plan for bug fixes and improvements drawn from the four review reports
(`fable`, `GPT5.5`, `Gemini3.1Pro`, `grok4.5highfast`) plus the synthesis. This file tracks progress
step-by-step. It is intended to be reviewed (by the user and by GPT 5.6) before and during implementation.

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked/needs decision · `[-]` deferred/won't-do

**Verification tag:** ✅ = claim independently confirmed against current code during planning. Line numbers are from the
working tree at planning time and should be re-checked at edit time.

**Source key:** F=fable, G=GPT5.5, Ge=Gemini3.1Pro, Gr=grok4.5, S=synthesis. "(all)" = 3+ reports agree.

---

## How this tracker is organized

Work is grouped into phases ordered so that correctness fixes land before the efficiency and capability work
that would otherwise be measured on a broken loop. Each phase is independently shippable and testable. Within a
phase, items are roughly ordered by (impact ÷ effort).

- **Phase 0 — Control-loop correctness** (gates everything; runs currently truncate/misbehave)
- **Phase 1 — Efficiency & cost** (cut LLM calls ~half; make cost visible)
- **Phase 2 — Persistence & curator honesty** (stop stale facts + permanent curator tax)
- **Phase 3 — RE technique gains** (biggest answer-quality jumps)
- **Phase 4 — Correctness/robustness papercuts** (cheap, high-value fixes)
- **Phase 5 — Observability & evaluation** (make every change above measurable)
- **Phase 6 — Product/UX** (research-notebook direction; larger, optional)
- **Phase 7 — Hygiene/packaging**

A per-item **"My call"** note states whether I judge it worth doing now, later, or skipping — the user asked for
the improvements *I* see as valuable, not a verbatim copy of all four reports.

## Implementation status table

This is the coding agent's progress ledger and should be updated in the same change that starts, completes, blocks,
or defers an item. Keep its status consistent with the matching detailed entry below. Valid status values are the
ones in the legend above.

| ID | Recommended bug fix / improvement | Status |
|---|---|---|
| 0.1 | Raise LangGraph's recursion limit and still emit a best-effort report if recursion is exhausted. | [x] Done |
| 0.2 | Compare KB growth with the previous critic snapshot so the dead-end detector does not stop productive replans. | [x] Done |
| 0.3 | Retry failed tool steps with a cap and report persistent failures instead of marking every attempted step complete. | [x] Done |
| 0.4 | Cap consecutive analyst/critic revisions so the revise loop cannot run indefinitely. | [x] Done |
| 0.5 | Distinguish blocking critic gaps from optional follow-ups so accepted answers are not incorrectly revised. | [x] Done |
| 0.6 | Detect dependency cycles or unsatisfied dependencies and replan instead of running blocked steps with null arguments. | [x] Done |
| 0.7 | Use one canonical game slug everywhere while preserving aliases for existing session directories. | [x] Done |
| 1.1 | Skip the executor LLM when a plan step already has complete, concrete tool arguments. | [x] Done |
| 1.2 | Extract structured tool results deterministically and batch the remaining free-text LLM synthesis work. | [x] Done |
| 1.3 | Cap and prioritize automatic Layer-1 routine annotations to prevent hidden LLM-call fan-out. | [x] Done |
| 1.4 | Track real per-role token, cost, LLM-call, tool-call, and VICE-call usage against the run budget. | [x] Done |
| 1.5 | Reuse and memoize KB digests, partial-asm metadata, and question embeddings instead of rebuilding them per node. | [x] Done |
| 1.6 | Replace repeated full event-log scans with indexed lookups and replay high-water marks. | [x] Done |
| 1.7 | Run independent read-only plan steps in parallel with LangGraph fan-out after batching and retry semantics stabilize. | [-] Deferred |
| 2.1 | Compact only new, uncompacted events and expose consolidated observations in question digests. | [x] Done |
| 2.2 | Detect changed dump and assembly inputs by content metadata and invalidate stale derived analysis. | [x] Done |
| 2.3 | Store routine confidence and prevent lower-confidence replay events from overwriting stronger facts. | [x] Done |
| 2.4 | Implement durable CLI checkpoint resume with `SqliteSaver`, or correct the resume claim if KB-only persistence is retained. | [x] Done |
| 2.5 | Persist semantic vectors and embed only cache misses once semantic search is enabled by default. | [-] Deferred |
| 3.1 | Add named memory snapshots, diffs, and monotonic scans for locating changing game-state variables. | [ ] Not started |
| 3.2 | Add a safe VICE watchpoint/trace workflow and allow register reads when a checkpoint is armed. | [ ] Not started |
| 3.3 | Add deterministic 6502 counter heuristics for lives, score, timers, HUD digits, and similar state. | [ ] Not started |
| 3.4 | Index data reads/writes and zero-page use-def relationships, with query modes for references and hardware access. | [ ] Not started |
| 3.5 | Pre-tag common C64/6502 code idioms before Layer-1 LLM annotation. | [ ] Not started |
| 3.6 | Make disassembly bank-aware, avoid false ROM assumptions, and improve illegal-opcode handling. | [ ] Not started |
| 3.7 | Decode PETSCII, screen RAM, and color RAM into searchable KB text. | [ ] Not started |
| 3.8 | Explore refined loop scoring, pseudocode decompilation, and multimodal poke-and-peek after prerequisite work. | [-] Deferred |
| 4.1 | Make BASIC `SYS` detection tolerate spaces and common separators before the entry-point digits. | [ ] Not started |
| 4.2 | Require an explicit Capstone linear-disassembly start address instead of silently using `$0801`. | [ ] Not started |
| 4.3 | Match hexadecimal addresses from the question when retrieving relevant KB labels. | [ ] Not started |
| 4.4 | Give screenshot analysis a dedicated neutral vision prompt and preserve usable failure/output metadata. | [ ] Not started |
| 4.5 | Encode VICE memory reads as compact hex dumps before truncation so most returned bytes are not lost. | [ ] Not started |
| 4.6 | Broaden configurable Tavily research domains and support advanced search depth. | [ ] Not started |
| 4.7 | Honor per-role output-token limits so planner and analyst JSON is not truncated mid-response. | [ ] Not started |
| 4.8 | Bound or compact `tool_results` in graph state after persisted results no longer need full payloads. | [-] Deferred |
| 5.1 | Add a golden-question regression suite that checks expected evidence, confidence, cost, and runtime. | [ ] Not started |
| 5.2 | Add deterministic unit tests and compact synthetic dump fixtures for core parsing, routing, and disassembly behavior. | [ ] Not started |
| 5.3 | Add a per-role calls, tokens, failures, and fallback-activation table to generated reports. | [x] Done (delivered with 1.4) |
| 5.4 | Persist one queryable run-summary event containing outcome, iterations, confidence, and cost. | [ ] Not started |
| 5.5 | Add human ratings and LangSmith datasets after the product/UI work is ready to collect them. | [-] Deferred |
| 6.1 | Add human approval/edit interrupts for plans and mutating VICE operations, plus UI cancellation. | [ ] Not started |
| 6.2 | Turn sessions into a multi-turn research notebook with archived turns, versioned reports, and prior-answer context. | [ ] Not started |
| 6.3 | Track hypotheses through open, supported, and refuted lifecycle states as evidence changes. | [ ] Not started |
| 6.4 | Share one evidence-reference resolver between the CLI and Streamlit UI. | [ ] Not started |
| 6.5 | Catalog named dumps from multiple game states so offline snapshot comparisons are reproducible. | [ ] Not started |
| 6.6 | Export high-confidence labels as VICE-compatible symbol maps. | [ ] Not started |
| 6.7 | Allow synthesizer-discovered Layer-1 facts into Code KB with a verified window or explicit unverified provenance. | [ ] Not started |
| 6.8 | Revisit DAG planning, deeper Code-KB layers, and cross-provider structured output after loop stabilization. | [-] Deferred |
| 7.1 | Package all runtime Python modules and configuration data instead of shipping only `graph/`. | [ ] Not started |
| 7.2 | Remove unused coordinator/researcher role configuration or wire those roles into real graph paths. | [ ] Not started |
| 7.3 | Redact secret-like values from digests and SQL previews and document trusted note inputs. | [ ] Not started |

---

## Phase 0 — Control-loop correctness (do first)

These four+ defects silently cap or distort *every* run. Fix and add a regression test for each before touching
anything downstream, because otherwise the efficiency/technique work is measured on a broken loop.

> **STATUS: Phase 0 implemented (2026-07-16); GPT 5.6 Sol review-round fixes applied (2026-07-17) — see
> "Phase 0 implementation notes" at the end of this section, including the "Review-round changes" list.**
> 64 unit/integration tests in `tests/` pass; `compileall` clean; `git diff --check` clean; graph compiles;
> `main.py` / `app.py` / `agent_runner.py` import cleanly.

- [x] **0.1 Set `recursion_limit` + emit a best-effort report on `GraphRecursionError`** ✅
  - **Bug:** Neither entry point raises LangGraph's default 25-super-step limit. `main.py:206` and
    `tools/agent_runner.py:144` pass only `{"configurable": {"thread_id": ...}}`. Each plan step costs 3 super-steps
    (executor→tool→synthesizer); a 7-step plan + load/plan/analyst/critic ≈ 25, so the **first** iteration exhausts
    the budget and any replan/revise raises `GraphRecursionError` — aborting *without* writing a report.
    `MAX_ITERS = 12` (`graph/routers.py:23`) is unreachable.
  - **Fix:** pass `recursion_limit` (~400) in both runners; wrap the stream in a try/except that, on
    `GraphRecursionError`, still calls the report path from the last good state (degrade to a low-confidence answer).
  - **Files:** `main.py`, `tools/agent_runner.py` (and optionally a shared helper).
  - **Effort:** ~1h · **Impact:** Critical · **Source:** F §1.1, S
  - **My call:** Do first. Nothing else can be trusted until runs actually complete.

- [x] **0.2 Fix the vacuous dead-end detector (snapshot ordering)** ✅
  - **Bug:** `_dead_end_detected` (`graph/routers.py:53-64`) wants "two consecutive replans AND KB stopped growing,"
    but `critic_node` (`graph/nodes.py:1625-1629`) writes `last_kb_event_count = events_total` in the *same* state
    update the router then reads. Nothing writes events between critic and router, so `count <= last_count` is always
    true — every second consecutive replan ends as `budget_exceeded` even when the intervening tools added evidence.
  - **Fix:** compare against the snapshot from the *previous* verdict. Compute `kb_grew_since_last_verdict` inside
    `critic_node` *before* overwriting the snapshot and emit it into the verdict; have the router read that boolean.
    (Alternative: keep a 2-deep history of counts in state.)
  - **Files:** `graph/nodes.py`, `graph/routers.py`, possibly `graph/state.py`.
  - **Effort:** ~1h · **Impact:** High · **Source:** F §1.2, S
  - **My call:** Do first.

- [x] **0.3 Failed tool steps are treated as "done"** ✅
  - **Bug:** `_pending_steps` (`graph/routers.py:27-29`) and the executor's `done` set (`graph/nodes.py:1016`) are
    built from `tool_results` step IDs with **no `ok` check**. A missing `TAVILY_API_KEY`, a banned VICE method, or
    bad SQL permanently "completes" the step; the loop advances as if it succeeded, and the analyst can read absence
    of evidence as negative evidence.
  - **Fix:** treat `ok=False` infra failures as retryable with a per-step cap (e.g. 2), distinct from "ran, empty
    result." Surface still-failing steps in the digest/report so they aren't silently swallowed.
  - **Files:** `graph/routers.py`, `graph/nodes.py`, `graph/state.py` (retry counters).
  - **Effort:** ~half day · **Impact:** High (prevents silent research corruption) · **Source:** Gr §1.1
  - **My call:** Do first — cheap and it directly corrupts answers today.

- [x] **0.4 Cap the `revise` loop** ✅ (uncapped confirmed)
  - **Bug:** `verdict_router` has no consecutive-`revise` counter; `iteration` only bumps in the planner, so
    analyst↔critic (incl. the auto-guardrail at `graph/nodes.py:1609`) can ping-pong until recursion kills the run.
  - **Fix:** track `revise_count` like `replan_count`; after 2–3 consecutive revises, force `accept` (attach critique
    to report) or escalate to `replan`.
  - **Files:** `graph/state.py`, `graph/nodes.py` (critic), `graph/routers.py`.
  - **Effort:** ~2h · **Impact:** Medium-High · **Source:** F §1.5
  - **My call:** Do first — bundles naturally with 0.1/0.2.

- [x] **0.5 Critic `accept` + `suggested_steps` is silently demoted to `revise`** ✅
  - **Bug:** `graph/nodes.py:1603-1604` — comment says "Force `replan`" but code sets `decision = "revise"` whenever
    `suggested` is non-empty on an otherwise-`accept`. Productive accepts bounce back into revise ping-pong.
  - **Fix:** split critic output into **blocking gaps** vs **optional next steps**. `accept` + only-optional →
    write report, append suggestions as open questions / follow-up stubs. `accept` + blocking tool-call needs →
    force `replan` (match the comment).
  - **Files:** `graph/nodes.py`, `graph/prompts.py` (critic contract).
  - **Effort:** ~half day · **Impact:** Medium-High · **Source:** Gr §1.2
  - **My call:** Do — orthogonal to 0.4 and removes a real stall.

- [x] **0.6 Dependency-cycle fallback runs blocked steps with null args** ✅
  - **Bug:** When no step is runnable but some remain, the executor picks FIFO pending anyway
    (`graph/nodes.py:1027-1033`), often with unresolved null addresses from unmet deps.
  - **Fix:** on unrunnable-but-pending, fail the plan with a structured `dependency_unsatisfied`/`cycle` error and
    force `replan` rather than dispatching a guaranteed-bad call.
  - **Files:** `graph/nodes.py`, `graph/routers.py`.
  - **Effort:** ~2h · **Impact:** Medium · **Source:** Gr §1.4
  - **My call:** Do — small, prevents wasted cycles.

- [x] **0.7 Unify the game-slug function (hyphen vs underscore)** ✅
  - **Bug:** CLI `thread_id` default uses hyphens (`main.py:205` `replace(" ", "-")`); session dirs / `_slug` / UI use
    underscores (`graph/nodes.py:444`). Resume/paths can point at the wrong place.
  - **Fix:** one canonical `slugify()` used everywhere (sessions, thread ids, prefs, report paths). Prefer underscores
    to match existing `sessions/` trees; alias old hyphen dirs for back-compat.
  - **Files:** shared util, `main.py`, `graph/nodes.py`, `app.py`, `tools/agent_runner.py`.
  - **Effort:** ~2h · **Impact:** Medium · **Source:** Gr §1.5
  - **My call:** Do — trivial and prevents confusing "lost session" reports.

### Phase 0 implementation notes (2026-07-16; review-round fixes applied 2026-07-17)

> A GPT 5.6 Sol review of the first pass required six corrections — all applied and re-tested (see
> "Review-round changes" below for what changed vs. the first pass). Full suite: **64 tests passing**;
> `compileall` clean; `git diff --check` clean.

**New module `graph/plan_utils.py`** — pure, unit-testable helpers importable by both `nodes` and `routers`
without a cycle: `slugify()` / `resolve_session_slug()`, `step_status()` / `pending_steps()` / `runnable_steps()` /
`failed_step_notes()`, `is_permanent_failure()`, `normalize_truncated_state()`, and the loop constants
`MAX_ITERS=12` (moved here; re-exported by `graph.routers`), `MAX_PLAN_STEPS=12`, `MAX_STEP_ATTEMPTS=2`,
`MAX_CONSECUTIVE_REVISES=2`, `RECURSION_LIMIT` (derived — see 0.1), `TRUNCATED_CONFIDENCE_CAP=0.4`.

Per item (final behavior):

- **0.1** — Both runners pass `recursion_limit: RECURSION_LIMIT`. The limit is **derived**, not guessed:
  `MAX_ITERS × (MAX_PLAN_STEPS × MAX_STEP_ATTEMPTS × 4 + 3 + MAX_CONSECUTIVE_REVISES × 2) + 2 = 1238`, where 4 is
  the worst-case super-steps per step (executor→tool→synthesizer→**curator**) and the planner enforces
  `MAX_PLAN_STEPS` (truncating oversized plans and dropping dangling deps), making the bound real. On
  `GraphRecursionError`, both runners **normalize** the checkpoint state via `normalize_truncated_state()`:
  explicit `termination_reason="recursion_exhausted"`, verdict decision rewritten (prior decision preserved),
  candidate confidence capped at 0.4, and a "Research truncated…" open question appended — and that normalized
  state feeds both `write_report()` and the CLI print / `TurnResult`, so a truncated run can't masquerade as a
  validated answer.
- **0.2** — `critic_node` compares against the snapshot from the *previous* verdict **before** overwriting it and
  ships `kb_grew_since_last_verdict` on the verdict; `_dead_end_detected` reads the flag (default alive).
  Growth is measured as **distinct substantive facts**, not raw event count: bookkeeping kinds
  (analysis/verdict/consolidated) never count; **failed tool attempts never count**; tool results are keyed on
  `(tool, data prefix)` — *not* step_id — so a replan re-running the identical call adds nothing; derived facts key
  on stable payload identity (label `addr+name`, routine `start+end`, hypothesis text). `load_inputs` seeds the
  same measure. (O(events) per verdict for now — indexing folded into 1.6.)
- **0.3** — A step is *terminal* when it succeeded or is exhausted (≥2 failed attempts, or `retryable: False` at
  the source / `rejection ∈ {banned_method, dependency_unsatisfied}`). Retries run untried-steps-first and the
  executor LLM sees the last error on retry. `analyst_node` gets a "steps that FAILED — evidence never gathered"
  block. **Terminal ≠ satisfied**: see 0.6.
- **0.4** — `revise_count` in state; cap applied **last** in `critic_node` so no demotion path can ping-pong past
  it; 3rd consecutive revise forces `accept` with the note in the critique.
- **0.5** — `suggested_steps` are **blocking** tool work: `accept`+suggested and `revise`+suggested both
  auto-escalate to `replan` (steps flow to the planner). Only `optional_followups` may accompany `accept`; they
  are merged into the candidate's `open_questions` so they reach the report. `CRITIC_ROLE` documents both fields
  and the escalation.
- **0.6** — Only a **succeeded** dependency satisfies `depends_on`; permanently-failed or retry-exhausted
  prerequisites block their dependents. When pending steps exist but none are runnable (failed prerequisite,
  missing dep id, or cycle), the executor records one structured
  `ok=False, retryable=False, rejection="dependency_unsatisfied"` result per blocked step (with a per-dependency
  reason) and raises `plan_blocked` in state; `route_tool` then routes **deterministically to `planner`** for a
  replan (new `executor→planner` edge), with the blocked reasons injected into the planner prompt. Blocked-plan
  replanning is bounded: past `MAX_ITERS` the route falls to synthesizer→analyst→critic→`budget_exceeded` and the
  report is written.
- **0.7** — Canonical `slugify()` used by `graph/nodes`, `main.py` (thread id + read paths), `tools/agent_runner.py`,
  `app.py`, `scripts/purge_persistence.py` (lazy import + fallback; also fixes empty-game → `""` which could have
  targeted `sessions/` itself), and `code_kb/scoping.py` (lazy import — `code_kb` must not import `graph` at module
  level or it would cycle through `graph.build`). Legacy compatibility via `resolve_session_slug()`: read/extend
  paths (session KBs, code-KB lookup, report paths, purge targets) prefer the canonical slug but fall back to an
  existing hyphen/underscore-variant directory, so previously-created sessions keep working.

**Review-round changes vs. the first pass** (what the GPT 5.6 Sol review corrected):
1. 0.6: exhausted deps previously *satisfied* dependents and blocked plans drifted to analyst/critic — now
   failed prerequisites block, and blocked plans replan deterministically (bounded by `MAX_ITERS`).
2. 0.1: recovery previously reported the raw checkpoint state — now normalized (reason/confidence-cap/open
   question) in both runners.
3. 0.5: `accept`+`suggested_steps` previously converted the steps to optional follow-ups — now escalates to
   `replan`; only `optional_followups` accompany accept.
4. 0.2: growth previously counted raw substantive events (failed attempts and re-recorded duplicates inflated
   it) — now distinct facts with step-id-independent identity.
5. 0.1: `RECURSION_LIMIT` was a flat 400 ignoring the curator path — now derived (1238) with `MAX_PLAN_STEPS`
   enforced.
6. 0.7: `scripts/purge_persistence.py`, `code_kb/scoping.py`, and two leftover `app.py` sites still had manual
   slugs — now centralized, with legacy-dir resolution and tests.

**Tests:** `tests/` — **64 passing**: `test_plan_utils.py` (step status, retries, permanence, dependency blocking,
ordering, slugify + legacy resolution, RECURSION_LIMIT derivation, truncation normalizer), `test_routers.py`
(dead-end scenarios, route_tool blocked/complete paths, verdict_router caps), `test_executor_blocked.py`
(blocked/complete paths never touch the LLM + **integration routing**: executor output → reducers → route_tool for
cycles, missing deps, failed prerequisites, iteration cap), `test_critic_guardrails.py` (0.4/0.5 guardrails, mocked
LLM), `test_kb_growth.py` (real KnowledgeStore: bookkeeping/failures/duplicates don't count, new facts do),
`test_recursion_recovery.py` (fake graph raising `GraphRecursionError` through **both** runners),
`test_slug_paths.py` (purge targets incl. legacy dirs + empty-game safety, scoping delegation, UI-helper
resolution). `pytest>=8` in the `dev` extra.

**Known deferrals within Phase 0 scope:** failed steps are surfaced in the analyst prompt and the report's
tool-result list, but not yet as a dedicated digest section (belongs with the 2.1 digest work). The distinct-fact
growth scan is O(events) per verdict — indexing belongs to 1.6. `RECURSION_LIMIT=1238` supersedes open question 2.

---

## Phase 1 — Efficiency & cost

Roughly halves LLM calls per iteration and makes spend visible. Do after Phase 0 so savings are measured on a loop
that actually completes.

- [x] **1.1 Skip the executor LLM for already-concrete steps** ✅ (LLM-per-step confirmed)
  - **Now:** `executor_node` (`graph/nodes.py:1001-1083`) makes an LLM call for *every* step (with an 8k-char digest),
    even `{"mode":"stats"}` or fully-specified capstone steps.
  - **Fix:** per-tool/mode `is_step_complete(step)` validator; call the executor LLM only when a required arg is
    missing/null or references prior discoveries. Record `executor_llm_skipped` in the message for observability.
    (Mechanical dependency selection already runs without the LLM.)
  - **Files:** `graph/nodes.py`, `graph/prompts.py`.
  - **Effort:** ~half day · **Impact:** High · **Source:** F §2.1, G #1, S (strong overlap)
  - **My call:** Do — highest-ROI first patch per GPT.

- [x] **1.2 Deterministic synthesis for structured tool output; batch the LLM rest** ✅ (all 3 reports)
  - **Now:** `synthesizer_node` runs an LLM extraction after *every* step, including structured JSON (`capstone
    find_entry/vectors/find_loops`, `code_kb` rows, `kb` rows), failed steps, and `kb stats`.
  - **Fix:** mechanical extractors for structured modes (find_loops→hypotheses, vectors→labels, code_kb routine
    windows are already KB content); invoke the LLM only for free-text (disassembly listings, Tavily snippets,
    screenshot descriptions). Accumulate + synthesize once per *batch* of completed steps. Add an extraction
    provenance field (`mechanical|llm|hybrid`).
  - **Files:** `graph/nodes.py`, `tools/c64_disasm.py`, `graph/code_kb_node.py`, `graph/state.py`.
  - **Effort:** ~1 day · **Impact:** High (highest cross-report consensus) · **Source:** F §2.2, G #3, Ge §1.3, S
  - **My call:** Do — pairs with 1.1 as the flagship efficiency patch.

- [x] **1.3 Cap the hidden Layer-1 auto-annotate fan-out** ✅
  - **Now:** synthesizer fires `_mode_annotate` (a full LLM call, possibly with auto-disasm) for *every* routine with
    confidence ≥ 0.5 (`graph/nodes.py:1283-1303`) — six routines = six invisible extra LLM calls.
  - **Fix:** cap at 1–2 per synthesizer invocation, prioritized by (confidence × question-term relevance); log each as
    a transcript message and count it against budget (see 1.4).
  - **Files:** `graph/nodes.py`.
  - **Effort:** ~2h · **Impact:** Medium-High · **Source:** F §2.3
  - **My call:** Do — silent unbounded cost.

- [x] **1.4 Real budget/cost accounting + per-role report table** ✅ (dead code confirmed)
  - **Now:** `budget_used` is checked (`graph/routers.py:98`) but only ever set to `0.0` in `load_inputs`
    (`graph/nodes.py:557`). Zero token telemetry.
  - **Fix:** read `usage_metadata` off each `AIMessage` in `_invoke_one`; accumulate (input,output) tokens per role;
    convert via a small price table (or track tokens); return a `budget_used` delta with an `operator.add` reducer on
    the field. Add `llm_calls`/`tool_calls`/`vice_calls` counters. Surface a per-role table in `write_report`.
  - **Files:** `graph/nodes.py`, `graph/llm.py`, `graph/routers.py`, `graph/state.py`.
  - **Effort:** ~half day · **Impact:** High (unblocks measuring everything else) · **Source:** F §1.4, G #4a, S
  - **My call:** Do early in Phase 1 so 1.1/1.2 savings are quantifiable.

- [x] **1.5 Stop rebuilding the digest / re-embedding per node** ✅ (partial-asm scan confirmed hot)
  - **Now:** `planner_node` and `executor_node` both call `_kb_digest_for_state` (full SQL rebuild) though
    `state["kb_digest"]` is refreshed by the synthesizer each step; every digest build calls `partial_asm_excerpt`
    which re-reads/JSON-parses the entire `kb.json` (O(events) per digest); semantic-on re-embeds the question each node.
  - **Fix:** reuse `state["kb_digest"]` in planner/executor; cache the partial-asm path in `meta` at ingest;
    memoize digest keyed on `(question, events_total)`; memoize the question embedding.
  - **Files:** `graph/nodes.py`, `memory/store.py`.
  - **Effort:** ~half day · **Impact:** Medium (scales with KB) · **Source:** F §2.4
  - **My call:** Do — cheap once budget telemetry shows the waste.

- [x] **1.6 Remove O(N) full-log scans on hot paths** ✅ (dedup scan confirmed)
  - **Now:** synthesizer stage-1 dedup loads+parses *all* `tool_result` events every call (`graph/nodes.py:1107-1116`);
    `ingest_text_file`/`ingest_dump`/`ingest_partial_asm`/`load_or_init` do full scans at startup.
  - **Fix:** index dedup keys (`(tool, step_id, hash(data))`); query `text_docs` by `path/mtime`; store latest-dump
    metadata in `meta`; keep a `meta.last_replayed_event_id` high-water so `load_or_init` doesn't full-replay every start.
  - **Files:** `memory/store.py`, `memory/schema.py`, `graph/nodes.py`.
  - **Effort:** ~1 day · **Impact:** Medium · **Source:** F §2.5, G #9
  - **My call:** Do after 1.5; defer the checkpoint-snapshot half if time-boxed.

- [-] **1.7 Parallel fan-out (LangGraph `Send`) for independent read-only steps**
  - **Source:** F §2.8, G #2 · **Effort:** ~1 day+
  - **My call:** Defer. Real wall-clock win, but it interacts with the batch-synthesis and failed-step-retry changes;
    land those first, then revisit so we don't design the batching twice.

### Phase 1 implementation notes (2026-07-17; hardening round + final hardening pass applied — for reviewer)

> **STATUS: Phase 1 implemented (1.1–1.6; 1.7 stays deferred), hardened per the GPT 5.6 Sol review
> (6 findings — "Hardening-round changes" below), and re-hardened per the final review's 3 reproduced
> probes ("Final hardening pass" below — all three probes are now permanent passing tests).
> Full suite: 182 tests passing; `compileall` clean; `git diff --check` clean; graph compiles; both
> runners import. Telemetry is trustworthy under nested (Layer-1), threaded, AND asyncio execution —
> the condition set for keeping 1.4/5.3 marked complete.**

**New module `graph/usage.py`** (1.4) — thread-safe usage collector + cost estimation + per-role aggregation.
Every LLM call (`_invoke_one`, the screenshot vision path, failures included) records
`{role, as_role, model, input/output_tokens, cost_usd, ok, error}`; each node drains pending entries at return
time (`_usage_update()` in nodes.py) into two new state fields: `llm_usage` (add-reducer list) and `budget_used`
(now an **add-reducer float** actually incremented — was dead code). The router's `BUDGET_CAP` check is live once
pricing exists. **Cost policy (deliberate):** no fabricated price table — `config/llm.json` MAY define an optional
top-level `"pricing"` map (`model-prefix → {input_per_mtok, output_per_mtok}`, longest prefix wins); without it,
costs report 0.0 and tokens are still tracked (the report says so explicitly). I did **not** edit the user's
`config/llm.json`. The report gained `## LLM usage (per role)` (calls/failures/backup-activations/tokens/USD —
this also delivers item 5.3) and `## Tool calls` (per-tool call/failure counts; vice calls visible per-tool).

Per item:

- **1.1** — `plan_utils.step_is_concrete(step)`: per-tool/mode required-arg validator (kb sql/text need
  `sql`/`q`; capstone `linear` needs `start`; tavily needs `q`; vice disassemble/memory.read need an address
  alias (+`size`); any null/empty arg value returns False). The executor dispatches concrete steps directly —
  no LLM call, no digest build — logging "(LLM skipped — args already concrete)". **Retries always go through
  the LLM** so it can repair args from the recorded failure. The planner's own fallback plan is now fully
  LLM-free at the executor. `VICE_ADDRESS_ALIASES` moved to plan_utils (shared with nodes' arg normalizer).
- **1.2** — Synthesizer split into: stage 1 record (indexed dedup, see 1.6) → stage 1.5 **mechanical extraction**
  (capstone `vectors` → vector labels; `find_loops` candidates → deterministic-id hypotheses `h_loop_<addr>`;
  payloads carry `provenance: "mechanical"`) → stage 2 **batched LLM extraction**: LLM-worthy results (free text
  only — failed results and `kb`/`code_kb` reads are never LLM'd; `vectors`/`find_entry` are mechanical-only;
  `find_loops` is hybrid because it embeds an auto-disassembly listing) accumulate across steps via a
  `synth_processed_count` index high-water in state and flush in ONE call when the plan drains (analyst next)
  or the batch fills (`SYNTH_BATCH_MAX_RESULTS=4` / `SYNTH_BATCH_MAX_CHARS=20k`). LLM-extracted payloads carry
  `provenance: "llm"`. On a pure defer the digest rebuild is also skipped.
- **1.3** — The per-routine auto-`_mode_annotate` calls are queued as candidates, ranked by
  confidence × (1 + question-term relevance), capped at `MAX_AUTO_ANNOTATE_PER_SYNTH=2` per synthesizer pass,
  and logged in the transcript (including how many candidates were deferred) instead of firing silently.
- **1.4** — See `graph/usage.py` above.
- **1.5** — Planner and executor reuse `state["kb_digest"]` (≤1 step stale) instead of rebuilding;
  `digest_for_question` memoized on `(question, events_total, max_chars)` (events_total covers all derived
  content, so growth self-invalidates); `partial_asm_excerpt` resolves via the events *table* + caches the head
  (was: re-read + JSON-parse the entire kb.json per digest build); the semantic question-embedding is memoized
  per query text (bounded cache).
- **1.6** — (a) synthesizer dedup via new derived table `tool_result_keys(key PK, event_id)` where key =
  sha256(tool|step_id|**full** data) — indexed lookup instead of parsing every tool_result event per pass, and
  full-content identity matches the 0.2 review direction (a 256-char-prefix would drop evidence past it);
  (b) `ingest_dump`/`ingest_partial_asm` idempotency and `_warm_dump_cache` now query the derived events table;
  `ingest_text_file` checks `text_docs.path/mtime` directly (was O(files × events)); (c) **incremental replay**:
  the derived SQLite view persists across processes with `meta.schema_version` + `meta.replay_offset` (byte
  high-water into kb.json, advanced on every append); startup replays only the JSONL tail; full rebuild on
  version mismatch, truncated/replaced log, or a corrupt tail (`SCHEMA_VERSION="2"` bumped in memory/schema.py —
  existing derived DBs rebuild once, transparently). Side benefit: two processes sharing a KB no longer clobber
  each other's derived rows at open (the old code dropped all tables on every start).

**Tests:** 182 passing (113 added across Phase 1 + both hardening rounds): `test_executor_skip.py`
(concrete-step table incl. mode-exact `code_kb` aliases with positive AND negative cases per address-bearing
mode, vice checkpoint/execution cases, bypass/retry integration), `test_synth_batch.py` (defer/flush/batch-fill,
unworthy results never reach the LLM, mechanical vectors-only, hybrid find_loops, no duplicate mechanical facts,
annotate cap + relevance priority, **annotate usage reaches the state update**, **failed annotation not logged
as success** — real KB + code-KB stores), `test_usage.py` (extraction, longest-prefix pricing, honest-zero for
unknown models, aggregation, success/transport-failure/backup recording, **empty/non-JSON/invalid-contract
responses counted as failures even when salvaged**, **thread AND asyncio contexts cannot cross-drain** (the
reviewer's exact repro), `reset()` behaviour, copy-on-write `mark_failed` amendment, **wrong-shaped dicts
rejected per role / valid minimal contracts accepted / envelope unwrap / invalid critic decision**, **fallback
vs backup activation distinct + counted exactly once**, report table + budget block content), `test_routers.py`
(+ **token budget terminates an unpriced over-budget run**, env resolution), `test_store_perf.py` (incremental
reopen without full rebuild, foreign-tail replay, truncation/schema-mismatch rebuilds, **same-size log
replacement and prefix-edit rebuilds**, full-content dedup identity, ingest idempotency incl. mtime refresh,
digest/partial-asm memoization).

**Hardening-round changes (GPT 5.6 Sol review of Phase 1 — all six findings fixed, 2026-07-17):**
1. **Nested Layer-1 telemetry no longer discarded.** `_mode_annotate()` returns through `_record_result()`,
   whose drain had moved the Layer-1 usage entries into a return value the synthesizer threw away. The
   synthesizer now captures that return, **re-records** its `llm_usage` entries so its own drain ships them to
   state, and inspects the nested tool result: "auto-annotated $XXXX" is logged only on `ok=True`; handled
   failures log "auto-annotation of $XXXX FAILED: <reason>" (crashes too). Cap of 2/pass preserved.
2. **Usage collection is invocation-local.** `graph/usage.py` replaced the module-global pending list with a
   `contextvars.ContextVar` bucket: concurrent Streamlit/LangGraph runs (threads, or asyncio tasks with copied
   contexts) cannot drain each other's entries; nested calls within a node still land in that node's drain.
   Residual (documented): a reused executor thread can inherit stragglers only if a node crashed between
   record and drain — `drain()` always clears, so they never accumulate.
3. **Token-budget fallback.** `tokens_used` (new add-reducer state field, distinct from USD `budget_used` — no
   unit mixing) always accrues; `verdict_router` enforces `routers.token_budget()`
   (env `C64RE_TOKEN_BUDGET` → config `defaults.token_budget` → `DEFAULT_TOKEN_BUDGET = 3,000,000`).
   The report's Budget block shows total tokens vs token cap, est. USD vs USD cap, priced/unpriced call
   counts, and **which limit was enforced** — so unpriced models are protected at runtime.
4. **`step_is_concrete` per-mode validation.** `code_kb` now mirrors the real mode handlers: metadata modes
   (stats/schema/hardware/routines/smc/export) are arg-free; routine/annotate/disasm/xrefs_to/xrefs_from
   require an address; search requires `q`; sql requires `sql`; unknown modes → enrich. VICE
   checkpoint/breakpoint methods now require an address (the old blanket-True false positive); unknown vice
   methods → enrich. Retries still always use the LLM.
5. **Honest failure accounting.** A usage entry's `ok` now means "produced a usable role-contract response";
   `transport_ok` tracks HTTP success separately. Empty bodies are demoted at record time; `_safe_invoke`
   demotes non-JSON / non-contract responses in place via `usage.mark_failed()` (the salvaged raw-text
   fallback is still handed to the caller but counted as a failure). Same treatment in the screenshot-vision
   path (empty description = failure) and `_invoke_layer1` (empty/non-JSON = failure).
6. **Replaced-log detection in incremental replay.** The derived view persists a `replay_fingerprint`
   (sha256 over the first and last 4KB of the replayed prefix + the offset) alongside `replay_offset`;
   same-size replacement, head edits, and edits near the replay boundary all trigger a full rebuild while
   normal appended-tail replay stays O(tail). **Accepted blind spot (documented in code):** an edit strictly
   inside the un-probed middle of a >8KB prefix; full-content hashing per startup would reinstate the
   O(all-events) cost the feature removes.

**Final hardening pass (2026-07-17 — three defects the final review reproduced, all fixed):**
1. **Asyncio usage cross-draining.** The ContextVar held a *mutable list*; asyncio context copies are shallow,
   so tasks inherited the SAME list and one task's `drain()` (clear) emptied another's (reproduced as
   `A=['a','b'], B=[]`). Storage is now an **immutable tuple with copy-on-write `set()`** — an inherited value
   is never mutated, so threads and asyncio tasks are both isolated while nested calls within one node still
   drain together. `record()` still returns the live entry dict for `mark_failed()`. New `usage.reset()` is
   called at run start (`load_inputs`) so a reused execution context cannot leak prior-run stragglers.
   The reviewer's exact two-task probe is a permanent test (`test_asyncio_tasks_cannot_cross_drain`).
2. **Role-contract dict validation.** `_safe_invoke` accepted ANY parsed dict — `{"unexpected": 1}` was
   returned as analyst "success", recorded ok=True, and blocked backups. `_dict_contract_error()` now
   validates the minimal discriminator per PRIMARY role (planner: non-empty `plan` list; executor: dict
   `args` + step identity; synthesizer: any extraction key; curator: any compaction key; analyst: non-empty
   `answer`; critic: `decision ∈ {accept, revise, replan}`), with one level of envelope unwrapping preserved
   (`candidate_answer`/`result`/…). Rejection marks the usage entry failed with the contract reason and tries
   the next backup; the heuristic fallback runs only after the chain is exhausted. **Fallback telemetry:**
   heuristic/raw-text fallback activation is flagged on the final attempt's entry
   (`fallback_activated=True` — never an extra call), aggregated separately from backup activations, and shown
   in a new `fallback` column of the report table.
3. **`code_kb` mode-exact argument aliases.** The shared alias list accepted keys handlers ignore
   (`xrefs_to`+`src` and `xrefs_from`+`dst` were confirmed false positives). `step_is_concrete` now mirrors
   each handler exactly: routine/annotate → start|addr|address; xrefs_to → addr|dst; xrefs_from → addr|src;
   disasm → address|addr (vice engine) / start|addr (capstone/default); search → q|query; sql → sql; metadata
   modes arg-free; unknown modes → enrich. Positive AND negative alias cases per mode are in the test table;
   retries still always use executor enrichment.

**Known limitations / deliberate choices for review:**
1. On a deferred batch the executor sees a digest that is at most one batch stale (fresh facts still reach the
   analyst — a flush always precedes it).
2. `budget_used` counts only LLM cost; tool/VICE calls are counted (report) but not priced.
3. The fingerprint blind spot from hardening item 6 above (middle-of-prefix edits on >8KB logs).
4. Straggler entries from a node that crashed between record and drain are dropped at the next run's
   `usage.reset()` (in `load_inputs`) rather than attributed — they can no longer leak across runs or tasks.
5. Contract validation checks each role's minimal discriminator, not full schemas — a dict with the right
   discriminator but malformed optional fields still reaches the node's existing permissive parsing.

---

## Phase 2 — Persistence & curator honesty

- [x] **2.1 Curator: uncompacted-size gate + high-water mark + digest section** ✅ (invisible output confirmed)
  - **Bug (3 parts):** (a) `post_synth_router` gates on total `kb.json` size (`graph/routers.py:84`), which only grows,
    so once a game crosses ~80k tokens *every* future synthesizer step pays a curator call forever; (b) `curator_node`
    always takes the last 20 tool_results by ts and never marks them compacted, so it re-summarizes near-identical
    inputs; (c) `EVT_CONSOLIDATED` output has no section in `digest_for_question`, and the only consumer (semantic
    index) is disabled by default — the curator's work is never read.
  - **Fix:** gate on token size of *uncompacted* tool_results (newer than the last consolidated high-water id); store
    the high-water event id in the consolidated payload and only feed newer events forward; add a
    `## Consolidated observations` section to the digest.
  - **Files:** `graph/routers.py`, `graph/nodes.py`, `memory/store.py`.
  - **Effort:** ~half day · **Impact:** High · **Source:** F §1.3, G #8, S
  - **My call:** Do — removes a permanent tax and restores long-term memory to the analyst.

- [x] **2.2 Dump/asm freshness (content hash, not path-idempotence)** ✅
  - **Bug:** `ingest_dump` skips on path-seen-before (no hash/mtime); `already_ingested_asm` keys on path alone. Text
    docs *do* re-ingest on newer mtime — dumps/asm should too. Overwriting `petch_ingame.bin` silently poisons later turns.
  - **Fix:** key dumps/asm by `(path, sha256|mtime, size)`; on change append a new ingest event, invalidate derived
    Capstone windows / Layer-0 rows for that source, note the refresh in the digest (+ optional UI banner).
  - **Files:** `memory/store.py`, `code_kb/store.py`.
  - **Effort:** ~half day · **Impact:** Medium-High · **Source:** Gr §1.3
  - **My call:** Do — real correctness bug for iterative users.

- [x] **2.3 Confidence-aware replay + add `confidence` to `routines`** ✅
  - **Bug:** `SYNTHESIZER_ROLE` promises "KB overwrites when confidence improves," but `_apply_event_to_sqlite`
    (`memory/store.py:368-448`) uses unconditional `INSERT OR REPLACE`; `routines` has **no confidence column**
    (`memory/schema.py:38-45`), and a later lower-confidence label re-emit clobbers a higher one.
  - **Fix:** add `confidence REAL` to `routines`; make replay compare-and-swap (replace only when
    `new.confidence >= old.confidence`) for labels and routines. Pure derived-view change; event log untouched.
  - **Files:** `memory/schema.py`, `memory/store.py`.
  - **Effort:** ~2h · **Impact:** Medium · **Source:** F §1.6
  - **My call:** Do — stops silent fact downgrades.

- [x] **2.4 Make CLI resume real (`SqliteSaver`) or fix the help text** ✅
  - **Bug:** `main.py:189` compiles with `MemorySaver`, so `--thread-id` ("Resume an existing session") never resumes
    across processes; only the KB persists. `tools/agent_runner.py:40` same, plus a fresh thread_id per UI turn.
  - **Fix:** use `SqliteSaver` at `sessions/<slug>/checkpoint.sqlite` (also gives crash-resume mid-iteration), or, if we
    intentionally rely on KB-only persistence, correct the flag help. **Decision needed** (see Open Questions).
  - **Files:** `main.py`, `tools/agent_runner.py`.
  - **Effort:** ~2–4h · **Impact:** Medium · **Source:** F §1.9
  - **My call:** Do the SqliteSaver route — it composes with the multi-turn archive (6.2).

- [-] **2.5 Persist the semantic vector index (embed only cache misses)**
  - **Source:** F §2.7, G #10 · **Effort:** ~1 day
  - **My call:** Defer until semantic search is actually enabled by default (it's `enabled:false` today). Track behind 2.1.

### Phase 2 implementation notes (2026-07-17; hardening-round fixes applied 2026-07-18 — for reviewer)

> **STATUS: Phase 2 implemented (2.1–2.4; 2.5 stays deferred), hardened per the GPT 5.6 Sol review's two
> reproduced defects, and finished per the follow-up review's three source-invalidation defects (see
> "Hardening-round changes" below — including the final-round provenance/re-materialization design and the
> append-order `seq` ordering fix). Full suite: 214 tests passing; `git diff --check` clean; `compileall`
> clean; `graph.build`/`main`/`tools.agent_runner` import; no checkpoint or test artifact under the
> repository's real `sessions/` tree.**

Per item (final behavior):

- **2.1 (curator honesty)** — The gate now keys on the token volume of **uncompacted** tool_result events
  (`store.uncompacted_tool_result_tokens()`, threshold `UNCOMPACTED_TOKEN_THRESHOLD = 40_000` — replaces the
  total-kb.json-size gate that could never close). The curator consumes the **oldest** uncompacted events
  (`uncompacted_tool_results()`, oldest-first, ties broken by id) and records the mechanically-derived
  `events_compacted` ids on the consolidated payload — **never** the LLM's claimed list; an LLM failure writes
  nothing, so no events are marked compacted without a real summary. **Bookkeeping is exact and ID-based**
  (hardening — see below): replaying a consolidated event registers its listed ids in the derived
  `compacted_tool_results` table (`SCHEMA_VERSION` → "4"), and "uncompacted" everywhere means
  `id NOT IN compacted_tool_results` — never a timestamp comparison. `through_ts` on the payload is
  informational only. `digest_for_question` gained a `## Consolidated observations` section (latest 3
  summaries) and its raw "Recent tool results" section excludes only **genuinely** compacted events —
  restoring the long-term memory the curator's output previously never reached.
- **2.2 (content freshness)** — General KB: `ingest_dump` is idempotent on **content sha256**, not path; a dump
  overwritten in place appends a fresh ingest event with `refreshed_from: <old-sha>` and the digest's KB-stats
  section flags `dump_refreshed: True` (warning that older dump-derived facts may be stale). Legacy ingest
  events without a hash get a one-time `sha_backfill` re-ingest, not a false "refreshed" flag. Code KB:
  `asm_freshness(path, content)` returns new/current/changed (sha-based; size+mtime fallback for legacy
  events) replacing the path-only `already_ingested_asm` in `_hydrate_code_kb`; a changed file triggers
  `invalidate_source(path)` — an **event** (`EVT_INVALIDATE_SOURCE`) whose replay deletes that source's rows
  from `code_routines`/`code_labels`/`instructions`/`hypotheses`/`asm_docs` *before* the re-ingest events that
  follow, so full rebuilds reproduce the same state. `code_kb.ingest_dump` returns `(event_id, changed)`; on a
  changed dump the hydrator invalidates `capstone:%`/`vice:%` disassembly-window rows (LIKE-pattern
  invalidation). **Invalidation is now COMPLETE** (hardening — see below): `code_xrefs`, `code_smc_sites`,
  `code_class`, and the canonical `annotations` rows all carry `source_file` provenance and are deleted with
  the rest; the old "documented residue" no longer exists.
- **2.3 (confidence-aware replay)** — `routines` gained a `confidence REAL` column
  (`SCHEMA_VERSION` bumped to "3" → existing derived DBs rebuild once, transparently). Labels and routines now
  replay via `ON CONFLICT … DO UPDATE … WHERE excluded.confidence >= existing.confidence`: a later
  lower-confidence re-emission no longer clobbers a stronger fact (the synthesizer prompt's overwrite promise
  is finally true). Pure derived-view change — the event log still records every emission. Equal confidence
  refreshes (allows corrections at the same strength). Digest routine lines and `routines_summary` now show
  confidence; the KB schema hint documents the CAS.
- **2.4 (real resume)** — The CLI compiles with **`SqliteSaver`** at `sessions/<slug>/checkpoint.sqlite`
  (new runtime dep `langgraph-checkpoint-sqlite`; graceful in-memory fallback with a warning if missing;
  purged together with the game by `purge_persistence.py --game`; legacy-slug dirs honoured). Default
  `--thread-id` is now **per-question** (`<game-slug>-<question-sha1[:8]>`): re-running the same game+question
  resumes that thread's checkpointed state after a crash, while a different question gets a clean thread so
  add-reducer state (tool_results/messages/llm_usage) can't bleed across questions. **Honest limitation
  (documented in the flag's help):** re-invocation restarts the graph from the planner *on top of* the
  checkpointed state — mid-node resume (input=None continuation) is left to the 6.2 notebook work. The
  Streamlit runner intentionally keeps MemorySaver + fresh thread ids until 6.2.

**Hardening-round changes (GPT 5.6 Sol review of Phase 2 — both reproduced defects fixed, 2026-07-18):**
1. **Timestamp cursors eliminated from compaction bookkeeping.** The first pass used
   `ts > last_consolidated.through_ts` as the boundary — an equal-timestamp event split across a batch limit
   was silently skipped, and a legacy consolidated event (no `through_ts`) fell back to its own timestamp,
   wrongly hiding EVERY earlier tool result even though the old curator summarised only its newest 20.
   Design now: a derived, indexed `compacted_tool_results(event_id PK, consolidated_id)` table is populated on
   replay of each consolidated event's `events_compacted` list; `uncompacted_tool_results()`,
   `uncompacted_tool_result_tokens()`, and the digest's excerpt filter all use `id NOT IN (…)` membership.
   Ordering is `ts ASC, id ASC` (deterministic, replay-stable); exclusion never depends on order. Legacy
   consolidated events hide only the ids they actually listed — worst case is harmless re-compaction, never
   hidden evidence. `last_consolidated_through_ts()` is deleted. **Migration:** `SCHEMA_VERSION` "3"→"4"
   triggers one transparent full rebuild of existing derived DBs; the event log is untouched.
2. **Source invalidation made complete.** `code_xrefs`, `code_smc_sites`, `code_class`, and `annotations`
   gained `source_file` provenance (populated from annotation payloads; layer0's classify payloads now carry
   it too; xref/SMC payloads already did in both the asm and disasm paths). The `EVT_INVALIDATE_SOURCE`
   replay handler now deletes from all eight tables (+ `asm_docs`), works live, after reopen, and under full
   replay (code_kb rebuilds its derived view on every open, so no migration machinery is needed).
   **Multi-source provenance:** xref and SMC rows are per-(fact, source) — dedup is NULL-safe per source,
   so the same edge asserted by two files keeps two rows and invalidating one source cannot delete the
   other's provenance; consumers were updated to `SELECT DISTINCT` / `COUNT(DISTINCT …)`
   (`code_kb_node` xrefs modes, `call_graph.local_dot`, `agent_runner.top_routines`).
3. **Final round (2026-07-18): three further reproduced invalidation defects fixed.**
   (a) *Indirect-jump xrefs lost provenance* — the `ANN_INDIRECT` projection called `_upsert_xref()` without
   `source_file`, so its `code_xrefs` rows were NULL-attributed and survived invalidation. The annotation's
   provenance now passes through every xref projection path.
   (b) *Legacy source-less classifications survived invalidation* — pre-hardening layer-0 classify payloads
   had no `source_file`. **Provenance recovery at replay time**: the legacy evidence templates
   ("… consecutive parsed instructions in `<path>`", "gap marker in `<path>` (N bytes)") embed the path, and
   `_recover_classify_source()` extracts it when the canonical row is applied — the append-only event log is
   never modified. Recovered rows are removed by path invalidation like any other, removed ranges do not
   survive when the new file emits no replacement, and unrelated sources are untouched.
   **Conservative residue policy (documented):** a deterministic layer-0 classify row whose provenance cannot
   be recovered (hand-altered evidence text) is dropped on ANY invalidation — it cannot be proven current, so
   it must never be presented as such; a re-ingest re-creates it with full provenance.
   (c) *Singleton projections discarded the surviving source* — with singleton keys
   (`code_routines` by start, `code_labels` by addr+name, `instructions` by addr, `code_class` by addr), if
   source A contributed a fact, B overwrote the key, and B was invalidated, the key was left empty though A's
   canonical annotation still asserted the fact. **Re-materialization design**: the invalidation replay now
   deletes the source's canonical `annotations` rows (single provenance-carrying source of truth), then
   rebuilds ALL typed projections from the surviving annotations via the factored `_project_annotation()` —
   invalidation is rare, so a full projection rebuild is cheap and provably consistent live, after reopen,
   and under full event-log replay (the code KB replays its whole log on every open). Per-source xref/SMC
   rows re-materialize exactly; the Layer-1 name-promotion path replays in order. The stale
   `invalidate_source()` docstring was rewritten to match.
   **Ordering fix (final):** the first re-materialization sorted by `(ts, id)` — but event ids are random
   UUID fragments, so two equal-timestamp annotations could replay in REVERSED append order, flipping which
   fact wins a singleton key after an *unrelated* invalidation (reproduced: A appended first with id
   `zzzzzzzzzzzz`, B second with id `aaaaaaaaaaaa`, same timestamp → B correctly won live, then A wrongly won
   after rematerialization and persisted across reopen). `annotations` now carries an explicit monotonic
   **`seq`** column: assigned 1..N in event-log order during replay, `MAX(seq)+1` on live appends (under the
   store's append lock; a re-apply of a known annotation id keeps its original seq). Invariants: deletions
   never renumber survivors, so surviving relative order is immutable; a later append always sorts after
   every survivor both live and in a from-scratch replay — hence identical winners live, after unrelated
   invalidation, after overlapping invalidation, after reopen, and after full replay. Rematerialization
   sorts by `ORDER BY seq ASC` only. (SQLite `rowid` was rejected: `INSERT OR REPLACE` churns it and VACUUM
   may renumber implicit rowids — `seq` makes the invariant explicit.)

**Tests:** 214 passing (32 net new across Phase 2 + all hardening rounds). Ordering-fix additions in
`test_freshness.py`: the exact equal-timestamp/reversed-lexical-id reproducer, **parameterized across
routines, labels, instructions, and classifications** — B (appended second, lexically earlier id) must remain
the winner after an unrelated invalidation, after reopen/full replay, and A is restored only when B itself is
invalidated; plus per-source xref/SMC rows verified byte-identical through a seq-ordered rematerialization.
An empirical probe confirmed the old `(ts, id)` ordering reverses this exact case. Earlier final-round
additions: indirect-xref provenance carried and invalidated (live + full replay); legacy classify
fixture using the EXACT pre-hardening payload shape — recovered, invalidated, ranges stay removed without
re-emission, unrelated source preserved; unattributable legacy classify swept conservatively on any
invalidation; singleton restoration for routines/labels/instructions/classifications (B invalidated → A
restored, live + replay); invalidating the older source preserves the current one. Earlier inventory:
`test_curator.py` (gate opens on uncompacted
volume and CLOSES after compaction; oldest-first, never-twice compaction with mechanical bookkeeping; failed
LLM marks nothing compacted; digest shows consolidated section, hides compacted raw, keeps fresh evidence;
**legacy consolidated event hides only its listed 20 of 25 — the older 5 stay visible**; **equal-timestamp
events with limit=1 cannot be skipped** (frozen-clock); **bookkeeping survives a full derived-view rebuild**),
`test_confidence_replay.py` (label/routine downgrade protection, equal/higher updates, CAS holds under
from-scratch replay), `test_freshness.py` (dump content-change detection + `refreshed_from` provenance +
digest flag + cache refresh; asm new/current/changed; event-based invalidation survives full replay;
**invalidation covers xrefs/SMC/classification/annotations, preserves another source's rows including its
provenance row for a shared identical edge, and holds after reopen/full replay**; **`capstone:%`/`vice:%`
LIKE invalidation reaches xref and SMC rows**; code-KB dump change tuple), `test_resume.py` (per-question
stable thread ids, checkpoint file creation, compiled-graph acceptance, legacy-dir reuse).
`test_recursion_recovery` runs the durable checkpointer against tmp only; verified: no `checkpoint.sqlite`
exists anywhere under the repository's real `sessions/` tree.

---

## Phase 3 — RE technique gains (biggest answer-quality jumps)

- [ ] **3.1 Memory snapshot diffing (`vice.memory.diff` composite)** — flagship
  - **Why:** For "where is the lives/score/level counter?", differential memory analysis beats any static disassembly.
    Building blocks exist (`vice.memory.read` up to 64KB; the on-disk dump is itself a snapshot).
  - **Fix:** agent-side composite with modes `snapshot` (store named RAM snapshot event), `diff` (changed addresses
    classified by region, old→new), `monotonic_scan` (intersect ≥3 snapshots by known delta, e.g. −1 for lives).
    Even diffing the ingested dump vs a live VICE read finds actively-mutating state. Expose in the planner cheat-sheet:
    prefer snapshot-diff when the question names an in-game quantity.
  - **Files:** `graph/nodes.py`, `graph/prompts.py`, `tools/vice_mcp.py`.
  - **Effort:** ~1–2 days · **Impact:** Highest quality gain · **Source:** F §3.1, G #6, Ge §2.1, S (all)
  - **My call:** Do — the single highest-value capability, but it needs a live VICE to shine (see 3.3/dump-catalog for the offline path).

- [ ] **3.2 Watchpoint/trace composite (`vice.trace`) + un-ban registers-after-break** ✅ (contradiction confirmed)
  - **Now:** `checkpoint_add`/`breakpoint` map to `vice.checkpoint.add` (`graph/nodes.py:1712-1713`) but have no arg
    normalization and aren't in the cheat-sheet; `vice.execution.step/run/pause` unexposed; the analyst prompt asks for
    breakpoint follow-ups it can't run; `PLANNER_ROLE` line 215 recommends `vice.registers.get` while
    `vice_mcp_node:2423-2430` hard-bans it.
  - **Fix:** `vice.trace` composite: watchpoint on candidate addr → run N frames/until-hit → read PC+registers →
    resolve PC to the writing instruction → auto-disasm ±16 bytes. Un-ban `registers.get` only when a checkpoint is
    armed; reconcile prompt and ban-list.
  - **Files:** `graph/nodes.py`, `graph/prompts.py`, `tools/vice_mcp.py`.
  - **Effort:** ~1–2 days · **Impact:** High (unlocks the dynamic verification the critic demands) · **Source:** F §3.2/§1.7, G #6, Ge §2.3, S (all)
  - **My call:** Do — pairs with 3.1 (static candidate ∩ dynamic trace = near-certainty).

- [ ] **3.3 `find_counters` deterministic static heuristic**
  - **Why:** `find_loops` finds control flow; nothing finds *state*. 6502 game-state idioms are regular and cheap to
    scan: `DEC/INC abs` below `$D000` (lives/timers), `SED…ADC/SBC…CLD` clusters (BCD score), stores into screen RAM
    `$0400-$07E7` with `ORA/ADC #$30` digit chains (HUD), `CMP #$0A` after `INC abs` (decimal rollover / multi-byte).
  - **Fix:** emit labels/hypotheses with evidence (`kind:"ram_var"`, e.g. `candidate_lives_counter`) *before* the LLM
    sees the question. Expose `code_kb mode='suspects' kind='lives|score|timer|sprite|loader|protection'`.
  - **Files:** `tools/c64_disasm.py`, `code_kb/layer0.py`, `graph/code_kb_node.py`, `graph/prompts.py`.
  - **Effort:** ~1 day · **Impact:** High (first-iteration answers for flagship questions) · **Source:** F §3.3, G #5, S
  - **My call:** Do — pure/testable, works offline, composes with 3.1.

- [ ] **3.4 Index data references / zero-page use-def in Layer 0** ✅ (only ctrl-flow xrefs today)
  - **Now:** `code_kb` Layer 0 records xrefs only for JSR/JMP/branches; every `LDA/STA/CMP abs` operand is discarded, so
    "what writes $D012?" needs fresh disasm + LLM eyeballing.
  - **Fix:** `ANN_DATAREF` (`src, dst, access∈{r,w,rmw}, index∈{none,x,y}`) captured in the same parse pass; query modes
    `writes_to`/`refs_to`/`zp_roles`/`hardware_refs`. Add zero-page use-def grouping to discover array/table base
    pointers (populates the near-empty `data_structures`).
  - **Files:** `code_kb/schema.py`, `code_kb/layer0.py`, `code_kb/store.py`, `graph/code_kb_node.py`.
  - **Effort:** ~2 days · **Impact:** Medium-High · **Source:** F §3.4, G #4, Ge §3.3, S (all)
  - **My call:** Do — turns many multi-iteration hunts into one SQL query.

- [ ] **3.5 Deterministic idiom pre-tagging before Layer-1**
  - **Fix:** table-driven matcher for rigid signatures (raster wait `LDA $D012/CMP/BNE`, `DEX/BNE` delay, memcpy
    `LDA abs,X/STA abs,X/DEX/BNE`, jump-table dispatch, KERNAL trampolines `JMP $FFxx`, SID tick to `$D400-$D418`).
    Feed matches into the Layer-1 prompt as "pre-analysis found: raster_wait at $XXXX" so the LLM confirms rather than
    discovers → higher accuracy + cheaper Layer-1 model.
  - **Files:** `tools/c64_disasm.py`, `code_kb/layer1.py`.
  - **Effort:** ~2 days · **Impact:** Medium · **Source:** F §3.7, G #5, Ge §3.1, S (all)
  - **My call:** Do after 3.4 (shares the instruction index).

- [ ] **3.6 Bank-awareness / RAM-under-ROM + illegal-opcode handling** ✅ (E000 penalty confirmed)
  - **Now:** `find_loops` scans only `$0800-$CFFF` and penalizes ≥`$E000` by −30 (`tools/c64_disasm.py:464-468`) even
    though the input is a RAM dump; ROM-range vector seeds burn the `max_insns` budget on KERNAL; `bank` arg exists but
    the planner is never told bank names.
  - **Fix:** condition the high-range penalty on evidence (`$0001` banking, HW vectors pointing into plausible RAM);
    skip ROM-range seeds unless banking says RAM is mapped there; document bank names (`cpu/ram/rom/io/cart`) in the
    cheat-sheet; mark banked-region disasm lower-confidence; optionally add an illegal-opcode decoder or dual VICE
    interpretation.
  - **Files:** `tools/c64_disasm.py`, `code_kb/disasm.py`, `graph/prompts.py`, `tools/vice_mcp.py`.
  - **Effort:** ~1–2 days · **Impact:** Medium (fewer hallucinations) · **Source:** F §3.5, G #7, Ge §3.2, S (all)
  - **My call:** Do the penalty/seed fixes (cheap, high value); treat the illegal-opcode decoder as optional stretch.

- [ ] **3.7 PETSCII / screen + color RAM decode tool** (C64-specific, cheap)
  - **Fix:** deterministic dump→string extractor: given `$0001`/VIC state or defaults, decode screen RAM (+color RAM)
    to ASCII/PETSCII and write into the KB. Often answers "what does the HUD say / where is score text?" without vision.
  - **Files:** new tool + `graph/nodes.py`/`code_kb`, reuse `code_kb/hardware_pack.py`.
  - **Effort:** ~half day · **Impact:** Medium (unique C64 value) · **Source:** Gr §3.1
  - **My call:** Do — cheap, offline, complements 3.1.

- [-] **3.8 Loop-scoring refinements / pseudocode decompilation / multimodal poke-and-peek**
  - **Source:** F §3.6, Ge §2.2, Ge §2.1
  - **My call:** Defer. Loop-scoring refinements are nice-to-have polish; pseudocode decompilation and poke-and-peek are
    larger bets that depend on 3.1/3.2 landing first and (for poke-and-peek) on `vice.memory.write` + HITL gating (6.1).

---

## Phase 4 — Correctness/robustness papercuts (cheap, bundle together)

- [ ] **4.1 `detect_basic_sys` misses `SYS 2064` (space after SYS)** ✅
  - `tools/c64_disasm.py:551-568`: digit loop breaks on the first non-digit, so a space/`(`/shifted-space after `$9E`
    yields `None`. Skip `$20`/`$28`/`$3A` (tolerate leading `+`) before collecting digits; consider scanning the BASIC
    line link-chain instead of a fixed `$0801-$0830` window. **Source:** F §1.8.

- [ ] **4.2 `_capstone_linear` silently defaults `start` to `$0801`** ✅
  - `graph/nodes.py:1926` — same silent-default class deliberately fixed for VICE args. Make a missing `start` an error
    for consistency. **Source:** F §3.9.

- [ ] **4.3 `kb mode='labels'` / digest matches names only, ignores addresses in the question**
  - `relevant_labels` matches question terms against label *names*; a question naming `$C145` never matches by address.
    Extract `$XXXX` tokens from the question and add an `addr IN (...)` clause. **Source:** F §3.9.

- [ ] **4.4 Vision call reuses the analyst JSON-only system prompt**
  - `_describe_screenshot` (`graph/nodes.py:2270-2278`) tells the model to emit analyst JSON while asking for prose. Use
    a minimal neutral system prompt for vision calls; store description + thumbnail path; stop burning three roles on
    silent vision failures (dedicated vision-capable role/flag). **Source:** F §3.8, Gr §3.4.

- [ ] **4.5 `vice.memory.read` truncated to 4,000 chars loses data**
  - `graph/nodes.py:2459` — for JSON byte arrays that's ~250 bytes of RAM. Format as compact hex-dump lines
    (16 bytes/line) *before* truncation, as `write_report` already does. **Source:** F §2.9.

- [ ] **4.6 Expand Tavily `include_domains` + advanced depth for researcher**
  - `graph/nodes.py:2196-2197` locked to 4 sites; add archive.org, forum64.de, C64 wiki (or make it a config knob);
    pass `search_depth="advanced"`. **Source:** F §3.9.

- [ ] **4.7 Per-role `max_tokens` (planner/analyst) to avoid mid-JSON truncation**
  - `defaults.max_tokens: 4096` can truncate an 8–10 step plan mid-JSON, burning the backup chain. Wire the per-agent
    `max_tokens` (config already supports per-agent keys) through `get_llm`. **Source:** F §2.9.

- [-] **4.8 `tool_results` grows unbounded in state**
  - Reducer is `add`; curator can't trim it. Replace with a drop-sentinel reducer or keep only
    `(step_id, ok, event_id)` tuples once persisted. **Source:** F §2.9. **My call:** Defer to Phase 1.7 (batching)
    where the state shape is already being reworked.

---

## Phase 5 — Observability & evaluation (do alongside Phases 1–3)

- [ ] **5.1 Golden-question regression suite** (`evals/golden.jsonl`)
  - ~10–20 `(game, question, expected_addresses, expected_keywords)` rows for manually verified facts (5 dumps + asm
    already exist). Pytest-marked slow suite runs the graph, asserts the expected `$XXXX` appears with confidence ≥
    threshold, records LLM calls/tokens/wall-time. **Source:** F §4, G #12, S.
  - **My call:** Do — without it every heuristic change is a guess. Build early so Phase 1–3 have a target.

- [ ] **5.2 Unit tests for the deterministic layer**
  - Pure functions with obvious fixtures: `detect_basic_sys`, recursive/SMC disasm, `_rewrite_kb_sql`,
    `_normalize_plan_ids`, `scoping.select_asm_files`, layer0 routine-boundary, "curator doesn't compact twice."
    Add tiny synthetic 64KB fixture dumps. **Source:** F §4, G #12, S.
  - **My call:** Do — cheap and protects exactly the code Phases 0/3/4 touch.

- [x] **5.3 Per-role cost telemetry table in `write_report`** (depends on 1.4) — _delivered with 1.4_
  - role → calls / tokens / failures / backup-role activations. **Source:** F §4.

- [ ] **5.4 `run_summary` event per completed run** (question, verdict, confidence, iterations, cost)
  - Makes cross-run behavior queryable and feedable to the planner ("previous runs already established…"). **Source:** F §4.

- [-] **5.5 Human ratings + LangSmith datasets** — **Source:** Gr §4.3. **My call:** Defer until UI work (Phase 6).

---

## Phase 6 — Product/UX (research-notebook direction; larger, optional)

- [ ] **6.1 Human-in-the-loop interrupts** (plan edit + mutating-VICE gate)
  - LangGraph `interrupt_before` after planner (edit/reorder/delete steps) and before mutating VICE calls
    (`memory.write`, poke-and-peek); Stop/cancel in Streamlit. **Source:** Gr §2.1.
  - **My call:** High-trust, but prerequisite for any `memory.write`/poke-and-peek work; sequence before 3.8's poke-and-peek.

- [ ] **6.2 Multi-turn research notebook** (turn archive, versioned reports, digest "prior answers" section)
  - Persist `sessions/<game>/turns.jsonl`; version reports (`report_<ts>.md` + latest copy); add prior accepted answers
    + open questions to the digest; open-question chips in Streamlit. **Source:** Gr §2.2. Pairs with 2.4 (real resume).

- [ ] **6.3 Hypothesis lifecycle** (mark supported/refuted on accept/contrary evidence) — **Source:** Gr §2.3.
- [ ] **6.4 Share one evidence resolver between CLI and UI** — **Source:** Gr §2.4.
- [ ] **6.5 Named multi-state dump catalog** (title/ingame/death/level2; diff frozen stages offline) — **Source:** Gr §3.3. Offline complement to 3.1.
- [ ] **6.6 Symbol-map export** (VICE monitor `.sym`/`.labels` from high-confidence labels) — **Source:** Gr §3.2.
- [ ] **6.7 Synthesizer→Code-KB Layer-1 without Layer-0 gate** — require enclosing Layer-0 window or tag `unverified_llm` so the call-graph UI stays honest. **Source:** Gr §3.5.

- [-] **6.8 DAG-based planner / Code-KB Layer 2+3 / structured outputs (`with_structured_output`)**
  - **Source:** Ge §1.2, G #10a, F §2.6. **My call:** Defer. DAG planning and Layer 2/3 are large architectural bets the
    reports themselves rank as ambitious; the fable report explicitly recommends stabilizing the linear loop first.
    `with_structured_output` is attractive (deletes a lot of JSON-salvage code) but risky across 5 providers — schedule
    it as its own hardening pass *after* the loop is stable and the golden suite can catch regressions.

---

## Phase 7 — Hygiene / packaging

- [ ] **7.1 Wheel ships only `graph/`** ✅ — `pyproject.toml:37` `packages = ["graph"]` while runtime needs `memory/`,
  `tools/`, `code_kb/`, `config/`. Package all importable packages + config data (or clearly document editable-only). **Source:** Gr §4.1.
- [ ] **7.2 Dead `coordinator`/`researcher` roles** — defined in `config/llm.json` + `graph/llm.py`, never invoked, no
  `ROLE_BLOCKS` entries. Remove them or wire real paths (researcher = owned web research; coordinator = multi-turn
  agenda, pairs with 6.2). Dead config invites accidental provider spend. **Source:** Gr §4.2.
- [ ] **7.3 Secret redaction in digests / SQL previews** — `kb mode=sql` can `SELECT` full `text_docs.content`; redact
  API-key-shaped strings; document `text_dir` as trusted LLM-context input. **Source:** Gr §4.4.

---

## Suggested execution order (condensed)

1. **Phase 0** (all) — one PR per item or one "control-loop" PR, each with a regression test.
2. **5.1 + 5.2** (golden + unit harness) — stand up before Phase 1/3 so savings and heuristics are measurable.
3. **1.4** (cost accounting) → **1.1 + 1.2 + 1.3** (executor skip + deterministic/batch synthesis + fan-out cap).
4. **2.1 + 2.2 + 2.3** (curator honesty + freshness + confidence replay).
5. **Phase 3** in order **3.3 → 3.4 → 3.7 → 3.1 → 3.2 → 3.5 → 3.6** (offline/static heuristics first — testable without
   live VICE — then the dynamic VICE composites, then idioms/banking).
6. **Phase 4** papercuts bundled into a single sweep (each has a unit test from 5.2).
7. **1.5 + 1.6 + 2.4** (perf/scan cleanup + real resume).
8. **Phase 6/7** as product/hygiene follow-ups.

---

## Open questions for review (user + GPT 5.6)

1. **Resume strategy (2.4):** switch to `SqliteSaver` checkpointing, or keep KB-only persistence and just fix the help
   text? SqliteSaver adds crash-resume + composes with the multi-turn archive but adds a checkpoint file to manage.
2. **`recursion_limit` value (0.1):** 400 is a rough fit for 12 iterations × 8-step plans. Confirm the target
   iteration/plan sizes so we set it deliberately rather than "big enough."
3. **VICE availability for testing (3.1/3.2):** is a live vice-mcp server available in the dev/CI loop, or should the
   dynamic composites be validated only via the offline dump-catalog path (6.5) until then?
4. **`with_structured_output` (6.8):** worth the cross-provider risk, or keep the current regex-salvage path? Leaning
   "later, behind the golden suite."
5. **Scope of this milestone:** do we target Phases 0–3 + eval harness as the first shippable milestone, or a tighter
   Phase 0 + efficiency-only cut first?

---

_Last updated: 2026-07-16 · Planning pass by Claude (Opus 4.8). Line numbers reference the working tree at planning
time; re-verify at edit time. Items tagged ✅ were confirmed against the code during planning._
