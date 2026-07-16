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
| 1.1 | Skip the executor LLM when a plan step already has complete, concrete tool arguments. | [ ] Not started |
| 1.2 | Extract structured tool results deterministically and batch the remaining free-text LLM synthesis work. | [ ] Not started |
| 1.3 | Cap and prioritize automatic Layer-1 routine annotations to prevent hidden LLM-call fan-out. | [ ] Not started |
| 1.4 | Track real per-role token, cost, LLM-call, tool-call, and VICE-call usage against the run budget. | [ ] Not started |
| 1.5 | Reuse and memoize KB digests, partial-asm metadata, and question embeddings instead of rebuilding them per node. | [ ] Not started |
| 1.6 | Replace repeated full event-log scans with indexed lookups and replay high-water marks. | [ ] Not started |
| 1.7 | Run independent read-only plan steps in parallel with LangGraph fan-out after batching and retry semantics stabilize. | [-] Deferred |
| 2.1 | Compact only new, uncompacted events and expose consolidated observations in question digests. | [ ] Not started |
| 2.2 | Detect changed dump and assembly inputs by content metadata and invalidate stale derived analysis. | [ ] Not started |
| 2.3 | Store routine confidence and prevent lower-confidence replay events from overwriting stronger facts. | [ ] Not started |
| 2.4 | Implement durable CLI checkpoint resume with `SqliteSaver`, or correct the resume claim if KB-only persistence is retained. | [ ] Not started |
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
| 5.3 | Add a per-role calls, tokens, failures, and fallback-activation table to generated reports. | [ ] Not started |
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

- [ ] **1.1 Skip the executor LLM for already-concrete steps** ✅ (LLM-per-step confirmed)
  - **Now:** `executor_node` (`graph/nodes.py:1001-1083`) makes an LLM call for *every* step (with an 8k-char digest),
    even `{"mode":"stats"}` or fully-specified capstone steps.
  - **Fix:** per-tool/mode `is_step_complete(step)` validator; call the executor LLM only when a required arg is
    missing/null or references prior discoveries. Record `executor_llm_skipped` in the message for observability.
    (Mechanical dependency selection already runs without the LLM.)
  - **Files:** `graph/nodes.py`, `graph/prompts.py`.
  - **Effort:** ~half day · **Impact:** High · **Source:** F §2.1, G #1, S (strong overlap)
  - **My call:** Do — highest-ROI first patch per GPT.

- [ ] **1.2 Deterministic synthesis for structured tool output; batch the LLM rest** ✅ (all 3 reports)
  - **Now:** `synthesizer_node` runs an LLM extraction after *every* step, including structured JSON (`capstone
    find_entry/vectors/find_loops`, `code_kb` rows, `kb` rows), failed steps, and `kb stats`.
  - **Fix:** mechanical extractors for structured modes (find_loops→hypotheses, vectors→labels, code_kb routine
    windows are already KB content); invoke the LLM only for free-text (disassembly listings, Tavily snippets,
    screenshot descriptions). Accumulate + synthesize once per *batch* of completed steps. Add an extraction
    provenance field (`mechanical|llm|hybrid`).
  - **Files:** `graph/nodes.py`, `tools/c64_disasm.py`, `graph/code_kb_node.py`, `graph/state.py`.
  - **Effort:** ~1 day · **Impact:** High (highest cross-report consensus) · **Source:** F §2.2, G #3, Ge §1.3, S
  - **My call:** Do — pairs with 1.1 as the flagship efficiency patch.

- [ ] **1.3 Cap the hidden Layer-1 auto-annotate fan-out** ✅
  - **Now:** synthesizer fires `_mode_annotate` (a full LLM call, possibly with auto-disasm) for *every* routine with
    confidence ≥ 0.5 (`graph/nodes.py:1283-1303`) — six routines = six invisible extra LLM calls.
  - **Fix:** cap at 1–2 per synthesizer invocation, prioritized by (confidence × question-term relevance); log each as
    a transcript message and count it against budget (see 1.4).
  - **Files:** `graph/nodes.py`.
  - **Effort:** ~2h · **Impact:** Medium-High · **Source:** F §2.3
  - **My call:** Do — silent unbounded cost.

- [ ] **1.4 Real budget/cost accounting + per-role report table** ✅ (dead code confirmed)
  - **Now:** `budget_used` is checked (`graph/routers.py:98`) but only ever set to `0.0` in `load_inputs`
    (`graph/nodes.py:557`). Zero token telemetry.
  - **Fix:** read `usage_metadata` off each `AIMessage` in `_invoke_one`; accumulate (input,output) tokens per role;
    convert via a small price table (or track tokens); return a `budget_used` delta with an `operator.add` reducer on
    the field. Add `llm_calls`/`tool_calls`/`vice_calls` counters. Surface a per-role table in `write_report`.
  - **Files:** `graph/nodes.py`, `graph/llm.py`, `graph/routers.py`, `graph/state.py`.
  - **Effort:** ~half day · **Impact:** High (unblocks measuring everything else) · **Source:** F §1.4, G #4a, S
  - **My call:** Do early in Phase 1 so 1.1/1.2 savings are quantifiable.

- [ ] **1.5 Stop rebuilding the digest / re-embedding per node** ✅ (partial-asm scan confirmed hot)
  - **Now:** `planner_node` and `executor_node` both call `_kb_digest_for_state` (full SQL rebuild) though
    `state["kb_digest"]` is refreshed by the synthesizer each step; every digest build calls `partial_asm_excerpt`
    which re-reads/JSON-parses the entire `kb.json` (O(events) per digest); semantic-on re-embeds the question each node.
  - **Fix:** reuse `state["kb_digest"]` in planner/executor; cache the partial-asm path in `meta` at ingest;
    memoize digest keyed on `(question, events_total)`; memoize the question embedding.
  - **Files:** `graph/nodes.py`, `memory/store.py`.
  - **Effort:** ~half day · **Impact:** Medium (scales with KB) · **Source:** F §2.4
  - **My call:** Do — cheap once budget telemetry shows the waste.

- [ ] **1.6 Remove O(N) full-log scans on hot paths** ✅ (dedup scan confirmed)
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

---

## Phase 2 — Persistence & curator honesty

- [ ] **2.1 Curator: uncompacted-size gate + high-water mark + digest section** ✅ (invisible output confirmed)
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

- [ ] **2.2 Dump/asm freshness (content hash, not path-idempotence)** ✅
  - **Bug:** `ingest_dump` skips on path-seen-before (no hash/mtime); `already_ingested_asm` keys on path alone. Text
    docs *do* re-ingest on newer mtime — dumps/asm should too. Overwriting `petch_ingame.bin` silently poisons later turns.
  - **Fix:** key dumps/asm by `(path, sha256|mtime, size)`; on change append a new ingest event, invalidate derived
    Capstone windows / Layer-0 rows for that source, note the refresh in the digest (+ optional UI banner).
  - **Files:** `memory/store.py`, `code_kb/store.py`.
  - **Effort:** ~half day · **Impact:** Medium-High · **Source:** Gr §1.3
  - **My call:** Do — real correctness bug for iterative users.

- [ ] **2.3 Confidence-aware replay + add `confidence` to `routines`** ✅
  - **Bug:** `SYNTHESIZER_ROLE` promises "KB overwrites when confidence improves," but `_apply_event_to_sqlite`
    (`memory/store.py:368-448`) uses unconditional `INSERT OR REPLACE`; `routines` has **no confidence column**
    (`memory/schema.py:38-45`), and a later lower-confidence label re-emit clobbers a higher one.
  - **Fix:** add `confidence REAL` to `routines`; make replay compare-and-swap (replace only when
    `new.confidence >= old.confidence`) for labels and routines. Pure derived-view change; event log untouched.
  - **Files:** `memory/schema.py`, `memory/store.py`.
  - **Effort:** ~2h · **Impact:** Medium · **Source:** F §1.6
  - **My call:** Do — stops silent fact downgrades.

- [ ] **2.4 Make CLI resume real (`SqliteSaver`) or fix the help text** ✅
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

- [ ] **5.3 Per-role cost telemetry table in `write_report`** (depends on 1.4)
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
