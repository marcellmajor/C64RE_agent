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
| 1.7 | Run independent read-only plan steps in parallel with LangGraph fan-out after batching and retry semantics stabilize. | [x] Done |
| 2.1 | Compact only new, uncompacted events and expose consolidated observations in question digests. | [x] Done |
| 2.2 | Detect changed dump and assembly inputs by content metadata and invalidate stale derived analysis. | [x] Done |
| 2.3 | Store routine confidence and prevent lower-confidence replay events from overwriting stronger facts. | [x] Done |
| 2.4 | Implement durable CLI checkpoint resume with `SqliteSaver`, or correct the resume claim if KB-only persistence is retained. | [x] Done |
| 2.5 | Persist semantic vectors and embed only cache misses once semantic search is enabled by default. | [x] Cache/provider complete; activation remains explicit |
| 3.1 | Add named memory snapshots, diffs, and monotonic scans for locating changing game-state variables. | [x] Done |
| 3.2 | Add a safe VICE watchpoint/trace workflow and allow register reads when a checkpoint is armed. | [x] Done |
| 3.3 | Add deterministic 6502 counter heuristics for lives, score, timers, HUD digits, and similar state. | [x] Done |
| 3.4 | Index data reads/writes and zero-page use-def relationships, with query modes for references and hardware access. | [x] Done |
| 3.5 | Pre-tag common C64/6502 code idioms before Layer-1 LLM annotation. | [x] Done |
| 3.6 | Make disassembly bank-aware, avoid false ROM assumptions, and improve illegal-opcode handling. | [x] Done |
| 3.7 | Decode PETSCII, screen RAM, and color RAM into searchable KB text. | [x] Done |
| 3.8 | Explore refined loop scoring, pseudocode decompilation, and multimodal poke-and-peek after prerequisite work. | [x] Done; live baseline passed |
| 4.1 | Make BASIC `SYS` detection tolerate spaces and common separators before the entry-point digits. | [x] Done |
| 4.2 | Require an explicit Capstone linear-disassembly start address instead of silently using `$0801`. | [x] Done |
| 4.3 | Match hexadecimal addresses from the question when retrieving relevant KB labels. | [x] Done |
| 4.4 | Give screenshot analysis a dedicated neutral vision prompt and preserve usable failure/output metadata. | [x] Done |
| 4.5 | Encode VICE memory reads as compact hex dumps before truncation so most returned bytes are not lost. | [x] Done |
| 4.6 | Broaden configurable Tavily research domains and support advanced search depth. | [x] Done |
| 4.7 | Honor per-role output-token limits so planner and analyst JSON is not truncated mid-response. | [x] Done |
| 4.8 | Bound or compact `tool_results` in graph state after persisted results no longer need full payloads. | [x] Done |
| 5.1 | Add a golden-question regression suite that checks expected evidence, confidence, cost, and runtime. | [x] Done |
| 5.2 | Add deterministic unit tests and compact synthetic dump fixtures for core parsing, routing, and disassembly behavior. | [x] Done |
| 5.3 | Add a per-role calls, tokens, failures, and fallback-activation table to generated reports. | [x] Done (delivered with 1.4) |
| 5.4 | Persist one queryable run-summary event containing outcome, iterations, confidence, and cost. | [x] Done |
| 5.5 | Add human ratings and LangSmith datasets after the product/UI work is ready to collect them. | [x] Done |
| 6.1 | Add human approval/edit interrupts for plans and mutating VICE operations, plus UI cancellation. | [x] Done |
| 6.2 | Turn sessions into a multi-turn research notebook with archived turns, versioned reports, and prior-answer context. | [x] Done |
| 6.3 | Track hypotheses through open, supported, and refuted lifecycle states as evidence changes. | [x] Done |
| 6.4 | Share one evidence-reference resolver between the CLI and Streamlit UI. | [x] Done |
| 6.5 | Catalog named dumps from multiple game states so offline snapshot comparisons are reproducible. | [x] Done |
| 6.6 | Export high-confidence labels as VICE-compatible symbol maps. | [x] Done |
| 6.7 | Allow synthesizer-discovered Layer-1 facts into Code KB with a verified window or explicit unverified provenance. | [x] Done |
| 6.8 | Revisit DAG planning, deeper Code-KB layers, and cross-provider structured output after loop stabilization. | [x] Done opt-in; analyst/critic live-canary passed |
| 7.1 | Package all runtime Python modules and configuration data instead of shipping only `graph/`. | [x] Done |
| 7.2 | Remove unused coordinator/researcher role configuration or wire those roles into real graph paths. | [x] Done |
| 7.3 | Redact secret-like values from digests and SQL previews and document trusted note inputs. | [x] Done |

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

- [x] **1.7 Parallel fan-out (LangGraph `Send`) for independent read-only steps**
  - **Source:** F §2.8, G #2 · **Effort:** ~1 day+
  - **My call:** Originally deferred until batching/retries stabilized; completed in the deferred-work pass after 4.8.

### Phase 1 implementation notes (2026-07-17; hardening round + final hardening pass applied — for reviewer)

> **STATUS: Phase 1 implemented and hardened; 1.7 was subsequently completed in the deferred-work pass. The original
> 1.1–1.6 hardening baseline below is preserved for reviewer history. Hardened per the GPT 5.6 Sol review
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
  stable processed-result identities in state and flush in ONE call when the plan drains (analyst next)
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

**Deferred-work completion (1.7):** the executor now selects up to four fresh, concrete, dependency-ready read-only
steps and the conditional router dispatches them with native LangGraph `Send`. Static Capstone, parent-KB queries,
Tavily, and an explicit allow-list of query-only Code-KB modes may overlap. VICE, retries, unresolved arguments, and
Code-KB disassembly/annotation/export stay serial. Reducer-backed results/counters merge before exactly one shared
synthesizer pass. Tests prove the compiled graph's branches overlap through a barrier, the synthesizer sees one merged
batch, policy exclusions hold, fan-out is bounded, and real concurrent Capstone/parent-KB reads share a store safely.

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

- [x] **2.5 Persist the semantic vector index (embed only cache misses)**
  - **Source:** F §2.7, G #10 · **Effort:** ~1 day
  - **My call:** Defer until semantic search is actually enabled by default (it's `enabled:false` today). Track behind 2.1.

### Phase 2 implementation notes (2026-07-17; hardening-round fixes applied 2026-07-18 — for reviewer)

> **STATUS: Phase 2 implemented and hardened. The 2.5 persistent cache is complete and the supported embedding
> model is OpenAI `text-embedding-3-small` through the executor provider. Automatic indexing remains explicit
> (`enabled=false`) so opening a session cannot silently spend money. Hardened per the GPT 5.6 Sol review's two
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

- **2.5 (deferred-work implementation).** Semantic embeddings are cached in
  `sessions/<game>/kb/vectors.sqlite` as float32 rows keyed by provider, embedding model, requested dimension, and
  exact content hash. Bootstrap and incremental indexing hydrate cached rows and send only unique misses to the
  embedding provider; reopening the same store makes no embedding call. Model/dimension changes invalidate by key,
  corrupt-size rows degrade to cache misses, and focused tests cover process reopen and numeric round trips. The
  bundled config uses `text-embedding-3-small` through the OpenAI executor provider. It remains `enabled=false`
  until the operator explicitly opts into automatic embedding spend.

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

- [x] **3.1 Memory snapshot diffing (`vice.memory.diff` composite)** — flagship
  - **Why:** For "where is the lives/score/level counter?", differential memory analysis beats any static disassembly.
    Building blocks exist (`vice.memory.read` up to 64KB; the on-disk dump is itself a snapshot).
  - **Fix:** agent-side composite with modes `snapshot` (store named RAM snapshot event), `diff` (changed addresses
    classified by region, old→new), `monotonic_scan` (intersect ≥3 snapshots by known delta, e.g. −1 for lives).
    Even diffing the ingested dump vs a live VICE read finds actively-mutating state. Expose in the planner cheat-sheet:
    prefer snapshot-diff when the question names an in-game quantity.
  - **Files:** `graph/nodes.py`, `graph/prompts.py`, `tools/vice_mcp.py`.
  - **Effort:** ~1–2 days · **Impact:** Highest quality gain · **Source:** F §3.1, G #6, Ge §2.1, S (all)
  - **My call:** Do — the single highest-value capability, but it needs a live VICE to shine (see 3.3/dump-catalog for the offline path).

- [x] **3.2 Watchpoint/trace composite (`vice.trace`) + un-ban registers-after-break** ✅ (contradiction confirmed)
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

- [x] **3.3 `find_counters` deterministic static heuristic**
  - **Why:** `find_loops` finds control flow; nothing finds *state*. 6502 game-state idioms are regular and cheap to
    scan: `DEC/INC abs` below `$D000` (lives/timers), `SED…ADC/SBC…CLD` clusters (BCD score), stores into screen RAM
    `$0400-$07E7` with `ORA/ADC #$30` digit chains (HUD), `CMP #$0A` after `INC abs` (decimal rollover / multi-byte).
  - **Fix:** emit labels/hypotheses with evidence (`kind:"ram_var"`, e.g. `candidate_lives_counter`) *before* the LLM
    sees the question. Expose `code_kb mode='suspects' kind='lives|score|timer|sprite|loader|protection'`.
  - **Files:** `tools/c64_disasm.py`, `code_kb/layer0.py`, `graph/code_kb_node.py`, `graph/prompts.py`.
  - **Effort:** ~1 day · **Impact:** High (first-iteration answers for flagship questions) · **Source:** F §3.3, G #5, S
  - **My call:** Do — pure/testable, works offline, composes with 3.1.

- [x] **3.4 Index data references / zero-page use-def in Layer 0** ✅ (only ctrl-flow xrefs today)
  - **Now:** `code_kb` Layer 0 records xrefs only for JSR/JMP/branches; every `LDA/STA/CMP abs` operand is discarded, so
    "what writes $D012?" needs fresh disasm + LLM eyeballing.
  - **Fix:** `ANN_DATAREF` (`src, dst, access∈{r,w,rmw}, index∈{none,x,y}`) captured in the same parse pass; query modes
    `writes_to`/`refs_to`/`zp_roles`/`hardware_refs`. Add zero-page use-def grouping to discover array/table base
    pointers (populates the near-empty `data_structures`).
  - **Files:** `code_kb/schema.py`, `code_kb/layer0.py`, `code_kb/store.py`, `graph/code_kb_node.py`.
  - **Effort:** ~2 days · **Impact:** Medium-High · **Source:** F §3.4, G #4, Ge §3.3, S (all)
  - **My call:** Do — turns many multi-iteration hunts into one SQL query.

- [x] **3.5 Deterministic idiom pre-tagging before Layer-1**
  - **Fix:** table-driven matcher for rigid signatures (raster wait `LDA $D012/CMP/BNE`, `DEX/BNE` delay, memcpy
    `LDA abs,X/STA abs,X/DEX/BNE`, jump-table dispatch, KERNAL trampolines `JMP $FFxx`, SID tick to `$D400-$D418`).
    Feed matches into the Layer-1 prompt as "pre-analysis found: raster_wait at $XXXX" so the LLM confirms rather than
    discovers → higher accuracy + cheaper Layer-1 model.
  - **Files:** `tools/c64_disasm.py`, `code_kb/layer1.py`.
  - **Effort:** ~2 days · **Impact:** Medium · **Source:** F §3.7, G #5, Ge §3.1, S (all)
  - **My call:** Do after 3.4 (shares the instruction index).

- [x] **3.6 Bank-awareness / RAM-under-ROM + illegal-opcode handling** ✅ (E000 penalty confirmed)
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

- [x] **3.7 PETSCII / screen + color RAM decode tool** (C64-specific, cheap)
  - **Fix:** deterministic dump→string extractor: given `$0001`/VIC state or defaults, decode screen RAM (+color RAM)
    to ASCII/PETSCII and write into the KB. Often answers "what does the HUD say / where is score text?" without vision.
  - **Files:** new tool + `graph/nodes.py`/`code_kb`, reuse `code_kb/hardware_pack.py`.
  - **Effort:** ~half day · **Impact:** Medium (unique C64 value) · **Source:** Gr §3.1
  - **My call:** Do — cheap, offline, complements 3.1.

- [x] **3.8 Loop-scoring refinements / pseudocode decompilation / multimodal poke-and-peek**
  - **Source:** F §3.6, Ge §2.2, Ge §2.1
  - **My call:** Defer. Loop-scoring refinements are nice-to-have polish; pseudocode decompilation and poke-and-peek are
    larger bets that depend on 3.1/3.2 landing first and (for poke-and-peek) on `vice.memory.write` + HITL gating (6.1).

### Phase 3 implementation notes (2026-07-18 — for reviewer)

> **STATUS: Phase 3 implemented and hardened. Item 3.8 was completed later and its approval-gated visual experiment
> passed a disposable live VICE baseline. Original Phase 3 baseline:
> 247 tests passing (33 new);
> `git diff --check` clean; `compileall` clean; graph + both runners import; no snapshot/checkpoint artifact
> under the repository's real `sessions/` tree. Awaiting GPT 5.6 Sol review before Phase 4.**

Per item (final behavior):

- **3.3 `find_counters` (flagship, offline).** `tools/c64_disasm.find_counters(insns)` scans a disassembled
  stream for the regular 6502 game-state idioms: `DEC/INC abs` into RAM (< $D000) → lives/timer/counter (DEC
  weighted higher for lives); stores inside a `SED…CLD` window → BCD score bytes; `ORA/ADC #$30` then STA into
  screen RAM → HUD digit; `INC abs` then `CMP #$0A` → multibyte-counter rollover. Exposed as capstone
  `mode='find_counters'` (recursive-disasms from the detected entry first) with an optional `kind` filter.
  The synthesizer extracts the top candidates **mechanically** into `ram_var` labels (`candidate_<kind>_<addr>`)
  — turning "where is the lives counter?" into a first-iteration KB lookup, no LLM.
- **3.4 data-reference index.** New `ANN_DATAREF` annotation + `code_data_refs(src,dst,access,index_reg,
  indirect,source_file,annotation_id)` table (per-source, NULL-safe dedup mirroring `code_xrefs`). Emitted by
  **both** Layer-0 paths (asm parse + on-demand disasm window) for every load/store/cmp/RMW memory operand.
  New code_kb modes `writes_to` / `refs_to` / `hardware_refs` (+ `SELECT DISTINCT` for cross-source dedup)
  answer "who writes $D012 / all SID writes?" as one query instead of fresh disasm + LLM eyeballing. Rows carry
  provenance, so a changed source invalidates its data-refs and they re-materialize with the survivors
  (verified live + full replay). `stats()` reports `data_refs`. (Zero-page use-def *grouping* into
  `data_structures` is the one sub-item I left for a follow-up — the index it needs now exists.)
- **3.5 idiom pre-tagging.** `find_idioms(insns)` table-matches raster wait, delay loop, memcpy, jump-table
  dispatch, KERNAL trampoline, and SID tick; capstone `mode='idioms'`; the synthesizer records each as a
  deterministic-id hypothesis (`h_idiom_<addr>_<idiom>`) so Layer-1 confirms rather than rediscovers.
- **3.6 bank-awareness.** New `banking_state(mem)` reads the dumped `$0001` (LORAM/HIRAM/CHAREN). `find_loops`
  now only penalises the `$E000+` range as ROM when the KERNAL is actually banked in, and extends its scan to
  `$FFF0` under RAM-under-ROM (the old flat −30 systematically hid RAM-under-KERNAL game loops).
  `recursive_disasm` skips seeding HW-vector targets in `$E000+` when the KERNAL ROM is banked in (stops
  burning the insn budget on KERNAL). Capstone `mode='bank'` reports the interpreted port; `vectors` now
  includes the banking block. The illegal-opcode decoder remains the deliberate optional-stretch deferral.
- **3.7 PETSCII/screen decode.** `decode_screen_ram(mem, screen_base, color_base)` maps screen codes → ASCII
  (screen code $00 treated as padding, not `@`, to avoid all-zero-screen noise) and returns printable runs;
  capstone `mode='screen_text'`. Answers "what does the HUD/score display say?" offline.
- **3.1 memory diffing (flagship dynamic).** Pure engine `tools/mem_diff.py` (`classify_region`,
  `diff_snapshots`, `monotonic_scan`, `summarize_diff`) — fully offline/unit-tested. Agent-side VICE
  composites `vice.memory.snapshot` (saves the live 64 KB image to `sessions/<slug>/snapshots/<name>.bin`),
  `vice.memory.diff` (two named snapshots; reserved name `dump` = the ingested dump; omit `b` to diff against a
  fresh live read — I/O regions excluded by default so raster/timer noise doesn't drown the signal), and
  `vice.memory.monotonic_scan` (addresses that changed by a fixed delta across ≥2 snapshots — the lives-counter
  finder, restricted to plausible state regions).
- **3.2 watchpoint trace + registers policy.** `vice.trace {address, frames}` composite: arm a write
  watchpoint → run → read registers (via the internal call path) → parse PC → disasm ±16 bytes around the
  writer — one call answers "how is $XXXX updated?", each sub-call degrading gracefully.
  **Note (superseded by the Hardening pass below):** the first cut un-banned standalone `vice.registers.get`
  on a planner `armed:` flag; that honor-system bypass was removed — see Hardening item 5 (option c):
  `vice.registers.get` is now **always** rejected standalone and registers-at-hit come only through
  `vice.trace`. Prompt cheat-sheet reflects the final policy.

**Wiring:** all new capstone modes (`find_counters`/`idioms`/`screen_text`/`bank`) and code_kb modes
(`writes_to`/`refs_to`/`hardware_refs`) are registered, documented in the planner cheat-sheet, validated by
`step_is_concrete` (so complete steps skip the executor LLM, per 1.1), and — for the structured ones — routed
through the synthesizer's deterministic/mechanical extraction (per 1.2). Snapshots/trace never touch the real
`sessions/` tree in tests (all use `tmp_path`).

**Tests:** 247 passing (33 new): `test_analyzers.py` (find_counters lives/score/HUD/rollover + IO/immediate
exclusion; idioms raster/delay/trampoline/memcpy/SID; extract_data_refs access+index; screen decode +
zero-padding-is-not-noise; banking_state; bank-aware find_loops penalty; ROM-vector seed skip),
`test_data_refs.py` (projection + writes_to/refs_to/hardware_refs + stats + per-source invalidation survives
replay), `test_mem_diff.py` (region classify, IO exclusion, region filter, monotonic scan incl. ≥2-snapshot
guard and IO exclusion, summary), `test_vice_composites.py` (snapshot→diff against dump, diff-against-live,
monotonic over saved snapshots, missing-snapshot error, trace resolves writer / requires address / degrades —
`_vice_call` mocked, snapshots under tmp_path). _(The registers-when-armed test was replaced in the Hardening
pass: `armed:` no longer bypasses the ban — see below.)_

**Known limitations / deliberate choices for review:**
1. Zero-page use-def *grouping* into `data_structures` (part of 3.4) is deferred; the `code_data_refs` index it
   builds on is done and queryable.
2. Illegal/undocumented-opcode decoding (optional stretch of 3.6) is deferred — Capstone MOS65XX still decodes
   the documented set only; banked-region confidence is signalled via `bank`/`vectors`, not a second decoder.
3. 3.1/3.2 composites need a live vice-mcp server to capture snapshots/traces; the offline value is
   dump-vs-live diffing and the pure engine. Multi-frozen-dump cataloguing is the deferred 6.5 path.
4. `find_counters`/`idioms` operate on a recursive-disasm stream from the detected entry, but that stream
   **already seeds the RAM IRQ/BRK/NMI vectors ($0314/$0316/$0318) and the HW vectors** (bank-aware: KERNAL-ROM
   targets are skipped when the ROM is banked in), so IRQ-driven game logic is reached. Code reachable only via
   computed/indexed jumps is still missed — the standard static-analysis reachability limit, unchanged.

### Hardening pass (2026-07-18 — GPT 5.6 Sol review follow-up)

Six concrete review gaps fixed; full suite **284 passing** (was 247), `git diff --check` clean, `compileall`
clean, `graph.build`/`main`/`tools.agent_runner` import, no snapshot/checkpoint artifact under the real
`sessions/` tree.

1. **Broken test assertion (H1).** `test_snapshot_then_diff_against_dump` asserted a truthy `Path`; it now
   asserts the `.bin` exists at `_snapshot_dir(state)/…` with the right size, plus a new
   `test_snapshot_writes_bin_to_derived_path` regression that fails if the file is missing.
2. **Mechanical extraction for the remaining Phase 3 structured outputs (H2).** `_mechanical_extract` now also
   handles: capstone `screen_text` → `text` labels (`screen_text_<addr>`); `vice.memory.monotonic_scan`
   candidates → `ram_var` labels with kind from delta (Δ−1 → lives), confidence capped at 0.7;
   `vice.memory.diff` → the **top ≤12 state-region** changed bytes as short-list `ram_var` labels (I/O
   excluded, region-prioritised — never dumps thousands of bytes). These vice methods (+ snapshot as
   no-fact) are excluded from `_llm_worthy_result`; `vice.trace` stays LLM-worthy. All names are
   address-deterministic so re-runs are idempotent (label PK `(addr,name)`).
3. **find_counters KB noise (H3).** `find_counters` now returns **one candidate per address** — highest-scoring
   kind as primary, the rest folded into `alias_kinds` (merged into evidence). A single `DEC` no longer emits
   both a lives and a timer first-class label; the capstone `kind=` filter matches primary *or* alias.
   (Rollover `multibyte_counter` weighted above a bare `counter` so it wins primary.) Regression:
   `test_find_counters_one_label_per_site`.
4. **vice.trace correctness + hygiene (H4).** Success is now claimed only on a **confirmed hit** — either the
   checkpoint's hit count (`vice.checkpoint.list`) or the resolved PC's instruction demonstrably writing the
   watched address (`_disasm_writes_addr`); a bare PC read is no longer "success". The armed watchpoint is
   **always deleted in a `try/finally`**, including on run failure. Tests: hit resolves writer + cleanup;
   hit-via-checkpoint-list (indexed write); no-hit → `ok=False` + cleanup; run-raises → cleanup still runs.
5. **registers.get un-ban tightened (H5) — chose option (c).** A planner-supplied `armed: true` no longer
   bypasses the ban (it isn't trustworthy proof of a hit). `vice.registers.get` is **always** rejected as a
   standalone step; registers at a hit are available only through `vice.trace`, which arms the checkpoint
   in-process, verifies the hit, and reads registers via the internal `_vice_call` path. Prompt updated;
   `test_registers_get_armed_flag_does_not_bypass_ban` locks it in.
6. **Wiring/coverage holes (H6).** `step_is_concrete` parametrization extended to all Phase 3 modes
   (capstone find_counters/idioms/screen_text/bank; code_kb writes_to/refs_to/hardware_refs; vice
   snapshot/diff/monotonic_scan/trace) including the failure cases (snapshot without name, trace without
   address, writes_to without addr). Added **code_kb_node integration tests** (dispatched through the node,
   not raw SQL) for `writes_to`/`refs_to`/`hardware_refs chip=sid` incl. multi-source DISTINCT dedup, and a
   `find_idioms` **jump_table** test.

**Residual-nit cleanup (post-acceptance, non-blocking).** Two of the reviewer's three residual nits fixed
(the third — no standalone "break then read regs" outside `vice.trace` — is intentional, option c):
- Stale Phase 3 prose that still said registers are allowed with `armed:` now points to the option-(c)
  final policy (this section), removing the contradiction with the prompt.
- `_checkpoint_hit` hardened: it now returns False without a concrete `checkpoint_id`, and `vice.trace` only
  consults the checkpoint list when it captured our id — so a hit on an unrelated breakpoint can't be
  misattributed to our watchpoint when `checkpoint.add` returns no id (falls back to PC/disasm proof).
  Regression: `test_trace_no_checkpoint_id_ignores_unrelated_hit`.

**Deferred-work implementation (3.8, offline portions).** Loop candidates now reward distinct JSR targets and
backward-branch closure, penalize delay/poll loops, expose scoring metrics, and distinguish main loops, IRQ handlers,
branch loops, and tight loops instead of labelling every recurring region a main loop. Code-KB gained a conservative
`pseudocode` mode that requires a verified Layer-0 routine and emits one address-preserving line per instruction;
unsupported instructions remain explicit assembly and no control structures or variable meanings are invented.
The approval-gated `vice.poke_verify` composite requires an evidence-backed address, one candidate byte, and an
explicit visible expectation. It saves a full emulator snapshot, reads the original byte from an optional named bank,
captures a baseline, writes and verifies the candidate by same-bank read-back, captures the after image, and reloads
the snapshot in `finally` before comparison. Older servers fall back to byte restoration; restore failure is critical
and non-retryable. Byte-identical PNGs mechanically contradict the visual hypothesis without spending a vision call.
`frames>0` remains an explicit execution-resume request, not a true frame bound on the current server.

**Live baseline (2026-07-18).** The installed server registers underscore MCP names and requires
`vice_memory_write {address,data:[...]}`; the transport now maps documented dotted names at its boundary, requests
array reads, and decodes the returned two-digit strings as hexadecimal. Against a loaded Artillery Duel session,
`$D020: $FE → $F2` produced distinct before/after PNGs and the configured vision role marked the expected
light-blue→red border change **supported**. Same-bank read-back succeeded, full-snapshot restoration succeeded, and a
post-restore read returned `$FE`. The first pre-experiment game snapshot was reloaded and VICE was left paused so warp
mode would not immediately run the game back to BASIC. No Tavily or golden-suite call was involved.

**Still gated / optional:** zero-page use-def → `data_structures` grouping;
illegal-opcode decoder; multi-dump catalog (6.5); live-VICE CI. The computed/indexed-jump reachability limit
also remains (RAM/HW vectors are seeded, so IRQ logic is covered — see limitation 4 above).

---

## Phase 4 — Correctness/robustness papercuts (cheap, bundle together)

> **STATUS: Phase 4 implemented and hardened (2026-07-18); 4.1–4.8 complete.**
> Full suite **308 passing**; Phase 4 has 17 focused cases plus 6 VICE-alias hardening regressions;
> `compileall` and `git diff --check` clean;
> configured planner/analyst/vision clients instantiate with the expected output budgets; graph and CLI entry-point
> imports are clean. Awaiting independent model review.

- [x] **4.1 `detect_basic_sys` misses `SYS 2064` (space after SYS)** ✅
  - `tools/c64_disasm.py:551-568`: digit loop breaks on the first non-digit, so a space/`(`/shifted-space after `$9E`
    yields `None`. Skip `$20`/`$28`/`$3A` (tolerate leading `+`) before collecting digits; consider scanning the BASIC
    line link-chain instead of a fixed `$0801-$0830` window. **Source:** F §1.8.

- [x] **4.2 `_capstone_linear` silently defaults `start` to `$0801`** ✅
  - `graph/nodes.py:1926` — same silent-default class deliberately fixed for VICE args. Make a missing `start` an error
    for consistency. **Source:** F §3.9.

- [x] **4.3 `kb mode='labels'` / digest matches names only, ignores addresses in the question**
  - `relevant_labels` matches question terms against label *names*; a question naming `$C145` never matches by address.
    Extract `$XXXX` tokens from the question and add an `addr IN (...)` clause. **Source:** F §3.9.

- [x] **4.4 Vision call reuses the analyst JSON-only system prompt**
  - `_describe_screenshot` (`graph/nodes.py:2270-2278`) tells the model to emit analyst JSON while asking for prose. Use
    a minimal neutral system prompt for vision calls; store description + thumbnail path; stop burning three roles on
    silent vision failures (dedicated vision-capable role/flag). **Source:** F §3.8, Gr §3.4.

- [x] **4.5 `vice.memory.read` truncated to 4,000 chars loses data**
  - `graph/nodes.py:2459` — for JSON byte arrays that's ~250 bytes of RAM. Format as compact hex-dump lines
    (16 bytes/line) *before* truncation, as `write_report` already does. **Source:** F §2.9.

- [x] **4.6 Expand Tavily `include_domains` + advanced depth for researcher**
  - `graph/nodes.py:2196-2197` locked to 4 sites; add archive.org, forum64.de, C64 wiki (or make it a config knob);
    pass `search_depth="advanced"`. **Source:** F §3.9.

- [x] **4.7 Per-role `max_tokens` (planner/analyst) to avoid mid-JSON truncation**
  - `defaults.max_tokens: 4096` can truncate an 8–10 step plan mid-JSON, burning the backup chain. Wire the per-agent
    `max_tokens` (config already supports per-agent keys) through `get_llm`. **Source:** F §2.9.

- [x] **4.8 `tool_results` grows unbounded in state**
  - Reducer is `add`; curator can't trim it. Replace with a drop-sentinel reducer or keep only
    `(step_id, ok, event_id)` tuples once persisted. **Source:** F §2.9. **My call:** Defer to Phase 1.7 (batching)
    where the state shape is already being reworked.

### Phase 4 implementation notes (2026-07-18)

1. **BASIC SYS parsing (4.1).** `detect_basic_sys` now follows the tokenized BASIC line-link chain (bounded to
   256 lines) and accepts space, `(`, `:`, `+`, and shifted-space before decimal digits. A bounded legacy-window
   fallback remains for corrupt/live-changing line links. Tests cover every separator and a `SYS` on a linked line
   beyond the old `$0801-$0830` scan window.
2. **Explicit Capstone origin (4.2).** Both dump-backed linear disassembly and no-dump/VICE fallback reject a missing
   or blank `start` with `rejection=missing_arg`; invalid addresses receive `invalid_arg`. Neither path can silently
   reach `$0801`. Tests exercise both dump-present and dump-absent cases and fail if VICE fallback is attempted.
3. **Address-aware label retrieval (4.3).** A shared `$XXXX`/`0xXXXX` extractor feeds `KnowledgeStore.relevant_labels`,
   question digests, and `kb mode=labels`; exact address matches are ORed with name terms. Regression coverage proves
   that an opaque label at `$C145` is returned even when a higher-confidence unrelated label exists.
4. **Dedicated screenshot vision path (4.4).** Screenshot calls now make exactly one `vision`-role request under a
   neutral, prose-oriented system prompt. The returned image is persisted under the session's `screenshots/`
   directory, and the tool result keeps its description, model/role/status, byte/hash metadata, and any save or
   vision error. `screenshot_path` and `thumbnail_path` deliberately point to the same browser-usable source image;
   this avoids adding a raster dependency solely to create a duplicate thumbnail. Even if vision fails, the image
   path and explicit failure metadata survive. Tests lock in one-call behavior, prompt/message shape, persistence,
   successful description metadata, and failure metadata.
5. **Compact memory reads (4.5).** Structured VICE byte arrays are rendered as 16-byte `$ADDR: XX …` hex lines before
   the result limit is applied. Oversized reads retain both head and tail plus an explicit middle-byte omission marker;
   metadata records returned/omitted byte counts, base address, and format. A 4 KiB node-level regression verifies
   that the result contains both `$2000` and `$2FF0`, not a truncated pretty-printed JSON array.
6. **Configurable Tavily research (4.6).** Tavily settings live under `tools.tavily` in `config/llm.json`, with the
   original sources plus `archive.org`, `forum64.de`, and `c64-wiki.com`, `search_depth=advanced`, and a configurable
   result limit. Per-step arguments can override all three settings; limits remain bounded to 1–20. The planner
   cheat-sheet documents the options, and a fake-client test verifies the values passed to Tavily.
7. **Per-role output budgets (4.7).** `get_llm` now validates and honors `agents.<role>.max_tokens` before falling back
   to the global default; the existing GPT-5 minimum-budget guard still applies afterward. Planner and analyst are
   configured for 8192 tokens; the dedicated vision role uses 2048. Factory regression coverage and a real local
   configuration-instantiation check confirm the effective values.
8. **Bounded tool-result state (4.8, completed with the deferred-work pass).** The state reducer now assigns stable,
   occurrence-safe result identities, compacts LLM-considered results to status/provenance rows after their raw payload
   reaches the evidence KB, and retains a hard 96-row recent window. Failed rows keep a bounded repair preview, exact
   per-tool counters survive eviction, reports rehydrate recent raw payloads by event id, and the synthesizer tracks
   processed identities instead of an unsafe list offset. Legacy numeric cursors migrate on first synthesis.

### Hardening pass (2026-07-18)

VICE methods are now canonicalized before policy checks and composite dispatch, closing the `ping`/register-alias
bypass while preserving option (c): standalone register reads remain banned and `_vice_call` is never reached.
Composite shorthands normalize explicitly. Added dump-present/absent `invalid_arg` coverage for Capstone linear mode,
tightened hex extraction to 2–4 digits, and added a real-config/mocked-client budget smoke test. Item 4.8 was completed
later in the deferred-work pass described above.

**Verification:** `tests/test_phase4.py` contains 17 Phase 4 cases and `tests/test_vice_composites.py` adds 6 alias-ban
regressions. The complete suite is **308 passing**; `compileall -q`
passes for `graph`, `memory`, `tools`, `code_kb`, and `tests`; `graph.build`, `main`, and `tools.agent_runner` import;
`git diff --check` reports no whitespace errors. No live Tavily, VICE, or paid vision endpoint is needed by the tests.

---

## Phase 5 — Observability & evaluation (do alongside Phases 1–3)

> **STATUS: Phase 5 implemented (2026-07-18); 5.1–5.5 complete.**
> Offline suite **345 passing, 15 skipped** (the 15 paid/live golden graph cases are opt-in); `compileall` and
> `git diff --check` clean at the original Phase 5 boundary. Phase 6 and the later 5.5 work are documented below.

- [x] **5.1 Golden-question regression suite** (`evals/golden.jsonl`)
  - ~10–20 `(game, question, expected_addresses, expected_keywords)` rows for manually verified facts (5 dumps + asm
    already exist). Pytest-marked slow suite runs the graph, asserts the expected `$XXXX` appears with confidence ≥
    threshold, records LLM calls/tokens/wall-time. **Source:** F §4, G #12, S.
  - **My call:** Do — without it every heuristic change is a guess. Build early so Phase 1–3 have a target.

- [x] **5.2 Unit tests for the deterministic layer**
  - Pure functions with obvious fixtures: `detect_basic_sys`, recursive/SMC disasm, `_rewrite_kb_sql`,
    `_normalize_plan_ids`, `scoping.select_asm_files`, layer0 routine-boundary, "curator doesn't compact twice."
    Add tiny synthetic 64KB fixture dumps. **Source:** F §4, G #12, S.
  - **My call:** Do — cheap and protects exactly the code Phases 0/3/4 touch.

- [x] **5.3 Per-role cost telemetry table in `write_report`** (depends on 1.4) — _delivered with 1.4_
  - role → calls / tokens / failures / backup-role activations. **Source:** F §4.

- [x] **5.4 `run_summary` event per completed run** (question, verdict, confidence, iterations, cost)
  - Makes cross-run behavior queryable and feedable to the planner ("previous runs already established…"). **Source:** F §4.

- [x] **5.5 Human ratings + LangSmith datasets** — **Source:** Gr §4.3. Implemented after the Phase 6 notebook/UI.

### Phase 5 implementation notes (2026-07-18)

1. **Golden graph evaluations (5.1).** `evals/golden.jsonl` contains 15 manually anchored questions across all five
   corpus dumps (three per game), with expected addresses, keywords, and confidence floors. `evals.runner` executes
   each through the real `run_question` graph, isolates KB/report output under `evals/.sessions/`, and writes JSONL
   records containing pass/fail reasons, verdict, confidence, matched/missing evidence, LLM calls/tokens/cost,
   tool calls, run id, and wall time. VICE/Tavily are disabled by default so mutable external state cannot move the
   baseline; `--allow-external-tools` is explicit. Ordinary pytest validates the manifest and evaluator offline;
   `C64RE_RUN_GOLDEN=1 pytest -m golden` (or `python -m evals.runner`) enables the real, potentially paid graph runs.
   Those 15 cases were deliberately **not** invoked during implementation, so the normal suite reports them skipped.
2. **Deterministic protection (5.2).** A reusable sparse 64 KiB fixture exercises BASIC SYS detection, recursive
   JSR reachability, and self-modifying-code detection. Focused tests cover SQL column rewriting without mutating
   string literals, plan-id/dependency normalization, per-game asm scoping/overrides, and Layer-0 routine boundaries
   from JSR targets and explicit gaps. The existing curator regression continues to prove events are never compacted
   twice. Coverage exposed and fixed a real edge case: duplicate planner step ids now receive stable unique suffixes,
   while dependencies resolve to the first occurrence.
3. **Queryable run outcomes (5.4).** Schema v5 adds `run_summary` events plus a derived `run_summaries` table. CLI,
   UI runner, and Studio/load fallback stamp a unique `run_id` and UTC start time; successful report generation
   records question, verdict, confidence, iterations, estimated cost, tokens, LLM/tool calls, elapsed time,
   termination reason, answer excerpt, and report path. Re-reporting the same run is idempotent. Recent outcomes are
   included in the planner/analyst digest (same question first), and summary bookkeeping does not count as new
   substantive evidence for the dead-end detector. Replay, digest visibility, metrics, and idempotence use real stores.
4. **Existing cost table retained (5.3).** The per-role report table delivered with 1.4 remains the human-readable
   view; run summaries and golden JSONL provide the cross-run/machine-readable layer.
5. **Human ratings and explicit dataset export (5.5, deferred-work implementation).** Every archived Streamlit turn
   can be rated `helpful` or `not_helpful` with an optional note. Rating revisions append locally to
   `ratings.jsonl`, identical submissions are idempotent, and the latest revision joins to the archived question,
   answer, evidence, and run metadata. `python -m evals.ratings --session-dir ... --dataset ...` explicitly creates
   or updates deterministic LangSmith examples; the UI never performs an implicit network write. `--local-only`
   previews the joined examples offline. LangSmith behavior is covered with a mocked client; no dataset was uploaded.

### Hardening pass (2026-07-18)

Default pytest now anchors every golden address to its 64 KiB dump, records local bytes/disassembly or decoded 6502
references, and rejects exact-address asm/dump byte conflicts without an LLM call. Dump-derived Blood and Petch
snippets replace or supplement incomplete listings; keyword checks use whole-token patterns, with explicit
entity/animation-state context and `limit`/`limited` morphology, and negative evaluator paths are locked down. The
paid/live suite has not yet been baselined and was not run during hardening; this offline dump gate is the current
baseline. Item 5.5 was completed later, after the Phase 6 notebook supplied stable turn identifiers and UI surfaces.

**Verification boundary:** default `pytest -q` is **345 passing, 15 skipped**; skips are only the explicitly gated
live golden cases. Manifest validation, all evaluator arithmetic, run-summary persistence/replay/digest integration,
and deterministic fixtures run offline on every test invocation. `compileall -q`, entry-point imports, CLI evaluator
help, and `git diff --check` are clean. The later 5.5 tests remain entirely offline.

### Paid golden baseline follow-up (2026-07-18, capped at $1)

A representative three-case live baseline was run against the default provider mix with native structured output
left off and VICE/Tavily disabled. It consumed **142,973 tokens**, **22 LLM calls**, and a deliberately conservative
**$0.7935 estimated cost** (the pricing table uses the highest applicable published context tier where a provider has
tiered rates). Results were **1/3 passing**:

- `wor_border_background` passed at confidence 1.0, finding `$D020` and `$D021` ($0.2995 estimated).
- `bubble_lives_storage` failed: `$045A` missing and confidence 0.45 ($0.3132 estimated).
- `vultures_lives_storage` failed: `$008B` missing and confidence 0.30 ($0.1809 estimated).

The run exposed two rollout constraints. First, Gemini plain-JSON responses repeatedly exhausted a small completion
allowance on reasoning or returned truncated/non-contract JSON, activating expensive backup chains; 2,048 output
tokens helped but did not eliminate critic failures. Second, the graph's USD cap is checked at verdict boundaries,
so a final node's fallback chain can overshoot a per-run cap before routing stops. `C64RE_USD_BUDGET` now makes that
boundary configurable and `C64RE_MAX_OUTPUT_TOKENS` provides an authoritative evaluation-time completion ceiling,
but the USD value is a guardrail rather than a transactional billing limit. A future hard-cap implementation should
reserve worst-case call cost before each primary/backup invocation. The remaining 12 live cases were not run under
this budget; the offline 15-case dump-anchor gate remains complete.

### Paid golden hardening follow-up (2026-07-18, additional $6 authorization)

The first baseline's failures led to three concrete fixes before broader paid
testing: explicitly selected ASM is now included in the planner/analyst digest
(including compact instruction-only excerpts with no lexical keyword match),
every evaluation case uses a fresh session root, and each provider call must
pass a conservative pre-call cost reservation. The runner also enforces a
cumulative `--max-cost-usd` ceiling. Gemini analyst/critic calls used the
role-scoped native structured-output canary with `reasoning_effort=low`; plain
JSON fallback remains available and global native structured output remains
off by default.

Live coverage reached **11/15 cases** with VICE and Tavily disabled. After the
grounding fix, the three representative storage/display cases passed. Bubble
initial/decrement, all three Blood PRNG cases, and all three Petch cases then
found every expected address at or above the configured confidence floor.
Four otherwise correct answers exposed evaluator morphology rather than model
quality (`life/lives`, `limit/limits/limited/bound/bounded`, `flag/flags`, and
qualified phrases such as `animation and setup state`); whole-word patterns
now admit those natural forms while the negative `I will state ...` regression
still fails. The four cases not yet live-baselined are
`vultures_lives_initial`, `vultures_lives_decrement`, `wor_sprite_enable`, and
`wor_screen_ram`.

Machine-readable completed batches are in
`evals/results/golden-smoke-after-grounding-part2.jsonl`,
`golden-blood-after-digest.jsonl`, and `golden-petch-baseline.jsonl`; interrupted
batches retain their per-case reports under `evals/.sessions/`. Completed
hardening calls reported **$3.6018**. Conservatively charging each interrupted
in-flight case its entire remaining case allowance brings the additional-round
upper estimate to **$5.6518**, under the approved $6 ceiling. A credentials-not-
loaded diagnostic attempt recorded zero tokens and zero estimated cost.

The live runs also exposed and fixed a planner-contract edge: descriptive prose
in a Code-KB `role` argument is now normalized to a configured Layer-LLM role,
and a string-valued backup role is treated as one role rather than a character
list. Budget exhaustion now produces an empty planner plan instead of launching
a free but noisy seven-step fallback loop. Current default verification after
the Phase 7/deferred review hardening below is **445 passed, 15 skipped**;
`compileall -q` and `git diff --check` are clean.

---

## Phase 6 — Product/UX (research-notebook direction; larger, optional)

- [x] **6.1 Human-in-the-loop interrupts** (plan edit + mutating-VICE gate)
  - LangGraph `interrupt_after=["planner"]` for plan review (edit/reorder/delete steps), plus approval before mutating VICE calls
    (`memory.write`, poke-and-peek); Stop/cancel in Streamlit. **Source:** Gr §2.1.
  - **My call:** High-trust, but prerequisite for any `memory.write`/poke-and-peek work; sequence before 3.8's poke-and-peek.

- [x] **6.2 Multi-turn research notebook** (turn archive, versioned reports, digest "prior answers" section)
  - Persist `sessions/<game>/turns.jsonl`; version reports (`report_<ts>.md` + latest copy); add prior accepted answers
    + open questions to the digest; open-question chips in Streamlit. **Source:** Gr §2.2. Pairs with 2.4 (real resume).

- [x] **6.3 Hypothesis lifecycle** (mark supported/refuted on accept/contrary evidence) — **Source:** Gr §2.3.
- [x] **6.4 Share one evidence resolver between CLI and UI** — **Source:** Gr §2.4.
- [x] **6.5 Named multi-state dump catalog** (title/ingame/death/level2; diff frozen stages offline) — **Source:** Gr §3.3. Offline complement to 3.1.
- [x] **6.6 Symbol-map export** (VICE monitor `.sym`/`.labels` from high-confidence labels) — **Source:** Gr §3.2.
- [x] **6.7 Synthesizer→Code-KB Layer-1 without Layer-0 gate** — require enclosing Layer-0 window or tag `unverified_llm` so the call-graph UI stays honest. **Source:** Gr §3.5.

- [!] **6.8 DAG-based planner / Code-KB Layer 2+3 / structured outputs (`with_structured_output`)**
  - **Source:** Ge §1.2, G #10a, F §2.6. The offline implementation is complete behind conservative boundaries.
    Cross-provider native structured output remains disabled by default pending a paid golden baseline.

### Phase 6 implementation notes

- **6.1:** the Streamlit runner compiles with `interrupt_after=["planner"]`, pausing after every planner pass. The user can
  edit/reorder/delete the JSON plan, while validation rejects duplicate IDs, missing dependencies, cycles, and unknown
  tools. Mutating VICE methods and aliases are normalized and rejected before MCP/composite dispatch unless that exact
  step was explicitly approved; the CLI stays deny-by-default unless `--approve-vice-mutations` is supplied. The UI can
  cancel a pending run before tool execution and asks again after every replan.
- **6.2:** every completed run is idempotently appended to `sessions/<game>/turns.jsonl`; reports are written to a
  run-specific `report_<timestamp>_<run>.md` plus the `report.md` latest copy. The planner digest includes bounded prior
  accepted answers/open questions, the UI reloads archived turns, and open questions have one-click investigation actions.
- **6.3–6.4:** hypothesis status changes are append-only events (`open` → `supported`/`refuted`) driven by explicit
  critic references, including hypothesis event IDs. Reports, the CLI, and Streamlit use the same resolver for typed
  address, hypothesis, and event references.
- **6.5:** named 64 KiB states are immutable, SHA-256 catalogued copies under the game session; source files can change
  without changing frozen evidence, and any two states can be diffed offline from the UI. Live VICE snapshots are also
  added to this catalog on a best-effort basis.
- **6.6:** Code-KB exports VICE monitor `.labels` commands for names at or above the confidence threshold, excluding
  generic fallbacks, low-confidence names, and `unverified_llm` suggestions; Streamlit exposes the result as a download.
- **6.7:** synthesizer semantics are mirrored only as Layer-1 hypotheses/labels. Claims inside a deterministic Layer-0
  routine bind to that verified window and may enter annotation; claims outside Layer 0 remain queryable but carry
  `unverified_llm` and never create routine/xref ground truth.
- **6.8 (deferred-work implementation):** dependency-ready, independent read-only steps use bounded LangGraph `Send`
  fan-out (item 1.7), while retries, VICE, mutation, and LLM-backed Code-KB writes remain serial. Explicit Code-KB
  `layer2` groups at least two verified Layer-0 routines and rejects hallucinated routine IDs; `layer3` appends
  support/question/reject critiques against exact Layer-1/2 annotation IDs and never mutates Layer 0. Read-only
  `groups`/`critiques` modes expose the results. Native JSON-schema output is opt-in through
  `defaults/agents.*.structured_output` or `C64RE_STRUCTURED_OUTPUT`; provider rejection automatically retries the
  same role through the existing plain-JSON contract before the backup chain. The default remains off until the live
  golden suite establishes provider compatibility and answer-quality parity.

### Hardening pass (2026-07-18)

Hypothesis lifecycle/evidence lookup now uses exact semantic IDs; event provenance uses a separate, unambiguous
exact-or-unique-prefix path. The VICE gate covers reset, autostart, disk/tape/cartridge attachment, snapshot loading,
resource changes, and normalized mutation aliases, with approved and read-only paths locked down offline. Unverified
routine guesses no longer materialize as parent-KB routines; they remain explicit `unverified_llm` hypotheses.

**Verification boundary:** default offline `pytest -q` is **367 passing, 15 skipped**; the skips remain only the
explicitly gated paid/live golden cases. `compileall -q` and `git diff --check` are clean. No live VICE, Tavily, vision,
or paid golden calls were run. The later 6.8 implementation is covered offline; its native-output rollout remains
gated on the paid golden baseline.

---

## Phase 7 — Hygiene / packaging

> **STATUS: Phase 7 implemented and verified (2026-07-18); 7.1–7.3 complete.**

- [x] **7.1 Wheel ships only `graph/`** ✅ — `pyproject.toml:37` `packages = ["graph"]` while runtime needs `memory/`,
  `tools/`, `code_kb/`, `config/`. Package all importable packages + config data (or clearly document editable-only). **Source:** Gr §4.1.
- [x] **7.2 Dead `coordinator`/`researcher` roles** — defined in `config/llm.json` + `graph/llm.py`, never invoked, no
  `ROLE_BLOCKS` entries. Remove them or wire real paths (researcher = owned web research; coordinator = multi-turn
  agenda, pairs with 6.2). Dead config invites accidental provider spend. **Source:** Gr §4.2.
- [x] **7.3 Secret redaction in digests / SQL previews** — `kb mode=sql` can `SELECT` full `text_docs.content`; redact
  API-key-shaped strings; document `text_dir` as trusted LLM-context input. **Source:** Gr §4.4.

### Phase 7 implementation notes

- **7.1:** the wheel now includes `graph`, `memory`, `tools`, `code_kb`, `evals`, a small `c64re_agent` runtime
  package, bundled configuration, the CLI/UI modules, and `langgraph.json`. Runtime paths distinguish packaged
  read-only config from writable workspace/session data and support `C64RE_CONFIG_DIR`, `C64RE_WORKSPACE_DIR`, and
  `C64RE_SESSIONS_DIR` overrides. An offline wheel build plus an isolated target install verified imports, bundled
  config lookup, relocated sessions, and the `c64re --help` entry point.
- **7.2:** removed the configured-but-unreachable coordinator and researcher clients instead of inventing new graph
  paths for them. Tavily remains a planner-owned tool, the dedicated vision path remains explicit, README role/config
  examples now match the graph, and a config regression prevents the dead roles from silently returning.
- **7.3:** original session evidence remains intact, while a shared recursive redactor now protects LLM-facing KB
  digests, note/asm search excerpts, semantic/event previews, and arbitrary parent/Code-KB SQL results and metadata.
  README documents `text_dir`/asm inputs as trusted, locally persisted material and makes the best-effort boundary clear.

### Deferred-work pass verification (2026-07-18)

The complete combined tree passes **445 tests with 15 skipped**; the skips are exactly the opt-in paid/live golden graph
cases. `compileall -q`, entry-point imports, and `git diff --check` are clean. A fresh offline wheel build succeeded,
and an isolated target install verified bundled configuration, relocated session paths, runtime imports, and
`c64re --help`. All VICE, vision, LangSmith, and structured-provider paths added in this pass were exercised with
mocks only. No paid golden, live VICE, Tavily, vision, embedding, or LangSmith dataset call was made.

The later Phase 3.8 live-baseline follow-up is documented in its own section above. It added the real-server transport,
schema, read-back, snapshot-restore, and visual-verdict regressions; the no-live statement here remains the boundary of
the original offline pass, not the final project state.

### Phase 7 / deferred review hardening (2026-07-18)

- Both dedicated vision paths now reserve a conservative image-inclusive
  worst-case call cost before invoking the provider. Reservation denial records
  ordinary usage/rejection telemetry, sets the budget-exhausted signal, skips
  the provider, and preserves the existing vision-unavailable tool fallback;
  successful description and before/after comparison calls still settle actual
  token/cost usage. Generic `_vice_text` screenshot flattening is non-billable
  unless its stateful caller supplies an explicit remaining-budget context.
- `vice.poke_verify` now applies the selected bank consistently to the original
  read, candidate write, verification read, and byte-fallback restore. Snapshot
  restoration remains preferred and the approval/alias gate is unchanged.
- Redaction is idempotent and credential-shape-aware at text assignments:
  C64 addresses such as `secret=$0780` survive, while prefixed API keys,
  bearer tokens, JWTs, and recursive secret fields remain protected. Persisted
  source evidence is still never mutated.
- A failed native-structured call can no longer reuse its uncertain transport
  reservation for a same-role plain retry. The held allowance is deducted
  before `_invoke_one` performs the retry's own reservation check.
- The 96-row limit now applies only to bulky tool-result payloads. Every result
  attempt keeps an unbounded compact status identity, so an older success
  remains done and continues satisfying dependencies after payload pressure;
  event IDs and durable aggregate tool-call statistics remain intact.
- Runtime path helpers remain dynamic, while CLI/UI/graph/semantic modules
  deliberately snapshot their paths at import. The environment-before-import
  contract is documented and tested in a fresh subprocess, including the
  evaluation runner's explicit ability to patch `graph.nodes.SESSIONS_DIR`.

Offline verification: **445 passed, 15 skipped**; the exact focused review
suite, `compileall -q`, and `git diff --check` are clean. No paid/live model,
VICE, Tavily, vision, LangSmith, or embedding call was made, and semantic
indexing remains disabled by default.

---

## Suggested execution order (condensed)

The deferred-work pass used this dependency order after Phases 0–6 were stable:

1. **7.3 → 7.1 → 7.2:** protect prompt boundaries first, make the wheel/install paths real, then remove dead roles.
2. **4.8 → 1.7:** bound persisted graph payloads before allowing concurrent result fan-in; fan-out is read-only and
   capped at four branches.
3. **2.5 cache → 3.8 offline:** remove repeat embedding work, then add loop scoring and conservative pseudocode without
   requiring external services.
4. **5.5:** collect ratings locally only after stable notebook run IDs exist; make LangSmith export explicit.
5. **6.8:** reuse the proven fan-out as the DAG execution layer, add evidence-gated Layer 2/3, and put native
   structured output behind an opt-in compatibility flag.
6. **External baselines last:** validate poke-and-peek against a disposable live VICE session (completed), select
   `text-embedding-3-small` while keeping automatic spend opt-in, then run the paid golden baseline before changing
   provider-facing defaults (11/15 cases complete).

---

## Remaining rollout decisions for review (user + GPT 5.6)

1. **Semantic activation (2.5):** `text-embedding-3-small` is configured and the persistent miss-only cache is
   complete. Decide whether a future release should change `enabled` to `true`; it remains explicit opt-in so opening
   an ordinary session cannot silently incur embedding spend.
2. **Provider rollout (6.8):** role-scoped native structured output for Gemini analyst/critic passed the live canary,
   with compatible plain-JSON fallback. Keep the global default off until the final four golden cases are baselined,
   then decide whether to enable only those roles or widen the rollout.
3. **Paid golden completion (5.1):** 11/15 live cases have been exercised after grounding/contract fixes. The remaining
   Vultures initial/decrement and Wizard of Wor sprite/screen cases need a separate paid completion allowance; pre-call
   reservations and the runner's cumulative cap are now implemented and tested.

### Future live-validation / A-B backlog — do not lose

- **Finish the fixed-input live baseline:** run `vultures_lives_initial`, `vultures_lives_decrement`,
  `wor_sprite_enable`, and `wor_screen_ram` with fresh per-case sessions, VICE/Tavily disabled, and both the per-case
  reservation guard and cumulative runner ceiling enabled. Do not describe the 15-case suite as fully baselined until
  all four have machine-readable results.
- **Structured-output A/B:** on the same manifest and provider/model versions, compare the current plain-JSON default
  (control) with `C64RE_STRUCTURED_OUTPUT_ROLES=analyst,critic` (treatment). Use fresh session roots and at least two
  repetitions per arm when budget permits; compare pass rate, address/keyword misses, confidence, contract/fallback
  failures, input/output tokens, estimated cost, LLM-call count, and wall time. Keep VICE/Tavily off so external state
  does not confound the result. Do not enable native structured output globally unless the treatment has no material
  quality regression and its fallback behavior remains bounded.
- **Optional semantic-search A/B:** before changing `config/kb_semantic.json` to `enabled=true`, compare semantic-off
  against a cold-cache and warm-cache `text-embedding-3-small` run on a small fixed subset. Record embedding calls,
  cache misses/hits, incremental cost, retrieval relevance, answer quality, and reopen behavior. Automatic embedding
  remains opt-in until this establishes a worthwhile benefit and an acceptable spend policy.
- Preserve raw JSONL results and the exact config/environment metadata for every arm. A future comparison should use
  paired case-level deltas rather than mixing results from different manifests, model revisions, or live emulator
  states.

---

_Last updated: 2026-07-18 · Original planning pass plus implementation updates. Historical line numbers reference
the working tree at planning time; re-verify at edit time. Items tagged ✅ were confirmed during planning._
