"""Shared LangGraph state for the C64-RE Agent.

Mirrors `CLAUDE_graph.md` §2. Reducers are declared via `Annotated` so that
LangGraph merges parallel/looped updates without clobbering history.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class C64State(TypedDict, total=False):
    # --- inputs (set once at load_inputs) ---
    game: str
    question: str
    dump_path: str
    partial_asm_path: str | None
    text_dir: str | None
    # Directory of partial-asm files fed to the layered code-comprehension
    # sub-agent (`code_kb`). Independent from `partial_asm_path` so a run
    # can mix one curated partial asm with a directory of community
    # disassemblies, and from `text_dir` because the asm pipeline owns
    # its own parser/store.
    asm_dir: str | None
    # Explicit per-game asm-file selection. When set, this overrides the
    # auto-scoping done by `code_kb.scoping.select_asm_files`. Each entry
    # may be absolute or relative to `asm_dir`. Used when `asm_dir` holds
    # files belonging to several games and the game's slug-token heuristic
    # would otherwise pick up the wrong subset.
    asm_files: list[str] | None

    # --- working memory ---
    kb_handle: str
    # Handle to the *separate* CodeKnowledgeStore that owns Layer-0
    # ground truth and Layer-1+ LLM annotations of the disassembly. Set
    # by `load_inputs` when an asm dir / partial asm is supplied.
    code_kb_handle: str | None
    plan: list[dict[str, Any]]
    current_step_id: str | None
    tool_results: Annotated[list[dict[str, Any]], add]

    # --- analysis ---
    candidate_answer: dict[str, Any] | None
    verdict: dict[str, Any] | None
    history: Annotated[list[dict[str, Any]], add]

    # Question-relevant KB digest, refreshed by the synthesizer / curator
    # so the analyst and critic see the same evidence sheet.
    kb_digest: str

    # Snapshot of the *distinct substantive fact* count (bookkeeping
    # kinds excluded; failed tool attempts excluded; identity is stable
    # across replanned step IDs) at the last critic verdict.
    # `critic_node` compares the fresh count against this snapshot BEFORE
    # overwriting it and ships the result on the verdict as
    # `kb_grew_since_last_verdict`, which the dead-end router reads.
    last_kb_event_count: int

    # Set by the executor when every pending step has an unsatisfiable
    # dependency (failed prerequisite, missing id, or cycle); routes
    # deterministically to a replan. Cleared by the planner.
    plan_blocked: bool

    # Stamped post-hoc by the runners when a run ends without a critic
    # accept (e.g. "recursion_exhausted") — see
    # `plan_utils.normalize_truncated_state`.
    termination_reason: str | None

    # Index into `tool_results` up to which the synthesizer has already
    # considered results for LLM extraction. Results past this index are
    # accumulated and extracted in one batched LLM call when the plan
    # drains or the batch fills (tracker 1.2). `tool_results` only grows
    # (its reducer is `add`), so an index high-water mark is stable.
    synth_processed_count: int

    # --- control ---
    iteration: int
    replan_count: int
    # Consecutive `revise` verdicts; reset on accept/replan. The critic
    # forces `accept` once this reaches MAX_CONSECUTIVE_REVISES so the
    # analyst↔critic ping-pong terminates (tracker 0.4).
    revise_count: int
    # Accumulated estimated USD across all LLM calls (tracker 1.4).
    # Nodes return per-drain deltas; stays 0.0 when config/llm.json has
    # no "pricing" section (tokens are still tracked in `llm_usage`).
    budget_used: Annotated[float, add]
    # Accumulated total tokens (input+output) across all LLM calls —
    # the always-available budget fallback (`routers.token_budget()`)
    # so unpriced models still hit a runtime cap. Deliberately a
    # separate field from `budget_used`: no unit mixing.
    tokens_used: Annotated[int, add]
    # One entry per LLM call: role, model, tokens, cost, ok (usable
    # contract response) / transport_ok / error. Aggregated per-role in
    # the report (tracker 1.4).
    llm_usage: Annotated[list[dict[str, Any]], add]

    # --- transcript (LangChain message log; viewable in Studio/LangSmith) ---
    messages: Annotated[list, add_messages]
