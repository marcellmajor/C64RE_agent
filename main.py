"""Thin CLI runner for the C64-RE Agent.

Usage:
    python main.py \\
        --game "Boulder Dash" \\
        --question "Where is the diamond counter stored?" \\
        --dump inputs/boulderdash.dump

For an interactive view of the graph, run instead:
    langgraph dev          # opens LangGraph Studio (and streams to LangSmith)
"""

from __future__ import annotations

# load_dotenv MUST run before any langchain/langgraph imports so that
# LANGCHAIN_TRACING_V2 / LANGSMITH_* are visible when LangChain first
# initialises its callback managers.
import os
from dotenv import load_dotenv
load_dotenv()

import argparse
import hashlib
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from graph.build import build_graph
from graph.plan_utils import (
    RECURSION_LIMIT,
    normalize_truncated_state,
    resolve_session_slug,
    slugify,
)
from c64re_agent.paths import sessions_dir

SESSIONS_DIR = sessions_dir()


def _default_thread_id(game: str, question: str) -> str:
    """Durable per-question thread id (tracker 2.4).

    Same game + question → same thread, so re-running after a crash
    continues on top of the checkpointed state. A different question
    gets a fresh thread: accumulated state (bounded tool results, messages,
    LLM usage) would otherwise bleed between questions.
    `--thread-id` overrides for explicit cross-run continuation.
    """
    q_hash = hashlib.sha1((question or "").encode("utf-8")).hexdigest()[:8]
    return f"{slugify(game)}-{q_hash}"


def _open_checkpointer(game: str):
    """Context manager yielding the durable per-game checkpointer.

    `--thread-id` claimed "Resume an existing session" but the CLI
    compiled with `MemorySaver`, so nothing ever survived the process
    (tracker 2.4). Checkpoints now live in
    ``sessions/<slug>/checkpoint.sqlite`` (wiped together with the game
    by `scripts/purge_persistence.py --game`). Falls back to in-memory
    checkpoints with a warning when langgraph-checkpoint-sqlite is
    unavailable — the run still works, only resume is disabled.
    """
    slug = resolve_session_slug(game, SESSIONS_DIR)
    ckpt_dir = SESSIONS_DIR / slug
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        print(
            "WARNING: langgraph-checkpoint-sqlite not installed — "
            "checkpoints are in-memory and --thread-id will NOT resume "
            "across runs."
        )
        return nullcontext(MemorySaver())
    return SqliteSaver.from_conn_string(str(ckpt_dir / "checkpoint.sqlite"))


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _configure_langsmith_tracing() -> None:
    """Normalize tracing env vars and avoid silent misconfiguration."""

    # Accept either modern LangSmith or legacy LangChain variable names.
    if os.getenv("LANGSMITH_TRACING") and not os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "")
    if os.getenv("LANGCHAIN_TRACING_V2") and not os.getenv("LANGSMITH_TRACING"):
        os.environ["LANGSMITH_TRACING"] = os.getenv("LANGCHAIN_TRACING_V2", "")

    if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    if os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

    tracing_on = _truthy_env("LANGCHAIN_TRACING_V2") or _truthy_env("LANGSMITH_TRACING")
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if tracing_on and not api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        print(
            "LangSmith tracing disabled: tracing=true but no API key found "
            "(set LANGSMITH_API_KEY or LANGCHAIN_API_KEY)."
        )
        return

    if tracing_on:
        project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "default"
        endpoint = os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
        print(
            "LangSmith tracing enabled: "
            f"project={project!r}, endpoint={endpoint}"
        )
    else:
        print("LangSmith tracing is off (set LANGSMITH_TRACING=true to enable).")


def _resolve_evidence(ref: str, store: "Any") -> str:
    from memory.evidence import resolve_evidence_ref

    return resolve_evidence_ref(ref, store)


def main() -> None:
    _configure_langsmith_tracing()

    parser = argparse.ArgumentParser(description="C64 Reverse Engineering Agent")
    parser.add_argument("--game", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--partial-asm", type=Path, default=None)
    default_text_dir = Path("text_dir")
    parser.add_argument(
        "--text-dir", type=Path,
        default=default_text_dir if default_text_dir.is_dir() else None,
        help="Directory of human-written notes (.txt/.md/.rst/...). "
             "Recursively ingested into the KB so the agent can search them. "
             "Default: ./text_dir if it exists.",
    )
    default_asm_dir = Path("asm_dir")
    parser.add_argument(
        "--asm-dir", type=Path,
        default=default_asm_dir if default_asm_dir.is_dir() else None,
        help="Directory of partial-asm files (.asm/.txt/.s) fed into the "
             "layered code-comprehension sub-agent (`code_kb`). "
             "Default: ./asm_dir if it exists.",
    )
    parser.add_argument(
        "--asm-files", type=str, nargs="*", default=None,
        help="Explicit list of asm files for this game (overrides the "
             "filename-token heuristic). Paths may be absolute or "
             "relative to --asm-dir. Use this when --asm-dir holds files "
             "from multiple games and you want to scope this run to one.",
    )
    parser.add_argument(
        "--reset-code-kb", action="store_true",
        help="Wipe sessions/<game>/code_kb/ before running. Use when a "
             "previous run polluted the code KB with files from a "
             "different game.",
    )
    parser.add_argument(
        "--thread-id", default=None,
        help="Durable checkpoint thread to use/resume "
             "(default: <game-slug>-<question-hash>). Checkpoints persist "
             "in sessions/<slug>/checkpoint.sqlite: re-running with the "
             "same thread id continues on top of that thread's saved "
             "state (e.g. after a crash). NOTE: re-invocation restarts "
             "the graph from the planner over the checkpointed state; "
             "mid-node resume is future work (tracker 6.2).",
    )
    parser.add_argument("--no-trace", action="store_true",
                        help="Disable LangSmith tracing for this run.")
    parser.add_argument(
        "--approve-vice-mutations", action="store_true",
        help="Allow plan steps that mutate live VICE state. Without this "
             "flag, memory writes, execution controls, checkpoints, input "
             "injection, and vice.trace are rejected before the MCP call.",
    )
    args = parser.parse_args()

    if args.reset_code_kb:
        slug = resolve_session_slug(args.game, SESSIONS_DIR)
        ck = SESSIONS_DIR / slug / "code_kb"
        if ck.exists():
            import shutil as _sh
            _sh.rmtree(ck)
            print(f"Reset code_kb at {ck}.")

    if args.no_trace:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    initial_state = {
        "run_id": uuid.uuid4().hex,
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "game": args.game,
        "question": args.question,
        "dump_path": str(args.dump),
        "partial_asm_path": str(args.partial_asm) if args.partial_asm else None,
        "text_dir": str(args.text_dir) if args.text_dir else None,
        "asm_dir": str(args.asm_dir) if args.asm_dir else None,
        "asm_files": list(args.asm_files) if args.asm_files else None,
        "plan": [],
        "current_step_ids": [],
        "tool_results": [],
        "tool_call_stats": {},
        "history": [],
        "messages": [],
        "require_vice_approval": True,
        "approved_mutation_steps": [],
        "approve_all_vice_mutations": bool(args.approve_vice_mutations),
    }

    thread_id = args.thread_id or _default_thread_id(args.game, args.question)
    # LangGraph defaults to 25 super-steps — one 7-step plan iteration.
    # RECURSION_LIMIT budgets the full MAX_ITERS loop (tracker 0.1).
    config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": thread_id},
    }

    print(f"=== Running C64-RE on {args.game!r} (thread_id={thread_id}) ===\n")
    seen = 0
    recursion_exhausted = False
    # Durable checkpoints (tracker 2.4): everything that touches the
    # graph must run inside the saver's context so its connection stays
    # open through the final `get_state`.
    with _open_checkpointer(args.game) as saver:
        graph = build_graph().compile(checkpointer=saver)
        try:
            for event in graph.stream(initial_state, config,
                                      stream_mode="values"):
                msgs = event.get("messages") or []
                for m in msgs[seen:]:
                    print(f"  · {m.content}", flush=True)
                seen = len(msgs)
        except GraphRecursionError:
            # Degrade into a best-effort low-confidence report instead of a
            # stack trace: write_report is only reachable via the critic
            # router, so on recursion exhaustion it never ran.
            recursion_exhausted = True
            print(
                f"\n!! Recursion limit ({RECURSION_LIMIT} super-steps) "
                "exhausted — emitting best-effort report from the last "
                "completed state."
            )

        final = graph.get_state(config).values or {}

    if recursion_exhausted:
        # Normalize BEFORE reporting/printing: stamp the termination
        # reason, cap the (never critic-accepted) confidence, and append
        # an open question explaining the truncation, so neither the
        # report nor the CLI output presents the checkpoint state as a
        # validated answer.
        final = normalize_truncated_state(
            final,
            reason="recursion_exhausted",
            detail=(
                f"the {RECURSION_LIMIT}-super-step recursion limit was "
                "reached before the critic accepted an answer; findings "
                "reflect the last completed iteration only."
            ),
        )
        try:
            from graph.nodes import write_report
            write_report(final)
        except Exception as e:  # noqa: BLE001
            print(f"Could not write best-effort report: {type(e).__name__}: {e}")
    verdict = (final.get("verdict") or {}).get("decision", "n/a")
    candidate = final.get("candidate_answer") or {}

    print("\n" + "=" * 72)
    print(f" FINAL VERDICT: {verdict}   ·   confidence: "
          f"{candidate.get('confidence', 0)}")
    print("=" * 72)
    print("\n--- Answer ---")
    print(candidate.get("answer") or "(no answer produced)")

    evidence = candidate.get("evidence") or []
    if evidence:
        print("\n--- Evidence ---")
        try:
            from memory.store import get_store
            sessions = SESSIONS_DIR
            game_slug = resolve_session_slug(args.game, sessions)
            ev_store = get_store(sessions / game_slug / "kb")
            for e in evidence:
                print(f"  · {_resolve_evidence(str(e), ev_store)}")
        except Exception:
            for e in evidence:
                print(f"  · {e}")

    open_qs = candidate.get("open_questions") or []
    if open_qs:
        print("\n--- Open questions ---")
        for q in open_qs:
            print(f"  · {q}")

    critique = (final.get("verdict") or {}).get("critique")
    if critique:
        print("\n--- Critic notes ---")
        print(critique)

    sessions = SESSIONS_DIR
    report_path = (
        sessions / resolve_session_slug(args.game, sessions) / "report.md"
    )
    if report_path.exists():
        print(f"\nFull report: {report_path}")

    # Flush any buffered LangSmith traces before the process exits.
    # Without this, background-thread batches are silently dropped.
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        pass


if __name__ == "__main__":
    main()
