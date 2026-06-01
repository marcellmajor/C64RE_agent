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
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from graph.build import build_graph


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
    """Turn an analyst evidence ref into a human-readable description.

    Handles:
    - $XXXX / $XXXX-$YYYY  → label name(s) from KB
    - h<n>/...             → hypothesis text
    - UUID (event id)      → event kind + source + payload snippet
    - file path            → kept as-is (already readable)
    - bare word / unknown  → kept as-is
    """
    import re as _re

    # --- address ref: $0780 or $0600-$063F ---
    addr_match = _re.match(r"^\$([0-9A-Fa-f]{1,4})(?:-\$[0-9A-Fa-f]{1,4})?$", ref.strip())
    if addr_match:
        addr = int(addr_match.group(1), 16)
        rows = store.query(
            "SELECT name, kind, confidence FROM labels WHERE addr=? ORDER BY confidence DESC LIMIT 3",
            (addr,),
        )
        if rows:
            names = ", ".join(
                f"{r['name']} ({r['kind']}, conf={r['confidence']:.1f})" for r in rows
            )
            return f"{ref}  →  {names}"
        return ref

    # --- hypothesis ref: h3/open, h7/confirmed, etc. ---
    hyp_match = _re.match(r"^(h\d+)(/\w+)?$", ref.strip())
    if hyp_match:
        hid = hyp_match.group(1)
        rows = store.query(
            "SELECT text, status FROM hypotheses WHERE id LIKE ? LIMIT 1",
            (f"%{hid}%",),
        )
        if rows:
            text = (rows[0]["text"] or "")[:120]
            return f"{ref}  →  [{rows[0]['status']}] {text}"
        return ref

    # --- UUID event id ---
    uuid_match = _re.match(
        r"^[0-9a-f]{8,12}$|^[0-9a-f]{8}-[0-9a-f]{4}-", ref.strip(), _re.IGNORECASE
    )
    if uuid_match:
        rows = store.query(
            "SELECT kind, source, substr(payload_json,1,160) AS snippet"
            " FROM events WHERE id=? OR id LIKE ? LIMIT 1",
            (ref.strip(), f"{ref.strip()}%"),
        )
        if rows:
            r = rows[0]
            snippet = (r["snippet"] or "").replace("\n", " ")
            return f"{ref}  →  [{r['kind']}] {r['source']}: {snippet}"
        return ref

    # --- file path or anything else — return verbatim ---
    return ref


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
    parser.add_argument("--thread-id", default=None,
                        help="Resume an existing session (default: <game-slug>)")
    parser.add_argument("--no-trace", action="store_true",
                        help="Disable LangSmith tracing for this run.")
    args = parser.parse_args()

    if args.reset_code_kb:
        slug = args.game.strip().lower().replace(" ", "_")
        ck = Path("sessions") / slug / "code_kb"
        if ck.exists():
            import shutil as _sh
            _sh.rmtree(ck)
            print(f"Reset code_kb at {ck}.")

    if args.no_trace:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    graph = build_graph().compile(checkpointer=MemorySaver())

    initial_state = {
        "game": args.game,
        "question": args.question,
        "dump_path": str(args.dump),
        "partial_asm_path": str(args.partial_asm) if args.partial_asm else None,
        "text_dir": str(args.text_dir) if args.text_dir else None,
        "asm_dir": str(args.asm_dir) if args.asm_dir else None,
        "asm_files": list(args.asm_files) if args.asm_files else None,
        "plan": [],
        "tool_results": [],
        "history": [],
        "messages": [],
    }

    thread_id = args.thread_id or args.game.lower().replace(" ", "-")
    config = {"configurable": {"thread_id": thread_id}}

    print(f"=== Running C64-RE on {args.game!r} (thread_id={thread_id}) ===\n")
    seen = 0
    for event in graph.stream(initial_state, config, stream_mode="values"):
        msgs = event.get("messages") or []
        for m in msgs[seen:]:
            print(f"  · {m.content}", flush=True)
        seen = len(msgs)

    final = graph.get_state(config).values
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
            game_slug = (args.game or "unknown").strip().lower().replace(" ", "_")
            ev_store = get_store(Path("sessions") / game_slug / "kb")
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

    game_slug = (args.game or "unknown").strip().lower().replace(" ", "_")
    report_path = Path("sessions") / game_slug / "report.md"
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
