"""C64-RE Agent — Streamlit chat UI with a code-lens panel.

Run with:

    streamlit run app.py

The app keeps each user message as a turn against the LangGraph agent.
After a turn finishes, addresses (`$XXXX`) cited in the analyst answer
auto-populate a "code lens" panel below the chat: disassembly snippets
(pulled from the layered `code_kb`) + an interactive call graph.

Multi-step prompting is just the regular chat loop — every send-button
press is a fresh agent invocation. Knowledge accumulates across turns
because the underlying `KnowledgeStore` and `CodeKnowledgeStore` are
file-backed and keyed off the game slug.
"""

from __future__ import annotations

# `load_dotenv` MUST run before any langchain / langgraph imports so that
# LANGCHAIN_TRACING_V2 / LANGSMITH_* are visible at import time.
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
from pathlib import Path

import streamlit as st

# Make sibling packages importable when streamlit launches us from
# anywhere; without this the app blows up if the user runs `streamlit
# run app.py` from a parent directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_UI_PREFS_PATH = _HERE / "sessions" / ".ui_prefs.json"

from tools.agent_runner import (  # noqa: E402
    annotate_routine,
    TurnResult,
    call_graph_dot,
    code_kb_summary,
    disasm_for_addresses,
    list_dump_candidates,
    parse_addresses,
    preview_asm_scoping,
    reset_code_kb,
    run_question,
    top_routines_by_xrefs,
)


def _slugify_game_name(game: str) -> str:
    return (game or "").strip().lower().replace(" ", "_")


def _known_game_slugs() -> list[str]:
    sessions = _HERE / "sessions"
    if not sessions.exists():
        return []
    return sorted(
        p.name for p in sessions.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def _slug_to_display(slug: str) -> str:
    return slug.replace("_", " ")


def _load_ui_prefs() -> dict:
    if not _UI_PREFS_PATH.exists():
        return {}
    try:
        data = json.loads(_UI_PREFS_PATH.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_ui_prefs(prefs: dict) -> None:
    _UI_PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _UI_PREFS_PATH.write_text(json.dumps(prefs, indent=2, sort_keys=True))


def _persist_sidebar_prefs(game: str, text_dir: str) -> None:
    prefs = _load_ui_prefs()
    changed = False

    if prefs.get("global_text_dir", "") != (text_dir or ""):
        prefs["global_text_dir"] = text_dir or ""
        changed = True

    game_name = (game or "").strip()
    if game_name:
        if prefs.get("last_game", "") != game_name:
            prefs["last_game"] = game_name
            changed = True
        slug = _slugify_game_name(game_name)
        by_game = prefs.get("text_dir_by_game")
        if not isinstance(by_game, dict):
            by_game = {}
            prefs["text_dir_by_game"] = by_game
        if by_game.get(slug, "") != (text_dir or ""):
            by_game[slug] = text_dir or ""
            changed = True

    if changed:
        _save_ui_prefs(prefs)


# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #


st.set_page_config(
    page_title="C64-RE Agent", layout="wide", page_icon="🕹️",
)


def _init_state() -> None:
    """Initialise session state with sensible defaults."""
    prefs = _load_ui_prefs()
    default_notes_dir = str(_HERE / "text_dir") if (_HERE / "text_dir").is_dir() else ""
    ss = st.session_state
    ss.setdefault("history", [])               # list[dict] — chat turns
    ss.setdefault("game", str(prefs.get("last_game") or ""))
    ss.setdefault("dump_path", "")
    ss.setdefault("memdump_dir",
                  str(_HERE / "memdump_dir") if (_HERE / "memdump_dir").is_dir() else "")
    ss.setdefault("asm_dir", str(_HERE / "asm_dir") if (_HERE / "asm_dir").is_dir() else "")
    ss.setdefault(
        "text_dir",
        str(prefs.get("global_text_dir") or default_notes_dir),
    )
    ss.setdefault("partial_asm", "")
    ss.setdefault("asm_files_selected", [])   # list[str] — explicit per-game selection
    ss.setdefault("running", False)
    ss.setdefault("focus_addr", None)
    ss.setdefault("focus_hops", 1)
    # Inputs the user types but hasn't submitted yet — kept here so a
    # rerun (e.g. clicking "show call graph") doesn't blow them away.
    ss.setdefault("pending_question", "")
    # Track the slug we last computed defaults for so the asm-file
    # multi-select auto-refreshes when the user changes the game.
    ss.setdefault("last_scoped_slug", "")


_init_state()


# --------------------------------------------------------------------------- #
# Sidebar — session config
# --------------------------------------------------------------------------- #


with st.sidebar:
    st.header("Session")
    # Apply any pending game switch BEFORE rendering the text_input widget.
    # (Cannot write st.session_state.game after key='game' widget is rendered.)
    _pending_game = st.session_state.pop("_pending_game", None)
    if _pending_game is not None:
        st.session_state.game = _pending_game

    # Bind directly to session_state via `key=` instead of mirroring with
    # `st.session_state.x = st.text_input(..., value=st.session_state.x)`.
    # The mirror pattern silently breaks set_value() and confuses the
    # widget's own state ↔ default-value precedence.
    st.text_input(
        "Game", key="game",
        placeholder="Boulder Dash",
        help="Used to slugify the persistent KB directory under sessions/. "
             "Also drives auto-selection of dumps and asm files when "
             "memdump_dir / asm_dir hold material from several games.",
    )
    known_slugs = _known_game_slugs()
    if known_slugs:
        # Apply any pending picker-slug update BEFORE the selectbox renders.
        # Always drive the widget via session_state; never pass index= alongside
        # a keyed widget (Streamlit warns/errors when both are present).
        _pending_picker_slug = st.session_state.pop("_pending_game_picker_slug", None)
        if _pending_picker_slug is not None:
            st.session_state["_game_picker_slug"] = _pending_picker_slug

        current_slug = _slugify_game_name(st.session_state.game)
        picker_options = ["(keep typed game)", *known_slugs]
        # Seed the key on first render so the widget shows the right default.
        if "_game_picker_slug" not in st.session_state:
            st.session_state["_game_picker_slug"] = (
                current_slug if current_slug in picker_options else "(keep typed game)"
            )

        def _on_game_picker_change():
            slug = st.session_state.get("_game_picker_slug")
            if slug and slug != "(keep typed game)":
                st.session_state["_pending_game"] = _slug_to_display(slug)
                # Also stage the saved notes dir for that game so it can be
                # applied before the text_dir widget renders on the next rerun.
                _prefs = _load_ui_prefs()
                _by_game = _prefs.get("text_dir_by_game")
                if isinstance(_by_game, dict):
                    _saved = _by_game.get(slug)
                    if isinstance(_saved, str) and _saved:
                        st.session_state["_pending_text_dir"] = _saved

        st.selectbox(
            "Analyzed games",
            options=picker_options,
            format_func=(
                lambda s: s
                if s == "(keep typed game)"
                else _slug_to_display(s)
            ),
            help="Quickly switch to a game that already has session data.",
            key="_game_picker_slug",
            on_change=_on_game_picker_change,
        )

    game_slug = (
        (st.session_state.game or "").strip().lower().replace(" ", "_")
    )
    game_just_changed = game_slug != st.session_state.last_scoped_slug

    if game_just_changed:
        # Restore per-game notes dir via the staging key so the text_dir
        # widget picks it up consistently (same pattern as the picker path).
        prefs = _load_ui_prefs()
        by_game = prefs.get("text_dir_by_game")
        if isinstance(by_game, dict):
            saved_notes = by_game.get(game_slug)
            if isinstance(saved_notes, str) and saved_notes:
                st.session_state["_pending_text_dir"] = saved_notes
        elif isinstance(prefs.get("global_text_dir"), str):
            st.session_state["_pending_text_dir"] = (
                prefs.get("global_text_dir") or st.session_state.text_dir
            )
        # Sync the "Analyzed games" picker to reflect the typed game name.
        # The selectbox ignores index= once its key exists in widget state, so
        # we stage the update and apply it before the widget renders next run.
        if game_slug in known_slugs:
            st.session_state["_pending_game_picker_slug"] = game_slug
        else:
            st.session_state["_pending_game_picker_slug"] = "(keep typed game)"

    # ---------- Memory dump picker ---------- #
    st.text_input(
        "Memory-dump dir", key="memdump_dir",
        help="Folder containing one or more .bin/.dump files. "
             "The picker below highlights the file matching the game name.",
    )
    dump_choices = list_dump_candidates(
        st.session_state.memdump_dir or None, game=st.session_state.game,
    )
    matched_dumps = dump_choices["matched"]
    other_dumps = dump_choices["others"]
    all_dumps = matched_dumps + other_dumps

    if game_just_changed:
        # Force the dump back to the new token-matched file. Without
        # this, Streamlit's widget state would stick to the previous
        # game's dump even after the user retypes the game name.
        if matched_dumps:
            st.session_state.dump_path = matched_dumps[0]
        elif st.session_state.dump_path not in all_dumps:
            st.session_state.dump_path = all_dumps[0] if all_dumps else ""

    if all_dumps:
        labelled = (
            [f"★ {Path(p).name}" for p in matched_dumps]
            + [Path(p).name for p in other_dumps]
        )
        try:
            default_idx = all_dumps.index(st.session_state.dump_path)
        except ValueError:
            default_idx = 0
        # Per-game key → fresh widget when the user switches games, so
        # the selectbox picks up the new default_idx instead of clinging
        # to the previous game's selection.
        idx = st.selectbox(
            "Pick dump file", range(len(labelled)),
            format_func=lambda i: labelled[i],
            index=default_idx,
            help="★ = filename token-matches the game name.",
            key=f"_dump_picker_{game_slug}",
        )
        st.session_state.dump_path = all_dumps[idx]
    elif st.session_state.memdump_dir:
        st.caption(f"_no dumps found in {st.session_state.memdump_dir}_")

    # Editable canonical path. Bound to st.session_state.dump_path so
    # the agent's run_question() picks up whatever's here regardless of
    # whether the user touched the picker.
    st.text_input(
        "Memory-dump path", key="dump_path",
        help="Resolved path passed to the agent. The picker above and "
             "the game-name auto-match update this; you can also edit "
             "it directly to point outside memdump_dir/.",
    )

    # ---------- Asm-file selection ---------- #
    st.divider()
    st.subheader("Asm files for this game")
    st.text_input(
        "Partial-asm dir", key="asm_dir",
        help="Directory of community / hand-written disassemblies. "
             "Files belonging to other games will be skipped automatically "
             "by token-matching their filenames against the game name. "
             f"Tip: put files under {st.session_state.asm_dir}/{game_slug}/ "
             "to lock them to this game.",
    )

    if game_just_changed:
        # Reset the multiselect's bound state so the per-game heuristic
        # default takes over. We re-prime it below once we know what
        # the heuristic picked.
        st.session_state.asm_files_selected = []
        st.session_state.last_scoped_slug = game_slug

    scoping_preview = preview_asm_scoping(
        game=st.session_state.game,
        asm_dir=st.session_state.asm_dir or None,
        override=st.session_state.asm_files_selected or None,
        extra_files=(
            [st.session_state.partial_asm]
            if st.session_state.partial_asm else None
        ),
    )

    # All files under asm_dir, used to populate the multi-select.
    asm_dir_path = Path(st.session_state.asm_dir) if st.session_state.asm_dir else None
    all_asm_files: list[str] = []
    if asm_dir_path and asm_dir_path.is_dir():
        all_asm_files = sorted(
            str(p) for p in asm_dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".asm", ".txt", ".s"}
        )

    if all_asm_files:
        # Pre-prime the bound state with the heuristic suggestion the
        # first time this game is seen; subsequent reruns honour the
        # user's edits because the widget owns the state from then on.
        if game_just_changed and not st.session_state.asm_files_selected:
            st.session_state.asm_files_selected = [
                p for p in scoping_preview["selected"] if p in all_asm_files
            ]

        rel_label = lambda p: str(Path(p).relative_to(asm_dir_path)) \
            if asm_dir_path and Path(p).is_relative_to(asm_dir_path) else p

        st.multiselect(
            f"Files to ingest for **{game_slug or '?'}**",
            options=all_asm_files,
            key="asm_files_selected",
            format_func=rel_label,
            help="Defaults to the auto-scoped subset. Edit freely; "
                 "your picks override the heuristic.",
        )

        st.caption(
            f"_strategy: **{scoping_preview['strategy']}**_ · "
            f"tokens: `{scoping_preview['tokens']}` · "
            f"{scoping_preview['suggestion']}"
        )
        if scoping_preview["skipped"]:
            with st.expander(f"Skipped ({len(scoping_preview['skipped'])})"):
                for s in scoping_preview["skipped"]:
                    st.markdown(f"- `{Path(s['path']).name}` — _{s['reason']}_")
    else:
        st.caption("_no asm files in the given directory_")

    st.text_input(
        "Single partial-asm file (always included)",
        key="partial_asm",
        help="Optional. Useful when one curated disassembly should "
             "always join whatever the multi-select chose.",
    )

    # ---------- Notes dir ---------- #
    st.divider()
    # Apply any pending notes-dir switch BEFORE the widget renders.
    _pending_text_dir = st.session_state.pop("_pending_text_dir", None)
    if _pending_text_dir is not None:
        st.session_state.text_dir = _pending_text_dir
    st.text_input(
        "Notes dir", key="text_dir",
        help="Directory of free-form notes (.txt/.md/.rst) ingested into "
             "the parent KB for retrieval.",
    )
    _persist_sidebar_prefs(st.session_state.game, st.session_state.text_dir)

    # ---------- Maintenance ---------- #
    st.divider()
    col_a, col_b = st.columns(2)
    if col_a.button("Clear chat"):
        st.session_state.history = []
        st.session_state.focus_addr = None
        st.rerun()
    if col_b.button("Reset code KB", help=(
        "Wipe sessions/<game>/code_kb/. Use after you change the asm-file "
        "selection so the next run rebuilds cleanly without leftovers."
    )):
        if st.session_state.game:
            removed = reset_code_kb(st.session_state.game)
            st.toast(
                "Code KB reset." if removed else "No code KB to reset.",
                icon="🧹" if removed else "ℹ️",
            )
            st.rerun()
        else:
            st.warning("Set a game name first.")

    st.divider()
    st.subheader("Code KB")
    summary = code_kb_summary(st.session_state.game) if st.session_state.game else None
    if summary is None:
        st.caption(
            "_no code-KB on disk yet — submit a question to populate it._"
        )
    else:
        c1, c2 = st.columns(2)
        c1.metric("Routines", summary.get("routines", 0))
        c2.metric("Xrefs", summary.get("xrefs", 0))
        c3, c4 = st.columns(2)
        c3.metric("Instructions", summary.get("instructions", 0))
        c4.metric("SMC sites", summary.get("smc_sites", 0))
        st.caption(
            f"asm_docs: {summary.get('asm_docs', 0)} · "
            f"labels: {summary.get('labels', 0)} · "
            f"events: {summary.get('events_total', 0)}"
        )


# --------------------------------------------------------------------------- #
# Main column — chat
# --------------------------------------------------------------------------- #


st.title("C64-RE Agent")
st.caption(
    "Multi-turn deep research over a Commodore 64 binary. "
    "Each question runs the full plan → tool-loop → critic pipeline; "
    "addresses cited in the answer light up the **Code Lens** below."
)


# --------------------------------------------------------------------------- #
# Agent message renderer — called during the streaming progress pane.
# Without this, Streamlit's markdown renderer treats $XXXX hex addresses as
# LaTeX math delimiters, producing character-by-character garbling.
# --------------------------------------------------------------------------- #

import re as _re

# Backtick-wrap every bare hex address so it renders as inline code.
# Segments already inside a backtick code span are left untouched to
# avoid double-wrapping (which breaks markdown rendering).
_ADDR_ESC_RE = _re.compile(r'\$([0-9A-Fa-f]{1,4})\b')
_CODE_SPAN_RE = _re.compile(r'(`+[^`]*?`+)')

def _esc_addrs(text: str) -> str:
    parts = _CODE_SPAN_RE.split(text)
    return "".join(
        part if part.startswith("`") else _ADDR_ESC_RE.sub(r'`$\1`', part)
        for part in parts
    )

_TOOL_LINE_RE   = _re.compile(r'^\[tool:(?P<tool>\w+)\]\s*(?P<ok>[✓✗])\s*step=(?P<step>\S+)\s*[—–-]\s*(?P<rest>.*)', _re.DOTALL)
_EXEC_RE        = _re.compile(r'^Executor selected step\s+(?P<step>\S+)\s+→\s+(?P<tool>\S+)\.\s+Rationale:\s*(?P<rat>.*)', _re.DOTALL)
_SYNTH_RE       = _re.compile(r'^Synthesizer:\s*(?P<counts>[^·]+)·\s*KB events=(?P<events>\d+)\.\s*(?P<note>.*)', _re.DOTALL)
_ANALYST_RE     = _re.compile(r'^Analyst confidence=(?P<conf>[0-9.]+)\.')
_CRITIC_RE      = _re.compile(r'^Critic verdict:\s*(?P<verdict>\w+)\.')
_PLANNER_RE     = _re.compile(r'^Planner produced (?P<n>\d+) steps\.')
_CURATOR_RE     = _re.compile(r'^Curator')
_LOAD_RE        = _re.compile(r'^(KB ready|Ingested|Text dir|KB stats|code_kb|thread_id)')
_WROTE_REPORT_RE= _re.compile(r'^Wrote report')


def _render_agent_msg(msg: str) -> None:
    """Render a single streamed agent message with appropriate formatting."""
    msg = (msg or "").strip()
    if not msg:
        return

    # Tool result: [tool:X] ✓/✗ step=Y — ...
    m = _TOOL_LINE_RE.match(msg)
    if m:
        ok_icon = "✅" if m.group("ok") == "✓" else "❌"
        tool = m.group("tool")
        step = m.group("step")
        rest = _esc_addrs(m.group("rest").strip())
        first_line = rest.split("\n")[0][:300]
        st.markdown(f"{ok_icon} **`{tool}`** `{step}` — {first_line}")
        return

    # Executor step selection
    m = _EXEC_RE.match(msg)
    if m:
        st.markdown(
            f"⚙️ **Step** `{m.group('step')}` → **{m.group('tool')}**"
            f" · _{_esc_addrs(m.group('rat').strip())}_"
        )
        return

    # Synthesizer summary
    m = _SYNTH_RE.match(msg)
    if m:
        counts = m.group("counts").strip().rstrip(",")
        events = m.group("events")
        note = _esc_addrs(m.group("note").strip())
        detail = f" · {note}" if note else ""
        st.markdown(f"🧩 **Synthesizer** {counts} · KB={events}{detail}")
        return

    # Analyst confidence
    m = _ANALYST_RE.match(msg)
    if m:
        conf = float(m.group("conf"))
        bar = "🟢" if conf >= 0.7 else "🟡" if conf >= 0.4 else "🔴"
        st.markdown(f"{bar} **Analyst** confidence = {conf:.0%}")
        return

    # Critic verdict
    m = _CRITIC_RE.match(msg)
    if m:
        verdict = m.group("verdict")
        icons = {"accept": "✅", "replan": "🔁", "revise": "🔄", "budget_exceeded": "⛔"}
        st.markdown(f"{icons.get(verdict, '🧐')} **Critic verdict:** `{verdict}`")
        return

    # Planner step count
    m = _PLANNER_RE.match(msg)
    if m:
        st.markdown(f"📋 **Planner** produced **{m.group('n')}** steps")
        return

    # Curator
    if _CURATOR_RE.match(msg):
        st.markdown(f"🗜️ _{_esc_addrs(msg)}_")
        return

    # Wrote report
    if _WROTE_REPORT_RE.match(msg):
        st.markdown(f"📄 {_esc_addrs(msg)}")
        return

    # Load-inputs / KB status lines
    if _LOAD_RE.match(msg):
        st.caption(_esc_addrs(msg))
        return

    # Fallback — escape addresses and render as markdown
    st.markdown(_esc_addrs(msg))


def _render_turn(turn: dict) -> None:
    """Render one chat turn (question + answer + meta) in the chat column."""
    with st.chat_message("user"):
        st.markdown(_esc_addrs(turn["question"]))
    with st.chat_message("assistant"):
        verdict = turn.get("verdict", "n/a")
        conf = turn.get("confidence")
        meta_bits = [f"verdict: **{verdict}**"]
        if conf is not None:
            meta_bits.append(f"confidence: **{conf:.2f}**")
        if turn.get("elapsed_s"):
            meta_bits.append(f"⏱ {turn['elapsed_s']:.1f}s")
        st.caption(" · ".join(meta_bits))
        st.markdown(_esc_addrs(turn["answer"]))
        if turn.get("evidence"):
            with st.expander(f"Evidence ({len(turn['evidence'])})", expanded=False):
                for e in turn["evidence"]:
                    st.markdown(f"- `{e}`")
        if turn.get("open_questions"):
            with st.expander(f"Open questions ({len(turn['open_questions'])})"):
                for q in turn["open_questions"]:
                    st.markdown(f"- {_esc_addrs(q)}")
        if turn.get("critique"):
            with st.expander("Critic notes", expanded=False):
                st.markdown(_esc_addrs(turn["critique"]))
        if turn.get("addresses"):
            addr_chips = " ".join(
                f"`${a:04X}`" for a in turn["addresses"][:24]
            )
            st.caption(f"Cited addresses: {addr_chips}")


for turn in st.session_state.history:
    _render_turn(turn)


# --------------------------------------------------------------------------- #
# Chat input — submitting a question kicks the agent
# --------------------------------------------------------------------------- #


def _run_turn(question: str) -> TurnResult | None:
    """Execute one turn of the agent and stream progress into the UI."""
    if not st.session_state.game:
        st.error("Set a game name in the sidebar first.")
        return None
    if not st.session_state.dump_path:
        st.error("Set the dump path in the sidebar first.")
        return None
    if not Path(st.session_state.dump_path).exists():
        st.error(f"Dump file not found: {st.session_state.dump_path}")
        return None

    progress_box = st.status(
        "Running deep research…", expanded=True, state="running",
    )
    final: TurnResult | None = None
    error_msg: str | None = None
    with progress_box:
        for kind, payload in run_question(
            game=st.session_state.game,
            question=question,
            dump_path=st.session_state.dump_path,
            asm_dir=st.session_state.asm_dir or None,
            asm_files=(
                st.session_state.asm_files_selected or None
            ),
            partial_asm=st.session_state.partial_asm or None,
            text_dir=st.session_state.text_dir or None,
        ):
            if kind == "status":
                st.caption(payload)
            elif kind == "step":
                msgs = payload.get("messages") or []
                for m in msgs:
                    _render_agent_msg(m)
            elif kind == "error":
                error_msg = str(payload)
                st.error(error_msg)
            elif kind == "done":
                final = payload  # type: ignore[assignment]
    if error_msg:
        progress_box.update(label=f"Failed: {error_msg}", state="error")
        return None
    if final is None:
        progress_box.update(label="No final state produced.", state="error")
        return None
    progress_box.update(
        label=f"Done · verdict={final.verdict} · {final.elapsed_s:.1f}s",
        state="complete", expanded=False,
    )
    return final


question = st.chat_input("Ask the agent something specific about the game…")
if question:
    st.session_state.pending_question = question
    st.session_state.running = True
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        result = _run_turn(question)
    st.session_state.running = False
    if result is not None:
        # Auto-focus the first cited address so the call graph tab is
        # already aimed somewhere useful when the user opens it.
        if result.addresses and st.session_state.focus_addr is None:
            st.session_state.focus_addr = result.addresses[0]
        st.session_state.history.append({
            "question": result.question,
            "answer": result.answer,
            "confidence": result.confidence,
            "verdict": result.verdict,
            "critique": result.critique,
            "evidence": result.evidence,
            "open_questions": result.open_questions,
            "addresses": result.addresses,
            "elapsed_s": result.elapsed_s,
            "thread_id": result.thread_id,
        })
        st.rerun()


# --------------------------------------------------------------------------- #
# Code Lens — auto-populated from the latest turn
# --------------------------------------------------------------------------- #


st.markdown("---")
st.subheader("Code Lens")

latest_turn = st.session_state.history[-1] if st.session_state.history else None
latest_addrs: list[int] = (
    latest_turn["addresses"] if latest_turn else []
)

# Let the user override / add addresses by hand. Useful when the
# answer prose phrases an address indirectly (e.g. "the JSR table at
# $C100") and the user wants to drill into a specific value.
manual_addrs_str = st.text_input(
    "Addresses to inspect (comma-separated, $XXXX or 0xXXXX)",
    value=", ".join(f"${a:04X}" for a in latest_addrs),
    help=(
        "Defaults to addresses cited in the most recent answer. "
        "Edit freely — anything you type is treated as the working set."
    ),
)
working_addrs = parse_addresses(manual_addrs_str) or latest_addrs

tab_disasm, tab_graph, tab_routines, tab_kb = st.tabs([
    "Disassembly", "Call graph", "Routines", "Raw KB",
])


# ----- Disassembly tab ----------------------------------------------------- #


with tab_disasm:
    if not st.session_state.game:
        st.info("Set a game name and run a question first.")
    elif not working_addrs:
        st.info(
            "No addresses on the radar yet. Ask a code question (or paste "
            "addresses above) and the relevant routines will appear here."
        )
    else:
        snippets = disasm_for_addresses(
            st.session_state.game, working_addrs, max_routines=8,
        )
        if not snippets:
            st.warning(
                "No routines found in the code-KB enclosing those "
                "addresses. The agent may need to disassemble that area "
                "first — try a follow-up question like "
                f"`disassemble ${working_addrs[0]:04X} for 256 bytes`."
            )
        for snip in snippets:
            fallback_name = f"sub_{snip['start_addr']:04x}"
            display_name = snip.get("name") or fallback_name
            head = (
                f"**{display_name}** "
                f"`${snip['start_addr']:04X}-${snip['end_addr']:04X}` "
                f"· {snip['instruction_count']} insns"
            )
            if snip.get("source_file"):
                head += f" · _{Path(snip['source_file']).name}_"
            st.markdown(head)
            if snip.get("layer1"):
                l1 = snip["layer1"]
                conf = l1.get("confidence")
                conf_str = f" (conf={conf:.2f})" if isinstance(conf, (int, float)) else ""
                st.markdown(
                    f"> _Layer-1 hypothesis{conf_str}_: "
                    f"{_esc_addrs((l1.get('text') or '').strip()[:600])}"
                )
            st.code(snip["listing"], language="text")
            cols = st.columns([1, 1, 4])
            if cols[0].button(
                "Center call graph here",
                key=f"focus_{snip['start_addr']:04X}",
            ):
                st.session_state.focus_addr = snip["start_addr"]
                st.rerun()


# ----- Call graph tab ------------------------------------------------------ #


with tab_graph:
    if not st.session_state.game:
        st.info("Set a game name in the sidebar.")
    else:
        center_default = (
            st.session_state.focus_addr
            if st.session_state.focus_addr is not None
            else (working_addrs[0] if working_addrs else None)
        )
        c1, c2, c3 = st.columns([2, 1, 1])
        center_str = c1.text_input(
            "Centre on routine (start address)",
            value=f"${center_default:04X}" if center_default is not None else "",
            help="Any $XXXX address inside the routine works.",
        )
        hops = c2.slider("Hops", 1, 3, value=int(st.session_state.focus_hops))
        st.session_state.focus_hops = hops
        max_nodes = c3.slider("Max nodes", 10, 200, value=60)

        center_list = parse_addresses(center_str)
        if not center_list:
            st.info("Type or pick an address to render a call graph.")
        else:
            res = call_graph_dot(
                st.session_state.game,
                start=center_list[0], hops=hops, max_nodes=max_nodes,
            )
            if res is None:
                st.warning("No code-KB on disk for this game yet.")
            else:
                st.graphviz_chart(res["dot"], width='stretch')
                ctr = res.get("center") or {}
                if ctr:
                    centre_name = ctr.get("name") or f"sub_{ctr['start']:04x}"
                    st.caption(
                        f"Centre: **{centre_name}** "
                        f"`${ctr['start']:04X}-${ctr['end']:04X}` · "
                        f"{len(res['nodes'])} nodes / {len(res['edges'])} edges"
                    )
                with st.expander("Edges (table)"):
                    if res["edges"]:
                        st.dataframe(
                            [
                                {
                                    "src": f"${e['src']:04X}",
                                    "dst": f"${e['dst']:04X}" if e.get("dst") else "—",
                                    "kind": e["kind"],
                                    "dst_idiom": e.get("dst_idiom") or "",
                                    "dst_conf": (
                                        f"{e['dst_confidence']:.2f}"
                                        if isinstance(e.get("dst_confidence"), (int, float))
                                        else ""
                                    ),
                                    "via": (
                                        f"${e['via_vector']:04X}"
                                        if e.get("via_vector")
                                        else ""
                                    ),
                                }
                                for e in res["edges"]
                            ],
                            hide_index=True, width='stretch',
                        )
                    else:
                        st.caption("(no outgoing or incoming xrefs in scope)")
                with st.expander("Nodes (semantic annotations)"):
                    nodes = res.get("nodes") or []
                    if nodes:
                        st.dataframe(
                            [
                                {
                                    "start": f"${int(n['start']):04X}",
                                    "end": f"${int(n['end']):04X}",
                                    "name": n.get("name") or f"sub_{int(n['start']):04x}",
                                    "idiom": (
                                        ((n.get("layer1") or {}).get("idiom_match"))
                                        or ""
                                    ),
                                    "conf": (
                                        f"{((n.get('layer1') or {}).get('confidence')):.2f}"
                                        if isinstance((n.get("layer1") or {}).get("confidence"), (int, float))
                                        else ""
                                    ),
                                    "hypothesis": (
                                        ((n.get("layer1") or {}).get("text") or "")[:180]
                                    ),
                                }
                                for n in nodes
                            ],
                            hide_index=True,
                            width='stretch',
                        )
                    else:
                        st.caption("(no nodes in graph)")
                with st.expander("DOT source"):
                    st.code(res["dot"], language="dot")


# ----- Routines tab -------------------------------------------------------- #


with tab_routines:
    if not st.session_state.game:
        st.info("Set a game name in the sidebar.")
    else:
        rows = top_routines_by_xrefs(st.session_state.game, limit=25)
        if not rows:
            st.info("No routines yet — run a question to populate the KB.")
        else:
            display_rows = [
                {
                    "start": f"${int(r['start_addr']):04X}",
                    "end": f"${int(r['end_addr']):04X}",
                    "name": r.get("name") or f"sub_{int(r['start_addr']):04x}",
                    "callers": int(r.get("callers") or 0),
                    "analyzed": "yes" if int(r.get("has_layer1") or 0) else "no",
                    "idiom": r.get("idiom_match") or "",
                    "conf": (
                        f"{float(r['layer1_confidence']):.2f}"
                        if isinstance(r.get("layer1_confidence"), (int, float))
                        else ""
                    ),
                    "hypothesis": (r.get("hypothesis_text") or "")[:140],
                }
                for r in rows
            ]
            analyzed_count = sum(1 for rr in display_rows if rr["analyzed"] == "yes")
            st.caption(
                f"Layer-1 coverage in this table: {analyzed_count}/{len(display_rows)} routines. "
                "Rows are prioritized to show analyzed routines first."
            )
            st.dataframe(
                display_rows, hide_index=True, width='stretch',
            )
            choice = st.selectbox(
                "Centre call graph on routine",
                options=[f"{r['start']} — {r['name']}" for r in display_rows],
                index=0,
            )
            c_btn1, c_btn2 = st.columns([1, 1])
            if c_btn1.button("Show call graph for selection"):
                addrs = parse_addresses(choice)
                if addrs:
                    st.session_state.focus_addr = addrs[0]
                    st.rerun()
            if c_btn2.button("Annotate selected routine"):
                addrs = parse_addresses(choice)
                if not addrs:
                    st.error("Could not parse routine address from selection.")
                else:
                    with st.spinner(
                        f"Running Layer-1 annotate for ${addrs[0]:04X}..."
                    ):
                        ann = annotate_routine(
                            st.session_state.game,
                            start=addrs[0],
                        )
                    if ann.get("ok"):
                        conf = ann.get("confidence")
                        conf_txt = (
                            f"{float(conf):.2f}"
                            if isinstance(conf, (int, float))
                            else "n/a"
                        )
                        st.success(
                            "Annotated routine "
                            f"${addrs[0]:04X} "
                            f"(idiom={ann.get('idiom_match') or 'n/a'}, conf={conf_txt})."
                        )
                        st.rerun()
                    else:
                        st.error(str(ann.get("data") or ann.get("error") or "annotate failed"))

            st.divider()
            c_bulk_n, c_bulk_only, c_bulk_run = st.columns([1, 1, 2])
            bulk_n = c_bulk_n.number_input(
                "Top N",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="How many routines to annotate in one batch.",
                key="bulk_annotate_n",
            )
            only_unanalyzed = c_bulk_only.checkbox(
                "Only unanalyzed",
                value=True,
                help="When enabled, skips rows that already have Layer-1 data.",
                key="bulk_annotate_only_unanalyzed",
            )
            if c_bulk_run.button("Annotate top routines"):
                candidates = [
                    r for r in display_rows
                    if (not only_unanalyzed or r.get("analyzed") != "yes")
                ]
                targets = candidates[: int(bulk_n)]
                if not targets:
                    st.info("No routines match the current bulk-annotation filter.")
                else:
                    ok = 0
                    failed = 0
                    failed_items: list[str] = []
                    with st.spinner(
                        f"Annotating {len(targets)} routine(s)..."
                    ):
                        for row in targets:
                            addrs = parse_addresses(row.get("start") or "")
                            if not addrs:
                                failed += 1
                                failed_items.append(f"{row.get('start')}: bad start")
                                continue
                            ann = annotate_routine(
                                st.session_state.game,
                                start=addrs[0],
                            )
                            if ann.get("ok"):
                                ok += 1
                            else:
                                failed += 1
                                failed_items.append(
                                    f"${addrs[0]:04X}: "
                                    f"{ann.get('data') or ann.get('error') or 'failed'}"
                                )
                    if failed_items:
                        with st.expander(f"Bulk annotate errors ({len(failed_items)})"):
                            for msg in failed_items[:20]:
                                st.markdown(f"- {msg}")
                    if ok:
                        st.success(f"Bulk annotation complete: success={ok}, failed={failed}.")
                        st.rerun()
                    else:
                        st.error(f"Bulk annotation failed for all {failed} routine(s).")


# ----- Raw KB tab ---------------------------------------------------------- #


with tab_kb:
    if not st.session_state.game:
        st.info("Set a game name in the sidebar.")
    else:
        summary = code_kb_summary(st.session_state.game)
        if summary is None:
            st.info("No code KB on disk yet.")
        else:
            st.json(summary, expanded=False)
        slug = st.session_state.game.strip().lower().replace(" ", "_")
        report_path = Path("sessions") / slug / "report.md"
        if report_path.exists():
            st.markdown(f"**Last full report:** `{report_path}`")
            with st.expander("Show last report.md"):
                st.markdown(report_path.read_text(errors="replace"))
        else:
            st.caption("_no `report.md` produced yet_")


# Footer
st.markdown(
    "<div style='opacity:0.5;font-size:0.85em;margin-top:2rem'>"
    "Tip: ask follow-up questions in the same chat — both the parent KB "
    "and the code KB persist across turns under "
    "<code>sessions/&lt;game&gt;/</code>."
    "</div>",
    unsafe_allow_html=True,
)
