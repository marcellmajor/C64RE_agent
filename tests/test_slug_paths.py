"""Slug centralization across entry points (tracker 0.7 review).

Covers the persistence-purge script, code-KB scoping, and the UI
helpers in tools/agent_runner — all of which previously carried their
own slug implementations — plus legacy hyphen-directory resolution.
(The CLI thread-id slug is asserted in test_recursion_recovery.)
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# scripts/purge_persistence.py
# ---------------------------------------------------------------------------

@pytest.fixture()
def purge_mod():
    spec = importlib.util.spec_from_file_location(
        "purge_persistence", ROOT / "scripts" / "purge_persistence.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_purge_targets_canonical_session_dir(purge_mod, monkeypatch, tmp_path):
    sessions = tmp_path / "sessions"
    (sessions / "bubble_bobble").mkdir(parents=True)
    monkeypatch.setattr(purge_mod, "SESSIONS_DIR", sessions)

    targets = purge_mod._collect_targets("Bubble Bobble", False, False, False)
    assert targets == [sessions / "bubble_bobble"]


def test_purge_finds_legacy_hyphen_session_dir(purge_mod, monkeypatch, tmp_path):
    sessions = tmp_path / "sessions"
    (sessions / "bubble-bobble").mkdir(parents=True)
    monkeypatch.setattr(purge_mod, "SESSIONS_DIR", sessions)

    targets = purge_mod._collect_targets("Bubble Bobble", False, False, False)
    assert targets == [sessions / "bubble-bobble"]


def test_purge_empty_game_never_targets_sessions_root(purge_mod, monkeypatch,
                                                      tmp_path):
    """The old local slug mapped '' to '' — `sessions/ / ''` IS the
    sessions root, so a stray empty --game could have purged everything."""
    monkeypatch.setattr(purge_mod, "SESSIONS_DIR", tmp_path / "sessions")
    assert purge_mod._slug("") == "unknown"


# ---------------------------------------------------------------------------
# code_kb/scoping.py
# ---------------------------------------------------------------------------

def test_scoping_slug_delegates_to_canonical():
    from code_kb.scoping import _slug

    assert _slug("Wizard of Wor") == "wizard_of_wor"
    assert _slug("") == "unknown"


# ---------------------------------------------------------------------------
# tools/agent_runner.py (UI helpers)
# ---------------------------------------------------------------------------

def test_resolve_code_store_finds_canonical_dir(monkeypatch, tmp_path):
    from tools.agent_runner import _resolve_code_store

    (tmp_path / "sessions" / "petch" / "code_kb").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert _resolve_code_store("Petch") is not None


def test_resolve_code_store_finds_legacy_hyphen_dir(monkeypatch, tmp_path):
    from tools.agent_runner import _resolve_code_store

    (tmp_path / "sessions" / "wizard-of-wor" / "code_kb").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert _resolve_code_store("Wizard of Wor") is not None


def test_resolve_code_store_missing_returns_none(monkeypatch, tmp_path):
    from tools.agent_runner import _resolve_code_store

    (tmp_path / "sessions").mkdir()
    monkeypatch.chdir(tmp_path)
    assert _resolve_code_store("No Such Game") is None


# ---------------------------------------------------------------------------
# main.py --reset-code-kb
# ---------------------------------------------------------------------------

def test_cli_reset_code_kb_finds_legacy_hyphen_dir(
    monkeypatch, tmp_path, capsys,
):
    import main as main_mod

    sessions = tmp_path / "sessions"
    legacy_code_kb = sessions / "wizard-of-wor" / "code_kb"
    legacy_code_kb.mkdir(parents=True)
    (legacy_code_kb / "sentinel.txt").write_text("stale")

    final_state = {
        "candidate_answer": {
            "answer": "done", "confidence": 0.0,
            "evidence": [], "open_questions": [],
        },
        "verdict": {"decision": "accept"},
        "messages": [],
    }

    class _FinishedGraph:
        def compile(self, checkpointer=None):
            return self

        def stream(self, initial_state, config, stream_mode="values"):
            yield final_state

        def get_state(self, config):
            return SimpleNamespace(values=final_state)

    monkeypatch.setattr(main_mod, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(main_mod, "build_graph", lambda: _FinishedGraph())
    monkeypatch.setattr(main_mod, "_configure_langsmith_tracing", lambda: None)
    monkeypatch.setattr("sys.argv", [
        "main.py",
        "--game", "Wizard of Wor",
        "--question", "q",
        "--dump", str(tmp_path / "missing.dump"),
        "--reset-code-kb",
        "--no-trace",
    ])

    main_mod.main()

    assert not legacy_code_kb.exists()
    assert "wizard-of-wor/code_kb" in capsys.readouterr().out
