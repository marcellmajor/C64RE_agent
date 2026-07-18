"""Durable CLI resume (tracker 2.4): SqliteSaver plumbing + thread ids."""

import main as main_mod


def test_default_thread_id_is_per_question_and_stable():
    a1 = main_mod._default_thread_id("Wizard of Wor", "where is the score?")
    a2 = main_mod._default_thread_id("Wizard of Wor", "where is the score?")
    b = main_mod._default_thread_id("Wizard of Wor", "how does the maze work?")

    assert a1 == a2                              # stable → rerun resumes
    assert a1.startswith("wizard_of_wor-")       # canonical slug prefix
    assert a1 != b                               # new question → new thread


def test_open_checkpointer_creates_durable_file(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "SESSIONS_DIR", tmp_path)

    with main_mod._open_checkpointer("My Game") as saver:
        assert saver is not None
        ckpt = tmp_path / "my_game" / "checkpoint.sqlite"
        assert ckpt.exists()
        # The saver is a real checkpointer: a compiled graph accepts it.
        from graph.build import build_graph
        graph = build_graph().compile(checkpointer=saver)
        assert graph is not None


def test_open_checkpointer_reuses_legacy_session_dir(monkeypatch, tmp_path):
    (tmp_path / "my-game").mkdir()               # legacy hyphen dir exists
    monkeypatch.setattr(main_mod, "SESSIONS_DIR", tmp_path)

    with main_mod._open_checkpointer("My Game"):
        assert (tmp_path / "my-game" / "checkpoint.sqlite").exists()
        assert not (tmp_path / "my_game").exists()
