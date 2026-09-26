"""Browse and annotate routines outside the former first-25 UI window."""

from pathlib import Path

import pytest

from code_kb import Annotation, get_code_store
from code_kb.schema import ANN_ROUTINE
from tools import agent_runner as runner


def test_routine_query_can_return_the_entire_catalog(tmp_path, monkeypatch):
    store = get_code_store(tmp_path / "code_kb")
    for index in range(60):
        start = 0x8000 + index * 16
        store.append_annotation(Annotation(
            layer=0, kind=ANN_ROUTINE, start_addr=start, end_addr=start + 15,
            producer="test", confidence=1.0,
            payload={"name": f"routine_{index:03d}", "source_file": "test.asm"},
        ), source="test")
    monkeypatch.setattr(runner, "_resolve_code_store", lambda game: store)
    assert len(runner.top_routines_by_xrefs("Test")) == 12
    assert len(runner.top_routines_by_xrefs("Test", limit=25)) == 25
    all_rows = runner.top_routines_by_xrefs("Test", limit=None)
    assert len(all_rows) == 60
    assert all_rows[-1]["name"] == "routine_059"
    assert [r["start_addr"] for r in all_rows] == sorted(r["start_addr"] for r in all_rows)


@pytest.fixture
def browser(tmp_path, monkeypatch):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    monkeypatch.setenv("C64RE_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr(runner, "_resolve_code_store", lambda game: None)
    rows = [{
        "start_addr": 0x8000 + i * 16, "end_addr": 0x800F + i * 16,
        "name": f"routine_{i:03d}", "has_layer1": int(i < 50), "callers": 0,
        "hypothesis_text": "X" * 180 + "needle-in-long-hypothesis" if i == 37 else "",
    } for i in range(60)]
    def routines(game, *, limit=12):
        assert limit is None  # The UI must never pre-limit its searchable catalog.
        return rows
    monkeypatch.setattr(runner, "top_routines_by_xrefs", routines)
    calls = []
    def annotate(game, *, start):
        calls.append(start)
        next(row for row in rows if row["start_addr"] == start)["has_layer1"] = 1
        return {"ok": True, "confidence": 0.8}
    monkeypatch.setattr(runner, "annotate_routine", annotate)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["game"] = "Browser Test"
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app, calls


def routine_table(app):
    return next(frame.value for frame in app.dataframe if "start" in frame.value.columns)


def test_all_pages_are_reachable_and_selection_can_annotate_the_last_routine(browser):
    app, calls = browser
    assert len(routine_table(app)) == 25
    app.number_input(key="routine_page").set_value(3).run()
    assert not app.exception
    assert routine_table(app)["name"].tolist() == [f"routine_{i:03d}" for i in range(50, 60)]
    chooser = next(s for s in app.selectbox if s.label == "Centre call graph on routine")
    chooser.select("$83B0 — routine_059").run()
    next(b for b in app.button if b.label == "Annotate selected routine").click().run()
    assert not app.exception
    assert calls == [0x83B0]


@pytest.mark.parametrize("query,expected", [
    ("routine_059", "routine_059"),
    ("$83B0", "routine_059"),
    ("needle-in-long-hypothesis", "routine_037"),
])
def test_search_covers_off_page_rows_and_full_hypothesis_text(browser, query, expected):
    app, _ = browser
    app.number_input(key="routine_page").set_value(3).run()
    app.text_input(key="routine_search").set_value(query).run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.number_input(key="routine_page").value == 1
    assert routine_table(app)["name"].tolist() == [expected]


def test_status_and_page_size_changes_reset_pagination(browser):
    app, _ = browser
    app.number_input(key="routine_page").set_value(3).run()
    app.selectbox(key="routine_page_size").select(50).run()
    assert app.number_input(key="routine_page").value == 1
    assert len(routine_table(app)) == 50
    app.selectbox(key="routine_filter").select("Unanalyzed").run()
    assert not app.exception
    assert len(routine_table(app)) == 10
    assert set(routine_table(app)["analyzed"]) == {"no"}


def test_bulk_annotation_finds_unanalyzed_routines_beyond_the_visible_page(browser):
    app, calls = browser
    assert set(routine_table(app)["analyzed"]) == {"yes"}
    assert any("10 eligible routines across all pages" in c.value for c in app.caption)
    app.button(key="bulk_annotate_run").click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert calls == [0x8000 + i * 16 for i in range(50, 55)]


def test_bulk_annotation_respects_search_and_empty_matches_disable_actions(browser):
    app, calls = browser
    app.text_input(key="routine_search").set_value("routine_059").run()
    app.button(key="bulk_annotate_run").click().run()
    assert not app.exception
    assert calls == [0x83B0]
    app.text_input(key="routine_search").set_value("no-such-routine").run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("No routines match these filters" in i.value for i in app.info)
    assert app.button(key="bulk_annotate_run").disabled
    assert next(b for b in app.button if b.label == "Annotate selected routine").disabled
