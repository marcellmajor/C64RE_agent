"""Local human ratings and explicit LangSmith export (tracker 5.5)."""

from __future__ import annotations

from types import SimpleNamespace

from langsmith.utils import LangSmithNotFoundError

from tools.research_notebook import (
    append_turn,
    latest_turn_ratings,
    load_turn_ratings,
    rated_turn_examples,
    record_turn_rating,
    sync_ratings_to_langsmith,
)


def _archive_turn(root, run_id="run-1"):
    assert append_turn(root, {
        "run_id": run_id,
        "game": "Test Game",
        "question": "Where is the counter?",
        "answer": "At $00C0.",
        "verdict": "accept",
        "confidence": 0.9,
        "evidence": ["evt_1"],
        "open_questions": [],
        "addresses": [0x00C0],
    })


def test_ratings_are_validated_idempotent_and_revisioned(tmp_path):
    _archive_turn(tmp_path)
    first, changed = record_turn_rating(
        tmp_path, run_id="run-1", rating="helpful", comment="grounded",
    )
    duplicate, duplicate_changed = record_turn_rating(
        tmp_path, run_id="run-1", rating="helpful", comment="grounded",
    )
    revised, revised_changed = record_turn_rating(
        tmp_path, run_id="run-1", rating="not_helpful", comment="wrong byte",
    )

    assert changed is True
    assert duplicate_changed is False
    assert duplicate == first
    assert revised_changed is True
    assert len(load_turn_ratings(tmp_path)) == 2
    assert latest_turn_ratings(tmp_path)["run-1"] == revised
    examples = rated_turn_examples(tmp_path)
    assert len(examples) == 1
    assert examples[0]["human_rating"]["rating"] == "not_helpful"


def test_rating_rejects_unknown_value_and_missing_run_id(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="helpful"):
        record_turn_rating(tmp_path, run_id="run", rating="maybe")
    with pytest.raises(ValueError, match="run_id"):
        record_turn_rating(tmp_path, run_id="", rating="helpful")


class _FakeLangSmithClient:
    def __init__(self):
        self.dataset = None
        self.examples = {}
        self.created = []
        self.updated = []

    def read_dataset(self, *, dataset_name):
        if self.dataset is None:
            raise LangSmithNotFoundError("missing dataset")
        return self.dataset

    def create_dataset(self, dataset_name, **kwargs):
        self.dataset = SimpleNamespace(id="dataset-id", name=dataset_name)
        return self.dataset

    def read_example(self, example_id):
        if example_id not in self.examples:
            raise LangSmithNotFoundError("missing example")
        return self.examples[example_id]

    def create_example(self, **kwargs):
        self.examples[kwargs["example_id"]] = dict(kwargs)
        self.created.append(dict(kwargs))

    def update_example(self, example_id, **kwargs):
        self.examples[example_id] = {"example_id": example_id, **kwargs}
        self.updated.append({"example_id": example_id, **kwargs})


def test_langsmith_sync_is_explicit_and_idempotently_updates(tmp_path):
    _archive_turn(tmp_path)
    record_turn_rating(
        tmp_path, run_id="run-1", rating="helpful", comment="useful",
    )
    client = _FakeLangSmithClient()

    first = sync_ratings_to_langsmith(
        tmp_path, dataset_name="c64re-human", client=client,
    )
    second = sync_ratings_to_langsmith(
        tmp_path, dataset_name="c64re-human", client=client,
    )

    assert first == {
        "dataset": "c64re-human", "rated_turns": 1,
        "created": 1, "updated": 0,
    }
    assert second == {
        "dataset": "c64re-human", "rated_turns": 1,
        "created": 0, "updated": 1,
    }
    created = client.created[0]
    assert created["dataset_id"] == "dataset-id"
    assert created["inputs"] == {
        "game": "Test Game", "question": "Where is the counter?",
    }
    assert created["metadata"]["rating_score"] == 1
    assert created["metadata"]["addresses"] == [0x00C0]
