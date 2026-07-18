"""Secret redaction stays at context/preview boundaries (tracker 7.3)."""

from __future__ import annotations

import json

from graph.nodes import kb_query_node
from memory import get_store
from memory.redaction import REDACTED, redact_text, redact_value


FAKE_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"


def test_redactor_covers_common_shapes_without_harming_c64_text():
    source = (
        f"OPENAI_API_KEY={FAKE_KEY}\n"
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz.1234567890\n"
        "database=https://alice:hunter2@example.invalid/db\n"
        "routine $C000 writes $D020; tokenized BASIC SYS 49152"
    )
    redacted = redact_text(source)

    assert FAKE_KEY not in redacted
    assert "hunter2" not in redacted
    assert redacted.count(REDACTED) >= 3
    assert "$C000 writes $D020" in redacted
    assert "tokenized BASIC SYS 49152" in redacted

    nested = redact_value({
        "api_key": "short-secret",
        "nested": [{"password": "pw"}, {"address": "$C000"}],
    })
    assert nested == {
        "api_key": REDACTED,
        "nested": [{"password": REDACTED}, {"address": "$C000"}],
    }


def test_secret_named_c64_addresses_survive_and_redaction_is_idempotent():
    source = (
        "lives at $0780; secret=$0780\n"
        f"OPENAI_API_KEY={FAKE_KEY}\n"
        "OPENAI_API_KEY=[REDACTED]"
    )

    redacted = redact_text(source)

    assert "lives at $0780" in redacted
    assert "secret=$0780" in redacted
    assert FAKE_KEY not in redacted
    assert "OPENAI_API_KEY=[REDACTED]" in redacted
    assert "[REDACTED]]" not in redacted
    assert redact_text(redacted) == redacted
    assert redact_value({"secret": "$0780"}) == {"secret": "$0780"}


def test_real_prefixed_bearer_and_jwt_values_still_redact():
    jwt = "abcdefgh.ijklmnop.qrstuvwx"
    source = (
        "xai-abcdefghijklmnopqrstuvwxyz123456 "
        "Bearer abcdefghijklmnopqrstuvwxyz123456 "
        f"{jwt}"
    )

    redacted = redact_text(source)

    assert "xai-" not in redacted
    assert "abcdefghijklmnopqrstuvwxyz123456" not in redacted
    assert jwt not in redacted
    assert redacted.count(REDACTED) == 3


def test_note_source_is_preserved_but_search_and_digest_are_redacted(tmp_path):
    note = tmp_path / "trusted_notes.md"
    note.write_text(
        "Lives counter research.\n"
        f"OPENAI_API_KEY={FAKE_KEY}\n"
        "The lives counter may be at $0780.\n",
    )
    store = get_store(tmp_path / "kb")
    assert store.ingest_text_file(note)

    raw = store.query(
        "SELECT content FROM text_docs WHERE path = ?", (str(note),),
    )[0]["content"]
    assert FAKE_KEY in raw

    hits = store.search_text("lives", limit=2, snippet_chars=500)
    assert hits and FAKE_KEY not in hits[0]["snippet"]
    assert REDACTED in hits[0]["snippet"]

    digest = store.digest_for_question("Where is the lives counter?")
    assert FAKE_KEY not in digest
    assert REDACTED in digest


def test_parent_kb_sql_redacts_rows_and_sql_metadata(tmp_path):
    store = get_store(tmp_path / "kb")
    note = tmp_path / "notes.md"
    note.write_text(f"secret research value {FAKE_KEY}\n")
    store.ingest_text_file(note)

    state = {
        "kb_handle": str(tmp_path / "kb"),
        "current_step_id": "sql1",
        "plan": [{
            "id": "sql1",
            "tool": "kb",
            "args": {
                "mode": "sql",
                "sql": (
                    "SELECT content, 'api_key=" + FAKE_KEY
                    + "' AS query_note FROM text_docs"
                ),
            },
        }],
    }
    result = kb_query_node(state)["tool_results"][0]
    rendered = json.dumps(result, sort_keys=True)

    assert result["ok"] is True
    assert FAKE_KEY not in rendered
    assert REDACTED in rendered
