"""SQLite schema + event-kind constants for the KB."""

from __future__ import annotations

# Event kinds (the `kind` column on the event log).
EVT_INGEST_DUMP = "ingest_dump"
EVT_INGEST_PARTIAL_ASM = "ingest_partial_asm"
EVT_INGEST_TEXT = "ingest_text"
EVT_TOOL_RESULT = "tool_result"
EVT_LABEL = "label"
EVT_ROUTINE = "routine"
EVT_DATA_STRUCTURE = "data_structure"
EVT_HYPOTHESIS = "hypothesis"
EVT_ANALYSIS = "analysis"
EVT_VERDICT = "verdict"
EVT_CONSOLIDATED = "consolidated_observation"

# Version of the DERIVED SQLite view. The view now persists between
# processes (tracker 1.6 — incremental replay from a high-water offset
# instead of a full rebuild per start); bump this whenever SCHEMA_DDL
# changes shape so stale derived DBs trigger a full rebuild from kb.json.
# v3: routines.confidence added (tracker 2.3 — confidence-aware replay).
# v4: compacted_tool_results added (tracker 2.1 hardening — exact,
#     event-ID-based curator bookkeeping instead of timestamp cursors).
SCHEMA_VERSION = "4"

# DDL — all tables created on `KnowledgeStore.load_or_init`.
SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id           TEXT PRIMARY KEY,
    ts           TEXT NOT NULL,
    kind         TEXT NOT NULL,
    source       TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);

CREATE TABLE IF NOT EXISTS labels (
    addr            INTEGER NOT NULL,
    name            TEXT    NOT NULL,
    kind            TEXT,
    confidence      REAL    DEFAULT 0.5,
    source_event_id TEXT,
    PRIMARY KEY (addr, name)
);

CREATE TABLE IF NOT EXISTS routines (
    start         INTEGER PRIMARY KEY,
    end           INTEGER NOT NULL,
    name          TEXT,
    summary       TEXT,
    calls_to_json TEXT,
    called_by_json TEXT,
    confidence    REAL DEFAULT 0.5
);

CREATE TABLE IF NOT EXISTS data_structures (
    start       INTEGER PRIMARY KEY,
    end         INTEGER NOT NULL,
    kind        TEXT,
    fields_json TEXT
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    status        TEXT DEFAULT 'open',
    evidence_json TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS text_docs (
    path            TEXT PRIMARY KEY,
    ts              TEXT NOT NULL,
    size            INTEGER NOT NULL,
    mtime           REAL,
    content         TEXT NOT NULL,
    source_event_id TEXT
);

-- Content-hash identity of every tool_result event: indexed dedup for
-- the synthesizer instead of a full event-log scan (tracker 1.6).
CREATE TABLE IF NOT EXISTS tool_result_keys (
    key      TEXT PRIMARY KEY,
    event_id TEXT NOT NULL
);

-- Exact curator bookkeeping (tracker 2.1): one row per tool_result event
-- a consolidated_observation actually summarised. Populated on replay of
-- consolidated events' `events_compacted` lists — never inferred from
-- timestamps, so equal-timestamp events can't be skipped and legacy
-- consolidated events can never hide evidence they did not summarise.
CREATE TABLE IF NOT EXISTS compacted_tool_results (
    event_id        TEXT PRIMARY KEY,
    consolidated_id TEXT NOT NULL
);
"""

# Drop all tables so the derived view can be rebuilt from kb.json without
# unlinking the file (unlinking breaks concurrent processes that hold an
# open connection to the old inode).
CLEAR_DDL = """
DROP TABLE IF EXISTS compacted_tool_results;
DROP TABLE IF EXISTS tool_result_keys;
DROP TABLE IF EXISTS text_docs;
DROP TABLE IF EXISTS meta;
DROP TABLE IF EXISTS hypotheses;
DROP TABLE IF EXISTS data_structures;
DROP TABLE IF EXISTS routines;
DROP TABLE IF EXISTS labels;
DROP TABLE IF EXISTS events;
"""
