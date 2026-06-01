"""SQLite schema + event-kind constants for the Code Knowledge Store.

This is the *layered code-comprehension* KB — completely separate from the
parent `memory.KnowledgeStore` so:

- Deterministic Layer-0 facts can never be overwritten by an LLM.
- The schema is purpose-built around 6502/6510 reality (multi-entry
  routines, SMC, xref graph, code/data ambiguity) rather than the parent
  agent's general label/routine/hypothesis triplet.

Following the v1 spec:
    L0  – deterministic ground truth (no LLM)
    L1  – per-window semantic annotation (LLM)
    L2  – global behavioural grouping (LLM)
    L3  – critique pass (LLM, ideally a different model)

Annotations are stored as a single uniform row keyed by `id`, and the
purpose-specific tables (`code_routines`, `code_xrefs`, …) are derived
views — they're rebuilt on every `load_or_init` from the append-only
event log so they can never drift.
"""

from __future__ import annotations

# ---- event kinds ---------------------------------------------------------- #

EVT_INGEST_ASM = "ingest_asm"               # an entire .asm/.txt file ingested
EVT_INGEST_DUMP = "ingest_dump"             # raw 64 KB binary attached
EVT_ANNOTATION = "annotation"               # a layered annotation row
EVT_DISASM_WINDOW = "disasm_window"         # results of a disasm tool call
EVT_LAYER_RUN = "layer_run"                 # bookkeeping: which layer ran when

# ---- annotation kinds (the `kind` column on `annotations`) ---------------- #
# v1 covers what's required by the build-spec; later layers can add new kinds
# without a schema migration since the only table that needs updating is
# `annotations` (everything else is derived).

ANN_ASM_DOC      = "asm_doc"           # source partial-asm document
ANN_LABEL        = "label"             # named address (deterministic or L1+)
ANN_ROUTINE      = "routine"           # contiguous instruction range
ANN_XREF         = "xref"              # call/jump/branch edge
ANN_SMC          = "smc"               # self-modifying code suspect
ANN_CLASSIFY     = "classify"          # per-byte code/data classification
ANN_INDIRECT     = "indirect_jump"     # JMP ($XXXX) site (target may be runtime)
ANN_DISASM       = "disasm_window"     # one disassembly window's listing
ANN_HYPOTHESIS   = "hypothesis"        # L1/L2 freeform hypothesis (idiom/purpose/motivation)
ANN_GROUP        = "group"             # L2 cross-routine grouping
ANN_CRITIQUE     = "critique"          # L3 flag on a prior annotation


# ---- DDL ------------------------------------------------------------------ #
# Notes:
#   - All addresses are stored as INTEGER (16-bit) — render hex on the way out.
#   - `events` is the source of truth; everything else is rebuilt on replay.
#   - `annotations` is the canonical typed view; layer-specific tables are
#     thin projections of the row's payload for fast indexed access.

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id           TEXT PRIMARY KEY,
    ts           TEXT NOT NULL,
    kind         TEXT NOT NULL,
    source       TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);

CREATE TABLE IF NOT EXISTS annotations (
    id              TEXT PRIMARY KEY,
    layer           INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    start_addr      INTEGER NOT NULL,
    end_addr        INTEGER NOT NULL,
    producer        TEXT NOT NULL,
    confidence      REAL    DEFAULT 0.5,
    payload_json    TEXT    NOT NULL,
    evidence_json   TEXT,
    supersedes_json TEXT,
    flags_json      TEXT,
    ts              TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ann_layer  ON annotations(layer);
CREATE INDEX IF NOT EXISTS idx_ann_kind   ON annotations(kind);
CREATE INDEX IF NOT EXISTS idx_ann_start  ON annotations(start_addr);
CREATE INDEX IF NOT EXISTS idx_ann_prod   ON annotations(producer);

CREATE TABLE IF NOT EXISTS code_routines (
    start_addr   INTEGER PRIMARY KEY,
    end_addr     INTEGER NOT NULL,
    name         TEXT,
    summary      TEXT,
    source_file  TEXT,
    entries_json TEXT,
    exits_json   TEXT,
    size_bytes   INTEGER,
    annotation_id TEXT,
    confidence   REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS code_xrefs (
    src_addr     INTEGER NOT NULL,
    dst_addr     INTEGER,
    via_vector   INTEGER,
    kind         TEXT NOT NULL,
    annotation_id TEXT,
    PRIMARY KEY (src_addr, dst_addr, kind, via_vector)
);
CREATE INDEX IF NOT EXISTS idx_xrefs_dst ON code_xrefs(dst_addr);

CREATE TABLE IF NOT EXISTS code_smc_sites (
    src_addr     INTEGER NOT NULL,
    dst_addr     INTEGER,
    mnemonic     TEXT,
    operand      TEXT,
    smc_kind     TEXT,
    annotation_id TEXT,
    PRIMARY KEY (src_addr, dst_addr, smc_kind)
);

CREATE TABLE IF NOT EXISTS code_class (
    addr         INTEGER PRIMARY KEY,
    classification TEXT NOT NULL,
    confidence   REAL,
    evidence     TEXT,
    annotation_id TEXT
);

CREATE TABLE IF NOT EXISTS code_labels (
    addr         INTEGER NOT NULL,
    name         TEXT NOT NULL,
    source_file  TEXT,
    annotation_id TEXT,
    PRIMARY KEY (addr, name)
);
CREATE INDEX IF NOT EXISTS idx_labels_addr ON code_labels(addr);

CREATE TABLE IF NOT EXISTS asm_docs (
    path           TEXT PRIMARY KEY,
    size           INTEGER,
    mtime          REAL,
    content        TEXT NOT NULL,
    instructions   INTEGER,
    routines       INTEGER,
    annotation_id  TEXT
);

CREATE TABLE IF NOT EXISTS instructions (
    addr         INTEGER PRIMARY KEY,
    bytes_hex    TEXT NOT NULL,
    mnemonic     TEXT NOT NULL,
    operand      TEXT,
    size_bytes   INTEGER NOT NULL,
    source_file  TEXT,
    annotation_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_insns_mnem ON instructions(mnemonic);

CREATE TABLE IF NOT EXISTS hypotheses (
    annotation_id TEXT PRIMARY KEY,
    layer         INTEGER NOT NULL,
    start_addr    INTEGER NOT NULL,
    end_addr      INTEGER NOT NULL,
    text          TEXT NOT NULL,
    name_suggestion TEXT,
    idiom_match   TEXT,
    hardware_touched_json TEXT,
    routine_id    TEXT,
    source_file   TEXT,
    confidence    REAL,
    flags_json    TEXT,
    producer      TEXT
);
CREATE INDEX IF NOT EXISTS idx_hyps_start ON hypotheses(start_addr);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Drop all tables so the derived view can be rebuilt without unlinking the
# file — unlinking breaks concurrent processes that hold an open connection
# to the old inode (they can no longer create journal/WAL files).
CLEAR_DDL = """
DROP TABLE IF EXISTS meta;
DROP TABLE IF EXISTS hypotheses;
DROP TABLE IF EXISTS instructions;
DROP TABLE IF EXISTS asm_docs;
DROP TABLE IF EXISTS code_labels;
DROP TABLE IF EXISTS code_class;
DROP TABLE IF EXISTS code_smc_sites;
DROP TABLE IF EXISTS code_xrefs;
DROP TABLE IF EXISTS code_routines;
DROP TABLE IF EXISTS annotations;
DROP TABLE IF EXISTS events;
"""
