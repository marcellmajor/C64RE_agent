# C64-RE Agent

An autonomous, multi-agent deep-research system that reverse-engineers Commodore 64 games from a memory dump and answers specific questions about the code — *"Where is the lives counter stored?", "How does the level loader work?", "What is the copy-protection check?"*

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). Designed for retro-computing researchers, ROM hackers, demoscene historians, and game preservationists.

---

## What it does

Given a game name, a research question, and a 64 KB memory dump (plus optional disassembly files and human notes), the agent:

1. **Plans** a sequence of investigative steps ordered from cheapest to most expensive.
2. **Executes** each step by calling tools: static disassembly (Capstone), live emulator introspection (VICE MCP), and web search (Tavily). Up to four dependency-ready, read-only calls may fan out concurrently; mutations, retries, and LLM-backed Code-KB writes stay serial.
3. **Synthesises** raw tool output into a typed knowledge base — labelled addresses, named routines, cross-references, hypotheses.
4. **Analyses** the accumulated KB to produce a candidate answer with evidence and a confidence score.
5. **Critiques** the answer adversarially; if unsatisfied, replans and loops (up to 12 iterations by default).
6. **Writes a versioned Markdown report** to
   `sessions/<game>/report_<timestamp>_<run>.md`, refreshes `report.md`, and
   archives the turn in `turns.jsonl`.

Knowledge is persisted across runs — restart the agent on the same game and it picks up from where it left off.

---

## Architecture

```
load_inputs
    └─▶ planner ──▶ executor ──▶ [vice | capstone | tavily | kb | code_kb]
                                    └─▶ synthesizer
                                            ├─ compaction needed ─▶ curator
                                            │                         ├─ more steps ─▶ executor
                                            │                         └─ done ───────▶ analyst
                                            ├─ more steps ─────────▶ executor
                                            └─ done ───────────────▶ analyst

analyst ──▶ critic
                ├─ accept ──────────▶ write_report
                ├─ revise ──────────▶ analyst
                ├─ replan ──────────▶ planner
                └─ budget/loop stop ▶ write_report
```

The compaction gate takes priority over pending steps. A blocked executor
plan can also return directly to the planner, within the iteration limit.

### Sub-agents

| Agent | Role |
|---|---|
| **Planner** | Decomposes the question into an ordered list of tool-call steps with hypotheses |
| **Executor** | Picks the next runnable step (respecting `depends_on`), enriches null arg placeholders from the live KB |
| **Synthesizer** | Extracts structured facts (labels, routines, data structures, hypotheses) from raw tool results and writes them to the KB |
| **Analyst** | Answers the question using only KB-backed evidence; emits confidence score and open questions |
| **Critic** | Adversarially reviews the answer; decides `accept` / `revise` / `replan` |
| **Curator** | Summarises uncompacted tool results when their estimated token count exceeds 40,000; original evidence stays in the append-only KB |
| **Vision** | Describes an explicitly requested VICE screenshot for downstream evidence extraction |

### Tools

| Tool | What it does |
|---|---|
| **Capstone** | Static 6502/6510 disassembly — linear, recursive, loop-finding, entry-point detection, polymorphic/SMC scan |
| **VICE MCP** | Live emulator introspection via MCP — disassembly/read, snapshots and diffs, watchpoint traces, screenshots, and approval-gated visual poke verification |
| **Tavily** | Web search scoped to retrocomputing sources (CSDb, Codebase64, Lemon64, GameBase64) |
| **KB** | SQL and text queries against the accumulated knowledge base |
| **Code KB** | Layered code-comprehension store built from parsed asm files — routines, xrefs, address-preserving pseudocode, semantic search, evidence-gated Layer-2 groups, and append-only Layer-3 critiques |

### Knowledge stores

Two separate SQLite-backed stores accumulate across sessions:

- `sessions/<game>/kb/` — general KB (labels, routines, hypotheses, raw tool results, user notes)
- `sessions/<game>/code_kb/` — code-specific KB built from asm/dump files (instructions, xrefs, LLM-discovered routines mirrored from the general KB)

---

## Setup

```bash
# 1. Clone and create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -e ".[ui]"           # includes Streamlit; omit [ui] for CLI-only

# 3. Copy and fill in API keys
cp .env.example .env             # edit .env with your keys
```

The package also builds as a standard wheel (`uv build --wheel`) and installs a
`c64re` command. By default, writable data is relative to the directory where
the command is launched. Set `C64RE_WORKSPACE_DIR` to relocate the default
session root and the UI's default input directories, `C64RE_SESSIONS_DIR` to
relocate only durable sessions, or
`C64RE_CONFIG_DIR` to use an external `llm.json`/`kb_semantic.json` directory.
The CLI's default `./asm_dir` and `./text_dir` still resolve relative to the
current working directory; pass `--asm-dir` and `--text-dir` explicitly to
use input directories elsewhere.
Set these overrides before starting/importing the CLI, Streamlit UI, graph, or
semantic configuration modules. Those application modules deliberately
snapshot paths at import so a running process cannot silently switch its
configuration or writable session root; restart after changing an override.

### `.env` keys

```
# Required — at least one LLM provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
XAI_API_KEY=xai-...
GEMINI_API_KEY=...

# Optional — web search
TAVILY_API_KEY=tvly-...

# Optional — live VICE emulator
VICE_MCP_URL=http://localhost:3000

# Optional — LangSmith tracing
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=c64re
```

### LLM configuration

Edit `config/llm.json` to choose which model each sub-agent uses. Any sub-agent
can use a different provider. The following is an illustrative `agents`
section, not a snapshot of the current configuration; preserve the file's
`providers` section and select models available to your accounts. Configure
credentials for the selected providers, or assign all roles to the provider
you intend to use:

```json
{
  "agents": {
    "planner":     { "provider": "anthropic", "model": "claude-opus-4-8",  "temperature": 0.2, "max_tokens": 8192 },
    "executor":    { "provider": "openai",    "model": "gpt-5.5",          "temperature": 0.0 },
    "synthesizer": { "provider": "grok",      "model": "grok-4.5",         "temperature": 0.3, "reasoning_effort": "low" },
    "analyst":     { "provider": "gemini",    "model": "gemini-pro-latest", "temperature": 0.2, "max_tokens": 8192, "reasoning_effort": "low" },
    "critic":      { "provider": "gemini",    "model": "gemini-pro-latest", "temperature": 0.1, "max_tokens": 8192, "reasoning_effort": "low" },
    "curator":     { "provider": "grok",      "model": "grok-4.5",         "temperature": 0.2, "reasoning_effort": "low" },
    "vision":      { "provider": "gemini",    "model": "gemini-pro-latest", "temperature": 0.1, "max_tokens": 2048, "reasoning_effort": "low" }
  }
}
```

Ollama (local) is also supported — set `"provider": "ollama"` and point `base_url` at `http://localhost:11434/v1`.

Native provider JSON-schema output is available as a compatibility-gated option. Set
`C64RE_STRUCTURED_OUTPUT=1`, `defaults.structured_output`, or an
`agents.<role>.structured_output` override to try it. It is disabled by
default until the paid golden suite has baselined every configured provider;
provider rejection automatically retries that role through the ordinary JSON
contract before using a backup provider.

### Semantic KB search

`config/kb_semantic.json` controls optional embedding search. When enabled,
vectors are persisted in `sessions/<game>/kb/vectors.sqlite`, keyed by the
provider/model/dimension and exact content hash. Startup and incremental
indexing embed only cache misses, so reopening an unchanged session incurs no
embedding calls. The supported default is OpenAI `text-embedding-3-small`
through the configured `executor` provider. Automatic indexing remains
disabled until `enabled` is explicitly set to `true`, so opening a session
cannot silently incur embedding spend.

---

## Usage

### Command line

```bash
python main.py \
  --game "Wizard of Wor" \
  --question "Where is the player score stored and how is it updated?" \
  --dump memdump_dir/wizofwor.dump
```

An installed wheel exposes the equivalent `c64re` command with the same
arguments.

**All options:**

```
--game          Game name (required)
--question      Research question (required)
--dump          Path to 64 KB memory dump (required)
--partial-asm   Single annotated .asm file with known labels/comments
--asm-dir       Directory of .asm/.txt disassembly files (default: ./asm_dir)
--asm-files     Explicit asm file list (overrides filename-token scoping)
--text-dir      Directory of human-written notes .txt/.md (default: ./text_dir)
--thread-id     Checkpoint thread (default: <game-slug>-<question-hash>)
--reset-code-kb Wipe the code KB before running (use if a prior run loaded wrong files)
--approve-vice-mutations
                Explicitly allow VICE steps that change emulator state
--no-trace      Disable LangSmith tracing for this run
```

**Continue researching the same game** — re-run with the same `--game` and a
new `--question`. The KB from prior sessions is loaded automatically, while
the new question gets a separate checkpoint thread:

```bash
python main.py \
  --game "Wizard of Wor" \
  --question "How does the maze generation work?" \
  --dump memdump_dir/wizofwor.dump
```

**Reuse a checkpoint thread** — repeat the same game and question, or pass
the previous `--thread-id`. Checkpoints persist in
`sessions/<game-slug>/checkpoint.sqlite`. Re-invocation starts graph
processing again over the saved state; it does not resume an interrupted
node in place.

### LangGraph Studio / LangSmith

```bash
uv run langgraph dev
```

Opens LangGraph Studio in the browser for step-by-step graph inspection, node-by-node state viewing, and full LangSmith tracing.

### Trusted research inputs

Files passed through `--text-dir`, `--partial-asm`, or `--asm-dir` are stored
in the local session KB and relevant excerpts may be sent to the configured
LLM providers. Only ingest material you trust and are permitted to share.
API-key, bearer-token, password, and similar credential shapes are redacted
from generated digests, search/SQL previews, and tool output, but this is a
best-effort safeguard rather than a substitute for keeping secrets out of
research notes. Original local evidence remains unchanged in the append-only
KB and should be protected as sensitive session data.

### Streamlit Web GUI

```bash
source .venv/bin/activate
streamlit run app.py
```

The web UI provides:

- **Chat interface** — type your question, pick a game and dump file from the sidebar, click Send
- **Plan review checkpoint** — edit, reorder, or delete planner steps before tools run; mutating VICE steps require their own checkboxes and a pending run can be cancelled
- **Research notebook** — prior accepted turns reload from `turns.jsonl`, inform later questions, and expose one-click open-question follow-ups
- **Named dump states** — freeze immutable title/ingame/death/level dumps and compare them offline
- **Code lens panel** — after each answer, addresses cited (`$XXXX`) auto-populate a disassembly viewer and interactive call graph
- **Routines panel** — top routines ranked by cross-reference count, with one-click jump to their disassembly
- **VICE symbol export** — download verified, high-confidence Code-KB names as a `.labels` monitor command file
- **Knowledge accumulation** — every turn enriches the same file-backed KB; close and reopen the browser and your session is intact
- **Human ratings** — mark archived answers helpful/not helpful with an optional note; revisions stay local until explicitly exported

---

## Directory layout

```
config/
  llm.json          LLM provider + model per sub-agent
  kb_semantic.json  Embedding model config for semantic KB search
graph/
  build.py          LangGraph StateGraph wiring
  nodes.py          All node implementations (planner, executor, synthesizer, …)
  routers.py        Conditional-edge routing functions
  state.py          C64State TypedDict
  prompts.py        System prompt + per-agent role blocks
  llm.py            Provider-agnostic LLM factory
memory/
  store.py          KnowledgeStore — general KB (SQLite + JSONL event log)
  evidence.py       Shared evidence-reference resolver for reports, CLI, and UI
  semantic_cache.py Persistent content-addressed embedding cache
  redaction.py      Best-effort secret redaction at prompt/preview boundaries
code_kb/
  store.py          CodeKnowledgeStore — layered code-comprehension KB
  layer0.py         Ground-truth layer: parsed asm / dump bytes
  layer1.py         LLM annotation layer: routines, labels, xrefs
  pseudocode.py     Conservative address-preserving 6502 transliteration
  call_graph.py     DOT call-graph builder for the web GUI
tools/
  vice_mcp.py       MCP client for VICE emulator
  c64_disasm.py     Capstone 6502/6510 disassembly wrapper
  agent_runner.py   Headless runner used by the Streamlit UI
  research_notebook.py  Turn archive, report-version, and dump-catalog services
evals/
  runner.py         Opt-in live golden runner and offline evaluator
  ratings.py        Explicit local-rating preview / LangSmith dataset export
asm_dir/            Partial disassembly files (game disassemblies go here)
memdump_dir/        Memory dumps (.dump files go here)
text_dir/           Human-written notes and research documents
sessions/           Persistent per-game knowledge bases and reports
main.py             CLI entry point
app.py              Streamlit web GUI
```

---

## Requirements

- Python 3.10+
- At least one LLM API key (OpenAI, Anthropic, xAI, Gemini, or a local Ollama instance)
- Streamlit + Graphviz for the web GUI (`pip install -e ".[ui]"`)
- VICE with the [vice-mcp](https://github.com/vice-emu/vice-mcp) plugin for live emulator features (optional)

Mutating VICE work is deny-by-default. The `vice.poke_verify` experiment is
limited to one byte, requires an explicit expected visible change, captures
before/after screenshots, and uses the selected bank consistently for the
original read, candidate write, verification read, and byte restore. It saves
then reloads a complete emulator snapshot in `finally`; older servers fall
back to restoring the original byte in that same bank. Treat optional execution resume as
experimental because the current VICE MCP run verb is not frame-bounded. Use
a disposable session and review the generated plan before approval.

---

## Evaluation

The normal test suite validates deterministic analysis, persistence, and the
golden manifest without making paid model calls. Its offline dump-anchor gate
checks every expected address for dump-backed bytes or 6502 references and
rejects exact-address assembly bytes that disagree with the selected dump.

Install the development extra to obtain `pytest` (use `.[ui,dev]` if you also
want the GUI). Before running the full suite, supply the dump and assembly
files at the paths listed in `evals/golden.jsonl`. The `memdump_dir/` and
`asm_dir/` corpus directories are Git-ignored and are not included in a fresh
clone; the golden-manifest and dump-anchor tests require those local files
even when live evaluation is disabled.

```bash
pip install -e ".[dev]"
pytest -q
```

A budget-limited live hardening baseline exercised 11 of the 15 cases on
2026-07-18. After grounding and evaluator-contract fixes, all 11 produced the
required addresses at or above their confidence floor; four cases remain
unbaselined (`vultures_lives_initial`, `vultures_lives_decrement`,
`wor_sprite_enable`, and `wor_screen_ram`). The dump-anchor tests therefore
remain the required offline gate. The end-to-end golden suite is opt-in. It runs the real graph,
stores its temporary KBs under `evals/.sessions/`, and writes per-case quality,
token, cost, tool-call, and wall-time metrics under `evals/results/`. VICE and
Tavily are disabled by default so the locally supplied corpus files remain the stable
inputs:

```bash
C64RE_RUN_GOLDEN=1 python -m evals.runner
```

Use `--case <id>` to run a subset. Live evaluation requires the configured LLM
credentials; it does not run as part of ordinary `pytest`. Pass
`--allow-external-tools` only when emulator/web variability is intentional.
`C64RE_USD_BUDGET` sets the estimated per-run graph guardrail and
`C64RE_MAX_OUTPUT_TOKENS` caps each completion. Each primary/backup call now
reserves a conservative worst-case input/output estimate before contacting a
provider. The runner's `--max-cost-usd` adds a cumulative batch ceiling and
`--min-case-budget-usd` prevents starting a case without useful headroom.
These are conservative application-side controls, not a transactional limit
inside a provider's billing system.

Future validation still includes the four named live cases above and a paired
structured-output A/B: plain JSON as the control versus
`C64RE_STRUCTURED_OUTPUT_ROLES=analyst,critic` as the treatment. Use fresh
session roots, identical model/config versions, VICE/Tavily disabled, and
compare quality, contract/fallback failures, tokens, estimated cost, calls, and
wall time before changing the default. A separate semantic-search A/B should
compare disabled, cold-cache, and warm-cache `text-embedding-3-small` runs
before automatic embedding is enabled.

Human ratings are recorded locally by the Streamlit UI. Preview the joined
turn/rating examples without network access, or explicitly synchronize them to
a LangSmith dataset:

```bash
python -m evals.ratings --session-dir sessions/wizard_of_wor --local-only
python -m evals.ratings --session-dir sessions/wizard_of_wor \
  --dataset c64re-human-ratings
```

There is no automatic LangSmith dataset write when a user clicks a rating.
