# C64-RE Agent

An autonomous, multi-agent deep-research system that reverse-engineers Commodore 64 games from a memory dump and answers specific questions about the code — *"Where is the lives counter stored?", "How does the level loader work?", "What is the copy-protection check?"*

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). Designed for retro-computing researchers, ROM hackers, demoscene historians, and game preservationists.

---

## What it does

Given a game name, a research question, and a 64 KB memory dump (plus optional disassembly files and human notes), the agent:

1. **Plans** a sequence of investigative steps ordered from cheapest to most expensive.
2. **Executes** each step by calling tools: static disassembly (Capstone), live emulator introspection (VICE MCP), and web search (Tavily).
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
                       ▲               └──────────────────────────────────▶ synthesizer
                       │                                                          │
                       │◀──── (more steps pending) ───────────────────────────────┤
                       │                                                          │
                    curator ◀─── (KB too large) ─────────────────────────────────▶│
                                                                                  ▼
                                                                               analyst
                                                                                  │
                                                                               critic
                                                                          ┌───────┴────────┐
                                                                       accept           replan/revise
                                                                          │                  │
                                                                     write_report         planner
```

### Sub-agents

| Agent | Role |
|---|---|
| **Coordinator** | Orchestrates the workflow, routes tasks, and manages overall progress |
| **Planner** | Decomposes the question into an ordered list of tool-call steps with hypotheses |
| **Executor** | Picks the next runnable step (respecting `depends_on`), enriches null arg placeholders from the live KB |
| **Synthesizer** | Extracts structured facts (labels, routines, data structures, hypotheses) from raw tool results and writes them to the KB |
| **Analyst** | Answers the question using only KB-backed evidence; emits confidence score and open questions |
| **Critic** | Adversarially reviews the answer; decides `accept` / `revise` / `replan` |
| **Curator** | Compacts the KB when it grows beyond the token budget of the cheapest model |
| **Researcher** | Performs targeted web searches and extracts relevant external knowledge |

### Tools

| Tool | What it does |
|---|---|
| **Capstone** | Static 6502/6510 disassembly — linear, recursive, loop-finding, entry-point detection, polymorphic/SMC scan |
| **VICE MCP** | Live emulator introspection via MCP — `disassemble`, `memory.read`, `registers.get`, `display.screenshot` |
| **Tavily** | Web search scoped to retrocomputing sources (CSDb, Codebase64, Lemon64, GameBase64) |
| **KB** | SQL and text queries against the accumulated knowledge base |
| **Code KB** | Layered code-comprehension store built from parsed asm files — routines, xrefs, instruction index, semantic search |

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

Edit `config/llm.json` to choose which model each sub-agent uses. Any sub-agent can use a different provider:

```json
{
  "agents": {
    "coordinator": { "provider": "grok",      "model": "grok-4.3",         "temperature": 0.1, "reasoning_effort": "low" },
    "planner":     { "provider": "anthropic", "model": "claude-opus-4-7",  "temperature": 0.2 },
    "executor":    { "provider": "openai",    "model": "gpt-5.3-codex",    "temperature": 0.0 },
    "synthesizer": { "provider": "grok",      "model": "grok-4.3",         "temperature": 0.3, "reasoning_effort": "low" },
    "analyst":     { "provider": "gemini",    "model": "gemini-pro-latest", "temperature": 0.2 },
    "critic":      { "provider": "gemini",    "model": "gemini-pro-latest", "temperature": 0.1 },
    "curator":     { "provider": "grok",      "model": "grok-4.3",         "temperature": 0.2, "reasoning_effort": "none" },
    "researcher":  { "provider": "grok",      "model": "grok-4.3",         "temperature": 0.3, "reasoning_effort": "medium" }
  }
}
```

Ollama (local) is also supported — set `"provider": "ollama"` and point `base_url` at `http://localhost:11434/v1`.

---

## Usage

### Command line

```bash
python main.py \
  --game "Wizard of Wor" \
  --question "Where is the player score stored and how is it updated?" \
  --dump memdump_dir/wizofwor.dump
```

**All options:**

```
--game          Game name (required)
--question      Research question (required)
--dump          Path to 64 KB memory dump (required)
--partial-asm   Single annotated .asm file with known labels/comments
--asm-dir       Directory of .asm/.txt disassembly files (default: ./asm_dir)
--asm-files     Explicit asm file list (overrides filename-token scoping)
--text-dir      Directory of human-written notes .txt/.md (default: ./text_dir)
--thread-id     Resume a previous session (default: derived from game name)
--reset-code-kb Wipe the code KB before running (use if a prior run loaded wrong files)
--approve-vice-mutations
                Explicitly allow VICE steps that change emulator state
--no-trace      Disable LangSmith tracing for this run
```

**Resume a previous run** — just re-run with the same `--game` and a new `--question`. The KB from prior sessions is loaded automatically:

```bash
python main.py \
  --game "Wizard of Wor" \
  --question "How does the maze generation work?" \
  --dump memdump_dir/wizofwor.dump
```

### LangGraph Studio / LangSmith

```bash
uv run langgraph dev
```

Opens LangGraph Studio in the browser for step-by-step graph inspection, node-by-node state viewing, and full LangSmith tracing.

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
code_kb/
  store.py          CodeKnowledgeStore — layered code-comprehension KB
  layer0.py         Ground-truth layer: parsed asm / dump bytes
  layer1.py         LLM annotation layer: routines, labels, xrefs
  call_graph.py     DOT call-graph builder for the web GUI
tools/
  vice_mcp.py       MCP client for VICE emulator
  c64_disasm.py     Capstone 6502/6510 disassembly wrapper
  agent_runner.py   Headless runner used by the Streamlit UI
  research_notebook.py  Turn archive, report-version, and dump-catalog services
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

---

## Evaluation

The normal test suite validates deterministic analysis, persistence, and the
golden manifest without making paid model calls. Its offline dump-anchor gate
checks every expected address for dump-backed bytes or 6502 references and
rejects exact-address assembly bytes that disagree with the selected dump:

```bash
pytest -q
```

The live suite has not yet been baselined; the dump-anchor tests are currently
the required offline gate. The 15-question end-to-end golden suite is opt-in.
It runs the real graph,
stores its temporary KBs under `evals/.sessions/`, and writes per-case quality,
token, cost, tool-call, and wall-time metrics under `evals/results/`. VICE and
Tavily are disabled by default so the checked-in dumps/asm remain the stable
inputs:

```bash
C64RE_RUN_GOLDEN=1 python -m evals.runner
```

Use `--case <id>` to run a subset. Live evaluation requires the configured LLM
credentials; it does not run as part of ordinary `pytest`. Pass
`--allow-external-tools` only when emulator/web variability is intentional.
