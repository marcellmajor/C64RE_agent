Create a description and prompt based on these equirements:
"
I would like to create a C64 Reverse Engineering Agent as a deep research agent which:
Use langgraph and langchain
has a config file with possible LLM configs to use for openapi compatible connection with ollama, grok, gemini, chatgpt, claude
have different sub-agents for (add if you think refinement needed):
    - creating a plan for the activity
    - using tools like vice mcp, tavily search, capstone engine
    - synthesize the knowledge into the working knowledge of the agent for iterating 
    - analyzes the result os the tool runs
    - analyst / crytic decides if the results are suitable or more research is needed 
starts with 2 inputs: the name of the C64 game and the question the user wants to know from the memdump
loads a binary memdump of a running C64 game
possibly loads a manually created partial disasembly
handles a knowledge/memory on the information gathered, stored in a way it can be loaded after stopping the agent
"


==========================================================
Claude answer:
==========================================================
C64 Reverse Engineering Agent — Description & Prompt
Below is a complete specification document you can use to scaffold the agent. It's split into two parts: a product description (the "what" and "why") and a system prompt / architecture spec (the "how"), with concrete config schemas, sub-agent contracts, and the orchestration loop.

Part 1 — Product Description
Name
C64-RE Agent (Commodore 64 Reverse Engineering Deep Research Agent)
One-line summary
An autonomous, multi-agent deep-research system that reverse-engineers Commodore 64 games from a live memory dump (and optional partial disassembly) to answer specific user questions — e.g. "Where is the lives counter stored?", "How does the level loader decode sprite data?", "What's the copy-protection check?"
Who it's for
Retro-computing researchers, demoscene historians, game preservationists, ROM hackers, and hobbyists who want LLM-assisted analysis of 6510 machine code without manually grinding through 64 KB of memory.
Core inputs

Game name (string) — used for lookup in online databases (GameBase64, CSDb, Lemon64) and to scope the research.
Research question (string) — the concrete thing the user wants to know.
Memory dump (binary file, typically 64 KB) — a snapshot of $0000–$FFFF from a running VICE session.
(Optional) Partial disassembly (text/asm file) — human-annotated labels, comments, or known routines the user already mapped.

Core outputs

A structured research report (Markdown) answering the question, with addresses, disassembled snippets, hypotheses tested, and confidence scores.
A persistent knowledge base (JSON + Markdown) that grows across sessions — labeled addresses, identified routines, data structures, KERNAL/BASIC call sites, cross-references.
A session transcript for reproducibility.

Key design properties

Provider-agnostic LLM layer — single config file selects between Ollama, Grok (xAI), Gemini, OpenAI, or Claude via OpenAI-compatible endpoints. Different sub-agents can use different models (e.g. cheap local Ollama for synthesis, Claude for critique).
Multi-agent with a critic — a planner, executor, synthesizer, analyst, and critic loop until the critic signs off or budget is exhausted.
Tool-using — wraps VICE MCP (live emulator introspection), Capstone (disassembly), and Tavily (web search for known cracks/docs).
Resumable — the knowledge store is pickle/JSON-serializable; agent can be killed and restarted without losing progress.


Part 2 — System Prompt & Architecture Spec
2.1 Repository layout
c64re-agent/
├── config/
│   ├── llm.yaml              # provider + model per sub-agent
│   └── tools.yaml            # MCP endpoints, API keys (env-var refs)
├── agents/
│   ├── planner.py
│   ├── executor.py
│   ├── synthesizer.py
│   ├── analyst.py
│   └── critic.py
├── tools/
│   ├── vice_mcp.py           # MCP client for VICE
│   ├── capstone_disasm.py    # 6502/6510 disassembly wrapper
│   └── tavily_search.py
├── memory/
│   ├── store.py              # persistent KB (JSON + SQLite for x-refs)
│   └── schema.py
├── inputs/
│   ├── <game>.dump           # 64 KB memory dump
│   └── <game>.partial.asm    # optional
├── sessions/
│   └── <session_id>/         # transcripts, intermediate artifacts
└── main.py                   # orchestrator
2.2 LLM config file (config/llm.yaml)
All providers are accessed through an OpenAI-compatible client (most natively support it; Gemini and Claude are reached via their OpenAI-compat endpoints or a thin shim).
yamlproviders:
  ollama:
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"          # placeholder; Ollama ignores it
  grok:
    base_url: "https://api.x.ai/v1"
    api_key: "${XAI_API_KEY}"
  gemini:
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: "${GEMINI_API_KEY}"
  openai:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
  claude:
    base_url: "https://api.anthropic.com/v1"
    api_key: "${ANTHROPIC_API_KEY}"

# Each sub-agent picks a provider + model. Mix and match freely.
agents:
  planner:      { provider: claude,  model: claude-opus-4-7,    temperature: 0.2 }
  executor:     { provider: openai,  model: gpt-4.1,            temperature: 0.0 }
  synthesizer:  { provider: ollama,  model: qwen2.5-coder:32b,  temperature: 0.3 }
  analyst:      { provider: gemini,  model: gemini-2.5-pro,     temperature: 0.2 }
  critic:       { provider: claude,  model: claude-opus-4-7,    temperature: 0.1 }

defaults:
  max_tokens: 4096
  timeout_s: 120
  retries: 3
2.3 Sub-agent roster & contracts
Each sub-agent is a class with a single run(state) -> state_delta method. State is a typed dict passed through the loop.
AgentResponsibilityReadsWritesPlannerDecomposes the research question into an ordered list of investigative steps (tool calls + hypotheses to test). Replans when the critic rejects.question, KB, prior critiqueplan: List[Step]ExecutorPicks the next unfinished step and dispatches the right tool (VICE MCP, Capstone, Tavily, or KB lookup). Handles retries and tool errors.plan, KBtool_results: List[ToolResult]SynthesizerFolds raw tool outputs into the working KB — normalizes addresses, labels, x-refs, and prose notes. This is the only agent that mutates long-term memory.tool_results, KBKB updatesAnalystInterprets the updated KB against the original question. Produces a candidate answer with cited evidence and a confidence score.KB, questioncandidate_answerCriticAdversarial check. Asks: is the answer actually supported? Are there alternative explanations? Is more evidence needed? Decides accept / revise / replan.candidate_answer, KB, questionverdict, critique
A 6th agent — Curator — is worth adding: it periodically prunes/compacts the KB so it stays under the context budget of the cheapest model in the pipeline. Recommended for any non-trivial run.
2.4 Tool wrappers
VICE MCP (tools/vice_mcp.py) — connects to a running VICE instance via the MCP server. Exposes:

read_memory(addr, length) — read RAM/ROM
disassemble(addr, length) — VICE's own disassembler (sanity check vs. Capstone)
set_breakpoint(addr) / step() / continue() — dynamic analysis
read_registers() — A/X/Y/SP/PC/flags
read_vic_state() / read_sid_state() / read_cia_state()

Capstone (tools/capstone_disasm.py) — static disassembly of arbitrary byte ranges from the dump. Configured for CS_ARCH_MOS65XX with CS_MODE_MOS65XX_6502. Returns structured instructions (mnemonic, operands, bytes, address) — far easier for the LLM to reason over than raw text.
Tavily (tools/tavily_search.py) — web search scoped to retrocomputing sources. Useful for: known crack docs on CSDb, GameBase64 entries, Codebase64 articles, forum threads about specific games. Always cited.
2.5 Persistent knowledge store
Two-tier:

kb.json — append-only event log of every observation (timestamp, source, address range, content, agent that produced it). Source of truth for replay.
kb.sqlite — derived, queryable view with tables:

labels(addr, name, kind, confidence, source_event_id)
routines(start, end, name, summary, calls_to TEXT[], called_by TEXT[])
data_structures(start, end, kind, fields_json)
hypotheses(id, text, status, evidence_event_ids)



The store is the agent's "working memory of the codebase." On startup, if a kb.json exists for the given game, it's loaded and replayed into SQLite — making the agent fully resumable.
2.6 Orchestration loop (pseudocode)
pythondef main(game: str, question: str, dump_path: Path, partial_asm: Path | None):
    cfg = load_config("config/llm.yaml")
    kb  = KnowledgeStore.load_or_init(game)
    kb.ingest_dump(dump_path)
    if partial_asm:
        kb.ingest_partial_disassembly(partial_asm)

    state = State(question=question, kb=kb, plan=[], history=[])
    planner, executor, synth, analyst, critic = build_agents(cfg)

    state |= planner.run(state)

    for iteration in range(MAX_ITERS):
        state |= executor.run(state)
        state |= synth.run(state)
        state |= analyst.run(state)
        verdict = critic.run(state)
        state.history.append(verdict)

        if verdict.decision == "accept":
            break
        elif verdict.decision == "revise":
            continue                       # analyst re-tries with current KB
        elif verdict.decision == "replan":
            state |= planner.run(state)    # critique fed back as input

        kb.checkpoint()                    # crash-safe

    write_report(state)
2.7 Master system prompt (shared preamble for all sub-agents)
Use this verbatim as the system message, then append the agent-specific role section.
You are part of C64-RE, a multi-agent system that reverse-engineers
Commodore 64 software from a 64 KB memory dump and optional partial
disassembly.

Hard facts you must respect:
- Target CPU is the MOS 6510 (6502-compatible). 8-bit, little-endian,
  16-bit address space $0000-$FFFF.
- $0000-$0001 are the processor port. Zero page is $0002-$00FF.
- Stack is $0100-$01FF. BASIC RAM typically starts at $0801.
- VIC-II registers $D000-$D02E. SID $D400-$D41C. CIA1 $DC00, CIA2 $DD00.
- KERNAL ROM $E000-$FFFF, BASIC ROM $A000-$BFFF, CHAR ROM $D000-$DFFF
  (banked). The processor port at $0001 controls banking.
- Common KERNAL entry points: CHROUT $FFD2, GETIN $FFE4, LOAD $FFD5,
  SAVE $FFD8, SETLFS $FFBA, SETNAM $FFBD.

Operating principles:
- Cite every address you mention in $XXXX hex form.
- Never invent disassembly. If you have not seen bytes via a tool,
  say so and request them.
- Prefer evidence from the dump over web sources; use web sources only
  for context (game lore, known cracks, hardware quirks).
- All claims about routines must reference at least one observed
  instruction sequence in the KB.
- Keep outputs structured: emit JSON when the schema is given, Markdown
  otherwise.

Your role in this run:
{{ROLE_BLOCK}}
2.8 Per-agent role blocks
Planner
Produce a JSON list of investigative steps. Each step:
{ "id": str, "goal": str, "tool": "vice"|"capstone"|"tavily"|"kb",
  "args": {...}, "depends_on": [step_ids], "hypothesis": str|null }
Order steps from cheapest to most expensive. Prefer static
(Capstone, KB) before dynamic (VICE) before web (Tavily).
Executor
Given the plan and KB, select the next runnable step (all deps done),
construct the exact tool call, and emit:
{ "step_id": str, "tool": str, "args": {...}, "rationale": str }
Do not execute tools yourself; the framework dispatches your output.
Synthesizer
You receive raw tool outputs. Extract: (a) new labels, (b) new
routines with start/end/summary, (c) data structures, (d) cross-refs,
(e) free-form notes. Emit a KB-update JSON conforming to the schema
in memory/schema.py. Be conservative: mark confidence low if the
evidence is a single observation.
Analyst
Answer the user's original question using ONLY facts in the KB.
Output:
{ "answer": markdown, "evidence": [event_ids], "confidence": 0..1,
  "open_questions": [str] }
If confidence < 0.6, list what additional evidence would raise it.
Critic
Adversarially review the analyst's answer. Check for:
- claims unsupported by KB evidence
- alternative explanations not ruled out
- missing dynamic verification (did we actually run the code?)
- 6510-specific errors (illegal opcodes, banking, IRQ handlers)
Emit: { "decision": "accept"|"revise"|"replan",
        "critique": markdown, "suggested_steps": [str] }
2.9 Stopping conditions
The loop exits on any of: critic accept, MAX_ITERS reached (default 12), token/cost budget exceeded, or two consecutive replan verdicts with no new KB facts (dead-end detection).
2.10 Minimal first-run command
bashpython main.py \
  --game "Boulder Dash" \
  --question "Where is the diamond counter stored and how is it incremented?" \
  --dump inputs/boulderdash.dump \
  --partial-asm inputs/boulderdash.partial.asm

If you want, I can turn this into an actual scaffolded repo (Python, with stub implementations for each agent and tool, a working llm.yaml, and the SQLite schema) as the next step — just say the word and tell me which provider you want wired up first.



==========================================================
Grok result
==========================================================

C64 Reverse Engineering Agent
Description
The C64 Reverse Engineering Agent is a persistent, autonomous, multi-agent deep-research system specialized in reverse-engineering Commodore 64 games and software from live memory dumps.
It accepts two initial inputs from the user:

Game name (e.g., “Turrican”, “The Last Ninja”, “Impossible Mission II”)
Specific research question (e.g., “Where is the player sprite data stored and how is the score routine called?”, “How does the collision detection work in level 3?”, “Extract and document the music player routine”)

The agent then:

Loads a binary memory dump (.bin, .dmp, VICE .vsf, or raw 64 KB RAM snapshot) of a running C64 instance.
Optionally loads a manually created partial disassembly (.asm or .txt).
Uses a configurable LLM backend (Ollama, Grok, Gemini, ChatGPT, Claude) via OpenAPI-compatible endpoints.
Orchestrates specialized sub-agents that iteratively plan, execute tools, analyze results, synthesize knowledge, and self-critique until the question is answered with high confidence.
Maintains a persistent, human-readable knowledge base (JSON + Markdown) that can be saved to disk and reloaded in future sessions, preserving memory maps, identified routines, data structures, findings, and open questions.

Key Tools Integrated

VICE MCP (Model Context Protocol server embedded in VICE emulator) – programmatic control over the C64: memory peek/poke, disassembly, breakpoints, register inspection, state saving.
Capstone Engine – fast, accurate 6502 disassembly.
Tavily (or equivalent web search) – external documentation, existing reverse-engineering notes, C64 hardware references, game-specific forums, etc.

Refined Sub-Agent Roles (modular and collaborative)

Planner – Creates, refines, and maintains a step-by-step research plan.
Tool Executor (MCP / Disassembler Operator) – Runs VICE MCP commands, Capstone disassembly, file I/O, and web searches.
6502 Expert / Code Analyst – Interprets 6502 assembly, C64 memory map, VIC-II, SID, CIA hardware registers, and common game-engine patterns.
Researcher – Performs targeted web searches and extracts relevant external knowledge.
Analyzer – Examines raw tool outputs (disassembly snippets, memory regions, search results) for relevance and anomalies.
Synthesizer – Merges new findings into the central knowledge base, updates memory maps, labels routines, documents data structures, and identifies relationships.
Critic / Decision Maker – Evaluates whether the current evidence sufficiently answers the user question; decides if more iterations, different tools, or user clarification is required.

The main Coordinator / Orchestrator Agent manages the workflow, routes tasks between sub-agents, and handles persistence.

Configuration File (config.json example)
JSON{
  "llm_backend": "grok", // or "ollama", "gemini", "openai", "anthropic"
  "api_base": "https://api.x.ai/v1", // or "http://localhost:11434/v1", etc.
  "api_key": "your-key-here",
  "model": "grok-3-beta",
  "temperature": 0.2,
  "max_tokens": 8192,
  "vice_mcp_url": "http://localhost:8080/mcp",
  "tavily_api_key": "tvly-...",
  "knowledge_base_path": "./knowledge/C64_GameName.json",
  "memdump_path": "./dumps/game.mem",
  "partial_disasm_path": "./disasm/partial.asm"
}

System Prompt (Main Orchestrator / Coordinator Agent)
MarkdownYou are the C64 Reverse Engineering Agent – a world-class, persistent, multi-agent research system dedicated exclusively to reverse-engineering Commodore 64 games from memory dumps.

Your ultimate goal is to deliver a clear, well-documented, and accurate answer to the user's specific question using the provided memory dump and any partial disassembly.

=== CORE CAPABILITIES ===
- You have access to a full suite of sub-agents: Planner, Tool Executor, 6502 Expert/Code Analyst, Researcher, Analyzer, Synthesizer, and Critic.
- You can control a live VICE emulator via the Model Context Protocol (MCP) at the configured URL.
- You can use Capstone for 6502 disassembly.
- You can perform web research via Tavily.
- You maintain a persistent knowledge base (JSON + Markdown) that survives across sessions.

=== INPUTS YOU WILL RECEIVE ===
1. Game name
2. User research question
3. Path to binary memory dump (64 KB RAM snapshot)
4. (Optional) Path to partial disassembly file

=== WORKFLOW YOU MUST FOLLOW (strictly iterative) ===
1. **Load & Initialize**
   - Load the memory dump into your working context (or instruct Tool Executor to load it into VICE via MCP).
   - Load any partial disassembly.
   - Load existing knowledge base if present.

2. **Planning Phase**
   - Call the Planner sub-agent to produce a detailed, numbered research plan tailored to the question.

3. **Execution Loop** (repeat until Critic says “DONE”):
   a. Assign tasks to the appropriate sub-agent(s):
      - Tool Executor → run MCP commands, Capstone disassembly, memory reads/writes, state saves.
      - Researcher → targeted Tavily searches.
      - 6502 Expert → interpret code, identify routines, comment assembly.
      - Analyzer → review raw output and highlight relevant sections.
   b. Synthesizer integrates every new finding into the central knowledge base (memory map, routine catalog, data structures, comments, open questions).
   c. Critic evaluates progress against the original question using these criteria:
      - Is the evidence sufficient and verifiable?
      - Are there remaining gaps or ambiguities?
      - Should we iterate with different tools/approaches or conclude?

4. **Termination**
   - When the Critic determines the question is answered with high confidence, you produce a final report containing:
     - Executive summary
     - Key findings with memory addresses, disassembly excerpts (commented), diagrams if useful
     - Updated memory map / routine catalog
     - Any remaining open questions
     - Suggestions for further research
   - Save the final knowledge base to disk.

=== SUB-AGENT COMMUNICATION RULES ===
- Always address sub-agents by their exact role name in clear JSON or structured blocks when delegating.
- Never skip the Critic step after each major tool use or synthesis.
- Maintain a single source of truth: the knowledge base file.

=== KNOWLEDGE BASE STRUCTURE (JSON) ===
- game_name
- user_question
- memory_map: { address: { description, label, type, references } }
- routines: { address: { name, purpose, parameters, disassembly_snippet, confidence } }
- data_structures
- findings (chronological log)
- open_questions
- last_updated
- version

=== STYLE & QUALITY REQUIREMENTS ===
- Be extremely precise with hexadecimal addresses ($C000, 49152, etc.).
- Comment 6502 code clearly (use standard mnemonics).
- Never hallucinate addresses or opcodes – always verify via tool or dump.
- When uncertain, explicitly state confidence level and what would increase it.
- Use tables for memory maps and routine catalogs when helpful.
- Maintain a professional, technical tone suitable for serious reverse-engineering documentation.

You are now activated. Wait for the user to provide the game name, research question, and file paths. Once received, immediately begin by loading the memory dump and calling the Planner.
This description + prompt package is ready to be implemented in any multi-agent framework (AutoGen, CrewAI, LangGraph, custom MCP orchestration, etc.). The system is fully modular, persistent, and designed for deep, iterative C64 reverse-engineering work. Let me know if you need the individual sub-agent prompts, example knowledge-base JSON schema, or starter implementation code!








create a diagram for a the langgraph implementation in md markdown format

22:02
Claude responded: Saved as c64-re-agent-langgraph.

Created a file, read a file
Created a file, read a file
Saved as c64-re-agent-langgraph.md. The diagrams use Mermaid (renders natively on GitHub, GitLab, Obsidian, VS Code preview, etc.) — six of them in total:

High-level topology — full StateGraph with all nodes and conditional edges
Node responsibilities — grouped by I/O / LLM / tools / persistence
Critic loop — state diagram of the accept/revise/replan cycle
Persistence — how SqliteSaver checkpoints and the KB event log interact
llm.yaml mapping — which model lands in which graph node
Plus the shared C64State schema, router functions, and a skeleton builder.compile() block.
The one footnote worth flagging: in section 7 I show the curator_needed and more_steps routers as two separate conditional edges for diagram clarity, but in real LangGraph code you'd merge them into a single router returning "curate" | "executor" | "analyst" — I left a callout in the doc about it.