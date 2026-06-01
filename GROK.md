# C64 Reverse Engineering Agent

**Description**

The **C64 Reverse Engineering Agent** is a persistent, autonomous, multi-agent deep-research system specialized in reverse-engineering Commodore 64 games and software from live memory dumps.

It accepts two initial inputs from the user:
1. **Game name** (e.g., “Turrican”, “The Last Ninja”, “Impossible Mission II”)
2. **Specific research question** (e.g., “Where is the player sprite data stored and how is the score routine called?”, “How does the collision detection work in level 3?”, “Extract and document the music player routine”)

The agent then:
- Loads a binary memory dump (`.bin`, `.dmp`, VICE `.vsf`, or raw 64 KB RAM snapshot) of a running C64 instance.
- Optionally loads a manually created partial disassembly (`.asm` or `.txt`).
- Uses a configurable LLM backend (Ollama, Grok, Gemini, ChatGPT, Claude) via OpenAPI-compatible endpoints.
- Orchestrates specialized sub-agents that iteratively plan, execute tools, analyze results, synthesize knowledge, and self-critique until the question is answered with high confidence.
- Maintains a persistent, human-readable knowledge base (JSON + Markdown) that can be saved to disk and reloaded in future sessions, preserving memory maps, identified routines, data structures, findings, and open questions.

**Key Tools Integrated**
- **VICE MCP** (Model Context Protocol server embedded in VICE emulator) – programmatic control over the C64: memory peek/poke, disassembly, breakpoints, register inspection, state saving.
- **Capstone Engine** – fast, accurate 6502 disassembly.
- **Tavily** (or equivalent web search) – external documentation, existing reverse-engineering notes, C64 hardware references, game-specific forums, etc.

**Refined Sub-Agent Roles** (modular and collaborative)
1. **Planner** – Creates, refines, and maintains a step-by-step research plan.
2. **Tool Executor (MCP / Disassembler Operator)** – Runs VICE MCP commands, Capstone disassembly, file I/O, and web searches.
3. **6502 Expert / Code Analyst** – Interprets 6502 assembly, C64 memory map, VIC-II, SID, CIA hardware registers, and common game-engine patterns.
4. **Researcher** – Performs targeted web searches and extracts relevant external knowledge.
5. **Analyzer** – Examines raw tool outputs (disassembly snippets, memory regions, search results) for relevance and anomalies.
6. **Synthesizer** – Merges new findings into the central knowledge base, updates memory maps, labels routines, documents data structures, and identifies relationships.
7. **Critic / Decision Maker** – Evaluates whether the current evidence sufficiently answers the user question; decides if more iterations, different tools, or user clarification is required.

The **main Coordinator / Orchestrator Agent** manages the workflow, routes tasks between sub-agents, and handles persistence.

---

**Configuration File** (`config.json` example)

```json
{
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
