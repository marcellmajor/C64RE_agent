# C64 Reverse Engineering Agent – LangGraph Implementation Diagram

**LangGraph Architecture Overview**

The C64 Reverse Engineering Agent is implemented as a **LangGraph StateGraph** with:
- A single shared **State** object (containing the knowledge base, current plan, tool results, memory map, etc.).
- A **Coordinator / Supervisor** node that routes tasks to specialized sub-agents.
- Conditional edges driven by the **Critic** node (the main loop controller).
- Persistence via LangGraph’s built-in checkpointing (JSON knowledge base saved/loaded automatically).

```mermaid
flowchart TD
    %% Main flow
    Start[Start] 
        --> UserInput[User Input:\n• Game Name\n• Research Question\n• Memory Dump Path\n• Optional Partial Disasm]

    UserInput 
        --> Initialize[Initialize Node\n• Load 64KB memory dump\n• Load partial disassembly\n• Load / create Knowledge Base JSON]

    Initialize 
        --> Coordinator[Coordinator / Orchestrator\nSupervisor Agent]

    %% Sub-agent routing from Coordinator
    Coordinator 
        --> Planner[Planner Agent\nCreate / refine step-by-step research plan]

    Planner 
        --> Coordinator

    %% Tool & Research branches
    Coordinator 
        --> ToolExecutor[Tool Executor Agent\n• VICE MCP commands\n• Capstone 6502 disassembly\n• File I/O]

    Coordinator 
        --> Researcher[Researcher Agent\nTavily web search\nC64 docs & existing RE notes]

    Coordinator 
        --> Expert[6502 Expert / Code Analyst\nInterpret assembly\nIdentify routines & data structures]

    %% Analysis & Synthesis chain
    ToolExecutor --> Analyzer[Analyzer Agent\nReview raw tool output\nHighlight relevant sections & anomalies]
    Researcher --> Analyzer
    Expert --> Analyzer

    Analyzer 
        --> Synthesizer[Synthesizer Agent\nMerge findings into KB\nUpdate memory map, routines, data structures]

    Synthesizer 
        --> Critic[Critic / Decision Maker Agent\nEvaluate evidence vs. question\nConfidence check]

    %% Loop control
    Critic 
        -->|More Research Needed| Coordinator

    Critic 
        -->|Question Answered with High Confidence| FinalReport[Final Report Node\n• Executive Summary\n• Commented disassembly\n• Updated Memory Map\n• Routine Catalog\n• Open Questions]

    FinalReport 
        --> SaveKB[Save Knowledge Base\nJSON + Markdown]

    SaveKB 
        --> End[End]

    %% Subgraph for clarity
    subgraph LangGraph_State ["Shared LangGraph State (Checkpointed)"]
        KB[Knowledge Base JSON\n• memory_map\n• routines\n• findings\n• open_questions]
        Plan[Current Research Plan]
        Results[Latest Tool Results]
    end

    %% Connect state to key nodes
    Initialize -.-> KB
    Synthesizer -.-> KB
    SaveKB -.-> KB
    Planner -.-> Plan
    ToolExecutor -.-> Results
    Researcher -.-> Results
    Expert -.-> Results

    classDef coordinator fill:#4a90e2,stroke:#fff,stroke-width:2px,color:#fff;
    classDef subagent fill:#50c878,stroke:#fff,stroke-width:2px,color:#fff;
    classDef decision fill:#f39c12,stroke:#fff,stroke-width:2px,color:#fff;
    classDef final fill:#e74c3c,stroke:#fff,stroke-width:2px,color:#fff;

    class Coordinator coordinator;
    class Planner,ToolExecutor,Researcher,Expert,Analyzer,Synthesizer subagent;
    class Critic decision;
    class FinalReport,SaveKB,End final;
