"""Wire the C64-RE StateGraph and expose a compiled `graph`.

The compiled graph is the LangGraph Studio entrypoint (referenced from
`langgraph.json`). When `langgraph dev` runs, Studio injects its own
checkpointer, so we deliberately compile without one here.

Topology follows `CLAUDE_graph.md` §7, with the curator-gate and
more-steps gate merged into a single post-synthesizer router (per the
footnote in §7).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.code_kb_node import code_kb_node
from graph.nodes import (
    analyst_node,
    capstone_node,
    critic_node,
    curator_node,
    executor_node,
    kb_query_node,
    load_inputs,
    planner_node,
    synthesizer_node,
    tavily_node,
    vice_mcp_node,
    write_report,
)
from graph.routers import (
    post_curator_router,
    post_synth_router,
    route_tool,
    verdict_router,
)
from graph.state import C64State


def build_graph() -> StateGraph:
    builder = StateGraph(C64State)

    # ---- nodes ----
    builder.add_node("load_inputs", load_inputs)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("vice", vice_mcp_node)
    builder.add_node("capstone", capstone_node)
    builder.add_node("tavily", tavily_node)
    builder.add_node("kb", kb_query_node)
    builder.add_node("code_kb", code_kb_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("curator", curator_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("critic", critic_node)
    builder.add_node("write_report", write_report)

    # ---- linear segment ----
    builder.add_edge(START, "load_inputs")
    builder.add_edge("load_inputs", "planner")
    builder.add_edge("planner", "executor")

    # ---- tool dispatch fan-out ----
    # Two no-step paths (tracker 0.6): a blocked plan (unsatisfiable
    # dependencies — the executor records a structured failure per
    # blocked step) routes deterministically to "planner" for a replan,
    # bounded by MAX_ITERS; a completed plan routes to "synthesizer".
    builder.add_conditional_edges(
        "executor",
        route_tool,
        {
            "vice": "vice",
            "capstone": "capstone",
            "tavily": "tavily",
            "kb": "kb",
            "code_kb": "code_kb",
            "synthesizer": "synthesizer",
            "planner": "planner",
        },
    )
    for tool in ("vice", "capstone", "tavily", "kb", "code_kb"):
        builder.add_edge(tool, "synthesizer")

    # ---- merged post-synth gate (curate / loop / analyze) ----
    builder.add_conditional_edges(
        "synthesizer",
        post_synth_router,
        {
            "curate": "curator",
            "executor": "executor",
            "analyst": "analyst",
        },
    )
    builder.add_conditional_edges(
        "curator",
        post_curator_router,
        {
            "executor": "executor",
            "analyst": "analyst",
        },
    )

    # ---- outer critic loop ----
    builder.add_edge("analyst", "critic")
    builder.add_conditional_edges(
        "critic",
        verdict_router,
        {
            "accept": "write_report",
            "revise": "analyst",
            "replan": "planner",
            "budget_exceeded": "write_report",
        },
    )
    builder.add_edge("write_report", END)

    return builder


# Compiled graph — this name is referenced from `langgraph.json`.
graph = build_graph().compile()
graph.name = "c64re_agent"
