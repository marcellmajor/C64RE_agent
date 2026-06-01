"""C64 layered code-comprehension knowledge base.

A separate, additive knowledge store that sits next to the parent
agent's `memory.KnowledgeStore`. Built around the v1 spec from
`CLAUDE.md`'s code-comprehension extension:

    Layer 0 — deterministic ground truth (no LLM)
    Layer 1 — per-routine semantic annotation (LLM, narrow window)
    Layer 2 — global behavioural grouping  (LLM, deferred to v2)
    Layer 3 — adversarial critique         (LLM, deferred to v2)

Public re-exports keep the call sites short:

    from code_kb import (
        CodeKnowledgeStore, get_code_store, Annotation,
        build_from_parsed_asm, build_from_disasm_window,
        disasm_capstone, disasm_vice,
        DEEP_RETRO_RE_PREAMBLE,
    )
"""

from code_kb.layer0 import (
    Layer0Stats,
    build_from_disasm_window,
    build_from_parsed_asm,
)
from code_kb.layer1 import (
    DEEP_RETRO_RE_PREAMBLE,
    RoutineWindow,
    annotation_from_layer1_json,
    build_window,
    fetch_routines,
    render_user_prompt,
)
from code_kb.store import Annotation, CodeKnowledgeStore, get_code_store
from code_kb.disasm import disasm_capstone, disasm_vice
from code_kb.exporters import export_commented_asm
from code_kb.call_graph import local_dot, routines_dot
from code_kb.scoping import (
    Scoping,
    candidate_dumps,
    game_tokens,
    select_asm_files,
)

__all__ = [
    "Annotation",
    "CodeKnowledgeStore",
    "DEEP_RETRO_RE_PREAMBLE",
    "Layer0Stats",
    "RoutineWindow",
    "Scoping",
    "annotation_from_layer1_json",
    "build_from_disasm_window",
    "build_from_parsed_asm",
    "build_window",
    "candidate_dumps",
    "disasm_capstone",
    "disasm_vice",
    "export_commented_asm",
    "fetch_routines",
    "game_tokens",
    "get_code_store",
    "local_dot",
    "render_user_prompt",
    "routines_dot",
    "select_asm_files",
]
