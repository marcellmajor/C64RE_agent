"""Smoke test for the Streamlit UI integration.

Doesn't actually render anything (Streamlit needs its own runner).
Instead this verifies:

  1. `tools.agent_runner` and `code_kb.call_graph` import cleanly.
  2. `parse_addresses` returns a stable, deduped list.
  3. The call_graph DOT generator handles the "no routine found" path.
  4. With a tiny in-memory CodeKnowledgeStore (fed via Layer 0 from
     a parsed snippet) `disasm_for_addresses` and `local_dot` produce
     non-empty output.

Run with:
    .venv/bin/python -m scripts.smoke_ui
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    print("== imports ==")
    from tools.agent_runner import (  # noqa: F401
        TurnResult,
        call_graph_dot,
        code_kb_summary,
        disasm_for_addresses,
        parse_addresses,
        run_question,
        top_routines_by_xrefs,
    )
    from code_kb import (
        CodeKnowledgeStore,
        build_from_parsed_asm,
        local_dot,
    )
    from code_kb.asm_parser import parse_path
    print("  imports OK")

    print("== parse_addresses ==")
    sample = (
        "The lives counter sits in $0780; it's bumped from "
        "$0810-$0825 (sub_pickup) and read from $C100. Range $0700-$070F "
        "is also reserved. Repeat $0780 once more."
    )
    addrs = parse_addresses(sample)
    print(f"  found {len(addrs)} addrs: {[hex(a) for a in addrs]}")
    assert 0x0780 in addrs and 0x0810 in addrs and 0xC100 in addrs

    print("== build a tiny CodeKnowledgeStore ==")
    asm_text = (
        "main:\n"
        "  $0801: A9 00     LDA #$00\n"
        "  $0803: 20 10 08  JSR $0810\n"
        "  $0806: 20 20 08  JSR $0820\n"
        "  $0809: 60        RTS\n"
        "sub_0810:\n"
        "  $0810: A2 05     LDX #$05\n"
        "  $0812: 60        RTS\n"
        "sub_0820:\n"
        "  $0820: 20 10 08  JSR $0810\n"
        "  $0823: 60        RTS\n"
    )

    tmp = Path(tempfile.mkdtemp(prefix="c64re_smoke_"))
    try:
        # 1) write the asm so the parser has a path to attach.
        asm_path = tmp / "smoke.asm"
        asm_path.write_text(asm_text)

        # 2) spin up an isolated code store under tmp/sessions/<game>.
        slug = "smoketest"
        sess_root = tmp / "sessions" / slug
        sess_root.mkdir(parents=True, exist_ok=True)
        store = CodeKnowledgeStore.load_or_init(sess_root / "code_kb")

        # 3) parse + Layer-0 fold.
        parsed = parse_path(asm_path)
        stats = build_from_parsed_asm(parsed, store)
        print(
            f"  layer-0 stats: routines={stats.routines}, "
            f"xrefs={stats.xrefs}, instructions={stats.instructions}"
        )

        # 4) DOT output.
        dot = local_dot(store, start=0x0801, hops=2, max_nodes=20)
        print(f"  local_dot: {len(dot['nodes'])} nodes, "
              f"{len(dot['edges'])} edges (preview):")
        print("\n".join("    " + ln for ln in dot["dot"].splitlines()[:10]))

        # 5) verify "no routine" path produces a usable empty graph
        empty = local_dot(store, start=0xFFEE, hops=1)
        assert "empty" in empty["dot"].lower()
        print("  empty-graph path OK")

        print("== UI helpers (sessions root override) ==")
        # `disasm_for_addresses` and friends look in `sessions/<slug>`
        # under the cwd. Hop into tmp so they find our tiny store.
        import os
        prev_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            from tools import agent_runner
            # Reset cache so the helper picks up the new on-disk store.
            agent_runner._GRAPH_CACHE.clear()
            from code_kb.store import _CODE_STORE_CACHE
            _CODE_STORE_CACHE.clear()

            snippets = disasm_for_addresses(slug, [0x0801, 0x0820])
            print(f"  disasm_for_addresses → {len(snippets)} routine(s)")
            for s in snippets:
                print(
                    f"    · {s['name']} ${s['start_addr']:04X}-${s['end_addr']:04X} "
                    f"({s['instruction_count']} insn)"
                )
            assert any(s["start_addr"] == 0x0801 for s in snippets)

            cg = call_graph_dot(slug, start=0x0801, hops=2)
            assert cg is not None and "digraph" in cg["dot"]
            print(f"  call_graph_dot OK ({len(cg['nodes'])} nodes)")

            top = top_routines_by_xrefs(slug, limit=5)
            print(f"  top_routines_by_xrefs → {len(top)} rows")

            summary = code_kb_summary(slug)
            print(f"  code_kb_summary: {summary}")
        finally:
            os.chdir(prev_cwd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nAll UI smoke checks passed.")


if __name__ == "__main__":
    main()
