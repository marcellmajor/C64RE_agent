"""Deterministic question-relevant ASM evidence in the shared digest."""

from code_kb import get_code_store
from graph.nodes import _code_kb_digest_for_question


def test_code_digest_surfaces_explicit_lives_evidence(tmp_path):
    root = tmp_path / "code-kb"
    asm = tmp_path / "game.asm"
    content = """
090C: A9 03       LDA #$03         ; 3 lives
090E: 85 8B       STA $8B          ; Store in zero-page lives counter
113E: C6 8B       DEC $8B          ; Lives -= 1
""".strip()
    asm.write_text(content)
    store = get_code_store(root)
    store.ingest_asm_doc(asm, content=content, instructions=3, routines=1)

    digest = _code_kb_digest_for_question(
        str(root), "Where is the player's lives counter stored?",
    )

    assert "partial-assembly excerpts" in digest
    assert "STA $8B" in digest
    assert "DEC $8B" in digest
    assert "lives counter" in digest


def test_code_digest_keeps_compact_selected_asm_without_lexical_overlap(tmp_path):
    root = tmp_path / "code-kb"
    asm = tmp_path / "prng.asm"
    content = """
1D1B  86 A1       STX $A1
1D30  20 1B 1D    JSR $1D1B
1D33  48          PHA
1D34  20 1B 1D    JSR $1D1B
1D39  60          RTS
""".strip()
    asm.write_text(content)
    store = get_code_store(root)
    store.ingest_asm_doc(asm, content=content, instructions=5, routines=2)

    digest = _code_kb_digest_for_question(
        str(root), "Which helper produces a 16-bit random value?",
    )

    assert "primary static evidence" in digest
    assert "1D30  20 1B 1D" in digest
    assert digest.count("1D30  20 1B 1D") == 1
