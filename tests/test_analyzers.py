"""Deterministic static analyzers (tracker 3.3 / 3.5 / 3.6 / 3.7)."""

import tools.c64_disasm as d


def _stream(pairs):
    """Build an int-keyed insns dict from (addr, mnemonic, op_str) tuples."""
    return {a: {"mnemonic": m, "op_str": o} for a, m, o in pairs}


# ---------------------------------------------------------------------------
# 3.3 find_counters
# ---------------------------------------------------------------------------

def test_find_counters_lives_and_score_and_hud():
    insns = _stream([
        (0xC000, "dec", "0x0780"),          # lives (DEC RAM)
        (0xC003, "sed", ""),
        (0xC004, "lda", "0x0900"),
        (0xC007, "adc", "0x0901"),
        (0xC00A, "sta", "0x0900"),          # score inside SED…CLD
        (0xC00D, "cld", ""),
        (0xC00E, "ora", "#0x30"),           # digit
        (0xC010, "sta", "0x0400"),          # HUD digit → screen RAM
    ])
    cands = d.find_counters(insns)
    kinds = {(c["address"], c["kind"]) for c in cands}
    assert ("$0780", "lives") in kinds
    assert ("$0900", "score") in kinds
    assert ("$0400", "hud_digit") in kinds
    # DEC weighted higher for lives than the timer alias.
    lives = next(c for c in cands if c["kind"] == "lives")
    assert lives["score"] >= 12.0


def test_find_counters_multibyte_rollover():
    insns = _stream([
        (0xC000, "inc", "0x0820"),
        (0xC003, "cmp", "#0x0a"),           # decimal rollover
    ])
    cands = d.find_counters(insns)
    assert any(c["kind"] == "multibyte_counter" and c["address"] == "$0820"
               for c in cands)


def test_find_counters_ignores_io_and_immediates():
    insns = _stream([
        (0xC000, "dec", "0xd020"),          # I/O register, not a RAM var
        (0xC003, "lda", "#0x05"),           # immediate, no memory operand
    ])
    assert d.find_counters(insns) == []


# ---------------------------------------------------------------------------
# 3.5 find_idioms
# ---------------------------------------------------------------------------

def test_find_idioms_raster_and_trampoline_and_delay():
    insns = _stream([
        (0xC000, "lda", "0xd012"),
        (0xC003, "cmp", "#0x80"),
        (0xC005, "bne", "0xc000"),          # raster wait
        (0xC007, "dex", ""),
        (0xC008, "bne", "0xc007"),          # delay loop
        (0xC00A, "jmp", "0xffd2"),          # KERNAL trampoline
    ])
    got = {(i["address"], i["idiom"]) for i in d.find_idioms(insns)}
    assert ("$C000", "raster_wait") in got
    assert ("$C007", "delay_loop") in got
    assert ("$C00A", "kernal_trampoline") in got


def test_find_idioms_memcpy_and_sid_tick():
    insns = _stream([
        (0xC000, "lda", "0x2000, x"),
        (0xC003, "sta", "0x0400, x"),
        (0xC006, "inx", ""),
        (0xC007, "bne", "0xc000"),          # memcpy
    ])
    got = {(i["address"], i["idiom"]) for i in d.find_idioms(insns)}
    assert ("$C000", "memcpy") in got

    sid = _stream([
        (0xC100, "sta", "0xd400"),
        (0xC103, "sta", "0xd401"),
        (0xC106, "sta", "0xd404"),          # 3 SID stores → tick
    ])
    assert any(i["idiom"] == "sid_tick" for i in d.find_idioms(sid))


def test_find_idioms_jump_table():
    insns = _stream([
        (0xC000, "asl", ""),
        (0xC001, "tax", ""),
        (0xC002, "lda", "0xc020, x"),       # table lookup
        (0xC005, "sta", "0x02"),
        (0xC007, "jmp", "(0x0002)"),        # indirect dispatch
    ])
    got = {(i["address"], i["idiom"]) for i in d.find_idioms(insns)}
    assert ("$C000", "jump_table") in got


# ---------------------------------------------------------------------------
# 3.4 extract_data_refs (pure)
# ---------------------------------------------------------------------------

def test_extract_data_refs_access_and_index():
    insns = _stream([
        (0xC000, "sta", "0xd020"),          # write
        (0xC003, "lda", "0x0780"),          # read
        (0xC006, "inc", "0x0781"),          # rmw
        (0xC009, "lda", "0x2000, x"),       # indexed read
        (0xC00C, "lda", "#0x05"),           # immediate — skipped
    ])
    refs = d.extract_data_refs(insns)
    idx = {(r["src"], r["dst"]): r for r in refs}
    assert idx[(0xC000, 0xD020)]["access"] == "w"
    assert idx[(0xC003, 0x0780)]["access"] == "r"
    assert idx[(0xC006, 0x0781)]["access"] == "rmw"
    assert idx[(0xC009, 0x2000)]["index"] == "x"
    assert all(r["src"] != 0xC00C for r in refs)   # immediate excluded


# ---------------------------------------------------------------------------
# 3.7 decode_screen_ram
# ---------------------------------------------------------------------------

def test_decode_screen_ram_finds_text():
    mem = bytearray(0x10000)
    # "SCORE" at $0400: S=19 C=3 O=15 R=18 E=5
    for i, c in enumerate([19, 3, 15, 18, 5]):
        mem[0x0400 + i] = c
    res = d.decode_screen_ram(bytes(mem))
    assert res["runs"], "expected at least one text run"
    assert res["runs"][0]["text"] == "SCORE"
    assert res["runs"][0]["addr_hex"] == "$0400"


def test_decode_screen_ram_zero_padding_is_not_noise():
    # An all-zero screen must not decode to "@@@@…" runs.
    res = d.decode_screen_ram(bytes(0x10000))
    assert res["runs"] == []


# ---------------------------------------------------------------------------
# 3.6 banking_state + bank-aware find_loops
# ---------------------------------------------------------------------------

def test_banking_state_reads_processor_port():
    mem = bytearray(0x10000)
    mem[0x0001] = 0x37                       # default: all ROM in
    b = d.banking_state(bytes(mem))
    assert b["kernal_rom_visible"] and not b["ram_under_kernal"]

    mem[0x0001] = 0x35                        # HIRAM off → RAM under KERNAL
    b = d.banking_state(bytes(mem))
    assert b["ram_under_kernal"] and not b["kernal_rom_visible"]


def test_find_loops_penalizes_e000_only_when_rom_banked_in():
    # A tight IRQ-style loop at $E000 closed by JMP back to itself.
    def build(port):
        mem = bytearray(0x10000)
        mem[0x0001] = port
        # $E000: LDA $D012 / CMP #$80 / BNE $E000 / JMP $E000
        prog = bytes([0xAD, 0x12, 0xD0, 0xC9, 0x80, 0xD0, 0xF9,
                      0x4C, 0x00, 0xE0])
        mem[0xE000:0xE000 + len(prog)] = prog
        # Point the RAM IRQ vector at it so heuristic 1 surfaces it.
        mem[0x0314] = 0x00
        mem[0x0315] = 0xE0
        return bytes(mem)

    rom_in = {c["address"]: c["score"] for c in d.find_loops(build(0x37))}
    ram_under = {c["address"]: c["score"] for c in d.find_loops(build(0x35))}
    # With RAM under KERNAL the $E000 loop scores higher (no −30 penalty).
    if "$E000" in rom_in and "$E000" in ram_under:
        assert ram_under["$E000"] > rom_in["$E000"]
    else:
        # At minimum, RAM-under-ROM must surface it when ROM-in may not.
        assert "$E000" in ram_under


def test_find_loops_explains_main_loop_signals_and_distinct_calls():
    mem = bytearray(0x10000)
    mem[0x0001] = 0x37
    # $C000: two distinct calls, input read, VIC write, backward JMP.
    program = bytes([
        0x20, 0x00, 0xC1,       # JSR $C100
        0x20, 0x10, 0xC1,       # JSR $C110
        0xAD, 0x00, 0xDC,       # LDA $DC00
        0x8D, 0x20, 0xD0,       # STA $D020
        0x4C, 0x00, 0xC0,       # JMP $C000
    ])
    mem[0xC000:0xC000 + len(program)] = program

    candidate = next(
        row for row in d.find_loops(bytes(mem)) if row["address"] == "$C000"
    )

    assert candidate["candidate_type"] == "main_loop"
    assert candidate["source"] == "backward_jmp"
    assert candidate["metrics"]["jsr_count"] == 2
    assert candidate["metrics"]["distinct_jsr_targets"] == ["$C100", "$C110"]
    assert set(candidate["metrics"]["c64_io_signals"]) >= {"input", "vic"}


def test_find_loops_types_irq_vector_candidate_instead_of_calling_it_main_loop():
    mem = bytearray(0x10000)
    mem[0x0001] = 0x37
    mem[0x0314:0x0316] = bytes([0x00, 0xC2])
    program = bytes([
        0xAD, 0x12, 0xD0,       # LDA $D012
        0x8D, 0x19, 0xD0,       # STA $D019
        0xEA, 0xEA, 0xEA,       # NOP x3
        0x40,                   # RTI
    ])
    mem[0xC200:0xC200 + len(program)] = program

    candidate = next(
        row for row in d.find_loops(bytes(mem)) if row["address"] == "$C200"
    )

    assert candidate["candidate_type"] == "irq_handler"
    assert candidate["source"] == "ram_irq_vector"
    assert candidate["metrics"]["returns"] == ["RTI"]


def test_find_loops_discovers_backward_conditional_branch_candidate():
    mem = bytearray(0x10000)
    mem[0x0001] = 0x37
    program = bytes([
        0xAD, 0x12, 0xD0,       # $C300 LDA $D012
        0xC9, 0x80,             # $C303 CMP #$80
        0xEA, 0xEA, 0xEA,       # $C305-$C307 NOP
        0xD0, 0xF6,             # $C308 BNE $C300
    ])
    mem[0xC300:0xC300 + len(program)] = program

    candidate = next(
        row for row in d.find_loops(bytes(mem)) if row["address"] == "$C300"
    )

    assert candidate["source"] == "backward_branch"
    assert any("BNE $C300" in reason for reason in candidate["reasons"])


def test_recursive_disasm_skips_rom_vectors_when_rom_banked_in():
    mem = bytearray(0x10000)
    mem[0x0001] = 0x37                        # KERNAL ROM in
    # A trivial RTS program at $0810 (BASIC-area entry).
    mem[0x0810] = 0x60
    # IRQ vector $FFFE → $E043 (KERNAL). Should NOT be seeded.
    mem[0xFFFE] = 0x43
    mem[0xFFFF] = 0xE0
    mem[0xE043] = 0xEA                        # NOP in "ROM"
    rec = d.recursive_disasm(bytes(mem), 0x0810, max_insns=50)
    assert 0xE043 not in {int(k.lstrip("$"), 16) for k in rec["insns"]}
