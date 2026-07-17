"""Layer 0 — deterministic ground truth (no LLM).

Inputs: a `ParsedAsmFile` (from `code_kb.asm_parser`) and/or a raw memory
dump. Outputs: a stream of `Annotation` rows committed to the
`CodeKnowledgeStore` covering:

    * routines (`ANN_ROUTINE`) with start/end + entry-point set
    * cross-references (`ANN_XREF`) for every JSR / JMP / branch
    * indirect-jump sites (`ANN_INDIRECT`)
    * self-modifying-code suspects (`ANN_SMC`)
    * per-byte classification (`ANN_CLASSIFY`) — code | data | ambiguous
    * labels (`ANN_LABEL`) carried over from the partial asm
    * raw instruction rows (`ANN_DISASM`) so other layers can show context
      without re-reading the asm document.

Every Layer-0 annotation has `producer="deterministic"` and is treated as
immutable by the higher layers. The C64 caveats from the build spec —
multiple entry points per routine, fall-through, SMC, code/data
ambiguity — are encoded explicitly:

    - Routine boundaries are derived from explicit gap markers and from
      `sub_*`/`entry`-style labels. Fall-through across a `RTS`/`RTI`/
      unconditional `JMP` *closes* a routine; the next instruction starts
      a new one only if no fall-through edge connects them.
    - Multiple labels on the same address become *entry points* on the
      enclosing routine, not separate routines.
    - SMC sites are flagged whenever a store's destination address has an
      instruction parsed at it.
    - Classification is conservative: addresses with a parsed instruction
      are `code`; addresses inside a gap are `data`; the rest are
      `ambiguous`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from code_kb.asm_parser import (
    ParsedAsmFile,
    ParsedInstruction,
    operand_direct_target,
    operand_indirect_vector,
    operand_label_ref,
    operand_store_target,
)
from code_kb.schema import (
    ANN_CLASSIFY,
    ANN_DISASM,
    ANN_INDIRECT,
    ANN_LABEL,
    ANN_ROUTINE,
    ANN_SMC,
    ANN_XREF,
)
from code_kb.store import Annotation, CodeKnowledgeStore


BRANCH_MNEMONICS = frozenset({"bpl", "bmi", "bvc", "bvs",
                              "bcc", "bcs", "bne", "beq", "bra"})
JUMP_MNEMONICS = frozenset({"jmp"})
CALL_MNEMONICS = frozenset({"jsr"})
RETURN_MNEMONICS = frozenset({"rts", "rti"})
HALT_MNEMONICS = frozenset({"brk"})
STORE_MNEMONICS = frozenset({"sta", "stx", "sty", "stz"})


# --------------------------------------------------------------------------- #
# Helper structures
# --------------------------------------------------------------------------- #


@dataclass
class Layer0Stats:
    instructions: int = 0
    routines: int = 0
    xrefs: int = 0
    indirect_sites: int = 0
    smc_sites: int = 0
    labels: int = 0
    classified_bytes: int = 0
    sources: list[str] = None

    def __post_init__(self) -> None:
        if self.sources is None:
            self.sources = []

    def merge(self, other: "Layer0Stats") -> None:
        self.instructions += other.instructions
        self.routines += other.routines
        self.xrefs += other.xrefs
        self.indirect_sites += other.indirect_sites
        self.smc_sites += other.smc_sites
        self.labels += other.labels
        self.classified_bytes += other.classified_bytes
        self.sources.extend(other.sources or [])


# --------------------------------------------------------------------------- #
# Routine boundary detection
# --------------------------------------------------------------------------- #


def _is_routine_boundary_label(name: str) -> bool:
    n = name.lower()
    return n.startswith(("sub_", "entry"))


def _detect_routines(
    insns: list[ParsedInstruction],
    labels_by_addr: dict[int, list[str]],
    gaps: list[tuple[int, int]],
    jsr_targets: set[int] | None = None,
    label_to_addr: dict[str, int] | None = None,
) -> list[dict]:
    """Group instructions into routines using boundary heuristics.

    Boundary signals (any of these starts a new routine):
        1. The first parsed instruction (file head).
        2. The first instruction after an explicit gap.
        3. Any label whose name conventionally marks a routine
           (`sub_*`, `entry*`).
        4. Any address that is the target of a `JSR` somewhere in the
           file — a *called* address is by definition a routine entry.
        5. Any instruction immediately after an unconditional terminator
           (`RTS`/`RTI`/`JMP`/`BRK`) when the next instruction also has
           a label attached to it (rules out fall-through tails).
    """
    if not insns:
        return []

    sorted_insns = sorted(insns, key=lambda i: i.addr)
    addr_set = {i.addr for i in sorted_insns}
    addrs_sorted = sorted(addr_set)

    boundary_set: set[int] = {sorted_insns[0].addr}
    for addr, names in labels_by_addr.items():
        if any(_is_routine_boundary_label(n) for n in names):
            boundary_set.add(addr)
    for g_start, _g_end in gaps:
        # The instruction *after* a gap is a new routine start.
        for a in addrs_sorted:
            if a > _g_end:
                boundary_set.add(a)
                break

    if jsr_targets:
        for t in jsr_targets:
            if t in addr_set:
                boundary_set.add(t)

    # Rule 5: instruction after an unconditional terminator that ALSO has
    # a label (avoids splitting fall-through tails).
    UNCOND_TERMINATORS = RETURN_MNEMONICS | HALT_MNEMONICS | JUMP_MNEMONICS
    prev_was_terminator = False
    for ins in sorted_insns:
        if prev_was_terminator and ins.addr in labels_by_addr:
            boundary_set.add(ins.addr)
        prev_was_terminator = ins.mnemonic in UNCOND_TERMINATORS

    boundaries = sorted(b for b in boundary_set if b in addr_set)
    routines: list[dict] = []
    for i, start in enumerate(boundaries):
        if i + 1 < len(boundaries):
            next_boundary = boundaries[i + 1]
            range_insns = [ins for ins in sorted_insns
                           if start <= ins.addr < next_boundary]
        else:
            range_insns = [ins for ins in sorted_insns if ins.addr >= start]
        if not range_insns:
            continue
        last = range_insns[-1]
        end_addr = last.addr + last.size - 1
        # Entry points = every label inside the routine range
        entries = sorted({
            a for a in labels_by_addr
            if start <= a <= end_addr
        }) or [start]
        # Exits = addresses of RTS/RTI/JMP/BRK
        exits = [
            ins.addr for ins in range_insns
            if ins.mnemonic in RETURN_MNEMONICS
            or ins.mnemonic in HALT_MNEMONICS
            or ins.mnemonic in JUMP_MNEMONICS
        ]
        # Pick the routine "name" from the first label at `start`.
        name = None
        for n in labels_by_addr.get(start, []):
            if _is_routine_boundary_label(n):
                name = n
                break
        if name is None and start in labels_by_addr and labels_by_addr[start]:
            name = labels_by_addr[start][0]
        if name is None:
            name = f"sub_{start:04x}"
        routines.append({
            "start_addr": start,
            "end_addr": end_addr,
            "name": name,
            "entries": entries,
            "exits": exits,
        })
    return routines


# --------------------------------------------------------------------------- #
# Public Layer-0 builder
# --------------------------------------------------------------------------- #


def build_from_parsed_asm(
    parsed: ParsedAsmFile, store: CodeKnowledgeStore,
) -> Layer0Stats:
    """Run the deterministic Layer-0 pipeline on one parsed asm file."""
    stats = Layer0Stats(sources=[parsed.path])

    if not parsed.instructions:
        return stats

    with store.bulk_writes():
        return _build_from_parsed_asm_inner(parsed, store, stats)


def _build_from_parsed_asm_inner(
    parsed: ParsedAsmFile, store: CodeKnowledgeStore, stats: Layer0Stats,
) -> Layer0Stats:

    # ---- 1. Persist raw instructions as a single ANN_DISASM annotation
    #         (chunked so individual rows stay queryable but the event
    #         payload doesn't blow up the JSONL log).
    insn_rows = [
        {
            "addr": ins.addr,
            "bytes_hex": ins.bytes_.hex(),
            "mnemonic": ins.mnemonic,
            "operand": ins.operand,
            "size_bytes": ins.size,
        }
        for ins in parsed.instructions
    ]
    chunk_size = 1000
    for chunk_idx in range(0, len(insn_rows), chunk_size):
        chunk = insn_rows[chunk_idx : chunk_idx + chunk_size]
        first_addr = chunk[0]["addr"]
        last_addr = chunk[-1]["addr"]
        store.append_annotation(
            Annotation(
                layer=0, kind=ANN_DISASM,
                start_addr=first_addr, end_addr=last_addr,
                producer="deterministic", confidence=1.0,
                payload={
                    "source_file": parsed.path,
                    "instructions": chunk,
                },
            ),
            source="layer0.asm",
        )
        stats.instructions += len(chunk)

    # ---- 2. Carry over labels from the asm.
    labels_by_addr: dict[int, list[str]] = defaultdict(list)
    for lab in parsed.labels:
        labels_by_addr[lab.addr].append(lab.name)
    for addr, names in labels_by_addr.items():
        for n in names:
            store.append_annotation(
                Annotation(
                    layer=0, kind=ANN_LABEL,
                    start_addr=addr, end_addr=addr,
                    producer="deterministic", confidence=0.9,
                    payload={"name": n, "source_file": parsed.path},
                ),
                source="layer0.asm",
            )
            stats.labels += 1

    # ---- 3. Pre-compute the symbol & xref tables once.
    code_addrs = {ins.addr for ins in parsed.instructions}
    label_to_addr: dict[str, int] = {}
    for addr, names in labels_by_addr.items():
        for n in names:
            label_to_addr.setdefault(n, addr)

    jsr_targets: set[int] = set()
    for ins in parsed.instructions:
        if ins.mnemonic in CALL_MNEMONICS:
            t = operand_direct_target(ins.operand)
            if t is None:
                ref = operand_label_ref(ins.operand)
                if ref:
                    t = label_to_addr.get(ref)
            if t is not None and t in code_addrs:
                jsr_targets.add(t)

    # ---- 4. Routine detection (uses JSR targets to widen boundaries).
    gap_pairs = [(g.start, g.end) for g in parsed.gaps]
    routines = _detect_routines(
        parsed.instructions, labels_by_addr, gap_pairs,
        jsr_targets=jsr_targets, label_to_addr=label_to_addr,
    )
    for r in routines:
        store.append_annotation(
            Annotation(
                layer=0, kind=ANN_ROUTINE,
                start_addr=r["start_addr"], end_addr=r["end_addr"],
                producer="deterministic", confidence=0.9,
                payload={
                    "name": r["name"],
                    "source_file": parsed.path,
                    "entries": r["entries"],
                    "exits": r["exits"],
                    "size_bytes": r["end_addr"] - r["start_addr"] + 1,
                },
            ),
            source="layer0.asm",
        )
        stats.routines += 1

    # ---- 5. Xref + indirect + SMC extraction.

    for ins in parsed.instructions:
        # JSR / JMP / branch -> direct xref
        if ins.mnemonic in (CALL_MNEMONICS | JUMP_MNEMONICS | BRANCH_MNEMONICS):
            tgt = operand_direct_target(ins.operand)
            if tgt is None:
                # Try to resolve symbolic operand against label table.
                lab_ref = operand_label_ref(ins.operand)
                if lab_ref is not None:
                    tgt = label_to_addr.get(lab_ref)
            if tgt is not None:
                xkind = (
                    "jsr"     if ins.mnemonic in CALL_MNEMONICS
                    else "branch" if ins.mnemonic in BRANCH_MNEMONICS
                    else "jmp"
                )
                store.append_annotation(
                    Annotation(
                        layer=0, kind=ANN_XREF,
                        start_addr=ins.addr, end_addr=ins.addr,
                        producer="deterministic", confidence=1.0,
                        payload={
                            "src_addr": ins.addr,
                            "dst_addr": tgt,
                            "xref_kind": xkind,
                            "via_vector": None,
                            "source_file": parsed.path,
                        },
                    ),
                    source="layer0.asm",
                )
                stats.xrefs += 1
            else:
                # Indirect JMP via vector — capture the site.
                if ins.mnemonic == "jmp":
                    vec = operand_indirect_vector(ins.operand)
                    if vec is not None:
                        store.append_annotation(
                            Annotation(
                                layer=0, kind=ANN_INDIRECT,
                                start_addr=ins.addr, end_addr=ins.addr,
                                producer="deterministic", confidence=1.0,
                                payload={
                                    "src_addr": ins.addr,
                                    "via_vector": vec,
                                    "resolved_target": None,
                                    "source_file": parsed.path,
                                },
                            ),
                            source="layer0.asm",
                        )
                        stats.indirect_sites += 1

        # Stores into known code addresses → SMC suspect.
        if ins.mnemonic in STORE_MNEMONICS:
            dst = operand_store_target(ins.operand)
            if dst is not None and dst in code_addrs:
                store.append_annotation(
                    Annotation(
                        layer=0, kind=ANN_SMC,
                        start_addr=ins.addr, end_addr=ins.addr,
                        producer="deterministic", confidence=0.7,
                        payload={
                            "src_addr": ins.addr,
                            "dst_addr": dst,
                            "mnemonic": ins.mnemonic,
                            "operand": ins.operand,
                            "smc_kind": "absolute_into_code",
                            "source_file": parsed.path,
                        },
                    ),
                    source="layer0.asm",
                )
                stats.smc_sites += 1

    # ---- 6. Per-byte classification — sparse, only for addrs we touched.
    # Run-length encode to keep the event log compact.
    sorted_code = sorted(code_addrs)
    if sorted_code:
        run_start = sorted_code[0]
        prev = run_start
        for a in sorted_code[1:]:
            if a == prev + 1:
                prev = a
                continue
            store.append_annotation(
                Annotation(
                    layer=0, kind=ANN_CLASSIFY,
                    start_addr=run_start, end_addr=prev,
                    producer="deterministic", confidence=1.0,
                    payload={
                        "classification": "code",
                        "source_file": parsed.path,
                        "evidence_text": (
                            f"{prev - run_start + 1} consecutive parsed "
                            f"instructions in {parsed.path}"
                        ),
                    },
                ),
                source="layer0.asm",
            )
            stats.classified_bytes += prev - run_start + 1
            run_start = a
            prev = a
        # Flush the last run.
        store.append_annotation(
            Annotation(
                layer=0, kind=ANN_CLASSIFY,
                start_addr=run_start, end_addr=prev,
                producer="deterministic", confidence=1.0,
                payload={
                    "classification": "code",
                    "source_file": parsed.path,
                    "evidence_text": (
                        f"{prev - run_start + 1} consecutive parsed "
                        f"instructions in {parsed.path}"
                    ),
                },
            ),
            source="layer0.asm",
        )
        stats.classified_bytes += prev - run_start + 1

    for g in parsed.gaps:
        if g.end < g.start:
            continue
        store.append_annotation(
            Annotation(
                layer=0, kind=ANN_CLASSIFY,
                start_addr=g.start, end_addr=g.end,
                producer="deterministic", confidence=0.6,
                payload={
                    "classification": "data",
                    "source_file": parsed.path,
                    "evidence_text": (
                        f"gap marker in {parsed.path} ({g.end - g.start + 1} bytes)"
                    ),
                },
            ),
            source="layer0.asm",
        )
        stats.classified_bytes += g.end - g.start + 1

    return stats


def build_from_disasm_window(
    insns: Iterable[dict],
    store: CodeKnowledgeStore,
    *,
    source_file: str = "<runtime>",
) -> Layer0Stats:
    """Layer 0 update from on-demand disassembly results.

    `insns` is the same `dict` shape produced by `tools.c64_disasm`'s
    recursive_disasm — `{addr, bytes, mnemonic, op_str}` — keyed by the
    instruction address.
    """
    rows = list(insns or [])
    stats = Layer0Stats(sources=[source_file])
    if not rows:
        return stats

    # Persist the disassembly window as one ANN_DISASM annotation.
    serial_rows = []
    for r in rows:
        try:
            addr = int(r["addr"]) & 0xFFFF
        except (KeyError, TypeError, ValueError):
            continue
        b = r.get("bytes")
        if isinstance(b, (bytes, bytearray)):
            b_hex = bytes(b).hex()
            sz = len(b) or 1
        elif isinstance(b, str):
            b_hex = b
            sz = max(1, len(b) // 2)
        else:
            b_hex = ""
            sz = int(r.get("size") or 1)
        serial_rows.append({
            "addr": addr,
            "bytes_hex": b_hex,
            "mnemonic": str(r.get("mnemonic") or "").lower(),
            "operand": r.get("op_str") or r.get("operand"),
            "size_bytes": sz,
        })

    if not serial_rows:
        return stats

    serial_rows.sort(key=lambda r: r["addr"])
    first_addr = serial_rows[0]["addr"]
    last_addr = serial_rows[-1]["addr"]
    store.append_annotation(
        Annotation(
            layer=0, kind=ANN_DISASM,
            start_addr=first_addr, end_addr=last_addr,
            producer="deterministic", confidence=1.0,
            payload={"source_file": source_file, "instructions": serial_rows},
        ),
        source="layer0.disasm",
    )
    stats.instructions += len(serial_rows)

    # Same xref/SMC extraction logic as the asm path.
    code_addrs = {r["addr"] for r in serial_rows}
    for r in serial_rows:
        mnem = r["mnemonic"]
        op = r.get("operand") or ""
        if not mnem:
            continue
        if mnem in (CALL_MNEMONICS | JUMP_MNEMONICS | BRANCH_MNEMONICS):
            tgt = operand_direct_target(op)
            if tgt is not None:
                xkind = (
                    "jsr"     if mnem in CALL_MNEMONICS
                    else "branch" if mnem in BRANCH_MNEMONICS
                    else "jmp"
                )
                store.append_annotation(
                    Annotation(
                        layer=0, kind=ANN_XREF,
                        start_addr=r["addr"], end_addr=r["addr"],
                        producer="deterministic", confidence=1.0,
                        payload={
                            "src_addr": r["addr"], "dst_addr": tgt,
                            "xref_kind": xkind, "via_vector": None,
                            "source_file": source_file,
                        },
                    ),
                    source="layer0.disasm",
                )
                stats.xrefs += 1
            elif mnem == "jmp":
                vec = operand_indirect_vector(op)
                if vec is not None:
                    store.append_annotation(
                        Annotation(
                            layer=0, kind=ANN_INDIRECT,
                            start_addr=r["addr"], end_addr=r["addr"],
                            producer="deterministic", confidence=1.0,
                            payload={
                                "src_addr": r["addr"],
                                "via_vector": vec,
                                "resolved_target": None,
                                "source_file": source_file,
                            },
                        ),
                        source="layer0.disasm",
                    )
                    stats.indirect_sites += 1
        if mnem in STORE_MNEMONICS:
            dst = operand_store_target(op)
            if dst is not None and dst in code_addrs:
                store.append_annotation(
                    Annotation(
                        layer=0, kind=ANN_SMC,
                        start_addr=r["addr"], end_addr=r["addr"],
                        producer="deterministic", confidence=0.7,
                        payload={
                            "src_addr": r["addr"], "dst_addr": dst,
                            "mnemonic": mnem, "operand": op,
                            "smc_kind": "absolute_into_code",
                            "source_file": source_file,
                        },
                    ),
                    source="layer0.disasm",
                )
                stats.smc_sites += 1
    return stats
