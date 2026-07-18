"""Opt-in golden-question runner with machine-readable metrics.

The pytest integration is marked ``slow``/``golden`` and is disabled unless
``C64RE_RUN_GOLDEN=1``. Manifest validation and evaluator logic remain part of
the normal offline suite. Live runs use an isolated evaluation session root,
never the user's normal ``sessions/`` tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = Path(__file__).with_name("golden.jsonl")
_ADDRESS_RE = re.compile(r"^\$[0-9A-Fa-f]{2,4}$")


@dataclass(frozen=True)
class GoldenCase:
    id: str
    game: str
    question: str
    dump_path: str
    asm_files: tuple[str, ...]
    expected_addresses: tuple[str, ...]
    expected_keywords: tuple[str, ...]
    min_confidence: float
    keyword_patterns: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GoldenCase":
        case = cls(
            id=str(raw.get("id") or "").strip(),
            game=str(raw.get("game") or "").strip(),
            question=str(raw.get("question") or "").strip(),
            dump_path=str(raw.get("dump_path") or "").strip(),
            asm_files=tuple(str(p) for p in (raw.get("asm_files") or [])),
            expected_addresses=tuple(
                str(a).upper() for a in (raw.get("expected_addresses") or [])
            ),
            expected_keywords=tuple(
                str(k).strip() for k in (raw.get("expected_keywords") or [])
            ),
            keyword_patterns=tuple(
                (str(keyword).strip(), str(pattern))
                for keyword, pattern in (raw.get("keyword_patterns") or {}).items()
            ),
            min_confidence=float(raw.get("min_confidence", 0.0)),
        )
        case.validate()
        return case

    def validate(self) -> None:
        if not self.id or not re.fullmatch(r"[a-z0-9_]+", self.id):
            raise ValueError(f"invalid golden id: {self.id!r}")
        if not self.game or not self.question or not self.dump_path:
            raise ValueError(f"golden case {self.id!r} has missing inputs")
        if not self.expected_addresses or not self.expected_keywords:
            raise ValueError(f"golden case {self.id!r} has no expectations")
        invalid = [a for a in self.expected_addresses if not _ADDRESS_RE.fullmatch(a)]
        if invalid:
            raise ValueError(f"golden case {self.id!r} has invalid addresses: {invalid}")
        expected = set(self.expected_keywords)
        pattern_keys = [keyword for keyword, _pattern in self.keyword_patterns]
        unknown = sorted(set(pattern_keys) - expected)
        if unknown:
            raise ValueError(
                f"golden case {self.id!r} has patterns for unknown keywords: {unknown}",
            )
        if len(pattern_keys) != len(set(pattern_keys)):
            raise ValueError(f"golden case {self.id!r} has duplicate keyword patterns")
        for keyword, pattern in self.keyword_patterns:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(
                    f"golden case {self.id!r} has invalid pattern for "
                    f"{keyword!r}: {exc}",
                ) from exc
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"golden case {self.id!r} has invalid confidence")

    def pattern_for_keyword(self, keyword: str) -> str:
        patterns = dict(self.keyword_patterns)
        if keyword in patterns:
            return patterns[keyword]
        escaped = re.escape(keyword)
        return rf"(?<![0-9A-Za-z_]){escaped}(?![0-9A-Za-z_])"

    @property
    def dump(self) -> Path:
        return (ROOT / self.dump_path).resolve()

    @property
    def asm_paths(self) -> list[Path]:
        return [(ROOT / p).resolve() for p in self.asm_files]


@dataclass
class GoldenResult:
    case_id: str
    game: str
    question: str
    passed: bool
    verdict: str
    confidence: float | None
    expected_addresses: list[str]
    found_addresses: list[str]
    missing_addresses: list[str]
    expected_keywords: list[str]
    missing_keywords: list[str]
    min_confidence: float
    wall_time_s: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    tool_calls: int
    run_id: str | None
    failure_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DumpAnchorResult:
    """Offline evidence tying one expected address to its checked-in dump."""

    address: str
    supported: bool
    bytes_hex: str
    disassembly: str
    reference_sites: tuple[str, ...]
    asm_matches: tuple[str, ...]
    evidence: tuple[str, ...]
    failures: tuple[str, ...]


# 6502 opcodes whose following operand is an absolute or zero-page address.
# This deliberately excludes immediate/relative modes so a byte coincidence is
# less likely to be mistaken for a reference to an empty data location.
_ABSOLUTE_REFERENCE_OPCODES = frozenset({
    0x0D, 0x0E, 0x19, 0x1D, 0x1E, 0x20, 0x2C, 0x2D, 0x2E, 0x39, 0x3D, 0x3E,
    0x4C, 0x4D, 0x4E, 0x59, 0x5D, 0x5E, 0x6C, 0x6D, 0x6E, 0x79, 0x7D, 0x7E,
    0x8C, 0x8D, 0x8E, 0x99, 0x9D, 0xAC, 0xAD, 0xAE, 0xB9, 0xBC, 0xBD, 0xBE,
    0xCC, 0xCD, 0xCE, 0xD9, 0xDD, 0xDE, 0xEC, 0xED, 0xEE, 0xF9, 0xFD, 0xFE,
})
_ZERO_PAGE_REFERENCE_OPCODES = frozenset({
    0x05, 0x06, 0x15, 0x16, 0x24, 0x25, 0x26, 0x35, 0x36, 0x45, 0x46, 0x55,
    0x56, 0x65, 0x66, 0x75, 0x76, 0x84, 0x85, 0x86, 0x94, 0x95, 0x96, 0xA4,
    0xA5, 0xA6, 0xB4, 0xB5, 0xB6, 0xC4, 0xC5, 0xC6, 0xD5, 0xD6, 0xE4, 0xE5,
    0xE6, 0xF5, 0xF6,
})


def _reference_sites(memory: bytes, address: int, *, limit: int = 8) -> tuple[int, ...]:
    low = address & 0xFF
    high = (address >> 8) & 0xFF
    found: list[int] = []
    for offset in range(max(0, len(memory) - 2)):
        opcode = memory[offset]
        if (
            opcode in _ABSOLUTE_REFERENCE_OPCODES
            and memory[offset + 1] == low
            and memory[offset + 2] == high
        ):
            found.append(offset)
        elif (
            address <= 0xFF
            and opcode in _ZERO_PAGE_REFERENCE_OPCODES
            and memory[offset + 1] == low
        ):
            found.append(offset)
        if len(found) >= limit:
            break
    return tuple(found)


def inspect_dump_anchors(case: GoldenCase) -> list[DumpAnchorResult]:
    """Validate every expected address against dump bytes and asm listings.

    This is intentionally deterministic and makes no model, web, or emulator
    calls. A locally empty (all ``00``/``FF``) address needs at least one
    decoded 6502 address operand elsewhere in the dump to count as supported.
    Whenever a listed asm file has an instruction at the exact address, its
    encoded bytes must match the dump.
    """
    from code_kb.asm_parser import parse_path
    from tools.c64_disasm import linear_disasm

    memory = case.dump.read_bytes()
    parsed_asm = [(path, parse_path(path)) for path in case.asm_paths]
    results: list[DumpAnchorResult] = []
    for canonical in case.expected_addresses:
        address = int(canonical[1:], 16)
        failures: list[str] = []
        evidence: list[str] = []
        asm_matches: list[str] = []
        if not 0 <= address < len(memory):
            failures.append(
                f"{canonical} is outside the {len(memory)}-byte dump",
            )
            results.append(DumpAnchorResult(
                address=canonical,
                supported=False,
                bytes_hex="",
                disassembly="",
                reference_sites=(),
                asm_matches=(),
                evidence=(),
                failures=tuple(failures),
            ))
            continue

        window = memory[address:min(address + 16, len(memory))]
        bytes_hex = window.hex(" ").upper()
        signal_bytes = sum(byte not in {0x00, 0xFF} for byte in window)
        references = _reference_sites(memory, address)
        if signal_bytes:
            evidence.append(
                f"{signal_bytes}/{len(window)} local bytes differ from 00/FF",
            )
        if references:
            sites = ", ".join(f"${site:04X}" for site in references)
            evidence.append(f"6502 operand reference(s) at {sites}")
        if not signal_bytes and not references:
            failures.append(
                f"{canonical} looks like empty RAM and has no 6502 operand reference",
            )

        try:
            disassembly = linear_disasm(
                memory, address, length=len(window), max_lines=8,
            )
        except (RuntimeError, ValueError):
            disassembly = ""

        for path, parsed in parsed_asm:
            for instruction in parsed.instructions:
                if instruction.addr != address:
                    continue
                actual = memory[address:address + len(instruction.bytes_)]
                label = f"{path.name}:{canonical}"
                if actual != instruction.bytes_:
                    failures.append(
                        f"{label} lists {instruction.bytes_.hex(' ').upper()} "
                        f"but dump has {actual.hex(' ').upper()}",
                    )
                else:
                    asm_matches.append(label)

        results.append(DumpAnchorResult(
            address=canonical,
            supported=not failures,
            bytes_hex=bytes_hex,
            disassembly=disassembly,
            reference_sites=tuple(f"${site:04X}" for site in references),
            asm_matches=tuple(asm_matches),
            evidence=tuple(evidence),
            failures=tuple(failures),
        ))
    return results


def load_cases(path: str | Path = DEFAULT_MANIFEST) -> list[GoldenCase]:
    manifest = Path(path)
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for line_no, raw_line in enumerate(manifest.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{manifest}:{line_no}: invalid JSON: {exc}") from exc
        case = GoldenCase.from_dict(raw)
        if case.id in seen:
            raise ValueError(f"{manifest}:{line_no}: duplicate id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return cases


def evaluate_turn(case: GoldenCase, turn: Any, *, wall_time_s: float) -> GoldenResult:
    """Compare a completed ``TurnResult`` against one golden contract."""
    answer_blob = "\n".join([
        str(getattr(turn, "answer", "") or ""),
        *[str(e) for e in (getattr(turn, "evidence", None) or [])],
    ])
    found_ints = {int(a) & 0xFFFF for a in (getattr(turn, "addresses", None) or [])}
    found_addresses = [f"${a:04X}" for a in sorted(found_ints)]
    missing_addresses = [
        a for a in case.expected_addresses
        if int(a[1:], 16) not in found_ints
    ]
    missing_keywords = [
        keyword for keyword in case.expected_keywords
        if not re.search(
            case.pattern_for_keyword(keyword), answer_blob, re.IGNORECASE,
        )
    ]
    confidence_raw = getattr(turn, "confidence", None)
    confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None
    failures: list[str] = []
    if missing_addresses:
        failures.append("missing addresses: " + ", ".join(missing_addresses))
    if missing_keywords:
        failures.append("missing keywords: " + ", ".join(missing_keywords))
    if confidence is None or confidence < case.min_confidence:
        failures.append(
            f"confidence {confidence!r} below {case.min_confidence:.2f}",
        )

    state = getattr(turn, "raw_state", None) or {}
    usage = state.get("llm_usage") or []
    input_tokens = sum(int(e.get("input_tokens") or 0) for e in usage)
    output_tokens = sum(int(e.get("output_tokens") or 0) for e in usage)
    return GoldenResult(
        case_id=case.id,
        game=case.game,
        question=case.question,
        passed=not failures,
        verdict=str(getattr(turn, "verdict", "n/a") or "n/a"),
        confidence=confidence,
        expected_addresses=list(case.expected_addresses),
        found_addresses=found_addresses,
        missing_addresses=missing_addresses,
        expected_keywords=list(case.expected_keywords),
        missing_keywords=missing_keywords,
        min_confidence=case.min_confidence,
        wall_time_s=round(float(wall_time_s), 6),
        llm_calls=len(usage),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=int(state.get("tokens_used") or input_tokens + output_tokens),
        cost_usd=float(state.get("budget_used") or 0.0),
        tool_calls=(
            sum(
                int(values.get("calls", 0))
                for values in (state.get("tool_call_stats") or {}).values()
            )
            or len(state.get("tool_results") or [])
        ),
        run_id=str(state.get("run_id")) if state.get("run_id") else None,
        failure_reasons=failures,
    )


def execute_case(
    case: GoldenCase, *, session_root: str | Path | None = None,
    allow_external_tools: bool = False,
) -> GoldenResult:
    """Run one case through the real graph in an isolated session tree.

    VICE and Tavily are disabled by default so the five checked-in dumps/asm
    files, not mutable emulator/web state, define the regression input.
    """
    from graph import nodes
    from tools.agent_runner import run_question

    root = Path(session_root) if session_root else ROOT / "evals" / ".sessions" / case.id
    prior_sessions_dir = nodes.SESSIONS_DIR
    external_env: dict[str, str] = {}
    if not allow_external_tools:
        for name in ("VICE_MCP_URL", "TAVILY_API_KEY"):
            if name in os.environ:
                external_env[name] = os.environ.pop(name)
    nodes.SESSIONS_DIR = root
    started = time.monotonic()
    try:
        events = list(run_question(
            game=case.game,
            question=case.question,
            dump_path=case.dump,
            asm_dir=ROOT / "asm_dir",
            asm_files=case.asm_paths,
            thread_id=f"golden-{case.id}-{uuid.uuid4().hex[:8]}",
        ))
    finally:
        nodes.SESSIONS_DIR = prior_sessions_dir
        os.environ.update(external_env)
    errors = [payload for kind, payload in events if kind == "error"]
    if errors:
        raise RuntimeError(f"golden case {case.id} graph error: {errors[-1]}")
    completed = [payload for kind, payload in events if kind == "done"]
    if len(completed) != 1:
        raise RuntimeError(f"golden case {case.id} produced {len(completed)} done events")
    return evaluate_turn(
        case, completed[0], wall_time_s=time.monotonic() - started,
    )


def write_results(results: Iterable[GoldenResult], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    with output.open("w") as handle:
        for result in rows:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run C64-RE golden questions")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--session-root", type=Path, default=None,
        help=(
            "fresh batch session root (default: "
            "evals/.sessions/<UTC stamp>-<nonce>)"
        ),
    )
    parser.add_argument(
        "--allow-external-tools", action="store_true",
        help="allow configured VICE/Tavily calls (disabled by default)",
    )
    parser.add_argument(
        "--max-cost-usd", type=float, default=None,
        help="stop before another case would exceed this cumulative estimate",
    )
    parser.add_argument(
        "--min-case-budget-usd", type=float, default=0.35,
        help="do not start another case with less remaining allowance",
    )
    args = parser.parse_args()

    if os.getenv("C64RE_RUN_GOLDEN", "").strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        parser.error("set C64RE_RUN_GOLDEN=1 to authorize live LLM evaluation")
    if args.max_cost_usd is not None and args.max_cost_usd <= 0:
        parser.error("--max-cost-usd must be positive")
    if args.min_case_budget_usd <= 0:
        parser.error("--min-case-budget-usd must be positive")

    selected = load_cases(args.manifest)
    if args.case:
        wanted = set(args.case)
        selected = [case for case in selected if case.id in wanted]
        missing = wanted - {case.id for case in selected}
        if missing:
            parser.error("unknown case id(s): " + ", ".join(sorted(missing)))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_root = args.session_root or (
        ROOT / "evals" / ".sessions" / f"{stamp}-{uuid.uuid4().hex[:8]}"
    )
    results: list[GoldenResult] = []
    cumulative_cost = 0.0
    for case in selected:
        remaining = (
            args.max_cost_usd - cumulative_cost
            if args.max_cost_usd is not None else None
        )
        if remaining is not None and remaining < args.min_case_budget_usd:
            print(
                f"stopping before {case.id}: ${remaining:.4f} remains, below "
                f"the ${args.min_case_budget_usd:.4f} minimum case reserve",
            )
            break

        previous_case_cap = os.environ.get("C64RE_USD_BUDGET")
        if remaining is not None:
            configured_cap: float | None = None
            if previous_case_cap:
                try:
                    parsed_cap = float(previous_case_cap)
                    if parsed_cap > 0:
                        configured_cap = parsed_cap
                except ValueError:
                    pass
            case_cap = min(remaining, configured_cap or remaining)
            os.environ["C64RE_USD_BUDGET"] = f"{case_cap:.8f}"
        try:
            result = execute_case(
                case,
                session_root=batch_root / case.id,
                allow_external_tools=args.allow_external_tools,
            )
        finally:
            if remaining is not None:
                if previous_case_cap is None:
                    os.environ.pop("C64RE_USD_BUDGET", None)
                else:
                    os.environ["C64RE_USD_BUDGET"] = previous_case_cap
        results.append(result)
        cumulative_cost += result.cost_usd
        badge = "PASS" if result.passed else "FAIL"
        print(
            f"{badge} {case.id}: conf={result.confidence} "
            f"tokens={result.total_tokens} cost=${result.cost_usd:.4f} "
            f"wall={result.wall_time_s:.2f}s",
        )
        if result.failure_reasons:
            print("  " + "; ".join(result.failure_reasons))
        if args.max_cost_usd is not None:
            print(
                f"  cumulative=${cumulative_cost:.4f} / "
                f"${args.max_cost_usd:.4f}",
            )

    output = args.output
    if output is None:
        output = ROOT / "evals" / "results" / f"golden-{stamp}.jsonl"
    write_results(results, output)
    print(f"wrote {output}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
