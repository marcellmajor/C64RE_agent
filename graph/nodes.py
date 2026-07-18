"""Node implementations.

LLM nodes call the configured provider via `graph.llm.get_llm`. Tool
nodes are real implementations with graceful fallback when their
external dependency is unreachable (Tavily without API key, VICE-MCP
without a running server, etc.).

KB persistence is delegated to `memory.KnowledgeStore`; nodes never
touch sqlite/JSON directly.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from graph.llm import get_llm, load_config, structured_output_enabled
from graph import usage as llm_usage
from graph.plan_utils import (
    MAX_CONSECUTIVE_REVISES,
    MAX_PLAN_STEPS,
    VICE_ADDRESS_ALIASES as _VICE_ADDRESS_ALIASES,
    failed_step_notes,
    parallel_runnable_steps,
    pending_steps,
    resolve_session_slug,
    runnable_steps,
    slugify,
    step_is_concrete,
    step_status,
)
from graph.prompts import MASTER_PREAMBLE, system_message
from graph.state import (
    C64State,
    compact_tool_results_update,
    identify_tool_results,
    tool_call_total,
)
from c64re_agent.paths import sessions_dir
from memory import KnowledgeStore, extract_hex_addresses, get_store
from memory.redaction import redact_text, redact_value
from memory.schema import (
    EVT_ANALYSIS,
    EVT_CONSOLIDATED,
    EVT_DATA_STRUCTURE,
    EVT_HYPOTHESIS,
    EVT_LABEL,
    EVT_RUN_SUMMARY,
    EVT_ROUTINE,
    EVT_TOOL_RESULT,
    EVT_VERDICT,
)
from tools import c64_disasm

SESSIONS_DIR = sessions_dir()

# Kept for back-compat with anything that imported the old name. New code
# should call `graph.prompts.system_message(role)` instead.
SYSTEM_PREAMBLE = MASTER_PREAMBLE


def _try_parse_json(content: str) -> Any | None:
    """Best-effort: pull the first plausible JSON object/array out of text."""
    if not content:
        return None
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, re.I)
    candidates: list[str] = []
    if fenced:
        candidates.append(fenced.group(1).strip())
    obj_start, obj_end = content.find("{"), content.rfind("}")
    arr_start, arr_end = content.find("["), content.rfind("]")
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(content[obj_start : obj_end + 1])
    if arr_start != -1 and arr_end > arr_start:
        candidates.append(content[arr_start : arr_end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:  # noqa: BLE001
            continue
    return None


def _looks_like_plan_steps(payload: Any) -> bool:
    """True iff *payload* looks like planner output: [{id,tool,...}, ...]."""
    if not isinstance(payload, list) or not payload:
        return False
    for step in payload:
        if not isinstance(step, dict):
            return False
        sid = step.get("id")
        tool = step.get("tool")
        sid_ok = str(sid or "").strip() != ""
        tool_ok = str(tool or "").strip() != ""
        if not sid_ok or not tool_ok:
            return False
    return True


def _unwrap_json_array_to_contract_dict(
    payload: list[Any],
    *,
    fallback: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract the contract object from accidental JSON-array wrappers.

    Models occasionally emit ``[{"answer": ...}]`` (array of one object) or nest under
    ``candidate_answer`` / ``response`` despite the analyst/critic/etc.
    role contract requesting a lone object.
    """
    fk = frozenset(k for k in fallback if not str(k).startswith("_"))
    envelope_keys = (
        "candidate_answer", "result", "data", "output",
        "response", "parsed", "verdict",
    )

    def _maybe_env_dict(d: dict[str, Any]) -> dict[str, Any] | None:
        if fk.intersection(d.keys()):
            return dict(d)
        for ek in envelope_keys:
            inner = d.get(ek)
            if isinstance(inner, dict) and fk.intersection(inner.keys()):
                return dict(inner)
            if isinstance(inner, dict) and not fk.isdisjoint(
                set(inner.keys()),
            ):
                return dict(inner)
        for ek in envelope_keys:
            inner = d.get(ek)
            if isinstance(inner, dict):
                return dict(inner)
        return None

    if len(payload) == 1:
        only = payload[0]
        if isinstance(only, dict):
            got = _maybe_env_dict(only)
            return got
        return None

    best: dict[str, Any] | None = None
    best_score = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        cand = _maybe_env_dict(item)
        if cand is None:
            continue
        score = len(fk.intersection(cand.keys()))
        if score > best_score:
            best_score = score
            best = cand
    return best


def _coerce_parsed_llm_json(
    parsed: Any,
    *,
    fallback: dict[str, Any],
    allow_plan_steps_array: bool = False,
) -> Any:
    """Normalize planner/analyst payloads so callers always see dict-or-plan-list."""
    if not isinstance(parsed, list):
        return parsed

    if allow_plan_steps_array and _looks_like_plan_steps(parsed):
        return parsed

    inner = _unwrap_json_array_to_contract_dict(parsed, fallback=fallback)
    if inner is not None:
        return inner

    return parsed


# Envelope keys models wrap their payloads in despite the role contracts
# requesting a bare object (same family `_unwrap_json_array_to_contract_dict`
# tolerates for array-wrapped payloads).
_DICT_ENVELOPE_KEYS = (
    "candidate_answer", "result", "data", "output", "response", "parsed",
    "verdict",
)

_CRITIC_DECISIONS = frozenset({"accept", "revise", "replan"})


def _dict_contract_error(role: str, payload: dict[str, Any]) -> str | None:
    """None when *payload* satisfies the role's minimal contract.

    Review finding: `_safe_invoke` accepted ANY parsed dict as success —
    an analyst reply of ``{"unexpected": 1}`` was returned, recorded
    ok=True, and prevented backups from running. Validation is minimal
    (the discriminator key per role, tolerant of optional fields) so
    legitimate sparse replies still pass.
    """
    if role == "planner":
        plan = payload.get("plan")
        if not isinstance(plan, list) or not plan:
            return "planner contract violated: no non-empty `plan` list"
        return None
    if role == "executor":
        if not isinstance(payload.get("args"), dict):
            return "executor contract violated: `args` missing or not a dict"
        if not (payload.get("step_id") or payload.get("tool")):
            return "executor contract violated: no step identity (step_id/tool)"
        return None
    if role == "synthesizer":
        keys = ("labels", "routines", "data_structures", "hypotheses", "notes")
        if not any(k in payload for k in keys):
            return (
                "synthesizer contract violated: none of "
                f"{'/'.join(keys)} present"
            )
        return None
    if role == "curator":
        keys = ("summary", "addresses_kept", "events_compacted")
        if not any(k in payload for k in keys):
            return (
                "curator contract violated: none of "
                f"{'/'.join(keys)} present"
            )
        return None
    if role == "analyst":
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return "analyst contract violated: no non-empty `answer`"
        return None
    if role == "critic":
        decision = str(payload.get("decision") or "").strip().lower()
        if decision not in _CRITIC_DECISIONS:
            return (
                "critic contract violated: `decision` missing or not one of "
                "accept/revise/replan"
            )
        return None
    return None  # unknown roles: no contract to enforce


def _extract_role_contract_dict(
    role: str, payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (usable_dict, None) or (None, contract_error).

    Tolerates one level of harmless envelope nesting: if the outer dict
    fails the role contract but a known envelope key holds an inner dict
    that passes, the inner dict is what the node should consume.
    """
    err = _dict_contract_error(role, payload)
    if err is None:
        return payload, None
    for ek in _DICT_ENVELOPE_KEYS:
        inner = payload.get(ek)
        if isinstance(inner, dict) and _dict_contract_error(role, inner) is None:
            return dict(inner), None
    return None, err


def _flatten_lc_ai_message_content(msg: Any) -> str:
    """Turn an ``AIMessage`` body into plain text JSON-like consumers expect.

    Newer OpenAI models (gpt-5.x, o-series) return ``content`` as a **list**
    of blocks: ``[{"type":"reasoning",...}, {"type":"text","text":"..."}]``.
    The old path did ``str(msg.content)``, which produced non-JSON noise like
    ``"[{'type': 'reasoning', ...}]"`` with no usable assistant text —
    poisoning ``_try_parse_json`` for planner / analyst / critic.

    This flattener:
    • concatenates ``type in {text, output_text}`` string payloads,
    • optionally harvests prose from reasoning ``summary`` sub-blocks,
    • falls back to ``additional_kwargs["parsed"]`` JSON when the visible
      body is empty (structured-output mode),
    • falls back to ``additional_kwargs["refusal"]``.
    """

    extras = getattr(msg, "additional_kwargs", None) or {}

    refusal = extras.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        return refusal.strip()

    raw = getattr(msg, "content", None)

    def _block_collect(block: Any, into: list[str]) -> None:
        if isinstance(block, str):
            if block.strip():
                into.append(block)
            return

        if not isinstance(block, dict):
            txt = getattr(block, "text", None)
            if isinstance(txt, str) and txt.strip():
                into.append(txt)
            return

        btype = str(block.get("type") or "").lower()

        # OpenAI Responses / Chat Completions content parts
        if btype in ("text", "output_text"):
            t = block.get("text")
            if isinstance(t, str) and t.strip():
                into.append(t)
            elif isinstance(t, list):
                for x in t:
                    _block_collect(x, into)
            return

        # Reasoning trace — occasionally carries human-readable summaries
        if btype == "reasoning":
            summ = block.get("summary")
            if isinstance(summ, list):
                for s in summ:
                    _block_collect(s, into)
            elif isinstance(summ, str) and summ.strip():
                into.append(summ)
            return

        # Generic nested text (Anthropic, etc.)
        if isinstance(block.get("text"), str) and block["text"].strip():
            into.append(block["text"])

        # Claude / multi-part "content" inside a dict payload
        inner = block.get("content")
        if isinstance(inner, list):
            for x in inner:
                _block_collect(x, into)
        elif isinstance(inner, str) and inner.strip():
            into.append(inner)

    pieces: list[str] = []

    if raw is None:
        pass
    elif isinstance(raw, str):
        if raw.strip():
            pieces.append(raw)
    elif isinstance(raw, dict):
        _block_collect(raw, pieces)
    elif isinstance(raw, list):
        for item in raw:
            _block_collect(item, pieces)
    else:
        s = str(raw)
        if s.strip():
            pieces.append(s)

    # LangChain v0.3 compatibility: Responses API reasoning may be stripped out of
    # `content` and kept only under `additional_kwargs["reasoning"]`, which makes
    # `content` look empty despite a successful completion.
    rk = extras.get("reasoning")
    if isinstance(rk, dict):
        rb = dict(rk)
        rb.setdefault("type", "reasoning")
        _block_collect(rb, pieces)

    text = "\n".join(pieces).strip()

    # Structured outputs: LangChain/OpenAI sometimes put JSON only here.
    parsed = extras.get("parsed")
    if not text and isinstance(parsed, (dict, list)):
        try:
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            pass

    return text


def _call_cost_reservation_usd(
    llm: Any,
    model_name: str | None,
    prompt: str,
    *,
    system_role: str | None = None,
    system_text: str | None = None,
) -> float:
    """Conservative worst-case cost for one text call before it is sent.

    One character per input token intentionally over-reserves source-heavy
    prompts, and the configured maximum completion is reserved in full. The
    cushion covers provider wrappers and native JSON-schema tokens that are
    not represented in the user/system strings.
    """
    max_output = (
        getattr(llm, "max_tokens", None)
        or getattr(llm, "max_completion_tokens", None)
        or 0
    )
    try:
        max_output = max(0, int(max_output))
    except (TypeError, ValueError):
        max_output = 0
    system = (
        system_text
        if system_text is not None
        else system_message(system_role or "")
    )
    estimated_input = len(system) + len(prompt) + 4_096
    return llm_usage.estimate_cost_usd(
        model_name, estimated_input, max_output,
    )


def _remaining_llm_budget(state: C64State) -> float:
    """Remaining priced-call allowance including this node's pending calls."""
    from graph.routers import budget_cap

    return max(
        0.0,
        budget_cap()
        - float(state.get("budget_used") or 0.0)
        - llm_usage.pending_cost_usd(),
    )


def _invoke_one(
    role: str,
    prompt: str,
    *,
    system_role: str | None = None,
    max_cost_usd: float | None = None,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """Run a single LLM call. Returns (content, error_or_None, usage_entry).

    `system_role` overrides which role-block is attached to the master
    preamble — useful when a backup model takes over for another role
    (e.g. critic backing up the analyst); we want it to *behave* like
    the original role, not its own.

    An empty content string is treated as an error: Gemini in particular
    silently returns empty bodies when its safety filter triggers
    (`finish_reason=SAFETY`), which would otherwise look like success.

    Usage accounting (review finding 5): the recorded entry's ``ok``
    means "produced usable output", not "the HTTP call succeeded" —
    ``transport_ok`` carries that separately. The live entry is returned
    so `_safe_invoke` can demote it once contract parsing fails.
    """
    as_role = system_role if system_role and system_role != role else None
    model_name = None
    try:
        llm = get_llm(role)
        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
        sys_prompt = system_message(system_role or role)
        reservation = _call_cost_reservation_usd(
            llm, model_name, prompt, system_role=system_role or role,
        )
        if max_cost_usd is not None and reservation > max_cost_usd:
            error = (
                f"cost reservation ${reservation:.4f} exceeds remaining "
                f"run allowance ${max_cost_usd:.4f}"
            )
            entry = llm_usage.record({
                "role": role,
                "as_role": as_role,
                "model": model_name,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "reserved_cost_usd": reservation,
                "ok": False,
                "transport_ok": False,
                "rejection": "cost_reservation",
                "error": error,
            })
            return "", error, entry
        msg = llm.invoke([SystemMessage(sys_prompt), HumanMessage(prompt)])
    except Exception as e:  # noqa: BLE001
        entry = llm_usage.record({
            "role": role, "as_role": as_role, "model": model_name,
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            "ok": False, "transport_ok": False,
            "error": f"{type(e).__name__}: {e}"[:200],
        })
        return "", f"{type(e).__name__}: {e}", entry

    in_tok, out_tok = llm_usage.extract_usage(msg)
    entry = llm_usage.record({
        "role": role, "as_role": as_role, "model": model_name,
        "input_tokens": in_tok, "output_tokens": out_tok,
        "cost_usd": llm_usage.estimate_cost_usd(model_name, in_tok, out_tok),
        "ok": True, "transport_ok": True,
    })

    content = _flatten_lc_ai_message_content(msg)
    if not content.strip():
        meta = getattr(msg, "response_metadata", None) or {}
        finish = meta.get("finish_reason") or meta.get("stop_reason") or "?"
        extras = getattr(msg, "additional_kwargs", None) or {}
        hint = ""
        if extras.get("reasoning") and not extras.get("refusal"):
            hint = (
                "; model returned reasoning without visible text — "
                "try increasing defaults.max_tokens, set agents.*.reasoning_effort, "
                "or use C64RE_GPT5_MIN_MAX_TOKENS"
            )
        err = f"empty response (finish_reason={finish}){hint}"
        llm_usage.mark_failed(entry, err)
        return "", err, entry
    return content, None, entry


_ROLE_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "planner": {
        "title": "C64REPlanner",
        "type": "object",
        "properties": {"plan": {"type": "array", "items": {"type": "object"}}},
        "required": ["plan"],
    },
    "executor": {
        "title": "C64REExecutor",
        "type": "object",
        "properties": {
            "step_id": {"type": "string"},
            "tool": {"type": "string"},
            "args": {"type": "object"},
            "rationale": {"type": "string"},
        },
        "required": ["step_id", "tool", "args"],
    },
    "synthesizer": {
        "title": "C64RESynthesizer",
        "type": "object",
        "properties": {
            "labels": {"type": "array"},
            "routines": {"type": "array"},
            "data_structures": {"type": "array"},
            "hypotheses": {"type": "array"},
            "notes": {"type": "string"},
        },
        "minProperties": 1,
    },
    "analyst": {
        "title": "C64REAnalyst",
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "evidence": {"type": "array"},
            "open_questions": {"type": "array"},
        },
        "required": ["answer", "confidence", "evidence", "open_questions"],
    },
    "critic": {
        "title": "C64RECritic",
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["accept", "revise", "replan"]},
            "critique": {"type": "string"},
            "suggested_steps": {"type": "array"},
            "refuted_hypotheses": {"type": "array"},
            "optional_followups": {"type": "array"},
        },
        "required": ["decision", "critique"],
    },
    "curator": {
        "title": "C64RECurator",
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "addresses_kept": {"type": "array"},
            "events_compacted": {"type": "array"},
        },
        "required": ["summary", "addresses_kept", "events_compacted"],
    },
}


def _invoke_one_structured(
    role: str,
    prompt: str,
    *,
    contract_role: str,
    max_cost_usd: float | None = None,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """Try one native JSON-schema call and preserve ordinary usage telemetry."""
    model_name = None
    reservation = 0.0
    try:
        llm = get_llm(role)
        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
        reservation = _call_cost_reservation_usd(
            llm, model_name, prompt, system_role=contract_role,
        )
        if max_cost_usd is not None and reservation > max_cost_usd:
            error = (
                f"cost reservation ${reservation:.4f} exceeds remaining "
                f"run allowance ${max_cost_usd:.4f}"
            )
            entry = llm_usage.record({
                "role": role,
                "as_role": contract_role if contract_role != role else None,
                "model": model_name,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "reserved_cost_usd": reservation,
                "ok": False,
                "transport_ok": False,
                "structured_output": True,
                "rejection": "cost_reservation",
                "error": error,
            })
            return "", error, entry
        schema = _ROLE_JSON_SCHEMAS[contract_role]
        structured = llm.with_structured_output(
            schema,
            method="json_schema",
            include_raw=True,
            strict=False,
        )
        result = structured.invoke([
            SystemMessage(system_message(contract_role)),
            HumanMessage(prompt),
        ])
    except Exception as exc:  # noqa: BLE001
        entry = llm_usage.record({
            "role": role,
            "as_role": contract_role if contract_role != role else None,
            "model": model_name,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "reserved_cost_usd": reservation,
            "ok": False,
            "transport_ok": False,
            "structured_output": True,
            "error": f"{type(exc).__name__}: {exc}"[:200],
        })
        return "", f"{type(exc).__name__}: {exc}", entry

    raw_message = None
    parsing_error = None
    if isinstance(result, dict) and (
        "parsed" in result or "parsing_error" in result or "raw" in result
    ):
        parsed = result.get("parsed")
        raw_message = result.get("raw")
        parsing_error = result.get("parsing_error")
    else:
        parsed = result
    if hasattr(parsed, "model_dump"):
        parsed = parsed.model_dump()

    in_tok, out_tok = (
        llm_usage.extract_usage(raw_message)
        if raw_message is not None else (0, 0)
    )
    entry = llm_usage.record({
        "role": role,
        "as_role": contract_role if contract_role != role else None,
        "model": model_name,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": llm_usage.estimate_cost_usd(
            model_name, in_tok, out_tok,
        ),
        "ok": parsed is not None and parsing_error is None,
        "transport_ok": True,
        "structured_output": True,
    })
    if parsing_error is not None or parsed is None:
        err = (
            f"structured parse failed: {parsing_error}"
            if parsing_error is not None else "structured response had no parsed value"
        )
        llm_usage.mark_failed(entry, err)
        return "", err, entry
    try:
        return json.dumps(parsed, ensure_ascii=False), None, entry
    except (TypeError, ValueError) as exc:
        err = f"structured response is not JSON serializable: {exc}"
        llm_usage.mark_failed(entry, err)
        return "", err, entry


def _safe_invoke(
    role: str,
    prompt: str,
    fallback: dict[str, Any],
    backup_roles: list[str] | None = None,
    *,
    allow_plan_steps_array: bool = False,
    remaining_budget_usd: float | None = None,
) -> dict[str, Any] | list[Any]:
    """Try the LLM, with optional backup providers, then degrade to fallback.

    The backup chain is preserved for resilience (Gemini safety filters
    can blackhole answers entirely), but every backup is prompted with
    the *primary role's* system block, so a critic-model standing in for
    an analyst still produces an analyst-shaped answer.

    Diagnostics: errors, empty responses, and unparseable bodies are
    printed to stderr so the user can see why a node degraded into its
    heuristic fallback rather than silently getting a stub answer.
    """
    chain = [role, *(backup_roles or [])]
    errors: list[str] = []
    last_entry: dict[str, Any] | None = None
    reservation_denied = False
    remaining = remaining_budget_usd

    def _charge(entry: dict[str, Any] | None) -> None:
        nonlocal remaining
        if remaining is not None and entry is not None:
            remaining = max(
                0.0, remaining - float(entry.get("cost_usd") or 0.0),
            )

    def _hold_unknown_transport_reservation(
        entry: dict[str, Any] | None,
    ) -> None:
        """Do not reuse worst-case allowance after an unknown transport failure."""
        nonlocal remaining
        if (
            remaining is None
            or entry is None
            or entry.get("transport_ok") is not False
            or entry.get("rejection") == "cost_reservation"
        ):
            return
        reserved = float(entry.get("reserved_cost_usd") or 0.0)
        actual = float(entry.get("cost_usd") or 0.0)
        remaining = max(0.0, remaining - max(0.0, reserved - actual))

    for r in chain:
        if structured_output_enabled(r) and role in _ROLE_JSON_SCHEMAS:
            content, err, entry = _invoke_one_structured(
                r, prompt, contract_role=role, max_cost_usd=remaining,
            )
            _charge(entry)
            if err:
                if entry and entry.get("rejection") == "cost_reservation":
                    reservation_denied = True
                else:
                    # A structured transport exception may occur after the
                    # provider accepted/billed the request but before usage
                    # metadata returned. Hold its full reservation before the
                    # same-role plain retry; `_invoke_one` then performs a
                    # fresh reservation check against the reduced remainder.
                    _hold_unknown_transport_reservation(entry)
                    print(
                        f"[llm:{r}] native structured output unavailable — "
                        f"falling back to plain JSON: {err}",
                        file=sys.stderr,
                        flush=True,
                    )
                    errors.append(f"{r} structured: {err}")
                    content, err, entry = _invoke_one(
                        r, prompt, system_role=role, max_cost_usd=remaining,
                    )
                    _charge(entry)
        else:
            content, err, entry = _invoke_one(
                r, prompt, system_role=role, max_cost_usd=remaining,
            )
            _charge(entry)
        if entry is not None:
            last_entry = entry
        if err:
            if entry and entry.get("rejection") == "cost_reservation":
                reservation_denied = True
            print(
                f"[llm:{r}] CALL FAILED — {err}",
                file=sys.stderr, flush=True,
            )
            errors.append(f"{r}: {err}")
            continue

        parsed = _try_parse_json(content)
        if parsed is not None:
            shaped = _coerce_parsed_llm_json(
                parsed,
                fallback=fallback,
                allow_plan_steps_array=allow_plan_steps_array,
            )
            if isinstance(shaped, dict):
                # Role-aware contract check (review): ANY dict used to be
                # accepted, so `{"unexpected": 1}` was returned as success
                # and blocked backups. Validation is against the PRIMARY
                # role — a backup stands in for it and must produce its
                # shape. Envelope-wrapped payloads are unwrapped.
                usable, contract_err = _extract_role_contract_dict(role, shaped)
                if usable is not None:
                    if r != role:
                        print(
                            f"[llm:{role}] using backup `{r}` (primary failed)",
                            file=sys.stderr, flush=True,
                        )
                    usable.setdefault("_role_used", r)
                    return usable
                llm_usage.mark_failed(entry, contract_err or "contract violated")
                print(
                    f"[llm:{r}] {contract_err}",
                    file=sys.stderr, flush=True,
                )
                errors.append(f"{r}: {contract_err}")
                continue  # try the next backup role
            if isinstance(shaped, list):
                if allow_plan_steps_array and _looks_like_plan_steps(shaped):
                    if r != role:
                        print(
                            f"[llm:{role}] using backup `{r}` (primary failed)",
                            file=sys.stderr, flush=True,
                        )
                    return shaped
                preview_json = json.dumps(shaped, default=str)[:200]
                print(
                    f"[llm:{r}] JSON parsed to a non-contract array "
                    f"({preview_json!r}…)",
                    file=sys.stderr, flush=True,
                )
                errors.append(f"{r}: JSON array lacks role contract")

        # Contract failure detected after transport success — demote the
        # usage entry so the report's failure column is honest (review
        # finding 5). This also covers the raw-text fallback below: text
        # that couldn't be parsed is NOT a usable contract response even
        # though the caller salvages it.
        llm_usage.mark_failed(entry, f"non-contract response ({len(content)} chars)")

        preview = content[:160].replace("\n", " ")
        print(
            f"[llm:{r}] response was not JSON-parseable "
            f"({len(content)} chars); preview: {preview!r}",
            file=sys.stderr, flush=True,
        )
        errors.append(f"{r}: non-JSON ({len(content)} chars)")
        if r == chain[-1]:
            if r != role:
                print(
                    f"[llm:{role}] using backup `{r}` text (no JSON)",
                    file=sys.stderr, flush=True,
                )
            # Raw-text salvage IS a fallback activation — flag it on the
            # final attempt's entry rather than counting an extra call.
            if entry is not None:
                entry["fallback_activated"] = True
            return {"_text": content, "_role_used": r, **fallback}

    # Every role in the chain failed to produce a usable contract —
    # heuristic fallback. Flag the final attempt (no extra call counted).
    if last_entry is not None:
        last_entry["fallback_activated"] = True
        if reservation_denied:
            last_entry["budget_reservation_exhausted"] = True
    return {
        "_error": " ; ".join(errors) or "all LLM calls failed",
        "_budget_exhausted": reservation_denied,
        **fallback,
    }


def _usage_update() -> dict[str, Any]:
    """Drain pending LLM-usage entries into a node state delta (tracker 1.4).

    Merged into every LLM-calling node's return so `budget_used` (add
    reducer) and `llm_usage` accumulate in state; nested calls made while
    the node ran (backups, vision, auto-annotation) are attributed to it.
    """
    entries = llm_usage.drain()
    if not entries:
        return {}
    update: dict[str, Any] = {
        "llm_usage": entries,
        "budget_used": sum(float(e.get("cost_usd") or 0.0) for e in entries),
        # Token fallback budget (review finding 3): always populated, so
        # runs on unpriced models still hit a runtime cap.
        "tokens_used": sum(
            int(e.get("input_tokens") or 0) + int(e.get("output_tokens") or 0)
            for e in entries
        ),
    }
    if any(bool(e.get("budget_reservation_exhausted")) for e in entries):
        update["budget_reservation_exhausted"] = True
    return update


# --------------------------------------------------------------------------- #
# Helpers shared by synthesizer / analyst / critic
# --------------------------------------------------------------------------- #

def _addr_to_int(value: Any) -> int | None:
    """Parse ``"$C145"`` / ``"0xC145"`` / ``49477`` into an int (or None)."""
    if value is None:
        return None
    if isinstance(value, int):
        return value & 0xFFFF
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            if s.startswith("$"):
                return int(s[1:], 16) & 0xFFFF
            if s.lower().startswith("0x"):
                return int(s, 16) & 0xFFFF
            return int(s) & 0xFFFF
        except ValueError:
            try:
                return int(s, 16) & 0xFFFF
            except ValueError:
                return None
    return None


_CODE_DIGEST_STOP_WORDS = frozenset({
    "about", "after", "around", "before", "begin", "caller", "control",
    "current", "does", "from", "have", "into", "player", "players",
    "produce", "provided", "remaining", "starts", "stored", "their",
    "there", "these", "value", "where", "which", "with",
})


def _code_digest_terms(question: str) -> list[str]:
    """Small stable keyword set for deterministic partial-ASM retrieval."""
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_]+", question.lower()):
        if len(token) < 4 or token in _CODE_DIGEST_STOP_WORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:8]


def _code_kb_digest_for_question(
    handle: str | None,
    question: str,
    *,
    max_chars: int = 10_000,
) -> str:
    """Return high-density excerpts from explicitly selected ASM documents.

    Partial assembly was previously ingested only into Code-KB while the
    planner/analyst digest contained only parent-KB facts. That made exact,
    human-annotated evidence invisible unless the planner happened to choose a
    ``code_kb search`` step. Retrieval here is deterministic and local: rank
    windows by distinct question-keyword density, then retain non-overlapping
    excerpts under a hard character cap.
    """
    if not handle or not question.strip():
        return ""
    terms = _code_digest_terms(question)
    if not terms:
        return ""
    try:
        from code_kb import get_code_store

        rows = get_code_store(handle).query(
            "SELECT path, content FROM asm_docs ORDER BY path",
        )
    except Exception:  # noqa: BLE001
        return ""

    candidates: list[tuple[int, str, int, int, str]] = []
    for row in rows:
        content = str(row.get("content") or "")
        lower = content.lower()
        path = str(row.get("path") or "?")
        positions: set[int] = set()
        for term in terms:
            positions.update(m.start() for m in re.finditer(re.escape(term), lower))
        for pos in sorted(positions):
            start = max(0, pos - 500)
            end = min(len(content), pos + 900)
            window_lower = lower[start:end]
            matched = sum(1 for term in terms if term in window_lower)
            occurrences = sum(window_lower.count(term) for term in terms)
            score = matched * 100 + min(occurrences, 20)
            candidates.append((score, path, start, end, content[start:end]))

        # A deliberately selected, compact excerpt may encode the answer only
        # in its instructions and addresses (for example a two-call wrapper)
        # without repeating the natural-language words from the question.
        # Keep those small source documents visible as a low-ranked fallback;
        # lexical matches still sort first and large listings remain bounded.
        compact_limit = min(4_000, max(800, max_chars // 2))
        if content.strip() and len(content) <= compact_limit:
            candidates.append((1, path, 0, len(content), content))

    chosen: list[tuple[str, int, int, str]] = []
    for _score, path, start, end, snippet in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2]),
    ):
        if any(
            path == prior_path and start < prior_end and end > prior_start
            for prior_path, prior_start, prior_end, _ in chosen
        ):
            continue
        chosen.append((path, start, end, snippet))
        if len(chosen) >= 6:
            break
    if not chosen:
        return ""

    blocks = [
        "## Question-relevant partial-assembly excerpts",
        "These are deterministic excerpts from explicitly selected ASM files. "
        "Treat their dump-consistent instructions as primary static evidence, "
        "even when the source has no natural-language labels.",
    ]
    used = sum(len(line) + 1 for line in blocks)
    for path, start, _end, snippet in chosen:
        block = (
            f"\n### {Path(path).name} (character offset {start})\n"
            f"{redact_text(snippet).strip()}"
        )
        remaining = max_chars - used
        if remaining <= 120:
            break
        if len(block) > remaining:
            block = block[: max(0, remaining - 16)].rstrip() + "\n…[truncated]"
        blocks.append(block)
        used += len(block) + 1
    return "\n".join(blocks)[:max_chars]


def _combined_kb_digest(
    state: C64State,
    store: KnowledgeStore,
    *,
    max_chars: int = 32_000,
) -> str:
    code_digest = state.get("code_kb_digest") or _code_kb_digest_for_question(
        state.get("code_kb_handle"), state.get("question", "") or "",
    )
    parent_limit = max(4_000, max_chars - len(code_digest) - 2)
    parent = store.digest_for_question(
        state.get("question", "") or "", max_chars=parent_limit,
    )
    if not code_digest:
        return parent[:max_chars]
    return (parent + "\n\n" + code_digest)[:max_chars]


def _kb_digest_for_state(state: C64State) -> str:
    """Build the combined parent-KB and selected-ASM evidence digest."""
    handle = state.get("kb_handle")
    if not handle:
        return "_(no KB handle yet)_"
    store = get_store(handle)
    return _combined_kb_digest(state, store)


def _step_for(state: C64State, step_id: str | None) -> dict[str, Any]:
    if not step_id:
        return {}
    for s in state.get("plan", []):
        if s.get("id") == step_id:
            return s
    return {}


def _slug(game: str) -> str:
    # Canonical slug lives in graph.plan_utils so main.py / app.py /
    # agent_runner.py share it (tracker 0.7).
    return slugify(game)


def _session_dir(game: str) -> Path:
    """Per-game session directory, honouring existing legacy-slug dirs.

    All session paths (kb/, code_kb/, report.md) must resolve through
    the same function or a legacy directory would receive the KB while
    the report lands in the canonical one.
    """
    return SESSIONS_DIR / resolve_session_slug(game, SESSIONS_DIR)


# Event kinds that are loop bookkeeping, not evidence. The dead-end
# detector must ignore them: the analyst and critic append one each per
# iteration, so counting them would make the KB "grow" every verdict
# and the detector could never fire.
_BOOKKEEPING_EVENT_KINDS = (
    EVT_ANALYSIS, EVT_VERDICT, EVT_CONSOLIDATED, EVT_RUN_SUMMARY,
)


def _stable_content_hash(value: Any) -> str:
    """Lossless identity for persisted evidence payloads.

    Hash the complete normalized value rather than a text prefix: expanded
    disassembly windows commonly share their opening lines while adding the
    decisive evidence near the end.
    """
    if isinstance(value, str):
        normalized = value
    else:
        normalized = json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str,
        )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _tool_result_is_substantive(payload: dict[str, Any]) -> bool:
    """Whether a successful tool result represents newly gathered evidence.

    Parent-KB calls are read-only views of facts/events already counted by
    this function; recording another query must not manufacture progress.
    Likewise Code-KB metadata/export modes are introspection, not evidence.
    Other Code-KB modes can expose code evidence from the separate layered
    store, so they remain substantive.
    """
    if not payload.get("ok"):
        return False
    tool = str(payload.get("tool") or "").strip().lower()
    mode = str(payload.get("mode") or "").strip().lower()
    if tool == "kb":
        return False
    if tool == "code_kb" and mode in {"stats", "schema", "export"}:
        return False
    return True


def _substantive_event_count(store: KnowledgeStore) -> int:
    """Count DISTINCT substantive facts in the KB (dead-end growth signal).

    Identity rules (tracker 0.2):
    - bookkeeping kinds (analysis / verdict / consolidated) never count;
    - failed and read-only/introspective tool calls never count — a retry or
      KB-query loop is not progress;
    - evidence-bearing tool results are keyed on a full normalized-content
      hash, NOT step_id, so identical replans add nothing while an expanded
      result with new evidence at the tail does count;
    - derived facts are keyed on their stable payload identity (label
      addr+name, routine start+end, hypothesis text, …), so re-emitting
      a known fact adds nothing while a genuinely new one counts.

    O(substantive events) per critic verdict — acceptable for now;
    indexing this is part of tracker 1.6.
    """
    try:
        placeholders = ",".join("?" for _ in _BOOKKEEPING_EVENT_KINDS)
        rows = store.query(
            f"SELECT kind, payload_json FROM events"
            f" WHERE kind NOT IN ({placeholders})",
            _BOOKKEEPING_EVENT_KINDS,
        )
    except Exception:  # noqa: BLE001
        return int(store.stats().get("events_total", 0))

    idents: set[tuple] = set()
    for r in rows:
        kind = r.get("kind")
        try:
            p = json.loads(r.get("payload_json") or "{}")
        except Exception:  # noqa: BLE001
            p = {}
        if not isinstance(p, dict):
            p = {}

        if kind == EVT_TOOL_RESULT:
            if not _tool_result_is_substantive(p):
                continue
            ident = (
                kind,
                str(p.get("tool") or "").strip().lower(),
                _stable_content_hash(p.get("data", "")),
            )
        elif kind == EVT_LABEL:
            ident = (kind, p.get("addr"), p.get("name"))
        elif kind == EVT_ROUTINE:
            ident = (kind, p.get("start"), p.get("end"))
        elif kind == EVT_DATA_STRUCTURE:
            ident = (kind, p.get("start"), p.get("end"), p.get("kind"))
        elif kind == EVT_HYPOTHESIS:
            ident = (kind, (p.get("text") or "").strip())
        else:
            # ingest_* and future kinds: stable payload identity with
            # per-run volatile keys stripped.
            stable = {
                k: v for k, v in p.items() if k not in ("step_id", "ts")
            }
            ident = (
                kind,
                _stable_content_hash(stable),
            )
        idents.add(ident)
    return len(idents)


def _normalize_plan_ids(plan: list[dict[str, Any]], state: C64State) -> list[dict[str, Any]]:
    """Ensure plan step IDs are unique across replans/runs.

    Planner outputs often reuse `s1/s2/s3`; that collides with historic
    `tool_results` and causes the executor to think all steps are done.
    """
    prefix = f"i{state.get('iteration', 0) + 1}_"
    normalized: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}
    used_ids: set[str] = set()

    for idx, step in enumerate(plan, start=1):
        old_id = str(step.get("id") or f"s{idx}")
        base_id = old_id if old_id.startswith(prefix) else f"{prefix}{old_id}"
        new_id = base_id
        suffix = 2
        while new_id in used_ids:
            new_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(new_id)
        # A dependency on a duplicated planner id resolves to its first
        # occurrence; later duplicates are renamed solely to keep state keys
        # collision-free (tracker 5.2 regression coverage).
        id_map.setdefault(old_id, new_id)
        s = dict(step)
        s["id"] = new_id
        s["tool"] = str(s.get("tool") or "kb")
        if not isinstance(s.get("args"), dict):
            s["args"] = {}
        normalized.append(s)

    for s in normalized:
        deps = s.get("depends_on", []) or []
        s["depends_on"] = [id_map.get(str(d), str(d)) for d in deps]

    return normalized


# --------------------------------------------------------------------------- #
# I/O nodes
# --------------------------------------------------------------------------- #

def load_inputs(state: C64State) -> dict[str, Any]:
    """Hydrate the KB: open store, ingest dump + partial asm idempotently."""
    # Run start: a reused execution context (pooled thread) must not
    # leak a prior run's un-drained usage entries into this run.
    llm_usage.reset()
    game = state.get("game", "unknown")
    kb_handle = str(_session_dir(game) / "kb")
    Path(kb_handle).mkdir(parents=True, exist_ok=True)

    store = get_store(kb_handle)

    msgs: list[str] = [f"KB ready at {kb_handle}"]

    dump_path_raw = state.get("dump_path")
    if dump_path_raw:
        dump_path = Path(dump_path_raw)
        if dump_path.exists():
            evt_id = store.ingest_dump(dump_path)
            msgs.append(
                f"Ingested dump ({dump_path.name}, "
                f"{dump_path.stat().st_size} bytes)"
                if evt_id
                else f"Dump already in KB ({dump_path.name})."
            )
        else:
            msgs.append(f"Warning: dump not found at {dump_path}")

    partial_asm_raw = state.get("partial_asm_path")
    if partial_asm_raw:
        asm_path = Path(partial_asm_raw)
        if asm_path.exists():
            evt_id = store.ingest_partial_asm(asm_path)
            msgs.append(
                f"Ingested partial asm ({asm_path.name})"
                if evt_id
                else f"Partial asm already in KB ({asm_path.name})."
            )
        else:
            msgs.append(f"Warning: partial asm not found at {asm_path}")

    text_dir_raw = state.get("text_dir")
    if text_dir_raw:
        td = Path(text_dir_raw)
        if td.is_dir():
            res = store.ingest_text_dir(td)
            if res["files_scanned"] == 0:
                msgs.append(
                    f"Text dir {td}/ has no .txt/.md/.rst files."
                )
            else:
                msgs.append(
                    f"Text dir {td}/: scanned={res['files_scanned']}, "
                    f"ingested={res['files_ingested']}, "
                    f"already-current={res['files_skipped']}"
                )
        else:
            msgs.append(f"Warning: text dir not found at {td}")

    s = store.stats()
    msgs.append(
        f"KB stats: events={s['events_total']} labels={s['labels']} "
        f"text_docs={s.get('text_docs', 0)} dump_bytes={s['dump_bytes']}"
    )

    # ---- Code KB hydration -------------------------------------------------
    # The layered code-comprehension subagent owns its own SQLite/JSONL store.
    # We populate it from the same dump + any partial-asm + everything in
    # `--asm-dir`. Failure here must NOT crash the parent agent.
    code_kb_handle, code_kb_msgs = _hydrate_code_kb(state, game)
    msgs.extend(code_kb_msgs)

    # Build the initial question-relevant digest so the planner can avoid
    # repeating work the partial-asm or text notes already answered.
    question = state.get("question", "") or ""
    code_kb_digest = _code_kb_digest_for_question(
        code_kb_handle, question, max_chars=10_000,
    )
    initial_digest = store.digest_for_question(
        question, max_chars=max(4_000, 32_000 - len(code_kb_digest) - 2),
    )
    if code_kb_digest:
        initial_digest = (initial_digest + "\n\n" + code_kb_digest)[:32_000]
    try:
        from tools.research_notebook import prior_answers_digest

        prior_answers = prior_answers_digest(
            _session_dir(game), current_question=question, max_chars=8_000,
        )
    except Exception:  # noqa: BLE001
        prior_answers = ""
    if prior_answers:
        initial_digest = (initial_digest + "\n\n" + prior_answers)[:32_000]

    prior_run_completed = bool(state.get("run_completed_at"))
    return {
        "run_id": (
            uuid.uuid4().hex
            if prior_run_completed or not state.get("run_id")
            else state["run_id"]
        ),
        "run_started_at": (
            datetime.now(timezone.utc).isoformat()
            if prior_run_completed or not state.get("run_started_at")
            else state["run_started_at"]
        ),
        "run_completed_at": None,
        "require_vice_approval": bool(
            state.get("require_vice_approval", True)
        ),
        "approved_mutation_steps": list(
            state.get("approved_mutation_steps") or []
        ),
        "approve_all_vice_mutations": bool(
            state.get("approve_all_vice_mutations", False)
        ),
        "kb_handle": kb_handle,
        "code_kb_handle": code_kb_handle,
        "code_kb_digest": code_kb_digest,
        "kb_digest": initial_digest,
        "iteration": 0,
        "replan_count": 0,
        "revise_count": 0,
        "budget_used": 0.0,
        "budget_reservation_exhausted": False,
        "tokens_used": 0,
        # Substantive (non-bookkeeping) events only — same measure the
        # critic uses, so the first verdict's growth check is honest.
        "last_kb_event_count": _substantive_event_count(store),
        "messages": [AIMessage(content=m) for m in msgs],
    }


def _hydrate_code_kb(
    state: C64State, game: str,
) -> tuple[str | None, list[str]]:
    """Initialize the layered Code Knowledge Store.

    Activated by either `--asm-dir` (a directory of partial-asm files) or
    the existing `--partial-asm` (one specific file). The parent agent
    keeps working without it; we only return a handle when at least one
    asm document or the dump made it through.
    """
    msgs: list[str] = []
    asm_dir_raw = state.get("asm_dir")
    partial_asm_raw = state.get("partial_asm_path")
    dump_path_raw = state.get("dump_path")
    asm_files_override = state.get("asm_files")  # explicit per-game list

    if (
        not asm_dir_raw and not partial_asm_raw and not asm_files_override
        and not dump_path_raw
    ):
        return None, msgs  # explicit opt-in only

    try:
        from code_kb import (
            build_from_parsed_asm, get_code_store, select_asm_files,
        )
        from code_kb.asm_parser import parse_path as _parse_asm
    except Exception as e:  # noqa: BLE001
        msgs.append(f"code_kb disabled (import failed): {e}")
        return None, msgs

    handle_path = _session_dir(game) / "code_kb"
    handle_path.mkdir(parents=True, exist_ok=True)
    code_store = get_code_store(str(handle_path))

    if dump_path_raw:
        dp = Path(dump_path_raw)
        if dp.exists():
            evt, dump_changed = code_store.ingest_dump(dp)
            if dump_changed:
                # The dump was overwritten in place (tracker 2.2):
                # disassembly windows derived from the OLD bytes are
                # stale — drop them so they get re-derived on demand.
                code_store.invalidate_source("capstone:%", like=True)
                code_store.invalidate_source("vice:%", like=True)
                msgs.append(
                    f"code_kb: dump {dp.name} CHANGED on disk — re-ingested "
                    "and invalidated stale disassembly windows."
                )
            elif evt:
                msgs.append(
                    f"code_kb: ingested dump {dp.name} ({dp.stat().st_size} bytes)."
                )

    # Game-aware asm-file scoping. Refuses to ingest cross-game files.
    extra_files: list[str] = []
    if partial_asm_raw:
        extra_files.append(str(partial_asm_raw))
    scoping = select_asm_files(
        game=game,
        asm_dir=asm_dir_raw,
        override=[str(p) for p in (asm_files_override or [])] or None,
        extra_files=extra_files or None,
    )
    if scoping.suggestion:
        msgs.append(f"code_kb scope[{scoping.strategy}]: {scoping.suggestion}")
    if scoping.skipped:
        # Cap skip-noise: top 3 examples per pollution-prone run.
        head = ", ".join(p.name for p, _ in scoping.skipped[:3])
        more = len(scoping.skipped) - 3
        msgs.append(
            "code_kb scope: skipped "
            f"{len(scoping.skipped)} file(s) "
            f"(e.g. {head}{', …' if more > 0 else ''}). "
            "If something belongs to this game, pass --asm-files."
        )

    ingested = 0
    skipped = 0
    for p in scoping.selected:
        try:
            content = p.read_text(errors="replace")
        except OSError as e:
            msgs.append(f"code_kb: cannot read {p.name}: {e}")
            continue
        # Content-aware freshness (tracker 2.2): a file edited under the
        # same name used to be skipped forever, leaving stale Layer-0 rows.
        freshness = code_store.asm_freshness(p, content)
        if freshness == "current":
            skipped += 1
            continue
        if freshness == "changed":
            code_store.invalidate_source(str(p))
            msgs.append(
                f"code_kb: {p.name} CHANGED on disk — invalidated its "
                "stale rows, re-indexing."
            )
        try:
            pf = _parse_asm(p)
        except Exception as e:  # noqa: BLE001
            msgs.append(f"code_kb: failed to parse {p.name}: {e}")
            continue
        try:
            with code_store.bulk_writes():
                stats = build_from_parsed_asm(pf, code_store)
                code_store.ingest_asm_doc(
                    p,
                    content=content,
                    instructions=stats.instructions,
                    routines=stats.routines,
                )
            ingested += 1
            msgs.append(
                f"code_kb: {p.name} → "
                f"{stats.instructions} insns / {stats.routines} routines / "
                f"{stats.xrefs} xrefs / {stats.smc_sites} SMC."
            )
        except Exception as e:  # noqa: BLE001
            msgs.append(f"code_kb: layer-0 build failed on {p.name}: {e}")

    if not scoping.selected and not code_store.has_dump():
        msgs.append("code_kb: no asm files or dump available — handle skipped.")
        return None, msgs

    cs = code_store.stats()
    msgs.append(
        f"code_kb ready at {handle_path} "
        f"(ingested={ingested}, already-current={skipped}, "
        f"routines={cs['routines']}, xrefs={cs['xrefs']}, "
        f"smc={cs['smc_sites']}, instructions={cs['instructions']})."
    )
    return str(handle_path), msgs


def _format_evidence(evidence: list[Any], store: "KnowledgeStore | None") -> str:
    from memory.evidence import format_evidence_markdown

    return format_evidence_markdown(evidence, store)


def write_report(state: C64State) -> dict[str, Any]:
    game = state.get("game", "unknown")
    out_dir = _session_dir(game)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = state.get("candidate_answer") or {}
    answer = candidate.get("answer", "(no answer)")
    confidence = candidate.get("confidence", 0.0)
    evidence = candidate.get("evidence", []) or []
    open_qs = candidate.get("open_questions", []) or []
    verdict = (state.get("verdict") or {}).get("decision", "n/a")
    critique = (state.get("verdict") or {}).get("critique", "")

    store: KnowledgeStore | None = None
    if state.get("kb_handle"):
        store = get_store(state["kb_handle"])
    s = store.stats() if store else {}

    completed_at = datetime.now(timezone.utc)
    started_at_raw = str(state.get("run_started_at") or "").strip()
    elapsed_s: float | None = None
    if started_at_raw:
        try:
            started_at = datetime.fromisoformat(started_at_raw.replace("Z", "+00:00"))
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            elapsed_s = max(0.0, (completed_at - started_at).total_seconds())
        except ValueError:
            elapsed_s = None

    run_id = str(state.get("run_id") or "").strip()
    if not run_id:
        # Direct/legacy callers that bypass load_inputs still get idempotent
        # reporting. Normal CLI/UI/Studio runs always carry a UUID.
        run_id = "legacy-" + _stable_content_hash({
            "game": game,
            "question": state.get("question", ""),
            "verdict": verdict,
            "iteration": state.get("iteration", 0),
            "answer": answer,
        })[:24]

    state_tool_results = list(state.get("tool_results", []) or [])
    report_tool_results = state_tool_results[-20:]
    if store and any(r.get("_event_id") for r in report_tool_results):
        event_ids = [
            str(r["_event_id"])
            for r in report_tool_results if r.get("_event_id")
        ]
        placeholders = ",".join("?" for _ in event_ids)
        persisted_by_id: dict[str, dict[str, Any]] = {}
        if placeholders:
            for row in store.query(
                f"SELECT id, payload_json FROM events WHERE id IN ({placeholders})",
                tuple(event_ids),
            ):
                try:
                    persisted_by_id[str(row["id"])] = json.loads(
                        row["payload_json"],
                    )
                except (TypeError, json.JSONDecodeError):
                    continue
        report_tool_results = [
            persisted_by_id.get(str(result.get("_event_id")), result)
            for result in report_tool_results
        ]

    tr_lines = []
    for r in report_tool_results:
        ok = "✓" if r.get("ok") else "✗"
        tool = r.get("tool", "?")
        step_id = r.get("step_id", "?")
        raw = str(r.get("data", ""))

        # Build a per-tool human-readable summary line + body
        summary = ""
        body = ""

        if tool == "kb":
            first_line = raw.split("\n")[0][:400]
            summary = first_line
            body = raw[:2_000]

        elif tool in ("vice", "vice.disassemble", "capstone"):
            import re as _re
            args_m = _re.search(r'args=(\{[^}]+\})', raw)
            args_str = args_m.group(1) if args_m else ""
            summary = args_str or raw[:120]
            if "disassemble" in tool or raw.strip().startswith("$"):
                body = raw[:3_000]
            else:
                hex_m = _re.search(r'"data":\s*\[([^\]]+)\]', raw, _re.DOTALL)
                if hex_m:
                    hex_bytes = [b.strip().strip('"') for b in hex_m.group(1).split(",")]
                    addr_m = _re.search(r'"address":\s*(\d+)', raw)
                    base = int(addr_m.group(1)) if addr_m else 0
                    rows_hex = []
                    for i in range(0, min(len(hex_bytes), 512), 16):
                        row_bytes = hex_bytes[i:i+16]
                        rows_hex.append(f"${base+i:04X}: {' '.join(row_bytes)}")
                    body = "\n".join(rows_hex)
                else:
                    body = raw[:2_000]

        elif tool == "tavily":
            summary = raw.split("\n")[0][:200]
            body = raw[:4_000]

        else:
            summary = raw[:200]
            body = raw[:2_000]

        # Trim body and indent for the report
        body_trimmed = body[:4_000].rstrip()
        if len(body) > 4_000:
            body_trimmed += "\n…(truncated)"
        indented = "\n".join("  " + line for line in body_trimmed.splitlines())

        tr_lines.append(
            f"- {ok} **{tool}** `{step_id}`\n"
            f"  **→** {summary}\n"
            f"  ```\n{indented}\n  ```"
        )

    plan_lines = [
        f"- `{step.get('id')}` ({step.get('tool')}): {step.get('goal')}"
        for step in state.get("plan", [])
    ]

    # ---- per-role LLM usage + budget + tool-call counters (tracker 1.4) ----
    usage_entries = state.get("llm_usage") or []
    usage_rows = llm_usage.summarize_by_role(usage_entries)
    budget_used = float(state.get("budget_used") or 0.0)
    if usage_rows:
        u_lines = [
            "| role | model | calls | failures | backup | fallback "
            "| in-tok | out-tok | est. USD |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for u in usage_rows:
            u_lines.append(
                f"| {u['role']} | {u['model']} | {u['calls']} "
                f"| {u['failures']} | {u['backup_activations']} "
                f"| {u['fallback_activations']} "
                f"| {u['input_tokens']} | {u['output_tokens']} "
                f"| {u['cost_usd']:.4f} |"
            )
        total_tok = sum(
            u["input_tokens"] + u["output_tokens"] for u in usage_rows
        )
        tokens_used = int(state.get("tokens_used") or 0) or total_tok
        priced_calls = sum(
            1 for e in usage_entries if float(e.get("cost_usd") or 0.0) > 0
        )
        unpriced_calls = len(usage_entries) - priced_calls

        # Which budget limit (if any) terminated the run — recomputed
        # against the same thresholds the router enforces.
        from graph.routers import budget_cap, token_budget
        usd_cap = budget_cap()
        token_cap = token_budget()
        enforced = []
        if budget_used >= usd_cap:
            enforced.append(f"USD cap (${usd_cap:.2f})")
        if tokens_used >= token_cap:
            enforced.append(f"token cap ({token_cap:,})")

        pricing_note = ""
        if priced_calls == 0 and total_tok > 0:
            pricing_note = (
                "\n_No `pricing` section in `config/llm.json` — USD shows "
                "0.00; the token budget above still protects the run._"
            )
        usage_block = (
            "\n".join(u_lines)
            + "\n\n**Budget:**\n"
            + f"- total tokens: {tokens_used:,} / {token_cap:,} (token cap)\n"
            + f"- estimated cost: ${budget_used:.4f} / ${usd_cap:.2f} (USD cap)\n"
            + f"- calls priced/unpriced: {priced_calls}/{unpriced_calls}\n"
            + f"- budget limit enforced: {', '.join(enforced) or 'none'}"
            + pricing_note
        )
    else:
        usage_block = "_no LLM usage recorded_"

    durable_tool_stats = state.get("tool_call_stats") or {}
    tool_counts: dict[str, list[int]] = {
        str(tool): [
            int(values.get("calls", 0)),
            int(values.get("failures", 0)),
        ]
        for tool, values in durable_tool_stats.items()
    }
    if not tool_counts:
        for r in state_tool_results:
            t = str(r.get("tool") or "?")
            row = tool_counts.setdefault(t, [0, 0])
            row[0] += 1
            if not r.get("ok"):
                row[1] += 1
    tool_lines = [
        f"- {t}: {n} call(s), {f} failure(s)"
        for t, (n, f) in sorted(tool_counts.items())
    ] or ["_none_"]

    report = f"""# C64-RE Report — {game}

**Question:** {state.get("question", "")}

**Run:** `{run_id}`  ·  **Verdict:** `{verdict}`  ·  **Confidence:** {confidence}  ·  **Iterations:** {state.get("iteration", 0)}

## Answer

{answer}

## Open questions

{chr(10).join(f"- {q}" for q in open_qs) if open_qs else "_none_"}

## Evidence

{_format_evidence(evidence, store)}

## Critic notes

{critique or "_none_"}

## Final plan

{chr(10).join(plan_lines) if plan_lines else "_no plan recorded_"}

## Tool results (latest 20)

{chr(10).join(tr_lines) if tr_lines else "_none_"}

## LLM usage (per role)

{usage_block}

## Tool calls

{chr(10).join(tool_lines)}

## KB stats

```
{json.dumps(s, indent=2)}
```

---
_generated {completed_at.isoformat()}_
"""

    from tools.research_notebook import (
        append_turn,
        versioned_report_path,
    )
    from memory.evidence import resolve_evidence

    report_path = out_dir / "report.md"
    report_version_path = versioned_report_path(
        out_dir, started_at=started_at_raw, run_id=run_id,
    )
    report_version_path.write_text(report)
    report_path.write_text(report)
    append_turn(out_dir, {
        "run_id": run_id,
        "game": str(game),
        "completed_at": completed_at.isoformat(),
        "question": str(state.get("question") or ""),
        "answer": str(answer),
        "confidence": confidence,
        "elapsed_s": elapsed_s,
        "verdict": str(verdict),
        "critique": str(critique or ""),
        "evidence": [str(item) for item in evidence],
        "resolved_evidence": resolve_evidence(evidence, store),
        "open_questions": [str(item) for item in open_qs],
        "addresses": sorted({
            int(address) for address in extract_hex_addresses(
                str(answer) + "\n" + "\n".join(map(str, evidence)),
            )
        }),
        "report_path": str(report_version_path),
    })

    summary_event_id: str | None = None
    summary_error: str | None = None
    if store is not None:
        try:
            try:
                summary_confidence = float(confidence)
            except (TypeError, ValueError):
                summary_confidence = 0.0
            usage_entries = state.get("llm_usage") or []
            total_tokens = int(state.get("tokens_used") or 0) or sum(
                int(e.get("input_tokens") or 0)
                + int(e.get("output_tokens") or 0)
                for e in usage_entries
            )
            summary_event_id = store.record_run_summary({
                "run_id": run_id,
                "completed_at": completed_at.isoformat(),
                "question": str(state.get("question") or ""),
                "verdict": str(verdict),
                "confidence": summary_confidence,
                "iterations": int(state.get("iteration") or 0),
                "cost_usd": float(state.get("budget_used") or 0.0),
                "tokens": total_tokens,
                "llm_calls": len(usage_entries),
                "tool_calls": tool_call_total(state),
                "elapsed_s": elapsed_s,
                "termination_reason": state.get("termination_reason"),
                "answer_excerpt": str(answer)[:1_000],
                "report_path": str(report_path),
            })
        except Exception as e:  # noqa: BLE001
            # The report is the primary user artifact; telemetry failure must
            # not turn a successfully completed research run into a graph error.
            summary_error = f"{type(e).__name__}: {e}"
            print(
                f"[write_report] could not persist run summary: {summary_error}",
                file=sys.stderr,
            )

    message = (
        f"Wrote report → {report_path} "
        f"(version={report_version_path.name})"
    )
    if summary_event_id:
        message += f" (run_summary={summary_event_id})"
    elif summary_error:
        message += " (run_summary failed; see stderr)"
    return {
        "run_completed_at": completed_at.isoformat(),
        "messages": [AIMessage(content=message)],
    }


# --------------------------------------------------------------------------- #
# LLM sub-agent nodes
# --------------------------------------------------------------------------- #

def planner_node(state: C64State) -> dict[str, Any]:
    # Surface the current KB digest + the critic's most recent feedback
    # so the planner can avoid duplicate work and incorporate suggested
    # follow-ups verbatim. The role-specific contract (tool args, ordering
    # rules) lives in `graph.prompts.PLANNER_ROLE` and is applied by
    # `_invoke_one` via the system message.
    #
    # Reuse the digest refreshed by the synthesizer/load_inputs instead of
    # rebuilding from SQL — it is at most one step stale (tracker 1.5).
    digest = state.get("kb_digest") or _kb_digest_for_state(state)
    last_verdict = state.get("verdict") or {}
    suggested = last_verdict.get("suggested_steps") or []
    critique = last_verdict.get("critique") or ""

    suggested_block = ""
    if suggested:
        suggested_block = (
            "Critic-suggested follow-up steps (incorporate verbatim "
            "near the front of your plan, renumbered):\n"
            f"{json.dumps(suggested, indent=2)}\n\n"
        )
    critique_block = (
        f"Most recent critic feedback:\n{critique}\n\n" if critique else ""
    )

    # When the previous plan blocked on unsatisfiable dependencies the
    # planner must see why, or it will re-emit the same broken shape
    # (tracker 0.6). The structured failures are in `tool_results`.
    blocked_block = ""
    if state.get("plan_blocked"):
        blocked = [
            r for r in state.get("tool_results", []) or []
            if r.get("rejection") == "dependency_unsatisfied"
        ][-8:]
        if blocked:
            lines = [
                f"- {r.get('step_id')}: {str(r.get('data', ''))[:220]}"
                for r in blocked
            ]
            blocked_block = (
                "**The previous plan BLOCKED** — these steps had failed or "
                "unsatisfiable dependencies. Your new plan MUST reference "
                "only step ids it itself contains in `depends_on`, and must "
                "not depend on tools/steps that failed permanently:\n"
                + "\n".join(lines) + "\n\n"
            )

    vice_mcp = os.getenv("VICE_MCP_URL", "").strip()
    if vice_mcp:
        vice_runtime = (
            "**Runtime — VICE MCP is configured** (`VICE_MCP_URL` is set). "
            "VICE is already confirmed reachable — do **NOT** emit `vice.ping` "
            "or `vice.registers.get` steps; those waste budget and produce no "
            "code facts. `vice.display.screenshot` is allowed when spatial/visual "
            "context would directly help answer the question. "
            "For any plan that needs instruction-level disassembly at "
            "concrete `$XXXX` addresses, order tools as: **vice first** "
            "(`method: vice.disassemble` with explicit `address` + `count`), "
            "then **capstone `linear`** on the same region with "
            "`depends_on` the vice step id — unless the KB digest already "
            "contains identical evidence.\n\n"
        )
    else:
        vice_runtime = (
            "**Runtime — VICE MCP is NOT configured** (no `VICE_MCP_URL`). "
            "Do **not** emit `vice` tool steps; use `capstone` and `kb` only.\n\n"
        )

    prompt = (
        f"Game: {state.get('game')}\n"
        f"Question: {state.get('question')}\n"
        f"Iteration: {state.get('iteration', 0)}  ·  "
        f"replan_count: {state.get('replan_count', 0)}\n\n"
        f"{vice_runtime}"
        f"{blocked_block}"
        f"{critique_block}"
        f"{suggested_block}"
        "KB schema (use these EXACT column names if you emit raw SQL —\n"
        "column-name drift like `address` instead of `addr` will be\n"
        "auto-rewritten where possible, but the round-trip wastes a step):\n"
        "----- BEGIN KB SCHEMA -----\n"
        f"{KB_SCHEMA_HINT}\n"
        "----- END KB SCHEMA -----\n\n"
        "Current KB digest (already-known facts — DO NOT repeat steps that\n"
        "would re-derive these):\n"
        "----- BEGIN DIGEST -----\n"
        f"{digest}\n"
        "----- END DIGEST -----\n\n"
        "Reply with JSON only — either a list of steps OR "
        '{"plan": [...]}. Follow your role contract exactly.'
    )
    q_keyword = (state.get("question") or "").split()[0][:8] or "rand"
    fallback_plan = [
        {"id": "s1", "goal": "Search user-provided notes for the question keywords",
         "tool": "kb",
         "args": {"mode": "text", "q": q_keyword},
         "depends_on": [], "hypothesis": None},
        {"id": "s2", "goal": "Find game-loop candidates via heuristics",
         "tool": "capstone", "args": {"mode": "find_loops", "top_n": 8},
         "depends_on": [], "hypothesis": None},
        {"id": "s3", "goal": "Detect entry point + RAM/HW vectors",
         "tool": "capstone", "args": {"mode": "find_entry"},
         "depends_on": [], "hypothesis": None},
        {"id": "s4", "goal": "Recursive disassembly from BASIC SYS / detected entry",
         "tool": "capstone",
         "args": {"mode": "recursive", "entry": "0x0801", "max_insns": 600},
         "depends_on": ["s3"], "hypothesis": None},
        {"id": "s5", "goal": "Search public C64 docs for context",
         "tool": "tavily",
         "args": {"q": (state.get("question") or "") + " Commodore 64"},
         "depends_on": [], "hypothesis": None},
        {"id": "s6", "goal": "Look up labels relevant to the question",
         "tool": "kb",
         "args": {"mode": "labels", "like": q_keyword},
         "depends_on": [], "hypothesis": None},
        {"id": "s7", "goal": "Summarize accumulated KB",
         "tool": "kb", "args": {"mode": "stats"},
         "depends_on": ["s1", "s2", "s3", "s4", "s5", "s6"],
         "hypothesis": None},
    ]
    out = _safe_invoke(
        "planner", prompt, {"plan": fallback_plan},
        backup_roles=["analyst", "executor", "critic"],
        allow_plan_steps_array=True,
        remaining_budget_usd=_remaining_llm_budget(state),
    )
    if isinstance(out, dict) and out.get("_budget_exhausted"):
        # Do not turn an exhausted invocation into the seven-step heuristic
        # fallback plan: that would re-enter tools/curator and repeatedly
        # attempt calls that the reservation guard must continue rejecting.
        plan_raw = []
    elif isinstance(out, list):
        plan_raw = out
    elif isinstance(out, dict):
        plan_raw = out.get("plan") or fallback_plan
    else:
        plan_raw = fallback_plan
    plan = _normalize_plan_ids(plan_raw, state)

    # Enforce the plan-size cap RECURSION_LIMIT is derived from
    # (tracker 0.1). Preserve dependency references exactly: if a retained
    # step depended on a dropped/later step, the executor must diagnose the
    # now-missing dependency and replan. Silently deleting the edge would
    # make the step runnable with unresolved inputs (tracker 0.6).
    truncated_note = ""
    if len(plan) > MAX_PLAN_STEPS:
        dropped = len(plan) - MAX_PLAN_STEPS
        plan = plan[:MAX_PLAN_STEPS]
        truncated_note = (
            f" (truncated: dropped {dropped} step(s) beyond the "
            f"{MAX_PLAN_STEPS}-step cap; dependency references preserved)"
        )

    # If we are replanning (we already have a verdict), bump replan_count
    # so the dead-end detector can react.
    new_replan_count = state.get("replan_count", 0)
    if state.get("verdict"):
        new_replan_count += 1

    return {
        "plan": plan,
        "plan_blocked": False,
        "iteration": state.get("iteration", 0) + 1,
        "replan_count": new_replan_count,
        "messages": [AIMessage(
            content=f"Planner produced {len(plan)} steps.{truncated_note}",
        )],
        **_usage_update(),
    }


def executor_node(state: C64State) -> dict[str, Any]:
    """Pick the next runnable step and ask the executor LLM to enrich its args.

    Two phases:

    1. **Mechanical selection** — respects ``depends_on`` so a step whose
       dependencies are not yet done is never dispatched early.
    2. **LLM enrichment** — the executor LLM receives the selected step
       (with its current, potentially ``null``-filled args) and the live
       KB digest.  It returns the same contract shape but with args filled
       in from KB-discovered facts (e.g. an address found by a previous
       Capstone step, a search term narrowed by a Tavily result, etc.).
       The enriched args are merged back into the plan so every downstream
       tool node sees them.
    """
    tool_results = state.get("tool_results", []) or []
    status = step_status(tool_results)
    plan = list(state.get("plan", []) or [])

    # A step is "done" only when it succeeded or its retries are
    # exhausted — `ok=False` results used to permanently burn the step
    # (tracker 0.3). Runnable steps come back untried-first so a flaky
    # step being retried never starves fresh work.
    runnable = runnable_steps(state)

    parallel = parallel_runnable_steps(state)
    if parallel:
        step_ids = [str(step["id"]) for step in parallel]
        return {
            "current_step_id": None,
            "current_step_ids": step_ids,
            "plan_blocked": False,
            "messages": [AIMessage(content=(
                "Executor selected read-only fan-out: "
                + ", ".join(
                    f"{step['id']}→{step['tool']}" for step in parallel
                )
                + "."
            ))],
            **_usage_update(),
        }

    if not runnable:
        pending = [s for s in plan if str(s.get("id")) not in status.done]
        if not pending:
            return {
                "current_step_id": None,
                "current_step_ids": [],
                "plan_blocked": False,
            }
        # Blocked plan: every pending step has an unsatisfied dependency
        # (a prerequisite that failed permanently, a dep id missing from
        # the plan, or a dependency cycle). The old fallback dispatched
        # pending[0] anyway — usually with unresolved null args producing
        # plausible-but-wrong results. Instead: record a structured
        # failure per blocked step and raise `plan_blocked`, which
        # `route_tool` turns into a deterministic replan (bounded by
        # MAX_ITERS) rather than letting the analyst/critic accept an
        # answer built on a broken plan (tracker 0.6).
        plan_ids = {str(s.get("id")) for s in plan}
        blocked_results: list[dict[str, Any]] = []
        blocked_ids: list[str] = []
        for s in pending:
            sid = str(s.get("id"))
            blocked_ids.append(sid)
            reasons: list[str] = []
            for d in (s.get("depends_on") or []):
                d = str(d)
                if d in status.succeeded:
                    continue
                if d in status.exhausted:
                    reasons.append(f"`{d}` failed permanently")
                elif d not in plan_ids:
                    reasons.append(f"`{d}` is missing from the plan")
                else:
                    reasons.append(f"`{d}` can never run (dependency cycle)")
            blocked_results.append({
                "step_id": sid,
                "tool": s.get("tool", "?"),
                "ok": False,
                "retryable": False,
                "rejection": "dependency_unsatisfied",
                "data": (
                    f"dependency_unsatisfied: step {sid} is blocked — "
                    + ("; ".join(reasons) or "no runnable prerequisite")
                    + ". Plan aborted for replan: the next plan must "
                    "reference only steps it contains and must not "
                    "depend on permanently-failed work."
                ),
            })
        return {
            "current_step_id": None,
            "current_step_ids": [],
            "plan_blocked": True,
            "tool_results": blocked_results,
            "tool_call_stats": {
                str(tool): {
                    "calls": sum(
                        1 for result in blocked_results
                        if str(result.get("tool") or "?") == str(tool)
                    ),
                    "failures": sum(
                        1 for result in blocked_results
                        if str(result.get("tool") or "?") == str(tool)
                    ),
                }
                for tool in {
                    str(result.get("tool") or "?")
                    for result in blocked_results
                }
            },
            "messages": [AIMessage(content=(
                f"Executor: plan blocked — {len(pending)} step(s) with "
                f"unsatisfiable dependencies ({', '.join(blocked_ids)}); "
                "routing to replan."
            ))],
        }

    nxt = runnable[0]
    step_id = nxt["id"]

    # When retrying a previously-failed step, show the executor LLM the
    # last error so it can repair the args instead of repeating them.
    prev_failures = [
        r for r in tool_results
        if str(r.get("step_id")) == str(step_id) and not r.get("ok")
    ]

    # LLM bypass (tracker 1.1): most planner steps arrive with complete,
    # concrete args — enrichment has nothing to do for them. Dispatch
    # directly and skip the executor LLM call (and its 8k-char digest
    # prompt). Retries always go through the LLM so it can repair the
    # args using the recorded failure.
    if not prev_failures and step_is_concrete(nxt):
        return {
            "current_step_id": step_id,
            "current_step_ids": [],
            "plan_blocked": False,
            "messages": [AIMessage(content=(
                f"Executor selected step {step_id} → {nxt['tool']} "
                "(LLM skipped — args already concrete)."
            ))],
            **_usage_update(),
        }

    retry_block = ""
    if prev_failures:
        last_err = str(prev_failures[-1].get("data", ""))[:600]
        retry_block = (
            f"NOTE: this step already FAILED {len(prev_failures)} time(s). "
            "Adjust the args so the retry can succeed.\n"
            f"Last error:\n{last_err}\n\n"
        )

    # Ask the executor LLM to enrich / validate the args using current KB
    # state. Reuse the synthesizer-refreshed digest (at most one step
    # stale; the prompt truncates it to 8k anyway — tracker 1.5).
    kb_snippet = state.get("kb_digest") or _kb_digest_for_state(state)
    prompt = (
        f"Question (context): {state.get('question')}\n\n"
        f"Selected step (next to execute):\n{json.dumps(nxt, default=str)}\n\n"
        f"{retry_block}"
        f"Current KB digest (facts discovered so far):\n{kb_snippet[:8_000]}\n\n"
        "Fill in any null arg values using facts from the KB digest above, "
        "then emit JSON exactly per your role contract: "
        '{"step_id": str, "tool": str, "args": {...}, "rationale": str}'
    )
    out = _safe_invoke(
        "executor", prompt,
        fallback={
            "step_id": step_id,
            "tool": nxt.get("tool", "kb"),
            "args": nxt.get("args") or {},
            "rationale": "fallback — LLM unavailable",
        },
        remaining_budget_usd=_remaining_llm_budget(state),
    )

    # Merge enriched args back into the plan step (never clobber id/tool/deps).
    enriched_args = out.get("args")
    if isinstance(enriched_args, dict) and enriched_args:
        updated_plan = []
        for s in plan:
            if s.get("id") == step_id:
                merged = {**(s.get("args") or {}), **{
                    k: v for k, v in enriched_args.items() if v is not None
                }}
                s = {**s, "args": merged}
            updated_plan.append(s)
    else:
        updated_plan = plan

    return {
        "current_step_id": step_id,
        "current_step_ids": [],
        "plan": updated_plan,
        "plan_blocked": False,
        "messages": [
            AIMessage(
                content=(
                    f"Executor selected step {step_id} → {nxt['tool']}. "
                    f"Rationale: {out.get('rationale', '—')}"
                )
            )
        ],
        **_usage_update(),
    }


# Batched LLM synthesis (tracker 1.2): the extraction LLM runs once per
# batch instead of once per step. A flush happens when the plan has no
# pending steps (the analyst runs next and must see extracted facts) or
# when the accumulated LLM-worthy material crosses either bound.
SYNTH_BATCH_MAX_RESULTS = 4
SYNTH_BATCH_MAX_CHARS = 20_000

# Layer-1 auto-annotation budget per synthesizer invocation (tracker 1.3).
# Previously every extracted routine with confidence ≥ 0.5 fired a full
# hidden LLM call — six routines meant six invisible calls.
MAX_AUTO_ANNOTATE_PER_SYNTH = 2

# Capstone modes whose structured `extra` payloads are extracted
# mechanically — the LLM adds nothing to them.
_MECHANICAL_CAPSTONE_MODES = {
    "find_entry", "entry", "vectors",
    "find_counters", "counters", "idioms",  # tracker 3.3 / 3.5
    "screen_text", "screen", "bank", "banking",  # tracker 3.6 / 3.7
}

# VICE composite methods whose structured `extra` payloads are extracted
# mechanically (tracker 3.1). `vice.trace` stays LLM-worthy — its body is
# prose disassembly the analyst reasons over.
_MECHANICAL_VICE_METHODS = {
    "vice.memory.diff", "vice.memory.monotonic_scan",
}
# VICE results that carry no extractable facts at all (status only).
_NOFACT_VICE_METHODS = {"vice.memory.snapshot"}


def _llm_worthy_result(r: dict[str, Any]) -> bool:
    """Only free-text evidence needs the extraction LLM (tracker 1.2).

    Failed results carry no facts; `kb`/`code_kb` results are reads of
    stores whose facts are already persisted; mechanical capstone/vice
    modes are handled deterministically. What remains — disassembly
    listings, vice.trace output, tavily snippets, screenshot
    descriptions — is prose.
    """
    if not r.get("ok"):
        return False
    tool = str(r.get("tool") or "").lower()
    if tool in ("kb", "code_kb"):
        return False
    if (
        tool == "capstone"
        and str(r.get("mode") or "").lower() in _MECHANICAL_CAPSTONE_MODES
    ):
        return False
    if tool == "vice":
        method = str(r.get("method") or "").lower()
        if method in _MECHANICAL_VICE_METHODS or method in _NOFACT_VICE_METHODS:
            return False
    return True


def _mechanical_extract(
    store: KnowledgeStore, result: dict[str, Any], ev_id: str,
) -> dict[str, int]:
    """Deterministically persist facts from structured tool outputs.

    Runs only for freshly recorded events (stage-1 dedup guarantees
    each result is seen once), so it cannot duplicate facts. Emitted
    payloads carry ``provenance: "mechanical"``.
    """
    counts = {"labels": 0, "hypotheses": 0}
    if not result.get("ok"):
        return counts
    tool = str(result.get("tool") or "").lower()
    if tool == "vice":
        return _mechanical_extract_vice(store, result, ev_id)
    if tool != "capstone":
        return counts
    mode = str(result.get("mode") or "").lower()

    if mode == "vectors":
        info = result.get("info") or {}
        for group in ("hw_vectors", "ram_vectors"):
            for row in info.get(group) or []:
                tgt = _addr_to_int(row.get("target"))
                name = str(row.get("name") or "").strip()
                if tgt is None or tgt == 0 or not name:
                    continue
                slug_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
                store.append_event(EVT_LABEL, "synthesizer_mechanical", {
                    "addr": tgt,
                    "name": f"{slug_name}_target",
                    "kind": "vector",
                    "confidence": 0.9,
                    "evidence": (
                        f"{group} entry {row.get('vector')} (event {ev_id})"
                    ),
                    "provenance": "mechanical",
                })
                counts["labels"] += 1

    elif mode in ("find_loops", "loops"):
        for c in (result.get("candidates") or [])[:3]:
            addr = _addr_to_int(c.get("address"))
            if addr is None:
                continue
            reasons = "; ".join(str(x) for x in (c.get("reasons") or [])[:4])
            candidate_type = str(
                c.get("candidate_type") or "loop",
            ).replace("_", " ")
            store.append_event(EVT_HYPOTHESIS, "synthesizer_mechanical", {
                # Deterministic id: re-detection updates instead of duplicating.
                "id": f"h_loop_{addr:04x}",
                "text": (
                    f"{candidate_type.title()} candidate at ${addr:04X} "
                    f"(score {c.get('score')}): {reasons}"
                ),
                "status": "open",
                "evidence": [ev_id],
                "provenance": "mechanical",
                "candidate_type": c.get("candidate_type"),
                "metrics": c.get("metrics") or {},
            })
            counts["hypotheses"] += 1

    elif mode in ("find_counters", "counters"):
        # Top game-state candidates → ONE ram_var label per address
        # (tracker 3.3). Deterministic name keyed on (kind, addr) so
        # re-runs on the same dump are idempotent (label PK (addr,name)).
        for c in (result.get("candidates") or [])[:8]:
            addr = _addr_to_int(c.get("address"))
            if addr is None:
                continue
            aliases = c.get("alias_kinds") or []
            ev = "; ".join(str(x) for x in (c.get("evidence") or [])[:3])
            if aliases:
                ev += f"  (also matched: {', '.join(aliases)})"
            store.append_event(EVT_LABEL, "synthesizer_mechanical", {
                "addr": addr,
                "name": f"candidate_{c.get('kind', 'var')}_{addr:04x}",
                "kind": "ram_var",
                "confidence": min(0.75, 0.4 + float(c.get("score", 0)) / 60.0),
                "evidence": ev,
                "provenance": "mechanical",
            })
            counts["labels"] += 1

    elif mode == "idioms":
        for f in (result.get("idioms") or [])[:32]:
            addr = _addr_to_int(f.get("address"))
            if addr is None:
                continue
            store.append_event(EVT_HYPOTHESIS, "synthesizer_mechanical", {
                "id": f"h_idiom_{addr:04x}_{f.get('idiom', 'x')}",
                "text": (
                    f"Idiom `{f.get('idiom')}` at ${addr:04X}: "
                    f"{f.get('evidence')}"
                ),
                "status": "open",
                "evidence": [ev_id],
                "provenance": "mechanical",
            })
            counts["hypotheses"] += 1

    elif mode in ("screen_text", "screen"):
        # Printable screen runs → text labels (tracker 3.7). Name keyed
        # on the address so a re-decode of the same screen is idempotent.
        for run in (result.get("runs") or [])[:32]:
            addr = _addr_to_int(run.get("addr") or run.get("addr_hex"))
            text = str(run.get("text") or "").strip()
            if addr is None or not text:
                continue
            store.append_event(EVT_LABEL, "synthesizer_mechanical", {
                "addr": addr,
                "name": f"screen_text_{addr:04x}",
                "kind": "text",
                "confidence": 0.7,
                "evidence": (
                    f"screen RAM row {run.get('row')} col {run.get('col')}: "
                    f"{text!r}"
                ),
                "provenance": "mechanical",
            })
            counts["labels"] += 1

    return counts


# Region-priority for diff short-listing: game state lives in these,
# most-likely first. I/O is already excluded upstream.
_DIFF_REGION_PRIORITY = {
    "zero_page": 0, "os_workspace": 1, "low_ram": 2, "upper_ram": 3,
    "screen_ram": 4, "color_ram": 5,
}


def _mechanical_extract_vice(
    store: KnowledgeStore, result: dict[str, Any], ev_id: str,
) -> dict[str, int]:
    """Deterministic extraction for VICE composite outputs (tracker 3.1)."""
    counts = {"labels": 0, "hypotheses": 0}
    method = str(result.get("method") or "").lower()

    if method == "vice.memory.monotonic_scan":
        delta = int(result.get("delta", -1))
        kind = {-1: "lives", 1: "counter"}.get(delta, "delta_var")
        for c in (result.get("candidates") or [])[:16]:
            addr = _addr_to_int(c.get("addr") if "addr" in c else c.get("addr_hex"))
            if addr is None:
                continue
            values = c.get("values") or []
            store.append_event(EVT_LABEL, "synthesizer_mechanical", {
                "addr": addr,
                "name": f"candidate_{kind}_{addr:04x}",
                "kind": "ram_var",
                # Strong dynamic signal, but capped — it's still a
                # candidate until confirmed against static evidence.
                "confidence": 0.7,
                "evidence": (
                    f"monotonic Δ{delta:+d} across snapshots "
                    f"[{', '.join(str(v) for v in values)}] "
                    f"in {c.get('region')} (event {ev_id})"
                ),
                "provenance": "mechanical",
            })
            counts["labels"] += 1

    elif method == "vice.memory.diff":
        # Short-list only the top state-region changed bytes — never dump
        # thousands of bytes into the KB.
        changed = list(result.get("changed") or [])
        changed.sort(key=lambda r: (
            _DIFF_REGION_PRIORITY.get(r.get("region"), 99),
            r.get("addr", 0),
        ))
        for c in changed[:12]:
            addr = _addr_to_int(c.get("addr") if "addr" in c else c.get("addr_hex"))
            region = str(c.get("region") or "")
            if addr is None or region.startswith("io_"):
                continue
            store.append_event(EVT_LABEL, "synthesizer_mechanical", {
                "addr": addr,
                "name": f"changed_{addr:04x}",
                "kind": "ram_var",
                "confidence": 0.5,
                "evidence": (
                    f"changed {c.get('old')}→{c.get('new')} "
                    f"(Δ{int(c.get('delta', 0)):+d}) in {region} "
                    f"(diff {result.get('a')}→{result.get('b')})"
                ),
                "provenance": "mechanical",
            })
            counts["labels"] += 1

    return counts


def _question_relevance(question: str, text: str) -> float:
    """Cheap word-overlap score in [0, 1] for annotate prioritisation."""
    q_words = {w for w in re.findall(r"[a-z0-9]{3,}", (question or "").lower())}
    t_words = {w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower())}
    if not q_words or not t_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


def synthesizer_node(state: C64State) -> dict[str, Any]:
    """Mechanical dedup + deterministic extraction + batched LLM extraction.

    Stages (tracker 1.2):

    1. **Dedup/record**: write previously-unrecorded `tool_results` as
       `EVT_TOOL_RESULT` events — indexed content-hash lookup, not a
       full log scan (tracker 1.6).
    1.5 **Mechanical extraction**: structured capstone outputs
       (`vectors`, `find_loops`) become labels/hypotheses with
       ``provenance: mechanical`` — no LLM.
    2. **Batched LLM extraction**: free-text results (disassembly,
       vice, tavily) accumulate across steps and are extracted in ONE
       LLM call when the plan drains or the batch fills. Failed and
       `kb`/`code_kb` results never reach the LLM.

    The extractor is the single biggest reason the multi-agent system
    can outperform a flat one-shot prompt: the KB accumulates *typed*
    knowledge across iterations rather than raw textual blobs.
    """
    if not state.get("kb_handle"):
        return {"messages": [AIMessage(content="Synthesizer: no KB handle.")]}

    store = get_store(state["kb_handle"])
    all_results = identify_tool_results(state.get("tool_results", []) or [])

    # ---- Stage 1: idempotent recording of raw tool_results ----
    new_raw: list[dict[str, Any]] = []
    new_event_ids: list[str] = []
    event_id_by_result: dict[str, str] = {}
    for r in all_results:
        result_id = str(r["_state_result_id"])
        compact_event_id = str(r.get("_event_id") or "")
        if compact_event_id:
            event_id_by_result[result_id] = compact_event_id
            continue
        existing_event_id = store.tool_result_event_id(r)
        if existing_event_id:
            event_id_by_result[result_id] = existing_event_id
            continue
        event_id = store.append_event(EVT_TOOL_RESULT, "synthesizer", r)
        new_event_ids.append(event_id)
        event_id_by_result[result_id] = event_id
        new_raw.append(r)

    # ---- Stage 1.5: mechanical extraction (structured outputs) ----
    mech_counts = {"labels": 0, "hypotheses": 0}
    for ev_id, r in zip(new_event_ids, new_raw):
        for k, v in _mechanical_extract(store, r, ev_id).items():
            mech_counts[k] += v

    # ---- Stage 2: batched LLM extraction of structured facts ----
    result_ids = [str(r["_state_result_id"]) for r in all_results]
    prior_processed_ids = {
        str(result_id)
        for result_id in (state.get("synth_processed_result_ids") or [])
    }
    if not prior_processed_ids:
        # Durable checkpoints written before tracker 4.8 carry only the
        # numeric prefix cursor. Migrate it once into stable identities.
        legacy_processed = max(
            0, min(int(state.get("synth_processed_count", 0) or 0), len(all_results)),
        )
        prior_processed_ids.update(result_ids[:legacy_processed])
    unprocessed = [
        result for result in all_results
        if str(result["_state_result_id"]) not in prior_processed_ids
    ]
    worthy = [r for r in unprocessed if _llm_worthy_result(r)]
    worthy_chars = sum(len(str(r.get("data", ""))) for r in worthy)
    # The analyst runs next once no step needs a (re)run — extraction
    # must flush before it reads the digest.
    plan_drained = not pending_steps(state)

    flush = bool(worthy) and (
        plan_drained
        or len(worthy) >= SYNTH_BATCH_MAX_RESULTS
        or worthy_chars >= SYNTH_BATCH_MAX_CHARS
    )

    extracted_counts = {"labels": 0, "routines": 0, "data_structures": 0,
                        "hypotheses": 0}
    notes_msg = ""
    annotate_notes: list[str] = []
    new_processed_count: int | None = None
    processed_ids = set(prior_processed_ids)
    if not worthy and unprocessed:
        # Nothing LLM-worthy in the tail — mark it considered.
        new_processed_count = len(all_results)
        processed_ids.update(
            str(result["_state_result_id"]) for result in unprocessed
        )

    if flush:
        new_processed_count = len(all_results)
        processed_ids.update(
            str(result["_state_result_id"]) for result in unprocessed
        )
        # Compact view of the batched tool outputs for the extractor.
        # (Results recorded in earlier deferred passes are labelled by
        # tool/step_id; their event ids live in the KB.)
        per_result_budget = max(
            4_000, SYNTH_BATCH_MAX_CHARS // max(1, len(worthy)),
        )
        observations: list[str] = []
        for r in worthy:
            data = str(r.get("data", ""))
            if len(data) > per_result_budget:
                data = data[:per_result_budget] + "\n... [truncated]"
            observations.append(
                f"### result {r.get('tool')}/{r.get('step_id')} "
                f"(ok={r.get('ok')})\n"
                f"extra={json.dumps({k: v for k, v in r.items() if k not in {'data', 'tool', 'step_id', 'ok'}}, default=str)[:2_000]}\n"
                f"data:\n{data}"
            )

        prompt = (
            f"Question (background only): {state.get('question')}\n\n"
            f"New tool outputs ({len(worthy)} results, batched) to "
            "synthesise into structured KB facts:\n\n"
            + "\n\n".join(observations)
            + "\n\nReply JSON only, exactly per your role contract."
        )
        out = _safe_invoke(
            "synthesizer", prompt,
            fallback={"labels": [], "routines": [], "data_structures": [],
                      "hypotheses": [], "notes": ""},
            backup_roles=["analyst", "planner", "executor"],
            remaining_budget_usd=_remaining_llm_budget(state),
        )

        # Layer-1 annotation candidates queued by the routines loop and
        # executed (capped + prioritised) after all extraction loops.
        annotate_candidates: list[tuple[float, int]] = []

        # Persist extracted facts. We're permissive about input shape
        # (LLMs occasionally emit "address" instead of "addr", etc.).

        # Resolve code_kb store once for the whole batch (may be None).
        # LLM-discovered labels/routines are mirrored here so the web GUI
        # call graph and routines panel show them alongside parser output.
        _code_store = None
        if state.get("code_kb_handle"):
            try:
                from code_kb import get_code_store as _get_code_store
                from code_kb.schema import (
                    ANN_HYPOTHESIS as _ANN_HYPOTHESIS,
                    ANN_LABEL as _ANN_LABEL,
                    ANN_ROUTINE as _ANN_ROUTINE,
                )
                from code_kb.store import Annotation as _Ann
                _code_store = _get_code_store(state["code_kb_handle"])
            except Exception:  # noqa: BLE001
                pass

        def _layer0_window(start: int, end: int | None = None) -> dict[str, Any] | None:
            if _code_store is None:
                return None
            end = start if end is None else end
            rows = _code_store.query(
                "SELECT id, start_addr, end_addr, payload_json"
                " FROM annotations WHERE layer = 0 AND kind = ?"
                " AND start_addr <= ? AND end_addr >= ?"
                " ORDER BY (end_addr - start_addr) ASC LIMIT 1",
                (_ANN_ROUTINE, start, end),
            )
            return rows[0] if rows else None

        for lab in out.get("labels") or []:
            addr = _addr_to_int(lab.get("addr") or lab.get("address"))
            name = (lab.get("name") or "").strip()
            if addr is None or not name:
                continue
            confidence = float(lab.get("confidence", 0.5))
            store.append_event(EVT_LABEL, "synthesizer", {
                "addr": addr,
                "name": name,
                "kind": lab.get("kind", "code"),
                "confidence": confidence,
                "evidence": lab.get("evidence"),
                "provenance": "llm",
            })
            # Mirror into code_kb so node labels appear in the call graph.
            if _code_store is not None:
                l0 = _layer0_window(addr)
                if l0 is not None:
                    _code_store.append_annotation(
                        _Ann(
                            layer=1,
                            kind=_ANN_LABEL,
                            start_addr=addr,
                            end_addr=addr,
                            producer="synthesizer",
                            confidence=confidence,
                            payload={
                                "name": name,
                                "source_file": "llm_synthesizer",
                            },
                            evidence=[str(lab.get("evidence") or "")],
                        ),
                        source="synthesizer",
                    )
                else:
                    _code_store.append_annotation(
                        _Ann(
                            layer=1,
                            kind=_ANN_HYPOTHESIS,
                            start_addr=addr,
                            end_addr=addr,
                            producer="synthesizer",
                            confidence=confidence,
                            payload={
                                "text": f"Unverified label suggestion: {name}",
                                "name_suggestion": name,
                                "source_file": "llm_synthesizer",
                            },
                            evidence=[str(lab.get("evidence") or "")],
                            flags=["unverified_llm"],
                        ),
                        source="synthesizer",
                    )
            extracted_counts["labels"] += 1

        for rt in out.get("routines") or []:
            start = _addr_to_int(rt.get("start"))
            end = _addr_to_int(rt.get("end") or rt.get("start"))
            if start is None or end is None:
                continue
            calls_to = [
                _addr_to_int(c) for c in (rt.get("calls_to") or [])
            ]
            calls_to = [c for c in calls_to if c is not None]
            called_by = [
                _addr_to_int(c) for c in (rt.get("called_by") or [])
            ]
            called_by = [c for c in called_by if c is not None]
            confidence = float(rt.get("confidence", 0.5))
            l0 = _layer0_window(start, end)
            if l0 is not None:
                store.append_event(EVT_ROUTINE, "synthesizer", {
                    "start": start,
                    "end": end,
                    "name": rt.get("name"),
                    "summary": rt.get("summary"),
                    "calls_to": calls_to,
                    "called_by": called_by,
                    "confidence": confidence,
                    "provenance": "llm_with_layer0_window",
                    "layer0_routine_id": str(l0["id"]),
                })
            else:
                # A semantic routine guess without deterministic bounds must
                # not materialise as parent-KB routine/call-graph truth.
                store.append_event(EVT_HYPOTHESIS, "synthesizer", {
                    "id": f"h_unverified_routine_{start:04x}",
                    "text": rt.get("summary") or (
                        f"Possible routine {rt.get('name') or 'purpose unknown'} "
                        f"at ${start:04X}-${end:04X}."
                    ),
                    "status": "open",
                    "evidence": [f"${start:04X}-${end:04X}"],
                    "name_suggestion": rt.get("name"),
                    "start": start,
                    "end": end,
                    "provenance": "unverified_llm",
                })

            # Mirror semantic claims into Code-KB Layer 1, but never invent
            # call-graph ground truth. Claims with no enclosing deterministic
            # Layer-0 routine remain queryable and carry `unverified_llm`.
            existing_conf = 0.0
            if _code_store is not None and l0 is not None:
                # Capture the prior semantic confidence before appending this
                # turn's hypothesis; otherwise the new row suppresses its own
                # eligible annotation candidate below.
                prior = _code_store.query(
                    "SELECT confidence FROM hypotheses"
                    " WHERE start_addr = ? AND layer = 1"
                    " ORDER BY confidence DESC LIMIT 1",
                    (int(l0["start_addr"]),),
                )
                existing_conf = float(prior[0]["confidence"]) if prior else 0.0
            if _code_store is not None:
                if l0 is not None:
                    l0_start = int(l0["start_addr"])
                    l0_end = int(l0["end_addr"])
                    flags: list[str] = []
                    routine_id = str(l0["id"])
                else:
                    l0_start = start
                    l0_end = end
                    flags = ["unverified_llm"]
                    routine_id = None
                _code_store.append_annotation(
                    _Ann(
                        layer=1,
                        kind=_ANN_HYPOTHESIS,
                        start_addr=l0_start,
                        end_addr=l0_end,
                        producer="synthesizer",
                        confidence=confidence,
                        payload={
                            "text": rt.get("summary") or (
                                f"Possible routine {rt.get('name') or 'purpose unknown'}"
                            ),
                            "name_suggestion": rt.get("name"),
                            "routine_id": routine_id,
                            "source_file": "llm_synthesizer",
                        },
                        evidence=[rt.get("summary") or ""],
                        flags=flags,
                    ),
                    source="synthesizer",
                )

            extracted_counts["routines"] += 1

            # Queue Layer-1 annotation candidates instead of firing a
            # hidden LLM call per routine (tracker 1.3). Candidates are
            # prioritised and capped after the extraction loops.
            if _code_store is not None and l0 is not None and confidence >= 0.50:
                try:
                    candidate_start = int(l0["start_addr"])
                    if confidence > existing_conf:
                        relevance = _question_relevance(
                            state.get("question", "") or "",
                            f"{rt.get('name') or ''} {rt.get('summary') or ''}",
                        )
                        annotate_candidates.append(
                            (confidence * (1.0 + relevance), candidate_start),
                        )
                except Exception:
                    pass

        for ds in out.get("data_structures") or []:
            start = _addr_to_int(ds.get("start"))
            end = _addr_to_int(ds.get("end") or ds.get("start"))
            if start is None or end is None:
                continue
            store.append_event(EVT_DATA_STRUCTURE, "synthesizer", {
                "start": start,
                "end": end,
                "kind": ds.get("kind"),
                "fields": ds.get("fields") or {},
                "provenance": "llm",
            })
            extracted_counts["data_structures"] += 1

        for hyp in out.get("hypotheses") or []:
            text = (hyp.get("text") or "").strip()
            if not text:
                continue
            store.append_event(EVT_HYPOTHESIS, "synthesizer", {
                "id": hyp.get("id") or f"h{store.stats().get('events_total', 0)}",
                "text": text,
                "status": hyp.get("status", "open"),
                "evidence": hyp.get("evidence") or [],
                "provenance": "llm",
            })
            extracted_counts["hypotheses"] += 1

        if out.get("notes"):
            notes_msg = " " + str(out.get("notes"))[:500]

        # Run the queued Layer-1 annotations — highest-priority first,
        # capped per invocation, and logged so the calls are visible in
        # the transcript instead of silent (tracker 1.3).
        if annotate_candidates and _code_store is not None:
            ranked = sorted(annotate_candidates, key=lambda t: -t[0])
            for _prio, a_start in ranked[:MAX_AUTO_ANNOTATE_PER_SYNTH]:
                addr_str = f"${a_start & 0xFFFF:04X}"
                try:
                    from graph.code_kb_node import _mode_annotate
                    ann_out = _mode_annotate(
                        state, _code_store,
                        {
                            "start": addr_str,
                            "auto_disasm_if_missing": True,
                        },
                        f"auto_ann_{a_start & 0xFFFF:04X}",
                    )
                except Exception as e:  # noqa: BLE001
                    annotate_notes.append(
                        f"auto-annotation of {addr_str} crashed: "
                        f"{type(e).__name__}"
                    )
                    continue

                # `_mode_annotate` returns through `_record_result`, whose
                # own drain moved the Layer-1 usage entries into this
                # (otherwise discarded) return value — put them back into
                # the collector so THIS node's drain ships them to state
                # (review finding 1: they were silently thrown away).
                for u in (ann_out or {}).get("llm_usage") or []:
                    llm_usage.record(u)

                ann_results = (ann_out or {}).get("tool_results") or []
                if ann_results and ann_results[0].get("ok"):
                    annotate_notes.append(f"auto-annotated {addr_str}")
                else:
                    detail = (
                        str(ann_results[0].get("data", ""))[:120]
                        if ann_results else "no result returned"
                    ).replace("\n", " ")
                    annotate_notes.append(
                        f"auto-annotation of {addr_str} FAILED: {detail}"
                    )
            skipped_ann = len(ranked) - MAX_AUTO_ANNOTATE_PER_SYNTH
            if skipped_ann > 0:
                annotate_notes.append(
                    f"{skipped_ann} annotation candidate(s) deferred "
                    f"(cap {MAX_AUTO_ANNOTATE_PER_SYNTH}/pass)"
                )

    # Refresh the digest only when the KB gained extractable facts or the
    # analyst runs next; on a pure defer the digest stays as-is (at most
    # one batch stale for the executor prompt — tracker 1.5).
    update: dict[str, Any] = {}
    if flush or plan_drained or any(mech_counts.values()) or not all_results:
        update["kb_digest"] = _combined_kb_digest(state, store)
    if new_processed_count is not None:
        update["synth_processed_count"] = new_processed_count
    bounded_processed_ids = [
        result_id for result_id in result_ids if result_id in processed_ids
    ]
    if bounded_processed_ids != list(
        state.get("synth_processed_result_ids") or [],
    ):
        update["synth_processed_result_ids"] = bounded_processed_ids
    compactable = {
        result_id: event_id_by_result[result_id]
        for result_id in bounded_processed_ids
        if event_id_by_result.get(result_id)
    }
    if compactable:
        update["tool_results"] = compact_tool_results_update(compactable)

    deferred = 0 if flush else len(worthy)
    s = store.stats()
    summary = (
        f"Synthesizer: +{len(new_raw)} tool_result(s), "
        f"+{mech_counts['labels']}L/+{mech_counts['hypotheses']}H mechanical, "
        + (
            f"LLM batch of {len(worthy)}: "
            f"+{extracted_counts['labels']}L "
            f"+{extracted_counts['routines']}R "
            f"+{extracted_counts['data_structures']}DS "
            f"+{extracted_counts['hypotheses']}H"
            if flush
            else f"LLM deferred ({deferred} result(s) batched)"
        )
        + f" · KB events={s['events_total']}."
        + notes_msg
        + (" · " + "; ".join(annotate_notes) if annotate_notes else "")
    )
    return {
        **update,
        "messages": [AIMessage(content=summary)],
        **_usage_update(),
    }


def curator_node(state: C64State) -> dict[str, Any]:
    """Compact verbose tool_result events into a single summary event.

    The router only routes here when the volume of UNCOMPACTED
    tool_result events exceeds `UNCOMPACTED_TOKEN_THRESHOLD`
    (tracker 2.1). Each pass consumes the OLDEST uncompacted events;
    bookkeeping is by EXACT event ids (`events_compacted` → the derived
    `compacted_tool_results` table), never by timestamp cursor, so
    equal-timestamp events can't be skipped and consecutive passes never
    re-summarise the same events (the old code always took the newest 20
    by timestamp). Once the backlog is compacted the gate closes instead
    of taxing every future synthesizer step forever. `through_ts` on the
    payload is informational only.
    """
    if not state.get("kb_handle"):
        return {"messages": [AIMessage(content="Curator: no KB handle.")]}

    store = get_store(state["kb_handle"])
    rows = store.uncompacted_tool_results(limit=20)
    if not rows:
        return {"messages": [AIMessage(content="Curator: nothing to compact.")]}

    inputs = []
    event_ids = []
    through_ts = None
    for r in rows:
        try:
            p = json.loads(r["payload_json"])
        except Exception:  # noqa: BLE001
            continue
        event_ids.append(r["id"])
        through_ts = max(through_ts or r["ts"], r["ts"])
        data = str(p.get("data", ""))[:4_000]
        inputs.append(
            f"### event {r['id']}  ({p.get('tool')}/{p.get('step_id')})\n{data}"
        )
    if not event_ids:
        return {"messages": [AIMessage(content="Curator: nothing to compact.")]}

    prompt = (
        f"Question (background): {state.get('question')}\n\n"
        "Compact the following events per your role contract:\n\n"
        + "\n\n".join(inputs)
    )
    out = _safe_invoke(
        "curator", prompt,
        fallback={"summary": "", "addresses_kept": [],
                  "events_compacted": event_ids},
        backup_roles=["analyst", "planner", "critic"],
        remaining_budget_usd=_remaining_llm_budget(state),
    )

    summary = (out.get("summary") or "").strip() or out.get("_text", "")
    if summary:
        # `through_ts` / `events_compacted` are OUR mechanically-derived
        # values, never the LLM's — the high-water mark must be exact.
        # When the LLM fails entirely (no summary) nothing is written,
        # so no events are marked compacted without a real summary.
        store.append_event(EVT_CONSOLIDATED, "curator", {
            "summary": summary,
            "addresses_kept": out.get("addresses_kept") or [],
            "events_compacted": event_ids,
            "through_ts": through_ts,
        })

    # Raw checkpoint copies are independently compacted by the synthesizer's
    # bounded result reducer (tracker 4.8). This curator owns semantic KB
    # compaction; the two mechanisms deliberately have separate provenance.
    return {
        "kb_digest": _combined_kb_digest(state, store),
        "messages": [
            AIMessage(content=f"Curator compacted {len(event_ids)} events."),
        ],
        **_usage_update(),
    }


def analyst_node(state: C64State) -> dict[str, Any]:
    """Draft an answer grounded in the question-focused KB digest.

    The previous implementation built a small ad-hoc context (LIMIT 20
    labels, last 8 truncated tool results); the analyst frequently
    answered without the routines / hypotheses / partial-asm it needed.
    We now hand it a single rich digest produced by
    `KnowledgeStore.digest_for_question`, the same one the critic sees,
    so the two agents argue over the same evidence.
    """
    store = get_store(state["kb_handle"]) if state.get("kb_handle") else None
    s = store.stats() if store else {}

    # Prefer a digest already computed by the synthesizer/curator this
    # iteration. Fall back to building one fresh.
    digest = state.get("kb_digest") or _kb_digest_for_state(state)

    # On a `revise` loop the previous critic feedback is the most
    # important context — without it the analyst would just re-emit the
    # same answer.
    last_verdict = state.get("verdict") or {}
    last_decision = last_verdict.get("decision")
    revision_block = ""
    if last_decision == "revise" and last_verdict.get("critique"):
        revision_block = (
            "You are RE-DRAFTING because the critic returned `revise`. "
            "Address every point in the critique below; do NOT request "
            "new tool calls (the critic decided the existing KB suffices).\n"
            f"Critic critique:\n{last_verdict['critique']}\n\n"
            "Your previous answer (for reference):\n"
            f"{json.dumps(state.get('candidate_answer') or {}, indent=2, default=str)[:4_000]}\n\n"
        )

    # Surface permanently-failed steps so the analyst never reads the
    # absence of that evidence as negative evidence (tracker 0.3).
    failed_notes = failed_step_notes(state)
    failed_block = ""
    if failed_notes:
        failed_block = (
            "The following plan steps FAILED — their evidence was NEVER "
            "gathered. Do NOT treat its absence as negative evidence; "
            "mention material gaps in `open_questions`:\n"
            + "\n".join(failed_notes) + "\n\n"
        )

    prompt = (
        f"Question: {state.get('question')}\n\n"
        f"{revision_block}"
        f"{failed_block}"
        "KB digest (the COMPLETE evidence available to you — do not invent "
        "anything outside it):\n"
        "----- BEGIN DIGEST -----\n"
        f"{digest}\n"
        "----- END DIGEST -----\n\n"
        "Reply JSON ONLY per your role contract."
    )

    fallback_answer_lines = [
        "I couldn't reach an LLM, so this is a heuristic summary based on the KB:",
        "",
        f"- KB events: {s.get('events_total', 0)}",
        f"- Labels learned: {s.get('labels', 0)}",
        f"- Dump bytes available: {s.get('dump_bytes', 0)}",
        "",
        "Digest preview (first 800 chars):",
        digest[:800],
    ]
    fallback = {
        "answer": "\n".join(fallback_answer_lines),
        "evidence": [],
        "confidence": 0.3,
        "open_questions": [
            "Primary analyst LLM unreachable — re-run when reachable.",
        ],
    }
    # Backup chain: prefer roles whose providers differ from the primary
    # so a Claude-side outage doesn't take everything down.
    out = _safe_invoke(
        "analyst", prompt, fallback,
        backup_roles=["critic", "synthesizer", "planner", "executor"],
        remaining_budget_usd=_remaining_llm_budget(state),
    )

    candidate = {k: out.get(k, fallback[k]) for k in fallback}
    role_used = out.get("_role_used")

    # If the LLM responded but couldn't be parsed as the expected JSON
    # object, prefer its freeform text as the answer instead of the
    # canned heuristic message.
    raw_text = out.get("_text")
    if raw_text and candidate["answer"] == fallback["answer"]:
        candidate["answer"] = raw_text.strip()
        candidate.setdefault("confidence", 0.5)
        candidate["open_questions"] = list(candidate.get("open_questions") or []) + [
            f"Analyst LLM ({role_used or 'unknown'}) returned text instead of "
            "JSON — used as answer verbatim."
        ]

    # Surface call failures so the user notices them in the final report.
    err = out.get("_error")
    if err:
        candidate["open_questions"] = list(candidate.get("open_questions") or []) + [
            f"All analyst LLM attempts failed: {err}"
        ]
        candidate["answer"] = (
            f"_(analyst LLM unavailable: {err})_\n\n" + candidate["answer"]
        )

    if role_used and role_used != "analyst" and not err:
        candidate["open_questions"] = list(candidate.get("open_questions") or []) + [
            f"Analyst answered via backup LLM `{role_used}` "
            f"(primary `analyst` failed or returned empty)."
        ]

    if store:
        store.append_event(EVT_ANALYSIS, "analyst", candidate)

    suffix = f" (via {role_used})" if role_used and role_used != "analyst" else ""
    return {
        "candidate_answer": candidate,
        "messages": [
            AIMessage(
                content=f"Analyst confidence={candidate['confidence']}{suffix}."
            )
        ],
        **_usage_update(),
    }


def critic_node(state: C64State) -> dict[str, Any]:
    """True adversarial LLM critic.

    Reviews the analyst's answer against the *same* KB digest the
    analyst saw, plus the candidate answer itself. Decides one of:

    * ``accept``  — well-grounded, no critical gaps;
    * ``revise``  — answer can be fixed in place from existing KB;
    * ``replan``  — needs fresh tool calls; ``suggested_steps`` is
                    populated and fed straight to the planner.

    The previous heuristic ``confidence >= 0.8 OR iteration >= 2`` rule
    is kept as a safety net for when the LLM itself is unreachable (so
    the loop still terminates).
    """
    iteration = state.get("iteration", 0)
    candidate = state.get("candidate_answer") or {}
    confidence = float(candidate.get("confidence", 0.0))

    digest = state.get("kb_digest") or _kb_digest_for_state(state)
    plan_json = json.dumps(state.get("plan") or [], default=str)[:4_000]

    prompt = (
        f"Original question: {state.get('question')}\n\n"
        f"Iteration: {iteration}  ·  replans so far: "
        f"{state.get('replan_count', 0)}\n\n"
        "Analyst's candidate answer (review this critically):\n"
        "----- BEGIN ANSWER -----\n"
        f"{json.dumps(candidate, indent=2, default=str)}\n"
        "----- END ANSWER -----\n\n"
        "Current plan (already-executed steps may be visible in the digest):\n"
        f"{plan_json}\n\n"
        "Same KB digest the analyst used:\n"
        "----- BEGIN DIGEST -----\n"
        f"{digest}\n"
        "----- END DIGEST -----\n\n"
        "Reply JSON ONLY per your role contract."
    )

    # Heuristic fallback if the LLM is unreachable — preserve old loop
    # termination behaviour so we never hang on a flaky provider.
    heuristic_decision = (
        "accept" if (confidence >= 0.8 or iteration >= 2) else "replan"
    )
    fallback = {
        "decision": heuristic_decision,
        "critique": (
            f"Critic LLM unreachable. Heuristic decision={heuristic_decision} "
            f"(confidence={confidence}, iteration={iteration})."
        ),
        "suggested_steps": [],
    }

    out = _safe_invoke(
        "critic", prompt, fallback,
        backup_roles=["analyst", "planner", "synthesizer"],
        remaining_budget_usd=_remaining_llm_budget(state),
    )

    decision = out.get("decision") or fallback["decision"]
    if decision not in {"accept", "revise", "replan"}:
        decision = fallback["decision"]

    critique = out.get("critique") or fallback["critique"]
    suggested = out.get("suggested_steps") or []
    optional_followups = list(out.get("optional_followups") or [])
    refuted_hypotheses = list(out.get("refuted_hypotheses") or [])

    # `suggested_steps` are BLOCKING tool work by contract; non-blocking
    # ideas belong in `optional_followups` (which accompanies `accept`
    # untouched). An `accept` or `revise` carrying suggested_steps is
    # inconsistent — the required work would otherwise silently vanish —
    # so both escalate to `replan` and the planner consumes the steps
    # (tracker 0.5).
    if suggested and decision == "accept":
        decision = "replan"
        critique += (
            "\n\n_(auto-guardrail: `accept` + `suggested_steps` — "
            "suggested steps are blocking tool work, so escalated to "
            "`replan`; use `optional_followups` for non-blocking ideas.)_"
        )
    elif suggested and decision == "revise":
        decision = "replan"
        critique += (
            "\n\n_(auto-guardrail: `revise` + `suggested_steps` — "
            "escalated to `replan` so the suggested tool calls run.)_"
        )

    # Late guardrail: very-high-confidence answer with no evidence cited
    # is suspicious — bump to revise unless the critic explicitly accepted.
    ev = (candidate.get("evidence") or [])
    if confidence >= 0.85 and not ev and decision == "accept":
        decision = "revise"
        critique += (
            "\n\n_(auto-guardrail: confidence ≥ 0.85 but evidence list is "
            "empty; demoted to `revise`.)_"
        )

    # Revise cap — applied LAST so no guardrail above can keep the
    # analyst↔critic ping-pong alive forever (tracker 0.4).
    revise_count = int(state.get("revise_count", 0))
    if decision == "revise" and revise_count >= MAX_CONSECUTIVE_REVISES:
        decision = "accept"
        critique += (
            f"\n\n_(auto-guardrail: {revise_count} consecutive `revise` "
            "verdicts — forcing `accept`; the remaining concerns above "
            "are recorded in the report.)_"
        )
    new_revise_count = revise_count + 1 if decision == "revise" else 0

    verdict = {
        "decision": decision,
        "critique": critique,
        "suggested_steps": suggested,
        "refuted_hypotheses": refuted_hypotheses,
        "optional_followups": optional_followups,
        "_role_used": out.get("_role_used"),
    }

    # Persist verdict + snapshot the substantive KB event count. The
    # growth comparison must run against the snapshot taken at the
    # PREVIOUS verdict, *before* we overwrite it — the old code wrote
    # the fresh count into the same state update the dead-end router
    # then read, so "KB grew" was always false (tracker 0.2).
    prev_count = int(state.get("last_kb_event_count", 0))
    new_count = prev_count
    if state.get("kb_handle"):
        store = get_store(state["kb_handle"])
        new_count = _substantive_event_count(store)
        verdict["kb_grew_since_last_verdict"] = new_count > prev_count
        store.append_event(EVT_VERDICT, "critic", verdict)
        if decision == "accept":
            store.transition_hypotheses(
                candidate.get("evidence") or [],
                status="supported",
                source="critic_accept",
                evidence=candidate.get("evidence") or [],
            )
        if refuted_hypotheses:
            store.transition_hypotheses(
                refuted_hypotheses,
                status="refuted",
                source="critic_contrary_evidence",
                evidence=candidate.get("evidence") or [],
            )
    else:
        # Unknown growth must never terminate a possibly-productive replan.
        verdict["kb_grew_since_last_verdict"] = True

    updates: dict[str, Any] = {
        "verdict": verdict,
        "history": [verdict],
        "last_kb_event_count": new_count,
        "revise_count": new_revise_count,
    }

    # On accept, surface the optional follow-ups in the final answer's
    # open questions so they reach the report instead of vanishing.
    if decision == "accept" and optional_followups:
        cand = dict(state.get("candidate_answer") or {})
        open_qs = list(cand.get("open_questions") or [])
        for f in optional_followups:
            text = f.get("goal") if isinstance(f, dict) else str(f)
            if text:
                open_qs.append(f"Follow-up suggested by critic: {text}")
        cand["open_questions"] = open_qs
        updates["candidate_answer"] = cand

    role_used = verdict.get("_role_used")
    suffix = f" (via {role_used})" if role_used and role_used != "critic" else ""
    updates["messages"] = [
        AIMessage(content=f"Critic verdict: {decision}{suffix}."),
    ]
    updates.update(_usage_update())
    return updates


# --------------------------------------------------------------------------- #
# Tool nodes — real implementations with graceful fallback
# --------------------------------------------------------------------------- #

def _record_result(
    tool: str,
    step_id: str,
    ok: bool,
    data: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {"step_id": step_id, "tool": tool, "ok": ok, "data": data}
    if extra:
        result.update(extra)
    badge = "✓" if ok else "✗"
    snippet = str(data)[:120].replace("\n", " ")
    return {
        "tool_results": [result],
        "tool_call_stats": {
            str(tool): {"calls": 1, "failures": 0 if ok else 1},
        },
        "messages": [
            AIMessage(content=f"[tool:{tool}] {badge} step={step_id} — {snippet}")
        ],
        # Tool nodes are LLM-free except the vice screenshot vision call;
        # draining here attributes that usage to the tool result (1.4).
        **_usage_update(),
    }


def _hex_to_int(v: Any, default: int = 0) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        v = v.strip()
        try:
            if v.startswith("$"):
                return int(v[1:], 16)
            if v.lower().startswith("0x"):
                return int(v, 16)
            return int(v)
        except ValueError:
            pass
    return default


def _vice_normalize_method(raw_method: str) -> str:
    """Map common aliases to real vice-mcp tool names."""
    m = (raw_method or "").strip().lower()
    aliases = {
        # ---- ping ----
        "ping": "vice.ping",
        # ---- disassemble ----
        "disassemble": "vice.disassemble",
        "vice.disassemble": "vice.disassemble",
        # ---- memory ----
        "read_memory": "vice.memory.read",
        "memory.read": "vice.memory.read",
        "memory_read": "vice.memory.read",
        "vice.memory.read": "vice.memory.read",
        "write_memory": "vice.memory.write",
        "memory.write": "vice.memory.write",
        "memory_write": "vice.memory.write",
        "poke": "vice.memory.write",
        "vice.memory.write": "vice.memory.write",
        "fill_memory": "vice.memory.fill",
        "memory.fill": "vice.memory.fill",
        "memory_fill": "vice.memory.fill",
        "vice.memory.fill": "vice.memory.fill",
        # ---- agent-side memory composites ----
        "snapshot": "vice.memory.snapshot",
        "memory.snapshot": "vice.memory.snapshot",
        "vice.memory.snapshot": "vice.memory.snapshot",
        "diff": "vice.memory.diff",
        "memory.diff": "vice.memory.diff",
        "vice.memory.diff": "vice.memory.diff",
        "monotonic_scan": "vice.memory.monotonic_scan",
        "memory.monotonic_scan": "vice.memory.monotonic_scan",
        "vice.memory.monotonic_scan": "vice.memory.monotonic_scan",
        "trace": "vice.trace",
        "vice.trace": "vice.trace",
        "poke_verify": "vice.poke_verify",
        "poke_and_peek": "vice.poke_verify",
        "vice.poke_verify": "vice.poke_verify",
        # ---- registers ----
        "registers": "vice.registers.get",
        "registers_get": "vice.registers.get",
        "registers.get": "vice.registers.get",
        "get_registers": "vice.registers.get",
        "vice.registers": "vice.registers.get",
        "vice.registers.get": "vice.registers.get",
        # ---- execution ----
        "pause": "vice.execution.pause",
        "execution.pause": "vice.execution.pause",
        "run": "vice.execution.run",
        "execution.run": "vice.execution.run",
        "step": "vice.execution.step",
        "execution.step": "vice.execution.step",
        "execution.reset": "vice.execution.reset",
        "vice.execution.reset": "vice.execution.reset",
        # ---- machine/media mutation ----
        "reset": "vice.machine.reset",
        "machine.reset": "vice.machine.reset",
        "vice.reset": "vice.machine.reset",
        "vice.machine.reset": "vice.machine.reset",
        "autostart": "vice.autostart",
        "machine.autostart": "vice.autostart",
        "vice.autostart": "vice.autostart",
        "attach_disk": "vice.disk.attach",
        "disk.attach": "vice.disk.attach",
        "disk_attach": "vice.disk.attach",
        "vice.disk.attach": "vice.disk.attach",
        "detach_disk": "vice.disk.detach",
        "disk.detach": "vice.disk.detach",
        "disk_detach": "vice.disk.detach",
        "vice.disk.detach": "vice.disk.detach",
        "attach_tape": "vice.tape.attach",
        "tape.attach": "vice.tape.attach",
        "vice.tape.attach": "vice.tape.attach",
        "detach_tape": "vice.tape.detach",
        "tape.detach": "vice.tape.detach",
        "vice.tape.detach": "vice.tape.detach",
        "attach_cartridge": "vice.cartridge.attach",
        "cartridge.attach": "vice.cartridge.attach",
        "vice.cartridge.attach": "vice.cartridge.attach",
        "detach_cartridge": "vice.cartridge.detach",
        "cartridge.detach": "vice.cartridge.detach",
        "vice.cartridge.detach": "vice.cartridge.detach",
        "snapshot.load": "vice.snapshot.load",
        "vice.snapshot.load": "vice.snapshot.load",
        "resources.set": "vice.resources.set",
        "vice.resources.set": "vice.resources.set",
        # ---- breakpoints ----
        "checkpoint_add": "vice.checkpoint.add",
        "breakpoint": "vice.checkpoint.add",
        "checkpoint.add": "vice.checkpoint.add",
        "vice.checkpoint.add": "vice.checkpoint.add",
        "checkpoint_delete": "vice.checkpoint.delete",
        "checkpoint.delete": "vice.checkpoint.delete",
        "vice.checkpoint.delete": "vice.checkpoint.delete",
        # ---- injected input ----
        "keyboard.type": "vice.keyboard.type",
        "keyboard_type": "vice.keyboard.type",
        "vice.keyboard.type": "vice.keyboard.type",
        "joystick.set": "vice.joystick.set",
        "joystick_set": "vice.joystick.set",
        "vice.joystick.set": "vice.joystick.set",
        # ---- screenshots ----
        "screenshot": "vice.display.screenshot",
        "display.screenshot": "vice.display.screenshot",
        "vice.screenshot": "vice.display.screenshot",
        # ---- VIC-II / SID / CIA state ----
        "vice.vicii": "vice.vicii.get_state",
        "vice.vic": "vice.vicii.get_state",
        "vice.sid": "vice.sid.get_state",
        "vice.cia": "vice.cia.get_state",
        "vice.cia1": "vice.cia.get_state",
        # ---- memory search/compare ----
        "vice.memory.search": "vice.memory.search",
        "memory.search": "vice.memory.search",
        "vice.memory_search": "vice.memory.search",
    }
    if m in aliases:
        return aliases[m]
    if m.startswith("vice."):
        return m
    # Last resort: treat as vice.<name>.
    return f"vice.{m}" if m else "vice.ping"


_MUTATING_VICE_METHODS = frozenset({
    "vice.trace",
    "vice.poke_verify",
    "vice.memory.write",
    "vice.memory.fill",
    "vice.execution.run",
    "vice.execution.step",
    "vice.execution.pause",
    "vice.execution.reset",
    "vice.machine.reset",
    "vice.autostart",
    "vice.disk.attach",
    "vice.disk.detach",
    "vice.tape.attach",
    "vice.tape.detach",
    "vice.tape.control",
    "vice.cartridge.attach",
    "vice.cartridge.detach",
    "vice.cartridge.freeze",
    "vice.snapshot.load",
    "vice.resources.set",
    "vice.checkpoint.add",
    "vice.checkpoint.delete",
    "vice.keyboard.type",
    "vice.joystick.set",
})


def vice_method_requires_approval(method: str) -> bool:
    """True when a VICE method can change emulator state."""
    canonical = _vice_normalize_method(method)
    return canonical in _MUTATING_VICE_METHODS or canonical.startswith((
        "vice.memory.write", "vice.memory.poke", "vice.keyboard.",
        "vice.joystick.",
    ))


def vice_step_requires_approval(step: dict[str, Any]) -> bool:
    args = step.get("args") or {}
    method = args.get("method") or step.get("method") or step.get("action") or ""
    return vice_method_requires_approval(str(method))


# Address-key aliases the planner LLM keeps inventing: shared constant
# `plan_utils.VICE_ADDRESS_ALIASES` (imported above as
# `_VICE_ADDRESS_ALIASES`) — `step_is_concrete` uses the same list.


def _coerce_vice_address(raw: Any) -> str | None:
    """Normalise an address-like value to ``$XXXX`` (or return None).

    Accepts: int, ``"$XXXX"``, ``"0xXXXX"``, ``"XXXX"`` (hex), or a
    decimal string that fits in 16 bits. The previous helper silently
    coerced ``0`` and empty strings into ``$0801``; we no longer do that
    — the caller decides how to react to a missing address.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None  # True/False shouldn't be treated as addresses
    if isinstance(raw, int):
        return f"${raw & 0xFFFF:04X}"
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        if s.startswith("$"):
            try:
                return f"${int(s[1:], 16) & 0xFFFF:04X}"
            except ValueError:
                return None
        if s.lower().startswith("0x"):
            try:
                return f"${int(s, 16) & 0xFFFF:04X}"
            except ValueError:
                return None
        # Bare hex-looking strings (e.g. "1135") — accept as hex.
        try:
            return f"${int(s, 16) & 0xFFFF:04X}"
        except ValueError:
            pass
        # Last resort: decimal.
        try:
            return f"${int(s) & 0xFFFF:04X}"
        except ValueError:
            return None
    return None


def _extract_vice_address(a: dict[str, Any]) -> str | None:
    """Find the first non-empty address-like key in ``a`` and canonicalise it."""
    for key in _VICE_ADDRESS_ALIASES:
        if key in a:
            canon = _coerce_vice_address(a[key])
            if canon is not None:
                return canon
    return None


class ViceArgsError(ValueError):
    """Raised when a vice-mcp call is missing required arguments."""


def _vice_normalize_args(method: str, args: dict[str, Any]) -> dict[str, Any]:
    """Translate generic args into the current vice-mcp schema.

    Raises ``ViceArgsError`` when a required arg is missing — previously
    these silently defaulted to ``$0801`` / ``$0000``, which produced
    plausible-looking but completely wrong tool results.
    """
    a = dict(args or {})

    if method == "vice.disassemble":
        address = _extract_vice_address(a)
        if address is None:
            raise ViceArgsError(
                "vice.disassemble requires an `address` argument "
                "(e.g. \"$1135\"). Accepted aliases: "
                + ", ".join(_VICE_ADDRESS_ALIASES) + "."
            )
        if address == "$0000":
            raise ViceArgsError(
                "vice.disassemble was given address $0000 — this is almost "
                "certainly a placeholder that was never resolved. "
                "Check the KB for the real target address and retry."
            )
        count = a.get("count")
        if count is None:
            length = _hex_to_int(a.get("length", 0), 0)
            count = max(1, min(100, (length // 2) if length else 16))
        return {
            "address": address,
            "count": int(max(1, min(100, int(count)))),
            "show_symbols": bool(a.get("show_symbols", True)),
        }

    if method == "vice.memory.read":
        address = _extract_vice_address(a)
        if address is None:
            raise ViceArgsError(
                "vice.memory.read requires an `address` argument "
                "(e.g. \"$03F0\")."
            )
        size = a.get("size") if a.get("size") is not None else a.get("length")
        if size is None:
            raise ViceArgsError(
                "vice.memory.read requires a `size` argument "
                "(byte count, 1..65535)."
            )
        out: dict[str, Any] = {
            "address": address,
            "size": int(max(1, min(65535, int(size)))),
            # The live vice-mcp server can return a compact hex string; the
            # agent's deterministic formatter needs explicit byte values.
            "encoding": "array",
        }
        if a.get("bank"):
            out["bank"] = str(a["bank"])
        return out

    if method == "vice.memory.write":
        address = _extract_vice_address(a)
        if address is None:
            raise ViceArgsError(
                "vice.memory.write requires an `address` argument."
            )
        raw_data = a.get("data")
        if raw_data is None:
            raw_data = a.get("values")
        if raw_data is None and a.get("value") is not None:
            raw_data = [a.get("value")]
        if isinstance(raw_data, (bytes, bytearray)):
            raw_data = list(raw_data)
        if not isinstance(raw_data, list) or not raw_data:
            raise ViceArgsError(
                "vice.memory.write requires `data` bytes or one `value`."
            )
        values = [_coerce_vice_byte(value) for value in raw_data]
        if any(value is None for value in values):
            raise ViceArgsError(
                "vice.memory.write data values must be bytes (0..255)."
            )
        return {"address": address, "data": values}

    if method == "vice.execution.run":
        # Current vice-mcp's run verb has an empty schema. Watchpoint-based
        # bounds are enforced by the checkpoint itself; unsupported planner
        # hints such as `frames` must not make the MCP request invalid.
        return {}

    if method == "vice.display.screenshot":
        # Always return base64 so the agent sees the image without needing
        # a filesystem path. Accept an explicit path if the caller provided one.
        result: dict[str, Any] = {"return_base64": True, "format": "PNG"}
        if a.get("path"):
            result["path"] = str(a["path"])
        return result

    return a


_VICE_FIRST_ADDR_RE = re.compile(r"\$([0-9A-Fa-f]{4})")


def _detect_vice_addr_mismatch(
    requested_addr: str | None, response_text: str,
) -> str | None:
    """If the response's first ``$XXXX`` is far from the requested address,
    return a warning string. Otherwise None.

    "Far" here means the first emitted address is more than 32 bytes
    before the requested address — vice can legitimately re-align
    forward (e.g. when starting mid-instruction), but it should never
    rewind significantly.
    """
    if not requested_addr or not response_text:
        return None
    try:
        req_int = int(requested_addr.lstrip("$"), 16)
    except ValueError:
        return None
    m = _VICE_FIRST_ADDR_RE.search(response_text)
    if not m:
        return None
    first_int = int(m.group(1), 16)
    if first_int < req_int - 32:
        return (
            f"WARNING: requested address {requested_addr} but response "
            f"begins at ${first_int:04X} (Δ = {req_int - first_int} bytes "
            "earlier). The disassembly below is NOT for the requested "
            "address — treat it as untrusted."
        )
    if first_int > req_int + 256:
        return (
            f"WARNING: requested address {requested_addr} but response "
            f"begins at ${first_int:04X} (Δ = {first_int - req_int} bytes "
            "later). Possibly a vice-mcp re-alignment — verify."
        )
    return None


# ---- capstone -------------------------------------------------------------

def _normalize_seeds(seeds_raw: Any) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    if not isinstance(seeds_raw, list):
        return out
    for s in seeds_raw:
        if isinstance(s, dict):
            addr = _hex_to_int(s.get("address") or s.get("addr") or 0, 0)
            label = str(s.get("label") or s.get("name") or f"seed_{addr:04x}")
        else:
            addr = _hex_to_int(s, 0)
            label = f"seed_{addr:04x}"
        if addr:
            out.append((addr, label))
    return out


def _capstone_linear(
    mem: bytes, args: dict[str, Any], step_id: str
) -> dict[str, Any]:
    if args.get("start") is None or (
        isinstance(args.get("start"), str) and not args["start"].strip()
    ):
        return _record_result(
            "capstone", step_id, False,
            "capstone mode='linear' requires an explicit `start` address "
            "(for example \"$C000\"); no $0801 default was applied.",
            extra={"mode": "linear", "rejection": "missing_arg"},
        )
    start = _hex_to_int(args.get("start"), -1)
    if not 0 <= start <= 0xFFFF:
        return _record_result(
            "capstone", step_id, False,
            f"invalid capstone linear start address: {args.get('start')!r}",
            extra={"mode": "linear", "rejection": "invalid_arg"},
        )
    length = int(args.get("length", 256))
    max_lines = int(args.get("max_lines", 64))
    try:
        text = c64_disasm.linear_disasm(mem, start, length, max_lines=max_lines)
        return _record_result(
            "capstone", step_id, True, text,
            extra={"mode": "linear", "start": start, "length": length},
        )
    except Exception as e:  # noqa: BLE001
        return _vice_disassemble_fallback(
            step_id, start, length, f"{type(e).__name__}: {e}"
        )


def _capstone_recursive(
    mem: bytes, args: dict[str, Any], step_id: str
) -> dict[str, Any]:
    entry = _hex_to_int(args.get("entry", args.get("start", "0x0801")), 0x0801)
    max_insns = int(args.get("max_insns", 600))
    seed_vectors = bool(args.get("seed_vectors", True))
    extra_seeds = _normalize_seeds(args.get("seeds"))

    # If the caller left the default $0801 BASIC-program-start address,
    # auto-advance to the real SYS target embedded in the stub — most
    # games land at $C000/$2000/etc. and the stub is ~10 bytes of BASIC.
    if entry == 0x0801:
        sys_target = c64_disasm.detect_basic_sys(mem)
        if sys_target and 0x0800 < sys_target <= 0xCFFF:
            entry = sys_target

    result = c64_disasm.recursive_disasm(
        mem, entry,
        max_insns=max_insns,
        extra_seeds=extra_seeds or None,
        seed_vectors=seed_vectors,
    )
    stats = result["stats"]
    header = (
        f"Recursive disassembly from ${entry:04X}: "
        f"{stats['instructions']} insns / {stats['subroutines']} subs / "
        f"{stats['labels']} labels"
        + (" (truncated)" if stats["truncated"] else "")
    )
    text = header + "\n\n" + result["listing"]
    return _record_result(
        "capstone", step_id, True, text,
        extra={
            "mode": "recursive",
            "entry": f"${entry:04X}",
            "stats": stats,
            "subroutines": result["subroutines"][:50],
        },
    )


def _capstone_find_loops(
    mem: bytes, args: dict[str, Any], step_id: str
) -> dict[str, Any]:
    top_n = int(args.get("top_n", 8))
    cands = c64_disasm.find_loops(mem, top_n=top_n)
    if not cands:
        text = "No game-loop candidates found."
    else:
        lines = [
            f"#{i + 1} {c['address']} {c.get('candidate_type', 'loop')} "
            f"score={c['score']} — {' | '.join(c['reasons'])}"
            for i, c in enumerate(cands)
        ]
        text = "\n".join(lines)
        # Automatically disassemble the top candidate so the synthesizer
        # can extract routines/labels without requiring an extra plan step.
        top_addr_str = cands[0]["address"]  # e.g. "$C000"
        top_addr = int(top_addr_str.lstrip("$"), 16)
        try:
            rec = c64_disasm.recursive_disasm(mem, top_addr, max_insns=300)
            text += (
                f"\n\nAuto-disassembly of top candidate {top_addr_str} "
                f"({rec['stats']['instructions']} insns, "
                f"{rec['stats']['subroutines']} subs):\n"
                + rec["listing"]
            )
        except Exception:  # noqa: BLE001
            pass
    return _record_result(
        "capstone", step_id, True, text,
        extra={"mode": "find_loops", "candidates": cands},
    )


def _capstone_find_entry(
    mem: bytes, args: dict[str, Any], step_id: str
) -> dict[str, Any]:
    hint = args.get("hint")
    hint_int = _hex_to_int(hint, 0) if hint else None
    info = c64_disasm.find_entry(mem, hint=hint_int)
    return _record_result(
        "capstone", step_id, True, json.dumps(info, indent=2),
        extra={"mode": "find_entry", "info": info},
    )


def _capstone_vectors(
    mem: bytes, args: dict[str, Any], step_id: str
) -> dict[str, Any]:
    info = c64_disasm.list_vectors(mem)
    info["banking"] = c64_disasm.banking_state(mem)
    return _record_result(
        "capstone", step_id, True, json.dumps(info, indent=2),
        extra={"mode": "vectors", "info": info},
    )


def _capstone_bank(
    mem: bytes, args: dict[str, Any], step_id: str
) -> dict[str, Any]:
    """Report the $0001 banking state captured in the dump (tracker 3.6)."""
    bank = c64_disasm.banking_state(mem)
    notes = []
    if bank["ram_under_kernal"]:
        notes.append("KERNAL ROM banked OUT → $E000-$FFFF is game RAM.")
    else:
        notes.append("KERNAL ROM banked IN → $E000-$FFFF is ROM.")
    if not bank["basic_rom_visible"]:
        notes.append("BASIC ROM banked out → $A000-$BFFF is RAM.")
    if not bank["io_visible"]:
        notes.append("I/O banked out → $D000-$DFFF is CHAR ROM, not VIC/SID/CIA.")
    text = (
        f"$0001 = {bank['port_hex']}\n" + "\n".join(f"- {n}" for n in notes)
    )
    return _record_result(
        "capstone", step_id, True, text,
        extra={"mode": "bank", "banking": bank},
    )


def _capstone_polymorphic(
    mem: bytes, args: dict[str, Any], step_id: str
) -> dict[str, Any]:
    entry = _hex_to_int(args.get("entry", args.get("start", "0x0801")), 0x0801)
    max_insns = int(args.get("max_insns", 1000))
    rec = c64_disasm.recursive_disasm(mem, entry, max_insns=max_insns)
    insns_int_keyed = {
        int(k.lstrip("$"), 16): v for k, v in rec["insns"].items()
    }
    findings = c64_disasm.detect_polymorphic(insns_int_keyed)
    if not findings:
        text = (
            f"No polymorphic / self-modifying-code suspects in "
            f"{rec['stats']['instructions']} disassembled instructions from "
            f"${entry:04X}."
        )
    else:
        lines = [
            f"{f['site']}  {f['mnemonic']:<4} {f['operand']:<14} "
            f"-> {f['writes_to']}  ({f['kind']})"
            for f in findings[:64]
        ]
        if len(findings) > 64:
            lines.append(f"... {len(findings) - 64} more")
        text = "\n".join(lines)
    return _record_result(
        "capstone", step_id, True, text,
        extra={
            "mode": "polymorphic",
            "entry": f"${entry:04X}",
            "findings_count": len(findings),
            "findings": findings[:64],
        },
    )


def _recursive_insns_int_keyed(
    mem: bytes, args: dict[str, Any], default_entry: int = 0x0801,
) -> dict[int, dict[str, Any]]:
    """Shared helper: recursive-disassemble from a resolved entry and
    return the int-keyed insns dict the 3.3/3.5 analysers consume."""
    entry = _hex_to_int(args.get("entry", args.get("start", default_entry)),
                        default_entry)
    if entry == 0x0801:
        sys_target = c64_disasm.detect_basic_sys(mem)
        if sys_target and 0x0800 < sys_target <= 0xCFFF:
            entry = sys_target
    max_insns = int(args.get("max_insns", 2000))
    rec = c64_disasm.recursive_disasm(mem, entry, max_insns=max_insns)
    return {int(k.lstrip("$"), 16): v for k, v in rec["insns"].items()}


def _capstone_find_counters(
    mem: bytes, args: dict[str, Any], step_id: str
) -> dict[str, Any]:
    """Game-state variable heuristics (tracker 3.3)."""
    insns = _recursive_insns_int_keyed(mem, args)
    kind_filter = str(args.get("kind") or "").strip().lower()
    cands = c64_disasm.find_counters(insns)
    if kind_filter:
        wanted = {k.strip() for k in kind_filter.split("|") if k.strip()}
        # Match the primary kind OR any alias so `kind='timer'` still
        # finds a site whose primary is `lives` but also matched timer.
        cands = [
            c for c in cands
            if c["kind"] in wanted or wanted & set(c.get("alias_kinds", []))
        ]
    top_n = int(args.get("top_n", 20))
    cands = cands[:top_n]
    if not cands:
        text = "No game-state variable candidates found."
    else:
        text = "\n".join(
            f"{c['address']}  {c['kind']:<18} score={c['score']}"
            + (f" (also: {', '.join(c['alias_kinds'])})"
               if c.get("alias_kinds") else "")
            + f"  [{'; '.join(c['evidence'][:3])}]"
            for c in cands
        )
    return _record_result(
        "capstone", step_id, True, text,
        extra={"mode": "find_counters", "candidates": cands},
    )


def _capstone_idioms(
    mem: bytes, args: dict[str, Any], step_id: str
) -> dict[str, Any]:
    """Deterministic 6502 idiom pre-tagging (tracker 3.5)."""
    insns = _recursive_insns_int_keyed(mem, args)
    found = c64_disasm.find_idioms(insns)
    if not found:
        text = "No recognised idioms."
    else:
        text = "\n".join(
            f"{f['address']}  {f['idiom']:<18} {f['evidence']}" for f in found
        )
    return _record_result(
        "capstone", step_id, True, text,
        extra={"mode": "idioms", "idioms": found},
    )


def _capstone_screen_text(
    mem: bytes, args: dict[str, Any], step_id: str
) -> dict[str, Any]:
    """Decode screen + colour RAM to PETSCII strings (tracker 3.7)."""
    screen_base = _hex_to_int(args.get("screen_base", "0x0400"), 0x0400)
    color_base = _hex_to_int(args.get("color_base", "0xD800"), 0xD800)
    min_run = int(args.get("min_run", 3))
    res = c64_disasm.decode_screen_ram(
        mem, screen_base=screen_base, color_base=color_base, min_run=min_run,
    )
    runs = res["runs"]
    if not runs:
        text = (
            f"No printable text in screen RAM at {res['screen_base']} "
            "(may be a bitmap/sprite screen, or a different base)."
        )
    else:
        text = f"Screen text at {res['screen_base']}:\n" + "\n".join(
            f"  {r['addr_hex']} (row {r['row']}, col {r['col']}): {r['text']!r}"
            for r in runs
        )
    return _record_result(
        "capstone", step_id, True, text,
        extra={"mode": "screen_text", "screen_base": res["screen_base"],
               "runs": runs},
    )


_CAPSTONE_MODES = {
    "linear": _capstone_linear,
    "recursive": _capstone_recursive,
    "find_loops": _capstone_find_loops,
    "loops": _capstone_find_loops,
    "find_entry": _capstone_find_entry,
    "entry": _capstone_find_entry,
    "vectors": _capstone_vectors,
    "polymorphic": _capstone_polymorphic,
    "poly": _capstone_polymorphic,
    "find_counters": _capstone_find_counters,
    "counters": _capstone_find_counters,
    "idioms": _capstone_idioms,
    "screen_text": _capstone_screen_text,
    "screen": _capstone_screen_text,
    "bank": _capstone_bank,
    "banking": _capstone_bank,
}


def capstone_node(state: C64State) -> dict[str, Any]:
    """Capstone static-analysis tool with multiple modes.

    Modes (planner picks via ``args["mode"]``):

    * ``linear`` (default): block disassembly from ``start`` for ``length``
      bytes. Falls back to VICE-MCP ``vice.disassemble`` if Capstone errors
      or the dump is missing.
    * ``recursive``: control-flow following from ``entry`` (default
      $0801), seeding HW + RAM vectors. Returns labelled listing plus
      subroutine list.
    * ``find_loops``: heuristic main-game-loop candidate scoring (IRQ
      vector, backward JMPs, BASIC SYS init chain).
    * ``find_entry``: BASIC SYS detection + vector-based entry candidates;
      validates the hinted entry.
    * ``vectors``: list HW ($FFFA/FC/FE) and RAM ($0314/16/18) vectors.
    * ``polymorphic``: scan a recursive disassembly for self-modifying or
      decryption-loop suspects.
    """
    step = _step_for(state, state.get("current_step_id"))
    args = step.get("args", {}) or {}
    mode = str(args.get("mode") or "linear").lower().strip()
    step_id = step.get("id", "?")

    if not state.get("kb_handle"):
        return _record_result(
            "capstone", step_id, False, "no KB handle",
            extra={"mode": mode, "retryable": False},
        )

    store = get_store(state["kb_handle"])
    if not store.has_dump():
        if mode == "linear":
            if args.get("start") is None or (
                isinstance(args.get("start"), str)
                and not args["start"].strip()
            ):
                return _record_result(
                    "capstone", step_id, False,
                    "capstone mode='linear' requires an explicit `start` "
                    "address; VICE fallback was not attempted.",
                    extra={"mode": "linear", "rejection": "missing_arg"},
                )
            start = _hex_to_int(args.get("start"), -1)
            if not 0 <= start <= 0xFFFF:
                return _record_result(
                    "capstone", step_id, False,
                    f"invalid capstone linear start address: "
                    f"{args.get('start')!r}",
                    extra={"mode": "linear", "rejection": "invalid_arg"},
                )
            length = int(args.get("length", 256))
            return _vice_disassemble_fallback(
                step_id, start, length, "no dump bytes available",
            )
        return _record_result(
            "capstone", step_id, False,
            f"capstone {mode} requires a dump, but none is loaded",
            extra={"mode": mode, "retryable": False},
        )

    handler = _CAPSTONE_MODES.get(mode)
    if handler is None:
        return _record_result(
            "capstone", step_id, False,
            f"unknown capstone mode {mode!r} "
            f"(supported: {', '.join(sorted(set(_CAPSTONE_MODES)))})",
            extra={"mode": mode, "args": args},
        )

    mem = store.full_dump()
    try:
        result = handler(mem, args, step_id)
    except Exception as e:  # noqa: BLE001
        return _record_result(
            "capstone", step_id, False,
            f"{type(e).__name__}: {e}",
            extra={"mode": mode, "args": args},
        )

    # For find_entry: promote the detected BASIC SYS target directly to
    # a durable KB label so the synthesizer always sees it, even if the
    # raw JSON text is later truncated from the digest.
    if mode == "find_entry":
        tr = (result.get("tool_results") or [{}])[0]
        info = tr.get("info") or {}
        sys_addr = _addr_to_int(info.get("basic_sys"))
        if sys_addr and sys_addr > 0x0400:
            store.append_event(EVT_LABEL, "capstone_find_entry", {
                "addr": sys_addr,
                "name": "game_entry",
                "kind": "code",
                "confidence": 0.9,
                "evidence": "BASIC SYS target detected in $0801 stub",
            })
    return result


# ---- tavily ---------------------------------------------------------------

def tavily_node(state: C64State) -> dict[str, Any]:
    step = _step_for(state, state.get("current_step_id"))
    args = step.get("args", {}) or {}
    query = args.get("q") or state.get("question") or ""

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return _record_result(
            "tavily", step.get("id", "?"), False,
            "TAVILY_API_KEY not set — skipped real search",
            extra={"query": query, "retryable": False},
        )

    try:
        try:
            from tavily import TavilyClient  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            return _record_result(
                "tavily", step.get("id", "?"), False,
                "tavily-python not installed in this interpreter "
                f"({sys.executable}). Install with: "
                "`pip install tavily-python` (or re-run `pip install -e .` "
                "after pulling pyproject changes).",
                extra={"query": query, "interpreter": sys.executable,
                       "retryable": False},
            )

        tool_cfg = (
            (load_config().get("tools") or {}).get("tavily") or {}
        )
        configured_domains = args.get("include_domains") or tool_cfg.get(
            "include_domains",
        ) or [
            "csdb.dk", "codebase64.org", "lemon64.com", "gamebase64.com",
            "archive.org", "forum64.de", "c64-wiki.com",
        ]
        if isinstance(configured_domains, str):
            configured_domains = configured_domains.split(",")
        include_domains = [
            str(d).strip() for d in configured_domains
            if str(d).strip()
        ]
        search_depth = str(
            args.get("search_depth") or tool_cfg.get("search_depth")
            or "advanced"
        ).strip().lower()
        if search_depth not in {"basic", "advanced", "fast", "ultra-fast"}:
            search_depth = "advanced"
        max_results = int(
            args.get("max_results") or tool_cfg.get("max_results") or 5,
        )
        max_results = max(1, min(20, max_results))

        client = TavilyClient(api_key=api_key)
        resp = client.search(
            query=query,
            max_results=max_results,
            include_domains=include_domains,
            search_depth=search_depth,
        )
        items = [
            {"title": r.get("title"), "url": r.get("url"),
             "snippet": (r.get("content") or "")[:2_000]}
            for r in (resp.get("results") or [])
        ]
        # Include title, URL and full content snippet so the synthesizer
        # can extract facts from the actual page text, not just the URL.
        text_parts = []
        for i in items:
            part = f"## {i['title']}\nURL: {i['url']}"
            if i["snippet"]:
                part += f"\n{i['snippet']}"
            text_parts.append(part)
        text = "\n\n".join(text_parts) or "(no hits)"
        return _record_result(
            "tavily", step.get("id", "?"), True, text,
            extra={
                "query": query, "results": items,
                "include_domains": include_domains,
                "search_depth": search_depth,
                "max_results": max_results,
            },
        )
    except Exception as e:  # noqa: BLE001
        return _record_result(
            "tavily", step.get("id", "?"), False,
            f"{type(e).__name__}: {e}", extra={"query": query},
        )


# ---- vice (MCP streamable-HTTP) ------------------------------------------

def _vice_call(method: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call a vice-mcp tool via the official MCP streamable-HTTP client.

    The transport spec is JSON-RPC 2.0 over HTTP; vice-mcp publishes its
    endpoint at ``$VICE_MCP_URL/mcp`` (Cursor's `mcp.json` convention).
    The helper tolerates a URL given without the ``/mcp`` suffix.
    """
    from tools import vice_mcp

    tool_name = _vice_normalize_method(method)
    tool_args = _vice_normalize_args(tool_name, args)
    # vice-mcp documents dotted logical methods but registers MCP tools with
    # underscore identifiers (vice.memory.read -> vice_memory_read).
    transport_name = tool_name.replace(".", "_")
    out = vice_mcp.call_tool(transport_name, tool_args)
    out["args"] = tool_args
    out["tool"] = tool_name
    if out.get("is_error"):
        raise RuntimeError(
            f"vice-mcp returned error for {tool_name}: "
            f"{out.get('data') or out.get('raw_content')}"
        )
    return out


_VISION_SYSTEM_PROMPT = (
    "You are a neutral visual inspection component for a reverse-engineering "
    "tool. Describe only visible image evidence in concise plain text. Do not "
    "emit JSON, do not infer unseen memory values, and clearly mark uncertain "
    "text or objects."
)


def _invoke_vision_messages(
    messages: list[Any],
    *,
    reservation_text: str,
    max_cost_usd: float | None,
) -> tuple[str, str | None, dict[str, Any], str | None]:
    """Invoke the vision role behind the same conservative cost gate."""
    role = "vision"
    model_name = None
    reservation = 0.0
    try:
        llm = get_llm(role)
        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
        reservation = _call_cost_reservation_usd(
            llm,
            model_name,
            reservation_text,
            system_text=_VISION_SYSTEM_PROMPT,
        )
        if max_cost_usd is not None and reservation > max_cost_usd:
            error = (
                f"cost reservation ${reservation:.4f} exceeds remaining "
                f"run allowance ${max_cost_usd:.4f}"
            )
            entry = llm_usage.record({
                "role": role,
                "model": model_name,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "reserved_cost_usd": reservation,
                "ok": False,
                "transport_ok": False,
                "rejection": "cost_reservation",
                "budget_reservation_exhausted": True,
                "error": error,
            })
            return "", error, entry, model_name
        response = llm.invoke(messages)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        entry = llm_usage.record({
            "role": role,
            "model": model_name,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "reserved_cost_usd": reservation,
            "ok": False,
            "transport_ok": False,
            "error": error[:200],
        })
        return "", error, entry, model_name

    in_tok, out_tok = llm_usage.extract_usage(response)
    content = _flatten_lc_ai_message_content(response).strip()
    error = None if content else "empty vision response"
    entry = llm_usage.record({
        "role": role,
        "model": model_name,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": llm_usage.estimate_cost_usd(
            model_name, in_tok, out_tok,
        ),
        "reserved_cost_usd": reservation,
        "ok": bool(content),
        "transport_ok": True,
        **({} if error is None else {"error": error}),
    })
    return content, error, entry, model_name


def _screenshot_data_uri(data: Any) -> str | None:
    """Extract/reconstruct a screenshot data URI from a VICE payload."""
    if isinstance(data, str) and data.startswith("data:image/"):
        return data
    if not isinstance(data, dict):
        return None
    if (
        isinstance(data.get("data_uri"), str)
        and data["data_uri"].startswith("data:image/")
    ):
        return data["data_uri"]
    if isinstance(data.get("base64"), str):
        fmt = str(data.get("format") or "png").strip().lower()
        return f"data:image/{fmt};base64,{data['base64']}"
    return None


def _persist_screenshot(
    state: C64State, step_id: str, data_uri: str,
) -> tuple[str | None, dict[str, Any]]:
    """Store the returned image under the run's session for later review."""
    meta: dict[str, Any] = {}
    try:
        header, encoded = data_uri.split(",", 1)
        if ";base64" not in header.lower():
            raise ValueError("screenshot data URI is not base64 encoded")
        image = base64.b64decode(encoded, validate=True)
        if not image:
            raise ValueError("screenshot image is empty")
        if len(image) > 20 * 1024 * 1024:
            raise ValueError("screenshot exceeds the 20 MiB safety limit")
        session_dir = (
            Path(state["kb_handle"]).parent
            if state.get("kb_handle") else _session_dir(state.get("game", "unknown"))
        )
        out_dir = session_dir / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_step = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(step_id))[:48] or "shot"
        digest = hashlib.sha256(image).hexdigest()[:12]
        suffix = ".jpg" if header.lower().startswith("data:image/jpeg") else ".png"
        path = out_dir / f"{safe_step}-{digest}{suffix}"
        if not path.exists():
            path.write_bytes(image)
        meta.update({"image_bytes": len(image), "image_sha256": digest})
        return str(path), meta
    except (ValueError, OSError, binascii.Error) as e:
        meta["screenshot_save_error"] = f"{type(e).__name__}: {e}"
        return None, meta


def _describe_screenshot_result(
    data_uri: str,
    *,
    remaining_budget_usd: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Describe one screenshot through the dedicated vision role.

    The description is stored as the tool result so the synthesizer can
    extract spatial/visual facts (sprites visible, score layout, colour
    RAM usage, etc.) without any downstream code needing multimodal support.

    Exactly one configured ``vision`` call is made. The former analyst →
    critic → planner chain reused JSON-only role prompts, burned up to three
    calls, and made prose output look contract-invalid (tracker 4.4).
    """
    prompt_text = (
        "You are looking at a screenshot of a Commodore 64 game running inside "
        "the VICE emulator. Describe exactly what you see on screen:\n"
        "- Game title or mode visible (title screen, gameplay, game over, etc.)\n"
        "- All text visible (score, lives, level, messages) with exact values\n"
        "- Sprites and characters visible and their screen positions (approx col/row)\n"
        "- Colour scheme (border, background, sprite colours)\n"
        "- Any HUD elements, status bars, or UI regions\n"
        "- Anything that looks like debug output or unusual artefacts\n"
        "Be precise and terse — this description goes into a reverse-engineering KB."
    )
    role = "vision"
    meta: dict[str, Any] = {"vision_role": role}
    multimodal_msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text", "text": prompt_text},
    ])
    remaining = (
        _remaining_llm_budget({})
        if remaining_budget_usd is None else remaining_budget_usd
    )
    description, error, entry, model_name = _invoke_vision_messages(
        [SystemMessage(_VISION_SYSTEM_PROMPT), multimodal_msg],
        reservation_text=prompt_text + "\n" + data_uri,
        max_cost_usd=remaining,
    )
    status = (
        "cost_reservation"
        if entry.get("rejection") == "cost_reservation"
        else ("ok" if description else "error")
    )
    meta.update({
        "vision_model": model_name,
        "vision_ok": bool(description),
        "vision_status": status,
        "description_chars": len(description),
    })
    if description:
        return f"[screenshot description via {role}]\n{description}", meta
    meta["vision_error"] = error
    if entry.get("rejection") != "cost_reservation":
        print(
            f"[vice/screenshot] dedicated vision call failed: {error}",
            file=sys.stderr, flush=True,
        )

    return _screenshot_unavailable_text(data_uri), meta


def _screenshot_unavailable_text(data_uri: str) -> str:
    b64_len = len(data_uri) - (data_uri.index(",") + 1) if "," in data_uri else 0
    return (
        f"[screenshot: base64 image, {b64_len} chars — "
        "vision description unavailable]"
    )


def _describe_screenshot(
    data_uri: str,
    *,
    remaining_budget_usd: float | None = None,
) -> str:
    """Compatibility wrapper; never bills without explicit run budget context."""
    if remaining_budget_usd is None:
        return _screenshot_unavailable_text(data_uri)
    return _describe_screenshot_result(
        data_uri, remaining_budget_usd=remaining_budget_usd,
    )[0]


def _compare_screenshot_pair(
    before_uri: str,
    after_uri: str,
    *,
    expectation: str,
    remaining_budget_usd: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Compare a labelled before/after pair with one dedicated vision call."""
    role = "vision"
    meta: dict[str, Any] = {"vision_role": role, "vision_comparison": True}
    prompt = (
        "Compare the labelled BEFORE and AFTER Commodore 64 screenshots. "
        "Report only visible differences, especially HUD text/numbers, sprites, "
        "screen corruption, and whether this expectation is visibly supported: "
        f"{expectation}. State supported, contradicted, or inconclusive and why."
    )
    message = HumanMessage(content=[
        {"type": "text", "text": "BEFORE"},
        {"type": "image_url", "image_url": {"url": before_uri}},
        {"type": "text", "text": "AFTER"},
        {"type": "image_url", "image_url": {"url": after_uri}},
        {"type": "text", "text": prompt},
    ])
    remaining = (
        _remaining_llm_budget({})
        if remaining_budget_usd is None else remaining_budget_usd
    )
    comparison, error, entry, model_name = _invoke_vision_messages(
        [SystemMessage(_VISION_SYSTEM_PROMPT), message],
        reservation_text=prompt + "\n" + before_uri + "\n" + after_uri,
        max_cost_usd=remaining,
    )
    meta.update({
        "vision_model": model_name,
        "vision_ok": bool(comparison),
        "vision_status": (
            "cost_reservation"
            if entry.get("rejection") == "cost_reservation"
            else ("ok" if comparison else "error")
        ),
    })
    if comparison:
        return f"[before/after comparison via {role}]\n{comparison}", meta
    meta["vision_error"] = error
    return "[before/after vision comparison unavailable]", meta


def _coerce_vice_byte(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 0xFF else None
    if isinstance(value, str):
        s = value.strip()
        try:
            if s.startswith("$"):
                n = int(s[1:], 16)
            elif s.lower().startswith("0x"):
                n = int(s, 16)
            else:
                n = int(s, 10)
        except ValueError:
            try:
                n = int(s, 16)
            except ValueError:
                return None
        return n if 0 <= n <= 0xFF else None
    return None


def _extract_vice_bytes(data: Any) -> tuple[list[int], Any | None]:
    """Return (byte values, payload address) from common vice-mcp shapes."""
    address = None
    candidate = data
    if isinstance(data, dict):
        address = data.get("address") or data.get("start") or data.get("base")
        for key in ("data", "bytes", "memory", "values"):
            if isinstance(data.get(key), list):
                candidate = data[key]
                break
    if isinstance(candidate, (bytes, bytearray)):
        return list(candidate), address
    if not isinstance(candidate, list):
        return [], address
    values: list[int] = []
    for value in candidate:
        # `vice_memory_read encoding=array` returns legacy two-digit HEX
        # strings. Keep user-supplied write values decimal by handling this
        # representation only at the response boundary.
        if isinstance(value, str) and re.fullmatch(r"[0-9A-Fa-f]{2}", value.strip()):
            byte = int(value.strip(), 16)
        else:
            byte = _coerce_vice_byte(value)
        if byte is None:
            return [], address
        values.append(byte)
    return values, address


def _format_vice_memory_read(
    data: Any,
    requested_addr: Any,
    *,
    max_chars: int = 4_000,
    remaining_budget_usd: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Compact a VICE byte array into 16-byte hex lines before truncation."""
    values, payload_addr = _extract_vice_bytes(data)
    if not values:
        return _vice_text(
            data, remaining_budget_usd=remaining_budget_usd,
        ), {"memory_bytes_returned": 0}
    addr_text = _coerce_vice_address(payload_addr or requested_addr) or "$0000"
    base = int(addr_text[1:], 16)
    lines = [
        f"${(base + off) & 0xFFFF:04X}: "
        + " ".join(f"{b:02X}" for b in values[off : off + 16])
        for off in range(0, len(values), 16)
    ]
    full = "\n".join(lines)
    omitted = 0
    if len(full) > max_chars:
        # Preserve both ends of a large read; middle omission is explicit.
        line_budget = max(2, (max_chars - 100) // 56)
        head_n = (line_budget + 1) // 2
        tail_n = line_budget // 2
        middle_end = len(lines) - tail_n
        omitted = sum(
            len(values[off : off + 16])
            for off in range(head_n * 16, middle_end * 16, 16)
        )
        marker = f"... [{omitted} middle byte(s) omitted] ..."
        full = "\n".join([*lines[:head_n], marker, *lines[-tail_n:]])
    return full, {
        "memory_bytes_returned": len(values),
        "memory_bytes_omitted": omitted,
        "memory_base": addr_text,
        "memory_format": "hex16",
    }


def _vice_text(
    data: Any,
    *,
    remaining_budget_usd: float | None = None,
) -> str:
    """Best-effort flatten of a vice-mcp response payload into a string."""
    if isinstance(data, str):
        # Screenshot data-URI: defer to the vision helper rather than discarding.
        if data.startswith("data:image/"):
            return _describe_screenshot(
                data, remaining_budget_usd=remaining_budget_usd,
            )
        return data
    if isinstance(data, dict):
        # Screenshot response: {status, format, data_uri, base64, size}
        data_uri = _screenshot_data_uri(data)
        if data_uri:
            return _describe_screenshot(
                data_uri, remaining_budget_usd=remaining_budget_usd,
            )
        if "lines" in data and isinstance(data["lines"], list):
            return "\n".join(
                (
                    line.get("text") or line.get("instruction") or json.dumps(line)
                    if isinstance(line, dict)
                    else str(line)
                )
                for line in data["lines"]
            )
        if "text" in data and isinstance(data["text"], str):
            return data["text"]
    if isinstance(data, list):
        parts = [
            _vice_text(d, remaining_budget_usd=remaining_budget_usd)
            for d in data
        ]
        return "\n".join(p for p in parts if p)
    return json.dumps(data, indent=2, default=str)


def _vice_disassemble_fallback(
    step_id: str,
    start: int,
    length: int,
    capstone_err: str,
) -> dict[str, Any]:
    """Used by `capstone_node` when Capstone fails or the dump is unavailable."""
    if not os.getenv("VICE_MCP_URL", "").strip():
        return _record_result(
            "capstone", step_id, False,
            f"capstone failed ({capstone_err}); VICE_MCP_URL not set for fallback",
            extra={"start": start, "length": length,
                   "fallback_attempted": False, "retryable": False},
        )
    try:
        requested_addr = f"${start:04X}"
        out = _vice_call(
            "vice.disassemble",
            {
                "address": requested_addr,
                "count": max(1, min(100, length // 2 if length else 16)),
                "show_symbols": True,
            },
        )
        raw_text = _vice_text(out.get("data"))
        warning = _detect_vice_addr_mismatch(requested_addr, raw_text)
        header = (
            f"[{out.get('tool', 'vice.disassemble')} "
            f"args={json.dumps(out.get('args') or {}, default=str)}]"
        )
        pieces = [header]
        if warning:
            pieces.append(warning)
        pieces.append(raw_text)
        text = "\n".join(pieces)
        return _record_result(
            "vice", step_id, True, text,
            extra={
                "start": start,
                "length": length,
                "fallback_from": f"capstone ({capstone_err})",
                "url": out.get("url"),
                "tool": out.get("tool"),
                "tool_args": out.get("args"),
                **({"addr_mismatch_warning": warning} if warning else {}),
            },
        )
    except Exception as e:  # noqa: BLE001
        return _record_result(
            "capstone", step_id, False,
            f"capstone failed ({capstone_err}); VICE fallback also failed: "
            f"{type(e).__name__}: {e}",
            extra={"start": start, "length": length, "fallback_attempted": True},
        )


# --------------------------------------------------------------------------- #
# VICE composites — memory diffing (3.1) + watchpoint trace (3.2)
# --------------------------------------------------------------------------- #

_VICE_COMPOSITE_METHODS = {
    "snapshot", "memory.snapshot", "vice.memory.snapshot",
    "diff", "memory.diff", "vice.memory.diff",
    "monotonic_scan", "memory.monotonic_scan", "vice.memory.monotonic_scan",
    "trace", "vice.trace",
    "poke_verify", "poke_and_peek", "vice.poke_verify",
}


def _snapshot_dir(state: C64State) -> Path | None:
    """`sessions/<slug>/snapshots/` derived from the KB handle."""
    handle = state.get("kb_handle")
    if not handle:
        return None
    d = Path(handle).parent / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_named_snapshot(state: C64State, name: str) -> bytes | None:
    """Load a snapshot by name. The reserved name ``dump`` returns the
    ingested 64 KB dump (so you can diff live RAM against the dump with no
    prior snapshot — the offline path, tracker 3.1)."""
    if str(name).strip().lower() == "dump":
        if state.get("kb_handle"):
            try:
                return get_store(state["kb_handle"]).full_dump()
            except Exception:  # noqa: BLE001
                return None
        return None
    try:
        from tools.research_notebook import normalize_dump_state_name

        safe_name = normalize_dump_state_name(str(name))
    except ValueError:
        return None
    d = _snapshot_dir(state)
    if d is not None:
        p = d / f"{safe_name}.bin"
        if p.exists():
            return p.read_bytes()
    handle = state.get("kb_handle")
    if handle:
        try:
            from tools.research_notebook import dump_state_path, load_dump_catalog

            rows = load_dump_catalog(Path(handle).parent)
            match = next(
                (row for row in rows if row.get("name") == safe_name), None,
            )
            if match:
                path = dump_state_path(Path(handle).parent, match)
                if path.exists():
                    return path.read_bytes()
        except Exception:  # noqa: BLE001
            pass
    return None


def _vice_read_full_ram(address: int = 0x0000, size: int = 0x10000) -> bytes:
    """Read a RAM range from the live emulator into a bytes buffer."""
    out = _vice_call(
        "vice.memory.read",
        {"address": f"${address:04X}", "size": size},
    )
    data = out.get("data")
    # vice-mcp returns bytes / list[int] / {"data":[...]} shapes.
    if isinstance(data, (bytes, bytearray)):
        buf = bytes(data)
    elif isinstance(data, list):
        buf = bytes(int(x) & 0xFF for x in data)
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        buf = bytes(int(x) & 0xFF for x in data["data"])
    else:
        raise RuntimeError(
            f"vice.memory.read returned an unparseable shape: {type(data)}"
        )
    # Zero-pad to a full 64 KB image so region classification lines up.
    if address == 0 and len(buf) < 0x10000:
        buf = buf + bytes(0x10000 - len(buf))
    return buf


def _vice_composite(
    state: C64State, method: str, args: dict[str, Any], step_id: str,
) -> dict[str, Any]:
    """Agent-side VICE composites (tracker 3.1 / 3.2).

    snapshot        : read live RAM, save under sessions/<slug>/snapshots/
    diff            : diff two snapshots (names; ``dump`` = ingested dump),
                      changed addresses classified by region, old→new
    monotonic_scan  : intersect ≥2 snapshots by a known delta (−1 lives)
    trace           : watchpoint on an address → run → read PC/registers →
                      resolve the writing instruction → disasm around it
    poke_verify     : save full state + capture before → write/read-back one
                      approved byte → capture after → restore in finally
    """
    from tools import mem_diff

    m = method.lower().rsplit(".", 1)[-1]  # normalise to the verb

    if m == "snapshot":
        name = str(args.get("name") or "snap").strip()
        from tools.research_notebook import normalize_dump_state_name

        try:
            safe_name = normalize_dump_state_name(name)
        except ValueError as exc:
            return _record_result(
                "vice", step_id, False, str(exc),
                extra={"method": "vice.memory.snapshot", "retryable": False},
            )
        d = _snapshot_dir(state)
        if d is None:
            return _record_result(
                "vice", step_id, False,
                "snapshot requires a KB handle (no session dir).",
                extra={"method": "vice.memory.snapshot", "retryable": False},
            )
        try:
            buf = _vice_read_full_ram()
        except Exception as e:  # noqa: BLE001
            return _record_result(
                "vice", step_id, False,
                f"vice.memory.snapshot failed: {type(e).__name__}: {e}",
                extra={"method": "vice.memory.snapshot"},
            )
        snapshot_path = d / f"{safe_name}.bin"
        snapshot_path.write_bytes(buf)
        try:
            from tools.research_notebook import freeze_dump_state

            freeze_dump_state(
                d.parent,
                name=name,
                source_path=snapshot_path,
                description="VICE live snapshot",
            )
        except Exception:  # noqa: BLE001
            pass
        return _record_result(
            "vice", step_id, True,
            f"Saved snapshot '{name}' ({len(buf)} bytes) → "
            f"snapshots/{safe_name}.bin. Diff it against another "
            "snapshot (or the reserved name 'dump') to find changed state.",
            extra={"method": "vice.memory.snapshot", "snapshot": name,
                   "size": len(buf)},
        )

    if m == "diff":
        a_name = str(args.get("a") or "dump")
        b_name = str(args.get("b") or "").strip()
        a = _load_named_snapshot(state, a_name)
        if b_name:
            # Explicit 'b' — must exist; no silent live fallback.
            b = _load_named_snapshot(state, b_name)
        elif os.getenv("VICE_MCP_URL", "").strip():
            # 'b' omitted → diff 'a' against a fresh live read.
            try:
                b = _vice_read_full_ram()
                b_name = "live"
            except Exception as e:  # noqa: BLE001
                return _record_result(
                    "vice", step_id, False,
                    f"vice.memory.diff live read failed: {type(e).__name__}: {e}",
                    extra={"method": "vice.memory.diff"},
                )
        else:
            b = None
        if a is None or b is None:
            return _record_result(
                "vice", step_id, False,
                f"vice.memory.diff needs two snapshots; got a={a_name!r} "
                f"(loaded={a is not None}), b={b_name or '(omitted)'!r} "
                f"(loaded={b is not None}). Take snapshots first.",
                extra={"method": "vice.memory.diff", "retryable": False},
            )
        rows = mem_diff.diff_snapshots(
            a, b, exclude_io=bool(args.get("exclude_io", True)),
        )
        return _record_result(
            "vice", step_id, True,
            f"diff {a_name} → {b_name}:\n" + mem_diff.summarize_diff(rows),
            extra={"method": "vice.memory.diff", "a": a_name, "b": b_name,
                   "changed": rows[:256]},
        )

    if m == "monotonic_scan":
        names = args.get("snapshots") or []
        if isinstance(names, str):
            names = [s.strip() for s in names.split(",") if s.strip()]
        delta = int(args.get("delta", -1))
        snaps = [_load_named_snapshot(state, n) for n in names]
        if len(names) < 2 or any(s is None for s in snaps):
            missing = [n for n, s in zip(names, snaps) if s is None]
            return _record_result(
                "vice", step_id, False,
                "vice.memory.monotonic_scan needs ≥2 existing snapshots; "
                f"missing/insufficient: {missing or names}.",
                extra={"method": "vice.memory.monotonic_scan",
                       "retryable": False},
            )
        hits = mem_diff.monotonic_scan(snaps, delta=delta)
        if not hits:
            text = (
                f"No address changed by Δ{delta:+d} across all "
                f"{len(names)} snapshots. Widen the delta or re-capture."
            )
        else:
            text = (
                f"{len(hits)} address(es) changed by Δ{delta:+d} across all "
                f"{len(names)} snapshots (strong game-state candidates):\n"
                + "\n".join(
                    f"  {h['addr_hex']} ({h['region']}): "
                    f"{' → '.join(str(v) for v in h['values'])}"
                    for h in hits[:40]
                )
            )
        return _record_result(
            "vice", step_id, True, text,
            extra={"method": "vice.memory.monotonic_scan", "delta": delta,
                   "candidates": hits},
        )

    if m == "trace":
        return _vice_trace(args, step_id)

    if m == "poke_verify":
        return _vice_poke_verify(state, args, step_id)

    return _record_result(
        "vice", step_id, False,
        f"unknown vice composite {method!r}",
        extra={"method": method, "retryable": False},
    )


def _vice_poke_verify(
    state: C64State,
    args: dict[str, Any],
    step_id: str,
) -> dict[str, Any]:
    """Approval-gated, restore-on-exit one-byte visual experiment.

    This deliberately supports one candidate byte only. A before screenshot
    must exist and the original byte must be readable before any mutation.
    A full VICE snapshot is preferred and reloaded in ``finally`` before the
    vision model sees the captured pair; older servers fall back to restoring
    the original byte. The write is read back from the same bank before an
    after image is accepted. Optional execution resume can change unrelated
    emulator state, which is why the full snapshot is the primary safety path.
    """
    address = _extract_vice_address(args)
    value = _coerce_vice_byte(
        args.get("value") if args.get("value") is not None
        else args.get("candidate"),
    )
    expectation = str(args.get("expect") or args.get("expectation") or "").strip()
    if address is None or value is None or not expectation:
        return _record_result(
            "vice",
            step_id,
            False,
            "vice.poke_verify requires address, one byte value (0..255), "
            "and an explicit expect string.",
            extra={
                "method": "vice.poke_verify",
                "rejection": "missing_arg",
                "retryable": False,
            },
        )
    try:
        frames = int(args.get("frames", 0))
    except (TypeError, ValueError):
        frames = -1
    if frames < 0 or frames > 60:
        return _record_result(
            "vice",
            step_id,
            False,
            "vice.poke_verify frames must be between 0 and 60.",
            extra={
                "method": "vice.poke_verify",
                "rejection": "invalid_arg",
                "retryable": False,
            },
        )

    snapshot_name: str | None = None
    snapshot_error: str | None = None
    try:
        candidate_name = f"c64re_poke_{uuid.uuid4().hex[:12]}"
        _vice_call("vice.snapshot.save", {
            "name": candidate_name,
            "description": (
                f"Automatic safety restore before {step_id} poke verification"
            ),
            "include_roms": False,
            "include_disks": False,
        })
        snapshot_name = candidate_name
    except Exception as exc:  # noqa: BLE001
        # Older servers may not expose snapshot save/load. The byte-level
        # fallback below remains mandatory and is still verified by tests.
        snapshot_error = f"{type(exc).__name__}: {exc}"

    try:
        read_args: dict[str, Any] = {"address": address, "size": 1}
        if args.get("bank"):
            read_args["bank"] = str(args["bank"])
        original_result = _vice_call("vice.memory.read", read_args)
        original_values, _payload_address = _extract_vice_bytes(
            original_result.get("data"),
        )
        if len(original_values) != 1:
            raise RuntimeError("memory.read did not return exactly one byte")
        original = original_values[0]
        if value == original:
            return _record_result(
                "vice",
                step_id,
                False,
                f"vice.poke_verify candidate ${value:02X} equals the current "
                "byte; no visual experiment was run.",
                extra={
                    "method": "vice.poke_verify",
                    "address": address,
                    "original_value": original,
                    "candidate_value": value,
                    "bank": args.get("bank"),
                    "rejection": "invalid_arg",
                    "retryable": False,
                },
            )

        before_result = _vice_call("vice.display.screenshot", {})
        before_uri = _screenshot_data_uri(before_result.get("data"))
        if not before_uri:
            raise RuntimeError("baseline screenshot contained no image")
        before_path, before_meta = _persist_screenshot(
            state, f"{step_id}-before", before_uri,
        )
    except Exception as exc:  # noqa: BLE001
        return _record_result(
            "vice",
            step_id,
            False,
            "vice.poke_verify aborted before mutation: "
            f"{type(exc).__name__}: {exc}",
            extra={"method": "vice.poke_verify", "address": address},
        )

    after_uri: str | None = None
    after_path: str | None = None
    after_meta: dict[str, Any] = {}
    experiment_error: str | None = None
    restore_error: str | None = None
    restore_method = "byte"
    write_verified = False
    try:
        write_args: dict[str, Any] = {"address": address, "value": value}
        if args.get("bank"):
            write_args["bank"] = str(args["bank"])
        _vice_call("vice.memory.write", write_args)
        verify_result = _vice_call("vice.memory.read", read_args)
        verify_values, _verify_address = _extract_vice_bytes(
            verify_result.get("data"),
        )
        if verify_values != [value]:
            observed = (
                f"${verify_values[0]:02X}" if len(verify_values) == 1
                else repr(verify_values)
            )
            raise RuntimeError(
                f"memory.write read-back mismatch: requested ${value:02X}, "
                f"observed {observed}"
            )
        write_verified = True
        if frames:
            _vice_call("vice.execution.run", {"frames": frames})
        after_result = _vice_call("vice.display.screenshot", {})
        after_uri = _screenshot_data_uri(after_result.get("data"))
        if not after_uri:
            raise RuntimeError("after screenshot contained no image")
        after_path, after_meta = _persist_screenshot(
            state, f"{step_id}-after", after_uri,
        )
    except Exception as exc:  # noqa: BLE001
        experiment_error = f"{type(exc).__name__}: {exc}"
    finally:
        snapshot_load_error: str | None = None
        if snapshot_name:
            try:
                _vice_call("vice.snapshot.load", {"name": snapshot_name})
                restore_method = "full_snapshot"
            except Exception as exc:  # noqa: BLE001
                snapshot_load_error = f"{type(exc).__name__}: {exc}"
        if restore_method != "full_snapshot":
            try:
                restore_args: dict[str, Any] = {
                    "address": address, "value": original,
                }
                if args.get("bank"):
                    restore_args["bank"] = str(args["bank"])
                _vice_call("vice.memory.write", restore_args)
            except Exception as exc:  # noqa: BLE001
                byte_error = f"{type(exc).__name__}: {exc}"
                restore_error = (
                    f"snapshot load failed ({snapshot_load_error}); "
                    f"byte fallback failed ({byte_error})"
                    if snapshot_load_error else byte_error
                )

    if restore_error:
        return _record_result(
            "vice",
            step_id,
            False,
            f"CRITICAL: vice.poke_verify could not restore {address} to "
            f"${original:02X}: {restore_error}",
            extra={
                "method": "vice.poke_verify",
                "address": address,
                "original_value": original,
                "candidate_value": value,
                "bank": args.get("bank"),
                "restored": False,
                "write_verified": write_verified,
                "restore_method": restore_method,
                "safety_snapshot": snapshot_name,
                "snapshot_save_error": snapshot_error,
                "rejection": "restore_failed",
                "retryable": False,
                "before_screenshot_path": before_path,
                "after_screenshot_path": after_path,
            },
        )
    if experiment_error or after_uri is None:
        return _record_result(
            "vice",
            step_id,
            False,
            "vice.poke_verify experiment failed after mutation; the original "
            f"byte was restored: {experiment_error or 'no after screenshot'}",
            extra={
                "method": "vice.poke_verify",
                "address": address,
                "original_value": original,
                "candidate_value": value,
                "bank": args.get("bank"),
                "restored": True,
                "write_verified": write_verified,
                "restore_method": restore_method,
                "safety_snapshot": snapshot_name,
                "snapshot_save_error": snapshot_error,
                "before_screenshot_path": before_path,
                "after_screenshot_path": after_path,
            },
        )

    images_identical = bool(
        before_meta.get("image_sha256")
        and before_meta.get("image_sha256") == after_meta.get("image_sha256")
    )
    if images_identical:
        comparison = (
            "[mechanical before/after comparison]\n"
            "Screenshots are byte-identical; the expected visible change "
            "is contradicted."
        )
        vision_meta = {
            "vision_role": "vision",
            "vision_ok": False,
            "vision_status": "skipped_identical_images",
            "visual_verdict": "contradicted",
        }
    else:
        comparison, vision_meta = _compare_screenshot_pair(
            before_uri,
            after_uri,
            expectation=expectation,
            remaining_budget_usd=_remaining_llm_budget(state),
        )
        comparison_lower = comparison.lower()
        if re.search(r"\bcontradicted\b", comparison_lower):
            visual_verdict = "contradicted"
        elif re.search(r"\binconclusive\b", comparison_lower):
            visual_verdict = "inconclusive"
        elif re.search(r"\bsupported\b", comparison_lower):
            visual_verdict = "supported"
        else:
            visual_verdict = "unknown"
        vision_meta["visual_verdict"] = visual_verdict
    return _record_result(
        "vice",
        step_id,
        True,
        f"poke verification at {address}: ${original:02X} → ${value:02X}; "
        f"captured before/after and restored via {restore_method}.\n"
        f"Expectation: {expectation}\n{comparison}",
        extra={
            "method": "vice.poke_verify",
            "address": address,
            "original_value": original,
            "candidate_value": value,
            "bank": args.get("bank"),
            "frames": frames,
            "expectation": expectation,
            "images_identical": images_identical,
            "restored": True,
            "write_verified": write_verified,
            "restore_method": restore_method,
            "safety_snapshot": snapshot_name,
            "snapshot_save_error": snapshot_error,
            "before_screenshot_path": before_path,
            "after_screenshot_path": after_path,
            "before_image": before_meta,
            "after_image": after_meta,
            **vision_meta,
        },
    )


_TRACE_WRITE_MNEMONICS = frozenset({
    "sta", "stx", "sty", "stz", "inc", "dec", "asl", "lsr", "rol", "ror",
})


def _extract_checkpoint_id(add_result: dict[str, Any]) -> Any:
    """Best-effort checkpoint id from a `vice.checkpoint.add` result."""
    data = add_result.get("data")
    if isinstance(data, dict):
        for k in ("checkpoint_id", "checkpointId", "number", "id"):
            if data.get(k) is not None:
                return data[k]
    text = _vice_text(data)
    m = re.search(r"(?:checkpoint|number|id)[^0-9]*([0-9]+)", text, re.I)
    return int(m.group(1)) if m else None


def _disasm_writes_addr(disasm_text: str, addr_int: int) -> bool:
    """True iff a line disassembles to a write/RMW targeting *addr_int*."""
    for line in (disasm_text or "").splitlines():
        toks = line.lower().split()
        if not any(t in _TRACE_WRITE_MNEMONICS for t in toks):
            continue
        for m in re.finditer(r"\$([0-9a-fA-F]{2,4})", line):
            if int(m.group(1), 16) == addr_int:
                return True
    return False


def _vice_trace(args: dict[str, Any], step_id: str) -> dict[str, Any]:
    """Watchpoint → run → resolve-writer composite (tracker 3.2, hardened).

    One tool call turns "candidate address" into "the routine that writes
    it". Success is claimed ONLY when a hit is actually confirmed — via
    the checkpoint's hit count, or by the resolved PC's instruction
    demonstrably writing the watched address — never merely because a PC
    was read. The armed watchpoint is always cleaned up (try/finally).
    """
    address = _extract_vice_address(args)
    if address is None:
        return _record_result(
            "vice", step_id, False,
            "vice.trace requires an `address` to watch.",
            extra={"method": "vice.trace", "retryable": False},
        )
    addr_int = int(address.lstrip("$"), 16)
    frames = int(args.get("frames", 2))
    steps: list[str] = []
    checkpoint_id: Any = None

    # 1. Arm a write watchpoint.
    try:
        add_out = _vice_call("vice.checkpoint.add", {
            "address": address, "stop_when_hit": True,
            "load": False, "store": True, "exec": False,
        })
        checkpoint_id = _extract_checkpoint_id(add_out)
        steps.append(
            f"armed write watchpoint at {address}"
            + (f" (id={checkpoint_id})" if checkpoint_id is not None else "")
        )
    except Exception as e:  # noqa: BLE001
        return _record_result(
            "vice", step_id, False,
            f"vice.trace could not arm a watchpoint at {address}: "
            f"{type(e).__name__}: {e}",
            extra={"method": "vice.trace", "address": address},
        )

    try:
        # 2. Run (bounded by frames).
        try:
            _vice_call("vice.execution.run", {"frames": frames})
            steps.append(f"ran up to {frames} frame(s)")
        except Exception as e:  # noqa: BLE001
            steps.append(f"run failed ({type(e).__name__}) — checking state anyway")

        # 3. Confirm a hit via the checkpoint list (hit_count > 0), if the
        #    server exposes it. Only trusted when we captured OUR
        #    checkpoint id — without one, a hit_count on some other
        #    breakpoint could be misattributed to our watchpoint, so we
        #    fall back to the PC/disasm proof below instead (review nit).
        hit_confirmed = False
        if checkpoint_id is not None:
            try:
                lst = _vice_call("vice.checkpoint.list", {})
                hit_confirmed = _checkpoint_hit(lst.get("data"), checkpoint_id)
                if hit_confirmed:
                    steps.append(f"checkpoint id={checkpoint_id} reports a hit")
            except Exception:  # noqa: BLE001
                pass  # optional verb
        else:
            steps.append(
                "no checkpoint id returned — relying on PC/disasm proof, "
                "not the checkpoint list"
            )

        # 4. Read registers (allowed — a checkpoint is armed) + resolve PC.
        pc = None
        reg_text = ""
        try:
            regs = _vice_call("vice.registers.get", {})
            reg_text = _vice_text(regs.get("data"))
            m = re.search(r"\bPC[:=\s]*\$?([0-9A-Fa-f]{4})", reg_text)
            if m:
                pc = int(m.group(1), 16)
            steps.append("read registers")
        except Exception as e:  # noqa: BLE001
            steps.append(f"registers.get failed ({type(e).__name__})")

        # 5. Disasm around PC and check it actually writes the address.
        writer_disasm = ""
        writer_proven = False
        if pc is not None:
            win_start = max(0, pc - 16)
            try:
                out = _vice_call("vice.disassemble", {
                    "address": f"${win_start:04X}", "count": 16,
                })
                writer_disasm = _vice_text(out.get("data"))
                writer_proven = _disasm_writes_addr(writer_disasm, addr_int)
                steps.append(
                    f"disasm around ${pc:04X}"
                    + (" confirms a write to the address"
                       if writer_proven else " (no write to the address seen)")
                )
            except Exception as e:  # noqa: BLE001
                steps.append(f"disasm around ${pc:04X} failed ({type(e).__name__})")

        ok = bool(hit_confirmed or writer_proven)
        body = [
            f"vice.trace of write watchpoint on {address}:",
            "steps: " + "; ".join(steps),
        ]
        if reg_text:
            body.append("registers:\n" + reg_text[:1_000])
        if writer_disasm:
            body.append(
                f"writing instruction context (±16 bytes around ${pc:04X}):\n"
                + writer_disasm[:2_000]
            )
        if not ok:
            body.append(
                "No confirmed write to the address within the frame budget "
                "(no hit reported and the PC's instruction does not write it). "
                "The value may be updated elsewhere or not during this window."
            )
        return _record_result(
            "vice", step_id, ok, "\n".join(body),
            extra={"method": "vice.trace", "address": address,
                   "writer_pc": f"${pc:04X}" if (ok and pc is not None) else None,
                   "hit_confirmed": bool(hit_confirmed),
                   "writer_proven": bool(writer_proven)},
        )
    finally:
        # Always disarm the watchpoint we added, even on early returns.
        try:
            del_args = ({"id": checkpoint_id} if checkpoint_id is not None
                        else {"address": address})
            _vice_call("vice.checkpoint.delete", del_args)
        except Exception:  # noqa: BLE001
            pass


def _checkpoint_hit(data: Any, checkpoint_id: Any) -> bool:
    """Parse a checkpoint-list payload for OUR checkpoint's hit count.

    Requires a concrete `checkpoint_id` and an exact id match: without a
    known id, a hit on some unrelated breakpoint could be misattributed
    to our watchpoint (review nit), so the caller uses PC/disasm proof
    instead. Belt-and-braces with the caller's own id guard.
    """
    if checkpoint_id is None:
        return False
    if isinstance(data, dict):
        items = data.get("checkpoints") or data.get("list") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = it.get("id") if it.get("id") is not None else it.get("number")
        if cid != checkpoint_id:
            continue
        hits = it.get("hit_count", it.get("hits", it.get("hit", 0)))
        if (isinstance(hits, bool) and hits) or (
            isinstance(hits, (int, float)) and hits > 0
        ):
            return True
    return False


def vice_mcp_node(state: C64State) -> dict[str, Any]:
    """Direct vice-mcp call (when the planner explicitly picks `vice`).

    The step's ``args`` dict should include a ``method`` key (e.g.
    ``disassemble``, ``memory.read``, ``registers.get``). As a fallback
    the node also accepts ``method`` at the step's top level (LLMs
    sometimes emit it there). Address-bearing methods (``disassemble`` /
    ``memory.read``) MUST also include an address — the previous code
    silently defaulted to ``$0801``, which produced plausible-looking but
    wrong tool results. We now error out loudly so the planner can
    self-correct on the next iteration.
    """
    step = _step_for(state, state.get("current_step_id"))
    args = dict(step.get("args", {}) or {})
    # LLMs sometimes put `method` inside args (per schema) or at the step
    # top level — accept both.  Also absorb top-level address/count/size
    # keys the LLM may have placed at step level instead of inside args.
    import logging as _logging
    _logging.getLogger(__name__).debug(
        "vice_mcp_node raw step: %s", json.dumps(step, default=str)
    )
    raw_method = (
        args.pop("method", None)
        or step.get("method")
        or step.get("action")
        or "vice.ping"
    )
    # Canonicalise once, before policy checks or composite dispatch. Calling
    # `_vice_call` later normalises again defensively, but policy must never
    # inspect a raw alias (`ping`, `registers`, `get_registers`, …).
    method = _vice_normalize_method(str(raw_method))
    # Absorb address / count / size at the step top level into args when
    # the LLM forgot to nest them (common with smaller models).
    for _top_key in ("address", "addr", "count", "size", "bank", "start"):
        if _top_key in step and _top_key not in args:
            args[_top_key] = step[_top_key]
    step_id = step.get("id", "?")

    if (
        state.get("require_vice_approval")
        and vice_method_requires_approval(method)
        and not state.get("approve_all_vice_mutations")
        and str(step_id) not in {
            str(item) for item in state.get("approved_mutation_steps") or []
        }
    ):
        return _record_result(
            "vice", step_id, False,
            f"{method} can mutate emulator state and requires explicit "
            "human approval before execution.",
            extra={
                "method": method,
                "args": args,
                "rejection": "human_approval_required",
                "retryable": False,
            },
        )

    if not os.getenv("VICE_MCP_URL", "").strip():
        return _record_result(
            "vice", step_id, False,
            "VICE_MCP_URL not set — skipped (start vice-mcp server to enable)",
            extra={"method": method, "args": args, "retryable": False},
        )

    # Agent-side composites (tracker 3.1 / 3.2): snapshot / diff /
    # monotonic_scan / trace. Dispatch before the direct-call path.
    if method in _VICE_COMPOSITE_METHODS:
        return _vice_composite(state, method, args, step_id)

    # Reject low-value diagnostic calls that produce no code facts.
    # `vice.registers.get` is ALWAYS banned as a standalone step
    # (tracker 3.2 hardening — option (c)): a planner-supplied
    # `armed: true` is not trustworthy proof a checkpoint actually
    # hit, so it cannot bypass the ban. Registers ARE useful right
    # after a watchpoint hit — but only through `vice.trace`, which
    # arms the checkpoint in-process, reads registers via the internal
    # `_vice_call` path (not this node), and verifies the hit.
    _BANNED_VICE_METHODS = {"vice.ping", "vice.registers.get"}
    if method in _BANNED_VICE_METHODS:
        hint = (
            " Use `vice.trace {address}` — it arms a watchpoint, reads "
            "registers at the verified hit, and resolves the writer."
            if method == "vice.registers.get" else ""
        )
        return _record_result(
            "vice", step_id, False,
            f"{method} produces no code facts — skipped (banned method).{hint} "
            "Plan a vice.disassemble, vice.memory.read, or vice.trace step "
            "instead.",
            extra={"method": method, "args": args, "rejection": "banned_method"},
        )

    try:
        out = _vice_call(method, args)
    except ViceArgsError as e:
        # Missing required argument — surface a precise error so the
        # planner / executor see exactly what to fix on retry.
        return _record_result(
            "vice", step_id, False,
            f"vice-mcp arg error: {e}",
            extra={"method": method, "args": args, "rejection": "missing_arg"},
        )
    except Exception as e:  # noqa: BLE001
        return _record_result(
            "vice", step_id, False,
            f"vice-mcp call failed: {type(e).__name__}: {e}",
            extra={"method": method, "args": args},
        )

    # Successful call — normalise high-volume / multimodal payloads before
    # wrapping them in the deterministic tool-result envelope.
    data = out.get("data")
    final_args = out.get("args", args) or {}
    requested_addr = final_args.get("address") if isinstance(final_args, dict) else None
    canonical_method = str(out.get("tool", method))
    payload_meta: dict[str, Any] = {}
    remaining_budget_usd = _remaining_llm_budget(state)
    if canonical_method == "vice.memory.read":
        raw_text, payload_meta = _format_vice_memory_read(
            data,
            requested_addr,
            max_chars=4_000,
            remaining_budget_usd=remaining_budget_usd,
        )
    elif canonical_method == "vice.display.screenshot":
        data_uri = _screenshot_data_uri(data)
        if data_uri:
            screenshot_path, save_meta = _persist_screenshot(
                state, step_id, data_uri,
            )
            raw_text, vision_meta = _describe_screenshot_result(
                data_uri,
                remaining_budget_usd=remaining_budget_usd,
            )
            payload_meta.update(save_meta)
            payload_meta.update(vision_meta)
            if screenshot_path:
                # The persisted source image is intentionally also the UI
                # thumbnail target: generating a second raster would add an
                # otherwise-unused Pillow dependency.
                payload_meta["screenshot_path"] = screenshot_path
                payload_meta["thumbnail_path"] = screenshot_path
        else:
            raw_text = _vice_text(
                data, remaining_budget_usd=remaining_budget_usd,
            )
            payload_meta.update({
                "vision_role": "vision",
                "vision_ok": False,
                "vision_status": "missing_image",
                "vision_error": "VICE response contained no image data URI",
            })
    else:
        raw_text = _vice_text(
            data, remaining_budget_usd=remaining_budget_usd,
        )
    header = (
        f"[{canonical_method} "
        f"args={json.dumps(final_args, default=str)}]"
    )
    warning = _detect_vice_addr_mismatch(requested_addr, raw_text)
    body = raw_text[:4_000]
    pieces = [header]
    if warning:
        pieces.append(warning)
    pieces.append(body)
    text = "\n".join(pieces)

    extra: dict[str, Any] = {
        "method": canonical_method,
        "args": final_args,
        "url": out.get("url"),
        **payload_meta,
    }
    if warning:
        extra["addr_mismatch_warning"] = warning

    return _record_result("vice", step_id, True, text, extra=extra)


# ---- kb -------------------------------------------------------------------

KB_SCHEMA_HINT = (
    "Tables (SQLite). Column names are FIXED — DO NOT invent aliases.\n"
    "  events(id TEXT PK, ts TEXT, kind TEXT, source TEXT, payload_json TEXT)\n"
    "    kind ∈ {ingest_dump, ingest_partial_asm, ingest_text, tool_result,\n"
    "            label, routine, hypothesis, analysis, verdict,\n"
    "            data_structure, consolidated_observation, run_summary}\n"
    "  labels(addr INTEGER, name TEXT, kind TEXT, confidence REAL,\n"
    "         source_event_id TEXT)\n"
    "    NOTE: the column is `addr`, NOT `address` / `address_hex` /\n"
    "    `addr_hex`. There is no separate hex column — format hex in\n"
    "    your client (or use kb mode='labels' which adds addr_hex).\n"
    "  routines(start INTEGER PK, end INTEGER, name, summary,\n"
    "           calls_to_json, called_by_json, confidence REAL)\n"
    "    NOTE: the columns are `start`/`end`, NOT `start_addr`/`end_addr`.\n"
    "    Replay is confidence-aware: re-emitting a label/routine with\n"
    "    LOWER confidence does not overwrite the stored fact.\n"
    "  data_structures(start INTEGER PK, end INTEGER, kind, fields_json)\n"
    "  hypotheses(id TEXT PK, text, status, evidence_json)\n"
    "  text_docs(path TEXT PK, ts, size, mtime, content TEXT,\n"
    "            source_event_id) -- user notes from --text-dir\n"
    "  run_summaries(run_id TEXT PK, event_id, completed_at, question,\n"
    "                verdict, confidence, iterations, cost_usd, tokens,\n"
    "                llm_calls, tool_calls, elapsed_s, termination_reason,\n"
    "                answer_excerpt, report_path)\n"
    "  meta(key TEXT PK, value TEXT)\n"
    "Useful queries:\n"
    "  - addresses near a label:\n"
    "      SELECT addr, name, confidence FROM labels\n"
    "      WHERE name LIKE '%rand%' OR name LIKE '%prng%' LIMIT 50;\n"
    "  - last N tool_result events:\n"
    "      SELECT id, ts, source, substr(payload_json,1,200) AS preview\n"
    "      FROM events WHERE kind='tool_result' ORDER BY ts DESC LIMIT 10;\n"
    "  - find capstone disassembly results in events:\n"
    "      SELECT id, ts, substr(payload_json,1,400) FROM events\n"
    "      WHERE kind='tool_result' AND payload_json LIKE '%\"tool\": \"capstone\"%'\n"
    "      ORDER BY ts DESC LIMIT 5;\n"
    "  - search user-provided text knowledge:\n"
    "      use mode='text' with q='<keyword>' — IMPORTANT: q must be a\n"
    "      SHORT keyword or short phrase (1-3 words) that will literally\n"
    "      appear in the file; e.g. q='planet' or q='BOW-BOW' or q='$0780'.\n"
    "      Long natural-language queries like 'discovered planet coordinates'\n"
    "      return NO results because they must match verbatim as a substring.\n"
    "      If keyword search misses, use mode='text_semantic' + q='<natural\n"
    "      language>' for cosine similarity (see config/kb_semantic.json).\n"
    "      To browse all docs: mode='sql' with\n"
    "        SELECT path, substr(content,1,400) FROM text_docs;"
)


# Common column-name drift the planner LLM falls into. Each rewrite is
# applied as an outside-of-string-literal substitution before the SQL
# executes; if the rewrite changes the SQL we retry once.
_KB_SQL_REWRITES: tuple[tuple[str, str], ...] = (
    # `address` is overwhelmingly the label-table mistake we keep seeing.
    (r"\baddress_hex\b", "addr"),
    (r"\baddress\b",     "addr"),
    (r"\baddr_hex\b",    "addr"),
    # Routines / data_structures use start/end, not start_addr/end_addr.
    (r"\bstart_addr\b",  "start"),
    (r"\bend_addr\b",    "end"),
    # Some LLMs output the JSON key name (`payload`) instead of the
    # actual column (`payload_json`).
    (r"\bpayload\b(?!_json)", "payload_json"),
    # `event_id` -> `id` when joining/selecting from events.
    (r"\bevent_id\b",    "id"),
)

# Reject any SQL that would mutate the KB. The KB is append-only via
# `KnowledgeStore.append_event`; raw write SQL would corrupt the
# replay invariant.
_KB_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|replace|"
    r"vacuum|pragma|reindex|truncate)\b",
    re.IGNORECASE,
)


def _rewrite_kb_sql(sql: str) -> tuple[str, list[str]]:
    """Apply the KB SQL rewrites *outside* of string literals.

    Returns ``(rewritten_sql, applied_rules)``. We tokenise on single
    quotes so column-name fixes don't accidentally clobber a literal
    such as ``WHERE name = 'address_table'``.
    """
    if not sql:
        return sql, []

    parts = re.split(r"('(?:[^']|'')*')", sql)  # keep '...'-literals intact
    applied: list[str] = []
    for i in range(0, len(parts), 2):  # even indices are non-literal
        seg = parts[i]
        for pattern, replacement in _KB_SQL_REWRITES:
            new_seg, n = re.subn(pattern, replacement, seg, flags=re.IGNORECASE)
            if n:
                applied.append(f"{pattern} -> {replacement} (×{n})")
                seg = new_seg
        parts[i] = seg
    return "".join(parts), applied


def kb_query_node(state: C64State) -> dict[str, Any]:
    """Query the persistent KB.

    Args:
        mode: "stats" (default) | "schema" | "sql" | "labels" | "events" |
             "text" | "text_semantic"
        sql: required when mode='sql'
        kind: optional filter for mode='events'
        limit: optional row cap (default 50)
        like: optional substring filter for mode='labels' (matches name)
        q:    required for mode='text' / 'text_semantic' —
              substring (text) / natural-language similarity (text_semantic)
        semantic: if true with mode='text', delegated to semantic search instead
          score_threshold: optional float for mode='text_semantic' (minimum
                     cosine score to keep a hit)
    """
    step = _step_for(state, state.get("current_step_id"))
    args = step.get("args", {}) or {}
    step_id = step.get("id", "?")
    if not state.get("kb_handle"):
        return _record_result(
            "kb", step_id, False, "no KB handle",
            extra={"retryable": False},
        )

    store = get_store(state["kb_handle"])
    mode = str(args.get("mode") or ("sql" if args.get("sql") else "stats")).lower()

    semantic_flag = args.get("semantic")
    semantic_on = semantic_flag is True or str(semantic_flag).strip().lower() in {
        "1", "true", "yes", "on",
    }
    if mode == "text" and semantic_on:
        mode = "text_semantic"

    limit = int(args.get("limit", 50))

    try:
        if mode == "schema":
            return _record_result(
                "kb", step_id, True, KB_SCHEMA_HINT,
                extra={"mode": "schema"},
            )

        if mode == "sql":
            sql = args.get("sql")
            if not sql:
                return _record_result(
                    "kb", step_id, False,
                    "kb mode='sql' requires args.sql.\n\n" + KB_SCHEMA_HINT,
                    extra={"mode": "sql", "schema": KB_SCHEMA_HINT},
                )
            sql = str(sql).strip().rstrip(";")

            if _KB_SQL_FORBIDDEN.search(sql):
                return _record_result(
                    "kb", step_id, False,
                    "kb mode='sql' is read-only — INSERT/UPDATE/DELETE/DDL "
                    "is rejected. Append-only writes go through "
                    "KnowledgeStore.append_event (synthesizer only).",
                    extra={"mode": "sql", "sql": redact_text(sql),
                           "rejection": "forbidden_statement"},
                )

            rewritten, applied_rewrites = _rewrite_kb_sql(sql)
            sql_history: list[dict[str, Any]] = []
            last_err: Exception | None = None

            for attempt_sql, label in (
                (sql, "original"),
                *([(rewritten, "auto-rewrite")]
                  if applied_rewrites and rewritten != sql else []),
            ):
                try:
                    rows = store.query(attempt_sql)
                    safe_rows = redact_value(rows[:limit])
                    text = json.dumps(safe_rows, indent=2, default=str)
                    extra: dict[str, Any] = {
                        "mode": "sql",
                        "sql": redact_text(attempt_sql),
                        "row_count": len(rows),
                    }
                    if label != "original":
                        extra["sql_was_auto_rewritten_from"] = redact_text(sql)
                        extra["sql_rewrites_applied"] = applied_rewrites
                    if sql_history:
                        extra["sql_attempts_before_success"] = redact_value(
                            sql_history,
                        )
                    return _record_result(
                        "kb", step_id, True, text, extra=extra,
                    )
                except Exception as e:  # noqa: BLE001
                    sql_history.append({
                        "label": label,
                        "sql": attempt_sql,
                        "error": f"{type(e).__name__}: {e}",
                    })
                    last_err = e

            # Both the original and (any) rewritten SQL failed. Surface a
            # detailed error + the schema cheat-sheet so the planner can
            # self-correct on the next iteration.
            err_msg = (
                f"{type(last_err).__name__ if last_err else 'Error'}: "
                f"{last_err if last_err else 'unknown SQL error'}\n\n"
                + ("Auto-rewrite tried: "
                   + ", ".join(applied_rewrites) + "\n\n"
                   if applied_rewrites else "")
                + KB_SCHEMA_HINT
            )
            return _record_result(
                "kb", step_id, False, redact_text(err_msg),
                extra={"mode": "sql", "sql": redact_text(sql),
                       "sql_attempts": redact_value(sql_history),
                       "schema": KB_SCHEMA_HINT},
            )

        if mode == "labels":
            like = args.get("like")
            addresses: list[int] = []
            for raw_addr in (
                args.get("addr"), args.get("address"),
                *(args.get("addresses") or []
                  if isinstance(args.get("addresses"), list) else []),
            ):
                if raw_addr is None:
                    continue
                parsed = _hex_to_int(raw_addr, -1)
                if 0 <= parsed <= 0xFFFF and parsed not in addresses:
                    addresses.append(parsed)
            addresses.extend(
                a for a in extract_hex_addresses(str(like or ""))
                if a not in addresses
            )

            where: list[str] = []
            params: list[Any] = []
            if addresses:
                where.append(
                    "addr IN (" + ",".join("?" for _ in addresses) + ")"
                )
                params.extend(addresses)
            # An address-looking `like` is an address query, not a label-name
            # substring (tracker 4.3).
            if like and not extract_hex_addresses(str(like)):
                where.append("name LIKE ?")
                params.append(f"%{like}%")

            if where:
                order_sql = "confidence DESC, addr"
                if addresses:
                    order_sql = (
                        "CASE WHEN addr IN ("
                        + ",".join("?" for _ in addresses)
                        + ") THEN 0 ELSE 1 END, confidence DESC, addr"
                    )
                    params.extend(addresses)
                params.append(limit)
                rows = store.query(
                    "SELECT addr, name, kind, confidence FROM labels"
                    f" WHERE {' OR '.join(where)}"
                    f" ORDER BY {order_sql} LIMIT ?",
                    tuple(params),
                )
            else:
                rows = store.query(
                    "SELECT addr, name, kind, confidence FROM labels"
                    " ORDER BY confidence DESC, addr LIMIT ?",
                    (limit,),
                )
            for r in rows:
                if isinstance(r.get("addr"), int):
                    r["addr_hex"] = f"${r['addr']:04X}"
            text = json.dumps(rows, indent=2, default=str)
            text = redact_text(text)
            return _record_result(
                "kb", step_id, True, text,
                extra={
                    "mode": "labels", "row_count": len(rows),
                    **({"addresses": [f"${a:04X}" for a in addresses]}
                       if addresses else {}),
                },
            )

        if mode == "text":
            q = args.get("q") or args.get("query") or ""
            if not q:
                return _record_result(
                    "kb", step_id, False,
                    "kb mode='text' requires args.q (search keywords).",
                    extra={"mode": "text"},
                )
            hits = store.search_text(str(q), limit=limit)
            if not hits:
                doc_count = store.text_doc_count()
                text = (
                    f"No matches for {q!r} across {doc_count} text docs.\n"
                    "Tip: ingest more notes via `--text-dir <path>`."
                )
            else:
                lines = [f"{len(hits)} hit(s) for {q!r}:"]
                for h in hits:
                    lines.append(
                        f"- {h['path']} (size={h.get('size')}, "
                        f"match_at={h['match_at']})"
                    )
                    lines.append(f"  …{h['snippet']}…")
                text = "\n".join(lines)
            text = redact_text(text)
            return _record_result(
                "kb", step_id, True, text,
                extra={
                    "mode": "text", "q": redact_text(q),
                    "hits": redact_value(hits),
                },
            )

        if mode == "text_semantic":
            q = args.get("q") or args.get("query") or ""
            if not q:
                return _record_result(
                    "kb", step_id, False,
                    "kb mode='text_semantic' requires args.q (question).",
                    extra={"mode": "text_semantic"},
                )
            if not store.semantic_search_enabled():
                return _record_result(
                    "kb", step_id, False,
                    "Semantic KB inactive. Set `\"enabled\": true` and a valid "
                    "`use_llm_provider` in config/kb_semantic.json whose "
                    "provider exposes `/v1/embeddings` "
                    "(e.g. OpenAI key on the executor role).\n\n"
                    "Keyword search stays available via mode='text'.",
                    extra={
                        "mode": "text_semantic",
                        "semantic_ready": False,
                    },
                )
            score_threshold = args.get("score_threshold")
            try:
                threshold = (
                    None
                    if score_threshold is None
                    else float(score_threshold)
                )
            except (TypeError, ValueError):
                threshold = None

            hits = store.search_semantic(
                str(q),
                limit=limit,
                score_threshold=threshold,
            )
            if not hits:
                text = (
                    f"No semantic hits for {q!r}. "
                    f"(index has {store.stats().get('semantic_kb_chunks')} chunks)."
                )
            else:
                lines = [
                    f"{len(hits)} semantic hit(s) for {q!r} "
                    f"(embedding similarity):",
                ]
                for h in hits:
                    body = (
                        str(h.get("text") or h.get("snippet_line") or "")
                    )
                    excerpt = (
                        body if len(body) <= 620
                        else body[:600] + "…"
                    )
                    lines.append(
                        f"- id={h.get('id')} score={h.get('score')} "
                        f"facet={(h.get('meta') or {}).get('facet')}\n"
                        f"  {excerpt}",
                    )
                text = "\n".join(lines)
            text = redact_text(text)
            return _record_result(
                "kb", step_id, True, text,
                extra={
                    "mode": "text_semantic",
                    "q": redact_text(q),
                    "score_threshold": threshold,
                    "hits": redact_value(hits),
                },
            )

        if mode == "events":
            kind = args.get("kind")
            if kind:
                rows = store.query(
                    "SELECT id, ts, kind, source, substr(payload_json,1,400) AS preview"
                    " FROM events WHERE kind = ? ORDER BY ts DESC LIMIT ?",
                    (kind, limit),
                )
            else:
                rows = store.query(
                    "SELECT id, ts, kind, source, substr(payload_json,1,400) AS preview"
                    " FROM events ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )
            safe_rows = redact_value(rows)
            text = json.dumps(safe_rows, indent=2, default=str)
            return _record_result(
                "kb", step_id, True, text,
                extra={"mode": "events", "row_count": len(rows)},
            )

        # default: stats
        s = store.stats()
        recent = store.query(
            "SELECT id, ts, kind, source FROM events"
            " ORDER BY ts DESC LIMIT 10"
        )
        text = json.dumps(
            {"stats": s, "recent_events": recent, "schema_hint": "use mode='schema' to see tables"},
            indent=2, default=str,
        )
        return _record_result(
            "kb", step_id, True, text, extra={"mode": "stats"},
        )

    except Exception as e:  # noqa: BLE001
        # Always echo the schema so the planner can self-correct on the next step.
        return _record_result(
            "kb", step_id, False,
            f"{type(e).__name__}: {e}\n\n{KB_SCHEMA_HINT}",
            extra={
                "mode": mode, "args": redact_value(args),
                "schema": KB_SCHEMA_HINT,
            },
        )
