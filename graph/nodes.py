"""Node implementations.

LLM nodes call the configured provider via `graph.llm.get_llm`. Tool
nodes are real implementations with graceful fallback when their
external dependency is unreachable (Tavily without API key, VICE-MCP
without a running server, etc.).

KB persistence is delegated to `memory.KnowledgeStore`; nodes never
touch sqlite/JSON directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from graph.llm import get_llm
from graph import usage as llm_usage
from graph.plan_utils import (
    MAX_CONSECUTIVE_REVISES,
    MAX_PLAN_STEPS,
    VICE_ADDRESS_ALIASES as _VICE_ADDRESS_ALIASES,
    failed_step_notes,
    pending_steps,
    resolve_session_slug,
    runnable_steps,
    slugify,
    step_is_concrete,
    step_status,
)
from graph.prompts import MASTER_PREAMBLE, system_message
from graph.state import C64State
from memory import KnowledgeStore, get_store
from memory.schema import (
    EVT_ANALYSIS,
    EVT_CONSOLIDATED,
    EVT_DATA_STRUCTURE,
    EVT_HYPOTHESIS,
    EVT_LABEL,
    EVT_ROUTINE,
    EVT_TOOL_RESULT,
    EVT_VERDICT,
)
from tools import c64_disasm

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"

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


def _invoke_one(
    role: str,
    prompt: str,
    *,
    system_role: str | None = None,
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


def _safe_invoke(
    role: str,
    prompt: str,
    fallback: dict[str, Any],
    backup_roles: list[str] | None = None,
    *,
    allow_plan_steps_array: bool = False,
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

    for r in chain:
        content, err, entry = _invoke_one(r, prompt, system_role=role)
        if entry is not None:
            last_entry = entry
        if err:
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
    return {"_error": " ; ".join(errors) or "all LLM calls failed", **fallback}


def _usage_update() -> dict[str, Any]:
    """Drain pending LLM-usage entries into a node state delta (tracker 1.4).

    Merged into every LLM-calling node's return so `budget_used` (add
    reducer) and `llm_usage` accumulate in state; nested calls made while
    the node ran (backups, vision, auto-annotation) are attributed to it.
    """
    entries = llm_usage.drain()
    if not entries:
        return {}
    return {
        "llm_usage": entries,
        "budget_used": sum(float(e.get("cost_usd") or 0.0) for e in entries),
        # Token fallback budget (review finding 3): always populated, so
        # runs on unpriced models still hit a runtime cap.
        "tokens_used": sum(
            int(e.get("input_tokens") or 0) + int(e.get("output_tokens") or 0)
            for e in entries
        ),
    }


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


def _kb_digest_for_state(state: C64State) -> str:
    """Build (or reuse) the question-relevant KB digest for this state."""
    handle = state.get("kb_handle")
    if not handle:
        return "_(no KB handle yet)_"
    store = get_store(handle)
    return store.digest_for_question(
        state.get("question", "") or "", max_chars=32_000,
    )


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
_BOOKKEEPING_EVENT_KINDS = (EVT_ANALYSIS, EVT_VERDICT, EVT_CONSOLIDATED)


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
        rows = store.query(
            "SELECT kind, payload_json FROM events WHERE kind NOT IN (?, ?, ?)",
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

    for idx, step in enumerate(plan, start=1):
        old_id = str(step.get("id") or f"s{idx}")
        new_id = old_id if old_id.startswith(prefix) else f"{prefix}{old_id}"
        id_map[old_id] = new_id
        s = dict(step)
        s["id"] = new_id
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
    initial_digest = store.digest_for_question(
        state.get("question", "") or "", max_chars=32_000,
    )

    return {
        "kb_handle": kb_handle,
        "code_kb_handle": code_kb_handle,
        "kb_digest": initial_digest,
        "iteration": 0,
        "replan_count": 0,
        "revise_count": 0,
        "budget_used": 0.0,
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
    """Resolve KB event IDs in the evidence list to human-readable lines.

    Each entry may be:
    - a file path (starts with '/' or contains path separators) → shown as-is
    - a KB event ID → resolved to kind + address/name/summary from payload
    - anything else → shown verbatim
    """
    if not evidence:
        return "_none recorded_"

    lines: list[str] = []
    for item in evidence:
        s = str(item).strip()
        if not s:
            continue

        # File path — show as a relative link
        if s.startswith("/") or (len(s) > 4 and ("/" in s or "\\" in s)):
            lines.append(f"- {s}")
            continue

        # Try to resolve as a KB event ID
        if store is not None:
            try:
                rows = store.query(
                    "SELECT kind, source, payload_json FROM events WHERE id = ?",
                    (s,),
                )
                if rows:
                    row = rows[0]
                    kind = row.get("kind", "?")
                    src = row.get("source", "?")
                    try:
                        payload = json.loads(row.get("payload_json") or "{}")
                    except Exception:
                        payload = {}

                    # Build a one-line human-readable summary from the payload
                    if kind == "label":
                        addr = payload.get("addr")
                        name = payload.get("name", "?")
                        desc = f"label `{name}` @ ${addr:04X}" if isinstance(addr, int) else f"label `{name}`"
                    elif kind == "routine":
                        start = payload.get("start")
                        name = payload.get("name") or "?"
                        summary = (payload.get("summary") or "")[:120]
                        desc = f"routine `{name}` @ ${start:04X}" if isinstance(start, int) else f"routine `{name}`"
                        if summary:
                            desc += f" — {summary}"
                    elif kind == "tool_result":
                        tool = payload.get("tool", "?")
                        step = payload.get("step_id", "?")
                        data = str(payload.get("data", ""))[:120].replace("\n", " ")
                        desc = f"tool result `{tool}/{step}`: {data}"
                    elif kind == "hypothesis":
                        text = (payload.get("text") or "")[:160]
                        status = payload.get("status", "?")
                        desc = f"hypothesis ({status}): {text}"
                    else:
                        # Generic fallback: show kind + first 160 chars of payload
                        preview = json.dumps(payload, default=str)[:160]
                        desc = f"{kind} ({src}): {preview}"

                    lines.append(f"- `{s}` — {desc}")
                    continue
            except Exception:  # noqa: BLE001
                pass

        # Unknown ID or no store — show raw
        lines.append(f"- `{s}`")

    return "\n".join(lines) if lines else "_none recorded_"


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

    tr_lines = []
    for r in state.get("tool_results", [])[-20:]:
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
        from graph.routers import BUDGET_CAP, token_budget
        token_cap = token_budget()
        enforced = []
        if budget_used >= BUDGET_CAP:
            enforced.append(f"USD cap (${BUDGET_CAP:.2f})")
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
            + f"- estimated cost: ${budget_used:.4f} / ${BUDGET_CAP:.2f} (USD cap)\n"
            + f"- calls priced/unpriced: {priced_calls}/{unpriced_calls}\n"
            + f"- budget limit enforced: {', '.join(enforced) or 'none'}"
            + pricing_note
        )
    else:
        usage_block = "_no LLM usage recorded_"

    all_tool_results = state.get("tool_results", []) or []
    tool_counts: dict[str, list[int]] = {}
    for r in all_tool_results:
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

**Verdict:** `{verdict}`  ·  **Confidence:** {confidence}  ·  **Iterations:** {state.get("iteration", 0)}

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
_generated {datetime.now(timezone.utc).isoformat()}_
"""

    report_path = out_dir / "report.md"
    report_path.write_text(report)

    return {
        "messages": [AIMessage(content=f"Wrote report → {report_path}")],
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
    )
    if isinstance(out, list):
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

    if not runnable:
        pending = [s for s in plan if str(s.get("id")) not in status.done]
        if not pending:
            return {"current_step_id": None, "plan_blocked": False}
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
            "plan_blocked": True,
            "tool_results": blocked_results,
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
_MECHANICAL_CAPSTONE_MODES = {"find_entry", "entry", "vectors"}


def _llm_worthy_result(r: dict[str, Any]) -> bool:
    """Only free-text evidence needs the extraction LLM (tracker 1.2).

    Failed results carry no facts; `kb`/`code_kb` results are reads of
    stores whose facts are already persisted; mechanical capstone modes
    are handled deterministically. What remains — disassembly listings,
    vice output, tavily snippets, screenshot descriptions — is prose.
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
    if not result.get("ok") or str(result.get("tool") or "").lower() != "capstone":
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
            store.append_event(EVT_HYPOTHESIS, "synthesizer_mechanical", {
                # Deterministic id: re-detection updates instead of duplicating.
                "id": f"h_loop_{addr:04x}",
                "text": (
                    f"Main-loop candidate at ${addr:04X} "
                    f"(score {c.get('score')}): {reasons}"
                ),
                "status": "open",
                "evidence": [ev_id],
                "provenance": "mechanical",
            })
            counts["hypotheses"] += 1

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
    all_results = state.get("tool_results", []) or []

    # ---- Stage 1: idempotent recording of raw tool_results ----
    new_raw: list[dict[str, Any]] = []
    new_event_ids: list[str] = []
    for r in all_results:
        if store.has_tool_result(r):
            continue
        new_event_ids.append(
            store.append_event(EVT_TOOL_RESULT, "synthesizer", r),
        )
        new_raw.append(r)

    # ---- Stage 1.5: mechanical extraction (structured outputs) ----
    mech_counts = {"labels": 0, "hypotheses": 0}
    for ev_id, r in zip(new_event_ids, new_raw):
        for k, v in _mechanical_extract(store, r, ev_id).items():
            mech_counts[k] += v

    # ---- Stage 2: batched LLM extraction of structured facts ----
    processed = int(state.get("synth_processed_count", 0) or 0)
    unprocessed = all_results[processed:]
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
    if not worthy and unprocessed:
        # Nothing LLM-worthy in the tail — mark it considered.
        new_processed_count = len(all_results)

    if flush:
        new_processed_count = len(all_results)
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
                    ANN_LABEL as _ANN_LABEL,
                    ANN_ROUTINE as _ANN_ROUTINE,
                    ANN_XREF as _ANN_XREF,
                )
                from code_kb.store import Annotation as _Ann
                _code_store = _get_code_store(state["code_kb_handle"])
            except Exception:  # noqa: BLE001
                pass

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
                _code_store.append_annotation(
                    _Ann(
                        layer=1,
                        kind=_ANN_LABEL,
                        start_addr=addr,
                        end_addr=addr,
                        producer="synthesizer",
                        confidence=confidence,
                        payload={"name": name, "source_file": "llm_synthesizer"},
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
            store.append_event(EVT_ROUTINE, "synthesizer", {
                "start": start,
                "end": end,
                "name": rt.get("name"),
                "summary": rt.get("summary"),
                "calls_to": calls_to,
                "called_by": called_by,
                "confidence": confidence,
                "provenance": "llm",
            })

            # Mirror into code_kb so the call graph + routines panel in
            # the web GUI see LLM-discovered routines, not only parser output.
            if _code_store is not None:
                _code_store.append_annotation(
                    _Ann(
                        layer=1,
                        kind=_ANN_ROUTINE,
                        start_addr=start,
                        end_addr=end,
                        producer="synthesizer",
                        confidence=confidence,
                        payload={
                            "name": rt.get("name"),
                            "summary": rt.get("summary"),
                            "source_file": "llm_synthesizer",
                            "entries": [start],
                            "exits": [],
                            "size_bytes": max(1, end - start + 1),
                        },
                        evidence=[rt.get("summary") or ""],
                    ),
                    source="synthesizer",
                )
                # Write call edges so local_dot can traverse the graph.
                for callee in calls_to:
                    _code_store.append_annotation(
                        _Ann(
                            layer=1,
                            kind=_ANN_XREF,
                            start_addr=start,
                            end_addr=start,
                            producer="synthesizer",
                            confidence=confidence,
                            payload={
                                "src_addr": start,
                                "dst_addr": callee,
                                "xref_kind": "jsr",
                            },
                        ),
                        source="synthesizer",
                    )

            extracted_counts["routines"] += 1

            # Queue Layer-1 annotation candidates instead of firing a
            # hidden LLM call per routine (tracker 1.3). Candidates are
            # prioritised and capped after the extraction loops.
            if _code_store is not None and confidence >= 0.50:
                try:
                    existing = _code_store.query(
                        "SELECT confidence FROM hypotheses"
                        " WHERE start_addr = ? AND layer = 1"
                        " ORDER BY confidence DESC LIMIT 1",
                        (start,),
                    )
                    existing_conf = float(existing[0]["confidence"]) if existing else 0.0
                    if confidence > existing_conf:
                        relevance = _question_relevance(
                            state.get("question", "") or "",
                            f"{rt.get('name') or ''} {rt.get('summary') or ''}",
                        )
                        annotate_candidates.append(
                            (confidence * (1.0 + relevance), start),
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
        update["kb_digest"] = store.digest_for_question(
            state.get("question", "") or "", max_chars=32_000,
        )
    if new_processed_count is not None:
        update["synth_processed_count"] = new_processed_count

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

    # NB: `tool_results` uses an `add` reducer in C64State, so we cannot
    # shrink it from a node return. The analyst is shielded from blow-up
    # because it reads via `digest_for_question` (which caps recent
    # excerpts), not by walking `state["tool_results"]` directly.
    return {
        "kb_digest": store.digest_for_question(
            state.get("question", "") or "", max_chars=32_000,
        ),
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
    )

    decision = out.get("decision") or fallback["decision"]
    if decision not in {"accept", "revise", "replan"}:
        decision = fallback["decision"]

    critique = out.get("critique") or fallback["critique"]
    suggested = out.get("suggested_steps") or []
    optional_followups = list(out.get("optional_followups") or [])

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
    m = (raw_method or "").strip()
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
        # ---- breakpoints ----
        "checkpoint_add": "vice.checkpoint.add",
        "breakpoint": "vice.checkpoint.add",
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
        }
        if a.get("bank"):
            out["bank"] = str(a["bank"])
        return out

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
    start = _hex_to_int(args.get("start", "0x0801"), 0x0801)
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
            f"#{i + 1} {c['address']} score={c['score']} — {' | '.join(c['reasons'])}"
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
    return _record_result(
        "capstone", step_id, True, json.dumps(info, indent=2),
        extra={"mode": "vectors", "info": info},
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
            start = _hex_to_int(args.get("start", "0x0801"), 0x0801)
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

        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=5,
                             include_domains=["csdb.dk", "codebase64.org",
                                              "lemon64.com", "gamebase64.com"])
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
            extra={"query": query, "results": items},
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
    out = vice_mcp.call_tool(tool_name, tool_args)
    out["args"] = tool_args
    out["tool"] = tool_name
    if out.get("is_error"):
        raise RuntimeError(
            f"vice-mcp returned error for {tool_name}: "
            f"{out.get('data') or out.get('raw_content')}"
        )
    return out


def _describe_screenshot(data_uri: str) -> str:
    """Send a C64 screenshot to a vision-capable LLM and return a text description.

    The description is stored as the tool result so the synthesizer can
    extract spatial/visual facts (sprites visible, score layout, colour
    RAM usage, etc.) without any downstream code needing multimodal support.

    Tries the analyst role first (most likely to be a capable vision model
    like Claude or GPT-4o), then falls back to the critic role.
    Gracefully degrades to a plain placeholder when no vision LLM is
    configured or the call errors.
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
    for role in ("analyst", "critic", "planner"):
        model_name = None
        try:
            llm = get_llm(role)
            model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
            sys_prompt = system_message(role)
            multimodal_msg = HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt_text},
            ])
            resp = llm.invoke([SystemMessage(sys_prompt), multimodal_msg])
            in_tok, out_tok = llm_usage.extract_usage(resp)
            description = _flatten_lc_ai_message_content(resp).strip()
            # ok = usable output produced (review finding 5) — an empty
            # description falls through to the next role and is a failure.
            llm_usage.record({
                "role": f"vision:{role}", "model": model_name,
                "input_tokens": in_tok, "output_tokens": out_tok,
                "cost_usd": llm_usage.estimate_cost_usd(
                    model_name, in_tok, out_tok,
                ),
                "ok": bool(description), "transport_ok": True,
                **({} if description else {"error": "empty description"}),
            })
            if description:
                return f"[screenshot description via {role}]\n{description}"
        except Exception as e:  # noqa: BLE001
            llm_usage.record({
                "role": f"vision:{role}", "model": model_name,
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "ok": False, "transport_ok": False,
                "error": f"{type(e).__name__}: {e}"[:200],
            })
            print(
                f"[vice/screenshot] vision call via {role} failed: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr, flush=True,
            )
    # All vision attempts failed — fall back to a size descriptor.
    b64_len = len(data_uri) - (data_uri.index(",") + 1) if "," in data_uri else 0
    return f"[screenshot: base64 PNG, {b64_len} chars — vision LLM unavailable]"


def _vice_text(data: Any) -> str:
    """Best-effort flatten of a vice-mcp response payload into a string."""
    if isinstance(data, str):
        # Screenshot data-URI: defer to the vision helper rather than discarding.
        if data.startswith("data:image/"):
            return _describe_screenshot(data)
        return data
    if isinstance(data, dict):
        # Screenshot response: {status, format, data_uri, base64, size}
        if "data_uri" in data:
            return _describe_screenshot(data["data_uri"])
        if "base64" in data:
            # Reconstruct a data-URI so the vision helper has a standard input.
            fmt = (data.get("format") or "PNG").lower()
            data_uri = f"data:image/{fmt};base64,{data['base64']}"
            return _describe_screenshot(data_uri)
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
        parts = [_vice_text(d) for d in data]
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
    method = (
        args.pop("method", None)
        or step.get("method")
        or step.get("action")
        or "vice.ping"
    )
    # Absorb address / count / size at the step top level into args when
    # the LLM forgot to nest them (common with smaller models).
    for _top_key in ("address", "addr", "count", "size", "bank", "start"):
        if _top_key in step and _top_key not in args:
            args[_top_key] = step[_top_key]
    step_id = step.get("id", "?")

    if not os.getenv("VICE_MCP_URL", "").strip():
        return _record_result(
            "vice", step_id, False,
            "VICE_MCP_URL not set — skipped (start vice-mcp server to enable)",
            extra={"method": method, "args": args, "retryable": False},
        )

    # Reject low-value diagnostic calls that produce no code facts.
    _BANNED_VICE_METHODS = {"vice.ping", "vice.registers.get"}
    if method in _BANNED_VICE_METHODS:
        return _record_result(
            "vice", step_id, False,
            f"{method} produces no code facts — skipped (banned method). "
            "Plan a vice.disassemble or vice.memory.read step instead.",
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

    # Successful call — wrap the raw text with a deterministic header and
    # an optional mismatch warning.
    raw_text = _vice_text(out.get("data"))
    final_args = out.get("args", args) or {}
    requested_addr = final_args.get("address") if isinstance(final_args, dict) else None
    header = (
        f"[{out.get('tool', method)} "
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
        "method": out.get("tool", method),
        "args": final_args,
        "url": out.get("url"),
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
    "            data_structure, consolidated_observation}\n"
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
                    extra={"mode": "sql", "sql": sql,
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
                    text = json.dumps(rows[:limit], indent=2, default=str)
                    extra: dict[str, Any] = {
                        "mode": "sql",
                        "sql": attempt_sql,
                        "row_count": len(rows),
                    }
                    if label != "original":
                        extra["sql_was_auto_rewritten_from"] = sql
                        extra["sql_rewrites_applied"] = applied_rewrites
                    if sql_history:
                        extra["sql_attempts_before_success"] = sql_history
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
                "kb", step_id, False, err_msg,
                extra={"mode": "sql", "sql": sql,
                       "sql_attempts": sql_history,
                       "schema": KB_SCHEMA_HINT},
            )

        if mode == "labels":
            like = args.get("like")
            if like:
                rows = store.query(
                    "SELECT addr, name, kind, confidence FROM labels"
                    " WHERE name LIKE ? ORDER BY confidence DESC, addr LIMIT ?",
                    (f"%{like}%", limit),
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
            return _record_result(
                "kb", step_id, True, text,
                extra={"mode": "labels", "row_count": len(rows)},
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
            return _record_result(
                "kb", step_id, True, text,
                extra={"mode": "text", "q": q, "hits": hits},
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
            return _record_result(
                "kb", step_id, True, text,
                extra={
                    "mode": "text_semantic",
                    "q": q,
                    "score_threshold": threshold,
                    "hits": hits,
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
            text = json.dumps(rows, indent=2, default=str)
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
            extra={"mode": mode, "args": args, "schema": KB_SCHEMA_HINT},
        )
