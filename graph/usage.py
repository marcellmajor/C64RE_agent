"""LLM usage / cost telemetry (tracker 1.4, hardened per review).

`budget_used` was checked by the router but never incremented, and there
was zero token visibility per role. Every LLM call now records a usage
entry through `record()`; each graph node drains the pending entries at
return time (`drain()`) and folds them into state (`llm_usage` list,
`budget_used` USD delta, `tokens_used` delta), where the router's budget
caps and the report's per-role table can see them.

**Scoping (review finding 2, hardened for asyncio):** pending entries
are held in a `contextvars.ContextVar` as an **immutable tuple** with
copy-on-write semantics — `record()` sets a NEW tuple, `drain()` sets
`()`; an inherited value is never mutated. This matters for asyncio:
context copies are shallow, so a mutable list stored in a ContextVar is
SHARED between a parent and its tasks — `append()`/`clear()` on it let
one task drain another's entries. With tuples, `set()` inside a task
only rebinds that task's context, so threads AND asyncio tasks are both
isolated, while nested calls within one graph node (same context) are
still collected by that node's drain. Entry dicts themselves stay
mutable on purpose: `record()` returns the live dict so `mark_failed()`
can amend it before the owning drain.

`reset()` clears the current context explicitly; `load_inputs` calls it
at run start so a reused execution context can never leak a prior run's
stragglers into a new run's telemetry.

**Failure accounting (review finding 5):** `ok` means "produced a
usable role-contract response", not "the HTTP call returned".
`transport_ok` tracks the latter separately. `record()` returns the
live stored dict so callers that detect contract failures only after
parsing (`_safe_invoke`) can demote the entry in place with
`mark_failed()` before the node drains.

Cost estimation is config-driven and honest: `config/llm.json` MAY
define a top-level ``"pricing"`` map of model-name prefixes →
``{"input_per_mtok": USD, "output_per_mtok": USD}``. Models without a
pricing entry contribute tokens (always tracked and enforced via the
token budget — see `graph.routers.token_budget`) but 0.0 USD — no
fabricated price table.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

# Immutable tuple + copy-on-write set(): NEVER a mutable container that
# a shallow context copy would share between asyncio tasks.
_PENDING_VAR: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
    "c64re_llm_usage_pending", default=(),
)


def record(entry: dict[str, Any]) -> dict[str, Any]:
    """Queue one LLM-call usage entry for the next drain in this context.

    Returns the *stored* dict so the caller can amend it in place
    (e.g. demote to a failure once contract parsing fails) before the
    owning node drains.
    """
    stored = dict(entry)
    _PENDING_VAR.set(_PENDING_VAR.get() + (stored,))
    return stored


def mark_failed(entry: dict[str, Any] | None, error: str) -> None:
    """Demote a previously-recorded entry to a failure, in place.

    Used when the failure is only detectable after the call returned
    (empty body already handled at record time; non-JSON / non-contract
    output detected during parsing in `_safe_invoke`).
    """
    if isinstance(entry, dict):
        entry["ok"] = False
        entry["error"] = str(error)[:200]


def drain() -> list[dict[str, Any]]:
    """Return and clear all pending usage entries in this context."""
    out = list(_PENDING_VAR.get())
    _PENDING_VAR.set(())
    return out


def reset() -> None:
    """Explicitly clear this context's pending entries.

    Called at run start (`load_inputs`) so a reused execution context —
    e.g. a pooled thread that previously crashed between record and
    drain — cannot leak prior-run entries into a new run's telemetry.
    """
    _PENDING_VAR.set(())


def extract_usage(msg: Any) -> tuple[int, int]:
    """Best-effort (input_tokens, output_tokens) from an AIMessage.

    Prefers LangChain's normalized ``usage_metadata``; falls back to the
    provider-shaped ``response_metadata["token_usage"]``.
    """
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict) and um:
        return (
            int(um.get("input_tokens") or 0),
            int(um.get("output_tokens") or 0),
        )
    meta = getattr(msg, "response_metadata", None) or {}
    tu = meta.get("token_usage") or meta.get("usage") or {}
    if isinstance(tu, dict):
        return (
            int(tu.get("prompt_tokens") or tu.get("input_tokens") or 0),
            int(tu.get("completion_tokens") or tu.get("output_tokens") or 0),
        )
    return (0, 0)


def estimate_cost_usd(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    *,
    pricing: dict[str, Any] | None = None,
) -> float:
    """USD estimate from the config's optional pricing map (0.0 if absent).

    Longest matching model-name prefix wins, so ``"gpt-5.3"`` can override
    a broader ``"gpt-5"`` entry.
    """
    if pricing is None:
        try:
            from graph.llm import load_config
            pricing = load_config().get("pricing") or {}
        except Exception:  # noqa: BLE001
            pricing = {}
    m = (model or "").lower()
    best_prefix = ""
    best_entry: dict[str, Any] | None = None
    for prefix, entry in (pricing or {}).items():
        p = str(prefix).lower()
        if m.startswith(p) and len(p) > len(best_prefix):
            best_prefix, best_entry = p, entry
    if not isinstance(best_entry, dict):
        return 0.0
    try:
        in_rate = float(best_entry.get("input_per_mtok") or 0.0)
        out_rate = float(best_entry.get("output_per_mtok") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def summarize_by_role(
    entries: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Aggregate usage entries per (role, model) for the report table.

    `failures` counts calls that did not yield a usable role-contract
    response (transport errors, empty bodies, non-JSON/contract output).
    `fallback_activations` counts chains that exhausted every role and
    degraded to the heuristic/raw-text fallback — flagged on the final
    attempt's entry (`fallback_activated=True`), never as an extra call —
    and is distinct from `backup_activations` (a backup role standing in).
    """
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for e in entries or []:
        key = (str(e.get("role") or "?"), str(e.get("model") or "?"))
        row = agg.setdefault(key, {
            "role": key[0], "model": key[1],
            "calls": 0, "failures": 0, "backup_activations": 0,
            "fallback_activations": 0,
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
        })
        row["calls"] += 1
        if not e.get("ok"):
            row["failures"] += 1
        if e.get("as_role"):
            row["backup_activations"] += 1
        if e.get("fallback_activated"):
            row["fallback_activations"] += 1
        row["input_tokens"] += int(e.get("input_tokens") or 0)
        row["output_tokens"] += int(e.get("output_tokens") or 0)
        row["cost_usd"] += float(e.get("cost_usd") or 0.0)
    return sorted(agg.values(), key=lambda r: (-r["cost_usd"], -r["calls"]))
