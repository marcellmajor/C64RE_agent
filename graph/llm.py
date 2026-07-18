"""Per-agent LLM client factory.

Reads `config/llm.json` (or `config/llm.yaml` as a fallback), expands
`${ENV_VAR}` placeholders, and returns a LangChain `ChatOpenAI` configured
against whichever OpenAI-compatible endpoint the agent is mapped to.

All five providers in the config (OpenAI, Anthropic, Gemini, xAI/Grok,
Ollama) expose OpenAI-compatible chat APIs, so a single client class
covers them all.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from c64re_agent.paths import config_dir

CONFIG_DIR = config_dir()
_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _is_openai_gpt5_reasoning_model(model_name: str) -> bool:
    """True for GPT-5 family reasoning models using the Responses-style stack.

    `gpt-5-chat` exempted: LangChain treats it like a chat model without the
    same reasoning/temperature coupling as flagship `gpt-5*` checkpoints.
    """
    m = (model_name or "").lower()
    return m.startswith("gpt-5") and "chat" not in m


def _reasoning_effort_openai(model_name: str, agent_cfg_effort: Any) -> str:
    """Pick a valid `reasoning.effort` for the Responses API — values differ by SKU.

    e.g. `gpt-5.5-pro` only allows ``medium``, ``high``, ``xhigh`` (not ``minimal``).
    Other GPT-5 models typically accept ``minimal`` … ``high``.
    """
    m = (model_name or "").lower()
    configured = (
        agent_cfg_effort.strip().lower()
        if isinstance(agent_cfg_effort, str) and agent_cfg_effort.strip()
        else ""
    )

    # 5.5 “pro” tier (and similar IDs): restrictive enum per OpenAI validation errors.
    if "gpt-5.5" in m:
        allowed = frozenset({"medium", "high", "xhigh"})
        if configured in allowed:
            return configured
        return "medium"

    # All other GPT-5 variants: "low" is universally accepted;
    # "minimal" is rejected by codex and certain other checkpoints.
    allowed = frozenset({"none", "low", "medium", "high", "xhigh"})
    if configured in allowed:
        return configured
    return "low"


def _is_grok_reasoning_model(model_name: str) -> bool:
    """True for grok models that support the `reasoning_effort` parameter.

    As of May 2026: grok-4.3 supports reasoning_effort
    ("none"/"low"/"medium"/"high").  grok-4.20-multi-agent uses the
    nested `reasoning.effort` form (handled separately).
    """
    m = (model_name or "").lower()
    # grok-4.3 family; exclude multi-agent variant (needs nested form)
    return ("grok-4" in m or m.startswith("grok-3")) and "multi-agent" not in m


def _reasoning_effort_grok(model_name: str, agent_cfg_effort: Any) -> str:
    """Pick a valid `reasoning_effort` for grok reasoning models.

    grok-4.3 accepts: "none", "low" (default), "medium", "high".
    Falls back to "low" (xAI default) when not configured or invalid.
    """
    # grok-4.5 rejects ``none``; older 4.x endpoints accepted it.
    allowed = (
        frozenset({"low", "medium", "high"})
        if "grok-4.5" in (model_name or "").lower()
        else frozenset({"none", "low", "medium", "high"})
    )
    configured = (
        agent_cfg_effort.strip().lower()
        if isinstance(agent_cfg_effort, str) and agent_cfg_effort.strip()
        else ""
    )
    return configured if configured in allowed else "low"


def _reasoning_effort_gemini(agent_cfg_effort: Any) -> str | None:
    """OpenAI-compat reasoning effort mapped by Gemini to thinking level."""
    configured = (
        agent_cfg_effort.strip().lower()
        if isinstance(agent_cfg_effort, str) and agent_cfg_effort.strip()
        else ""
    )
    if configured in {"minimal", "low", "medium", "high"}:
        return configured
    return None


def _supports_temperature(provider_name: str, model_name: str) -> bool:
    """Return whether passing `temperature` is generally safe for this model.

    Some provider/model combinations reject `temperature` entirely:
    - grok reasoning models (xAI)
    - anthropic/claude provider (claude-opus-4.x and newer deprecate it)
    - any model with 'reasoning' in its name
    """
    p = (provider_name or "").lower()
    m = (model_name or "").lower()
    # Both the old 'claude' and new 'anthropic' provider keys map to Anthropic.
    if p in {"claude", "anthropic", "grok"}:
        return False
    if "reasoning" in m:
        return False
    return True


def _expand_env(value: Any) -> Any:
    """Recursively replace `${VAR}` placeholders with env values."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    json_path = CONFIG_DIR / "llm.json"
    yaml_path = CONFIG_DIR / "llm.yaml"

    if json_path.exists():
        raw = json.loads(json_path.read_text())
    elif yaml_path.exists():
        import yaml  # lazy: only needed if json is missing

        raw = yaml.safe_load(yaml_path.read_text())
    else:
        raise FileNotFoundError(
            f"No LLM config found. Expected {json_path} or {yaml_path}."
        )

    return _expand_env(raw)


def structured_output_enabled(role: str) -> bool:
    """Resolve opt-in native JSON-schema output for one provider role.

    Environment override wins so a golden baseline can exercise all roles
    without editing config: ``C64RE_STRUCTURED_OUTPUT=1|0``.
    """
    env = os.getenv("C64RE_STRUCTURED_OUTPUT")
    if env is not None and env.strip():
        return _truthy(env)
    roles_env = os.getenv("C64RE_STRUCTURED_OUTPUT_ROLES", "").strip()
    if roles_env:
        enabled_roles = {
            item.strip() for item in roles_env.split(",") if item.strip()
        }
        return role in enabled_roles
    cfg = load_config()
    agent_cfg = (cfg.get("agents") or {}).get(role) or {}
    if "structured_output" in agent_cfg:
        return _truthy(agent_cfg.get("structured_output"))
    return _truthy((cfg.get("defaults") or {}).get("structured_output"))


@lru_cache(maxsize=None)
def get_llm(role: str) -> ChatOpenAI:
    """Build (and cache) the chat client for a given sub-agent role.

    Roles are the keys under `agents:` in `config/llm.json` —
    e.g. `planner`, `executor`, `synthesizer`, `analyst`, `critic`,
    `curator`, and `vision`.
    """
    cfg = load_config()
    try:
        agent_cfg = cfg["agents"][role]
    except KeyError as e:
        known = ", ".join(sorted(cfg.get("agents", {}).keys()))
        raise KeyError(f"Unknown agent role {role!r}. Known: {known}") from e

    provider_name = agent_cfg["provider"]
    provider = cfg["providers"][provider_name]
    defaults = cfg.get("defaults", {})

    model_name = str(agent_cfg["model"])
    # Output budgets are role-specific: the planner and analyst routinely
    # produce substantially larger structured payloads than the other roles.
    # Keep the global value as a fallback for old/custom configurations.
    max_tok = agent_cfg.get("max_tokens", defaults.get("max_tokens", 4096))
    try:
        max_tok = int(max_tok)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"agents.{role}.max_tokens must be an integer, got {max_tok!r}"
        ) from e
    if max_tok <= 0:
        raise ValueError(
            f"agents.{role}.max_tokens must be positive, got {max_tok!r}"
        )
    # Reasoning-heavy OpenAI models can spend the whole completion budget on
    # internal reasoning if effort is unchecked, yielding empty visible `content`.
    # Extra headroom plus low reasoning effort avoids planner/analyst silent failure.
    if provider_name == "openai" and _is_openai_gpt5_reasoning_model(model_name):
        floor = int(os.environ.get("C64RE_GPT5_MIN_MAX_TOKENS", "8192"))
        max_tok = max(max_tok, floor)

    # A live evaluation can impose a conservative completion ceiling without
    # mutating the checked-in role budgets. Apply it after model-specific
    # floors so the caller's explicit spend constraint remains authoritative.
    output_cap = os.getenv("C64RE_MAX_OUTPUT_TOKENS", "").strip()
    if output_cap:
        try:
            output_cap_int = int(output_cap)
        except ValueError as e:
            raise ValueError(
                "C64RE_MAX_OUTPUT_TOKENS must be a positive integer"
            ) from e
        if output_cap_int <= 0:
            raise ValueError("C64RE_MAX_OUTPUT_TOKENS must be a positive integer")
        max_tok = min(max_tok, output_cap_int)

    kwargs: dict[str, Any] = {
        "model": model_name,
        "base_url": provider["base_url"],
        "api_key": provider.get("api_key") or "missing",
        "max_tokens": max_tok,
        "timeout": defaults.get("timeout_s", 120),
        "max_retries": defaults.get("retries", 3),
    }
    if provider_name == "openai" and _is_openai_gpt5_reasoning_model(model_name):
        kwargs["reasoning_effort"] = _reasoning_effort_openai(
            model_name, agent_cfg.get("reasoning_effort")
        )
        cfg_verbosity = agent_cfg.get("verbosity")
        if isinstance(cfg_verbosity, str) and cfg_verbosity.strip():
            kwargs["verbosity"] = cfg_verbosity.strip()
    elif provider_name == "grok" and _is_grok_reasoning_model(model_name):
        kwargs["reasoning_effort"] = _reasoning_effort_grok(
            model_name, agent_cfg.get("reasoning_effort")
        )
    elif provider_name == "gemini":
        gemini_effort = _reasoning_effort_gemini(
            agent_cfg.get("reasoning_effort"),
        )
        if gemini_effort:
            kwargs["reasoning_effort"] = gemini_effort
    if _supports_temperature(provider_name, model_name):
        kwargs["temperature"] = agent_cfg.get("temperature", 0.2)

    return ChatOpenAI(**kwargs)
