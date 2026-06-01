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

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
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
    allowed = frozenset({"none", "low", "medium", "high"})
    configured = (
        agent_cfg_effort.strip().lower()
        if isinstance(agent_cfg_effort, str) and agent_cfg_effort.strip()
        else ""
    )
    return configured if configured in allowed else "low"


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


@lru_cache(maxsize=None)
def get_llm(role: str) -> ChatOpenAI:
    """Build (and cache) the chat client for a given sub-agent role.

    Roles are the keys under `agents:` in `config/llm.json` —
    e.g. `planner`, `executor`, `synthesizer`, `analyst`, `critic`,
    `curator`, `researcher`, `coordinator`.
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
    max_tok = defaults.get("max_tokens", 4096)
    # Reasoning-heavy OpenAI models can spend the whole completion budget on
    # internal reasoning if effort is unchecked, yielding empty visible `content`.
    # Extra headroom plus low reasoning effort avoids planner/analyst silent failure.
    if provider_name == "openai" and _is_openai_gpt5_reasoning_model(model_name):
        floor = int(os.environ.get("C64RE_GPT5_MIN_MAX_TOKENS", "8192"))
        max_tok = max(int(max_tok or 0), floor)

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
    if _supports_temperature(provider_name, model_name):
        kwargs["temperature"] = agent_cfg.get("temperature", 0.2)

    return ChatOpenAI(**kwargs)
