"""Strict argument parsing shared by planner checks and tool adapters."""

from __future__ import annotations

import re
from typing import Any


class ToolArgsError(ValueError):
    pass


def address(value: Any) -> int:
    """Integers are numeric addresses; strings are hexadecimal, with optional prefix."""
    if isinstance(value, bool):
        raise ToolArgsError("address must be an integer or hexadecimal string")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("$"):
            text = text[1:]
        elif text.lower().startswith("0x"):
            text = text[2:]
        if not re.fullmatch(r"[0-9a-fA-F]{1,4}", text):
            raise ToolArgsError(f"invalid address {value!r}; use $0000..$FFFF")
        parsed = int(text, 16)
    else:
        raise ToolArgsError(f"invalid address {value!r}; use $0000..$FFFF")
    if not 0 <= parsed <= 0xFFFF:
        raise ToolArgsError(f"address {value!r} is outside $0000..$FFFF")
    return parsed


def address_arg(args: dict, *keys: str, default: int | None = None) -> int:
    for key in keys:
        if key in args:
            return address(args[key])
    if default is not None:
        return address(default)
    raise ToolArgsError("requires an address: " + " / ".join(keys))


def integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ToolArgsError(f"{name} must be an integer in {low}..{high}")
    try:
        text = str(value).strip()
        parsed = int(text[1:], 16) if text.startswith("$") else int(text, 16 if text.lower().startswith("0x") else 10)
    except ValueError:
        raise ToolArgsError(f"{name} must be an integer in {low}..{high}") from None
    if not low <= parsed <= high:
        raise ToolArgsError(f"{name} must be an integer in {low}..{high}")
    return parsed


def boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ToolArgsError(f"{name} must be true or false")
