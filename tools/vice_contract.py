"""Shared VICE contracts; no graph imports, I/O, or emulator side effects."""
from __future__ import annotations
import json
from typing import Any
from tools.arguments import ToolArgsError, address, address_arg, boolean, integer

ViceArgsError = ToolArgsError
NATIVE_TOOLS = frozenset({
    'vice_autostart',
    'vice_backtrace',
    'vice_checkpoint_add',
    'vice_checkpoint_delete',
    'vice_checkpoint_group_add',
    'vice_checkpoint_group_create',
    'vice_checkpoint_group_list',
    'vice_checkpoint_group_toggle',
    'vice_checkpoint_list',
    'vice_checkpoint_set_condition',
    'vice_checkpoint_set_ignore_count',
    'vice_checkpoint_toggle',
    'vice_cia_get_state',
    'vice_cia_set_state',
    'vice_cycles_stopwatch',
    'vice_disassemble',
    'vice_disk_attach',
    'vice_disk_detach',
    'vice_disk_list',
    'vice_disk_read_sector',
    'vice_display_get_dimensions',
    'vice_display_screenshot',
    'vice_execution_pause',
    'vice_execution_run',
    'vice_execution_step',
    'vice_joystick_set',
    'vice_joystick_tap',
    'vice_keyboard_chord',
    'vice_keyboard_key_press',
    'vice_keyboard_key_release',
    'vice_keyboard_matrix',
    'vice_keyboard_petscii',
    'vice_keyboard_restore',
    'vice_keyboard_type',
    'vice_machine_config_get',
    'vice_machine_config_set',
    'vice_machine_reset',
    'vice_memory_banks',
    'vice_memory_compare',
    'vice_memory_fill',
    'vice_memory_read',
    'vice_memory_search',
    'vice_memory_write',
    'vice_ping',
    'vice_registers_get',
    'vice_registers_set',
    'vice_run_until',
    'vice_sid_get_state',
    'vice_sid_set_state',
    'vice_snapshot_list',
    'vice_snapshot_load',
    'vice_snapshot_save',
    'vice_sprite_get',
    'vice_sprite_inspect',
    'vice_sprite_set',
    'vice_symbols_load',
    'vice_symbols_lookup',
    'vice_vicii_get_state',
    'vice_vicii_set_state',
    'vice_watch_add',
})
READ_ONLY_TOOLS = frozenset({
    "vice_ping", "vice_disassemble", "vice_backtrace", "vice_checkpoint_list",
    "vice_checkpoint_group_list", "vice_memory_read", "vice_memory_search",
    "vice_memory_compare", "vice_memory_banks", "vice_registers_get",
    "vice_machine_config_get", "vice_display_get_dimensions", "vice_display_screenshot",
    "vice_vicii_get_state", "vice_sid_get_state", "vice_cia_get_state",
    "vice_sprite_get", "vice_sprite_inspect", "vice_disk_list", "vice_disk_read_sector",
    "vice_snapshot_list", "vice_symbols_lookup",
    # Agent-side operations only read the emulator (snapshot files are local).
    "vice_memory_snapshot", "vice_memory_diff", "vice_memory_monotonic_scan",
})


def vice_method_requires_approval(method: str, args: dict | None = None) -> bool:
    native = _vice_normalize_method(method).replace(".", "_")
    if native == "vice_cycles_stopwatch":
        return (args or {}).get("action") != "read"
    # Unknown methods fail closed: newly introduced mutations cannot bypass review.
    return native not in READ_ONLY_TOOLS


def validate_literal_options(args: dict) -> None:
    """Validate supplied options without rejecting deferred address resolution."""
    method = _vice_normalize_method(str(args.get("method") or ""))
    if method == "vice.memory.read" and "encoding" in args:
        if args["encoding"] not in (None, "array", "hex"):
            raise ViceArgsError("vice.memory.read encoding must be 'array' or 'hex'")
    if method == "vice.memory.read":
        for key in ("size", "length"):
            if key in args and args[key] is not None:
                integer(args[key], key, 1, 65535)
    if method == "vice.disassemble" and args.get("count") is not None:
        integer(args["count"], "count", 1, 100)
    if method == "vice.checkpoint.add":
        for key in ("exec", "execute", "on_exec", "load", "read", "on_read",
                    "store", "write", "on_write", "stop", "stop_when_hit", "stop_on_hit"):
            if key in args and args[key] is not None:
                boolean(args[key], key)

VICE_ADDRESS_ALIASES = ("address", "addr", "start", "pc", "at", "from", "loc", "location", "entry", "target", "where")
_VICE_ADDRESS_ALIASES = VICE_ADDRESS_ALIASES


def _coerce_vice_address(raw: Any) -> str | None:
    try:
        return f"${address(raw):04X}"
    except ToolArgsError:
        return None


def _extract_vice_address(args: dict) -> str | None:
    # Never replace an explicitly invalid primary address with another alias.
    for key in VICE_ADDRESS_ALIASES:
        if key in args:
            return _coerce_vice_address(args[key])
    return None

def _vice_normalize_method(raw_method: str) -> str:
    """Map common aliases to real vice-mcp tool names."""
    m = (raw_method or "").strip().lower()
    if m.startswith("vice_"):
        m = m.replace("_", ".")
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
        "execute": "vice.execution.advance",
        "vice.execute": "vice.execution.advance",
        "advance": "vice.execution.advance",
        "vice.advance": "vice.execution.advance",
        "execution.advance": "vice.execution.advance",
        "pause": "vice.execution.pause",
        "execution.pause": "vice.execution.pause",
        "run": "vice.execution.run",
        "execution.run": "vice.execution.run",
        "step": "vice.execution.step",
        "execution.step": "vice.execution.step",
        "execution.reset": "vice.machine.reset",
        "vice.execution.reset": "vice.machine.reset",
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
        "resources.set": "vice.machine.config.set",
        "vice.resources.set": "vice.machine.config.set",
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
    aliases["vice.run.until"] = "vice.run_until"
    if m in aliases:
        return aliases[m]
    if m.startswith("vice."):
        return m
    # Last resort: treat as vice.<name>.
    return f"vice.{m}" if m else "vice.ping"

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

def _vice_normalize_args(method: str, args: dict[str, Any]) -> dict[str, Any]:
    """Translate generic args into the current vice-mcp schema.

    Raises ``ViceArgsError`` when a required arg is missing — previously
    these silently defaulted to ``$0801`` / ``$0000``, which produced
    plausible-looking but completely wrong tool results.
    """
    a = dict(args or {})
    if method.replace(".", "_") not in NATIVE_TOOLS:
        raise ViceArgsError(f"unsupported VICE method {method!r}")

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
            length = integer(a["length"], "length", 1, 65536) if "length" in a else 0
            count = max(1, min(100, (length // 2) if length else 16))
        return {
            "address": address,
            "count": integer(count, "count", 1, 100),
            "show_symbols": boolean(a.get("show_symbols", True), "show_symbols"),
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
            "size": integer(size, "size", 1, 65535),
            # The agent's deterministic formatter needs explicit byte
            # values, so per-byte "array" stays the default; bulk readers
            # ask for the compact "hex" blob instead.
            "encoding": str(a.get("encoding") or "array"),
        }
        if out["encoding"] not in {"array", "hex"}:
            raise ViceArgsError("vice.memory.read encoding must be 'array' or 'hex'")
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

    if method == "vice.checkpoint.add":
        # vice-mcp names the address `start` (not `address`) and the
        # stop flag `stop` (not `stop_when_hit`); sending our own spelling
        # made the server reject the call with "start address required".
        address = _extract_vice_address(a)
        if address is None:
            raise ViceArgsError(
                "vice.checkpoint.add requires a `start` address "
                "(e.g. \"$D000\")."
            )
        out_cp: dict[str, Any] = {"start": address}
        if a.get("end") is not None:
            end = _coerce_vice_address(a.get("end"))
            if end is None:
                raise ViceArgsError(
                    f"vice.checkpoint.add got an unparseable `end`: "
                    f"{a.get('end')!r}"
                )
            out_cp["end"] = end
        for canonical, aliases in (
            ("exec", ("exec", "execute", "on_exec")),
            ("load", ("load", "read", "on_read")),
            ("store", ("store", "write", "on_write")),
            ("stop", ("stop", "stop_when_hit", "stop_on_hit")),
        ):
            for alias in aliases:
                if a.get(alias) is not None:
                    out_cp[canonical] = boolean(a[alias], canonical)
                    break
        return out_cp

    if method == "vice.checkpoint.delete":
        # The server deletes by number only — there is no address form.
        num = None
        for alias in ("checkpoint_num", "checkpoint_id", "id", "number", "num"):
            if a.get(alias) is not None:
                num = a[alias]
                break
        try:
            num_int = integer(num, "checkpoint_num", 0, 0xFFFFFFFF)
        except (TypeError, ValueError):
            raise ViceArgsError(
                "vice.checkpoint.delete requires the numeric "
                "`checkpoint_num` returned by vice.checkpoint.add."
            ) from None
        return {"checkpoint_num": num_int}

    if method == "vice.execution.run":
        # Unsupported bounds must never be silently dropped on a free run.
        if a:
            raise ViceArgsError("vice.execution.run accepts no bounds; use vice.execution.advance for frames")
        return {}

    if method == "vice.display.screenshot":
        # Always return base64 so the agent sees the image without needing
        # a filesystem path. Accept an explicit path if the caller provided one.
        result: dict[str, Any] = {"return_base64": True, "format": "PNG"}
        if a.get("path"):
            result["path"] = str(a["path"])
        return result

    if method in {"vice.memory.search", "vice.memory.fill"}:
        start = address_arg(a, "start")
        end = address_arg(a, "end")
        if end < start:
            raise ViceArgsError("end must be at or after start")
        pattern = a.get("pattern")
        if not isinstance(pattern, list) or not pattern:
            raise ViceArgsError("requires a non-empty byte pattern")
        result = {"start": f"${start:04X}", "end": f"${end:04X}",
                  "pattern": [integer(v, "pattern byte", 0, 255) for v in pattern]}
        if method == "vice.memory.search":
            if "mask" in a:
                if not isinstance(a["mask"], list) or len(a["mask"]) != len(pattern):
                    raise ViceArgsError("mask must contain one byte per pattern byte")
                result["mask"] = [integer(v, "mask byte", 0, 255) for v in a["mask"]]
            if "max_results" in a:
                result["max_results"] = integer(a["max_results"], "max_results", 1, 10000)
        return result
    if method == "vice.execution.step":
        return {"count": integer(a.get("count", 1), "count", 1, 65535),
                "stepOver": boolean(a.get("stepOver", a.get("step_over", False)), "stepOver")}
    if method == "vice.machine.config.set":
        resources = a.get("resources")
        if isinstance(resources, dict):
            resources = json.dumps(resources)
        if not isinstance(resources, str):
            raise ViceArgsError("resources must be a JSON object")
        try:
            parsed = json.loads(resources)
        except ValueError:
            raise ViceArgsError("resources must be a JSON object") from None
        if not isinstance(parsed, dict):
            raise ViceArgsError("resources must be a JSON object")
        return {"resources": resources}
    if method == "vice.run_until":
        # This server explicitly documents cycles as not implemented.
        if "cycles" in a or "frames" in a:
            raise ViceArgsError("run_until does not implement cycle bounds; use vice.execution.advance")
        return {"address": f"${address_arg(a, 'address'):04X}"}
    if method == "vice.cycles.stopwatch" and a.get("action") not in {"read", "reset", "reset_and_read"}:
        raise ViceArgsError("stopwatch action must be read, reset, or reset_and_read")
    return a
