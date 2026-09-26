"""Bounded emulator advancement using measured cycles, without free-running."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


# Full raster frame durations for the supported C64 video standards.
_CYCLES_PER_FRAME = {"PAL": 312 * 63, "NTSC": 263 * 65}
_MAX_STEP_INSTRUCTIONS = 8192
_ADVANCE_TIMEOUT_S = 30.0


def advance_frames(
    call: Callable[[str, dict], dict], frames: Any,
) -> dict[str, Any]:
    """Step to a frame-equivalent cycle budget and leave the machine paused.

    Stepping can cross the target by part of an instruction/interrupt. Report
    actual measured cycles rather than claiming exact raster-boundary stops.
    Never reset the user's stopwatch, resume freely, or retry partial progress.
    """
    if (
        isinstance(frames, bool) or not isinstance(frames, (int, str))
        or not str(frames).strip().isdigit() or not 1 <= int(frames) <= 300
    ):
        return {
            "ok": False, "retryable": False, "rejection": "invalid_frames",
            "message": "vice.execution.advance requires integer frames in 1..300.",
        }
    frames = int(frames)
    result: dict[str, Any] = {"requested_frames": frames, "cycles_advanced": 0}
    paused = False
    failure = None

    def read_cycles() -> int:
        data = call("vice.cycles.stopwatch", {"action": "read"}).get("data")
        value = data.get("cycles") if isinstance(data, dict) else None
        if (isinstance(value, bool) or not isinstance(value, (int, str))
                or not str(value).isdigit()):
            raise RuntimeError("VICE did not return a usable cycle counter")
        return int(value)

    try:
        config = call("vice.machine.config.get", {}).get("data")
        standard = str(config.get("video_standard", "")).upper() if isinstance(config, dict) else ""
        if standard not in _CYCLES_PER_FRAME:
            raise RuntimeError(f"unsupported or unknown video standard {standard!r}")
        per_frame = _CYCLES_PER_FRAME[standard]
        target = frames * per_frame
        result.update(video_standard=standard, target_cycles=target)
        call("vice.execution.pause", {})
        paused = True
        initial = previous = read_cycles()
        deadline = time.monotonic() + _ADVANCE_TIMEOUT_S
        while result["cycles_advanced"] < target:
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out before the requested cycle budget was reached")
            remaining = target - result["cycles_advanced"]
            count = min(_MAX_STEP_INSTRUCTIONS, max(1, remaining // 16))
            call("vice.execution.step", {"count": count, "stepOver": False})
            current = read_cycles()
            if current <= previous:
                raise RuntimeError("cycle counter stalled, reset, or wrapped during advancement")
            previous = current
            result["cycles_advanced"] = current - initial
        result["frames_advanced"] = result["cycles_advanced"] / per_frame
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if paused:
            try:
                call("vice.execution.pause", {})
            except Exception as exc:  # noqa: BLE001
                failure = f"{failure + '; ' if failure else ''}could not confirm pause: {exc}"
                paused = False

    result["paused"] = paused
    if failure:
        return {
            **result, "ok": False, "retryable": False,
            "rejection": "advance_failed",
            "message": (
                f"vice.execution.advance failed: {failure}. "
                f"Confirmed progress: {result['cycles_advanced']} cycles. "
                "Do not repeat automatically: partial progress may already have occurred."
            ),
        }
    return {
        **result, "ok": True,
        "message": (
            f"Advanced {result['cycles_advanced']} measured cycles "
            f"({result['frames_advanced']:.4f} {result['video_standard']} frame equivalents; "
            f"requested {frames}). Paused for the next snapshot. "
            "The stop is at an instruction boundary, not an exact raster boundary. "
            "Verify game-state changes with snapshot/diff; advancement alone does not prove them."
        ),
    }
