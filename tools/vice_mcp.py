"""Synchronous wrapper around vice-mcp's streamable-HTTP MCP transport.

The agent runs the LangGraph nodes synchronously; the official `mcp`
client is async. We bridge with a dedicated background thread + asyncio
loop, keeping a single persistent `ClientSession` open for the lifetime
of the process so subsequent tool calls don't pay the
``initialize`` round-trip again.

`VICE_MCP_URL` may be given with or without the ``/mcp`` suffix; we'll
append it if missing (matches Cursor's `mcp.json` convention).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any, Optional


_DEFAULT_PATH = "/mcp"


def resolve_url(raw: Optional[str] = None) -> str:
    raw = (raw if raw is not None else os.getenv("VICE_MCP_URL") or "").strip()
    if not raw:
        raise RuntimeError("VICE_MCP_URL not set")
    raw = raw.rstrip("/")
    last = raw.rsplit("/", 1)[-1]
    if last != "mcp":
        raw = raw + _DEFAULT_PATH
    return raw


# --------------------------------------------------------------------------- #
# Persistent background session
# --------------------------------------------------------------------------- #

class _ViceMcpClient:
    """Singleton: one event loop in a thread, one ClientSession kept open."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._url: str | None = None

    # ---- start ----------------------------------------------------------- #

    def _ensure_started(self, url: str, timeout: float = 10.0) -> None:
        if self._thread is not None and self._url == url:
            return
        with self._lock:
            if self._thread is not None and self._url == url:
                return
            if self._thread is not None and self._url != url:
                self._shutdown_locked()

            self._ready.clear()
            self._start_error = None
            self._url = url
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_loop, args=(url,),
                name="vice-mcp-client", daemon=True,
            )
            self._thread.start()
            if not self._ready.wait(timeout=timeout):
                raise RuntimeError(
                    f"vice-mcp session failed to start within {timeout}s "
                    f"(url={url})"
                )
            if self._start_error is not None:
                err = self._start_error
                self._thread = None
                self._loop = None
                raise RuntimeError(
                    f"vice-mcp session failed to initialise: "
                    f"{type(err).__name__}: {err}"
                )

    def _run_loop(self, url: str) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main(url))
        except BaseException as e:  # noqa: BLE001
            self._start_error = e
            self._ready.set()
        finally:
            try:
                self._loop.close()
            except Exception:  # noqa: BLE001
                pass

    async def _main(self, url: str) -> None:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ModuleNotFoundError as e:
            import sys as _sys
            self._start_error = ModuleNotFoundError(
                f"`mcp` not installed in this interpreter ({_sys.executable}). "
                "Install with: `pip install \"mcp>=1.0.0\"` "
                "(or re-run `pip install -e .` after pulling pyproject changes). "
                f"Original: {e}"
            )
            self._ready.set()
            return

        try:
            async with streamablehttp_client(url) as streams:
                read, write, *_rest = streams
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._stop_event = asyncio.Event()
                    self._ready.set()
                    await self._stop_event.wait()
        except BaseException as e:  # noqa: BLE001
            self._start_error = e
            self._ready.set()
        finally:
            self._session = None

    def _shutdown_locked(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._loop = None
        self._session = None
        self._stop_event = None
        self._url = None

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown_locked()

    # ---- calls ----------------------------------------------------------- #

    def call_tool(
        self,
        tool: str,
        arguments: dict[str, Any],
        timeout: float = 20.0,
        url: Optional[str] = None,
    ) -> dict[str, Any]:
        url = resolve_url(url)
        self._ensure_started(url)
        assert self._loop is not None and self._session is not None

        coro = self._session.call_tool(tool, arguments=arguments)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        result = fut.result(timeout=timeout)
        return _serialize_result(tool, arguments, url, result)

    def list_tools(
        self, timeout: float = 10.0, url: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        url = resolve_url(url)
        self._ensure_started(url)
        assert self._loop is not None and self._session is not None
        coro = self._session.list_tools()
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        res = fut.result(timeout=timeout)
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": getattr(t, "inputSchema", None),
            }
            for t in getattr(res, "tools", [])
        ]


_CLIENT = _ViceMcpClient()


# --------------------------------------------------------------------------- #
# Result normalization
# --------------------------------------------------------------------------- #

def _serialize_result(
    tool: str, args: dict[str, Any], url: str, result: Any,
) -> dict[str, Any]:
    is_error = bool(getattr(result, "isError", False))
    structured = getattr(result, "structuredContent", None)

    contents: list[Any] = []
    for c in getattr(result, "content", None) or []:
        text = getattr(c, "text", None)
        if text is not None:
            contents.append(text)
            continue
        data = getattr(c, "data", None)
        if data is not None:
            contents.append(data)
            continue
        try:
            contents.append(c.model_dump())  # pydantic v2
        except Exception:  # noqa: BLE001
            contents.append(str(c))

    payload: Any
    if structured is not None:
        payload = structured
    elif len(contents) == 1:
        only = contents[0]
        if isinstance(only, str):
            try:
                payload = json.loads(only)
            except Exception:  # noqa: BLE001
                payload = only
        else:
            payload = only
    else:
        payload = contents

    return {
        "ok": not is_error,
        "tool": tool,
        "url": url,
        "args": args,
        "data": payload,
        "is_error": is_error,
        "raw_content": contents,
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def call_tool(
    tool: str, arguments: dict[str, Any] | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    return _CLIENT.call_tool(tool, arguments or {}, timeout=timeout)


def list_tools(timeout: float = 10.0) -> list[dict[str, Any]]:
    return _CLIENT.list_tools(timeout=timeout)


def shutdown() -> None:
    _CLIENT.shutdown()
