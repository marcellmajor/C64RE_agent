"""Installation-safe configuration and writable workspace paths.

The helper functions read environment overrides on every call. Application
modules intentionally snapshot their returned paths into module constants, so
``C64RE_*_DIR`` overrides must be set before importing/starting the CLI, UI,
graph, or semantic configuration modules. This gives one process a stable
workspace while still allowing test/evaluation runners to patch the explicit
module constant (notably ``graph.nodes.SESSIONS_DIR``) for isolated cases.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_DIR.parent


def workspace_root() -> Path:
    """Return the writable project/data root for the current invocation."""
    override = os.getenv("C64RE_WORKSPACE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path.cwd().resolve()


def sessions_dir() -> Path:
    """Return the writable session root, honoring an explicit override."""
    override = os.getenv("C64RE_SESSIONS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return workspace_root() / "sessions"


def config_dir() -> Path:
    """Locate user-overridden, packaged, or source-tree configuration."""
    override = os.getenv("C64RE_CONFIG_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    packaged = PACKAGE_DIR / "config"
    if packaged.is_dir():
        return packaged
    source = SOURCE_ROOT / "config"
    if source.is_dir():
        return source
    return workspace_root() / "config"
