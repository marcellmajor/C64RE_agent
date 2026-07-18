"""Wheel manifest and installation-safe path regressions (tracker 7.1)."""

from __future__ import annotations

import tomllib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from c64re_agent.paths import config_dir, sessions_dir, workspace_root


ROOT = Path(__file__).resolve().parent.parent


def test_wheel_manifest_contains_every_runtime_package_and_config():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    wheel = config["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert set(wheel["packages"]) >= {
        "c64re_agent", "graph", "memory", "tools", "code_kb", "evals",
    }
    assert wheel["force-include"] == {
        "config": "c64re_agent/config",
        "main.py": "main.py",
        "app.py": "app.py",
        "langgraph.json": "langgraph.json",
    }


def test_runtime_paths_support_writable_and_config_overrides(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    sessions = tmp_path / "durable-sessions"
    config = tmp_path / "custom-config"
    monkeypatch.setenv("C64RE_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("C64RE_SESSIONS_DIR", str(sessions))
    monkeypatch.setenv("C64RE_CONFIG_DIR", str(config))

    assert workspace_root() == workspace
    assert sessions_dir() == sessions
    assert config_dir() == config


def test_module_path_snapshots_honor_overrides_set_before_import(tmp_path):
    workspace = (tmp_path / "workspace").resolve()
    sessions = (tmp_path / "sessions").resolve()
    config = (tmp_path / "config").resolve()
    env = dict(os.environ)
    env.update({
        "C64RE_WORKSPACE_DIR": str(workspace),
        "C64RE_SESSIONS_DIR": str(sessions),
        "C64RE_CONFIG_DIR": str(config),
    })
    script = """
import json
import app
import main
import graph.llm as llm
import graph.nodes as nodes
import memory.semantic_config as semantic
print(json.dumps({
    "app": str(app._SESSIONS_DIR),
    "main": str(main.SESSIONS_DIR),
    "llm": str(llm.CONFIG_DIR),
    "nodes": str(nodes.SESSIONS_DIR),
    "semantic": str(semantic.CONFIG_PATH),
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload == {
        "app": str(sessions),
        "main": str(sessions),
        "llm": str(config),
        "nodes": str(sessions),
        "semantic": str(config / "kb_semantic.json"),
    }


def test_eval_runner_temporarily_patches_and_restores_nodes_session_root(
    tmp_path, monkeypatch,
):
    import graph.nodes as nodes
    from evals.runner import execute_case, load_cases
    from tools import agent_runner

    original = nodes.SESSIONS_DIR
    isolated = tmp_path / "isolated-case"
    observed = []

    def fake_run_question(**_kwargs):
        observed.append(nodes.SESSIONS_DIR)
        raise RuntimeError("stop after observing patched session root")

    monkeypatch.setattr(agent_runner, "run_question", fake_run_question)

    with pytest.raises(RuntimeError, match="patched session root"):
        execute_case(load_cases()[0], session_root=isolated)

    assert observed == [isolated]
    assert nodes.SESSIONS_DIR == original


def test_source_checkout_resolves_checked_in_config():
    assert (config_dir() / "llm.json").is_file()
    assert (config_dir() / "kb_semantic.json").is_file()


def test_checked_in_llm_config_contains_only_live_graph_roles():
    config = json.loads((ROOT / "config" / "llm.json").read_text())

    assert set(config["agents"]) == {
        "planner",
        "executor",
        "synthesizer",
        "analyst",
        "critic",
        "curator",
        "vision",
    }
