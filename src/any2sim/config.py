"""Repository paths and machine configuration (configs/any2sim.yaml, overridden by configs/any2sim.local.yaml)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _expand(x):
    if isinstance(x, str):
        return x.replace("${ROOT}", str(ROOT))
    if isinstance(x, dict):
        return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v) for v in x]
    return x


@lru_cache(maxsize=1)
def config() -> dict:
    path = Path(os.environ.get("ANY2SIM_CONFIG", ROOT / "configs" / "any2sim.yaml"))
    cfg = yaml.safe_load(path.read_text()) or {}
    local = path.with_name("any2sim.local.yaml")
    if local.exists():
        cfg = _merge(cfg, yaml.safe_load(local.read_text()) or {})
    return _expand(cfg)


def env_python(name: str) -> Path:
    """Interpreter of a worker environment (envs/<name>/bin/python)."""
    p = Path(config()["environments"][name])
    if not p.exists():
        raise FileNotFoundError(f"environment {name!r} not installed: {p}")
    return p


def env_vars(name: str) -> dict:
    """Process environment for a tool running in worker env `name`: global env, then per-env additions
    (keys ending in PATH are prepended to the inherited value)."""
    cfg = config()
    env = dict(os.environ)
    for k, v in {**cfg.get("env", {}), **cfg.get("environment_env", {}).get(name, {})}.items():
        if v is None:
            continue
        v = str(v)
        env[k] = f"{v}:{env[k]}" if k.endswith("PATH") and env.get(k) else v
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(ROOT / "src")
    env.setdefault("LC_ALL", "C.UTF-8")
    env.setdefault("LANG", "C.UTF-8")
    return env


def model_path(key: str) -> Path:
    return Path(config()["models"][key])
