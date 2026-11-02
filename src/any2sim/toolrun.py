"""Run a tool in its worker environment and write a receipt.

A tool is a module any2sim.tools.<name> with
    ENV: str            worker environment (configs/any2sim.yaml: environments)
    GPU: bool           whether it needs a GPU (the least-loaded one is chosen)
    def run(project: Path, **params) -> dict     result; result["outputs"] lists produced paths
The runner (base env) starts `<env python> -m any2sim.toolexec`, so model envs never import the runner, and records
request, input/output hashes, code version, environment, duration, log and status in receipts/.
"""
from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import config as C
from .common import digest, read_json, write_json
from .project import Project, now


def tool_module(name: str):
    return importlib.import_module(f"any2sim.tools.{name}")


def _tool_meta(name: str) -> tuple[str, bool]:
    """ENV and GPU of a tool, read from its source without importing it (its imports may be env-specific)."""
    import ast
    src = Path(C.ROOT, "src", "any2sim", "tools", f"{name}.py")
    if not src.exists():
        raise KeyError(f"unknown tool {name!r}")
    env, gpu = "base", False
    for node in ast.parse(src.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "ENV":
                env = ast.literal_eval(node.value)
            elif node.targets[0].id == "GPU":
                gpu = ast.literal_eval(node.value)
    return env, gpu


def _tree_rss(pid: int) -> float:
    """Resident memory (bytes) of a process and all its descendants (Linux /proc)."""
    total, todo, seen = 0.0, [pid], set()
    while todo:
        q = todo.pop()
        if q in seen:
            continue
        seen.add(q)
        try:
            with open(f"/proc/{q}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += float(line.split()[1]) * 1024
            with open(f"/proc/{q}/task/{q}/children") as f:
                todo += [int(c) for c in f.read().split()]
        except (FileNotFoundError, ProcessLookupError, ValueError):
            continue
    return total


def pick_gpu() -> str | None:
    allowed = [str(g) for g in C.config().get("gpus", [])]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    rows = [l.split(",") for l in out.strip().splitlines() if l.strip()]
    rows = [(i.strip(), float(m)) for i, m in rows if not allowed or i.strip() in allowed]
    return min(rows, key=lambda r: r[1])[0] if rows else None


def _git_rev() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(C.ROOT), "status", "--porcelain", "src"], capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return r.stdout.strip() + ("+dirty" if dirty else "") if r.returncode == 0 else None
    except OSError:
        return None


def run_tool(project: str | Path, name: str, params: dict | None = None, gpu: str | None = None,
             inputs: list[str] | None = None, timeout: float | None = None) -> dict:
    """Execute tool `name` on `project`; returns the receipt (result under receipt['result'])."""
    proj = Project(project)
    params = params or {}
    env_name, needs_gpu = _tool_meta(name)
    py = C.env_python(env_name)
    env = C.env_vars(env_name)
    if needs_gpu:
        gpu = gpu or pick_gpu()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    rid = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}-{uuid.uuid4().hex[:6]}"
    log = proj.path("receipts", "logs", f"{rid}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    res_path = proj.path("receipts", "logs", f"{rid}.result.json")
    in_hashes = {p: digest(proj.path(p)) for p in (inputs or []) if proj.path(p).exists()}
    t0 = time.time()
    cmd = [str(py), "-m", "any2sim.toolexec", name, str(proj.root), json.dumps(params), str(res_path)]
    max_rss = float(C.config().get("max_rss_gb", 48)) * 1e9
    peak = 0.0
    killed = None
    with open(log, "w") as lf:
        # shared machine: a tool that exceeds the RSS cap (or the timeout) is killed, never left to OOM the host
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(proj.root),
                             start_new_session=True)
        while p.poll() is None:
            rss = _tree_rss(p.pid)
            peak = max(peak, rss)
            if rss > max_rss:
                killed = f"RSS {rss / 1e9:.1f} GB > cap {max_rss / 1e9:.0f} GB"
            elif timeout and time.time() - t0 > timeout:
                killed = f"timeout {timeout} s"
            if killed:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                lf.write(f"\n[toolrun] killed: {killed}\n")
                break
            time.sleep(1.0)
        rc = p.returncode
    result = read_json(res_path) if res_path.exists() else None
    status = "ok" if rc == 0 and result is not None and not (isinstance(result, dict) and "error" in result) else "failed"
    outputs = (result or {}).get("outputs", []) if isinstance(result, dict) else []
    receipt = {
        "receipt_id": rid, "tool": name, "status": status, "returncode": rc, "started": now(),
        "duration_s": round(time.time() - t0, 2), "params": params, "env": env_name,
        "interpreter": str(py), "gpu": env.get("CUDA_VISIBLE_DEVICES"), "code_version": _git_rev(),
        "peak_rss_gb": round(peak / 1e9, 2), "killed": killed,
        "input_hashes": in_hashes,
        "output_hashes": {o: digest(proj.path(o)) for o in outputs if proj.path(o).exists()},
        "log": str(log.relative_to(proj.root)), "result": result,
    }
    write_json(receipt, proj.path("receipts", f"{rid}.json"))
    if status != "ok":
        tail = log.read_text()[-3000:] if log.exists() else ""
        receipt["log_tail"] = tail
    return receipt


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("project"); ap.add_argument("tool"); ap.add_argument("params", nargs="?", default="{}")
    ap.add_argument("--gpu"); ap.add_argument("--inputs", nargs="*")
    a = ap.parse_args(argv)
    r = run_tool(a.project, a.tool, json.loads(a.params), a.gpu, a.inputs)
    print(json.dumps({k: v for k, v in r.items() if k != "output_hashes"}, indent=1, default=str)[:20000])
    return 0 if r["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
