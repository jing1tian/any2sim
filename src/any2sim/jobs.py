"""Detached tool jobs: a tool call survives the caller (MCP request timeouts, agent restarts) and can be polled.

projects/<id>/jobs/<job_id>.json  {job_id, tool, params, status: running|ok|failed|cancelled, pid, receipt, ...}
The job process runs any2sim.toolrun.run_tool (receipt, RSS cap, GPU choice) and writes the outcome.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import config as C
from .project import Project, now


def _job_path(project: Project, job_id: str) -> Path:
    return project.path("jobs", f"{job_id}.json")


def submit(project_root: str | Path, tool: str, params: dict, timeout: float | None = None) -> str:
    proj = Project(project_root)
    proj.path("jobs").mkdir(exist_ok=True)
    job_id = f"job-{time.strftime('%Y%m%d-%H%M%S')}-{tool}-{uuid.uuid4().hex[:5]}"
    jp = _job_path(proj, job_id)
    jp.write_text(json.dumps({"job_id": job_id, "tool": tool, "params": params, "status": "running",
                              "submitted": now(), "timeout": timeout}, indent=1))
    cmd = [sys.executable, "-m", "any2sim.jobs", str(proj.root), job_id]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{C.ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=open(proj.path("jobs", f"{job_id}.err"), "w"),
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(C.ROOT))
    rec = json.loads(jp.read_text())
    rec["pid"] = p.pid
    jp.write_text(json.dumps(rec, indent=1))
    return job_id


def status(project_root: str | Path, job_id: str) -> dict:
    proj = Project(project_root)
    rec = json.loads(_job_path(proj, job_id).read_text())
    if rec["status"] == "running" and rec.get("pid"):
        try:
            os.kill(rec["pid"], 0)
        except ProcessLookupError:
            rec = json.loads(_job_path(proj, job_id).read_text())     # finished between the two reads?
            if rec["status"] == "running":
                rec["status"] = "failed"
                rec["error"] = "job process died without a result (see jobs/<id>.err)"
    return rec


def cancel(project_root: str | Path, job_id: str) -> dict:
    proj = Project(project_root)
    rec = json.loads(_job_path(proj, job_id).read_text())
    if rec["status"] == "running" and rec.get("pid"):
        try:
            os.killpg(rec["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        rec["status"] = "cancelled"
        rec["finished"] = now()
        _job_path(proj, job_id).write_text(json.dumps(rec, indent=1))
    return rec


def wait(project_root: str | Path, job_id: str, wait_s: float) -> dict:
    t0 = time.time()
    while True:
        rec = status(project_root, job_id)
        if rec["status"] != "running" or time.time() - t0 > wait_s:
            return rec
        time.sleep(2.0)


def _main():
    root, job_id = sys.argv[1], sys.argv[2]
    proj = Project(root)
    jp = _job_path(proj, job_id)
    rec = json.loads(jp.read_text())
    from .toolrun import run_tool
    try:
        r = run_tool(proj.root, rec["tool"], rec["params"], timeout=rec.get("timeout"))
        rec.update({"status": r["status"], "receipt_id": r["receipt_id"], "duration_s": r["duration_s"],
                    "result": r["result"], "killed": r.get("killed"), "log": r["log"],
                    "log_tail": r.get("log_tail", "")[-2000:]})
    except Exception as e:  # recorded for the caller
        rec.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
    rec["finished"] = now()
    jp.write_text(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    _main()
