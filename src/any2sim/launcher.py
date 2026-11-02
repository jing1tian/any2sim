"""Headless agent sessions (Codex or Claude Code) on a project, with the full trace kept and token usage monitored live.

The launcher starts the runtime and records; it never decides stages, grades artifacts or edits the scene.

projects/<id>/sessions/                (<agent> = codex | claude)
  <agent>_<stamp>.command.json    command line, model, effort, budgets
  <agent>_<stamp>.events.jsonl    event stream: `codex exec --json`, or `claude -p --output-format stream-json`
                                  (each line stamped with its arrival time `_t`)
  codex_<stamp>.rollout.jsonl     Codex's own rollout of the main thread (reasoning summaries, tool I/O, token_count)
  claude_<stamp>.transcript.jsonl Claude Code's own transcript of the main thread
  <agent>_<stamp>.sub_*.jsonl     rollouts / transcripts of sub-agent threads (evaluators) started during the session
  <agent>_<stamp>.usage.jsonl     one row per model call: cumulative and last token usage, rate-limit use, thread
  <agent>_<stamp>.summary.json    totals per thread, tool-call counts, durations, exit status (updated live)

Token rows use one schema for both runtimes: input_tokens includes cached input (Claude: input + cache writes +
cache reads), cached_input_tokens are cache reads, total_tokens = input_tokens + output_tokens.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from . import config as C
from .project import Project, now

CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
CLAUDE_HOME = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
AGENTS = ("codex", "claude")
DEFAULT_MODEL = {"codex": "gpt-6-astra", "claude": "claude-opus-5-5"}
# Built-in Claude Code tools of the builder (sub-agents via Task, skills via Skill); web, cron, worktrees stay off.
CLAUDE_TOOLS = ["Bash", "Read", "Edit", "Write", "Task", "Skill"]

PROMPT = """You are the builder agent of the any2sim project at {project}. Read AGENTS.md, then {project}/request.yaml
and the skills it names (skills/*/SKILL.md). Work autonomously toward the request's goals with the tools (MCP server
`any2sim`: list_tools, describe_tool, run_tool, job_status, job_cancel) and your own code in the project folder.
Keep {project}/WORKING.md current (plan, decisions, evidence, open issues). Delegate every acceptance evaluation to the
stage-evaluator sub-agent. Stop when the request's acceptance criteria pass, or a budget is exhausted, and end with a
summary of what was produced, the scorecard, and what is unresolved."""


def codex_overrides(python: str) -> list[str]:
    q = json.dumps
    kv = {
        "mcp_servers.any2sim.command": q(python),
        "mcp_servers.any2sim.args": '["-m", "any2sim.mcp_server"]',
        "mcp_servers.any2sim.env": "{PYTHONPATH = %s}" % q(str(C.ROOT / "src")),
        "mcp_servers.any2sim.default_tools_approval_mode": q("approve"),
        "mcp_servers.any2sim.startup_timeout_sec": "60",
        "mcp_servers.any2sim.tool_timeout_sec": "3600",
        "agents.stage-evaluator.description": q("Independent evaluator of an any2sim stage candidate; give it the "
                                                "project path, the stage and the rubric."),
        "agents.stage-evaluator.config_file": q(str(C.ROOT / ".codex" / "agents" / "stage-evaluator.toml")),
    }
    out = []
    for k, v in kv.items():
        out += ["-c", f"{k}={v}"]
    return out


def claude_args(python: str) -> list[str]:
    """Headless Claude Code: the any2sim MCP server only, a fixed tool set, no permission prompts. The stage-evaluator
    sub-agent and the skills come from the repository (.claude/agents, .claude/skills)."""
    mcp = {"mcpServers": {"any2sim": {"type": "stdio", "command": python, "args": ["-m", "any2sim.mcp_server"],
                                      "env": {"PYTHONPATH": str(C.ROOT / "src")}}}}
    return ["--output-format", "stream-json", "--verbose", "--mcp-config", json.dumps(mcp), "--strict-mcp-config",
            "--tools", ",".join(CLAUDE_TOOLS), "--permission-mode", "dontAsk",
            "--allowedTools", ",".join(CLAUDE_TOOLS + ["mcp__any2sim"])]


def _find_rollouts(since: float) -> dict[str, dict]:
    """Rollout files touched since `since`: thread id -> {path, cwd, parent} (parent = spawning thread for
    sub-agents)."""
    found = {}
    for p in (CODEX_HOME / "sessions").rglob("rollout-*.jsonl"):
        try:
            if p.stat().st_mtime < since - 5:
                continue
            with open(p) as f:
                meta = json.loads(f.readline())
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("type") != "session_meta":
            continue
        pl = meta.get("payload") or {}
        src = pl.get("source")
        parent = None
        if isinstance(src, dict):
            sub = src.get("subagent") or {}
            parent = (sub.get("thread_spawn") or {}).get("parent_thread_id")
            if parent is None and sub:
                parent = "?"                                 # e.g. guardian reviewers
        found[pl.get("id")] = {"path": p, "cwd": pl.get("cwd"), "parent": parent}
    return found


def _family(rolls: dict, main: str | None) -> dict[str, str]:
    """Threads that belong to this session: the main thread and its (transitive) sub-agents -> role."""
    if main is None or main not in rolls:
        return {}
    fam, changed = {main: "main"}, True
    while changed:
        changed = False
        for tid, r in rolls.items():
            if tid not in fam and r["parent"] in fam:
                fam[tid] = "sub"
                changed = True
    return fam


def _usage_rows(path: Path, thread: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if '"token_count"' not in line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # Live rollout may end in a partially written token event; retry next poll.
            pl = d.get("payload") or {}
            info = pl.get("info") or {}
            if not info or not isinstance(info.get("total_token_usage"), dict):
                continue
            rl = ((pl.get("rate_limits") or {}).get("primary") or {})
            rows.append({"t": d.get("timestamp"), "thread": thread, "total": info.get("total_token_usage"),
                         "last": info.get("last_token_usage"), "context_window": info.get("model_context_window"),
                         "rate_limit_used_percent": rl.get("used_percent")})
    return rows


def _tool_counts(events: Path) -> dict:
    counts, mcp_tools = {}, {}
    if not events.exists():
        return {}
    with open(events) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "item.completed":
                continue
            it = d.get("item") or {}
            t = it.get("type")
            counts[t] = counts.get(t, 0) + 1
            if t == "mcp_tool_call":
                name = it.get("tool")
                if name == "run_tool":
                    name = "run_tool:" + str((it.get("arguments") or {}).get("tool"))
                mcp_tools[name] = mcp_tools.get(name, 0) + 1
    return {"items": counts, "mcp_tool_calls": mcp_tools}


def monitor(proj: Project, stamp: str, since: float, stop: threading.Event, period: float = 30.0,
            max_tokens: int | None = None, proc: subprocess.Popen | None = None):
    sdir = proj.path("sessions")
    main_tid = {"id": None}
    while True:
        rolls = _find_rollouts(since)
        ev = sdir / f"codex_{stamp}.events.jsonl"
        if main_tid["id"] is None and ev.exists():
            with open(ev) as f:
                for line in f:
                    if '"thread.started"' in line:
                        main_tid["id"] = json.loads(line).get("thread_id")
                        break
        rows, per_thread = [], {}
        for tid, role in _family(rolls, main_tid["id"]).items():
            p = rolls[tid]["path"]
            r = _usage_rows(p, f"{role}:{tid}")
            rows += r
            if r:
                per_thread[f"{role}:{tid}"] = r[-1]["total"]
            dst = sdir / (f"codex_{stamp}.rollout.jsonl" if role == "main" else f"codex_{stamp}.sub_{tid}.jsonl")
            try:
                shutil.copyfile(p, dst)
            except OSError:
                pass
        rows.sort(key=lambda r: r["t"] or "")
        with open(sdir / f"codex_{stamp}.usage.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        tot = {k: sum((v or {}).get(k, 0) for v in per_thread.values())
               for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens", "total_tokens")}
        summary = {"agent": "codex", "stamp": stamp, "updated": now(), "main_thread": main_tid["id"], "threads": per_thread,
                   "totals": tot, "uncached_input_tokens": tot["input_tokens"] - tot["cached_input_tokens"],
                   "model_calls": len(rows), "rate_limit_used_percent": rows[-1]["rate_limit_used_percent"] if rows else None,
                   "elapsed_h": round((time.time() - since) / 3600, 3), "tools": _tool_counts(ev)}
        summary_path = sdir / f"codex_{stamp}.summary.json"
        temporary = summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=1))
        temporary.replace(summary_path)
        if max_tokens and tot["total_tokens"] > max_tokens and proc is not None and proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            with open(ev, "a") as f:
                f.write(json.dumps({"any2sim": "token budget reached", "total_tokens": tot["total_tokens"]}) + "\n")
        if stop.is_set():
            return summary
        stop.wait(period)


def _utc(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def _pump(src, f, lock: threading.Lock):
    """Copy the runtime's stdout to the event log, stamping each JSON line with its arrival time."""
    for line in src:
        try:
            line = json.dumps({"_t": _utc(), **json.loads(line)}) + "\n"
        except (json.JSONDecodeError, TypeError):
            pass
        with lock:
            f.write(line)
            f.flush()


def _claude_transcripts(sid: str, since: float) -> dict[str, Path]:
    """Claude Code's own session files touched since `since`: thread -> path (main transcript, sub-agent ones)."""
    found = {}
    for p in (CLAUDE_HOME / "projects").glob(f"*/{sid}.jsonl"):
        found[f"main:{sid}"] = p
        for s in sorted(p.with_suffix("").glob("subagents/*.jsonl")):
            try:
                if s.stat().st_mtime >= since - 5:
                    found["sub:" + s.stem.removeprefix("agent-")] = s
            except OSError:
                pass
    return found


def _claude_calls(path: Path, thread: str, stream: bool = False) -> tuple[dict, dict, dict | None]:
    """Model calls (deduplicated by message id; one message spans several lines), tool_use blocks and the final
    result event of a transcript or a stream-json log."""
    calls, tools, result = {}, {}, None
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # Live file may end in a partially written line; retry next poll.
            if d.get("type") == "result":
                result = d
            m = d.get("message")
            if d.get("type") != "assistant" or not isinstance(m, dict) or not m.get("id") or m.get("model") == "<synthetic>":
                continue
            th = thread
            if stream and d.get("parent_tool_use_id"):
                th = "sub:" + d["parent_tool_use_id"]
            elif d.get("isSidechain"):
                th = "sub:" + str(d.get("agentId") or "sidechain")
            t = d.get("timestamp") or d.get("_t")
            if isinstance(m.get("usage"), dict):
                calls[(th, m["id"])] = {"t": t, "usage": m["usage"]}
            for b in m.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tools[b.get("id")] = {**b, "_t": t}
    return calls, tools, result


def _claude_rows(calls: dict, since: float) -> list[dict]:
    rows, cum, t0 = [], {}, _utc(since - 5)
    for (th, _), c in sorted(calls.items(), key=lambda kv: kv[1]["t"] or ""):
        if c["t"] and c["t"] < t0:
            continue                                         # earlier part of a resumed conversation
        u = c["usage"]
        cached = u.get("cache_read_input_tokens") or 0
        inp = (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) + cached
        out = u.get("output_tokens") or 0
        last = {"input_tokens": inp, "cached_input_tokens": cached, "output_tokens": out,
                "reasoning_output_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens") or 0,
                "total_tokens": inp + out}
        tot = cum.setdefault(th, dict.fromkeys(last, 0))
        for k, v in last.items():
            tot[k] += v
        rows.append({"t": c["t"], "thread": th, "total": dict(tot), "last": last, "context_window": None,
                     "rate_limit_used_percent": None})
    return rows


def _claude_tool_counts(tools: dict, since: float) -> dict:
    counts, mcp_tools, t0 = {}, {}, _utc(since - 5)
    for b in tools.values():
        if b["_t"] and b["_t"] < t0:
            continue                                         # earlier part of a resumed conversation
        name = b.get("name") or "?"
        counts[name] = counts.get(name, 0) + 1
        if name.startswith("mcp__any2sim__"):
            name = name.removeprefix("mcp__any2sim__")
            if name == "run_tool":
                name = "run_tool:" + str((b.get("input") or {}).get("tool"))
            mcp_tools[name] = mcp_tools.get(name, 0) + 1
    return {"items": counts, "mcp_tool_calls": mcp_tools}


def monitor_claude(proj: Project, stamp: str, sid: str, since: float, stop: threading.Event, note,
                   period: float = 30.0, max_tokens: int | None = None, proc: subprocess.Popen | None = None):
    """Usage from Claude Code's transcripts (main + sub-agents); the stream-json log is the fallback when they
    cannot be found."""
    sdir = proj.path("sessions")
    ev = sdir / f"claude_{stamp}.events.jsonl"
    while True:
        calls, tools = {}, {}
        for th, p in _claude_transcripts(sid, since).items():
            c, t, _ = _claude_calls(p, th)
            calls.update(c)
            tools.update(t)
            dst = sdir / (f"claude_{stamp}.transcript.jsonl" if th.startswith("main:") else f"claude_{stamp}.sub_{th[4:]}.jsonl")
            try:
                shutil.copyfile(p, dst)
            except OSError:
                pass
        s_calls, s_tools, result = _claude_calls(ev, f"main:{sid}", stream=True) if ev.exists() else ({}, {}, None)
        rows = _claude_rows(calls or s_calls, since)
        with open(sdir / f"claude_{stamp}.usage.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        per_thread = {}
        for r in rows:
            per_thread[r["thread"]] = r["total"]
        tot = {k: sum(v.get(k, 0) for v in per_thread.values())
               for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens", "total_tokens")}
        summary = {"agent": "claude", "stamp": stamp, "updated": now(), "main_thread": sid, "threads": per_thread,
                   "totals": tot, "uncached_input_tokens": tot["input_tokens"] - tot["cached_input_tokens"],
                   "model_calls": len(rows), "rate_limit_used_percent": None,
                   "elapsed_h": round((time.time() - since) / 3600, 3),
                   "tools": _claude_tool_counts({**s_tools, **tools}, since),
                   "usage_source": "transcripts" if calls else "stream"}
        if result:                                           # Claude's own figures; cost and modelUsage cover
            summary["result"] = {k: result.get(k) for k in ("subtype", "is_error", "num_turns", "terminal_reason",
                                                            "total_cost_usd", "duration_ms", "modelUsage")}
            summary["result"]["cost_scope"] = "conversation"    # the whole conversation, earlier parts when resumed
        summary_path = sdir / f"claude_{stamp}.summary.json"
        temporary = summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=1))
        temporary.replace(summary_path)
        if max_tokens and tot["total_tokens"] > max_tokens and proc is not None and proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            note({"any2sim": "token budget reached", "total_tokens": tot["total_tokens"]})
        if stop.is_set():
            return summary
        stop.wait(period)


RESUME_PROMPT = """The previous session on {project} was interrupted. Re-read {project}/WORKING.md and the latest
evaluation reports, check which jobs and tools finished (job_status, receipts/), then continue toward the request's
acceptance criteria within the remaining budgets."""


def session_files(proj: Project, suffix: str, agent: str | None = None) -> list[Path]:
    """A project's session files with this suffix (e.g. "summary.json"), oldest first across agents."""
    files = [p for a in ((agent,) if agent else AGENTS) for p in proj.path("sessions").glob(f"{a}_*.{suffix}")]
    return sorted(files, key=lambda p: p.name.split("_", 1)[1])


def last_session(proj: Project, agent: str = "codex") -> str | None:
    for p in reversed(session_files(proj, "summary.json", agent)):
        tid = json.loads(p.read_text()).get("main_thread")
        if tid:
            return tid
    return None


def launch(project: str | Path, model: str | None = None, effort: str = "medium", max_hours: float = 8.0,
           max_tokens: int | None = None, prompt: str | None = None, resume: str | None = None,
           agent: str = "codex") -> dict:
    """agent: "codex" | "claude". model: None = DEFAULT_MODEL[agent].
    resume: None (new session) | "last" (this project's last main thread of that agent) | a session/thread id."""
    if agent not in AGENTS:
        raise ValueError(f"agent must be one of {AGENTS}")
    model = model or DEFAULT_MODEL[agent]
    proj = Project(project)
    sdir = proj.path("sessions")
    sdir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{agent}_{stamp}"
    py = str(C.env_python("base"))
    rel = proj.root.relative_to(C.ROOT)
    sid = None
    if resume:
        sid = last_session(proj, agent) if resume == "last" else resume
        if not sid:
            raise RuntimeError("no previous session to resume")
    text = (prompt or (RESUME_PROMPT if resume else PROMPT)).format(project=rel)
    if agent == "claude":
        sid = sid or str(uuid.uuid4())                       # known up front: transcripts are found by it
        cmd = [shutil.which("claude") or "claude", "-p", "--resume" if resume else "--session-id", sid,
               "--model", model, "--effort", effort, *claude_args(py)]   # prompt goes in on stdin
    elif resume:
        exe = shutil.which("codex") or "codex"
        cmd = [exe, "exec", "resume", "--json", "--skip-git-repo-check", "-m", model,
               "-c", 'sandbox_mode="workspace-write"', "-c", f'model_reasoning_effort="{effort}"',
               *codex_overrides(py), sid, text]
    else:
        exe = shutil.which("codex") or "codex"
        cmd = [exe, "exec", "--json", "--skip-git-repo-check", "-C", str(C.ROOT), "--sandbox", "workspace-write",
               "-m", model, "-c", f'model_reasoning_effort="{effort}"', *codex_overrides(py), text]
    (sdir / f"{name}.command.json").write_text(json.dumps({"agent": agent, "cmd": cmd, "prompt": text, "model": model,
                                                           "effort": effort, "resume": resume,
                                                           "max_hours": max_hours, "max_tokens": max_tokens,
                                                           "started": now()}, indent=1))
    env = dict(os.environ)
    env["PATH"] = f"{Path(py).parent}:{env.get('PATH', '')}"
    if agent == "claude":
        env.setdefault("MCP_TIMEOUT", "60000")              # server startup, ms
        env.setdefault("MCP_TOOL_TIMEOUT", "3600000")       # one tool call, ms (long tools run as jobs anyway)
    since = time.time()
    stop = threading.Event()
    lock = threading.Lock()
    ev = sdir / f"{name}.events.jsonl"
    with open(ev, "w") as f:
        def note(obj):
            with lock:
                f.write(json.dumps(obj) + "\n")
                f.flush()

        pump = None
        if agent == "claude":
            proc = subprocess.Popen(cmd, cwd=str(C.ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True)
            pump = threading.Thread(target=_pump, args=(proc.stdout, f, lock), daemon=True)
            pump.start()
            try:
                proc.stdin.write(text)
                proc.stdin.close()
            except BrokenPipeError:
                pass
            th = threading.Thread(target=monitor_claude, args=(proj, stamp, sid, since, stop, note), daemon=True,
                                  kwargs={"max_tokens": max_tokens, "proc": proc})
        else:
            proc = subprocess.Popen(cmd, cwd=str(C.ROOT), stdout=f, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
            th = threading.Thread(target=monitor, args=(proj, stamp, since, stop), daemon=True,
                                  kwargs={"max_tokens": max_tokens, "proc": proc})
        th.start()
        try:
            while proc.poll() is None:
                if time.time() - since > max_hours * 3600:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    note({"any2sim": "wall-clock limit reached", "hours": max_hours})
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        rc = proc.wait()
        if pump is not None:
            pump.join(timeout=60)
    stop.set()
    th.join(timeout=120)
    summary = json.loads((sdir / f"{name}.summary.json").read_text())
    summary.update({"exit_code": rc, "finished": now()})
    (sdir / f"{name}.summary.json").write_text(json.dumps(summary, indent=1))
    plot_usage(proj, name)
    return summary


def plot_usage(proj: Project, session: str | None = None) -> str | None:
    """session: "<agent>_<stamp>" (a bare stamp means codex); None = the project's latest session."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sdir = proj.path("sessions")
    if session is None:
        cands = session_files(proj, "usage.jsonl")
        if not cands:
            return None
        session = cands[-1].name.split(".")[0]
    elif not session.startswith(tuple(a + "_" for a in AGENTS)):
        session = "codex_" + session
    stamp = session.split("_", 1)[1]
    rows = [json.loads(l) for l in open(sdir / f"{session}.usage.jsonl")]
    if not rows:
        return None
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for thread in sorted({r["thread"] for r in rows}):
        rr = [r for r in rows if r["thread"] == thread]
        t = [dt.datetime.fromisoformat(r["t"].replace("Z", "+00:00")) for r in rr]
        lab = thread.split(":")[0] + ":" + thread.split(":")[1][:8]
        ax[0].plot(t, [r["total"]["total_tokens"] / 1e6 for r in rr], label=lab)
        ax[1].plot(t, [r["last"]["input_tokens"] / 1e3 for r in rr], ".", ms=3, label=lab)
    ax[0].set_ylabel("cumulative tokens (M)")
    ax[1].set_ylabel("input tokens per call (k)")
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"token usage, {session.split('_')[0]} session {stamp}")
    fig.autofmt_xdate()
    fig.tight_layout()
    out = sdir / f"{session}.usage.png"
    fig.savefig(out, dpi=110)
    return str(out)
