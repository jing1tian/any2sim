"""any2sim command line.

  any2sim new <id> --video demo.mp4 [--task "..."]      create projects/<id> with a video-only ObservationBundle
  any2sim new <id> --from <project>[@<cand>] --stage trajectory [--tracks T --tracks-map M --robot assets:R]
                                                        resume at a later stage from another project's result
  any2sim new <id> --from <traj project>[@<cand>] --stage synthesis
                                                        data synthesis from an independently reviewed realism result
  any2sim tool <project> <tool> '<json params>'         run one tool in its env, write a receipt
  any2sim tools                                         list tools (name, env, gpu, summary)
  any2sim publish-robot <project> [--name N]           copy a verified bundle (projects/<id>/robot) to assets/robots
  any2sim status <project>                              receipts and scene state
  any2sim run <project> [--agent codex|claude --model M --effort E --max-hours H --max-tokens N] [--resume [SESSION]]
                                                        headless agent session, Codex (default) or Claude Code
                                                        (trace + tokens); --resume continues the project's last
                                                        session of that agent (or the given one)
  any2sim usage <project>                               token usage of the latest session (summary + plot)
  any2sim mcp                                           MCP tool server (stdio), started by the agent runtime
  any2sim validate <project>                            Scene IR structural validation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as C


def cmd_new(a):
    from .project import Project
    if a.from_:
        from .stages import new_from
        p = new_from(a.id, a.from_, a.stage, a.tracks, a.tracks_map, a.robot, a.import_work, a.task, a.notes or "",
                     a.force)
        print(json.dumps({"project": str(p.root), "stage": a.stage, "from": a.from_}, indent=1))
        return
    if not a.video:
        sys.exit("--video or --from is required")
    root = Path(C.config()["projects_dir"]) / a.id
    if (root / "project.yaml").exists() and not a.force:
        sys.exit(f"exists: {root} (use --force to re-ingest)")
    proj = Project.create(root, {"video": str(Path(a.video).resolve())}, a.notes or "")
    from .toolrun import run_tool
    r = run_tool(proj.root, "ingest_video", {"video": str(Path(a.video).resolve()), "task": a.task,
                                             "frame_step": a.frame_step})
    print(json.dumps({"project": str(proj.root), "status": r["status"], "result": r["result"]}, indent=1))


def cmd_tool(a):
    from .toolrun import run_tool
    r = run_tool(a.project, a.tool, json.loads(a.params), a.gpu, a.inputs)
    print(json.dumps({k: r[k] for k in ("receipt_id", "status", "duration_s", "result")}, indent=1, default=str)[:20000])
    if r["status"] != "ok":
        print(r.get("log_tail", ""), file=sys.stderr)
        sys.exit(1)


def cmd_tools(a):
    from .toolrun import _tool_meta
    for p in sorted((C.ROOT / "src" / "any2sim" / "tools").glob("*.py")):
        if p.stem.startswith("_"):
            continue
        env, gpu = _tool_meta(p.stem)
        doc = p.read_text().split('"""')[1].strip().splitlines()[0] if '"""' in p.read_text() else ""
        print(f"{p.stem:28s} env={env:7s} gpu={str(gpu):5s} {doc}")


def cmd_status(a):
    from .project import Project
    proj = Project(a.project)
    for r in proj.receipts():
        print(f"{r['receipt_id']:52s} {r['status']:6s} {r['duration_s']:8.1f}s")
    s = proj.path("scene", "scene.yaml")
    print("scene:", s if s.exists() else "none")


def cmd_run(a):
    from .launcher import launch
    print(json.dumps(launch(a.project, a.model, a.effort, a.max_hours, a.max_tokens, resume=a.resume, agent=a.agent),
                     indent=1))


def cmd_usage(a):
    from .launcher import plot_usage, session_files
    from .project import Project
    proj = Project(a.project)
    s = session_files(proj, "summary.json")
    if not s:
        sys.exit("no sessions")
    print(s[-1].read_text())
    print("plot:", plot_usage(proj, s[-1].name.split(".")[0]))


def cmd_publish_robot(a):
    from .stages import publish_robot
    print(json.dumps(publish_robot(a.project, a.name), indent=1))


def cmd_validate(a):
    from .scene_ir import validate
    r = validate(Path(a.project) / "scene")
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["ok"] else 1)


def cmd_mcp(a):
    from .mcp_server import main as mcp_main
    mcp_main()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="any2sim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new"); p.add_argument("id"); p.add_argument("--video")
    p.add_argument("--from", dest="from_"); p.add_argument("--stage", choices=("trajectory", "realism", "synthesis"), default="trajectory")
    p.add_argument("--tracks"); p.add_argument("--tracks-map"); p.add_argument("--robot")
    p.add_argument("--import-work", nargs="*", default=[])
    p.add_argument("--task"); p.add_argument("--notes"); p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_new)
    p = sub.add_parser("tool"); p.add_argument("project"); p.add_argument("tool")
    p.add_argument("params", nargs="?", default="{}"); p.add_argument("--gpu"); p.add_argument("--inputs", nargs="*")
    p.set_defaults(fn=cmd_tool)
    p = sub.add_parser("tools"); p.set_defaults(fn=cmd_tools)
    p = sub.add_parser("status"); p.add_argument("project"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("run"); p.add_argument("project"); p.add_argument("--agent", choices=("codex", "claude"), default="codex")
    p.add_argument("--model", help="default: gpt-6-astra (codex), claude-opus-5-5 (claude)")
    p.add_argument("--effort", default="medium"); p.add_argument("--max-hours", type=float, default=8.0)
    p.add_argument("--max-tokens", type=int); p.add_argument("--resume", nargs="?", const="last")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("usage"); p.add_argument("project"); p.set_defaults(fn=cmd_usage)
    p = sub.add_parser("mcp"); p.set_defaults(fn=cmd_mcp)
    p = sub.add_parser("validate"); p.add_argument("project"); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("publish-robot"); p.add_argument("project"); p.add_argument("--name")
    p.set_defaults(fn=cmd_publish_robot)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
