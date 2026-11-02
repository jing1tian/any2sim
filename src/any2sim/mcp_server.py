"""MCP server: the agent's door to the tool catalog. Tools run outside the agent's sandbox (GPUs, model envs).

Five tools: list_tools, describe_tool, run_tool, job_status, job_cancel. Everything else the agent does with its own
shell and editor inside the repository (reading evidence, writing project scripts, editing the Scene IR).
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import config as C
from . import jobs

mcp = FastMCP("any2sim")
TOOLS = C.ROOT / "src" / "any2sim" / "tools"


def _project(project: str) -> Path:
    p = Path(project)
    if not p.is_absolute():
        p = C.ROOT / p
    return p.resolve()


def _doc_and_signature(name: str) -> tuple[str, str, str, bool]:
    src = (TOOLS / f"{name}.py").read_text()
    mod = ast.parse(src)
    doc = ast.get_docstring(mod) or ""
    env, gpu, sig = "base", False, ""
    for node in mod.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "ENV":
                env = ast.literal_eval(node.value)
            if node.targets[0].id == "GPU":
                gpu = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            args = node.args
            defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
            parts = []
            for a, d in zip(args.args[1:], defaults[1:]):
                ann = ast.unparse(a.annotation) if a.annotation else ""
                parts.append(f"{a.arg}: {ann}" + (f" = {ast.unparse(d)}" if d is not None else ""))
            sig = f"run(project, {', '.join(parts)})"
            fdoc = ast.get_docstring(node)
            if fdoc:
                doc += "\n\nrun(): " + fdoc
    return doc, sig, env, gpu


@mcp.tool()
def list_tools(query: str = "") -> str:
    """List any2sim tools (name, env, gpu, one-line summary); optional substring filter."""
    rows = []
    for p in sorted(TOOLS.glob("*.py")):
        if p.stem.startswith("_"):
            continue
        doc, _, env, gpu = _doc_and_signature(p.stem)
        line = doc.strip().splitlines()[0] if doc.strip() else ""
        if query.lower() in (p.stem + " " + doc).lower():
            rows.append({"tool": p.stem, "env": env, "gpu": gpu, "summary": line})
    return json.dumps(rows, indent=1)


@mcp.tool()
def describe_tool(name: str) -> str:
    """Full description of one tool: what it does, inputs/outputs, parameters with defaults."""
    doc, sig, env, gpu = _doc_and_signature(name)
    return json.dumps({"tool": name, "env": env, "gpu": gpu, "signature": sig, "doc": doc}, indent=1)


@mcp.tool()
def run_tool(project: str, tool: str, params: dict | None = None, wait_s: float = 600,
             timeout_s: float | None = None) -> str:
    """Run a tool on a project (path relative to the repository or absolute). Waits up to wait_s seconds; a longer
    run returns its job_id (poll with job_status). Every run writes a receipt under <project>/receipts/."""
    job_id = jobs.submit(_project(project), tool, params or {}, timeout_s)
    rec = jobs.wait(_project(project), job_id, wait_s)
    return json.dumps(rec, indent=1, default=str)[:60000]


@mcp.tool()
def job_status(project: str, job_id: str, wait_s: float = 0) -> str:
    """Status/result of a job; optionally wait up to wait_s seconds for it to finish."""
    return json.dumps(jobs.wait(_project(project), job_id, wait_s), indent=1, default=str)[:60000]


@mcp.tool()
def job_cancel(project: str, job_id: str) -> str:
    """Cancel a running job."""
    return json.dumps(jobs.cancel(_project(project), job_id), indent=1, default=str)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
