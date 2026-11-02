"""Execution-only GPU OIDN acceleration of the frozen synthesis renderer.

Uses the standard render orchestration, labels and image processing. Existing renders
may be reused only when their full baseline/motion fingerprint and appearance config
match. Newly rendered episodes record the execution settings and this source hash in
their fingerprint. CPU/GPU OIDN equivalence must be checked for the selected scene.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

ENV = "render"
GPU = True


def execution():
    return {"denoiser": "OPENIMAGEDENOISE", "denoising_use_gpu": True,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def reusable(project, eid, cfg):
    from .. import render_pipeline as RP
    ep = Path(project) / "synthesis/episodes" / eid
    f = Path(project) / "synthesis/renders" / eid / "render.json"
    if not cfg.get("resume", True) or not f.exists():
        return False
    old = json.loads(f.read_text())
    previous = old.get("render_config") or {}
    requested = {k: v for k, v in cfg.items() if k not in ("baseline", "realism_review", "execution", "resume")}
    appearance = {k: v for k, v in previous.items() if k not in ("execution", "resume")}
    return (old.get("complete") and not old.get("preview") and appearance == requested
            and old.get("baseline_fingerprint") == cfg["baseline"]["fingerprint"]
            and old.get("render_fingerprint") == RP.episode_fingerprint(ep, cfg["baseline"], previous)
            and all((f.parent / (cn + ".mp4")).is_file() for cn in cfg["cameras"]))


def worker(args):
    from . import synth_render as SR
    from ..backends import blender_scene as B
    args["episodes"] = [eid for eid in args["episodes"] if not reusable(args["project"], eid, args["cfg"])]
    args["cfg"]["execution"] = execution()
    original = B.setup_cycles
    def setup(*a, **kw):
        device = original(*a, **kw)
        import bpy
        bpy.context.scene.cycles.denoiser = "OPENIMAGEDENOISE"
        bpy.context.scene.cycles.denoising_use_gpu = True
        return device
    B.setup_cycles = setup
    SR._worker_main(args)


def run(project: Path, **params):
    from . import synth_render as SR
    original = SR.subprocess
    def popen(cmd, *a, **kw):
        if len(cmd) >= 4 and cmd[1:3] == ["-m", "any2sim.tools.synth_render"]:
            cmd = [cmd[0], "-m", "any2sim.tools.synth_render_accelerated", cmd[3]]
        return subprocess.Popen(cmd, *a, **kw)
    SR.subprocess = SimpleNamespace(Popen=popen, run=subprocess.run, STDOUT=subprocess.STDOUT)
    try:
        return SR.run(project, **params)
    finally:
        SR.subprocess = original


if __name__ == "__main__":
    worker(json.loads(sys.argv[1]))
