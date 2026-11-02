"""Initialize/freeze the realism protocol, measure diagnostics, or check the independent stage gate.

Actions: init (protect achieved trajectory and create editable protocol), freeze
(lock declared frames/regions/references), check (measure signal-chain images and
ISP held-out regressions), fallback_isp (explicit identity for rejected cameras;
rerender required), adopt (validate and publish the evaluator's unchanged decision),
gate (verify current review, assets and evidence).
Never writes a passing visual review. Missing evidence blocks the stage.
"""
from pathlib import Path

ENV = "base"
GPU = False


def run(project: Path, action: str = "check", cameras: list | None = None,
        manifest: str = "realism/renders/nominal/diagnostics.json", out: str = "realism/check.json",
        review: str | None = None) -> dict:
    from .. import realism as R
    if action == "init":
        return R.initialize(project, cameras)
    if action == "freeze":
        return R.freeze_protocol(project)
    if action == "check":
        return R.check(project, manifest, out)
    if action == "gate":
        from .. import render_pipeline as RP
        from ..scene_ir import load
        saved = R.read_json(project / "render/realism_baseline.json")
        snapshot = RP.baseline(project, load(project / "scene"), saved["settings"])
        return RP.review(project, snapshot, cameras or saved["settings"]["cameras"], required=True)
    if action == "adopt" and review:
        return R.adopt_review(project, review)
    if action == "fallback_isp":
        return R.fallback_isp(project, out)
    raise ValueError("action must be init, freeze, check, fallback_isp, gate or adopt (with review path)")
