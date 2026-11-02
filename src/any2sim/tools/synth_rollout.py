"""Synthesize physically executed episodes: sample variants of the scene, re-anchor the demonstration, simulate, keep
the successful ones (synthesis stage; any2sim.synthkit does the work, this tool runs it in parallel and reports).

Each attempt k (deterministic for the spec seed and k): sample a variant (object/target placement, materials,
lights, camera mounts, physics, sensor, trajectory speed/grasp/via-point) -> synthkit.transfer (object-centric palm
re-anchoring + arm IK) -> MuJoCo rollout of the variant scene with the robot's own actuators -> task predicate +
physics checks. Accepted episodes go to synthesis/episodes/<id>/ (variant.yaml, commanded.npz, achieved.npz,
contacts.json, rollout.json; the MJCF is synthesis/mujoco/<id>.xml); every attempt is a row of
synthesis/attempts.jsonl (accepted | rejected with stage and reason). The nominal episode (no randomisation: the
demo itself) is simulated first and must reproduce the demo's achieved motion.

params:
  spec: synthesis/spec.yaml           randomisation spec (synthkit docstring)
  episodes: 50                        accepted episodes wanted (excluding nominal)
  max_attempts: null                  default 5 x episodes
  workers: 12                         parallel CPU processes (shared machine: keep it moderate)
  resume: true                        keep accepted episodes of an earlier run with the same spec; continue
  keep_rejected: false                keep files of rejected episodes (debugging)
  check_determinism: true             re-simulate the first accepted episode in this process and compare
  sheet: 8                            MuJoCo first/last frames of this many episodes -> evidence sheet
Writes synthesis/episodes.yaml (accepted ids, spec sha, stats), evidence/synth_rollout.json, synth_rollout_sheet.jpg,
synth_rollout_layout.png (sampled placements, accepted vs rejected).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..common import write_json

ENV = "base"
GPU = False

_CTX = {}


def _worker(args):
    project, spec_rel, attempt, nominal, keep = args
    from .. import synthkit as sy
    key = (project, spec_rel)
    if key not in _CTX:
        _CTX[key] = sy.Context(project, spec_rel)
    try:
        return sy.run_attempt(_CTX[key], attempt, Path(project) / "synthesis", nominal=nominal, keep_rejected=keep)
    except Exception as ex:                                    # noqa: BLE001  (one bad attempt never kills the run)
        import traceback
        return {"id": "nominal" if nominal else f"ep_{attempt:05d}", "attempt": attempt, "status": "rejected",
                "stage": "error", "reason": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:]}


def _sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sheet(project: Path, ctx, ids: list, out: Path):
    from .. import simkit as sk
    from ..common import contact_sheet, write_rgb
    from ..robot.render import CalibRenderer
    from ..scene_ir import camera_poses, load_motion, render_head_camera
    from .. import synthkit as sy
    import yaml
    ims, labels = [], []
    for eid in ids:
        ep = project / "synthesis" / "episodes" / eid
        var = yaml.safe_load((ep / "variant.yaml").read_text())
        sc = sy.materialize(ctx, var, ep)
        sim = sk.Sim(project, scene=sc, mdir=project / "synthesis" / "mujoco", xml_name=f"{eid}.xml", build=False)
        cam = sc["cameras"][render_head_camera(sc)]
        K = np.asarray(cam["K"], float) * [[0.5], [0.5], [1]]
        r = CalibRenderer(sim.m, cam["width"] // 2, cam["height"] // 2)
        mo = load_motion(project / "scene", sc, "achieved")
        Tc = camera_poses(project / "scene", sc, render_head_camera(sc))[0]
        for k, lab in ((0, "start"), (len(mo["t"]) - 1, "end")):
            q = dict(zip(mo["joint_names"], mo["q"][k]))
            objs = {n: mo["obj_T"][k, i] for i, n in enumerate(mo["obj_names"])}
            sim.reset(q, objs)
            ims.append(r.rgb(sim.d, K, Tc))
            labels.append(f"{eid} {lab}")
        r.close()
    if ims:
        write_rgb(out, contact_sheet(ims, cols=4, width=480, labels=labels))


def _layout(ctx, rows: list, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..scene_ir import pose_at
    from .. import synthkit as sy
    sp = ctx.spec.get("spatial") or {}
    fig, ax = plt.subplots(figsize=(6, 6))
    if sp.get("support"):
        poly, _ = sy._support_polygon(ctx, sp["support"], float(sp.get("support_margin_m", 0.03)))
        ax.fill(np.r_[poly[:, 0], poly[:1, 0]], np.r_[poly[:, 1], poly[:1, 1]], color="0.92")
    marks = {}
    for n in (sp.get("objects") or {}):
        p0 = pose_at(ctx.scene["entities"][n], None)[:3, 3]
        ax.plot(p0[0], p0[1], "k*", ms=12)
        ax.annotate(n, p0[:2])
        marks[n] = "o" if n == ctx.demo.obj else "s"
    for r in rows:
        v = r.get("variant") or {}
        if r["status"] == "accepted":
            import yaml
            f = ctx.project / "synthesis" / "episodes" / r["id"] / "variant.yaml"
            v = yaml.safe_load(f.read_text()) if f.exists() else {}
        for n, e in (v.get("entities") or {}).items():
            if "T_world_obj" in e and n in marks:
                p = np.asarray(e["T_world_obj"])[:3, 3]
                ax.plot(p[0], p[1], marks[n], color="tab:green" if r["status"] == "accepted" else "tab:red",
                        ms=4, alpha=0.7)
    pts = np.array([l.get_xydata()[0] for l in ax.lines]) if ax.lines else np.zeros((1, 2))
    lo, hi = pts.min(0) - 0.12, pts.max(0) + 0.12
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("sampled placements (green accepted, red rejected; o object, s target)")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run(project: Path, spec: str = "synthesis/spec.yaml", episodes: int = 50, max_attempts: int | None = None,
        workers: int = 12, resume: bool = True, keep_rejected: bool = False, check_determinism: bool = True,
        sheet: int = 8) -> dict:
    import multiprocessing as mp
    import shutil
    import yaml
    from .. import simkit as sk
    from .. import synthkit as sy
    from ..scene_ir import _plain, load_motion
    from ..backends.mujoco_scene import write_scene
    if episodes > 0:
        from .realism_check import run as check_realism
        check_realism(project, action="gate")
    out = project / "synthesis"
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    spec_sha = _sha(project / spec)
    idx_f = out / "episodes.yaml"
    att_f = out / "attempts.jsonl"
    prev = yaml.safe_load(idx_f.read_text()) if idx_f.exists() else None
    rows = []
    if resume and prev and prev.get("spec_sha256") == spec_sha and att_f.exists():
        rows = [json.loads(l) for l in att_f.read_text().splitlines() if l.strip()]
    else:
        for d in (out / "episodes", out / "mujoco"):
            shutil.rmtree(d, ignore_errors=True)
        (out / "episodes").mkdir(parents=True, exist_ok=True)
        att_f.write_text("")
    ctx = sy.Context(project, spec)
    # shared MuJoCo assets once (workers then only read them)
    write_scene(project, ctx.scene, mdir=out / "mujoco", name="_base.xml")
    done = {r["attempt"] for r in rows if not r.get("nominal")}
    max_attempts = int(max_attempts or 5 * episodes)
    res = {"spec": spec, "spec_sha256": spec_sha, "demo": {"commanded": ctx.demo.cmd_id, "achieved": ctx.demo.ach_id,
           "events": ctx.demo.events}}
    pool = mp.get_context("spawn").Pool(max(1, int(workers)))
    try:
        if not any(r.get("nominal") for r in rows):
            r = pool.apply(_worker, ((str(project), spec, -1, True, keep_rejected),))
            rows.append(r)
            with att_f.open("a") as f:
                f.write(json.dumps(_plain(r), default=float) + "\n")
        nominal = next(r for r in rows if r.get("nominal"))
        accepted = sorted(r["attempt"] for r in rows if r["status"] == "accepted" and not r.get("nominal"))
        k = (max(done) + 1) if done else 0
        bs = max(1, int(workers)) * 2
        while len(accepted) < episodes and k < max_attempts:
            batch = list(range(k, min(k + bs, max_attempts)))
            k = batch[-1] + 1
            got = pool.map(_worker, [(str(project), spec, a, False, keep_rejected) for a in batch])
            got.sort(key=lambda r: r["attempt"])
            with att_f.open("a") as f:
                for r in got:
                    if r["status"] == "accepted" and len(accepted) >= episodes:
                        r["status"] = "surplus"                    # beyond the target: dropped (determinism)
                        shutil.rmtree(out / "episodes" / r["id"], ignore_errors=True)
                    if r["status"] == "accepted":
                        accepted.append(r["attempt"])
                    rows.append(r)
                    f.write(json.dumps(_plain(r), default=float) + "\n")
    finally:
        pool.close()
        pool.join()
    acc_rows = [r for r in rows if r["status"] == "accepted" and not r.get("nominal")]
    ids = [r["id"] for r in sorted(acc_rows, key=lambda r: r["attempt"])]
    # nominal must reproduce the demo's achieved motion
    nom = {"status": nominal["status"], "reason": nominal.get("reason")}
    if nominal["status"] == "accepted":
        from ..scene_ir import load
        ep = out / "episodes" / "nominal"
        a = dict(np.load(ep / "achieved.npz"))
        d = ctx.demo.ach
        n = min(len(a["t"]), len(d["t"]))
        jn = [str(x) for x in a["joint_names"]]
        dj = [list(d["joint_names"]).index(j) for j in jn]
        oi = list(d["obj_names"]).index(ctx.demo.obj)
        ai = [str(x) for x in a["obj_names"]].index(ctx.demo.obj)
        nom.update({"q_max_diff_rad": float(np.abs(a["q"][:n] - d["q"][:n][:, dj]).max()),
                    "object_final_diff_m": float(np.linalg.norm(a["obj_T"][n - 1, ai, :3, 3] - d["obj_T"][n - 1, oi, :3, 3]))})
        nom["reproduces_demo"] = nom["q_max_diff_rad"] < 1e-3 and nom["object_final_diff_m"] < 0.005
    res["nominal"] = nom
    # determinism: re-simulate the first accepted episode here
    if check_determinism and ids:
        eid = ids[0]
        ep = out / "episodes" / eid
        var = yaml.safe_load((ep / "variant.yaml").read_text())
        sc = sy.materialize(ctx, var, ep)
        sim = sk.Sim(project, scene=sc, mdir=out / "mujoco", xml_name=f"{eid}.xml", build=False)
        cm = load_motion(project / "scene", sc, "commanded")
        R = sim.rollout({"t": cm["t"], "joint_names": cm["joint_names"], "q": cm["q"]}, start="first_command",
                        settle_s=float((ctx.spec.get("rollout") or {}).get("settle_s", 0.0)), record_contacts=False)
        a = load_motion(project / "scene", sc, "achieved")
        res["determinism"] = {"episode": eid, "q_max_diff": float(np.abs(R["q"] - a["q"]).max()),
                              "obj_max_diff_m": float(np.abs(R["obj_T"][..., :3, 3] - a["obj_T"][..., :3, 3]).max())}
    # stats
    tried = [r for r in rows if not r.get("nominal") and r["status"] != "surplus"]
    by_stage = {}
    for r in tried:
        if r["status"] == "rejected":
            by_stage[r.get("stage", "?")] = by_stage.get(r.get("stage", "?"), 0) + 1
    reasons = {}
    for r in tried:
        if r["status"] == "rejected":
            for part in str(r.get("reason", "")).split("; "):
                import re
                key = re.split(r"[\d:{(>]", part, maxsplit=1)[0].strip() or part[:30]
                reasons[key] = reasons.get(key, 0) + 1
    res.update({"accepted": len(ids), "attempts": len(tried), "acceptance_rate": len(ids) / max(len(tried), 1),
                "rejected_by_stage": by_stage, "rejected_by_reason": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:12]),
                "seconds_per_attempt_median": float(np.median([r.get("s", 0) for r in tried])) if tried else None})
    succ = [r["success"] for r in acc_rows]
    if acc_rows:
        res["accepted_metrics"] = {
            "end_speed_mps_max": float(max(s.get("end_speed_mps", 0) for s in succ)),
            "penetration_mm_max": {c: float(max((r["metrics"]["max_penetration_mm"].get(c, 0) for r in acc_rows)))
                                   for c in {c for r in acc_rows for c in r["metrics"]["max_penetration_mm"]}},
            "saturation_max": float(max(r["metrics"]["actuator_saturation_frac"] for r in acc_rows)),
            "ik_pos_err_mm_max": float(max(r["transfer"]["ik_pos_err_mm_max"] for r in acc_rows)),
            "duration_s": [float(min(r["transfer"]["duration_s"] for r in acc_rows)),
                           float(max(r["transfer"]["duration_s"] for r in acc_rows))]}
    idx_f.write_text(yaml.safe_dump(_plain({"spec": spec, "spec_sha256": spec_sha, "base_core_sha256": ctx.core,
                                            "nominal": "nominal" if nominal["status"] == "accepted" else None,
                                            "episodes": ids, "stats": {k: res[k] for k in ("accepted", "attempts", "acceptance_rate")}}),
                                     sort_keys=False))
    if sheet:
        try:
            _sheet(project, ctx, (["nominal"] if nominal["status"] == "accepted" else []) + ids[:int(sheet)],
                   project / "evidence" / "synth_rollout_sheet.jpg")
        except Exception as ex:                                # noqa: BLE001
            res["sheet_error"] = str(ex)
    _layout(ctx, rows, project / "evidence" / "synth_rollout_layout.png")
    write_json(res, project / "evidence" / "synth_rollout.json")
    return {**res, "outputs": ["synthesis/episodes.yaml", "synthesis/attempts.jsonl", "synthesis/episodes",
                               "evidence/synth_rollout.json", "evidence/synth_rollout_sheet.jpg",
                               "evidence/synth_rollout_layout.png"]}
