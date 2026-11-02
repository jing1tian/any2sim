"""Execute a commanded robot motion of the Scene IR in MuJoCo physics and record what actually happens.

The robot is driven only through its position actuators; objects move only by contact and gravity (simkit). The
achieved motion (robot joints + object poses at every video frame) is written to scene/motions/<out>.npz and
registered in the Scene IR (kind achieved); entities get `trajectory: {motion: <out>}` and render.motion is set,
so blender_render / export_usd / video_compare replay exactly this rollout. Also renders a MuJoCo preview from the
source camera. Re-running reproduces the result (deterministic).

params:
  motion: <commanded motion id in the Scene IR>        (kind commanded; q are actuator targets)
  out: null                                            achieved motion id (default <motion>_achieved)
  start: "first_command" | "initial_q";  settle_s: 0.0
  record_fps: null (= source video fps)
  register: true                                       write the motion into scene.yaml (+ entity trajectories)
  preview: {scale: 0.5, every: 1} | null               mujoco/preview_<out>.mp4 (geometry colours, contacts)
  check_determinism: true                              a second rollout must reproduce q and object poses
Returns metrics (penetration by category, tracking error, saturation, limits, object start/end, robot-object
contact frames), the model audit and outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import write_json, write_video

ENV = "base"
GPU = False


def run(project: Path, motion: str, out: str | None = None, start: str = "first_command", settle_s: float = 0.0,
        record_fps: float | None = None, register: bool = True, preview: dict | None = None,
        check_determinism: bool = True) -> dict:
    from .. import simkit as sk
    from ..scene_ir import camera_poses, load, load_motion, save, validate
    scene = load(project / "scene")
    mv = scene["motions"][motion]
    if mv["kind"] != "commanded":
        raise ValueError(f"motion {motion} is {mv['kind']}; mujoco_rollout executes commanded motions")
    M = load_motion(project / "scene", scene, motion)
    cmd = {"t": M["t"], "joint_names": M["joint_names"], "q": M.get("ctrl", M["q"])}
    sim = sk.Sim(project)
    R = sim.rollout(cmd, record_fps=record_fps, start=start, settle_s=settle_s)
    res = {"metrics": R["metrics"], "audit": sim.audit()}
    if check_determinism:
        R2 = sim.rollout(cmd, record_fps=record_fps, start=start, settle_s=settle_s, record_contacts=False)
        res["determinism"] = {"max_q_diff": float(np.abs(R2["q"] - R["q"]).max()),
                              "max_obj_diff_m": float(np.abs(R2["obj_T"] - R["obj_T"]).max()) if R["obj_T"].size else 0.0}
    oid = out or f"{motion}_achieved"
    outs = []
    if register:
        rel = sk.save_achieved(project, scene, R, oid, mv.get("robot"), motion)
        for n in sim.rigid:
            scene["entities"][n]["trajectory"] = {"motion": oid}
        scene["render"] = {"motion": oid}
        save(project / "scene", scene)
        v = validate(project / "scene", scene)
        res["validate"] = {"ok": v["ok"], "errors": v["errors"][:10]}
        outs.append(f"scene/{rel}")
    write_json({k: v for k, v in R.items() if k == "contacts"}, project / "mujoco" / f"contacts_{oid}.json")
    if preview is not False:
        pv = preview or {}
        from ..robot.render import CalibRenderer
        c = scene["cameras"]["source_cam"]
        s = pv.get("scale", 0.5)
        W, H = int(c["width"] * s), int(c["height"] * s)
        K = np.asarray(c["K"], float).copy()
        K[:2] *= s
        poses = camera_poses(project / "scene", scene)
        ren = CalibRenderer(sim.m, W, H, groups=(0, 1, 2))
        frames = []
        jn = R["joint_names"]
        for k in range(0, len(R["t"]), pv.get("every", 1)):
            sim.reset({j: v for j, v in zip(jn, R["q"][k])}, {n: R["obj_T"][k, i] for i, n in enumerate(sim.rigid)})
            f = int(round(R["frame"][k]))
            Tc = poses[min(f, len(poses) - 1)] if len(poses) > 1 else poses[0]
            frames.append(ren.rgb(sim.d, K, Tc))
        ren.close()
        pth = f"mujoco/preview_{oid}.mp4"
        write_video(frames, project / pth, fps=sim.fps)
        outs.append(pth)
    write_json(res, project / "mujoco" / f"rollout_{oid}.json")
    return {**res, "motion": oid, "outputs": outs + [f"mujoco/rollout_{oid}.json", "mujoco/scene.xml"]}
