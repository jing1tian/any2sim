"""Make rest-state poses physically consistent: resolve penetrations, then settle under gravity in MuJoCo.

For every state set (the initial one and each other rest state, with the other objects at their pose at that
time), objects are processed bottom-up: an object penetrating the table or another object is lifted in 1 mm
steps (at most `max_lift_m`) until the deepest contact is above -`contact_tol_m`; then all objects of the set are
simulated for `settle_s`. The settled pose replaces the fitted one when the change is small (drift < `max_drift_m`,
rotation < `max_rot_deg`) and the source-view silhouette IoU (MuJoCo segmentation, openings filled) on the
state's fit and held-out frames stays >= `min_iou` and drops by at most `max_iou_drop`. Every change is a ScenePatch in
scene/patches.jsonl (entity, field, before, after, rationale, evidence) and is written back to scene/scene.yaml.
Run compile_mjcf afterwards (the MJCF is regenerated from the IR).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..common import read_json, read_mask, write_json
from ..geom3d import cv_to_gl, mat_to_quat, quat_to_mat
from ..project import now

ENV = "base"
GPU = False


def run(project: Path, settle_s: float = 1.5, contact_tol_m: float = 0.001, max_lift_m: float = 0.02,
        max_drift_m: float = 0.02, max_rot_deg: float = 10.0, max_iou_drop: float = 0.02,
        min_iou: float = 0.85) -> dict:
    import cv2
    import mujoco
    import yaml
    from .compile_mjcf import build_xml
    scene_p = project / "scene" / "scene.yaml"
    scene = yaml.safe_load(scene_p.read_text())
    cams = read_json(project / "work" / "world" / "cameras.json")
    Ts = np.asarray(cams["T_world_cam"])
    ref = scene["cameras"]["source_cam"]["ref_frame"]
    rigid = {n: e for n, e in scene["entities"].items() if e.get("capability") == "rigid"}
    init = {n: e["initial_state"] for n, e in rigid.items()}
    (project / "mujoco").mkdir(exist_ok=True)
    (project / "mujoco" / "scene.xml").write_text(build_xml(project, scene, init, Ts[ref]))
    model = mujoco.MjModel.from_xml_path(str(project / "mujoco" / "scene.xml"))
    labels = read_json(project / "work" / "masks" / "labels.json")["labels"]
    name_to_label = {v: int(k) for k, v in labels.items()}
    H, W = cams["height"], cams["width"]
    renderer = mujoco.Renderer(model, H, W)
    renderer.enable_segmentation_rendering()

    def set_pose(data, n, T):
        adr = model.jnt_qposadr[model.joint(f"{n}_free").id]
        data.qpos[adr:adr + 3] = T[:3, 3]
        data.qpos[adr + 3:adr + 7] = mat_to_quat(T[:3, :3])
        data.qvel[model.jnt_dofadr[model.joint(f"{n}_free").id]:][:6] = 0

    def get_pose(data, n):
        adr = model.jnt_qposadr[model.joint(f"{n}_free").id]
        T = np.eye(4)
        T[:3, 3] = data.qpos[adr:adr + 3]
        T[:3, :3] = quat_to_mat(data.qpos[adr + 3:adr + 7])
        return T

    def deepest(data, n, below):
        """Deepest contact of body n with the world or the bodies in `below` (objects placed before it)."""
        mujoco.mj_forward(model, data)
        bid = model.body(n).id
        ok = {0} | {model.body(b).id for b in below}
        d = []
        for c in data.contact[:data.ncon]:
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if bid in (b1, b2) and (b2 if b1 == bid else b1) in ok:
                d.append(c.dist)
        return min(d) if d else 0.0

    def iou_at(poses, n, frames):
        out = []
        for f in frames:
            data = mujoco.MjData(model)
            for m, T in poses.items():
                set_pose(data, m, T if T is not None else np.r_[np.c_[np.eye(3), [10 + 2 * len(m), 10, 1]], [[0, 0, 0, 1]]])
            cid = model.camera("source_cam").id
            Tgl = cv_to_gl(Ts[f])
            model.cam_pos[cid] = Tgl[:3, 3]
            model.cam_quat[cid] = mat_to_quat(Tgl[:3, :3])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="source_cam")
            seg = renderer.render()[..., 0]
            gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(n).id and model.geom_group[g] == 1]
            others = [g for g in range(model.ngeom) if model.geom_group[g] == 1 and model.geom_bodyid[g] not in (0, model.body(n).id)]
            p = np.isin(seg, gids).astype(np.uint8)
            cn, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            p = (cv2.drawContours(np.zeros_like(p), cn, -1, 1, -1) > 0) & ~np.isin(seg, others)
            m = read_mask(project / "work" / "masks" / "objects" / f"frame{f:05d}.png") == name_to_label[n]
            v = ~(read_mask(project / "work" / "masks" / "dynamic" / f"frame{f:05d}.png") > 0)
            out.append(float((p & m & v).sum() / max(((p | m) & v).sum(), 1)))
        return float(np.mean(out)) if out else None

    sets = [("initial", init, ref)]
    for n, e in rigid.items():
        for s, st in e["states"].items():
            if s == init[n]:
                continue
            t = st["frames"][0]
            combo = {}
            for m, f in rigid.items():
                cur = [k for k, v in f["states"].items() if v["frames"][0] <= t <= v["frames"][1]]
                combo[m] = s if m == n else (cur[0] if cur else None)
            sets.append((f"{n}:{s}", combo, t))

    patches, report = [], []
    for set_name, combo, _ in sets:
        present = {n: s for n, s in combo.items() if s is not None}
        poses = {n: np.asarray(rigid[n]["states"][s]["T_world_obj"]) for n, s in present.items()}
        # bottom-up: lowest object first
        order = sorted(present, key=lambda n: poses[n][2, 3])
        data = mujoco.MjData(model)
        for n in combo:
            set_pose(data, n, poses[n] if n in poses else np.r_[np.c_[np.eye(3), [10 + 2 * len(n), 10, 1]], [[0, 0, 0, 1]]])
        lifts = {}
        for k, n in enumerate(order):
            lift = 0.0
            while deepest(data, n, order[:k]) < -contact_tol_m and lift < max_lift_m:
                lift += 0.001
                T = poses[n].copy()
                T[2, 3] += lift
                set_pose(data, n, T)
            poses[n] = get_pose(data, n)
            lifts[n] = lift
        for _ in range(int(settle_s / model.opt.timestep)):
            mujoco.mj_step(model, data)
        rec = {"set": set_name, "objects": {}}
        for n, s in present.items():
            # only the state this set is about is adopted (others are judged in their own set)
            if set_name != "initial" and not set_name.startswith(f"{n}:"):
                continue
            T0 = np.asarray(rigid[n]["states"][s]["T_world_obj"])
            T1 = get_pose(data, n)
            drift = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
            dR = T0[:3, :3].T @ T1[:3, :3]
            rot = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
            st = rigid[n]["states"][s]
            frames = st["fit_frames"] + st["heldout_frames"]
            others0 = {m: (np.asarray(rigid[m]["states"][q]["T_world_obj"]) if q else None) for m, q in combo.items()}
            others1 = {**others0, n: T1}
            iou0, iou1 = iou_at(others0, n, frames), iou_at(others1, n, frames)
            ok = drift < max_drift_m and rot < max_rot_deg and (iou0 - iou1) <= max_iou_drop and iou1 >= min_iou
            rec["objects"][n] = {"state": s, "lift_m": lifts[n], "drift_m": drift, "rotation_deg": rot,
                                 "iou_before": iou0, "iou_after": iou1, "adopted": ok}
            rigid[n]["states"][s]["physics_refine"] = {"iou_before": iou0, "iou_settled": iou1, "adopted": bool(ok),
                                                       "lift_m": lifts[n], "drift_m": drift, "rotation_deg": rot}
            if ok:
                rigid[n]["states"][s]["T_world_obj"] = T1.tolist()
                rigid[n]["states"][s]["eval_summary_note"] = f"eval_summary is pre-refine; source-view IoU after refine {iou1:.3f}"
                rigid[n]["states"][s]["pose_provenance"] = "fitted, then penetration-resolved and settled (MuJoCo)"
                patches.append({"time": now(), "entity": n, "field": f"states.{s}.T_world_obj", "before": T0.tolist(),
                                "after": T1.tolist(), "rationale": f"physics_refine: lift {lifts[n] * 1000:.0f} mm, "
                                f"settle drift {drift * 1000:.1f} mm / {rot:.1f} deg, IoU {iou0:.3f} -> {iou1:.3f}",
                                "evidence": "mujoco/physics_refine.json"})
        report.append(rec)
    renderer.close()
    for n, e in rigid.items():
        e["T_world_entity"] = e["states"][e["initial_state"]]["T_world_obj"]
    scene_p.write_text(yaml.safe_dump(scene, sort_keys=False, allow_unicode=True))
    with open(project / "scene" / "patches.jsonl", "a") as f:
        for p in patches:
            f.write(json.dumps(p) + "\n")
    write_json({"sets": report, "patches": len(patches)}, project / "mujoco" / "physics_refine.json")
    return {"sets": report, "patches": len(patches), "outputs": ["scene/scene.yaml", "scene/patches.jsonl",
                                                                 "mujoco/physics_refine.json"]}
