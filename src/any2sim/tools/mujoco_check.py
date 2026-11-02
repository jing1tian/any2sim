"""Compile the Scene IR to MuJoCo (mujoco/scene.xml) and measure physical plausibility of rest-state sets.

For each state set (which state every rigid entity is in; default: the initial set and one set per non-initial
state with the others at the state valid at its first frame): deepest penetration per body pair at the given poses,
then free simulation for `settle_s`: drift, rotation and the settled pose of every body. Measurement only:
whether to accept a settled pose, change geometry or add support is the caller's decision.

params:
  sets: null | {set_name: {entity: state | "hidden"}}
  settle_s: 2.0, timestep: 0.002
  probes: [{"name", "body": entity, "points": [[x,y,z],...]}]   optional: are these object-frame points inside
                                       any collision geom (e.g. a container's cavity must stay free)?
Writes mujoco/scene.xml (objects.xml when the scene has robots: mujoco_rollout writes the full scene.xml) +
assets, mujoco/checks.json (per set: contacts, deepest_m, per body drift_m,
rotation_deg, settled T_world), returns the same.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..common import write_json
from ..geom3d import mat_to_quat, quat_to_mat

ENV = "base"
GPU = False


def _probe(project: Path, scene: dict, pr: dict) -> dict:
    """Object-frame points tested against the entity's own collision geometry (exact containment)."""
    import trimesh
    e = scene["entities"][pr["body"]]
    P = np.asarray(pr["points"], float)
    inside = np.zeros(len(P), bool)
    for c in e["geometry"].get("collision") or []:
        T = np.asarray(c.get("T", np.eye(4)), float)
        Pl = (P - T[:3, 3]) @ T[:3, :3]
        if c["type"] == "mesh":
            inside |= trimesh.load(project / "scene" / c["uri"], force="mesh").contains(Pl)
        else:
            inside |= np.all(np.abs(Pl) <= np.asarray(c["size"]) / 2, axis=1)
    return {"points": int(len(P)), "inside_collision": int(inside.sum()), "inside_mask": inside.tolist()}


def run(project: Path, sets: dict | None = None, settle_s: float = 2.0, timestep: float = 0.002,
        probes: list | None = None) -> dict:
    import mujoco
    from ..scene_ir import load
    scene = load(project / "scene")
    (project / "mujoco").mkdir(exist_ok=True)
    from ..backends.mujoco_scene import write_scene
    # objects only (robots, if any, are simulated by mujoco_rollout, which writes the full scene.xml)
    path, _ = write_scene(project, scene, robots=False, timestep=timestep,
                          name="objects.xml" if scene.get("robots") else "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    rigid = {n: e for n, e in scene["entities"].items() if e.get("capability") == "rigid"}
    if sets is None:
        sets = {"initial": {}}
        for n, e in rigid.items():
            for s, st in e["states"].items():
                if s == e["initial_state"]:
                    continue
                t = st["frames"][0]
                spec = {}
                for m, f in rigid.items():
                    cur = [k for k, v in f["states"].items() if v["frames"][0] <= t <= v["frames"][1]]
                    spec[m] = s if m == n else (cur[0] if cur else "hidden")
                sets[f"{n}:{s}"] = spec
    checks = {}
    for sname, spec in sets.items():
        d = mujoco.MjData(model)
        present = []
        for n, e in rigid.items():
            jid = model.joint(f"{n}_free").id
            adr = model.jnt_qposadr[jid]
            s = spec.get(n, e["initial_state"])
            if s == "hidden":
                d.qpos[adr:adr + 3] = [10 + 3 * len(present) + 5 * hash(n) % 7, 10, 1]
                continue
            T = np.asarray(e["states"][s]["T_world_obj"], float)
            d.qpos[adr:adr + 3] = T[:3, 3]
            d.qpos[adr + 3:adr + 7] = mat_to_quat(T[:3, :3])
            present.append(n)
        mujoco.mj_forward(model, d)
        pairs = {}
        for c in d.contact[:d.ncon]:
            b1, b2 = model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name
            key = "|".join(sorted([b1 or "world", b2 or "world"]))
            pairs[key] = min(pairs.get(key, 0.0), float(c.dist))
        probe_res = {}
        for pr in probes or []:
            probe_res[pr["name"]] = _probe(project, scene, pr)
        p0 = {n: (d.body(n).xpos.copy(), d.body(n).xmat.reshape(3, 3).copy()) for n in present}
        for _ in range(int(settle_s / timestep)):
            mujoco.mj_step(model, d)
        bodies = {}
        for n in present:
            R1 = d.body(n).xmat.reshape(3, 3)
            dR = p0[n][1].T @ R1
            T1 = np.eye(4)
            jid = model.joint(f"{n}_free").id
            adr = model.jnt_qposadr[jid]
            T1[:3, 3] = d.qpos[adr:adr + 3]
            T1[:3, :3] = quat_to_mat(d.qpos[adr + 3:adr + 7])
            bodies[n] = {"drift_m": float(np.linalg.norm(d.body(n).xpos - p0[n][0])),
                         "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))),
                         "settled_T_world": T1.tolist()}
        checks[sname] = {"states": spec, "contacts_min_dist_m": pairs,
                         "deepest_m": min(pairs.values()) if pairs else 0.0, "bodies": bodies, "probes": probe_res}
    res = {"model": {"nbody": model.nbody, "ngeom": model.ngeom}, "mujoco_version": mujoco.__version__,
           "settle_s": settle_s, "sets": checks}
    write_json(res, project / "mujoco" / "checks.json")
    return {**res, "outputs": ["mujoco"]}
