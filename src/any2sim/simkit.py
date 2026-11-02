"""Physics rollouts of a Scene IR with robots (MuJoCo, CPU) for agent-written optimisers and the mujoco_rollout tool.

    from any2sim import simkit as sk
    sim = sk.Sim(project)                                     # compiles mujoco/scene.xml from scene/scene.yaml
    res = sim.rollout(cmd)                                    # cmd: {"t": (N,), "joint_names": [...], "q": (N,J)}
    res["metrics"]["max_penetration_mm"], res["q"], res["obj_T"], res["contacts"]

Robots are driven only through their position actuators (the real controller's job): commanded joint targets are
interpolated linearly in time and applied as actuator controls; objects move only through contacts and gravity.
Nothing is welded, teleported or pushed by invisible forces (`audit()` checks the compiled model for that).
Recording happens at `record_fps` (default: the source video fps) so results map 1:1 onto video frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Sim:
    def __init__(self, project, timestep: float | None = None, build: bool = True, robots: bool = True,
                 scene: dict | None = None, mdir=None, xml_name: str = "scene.xml"):
        """scene: an in-memory Scene IR (e.g. scene_ir.apply_variant(...)) instead of scene/scene.yaml; mdir/xml_name:
        where the MJCF is written (default <project>/mujoco/scene.xml)."""
        import mujoco
        from .backends.mujoco_scene import write_scene
        from .scene_ir import load
        self.mj = mujoco
        self.project = Path(project)
        self.scene = scene if scene is not None else load(self.project / "scene")
        mdir = Path(mdir) if mdir else self.project / "mujoco"
        xml_path = mdir / xml_name
        if build or not xml_path.exists():
            xml_path, self.info = write_scene(self.project, self.scene, robots=robots, timestep=timestep,
                                              name=xml_name, mdir=mdir)
        else:
            from .backends.mujoco_scene import build_xml
            _, self.info = build_xml(self.project, self.scene, robots=robots, timestep=timestep, mdir=mdir)
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)
        self.fps = float(self.scene["cameras"]["source_cam"].get("fps") or 20.0)
        m = self.m
        self.act = {m.actuator(a).name: a for a in range(m.nu)}
        self.jadr = {m.joint(j).name: int(m.jnt_qposadr[j]) for j in range(m.njnt)}
        self.rigid = [n for n in self.info["rigid"]]
        self.body_owner = {}                                 # body id -> ("robot", rid, link) | ("entity", name)
        for b in range(1, m.nbody):
            name = m.body(b).name
            if name in self.scene["entities"]:
                self.body_owner[b] = ("entity", name)
        for rid, r in self.info["robots"].items():
            for b in range(1, m.nbody):
                name = m.body(b).name
                if b not in self.body_owner and name.startswith(r["prefix"]):
                    self.body_owner[b] = ("robot", rid, name[len(r["prefix"]):])
        self.static = {n for n, e in self.scene["entities"].items() if e.get("capability") == "static"}

    # ------------------------------------------------------------------ state
    def robot_joints(self, rid: str | None = None) -> list[str]:
        out = []
        for r, v in self.info["robots"].items():
            if rid is None or r == rid:
                out += v["joints"]
        return out

    def reset(self, q: dict | None = None, objects: dict | None = None):
        """Robots at q (default each robot's initial_q / bundle init pose), rigid entities at their initial state
        (or `objects` {name: 4x4}); actuator targets = current joint values; velocities zero."""
        from .geom3d import mat_to_quat
        from .scene_ir import pose_at, robot_bundle
        mj, m, d = self.mj, self.m, self.d
        mj.mj_resetData(m, d)
        qq = {}
        for rid, r in (self.scene.get("robots") or {}).items():
            pre = self.info["robots"][rid]["prefix"]
            b = robot_bundle(self.project / "scene", self.scene, rid)
            init = b.init_q()
            init.update(r.get("initial_q") or {})
            qq.update({pre + j: v for j, v in init.items()})
        qq.update(q or {})
        for j, v in qq.items():
            if j in self.jadr:
                d.qpos[self.jadr[j]] = v
        for n in self.rigid:
            T = (objects or {}).get(n)
            if T is None:
                T = pose_at(self.scene["entities"][n], None)
            a = self.jadr[f"{n}_free"]
            d.qpos[a:a + 3] = T[:3, 3]
            d.qpos[a + 3:a + 7] = mat_to_quat(T[:3, :3])
        for name, a in self.act.items():
            if name in self.jadr:
                d.ctrl[a] = d.qpos[self.jadr[name]]
        mj.mj_forward(m, d)

    def object_pose(self, name: str) -> np.ndarray:
        from .geom3d import quat_to_mat
        a = self.jadr[f"{name}_free"]
        T = np.eye(4)
        T[:3, 3] = self.d.qpos[a:a + 3]
        T[:3, :3] = quat_to_mat(self.d.qpos[a + 3:a + 7])
        return T

    def q(self, joints) -> np.ndarray:
        return np.array([self.d.qpos[self.jadr[j]] for j in joints])

    def contacts(self) -> list[dict]:
        """Current contacts: owners, depth (m, >0 = penetration), normal force (N)."""
        out = []
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            o1, o2 = self.body_owner.get(b1, ("world",)), self.body_owner.get(b2, ("world",))
            self.mj.mj_contactForce(self.m, self.d, i, f6)
            out.append({"a": o1, "b": o2, "depth": float(-c.dist), "force": float(f6[0]), "pos": c.pos.copy()})
        return out

    # ------------------------------------------------------------------ rollout
    def rollout(self, cmd: dict, record_fps: float | None = None, duration: float | None = None,
                start: str = "first_command", settle_s: float = 0.0, objects: dict | None = None,
                record_contacts: bool = True) -> dict:
        """Execute commanded joint targets. cmd: {"t": (N,), "joint_names": [...], "q": (N,J)} (times in seconds of
        the source video, i.e. t = frame / fps). Returns recorded q, object poses, contacts per recorded frame and
        metrics. start: "first_command" (robot starts at the first target) | "initial_q". Targets reach the
        actuators scene.simulation.control_latency_s later (default 0)."""
        mj, m, d = self.mj, self.m, self.d
        t_cmd = np.asarray(cmd["t"], float)
        names = list(cmd["joint_names"])
        Q = np.asarray(cmd["q"], float)
        acts = [self.act.get(j) for j in names]
        q0 = dict(zip(names, Q[0])) if start == "first_command" else None
        self.reset(q0, objects)
        fps = record_fps or self.fps
        t0 = float(t_cmd[0])
        t_end = float(duration + t0 if duration else t_cmd[-1])
        if settle_s > 0:                                     # let objects settle under the held first command
            n = int(settle_s / m.opt.timestep)
            for _ in range(n):
                mj.mj_step(m, d)
        rec_t = np.arange(np.ceil(t0 * fps), np.floor(t_end * fps) + 1) / fps
        robot_j = self.robot_joints()
        R = {"t": rec_t, "frame": rec_t * fps, "joint_names": robot_j, "q": [], "ctrl": [], "obj_names": self.rigid,
             "obj_T": [], "contacts": []}
        sim_t = t0
        dt = m.opt.timestep
        lat = float((self.scene.get("simulation") or {}).get("control_latency_s") or 0.0)
        max_pen = {}
        sat, steps = np.zeros(int(sum(a is not None for a in acts))), 0
        lim_viol = 0.0
        jlim = {j: m.jnt_range[m.joint(j).id] for j in robot_j if m.jnt_limited[m.joint(j).id]}
        fr_lo = m.actuator_forcerange[:, 0]
        fr_hi = m.actuator_forcerange[:, 1]
        ok = np.array([a is not None for a in acts])
        aidx = np.array([a for a in acts if a is not None], int)
        for k, tr in enumerate(rec_t):
            while sim_t < tr - 1e-9:
                tgt = np.array([np.interp(sim_t - lat, t_cmd, Q[:, i]) for i in range(len(names))])
                d.ctrl[aidx] = tgt[ok]
                mj.mj_step(m, d)
                sim_t += dt
                steps += 1
                af = d.actuator_force[aidx]
                sat += (af <= fr_lo[aidx] + 1e-6) | (af >= fr_hi[aidx] - 1e-6)
                for i in range(d.ncon):
                    c = d.contact[i]
                    if c.dist < 0:
                        o1 = self.body_owner.get(m.geom_bodyid[c.geom1], ("world",))
                        o2 = self.body_owner.get(m.geom_bodyid[c.geom2], ("world",))
                        key = "|".join(sorted([o1[0] + ":" + (o1[2] if o1[0] == "robot" else o1[-1]),
                                               o2[0] + ":" + (o2[2] if o2[0] == "robot" else o2[-1])]))
                        max_pen[key] = max(max_pen.get(key, 0.0), -c.dist)
            R["q"].append(self.q(robot_j))
            R["ctrl"].append(np.array([np.interp(tr, t_cmd, Q[:, i]) for i in range(len(names))]))
            R["obj_T"].append(np.array([self.object_pose(n) for n in self.rigid]) if self.rigid else np.zeros((0, 4, 4)))
            if record_contacts:
                R["contacts"].append([{**c, "pos": c["pos"].round(4).tolist()} for c in self.contacts()])
            for j, (lo, hi) in jlim.items():
                v = d.qpos[self.jadr[j]]
                lim_viol = max(lim_viol, lo - v, v - hi)
        R["q"] = np.asarray(R["q"])
        R["ctrl"] = np.asarray(R["ctrl"])
        R["obj_T"] = np.asarray(R["obj_T"])
        R["ctrl_joint_names"] = names
        satf = sat / max(steps, 1)
        R["metrics"] = self.metrics(R, max_pen, satf, lim_viol,
                                    {m.actuator(a).name: float(f) for a, f in zip(aidx, satf)})
        return R

    def metrics(self, R: dict, max_pen: dict, sat_frac, lim_viol: float, sat_by_act: dict | None = None) -> dict:
        names = R["joint_names"]
        cn = R["ctrl_joint_names"]
        common = [j for j in cn if j in names]
        track = np.abs(R["q"][:, [names.index(j) for j in common]] - R["ctrl"][:, [cn.index(j) for j in common]])
        def cat(k):
            a, b = k.split("|")
            ka, kb = a.split(":")[0], b.split(":")[0]
            return "|".join(sorted([ka if ka != "entity" else ("static" if a.split(":")[1] in self.static else "object"),
                                    kb if kb != "entity" else ("static" if b.split(":")[1] in self.static else "object")]))
        by_cat = {}
        for k, v in max_pen.items():
            c = cat(k)
            by_cat[c] = max(by_cat.get(c, 0.0), v)
        obj = {}
        for i, n in enumerate(R["obj_names"]):
            P = R["obj_T"][:, i, :3, 3]
            obj[n] = {"start": P[0].round(4).tolist(), "end": P[-1].round(4).tolist(),
                      "max_speed_mps": float(np.max(np.linalg.norm(np.diff(P, axis=0), axis=1)) * self.fps) if len(P) > 1 else 0.0,
                      "min_z": float(P[:, 2].min())}
        hold = {}
        for f, cs in enumerate(R["contacts"]):
            for c in cs:
                for x, y in ((c["a"], c["b"]), (c["b"], c["a"])):
                    if x[0] == "robot" and y[0] == "entity" and y[1] not in self.static:
                        hold.setdefault(y[1], set()).add(f)
        return {"frames": len(R["t"]), "max_penetration_mm": {k: round(v * 1000, 2) for k, v in by_cat.items()},
                "max_penetration_pairs_mm": {k: round(v * 1000, 2) for k, v in sorted(max_pen.items(), key=lambda kv: -kv[1])[:12]},
                "tracking_err_rad": {"median": float(np.median(track)) if track.size else 0.0,
                                     "p99": float(np.percentile(track, 99)) if track.size else 0.0},
                "actuator_saturation_frac": float(np.max(sat_frac)) if np.size(sat_frac) else 0.0,
                "actuator_saturation_top": dict(sorted((sat_by_act or {}).items(), key=lambda kv: -kv[1])[:6]),
                "actuator_saturation_all": {k: round(v, 5) for k, v in (sat_by_act or {}).items() if v > 0},
                "joint_limit_violation_rad": float(max(lim_viol, 0.0)),
                "objects": obj,
                "robot_contact_frames": {k: [int(min(v)), int(max(v)), len(v)] for k, v in hold.items()}}

    def audit(self) -> dict:
        """Model checks for 'physically honest' execution: no welds/connects except mimic joint couplings, no
        mocap bodies, no applied external forces, objects have free joints, robots are actuated."""
        m = self.m
        eq = [int(m.eq_type[i]) for i in range(m.neq)]
        bad_eq = [i for i, t in enumerate(eq) if t != int(self.mj.mjtEq.mjEQ_JOINT)]
        return {"equality_non_joint": len(bad_eq), "mocap_bodies": int(m.nmocap),
                "xfrc_applied_nonzero": bool(np.any(self.d.xfrc_applied)), "actuators": int(m.nu),
                "free_objects": len(self.rigid), "ok": not bad_eq and m.nmocap == 0 and not np.any(self.d.xfrc_applied)}


def save_achieved(project, scene: dict, R: dict, motion_id: str, robot: str, derived_from: str | None,
                  note: str = "") -> str:
    """Write a rollout as an achieved motion (scene/motions/<id>.npz) and register it in the Scene IR (not saved:
    the caller saves the scene)."""
    import json
    from .scene_ir import save_motion
    rel = f"motions/{motion_id}.npz"
    save_motion(Path(project) / "scene" / rel, R["t"], R["frame"], R["joint_names"], R["q"], ctrl=R["ctrl"],
                obj_names=R["obj_names"], obj_T=R["obj_T"])
    (Path(project) / "scene" / f"motions/{motion_id}.metrics.json").write_text(json.dumps(R["metrics"], indent=1))
    scene.setdefault("motions", {})[motion_id] = {
        "robot": robot, "kind": "achieved", "uri": rel, "rate_hz": float(1.0 / np.median(np.diff(R["t"]))) if len(R["t"]) > 1 else None,
        "derived_from": derived_from, "provenance": {"q": f"MuJoCo rollout of {derived_from} {note}".strip()}}
    return rel
