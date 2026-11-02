"""Robot kinematics for agent-written code (CPU; MuJoCo): FK, Jacobians, IK and trajectory IK for any RobotBundle.

    from any2sim import robokit as rk
    kin = rk.Kin("assets:my_robot")                     # or a bundle dir, or rk.Kin.from_scene(project, "robot")
    T = kin.fk(q)["right_palm_link"]                       # q: {joint: value} (missing joints: init pose)
    q, info = kin.ik({"right_palm_link": T_target}, q0=q, joints=kin.group("right_arm"))
    Q, info = kin.ik_traj({"right_palm_link": Ts}, joints=kin.group("right_arm"), smooth=1e-2)

Frames: everything in the robot base frame unless T_world_base is given to Kin (then world). Joint limits are
always respected (clipped projected steps). Positions in metres, angles in radians.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _bundle(spec):
    from .robot.bundle import Bundle
    from .scene_ir import resolve_bundle
    if hasattr(spec, "profile"):
        return spec
    return Bundle(resolve_bundle(Path("."), str(spec)))


class Kin:
    """Kinematic model of a bundle (no contacts; visual geoms dropped)."""

    def __init__(self, bundle, T_world_base=None):
        import mujoco
        from .robot import mjcf
        self.mj = mujoco
        self.bundle = _bundle(bundle)
        fr = mjcf.robot_fragment(self.bundle, with_visuals=False, T_world_base=T_world_base)
        xml = (f'<mujoco>{mjcf.compiler_xml()}<asset>{"".join(fr["assets"])}</asset><worldbody>{fr["body"]}</worldbody>'
               + (f'<equality>{"".join(fr["equality"])}</equality>' if fr["equality"] else "") + "</mujoco>")
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.joints = [self.m.joint(j).name for j in range(self.m.njnt)]
        self.qadr = {j: int(self.m.jnt_qposadr[self.m.joint(j).id]) for j in self.joints}
        self.dadr = {j: int(self.m.jnt_dofadr[self.m.joint(j).id]) for j in self.joints}
        self.lim = {j: tuple(self.m.jnt_range[self.m.joint(j).id]) if self.m.jnt_limited[self.m.joint(j).id]
                    else (-np.inf, np.inf) for j in self.joints}
        self.q0 = {j: 0.0 for j in self.joints}
        self.q0.update(self.bundle.init_q())

    @classmethod
    def from_scene(cls, project, rid: str):
        from .scene_ir import load, robot_bundle
        sc = load(Path(project) / "scene")
        return cls(robot_bundle(Path(project) / "scene", sc, rid), np.asarray(sc["robots"][rid]["T_world_base"], float))

    def group(self, name: str) -> list[str]:
        return list(self.bundle.profile["groups"][name]["joints"])

    def limits(self, joints) -> np.ndarray:
        return np.array([self.lim[j] for j in joints], float)

    def set(self, q: dict | None = None):
        qq = dict(self.q0)
        qq.update(q or {})
        for j, v in qq.items():
            if j in self.qadr:
                self.d.qpos[self.qadr[j]] = v
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_comPos(self.m, self.d)

    def pose(self, link: str) -> np.ndarray:
        b = self.m.body(link).id
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[b].reshape(3, 3)
        T[:3, 3] = self.d.xpos[b]
        return T

    def fk(self, q: dict | None = None, links: list | None = None) -> dict:
        self.set(q)
        names = links or [self.m.body(b).name for b in range(1, self.m.nbody)]
        return {n: self.pose(n) for n in names}

    def point(self, link: str, offset=(0, 0, 0)) -> np.ndarray:
        T = self.pose(link)
        return T[:3, :3] @ np.asarray(offset, float) + T[:3, 3]

    def jac(self, link: str, joints: list, offset=(0, 0, 0)) -> tuple[np.ndarray, np.ndarray]:
        """(Jp 3xn, Jr 3xn) of a link point w.r.t. the given joints at the current configuration."""
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        self.mj.mj_jac(self.m, self.d, jp, jr, self.point(link, offset), self.m.body(link).id)
        cols = [self.dadr[j] for j in joints]
        return jp[:, cols], jr[:, cols]

    def ik(self, targets: dict, q0: dict | None = None, joints: list | None = None, iters: int = 100,
           pos_w: float = 1.0, rot_w: float = 0.3, damping: float = 1e-3, tol: float = 1e-5,
           q_rest: dict | None = None, rest_w: float = 0.0) -> tuple[dict, dict]:
        """Damped least squares with joint limits. targets: {link: 4x4 | {"T": 4x4, "pos_w", "rot_w", "offset"}
        | {"p": [x,y,z], "offset"}} (position-only); a dict target may name its body with "link" (then the key is
        free, e.g. several points on one link). Returns (q, {pos_err_m, rot_err_rad per link, iters})."""
        joints = joints or [j for j in self.joints]
        q = dict(self.q0)
        q.update(q0 or {})
        lim = self.limits(joints)
        x = np.array([q[j] for j in joints])
        xr = np.array([(q_rest or q)[j] for j in joints])
        specs = []
        for key, t in targets.items():
            if isinstance(t, dict):
                specs.append((t.get("link", key), t.get("T"), t.get("p"), t.get("offset", (0, 0, 0)),
                              t.get("pos_w", pos_w), t.get("rot_w", rot_w)))
            else:
                specs.append((key, np.asarray(t, float), None, (0, 0, 0), pos_w, rot_w))
        info = {}
        for it in range(iters):
            q.update(dict(zip(joints, x)))
            self.set(q)
            J, e = [], []
            for link, T, p, off, pw, rw in specs:
                cur = self.pose(link)
                pc = cur[:3, :3] @ np.asarray(off, float) + cur[:3, 3]
                jp, jr = self.jac(link, joints, off)
                pt = np.asarray(T[:3, 3] if T is not None else p, float)
                if T is not None:
                    pt = T[:3, :3] @ np.asarray(off, float) + T[:3, 3]
                e.append(pw * (pt - pc))
                J.append(pw * jp)
                if T is not None and rw > 0:
                    Re = T[:3, :3] @ cur[:3, :3].T
                    from scipy.spatial.transform import Rotation as Rot
                    e.append(rw * Rot.from_matrix(Re).as_rotvec())
                    J.append(rw * jr)
            if rest_w > 0:
                e.append(rest_w * (xr - x))
                J.append(rest_w * np.eye(len(x)))
            J, e = np.vstack(J), np.concatenate(e)
            if np.linalg.norm(e) < tol:
                break
            dx = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(len(e)), e)
            x = np.clip(x + dx, lim[:, 0], lim[:, 1])
        q.update(dict(zip(joints, x)))
        self.set(q)
        for key, (link, T, p, off, pw, rw) in zip(targets, specs):
            cur = self.pose(link)
            pc = cur[:3, :3] @ np.asarray(off, float) + cur[:3, 3]
            pt = (T[:3, :3] @ np.asarray(off, float) + T[:3, 3]) if T is not None else np.asarray(p, float)
            r = {"pos_err_m": float(np.linalg.norm(pt - pc))}
            if T is not None:
                r["rot_err_rad"] = float(np.arccos(np.clip((np.trace(T[:3, :3] @ cur[:3, :3].T) - 1) / 2, -1, 1)))
            info[key] = r
        info["iters"] = it + 1
        return q, info

    def ik_traj(self, targets: dict, joints: list, q0: dict | None = None, smooth: float = 0.0, **kw) -> tuple[np.ndarray, list]:
        """IK along a trajectory: targets {link: (T,4,4) | {"T": (T,4,4), ...} | {"p": (T,3), ...}}, warm-started
        from the previous solution; smooth > 0 adds a pull towards the previous frame's solution (rest_w).
        Returns (Q (T, len(joints)), per-frame info)."""
        n = len(next(iter(v if not isinstance(v, dict) else (v.get("T") if v.get("T") is not None else v["p"])
                           for v in targets.values())))
        q = dict(self.q0)
        q.update(q0 or {})
        Q, infos = [], []
        for t in range(n):
            tt = {}
            for link, v in targets.items():
                if isinstance(v, dict):
                    tt[link] = {k: (vv[t] if k in ("T", "p") else vv) for k, vv in v.items() if vv is not None}
                else:
                    tt[link] = v[t]
            q, info = self.ik(tt, q0=q, joints=joints, q_rest=dict(q) if smooth > 0 else None, rest_w=smooth, **kw)
            Q.append([q[j] for j in joints])
            infos.append(info)
        return np.asarray(Q), infos

    def palm_points(self, links_offsets: list) -> np.ndarray:
        return np.array([self.point(l, o) for l, o in links_offsets])


def finite_diff(Q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Velocities and accelerations of a (T, n) joint trajectory sampled at dt."""
    v = np.gradient(Q, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    return v, a
