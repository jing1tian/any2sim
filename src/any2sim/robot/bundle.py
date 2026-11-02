"""RobotBundle: a self-contained, verified robot description (assets/robots/<name>/ or projects/<id>/robot/).

<bundle>/
  robot.yaml          profile (this schema), the source of truth together with robot.urdf
  robot.urdf          kinematics, inertials; visual/collision geometry (paths relative to the bundle)
  robot.xml           compiled MJCF (robot only; regenerate with `compile_mjcf`)
  meshes/             source meshes (kept as supplied)
  visual/<link>/      render visuals per link: <link>.glb (materials, normal maps) + part_<k>.obj/.json (MuJoCo)
  collision/          collision meshes per link (when the URDF ones are replaced)
  checks/             verification evidence (report.yaml, overlays, FK residuals)

robot.yaml:
  schema_version: "0.1"
  name: <text>
  urdf: robot.urdf
  base_link: <link>                       # the robot's mounting frame (T_world_base places it)
  frames: {conventions text}              # e.g. "x forward, y left, z up; palm +z along the fingers"
  groups:                                  # kinematic groups (arms, hands, grippers, heads)
    <group>: {kind: arm|hand|gripper|head|torso, joints: [...], tip_link: <link>, side: left|right|null,
              tcp: {link, T: 4x4}}         # optional tool frame
  init_pose: {<joint>: value}              # the real robot's start/rest configuration
  named_poses: {<name>: {<joint>: value}}
  actuators:                               # position servos (what the real controller does)
    default: {kp, kv, armature, damping, frictionloss}
    <group>: {kp, kv, ...}                 # per-group overrides; forcerange = URDF effort, ctrlrange = limits
  controller: {gravity_compensation: true, rate_hz: <float>}
  contact: {<group>: {friction: [slide, spin, roll], condim, solref, solimp}}
  collision_exclude: [[linkA, linkB], ...]
  visuals: {<link>: {glb: visual/<link>/<link>.glb, parts: visual/<link>/parts.json, source: <text>}}
  collisions: {<link>: [collision/<file>.obj, ...]}
  appearance: {<link pattern>: {base_color: [r,g,b] (linear), roughness, metallic}}   # for STL-only visuals
  cameras:                                 # robot-mounted cameras (OpenCV optical frame)
    <name>: {link: <link>, T_link_cam: 4x4, K: 3x3, width, height, distortion: {model, coeffs}, provenance}
  mount: {T_world_base: 4x4 | null, notes, provenance}            # how the robot stands (e.g. on the table edge)
  data_mapping: {<source>: {<column>: {joints: [...], slice: [a, b], kind: joint_pos|ee_pose, ...}}}
  retarget: {<group>: {human_model: mano21, keypoint_links: {<mano index>: <link or link+offset>}, T_palm_wrist: 4x4}}
  provenance: {<field>: observed | fitted | prior | supplied (+ text)}
  checks: {<check>: {status: pass|fail|not_run, metric..., evidence: checks/...}}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import urdf as U


class Bundle:
    def __init__(self, root: str | Path):
        import yaml
        self.root = Path(root).resolve()
        self.profile = yaml.safe_load((self.root / "robot.yaml").read_text())
        self.model = U.load(self.root / self.profile.get("urdf", "robot.urdf"))

    @property
    def name(self) -> str:
        return self.profile["name"]

    def save_profile(self):
        from ..scene_ir import _plain
        import yaml
        (self.root / "robot.yaml").write_text(yaml.safe_dump(_plain(self.profile), sort_keys=False, allow_unicode=True))

    def joints(self, group: str | None = None) -> list[str]:
        if group:
            return list(self.profile["groups"][group]["joints"])
        return U.movable_joints(self.model)

    def limits(self, joints=None) -> np.ndarray:
        joints = joints or self.joints()
        lim = []
        for j in joints:
            L = self.model["joints"][j].get("limit") or {}
            lim.append([L.get("lower", -np.pi), L.get("upper", np.pi)])
        return np.asarray(lim, float)

    def init_q(self) -> dict:
        q = {j: 0.0 for j in self.joints()}
        q.update(self.profile.get("init_pose") or {})
        return q

    def fk(self, q: dict | None = None, T_world_base=None) -> dict:
        """World pose of every link (numpy FK on the URDF)."""
        return U.fk(self.model, q if q is not None else self.init_q(), T_world_base)

    def camera(self, name: str, q: dict | None = None, T_world_base=None) -> dict:
        """{K, width, height, T_world_cam} of a robot-mounted camera at configuration q."""
        c = self.profile["cameras"][name]
        T = self.fk(q, T_world_base)[c["link"]] @ np.asarray(c["T_link_cam"], float)
        return {"K": np.asarray(c["K"], float), "width": c["width"], "height": c["height"], "T_world_cam": T}

    def path(self, rel: str) -> Path:
        return self.root / rel


def validate(root: str | Path) -> dict:
    """Structural checks of a bundle: files exist, groups name real joints, init pose within limits, visuals and
    collisions present, cameras well formed. {"ok", "errors", "warnings"}."""
    err, warn = [], []
    try:
        b = Bundle(root)
    except Exception as e:                                           # noqa: BLE001
        return {"ok": False, "errors": [f"cannot load: {e}"], "warnings": []}
    P, M = b.profile, b.model
    for k in ("schema_version", "name", "urdf", "base_link", "groups"):
        if k not in P:
            err.append(f"robot.yaml: missing {k}")
    if P.get("base_link") not in M["links"]:
        err.append(f"base_link {P.get('base_link')!r} not a URDF link")
    for g, G in (P.get("groups") or {}).items():
        for j in G.get("joints", []):
            if j not in M["joints"]:
                err.append(f"group {g}: joint {j} not in URDF")
        if G.get("tip_link") and G["tip_link"] not in M["links"]:
            err.append(f"group {g}: tip_link {G['tip_link']} not in URDF")
    lim = dict(zip(b.joints(), b.limits()))
    for j, v in (P.get("init_pose") or {}).items():
        if j not in lim:
            err.append(f"init_pose: {j} is not a movable joint")
        elif not (lim[j][0] - 1e-3 <= v <= lim[j][1] + 1e-3):
            warn.append(f"init_pose: {j}={v:.4f} outside [{lim[j][0]:.4f}, {lim[j][1]:.4f}]")
    for ln, l in M["links"].items():
        for g in l["visuals"] + l["collisions"]:
            if g["type"] == "mesh" and not Path(g["uri"]).exists():
                err.append(f"{ln}: mesh {g['uri']} missing")
    for ln, v in (P.get("visuals") or {}).items():
        if ln not in M["links"]:
            err.append(f"visuals: {ln} not a link")
        for k in ("glb", "parts"):
            if v.get(k) and not (b.root / v[k]).exists():
                err.append(f"visuals.{ln}: {v[k]} missing")
    for ln, fs in (P.get("collisions") or {}).items():
        for f in fs:
            if not (b.root / f).exists():
                err.append(f"collisions.{ln}: {f} missing")
    for cn, c in (P.get("cameras") or {}).items():
        if c.get("link") not in M["links"]:
            err.append(f"camera {cn}: link {c.get('link')} not in URDF")
        for k in ("T_link_cam", "K", "width", "height"):
            if k not in c:
                err.append(f"camera {cn}: missing {k}")
    for k, c in (P.get("checks") or {}).items():
        if c.get("status") == "fail":
            warn.append(f"check {k} failed: {c.get('note', '')}")
    if not (b.root / "robot.xml").exists():
        warn.append("robot.xml not compiled")
    return {"ok": not err, "errors": err, "warnings": warn}
