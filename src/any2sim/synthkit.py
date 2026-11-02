"""Synthesis kit: diversify one physically executed demonstration into many randomised episodes (synthesis stage).

    from any2sim import synthkit as sy
    ctx = sy.Context(project)                         # base scene + synthesis/spec.yaml + the demo it names
    ctx.demo.events                                   # {side, contact: [a, b], lift, place, release, settle} (frames)
    var = sy.sample_variant(ctx, attempt=7)           # deterministic for (spec seed, attempt)
    cmd, info = sy.transfer(ctx.demo, var)            # object-centric re-anchoring + IK -> commanded joint targets
    scene_v = sy.materialize(ctx, var, episode_dir)   # scene_ir.apply_variant + the episode's motions
    ok, det = sy.success(ctx, scene_v, R)             # task predicate on a rollout (simkit.Sim(...).rollout)

Trajectory transfer (MimicGen-style, per active hand): the demo's palm path P(t) is split at its events. Before the
approach it is unchanged (the robot starts where the demo starts); from the pre-grasp to the lift it follows the
manipulated object (P' = T_obj' . grasp_delta . T_obj^-1 . P); after transport it follows the target (P' = T_tgt' .
place_delta . T_tgt^-1 . P); in between the two candidate palm poses are blended (minimum-jerk, slerp), plus an
optional smooth via-point offset. Fingers keep the demo's joint targets (the grasp is defined relative to the
object). The arm joints come from robokit IK; everything else of the demo is kept. Physics decides: every episode is
rolled out with the robot's own actuators and kept only if the task succeeds and contacts/limits are plausible.

spec (synthesis/spec.yaml) - ranges, never sampled values:
  seed: 0
  demo: {commanded: <motion id>, achieved: <motion id>}
  task: {object: <entity>, target: <entity> | null, side: left | right | null (auto),
         success: {type: in_container | at_pose, margin_m: 0.0, settle_s: 0.5, released: true,
                   max_end_speed_mps: 0.03, max_target_tilt_deg: 30, (at_pose: relative_to, T, pos_tol_m, rot_tol_deg)},
         instruction: {instruction: <text>, detailed_instruction: ..., task: <name>, ...}}
  spatial:
    objects: {<entity>: {dx: [lo, hi], dy: [lo, hi], yaw_deg: [lo, hi]}}     # world axes, about the entity origin
    min_separation_m: 0.10          # between every pair of rigid entities (xy, origins)
    support: <static entity> | null # origins must stay on its top footprint (shrunk by support_margin_m)
    support_margin_m: 0.03
    in_view_margin_px: 40           # origins project inside the source camera image (first pose)
  camera: {source_cam: {rot_deg: 0.5, trans_m: 0.005}, robot: {<bundle camera>: {rot_deg, trans_m}}}
  lighting: {strength_scale: [0.6, 1.5], world_strength_scale: [0.5, 2.0], color_temp_k: [3500, 7500],
             position_jitter_m: 0.2, size_scale: [0.7, 1.4]}
  textures: {<entity>: {p_keep: 0.3, hsv_shift: {h_deg: 8, s: 0.15, v: 0.2},
                        procedural: {types: [noise, wave, voronoi, checker, brick], hue: [0, 360], sat: [0, 0.4],
                                     val: [0.15, 0.8], contrast: [0.05, 0.3], scale: [4, 40]},
                        images: [<project-relative png>, ...], roughness: [0.3, 0.9]}}
  physics: {objects: {<entity>: {mass_scale: [0.7, 1.3], friction_scale: [0.7, 1.3]}},
            actuators: {<group | "*">: {kp_scale: [0.85, 1.15], kv_scale: [0.85, 1.15]}},
            control_latency_s: [0.0, 0.03]}
  sensor: {exposure_ev: [-0.4, 0.4], wb_gain: 0.06, noise_std: [0.0, 0.015], blur_px: [0.0, 0.8]}
                                    # on top of the camera model (Scene IR cameras.<cam>.isp, camera_calibrate)
  dressing:                         # render-only clutter: never touches physics (MuJoCo/USD ignore it)
    surface: table                  # static entity whose top gets cables and grime
    margin_m: 0.06                  # keep-out around the robot's swept palm paths and the rigid objects
    edge_margin_m: 0.06             # stay this far inside the surface's footprint
    cables: {count: [0, 3], radius_m: [0.002, 0.005], length_m: [0.3, 0.9],
             colors: [[0.01, 0.01, 0.01], [0.6, 0.6, 0.6], [0.02, 0.02, 0.15]]}
    grime: {p: 0.8, amount: [0.1, 0.45], scale: [6, 30]}       # stains / wear on the surface material
    shadow_casters: {count: [0, 2], size_m: [0.15, 0.45], height_m: [0.35, 0.8]}   # unseen occluders: soft shadows
  trajectory: {speed: [0.85, 1.15], grasp_offset_m: 0.008, grasp_rot_deg: 4, place_offset_m: 0.02,
               via_noise_m: 0.015, approach_lead_s: 0.5, tail_s: 1.0,   # tail: hold the last targets (settling)
               place_anchor: position | pose}   # position (default for in_container): the place follows the
                                                # target's position, not its yaw (containers); pose: both
  accept: {penetration_mm_max: {object_robot: 3, robot_static: 3, object_static: 2, robot_robot: 1},
           actuator_saturation_frac_max: 0.05,   # every actuator except fingers (fraction of steps at the effort limit)
           hand_saturation_frac_max: 0.5,        # finger actuators may sit at their effort limit while gripping
           joint_limit_violation_rad_max: 0.01,
           ik_pos_err_mm_max: 5, ik_rot_err_deg_max: 3, joint_speed_ratio_max: 1.0,
           robust_replay: true}           # the commands rounded to float32 (= the exported actions) must succeed too
Every axis is optional; a missing axis is not randomised. Values are absolute in the variant (variant.yaml), so an
episode is reproducible from (base scene, variant) alone.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

ACCEPT_DEFAULT = {"penetration_mm_max": {"object_robot": 3.0, "robot_static": 3.0, "object_static": 2.0,
                                         "robot_robot": 1.0},
                  "actuator_saturation_frac_max": 0.05, "hand_saturation_frac_max": 0.5,
                  "joint_limit_violation_rad_max": 0.01, "ik_pos_err_mm_max": 5.0, "ik_rot_err_deg_max": 3.0,
                  "joint_speed_ratio_max": 1.0, "robust_replay": True}


# ---------------------------------------------------------------- small geometry helpers
def min_jerk(u):
    u = np.clip(u, 0.0, 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def blend_pose(A, B, s):
    """Pose between A and B: position linear, rotation slerp (s in [0, 1])."""
    from scipy.spatial.transform import Rotation as R, Slerp
    if s <= 0:
        return np.asarray(A, float).copy()
    if s >= 1:
        return np.asarray(B, float).copy()
    T = np.eye(4)
    T[:3, 3] = (1 - s) * A[:3, 3] + s * B[:3, 3]
    T[:3, :3] = Slerp([0, 1], R.from_matrix([A[:3, :3], B[:3, :3]]))([s]).as_matrix()[0]
    return T


def small_transform(rng, trans_m: float, rot_deg: float) -> np.ndarray:
    """Random rigid transform: translation uniform in a ball of radius trans_m, rotation about a random axis."""
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    if trans_m:
        d = rng.normal(size=3)
        T[:3, 3] = d / (np.linalg.norm(d) + 1e-12) * trans_m * rng.uniform() ** (1 / 3)
    if rot_deg:
        a = rng.normal(size=3)
        T[:3, :3] = R.from_rotvec(a / (np.linalg.norm(a) + 1e-12) * np.radians(rot_deg) * rng.uniform()).as_matrix()
    return T


def planar_shift(T, dx, dy, yaw_deg):
    """T moved by (dx, dy) in world xy and rotated by yaw about the world z axis through its own origin."""
    from scipy.spatial.transform import Rotation as R
    T = np.asarray(T, float)
    out = T.copy()
    Rz = R.from_euler("z", yaw_deg, degrees=True).as_matrix()
    out[:3, :3] = Rz @ T[:3, :3]
    out[:3, 3] = T[:3, 3] + [dx, dy, 0.0]
    return out


def support_height_poses(scene, spatial, rng):
    """Translate a support and its explicitly listed contents together along world z.

    This changes robot/support relative height without changing the robot or camera.
    Poses are initial conditions only; the normal actuator/contact rollout follows.
    """
    from .scene_ir import pose_at
    config = spatial.get("support_height")
    if not config:
        return {}, None
    support = spatial.get("support")
    contents = config.get("contents", [])
    if not support or not isinstance(contents, list):
        raise ValueError("support_height requires spatial.support and a contents list")
    bounds = np.asarray(config["dz_m"], float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise ValueError("support_height.dz_m must be finite ordered [min, max]")
    names = list(dict.fromkeys([support, *contents]))
    if not set(spatial.get("objects", {})).issubset(names):
        raise ValueError("support_height.contents must include all randomized supported objects")
    dz = float(rng.uniform(*bounds))
    poses = {}
    for name in names:
        T = np.asarray(pose_at(scene["entities"][name], None), float).copy()
        T[2, 3] += dz
        poses[name] = T
    return poses, {"dz_m": dz, "entities": names, "axis": "world_z"}


def kelvin_rgb(k: float) -> list:
    """Approximate linear RGB of a black body at k kelvin, normalised to the colour at 6500 K (Tanner Helland fit)."""
    def srgb(t):
        t = t / 100.0
        r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
        g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
        c = np.clip([r, g, b], 0, 255) / 255.0
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    c = srgb(k) / srgb(6500.0)
    return (c / c.max()).round(4).tolist()


def _u(rng, rng_spec, default=None):
    if rng_spec is None:
        return default
    if isinstance(rng_spec, (int, float)):
        return float(rng_spec)
    return float(rng.uniform(rng_spec[0], rng_spec[1]))


# ---------------------------------------------------------------- the demonstration
class Demo:
    """The executed demonstration: commanded targets, achieved rollout, events, palm path of the active hand."""

    def __init__(self, project, scene: dict, commanded: str, achieved: str, obj: str, target: str | None = None,
                 side: str | None = None):
        import json
        from . import robokit as rk
        from .scene_ir import load_motion, pose_at, robot_bundle
        self.project = Path(project)
        self.scene = scene
        sd = self.project / "scene"
        self.cmd = load_motion(sd, scene, commanded)
        self.ach = load_motion(sd, scene, achieved)
        self.cmd_id, self.ach_id = commanded, achieved
        self.rid = scene["motions"][commanded]["robot"]
        self.bundle = robot_bundle(sd, scene, self.rid)
        self.T_world_base = np.asarray(scene["robots"][self.rid]["T_world_base"], float)
        self.kin = rk.Kin(self.bundle, self.T_world_base)
        self.obj, self.target = obj, target
        self.T_obj0 = pose_at(scene["entities"][obj], None)
        self.T_tgt0 = pose_at(scene["entities"][target], None) if target else None
        self.t = np.asarray(self.cmd["t"], float)
        self.dt = float(np.median(np.diff(self.t)))
        self.joint_names = list(self.cmd["joint_names"])
        self.Q = np.asarray(self.cmd["q"], float)
        mfile = sd / Path(scene["motions"][achieved]["uri"]).with_suffix(".metrics.json")
        self.ach_metrics = json.loads(mfile.read_text()) if mfile.exists() else {}
        self.events = self._events(side)
        self.side = self.events["side"]
        self.palm_link = self.bundle.profile["groups"][f"{self.side}_hand"].get("tip_link", f"{self.side}_palm_link")
        self.arm = list(self.bundle.profile["groups"][f"{self.side}_arm"]["joints"])
        self.palm = self.palm_path(self.Q)

    def q_dict(self, Q, k) -> dict:
        return dict(zip(self.joint_names, Q[k]))

    def palm_path(self, Q, link: str | None = None) -> np.ndarray:
        link = link or self.palm_link
        return np.array([self.kin.fk(self.q_dict(Q, k), [link])[link] for k in range(len(Q))])

    def _events(self, side) -> dict:
        """Frames (rows of the commanded motion) of contact, lift, place, release, settle from the achieved rollout."""
        a = self.ach
        names = list(a.get("obj_names", []))
        P = a["obj_T"][:, names.index(self.obj), :3, 3]
        fa = np.asarray(a["frame"], float)
        fc = np.asarray(self.cmd["frame"], float)
        row = lambda f: int(np.clip(np.searchsorted(fc, f), 0, len(fc) - 1))   # noqa: E731
        z0 = P[0, 2]
        moved = np.linalg.norm(P - P[0], axis=1) > 0.005
        lift_i = int(np.argmax(P[:, 2] > z0 + 0.01)) if np.any(P[:, 2] > z0 + 0.01) else int(np.argmax(moved))
        d_end = np.linalg.norm(P[:, :2] - P[-1, :2], axis=1)
        after = np.arange(len(P)) > lift_i
        place_i = int(np.argmax(after & (d_end < 0.02))) if np.any(after & (d_end < 0.02)) else len(P) - 1
        sp = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)] / max(float(np.median(np.diff(a["t"]))), 1e-6)
        settle_i = int(np.nonzero(sp > 0.02)[0].max()) if np.any(sp > 0.02) else len(P) - 1
        rc = (self.ach_metrics.get("robot_contact_frames") or {}).get(self.obj)
        if rc:
            c0, c1 = int(rc[0]), int(rc[1])
        else:                                                   # fallback: the object starts moving / stops
            c0, c1 = int(np.argmax(moved)), settle_i
        if side is None:
            dist = {}
            for s in ("left", "right"):
                g = self.bundle.profile["groups"].get(f"{s}_hand")
                if not g:
                    continue
                link = g.get("tip_link", f"{s}_palm_link")
                k = row(fa[c0])
                dist[s] = float(np.linalg.norm(self.kin.fk(self.q_dict(self.Q, k), [link])[link][:3, 3] - P[c0]))
            side = min(dist, key=dist.get)
        return {"side": side, "contact": [row(fa[c0]), row(fa[c1])], "lift": row(fa[lift_i]),
                "place": row(fa[place_i]), "release": row(fa[c1]), "settle": row(fa[settle_i]),
                "obj_start": P[0].round(4).tolist(), "obj_end": P[-1].round(4).tolist()}


# ---------------------------------------------------------------- context (base scene + spec + demo)
class Context:
    def __init__(self, project, spec: dict | str | None = None):
        import yaml
        from .scene_ir import core_digest, load
        self.project = Path(project)
        self.scene = load(self.project / "scene")
        if spec is None or isinstance(spec, (str, Path)):
            spec = yaml.safe_load((self.project / (spec or "synthesis/spec.yaml")).read_text())
        self.spec = spec
        task = spec["task"]
        dm = spec.get("demo") or {}
        cmd = dm.get("commanded") or next(k for k, m in self.scene["motions"].items() if m["kind"] == "commanded")
        ach = dm.get("achieved") or (self.scene.get("render") or {}).get("motion")
        self.demo = Demo(self.project, self.scene, cmd, ach, task["object"], task.get("target"), task.get("side"))
        self.core = core_digest(self.scene)
        self.accept = {**ACCEPT_DEFAULT, **(spec.get("accept") or {})}
        self.accept["penetration_mm_max"] = {**ACCEPT_DEFAULT["penetration_mm_max"],
                                             **((spec.get("accept") or {}).get("penetration_mm_max") or {})}

    def rigid(self) -> list:
        return [n for n, e in self.scene["entities"].items() if e.get("capability") == "rigid"]


# ---------------------------------------------------------------- variants
def _support_polygon(ctx, name, margin):
    """World xy polygon (convex hull) of a static entity's top footprint, shrunk by margin; z of its top."""
    import trimesh
    from scipy.spatial import ConvexHull
    from .scene_ir import pose_at
    e = ctx.scene["entities"][name]
    T = pose_at(e, None)
    pts = []
    for c in (e["geometry"].get("collision") or []):
        if c["type"] == "box":
            s = np.asarray(c["size"], float) / 2
            corners = np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])
            Tc = np.asarray(c.get("T", np.eye(4)), float)
            pts.append(corners @ Tc[:3, :3].T + Tc[:3, 3])
    if not pts:
        V = np.asarray(trimesh.load(ctx.project / "scene" / e["geometry"]["visual"]["obj_uri"], force="mesh").vertices)
        pts.append(V)
    P = np.vstack(pts) @ T[:3, :3].T + T[:3, 3]
    hull = ConvexHull(P[:, :2])
    poly = P[hull.vertices, :2]
    c = poly.mean(0)
    d = poly - c
    poly = c + d * np.clip(1 - margin / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9), 0, 1)
    return poly, float(P[:, 2].max())


def _in_poly(p, poly):
    from matplotlib.path import Path as MPath
    return bool(MPath(poly).contains_point(p))


def sample_variant(ctx: Context, attempt: int, nominal: bool = False) -> dict:
    """One variant (absolute values) for attempt index `attempt`; nominal: no randomisation at all.
    Spatial constraints are met by rejection inside the sampler (up to 200 draws); the result records the draws."""
    from .geom3d import inv, project as proj
    from .scene_ir import camera_poses, pose_at, robot_bundle, render_head_camera
    sp = ctx.spec
    seed = int(sp.get("seed", 0))
    rng = np.random.default_rng([seed, int(attempt) + 1])          # attempt -1 = nominal
    var = {"id": "nominal" if nominal else f"ep_{attempt:05d}", "attempt": int(attempt), "seed": [seed, int(attempt) + 1],
           "base_core_sha256": ctx.core, "nominal": bool(nominal), "entities": {}}
    sc = ctx.scene
    if nominal:
        var["trajectory"] = {"speed": 1.0, "grasp_delta": np.eye(4).tolist(), "place_delta": np.eye(4).tolist(),
                             "via_offset": [0.0, 0.0, 0.0],
                             "approach_lead_s": float((sp.get("trajectory") or {}).get("approach_lead_s", 0.5))}
        var["sensor"] = {}
        return var
    # spatial
    S = sp.get("spatial") or {}
    objs = S.get("objects") or {}
    head_camera = render_head_camera(sc)
    cam = sc["cameras"][head_camera]
    K = np.asarray(cam["K"], float)
    Tc = camera_poses(ctx.project / "scene", sc, head_camera)[0]
    W, H = cam["width"], cam["height"]
    support = None
    if S.get("support"):
        support = _support_polygon(ctx, S["support"], float(S.get("support_margin_m", 0.03)))
    rigid = ctx.rigid()
    base = {n: pose_at(sc["entities"][n], None) for n in rigid}
    raised, height = support_height_poses(sc, S, rng)
    if height is not None:
        var["support_height"] = height
        for name, T in raised.items():
            var["entities"].setdefault(name, {})["T_world_obj"] = T.tolist()
            if name in base:
                base[name] = T
    draws = 0
    for draws in range(1, 201):
        new = dict(base)
        for n, r in objs.items():
            new[n] = planar_shift(base[n], _u(rng, r.get("dx"), 0.0), _u(rng, r.get("dy"), 0.0),
                                  _u(rng, r.get("yaw_deg"), 0.0))
        ok = True
        ms = float(S.get("min_separation_m", 0.0))
        names = list(new)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if np.linalg.norm(new[names[i]][:2, 3] - new[names[j]][:2, 3]) < ms:
                    ok = False
        if ok and support is not None:
            ok = all(_in_poly(new[n][:2, 3], support[0]) for n in objs)
        if ok and S.get("in_view_margin_px") is not None:
            m = float(S["in_view_margin_px"])
            for n in objs:
                uv, z = proj(K, inv(Tc), new[n][:3, 3][None])
                if not (z[0] > 0 and m <= uv[0, 0] <= W - m and m <= uv[0, 1] <= H - m):
                    ok = False
        if ok:
            break
    else:
        var["invalid"] = "spatial constraints not met in 200 draws"
    var["spatial_draws"] = draws
    for n in objs:
        var["entities"].setdefault(n, {})["T_world_obj"] = new[n].tolist()
    # textures / materials (render-only except base_color for MuJoCo previews)
    for n, r in (sp.get("textures") or {}).items():
        mat = {}
        if rng.uniform() >= float(r.get("p_keep", 0.0)):
            opts = [k for k in ("hsv_shift", "procedural", "images") if r.get(k)]
            kind = opts[int(rng.integers(len(opts)))] if opts else None
            if kind == "hsv_shift":
                h = r["hsv_shift"]
                mat["hsv_shift"] = [float(rng.uniform(-1, 1) * h.get("h_deg", 0)), float(rng.uniform(-1, 1) * h.get("s", 0)),
                                    float(rng.uniform(-1, 1) * h.get("v", 0))]
            elif kind == "procedural":
                p = r["procedural"]
                import colorsys
                hue = _u(rng, p.get("hue", [0, 360])) / 360.0
                sat, val = _u(rng, p.get("sat", [0, 0.4])), _u(rng, p.get("val", [0.15, 0.8]))
                con = _u(rng, p.get("contrast", [0.05, 0.3]))
                c1 = np.array(colorsys.hsv_to_rgb(hue, sat, np.clip(val - con / 2, 0, 1))) ** 2.2
                c2 = np.array(colorsys.hsv_to_rgb(hue, sat, np.clip(val + con / 2, 0, 1))) ** 2.2
                types = p.get("types") or ["noise"]
                mat["procedural"] = {"type": types[int(rng.integers(len(types)))], "colors": [c1.round(4).tolist(), c2.round(4).tolist()],
                                     "scale": _u(rng, p.get("scale", [4, 40]))}
                mat["base_color"] = ((c1 + c2) / 2).round(4).tolist()
                mat["base_color_texture"] = None
            elif kind == "images":
                im = r["images"][int(rng.integers(len(r["images"])))]
                import os
                mat["base_color_texture"] = os.path.relpath(ctx.project / im, ctx.project / "scene")
                mat["procedural"] = None
        if r.get("roughness"):
            mat["roughness"] = _u(rng, r["roughness"])
        base_mat = sc["entities"][n].get("material") or {}
        for field in ("normal_strength", "roughness_map"):
            if r.get(field + "_scale"):
                key = field if field == "normal_strength" else "roughness_map_scale"
                mat[key] = float(base_mat.get(key, 1.0)) * _u(rng, r[field + "_scale"])
        if mat:
            var["entities"].setdefault(n, {})["material"] = mat
    # lighting
    L = sp.get("lighting")
    if L:
        lights = copy.deepcopy(sc.get("lights") or [])
        temp = _u(rng, L.get("color_temp_k"))
        for li in lights:
            if li["type"] == "world":
                li["strength"] = float(li.get("strength", 0.3)) * _u(rng, L.get("world_strength_scale"), 1.0)
                continue
            li["strength_w"] = float(li.get("strength_w", 100.0)) * _u(rng, L.get("strength_scale"), 1.0)
            if temp:
                li["color"] = kelvin_rgb(temp)
            j = float(L.get("position_jitter_m", 0.0))
            if j and "position" in li:
                li["position"] = (np.asarray(li["position"], float) + small_transform(rng, j, 0)[:3, 3]).tolist()
            if L.get("size_scale") and "size" in li:
                s = _u(rng, L["size_scale"])
                li["size"] = [float(x) * s for x in li["size"]]
        var["lights"] = lights
        var["light_params"] = {"color_temp_k": temp}
    # cameras
    C = sp.get("camera") or {}
    for cn, config in C.items():
        if cn in sc["cameras"]:
            var.setdefault("cameras", {})[cn] = {"T_delta": small_transform(
                rng, float(config.get("trans_m", 0)), float(config.get("rot_deg", 0))).tolist()}
    rid = ctx.demo.rid
    rv = {}
    if C.get("robot"):
        rv["camera_deltas"] = {cn: small_transform(rng, float(c.get("trans_m", 0)), float(c.get("rot_deg", 0))).tolist()
                               for cn, c in C["robot"].items()}
    # physics
    Ph = sp.get("physics") or {}
    for n, r in (Ph.get("objects") or {}).items():
        from .scene_ir import phys
        ph0 = sc["entities"][n].get("physics") or {}
        ph = {}
        if r.get("mass_scale"):
            ph["mass_kg"] = float(phys(ph0.get("mass_kg"), 0.1)) * _u(rng, r["mass_scale"])
        if r.get("friction_scale"):
            ph["friction"] = float(phys(ph0.get("friction"), 0.8)) * _u(rng, r["friction_scale"])
        if ph:
            var["entities"].setdefault(n, {})["physics"] = ph
    if Ph.get("actuators"):
        prof = robot_bundle(ctx.project / "scene", sc, rid).profile
        acts = prof.get("actuators") or {}
        ov = {}
        for g in [k for k in prof["groups"]]:
            r = Ph["actuators"].get(g) or Ph["actuators"].get("*")
            if not r:
                continue
            a0 = {**(acts.get("default") or {}), **(acts.get(g) or {})}
            ov[g] = {"kp": float(a0.get("kp", 200.0)) * _u(rng, r.get("kp_scale"), 1.0),
                     "kv": float(a0.get("kv", 10.0)) * _u(rng, r.get("kv_scale"), 1.0)}
        rv["overrides"] = {"actuators": ov}
    if rv:
        var["robots"] = {rid: rv}
    if Ph.get("control_latency_s") is not None:
        var["simulation"] = {"control_latency_s": _u(rng, Ph["control_latency_s"])}
    # dressing (render-only): cables on the surface outside the robot's workspace, grime, unseen shadow casters
    Dr = sp.get("dressing") or {}
    if Dr.get("surface"):
        var["dressing"] = sample_dressing(ctx, Dr, rng, var)
        gr = Dr.get("grime")
        if gr and rng.uniform() < float(gr.get("p", 1.0)):
            m_ = var["entities"].setdefault(Dr["surface"], {}).setdefault("material", {})
            m_["grime"] = {"amount": _u(rng, gr.get("amount"), 0.25), "scale": _u(rng, gr.get("scale"), 15.0),
                           "seed": float(rng.uniform(0, 100))}
    # sensor (image space, applied by synth_render)
    Se = sp.get("sensor") or {}
    var["sensor"] = {"exposure_ev": _u(rng, Se.get("exposure_ev"), 0.0),
                     "wb_gains": (1 + rng.uniform(-1, 1, 3) * float(Se.get("wb_gain", 0.0))).round(4).tolist(),
                     "noise_std": _u(rng, Se.get("noise_std"), 0.0), "blur_px": _u(rng, Se.get("blur_px"), 0.0),
                     "noise_seed": int(rng.integers(2 ** 31))}
    # trajectory
    Tr = sp.get("trajectory") or {}
    via = rng.normal(size=3)
    via = via / (np.linalg.norm(via) + 1e-12) * float(Tr.get("via_noise_m", 0.0)) * rng.uniform()
    var["trajectory"] = {"speed": _u(rng, Tr.get("speed"), 1.0),
                         "grasp_delta": small_transform(rng, float(Tr.get("grasp_offset_m", 0.0)),
                                                        float(Tr.get("grasp_rot_deg", 0.0))).tolist(),
                         "place_delta": small_transform(rng, float(Tr.get("place_offset_m", 0.0)), 0.0).tolist(),
                         "via_offset": via.round(5).tolist(), "approach_lead_s": float(Tr.get("approach_lead_s", 0.5)),
                         "tail_s": float(Tr.get("tail_s", 1.0)),
                         "place_anchor": Tr.get("place_anchor") or ("position" if ((sp["task"].get("success") or {})
                                                                                   .get("type", "in_container") == "in_container") else "pose")}
    return var


def keep_out_points(ctx, var) -> np.ndarray:
    """xy points the robot sweeps (both palms over the demo, re-anchored for this variant when possible) and the
    rigid objects' start positions: dressing stays away from them."""
    pts = []
    try:
        T, _ = palm_targets(ctx.demo, var) if not var.get("nominal") else (ctx.demo.palm, None)
        pts.append(T[:, :2, 3])
    except Exception:                                             # noqa: BLE001
        pts.append(ctx.demo.palm[:, :2, 3])
    other = "left" if ctx.demo.side == "right" else "right"
    g = ctx.demo.bundle.profile["groups"].get(f"{other}_hand")
    if g:
        link = g.get("tip_link", f"{other}_palm_link")
        pts.append(ctx.demo.palm_path(ctx.demo.Q, link)[:, :2, 3])
    from .scene_ir import pose_at
    for n in ctx.rigid():
        T = (var.get("entities") or {}).get(n, {}).get("T_world_obj")
        pts.append(np.asarray(T if T is not None else pose_at(ctx.scene["entities"][n], None), float)[None, :2, 3])
    return np.vstack(pts)


def sample_dressing(ctx, Dr: dict, rng, var) -> dict:
    """Render-only clutter for one variant (world coordinates)."""
    poly, top = _support_polygon(ctx, Dr["surface"], float(Dr.get("edge_margin_m", 0.06)))   # collision footprints
                                                                       # can be wider than the visible top
    keep = keep_out_points(ctx, var)
    margin = float(Dr.get("margin_m", 0.06))
    out = {"cables": [], "shadow_casters": []}
    C = Dr.get("cables") or {}
    n = int(round(_u(rng, C.get("count"), 0)))
    lo, hi = poly.min(0), poly.max(0)
    for _ in range(n):
        for _try in range(60):
            r = _u(rng, C.get("radius_m"), 0.003)
            L = _u(rng, C.get("length_m"), 0.5)
            p = rng.uniform(lo, hi)
            ang = rng.uniform(0, 2 * np.pi)
            P = [p]
            step = L / 6
            for _k in range(6):
                ang += rng.normal(0, 0.6)
                P.append(P[-1] + step * np.array([np.cos(ang), np.sin(ang)]))
            P = np.asarray(P)
            inside = all(_in_poly(q, poly) for q in P)
            clear = np.min(np.linalg.norm(P[:, None] - keep[None], axis=2)) > margin
            if inside and clear:
                cols = C.get("colors") or [[0.01, 0.01, 0.01]]
                out["cables"].append({"points": np.c_[P, np.full(len(P), top + r)].round(4).tolist(), "radius": r,
                                      "color": cols[int(rng.integers(len(cols)))]})
                break
    Sc = Dr.get("shadow_casters") or {}
    lights = [l for l in (var.get("lights") or ctx.scene.get("lights") or []) if l.get("type") in ("area", "point", "spot")]
    for _ in range(int(round(_u(rng, Sc.get("count"), 0)))):
        if not lights:
            break
        Lp = np.asarray(lights[0]["position"], float)
        t = np.r_[rng.uniform(lo, hi), top]                       # a table point whose light gets blocked
        h = _u(rng, Sc.get("height_m"), 0.5)
        c = t + (Lp - t) * min(0.95, h / max(Lp[2] - top, 1e-3))
        a, b_ = _u(rng, Sc.get("size_m"), 0.3), _u(rng, Sc.get("size_m"), 0.3)
        out["shadow_casters"].append({"center": c.round(4).tolist(), "normal": ((Lp - t) / np.linalg.norm(Lp - t)).round(4).tolist(),
                                      "size": [round(a, 3), round(b_, 3)], "yaw": float(rng.uniform(0, np.pi))})
    return out


# ---------------------------------------------------------------- trajectory transfer
def time_warp(t, Q, speed: float):
    """Resample a (N, J) target trajectory played `speed` times faster, at the same row rate."""
    t = np.asarray(t, float)
    if abs(speed - 1.0) < 1e-9:
        return t.copy(), np.asarray(Q, float).copy()
    dt = float(np.median(np.diff(t)))
    dur = (t[-1] - t[0]) / speed
    tn = t[0] + np.arange(0, dur + 1e-9, dt)
    src = t[0] + (tn - t[0]) * speed
    Qn = np.stack([np.interp(src, t, Q[:, j]) for j in range(Q.shape[1])], 1)
    return tn, Qn


def palm_targets(demo: Demo, var: dict) -> tuple[np.ndarray, dict]:
    """(N,4,4) palm targets of the active hand for a variant (before time warping) and the schedule."""
    ev = demo.events
    n = len(demo.palm)
    tr = var.get("trajectory") or {}
    G = np.asarray(tr.get("grasp_delta", np.eye(4)), float)
    Pd = np.asarray(tr.get("place_delta", np.eye(4)), float)
    ents = var.get("entities") or {}
    To1 = np.asarray((ents.get(demo.obj) or {}).get("T_world_obj", demo.T_obj0), float)
    D_obj = To1 @ G @ np.linalg.inv(demo.T_obj0)
    if demo.target:
        Tt1 = np.asarray((ents.get(demo.target) or {}).get("T_world_obj", demo.T_tgt0), float)
        if tr.get("place_anchor", "pose") == "position":       # e.g. a container: only where it is matters
            Tt1 = Tt1.copy()
            Tt1[:3, :3] = demo.T_tgt0[:3, :3]
        D_tgt = Tt1 @ Pd @ np.linalg.inv(demo.T_tgt0)
    else:
        D_tgt = D_obj
    lead = int(round(float(tr.get("approach_lead_s", 0.5)) / demo.dt))
    p0 = demo.palm[0][:3, 3]
    mv = np.linalg.norm(demo.palm[:, :3, 3] - p0, axis=1) > 0.01
    f_move = int(np.argmax(mv)) if mv.any() else 0
    f_pre = int(np.clip(ev["contact"][0] - lead, f_move + 1, n - 1))
    f_lift = int(np.clip(max(ev["lift"], ev["contact"][0]) + 3, f_pre, n - 1))
    f_place = int(np.clip(ev["place"], f_lift + 1, n - 1))
    via = np.asarray(tr.get("via_offset", [0, 0, 0]), float)
    out = np.empty_like(demo.palm)
    for k in range(n):
        P = demo.palm[k]
        if k <= f_move:
            T = P.copy()
        elif k < f_pre:
            T = blend_pose(P, D_obj @ P, min_jerk((k - f_move) / max(f_pre - f_move, 1)))
        elif k <= f_lift:
            T = D_obj @ P
        elif k < f_place:
            s = (k - f_lift) / max(f_place - f_lift, 1)
            T = blend_pose(D_obj @ P, D_tgt @ P, min_jerk(s))
            T[:3, 3] += np.sin(np.pi * s) * via
        else:
            T = D_tgt @ P
        out[k] = T
    return out, {"f_move": f_move, "f_pre": f_pre, "f_lift": f_lift, "f_place": f_place}


def transfer(demo: Demo, var: dict, iters: int = 60, smooth: float = 0.05) -> tuple[dict, dict]:
    """Commanded motion {t, frame, joint_names, q} for a variant: active-arm IK on the re-anchored palm path, the rest
    of the demo's targets unchanged, then the variant's speed (time warp). info: IK errors, joint speed ratio."""
    targets, sched = palm_targets(demo, var)
    f0 = sched["f_move"]
    Q = demo.Q.copy()
    if var.get("nominal"):                                  # the demo itself, untouched
        pe, re = np.zeros(1), np.zeros(1)
    else:
        idx = [demo.joint_names.index(j) for j in demo.arm]
        q0 = demo.q_dict(demo.Q, f0)
        Qa, infos = demo.kin.ik_traj({demo.palm_link: targets[f0:]}, joints=demo.arm, q0=q0, smooth=smooth, iters=iters)
        Q[f0:, idx] = Qa
        pe = np.array([i[demo.palm_link]["pos_err_m"] for i in infos]) * 1000
        re = np.degrees([i[demo.palm_link].get("rot_err_rad", 0.0) for i in infos])
    speed = float((var.get("trajectory") or {}).get("speed", 1.0))
    t, Qw = time_warp(demo.t, Q, speed)
    fps = 1.0 / demo.dt
    tail = 0 if var.get("nominal") else int(round(float((var.get("trajectory") or {}).get("tail_s", 1.0)) * fps))
    if tail > 0:                                            # hold the last targets so objects settle before the end
        t = np.r_[t, t[-1] + np.arange(1, tail + 1) / fps]
        Qw = np.vstack([Qw, np.repeat(Qw[-1:], tail, axis=0)])
    # joint speed vs URDF velocity limits
    vlim = []
    for j in demo.joint_names:
        L = (demo.bundle.model["joints"].get(j) or {}).get("limit") or {}
        vlim.append(float(L.get("velocity") or np.inf))
    vel = np.abs(np.diff(Qw, axis=0)) * fps
    ratio = float(np.max(vel / np.asarray(vlim)[None])) if len(vel) else 0.0
    info = {"schedule": sched, "ik_pos_err_mm_max": float(pe.max()), "ik_pos_err_mm_p95": float(np.percentile(pe, 95)),
            "ik_rot_err_deg_max": float(re.max()), "joint_speed_ratio_max": ratio, "speed": speed,
            "frames": len(t), "duration_s": float(t[-1] - t[0])}
    cmd = {"t": t, "frame": (t - t[0]) * fps + float(demo.cmd["frame"][0]), "joint_names": demo.joint_names, "q": Qw}
    return cmd, info


# ---------------------------------------------------------------- materialise, success
def materialize(ctx: Context, var: dict, ep_dir: Path | None = None) -> dict:
    """Scene IR of an episode: base scene + variant (+ its commanded/achieved motions when ep_dir has them)."""
    import os
    from .scene_ir import apply_variant
    v = copy.deepcopy(var)
    if ep_dir is not None:
        rel = lambda f: os.path.relpath(Path(ep_dir) / f, ctx.project / "scene")   # noqa: E731
        mot = {}
        if (Path(ep_dir) / "commanded.npz").exists():
            mot["commanded"] = {"robot": ctx.demo.rid, "kind": "commanded", "uri": rel("commanded.npz"),
                                "rate_hz": 1.0 / ctx.demo.dt, "derived_from": ctx.demo.cmd_id,
                                "provenance": {"q": f"synthkit.transfer of {ctx.demo.cmd_id} for variant {var['id']}"}}
        if (Path(ep_dir) / "achieved.npz").exists():
            mot["achieved"] = {"robot": ctx.demo.rid, "kind": "achieved", "uri": rel("achieved.npz"),
                               "rate_hz": 1.0 / ctx.demo.dt, "derived_from": "commanded",
                               "provenance": {"q": "MuJoCo rollout (synth_rollout)"}}
            v["render"] = {"motion": "achieved"}
            v["trajectory_entities"] = ctx.rigid()
        v["motions"] = mot
    return apply_variant(ctx.scene, v)


def _mesh_bounds(ctx, name):
    import trimesh
    m = trimesh.load(ctx.project / "scene" / ctx.scene["entities"][name]["geometry"]["visual"]["obj_uri"],
                     force="mesh", process=False)
    V = np.asarray(m.vertices)
    return V.min(0), V.max(0), V.mean(0)


def success(ctx: Context, R: dict, task: dict | None = None) -> tuple[bool, dict]:
    """Task predicate on a rollout R (simkit rollout result). Returns (ok, details)."""
    from scipy.spatial.transform import Rotation as Rot
    task = task or ctx.spec["task"]
    sc = task.get("success") or {"type": "in_container"}
    names = list(R["obj_names"])
    obj = task["object"]
    To = R["obj_T"][:, names.index(obj)]
    fps = 1.0 / float(np.median(np.diff(R["t"])))
    lo, hi, cen = _mesh_bounds(ctx, obj)
    pc = To[:, :3, :3] @ cen + To[:, :3, 3]                    # object centre path (world)
    det = {"type": sc.get("type", "in_container")}
    ok = True
    if det["type"] == "in_container":
        tgt = task["target"]
        Tt = R["obj_T"][-1, names.index(tgt)]
        tlo, thi, _ = _mesh_bounds(ctx, tgt)
        m = float(sc.get("margin_m", 0.0))
        p = np.linalg.inv(Tt) @ np.r_[pc[-1], 1]
        inside = bool(np.all(p[:2] >= tlo[:2] + m) and np.all(p[:2] <= thi[:2] - m) and tlo[2] - 0.01 <= p[2] <= thi[2] + 0.03)
        tilt = float(np.degrees(np.arccos(np.clip(Tt[2, 2], -1, 1))))
        det.update({"inside": inside, "obj_in_target_frame": p[:3].round(4).tolist(), "target_tilt_deg": tilt,
                    "target_moved_m": float(np.linalg.norm(R["obj_T"][-1, names.index(tgt), :3, 3] - R["obj_T"][0, names.index(tgt), :3, 3]))})
        ok &= inside and tilt <= float(sc.get("max_target_tilt_deg", 30))
    elif det["type"] == "at_pose":
        ref = sc.get("relative_to")
        Tref = R["obj_T"][-1, names.index(ref)] if ref and ref != "world" else np.eye(4)
        Tg = Tref @ np.asarray(sc["T"], float)
        dp = float(np.linalg.norm(To[-1, :3, 3] - Tg[:3, 3]))
        dr = float(np.degrees(Rot.from_matrix(To[-1, :3, :3] @ Tg[:3, :3].T).magnitude()))
        det.update({"pos_err_m": dp, "rot_err_deg": dr})
        ok &= dp <= float(sc.get("pos_tol_m", 0.03)) and dr <= float(sc.get("rot_tol_deg", 25))
    n_settle = max(2, int(round(float(sc.get("settle_s", 0.5)) * fps)))
    v_end = float(np.max(np.linalg.norm(np.diff(pc[-n_settle:], axis=0), axis=1)) * fps)
    det["end_speed_mps"] = v_end
    ok &= v_end <= float(sc.get("max_end_speed_mps", 0.03))
    if sc.get("released", True) and R.get("contacts"):
        held = 0
        for cs in R["contacts"][-n_settle:]:
            if any((c["a"][0] == "robot" and c["b"][:2] == ("entity", obj)) or
                   (c["b"][0] == "robot" and c["a"][:2] == ("entity", obj)) for c in cs):
                held += 1
        det["robot_contact_frames_at_end"] = held
        ok &= held == 0
    det["obj_min_z"] = float(pc[:, 2].min())
    return bool(ok), det


def physics_ok(ctx: Context, metrics: dict, info: dict | None = None) -> tuple[bool, list]:
    """Per-episode acceptance on rollout metrics (and transfer info): list of failed checks."""
    a = ctx.accept
    bad = []
    groups = ctx.demo.bundle.profile["groups"]
    hand = {j for g in groups.values() if g.get("kind") == "hand" for j in g["joints"]}
    sat = metrics.get("actuator_saturation_all")
    if sat is not None:
        arm_sat = max([v for k, v in sat.items() if k.split("/")[-1] not in hand] or [0.0])
        hand_sat = max([v for k, v in sat.items() if k.split("/")[-1] in hand] or [0.0])
        if arm_sat > a["actuator_saturation_frac_max"]:
            bad.append(f"saturation {arm_sat:.3f} > {a['actuator_saturation_frac_max']}")
        if hand_sat > a["hand_saturation_frac_max"]:
            bad.append(f"hand saturation {hand_sat:.3f} > {a['hand_saturation_frac_max']}")
    elif metrics.get("actuator_saturation_frac", 0) > a["actuator_saturation_frac_max"]:
        bad.append(f"saturation {metrics['actuator_saturation_frac']:.3f} > {a['actuator_saturation_frac_max']}")
    for cat, v in (metrics.get("max_penetration_mm") or {}).items():
        key = {"object|robot": "object_robot", "robot|static": "robot_static", "object|static": "object_static",
               "robot|robot": "robot_robot", "object|object": "object_object"}.get(cat, cat)
        lim = a["penetration_mm_max"].get(key)
        if lim is not None and v > lim:
            bad.append(f"penetration {key} {v:.2f} mm > {lim}")
    if metrics.get("joint_limit_violation_rad", 0) > a["joint_limit_violation_rad_max"]:
        bad.append(f"joint limits {metrics['joint_limit_violation_rad']:.3f} rad")
    if info:
        if info["ik_pos_err_mm_max"] > a["ik_pos_err_mm_max"]:
            bad.append(f"ik position error {info['ik_pos_err_mm_max']:.1f} mm")
        if info["ik_rot_err_deg_max"] > a["ik_rot_err_deg_max"]:
            bad.append(f"ik rotation error {info['ik_rot_err_deg_max']:.1f} deg")
        if info["joint_speed_ratio_max"] > a["joint_speed_ratio_max"]:
            bad.append(f"joint speed {info['joint_speed_ratio_max']:.2f} x limit")
    return not bad, bad


# ---------------------------------------------------------------- one attempt (used by synth_rollout workers)
def run_attempt(ctx: Context, attempt: int, out_dir: Path, nominal: bool = False, keep_rejected: bool = False) -> dict:
    """Sample, transfer, simulate, judge one attempt. Accepted episodes are written to out_dir/episodes/<id>/
    (variant.yaml, commanded.npz, achieved.npz, rollout.json); returns a summary row."""
    import json
    import time
    import yaml
    from . import simkit as sk
    from .scene_ir import _plain, save_motion
    t0 = time.time()
    var = sample_variant(ctx, attempt, nominal=nominal)
    row = {"id": var["id"], "attempt": attempt, "nominal": nominal}
    if var.get("invalid"):
        return {**row, "status": "rejected", "stage": "sample", "reason": var["invalid"], "s": time.time() - t0}
    cmd, info = transfer(ctx.demo, var)
    row["transfer"] = info
    ok, bad = physics_ok(ctx, {}, info)
    if not ok:
        return {**row, "status": "rejected", "stage": "transfer", "reason": "; ".join(bad), "variant": var,
                "s": time.time() - t0}
    ep = Path(out_dir) / "episodes" / var["id"]
    ep.mkdir(parents=True, exist_ok=True)
    save_motion(ep / "commanded.npz", cmd["t"], cmd["frame"], cmd["joint_names"], cmd["q"], ctrl=cmd["q"])
    scene_v = materialize(ctx, var, ep)
    sim = sk.Sim(ctx.project, scene=scene_v, mdir=Path(out_dir) / "mujoco", xml_name=f"{var['id']}.xml")
    audit = sim.audit()
    R = sim.rollout(cmd, start="first_command", settle_s=float((ctx.spec.get("rollout") or {}).get("settle_s", 0.0)))
    good, det = success(ctx, R)
    pok, pbad = physics_ok(ctx, R["metrics"], info)
    row.update({"success": det, "metrics": R["metrics"], "audit": audit})
    if good and pok and ctx.accept.get("robust_replay", True) and not nominal:
        # the exported actions are float32: contact-rich episodes that only succeed for the exact float64 targets
        # are too sensitive to be useful demonstrations
        cmd32 = {**cmd, "q": np.asarray(cmd["q"], np.float32).astype(float)}
        R32 = sim.rollout(cmd32, start="first_command", settle_s=float((ctx.spec.get("rollout") or {}).get("settle_s", 0.0)))
        g32, d32 = success(ctx, R32)
        row["robust_replay"] = {"success": g32, "q_max_diff_rad": float(np.abs(R32["q"] - R["q"]).max()),
                                "end_speed_mps": d32.get("end_speed_mps")}
        if not g32:
            good = False
            det = {**det, "robust_replay": d32}
    if not (good and pok and audit["ok"]):
        reason = ([] if good else ["task failed: " + json.dumps({k: v for k, v in det.items() if k != "type"}, default=float)[:200]]) + pbad + ([] if audit["ok"] else ["model audit"])
        row.update({"status": "rejected", "stage": "physics" if good else "task", "reason": "; ".join(reason),
                    "variant": var, "s": time.time() - t0})
        if not keep_rejected:
            import shutil
            shutil.rmtree(ep, ignore_errors=True)
            (Path(out_dir) / "mujoco" / f"{var['id']}.xml").unlink(missing_ok=True)
        return row
    save_motion(ep / "achieved.npz", R["t"], R["frame"], R["joint_names"], R["q"], ctrl=R["ctrl"],
                obj_names=R["obj_names"], obj_T=R["obj_T"])
    (ep / "variant.yaml").write_text(yaml.safe_dump(_plain(var), sort_keys=False))
    contacts = [[{"a": list(c["a"]), "b": list(c["b"]), "depth": c["depth"], "force": c["force"]} for c in cs]
                for cs in R["contacts"]]
    (ep / "contacts.json").write_text(json.dumps(contacts, default=float))
    res = {**row, "status": "accepted", "s": time.time() - t0}
    (ep / "rollout.json").write_text(json.dumps(_plain(res), indent=1, default=float))
    return res
