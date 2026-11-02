"""Scene IR: the single source of truth for a reconstructed scene (projects/<id>/scene/scene.yaml).

Backends (blender_render, blender_bake, mujoco_check, geometry_metrics, render_views) only read it; the agent writes
it (with its own code) and validates it with `validate()`. Paths are relative to the scene/ folder.

schema_version: "0.4"                  # 0.2/0.3 scenes remain valid (0.4 adds the variant fields below)
units: {length: m, time: s, angle: rad}
frames: {world: <text>, quaternion_order: wxyz, camera_optical: "OpenCV (x right, y down, z forward)"}
source: {video: <uri>, sha256: <hex>, withheld: [...]}
scale: {source: <text>, uncertainty_rel: <float>, cues: {...}, rejected: {...}}
cameras:
  source_cam: {model: pinhole, width, height, K: 3x3, distortion: null | [...], fps,
               poses_uri: <json with T_world_cam: [N x 4x4]> | T_world_cam: 4x4, provenance: {...}}
  <name>: {...}                        # extra cameras (robot, review); T_world_cam maps camera points to world
lights:                                # one rig for the whole clip
  - {name, type: area|sun|point, position: [x,y,z], target: [x,y,z], size: [a,b], strength_w, color: [r,g,b]}
  - {type: world, color: [r,g,b], strength}
entities:
  <name>:
    category: <text>                   # table, container, produce, background, ...
    capability: static | rigid | visual_only
    T_world_entity: 4x4                # static / visual_only placement (rigid: pose of the initial state)
    geometry:
      visual: {obj_uri: <obj with UVs>}          # mesh in the entity frame
      collision: [ {type: box, size: [x,y,z], T: 4x4} | {type: mesh, uri: <convex obj>} ... ]   # rigid/static
    material:                                    # one material per entity
      shader: principled | emission
      base_color: [r,g,b]                         # linear 0-1, used when no texture
      base_color_texture: <png>                   # sRGB albedo (principled) or radiance (emission)
      tile_m: null | <float>                      # if set, mesh UVs are in metres and the texture repeats every tile_m
      roughness: 0.6, specular: 0.0, emission_strength: 1.0
      metallic: 0.0
      roughness_texture: <png>                   # linear scalar data, shares albedo UV / tile_m
      normal_texture: <png>, normal_strength: 1.0 # tangent-space normal map, Non-Color data
    physics: {mass_kg: <float>, friction: <float>}            # rigid; each value may be {value, kind}
    states: {<state>: {frames: [a, b], T_world_obj: 4x4}}      # rigid: rest intervals in source-video frames
    initial_state: <state>
    trajectory: {motion: <motion id>}            # optional: per-frame poses (e.g. a simulated rollout) used for
                                                 # rendering instead of rest states (shown while moving too)
    provenance: {<field>: observed | fitted | completed | prior | generated (+ text)}
simulation: {timestep: 0.002, integrator: implicitfast, cone: elliptic, impratio: 10, gravity: [0, 0, -9.81],
             noslip_iterations: 0}        # physics settings shared by every simulator backend
robots:                                  # RobotBundles placed in the scene (any2sim.robot.bundle)
  <robot id>:
    bundle: <dir>                        # relative to scene/, absolute, or "assets:<name>" (configs: assets.robots)
    bundle_sha256: <hex>                 # robot.yaml + robot.urdf digest the scene was built against (later
                                         # appearance-only revisions are fine: scene_ir.bundle_compatible)
    T_world_base: 4x4                    # the robot's base_link in the world
    initial_q: {<joint>: value}          # default: the bundle's init_pose
    provenance: {T_world_base: ..., initial_q: ...}
motions:                                 # time-indexed trajectories (MotionBundle)
  <motion id>:
    robot: <robot id> | null             # null: object-only motion
    kind: reference | commanded | achieved
    uri: <npz>                           # t (N,) s; frame (N,) video frame index (float); joint_names (J,);
                                         # q (N,J); optional ctrl (N,J) commanded targets; obj_names (M,);
                                         # obj_T (N,M,4,4) world poses; contact/metric sidecars next to it
    rate_hz: <float>
    derived_from: <motion id> | null
    provenance: {...}
render: {motion: <motion id>}            # which (achieved) motion drives renders of robots/object trajectories
render.head_camera: source_cam          # scene camera used for dataset head output; source_cam stays the reconstruction reference
render.camera_isp: {head: {...}, wrist_left: {...}, wrist_right: {...}} # per-camera override; {} = identity;
                                         # otherwise use that camera's own ISP, never another camera's calibration
variants:                                # (v0.4) synthesis stage: the randomisation spec is synthesis/spec.yaml;
                                         # each episode is this scene + one variant overlay (apply_variant), so
                                         # MuJoCo, Blender and Isaac see exactly the same randomised scene
cameras.<name>.T_delta: 4x4              # (v0.4, optional) camera mount perturbation: T_world_cam @ T_delta
cameras.<name>.isp: {...}                # (v0.4, optional) camera image model (any2sim.camera_isp; camera_calibrate):
                                         # applied to renders; not part of the core digest
robots.<id>.overrides: {actuators: {<group>: {kp, kv, damping, armature}}, contact: {<group>: {friction, ...}}}
                                         # (v0.4, optional) scene-level physics overrides of the bundle (declared,
                                         # the bundle itself is never edited)
robots.<id>.camera_deltas: {<bundle camera>: 4x4}   # (v0.4, optional) robot-camera mount perturbations
simulation.control_latency_s: 0.0        # (v0.4) commanded targets reach the actuators this much later
entities.<name>.material.hsv_shift: [dh_deg, ds, dv]          # (v0.4, render-only) hue/saturation/value change
entities.<name>.material.procedural: {type: noise|wave|checker|voronoi|brick, colors: [[r,g,b], [r,g,b]],
                                      scale: <1/m>}           # (v0.4, render-only) replaces the base colour
registrations:                           # external metric observations placed in the world (rigid, never scaled)
  <name>:                                # e.g. hand_tracks
    uri: <path relative to the project>  # e.g. inputs/hand_tracks.npz
    T_world_tracks: 4x4                  # rigid: tracks frame -> world (if the scene disagrees, rescale the scene)
    frame_offset: 0                      # video frame = track row + frame_offset
    residuals: {...}, provenance: {...}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REQUIRED_TOP = ("schema_version", "units", "frames", "cameras", "entities")
CAPABILITIES = ("static", "rigid", "visual_only")


def load(scene_dir: str | Path) -> dict:
    import yaml
    return yaml.safe_load((Path(scene_dir) / "scene.yaml").read_text())


def save(scene_dir: str | Path, scene: dict) -> str:
    import yaml
    p = Path(scene_dir) / "scene.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(_plain(scene), sort_keys=False, allow_unicode=True))
    return str(p)


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def _is_rigid_T(T, tol=1e-4) -> bool:
    T = np.asarray(T, float)
    if T.shape != (4, 4):
        return False
    R = T[:3, :3]
    return bool(np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol
                and np.allclose(T[3], [0, 0, 0, 1]))


def render_head_camera(scene: dict) -> str:
    """Select the dataset head view without replacing the reconstruction reference camera."""
    name = scene.get("render", {}).get("head_camera", "source_cam")
    if name not in scene.get("cameras", {}):
        raise ValueError(f"render.head_camera {name!r} is not a scene camera")
    return name


def camera_poses(scene_dir: str | Path, scene: dict, cam: str = "source_cam") -> np.ndarray:
    """(N,4,4) T_world_cam of a camera (N = 1 for a fixed camera)."""
    import json
    c = scene["cameras"][cam]
    if "poses_uri" in c:
        d = json.loads((Path(scene_dir) / c["poses_uri"]).read_text())
        P = np.asarray(d["T_world_cam"] if isinstance(d, dict) else d, float)
    else:
        P = np.asarray(c["T_world_cam"], float)[None]
    if c.get("T_delta") is not None:
        P = P @ np.asarray(c["T_delta"], float)
    return P


def state_at(entity: dict, frame: int) -> str | None:
    for s, st in (entity.get("states") or {}).items():
        if st["frames"][0] <= frame <= st["frames"][1]:
            return s
    return None


def pose_at(entity: dict, frame: int | None) -> np.ndarray | None:
    """World pose of an entity at a source frame (None: rigid and moving at that frame)."""
    if entity.get("capability") != "rigid":
        return np.asarray(entity.get("T_world_entity", np.eye(4)), float)
    if frame is None:
        return np.asarray(entity["states"][entity["initial_state"]]["T_world_obj"], float)
    s = state_at(entity, frame)
    return None if s is None else np.asarray(entity["states"][s]["T_world_obj"], float)


def core_digest(scene: dict) -> str:
    """Digest of the reconstructed scene proper (cameras, lights, scale, entities without their trajectory links):
    later stages add robots/motions without changing it; any change means the upstream scene was revised."""
    import hashlib
    import json
    ents = {n: {k: v for k, v in e.items() if k != "trajectory"} for n, e in (scene.get("entities") or {}).items()}
    core = {k: scene.get(k) for k in ("units", "frames", "cameras", "lights", "scale")}
    core["cameras"] = {n: {k: v for k, v in c.items() if k != "isp"} for n, c in (scene.get("cameras") or {}).items()}
    core["entities"] = ents
    return hashlib.sha256(json.dumps(_plain(core), sort_keys=True).encode()).hexdigest()


def resolve_bundle(scene_dir: str | Path, spec: str) -> Path:
    if spec.startswith("assets:"):
        from .config import config
        return Path(config()["assets"]["robots"]) / spec.split(":", 1)[1]
    p = Path(spec)
    return p if p.is_absolute() else (Path(scene_dir) / p).resolve()


def bundle_digest(bundle_dir: str | Path) -> str:
    import hashlib
    h = hashlib.sha256()
    for f in ("robot.yaml", "robot.urdf"):
        h.update((Path(bundle_dir) / f).read_bytes())
    return h.hexdigest()


APPEARANCE_KEYS = ("visuals", "appearance", "materials", "cables", "checks", "provenance", "revisions", "onboarding_project",
                   "source", "decals")


def physics_digest(bundle_dir: str | Path) -> str:
    """Digest of what simulation depends on (robot.urdf + robot.yaml without appearance keys and camera image
    models): appearance-only revisions of a bundle (visuals, cables, materials, camera ISP) keep it."""
    import hashlib
    import json
    import yaml
    bd = Path(bundle_dir)
    prof = yaml.safe_load((bd / "robot.yaml").read_text())
    prof = {k: v for k, v in prof.items() if k not in APPEARANCE_KEYS}
    for c in (prof.get("cameras") or {}).values():
        c.pop("isp", None)
    h = hashlib.sha256()
    h.update((bd / "robot.urdf").read_bytes())
    h.update(json.dumps(_plain(prof), sort_keys=True).encode())
    return h.hexdigest()


def bundle_compatible(bundle_dir: str | Path, sha256: str) -> str:
    """'same' | 'appearance' (an earlier revision with the same physics, robot.yaml revisions) | 'changed'."""
    import yaml
    bd = Path(bundle_dir)
    if sha256 == bundle_digest(bd):
        return "same"
    phys = physics_digest(bd)
    for r in (yaml.safe_load((bd / "robot.yaml").read_text()).get("revisions") or []):
        if r.get("sha256") == sha256 and r.get("physics_sha256") == phys:
            return "appearance"
    return "changed"


def robot_bundle(scene_dir: str | Path, scene: dict, rid: str):
    """The robot's Bundle with the scene's declared overrides (actuators, contact) merged into its profile (in
    memory only; the bundle on disk is never changed)."""
    import copy
    from .robot.bundle import Bundle
    r = scene["robots"][rid]
    b = Bundle(resolve_bundle(scene_dir, r["bundle"]))
    ov = r.get("overrides") or {}
    if ov:
        b.profile = copy.deepcopy(b.profile)
        for sec in ("actuators", "contact"):
            for g, vals in (ov.get(sec) or {}).items():
                b.profile.setdefault(sec, {}).setdefault(g, {})
                b.profile[sec][g] = {**(b.profile[sec].get(g) or b.profile[sec].get("default") or {}), **vals}
    return b


def apply_variant(scene: dict, variant: dict) -> dict:
    """The scene with one synthesis variant applied (a new dict; the input is not modified).

    variant (synthesis/episodes/<id>/variant.yaml):
      entities: {<name>: {T_world_obj: 4x4 (rigid initial pose) | T_world_entity: 4x4 (static),
                          material: {...overrides}, physics: {...overrides}}}
      lights: [...]                         replaces the light rig
      cameras: {<name>: {T_delta: 4x4}}     camera mount perturbation
      robots: {<id>: {overrides: {...}, camera_deltas: {...}, initial_q: {...}}}
      simulation: {...}                     merged (e.g. control_latency_s)
      motions: {<id>: {...}}                replaces the motion table (episode motions); render: {motion: <id>}
    Rigid entities get a single state valid for the whole episode at their variant pose; entities moved by the
    episode's achieved motion get trajectory links (set by the caller through `motions`/`render`)."""
    import copy
    sc = copy.deepcopy(scene)
    v = variant or {}
    for n, ov in (v.get("entities") or {}).items():
        e = sc["entities"][n]
        if ov.get("T_world_obj") is not None:
            if e.get("capability") == "rigid":
                e["states"] = {"variant_initial": {"frames": [0, 10 ** 7], "T_world_obj": _plain(ov["T_world_obj"])}}
                e["initial_state"] = "variant_initial"
            else:
                e["T_world_entity"] = _plain(ov["T_world_obj"])
        if ov.get("T_world_entity") is not None:
            e["T_world_entity"] = _plain(ov["T_world_entity"])
        if ov.get("material"):
            e["material"] = {**(e.get("material") or {}), **ov["material"]}
        if ov.get("physics"):
            e["physics"] = {**(e.get("physics") or {}), **ov["physics"]}
        e.pop("trajectory", None)
    if v.get("lights") is not None:
        sc["lights"] = copy.deepcopy(v["lights"])
    for cn, ov in (v.get("cameras") or {}).items():
        sc["cameras"][cn].update(copy.deepcopy(ov))
    for rid, ov in (v.get("robots") or {}).items():
        r = sc["robots"][rid]
        for k in ("overrides", "camera_deltas"):
            if ov.get(k):
                r[k] = copy.deepcopy(ov[k])
        if ov.get("initial_q"):
            r["initial_q"] = {**(r.get("initial_q") or {}), **ov["initial_q"]}
    if v.get("simulation"):
        sc["simulation"] = {**(sc.get("simulation") or {}), **v["simulation"]}
    if v.get("motions") is not None:
        sc["motions"] = copy.deepcopy(v["motions"])
        for e in sc["entities"].values():
            e.pop("trajectory", None)
        if sc.get("render"):
            sc["render"].pop("motion", None)
    if v.get("render"):
        sc.setdefault("render", {}).update(copy.deepcopy(v["render"]))
        for n in (v.get("trajectory_entities") or []):
            sc["entities"][n]["trajectory"] = {"motion": v["render"]["motion"]}
    sc["variant"] = {"id": v.get("id"), "seed": v.get("seed"), "base_core_sha256": v.get("base_core_sha256")}
    return sc


def load_motion(scene_dir: str | Path, scene: dict, mid: str) -> dict:
    d = np.load(Path(scene_dir) / scene["motions"][mid]["uri"], allow_pickle=False)
    out = {k: d[k] for k in d.files}
    for k in ("joint_names", "obj_names"):
        if k in out:
            out[k] = [str(x) for x in out[k]]
    return out


def save_motion(path: str | Path, t, frame, joint_names=(), q=None, ctrl=None, obj_names=(), obj_T=None, **extra) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = {"t": np.asarray(t, float), "frame": np.asarray(frame, float),
           "joint_names": np.asarray(list(joint_names), dtype="U64")}
    if q is not None:
        arr["q"] = np.asarray(q, float)
    if ctrl is not None:
        arr["ctrl"] = np.asarray(ctrl, float)
    if obj_T is not None:
        arr["obj_names"] = np.asarray(list(obj_names), dtype="U64")
        arr["obj_T"] = np.asarray(obj_T, float)
    arr.update({k: np.asarray(v) for k, v in extra.items()})
    np.savez_compressed(path, **arr)
    return str(path)


def motion_at(motion: dict, frame: float) -> tuple[dict, dict]:
    """(q {joint: value}, object poses {name: 4x4}) at a video frame (linear in q, nearest for object poses)."""
    fr = motion["frame"]
    i = int(np.clip(np.searchsorted(fr, frame), 1, len(fr) - 1))
    a = float(np.clip((frame - fr[i - 1]) / max(fr[i] - fr[i - 1], 1e-9), 0, 1))
    q = {}
    if "q" in motion:
        qq = (1 - a) * motion["q"][i - 1] + a * motion["q"][i]
        q = dict(zip(motion["joint_names"], qq.tolist()))
    objs = {}
    if "obj_T" in motion:
        k = i if a >= 0.5 else i - 1
        objs = {n: motion["obj_T"][k, j] for j, n in enumerate(motion["obj_names"])}
    return q, objs


def validate(scene_dir: str | Path, scene: dict | None = None) -> dict:
    """Structural checks: required keys, files exist, transforms are rigid, states and cameras consistent.
    Returns {"ok": bool, "errors": [...], "warnings": [...]}."""
    sd = Path(scene_dir)
    scene = scene or load(sd)
    err, warn = [], []
    for k in REQUIRED_TOP:
        if k not in scene:
            err.append(f"missing top-level key {k!r}")
    cams = scene.get("cameras") or {}
    if "source_cam" not in cams:
        err.append("cameras.source_cam missing")
    try:
        render_head_camera(scene)
    except ValueError as exc:
        err.append(str(exc))
    for cn, c in cams.items():
        for k in ("width", "height", "K"):
            if k not in c:
                err.append(f"camera {cn}: missing {k}")
        if "poses_uri" in c and not (sd / c["poses_uri"]).exists():
            err.append(f"camera {cn}: poses_uri {c['poses_uri']} not found")
        if "poses_uri" not in c and "T_world_cam" not in c:
            err.append(f"camera {cn}: needs poses_uri or T_world_cam")
        if "provenance" not in c:
            warn.append(f"camera {cn}: no provenance")
    if not scene.get("lights"):
        warn.append("no lights: renders use only the world background")
    for name, e in (scene.get("entities") or {}).items():
        cap = e.get("capability")
        if cap not in CAPABILITIES:
            err.append(f"{name}: capability {cap!r} not in {CAPABILITIES}")
        vis = (e.get("geometry") or {}).get("visual") or {}
        if not vis.get("obj_uri"):
            err.append(f"{name}: geometry.visual.obj_uri missing")
        elif not (sd / vis["obj_uri"]).exists():
            err.append(f"{name}: {vis['obj_uri']} not found")
        mat = e.get("material") or {}
        if not mat:
            err.append(f"{name}: material missing")
        for key in ("base_color_texture", "roughness_texture", "normal_texture"):
            tex = mat.get(key)
            if tex and not (sd / tex).exists():
                err.append(f"{name}: {key} {tex} not found")
        if cap in ("static", "visual_only") and not _is_rigid_T(e.get("T_world_entity", np.eye(4))):
            err.append(f"{name}: T_world_entity is not a rigid transform")
        if cap in ("static", "rigid"):
            for c in (e.get("geometry") or {}).get("collision") or []:
                if c.get("type") == "mesh" and not (sd / c["uri"]).exists():
                    err.append(f"{name}: collision mesh {c['uri']} not found")
                if c.get("type") not in ("mesh", "box"):
                    err.append(f"{name}: collision type {c.get('type')!r} (box | mesh)")
            if not (e.get("geometry") or {}).get("collision"):
                warn.append(f"{name}: no collision geometry")
        if cap == "rigid":
            st = e.get("states") or {}
            if not st:
                err.append(f"{name}: rigid entity without states")
            if e.get("initial_state") not in st:
                err.append(f"{name}: initial_state {e.get('initial_state')!r} not in states")
            for s, v in st.items():
                if not _is_rigid_T(v.get("T_world_obj", np.zeros((4, 4)))):
                    err.append(f"{name}.{s}: T_world_obj is not a rigid transform")
                if len(v.get("frames", [])) != 2:
                    err.append(f"{name}.{s}: frames must be [first, last]")
            if "mass_kg" not in (e.get("physics") or {}):
                warn.append(f"{name}: physics.mass_kg missing (default used)")
        if not e.get("provenance"):
            warn.append(f"{name}: no provenance")
        tr = e.get("trajectory")
        if tr and tr.get("motion") not in (scene.get("motions") or {}):
            err.append(f"{name}: trajectory motion {tr.get('motion')!r} not in motions")
    for rid, r in (scene.get("robots") or {}).items():
        try:
            bd = resolve_bundle(sd, r["bundle"])
            if not (bd / "robot.yaml").exists():
                err.append(f"robot {rid}: bundle {bd} not found")
            elif r.get("bundle_sha256") and bundle_compatible(bd, r["bundle_sha256"]) == "changed":
                warn.append(f"robot {rid}: bundle changed since the scene was built (physics digest differs)")
        except Exception as ex:                                       # noqa: BLE001
            err.append(f"robot {rid}: {ex}")
        if not _is_rigid_T(r.get("T_world_base", np.zeros((4, 4)))):
            err.append(f"robot {rid}: T_world_base is not a rigid transform")
        if not r.get("provenance"):
            warn.append(f"robot {rid}: no provenance")
    for mid, m in (scene.get("motions") or {}).items():
        if m.get("robot") and m["robot"] not in (scene.get("robots") or {}):
            err.append(f"motion {mid}: robot {m['robot']!r} not in robots")
        if m.get("kind") not in ("reference", "commanded", "achieved"):
            err.append(f"motion {mid}: kind must be reference | commanded | achieved")
        if not (sd / m.get("uri", "")).is_file():
            err.append(f"motion {mid}: {m.get('uri')} not found")
            continue
        try:
            mm = load_motion(sd, scene, mid)
            n = len(mm["t"])
            if len(mm["frame"]) != n or ("q" in mm and mm["q"].shape != (n, len(mm["joint_names"]))):
                err.append(f"motion {mid}: inconsistent array lengths")
            if np.any(np.diff(mm["t"]) <= 0):
                err.append(f"motion {mid}: t not strictly increasing")
        except Exception as ex:                                       # noqa: BLE001
            err.append(f"motion {mid}: cannot load ({ex})")
    for rn, r in (scene.get("registrations") or {}).items():
        T = r.get("T_world_tracks")
        if T is None or not _is_rigid_T(T, tol=1e-3):
            err.append(f"registration {rn}: T_world_tracks must be a rigid transform (metric tracks are never scaled; "
                       "rescale the scene instead)")
        if r.get("scale") not in (None, 1, 1.0):
            err.append(f"registration {rn}: scale {r.get('scale')} != 1 (rescale the scene, not the tracks)")
    rm = (scene.get("render") or {}).get("motion")
    if rm and rm not in (scene.get("motions") or {}):
        err.append(f"render.motion {rm!r} not in motions")
    return {"ok": not err, "errors": err, "warnings": warn}


def phys(v, default):
    """Physics value that may be given as a number or {value, kind}."""
    if isinstance(v, dict):
        return v.get("value", default)
    return default if v is None else v
