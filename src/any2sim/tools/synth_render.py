"""Photo-realistic multi-camera renders of synthesized episodes (Blender Cycles), one Blender worker per GPU.

Per episode (synthesis/episodes/<id>/): the variant scene (synthkit.materialize: base scene + variant lights,
materials incl. render-only hsv_shift/procedural, camera mount deltas) with the robot posed by FK and the objects by
the achieved motion at every frame, seen by
  head          render.head_camera (default source_cam), with its variant T_delta
  wrist_left/right  the bundle's wrist cameras on their links (FK of the achieved joints) x variant camera_deltas;
                distortion from robot.yaml is applied (rendered on a larger pinhole canvas, then remapped)
then the camera's image model (Scene IR cameras.<cam>.isp from camera_calibrate: colour/tone/vignetting/softness/
noise; each wrist uses its own bundle calibration, with render.camera_isp overrides) and the variant's sensor
effects (exposure, white balance, blur, noise; image space, seeded). The episode is keyframed, so motion blur follows
the real motion; render-only dressing (loose cables, grime, unseen shadow casters) comes from the variant. Videos are written at
the motion rate (h264). At `check_frames` it also renders label masks (Workbench, flat colours: 1 robot, 2 task
object, 3 target, 4 robot cables) for the label-consistency audit. The nominal episode also gets head PNGs of every frame
(synthesis/renders/nominal/head_frames/src_%05d.png) when using source_cam. Other selected head cameras
write <camera>_frames/ so they cannot be mistaken for paired reconstruction-reference images.

params:
  episodes: "all" | [ids]          (episodes.yaml order; the nominal episode is included when present)
  cameras: [head, wrist_left, wrist_right]
  engine: eevee | cycles           default: scene.render.realism_baseline.engine, otherwise cycles; rasterised with ray-traced shadows, close to Cycles at ~1/3 of
                                   the GPU time; cycles: path traced (~1.5-3.5 s per camera-frame on shared GPUs).
                                   Use one engine for a whole dataset; the nominal fidelity check uses the same.
  samples: 128, denoise: true     (scene.render.realism_baseline can override these). OIDN runs on the CPU here: with several workers on a loaded
                                   host it dominates (4-6 s per frame); 48-64 spp without it look close (mild grain)
  res_scale: 1.0                   < 1 for quick looks only (dataset export requires full size)
  motion_blur: 0.5                 shutter as a fraction of the frame interval (0: off); the episode is keyframed
  preview_frames: null | [k, ...]  render only these frames (check PNGs, no videos; render.json marked preview, not
                                   complete: not eligible for dataset export) - for quick looks while choosing settings
  check_frames: 4                  frames per episode (evenly spaced) with label masks + kept PNGs per camera
  gpus: "auto" | [ids]             auto: configured GPUs with < 40 GB used and < 60 % utilisation
  workers_per_gpu: 2              Blender is CPU-bound between GPU kernels (scene sync ~0.35 s per render): overlap
  resume: true                     skip only complete renders matching baseline, settings, variant and motions
Full non-nominal renders require an independently passed render/realism_review.json matching the current
render/realism_baseline.json. Render nominal or preview_frames first to establish and review the baseline.
Fixed camera ISP and material/light refinement are separate from randomization; empty sensor variation does
not disable calibrated camera processing. Raw check frames are retained under raw/check for every camera. RGB is rendered before the label pass to
avoid repeated Cycles/Workbench cache resets. seconds_per_frame measures RGB frame work; label_seconds
is reported separately (seconds includes setup, RGB, encoding and labels).
Writes synthesis/renders/<id>/{<camera>.mp4, check/<camera>_<f>.png, check/<camera>_<f>_label.png, render.json} and
evidence/synth_render.json + synth_render_sheet.jpg (head/wrist frames of several episodes).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ENV = "render"
GPU = True

LABELS = {"robot": 1, "object": 2, "target": 3, "cable": 4}     # cables are appearance (not in the physics model)


def output_intrinsics(K, width, height, resolution=None, res_scale=1.0):
    """Uniform resize followed by a centred crop; never stretch camera rays."""
    K = np.asarray(K, float).copy()
    if resolution is None:
        scale = float(res_scale)
        W, H = int(round(width * scale)), int(round(height * scale))
        dx = dy = 0.0
    else:
        if len(resolution) != 2 or any(int(x) != x or x <= 0 for x in resolution):
            raise ValueError("resolution must be [positive integer width, height]")
        W, H = map(int, resolution)
        scale = max(W / width, H / height)
        dx, dy = (width * scale - W) / 2, (height * scale - H) / 2
    K[:2] *= scale
    K[0, 2] -= dx
    K[1, 2] -= dy
    return K, W, H


# ---------------------------------------------------------------- image-space sensor model
def sensor(img: np.ndarray, s: dict, key: tuple) -> np.ndarray:
    """uint8 sRGB -> exposure (EV) and white balance in linear light, blur, then Gaussian noise (sRGB units)."""
    import cv2
    if not s:
        return img
    x = img.astype(np.float32) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    lin = lin * (2.0 ** float(s.get("exposure_ev", 0.0))) * np.asarray(s.get("wb_gains", [1, 1, 1]), np.float32)
    lin = np.clip(lin, 0, 1)
    x = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    b = float(s.get("blur_px", 0.0))
    if b > 0.05:
        x = cv2.GaussianBlur(x, (0, 0), b)
    n = float(s.get("noise_std", 0.0))
    if n > 0:
        import zlib
        rng = np.random.default_rng([int(s.get("noise_seed", 0))] + [zlib.crc32(str(k).encode()) for k in key])
        x = x + rng.normal(0, n, x.shape).astype(np.float32)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def distortion_maps(K, W, H, coeffs):
    """Pinhole canvas (K_big, W_big, H_big) covering the distorted image, and cv2.remap maps big -> distorted."""
    import cv2
    K = np.asarray(K, float)
    D = np.asarray(coeffs, float)
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([u.ravel(), v.ravel()], 1)[:, None]
    und = cv2.undistortPoints(pts, K, D).reshape(-1, 2)       # normalised pinhole coordinates of each pixel
    x = und[:, 0] * K[0, 0] + K[0, 2]
    y = und[:, 1] * K[1, 1] + K[1, 2]
    pad_l, pad_t = max(0, int(np.ceil(-x.min())) + 2), max(0, int(np.ceil(-y.min())) + 2)
    Wb = int(np.ceil(x.max())) + 3 + pad_l
    Hb = int(np.ceil(y.max())) + 3 + pad_t
    Kb = K.copy()
    Kb[0, 2] += pad_l
    Kb[1, 2] += pad_t
    mx = (x + pad_l).reshape(H, W).astype(np.float32)
    my = (y + pad_t).reshape(H, W).astype(np.float32)
    return Kb, Wb, Hb, mx, my


# ---------------------------------------------------------------- worker (one Blender session, several episodes)
def _render_episode(project: Path, ctx, eid: str, cfg: dict, cams_cache: dict):
    import bpy
    import cv2
    import imageio.v2 as imageio
    import yaml
    from .. import synthkit as sy
    from ..backends import blender_scene as B
    from ..common import read_rgb, write_rgb
    from .. import camera_isp as ISP
    from ..scene_ir import camera_poses, load_motion, robot_bundle
    diagnostic = cfg.get("realism_nominal", False)
    ep = project / "realism/episode" if diagnostic else project / "synthesis" / "episodes" / eid
    out = project / "realism/renders/nominal" if diagnostic else project / "synthesis" / "renders" / eid
    (out / "check").mkdir(parents=True, exist_ok=True)
    var = yaml.safe_load((ep / "variant.yaml").read_text())
    vsha = __import__("hashlib").sha256((ep / "variant.yaml").read_bytes()).hexdigest()
    rj = out / "render.json"
    from .. import render_pipeline as RP
    fingerprint = RP.episode_fingerprint(ep, cfg["baseline"], cfg)
    if cfg.get("resume", True) and rj.exists():
        old = json.loads(rj.read_text())
        if old.get("render_fingerprint") == fingerprint and old.get("complete"):
            return old
    # A failed rerender must never leave an old complete manifest eligible for export.
    rj.unlink(missing_ok=True)
    if diagnostic:
        (out / "diagnostics.json").unlink(missing_ok=True)
    (out / "render_error.json").unlink(missing_ok=True)
    t0 = time.time()
    sc = ctx.scene if diagnostic else sy.materialize(ctx, var, ep)
    mo = load_motion(project / "scene", sc, sc["render"]["motion"] if diagnostic else "achieved")
    rid = ctx.demo.rid
    bundle = robot_bundle(project / "scene", sc, rid)
    T_base = np.asarray(sc["robots"][rid]["T_world_base"], float)
    deltas = (sc["robots"][rid].get("camera_deltas") or {})
    B.reset()
    obs, src_cam, dev = B.build(project, sc, samples=int(cfg.get("samples", 48)))
    robots = B.add_robots(project, sc)
    dressing = B.add_dressing(var.get("dressing"))
    scn = bpy.context.scene
    RP.apply_settings(scn, cfg)
    scn.render.use_persistent_data = True                 # keep the BVH between renders (big win per frame)
    scn.render.image_settings.compression = 0
    if cfg.get("engine", "eevee") == "eevee":
        scn.render.engine = "BLENDER_EEVEE"
        for k, v in (("taa_render_samples", int(cfg.get("samples", 48))), ("use_shadows", True),
                     ("use_raytracing", True), ("shadow_ray_count", 2), ("shadow_step_count", 6)):
            if hasattr(scn.eevee, k):
                setattr(scn.eevee, k, v)
    shutter = float(cfg.get("motion_blur", 0.5))
    scn.render.use_motion_blur = shutter > 0
    if shutter > 0:
        scn.render.motion_blur_shutter = shutter
    rs = float(cfg.get("res_scale", 1.0))
    if cfg.get("view_transform"):
        scn.view_settings.view_transform = cfg["view_transform"]
        scn.view_settings.look = "None"
    # the whole episode is keyframed (robot links, cables, objects), so motion blur sees the real motion
    n = len(mo["t"])
    # Video frames are achieved-motion rows; source frame ids are for reference pairing.
    B.keyframe_motion(sc, obs, robots, {**mo, "frame": np.arange(n)}, n, rid)
    scn.frame_start, scn.frame_end = 0, n - 1
    # Keep the reconstruction source camera separate from the chosen dataset head view.
    import mathutils
    cams = {}
    from ..camera_isp import for_camera
    from ..scene_ir import render_head_camera
    head_camera = render_head_camera(sc)
    for cn in cfg.get("cameras", ["head", "wrist_left", "wrist_right"]):
        if cn == "head":
            c = sc["cameras"][head_camera]
            K = np.asarray(c["K"], float)
            W, H = c["width"], c["height"]
            dist = (c.get("distortion") or {}).get("coeffs")
        else:
            bc = bundle.profile["cameras"][cn]
            K = np.asarray(bc["K"], float)
            W, H = bc["width"], bc["height"]
            dist = (bc.get("distortion") or {}).get("coeffs")
        isp = for_camera(sc, cn, bundle.profile.get("cameras", {}))
        K, W, H = output_intrinsics(K, W, H, cfg.get("resolution"), rs)
        if dist and any(abs(x) > 1e-9 for x in dist):
            key = (cn, W, H)
            if key not in cams_cache:
                cams_cache[key] = distortion_maps(K, W, H, dist)
            Kb, Wb, Hb, mx, my = cams_cache[key]
        else:
            Kb, Wb, Hb, mx, my = K, W, H, None, None
        ob = B.add_camera(f"cam_{cn}", Kb, Wb, Hb)
        if cn == "head":
            head_poses = camera_poses(project / "scene", sc, head_camera)
            previous = {}
            ob.rotation_mode = "QUATERNION"
            for row in range(n):
                source_frame = int(np.clip(round(mo["frame"][row]), 0, len(head_poses) - 1))
                B.set_cv_pose(ob, head_poses[source_frame])
                B._keyframe_pose(ob, row, previous)
            B._linear_animation(ob)
        else:
            bc = bundle.profile["cameras"][cn]
            Tl = np.asarray(bc["T_link_cam"], float)
            if deltas.get(cn) is not None:
                Tl = Tl @ np.asarray(deltas[cn], float)
            ob.parent = robots[rid]["links"][bc["link"]]
            ob.matrix_parent_inverse.identity()
            ob.matrix_basis = mathutils.Matrix((Tl @ B.CV_TO_GL).tolist())
        cams[cn] = {"obj": ob, "W": W, "H": H, "Wb": Wb, "Hb": Hb, "mx": mx, "my": my, "isp": isp,
                    "K": K.tolist(), "distortion": dist}
    checks = sorted(set(np.linspace(0, n - 1, int(cfg.get("check_frames", 4))).round().astype(int).tolist())) if cfg.get("check_frames") else []
    if diagnostic:
        checks = cfg["diagnostic_frames"]
    preview = cfg.get("preview_frames")
    if preview:
        checks = sorted(int(k) for k in preview if 0 <= int(k) < n)
    fps = float(1.0 / np.median(np.diff(mo["t"])))

    def writer(cn):
        kb = cams[cn]["isp"].get("bitrate_kbps")
        if kb:                                              # the real camera's bitrate (compression artefacts)
            scale = rs * rs  # bitrate belongs to this camera at its native resolution
            return imageio.get_writer(str(out / f"{cn}.mp4"), fps=fps, codec="libx264", macro_block_size=8,
                                      ffmpeg_params=["-b:v", f"{max(200, int(kb * scale))}k"])
        return imageio.get_writer(str(out / f"{cn}.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    writers = {} if preview else {cn: writer(cn) for cn in cams}
    tmp = out / f"_tmp_{os.getpid()}.png"
    labels_of = {}
    for R in robots.values():
        for e in R["links"].values():
            for o in e.children_recursive:
                labels_of[o.name] = LABELS["robot"]
        for c in (R.get("cables") or {}).values():
            labels_of[c["obj"].name] = LABELS["cable"]
    if ctx.demo.obj:
        labels_of[ctx.demo.obj] = LABELS["object"]
    if ctx.demo.target:
        labels_of[ctx.demo.target] = LABELS["target"]
    nominal_png = eid == "nominal" and "head" in cams
    head_frames_dir = "head_frames" if head_camera == "source_cam" else f"{head_camera}_frames"
    if nominal_png:
        (out / head_frames_dir).mkdir(exist_ok=True)
    fr_times = []
    captures = {cn: {} for cn in cams}
    for k in (checks if preview else range(n)):
        tf = time.time()
        scn.frame_set(k)
        for cn, c in cams.items():
            scn.camera = c["obj"]
            scn.render.resolution_x, scn.render.resolution_y = c["Wb"], c["Hb"]
            scn.render.resolution_percentage = 100
            B.render_to(tmp)
            im = read_rgb(tmp)
            if c["mx"] is not None:
                im = cv2.remap(im, c["mx"], c["my"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if nominal_png and cn == "head":               # before the camera model (camera_calibrate fits it)
                write_rgb(out / head_frames_dir / f"src_{int(round(mo['frame'][k])):05d}.png", im)
            if k in checks:
                (out / "raw" / "check").mkdir(parents=True, exist_ok=True)
                write_rgb(out / "raw" / "check" / f"{cn}_{k:05d}.png", im)
            im = ISP.apply(im, c["isp"], key=(eid, cn, k))
            im = sensor(im, var.get("sensor") or {}, (eid, cn, k))
            if not preview:
                writers[cn].append_data(im)
            if k in checks:
                write_rgb(out / "check" / f"{cn}_{k:05d}.png", im)
                if diagnostic:
                    from .realism_render import capture_noisy
                    noisy = out / "check" / f"{cn}_{k:05d}_noisy.png"
                    exr = out / "check" / f"{cn}_{k:05d}_light.exr"
                    capture_noisy(scn, noisy, exr)
                    if c["mx"] is not None:
                        write_rgb(noisy, cv2.remap(read_rgb(noisy), c["mx"], c["my"], cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_REPLICATE))
                    captures[cn][str(k)] = {
                        "noisy": str(noisy.relative_to(project)),
                        "denoised": str((out / "raw/check" / f"{cn}_{k:05d}.png").relative_to(project)),
                        "processed": str((out / "check" / f"{cn}_{k:05d}.png").relative_to(project)),
                        "light_passes": str(exr.relative_to(project)),
                        "light_passes_space": "linear undistorted render canvas; camera maps in render.json",
                        "decoded": str((out / "check" / f"{cn}_{k:05d}_decoded.png").relative_to(project))}
                    (project / captures[cn][str(k)]["decoded"]).unlink(missing_ok=True)
        fr_times.append(time.time() - tf)
    for w in writers.values():
        w.close()
    if diagnostic and not preview:
        for cn in cams:
            reader = imageio.get_reader(str(out / f"{cn}.mp4"))
            try:
                for k, im in enumerate(reader):
                    if str(k) in captures[cn]:
                        write_rgb(project / captures[cn][str(k)]["decoded"], im)
            finally:
                reader.close()
    # Finish RGB before switching engines: interleaved Workbench labels discard
    # Cycles' persistent render data. Labels use the same achieved frame/camera.
    label_t0 = time.time()
    for k in (checks if cfg.get("render_labels", True) else []):
        scn.frame_set(k)
        for cn, c in cams.items():
            scn.camera = c["obj"]
            scn.render.resolution_x, scn.render.resolution_y = c["Wb"], c["Hb"]
            for o in dressing["casters"]:
                o.hide_render = True
            mb = scn.render.use_motion_blur
            scn.render.use_motion_blur = False
            lab = _label_render(scn, obs, labels_of, tmp, cfg)
            scn.render.use_motion_blur = mb
            for o in dressing["casters"]:
                o.hide_render = False
            if c["mx"] is not None:
                lab = cv2.remap(lab, c["mx"], c["my"], cv2.INTER_NEAREST)
            cv2.imwrite(str(out / "check" / f"{cn}_{k:05d}_label.png"), lab)
    label_seconds = time.time() - label_t0
    tmp.unlink(missing_ok=True)
    res = {"episode": eid, "variant_sha256": vsha, "render_fingerprint": fingerprint,
           "baseline_fingerprint": cfg["baseline"]["fingerprint"],
           "render_settings": cfg["baseline"]["settings"], "realism_review": cfg["realism_review"],
           "render_config": {k: v for k, v in cfg.items() if k not in ("baseline", "realism_review")},
           "frames": n, "cameras": {cn: {"width": c["W"], "height": c["H"],
           "canvas": [c["Wb"], c["Hb"]], "distorted": c["mx"] is not None,
           "K": c["K"], "distortion": c["distortion"],
           "resize_policy": "uniform_scale_center_crop"} for cn, c in cams.items()},
           "head_camera": head_camera, "head_frames_dir": head_frames_dir if nominal_png else None,
           "check_frames": checks, "device": dev, "engine": cfg.get("engine", "eevee"),
           "samples": int(cfg.get("samples", 48)), "res_scale": rs,
           "seconds": time.time() - t0, "seconds_per_frame": float(np.mean(fr_times)), "complete": not preview,
           "rgb_seconds_per_frame": float(np.mean(fr_times)), "label_seconds": label_seconds,
           "preview": bool(preview),
           "sensor": var.get("sensor"), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "motion_blur": shutter,
           "isp": {cn: sorted(c["isp"]) for cn, c in cams.items()},
           "camera_models": {cn: {"parameters": c["isp"],
                                   "pixel_processing": RP.pixel_processing(c["isp"])}
                             for cn, c in cams.items()},
           "dressing": {k: len(v) for k, v in dressing.items()}}
    rj.write_text(json.dumps(res, indent=1, default=float))
    if diagnostic:
        (out / "diagnostics.json").write_text(json.dumps({"fingerprint": cfg["baseline"]["fingerprint"],
            "settings": cfg["baseline"]["settings"], "captures": captures, "complete": not preview,
            "videos": {cn: str((out / f"{cn}.mp4").relative_to(project)) for cn in cams}}, indent=2))
    return res


def _label_render(scn, obs, labels_of, tmp, cfg) -> np.ndarray:
    """Flat-colour Workbench render: 1 robot, 2 task object, 3 target, 0 everything else."""
    import bpy
    from ..backends import blender_scene as B
    from ..common import read_rgb
    colors = {0: (0, 0, 0, 1), 1: (1, 1, 1, 1), 2: (1, 0, 0, 1), 3: (0, 1, 0, 1), 4: (0, 0, 1, 1)}
    for o in scn.objects:
        if o.type in ("MESH", "CURVE"):
            o.color = colors[labels_of.get(o.name, labels_of.get(o.name.split(":")[0], 0))]
    eng = scn.render.engine
    scn.render.engine = "BLENDER_WORKBENCH"
    sh = scn.display.shading
    sh.light, sh.color_type = "FLAT", "OBJECT"
    sh.show_shadows = sh.show_cavity = sh.show_object_outline = False
    scn.display.render_aa = "OFF"
    vt = scn.view_settings.view_transform
    scn.view_settings.view_transform = "Standard"
    scn.render.film_transparent = True
    B.render_to(tmp)
    scn.render.film_transparent = False
    scn.view_settings.view_transform = vt
    scn.render.engine = eng
    im = read_rgb(tmp).astype(int)
    lab = np.zeros(im.shape[:2], np.uint8)
    lab[(im[..., 0] > 200) & (im[..., 1] > 200) & (im[..., 2] > 200)] = 1
    lab[(im[..., 0] > 200) & (im[..., 1] < 60)] = 2
    lab[(im[..., 1] > 200) & (im[..., 0] < 60)] = 3
    lab[(im[..., 2] > 200) & (im[..., 0] < 60) & (im[..., 1] < 60)] = 4
    return lab


def _worker_main(args: dict):
    from .. import synthkit as sy
    project = Path(args["project"])
    ctx = sy.Context(project, args["spec"])
    cache = {}
    done = []
    for eid in args["episodes"]:
        try:
            done.append(_render_episode(project, ctx, eid, args["cfg"], cache))
        except Exception as ex:                                # noqa: BLE001
            import traceback
            err = {"episode": eid, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-2000:]}
            (project / "synthesis" / "renders" / eid).mkdir(parents=True, exist_ok=True)
            (project / "synthesis" / "renders" / eid / "render_error.json").write_text(json.dumps(err, indent=1))
            done.append(err)
    print(json.dumps({"worker_done": len(done)}))


def _gpus(spec) -> list:
    from .. import config as C
    if spec != "auto":
        return [str(g) for g in spec]
    allowed = [str(g) for g in C.config().get("gpus", [])]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        ok = [l.split(",")[0].strip() for l in out.strip().splitlines()
              if float(l.split(",")[1]) < 40000 and float(l.split(",")[2]) < 60]
        ok = [g for g in ok if g in allowed] if allowed else ok
        return ok or allowed[:1] or ["0"]
    except Exception:                                          # noqa: BLE001
        return allowed[:1] or ["0"]


def run(project: Path, episodes="all", cameras: list | None = None, engine: str | None = None, samples: int | None = None,
        denoise: bool | None = None, motion_blur: float | None = None, preview_frames: list | None = None,
        res_scale: float = 1.0, check_frames: int = 4, gpus="auto", workers_per_gpu: int = 2, resume: bool = True,
        spec: str = "synthesis/spec.yaml", realism_override_reason: str | None = None,
        resolution: list | None = None, view_transform: str | None = None) -> dict:
    import yaml
    from ..common import contact_sheet, read_rgb, write_json, write_rgb
    from ..scene_ir import load
    from .. import render_pipeline as RP
    scene = load(project / "scene")
    fixed = RP.settings(scene, engine=engine, samples=samples, denoise=denoise, motion_blur=motion_blur)
    idx = yaml.safe_load((project / "synthesis" / "episodes.yaml").read_text())
    ids = ([idx["nominal"]] if idx.get("nominal") else []) + list(idx["episodes"])
    if episodes != "all":
        ids = [e for e in ids if e in set(episodes)]
    cfg = {**fixed, "cameras": cameras or ["head", "wrist_left", "wrist_right"], "preview_frames": preview_frames,
           "res_scale": res_scale, "check_frames": check_frames, "resume": resume}
    if resolution is not None:
        output_intrinsics(np.eye(3), 640, 480, resolution)
        cfg["resolution"] = list(resolution)
    if view_transform is not None:
        if view_transform not in ("Standard", "AgX"):
            raise ValueError("view_transform must be Standard or AgX")
        cfg["view_transform"] = view_transform
    snapshot = RP.baseline(project, scene, cfg)
    (project / "render").mkdir(exist_ok=True)
    write_json(snapshot, project / "render" / "realism_baseline.json")
    if realism_override_reason is not None and not realism_override_reason.strip():
        raise ValueError("realism_override_reason must document explicit user authorization")
    cfg["realism_review"] = RP.review(project, snapshot, cfg["cameras"],
                                     required=not realism_override_reason and not preview_frames
                                     and any(e != idx.get("nominal") for e in ids))
    if realism_override_reason:
        # Explicit user-requested rendering is not an independent realism pass.
        cfg["realism_review"]["user_authorized_render_override"] = realism_override_reason
    cfg["baseline"] = {k: snapshot[k] for k in ("fingerprint", "settings")}
    gl = _gpus(gpus)
    slots = [g for g in gl for _ in range(max(1, int(workers_per_gpu)))]
    shards = [ids[i::len(slots)] for i in range(len(slots))]
    logs = project / "synthesis" / "renders" / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    procs = []
    for k, (g, sh) in enumerate(zip(slots, shards)):
        if not sh:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        arg = json.dumps({"project": str(project), "spec": spec, "episodes": sh, "cfg": cfg})
        f = open(logs / f"worker_{k}_gpu{g}.log", "w")
        procs.append((subprocess.Popen([sys.executable, "-m", "any2sim.tools.synth_render", arg], env=env,
                                       stdout=f, stderr=subprocess.STDOUT), f))
    for p, f in procs:
        p.wait()
        f.close()
    res = {"gpus": slots, "episodes": len(ids), "wall_s": time.time() - t0, "rendered": [], "errors": []}
    for eid in ids:
        d = project / "synthesis" / "renders" / eid
        if (d / "render.json").exists():
            res["rendered"].append(json.loads((d / "render.json").read_text()))
        elif (d / "render_error.json").exists():
            res["errors"].append(json.loads((d / "render_error.json").read_text()))
        else:
            res["errors"].append({"episode": eid, "error": "no output (see synthesis/renders/_logs)"})
    r = res["rendered"]
    res["summary"] = {"ok": len(r), "failed": len(res["errors"]),
                      "seconds_per_episode_median": float(np.median([x["seconds"] for x in r])) if r else None,
                      "seconds_per_frame_median": float(np.median([x["seconds_per_frame"] for x in r])) if r else None}
    ims, labels = [], []
    for x in r[:6]:
        d = project / "synthesis" / "renders" / x["episode"] / "check"
        f = x["check_frames"][len(x["check_frames"]) // 2] if x["check_frames"] else None
        if f is None:
            continue
        for cn in cfg["cameras"]:
            p = d / f"{cn}_{f:05d}.png"
            if f is not None and p.exists():
                ims.append(read_rgb(p))
                labels.append(f"{x['episode']} {cn} f{f}")
    if ims:
        write_rgb(project / "evidence" / "synth_render_sheet.jpg", contact_sheet(ims, cols=len(cfg["cameras"]), width=420, labels=labels))
    write_json({**res, "rendered": [{k: v for k, v in x.items() if k != "sensor"} for x in r]},
               project / "evidence" / "synth_render.json")
    return {"summary": res["summary"], "errors": res["errors"][:5], "wall_s": res["wall_s"], "gpus": slots,
            "outputs": ["synthesis/renders", "evidence/synth_render.json", "evidence/synth_render_sheet.jpg"]}


if __name__ == "__main__":
    _worker_main(json.loads(sys.argv[1]))
