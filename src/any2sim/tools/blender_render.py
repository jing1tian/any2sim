"""Photo-realistic renders of the Scene IR with Blender Cycles, and the deliverable .blend.

Everything comes from scene/scene.yaml: meshes, materials (principled/emission, textures, tiling), lights, cameras,
rigid states (an entity is shown at the rest pose valid at a frame, hidden while moving), robots (bundle visuals,
posed by FK) and the motion named by render.motion (robot joints and object trajectories at every video frame).

params:
  frames: [source frame ids] | "states" | "all"  ("states": first/middle/last frame of every rest state)
  step: 1                              with frames "all": every step-th frame
  video: false                         also write <out_dir>/video.mp4 (and side_by_side.mp4 with the source frames)
  novel: "auto" | [{"name", "eye": [x,y,z], "target": [x,y,z]}] | null    review cameras around the scene
  turntables: [entity names]           each entity alone on a neutral background, 8 views + a top view
  background_only: false               also render the source frames with all rigid entities hidden
                                       (shows what the static scene looks like where objects/hands were)
  camera_background: null | [r,g,b]    linear colour seen by camera rays that miss everything
  samples: 128, res_scale: 1.0
  camera_model: true                   apply the source camera's fixed image model (including render.camera_isp override,
                                       camera_calibrate) to src_* frames; raw renders are kept in <out_dir>/raw
  out_dir: project-relative (default blender/renders)
  save_blend: true                     blender/scene.blend (timeline = video, keyed camera/objects); reopened and
                                       re-rendered to prove the delivered file reproduces the in-memory scene
Writes src_%05d.png, bg_%05d.png, novel_<name>.png, turntable_<entity>_<k>.png, render.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import read_rgb, write_json

ENV = "render"
GPU = True


def run(project: Path, frames="states", novel="auto", turntables: list | None = None, background_only: bool = False,
        camera_background: list | None = None, samples: int | None = None, res_scale: float = 1.0,
        out_dir: str = "blender/renders", save_blend: bool = True, step: int = 1, video: bool = False,
        camera_model: bool = True, denoise: bool | None = None, motion_blur: float | None = None) -> dict:
    import bpy
    from ..backends import blender_scene as B
    from ..scene_ir import camera_poses, load, pose_at
    scene = load(project / "scene")
    from .. import render_pipeline as RP
    cfg = RP.settings(scene, engine="cycles", samples=samples, denoise=denoise, motion_blur=motion_blur)
    samples = int(cfg["samples"])
    poses = camera_poses(project / "scene", scene)
    if len(poses) == 1:
        # fixed camera: one pose for every video frame, so rest states still apply per frame
        from ..common import frame_files
        n_video = len(frame_files(project / "work" / "frames"))
        n_states = 1 + max([st["frames"][1] for e in scene["entities"].values()
                            for st in (e.get("states") or {}).values()] or [0])
        poses = np.repeat(poses, max(n_video, n_states), axis=0)
    rdir = project / out_dir
    rdir.mkdir(parents=True, exist_ok=True)
    B.reset()
    obs, cam, dev = B.build(project, scene, samples=samples)
    robots = B.add_robots(project, scene)
    rmot = (scene.get("render") or {}).get("motion")
    motion = None
    if rmot:
        from ..scene_ir import load_motion
        motion = load_motion(project / "scene", scene, rmot)
        n_m = int(np.floor(motion["frame"][-1])) + 1
        if len(poses) < n_m:
            poses = np.concatenate([poses, np.repeat(poses[-1:], n_m - len(poses), axis=0)])
    elif robots:
        B.pose_robots(robots)
    sc = bpy.context.scene
    RP.apply_settings(sc, cfg, animated=motion is not None)
    sc.render.resolution_percentage = int(100 * res_scale)
    if camera_background is not None:
        B.set_camera_background(tuple(camera_background))
    n = len(poses)
    if n > 1:
        B.keyframe_timeline(scene, obs, cam, poses)
    if motion is not None:
        B.keyframe_motion(scene, obs, robots, motion, n, scene["motions"][rmot].get("robot"))
    if frames == "all":
        frames = list(range(0, n, step))
    elif frames == "states":
        fs = {0}
        for e in scene["entities"].values():
            for st in (e.get("states") or {}).values():
                a, b = st["frames"]
                fs |= {a, (a + b) // 2, b}
        frames = sorted(f for f in fs if f < max(n, 1))
    out = {"device": dev, "src": [], "bg": [], "novel": [], "turntables": {}}
    for f in frames or []:
        if n > 1:
            sc.frame_set(int(f))
        out["src"].append(B.render_to(rdir / f"src_{int(f):05d}.png"))
    from .. import camera_isp as ISP
    isp = ISP.for_camera(scene, "source_cam") if camera_model else {}
    if isp and out["src"]:
        import shutil
        from .. import camera_isp as ISP
        from ..common import write_rgb
        (rdir / "raw").mkdir(exist_ok=True)
        for f_, pth in zip(frames, out["src"]):
            shutil.copy(pth, rdir / "raw" / Path(pth).name)
            write_rgb(pth, ISP.apply(read_rgb(pth), isp, key=("blender_render", int(f_))))
    rigid = [k for k, e in scene["entities"].items() if e.get("capability") == "rigid"]
    if background_only and frames:
        for k in rigid:
            if obs[k].animation_data:
                obs[k].animation_data_clear()
            obs[k].hide_render = True
        for f in frames:
            if n > 1:
                sc.frame_set(int(f))
            for k in rigid:
                obs[k].hide_render = True
            out["bg"].append(B.render_to(rdir / f"bg_{int(f):05d}.png"))
    # review cameras use their own (unkeyed) camera object
    rc = B.add_camera("review_cam", np.asarray(scene["cameras"]["source_cam"]["K"]),
                      scene["cameras"]["source_cam"]["width"], scene["cameras"]["source_cam"]["height"])
    sc.camera = rc
    if video and out["src"]:
        from ..common import frame_files, write_video
        ims = [read_rgb(p) for p in out["src"]]
        write_video(ims, rdir / "video.mp4", fps=float(scene["cameras"]["source_cam"].get("fps") or 20) / max(step, 1),
                    bitrate_kbps=isp.get("bitrate_kbps", 0) * res_scale ** 2)
        src = frame_files(project / "work" / "frames")
        import cv2
        sbs = [np.hstack([cv2.resize(read_rgb(src[int(f)]), (im.shape[1], im.shape[0])), im])
               for f, im in zip(frames, ims) if int(f) < len(src)]
        if sbs:
            write_video(sbs, rdir / "side_by_side.mp4", fps=float(scene["cameras"]["source_cam"].get("fps") or 20) / max(step, 1))
        out["video"] = [str(rdir / "video.mp4"), str(rdir / "side_by_side.mp4")]
    if robots:
        for R in robots.values():
            for e in R["links"].values():
                if e.animation_data:
                    e.animation_data_clear()
        B.pose_robots(robots)
    for k in rigid:                                       # initial state for review views
        if obs[k].animation_data:
            obs[k].animation_data_clear()
        T = pose_at(scene["entities"][k], None)
        obs[k].hide_render = False
        B.set_matrix(obs[k], T)
    pts = [pose_at(scene["entities"][k], None)[:3, 3] for k in rigid] or [np.zeros(3)]
    tgt = np.mean(pts, 0)
    cam0 = poses[0][:3, 3]
    if novel == "auto":
        d = np.r_[cam0[:2] - tgt[:2], 0]
        d /= np.linalg.norm(d) + 1e-9
        side = np.r_[-d[1], d[0], 0]
        novel = [{"name": "top", "eye": (tgt + [0.001, 0.001, 1.1]).tolist(), "target": tgt.tolist()},
                 {"name": "front_low", "eye": (tgt + 0.6 * d + [0, 0, 0.25]).tolist(), "target": tgt.tolist()},
                 {"name": "left", "eye": (tgt + 0.6 * side + [0, 0, 0.35]).tolist(), "target": tgt.tolist()},
                 {"name": "right", "eye": (tgt - 0.6 * side + [0, 0, 0.35]).tolist(), "target": tgt.tolist()},
                 {"name": "back", "eye": (tgt - 0.6 * d + [0, 0, 0.45]).tolist(), "target": tgt.tolist()}]
    for v in novel or []:
        up = (0, 1, 0) if abs(np.asarray(v["eye"])[2] - np.asarray(v["target"])[2]) > 0.9 * np.linalg.norm(
            np.asarray(v["eye"]) - np.asarray(v["target"])) else (0, 0, 1)
        B.set_cv_pose(rc, B.look_at(v["eye"], v["target"], up=up))
        out["novel"].append(B.render_to(rdir / f"novel_{v['name']}.png"))
    for ent in turntables or []:
        others = {k: o.hide_render for k, o in obs.items()}
        for k, o in obs.items():
            o.hide_render = k != ent
        e = scene["entities"][ent]
        T = pose_at(e, None)
        Vb = np.array([list(v.co) for v in obs[ent].data.vertices])
        c = T[:3, :3] @ ((Vb.min(0) + Vb.max(0)) / 2) + T[:3, 3]
        r = 2.2 * np.linalg.norm(Vb.max(0) - Vb.min(0)) / 2
        files = []
        for k in range(8):
            a = 2 * np.pi * k / 8
            eye = c + [r * np.cos(a), r * np.sin(a), 0.5 * r]
            B.set_cv_pose(rc, B.look_at(eye, c))
            files.append(B.render_to(rdir / f"turntable_{ent}_{k}.png"))
        B.set_cv_pose(rc, B.look_at(c + [0.001, 0.001, 1.3 * r], c, up=(0, 1, 0)))
        files.append(B.render_to(rdir / f"turntable_{ent}_top.png"))
        out["turntables"][ent] = files
        for k, o in obs.items():
            o.hide_render = others[k]
    sc.camera = cam
    res = {"device": dev, "samples": samples, "frames": frames, "renders": out,
           "render_settings": cfg, "camera_model": {"enabled": camera_model, "parameters": isp,
           "pixel_processing": RP.pixel_processing(isp)},
           "realism_status": "unreviewed", "raw_dir": str(rdir / "raw") if isp else out_dir}
    if save_blend:
        bpy.data.objects.remove(rc)
        if n > 1:
            B.keyframe_timeline(scene, obs, cam, poses)
        if motion is not None:
            B.keyframe_motion(scene, obs, robots, motion, n, scene["motions"][rmot].get("robot"))
        blend = project / "blender" / "scene.blend"
        blend.parent.mkdir(parents=True, exist_ok=True)
        f0 = int(frames[0]) if frames else 0
        if n > 1:
            sc.frame_set(f0)
        ref = B.render_to(rdir / f"_prereopen_{f0:05d}.png")
        for im in list(bpy.data.images):              # pack what exists; drop dangling references
            if im.source == "FILE" and im.filepath and not Path(bpy.path.abspath(im.filepath)).exists():
                bpy.data.images.remove(im)
            elif im.source == "FILE" and im.filepath and im.packed_file is None:
                im.pack()
        bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=True)
        bpy.ops.wm.open_mainfile(filepath=str(blend))
        B.setup_cycles(samples, scene["cameras"]["source_cam"]["width"], scene["cameras"]["source_cam"]["height"])
        RP.apply_settings(bpy.context.scene, cfg, animated=motion is not None)
        bpy.context.scene.render.resolution_percentage = int(100 * res_scale)
        if n > 1:
            bpy.context.scene.frame_set(f0)
        chk = B.render_to(rdir / f"_reopen_{f0:05d}.png")
        d = np.abs(read_rgb(chk).astype(int) - read_rgb(ref).astype(int))
        res["blend"] = str(blend.relative_to(project))
        res["reopen_check"] = {"frame": f0, "mean_abs_diff": float(d.mean()), "p99_abs_diff": float(np.percentile(d, 99))}
    write_json(res, rdir / "render.json")
    return {**{k: v for k, v in res.items() if k != "renders"}, "n_renders": sum(len(v) if isinstance(v, list) else
            sum(len(x) for x in v.values()) for v in out.values() if not isinstance(v, str)), "outputs": [out_dir]}
