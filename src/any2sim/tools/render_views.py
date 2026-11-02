"""Fast geometry renders (nvdiffrast): object ids, silhouettes, depth, normals, outline overlays on real frames.

For checking geometry against the video at any frame, or rendering from any camera. Not photo-realistic (use
blender_render for appearance).

params:
  meshes: "scene" (all Scene IR entities at their pose for each source frame; rigid entities hidden while moving) |
          [{"name", "mesh": path, "T_world": 4x4 | "per_frame": {frame: 4x4}}]   (paths project-relative)
  frames: [source frame ids]          (cameras from the Scene IR source camera, or `cameras`)
  cameras: "scene" | project-relative json {K, width, height, T_world_cam: [N x 4x4]}
  extra_cams: [{"name", "K", "width", "height", "T_world_cam"}]     (renders from arbitrary cameras)
  out_dir: project-relative (default work/render_views)
  overlay: true                        outline of every mesh drawn on the real frame
Writes <out_dir>/<cam or frame>_{ids.png, depth.npz, normal.png, overlay.jpg}, index.json (id -> name).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import PALETTE, frame_files, read_rgb, write_json, write_rgb

ENV = "sam3d"
GPU = True


def _mesh_set(project, meshes, frame):
    from ..meshio import load_mesh
    from ..scene_ir import load, pose_at
    out = []
    if meshes == "scene":
        sc = load(project / "scene")
        for name, e in sc["entities"].items():
            T = pose_at(e, frame)
            if T is None:
                continue
            V, F, *_ = load_mesh(project / "scene" / e["geometry"]["visual"]["obj_uri"])
            out.append((name, V, F, T))
    else:
        for m in meshes:
            T = m.get("per_frame", {}).get(str(frame), m.get("T_world", np.eye(4))) if frame is not None else m.get("T_world", np.eye(4))
            V, F, *_ = load_mesh(project / m["mesh"])
            out.append((m["name"], V, F, np.asarray(T, float)))
    return out


def run(project: Path, frames: list | None = None, meshes="scene", cameras="scene", extra_cams: list | None = None,
        out_dir: str = "work/render_views", overlay: bool = True) -> dict:
    import cv2
    import torch
    from ..meshio import load_cameras
    from ..models.raster import Rasterizer
    R = Rasterizer()
    out = project / out_dir
    out.mkdir(parents=True, exist_ok=True)
    files = frame_files(project / "work" / "frames")
    jobs = []
    if frames:
        cams = load_cameras(project, cameras)
        for f in frames:
            T = cams["T_world_cam"][int(f)] if len(cams["T_world_cam"]) > 1 else cams["T_world_cam"][0]
            jobs.append((f"frame{int(f):05d}", int(f), cams["K"], cams["width"], cams["height"], T))
    for c in extra_cams or []:
        jobs.append((c["name"], None, np.asarray(c["K"], float), c["width"], c["height"], np.asarray(c["T_world_cam"], float)))
    index = {}
    for name, f, K, W, H, Twc in jobs:
        ms = _mesh_set(project, meshes, f)
        if not ms:
            continue
        Tcw = np.linalg.inv(Twc)
        Vc = [torch.tensor((V @ T[:3, :3].T + T[:3, 3]) @ Tcw[:3, :3].T + Tcw[:3, 3], dtype=torch.float32, device="cuda")
              for _, V, _, T in ms]
        Fs = [torch.tensor(F, dtype=torch.int64, device="cuda") for _, _, F, _ in ms]
        obj, depth, nrm, _ = R.render_full(Vc, Fs, K, W, H)
        cv2.imwrite(str(out / f"{name}_ids.png"), (obj + 1).astype(np.uint8))
        np.savez_compressed(out / f"{name}_depth.npz", depth=depth.astype(np.float32))
        write_rgb(out / f"{name}_normal.png", ((nrm * 0.5 + 0.5) * 255 * (obj >= 0)[..., None]).astype(np.uint8))
        index = {str(k + 1): n for k, (n, *_rest) in enumerate(ms)}
        if overlay and f is not None:
            im = read_rgb(files[f])
            for k in range(len(ms)):
                cn, _ = cv2.findContours((obj == k).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cn, -1, tuple(int(x) for x in PALETTE[(k + 1) % len(PALETTE)]), 1)
            write_rgb(out / f"{name}_overlay.jpg", im)
        write_json({"ids": index, "camera": {"K": np.asarray(K).tolist(), "width": W, "height": H,
                                            "T_world_cam": np.asarray(Twc).tolist()}}, out / f"{name}_index.json")
    return {"views": [j[0] for j in jobs], "outputs": [out_dir]}
