"""Generate a textured 3-D mesh of one object from one image + mask (SAM 3D Objects), placed in that camera.

A generative draft: the visible side follows the image, hidden parts are invented, fine structure (slots, holes,
handles, thin parts) is often wrong. Check it against the video and refine or rebuild it.

params:
  frame: frame id (or image path relative to the project)
  mask: project-relative png of the object in that image
  pointmap: optional project-relative npz with "points" (H,W,3, OpenCV camera frame; e.g. a moge_geometry output):
            the layout is then in that metric camera frame
  out_dir: project-relative folder (default work/sam3d/<mask stem>)
  seed: 42
Writes <out_dir>/sam3d.glb (raw), mesh_cam.glb (textured mesh in the OpenCV camera frame of `frame`),
layout.json (quaternion/translation/scale of SAM 3D, silhouette IoU against the mask), overlay.png.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from ..common import frame_files, read_mask, read_rgb, write_json, write_rgb

ENV = "sam3d"
GPU = True
P3D_FROM_CV = np.diag([-1.0, -1.0, 1.0])
GLB_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # inverse of to_glb's export rotation


def silhouette(K, V_cam, F, H, W):
    import cv2
    z = V_cam[:, 2]
    uv = V_cam[:, :2] / np.maximum(z, 1e-6)[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    m = np.zeros((H, W), np.uint8)
    tri = np.round(uv[F] * 4).astype(np.int32)
    for t in tri[(z[F] > 0).all(1)]:
        cv2.fillConvexPoly(m, t, 1, lineType=cv2.LINE_8, shift=2)
    return m > 0


def run(project: Path, frame, mask: str, pointmap: str | None = None, out_dir: str | None = None, seed: int = 42,
        K: list | None = None) -> dict:
    """K: intrinsics used for the overlay/IoU (default: the point map's K, else a 60 deg guess)."""
    import cv2
    import torch
    import trimesh
    from scipy.spatial.transform import Rotation
    from ..config import model_path
    files = frame_files(project / "work" / "frames")
    img_path = files[int(frame)] if isinstance(frame, int) or str(frame).isdigit() else project / frame
    rgb = read_rgb(img_path)
    H, W = rgb.shape[:2]
    m = read_mask(project / mask) > 0
    out = project / (out_dir or f"work/sam3d/{Path(mask).stem}")
    out.mkdir(parents=True, exist_ok=True)
    repo = model_path("sam3d_repo")
    sys.path.insert(0, str(repo / "notebook"))
    os.environ.setdefault("CONDA_PREFIX", os.environ.get("CUDA_HOME", ""))
    from inference import Inference
    inf = Inference(str(repo / "checkpoints" / "hf" / "pipeline.yaml"), compile=False)
    pm, Kp = None, None
    if pointmap:
        d = np.load(project / pointmap)
        P = d["points"].astype(np.float32)
        P[~np.isfinite(P).all(-1)] = np.nan
        pm = torch.from_numpy(P @ P3D_FROM_CV.T.astype(np.float32))
        Kp = d["K"] if "K" in d else None
    o = inf._pipeline.run(inf.merge_mask_to_rgba(rgb, m.astype(np.uint8)), None, int(seed), stage1_only=False,
                          with_mesh_postprocess=True, with_texture_baking=True, with_layout_postprocess=False,
                          use_vertex_color=False, pointmap=pm)
    o["glb"].export(str(out / "sam3d.glb"))
    q = np.asarray(o["rotation"].detach().cpu()).reshape(-1)
    t = np.asarray(o["translation"].detach().cpu()).reshape(-1)
    s = np.asarray(o["scale"].detach().cpu()).reshape(-1)
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    glb = trimesh.load(out / "sam3d.glb")
    mesh = glb.dump(concatenate=True) if isinstance(glb, trimesh.Scene) else glb
    # SAM 3D: V_glb = V_local @ M (y-up export); layout is a PyTorch3D row-vector transform p_cam = s p @ R + t
    Vl = np.asarray(mesh.vertices) @ GLB_YUP_TO_ZUP.T
    Vc = ((Vl * s) @ R + t) @ P3D_FROM_CV.T
    mc = mesh.copy()
    mc.vertices = Vc
    mc.export(str(out / "mesh_cam.glb"))
    Kv = np.asarray(K, float) if K is not None else (np.asarray(Kp, float) if Kp is not None else
                                                     np.array([[W / (2 * np.tan(np.radians(30))), 0, W / 2],
                                                               [0, W / (2 * np.tan(np.radians(30))), H / 2], [0, 0, 1]]))
    sil = silhouette(Kv, Vc, np.asarray(mesh.faces), H, W)
    iou = float((sil & m).sum() / max((sil | m).sum(), 1))
    ov = rgb.copy()
    ov[sil] = (0.5 * ov[sil] + [0, 127, 127]).astype(np.uint8)
    cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cn, -1, (255, 0, 255), 2)
    write_rgb(out / "overlay.png", ov)
    lay = {"frame": str(frame), "pointmap": pointmap, "quat_wxyz_local_to_p3dcam": q.tolist(),
           "translation_p3dcam": t.tolist(), "scale": s.tolist(), "silhouette_iou": iou,
           "extent_cam_m": (Vc.max(0) - Vc.min(0)).tolist(), "n_faces": int(len(mesh.faces)),
           "canonical_up_in_camera": (P3D_FROM_CV @ R.T @ np.array([0, 0, 1.0])).tolist()}
    write_json(lay, out / "layout.json")
    return {**lay, "outputs": [str(out.relative_to(project))]}
