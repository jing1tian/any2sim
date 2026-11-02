"""Shared mesh / camera / mask loading for the GPU tools (dependency-light: numpy, trimesh, opencv)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_mesh(path):
    """(V, F, UV|None, FUV|None, texture image|None, trimesh) from an OBJ/GLB/PLY."""
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    uv = getattr(m.visual, "uv", None)
    tex = None
    mat = getattr(m.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is not None:
        tex = np.asarray(img.convert("RGB"))
    F = np.asarray(m.faces)
    return np.asarray(m.vertices, float), F, (np.asarray(uv, float) if uv is not None and len(uv) == len(m.vertices) else None), \
        (F if uv is not None else None), tex, m


def load_cameras(project: Path, spec="scene") -> dict:
    """{"K", "width", "height", "T_world_cam": (N,4,4)} from the Scene IR source camera ("scene") or a json file
    (project-relative) with the same keys."""
    if spec == "scene":
        from .scene_ir import camera_poses, load
        sc = load(project / "scene")
        c = sc["cameras"]["source_cam"]
        return {"K": np.asarray(c["K"], float), "width": c["width"], "height": c["height"],
                "T_world_cam": camera_poses(project / "scene", sc)}
    d = json.loads((project / spec).read_text())
    return {"K": np.asarray(d["K"], float), "width": d["width"], "height": d["height"],
            "T_world_cam": np.asarray(d["T_world_cam"], float).reshape(-1, 4, 4)}


def frame_mask(project: Path, spec: dict, frame: int) -> np.ndarray:
    """Binary mask of one frame from {"masks": dir} (frame%05d.png, >0) or {"label_maps": dir, "label": k} or
    {"label_maps": dir, "labels": [k, ...]}. Missing files give an empty mask."""
    from .common import read_mask
    d = project / (spec.get("masks") or spec.get("label_maps"))
    p = d / f"frame{frame:05d}.png"
    if not p.exists():
        return None
    m = read_mask(p)
    if "masks" in spec:
        return m > 0
    labs = spec.get("labels") or [spec["label"]]
    return np.isin(m, labs)


def any_mask(project: Path, specs: list[dict] | None, frame: int, shape, dilate_px: int = 0) -> np.ndarray:
    import cv2
    out = np.zeros(shape, bool)
    for s in specs or []:
        m = frame_mask(project, s, frame)
        if m is not None:
            out |= m
    if dilate_px:
        out = cv2.dilate(out.astype(np.uint8), np.ones((dilate_px, dilate_px), np.uint8)) > 0
    return out


def fill_holes(m: np.ndarray) -> np.ndarray:
    import cv2
    cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(np.zeros(m.shape, np.uint8), cn, -1, 1, -1) > 0


def visible_filled(obj_all, depth_all, k, obj_alone, hull_front_depth, movable_ids=()):
    """Silhouette of entity k as a tracked (solid) mask sees it: k rendered alone with openings filled, minus
    (a) other movable objects visible inside it (contents such as a pepper in a basket, which tracked masks
    exclude) and (b) other geometry in front of k's convex-hull front surface. Static geometry seen *through*
    an opening (a table under a slotted basket) is not an occluder."""
    filled = fill_holes(obj_alone == 0)
    if not filled.any():
        return filled
    other = (obj_all >= 0) & (obj_all != k)
    movable = other & np.isin(obj_all, list(movable_ids))
    in_front = other & (depth_all > 0) & (hull_front_depth > 0) & (depth_all < hull_front_depth - 1e-4)
    return filled & ~(movable | in_front)


def hull_front_depth(R, V_cam, K, W, H):
    """Depth of the front surface of the convex hull of camera-frame vertices V_cam (torch (N,3))."""
    import torch
    import trimesh
    h = trimesh.Trimesh(V_cam.detach().cpu().numpy(), process=False).convex_hull
    Vh = torch.tensor(h.vertices, dtype=torch.float32, device=V_cam.device)
    Fh = torch.tensor(h.faces, dtype=torch.int64, device=V_cam.device)
    _, d, _, _ = R.render_full([Vh], [Fh], K, W, H)
    return d


def boundary_stats(pred, gt, ignore):
    """Mean / P95 distance (px) between the outlines of pred and gt (outline pixels near ignored pixels skipped)."""
    import cv2
    def outline(m):
        m = m.astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0
    near = cv2.dilate(ignore.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    op, og = outline(pred) & ~near, outline(gt) & ~near
    if not op.any() or not og.any():
        return None, None
    dg = cv2.distanceTransform((~og).astype(np.uint8), cv2.DIST_L2, 3)
    dp = cv2.distanceTransform((~op).astype(np.uint8), cv2.DIST_L2, 3)
    d = np.r_[dg[op], dp[og]]
    return float(d.mean()), float(np.percentile(d, 95))


def edge_alignment(real_rgb, tri_map, depth, normals, region, ignore, max_px: float = 10.0):
    """Interior structure agreement: Canny edges of the real image vs geometric edges of the render (triangle-id
    changes with depth jumps or creases) inside `region`. Returns symmetric mean chamfer distance (px, capped) and
    the edge maps. Small = slots, rims, handles line up."""
    import cv2
    g = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY)
    e_real = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 120) > 0
    dz = np.zeros(depth.shape, bool)
    crease = 1 - (normals[:-1] * normals[1:]).sum(-1) > 0.13, 1 - (normals[:, :-1] * normals[:, 1:]).sum(-1) > 0.13
    dz[:-1] |= (np.abs(np.diff(depth, axis=0)) > 0.01 * np.maximum(depth[:-1], 1e-3)) | crease[0]
    dz[:, :-1] |= (np.abs(np.diff(depth, axis=1)) > 0.01 * np.maximum(depth[:, :-1], 1e-3)) | crease[1]
    e_rend = dz & (tri_map >= 0)
    reg = region & ~ignore
    er, en = e_real & reg, e_rend & reg
    if er.sum() < 20 or en.sum() < 20:
        return None, er, en
    dr_ = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
    dn_ = cv2.distanceTransform((~en).astype(np.uint8), cv2.DIST_L2, 3)
    d = 0.5 * (np.minimum(dr_[en], max_px).mean() + np.minimum(dn_[er], max_px).mean())
    return float(d), er, en


def edge_fscore(real_rgb, e_real, e_rend, depth, tol_px: float = 2.0, max_angle_deg: float = 30.0):
    """Structure agreement that dense patterns cannot game: an edge pixel matches when an edge of the other map lies
    within tol_px AND their gradient orientations agree within max_angle_deg. Returns precision (rendered edges
    explained by real ones), recall (real edges reproduced) and F-score."""
    import cv2
    if e_real.sum() < 20 or e_rend.sum() < 20:
        return None
    g = cv2.GaussianBlur(cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0)
    th_r = np.arctan2(cv2.Sobel(g, cv2.CV_32F, 0, 1), cv2.Sobel(g, cv2.CV_32F, 1, 0))
    dd = cv2.GaussianBlur(depth.astype(np.float32), (3, 3), 0)
    th_n = np.arctan2(cv2.Sobel(dd, cv2.CV_32F, 0, 1), cv2.Sobel(dd, cv2.CV_32F, 1, 0))
    def matched(src, src_th, dst, dst_th):
        ys, xs = np.nonzero(src)
        H, W = src.shape
        r = int(np.ceil(tol_px))
        ok = np.zeros(len(ys), bool)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx > tol_px * tol_px:
                    continue
                y2, x2 = np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)
                hit = dst[y2, x2]
                da = np.abs(((src_th[ys, xs] - dst_th[y2, x2]) + np.pi / 2) % np.pi - np.pi / 2)   # orientation, mod 180
                ok |= hit & (np.degrees(da) <= max_angle_deg)
        return ok.mean()
    p = float(matched(e_rend, th_n, e_real, th_r))
    r_ = float(matched(e_real, th_r, e_rend, th_n))
    return {"precision": p, "recall": r_, "f": 2 * p * r_ / max(p + r_, 1e-9)}
