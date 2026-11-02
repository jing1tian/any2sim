"""Project video frames onto a mesh's UV map: per-texel observed colour from every frame where the surface point is
visible (depth test against all given geometry), inside the object's mask, outside ignored pixels, facing the camera.

A measurement primitive: what to do with the observations (de-lighting, filling unseen texels, replacing them by a
procedural or synthesized material) is the caller's decision.

params:
  mesh: project-relative OBJ/GLB with UVs (object frame)
  poses: [{"T_world": 4x4, "frames": [ids]}]     object pose for those source frames (one entry per rest state)
  cameras: "scene" | project-relative json
  visible: {label_maps: dir, label: k} | {masks: dir}      pixel must be on this object
  ignore: [{masks: dir} | {label_maps: dir, labels: [..]}]   (hands, other occluders not modelled)
  occluders: [{"mesh": path, "T_world": 4x4, "frames": [ids] | null}]   modelled geometry for the depth test
  color_gains: null | project-relative json {"gains": [[r,g,b] per frame]}   linear per-frame camera response to undo
  size: [w, h] (default: mesh texture size or 1024)
  min_cos: 0.2, depth_tol_m: 0.005, erode_px: 2
  out_dir: project-relative (default work/texproj/<mesh stem>)
Writes <out_dir>/: observed.png (sRGB median of samples; unobserved = black), observed_lin.npy (H,W,3 linear,
NaN unobserved), count.png/.npy (samples per texel), coverage.png (texels used by the UV map),
position.npy / normal.npy (object-frame texel position/normal; for the caller's own computations), summary.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import frame_files, read_rgb, write_json, write_mask, write_rgb

ENV = "sam3d"
GPU = True


def _lin(x):
    x = np.asarray(x, np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def _bilinear(img, u, v):
    H, W = img.shape[:2]
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1] +
            (1 - fx) * fy * img[y0 + 1, x0] + fx * fy * img[y0 + 1, x0 + 1])


def run(project: Path, mesh: str, poses: list[dict], visible: dict, cameras="scene", ignore: list | None = None,
        occluders: list | None = None, color_gains: str | None = None, size: list | None = None,
        min_cos: float = 0.2, depth_tol_m: float = 0.005, erode_px: int = 2, out_dir: str | None = None) -> dict:
    import json
    import cv2
    import torch
    from ..meshio import any_mask, frame_mask, load_cameras, load_mesh
    from ..models.raster import Rasterizer
    dev = "cuda"
    R = Rasterizer()
    V, F, UV, FUV, tex, _ = load_mesh(project / mesh)
    if UV is None:
        raise ValueError("mesh has no UVs (use meshkit.uv_unwrap first)")
    w, h = size or ((tex.shape[1], tex.shape[0]) if tex is not None else (1024, 1024))
    Vt = torch.tensor(V, dtype=torch.float32, device=dev)
    Ft = torch.tensor(F, dtype=torch.int64, device=dev)
    fn = torch.nn.functional.normalize(torch.cross(Vt[Ft[:, 1]] - Vt[Ft[:, 0]], Vt[Ft[:, 2]] - Vt[Ft[:, 0]], dim=1), dim=1)
    vn = torch.zeros_like(Vt).index_add_(0, Ft.reshape(-1), fn.repeat_interleave(3, 0))
    vn = torch.nn.functional.normalize(vn, dim=1)
    cov, (pos, nrm) = R.rasterize_uv(Vt, F, UV, FUV, w, h, [Vt, vn])
    cov = cov & np.isfinite(pos).all(-1)
    P = pos[cov].astype(np.float64)
    N = nrm[cov].astype(np.float64)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
    cams = load_cameras(project, cameras)
    K, W, H = cams["K"], cams["width"], cams["height"]
    files = frame_files(project / "work" / "frames")
    gains = np.asarray(json.loads((project / color_gains).read_text())["gains"], np.float32) if color_gains else None
    occ_meshes = [(load_mesh(project / o["mesh"])[:2], np.asarray(o["T_world"], float), o.get("frames")) for o in occluders or []]
    ke = np.ones((2 * erode_px + 1,) * 2, np.uint8)
    samples = []
    for ps in poses:
        T = np.asarray(ps["T_world"], float)
        Pw = P @ T[:3, :3].T + T[:3, 3]
        Nw = N @ T[:3, :3].T
        Vw = torch.tensor(V @ T[:3, :3].T + T[:3, 3], dtype=torch.float32, device=dev)
        for f in ps["frames"]:
            Twc = cams["T_world_cam"][f] if len(cams["T_world_cam"]) > 1 else cams["T_world_cam"][0]
            Tcw = np.linalg.inv(Twc)
            Tt = torch.tensor(Tcw, dtype=torch.float32, device=dev)
            Vs, Fs = [Vw @ Tt[:3, :3].T + Tt[:3, 3]], [Ft]
            for (Vo, Fo), To, fr in occ_meshes:
                if fr is None or f in fr:
                    Vs.append(torch.tensor((Vo @ To[:3, :3].T + To[:3, 3]) @ Tcw[:3, :3].T + Tcw[:3, 3],
                                           dtype=torch.float32, device=dev))
                    Fs.append(torch.tensor(Fo, dtype=torch.int64, device=dev))
            with torch.no_grad():
                _, Z = R.render_ids(Vs, Fs, K, W, H)
            Z = Z.cpu().numpy()
            pc = Pw @ Tcw[:3, :3].T + Tcw[:3, 3]
            z = pc[:, 2]
            u = K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
            v = K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
            ins = (z > 0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
            ui, vi = np.clip(u.round().astype(int), 0, W - 1), np.clip(v.round().astype(int), 0, H - 1)
            zb = Z[vi, ui]
            vis = ins & (zb > 0) & (np.abs(z - zb) < depth_tol_m + 0.01 * z)
            view = Twc[:3, 3] - Pw
            view /= np.linalg.norm(view, axis=1, keepdims=True)
            cosv = (Nw * view).sum(1)
            m = frame_mask(project, visible, f)
            if m is None:
                continue
            m = cv2.erode(m.astype(np.uint8), ke) > 0
            ig = any_mask(project, ignore, f, (H, W), dilate_px=7)
            ok = vis & (cosv > min_cos) & m[vi, ui] & ~ig[vi, ui]
            img = _lin(read_rgb(files[f]))
            if gains is not None:
                img = img / gains[f]
            col = np.full((len(P), 3), np.nan, np.float32)
            if ok.any():
                col[ok] = _bilinear(img, u[ok], v[ok])
            samples.append(col)
    S = np.stack(samples) if samples else np.full((1, len(P), 3), np.nan, np.float32)
    cnt = np.isfinite(S[..., 0]).sum(0)
    med = np.full((len(P), 3), np.nan, np.float32)
    if (cnt > 0).any():
        med[cnt > 0] = np.nanmedian(S[:, cnt > 0], axis=0)
    out = project / (out_dir or f"work/texproj/{Path(mesh).stem}")
    out.mkdir(parents=True, exist_ok=True)
    obs = np.full((h, w, 3), np.nan, np.float32)
    obs[cov] = med
    count = np.zeros((h, w), np.int32)
    count[cov] = cnt
    np.save(out / "observed_lin.npy", obs)
    np.save(out / "count.npy", count)
    posm = np.full((h, w, 3), np.nan, np.float32)
    posm[cov] = P
    nrmm = np.zeros((h, w, 3), np.float32)
    nrmm[cov] = N
    np.save(out / "position.npy", posm)
    np.save(out / "normal.npy", nrmm)
    write_rgb(out / "observed.png", (np.nan_to_num(_srgb(np.nan_to_num(obs))) * 255).astype(np.uint8))
    write_rgb(out / "count.png", np.stack([np.clip(count * 25, 0, 255).astype(np.uint8)] * 3, -1))
    write_mask(out / "coverage.png", cov)
    s = {"texels": int(cov.sum()), "observed_fraction": float((cnt > 0).mean()) if len(cnt) else 0.0,
         "samples_per_observed_texel": float(cnt[cnt > 0].mean()) if (cnt > 0).any() else 0.0,
         "frames": int(len(samples)), "size": [w, h]}
    write_json(s, out / "summary.json")
    return {**s, "outputs": [str(out.relative_to(project))]}
