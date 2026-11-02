"""Per-frame camera response (auto white balance / exposure) relative to the reference frame.

Consumer cameras change white balance and exposure during a clip; that is a property of the camera's processing,
not of the scene. For every frame, static background pixels (table and backdrop, far from hands and tracked
objects) are reprojected into the reference view with the estimated cameras and a diagonal linear-RGB gain
g_f = median(frame / reference) is measured; gains are median-filtered over time. Object pixels are never used, so
the response cannot absorb object appearance errors.
Consumers: build_background and solve_albedo divide samples by g_f; evaluate_render reports metrics raw and with
the render multiplied by g_f (declared separately).
Output: work/camera/response.json {gains: [[r, g, b] per frame], ...}, evidence/camera_response.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import frame_files, read_json, read_mask, read_rgb, write_json

ENV = "base"
GPU = False


def srgb_to_lin(x):
    x = np.asarray(x, np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load(project: Path, n: int) -> np.ndarray:
    p = project / "work" / "camera" / "response.json"
    return np.asarray(read_json(p)["gains"], np.float32) if p.exists() else np.ones((n, 3), np.float32)


def run(project: Path, dilate_px: int = 61, smooth: int = 5) -> dict:
    import cv2
    from ..geom3d import pixel_rays
    cams = read_json(project / "work" / "world" / "cameras.json")
    world = read_json(project / "work" / "world" / "world.json")
    K, Ts = np.asarray(cams["K"]), np.asarray(cams["T_world_cam"])
    ref = world["ref_frame"]
    files = frame_files(project / "work" / "frames")
    mdir = project / "work" / "masks"
    H, W = cams["height"], cams["width"]
    kd = np.ones((dilate_px, dilate_px), np.uint8)
    static = ((read_mask(mdir / "static" / "table.png") > 0) | (read_mask(mdir / "static" / "backdrop.png") > 0))
    occ_ref = cv2.dilate(((read_mask(mdir / "dynamic" / f"frame{ref:05d}.png") > 0) |
                          (read_mask(mdir / "objects" / f"frame{ref:05d}.png") > 0)).astype(np.uint8), kd) > 0
    pw = np.load(project / "work" / "world" / "points_ref_world.npz")
    Pw = pw["points"].astype(np.float64)
    tbl = read_mask(mdir / "static" / "table.png") > 0
    ys, xs = np.nonzero(tbl)
    d = pixel_rays(K, np.c_[xs, ys]) @ Ts[ref][:3, :3].T
    t = -Ts[ref][2, 3] / d[:, 2]
    Pw[ys, xs] = Ts[ref][:3, 3] + t[:, None] * d
    sel = static & ~occ_ref & np.isfinite(Pw).all(-1)
    yy, xx = np.nonzero(sel)
    yy, xx = yy[::7], xx[::7]
    X = Pw[yy, xx]
    ref_lin = srgb_to_lin(read_rgb(files[ref])[yy, xx])
    gains, used = [], []
    for i, f in enumerate(files):
        Tcw = np.linalg.inv(Ts[i])
        pc = X @ Tcw[:3, :3].T + Tcw[:3, 3]
        u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
        v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
        ok = (pc[:, 2] > 0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        occ = cv2.dilate(((read_mask(mdir / "dynamic" / f"frame{i:05d}.png") > 0) |
                          (read_mask(mdir / "objects" / f"frame{i:05d}.png") > 0)).astype(np.uint8), kd) > 0
        ui, vi = u.round().astype(int).clip(0, W - 1), v.round().astype(int).clip(0, H - 1)
        ok &= ~occ[vi, ui]
        im = srgb_to_lin(read_rgb(f)[vi[ok], ui[ok]])
        r = im / np.maximum(ref_lin[ok], 1e-3)
        good = (ref_lin[ok] > 0.01).all(1) & (im > 0.01).all(1) & (im < 0.98).all(1)
        gains.append(np.median(r[good], 0) if good.sum() > 200 else np.array([np.nan] * 3))
        used.append(int(good.sum()))
    G = np.asarray(gains)
    for c in range(3):                                    # fill gaps, then median-filter over time
        bad = ~np.isfinite(G[:, c])
        if bad.any() and (~bad).any():
            G[bad, c] = np.interp(np.nonzero(bad)[0], np.nonzero(~bad)[0], G[~bad, c])
    k = smooth // 2
    Gs = np.stack([np.median(G[max(0, i - k):i + k + 1], 0) for i in range(len(G))])
    out = project / "work" / "camera"
    out.mkdir(parents=True, exist_ok=True)
    write_json({"reference_frame": ref, "space": "linear sRGB, diagonal gain relative to the reference frame",
                "gains": Gs.tolist(), "raw_gains": G.tolist(), "pixels_used": used,
                "range": {"min": Gs.min(0).tolist(), "max": Gs.max(0).tolist()}}, out / "response.json")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    for c, col in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
        ax.plot(Gs[:, c], color=col, label="RGB"[c])
    ax.set_xlabel("frame")
    ax.set_ylabel("gain vs reference")
    ax.legend()
    ax.set_title("camera response (static background)")
    fig.tight_layout()
    fig.savefig(project / "evidence" / "camera_response.png", dpi=110)
    return {"gain_min": Gs.min(0).tolist(), "gain_max": Gs.max(0).tolist(), "min_pixels": int(min(used)),
            "outputs": ["work/camera/response.json", "evidence/camera_response.png"]}
