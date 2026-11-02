"""Static background from the video: clean plate, textured table top and backdrop meshes.

1. clean plate in the reference view: reference pixels (table: exact plane points; elsewhere MoGe-2 points) are
   reprojected into every k-th frame with the estimated cameras, occluders (dynamic + tracked objects, dilated)
   are masked, and the per-pixel median of the remaining observations is taken. Pixels never seen uncovered are
   inpainted and flagged (`plate_observed.png`).
2. table: a slab whose top is the table plane (z = 0). Visible outline from fit_world; edges on the image border
   are extended by `extend_unseen_m` (completed, not observed). Top texture: orthographic bake of the plate at
   `texel_m`; unobserved texels get the observed texture statistics (flagged in table_observed.png).
3. backdrop: depth-grid mesh of the reference frame's backdrop pixels (MoGe-2, world frame), UVs = reference
   pixel coordinates, texture = plate. Visual only (no collision).
Outputs: scene/assets/background/{table.obj, table_top.png, backdrop.obj, plate.png}, work/background/*.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import contact_sheet, frame_files, read_json, read_mask, read_rgb, write_json, write_mask, write_rgb
from ..geom3d import project as project_points
from ..geom3d import inv

ENV = "base"
GPU = False


def _write_obj(path, V, F, UV=None, mtl_name=None, texture=None, FT=None):
    """OBJ writer; FT: per-face texture-coordinate indices when they differ from the vertex indices."""
    path = Path(path)
    lines = []
    if mtl_name:
        (path.with_suffix(".mtl")).write_text(
            f"newmtl {mtl_name}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\n" + (f"map_Kd {texture}\n" if texture else ""))
        lines += [f"mtllib {path.with_suffix('.mtl').name}", f"usemtl {mtl_name}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V]
    if UV is not None:
        lines += [f"vt {u:.6f} {v:.6f}" for u, v in UV]
        FT = F if FT is None else FT
        lines += [f"f {a + 1}/{ta + 1} {b + 1}/{tb + 1} {c + 1}/{tc + 1}" for (a, b, c), (ta, tb, tc) in zip(F, FT)]
    else:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in F]
    path.write_text("\n".join(lines) + "\n")


def push_pull_fill(img, known):
    """Smooth hole filling: pyramid of known-weighted colours (push), then coarse-to-fine filling (pull).
    No streaks; holes take the low-frequency colour of their surroundings."""
    import cv2
    levels = []
    c = img.astype(np.float32) * known[..., None]
    w = known.astype(np.float32)
    while min(w.shape) > 4:
        levels.append((c, w))
        c = cv2.resize(c, (c.shape[1] // 2, c.shape[0] // 2), interpolation=cv2.INTER_AREA)
        w = cv2.resize(w, (w.shape[1] // 2, w.shape[0] // 2), interpolation=cv2.INTER_AREA)
    est = c / np.maximum(w, 1e-6)[..., None]
    for c, w in reversed(levels):
        up = cv2.resize(est, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)
        here = c / np.maximum(w, 1e-6)[..., None]
        a = np.clip(w, 0, 1)[..., None]
        est = a * here + (1 - a) * up
    return np.where(known[..., None], img, est.clip(0, 255)).astype(np.uint8)


def clean_plate(project, ref, step, dilate_px, files, K, Ts, min_obs=3):
    """Median of the static scene seen from the reference camera. Each reference pixel's 3-D point (table pixels:
    ray / table-plane intersection; others: MoGe-2 metric point map) is reprojected into every k-th frame with the
    estimated cameras; samples on that frame's occluders (dynamic + tracked objects, dilated) are dropped."""
    import cv2
    from ..geom3d import pixel_rays
    rgb_ref = read_rgb(files[ref])
    H, W = rgb_ref.shape[:2]
    mdir = project / "work" / "masks"
    kd = np.ones((dilate_px, dilate_px), np.uint8)
    pw = np.load(project / "work" / "world" / "points_ref_world.npz")
    Pw = pw["points"].astype(np.float64)                     # world-frame points of reference pixels (MoGe-2)
    table = cv2.morphologyEx((read_mask(mdir / "static" / "table.png") > 0).astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((31, 31), np.uint8)) > 0
    # table pixels: exact ray / plane (z = 0) intersection instead of noisy depth
    ys, xs = np.nonzero(table)
    T0 = Ts[ref]
    d = pixel_rays(K, np.c_[xs, ys]) @ T0[:3, :3].T
    o = T0[:3, 3]
    t = -o[2] / d[:, 2]
    Pw[ys, xs] = o + t[:, None] * d
    ok3 = np.isfinite(Pw).all(-1)
    Pw = np.nan_to_num(Pw)
    ids = sorted(set(range(0, len(files), step)) | {ref})
    from .estimate_camera_response import load as load_response
    gains = load_response(project, len(files))           # camera auto white balance / exposure, per frame
    lut = np.arange(256, dtype=np.float32) / 255
    lut = np.where(lut <= 0.04045, lut / 12.92, ((lut + 0.055) / 1.055) ** 2.4)
    S, V = [], []
    for i in ids:
        Tcw = np.linalg.inv(Ts[i])
        pc = Pw.reshape(-1, 3) @ Tcw[:3, :3].T + Tcw[:3, 3]
        z = pc[:, 2]
        mx = (K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]).reshape(H, W).astype(np.float32)
        my = (K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]).reshape(H, W).astype(np.float32)
        im = read_rgb(files[i])
        occ = (read_mask(mdir / "dynamic" / f"frame{i:05d}.png") > 0) | (read_mask(mdir / "objects" / f"frame{i:05d}.png") > 0)
        occ = cv2.dilate(occ.astype(np.uint8), kd)
        w = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        lin = lut[w] / gains[i]                            # to the reference frame's camera response
        S.append(np.clip(np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055) * 255, 0, 255))
        inside = (mx >= 0) & (mx < W - 1) & (my >= 0) & (my < H - 1) & (z.reshape(H, W) > 0) & ok3
        V.append(inside & (cv2.remap(occ, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1) == 0))
    S = np.stack(S).astype(np.float32)
    V = np.stack(V)
    seen = V.sum(0)
    S[~V] = np.nan
    plate = np.nanmedian(S, axis=0)
    observed = seen >= min_obs
    plate = np.nan_to_num(plate, nan=0).clip(0, 255).astype(np.uint8)
    plate = push_pull_fill(plate, observed)
    return plate, observed, seen, [{"frame": int(i), "valid_fraction": float(v.mean())} for i, v in zip(ids, V)]


def run(project: Path, ref_frame: int | None = None, step: int = 2, dilate_px: int = 21, texel_m: float = 0.002,
        extend_unseen_m: float = 0.15, table_thickness_m: float = 0.03, table_height_m: float = 0.75,
        backdrop_grid_px: int = 8, max_edge_ratio: float = 0.06) -> dict:
    import cv2
    world = read_json(project / "work" / "world" / "world.json")
    cams = read_json(project / "work" / "world" / "cameras.json")
    ref = world["ref_frame"] if ref_frame is None else ref_frame
    files = frame_files(project / "work" / "frames")
    K = np.asarray(cams["K"])
    T_wc = np.asarray(cams["T_world_cam"][ref])
    T_cw = inv(T_wc)
    W, H = cams["width"], cams["height"]
    out = project / "work" / "background"
    assets = project / "scene" / "assets" / "background"
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    # 1. clean plate
    plate, observed, seen, info = clean_plate(project, ref, step, dilate_px, files, K, np.asarray(cams["T_world_cam"]))
    write_rgb(assets / "plate.png", plate)
    write_mask(out / "plate_observed.png", observed)

    # 2. table top: outline -> rectangle in world xy, extended on unseen sides
    outline = np.asarray(world["table"]["outline_all_xy"])
    on_border = np.asarray(world["table"]["outline_on_image_border"], bool)
    rect = cv2.minAreaRect(outline.astype(np.float32))
    (cx, cy), (w, h), ang = rect
    a = np.radians(ang)
    u = np.array([np.cos(a), np.sin(a)])
    v = np.array([-np.sin(a), np.cos(a)])
    c = np.array([cx, cy])
    pu, pv = (outline - c) @ u, (outline - c) @ v
    lim = {"-u": pu.min(), "+u": pu.max(), "-v": pv.min(), "+v": pv.max()}
    observed_sides = {}
    for side, proj, sgn in (("-u", pu, -1), ("+u", pu, 1), ("-v", pv, -1), ("+v", pv, 1)):
        near = np.abs(proj - lim[side]) < 0.02
        frac_border = float(on_border[near].mean()) if near.any() else 1.0
        observed_sides[side] = frac_border < 0.5
        if not observed_sides[side]:
            lim[side] += sgn * extend_unseen_m
    corners = np.array([c + lim["-u"] * u + lim["-v"] * v, c + lim["+u"] * u + lim["-v"] * v,
                        c + lim["+u"] * u + lim["+v"] * v, c + lim["-u"] * u + lim["+v"] * v])
    xmin, ymin = corners.min(0)
    xmax, ymax = corners.max(0)
    nx, ny = int(np.ceil((xmax - xmin) / texel_m)), int(np.ceil((ymax - ymin) / texel_m))
    gx, gy = np.meshgrid(xmin + (np.arange(nx) + 0.5) * texel_m, ymax - (np.arange(ny) + 0.5) * texel_m)
    pts = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], -1)
    uv, z = project_points(K, T_cw, pts)
    inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
    mapx = uv[:, 0].reshape(ny, nx).astype(np.float32)
    mapy = uv[:, 1].reshape(ny, nx).astype(np.float32)
    tex = cv2.remap(plate, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    table_mask = cv2.morphologyEx((read_mask(project / "work" / "masks" / "static" / "table.png") > 0).astype(np.uint8),
                                  cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    on_table = cv2.remap(table_mask, mapx, mapy, cv2.INTER_NEAREST, borderValue=0) > 0
    seen_tex = cv2.remap(observed.astype(np.uint8), mapx, mapy, cv2.INTER_NEAREST, borderValue=0) > 0
    in_view = inside.reshape(ny, nx)
    tex_obs = in_view & seen_tex                          # observed texels (reporting); in view = plate (incl. its fill)
    # unobserved texels: observed colour statistics (tiled noise-free patches of the observed table)
    obs_px = tex[tex_obs & on_table]
    mean = np.median(obs_px, 0)
    fill = np.empty_like(tex)
    fill[:] = mean.astype(np.uint8)
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(rng.normal(0, obs_px.std(0).mean() * 0.25, fill.shape[:2]).astype(np.float32), (0, 0), 2)
    fill = np.clip(fill + noise[..., None], 0, 255).astype(np.uint8)
    tex = np.where(in_view[..., None], tex, fill)
    # soften the seam between the camera's view and the statistical fill outside it
    seam = cv2.dilate(in_view.astype(np.uint8), np.ones((5, 5), np.uint8)) ^ cv2.erode(in_view.astype(np.uint8), np.ones((5, 5), np.uint8))
    tex = np.where(seam[..., None] > 0, cv2.GaussianBlur(tex, (7, 7), 0), tex)
    write_rgb(assets / "table_top.png", tex)
    write_mask(out / "table_observed.png", tex_obs)

    # table slab: top (textured) + sides/bottom (untextured), world frame
    top = np.c_[corners, np.zeros(4)]
    bot = np.c_[corners, -table_thickness_m * np.ones(4)]
    Vt = np.vstack([top, bot])
    uv_c = np.c_[(corners[:, 0] - xmin) / (xmax - xmin), (corners[:, 1] - ymin) / (ymax - ymin)]
    UV = np.vstack([uv_c, [[1.0, 1.0]]])                  # sides/bottom: one zero-area UV (never baked over the top)
    o = 1 if np.cross(np.r_[corners[1] - corners[0], 0], np.r_[corners[2] - corners[1], 0])[2] > 0 else -1
    F = [[0, 1, 2], [0, 2, 3]] if o > 0 else [[0, 2, 1], [0, 3, 2]]
    F += [[4, 6, 5], [4, 7, 6]] if o > 0 else [[4, 5, 6], [4, 6, 7]]
    for k in range(4):
        a_, b_ = k, (k + 1) % 4
        F += [[a_, b_ + 4, b_], [a_, a_ + 4, b_ + 4]] if o > 0 else [[a_, b_, b_ + 4], [a_, b_ + 4, a_ + 4]]
    FT = [f if k < 2 else [4, 4, 4] for k, f in enumerate(F)]
    _write_obj(assets / "table.obj", Vt, np.asarray(F), UV, "table_top", "table_top.png", FT=FT)

    # 3. backdrop: depth-grid mesh of backdrop pixels (reference frame), textured by the plate
    pw = np.load(project / "work" / "world" / "points_ref_world.npz")
    P, valid = pw["points"], pw["valid"]
    bd = read_mask(project / "work" / "masks" / "static" / "backdrop.png") > 0
    bd = cv2.dilate(bd.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    g = backdrop_grid_px
    ys = np.arange(0, H, g)
    xs = np.arange(0, W, g)
    if ys[-1] != H - 1:
        ys = np.r_[ys, H - 1]
    if xs[-1] != W - 1:
        xs = np.r_[xs, W - 1]
    GX, GY = np.meshgrid(xs, ys)
    Pg = P[GY, GX]
    from matplotlib.path import Path as MPath
    over_table = MPath(corners).contains_points(Pg[..., :2].reshape(-1, 2)).reshape(Pg.shape[:2])
    # the backdrop is what lies beyond the table: no vertices on or just above the table top (no ridges on it)
    ok = valid[GY, GX] & bd[GY, GX] & np.isfinite(Pg).all(-1) & (Pg[..., 2] > -0.05) & ~(over_table & (Pg[..., 2] < 0.05))
    # backdrop must not sit in front of the table (MoGe edges bleed): keep points behind the table outline
    idx = -np.ones(ok.shape, int)
    idx[ok] = np.arange(ok.sum())
    Vb = Pg[ok]
    UVb = np.c_[GX[ok] / (W - 1), 1 - GY[ok] / (H - 1)]
    cam = T_wc[:3, 3]
    Fb = []
    for r in range(len(ys) - 1):
        for q in range(len(xs) - 1):
            a_, b_, c_, d_ = idx[r, q], idx[r, q + 1], idx[r + 1, q + 1], idx[r + 1, q]
            for tri in ((a_, d_, c_), (a_, c_, b_)):
                if min(tri) < 0:
                    continue
                tp = Vb[list(tri)]
                dist = np.linalg.norm(tp - cam, axis=1)
                edge = max(np.linalg.norm(tp[0] - tp[1]), np.linalg.norm(tp[1] - tp[2]), np.linalg.norm(tp[0] - tp[2]))
                if edge / dist.mean() < max_edge_ratio:
                    Fb.append(tri)
    Fb = np.asarray(Fb)
    # winding: normals towards the reference camera
    if len(Fb):
        nrm = np.cross(Vb[Fb[:, 1]] - Vb[Fb[:, 0]], Vb[Fb[:, 2]] - Vb[Fb[:, 0]])
        flip = (nrm * (cam - Vb[Fb[:, 0]])).sum(1) < 0
        Fb[flip] = Fb[flip][:, ::-1]
    _write_obj(assets / "backdrop.obj", Vb, Fb, UVb, "backdrop", "plate.png")

    res = {
        "plate": {"frames_used": len(info), "method": "reprojection of reference pixels with the estimated cameras",
                  "observed_fraction": float(observed.mean()), "inpainted_px": int((~observed).sum()),
                  "median_observations": float(np.median(seen))},
        "table": {"corners_xy": corners.tolist(), "size_m": [float(lim["+u"] - lim["-u"]), float(lim["+v"] - lim["-v"])],
                  "observed_sides": observed_sides, "extended_unseen_sides_by_m": extend_unseen_m,
                  "thickness_m": table_thickness_m, "height_prior_m": table_height_m,
                  "texture_px": [nx, ny], "texel_m": texel_m, "texture_observed_fraction": float(tex_obs.mean())},
        "backdrop": {"vertices": int(len(Vb)), "faces": int(len(Fb)), "grid_px": g,
                     "z_range_m": [float(Vb[:, 2].min()), float(Vb[:, 2].max())] if len(Vb) else None},
        "frames": info,
    }
    write_json(res, out / "background.json")
    ref_rgb = read_rgb(files[ref])
    write_rgb(project / "evidence" / "background_plate.jpg", contact_sheet(
        [ref_rgb, plate, (np.stack([observed] * 3, -1) * 255).astype(np.uint8),
         cv2.resize(tex, (int(tex.shape[1] * 720 / tex.shape[0]), 720)) if tex.shape[0] > 0 else plate],
        cols=2, width=640, labels=["reference", "clean plate", "plate observed (white)", "table top texture (ortho)"]))
    return {**{k: v for k, v in res.items() if k != "frames"},
            "outputs": ["scene/assets/background", "work/background", "evidence/background_plate.jpg"]}
