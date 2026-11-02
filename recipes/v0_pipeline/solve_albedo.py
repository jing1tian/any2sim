"""De-lit albedo textures from the video, so Blender's lighting reproduces the observed appearance.

Per texel: albedo = observed linear radiance / baked irradiance E (blender_bake), in linear sRGB.
  * table: observed radiance = the clean-plate table texture (already in texel space)
  * objects: texel positions (state pose) are projected into the fit frames of each rest state; a sample counts
    when the texel is visible (depth test against all objects rendered with nvdiffrast), faces the camera, lies
    inside the object's tracked mask and outside dilated hand/arm pixels. Per-texel median over samples.
    Texels never observed keep the generated (SAM 3D) texture, colour-matched to the observed texels with a
    per-channel gain; the seam is feathered.
A global light-strength factor puts the 99.5th-percentile albedo at `max_albedo` (light energy and albedo are only
known up to a common scale); it is written to work/appearance/lighting.json and applied by blender_render.
Outputs: work/appearance/<entity>/albedo.png (16-bit sRGB), work/appearance/albedo.json
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np

from ..common import contact_sheet, frame_files, read_json, read_mask, read_rgb, write_json, write_rgb
from ..geom3d import inv

ENV = "sam3d"
GPU = True


def srgb_to_lin(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def bilinear(img, u, v):
    """Bilinear samples (N,C) of an (H,W,C) image at float pixel coordinates (any N; cv2.remap caps at 32767)."""
    H, W = img.shape[:2]
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1] +
            (1 - fx) * fy * img[y0 + 1, x0] + fx * fy * img[y0 + 1, x0 + 1])


def read_exr(p):
    import cv2
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    return im[..., :3][..., ::-1].astype(np.float32) if im.ndim == 3 else im.astype(np.float32)


def write_png16(p, srgb01):
    import cv2
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), (np.clip(srgb01, 0, 1) * 65535).astype(np.uint16)[..., ::-1])


def run(project: Path, max_albedo: float = 0.9, min_E_frac: float = 0.05, min_E_obs: float = 0.15, bg_tol: float = 14.0,
        table_lowfreq_m: float | None = None, depth_tol_m: float = 0.004,
        min_cos: float = 0.25, dyn_dilate_px: int = 9, mask_erode_px: int = 3, seam_px: int = 12) -> dict:
    import cv2
    import torch
    import trimesh
    import yaml
    from ..models.raster import Rasterizer
    scene = yaml.safe_load((project / "scene" / "scene.yaml").read_text())
    cams = read_json(project / "work" / "world" / "cameras.json")
    Ts = np.asarray(cams["T_world_cam"])
    K = np.asarray(cams["K"])
    W, H = cams["width"], cams["height"]
    bake = read_json(project / "work" / "appearance" / "bake" / "bake.json")
    labels = read_json(project / "work" / "masks" / "labels.json")["labels"]
    name_to_label = {v: int(k) for k, v in labels.items()}
    files = frame_files(project / "work" / "frames")
    rigid = {n: e for n, e in scene["entities"].items() if e.get("capability") == "rigid"}
    sdir = project / "scene"
    app = project / "work" / "appearance"
    dev = "cuda"
    R = Rasterizer()
    meshes = {}
    for n, e in rigid.items():
        m = trimesh.load(sdir / e["geometry"]["visual"]["obj_uri"], force="mesh", process=False)
        meshes[n] = (torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=dev),
                     torch.tensor(np.asarray(m.faces), dtype=torch.int64, device=dev))

    def state_at(e, f):
        for s, st in e["states"].items():
            if st["frames"][0] <= f <= st["frames"][1]:
                return s
        return None

    def depth_at(f):
        Tcw = torch.tensor(inv(Ts[f]), dtype=torch.float32, device=dev)
        Vs, Fs = [], []
        for n, e in rigid.items():
            s = state_at(e, f)
            if s is None:
                continue
            T = torch.tensor(e["states"][s]["T_world_obj"], dtype=torch.float32, device=dev)
            V, F = meshes[n]
            Vw = V @ T[:3, :3].T + T[:3, 3]
            Vs.append(Vw @ Tcw[:3, :3].T + Tcw[:3, 3])
            Fs.append(F)
        with torch.no_grad():
            _, z = R.render_ids(Vs, Fs, K, W, H)
        return z.cpu().numpy()

    raw = {}           # entity -> (albedo_lin HxWx3 with NaN where unobserved, base_lin, cov)
    report = {}
    # table: plate texture / E
    tb = bake["entities"]["table"]
    seen = cv2.resize(read_mask(project / "work" / "background" / "table_observed.png"), tuple(tb["size"]),
                      interpolation=cv2.INTER_NEAREST) > 0
    E = np.where(seen[..., None], read_exr(tb["E"]["static"]), read_exr(tb["E"]["empty"]))
    tex = read_rgb(sdir / scene["entities"]["table"]["materials"]["top"]["base_color_texture"])
    tex = cv2.resize(tex, tuple(tb["size"]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    cov = read_exr(tb["cov"])[..., 0] > 0.5
    Eref = np.median(E[cov]) if cov.any() else 1.0
    a = srgb_to_lin(tex) / np.maximum(E, min_E_frac * Eref)
    report["table"] = {"observed_texel_fraction": float((seen & cov).mean()), "uv_coverage": float(cov.mean()),
                       "E_median": float(Eref)}
    if table_lowfreq_m:
        # unmodelled occluders (the demonstrator's arms and body) shadow the table in the video; that shading is
        # not part of the table: divide out low-frequency luminance the rig does not explain, and replace dark
        # outliers (equipment remnants in the plate) with the local median
        texel_m = read_json(project / "work" / "background" / "background.json")["table"]["texel_m"]
        sig = table_lowfreq_m / texel_m
        lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        low = cv2.GaussianBlur(lum, (0, 0), sig) / max(float(np.median(lum)), 1e-6)
        a = a / np.clip(low, 0.3, 3.0)[..., None]
        med = cv2.medianBlur((np.clip(lin_to_srgb(a), 0, 1) * 255).astype(np.uint8), 31).astype(np.float32) / 255
        lum2 = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        dark = lum2 < 0.6 * (srgb_to_lin(med) @ np.array([0.2126, 0.7152, 0.0722], np.float32))
        dark = cv2.dilate(dark.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        a = np.where(dark[..., None], srgb_to_lin(med), a)
        report["table"]["lowfreq_shading_removed_m"] = table_lowfreq_m
        report["table"]["dark_outlier_texels"] = float(dark.mean())
    raw["table"] = (np.where(cov[..., None], a, np.nan), srgb_to_lin(tex), cov)

    # objects
    ref = scene["cameras"]["source_cam"]["ref_frame"]
    plate_f = read_rgb(sdir / "assets" / "background" / "plate.png").astype(np.float32)
    pw = np.load(project / "work" / "world" / "points_ref_world.npz")["points"]

    def warp_to_ref(f, u, v):
        """Pixel (u, v) of frame f -> reference pixel, through the static scene seen there (table plane z = 0)."""
        T = Ts[f]
        d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ T[:3, :3].T
        t = -T[2, 3] / d[:, 2]
        X = T[:3, 3] + t[:, None] * d
        Tr = inv(Ts[ref])
        pc = X @ Tr[:3, :3].T + Tr[:3, 3]
        return K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    del pw
    from .estimate_camera_response import load as load_response
    gains = load_response(project, len(files))
    kd = np.ones((dyn_dilate_px, dyn_dilate_px), np.uint8)
    ke = np.ones((mask_erode_px * 2 + 1,) * 2, np.uint8)
    depth_cache = {}
    for n, e in rigid.items():
        b = bake["entities"][n]
        w, h = b["size"]
        pos = read_exr(b["pos"])
        nrm = read_exr(b["nrm"])
        cov = read_exr(b["cov"])[..., 0] > 0.5
        base = read_rgb(sdir / e["geometry"]["visual"]["texture_uri"])
        base = cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        samples = []
        for s, st in e["states"].items():
            E = read_exr(b["E"][s])
            T = np.asarray(st["T_world_obj"])
            P = pos[cov] @ T[:3, :3].T + T[:3, 3]
            N = nrm[cov] @ T[:3, :3].T
            N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
            Ecov = E[cov]
            for f in st["fit_frames"]:                      # held-out frames stay unseen
                if f not in depth_cache:
                    depth_cache[f] = depth_at(f)
                Z = depth_cache[f]
                Tcw = inv(Ts[f])
                pc = P @ Tcw[:3, :3].T + Tcw[:3, 3]
                z = pc[:, 2]
                u = K[0, 0] * pc[:, 0] / z + K[0, 2]
                v = K[1, 1] * pc[:, 1] / z + K[1, 2]
                inside = (z > 0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
                ui, vi = np.clip(u.round().astype(int), 0, W - 1), np.clip(v.round().astype(int), 0, H - 1)
                zb = Z[vi, ui]
                visible = inside & (zb > 0) & (np.abs(z - zb) < depth_tol_m + 0.01 * z)
                camc = Ts[f][:3, 3]
                view = camc - P
                view /= np.linalg.norm(view, axis=1, keepdims=True)
                cosv = (N * view).sum(1)
                m = read_mask(project / "work" / "masks" / "objects" / f"frame{f:05d}.png") == name_to_label[n]
                m = cv2.erode(m.astype(np.uint8), ke) > 0
                dyn = cv2.dilate((read_mask(project / "work" / "masks" / "dynamic" / f"frame{f:05d}.png") > 0).astype(np.uint8), kd) > 0
                lit = Ecov.mean(1) > min_E_obs * np.median(Ecov.mean(1))     # deep shadow: albedo unreliable
                ok = visible & (cosv > min_cos) & m[vi, ui] & ~dyn[vi, ui] & lit
                rgb_f = read_rgb(files[f]).astype(np.float32)
                img = srgb_to_lin(rgb_f / 255) / gains[f]      # to the reference frame's camera response
                col = np.full((len(P), 3), np.nan, np.float32)
                if ok.any():
                    # background seen through openings (basket slots): the pixel looks like the clean plate there
                    plate_px = bilinear(plate_f, *(np.r_[[u[ok]], [v[ok]]] if f == ref else warp_to_ref(f, u[ok], v[ok])))
                    px = bilinear(img, u[ok], v[ok])
                    px = np.where(px <= 0.0031308, 12.92 * px, 1.055 * np.clip(px, 0, None) ** (1 / 2.4) - 0.055) * 255
                    through = np.abs(px - plate_px).mean(1) < bg_tol
                    idx = np.nonzero(ok)[0]
                    ok[idx[through]] = False
                if ok.any():
                    col[ok] = bilinear(img, u[ok], v[ok])
                    col[ok] /= np.maximum(Ecov[ok], min_E_frac * np.median(Ecov))
                samples.append(col)
        S = np.stack(samples) if samples else np.full((1, int(cov.sum()), 3), np.nan)
        n_obs = np.isfinite(S[..., 0]).sum(0)
        med = np.nanmedian(np.where(np.isfinite(S), S, np.nan), axis=0) if n_obs.any() else np.full((int(cov.sum()), 3), np.nan)
        A = np.full((h, w, 3), np.nan, np.float32)
        A[cov] = np.where((n_obs > 0)[:, None], med, np.nan)
        raw[n] = (A, srgb_to_lin(base), cov)
        report[n] = {"texels": int(cov.sum()), "observed_texel_fraction": float((n_obs > 0).mean()),
                     "samples_per_observed_texel": float(n_obs[n_obs > 0].mean()) if (n_obs > 0).any() else 0.0,
                     "frames": int(len(samples))}

    # global light scale: albedo <= max_albedo
    allv = np.concatenate([v[0][np.isfinite(v[0][..., 0])] for v in raw.values()])
    p995 = float(np.percentile(allv.max(1), 99.5))
    light_scale = p995 / max_albedo          # albedo and light energy share one unknown scale: fix it here
    tiles, caps = [], []
    for n, (A, base, cov) in raw.items():
        A = A / light_scale
        obs = np.isfinite(A[..., 0])
        if n == "table":
            out = np.where(obs[..., None], A, base / light_scale)
            gain = None
        else:
            # unseen texels: the observed median albedo, modulated by the generated texture's relative brightness
            # (generated textures carry baked shading and wrong hues; only their pattern is kept)
            if obs.any():
                lum = base @ np.array([0.2126, 0.7152, 0.0722])
                rel = np.clip(lum / max(np.median(lum[cov]), 1e-4), 0.6, 1.4)
                ma = np.median(A[obs], 0)
                fill = np.clip(ma * rel[..., None], 0, 1)
                gain = {"median_observed_albedo_lin": ma.tolist(), "pattern": "generated luminance, clipped 0.6-1.4"}
            else:
                fill, gain = base, None
            alpha = np.zeros(obs.shape, np.float32)
            if obs.any():
                dist = cv2.distanceTransform((~obs).astype(np.uint8), cv2.DIST_L2, 3)
                alpha = np.clip(1 - dist / seam_px, 0, 1) * (~obs) + obs
            out = np.where(obs[..., None], A, fill)
            out = alpha[..., None] * np.nan_to_num(out) + (1 - alpha[..., None]) * fill
            # dilate observed values into the feather band so the blend has colour to blend with
            out = np.where(np.isfinite(out), out, fill)
            report[n]["hidden_colour_transfer"] = gain
        out = np.clip(out, 0, 1)
        srgb = lin_to_srgb(out)
        write_png16(app / n / "albedo.png", srgb)
        report[n]["albedo_srgb_median"] = np.median(srgb[cov], 0).tolist() if cov.any() else None
        tiles += [(lin_to_srgb(base) * 255).astype(np.uint8), (srgb * 255).astype(np.uint8),
                  (np.stack([obs] * 3, -1) * 255).astype(np.uint8)]
        caps += [f"{n}: generated/plate texture", f"{n}: de-lit albedo", f"{n}: observed texels"]
    lit = read_json(app / "lighting.json")
    lit["light_scale"] = light_scale
    lit["albedo_p995_before_scale"] = p995
    write_json(lit, app / "lighting.json")
    write_json({"entities": report, "light_scale": light_scale}, app / "albedo.json")
    write_rgb(project / "evidence" / "albedo_textures.jpg", contact_sheet(tiles, cols=3, width=400, labels=caps))
    return {"entities": report, "light_scale": light_scale,
            "outputs": ["work/appearance", "evidence/albedo_textures.jpg"]}
