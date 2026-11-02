"""Fixed-protocol comparison of Blender renders with the real frames (source_reconstruction protocol).

Protocol (recorded in the report): source frames = rendered src_%05d.png frames, each frame's own camera,
full resolution, sRGB, no colour correction ("raw"); a second set of numbers after one global per-channel gain
fitted on static regions of the fit frames ("declared colour correction"). Hands/arms/controllers (dynamic masks,
dilated) are not comparable and excluded; objects that are moving at a frame are not rendered and their pixels
are excluded. Regions: backdrop, table, each object (pixels inside the tracked mask or the rendered silhouette).
Metrics per region: masked PSNR, MAE, SSIM (mean of the SSIM map); LPIPS when the package is available;
object silhouette IoU (rendered vs tracked). Frames are split into fit / held-out per the rest states.
Outputs: evidence/eval/compare_%05d.jpg (real | blender | |diff| heatmap), evidence/eval/metrics.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import contact_sheet, frame_files, read_json, read_mask, read_rgb, write_json, write_rgb
from ..geom3d import inv

ENV = "sam3d"
GPU = True


def _ssim_map(a, b):
    import cv2
    a, b = a.astype(np.float32), b.astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    saa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    sbb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return s.mean(-1) if s.ndim == 3 else s


def _metrics(real, ren, m, ssim, lp=None):
    if m.sum() < 50:
        return None
    d = real[m].astype(np.float32) - ren[m].astype(np.float32)
    mse = float((d ** 2).mean())
    r = {"pixels": int(m.sum()), "psnr_db": float(10 * np.log10(255 ** 2 / max(mse, 1e-6))),
         "mae": float(np.abs(d).mean()), "ssim": float(ssim[m].mean())}
    if lp is not None:
        r["lpips"] = float(lp[m].mean())
    return r


def run(project: Path, dyn_dilate_px: int = 15, lpips_net: str = "alex") -> dict:
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
    labels = read_json(project / "work" / "masks" / "labels.json")["labels"]
    name_to_label = {v: int(k) for k, v in labels.items()}
    files = frame_files(project / "work" / "frames")
    rdir = project / "blender" / "renders"
    rigid = {n: e for n, e in scene["entities"].items() if e.get("capability") == "rigid"}
    split = {}
    for e in rigid.values():
        for st in e["states"].values():
            for f in st["fit_frames"]:
                split.setdefault(f, "fit")
            for f in st["heldout_frames"]:
                split[f] = "heldout"
    frames = sorted(int(p.stem.split("_")[1]) for p in rdir.glob("src_*.png"))
    dev = "cuda"
    R = Rasterizer()
    meshes = {n: trimesh.load(project / "scene" / e["geometry"]["visual"]["obj_uri"], force="mesh", process=False)
              for n, e in rigid.items()}
    try:
        import lpips
        lp_model = lpips.LPIPS(net=lpips_net, spatial=True).to(dev).eval()
    except Exception:
        lp_model = None
    table = read_mask(project / "work" / "masks" / "static" / "table.png") > 0
    backdrop = read_mask(project / "work" / "masks" / "static" / "backdrop.png") > 0
    kd = np.ones((dyn_dilate_px, dyn_dilate_px), np.uint8)
    per_frame, gains_num, gains_den = {}, np.zeros(3), np.zeros(3)
    cache = {}
    for f in frames:
        real = read_rgb(files[f])
        ren = read_rgb(rdir / f"src_{f:05d}.png")
        dyn = cv2.dilate((read_mask(project / "work" / "masks" / "dynamic" / f"frame{f:05d}.png") > 0).astype(np.uint8), kd) > 0
        lab = read_mask(project / "work" / "masks" / "objects" / f"frame{f:05d}.png")
        Tcw = inv(Ts[f])
        Vs, Fs, names = [], [], []
        moving = np.zeros((H, W), bool)
        for n, e in rigid.items():
            s = next((s for s, st in e["states"].items() if st["frames"][0] <= f <= st["frames"][1]), None)
            if s is None:
                moving |= lab == name_to_label[n]
                continue
            T = np.asarray(e["states"][s]["T_world_obj"])
            V = np.asarray(meshes[n].vertices) @ T[:3, :3].T + T[:3, 3]
            Vs.append(torch.tensor(V @ Tcw[:3, :3].T + Tcw[:3, 3], dtype=torch.float32, device=dev))
            Fs.append(torch.tensor(np.asarray(meshes[n].faces), dtype=torch.int64, device=dev))
            names.append(n)
        sil = {}
        if Vs:
            with torch.no_grad():
                S, _ = R.render_ids(Vs, Fs, K, W, H)
            raw = {n: S[k].cpu().numpy() > 0.5 for k, n in enumerate(names)}
            # same silhouette definition as registration: openings of the visual mesh (slots) filled, minus other
            # objects in front (tracked masks are solid)
            for n in names:
                cn, _ = cv2.findContours(raw[n].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                f_ = cv2.drawContours(np.zeros((H, W), np.uint8), cn, -1, 1, -1) > 0
                sil[n] = f_ & ~np.any([raw[m] for m in names if m != n] or [np.zeros((H, W), bool)], axis=0)
        valid = ~dyn & ~moving
        regions = {}
        obj_any = np.zeros((H, W), bool)
        for n in names:
            m_src = lab == name_to_label[n]
            regions[n] = (m_src | sil[n]) & valid
            # interior: inside both the tracked mask and the render, 3 px from either boundary (appearance only;
            # boundary/placement error is measured by the silhouette metrics)
            regions[f"{n}_interior"] = cv2.erode((m_src & sil[n]).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & valid
            obj_any |= m_src | sil[n]
        regions["table"] = table & ~obj_any & valid
        regions["backdrop"] = backdrop & ~table & ~obj_any & valid
        regions["all_comparable"] = valid
        cache[f] = (real, ren, regions, dyn | moving)
        if split.get(f) == "fit":
            st_m = (regions["table"] | regions["backdrop"])
            gains_num += (real[st_m].astype(np.float64) * ren[st_m]).sum(0)
            gains_den += (ren[st_m].astype(np.float64) ** 2).sum(0)
        per_frame[f] = {"split": split.get(f, "reference"),
                        "silhouette_iou": {n: float(((lab == name_to_label[n]) & sil[n] & ~dyn).sum() /
                                                    max((((lab == name_to_label[n]) | sil[n]) & ~dyn).sum(), 1)) for n in names}}
    gain = (gains_num / np.maximum(gains_den, 1)).tolist() if gains_den.any() else [1.0, 1.0, 1.0]
    out = project / "evidence" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    from .estimate_camera_response import load as load_response
    resp = load_response(project, len(files))

    def through_camera(ren, f):
        x = ren.astype(np.float32) / 255
        lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4) * resp[f]
        return np.clip(np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055), 0, 1) * 255

    for f, (real, ren, regions, excl) in cache.items():
        ren_cam = through_camera(ren, f)
        ss_cam = _ssim_map(real, ren_cam)
        lp_cam = None
        ren_cc = np.clip(ren.astype(np.float32) * np.asarray(gain), 0, 255)
        ss_raw, ss_cc = _ssim_map(real, ren), _ssim_map(real, ren_cc)
        lp_raw = lp_cc = None
        if lp_model is not None:
            def lpmap(x):
                t = lambda im: torch.tensor(im, dtype=torch.float32, device=dev).permute(2, 0, 1)[None] / 127.5 - 1
                with torch.no_grad():
                    return lp_model(t(real), t(x))[0, 0].cpu().numpy()
            lp_raw, lp_cc = lpmap(ren), lpmap(ren_cc)
        per_frame[f]["raw"] = {k: _metrics(real, ren, m, ss_raw, lp_raw) for k, m in regions.items()}
        per_frame[f]["color_corrected"] = {k: _metrics(real, ren_cc, m, ss_cc, lp_cc) for k, m in regions.items()}
        if lp_model is not None:
            lp_cam = lpmap(ren_cam.astype(np.uint8))
        per_frame[f]["camera_response"] = {k: _metrics(real, ren_cam, m, ss_cam, lp_cam) for k, m in regions.items()}
        d = np.abs(real.astype(np.float32) - ren.astype(np.float32)).mean(-1)
        heat = cv2.applyColorMap(np.clip(d * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].copy()
        heat[excl] = (heat[excl] * 0.2 + 80).astype(np.uint8)
        p = per_frame[f]["raw"]["all_comparable"]
        write_rgb(out / f"compare_{f:05d}.jpg", contact_sheet(
            [real, ren, heat], cols=3, width=640,
            labels=[f"real f{f} ({per_frame[f]['split']})", "blender",
                    f"|diff| x4 (grey = excluded)  PSNR {p['psnr_db']:.1f} dB" if p else "|diff|"]))

    def agg(split_name, kind):
        rows = [v[kind] for v in per_frame.values() if v["split"] == split_name and kind in v]
        res = {}
        for reg in {k for r in rows for k in r if r[k]}:
            vals = [r[reg] for r in rows if r.get(reg)]
            res[reg] = {m: float(np.mean([v[m] for v in vals])) for m in vals[0] if m != "pixels"}
            res[reg]["frames"] = len(vals)
        return res

    summary = {s: {"raw": agg(s, "raw"), "color_corrected": agg(s, "color_corrected"),
                   "camera_response": agg(s, "camera_response"),
                   "silhouette_iou": {n: float(np.mean([v["silhouette_iou"][n] for v in per_frame.values()
                                                        if v["split"] == s and n in v["silhouette_iou"]] or [np.nan]))
                                      for n in rigid}}
               for s in ("reference", "fit", "heldout")}
    report = {"protocol": {"protocol": "source_reconstruction", "resolution": [W, H], "color_space": "sRGB 8-bit",
                           "excluded": "dynamic masks dilated by %d px; objects while moving" % dyn_dilate_px,
                           "color_correction": {"type": "global per-channel gain on static regions of fit frames",
                                                "gain_rgb": gain},
                           "camera_response": "per-frame diagonal gains of the camera (auto white balance / exposure) "
                                              "from static background only (work/camera/response.json)",
                           "lpips": lpips_net if lp_model is not None else "unavailable"},
              "summary": summary, "per_frame": per_frame}
    write_json(report, out / "metrics.json")
    return {"summary": summary, "color_gain": gain, "lpips": lp_model is not None,
            "outputs": ["evidence/eval"]}
