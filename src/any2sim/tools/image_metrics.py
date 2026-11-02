"""Masked image comparison: PSNR, MAE, SSIM, LPIPS (AlexNet) per region, for any pairs of images.

params:
  pairs: [{"id", "a": path, "b": path, "regions": {name: mask path | "all"}, "ignore": mask path | null,
           "ignore_dilate_px": 0}]      paths project-relative; a and b same size
  out: project-relative json (default evidence/image_metrics.json)
  heatmaps: true                       <out stem>_<id>.jpg: a | b | |a-b| x4 (ignored pixels grey)
Returns per pair, per region: pixels, psnr_db, mae, ssim, lpips.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import contact_sheet, read_mask, read_rgb, write_json, write_rgb

ENV = "sam3d"
GPU = True


def ssim_map(a, b):
    import cv2
    a, b = a.astype(np.float32), b.astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    saa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    sbb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return s.mean(-1)


def run(project: Path, pairs: list[dict], out: str = "evidence/image_metrics.json", heatmaps: bool = True) -> dict:
    import cv2
    import torch
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).cuda().eval()
    except Exception:
        lp = None
    res = {}
    for p in pairs:
        a, b = read_rgb(project / p["a"]), read_rgb(project / p["b"])
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        ign = np.zeros(a.shape[:2], bool)
        if p.get("ignore"):
            ign = read_mask(project / p["ignore"]) > 0
            if p.get("ignore_dilate_px"):
                k = int(p["ignore_dilate_px"])
                ign = cv2.dilate(ign.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        ss = ssim_map(a, b)
        lmap = None
        if lp is not None:
            t = lambda im: torch.tensor(im, dtype=torch.float32, device="cuda").permute(2, 0, 1)[None] / 127.5 - 1
            with torch.no_grad():
                lmap = lp(t(a), t(b))[0, 0].cpu().numpy()
        rr = {}
        for rn, mp in (p.get("regions") or {"all": "all"}).items():
            m = np.ones(a.shape[:2], bool) if mp == "all" else read_mask(project / mp) > 0
            m &= ~ign
            if m.sum() < 50:
                rr[rn] = None
                continue
            d = a[m].astype(np.float32) - b[m].astype(np.float32)
            mse = float((d ** 2).mean())
            rr[rn] = {"pixels": int(m.sum()), "psnr_db": float(10 * np.log10(255 ** 2 / max(mse, 1e-6))),
                      "mae": float(np.abs(d).mean()), "ssim": float(ss[m].mean()),
                      "lpips": float(lmap[m].mean()) if lmap is not None else None}
        res[p["id"]] = rr
        if heatmaps:
            dd = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1)
            heat = cv2.applyColorMap(np.clip(dd * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].copy()
            heat[ign] = 110
            write_rgb((project / out).with_name(f"{Path(out).stem}_{p['id']}.jpg"),
                      contact_sheet([a, b, heat], cols=3, width=560, labels=[p["a"], p["b"], "|a-b| x4"]))
    write_json(res, project / out)
    return {"pairs": res, "outputs": [out]}
