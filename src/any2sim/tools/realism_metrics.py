"""How distinguishable are rendered frames from real ones? Unpaired realism statistics on image crops.

Real and rendered frames rarely show the same content (a human demo vs the robot), so whole frames would be told
apart by content. The comparison uses 224 px crops at native resolution from regions both sides share: real crops
avoid excluded masks (people, hands, controllers); rendered crops avoid the robot and its cables (synth_render label
masks). Measures:
  kid_x1000      kernel inception distance on DINOv2 (ViT-L/14 reg) crop features (lower = closer; 0 = same)
  c2st_auc       a linear classifier real vs rendered on those features, 5-fold cross-validated AUC (0.5 =
                 indistinguishable, 1.0 = trivially separable)
  low_level      per set: noise (MAD of the Laplacian on flat crops), sharpness (high-frequency power share),
                 luminance p5/p50/p95, saturation, colourfulness; and the rendered / real ratio of each
params:
  real: {frames: work/frames, step: 4, exclude: [{masks: dir} | {label_maps: dir, labels: [..]}]}
  synth: {renders: synthesis/renders, camera: head, episodes: null | [ids], source: video | png}
         (check frames + label masks; video: the encoded dataset frame, png: the pre-encoding check image)
         | {videos: <glob>, step: 10} (no masks: whole frames)
  crops_per_image: 4, max_crops: 1200, seed: 0
  out: evidence/realism_metrics.json
Writes the json and realism_crops.jpg (random real | rendered crops).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from ..common import contact_sheet, frame_files, read_mask, read_rgb, write_json, write_rgb

ENV = "vision"
GPU = True


def _crops(img, valid, n, rng, size=224, min_valid=0.9):
    H, W = img.shape[:2]
    out = []
    if H < size or W < size:
        return out
    for _ in range(n * 8):
        y, x = int(rng.integers(0, H - size + 1)), int(rng.integers(0, W - size + 1))
        if valid is None or valid[y:y + size, x:x + size].mean() >= min_valid:
            out.append(img[y:y + size, x:x + size])
            if len(out) >= n:
                break
    return out


def _low_level(crops):
    import cv2
    st = {"noise": [], "sharpness": [], "luma": [], "sat": [], "colourfulness": []}
    for c in crops:
        g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)                # fine-scale residual: sensor noise + compression
        st["noise"].append(float(np.median(np.abs(hp))) * 1.4826)
        F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
        h, w = F.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
        st["sharpness"].append(float(F[r > 0.5].sum() / max(F.sum(), 1e-9)))
        st["luma"].append(g.ravel())
        hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
        st["sat"].append(float(hsv[..., 1].mean()))
        R, G, B = [c[..., i].astype(float) for i in range(3)]
        rg, yb = R - G, 0.5 * (R + G) - B
        st["colourfulness"].append(float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())))
    lum = np.concatenate(st["luma"])
    return {"noise": float(np.median(st["noise"])), "sharpness": float(np.median(st["sharpness"])),
            "luma_p5": float(np.percentile(lum, 5)), "luma_p50": float(np.percentile(lum, 50)),
            "luma_p95": float(np.percentile(lum, 95)), "saturation": float(np.mean(st["sat"])),
            "colourfulness": float(np.mean(st["colourfulness"]))}


def _features(crops, device):
    import torch
    from .. import config as C
    hub = Path(C.config()["models"]["torch_home"]) / "hub"
    model = torch.hub.load(str(hub / "facebookresearch_dinov2_main"), "dinov2_vitl14_reg", source="local", pretrained=False)
    sd = torch.load(hub / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth", map_location="cpu")
    model.load_state_dict(sd)
    model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            x = torch.from_numpy(np.stack(crops[i:i + 64])).to(device).permute(0, 3, 1, 2).float() / 255
            x = (x - mean) / std
            feats.append(model(x).float().cpu().numpy())
    return np.vstack(feats)


def kid(a, b, subsets=20, size=200, seed=0):
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    vals = []
    m = min(size, len(a), len(b))
    for _ in range(subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def c2st_auc(a, b, folds=5, seed=0):
    """Cross-validated AUC of an L2-regularised logistic regression (torch, full batch)."""
    import torch
    X = np.vstack([a, b]).astype(np.float32)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))].astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    aucs = []
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b0 = torch.zeros(1, requires_grad=True)
        Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
        opt = torch.optim.LBFGS([w, b0], max_iter=200)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b0, yt) + 1e-2 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        s = (torch.from_numpy(X[te]) @ w + b0).detach().numpy()
        yy = y[te]
        pos, neg = s[yy == 1], s[yy == 0]
        if len(pos) and len(neg):
            aucs.append(float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))
    return float(np.mean(aucs)), float(np.std(aucs))


def run(project: Path, real: dict, synth: dict, crops_per_image: int = 4, max_crops: int = 1200, seed: int = 0,
        out: str = "evidence/realism_metrics.json") -> dict:
    import cv2
    import torch
    rng = np.random.default_rng(seed)
    # real crops
    src = frame_files(project / real.get("frames", "work/frames"))
    rc = []
    for f in range(0, len(src), int(real.get("step", 4))):
        im = read_rgb(src[f])
        valid = np.ones(im.shape[:2], bool)
        for ex in real.get("exclude") or []:
            if "masks" in ex:
                for p in sorted((project / ex["masks"]).glob(f"*{f:05d}*.png"))[:1]:
                    valid &= ~(read_mask(p) > 0)
            elif "label_maps" in ex:
                for p in sorted((project / ex["label_maps"]).glob(f"*{f:05d}*.png"))[:1]:
                    L = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    valid &= ~np.isin(L, ex.get("labels") or [l for l in np.unique(L) if l != 0])
        valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        rc += _crops(im, valid, crops_per_image, rng)
    # rendered crops
    sc = []
    if synth.get("renders"):
        cam = synth.get("camera", "head")
        eps = synth.get("episodes")                         # e.g. ["nominal"]: realism without randomisation
        for lab in sorted((project / synth["renders"]).glob(f"*/check/{cam}_*_label.png")):
            if eps and lab.parent.parent.name not in eps:
                continue
            img = lab.with_name(lab.name.replace("_label", ""))
            if not img.exists():
                continue
            L = cv2.imread(str(lab), cv2.IMREAD_UNCHANGED)
            valid = ~np.isin(L, [1, 4])
            valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            vid = lab.parent.parent / f"{cam}.mp4"
            if synth.get("source", "video") == "video" and vid.exists():
                # what the dataset contains: the encoded video frame (compression included)
                import imageio.v2 as imageio
                k = int(lab.stem.split("_")[-2])
                rd = imageio.get_reader(str(vid))
                frame = rd.get_data(k)
                rd.close()
            else:
                frame = read_rgb(img)
            sc += _crops(frame, valid, crops_per_image * 3, rng)
    else:
        import imageio.v2 as imageio
        for v in sorted(glob.glob(str(project / synth["videos"]))):
            r = imageio.get_reader(v)
            n = r.count_frames()
            for f in range(0, n, int(synth.get("step", 10))):
                sc += _crops(r.get_data(f), None, crops_per_image, rng)
            r.close()
    rng.shuffle(rc)
    rng.shuffle(sc)
    rc, sc = rc[:max_crops], sc[:max_crops]
    if len(rc) < 30 or len(sc) < 30:
        raise ValueError(f"too few crops (real {len(rc)}, rendered {len(sc)})")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fr, fs = _features(rc, dev), _features(sc, dev)
    k, ks = kid(fr, fs)
    auc, aucs = c2st_auc(fr, fs)
    lr, ls = _low_level(rc), _low_level(sc)
    res = {"crops": {"real": len(rc), "rendered": len(sc)}, "kid_x1000": round(1000 * k, 3), "kid_std_x1000": round(1000 * ks, 3),
           "c2st_auc": round(auc, 4), "c2st_auc_std": round(aucs, 4),
           "low_level": {"real": lr, "rendered": ls,
                         "ratio_rendered_over_real": {kk: round(ls[kk] / lr[kk], 3) if lr[kk] else None for kk in lr}}}
    tiles = rc[:8] + sc[:8]
    labels = ["real"] * len(rc[:8]) + ["rendered"] * len(sc[:8])
    write_rgb(project / Path(out).with_name("realism_crops.jpg"), contact_sheet(tiles, cols=8, width=224, labels=labels))
    write_json(res, project / out)
    return {**res, "outputs": [out, str(Path(out).with_name("realism_crops.jpg"))]}
