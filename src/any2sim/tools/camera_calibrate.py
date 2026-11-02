"""Fit the camera's image model (any2sim.camera_isp) so renders look like the real camera: colour, tone, vignetting,
softness, chromatic shift, noise and compression, from renders aligned with real frames (same camera pose).

Paired fit on static pixels (moving things, objects and the simulated robot excluded): a near-identity colour matrix
and radial vignetting in linear light plus a per-channel tone curve (gain, gamma, black level), fitted jointly in
CIELAB with a robust loss and a ridge towards the identity; softness/sharpening chosen so the high frequencies match the real frames after the noise model and an H.264 round
trip at the real bitrate (what the dataset videos go through); chromatic shift by grid search. Noise comes from the real video alone (temporal differences of static pixels between consecutive frames:
variance vs intensity per channel, spatial correlation); compression from the video's bitrate. Every component is
validated on held-out frames and kept only if it does not make them worse.

params:
  camera: source_cam
  renders: project-relative dir with src_%05d.png (renders WITHOUT camera effects, e.g. synthesis/renders/nominal/head_frames)
  frames: "auto" | [ids]            paired frames (auto: every 6th frame with a render)
  holdout: 0.3                      fraction of paired frames used only for validation
  exclude: [{label_maps: dir, labels: [..]} | {masks: dir}]   moving/dynamic pixels (hands, objects, robot)
  exclude_robot: true               also exclude the robot and moving objects as simulated (render.motion; MuJoCo masks)
  exclude_dilate_px: 15
  video: null | project-relative video  (default: observation.yaml videos[0].uri) for bitrate and noise
  write: false                      store as scene.yaml cameras.<camera>.isp
Writes evidence/camera_calibrate.json and camera_calibrate.jpg (held-out: real | render | calibrated | |diff|).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import contact_sheet, frame_files, read_mask, read_rgb, write_json, write_rgb

ENV = "base"
GPU = False


def _nonregressing(candidate, baseline):
    """Both colour and structure must survive a held-out comparison."""
    return (candidate["ssim"] >= baseline["ssim"] - 1e-6
            and candidate["deltaE_median"] <= baseline["deltaE_median"] + 1e-6)


def _isotonic(x, y, w=None):
    """Pool-adjacent-violators: non-decreasing fit of y over sorted x."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wt, c = wts[-2] + wts[-1], cnt[-2] + cnt[-1]
            vals[-2:], wts[-2:], cnt[-2:] = [v], [wt], [c]
    return np.repeat(vals, cnt)


def _metrics(real, img, m):
    import cv2
    from skimage.metrics import structural_similarity as ssim
    g0 = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s = ssim(g0, g1, full=True)[1][m].mean()
    l0 = cv2.cvtColor(real, cv2.COLOR_RGB2LAB).astype(float)
    l1 = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(float)
    de = np.linalg.norm(l0 - l1, axis=-1)[m]
    h0 = np.histogram(g0[m], 64, (0, 255))[0].astype(float)
    h1 = np.histogram(g1[m], 64, (0, 255))[0].astype(float)
    emd = float(np.abs(np.cumsum(h0 / h0.sum()) - np.cumsum(h1 / h1.sum())).sum() * 4)
    return {"ssim": float(s), "deltaE_median": float(np.median(de)), "luma_hist_emd": emd}


def run(project: Path, renders: str, camera: str = "source_cam", frames="auto", holdout: float = 0.3,
        exclude: list | None = None, exclude_robot: bool = True, exclude_dilate_px: int = 15, video: str | None = None,
        write: bool = False) -> dict:
    import cv2
    import yaml
    from .. import camera_isp as ISP
    from ..scene_ir import load, save
    if camera != "source_cam":
        raise ValueError("camera_calibrate uses work/frames from source_cam; other cameras need their own paired reference frames")
    src = frame_files(project / "work" / "frames")
    from ..render_pipeline import calibration_renders
    rdir = calibration_renders(project / renders)
    have = sorted(int(p.stem.split("_")[-1]) for p in rdir.glob("src_*.png"))
    ids = [f for f in (have[::6] if frames == "auto" else frames) if f < len(src)]
    if len(ids) < 4:
        raise ValueError("need >= 4 paired frames")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ids))
    n_hold = max(2, int(round(holdout * len(ids))))
    hold = sorted(ids[i] for i in perm[:n_hold])
    fit = sorted(ids[i] for i in perm[n_hold:])

    sim_mask = None
    if exclude_robot:
        from .. import simkit as sk
        from ..robot.render import CalibRenderer
        from ..scene_ir import camera_poses, load_motion, motion_at
        scn = load(project / "scene")
        sim = sk.Sim(project, mdir=project / "work" / "camera_calibrate_mjcf")
        cc = scn["cameras"][camera]
        ren_ = CalibRenderer(sim.m, cc["width"], cc["height"], groups=(0, 1, 2))
        mot = load_motion(project / "scene", scn, scn["render"]["motion"]) if (scn.get("render") or {}).get("motion") else None
        Tc_all = camera_poses(project / "scene", scn, camera)
        dyn = np.array([b for b, o in sim.body_owner.items() if o[0] == "robot" or (o[0] == "entity" and o[1] not in sim.static)])

        def sim_mask(f):
            q, objs = motion_at(mot, f) if mot is not None else ({}, {})
            sim.reset(q, objs)
            seg = ren_.segment(sim.d, np.asarray(cc["K"], float), Tc_all[min(f, len(Tc_all) - 1)])
            return np.isin(seg, dyn)

    def static_mask(f, shape):
        m = np.ones(shape[:2], bool)
        if sim_mask is not None:
            mm = sim_mask(f)
            if mm.shape == m.shape:
                m &= ~mm
        for ex in exclude or []:
            if "label_maps" in ex:
                p = project / ex["label_maps"] / f"{f:05d}.png"
                if not p.exists():
                    cands = sorted((project / ex["label_maps"]).glob(f"*{f:05d}*"))
                    p = cands[0] if cands else None
                if p is not None and p.exists():
                    L = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    m &= ~np.isin(L, ex.get("labels") or [l for l in np.unique(L) if l != 0])
            elif "masks" in ex:
                cands = sorted((project / ex["masks"]).glob(f"*{f:05d}*.png"))
                for p in cands[:1]:
                    m &= ~(read_mask(p) > 0)
        if exclude_dilate_px:
            m = ~cv2.dilate((~m).astype(np.uint8), np.ones((2 * exclude_dilate_px + 1,) * 2, np.uint8)).astype(bool)
        return m

    pairs = {}
    for f in ids:
        real = read_rgb(src[f])
        ren = read_rgb(rdir / f"src_{f:05d}.png")
        if ren.shape != real.shape:
            ren = cv2.resize(ren, (real.shape[1], real.shape[0]), interpolation=cv2.INTER_AREA)
        sm = static_mask(f, real.shape)
        sm &= (ren.max(-1) > 3)                               # rendered background (camera rays missing everything)
        pairs[f] = (real, ren, sm)
    H, W = next(iter(pairs.values()))[0].shape[:2]
    r2 = ISP.radius2(H, W)
    # ---------------------------------------------------------- colour matrix + vignette (linear light)
    X, Y, R = [], [], []
    for f in fit:
        real, ren, sm = pairs[f]
        idx = np.flatnonzero(sm)
        idx = rng.choice(idx, min(len(idx), 40000), replace=False)
        X.append(ISP.srgb_to_lin(ren.reshape(-1, 3)[idx] / 255.0))
        Y.append(ISP.srgb_to_lin(real.reshape(-1, 3)[idx] / 255.0))
        R.append(r2.reshape(-1)[idx])
    X, Y, R = np.vstack(X), np.vstack(Y), np.concatenate(R)
    ok = X.max(1) < 0.995                                    # unclipped render values (real clipping is information)
    X, Y, R = X[ok], Y[ok], R[ok]
    # joint fit in CIELAB (perceptual): colour matrix + radial vignetting (linear light) + per-channel tone curve
    # (gain, gamma, black level on sRGB values); robust loss, ridge towards the identity
    from scipy.optimize import least_squares
    from skimage.color import rgb2lab
    sub = rng.choice(len(X), min(len(X), 30000), replace=False)
    Xs, Rs = X[sub], R[sub]
    Lab_t = rgb2lab(np.clip(ISP.lin_to_srgb(Y[sub]), 0, 1)[None])[0]

    def model(pv, Xl, Rr):
        Mx = pv[:9].reshape(3, 3)
        a1, a2 = pv[9:11]
        lin = (Xl @ Mx.T) * (1 + a1 * Rr + a2 * Rr * Rr)[:, None]
        sr = ISP.lin_to_srgb(lin)
        g, gam, blk = pv[11:14], pv[14:17], pv[17:20]
        return np.clip(blk + (1 - blk) * g * np.power(np.clip(sr, 1e-6, 1), gam), 0, 1)

    def resid(pv):
        lab = rgb2lab(model(pv, Xs, Rs)[None])[0]
        reg = np.r_[(pv[:9] - np.eye(3).ravel()) * 20, pv[9:11] * 5, (pv[14:17] - 1) * 10]
        return np.r_[(lab - Lab_t).ravel() / 4.0, reg]
    p0 = np.r_[np.eye(3).ravel(), 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    lo = np.r_[np.full(9, -0.5), -0.8, -0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, -0.05, -0.05, -0.05]
    hi = np.r_[np.full(9, 1.6), 0.3, 0.5, 1.4, 1.4, 1.4, 1.8, 1.8, 1.8, 0.1, 0.1, 0.1]
    sol = least_squares(resid, p0, bounds=(lo, hi), loss="soft_l1", f_scale=2.0, max_nfev=200, diff_step=1e-3)
    pv = sol.x
    M = pv[:9].reshape(3, 3)
    vig = pv[9:11]
    knots = np.linspace(0, 1, 33)
    g, gam, blk = pv[11:14], pv[14:17], pv[17:20]
    lut = {"x": knots.round(4).tolist(),
           "y": [np.clip(blk[c] + (1 - blk[c]) * g[c] * np.power(knots, gam[c]), 0, 1).round(4).tolist() for c in range(3)]}

    def render(img, isp):
        return ISP.apply(img, {k: v for k, v in isp.items() if k != "noise"}, key="cal")

    stages = [("identity", {}),
              ("colour+vignette", {"color_matrix": M.round(5).tolist(), "vignette": vig.round(5).tolist()}),
              ("+tone", {"color_matrix": M.round(5).tolist(), "vignette": vig.round(5).tolist(), "tone_lut": lut})]
    # Stages are nested: keep improvements without a held-out colour/structure regression.

    def score(isp, frames_):
        ms = [_metrics(pairs[f][0], render(pairs[f][1], isp), pairs[f][2]) for f in frames_]
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    hist = {}
    best_name, best = "identity", {}
    hist["identity"] = score({}, hold)
    for name, isp in stages[1:]:
        s = score(isp, hold)
        hist[name] = s
        b = hist[best_name]
        if _nonregressing(s, b):
            best_name, best = name, isp
    # ---------------------------------------------------------- tone by histogram matching (pooled static pixels)
    # insensitive to where light falls: matches contrast, black level and colour balance of the distributions when
    # the scene's lighting layout differs from reality (a per-pixel fit then trades colour for layout)
    qs_ = np.linspace(0.5, 99.5, 41)
    rq, tq = [], []
    for c in range(3):
        rv = np.concatenate([pairs[f][1][..., c][pairs[f][2]] for f in fit]).astype(np.float32) / 255
        tv = np.concatenate([pairs[f][0][..., c][pairs[f][2]] for f in fit]).astype(np.float32) / 255
        rq.append(np.percentile(rv, qs_))
        tq.append(np.percentile(tv, qs_))
    knots = np.linspace(0, 1, 33)
    hm = []
    for c in range(3):
        xr, yr = np.maximum.accumulate(rq[c]) + np.arange(len(qs_)) * 1e-6, np.maximum.accumulate(tq[c])
        y = np.interp(knots, xr, yr, left=max(0.0, yr[0] - (xr[0] - 0)), right=min(1.0, yr[-1] + (1 - xr[-1])))
        hm.append(np.clip(0.5 * y + 0.5 * np.interp(knots, knots, y), 0, 1).round(4).tolist())
    isp_h = {"tone_lut": {"x": knots.round(4).tolist(), "y": hm}}
    s = score(isp_h, hold)
    hist["histogram tone"] = s
    b_ = hist[best_name]
    if s["luma_hist_emd"] < b_["luma_hist_emd"] * 0.8 and _nonregressing(s, b_):
        best_name, best = "histogram tone", isp_h
    # ---------------------------------------------------------- levels + saturation (pooled statistics, layout-free)
    def pooled(frames_, which, isp=None):
        Ys, Ss = [], []
        for f in frames_:
            img = pairs[f][which] if isp is None else render(pairs[f][1], isp)
            x = img.astype(np.float32) / 255
            Y = x @ np.array([0.299, 0.587, 0.114], np.float32)
            m = pairs[f][2]
            Ys.append(Y[m])
            Ss.append((np.abs(x - Y[..., None]).mean(-1))[m])
        Y, S_ = np.concatenate(Ys), np.concatenate(Ss)
        return np.percentile(Y, [5, 50, 95]), float(np.mean(S_))
    pr_t, sat_t = pooled(fit, 0)
    pr_r, sat_r = pooled(fit, 1, best)
    lv = {**best, "levels": {"in": pr_r.round(4).tolist(), "out": pr_t.round(4).tolist()}}
    pr_r2, sat_r2 = pooled(fit, 1, lv)
    lv["saturation"] = round(float(np.clip(sat_t / max(sat_r2, 1e-4), 0.6, 1.8)), 3)
    s = score(lv, hold)
    hist["levels+saturation"] = s
    b_ = hist[best_name]
    if s["luma_hist_emd"] < b_["luma_hist_emd"] and _nonregressing(s, b_):
        best_name, best = "levels+saturation", lv
    # ---------------------------------------------------------- noise from the real video (temporal, static pixels)
    a_s, b_s, corr = [], [], []
    used = 0
    for f in ids:
        if f + 1 >= len(src) or used >= 12 or f not in pairs:
            continue
        r0 = read_rgb(src[f]).astype(np.float32) / 255
        r1 = read_rgb(src[f + 1]).astype(np.float32) / 255
        if np.abs(r1 - r0).mean() < 1e-4:                    # repeated frames (e.g. a static ramp): no sensor noise
            continue
        used += 1
        sm = pairs[f][2] & static_mask(f + 1, r0.shape)
        dif = (r1 - r0)
        sm &= np.abs(dif).max(-1) < 0.08                       # no motion
        mean = 0.5 * (r0 + r1)
        for c in range(3):
            v, mu = dif[..., c][sm], mean[..., c][sm]
            if len(v) < 1000:
                continue
            bins = np.linspace(0.05, 0.95, 10)
            bi = np.digitize(mu, bins)
            xs, vs = [], []
            for k in range(1, len(bins)):
                sel = bi == k
                if sel.sum() > 200:
                    xs.append(mu[sel].mean())
                    vs.append(np.var(v[sel]) / 2)
            if len(xs) >= 3:
                bb, aa = np.polyfit(xs, vs, 1)
                a_s.append((c, max(aa, 1e-6)))
                b_s.append((c, max(bb, 0.0)))
        d = dif[..., 1].copy()
        d[~sm] = 0
        if sm.sum() > 1000:
            ac = np.mean(d[:, 1:][sm[:, 1:] & sm[:, :-1]] * d[:, :-1][sm[:, 1:] & sm[:, :-1]]) / max(np.mean(d[sm] ** 2), 1e-9)
            corr.append(float(np.clip(ac, 0, 0.95)))
    noise = None
    if a_s:
        A = [float(np.median([v for c, v in a_s if c == k] or [1e-5])) for k in range(3)]
        Bv = [float(np.median([v for c, v in b_s if c == k] or [0.0])) for k in range(3)]
        rho = float(np.median(corr)) if corr else 0.0
        corr_px = float(np.sqrt(-1 / (4 * np.log(rho)))) if 0.05 < rho < 0.95 else 0.0   # Gaussian blur with that lag-1 correlation
        noise = {"a": np.round(A, 7).tolist(), "b": np.round(Bv, 7).tolist(), "corr_px": round(corr_px, 3)}
    # ---------------------------------------------------------- bitrate of the real video
    kbps_real = None
    vp = project / (video or ((yaml.safe_load((project / "observation.yaml").read_text()).get("videos") or [{}])[0].get("uri", "")))
    if vp.is_file():
        import imageio.v2 as imageio
        r = imageio.get_reader(str(vp))
        md = r.get_meta_data()
        dur = md.get("duration") or (r.count_frames() / max(md.get("fps", 20), 1))
        r.close()
        kbps_real = int(vp.stat().st_size * 8 / max(dur, 1e-3) / 1000)
    # ---------------------------------------------------------- softness / sharpening by matching high frequencies
    def edge_flat(img, m):
        """High-frequency amplitude on the image's own edges (top 15 % gradients: acutance) and flats (bottom 40 %:
        grain), inside the static mask m."""
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F, 0, 1))
        hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.5))
        hi, lo = np.percentile(gm[m], 85), np.percentile(gm[m], 40)
        return float(np.median(hp[m & (gm >= hi)])), float(np.median(hp[m & (gm <= lo)]))
    real_edge = np.median([edge_flat(pairs[f][0], pairs[f][2])[0] for f in fit[:6]])
    real_flat = np.median([edge_flat(pairs[f][0], pairs[f][2])[1] for f in fit[:6]])

    def encode(frames_):
        """Round trip through libx264 at the real bitrate: the dataset videos lose high frequencies the same way."""
        if not kbps_real:
            return frames_
        import tempfile
        import imageio.v2 as imageio
        with tempfile.TemporaryDirectory() as td:
            fn = str(Path(td) / "t.mp4")
            with imageio.get_writer(fn, fps=20, codec="libx264", macro_block_size=8,
                                    ffmpeg_params=["-b:v", f"{kbps_real}k"]) as w:
                for fr in frames_:
                    w.append_data(fr)
            r = imageio.get_reader(fn)
            out_ = [r.get_data(i) for i in range(len(frames_))]
            r.close()
        return out_

    def stats_of(isp):
        imgs = encode([ISP.apply(pairs[f][1], isp, key=("hf", f)) for f in fit[:6]])
        ef = [edge_flat(im, pairs[f][2]) for im, f in zip(imgs, fit[:6])]
        return np.median([e for e, _ in ef]), np.median([fl for _, fl in ef])
    # 1) optics by edge acutance (noise included), 2) noise amplitude by flat-area grain
    nz0 = noise or {}
    cands = [dict(best)] + [{**best, "blur_sigma_px": sb, "sharpen_amount": 0.8} for sb in (-1.2, -0.8, -0.5, 0.4, 0.7, 1.0)]
    scored = [(abs(np.log(max(stats_of({**c, **({"noise": nz0} if nz0 else {})})[0], 1e-6) / max(real_edge, 1e-6))), i)
              for i, c in enumerate(cands)]
    cand = cands[min(scored)[1]]
    if nz0:
        best_k, best_e = 1.0, np.inf
        for k_ in (0.5, 0.7, 1.0, 1.4, 2.0):
            nzk = {**nz0, "a": (np.asarray(nz0["a"]) * k_ ** 2).tolist(), "b": (np.asarray(nz0["b"]) * k_ ** 2).tolist()}
            e = abs(np.log(max(stats_of({**cand, "noise": nzk})[1], 1e-6) / max(real_flat, 1e-6)))
            if e < best_e:
                best_k, best_e = k_, e
        noise = {**nz0, "a": (np.asarray(nz0["a"]) * best_k ** 2).round(8).tolist(),
                 "b": (np.asarray(nz0["b"]) * best_k ** 2).round(8).tolist(), "scale_after_encoding": best_k}
    s = score(cand, hold)
    hist["+optics"] = s
    if _nonregressing(s, hist[best_name]):
        best = cand
        best_name = "+optics"
    def encoded_score(isp):
        imgs = encode([ISP.apply(pairs[f][1], isp, key=("heldout", f)) for f in hold])
        ms = [_metrics(pairs[f][0], im, pairs[f][2]) for im, f in zip(imgs, hold)]
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}

    hist["encoded_without_noise"] = encoded_score(best)
    if noise:
        hist["encoded_with_noise"] = encoded_score({**best, "noise": noise})
        if _nonregressing(hist["encoded_with_noise"], hist["encoded_without_noise"]):
            best = {**best, "noise": noise}
    if kbps_real:
        best["bitrate_kbps"] = kbps_real
    # ---------------------------------------------------------- sheet + result
    tiles, labels = [], []
    for f in hold[:3]:
        real, ren, sm = pairs[f]
        cal = ISP.apply(ren, best, key=("cal", f))
        tiles += [real, ren, cal, (np.abs(real.astype(int) - cal.astype(int)).clip(0, 255) * 3).clip(0, 255).astype(np.uint8)]
        labels += [f"real {f}", "render", "render + camera model", "|real - calibrated| x3"]
    write_rgb(project / "evidence" / "camera_calibrate.jpg", contact_sheet(tiles, cols=4, width=480, labels=labels))
    final = score({k: v for k, v in best.items() if k not in ("noise", "bitrate_kbps")}, hold)
    hist["encoded_final"] = encoded_score(best)
    res = {"camera": camera, "raw_renders": str(rdir.relative_to(project)),
           "fit_frames": fit, "holdout_frames": hold, "kept": best_name if best_name != "+optics" else "+optics",
           "holdout_metrics": {**hist, "final": final}, "isp": best}
    if write:
        sc = load(project / "scene")
        calibrated = {**best, "provenance": f"camera_calibrate: {len(fit)} paired frames, held-out "
                                        f"deltaE {hist['identity']['deltaE_median']:.1f} -> {final['deltaE_median']:.1f}, "
                                        f"SSIM {hist['identity']['ssim']:.3f} -> {final['ssim']:.3f}"}
        sc["cameras"][camera]["isp"] = calibrated
        if camera in (sc.get("render", {}).get("camera_isp") or {}):
            sc["render"]["camera_isp"][camera] = calibrated
        if (sc.get("render", {}).get("head_camera", "source_cam") == camera
                and "head" in (sc.get("render", {}).get("camera_isp") or {})):
            sc["render"]["camera_isp"]["head"] = calibrated
        save(project / "scene", sc)
    write_json(res, project / "evidence" / "camera_calibrate.json")
    return {**{k: v for k, v in res.items() if k != "isp"}, "isp_keys": sorted(best),
            "outputs": ["evidence/camera_calibrate.json", "evidence/camera_calibrate.jpg"]}
