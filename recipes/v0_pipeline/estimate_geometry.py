"""Cameras and metric geometry from the video alone.

  * MoGe-2 on the reference frame (and every `moge_every`-th frame): metric point map, intrinsics, normals.
    The per-frame focal lengths and static-scene depth scales are compared (consistency evidence).
  * camera poses, method "pnp": SIFT matches between each frame and the reference on static pixels (not dynamic,
    not tracked objects), 3-D points from the reference MoGe-2 point map, RANSAC PnP -> metric poses relative to
    the reference camera. Works for static, quasi-static and moving cameras that keep overlap with the reference.
  * Pi3X (when its checkpoint is present): an independent multi-frame estimate of intrinsics, poses and metric
    scale, reported as a cross-check (rotation / translation agreement, scale ratio, focal ratio).
Outputs work/geometry/: poses.npz (c2w per frame, reference = identity), moge_%05d.npz, pi3.npz (optional),
geometry.json; evidence/geometry_depth.jpg
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import contact_sheet, frame_files, read_mask, read_rgb, write_json, write_rgb

ENV = "geom"
GPU = True


def _colorize(d, valid=None):
    import cv2
    v = d[valid] if valid is not None and valid.any() else d[np.isfinite(d)]
    lo, hi = np.percentile(v, [2, 98]) if v.size else (0, 1)
    x = np.clip((np.nan_to_num(d) - lo) / max(hi - lo, 1e-6), 0, 1)
    c = cv2.applyColorMap((255 * (1 - x)).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    if valid is not None:
        c[~valid] = 0
    return c


def _static_mask(project, i, shape, dilate=15):
    import cv2
    m = np.zeros(shape, bool)
    for sub in ("dynamic", "objects"):
        p = project / "work" / "masks" / sub / f"frame{i:05d}.png"
        if p.exists():
            m |= read_mask(p) > 0
    return ~(cv2.dilate(m.astype(np.uint8), np.ones((dilate, dilate), np.uint8)) > 0)


def pnp_poses(project, files, ref, P_ref, valid_ref, K, min_inliers=40):
    """c2w (camera i -> reference camera) for every frame; None where PnP fails."""
    import cv2
    sift = cv2.SIFT_create(6000)
    matcher = cv2.BFMatcher()
    g_ref = cv2.cvtColor(read_rgb(files[ref]), cv2.COLOR_RGB2GRAY)
    m_ref = _static_mask(project, ref, g_ref.shape) & valid_ref
    k_ref, d_ref = sift.detectAndCompute(g_ref, m_ref.astype(np.uint8) * 255)
    pts_ref = np.float32([k.pt for k in k_ref])
    X_ref = P_ref[pts_ref[:, 1].round().astype(int).clip(0, P_ref.shape[0] - 1),
                  pts_ref[:, 0].round().astype(int).clip(0, P_ref.shape[1] - 1)]
    out, info = [], []
    for i, f in enumerate(files):
        if i == ref:
            out.append(np.eye(4))
            info.append({"frame": i, "inliers": len(k_ref), "reproj_px": 0.0})
            continue
        g = cv2.cvtColor(read_rgb(f), cv2.COLOR_RGB2GRAY)
        k, d = sift.detectAndCompute(g, _static_mask(project, i, g.shape).astype(np.uint8) * 255)
        if d is None or len(k) < min_inliers:
            out.append(None)
            info.append({"frame": i, "error": "few features"})
            continue
        mm = matcher.knnMatch(d_ref, d, k=2)
        good = [a for a, b in (x for x in mm if len(x) == 2) if a.distance < 0.75 * b.distance]
        X = np.float32([X_ref[g_.queryIdx] for g_ in good])
        uv = np.float32([k[g_.trainIdx].pt for g_ in good])
        fin = np.isfinite(X).all(1)
        X, uv = X[fin], uv[fin]
        if len(X) < min_inliers:
            out.append(None)
            info.append({"frame": i, "error": f"{len(X)} matches"})
            continue
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X, uv, K, None, reprojectionError=2.0, iterationsCount=2000,
                                                 flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < min_inliers:
            out.append(None)
            info.append({"frame": i, "error": "pnp failed"})
            continue
        inl = inl[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(X[inl], uv[inl], K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        T_cam_ref = np.eye(4)
        T_cam_ref[:3, :3], T_cam_ref[:3, 3] = R, tvec[:, 0]
        proj, _ = cv2.projectPoints(X[inl], rvec, tvec, K, None)
        err = np.linalg.norm(proj[:, 0] - uv[inl], axis=1)
        out.append(np.linalg.inv(T_cam_ref))
        info.append({"frame": i, "inliers": int(len(inl)), "reproj_px": float(np.median(err))})
    return out, info


def plane_ecc_refine(project, files, ref, K, n_plane, d_plane, table_mask, init_c2w, scale=0.5, iters=100,
                     min_cc=0.7, max_rot_deg=1.0, max_trans_m=0.02):
    """Per-frame dense refinement of initial poses on the table plane: the plane-induced homography of the initial
    pose seeds an ECC homography fit (reference -> frame i) on static table pixels (frame i's hands/objects
    excluded); the refined homography is decomposed with the known metric plane (n.X = d, reference camera).
    A refinement is kept only if ECC correlation >= min_cc and it moves the pose by less than the limits."""
    import cv2
    from ..geom3d import rot_angle_deg
    S = np.diag([scale, scale, 1.0])
    Kinv = np.linalg.inv(K)
    def prep(i):
        g = cv2.cvtColor(read_rgb(files[i]), cv2.COLOR_RGB2GRAY)
        return cv2.GaussianBlur(cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    g_ref = prep(ref)
    h, w = g_ref.shape
    base = cv2.resize((table_mask & _static_mask(project, ref, table_mask.shape)).astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)
    base = cv2.erode(base, np.ones((5, 5), np.uint8))
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
    out, info = [], []
    for i, f in enumerate(files):
        T0 = np.linalg.inv(init_c2w[i])                     # reference camera -> camera i
        if i == ref:
            out.append(np.eye(4))
            info.append({"frame": i, "status": "reference"})
            continue
        H0 = K @ (T0[:3, :3] + np.outer(T0[:3, 3], n_plane) / d_plane) @ Kinv
        Wm = (S @ H0 @ np.linalg.inv(S)).astype(np.float32)
        Wm /= Wm[2, 2]
        g = prep(i)
        free = cv2.resize(_static_mask(project, i, table_mask.shape).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        try:
            tbl_i = cv2.warpPerspective(base, Wm, (w, h), flags=cv2.INTER_NEAREST)
            cc, Wr = cv2.findTransformECC(g_ref, g, Wm.copy(), cv2.MOTION_HOMOGRAPHY, crit, (tbl_i & free).astype(np.uint8), 5)
        except cv2.error:
            out.append(init_c2w[i])
            info.append({"frame": i, "status": "init (ECC failed)"})
            continue
        H_ri = np.linalg.inv(S) @ Wr.astype(np.float64) @ S
        n_, Rs, ts, ns = cv2.decomposeHomographyMat(H_ri, K)
        Hn = H_ri / H_ri[2, 2]
        best, best_err = None, np.inf
        for k in range(n_):
            for sgn in (1.0, -1.0):
                R, t = Rs[k], sgn * ts[k][:, 0] * d_plane
                Hp = K @ (R + np.outer(t, n_plane) / d_plane) @ Kinv
                err = np.abs(Hp / Hp[2, 2] - Hn).max()
                if err < best_err:
                    best, best_err = (R, t), err
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = best
        dr = rot_angle_deg(T[:3, :3] @ T0[:3, :3].T)
        dt = float(np.linalg.norm(T[:3, 3] - T0[:3, 3]))
        if cc >= min_cc and dr <= max_rot_deg and dt <= max_trans_m:
            out.append(np.linalg.inv(T))
            info.append({"frame": i, "status": "refined", "ecc": float(cc), "d_rot_deg": dr, "d_trans_m": dt})
        else:
            out.append(init_c2w[i])
            info.append({"frame": i, "status": "init (refinement rejected)", "ecc": float(cc), "d_rot_deg": dr, "d_trans_m": dt})
    return np.stack(out), info


def run(project: Path, ref_frame: int = 0, moge_every: int = 40, pixel_limit: int = 255000, pi3_step: int = 3,
        use_pi3x: bool = True) -> dict:
    import cv2
    from ..config import model_path
    from ..geom3d import interpolate_poses, rot_angle_deg
    from ..models.geometry import run_moge2, run_pi3x
    files = frame_files(project / "work" / "frames")
    n = len(files)
    out = project / "work" / "geometry"
    out.mkdir(parents=True, exist_ok=True)
    ref_rgb = read_rgb(files[ref_frame])
    H, W = ref_rgb.shape[:2]

    # 1. MoGe-2 on the reference and spread frames
    moge_ids = sorted(set([ref_frame] + list(range(0, n, moge_every))))
    moge = {}
    for i in moge_ids:
        r = run_moge2(read_rgb(files[i]), str(model_path("moge2_ckpt")))
        moge[i] = r
        np.savez_compressed(out / f"moge_{i:05d}.npz", **r)
    mr = moge[ref_frame]
    K = mr["K"].astype(np.float64)

    # 2. poses: Pi3X (robust on low texture, all keyframes) gives the initial pose of every frame, scaled to
    #    MoGe-2 metres on static reference pixels; each frame is then refined independently by ECC on the metric
    #    table plane. Sparse SIFT + PnP (accurate where texture allows) cross-checks the result.
    from ..geom3d import fit_plane_ransac
    P_ref = np.where(mr["mask"][..., None], mr["points"], np.nan)
    tmask = read_mask(project / "work" / "masks" / "static" / "table.png") > 0
    st_ref = _static_mask(project, ref_frame, (H, W))
    tpx = tmask & st_ref & mr["mask"] & np.isfinite(P_ref).all(-1)
    n_pl, d_pl, _ = fit_plane_ransac(P_ref[tpx], thresh=0.006)
    if d_pl < 0:
        n_pl, d_pl = -n_pl, -d_pl                           # n.X + d = 0, normal towards the camera
    n_away, d_away = -n_pl, d_pl                            # n'.X = d' with n' pointing away from the camera
    pi3, init, init_src = None, None, "identity (no Pi3X)"
    ck = model_path("pi3x_ckpt")
    if use_pi3x and ck.exists():
        ids = sorted(set(range(0, n, pi3_step)) | {ref_frame, n - 1})
        ids.remove(ref_frame)
        ids = [ref_frame] + ids
        pi = run_pi3x([read_rgb(files[i]) for i in ids], str(model_path("pi3_repo")), str(ck), pixel_limit)
        w, h = pi["size"]
        Kp = pi["K"].copy()
        Kp[0] *= W / w
        Kp[1] *= H / h
        d_pi = cv2.resize(pi["local_points"][0][..., 2], (W, H))
        sel = st_ref & mr["mask"] & (d_pi > 0)
        scale = float(np.median(mr["depth"][sel] / d_pi[sel]))   # MoGe metres per Pi3X (metric) unit
        order = np.argsort(ids)
        kid = np.asarray(ids)[order]
        kc2w = pi["c2w"][order].astype(np.float64)
        kc2w[:, :3, 3] *= scale
        init = interpolate_poses(kid, kc2w, np.arange(n))
        init_src = "Pi3X keyframes (every %d frames), translation in MoGe-2 metres" % pi3_step
        np.savez_compressed(out / "pi3.npz", frame_ids=kid, c2w=pi["c2w"][order], K_full=Kp,
                            local_points=pi["local_points"][order].astype(np.float16),
                            conf=pi["conf"][order].astype(np.float16), metric=np.float32(pi["metric"] or np.nan))
        pi3 = {"K_full": Kp.tolist(), "focal_ratio_pi3x_over_moge": float(Kp[0, 0] / K[0, 0]),
               "metric_scale_ratio_moge_over_pi3x": scale, "frames": len(ids)}
    if init is None:
        init = np.tile(np.eye(4), (n, 1, 1))
    allT, einfo = plane_ecc_refine(project, files, ref_frame, K, n_away, d_away, tmask, init)
    # an image identical to the previous one has the previous one's pose (static camera, duplicate frames)
    prev = cv2.resize(cv2.cvtColor(read_rgb(files[0]), cv2.COLOR_RGB2GRAY), (320, 180)).astype(np.float32)
    snapped = []
    for i in range(1, n):
        cur = cv2.resize(cv2.cvtColor(read_rgb(files[i]), cv2.COLOR_RGB2GRAY), (320, 180)).astype(np.float32)
        if float(np.abs(cur - prev).mean()) < 0.5:
            allT[i] = allT[i - 1]
            snapped.append(i)
        prev = cur
    pnp, pinfo = pnp_poses(project, files, ref_frame, P_ref, mr["mask"], K)
    agree = []
    for p in pinfo:
        i = p["frame"]
        if i != ref_frame and pnp[i] is not None and p.get("inliers", 0) >= 60:
            agree.append({"frame": i, "rot_deg": rot_angle_deg(pnp[i][:3, :3].T @ allT[i][:3, :3]),
                          "trans_m": float(np.linalg.norm(pnp[i][:3, 3] - allT[i][:3, 3])),
                          "init_rot_deg": rot_angle_deg(pnp[i][:3, :3].T @ init[i][:3, :3])})
    np.savez_compressed(out / "poses.npz", c2w=allT, measured=np.ones(n, bool), K=K, init_c2w=init,
                        method="Pi3X init + per-frame table-plane ECC refinement (MoGe-2 metric plane)")
    ang = [rot_angle_deg(T[:3, :3]) for T in allT]
    trans = [float(np.linalg.norm(T[:3, 3])) for T in allT]
    motion = "static" if max(ang) < 0.5 and max(trans) < 0.005 else \
        "quasi_static" if max(ang) < 5 and max(trans) < 0.05 else "moving"

    # 3. MoGe-2 consistency across frames: focal and static depth scale (depth vs. reference geometry seen by PnP)
    moge_cons = []
    for i, r in moge.items():
        st = _static_mask(project, i, (H, W)) & r["mask"]
        T_i_ref = np.linalg.inv(allT[i])
        Pr = P_ref.reshape(-1, 3)
        pc = Pr @ T_i_ref[:3, :3].T + T_i_ref[:3, 3]
        z = pc[:, 2]
        u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).round()
        v = (K[1, 1] * pc[:, 1] / z + K[1, 2]).round()
        okp = np.isfinite(z) & (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui, vi = u[okp].astype(int), v[okp].astype(int)
        sel = st[vi, ui]
        ratio = float(np.median(r["depth"][vi[sel], ui[sel]] / z[okp][sel])) if sel.sum() > 100 else None
        moge_cons.append({"frame": int(i), "focal_px": float(r["K"][0, 0]), "depth_ratio_vs_ref_geometry": ratio})

    good = [p for p in pinfo if "reproj_px" in p]
    info = {
        "ref_frame": ref_frame, "image_size": [W, H], "K": K.tolist(), "K_source": "MoGe-2 (reference frame)",
        "fov_x_deg": float(np.degrees(2 * np.arctan(W / (2 * K[0, 0])))),
        "poses": {"method": "initial poses from " + init_src + "; per-frame ECC refinement on the metric table plane",
                  "refined_frames": sum(e["status"] == "refined" for e in einfo),
                  "kept_initial_frames": [e["frame"] for e in einfo if e["status"].startswith("init")],
                  "ecc_median": float(np.median([e["ecc"] for e in einfo if "ecc" in e] or [0])),
                  "duplicate_frames_snapped": len(snapped),
                  "plane_cam": {"n": n_pl.tolist(), "d": float(d_pl)},
                  "pnp_crosscheck": {"frames": len(agree),
                                     "init_rot_deg_median": float(np.median([a["init_rot_deg"] for a in agree])) if agree else None,
                                     "rot_deg_median": float(np.median([a["rot_deg"] for a in agree])) if agree else None,
                                     "rot_deg_max": float(np.max([a["rot_deg"] for a in agree])) if agree else None,
                                     "trans_m_median": float(np.median([a["trans_m"] for a in agree])) if agree else None,
                                     "trans_m_max": float(np.max([a["trans_m"] for a in agree])) if agree else None,
                                     "pnp_frames_ok": len(good), "pnp_median_reproj_px": float(np.median(
                                         [p["reproj_px"] for p in good if p["frame"] != ref_frame] or [0]))}},
        "camera_motion": {"class": motion, "max_rotation_deg": max(ang), "max_translation_m": max(trans)},
        "moge_consistency": moge_cons,
        "pi3x": pi3 if pi3 else "Pi3X checkpoint not available; not run",
        "median_static_depth_m": float(np.median(mr["depth"][_static_mask(project, ref_frame, (H, W)) & mr["mask"]])),
    }
    write_json(info, out / "geometry.json")
    write_json({"pnp": pinfo, "ecc": einfo, "agreement": agree}, out / "pose_frames.json")
    static = _static_mask(project, ref_frame, (H, W))
    write_rgb(project / "evidence" / "geometry_depth.jpg", contact_sheet(
        [ref_rgb, _colorize(mr["depth"], mr["mask"]), (np.clip(mr["normal"] * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
         if "normal" in mr else ref_rgb, (static[..., None] * ref_rgb).astype(np.uint8)],
        cols=2, width=640, labels=["reference frame", "MoGe-2 metric depth", "MoGe-2 normals", "static pixels (pose + scale)"]))
    return {**{k: v for k, v in info.items() if k != "moge_consistency"},
            "moge_focal_px": [m["focal_px"] for m in moge_cons],
            "moge_depth_ratio": [m["depth_ratio_vs_ref_geometry"] for m in moge_cons],
            "outputs": ["work/geometry", "evidence/geometry_depth.jpg"]}
