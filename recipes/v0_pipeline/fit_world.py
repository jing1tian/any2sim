"""World (table) frame, metric scale decision, per-frame cameras and the table's visible extent.

Inputs: work/geometry (estimate_geometry), work/masks (segment_scene).
  * table plane: RANSAC on MoGe-2 metric points of static table pixels -> world z up, origin where the reference
    optical axis meets the table, x = the reference camera's x axis projected on the table
  * scale: MoGe-2 metric by default; Pi3X metric (when run), MoGe-2 on other frames and object-size priors are
    compared and reported (disagreement > tolerance is a finding, not silently averaged)
  * cameras: per-frame poses from estimate_geometry (metric, relative to the reference) moved into the world
  * table extent: the table mask's outline on the plane; outline points on the image border are unobserved limits
Outputs: work/world/{world.json, cameras.json, points_ref_world.npz}, evidence/world_overlay.jpg
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import frame_files, read_json, read_mask, read_rgb, write_json, write_rgb
from ..geom3d import project as project_points
from ..geom3d import apply, fit_plane_ransac, hom, inv, pixel_rays, ray_plane

ENV = "base"
GPU = False


def _object_extent(P):
    """Horizontal max extent (PCA in xy) and height above the table of an object's visible points."""
    if len(P) < 30:
        return None
    xy = P[:, :2] - P[:, :2].mean(0)
    ev, evec = np.linalg.eigh(np.cov(xy.T))
    proj = xy @ evec
    lo, hi = np.percentile(proj, [2, 98], axis=0)
    return {"extent_xy_m": (hi - lo).tolist(), "max_extent_m": float((hi - lo).max()),
            "height_m": float(np.percentile(P[:, 2], 98)), "n_points": int(len(P))}


def run(project: Path, ref_frame: int | None = None, scale_source: str = "moge", size_priors: dict | None = None,
        scale_tolerance: float = 0.15, plane_thresh_m: float = 0.006) -> dict:
    import cv2
    geo = read_json(project / "work" / "geometry" / "geometry.json")
    ref = geo["ref_frame"] if ref_frame is None else ref_frame
    moge = np.load(project / "work" / "geometry" / f"moge_{ref:05d}.npz")
    poses = np.load(project / "work" / "geometry" / "poses.npz")
    files = frame_files(project / "work" / "frames")
    rgb = read_rgb(files[ref])
    H, W = rgb.shape[:2]
    P, valid, K = moge["points"], moge["mask"], moge["K"].astype(float)
    labels = read_json(project / "work" / "masks" / "labels.json")
    dyn = read_mask(project / "work" / "masks" / "dynamic" / f"frame{ref:05d}.png") > 0
    obj = read_mask(project / "work" / "masks" / "objects" / f"frame{ref:05d}.png")
    table = read_mask(project / "work" / "masks" / "static" / "table.png") > 0
    k15 = np.ones((15, 15), np.uint8)
    busy = cv2.dilate(((obj > 0) | dyn).astype(np.uint8), k15) > 0
    tbl_px = table & ~busy & valid & np.isfinite(P).all(-1)

    # 1. scale (MoGe-2 metres unless overridden by a number)
    s = 1.0 if scale_source == "moge" else float(scale_source)
    P = P * s

    # 2. table plane and world frame
    n, d, inl = fit_plane_ransac(P[tbl_px], thresh=plane_thresh_m)
    if d < 0:                                               # normal towards the camera (camera at the origin)
        n, d = -n, -d
    o = ray_plane(np.zeros((1, 3)), np.array([[0, 0, 1.0]]), n, d)[0]
    ex = np.array([1.0, 0, 0]) - n * n[0]
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    R = np.stack([ex, ey, n], 1)                             # world axes in camera coordinates
    T_world_camref = hom(R.T, -R.T @ o)                      # p_world = R^T (p_cam - o)
    res_tbl = np.abs(P[tbl_px] @ n + d)
    Pw = apply(T_world_camref, P.reshape(-1, 3)).reshape(P.shape)

    # 3. per-frame cameras (reference = identity, metres) moved into the world frame
    c2w = poses["c2w"].astype(float).copy()
    c2w[:, :3, 3] *= s
    allT = np.einsum("ij,njk->nik", T_world_camref, c2w)
    ids = np.nonzero(poses["measured"])[0]
    cam_pos = allT[:, :3, 3]

    # 4. table extent from the table mask outline (holes from objects/hands filled)
    tm = cv2.morphologyEx(table.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    cnts, _ = cv2.findContours(tm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    border = (c[:, 0] <= 2) | (c[:, 1] <= 2) | (c[:, 0] >= W - 3) | (c[:, 1] >= H - 3)
    rays = pixel_rays(K, c)
    hit = ray_plane(np.zeros((len(c), 3)), rays, n, d)
    hit_w = apply(T_world_camref, hit)
    obs_pts = hit_w[~border][:, :2]
    rect = cv2.minAreaRect(hit_w[:, :2].astype(np.float32))
    box = cv2.boxPoints(rect)
    foot = cam_pos[ref].copy()
    foot[2] = 0

    # 5. object extents (visible points, reference frame) vs priors
    objects = {}
    for k, name in labels["labels"].items():
        m = (obj == int(k)) & ~cv2.dilate(dyn.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & valid
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        objects[name] = _object_extent(Pw[m])
    checks = {}
    for name, pr in (size_priors or {}).items():
        e = objects.get(name)
        if not e:
            continue
        val = e[pr.get("measure", "max_extent_m")]
        lo, hi = pr["range_m"]
        mid = 0.5 * (lo + hi)
        checks[name] = {"measured_m": val, "prior_range_m": [lo, hi], "implied_scale": mid / val,
                        "within_prior": bool(lo <= val <= hi), "note": pr.get("note")}
    implied = [v["implied_scale"] for v in checks.values()]
    pi3 = geo.get("pi3x")
    pi3_vs_moge = 1.0 / pi3["metric_scale_ratio_moge_over_pi3x"] if isinstance(pi3, dict) else None
    other = [m["depth_ratio_vs_ref_geometry"] for m in geo["moge_consistency"]
             if m["frame"] != ref and m["depth_ratio_vs_ref_geometry"]]
    findings, rejected = [], {}
    if pi3_vs_moge is not None and abs(pi3_vs_moge - 1) > scale_tolerance:
        findings.append(f"Pi3X metric scale differs from MoGe-2 by {100 * (pi3_vs_moge - 1):+.1f}%")
        # a cue is rejected when the object sizes it implies fall outside every category prior
        outside = [nm for nm, v in checks.items() if not (v["prior_range_m"][0] <= v["measured_m"] * pi3_vs_moge <= v["prior_range_m"][1])]
        if checks and len(outside) == len(checks):
            sizes = ", ".join("%s %.3f m" % (nm, checks[nm]["measured_m"] * pi3_vs_moge) for nm in outside)
            rejected["pi3x_metric"] = f"implies {sizes}, outside the category priors"
    for r in other:
        if abs(r - 1) > scale_tolerance:
            findings.append(f"MoGe-2 on another frame disagrees with the reference scale by {100 * (r - 1):+.1f}%")
    for nm, v in checks.items():
        if not v["within_prior"]:
            findings.append(f"{nm}: measured {v['measured_m']:.3f} m outside prior {v['prior_range_m']}")

    world = {
        "frame": "table: +z up (plane normal), origin = reference optical axis on the table, x = reference camera x projected",
        "ref_frame": int(ref), "scale_source": scale_source, "scale_factor_applied": s,
        "scale_cues": {"moge2_metric_reference": 1.0, "moge2_other_frames_relative": other,
                       "pi3x_metric_relative_to_moge": pi3_vs_moge,
                       "object_priors": checks, "object_prior_implied_mean": float(np.mean(implied)) if implied else None},
        "rejected_cues": rejected,
        "scale_uncertainty_rel": float(max([abs(x - 1) for x in implied + other] +
                                           ([abs(pi3_vs_moge - 1)] if pi3_vs_moge is not None and "pi3x_metric" not in rejected else []) + [0.0])),
        "scale_uncertainty_note": "max relative deviation among accepted cues (MoGe-2 on other frames, object priors)",
        "findings": findings,
        "table_plane_cam": {"n": n.tolist(), "d": d, "inliers": int(inl.sum()), "points": int(tbl_px.sum()),
                            "residual_p50_mm": float(1000 * np.median(res_tbl)),
                            "residual_p90_mm": float(1000 * np.percentile(res_tbl, 90))},
        "T_world_camref": T_world_camref.tolist(),
        "camera_height_m": float(cam_pos[ref, 2]),
        "camera_pitch_down_deg": float(np.degrees(np.arcsin(-(T_world_camref[:3, :3] @ [0, 0, 1])[2]))),
        "table": {"observed_outline_xy": obs_pts[::10].tolist(), "outline_all_xy": hit_w[::10, :2].tolist(),
                  "outline_on_image_border": border[::10].tolist(),
                  "min_area_rect_xy": box.tolist(), "rect_size_m": list(rect[1]), "rect_angle_deg": rect[2],
                  "camera_foot_xy": foot[:2].tolist(),
                  "note": "outline points on the image border are view limits, not table edges"},
        "objects_visible_extent": objects,
        "camera_motion": geo["camera_motion"],
    }
    out = project / "work" / "world"
    out.mkdir(parents=True, exist_ok=True)
    write_json(world, out / "world.json")
    cams = {"model": "pinhole", "convention": "OpenCV optical frame; T_world_cam maps camera points into world",
            "width": W, "height": H, "K": K.tolist(), "K_source": "MoGe-2 (reference frame)",
            "K_pi3x": pi3["K_full"] if isinstance(pi3, dict) else None, "distortion": None, "distortion_note": "assumed none (not estimated)",
            "fps": 20.0, "T_world_cam": allT.tolist(), "keyframes": ids.tolist()}
    write_json(cams, out / "cameras.json")
    np.savez_compressed(out / "points_ref_world.npz", points=Pw.astype(np.float32), valid=valid,
                        rgb=rgb, table_px=tbl_px)

    # evidence: world axes, table outline and a 10 cm grid on the reference frame
    vis = rgb.copy()
    T_cam_w = inv(T_world_camref)
    for x in np.arange(-1.0, 1.01, 0.1):
        for a, b in (((x, -1, 0), (x, 1, 0)), ((-1, x, 0), (1, x, 0))):
            seg = np.linspace(a, b, 60)
            uv, z = project_points(K, T_cam_w, seg)
            uv = uv[z > 0].astype(int)
            cv2.polylines(vis, [uv.reshape(-1, 1, 2)], False, (255, 255, 255), 1, cv2.LINE_AA)
    for axis, col in ((0, (255, 0, 0)), (1, (0, 255, 0)), (2, (0, 0, 255))):
        e = np.zeros((2, 3))
        e[1, axis] = 0.1
        uv, _ = project_points(K, T_cam_w, e)
        cv2.arrowedLine(vis, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)), col, 3, cv2.LINE_AA, tipLength=0.2)
    uvb, _ = project_points(K, T_cam_w, np.c_[box, np.zeros(4)])
    cv2.polylines(vis, [uvb.astype(int).reshape(-1, 1, 2)], True, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f"cam h={world['camera_height_m']:.3f} m pitch={world['camera_pitch_down_deg']:.1f} deg",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    write_rgb(project / "evidence" / "world_overlay.jpg", vis)
    return {**{k: world[k] for k in ("scale_cues", "rejected_cues", "scale_uncertainty_rel", "findings", "table_plane_cam",
                                      "camera_height_m", "camera_pitch_down_deg", "objects_visible_extent")},
            "table_rect_m": list(rect[1]),
            "outputs": ["work/world", "evidence/world_overlay.jpg"]}
