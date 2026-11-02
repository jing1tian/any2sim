"""Metric hand tracks vs video and scene: events, scale cues, rigid registration, reprojection (numpy/opencv/scipy).

Hand tracks from VR / gloves are metric and accurate: they are the scale reference. They are placed rigidly
(T_world_tracks: rotation + translation, no scale). If the reconstructed scene disagrees, the scene is rescaled
(tool scene_rescale, a declared revision); the tracks are never stretched.

    from any2sim import handkit as hk
    tr = hk.load_tracks(project)                        # {side: {"kp": (T,21,3), "names": {...}, "wrist_T": ...}}
    ev = hk.track_events(tr["right"], fps=20)           # still/move segments, grasp/release dwell guesses
    cues = hk.scale_cues(project, tr["right"], ev, "pepper")   # registration-free scene-scale cues
    reg = hk.register_tracks(project, tr, frames, glove, contacts)   # rigid T + scene factor f, residuals
    chk = hk.reprojection(project, tr, reg["T_world_tracks"], frames, glove, scene_factor=1.0)

Conventions: scene factor f = (true metres) / (scene units); a correct scene has f = 1. A tracks point p (metres) is
at f^-1 * (T_world_tracks @ p) in the current scene's units (T_world_tracks maps into metric world coordinates of the
scene rescaled by f). Keypoint names: wrist, <finger>_<0..3> (tools.hand_retarget.layout_names).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


# ---------------------------------------------------------------- tracks
def load_tracks(project, npz: str = "inputs/hand_tracks.npz", layout="auto") -> dict:
    from .tools.hand_retarget import layout_names
    d = np.load(Path(project) / npz)
    out = {}
    for side in ("left", "right"):
        if f"{side}_keypoints" not in d.files:
            continue
        kp = d[f"{side}_keypoints"].reshape(len(d["t"]), -1, 3)
        W = np.tile(np.eye(4), (len(kp), 1, 1))
        W[:, :3, :3] = d[f"{side}_wrist_rotmat"].reshape(-1, 3, 3)
        W[:, :3, 3] = d[f"{side}_wrist_pos"]
        out[side] = {"kp": kp, "names": layout_names(layout, kp), "wrist_T": W, "t": d["t"], "frame": d["frame"]}
    return out


def pts(tr: dict, names) -> np.ndarray:
    """(T, len(names), 3) keypoints by semantic name."""
    return tr["kp"][:, [tr["names"][n] for n in names]]


def palm_centre(tr) -> np.ndarray:
    return pts(tr, ["wrist", "index_0", "middle_0", "ring_0", "pinky_0"]).mean(1)


def aperture(tr) -> np.ndarray:
    """Thumb tip to the nearer of index/middle tip (m), per frame."""
    p = pts(tr, ["thumb_3", "index_3", "middle_3"])
    return np.minimum(np.linalg.norm(p[:, 0] - p[:, 1], axis=1), np.linalg.norm(p[:, 0] - p[:, 2], axis=1))


def hand_length(tr) -> float:
    return float(np.median(np.linalg.norm(pts(tr, ["middle_3"])[:, 0] - pts(tr, ["wrist"])[:, 0], axis=1)))


def track_events(tr: dict, fps: float, still_mps: float = 0.05, min_frames: int = 5, smooth: int = 5) -> dict:
    """Still / moving segments of the palm with fingertip height and grip aperture, and a pick-and-place guess:
    transport = the motion with the largest horizontal displacement; grasp = the dwell just before it (or the last
    8 frames before it); release = the lowest fingertip moment within 1.5 s after it (+-3 frames). Frames are
    track rows. Check the guesses against the video before using them as contacts."""
    c = palm_centre(tr)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        pad = np.pad(c, ((smooth // 2, smooth // 2), (0, 0)), mode="edge")
        c = np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(3)], 1)
    v = np.r_[0, np.linalg.norm(np.diff(c, axis=0), axis=1)] * fps
    tips_z = pts(tr, [f"{f}_3" for f in FINGERS])[:, :, 2].min(1)
    ap = aperture(tr)
    still = v < still_mps
    def runs(mask):
        out, a = [], 0
        for i in range(1, len(mask) + 1):
            if i == len(mask) or mask[i] != mask[a]:
                out.append((a, i - 1, bool(mask[a])))
                a = i
        return out
    for a, b, st in runs(still):                           # short dwells are part of the motion
        if st and b - a + 1 < min_frames:
            still[a:b + 1] = False
    segs = []
    for a, b, st in runs(still):
        segs.append({"kind": "still" if st else "move", "frames": [int(a), int(b)],
                     "palm": c[a:b + 1].mean(0).round(4).tolist(), "tip_z_min": float(tips_z[a:b + 1].min()),
                     "aperture_min": float(ap[a:b + 1].min()), "aperture_mean": float(ap[a:b + 1].mean()),
                     "horizontal_m": float(np.linalg.norm(c[b, :2] - c[a, :2]))})
    out = {"segments": segs, "speed_mps": v}
    # pick-and-place guess: grasp = the lowest dwell that is neither the first nor the last segment; release = the
    # next fingertip low point within 2.5 s after it; transport = between them
    inner = [s for s in segs[1:-1] if s["kind"] == "still"] or [s for s in segs if s["kind"] == "still"]
    if inner:
        g = min(inner, key=lambda s: s["tip_z_min"])
        g0, g1 = g["frames"]
        hi = min(len(tips_z), g1 + int(2.5 * fps))
        if hi > g1 + 6:
            zz = tips_z[g1 + 6:hi]
            lo = g1 + 6 + int(np.argmin(zz))
            out["grasp"] = [int(g0), int(g1)]
            out["release"] = [int(max(g1 + 1, lo - 3)), int(min(len(tips_z) - 1, lo + 3))]
            out["transport"] = [int(g1), int(lo)]
            r = out["release"]
            out["transport_m"] = float(np.linalg.norm(c[r[0]:r[1] + 1, :2].mean(0) - c[g0:g1 + 1, :2].mean(0)))
    return out


# ---------------------------------------------------------------- scale cues (registration-free)
def _entity_vertices(project, scene, name, T):
    import trimesh
    m = trimesh.load(Path(project) / "scene" / scene["entities"][name]["geometry"]["visual"]["obj_uri"], force="mesh")
    V = np.asarray(m.vertices)
    return V @ T[:3, :3].T + T[:3, 3]


def scale_cues(project, tr: dict, events: dict, obj: str, finger_thickness_m: float = 0.016) -> dict:
    """Scene factor f (true m / scene unit) from cues that need no registration:
      transport: horizontal hand displacement between grasp and release dwells (m) / object displacement between
                 its first and last rest states (scene units)
      aperture:  grip aperture at the grasp dwell + finger thickness (m) / object's smallest horizontal width at
                 its first rest pose (scene units; the object is assumed grasped across that width)
    Both assume a pick-and-place of `obj`; check the events first."""
    from .scene_ir import load
    sc = load(Path(project) / "scene")
    e = sc["entities"][obj]
    st = sorted((e.get("states") or {}).items(), key=lambda kv: kv[1]["frames"][0])
    out = {}
    if "grasp" in events and len(st) >= 2:
        c = palm_centre(tr)
        g0, g1 = events["grasp"]
        r0, r1 = events["release"]
        hand = float(np.linalg.norm(c[r0:r1 + 1, :2].mean(0) - c[g0:g1 + 1, :2].mean(0)))
        A = np.asarray(st[0][1]["T_world_obj"], float)[:3, 3]
        B = np.asarray(st[-1][1]["T_world_obj"], float)[:3, 3]
        objd = float(np.linalg.norm(B[:2] - A[:2]))
        out["transport"] = {"hand_m": hand, "object_scene": objd, "f": hand / objd if objd > 0 else None}
        V = _entity_vertices(project, sc, obj, np.asarray(st[0][1]["T_world_obj"], float))
        Vc = V[:, :2] - V[:, :2].mean(0)
        widths = [float(np.ptp(Vc @ np.array([np.cos(a), np.sin(a)]))) for a in np.linspace(0, np.pi, 36, endpoint=False)]
        ap = float(np.median(aperture(tr)[g0:g1 + 1]))            # holding: median over the grasp dwell
        out["aperture"] = {"aperture_m": ap, "finger_thickness_m": finger_thickness_m, "object_width_scene": min(widths),
                           "f": (ap + finger_thickness_m) / min(widths)}
    fs = [v["f"] for v in out.values() if v.get("f")]
    out["f_median"] = float(np.median(fs)) if fs else None
    out["spread"] = float(np.ptp(fs) / np.median(fs)) if len(fs) > 1 else None
    return out


# ---------------------------------------------------------------- projection helpers
def to_scene(T_world_tracks, P, f: float = 1.0):
    """Track points (m) -> current scene units."""
    P = np.asarray(P, float)
    return (P @ np.asarray(T_world_tracks)[:3, :3].T + np.asarray(T_world_tracks)[:3, 3]) / f


BONES = [("wrist", "thumb_0"), ("thumb_0", "thumb_1"), ("thumb_1", "thumb_2"), ("thumb_2", "thumb_3")] + \
        [(a, b) for fg in FINGERS[1:] for a, b in (("wrist", f"{fg}_0"), (f"{fg}_0", f"{fg}_1"),
                                                   (f"{fg}_1", f"{fg}_2"), (f"{fg}_2", f"{fg}_3"))]


def skeleton_silhouette(Pw: dict, K, T_world_cam, W: int, H: int, radius: float) -> np.ndarray:
    """Binary silhouette of a hand skeleton {name: world point} drawn as capsules of `radius` (world units) plus the
    palm polygon."""
    import cv2
    from .geom3d import inv, project
    names = list(Pw)
    uv, z = project(np.asarray(K, float), inv(np.asarray(T_world_cam, float)), np.array([Pw[n] for n in names]))
    U = dict(zip(names, uv))
    Z = dict(zip(names, z))
    m = np.zeros((H, W), np.uint8)
    fx = float(np.asarray(K)[0, 0])
    if min(z) <= 0.02:
        return m.astype(bool)
    for a, b in BONES:
        if a in U and b in U:
            r = max(1, int(round(fx * radius / max(0.5 * (Z[a] + Z[b]), 1e-3))))
            cv2.line(m, tuple(np.round(U[a]).astype(int)), tuple(np.round(U[b]).astype(int)), 1, 2 * r)
    palm = [n for n in ("wrist", "thumb_0", "index_0", "middle_0", "ring_0", "pinky_0") if n in U]
    if len(palm) >= 3:
        hull = cv2.convexHull(np.round(np.array([U[n] for n in palm])).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def glove_mask(project, spec: dict, frame: int, H: int, W: int) -> np.ndarray | None:
    """{instances: dir} (frame%05d_<k>.png) | {masks: dir} | {label_maps: dir, labels: [..]}."""
    import cv2
    from .meshio import frame_mask
    p = Path(project)
    if "instances" in spec:
        fs = sorted((p / spec["instances"]).glob(f"frame{frame:05d}_*.png"))
        if not fs:
            return None
        m = np.zeros((H, W), bool)
        for f in fs:
            m |= cv2.imread(str(f), 0) > 0
        return m
    return frame_mask(p, spec, frame)


def _component_near(mask, ref):
    """Union of the connected components of `mask` that overlap `ref` (dilated)."""
    import cv2
    if mask is None or not mask.any():
        return mask
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    near = cv2.dilate(ref.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    keep = [k for k in range(1, n) if (near & (lab == k)).any()]
    return np.isin(lab, keep)


def _seg_dist(G, A, B):
    """Distance (px) from points G (n,2) to the nearest segment A[i]-B[i]."""
    AB = B - A
    t = np.clip(((G[:, None] - A[None]) * AB[None]).sum(-1) / np.maximum((AB ** 2).sum(-1), 1e-9)[None], 0, 1)
    return np.linalg.norm(G[:, None] - (A[None] + t[..., None] * AB[None]), axis=-1).min(1)


def reprojection(project, tracks: dict, T_world_tracks, frames, glove: dict, scene_factor: float = 1.0,
                 finger_radius_m: float = 0.009, frame_offset: int = 0) -> dict:
    """Tracked hand skeletons (rigid, unscaled) projected through the scene cameras vs glove masks, per frame and
    side: keypoint_to_glove (px from each projected keypoint to the glove mask: the hand lands on the glove),
    glove_to_hand (px from glove pixels to the nearest projected bone: the glove is explained by the hand), IoU and
    size ratio sqrt(area_mask / area_skeleton) of the capsule silhouette, centroid distance. Glove masks usually
    extend over cuffs and sleeves: the keypoint/coverage distances are the robust measures. frame = track row +
    frame_offset."""
    import cv2
    from .geom3d import inv, project as proj
    from .scene_ir import camera_poses, load
    sc = load(Path(project) / "scene")
    c = sc["cameras"]["source_cam"]
    K, W, H = np.asarray(c["K"], float), c["width"], c["height"]
    poses = camera_poses(Path(project) / "scene", sc)
    rows = []
    for f in frames:
        gm = glove_mask(project, glove, f + frame_offset, H, W)
        if gm is None:
            continue
        Tc = poses[min(f + frame_offset, len(poses) - 1)]
        for side, tr in tracks.items():
            P = to_scene(T_world_tracks, tr["kp"][f], scene_factor)
            sil = skeleton_silhouette({n: P[i] for n, i in tr["names"].items()}, K, Tc, W, H,
                                      finger_radius_m / scene_factor)
            if sil.sum() < 50:
                rows.append({"frame": int(f), "side": side, "visible": False, "reason": "tracked hand off image"})
                continue
            g = _component_near(gm, sil)
            if g is None or g.sum() < 50:
                rows.append({"frame": int(f), "side": side, "visible": False, "reason": "no glove under the tracked hand"})
                continue
            ys, xs = np.nonzero(sil)
            yg, xg = np.nonzero(g)
            uv, _ = proj(K, inv(Tc), P)
            dt = cv2.distanceTransform((~g).astype(np.uint8), cv2.DIST_L2, 3)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            kd = np.full(len(uv), 999.0)
            kd[ok] = dt[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
            nm = tr["names"]
            A = np.array([uv[nm[a]] for a, b in BONES if a in nm and b in nm])
            B = np.array([uv[nm[b]] for a, b in BONES if a in nm and b in nm])
            sel = np.random.default_rng(f).choice(len(xg), min(300, len(xg)), replace=False)
            gd = _seg_dist(np.stack([xg[sel], yg[sel]], 1).astype(float), A, B)
            rows.append({"frame": int(f), "side": side, "visible": True,
                         "keypoint_to_glove_px": float(np.median(kd)), "glove_to_hand_px": float(np.median(gd)),
                         "iou": float((sil & g).sum() / (sil | g).sum()),
                         "size_ratio": float(np.sqrt(g.sum() / sil.sum())),
                         "centroid_px": float(np.hypot(xs.mean() - xg.mean(), ys.mean() - yg.mean()))})
    vis = [r for r in rows if r.get("visible")]
    summ = {"frames": len({r["frame"] for r in rows}), "visible": len(vis),
            "on_glove_frac": float(len(vis) / max(len(rows), 1))}
    if vis:
        for k in ("keypoint_to_glove_px", "glove_to_hand_px", "iou", "size_ratio", "centroid_px"):
            v = np.array([r[k] for r in vis])
            summ[k] = {"median": float(np.median(v)), "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))}
    return {"summary": summ, "per_frame": rows}


# ---------------------------------------------------------------- rigid registration (+ scene factor)
def register_tracks(project, tracks: dict, frames, glove: dict, contacts: list | None = None,
                    fit_scene_factor: bool = True, scene_factor: float = 1.0, init: dict | None = None,
                    tilt: bool = True, frame_offset: int = 0, cap_px: float = 80.0, contact_share: float = 0.3,
                    px_per_cm: float = 10.0, samples: int = 80) -> dict:
    """Rigid placement of the tracks in the scene (+ optional scene factor f) from video evidence and contacts.

    Image term (two-sided, px, capped): tracked keypoints projected into frames (track row + frame_offset) must fall
    on the glove mask, and sampled glove pixels must lie near a projected hand bone (so two hands cannot share one
    glove). Contact term: contacts [{frame, side, keypoints: [names], entity | plane: "table", tol_m}] put the
    keypoints on the entity surface (its pose at that frame) or on the table plane z = 0, in metres (scene units *
    f), converted at px_per_cm and weighted to `contact_share` of the total. Both frames are taken as z up
    (gravity / table normal): yaw + translation (+ small tilt) and log f are optimised from a grid of yaws.
    Returns T_world_tracks (metric world of the scene rescaled by f), scene_factor, residuals."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation as Rot
    from .geom3d import inv, project as proj
    from .scene_ir import camera_poses, load, pose_at
    sc = load(Path(project) / "scene")
    c = sc["cameras"]["source_cam"]
    K, W, H = np.asarray(c["K"], float), c["width"], c["height"]
    poses = camera_poses(Path(project) / "scene", sc)
    rng = np.random.default_rng(0)
    dts, gpix = {}, {}
    for f in frames:
        g = glove_mask(project, glove, f + frame_offset, H, W)
        if g is not None and g.sum() > 100:
            dts[f] = cv2.distanceTransform((~g).astype(np.uint8), cv2.DIST_L2, 3)
            ys, xs = np.nonzero(g)
            k = rng.choice(len(xs), min(samples, len(xs)), replace=False)
            gpix[f] = np.stack([xs[k], ys[k]], 1).astype(float)
    fr = sorted(dts)
    surf = []
    for ct in contacts or []:
        if ct.get("plane") == "table":
            surf.append((ct, None))
        else:
            e = sc["entities"][ct["entity"]]
            T = pose_at(e, ct["frame"] + frame_offset)
            if T is None:
                T = pose_at(e, None)
            surf.append((ct, cKDTree(_entity_vertices(project, sc, ct["entity"], T))))
    bones = {side: [(tr["names"][a], tr["names"][b]) for a, b in BONES if a in tr["names"] and b in tr["names"]]
             for side, tr in tracks.items()}

    def unpack(x):
        R = Rot.from_euler("zyx", [x[0], x[1] if tilt else 0, x[2] if tilt else 0]).as_matrix()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, x[3:6]
        f = float(np.exp(x[6])) if fit_scene_factor else scene_factor
        return T, f

    def image_terms(T, f):
        a_, b_ = [], []
        for fr_ in fr:
            Tc = inv(poses[min(fr_ + frame_offset, len(poses) - 1)])
            segs = []
            for side, tr in tracks.items():
                uv, z = proj(K, Tc, to_scene(T, tr["kp"][fr_], f))
                ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                d = np.full(len(uv), cap_px)
                d[ok] = np.minimum(dts[fr_][uv[ok, 1].astype(int), uv[ok, 0].astype(int)], cap_px)
                a_.append(d)
                if (z > 0).all():
                    segs += [(uv[i], uv[j]) for i, j in bones[side]]
            G = gpix[fr_]
            if segs:
                A = np.array([s_[0] for s_ in segs])
                B = np.array([s_[1] for s_ in segs])
                AB = B - A
                tt = np.clip(((G[:, None] - A[None]) * AB[None]).sum(-1) / np.maximum((AB ** 2).sum(-1), 1e-9)[None], 0, 1)
                dd = np.linalg.norm(G[:, None] - (A[None] + tt[..., None] * AB[None]), axis=-1).min(1)
                b_.append(np.minimum(dd, cap_px))
            else:
                b_.append(np.full(len(G), cap_px))
        return np.concatenate(a_), np.concatenate(b_)

    n_img = None

    def resid(x):
        nonlocal n_img
        T, f = unpack(x)
        ia, ib = image_terms(T, f)
        img = np.r_[ia, ib]
        n_img = len(img)
        out = [img]
        cm = []
        for ct, tree in surf:
            tr = tracks[ct["side"]]
            P = to_scene(T, tr["kp"][ct["frame"], [tr["names"][n] for n in ct["keypoints"]]], f)
            dm = np.abs(P[:, 2]) * f if tree is None else tree.query(P)[0] * f
            cm.append(np.maximum(dm - ct.get("tol_m", 0.01), 0.0) * 100 * px_per_cm)
        if cm:
            cm = np.concatenate(cm)
            w = np.sqrt(contact_share / (1 - contact_share) * len(img) / len(cm))
            out.append(w * cm)
        return np.concatenate(out)

    if init is not None:
        starts = [np.r_[init["yaw"], 0, 0, init["t"], np.log(init.get("f", scene_factor))]]
    else:
        T0 = poses[0]
        look = T0[:3, 2] / np.linalg.norm(T0[:3, 2])
        tgt = T0[:3, 3] + 0.6 * look
        centre = np.mean([tr["kp"][fr[len(fr) // 2]].mean(0) for tr in tracks.values()], 0)
        starts = []
        for yaw in np.linspace(-np.pi, np.pi, 24, endpoint=False):
            R = Rot.from_euler("z", yaw).as_matrix()
            for fs in ([scene_factor] if not fit_scene_factor else [0.7, 1.0, 1.4]):
                starts.append(np.r_[yaw, 0, 0, tgt * fs - R @ centre, np.log(fs)])
    scored = sorted(((float(np.mean(resid(x0) ** 2)), i) for i, x0 in enumerate(starts)))[:4]
    best = None
    for _, i in scored:
        sol = least_squares(resid, starts[i], loss="linear", max_nfev=400, diff_step=1e-3)
        if best is None or sol.cost < best.cost:
            best = sol
    T, f = unpack(best.x)
    r = resid(best.x)
    ia, ib = image_terms(T, f)
    cm = []
    for ct, tree in surf:
        tr = tracks[ct["side"]]
        P = to_scene(T, tr["kp"][ct["frame"], [tr["names"][n] for n in ct["keypoints"]]], f)
        cm += ((np.abs(P[:, 2]) if tree is None else tree.query(P)[0]) * f).tolist()
    return {"T_world_tracks": T, "scene_factor": f, "frames": fr,
            "keypoint_to_glove_px": {"median": float(np.median(ia)), "p90": float(np.percentile(ia, 90))},
            "glove_to_hand_px": {"median": float(np.median(ib)), "p90": float(np.percentile(ib, 90))},
            "contact_m": [round(x, 4) for x in cm]}
