"""Fit the visible cables / wires of a robot (robot.yaml `cables`, any2sim.robot.cables) to real images: anchor
points in link frames, cable length (sag) and service-loop parameters, from cable masks on frames with logged joints.

Each cable is projected through the bundle camera at the frame's joint values; samples hidden behind the robot
(MuJoCo depth) are ignored. The cost is symmetric: projected visible samples -> distance to the cable mask, and the
mask pixels assigned to this cable (near its initial projection) -> distance to the projected curve. Free parameters
move inside a box around the initial guess (least squares, robust loss).

Initial anchors can be given as link-frame points or as pixels on a frame: {"pixel": [u, v], "row": r, "link": opt}
is lifted onto the robot surface by a ray cast (nearest robot pixel within 80 px) and expressed in "link" (default:
the link that was hit), so anchors can be picked where a cable visibly leaves a part; "surface": "far" takes the
back side of the part (cables running behind it). A loop's centre can be a pixel anchor too.

params:
  bundle: "robot"                    project-relative bundle folder
  camera: <bundle camera name>
  frames: {images: dir, masks: dir, rows: "frame"} | [{image, mask, row}]   cable masks (e.g. sam3 "black cable")
  log, joints, q_fixed               joint values per row (as robot_overlay)
  cables: {<name>: <cable spec (robot/cables.py)> + fit: {points: [indices] | "all", box_m: 0.03,
                                                          length: [lo, hi] | null,
                                                          slack_dir: null | <link> (fit the slack direction)}}
  assign_px: 60                      mask pixels within this distance of a cable's initial projection belong to it
  scale: 0.5, iters: 200, starts: 300  (random restarts inside the box before least squares)
  write: false                       store the fitted cables in robot.yaml (provenance: fitted ...)
Writes <bundle>/checks/cables.json and cables_overlay.jpg (mask magenta, fitted cables green, initial red).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ..common import contact_sheet, read_mask, read_rgb, write_json, write_rgb

ENV = "base"
GPU = False


def _frames(project, frames):
    if isinstance(frames, list):
        return [(project / f["image"], project / f["mask"], int(f["row"])) for f in frames]
    imgs = sorted((project / frames["images"]).glob("frame*.jpg")) + sorted((project / frames["images"]).glob("frame*.png"))
    out = []
    for p in imgs:
        m = project / frames["masks"] / f"{p.stem}.png"
        if m.exists():
            out.append((p, m, int(p.stem[5:])))
    return out


def run(project: Path, camera: str, frames, log: str, joints: list, cables: dict, bundle: str = "robot",
        q_fixed: dict | None = None, assign_px: float = 60.0, scale: float = 0.5, iters: int = 200,
        starts: int = 300, write: bool = False) -> dict:
    import cv2
    import mujoco
    from scipy.optimize import least_squares
    from ..geom3d import inv, project as proj
    from ..robot import cables as CB
    from ..robot import datalog as DL
    from ..robot.bundle import Bundle
    from ..robot.render import CalibRenderer
    from ..robot.sim import kinematic_model, link_pose, set_q
    broot = Path(bundle) if Path(bundle).is_absolute() else project / bundle
    b = Bundle(broot)
    cam = b.profile["cameras"][camera]
    W, H = cam["width"], cam["height"]
    w, h = max(16, round(W * scale)), max(16, round(H * scale))
    K = np.asarray(cam["K"], float)
    Ks = K * [[scale], [scale], [1]]
    tab = DL.load_table(Path(log) if Path(log).is_absolute() else project / log)
    qs = DL.joint_series(tab, joints)
    m, d = kinematic_model(b, W, H)
    ren = CalibRenderer(m, w, h, groups=(1,))
    links = [m.body(i).name for i in range(1, m.nbody)]
    F = []
    for img, msk, row in _frames(project, frames):
        q = dict(b.init_q())
        q.update(q_fixed or {})
        q.update({j: float(v[row]) for j, v in qs.items()})
        set_q(m, d, q)
        T_cam = link_pose(m, d, cam["link"]) @ np.asarray(cam["T_link_cam"], float)
        LT = {ln: link_pose(m, d, ln) for ln in links}
        depth = ren.depth(d, Ks, T_cam)
        depth[depth <= 0] = np.inf
        mk = cv2.resize(read_mask(msk).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        dt = cv2.distanceTransform((~mk).astype(np.uint8), cv2.DIST_L2, 3)
        F.append({"img": img, "row": row, "q": q, "T_cam": T_cam, "LT": LT, "depth": depth, "mask": mk, "dt": dt})
    if not F:
        raise ValueError("no frames with masks")
    # pixel anchors -> link-frame points (ray cast on the robot)
    def lift(pt):
        if "p" in pt:
            return pt
        fr = next(f for f in F if f["row"] == int(pt["row"]))
        set_q(m, d, fr["q"])
        u, v = pt["pixel"]
        Rw = fr["T_cam"][:3, :3]
        o = fr["T_cam"][:3, 3]
        gid = np.array([-1], np.int32)
        geomgroup = np.array([0, 1, 0, 0, 0, 0], np.uint8)
        want = pt.get("link")
        hit = None
        for r_px in range(0, 81, 4):                        # nearest robot surface around the pixel
            for du, dv in ([(0, 0)] if r_px == 0 else [(r_px * np.cos(a_), r_px * np.sin(a_)) for a_ in np.linspace(0, 2 * np.pi, 12, endpoint=False)]):
                ray = np.linalg.inv(K) @ np.array([u + du, v + dv, 1.0])
                dirw = Rw @ (ray / np.linalg.norm(ray))
                dist = mujoco.mj_ray(m, d, o, dirw, geomgroup, 1, -1, gid)
                if dist >= 0:
                    hit = o + dist * dirw
                    if pt.get("surface") == "far":              # exit point of the ray (the part's back side)
                        o2 = hit + 0.6 * dirw
                        d2 = mujoco.mj_ray(m, d, o2, -dirw, geomgroup, 1, -1, gid)
                        if d2 >= 0:
                            hit = o2 - d2 * dirw
                    break
            if hit is not None:
                break
        if hit is None:
            raise ValueError(f"pixel anchor {pt} does not hit the robot within 80 px")
        link = want or m.body(int(m.geom_bodyid[gid[0]])).name      # the frame to express the point in
        T = fr["LT"][link]
        return {"link": link, "p": (np.linalg.inv(T) @ np.r_[hit, 1])[:3].round(5).tolist(), "from_pixel": pt}
    specs = {}
    for name, c in cables.items():
        c = copy.deepcopy(c)
        if c.get("shape", "spline") != "loop":
            c["points"] = [lift(p) for p in c["points"]]
        elif isinstance(c.get("center"), dict):                 # loop centre picked as a pixel
            c["center"] = lift({**c["center"], "link": c["link"]})["p"]
        specs[name] = c
    # ---------------------------------------------------------------- parameterisation
    def unpack(name, x, base):
        s = copy.deepcopy(base)
        fit = base.get("fit") or {}
        k = 0
        if s.get("shape", "spline") == "loop":
            s["center"] = (np.asarray(base["center"]) + x[k:k + 3]).tolist(); k += 3
            s["radius"] = float(base["radius"] * np.exp(x[k])); k += 1
            s["pitch"] = float(base.get("pitch", 0.0) + x[k]); k += 1
            s["phase"] = float(base.get("phase", 0.0) + x[k]); k += 1
        else:
            idx = range(len(s["points"])) if fit.get("points", "all") == "all" else fit.get("points", [])
            for i in idx:
                s["points"][i]["p"] = (np.asarray(base["points"][i]["p"]) + x[k:k + 3]).tolist(); k += 3
            if fit.get("length"):
                s["length_m"] = float(x[k]); k += 1
            if fit.get("slack_dir"):                    # azimuth / elevation of the slack direction in its link
                az, el = x[k], x[k + 1]; k += 2
                s["slack"] = {"link": base.get("slack", {}).get("link") or fit["slack_dir"],
                              "dir": [float(np.cos(el) * np.cos(az)), float(np.cos(el) * np.sin(az)), float(np.sin(el))]}
        return s, k

    def x0_bounds(base):
        fit = base.get("fit") or {}
        box = float(fit.get("box_m", 0.03))
        x0, lo, hi = [], [], []
        rm = []
        if base.get("shape", "spline") == "loop":
            x0 += [0, 0, 0, 0, 0, 0]
            lo += [-box] * 3 + [np.log(0.5), -0.02, -np.pi]
            hi += [box] * 3 + [np.log(2.0), 0.02, np.pi]
            rm += [1, 1, 1, 0, 0, 0]
        else:
            n = len(base["points"]) if fit.get("points", "all") == "all" else len(fit.get("points", []))
            x0 += [0.0] * 3 * n
            lo += [-box] * 3 * n
            hi += [box] * 3 * n
            rm += [1] * 3 * n
            if fit.get("length"):
                l0, l1 = fit["length"]
                x0.append(float(np.clip(base.get("length_m") or 0.5 * (l0 + l1), l0, l1)))
                lo.append(l0)
                hi.append(l1)
                rm.append(0)
            if fit.get("slack_dir"):
                x0 += [0.0, -0.5]
                lo += [-np.pi, -np.pi / 2]
                hi += [np.pi, np.pi / 2]
                rm += [0, 0]
        return np.array(x0, float), np.array(lo, float), np.array(hi, float), np.array(rm, float)

    def project_curve(s, fr):
        P = CB.centerline(s, fr["LT"], n=64)
        uv, z = proj(Ks, inv(fr["T_cam"]), P)
        ok = (z > 0.02) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        vis = ok.copy()
        iu, iv = np.clip(uv[:, 0].astype(int), 0, w - 1), np.clip(uv[:, 1].astype(int), 0, h - 1)
        vis &= z <= fr["depth"][iv, iu] + 0.01                  # not hidden behind the robot
        return uv, vis, ok

    # mask pixels assigned to each cable: near its initial projection (only mask pixels not covered by the robot)
    assign = {}
    for name, s in specs.items():
        assign[name] = []
        for fr in F:
            uv, vis, ok = project_curve(s, fr)
            canvas = np.zeros((h, w), np.uint8)
            pts = uv[ok].astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(canvas, [pts], False, 255, 1)
            near = cv2.distanceTransform((canvas == 0).astype(np.uint8), cv2.DIST_L2, 3) < assign_px * scale
            ys, xs = np.nonzero(fr["mask"] & near)
            if len(xs) > 300:
                sel = np.random.default_rng(0).choice(len(xs), 300, replace=False)
                xs, ys = xs[sel], ys[sel]
            assign[name].append(np.stack([xs, ys], 1))

    def seg_dist(A, uv):
        """Distance of points A (k,2) to the polyline uv (n,2)."""
        P0, P1 = uv[:-1], uv[1:]
        D = P1 - P0
        L2 = np.maximum((D ** 2).sum(1), 1e-9)
        t = np.clip(((A[:, None, :] - P0[None]) * D[None]).sum(2) / L2[None], 0, 1)
        C = P0[None] + t[..., None] * D[None]
        return np.sqrt(((A[:, None, :] - C) ** 2).sum(2)).min(1)

    def residuals(name, x, base):
        from scipy.ndimage import map_coordinates
        s, _ = unpack(name, x, base)
        out = []
        for fi, fr in enumerate(F):                         # fixed-length, smooth residual vector
            uv, vis, ok = project_curve(s, fr)
            u = np.clip(uv[:, 0], 0, w - 1)
            v = np.clip(uv[:, 1], 0, h - 1)
            dtv = map_coordinates(fr["dt"], [v, u], order=1, mode="nearest")
            out.append(np.where(vis, np.minimum(dtv, 40.0), 0.0) / 4.0)
            A = assign[name][fi]
            if len(A):
                if ok.sum() >= 2:
                    out.append(np.minimum(seg_dist(A.astype(float), uv[ok]), 40.0) / 8.0)
                else:
                    out.append(np.full(len(A), 40.0 / 8.0))
        r = np.concatenate(out) if out else np.zeros(1)
        # weak pull towards the initial anchors (metres -> comparable to a few px)
        reg = (x - x0_bounds(base)[0]) * x0_bounds(base)[3] * 30.0
        return np.r_[r, reg]

    fitted, report = {}, {}
    for name, base in specs.items():
        x0, lo, hi, _ = x0_bounds(base)
        before = residuals(name, x0, base)
        if len(x0):
            # coarse random search inside the box (the initial projection can be tens of px off), then least squares
            rng = np.random.default_rng(0)
            cost = lambda x: float(np.sum(np.sqrt(1 + residuals(name, x, base) ** 2) - 1))   # noqa: E731 (soft_l1)
            best, bc = x0.copy(), cost(x0)
            for _ in range(int(starts)):
                xc = rng.uniform(lo, hi) if rng.uniform() < 0.3 else np.clip(best + rng.normal(0, 0.25, len(x0)) * (hi - lo), lo, hi)
                c = cost(xc)
                if c < bc:
                    best, bc = xc, c
            sol = least_squares(lambda x: residuals(name, x, base), best, bounds=(lo, hi), loss="soft_l1", f_scale=1.0,
                                max_nfev=int(iters), diff_step=2e-3)
            x = sol.x
        else:
            x = x0
        s, _ = unpack(name, x, base)
        s.pop("fit", None)
        after = residuals(name, x, base)
        # per-frame visible-sample distance (px at full resolution)
        per = []
        for fr in F:
            uv, vis, ok = project_curve(s, fr)
            if vis.sum():
                iu, iv = uv[vis, 0].astype(int), uv[vis, 1].astype(int)
                per.append(float(np.median(fr["dt"][iv, iu]) / scale))
        cov = []
        for fi, fr in enumerate(F):
            A = assign[name][fi]
            uv, vis, ok = project_curve(s, fr)
            if len(A) and ok.sum() >= 2:
                canvas = np.zeros((h, w), np.uint8)
                cv2.polylines(canvas, [uv[ok].astype(np.int32)], False, 255, 1)
                dtc = cv2.distanceTransform((canvas == 0).astype(np.uint8), cv2.DIST_L2, 3)
                cov.append(float(np.mean(dtc[A[:, 1], A[:, 0]] < 6 * scale)))
        s["provenance"] = (f"fitted: robot_cables on {len(F)} {camera} frames, curve->mask median "
                           f"{np.median(per) if per else float('nan'):.1f} px, mask coverage {np.mean(cov) if cov else 0:.2f}")
        fitted[name] = s
        report[name] = {"cost_before": float(np.mean(before)), "cost_after": float(np.mean(after)),
                        "curve_to_mask_px_median": float(np.median(per)) if per else None,
                        "mask_coverage": float(np.mean(cov)) if cov else None,
                        "length_m": CB.arc_length(CB.centerline(s, F[0]["LT"], n=64))}
    # overlay
    tiles, labels = [], []
    busy = np.argsort([-sum(len(assign[nm][fi]) for nm in assign) for fi in range(len(F))])[:8]
    for fr in [F[i] for i in sorted(busy)]:
        im = read_rgb(fr["img"]).copy()
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        mk = cv2.resize(fr["mask"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        cnt, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnt, -1, (255, 0, 255), 1)
        for name in fitted:
            for s, col, dim in ((specs[name], (255, 60, 60), (110, 30, 30)), (fitted[name], (60, 255, 60), (20, 90, 20))):
                uvs, vis, ok = project_curve(s, fr)            # bright: visible, dark: behind the robot
                uvs = (uvs / scale).astype(np.int32)
                for k in range(len(uvs) - 1):
                    if ok[k] and ok[k + 1]:
                        cv2.line(im, tuple(uvs[k]), tuple(uvs[k + 1]), col if (vis[k] and vis[k + 1]) else dim, 2)
        tiles.append(im)
        labels.append(f"row {fr['row']}")
    (broot / "checks").mkdir(parents=True, exist_ok=True)
    write_rgb(broot / "checks" / "cables_overlay.jpg", contact_sheet(tiles, cols=2, width=720, labels=labels))
    res = {"camera": camera, "frames": [f["row"] for f in F], "cables": report, "fitted": fitted}
    write_json(res, broot / "checks" / "cables.json")
    if write:
        b.profile.setdefault("cables", {}).update(fitted)
        b.save_profile()
    return {"cables": report, "outputs": [str((broot / "checks" / "cables.json").relative_to(project)),
                                          str((broot / "checks" / "cables_overlay.jpg").relative_to(project))]}
