"""Register each object's mesh to the video: one shared scale per object, one rigid pose per rest state.

Loss per state (summed over its fit frames, each with its own camera): 1 - soft IoU between the rendered visible
silhouette (all objects rendered together, so other objects occlude) and the tracked mask, ignoring dynamic
(hand/arm) pixels; + MoGe-2 depth agreement on frames that have a MoGe point map; + support: an object resting on
the table has its lowest point on z = 0, and nothing goes below the table. Objects are registered largest first,
so earlier objects act as fixed occluders. States away from the SAM 3D anchor start from several orientations.
Held-out frames are only evaluated.
Outputs: scene/assets/objects/<name>/visual.glb (object frame: origin at the bottom centre, scale baked in),
work/objects/<name>/registration.json, evidence/registration_<name>_<state>.jpg
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import contact_sheet, frame_files, read_json, read_mask, read_rgb, write_json, write_rgb
from ..geom3d import inv

ENV = "sam3d"
GPU = True


def _so3(w):
    import torch
    th = torch.linalg.norm(w) + 1e-12
    k = w / th
    Kx = torch.zeros(3, 3, device=w.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def _rot(axis, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler(axis, deg, degrees=True).as_matrix()


def _boundary_stats(pred, gt, ignore):
    """Mean / P95 distance (px) from predicted outline pixels to the target outline, outlines near ignored
    pixels excluded."""
    import cv2
    def outline(m):
        m = m.astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0
    near_ign = cv2.dilate(ignore.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    op, og = outline(pred) & ~near_ign, outline(gt) & ~near_ign
    if not op.any() or not og.any():
        return None, None
    dt_g = cv2.distanceTransform((~og).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((~op).astype(np.uint8), cv2.DIST_L2, 3)
    d = np.r_[dt_g[op], dt_p[og]]
    return float(d.mean()), float(np.percentile(d, 95))


def run(project: Path, objects: list[str] | None = None, supports: dict | None = None, iters: int = 300,
        scale_res: float = 0.5, lr: float = 0.01, w_depth: float = 0.0, w_support: float = 0.1,
        dyn_dilate_px: int = 7, n_starts: int = 12, level_max_deg: float = 25.0) -> dict:
    import cv2
    import torch
    import trimesh
    from ..models.raster import Rasterizer
    dev = "cuda"
    labels = read_json(project / "work" / "masks" / "labels.json")["labels"]
    name_to_label = {v: int(k) for k, v in labels.items()}
    states = read_json(project / "work" / "objects" / "states.json")
    cams = read_json(project / "work" / "world" / "cameras.json")
    Ts = np.asarray(cams["T_world_cam"])
    W0, H0 = cams["width"], cams["height"]
    W, H = int(W0 * scale_res), int(H0 * scale_res)
    K0 = np.asarray(cams["K"])
    K = K0.copy()
    K[:2] *= scale_res
    files = frame_files(project / "work" / "frames")
    vis = read_json(project / "work" / "masks" / "visibility.json")
    names = objects or sorted(labels.values(), key=lambda n: -np.median([v["area_px"] for v in vis[n]]))
    supports = supports or {}
    R = Rasterizer()
    kd = np.ones((dyn_dilate_px, dyn_dilate_px), np.uint8)

    def target(i, name):
        m = read_mask(project / "work" / "masks" / "objects" / f"frame{i:05d}.png") == name_to_label[name]
        dyn = cv2.dilate((read_mask(project / "work" / "masks" / "dynamic" / f"frame{i:05d}.png") > 0).astype(np.uint8), kd) > 0
        return m, dyn

    def small(x):
        return cv2.resize(x.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    registered = {}          # name -> {"V": torch local verts (scaled), "F": torch, "poses": {state: 4x4}, "frames": {...}}

    def occluders(i, skip):
        out = []
        for nm, r in registered.items():
            if nm == skip:
                continue
            for st in r["states"]:
                a, b = st["frames"]
                if a <= i <= b:
                    T = torch.tensor(st["T_world_obj"], dtype=torch.float32, device=dev)
                    out.append((r["V"] @ T[:3, :3].T + T[:3, 3], r["F"]))
        return out

    def cam_verts(Vw, i):
        Tcw = torch.tensor(inv(Ts[i]), dtype=torch.float32, device=dev)
        return Vw @ Tcw[:3, :3].T + Tcw[:3, 3]

    results = {}
    for name in names:
        init = trimesh.load(project / "scene" / "assets" / "objects" / name / "init_world.glb", force="mesh")
        lay = read_json(project / "work" / "objects" / name / "layout.json")
        anchor = lay["anchor_frame"]
        Vw0 = np.asarray(init.vertices)
        # resting prior for the anchor state: SAM 3D's canonical up axis (local +z) is levelled to world +z when
        # the tilt is small, and the object is lifted/lowered onto the table; recorded in the result
        from scipy.spatial.transform import Rotation
        q = lay["layout"]["quat_wxyz_local_to_p3dcam"]
        R_l2c = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        up_cv = np.diag([-1.0, -1.0, 1.0]) @ R_l2c.T @ np.array([0, 0, 1.0])
        up_w = np.asarray(cams["T_world_cam"][anchor])[:3, :3] @ up_cv
        up_w /= np.linalg.norm(up_w)
        tilt = float(np.degrees(np.arccos(np.clip(up_w[2], -1, 1))))
        levelled = False
        ctr = Vw0.mean(0)
        if tilt < level_max_deg and supports.get(name, {}).get("rest0", "table") == "table":
            axis = np.cross(up_w, [0, 0, 1.0])
            if np.linalg.norm(axis) > 1e-9:
                Rl = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.radians(tilt)).as_matrix()
                Vw0 = (Vw0 - ctr) @ Rl.T + ctr
            Vw0 = Vw0 - [0, 0, Vw0[:, 2].min()]
            levelled = True
        c = np.array([(Vw0[:, 0].min() + Vw0[:, 0].max()) / 2, (Vw0[:, 1].min() + Vw0[:, 1].max()) / 2, Vw0[:, 2].min()])
        Vloc_vis = torch.tensor(Vw0 - c, dtype=torch.float32, device=dev)
        # silhouettes are fitted with the convex hull: tracked masks are solid while generated meshes can have
        # real openings (a slotted basket renders as a lattice, and the fit would inflate it to cover the mask)
        hull = trimesh.Trimesh(Vw0 - c, np.asarray(init.faces), process=False).convex_hull
        Vloc = torch.tensor(np.asarray(hull.vertices), dtype=torch.float32, device=dev)
        F = torch.tensor(np.asarray(hull.faces), dtype=torch.int64, device=dev)
        st_list = states.get(name, [])
        if not st_list:
            results[name] = {"error": "no rest state"}
            continue
        anchor_state = min(range(len(st_list)), key=lambda s: 0 if st_list[s]["frames"][0] <= anchor <= st_list[s]["frames"][1]
                           else min(abs(anchor - st_list[s]["frames"][0]), abs(anchor - st_list[s]["frames"][1])))
        log_s = torch.zeros(1, device=dev, requires_grad=True)
        data = {}
        for s, st in enumerate(st_list):
            fr = []
            for i in st["fit"]:
                m, dyn = target(i, name)
                mf = project / "work" / "geometry" / f"moge_{i:05d}.npz"
                dep = None
                if mf.exists() and w_depth > 0:
                    d = np.load(mf)["depth"].astype(np.float32)
                    d = cv2.resize(np.nan_to_num(d, nan=0, posinf=0), (W, H), interpolation=cv2.INTER_NEAREST)
                    dep = torch.tensor(d, device=dev)
                fr.append({"i": i, "m": torch.tensor(small(m), dtype=torch.float32, device=dev),
                           "valid": torch.tensor(~small(dyn), dtype=torch.float32, device=dev), "depth": dep})
            data[s] = fr

        def support_of(s):
            """'table', 'on:<object>' (anchor xy over another object's footprint at that time) or 'free'."""
            st = st_list[s]
            given = supports.get(name, {}).get(st["state"])
            if given:
                return given
            if s == anchor_state:
                return "table"
            ax, ay = st["anchor_xy"]
            for nm, r in registered.items():
                for o in r["states"]:
                    if o["frames"][0] <= st["frames"][0] <= o["frames"][1]:
                        T = torch.tensor(o["T_world_obj"], dtype=torch.float32, device=dev)
                        Vo = (r["V"] @ T[:3, :3].T + T[:3, 3]).cpu().numpy()
                        lo, hi = Vo[:, :2].min(0), Vo[:, :2].max(0)
                        if lo[0] < ax < hi[0] and lo[1] < ay < hi[1]:
                            return f"on:{nm}"
            return "table"

        def support_height(sup, s, xy, radius):
            """Height of the support surface under xy: 0 for the table; for 'on:X' the floor of X there
            (90th percentile height of X's lower-half vertices within `radius`)."""
            if not sup.startswith("on:"):
                return 0.0
            nm = sup[3:]
            r = registered[nm]
            t0 = st_list[s]["frames"][0]
            o = next(o for o in r["states"] if o["frames"][0] <= t0 <= o["frames"][1])
            T = torch.tensor(o["T_world_obj"], dtype=torch.float32, device=dev)
            Vo = (r["V"] @ T[:3, :3].T + T[:3, 3]).cpu().numpy()
            zmid = Vo[:, 2].min() + 0.5 * (Vo[:, 2].max() - Vo[:, 2].min())
            near = (np.linalg.norm(Vo[:, :2] - xy, axis=1) < radius) & (Vo[:, 2] < zmid)
            return float(np.percentile(Vo[near, 2], 90)) if near.sum() > 10 else float(Vo[:, 2].min())

        def place(Rm, txy, sc, z_sup):
            """Resting pose: lowest points of the rotated, scaled mesh at the support height."""
            V = (Vloc * sc) @ Rm.T
            zlow = -torch.topk(-V[:, 2], 32).values.mean()
            return V + torch.cat([txy, (z_sup - zlow).reshape(1)])

        def state_loss(s, Vw):
            loss = 0.0
            for f in data[s]:
                Vc = cam_verts(Vw, f["i"])
                occ = [(cam_verts(V, f["i"]), Fo) for V, Fo in occluders(f["i"], name)]
                sil, z = R.render_ids([Vc] + [o[0] for o in occ], [F] + [o[1] for o in occ], K, W, H)
                p, g, v = sil[0], f["m"], f["valid"]
                inter = (p * g * v).sum()
                union = ((p + g - p * g) * v).sum() + 1e-6
                loss = loss + (1 - inter / union)
                if f["depth"] is not None:
                    sel = (g * v > 0) & (z > 0) & (f["depth"] > 0) & (sil[0] > 0.99)
                    if sel.any():
                        loss = loss + w_depth * torch.nn.functional.smooth_l1_loss(z[sel], f["depth"][sel], beta=0.01) / 0.01
            loss = loss / len(data[s])
            zmin = -torch.topk(-Vw[:, 2], 32).values.mean()
            return loss + 10 * w_support * (torch.relu(-zmin) / 0.005) ** 2   # never below the table

        def optimise(s, R0, t0, n_it, fit_scale, sup):
            """sup 'free': 6-DoF. Otherwise planar resting: yaw + x, y (z from the support height)."""
            w = torch.zeros(3, device=dev, requires_grad=True)
            dt = torch.zeros(3, device=dev, requires_grad=True)
            opt = torch.optim.Adam([{"params": [w], "lr": lr}, {"params": [dt], "lr": lr * 0.05}] +
                                   ([{"params": [log_s], "lr": lr * 0.5}] if fit_scale else []))
            R0t = torch.tensor(R0, dtype=torch.float32, device=dev)
            t0t = torch.tensor(t0, dtype=torch.float32, device=dev)
            planar = sup != "free"
            ez = torch.tensor([0.0, 0.0, 1.0], device=dev)
            z_sup = support_height(sup, s, np.asarray(t0[:2]), 0.5 * float((Vloc[:, :2].max(0).values - Vloc[:, :2].min(0).values).max())) if planar else 0.0
            hist = []

            def pose(w, dt, sc):
                Rm = _so3(w * ez if planar else w) @ R0t
                if planar:
                    return Rm, place(Rm, (t0t + dt)[:2], sc, z_sup)
                return Rm, (Vloc * sc) @ Rm.T + t0t + dt
            for it in range(n_it):
                opt.zero_grad()
                sc = torch.exp(log_s) if fit_scale else torch.exp(log_s).detach()
                Rm, Vw = pose(w, dt, sc)
                L = state_loss(s, Vw)
                L.backward()
                opt.step()
                hist.append(float(L))
            with torch.no_grad():
                sc = torch.exp(log_s)
                Rm, Vw = pose(w, dt, sc)
                Rm = Rm.cpu().numpy()
                t = (Vw - (Vloc * sc) @ torch.tensor(Rm, dtype=torch.float32, device=dev).T)[0].cpu().numpy()
            return Rm, t, hist, z_sup

        rec_states = []
        order = [anchor_state] + [s for s in range(len(st_list)) if s != anchor_state]
        for s in order:
            st = st_list[s]
            sup = support_of(s)
            if s == anchor_state:
                Rm, t, hist, z_sup = optimise(s, np.eye(3), c, iters, True, sup)
                starts = 1
            else:
                # unknown orientation: several starts (yaw x upright / on its side / upside down) at the state's
                # anchor position; best after a short fit, then refined
                ax, ay = st["anchor_xy"]
                cands = [_rot("z", yaw) @ _rot("x", flip) for flip in (0, 90, 180)
                         for yaw in np.linspace(0, 360, max(n_starts // 3, 1), endpoint=False)]
                best = None
                for R0 in cands:
                    Rm_, t_, h_, _ = optimise(s, R0, np.array([ax, ay, 0.0]), max(iters // 4, 40), False, sup)
                    if best is None or h_[-1] < best[2][-1]:
                        best = (Rm_, t_, h_)
                Rm, t, hist, z_sup = optimise(s, best[0], best[1], iters, False, sup)
                starts = len(cands)
            T = np.eye(4)
            T[:3, :3] = Rm
            T[:3, 3] = t
            rec_states.append({**st, "T_world_obj": T.tolist(), "loss_first": hist[0], "loss_last": hist[-1],
                               "n_starts": starts, "support": sup, "support_height_m": z_sup,
                               "dof": "6-DoF" if sup == "free" else "planar resting (yaw, x, y)"})
        scale = float(torch.exp(log_s).detach())
        Vvis = (Vloc_vis * scale).detach()
        Fvis = torch.tensor(np.asarray(init.faces), dtype=torch.int64, device=dev)
        registered[name] = {"V": Vvis, "F": Fvis, "states": rec_states}   # visual mesh occludes (keeps openings)
        mesh = init.copy()                                   # object frame: origin at the bottom centre, scale baked
        mesh.vertices = Vvis.cpu().numpy()
        mesh.export(str(project / "scene" / "assets" / "objects" / name / "visual.glb"))
        ext = mesh.vertices.max(0) - mesh.vertices.min(0)
        results[name] = {"scale_vs_sam3d_moge": scale, "extent_m": ext.tolist(),
                         "init_tilt_deg": tilt, "levelled_and_placed_on_table": levelled, "anchor_frame": anchor,
                         "fit_proxy": "convex hull silhouettes; evaluation fills openings of the visual mesh"}

    # evaluation at full resolution on fit and held-out frames, after all objects are placed (mutual occlusion)
    for name, reg in registered.items():
        for st in reg["states"]:
            T = torch.tensor(st["T_world_obj"], dtype=torch.float32, device=dev)
            Vw = reg["V"] @ T[:3, :3].T + T[:3, 3]
            ev, tiles, caps = {}, [], []
            for kind in ("fit", "heldout"):
                for i in st[kind]:
                    m, dyn = target(i, name)
                    occ = [(cam_verts(V, i), Fo) for V, Fo in occluders(i, name)]
                    with torch.no_grad():
                        sil, _ = R.render_ids([cam_verts(Vw, i)] + [o[0] for o in occ], [reg["F"]] + [o[1] for o in occ], K0, W0, H0)
                    p = sil[0].cpu().numpy() > 0.5
                    # openings of the visual mesh (basket slots) are filled: tracked masks are solid
                    cn, _ = cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    p = cv2.drawContours(np.zeros(p.shape, np.uint8), cn, -1, 1, -1) > 0
                    if len(sil) > 1:                         # ...but not where another object is in front
                        p &= ~(sil[1:].amax(0).cpu().numpy() > 0.5)
                    v = ~dyn
                    iou = float((p & m & v).sum() / max(((p | m) & v).sum(), 1))
                    bm, b95 = _boundary_stats(p, m, dyn)
                    ev.setdefault(kind, []).append({"frame": i, "iou": round(iou, 4), "boundary_mean_px": bm,
                                                    "boundary_p95_px": b95})
                    if len(tiles) < 6:
                        im = read_rgb(files[i])
                        for msk, col in ((m, (255, 0, 255)), (p, (0, 255, 0))):
                            cn, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            cv2.drawContours(im, cn, -1, col, 2)
                        im[dyn] = (im[dyn] * 0.4).astype(np.uint8)
                        tiles.append(im)
                        caps.append(f"{kind} frame {i} IoU {iou:.3f}")
            st["eval"] = ev
            st["eval_summary"] = {k: {"iou_mean": float(np.mean([e["iou"] for e in v])),
                                      "iou_min": float(np.min([e["iou"] for e in v])),
                                      "boundary_p95_px_max": float(np.max([e["boundary_p95_px"] or 0 for e in v]))}
                                  for k, v in ev.items()}
            if tiles:
                write_rgb(project / "evidence" / f"registration_{name}_{st['state']}.jpg",
                          contact_sheet(tiles, cols=3, width=640, labels=caps))
        results[name]["states"] = [{k: v for k, v in st.items() if k != "eval"} for st in reg["states"]]
        write_json({**results[name], "states": reg["states"]}, project / "work" / "objects" / name / "registration.json")
    return {"objects": results, "outputs": ["scene/assets/objects", "work/objects"] +
            [f"evidence/registration_{n}_{st['state']}.jpg" for n in results if "states" in results[n] for st in results[n]["states"]]}
