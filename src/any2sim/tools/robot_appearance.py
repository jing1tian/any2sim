"""Robot appearance from real images (robot stage): per-part materials (default) or per-texel albedo textures of the
links, from frames with logged joints, so renders show the real robot's colours (white shells, silver plates, dark
rings) instead of one CAD colour.

Observation model (both modes): surface samples of a link are projected through the bundle camera at the frame's FK;
a sample counts where it faces the camera, passes the depth test against the whole robot (MuJoCo), lies inside the
eroded robot mask and outside ignored masks (cables, people). Colours are sRGB-linearised. De-lighting (`delight: sh`)
fits a 2nd-order spherical-harmonics irradiance fixed in the robot base frame (the lab's lights do not move with the
robot) jointly with the albedos, so the real lab's shading is not baked in.

mode "parts" (default; robust to a few px of camera/kinematic misregistration): each link mesh is split into its
connected components (CAD bodies: shells, plates, rings, screws); a component's albedo is the median of its samples;
observed components are clustered into material classes (k-means in linear RGB over all links, <= `classes`);
unobserved components take the class of the nearest observed component of the same link (else the link's majority).
Classes get principled parameters by brightness (white plastic / grey-silver metal / dark), overridable afterwards
in robot.yaml `materials`. Output: one OBJ per link with a material group per class.
mode "texels": UV-unwrapped albedo textures (needs accurate registration; logos and stickers survive).

params:
  bundle: "robot"
  links: [globs]                    e.g. ["right_arm_*", "right_hand_bar"]
  camera: <bundle camera>; frames: {images, visible: {instances: dir} | {masks: dir}, ignore: [{masks: dir} ...]}
  log, joints, q_fixed
  mode: parts | texels; classes: 4; min_samples: 30 (parts) / 2 (texels)
  size: 512 (texels), min_cos: 0.25, depth_tol_m: 0.008, erode_px: 3, delight: "sh" | null
  copy_to: {<prefix>: <prefix>}     e.g. {"left_": "right_"}: twin links without observations reuse the result
  write: false                      bundle visual/<link>/<link>_app.obj (+png) and robot.yaml visuals[link].obj,
                                    robot.yaml materials (class parameters)
Writes <bundle>/checks/appearance.json and appearance_sheet.jpg.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np

from ..common import contact_sheet, read_mask, read_rgb, write_json, write_rgb

ENV = "base"
GPU = False


def _lin(x):
    x = np.asarray(x, np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def sh9(n):
    """Real SH basis up to order 2 (9 terms) for unit normals (N, 3)."""
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    return np.stack([np.ones_like(x), y, z, x, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y], 1)


def link_mesh(bundle, link):
    import trimesh
    L = bundle.model["links"][link]
    parts = []
    for g in L["visuals"]:
        if g["type"] != "mesh":
            continue
        m = trimesh.load(g["uri"], force="mesh", process=False)
        if g.get("scale") is not None:
            m.apply_scale(g["scale"])
        m.apply_transform(np.asarray(g["T"], float))
        parts.append(m)
    return trimesh.util.concatenate(parts) if parts else None


def texel_geometry(m, size):
    """UV-unwrapped mesh, texel face ids, link-frame texel positions and normals."""
    import cv2
    from ..meshkit import uv_unwrap
    mu = uv_unwrap(m)
    uv = np.asarray(mu.visual.uv)
    F = np.asarray(mu.faces)
    V = np.asarray(mu.vertices)
    px = uv * [size - 1, size - 1]
    px[:, 1] = (size - 1) - px[:, 1]                      # image rows go down
    fid = np.full((size, size), -1, np.int64)
    ids = np.zeros((size, size, 3), np.uint8)
    for f, tri in enumerate(F):
        c = ((f + 1) & 255, ((f + 1) >> 8) & 255, ((f + 1) >> 16) & 255)
        cv2.fillConvexPoly(ids, np.round(px[tri]).astype(np.int32), c)
    fid = ids[..., 0].astype(np.int64) + (ids[..., 1].astype(np.int64) << 8) + (ids[..., 2].astype(np.int64) << 16) - 1
    ys, xs = np.nonzero(fid >= 0)
    f = fid[ys, xs]
    a, b, c = px[F[f, 0]], px[F[f, 1]], px[F[f, 2]]
    q = np.stack([xs, ys], 1).astype(float)
    v0, v1, v2 = b - a, c - a, q - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-12, 1e-12, d00 * d11 - d01 * d01)
    wv, ww = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
    wu = 1 - wv - ww
    W = np.clip(np.stack([wu, wv, ww], 1), 0, 1)
    W /= W.sum(1, keepdims=True)
    P = (V[F[f]] * W[..., None]).sum(1)
    fn = np.asarray(mu.face_normals)[f]
    return mu, (ys, xs), P, fn




def _group_median(v, group, n_groups):
    """Per-group median of rows of v (groups as contiguous segments after one sort)."""
    order = np.argsort(group, kind="stable")
    g, vs = group[order], v[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    out = np.full((n_groups, v.shape[1]), np.nan)
    for s_, e_ in zip(starts, ends):
        out[g[s_]] = np.median(vs[s_:e_], 0)
    return out


def delight_sh(obs, normals, group, n_groups, iters: int = 3):
    """obs (N,3) linear colours, normals (N,3) base-frame, group (N,) albedo index -> (albedo (G,3), SH (9,3), err)."""
    Y = sh9(normals)
    alb = _group_median(obs, group, n_groups)
    E = np.ones_like(obs)
    for _ in range(iters):
        a = alb[group]
        L = np.stack([np.linalg.lstsq(Y * a[:, c:c + 1], obs[:, c], rcond=None)[0] for c in range(3)], 1)
        E = np.clip(Y @ L, 0.05, None)
        Em = np.median(E, 0)
        L /= Em
        E /= Em
        alb = _group_median(obs / E, group, n_groups)
    err = float(np.median(np.abs(alb[group] * E - obs) / (obs + 1e-3)))
    return alb, L, err


def class_params(c):
    """Principled parameters of a material class from its linear albedo (heuristic, overridable)."""
    v = float(np.max(c))
    sat = float((np.max(c) - np.min(c)) / (np.max(c) + 1e-6))
    if v > 0.45:
        return {"kind": "white plastic", "roughness": 0.38, "metallic": 0.0, "specular": 0.5}
    if v > 0.12 and sat < 0.25:                           # albedo was measured with a diffuse model: keep it mostly
        return {"kind": "grey metal", "roughness": 0.35, "metallic": 0.25, "specular": 0.6}   # diffuse
    if v > 0.12:
        return {"kind": "coloured plastic", "roughness": 0.4, "metallic": 0.0, "specular": 0.5}
    return {"kind": "dark", "roughness": 0.5, "metallic": 0.1, "specular": 0.4}


def run(project: Path, links: list, camera: str, frames: dict, log: str, joints: list, bundle: str = "robot",
        q_fixed: dict | None = None, mode: str = "parts", classes: int = 4, min_samples: int | None = None,
        size: int = 512, min_cos: float = 0.25, depth_tol_m: float = 0.008, erode_px: int = 3,
        delight: str | None = "sh", copy_to: dict | None = None, write: bool = False) -> dict:
    import cv2
    import trimesh
    from ..geom3d import inv, project as proj
    from ..robot import datalog as DL
    from ..robot.bundle import Bundle
    from ..robot.render import CalibRenderer
    from ..robot.sim import kinematic_model, link_pose, set_q
    broot = Path(bundle) if Path(bundle).is_absolute() else project / bundle
    b = Bundle(broot)
    cam = b.profile["cameras"][camera]
    W, H = cam["width"], cam["height"]
    K = np.asarray(cam["K"], float)
    tab = DL.load_table(Path(log) if Path(log).is_absolute() else project / log)
    qs = DL.joint_series(tab, joints)
    m, d = kinematic_model(b, W, H)
    ren = CalibRenderer(m, W, H, groups=(1,))
    sel = [ln for ln in b.model["links"] if any(fnmatch.fnmatch(ln, g) for g in links)]
    min_samples = min_samples or (30 if mode == "parts" else 2)
    imgs = sorted((project / frames["images"]).glob("frame*.jpg")) + sorted((project / frames["images"]).glob("frame*.png"))
    FR = []
    for p in imgs:
        row = int(p.stem[5:])
        vis = frames["visible"]
        if "instances" in vis:
            ms = sorted((project / vis["instances"]).glob(f"{p.stem}_*.png"))
            if not ms:
                continue
            mk = np.any([read_mask(x) for x in ms], axis=0)
        else:
            f = project / vis["masks"] / f"{p.stem}.png"
            if not f.exists():
                continue
            mk = read_mask(f)
        for ig in frames.get("ignore") or []:
            f = project / ig["masks"] / f"{p.stem}.png"
            if f.exists():
                mk &= ~cv2.dilate(read_mask(f).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        if erode_px:
            mk = cv2.erode(mk.astype(np.uint8), np.ones((2 * erode_px + 1,) * 2, np.uint8)).astype(bool)
        q = dict(b.init_q())
        q.update(q_fixed or {})
        q.update({j: float(v[row]) for j, v in qs.items()})
        set_q(m, d, q)
        T_cam = link_pose(m, d, cam["link"]) @ np.asarray(cam["T_link_cam"], float)
        FR.append({"row": row, "img": _lin(read_rgb(p)), "mask": mk, "T_cam": T_cam, "depth": ren.depth(d, K, T_cam),
                   "LT": {ln: link_pose(m, d, ln) for ln in sel}})
    if not FR:
        raise ValueError("no frames with robot masks")

    def observe(ln, P, N):
        """(frames, samples, 3) linear colours (NaN where not observed) and base-frame normals."""
        S = np.full((len(FR), len(P), 3), np.nan, np.float32)
        Nb = np.zeros((len(FR), len(P), 3), np.float32)
        for fi, fr in enumerate(FR):
            T = fr["LT"][ln]
            Pw = P @ T[:3, :3].T + T[:3, 3]
            Nw = N @ T[:3, :3].T
            Nb[fi] = Nw
            uv, z = proj(K, inv(fr["T_cam"]), Pw)
            inb = (z > 0.02) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
            vdir = fr["T_cam"][:3, 3] - Pw
            vdir /= np.linalg.norm(vdir, axis=1, keepdims=True) + 1e-12
            iu = np.clip(uv[:, 0].round().astype(int), 0, W - 1)
            iv = np.clip(uv[:, 1].round().astype(int), 0, H - 1)
            ok = (inb & ((Nw * vdir).sum(1) > min_cos) & fr["mask"][iv, iu]
                  & (np.abs(fr["depth"][iv, iu] - z) < depth_tol_m))
            S[fi, ok] = fr["img"][iv[ok], iu[ok]]
        return S, Nb

    report, results, sheet, labels = {}, {}, [], []
    if mode == "parts":
        comp = {}
        obs_all, nrm_all, grp_all, gid = [], [], [], 0
        for ln in sel:
            mesh = link_mesh(b, ln)
            if mesh is None:
                continue
            mesh.merge_vertices()                            # STL: unmerged triangles would each be a component
            cc = mesh.split(only_watertight=False)
            rng = np.random.default_rng(0)
            info = []
            for ci, c in enumerate(cc):
                n = int(np.clip(c.area / 2e-5, 20, 1500))      # ~ one sample per 0.2 cm^2
                P, fi_ = trimesh.sample.sample_surface(c, n, seed=int(rng.integers(1 << 30)))
                N = np.asarray(c.face_normals)[fi_]
                S, Nb = observe(ln, P, N)
                ok = np.isfinite(S[..., 0])
                info.append({"mesh": c, "gid": gid, "n_obs": int(ok.sum()), "centroid": np.asarray(c.centroid)})
                if ok.sum():
                    obs_all.append(S[ok])
                    nrm_all.append(Nb[ok])
                    grp_all.append(np.full(int(ok.sum()), gid))
                gid += 1
            comp[ln] = info
        obs = np.vstack(obs_all) if obs_all else np.zeros((0, 3))
        nrm = np.vstack(nrm_all) if nrm_all else np.zeros((0, 3))
        grp = np.concatenate(grp_all) if grp_all else np.zeros(0, int)
        counts = np.bincount(grp, minlength=gid)
        good = counts >= min_samples
        keep = good[grp]
        if delight == "sh" and keep.sum() > 500:
            alb, L, err = delight_sh(obs[keep], nrm[keep], grp[keep], gid)
            report["_delight"] = {"sh_coeffs": L.round(4).tolist(), "fit_rel_err": err}
        else:
            alb = _group_median(obs[keep], grp[keep], gid)
        # material classes: k-means (weighted by surface samples) on observed component albedos
        from scipy.cluster.vq import kmeans2
        idx = np.nonzero(good)[0]
        X = np.log(np.clip(alb[idx], 1e-3, 1))               # log space: separates dark parts
        k = int(min(classes, max(1, len(idx))))
        rep = np.repeat(np.arange(len(idx)), np.clip(counts[idx] // 30, 1, 50))
        cent, lab = kmeans2(X[rep], k, seed=0, minit="++")
        dist = ((X[:, None] - cent[None]) ** 2).sum(-1)
        cls_of = {int(g): int(np.argmin(dist[i])) for i, g in enumerate(idx)}
        cls_col = [np.median(alb[[g for g in idx if cls_of[int(g)] == c]], 0) if any(cls_of[int(g)] == c for g in idx)
                   else np.exp(cent[c]) for c in range(k)]
        mats = {f"cls{c}": {"base_color": np.round(cls_col[c], 4).tolist(), **class_params(cls_col[c])} for c in range(k)}
        all_obs = [it for info in comp.values() for it in info if cls_of.get(it["gid"]) is not None]
        global_maj = int(np.bincount([cls_of[it["gid"]] for it in all_obs], weights=[it["mesh"].area for it in all_obs],
                                     minlength=k).argmax()) if all_obs else 0
        for ln, info in comp.items():
            obs_c = [it for it in info if cls_of.get(it["gid"]) is not None]
            maj = (np.bincount([cls_of[it["gid"]] for it in obs_c], weights=[it["mesh"].area for it in obs_c]).argmax()
                   if obs_c else None)
            parts = []
            for it in info:
                c = cls_of.get(it["gid"])
                if c is None and obs_c:                         # nearest observed component of this link
                    c = cls_of[min(obs_c, key=lambda o: np.linalg.norm(o["centroid"] - it["centroid"]))["gid"]]
                if c is None:
                    c = maj if maj is not None else global_maj      # unobserved link: the robot's main material
                parts.append((it["mesh"], int(c)))
            results[ln] = {"parts": parts, "observed": bool(obs_c)}
            area = sum(it["mesh"].area for it in info)
            report[ln] = {"components": len(info), "observed_components": len(obs_c),
                          "observed_area_frac": float(sum(it["mesh"].area for it in obs_c) / max(area, 1e-9)),
                          "class_area": {f"cls{c}": round(float(sum(mm.area for mm, cc in parts if cc == c) / max(area, 1e-9)), 3)
                                         for c in range(k)}}
        report["_classes"] = {kk: {**v, "base_color_srgb": (_srgb(np.asarray(v["base_color"])) * 255).round(0).tolist()}
                              for kk, v in mats.items()}
        sw = []
        for kk, v in mats.items():
            sw.append(np.full((80, 160, 3), (_srgb(np.asarray(v["base_color"])) * 255).astype(np.uint8)))
        sheet, labels = sw, [f"{kk} {v['kind']}" for kk, v in mats.items()]
    else:
        mats = None
        for ln in sel:
            mesh = link_mesh(b, ln)
            if mesh is None:
                continue
            mu, (ys, xs), P, N = texel_geometry(mesh, size)
            S, Nb = observe(ln, P, N)
            cnt = np.isfinite(S[..., 0]).sum(0)
            seen = cnt >= min_samples
            alb = np.nanmedian(S, axis=0) if seen.any() else np.full((len(P), 3), np.nan)
            if delight == "sh" and seen.sum() > 200:
                fsel, tsel = np.nonzero(np.isfinite(S[:, seen, 0]))
                tidx = np.nonzero(seen)[0][tsel]
                a2, L, err = delight_sh(S[fsel, tidx], Nb[fsel, tidx], tidx, len(P))
                alb[seen] = np.where(np.isfinite(a2[seen]), a2[seen], alb[seen])
            med = np.nanmedian(alb[seen], axis=0) if seen.any() else np.array([0.5, 0.5, 0.5])
            tex = np.zeros((size, size, 3), np.float32)
            tex[ys, xs] = np.where(seen[:, None], alb, med)
            img8 = (_srgb(np.clip(tex, 0, 1)) * 255).astype(np.uint8)
            used = np.zeros((size, size), np.uint8)
            used[ys, xs] = 255
            img8 = cv2.inpaint(img8, 255 - used, 3, cv2.INPAINT_TELEA)
            results[ln] = {"mesh": mu, "img": img8, "observed": bool(seen.any())}
            report[ln] = {"texels": int(len(P)), "observed_frac": float(seen.mean())}
            sheet.append(img8)
            labels.append(f"{ln} observed {seen.mean():.2f}")
    for dst_pre, src_pre in (copy_to or {}).items():           # twin links reuse their counterpart's result
        for ln in b.model["links"]:
            s = src_pre + ln[len(dst_pre):] if ln.startswith(dst_pre) else None
            if s in results and not (results.get(ln) or {}).get("observed") and link_mesh(b, ln) is not None:
                results[ln] = {**results[s], "copied_from": s}
                report[ln] = {"copied_from": s}
    (broot / "checks").mkdir(parents=True, exist_ok=True)
    if sheet:
        write_rgb(broot / "checks" / "appearance_sheet.jpg", contact_sheet(sheet, cols=4, width=320, labels=labels))
    if write:
        from ..meshkit import save_obj
        vis = b.profile.setdefault("visuals", {})
        if mats:
            b.profile["materials"] = mats
        for ln, r in results.items():
            od = broot / "visual" / ln
            od.mkdir(parents=True, exist_ok=True)
            if mode == "parts":
                lines, vt, nv = [f"mtllib {ln}_app.mtl"], [], 0
                groups = {}
                for mm, c in r["parts"]:
                    groups.setdefault(c, []).append(mm)
                for c, ms in groups.items():
                    mm = trimesh.util.concatenate(ms)
                    lines.append(f"usemtl cls{c}")
                    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in mm.vertices]
                    lines += [f"f {a + 1 + nv} {bb + 1 + nv} {cc + 1 + nv}" for a, bb, cc in mm.faces]
                    nv += len(mm.vertices)
                (od / f"{ln}_app.obj").write_text("\n".join(lines) + "\n")
                (od / f"{ln}_app.mtl").write_text("".join(
                    f"newmtl cls{c}\nKd {' '.join(f'{x:.4f}' for x in _srgb(np.asarray(mats[f'cls{c}']['base_color'])))}\n"
                    for c in groups))
                entry = vis.setdefault(ln, {})
                entry["obj"] = f"visual/{ln}/{ln}_app.obj"
                entry.pop("material", None)
            else:
                write_rgb(od / f"{ln}_albedo.png", r["img"])
                save_obj(r["mesh"], od / f"{ln}_tex.obj", texture=f"{ln}_albedo.png", normals="crease", crease_deg=40)
                entry = vis.setdefault(ln, {})
                entry["obj"] = f"visual/{ln}/{ln}_tex.obj"
                entry["material"] = {"roughness": 0.4, "specular": 0.5}
            entry["appearance_source"] = (f"robot_appearance ({mode}): {camera} frames with logged joints"
                                          + (", SH de-lit" if delight == "sh" else "")
                                          + (f", copied from {r['copied_from']}" if r.get("copied_from") else ""))
        b.save_profile()
    res = {"camera": camera, "frames": len(FR), "mode": mode, "links": report}
    write_json(res, broot / "checks" / "appearance.json")
    return {**res, "outputs": ["checks/appearance.json", "checks/appearance_sheet.jpg"]}
