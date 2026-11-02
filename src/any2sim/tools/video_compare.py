"""Compare the simulated/rendered episode with the source video, frame by frame (the trajectory stage's evidence).

Sim side: the achieved motion (render.motion) replayed in MuJoCo and segmented through the source camera of each
frame (robot hand links, robot arm, every rigid entity, with occlusion). Real side: masks supplied by the caller.

Per frame:
  hand      robot hand silhouette vs human hand mask: IoU, precision (robot hand inside the human hand region),
            recall; optional keypoints: projected robot hand points vs projected human keypoints (px)
  tracks    (with `tracks`) the registered metric hand tracks (rigid) projected vs glove masks: keypoint->glove and
            glove->hand distances (px), IoU, size ratio (handkit.reprojection)
  contact_timing (with `tracks`) first/last sim robot-object contact (mujoco contacts json) vs the tracks'
            grasp/release events (handkit.track_events)
  robot_hand_scale  (with `glove`) rendered robot hand vs the real glove next to it, per side: sqrt(area) ratio
            divided by the metric length ratio robot hand / human hand (about 1 when the scene scale is right;
            glove masks include cuffs, so expect slightly below 1)
  objects   visible sim object vs real object mask (IoU, centroid px), pixels of either hand ignored
  static    SSIM of the rendered frame vs the real frame outside hands, robot, objects (dilated)
Episode:
  events    per object: first frame it leaves its initial pose / last frame it moves (sim) vs the video's rest-state
            boundaries (Scene IR states); final pose error vs the last rest state (cm, deg)
Writes evidence/video_compare.json, video_compare_worst.jpg and video_compare.mp4 (real frame with sim outlines:
robot hand green, human hand magenta, sim objects cyan, real objects yellow | rendered frame).

params:
  renders: project-relative dir with src_%05d.png (blender_render frames "all") | null (no static SSIM)
  hand_masks: {masks: dir} | {label_maps: dir, labels: [..]} | null
  hand_groups: null | [group names] (default: every group of kind hand/gripper)
  object_masks: {entity: {masks: dir} | {label_maps: dir, label: k}}
  keypoints: null | {npz: path, key: "kp_world" (T,K,3) world frame, robot_points: [[link, [ox,oy,oz]], ...]}
  tracks: null | {registration: <scene registrations key> | {T_world_tracks, frame_offset}, uri:
          inputs/hand_tracks.npz, scene_factor: 1.0}
  glove: null | {instances: dir} | {masks: dir}        glove-only masks (no controllers/sleeves) for the two checks
  frames: "all" | [ids], step: 1
  move_thresh_m: 0.01
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import contact_sheet, frame_files, read_rgb, write_json, write_rgb, write_video

ENV = "base"
GPU = False


def _ssim(a, b, mask):
    import cv2
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    s_aa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    s_bb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    s_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))
    return float(m[mask].mean()) if mask.any() else None


def run(project: Path, object_masks: dict | None = None, hand_masks: dict | None = None, renders: str | None = None,
        hand_groups: list | None = None, keypoints: dict | None = None, frames="all", step: int = 1,
        move_thresh_m: float = 0.01, out: str = "evidence/video_compare.json", tracks: dict | None = None,
        glove: dict | None = None) -> dict:
    import cv2
    from .. import simkit as sk
    from ..geom3d import project as proj, inv
    from ..meshio import fill_holes, frame_mask
    from ..robot.render import CalibRenderer
    from ..scene_ir import camera_poses, load, load_motion, motion_at, robot_bundle
    scene = load(project / "scene")
    rm = (scene.get("render") or {}).get("motion")
    if not rm:
        raise ValueError("scene has no render.motion (run mujoco_rollout first)")
    M = load_motion(project / "scene", scene, rm)
    sim = sk.Sim(project, build=False)
    c = scene["cameras"]["source_cam"]
    W, H = c["width"], c["height"]
    K = np.asarray(c["K"], float)
    poses = camera_poses(project / "scene", scene)
    files = frame_files(project / "work" / "frames")
    nM = int(np.floor(M["frame"][-1])) + 1
    fr = list(range(0, min(nM, len(files)), step)) if frames == "all" else list(frames)
    ren = CalibRenderer(sim.m, W, H, groups=(0, 1, 2))
    # body groups
    hand_bodies, arm_bodies = set(), set()
    side_bodies, robot_len = {}, {}
    for rid in (scene.get("robots") or {}):
        b = robot_bundle(project / "scene", scene, rid)
        pre = sim.info["robots"][rid]["prefix"]
        hg = hand_groups or [g for g, G in b.profile["groups"].items() if G["kind"] in ("hand", "gripper")]
        hand_links = set()
        for g in hg:
            G = b.profile["groups"][g]
            j0 = b.model["joints"][G["joints"][0]]                 # the hand: its mounting link and everything below
            todo = [j0["parent"]]
            links_g = set()
            while todo:
                l = todo.pop()
                links_g.add(l)
                todo += [j["child"] for j in b.model["joints"].values() if j["parent"] == l]
            hand_links |= links_g
            sd_ = G.get("side") or g
            side_bodies.setdefault(sd_, set()).update(
                bi for bi, own in sim.body_owner.items() if own[0] == "robot" and own[1] == rid and own[2] in links_g)
            rt = (b.profile.get("retarget") or {}).get(g)
            if rt and "middle" in rt.get("chains", {}):
                wl = rt["rigid"]["wrist"]
                tl = rt["chains"]["middle"]["keypoints"]["middle_3"]
                wl = wl if isinstance(wl, str) else wl["link"]
                tl = tl if isinstance(tl, str) else tl["link"]
                Tq = b.fk(b.init_q())
                robot_len[sd_] = float(np.linalg.norm(Tq[tl][:3, 3] - Tq[wl][:3, 3]))
        for bi, own in sim.body_owner.items():
            if own[0] == "robot" and own[1] == rid:
                (hand_bodies if own[2] in hand_links else arm_bodies).add(bi)
    ent_body = {n: sim.m.body(n).id for n in sim.rigid}
    trk, reg = None, None
    if tracks:
        from .. import handkit as hk
        trk = hk.load_tracks(project, tracks.get("uri", "inputs/hand_tracks.npz"))
        reg = tracks["registration"]
        if isinstance(reg, str):
            reg = scene["registrations"][reg]
        human_len = {sd_: hk.hand_length(t) for sd_, t in trk.items()}
    scale_rows = []
    kp = None
    if keypoints:
        kd = np.load(project / keypoints["npz"])
        kp = kd[keypoints.get("key", "kp_world")]
    per, worst, vid = [], [], []
    for f in fr:
        q, objs = motion_at(M, f)
        sim.reset(q, objs)
        Tc = poses[min(f, len(poses) - 1)]
        seg = ren.segment(sim.d, K, Tc)
        robot_hand = np.isin(seg, list(hand_bodies))
        robot_any = robot_hand | np.isin(seg, list(arm_bodies))
        row = {"frame": int(f)}
        real = read_rgb(files[f])
        hm = frame_mask(project, hand_masks, f) if hand_masks else None
        if hm is not None:
            inter = (robot_hand & hm).sum()
            row["hand"] = {"iou": float(inter / max((robot_hand | hm).sum(), 1)),
                           "precision": float(inter / max(robot_hand.sum(), 1)) if robot_hand.any() else None,
                           "recall": float(inter / max(hm.sum(), 1)) if hm.any() else None,
                           "robot_px": int(robot_hand.sum()), "human_px": int(hm.sum())}
        if kp is not None and f < len(kp) and keypoints.get("robot_points"):
            pts = []
            for link, off in keypoints["robot_points"]:
                bid = sim.m.body(link).id
                pts.append(sim.d.xmat[bid].reshape(3, 3) @ np.asarray(off, float) + sim.d.xpos[bid])
            uv_r, _ = proj(K, inv(Tc), np.asarray(pts))
            uv_h, zh = proj(K, inv(Tc), kp[f])
            ok = zh > 0
            if ok.any():
                row["keypoint_px"] = float(np.median(np.linalg.norm(uv_r[ok] - uv_h[ok], axis=1)))
        ign = robot_any | (hm if hm is not None else False)
        objrows = {}
        for n, spec in (object_masks or {}).items():
            om = frame_mask(project, spec, f)
            if om is None:
                continue
            sm = fill_holes(seg == ent_body[n]) & ((seg == ent_body[n]) | (seg < 0) | np.isin(seg, list(sim.body_owner)))
            valid = ~ign                                # openings filled (tracked masks are solid), hands ignored
            u = ((sm | om) & valid).sum()
            r = {"iou": float(((sm & om) & valid).sum() / u) if u else None, "sim_px": int(sm.sum()), "real_px": int(om.sum())}
            if sm.any() and om.any():
                ys, xs = np.nonzero(sm)
                yr, xr = np.nonzero(om)
                r["centroid_px"] = float(np.hypot(xs.mean() - xr.mean(), ys.mean() - yr.mean()))
            objrows[n] = r
        row["objects"] = objrows
        rend = None
        if renders:
            p = project / renders / f"src_{f:05d}.png"
            if p.exists():
                rend = read_rgb(p)
                dyn = robot_any.copy()
                if hm is not None:
                    dyn |= hm
                for n in ent_body:
                    dyn |= seg == ent_body[n]
                for n, spec in (object_masks or {}).items():
                    om = frame_mask(project, spec, f)
                    if om is not None:
                        dyn |= om
                dyn = cv2.dilate(dyn.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
                row["static_ssim"] = _ssim(real, rend if rend.shape == real.shape else cv2.resize(rend, (W, H)), ~dyn)
        if glove and robot_len:
            from ..handkit import _component_near, glove_mask
            gm = glove_mask(project, glove, f, H, W)
            for sd_, bodies in side_bodies.items():
                rs = np.isin(seg, list(bodies))
                if gm is None or rs.sum() < 300:
                    continue
                gc = _component_near(gm, rs)
                if gc is None or gc.sum() < 300:
                    continue
                hl = human_len.get(sd_) if trk else None
                exp = robot_len[sd_] / hl if hl else None
                r_ = float(np.sqrt(rs.sum() / gc.sum()))
                scale_rows.append({"frame": int(f), "side": sd_, "area_ratio_sqrt": r_,
                                   "expected": exp, "scale": r_ / exp if exp else None})
                row.setdefault("robot_hand_scale", {})[sd_] = scale_rows[-1]["scale"]
        per.append(row)
        # overlay frame
        im = real.copy()
        for m_, col in ((robot_hand, (0, 255, 0)), (hm, (255, 0, 255))):
            if m_ is not None:
                cn, _ = cv2.findContours(m_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cn, -1, col, 2)
        for n, spec in (object_masks or {}).items():
            om = frame_mask(project, spec, f)
            for m_, col in ((seg == ent_body[n], (0, 255, 255)), (om, (255, 255, 0))):
                if m_ is not None:
                    cn, _ = cv2.findContours(m_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(im, cn, -1, col, 2)
        half = lambda x: cv2.resize(x, (W // 2, H // 2))
        vid.append(np.hstack([half(im), half(rend if rend is not None else np.zeros_like(real))]))
        score = min([v["iou"] for v in objrows.values() if v.get("iou") is not None] or [1.0])
        worst.append((score, f, vid[-1]))
    ren.close()
    # events vs rest states
    events = {}
    for i, n in enumerate(M.get("obj_names", [])):
        P = M["obj_T"][:, i, :3, 3]
        dist = np.linalg.norm(P - P[0], axis=1)
        moving = np.nonzero(dist > move_thresh_m)[0]
        e = scene["entities"][n]
        st = sorted((e.get("states") or {}).items(), key=lambda kv: kv[1]["frames"][0])
        ev = {"sim_leaves_initial_frame": float(M["frame"][moving[0]]) if len(moving) else None}
        spd = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)] * sim.fps
        mv = np.nonzero(spd > 0.02)[0]
        ev["sim_last_moving_frame"] = float(M["frame"][mv[-1]]) if len(mv) else None
        if len(st) >= 2:
            ev["video_leaves_rest_after_frame"] = st[0][1]["frames"][1]
            ev["video_next_rest_from_frame"] = st[-1][1]["frames"][0]
            if ev["sim_leaves_initial_frame"] is not None:
                ev["lift_off_delay_frames"] = ev["sim_leaves_initial_frame"] - st[0][1]["frames"][1]
            if ev["sim_last_moving_frame"] is not None:
                ev["settle_delay_frames"] = ev["sim_last_moving_frame"] - st[-1][1]["frames"][0]
        if st:
            Tf = np.asarray(st[-1][1]["T_world_obj"], float)
            Ts = M["obj_T"][-1, i]
            ev["final_pos_err_cm"] = float(np.linalg.norm(Ts[:3, 3] - Tf[:3, 3]) * 100)
            ev["final_rot_err_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(Ts[:3, :3].T @ Tf[:3, :3]) - 1) / 2, -1, 1))))
        events[n] = ev
    def agg(vals):
        vals = [v for v in vals if v is not None]
        return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "n": len(vals)} if vals else None
    tracks_check = None
    if trk is not None and glove:
        from ..handkit import reprojection
        tracks_check = reprojection(project, trk, np.asarray(reg["T_world_tracks"], float), fr, glove,
                                    float(tracks.get("scene_factor", 1.0)), frame_offset=int(reg.get("frame_offset", 0)))
    contact_timing = None
    cpath = project / "mujoco" / f"contacts_{rm}.json"
    if trk is not None and cpath.exists():
        from ..handkit import track_events
        import json as _json
        cons = _json.loads(cpath.read_text()).get("contacts") or []
        contact_timing = {}
        for side, t in trk.items():
            ev = track_events(t, fps=sim.fps)
            if "grasp" not in ev:
                continue
            off = int(reg.get("frame_offset", 0))
            for n in sim.rigid:
                if n in sim.static:
                    continue
                fr_c = [int(round(M["frame"][k])) for k, cs in enumerate(cons)
                        if any((c_["a"][0] == "robot" and c_["b"][0] == "entity" and c_["b"][1] == n) or
                               (c_["b"][0] == "robot" and c_["a"][0] == "entity" and c_["a"][1] == n) for c_ in cs)]
                if not fr_c:
                    continue
                contact_timing[f"{side}:{n}"] = {
                    "track_grasp_start": ev["grasp"][0] + off, "sim_first_contact": min(fr_c),
                    "track_release": ev["release"][0] + off, "sim_last_contact": max(fr_c),
                    "grasp_delay_frames": min(fr_c) - (ev["grasp"][0] + off),
                    "release_delay_frames": max(fr_c) - (ev["release"][1] + off)}
    sv = [r_["scale"] for r_ in scale_rows if r_.get("scale")]
    summary = {"frames": len(per),
               "robot_hand_scale": {"median": float(np.median(sv)), "p25": float(np.percentile(sv, 25)),
                                    "p75": float(np.percentile(sv, 75)), "n": len(sv)} if sv else None,
               "tracks_reprojection": tracks_check["summary"] if tracks_check else None,
               "contact_timing": contact_timing,
               "hand_iou": agg([r.get("hand", {}).get("iou") for r in per if r.get("hand", {}).get("human_px")]),
               "hand_precision": agg([r.get("hand", {}).get("precision") for r in per if r.get("hand", {}).get("human_px")]),
               "keypoint_px": agg([r.get("keypoint_px") for r in per]),
               "objects": {n: {"iou": agg([r["objects"].get(n, {}).get("iou") for r in per]),
                               "centroid_px": agg([r["objects"].get(n, {}).get("centroid_px") for r in per])}
                           for n in (object_masks or {})},
               "static_ssim": agg([r.get("static_ssim") for r in per]), "events": events}
    write_json({"summary": summary, "per_frame": per, "robot_hand_scale_rows": scale_rows,
                "tracks_reprojection_rows": tracks_check["per_frame"] if tracks_check else None}, project / out)
    worst.sort(key=lambda x: x[0])
    write_rgb((project / out).with_name(Path(out).stem + "_worst.jpg"),
              contact_sheet([w[2] for w in worst[:6]], cols=2, width=960, labels=[f"frame {w[1]} obj IoU {w[0]:.2f}" for w in worst[:6]]))
    vpath = str(Path(out).with_suffix(".mp4"))
    write_video(vid, project / vpath, fps=sim.fps / step)
    return {"summary": summary, "outputs": [out, vpath]}
