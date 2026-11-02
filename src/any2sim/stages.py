"""Stages and resuming: start a project at a later stage from another project's result.

    any2sim new <id> --from <project>[@<candidate>] --stage trajectory [--tracks <table> --tracks-map <yaml>]
                     [--robot assets:my_robot] [--import-work track ...]
    any2sim new <id> --from <trajectory project>@<candidate> --stage realism
    any2sim new <id> --from <realism project>[@<candidate>] --stage synthesis [--import-work track ...]

The upstream result is copied (frozen: scene/, the source video and frames, chosen work folders, its evaluation
report) and pinned by hash in project.yaml `stages` and observation.yaml `upstream`. Supplied trajectories are
converted to one canonical table (inputs/hand_tracks.npz + .yaml) so the agent never needs the raw source.

Canonical hand tracks (per side in {left, right}; T rows = video frames):
  t (T,), frame (T,)                      seconds / video frame index
  <side>_wrist_pos (T,3), <side>_wrist_rotmat (T,9)   human wrist pose in the tracks' frame (row-major R)
  <side>_keypoints (T,K*3)                hand keypoints in the same frame (layout in the .yaml)
  <side>_hand_q (T,J)                     optional robot-hand joints from an external retargeter
"""
from __future__ import annotations

import fnmatch
import json
import shutil
from pathlib import Path

import numpy as np

from . import config as C
from .common import sha256_file
from .project import Project, now

STAGES = ("scene", "trajectory", "realism", "synthesis")


def _src_dir(spec: str) -> tuple[Path, Path, str | None]:
    name, _, cand = spec.partition("@")
    proj = Path(name) if Path(name).is_absolute() else Path(C.config()["projects_dir"]) / name
    if not (proj / "project.yaml").exists():
        raise FileNotFoundError(f"no project {proj}")
    src = proj / "candidates" / cand if cand else proj
    if not (src / "scene" / "scene.yaml").exists():
        raise FileNotFoundError(f"no scene in {src}")
    return proj, src, cand or None


def convert_tracks(table: str, mapping: str, out_dir: Path) -> dict:
    """Raw per-frame table + mapping yaml -> inputs/hand_tracks.npz/.yaml (canonical)."""
    import yaml
    from .robot import datalog as DL
    mp = yaml.safe_load(Path(mapping).read_text())
    tab = DL.load_table(table)
    n = len(next(iter(tab.values())))
    fps = float(mp["time"]["fps"])
    arr = {"t": np.arange(n) / fps, "frame": np.arange(n, dtype=float)}
    meta = {"frame": mp["frame"], "time": mp["time"], "units": {"length": "m", "angle": "rad"}, "sides": {}}
    for side, s in mp["sides"].items():
        R = DL.to_matrix(DL.column(tab, s["wrist_rotation"]), s["rotation_format"])
        arr[f"{side}_wrist_pos"] = DL.column(tab, s["wrist_position"])
        arr[f"{side}_wrist_rotmat"] = R.reshape(n, 9)
        arr[f"{side}_keypoints"] = DL.column(tab, s["keypoints"])
        m = {"wrist": "wrist_pos + wrist_rotmat (row-major) = human wrist pose", "keypoints": s.get("keypoint_description", ""),
             "keypoint_layout": s.get("keypoint_layout", "auto")}
        if s.get("hand_q"):
            arr[f"{side}_hand_q"] = DL.column(tab, s["hand_q"])
            m["hand_q_joints"] = s["hand_q_joints"]
            m["hand_q"] = s.get("hand_q_description", "robot-hand joints from an external retargeter (unverified)")
        meta["sides"][side] = m
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "hand_tracks.npz", **arr)
    meta["source_columns_used"] = sorted({v for s in mp["sides"].values() for k, v in s.items()
                                          if k in ("wrist_position", "wrist_rotation", "keypoints", "hand_q")})
    meta["withheld_columns"] = sorted(k for k in tab if any(fnmatch.fnmatch(k, p) for p in mp.get("withheld", [])))
    (out_dir / "hand_tracks.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    return meta


def new_from(pid: str, src_spec: str, stage: str, tracks: str | None = None, tracks_map: str | None = None,
             robot: str | None = None, import_work: list | None = None, task: str | None = None,
             notes: str = "", force: bool = False) -> Project:
    import yaml
    if stage not in STAGES[1:]:
        raise ValueError(f"stage must be one of {STAGES[1:]}")
    sproj, src, cand = _src_dir(src_spec)
    if stage in ("realism", "synthesis"):
        from .scene_ir import load as _load
        from .realism import require_achieved
        _sc = _load(src / "scene")
        require_achieved(src, _sc)
        if tracks or robot:
            raise ValueError("realism/synthesis must preserve upstream tracks and robot; revise the trajectory first")
    if stage == "realism":
        if not cand:
            raise ValueError("realism needs a passed frozen trajectory: use --from project@candidate, not a mutable project root")
        report = sproj / "evaluations" / (cand or "") / "report.yaml"
        data = (yaml.safe_load(report.read_text()) or {}) if report.is_file() else {}
        criteria = data.get("criteria", {})
        report_stage = data.get("stage", data.get("rubric"))
        candidate_path = Path(str(data.get("candidate", "")))
        candidate_path = candidate_path if candidate_path.is_absolute() else sproj / candidate_path
        if (report_stage not in ("trajectory", "rubrics/trajectory.yaml") or data.get("verdict") != "pass"
                or candidate_path.resolve() != src.resolve() or not criteria
                or any(c.get("status") != "pass" for c in criteria.values())):
            raise ValueError("realism needs a passed frozen trajectory candidate and its independent report")
    if stage == "synthesis":
        from . import render_pipeline as RP
        saved = json.loads((src / "render/realism_baseline.json").read_text()) if (src / "render/realism_baseline.json").exists() else {}
        if not saved:
            raise ValueError("synthesis needs an independently passed realism stage; run --stage realism first")
        snapshot = RP.baseline(src, _sc, saved["settings"])
        RP.review(src, snapshot, saved["settings"]["cameras"], required=True)
    root = Path(C.config()["projects_dir"]) / pid
    if (root / "project.yaml").exists() and not force:
        raise FileExistsError(f"{root} exists (use --force)")
    proj = Project.create(root, {"from": src_spec, "stage": stage}, notes)
    # frozen upstream: scene, video, frames, chosen work folders, evaluation report
    if (root / "scene").exists():
        shutil.rmtree(root / "scene")
    shutil.copytree(src / "scene", root / "scene")
    # Preserve local bundle/asset paths, including appearance-only derived bundles.
    # Only references explicitly present in the authorized IR are followed.
    from .scene_ir import load as _load
    def copy_refs(value):
        if isinstance(value, dict):
            for v in value.values(): copy_refs(v)
        elif isinstance(value, list):
            for v in value: copy_refs(v)
        elif isinstance(value, str) and len(value) < 1024 and "\n" not in value and not Path(value).is_absolute():
            try:
                original = (src / "scene" / value).resolve()
                destination = (root / "scene" / value).resolve()
            except OSError:
                return  # Provenance text need not be a valid pathname.
            if original.is_relative_to(src.resolve()) and destination.is_relative_to(root.resolve()) and not destination.exists():
                if original.is_dir():
                    shutil.copytree(original, destination)
                elif original.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, destination)
    copy_refs(_load(root / "scene"))
    if stage == "synthesis":
        # Review evidence is copied by hash, never by an unbounded project tree walk.
        review = json.loads((src / "render/realism_review.json").read_text())
        names = {"render/realism_review.json", "render/realism_baseline.json",
                 "realism/protocol.yaml", "realism/protected.json", "realism/protocol.lock.json",
                 *review["evidence_hashes"]}
        for name in names:
            from .realism import local_file
            f = local_file(src, name)
            dst = root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
    for f in (src / "inputs").glob("*"):
        shutil.copy2(f, root / "inputs" / f.name)
    frames_source = src / "work/frames" if (src / "work/frames").is_dir() else sproj / "work/frames"
    shutil.copytree(frames_source, root / "work/frames", dirs_exist_ok=True)
    for w in import_work or []:
        work_source = src / "work" / w if (src / "work" / w).is_dir() else sproj / "work" / w
        shutil.copytree(work_source, root / "work" / w, dirs_exist_ok=True)
    up = root / "upstream"
    up.mkdir(exist_ok=True)
    up_stage = STAGES[STAGES.index(stage) - 1]
    rep = sproj / "evaluations" / (cand or "") / "report.yaml"
    if rep.exists():
        shutil.copy2(rep, up / f"{up_stage}_report.yaml")
    for previous in STAGES[:STAGES.index(up_stage)]:
        prior = sproj / "upstream" / f"{previous}_report.yaml"
        if prior.exists():
            shutil.copy2(prior, up / prior.name)
    for f in ("DELIVERABLES.md", "WORKING.md"):
        if (src / f).exists() or (sproj / f).exists():
            shutil.copy2(src / f if (src / f).exists() else sproj / f, up / f)
    from .scene_ir import core_digest, load
    scene_sha = core_digest(load(root / "scene"))
    obs = yaml.safe_load((sproj / "observation.yaml").read_text())
    sobs = obs.get("upstream") or {}
    obs["upstream"] = {"scene": {"from": src_spec if stage == "trajectory" else (sobs.get("scene") or {}).get("from", src_spec),
                                 "scene_core_sha256": scene_sha, "imported": now(),
                                 "work": import_work or [], "report": "upstream/scene_report.yaml" if (up / "scene_report.yaml").exists() else None}}
    if stage in ("realism", "synthesis"):
        from .scene_ir import load as _load
        _sc = _load(root / "scene")
        obs["upstream"]["trajectory"] = {
            "from": src_spec if stage == "realism" else (sobs.get("trajectory") or {}).get("from", src_spec), "imported": now(), "render_motion": _sc["render"]["motion"],
            "motions_sha256": {k: sha256_file(root / "scene" / m["uri"]) for k, m in _sc["motions"].items()},
            "report": "upstream/trajectory_report.yaml" if (up / "trajectory_report.yaml").exists() else None}
        rb = next(iter(_sc["robots"].values()))
        obs["robot_description"] = {"bundle": rb["bundle"], "bundle_sha256": rb.get("bundle_sha256"),
                                    "note": "verified RobotBundle used by the upstream trajectory; use as is"}
    if stage == "synthesis":
        obs["upstream"]["realism"] = {"from": src_spec, "imported": now(),
            "fingerprint": snapshot["fingerprint"], "report": "render/realism_review.json"}
    if tracks:
        meta = convert_tracks(tracks, tracks_map, root / "inputs")
        obs["hand_tracks"] = {"uri": "inputs/hand_tracks.npz", "description": "inputs/hand_tracks.yaml",
                              "sha256": sha256_file(root / "inputs" / "hand_tracks.npz"),
                              "frame": meta["frame"], "time_base": meta["time"],
                              "source": "external estimate (VR tracking + retargeting); verify against the video"}
        obs["withheld"] = sorted((set(obs.get("withheld") or []) - {"hand_tracks"}) |
                                 {"other columns of the source trajectory table (hand_tracks.yaml withheld_columns)",
                                  "camera_calibration", "robot logs / real robot data"})
    if robot:
        from .scene_ir import bundle_digest, resolve_bundle
        bd = resolve_bundle(root / "scene", robot)
        obs["robot_description"] = {"bundle": robot, "bundle_sha256": bundle_digest(bd),
                                    "note": "verified RobotBundle (robot.yaml checks); use as is"}
    obs["actor_type"] = obs.get("actor_type", "human")
    if task:
        obs["task_hint"] = task
    obs["notes"] = (f"{stage} stage resumed from {src_spec}: the source video, the frozen scene"
                    + (" with the executed robot trajectory (robots, motions)" if stage in ("realism", "synthesis") else "")
                    + f" (+ imported work: {', '.join(import_work or []) or 'none'})"
                    + (", estimated hand tracks" if tracks else "")
                    + (f", robot bundle {robot}" if robot else "") + ". Everything else is withheld.")
    (root / "observation.yaml").write_text(yaml.safe_dump(obs, sort_keys=False, allow_unicode=True))
    meta = yaml.safe_load((root / "project.yaml").read_text())
    meta["stages"] = {"scene": {"status": "imported", "from": obs["upstream"]["scene"]["from"], "scene_sha256": scene_sha},
                      stage: {"status": "open"}}
    if stage in ("realism", "synthesis"):
        meta["stages"]["trajectory"] = {"status": "imported", "from": src_spec,
                                        "motions_sha256": obs["upstream"]["trajectory"]["motions_sha256"]}
    if stage == "synthesis":
        meta["stages"]["realism"] = {"status": "imported", **obs["upstream"]["realism"]}
    (root / "project.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    tpl = (C.ROOT / "templates" / f"request_{stage}.yaml").read_text()
    (root / "request.yaml").write_text(tpl.replace("{project}", pid).replace("{task}", task or obs.get("task_hint") or "")
                                       .replace("{robot}", robot or (obs.get("robot_description") or {}).get("bundle", ""))
                                       .replace("{upstream}", src_spec))
    if stage == "realism":
        from .realism import initialize
        initialize(root)
    if stage == "synthesis":
        # Path relocation must not silently invalidate the inherited review.
        RP.review(root, RP.baseline(root, _load(root / "scene"), saved["settings"]),
                  saved["settings"]["cameras"], required=True)
    return proj


def scene_drift(project: Path) -> dict:
    """Has the imported upstream scene been modified? (The trajectory stage may revise it only as a declared new
    scene version.)"""
    import yaml
    from .scene_ir import core_digest, load
    meta = yaml.safe_load((project / "project.yaml").read_text())
    st = (meta.get("stages") or {}).get("scene") or {}
    cur = core_digest(load(project / "scene"))
    return {"imported_sha256": st.get("scene_sha256"), "current_sha256": cur, "changed": cur != st.get("scene_sha256")}


PUBLIC_CHECKS = ("import.json", "preview.jpg", "visuals.json")


def publish_robot(project: str | Path, name: str | None = None, checks_include=PUBLIC_CHECKS) -> dict:
    """Copy a project's verified bundle (projects/<id>/robot) to assets/robots/<name> (keeps <name>/source/).
    Evidence derived from real robot data (overlays of real frames, log fits) stays in the onboarding project: the
    published robot.yaml points there, so agent runs that must not see real robot data can still use the bundle."""
    import yaml
    from .robot.bundle import Bundle, validate
    from .scene_ir import bundle_digest
    src = Path(project) / "robot"
    v = validate(src)
    if not v["ok"]:
        raise ValueError(f"bundle invalid: {v['errors']}")
    name = name or Bundle(src).name
    dst = Path(C.config()["assets"]["robots"]) / name
    old = None
    if (dst / "robot.yaml").exists():                    # remember the replaced revision (appearance-only updates
        from .scene_ir import physics_digest              # keep scenes built against it valid)
        old = {"sha256": bundle_digest(dst), "physics_sha256": physics_digest(dst), "replaced": now(),
               "revisions": yaml.safe_load((dst / "robot.yaml").read_text()).get("revisions") or []}
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("robot.urdf", "robot.xml"):
        shutil.copy2(src / f, dst / f)
    for d in ("meshes", "visual", "collision", "checks"):
        if (dst / d).exists():
            shutil.rmtree(dst / d)
        if (src / d).exists() and d != "checks":
            shutil.copytree(src / d, dst / d)
    (dst / "checks").mkdir(exist_ok=True)
    for f in checks_include:
        if (src / "checks" / f).exists():
            shutil.copy2(src / "checks" / f, dst / "checks" / f)
    prof = yaml.safe_load((src / "robot.yaml").read_text())
    origin = str(Path(project).resolve().relative_to(C.ROOT)) if Path(project).resolve().is_relative_to(C.ROOT) else str(project)
    for c in (prof.get("checks") or {}).values():
        ev = c.get("evidence")
        if ev:
            c["evidence"] = ", ".join(e.strip() if e.strip().split("/")[-1] in checks_include
                                      else f"{origin}/robot/{e.strip()}" for e in ev.split(","))
    prof["onboarding_project"] = origin
    revs = list(prof.get("revisions") or [])
    if old:
        known = {r.get("sha256") for r in revs}
        revs += [r for r in old["revisions"] if r.get("sha256") not in known]
        if old["sha256"] not in known:
            revs.append({k: old[k] for k in ("sha256", "physics_sha256", "replaced")})
    if revs:
        prof["revisions"] = revs
    (dst / "robot.yaml").write_text(yaml.safe_dump(prof, sort_keys=False, allow_unicode=True))
    from .scene_ir import physics_digest
    return {"published": str(dst), "bundle_sha256": bundle_digest(dst), "physics_sha256": physics_digest(dst),
            "physics_unchanged": (old or {}).get("physics_sha256") == physics_digest(dst) if old else None,
            "validate": validate(dst)}
