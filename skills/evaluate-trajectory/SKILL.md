---
name: evaluate-trajectory
description: Independent evaluation of a frozen trajectory-stage candidate against rubrics/trajectory.yaml - inputs, robot integrity, registration, placement, retargeting, physics honesty, feasibility, task success, motion fidelity, render, deliverables. Used by the stage-evaluator sub-agent.
---

# evaluate-trajectory (evaluator)

You judge a frozen candidate (`projects/<id>/candidates/<k>/`); never modify it. Re-measure; the builder's
numbers are claims. Write everything to `projects/<id>/evaluations/<k>/`.

## Protocol (fixed)
1. **Validity / deliverables**: `scene_ir.validate` (v0.3: robots, commanded + achieved motions, render.motion);
   `stages.scene_drift(project)`; bundle sha in the scene vs observation.yaml robot_description; files exist
   (mujoco/scene.xml, blender/scene.blend, renders video, isaac/stage.usda + playback.usda, evidence).
2. **Physics**: re-run `mujoco_rollout` on the candidate's commanded motion with `out: eval_<k>` and `register:
   false` (it must reproduce the achieved motion: compare q and object poses); read audit, determinism, penetration
   by category, limits, saturation, tracking error.
3. **Task success**: object final pose vs the scene's last rest state (video_compare events); stability after release
   (object speed at the end); no drops (object z, contacts).
4. **Registration and metric scale**: `scene_ir.validate` (registrations rigid); read the builder's motion code to
   confirm no scale is applied to track positions; re-derive the reference palm path from `T_world_tracks @ tracks`
   and compare. `handkit.reprojection` on frames you choose (not the builder's fitting frames). Recompute the
   scale cues yourself (`handkit.track_events` + `scale_cues`, `register_tracks` with `fit_scene_factor` on your
   frames) and compare with the applied scale (scene/revisions.yaml). `video_compare` robot_hand_scale.
5. **Contact timing**: first/last robot-object contact of the manipulated object (mujoco contacts json) vs the
   tracks' grasp/release events.
6. **Motion fidelity**: `video_compare` into your folder (hand masks, object masks, renders): object IoU over the
   episode, hand precision, event delays. Watch `video_compare.mp4` and the side-by-side render video.
7. **Render**: look at frames from approach, grasp, transport, release, end: robot visuals (glove/skin, arm shells),
   lighting consistent, shadows, no flicker or popping, nothing floating; sam3_segment on a few renders with
   ["hand", "person", "controller"] (human parts must not remain in the robot render, apart from the scene's own
   known issues).
8. **Isaac/USD**: `export_usd` checks ok (joint frames consistent, bodies with colliders, articulation root).

## Report
`evaluations/<k>/report.yaml` (same format as evaluate-scene: verdict, criteria with status/metrics/evidence/note,
issues with owner and concrete action). Unknown blocks pass. A false pass is worse than a false fail.
Include `stage: trajectory` and `candidate: candidates/<k>` so the realism importer can
verify which frozen execution passed; a report about the mutable project root is insufficient.


Appearance acceptance belongs to the following `realism` stage. Judge the trajectory render
for achieved-motion playback, camera/registration, visibility and temporal continuity. Report
photometric discrepancies descriptively; do not certify materials/lighting/ISP here or require
that later stage's pass to accept a physically valid trajectory.
