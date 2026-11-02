---
name: synthesize-data
description: Synthesis stage - turn one physically executed robot episode (frozen realism-stage result) into a randomised, physically verified, photo-realistically rendered training dataset (spatial, lighting, texture, physics, sensor and trajectory variation), exported in the robot's data format and audited. Use when request.yaml has stage synthesis.
---

# synthesize-data

Inputs (request.yaml / observation.yaml): the frozen metric scene with the executed episode (`scene/`: robots,
reference/commanded/achieved motions, `render.motion`; evaluations in `upstream/`), the source video frames
(`work/frames`), imported masks (`work/track`), the verified RobotBundle (cameras head/wrist_*, actuators,
data_mapping) and the dataset format (`configs/datasets/<format>.yaml`). The scene and the demo are input: never
edit them; everything you vary goes through the variant overlay (Scene IR v0.4, `scene_ir.apply_variant`).

Before synthesis, run `realism_check action=gate` on the imported realism-stage result.
An old trajectory project with only a baseline render has not completed this stage. Keep
reviewed fixed appearance frozen; changes to it require a fresh realism evaluation.

## What to produce
- `synthesis/spec.yaml`: the randomisation spec (`any2sim.synthkit` docstring), each range justified in WORKING.md.
- `synthesis/episodes/<id>/` (synth_rollout): variant.yaml, commanded.npz, achieved.npz, rollout.json; the nominal
  episode (no randomisation) first.
- `synthesis/renders/<id>/` (synth_render): head.mp4, wrist_left.mp4, wrist_right.mp4, check frames + label masks.
- `dataset/<name>/` (dataset_export): the user-supplied format (camera videos, state/action arrays,
  .json, instruction.json, sim_meta.json) + dataset_card.yaml.
- Evidence: synth_rollout.json + sheet + layout, synth_render.json + sheet, dataset_export.json, dataset_audit.json +
  label/diversity sheets, video_compare of the nominal episode.

## Steps (each checked before the next)
1. **Understand the demo**: read the upstream report and the achieved motion: which hand, contact/lift/place/release
   frames (`synthkit.Context(project, spec).demo.events`), how close the arm is to its joint limits along the demo,
   how robust the grasp is. Watch the upstream render. Note what the episode can and cannot be stretched to.
2. **Task predicate**: `task.success` that states the task (e.g. in_container: object at rest inside the target,
   released, target upright). Check it on the nominal rollout (`synthkit.run_attempt(ctx, -1, ..., nominal=True)`);
   it must pass, and must fail on an obvious failure (e.g. the object left on the table).
3. **Ranges from evidence**, one axis at a time (start small, widen while the acceptance stays reasonable):
   - spatial: where the object/target can be (support surface, head-camera view, robot reach: run IK on the
     re-anchored palm path, `synthkit.transfer` info, for a grid of placements; keep margins); yaw ranges that
     the hand can follow (wrist limits); `place_anchor: position` for containers.
   - physics: mass/friction around the IR's priors (their provenance says how uncertain they are), actuator gains
     within the controller's plausible spread, small control latency.
   - lighting: around the reconstructed rig (strength, colour temperature, position, size); materials: hsv shifts of
     the reconstructed textures, procedural/image swaps for large surfaces (table, backdrop) only where the task
     allows (do not recolour the object into something else unless the task is defined by shape).
   - camera mounts: small (mm, sub-degree) perturbations; sensor: exposure, white balance, mild blur/noise.
   - trajectory: speed, via-point offsets during transport, grasp offsets only if the grasp tolerates them (measure:
     ablate one axis at a time on a few attempts; a fragile grasp shows up as drops right after lift).
   Record every decision and its evidence in WORKING.md.
4. **Pilot** (`synth_rollout` with a few episodes): read the rejection table (by stage and reason) and the layout
   plot; look at the MuJoCo sheet. Rejections are information: a range that is mostly rejected is wrong (shrink it)
   or reveals a transfer weakness (fix the transfer in your code, e.g. a better approach for large yaws). Never
   relax the predicate or the physics thresholds to raise acceptance. If accepted placements cluster in one corner,
   the dataset is less diverse than the spec claims: fix the spec.
5. **Generate** the requested number of episodes (`synth_rollout`, resume keeps earlier accepted episodes when the
   spec is unchanged; a changed spec starts over).
6. **Required realism baseline** (skill render-realism, `docs/render_pipeline.md`): render nominal/native previews,
   diagnose and fix materials/illumination, measure (`realism_metrics`), fit the camera model on paired nominal
   views (`camera_calibrate` for source_cam only), and inspect raw/processed/encoded results. Use fixed calibrated
   processing regardless of which randomization axes are enabled; do not introduce unrequested random axes.
   Obtain independent `render/realism_review.json` for the current baseline fingerprint before batch rendering.
   Identity/bitrate-only calibration and using Cycles do not prove realism; unresolved appearance needs repair.
   **Render** (`synth_render`): look at the sheet and at full videos of several episodes, every camera: lighting
   plausible, no artefacts, wrist views like a real wrist camera. Run `video_compare` on the nominal episode
   (`renders: synthesis/renders/nominal/head_frames`, the masks imported from upstream) for render fidelity.
7. **Export** (`dataset_export`) and **audit** (`dataset_audit`): format errors 0, replay from the exported actions
   succeeds, label IoU (exported state vs pixels) high, diversity as the spec claims, images well exposed.
8. **Freeze** a candidate (spec, episodes index + episodes, renders, dataset, evidence; hard links for the large
   files) and ask the `stage-evaluator` (rubric rubrics/synthesis.yaml, skill evaluate-dataset).

## Your own code
Task-specific logic goes in `work/code/` (e.g. a better approach path, an alternative grasp from a different demo
segment, a custom success predicate). Use `synthkit` functions as building blocks; write accepted episodes in the
same layout (variant.yaml, commanded.npz, achieved.npz, rollout.json) so the render/export/audit tools apply.

## Pitfalls
- A demo near the robot's joint limits leaves little room: spatial ranges must respect it (IK errors at place time).
- Grasps found by repair can be fragile: grasp perturbations and heavier objects drop them; measure before widening.
- Rendering dominates cost (~1-3 s per camera-frame): pilot renders on a few episodes, then render everything once.
- The dataset's state is the achieved rollout and the action is the commanded target: never export the commands as
  the state.
- Columns a simulator cannot provide stay zero and are listed as unavailable; do not invent them.

Public-release integration requirement: `dataset_export` and `dataset_audit` name external integration roles,
not bundled tools. Supply independently validated adapters for the requested format. Their absence blocks
dataset export, audit and synthesis acceptance; do not skip the required evidence.
