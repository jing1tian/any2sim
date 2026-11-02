---
name: evaluate-dataset
description: Independent evaluation of a frozen synthesis-stage candidate against rubrics/synthesis.yaml - inputs, upstream integrity, spec justification, physics honesty, per-episode validity, replay from exported actions, nominal fidelity, diversity, rendering, visibility, label consistency, format, deliverables. Used by the stage-evaluator sub-agent.
---

# evaluate-dataset (evaluator)

You judge a frozen candidate (`projects/<id>/candidates/<k>/`); never modify it. Re-measure; the builder's numbers
and WORKING.md are claims. Write everything to `projects/<id>/evaluations/<k>/`.

## Protocol (fixed)
1. **Inputs and upstream**: observation.yaml (withheld items untouched; nothing under real-data paths read: grep the
   builder's code and logs); `scene_ir.core_digest(load(scene))` equals `upstream.scene.scene_core_sha256` and
   synthesis/episodes.yaml base_core_sha256; upstream motion files hash to `upstream.trajectory.motions_sha256`;
   the robot bundle sha matches.
2. **Spec**: read synthesis/spec.yaml and the WORKING.md justification. Check a few claims yourself: e.g. sample
   placements at the range corners and run `synthkit.transfer` (IK errors), project them into the head camera, check
   the support polygon. Ranges chosen only to raise acceptance, or that exclude the hard cases silently, fail.
3. **Episodes**: for several accepted episodes (your choice, include the worst by rollout metrics) re-run the
   attempt from its variant (`synthkit.materialize` + `simkit.Sim(...).rollout` of commanded.npz): same achieved
   motion, success predicate true, physics thresholds met, audit ok. Check that variant poses are only initial
   states and that nothing else changes objects during the episode.
4. **Replay**: `dataset_audit` into your folder with `replay` episodes you pick (not the builder's seed): replay
   success and q differences.
5. **Nominal fidelity**: synth_rollout nominal `reproduces_demo`; `video_compare` of
   `synthesis/renders/nominal/head_frames` vs the source video with the imported masks: static SSIM, object IoU.
6. **Diversity**: dataset_audit diversity vs the declared ranges; open synth_rollout_layout.png (accepted vs
   rejected) and dataset_audit_diversity.png; judge whether rejection collapsed a range.
7. **Rendering and visibility**: watch full videos (all cameras) of at least 3 episodes with different lighting and
   materials; check check-frames and label masks; luminance statistics; object visible at the start in the head
   camera; wrist views plausible (distortion, hand in view, not black).
8. **Realism** (skill render-realism): read evidence/camera_calibrate.json (held-out metrics, rejected components and
   why); run `realism_metrics` yourself on the nominal episode and on all episodes (source video); compare the
   low-level ratios with the band; look at the crop sheet; check that dressing never intrudes on the robot's
   workspace (watch videos) and is absent from MuJoCo.
9. **Labels and format**: dataset_audit labels (IoU per camera) and format (0 errors); open the label sheet; read one
   episode's .npy.meta.json, .json and sim_meta.json yourself (state = achieved, action = commanded, unavailable
   columns zero and listed).
10. **Report** `evaluations/<k>/report.yaml`:
```yaml
candidate: candidates/<k>   rubric: rubrics/synthesis.yaml <version>   fidelity: <level>
verdict: pass | fail
criteria:
  <criterion>: {status: pass | fail | unknown, metrics: {...}, evidence: [files], notes: <text>}
most_valuable_fix: <text>
```
A false pass is worse than a false fail; unknown blocks pass.

Public-release integration requirement: `dataset_export` and `dataset_audit` name external integration roles,
not bundled tools. Supply independently validated adapters for the requested format. Their absence blocks
dataset export, audit and synthesis acceptance; do not skip the required evidence.
