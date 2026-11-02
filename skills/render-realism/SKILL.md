---
name: render-realism
description: Improve and evaluate robot-scene rendering against real video through camera/view alignment, surface detail, room lighting, per-camera image models and controlled comparisons. Use for the independent realism stage after trajectory reconstruction, and for revalidation when synthesis changes the fixed rendering baseline.
---

# render-realism

A render looks synthetic for measurable reasons: wrong tonal range (lifted blacks, compressed highlights), no sensor
noise/compression, too-clean surfaces, flat lighting, CAD-perfect robot, no motion blur. Fix the cause where it lives;
measure before and after; never tune the measurement.

## Independent pipeline stage

The default order is `scene -> trajectory -> realism -> synthesis`. Start from the accepted
achieved trajectory; use `any2sim new <id> --from <trajectory>@<candidate> --stage realism`.
The importer creates `realism/protected.json` and a draft protocol. Keep commanded/achieved
bytes, timing, placement, dynamics, collision and camera geometry fixed. Declare visual-only
revisions; geometry/registration repairs that affect these invariants return to their owning
stage. Do not reproduce or regenerate the trajectory merely to render it.

Follow [the executable stage contract](../../docs/realism_stage.md): edit/freeze the protocol,
run `realism_render`, measure with `realism_check`, freeze a candidate and delegate to
`evaluate-realism`. Publish the evaluator's passing decision through `action=adopt`.
Full synthesis requires that stage's current evidence; finishing a batch is not a pass.

## Establish the comparison

- Use authorized reference views and synchronized logged joints when available. Compare real / accepted baseline /
  candidate at the same pose, camera and image scale. Keep render settings and camera response fixed when isolating
  a material or geometry change; evaluate a camera-view correction separately from the appearance change.
- Verify the output camera's intrinsics, distortion, resolution and pose before tuning surfaces. A human-demo
  `source_cam` is not necessarily the robot head view. Keep it for reconstruction checks; use `render.head_camera`
  to select a separate scene camera when needed. Carry that selection through rendering, visibility sampling,
  label audits, previews and any exported calibration. Do not hide registration errors by changing albedo.
- Choose fitting and validation frames/regions before fitting, spanning poses and exposure conditions. Repeated
  use of validation frames to select candidates is not fresh unseen-data evidence; do not change the comparison
  masks, thresholds or frame selection to rescue a candidate.
- Check validation frames for duplicate content, including repeated poses in static ramps: disjoint frame indices
  do not establish independent evidence. Predeclare the minimum eligible static regions and pixel support for
  camera fitting; too few valid regions means insufficient evidence, even if every remaining region improves.
  Retaining identity ISP after rejection is a fallback, not successful camera calibration.
- With different task props or deliberately omitted hardware, compare shared surfaces and registered robot crops.
  Label hidden props or other diagnostic changes; a logged-pose render is not an executed simulation of that task.
  Background realism metrics may exclude robots, but robot appearance still needs its own visual comparison.

## Workflow

1. **Diagnose the raw render.** Use `realism_metrics` where available: `c2st_auc`, `kid_x1000`, noise/sharpness ratios,
   luminance percentiles, saturation and colourfulness, together with `realism_crops.jpg`. Fixed shared-surface
   RGB/Lab statistics can localize brightness and colour errors. State what was measured; a regional colour
   statistic is not a whole-image realism score. Record the diagnosis in WORKING.md.
2. **Fix structure, materials and illumination.** Use [refine-appearance](../refine-appearance/SKILL.md) for room
   structure, clean surfaces, causal wear and light transport. Use [onboard-robot](../onboard-robot/SKILL.md) for
   candidate bundle revisions, including gloves and markings; keep the supplied bundle immutable. Missing camera
   housings, fixtures or cables are scope decisions: do not reintroduce hardware the user excluded.
   Compare several real frames before changing light power. If the reference exposure varies, do not fit the rig
   to its brightest frame or bake that exposure into albedo. If channel ratios, saturation or spatial falloff
   remain wrong, inspect albedo and light layout instead of repeating global brightness changes or adding noise.
3. **Fit the camera image model after checking the raw appearance.** Keep raw and processed outputs separately.
   Current `camera_calibrate` pairs `work/frames` with `source_cam`; other head/wrist cameras need their own paired
   reference workflow. Realism-stage source-view raw images are in `realism/renders/nominal/head_frames`
   (synthesis uses `synthesis/renders/nominal/head_frames`); another
   selected head uses `<camera>_frames` so it cannot silently become a source-video calibration input.
   Fit on static shared pixels, excluding robots/people, moving objects and their changing shadows. Define each
   component's acceptance checks before fitting; do not relax a failed calibration gate. Retain tone/blur/
   sharpening only when their held-out checks do not regress; inspect encoded outputs too.
   Resolve ISP independently per camera (`camera_isp.for_camera`): `render.camera_isp` overrides its own camera
   calibration, and an explicit `{}` means identity. After a current failed diagnosis, `realism_check action=fallback_isp`
   records and applies that fallback; rerender and re-evaluate before adoption. Never let wrists inherit the head's tone, noise or bitrate.
   When estimating noise from adjacent frames, reject motion, clipping, duplicates and exposure drift; compression
   can give zero or unstable residuals. Retain only stable estimates and call decoded-video noise an empirical
   image model, not raw sensor calibration. Validate pixel-scale blur/noise at native resolution; a smaller
   preview is not equivalent calibration evidence.
4. **Establish the nominal baseline before widening variation.** A frozen scene change needs a declared revision;
   synthesis variations belong in `synthesis/spec.yaml` overlays. Start with small changes around the accepted
   deployment appearance. Keep broad hue/checkerboard/dark-scene robustness profiles separate from fidelity
   comparisons. Render-only dressing is suitable for surface appearance or geometry safely outside the swept
   workspace; props that can contact the robot or task objects need collision geometry and physics validation.
5. **Render and inspect all relevant views.** Compare Eevee/Cycles keyframes with the same pose and lights when
   diagnosing indirect light or reflections; switching engines alone cannot supply missing scene structure.
   Choose the engine/sample budget from that evidence and keep the dataset renderer consistent. Check native
   head and wrist views, then a continuous achieved-motion excerpt with appropriate keyframed motion blur
   (half a frame is a starting estimate, not a calibrated shutter). Measure nominal fidelity separately from
   the deliberately wider distribution of randomized episodes.

### Sampling and preview evidence

- For material-strength previews, keep the procedural seed, pose, albedo, lights, camera response and sampling
  fixed; separate surface-only and combined edits when needed to attribute the change. Show the real reference,
  previous selected sample and new sample in matching native crops alongside an overview; label enlarged crops.
  Reuse an unchanged baseline render only when its inputs and settings still match. Render only affected variants.
- Preserve each trial's assets and evidence, and record the selected revision explicitly when the user reverts.
  A choice between previews establishes the preferred material direction, not an independent realism pass or
  completion of a full episode rerender. Do not promote the most recently generated trial merely because it is latest.
- Compare sample counts at fixed materials, lights, pose and camera response before increasing the render budget.
  Less Monte Carlo speckle does not recover cloth weave, wear or camera noise; it may expose an overly smooth
  material. Select the lowest budget that preserves the required evidence rather than making a costly diagnostic
  setting the universal default.
- Inspect native raw and processed frames separately from decoded preview video. Encoding can change sharpness,
  colour and noise even with identity ISP; codec settings are not a calibrated sensor response. Use continuous
  achieved-motion excerpts for highlight stability, texture swimming and exposure jumps, plus representative
  lighting variants. State the inspected duration, views and coverage; a short excerpt cannot certify an entire
  episode or all randomized episodes.

## Verification and adoption

- This is a required part of standard rendering, not an optional post-process. Fixed materials, light transport
  and per-camera ISP must be established even when the user permits only spatial/lighting randomization.
  Do not add random texture, sensor or dressing axes to satisfy this requirement.
- `blender_render` applies the source camera ISP by default; `camera_model: false` is for raw diagnostic fits.
  Both renderers resolve fixed settings from `render.realism_baseline` (fallback Cycles, 128 samples, denoising,
  half-frame motion blur). Those settings alone are not evidence of realism. Retain identity ISP if the paired
  validation rejects correction; fix underlying appearance instead of adding arbitrary noise or tone.
- Complete the realism stage before synthesis: use `realism_render` for native signal-chain checkpoints and
  continuous achieved-motion videos, then delegate independent baseline review. The renderer writes
  `render/realism_baseline.json` containing the exact scene/assets/calibration/settings fingerprint. The
  evaluator writes schema-v2 `evaluations/<k>/realism_review.json` with all required findings and evidence
  hashes; `realism_check action=adopt` validates and publishes the unchanged decision. See
  [the executable contract](../../docs/realism_stage.md) for the full checks. Do not self-approve.
  Missing paired wrist references must be stated; never reuse the head ISP or claim calibrated wrist fidelity.
- A missing, failed or stale review blocks full variant rendering and dataset export. Nominal and previews stay
  available to repair the baseline. Asset, camera ISP or render-setting changes require fresh evidence/review;
  old variant-only render manifests must be rerendered. Preserve raw and processed check images for every camera.

- Compare each candidate with both the real reference and the accepted baseline, not only a rejected trial.
  Review shape/texture structure, colour, exposure and physical validity separately. More detailed geometry can
  coexist with worse colour: retain verified improvements and revert a failed light/tone component rather than
  accepting the whole trial or continuing to raise global brightness.
- For a reviewed candidate, finish building assets before freezing hashes and starting its evidence renders.
  Record the scene core plus referenced mesh/texture/GLB hashes and render settings; a render started while those files are changing is not a frozen
  candidate comparison. Obtain independent visual evaluation under the project rules; structural checks alone
  do not establish likeness. Report remaining differences and metric regressions. Limit acceptance claims to
  what the comparisons establish; a scoped surface improvement is not full reconstruction certification.
- For an appearance-only revision, verify unchanged physics and compare paired actuator-driven rollouts where
  contact equivalence matters. Then update the declared Scene IR revision, rebuild compiled assets and exercise
  the actual rendering/label path. Mark old render manifests stale. Distinguish native stills, a short or reduced-
  resolution preview, full episode rendering and dataset validation in the handoff.
- If a backend cannot rerender labels (for example, its graphics context is unavailable), mark that check
  `not_run`/unknown. Unchanged compiled physics and matching prior/current renderer labels support nonregression;
  they do not substitute for a fresh cross-backend label audit or dataset validation.
- Keep a component or comparison-phase pass separate from the overall realism gate. Accepted local improvements
  do not authorize full batch rendering when the integrated review still fails. If the evaluation budget ends,
  record the unresolved differences and budget stop explicitly rather than declaring likeness complete.

## Causal diagnosis and stopping

- Trace detail through matched noisy, denoised, processed and actual video-decoded pixels
  before strengthening a normal map. Raw here means pre-ISP display RGB, not linear radiance;
  use the separate linear light passes for transport diagnosis.
- For colour spill or washed-out highlights, inspect diffuse/glossy direct and indirect
  contributions. A temporary neutral-albedo room can isolate coloured bounce; preserve the
  comparison and never adopt that diagnostic substitution as the reconstruction. Avoid
  compensating a transport error with tinted white paint or a global LUT.
- Validate microstructure in native close crops and final-size overviews. Report physical
  wavelength and projected size; head-view invisibility alone is not grounds for amplification.
- Record trials in `realism/trials.json` with hypothesis, changed parameter, controlled inputs,
  outcome and evidence. Repeated ineffective trials require a different hypothesis or an
  explicit stop reason; do not spend the remaining budget repeating the same adjustment.
- Keep natural appearance and reference correspondence as separate review findings. Neither
  a good pixel score nor a plausible novel view can substitute for the other.

## Metric interpretation

- Paired per-pixel metrics (SSIM, deltaE) reward matching the scene's lighting layout, not realism: a global tone
  curve that stretches a flat render to the real image's spread looks worse. Use them to guard, not to optimise.
