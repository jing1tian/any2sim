# Independent realism stage

The default pipeline is **scene → trajectory → realism → synthesis**. Robot onboarding is
independent. Trajectory acceptance concerns execution, contact and motion correspondence;
realism acceptance concerns the appearance of that protected execution. Existing projects
are not automatically rerendered or reclassified.

## Entry and execution

```bash
any2sim new appearance --from trajectory_project@001 --stage realism
# Edit projects/appearance/realism/protocol.yaml using permitted reference inputs.
any2sim tool projects/appearance realism_check '{"action":"freeze"}'
any2sim tool projects/appearance realism_render '{}'
any2sim tool projects/appearance realism_check '{}'
# Freeze the candidate and obtain an independent evaluate-realism review.
any2sim tool projects/appearance realism_check '{"action":"adopt","review":"evaluations/001/realism_review.json"}'
any2sim tool projects/appearance realism_check '{"action":"gate"}'
any2sim new dataset --from appearance@001 --stage synthesis
```

`new --stage realism` requires an explicit frozen `project@candidate`, an achieved motion and a passed upstream report naming that candidate;
it creates the protection manifest and protocol draft. For an existing project, `realism_check
{"action":"init"}` creates the same manifests without changing request/stage metadata. This
is useful for migration, not evidence that its trajectory or realism passed. The synthesis
importer requires the new realism review, copies its exact evidence and rechecks portability.

Before freezing a candidate, include scene, project-local referenced robot assets, `realism/`,
`render/`, permitted reference frames/masks and all review evidence. Do not copy only `.blend`.
Freeze by copying, not mutable symlinks. The evaluator's JSON uses paths relative to the
candidate root; the same paths must exist in the live project before adoption. Publish the
accepted JSON into the frozen delivery only through the project's established packaging step;
do not edit previously frozen scene/assets. If evidence layout differs, package a new candidate
and review that exact payload. Hashes must not be rewritten to make an old review look current.

## Frozen protocol

The generated YAML lists actual camera native sizes and proposed achieved-motion row indices.
Edit its fit/validation split and regions **before** `action=freeze`. Source video indices can
differ from motion rows: pair them explicitly; use achieved `frame` values and synchronization
evidence. Unavailable references need an explicit explanation. Each camera has:

```yaml
reference_status: paired  # or unavailable, with reference_note
reference_note: Head source view, synchronized from permitted recording.
native_size: [1280, 720]
fit_frames: [0, 160]
validation_frames: [80, 300]
pairs:
  0: {reference: work/frames/00000.jpg, static_mask: realism/masks/head_0.png}
  80: {reference: work/frames/00080.jpg, static_mask: realism/masks/head_80.png}
  160: {reference: work/frames/00160.jpg, static_mask: realism/masks/head_160.png}
  300: {reference: work/frames/00300.jpg, static_mask: realism/masks/head_300.png}
regions:
  - {name: table_surface, role: surface, box: [500, 450, 780, 600]}
  - {name: cup_support, role: contact, box: [600, 380, 740, 470]}
min_static_pixels: 256
max_mae_regression_dn: 0.0
```

These coordinates are examples, not a reusable mask. Define eligibility and tolerances from
reference evidence, not trial scores. Masks are native-size grayscale, nonzero = eligible;
exclude moving objects, their changing shadows and clipped pixels. Qualified static masks
are required for nonidentity ISP, not for an explicitly uncalibrated identity fallback.
The independent evaluator validates mask meaning and near-duplicates; automatic checks detect
insufficient support, exact duplicate reference pixels and fit/validation index overlap.

Freeze hashes protocol, protected state and reference/mask bytes. Changed frames, thresholds
or eligibility are a new declared protocol requiring a new evaluation, not a retry that can
inherit the prior pass. `init` and `freeze` refuse to overwrite changed protection/locks.

## Render and measure

`realism_render` directly renders the protected achieved motion with the shared synthesis RGB,
camera and encoding path. It does not use synthesis predicates or rerun IK/physics.
It produces full native nominal videos and selected checkpoints:

- `*_noisy.png`: same settings/seed/pose, denoising disabled.
- `raw/check/*.png`: selected baseline output before per-camera ISP. The name `denoised`
  in the diagnostic manifest denotes this stage; if baseline denoising is off it is unfiltered.
- `check/*.png`: after fixed camera processing.
- `*_decoded.png`: decoded from the actual encoded video, reverified by the checker.
- `*_light.exr`: linear multilayer diffuse/glossy direct/indirect/color passes. Distorted
  cameras retain the undistorted render canvas in EXR; RGB checkpoints use the camera map.

`preview_frames` is for cheap isolated trials only. It cannot satisfy the video/decoded gate.
The baseline must be frozen during a run. Uniform achieved timestamps are required by the
constant-rate video path; irregular timing fails instead of silently retiming the trajectory.

`realism_check` measures per-region RGB, clipping and gradient energy; stage-to-stage MAE;
paired static held-out MAE; decoded video frame count/fps, changes and duplicates. For
nonidentity ISP, every declared held-out checkpoint must have sufficient static support and
meet its frozen nonregression tolerance. Rejected/unsupported correction blocks readiness and
reports the explicit per-camera `{}` fallback to apply **and rerender**. It does not silently
change the scene. `realism_check {"action":"fallback_isp"}` explicitly applies identity only
to rejected cameras, preserves the previous model in `realism/isp_fallback_history.json`,
and returns `rerender_required`; stale diagnoses cannot apply a fallback. Identity processing must actually preserve pixels and is not calibration.
No automatic image metric establishes material likeness or perceptual realism.

Protection covers motions and file bytes, physics/collision (including robot collision meshes),
placement, registration and geometric cameras. Visual geometry/material changes are allowed
but remain subject to contact/shape review and declared revisions. If physics or registration
must change, return to trajectory reconstruction and establish a new protected handoff.

Optional `realism/trials.json` is an array of trial records: `hypothesis`, `parameter`,
`outcome` (`improved`, `regressed`, `ineffective`), evidence, and controlled-input descriptions.
At `ineffective_trial_limit` repetitions of a hypothesis (default 2), record a genuinely new
`next_hypothesis` or `stop_reason`; the checker blocks an unexplained repeated dead end.
This log records causal investigation, not a numeric optimizer or an automatic visual grade.

## Independent review and promotion

The evaluator follows `skills/evaluate-realism` and `rubrics/realism.yaml`. It writes a normal
criterion-by-criterion YAML report and independently remeasures the frozen candidate without
writing tool receipts into it. For example, in the base Python environment:

```python
from pathlib import Path
from any2sim.realism import check

check(Path("projects/appearance/candidates/001"),
      out=str(Path("projects/appearance/evaluations/001/check.json").resolve()))
```

Use `evaluations/001/check.json` as the review's `diagnostics` path, and preserve that path
when packaging the accepted decision and evidence. Extra rendering/measurement tools run
on an isolated copy inside the evaluation folder. A JSON review has this structure:

```json
{
  "schema_version": 2,
  "stage": "realism",
  "fingerprint": "exact current baseline fingerprint",
  "builder": "builder session identifier",
  "evaluator": "independent evaluator session identifier",
  "status": "pass",
  "diagnostics": "evaluations/001/check.json",
  "checks": {
    "materials": {"status": "pass", "note": "Specific observed finding and limitations", "evidence": ["evaluations/001/findings.md"]}
  },
  "evidence_hashes": {"evaluations/001/findings.md": "sha256 of actual bytes"}
}
```

The abbreviated example needs **all** checks: materials, lighting, signal_chain, surface_scale,
contact, thin_surfaces, temporal, perceptual_realism, reference_fidelity, integrity, and
`camera:<name>` for every output camera. Each requires `pass`, a specific note and evidence.
When no thin surface exists, that check documents evidenced absence rather than inventing a
material experiment. Missing paired views at usable fidelity require explicit scope limits;
at precise fidelity they block pass. Perceptual realism and reference correspondence are
separate findings. No averaging compensates for a failed item.

`evidence_hashes` includes the independent diagnostic JSON, every input it hashed, and every
file cited by every check. All evidence must be project-relative and resolve inside the
project. Evaluator identities are auditable declarations, not access-control/authentication.
`adopt` validates and copies a passing evaluator decision; it never fills missing hashes or
creates a pass. `gate`, non-nominal `synth_render` and `dataset_export` recheck assets, settings,
protected state, protocol and evidence hashes. Legacy reviews require re-evaluation.

Existing explicit `realism_override_reason` options remain diagnostic user-authorized
exceptions. They do not create a stage pass or permit the normal synthesis-stage handoff.
Camera/pixel/material/physics failures remain visible in their output provenance.

Public-release integration requirement: `dataset_export` and `dataset_audit` name external integration roles,
not bundled tools. Supply independently validated adapters for the requested format. Their absence blocks
dataset export, audit and synthesis acceptance; do not skip the required evidence.
