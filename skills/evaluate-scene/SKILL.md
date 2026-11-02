---
name: evaluate-scene
description: Independent evaluation of a frozen any2sim scene candidate against rubrics/scene.yaml - geometry, structure, scale, cleanliness, materials, lighting, physics, appearance, deliverables - each judged separately with measurements and by looking. Used by the stage-evaluator sub-agent.
---

# evaluate-scene (evaluator)

You judge a frozen candidate (`projects/<id>/candidates/<k>/`); you never modify it. Measure with the tools,
pointing them at the candidate (copy the candidate's scene to `projects/<id>/evaluations/<k>/scene_copy` if a
tool needs `scene/` and the project's live `scene/` differs; record what you evaluated by hash).

## Protocol (fixed; same for every candidate)
1. **Validity**: `any2sim.scene_ir.validate`; deliverables exist (scene.yaml, scene.blend reopen check, MJCF).
2. **Geometry**: `geometry_metrics` on all distinct frames of every rest state (targets = the project's tracked
   object masks, ignore = dynamic masks). Look at `_worst.jpg`.
3. **Structure**: for each object, crops of 3 real frames next to `render_views` overlays and Blender turntables.
   Count and place salient parts (openings, slots, rims, handles, lugs, lids). Generated-looking or missing structure
   fails `object_structure` even with high IoU.
4. **Scale**: object extents vs category priors and the project's scale cues.
5. **Cleanliness**: `blender_render` with `background_only`, `novel: auto`, `turntables` (all objects) into your
   evaluation folder; run `sam3_segment` on those renders with prompts
   ["hand", "arm", "glove", "person", "controller", "strap", "cable", "shadow"]; inspect every material texture image.
   Any hand/equipment detection or baked shadow on a background-only render fails `scene_clean`.
6. **Materials/lighting**: static surfaces not raw video copies; object textures consistent; cast shadows present
   and resembling the video's.
7. **Physics**: `mujoco_check` (default sets).
8. **Appearance**: `image_metrics` for source frames (real vs render), regions = static surfaces and each object,
   ignore = dynamic masks dilated 15 px. Report PSNR/SSIM/LPIPS; judge colour/brightness agreement.

## Report
`projects/<id>/evaluations/<k>/report.yaml`:
```yaml
candidate: candidates/<k>   rubric: rubrics/scene.yaml <version>   fidelity: <level>
verdict: pass | revise | blocked
criteria: {<criterion>: {status: pass|fail|unknown, metrics: {...}, evidence: [files], note: <one line>}}
issues: [{code, criterion, evidence, owner: geometry|appearance|physics|cameras, action: <concrete next step>}]
```
A criterion you could not verify is `unknown`, which blocks `pass`. A false pass is worse than a false fail.
