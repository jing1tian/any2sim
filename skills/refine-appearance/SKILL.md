---
name: refine-appearance
description: Give the reconstructed scene clean, physically-lit, photo-realistic materials and lights in Blender (Cycles) - no baked shadows, hands or artifacts - and verify them from the source view and from new views. Use after the scene geometry is accepted or stable.
---

# refine-appearance

## Goal
An editable `blender/scene.blend` (compiled from the Scene IR) that looks like the real scene from the source
camera and from new viewpoints, with materials that are **clean** and lights that produce the shadows.

## Requirements (checked by rubrics/scene.yaml)
- No artifacts in any material: no baked shadows (from objects, hands, the demonstrator's body), no hands, gloves,
  equipment, cables, inpainting blobs, ghost outlines, seams, stretched or smeared regions. Checked on
  background-only renders, novel views, turntables and the texture images themselves.
- Static surfaces (tables, floors, walls, backdrops) are not raw video copies: use a material whose texture is
  synthesized from a verified-clean exemplar, or procedural/constant, calibrated in colour and brightness to the
  video under the scene lights.
- Objects: observed colours where they are reliable (de-lit), and plausible consistent completion elsewhere (no
  visible boundary between observed and completed regions).
- One lighting rig for the whole clip; objects cast and receive shadows; shadow direction and softness resemble the
  video. Emission only for parts that are genuinely background imagery (and then they must be clean too).
- From the source camera, colours, brightness and structure agree with the video in regions without unmodelled
  content (hands and their shadows). Report raw numbers; a per-frame camera response may be modelled only from
  static background pixels and must be reported separately.

## Tools that help
blender_bake (irradiance/position/normal maps), project_texture (observed colours per texel), synthesize_texture
(clean tileable textures), blender_render (src / background-only / novel / turntables), sam3_segment (look for
hands, shadows, equipment in renders), image_metrics, render_views.

## Strategies that work
- Decide per surface: *observed texture* (only if clean after masking), *exemplar-synthesized* texture (pick a
  clean patch: no shadows, no masks nearby, low outlier fraction), or *procedural/constant* colour.
- De-lighting: albedo ~ observed linear radiance / baked irradiance E (bake E with the geometry that is present in
  those frames). Then set one global light scale so albedos are plausible (< 0.9).
- Lights: estimate the key light direction from real cast shadows (object shadow offsets on the support), its
  size from shadow softness, ambient from shadowed-region brightness; iterate on renders vs frames.
- Always look at: background-only renders (are there ghosts where objects/hands were?), turntables (hidden sides,
  seams), novel views (stretching, floating parts), and the texture images.

## Room lighting and surface detail

- Model enough of the surrounding room to explain bounce light, occlusion and reflections: walls, ceiling and
  table edges/underside can matter even outside the camera view. Place key/fill lights using observed shadow
  direction, softness and brightness gradients. Keep completed room dimensions and unobserved light locations
  labelled as priors; more lights or a path tracer do not establish a measured lighting rig.
- Separate illumination from camera exposure. Fit across representative frames, avoiding clipped white surfaces.
  A rig that matches a bright frame may wash out the table and shells elsewhere. Check channel ratios and spatial
  falloff as well as mean luminance before changing every light together. A per-frame response needs evidence
  from static shared pixels and separate validation; it must not become a sequence of hand-tuned colour fixes.
- Give wear a location, scale and cause: rubbing in the handling area, scratches along repeated tool motion,
  adhesive residue near fixtures and small edge damage. Use restrained albedo and roughness changes; scuff
  brightness should respond to the light. Uniform dirt, low-frequency colour clouds and random marks everywhere
  usually add a new synthetic pattern. Clean material means free of baked illumination/foreign content, not
  necessarily a pristine object.
- Distinguish persistent wear from transient shadows across reference frames before placing marks. Match the
  observed contrast, irregular edges and edge softness as well as location; crisp procedural strokes can look
  less faithful than subtle abrasion. Do not add a proxy for an absent demonstrator just to reproduce their shadow.
  Compare material-only, light-only and combined candidates under the same camera response; report regional
  regressions even when the aggregate colour error improves.
- Match detail to its physical scale. Cloth folds that change silhouette, parallax or cast shadows need geometry;
  weave belongs in fine normal/roughness detail. Generate folds with plausible directions, slack and local
  tension rather than regular waves or dense sharp ridges. Check close wrist views as well as the head view:
  wrinkles that read well at a distance can look like stiff folded paper nearby. Generated folds and wear may
  reproduce a surface type without recovering the exact observed marks; state that distinction.
- Inspect texture colour spaces, UV scale and actual backend material support. Roughness/normal maps are data,
  not sRGB colour; verify their physical scale and strength in the compiled scene. Compare a representative
  same-pose Eevee/Cycles pair when transport is uncertain, rather than assuming more samples fix a material error.

## Tuning surfaces that look too smooth

- Distinguish highlight roughness from visible surface relief. Raising an already high roughness value may only
  flatten reflections; use normal/bump detail for small relief, and geometry for silhouette or parallax. Derive
  normals and roughness variation from related surface structure, preserving the established albedo while
  isolating relief. Normal maps do not change contact geometry.
- Tune physical wavelength and height together: normal response depends on slope, not height alone. Verify the
  mesh's UV-to-metre mapping and the detail's projected size at native resolution. If fine detail disappears in
  the head view, inspect a closer view before amplifying it. Excess amplitude can turn a coated tabletop into
  apparent coarse rubber or stone, even with unchanged colour. Label generated scales as priors, not measurements.
- Strength increases expose procedural regularity. Repeated sinusoidal micro-folds can become conspicuous ribs;
  use irregular spacing, direction and amplitude supported by the surface type. Directionally smoothed noise
  can model plausible small cloth wrinkles, but it does not recover observed folds. Inspect tiling and repetition
  as well as strength, and retain a weaker version when the stronger one changes the apparent material class.

For real-versus-render comparison, per-camera ISP and adoption checks in trajectory/synthesis projects, use
[render-realism](../render-realism/SKILL.md). Preserve an accepted light/colour baseline when a new structural
detail helps but its accompanying photometric fit regresses.

## Manufacturing structure, thin walls and contact

- Couple albedo, roughness and relief to the same supported physical structure, without
  copying one noise image into every channel. Cloth tension/seams, grip-pad construction,
  print boundaries and handling wear are more informative than uniformly distributed dirt.
- Inspect thin objects separately: wall/rim/base thickness, inner/outer surfaces and nested
  clearance. Test transmission or coating only when reference illumination supports it;
  a bright cavity is not by itself evidence for translucency or a need to darken the whole rig.
- Inspect support, grasp and nesting boundaries in native crops. Check visible gaps, occlusion
  and contact shadows against the visual and collision geometry. A physics pass does not
  establish visual contact; painted AO must not hide a gap or incorrect pose.
- Before adding relief, locate detail loss through the recorded noisy/denoised/processed/decoded
  chain. Stop amplitude increases that change the apparent material class; inspect UV scale,
  procedural regularity and the reflection model instead.
