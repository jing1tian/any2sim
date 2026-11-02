---
name: reconstruct-scene
description: Reconstruct the geometry of a scene from video - cameras, metric scale, world frame, static structure, every interactive object with a shape that matches the real one, rest-state poses, collision and physics - into the Scene IR. Use when a project has observations but no accepted scene geometry.
---

# reconstruct-scene

## Goal
A Scene IR whose geometry matches the real scene: cameras and metric scale with stated uncertainty, a world frame
with gravity along -z, static structure (supports, background), and every interactive object with the **real
object's shape and structure** (proportions, openings, slots, rims, handles, legs, lids), posed in every rest state,
with collision geometry and physics that hold.

## Requirements (checked by rubrics/scene.yaml)
- Object silhouettes match the video on all frames of every rest state (not only the frames you fitted).
- Interior structure matches: compare close-up crops of real frames with renders; count and place salient parts
  (slots, holes, handles, lugs, rims, wall thickness). `geometry_metrics` reports interior edge alignment; you must
  also look.
- Sizes are plausible for the object category and consistent with the scale cues; scale uncertainty is stated.
- Hidden parts are completed plausibly and labelled `completed`; generated parts are labelled `generated`.
- Rest states are physically consistent: no penetration, stable when simulated (`mujoco_check`), objects in
  containers stay inside, containers keep their cavity in collision.

## Tools that help (see `describe_tool`)
cameras/scale: pi3x_geometry, moge_geometry · masks: sam3_segment, sam2_track · object drafts: sam3d_generate ·
checking: render_views, geometry_metrics · fitting: fit_render (gradient: pose/scale/offsets; builder: your shape
code) · collision: convex_decompose · physics: mujoco_check. Library: meshkit, geom3d, scene_ir.

## Strategies that work
- **Cameras and scale first.** Learned multi-frame poses (Pi3X) and single-image metric depth (MoGe-2) are
  complementary; cross-check them (feature matches/PnP, plane fits) and compare independent scale cues (several
  frames, object size priors, known-size items in view). Reject a cue only with a recorded reason.
- **World frame** from the dominant support (fit a plane to static support pixels; z up).
- **Masks:** prompt objects and every dynamic distractor (hands, gloves, arms, tools, trackers, straps, cables);
  check that distractor masks do not swallow objects.
- **Objects: draft, inspect, rebuild.** Start from a SAM 3D draft (with the metric point map), then compare it with
  the video at several frames (render_views overlays, close-up crops). If proportions or structure are wrong,
  write a parametric builder (`work/code/<object>.py`, `build(params) -> Trimesh` with meshkit: lathe profiles,
  shells, slot/hole arrays via booleans, handles) and fit its parameters and pose with `fit_render` builder mode
  (masks + `w_edges` for interior structure). Use gradient mode with vertex offsets for organic shapes (produce,
  cloth-free deformable-looking rigid objects). Keep the better of draft and rebuild by the metrics *and* by eye.
- **Rest states:** find intervals where each object is static and untouched (masks + world-frame anchors); fit one
  pose per state with the shape shared across states; resting objects are planar (yaw, x, y on the support).
- **Physics:** decompose collision (low concavity threshold for containers), check with `mujoco_check`; fix the
  cause (geometry, support, pose) rather than hiding it; keep fitted poses unless a settled pose stays within the
  silhouette tolerance.

## Pitfalls seen before (recipes/v0_pipeline/README.md)
N x 3 SVD memory blow-up; SAM 3D export axes; meshes with openings vs solid masks; single-plane homography
ambiguity; camera auto white balance drift; duplicate frames in static clips (use distinct frames for held-out).

## Metric scale
A uniform rescale of the whole scene is invisible in the source video, so no pixel metric can confirm the scale.
Use metric observations when they exist (hand tracks, robot geometry, calibration, a measured object) as scale cues;
category priors are a fallback with a stated uncertainty. A later stage with metric tracks may rescale the scene
(`scene_rescale`, declared in scene/revisions.yaml).
