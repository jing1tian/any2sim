# any2sim — agent guide

You turn observations (a video at minimum) into a digital twin: a Scene IR (`projects/<id>/scene/scene.yaml`)
with correct geometry, clean physically-lit materials and physics, robots and physically executed motions, compiled
to MuJoCo, Blender and Isaac (USD). You are the builder: you decide the approach, write the task-specific code, run
tools, look at the results, and hand frozen candidates to an independent evaluator. Design reference:
`docs/real2sim_architecture_summary.md`.

Stages (request.yaml `stage`): **scene** (reconstruct the scene), **trajectory** (a robot executes the demonstrated
task in that scene; starts from a frozen scene result, `observation.yaml upstream`), **realism** (refine and independently review appearance while preserving the
accepted trajectory and physics), **synthesis** (diversify the
executed episode into a randomised, physically verified, rendered training dataset; starts from a reviewed realism
result), **robot** (onboard an embodiment as a verified RobotBundle). A project can start at any stage from another
project's result (`any2sim new <id> --from <project>[@<candidate>] --stage trajectory|realism|synthesis ...`).

## What you have
1. **General tools** (MCP server `any2sim`: `list_tools`, `describe_tool`, `run_tool`, `job_status`, `job_cancel`).
   They run outside your sandbox on the GPUs. Read `describe_tool` before first use. Each run writes a receipt.
   | Tool | Capability |
   |---|---|
   | ingest_video | video -> frames + ObservationBundle |
   | sam3_segment | text-prompted instance masks on frames or any image (also: find artifacts in renders) |
   | sam2_track | propagate seed masks through the video with identities |
   | moge_geometry | per-image metric point map, depth, normals, intrinsics |
   | pi3x_geometry | multi-frame camera poses, intrinsics, point maps |
   | sam3d_generate | image + mask (+ point map) -> textured mesh draft in that camera |
   | render_views | fast geometry renders (ids, depth, normals, outline overlays) from any camera |
   | fit_render | render-and-compare fitting: pose / scale / vertex offsets, or the parameters of YOUR `build(params)` shape code, to masks (+ interior edges, depth) |
   | project_texture | per-texel observed colours from the video for a UV-mapped mesh (visibility-tested) |
   | synthesize_texture | clean tileable texture from an exemplar region (image quilting) |
   | convex_decompose | CoACD collision hulls + visual/collision distance |
   | blender_bake | per-texel irradiance / position / normal / coverage maps under the IR lights |
   | blender_render | Cycles renders of the IR (robots + render.motion included): source frames / all frames + video, background-only, novel views, turntables; scene.blend |
   | mujoco_check | IR -> MJCF; penetration, settling, settled poses, cavity probes |
   | geometry_metrics | IR vs video over all frames: IoU, outline error, interior edge alignment, depth |
   | image_metrics | masked PSNR / SSIM / LPIPS per region |
   | media_frames | frames of any extra video (other views, robot cameras), optional undistortion |
   | robot_import | URDF -> RobotBundle (groups, profile, MJCF), load/self-contact checks, preview |
   | robot_visuals | attach realistic per-link visuals (USD/GLB/OBJ/blend), align frames, derive collisions |
   | robot_log_check | FK vs logged EE poses (link/frame found), joint ranges, velocities |
   | robot_fit_camera | robot-mounted camera pose (+focal) from images, robot masks and joint logs |
   | robot_overlay | robot rendered at logged joints over real images (side by side / blend / outline) |
   | hand_retarget | human hand keypoints (+wrist) -> robot palm pose + finger joints, residuals, layout detection |
   | mujoco_rollout | execute a commanded motion with the robot's actuators; achieved motion, contacts, metrics, preview |
   | video_compare | achieved episode vs the video: robot hand vs human hand, object overlap, static SSIM, events |
   | export_usd | Scene IR -> Isaac USD (articulations, drives, rigid bodies, materials, camera) + playback layer |
   | scene_rescale | uniform rescale of the Scene IR (declared revision), verifies projections are unchanged |
   | camera_calibrate | camera image model (tone, vignetting, softness, noise, bitrate) from renders vs the real video; held-out validated |
   | realism_render | direct achieved-motion nominal videos + noisy/denoised/processed/decoded checkpoints and light passes |
   | realism_check | protect execution, freeze protocol, measure evidence, adopt independent review and gate synthesis |
   | realism_metrics | real vs rendered crops: KID and classifier AUC on DINOv2 features, low-level image statistics |
   | robot_cables | fit robot cables/wires (anchors in links, length/sag, loops) to cable masks on frames with joint logs |
   | robot_appearance | per-part robot materials (or textures) from de-lit real observations with joint logs |
   | synth_rollout | sample scene variants, re-anchor the demo (IK), simulate, keep successful episodes (parallel CPU) |
   | synth_render | Cycles multi-camera renders of episodes (head + wrist cams, variant lights/materials/sensor), label masks |
2. **Shared library** (import from your scripts with `envs/base/bin/python`): `any2sim.scene_ir` (schema, load/save,
   `validate`, motions: `save_motion`/`load_motion`/`motion_at`), `any2sim.meshkit` (lathe, booleans, arrays, UV
   unwrap, OBJ export), `any2sim.geom3d` (transforms, projection, planes, conventions), `any2sim.common` (frames,
   masks, images, contact sheets), `any2sim.robokit` (robot FK, Jacobians, IK and trajectory IK with limits),
   `any2sim.simkit` (in-process MuJoCo rollouts of the scene with robots for your optimisers: `Sim(project).rollout`),
   `any2sim.handkit` (metric hand tracks: events, scale cues, rigid registration, reprojection vs glove masks),
   `any2sim.synthkit` (synthesis: demo events, variant sampling from synthesis/spec.yaml, object-centric trajectory
   transfer, success predicates, one attempt end to end), `any2sim.camera_isp` (camera image model applied to renders),
   `any2sim.robot.cables` (cable centrelines/tubes from link poses),
   `any2sim.robot.bundle` / `robot.datalog` (RobotBundle schema, log tables, rotation formats).
3. **Your own code**: everything specific to the scene (world frame, background/static structure, object models,
   rest states, material choices, de-lighting, physics fixes) goes in `projects/<id>/work/code/*.py`, written by you.
   Reference recipes from an earlier hand-built run are in `recipes/` (ideas and pitfalls, not tools).
4. **Skills**: `skills/real2sim` (how to run a project), scene: `skills/reconstruct-scene`,
   `skills/refine-appearance`, `skills/evaluate-scene`; trajectory: `skills/generate-trajectory`,
   `skills/evaluate-trajectory`; synthesis: `skills/synthesize-data`, `skills/evaluate-dataset`; rendering realism
   (independent stage after trajectory): `skills/render-realism`, `skills/evaluate-realism`; robot:
   `skills/onboard-robot`, `skills/evaluate-robot`. Isaac scene/trajectory migration and native policy evaluation: `skills/isaac-scene-policy`. **Rubrics**: `rubrics/scene.yaml`, `rubrics/trajectory.yaml`,
   `rubrics/realism.yaml`, `rubrics/synthesis.yaml`, `rubrics/robot.yaml` (the request names the one that applies).
5. **Robots**: verified bundles in `assets/robots/<name>/` (referenced as `assets:<name>`); read robot.yaml (groups,
   init pose, actuators, contact, retarget map, cameras, checks) before using one.

## Rules
- The Scene IR is the source of truth; `.blend`, MJCF and USD are compiled from it. Validate it with
  `envs/base/bin/python -c "from any2sim.scene_ir import validate; print(validate('projects/<id>/scene'))"`.
- Respect `observation.yaml`: never read withheld inputs (calibration, trajectories, other views), even if they sit
  next to the video.
- Every value in the IR says where it came from: observed / fitted / completed / prior / generated.
- Generated geometry (SAM 3D) is a draft. The deliverable must match the real object's shape and structure.
- Materials must be clean: no baked shadows, hands, equipment, inpainting blobs, seams or smears. Shadows come from
  the lights, not from textures. Never ship raw video pixels as a surface texture without proving they are clean.
- Do not tune the evaluation (masks, thresholds, frames) to pass. The builder never judges its own candidate.
- Robots move only through their actuators; objects move only through contact. No welds, attachments, teleports,
  mocap or applied forces to make a task succeed. A RobotBundle you were given is input: never edit it.
- An upstream (imported) scene is frozen; revise it only as a declared revision (`scene/revisions.yaml`) with evidence.
- Metric observations (hand tracks, robots) are never scaled; when they disagree with the scene, the scene is rescaled.
- Synthesis: randomise only through the variant overlay (synthesis/spec.yaml -> variant.yaml); the base scene and the
  demo stay frozen. Every exported episode is physically executed and succeeds; failures are rejected and reported,
  never patched (no relaxed predicates per episode, no object teleports between phases).
- Shared machine: tools have an RSS cap; keep jobs reasonable, cancel what you no longer need.
- Tokens are monitored: read compact outputs (`summary.json`, contact sheets) instead of dumping large files; look
  at images that answer a question.

## Conventions
Right-handed, metres, world z up. `T_a_b` maps points from frame b into frame a. Cameras: OpenCV optical frame.
Quaternions wxyz. Object meshes: object frame z up, origin at the bottom centre. Motions: t = video frame / fps.

Public release: dataset export and audit integrations are not bundled. Supply and independently validate them
before claiming a completed synthesis dataset; missing adapters must never be treated as a passing evaluation.
