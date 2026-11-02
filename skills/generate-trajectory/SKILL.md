---
name: generate-trajectory
description: Trajectory stage - turn an estimated human wrist/hand trajectory into a physically executed robot trajectory in the reconstructed scene (register to the video, place the robot, retarget, IK, repair contacts in MuJoCo), render it diagnostically in Blender, compare with the real video, export to MuJoCo/Blender/Isaac. Use when request.yaml has stage trajectory.
---

# generate-trajectory

Inputs (request.yaml / observation.yaml): the frozen scene (`scene/`, its evaluation in `upstream/`), the source
video frames (`work/frames`), tracked masks imported from the scene stage (`work/track` etc.), the hand tracks
(`inputs/hand_tracks.npz` + `.yaml`: per side wrist pose, keypoints, optional externally retargeted hand joints,
in their own frame) and a verified RobotBundle (`robot.yaml`: groups, init pose, actuators, contact, retarget map,
cameras, mount notes, checks). The robot bundle is input, not a draft: never edit it.

## What to produce
Scene IR v0.3 (`any2sim.scene_ir` docstring): `robots.<id>` (bundle, bundle_sha256, T_world_base, initial_q,
provenance), `motions` (a `reference` motion, the `commanded` motion you execute, the `achieved` rollout),
`simulation` settings, `render.motion`, entity `trajectory` links. Then the backends: mujoco/scene.xml (mujoco_rollout),
blender renders + video + scene.blend (blender_render frames "all", video true), isaac/ (export_usd), evidence
(video_compare). Motions are `.npz` written with `scene_ir.save_motion` (t = video frame / fps).

## Steps (each checked before the next)
1. **Understand the episode**: look at the frames, the hand masks, the rest states of the objects in the scene, the
   upstream report (known weaknesses: e.g. an object's alignment). Note the time line: when the hand approaches,
   grasps, transports, releases; static ramp frames at the start, if any.
2. **Register the tracks to the world; fix the scene's scale if needed** (`any2sim.handkit`). The hand tracks
   (VR/glove) are metric and accurate: they are the scale reference and are placed **rigidly** (rotation +
   translation, never scaled). A video-only scene's metric scale is a prior-based estimate that pixels cannot
   check (a uniform rescale is invisible in the source video); the tracks can.
   - Events from the tracks themselves (`handkit.track_events`): grasp dwell, transport, release. Use them for
     contacts and timing; do not guess contact frames from the video (the hand may hover for seconds before it
     grips).
   - Registration-free cues (`handkit.scale_cues`): hand transport vs object displacement, grip aperture vs object
     width; registration with contacts (`handkit.register_tracks` with `fit_scene_factor`: glove masks + the grasp
     contact) gives a third. Compare them; explain disagreements (aperture is weak when the grasp is not across
     the object's width).
   - If the agreed factor differs from 1 beyond the rubric tolerance: `scene_rescale` (a declared revision; it
     verifies that no projection changes), then register rigidly again (`fit_scene_factor: false`) and store it in
     `scene.registrations.hand_tracks`. Check `handkit.reprojection` on frames you did not fit: tracked keypoints on
     the gloves, gloves explained by the hands, both hands on image.
   - Never multiply track positions by a scale to make contacts work: that is the symptom of a wrong scene scale
     (or a wrong contact frame), and the robot (fixed metric size) will then look too small or too large.
3. **Place the robot**: T_world_base. If the tracks are expressed in a frame meant as this robot's base (read the
   tracks yaml), that is the natural first guess; check it: the whole reference within reach (IK succeeds with
   margin), mount plausible for the robot (robot.yaml mount notes: e.g. base on the table edge, base below/at the table
   top), no contact with the table at the initial configuration, camera-to-robot geometry plausible.
4. **Retarget the hand** (`hand_retarget`, or your own): palm pose + finger joints per frame, residuals per keypoint;
   if external hand joints are supplied, evaluate them the same way before using them. Fingers of a robot hand are
   not human fingers: thumb opposition and finger lengths differ; prefer contact-relevant points (tips) near grasps.
5. **Arm IK** (`any2sim.robokit`: `Kin.ik_traj` with limits and smoothing): palm targets (world -> base frame) to arm
   joints; check reach, limits, velocities (`robokit.finite_diff` vs URDF limits), elbow flips, collisions with the
   table (`simkit.Sim` contacts at the reference states).
6. **Reference -> commanded**: write the reference motion (palm + fingers, kinematic) and a first commanded motion
   (joint targets for every actuated joint at each video frame). Run `mujoco_rollout`. Read its metrics and preview.
7. **Physical repair** (your code with `simkit`, in-process rollouts): the kinematic reference usually penetrates the
   object or misses it. Typical repairs: approach with clearance and close the fingers onto the object (targets
   beyond contact give grip force; servo + effort limits bound it), adjust the palm pose at the grasp relative to the
   object (search/CEM over a few parameters), lift before transport, open near the observed release frame, keep timing.
   Optimise a few interpretable parameters with whole-episode rollouts; score: object final pose vs the last rest
   state, no drop, penetration, limits, saturation, deviation from the reference, event timing. Never weld, attach,
   teleport or push objects with invisible forces; never change object physics to make a grasp work without evidence
   (mass, friction priors are allowed with provenance).
8. **Execute and register**: final commanded motion -> `mujoco_rollout` (registers the achieved motion, sets
   render.motion and entity trajectories, checks determinism and the model audit).
9. **Render and compare motion**: `blender_render` (frames "all" or a step, video and saved
   blend) and `video_compare`. Check robot/object placement, contact/event correspondence,
   achieved-motion playback and temporal continuity. Record appearance deficiencies without
   claiming realism acceptance. The next independent `realism` stage owns material/light/ISP
   repair and its native multi-camera acceptance; it must preserve the accepted execution.
10. **Export** `export_usd`; validate the Scene IR; freeze a candidate (scene/, mujoco/, blender/, isaac/, evidence)
    and ask the `stage-evaluator` (rubric rubrics/trajectory.yaml, skill evaluate-trajectory).

## Pitfalls
- Metric consistency: the robot is metric; if the scene is not, the rendered robot hand is too small/large next to
  the real hand (video_compare robot_hand_scale). Hand precision alone does not show it.
- The first frames may be a static ramp (repeated frame 0); the hands do not move there.
- Keypoint layouts differ between sources (OpenPose vs MANO chain order); `hand_retarget` detects it, check the
  reported layout.
- Hand masks from the scene stage may include sleeves, controllers or trackers worn on the hand; they are larger than
  the robot hand: use precision (robot inside human region) more than IoU.
- Contact solver: timestep 2 ms; stiff fingers with tiny inertia need armature (the bundle sets it); check
  determinism and that the object does not jitter at rest before and after the episode.
- The scene's object geometry may be imperfect (upstream report): grasps that depend on exact shape need margin.
- Robot links can hit the table or the basket rim during transport even when the hand does not: check all contacts.
