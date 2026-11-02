# Synthesis stage: from one executed episode to a training dataset

Input: a frozen trajectory-stage candidate (metric scene + robot + an episode executed in MuJoCo with the robot's own
actuators). Output: N randomised episodes, each physically executed and successful, rendered from the robot's
cameras and exported in the robot's data format, plus the evidence that the dataset is valid and diverse.

```
any2sim new <id> --from <traj project>@<cand> --stage synthesis      scene + motions + robot frozen, hashes pinned
synthesis/spec.yaml (agent)                                          ranges, task predicate, acceptance limits
synth_rollout   variant k -> re-anchored commanded motion -> MuJoCo -> success?   (parallel CPU, deterministic)
synth_render    accepted episodes -> head + wrist videos (+ label masks)          (Blender, workers per GPU)
```

## Variants (Scene IR v0.4)
An episode = base scene + one variant overlay (`scene_ir.apply_variant`), so every backend sees the same randomised
scene. The variant stores absolute values (not scales), so an episode is reproducible from (base scene, variant):

| axis | in the variant | used by |
|---|---|---|
| spatial | `entities.<n>.T_world_obj` (initial pose; planar shift + yaw about the world z axis) | all |
| physics | `entities.<n>.physics` (mass, friction), `robots.<id>.overrides.actuators` (kp, kv), `simulation.control_latency_s` | MuJoCo (USD: mass/friction) |
| lighting | `lights` (strength, colour temperature, position, size, world fill) | Blender (USD lights) |
| materials | `entities.<n>.material` (`hsv_shift`, `procedural`, texture swap, roughness) | Blender (MuJoCo: base colour) |
| cameras | `cameras.source_cam.T_delta`, `robots.<id>.camera_deltas.<cam>` | renders, audit |
| sensor | `sensor` (exposure, white balance, blur, noise; seeded) | image post-processing |
| trajectory | `trajectory` (speed, grasp_delta, place_delta, via_offset, place_anchor) | synthkit.transfer |

## Trajectory transfer
Per active hand, the demo's palm path is re-anchored by phase (events from the achieved rollout: first contact,
lift, place, release): unchanged until the arm starts moving, blended (minimum jerk, slerp) onto the object frame by
the pre-grasp (`approach_lead_s` before contact), rigidly attached to the object frame until the lift, blended onto
the target frame by the place, then attached to the target (for containers only the target position:
`place_anchor: position`). Fingers keep the demo's targets. Arm joints by robokit IK (limits, smoothing); IK errors and
joint speeds are acceptance checks. Then the variant's time warp. Physics decides the rest.

## Acceptance (per episode)
Task predicate (`in_container` / `at_pose`: object at rest, released, target upright) + rollout physics: penetration
by category, arm saturation (≤ 5 %), finger saturation (≤ 50 %: fingers may sit at their effort limit while
gripping), joint limits, IK errors, joint speed. Rejections are logged with stage and reason
(`synthesis/attempts.jsonl`, `evidence/synth_rollout.json`), never patched.

## Dataset integration

This release provides episode rollout and rendering. Supply your own dataset schema, export adapter and
independent auditor. Preserve commanded targets separately from achieved states, include provenance, and
verify replay, dimensions, frame alignment, visibility and labels. Missing adapters block dataset acceptance.
