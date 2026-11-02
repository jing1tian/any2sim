---
name: isaac-scene-policy
description: Export reconstructed robot scenes and commanded trajectories to native Isaac Sim, diagnose MuJoCo-to-PhysX replay differences, and expose policy rollout/evaluation including OpenPI WebSocket inference. Use for simulation migration and evaluation, not Blender-only dataset rendering.
---

# Reconstructed scenes to Isaac policy evaluation

Build on the frozen scene, RobotBundle and commanded trajectory. Distinguish visual playback, native physics execution and policy evaluation: success in one does not validate the others. Read the project's observation permissions and upstream acceptance reports before accessing references. Keep project-specific adapters in project code; record declared revisions instead of silently modifying upstream assets.

## Import contract

Record units, world up axis, transform direction, quaternion order, robot root pose and joint-name ordering. USD angular attributes may use degrees while runtime articulation actions use radians. Match joints by name; do not rely on USD traversal or PhysX DOF order. Check reset joint values, palm FK, object support heights and camera extrinsics before testing contact. Validate quaternion conversion explicitly. Never scale measured robot joint/hand trajectories to repair a scene-registration error.

Export portable USD dependencies, including collision partitions, textures and sublayers. Verify that copied runtime assets resolve outside the development workspace. A visual animation layer that drives object transforms or joints is unsuitable as the physics/policy execution layer: remove active time samples and kinematic playback for dynamic bodies. Keep state setters limited to reset; objects must move through native contact during rollout.

Transfer masses, centers of mass, inertia, armature, joint limits, actuator effort limits, PD gains, passive damping, gravity treatment, collision exclusions and contact materials. Verify authored properties on the **active collision shapes after decomposition/partitioning**, not just the original mesh. See [physics-transfer.md](references/physics-transfer.md) for differential tests and controller pitfalls.

## Runtime and evaluation

Expose reset/step/observe/evaluate/close with declared action shape, order, units and timing. For ordinary policy stepping, hold each action over the physics substeps. For reconstructed-command diagnostics, match the source's interpolation at each physics step; logging only the final substep target cannot reproduce an interpolated schedule. Record exact substep targets when needed.

Separate requested, clipped/applied and achieved actions/states. Evaluate from achieved object poses and recorded contacts; do not use reference object animation as task ground truth. Preserve the source success predicate and report penetration, joint-limit and torque/saturation checks separately. A successful task is not a full physics certification. Sample several generated trajectories with their own variant poses, sizes and physics; never silently omit unsupported variant axes. Report sample scope instead of extrapolating a few tests to an entire dataset.

Optional randomization should be applied at reset, use reproducible seeds, and leave the frozen base scene unchanged. Validate support, overlap and limits. Placement validity alone does not guarantee reachability or task completion. Separate spatial and lighting RNG streams so toggling visual randomization does not change physical placements.

For remote policies read [policy-websocket.md](references/policy-websocket.md). Check state/image fields and action semantics against the actual trained server configuration before a benchmark. Do not treat a mock-hold smoke run as policy performance.

## Cameras and graphics

Attach wrist cameras to live articulated links, and verify first-frame head framing against the intended robot reference. Preserve calibrated camera conventions. Render observations without advancing simulation time; test this by comparing time and joint state before/after capture. Verify image channel order, resolution and moving-camera updates over multiple resets.

A green or flat-looking imported scene can come from material color-space, UV, normal/roughness repeat, missing specular mapping, renderer or exposure differences. Inspect those separately from dynamics. Set renderer and tone mapping after stage initialization if the stage overwrites settings. Cycles sample count and Isaac RTX sample count do not imply matching images. Distinguish raw rendered RGB from calibrated sensor ISP.

For headless startup, separate missing userspace libraries (e.g. libXfixes) from missing NVIDIA graphics driver mounts. Setting driver-capability variables inside an already-running container does not mount host libraries. Use an import/graphics preflight and explicit failure logs; a shutdown routine can obscure the process exit status. Put large texture caches on appropriate storage. Start with state-only checks, then a small RGB smoke run, before expensive episode rendering.

## Evidence to retain

Keep config, source hashes, reset state, joint names, applied actions, achieved states, object poses, contact history, seeds, task metrics and failure artifacts. Retain failed episodes and label optional diagnostic controllers; do not silently replace the default with a candidate that rescued one grasp. Record the runtime/controller mode in policy reports. Use independent evaluation when the repository workflow requires it.
