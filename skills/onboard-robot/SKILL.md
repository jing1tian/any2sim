---
name: onboard-robot
description: Integrate an embodiment as a verified RobotBundle from URDF/MJCF, real images/videos, joint and EE logs, calibration and visual assets; refine an existing bundle's robot materials and markings against real observations. Use for robot onboarding or robot appearance refinement, including an existing robot used by trajectory or synthesis projects.
---

# onboard-robot

The bundle (`robot/`: robot.yaml + robot.urdf + robot.xml + visual/ + collision/ + checks/) is what every later
stage trusts: kinematics, joint conventions, frames, tool/hand mounting, init pose, actuation, contacts, cameras and
appearance must agree with the real robot. Schema: `any2sim.robot.bundle` docstring.

For an existing bundle's material-only revision, use **Robot material refinement** below and the applicable
validation steps; preserve accepted kinematics and calibration instead of repeating onboarding.

## Evidence -> checks it enables
| Supplied | Check | Tools |
|---|---|---|
| description only | loads, masses, limits, rest self-contact, preview | robot_import |
| visual assets (covers, gloves, skins: USD/GLB/OBJ/blend) | per-link alignment to the description's own visuals; frame conventions | robot_visuals (align: reference description, or ICP on the bare part) |
| joint logs + EE logs | FK vs logged EE (which link, which frame), joint ranges, velocities | robot_log_check |
| images/videos + joint logs (same clock) | camera extrinsics (+focal), overlays: mounting, init pose, joint signs | media_frames, sam3_segment, robot_fit_camera, robot_overlay |
| calibration files | intrinsics/extrinsics directly (then overlays verify them) | robot_overlay |
| photos of the robot | appearance (materials per link, covers), missing visual parts | your judgement + blender renders |
| images + joint logs (with masks) | cables/wires routed between links (wrist leads, service loops, harnesses) | sam3_segment ("black cable"), robot_cables |
| images + joint logs (with masks) | real per-part colours of the links (white shells, metal plates, dark rings), de-lit | robot_appearance |
| videos of the robot cameras | camera image models (noise, softness, bitrate) of robot-mounted cameras | camera_calibrate for source_cam; own paired-reference workflow for other cameras (see render-realism) |
| hints (home, gains, open/closed) | named poses, actuators (with provenance) | robot_import params |
Without logs or images, kinematics and conventions stay `not_run` in checks: say what would verify them.

## Steps
1. `robot_import` (any description): groups are detected from the tree (arms = serial chains, hands = short
   branching subtrees); fix names/kinds if needed; rest self-contacts between near links are excluded automatically
   (listed in the result: check they are mesh artefacts, not real interference).
2. Visuals: if the description's meshes are bare (skeleton hand, missing covers) and realistic assets exist, attach
   them with `robot_visuals`; if the asset comes with its own description (another URDF/MJCF of the same hand),
   align by `reference` (exact); otherwise ICP on the bare part shipped with the asset. Every link must stay within
   ~1 mm of the description's own mesh (P95), except the intended cover thickness. Derive collisions from the covers
   when they are what touches the world (gloves).
3. Kinematics: `robot_log_check` with every EE log column; the best link/frame and residuals (sub-mm to ~1 mm for a
   consistent model). Large residuals: wrong joint order/sign/offset or wrong link lengths; test hypotheses.
4. Cameras and overlays: extract frames (`media_frames`, undistort with the supplied intrinsics), robot masks
   (`sam3_segment`: e.g. "robot arm", "black glove"; check them), `robot_fit_camera` (search starts around the
   expected mount; use `bodies` to fit on the parts the masks cover; a camera rigid to a tool only verifies that
   tool's joints, not its mounting). Then `robot_overlay` side by side at many log rows: the rendered robot must
   look like the real one in pose (fingers too) and appearance.
5. Conventions under doubt: compare hypotheses (sign flips, swapped joints, offsets) by refitting and comparing
   IoU / residuals; record the comparison.
6. Realism (renders must look like the real robot, not CAD):
   - Cables and wires: segment them (`sam3_segment` "black cable" on the frames with logged joints), map the parts
     they attach to (overlay links on a frame), then `robot_cables` with anchors picked as pixels where a cable
     visibly leaves a part ("surface": "far" for cables running behind a part), a length (sag) or a fitted slack
     direction for service loops, loops for coils around a joint axle. Check the overlay (bright = visible, dark =
     behind the robot) on frames you did not pick anchors from; twin arms reuse a fit only with that provenance.
   - Materials and markings: follow **Robot material refinement** below. Use `robot_appearance` (mode `parts`)
     for de-lit per-part colour estimates with robot masks and cables ignored; use texel fitting only when
     registration is accurate. Validate the semantic assignments and rendered finish against real observations.
   - Appearance-only changes keep the physics digest (scene_ir.physics_digest): publishing records the replaced
     revision, so scenes built on the earlier bundle stay valid.
7. Profile decisions with provenance: init_pose (first/rest frame of real logs), actuators (servo gains, effort and
   velocity limits), contact (materials of covers), appearance (per-link materials for meshes without textures),
   retarget map for hands (semantic keypoints wrist / <finger>_0..3 -> links), data_mapping (log columns -> joints,
   EE frames), cameras, mount notes, checks (status, metric, evidence file).
8. `any2sim.robot.bundle.validate`, freeze, evaluate (rubrics/robot.yaml, skill evaluate-robot). Publishing to
   `assets/robots/<name>` is the user's decision.

## Robot material refinement

### Scope and reference views

- Record the permitted use of any supplemental robot recording. A different task can support robot finish and
  logged-pose/view comparisons without supplying the target scene's illumination, props or trajectory. Keep its
  diagnostic lighting separate from the target rig, and preserve restrictions on other reference inputs.
- Work in a candidate copied from the accepted bundle; keep the supplied bundle immutable. For material-only
  work, preserve geometry, hand assets, dynamics and virtual camera calibration. Add or remove camera housings,
  fixtures and cables only within the requested scope. Removing a visible camera mesh does not imply changing
  the corresponding virtual sensor or its calibration.
- When joint logs are available, align video frames with their logged poses; otherwise use the supplied pose/view
  evidence and record registration uncertainty. Inspect the actual downstream camera (often the head view),
  close crops of each relevant surface, and other poses that expose its sides. Separate fitting frames from
  verification frames. A single view can conceal missing graphics, wrong material regions or UV clipping.
- Diagnose geometry/registration, albedo, surface response, lighting and camera exposure separately. Keep one
  fixed lighting rig and camera response across before/after renders. Also inspect the adopted scene's lighting;
  a neutral robot-only preview does not establish the appearance in the synthesis scene.

### Surface finish

- Assign materials by surface construction: painted coating, exposed metal, dark joint housing, rubber and fabric.
  A grey cluster from video may be shaded white paint; it is not evidence of high metallicity. Check the spatial
  assignments from `robot_appearance` before retaining or remapping its colour classes.
- Tune base colour, roughness and metallic response together. Painted metal has a dielectric coating; satin metal
  needs broad highlights rather than a dark chrome response. Inspect highlight width and shadow-side brightness
  over several poses. PBR values chosen visually are estimates, not measured BRDFs or calibrated albedos.
- Do not fit white albedo to clipped/overexposed video pixels or copy exposure gradients into textures. Use
  de-lit observations where supported, otherwise clean constant/procedural surfaces. Preserve measured glove
  textures and normal maps unless they are part of the requested change; retain hard mechanical edges when
  adjusting shading normals. Fine surface variation should be supported by the reference, not invented wear.

### Gloves and small surface relief

- Check every relevant part, including palms and finger segments, by material role rather than assuming mesh
  names or material indices are uniform. Distinguish fabric, rubber grips, plastic housings and exposed metal.
  Fabric sheen/roughness, grip highlight width and weave normals affect black-surface readability differently;
  lifting all black albedos can merely turn the glove grey. Check both lit and backlit views in the target scene.
- Isolate grip-dot and fabric/backing edits before combining them: reducing rubber highlights can leave broad
  backing reflections unchanged. Compare each role against the accepted bundle and real reference at matched
  poses. A nearly static reference clip provides little evidence about highlight motion through a changing grasp;
  label that limitation even when the target motion excerpt is temporally stable.
- Inspect the actual mesh cross-section when grip dots look like beads. Excess geometric relief is not fixed by
  colour or roughness alone. A requested visual-shape refinement is a separate change from material-only work:
  measure displacement against the accepted mesh and its contact surfaces, and record the allowed mismatch.
  Small displacement or an unchanged physics digest alone does not prove contact fidelity. If the changed
  surface materially alters contact, revise collision geometry through a declared physics change and revalidate.
- After a visual mesh edit, update shading normals and exported bounds, retain UVs/markings, and synchronize the
  corresponding visual meshes used by other backends. Compare GLB and MuJoCo OBJ geometry in the same link frame,
  accounting for axis/unit conversions. Otherwise image/label comparisons may use different hand shapes even
  though both files load. Hash untouched arm/logo assets to verify that a glove edit preserved them.

Use [render-realism](../render-realism/SKILL.md) for output-camera selection and independent per-camera image
models; verify the downstream view before attributing a framing or exposure difference to the robot material.

### Logos, lettering and equipment labels

- Treat the symbol, wordmark and equipment labels as distinct observations. Inspect multiple frames before
  identifying a blurred mark: for example, a top-cover mark may be an equipment tag rather than a repeated logo.
  Leave unreadable content unresolved rather than inventing text or turning it into blurred texture blobs.
- Prefer an existing authentic asset or the manufacturer's vector artwork, checking that its version matches the
  recorded hardware. Retain the full symbol and letter paths, case, spacing and proportions. A readable brand
  name in a generic font is not a matching logo. Record the source URL/file, content hash and any transformations;
  if authentic artwork is unavailable, label reconstruction uncertainty explicitly.
- Build a clean monochrome/colour stencil on the surface material. Preserve SVG transforms, holes and clipping
  when rasterizing. Do not paste the source frame's white background, shadows, reflections or JPEG noise into
  the albedo; separately inspect the artwork and the rendered result.
- Fit size, position and rotation in the link's surface coordinates using the logged pose and camera projection.
  Use observed cover edges/fasteners to distinguish decal placement from camera or mesh registration error.
  Report apparent video-fitted dimensions as such; do not call them factory measurements. Record symmetric
  completion on an unobserved arm or opposite face as a prior.
- Check the actual exterior faces and exported UVs. A narrow face-selection rule can omit part or all of the
  marking; opposite faces can mirror the text. Verify the complete symbol and every letter over multiple poses,
  including curved boundaries, without moving geometry or changing calibration to hide a decal error.

### Verification and adoption

- Produce real / previous / candidate comparisons at the same log rows, crop coordinates and image scale, with
  fixed simulated lighting/exposure. Show a close-up large enough to judge symbol, letter shapes, spacing, size,
  position and angle as well as an overall robot view. Label rectified or warped diagnostic images separately;
  they must not masquerade as unmodified render agreement.
- Freeze the candidate and have the independent evaluator compare it with real observations on frames not used
  for fitting. Require evidence for the requested visual match: “text is readable”, a physics pass or a plausible
  material alone does not establish likeness. Keep a limited material/decal acceptance separate from full robot
  certification, and report remaining lighting, registration and unresolved-label limitations.
- For a material-only change, verify unchanged `physics_digest`, camera parameters, URDF, collision/inertia and
  link geometry where re-exported; use representative or logged FK checks when transforms were touched. Camera
  extrinsics/intrinsics changes affect `physics_digest` and require a separately declared calibration revision.
  `bundle_digest` hashes only `robot.yaml` and `robot.urdf`: also hash the referenced meshes, textures and GLBs so
  a texture-only edit cannot silently invalidate a frozen review.
- After acceptance, update the consuming Scene IR's bundle reference/hash through a declared revision, rebuild
  its compiled assets and regenerate the affected camera previews. Verify texture/material support in the
  downstream renderer; record any backend that retains only flat colours. Preserve old renders but mark their
  manifests stale, and distinguish updated still previews from a newly rendered full video. Include the adopted
  scene/Blender artifact and the real-versus-render comparison in the handoff.


For a downstream realism-stage revision, retain `realism/protected.json` invariants.
Judge glove backing, grip pads, paint and exposed metal in separate native crops through
noisy/denoised/processed/decoded outputs before further material-strength trials. Record
ineffective hypotheses and return geometry/contact changes to the appropriate stage;
material-only permission does not justify reshaping fingers to improve the silhouette.
