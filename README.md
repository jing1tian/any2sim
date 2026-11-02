# any2sim

Video (+ optional calibration, trajectories, robot description) → versioned digital twin, built by a coding agent
working through stage skills and judged by an independent evaluator. Architecture: `docs/real2sim_architecture_summary.md`.

Implemented: **scene and object reconstruction** + **photo-realistic rendering in Blender** from a video alone;
**robot trajectory generation**: an estimated human wrist/hand trajectory turned into a robot motion executed in
MuJoCo physics in that scene, rendered in Blender, compared with the real video and exported to Isaac (USD);
**independent realism refinement** after trajectory acceptance, with protected execution, diagnostic renders and review gates;
**data synthesis**: the executed episode diversified into a randomised (placement, lighting, materials, physics,
camera mounts, sensor, trajectory), physically verified, multi-camera rendered dataset in the robot's format;
**robot onboarding** of any embodiment as a verified RobotBundle. One Scene IR (v0.4: entities, robots, motions,
variant overlays) compiles to MuJoCo, Blender and USD.

## How it works

A coding agent (Codex or Claude Code, `any2sim run [--agent claude]`) is the builder: it reads `projects/<id>/request.yaml`, the skills and the
rubric, and decides how to reconstruct the scene. It has **general tools** (MCP server `any2sim`, run outside its
sandbox on the GPUs; see AGENTS.md for the catalog), a **shared library** (`any2sim.scene_ir`, `meshkit`, `geom3d`,
`common`, `robokit`, `simkit`, `handkit`, `synthkit`) and writes **all task-specific code itself** in `projects/<id>/work/code/`. An
independent `stage-evaluator` sub-agent judges frozen candidates against the stage's rubric (`rubrics/scene.yaml`,
`trajectory.yaml`, `realism.yaml`, `synthesis.yaml`, `robot.yaml` - every criterion separately).

`recipes/v0_pipeline/` keeps the first hand-built pipeline as reference (not tools).

Sessions record the full trace and token usage in `projects/<id>/sessions/` (events, rollouts of the main thread
and sub-agents, per-call token usage, summary, plot): `any2sim usage <project>`.

This public source release includes generic reconstruction, robot onboarding, simulation and rendering.
Dataset export and audit adapters, replay mappings, robot assets and private experiment fixtures are not included.
Synthesis evaluation requires a user-supplied exporter and auditor before a dataset can pass review.

## Setup (server)

```bash
EGO2SIM_WORKSPACE=/path/to/source/workspace bash environments/copy_from_ego2sim.sh # envs (vision, sam3d, render, cuda-12.1), models, robot assets
bash environments/install_base.sh           # envs/base: any2sim CLI, MuJoCo, CoACD
bash environments/install_geom.sh           # envs/geom: Pi3X + MoGe-2 (+ weights via hf-mirror, parallel fetch)
bash environments/install_extras.sh         # LPIPS in envs/sam3d
```

`envs/`, `models/` (HF cache, torch hub, model sources) and `assets/` (robots) are git-ignored.
The copy script requires an existing compatible workspace; environments, weights and robot assets are not included.
Set private machine paths and proxy settings in `configs/any2sim.local.yaml` (ignored by Git).
Installation scripts inherit proxy environment variables. Never commit credentials or local runtime data.

## Run

```bash
envs/base/bin/any2sim new <id> --video demo.mp4          # project + frames + observation.yaml
cp templates/request_scene.yaml projects/<id>/request.yaml # edit goals / budgets
envs/base/bin/any2sim run projects/<id> --model gpt-6-astra --effort medium --max-hours 8
envs/base/bin/any2sim run projects/<id> --agent claude --effort high    # same with Claude Code (default claude-opus-5-5)
envs/base/bin/any2sim usage projects/<id>                  # token usage so far (live during a run)
envs/base/bin/any2sim tool projects/<id> <tool> '<json>'   # run a tool by hand
```

### Trajectory stage (resume from a scene result)

```bash
envs/base/bin/any2sim new <id> --from <scene project>[@<candidate>] --stage trajectory \
    --tracks <per-frame table> --tracks-map configs/tracks/<source>.yaml \
    --robot assets:my_robot --import-work track distractors --task "Put the pepper in the basket."
envs/base/bin/any2sim run projects/<id> --model gpt-6-astra --effort medium     # request.yaml is written for the stage
envs/base/bin/any2sim run projects/<id> --resume                                # continue an interrupted session
envs/base/bin/any2sim run projects/<id> --agent claude --resume                 # (of that agent)
```
The scene, frames, chosen work folders and the upstream evaluation are copied and pinned by hash; the tracks are
converted to `inputs/hand_tracks.npz` (canonical: wrist pose, keypoints, optional retargeted hand joints).

### Realism stage (after trajectory reconstruction)

```bash
envs/base/bin/any2sim new <appearance-id> --from <trajectory-project>@<candidate> --stage realism
envs/base/bin/any2sim run projects/<appearance-id>
```

The independent stage protects executed motion/physics, captures image-processing and lighting
comparisons, validates each camera and requires an independent review. See
[the executable stage contract](docs/realism_stage.md). Legacy reviews require re-evaluation.

### Synthesis stage (resume from a reviewed realism result)

```bash
envs/base/bin/any2sim new <id> --from <realism project>[@<candidate>] --stage synthesis --import-work track \
    --task "Put the pepper in the basket."
envs/base/bin/any2sim run projects/<id> --model gpt-6-astra --effort medium
```
The agent writes `synthesis/spec.yaml` (ranges) and runs synth_rollout (variants -> object-centric re-anchoring + IK
-> MuJoCo -> keep successes), synth_render (head + wrist cameras, EEVEE or Cycles), a user-supplied dataset exporter and independent auditor. See `docs/synthesis.md`.
Rendering realism (robot cables and materials from real frames, camera model from the source video, render-only
dressing, motion blur, realism metrics): `docs/realism.md`, skill `skills/render-realism`.
The required standard render workflow (fixed processing, nominal review, batch gate and stale-render checks)
is documented in `docs/render_pipeline.md`. Spatial/lighting-only randomization still uses fixed realism processing.

### Robots

Supply your own verified RobotBundle in `assets/robots/<name>` (`assets:<name>`).
Onboarding another embodiment: a project with request stage `robot` (`templates/request_robot.yaml`, skill
`skills/onboard-robot`), tools robot_import / robot_visuals / robot_log_check / robot_fit_camera / robot_overlay,
then `envs/base/bin/any2sim publish-robot projects/<id>`.

See `skills/reconstruct-scene/SKILL.md` and `skills/refine-appearance/SKILL.md` for the step order and checks,
and `docs/pepper_basket_video.md` for a generic video-input example.
Long tool runs: `bash environments/run_chain.sh <project> <name> tool='{json}' ...` (detached, survives SSH drops).
Every tool runs under an RSS cap (`max_rss_gb`, default 48 GB): this is a shared machine.
