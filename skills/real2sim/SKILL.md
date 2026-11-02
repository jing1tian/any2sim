---
name: real2sim
description: >-
  Run an any2sim project end to end (any stage: scene, trajectory, realism, synthesis, robot) - read the request, plan, build with general tools and your own code, freeze candidates, get independent evaluations, revise until the acceptance criteria pass or the budget ends. Use at the start of every project session.
---

# real2sim: running a project

## Start
1. Read `projects/<id>/request.yaml` (stage, goals, inputs, acceptance, budgets) and `observation.yaml` (what was
   given, what was withheld, `upstream` results this stage starts from). Read `AGENTS.md`, then the stage skill:
   scene -> reconstruct-scene + refine-appearance; trajectory -> generate-trajectory; realism -> render-realism + refine-appearance; synthesis -> synthesize-data;
   robot -> onboard-robot.
   A project resumed from another stage (`upstream`) starts from that frozen result: read its report first.
2. Look at the input yourself: `evidence/frames_overview.jpg`, a few frames, the motion in the clip. Write down in
   `WORKING.md`: what the scene contains (static structure, interactive objects, dynamic distractors such as hands,
   tools, cables), how the camera moves, what is hard (thin structures, openings, transparency, occlusion).
3. Plan in `WORKING.md`: which quantities you need (cameras, scale, world frame, masks, each object's shape, rest
   states, materials, lights) and how you will get and verify each one.

The normal stage order is `scene -> trajectory -> realism -> synthesis`. A passed trajectory
is the immutable execution input to realism. A passed, current realism review is required
before synthesis; do not fold appearance acceptance back into trajectory acceptance.

## Loop
- Build with tools + your own code (`work/code/`). Check every intermediate result visually and numerically before
  building on it (cameras and scale before contact; geometry before appearance).
- When a version is ready, freeze it: copy `scene/`, `blender/`, `mujoco/` (+ `isaac/`, `robot/`, `realism/`, `render/`, `synthesis/`
  and `dataset/` when the stage has them; use copies or copy-on-write reflinks, never mutable hard links) and the evidence you rely on to
  `candidates/<k>/` (never edit a frozen candidate) and note it in `WORKING.md`.
  Realism candidates also preserve permitted reference frames/masks and the exact evidence paths in
  [the stage contract](../../docs/realism_stage.md).
- Delegate the evaluation to the `stage-evaluator` sub-agent with: project path, candidate folder, the rubric and
  fidelity level from the request, the stage. Do not tell it what to conclude.
- Read the report, fix the root cause at the stage that owns it, record the decision, re-freeze, re-evaluate.
- Stop when the evaluator passes the candidate, or when a budget (hours, evaluations) is exhausted; then write the
  final summary in `WORKING.md` (deliverables, scorecard, open issues, what you would do next).

## Budget discipline
Prefer cheap checks (render_views, a few frames) before expensive runs (full Blender renders, long fits). Do not
re-run heavy tools without a reason recorded in WORKING.md. Keep tool outputs you read short.
