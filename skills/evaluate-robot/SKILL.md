---
name: evaluate-robot
description: Independent evaluation of a RobotBundle against rubrics/robot.yaml using the supplied real-robot evidence. Used by the stage-evaluator sub-agent for robot onboarding.
---

# evaluate-robot (evaluator)

Judge the frozen bundle (`candidates/<k>/robot/`) without editing it. Re-run in your evaluation folder: bundle
`validate`; `robot_import`-style load check (MuJoCo compile of robot.xml); `robot_log_check` on log segments the
builder did not use; `robot_overlay` on frames the builder did not show (pick frames with large motions and with
the hand/tool open and closed); look at visuals in a Blender or MuJoCo preview next to real frames/photos. Every
check in robot.yaml `checks` must point to evidence you can reproduce; checks the evidence cannot support must be
`not_run` (never `pass`). Report format as in evaluate-scene.
