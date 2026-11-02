# Rendering realism

Use `robot_cables` and `robot_appearance` with authorized robot observations to fit appearance.
Keep appearance revisions separate from physics revisions.

Use `camera_calibrate` to estimate a camera model from aligned static regions. Validate changes on held-out
frames. Check sharpness and noise separately; image similarity alone can reward excessive smoothing.

Use `realism_metrics` for feature-based and low-level image comparisons. Preserve evidence for each trial.
See `realism_stage.md` and `render_pipeline.md` for the review protocol and rendering gates.
