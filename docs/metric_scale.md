# Metric scale

Metric hand tracks and robot kinematics are fixed observations. When they disagree with reconstructed scene
scale, revise the scene explicitly rather than scaling those observations.

Compare at least two independent cues: contact registration, transport distances, known geometry or grip
aperture. Record uncertainty and explain disagreements. Use `scene_rescale` for a declared scene revision;
verify projections and contact timing again after the change.
