# Reference recipes: v0 hand-built pipeline (not tools)

These modules are the first, hand-written pipeline for a tabletop demonstration (tabletop, static head
camera). They encode scene-specific decisions (table-plane world frame, clean plate, rest-state detection, planar
resting registration, flat support feet, diffuse de-lighting of a table, ...). They are **not** in the tool catalog
and are not maintained against the current library. Read them for ideas and pitfalls; write your own project code
(`projects/<id>/work/code/`) for the scene at hand, using the general tools and the shared library.

Known pitfalls recorded while building them (validate these against your own observations):
SVD of N x 3 point sets allocates N x N (use eigh of the 3x3 covariance); SAM 3D GLB axes are y-up
(`to_glb` applies [[1,0,0],[0,0,-1],[0,1,0]]) and its layout is a PyTorch3D row-vector transform; generated meshes
can have openings (fit silhouettes with a filled/convex proxy); a single plane-induced homography leaves rotation and
translation ambiguous; the head camera's auto white balance drifts during the clip; tracked "dynamic" prompts also
fire on objects; meshes with shared UVs between faces corrupt Blender bakes.
