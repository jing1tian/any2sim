# Standard rendering and realism

The standard sequence is **nominal raw render → appearance/lighting repair → per-camera image-model
validation → nominal processed/encoded comparison → independent review → batch rendering → export**.
Cycles, noise, tone mapping or a successful calibration command do not by themselves establish realism.

Fixed processing is separate from episode randomization. A spatial-and-lighting-only dataset still uses its
reviewed materials, optics and camera response on every episode. Leave sensor, textures and dressing variation
disabled; do not synthesize uncalibrated noise just to make processing nonidentity. Repair a frozen scene only
as a declared revision; preserve physics and recheck labels after appearance changes.

Both render entry points resolve `scene.render.realism_baseline`; explicit tool parameters take precedence:

```yaml
render:
  realism_baseline:
    engine: cycles
    samples: 128
    denoise: true
    motion_blur: 0.5
    max_bounces: 12
    diffuse_bounces: 4
    glossy_bounces: 4
    adaptive_threshold: 0.01
```

These are starting settings, not calibrated shutter values or guaranteed quality. A reviewed scene can choose
different settings, including Eevee for synthesis. `blender_render` remains Cycles. Motion blur in trajectory
renders is enabled only for executed/keyframed motion. Labels are rendered without ISP or motion blur.
For Cycles, `samples` is a ceiling when adaptive sampling is enabled. Lower `adaptive_threshold` when measured
render noise stops improving as the ceiling rises; zero disables adaptive sampling. This setting is included
in the baseline fingerprint and therefore requires fresh evidence and review when changed.

`blender_render` now defaults to `camera_model: true`. It resolves `render.camera_isp.source_cam` before the
source camera's ISP, retains raw images, and uses its fixed bitrate for the video. Explicit `camera_model: false`
produces raw diagnostics. Synthesis resolves each camera independently, with `render.camera_isp.head` and
`render.camera_isp.wrist_*` overrides. An explicit empty override is identity; wrists never inherit the head.
The render manifest records the actual ISP parameters and whether any pixel-processing fields are configured.
Bitrate-only settings are identified as no pixel processing, not reported as a successful realism review.

## Independent realism stage and synthesis handoff

The default order is `scene -> trajectory -> realism -> synthesis`. Use the
[realism stage contract](realism_stage.md) for protocol freezing, direct achieved-motion
rendering, signal-chain/lighting diagnostics, independent review schema v2 and stage adoption.
`realism_render` shares the synthesis RGB/camera path but requires no synthetic rollout.

Full non-nominal rendering and export revalidate the current baseline, protected execution,
frozen protocol and all review evidence hashes. Nominal/previews remain available for repairs.
Legacy reviews with only materials/lighting/temporal status fields no longer pass the gate.
An explicit diagnostic override remains unaccepted and cannot satisfy stage advancement.

Existing jobs retain their loaded implementation. Existing videos and datasets are not
retroactively improved or rerendered. Adopt the new protocol through a fresh realism review.
