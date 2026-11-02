# Video-input example

Supply a demonstration video you are authorized to use:

```bash
envs/base/bin/any2sim new demo --video /path/to/demo.mp4
cp templates/request_scene.yaml projects/demo/request.yaml
envs/base/bin/any2sim run projects/demo
```

Edit the request to describe the task and allowed inputs. Follow the reconstruction and appearance skills,
freeze a candidate, and obtain an independent evaluation. No example recordings or experiment results are
distributed with this release.
