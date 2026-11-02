"""Project folders: one per source episode / task.

projects/<id>/
  project.yaml        id, created, source, notes
  inputs/             the observations exactly as supplied (video only for a video-only run)
  observation.yaml    ObservationBundle: what was supplied, what was withheld, timing, provenance
  work/<step>/        tool outputs, one folder per step (frames, masks, geometry, objects, background, ...)
  scene/              Scene IR (scene.yaml) + registered assets: the source of truth for every backend
  blender/            compiled .blend + renders      (backend artifact, regenerated from scene/)
  mujoco/             compiled MJCF + physics checks (backend artifact, regenerated from scene/)
  evidence/           comparisons, metrics, review packs
  evaluations/        independent evaluator reports
  receipts/           one JSON per tool execution
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

LAYOUT = ("inputs", "work", "scene", "blender", "mujoco", "evidence", "evaluations", "receipts")


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Project:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not (self.root / "project.yaml").exists():
            raise FileNotFoundError(f"not a project: {self.root}")

    @classmethod
    def create(cls, root: str | Path, source: dict, notes: str = "") -> "Project":
        import yaml
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        for d in LAYOUT:
            (root / d).mkdir(exist_ok=True)
        meta = {"id": root.name, "created": now(), "source": source, "notes": notes}
        (root / "project.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
        return cls(root)

    def path(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def work(self, step: str) -> Path:
        p = self.root / "work" / step
        p.mkdir(parents=True, exist_ok=True)
        return p

    def frames_dir(self) -> Path:
        return self.root / "work" / "frames"

    def observation(self) -> dict:
        import yaml
        return yaml.safe_load((self.root / "observation.yaml").read_text())

    def receipts(self, tool: str | None = None) -> list[dict]:
        out = []
        for p in sorted((self.root / "receipts").glob("*.json")):
            r = json.loads(p.read_text())
            if tool is None or r.get("tool") == tool:
                out.append(r)
        return out
