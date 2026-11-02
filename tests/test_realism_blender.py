"""Tiny CPU Blender integration: actual passes, video encoding/decoding and protected motion.

Run with envs/render/bin/python -m unittest discover -s tests -p test_realism_blender.py.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

try:
    import bpy
except ImportError:
    raise unittest.SkipTest("requires the Blender render environment")

from any2sim import realism as R, scene_ir as IR, render_pipeline as RP
from any2sim.backends import blender_scene as B
from any2sim.tools.realism_render import run


class RealismRenderTest(unittest.TestCase):
    def test_real_passes_and_encoded_checkpoints(self):
        with tempfile.TemporaryDirectory(prefix="any2sim-realism-render-") as directory:
            p = Path(directory)
            for folder in ("scene", "robot", "inputs", "work/frames"):
                (p / folder).mkdir(parents=True)
            (p / "project.yaml").write_text("id: test\n")
            (p / "robot/robot.yaml").write_text("name: fixture\n")
            (p / "robot/robot.urdf").write_text('<robot name="fixture"><link name="base"/></robot>')
            (p / "scene/plane.obj").write_text("v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nf 1 2 3 4\n")
            camera = np.diag([1., -1., -1., 1.]); camera[2, 3] = 2
            scene = {"schema_version": "0.4", "units": "m", "frames": {}, "entities": {
                "table": {"capability": "static", "T_world_entity": np.eye(4).tolist(),
                          "geometry": {"visual": {"obj_uri": "plane.obj"}, "collision": []},
                          "trajectory": {"motion": "ach"},
                          "material": {"base_color": [.2, .3, .1], "roughness": .6}}},
                "cameras": {"source_cam": {"width": 32, "height": 32,
                    "K": [[32, 0, 16], [0, 32, 16], [0, 0, 1]], "poses_uri": "camera.json"}},
                "robots": {"R": {"bundle": "../robot", "T_world_base": np.eye(4).tolist()}},
                "motions": {"cmd": {"uri": "cmd.npz", "kind": "commanded", "robot": "R"},
                            "ach": {"uri": "ach.npz", "kind": "achieved", "robot": "R", "derived_from": "cmd"}},
                "render": {"motion": "ach", "realism_baseline": {"samples": 4, "motion_blur": 0}},
                "lights": [{"type": "world", "color": [1, 1, 1], "strength": .2}]}
            IR.save(p / "scene", scene)
            camera_poses = np.repeat(camera[None], 20, axis=0)
            camera_poses[:, 0, 3] = np.arange(20) * .01
            R.write_json(p / "scene/camera.json", {"T_world_cam": camera_poses.tolist()})
            source_frames = np.array([10, 12, 14, 16])
            object_poses = np.repeat(np.eye(4)[None, None], 4, axis=0)
            object_poses[:, 0, 0, 3] = np.arange(4) * .02
            for name in ("cmd", "ach"):
                IR.save_motion(p / f"scene/{name}.npz", np.arange(4) / 20, source_frames,
                               joint_names=[], q=np.zeros((4, 0)), obj_names=["table"], obj_T=object_poses)
            original_motion = (p / "scene/ach.npz").read_bytes()
            R.initialize(p, ["head"])
            protocol = yaml.safe_load((p / "realism/protocol.yaml").read_text())
            protocol["cameras"]["head"].update(
                reference_note="No real reference for this synthetic integration fixture.",
                regions=[{"name": "surface", "role": "surface", "box": [0, 0, 32, 32]}])
            (p / "realism/protocol.yaml").write_text(yaml.safe_dump(protocol))
            R.freeze_protocol(p)

            def setup_cpu(samples, width, height):
                sc = bpy.context.scene
                RP.apply_settings(sc, RP.settings({}, samples=samples, motion_blur=0))
                sc.cycles.device = "CPU"
                sc.render.threads_mode = "FIXED"; sc.render.threads = 2
                sc.render.resolution_x, sc.render.resolution_y = width, height
                sc.render.resolution_percentage = 100
                sc.render.image_settings.file_format = "PNG"
                sc.render.image_settings.color_mode = "RGB"
                return "CPU"

            with patch.object(B, "setup_cycles", setup_cpu):
                rendered = run(p)
            self.assertTrue(rendered["complete"])
            self.assertEqual((p / "scene/ach.npz").read_bytes(), original_motion)
            checked = R.check(p)
            self.assertEqual(checked["status"], "ready_for_review", checked["issues"])
            self.assertEqual(checked["cameras"]["head"]["temporal"]["frames"], 4)
            self.assertFalse((p / "render/realism_review.json").exists())
            exr = p / "realism/renders/nominal/check/head_00000_light.exr"
            self.assertGreaterEqual(len(R.exr_metadata(exr)["channels"]), 18)
            self.assertTrue(bpy.context.scene.cycles.use_denoising)
            self.assertEqual(bpy.context.scene.render.image_settings.file_format, "PNG")
            for row in range(4):
                bpy.context.scene.frame_set(row)
                self.assertAlmostEqual(bpy.data.objects["table"].matrix_world.translation.x, .02 * row, places=5)
                self.assertAlmostEqual(bpy.data.objects["cam_head"].matrix_world.translation.x,
                                       .01 * source_frames[row], places=5)


if __name__ == "__main__":
    unittest.main()
