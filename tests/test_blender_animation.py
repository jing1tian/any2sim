"""Run in the render environment; verify actual Blender subframe transforms."""
import unittest
from types import SimpleNamespace

import numpy as np

try:
    import bpy
except ImportError:
    raise unittest.SkipTest('requires the Blender render environment')

from mathutils import Matrix
from any2sim.backends import blender_scene as B


def pose(degrees, x=0):
    a = np.radians(degrees)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[0, 3] = x
    return T


class SampledAnimationTest(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def empty(self, name):
        ob = bpy.data.objects.new(name, None)
        bpy.context.collection.objects.link(ob)
        return ob

    def assert_pose(self, ob, expected):
        np.testing.assert_allclose(np.array(ob.matrix_world), expected, atol=2e-6)

    def test_object_wrap_and_linear_translation(self):
        ob = self.empty('prop')
        motion = {'t': np.array([0., .05]), 'frame': np.array([0., 1.]),
                  'joint_names': [], 'q': np.empty((2, 0)), 'obj_names': ['prop'],
                  'obj_T': np.array([[pose(179, 0)], [pose(181, 1)]])}
        scene = {'entities': {'prop': {'trajectory': {'motion': 'achieved'}}}}
        B.keyframe_motion(scene, {'prop': ob}, {}, motion, 2, None)
        for f in [0, 1]:
            bpy.context.scene.frame_set(f)
            self.assert_pose(ob, pose(179 + 2*f, f))
        bpy.context.scene.frame_set(0, subframe=.5)
        self.assert_pose(ob, pose(180, .5))
        bpy.context.scene.frame_set(0, subframe=.25)
        self.assertAlmostEqual(ob.matrix_world.translation.x, .25, places=6)

    def test_robot_wrap_and_attached_camera(self):
        root, link, cam = (self.empty(n) for n in ['base', 'link', 'camera'])
        cam.parent = link
        cam.matrix_basis = Matrix.Translation((.02, 0, .01))
        model = {'root': 'base', 'links': {'base': {}, 'link': {}}, 'joints': {
            'joint': {'type': 'revolute', 'parent': 'base', 'child': 'link',
                      'T': np.eye(4), 'axis': [0, 0, 1]}}}
        robot = {'bundle': SimpleNamespace(model=model), 'q0': {'joint': 0},
                 'T_world_base': np.eye(4), 'links': {'base': root, 'link': link}, 'cables': {}}
        motion = {'t': np.array([0., .05]), 'frame': np.array([0., 1.]),
                  'joint_names': ['joint'], 'q': np.radians([[179.], [181.]]),
                  'obj_names': [], 'obj_T': np.empty((2, 0, 4, 4))}
        B.keyframe_motion({'entities': {}}, {}, {'robot': robot}, motion, 2, 'robot')
        bpy.context.scene.frame_set(0, subframe=.5)
        self.assert_pose(link, pose(180))
        expected = pose(180) @ np.array(Matrix.Translation((.02, 0, .01)))
        self.assert_pose(cam, expected)


if __name__ == '__main__':
    unittest.main()
