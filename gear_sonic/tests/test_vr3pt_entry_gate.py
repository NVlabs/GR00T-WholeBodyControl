import math
import unittest

import numpy as np

from gear_sonic.scripts.pico_manager_thread_server import evaluate_vr3pt_entry_gate


def _identity_pose():
    pose = np.zeros((3, 7), dtype=np.float64)
    pose[:, 3] = 1.0
    return pose


def _g1_poses():
    return {
        "left_wrist": {
            "position": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "orientation_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        },
        "right_wrist": {
            "position": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "orientation_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        },
        "torso": {
            "position": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        },
    }


class Vr3PtEntryGateTest(unittest.TestCase):
    def test_passes_when_position_and_orientation_are_within_limits(self):
        pose = _identity_pose()
        pose[0, 0] = 0.02
        pose[1, 1] = -0.03
        pose[2, 2] = 0.04

        passed, _ = evaluate_vr3pt_entry_gate(pose, _g1_poses())

        self.assertTrue(passed)

    def test_fails_large_wrist_position_error(self):
        pose = _identity_pose()
        pose[0, 0] = 0.30

        passed, details = evaluate_vr3pt_entry_gate(pose, _g1_poses())

        self.assertFalse(passed)
        self.assertIn("left_wrist_pos=0.300m", details)

    def test_fails_large_torso_position_error(self):
        pose = _identity_pose()
        pose[2, 0] = 0.24

        passed, details = evaluate_vr3pt_entry_gate(pose, _g1_poses())

        self.assertFalse(passed)
        self.assertIn("torso_pos=0.240m", details)

    def test_fails_large_wrist_orientation_error(self):
        pose = _identity_pose()
        pose[0, 3] = math.cos(math.pi / 4.0)
        pose[0, 4] = math.sin(math.pi / 4.0)

        passed, details = evaluate_vr3pt_entry_gate(pose, _g1_poses())

        self.assertFalse(passed)
        self.assertIn("left_wrist_orn=90.0deg", details)


if __name__ == "__main__":
    unittest.main()
