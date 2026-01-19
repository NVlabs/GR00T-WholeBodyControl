"""
Integration tests for G1Env.

Tests the gravity compensation fix from commit c28acee.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from gr00t_wbc.control.robot_model.instantiation import instantiate_g1_robot_model


class TestG1EnvGravityCompensation:
    """
    Integration test for G1Env.queue_action() gravity compensation.

    This test WILL FAIL if the fix is reverted (old code always sent body_tau=zeros).
    """

    @pytest.mark.parametrize("enable_gravity,expect_nonzero", [
        (True, True),   # Gravity enabled → expect non-zero torques
        (False, False), # Gravity disabled → expect zero torques
    ])
    def test_queue_action_gravity_compensation(self, enable_gravity, expect_nonzero):
        """
        Test that G1Env.queue_action() respects the gravity compensation flag.
        """
        from gr00t_wbc.control.envs.g1.g1_env import G1Env

        robot_model = instantiate_g1_robot_model(waist_location="lower_body")

        # Create minimal mock G1Env with real queue_action method
        env = object.__new__(G1Env)
        env.robot_model = robot_model
        env.enable_gravity_compensation = enable_gravity
        env.gravity_compensation_joints = ["arms"]
        env.with_hands = False  # Skip hand commands in queue_action

        # Set up last_obs with arm configuration
        arm_indices = robot_model.get_joint_group_indices("arms")
        q = np.zeros(robot_model.num_dofs)
        for idx in arm_indices:
            q[idx] = 0.3
        env.last_obs = {"q": q}

        # Mock safety_monitor
        mock_safety = MagicMock()
        mock_safety.handle_violations = lambda obs, action: {"action": action}
        env.safety_monitor = mock_safety

        # Capture what queue_action sends
        captured_actions = []
        mock_body = MagicMock()
        mock_body.queue_action = lambda a: captured_actions.append(a)
        env.body = lambda: mock_body

        # Call actual queue_action
        env.queue_action({"q": np.zeros(robot_model.num_dofs)})

        # Verify
        assert len(captured_actions) == 1
        body_tau = captured_actions[0]["body_tau"]

        if expect_nonzero:
            assert not np.allclose(body_tau, 0), \
                "BUG: body_tau is zeros when gravity compensation is enabled!"
        else:
            assert np.allclose(body_tau, 0), \
                "body_tau should be zeros when gravity compensation is disabled"
