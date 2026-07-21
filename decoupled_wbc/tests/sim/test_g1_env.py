from unittest.mock import MagicMock
import sys
import types

from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
import numpy as np


def _install_ros_stubs() -> None:
    if "rclpy" in sys.modules:
        return

    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: False
    rclpy.init = lambda *args, **kwargs: None
    rclpy.create_node = lambda *args, **kwargs: None
    rclpy.spin = lambda *args, **kwargs: None
    rclpy.get_global_executor = lambda: types.SimpleNamespace(get_nodes=lambda: [])
    rclpy.shutdown = lambda *args, **kwargs: None
    rclpy.exceptions = types.SimpleNamespace(ROSInterruptException=Exception)

    rclpy_executors = types.ModuleType("rclpy.executors")
    rclpy_executors.SingleThreadedExecutor = object

    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = object

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = object
    sensor_msgs.msg = sensor_msgs_msg

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.ByteMultiArray = object
    std_msgs.msg = std_msgs_msg

    std_srvs = types.ModuleType("std_srvs")
    std_srvs_srv = types.ModuleType("std_srvs.srv")
    std_srvs_srv.Trigger = object
    std_srvs.srv = std_srvs_srv

    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.executors"] = rclpy_executors
    sys.modules["rclpy.node"] = rclpy_node
    sys.modules["sensor_msgs"] = sensor_msgs
    sys.modules["sensor_msgs.msg"] = sensor_msgs_msg
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs_msg
    sys.modules["std_srvs"] = std_srvs
    sys.modules["std_srvs.srv"] = std_srvs_srv


_install_ros_stubs()

from decoupled_wbc.control.envs.g1.g1_env import G1Env


class TestG1EnvGravityCompensation:
    def test_queue_action_applies_gravity_compensation_when_enabled(self):
        robot_model = instantiate_g1_robot_model(waist_location="lower_body")

        env = object.__new__(G1Env)
        env.robot_model = robot_model
        env.enable_gravity_compensation = True
        env.gravity_compensation_joints = ["arms"]
        env.with_hands = False

        arm_indices = robot_model.get_joint_group_indices("arms")
        q = np.zeros(robot_model.num_dofs)
        for idx in arm_indices:
            q[idx] = 0.3
        env.last_obs = {"q": q}

        mock_safety = MagicMock()
        mock_safety.handle_violations = lambda obs, action: {"action": action}
        env.safety_monitor = mock_safety

        captured_actions = []
        mock_body = MagicMock()
        mock_body.queue_action = lambda a: captured_actions.append(a)
        env.body = lambda: mock_body

        env.queue_action({"q": np.zeros(robot_model.num_dofs)})

        assert len(captured_actions) == 1
        body_tau = captured_actions[0]["body_tau"]
        assert not np.allclose(body_tau, 0)

    def test_queue_action_keeps_zero_torques_when_gravity_compensation_disabled(self):
        robot_model = instantiate_g1_robot_model(waist_location="lower_body")

        env = object.__new__(G1Env)
        env.robot_model = robot_model
        env.enable_gravity_compensation = False
        env.gravity_compensation_joints = ["arms"]
        env.with_hands = False

        arm_indices = robot_model.get_joint_group_indices("arms")
        q = np.zeros(robot_model.num_dofs)
        for idx in arm_indices:
            q[idx] = 0.3
        env.last_obs = {"q": q}

        mock_safety = MagicMock()
        mock_safety.handle_violations = lambda obs, action: {"action": action}
        env.safety_monitor = mock_safety

        captured_actions = []
        mock_body = MagicMock()
        mock_body.queue_action = lambda a: captured_actions.append(a)
        env.body = lambda: mock_body

        env.queue_action({"q": np.zeros(robot_model.num_dofs)})

        assert len(captured_actions) == 1
        body_tau = captured_actions[0]["body_tau"]
        assert np.allclose(body_tau, 0)


def test_control_loop_config_defaults_gravity_compensation_joints_to_arms():
    config = ControlLoopConfig(
        interface="sim",
        enable_gravity_compensation=True,
        gravity_compensation_joints=None,
    )

    wbc_config = config.load_wbc_yaml()

    assert wbc_config["gravity_compensation_joints"] == ["arms"]
