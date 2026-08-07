import pathlib
import time

import numpy as np
import pytest

from decoupled_wbc.control.robot_model import RobotModel
from decoupled_wbc.control.robot_model.supplemental_info.g1.g1_supplemental_info import (
    G1SupplementalInfo,
)


@pytest.fixture
def env_fixture():
    """
    Pytest fixture that creates and yields the MeshcatVisualizerEnv.
    After the test, it closes the environment to clean up.
    """
    from decoupled_wbc.control.visualization.meshcat_visualizer_env import MeshcatVisualizerEnv

    root_dir = pathlib.Path(__file__).parent.parent.parent.parent
    urdf_path = str(
        root_dir / "decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
    )
    asset_path = str(root_dir / "decoupled_wbc/control/robot_model/model_data/g1")
    robot_config = {
        "asset_path": asset_path,
        "urdf_path": urdf_path,
    }
    robot_model = RobotModel(
        robot_config["urdf_path"],
        robot_config["asset_path"],
        supplemental_info=G1SupplementalInfo(),
    )
    env = MeshcatVisualizerEnv(robot_model)
    time.sleep(0.5)
    yield env
    env.close()


def test_meshcat_env_init(env_fixture):
    """
    Test that the environment initializes without errors
    and that reset() returns the proper data structure.
    """
    env = env_fixture
    initial_obs = env.reset()
    assert isinstance(initial_obs, dict), "reset() should return a dictionary."
    assert "q" in initial_obs, "The returned dictionary should contain key 'q'."
    assert (
        len(initial_obs["q"]) == env.robot_model.num_dofs
    ), "Length of 'q' should match the robot's DOF."


def test_meshcat_env_observation(env_fixture):
    """
    Test that the observe() method returns a valid observation
    conforming to the environment's observation space.
    """
    env = env_fixture
    observation = env.observe()
    assert isinstance(observation, dict), "observe() should return a dictionary."
    assert "q" in observation, "The returned dictionary should contain key 'q'."
    assert (
        len(observation["q"]) == env.robot_model.num_dofs
    ), "Length of 'q' should match the robot's DOF."


def test_meshcat_env_action(env_fixture):
    """
    Test that we can queue an action and visualize it without error.
    """
    env = env_fixture
    # Build a dummy action within the action space
    test_action = {"q": 0.2 * np.ones(env.robot_model.num_dofs)}

    # This should not raise an exception and should visualize the correct configuration
    env.queue_action(test_action)


def test_meshcat_env_close(env_fixture):
    """
    Test closing the environment. (Though the fixture calls env.close()
    automatically, we can invoke it here to ensure it's safe to do so.)
    """
    env = env_fixture
    env.close()
    # If close() triggers no exceptions, we're good.
