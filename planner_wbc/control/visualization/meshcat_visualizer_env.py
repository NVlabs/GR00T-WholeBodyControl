from contextlib import contextmanager
import time

import gymnasium as gym
import numpy as np
from pinocchio.visualize import MeshcatVisualizer

from decoupled_wbc.control.base.env import Env
from decoupled_wbc.control.robot_model import RobotModel


class MeshcatVisualizerEnv(Env):
    def __init__(self, robot_model: RobotModel):
        self.robot_model = robot_model
        self.viz = MeshcatVisualizer(
            self.robot_model.pinocchio_wrapper.model,
            self.robot_model.pinocchio_wrapper.collision_model,
            self.robot_model.pinocchio_wrapper.visual_model,
        )
        try:
            self.viz.initViewer(open=True)

        except ImportError as err:
            print("Error while initializing the viewer. It seems you should install Python meshcat")
            print(err)
            exit(0)

        self.viz.loadViewerModel()
        self.visualize(self.robot_model.pinocchio_wrapper.q0)
        time.sleep(1.0)

        self._observation_space = gym.spaces.Dict(
            {
                "q": gym.spaces.Box(
                    low=-2 * np.pi, high=2 * np.pi, shape=(self.robot_model.num_dofs,)
                )
            }
        )
        self._action_space = gym.spaces.Dict(
            {
                "q": gym.spaces.Box(
                    low=-2 * np.pi, high=2 * np.pi, shape=(self.robot_model.num_dofs,)
                )
            }
        )

    def visualize(self, robot_state: np.ndarray):
        # visualize robot state
        if robot_state is not None:
            self.viz.display(robot_state)

    def observe(self):
        # Dummy observation
        return {"q": self.robot_model.pinocchio_wrapper.q0}

    def queue_action(self, action: dict[str, np.ndarray]):
        self.visualize(action["q"])

    def reset(self, **kwargs):
        self.visualize(self.robot_model.pinocchio_wrapper.q0)
        return {"q": self.robot_model.pinocchio_wrapper.q0}

    def sensors(self) -> dict[str, any]:
        return {}

    def observation_space(self) -> gym.Space:
        return self._observation_space

    def action_space(self) -> gym.Space:
        return self._action_space

    def close(self):
        return

    @contextmanager
    def activate(self):
        yield
