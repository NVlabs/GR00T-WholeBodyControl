from typing import Any, Dict

import gymnasium as gym
import numpy as np

from decoupled_wbc.control.base.env import Env
from decoupled_wbc.control.envs.g1.utils.command_sender import BodyCommandSender
from decoupled_wbc.control.envs.g1.utils.state_processor import BodyStateProcessor


class G1Body(Env):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.body_state_processor = BodyStateProcessor(config=config)
        self.body_command_sender = BodyCommandSender(config=config)

    def observe(self) -> dict[str, any]:
        body_state = self.body_state_processor._prepare_low_state()  # (1, 148)
        assert body_state.shape == (1, 148)
        body_q = body_state[
            0, 7 : 7 + 12 + 3 + 7 + 7
        ]  # leg (12) + waist (3) + left arm (7) + right arm (7)
        body_dq = body_state[0, 42 : 42 + 12 + 3 + 7 + 7]
        body_ddq = body_state[0, 112 : 112 + 12 + 3 + 7 + 7]
        body_tau_est = body_state[0, 77 : 77 + 12 + 3 + 7 + 7]
        floating_base_pose = body_state[0, 0:7]
        floating_base_vel = body_state[0, 36:42]
        floating_base_acc = body_state[0, 106:112]
        torso_quat = body_state[0, 141:145]
        torso_ang_vel = body_state[0, 145:148]

        return {
            "body_q": body_q,
            "body_dq": body_dq,
            "body_ddq": body_ddq,
            "body_tau_est": body_tau_est,
            "floating_base_pose": floating_base_pose,
            "floating_base_vel": floating_base_vel,
            "floating_base_acc": floating_base_acc,
            "torso_quat": torso_quat,
            "torso_ang_vel": torso_ang_vel,
        }

    def queue_action(self, action: dict[str, any]):
        # action should contain body_q, body_dq, body_tau
        self.body_command_sender.send_command(
            action["body_q"], action["body_dq"], action["body_tau"]
        )

    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "body_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,)),
                "body_dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,)),
                "floating_base_pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "floating_base_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,)),
            }
        )

    def action_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "body_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,)),
                "body_dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,)),
                "body_tau": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,)),
            }
        )
