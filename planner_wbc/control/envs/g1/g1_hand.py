import time

import gymnasium as gym
import numpy as np

from decoupled_wbc.control.base.env import Env
from decoupled_wbc.control.envs.g1.utils.command_sender import HandCommandSender
from decoupled_wbc.control.envs.g1.utils.state_processor import HandStateProcessor


class G1ThreeFingerHand(Env):
    def __init__(self, is_left: bool = True):
        super().__init__()
        self.is_left = is_left
        self.hand_state_processor = HandStateProcessor(is_left=self.is_left)
        self.hand_command_sender = HandCommandSender(is_left=self.is_left)
        self.hand_q_offset = np.zeros(7)

    def observe(self) -> dict[str, any]:
        hand_state = self.hand_state_processor._prepare_low_state()  # (1, 28)
        assert hand_state.shape == (1, 28)

        # Apply offset to the hand state
        hand_state[0, :7] = hand_state[0, :7] + self.hand_q_offset

        hand_q = hand_state[0, :7]
        hand_dq = hand_state[0, 7:14]
        hand_ddq = hand_state[0, 21:28]
        hand_tau_est = hand_state[0, 14:21]

        # Return the state for this specific hand (left or right)
        return {
            "hand_q": hand_q,
            "hand_dq": hand_dq,
            "hand_ddq": hand_ddq,
            "hand_tau_est": hand_tau_est,
        }

    def queue_action(self, action: dict[str, any]):
        # Apply offset to the hand target
        action["hand_q"] = action["hand_q"] - self.hand_q_offset

        # action should contain hand_q
        self.hand_command_sender.send_command(action["hand_q"])

    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "hand_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_ddq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_tau_est": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
            }
        )

    def action_space(self) -> gym.Space:
        return gym.spaces.Dict({"hand_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))})

    def calibrate_hand(self):
        hand_obs = self.observe()
        hand_q = hand_obs["hand_q"]

        hand_q_target = np.zeros_like(hand_q)
        hand_q_target[0] = hand_q[0]

        # joint limit
        hand_q0_upper_limit = np.deg2rad(60)  # lower limit is -60

        # move the figure counterclockwise until the limit
        while True:

            if hand_q_target[0] - hand_q[0] < np.deg2rad(60):
                hand_q_target[0] += np.deg2rad(10)
            else:
                self.hand_q_offset[0] = hand_q0_upper_limit - hand_q[0]
                break

            self.queue_action({"hand_q": hand_q_target})

            hand_obs = self.observe()
            hand_q = hand_obs["hand_q"]

            time.sleep(0.1)

        print("done calibration, q0 offset (deg):", np.rad2deg(self.hand_q_offset[0]))

        # done calibrating, set target to zero
        self.hand_q_target = np.zeros_like(hand_q)
        self.queue_action({"hand_q": self.hand_q_target})
