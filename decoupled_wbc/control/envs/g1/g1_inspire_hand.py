import gymnasium as gym
import numpy as np

from decoupled_wbc.control.base.env import Env
from decoupled_wbc.control.envs.g1.utils import inspire_hand_spec
from decoupled_wbc.control.envs.g1.utils.command_sender import InspireHandCommandSender
from decoupled_wbc.control.envs.g1.utils.state_processor import InspireHandStateProcessor


class G1InspireHand(Env):
    """Single Inspire RH56 6-DOF hand exposed through the standard Env interface.

    Mirrors ``G1ThreeFingerHand`` so it is a drop-in replacement, but operates on
    6 actuated drives and communicates over the Inspire DDS SDK topics
    (``rt/inspire_hand/{ctrl,state}/{l,r}``). Joint angles are in radians in the
    actuator order of ``inspire_hand_spec.JOINT_ORDER``; the sender/processor handle
    the radian<->drive-unit conversion.
    """

    NUM_DOF = inspire_hand_spec.NUM_INSPIRE_DOF

    def __init__(self, is_left: bool = True):
        super().__init__()
        self.is_left = is_left
        self.hand_state_processor = InspireHandStateProcessor(is_left=self.is_left)
        self.hand_command_sender = InspireHandCommandSender(is_left=self.is_left)

    def observe(self) -> dict[str, any]:
        hand_state = self.hand_state_processor._prepare_low_state()  # (1, 4 * NUM_DOF)
        assert hand_state.shape == (1, 4 * self.NUM_DOF)

        n = self.NUM_DOF
        hand_q = hand_state[0, 0:n]
        hand_dq = hand_state[0, n : 2 * n]
        hand_tau_est = hand_state[0, 2 * n : 3 * n]
        hand_ddq = hand_state[0, 3 * n : 4 * n]

        return {
            "hand_q": hand_q,
            "hand_dq": hand_dq,
            "hand_ddq": hand_ddq,
            "hand_tau_est": hand_tau_est,
        }

    def queue_action(self, action: dict[str, any]):
        # action should contain hand_q (6 joint angles in radians, actuator order)
        self.hand_command_sender.send_command(action["hand_q"])

    def observation_space(self) -> gym.Space:
        box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.NUM_DOF,))
        return gym.spaces.Dict(
            {
                "hand_q": box,
                "hand_dq": box,
                "hand_ddq": box,
                "hand_tau_est": box,
            }
        )

    def action_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {"hand_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.NUM_DOF,))}
        )

    def calibrate_hand(self):
        # The Inspire hand uses absolute encoders and the driver handles homing, so
        # no runtime calibration is required (unlike the dex3 three-finger hand).
        print("Inspire hand: no calibration required")
