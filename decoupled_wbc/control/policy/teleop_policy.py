from contextlib import contextmanager
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from decoupled_wbc.control.base.policy import Policy
from decoupled_wbc.control.robot_model import RobotModel
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from decoupled_wbc.control.teleop.teleop_streamer import TeleopStreamer


class TeleopPolicy(Policy):
    """
    Robot-agnostic teleop policy.
    Clean separation: IK processing vs command passing.
    All robot-specific properties are abstracted through robot_model and hand_ik_solvers.
    """

    def __init__(
        self,
        body_control_device: str,
        hand_control_device: str,
        robot_model: RobotModel,
        retargeting_ik: TeleopRetargetingIK,
        body_streamer_ip: str = "192.168.?.?",
        body_streamer_keyword: str = "shoulder",
        enable_real_device: bool = True,
        replay_data_path: Optional[str] = None,
        replay_speed: float = 1.0,
        wait_for_activation: int = 5,
        activate_keyboard_listener: bool = True,
    ):
        if activate_keyboard_listener:
            from decoupled_wbc.control.utils.keyboard_dispatcher import KeyboardListenerSubscriber

            self.keyboard_listener = KeyboardListenerSubscriber()
        else:
            self.keyboard_listener = None

        self.wait_for_activation = wait_for_activation

        self.teleop_streamer = TeleopStreamer(
            robot_model=robot_model,
            body_control_device=body_control_device,
            hand_control_device=hand_control_device,
            enable_real_device=enable_real_device,
            body_streamer_ip=body_streamer_ip,
            body_streamer_keyword=body_streamer_keyword,
            replay_data_path=replay_data_path,
            replay_speed=replay_speed,
        )
        self.robot_model = robot_model
        self.retargeting_ik = retargeting_ik
        self.is_active = False

        self.latest_left_wrist_data = np.eye(4)
        self.latest_right_wrist_data = np.eye(4)
        self.latest_left_fingers_data = {"position": np.zeros((25, 4, 4))}
        self.latest_right_fingers_data = {"position": np.zeros((25, 4, 4))}

    def set_goal(self, goal: dict[str, any]):
        # The current teleop policy doesn't take higher level commands yet.
        pass

    def get_action(self) -> dict[str, any]:
        # Get structured data
        streamer_output = self.teleop_streamer.get_streamer_data()

        # Handle activation using teleop_data commands
        self.check_activation(
            streamer_output.teleop_data, wait_for_activation=self.wait_for_activation
        )

        action = {}

        # Process streamer data if active
        if self.is_active and streamer_output.ik_data:
            body_data = streamer_output.ik_data["body_data"]
            left_hand_data = streamer_output.ik_data["left_hand_data"]
            right_hand_data = streamer_output.ik_data["right_hand_data"]

            left_wrist_name = self.robot_model.supplemental_info.hand_frame_names["left"]
            right_wrist_name = self.robot_model.supplemental_info.hand_frame_names["right"]
            self.latest_left_wrist_data = body_data[left_wrist_name]
            self.latest_right_wrist_data = body_data[right_wrist_name]
            self.latest_left_fingers_data = left_hand_data
            self.latest_right_fingers_data = right_hand_data

            # TODO: This stores the same data again
            ik_data = {
                "body_data": body_data,
                "left_hand_data": left_hand_data,
                "right_hand_data": right_hand_data,
            }
            action["ik_data"] = ik_data

        # Wrist poses (pos and quat)
        # TODO: This stores the same wrist poses in two different formats
        left_wrist_matrix = self.latest_left_wrist_data
        right_wrist_matrix = self.latest_right_wrist_data
        left_wrist_pose = np.concatenate(
            [
                left_wrist_matrix[:3, 3],
                R.from_matrix(left_wrist_matrix[:3, :3]).as_quat(scalar_first=True),
            ]
        )
        right_wrist_pose = np.concatenate(
            [
                right_wrist_matrix[:3, 3],
                R.from_matrix(right_wrist_matrix[:3, :3]).as_quat(scalar_first=True),
            ]
        )

        # Combine IK results with control commands (no teleop_data commands)
        action.update(
            {
                "left_wrist": self.latest_left_wrist_data,
                "right_wrist": self.latest_right_wrist_data,
                "left_fingers": self.latest_left_fingers_data,
                "right_fingers": self.latest_right_fingers_data,
                "wrist_pose": np.concatenate([left_wrist_pose, right_wrist_pose]),
                **streamer_output.control_data,  # Only control & data collection commands pass through
                **streamer_output.data_collection_data,
            }
        )

        # Run retargeting IK
        if "ik_data" in action:
            self.retargeting_ik.set_goal(action["ik_data"])
        action["target_upper_body_pose"] = self.retargeting_ik.get_action()

        return action

    def close(self) -> bool:
        self.teleop_streamer.stop_streaming()
        return True

    def check_activation(self, teleop_data: dict, wait_for_activation: int = 5):
        """Activation logic only looks at teleop data commands"""
        key = self.keyboard_listener.read_msg() if self.keyboard_listener else ""
        toggle_activation_by_keyboard = key == "l"
        reset_teleop_policy_by_keyboard = key == "k"
        toggle_activation_by_teleop = teleop_data.get("toggle_activation", False)

        if reset_teleop_policy_by_keyboard:
            print("Resetting teleop policy")
            self.reset()

        if toggle_activation_by_keyboard or toggle_activation_by_teleop:
            self.is_active = not self.is_active
            if self.is_active:
                print("Starting teleop policy")

                if wait_for_activation > 0 and toggle_activation_by_keyboard:
                    print(f"Sleeping for {wait_for_activation} seconds before starting teleop...")
                    for i in range(wait_for_activation, 0, -1):
                        print(f"Starting in {i}...")
                        time.sleep(1)

                # dda: calibration logic should use current IK data
                self.teleop_streamer.calibrate()
                print("Teleop policy calibrated")
            else:
                print("Stopping teleop policy")

    @contextmanager
    def activate(self):
        try:
            yield self
        finally:
            self.close()

    def handle_keyboard_button(self, keycode):
        """
        Handle keyboard input with proper state toggle.
        """
        if keycode == "l":
            # Toggle start state
            self.is_active = not self.is_active
            # Reset initialization when stopping
            if not self.is_active:
                self._initialized = False
        if keycode == "k":
            print("Resetting teleop policy")
            self.reset()

    def activate_policy(self, wait_for_activation: int = 5):
        """activate the teleop policy"""
        self.is_active = False
        self.check_activation(
            teleop_data={"toggle_activation": True}, wait_for_activation=wait_for_activation
        )

    def reset(self, wait_for_activation: int = 5, auto_activate: bool = False):
        """Reset the teleop policy to the initial state, and re-activate it."""
        self.teleop_streamer.reset()
        self.retargeting_ik.reset()
        self.is_active = False
        self.latest_left_wrist_data = np.eye(4)
        self.latest_right_wrist_data = np.eye(4)
        self.latest_left_fingers_data = {"position": np.zeros((25, 4, 4))}
        self.latest_right_fingers_data = {"position": np.zeros((25, 4, 4))}

        if auto_activate:
            self.activate_policy(wait_for_activation)
