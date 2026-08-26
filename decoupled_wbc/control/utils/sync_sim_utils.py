from datetime import datetime
import pathlib
import shutil
from typing import Any, Dict, Optional

import numpy as np

from decoupled_wbc.control.envs.robocasa.sync_env import G1SyncEnv, SyncEnv
from decoupled_wbc.control.envs.robocasa.utils.controller_utils import (
    get_body_ik_solver_settings_type,
    update_robosuite_controller_configs,
)
from decoupled_wbc.control.main.constants import DEFAULT_BASE_HEIGHT, DEFAULT_NAV_CMD
from decoupled_wbc.control.main.teleop.configs.configs import SyncSimDataCollectionConfig
from decoupled_wbc.control.policy.teleop_policy import TeleopPolicy
from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from decoupled_wbc.control.robot_model.instantiation import get_robot_type_and_model
from decoupled_wbc.control.robot_model.robot_model import RobotModel
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from decoupled_wbc.control.utils.episode_state import EpisodeState
from decoupled_wbc.control.utils.text_to_speech import TextToSpeech
from decoupled_wbc.data.exporter import Gr00tDataExporter
from decoupled_wbc.data.utils import get_dataset_features, get_modality_config

MAX_MUJOCO_STATE_LEN = 800
COLLECTION_KEY = "c"
SKIP_KEY = "x"


class EpisodeManager:
    """Manages episode state transitions, done flags, and task completion hold counts.

    This class encapsulates the logic for:
    - Episode state management (IDLE -> RECORDING -> NEED_TO_SAVE)
    - Done flag handling
    - Task completion hold count tracking
    - Data collection triggering based on manual/auto mode
    - Step counting within episodes
    """

    def __init__(self, config: SyncSimDataCollectionConfig):
        self.config = config
        self.task_completion_hold_count = -1
        self.done = False
        self.step_count = 0

        # Initialize episode state and text-to-speech for both manual and automatic modes
        self.episode_state = EpisodeState()
        self.text_to_speech = TextToSpeech()

    def should_collect_data(self) -> bool:
        """Determine if data should be collected at this timestep."""
        if not self.config.data_collection:
            return False

        if self.config.manual_control:
            # Manual mode: only collect when RECORDING or NEED_TO_SAVE
            return self.episode_state.get_state() in [
                self.episode_state.RECORDING,
                self.episode_state.NEED_TO_SAVE,
            ]
        else:
            # Auto mode: collect when RECORDING or NEED_TO_SAVE
            return self.episode_state.get_state() in [
                self.episode_state.RECORDING,
                self.episode_state.NEED_TO_SAVE,
            ]

    def increment_step(self):
        """Increment the step counter."""
        self.step_count += 1

    def reset_step_count(self):
        """Reset the step counter to 0."""
        self.step_count = 0

    def get_step_count(self) -> int:
        """Get the current step count."""
        return self.step_count

    def handle_collection_trigger(
        self, wbc_goal: dict, keyboard_input: str | None, step_info: dict
    ):
        """Handle data collection start/stop triggers.

        Args:
            wbc_goal: WBC goal dictionary
            keyboard_input: Keyboard input from user (can be None)
            step_info: Step information from environment containing success flag
        """
        self.done = False
        state_changed = False

        if self.config.manual_control:
            if wbc_goal.get("toggle_data_collection", False) or (
                keyboard_input and keyboard_input == COLLECTION_KEY
            ):
                self.episode_state.change_state()
                # IDLE -> RECORDING or RECORDING -> NEED_TO_SAVE
                state_changed = True
        else:
            # Auto mode: automatically start collecting data for new env
            if self.episode_state.get_state() == self.episode_state.IDLE:
                self.episode_state.change_state()  # IDLE -> RECORDING
                state_changed = True
            self.done = step_info.get("success", False)
            if (
                self.episode_state.get_state() == self.episode_state.RECORDING
                and self.done
                and self.task_completion_hold_count == 0
            ):
                self.episode_state.change_state()  # RECORDING -> NEED_TO_SAVE
                state_changed = True

            # Check for CI test completion
            if self.config.ci_test and self.step_count >= self._get_ci_test_steps():
                self.episode_state.change_state()  # RECORDING -> NEED_TO_SAVE
                state_changed = True

        if state_changed:
            if self.episode_state.get_state() == self.episode_state.RECORDING:
                self.text_to_speech.print_and_say("Started recording episode")
            elif self.episode_state.get_state() == self.episode_state.NEED_TO_SAVE:
                self.done = True
                self.task_completion_hold_count = 0
                self.text_to_speech.print_and_say("Stopping recording, preparing to save")

    def check_export_and_completion(self, exporter: Gr00tDataExporter) -> bool:
        """Check if episode should be exported and update completion state.

        Args:
            exporter: Data exporter instance

        Returns:
            bool: True if environment needs to be reset
        """
        need_reset = False

        # Check if we should save the episode
        if self.task_completion_hold_count == 0:
            exporter.save_episode()
            need_reset = True
            self.task_completion_hold_count = -1

            if self.episode_state.get_state() == self.episode_state.NEED_TO_SAVE:
                self.episode_state.change_state()  # NEED_TO_SAVE -> IDLE
                self.text_to_speech.print_and_say("Episode saved.")

        # State machine to check for having a success for N consecutive timesteps
        elif self.done:
            print(
                f"Task success detected! Will collect {self.config.success_hold_steps} additional steps..."
            )
            print(f"currently {self.task_completion_hold_count}")
            if self.task_completion_hold_count > 0:
                self.task_completion_hold_count -= 1  # latched state, decrement count
                print(f"Task completed! Collecting {self.task_completion_hold_count} more steps...")
            else:
                self.task_completion_hold_count = (
                    self.config.success_hold_steps
                )  # reset count on first success timestep
                print(
                    f"Task success detected! Will collect {self.config.success_hold_steps} additional steps..."
                )
        else:
            self.task_completion_hold_count = -1  # null the counter if there's no success

        return need_reset

    def handle_skip(
        self, wbc_goal: dict, keyboard_input: str | None, exporter: Gr00tDataExporter
    ) -> bool:
        """Handle episode skip/abort.

        Args:
            wbc_goal: WBC goal dictionary
            keyboard_input: Keyboard input from user (can be None)
            exporter: Data exporter instance

        Returns:
            bool: True if episode was skipped and needs reset
        """
        if wbc_goal.get("toggle_data_abort", False) or (
            keyboard_input and keyboard_input == SKIP_KEY
        ):
            exporter.skip_and_start_new_episode()
            self.episode_state.reset_state()
            self.text_to_speech.print_and_say("Episode discarded, starting new episode")
            return True
        return False

    def _get_ci_test_steps(self) -> int:
        """Get CI test steps based on CI test mode."""
        if self.config.get("ci_test_mode", "unit") == "unit":
            return 50
        else:  # pre_merge
            return 500


class CITestManager:
    def __init__(self, config: SyncSimDataCollectionConfig):
        self.config = config
        self.ci_test_steps = 50 if config.ci_test_mode == "unit" else 500
        self.enable_tracking_check = True if config.ci_test_mode == "pre_merge" else False
        self.upper_body_speed_hist = []
        self.last_q_upper_body = None
        self.end_effector_tracking_errors = []

    def check_upper_body_motion(
        self, robot_model: RobotModel, wbc_action: dict, config: SyncSimDataCollectionConfig
    ):
        upper_body_joint_indices = robot_model.get_joint_group_indices("upper_body")
        q_upper_body = wbc_action["q"][upper_body_joint_indices]
        if self.last_q_upper_body is None:
            self.last_q_upper_body = q_upper_body.copy()
        self.upper_body_speed_hist.append(
            np.abs(q_upper_body - self.last_q_upper_body).mean() * config.control_frequency
        )
        self.last_q_upper_body = q_upper_body.copy()
        assert q_upper_body.mean() != 0, "Upper body joints should not be zero"

    def check_end_effector_tracking(
        self, teleop_cmd: dict, obs: dict, config: SyncSimDataCollectionConfig, i: int
    ):
        """
        Check end effector tracking error and validate thresholds.

        Args:
            teleop_cmd: Teleoperation command containing target poses
            obs: Environment observation containing current poses
            end_effector_tracking_errors: List to store tracking errors
            config: Configuration object containing robot type
            i: Current step index
            ci_test_steps: Number of steps for CI test
            upper_body_speed_hist: History of upper body joint speeds
        """
        from scipy.spatial.transform import Rotation as R

        # Get target poses from teleop command (replay data format)
        target_left_wrist = teleop_cmd.get("left_wrist")
        target_right_wrist = teleop_cmd.get("right_wrist")

        wrist_pose = obs.get("wrist_pose")

        # Extract left and right wrist poses from environment observation (7 for left + 7 for right)
        left_pos = wrist_pose[:3]
        left_quat = wrist_pose[3:7]
        right_pos = wrist_pose[7:10]
        right_quat = wrist_pose[10:14]

        # Convert quaternions to rotation matrices for error calculation
        left_rot_matrix = R.from_quat(left_quat, scalar_first=True).as_matrix()
        right_rot_matrix = R.from_quat(right_quat, scalar_first=True).as_matrix()

        # Construct 4x4 transformation matrices
        actual_left_wrist = np.eye(4)
        actual_left_wrist[:3, 3] = left_pos
        actual_left_wrist[:3, :3] = left_rot_matrix

        actual_right_wrist = np.eye(4)
        actual_right_wrist[:3, 3] = right_pos
        actual_right_wrist[:3, :3] = right_rot_matrix

        # Calculate position error
        left_pos_error = np.linalg.norm(target_left_wrist[:3, 3] - actual_left_wrist[:3, 3])
        right_pos_error = np.linalg.norm(target_right_wrist[:3, 3] - actual_right_wrist[:3, 3])

        # Calculate rotation error (similar to test_teleop_retargeting_ik)
        left_rot_diff = actual_left_wrist[:3, :3] @ target_left_wrist[:3, :3].T
        left_rot_error = np.arccos(np.clip((np.trace(left_rot_diff) - 1) / 2, -1, 1))
        right_rot_diff = actual_right_wrist[:3, :3] @ target_right_wrist[:3, :3].T
        right_rot_error = np.arccos(np.clip((np.trace(right_rot_diff) - 1) / 2, -1, 1))

        # Store max error for this timestep
        max_pos_error = max(left_pos_error, right_pos_error)
        max_rot_error = max(left_rot_error, right_rot_error)
        self.end_effector_tracking_errors.append((max_pos_error, max_rot_error))

        if i >= self.ci_test_steps:
            max_pos_errors = [error[0] for error in self.end_effector_tracking_errors]
            max_rot_errors = [error[1] for error in self.end_effector_tracking_errors]

            max_pos_error = np.max(max_pos_errors)
            max_rot_error = np.max(max_rot_errors)

            average_pos_error = np.mean(max_pos_errors)
            average_rot_error = np.mean(max_rot_errors)

            # More realistic thresholds based on observed data
            max_pos_threshold = 0.07  # 7cm threshold
            max_rot_threshold = np.deg2rad(17)  # 17 degree threshold
            average_pos_threshold = 0.05  # 5cm threshold
            average_rot_threshold = np.deg2rad(12)  # 12 degree threshold

            print(f"  Position errors - Max: {max_pos_error:.4f}")
            print(f"  Rotation errors - Max: {np.rad2deg(max_rot_error):.2f}°")
            print(f"  Average position error: {average_pos_error:.4f}")
            print(f"  Average rotation error: {np.rad2deg(average_rot_error):.2f}°")

            assert (
                max_pos_error < max_pos_threshold
            ), "Maximum end effector position tracking error exceeds threshold"
            assert (
                max_rot_error < max_rot_threshold
            ), "Maximum end effector rotation tracking error exceeds threshold"
            assert (
                average_pos_error < average_pos_threshold
            ), "Average end effector position tracking error exceeds threshold"
            assert (
                average_rot_error < average_rot_threshold
            ), "Average end effector rotation tracking error exceeds threshold"

            assert (
                np.array(self.upper_body_speed_hist).mean() > 0.03
            ), "Mean upper body joint velocities should be larger. Robot might not be moving."

            print("End effector tracking validation passed.")

            # Ensure end effector tracking validation has run
            if not self.end_effector_tracking_errors:
                assert False, "No end effector tracking data collected during CI test"
            elif len(self.end_effector_tracking_errors) < self.ci_test_steps:
                assert (
                    False
                ), f"Only {len(self.end_effector_tracking_errors)} end effector tracking samples collected"


def get_features(
    robot_model: RobotModel, save_img_obs: bool = False, image_obs_configs: dict[str, dict] = {}
) -> dict[str, dict]:
    """Fixture providing test features dict."""
    features = get_dataset_features(robot_model)
    features.update(
        {
            "observation.sim.seed": {
                "dtype": "int32",
                "shape": (1,),
            },
            "observation.sim.max_mujoco_state_len": {
                "dtype": "int32",
                "shape": (1,),
            },
            "observation.sim.mujoco_state_len": {
                "dtype": "int32",
                "shape": (1,),
            },
            "observation.sim.mujoco_state": {
                "dtype": "float64",
                "shape": (MAX_MUJOCO_STATE_LEN,),
            },
            "observation.sim.left_wrist": {
                "dtype": "float64",
                "shape": (16,),
            },
            "observation.sim.right_wrist": {
                "dtype": "float64",
                "shape": (16,),
            },
            "observation.sim.left_fingers": {
                "dtype": "float64",
                "shape": (400,),
            },
            "observation.sim.right_fingers": {
                "dtype": "float64",
                "shape": (400,),
            },
            "observation.sim.target_upper_body_pose": {
                "dtype": "float64",
                "shape": (len(robot_model.get_joint_group_indices("upper_body")),),
            },
            # TODO: support different reduced robot models
        }
    )

    features.pop("observation.img_state_delta")
    features.pop("observation.images.ego_view")

    if save_img_obs:
        for key, value in image_obs_configs.items():
            features.update(
                {
                    f"observation.images.{key.replace('_image', '')}": {
                        "dtype": "video",
                        "shape": value["shape"],
                        "names": ["height", "width", "channel"],
                    }
                }
            )

    return features


def generate_frame(
    obs: Dict[str, Any],
    wbc_action: Dict[str, Any],
    seed: int,
    mujoco_state: np.ndarray,
    mujoco_state_len: int,
    max_mujoco_state_len: int,
    teleop_cmd: Dict[str, Any],
    wbc_goal: Dict[str, Any],
    save_img_obs: bool = False,
):
    frame = {
        "observation.state": np.array(obs["q"], dtype=np.float64),
        "observation.eef_state": np.array(obs["wrist_pose"], dtype=np.float64),
        "action": np.array(wbc_action["q"], dtype=np.float64),
        "action.eef": np.array(wbc_goal["wrist_pose"], dtype=np.float64),
        "teleop.navigate_command": np.array(
            wbc_goal.get("navigate_cmd", DEFAULT_NAV_CMD), dtype=np.float64
        ),
        "teleop.base_height_command": np.array(
            teleop_cmd.get("base_height_command", [DEFAULT_BASE_HEIGHT]), dtype=np.float64
        ).reshape(
            1,
        ),
        "observation.sim.seed": np.array([seed], dtype=np.int32),
        "observation.sim.mujoco_state_len": np.array([mujoco_state_len], dtype=np.int32),
        "observation.sim.max_mujoco_state_len": np.array([max_mujoco_state_len], dtype=np.int32),
        "observation.sim.mujoco_state": mujoco_state.astype(np.float64),
        "observation.sim.left_wrist": teleop_cmd["left_wrist"].flatten().astype(np.float64),
        "observation.sim.right_wrist": teleop_cmd["right_wrist"].flatten().astype(np.float64),
        "observation.sim.left_fingers": teleop_cmd["left_fingers"]["position"]
        .flatten()
        .astype(np.float64),
        "observation.sim.right_fingers": teleop_cmd["right_fingers"]["position"]
        .flatten()
        .astype(np.float64),
        "observation.sim.target_upper_body_pose": wbc_goal["target_upper_body_pose"].astype(
            np.float64
        ),
        # "observation.sim.target_time": np.array([wbc_goal["target_time"]], dtype=np.float64),
        # "observation.sim.interpolation_garbage_collection_time": np.array(
        #     [wbc_goal["interpolation_garbage_collection_time"]], dtype=np.float64
        # ),
    }

    if save_img_obs:
        for key, value in obs.items():
            if key.endswith("image"):
                frame[f"observation.images.{key.replace('_image', '')}"] = value
    return frame


def get_data_exporter(
    config: SyncSimDataCollectionConfig,
    obs: Dict[str, Any],
    robot_model: RobotModel,
    save_path: Optional[pathlib.Path] = None,
) -> Gr00tDataExporter:
    if save_path is None:
        save_path = pathlib.Path(
            f"./outputs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{config.robot}-sim-{config.task_name}/"
        )

    if config.remove_existing_dir:
        if save_path.exists():
            shutil.rmtree(save_path)

    image_obs_configs = {}
    for key, value in obs.items():
        if key.endswith("image"):
            image_obs_configs[key] = {
                "dtype": "uint8",
                "shape": value.shape,
            }

    # TODO: use standardized keys for training dataset
    modality_config = get_modality_config(robot_model)
    exporter = Gr00tDataExporter.create(
        save_root=save_path,
        fps=config.control_frequency,
        features=get_features(
            robot_model, save_img_obs=config.save_img_obs, image_obs_configs=image_obs_configs
        ),
        modality_config=modality_config,
        task=config.task_name,
        script_config=config,
        robot_type=config.robot,
        vcodec="libx264",  # Use a common codec that should be available
    )
    return exporter


def get_env(config: SyncSimDataCollectionConfig, **kwargs) -> SyncEnv:
    robot_type, _ = get_robot_type_and_model(config.robot, enable_waist_ik=config.enable_waist)
    print("Instantiating environment:", config.env_name, config.robot)
    controller_configs = update_robosuite_controller_configs(
        robot=config.robot,
        wbc_version=config.wbc_version,
        enable_gravity_compensation=config.enable_gravity_compensation,
    )

    kwargs.update(
        {
            "ik_indicator": config.ik_indicator,
            "control_freq": config.control_frequency,
            "renderer": config.renderer,
            "controller_configs": controller_configs,
            "enable_waist": config.enable_waist,
            "enable_gravity_compensation": config.enable_gravity_compensation,
            "gravity_compensation_joints": config.gravity_compensation_joints,
        }
    )
    if robot_type == "g1":
        env_type = G1SyncEnv
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    env = env_type(
        env_name=config.task_name,  # TODO: should merge with config.env_name
        robot_name=config.robot,
        **kwargs,
    )
    return env


def get_env_name(robot: str, task_name: str, enable_waist_ik: bool = False) -> str:
    robot_type, _ = get_robot_type_and_model(robot, enable_waist_ik=enable_waist_ik)
    env_name = f"gr00tlocomanip_{robot_type}_sim/{task_name}_{robot}_Env"
    return env_name


def get_body_teleoped_joint_groups(robot: str) -> list[str]:
    robot2body_teleoped_joint_groups = {
        "G1FixedLowerBody": ["arms"],
        "G1FixedBase": ["arms"],
        "G1ArmsOnly": ["arms"],
        "G1": ["upper_body"],
    }
    return robot2body_teleoped_joint_groups[robot]


def get_teleop_policy(
    robot_type: str,
    robot_model: RobotModel,
    config: SyncSimDataCollectionConfig,
    activate_keyboard_listener: bool = True,
) -> TeleopPolicy:
    if robot_type == "g1":
        left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()
    else:
        raise ValueError(f"Invalid robot type: {robot_type}")

    # Initializing the teleop policy will block the main process until the Leap Motion is ready.
    retargeting_ik = TeleopRetargetingIK(
        robot_model=robot_model,
        left_hand_ik_solver=left_hand_ik_solver,
        right_hand_ik_solver=right_hand_ik_solver,
        enable_visualization=config.enable_visualization,
        body_active_joint_groups=get_body_teleoped_joint_groups(config.robot),
        body_ik_solver_settings_type=get_body_ik_solver_settings_type(config.robot),
    )
    teleop_policy = TeleopPolicy(
        robot_model=robot_model,
        retargeting_ik=retargeting_ik,
        body_control_device=config.body_control_device,
        hand_control_device=config.hand_control_device,
        body_streamer_ip=config.body_streamer_ip,
        body_streamer_keyword=config.body_streamer_keyword,
        enable_real_device=config.enable_real_device,
        replay_data_path=config.replay_data_path,
        replay_speed=config.replay_speed,
        activate_keyboard_listener=activate_keyboard_listener,
    )
    return teleop_policy


def get_wbc_config(config: SyncSimDataCollectionConfig):
    wbc_config = config.load_wbc_yaml()
    wbc_config["upper_body_policy_type"] = "identity"
    return wbc_config


def get_policies(
    config: SyncSimDataCollectionConfig,
    robot_type: str,
    robot_model: RobotModel,
    activate_keyboard_listener: bool = True,
):
    wbc_config = get_wbc_config(config)
    wbc_policy = get_wbc_policy(robot_type, robot_model, wbc_config, init_time=0.0)
    wbc_policy.activate_policy()
    teleop_policy = get_teleop_policy(robot_type, robot_model, config, activate_keyboard_listener)
    if not config.manual_control:
        teleop_policy.activate_policy()
    return wbc_policy, teleop_policy
