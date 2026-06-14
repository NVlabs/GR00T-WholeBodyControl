from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, Optional

import yaml

import decoupled_wbc
from decoupled_wbc.control.main.config_template import ArgsConfig as ArgsConfigTemplate
from decoupled_wbc.control.policy.wbc_policy_factory import WBC_VERSIONS
from decoupled_wbc.control.utils.network_utils import resolve_interface


def override_wbc_config(
    wbc_config: dict, config: "BaseConfig", missed_keys_only: bool = False
) -> dict:
    """Override WBC YAML values with dataclass values.

    Args:
        wbc_config: The loaded WBC YAML configuration dictionary
        config: The BaseConfig dataclass instance with override values
        missed_keys_only: If True, only add keys that don't exist in wbc_config.
                         If False, validate all keys exist and override all.

    Returns:
        Updated wbc_config dictionary with overridden values

    Raises:
        KeyError: If any required keys are missing from the WBC YAML configuration
                  (only when missed_keys_only=False)
    """
    # Override yaml values with dataclass values
    key_to_value = {
        "INTERFACE": config.interface,
        "ENV_TYPE": config.env_type,
        "VERSION": config.wbc_version,
        "SIMULATOR": config.simulator,
        "SIMULATE_DT": 1 / float(config.sim_frequency),
        "ENABLE_OFFSCREEN": config.enable_offscreen,
        "ENABLE_ONSCREEN": config.enable_onscreen,
        "model_path": config.wbc_model_path,
        "enable_waist": config.enable_waist,
        "with_hands": config.with_hands,
        "verbose": config.verbose,
        "verbose_timing": config.verbose_timing,
        "upper_body_max_joint_speed": config.upper_body_joint_speed,
        "keyboard_dispatcher_type": config.keyboard_dispatcher_type,
        "enable_gravity_compensation": config.enable_gravity_compensation,
        "gravity_compensation_joints": config.gravity_compensation_joints,
        "high_elbow_pose": config.high_elbow_pose,
    }

    if missed_keys_only:
        # Only add keys that don't exist in wbc_config
        for key in key_to_value:
            if key not in wbc_config:
                wbc_config[key] = key_to_value[key]
    else:
        # Set all keys (overwrite existing)
        for key in key_to_value:
            wbc_config[key] = key_to_value[key]

    # g1 kp, kd, sim2real gap
    if config.env_type == "real":
        # update waist pitch damping, index 14
        wbc_config["MOTOR_KD"][14] = wbc_config["MOTOR_KD"][14] - 10

    return wbc_config


@dataclass
class BaseConfig(ArgsConfigTemplate):
    """Base config inherited by all G1 control loops"""

    # WBC Configuration
    wbc_version: Literal[tuple(WBC_VERSIONS)] = "gear_wbc"
    """Version of the whole body controller."""

    wbc_model_path: str = (
        "policy/GR00T-WholeBodyControl-Balance.onnx," "policy/GR00T-WholeBodyControl-Walk.onnx"
    )
    """Path to WBC model file (relative to decoupled_wbc/sim2mujoco/resources/robots/g1)"""
    """gear_wbc model path: policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx"""

    wbc_policy_class: str = "G1DecoupledWholeBodyPolicy"
    """Whole body policy class."""

    # System Configuration
    interface: str = "sim"
    """Interface to use for the control loop. [sim, real, lo, enxe8ea6a9c4e09]"""

    simulator: str = "mujoco"
    """Simulator to use."""

    sim_sync_mode: bool = False
    """Whether to run the control loop in sync mode."""

    control_frequency: int = 50
    """Frequency of the control loop."""

    sim_frequency: int = 200
    """Frequency of the simulation loop."""

    # Robot Configuration
    enable_waist: bool = False
    """Whether to include waist joints in IK."""

    with_hands: bool = True
    """Enable hand functionality. When False, robot operates without hands."""

    high_elbow_pose: bool = False
    """Enable high elbow pose configuration for default joint positions."""

    verbose: bool = True
    """Whether to print verbose output."""

    # Additional common fields
    enable_offscreen: bool = False
    """Whether to enable offscreen rendering."""

    enable_onscreen: bool = True
    """Whether to enable onscreen rendering."""

    upper_body_joint_speed: float = 1000
    """Upper body joint speed."""

    env_name: str = "default"
    """Environment name."""

    ik_indicator: bool = False
    """Whether to draw IK indicators."""

    verbose_timing: bool = False
    """Enable verbose timing output every iteration."""

    keyboard_dispatcher_type: str = "raw"
    """Keyboard dispatcher to use. [raw, ros]"""

    # Gravity Compensation Configuration
    enable_gravity_compensation: bool = False
    """Enable gravity compensation using pinocchio dynamics."""

    gravity_compensation_joints: Optional[list[str]] = None
    """Joint groups to apply gravity compensation to (e.g., ['arms', 'left_arm', 'right_arm'])."""
    # Teleop/Device Configuration
    body_control_device: str = "dummy"
    """Device to use for body control. Options: dummy, vive, iphone, leapmotion, joycon."""

    hand_control_device: Optional[str] = "dummy"
    """Device to use for hand control. Options: None, manus, joycon, iphone."""

    body_streamer_ip: str = "10.112.210.229"
    """IP address for body streamer (vive only)."""

    body_streamer_keyword: str = "knee"
    """Body streamer keyword (vive only)."""

    enable_visualization: bool = False
    """Whether to enable visualization."""

    enable_real_device: bool = True
    """Whether to enable real device."""

    teleop_frequency: int = 20
    """Teleoperation frequency (Hz)."""

    teleop_replay_path: Optional[str] = None
    """Path to teleop replay data."""

    # Deployment/Camera Configuration
    robot_ip: str = "192.168.123.164"
    """Robot IP address"""
    # Data collection settings
    data_collection: bool = True
    """Enable data collection"""

    data_collection_frequency: int = 20
    """Data collection frequency (Hz)"""

    root_output_dir: str = "outputs"
    """Root output directory"""

    # Policy settings
    enable_upper_body_operation: bool = True
    """Enable upper body operation"""

    upper_body_operation_mode: Literal["teleop", "inference"] = "teleop"
    """Upper body operation mode"""

    def __post_init__(self):
        # Resolve interface (handles sim/real shortcuts, platform differences, and error handling)
        self.interface, self.env_type = resolve_interface(self.interface)

    def load_wbc_yaml(self) -> dict:
        """Load and merge wbc yaml with dataclass overrides"""
        # Get the base path to decoupled_wbc and convert to Path object
        package_path = Path(os.path.dirname(decoupled_wbc.__file__))

        if self.wbc_version == "gear_wbc":
            config_path = str(package_path / "control/main/teleop/configs/g1_29dof_gear_wbc.yaml")
        else:
            raise ValueError(
                f"Invalid wbc_version: {self.wbc_version}, please use one of: " f"gear_wbc"
            )

        with open(config_path) as file:
            wbc_config = yaml.load(file, Loader=yaml.FullLoader)

        # Override yaml values with dataclass values
        wbc_config = override_wbc_config(wbc_config, self)

        return wbc_config


@dataclass
class ControlLoopConfig(BaseConfig):
    """Config for running the G1 control loop."""

    pass


@dataclass
class TeleopConfig(BaseConfig):
    """Config for running the G1 teleop policy loop."""

    robot: Literal["g1"] = "g1"
    """Name of the robot to use, e.g., 'g1'."""

    lerobot_replay_path: Optional[str] = None
    """Path to lerobot replay data."""

    # Override defaults for teleop-specific values
    body_streamer_ip: str = "10.110.67.24"
    """IP address for body streamer (vive only)."""

    body_streamer_keyword: str = "foot"
    """Keyword for body streamer (vive only)."""

    teleop_frequency: float = 20  # Override to be float instead of int
    """Frequency of the teleop loop."""

    binary_hand_ik: bool = True
    """Whether to use binary IK."""


@dataclass
class ComposedCameraClientConfig:
    """Config for running the composed camera client."""

    camera_port: int = 5555
    """Port number"""

    camera_host: str = "localhost"
    """Host IP address"""

    fps: float = 20.0
    """FPS of the camera viewer"""


@dataclass
class DataExporterConfig(BaseConfig, ComposedCameraClientConfig):
    """Config for running the G1 data exporter."""

    dataset_name: Optional[str] = None
    """Name of the dataset to save the data to. If the dataset already exists,
    the new episodes will be appended to existing dataset. If the dataset does not exist,
    episodes will be saved under root_output_dir/dataset_name.
    """

    task_prompt: str = "demo"
    """Language Task prompt for the dataset."""

    state_dim: int = 43
    """Size of the state."""

    action_dim: int = 43
    """Size of the action."""

    teleoperator_username: Optional[str] = None
    """Teleoperator username."""

    support_operator_username: Optional[str] = None
    """Support operator username."""

    robot_id: Optional[str] = None
    """Robot ID."""

    lower_body_policy: Optional[str] = None
    """Lower body policy."""

    img_stream_viewer: bool = False
    """Whether to open a matplot lib window to view the camera images."""

    text_to_speech: bool = True
    """Whether to use text-to-speech for voice feedback."""

    add_stereo_camera: bool = True
    """Whether to add stereo camera for data collection. If False, only use a signle ego view camera."""


@dataclass
class SyncSimDataCollectionConfig(ControlLoopConfig, TeleopConfig):
    """Args Config for running the data collection loop."""

    robot: str = "G1"
    """Name of the robot to collect data for (e.g., G1 variants)."""

    task_name: str = "GroundOnly"
    """Name of the task to collect data for. [PnPBottle, GroundOnly, ...]"""

    body_control_device: str = "dummy"
    """Device to use for body control. Options: dummy, vive, iphone, leapmotion, joycon."""

    hand_control_device: Optional[str] = "dummy"
    """Device to use for hand control. Options: None, manus, joycon, iphone."""

    remove_existing_dir: bool = False
    """Whether to remove existing output directory if it exists."""

    hardcode_teleop_cmd: bool = False
    """Whether to hardcode the teleop command for testing purposes."""

    ik_indicator: bool = False
    """Whether to draw IK indicators."""

    enable_onscreen: bool = True
    """Whether to show the onscreen rendering."""

    save_img_obs: bool = False
    """Whether to save image observations."""

    success_hold_steps: int = 50
    """Number of steps to collect after task completion before saving."""

    renderer: Literal["mjviewer", "mujoco", "rerun"] = "mjviewer"
    """Renderer to use for the environment. """

    replay_data_path: str | None = None
    """Path to the data (.pkl) to replay. If None, will not replay. Used for CI/CD."""

    replay_speed: float = 2.5
    """Speed multiplier for replay data. Higher values make replay slower (e.g., 2.5 for sync sim tests)."""

    ci_test: bool = False
    """Whether to run the CI test."""

    ci_test_mode: Literal["unit", "pre_merge"] = "pre_merge"
    """'unit' for fast 50-step tests, 'pre_merge' for 500-step test with tracking checks."""

    manual_control: bool = False
    """Enable manual control of data collection start/save. When True, use toggle_data_collection
    to manually control episode states (idle -> recording -> need_to_save -> idle).
    When False (default), automatically starts and stops data collection based on task completion."""


@dataclass
class SyncSimPlaybackConfig(SyncSimDataCollectionConfig):
    """Configuration class for playback script arguments."""

    enable_real_device: bool = False
    """Whether to enable real device"""

    dataset: str | None = None
    """Path to the demonstration dataset, either an HDF5 file or a LeRobot folder path."""

    use_actions: bool = False
    """Whether to use actions for playback"""

    use_wbc_goals: bool = False
    """Whether to use WBC goals for control"""

    use_teleop_cmd: bool = False
    """Whether to use teleop IK for action generation"""

    # Video recording arguments.
    # Warning: enabling this key will leads to divergence between playback and recording.
    save_video: bool = False
    """Whether to save video of the playback"""

    # Saving to LeRobot dataset.
    # Warning: enabling this key will leads to divergence between playback and recording.
    save_lerobot: bool = False
    """Whether to save the playback as a new LeRobot dataset"""

    video_path: str | None = None
    """Path to save the output video. If not specified, 
       will use the nearest folder to dataset and save as playback_video.mp4"""

    num_episodes: int = 1
    """Number of episodes to load and playback/record (loads only the first N episodes from dataset)"""

    intervention: bool = False
    """Whether to denote intervention timesteps with colored borders in video frames"""

    ci_test: bool = False
    """Whether this is a CI test run, which limits the number of steps for testing purposes"""

    def validate_args(self):
        # Validate argument combinations
        if self.use_teleop_cmd and not self.use_actions:
            raise ValueError("--use-teleop-cmd requires --use-actions to be set")

        # Note: using teleop cmd has playback divergence unlike using wbc goals, as TeleopPolicy has a warmup loop
        if self.use_teleop_cmd and self.use_wbc_goals:
            raise ValueError("--use-teleop-cmd and --use-wbc-goals are mutually exclusive")

        if (self.use_teleop_cmd or self.use_wbc_goals) and not self.use_actions:
            raise ValueError(
                "You are using --use-teleop-cmd or --use-wbc-goals but not --use-actions. "
                "This will not play back actions whether via teleop or wbc goals. "
                "Instead, it'll play back states only."
            )

        if self.save_img_obs and not self.save_lerobot:
            raise ValueError("--save-img-obs is only supported with --save-lerobot")

        if self.intervention and not self.save_video:
            raise ValueError("--intervention requires --save-video to be enabled for visualization")


@dataclass
class WebcamRecorderConfig(BaseConfig):
    """Config for running the webcam recorder."""

    output_dir: str = "logs_experiment"
    """Output directory for webcam recordings"""

    device_id: int = 0
    """Camera device ID"""

    fps: int = 30
    """Recording frame rate"""

    duration: Optional[int] = None
    """Recording duration in seconds (None for continuous)"""


@dataclass
class SimLoopConfig(BaseConfig):
    """Config for running the simulation loop."""

    mp_start_method: str = "spawn"
    """Multiprocessing start method"""

    enable_image_publish: bool = False
    """Enable image publishing in simulation"""

    camera_port: int = 5555
    """Camera port for image publishing"""

    verbose: bool = False
    """Verbose output, override the base config verbose"""


@dataclass
class DeploymentConfig(BaseConfig, ComposedCameraClientConfig):
    """G1 Robot Deployment Configuration

    Simplified deployment config that inherits all common fields from G1BaseConfig.
    All deployment settings are now available in the base config.
    """

    camera_publish_rate: float = 30.0
    """Camera publish rate (Hz)"""

    view_camera: bool = True
    """Enable camera viewer"""
    # Webcam recording settings
    enable_webcam_recording: bool = True
    """Enable webcam recording for real robot deployment monitoring"""

    webcam_output_dir: str = "logs_experiment"
    """Output directory for webcam recordings"""

    skip_img_transform: bool = False
    """Skip image transformation in the model (for faster internet)"""

    sim_in_single_process: bool = False
    """Run simulator in a separate process. When True, sets simulator to None in main control loop
    and launches run_sim_loop.py separately."""

    image_publish: bool = False
    """Enable image publishing in simulation loop (passed to run_sim_loop.py)"""
