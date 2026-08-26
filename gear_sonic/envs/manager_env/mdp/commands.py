"""Motion tracking command terms for humanoid whole-body control RL environments."""

from __future__ import annotations

from collections.abc import Sequence
import copy
import dataclasses
import glob
import os
from typing import TYPE_CHECKING

import easydict
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import DEFORMABLE_TARGET_MARKER_CFG
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_yaw,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
)
import numpy as np
import torch

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.isaac_utils import rotations
from gear_sonic.trl.utils import common, order_converter, torch_transform
from gear_sonic.utils.motion_lib import motion_lib_robot

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Constants for multi-object mode: inactive objects placed far away
# Objects spread vertically (Z) since envs only vary in X,Y - much simpler!
INACTIVE_OBJECT_BASE_OFFSET = torch.tensor([1000.0, 0.0, -50.0])  # 1km away in X, 50m underground
INACTIVE_OBJECT_Z_SPACING = 10.0  # 10m vertical spacing between objects (must be > chair height)


def _init_variable_frames(
    enabled: bool,
    min_frames: int,
    num_future_frames: int,
    step: int,
    num_envs: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Initialize variable frame support tensors.

    Returns (per_env_num_frames, frame_choices) or (None, None) if disabled.
    """
    if not enabled:
        return None, None
    assert step > 0, f"variable_frames_step must be positive, got {step}"
    per_env_num_frames = torch.full((num_envs,), num_future_frames, device=device, dtype=torch.long)
    frame_choices = torch.arange(min_frames, num_future_frames + 1, step, device=device)
    assert len(frame_choices) > 0, (
        f"No valid frame choices: variable_frames_min={min_frames} "
        f"> num_future_frames={num_future_frames}"
    )
    return per_env_num_frames, frame_choices


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = None
    force = None


class TrackingCommand(CommandTerm):
    """Provide reference motion trajectories for motion-tracking RL.

    This is the primary command term for SONIC-style humanoid control. It manages
    a motion library of pre-recorded motion clips (robot joint trajectories, SMPL
    poses, and optionally object trajectories) and serves reference frames to the
    policy and reward system at each simulation step.

    Key responsibilities:
        - Load and index motion clips from the motion library (MotionLibRobot).
        - Sample motion IDs and start times for each environment at episode reset.
        - Advance the time cursor each step and provide current + multi-future
          reference frames (positions, orientations, velocities, joint states).
        - Transform reference quantities into robot-local, egocentric, or
          heading-canonicalized coordinate frames for observation terms.
        - Handle DOF mismatch between motion data and robot (e.g., extra finger
          joints not present in the motion library).
        - Support adaptive sampling, contact-based initialization, variable future
          frame counts, encoder mode sampling (G1/SMPL/teleop), and multi-object
          scene management.

    The command exposes ~60 property-based accessors that observation and reward
    terms query each step. Properties are named by the pattern::

        {quantity}[_multi_future][_dif][_{frame}]

    where ``quantity`` is body_pos, joint_pos, smpl_pose, etc.; ``multi_future``
    means values for all ``num_future_frames`` reference frames stacked;
    ``dif`` is the difference relative to the robot; and ``frame`` is ``w``
    (world), ``l`` (robot-local / de-headed), or ``b`` (body).
    """

    cfg: TrackingCommandCfg

    def __init__(
        self, cfg: TrackingCommandCfg, env: ManagerBasedRLEnv, max_num_load_motions: int = None  # noqa: RUF013
    ):
        """Initialize the tracking command with motion library and body mappings.

        Loads motion data, builds DOF/body index mappings between the motion
        library (MuJoCo ordering) and the IsaacLab robot, pre-allocates per-env
        tensors for motion IDs, time steps, and future frame offsets, and
        optionally sets up height-map raycasting.

        Args:
            cfg: Configuration dataclass specifying motion files, body names,
                future frame settings, encoder sampling probabilities, etc.
            env: The manager-based RL environment that owns this command.
            max_num_load_motions: Cap on how many unique motion clips to load
                into GPU memory. Defaults to ``min(num_envs, 1024)`` or the
                full library when ``use_paired_motions`` is enabled.
        """
        super().__init__(cfg, env)

        self.is_evaluating = False
        self.cmd_body_names = self.cfg.body_names
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)
        self.vr_3point_body_indices = [
            self.robot.body_names.index(name) for name in self.cfg.vr_3point_body
        ]
        self.vr_3point_body_indices_motion = [
            self.cfg.body_names.index(name) for name in self.cfg.vr_3point_body
        ]
        self.vr_3point_body_offsets = (
            torch.tensor(self.cfg.vr_3point_body_offset, dtype=torch.float32, device=self.device)
            .view(1, -1, 3)
            .repeat(self.num_envs, 1, 1)
        )

        self.reward_point_body_indices = [
            self.robot.body_names.index(name) for name in self.cfg.reward_point_body
        ]
        self.reward_point_body_offsets = (
            torch.tensor(self.cfg.reward_point_body_offset, dtype=torch.float32, device=self.device)
            .view(1, -1, 3)
            .repeat(self.num_envs, 1, 1)
        )
        self.reward_point_body_indices_motion = [
            self.cfg.body_names.index(name) for name in self.cfg.reward_point_body
        ]

        self.down_dir = (
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
            .view(1, -1)
            .repeat(self.num_envs, 1)
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        isaac_lab_joints = env.cfg.isaaclab_to_mujoco_mapping["isaaclab_joints"]

        self.isaaclab_to_mujoco_dof = env.cfg.isaaclab_to_mujoco_mapping["isaaclab_to_mujoco_dof"]
        self.mujoco_to_isaaclab_dof = env.cfg.isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_dof"]
        self.lower_joint_indices_mujoco = list(range(12))
        self.lower_joint_isaaclab_indices = [
            self.isaaclab_to_mujoco_dof[i] for i in self.lower_joint_indices_mujoco
        ]
        self.isaaclab_to_mujoco_body = env.cfg.isaaclab_to_mujoco_mapping["isaaclab_to_mujoco_body"]
        self.mujoco_to_isaaclab_body = env.cfg.isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_body"]
        self.running_ref_root_height = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.body_indexes_data = [isaac_lab_joints.index(name) for name in self.cfg.body_names]
        if self.cfg.motion_lib_cfg is not None:
            motion_lib_cfg = easydict.EasyDict(self.cfg.motion_lib_cfg)
        else:
            motion_lib_cfg = easydict.EasyDict(
                {
                    "motion_file": self.cfg.motion_file,
                    "smpl_motion_file": getattr(self.cfg, "smpl_motion_file", None),
                    "asset": {
                        "assetRoot": "gear_sonic/data/assets/robot_description/mjcf/",
                        "assetFileName": "g1_29dof_rev_1_0.xml",
                        "urdfFileName": "",
                    },
                    "extend_config": [],
                    "target_fps": 50,
                    "multi_thread": True,
                    "filter_motion_keys": self.cfg.filter_motion_keys,
                }
            )
        # Only override filter_motion_keys if explicitly set at command level
        # (don't overwrite the value from motion_lib_cfg if it exists there)
        filter_keys = self.cfg.filter_motion_keys
        if filter_keys is None:
            filter_keys = motion_lib_cfg.get("filter_motion_keys", None)

        motion_lib_cfg.update(
            {
                "mujoco_to_isaaclab_dof": self.mujoco_to_isaaclab_dof,
                "mujoco_to_isaaclab_body": self.mujoco_to_isaaclab_body,
                "isaaclab_to_mujoco_dof": self.isaaclab_to_mujoco_dof,
                "isaaclab_to_mujoco_body": self.isaaclab_to_mujoco_body,
                "body_indexes": self.body_indexes,
                "body_indexes_data": self.body_indexes_data,
                "filter_motion_keys": filter_keys,
                "lower_joint_indices_mujoco": self.lower_joint_indices_mujoco,
                "cat_upper_body_poses": self.cfg.cat_upper_body_poses,
                "cat_upper_body_poses_prob": self.cfg.cat_upper_body_poses_prob,
                "randomize_heading": self.cfg.randomize_heading,
                "freeze_frame_aug": self.cfg.freeze_frame_aug,
                "freeze_frame_aug_prob": self.cfg.freeze_frame_aug_prob,
                "randomize_wrist_poses": self.cfg.randomize_wrist_poses,
                "randomize_wrist_prob": self.cfg.randomize_wrist_prob,
                "randomize_wrist_std": self.cfg.randomize_wrist_std,
            }
        )

        self.motion_lib = motion_lib_robot.MotionLibRobot(
            motion_lib_cfg, self.num_envs, self.device
        )
        if max_num_load_motions is None:
            if self.cfg.use_paired_motions:
                self.max_num_load_motions = self.motion_lib._num_unique_motions  # noqa: SLF001
            else:
                self.max_num_load_motions = min(self.num_envs, 1024)
        else:
            self.max_num_load_motions = max_num_load_motions
        self.motion_lib.load_motions_for_training(max_num_seqs=self.max_num_load_motions)
        self.use_adaptive_sampling = self.motion_lib.use_adaptive_sampling

        # Load contact data for contact-based initialization
        self._load_contact_data()

        # Setup DOF mapping for handling mismatch between motion library and robot

        self.robot_num_dof = self.robot.num_joints
        self.motion_lib_num_dof = self.cfg.motion_lib_num_dof
        if self.motion_lib_num_dof is None:
            self.motion_lib_num_dof = self.robot_num_dof

        self.extra_num_dof = self.robot_num_dof - self.motion_lib_num_dof
        self.has_dof_mismatch = self.extra_num_dof > 0

        if self.has_dof_mismatch:
            self.body_joint_indices = joint_utils.get_body_joint_indices(self.robot)
            self.extra_joint_indices = joint_utils.get_hand_joint_indices(self.robot)
            self.extra_default_positions = torch.tensor(
                self.cfg.hand_default_positions or [0.0] * self.extra_num_dof,
                dtype=torch.float32,
                device=self.device,
            )
            self.extra_default_velocities = torch.tensor(
                self.cfg.hand_default_velocities or [0.0] * self.extra_num_dof,
                dtype=torch.float32,
                device=self.device,
            )

        # Step 1: Select which motions to use
        if self.cfg.use_paired_motions:
            # Assign motion IDs sequentially (wraps around if more envs than motions)
            self.motion_ids = (
                torch.arange(self.num_envs, device=self.device)
                % self.motion_lib._num_motions  # noqa: SLF001
            )
        elif getattr(self.cfg, "sample_unique_motions", False):
            # Sample without replacement - each env gets a unique motion
            num_available = len(self.motion_lib._curr_motion_ids)  # noqa: SLF001
            if self.num_envs > num_available:
                raise ValueError(
                    f"sample_unique_motions=True requires num_envs ({self.num_envs}) <= "
                    f"num_available_motions ({num_available})"
                )
            perm = torch.randperm(num_available, device=self.device)[: self.num_envs]
            self.motion_ids = perm
            print(  # noqa: T201
                f"[TrackingCommand] Sampled {self.num_envs} unique motions (no duplicates)"
            )
        else:
            # Random sampling (can have duplicates)
            self.motion_ids = self.motion_lib.sample_motions(self.num_envs)

        # Step 2: Sample start time steps for selected motions
        self.motion_start_time_steps = self.motion_lib.sample_time_steps(
            self.motion_ids, truncate_time=None
        )

        # # Debug: print motion assignments
        # motion_keys = self.motion_lib._motion_data_keys
        # for env_id in range(self.num_envs):
        #     motion_id = int(self.motion_ids[env_id].item())
        #     motion_key = motion_keys[motion_id]
        #     print(f"env {env_id}: motion_id={motion_id}, motion_key={motion_key}")

        # Step 3: Override start time steps if configured
        if self.cfg.sample_from_n_initial_frames is not None:
            # Sample uniformly from first N frames
            n_frames = self.cfg.sample_from_n_initial_frames
            self.motion_start_time_steps = torch.randint(
                0,
                n_frames,
                (self.num_envs,),
                dtype=self.motion_start_time_steps.dtype,
                device=self.device,
            )
        elif self.cfg.start_from_first_frame:
            self.motion_start_time_steps.zero_()
        self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Object position randomization offset (per-env, resampled at reset)
        self._object_position_offset = torch.zeros(self.num_envs, 3, device=self.device)

        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.num_future_frames = self.cfg.num_future_frames
        # Motion lib is at target_fps; ref frames are spaced by dt_future_ref_frames (seconds).
        # frame_skips = number of motion-lib steps between consecutive ref frames (integer).
        # Effective spacing equals dt_future_ref_frames only when (dt_future_ref_frames * target_fps) is an integer;  # noqa: E501
        # otherwise integer division truncates and the velocity calculation will be incorrect.
        self.frame_skips = self.cfg.dt_future_ref_frames // (1.0 / motion_lib_cfg.target_fps)
        steps_exact = self.cfg.dt_future_ref_frames * motion_lib_cfg.target_fps
        if abs(steps_exact - round(steps_exact)) > 1e-9:
            import warnings

            warnings.warn(
                f"dt_future_ref_frames={self.cfg.dt_future_ref_frames} * target_fps={motion_lib_cfg.target_fps} "
                f"= {steps_exact} is not an integer; "
                f"using frame_skips={self.frame_skips} so effective ref-frame spacing is "
                f"{self.frame_skips / motion_lib_cfg.target_fps:.4f}s (not {self.cfg.dt_future_ref_frames}s). "
                "Velocity-based losses may use incorrect dt.",
                stacklevel=2,
            )

        self.future_time_steps_init = (
            (
                torch.arange(self.num_future_frames, device=self.device, dtype=torch.long)
                * self.frame_skips
            )
            .view(1, -1)
            .repeat(self.num_envs, 1)
        )

        if self.cfg.smpl_num_future_frames is None:
            self.smpl_num_future_frames = self.num_future_frames
        else:
            self.smpl_num_future_frames = self.cfg.smpl_num_future_frames
        if self.cfg.smpl_dt_future_ref_frames is None:
            self.smpl_dt_future_ref_frames = self.cfg.dt_future_ref_frames
        else:
            self.smpl_dt_future_ref_frames = self.cfg.smpl_dt_future_ref_frames

        self.smpl_frame_skips = self.smpl_dt_future_ref_frames // (1.0 / motion_lib_cfg.target_fps)
        self.smpl_future_time_steps_init = (
            (
                torch.arange(self.smpl_num_future_frames, device=self.device, dtype=torch.long)
                * self.smpl_frame_skips
            )
            .view(1, -1)
            .repeat(self.num_envs, 1)
        )

        self.future_motion_ids = self.motion_ids.repeat_interleave(self.num_future_frames)
        self.smpl_future_motion_ids = self.motion_ids.repeat_interleave(self.smpl_num_future_frames)

        # Variable frame support
        self.variable_frames_enabled = getattr(self.cfg, "variable_frames_enabled", False)
        self.per_env_num_frames, self._frame_choices = _init_variable_frames(
            self.variable_frames_enabled,
            getattr(self.cfg, "variable_frames_min", 16),
            self.num_future_frames,
            getattr(self.cfg, "variable_frames_step", 4),
            self.num_envs,
            self.device,
        )

        self.encoder_sample_probs_dict = self.cfg.encoder_sample_probs
        self.optimize_encoders_ratio_for_CHIP = getattr(
            self.cfg, "optimize_encoders_ratio_for_CHIP", False
        )
        self.encoder_sample_probs = None
        if self.encoder_sample_probs_dict is not None:
            self.encoder_sample_probs = torch.tensor(
                list(self.encoder_sample_probs_dict.values()), device=self.device
            )
            self.encoder_sample_probs = self.encoder_sample_probs / self.encoder_sample_probs.sum()
            self.encoder_sample_probs_no_smpl_dict = copy.deepcopy(self.encoder_sample_probs_dict)

            if "smpl" in self.encoder_sample_probs_no_smpl_dict:
                self.encoder_sample_probs_no_smpl_dict["smpl"] = 0.0
            self.encoder_sample_probs_no_smpl = torch.tensor(
                list(self.encoder_sample_probs_no_smpl_dict.values()), device=self.device
            )
            self.encoder_sample_probs_no_smpl = (
                self.encoder_sample_probs_no_smpl / self.encoder_sample_probs_no_smpl.sum()
            )
            self.g1_encoder_index = list(self.encoder_sample_probs_dict.keys()).index("g1")
            if "smpl" in self.encoder_sample_probs_dict:
                self.smpl_encoder_index = list(self.encoder_sample_probs_dict.keys()).index("smpl")
            else:
                self.smpl_encoder_index = None

            if "teleop" in self.encoder_sample_probs_dict:
                self.teleop_encoder_index = list(self.encoder_sample_probs_dict.keys()).index(
                    "teleop"
                )
            else:
                self.teleop_encoder_index = None

            if "soma" in self.encoder_sample_probs_dict:
                self.soma_encoder_index = list(self.encoder_sample_probs_dict.keys()).index("soma")
                encoder_sample_probs_no_soma_dict = copy.deepcopy(self.encoder_sample_probs_dict)
                encoder_sample_probs_no_soma_dict["soma"] = 0.0
                self.encoder_sample_probs_no_soma = torch.tensor(
                    list(encoder_sample_probs_no_soma_dict.values()), device=self.device
                )
                no_soma_sum = self.encoder_sample_probs_no_soma.sum()
                if no_soma_sum > 0:
                    self.encoder_sample_probs_no_soma = (
                        self.encoder_sample_probs_no_soma / no_soma_sum
                    )
                else:
                    # All probs zero (e.g. use_encoder=soma forced) — fall back to G1
                    self.encoder_sample_probs_no_soma[self.g1_encoder_index] = 1.0
                encoder_sample_probs_no_smpl_no_soma_dict = copy.deepcopy(
                    self.encoder_sample_probs_no_smpl_dict
                )
                if "soma" in encoder_sample_probs_no_smpl_no_soma_dict:
                    encoder_sample_probs_no_smpl_no_soma_dict["soma"] = 0.0
                self.encoder_sample_probs_no_smpl_no_soma = torch.tensor(
                    list(encoder_sample_probs_no_smpl_no_soma_dict.values()), device=self.device
                )
                no_smpl_no_soma_sum = self.encoder_sample_probs_no_smpl_no_soma.sum()
                if no_smpl_no_soma_sum > 0:
                    self.encoder_sample_probs_no_smpl_no_soma = (
                        self.encoder_sample_probs_no_smpl_no_soma / no_smpl_no_soma_sum
                    )
                else:
                    # All probs zero — fall back to G1
                    self.encoder_sample_probs_no_smpl_no_soma[self.g1_encoder_index] = 1.0
            else:
                self.soma_encoder_index = None
                self.encoder_sample_probs_no_soma = None
                self.encoder_sample_probs_no_smpl_no_soma = None

            self.teleop_sample_prob_when_smpl = self.cfg.teleop_sample_prob_when_smpl

            self.encoder_index = torch.zeros(
                (self.num_envs, self.encoder_sample_probs.shape[0]),
                dtype=torch.long,
                device=self.device,
            )

            if "smpl" in self.encoder_sample_probs_dict:
                assert (
                    self.smpl_encoder_index > self.g1_encoder_index
                ), f"SMPL encoder index {self.smpl_encoder_index} must be greater than G1 encoder index {self.g1_encoder_index} to when both exist!"  # noqa: E501

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

        self.use_ref_motion_root_quat_w_as_anchor = False
        self.ref_motion_root_rotation_noise = None
        self.num_bodies = len(self.cfg.body_names)

        # Multi-object mode detection: check for multiple object_* entries in scene
        # (dynamically detected to avoid storing metadata on scene config)
        self._multi_object_mode = False
        self._object_names = []  # List of (safe_name, original_name) tuples
        self._active_object_name = None

        # Check if we have multiple objects (object_*) vs single object ("object")
        if hasattr(self._env, "scene") and hasattr(self._env.scene, "rigid_objects"):
            rigid_obj_keys = list(self._env.scene.rigid_objects.keys())
            multi_obj_keys = [k for k in rigid_obj_keys if k.startswith("object_")]
            if len(multi_obj_keys) > 0:
                self._multi_object_mode = True
                # Build object names list: (safe_name, original_name)
                # safe_name has underscores, original_name has hyphens (for motion lookup)
                for key in multi_obj_keys:
                    safe_name = key[7:]  # Remove "object_" prefix
                    original_name = safe_name.replace(  # noqa: F841
                        "_", "-"
                    )  # Convert back for motion lookup
                    # But we need to be careful - only convert the ones that were originally hyphens
                    # Actually we don't know which underscores were hyphens, so let's store just safe_name
                    # and try both when looking up motions
                    self._object_names.append(safe_name)
                print(  # noqa: T201
                    f"[Multi-Object Mode] Detected {len(self._object_names)} objects in scene"
                )

        # Cache for table metadata loaded from pkl files (loaded once per motion key)
        self._table_meta_cache = {}  # motion_key -> {'table_pos': tensor, 'table_quat': tensor}

        if self.cfg.use_height_map:
            try:
                from simple_raycaster.raycaster import MultiMeshRaycaster
            except ImportError:
                command_install = "pip install -e git+https://github.com/Agent-3154/simple-raycaster.git@197daa6dcb146c5ce3e675a173328e17df6b9777#egg=simple-raycaster"
                raise ImportError(  # noqa: B904
                    f"simple-raycaster is required for height map observation. Install with: {command_install}"
                )
            import omni

            prim_path_patterns = [
                # r"/World/ground",
                r"/World/envs/env_\d+/Object",
            ]
            stage = omni.usd.get_context().get_stage()

            self.height_map = MultiMeshRaycaster.from_prim_paths(
                paths=prim_path_patterns,
                stage=stage,
                device=str(self.device),
            )
            mesh_filters = [[f"/World/envs/env_{i}/Object"] for i in range(self.num_envs)]
            self.num_mesh_per_cam, self.mesh_ids, self.cam_ids = self.height_map.get_mesh_ids(
                mesh_filters, device=self.device
            )

            lin = torch.linspace(
                -0.5 * self.cfg.height_map_size,
                0.5 * self.cfg.height_map_size,
                int(self.cfg.height_map_size / self.cfg.height_map_resolution) + 1,
                device=self.device,
            )
            grid_x, grid_y = torch.meshgrid(lin, lin)
            zeros = torch.zeros_like(grid_x)
            scan_offsets = torch.stack(
                [grid_x, grid_y, zeros],
                dim=-1,
            ).view(-1, 3)
            scan_offsets[:, 2] = -1.0
            ray_dirs_local = torch.nn.functional.normalize(scan_offsets, dim=-1)
            self.ray_dirs_local = ray_dirs_local.expand(self.num_envs, -1, -1)
            self.num_rays = self.ray_dirs_local.shape[1]
            self.num_rays_x, self.num_rays_y = grid_x.shape
            self.scan_dot_pos_w = torch.zeros(
                self.num_envs, self.num_rays_x, self.num_rays_y, 3, device=self.device
            )

    # =========================================================================
    # Offline (kinematic-only) factory
    # =========================================================================

    @classmethod
    def create_offline(
        cls,
        motion_lib_cfg,
        device,
    ):
        """Create a MotionCommand without env/sim for offline kinematic use.

        All @property methods that only use motion_lib data work unchanged.
        Properties that require robot state (self.robot) will use the reference
        motion root as the anchor (equivalent to use_ref_motion_root_quat_w_as_anchor=True).

        Expected keys in motion_lib_cfg:
            num_future_frames: Number of future reference frames.
            dt_future_ref_frames: Time delta between future reference frames (seconds).
        """
        inst = object.__new__(cls)
        inst._offline = True  # noqa: SLF001
        inst._debug_vis_handle = None  # noqa: SLF001
        num_envs = 1

        # device / num_envs are read-only properties on ManagerTermBase that
        # delegate to self._env, so we provide a lightweight stand-in.
        class _MinimalEnv:
            pass

        _env = _MinimalEnv()
        _env.device = device
        _env.num_envs = num_envs
        inst._env = _env  # noqa: SLF001

        # Motion lib setup
        motion_lib_cfg = (
            easydict.EasyDict(motion_lib_cfg)
            if not isinstance(motion_lib_cfg, easydict.EasyDict)
            else motion_lib_cfg
        )

        num_future_frames = motion_lib_cfg.get("num_future_frames", 8)
        dt_future_ref_frames = motion_lib_cfg.get("dt_future_ref_frames", 0.1)

        # Inject body/DOF mapping into motion_lib_cfg so motion_lib handles
        # body reordering and xyzw→wxyz quaternion conversion at load time.

        isaaclab_to_mujoco_mapping = order_converter.G1Converter().get_isaaclab_to_mujoco_mapping()
        motion_lib_cfg.update(
            {
                "mujoco_to_isaaclab_body": isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_body"],
                "mujoco_to_isaaclab_dof": isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_dof"],
                "isaaclab_to_mujoco_body": isaaclab_to_mujoco_mapping["isaaclab_to_mujoco_body"],
                "isaaclab_to_mujoco_dof": isaaclab_to_mujoco_mapping["isaaclab_to_mujoco_dof"],
            }
        )
        # body_indexes_data: same logic as __init__ — use body_names to select
        # a subset when provided, otherwise default to all bodies.
        isaaclab_joints = isaaclab_to_mujoco_mapping["isaaclab_joints"]
        body_names = motion_lib_cfg.get("body_names", None)
        if "body_indexes_data" not in motion_lib_cfg:
            if body_names is not None:
                motion_lib_cfg.body_indexes_data = [
                    isaaclab_joints.index(name) for name in body_names
                ]
            else:
                num_bodies_full = len(isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_body"])
                motion_lib_cfg.body_indexes_data = list(range(num_bodies_full))

        inst.motion_lib = motion_lib_robot.MotionLibRobot(motion_lib_cfg, num_envs, device)
        max_num_motions = motion_lib_cfg.get("max_num_motions", None)
        inst.motion_lib.load_motions_for_training(max_num_seqs=max_num_motions)

        # Timing
        target_fps = motion_lib_cfg.get("target_fps", 50)
        inst.num_future_frames = num_future_frames
        inst.frame_skips = dt_future_ref_frames // (1.0 / target_fps)
        steps_exact = dt_future_ref_frames * target_fps
        if abs(steps_exact - round(steps_exact)) > 1e-9:
            import warnings

            warnings.warn(
                f"dt_future_ref_frames={dt_future_ref_frames} * target_fps={target_fps} "
                f"= {steps_exact} is not an integer; using frame_skips={inst.frame_skips}",
                stacklevel=2,
            )
        inst.future_time_steps_init = (
            (torch.arange(num_future_frames, device=device, dtype=torch.long) * inst.frame_skips)
            .view(1, -1)
            .repeat(num_envs, 1)
        )
        # Anchor body index (from the body_names list, matching __init__)
        anchor_body = motion_lib_cfg.get("anchor_body", None)
        if body_names is not None and anchor_body is not None:
            inst.motion_anchor_body_index = body_names.index(anchor_body)
        else:
            inst.motion_anchor_body_index = 0  # Default to root
        inst.num_bodies = len(motion_lib_cfg.body_indexes_data)

        # DOF mapping
        if isaaclab_to_mujoco_mapping is not None:
            inst.isaaclab_to_mujoco_dof = isaaclab_to_mujoco_mapping["isaaclab_to_mujoco_dof"]
            inst.mujoco_to_isaaclab_dof = isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_dof"]
        inst.lower_joint_indices_mujoco = list(range(12))
        if hasattr(inst, "isaaclab_to_mujoco_dof"):
            inst.lower_joint_isaaclab_indices = [
                inst.isaaclab_to_mujoco_dof[i] for i in inst.lower_joint_indices_mujoco
            ]

        # Variable frame support (offline)
        inst.variable_frames_enabled = motion_lib_cfg.get("variable_frames_enabled", False)
        inst.per_env_num_frames, inst._frame_choices = _init_variable_frames(  # noqa: SLF001
            inst.variable_frames_enabled,
            motion_lib_cfg.get("variable_frames_min", 16),
            num_future_frames,
            motion_lib_cfg.get("variable_frames_step", 4),
            num_envs,
            device,
        )

        # Motion state (updated per-sample via set_motion_state)
        inst.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        inst.time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        inst.motion_start_time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        inst.future_motion_ids = inst.motion_ids.repeat_interleave(num_future_frames)
        inst.motion_num_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Offline mode: use ref motion root as robot anchor
        inst.use_ref_motion_root_quat_w_as_anchor = True
        inst.ref_motion_root_rotation_noise = None

        # SMPL future frames (may differ from regular future frames)
        smpl_num_future_frames = motion_lib_cfg.get("smpl_num_future_frames", None)
        smpl_dt_future_ref_frames = motion_lib_cfg.get("smpl_dt_future_ref_frames", None)
        inst.smpl_num_future_frames = (
            smpl_num_future_frames if smpl_num_future_frames is not None else num_future_frames
        )
        inst.smpl_dt_future_ref_frames = (
            smpl_dt_future_ref_frames
            if smpl_dt_future_ref_frames is not None
            else dt_future_ref_frames
        )
        inst.smpl_frame_skips = inst.smpl_dt_future_ref_frames // (1.0 / target_fps)
        smpl_steps_exact = inst.smpl_dt_future_ref_frames * target_fps
        if abs(smpl_steps_exact - round(smpl_steps_exact)) > 1e-9:
            import warnings

            warnings.warn(
                f"smpl_dt_future_ref_frames={inst.smpl_dt_future_ref_frames} * target_fps={target_fps} "
                f"= {smpl_steps_exact} is not an integer; using smpl_frame_skips={inst.smpl_frame_skips}",
                stacklevel=2,
            )
        inst.smpl_future_time_steps_init = (
            (
                torch.arange(inst.smpl_num_future_frames, device=device, dtype=torch.long)
                * inst.smpl_frame_skips
            )
            .view(1, -1)
            .repeat(num_envs, 1)
        )
        inst.smpl_future_motion_ids = inst.motion_ids.repeat_interleave(inst.smpl_num_future_frames)

        # Encoder sampling (not used offline, but set for compat)
        inst.encoder_sample_probs_dict = None
        inst.encoder_sample_probs = None
        inst.is_evaluating = False

        # Metrics dict (normally set by CommandTerm.__init__)
        inst.metrics = {}

        return inst

    def set_is_evaluating(self, is_evaluating: bool = True):
        """Toggle evaluation mode, which disables reset randomizations."""
        self.is_evaluating = is_evaluating

    def forward_motion_samples(self, env_ids: Sequence[int]):
        """Assign sequential motion IDs and reset time steps for given envs.

        Used during paired/evaluation mode to deterministically cycle through
        motions. Updates motion IDs, start times, motion lengths, and caches
        the initial body pose for relative-frame computation.

        Args:
            env_ids: Environment indices to reassign motions for.
        """
        self.motion_ids[env_ids] = (
            torch.arange(self.num_envs).to(self.device)
            % self.motion_lib._num_motions  # noqa: SLF001
        )[env_ids]
        sampled_times = self.motion_lib.sample_time_steps(
            self.motion_ids[env_ids], truncate_time=None
        )
        if self.cfg.sample_from_n_initial_frames is not None:
            # Sample uniformly from first N frames
            n_frames = self.cfg.sample_from_n_initial_frames
            sampled_times = torch.randint(
                0, n_frames, (len(env_ids),), dtype=sampled_times.dtype, device=self.device
            )
        elif self.cfg.start_from_first_frame:
            sampled_times.zero_()
        self.motion_start_time_steps[env_ids] = sampled_times
        self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(
            self.motion_ids[env_ids]
        )
        self.time_steps[env_ids] = 0
        self.body_pos_relative_w[env_ids] = self.motion_lib.get_body_pos_w(
            self.motion_ids[env_ids], self.motion_start_time_steps[env_ids]
        )
        self.body_quat_relative_w[env_ids] = self.motion_lib.get_body_quat_w(
            self.motion_ids[env_ids], self.motion_start_time_steps[env_ids]
        )

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation  # noqa: TD002, TD003, TD004
        """Return current-frame joint positions and velocities concatenated.

        Returns:
            Tensor of shape ``(num_envs, 2 * num_dof)``.
        """
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def command_z(self) -> torch.Tensor:
        """Return reference root height (z) for the current frame.

        Returns:
            Tensor of shape ``(num_envs, 1)``.
        """
        return self.root_z

    @property
    def command_z_multi_future(self) -> torch.Tensor:
        """Return reference root height (z) for the first future frame.

        Returns:
            Tensor of shape ``(num_envs, 1)``.
        """
        return self.root_z_multi_future

    @property
    def command_vel(self) -> torch.Tensor:  # TODO Consider again if this is the best observation  # noqa: TD002, TD003, TD004
        """Return reference root velocity (2D linear + 1D angular) in body frame.

        Returns:
            Tensor of shape ``(num_envs, 3)``.
        """
        return torch.cat([self.root_lin_vel_b_2d, self.root_ang_vel_b_1d], dim=1)

    @property
    def command_max(self) -> torch.Tensor:
        """Return all body state (pos, quat, lin_vel, ang_vel) flattened.

        Returns:
            Tensor of shape ``(num_envs, num_bodies * 13)``.
        """
        return torch.cat(
            [self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w], dim=-1
        ).view(self.num_envs, -1)

    @property
    def command_max(self) -> torch.Tensor:  # noqa: F811
        """Return all body state (pos, quat, lin_vel, ang_vel) flattened.

        Returns:
            Tensor of shape ``(num_envs, num_bodies * 13)``.
        """
        return torch.cat(
            [self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w], dim=-1
        ).view(self.num_envs, -1)

    @property
    def command_max_diff_l(self) -> torch.Tensor:
        """Return body state differences in robot-local frame, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_bodies * 13)``.
        """
        return torch.cat(
            [
                self.body_pos_dif_l.view(self.num_envs, -1),
                self.body_quat_dif_l.view(self.num_envs, -1),
                self.body_lin_vel_l,
                self.body_ang_vel_l,
            ],
            dim=-1,
        ).view(self.num_envs, -1)

    @property
    def command_max_diff_l_multi_future(self) -> torch.Tensor:
        """Return multi-future body state diffs and local poses concatenated.

        Returns:
            Tensor of shape ``(num_envs, <variable>)`` depending on body count and future frames.
        """
        # TODO: this is not done.  # noqa: TD002, TD003
        return torch.cat(
            [
                self.body_pos_dif_l_multi_future.view(self.num_envs, -1),
                self.body_quat_dif_l_multi_future.view(self.num_envs, -1),
                self.body_lin_vel_dif_l_multi_future,
                self.body_ang_vel_dif_l_multi_future,
                self.bod_pos_local_multi_future,
                self.body_quat_local_multi_future,
                self.body_lin_vel_l_multi_future,
                self.body_ang_vel_l_multi_future,
            ],
            dim=-1,
        ).view(self.num_envs, -1)

    @property
    def command_max_multi_future(self) -> torch.Tensor:
        """Return all body state for all future frames in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_bodies * 13)``.
        """
        return torch.cat(
            [
                self.body_pos_w_multi_future,
                self.body_quat_w_multi_future,
                self.body_lin_vel_w_multi_future,
                self.body_ang_vel_w_multi_future,
            ],
            dim=1,
        )

    @property
    def command_multi_future(self) -> torch.Tensor:
        """Return joint positions and velocities for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, 2 * num_future_frames * num_dof)``.
        """
        return torch.cat([self.joint_pos_multi_future, self.joint_vel_multi_future], dim=1)

    @property
    def command_multi_future_joint_pos(self) -> torch.Tensor:
        """Return joint positions for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof)``.
        """
        return self.joint_pos_multi_future

    @property
    def command_multi_future_joint_body_pos(self) -> torch.Tensor:
        """Return joint positions and anchor-relative body positions for all future frames.

        Body positions are expressed relative to each frame's anchor position,
        rotated into the anchor's yaw-inverse frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof + num_future_frames * num_bodies * 3)``.
        """
        body_pos_multi_frame = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, self.num_bodies, 3)
        anchor_pos_multi_frame = (
            self.motion_lib.get_body_pos_w(self.future_motion_ids, self.future_time_steps)[
                :, self.motion_anchor_body_index
            ]
            .view(self.num_envs, self.num_future_frames, 1, 3)
            .expand(self.num_envs, self.num_future_frames, self.num_bodies, 3)
        )
        body_pos_relative_to_anchor_multi_frame = body_pos_multi_frame - anchor_pos_multi_frame
        anchor_quat_w_repeat = self.anchor_quat_w_multi_future.view(
            self.num_envs, self.num_future_frames, 1, 4
        ).expand(self.num_envs, self.num_future_frames, self.num_bodies, 4)
        body_pos_multi_frame = quat_apply_yaw(
            quat_inv(anchor_quat_w_repeat), body_pos_relative_to_anchor_multi_frame
        ).reshape(self.num_envs, -1)
        return torch.cat([self.joint_pos_multi_future, body_pos_multi_frame], dim=1)

    @property
    def command_multi_future_joint_body_abs_pos(self) -> torch.Tensor:
        """Return joint positions and anchor-relative body positions (without rotation).

        Unlike ``command_multi_future_joint_body_pos``, the body position deltas
        are not rotated into the anchor's local frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof + num_future_frames * num_bodies * 3)``.
        """
        body_pos_multi_frame = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).reshape(self.num_envs, -1)
        anchor_pos_multi_frame = (
            self.motion_lib.get_body_pos_w(self.future_motion_ids, self.future_time_steps)[
                :, self.motion_anchor_body_index
            ]
            .view(self.num_envs, self.num_future_frames, 1, 3)
            .expand(self.num_envs, self.num_future_frames, self.num_bodies, 3)
            .reshape(self.num_envs, -1)
        )
        body_pos_relative_to_anchor_multi_frame = body_pos_multi_frame - anchor_pos_multi_frame
        return torch.cat(
            [self.joint_pos_multi_future, body_pos_relative_to_anchor_multi_frame], dim=1
        )

    # @property
    # def command_multi_future_joint_body_diff_pos(self) -> torch.Tensor:
    #     body_pos_w = self.motion_lib.get_body_pos_w(self.future_motion_ids, self.future_time_steps).view(self.num_envs, self.num_future_frames, -1, 3)  # noqa: E501
    #     body_pos_w_env = body_pos_w + self._env.scene.env_origins[:, None, None, :]
    #     body_pos_dif = body_pos_w_env - self.robot_body_pos_w[:, None, :, :]
    #     body_pos_relative_to_robot_anchor_multi_frame = quat_apply_yaw(quat_inv(self.robot_anchor_quat_w.repeat(1, self.num_future_frames, self.num_bodies, 1)), body_pos_dif).reshape(self.num_envs, -1)  # noqa: E501
    #     return torch.cat([self.joint_pos_multi_future, body_pos_relative_to_robot_anchor_multi_frame], dim=1)

    @property
    def command_multi_future_joint_body_diff_pos(self) -> torch.Tensor:
        """Return joint positions and robot-relative body position differences.

        Computes body positions in robot-anchor-relative frame: positions are
        translated so the XY comes from the robot root and Z from the reference
        anchor, then rotated by the heading difference. The result is the
        difference from the current robot body positions, expressed in the
        robot's local frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof + num_future_frames * num_bodies * 3)``.
        """
        N, F, B = self.num_envs, self.num_future_frames, self.num_bodies
        anchor_pos_w_repeat = self.anchor_pos_w_multi_future.view(N, F, 1, 3).expand(N, F, B, 3)
        anchor_quat_w_repeat = self.anchor_quat_w_multi_future.view(N, F, 1, 4).expand(N, F, B, 4)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, None, :].expand(N, F, B, 3)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, None, :].expand(N, F, B, 4)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()  # Root position of the robot
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = torch_transform.get_heading_q(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )
        body_pos_relative_w_multi_frame = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w_multi_future.view(N, F, B, 3) - anchor_pos_w_repeat
        )
        body_pos_dif = body_pos_relative_w_multi_frame - self.robot_body_pos_w.view(
            N, 1, B, 3
        ).expand(N, F, B, 3)
        body_pos_relative_to_robot_anchor_multi_frame = quat_apply_yaw(
            quat_inv(robot_anchor_quat_w_repeat), body_pos_dif
        ).reshape(N, -1)
        return torch.cat(
            [self.joint_pos_multi_future, body_pos_relative_to_robot_anchor_multi_frame], dim=1
        )

    @property
    def command_multi_future_lower_body(self) -> torch.Tensor:
        return torch.cat(
            [self.joint_pos_lower_body_multi_future, self.joint_vel_lower_body_multi_future], dim=1
        )

    @property
    def command_multi_future_lower_body_joint_pos(self) -> torch.Tensor:
        return self.joint_pos_lower_body_multi_future

    # =========================================================================
    # Egocentric joint transforms (positions + rotations) for reference frames
    # =========================================================================

    @property
    def num_bodies_full(self) -> int:
        """Get the full number of bodies (all bodies, not just selected body_indexes)."""
        return self.motion_lib.num_bodies_full

    @property
    def egocentric_joint_positions_multi_future(self) -> torch.Tensor:
        """Get body positions (including pelvis) in egocentric frame for all future frames.

        Egocentric frame = projected root frame (heading/yaw only rotation, z=0 projected).
        For each future frame, positions are relative to that frame's projected root.

        Returns:
            torch.Tensor: [num_envs, num_future_frames, num_bodies_full, 3] joint positions
                          in egocentric (projected root) frame
        """
        N, F, B = self.num_envs, self.num_future_frames, self.num_bodies_full

        # Get full body positions in world frame for all future frames
        body_pos_w = self.motion_lib.get_body_pos_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 3)

        # Get anchor (pelvis) positions for each future frame
        anchor_body_idx_full = self.motion_lib.m_cfg.get("anchor_body_idx_full", 0)
        anchor_pos_w = body_pos_w[:, :, anchor_body_idx_full, :]  # [N, F, 3]

        # Project anchor to ground plane (z=0) for egocentric frame
        anchor_pos_projected = anchor_pos_w.clone()
        anchor_pos_projected[..., 2] = 0  # Set z to 0
        anchor_pos_projected_expanded = anchor_pos_projected.view(N, F, 1, 3).expand(N, F, B, 3)

        # Get anchor quaternions for each future frame
        body_quat_w = self.motion_lib.get_body_quat_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 4)
        anchor_quat_w = body_quat_w[:, :, anchor_body_idx_full, :]  # [N, F, 4]

        # Extract heading quaternion using get_heading_q (canonicalize)
        anchor_heading_quat = torch_transform.get_heading_q(anchor_quat_w.reshape(-1, 4)).reshape(
            N, F, 4
        )
        anchor_heading_quat_expanded = anchor_heading_quat.view(N, F, 1, 4).expand(N, F, B, 4)

        # Compute egocentric positions:
        # 1. Translate to projected anchor origin (z=0)
        body_pos_relative = body_pos_w - anchor_pos_projected_expanded

        # 2. Rotate by inverse of heading quaternion
        body_pos_egocentric = quat_apply(
            quat_inv(anchor_heading_quat_expanded.reshape(-1, 4)), body_pos_relative.reshape(-1, 3)
        ).reshape(N, F, B, 3)

        return body_pos_egocentric

    @property
    def egocentric_joint_rotations_multi_future(self) -> torch.Tensor:
        """Get body rotations (including pelvis) in egocentric frame for all future frames.

        Returns 6D rotation representation (first two columns of rotation matrix).
        For each future frame, rotations are relative to that frame's projected root.

        Returns:
            torch.Tensor: [num_envs, num_future_frames, num_bodies_full, 6] joint rotations
                          in 6D representation (first two columns of rotation matrix)
        """
        N, F, B = self.num_envs, self.num_future_frames, self.num_bodies_full

        # Get full body quaternions in world frame for all future frames
        body_quat_w = self.motion_lib.get_body_quat_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 4)

        # Get anchor quaternions for each future frame
        anchor_body_idx_full = self.motion_lib.m_cfg.get(
            "anchor_body_idx_full", 0
        )  # pelvis is usually index 0
        anchor_quat_w = body_quat_w[:, :, anchor_body_idx_full, :]  # [N, F, 4]

        # Extract heading quaternion using get_heading_q (canonicalize)
        anchor_heading_quat = torch_transform.get_heading_q(anchor_quat_w.reshape(-1, 4)).reshape(
            N, F, 4
        )
        anchor_heading_quat_expanded = anchor_heading_quat.view(N, F, 1, 4).expand(N, F, B, 4)

        # Compute relative rotation: q_relative = q_heading_inv * q_body
        body_quat_egocentric = quat_mul(
            quat_inv(anchor_heading_quat_expanded.reshape(-1, 4)), body_quat_w.reshape(-1, 4)
        ).reshape(N, F, B, 4)

        # Convert to 6D representation (first two columns of rotation matrix)
        mat = matrix_from_quat(body_quat_egocentric)  # [N, F, B, 3, 3]
        body_rot_6d = rotations.mat_to_rot6d_first_two_cols(mat).reshape(N, F, B, 6)  # [N, F, B, 6]

        return body_rot_6d

    @property
    def egocentric_joint_transforms_multi_future(self) -> torch.Tensor:
        """Combined egocentric joint transforms (positions + rotations) for all future frames.

        Returns:
            torch.Tensor: [num_envs, num_future_frames, num_bodies_full, 9]
                          (3 for position + 6 for 6D rotation)
        """
        positions = self.egocentric_joint_positions_multi_future  # [N, F, B, 3]
        rotations_ = self.egocentric_joint_rotations_multi_future  # [N, F, B, 6]
        return torch.cat([positions, rotations_], dim=-1)

    @property
    def root_transforms_relative_to_first_frame(self) -> torch.Tensor:
        """Get root transforms (position + rotation) relative to the first reference frame's
        projected root (z=0, heading only).

        Position: Delta from first frame's projected root position, rotated to first frame's heading frame.
        Rotation: Relative rotation from first frame's heading quaternion (6D representation).

        Returns:
            torch.Tensor: [num_envs, num_future_frames, 9]
                          (3 for relative position + 6 for 6D relative rotation)
        """  # noqa: D205
        N, F = self.num_envs, self.num_future_frames

        # Get root positions for all future frames
        root_pos_w = self.motion_lib.get_root_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, 3)

        # Get root quaternions for all future frames
        root_quat_w = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, 4)

        # First frame as reference - use projected root (z=0)
        first_frame_pos = root_pos_w[:, 0:1, :]  # [N, 1, 3]
        first_frame_pos_projected = first_frame_pos.clone()
        first_frame_pos_projected[..., 2] = 0  # Project to ground plane

        # Extract heading quaternion using get_heading_q (canonicalize)
        first_frame_quat = root_quat_w[:, 0:1, :]  # [N, 1, 4]
        first_frame_heading = torch_transform.get_heading_q(
            first_frame_quat.reshape(-1, 4)
        ).reshape(N, 1, 4)

        # Relative position: delta from projected first frame, rotated to heading frame
        delta_pos_w = root_pos_w - first_frame_pos_projected.expand(N, F, 3)

        # Rotate delta position by inverse of first frame's heading
        delta_pos_local = quat_apply(
            quat_inv(first_frame_heading.expand(N, F, 4).reshape(-1, 4)), delta_pos_w.reshape(-1, 3)
        ).reshape(N, F, 3)

        # Relative rotation: q_relative = q_heading_inv * q_current
        relative_quat = quat_mul(
            quat_inv(first_frame_heading.expand(N, F, 4).reshape(-1, 4)), root_quat_w.reshape(-1, 4)
        ).reshape(N, F, 4)

        # Convert to 6D representation (first two columns of rotation matrix)
        mat = matrix_from_quat(relative_quat)  # [N, F, 3, 3]
        relative_rot_6d = rotations.mat_to_rot6d_first_two_cols(mat).reshape(N, F, 6)  # [N, F, 6]

        return torch.cat([delta_pos_local, relative_rot_6d], dim=-1)

    @property
    def smpl_joints(self) -> torch.Tensor:
        """Return SMPL joint positions for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_smpl_joints, 3)``.
        """
        return self.motion_lib.get_smpl_joints(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def smpl_joints_multi_future(self) -> torch.Tensor:
        """Return SMPL joint positions for all SMPL future frames.

        Returns:
            Tensor of shape ``(num_envs, smpl_num_future_frames, num_smpl_joints, 3)``.
        """
        smpl_joints_mf = self.motion_lib.get_smpl_joints(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return smpl_joints_mf.view(
            self.num_envs, self.smpl_num_future_frames, *smpl_joints_mf.shape[1:]
        )

    @property
    def smpl_transl_multi_future(self) -> torch.Tensor:
        smpl_transl_mf = self.motion_lib.get_smpl_transl(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return smpl_transl_mf.view(
            self.num_envs, self.smpl_num_future_frames, *smpl_transl_mf.shape[1:]
        )

    @property
    def smpl_transl_z_multi_future(self) -> torch.Tensor:
        smpl_transl_mf = self.motion_lib.get_smpl_transl(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        # Only return y dimension (height) --- y is the z for smpl
        return smpl_transl_mf[..., 1:2].view(self.num_envs, self.smpl_num_future_frames, 1)

    @property
    def smpl_pose(self) -> torch.Tensor:
        """Return full SMPL pose (root + body joints) in axis-angle for current frame.

        Returns:
            Tensor of shape ``(num_envs, 72)`` (24 joints * 3 axis-angle).
        """
        return self.motion_lib.get_smpl_pose(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def smpl_body_pose(self) -> torch.Tensor:
        """Return SMPL body pose (excluding root) in axis-angle for current frame.

        Returns:
            Tensor of shape ``(num_envs, 69)`` (23 body joints * 3 axis-angle).
        """
        return self.motion_lib.get_smpl_pose(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[..., 3:]

    @property
    def smpl_body_pose_6d(self) -> torch.Tensor:
        """Return SMPL body pose in 6D rotation representation for current frame.

        Returns:
            Tensor of shape ``(num_envs, 23 * 6)``.
        """
        smpl_body_pose = self.smpl_body_pose.reshape(-1, 23, 3)
        quat = torch_transform.angle_axis_to_quaternion(smpl_body_pose)
        mat = matrix_from_quat(quat)
        smpl_body_pose_6d = mat[..., :2].reshape(smpl_body_pose.shape[0], -1)
        return smpl_body_pose_6d

    def smpl_root_ytoz_up(self, root_quat_y_up: torch.Tensor) -> torch.Tensor:
        """Convert SMPL root quaternion from Y-up to Z-up convention.

        Args:
            root_quat_y_up: Root quaternions in Y-up frame, ``(N, 4)``.

        Returns:
            Root quaternions rotated to Z-up frame, ``(N, 4)``.
        """
        base_rot = torch_transform.angle_axis_to_quaternion(
            torch.tensor([[np.pi / 2, 0.0, 0.0]]).to(root_quat_y_up)
        )
        root_quat_z_up = rotations.quat_mul(
            base_rot.repeat(root_quat_y_up.shape[0], 1), root_quat_y_up, w_last=False
        )
        return root_quat_z_up

    @property
    def smpl_pose_noheading(self) -> torch.Tensor:
        """Return SMPL pose with root heading removed (preserving pitch/roll).

        Converts SMPL root from Y-up if needed, removes the base rotation offset,
        extracts and removes heading, then re-encodes as axis-angle.

        Returns:
            Tensor of shape ``(num_envs, 72)``.
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_smpl_base_rot(root_quat, w_last=False)
        root_heading_inv = rotations.calc_heading_quat_inv(root_quat, w_last=False)
        root_quat_noheading = rotations.quat_mul(root_heading_inv, root_quat, w_last=False)
        smpl_pose[..., :3] = torch_transform.quaternion_to_angle_axis(root_quat_noheading).view(
            smpl_pose.shape[:-1] + (3,)
        )
        return smpl_pose

    @property
    def smpl_root_quat_w(self) -> torch.Tensor:
        """Return SMPL root quaternion in Z-up world frame for current frame.

        Returns:
            Tensor of shape ``(num_envs, 4)``.
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_smpl_base_rot(root_quat, w_last=False)
        return root_quat

    @property
    def smpl_root_quat_w_multi_future(self) -> torch.Tensor:
        """Return SMPL root quaternions in Z-up world frame for all SMPL future frames.

        Returns:
            Tensor of shape ``(num_envs, smpl_num_future_frames, 4)``.
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_smpl_base_rot(root_quat, w_last=False).view(
            self.num_envs, self.smpl_num_future_frames, 4
        )
        return root_quat

    @property
    def smpl_root_quat_w_dif_l_multi_future(self) -> torch.Tensor:
        """Return SMPL root orientation relative to robot orientation for all SMPL future frames.

        Computes ``quat_inv(robot_anchor) * smpl_root`` and returns 6D rotation
        matrix representation (first 2 columns).

        Returns:
            Tensor of shape ``(num_envs, smpl_num_future_frames * 6)``.
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_smpl_base_rot(root_quat, w_last=False)
        root_rot_dif = quat_mul(
            quat_inv(
                self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(
                    1, self.smpl_num_future_frames, 1
                )
            ),
            root_quat.view(self.num_envs, self.smpl_num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_dif_l_mat

    @property
    def smpl_root_quat_w_dif_refheading_multi_future(self) -> torch.Tensor:
        """SMPL root orientation canonicalized by the first SMPL future frame's heading.

        Same extraction as smpl_root_quat_w_dif_l_multi_future but uses the heading
        of the first SMPL future frame instead of the robot's orientation.

        Returns:
            torch.Tensor: 6D rotation matrix representation,
                shape (num_envs, smpl_num_future_frames * 6)
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        smpl_root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            smpl_root_quat = self.smpl_root_ytoz_up(smpl_root_quat)
        smpl_root_quat = rotations.remove_smpl_base_rot(smpl_root_quat, w_last=False)
        smpl_root_quat = smpl_root_quat.view(self.num_envs, self.smpl_num_future_frames, 4)
        ref_first_heading = torch_transform.get_heading_q(smpl_root_quat[:, 0, :])
        root_rot_dif = quat_mul(
            quat_inv(
                ref_first_heading.view(self.num_envs, 1, 4).expand(
                    -1, self.smpl_num_future_frames, -1
                )
            ),
            smpl_root_quat,
        )
        mat = matrix_from_quat(root_rot_dif)
        return mat[..., :2].reshape(mat.shape[0], -1)

    @property
    def smpl_root_quat_w_dif_heading_multi_future(self) -> torch.Tensor:
        """SMPL root orientation canonicalized by robot heading (yaw) only.

        Same extraction as smpl_root_quat_w_dif_l_multi_future but uses the
        robot's heading (yaw) instead of full orientation for canonicalization.

        Returns:
            torch.Tensor: 6D rotation matrix representation,
                shape (num_envs, smpl_num_future_frames * 6)
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_smpl_base_rot(root_quat, w_last=False)
        root_rot_dif = quat_mul(
            quat_inv(
                self.anchor_heading_quat.view(self.num_envs, 1, 4).expand(
                    -1, self.smpl_num_future_frames, -1
                )
            ),
            root_quat.view(self.num_envs, self.smpl_num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        return mat[..., :2].reshape(mat.shape[0], -1)

    @property
    def smpl_pose_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        ).reshape(self.num_envs, self.smpl_num_future_frames, -1)

    @property
    def smpl_body_pose_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )[..., 3:].reshape(self.num_envs, -1)

    @property
    def smpl_body_pose_multi_future_6d(self) -> torch.Tensor:
        smpl_body_pose = self.smpl_body_pose_multi_future.reshape(
            -1, self.smpl_num_future_frames, 23, 3
        )
        quat = torch_transform.angle_axis_to_quaternion(smpl_body_pose)
        mat = matrix_from_quat(quat)
        smpl_body_pose_6d = mat[..., :2].reshape(smpl_body_pose.shape[0], -1)
        return smpl_body_pose_6d

    # --- SOMA skeleton properties ---

    @property
    def soma_joints(self) -> torch.Tensor:
        """SOMA joints in Z-up body-local frame. Stored Z-up in PKL (same as SMPL)."""
        return self.motion_lib.get_soma_joints(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def soma_joints_multi_future(self) -> torch.Tensor:
        """SOMA joints multi-future in Z-up body-local frame."""
        soma_joints_mf = self.motion_lib.get_soma_joints(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return soma_joints_mf.view(
            self.num_envs, self.smpl_num_future_frames, *soma_joints_mf.shape[1:]
        )

    @property
    def soma_root_quat_w(self) -> torch.Tensor:
        """SOMA root quaternion in Z-up world frame (wxyz)."""
        root_quat = self.motion_lib.get_soma_root_quat(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        if self.motion_lib.soma_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_bvh_base_rot(root_quat, w_last=False)
        return root_quat

    @property
    def soma_root_quat_w_multi_future(self) -> torch.Tensor:
        """SOMA root quaternion multi-future in Z-up world frame (wxyz)."""
        root_quat = self.motion_lib.get_soma_root_quat(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        if self.motion_lib.soma_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_bvh_base_rot(root_quat, w_last=False)
        return root_quat.view(self.num_envs, self.smpl_num_future_frames, 4)

    @property
    def soma_root_quat_w_dif_l_multi_future(self) -> torch.Tensor:
        """SOMA root orientation relative to robot anchor, as 6D rotation matrix."""
        root_quat = self.motion_lib.get_soma_root_quat(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        if self.motion_lib.soma_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_bvh_base_rot(root_quat, w_last=False)
        root_rot_dif = quat_mul(
            quat_inv(
                self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(
                    1, self.smpl_num_future_frames, 1
                )
            ),
            root_quat.view(self.num_envs, self.smpl_num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_dif_l_mat

    @property
    def soma_transl_multi_future(self) -> torch.Tensor:
        """SOMA hips translation multi-future. Stored Y-up (same as SMPL transl)."""
        soma_transl_mf = self.motion_lib.get_soma_transl(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return soma_transl_mf.view(
            self.num_envs, self.smpl_num_future_frames, *soma_transl_mf.shape[1:]
        )

    @property
    def object_root_pos(self) -> torch.Tensor:
        """Return object root position in world frame with env origin and z-offset.

        Returns:
            Tensor of shape ``(num_envs, num_objects, 3)``.
        """
        object_root_pos = self.motion_lib.get_object_root_pos(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        # Apply z-offset if configured (e.g., to lower chair into ground)
        z_offset = getattr(self.cfg, "object_z_offset", 0.0)
        if z_offset != 0.0:
            object_root_pos = object_root_pos.clone()
            object_root_pos[..., 2] += z_offset
        return object_root_pos + self._env.scene.env_origins[:, None, :]

    @property
    def object_root_quat(self) -> torch.Tensor:
        return self.motion_lib.get_object_root_quat(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    def _get_object_pos_with_offset(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Get object position with randomization offset applied.

        This method also resamples the offset for the given env_ids.
        Called at environment reset to place object with random offset.

        Args:
            env_ids: Environment indices being reset

        Returns:
            Object positions with offset applied, shape [len(env_ids), 3]
        """
        obj_pos = self.object_root_pos[env_ids, 0].clone()

        if self.cfg.object_position_randomize:
            # Resample offset for these environments
            rand_cfg = self.cfg.object_position_randomization or {}
            x_range = rand_cfg.get("x", 0.0)
            y_range = rand_cfg.get("y", 0.0)
            z_range = rand_cfg.get("z", 0.0)

            # Generate new random offsets
            self._object_position_offset[env_ids, 0] = (
                torch.rand(len(env_ids), device=self.device) * 2 - 1
            ) * x_range
            self._object_position_offset[env_ids, 1] = (
                torch.rand(len(env_ids), device=self.device) * 2 - 1
            ) * y_range
            self._object_position_offset[env_ids, 2] = (
                torch.rand(len(env_ids), device=self.device) * 2 - 1
            ) * z_range

            # Apply offset
            obj_pos = obj_pos + self._object_position_offset[env_ids]

        return obj_pos

    @property
    def object_root_pos_multi_future(self) -> torch.Tensor:
        """Return object root positions for all future frames in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames, num_objects, 3)``.
        """
        object_root_pos = self.motion_lib.get_object_root_pos(
            self.future_motion_ids, self.future_time_steps
        )
        object_root_pos_view = object_root_pos.view(self.num_envs, self.num_future_frames, -1, 3)
        # Apply z-offset if configured (e.g., to lower chair into ground)
        z_offset = getattr(self.cfg, "object_z_offset", 0.0)
        if z_offset != 0.0:
            object_root_pos_view = object_root_pos_view.clone()
            object_root_pos_view[..., 2] += z_offset
        return object_root_pos_view + self._env.scene.env_origins[:, None, None, :]

    @property
    def object_root_quat_multi_future(self) -> torch.Tensor:
        object_root_quat = self.motion_lib.get_object_root_quat(
            self.future_motion_ids, self.future_time_steps
        )
        return object_root_quat.view(self.num_envs, self.num_future_frames, -1, 4)

    def _get_contact_center_world(self, hand: str) -> torch.Tensor | None:
        """Get object contact center in world frame for the given hand.

        Args:
            hand: "left_hand" or "right_hand"

        Returns:
            Tensor of shape (num_envs, 3) in world frame, or None if not available.
        """
        contact_center = self.motion_lib.get_object_contact_center(
            self.motion_ids, self.motion_start_time_steps + self.time_steps, hand=hand
        )
        if contact_center is None:
            return None

        # Transform from object-local to world frame
        obj_pos = self.object_root_pos[:, 0, :]  # (num_envs, 3)
        obj_quat = self.object_root_quat[:, 0, :]  # (num_envs, 4)

        from gear_sonic.isaac_utils import rotations

        rotated_center = rotations.quat_rotate(obj_quat, contact_center, w_last=False)
        world_center = rotated_center + obj_pos
        return world_center

    @property
    def object_contact_center_left(self) -> torch.Tensor | None:
        """Get left hand object contact center in world frame. Shape: (num_envs, 3)."""
        return self._get_contact_center_world("left_hand")

    @property
    def object_contact_center_right(self) -> torch.Tensor | None:
        """Get right hand object contact center in world frame. Shape: (num_envs, 3)."""
        return self._get_contact_center_world("right_hand")

    def get_in_contact(self, hand: str = "right_hand") -> torch.Tensor | None:
        """Get binary in_contact label for the given hand at current timestep.

        Args:
            hand: "left_hand" or "right_hand"

        Returns:
            Tensor of shape (num_envs,) with 1.0 if in contact, 0.0 otherwise,
            or None if not available.
        """
        return self.motion_lib.get_object_in_contact(
            self.motion_ids, self.motion_start_time_steps + self.time_steps, hand=hand
        )

    def get_hand_action(self, hand: str = "right_hand") -> torch.Tensor | None:
        """Get discrete hand action (open/closed) for the given hand at current timestep.

        Args:
            hand: "left_hand" or "right_hand"

        Returns:
            Tensor of shape (num_envs,) with -1.0 = open, +1.0 = closed,
            or None if not available.
        """
        return self.motion_lib.get_hand_action(
            self.motion_ids, self.motion_start_time_steps + self.time_steps, hand=hand
        )

    @property
    def joint_pos(self) -> torch.Tensor:
        """Return reference joint positions for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_dof)``.
        """
        return self.motion_lib.get_dof_pos(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def joint_pos_multi_future(self) -> torch.Tensor:
        """Return reference joint positions for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof)``.
        """
        return self.motion_lib.get_dof_pos(self.future_motion_ids, self.future_time_steps).view(
            self.num_envs, -1
        )

    @property
    def joint_pos_multi_future_for_smpl(self) -> torch.Tensor:
        return self.motion_lib.get_dof_pos(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        ).view(self.num_envs, -1)

    @property
    def joint_pos_lower_body_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_dof_pos(self.future_motion_ids, self.future_time_steps)[
            ..., self.lower_joint_isaaclab_indices
        ].view(self.num_envs, -1)

    @property
    def joint_vel(self) -> torch.Tensor:
        """Return reference joint velocities for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_dof)``.
        """
        return self.motion_lib.get_dof_vel(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def joint_vel_multi_future(self) -> torch.Tensor:
        """Return reference joint velocities for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof)``.
        """
        return self.motion_lib.get_dof_vel(self.future_motion_ids, self.future_time_steps).view(
            self.num_envs, -1
        )

    @property
    def root_pos_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_root_pos_w(self.future_motion_ids, self.future_time_steps).view(
            self.num_envs, -1
        )

    @property
    def root_quat_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_root_quat_w(self.future_motion_ids, self.future_time_steps).view(
            self.num_envs, -1
        )

    @property
    def root_z(self) -> torch.Tensor:
        return self.motion_lib.get_root_pos_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, 2:3].view(self.num_envs, -1)

    @property
    def root_z_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_root_pos_w(self.future_motion_ids, self.future_time_steps)[
            :, 2:3
        ].view(self.num_envs, -1)

    @property
    def joint_vel_lower_body_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_dof_vel(self.future_motion_ids, self.future_time_steps)[
            ..., self.lower_joint_isaaclab_indices
        ].view(self.num_envs, -1)

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Return reference body positions in world frame for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 3)``.
        """
        return (
            self.motion_lib.get_body_pos_w(
                self.motion_ids, self.motion_start_time_steps + self.time_steps
            )
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_pos_w_multi_future(self) -> torch.Tensor:
        """Return reference body positions in world frame for all future frames.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_bodies * 3)``.
        """
        body_pos_w = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, -1, 3)
        body_pos_w_env = body_pos_w + self._env.scene.env_origins[:, None, None, :]
        return body_pos_w_env.reshape(self.num_envs, -1)

    @property
    def body_pos_dif_w(self) -> torch.Tensor:
        """Return position difference (reference - robot) in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 3)``.
        """
        body_pos_w = self.motion_lib.get_body_pos_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        body_pos_w_env = body_pos_w + self._env.scene.env_origins[:, None, :]
        body_pos_dif = body_pos_w_env - self.robot_body_pos_w
        return body_pos_dif

    @property
    def body_pos_dif_w_multi_future(self) -> torch.Tensor:
        body_pos_w = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, -1, 3)
        body_pos_w_env = body_pos_w + self._env.scene.env_origins[:, None, None, :]
        body_pos_dif = body_pos_w_env - self.robot_body_pos_w[:, None, :, :]
        return body_pos_dif.reshape(self.num_envs, -1)

    @property
    def body_pos_dif_l(self) -> torch.Tensor:
        """Return body position difference de-headed into robot-local frame.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 3)``.
        """
        body_pos_dif_w = self.body_pos_dif_w
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(1, self.num_bodies, 1)
        deheaded_dif_l = quat_apply_yaw(quat_inv(root_quat), body_pos_dif_w)
        return deheaded_dif_l

    @property
    def body_pos_dif_l_multi_future(self) -> torch.Tensor:
        body_pos_dif_w = self.body_pos_dif_w_multi_future
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 1, 4).repeat(
            1, self.num_future_frames, self.num_bodies, 1
        )
        deheaded_dif_l = quat_apply_yaw(quat_inv(root_quat), body_pos_dif_w)
        return deheaded_dif_l

    @property
    def root_lin_vel_b_2d(self) -> torch.Tensor:
        """Return reference root linear velocity (XY only) in body frame.

        Returns:
            Tensor of shape ``(num_envs, 2)``.
        """
        root_lin_vel_w = self.motion_lib.get_root_lin_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_quat = self.anchor_quat_w.view(self.num_envs, 1, 4)
        root_lin_vel_l = quat_apply_yaw(quat_inv(root_quat), root_lin_vel_w)[:, :2]
        return root_lin_vel_l

    @property
    def root_ang_vel_b_1d(self) -> torch.Tensor:
        """Return reference root angular velocity (yaw only) in body frame.

        Returns:
            Tensor of shape ``(num_envs, 1)``.
        """
        root_ang_vel_w = self.motion_lib.get_root_ang_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_quat = self.anchor_quat_w.view(self.num_envs, 1, 4)
        root_ang_vel_l = quat_apply_yaw(quat_inv(root_quat), root_ang_vel_w)[:, 2:3]
        return root_ang_vel_l

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Return reference body quaternions in world frame for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 4)``.
        """
        return self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def body_quat_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(self.future_motion_ids, self.future_time_steps).view(
            self.num_envs, -1
        )

    @property
    def body_quat_dif_w(self) -> torch.Tensor:
        body_quat_w = self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        body_quat_dif = quat_mul(quat_inv(body_quat_w), self.robot_body_quat_w)
        return body_quat_dif

    @property
    def body_quat_dif_w_multi_future(self) -> torch.Tensor:
        ref_body_quat_w = self.motion_lib.get_body_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, -1, 4)
        robot_body_quat_w = self.robot_body_quat_w.view(
            self.num_envs, 1, self.num_bodies, 4
        ).repeat(1, self.num_future_frames, 1, 1)
        body_quat_dif = quat_mul(quat_inv(ref_body_quat_w), robot_body_quat_w)
        return body_quat_dif

    @property
    def anchor_heading_quat(self) -> torch.Tensor:
        """Return robot anchor heading quaternion (yaw-only, pitch/roll removed).

        Returns:
            Tensor of shape ``(num_envs, 4)``.
        """
        return torch_transform.get_heading_q(self.robot_anchor_quat_w)

    # @property
    # def body_quat_dif_l(self) -> torch.Tensor:
    #     body_quat_dif_w = self.body_quat_dif_w
    #     root_heading = self.anchor_heading_quat.view(self.num_envs, 1, 4).repeat(1, self.num_bodies, 1)
    #     root_heading_inv = quat_inv(self.anchor_heading_quat).view(self.num_envs, 1, 4).repeat(1, self.num_bodies, 1)  # noqa: E501
    #     deheaded_dif_l = quat_mul(quat_mul(root_heading_inv, body_quat_dif_w), root_heading)
    #     mat = matrix_from_quat(deheaded_dif_l)
    #     deheaded_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
    #     return deheaded_dif_l_mat

    # @property
    # def body_quat_dif_l_multi_future(self) -> torch.Tensor:
    #     body_quat_dif_w = self.body_quat_dif_w_multi_future
    #     root_heading = self.anchor_heading_quat.view(self.num_envs, 1, 1, 4).repeat(1, self.num_future_frames, self.num_bodies, 1)  # noqa: E501
    #     root_heading_inv = quat_inv(self.anchor_heading_quat).view(self.num_envs, 1, 1, 4).repeat(1, self.num_future_frames, self.num_bodies, 1)  # noqa: E501
    #     deheaded_dif_l = quat_mul(quat_mul(root_heading_inv, body_quat_dif_w), root_heading)
    #     mat = matrix_from_quat(deheaded_dif_l)
    #     deheaded_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
    #     return deheaded_dif_l_mat

    @property
    def root_rot_dif_l(self) -> torch.Tensor:
        """Return reference root orientation relative to robot orientation in 6D repr.

        Returns:
            Tensor of shape ``(num_envs, 6)``.
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_rot_dif_w = quat_mul(quat_inv(self.robot_anchor_quat_w), ref_root_quat)
        # root_heading = self.anchor_heading_quat.view(self.num_envs, 1, 4).repeat(1, 1, 1)
        # root_heading_inv = quat_inv(self.anchor_heading_quat).view(self.num_envs, 1, 4).repeat(1, 1, 1)
        # deheaded_rot_dif_l = quat_mul(quat_mul(root_heading_inv, root_rot_dif_w), root_heading)
        mat = matrix_from_quat(root_rot_dif_w)
        deheaded_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return deheaded_rot_dif_l_mat

    @property
    def root_rot_dif_l_multi_future(self) -> torch.Tensor:
        """Return reference root orientation relative to robot for all future frames.

        Uses the full robot orientation for canonicalization (preserves heading diff).

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * 6)``.
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        )
        root_rot_dif = quat_mul(
            quat_inv(
                self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(
                    1, self.num_future_frames, 1
                )
            ),
            ref_root_quat.view(self.num_envs, self.num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_dif_l_mat

    @property
    def root_rot_dif_heading_multi_future(self) -> torch.Tensor:
        """Reference root orientation canonicalized by robot heading (yaw) only.

        Unlike root_rot_dif_l_multi_future which uses the full robot orientation,
        this version only uses the heading (yaw) for canonicalization. This preserves
        the reference motion's pitch/roll relative to gravity while removing the
        heading difference.

        Returns:
            torch.Tensor: 6D rotation matrix representation (first 2 columns),
                shape (num_envs, num_future_frames * 6)
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        )
        # Use only the heading (yaw) of the robot orientation for canonicalization
        root_rot_dif = quat_mul(
            quat_inv(
                self.anchor_heading_quat.view(self.num_envs, 1, 4).expand(
                    -1, self.num_future_frames, -1
                )
            ),
            ref_root_quat.view(self.num_envs, self.num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_heading_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_dif_heading_mat

    @property
    def root_rot_dif_refheading_multi_future(self) -> torch.Tensor:
        """Reference root orientation canonicalized by the first future frame's heading.

        Instead of using the robot's current heading for canonicalization, this uses
        the heading of the first (immediate) target frame from the reference motion.
        This makes the trajectory representation independent of the robot's heading.

        Returns:
            torch.Tensor: 6D rotation matrix representation (first 2 columns),
                shape (num_envs, num_future_frames * 6)
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, 4)
        # Use the heading of the first future frame as the canonical frame
        ref_first_heading = torch_transform.get_heading_q(ref_root_quat[:, 0, :])
        root_rot_dif = quat_mul(
            quat_inv(
                ref_first_heading.view(self.num_envs, 1, 4).expand(-1, self.num_future_frames, -1)
            ),
            ref_root_quat,
        )
        mat = matrix_from_quat(root_rot_dif)
        return mat[..., :2].reshape(mat.shape[0], -1)

    @property
    def heading_diff_robot_ref(self) -> torch.Tensor:
        """Relative heading rotation from robot heading to reference first frame heading.

        Computes quat_mul(quat_inv(robot_heading), ref_first_heading), expressing
        the reference motion's heading direction in the robot's heading frame.

        Returns:
            torch.Tensor: 6D rotation matrix representation, shape (num_envs, 6)
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, 4)
        ref_first_heading = torch_transform.get_heading_q(ref_root_quat[:, 0, :])
        heading_diff = quat_mul(quat_inv(self.anchor_heading_quat), ref_first_heading)
        mat = matrix_from_quat(heading_diff)
        return mat[..., :2].reshape(self.num_envs, -1)  # (num_envs, 6)

    @property
    def raw_root_quat_w_multi_future(self) -> torch.Tensor:
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        )
        return ref_root_quat.reshape(self.num_envs, self.num_future_frames, 4)

    @property
    def root_rot_w_multi_future(self) -> torch.Tensor:
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        )
        mat = matrix_from_quat(ref_root_quat)
        root_rot_w_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_w_mat

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_lin_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def body_lin_vel_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_lin_vel_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, -1)

    @property
    def body_lin_vel_l(self) -> torch.Tensor:
        body_lin_vel_w = self.body_lin_vel_w
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(1, self.num_bodies, 1)
        deheaded_vel_l = quat_apply_yaw(quat_inv(root_quat), body_lin_vel_w)
        return deheaded_vel_l

    @property
    def body_lin_vel_l_multi_future(self) -> torch.Tensor:
        body_lin_vel_w = self.body_lin_vel_w_multi_future
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 1, 4).repeat(
            1, self.num_future_frames, self.num_bodies, 1
        )
        deheaded_vel_l = quat_apply_yaw(quat_inv(root_quat), body_lin_vel_w)
        return deheaded_vel_l

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_ang_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def body_ang_vel_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_ang_vel_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, -1)

    @property
    def body_ang_vel_l(self) -> torch.Tensor:
        body_ang_vel_w = self.body_ang_vel_w
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(1, self.num_bodies, 1)
        deheaded_vel_l = quat_apply_yaw(quat_inv(root_quat), body_ang_vel_w)
        return deheaded_vel_l

    @property
    def body_ang_vel_l_multi_future(self) -> torch.Tensor:
        body_ang_vel_w = self.body_ang_vel_w_multi_future
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 1, 4).repeat(
            1, self.num_future_frames, self.num_bodies, 1
        )
        deheaded_vel_l = quat_apply_yaw(quat_inv(root_quat), body_ang_vel_w)
        return deheaded_vel_l

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """Return reference anchor body position in world frame.

        Returns:
            Tensor of shape ``(num_envs, 3)``.
        """
        return (
            self.motion_lib.get_body_pos_w(
                self.motion_ids, self.motion_start_time_steps + self.time_steps
            )[:, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_pos_w_multi_future(self) -> torch.Tensor:
        """Return reference anchor positions for all future frames in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * 3)``.
        """
        anchor_pos_w = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        )[:, self.motion_anchor_body_index].view(self.num_envs, self.num_future_frames, -1)
        anchor_pos_w_env = anchor_pos_w + self._env.scene.env_origins[:, None, :]
        return anchor_pos_w_env.reshape(self.num_envs, -1)

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """Return reference anchor body quaternion in world frame.

        Returns:
            Tensor of shape ``(num_envs, 4)``.
        """
        return self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w_multi_future(self) -> torch.Tensor:
        """Return reference anchor quaternions for all future frames.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * 4)``.
        """
        return self.motion_lib.get_body_quat_w(self.future_motion_ids, self.future_time_steps)[
            :, self.motion_anchor_body_index
        ].reshape(self.num_envs, -1)

    @property
    def anchor_ori_refheading(self) -> torch.Tensor:
        """Current anchor orientation canonicalized by its own heading.

        Uses get_heading_q(anchor_quat_w) as the canonical frame, removing the
        heading component and preserving pitch/roll relative to gravity.

        Returns:
            torch.Tensor: 6D rotation matrix representation, shape (num_envs, 6)
        """
        ref_heading = torch_transform.get_heading_q(self.anchor_quat_w)
        ori = quat_mul(quat_inv(ref_heading), self.anchor_quat_w)
        mat = matrix_from_quat(ori)
        return mat[..., :2].reshape(self.num_envs, -1)

    @property
    def anchor_ori_heading(self) -> torch.Tensor:
        """Current anchor orientation canonicalized by robot heading (yaw).

        Uses get_heading_q(robot_anchor_quat_w) as the canonical frame, preserving
        the reference motion's pitch/roll relative to gravity while removing the
        robot's heading.

        Returns:
            torch.Tensor: 6D rotation matrix representation, shape (num_envs, 6)
        """
        robot_heading = self.anchor_heading_quat
        ori = quat_mul(quat_inv(robot_heading), self.anchor_quat_w)
        mat = matrix_from_quat(ori)
        return mat[..., :2].reshape(self.num_envs, -1)

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_lin_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_lin_vel_w(self.future_motion_ids, self.future_time_steps)[
            :, self.motion_anchor_body_index
        ].view(self.num_envs, -1)

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_ang_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_ang_vel_w(self.future_motion_ids, self.future_time_steps)[
            :, self.motion_anchor_body_index
        ].view(self.num_envs, -1)

    @property
    def vr_3point_body_quat_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.vr_3point_body_indices_motion]

    @property
    def reward_point_body_quat_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.reward_point_body_indices_motion]

    @property
    def vr_3point_body_quat_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(self.future_motion_ids, self.future_time_steps)[
            :, self.vr_3point_body_indices_motion
        ].view(self.num_envs, self.num_future_frames, len(self.cfg.vr_3point_body), -1)

    @property
    def head_orn_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(self.future_motion_ids, self.future_time_steps)[
            :, self.vr_3point_body_indices_motion[2]
        ].view(self.num_envs, self.num_future_frames, -1)

    @property
    def reward_point_body_pos_w(self) -> torch.Tensor:
        reward_point_original = self.motion_lib.get_body_pos_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.reward_point_body_indices_motion]
        return (
            reward_point_original
            + quat_apply(self.reward_point_body_quat_w, self.reward_point_body_offsets)
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def vr_3point_body_pos_w(self) -> torch.Tensor:
        """Return reference VR 3-point body positions (with offsets) in world frame.

        The 3 points are typically left wrist, right wrist, and head. Offsets
        allow tracking a point displaced from the body origin (e.g., palm center).

        Returns:
            Tensor of shape ``(num_envs, 3, 3)``.
        """
        vr_3point_original = self.motion_lib.get_body_pos_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.vr_3point_body_indices_motion]
        return (
            vr_3point_original
            + quat_apply(self.vr_3point_body_quat_w, self.vr_3point_body_offsets)
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def vr_3point_body_pos_w_multi_future(self) -> torch.Tensor:
        """Return reference VR 3-point positions for all future frames in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames, 3, 3)``.
        """
        vr_3point_original = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        )[:, self.vr_3point_body_indices_motion].view(
            self.num_envs, self.num_future_frames, len(self.cfg.vr_3point_body), -1
        )
        vr_3point_offset_extend = self.vr_3point_body_offsets.unsqueeze(1).repeat(
            1, self.num_future_frames, 1, 1
        )
        vr_3point_pos_w = (
            vr_3point_original
            + quat_apply(self.vr_3point_body_quat_w_multi_future, vr_3point_offset_extend)
            + self._env.scene.env_origins[:, None, None, :]
        )
        return vr_3point_pos_w

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """Return the robot's current anchor body position in world frame.

        In offline mode, falls back to the reference motion root position.

        Returns:
            Tensor of shape ``(num_envs, 3)``.
        """
        if getattr(self, "_offline", False):
            return self.motion_lib.get_root_pos_w(
                self.motion_ids, self.motion_start_time_steps + self.time_steps
            )
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """Return the robot's current anchor body quaternion in world frame.

        When ``use_ref_motion_root_quat_w_as_anchor`` is True or in offline mode,
        returns the reference motion root orientation (optionally with added noise)
        instead of the simulated robot state.

        Returns:
            Tensor of shape ``(num_envs, 4)``.
        """
        if self.use_ref_motion_root_quat_w_as_anchor or getattr(self, "_offline", False):
            ref_root_quat = self.motion_lib.get_root_quat_w(
                self.motion_ids, self.motion_start_time_steps + self.time_steps
            )
            if self.ref_motion_root_rotation_noise is not None:
                ref_root_quat = quat_mul(ref_root_quat, self.ref_motion_root_rotation_noise)
            return ref_root_quat
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        assert not getattr(
            self, "_offline", False
        ), "robot_anchor_lin_vel_w is not available in offline mode"
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        assert not getattr(
            self, "_offline", False
        ), "robot_anchor_ang_vel_w is not available in offline mode"
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_vr_3point_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.vr_3point_body_indices]

    @property
    def robot_reward_point_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.reward_point_body_indices] + quat_apply(
            self.robot.data.body_quat_w[:, self.reward_point_body_indices],
            self.reward_point_body_offsets,
        )

    @property
    def robot_vr_3point_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.vr_3point_body_indices] + quat_apply(
            self.robot.data.body_quat_w[:, self.vr_3point_body_indices], self.vr_3point_body_offsets
        )

    @property
    def feet_l(self) -> torch.Tensor:
        return self.motion_lib.get_feet_l(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def feet_r(self) -> torch.Tensor:
        return self.motion_lib.get_feet_r(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def episode_encoder_index(self) -> torch.Tensor:
        return self.encoder_index

    def _update_metrics(self):
        """Compute tracking error metrics between reference motion and robot state.

        Populates ``self.metrics`` with per-env errors for anchor position/rotation,
        body position/rotation, and joint position/velocity. When there is a DOF
        mismatch (extra finger joints), only body joints are compared.
        """
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        # Compare only body joints when there's a DOF mismatch (motion lib has fewer DOFs than robot)
        if self.has_dof_mismatch:
            robot_body_joint_pos = self.robot_joint_pos[:, self.body_joint_indices]
            robot_body_joint_vel = self.robot_joint_vel[:, self.body_joint_indices]
            self.metrics["error_joint_pos"] = torch.abs(self.joint_pos - robot_body_joint_pos).mean(
                dim=-1
            )
            self.metrics["error_joint_vel"] = torch.abs(self.joint_vel - robot_body_joint_vel).mean(
                dim=-1
            )
        else:
            self.metrics["error_joint_pos"] = torch.abs(self.joint_pos - self.robot_joint_pos).mean(
                dim=-1
            )
            self.metrics["error_joint_vel"] = torch.abs(self.joint_vel - self.robot_joint_vel).mean(
                dim=-1
            )

    def resample_all_commands(self):
        """Resample motion clips and reset state for all environments at once."""
        self._resample_command(torch.arange(self.num_envs))

    def _load_contact_data(self):
        """Load contact data from file or directory and validate frame counts against motion library.

        Supports both:
        - Single file: loads one pkl file
        - Directory: loads all pkl files and merges them

        Sets:
            self._contact_data: Raw contact data dict
            self._first_contact_frame: Dict mapping motion_name -> first contact frame index
        """
        import joblib

        self._first_contact_frame = None
        self._contact_data = None
        self._first_contact_lookup = None
        self._per_env_first_contact = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._motion_contact_flags = None

        if self.cfg.contact_file is None or not os.path.exists(self.cfg.contact_file):
            # Fallback: derive first contact from motion lib's in_contact labels
            self._derive_first_contact_from_in_contact_labels()
            return

        # Support both file and directory modes
        if os.path.isfile(self.cfg.contact_file):
            contact_data = joblib.load(self.cfg.contact_file)
        elif os.path.isdir(self.cfg.contact_file):
            # Directory mode: load all pkl files
            # Only load data if internal key EXACTLY matches filename (without .pkl)
            contact_data = {}
            pkl_files = glob.glob(os.path.join(self.cfg.contact_file, "*.pkl"))
            for pkl_file in sorted(pkl_files):
                expected_key = os.path.splitext(os.path.basename(pkl_file))[0]
                try:
                    data = joblib.load(pkl_file)
                    if expected_key in data:
                        contact_data[expected_key] = data[expected_key]
                except Exception as e:  # noqa: BLE001
                    print(f"  Warning: Failed to load {pkl_file}: {e}")  # noqa: T201
            print(  # noqa: T201
                f"[TrackingCommand] Loaded {len(contact_data)} contact sequences from {self.cfg.contact_file}"
            )
        else:
            print(  # noqa: T201
                f"[TrackingCommand] Warning: contact_file path invalid: {self.cfg.contact_file}"
            )
            return

        # Filter contact data to only include motions that exist in motion_lib
        # This respects filter_motion_keys used by motion_lib
        motion_keys_set = set(self.motion_lib.curr_motion_keys)
        filtered_contact_data = {k: v for k, v in contact_data.items() if k in motion_keys_set}
        if len(filtered_contact_data) < len(contact_data):
            print(  # noqa: T201
                f"[TrackingCommand] Filtered contact data: {len(filtered_contact_data)}/{len(contact_data)} "
                f"motions match loaded motion keys"
            )
        self._contact_data = filtered_contact_data

        # Find first frame with contact for each motion
        # Contact file structure: {motion_name: {body: (N, 10475), object: (N, obj_verts), ...}}
        first_contact_frames = {}
        for motion_name, motion_data in filtered_contact_data.items():
            object_contact = motion_data.get("object", None)
            if object_contact is not None:
                # Find first frame where any body vertex is in contact
                contact_per_frame = (object_contact != 0).sum(axis=1)
                contact_frames = np.where(contact_per_frame > 0)[0]
                if len(contact_frames) > 0:
                    first_contact_frames[motion_name] = int(contact_frames[0])
                else:
                    # No contact, use last frame
                    first_contact_frames[motion_name] = object_contact.shape[0]
            else:
                first_contact_frames[motion_name] = 0

        self._first_contact_frame = first_contact_frames
        print(f"[TrackingCommand] Loaded contact data from {self.cfg.contact_file}")  # noqa: T201
        for motion_name, first_frame in first_contact_frames.items():
            print(f"  {motion_name}: first contact at frame {first_frame}")  # noqa: T201

        # Validate and align contact data frame counts to match motion data
        self._validate_and_align_contact_frame_counts(filtered_contact_data)

        # Build motion_key -> first_contact_frame lookup tensor for efficient per-env updates
        self._build_first_contact_lookup()

        # Preprocess contact flags for each motion: motion_id -> (num_frames,) bool tensor
        self._build_motion_contact_flags()

    def _derive_first_contact_from_in_contact_labels(self):
        """Derive first contact frames from motion lib's in_contact labels.

        When no contact_file is provided, but the motion lib has per-frame
        in_contact labels (from contact_points_left_hand/right_hand in the
        object motion data), derive the first contact frame for each motion.
        """
        hand = getattr(self.cfg, "sample_before_contact_hand", "right_hand")
        side = "left" if hand == "left_hand" else "right"
        attr = f"_motion_object_in_contact_{side}"

        if not hasattr(self.motion_lib, attr):
            return

        in_contact_tensor = getattr(self.motion_lib, attr)  # (total_frames,)
        length_starts = self.motion_lib.length_starts
        num_motions = len(self.motion_lib.curr_motion_keys)

        first_contact_frames = {}
        for motion_idx in range(num_motions):
            motion_key = self.motion_lib.curr_motion_keys[motion_idx]
            start = length_starts[motion_idx].item()
            num_frames = self.motion_lib._motion_num_frames[motion_idx].item()  # noqa: SLF001
            end = start + num_frames

            motion_in_contact = in_contact_tensor[start:end]
            contact_indices = torch.nonzero(motion_in_contact > 0.5, as_tuple=False)
            if len(contact_indices) > 0:
                first_contact_frames[motion_key] = int(contact_indices[0].item())
            else:
                first_contact_frames[motion_key] = num_frames

        self._first_contact_frame = first_contact_frames
        print(  # noqa: T201
            f"[TrackingCommand] Derived first contact from in_contact labels ({hand}):"
        )
        for motion_name, first_frame in first_contact_frames.items():
            print(f"  {motion_name}: first contact at frame {first_frame}")  # noqa: T201

        self._build_first_contact_lookup()

    def _build_first_contact_lookup(self):
        """Build a tensor for efficient per-env first contact frame lookup."""
        if self._first_contact_frame is None:
            self._first_contact_lookup = None
            return

        # Create a tensor indexed by motion_id for O(1) lookup
        num_motions = len(self.motion_lib.curr_motion_keys)
        self._first_contact_lookup = torch.zeros(num_motions, device=self.device, dtype=torch.long)

        for motion_idx, motion_key in enumerate(self.motion_lib.curr_motion_keys):
            if motion_key in self._first_contact_frame:
                self._first_contact_lookup[motion_idx] = self._first_contact_frame[motion_key]
            else:
                # Fallback to 0 if not found
                self._first_contact_lookup[motion_idx] = 0

    def _build_motion_contact_flags(self):
        """Preprocess contact flags for each motion: motion_id -> (num_frames,) bool tensor."""
        if self._contact_data is None or len(self._contact_data) == 0:
            self._motion_contact_flags = None
            return

        self._motion_contact_flags = {}

        for motion_idx, motion_key in enumerate(self.motion_lib.curr_motion_keys):
            if motion_key not in self._contact_data:
                continue

            motion_contact_data = self._contact_data[motion_key]
            object_contact = motion_contact_data.get("object", None)

            if object_contact is not None:
                # Convert to torch tensor if needed
                if not isinstance(object_contact, torch.Tensor):
                    object_contact_tensor = torch.from_numpy(object_contact).to(self.device)
                else:
                    object_contact_tensor = object_contact.to(self.device)

                # Compute contact flags: (num_frames,) - True if any vertex has contact
                contact_per_frame = (object_contact_tensor != 0).sum(dim=1)  # (num_frames,)
                contact_flags = contact_per_frame > 0  # (num_frames,) bool

                self._motion_contact_flags[motion_idx] = contact_flags

    def _update_per_env_first_contact(self, env_ids):
        """Update _per_env_first_contact for given env_ids based on their assigned motion."""
        if self._first_contact_lookup is None:
            return

        # Vectorized lookup using motion_ids as indices
        motion_ids = self.motion_ids[env_ids]
        self._per_env_first_contact[env_ids] = self._first_contact_lookup[motion_ids]

    def _validate_and_align_contact_frame_counts(self, contact_data: dict):
        """Validate contact data frame counts match the loaded motion frame counts.

        Allows a tolerance of ±3 frames due to slight duration differences between
        GRAB (120Hz) and robot motion (30Hz) sources.

        Args:
            contact_data: Dict mapping motion_name -> contact arrays

        Raises:
            AssertionError: If frame count difference exceeds 3 frames
        """
        FRAME_TOLERANCE = 3

        for motion_name, motion_contact in contact_data.items():
            # Get contact frame count
            contact_frames = None
            if "object" in motion_contact and motion_contact["object"] is not None:
                contact_frames = motion_contact["object"].shape[0]
            elif "body" in motion_contact and motion_contact["body"] is not None:
                contact_frames = motion_contact["body"].shape[0]

            if contact_frames is None:
                continue

            # Find matching motion in motion_lib and get actual frame count
            motion_idx = None
            for idx, key in enumerate(self.motion_lib.curr_motion_keys):
                if key == motion_name:
                    motion_idx = idx
                    break

            if motion_idx is None:
                print(  # noqa: T201
                    f"[TrackingCommand] Warning: Contact motion '{motion_name}' "
                    f"not found in loaded motions"
                )
                continue

            # Get actual frame count from loaded motion
            motion_frames = int(
                self.motion_lib._motion_num_frames[motion_idx].item()  # noqa: SLF001
            )
            frame_diff = abs(contact_frames - motion_frames)

            if frame_diff == 0:
                print(f"  {motion_name}: {contact_frames} frames (exact match)")  # noqa: T201
            elif frame_diff <= FRAME_TOLERANCE:
                print(  # noqa: T201
                    f"  {motion_name}: {contact_frames} frames "
                    f"(motion lib: {motion_frames}, diff: {contact_frames - motion_frames})"
                )
            else:
                raise AssertionError(
                    f"[TrackingCommand] Frame count mismatch too large for motion '{motion_name}':\n"
                    f"  Contact data: {contact_frames} frames\n"
                    f"  Motion lib: {motion_frames} frames\n"
                    f"  Difference: {frame_diff} frames (max allowed: ±{FRAME_TOLERANCE})"
                )

    def _load_table_meta(self, motion_key: str):
        """Load and cache table meta info (table_pos, table_quat, table_size) for a motion.
        Derives meta path from motion_file path. Falls back to motion file if meta not found.

        For GeniHOI data:
            - table_pos: [x, y, z] center position of the table
            - table_quat: [w, x, y, z] quaternion (identity for cuboid tables)
            - table_size: [width, depth, thickness] dimensions of the table

        For GRAB data:
            - table_pos, table_quat: from meta file
            - table_size: not provided (uses USD with scene_scale)
        """  # noqa: D205
        import joblib

        try:
            # Derive meta path from motion_file path
            motion_file = (
                self.cfg.motion_lib_cfg.get("motion_file", "") if self.cfg.motion_lib_cfg else ""
            )
            if motion_file:
                if os.path.isdir(motion_file):
                    meta_dir = motion_file.replace("/robot", "/meta")
                elif "/robot" in motion_file:
                    meta_dir = os.path.dirname(motion_file).replace("/robot", "/meta")
                else:
                    meta_dir = "data/motion_lib_grab/meta"
                meta_file = os.path.join(meta_dir, f"{motion_key}.pkl")
            else:
                meta_file = f"data/motion_lib_grab/meta/{motion_key}.pkl"

            if os.path.exists(meta_file):
                meta = joblib.load(meta_file)
                self._table_meta_cache[motion_key] = {
                    "table_pos": torch.tensor(
                        meta.get("table_pos", [0.0, 0.0, 0.8]), device=self.device
                    ).float(),
                    "table_quat": torch.tensor(
                        meta.get("table_quat", [1.0, 0.0, 0.0, 0.0]), device=self.device
                    ).float(),
                    "table_size": torch.tensor(
                        meta.get("table_size", [1.0, 0.6, 0.04]), device=self.device
                    ).float(),
                }
                return

            # Fallback: try to get table data from motion file directly
            # Cache ALL motions at once to avoid reloading the file
            if motion_file and os.path.isfile(motion_file):
                motion_file_data = joblib.load(motion_file)
                # Cache table data for ALL motions in the file
                for mk, motion_data in motion_file_data.items():
                    if mk in self._table_meta_cache:
                        continue  # Already cached
                    if (
                        isinstance(motion_data, dict)
                        and "table_pos" in motion_data
                        and "table_quat" in motion_data
                    ):
                        self._table_meta_cache[mk] = {
                            "table_pos": torch.tensor(
                                motion_data["table_pos"], device=self.device
                            ).float(),
                            "table_quat": torch.tensor(
                                motion_data["table_quat"], device=self.device
                            ).float(),
                        }
                    else:
                        self._table_meta_cache[mk] = None
                # After caching all, check if we got the one we needed
                if motion_key in self._table_meta_cache:
                    return
        except Exception:  # noqa: BLE001, S110
            pass

        self._table_meta_cache[motion_key] = None

    def _sample_before_contact(
        self, env_ids: Sequence[int], sampled_times: torch.Tensor
    ) -> torch.Tensor:
        """Sample timestamps before the first contact frame for contact-based initialization.

        Args:
            env_ids: Environment indices to resample
            sampled_times: Originally sampled time steps

        Returns:
            Modified sampled_times with timestamps clamped to before first contact
        """
        if self._first_contact_frame is None or len(self._first_contact_frame) == 0:
            return sampled_times
        # Get motion keys from motion library
        curr_motion_keys = getattr(self.motion_lib, "curr_motion_keys", None)

        for i, env_idx in enumerate(env_ids):
            motion_id = self.motion_ids[env_idx].item()

            # Get motion key for this motion_id
            if curr_motion_keys is not None and motion_id < len(curr_motion_keys):
                motion_key = curr_motion_keys[motion_id]
            else:
                # Fallback: use first contact key
                motion_key = list(self._first_contact_frame.keys())[0]  # noqa: RUF015
            first_contact = None
            if motion_key in self._first_contact_frame:
                first_contact = self._first_contact_frame[motion_key]
            else:
                for contact_key in self._first_contact_frame.keys():  # noqa: SIM118
                    if contact_key in motion_key or motion_key in contact_key:
                        first_contact = self._first_contact_frame[contact_key]
                        break
                if first_contact is None:
                    first_contact = list(self._first_contact_frame.values())[0]  # noqa: RUF015

            # Sample uniformly from [0, first_contact - margin)
            margin = getattr(self.cfg, "sample_before_contact_margin", 10)
            if first_contact > margin:
                sampled_times[i] = torch.randint(
                    0, first_contact - margin, (1,), device=self.device, dtype=sampled_times.dtype
                )
            else:
                sampled_times[i] = 0

        return sampled_times

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample motion clips, reset robot state, and position objects for given envs.

        This is the main episode-reset handler. It performs the following in order:

        1. Sample new motion IDs and start times (respecting evaluation mode,
           paired motions, multi-object mode, and adaptive sampling).
        2. Resample encoder mode (G1/SMPL/teleop) per env.
        3. Apply pose and velocity randomization (skipped during evaluation).
        4. Handle DOF mismatch by mapping motion lib joints to robot joints.
        5. Write joint state and root state to the simulator.
        6. Position scene objects (single or multi-object mode) and tables.
        7. Cache body-relative poses for reward computation.

        Args:
            env_ids: Environment indices being reset.
        """
        self.time_steps[env_ids] = 0
        # Variable frames: resample per-env num_frames at episode reset
        if self.variable_frames_enabled and len(env_ids) > 0:
            idx = torch.randint(0, len(self._frame_choices), (len(env_ids),), device=self.device)
            self.per_env_num_frames[env_ids] = self._frame_choices[idx]
        if len(env_ids) > 0:
            if self.is_evaluating:
                self.motion_ids[env_ids] = (
                    torch.arange(self.num_envs).to(self.device)
                    % self.motion_lib._num_motions  # noqa: SLF001
                )[env_ids]
                self.motion_start_time_steps[env_ids] = 0
            elif self.cfg.use_paired_motions:
                self.motion_ids[env_ids] = (
                    torch.arange(self.num_envs).to(self.device)
                    % self.motion_lib._num_motions  # noqa: SLF001
                )[env_ids]

            elif self._multi_object_mode:
                # MULTI-OBJECT MODE: Resetting envs sample a new motion (and corresponding object)
                # Over time, staggered resets lead to different envs using different objects,
                # which provides training diversity. Object positioning (below) handles per-env instances.
                new_motion_id = self.motion_lib.sample_motions(1)[0]
                self.motion_ids[env_ids] = new_motion_id
                self.motion_start_time_steps[env_ids] = self.motion_lib.sample_time_steps(
                    self.motion_ids[env_ids], truncate_time=None
                )
            else:
                if self.use_adaptive_sampling:
                    sampled_ids, sampled_times = self.motion_lib.sample_motion_ids_and_time_steps(
                        len(env_ids)
                    )
                    self.motion_ids[env_ids] = sampled_ids.to(self.motion_ids.dtype)
                    sampled_times = sampled_times.to(self.motion_start_time_steps.dtype)
                else:
                    self.motion_ids[env_ids] = self.motion_lib.sample_motions(len(env_ids))
                    sampled_times = self.motion_lib.sample_time_steps(
                        self.motion_ids[env_ids], truncate_time=None
                    )

                # Override to sample from initial frames if configured
                if self.cfg.sample_from_n_initial_frames is not None:
                    # Sample uniformly from first N frames
                    n_frames = self.cfg.sample_from_n_initial_frames
                    sampled_times = torch.randint(
                        0, n_frames, (len(env_ids),), dtype=sampled_times.dtype, device=self.device
                    )
                elif self.cfg.start_from_first_frame:
                    sampled_times.zero_()

                # Contact-based initialization: sample timestamps before first contact frame
                if self.cfg.sample_before_contact and self._first_contact_frame is not None:
                    sampled_times = self._sample_before_contact(env_ids, sampled_times)

                self.motion_start_time_steps[env_ids] = sampled_times

            if self.encoder_sample_probs is not None:
                has_smpl = self.motion_lib.motion_has_smpl[self.motion_ids[env_ids]]
                if self.soma_encoder_index is not None and hasattr(
                    self.motion_lib, "motion_has_soma"
                ):
                    has_soma = self.motion_lib.motion_has_soma[self.motion_ids[env_ids]]
                    sampling_cases = [
                        (env_ids[has_smpl & has_soma], self.encoder_sample_probs),
                        (env_ids[has_smpl & ~has_soma], self.encoder_sample_probs_no_soma),
                        (env_ids[~has_smpl & has_soma], self.encoder_sample_probs_no_smpl),
                        (
                            env_ids[~has_smpl & ~has_soma],
                            self.encoder_sample_probs_no_smpl_no_soma,
                        ),
                    ]
                else:
                    sampling_cases = [
                        (env_ids[has_smpl], self.encoder_sample_probs),
                        (env_ids[~has_smpl], self.encoder_sample_probs_no_smpl),
                    ]
                for subset_ids, probs in sampling_cases:
                    if len(subset_ids) > 0:
                        encoder_index = torch.multinomial(
                            probs, len(subset_ids), replacement=True
                        ).to(self.device)
                        self.encoder_index[subset_ids] = 0
                        self.encoder_index[subset_ids, encoder_index] = 1

                # =============================================================
                # Legacy behavior: SMPL-native envs also activate G1 encoder
                # This causes G1 tokens to be computed (then overwritten by SMPL)
                # in the main encoding loop, enabling G1-SMPL latent alignment.
                #
                # When optimize_encoders_ratio_for_CHIP=True:
                #   - Skip this OR logic (cleaner native encoder selection)
                #   - G1 encoder only runs for G1-native envs in main loop
                #   - G1 latents for SMPL-native are computed separately in
                #     aux losses, ONLY when compliance=0 (stiff mode)
                # =============================================================
                if (
                    self.smpl_encoder_index is not None
                    and not self.optimize_encoders_ratio_for_CHIP
                ):
                    use_smpl = self.encoder_index[env_ids, self.smpl_encoder_index]
                    self.encoder_index[env_ids, self.g1_encoder_index] = (
                        self.encoder_index[env_ids, self.g1_encoder_index] | use_smpl
                    )
                    # Also sample teleop mode when smpl mode is active (for latent alignment)
                    if (
                        self.teleop_encoder_index is not None
                        and self.teleop_sample_prob_when_smpl > 0.0
                    ):
                        smpl_env_ids = env_ids[use_smpl.bool()]
                        if len(smpl_env_ids) > 0:
                            sample_teleop = (
                                torch.rand(len(smpl_env_ids), device=self.device)
                                < self.teleop_sample_prob_when_smpl
                            )
                            self.encoder_index[smpl_env_ids, self.teleop_encoder_index] = (
                                self.encoder_index[smpl_env_ids, self.teleop_encoder_index]
                                | sample_teleop.long()
                            )

                # When soma is sampled, also activate g1 (for g1-soma latent alignment)
                if (
                    self.soma_encoder_index is not None
                    and not self.optimize_encoders_ratio_for_CHIP
                ):
                    use_soma = self.encoder_index[env_ids, self.soma_encoder_index]
                    self.encoder_index[env_ids, self.g1_encoder_index] = (
                        self.encoder_index[env_ids, self.g1_encoder_index] | use_soma
                    )

            self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(
                self.motion_ids[env_ids]
            )
            if self.num_future_frames > 1:
                self.future_motion_ids = self.motion_ids.repeat_interleave(self.num_future_frames)
                self.smpl_future_motion_ids = self.motion_ids.repeat_interleave(
                    self.smpl_num_future_frames
                )
                self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(
                    self.motion_ids[env_ids]
                )

            # Update per-env first contact frame based on assigned motion
            if self._first_contact_frame is not None:
                self._update_per_env_first_contact(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        self.running_ref_root_height[env_ids] = self.anchor_pos_w[env_ids, 2]

        # Skip reset randomizations during evaluation — they cause visible stumbling
        # at the start of rendered episodes and are only needed for training robustness.
        if not self.is_evaluating:
            range_list = [
                self.cfg.pose_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
            )
            root_pos[env_ids] += rand_samples[:, 0:3]
            orientations_delta = quat_from_euler_xyz(
                rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
            )
            root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
            range_list = [
                self.cfg.velocity_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
            )
            root_lin_vel[env_ids] += rand_samples[:, :3]
            root_ang_vel[env_ids] += rand_samples[:, 3:]

        # Handle DOF mismatch between motion library and robot
        motion_lib_joint_pos = self.joint_pos.clone()  # Shape: [num_envs, motion_lib_num_dof]
        motion_lib_joint_vel = self.joint_vel.clone()  # Shape: [num_envs, motion_lib_num_dof]

        if self.has_dof_mismatch:
            # Create full robot joint tensors and map using name-based indices
            joint_pos = torch.zeros(
                self.num_envs, self.robot_num_dof, dtype=torch.float32, device=self.device
            )
            joint_vel = torch.zeros(
                self.num_envs, self.robot_num_dof, dtype=torch.float32, device=self.device
            )
            joint_pos[:, self.body_joint_indices] = motion_lib_joint_pos
            joint_vel[:, self.body_joint_indices] = motion_lib_joint_vel
            joint_pos[:, self.extra_joint_indices] = self.extra_default_positions
            joint_vel[:, self.extra_joint_indices] = self.extra_default_velocities
        else:
            joint_pos = motion_lib_joint_pos
            joint_vel = motion_lib_joint_vel

        if not self.is_evaluating:
            joint_pos += sample_uniform(
                *self.cfg.joint_position_range, joint_pos.shape, joint_pos.device
            )
            joint_vel += sample_uniform(
                *self.cfg.joint_velocity_range, joint_vel.shape, joint_vel.device
            )

        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )

        ####### Resetting Humaonid States #######
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos[env_ids],
                    root_ori[env_ids],
                    root_lin_vel[env_ids],
                    root_ang_vel[env_ids],
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )
        # Handle object positioning
        if self._multi_object_mode and len(self._object_names) > 0:
            # MULTI-OBJECT MODE: Position active object, move others far away
            # Get active object name from motion key (original name with hyphens)
            active_motion_id = self.motion_ids[env_ids[0]].item() if len(env_ids) > 0 else 0
            active_motion_key = self.motion_lib.curr_motion_keys[active_motion_id]
            # Convert motion key to safe name (hyphens → underscores) for scene lookup
            active_obj_safe_name = active_motion_key.replace("-", "_")
            self._active_object_name = active_motion_key

            # Position the active object from motion trajectory
            active_key = f"object_{active_obj_safe_name}"
            if active_key in self._env.scene.rigid_objects:
                obj = self._env.scene[active_key]
                obj_pos = self._get_object_pos_with_offset(env_ids)
                # Reset with zero velocity to prevent velocity carryover between episodes
                zero_vel = torch.zeros(len(env_ids), 6, device=self.device)
                obj.write_root_state_to_sim(
                    torch.cat(
                        [obj_pos, self.object_root_quat[env_ids, 0], zero_vel],
                        dim=-1,
                    ),
                    env_ids=env_ids,
                )
            else:
                print(  # noqa: T201
                    f"[Warning] Active object '{active_key}' not found in scene. Available: {list(self._env.scene.rigid_objects.keys())[:5]}..."  # noqa: E501
                )

            # Move all other objects to inactive positions (far away from robot)
            # Spread objects vertically (Z-axis) - envs naturally separate in X,Y via env_origins
            inactive_base = INACTIVE_OBJECT_BASE_OFFSET.to(self.device)
            inactive_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(
                len(env_ids), -1
            )

            inactive_idx = 0
            for safe_name in self._object_names:
                obj_key = f"object_{safe_name}"
                if obj_key != active_key and obj_key in self._env.scene.rigid_objects:
                    # Spread in Z: each object type at different depth underground
                    # X,Y comes from env_origins → naturally separates different envs
                    z_offset = torch.tensor(
                        [0.0, 0.0, -inactive_idx * INACTIVE_OBJECT_Z_SPACING], device=self.device
                    )
                    inactive_pos = self._env.scene.env_origins[env_ids] + inactive_base + z_offset
                    # Reset with zero velocity to prevent velocity carryover between episodes
                    inactive_zero_vel = torch.zeros(len(env_ids), 6, device=self.device)
                    inactive_state = torch.cat(
                        [inactive_pos, inactive_quat, inactive_zero_vel], dim=-1
                    )
                    self._env.scene[obj_key].write_root_state_to_sim(
                        inactive_state, env_ids=env_ids
                    )
                    inactive_idx += 1
        elif "object" in self._env.scene.rigid_objects:
            # SINGLE OBJECT MODE: Existing behavior
            obj = self._env.scene["object"]
            obj_pos = self._get_object_pos_with_offset(env_ids)
            # Reset with zero velocity to prevent velocity carryover between episodes
            zero_vel = torch.zeros(len(env_ids), 6, device=self.device)
            obj.write_root_state_to_sim(
                torch.cat([obj_pos, self.object_root_quat[env_ids, 0], zero_vel], dim=-1),
                env_ids=env_ids,
            )
        if "table" in self._env.scene.rigid_objects:
            table = self._env.scene["table"]

            # Per-env table positions based on each env's assigned motion
            table_pos_list = []
            table_quat_list = []

            # Prefetch motion_ids to CPU to avoid per-iteration GPU sync
            motion_ids_cpu = self.motion_ids.cpu()

            for i, env_idx in enumerate(env_ids):  # noqa: B007
                motion_idx = motion_ids_cpu[env_idx].item()
                motion_key = self.motion_lib.curr_motion_keys[motion_idx]

                # Load and cache meta if not already cached
                if motion_key not in self._table_meta_cache:
                    self._load_table_meta(motion_key)

                # Use cached meta if available
                cached = self._table_meta_cache.get(motion_key)
                if cached is not None:
                    pos = cached["table_pos"].clone()
                    quat = cached["table_quat"].clone()
                    # Add env_origin offset
                    pos = pos + self._env.scene.env_origins[env_idx]
                    table_pos_list.append(pos)
                    table_quat_list.append(quat)
                else:
                    # Fallback: derive from object position with hardcoded offset
                    pos = self.object_root_pos[env_idx, 0].clone()
                    pos[2] = 0.76  # Table height
                    pos[1] -= 0.15
                    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._env.device)
                    table_pos_list.append(pos)
                    table_quat_list.append(quat)

            if len(table_pos_list) > 0:
                table_pos = torch.stack(table_pos_list, dim=0)
                table_quat = torch.stack(table_quat_list, dim=0)

                # Apply table_offset if configured
                if self.cfg.table_offset is not None:
                    table_offset = torch.tensor(
                        self.cfg.table_offset, device=self._env.device, dtype=table_pos.dtype
                    )
                    table_pos = table_pos + table_offset

                table_root_pose = torch.cat([table_pos, table_quat], dim=-1)
                table.write_root_pose_to_sim(table_root_pose, env_ids=env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )

        delta_pos_w = robot_anchor_pos_w_repeat  # Root position of the robot
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = torch_transform.get_heading_q(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

    def _update_command(self):
        """Advance the motion time cursor by one step and handle episode wrap-around.

        Called once per simulation step. Updates adaptive sampling statistics,
        increments the time cursor, resamples any environments that have reached
        the end of their motion clip, updates the running reference root height
        EMA, and optionally performs height-map raycasting for object-aware
        observations.
        """
        if self.use_adaptive_sampling:
            with common.Timer("update_adaptive_sampling"):
                cur_time_steps = self.motion_start_time_steps + self.time_steps
                self.motion_lib.update_adaptive_sampling(
                    self._env.reset_terminated, self.motion_ids, cur_time_steps
                )
        self.time_steps += 1
        env_ids = torch.where(
            self.time_steps + self.motion_start_time_steps
            >= self.motion_lib.get_time_step_total(self.motion_ids)
        )[0]
        self._resample_command(env_ids)

        # Exponential moving average update for running_ref_root_height.
        # ZL this should be moved to the recorders???
        ema_alpha = 0.1  # Smoothing factor, adjust as needed
        self.running_ref_root_height = (
            ema_alpha * self.anchor_pos_w[:, 2] + (1 - ema_alpha) * self.running_ref_root_height
        )

        if self.cfg.use_height_map:
            root_pos_w = self.robot.data.root_pos_w
            root_quat_w = self.robot.data.root_quat_w

            ray_starts_w = root_pos_w.unsqueeze(1).expand(-1, self.num_rays, -1)
            root_quat_expanded = (
                torch_transform.get_heading_q(root_quat_w)
                .unsqueeze(1)
                .expand(-1, self.num_rays, -1)
            )
            ray_dirs_w = quat_apply(
                root_quat_expanded,
                self.ray_dirs_local,
            )

            scan_dot_pos_w, _ = self.height_map.raycast_fused(
                self.object_root_pos,
                self.object_root_quat,
                ray_starts_w,
                ray_dirs_w,
                n_mesh_per_cam=self.num_mesh_per_cam,
                mesh_ids_flattened=self.mesh_ids,
                cam_ids_flattened=self.cam_ids,
                min_dist=0.0,
                max_dist=self.cfg.height_map_max_dist,
            )
            # Adjust hits that fall below the ground plane so all zs are non-negative.
            denom = ray_starts_w[:, :, 2] - scan_dot_pos_w[:, :, 2]
            denom = torch.clamp_min(denom, 1e-8)
            scale = torch.clamp(ray_starts_w[:, :, 2] / denom, max=1.0)
            self.scan_dot_pos_w[:] = (
                ray_starts_w + (scan_dot_pos_w - ray_starts_w) * scale.unsqueeze(-1)
            ).view(self.num_envs, self.num_rays_x, self.num_rays_y, 3)

    @property
    def future_time_steps(self) -> torch.Tensor:
        """Compute absolute time-step indices for all future reference frames.

        Clamps to the last valid frame of each motion to avoid out-of-bounds access.

        Returns:
            Flattened tensor of shape ``(num_envs * num_future_frames,)``.
        """
        return (
            torch.clip(
                self.future_time_steps_init
                + self.time_steps[:, None]
                + self.motion_start_time_steps[:, None],
                max=self.motion_num_steps[:, None] - 1,
            )
            .flatten()
            .long()
        )

    @property
    def smpl_future_time_steps(self) -> torch.Tensor:
        """Compute absolute time-step indices for SMPL future reference frames.

        SMPL future frames may use different count and spacing than robot frames.

        Returns:
            Flattened tensor of shape ``(num_envs * smpl_num_future_frames,)``.
        """
        return (
            torch.clip(
                self.smpl_future_time_steps_init
                + self.time_steps[:, None]
                + self.motion_start_time_steps[:, None],
                max=self.motion_num_steps[:, None] - 1,
            )
            .flatten()
            .long()
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle visibility of debug visualization markers.

        Lazily initializes feet contact markers, height-map dot markers, and
        contact center sphere markers on first enable. Subsequent calls toggle
        visibility without re-creating prims.

        Args:
            debug_vis: Whether to enable or disable debug visualization.
        """
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer = VisualizationMarkers(
                    self.cfg.body_pos_visualizer_cfg.replace(
                        prim_path="/Visuals/goal_marker_sphere"
                    )
                )

                self.feet_contact_goal_visualizers = []

                for name in self.cfg.feet_body_names:
                    self.feet_contact_goal_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.feet_contact_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/goal/" + name
                            )
                        )
                    )

            self.goal_pos_visualizer.set_visibility(True)
            for i in range(len(self.cfg.feet_body_names)):
                self.feet_contact_goal_visualizers[i].set_visibility(True)

            if self.cfg.use_height_map:
                if not hasattr(self, "height_map_visualizer"):
                    height_map_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/height_map",
                        markers={
                            "scan_dots": sim_utils.SphereCfg(
                                radius=0.05,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(1.0, 1.0, 0.0)
                                ),
                            ),
                        },
                    )
                    self.height_map_visualizer = VisualizationMarkers(height_map_cfg)
                self.height_map_visualizer.set_visibility(True)

            # Contact center visualizers: deferred to _debug_vis_callback (lazy init)
            # because motion_lib is not yet available during super().__init__()
            if not hasattr(self, "contact_center_visualizers"):
                self.contact_center_visualizers = None

        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "feet_contact_goal_visualizers"):
                for vis in self.feet_contact_goal_visualizers:
                    vis.set_visibility(False)
            if hasattr(self, "height_map_visualizer"):
                self.height_map_visualizer.set_visibility(False)
            if hasattr(self, "contact_center_visualizers") and self.contact_center_visualizers:
                for vis in self.contact_center_visualizers.values():
                    vis.set_visibility(False)

    def _debug_vis_callback(self, event):  # noqa: ARG002
        """Update debug visualization marker positions each render frame.

        Draws current robot body frames, reference (goal) body frames, feet
        contact indicators, height-map hit points, and object contact centers.

        Args:
            event: Render event from the simulation (unused).
        """
        if not self.robot.is_initialized:
            return

        if not hasattr(self, "goal_pos_visualizer"):
            return

        self.goal_pos_visualizer.visualize(self.body_pos_w.view(-1, 3))

        if hasattr(self, "feet_contact_goal_visualizers"):
            for i in range(len(self.cfg.body_names)):
                if self.cfg.body_names[i] == "left_ankle_roll_link":
                    self.feet_contact_goal_visualizers[0].visualize(
                        translations=self.body_pos_relative_w[:, i],
                        marker_indices=self.feet_l.int().reshape(-1),
                    )
                if self.cfg.body_names[i] == "right_ankle_roll_link":
                    self.feet_contact_goal_visualizers[1].visualize(
                        translations=self.body_pos_relative_w[:, i],
                        marker_indices=self.feet_r.int().reshape(-1),
                    )

        if self.cfg.use_height_map:
            self.height_map_visualizer.visualize(
                translations=self.scan_dot_pos_w.view(-1, 3),
            )

        # Contact center visualization (lazy init on first callback)
        if hasattr(self, "contact_center_visualizers") and self.contact_center_visualizers is None:  # noqa: SIM102
            if hasattr(self, "motion_lib"):
                self.contact_center_visualizers = {}
                if hasattr(self.motion_lib, "_motion_object_contact_center_left"):
                    left_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/contact_center_left",
                        markers={
                            "contact": sim_utils.SphereCfg(
                                radius=0.05,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(0.0, 0.0, 1.0),
                                ),
                            ),
                        },
                    )
                    self.contact_center_visualizers["left_hand"] = VisualizationMarkers(left_cfg)
                if hasattr(self.motion_lib, "_motion_object_contact_center_right"):
                    right_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/contact_center_right",
                        markers={
                            "contact": sim_utils.SphereCfg(
                                radius=0.05,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(0.0, 1.0, 1.0),
                                ),
                            ),
                        },
                    )
                    self.contact_center_visualizers["right_hand"] = VisualizationMarkers(right_cfg)
                for vis in self.contact_center_visualizers.values():
                    vis.set_visibility(True)

        if self.contact_center_visualizers:
            hidden_pos = torch.tensor([[0.0, 0.0, -1000.0]], device=self._env.device)
            for hand, visualizer in self.contact_center_visualizers.items():
                contact_center = self.motion_lib.get_object_contact_center(
                    self.motion_ids, self.motion_start_time_steps + self.time_steps, hand=hand
                )
                if contact_center is None:
                    visualizer.visualize(translations=hidden_pos)
                    continue
                valid_mask = torch.norm(contact_center, dim=-1) > 1e-6
                world_center = self._get_contact_center_world(hand)
                world_center[~valid_mask] = hidden_pos
                visualizer.visualize(translations=world_center)

    def set_motion_state(self, motion_ids, time_steps, motion_start_time_steps=None):
        """Update which motion clip/frame this command serves (for offline use)."""
        self.motion_ids = motion_ids
        self.time_steps = time_steps
        self.motion_start_time_steps = (
            motion_start_time_steps
            if motion_start_time_steps is not None
            else torch.zeros_like(motion_ids)
        )
        self.future_motion_ids = motion_ids.repeat_interleave(self.num_future_frames)
        self.smpl_future_motion_ids = motion_ids.repeat_interleave(self.smpl_num_future_frames)
        self.motion_num_steps = self.motion_lib.get_time_step_total(motion_ids)


class ForceTrackingCommand(CommandTerm):
    """Apply external perturbation forces and manage compliance state for training robustness.

    This command term works alongside ``TrackingCommand`` to add domain-randomized
    external forces on specified robot bodies (typically wrists and torso). It also
    tracks per-env compliance levels that control how stiffly the end-effectors
    track their targets, enabling the policy to learn compliant manipulation
    behaviors.

    Key responsibilities:
        - Maintain per-body force direction and magnitude buffers that event terms
          sample into periodically (every ``force_update_frequency`` steps).
        - Track per-env compliance levels for left wrist, right wrist, and head
          via ``eef_stiffness_buf``.
        - Compute Jacobian-based end-effector analysis for compliance calculations.
        - Report detailed force and compliance metrics per encoder type
          (G1/teleop/SMPL) for W&B logging.

    NOTE: Force application itself is handled by event terms that read and write
    this command's buffers. This command only manages the state and metrics.
    """

    cfg: ForceTrackingCommandCfg

    def __init__(self, cfg: ForceTrackingCommandCfg, env: ManagerBasedRLEnv):
        """Initialize force tracking state, Jacobian indices, and compliance buffers.

        Args:
            cfg: Configuration specifying force bodies, max force, update frequency,
                joint dependencies for Jacobian computation, and debug settings.
            env: The manager-based RL environment that owns this command.
        """
        super().__init__(cfg, env)

        self.is_evaluating = False
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)

        self.left_eef_deps_ids = np.array(
            [self.robot.joint_names.index(joint) for joint in self.cfg.left_eef_deps]
        )
        self.right_eef_deps_ids = np.array(
            [self.robot.joint_names.index(joint) for joint in self.cfg.right_eef_deps]
        )
        self.waist_deps_ids = np.array(
            [self.robot.joint_names.index(joint) for joint in self.cfg.waist_joints]
        )
        self.upper_body_deps_ids = np.concatenate(
            [self.waist_deps_ids, self.left_eef_deps_ids, self.right_eef_deps_ids]
        )

        self.kp = self.robot.data.joint_stiffness.clone()
        self.kd = self.robot.data.joint_damping.clone()

        self.force_update_frequency = self.cfg.force_update_frequency
        self.max_force = self.cfg.max_force
        self.vr_3point_body_indices = [
            self.robot.body_names.index(name) for name in self.cfg.vr_3point_body
        ]
        self.vr_3point_body_indices_motion = [
            self.cfg.body_names.index(name) for name in self.cfg.vr_3point_body
        ]
        self.vr_3point_body_offsets = (
            torch.tensor(self.cfg.vr_3point_body_offset, dtype=torch.float32, device=self.device)
            .view(1, -1, 3)
            .repeat(self.num_envs, 1, 1)
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        ### Force related
        self.num_bodies = len(self.cfg.body_names)
        self.body_force_dir_buf = torch.randn(
            self.num_envs,
            self.num_bodies,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.body_force_dir_buf /= torch.norm(
            self.body_force_dir_buf, dim=-1, keepdim=True
        )  # normalize

        # NOTE: We initialize force_push_ids first so we can use len(force_push_ids) for buffer shape
        self.force_push_ids = self.robot.find_bodies(self.cfg.force_push_body, preserve_order=True)[
            0
        ]
        self.num_force_push_bodies = len(self.force_push_ids)

        # Per-body force magnitude buffer: [num_envs, num_force_push_bodies]
        # Each body can have different force magnitudes, enabling differentiated
        # force application (e.g., stronger forces on wrists than torso)
        self.body_force_magnitude_buf = torch.rand(
            self.num_envs,
            self.num_force_push_bodies,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # [0, 1] per body

        self.force_push_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.force_duration_per_env = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )
        self.force_config_init = False
        self.non_force_push_ids_rel = []
        self.force_push_ids_rel = []
        for i, idx in enumerate(self.body_indexes.tolist()):
            if idx not in self.force_push_ids:
                self.non_force_push_ids_rel.append(i)
            else:
                self.force_push_ids_rel.append(i)
        # self.non_force_push_ids = [i for i in self.body_indexes.tolist() if i not in self.force_push_ids]
        self.force_push_body_offsets = (
            torch.tensor(self.cfg.force_push_body_offset, dtype=torch.float32, device=self.device)
            .view(1, -1, 3)
            .repeat(self.num_envs, 1, 1)
        )
        self.last_force_applied = torch.zeros(
            self.num_envs,
            len(self.force_push_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # compliance related counters
        self.compliance_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.compliance_duration_per_env = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )
        self.eef_stiffness_buf = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.compliance_config_init = False

        # Compliance monitoring - track cumulative stats for sanity checks
        self._force_update_count = 0  # Tracks how many times force was applied (non-zero)
        self._compliance_update_count = 0  # Tracks how many times compliance was updated
        self._total_steps = 0
        self._warned_no_force = False  # Prevent spamming warnings
        # Debug print frequency (0 = disabled, nonzero = print every N steps)
        # Can be set via config: manager_env.commands.force.debug_print_every_n_steps=10
        self._debug_print_every_n_steps = self.cfg.debug_print_every_n_steps
        # Note: Metrics are now created dynamically in _update_metrics()
        # The old "force applied" metric (with space) has been removed to avoid
        # confusion with "force_applied" (with underscore)

    def set_is_evaluating(self, is_evaluating: bool):
        """Toggle evaluation mode."""
        self.is_evaluating = is_evaluating

    @property
    def jacobian(self) -> torch.Tensor:
        """Return the full articulation Jacobian from PhysX.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 6, num_dof + 6)``.
        """
        return self.robot.root_physx_view.get_jacobians()

    @property
    def eef_jacobian(self) -> torch.Tensor:
        """Return the translational Jacobian for left and right end-effectors.

        Returns:
            Tensor of shape ``(num_envs, 2, 3, num_dof + 6)``.
        """
        return self.jacobian[:, self.vr_3point_body_indices[:2], :3, :]

    @property
    def matrix_M(self) -> torch.Tensor:
        """Compute the combined upper-body Jacobian mapping for compliance control.

        Assembles a ``(6, 17)`` matrix per env that maps the 17 upper-body joint
        velocities (3 waist + 7 left arm + 7 right arm) to 6D end-effector
        velocities (3 left + 3 right translational).

        Returns:
            Tensor of shape ``(num_envs, 6, 17)``.
        """
        eef_jacobian = self.eef_jacobian
        left_eef_jac = eef_jacobian[:, 0, :, self.left_eef_deps_ids + 6]
        right_eef_jac = eef_jacobian[:, 1, :, self.right_eef_deps_ids + 6]
        waist_jac_left = eef_jacobian[:, 0, :, self.waist_deps_ids + 6]
        waist_jac_right = eef_jacobian[:, 1, :, self.waist_deps_ids + 6]

        M = torch.zeros(self.num_envs, 6, 17).to(self.device)
        M[:, :3, :3] = waist_jac_left
        M[:, 3:, :3] = waist_jac_right
        M[:, :3, 3:10] = left_eef_jac
        M[:, 3:, 10:] = right_eef_jac
        return M

    def _resample_command(self, env_ids: Sequence[int]):
        """Reset force-related buffers when environments are reset.

        This prevents stale force values from affecting compliance calculations
        immediately after environment reset.

        NOTE: We intentionally do NOT reset force_push_counter here.
        The counter needs to reach force_update_frequency (default: 100) before
        forces are applied. If we reset it on every env reset, and episodes are
        shorter than 100 steps, forces would never be applied!
        """
        if len(env_ids) > 0:
            self.last_force_applied[env_ids] = 0.0

    def _update_command(self):
        """No-op; force state is updated by event terms, not per-step."""
        pass

    def _update_compliance_force_push_related_metrics(self):
        """Compute and log per-body force magnitudes, compliance levels, and encoder ratios.

        Populates ``self.metrics`` with force norms per body, compliance levels
        per encoder type (G1/teleop/SMPL), stiff vs. compliant environment ratios,
        and cross-referenced force-by-compliance-status metrics. Optionally prints
        debug summaries every ``_debug_print_every_n_steps`` steps.
        """
        self._total_steps += 1

        # =====================================================================
        # Get encoder masks from TrackingCommand (for per-encoder metrics)
        # encoder_index: [num_envs, num_encoders] one-hot encoding
        # =====================================================================
        motion_command = None
        encoder_g1_mask = None
        encoder_teleop_mask = None
        encoder_smpl_mask = None
        has_encoder_info = False

        try:
            motion_command = self._env.command_manager.get_term("motion")
            if hasattr(motion_command, "encoder_index") and hasattr(
                motion_command, "encoder_sample_probs_dict"
            ):
                encoder_names = list(motion_command.encoder_sample_probs_dict.keys())
                encoder_index = motion_command.encoder_index  # [num_envs, num_encoders]

                # Get masks for each encoder type
                if "g1" in encoder_names:
                    g1_idx = encoder_names.index("g1")
                    encoder_g1_mask = encoder_index[:, g1_idx].bool()
                if "teleop" in encoder_names:
                    teleop_idx = encoder_names.index("teleop")
                    encoder_teleop_mask = encoder_index[:, teleop_idx].bool()
                if "smpl" in encoder_names:
                    smpl_idx = encoder_names.index("smpl")
                    encoder_smpl_mask = encoder_index[:, smpl_idx].bool()

                has_encoder_info = True
        except Exception:  # noqa: BLE001, S110
            pass  # Motion command not available, skip encoder-specific metrics

        # =====================================================================
        # Per-body force metrics
        # last_force_applied: [num_envs, num_force_push_bodies, 3]
        # force_push_body order: ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
        # =====================================================================
        force_norm = torch.norm(self.last_force_applied, dim=-1)  # [num_envs, num_bodies]
        num_force_bodies = self.last_force_applied.shape[1]

        # Per-body force magnitudes (for wrists and torso)
        if num_force_bodies >= 1:
            self.metrics["force_left_wrist"] = force_norm[:, 0]
        if num_force_bodies >= 2:
            self.metrics["force_right_wrist"] = force_norm[:, 1]
        if num_force_bodies >= 3:
            self.metrics["force_torso"] = force_norm[:, 2]

        # Combined wrist force
        if num_force_bodies >= 2:
            self.metrics["force_wrists_sum"] = force_norm[:, 0] + force_norm[:, 1]
            self.metrics["force_wrists_max"] = torch.max(force_norm[:, 0], force_norm[:, 1])

        # Overall force stats
        self.metrics["force_applied"] = force_norm.mean(dim=-1)
        self.metrics["force_applied_max"] = force_norm.max(dim=-1).values

        # Track if any force was actually applied
        force_nonzero = (force_norm.sum() > 0.01).float()
        if force_nonzero > 0:
            self._force_update_count += 1
        self.metrics["force_nonzero_ratio"] = torch.tensor(
            self._force_update_count / max(1, self._total_steps), device=self.device
        ).expand(self.num_envs)

        # =====================================================================
        # Compliance distribution masks
        # eef_stiffness_buf: [num_envs, 3] -> [left_wrist, right_wrist, head]
        # =====================================================================
        compliance_threshold = 0.001  # Threshold to consider compliance as "active"
        is_compliant = self.eef_stiffness_buf[:, :2].abs().sum(dim=-1) > compliance_threshold
        is_stiff = ~is_compliant

        # =====================================================================
        # ENCODER RATIO METRICS
        # ratio_of_total_envs/encoder_*: Fraction of envs using each encoder
        # =====================================================================
        if has_encoder_info:
            if encoder_g1_mask is not None:
                n_g1 = encoder_g1_mask.sum().float()
                self.metrics["ratio_of_total_envs/encoder_g1"] = (n_g1 / self.num_envs).expand(
                    self.num_envs
                )
            if encoder_teleop_mask is not None:
                n_teleop = encoder_teleop_mask.sum().float()
                self.metrics["ratio_of_total_envs/encoder_teleop"] = (
                    n_teleop / self.num_envs
                ).expand(self.num_envs)
            if encoder_smpl_mask is not None:
                n_smpl = encoder_smpl_mask.sum().float()
                self.metrics["ratio_of_total_envs/encoder_smpl"] = (n_smpl / self.num_envs).expand(
                    self.num_envs
                )

        # =====================================================================
        # COMPLIANCE LEVEL METRICS PER ENCODER
        # compliance_level_LH/encoder_*: Mean compliance for each encoder type
        # =====================================================================
        if has_encoder_info:
            # G1 encoder (should always be stiff = 0)
            if encoder_g1_mask is not None and encoder_g1_mask.sum() > 0:
                g1_compliance = self.eef_stiffness_buf[encoder_g1_mask]
                self.metrics["compliance_level_LH/encoder_g1"] = (
                    g1_compliance[:, 0].mean().expand(self.num_envs)
                )
                self.metrics["compliance_level_RH/encoder_g1"] = (
                    g1_compliance[:, 1].mean().expand(self.num_envs)
                )
                self.metrics["compliance_level_Head/encoder_g1"] = (
                    g1_compliance[:, 2].mean().expand(self.num_envs)
                )
            else:
                self.metrics["compliance_level_LH/encoder_g1"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliance_level_RH/encoder_g1"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliance_level_Head/encoder_g1"] = torch.zeros(
                    self.num_envs, device=self.device
                )

            # Teleop encoder
            if encoder_teleop_mask is not None and encoder_teleop_mask.sum() > 0:
                teleop_compliance = self.eef_stiffness_buf[encoder_teleop_mask]
                self.metrics["compliance_level_LH/encoder_teleop"] = (
                    teleop_compliance[:, 0].mean().expand(self.num_envs)
                )
                self.metrics["compliance_level_RH/encoder_teleop"] = (
                    teleop_compliance[:, 1].mean().expand(self.num_envs)
                )
                self.metrics["compliance_level_Head/encoder_teleop"] = (
                    teleop_compliance[:, 2].mean().expand(self.num_envs)
                )
            else:
                self.metrics["compliance_level_LH/encoder_teleop"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliance_level_RH/encoder_teleop"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliance_level_Head/encoder_teleop"] = torch.zeros(
                    self.num_envs, device=self.device
                )

            # SMPL encoder
            if encoder_smpl_mask is not None and encoder_smpl_mask.sum() > 0:
                smpl_compliance = self.eef_stiffness_buf[encoder_smpl_mask]
                self.metrics["compliance_level_LH/encoder_smpl"] = (
                    smpl_compliance[:, 0].mean().expand(self.num_envs)
                )
                self.metrics["compliance_level_RH/encoder_smpl"] = (
                    smpl_compliance[:, 1].mean().expand(self.num_envs)
                )
                self.metrics["compliance_level_Head/encoder_smpl"] = (
                    smpl_compliance[:, 2].mean().expand(self.num_envs)
                )
            else:
                self.metrics["compliance_level_LH/encoder_smpl"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliance_level_RH/encoder_smpl"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliance_level_Head/encoder_smpl"] = torch.zeros(
                    self.num_envs, device=self.device
                )

        # =====================================================================
        # FORCE METRICS BY COMPLIANCE STATUS
        # compliance_related_force_on_LH/nonzero_compliance_envs: Force on envs with compliance > threshold
        # compliance_related_force_on_LH/stiff_envs: Force on stiff envs (G1 + stiff teleop/smpl)
        # =====================================================================
        if num_force_bodies >= 1:
            # Forces on envs with nonzero compliance (teleop/SMPL with active compliance)
            if is_compliant.sum() > 0:
                force_compliant = force_norm[is_compliant]
                self.metrics["compliance_related_force_on_LH/nonzero_compliance_envs"] = (
                    force_compliant[:, 0].mean().expand(self.num_envs)
                )
                if num_force_bodies >= 2:
                    self.metrics["compliance_related_force_on_RH/nonzero_compliance_envs"] = (
                        force_compliant[:, 1].mean().expand(self.num_envs)
                    )
                if num_force_bodies >= 3:
                    self.metrics["compliance_related_force_on_Head/nonzero_compliance_envs"] = (
                        force_compliant[:, 2].mean().expand(self.num_envs)
                    )
            else:
                self.metrics["compliance_related_force_on_LH/nonzero_compliance_envs"] = (
                    torch.zeros(self.num_envs, device=self.device)
                )
                if num_force_bodies >= 2:
                    self.metrics["compliance_related_force_on_RH/nonzero_compliance_envs"] = (
                        torch.zeros(self.num_envs, device=self.device)
                    )
                if num_force_bodies >= 3:
                    self.metrics["compliance_related_force_on_Head/nonzero_compliance_envs"] = (
                        torch.zeros(self.num_envs, device=self.device)
                    )

            # Forces on stiff envs (G1 envs + teleop/smpl envs with stiff mode)
            if is_stiff.sum() > 0:
                force_stiff = force_norm[is_stiff]
                self.metrics["compliance_related_force_on_LH/stiff_envs"] = (
                    force_stiff[:, 0].mean().expand(self.num_envs)
                )
                if num_force_bodies >= 2:
                    self.metrics["compliance_related_force_on_RH/stiff_envs"] = (
                        force_stiff[:, 1].mean().expand(self.num_envs)
                    )
                if num_force_bodies >= 3:
                    self.metrics["compliance_related_force_on_Head/stiff_envs"] = (
                        force_stiff[:, 2].mean().expand(self.num_envs)
                    )
            else:
                self.metrics["compliance_related_force_on_LH/stiff_envs"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                if num_force_bodies >= 2:
                    self.metrics["compliance_related_force_on_RH/stiff_envs"] = torch.zeros(
                        self.num_envs, device=self.device
                    )
                if num_force_bodies >= 3:
                    self.metrics["compliance_related_force_on_Head/stiff_envs"] = torch.zeros(
                        self.num_envs, device=self.device
                    )

        # =====================================================================
        # Debug print every N steps
        # =====================================================================
        if (
            self._debug_print_every_n_steps > 0
            and self._total_steps % self._debug_print_every_n_steps == 0
        ):
            force_mean = force_norm.mean(dim=0)
            force_vals = [f"{force_mean[i].item():.2f}" for i in range(min(3, num_force_bodies))]
            force_str = ", ".join(force_vals) if force_vals else "N/A"
            body_labels = ["L_wrist", "R_wrist", "Torso"][:num_force_bodies]

            n_compliant = is_compliant.sum().item()
            n_stiff = is_stiff.sum().item()

            # Per-encoder counts
            encoder_str = ""
            if has_encoder_info:
                n_g1 = encoder_g1_mask.sum().item() if encoder_g1_mask is not None else 0
                n_teleop = (
                    encoder_teleop_mask.sum().item() if encoder_teleop_mask is not None else 0
                )
                n_smpl = encoder_smpl_mask.sum().item() if encoder_smpl_mask is not None else 0
                encoder_str = f" | Encoders(G1/Teleop/SMPL): {n_g1}/{n_teleop}/{n_smpl}"

            # Compliance stats for compliant envs only
            if n_compliant > 0:
                compliance_compliant_mean = self.eef_stiffness_buf[is_compliant].mean(dim=0)
                comp_str = f"[{compliance_compliant_mean[0].item():.3f}, {compliance_compliant_mean[1].item():.3f}, {compliance_compliant_mean[2].item():.3f}]"  # noqa: E501
            else:
                comp_str = "[N/A - all stiff]"

            # Debug info about event calls
            event_calls = getattr(self, "_event_call_count", 0)
            counter_stats = f"counter_min={self.force_push_counter.min().item()} max={self.force_push_counter.max().item()}"  # noqa: E501

            print(  # noqa: T201
                f"[ForceDebug @ step {self._total_steps}] "
                f"Force({'/'.join(body_labels)}): [{force_str}] | "
                f"Compliant: {n_compliant} Stiff: {n_stiff}{encoder_str} | "
                f"Compliance(L/R/H): {comp_str} | "
                f"Force ratio: {self._force_update_count}/{self._total_steps} | "
                f"Event calls: {event_calls} | {counter_stats}"
            )

        # =====================================================================
        # Legacy compliance metrics (kept for backward compatibility)
        # =====================================================================
        self.metrics["compliance_left_wrist"] = self.eef_stiffness_buf[:, 0]
        self.metrics["compliance_right_wrist"] = self.eef_stiffness_buf[:, 1]
        self.metrics["compliance_head"] = self.eef_stiffness_buf[:, 2]
        self.metrics["compliance_mean"] = self.eef_stiffness_buf.mean(dim=-1)

        # =====================================================================
        # Compliance ratio metrics FOR NON-G1 ENCODER ENVS ONLY
        # G1 envs are always forced to stiff, so including them is misleading.
        # These metrics show the stiff/compliant distribution among TELEOP/SMPL envs.
        # =====================================================================
        if has_encoder_info and encoder_g1_mask is not None:
            non_g1_mask = ~encoder_g1_mask
            n_non_g1 = non_g1_mask.sum()
            if n_non_g1 > 0:
                stiff_ratio_non_g1 = is_stiff[non_g1_mask].float().mean()
                self.metrics["stiff_ratio_for_non_G1_encoder_envs"] = stiff_ratio_non_g1.expand(
                    self.num_envs
                )
                self.metrics["compliant_ratio_for_non_G1_encoder_envs"] = (
                    1.0 - stiff_ratio_non_g1
                ).expand(self.num_envs)
            else:
                # All envs are G1, ratio is undefined (report 0)
                self.metrics["stiff_ratio_for_non_G1_encoder_envs"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics["compliant_ratio_for_non_G1_encoder_envs"] = torch.zeros(
                    self.num_envs, device=self.device
                )
        else:
            # No encoder info available, fall back to global ratio
            stiff_ratio = is_stiff.float().mean()
            self.metrics["stiff_ratio_for_non_G1_encoder_envs"] = stiff_ratio.expand(self.num_envs)
            self.metrics["compliant_ratio_for_non_G1_encoder_envs"] = (1.0 - stiff_ratio).expand(
                self.num_envs
            )

        # Sanity check warning: If compliance is being varied but no forces are applied
        compliance_is_active = self.compliance_config_init and (
            self.eef_stiffness_buf.abs().sum() > 0
        )
        if compliance_is_active and self._total_steps > 1000:  # noqa: SIM102
            if self._force_update_count == 0 and not self._warned_no_force:
                import warnings

                warnings.warn(  # noqa: B028
                    "\n" + "=" * 80 + "\n"
                    "[COMPLIANCE SANITY CHECK FAILED]\n"
                    "Compliance values are being changed, but NO external forces have been applied!\n"
                    "This means vr_3point_local_target_compliant == vr_3point_local_target always.\n"
                    "The policy will NOT learn to use the compliance signal.\n\n"
                    "FIX: Add 'compliance_force_push' event to your events config:\n"
                    "  defaults:\n"
                    "    - terms/compliance_force_push@_here_\n"
                    "=" * 80
                )
                self._warned_no_force = True

    def _update_metrics(self):
        """Refresh max_force from config and compute all force/compliance metrics."""
        self.max_force = self.cfg.max_force
        self._update_compliance_force_push_related_metrics()

    @property
    def command(self):
        """Return None; ForceTrackingCommand has no direct observation tensor."""
        return None


@configclass
class TrackingCommandCfg(CommandTermCfg):
    """Configuration for TrackingCommand motion tracking.

    Controls motion library loading, future reference frame layout, encoder
    sampling probabilities (G1/SMPL/teleop), episode initialization strategy,
    object handling, and debug visualization marker styles.
    """

    class_type: type = TrackingCommand

    asset_name: str = dataclasses.MISSING

    motion_lib_cfg: dict = None
    motion_file: str = None
    smpl_motion_file: str = None
    filter_motion_keys: list[str] = None
    use_paired_motions: bool = False
    # Contact-based initialization: sample timestamps before first contact frame
    # Path to contact pickle file (e.g., data/motion_lib_grab/contact/s2_apple_lift.pkl)
    contact_file: str = None
    # If True, sample only timestamps before the first contact frame
    sample_before_contact: bool = False
    # Margin (in frames) before first contact to sample from
    sample_before_contact_margin: int = 10
    # Which hand's in_contact label to use for deriving first contact frame
    sample_before_contact_hand: str = "right_hand"
    anchor_body: str = dataclasses.MISSING
    body_names: list[str] = dataclasses.MISSING

    vr_3point_body: list[str] = []  # noqa: RUF012
    vr_3point_body_offset: list[list[float]] = []  # noqa: RUF012

    reward_point_body: list[str] = []  # noqa: RUF012
    reward_point_body_offset: list[list[float]] = []  # noqa: RUF012

    # For backward compatibility (to remove)
    force_push_body: list[str] = []  # noqa: RUF012
    force_push_body_offset: list[list[float]] = []  # noqa: RUF012

    num_future_frames: int = 1
    dt_future_ref_frames: float = 0.1
    randomize_heading: bool = False

    # Variable frame support: when enabled, num_future_frames serves as max_frames
    # and each environment/sample gets a random num_frames from
    # [variable_frames_min, num_future_frames] with step variable_frames_step.
    # Step must be a power of 2 compatible with down_t (step=4 works for down_t=2).
    variable_frames_enabled: bool = False
    variable_frames_min: int = 16
    variable_frames_step: int = 4

    freeze_frame_aug: bool = False
    freeze_frame_aug_prob: float = 0.1

    smpl_num_future_frames: int = None
    smpl_dt_future_ref_frames: float = None

    smpl_num_future_frames: int = None  # noqa: PIE794
    smpl_dt_future_ref_frames: float = None  # noqa: PIE794

    encoder_sample_probs: dict[str, float] = None

    # ==========================================================================
    # CHIP Compliance Training Optimization Flag
    # ==========================================================================
    # When True, enables cleaner encoder logic for compliance-aware training:
    #   1. SMPL-native envs do NOT automatically activate G1 encoder in main loop
    #   2. G1 encoder only runs for G1-native envs (policy tokens)
    #   3. G1 latents for SMPL-native envs are computed ONLY in aux losses
    #      and ONLY when compliance=0 (stiff mode)
    #
    # This eliminates wasted computation where G1 tokens were computed for
    # SMPL-native envs but immediately overwritten by SMPL tokens.
    #
    # Set to False (default) for backward compatibility with non-compliance runs.
    # ==========================================================================
    optimize_encoders_ratio_for_CHIP: bool = False

    # Probability to also sample teleop mode when smpl mode is active
    # This enables teleop-smpl and g1-teleop latent alignment losses
    teleop_sample_prob_when_smpl: float = 0.0

    # Always start from the first frame of the motion file during resampling
    # Useful for debugging and replaying specific motions from the beginning
    start_from_first_frame: bool = False

    # Sample each motion at most once (no duplicates across environments)
    # Requires num_envs <= num_available_motions, otherwise will error
    # Useful for replay/evaluation to ensure coverage of all unique motions
    sample_unique_motions: bool = False

    # Sample from the first N frames of the motion (random uniform in [0, N-1])
    # If set, this takes precedence over start_from_first_frame
    # Useful for adding slight variation while still starting near the beginning
    sample_from_n_initial_frames: int = None

    # Object position randomization (adds random offset to object position at reset)
    object_position_randomize: bool = False
    # Randomization range for each axis: {"x": 0.05, "y": 0.05, "z": 0.0}
    # Values are half-range, so 0.05 means uniform random in [-0.05, 0.05]
    object_position_randomization: dict[str, float] = None

    # Table position offset: [x, y, z] added to table position from meta file
    # Useful for adjusting table position relative to the robot
    table_offset: list[float] = None

    pose_range: dict[str, tuple[float, float]] = {}  # noqa: RUF012
    velocity_range: dict[str, tuple[float, float]] = {}  # noqa: RUF012

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    joint_velocity_range: tuple[float, float] = (-0, 0)

    body_pos_visualizer_cfg: VisualizationMarkersCfg = DEFORMABLE_TARGET_MARKER_CFG.replace(
        prim_path="/Visuals/goal_marker_sphere"
    )
    body_pos_visualizer_cfg.markers["target"].radius = 0.05
    body_pos_visualizer_cfg.markers["target"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 1.0, 0.0)
    )

    feet_contact_visualizer_cfg: VisualizationMarkersCfg = DEFORMABLE_TARGET_MARKER_CFG.replace(
        prim_path="/Visuals/goal_marker_sphere"
    )
    feet_contact_visualizer_cfg.markers = {  # noqa: RUF012
        "target_small": sim_utils.SphereCfg(
            radius=0.0001,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "target_big": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }

    waist_joints = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]  # noqa: RUF012

    left_eef_deps = [  # noqa: RUF012
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
    ]

    right_eef_deps = [  # noqa: RUF012
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
    ]

    force_push_body: list[str] = []  # default only two wrists  # noqa: PIE794, RUF012
    force_push_body_offset: list[list[float]] = []  # noqa: PIE794, RUF012

    feet_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]  # noqa: RUF012
    cat_upper_body_poses: bool = False
    cat_upper_body_poses_prob: float = 0.5
    randomize_wrist_poses: bool = False
    randomize_wrist_prob: float = 0.3
    randomize_wrist_std: float = 0.1  # radians (~5.7 degrees)

    use_height_map: bool = False
    height_map_resolution: float = 0.15
    height_map_size: float = 1.5
    height_map_max_dist: float = 5.0
    motion_lib_num_dof: int | None = None  # If None, assumes motion lib DOF matches robot DOF
    hand_default_positions: list[float] | None = (
        None  # Default positions for extra joints (e.g., hands)
    )
    hand_default_velocities: list[float] | None = (
        None  # Default velocities for extra joints (defaults to 0)
    )
    # Object z-offset (e.g., -0.05 to lower chair 5cm into ground)
    object_z_offset: float = 0.0


@configclass
class ForceTrackingCommandCfg(CommandTermCfg):
    """Configuration for ForceTrackingCommand external perturbation and compliance."""

    class_type: type = ForceTrackingCommand

    asset_name: str = dataclasses.MISSING

    anchor_body: str = dataclasses.MISSING
    body_names: list[str] = dataclasses.MISSING

    force_update_frequency: int = 100
    max_force: float = 20.0

    vr_3point_body: list[str] = []  # noqa: RUF012
    vr_3point_body_offset: list[list[float]] = []  # noqa: RUF012

    force_push_body: list[str] = []  # default only two wrists  # noqa: RUF012
    force_push_body_offset: list[list[float]] = []  # noqa: RUF012

    # Debug print frequency (0 = disabled, nonzero = print every N steps)
    # Usage: manager_env.commands.force.debug_print_every_n_steps=10
    debug_print_every_n_steps: int = 0

    waist_joints = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]  # noqa: RUF012

    left_eef_deps = [  # noqa: RUF012
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
    ]

    right_eef_deps = [  # noqa: RUF012
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
    ]


def _get_body_indexes(command: TrackingCommand, body_names: list[str] | None) -> list[int]:
    """Return indices into command.cfg.body_names for the requested subset.

    Args:
        command: TrackingCommand instance whose cfg.body_names defines the full body list.
        body_names: Subset of body names to select. If None, return all indices.

    Returns:
        List of integer indices into ``command.cfg.body_names``.
    """
    return [
        idx
        for idx, body_name in enumerate(command.cfg.body_names)
        if body_names is None or body_name in body_names
    ]


# Backward compat — remove after all checkpoints migrated
MotionCommand = TrackingCommand
MotionCommandCfg = TrackingCommandCfg
ForceCommand = ForceTrackingCommand
ForceCommandCfg = ForceTrackingCommandCfg
