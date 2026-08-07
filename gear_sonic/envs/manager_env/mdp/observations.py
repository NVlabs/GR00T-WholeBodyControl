"""Observation functions for the manager-based RL environment MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_apply_yaw,
    quat_conjugate,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
import torch

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.envs.manager_env.mdp import commands, utils
from gear_sonic.trl.utils import torch_transform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# Joint ordering constants (Mujoco order for compatibility)
G1_MUJOCO_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Index mappings for 29 DOF
isaaclab_to_mujoco_dof = [joint_utils.G1_ISAACLab_ORDER.index(i) for i in G1_MUJOCO_ORDER]
mujoco_to_isaaclab = [G1_MUJOCO_ORDER.index(i) for i in joint_utils.G1_ISAACLab_ORDER]


@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    command = None
    command_vel = None
    command_max = None
    command_multi_future = None
    command_multi_future_joint_pos = None
    command_multi_future_joint_body_pos = None
    command_multi_future_joint_body_diff_pos = None
    command_multi_future_joint_body_abs_pos = None
    command_multi_future_lower_body = None
    command_max_multi_future = None
    command_max_diff_w = None
    command_max_diff_w_multi_future = None
    command_max_diff_l = None
    command_max_diff_l_multi_future = None
    command_z_multi_future = None
    root_pos_multi_future = None
    root_quat_multi_future = None
    joint_pos_multi_future = None
    smpl_pose = None
    smpl_body_pose = None
    smpl_joints_multi_future = None
    smpl_joints_lower_multi_future = None
    smpl_root_ori_b = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b_mf = None
    motion_anchor_pos_b_xy = None
    motion_anchor_ori_w_mf = None
    motion_anchor_ori_heading_mf = None
    motion_anchor_ori_b = None
    motion_anchor_ori_w = None
    motion_anchor_yaw_b = None
    robot_anchor_ori_w = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    # Body-only observations (for pre-trained action_transform_module)
    joint_pos_wo_hand = None
    joint_vel_wo_hand = None
    actions_wo_hand = None
    vr_3point_target = None
    vr_3point_target_compliant = None
    vr_3point_target_compliant_multi_future = None
    vr_3point_target_multi_future = None
    vr_3point_orn_target_multi_future = None
    vr_3point_local_target = None
    vr_3point_local_target_compliant = None
    vr_3point_local_target_multi_future = None
    vr_3point_local_orn_target = None
    head_orn_target_multi_future = None
    vr_wrists_local_pos_target = None
    vr_wrists_local_orn_target = None
    vr_head_local_orn_target = None
    gravity_dir = None
    compliance = None
    ext_forces = None
    motion_anchor_gravity_dir = None
    body_pos = None
    body_pos_diff_l = None
    # HOI manipulation task specific
    target_object_pos = None
    hand_object_transform = None
    finger_tips_force = None
    # Object future motion observations
    object_pos_b_multi_future = None
    object_ori_b_multi_future = None
    object_pos_delta_multi_future = None
    object_ori_delta_multi_future_6d = None
    grab_contact_flag = None
    # Hand-object transform
    hand_object_transform_6d = None
    # Object-to-root observations (current state in body frame)
    object_pos_b = None
    object_ori_b_6d = None
    # Table observations
    table_pos_b = None
    table_ori_b = None

    # Policy output from last step (latent residual + primitives, e.g., 64+2=66 dims)
    last_meta_action = None

    # Terrain observations
    height_map_flat = None


@configclass
class PolicyAtmCfg(ObsGroup):
    """Observations for action_transform_module (ATM).

    This observation group provides body-only observations (29 DOF) for use with
    pre-trained action_transform_module. When using a 43 DOF robot (29 body + 14 hand),
    this group extracts only the body joint observations matching ATM's expected input.

    NOTE: Order must match the observations the pretrained ATM was trained with.
    """

    # Order matches PolicyCfg: base_ang_vel, joint_pos, joint_vel, actions, gravity_dir
    base_ang_vel = None
    joint_pos_wo_hand = None
    joint_vel_wo_hand = None
    actions_wo_hand = None
    gravity_dir = None


@configclass
class TeacherCfg(ObsGroup):
    """Teacher observations for distillation.

    This provides privileged state observations that the teacher policy uses.
    The student policy learns to imitate the teacher's actions using vision instead.
    """

    # # Basic proprioception
    # command = None
    # motion_anchor_pos_b = None
    # motion_anchor_ori_b = None
    # base_lin_vel = None
    # base_ang_vel = None
    # joint_pos = None
    # joint_vel = None
    # actions = None
    # # Object and manipulation observations
    # target_object_pos = None
    # hand_object_transform = None
    # finger_tips_force = None
    # grab_contact_flag = None
    # # Object future motion observations
    # object_pos_b_multi_future = None
    # object_ori_b_multi_future = None
    # # Body multi-future observations
    # command_multi_future = None
    # motion_anchor_ori_b_mf = None

    # observation terms (order preserved)
    command = None
    command_vel = None
    command_max = None
    command_multi_future = None
    command_multi_future_joint_pos = None
    command_multi_future_joint_body_pos = None
    command_multi_future_joint_body_diff_pos = None
    command_multi_future_joint_body_abs_pos = None
    command_multi_future_lower_body = None
    command_max_multi_future = None
    command_max_diff_w = None
    command_max_diff_w_multi_future = None
    command_max_diff_l = None
    command_max_diff_l_multi_future = None
    command_z_multi_future = None
    root_pos_multi_future = None
    root_quat_multi_future = None
    joint_pos_multi_future = None
    smpl_pose = None
    smpl_body_pose = None
    smpl_joints_multi_future = None
    smpl_joints_lower_multi_future = None
    smpl_root_ori_b = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b_mf = None
    motion_anchor_pos_b_xy = None
    motion_anchor_ori_w_mf = None
    motion_anchor_ori_b = None
    motion_anchor_ori_w = None
    motion_anchor_yaw_b = None
    robot_anchor_ori_w = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    # Body-only observations (for pre-trained action_transform_module)
    joint_pos_wo_hand = None
    joint_vel_wo_hand = None
    actions_wo_hand = None
    vr_3point_target = None
    vr_3point_target_compliant = None
    vr_3point_target_compliant_multi_future = None
    vr_3point_target_multi_future = None
    vr_3point_orn_target_multi_future = None
    vr_3point_local_target = None
    vr_3point_local_target_compliant = None
    vr_3point_local_target_multi_future = None
    vr_3point_local_orn_target = None
    head_orn_target_multi_future = None
    vr_wrists_local_pos_target = None
    vr_wrists_local_orn_target = None
    vr_head_local_orn_target = None
    gravity_dir = None
    compliance = None
    ext_forces = None
    motion_anchor_gravity_dir = None
    body_pos = None
    body_pos_diff_l = None
    # HOI manipulation task specific
    target_object_pos = None
    hand_object_transform = None
    finger_tips_force = None
    # Object future motion observations
    object_pos_b_multi_future = None
    object_ori_b_multi_future = None
    grab_contact_flag = None
    # Policy output from last step (latent residual + primitives, e.g., 64+2=66 dims)
    last_meta_action = None


@configclass
class PrivilegedCfg(ObsGroup):
    """Privileged observations for the critic network (asymmetric actor-critic)."""

    command = None
    command_max = None
    command_multi_future = None
    command_multi_future_lower_body = None
    command_multi_future_lower_body_joint_pos = None
    command_max_multi_future = None
    command_max_diff_w = None
    command_max_diff_w_multi_future = None
    command_max_diff_l = None
    command_max_diff_l_multi_future = None
    command_z_multi_future = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b = None
    motion_anchor_ori_b_mf = None
    motion_anchor_ori_heading_mf = None
    body_pos = None
    body_ori = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    vr_3point_target = None
    vr_3point_target_compliant_multi_future = None
    vr_3point_target_multi_future = None
    vr_3point_orn_target_multi_future = None
    head_orn_target_multi_future = None
    vr_3point_local_target = None
    vr_3point_local_target_compliant = None
    vr_3point_local_target_multi_future = None
    vr_3point_local_orn_target = None
    vr_wrists_local_pos_target = None
    vr_wrists_local_orn_target = None
    vr_head_local_orn_target = None
    gravity_dir = None
    compliance = None
    ext_forces = None
    motion_anchor_gravity_dir = None
    # HOI manipulation task specific
    target_object_pos = None
    hand_object_transform = None
    finger_tips_force = None
    # Object future motion observations
    object_pos_b_multi_future = None
    object_ori_b_multi_future = None
    object_pos_delta_multi_future = None
    object_ori_delta_multi_future_6d = None
    # Hand-object transform
    hand_object_transform_6d = None
    # Object-to-root observations (current state in body frame)
    object_pos_b = None
    object_ori_b_6d = None
    # Table observations
    table_pos_b = None
    table_ori_b = None
    # Staged training
    task_stage = None

    # g1 token obs
    ref_root_pos_future_b = None
    ref_root_ori_future_b = None
    diff_body_pos_future_local = None
    diff_body_ori_future_local = None
    diff_body_lin_vel_future_local = None
    diff_body_ang_vel_future_local = None
    grab_contact_flag = None

    # Terrain observations
    height_map_flat = None


@configclass
class DiscriminatorCfg:
    """Observation specifications for the discriminator."""

    disc_obs = None
    ref_disc_obs = None


@configclass
class TokenizerCfg(ObsGroup):
    """Observations for the tokenizer (SONIC/ATM encoder input)."""

    encoder_index = None
    command_multi_future_nonflat = None
    motion_anchor_ori_w = None
    command_z_multi_future_nonflat = None
    command_z = None
    motion_anchor_ori_b = None
    motion_anchor_ori_heading_b = None
    motion_anchor_ori_b_nonflat = None
    motion_anchor_ori_b_mf_nonflat = None
    motion_anchor_ori_w_mf_nonflat = None
    command_multi_future_egocentric_joint_transforms = None
    command_multi_future_egocentric_joint_transforms_nonflat = None
    command_multi_future_egocentric_joint_positions = None
    command_multi_future_egocentric_joint_positions_nonflat = None
    command_multi_future_egocentric_joint_rotations = None
    command_multi_future_egocentric_joint_rotations_nonflat = None
    command_multi_future_root_transforms = None
    command_multi_future_root_transforms_nonflat = None
    motion_anchor_ori_heading_mf_nonflat = None
    motion_anchor_ori_refheading_mf_nonflat = None
    heading_diff_robot_ref = None
    motion_anchor_ori_refheading = None
    motion_anchor_ori_heading = None
    command_multi_future_lower_body = None
    vr_3point_local_target = None
    vr_3point_local_orn_target = None
    vr_3point_local_target_compliant = None
    smpl_joints_multi_future_nonflat = None
    smpl_joints_multi_future_local_nonflat = None
    smpl_joints_multi_future_local_flatten = None
    smpl_lower_body_joints_multi_future_local_nonflat = None
    smpl_lower_body_joints_multi_future_local_flatten = None
    smpl_joints_lower_multi_future_local_flatten = None
    smpl_transl_z_multi_future_nonflat = None
    smpl_root_ori_b_multi_future = None
    smpl_root_ori_b_multi_future_flatten = None
    smpl_root_ori_refheading_multi_future = None
    smpl_root_ori_heading_multi_future = None
    smpl_elbow_wrist_pose_multi_future = None
    smpl_wrist_pose_multi_future = None
    joint_pos_multi_future_wrist = None
    joint_pos_multi_future_wrist_flatten = None
    joint_pos_multi_future_wrist_for_smpl = None
    # SOMA skeleton observations
    soma_joints_multi_future_local_nonflat = None
    soma_root_ori_b_multi_future = None
    joint_pos_multi_future_wrist_for_soma = None
    # Object goal observations (position and orientation in robot body frame)
    object_pos_b = None
    object_ori_b = None
    object_ori_b_6d = None
    ref_root_pos_future_b = None
    ref_root_ori_future_b = None
    diff_body_pos_future_local = None
    diff_body_ori_future_local = None
    diff_body_lin_vel_future_local = None
    diff_body_ang_vel_future_local = None
    compliance = None
    # HOI encoder observations (for end-to-end SONIC-HOI training)
    command = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b_mf = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    target_object_pos = None
    hand_object_transform_6d = None
    finger_tips_force = None
    table_pos_b = None
    table_ori_b = None
    object_pos_delta_multi_future = None
    object_ori_delta_multi_future_6d = None
    command_multi_future = None


@configclass
class HeightMapCfg(ObsGroup):
    """Height map observation group for terrain-aware locomotion."""

    height_map = None


@configclass
class CameraRGBCfg(ObsGroup):
    """Camera RGB image observation group.

    This is a separate observation group for vision observations.
    It will be passed through the wrapper as 'camera_rgb' key in obs_dict,
    not concatenated with other policy observations.
    """

    camera_rgb = None


@configclass
class ResidualAction(ObsGroup):
    """Observation group for residual action feedback."""

    residual_action = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups
    policy: PolicyCfg = None
    critic: PrivilegedCfg = None
    disc: DiscriminatorCfg = None
    tokenizer: TokenizerCfg = None
    policy_atm: PolicyAtmCfg = None  # Body-only obs for pre-trained action_transform_module
    height_map: HeightMapCfg = None
    teacher: TeacherCfg = None  # Teacher observations for distillation
    camera_rgb: CameraRGBCfg = None  # Separate vision observation group
    residual_action: ResidualAction = None


def command_max(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get maximum reference body positions in heading-local frame.

    Returns:
        torch.Tensor: Flattened body positions, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    return command.command_max


def command_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference body velocities.

    Returns:
        torch.Tensor: Flattened body velocities, shape (num_envs, vel_dim).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_vel


def command_max_diff(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get difference between reference and robot body positions.

    Returns:
        torch.Tensor: Flattened position differences, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_diff


def command_max_diff_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference-minus-robot body position differences for multiple future frames.

    Returns:
        torch.Tensor: Flattened multi-future differences,
            shape (num_envs, num_future_frames * num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_diff_multi_future


def command_z(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference root z-height.

    Returns:
        torch.Tensor: Root z-height, shape (num_envs, 1).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_z


def command_z_multi_future(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference root z-heights for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, 1).

    Returns:
        torch.Tensor: Z-heights, shape (num_envs, num_future_frames) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_z_multi_future.reshape(
            command.num_envs, command.num_future_frames, -1
        )
    else:
        return command.command_z_multi_future


def command_max_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get maximum reference body positions for multiple future frames in heading-local frame.

    Returns:
        torch.Tensor: Flattened multi-future body positions,
            shape (num_envs, num_future_frames * num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_multi_future


def command_multi_future(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference body positions in body-local frame for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, num_bodies * 3).

    Returns:
        torch.Tensor: Body positions, shape (num_envs, num_future_frames * num_bodies * 3)
            when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_multi_future.reshape(command.num_envs, command.num_future_frames, -1)
    else:
        return command.command_multi_future


def command_multi_future_joint_pos(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference joint positions for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, num_joints).

    Returns:
        torch.Tensor: Joint positions, shape (num_envs, num_future_frames * num_joints)
            when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_multi_future_joint_pos.reshape(
            command.num_envs, command.num_future_frames, -1
        )
    else:
        return command.command_multi_future_joint_pos


def command_multi_future_joint_body_pos(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference joint body positions for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, num_joints * 3).

    Returns:
        torch.Tensor: Joint body positions, shape (num_envs, num_future_frames * num_joints * 3)
            when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_multi_future_joint_body_pos.reshape(
            command.num_envs, command.num_future_frames, -1
        )
    else:
        return command.command_multi_future_joint_body_pos


def command_multi_future_joint_body_diff_pos(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference-minus-robot joint body position differences for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, num_joints * 3).

    Returns:
        torch.Tensor: Joint body position differences,
            shape (num_envs, num_future_frames * num_joints * 3) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_multi_future_joint_body_diff_pos.reshape(
            command.num_envs, command.num_future_frames, -1
        )
    else:
        return command.command_multi_future_joint_body_diff_pos


def command_multi_future_joint_body_abs_pos(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get absolute reference joint body positions for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, num_joints * 3).

    Returns:
        torch.Tensor: Absolute joint body positions,
            shape (num_envs, num_future_frames * num_joints * 3) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_multi_future_joint_body_abs_pos.reshape(
            command.num_envs, command.num_future_frames, -1
        )
    else:
        return command.command_multi_future_joint_body_abs_pos


def command_multi_future_lower_body(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference lower-body positions for multiple future frames in body-local frame.

    Returns:
        torch.Tensor: Flattened lower-body positions,
            shape (num_envs, num_future_frames * lower_body_dim).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_multi_future_lower_body


def command_multi_future_lower_body_joint_pos(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Get reference lower-body joint positions for multiple future frames.

    Returns:
        torch.Tensor: Flattened lower-body joint positions,
            shape (num_envs, num_future_frames * lower_body_joints).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_multi_future_lower_body_joint_pos


def command_max_diff_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference-minus-robot body position differences in world frame.

    Returns:
        torch.Tensor: Flattened position differences, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_diff_w


def command_max_diff_w_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference-minus-robot body position differences in world frame for multiple future frames.

    Returns:
        torch.Tensor: Flattened multi-future differences,
            shape (num_envs, num_future_frames * num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_diff_w_multi_future


def command_max_diff_l(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference-minus-robot body position differences in body-local frame.

    Returns:
        torch.Tensor: Flattened position differences, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_diff_l


def command_max_diff_l_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference-minus-robot body position differences in body-local frame for multiple future frames.

    Returns:
        torch.Tensor: Flattened multi-future differences,
            shape (num_envs, num_future_frames * num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_max_diff_l_multi_future


def command_num_frames(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Per-env number of valid future frames. Shape [num_envs, 1]."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if getattr(command, "per_env_num_frames", None) is not None:
        return command.per_env_num_frames.float().reshape(-1, 1)
    return torch.full(
        (command.num_envs, 1), float(command.num_future_frames), device=command.device
    )


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get robot anchor (pelvis) orientation in world frame as 6D rotation.

    Returns:
        torch.Tensor: First two columns of rotation matrix, shape (num_envs, 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_gravity_dir(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Compute gravity direction in the reference motion anchor frame.

    Transforms the world-frame down vector into the reference anchor's local frame.

    Returns:
        torch.Tensor: Gravity direction vector, shape (num_envs, 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    gravity_dir = quat_apply(quat_inv(command.anchor_quat_w), command.down_dir)
    return gravity_dir


def gravity_dir(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Compute gravity direction in the robot anchor (pelvis) frame.

    Transforms the world-frame down vector into the robot's local frame.
    Provides the policy with tilt/orientation information.

    Returns:
        torch.Tensor: Gravity direction vector, shape (num_envs, 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    gravity_dir = quat_apply(quat_inv(command.robot_anchor_quat_w), command.down_dir)
    return gravity_dir


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get robot anchor (pelvis) linear velocity in world frame.

    Returns:
        torch.Tensor: Linear velocity xyz, shape (num_envs, 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get robot anchor (pelvis) angular velocity in world frame.

    Returns:
        torch.Tensor: Angular velocity xyz, shape (num_envs, 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get robot body positions in robot anchor (pelvis) local frame.

    Transforms all tracked body positions from world frame into the robot's
    anchor frame using subtract_frame_transforms.

    Returns:
        torch.Tensor: Flattened body positions, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_pos_diff_l(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference-minus-robot body position differences in world frame.

    Computes (reference_body_pos - robot_body_pos) for each tracked body.

    NOTE: Despite the ``_l`` suffix, the subtraction is done in world frame
    (body_pos_relative_w - robot_body_pos_w).

    Returns:
        torch.Tensor: Flattened position differences, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = commands._get_body_indexes(command, command.cfg.body_names)  # noqa: SLF001
    body_pos_diff_l = (
        command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]
    )
    return body_pos_diff_l.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get robot body orientations in robot anchor (pelvis) local frame as 6D rotation.

    Returns:
        torch.Tensor: First two columns of each body's rotation matrix,
            shape (num_envs, num_bodies * 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str, mask_out_z=False) -> torch.Tensor:
    """Get reference motion anchor position relative to robot anchor in robot-local frame.

    Computes the position of the reference motion's root relative to the robot's
    root, expressed in the robot's local coordinate frame.

    Args:
        command_name: Name of the tracking command term.
        mask_out_z: If True, return only xy components (discard z).

    Returns:
        torch.Tensor: Position offset, shape (num_envs, 3) or (num_envs, 2) if mask_out_z.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    if mask_out_z:
        pos = pos[:, :2]

    return pos.view(env.num_envs, -1)


def motion_anchor_yaw_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get yaw (heading) difference between reference anchor and robot anchor.

    Extracts only the heading component of the relative orientation between the
    reference motion root and the robot root.

    Returns:
        torch.Tensor: Heading quaternion, shape (num_envs, 4).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    yaw = torch_transform.get_heading_q(ori)
    return yaw.view(env.num_envs, -1)


def motion_anchor_ori_heading_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference root orientation with heading removed as 6D rotation.

    Removes the reference anchor's own heading (yaw) from the reference root
    quaternion, preserving pitch/roll relative to gravity.

    Returns:
        torch.Tensor: 6D rotation (first two columns of rotation matrix),
            shape (num_envs, 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.motion_lib.get_root_quat_w(
        command.motion_ids, command.motion_start_time_steps + command.time_steps
    )
    root_heading_inv = quat_inv(command.anchor_heading_quat).view(env.num_envs, 4)
    deheaded_ref_rot = quat_mul(root_heading_inv, ref_root_quat)
    mat = matrix_from_quat(deheaded_ref_rot)
    deheaded_ref_root_mat = mat[..., :2].reshape(mat.shape[0], -1)
    return deheaded_ref_root_mat


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str, non_flatten=False) -> torch.Tensor:
    """Get reference anchor orientation relative to robot anchor as 6D rotation.

    Computes the orientation difference between the reference motion root and the
    robot root, expressed in the robot's local frame.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, repeat across future frames and return
            shape (num_envs, num_future_frames, 6).

    Returns:
        torch.Tensor: 6D rotation representation, shape (num_envs, 6) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,  # robot
        command.robot_anchor_quat_w,
        command.anchor_pos_w,  # reference
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    ori = mat[..., :2].reshape(mat.shape[0], -1)
    if non_flatten:
        return ori.unsqueeze(1).repeat(1, command.num_future_frames, 1)
    else:
        return ori


def motion_anchor_ori_refheading(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Motion anchor orientation canonicalized by its own heading.

    Removes the heading component from the anchor orientation, preserving
    pitch/roll relative to gravity.

    Returns:
        torch.Tensor: 6D rotation matrix representation,
            shape (num_envs, 6) or (num_envs, num_future_frames, 6) if non_flatten
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ori = command.anchor_ori_refheading  # (num_envs, 6)
    if non_flatten:
        return ori.unsqueeze(1).repeat(1, command.num_future_frames, 1)
    return ori


def motion_anchor_ori_heading(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Motion anchor orientation canonicalized by robot heading (yaw).

    Uses the robot's heading for canonicalization, preserving the reference
    motion's pitch/roll relative to gravity while removing heading difference.

    Returns:
        torch.Tensor: 6D rotation matrix representation,
            shape (num_envs, 6) or (num_envs, num_future_frames, 6) if non_flatten
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ori = command.anchor_ori_heading  # (num_envs, 6)
    if non_flatten:
        return ori.unsqueeze(1).repeat(1, command.num_future_frames, 1)
    return ori


def motion_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference motion anchor orientation in world frame as 6D rotation.

    Returns:
        torch.Tensor: First two columns of rotation matrix, shape (num_envs, 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_ori_b_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference-vs-robot root orientation difference for multiple future frames in body-local frame.

    Uses the robot's full orientation (including pitch/roll) for normalization.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, 6).

    Returns:
        torch.Tensor: 6D rotation differences,
            shape (num_envs, num_future_frames * 6) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.root_rot_dif_l_multi_future.reshape(
            env.num_envs, command.num_future_frames, -1
        )
    else:
        return command.root_rot_dif_l_multi_future.reshape(env.num_envs, -1)


def motion_anchor_ori_w_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference root orientations in world frame for multiple future frames as 6D rotation.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, 6).

    Returns:
        torch.Tensor: 6D rotation representations,
            shape (num_envs, num_future_frames * 6) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.root_rot_w_multi_future.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return command.root_rot_w_multi_future.reshape(env.num_envs, -1)


# =============================================================================
# Egocentric joint transforms and root transforms relative to first frame
# =============================================================================


def command_multi_future_egocentric_joint_transforms(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Egocentric joint transforms (positions + rotations) for reference frames.

    For each future frame, joint positions and rotations are relative to that frame's
    projected root (heading/yaw only rotation, z=0).

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, num_bodies_full * 9] (3 pos + 6 rot per body, flattened)
        else:
            [num_envs, num_future_frames * num_bodies_full * 9]
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    transforms = command.egocentric_joint_transforms_multi_future
    if non_flatten:
        return transforms.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return transforms.reshape(env.num_envs, -1)


def command_multi_future_egocentric_joint_positions(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Egocentric joint positions for reference frames.

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, num_bodies_full * 3]
        else:
            [num_envs, num_future_frames * num_bodies_full * 3]
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    positions = command.egocentric_joint_positions_multi_future
    if non_flatten:
        return positions.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return positions.reshape(env.num_envs, -1)


def command_multi_future_egocentric_joint_rotations(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Egocentric joint rotations (6D) for reference frames.

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, num_bodies_full * 6]
        else:
            [num_envs, num_future_frames * num_bodies_full * 6]
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    rotations = command.egocentric_joint_rotations_multi_future
    if non_flatten:
        return rotations.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return rotations.reshape(env.num_envs, -1)


def command_multi_future_root_transforms(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Root transforms (position + rotation) relative to the first reference frame.

    Position: Delta from first frame's projected root position, in first frame's heading frame.
    Rotation: Relative rotation from first frame's heading quaternion (6D representation).

    Returns:
        if non_flatten:
            [num_envs, num_future_frames, 1 * 9] (3 pos + 6 rot)
        else:
            [num_envs, num_future_frames * 1 * 9]
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    transforms = command.root_transforms_relative_to_first_frame
    if non_flatten:
        return transforms.reshape(env.num_envs, command.num_future_frames, -1)
    else:
        return transforms.reshape(env.num_envs, -1)


def motion_anchor_ori_heading_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Motion anchor orientation normalized by robot heading (yaw) only, for multi-future frames.

    Unlike motion_anchor_ori_b_mf which normalizes by the robot's full orientation (including
    pitch/roll), this version only normalizes by the robot's heading (yaw). This preserves
    the reference motion's pitch/roll relative to gravity while removing the heading difference.

    Returns:
        torch.Tensor: Orientation as 6D rotation matrix representation (first 2 columns of
            rotation matrix), shape (num_envs, num_future_frames, 6) if non_flatten else
            (num_envs, num_future_frames * 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.root_rot_dif_heading_multi_future.reshape(
            env.num_envs, command.num_future_frames, -1
        )
    else:
        return command.root_rot_dif_heading_multi_future.reshape(env.num_envs, -1)


def motion_anchor_ori_refheading_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Motion anchor orientation canonicalized by the first target frame's heading.

    Instead of using the robot's heading, this uses the heading of the first (immediate)
    future frame from the reference motion. The trajectory is expressed in the reference
    motion's own heading frame.

    Returns:
        torch.Tensor: Orientation as 6D rotation matrix representation,
            shape (num_envs, num_future_frames, 6) if non_flatten else
            (num_envs, num_future_frames * 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.root_rot_dif_refheading_multi_future.reshape(
            env.num_envs, command.num_future_frames, -1
        )
    else:
        return command.root_rot_dif_refheading_multi_future.reshape(env.num_envs, -1)


def heading_diff_robot_ref(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Relative heading from robot to reference motion's first target frame.

    Computes the rotation from the robot's heading to the reference motion's first
    future frame heading as a single-frame 6D rotation.

    Returns:
        torch.Tensor: 6D rotation matrix representation, shape (num_envs, 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.heading_diff_robot_ref


### 3 point force based tracking
def vr_3point_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point tracking target positions in heading-local frame.

    Transforms the reference 3-point body positions (left wrist, right wrist, head)
    relative to the robot anchor, then de-heads (removes yaw) to get a
    heading-invariant representation.

    Returns:
        torch.Tensor: Flattened 3-point positions, shape (num_envs, 9).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    pos_b = command.vr_3point_body_pos_w - command.robot_anchor_pos_w[:, None, :]

    # transform pos_b in to deheaded root frame
    root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    deheaded_pos_b = quat_apply_yaw(quat_inv(root_quat), pos_b)
    return deheaded_pos_b.view(env.num_envs, -1)


def _phase_to_weight_pyramid(phase, start=0.25, end=0.75):
    """Compute pyramid-shaped weight from a [0,1] phase value.

    Ramps linearly from 0 to 1 over [0, start], holds at 1 over [start, end],
    then ramps linearly from 1 to 0 over [end, 1].
    """
    weight = torch.zeros_like(phase)
    mask1 = phase < start
    mask2 = phase > end
    weight[mask1] = 1 / start * phase[mask1]
    weight[mask2] = 1 / (1 - end) * (1 - phase[mask2])
    weight[~(mask1 | mask2)] = 1.0
    return weight


def vr_3point_target_compliant_multi_future(
    env: ManagerBasedEnv, motion_command_name: str, force_command_name: str
) -> torch.Tensor:
    """Get force-compliant VR 3-point targets for multiple future frames in heading-local frame.

    Modifies the reference 3-point positions by subtracting external force displacements
    (scaled by compliance/stiffness) to create compliant tracking targets. The force
    profile uses a pyramid-shaped phase weighting across future frames.

    Returns:
        torch.Tensor: Flattened compliant targets,
            shape (num_envs, num_future_frames * num_points * 3).
    """
    motion_command: commands.TrackingCommand = env.command_manager.get_term(motion_command_name)
    force_command: commands.ForceTrackingCommand = env.command_manager.get_term(force_command_name)

    future_phases = (
        (
            force_command.force_push_counter[:, None]
            + motion_command.future_time_steps_init
            - force_command.force_update_frequency
        )
        / force_command.force_duration_per_env[:, None]
    ).clamp(min=0.0, max=1.0)
    # future_phases shape: (num_envs, num_future_frames)
    future_forces = (
        force_command.body_force_magnitude_buf[:, None, None, None]
        * _phase_to_weight_pyramid(future_phases)[:, :, None, None]
        * (force_command.body_force_dir_buf[:, force_command.force_push_ids_rel])[:, None, :, :]
        * force_command.max_force
    )

    # future_forces [E, 5, 2, 3]
    ext_force_disp = future_forces * force_command.eef_stiffness_buf[:, None, :, None]

    mod_3point = motion_command.vr_3point_body_pos_w_multi_future
    mod_3point -= ext_force_disp
    pos_b = mod_3point - motion_command.robot_anchor_pos_w[:, None, None, :]
    # transform pos_b in to deheaded root frame
    root_quat = motion_command.robot_anchor_quat_w.view(env.num_envs, 1, 1, 4).repeat(
        1, motion_command.num_future_frames, len(motion_command.cfg.vr_3point_body), 1
    )
    deheaded_pos_b = quat_apply_yaw(quat_inv(root_quat), pos_b)
    return deheaded_pos_b.view(env.num_envs, -1)


def vr_3point_target_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point tracking target positions for multiple future frames in heading-local frame.

    Returns:
        torch.Tensor: Flattened multi-future 3-point positions,
            shape (num_envs, num_future_frames * num_points * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    pos_b = command.vr_3point_body_pos_w_multi_future - command.robot_anchor_pos_w[:, None, None, :]
    # transform pos_b in to deheaded root frame
    root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 1, 4).repeat(
        1, command.num_future_frames, len(command.cfg.vr_3point_body), 1
    )
    deheaded_pos_b = quat_apply_yaw(quat_inv(root_quat), pos_b)
    return deheaded_pos_b.view(env.num_envs, -1)


def head_orn_target_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference head orientation for multiple future frames, de-headed by robot yaw.

    Returns:
        torch.Tensor: Flattened head quaternions,
            shape (num_envs, num_future_frames * 4).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    head_quat = command.head_orn_w_multi_future
    root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, command.num_future_frames, 1
    )
    # deheaded_head_quat = quat_mul(head_quat, get_heading_q(quat_inv(root_quat)))
    deheaded_head_quat = quat_mul(torch_transform.get_heading_q(quat_inv(root_quat)), head_quat)
    return deheaded_head_quat.view(env.num_envs, -1)


def vr_3point_orn_target_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point body orientations for multiple future frames, de-headed by robot yaw.

    Returns:
        torch.Tensor: Flattened body quaternions,
            shape (num_envs, num_future_frames * num_points * 4).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    vr_3point_quat = command.vr_3point_body_quat_w_multi_future
    root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 1, 4).repeat(
        1, command.num_future_frames, len(command.cfg.vr_3point_body), 1
    )
    # deheaded_vr_3point_quat = quat_mul(vr_3point_quat, get_heading_q(quat_inv(root_quat)))
    deheaded_vr_3point_quat = quat_mul(
        torch_transform.get_heading_q(quat_inv(root_quat)), vr_3point_quat
    )
    return deheaded_vr_3point_quat.view(env.num_envs, -1)


def vr_3point_local_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point positions in reference motion anchor local frame.

    NOTE: "local" here means relative to the reference motion root, not the robot root.
    This gives the policy the reference pose structure independent of global position.

    Returns:
        torch.Tensor: Flattened 3-point positions, shape (num_envs, num_points * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    return ref_3point_root.view(env.num_envs, -1)


def vr_3point_local_target_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point positions in reference anchor local frame for multiple future frames.

    All future frames use the current reference anchor orientation for canonicalization.

    Returns:
        torch.Tensor: Flattened multi-future local positions,
            shape (num_envs, num_future_frames * num_points * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 1, 4).repeat(
        1, command.num_future_frames, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_diff = (
        command.vr_3point_body_pos_w_multi_future - command.anchor_pos_w[:, None, None, :]
    )  # All w.r.t current anchor
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    return ref_3point_root.view(env.num_envs, -1)


def vr_3point_local_target_compliant(
    env: ManagerBasedEnv,
    motion_command_name: str,
    force_command_name: str,
    zero_out_head_position: bool = False,
) -> torch.Tensor:
    """Get force-compliant VR 3-point targets in reference anchor local frame.

    Subtracts external force displacements (in robot-local frame) from the reference
    3-point positions to create compliant tracking targets.

    Args:
        motion_command_name: Name of the motion tracking command.
        force_command_name: Name of the force tracking command.
        zero_out_head_position: If True, zero out the head position component.

    Returns:
        torch.Tensor: Flattened compliant targets, shape (num_envs, num_points * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(motion_command_name)
    force_command: commands.ForceTrackingCommand = env.command_manager.get_term(force_command_name)
    ext_force_disp_w = (
        force_command.last_force_applied * force_command.eef_stiffness_buf[:, :, None]
    )  # delta x external force world frame
    root_quat = command.robot_anchor_quat_w[:, None, :].repeat(
        1, len(command.cfg.vr_3point_body), 1
    )  # robot root pos not ref motion root pose as force is applied on robot, ref motion root frame is meaningless
    ext_force_disp_l = quat_apply(
        quat_inv(root_quat), ext_force_disp_w
    )  # delta x external force local frame
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    ref_3point_root -= ext_force_disp_l  # [E, 3, 3] -- only a single frame VR 3 point is returned

    # Conditionally zero out the head position (index -1)
    if zero_out_head_position:
        ref_3point_root[:, -1, :] = 0.0  # Head position

    return ref_3point_root.view(env.num_envs, -1)


def vr_3point_local_orn_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point body orientations in reference anchor local frame.

    Returns:
        torch.Tensor: Flattened 3-point quaternions, shape (num_envs, num_points * 4).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_quat = command.vr_3point_body_quat_w
    ref_3point_root = quat_mul(quat_inv(ref_root_quat), ref_3point_quat)
    return ref_3point_root.view(env.num_envs, -1)


def vr_wrists_local_pos_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get wrist positions in reference anchor local frame (wrists only, no head).

    Returns:
        torch.Tensor: Flattened 2-wrist positions, shape (num_envs, 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    ref_2point_diff = command.vr_3point_body_pos_w[:, :2] - command.anchor_pos_w[:, None, :]
    ref_2point_root = quat_apply(quat_inv(ref_root_quat), ref_2point_diff)
    return ref_2point_root.view(env.num_envs, -1)


def vr_wrists_local_orn_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get wrist orientations in reference anchor local frame (wrists only, no head).

    Returns:
        torch.Tensor: Flattened 2-wrist quaternions, shape (num_envs, 8).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    ref_2point_quat = command.vr_3point_body_quat_w[:, :2]
    ref_2point_root = quat_mul(quat_inv(ref_root_quat), ref_2point_quat)
    return ref_2point_root.view(env.num_envs, -1)


def vr_head_local_orn_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get head orientation in reference anchor local frame.

    Returns:
        torch.Tensor: Head quaternion in anchor frame, shape (num_envs, 4).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w
    head_quat = command.vr_3point_body_quat_w[:, 2]
    ref_head_quat_root = quat_mul(quat_inv(ref_root_quat), head_quat)
    return ref_head_quat_root.view(env.num_envs, -1)


def get_command_obs(env: ManagerBasedEnv, command_name: str, obs_name: str) -> torch.Tensor:
    """Get an arbitrary attribute from the tracking command by name.

    Generic accessor for command observations not covered by dedicated functions.

    Args:
        command_name: Name of the tracking command term.
        obs_name: Attribute name on the command object.

    Returns:
        torch.Tensor: The requested observation tensor.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return getattr(command, obs_name)


def smpl_pose_multi_future_select_joints(
    env: ManagerBasedEnv, command_name: str, joints_idx: list
) -> torch.Tensor:
    """Extract SMPL joint axis-angle poses for selected joints across future frames.

    Args:
        command_name: Name of the tracking command term.
        joints_idx: Indices of SMPL joints to select.

    Returns:
        torch.Tensor: Selected joint poses, shape (..., len(joints_idx) * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    smpl_pose = command.smpl_pose_multi_future
    smpl_pose_selected = smpl_pose.view(*smpl_pose.shape[:-1], -1, 3)[..., joints_idx, :].view(
        *smpl_pose.shape[:-1], -1
    )
    return smpl_pose_selected


def joint_pos_multi_future_select_joints(
    env: ManagerBasedEnv, command_name: str, joints_idx: list, non_flatten=True
) -> torch.Tensor:
    """Extract reference joint positions for selected joints across future frames.

    Args:
        command_name: Name of the tracking command term.
        joints_idx: Indices of joints to select from the full joint position vector.
        non_flatten: If True, return shape (num_envs, num_future_frames, len(joints_idx)).

    Returns:
        torch.Tensor: Selected joint positions.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos = command.joint_pos_multi_future
    # Flatten case: (num_envs, num_future_frames * num_joints)
    joint_pos_reshaped = joint_pos.view(env.num_envs, command.num_future_frames, -1)
    joint_pos_selected = joint_pos_reshaped[..., joints_idx]
    if not non_flatten:
        joint_pos_selected = joint_pos_selected.view(env.num_envs, -1)
    return joint_pos_selected


def joint_pos_multi_future_select_joints_for_smpl(
    env: ManagerBasedEnv, command_name: str, joints_idx: list
) -> torch.Tensor:
    """Extract reference joint positions for selected joints across SMPL-aligned future frames.

    Uses smpl_num_future_frames (which may differ from num_future_frames) for
    alignment with SMPL motion data.

    Args:
        command_name: Name of the tracking command term.
        joints_idx: Indices of joints to select.

    Returns:
        torch.Tensor: Selected joint positions,
            shape (num_envs, smpl_num_future_frames, len(joints_idx)).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos = command.joint_pos_multi_future_for_smpl
    # Flatten case: (num_envs, num_future_frames * num_joints)
    joint_pos_reshaped = joint_pos.view(env.num_envs, command.smpl_num_future_frames, -1)
    joint_pos_selected = joint_pos_reshaped[..., joints_idx]
    return joint_pos_selected


def joint_pos_multi_future_select_joints_for_smpl(  # noqa: F811
    env: ManagerBasedEnv, command_name: str, joints_idx: list
) -> torch.Tensor:
    """Extract reference joint positions for selected joints across SMPL-aligned future frames.

    NOTE: This is a duplicate definition that shadows the previous one. Only this
    version is active at runtime.

    Args:
        command_name: Name of the tracking command term.
        joints_idx: Indices of joints to select.

    Returns:
        torch.Tensor: Selected joint positions,
            shape (num_envs, smpl_num_future_frames, len(joints_idx)).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos = command.joint_pos_multi_future_for_smpl
    # Flatten case: (num_envs, num_future_frames * num_joints)
    joint_pos_reshaped = joint_pos.view(env.num_envs, command.smpl_num_future_frames, -1)
    joint_pos_selected = joint_pos_reshaped[..., joints_idx]
    return joint_pos_selected


def smpl_elbow_wrist_pose_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Extract SMPL elbow and wrist joint poses for multiple future frames.

    Selects indices 54:66 from the SMPL pose vector, corresponding to
    left elbow, left wrist, right elbow, right wrist (4 joints x 3 axis-angle).

    Returns:
        torch.Tensor: Elbow/wrist poses, shape (..., 12).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    smpl_pose = command.smpl_pose_multi_future
    elbow_wrist_pose = smpl_pose[..., 54:66]
    return elbow_wrist_pose


def smpl_root_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get SMPL root orientation relative to robot anchor as 6D rotation.

    Returns:
        torch.Tensor: 6D rotation representation, shape (num_envs, 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        None,
        command.smpl_root_quat_w,
    )
    # diff = quat_mul(command.smpl_root_quat_w, quat_inv(command.anchor_quat_w))
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def smpl_root_ori_b_mf(env: ManagerBasedEnv, command_name: str, non_flatten=True) -> torch.Tensor:
    """Get SMPL root orientation difference in body-local frame for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, smpl_num_future_frames, 6).

    Returns:
        torch.Tensor: 6D rotation differences.
    """
    # ZL: non-flatten set to true temporarily to match previous jobs, will be set to false in the future.
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        rot = command.smpl_root_quat_w_dif_l_multi_future.reshape(
            env.num_envs, command.smpl_num_future_frames, -1
        )
    else:
        rot = command.smpl_root_quat_w_dif_l_multi_future.reshape(env.num_envs, -1)
    return rot


def smpl_root_ori_refheading_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=True
) -> torch.Tensor:
    """SMPL root orientation canonicalized by the first SMPL future frame's heading.

    Uses the heading of the first SMPL future frame instead of the robot's orientation
    for canonicalization.

    Returns:
        torch.Tensor: 6D rotation matrix representation,
            shape (num_envs, smpl_num_future_frames, 6) if non_flatten else
            (num_envs, smpl_num_future_frames * 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.smpl_root_quat_w_dif_refheading_multi_future.reshape(
            env.num_envs, command.smpl_num_future_frames, -1
        )
    else:
        return command.smpl_root_quat_w_dif_refheading_multi_future.reshape(env.num_envs, -1)


def smpl_root_ori_heading_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=True
) -> torch.Tensor:
    """SMPL root orientation canonicalized by robot heading (yaw) only.

    Uses the robot's heading (yaw) instead of full orientation for canonicalization,
    preserving the SMPL root's pitch/roll relative to gravity.

    Returns:
        torch.Tensor: 6D rotation matrix representation,
            shape (num_envs, smpl_num_future_frames, 6) if non_flatten else
            (num_envs, smpl_num_future_frames * 6)
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.smpl_root_quat_w_dif_heading_multi_future.reshape(
            env.num_envs, command.smpl_num_future_frames, -1
        )
    else:
        return command.smpl_root_quat_w_dif_heading_multi_future.reshape(env.num_envs, -1)


def smpl_joints_multi_future(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get SMPL joint positions relative to SMPL root for multiple future frames.

    Canonicalizes joint positions using the first SMPL frame's root orientation.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, smpl_num_future_frames, num_joints * 3).

    Returns:
        torch.Tensor: Root-relative SMPL joint positions.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_joints = command.smpl_joints_multi_future
    ref_root_quat = command.smpl_root_quat_w.view(env.num_envs, 1, 1, 4).repeat(
        1, command.num_future_frames, ref_joints.shape[-2], 1
    )
    ref_joints_root = quat_apply(quat_inv(ref_root_quat), ref_joints)
    if non_flatten:
        return ref_joints_root.reshape(env.num_envs, command.smpl_num_future_frames, -1)
    else:
        return ref_joints_root.view(env.num_envs, -1)


def smpl_joints_multi_future_local(
    env: ManagerBasedEnv, command_name: str, non_flatten=False, joints_idx=None
) -> torch.Tensor:
    """Get SMPL joint positions relative to each frame's own root orientation.

    Unlike smpl_joints_multi_future which uses only the first frame's root,
    this version canonicalizes each future frame independently using its own
    root orientation. Optionally select a subset of joints.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, smpl_num_future_frames, num_joints * 3).
        joints_idx: Optional list of joint indices to select.

    Returns:
        torch.Tensor: Per-frame root-relative SMPL joint positions.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    ref_joints = command.smpl_joints_multi_future
    ref_root_quat = command.smpl_root_quat_w_multi_future.unsqueeze(-2).repeat(
        1, 1, ref_joints.shape[-2], 1
    )
    ref_joints_root = quat_apply(quat_inv(ref_root_quat), ref_joints)
    if joints_idx is not None:
        ref_joints_root = ref_joints_root[..., joints_idx, :]
    if non_flatten:
        return ref_joints_root.reshape(env.num_envs, command.smpl_num_future_frames, -1)
    else:
        return ref_joints_root.view(env.num_envs, -1)


def smpl_joints_lower_multi_future_local(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get lower-body SMPL joint positions relative to each frame's own root orientation.

    Selects joints [0,1,2,4,5,7,8,10,11] (hips, knees, ankles) and canonicalizes
    each future frame using its own root orientation.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, smpl_num_future_frames, 9 * 3).

    Returns:
        torch.Tensor: Per-frame root-relative lower-body SMPL joint positions.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    ref_joints = command.smpl_joints_multi_future[
        :, :, [0, 1, 2, 4, 5, 7, 8, 10, 11], :
    ]  # only first 12 joints (lower body)
    ref_root_quat = command.smpl_root_quat_w_multi_future.unsqueeze(-2).repeat(
        1, 1, ref_joints.shape[-2], 1
    )
    ref_joints_root = quat_apply(quat_inv(ref_root_quat), ref_joints)
    if non_flatten:
        return ref_joints_root.reshape(env.num_envs, command.smpl_num_future_frames, -1)
    else:
        return ref_joints_root.view(env.num_envs, -1)


def smpl_transl_z_multi_future(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get SMPL root translation z-height for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, preserve per-frame dimension.

    Returns:
        torch.Tensor: Z-heights, shape (num_envs, smpl_num_future_frames) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_transl_z = command.smpl_transl_z_multi_future
    if non_flatten:
        return ref_transl_z
    else:
        return ref_transl_z.view(env.num_envs, -1)


def soma_joints_multi_future_local(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """SOMA skeleton joint positions canonicalized to each frame's root orientation.

    Mirrors smpl_joints_multi_future_local but uses SOMA skeleton (26 joints).
    Root quat has Y→Z up conversion and base rotation removal applied.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    ref_joints = command.soma_joints_multi_future  # (batch, num_frames, 26, 3)
    ref_root_quat = command.soma_root_quat_w_multi_future.unsqueeze(-2).repeat(
        1, 1, ref_joints.shape[-2], 1  # (batch, num_frames, 26, 4)
    )
    ref_joints_root = quat_apply(quat_inv(ref_root_quat), ref_joints)
    if non_flatten:
        return ref_joints_root.reshape(env.num_envs, command.smpl_num_future_frames, -1)
    else:
        return ref_joints_root.view(env.num_envs, -1)


def soma_root_ori_b_mf(env: ManagerBasedEnv, command_name: str, non_flatten=True):
    """SOMA root orientation relative to robot anchor, as 6D rotation matrix.

    Returns:
        (num_envs, soma_num_future_frames, 6) if non_flatten
        (num_envs, soma_num_future_frames * 6) otherwise
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        rot = command.soma_root_quat_w_dif_l_multi_future.reshape(
            env.num_envs, command.smpl_num_future_frames, -1
        )
    else:
        rot = command.soma_root_quat_w_dif_l_multi_future.reshape(env.num_envs, -1)
    return rot


def compliance(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get end-effector compliance (stiffness) scaled for observation input.

    Returns:
        torch.Tensor: Scaled stiffness values, shape (num_envs, num_eef).
    """
    command: commands.ForceTrackingCommand = env.command_manager.get_term(command_name)
    return command.eef_stiffness_buf * 10.0  # rescaling observation


def ext_forces(
    env: ManagerBasedEnv, force_command_name: str, motion_command_name: str
) -> torch.Tensor:
    """Get last-applied external forces in heading-local frame.

    Transforms the world-frame external forces into a heading-invariant frame
    by removing the robot's yaw rotation.

    Returns:
        torch.Tensor: Flattened forces, shape (num_envs, num_force_bodies * 3).
    """
    force_command: commands.ForceTrackingCommand = env.command_manager.get_term(force_command_name)
    motion_command: commands.TrackingCommand = env.command_manager.get_term(motion_command_name)
    ext_force_w = force_command.last_force_applied
    root_quat = motion_command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, ext_force_w.shape[1], 1
    )
    deheaded_force = quat_apply_yaw(quat_inv(root_quat), ext_force_w)
    return deheaded_force.view(env.num_envs, -1)


# =============================================================================
# Body-only observation functions (for pre-trained action_transform_module)
# Uses get_body_joint_indices from joint_utils.py


def joint_pos_wo_hand(env: ManagerBasedEnv, asset_cfg) -> torch.Tensor:
    """Get joint positions excluding hand joints (29 DOF body only)."""
    asset = env.scene[asset_cfg.name]
    body_indices = joint_utils.get_body_joint_indices(asset)
    return asset.data.joint_pos[:, body_indices] - asset.data.default_joint_pos[:, body_indices]


def joint_vel_wo_hand(env: ManagerBasedEnv, asset_cfg) -> torch.Tensor:
    """Get joint velocities excluding hand joints (29 DOF body only)."""
    asset = env.scene[asset_cfg.name]
    body_indices = joint_utils.get_body_joint_indices(asset)
    return asset.data.joint_vel[:, body_indices] - asset.data.default_joint_vel[:, body_indices]


def last_action_wo_hand(env: ManagerBasedEnv, asset_cfg) -> torch.Tensor:
    """Get last actions excluding hand joints."""
    asset = env.scene[asset_cfg.name]
    body_indices = joint_utils.get_body_joint_indices(asset)
    return env.action_manager.action[:, body_indices]


def last_meta_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Get last meta action (policy output: latent residual + finger primitives).

    This returns the policy's output from the previous step, not the joint-level
    actions that were applied to the simulation. For a student policy using
    latent residual mode, this is typically:
    - 64 dims: latent residual (tokenizer space)
    - 2 dims: finger primitive actions (left + right hand)
    Total: 66 dims

    The buffer is stored on env by the ManagerEnvWrapper and initialized to zeros
    on reset.
    """
    if hasattr(env, "_last_meta_action"):
        return env._last_meta_action  # noqa: SLF001
    else:
        # Fallback: return zeros if buffer not initialized (shouldn't happen)
        meta_action_dim = 66  # Default: 64 latent + 2 primitives
        return torch.zeros(env.num_envs, meta_action_dim, dtype=torch.float32, device=env.device)


def ref_root_pos_future_b(env, command_name: str, flatten: bool = False) -> torch.Tensor:
    """Get reference root positions for future frames in robot anchor frame.

    Args:
        command_name: Name of the tracking command term.
        flatten: If True, return shape (num_envs, num_future_frames * 3).

    Returns:
        torch.Tensor: Reference root positions,
            shape (num_envs, num_future_frames, 3) or flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_pos_future_w = command.anchor_pos_w_multi_future.view(
        command.num_envs, command.num_future_frames, -1
    )
    robot_root_pos_w = command.robot_anchor_pos_w.unsqueeze(1)
    robot_root_quat_w = command.robot_anchor_quat_w.unsqueeze(1)
    robot_root_quat_w = robot_root_quat_w.expand(-1, command.num_future_frames, -1)

    ref_root_pos_future_b = quat_apply_inverse(
        robot_root_quat_w, ref_root_pos_future_w - robot_root_pos_w
    )
    if flatten:
        return ref_root_pos_future_b.reshape(command.num_envs, -1)
    return ref_root_pos_future_b.reshape(command.num_envs, command.num_future_frames, -1)


def ref_root_ori_future_b(env, command_name: str, flatten: bool = False) -> torch.Tensor:
    """Get reference root orientations for future frames in robot anchor frame as 6D rotation.

    Args:
        command_name: Name of the tracking command term.
        flatten: If True, return shape (num_envs, num_future_frames * 6).

    Returns:
        torch.Tensor: 6D rotation representations,
            shape (num_envs, num_future_frames, 6) or flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat_future_w = command.anchor_quat_w_multi_future.view(
        command.num_envs, command.num_future_frames, -1
    )
    robot_root_quat_w = command.robot_anchor_quat_w.unsqueeze(1)
    robot_root_quat_w = robot_root_quat_w.expand(-1, command.num_future_frames, -1)

    ref_root_quat_future_b = quat_mul(
        quat_conjugate(robot_root_quat_w),
        ref_root_quat_future_w,
    )
    ref_root_ori_future_b = matrix_from_quat(ref_root_quat_future_b)
    ref_root_ori_future_b = ref_root_ori_future_b[:, :, :2, :]
    if flatten:
        return ref_root_ori_future_b.reshape(command.num_envs, -1)
    return ref_root_ori_future_b.reshape(command.num_envs, command.num_future_frames, -1)


def diff_body_pos_future_local(env, command_name: str, flatten: bool = False) -> torch.Tensor:
    """Compute reference-minus-robot body position differences in their respective heading frames.

    Reference body positions are expressed in the reference motion's heading (yaw-only,
    z=0 projected) root frame; robot body positions are expressed in the robot's heading
    root frame. The difference captures the tracking error per body per future frame.

    Args:
        command_name: Name of the tracking command term.
        flatten: If True, return shape (num_envs, num_future_frames * num_bodies * 3).

    Returns:
        torch.Tensor: Position differences,
            shape (num_envs, num_future_frames, num_bodies * 3) or flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_body_pos_future_w = command.body_pos_w_multi_future.view(
        command.num_envs, command.num_future_frames, command.num_bodies, -1
    )

    ref_root_pos_w = command.anchor_pos_w.unsqueeze(1).unsqueeze(2)
    # shape: (num_envs, 1, 1, 3)
    ref_root_quat_w = command.anchor_quat_w.unsqueeze(1).unsqueeze(2)
    # shape: (num_envs, 1, 1, 4)
    ref_root_pos_w = ref_root_pos_w.clone()
    ref_root_pos_w[..., 2] = 0.0
    ref_root_quat_w = torch_transform.get_heading_q(ref_root_quat_w)
    ref_root_quat_w = ref_root_quat_w.expand(
        command.num_envs, command.num_future_frames, len(command.cfg.body_names), -1
    )

    robot_body_pos_w = command.robot_body_pos_w.view(command.num_envs, command.num_bodies, -1)

    robot_root_pos_w = command.robot_anchor_pos_w.unsqueeze(1)
    robot_root_quat_w = command.robot_anchor_quat_w.unsqueeze(1)
    robot_root_pos_w = robot_root_pos_w.clone()
    robot_root_pos_w[..., 2] = 0.0
    robot_root_quat_w = torch_transform.get_heading_q(robot_root_quat_w)
    robot_root_quat_w = robot_root_quat_w.expand(command.num_envs, len(command.cfg.body_names), -1)

    robot_body_pos_local = quat_apply_inverse(
        robot_root_quat_w, robot_body_pos_w - robot_root_pos_w
    )
    ref_body_pos_future_local = quat_apply_inverse(
        ref_root_quat_w, ref_body_pos_future_w - ref_root_pos_w
    )

    diff = ref_body_pos_future_local - robot_body_pos_local.unsqueeze(1)
    if flatten:
        return diff.reshape(command.num_envs, -1)
    return diff.reshape(command.num_envs, command.num_future_frames, -1)


def diff_body_ori_future_local(env, command_name: str, flatten: bool = False) -> torch.Tensor:
    """Compute reference-minus-robot body orientation differences in their respective heading frames.

    Both reference and robot body orientations are first canonicalized by their respective
    heading (yaw-only) root frames, then the relative rotation between them is computed
    as a 6D rotation representation.

    Args:
        command_name: Name of the tracking command term.
        flatten: If True, return shape (num_envs, num_future_frames * num_bodies * 6).

    Returns:
        torch.Tensor: 6D orientation differences,
            shape (num_envs, num_future_frames, num_bodies * 6) or flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_body_quat_future_w = command.body_quat_w_multi_future.view(
        command.num_envs, command.num_future_frames, command.num_bodies, -1
    )

    ref_root_quat_w = command.anchor_quat_w.unsqueeze(1).unsqueeze(2)
    ref_root_quat_w = torch_transform.get_heading_q(ref_root_quat_w)
    ref_root_quat_w = ref_root_quat_w.expand(
        command.num_envs, command.num_future_frames, len(command.cfg.body_names), -1
    )

    robot_body_quat_w = command.robot_body_quat_w.view(command.num_envs, command.num_bodies, -1)

    robot_root_quat_w = command.robot_anchor_quat_w.unsqueeze(1)
    robot_root_quat_w = torch_transform.get_heading_q(robot_root_quat_w)
    robot_root_quat_w = robot_root_quat_w.expand(command.num_envs, len(command.cfg.body_names), -1)

    robot_body_quat_local = quat_mul(
        quat_conjugate(robot_root_quat_w),
        robot_body_quat_w,
    ).unsqueeze(1)
    ref_body_quat_future_local = quat_mul(
        quat_conjugate(ref_root_quat_w),
        ref_body_quat_future_w,
    )

    diff_body_quat_future = quat_mul(
        quat_conjugate(robot_body_quat_local).expand_as(ref_body_quat_future_local),
        ref_body_quat_future_local,
    )
    diff_body_ori_future_local = matrix_from_quat(diff_body_quat_future)
    diff = diff_body_ori_future_local[:, :, :, :2, :]
    if flatten:
        return diff.reshape(command.num_envs, -1)
    return diff.reshape(command.num_envs, command.num_future_frames, -1)


def diff_body_lin_vel_future_local(env, command_name: str, flatten: bool = False) -> torch.Tensor:
    """Compute reference-minus-robot body linear velocity differences in heading frames.

    Both reference and robot velocities are expressed in their respective heading
    (yaw-only) root frames. Output is clamped to [-25, 25] for numerical stability.

    Args:
        command_name: Name of the tracking command term.
        flatten: If True, return shape (num_envs, num_future_frames * num_bodies * 3).

    Returns:
        torch.Tensor: Velocity differences,
            shape (num_envs, num_future_frames, num_bodies * 3) or flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_body_lin_vel_future_w = command.motion_lib.get_body_lin_vel_w(
        command.future_motion_ids, command.future_time_steps
    ).view(command.num_envs, command.num_future_frames, -1, 3)

    ref_root_quat_w = command.anchor_quat_w.unsqueeze(1).unsqueeze(2)
    # shape: (num_envs, 1, 1, 4)
    ref_root_quat_w = torch_transform.get_heading_q(ref_root_quat_w)
    ref_root_quat_w = ref_root_quat_w.expand(
        command.num_envs, command.num_future_frames, command.num_bodies, -1
    )
    ref_body_lin_vel_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_lin_vel_future_w)

    robot_body_lin_vel_w = command.robot_body_lin_vel_w
    robot_root_quat_w = command.robot_anchor_quat_w.unsqueeze(1)
    robot_root_quat_w = torch_transform.get_heading_q(robot_root_quat_w).expand(
        command.num_envs, command.num_bodies, -1
    )
    robot_body_lin_vel_local = quat_apply_inverse(robot_root_quat_w, robot_body_lin_vel_w)

    diff = ref_body_lin_vel_future_local - robot_body_lin_vel_local.unsqueeze(1)
    diff.clamp_(min=-25, max=25)
    if flatten:
        return diff.reshape(command.num_envs, -1)
    return diff.reshape(command.num_envs, command.num_future_frames, -1)


def diff_body_ang_vel_future_local(env, command_name: str, flatten: bool = False) -> torch.Tensor:
    """Compute reference-minus-robot body angular velocity differences in heading frames.

    Both reference and robot angular velocities are expressed in their respective heading
    (yaw-only) root frames. Output is clamped to [-25, 25] for numerical stability.

    Args:
        command_name: Name of the tracking command term.
        flatten: If True, return shape (num_envs, num_future_frames * num_bodies * 3).

    Returns:
        torch.Tensor: Angular velocity differences,
            shape (num_envs, num_future_frames, num_bodies * 3) or flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_body_ang_vel_future_w = command.motion_lib.get_body_ang_vel_w(
        command.future_motion_ids, command.future_time_steps
    ).view(command.num_envs, command.num_future_frames, -1, 3)

    ref_root_quat_w = command.anchor_quat_w.unsqueeze(1).unsqueeze(2)
    # shape: (num_envs, 1, 1, 4)
    ref_root_quat_w = torch_transform.get_heading_q(ref_root_quat_w)
    ref_root_quat_w = ref_root_quat_w.expand(
        command.num_envs, command.num_future_frames, command.num_bodies, -1
    )
    ref_body_ang_vel_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_ang_vel_future_w)

    robot_body_ang_vel_w = command.robot_body_ang_vel_w
    robot_root_quat_w = command.robot_anchor_quat_w.unsqueeze(1)
    robot_root_quat_w = torch_transform.get_heading_q(robot_root_quat_w).expand(
        command.num_envs, command.num_bodies, -1
    )
    robot_body_ang_vel_local = quat_apply_inverse(robot_root_quat_w, robot_body_ang_vel_w)

    diff = ref_body_ang_vel_future_local - robot_body_ang_vel_local.unsqueeze(1)
    diff.clamp_(min=-25, max=25)
    if flatten:
        return diff.reshape(command.num_envs, -1)
    return diff.reshape(command.num_envs, command.num_future_frames, -1)


def height_map(env: ManagerBasedEnv, command_name, random=False) -> torch.Tensor:
    """Get height map observation from the environment."""
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)

    if not hasattr(command, "scan_dot_pos_w"):
        # Height map disabled - return zeros with expected shape
        n = int(command.cfg.height_map_size / command.cfg.height_map_resolution) + 1
        return torch.zeros(command.num_envs, n, n, 3, device=command.device)

    if not hasattr(command, "scan_dot_pos_w"):
        # Height map disabled - return zeros with expected shape
        n = int(command.cfg.height_map_size / command.cfg.height_map_resolution) + 1
        return torch.zeros(command.num_envs, n, n, 3, device=command.device)

    robot_root_pos_w, robot_root_quat_w = (
        command.robot.data.root_pos_w,
        command.robot.data.root_quat_w,
    )
    scan_dot_pos_w = command.scan_dot_pos_w

    robot_root_quat_w_yaw = (
        torch_transform.get_heading_q(robot_root_quat_w)
        .unsqueeze(1)
        .unsqueeze(2)
        .expand(-1, command.num_rays_x, command.num_rays_y, -1)
    )
    obs = quat_apply_inverse(
        robot_root_quat_w_yaw, scan_dot_pos_w - robot_root_pos_w.unsqueeze(1).unsqueeze(2)
    )
    obs.nan_to_num_(0.0)
    if random:
        obs = torch.rand_like(obs)
    return obs


def height_map_flat(env: ManagerBasedEnv, command_name="motion", random=False) -> torch.Tensor:
    """Flattened height map for concatenation with 1D policy observations."""
    hmap = height_map(env, command_name=command_name, random=random)
    return hmap.reshape(hmap.shape[0], -1)  # (B, n*n*3)


def residual_joint_pos_action(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Compute the action-space representation of reference joint positions.

    Converts the reference motion's joint positions into the normalized action space
    using the action manager's offset and scale.

    Returns:
        torch.Tensor: Normalized residual actions, shape (num_envs, num_action_joints).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    action_manager = env.action_manager.get_term("joint_pos")
    action_offset = action_manager._offset  # noqa: SLF001
    action_scale = action_manager._scale  # noqa: SLF001

    motion_joint_pos = command.joint_pos
    # motion_joint_names = command.robot.joint_names
    # action_joint_names = action_manager._joint_names
    action_joint_pos = motion_joint_pos[:, action_manager._joint_ids]  # noqa: SLF001
    residual_action = (action_joint_pos - action_offset) / action_scale
    return residual_action


# =============================================================================
# Task stage observation (for staged training)


def get_task_stage(env: ManagerBasedEnv) -> torch.Tensor:
    """Get current task stage."""
    if hasattr(env, "task_stage"):
        return env.task_stage.float().unsqueeze(-1)
    return torch.zeros(env.num_envs, 1, dtype=torch.float, device=env.device)


def get_tiled_camera_image(
    env: ManagerBasedEnv,
    camera_cfg: SceneEntityCfg,
    normalize: bool = True,
    normalize_mean: list[float] = [0.485, 0.456, 0.406],  # ImageNet mean (RGB)
    normalize_std: list[float] = [0.229, 0.224, 0.225],  # ImageNet std (RGB)
    debug_visualize: bool = False,  # Enable real-time visualization with cv2
    debug_env_idx: int = 0,  # Which environment's image to visualize
    debug_show_predicted: bool = False,  # Also show predicted object position (in red)
) -> torch.Tensor:
    """Get RGB image from tiled camera sensor.

    Returns normalized RGB image of shape (num_envs, H, W, 3) - NOT flattened.
    Image values can be normalized with configurable mean and std (default: ImageNet stats).

    Args:
        env: The environment object
        camera_cfg: Camera configuration with sensor name
        normalize: Whether to apply mean/std normalization (default: True)
        normalize_mean: RGB mean values for normalization (default: ImageNet [0.485, 0.456, 0.406])
        normalize_std: RGB std values for normalization (default: ImageNet [0.229, 0.224, 0.225])
        debug_visualize: Whether to display the image in real-time using cv2 (default: False)
        debug_env_idx: Which environment's image to visualize (default: 0)
        debug_show_predicted: Whether to also show predicted object position in red (default: False)

    Returns:
        RGB image tensor of shape (num_envs, H, W, 3) - channels last format
    """
    camera = env.scene[camera_cfg.name]

    # Get RGB image from camera - shape is (num_envs, H, W, 4) with RGBA
    rgb_image = camera.data.output["rgb"]

    # Take only RGB channels (drop alpha if present)
    if rgb_image.shape[-1] == 4:
        rgb_image = rgb_image[..., :3]

    # Debug visualization (before normalization)
    if debug_visualize:
        utils.debug_visualize_object_projection(
            env, camera, rgb_image, debug_env_idx, show_predicted=debug_show_predicted
        )

    # Convert to float if needed
    if rgb_image.dtype != torch.float:
        rgb_image = rgb_image.float()

    # Normalize to [0, 1] if in [0, 255] range
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0

    # Apply mean/std normalization if enabled
    if normalize:
        mean = torch.tensor(normalize_mean, device=rgb_image.device, dtype=rgb_image.dtype)
        std = torch.tensor(normalize_std, device=rgb_image.device, dtype=rgb_image.dtype)
        rgb_image = (rgb_image - mean) / std

    # Return shape: (num_envs, H, W, 3) - do NOT flatten
    # The vision encoder will handle permuting to (B, C, H, W) if needed
    return rgb_image
