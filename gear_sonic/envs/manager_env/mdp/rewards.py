"""Reward functions for the manager-based RL environment MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
)
import torch

from gear_sonic.envs.manager_env.mdp.commands import (
    ForceTrackingCommand,
    TrackingCommand,
    _get_body_indexes,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    tracking_anchor_pos = None
    tracking_anchor_ori = None
    tracking_relative_body_pos = None
    tracking_relative_body_ori = None
    tracking_relative_body_ori_weighted = None
    tracking_body_linvel = None
    tracking_body_angvel = None
    action_rate_l2 = None
    joint_limit = None
    undesired_contacts = None
    undesired_contacts_no_hands = None
    undesired_contacts_no_ankle_hand = None
    tracking_body_pos = None
    tracking_body_ori = None
    tracking_vr_3point_global = None
    tracking_vr_3point_local = None
    tracking_vr_3point_force = None
    tracking_vr_2wrists_ori_tight = None
    tracking_vr_2wrists_local_ori = None
    tracking_head_local_ori = None
    anti_shake_ang_vel = None
    tracking_vr_5point_local = None
    motion_5point_local_pos = None
    feet_acc = None
    is_terminated = None
    upright_penalty = None


def tracking_anchor_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute anchor position tracking reward using a Gaussian kernel.

    Encourages the robot's anchor (root) position to match the reference motion anchor.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel. Smaller values produce
            sharper falloff and stricter tracking.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    diff = command.anchor_pos_w - command.robot_anchor_pos_w
    sq_dist = (diff * diff).sum(dim=-1)
    return torch.exp(-sq_dist / (std * std))


def tracking_anchor_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute anchor orientation tracking reward using a Gaussian kernel.

    Encourages the robot's anchor (root) orientation to match the reference motion anchor.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    angular_err = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    return torch.exp(-angular_err.square() / (std * std))


def upright_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_name: str | None = None,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize tilt of bodies away from upright.

    Compute the squared magnitude of the x/y components of the gravity vector
    in each body's local frame, summed across all specified bodies. When a body
    is perfectly upright the local gravity is [0, 0, -1] and the penalty is 0.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        body_name: Single body name (for backwards compatibility).
        body_names: List of body names. If both are None, defaults to ["pelvis"].

    Returns:
        Penalty tensor of shape (num_envs,). Zero when upright, positive when tilted.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]

    if body_names is None:
        body_names = [body_name] if body_name else ["pelvis"]

    total_penalty = torch.zeros(env.num_envs, device=env.device)
    for name in body_names:
        body_idx = robot.body_names.index(name)
        body_quat = robot.data.body_quat_w[:, body_idx]
        g_local = quat_apply(quat_inv(body_quat), command.down_dir)
        total_penalty += g_local[:, 0] ** 2 + g_local[:, 1] ** 2

    return total_penalty


def tracking_body_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body position tracking reward in world frame using a Gaussian kernel.

    Encourages tracked body positions to match the reference motion. The reward is
    the mean squared distance across all tracked bodies, passed through an exponential.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_vr_3point_error(env: ManagerBasedRLEnv, command_name: str, std: float):
    """Compute VR 3-point tracking reward in world frame using a Gaussian kernel.

    Encourages the robot's 3 VR tracking points (typically left wrist, right wrist,
    head) to match their reference positions in world frame.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pos_diff = command.robot_vr_3point_pos_w - command.vr_3point_body_pos_w
    per_point_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_point_err.mean(dim=-1) / (std * std))


def tracking_vr_2wrists_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute wrist orientation tracking reward in world frame.

    Measure the orientation error of 2 wrist bodies against the reference motion,
    similar to tracking_relative_body_ori_error but restricted to wrist links.

    NOTE: The rigid extension defined in vr_3point_body_offset can be skipped for
    orientation error since it does not affect rotations.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: List of wrist body names (must be provided).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    assert body_names is not None, "body_names must be provided"
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_w[:, tracked], command.robot_body_quat_w[:, tracked]
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_local_vr_2wrists_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute wrist orientation tracking reward in the anchor's local frame.

    Transform both reference and robot wrist orientations into the anchor (root)
    frame before computing the angular error. This makes the reward invariant to
    global root orientation.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: List of wrist body names (must be provided).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    assert body_names is not None, "body_names must be provided"
    body_indexes = _get_body_indexes(command, body_names)
    num_bodies = len(body_indexes)

    # reference motion
    ref_wrist_quat_w = command.body_quat_w[:, body_indexes]
    ref_anchor_quat_w = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, num_bodies, 1)
    ref_wrist_quat_local = quat_mul(quat_inv(ref_anchor_quat_w), ref_wrist_quat_w)

    # robot
    robot_wrist_quat_w = command.robot_body_quat_w[:, body_indexes]
    robot_anchor_quat_w = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, num_bodies, 1
    )
    robot_wrist_quat_local = quat_mul(quat_inv(robot_anchor_quat_w), robot_wrist_quat_w)

    error = quat_error_magnitude(ref_wrist_quat_local, robot_wrist_quat_local) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def tracking_local_head_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute head orientation tracking reward in the anchor's local frame.

    Transform the head (torso_link) orientation into the anchor's local frame for
    both the reference motion and the robot, then compute the angular error. This
    encourages the robot to match the head-to-root relative orientation.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get the head body index (torso_link)
    head_body_names = ["torso_link"]
    body_indexes = _get_body_indexes(command, head_body_names)

    # reference motion: head orientation in world frame, transformed to anchor's local frame
    ref_head_quat_w = command.body_quat_w[:, body_indexes]  # [num_envs, 1, 4]
    ref_anchor_quat_w = command.anchor_quat_w.view(env.num_envs, 1, 4)
    ref_head_quat_local = quat_mul(quat_inv(ref_anchor_quat_w), ref_head_quat_w)

    # robot: head orientation in world frame, transformed to anchor's local frame
    robot_head_quat_w = command.robot_body_quat_w[:, body_indexes]  # [num_envs, 1, 4]
    robot_anchor_quat_w = command.robot_anchor_quat_w.view(env.num_envs, 1, 4)
    robot_head_quat_local = quat_mul(quat_inv(robot_anchor_quat_w), robot_head_quat_w)

    error = quat_error_magnitude(ref_head_quat_local, robot_head_quat_local) ** 2
    return torch.exp(-error.squeeze(-1) / std**2)


def tracking_local_vr_3point_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    point_weights: list[float] | None = None,
):
    """Compute VR 3-point tracking reward in the anchor's local frame.

    Transform tracking points into the anchor (root) local frame before computing
    position error, making the reward invariant to global root position/orientation.
    Supports optional per-point weighting.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        point_weights: Optional weights for each tracking point. Order matches
            vr_3point_body config (typically [left_wrist, right_wrist, head]).
            If None, all points are weighted equally.
            Example: [2, 2, 1] gives wrists 2x importance vs head.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_pos = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    robot_root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    robot_3point_diff = command.robot_vr_3point_pos_w - command.robot_anchor_pos_w[:, None, :]
    robot_3point_pos = quat_apply(quat_inv(robot_root_quat), robot_3point_diff)
    diff = robot_3point_pos - ref_3point_pos
    error = torch.sum(torch.square(diff), dim=-1)  # [num_envs, num_points]

    if point_weights is not None:
        # Weighted mean: sum(w_i * e_i) / sum(w_i)
        weights = torch.tensor(point_weights, dtype=error.dtype, device=error.device)
        weighted_error = (error * weights).sum(dim=-1) / weights.sum()
    else:
        # Simple mean (equal weights)
        weighted_error = error.mean(dim=-1)

    return torch.exp(-weighted_error / std**2)


def tracking_local_vr_5point_error(env: ManagerBasedRLEnv, command_name: str, std: float):
    """Compute VR 5-point tracking reward in the anchor's local frame.

    Same approach as tracking_local_vr_3point_error but with 5 tracking points
    (e.g., 2 wrists + head + 2 feet).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ref_5point_diff = command.reward_point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.reward_point_body), 1
    )
    ref_5point_pos = quat_apply(quat_inv(ref_root_quat), ref_5point_diff)
    robot_root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.reward_point_body), 1
    )
    robot_5point_diff = (
        command.robot_reward_point_body_pos_w - command.robot_anchor_pos_w[:, None, :]
    )
    robot_5point_pos = quat_apply(quat_inv(robot_root_quat), robot_5point_diff)
    diff = robot_5point_pos - ref_5point_pos
    error = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def tracking_vr_3point_error_pos_force(
    env: ManagerBasedRLEnv, motion_command_name: str, force_command_name: str, std: float
):
    """Compute VR 3-point tracking reward with force-based compliance correction.

    Add a force-proportional offset to the wrist tracking error so that applied
    external forces shift the tracking target, enabling compliant behavior under
    force perturbations.

    Args:
        env: The environment.
        motion_command_name: Name of the motion tracking command term.
        force_command_name: Name of the force tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    motion_command: TrackingCommand = env.command_manager.get_term(motion_command_name)
    force_command: ForceTrackingCommand = env.command_manager.get_term(force_command_name)
    diff = motion_command.robot_vr_3point_pos_w - motion_command.vr_3point_body_pos_w
    force_error_wrists = (
        force_command.last_force_applied * force_command.eef_stiffness_buf[:, :, None]
    )
    diff[:, :2] += force_error_wrists
    error = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def tracking_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body orientation tracking reward in world frame using a Gaussian kernel.

    Encourages tracked body orientations to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_w[:, tracked], command.robot_body_quat_w[:, tracked]
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_relative_body_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body position tracking reward using anchor-relative reference positions.

    Use reference body positions that have been shifted to share the robot's anchor
    (root) position, so only the relative pose matters rather than absolute position.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_relative_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_relative_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body orientation tracking reward using anchor-relative reference orientations.

    Use reference body orientations that have been transformed to share the robot's
    anchor (root) orientation, making the reward invariant to global heading.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_relative_w[:, tracked],
        command.robot_body_quat_w[:, tracked],
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_relative_body_ori_weighted_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    body_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute anchor-relative body orientation tracking reward with per-body weights.

    Same as tracking_relative_body_ori_error but allows different bodies to contribute
    differently to the mean error. Useful for relaxing tracking on certain joints
    (e.g., wrists during manipulation).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.
        body_weights: Dict mapping body name to weight multiplier. Bodies not listed
            default to 1.0. E.g. {"left_wrist_yaw_link": 0.1} to relax wrist tracking.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    if body_weights is not None:
        tracked_names = [command.cfg.body_names[i] for i in body_indexes]
        weights = torch.tensor(
            [body_weights.get(name, 1.0) for name in tracked_names],
            device=error.device,
            dtype=error.dtype,
        )
        weighted_error = (error * weights).sum(-1) / weights.sum()
    else:
        weighted_error = error.mean(-1)
    return torch.exp(-weighted_error / std**2)


def tracking_body_linvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body linear velocity tracking reward using a Gaussian kernel.

    Encourages tracked body linear velocities to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_lin_vel_w[:, tracked] - command.robot_body_lin_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_body_angvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body angular velocity tracking reward using a Gaussian kernel.

    Encourages tracked body angular velocities to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_ang_vel_w[:, tracked] - command.robot_body_ang_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def anti_shake_ang_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 1.5,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize excessive angular velocity on selected bodies with a deadzone.

    Discourage high-frequency jitter on small links (wrists, head) while allowing
    normal intentional motion within the threshold. Speeds below the threshold
    incur zero penalty.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        threshold: Angular velocity deadzone (rad/s). No penalty below this.
        body_names: Bodies to penalize. If None, uses all tracked bodies.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # [E, B, 3]
    ang_vel = command.robot_body_ang_vel_w[:, body_indexes]
    # magnitude per body: [E, B]
    speed = torch.linalg.norm(ang_vel, dim=-1)
    # deadzone then square: [E, B]
    excess = torch.relu(speed - threshold)
    penalty = (excess * excess).mean(dim=-1)
    return penalty
