"""Interpolation and frame-rate rescaling for pose sequences.

Provides linear interpolation (via scipy), quaternion slerp, and
functions to up/down-sample joint pose trajectories to a target frame rate.
"""

import torch
import numpy as np
from scipy.interpolate import interp1d
from .kornia_transform import angle_axis_to_quaternion, quaternion_to_angle_axis


def interp_tensor_with_scipy(x, new_len=None, scale=None, dim=-1):
    orig_len = x.shape[dim]
    if new_len is None:
        new_len = int(orig_len * scale)
    T = orig_len
    f = interp1d(
        np.linspace(0, T, orig_len),
        x.cpu().numpy(),
        axis=dim,
        assume_sorted=True,
        fill_value="extrapolate",
    )
    x_interp = torch.from_numpy(f(np.linspace(0, T, new_len))).type_as(x)
    return x_interp


def slerp(q0, q1, t):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor

    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()

    # Replace: q1[neg_mask] = -q1[neg_mask]
    # With: torch.where for safer broadcasting
    neg_mask_expanded = neg_mask.unsqueeze(-1).expand_as(q1)
    q1 = torch.where(neg_mask_expanded, -q1, q1)

    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t[:, None]) * half_theta) / sin_half_theta
    ratioB = torch.sin(t[:, None] * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


def _slerp_batch(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    slerped_quats = slerp(a, b, blend)
    return slerped_quats


def interpolate_quaternions(
    pose_quat: torch.Tensor, source_fps: float, target_fps: float
) -> torch.Tensor:
    """
    Interpolate quaternions from source_fps to target_fps.

    Args:
        pose_quat: Input quaternions, shape (1, T, 4)
        source_fps: Source frame rate
        target_fps: Target frame rate

    Returns:
        Interpolated quaternions
    """
    device = pose_quat.device
    in_shape = pose_quat.shape
    assert in_shape[0] == 1, "Only support single sequence for now"

    T = in_shape[1]
    duration = (T - 1) * (1 / source_fps)
    times = torch.arange(0, duration + 1e-6, 1 / target_fps, dtype=torch.float32, device=device)
    times = times[times <= duration]

    # Compute frame indices and blend factors
    frame_indices = times * source_fps
    index_0 = torch.floor(frame_indices).long()
    index_1 = torch.min(index_0 + 1, torch.tensor(T - 1, device=device))
    blend = frame_indices - index_0.float()

    pose_quat_interp = _slerp_batch(pose_quat[0, index_0], pose_quat[0, index_1], blend)
    pose_quat_interp = pose_quat_interp.unsqueeze(0)

    return pose_quat_interp


def interpolate_pose(
    pose_aa: torch.Tensor,
    source_fps: float,
    target_fps: float,
    device: str = "cpu",
    interpolation_type: str = "slerp",
    rot_type: str = "aa",
) -> torch.Tensor:
    """
    Interpolate pose_aa from source_fps to target_fps using specified interpolation method.

    Args:
        pose_aa: Input pose in angle-axis format, shape (T, N*3) where T is number of frames and N is number of joints
        source_fps: Source frame rate
        target_fps: Target frame rate
        device: Device to run computations on
        interpolation_type: Type of interpolation to use ("linear" or "slerp")

    Returns:
        Interpolated pose_aa with new frame rate, shape (T_new, N*3)
    """
    # pose_aa: (T, N*3)
    orig_shape = pose_aa.shape[1:]
    if pose_aa.ndim != 2:
        pose_aa = pose_aa.reshape(pose_aa.shape[0], -1)
    T, D = pose_aa.shape

    if interpolation_type == "linear":
        # Direct linear interpolation on angle-axis representation
        duration = (T - 1) * (1 / source_fps)
        times = torch.arange(0, duration + 1e-6, 1 / target_fps, dtype=torch.float32, device=device)
        times = times[times <= duration]

        # Compute frame indices and blend factors for linear interpolation
        frame_indices = times * source_fps
        index_0 = torch.floor(frame_indices).long()
        index_1 = torch.min(index_0 + 1, torch.tensor(T - 1, device=device))
        blend = frame_indices - index_0.float()

        # Linear interpolation on the entire 2D tensor
        pose_aa_interp = (1 - blend.unsqueeze(1)) * pose_aa[index_0] + blend.unsqueeze(1) * pose_aa[
            index_1
        ]
        pose_aa_interp = pose_aa_interp.view(pose_aa_interp.shape[0], *orig_shape)
        if pose_aa.dtype == torch.int64:
            pose_aa_interp = pose_aa_interp.round()
        pose_aa_interp = pose_aa_interp.type_as(pose_aa)
        return pose_aa_interp

    elif interpolation_type == "slerp":
        dim = 3 if rot_type == "aa" else 4
        N = D // dim
        pose_aa_reshaped = pose_aa.view(T, N, dim)
        # Original spherical linear interpolation on quaternions
        pose_aa_interp_list = []
        for i in range(N):
            # Convert angle-axis to quaternion for this joint
            if rot_type == "aa":
                pose_quat = angle_axis_to_quaternion(pose_aa_reshaped[:, i])  # (T, 4)
            else:
                pose_quat = pose_aa_reshaped[:, i]
            pose_quat_batch = pose_quat.unsqueeze(0)  # (1, T, 4)
            pose_quat_interp = interpolate_quaternions(pose_quat_batch, source_fps, target_fps)
            pose_quat_interp = pose_quat_interp[0]  # (T_new, 4)
            if rot_type == "aa":
                pose_aa_interp = quaternion_to_angle_axis(pose_quat_interp)  # (T_new, 3)
            else:
                pose_aa_interp = pose_quat_interp
            pose_aa_interp_list.append(pose_aa_interp)

        # Concatenate all joints: (T_new, N, 3) -> (T_new, N*3)
        pose_aa_interp = torch.stack(pose_aa_interp_list, dim=1)  # (T_new, N, 3)
        pose_aa_interp = pose_aa_interp.view(pose_aa_interp.shape[0], -1)  # (T_new, N*3)
        pose_aa_interp = pose_aa_interp.view(pose_aa_interp.shape[0], *orig_shape).to(pose_aa)
        return pose_aa_interp

    else:
        raise ValueError(
            f"Unsupported interpolation_type: {interpolation_type}. Must be 'linear' or 'slerp'."
        )
