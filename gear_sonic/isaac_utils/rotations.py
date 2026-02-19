"""JIT-compiled quaternion and rotation utilities for Isaac environments.

Provides quaternion arithmetic (multiply, inverse, conjugate, slerp), conversions
(axis-angle, rotation matrix, euler), and specialized helpers for SMPL root
orientation transforms (Y-up to Z-up, base rotation removal).
"""

import torch
from torch import Tensor
import torch.nn.functional as F
from gear_sonic.isaac_utils.maths import (
    normalize,
    copysign,
)
from gear_sonic.trl.utils.torch_transform import angle_axis_to_quaternion, quaternion_to_angle_axis
from typing import Tuple
import numpy as np
from typing import List, Optional


@torch.jit.script
def quat_unit(a):
    """Normalize quaternion to unit length."""
    return normalize(a)


@torch.jit.script
def quat_apply(a: Tensor, b: Tensor, w_last: bool) -> Tensor:
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    if w_last:
        xyz = a[:, :3]
        w = a[:, 3:]
    else:
        xyz = a[:, 1:]
        w = a[:, :1]
    t = xyz.cross(b, dim=-1) * 2
    return (b + w * t + xyz.cross(t, dim=-1)).view(shape)


def get_yaw_quat_from_quat(quat_angle):
    rpy = get_euler_xyz_in_tensor(quat_angle)
    roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    roll = torch.zeros_like(roll)
    pitch = torch.zeros_like(pitch)
    return quat_from_euler_xyz(roll, pitch, yaw)


@torch.jit.script
def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = quat.view(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = torch.zeros_like(quat_yaw)
    quat_yaw[:, 3] = torch.sin(yaw / 2)
    quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def quat_conjugate(a: Tensor, w_last: bool) -> Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    if w_last:
        return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)
    else:
        return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


 

@torch.jit.script
def quat_rotate(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    # Same as quat_rotate but with the cross-product term (b) negated,
    # which is equivalent to rotating by the conjugate quaternion (inverse rotation).
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_angle_axis(x: Tensor, w_last: bool) -> Tuple[Tensor, Tensor]:
    """
    The (angle, axis) representation of the rotation. The axis is normalized to unit length.
    The angle is guaranteed to be between [0, pi].
    """
    if w_last:
        w = x[..., -1]
        axis = x[..., :3]
    else:
        w = x[..., 0]
        axis = x[..., 1:]
    # cos(theta) = 2*w^2 - 1, derived from w = cos(theta/2) and double-angle formula
    s = 2 * (w**2) - 1
    angle = s.clamp(-1, 1).arccos()  # just to be safe
    axis /= axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    return angle, axis


@torch.jit.script
def quat_from_angle_axis(angle: Tensor, axis: Tensor, w_last: bool) -> Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    if w_last:
        return quat_unit(torch.cat([xyz, w], dim=-1))
    else:
        return quat_unit(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def vec_to_heading(h_vec):
    h_theta = torch.atan2(h_vec[..., 1], h_vec[..., 0])
    return h_theta


@torch.jit.script
def heading_to_quat(h_theta, w_last: bool):
    axis = torch.zeros(
        h_theta.shape
        + [
            3,
        ],
        device=h_theta.device,
    )
    axis[..., 2] = 1
    heading_q = quat_from_angle_axis(h_theta, axis, w_last=w_last)
    return heading_q


@torch.jit.script
def quat_axis(q: Tensor, axis: int, w_last: bool) -> Tensor:
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec, w_last)


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def get_basis_vector(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    return quat_rotate(q, v, w_last)


@torch.jit.script
def quat_to_angle_axis(q, w_last: bool):
    # type: (Tensor, bool) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    # ZL: could have issues.
    min_theta = 1e-5
    if w_last:
        qx, qy, qz, qw = 0, 1, 2, 3
    else:
        qw, qx, qy, qz = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
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

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map


@torch.jit.script
def my_quat_rotate(q, v, w_last=True):
    # type: (Tensor, Tensor, bool) -> Tensor
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_to_tan_norm(q, w_last):
    # type: (Tensor, bool) -> Tensor
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    if w_last:
        tan = my_quat_rotate(q, ref_tan)
    else:
        raise NotImplementedError

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    if w_last:
        norm = my_quat_rotate(q, ref_norm)
    else:
        raise NotImplementedError

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


@torch.jit.script
def calc_heading(q, w_last=True):
    # type: (Tensor, bool) -> Tensor
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    # this is the x axis heading
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = my_quat_rotate(q, ref_dir, w_last)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


@torch.jit.script
def quat_to_exp_map(q, w_last):
    # type: (Tensor, bool) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q, w_last)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map


@torch.jit.script
def calc_heading_quat(q, w_last):
    # type: (Tensor, bool) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q, w_last)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis, w_last=w_last)
    return heading_q


@torch.jit.script
def calc_heading_quat_inv(q, w_last):
    # type: (Tensor, bool) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q, w_last)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis, w_last=w_last)
    return heading_q


@torch.jit.script
def quat_inverse(x, w_last):
    # type: (Tensor, bool) -> Tensor
    """
    The inverse of the rotation
    """
    return quat_conjugate(x, w_last=w_last)


@torch.jit.script
def get_euler_xyz(q: Tensor, w_last: bool) -> Tuple[Tensor, Tensor, Tensor]:
    if w_last:
        qx, qy, qz, qw = 0, 1, 2, 3
    else:
        qw, qx, qy, qz = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (
        q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (
        q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


# @torch.jit.script
def get_euler_xyz_in_tensor(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (
        q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (
        q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


@torch.jit.script
def quat_pos(x):
    """
    make all the real part of the quaternion positive
    """
    q = x
    z = (q[..., 3:] < 0).float()
    q = (1 - 2 * z) * q
    return q


@torch.jit.script
def is_valid_quat(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return (w * w + x * x + y * y + z * z).allclose(torch.ones_like(w))


@torch.jit.script
def quat_normalize(q):
    """
    Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
    """
    q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
    return q


@torch.jit.script
def quat_mul(a, b, w_last: bool):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat



@torch.jit.script
def quat_mul_norm(x, y, w_last):
    # type: (Tensor, Tensor, bool) -> Tensor
    """
    Combine two set of 3D rotations together using \**\* operator. The shape needs to be
    broadcastable
    """
    return quat_unit(quat_mul(x, y, w_last))


@torch.jit.script
def quat_identity(shape: List[int]):
    """
    Construct 3D identity rotation given shape
    """
    w = torch.ones(shape + [1])
    xyz = torch.zeros(shape + [3])
    q = torch.cat([xyz, w], dim=-1)
    return quat_normalize(q)


@torch.jit.script
def quat_identity_like(x):
    """
    Construct identity 3D rotation with the same shape
    """
    return quat_identity(list(x.shape[:-1]))


@torch.jit.script
def transform_from_rotation_translation(
    r: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None
):
    """
    Construct a transform from a quaternion and 3D translation. Only one of them can be None.
    """
    assert r is not None or t is not None, "rotation and translation can't be all None"
    if r is None:
        assert t is not None
        r = quat_identity(list(t.shape))
    if t is None:
        t = torch.zeros(list(r.shape) + [3])
    return torch.cat([r, t], dim=-1)


@torch.jit.script
def transform_rotation(x):
    """Get rotation from transform"""
    return x[..., :4]


@torch.jit.script
def transform_translation(x):
    """Get translation from transform"""
    return x[..., 4:]


@torch.jit.script
def transform_mul(x, y):
    """
    Combine two transformation together
    """
    z = transform_from_rotation_translation(
        r=quat_mul_norm(transform_rotation(x), transform_rotation(y), w_last=True),
        t=quat_rotate(transform_rotation(x), transform_translation(y), w_last=True)
        + transform_translation(x),
    )
    return z


##################################### FROM PHC rotation_conversions.py #####################################
@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


@torch.jit.script
def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


# @torch.jit.script
def wxyz_to_xyzw(quat):
    return quat[..., [1, 2, 3, 0]]


# @torch.jit.script
def xyzw_to_wxyz(quat):
    return quat[..., [3, 0, 1, 2]]


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    w x y z
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quat_w_first(rot):
    rot = torch.cat([rot[..., [-1]], rot[..., :-1]], -1)
    return rot


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)



@torch.jit.script
def remove_smpl_base_rot(quat, w_last: bool):
    # [0.5,0.5,0.5,0.5] is a 120° rotation about the [1,1,1] axis — SMPL's default rest orientation.
    # Conjugating it out aligns with a neutral standing pose.
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat), w_last=w_last)  # SMPL
    return quat_mul(quat, base_rot.repeat(quat.shape[0], 1), w_last=w_last)


@torch.jit.script
def smpl_root_ytoz_up(root_quat_y_up) -> torch.Tensor:
    """Convert SMPL root quaternion from Y-up to Z-up coordinate system"""
    # 90° rotation about X-axis maps Y-up (SMPL convention) to Z-up (robot convention)
    base_rot = angle_axis_to_quaternion(torch.tensor([[np.pi / 2, 0.0, 0.0]]).to(root_quat_y_up))
    root_quat_z_up = quat_mul(
        base_rot.repeat(root_quat_y_up.shape[0], 1), root_quat_y_up, w_last=False
    )
    return root_quat_z_up


@torch.jit.script
def rotate_vectors_by_quaternion(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Rotate `vec` by `quat`, elementwise.

    Args:
        quat (torch.Tensor): Tensor of shape (..., 4), quaternions in [x, y, z, w] format.
        vec  (torch.Tensor): Tensor of shape (..., 3), vectors to rotate.

    Returns:
        torch.Tensor: Rotated vectors, same shape as `vec`.
    """
    q_xyz = quat[..., :3]  # (..., 3)
    q_w = quat[..., 3:].unsqueeze(-1)  # (..., 1, 1) -> we'll squeeze to (...,1)

    # Compute intermediate cross products
    # t = 2 * q_xyz × v
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)  # (..., 3)

    # v' = v + w * t + q_xyz × t
    rotated = vec + q_w.squeeze(-1) * t + torch.cross(q_xyz, t, dim=-1)
    return rotated


def rot6d_to_quat_first_two_cols(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation (first 2 columns of rotation matrix) to quaternion.

    This function handles the 6D representation where the first 6 elements represent
    the first 2 columns of a 3x3 rotation matrix (flattened). The third column is
    reconstructed via cross product of the first two columns.

    Args:
        rot_6d (torch.Tensor): Tensor of shape (..., 6) representing the first 2 columns
                               of rotation matrix flattened.

    Returns:
        torch.Tensor: Quaternion in (w, x, y, z) format, shape (..., 4).
    """
    # Reshape to get first 2 columns: (..., 3, 2)
    rot_2cols = rot_6d.reshape(*rot_6d.shape[:-1], 3, 2)

    # Extract the two column vectors
    col_0 = rot_2cols[..., :, 0]
    col_1 = rot_2cols[..., :, 1]

    # Normalize the columns to ensure they are unit vectors
    col_0 = F.normalize(col_0, dim=-1)
    col_1 = F.normalize(col_1, dim=-1)

    # Reconstruct the third column via cross product
    col_2 = torch.cross(col_0, col_1, dim=-1)

    # Stack to form full rotation matrix (..., 3, 3)
    rot_matrix = torch.stack([col_0, col_1, col_2], dim=-1)

    # Convert rotation matrix to quaternion (w, x, y, z format)
    quat = matrix_to_quaternion(rot_matrix)

    return quat
