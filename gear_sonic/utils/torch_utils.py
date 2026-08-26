"""PyTorch and USD/Gf utility functions for IsaacGym-based RL.

Provides quaternion arithmetic (multiply, apply, rotate, conjugate, inverse
transform, combine), Euler-angle conversions, tensor helpers (clamp, scale,
unscale, random float/direction), and a USD ``Gf.Matrix4d`` construction
helper.  Most functions are compiled with ``@torch.jit.script`` for
performance.

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
from pxr import Gf
import torch


def set_env_attr(self, attr_name, attr_val, env_ids):
    """Set a per-environment attribute on an env object for the given env ids.

    If the attribute already exists it is indexed by ``env_ids``; otherwise
    the value is set as a plain attribute (useful for first-time initialisation
    before the buffer exists).

    Args:
        self: Environment object that owns the attribute.
        attr_name: Name of the attribute to set.
        attr_val: Value(s) to assign.
        env_ids: Integer indices of the environments to update.
    """
    if hasattr(self, attr_name):
        getattr(self, attr_name)[env_ids] = attr_val
    else:
        setattr(self, attr_name, attr_val)


def to_torch(
    x: np.ndarray | torch.Tensor | list, device: torch.device | str, dtype=None, requires_grad=False
):
    """Convert a list, NumPy array, or Tensor to a ``torch.Tensor`` on ``device``.

    Args:
        x: Input data.  Lists are first converted to ``np.ndarray``.
        device: Target device (e.g. ``"cuda:0"`` or ``torch.device("cpu")``).
        dtype: Desired dtype.  Defaults to ``torch.float`` for non-Tensor
            inputs; for existing Tensors the current dtype is preserved when
            ``dtype`` is ``None``.
        requires_grad: Whether the result should track gradients.

    Returns:
        A ``torch.Tensor`` on ``device`` with the requested dtype and
        ``requires_grad`` setting.
    """
    # Convert list to np.ndarray for shape and dtype handling
    if isinstance(x, list):
        x = np.array(x)

    if not isinstance(x, torch.Tensor):
        if dtype is None:
            dtype = torch.float
        x = torch.tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)
    else:
        # torch.Tensor
        if dtype is None:
            x = x.to(device=device)
            if x.requires_grad != requires_grad:
                x = x.detach().requires_grad_(requires_grad)
        else:
            x = x.to(dtype=dtype, device=device)
            if x.requires_grad != requires_grad:
                x = x.detach().requires_grad_(requires_grad)
    return x


@torch.jit.script
def quat_mul(a, b):
    """Multiply two batches of quaternions (xyzw convention).

    Args:
        a: Quaternion tensor of shape ``(..., 4)`` in xyzw order.
        b: Quaternion tensor of the same shape as ``a``.

    Returns:
        Product quaternion tensor of the same shape as ``a``.
    """
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    """L2-normalize a tensor along its last dimension.

    Args:
        x: Input tensor of any shape.
        eps: Minimum norm value to clamp to, preventing division by zero.

    Returns:
        Tensor of the same shape as ``x`` with unit L2 norm along the last dim.
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_apply(a, b):
    """Rotate a 3-D vector by a quaternion (xyzw convention).

    Args:
        a: Quaternion tensor of shape ``(..., 4)`` in xyzw order.
        b: Vector tensor of shape ``(..., 3)``.

    Returns:
        Rotated vector tensor of the same shape as ``b``.
    """
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    """Rotate batched 3-D vectors by batched quaternions (xyzw convention).

    Uses the expanded form ``v' = 2(q_w^2 - 0.5)v + 2(q·v)q + 2q_w(q×v)``.

    Args:
        q: Quaternion tensor of shape ``(N, 4)`` in xyzw order.
        v: Vector tensor of shape ``(N, 3)``.

    Returns:
        Rotated vector tensor of shape ``(N, 3)``.
    """
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


# @torch.jit.script
def quat_rotate_inverse(q, v):
    """Rotate batched 3-D vectors by the *inverse* of batched quaternions.

    Equivalent to rotating by the conjugate quaternion (i.e. the transpose of
    the rotation matrix).

    Args:
        q: Quaternion tensor of shape ``(N, 4)`` in xyzw order.
        v: Vector tensor of shape ``(N, 3)``.

    Returns:
        Inversely-rotated vector tensor of shape ``(N, 3)``.
    """
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_conjugate(a):
    """Return the conjugate of a batch of quaternions (xyzw convention).

    The conjugate negates the imaginary (xyz) part while keeping the real (w)
    part, yielding the inverse rotation for unit quaternions.

    Args:
        a: Quaternion tensor of shape ``(..., 4)`` in xyzw order.

    Returns:
        Conjugate quaternion tensor of the same shape.
    """
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    """Normalize a batch of quaternions to unit length.

    Args:
        a: Quaternion tensor of shape ``(..., 4)``.

    Returns:
        Unit-length quaternion tensor of the same shape.
    """
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    """Construct a unit quaternion from an angle-axis representation.

    Args:
        angle: Rotation angle in radians, shape ``(N,)``.
        axis: Rotation axes, shape ``(N, 3)``.  Need not be unit vectors.

    Returns:
        Unit quaternion tensor of shape ``(N, 4)`` in xyzw order.
    """
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def normalize_angle(x):
    """Wrap angles into the range ``(-pi, pi]``.

    Args:
        x: Angle tensor (radians), any shape.

    Returns:
        Wrapped angle tensor of the same shape.
    """
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def tf_inverse(q, t):
    """Compute the inverse of a rigid transform (q, t).

    Args:
        q: Rotation quaternion, shape ``(N, 4)`` in xyzw order.
        t: Translation vector, shape ``(N, 3)``.

    Returns:
        Tuple ``(q_inv, t_inv)`` representing the inverse transform.
    """
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    """Apply a rigid transform (q, t) to a batch of points v.

    Computes ``R(q) * v + t``.

    Args:
        q: Rotation quaternion, shape ``(N, 4)`` in xyzw order.
        t: Translation vector, shape ``(N, 3)``.
        v: Points to transform, shape ``(N, 3)``.

    Returns:
        Transformed points of shape ``(N, 3)``.
    """
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    """Rotate a vector by a quaternion (no translation).

    Args:
        q: Quaternion, shape ``(..., 4)`` in xyzw order.
        v: Vector to rotate, shape ``(..., 3)``.

    Returns:
        Rotated vector of the same shape as ``v``.
    """
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    """Compose two rigid transforms T1 followed by T2.

    Computes the combined rotation ``q1 * q2`` and the combined translation
    ``R(q1) * t2 + t1``.

    Args:
        q1: First rotation quaternion, shape ``(N, 4)`` in xyzw order.
        t1: First translation, shape ``(N, 3)``.
        q2: Second rotation quaternion, shape ``(N, 4)`` in xyzw order.
        t2: Second translation, shape ``(N, 3)``.

    Returns:
        Tuple ``(q_combined, t_combined)`` of the composed transform.
    """
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def get_basis_vector(q, v):
    """Rotate a basis vector ``v`` by quaternion ``q``.

    Args:
        q: Quaternion tensor, shape ``(N, 4)`` in xyzw order.
        v: Basis vector, shape ``(N, 3)``.

    Returns:
        Rotated vector of shape ``(N, 3)``.
    """
    return quat_rotate(q, v)


def get_axis_params(value, axis_idx, x_value=0.0, dtype=np.float64, n_dims=3):
    """Construct a parameter list for a USD ``Vec`` along a specific axis.

    Creates an n-dimensional vector that is ``value`` along ``axis_idx`` and
    zero everywhere else, then overrides index 0 with ``x_value``.

    Args:
        value: Scalar value to place at position ``axis_idx``.
        axis_idx: Index of the axis to set to ``value``.
        x_value: Value to assign to index 0 after the axis fill.
        dtype: NumPy dtype of the output array.
        n_dims: Total number of dimensions in the vector.

    Returns:
        List of ``n_dims`` floats suitable for passing to a USD ``Vec``
        constructor.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    """Copy the sign of tensor ``b`` onto scalar ``a``.

    Returns a tensor of the same shape as ``b`` with magnitude ``|a|`` and
    sign matching each element of ``b``.

    Args:
        a: Scalar magnitude.
        b: Tensor whose signs are copied, shape ``(N,)``.

    Returns:
        Tensor of shape ``(N,)`` equal to ``|a| * sign(b)``.
    """
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    """Extract Euler XYZ angles (roll, pitch, yaw) from a batch of quaternions.

    Uses the standard ZYX intrinsic decomposition.  Handles the gimbal-lock
    singularity at |sinp| >= 1 via ``copysign``.

    Args:
        q: Quaternion tensor of shape ``(N, 4)`` in xyzw order.

    Returns:
        Tuple ``(roll, pitch, yaw)`` each of shape ``(N,)`` in radians,
        mapped to ``[0, 2*pi)``.
    """
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

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    """Construct unit quaternions from intrinsic XYZ Euler angles.

    Args:
        roll: Rotation around X axis (radians), shape ``(N,)``.
        pitch: Rotation around Y axis (radians), shape ``(N,)``.
        yaw: Rotation around Z axis (radians), shape ``(N,)``.

    Returns:
        Unit quaternion tensor of shape ``(N, 4)`` in xyzw order.
    """
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
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    """Sample uniform random floats in ``[lower, upper)``.

    Args:
        lower: Lower bound of the uniform distribution.
        upper: Upper bound of the uniform distribution.
        shape: Output shape as a 2-tuple ``(rows, cols)``.
        device: Target device string (e.g. ``"cuda:0"``).

    Returns:
        Float tensor of the given shape on ``device``.
    """
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def torch_random_dir_2(shape, device):
    # type: (Tuple[int, int], str) -> Tensor
    """Sample uniformly random unit vectors in 2-D.

    Args:
        shape: Shape of the angle samples as a 2-tuple ``(rows, 1)``.
        device: Target device string.

    Returns:
        Float tensor of shape ``(rows, 2)`` containing ``(cos θ, sin θ)``
        with ``θ`` drawn uniformly from ``[-π, π)``.
    """
    angle = torch_rand_float(-np.pi, np.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    """Element-wise clamp of tensor ``t`` to the range ``[min_t, max_t]``.

    Unlike ``torch.clamp`` this version accepts tensors for the bounds so that
    per-element limits are supported.

    Args:
        t: Input tensor.
        min_t: Lower bound tensor, same shape or broadcastable to ``t``.
        max_t: Upper bound tensor, same shape or broadcastable to ``t``.

    Returns:
        Clamped tensor of the same shape as ``t``.
    """
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def scale(x, lower, upper):
    """Map values from ``[-1, 1]`` to ``[lower, upper]``.

    Args:
        x: Input tensor in the normalised range ``[-1, 1]``.
        lower: Target range lower bound.
        upper: Target range upper bound.

    Returns:
        Tensor scaled to ``[lower, upper]``.
    """
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    """Map values from ``[lower, upper]`` to ``[-1, 1]``.

    Inverse of :func:`scale`.

    Args:
        x: Input tensor in the range ``[lower, upper]``.
        lower: Source range lower bound.
        upper: Source range upper bound.

    Returns:
        Tensor normalised to ``[-1, 1]``.
    """
    return (2.0 * x - upper - lower) / (upper - lower)


def unscale_np(x, lower, upper):
    """NumPy equivalent of :func:`unscale`.

    Args:
        x: Input array in the range ``[lower, upper]``.
        lower: Source range lower bound.
        upper: Source range upper bound.

    Returns:
        Array normalised to ``[-1, 1]``.
    """
    return (2.0 * x - upper - lower) / (upper - lower)


def euler_xyz_to_gf_matrix(angles):
    """Convert Euler XYZ angles (in radians) to a USD Gf.Matrix4d.

    Args:
        angles: [roll, pitch, yaw] in radians. Can be list, numpy array, or torch.Tensor.

    Returns:
        Gf.Matrix4d with the rotation applied.
    """
    if isinstance(angles, list | np.ndarray):
        angles = torch.tensor(angles, dtype=torch.float32)
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)  # [1, 3]

    roll, pitch, yaw = angles[:, 0], angles[:, 1], angles[:, 2]
    quat = quat_from_euler_xyz(roll, pitch, yaw)  # [1, 4] as [qx, qy, qz, qw]

    # Convert to Gf.Quatd (w, x, y, z order)
    qx, qy, qz, qw = quat[0].tolist()
    gf_quat = Gf.Quatd(qw, qx, qy, qz)

    m = Gf.Matrix4d()
    m.SetRotate(Gf.Rotation(gf_quat))
    return m
