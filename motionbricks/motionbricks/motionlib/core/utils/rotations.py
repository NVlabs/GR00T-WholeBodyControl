import einops
import numpy as np
import torch
import torch.nn.functional as F

from motionbricks.motionlib.core.utils.torch_utils import normalize_vec


def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:].clone()
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 0:1].clone() * t + xyz.cross(t, dim=-1)).view(shape)


def cont6d_to_matrix(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat


def matrix_to_cont6d(matrix):
    cont_6d = torch.concat([matrix[..., 0], matrix[..., 1]], dim=-1)
    return cont_6d


def quaternion_to_matrix(quaternions):
    """Convert rotations given as quaternions to rotation matrices.

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


def quaternion_to_cont6d(quaternions: torch.Tensor) -> torch.Tensor:
    rotation_mat = quaternion_to_matrix(quaternions)
    cont_6d = matrix_to_cont6d(rotation_mat)
    return cont_6d


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) subgradient is zero where x is 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

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

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def cont6d_to_quaternion(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    matrix = cont6d_to_matrix(cont6d)
    return matrix_to_quaternion(matrix)


def exp_map_to_matrix(exp_map):
    quat = exp_map_to_quat(exp_map)
    return quaternion_to_matrix(quat)


#
# Extra utils only needed for visualization
#


def quat_unit(a: torch.Tensor):
    return normalize_vec(a)


def angle_axis_to_quaternion(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize_vec(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([w, xyz], dim=-1))


def quat_between_two_vec(v1, v2, eps: float = 1e-6):
    """Quaternion for rotating v1 to v2."""
    orig_shape = v1.shape
    v1 = v1.reshape(-1, 3)
    v2 = v2.reshape(-1, 3)
    dot = (v1 * v2).sum(-1)
    cross = torch.cross(v1, v2, dim=-1)
    out = torch.cat([(1 + dot).unsqueeze(-1), cross], dim=-1)
    # handle v1 & v2 with same direction
    sind = dot > 1 - eps
    out[sind] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=v1.device)
    # handle v1 & v2 with opposite direction
    nind = dot < -1 + eps
    if torch.any(nind):
        vx = torch.tensor([1.0, 0.0, 0.0], device=v1.device)
        vxdot = (v1 * vx).sum(-1).abs()
        nxind = nind & (vxdot < 1 - eps)
        if torch.any(nxind):
            out[nxind] = exp_map_to_quat(
                normalize_vec(torch.cross(vx.expand_as(v1[nxind]), v1[nxind], dim=-1))
                * np.pi
            )
        # handle v1 & v2 with opposite direction and they are parallel to x axis
        pind = nind & (vxdot >= 1 - eps)
        if torch.any(pind):
            vy = torch.tensor([0.0, 1.0, 0.0], device=v1.device)
            out[pind] = exp_map_to_quat(
                normalize_vec(torch.cross(vy.expand_as(v1[pind]), v1[pind], dim=-1))
                * np.pi
            )
    # normalize and reshape
    out = normalize_vec(out).view(orig_shape[:-1] + (4,))
    return out


def quaternion_angle_diff(q1, q2, eps=1e-6):
    """Calculate the angle difference between two quaternions in radians. Handles arbitrary input
    shapes where the last dimension is 4.

    Args:
        q1: First quaternion tensor [..., 4]
        q2: Second quaternion tensor [..., 4]

    Returns:
        Angle in radians between the two quaternions, shape [...]
    """
    # Normalize quaternions
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)

    # Compute dot product
    dot = torch.clamp(torch.abs(torch.sum(q1 * q2, dim=-1)), -1.0, 1.0)
    # Calculate angle in radians
    angle = 2 * torch.acos(dot)
    angle[angle < eps] = 0.0

    return angle


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def exp_map_to_angle_axis(exp_map):
    min_theta = 1e-5

    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / angle_exp
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = torch.abs(angle) > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis


def exp_map_to_quat(exp_map):
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = angle_axis_to_quaternion(angle, axis)
    return q


def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qw, qx = 0, 1

    norms = torch.norm(q[..., qx:], p=2, dim=-1)
    half_angles = torch.atan2(
        norms, q[..., qw]
    )  # half_angles: [0,pi] because norms >= 0
    angle = 2 * half_angles  # angle: [0, 2pi]
    sin_theta = torch.sin(half_angles)  # sin_theta: [0, 1]
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:] / sin_theta_expand

    mask = sin_theta > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    # angle is within [0, 2pi].
    # if angle > pi, use shorter side of the arc(2*pi-angle) and flip the rotation axis
    flip_axis = angle > torch.pi
    angle = torch.where(flip_axis, 2 * torch.pi - angle, angle)
    axis = torch.where(flip_axis[..., None], -axis, axis)

    return angle, axis


def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map


def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map


def angle_to_Y_rotation_matrix(angle):
    cos, sin = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(angle), torch.zeros_like(angle)
    mat = torch.stack((cos, zero, sin, zero, one, zero, -sin, zero, cos), -1)
    mat = mat.reshape(angle.shape + (3, 3))
    return mat


def diff_angles(angles, fps: float):
    """Computes differences between angles.

    Args:
        angles (Tensor): [..., T] the batched sequences of rotation angles in radians.

    Returns:
        Tensor: [..., T-1] the difference between consecutive angles
    """

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    cos_diff = cos[..., 1:] * cos[..., :-1] + sin[..., 1:] * sin[..., :-1]
    sin_diff = sin[..., 1:] * cos[..., :-1] - cos[..., 1:] * sin[..., :-1]

    # should be close to angles.diff() but more robust
    # multiply by fps = 1 / dt
    angles_diff = fps * torch.arctan2(sin_diff, cos_diff)
    return angles_diff


def diff_between_two_angles(b, a, fps: float):
    # angle: b - a
    cos_a = np.cos(a)
    sin_a = np.sin(a)

    cos_b = np.cos(b)
    sin_b = np.sin(b)

    cos_diff = cos_b * cos_a + sin_b * sin_a
    sin_diff = sin_b * cos_a - cos_b * sin_a

    return fps * np.arctan2(sin_diff, cos_diff)


# Numpy-backed utils


def diff_angles_np(angles: np.array, fps: float) -> np.array:
    angles = torch.from_numpy(angles)
    return diff_angles(angles, fps).numpy()


def qmul_np(q: np.array, r: np.array) -> np.array:
    q = torch.from_numpy(q).contiguous().float()
    r = torch.from_numpy(r).contiguous().float()
    return quat_mul(q, r).numpy()


def qrot_np(q: np.array, v: np.array) -> np.array:
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return quat_apply(q, v).numpy()


def qinv_np(q: np.array) -> np.array:
    q = torch.from_numpy(q).contiguous().float()
    return quat_conjugate(q).numpy()


def quaternion_to_cont6d_np(q: np.array) -> np.array:
    q = torch.from_numpy(q).contiguous().float()
    return quaternion_to_cont6d(q).numpy()
