"""Rotation and gravity transform utilities for data collection."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_to_rot6d(q):
    """Convert scalar-first quaternion(s) (wxyz) to 6D rotation representation.

    The 6D representation consists of the first two columns of the rotation
    matrix, flattened (Zhou et al., CVPR 2019).

    Accepted input shapes:
        * ``(4,)``   -- single quaternion  -> returns ``(6,)``
        * ``(N, 4)`` -- batch of quats     -> returns ``(N, 6)``
        * ``(N*4,)`` -- flat concatenated  -> returns ``(N*6,)``
    """
    q = np.asarray(q)
    if q.ndim == 1 and q.shape[0] > 4:
        assert q.shape[0] % 4 == 0, f"Flat quat length {q.shape[0]} is not divisible by 4"
        q = q.reshape(-1, 4)
        rot_6d = quat_to_rot6d(q)
        return rot_6d.ravel()

    single = q.ndim == 1
    q = np.atleast_2d(q)
    q_xyzw = q[:, [1, 2, 3, 0]]
    rot_mat = R.from_quat(q_xyzw).as_matrix()  # (N, 3, 3)
    rot_6d = rot_mat[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)  # (N, 6)
    if single:
        return rot_6d[0].astype(q.dtype)
    return rot_6d.astype(q.dtype)


def rot6d_to_quat(r):
    """Convert 6D rotation representation to scalar-first quaternion(s) (wxyz).

    Accepted input shapes:
        * ``(6,)``   -- single rot6d  -> returns ``(4,)``
        * ``(N, 6)`` -- batch         -> returns ``(N, 4)``
        * ``(N*6,)`` -- flat concat   -> returns ``(N*4,)``
          (length must be divisible by 6)

    Args:
        r: 6D rotation array (first two columns of rotation matrix, row-major).

    Returns:
        Quaternion array in wxyz order.
    """
    r = np.asarray(r, dtype=np.float64)
    if r.ndim == 1 and r.shape[0] > 6:
        assert r.shape[0] % 6 == 0, f"Flat rot6d length {r.shape[0]} is not divisible by 6"
        r = r.reshape(-1, 6)
        quats = rot6d_to_quat(r)
        return quats.ravel()

    single = r.ndim == 1
    r = np.atleast_2d(r)  # (N, 6)
    col0 = r[:, :3]
    col1 = r[:, 3:]
    col0 = col0 / (np.linalg.norm(col0, axis=1, keepdims=True) + 1e-8)
    dot = np.sum(col0 * col1, axis=1, keepdims=True)
    col1 = col1 - dot * col0
    col1 = col1 / (np.linalg.norm(col1, axis=1, keepdims=True) + 1e-8)
    col2 = np.cross(col0, col1)
    rot_mat = np.stack([col0, col1, col2], axis=-1)  # (N, 3, 3)
    q_xyzw = R.from_matrix(rot_mat).as_quat()  # (N, 4) xyzw
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    if single:
        return q_wxyz[0].astype(np.float32)
    return q_wxyz.astype(np.float32)


def compute_projected_gravity(base_quat: np.ndarray) -> np.ndarray:
    """Compute projected gravity vector in robot's body frame from base quaternion.

    Projects the world gravity vector [0, 0, -1] into the robot's body frame by
    rotating it by the inverse of the base quaternion.

    Args:
        base_quat: Base quaternion [qw, qx, qy, qz] of shape (4,)

    Returns:
        Projected gravity vector [gx, gy, gz] of shape (3,) in robot's body frame
    """
    base_quat = np.asarray(base_quat, dtype=np.float64)
    if base_quat.shape != (4,):
        raise ValueError(f"base_quat must have shape (4,), got {base_quat.shape}")

    gravity_vec_world = np.array([0.0, 0.0, -1.0])
    base_rotation = R.from_quat(base_quat, scalar_first=True)
    projected_gravity = base_rotation.inv().apply(gravity_vec_world)

    return projected_gravity.astype(np.float32)
