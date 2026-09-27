"""Strict deployment-equivalent WXYZ Hamilton quaternion operations."""

from __future__ import annotations

import math
import numbers

import numpy as np

_DEGENERATE_NORM = 1e-12
_UNIT_NORM_TOLERANCE = 1e-5
_SLERP_LINEAR_DOT = 0.9995
_SLERP_RESULT_TOLERANCE = 1e-10


def _float64_wxyz(value: object) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("quaternion must be a numeric float64 quaternion with shape [4]") from error
    if source.shape != (4,):
        raise ValueError(f"quaternion must have shape [4]; got {source.shape}")
    if source.dtype.kind not in "iuf":
        raise ValueError("quaternion must be a numeric float64 quaternion")
    result = np.array(source, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("quaternion components must be finite")
    return result


def validate_wxyz(value: object) -> tuple[np.ndarray, bool]:
    """Validate and normalize one finite WXYZ quaternion under the source policy."""
    quaternion = _float64_wxyz(value)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm):
        raise ValueError("quaternion norm must be finite")
    if norm < _DEGENERATE_NORM:
        raise ValueError("quaternion norm is degenerate")
    if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
        raise ValueError("quaternion norm exceeds the unit-norm tolerance")

    normalized = np.array(quaternion / norm, dtype=np.float64, order="C", copy=True)
    changed = not np.array_equal(normalized.view(np.uint64), quaternion.view(np.uint64))
    return normalized, changed


def quat_conjugate(value: object) -> np.ndarray:
    """Return the conjugate of a policy-normalized WXYZ quaternion."""
    quaternion, _ = validate_wxyz(value)
    return np.array(
        [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
        dtype=np.float64,
        order="C",
    )


def quat_mul(left: object, right: object) -> np.ndarray:
    """Return normalized Hamilton product ``left ⊗ right`` in WXYZ order."""
    a, _ = validate_wxyz(left)
    b, _ = validate_wxyz(right)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    product = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
        order="C",
    )
    normalized, _ = validate_wxyz(product)
    return normalized


def heading_quat(value: object) -> np.ndarray:
    """Extract deployment yaw by projecting the rotated positive-X direction."""
    quaternion, _ = validate_wxyz(value)
    w, x, y, z = quaternion
    scale_a = 2.0 * w * w - 1.0
    a0 = scale_a
    b0 = 0.0
    c0 = x * x * 2.0
    rotated_x = (a0 + b0) + c0
    a1 = 0.0
    b1 = z * w * 2.0
    c1 = y * x * 2.0
    rotated_y = (a1 + b1) + c1
    heading = math.atan2(rotated_y, rotated_x)
    half = heading / 2.0
    result = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
    normalized, _ = validate_wxyz(result)
    return normalized


def _alpha(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("SLERP alpha must be a finite real scalar in [0,1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("SLERP alpha must be a finite real scalar in [0,1]")
    return result


def quat_slerp_deployment(q0: object, q1: object, alpha: object) -> np.ndarray:
    """Reproduce deployment ``quat_slerp_d`` on validated unit quaternions."""
    start, _ = validate_wxyz(q0)
    end, _ = validate_wxyz(q1)
    return _quat_slerp_deployment_normalized(start, end, alpha)


def _quat_slerp_deployment_normalized(
    start: np.ndarray,
    end: np.ndarray,
    alpha: object,
) -> np.ndarray:
    """Apply deployment SLERP to operands already normalized by the source policy."""
    weight = _alpha(alpha)

    dot = float(np.dot(start, end))
    adjusted_end = np.array(end, dtype=np.float64, order="C", copy=True)
    if dot < 0.0:
        adjusted_end *= -1.0
        dot = -dot

    if dot > _SLERP_LINEAR_DOT:
        linear = start + weight * (adjusted_end - start)
        linear_norm = float(np.linalg.norm(linear))
        if not math.isfinite(linear_norm) or linear_norm < _DEGENERATE_NORM:
            raise ValueError("SLERP linear result is not normalizable")
        result = linear / linear_norm
    else:
        clamped_dot = min(1.0, max(-1.0, dot))
        theta = math.acos(abs(clamped_dot))
        sin_theta = math.sin(theta)
        factor0 = math.sin((1.0 - weight) * theta) / sin_theta
        factor1 = math.sin(weight * theta) / sin_theta
        result = factor0 * start + factor1 * adjusted_end

    owned = np.array(result, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(owned).all():
        raise ValueError("SLERP result must be finite")
    unit_error = abs(float(np.linalg.norm(owned)) - 1.0)
    if unit_error > _SLERP_RESULT_TOLERANCE:
        raise ValueError("SLERP result violates unit-norm tolerance")
    return owned


def quat_to_encoder_rot6d(value: object) -> np.ndarray:
    """Pack the first two rotation-matrix columns in deployment row-major order."""
    quaternion, _ = validate_wxyz(value)
    w, x, y, z = quaternion
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
        ],
        dtype=np.float64,
        order="C",
    )
