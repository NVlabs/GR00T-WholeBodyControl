# Pico SMPL stream server for body tracking visualization

"""

# Recommended Command Line Arguments:
    # With VR3 PT visualization (by --vis_vr3pt) and optional SMPL body visualization (by --vis_smpl)
    # If you want to enable waist tracking in the VR3 PT visualization, please add --waist_tracking
    python pico_manager_thread_server.py --manager \
        --vis_vr3pt --vis_smpl \
        --waist_tracking

    # VR3 PT visualization only (without SMPL body) — lower latency
    python pico_manager_thread_server.py --manager --vis_vr3pt

# DEBUG VR3 PT VISUALIZATION:
    # A standalone test mode that captures one live frame and visualizes it.
    python pico_manager_thread_server.py --vr3pt_live

# TIMING COMPARISON:
    # The visualizer automatically reports timing every 5 seconds when running:
    #   [Vis Timing] vr3pt: X.XXms | smpl: X.XXms | render: X.XXms | vr3pt_only: X.XXms | both(vr3pt+smpl): X.XXms

"""

from collections import defaultdict, deque
from enum import Enum, IntEnum
import json
import os
import subprocess
import threading
import time

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation as sRot
import torch
import zmq

from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)

try:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        build_command_message,
        build_planner_message,
        pack_pose_message,
    )
except ImportError:

    def build_command_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_command_message unavailable")

    def build_planner_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_planner_message unavailable")

    def pack_pose_message(*args, **kwargs) -> bytes:
        raise RuntimeError("pack_pose_message unavailable")


try:
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
except ImportError:
    print("Warning: gear_sonic.isaac_utils.rotations not available.")
    remove_smpl_base_rot = None
    smpl_root_ytoz_up = None

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

try:
    from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
        G1GripperInverseKinematicsSolver,
    )
except ImportError:
    print("Warning: G1GripperInverseKinematicsSolver not available.")
    G1GripperInverseKinematicsSolver = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import VR3PtPoseVisualizer
except ImportError:
    print("Warning: VR3PtPoseVisualizer not available (pyvista may not be installed).")
    VR3PtPoseVisualizer = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import get_g1_key_frame_poses
except ImportError:
    print("Warning: get_g1_key_frame_poses not available (pyvista may not be installed).")
    get_g1_key_frame_poses = None


class LocomotionMode(IntEnum):
    """Locomotion mode enum for robot movement."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19


class StreamMode(Enum):
    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5


class XRStalenessWatchdog:
    """Track XR sample freshness after the stream has been armed."""

    def __init__(self, warn_ms: float = 50.0, estop_ms: float = 200.0):
        self.warn_ns = int(warn_ms * 1_000_000)
        self.estop_ns = int(estop_ms * 1_000_000)
        self._armed = False
        self._last_state = "idle"

    def reset(self) -> None:
        self._armed = False
        self._last_state = "idle"

    def poll(
        self,
        last_sample_ns: int | None,
        now_ns: int,
        active: bool = True,
    ) -> str:
        if not active:
            self.reset()
            return "idle"
        if last_sample_ns is None:
            state = "ok" if not self._armed else "estop"
        else:
            self._armed = True
            gap = now_ns - last_sample_ns
            if gap < self.warn_ns:
                state = "ok"
            elif gap < self.estop_ns:
                state = "warn"
            else:
                state = "estop"
        if state != self._last_state:
            print(f"[XRStalenessWatchdog] {self._last_state} -> {state}")
            self._last_state = state
        return state


### Parse 3 point pose from SMPL
#
# OFFSETS: Rotation corrections applied to each keypoint to align SMPL joint frames
# with the desired robot/visualization coordinate convention.
#
# Index mapping (based on [0, 22, 23, 12].index(joint_id)):
#   - OFFSETS[0]: Root/Pelvis (joint 0)
#   - OFFSETS[1]: Left Wrist (joint 22)
#   - OFFSETS[2]: Right Wrist (joint 23)
#   - OFFSETS[3]: Neck (joint 12) - more stable than Head (joint 15) for body tracking
#
# Scipy euler rotation convention:
#   - Lowercase "xyz" = EXTRINSIC rotations (about the FIXED/ORIGINAL frame's axes)
#   - Uppercase "XYZ" = INTRINSIC rotations (about the ROTATING body's axes)
#
# For EXTRINSIC "xyz" with angles [a, b, c]:
#   All rotations are about the ORIGINAL frame's axes (before any rotation):
#     R_total = R_z(c) @ R_y(b) @ R_x(a)   (matrix multiplication order)
#   Applied as: first rotate 'a' about original X, then 'b' about original Y, then 'c' about original Z
#
# For INTRINSIC "XYZ" with angles [a, b, c]:
#   Each rotation is about the CURRENT (rotated) frame's axis:
#     R_total = R_x(a) @ R_y(b) @ R_z(c)   (matrix multiplication order)
#   Applied as: first rotate 'a' about X, then 'b' about NEW Y, then 'c' about NEW Z
#
OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Root: yaw -90° about fixed Z
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),  # L-Wrist: roll +90° about fixed X
    sRot.from_euler(
        "xyz", [-90, 0, 180], degrees=True
    ),  # R-Wrist: roll -90° about fixed X, then yaw 180° about fixed Z
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Neck: yaw -90° about fixed Z
]


def _compute_rel_transform(pose, world_frame, scalar_first=True):
    """
    Transform a pose from Unity coordinate frame to robot coordinate frame.

    Args:
        pose: np.ndarray shape (7,) - [x, y, z, qx, qy, qz, qw] in Unity frame
        world_frame: np.ndarray shape (7,) - reference frame to compute relative transform
        scalar_first: bool - if True, quaternion is [qw, qx, qy, qz]; if False, [qx, qy, qz, qw]

    Returns:
        rel_pos: np.ndarray (3,) - position in robot frame
        rel_rot: np.ndarray (4,) - quaternion [qw, qx, qy, qz] in robot frame

    Coordinate transform matrix Q converts Unity (Y-up, left-handed) to Robot (Z-up, right-handed):
        Unity:  X-right, Y-up, Z-forward
        Robot:  X-forward, Y-left, Z-up
    """
    world_frame = world_frame.copy()

    # Q transforms Unity coordinates to Robot coordinates
    # Unity [x, y, z] -> Robot [-x, z, y]
    Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
    pose[:3] = Q @ pose[:3]
    world_frame[:3] = Q @ world_frame[:3]
    rot_base = sRot.from_quat(world_frame[3:], scalar_first=scalar_first).as_matrix()
    rot = sRot.from_quat(pose[3:], scalar_first=scalar_first).as_matrix()
    rel_rot = sRot.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    rel_pos = sRot.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
    return rel_pos, rel_rot.as_quat(scalar_first=True)


def _process_3pt_pose(smpl_pose_np):
    """
    Extract 3-point VR pose (L-Wrist, R-Wrist, Neck) from full SMPL body joint poses.

    NOTE: We use Neck (joint 12) instead of Head (joint 15) because:
      - Neck is more rigidly coupled to the torso
      - Head has high DoF (looking around) which doesn't reflect body pose
      - Neck provides more stable tracking for upper body orientation

    Args:
        smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints, each [x, y, z, qx, qy, qz, qw]
                      in Unity frame (scalar-last quaternion format)

    Returns:
        vr_3pt_pose: np.ndarray shape (3, 7) - 3 keypoints in robot frame
                     Each row is [x, y, z, qw, qx, qy, qz] (scalar-FIRST quaternion format)
                     Row 0: Left Wrist (SMPL joint 22)
                     Row 1: Right Wrist (SMPL joint 23)
                     Row 2: Neck (SMPL joint 12)

                     IMPORTANT: Positions and orientations are RELATIVE TO ROOT (pelvis).

    Processing Steps:
        1. Transform all 24 joints from Unity frame to robot frame
        2. Extract 4 keypoints: Root(0), L-Wrist(22), R-Wrist(23), Neck(12)
        3. Apply per-joint rotation OFFSETS to align joint frames
        4. Make L-Wrist, R-Wrist, Neck relative to Root (both position and orientation)
        5. Return only the 3 non-root keypoints

    Note: Position calibration (wrist offsets, neck kinematic chain) is done in
          ThreePointPose.apply_calibration() to ensure consistency with calibrated
          orientations.
    """

    # Defensive copy: _compute_rel_transform modifies pose[:3] in-place, which would
    # corrupt the caller's array (e.g. PicoReader._latest) and cause wrong results
    # if the same sample is processed more than once.
    smpl_pose_np = smpl_pose_np.copy()

    # =========================================================================
    # STEP 1: Transform all joints from Unity frame to robot frame
    # =========================================================================
    # Input: smpl_pose_np[i] = [x, y, z, qx, qy, qz, qw] in Unity frame (scalar-last)
    # Output: body_poses[i] = [x, y, z, qw, qx, qy, qz] in robot frame (scalar-first)
    body_poses = np.zeros((smpl_pose_np.shape[0], 7), dtype=np.float32)
    for i in range(smpl_pose_np.shape[0]):
        pos, orn = _compute_rel_transform(
            smpl_pose_np[i], [0, 0, 0, 0, 0, 0, 1], scalar_first=False
        )
        body_poses[i, :3] = pos  # Position in robot frame
        body_poses[i, 3:] = orn  # Quaternion [qw, qx, qy, qz] in robot frame

    # =========================================================================
    # STEP 2 & 3: Extract 4 keypoints and apply rotation OFFSETS
    # =========================================================================
    # We only care about these SMPL joint indices:
    #   - Joint 0:  Root/Pelvis (reference frame)
    #   - Joint 22: Left Wrist
    #   - Joint 23: Right Wrist
    #   - Joint 12: Neck (more stable than Head joint 15)
    #
    # kp_poses maps these to indices 0, 1, 2, 3 respectively
    positions = np.array([[p[0], p[1], p[2]] for p in body_poses])
    kp_poses = np.zeros((4, 7), dtype=np.float32)

    for i, pose in enumerate(body_poses):
        if i not in [0, 22, 23, 12]:
            continue  # Skip joints we don't care about

        pos = positions[i]

        # Map SMPL joint index to our keypoint index (0-3)
        # rel_i: 0=Root, 1=L-Wrist, 2=R-Wrist, 3=Neck
        rel_i = [0, 22, 23, 12].index(i)

        # Extract quaternion and apply rotation offset
        # pose[3:7] is [qw, qx, qy, qz] (scalar-first from _compute_rel_transform)
        quat = np.array([pose[3], pose[4], pose[5], pose[6]])

        # Apply offset: new_rotation = original_rotation * OFFSET
        # This post-multiplies the offset (intrinsic rotation)
        rot_quat = (sRot.from_quat(quat, scalar_first=True) * OFFSETS[rel_i]).as_quat(
            scalar_first=False
        )

        kp_poses[rel_i, 3:] = rot_quat  # Store as scalar-last temporarily for scipy compatibility
        kp_poses[rel_i, :3] = pos

    # =========================================================================
    # STEP 4: Make positions and orientations RELATIVE TO ROOT
    # =========================================================================
    # This transforms everything into the root's local coordinate frame.
    # After this step:
    #   - Root's position would be (0,0,0) and orientation identity (but we don't return root)
    #   - Other keypoints are expressed relative to root
    root_pos = kp_poses[0, :3].copy()
    root_quat = kp_poses[0, 3:].copy()  # Still scalar-last for scipy

    for i in range(1, 4):
        # Position: subtract root position, then rotate by inverse of root orientation
        kp_poses[i, :3] = sRot.from_quat(root_quat).inv().apply(kp_poses[i, :3] - root_pos)

        # Orientation: compute relative rotation (root_inv * keypoint_rot)
        # Result stored as scalar-FIRST [qw, qx, qy, qz]
        kp_poses[i, 3:] = (
            sRot.from_quat(root_quat).inv() * sRot.from_quat(kp_poses[i, 3:])
        ).as_quat(scalar_first=True)

    # =========================================================================
    # STEP 5: Return only L-Wrist, R-Wrist, Neck (skip Root)
    # =========================================================================
    # NOTE: Position and orientation calibration (including neck position via kinematic
    #       chain) is done in ThreePointPose.apply_calibration() to ensure consistency
    #       between calibrated orientation and computed neck position.
    # kp_poses[1:] = indices 1, 2, 3 = L-Wrist, R-Wrist, Neck
    # Each row: [x, y, z, qw, qx, qy, qz] relative to root, scalar-first quaternion
    return kp_poses[1:]


def _slerp_quat_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical interpolation for scalar-first quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return (result / max(np.linalg.norm(result), 1e-12)).astype(np.float32)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return (s0 * q0 + s1 * q1).astype(np.float32)


def _blend_vr_3pt_pose(
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend VR_3PT rows [xyz, qw, qx, qy, qz] from start to target."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    pose = np.asarray(target_pose, dtype=np.float32).copy()
    start_pose = np.asarray(start_pose, dtype=np.float32)
    pose[:, :3] = (1.0 - alpha) * start_pose[:, :3] + alpha * pose[:, :3]
    for row in range(pose.shape[0]):
        pose[row, 3:] = _slerp_quat_wxyz(start_pose[row, 3:], pose[row, 3:], alpha)
    return pose


CONTROLLER_POSE_CONVENTIONS = ("xrobotoolkit_unity", "openxr_unitree")


def _basis_matrix_for_controller_convention(convention: str) -> np.ndarray:
    """Return matrix that maps XR pose basis into robot basis."""
    if convention == "xrobotoolkit_unity":
        # Previous local implementation: XRoboToolkit Unity-like basis.
        # Unity: X-right, Y-up, Z-forward. Robot: X-forward, Y-left, Z-up.
        return np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]], dtype=np.float64)
    if convention == "openxr_unitree":
        # Unitree/televuer convention for OpenXR:
        # OpenXR: X-right, Y-up, Z-back. Robot: X-forward, Y-left, Z-up.
        return np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0.0]], dtype=np.float64)
    raise ValueError(
        f"Unknown controller pose convention {convention!r}; "
        f"expected one of {CONTROLLER_POSE_CONVENTIONS}"
    )


def _pose_xr_to_robot(
    pose: np.ndarray,
    scalar_first: bool = False,
    convention: str = "xrobotoolkit_unity",
) -> tuple[np.ndarray, sRot]:
    """Convert an XR controller/headset pose to the robot frame.

    XRoboToolkit poses are usually [x, y, z, qx, qy, qz, qw]. The returned
    quaternion rotation is a scipy Rotation in robot coordinates.
    """
    pose = np.asarray(pose, dtype=np.float64).copy()
    if pose.shape[0] < 7:
        raise ValueError(f"Expected pose with 7 values, got shape {pose.shape}")

    q_xr_to_robot = _basis_matrix_for_controller_convention(convention)
    pos = q_xr_to_robot @ pose[:3]
    rot = sRot.from_quat(pose[3:7], scalar_first=scalar_first)
    rot_robot = sRot.from_matrix(q_xr_to_robot @ rot.as_matrix() @ q_xr_to_robot.T)
    return pos, rot_robot


def _pose_unity_to_robot(pose: np.ndarray, scalar_first: bool = False) -> tuple[np.ndarray, sRot]:
    """Backward-compatible wrapper for the previous XRoboToolkit Unity transform."""
    return _pose_xr_to_robot(pose, scalar_first=scalar_first, convention="xrobotoolkit_unity")


def _valid_xrt_pose(pose: np.ndarray) -> bool:
    """Return True if an XRoboToolkit pose contains a usable quaternion."""
    pose = np.asarray(pose)
    if pose.shape[0] < 7:
        return False
    if not np.all(np.isfinite(pose[:7])):
        return False
    return float(np.linalg.norm(pose[3:7])) > 1e-6


def _process_controller_3pt_pose(
    left_controller_pose: np.ndarray,
    right_controller_pose: np.ndarray,
    headset_pose: np.ndarray,
    left_controller_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    right_controller_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    controller_pose_convention: str = "xrobotoolkit_unity",
    headset_pose_convention: str = "xrobotoolkit_unity",
    headset_orientation_convention: str | None = None,
) -> np.ndarray:
    """Build a raw VR 3-point pose from headset + two controller poses.

    Output rows match the existing VR_3PT contract:
      [left wrist, right wrist, head], each [x, y, z, qw, qx, qy, qz].

    Wrist positions are expressed relative to the headset position so the
    lower-body planner remains responsible for robot root motion. The controller
    and headset positions must be converted with the same basis before
    subtraction; otherwise mixed conventions create artificial wrist drift.
    """
    left_pos, left_rot = _pose_xr_to_robot(
        left_controller_pose, scalar_first=False, convention=controller_pose_convention
    )
    right_pos, right_rot = _pose_xr_to_robot(
        right_controller_pose, scalar_first=False, convention=controller_pose_convention
    )
    head_pos_for_wrist_relative, _ = _pose_xr_to_robot(
        headset_pose, scalar_first=False, convention=controller_pose_convention
    )
    head_pos, _ = _pose_xr_to_robot(
        headset_pose, scalar_first=False, convention=headset_pose_convention
    )
    _, head_rot = _pose_xr_to_robot(
        headset_pose,
        scalar_first=False,
        convention=headset_orientation_convention or headset_pose_convention,
    )
    left_rot = left_rot * sRot.from_euler("xyz", left_controller_offset_rpy, degrees=True)
    right_rot = right_rot * sRot.from_euler("xyz", right_controller_offset_rpy, degrees=True)

    vr_3pt_pose = np.zeros((3, 7), dtype=np.float32)
    vr_3pt_pose[0, :3] = left_pos - head_pos_for_wrist_relative
    vr_3pt_pose[1, :3] = right_pos - head_pos_for_wrist_relative
    vr_3pt_pose[2, :3] = head_pos
    vr_3pt_pose[0, 3:] = left_rot.as_quat(scalar_first=True)
    vr_3pt_pose[1, 3:] = right_rot.as_quat(scalar_first=True)
    vr_3pt_pose[2, 3:] = head_rot.as_quat(scalar_first=True)
    return vr_3pt_pose


def _as_list(value) -> list:
    return np.asarray(value, dtype=np.float64).tolist()


def _rotation_frame_debug(rot: sRot) -> dict:
    matrix = rot.as_matrix()
    return {
        "quat_wxyz": _as_list(rot.as_quat(scalar_first=True)),
        "rpy_xyz_deg": _as_list(rot.as_euler("xyz", degrees=True)),
        "axis_x": _as_list(matrix[:, 0]),
        "axis_y": _as_list(matrix[:, 1]),
        "axis_z": _as_list(matrix[:, 2]),
        "matrix_col_axes": _as_list(matrix),
    }


def _pose_frame_debug(position, rot: sRot) -> dict:
    return {
        "position": _as_list(position),
        "orientation": _rotation_frame_debug(rot),
    }


def _vr_3pt_row_debug(row: np.ndarray) -> dict:
    return _pose_frame_debug(row[:3], sRot.from_quat(row[3:], scalar_first=True))


def _controller_convention_debug(
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    head_pose: np.ndarray,
    left_controller_offset_rpy: tuple[float, float, float],
    right_controller_offset_rpy: tuple[float, float, float],
) -> dict:
    """Build side-by-side controller/head frames for each supported basis convention."""
    debug = {}
    for convention in CONTROLLER_POSE_CONVENTIONS:
        left_pos, left_rot = _pose_xr_to_robot(
            left_pose, scalar_first=False, convention=convention
        )
        right_pos, right_rot = _pose_xr_to_robot(
            right_pose, scalar_first=False, convention=convention
        )
        head_pos, head_rot = _pose_xr_to_robot(
            head_pose, scalar_first=False, convention=convention
        )
        left_ee_rot = left_rot * sRot.from_euler("xyz", left_controller_offset_rpy, degrees=True)
        right_ee_rot = right_rot * sRot.from_euler(
            "xyz", right_controller_offset_rpy, degrees=True
        )
        debug[convention] = {
            "left_controller_robot_position": left_pos.astype(np.float32),
            "right_controller_robot_position": right_pos.astype(np.float32),
            "headset_robot_position": head_pos.astype(np.float32),
            "left_controller_robot_orientation_wxyz": left_rot.as_quat(
                scalar_first=True
            ).astype(np.float32),
            "right_controller_robot_orientation_wxyz": right_rot.as_quat(
                scalar_first=True
            ).astype(np.float32),
            "headset_robot_orientation_wxyz": head_rot.as_quat(scalar_first=True).astype(
                np.float32
            ),
            "left_controller_ee_orientation_wxyz": left_ee_rot.as_quat(
                scalar_first=True
            ).astype(np.float32),
            "right_controller_ee_orientation_wxyz": right_ee_rot.as_quat(
                scalar_first=True
            ).astype(np.float32),
        }
    return debug


# =============================================================================
# VR 3-Point Pose Visualization Functions
# =============================================================================


def run_vr3pt_visualizer_test():
    """
    Standalone test for VR 3-point pose visualizer using PyVista.
    Run this to verify the reference frames are displayed correctly.
    """
    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Visualizer Test (PyVista)")
    print("=" * 60)
    print("\nExpected reference frames (all with RGB axes for XYZ):")
    print("  1. WHITE ball at origin (0, 0, 0) - World frame")
    print("  2. CYAN ball at (0, 0, 0.35) - Looking forward (identity)")
    print("  3. MAGENTA ball at (0, 0.4, 0.25) - Looking left (yaw +90°)")
    print("  4. YELLOW ball at (0.4, 0, 0.15) - Looking down (pitch +90°)")
    print("\nClose the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_static()


def run_vr3pt_live_visualizer():
    """
    Live visualizer for real VR 3-point pose data from Pico.
    Captures one frame from Pico and displays it alongside reference frames.
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use live visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Live Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Capturing VR 3-point pose...")

    # Capture body poses and compute vr_3pt_pose
    body_poses = xrt.get_body_joints_pose()
    body_poses_np = np.array(body_poses)

    # Process to get 3-point pose (L-Wrist, R-Wrist, Neck)
    vr_3pt_pose = _process_3pt_pose(body_poses_np)

    print(f"\nCaptured vr_3pt_pose shape: {vr_3pt_pose.shape}")
    print(f"  L-Wrist: pos={vr_3pt_pose[0, :3]}, quat_wxyz={vr_3pt_pose[0, 3:]}")
    print(f"  R-Wrist: pos={vr_3pt_pose[1, :3]}, quat_wxyz={vr_3pt_pose[1, 3:]}")
    print(f"  Neck:    pos={vr_3pt_pose[2, :3]}, quat_wxyz={vr_3pt_pose[2, 3:]}")

    print("\nDisplaying visualization...")
    print("Close the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_with_vr_pose(vr_3pt_pose)


def run_vr3pt_realtime_visualizer(update_hz: int = 10):
    """
    Real-time visualizer for VR 3-point pose data from Pico.
    Continuously updates the visualization with live data.

    Args:
        update_hz: Update rate in Hz (default 10)
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use realtime visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Real-time Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Starting real-time visualization...")
    print(f"Update rate: {update_hz} Hz")
    print("Close the window or press 'q' to exit.")
    print("=" * 60)

    # Use the VR3PtPoseVisualizer for real-time visualization with G1 robot
    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.create_realtime_plotter(interactive=True)

    try:
        while visualizer.is_open:
            # Get new data from Pico
            body_poses = xrt.get_body_joints_pose()
            body_poses_np = np.array(body_poses)
            vr_3pt_pose = _process_3pt_pose(body_poses_np)

            # Update visualization
            visualizer.update_vr_poses(vr_3pt_pose)
            visualizer.render()

            time.sleep(1.0 / update_hz)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        visualizer.close()


def process_smpl_joints(body_pose, global_orient, transl):
    """Process SMPL parameters to compute local joints.

    Args:
        body_pose: Body pose tensor, shape (T, 69)
        global_orient: Global orientation tensor, shape (T, 3)
        transl: Translation tensor, shape (T, 3)

    Returns:
        Dictionary with processed joints and parameters
    """
    # Convert global_orient to quaternion and apply transformations (robust if utils missing)
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    if smpl_root_ytoz_up is not None:
        global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    # Compute joints and vertices using SMPL model (single forward pass)
    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
    )  # (*, 24, 3)

    # Apply base rotation removal and compute local joints
    if remove_smpl_base_rot is not None:
        global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)
    global_orient_mat = quaternion_to_rotation_matrix(global_orient_quat)
    global_orient_6d = global_orient_mat[..., :2].reshape(1, 6)

    return {
        "smpl_pose": body_pose,
        "joints": joints,
        "smpl_joints_local": smpl_joints_local,
        "global_orient_quat": global_orient_quat,
        "global_orient_6d": global_orient_6d,
        "adjusted_transl": transl,
    }


def generate_finger_data(hand: str, trigger: float, grip: float) -> np.ndarray:
    """
    Generate finger position data from Pico controller button states.

    Args:
        hand: "left" or "right"
        trigger: Trigger button value (0-1)
        grip: Grip button value (0-1)

    Returns:
        Array of shape [25, 4, 4] representing fingertip positions
    """
    fingertips = np.zeros([25, 4, 4])

    thumb = 0
    middle = 10
    # Control thumb based on shoulder button state (index 4 is thumb tip)
    fingertips[4 + thumb, 0, 3] = 1.0  # open thumb
    if trigger > 0.5:
        fingertips[4 + middle, 0, 3] = 1.0  # close middle

    return fingertips


# Joystick deadzone threshold
JOYSTICK_DEADZONE = 0.15


class YawAccumulator:
    """Accumulates yaw heading angle based on joystick input."""

    def __init__(self, yaw_gain: float = 1.5, deadzone: float = JOYSTICK_DEADZONE):
        self.yaw_gain = yaw_gain
        self.deadzone = deadzone
        self.reset()

    def reset(self):
        """Reset facing direction to default (1,0,0)."""
        self.heading = [1.0, 0.0, 0.0]
        self.yaw_angle_rad = 0.0
        self.dyaw = 0.0
        print("YawAccumulator: reset yaw angle to 0.0")

    def yaw_angle(self) -> float:
        """Get current yaw angle in radians."""
        return self.yaw_angle_rad

    def yaw_angle_change(self) -> float:
        """Get current yaw angle change in radians."""
        return self.dyaw

    def update(self, rx: float, dt: float) -> list[float]:
        """
        Update facing direction based on right stick x-axis input.

        Args:
            rx: Right stick x-axis value (-1 to 1)
            dt: Time delta in seconds

        Returns:
            Facing direction as [x, y, 0.0]
        """
        self.dyaw = self.yaw_gain * (-rx) * dt
        if abs(rx) >= self.deadzone:
            self.yaw_angle_rad += self.dyaw
            self.heading = [np.cos(self.yaw_angle_rad), np.sin(self.yaw_angle_rad), 0.0]
        return self.heading


def compute_from_body_poses(parent_indices: list, device, body_poses_np: np.ndarray):
    """
    Compute local joints and body orientation from provided body_poses_np.
    """
    positions = body_poses_np[:, :3]
    global_quats = body_poses_np[:, [6, 3, 4, 5]]

    # Convert to local rotations
    global_rots = sRot.from_quat(global_quats, scalar_first=True)
    global_rots = global_rots * sRot.from_euler("y", 180, degrees=True)

    local_rots = []
    for i in range(24):
        if parent_indices[i] == -1:
            local_rots.append(global_rots[i])
        else:
            local_rot = global_rots[parent_indices[i]].inv() * global_rots[i]
            local_rots.append(local_rot)

    pose_aa = np.array([rot.as_rotvec() for rot in local_rots])

    body_pose = torch.from_numpy(pose_aa[1:].flatten()).float().to(device).unsqueeze(0)
    global_orient = torch.from_numpy(pose_aa[0]).float().to(device).unsqueeze(0)
    transl = torch.from_numpy(positions[0]).float().to(device).unsqueeze(0)

    return process_smpl_joints(body_pose, global_orient, transl)


# def compute_latest_frame(parent_indices: list, device) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Pull body data from XRoboToolkit, compute local SMPL joints and body orientation.
#     Returns (smpl_joints_local_np [24,3], global_orient_quat_np [4,])
#     """
#     body_poses = xrt.get_body_joints_pose()
#     body_poses_np = np.array(body_poses)
#     return compute_from_body_poses(parent_indices, device, body_poses_np)


def init_hand_ik_solvers():
    """Initialize hand IK solvers if available."""
    if G1GripperInverseKinematicsSolver is not None:
        left_solver = G1GripperInverseKinematicsSolver(side="left")
        right_solver = G1GripperInverseKinematicsSolver(side="right")
        print("Hand IK solvers initialized")
        return left_solver, right_solver
    print("Warning: Hand IK solvers not available")
    return None, None


def get_controller_inputs():
    """Fetch controller button/trigger states from XRoboToolkit."""
    left_trigger = xrt.get_left_trigger()
    right_trigger = xrt.get_right_trigger()
    left_grip = xrt.get_left_grip()
    right_grip = xrt.get_right_grip()
    left_menu_button = xrt.get_left_menu_button()
    return left_menu_button, left_trigger, right_trigger, left_grip, right_grip


def get_controller_axes():
    """Fetch joystick axes (lx, ly, rx, ry). Falls back to zeros if not available."""
    if xrt is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        left_axis = xrt.get_left_axis()  # expected [x, y]
        right_axis = xrt.get_right_axis()  # expected [x, y]
        lx = float(left_axis[0]) if len(left_axis) >= 1 else 0.0
        ly = float(left_axis[1]) if len(left_axis) >= 2 else 0.0
        rx = float(right_axis[0]) if len(right_axis) >= 1 else 0.0
        ry = float(right_axis[1]) if len(right_axis) >= 2 else 0.0
        return lx, ly, rx, ry
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def get_menu_buttons():
    """Fetch both menu buttons (left, right). Falls back to False if not available."""
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_menu_button")
    right = _safe_btn("get_right_menu_button")
    return left, right


def get_axis_clicks():
    """Fetch both axis click buttons (left, right). Falls back to False if not available."""
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_axis_click")
    right = _safe_btn("get_right_axis_click")
    return left, right


def get_face_buttons():
    """Fetch primary face buttons A and X. Returns (a_pressed, x_pressed)."""
    if xrt is None:
        return False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        x_pressed = bool(xrt.get_X_button())
        return a_pressed, x_pressed
    except Exception:
        return False, False


def get_abxy_buttons():
    """Fetch A,B,X,Y face buttons as booleans (a,b,x,y)."""
    if xrt is None:
        return False, False, False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        b_pressed = bool(xrt.get_B_button())
        x_pressed = bool(xrt.get_X_button())
        y_pressed = bool(xrt.get_Y_button())
        return a_pressed, b_pressed, x_pressed, y_pressed
    except Exception:
        return False, False, False, False


def compute_hand_joints_from_inputs(
    left_solver, right_solver, left_trigger, left_grip, right_trigger, right_grip
) -> tuple[np.ndarray, np.ndarray]:
    """Compute left/right hand joints using IK solvers, or zeros if unavailable."""
    if left_solver is not None and right_solver is not None:
        left_finger_data = generate_finger_data("left", left_trigger, left_grip)
        right_finger_data = generate_finger_data("right", right_trigger, right_grip)
        left_hand_joints = left_solver({"position": left_finger_data})
        right_hand_joints = right_solver({"position": right_finger_data})
    else:
        left_hand_joints = np.zeros((1, 7), dtype=np.float32)
        right_hand_joints = np.zeros((1, 7), dtype=np.float32)
    return left_hand_joints, right_hand_joints


def _quat_lerp_normalized(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Linear interpolate two quaternions and renormalize. Input shape (4,), xyzw order.
    Ensures shortest path by flipping sign if dot < 0.
    """
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q


def _interp_pose_axis_angle(
    prev_pose: np.ndarray, curr_pose: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Interpolate axis-angle joint poses by converting to quats, lerp-normalize, then back.
    prev_pose, curr_pose: (21,3) axis-angle (rotvec)
    Returns (21,3) axis-angle.
    """
    prev_quats = sRot.from_rotvec(prev_pose.reshape(-1, 3)).as_quat()  # (N,4) xyzw
    curr_quats = sRot.from_rotvec(curr_pose.reshape(-1, 3)).as_quat()
    out_quats = np.empty_like(prev_quats)
    for i in range(prev_quats.shape[0]):
        out_quats[i] = _quat_lerp_normalized(prev_quats[i], curr_quats[i], alpha)
    out_pose = sRot.from_quat(out_quats).as_rotvec().reshape(prev_pose.shape)
    return out_pose


class PicoReader:
    """
    Background reader that pulls Pico/XRT data as fast as possible and computes dt/FPS.
    """

    def __init__(self, max_queue_size: int = 15):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_t = None
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest = None
        self._lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            if not xrt.is_body_data_available():
                time.sleep(0.001)
                continue
            stamp_ns = xrt.get_time_stamp_ns()
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                time.sleep(0.000001)
                continue
            # Compute device-based dt/fps using timestamp deltas (ns -> s)
            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns
            t_realtime = time.time()
            t_monotonic = time.monotonic()
            try:
                body_poses = xrt.get_body_joints_pose()

                sample = {
                    "body_poses_np": np.array(body_poses),
                    "timestamp_realtime": t_realtime,
                    "timestamp_monotonic": t_monotonic,
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample
                now = time.time()
                if now - last_report >= 5.0:
                    print(
                        f"[PicoReader] dt_ts: {device_dt*1000.0:.2f} ms, fps: {self._fps_ema:.2f}"
                    )
                    last_report = now
            except Exception as e:
                print(f"[PicoReader] read error: {e}")


class ControllerPoseReader:
    """Background reader for headset + controller-only VR 3-point tracking."""

    def __init__(
        self,
        max_queue_size: int = 15,
        left_controller_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        right_controller_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        controller_pose_convention: str = "xrobotoolkit_unity",
        headset_pose_convention: str = "xrobotoolkit_unity",
        headset_orientation_convention: str | None = None,
    ):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_t = None
        self._fps_ema = 0.0
        self._latest = None
        self._lock = threading.Lock()
        self.left_controller_offset_rpy = left_controller_offset_rpy
        self.right_controller_offset_rpy = right_controller_offset_rpy
        self.controller_pose_convention = controller_pose_convention
        self.headset_pose_convention = headset_pose_convention
        self.headset_orientation_convention = (
            headset_orientation_convention or headset_pose_convention
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    def get_last_sample_monotonic_ns(self) -> int | None:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.get("sample_monotonic_ns")

    def _run(self):
        last_report = time.time()
        last_signature = None
        last_wait_report = 0.0
        while not self._stop.is_set():
            try:
                left_pose = np.array(xrt.get_left_controller_pose(), dtype=np.float32)
                right_pose = np.array(xrt.get_right_controller_pose(), dtype=np.float32)
                head_pose = np.array(xrt.get_headset_pose(), dtype=np.float32)
                if not (
                    _valid_xrt_pose(left_pose)
                    and _valid_xrt_pose(right_pose)
                    and _valid_xrt_pose(head_pose)
                ):
                    now = time.time()
                    if now - last_wait_report >= 2.0:
                        print(
                            "[ControllerPoseReader] waiting for valid headset/controller poses "
                            "(zero quaternion received)"
                        )
                        last_wait_report = now
                    time.sleep(0.02)
                    continue

                vr_3pt_pose = _process_controller_3pt_pose(
                    left_pose,
                    right_pose,
                    head_pose,
                    left_controller_offset_rpy=self.left_controller_offset_rpy,
                    right_controller_offset_rpy=self.right_controller_offset_rpy,
                    controller_pose_convention=self.controller_pose_convention,
                    headset_pose_convention=self.headset_pose_convention,
                    headset_orientation_convention=self.headset_orientation_convention,
                )
                convention_debug = _controller_convention_debug(
                    left_pose,
                    right_pose,
                    head_pose,
                    self.left_controller_offset_rpy,
                    self.right_controller_offset_rpy,
                )
                active_debug = convention_debug[self.controller_pose_convention]
                active_head_debug = convention_debug[self.headset_pose_convention]
                active_head_orientation_debug = convention_debug[
                    self.headset_orientation_convention
                ]
                left_pos_robot = active_debug["left_controller_robot_position"]
                right_pos_robot = active_debug["right_controller_robot_position"]
                head_pos_for_wrist_relative = active_debug["headset_robot_position"]
                head_pos_robot = active_head_debug["headset_robot_position"]
                left_rot_robot = sRot.from_quat(
                    active_debug["left_controller_robot_orientation_wxyz"], scalar_first=True
                )
                right_rot_robot = sRot.from_quat(
                    active_debug["right_controller_robot_orientation_wxyz"], scalar_first=True
                )
                head_rot_robot = sRot.from_quat(
                    active_head_orientation_debug["headset_robot_orientation_wxyz"],
                    scalar_first=True,
                )
                left_rot_ee = left_rot_robot * sRot.from_euler(
                    "xyz", self.left_controller_offset_rpy, degrees=True
                )
                right_rot_ee = right_rot_robot * sRot.from_euler(
                    "xyz", self.right_controller_offset_rpy, degrees=True
                )

                now = time.time()
                now_monotonic_ns = time.monotonic_ns()
                if self._last_t is None:
                    device_dt = 0.0
                else:
                    device_dt = now - self._last_t
                    if device_dt > 0.0:
                        inst = 1.0 / device_dt
                        self._fps_ema = (
                            inst if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * inst
                        )
                self._last_t = now

                signature = (
                    tuple(np.round(left_pose, 4)),
                    tuple(np.round(right_pose, 4)),
                    tuple(np.round(head_pose, 4)),
                )
                if signature == last_signature:
                    time.sleep(0.001)
                    continue
                last_signature = signature

                sample = {
                    "vr_3pt_pose_np": vr_3pt_pose,
                    "controller_pose_convention": self.controller_pose_convention,
                    "headset_pose_convention": self.headset_pose_convention,
                    "headset_orientation_convention": self.headset_orientation_convention,
                    "controller_pose_conventions": convention_debug,
                    "left_controller_pose_xrt": left_pose.copy(),
                    "right_controller_pose_xrt": right_pose.copy(),
                    "headset_pose_xrt": head_pose.copy(),
                    "left_controller_robot_position": left_pos_robot.astype(np.float32),
                    "right_controller_robot_position": right_pos_robot.astype(np.float32),
                    "headset_robot_position": head_pos_robot.astype(np.float32),
                    "headset_robot_position_for_wrist_relative": (
                        head_pos_for_wrist_relative.astype(np.float32)
                    ),
                    "left_controller_robot_orientation_wxyz": left_rot_robot.as_quat(
                        scalar_first=True
                    ).astype(np.float32),
                    "right_controller_robot_orientation_wxyz": right_rot_robot.as_quat(
                        scalar_first=True
                    ).astype(np.float32),
                    "headset_robot_orientation_wxyz": head_rot_robot.as_quat(
                        scalar_first=True
                    ).astype(np.float32),
                    "left_controller_ee_orientation_wxyz": left_rot_ee.as_quat(
                        scalar_first=True
                    ).astype(np.float32),
                    "right_controller_ee_orientation_wxyz": right_rot_ee.as_quat(
                        scalar_first=True
                    ).astype(np.float32),
                    "left_controller_offset_rpy": np.array(
                        self.left_controller_offset_rpy, dtype=np.float32
                    ),
                    "right_controller_offset_rpy": np.array(
                        self.right_controller_offset_rpy, dtype=np.float32
                    ),
                    "timestamp_realtime": now,
                    "timestamp_monotonic": time.monotonic(),
                    "sample_monotonic_ns": now_monotonic_ns,
                    "timestamp_ns": int(now * 1e9),
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample

                if now - last_report >= 5.0:
                    print(
                        f"[ControllerPoseReader] dt: {device_dt*1000.0:.2f} ms, "
                        f"fps: {self._fps_ema:.2f}"
                    )
                    last_report = now
            except Exception as e:
                print(f"[ControllerPoseReader] read error: {e}")
                time.sleep(0.05)


def _pose_stream_common(
    socket,
    buffer_size: int,
    num_frames_to_send: int,
    target_fps: int,
    use_cuda: bool,
    record_dir: str,
    record_format: str,
    stop_event: threading.Event | None = None,
    log_prefix: str = "PoseLoop",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
):
    """Shared pose streaming loop used by run_pico."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run pose streaming."
        )

    # Create reader and start it
    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()

    # Create 3-point pose processor with visualization settings
    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix=log_prefix,
    )

    streamer = PoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        log_prefix=log_prefix,
    )

    if stop_event is None:
        stop_event = threading.Event()

    try:
        while not stop_event.is_set():
            streamer.run_once()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup resources
        reader.stop()
        three_point.close()


class ThreePointPose:
    """
    Encapsulates everything around calculating 3-point pose from SMPL input.

    This includes:
    - Processing SMPL poses to extract 3-point VR pose (L-Wrist, R-Wrist, Neck)
    - Calibration logic to align VR poses with G1 robot
    - Optional visualization of 3-point poses

    Calibration is done in two steps:
    1. Neck orientation: Captures initial neck orientation to align subsequent poses as upright
    2. Wrist positions: Aligns wrist positions to match G1 robot key frame positions
    """

    # Kinematic chain constants for neck position (matches VR3PtPoseVisualizer)
    TORSO_LINK_OFFSET_Z = 0.05  # meters from root to torso_link
    NECK_LINK_LENGTH = 0.35  # meters from torso_link to neck along neck's local Z

    def __init__(
        self,
        enable_vis_vr3pt: bool = False,
        with_g1_robot: bool = True,
        enable_waist_tracking: bool = False,
        enable_smpl_vis: bool = False,
        log_prefix: str = "ThreePointPose",
        robot_model=None,
    ):
        """
        Initialize 3-point pose processor.

        Args:
            enable_vis_vr3pt: Whether to enable VR 3pt pose visualization (requires display)
            with_g1_robot: Whether to include G1 robot in visualization
            enable_waist_tracking: Whether to enable waist tracking in visualization
            enable_smpl_vis: Whether to render SMPL body joints in the VR3pt visualizer
            log_prefix: Prefix for log messages
            robot_model: Optional pre-instantiated RobotModel. If None, will create one.
                        Used for FK-based calibration (no display required).
        """
        self.log_prefix = log_prefix
        self.with_g1_robot = with_g1_robot
        self.enable_waist_tracking = enable_waist_tracking
        self.enable_smpl_vis = enable_smpl_vis

        # Robot model for FK-based calibration (headless, no display required)
        self._robot_model = robot_model
        if self._robot_model is None:
            from gear_sonic.data.robot_model.instantiation.g1 import (
                instantiate_g1_robot_model,
            )

            self._robot_model = instantiate_g1_robot_model()
            print(f"[{log_prefix}] Robot model loaded for FK calibration")

        # Optional visualization (requires display + PyVista)
        self.vr3pt_visualizer = None
        if enable_vis_vr3pt:
            if VR3PtPoseVisualizer is None:
                raise ImportError(
                    "VR3PtPoseVisualizer could not be imported but --vis_vr3pt was requested. "
                    "Ensure pyvista is installed: pip install pyvista"
                )
            self.vr3pt_visualizer = VR3PtPoseVisualizer(
                axis_length=0.08,
                ball_radius=0.015,
                with_g1_robot=with_g1_robot,
                robot_model=self._robot_model,
                enable_waist_tracking=enable_waist_tracking,
                enable_smpl_vis=enable_smpl_vis,
            )
            self.vr3pt_visualizer.create_realtime_plotter(interactive=True)
            g1_str = " with G1 robot" if with_g1_robot else ""
            waist_str = " + waist tracking" if enable_waist_tracking else ""
            smpl_str = " + SMPL body" if enable_smpl_vis else ""
            print(f"[{log_prefix}] VR 3pt pose visualization enabled{g1_str}{waist_str}{smpl_str}")

        # Calibration state — triggered explicitly by calibrate_now() or reset_with_measured_q()
        self._calibration_pending = False
        self._calibration_neck_quat_inv: np.ndarray | None = None  # inv(initial neck quat)
        self._calibration_neck_pos_offset: np.ndarray | None = None
        self._calibration_lwrist_offset: np.ndarray | None = None  # position offset
        self._calibration_rwrist_offset: np.ndarray | None = None
        self._calibration_lwrist_rot_offset: sRot | None = None  # orientation offset
        self._calibration_rwrist_rot_offset: sRot | None = None
        self._calibration_wrist_orientation_uses_neck = True
        # Override robot q for FK during recalibration (e.g. measured joints for VR 3PT)
        self._override_robot_q: np.ndarray | None = None

    @property
    def is_pending(self) -> bool:
        """Check if calibration is pending."""
        return self._calibration_pending

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration has been captured."""
        return self._calibration_neck_quat_inv is not None

    def process_smpl_pose(
        self,
        smpl_pose_np: np.ndarray,
        smpl_joints_local: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Process SMPL pose to extract and calibrate 3-point VR pose.

        Args:
            smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints
            smpl_joints_local: Optional np.ndarray shape (24, 3) - SMPL local joint
                               positions for body visualization. If provided and SMPL
                               visualization is enabled, the joint spheres are updated.

        Returns:
            vr_3pt_pose: np.ndarray shape (3, 7) - Calibrated 3-point pose
                         [L-Wrist, R-Wrist, Neck], each row [x, y, z, qw, qx, qy, qz]
        """
        # Extract raw 3-point pose from SMPL
        vr_3pt_pose_raw = _process_3pt_pose(smpl_pose_np)

        # Capture calibration on first valid frame (or after reset)
        if self._calibration_pending:
            self._capture_calibration(vr_3pt_pose_raw)

        # Apply calibration to get the final pose
        vr_3pt_pose = self._apply_calibration(vr_3pt_pose_raw)

        if self.vr3pt_visualizer is not None:
            self.vr3pt_visualizer.update_from_vr_pose(vr_3pt_pose, waist_scale=1.0)
            if smpl_joints_local is not None:
                self.vr3pt_visualizer.update_smpl_joints(smpl_joints_local)
            self.vr3pt_visualizer.render()

        return vr_3pt_pose

    def process_vr_3pt_pose(self, vr_3pt_pose_raw: np.ndarray) -> np.ndarray:
        """Process a raw headset/controller-derived VR 3-point pose."""
        if self._calibration_pending:
            self._capture_calibration(vr_3pt_pose_raw)

        vr_3pt_pose = self._apply_calibration(vr_3pt_pose_raw)
        if self.vr3pt_visualizer is not None:
            self.vr3pt_visualizer.update_from_vr_pose(vr_3pt_pose, waist_scale=1.0)
            self.vr3pt_visualizer.render()
        return vr_3pt_pose

    def close(self) -> None:
        """Close and cleanup visualizer resources."""
        if self.vr3pt_visualizer is not None:
            try:
                self.vr3pt_visualizer.close()
            except Exception as e:
                print(f"[{self.log_prefix}] Warning: Error closing VR3pt visualizer: {e}")

    def calibrate_now(self, body_poses_np: np.ndarray) -> bool:
        """Calibrate using current SMPL frame against FK of all-zero body joints.
        Operator should be in zero-reference pose when calling this."""
        try:
            vr_3pt_pose_raw = _process_3pt_pose(body_poses_np)
            self._override_robot_q = np.zeros(29, dtype=np.float64)
            self._capture_calibration(vr_3pt_pose_raw)
            print(f"[{self.log_prefix}] Calibration completed (zero-pose reference)")
            return True
        except Exception as e:
            print(f"[{self.log_prefix}] Calibration failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def calibrate_vr_3pt_now(self, vr_3pt_pose_raw: np.ndarray) -> bool:
        """Calibrate using current headset/controller 3-point frame."""
        try:
            self._override_robot_q = np.zeros(29, dtype=np.float64)
            self._calibration_wrist_orientation_uses_neck = False
            self._capture_calibration(vr_3pt_pose_raw)
            print(f"[{self.log_prefix}] Controller 3PT calibration completed")
            return True
        except Exception as e:
            print(f"[{self.log_prefix}] Controller 3PT calibration failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_g1_ee_debug(self, body_q_measured: np.ndarray | None = None) -> dict:
        """Return current/reference G1 wrist FK frames for calibration debugging."""
        if self._robot_model is None or get_g1_key_frame_poses is None:
            return {}
        if body_q_measured is not None:
            robot_q = self._robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_q_measured[:29]
            )
            source = "measured_q"
        else:
            robot_q = None
            source = "default_zero_q"
        g1_poses = get_g1_key_frame_poses(self._robot_model, q=robot_q)
        left_rot = sRot.from_quat(g1_poses["left_wrist"]["orientation_wxyz"], scalar_first=True)
        right_rot = sRot.from_quat(g1_poses["right_wrist"]["orientation_wxyz"], scalar_first=True)
        return {
            "source": source,
            "left_wrist": _pose_frame_debug(g1_poses["left_wrist"]["position"], left_rot),
            "right_wrist": _pose_frame_debug(g1_poses["right_wrist"]["position"], right_rot),
        }

    def _capture_calibration(self, vr_3pt_pose: np.ndarray) -> None:
        """Capture calibration offsets from vr_3pt_pose against G1 FK reference.
        If neck calibration already exists (e.g. from calibrate_now), it is preserved
        to avoid jumps from SMPL noise during recalibration."""

        # Step 1: Neck orientation — only capture if not already set
        if self._calibration_neck_quat_inv is None:
            neck_quat_wxyz = vr_3pt_pose[2, 3:].copy()
            neck_rot = sRot.from_quat(neck_quat_wxyz, scalar_first=True)
            self._calibration_neck_quat_inv = neck_rot.inv().as_quat(scalar_first=True)
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        # Step 2: Rotate VR wrist positions/orientations by neck inverse
        lwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[0, :3].copy())
        rwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[1, :3].copy())
        neck_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[2, :3].copy())
        lwrist_rot_raw = sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
        rwrist_rot_raw = sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)
        if self._calibration_wrist_orientation_uses_neck:
            lwrist_rot_corrected = calib_inv_rot * lwrist_rot_raw
            rwrist_rot_corrected = calib_inv_rot * rwrist_rot_raw
        else:
            lwrist_rot_corrected = lwrist_rot_raw
            rwrist_rot_corrected = rwrist_rot_raw

        # Step 3: Get G1 FK reference poses
        if self._robot_model is None:
            raise RuntimeError(
                "Robot model is required for calibration but was not loaded. "
                "Ensure the G1 robot model and URDF are available."
            )
        if get_g1_key_frame_poses is None:
            raise RuntimeError(
                "get_g1_key_frame_poses could not be imported. "
                "Ensure gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer is available."
            )

        # Convert 29-DOF override to full model config if needed
        if self._override_robot_q is not None:
            robot_q = self._robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=self._override_robot_q[:29]
            )
        else:
            robot_q = None
        g1_poses = get_g1_key_frame_poses(self._robot_model, q=robot_q)

        g1_lwrist_pos = g1_poses["left_wrist"]["position"]
        g1_rwrist_pos = g1_poses["right_wrist"]["position"]
        g1_neck_pos = g1_poses["torso"]["position"]
        g1_lwrist_rot = sRot.from_quat(
            g1_poses["left_wrist"]["orientation_wxyz"], scalar_first=True
        )
        g1_rwrist_rot = sRot.from_quat(
            g1_poses["right_wrist"]["orientation_wxyz"], scalar_first=True
        )

        # Compute position offsets: calibrated = neck_corrected - offset
        self._calibration_lwrist_offset = lwrist_pos_corrected - g1_lwrist_pos
        self._calibration_rwrist_offset = rwrist_pos_corrected - g1_rwrist_pos
        self._calibration_neck_pos_offset = neck_pos_corrected - g1_neck_pos

        # Compute orientation offsets: calibrated = rot_offset * neck_corrected
        self._calibration_lwrist_rot_offset = g1_lwrist_rot * lwrist_rot_corrected.inv()
        self._calibration_rwrist_rot_offset = g1_rwrist_rot * rwrist_rot_corrected.inv()

        self._calibration_pending = False
        self._override_robot_q = None

        # Log summary
        source = "override q" if g1_lwrist_pos.any() else "default/zero"
        print(
            f"[{self.log_prefix}] Calibration captured (FK ref: {source}):\n"
            f"  L-Wrist pos offset: [{self._calibration_lwrist_offset[0]:.4f}, "
            f"{self._calibration_lwrist_offset[1]:.4f}, {self._calibration_lwrist_offset[2]:.4f}]\n"
            f"  R-Wrist pos offset: [{self._calibration_rwrist_offset[0]:.4f}, "
            f"{self._calibration_rwrist_offset[1]:.4f}, {self._calibration_rwrist_offset[2]:.4f}]\n"
            f"  Neck pos offset: [{self._calibration_neck_pos_offset[0]:.4f}, "
            f"{self._calibration_neck_pos_offset[1]:.4f}, {self._calibration_neck_pos_offset[2]:.4f}]"
        )

    def _apply_calibration(self, vr_3pt_pose: np.ndarray) -> np.ndarray:
        """Apply stored calibration offsets to raw VR 3-point pose."""
        if self._calibration_neck_quat_inv is None:
            return vr_3pt_pose

        calibrated = vr_3pt_pose.copy()
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        # Neck orientation: calibrated = inv(initial) * current
        neck_rot = sRot.from_quat(vr_3pt_pose[2, 3:], scalar_first=True)
        calibrated[2, 3:] = (calib_inv_rot * neck_rot).as_quat(scalar_first=True)

        # Wrist positions: rotate by neck inverse, then subtract offset
        if self._calibration_lwrist_offset is not None:
            calibrated[0, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[0, :3]) - self._calibration_lwrist_offset
            )
        if self._calibration_rwrist_offset is not None:
            calibrated[1, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[1, :3]) - self._calibration_rwrist_offset
            )

        # Wrist orientations: rot_offset * (neck_inv * current)
        if self._calibration_lwrist_rot_offset is not None:
            lw_raw = sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
            lw_corrected = (
                calib_inv_rot * lw_raw if self._calibration_wrist_orientation_uses_neck else lw_raw
            )
            calibrated[0, 3:] = (self._calibration_lwrist_rot_offset * lw_corrected).as_quat(
                scalar_first=True
            )
        if self._calibration_rwrist_rot_offset is not None:
            rw_raw = sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)
            rw_corrected = (
                calib_inv_rot * rw_raw if self._calibration_wrist_orientation_uses_neck else rw_raw
            )
            calibrated[1, 3:] = (self._calibration_rwrist_rot_offset * rw_corrected).as_quat(
                scalar_first=True
            )

        # Neck/torso target position: track the calibrated headset translation.
        # The official VR_3PT target is a 3D torso/head proxy, so do not discard
        # headset position and synthesize this point from orientation alone.
        if self._calibration_neck_pos_offset is not None:
            calibrated[2, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[2, :3]) - self._calibration_neck_pos_offset
            ).astype(np.float32)

        return calibrated

    def _clear_calibration(self):
        """Clear all calibration state."""
        self._calibration_neck_quat_inv = None
        self._calibration_neck_pos_offset = None
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._calibration_wrist_orientation_uses_neck = True
        self._override_robot_q = None

    def reset(self) -> None:
        """Reset calibration. Next process_smpl_pose() call will recalibrate."""
        self._clear_calibration()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Calibration reset, will re-calibrate on next frame")

    def reset_with_measured_q(self, body_q_measured: np.ndarray) -> None:
        """Recalibrate wrist offsets using measured robot joints (29 DOFs).
        Preserves neck calibration to avoid jumps from SMPL noise.
        Next process_smpl_pose() will recompute wrist offsets against FK of these joints."""
        # Preserve neck calibration — only clear wrist offsets
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._calibration_wrist_orientation_uses_neck = False
        self._override_robot_q = body_q_measured.copy()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Wrist recalibration pending (neck preserved, measured q)")


class PoseStreamer:
    """Encapsulates the pose streaming loop state and logic."""

    def __init__(
        self,
        socket,
        reader: PicoReader,
        three_point: ThreePointPose,
        num_frames_to_send: int,
        target_fps: int,
        use_cuda: bool,
        record_dir: str,
        record_format: str,
        log_prefix: str = "PoseLoop",
    ):
        self.socket = socket
        self.reader = reader
        self.num_frames_to_send = num_frames_to_send
        self.target_fps = target_fps
        self.record_dir = record_dir
        self.log_prefix = log_prefix

        # Injected dependencies
        self.reader = reader
        self.three_point = three_point

        self.device = (
            torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")
        )

        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self.record_idx = 0

        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()
        self.parent_indices = [
            -1,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            9,
            9,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            22,
            23,
        ][:24]

        self.step = 0
        self.last_fps_report = time.time()
        self.fps_counter = 0
        # NOTE: Sleep budget set to 95% of the ideal frame period so that the actual
        # FPS lands closer to target_fps despite per-frame processing overhead.
        self.frame_time = 0.95 / max(1, target_fps)
        self.frame_buffer = defaultdict(lambda: deque(maxlen=num_frames_to_send))

        self.prev_stamp_ns = None
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.next_target_ns = None
        self.frame_start = time.time()

        # Data collection button state tracking (edge-triggered)
        self.toggle_data_collection_last = False
        self.toggle_data_abort_last = False

        self.buffer_cleared = (
            True  # Start with buffer cleared - wait for full buffer before first send
        )
        self.yaw_accumulator = YawAccumulator()

    def reset_yaw(self):
        """Called when entering pose mode. Resets yaw only.
        Calibration is triggered separately by the operator (A+B+X+Y → calibrate_now)."""
        self.yaw_accumulator.reset()

    def on_mode_exit(self):
        self.frame_buffer.clear()
        self.prev_stamp_ns = None
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.next_target_ns = None
        self.buffer_cleared = True
        self.step = 0

    def run_once(self):
        """Execute one iteration of the pose streaming loop."""
        sample = self.reader.get_latest()

        if sample is None:
            time.sleep(0.005)
            return

        latest_data = compute_from_body_poses(
            self.parent_indices, self.device, sample["body_poses_np"]
        )
        (left_menu_button, left_trigger, right_trigger, left_grip, right_grip) = (
            get_controller_inputs()
        )
        # Get A and B button states for data collection control
        a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()

        # Data collection toggle logic (edge-triggered)
        # Left grip + A = toggle_data_collection
        # Left grip + B = toggle_data_abort
        toggle_data_collection_tmp = a_pressed and left_grip > 0.5
        toggle_data_abort_tmp = b_pressed and left_grip > 0.5

        # Detect rising edge
        toggle_data_collection = toggle_data_collection_tmp and not self.toggle_data_collection_last
        toggle_data_abort = toggle_data_abort_tmp and not self.toggle_data_abort_last
        self.toggle_data_collection_last = toggle_data_collection_tmp
        self.toggle_data_abort_last = toggle_data_abort_tmp

        left_hand_joints, right_hand_joints = compute_hand_joints_from_inputs(
            self.left_hand_ik_solver,
            self.right_hand_ik_solver,
            left_trigger,
            left_grip,
            right_trigger,
            right_grip,
        )
        smpl_pose_np = (
            latest_data["smpl_pose"].detach().cpu().numpy()[:, :63].reshape(-1, 21, 3)[0]
        ).astype(np.float32)
        smpl_joints_np = (
            latest_data["smpl_joints_local"].detach().cpu().numpy()[0].astype(np.float32)
        )
        body_quat_np = (
            latest_data["global_orient_quat"].detach().cpu().numpy()[0].astype(np.float32)
        )
        curr_stamp_ns = int(sample.get("timestamp_ns", 0))
        step_ns = int(1e9 / max(1, self.target_fps))
        if self.prev_stamp_ns is None:
            self.prev_stamp_ns = curr_stamp_ns
            self.prev_smpl_pose_np = smpl_pose_np
            self.prev_smpl_joints_np = smpl_joints_np
            self.prev_body_quat_np = body_quat_np
            self.next_target_ns = curr_stamp_ns
            return
        if curr_stamp_ns <= self.prev_stamp_ns:
            return
        if self.next_target_ns is None:
            self.next_target_ns = self.prev_stamp_ns + step_ns
        if self.next_target_ns < self.prev_stamp_ns:
            self.next_target_ns = self.prev_stamp_ns
        if self.next_target_ns > curr_stamp_ns:
            return
        denom = float(curr_stamp_ns - self.prev_stamp_ns)
        alpha = float(self.next_target_ns - self.prev_stamp_ns) / denom if denom > 0.0 else 1.0
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0
        use_joints = (1.0 - alpha) * self.prev_smpl_joints_np + alpha * smpl_joints_np
        use_pose = _interp_pose_axis_angle(self.prev_smpl_pose_np, smpl_pose_np, alpha).astype(
            np.float32
        )
        use_body_quat = _quat_lerp_normalized(self.prev_body_quat_np, body_quat_np, alpha).astype(
            np.float32
        )
        N = len(self.frame_buffer["frame_index"])

        ##### From @Jiefeng for directly setting the joint position ######
        joint_pos = np.zeros(29)
        body_pose = use_pose.reshape(-1, 21, 3)

        SMPL_L_ELBOW_IDX = 17
        SMPL_L_WRIST_IDX = 19
        SMPL_R_ELBOW_IDX = 18
        SMPL_R_WRIST_IDX = 20

        # G1_L_ELBOW_IDX = 0
        G1_L_WRIST_ROLL_IDX = 23
        G1_L_WRIST_PITCH_IDX = 25
        G1_L_WRIST_YAW_IDX = 27

        # G1_R_ELBOW_IDX = 0
        G1_R_WRIST_ROLL_IDX = 24  # Done
        G1_R_WRIST_PITCH_IDX = 26
        G1_R_WRIST_YAW_IDX = 28
        smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
        smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
        smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
        smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

        g1_l_elbow_axis = np.array([0, 1, 0])
        g1_l_elbow_q_twist, g1_l_elbow_q_swing = decompose_rotation_aa(
            smpl_l_elbow_aa, g1_l_elbow_axis
        )

        g1_r_elbow_axis = np.array([0, 1, 0])
        g1_r_elbow_q_twist, g1_r_elbow_q_swing = decompose_rotation_aa(
            smpl_r_elbow_aa, g1_r_elbow_axis
        )

        # Move elbow roll/yaw into wrist while preserving wrist pitch from SMPL
        l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )
        r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )

        l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
        r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

        g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
        g1_l_wrist_pitch = -l_wrist_euler[:, 1]
        g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

        g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
        g1_r_wrist_pitch = -r_wrist_euler[:, 1]
        g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

        joint_pos[G1_L_WRIST_ROLL_IDX] = g1_l_wrist_roll[0]
        joint_pos[G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch[0]
        joint_pos[G1_L_WRIST_YAW_IDX] = g1_l_wrist_yaw[0]

        joint_pos[G1_R_WRIST_ROLL_IDX] = g1_r_wrist_roll[0]
        joint_pos[G1_R_WRIST_PITCH_IDX] = g1_r_wrist_pitch[0]
        joint_pos[G1_R_WRIST_YAW_IDX] = g1_r_wrist_yaw[0]

        # Process SMPL pose to get calibrated 3-point VR pose and update visualization
        # Pass SMPL local joints for optional body visualization in the VR3Pt viewer
        smpl_joints_for_vis = (
            latest_data["smpl_joints_local"].detach().cpu().numpy()[0]
            if self.three_point.enable_smpl_vis
            else None
        )
        vr_3pt_pose = self.three_point.process_smpl_pose(
            sample["body_poses_np"], smpl_joints_local=smpl_joints_for_vis
        )
        ##### From @Jiefeng for directly setting the joint position ######

        self.frame_buffer["smpl_pose"].append(use_pose)
        self.frame_buffer["smpl_joints"].append(use_joints)
        self.frame_buffer["body_quat_w"].append(use_body_quat)
        self.frame_buffer["frame_index"].append(int(self.step))
        self.frame_buffer["joint_pos"].append(joint_pos)
        pico_dt = float(sample.get("dt", 0.0))
        pico_fps = float(sample.get("fps", 0.0))
        N = len(self.frame_buffer["frame_index"])

        # Wait for buffer to be completely filled before sending first message after clearing
        buffer_is_full = len(self.frame_buffer["frame_index"]) >= self.num_frames_to_send
        if buffer_is_full and self.buffer_cleared:
            # Buffer is now full with fresh data, can start sending
            self.buffer_cleared = False

        # Get joystick axes for yaw accumulation
        _, _, rx, _ = get_controller_axes()
        self.yaw_accumulator.update(rx, self.frame_time)

        # Only send if buffer is full and we're not waiting for fresh data
        if buffer_is_full and not self.buffer_cleared:
            numpy_data = {
                "smpl_pose": np.stack((self.frame_buffer["smpl_pose"]), axis=0),
                "smpl_joints": np.stack((self.frame_buffer["smpl_joints"]), axis=0),
                "body_quat_w": np.stack((self.frame_buffer["body_quat_w"]), axis=0),
                "joint_pos": np.stack((self.frame_buffer["joint_pos"]), axis=0),
                "joint_vel": np.zeros((N, 29)),
                "vr_position": vr_3pt_pose[:, :3].flatten(),
                "vr_orientation": vr_3pt_pose[:, 3:].flatten(),
                "frame_index": np.array((self.frame_buffer["frame_index"]), dtype=np.int64),
                "left_trigger": np.array([left_trigger], dtype=np.float32),
                "right_trigger": np.array([right_trigger], dtype=np.float32),
                "left_grip": np.array([left_grip], dtype=np.float32),
                "right_grip": np.array([right_grip], dtype=np.float32),
                "pico_dt": np.array([pico_dt], dtype=np.float32),
                "pico_fps": np.array([pico_fps], dtype=np.float32),
                "timestamp_realtime": np.array(
                    [sample.get("timestamp_realtime", 0.0)], dtype=np.float64
                ),
                "timestamp_monotonic": np.array(
                    [sample.get("timestamp_monotonic", 0.0)], dtype=np.float64
                ),
                "left_hand_joints": left_hand_joints.reshape(-1).astype(np.float32),
                "right_hand_joints": right_hand_joints.reshape(-1).astype(np.float32),
                "toggle_data_collection": np.array([toggle_data_collection], dtype=bool),
                "toggle_data_abort": np.array([toggle_data_abort], dtype=bool),
                "heading_increment": np.array(
                    [self.yaw_accumulator.yaw_angle_change()], dtype=np.float32
                ),
            }

            packed_message = pack_pose_message(numpy_data, topic="pose")
            self.socket.send(packed_message)

            if self.record_dir:
                out_path = os.path.join(self.record_dir, f"pose_{self.record_idx:06d}.npz")
                np.savez_compressed(out_path, **numpy_data)
                self.record_idx += 1

        self.step += 1
        self.next_target_ns += step_ns
        self.prev_stamp_ns = curr_stamp_ns
        self.prev_smpl_pose_np = smpl_pose_np
        self.prev_smpl_joints_np = smpl_joints_np
        self.prev_body_quat_np = body_quat_np
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_report >= 5.0:
            fps = self.fps_counter / (current_time - self.last_fps_report)
            print(f"[{self.log_prefix}] FPS: {fps:.2f}, Step: {self.step}")
            self.fps_counter = 0
            self.last_fps_report = current_time
        elapsed = time.time() - self.frame_start
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        self.frame_start = time.time()


def run_pico(
    buffer_size: int = 15,
    port: int = 5556,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
):
    """Run Pico body tracking with real-time visualization and ZMQ streaming."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run Pico streaming."
        )
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"ZMQ socket bound to port {port}")
    if build_command_message is not None and build_planner_message is not None:
        try:
            socket.send(build_command_message(start=False, stop=False, planner=False))
            socket.send(build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0))
        except Exception as e:
            print(f"Warning: failed to send initial command/planner messages: {e}")
    try:
        _pose_stream_common(
            socket=socket,
            buffer_size=buffer_size,
            num_frames_to_send=num_frames_to_send,
            target_fps=target_fps,
            use_cuda=use_cuda,
            record_dir=record_dir,
            record_format=record_format,
            stop_event=None,
            log_prefix="Main",
            enable_vis_vr3pt=enable_vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=enable_waist_tracking,
            enable_smpl_vis=enable_smpl_vis,
        )
    finally:
        socket.close()
        context.term()
        print("Threads stopped, ZMQ socket closed")


class FeedbackReader:
    """Reads feedback from robot via ZMQ and processes measured upper body position to use as frozen targets."""

    def __init__(self, zmq_feedback_host: str = "localhost", zmq_feedback_port: int = 5557):
        self.poller = ZMQPoller(host=zmq_feedback_host, port=zmq_feedback_port, topic="g1_debug")

        self.upper_body_joint_indices = self._get_upper_body_joint_indices()

        self.upper_body_position_target = None
        self.left_hand_position_target = None
        self.right_hand_position_target = None
        # Full body joint configuration (29 DOFs) as measured from robot,
        # used for FK when recalibrating VR 3PT tracking against actual robot pose
        self.full_body_q_measured: np.ndarray | None = None

    def _get_upper_body_joint_indices(self) -> list[int]:
        # TODO: get from robot model, not hardcoded
        # robot_model = instantiate_g1_robot_model()
        # return robot_model.get_joint_group_indices("upper_body")
        return [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

    def poll_feedback(self):
        """Poll for feedback once, and update internal state."""
        (
            self.upper_body_position_target,
            self.left_hand_position_target,
            self.right_hand_position_target,
            self.full_body_q_measured,
        ) = self._process_upper_body_position_targets()
        print("[PlannerLoop] Saved upper body position target:", self.upper_body_position_target)

    def _process_upper_body_position_targets(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        data = self.poller.get_data()

        if data is None:
            print("[PlannerLoop] No feedback data received")
            return None, None, None, None

        unpacked = msgpack.unpackb(data, raw=False)
        full_body_q = None
        if "body_q_measured" in unpacked:
            body_q_swizzled = unpacked["body_q_measured"]
            full_body_q = np.array(body_q_swizzled, dtype=np.float64)
            body_q = [body_q_swizzled[i] for i in self.upper_body_joint_indices]
        else:
            print("[PlannerLoop] body_q_measured not in feedback data")
            body_q = None

        if "left_hand_q_measured" in unpacked:
            left_hand_q = unpacked["left_hand_q_measured"]
        else:
            print("[PlannerLoop] left_hand_q_measured not in feedback data")
            left_hand_q = None

        if "right_hand_q_measured" in unpacked:
            right_hand_q = unpacked["right_hand_q_measured"]
        else:
            print("[PlannerLoop] right_hand_q_measured not in feedback data")
            right_hand_q = None

        return body_q, left_hand_q, right_hand_q, full_body_q


class Controller3PtCalibrationLogger:
    """JSONL logger for comparing PICO controller frames against G1 EE FK frames."""

    def __init__(
        self,
        log_dir: str,
        interval_s: float = 1.0,
        posture_marker: "Controller3PtPostureMarker | None" = None,
    ):
        self.log_dir = log_dir
        self.interval_s = interval_s
        self.posture_marker = posture_marker
        self.last_log_time = 0.0
        self.path = ""
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.path = os.path.join(log_dir, f"controller_3pt_calibration_{stamp}.jsonl")
            print(f"[Controller3PtLogger] Writing calibration frames to {self.path}")

    def maybe_log(
        self,
        event: str,
        sample: dict,
        raw_vr_3pt_pose: np.ndarray,
        calibrated_vr_3pt_pose: np.ndarray,
        three_point: ThreePointPose,
        body_q_measured: np.ndarray | None = None,
    ) -> None:
        now = time.time()
        if now - self.last_log_time < self.interval_s:
            return
        self.last_log_time = now
        self.log(event, sample, raw_vr_3pt_pose, calibrated_vr_3pt_pose, three_point, body_q_measured)

    def log(
        self,
        event: str,
        sample: dict,
        raw_vr_3pt_pose: np.ndarray,
        calibrated_vr_3pt_pose: np.ndarray,
        three_point: ThreePointPose,
        body_q_measured: np.ndarray | None = None,
    ) -> None:
        g1_ee = three_point.get_g1_ee_debug(body_q_measured)
        posture = (
            self.posture_marker.snapshot()
            if self.posture_marker is not None
            else {"index": -1, "label": "unlabeled"}
        )
        record = {
            "event": event,
            "posture_marker": posture,
            "timestamp_realtime": float(sample.get("timestamp_realtime", time.time())),
            "timestamp_monotonic": float(sample.get("timestamp_monotonic", time.monotonic())),
            "timestamp_ns": int(sample.get("timestamp_ns", 0)),
            "pico": {
                "active_controller_pose_convention": sample.get(
                    "controller_pose_convention", "xrobotoolkit_unity"
                ),
                "active_headset_pose_convention": sample.get(
                    "headset_pose_convention", "xrobotoolkit_unity"
                ),
                "active_headset_orientation_convention": sample.get(
                    "headset_orientation_convention",
                    sample.get("headset_pose_convention", "xrobotoolkit_unity"),
                ),
                "left_controller_pose_xrt": _as_list(sample["left_controller_pose_xrt"]),
                "right_controller_pose_xrt": _as_list(sample["right_controller_pose_xrt"]),
                "headset_pose_xrt": _as_list(sample["headset_pose_xrt"]),
                "left_controller_robot_frame": _pose_frame_debug(
                    sample["left_controller_robot_position"],
                    sRot.from_quat(
                        sample["left_controller_robot_orientation_wxyz"], scalar_first=True
                    ),
                ),
                "right_controller_robot_frame": _pose_frame_debug(
                    sample["right_controller_robot_position"],
                    sRot.from_quat(
                        sample["right_controller_robot_orientation_wxyz"], scalar_first=True
                    ),
                ),
                "headset_robot_frame": _pose_frame_debug(
                    sample["headset_robot_position"],
                    sRot.from_quat(sample["headset_robot_orientation_wxyz"], scalar_first=True),
                ),
                "headset_robot_frame_for_wrist_relative": _pose_frame_debug(
                    sample.get(
                        "headset_robot_position_for_wrist_relative",
                        sample["headset_robot_position"],
                    ),
                    sRot.from_quat(sample["headset_robot_orientation_wxyz"], scalar_first=True),
                ),
                "left_controller_ee_frame": _pose_frame_debug(
                    sample["left_controller_robot_position"],
                    sRot.from_quat(
                        sample["left_controller_ee_orientation_wxyz"], scalar_first=True
                    ),
                ),
                "right_controller_ee_frame": _pose_frame_debug(
                    sample["right_controller_robot_position"],
                    sRot.from_quat(
                        sample["right_controller_ee_orientation_wxyz"], scalar_first=True
                    ),
                ),
                "left_controller_offset_rpy": _as_list(sample["left_controller_offset_rpy"]),
                "right_controller_offset_rpy": _as_list(sample["right_controller_offset_rpy"]),
            },
            "controller_pose_convention_debug": self._pose_convention_debug(sample),
            "vr_3pt_raw_relative_to_head": {
                "left_wrist": _vr_3pt_row_debug(raw_vr_3pt_pose[0]),
                "right_wrist": _vr_3pt_row_debug(raw_vr_3pt_pose[1]),
                "head": _vr_3pt_row_debug(raw_vr_3pt_pose[2]),
            },
            "vr_3pt_calibrated_target": {
                "left_wrist": _vr_3pt_row_debug(calibrated_vr_3pt_pose[0]),
                "right_wrist": _vr_3pt_row_debug(calibrated_vr_3pt_pose[1]),
                "head": _vr_3pt_row_debug(calibrated_vr_3pt_pose[2]),
            },
            "g1_ee_fk": g1_ee,
        }
        if g1_ee:
            self._add_target_vs_g1_errors(record)

        left_rpy = record["pico"]["left_controller_ee_frame"]["orientation"]["rpy_xyz_deg"]
        right_rpy = record["pico"]["right_controller_ee_frame"]["orientation"]["rpy_xyz_deg"]
        print(
            f"[Controller3PtLogger] {event}: "
            f"posture={posture['label']} "
            f"L controller EE rpy={np.round(left_rpy, 1).tolist()} "
            f"R controller EE rpy={np.round(right_rpy, 1).tolist()}"
        )

        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _pose_convention_debug(self, sample: dict) -> dict:
        debug = {}
        for convention, data in sample.get("controller_pose_conventions", {}).items():
            debug[convention] = {
                "left_controller_robot_frame": _pose_frame_debug(
                    data["left_controller_robot_position"],
                    sRot.from_quat(
                        data["left_controller_robot_orientation_wxyz"], scalar_first=True
                    ),
                ),
                "right_controller_robot_frame": _pose_frame_debug(
                    data["right_controller_robot_position"],
                    sRot.from_quat(
                        data["right_controller_robot_orientation_wxyz"], scalar_first=True
                    ),
                ),
                "headset_robot_frame": _pose_frame_debug(
                    data["headset_robot_position"],
                    sRot.from_quat(data["headset_robot_orientation_wxyz"], scalar_first=True),
                ),
                "left_controller_ee_frame": _pose_frame_debug(
                    data["left_controller_robot_position"],
                    sRot.from_quat(
                        data["left_controller_ee_orientation_wxyz"], scalar_first=True
                    ),
                ),
                "right_controller_ee_frame": _pose_frame_debug(
                    data["right_controller_robot_position"],
                    sRot.from_quat(
                        data["right_controller_ee_orientation_wxyz"], scalar_first=True
                    ),
                ),
            }
        return debug

    def _add_target_vs_g1_errors(self, record: dict) -> None:
        errors = {}
        for side in ("left", "right"):
            key = f"{side}_wrist"
            target = record["vr_3pt_calibrated_target"][key]
            g1 = record["g1_ee_fk"][key]
            target_pos = np.array(target["position"], dtype=np.float64)
            g1_pos = np.array(g1["position"], dtype=np.float64)
            target_rot = sRot.from_quat(target["orientation"]["quat_wxyz"], scalar_first=True)
            g1_rot = sRot.from_quat(g1["orientation"]["quat_wxyz"], scalar_first=True)
            rot_error = g1_rot.inv() * target_rot
            errors[key] = {
                "position_delta_target_minus_g1": _as_list(target_pos - g1_pos),
                "orientation_error_g1_inv_target_rpy_xyz_deg": _as_list(
                    rot_error.as_euler("xyz", degrees=True)
                ),
            }
            record["target_vs_g1_error"] = errors


class Controller3PtPostureMarker:
    """Controller-driven labels for calibration log rows."""

    LABELS = (
        "unlabeled",
        "neutral",
        "tilt_left",
        "tilt_right",
        "tilt_forward",
        "tilt_backward",
    )

    def __init__(self):
        self.index = 0
        self.label = self.LABELS[self.index]
        self.updated_realtime = time.time()
        self.updated_monotonic = time.monotonic()
        self.lock = threading.Lock()

    def advance(self) -> dict:
        with self.lock:
            self.index = (self.index + 1) % len(self.LABELS)
            self.label = self.LABELS[self.index]
            self.updated_realtime = time.time()
            self.updated_monotonic = time.monotonic()
            marker = self._snapshot_unlocked()
        print(
            "[Controller3PtMarker] "
            f"posture -> {marker['index']}: {marker['label']}"
        )
        return marker

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict:
        return {
            "index": int(self.index),
            "label": self.label,
            "updated_realtime": float(self.updated_realtime),
            "updated_monotonic": float(self.updated_monotonic),
        }


def _quat_angle_deg_wxyz(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    dot = abs(float(np.dot(q0, q1)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def evaluate_vr3pt_entry_gate(
    candidate_pose: np.ndarray,
    g1_poses: dict,
    wrist_pos_max_m: float = 0.25,
    torso_pos_max_m: float = 0.20,
    wrist_orn_max_deg: float = 45.0,
) -> tuple[bool, str]:
    """Compare calibrated VR_3PT target against current G1 FK."""
    reference_positions = np.vstack(
        [
            g1_poses["left_wrist"]["position"],
            g1_poses["right_wrist"]["position"],
            g1_poses["torso"]["position"],
        ]
    )
    position_errors = np.linalg.norm(candidate_pose[:, :3] - reference_positions, axis=1)
    left_orn_error = _quat_angle_deg_wxyz(
        candidate_pose[0, 3:], g1_poses["left_wrist"]["orientation_wxyz"]
    )
    right_orn_error = _quat_angle_deg_wxyz(
        candidate_pose[1, 3:], g1_poses["right_wrist"]["orientation_wxyz"]
    )

    passed = True
    if wrist_pos_max_m > 0.0:
        passed = passed and bool(position_errors[0] <= wrist_pos_max_m)
        passed = passed and bool(position_errors[1] <= wrist_pos_max_m)
    if torso_pos_max_m > 0.0:
        passed = passed and bool(position_errors[2] <= torso_pos_max_m)
    if wrist_orn_max_deg > 0.0:
        passed = passed and left_orn_error <= wrist_orn_max_deg
        passed = passed and right_orn_error <= wrist_orn_max_deg

    details = (
        f"left_wrist_pos={position_errors[0]:.3f}m/{wrist_pos_max_m:.3f}m, "
        f"right_wrist_pos={position_errors[1]:.3f}m/{wrist_pos_max_m:.3f}m, "
        f"torso_pos={position_errors[2]:.3f}m/{torso_pos_max_m:.3f}m, "
        f"left_wrist_orn={left_orn_error:.1f}deg/{wrist_orn_max_deg:.1f}deg, "
        f"right_wrist_orn={right_orn_error:.1f}deg/{wrist_orn_max_deg:.1f}deg"
    )
    return passed, details


class PlannerStreamer:
    """Encapsulates the planner control loop state and logic."""

    def __init__(
        self,
        socket,
        reader: PicoReader,
        three_point: ThreePointPose,
        poll_hz: int = 20,
        zmq_feedback_host: str = "localhost",
        zmq_feedback_port: int = 5557,
        controller_3pt: bool = False,
        controller_3pt_logger: Controller3PtCalibrationLogger | None = None,
        teleop_ramp_duration: float = 1.0,
        vr3pt_entry_wrist_pos_max_m: float = 0.25,
        vr3pt_entry_torso_pos_max_m: float = 0.20,
        vr3pt_entry_wrist_orn_max_deg: float = 45.0,
    ):
        self.socket = socket
        self.reader = reader
        self.three_point = three_point
        self.controller_3pt = controller_3pt
        self.controller_3pt_logger = controller_3pt_logger
        self.teleop_ramp_duration = max(0.0, float(teleop_ramp_duration))
        self.vr3pt_entry_wrist_pos_max_m = float(vr3pt_entry_wrist_pos_max_m)
        self.vr3pt_entry_torso_pos_max_m = float(vr3pt_entry_torso_pos_max_m)
        self.vr3pt_entry_wrist_orn_max_deg = float(vr3pt_entry_wrist_orn_max_deg)
        self.feedback_reader = FeedbackReader(
            zmq_feedback_host=zmq_feedback_host, zmq_feedback_port=zmq_feedback_port
        )

        self.dt = 1.0 / max(1, poll_hz)
        # Current locomotion mode, default IDLE
        self.mode = LocomotionMode.IDLE
        self.prev_ab = False
        self.prev_xy = False
        # Persistent facing buffer (unit vector on XY plane)
        self.yaw_accumulator = YawAccumulator()
        self.last_send = time.time()
        self.last_xrt_timestamp = None

        # Hand IK solvers for trigger-controlled hand open/close in VR 3PT mode
        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()
        self._vr3pt_ramp_start_pose: np.ndarray | None = None
        self._vr3pt_ramp_start_time: float | None = None
        self._vr3pt_ramp_active = False
        self._last_vr3pt_pose: np.ndarray | None = None
        self.freeze_vr3pt_target_once = False
        self._vr3pt_entry_gate_passed = False
        self._vr3pt_recalibrated_since_gate = False

    def reset_yaw(self):
        """Called when entering planner mode. Resets state for fresh start."""
        self.yaw_accumulator.reset()

    def start_vr3pt_ramp(self, require_recalibration: bool = True):
        """Start a short blend from the first calibrated VR_3PT target to live targets."""
        gate_enabled = (
            self.vr3pt_entry_wrist_pos_max_m > 0.0
            or self.vr3pt_entry_torso_pos_max_m > 0.0
            or self.vr3pt_entry_wrist_orn_max_deg > 0.0
        )
        if gate_enabled and not self._vr3pt_entry_gate_passed:
            raise RuntimeError("VR_3PT ramp requested before entry gate passed")
        if require_recalibration and not self._vr3pt_recalibrated_since_gate:
            raise RuntimeError("VR_3PT ramp requested before post-gate recalibration")
        if self.teleop_ramp_duration <= 0.0:
            self._vr3pt_ramp_active = False
            self._vr3pt_ramp_start_pose = None
            self._vr3pt_ramp_start_time = None
            self._vr3pt_entry_gate_passed = False
            self._vr3pt_recalibrated_since_gate = False
            return
        self._vr3pt_ramp_active = True
        self._vr3pt_ramp_start_pose = None
        self._vr3pt_ramp_start_time = None
        self._vr3pt_entry_gate_passed = False
        self._vr3pt_recalibrated_since_gate = False
        print(f"[PlannerLoop] VR_3PT ramp started ({self.teleop_ramp_duration:.2f}s)")

    def cancel_vr3pt_ramp(self):
        self._vr3pt_ramp_active = False
        self._vr3pt_ramp_start_pose = None
        self._vr3pt_ramp_start_time = None

    def _apply_vr3pt_ramp(self, vr_3pt_pose: np.ndarray) -> np.ndarray:
        if not self._vr3pt_ramp_active:
            return vr_3pt_pose

        now = time.monotonic()
        if self._vr3pt_ramp_start_pose is None:
            self._vr3pt_ramp_start_pose = np.asarray(vr_3pt_pose, dtype=np.float32).copy()
            self._vr3pt_ramp_start_time = now
            return self._vr3pt_ramp_start_pose.copy()

        elapsed = now - (self._vr3pt_ramp_start_time or now)
        alpha = np.clip(elapsed / self.teleop_ramp_duration, 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        if alpha >= 1.0:
            self.cancel_vr3pt_ramp()
            print("[PlannerLoop] VR_3PT ramp complete")
            return vr_3pt_pose
        return _blend_vr_3pt_pose(self._vr3pt_ramp_start_pose, vr_3pt_pose, alpha)

    def save_upper_body_position_target(self):
        """Poll feedback and save upper body position target."""
        self.feedback_reader.poll_feedback()

    def recalibrate_for_vr3pt(self):
        """
        Recalibrate VR 3-point pose tracking using the robot's current measured joints.

        Polls the g1_debug feedback to get the robot's actual joint state, then
        schedules recalibration so VR tracking aligns with the robot's current pose.
        This prevents sudden jumps when entering VR 3PT mode from PLANNER mode.
        """
        self.feedback_reader.poll_feedback()
        if self.feedback_reader.full_body_q_measured is not None:
            self.three_point.reset_with_measured_q(self.feedback_reader.full_body_q_measured)
            self._vr3pt_recalibrated_since_gate = True
            print("[PlannerLoop] VR 3PT recalibration scheduled with measured robot pose")
        else:
            # Fallback: use zeros if no feedback available
            print(
                "[PlannerLoop] WARNING: No feedback data for VR 3PT recalibration, "
                "using zero body_q as fallback"
            )
            self.three_point.reset_with_measured_q(np.zeros(29, dtype=np.float64))
            self._vr3pt_recalibrated_since_gate = True

    def check_vr3pt_entry_mismatch(self) -> bool:
        """Refuse VR_3PT if calibrated operator target is far from current robot FK."""
        gate_enabled = (
            self.vr3pt_entry_wrist_pos_max_m > 0.0
            or self.vr3pt_entry_torso_pos_max_m > 0.0
            or self.vr3pt_entry_wrist_orn_max_deg > 0.0
        )
        if not gate_enabled:
            print("[Manager] WARNING: VR_3PT entry gate disabled by thresholds <= 0")
            self._vr3pt_entry_gate_passed = True
            self._vr3pt_recalibrated_since_gate = False
            return True
        if self.three_point._robot_model is None or get_g1_key_frame_poses is None:
            print("[Manager] Refusing VR_3PT: robot model unavailable for entry gate")
            return False
        if not self.three_point.is_calibrated:
            print("[Manager] Refusing VR_3PT: 3-point calibration is not initialized")
            return False

        sample = self.reader.get_latest()
        if sample is None:
            print("[Manager] Refusing VR_3PT: no current Pico sample for entry gate")
            return False

        self.feedback_reader.poll_feedback()
        body_q = self.feedback_reader.full_body_q_measured
        if body_q is None:
            print("[Manager] Refusing VR_3PT: no robot feedback for entry gate")
            return False

        if self.controller_3pt:
            raw_vr_3pt_pose = sample["vr_3pt_pose_np"]
        else:
            raw_vr_3pt_pose = _process_3pt_pose(sample["body_poses_np"])
        candidate_pose = self.three_point._apply_calibration(raw_vr_3pt_pose)

        robot_q = self.three_point._robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_q[:29]
        )
        g1_poses = get_g1_key_frame_poses(self.three_point._robot_model, q=robot_q)
        passed, details = evaluate_vr3pt_entry_gate(
            candidate_pose,
            g1_poses,
            wrist_pos_max_m=self.vr3pt_entry_wrist_pos_max_m,
            torso_pos_max_m=self.vr3pt_entry_torso_pos_max_m,
            wrist_orn_max_deg=self.vr3pt_entry_wrist_orn_max_deg,
        )
        if not passed:
            self._vr3pt_entry_gate_passed = False
            self._vr3pt_recalibrated_since_gate = False
            print(
                "[Manager] Refusing VR_3PT: calibrated target mismatches current robot FK "
                f"({details})"
            )
            return False

        print(f"[Manager] VR_3PT entry gate passed ({details})")
        self._vr3pt_entry_gate_passed = True
        self._vr3pt_recalibrated_since_gate = False
        return True

    def run_once(self, stream_mode: StreamMode):
        """Execute one iteration of the planner control loop."""
        try:
            # Avoid sending old commands if XRT timestamp hasn't advanced, in case of headset disconnect
            if self.controller_3pt:
                latest_sample = self.reader.get_latest()
                if latest_sample is None:
                    return
                xrt_timestamp = latest_sample.get("timestamp_ns")
            else:
                latest_sample = None
                xrt_timestamp = xrt.get_time_stamp_ns()
            duplicate_xr_sample = xrt_timestamp == self.last_xrt_timestamp
            can_freeze_vr3pt = (
                stream_mode == StreamMode.PLANNER_VR_3PT
                and self.freeze_vr3pt_target_once
                and self._last_vr3pt_pose is not None
            )
            if duplicate_xr_sample and not can_freeze_vr3pt:
                return
            self.last_xrt_timestamp = xrt_timestamp

            # A+B => next mode; X+Y => previous mode (rising edges)
            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()
            ab_now = bool(a_pressed) and bool(b_pressed) and not bool(x_pressed) and not bool(y_pressed)
            xy_now = bool(x_pressed) and bool(y_pressed) and not bool(a_pressed) and not bool(b_pressed)
            if ab_now and not self.prev_ab:
                self.mode = LocomotionMode(min(LocomotionMode.INJURED_WALK, self.mode + 1))
                print(f"[PlannerLoop] Mode -> {self.mode.value}: {self.mode.name}")
            if xy_now and not self.prev_xy:
                self.mode = LocomotionMode(max(LocomotionMode.IDLE, self.mode - 1))
                print(f"[PlannerLoop] Mode -> {self.mode.value}: {self.mode.name}")
            self.prev_ab = ab_now
            self.prev_xy = xy_now

            # Read axes/joysticks to control movement, facing, speed and mode
            lx, ly, rx, ry = get_controller_axes()

            # Facing from RIGHT stick: continuous yaw based on rx (right = turn right, left = turn left)
            facing = self.yaw_accumulator.update(rx, self.dt)

            raw_mag = np.hypot(lx, ly)
            raw_mag = np.clip(raw_mag, 0.0, 1.0)
            if np.abs(raw_mag) < JOYSTICK_DEADZONE:
                mag = 0.0
                speed = -1.0
                mode_to_send = LocomotionMode.IDLE
            else:
                mag = (raw_mag - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
                if mag > 1.0:
                    mag = 1.0
                mode_to_send = self.mode

                if self.mode == LocomotionMode.SLOW_WALK:
                    speed = 0.1 + 0.5 * mag  # 0.1 .. 0.6
                elif self.mode == LocomotionMode.WALK:
                    speed = -1.0
                elif self.mode == LocomotionMode.RUN:
                    speed = 1.5 + 3 * mag  # 1.5 .. 4.5
                else:
                    speed = mag  # default 0 .. 1.0

            denom = raw_mag if raw_mag > 0.0 else 1.0
            scale = mag / denom
            movement_local = np.array([-lx, ly]) * scale
            perp_x, perp_y = -facing[1], facing[0]
            rotation_facing = np.array([[perp_x, perp_y], [facing[0], facing[1]]])
            movement_global = rotation_facing @ movement_local

            movement = [movement_global[0], movement_global[1], 0.0]

            upper_body_position = None
            left_hand_position = None
            right_hand_position = None
            if stream_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                upper_body_position = self.feedback_reader.upper_body_position_target
                left_hand_position = self.feedback_reader.left_hand_position_target
                right_hand_position = self.feedback_reader.right_hand_position_target

            vr_3pt_position = None
            vr_3pt_orientation = None
            vr_3pt_compliance = None
            if stream_mode == StreamMode.PLANNER_VR_3PT:
                vr_3pt_pose = None
                sample = latest_sample if latest_sample is not None else self.reader.get_latest()
                if self.freeze_vr3pt_target_once and self._last_vr3pt_pose is not None:
                    vr_3pt_pose = self._last_vr3pt_pose.copy()
                    self.freeze_vr3pt_target_once = False
                    print("[PlannerLoop] XR stale warning: re-sending last VR 3-point target")
                elif sample is not None:
                    print("[PlannerLoop] Sending VR 3-point pose as target")
                    if self.controller_3pt:
                        raw_vr_3pt_pose = sample["vr_3pt_pose_np"]
                        vr_3pt_pose = self.three_point.process_vr_3pt_pose(
                            raw_vr_3pt_pose
                        )
                        if self.controller_3pt_logger is not None:
                            self.controller_3pt_logger.maybe_log(
                                "vr3pt_stream",
                                sample,
                                raw_vr_3pt_pose,
                                vr_3pt_pose,
                                self.three_point,
                                self.feedback_reader.full_body_q_measured,
                            )
                    else:
                        vr_3pt_pose = self.three_point.process_smpl_pose(sample["body_poses_np"])
                    vr_3pt_pose = self._apply_vr3pt_ramp(vr_3pt_pose)
                    self._last_vr3pt_pose = np.asarray(vr_3pt_pose, dtype=np.float32).copy()
                    self.freeze_vr3pt_target_once = False
                if vr_3pt_pose is not None:
                    vr_3pt_position = (vr_3pt_pose[:, :3].flatten()).tolist()
                    vr_3pt_orientation = vr_3pt_pose[:, 3:].flatten().tolist()

                # Compute hand joints from trigger/grip inputs so operator can
                # control hand open/close while in VR 3PT mode
                (
                    left_menu_button,
                    left_trigger,
                    right_trigger,
                    left_grip,
                    right_grip,
                ) = get_controller_inputs()
                lh_joints, rh_joints = compute_hand_joints_from_inputs(
                    self.left_hand_ik_solver,
                    self.right_hand_ik_solver,
                    left_trigger,
                    left_grip,
                    right_trigger,
                    right_grip,
                )
                left_hand_position = lh_joints.reshape(-1).astype(np.float32).tolist()
                right_hand_position = rh_joints.reshape(-1).astype(np.float32).tolist()

            msg = build_planner_message(
                mode_to_send.value,
                movement,
                facing,
                speed=speed,
                height=-1.0,
                upper_body_position=upper_body_position,
                left_hand_position=left_hand_position,
                right_hand_position=right_hand_position,
                vr_3pt_position=vr_3pt_position,
                vr_3pt_orientation=vr_3pt_orientation,
                vr_3pt_compliance=vr_3pt_compliance,
            )
            self.socket.send(msg)
        except Exception as e:
            import traceback

            print(f"[PlannerLoop] error: {e}")
            traceback.print_exc()
            raise

        # pacing
        now = time.time()
        sleep_t = self.dt - (now - self.last_send)
        if sleep_t > 0:
            time.sleep(sleep_t)
        self.last_send = time.time()


def run_pico_manager(
    port: int = 5556,
    buffer_size: int = 15,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    zmq_feedback_host: str = "localhost",
    zmq_feedback_port: int = 5557,
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
    controller_3pt: bool = False,
    left_controller_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    right_controller_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    controller_pose_convention: str = "xrobotoolkit_unity",
    headset_pose_convention: str = "xrobotoolkit_unity",
    headset_orientation_convention: str | None = None,
    no_vr3pt_recalib_on_switch: bool = False,
    teleop_ramp_duration: float = 1.0,
    vr3pt_entry_wrist_pos_max_m: float = 0.25,
    vr3pt_entry_torso_pos_max_m: float = 0.20,
    vr3pt_entry_wrist_orn_max_deg: float = 45.0,
    controller_3pt_log_dir: str = "",
    controller_3pt_log_interval: float = 1.0,
    disable_xr_staleness_watchdog: bool = False,
):
    """
    Manager: creates shared PUB socket and runs pose/planner streamers based on current mode.
    Controller input:
      A+X: Toggle between planner and pose mode
      A+B+X+Y: Toggle policy start/stop
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run the manager."
        )
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    if controller_3pt:
        print("Controller 3PT mode enabled: using headset + two controller poses only.")
        print("Full-body POSE mode is disabled; use PLANNER and VR_3PT modes.")
        print(
            "Controller EE offsets (deg): "
            f"left={left_controller_offset_rpy}, right={right_controller_offset_rpy}"
        )
        print(f"Controller pose convention: {controller_pose_convention}")
        print(f"Headset pose convention: {headset_pose_convention}")
        headset_orientation_convention = (
            headset_orientation_convention or headset_pose_convention
        )
        print(f"Headset orientation convention: {headset_orientation_convention}")
        if no_vr3pt_recalib_on_switch:
            print(
                "WARNING: VR_3PT per-switch recalibration disabled. "
                "Use only for testing, not real robot teleop."
            )
        if (
            vr3pt_entry_wrist_pos_max_m <= 0.0
            and vr3pt_entry_torso_pos_max_m <= 0.0
            and vr3pt_entry_wrist_orn_max_deg <= 0.0
        ):
            print("WARNING: VR_3PT entry gate disabled by thresholds <= 0.")
        if not controller_3pt_log_dir:
            controller_3pt_log_dir = os.path.join("outputs", "controller_3pt_calibration")
    else:
        print("Waiting for body tracking data...")
        while not xrt.is_body_data_available():
            print("waiting for body data...")
            time.sleep(1)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"[Manager] ZMQ socket bound to port {port}")

    # Print available locomotion modes
    try:
        print("[Manager] Available modes:")
        for mode in LocomotionMode:
            print(f"  {mode.value}: {mode.name}")
    except Exception:
        pass

    # Create shared reader and 3-point pose processor
    reader = (
        ControllerPoseReader(
            max_queue_size=buffer_size,
            left_controller_offset_rpy=left_controller_offset_rpy,
            right_controller_offset_rpy=right_controller_offset_rpy,
            controller_pose_convention=controller_pose_convention,
            headset_pose_convention=headset_pose_convention,
            headset_orientation_convention=headset_orientation_convention,
        )
        if controller_3pt
        else PicoReader(max_queue_size=buffer_size)
    )
    reader.start()

    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix="PoseLoop",
    )
    posture_marker = Controller3PtPostureMarker() if controller_3pt else None
    controller_3pt_logger = (
        Controller3PtCalibrationLogger(
            controller_3pt_log_dir,
            controller_3pt_log_interval,
            posture_marker=posture_marker,
        )
        if controller_3pt
        else None
    )

    pose_streamer = PoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        log_prefix="PoseLoop",
    )
    planner_streamer = PlannerStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        poll_hz=20,
        zmq_feedback_host=zmq_feedback_host,
        zmq_feedback_port=zmq_feedback_port,
        controller_3pt=controller_3pt,
        controller_3pt_logger=controller_3pt_logger,
        teleop_ramp_duration=teleop_ramp_duration,
        vr3pt_entry_wrist_pos_max_m=vr3pt_entry_wrist_pos_max_m,
        vr3pt_entry_torso_pos_max_m=vr3pt_entry_torso_pos_max_m,
        vr3pt_entry_wrist_orn_max_deg=vr3pt_entry_wrist_orn_max_deg,
    )

    # State machine diagram:
    #
    #   Chain 1 (by_pressed enters/exits, left_axis_click toggles sub-mode):
    #     POSE <--(by)--> PLANNER_FROZEN_UPPER_BODY <--(left_axis_click)--> PLANNER_VR_3PT
    #                                          ▲                              |
    #                                          └───────────────(by)───────────┘
    #
    #   Chain 2 (ax_pressed enters/exits, left_axis_click toggles sub-mode):
    #     POSE <--(ax)--> PLANNER <--(left_axis_click)--> PLANNER_VR_3PT
    #                                                        |
    #                                                   (ax)--> POSE
    #
    #   Emergency stop from any mode: A+B+X+Y (start_combo) --> OFF
    #   POSE_PAUSE: left_menu_button held --> POSE_PAUSE, released --> POSE
    #
    if controller_3pt:
        print(
            "Manager controls: A+B+X+Y=start/stop, "
            "Left Stick Click=toggle VR_3PT upper-body teleop, "
            "B+Y=freeze upper body, "
            "Right Stick Click=advance calibration posture marker"
        )
        print(
            "Posture marker sequence: "
            + " -> ".join(Controller3PtPostureMarker.LABELS)
        )
    else:
        print("Manager controls: A+X=toggle mode, B+Y=freeze upper body, A+B+X+Y=start/stop policy")
    current_mode = StreamMode.OFF
    # Track which mode VR_3PT was entered from, so left_axis_click returns to it.
    # Will be either PLANNER or PLANNER_FROZEN_UPPER_BODY.
    vr3pt_parent_mode = StreamMode.PLANNER
    prev_toggle_dc = False
    prev_toggle_da = False
    xr_watchdog = XRStalenessWatchdog()
    if disable_xr_staleness_watchdog:
        print("WARNING: XR staleness watchdog disabled.")
    try:
        prev_ax_pressed = False
        prev_by_pressed = False
        prev_start_combo = False
        prev_left_axis_click = False
        prev_right_axis_click = False
        while True:
            # Poll Pico controller for buttons/axes
            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()

            left_menu_button, _, _, left_grip_mgr, _ = get_controller_inputs()

            left_axis_click, right_axis_click = get_axis_clicks()

            # Rising edge: A+X pressed together -> toggle POSE/PLANNER mode
            ax_pressed = bool(a_pressed) and bool(x_pressed) and not bool(b_pressed) and not bool(y_pressed)

            # Rising edge: B+Y pressed together -> toggle POSE/PLANNER_FROZEN_UPPER_BODY mode
            by_pressed = bool(b_pressed) and bool(y_pressed) and not bool(a_pressed) and not bool(x_pressed)

            # Rising edge: A+B+X+Y pressed together -> toggle policy start/stop (planner=True)
            start_combo = bool(a_pressed) and bool(b_pressed) and bool(x_pressed) and bool(y_pressed)

            if not disable_xr_staleness_watchdog:
                xr_state = xr_watchdog.poll(
                    reader.get_last_sample_monotonic_ns()
                    if controller_3pt and hasattr(reader, "get_last_sample_monotonic_ns")
                    else None,
                    time.monotonic_ns(),
                    active=controller_3pt and current_mode == StreamMode.PLANNER_VR_3PT,
                )
                if xr_state == "warn":
                    planner_streamer.freeze_vr3pt_target_once = True
                elif xr_state == "estop":
                    print("[Manager] XR staleness E-STOP: stopping policy and exiting")
                    socket.send(build_command_message(start=False, stop=True, planner=True))
                    exit()

            if (
                controller_3pt
                and right_axis_click
                and not prev_right_axis_click
                and posture_marker is not None
            ):
                posture_marker.advance()
                sample = reader.get_latest()
                if sample is not None and controller_3pt_logger is not None:
                    raw_vr_3pt_pose = sample["vr_3pt_pose_np"]
                    controller_3pt_logger.log(
                        "posture_marker",
                        sample,
                        raw_vr_3pt_pose,
                        three_point.process_vr_3pt_pose(raw_vr_3pt_pose),
                        three_point,
                        planner_streamer.feedback_reader.full_body_q_measured,
                    )

            new_mode = current_mode
            if current_mode == StreamMode.OFF:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.PLANNER
                    # Calibrate VR 3pt tracking NOW: operator should be in zero-ref pose.
                    # Uses the current Pico frame + FK of all-zero body joints.
                    sample = reader.get_latest()
                    if sample is not None:
                        if controller_3pt:
                            three_point.calibrate_vr_3pt_now(sample["vr_3pt_pose_np"])
                            if controller_3pt_logger is not None:
                                controller_3pt_logger.log(
                                    "initial_calibration",
                                    sample,
                                    sample["vr_3pt_pose_np"],
                                    three_point.process_vr_3pt_pose(sample["vr_3pt_pose_np"]),
                                    three_point,
                                    None,
                                )
                        else:
                            three_point.calibrate_now(sample["body_poses_np"])
                    else:
                        print("[Manager] WARNING: No Pico data available for calibration")

            elif current_mode == StreamMode.PLANNER:
                # Chain 2: POSE <--(ax)--> PLANNER <--(left_axis_click)--> VR_3PT
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.POSE
                elif left_axis_click and not prev_left_axis_click:
                    new_mode = StreamMode.PLANNER_VR_3PT

            elif current_mode == StreamMode.POSE:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.PLANNER  # Enter chain 2
                elif by_pressed and not prev_by_pressed:
                    new_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY  # Enter chain 1
                elif left_menu_button:
                    new_mode = StreamMode.POSE_PAUSE

            elif current_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                # Chain 1: POSE <--(by)--> FROZEN <--(left_axis_click)--> VR_3PT
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif by_pressed and not prev_by_pressed:
                    new_mode = StreamMode.POSE
                elif left_axis_click and not prev_left_axis_click:
                    new_mode = StreamMode.PLANNER_VR_3PT

            elif current_mode == StreamMode.POSE_PAUSE:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif not left_menu_button:
                    new_mode = StreamMode.POSE

            elif current_mode == StreamMode.PLANNER_VR_3PT:
                # VR_3PT is reachable from both chains:
                #   left_axis_click → return to parent (PLANNER or FROZEN)
                #   ax_pressed      → POSE (chain 2 exit)
                #   by_pressed      → freeze current upper body
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif left_axis_click and not prev_left_axis_click:
                    new_mode = vr3pt_parent_mode  # Return to parent mode
                elif ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.POSE
                elif by_pressed and not prev_by_pressed:
                    new_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY

            # Handle mode transitions before running loop
            if new_mode != current_mode:
                if controller_3pt and new_mode == StreamMode.POSE:
                    if current_mode == StreamMode.PLANNER_VR_3PT:
                        new_mode = vr3pt_parent_mode
                    else:
                        new_mode = StreamMode.PLANNER_VR_3PT
                    print("[Manager] Controller 3PT mode remapped POSE request to VR_3PT")

                if new_mode == StreamMode.PLANNER_VR_3PT:
                    if not planner_streamer.check_vr3pt_entry_mismatch():
                        new_mode = current_mode

                if new_mode != current_mode:
                    if current_mode == StreamMode.POSE:
                        pose_streamer.on_mode_exit()

                    if current_mode == StreamMode.PLANNER_VR_3PT:
                        planner_streamer.cancel_vr3pt_ramp()

                    # Track parent when entering VR_3PT
                    if new_mode == StreamMode.PLANNER_VR_3PT:
                        vr3pt_parent_mode = current_mode
                        print(f"[Manager] VR_3PT parent: {vr3pt_parent_mode.name}")

                    if new_mode == StreamMode.POSE:
                        pose_streamer.reset_yaw()
                    elif new_mode == StreamMode.PLANNER and current_mode != StreamMode.PLANNER_VR_3PT:
                        # Only reset yaw when freshly entering PLANNER from POSE,
                        # not when returning from VR_3PT sub-mode
                        planner_streamer.reset_yaw()
                    elif new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                        if current_mode != StreamMode.PLANNER_VR_3PT:
                            # Freshly entering from POSE: reset yaw and grab initial targets
                            planner_streamer.reset_yaw()
                        # Always re-grab the latest robot state as frozen targets,
                        # whether entering from POSE or returning from VR_3PT
                        # (the old targets are stale after VR_3PT moved the arms)
                        planner_streamer.save_upper_body_position_target()
                    elif new_mode == StreamMode.PLANNER_VR_3PT:
                        # Recalibrate VR tracking against the robot's actual current pose
                        # (read via g1_debug feedback + FK) to prevent sudden jumps
                        if no_vr3pt_recalib_on_switch:
                            print("[Manager] Skipping VR_3PT per-switch recalibration")
                        else:
                            planner_streamer.recalibrate_for_vr3pt()
                        planner_streamer.start_vr3pt_ramp(
                            require_recalibration=not no_vr3pt_recalib_on_switch
                        )

            # Run one iteration of the new mode
            if new_mode == StreamMode.POSE:
                pose_streamer.run_once()
            elif (
                new_mode == StreamMode.PLANNER
                or new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
                or new_mode == StreamMode.PLANNER_VR_3PT
            ):
                planner_streamer.run_once(new_mode)

            # Make sure to send command messages after loop iteration to ensure data arrives before mode switch
            if new_mode != current_mode:
                if new_mode == StreamMode.OFF:
                    socket.send(build_command_message(start=False, stop=True, planner=True))
                    exit()
                elif (
                    new_mode == StreamMode.PLANNER
                    or new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
                    or new_mode == StreamMode.PLANNER_VR_3PT
                ):
                    socket.send(build_command_message(start=True, stop=False, planner=True))
                elif new_mode == StreamMode.POSE:
                    socket.send(build_command_message(start=True, stop=False, planner=False))

                print(f"[Manager] StreamMode switch: {current_mode.name} -> {new_mode.name}")
                current_mode = new_mode

            # Mode-independent: send manager_state for data exporter
            toggle_dc_tmp = bool(a_pressed) and left_grip_mgr > 0.5
            toggle_da_tmp = bool(b_pressed) and left_grip_mgr > 0.5
            toggle_dc = toggle_dc_tmp and not prev_toggle_dc
            toggle_da = toggle_da_tmp and not prev_toggle_da
            prev_toggle_dc = toggle_dc_tmp
            prev_toggle_da = toggle_da_tmp
            socket.send(
                pack_pose_message(
                    {
                        "stream_mode": np.array([current_mode.value], dtype=np.int32),
                        "toggle_data_collection": np.array([toggle_dc], dtype=bool),
                        "toggle_data_abort": np.array([toggle_da], dtype=bool),
                    },
                    topic="manager_state",
                )
            )

            prev_ax_pressed = ax_pressed
            prev_by_pressed = by_pressed
            prev_start_combo = start_combo
            prev_left_axis_click = left_axis_click
            prev_right_axis_click = right_axis_click

    except KeyboardInterrupt:
        print("\nStopping manager...")
    finally:
        # Cleanup resources
        reader.stop()
        three_point.close()
        socket.close()
        context.term()
        print("[Manager] Shutdown complete")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=15, help="Sliding window buffer size")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ server port (default: 5556)")
    parser.add_argument(
        "--num_frames_to_send", type=int, default=5, help="Number of frames to send (default: 200)"
    )
    parser.add_argument("--target_fps", type=int, default=50, help="Target loop FPS (default: 50)")
    parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA for tensors and model (default: CPU)"
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default="",
        help="Directory to save sent batches (default: disabled)",
    )
    parser.add_argument(
        "--record_format",
        type=str,
        default="npz",
        help="Recording format: 'npz' or 'bin' (default: npz)",
    )
    parser.add_argument(
        "--manager",
        action="store_true",
        help="Run manager with planner and pose threads (interactive)",
    )
    parser.add_argument(
        "--zmq_feedback_host",
        type=str,
        default="localhost",
        help="ZMQ feedback host (default: localhost)",
    )
    parser.add_argument(
        "--zmq_feedback_port",
        type=int,
        default=5557,
        help="ZMQ feedback port (default: 5557)",
    )
    parser.add_argument(
        "--vr3pt_test",
        action="store_true",
        help="Run VR 3-point pose visualizer test (reference frames only)",
    )
    parser.add_argument(
        "--vr3pt_live",
        action="store_true",
        help="Capture one frame of VR 3-point pose and visualize with reference frames",
    )
    parser.add_argument(
        "--vr3pt_realtime",
        action="store_true",
        help="Run standalone real-time VR 3-point pose visualizer",
    )
    parser.add_argument(
        "--vis_vr3pt",
        action="store_true",
        help="Enable inline VR 3-point pose visualization in pose streaming mode",
    )
    parser.add_argument(
        "--vr3pt_hz",
        type=int,
        default=10,
        help="Update rate for real-time VR visualization in Hz (default: 10)",
    )
    parser.add_argument(
        "--no_g1",
        action="store_true",
        help="Disable G1 robot visualization in VR 3pt pose view (G1 is shown by default)",
    )
    parser.add_argument(
        "--waist_tracking",
        action="store_true",
        help="Enable G1 robot waist to follow VR head orientation (disabled by default for performance)",
    )
    parser.add_argument(
        "--vis_smpl",
        action="store_true",
        help="Enable SMPL body joint visualization (24 joint spheres) in the VR3pt viewer",
    )
    parser.add_argument(
        "--controller_3pt",
        action="store_true",
        help=(
            "Use only PICO headset + two controller poses for VR_3PT upper-body teleop. "
            "This does not require ankle trackers, but full-body POSE mode is disabled."
        ),
    )
    parser.add_argument(
        "--left_controller_offset_rpy",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help=(
            "Extra left controller-to-EE orientation offset in degrees, applied as "
            "extrinsic xyz roll/pitch/yaw before VR_3PT calibration."
        ),
    )
    parser.add_argument(
        "--right_controller_offset_rpy",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help=(
            "Extra right controller-to-EE orientation offset in degrees, applied as "
            "extrinsic xyz roll/pitch/yaw before VR_3PT calibration."
        ),
    )
    parser.add_argument(
        "--controller_pose_convention",
        type=str,
        choices=CONTROLLER_POSE_CONVENTIONS,
        default="xrobotoolkit_unity",
        help=(
            "Base coordinate transform for headset/controller poses. "
            "xrobotoolkit_unity is the previous behavior; openxr_unitree matches "
            "Unitree televuer's OpenXR-to-robot basis transform."
        ),
    )
    parser.add_argument(
        "--headset_pose_convention",
        type=str,
        choices=CONTROLLER_POSE_CONVENTIONS,
        default="xrobotoolkit_unity",
        help=(
            "Base coordinate transform for headset pose. Keep xrobotoolkit_unity for "
            "GR00T upper-body/head tracking; controller_pose_convention can be "
            "openxr_unitree independently for hand controller axes."
        ),
    )
    parser.add_argument(
        "--headset_orientation_convention",
        type=str,
        choices=CONTROLLER_POSE_CONVENTIONS,
        default=None,
        help=(
            "Base coordinate transform for headset orientation. Defaults to "
            "--headset_pose_convention. Use openxr_unitree if headset forward/back tilt "
            "is mapped to robot side lean."
        ),
    )
    parser.add_argument(
        "--no_vr3pt_recalib_on_switch",
        action="store_true",
        help=(
            "Do not recalibrate VR_3PT when entering it with Left Stick Click. "
            "Useful for testing wrist axis alignment after initial CALIB_FULL calibration."
        ),
    )
    parser.add_argument(
        "--teleop_ramp_duration",
        type=float,
        default=1.0,
        help=(
            "Seconds to ramp VR_3PT teleop targets after entering teleoperation mode. "
            "Set to 0 to disable ramping."
        ),
    )
    parser.add_argument(
        "--vr3pt_entry_max_mismatch",
        type=float,
        default=None,
        help=(
            "Deprecated alias for --vr3pt_entry_wrist_pos_max_m. "
            "Kept for old commands."
        ),
    )
    parser.add_argument(
        "--vr3pt_entry_wrist_pos_max_m",
        type=float,
        default=0.25,
        help="Maximum wrist position mismatch in meters when entering VR_3PT.",
    )
    parser.add_argument(
        "--vr3pt_entry_torso_pos_max_m",
        type=float,
        default=0.20,
        help="Maximum torso/head-point position mismatch in meters when entering VR_3PT.",
    )
    parser.add_argument(
        "--vr3pt_entry_wrist_orn_max_deg",
        type=float,
        default=45.0,
        help="Maximum wrist orientation mismatch in degrees when entering VR_3PT.",
    )
    parser.add_argument(
        "--controller_3pt_log_dir",
        type=str,
        default="",
        help=(
            "Directory for controller-only calibration JSONL logs. Defaults to "
            "outputs/controller_3pt_calibration when --controller_3pt is used."
        ),
    )
    parser.add_argument(
        "--controller_3pt_log_interval",
        type=float,
        default=1.0,
        help="Seconds between saved controller/G1 frame logs while in VR_3PT.",
    )
    parser.add_argument(
        "--disable_xr_staleness_watchdog",
        action="store_true",
        help="Disable manager-side XR stale-frame stop/freeze checks. Use only for harnessed diagnostics.",
    )
    args = parser.parse_args()

    # Standalone VR3Pt test modes (exit after finishing)
    if args.vr3pt_test:
        print("Running VR 3-point pose visualizer test...")
        run_vr3pt_visualizer_test()
        print("VR 3-point pose visualizer test completed")
        exit(0)

    if args.vr3pt_live:
        print("Running VR 3-point pose live capture...")
        run_vr3pt_live_visualizer()
        print("VR 3-point pose live visualizer completed")
        exit(0)

    if args.vr3pt_realtime:
        print("Running VR 3-point pose real-time visualizer...")
        run_vr3pt_realtime_visualizer(update_hz=args.vr3pt_hz)
        print("VR 3-point pose real-time visualizer completed")
        exit(0)

    # Main execution modes
    # G1 robot visualization is enabled by default when vis_vr3pt is used
    with_g1_robot = not args.no_g1

    if args.manager:
        run_pico_manager(
            port=args.port,
            buffer_size=args.buffer_size,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            zmq_feedback_host=args.zmq_feedback_host,
            zmq_feedback_port=args.zmq_feedback_port,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
            controller_3pt=args.controller_3pt,
            left_controller_offset_rpy=tuple(args.left_controller_offset_rpy),
            right_controller_offset_rpy=tuple(args.right_controller_offset_rpy),
            controller_pose_convention=args.controller_pose_convention,
            headset_pose_convention=args.headset_pose_convention,
            headset_orientation_convention=args.headset_orientation_convention,
            no_vr3pt_recalib_on_switch=args.no_vr3pt_recalib_on_switch,
            teleop_ramp_duration=args.teleop_ramp_duration,
            vr3pt_entry_wrist_pos_max_m=(
                args.vr3pt_entry_max_mismatch
                if args.vr3pt_entry_max_mismatch is not None
                else args.vr3pt_entry_wrist_pos_max_m
            ),
            vr3pt_entry_torso_pos_max_m=args.vr3pt_entry_torso_pos_max_m,
            vr3pt_entry_wrist_orn_max_deg=args.vr3pt_entry_wrist_orn_max_deg,
            controller_3pt_log_dir=args.controller_3pt_log_dir,
            controller_3pt_log_interval=args.controller_3pt_log_interval,
            disable_xr_staleness_watchdog=args.disable_xr_staleness_watchdog,
        )
    else:
        # Run legacy single-thread pose streaming
        run_pico(
            buffer_size=args.buffer_size,
            port=args.port,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
        )
