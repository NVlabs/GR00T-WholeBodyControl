# Pico full-body + BrainCo hand stream server

"""

# The server publishes only the ``pose`` topic consumed by Gear SONIC's plain
# ZMQ endpoint. Planner, manager-state and VR 3-point upper-body control are
# intentionally not part of this entry point.

    python "gear_sonic/scripts/pico_manager_thread_server_brainco _hand.py" \
        --port 5556 --brainco_hand_speed 1.0

"""

from collections import defaultdict, deque
import os
import subprocess
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation as sRot
import torch
import zmq

from gear_sonic.utils.teleop import input_readers
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
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
except ImportError:

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
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
except ImportError as exc:
    ChannelFactoryInitialize = None
    ChannelPublisher = None
    MotorCmds_ = None
    unitree_go_msg_dds__MotorCmd_ = None
    _BRAINCO_DDS_IMPORT_ERROR = exc
else:
    _BRAINCO_DDS_IMPORT_ERROR = None

# Full-body SMPL processing

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


BRAINCO_NUM_MOTORS = 6
BRAINCO_LEFT_COMMAND_TOPIC = "rt/brainco/left/cmd"
BRAINCO_RIGHT_COMMAND_TOPIC = "rt/brainco/right/cmd"
BRAINCO_DEFAULT_HAND_SPEED = 1.0

# XR_EXT_hand_tracking joint order used by Pico/XRoboToolkit (26 joints).
# BrainCo command order: thumb flexion, thumb opposition, index, middle, ring, pinky.
_PICO_THUMB_JOINTS = (2, 3, 4, 5)
_PICO_FINGER_JOINTS = (
    (6, 7, 8, 9, 10),
    (11, 12, 13, 14, 15),
    (16, 17, 18, 19, 20),
    (21, 22, 23, 24, 25),
)
_PICO_INDEX_METACARPAL = 6
_PICO_PINKY_METACARPAL = 21


def _chain_flexion(
    positions: np.ndarray, indices: tuple[int, ...], max_angle: float
) -> float:
    """Return normalized accumulated bend of a tracked finger chain."""
    points = positions[np.asarray(indices)]
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths < 1e-6):
        return 0.0
    segments /= lengths[:, None]
    bends = np.arccos(
        np.clip(np.sum(segments[:-1] * segments[1:], axis=1), -1.0, 1.0)
    )
    return float(np.clip(np.sum(bends) / max_angle, 0.0, 1.0))


def compute_brainco_hand_target(hand_joints: np.ndarray) -> np.ndarray:
    """Retarget one Pico hand (26x7 poses) to BrainCo's six normalized motors.

    Flexion is derived from the accumulated angles of each finger skeleton, so
    it is independent of wrist pose and hand size.  The thumb's second axis is
    its opposition across the palm, estimated from the thumb-tip distance to
    the pinky base and normalized by palm width.
    """
    joints = np.asarray(hand_joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[0] < 26 or joints.shape[1] < 3:
        raise ValueError(
            f"Pico hand joints must have shape (26, >=3), got {joints.shape}"
        )
    positions = joints[:26, :3]
    if not np.all(np.isfinite(positions)):
        raise ValueError("Pico hand joints contain non-finite positions")

    thumb_flexion = _chain_flexion(
        positions, _PICO_THUMB_JOINTS, np.deg2rad(150.0)
    )
    finger_flexions = [
        _chain_flexion(positions, indices, np.deg2rad(250.0))
        for indices in _PICO_FINGER_JOINTS
    ]

    palm_width = np.linalg.norm(
        positions[_PICO_INDEX_METACARPAL] - positions[_PICO_PINKY_METACARPAL]
    )
    if palm_width < 1e-6:
        thumb_opposition = 0.0
    else:
        thumb_to_pinky = np.linalg.norm(
            positions[_PICO_THUMB_JOINTS[-1]] - positions[_PICO_PINKY_METACARPAL]
        )
        distance_ratio = thumb_to_pinky / palm_width
        # Open thumb is normally >= 1.35 palm widths from the pinky base;
        # a fully opposed thumb is approximately <= 0.35 palm widths away.
        thumb_opposition = float(np.clip(1.35 - distance_ratio, 0.0, 1.0))

    return np.asarray(
        [thumb_flexion, thumb_opposition, *finger_flexions], dtype=np.float32
    )


def compute_brainco_hand_targets(
    left_hand_joints: np.ndarray, right_hand_joints: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Retarget both Pico hands to BrainCo commands."""
    return (
        compute_brainco_hand_target(left_hand_joints),
        compute_brainco_hand_target(right_hand_joints),
    )


class BraincoHandCommandPublisher:
    """Publishes continuous BrainCo commands retargeted from Pico hand tracking."""

    def __init__(
        self,
        dds_domain_id: int = 0,
        dds_network_interface: str | None = None,
        hand_speed: float = BRAINCO_DEFAULT_HAND_SPEED,
    ):
        if _BRAINCO_DDS_IMPORT_ERROR is not None:
            raise ImportError(
                "unitree_sdk2py is required for BrainCo hand control. "
                "Install/source the Unitree SDK or run with --disable_brainco_hand."
            ) from _BRAINCO_DDS_IMPORT_ERROR
        if not 0.0 <= hand_speed <= 1.0:
            raise ValueError(f"hand_speed must be in [0, 1], got {hand_speed}")
        self.hand_speed = float(hand_speed)
        self._last_left_q = np.zeros(BRAINCO_NUM_MOTORS, dtype=np.float32)
        self._last_right_q = np.zeros(BRAINCO_NUM_MOTORS, dtype=np.float32)
        self._lock = threading.Lock()
        self._closed = False

        ChannelFactoryInitialize(dds_domain_id, networkInterface=dds_network_interface)

        self.left_publisher = ChannelPublisher(BRAINCO_LEFT_COMMAND_TOPIC, MotorCmds_)
        self.left_publisher.Init()
        self.right_publisher = ChannelPublisher(BRAINCO_RIGHT_COMMAND_TOPIC, MotorCmds_)
        self.right_publisher.Init()

        self.left_msg = self._make_command_message()
        self.right_msg = self._make_command_message()
        print(
            "[BrainCoHand] DDS publishers ready: "
            f"{BRAINCO_LEFT_COMMAND_TOPIC}, {BRAINCO_RIGHT_COMMAND_TOPIC}"
        )

    def _make_command_message(self):
        msg = MotorCmds_()
        msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(BRAINCO_NUM_MOTORS)]
        for cmd in msg.cmds:
            cmd.q = 0.0
            cmd.dq = self.hand_speed
        return msg

    def publish(self, left_q_target: np.ndarray, right_q_target: np.ndarray) -> None:
        if self._closed:
            return
        left_q_target = np.asarray(left_q_target, dtype=np.float32).reshape(
            BRAINCO_NUM_MOTORS
        )
        right_q_target = np.asarray(right_q_target, dtype=np.float32).reshape(
            BRAINCO_NUM_MOTORS
        )

        with self._lock:
            for idx in range(BRAINCO_NUM_MOTORS):
                self.left_msg.cmds[idx].q = float(left_q_target[idx])
                self.right_msg.cmds[idx].q = float(right_q_target[idx])

            self.left_publisher.Write(self.left_msg)
            self.right_publisher.Write(self.right_msg)
            self._last_left_q = left_q_target.copy()
            self._last_right_q = right_q_target.copy()

    def publish_from_hand_joints(
        self, left_hand_joints: np.ndarray, right_hand_joints: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        left_q_target, right_q_target = compute_brainco_hand_targets(
            left_hand_joints, right_hand_joints
        )
        self.publish(left_q_target, right_q_target)
        return left_q_target, right_q_target

    def last_targets(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._last_left_q.copy(), self._last_right_q.copy()

    def close(self) -> None:
        self._closed = True
        publishers = (
            getattr(self, "left_publisher", None),
            getattr(self, "right_publisher", None),
        )
        for publisher in publishers:
            if publisher is not None:
                try:
                    publisher.Close()
                except Exception as e:
                    print(f"[BrainCoHand] Warning: failed to close DDS publisher: {e}")


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


# Readers that expose `get_controller_data()` returning the IsaacTeleop
# controller_data dict schema (left/right trigger/squeeze, thumbstick, clicks).
# Tuple form keeps the dispatch sites uniform if/when a second reader speaks
# the same schema.
_ISAAC_TELEOP_READERS = (input_readers.IsaacTeleopReader,)


def get_controller_inputs(reader=None):
    """Fetch controller triggers and grips used in pose metadata/recording."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(ctrl.get("left_trigger_value", 0.0)),
            float(ctrl.get("right_trigger_value", 0.0)),
            float(ctrl.get("left_squeeze_value", 0.0)),
            float(ctrl.get("right_squeeze_value", 0.0)),
        )
    left_trigger = xrt.get_left_trigger()
    right_trigger = xrt.get_right_trigger()
    left_grip = xrt.get_left_grip()
    right_grip = xrt.get_right_grip()
    return left_trigger, right_trigger, left_grip, right_grip


def get_controller_axes(reader=None):
    """Fetch joystick axes (lx, ly, rx, ry). Falls back to zeros if not available."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return 0.0, 0.0, 0.0, 0.0
        left_thumbstick = ctrl.get("left_thumbstick", [0.0, 0.0])
        right_thumbstick = ctrl.get("right_thumbstick", [0.0, 0.0])
        return (
            float(left_thumbstick[0]),
            float(left_thumbstick[1]),
            float(right_thumbstick[0]),
            float(right_thumbstick[1]),
        )
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


def get_abxy_buttons(reader=None):
    """Fetch A,B,X,Y face buttons as booleans (a,b,x,y)."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return False, False, False, False
        return (
            float(ctrl.get("right_primary_click", 0.0)) > 0.5,
            float(ctrl.get("right_secondary_click", 0.0)) > 0.5,
            float(ctrl.get("left_primary_click", 0.0)) > 0.5,
            float(ctrl.get("left_secondary_click", 0.0)) > 0.5,
        )
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

    @property
    def disconnected(self) -> bool:
        return False

    def clear_disconnect(self):
        pass

    def get_timestamp_ns(self) -> int:
        if xrt is None:
            return 0
        return int(xrt.get_time_stamp_ns())

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
                left_hand_joints = np.asarray(xrt.get_left_hand_tracking_state())
                right_hand_joints = np.asarray(xrt.get_right_hand_tracking_state())
                left_hand_active = bool(xrt.get_left_hand_is_active())
                right_hand_active = bool(xrt.get_right_hand_is_active())

                sample = {
                    "body_poses_np": np.array(body_poses),
                    "left_hand_joints": left_hand_joints,
                    "right_hand_joints": right_hand_joints,
                    "left_hand_active": left_hand_active,
                    "right_hand_active": right_hand_active,
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


def _brainco_targets_from_sample(
    sample: dict | None,
    brainco_hand: BraincoHandCommandPublisher | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Publish valid tracked hands and hold each side independently if tracking is lost."""
    if brainco_hand is not None:
        left_target, right_target = brainco_hand.last_targets()
    else:
        left_target = np.zeros(BRAINCO_NUM_MOTORS, dtype=np.float32)
        right_target = left_target.copy()

    publish = False
    if sample is not None:
        left_joints = sample.get("left_hand_joints")
        right_joints = sample.get("right_hand_joints")
        if sample.get("left_hand_active", False) and left_joints is not None:
            left_target = compute_brainco_hand_target(left_joints)
            publish = True
        if sample.get("right_hand_active", False) and right_joints is not None:
            right_target = compute_brainco_hand_target(right_joints)
            publish = True

    if publish and brainco_hand is not None:
        brainco_hand.publish(left_target, right_target)
    return left_target, right_target


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
    reader=None,
    brainco_hand: BraincoHandCommandPublisher | None = None,
):
    """Shared pose streaming loop used by run_pico."""
    if reader is None:
        if xrt is None:
            raise ImportError(
                "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run pose streaming."
            )

        # Create reader and start it
        reader = PicoReader(max_queue_size=buffer_size)
        reader.start()

    streamer = PoseStreamer(
        socket=socket,
        reader=reader,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        brainco_hand=brainco_hand,
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
        reader.stop()


class PoseStreamer:
    """Encapsulates the pose streaming loop state and logic."""

    def __init__(
        self,
        socket,
        reader: "PicoReader | input_readers.IsaacTeleopReader",
        num_frames_to_send: int,
        target_fps: int,
        use_cuda: bool,
        record_dir: str,
        record_format: str,
        brainco_hand: BraincoHandCommandPublisher | None = None,
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
        self.brainco_hand = brainco_hand

        self.device = (
            torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")
        )

        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self.record_idx = 0

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
        left_trigger, right_trigger, left_grip, right_grip = get_controller_inputs(self.reader)
        # Get A and B button states for data collection control
        a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons(self.reader)

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

        left_brainco_q, right_brainco_q = _brainco_targets_from_sample(
            sample, self.brainco_hand
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
        _, _, rx, _ = get_controller_axes(self.reader)
        self.yaw_accumulator.update(rx, self.frame_time)

        # Only send if buffer is full and we're not waiting for fresh data
        if buffer_is_full and not self.buffer_cleared:
            numpy_data = {
                "smpl_pose": np.stack((self.frame_buffer["smpl_pose"]), axis=0),
                "smpl_joints": np.stack((self.frame_buffer["smpl_joints"]), axis=0),
                "body_quat_w": np.stack((self.frame_buffer["body_quat_w"]), axis=0),
                "joint_pos": np.stack((self.frame_buffer["joint_pos"]), axis=0),
                "joint_vel": np.zeros((N, 29)),
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
                "left_brainco_q": left_brainco_q.astype(np.float32),
                "right_brainco_q": right_brainco_q.astype(np.float32),
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


def _init_input_source(
    input_source: str,
    buffer_size: int,
) -> "PicoReader | input_readers.IsaacTeleopReader":
    """Create, start, and wait for readiness of the requested teleop input source."""
    if input_source == "isaac-teleop":
        reader = input_readers.IsaacTeleopReader(max_queue_size=buffer_size)
        reader.start()
        print("Using Isaac Teleop (in-process CloudXR / DeviceIO), waiting for data...")
        while reader.get_latest() is None:
            print("waiting for Isaac Teleop body data (connect the headset to CloudXR)...")
            time.sleep(1)
        return reader

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

    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()
    return reader


def run_pico(
    buffer_size: int = 15,
    port: int = 5556,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    input_source: str = "xrt",
    enable_brainco_hand: bool = True,
    brainco_dds_domain_id: int = 0,
    brainco_network_interface: str | None = None,
    brainco_hand_speed: float = BRAINCO_DEFAULT_HAND_SPEED,
):
    """Stream full-body Pico motion to Gear SONIC and fingers to BrainCo."""
    if enable_brainco_hand and input_source != "xrt":
        raise ValueError(
            "BrainCo finger retargeting requires Pico/XRoboToolkit hand joints; "
            "use --input-source xrt or add --disable_brainco_hand"
        )
    reader = _init_input_source(input_source, buffer_size)
    brainco_hand = (
        BraincoHandCommandPublisher(
            dds_domain_id=brainco_dds_domain_id,
            dds_network_interface=brainco_network_interface,
            hand_speed=brainco_hand_speed,
        )
        if enable_brainco_hand
        else None
    )
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"Full-body pose ZMQ publisher bound to port {port}, topic=pose")
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
            reader=reader,
            brainco_hand=brainco_hand,
        )
    finally:
        if brainco_hand is not None:
            brainco_hand.close()
        socket.close()
        context.term()
        if input_source == "xrt" and xrt is not None:
            try:
                xrt.close()
            except Exception as e:
                print(f"Warning: failed to close XRoboToolkit cleanly: {e}")
        print("Threads stopped, ZMQ socket closed")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=15, help="Sliding window buffer size")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ server port (default: 5556)")
    parser.add_argument(
        "--num_frames_to_send", type=int, default=5, help="Number of frames to send (default: 5)"
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
        "--input-source",
        type=str,
        default="xrt",
        choices=["xrt", "isaac-teleop"],
        help=(
            "Input source: 'xrt' for XRoboToolkit SDK (default), "
            "'isaac-teleop' for in-process IsaacTeleop / CloudXR DeviceIO"
        ),
    )
    parser.add_argument(
        "--disable_brainco_hand",
        action="store_true",
        help="Disable direct BrainCo hand DDS control",
    )
    parser.add_argument(
        "--brainco_dds_domain",
        type=int,
        default=0,
        help="DDS domain id for BrainCo hand topics (default: 0)",
    )
    parser.add_argument(
        "--brainco_network_interface",
        type=str,
        default="enP8p1s0",
        help="Network interface for BrainCo DDS, e.g. eth0 (default: enP8p1s0)",
    )
    parser.add_argument(
        "--brainco_hand_speed",
        type=float,
        default=BRAINCO_DEFAULT_HAND_SPEED,
        help="Normalized BrainCo finger speed in [0, 1] (default: 1.0)",
    )
    args = parser.parse_args()

    run_pico(
        buffer_size=args.buffer_size,
        port=args.port,
        num_frames_to_send=args.num_frames_to_send,
        target_fps=args.target_fps,
        use_cuda=args.cuda,
        record_dir=args.record_dir,
        record_format=args.record_format,
        input_source=args.input_source,
        enable_brainco_hand=not args.disable_brainco_hand,
        brainco_dds_domain_id=args.brainco_dds_domain,
        brainco_network_interface=args.brainco_network_interface,
        brainco_hand_speed=args.brainco_hand_speed,
    )
