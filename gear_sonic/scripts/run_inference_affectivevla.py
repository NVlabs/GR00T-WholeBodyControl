"""
AffectiveVLA two-camera inference runner — NO ROS 2 DEPENDENCY.

Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (latent protocol v4: motion token + hand joints)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Uses the Isaac-GR00T PolicyClient (ZMQ REQ/REP) to communicate with a
running PolicyServer. The robot camera transport names are mapped to the
two video modality names used during training:

  ego_view -> depth_camera_rgb
  head     -> webcam

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> send initial pose and switch to POSE mode
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from dataclasses import dataclass
import queue
import threading
import time

import numpy as np
import os  # AGENT-PATCH
import sys as _sys  # AGENT-PATCH
_sys.path.insert(0, os.path.expanduser('~'))  # AGENT-PATCH
from tactile_agent import TactileAgent  # AGENT-PATCH
import os  # PROBE-PATCH
import sys as _sys  # PROBE-PATCH
_sys.path.append(os.path.expanduser('~'))  # PROBE-PATCH
from chunk_probe import ChunkProbe  # PROBE-PATCH
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    concat_action,
    prepare_observation_for_eval,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (Isaac-GR00T PolicyServer)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5550
    """The port of the Isaac-GR00T PolicyServer."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 50
    """Action horizon of the VLA policy (number of future actions per inference)."""

    rate: float = 1 / 0.2
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Embodiment
    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            if left_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"left_hand_joints must have shape [7], got {left_hand_joints.shape}"
                )
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            if right_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"right_hand_joints must have shape [7], got {right_hand_joints.shape}"
                )
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


# --- BrainCo hand state override (6 DOF, как при сборе данных) ---
_BRAINCO_CTRL = None

def _get_brainco():
    global _BRAINCO_CTRL
    if _BRAINCO_CTRL is None:
        import sys as _sys
        if "/home/unitree/gr00t-g1-bridge" not in _sys.path:
            _sys.path.insert(0, "/home/unitree/gr00t-g1-bridge")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        try:
            ChannelFactoryInitialize(0, "wlxfc23cd997021")
        except Exception as e:
            print(f"[BrainCo] ChannelFactoryInitialize: {e}", flush=True)
        from brainco_hand import BraincoHandController
        _BRAINCO_CTRL = BraincoHandController()
        _BRAINCO_CTRL.wait_ready(10.0)
        print("[BrainCo] hand state reader ready (6 DOF)", flush=True)
    return _BRAINCO_CTRL

def _to7(h):
    """6 моторов BrainCo -> 7-мерный формат, который ждёт WBC."""
    h = np.asarray(h, dtype=np.float32).reshape(-1)[:6]
    out = np.zeros(7, dtype=np.float32)
    out[:6] = h
    out[6] = h[5]
    return out


_slew_prev = [None]  # SLEW-PATCH


def _slew_limit(token):  # SLEW-PATCH
    """
    Cap the per-tick change of motion_token.

    Measured chunk-boundary jumps reach 0.49 in L2 against a normal playback
    step of 0.017, which is what the arm feels as a jolt. This spreads such a
    jump over several ticks. Returns the token unchanged when SLEW_ENABLE is
    not set, so the default path is untouched.
    """
    import os as _os
    if _os.environ.get("SLEW_ENABLE", "0") != "1":
        return token
    try:
        import numpy as _np
        cap = float(_os.environ.get("SLEW_MAX", "0.05"))
        cur = _np.asarray(token, dtype=_np.float32).reshape(-1)
        prev = _slew_prev[0]
        if prev is None or prev.shape != cur.shape:
            _slew_prev[0] = cur.copy()
            return token
        delta = cur - prev
        dist = float(_np.linalg.norm(delta))
        if dist > cap:
            cur = prev + delta * (cap / dist)
        _slew_prev[0] = cur.copy()
        return cur.reshape(_np.asarray(token).shape)
    except Exception as e:  # noqa: BLE001
        print(f"[slew] limiter failed, passing through: {e}", flush=True)
        return token


_slew_hand_prev = [None]  # SLEW-PATCH


def _slew_hand(vec):  # SLEW-PATCH
    """
    Cap the per-tick change of the hand command.

    Measured: at chunk boundaries the flex command snaps BACKWARD by up to
    0.499, because the new plan expects the hand less closed than it already
    is. A genuine release moves about 0.014 per tick, so the default cap of
    0.03 passes real motion at double speed and stretches a snap over ~14
    ticks. SLEW_HAND=0 disables just this part.
    """
    import os as _os
    if _os.environ.get("SLEW_ENABLE", "0") != "1":
        return vec
    try:
        import numpy as _np
        cap = float(_os.environ.get("SLEW_HAND", "0.03"))
        if cap <= 0:
            return vec
        cur = _np.asarray(vec, dtype=_np.float32).reshape(-1).copy()
        prev = _slew_hand_prev[0]
        if prev is None or prev.shape != cur.shape:
            _slew_hand_prev[0] = cur.copy()
            return vec
        step = _np.clip(cur - prev, -cap, cap)
        cur = prev + step
        _slew_hand_prev[0] = cur.copy()
        return cur.reshape(_np.asarray(vec).shape)
    except Exception as e:  # noqa: BLE001
        print(f"[slew] hand limiter failed, passing through: {e}", flush=True)
        return vec


def _send_brainco_hands(left, right):
    """Шлём 6-DOF команды кисти напрямую в BrainCo DDS (как при сборе данных)."""
    import os as _os
    enable = _os.environ.get("HAND_ENABLE", "0") == "1"
    try:
        ctrl = _get_brainco()
        l = np.clip(np.asarray(left, dtype=np.float32).reshape(-1)[:6], 0.0, 1.0)
        r = np.clip(np.asarray(right, dtype=np.float32).reshape(-1)[:6], 0.0, 1.0)
        ctrl.send_targets_normalized(l, r, dry_run=not enable)
        if not enable:
            print(f"[BrainCo] DRY-RUN right={r.round(3)}", flush=True)
    except Exception as e:
        print(f"[BrainCo] send failed: {e}", flush=True)


def _override_hand_state(observation):
    try:
        left, right = _get_brainco().get_state_normalized()
        observation["state"]["left_hand"] = np.asarray(left, dtype=np.float32)[np.newaxis, np.newaxis]
        observation["state"]["right_hand"] = np.asarray(right, dtype=np.float32)[np.newaxis, np.newaxis]
    except Exception as e:
        print(f"[BrainCo] override failed: {e}", flush=True)
    return observation


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
):
    """Read sensors and prepare observation for the VLA policy.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    images = camera_msg.get("images", {})
    missing_cameras = [name for name in ("ego_view", "head") if images.get(name) is None]
    if missing_cameras:
        if log_errors:
            print(
                "[DEBUG] prepare_observation: missing required camera stream(s) "
                f"{missing_cameras}; available={sorted(images)}",
                flush=True,
            )
        return None

    import cv2 as _cv2

    def resize_for_policy(image: np.ndarray) -> np.ndarray:
        """Reduce transport size without changing the camera aspect ratio."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 camera image, got {getattr(image, 'shape', None)}")
        height, width = image.shape[:2]
        if min(height, width) <= 256:
            return image
        scale = 256 / min(height, width)
        return _cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=_cv2.INTER_AREA,
        )

    ego_img = resize_for_policy(images["ego_view"])
    head_img = resize_for_policy(images["head"])

    # Copy index finger data to middle finger (hardware coupling)
    state_msg["left_hand_q"][5] = state_msg["left_hand_q"][3]
    state_msg["left_hand_q"][6] = state_msg["left_hand_q"][4]

    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )

    # These keys must exactly match the two-camera ModalityConfig saved in the
    # GR00T checkpoint. The source camera names remain ego_view and head.
    video = {
        "depth_camera_rgb": ego_img[np.newaxis, np.newaxis],
        "webcam": head_img[np.newaxis, np.newaxis],
    }

    observation = {
        "video": video,
        "state": {},
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }

    observation = prepare_observation_for_eval(robot_model, observation)
    observation = _override_hand_state(observation)

    # Projected gravity for Sonic latent embodiment
    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]

    return observation


def run_policy_inference_and_process(policy, observation, robot_model):
    """Run policy inference via Isaac-GR00T PolicyClient and process results.

    Returns:
        processed_action dict or None on error.
    """
    try:
        try:
            for k, vv in observation.get("video", {}).items():
                print(f"[DBG] video[{k}] type={type(vv)}", flush=True)
                if isinstance(vv, dict):
                    print(f"[DBG]   keys={list(vv.keys())[:5]}", flush=True)
            for k, vv in observation.get("state", {}).items():
                print(f"[DBG] state[{k}] shape={getattr(vv,'shape',None)}", flush=True)
        except Exception as e:
            print("[DBG] fail:", e, flush=True)
        action, _info = policy.get_action(observation)

        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = "motion_token" if "motion_token" in action else "action.motion_token"
        if np.abs(action[motion_key]).max() > 1.25:
            print(
                f"[Warning] action['{motion_key}'] max "
                f"({np.abs(action[motion_key]).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None

        processed_action = concat_action(robot_model, action)
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn()
                if observation is None:
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)

                if processed_action is not None:
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time))
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    """Compute closed hand joint positions using G1GripperInverseKinematicsSolver."""
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def main(config: InferenceConfig):
    pause_loop = True
    freeze_body = False        # body frozen: skip motion_token, WBC holds pose
    hand_override_ref = [None]  # agent puts a 6-vec here to override the RIGHT hand; None = use VLA

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    # Isaac-GR00T PolicyClient
    import sys; sys.path.insert(0, "/home/unitree/gr00t-g1-bridge")
    from groot_client_liza import Gr00tClient as PolicyClient

    n1_policy = PolicyClient(host=config.host, port=config.port)

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        print("WARNING: PolicyServer not reachable. Inference will fail until server is up.")

    state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host,
        port=config.state_zmq_port,
    )

    camera_subscriber = ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(
        f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    )
    print_green(f"Using embodiment tag: {config.embodiment_tag}")

    keyboard_listener = ZMQKeyboardSubscriber(
        port=config.keyboard_zmq_port, host=config.keyboard_zmq_host
    )

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    def publish_initial_pose():
        """Publish initial pose command to move robot to starting position."""
        print("Moving to initial pose")
        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        zmq_message = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)
        print("Initial pose done.")

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    agent = TactileAgent(hand=os.environ.get('AGENT_HAND', 'right'),
                         enabled=os.environ.get('AGENT_ENABLE') == '1',
                         observe=os.environ.get('AGENT_OBSERVE') == '1')  # AGENT-PATCH
    print(f'[agent] hand={agent.hand} enabled={agent.enabled} observe={agent.observe} adjust={agent.adjust}')  # AGENT-PATCH
    probe = ChunkProbe(enabled=os.environ.get('PROBE_ENABLE') == '1')  # PROBE-PATCH
    print(f'[probe] enabled={probe.enabled}')  # PROBE-PATCH
    inference_interval = 1.0 / config.rate

    zmq_frame_counter = 0

    PROMPT_MSG_PREFIX = "prompt:"

    def check_keyboard_input():
        nonlocal pause_loop, freeze_body, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            print("Moving to initial pose")
            zmq_frame_counter = 0
            print("Reset ZMQ frame counter")
            publish_initial_pose()
            cached_action_chunk = None
            action_chunk_index = 0
            print("Cleared cached action chunk")
            if agent.enabled:  # AGENT-PATCH
                agent.reset()  # AGENT-PATCH
            if cpp_loop_running and cpp_mode == "PLANNER":
                if send_cpp_control_command(start=True, planner=False):
                    print("Switched to POSE mode (from PLANNER mode)")
                else:
                    print("Warning: Failed to switch to POSE mode")
            elif not cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")
        elif key == "g":
            freeze_body = not freeze_body
            print(f"{'FROZEN body (fingers live)' if freeze_body else 'Body resumed'} - WBC holds pose")
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if pause_loop:
                print("Policy loop paused (C++ loop still running - press 'k' to stop)")
            else:
                print("Policy loop resumed")
        elif key == "k":
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(
                f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}"
            )
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(
                f"Initial pose right hand: "
                f"{'closed' if initial_pose_right_hand_closed else 'open'}"
            )

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
                log_errors=True,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                robot_model=robot_model,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                processed_action, inference_start_time = result_queue.get_nowait()
                inference_delay = time.monotonic() - inference_start_time
                action_chunk_index = calculate_latency_compensated_index(
                    inference_delay, config.action_publish_rate, config.action_horizon
                )
                cached_action_chunk = processed_action
                if probe.enabled:  # PROBE-PATCH
                    probe.note_delay(inference_delay, action_chunk_index)  # PROBE-PATCH
                last_inference_time = time.monotonic()
                print_green(
                    f'New action chunk (prompt: "{language_prompt_ref[0]}", '
                    f"latency: {inference_delay:.3f}s)"
                )
            except queue.Empty:
                pass

            worker_is_busy = inference_busy_event.is_set()
            should_start = should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=worker_is_busy,
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            )

            if should_start:
                try:
                    inference_queue.put_nowait(None)
                except queue.Full:
                    pass

            if pause_loop:
                print("Pausing...", end="", flush=True)
                time.sleep(0.2)
                print(".", end="", flush=True)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                else:
                    motion_token = np.asarray(
                        get_action_field(processed_action, "motion_token"),
                        dtype=np.float32,
                    )
                    left_hand_joints = np.asarray(
                        get_action_field(processed_action, "left_hand_joints"),
                        dtype=np.float32,
                    )
                    right_hand_joints = np.asarray(
                        get_action_field(processed_action, "right_hand_joints"),
                        dtype=np.float32,
                    )

                    # Action arrays arrive as (B, T, D) from the model.
                    # Squeeze batch dim to get (T, D), then index by time step.
                    if motion_token.ndim == 3:
                        motion_token = motion_token[0]
                    if left_hand_joints.ndim == 3:
                        left_hand_joints = left_hand_joints[0]
                    if right_hand_joints.ndim == 3:
                        right_hand_joints = right_hand_joints[0]

                    horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                    current_idx = min(action_chunk_index, horizon - 1)

                    # # AGENT-PATCH: chunk is still (T,6) — the agent looks ahead
                    if agent.enabled:  # AGENT-PATCH
                        agent.prompt = language_prompt_ref[0]  # AGENT-PATCH
                        _agent_hand = left_hand_joints if agent.hand == 'left' else right_hand_joints  # AGENT-PATCH
                        try:  # AGENT-PATCH
                            _agent_out = agent.step(_agent_hand, current_idx)  # AGENT-PATCH
                        except Exception as _ae:  # AGENT-PATCH
                            print(f'[agent] step failed, disabling: {_ae}')  # AGENT-PATCH
                            agent.enabled = False; agent.freeze_body = False  # AGENT-PATCH
                            _agent_out = None  # AGENT-PATCH
                        if agent.hand != 'left':  # AGENT-PATCH
                            hand_override_ref[0] = _agent_out  # AGENT-PATCH
                    if probe.enabled:  # PROBE-PATCH
                        probe.on_step(left_hand_joints if os.environ.get('PROBE_HAND')=='left' else right_hand_joints, current_idx)  # PROBE-PATCH

                    if motion_token.ndim == 2:
                        motion_token = motion_token[current_idx]
                    if left_hand_joints.ndim == 2:
                        left_hand_joints = left_hand_joints[current_idx]
                    if right_hand_joints.ndim == 2:
                        right_hand_joints = right_hand_joints[current_idx]

                    frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                    zmq_frame_counter += 1

                    # --- split control (freeze_body / hand_override) ---
                    _override = hand_override_ref[0]
                    right_hand_out = (
                        np.asarray(_override, dtype=right_hand_joints.dtype).reshape(right_hand_joints.shape)
                        if _override is not None
                        else right_hand_joints
                    )
                    # fingers ALWAYS sent (live hand): override if agent set one, else VLA
                    if agent.enabled and agent.hand == 'left' and _agent_out is not None:  # AGENT-PATCH
                        left_hand_joints = np.asarray(_agent_out, dtype=left_hand_joints.dtype).reshape(left_hand_joints.shape)  # AGENT-PATCH
                    # # THUMBLOCK-PATCH: большой палец защёлкивается от первой же
                    # команды на сгиб (13 авг) — не даём её вообще.
                    # Снять после ремонта: --revert
                    for _bi in [0]:  # THUMBLOCK-PATCH
                        right_hand_out[_bi] = 0.0  # THUMBLOCK-PATCH
                    left_hand_joints = _slew_hand(left_hand_joints)  # SLEW-PATCH
                    _send_brainco_hands(left_hand_joints, right_hand_out)
                    if probe.enabled:  # PROBE-PATCH
                        probe.on_sent(left_hand_joints if os.environ.get('PROBE_HAND')=='left' else right_hand_out, motion_token)  # PROBE-PATCH
                    # body/motion_token skipped when frozen - WBC holds the last pose itself
                    if not (freeze_body or (agent.enabled and agent.freeze_body)):  # AGENT-PATCH
                        motion_token = _slew_limit(motion_token)  # SLEW-PATCH
                        zmq_message = pack_latent_action_message(
                            motion_token,
                            frame_index,
                            left_hand_joints=_to7(left_hand_joints),
                            right_hand_joints=_to7(right_hand_out),
                        )
                        zmq_socket.send(zmq_message)
                        if zmq_frame_counter % 50 == 0:
                            print_green(
                                f"ZMQ: Sent latent action - "
                                f"frame: {frame_index[0]}, "
                                f"token shape: {motion_token.shape}"
                            )

                action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(
                    context="VLA Inference Loop Missed", threshold=0.001
                )

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
