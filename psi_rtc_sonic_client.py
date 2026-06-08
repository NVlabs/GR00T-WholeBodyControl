import os
import sys
import time
import threading
import json
import signal
import struct
from collections import deque

import cv2
import numpy as np
import zmq
import msgpack
from websocket import WebSocketApp

# Add project root to sys.path so `gear_sonic` is importable regardless of cwd.
# This file lives at the repo root (next to the gear_sonic/ package).
_GROOT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _GROOT_ROOT)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    pack_pose_message,
    build_command_message,
)

# ---------------- Configuration ----------------
TASK_INSTRUCTION = "default/grasp_the_silver_bottle_and_pour_water_into_the_kettle"

# FSQ configuration (must match g1_sonic_client / encoder)
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16


def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(continuous_value, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    quantized = np.clip(quantized, fsq_min, fsq_max)
    return quantized

# ---------------- Serialization utilities ----------------
from base64 import b64encode, b64decode
from numpy.lib.format import dtype_to_descr, descr_to_dtype


def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(dct["shape"]) if dct["shape"] else np_obj[0]
    return dct


def convert_numpy_in_dict(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    else:
        return data


# ---------------- RSCamera ----------------
class RSCamera:
    def __init__(self, address="tcp://192.168.123.164:5558"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(address)

    def get_frame(self):
        self.socket.send(b"get_frame")
        rgb_bytes, _, _ = self.socket.recv_multipart()
        rgb_array = np.frombuffer(rgb_bytes, np.uint8)
        rgb_image = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        return rgb_image


# ---------------- RobotStateSubscriber ----------------
class RobotStateSubscriber:
    """Subscribe to robot state published by g1_deploy_onnx_ref on ZMQ PUB port."""

    def __init__(self, host="localhost", port=5557, topic="g1_debug", queue_size=30):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout (for fast shutdown)
        self._socket.setsockopt(zmq.RCVHWM, 1)

        self._topic = topic
        self._lock = threading.Lock()
        self._state_queue = deque(maxlen=queue_size)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                msg = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break

            # Strip topic prefix
            topic_bytes = self._topic.encode("utf-8")
            if msg.startswith(topic_bytes):
                payload = msg[len(topic_bytes):]
            else:
                payload = msg

            try:
                state = msgpack.unpackb(payload, raw=False)
                with self._lock:
                    self._state_queue.append(state)
            except Exception as e:
                print(f"[StateSubscriber] Unpack error: {e}")

    def get_state(self):
        """Return the latest robot state dict, or None if queue is empty."""
        with self._lock:
            return self._state_queue[-1] if self._state_queue else None

    def get_all_states(self):
        """Return a list of all queued robot states, padded with earliest frame to reach queue_size."""
        with self._lock:
            if not self._state_queue:
                return []
            states = list(self._state_queue)
            if len(states) < self._state_queue.maxlen:
                earliest = states[0]
                padding = [earliest] * (self._state_queue.maxlen - len(states))
                return padding + states
            return states

    def get_state_queue(self):
        """Return a copy of the state queue (as deque)."""
        with self._lock:
            return deque(self._state_queue)

    def clear_states(self):
        """Clear all queued states."""
        with self._lock:
            self._state_queue.clear()

    def stop(self):
        self._running = False
        self._thread.join(timeout=0.5)
        self._socket.close(linger=0)
        self._context.term()


# ---------------- TokenPublisher ----------------
class TokenPublisher:
    """ZMQ publisher for token-only streaming (Protocol v4), same as g1_sonic_client."""

    def __init__(self, host="*", port=5556, topic="pose"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://{host}:{port}")
        self._topic = topic
        self._frame_index = 0

    def send_command(self, start=False, stop=False, planner=False):
        msg = build_command_message(start=start, stop=stop, planner=planner)
        self._socket.send(msg)
        print(f"[TokenPublisher] Command: start={start} stop={stop} planner={planner}")

    def publish_token(self, action):
        """
        Publish action token message (Protocol v4).

        Args:
            action: np.ndarray of shape (78,) — hand_joints(14) + token(64)
        """
        action = action.astype(np.float32).reshape(1, -1)
        pose_data = {
            "token_state": action[:, :64],       # (1, 64)
            "left_hand_joints": action[:, 64:71],    # (1, 7)
            "right_hand_joints": action[:, 71:78], # (1, 7)
        }
        msg = pack_pose_message(pose_data, topic=self._topic, version=4)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        self._socket.close()
        self._context.term()

# ---------------- Global state ----------------
running = threading.Event()
running.set()


# ---------------- RTCWebSocketClient ----------------
class RTCWebSocketClient:
    def __init__(self, server_url, state_subscriber, camera, token_publisher):
        self.server_url = server_url
        self._running = True
        self._connected = threading.Event()
        self._ws = None
        self._send_lock = threading.Lock()
        self.start_time = time.time()

        self._state_sub = state_subscriber
        self._camera = camera
        self._token_publisher = token_publisher

    def execute_action(self, action):
        """
        Publish action token (78D: hand_joints(14) + token(64)) via Protocol v4.
        """
        if action.ndim > 1:
            action = action[0]

        hand_joints = action[64:78]
        token_ori = action[:64]
        token_qtz = fsq_quantize(token_ori)

        action_out = np.concatenate([token_qtz, hand_joints])
        self._token_publisher.publish_token(action_out)

    def _on_open(self, ws):
        print("[client] Connected!")
        self._connected.set()

    def _on_message(self, ws, message):
        interval = time.time() - self.start_time
        self.start_time = time.time()
        print(f"[client] recv_action interval: {interval:.3f}s")

        try:
            data = json.loads(message)
            action_data = data.get("action")
            version = data.get("version", -1)

            if action_data is not None:
                action = convert_numpy_in_dict(action_data, numpy_deserialize)
                if isinstance(action, np.ndarray):
                    self.execute_action(action)
                    print(f"[client] Received action, version={version}, shape={action.shape}")

        except Exception as e:
            print(f"[client] Message processing error: {e}")

    def _on_error(self, ws, error):
        print(f"[client] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[client] Connection closed: {close_status_code} - {close_msg}")
        self._running = False
        running.clear()

    def _send_thread(self):
        print("[client] Send thread started, waiting for connection...")
        self._connected.wait()
        print("[client] Connected, starting observation loop")

        prev_tick = time.perf_counter()

        while self._running and running.is_set():
            try:
                # Get robot state
                # state = self._state_sub.get_state()
                states = self._state_sub.get_all_states()
                if len(states) == 0:
                    print("[client] No robot state yet, waiting...")
                    time.sleep(0.1)
                    continue
                
                # states_np = np.zeros((len(states), 38), dtype=np.float32)
                states_list = []
                for state in states:
                    body_q    = np.array(state["body_q_measured"],   dtype=np.float32)
                    left_hand_states = np.array(state["left_hand_q"], dtype=np.float32)
                    right_hand_states = np.array(state["right_hand_q"], dtype=np.float32)

                    state = np.concatenate((body_q, left_hand_states, right_hand_states), axis=0)
                    states_list.append(state)

                states_np = np.stack(states_list, axis=0)  # (N, Ds)

                # Get camera frame
                frame = self._camera.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.uint8)

                # Build observation payload
                img_obs = {"observation.images.egocentric": frame}
                state_obs = {"states": states_np}

                payload = {
                    "image": img_obs,
                    "state": state_obs,
                    "gt_action": None,
                    "dataset_name": None,
                    "instruction": TASK_INSTRUCTION,
                    "history": None,
                    "condition": None,
                    "timestamp": None,
                }
                payload = convert_numpy_in_dict(payload, numpy_serialize)
                message = json.dumps(payload)

                # Send (thread-safe)
                with self._send_lock:
                    if self._ws and self._ws.sock and self._ws.sock.connected:
                        self._ws.send(message)
                    else:
                        print("[client] WebSocket not connected, skipping send")
                        break

            except Exception as e:
                print(f"[client] Send error: {e}")
                break

            now = time.perf_counter()
            interval = now - prev_tick
            prev_tick = now
            print(f"[client] send interval: {interval:.3f}s")

        print("[client] Send thread stopped")

    def run(self):
        print(f"[client] Connecting to {self.server_url}")

        self._ws = WebSocketApp(
            self.server_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        send_thread = threading.Thread(target=self._send_thread, daemon=True)
        send_thread.start()

        self._ws.run_forever()

        self._running = False
        send_thread.join(timeout=0.5)
        print("[client] Client stopped")

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()


# ---------------- Main ----------------
def main(server_url, zmq_host, zmq_pub_port, zmq_sub_port, zmq_topic, zmq_sub_topic,
         camera_address, history_length=1):
    print("[MAIN] Initializing components...")

    # 1. Initialize token publisher (ZMQ PUB, Protocol v4)
    token_publisher = TokenPublisher(host="*", port=zmq_pub_port, topic=zmq_topic)
    print(f"[MAIN] TokenPublisher bound on port {zmq_pub_port}, topic='{zmq_topic}'")

    # 2. Initialize robot state subscriber (ZMQ SUB)
    state_sub = RobotStateSubscriber(host=zmq_host, port=zmq_sub_port, topic=zmq_sub_topic, queue_size=history_length)
    print(f"[MAIN] State subscriber connected to {zmq_host}:{zmq_sub_port}, topic='{zmq_sub_topic}'")

    # 3. Initialize camera
    camera = RSCamera(address=camera_address)
    print(f"[MAIN] Camera connected to {camera_address}")

    # 4. Wait briefly for ZMQ PUB socket to establish connections
    time.sleep(1.0)

    # 5. Send start command (planner mode for token streaming)
    token_publisher.send_command(start=True, stop=False, planner=True)

    # 6. Wait for first robot state
    print("[MAIN] Waiting for robot state...")
    for i in range(30):
        state = state_sub.get_state()
        if state is not None:
            print(f"[MAIN] Got robot state with keys: {list(state.keys())}")
            body_q = np.array(state.get("body_q_measured", []))
            print(f"[MAIN] body_q_measured shape: {body_q.shape}")
            break
        time.sleep(0.5)
    else:
        print("[MAIN] WARNING: No robot state received after 15s, proceeding anyway...")

    # 7. Start WebSocket client
    client = RTCWebSocketClient(
        server_url=server_url,
        state_subscriber=state_sub,
        camera=camera,
        token_publisher=token_publisher,
    )

    def websocket_thread():
        client.run()
        print("[WS] WebSocket thread stopped")

    t_ws = threading.Thread(target=websocket_thread, daemon=True)
    t_ws.start()

    print("[MAIN] Running. Ctrl+C to stop.")

    # 8. Wait for shutdown
    def signal_handler(sig, frame):
        print("\n[MAIN] Caught signal, shutting down...")
        running.clear()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while running.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[MAIN] Caught Ctrl+C, exiting...")
        running.clear()

    # 9. Shutdown
    print("[MAIN] Shutting down...")
    client.stop()

    # Send stop command
    try:
        token_publisher.send_command(start=False, stop=True, planner=True)
    except Exception as e:
        print(f"[MAIN] Error sending stop command: {e}")

    state_sub.stop()
    token_publisher.stop()
    print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLA Policy Inference with WBC Stabilization")
    parser.add_argument("--host", type=str, default="localhost",
                        help="VLA policy server host")
    parser.add_argument("--port", type=int, default=8014,
                        help="VLA policy server port")
    parser.add_argument("--zmq-host", type=str, default="localhost",
                        help="ZMQ host for robot state subscriber")
    parser.add_argument("--zmq-pub-port", type=int, default=5556,
                        help="ZMQ PUB port for sending pose to WBC")
    parser.add_argument("--zmq-sub-port", type=int, default=5557,
                        help="ZMQ SUB port for receiving robot state")
    parser.add_argument("--zmq-topic", type=str, default="pose",
                        help="ZMQ topic for pose messages")
    parser.add_argument("--zmq-sub-topic", type=str, default="g1_debug",
                        help="ZMQ topic for robot state subscription")
    parser.add_argument("--camera-address", type=str, default="tcp://192.168.123.164:5558",
                        help="Camera ZMQ address")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Task instruction for VLA policy")
    parser.add_argument("--state-history-length", type=int, default=1,
                        help="Number of past frames to include in state history")

    args = parser.parse_args()

    if args.instruction:
        TASK_INSTRUCTION = args.instruction

    server_url = f"ws://{args.host}:{args.port}/ws"
    main(
        server_url=server_url,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_sub_port=args.zmq_sub_port,
        zmq_topic=args.zmq_topic,
        zmq_sub_topic=args.zmq_sub_topic,
        camera_address=args.camera_address,
        history_length=args.state_history_length,
    )
