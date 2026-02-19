from copy import deepcopy
import pickle
import queue
import threading
import time
from typing import Dict, List, Tuple

import leap
import leap.events
import numpy as np
from scipy.spatial.transform import Rotation as R

from decoupled_wbc.control.teleop.streamers.base_streamer import BaseStreamer, StreamerOutput


def xyz2np_array(position: leap.datatypes.Vector) -> np.ndarray:
    return np.array([position.x, position.y, position.z])


def quat2np_array(quaternion: leap.datatypes.Quaternion) -> np.ndarray:
    return np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])


def get_raw_finger_points(hand: leap.datatypes.Hand) -> np.ndarray:
    target_points = np.array([np.eye(4) for _ in range(25)])
    target_points[0, :3, :3] = R.from_quat(quat2np_array(hand.palm.orientation)).as_matrix()
    target_points[0, :3, 3] = xyz2np_array(hand.arm.next_joint)
    target_points[1, :3, 3] = xyz2np_array(hand.thumb.bones[0].next_joint)
    target_points[2, :3, 3] = xyz2np_array(hand.thumb.bones[1].next_joint)
    target_points[3, :3, 3] = xyz2np_array(hand.thumb.bones[2].next_joint)
    target_points[4, :3, 3] = xyz2np_array(hand.thumb.bones[3].next_joint)
    target_points[5, :3, 3] = xyz2np_array(hand.index.bones[0].prev_joint)
    target_points[6, :3, 3] = xyz2np_array(hand.index.bones[0].next_joint)
    target_points[7, :3, 3] = xyz2np_array(hand.index.bones[1].next_joint)
    target_points[8, :3, 3] = xyz2np_array(hand.index.bones[2].next_joint)
    target_points[9, :3, 3] = xyz2np_array(hand.index.bones[3].next_joint)
    target_points[10, :3, 3] = xyz2np_array(hand.middle.bones[0].prev_joint)
    target_points[11, :3, 3] = xyz2np_array(hand.middle.bones[0].next_joint)
    target_points[12, :3, 3] = xyz2np_array(hand.middle.bones[1].next_joint)
    target_points[13, :3, 3] = xyz2np_array(hand.middle.bones[2].next_joint)
    target_points[14, :3, 3] = xyz2np_array(hand.middle.bones[3].next_joint)
    target_points[15, :3, 3] = xyz2np_array(hand.ring.bones[0].prev_joint)
    target_points[16, :3, 3] = xyz2np_array(hand.ring.bones[0].next_joint)
    target_points[17, :3, 3] = xyz2np_array(hand.ring.bones[1].next_joint)
    target_points[18, :3, 3] = xyz2np_array(hand.ring.bones[2].next_joint)
    target_points[19, :3, 3] = xyz2np_array(hand.ring.bones[3].next_joint)
    target_points[20, :3, 3] = xyz2np_array(hand.pinky.bones[0].prev_joint)
    target_points[21, :3, 3] = xyz2np_array(hand.pinky.bones[0].next_joint)
    target_points[22, :3, 3] = xyz2np_array(hand.pinky.bones[1].next_joint)
    target_points[23, :3, 3] = xyz2np_array(hand.pinky.bones[2].next_joint)
    target_points[24, :3, 3] = xyz2np_array(hand.pinky.bones[3].next_joint)

    return target_points


def get_fake_finger_points_from_pinch(hand: leap.datatypes.Hand) -> np.ndarray:
    # print(hand.pinch_strength)
    target_points = np.array([np.eye(4) for _ in range(25)])
    for i in range(25):
        target_points[i] = np.eye(4)

    # Control thumb based on shoulder button state (index 4 is thumb tip)
    if hand.pinch_strength > 0.3:
        target_points[4, 0, 3] = 0.0  # closed
    else:
        target_points[4, 0, 3] = 1000.0  # open, in mm

    return target_points


def get_raw_wrist_pose(hand: leap.datatypes.Hand) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hand_palm_position = xyz2np_array(hand.palm.position)
    hand_palm_normal = np.array([-hand.palm.normal.z, -hand.palm.normal.x, hand.palm.normal.y])
    hand_palm_direction = np.array(
        [-hand.palm.direction.z, -hand.palm.direction.x, hand.palm.direction.y]
    )

    return hand_palm_position, hand_palm_normal, hand_palm_direction


def get_finger_transform(target_points: np.ndarray, hand_type: str) -> np.ndarray:
    pose_palm_root = target_points[0, :3, 3].copy()
    rot_palm = target_points[0, :3, :3].copy()

    if hand_type == "right":
        rot_leap2base = R.from_euler("z", [180], degrees=True).as_matrix()
        rot_reverse = np.array(
            [[[0, 0, 1], [0, -1, 0], [1, 0, 0]]]
        )  # due to the target_base_rotation in hand IK solver
    else:
        rot_leap2base = np.eye(3)
        rot_reverse = np.array([[[0, 0, -1], [0, -1, 0], [-1, 0, 0]]])
    offset = (target_points[:, :3, 3] - pose_palm_root).copy() / 1000.0  # mm to m
    offset = offset @ rot_palm @ rot_leap2base @ rot_reverse

    target_points[:, :3, 3] = offset

    return target_points


def get_wrist_transformation(
    hand_palm_position: np.ndarray,
    hand_palm_normal: np.ndarray,
    hand_palm_direction: np.ndarray,
    hand_type: str,
    pos_sensitivity: float = 1.0 / 800.0,
) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = hand_palm_position[[2, 0, 1]] * pos_sensitivity * np.array([-1, -1, 1])
    direction_np = hand_palm_direction
    palm_normal_np = hand_palm_normal
    if hand_type == "left":
        transform = R.from_euler("y", -90, degrees=True).as_matrix()
        lh_thumb_np = -np.cross(direction_np, palm_normal_np)
        rotation_matrix = np.array([direction_np, -palm_normal_np, lh_thumb_np]).T
    else:
        transform = R.from_euler("xy", [-180, 90], degrees=True).as_matrix()
        rh_thumb_np = np.cross(direction_np, palm_normal_np)
        rotation_matrix = np.array([direction_np, palm_normal_np, rh_thumb_np]).T
    T[:3, :3] = np.dot(rotation_matrix, transform)
    return T


def get_raw_data(
    hands: List[leap.datatypes.Hand], data: Dict[str, np.ndarray] = None
) -> Dict[str, np.ndarray]:
    if data is None:
        data = {}
    for hand in hands:
        hand_type = "left" if str(hand.type) == "HandType.Left" else "right"
        data[hand_type + "_wrist"] = get_raw_wrist_pose(hand)
        # @runyud: this is a hack to get the finger points from the pinch strength
        data[hand_type + "_fingers"] = get_fake_finger_points_from_pinch(hand)
    assert len(data) == 4, f"Leapmotiondata length should be 4, but got {len(data)}"
    return data


def process_data(raw_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    data = {}
    for hand_type in ["left", "right"]:
        data[hand_type + "_wrist"] = get_wrist_transformation(
            *raw_data[hand_type + "_wrist"], hand_type
        )
        data[hand_type + "_fingers"] = {
            "position": get_finger_transform(raw_data[hand_type + "_fingers"], hand_type)
        }
    return data


class LeapMotionListener(leap.Listener):
    def __init__(self, verbose=False):
        self.data = None
        self.data_lock = threading.Lock()
        self.verbose = verbose

    def on_connection_event(self, event):
        print("Connected")

    def on_device_event(self, event: leap.events.DeviceEvent):
        try:
            with event.device.open():
                info = event.device.get_info()
        except AttributeError:
            # Handle the case where LeapCannotOpenDeviceError is not available
            try:
                info = event.device.get_info()
            except Exception as e:
                print(f"Error opening device: {e}")
                info = None

        print(f"Found Leap Motion device {info.serial}")

    def on_tracking_event(self, event: leap.events.TrackingEvent):
        if (
            len(event.hands) == 2 or self.data is not None
        ):  # only when two hands are detected, we update the data
            with self.data_lock:
                self.data: Dict[str, np.ndarray] = get_raw_data(event.hands, self.data)
        if self.verbose:
            for hand in event.hands:
                hand_type = "left" if str(hand.type) == "HandType.Left" else "right"
                print(
                    f"Hand id {hand.id} is a {hand_type} hand with position"
                    f"({hand.palm.position.x}, {hand.palm.position.y}, {hand.palm.position.z})."
                )

    def get_data(self):
        with self.data_lock:
            return deepcopy(self.data)

    def reset_status(self):
        """Reset the cache of the streamer."""
        with self.data_lock:
            self.data = None


class FakeLeapMotionListener:
    def __init__(self, data_path=None):
        self.data_lock = threading.Lock()
        with open(data_path, "rb") as f:
            self.data_list = pickle.load(f)
        self.data_index = 0

    def get_data(self) -> Dict[str, np.ndarray]:
        with self.data_lock:
            data = deepcopy(self.data_list[self.data_index % len(self.data_list)])
            self.data_index += 1
        return data


# Note currently it is a auto-polling based streamer.
# The connection will start another thread to continue polling data
# and the listener will continue to receive data from the connection.
class LeapMotionStreamer(BaseStreamer):
    """LeapMotion streamer that provides hand tracking data."""

    def __init__(self, verbose=False, record_data=False, **kwargs):
        self.connection = leap.Connection()
        self.listener = LeapMotionListener(verbose=verbose)
        self.connection.add_listener(self.listener)
        # self.listener = FakeLeapMotionListener(data_path="leapmotion_data_rot.pkl")

        self.connection.set_tracking_mode(leap.TrackingMode.Desktop)

        self.record_data = record_data
        self.data_queue = queue.Queue()

    def reset_status(self):
        """Reset the cache of the streamer."""
        self.listener.reset_status()
        while not self.listener.get_data():
            time.sleep(0.1)

    def start_streaming(self):
        self.connection.connect()
        print("Waiting for the first data...")
        time.sleep(0.5)
        while not self.listener.get_data():
            time.sleep(0.1)
        print("First data received!")

    def stop_streaming(self):
        self.connection.disconnect()

    def get(self) -> StreamerOutput:
        """Return hand tracking data as StreamerOutput."""
        # Get raw data and save if recording
        raw_data = self.listener.get_data()
        if self.record_data:
            self.data_queue.put(raw_data)

        # Process raw data into transformations
        processed_data = process_data(raw_data)

        # Initialize IK data (ik_keys) - LeapMotion provides hand/finger tracking
        ik_data = {}
        for hand_type in ["left", "right"]:
            # Add wrist poses and finger positions to IK data
            ik_data[f"{hand_type}_wrist"] = processed_data[f"{hand_type}_wrist"]
            ik_data[f"{hand_type}_fingers"] = processed_data[f"{hand_type}_fingers"]

        # Return structured output - LeapMotion only provides IK data
        return StreamerOutput(
            ik_data=ik_data,
            control_data={},  # No control commands from LeapMotion
            teleop_data={},  # No teleop commands from LeapMotion
            source="leapmotion",
        )

    def dump_data_to_file(self):
        """Save recorded data to file if recording was enabled."""
        if not self.record_data:
            return

        data_list = []
        while not self.data_queue.empty():
            data_list.append(self.data_queue.get())
        with open("leapmotion_data_trans.pkl", "wb") as f:
            pickle.dump(data_list, f)


if __name__ == "__main__":
    streamer = LeapMotionStreamer(verbose=True)
    streamer.start_streaming()
    for _ in range(100):
        streamer.get()
        time.sleep(0.1)
    streamer.stop_streaming()
    streamer.dump_data_to_file()
