"""Sonic collector integrating robot, BrainCo, camera, and event streams."""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from gear_sonic.scripts.run_data_exporter import GrootDataCollector, unpack_pose_message

from .constants import (
    BRAINCO_MAX_AGE_SEC, BRAINCO_NUM_MOTORS, DEPTH_CAMERA_FEATURE,
    DEPTH_CAMERA_RGB_FEATURE,
    EXTERNAL_VIEW_CAMERA_FEATURE, G1_DATASET_ACTION_FIELDS,
    G1_DATASET_OBSERVATION_FIELDS, HEAD_DEPTH_CAMERA_FEATURE, WEBCAM_FEATURE,
)

class BraincoGrootDataCollector(GrootDataCollector):
    def __init__(
        self,
        *args,
        brainco_subscriber,
        brainco_schema,
        depth_camera_rgb_image_key: str,
        depth_camera_image_key: str,
        webcam_image_key: str,
        head_depth_camera_image_key: str,
        depth_video_max_meters: float,
        record_raw_depth: bool,
        enabled_camera_streams: tuple[str, ...],
        telemetry_receiver,
        telemetry_max_age: float,
        record_raw_telemetry: bool,
        external_view_camera=None,
        **kwargs,
    ):
        self._brainco_subscriber = brainco_subscriber
        self._brainco_schema = brainco_schema
        self._telemetry_receiver = telemetry_receiver
        self._telemetry_max_age = float(telemetry_max_age)
        self._record_raw_telemetry = bool(record_raw_telemetry)
        self._external_view_camera = external_view_camera
        self._depth_camera_rgb_image_key = depth_camera_rgb_image_key
        self._depth_camera_image_key = depth_camera_image_key
        self._webcam_image_key = webcam_image_key
        self._head_depth_camera_image_key = head_depth_camera_image_key
        self._depth_video_max_meters = depth_video_max_meters
        self._record_raw_depth = record_raw_depth
        self._enabled_camera_streams = tuple(enabled_camera_streams)
        self._pending_grip_marker = False
        self._pending_trigger_events = {
            ("left", "press"): False,
            ("left", "release"): False,
            ("right", "press"): False,
            ("right", "release"): False,
        }
        self._pending_depth_frames: dict[str, tuple[np.ndarray, float | None]] = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def _fresh(snapshot: dict, key: str) -> np.ndarray | None:
        value = snapshot.get(key)
        timestamp = snapshot.get(f"{key}_time")
        if value is None or timestamp is None:
            return None
        if time.monotonic() - float(timestamp) > BRAINCO_MAX_AGE_SEC:
            return None
        return np.asarray(value, dtype=np.float32)

    def _handle_pose_message(self, raw: bytes) -> None:
        super()._handle_pose_message(raw)
        if self.latest_sonic_msg is None:
            return
        try:
            pose_data = unpack_pose_message(raw, topic="pose")
            for side in ("left", "right"):
                value = pose_data.get(f"{side}_brainco_q")
                if value is not None:
                    self.latest_sonic_msg[f"{side}_brainco_q"] = np.asarray(
                        value, dtype=np.float32
                    ).reshape(BRAINCO_NUM_MOTORS)
        except Exception as exc:
            print(f"[BrainCo] Warning: could not read pose command fields: {exc}")

    def _handle_manager_state(self, raw: bytes) -> None:
        super()._handle_manager_state(raw)
        try:
            data = unpack_pose_message(raw, topic="manager_state")
            if self._extract_bool(data, "grip_marker"):
                self._pending_grip_marker = True
            for side, event in self._pending_trigger_events:
                if self._extract_bool(data, f"{side}_trigger_{event}"):
                    self._pending_trigger_events[(side, event)] = True
        except Exception as exc:
            print(f"[G1State] Warning: could not read manager state fields: {exc}")

    def _record_pending_controller_events(self) -> None:
        timestamp_sec = self._current_episode_duration_sec()
        if self._pending_grip_marker:
            self.data_exporter.add_grip_timestamp(timestamp_sec)
        for (side, event), pending in self._pending_trigger_events.items():
            if pending:
                self.data_exporter.add_trigger_timestamp(side, event, timestamp_sec)

    def _clear_pending_controller_events(self) -> None:
        self._pending_grip_marker = False
        for key in self._pending_trigger_events:
            self._pending_trigger_events[key] = False

    def _check_recording_commands(self) -> None:
        events_pending = self._pending_grip_marker or any(
            self._pending_trigger_events.values()
        )
        was_recording = self._episode_state.get_state() == self._episode_state.RECORDING

        # For events arriving together with a stop command, capture the time
        # before the parent transitions RECORDING -> NEED_TO_SAVE.
        if events_pending and was_recording:
            self._record_pending_controller_events()

        super()._check_recording_commands()

        # For events arriving together with a start command, timestamp them at
        # the beginning of the newly started episode.
        if (
            events_pending
            and not was_recording
            and self._episode_state.get_state() == self._episode_state.RECORDING
        ):
            self._record_pending_controller_events()
        self._clear_pending_controller_events()

    @staticmethod
    def _depth_to_mm(depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth)
        if depth.ndim != 2:
            raise ValueError(f"Depth image must be HxW, got shape {depth.shape}")
        if np.issubdtype(depth.dtype, np.integer):
            return np.clip(depth, 0, np.iinfo(np.uint16).max).astype(np.uint16)

        depth_m = depth.astype(np.float32)
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        depth_mm[valid] = np.rint(
            np.clip(depth_m[valid] * 1000.0, 1.0, np.iinfo(np.uint16).max)
        ).astype(np.uint16)
        return depth_mm

    def _depth_to_rgb(self, depth_mm: np.ndarray) -> np.ndarray:
        if self._depth_video_max_meters <= 0:
            raise ValueError("depth_video_max_meters must be positive")
        normalized = np.clip(
            depth_mm.astype(np.float32) / (self._depth_video_max_meters * 1000.0),
            0.0,
            1.0,
        )
        gray = np.rint(normalized * 255.0).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)

    def _add_images_to_frame_data(self, frame_data: dict) -> None:
        self._pending_depth_frames = {}
        if self.latest_image_msg is None:
            return
        images = self.latest_image_msg["images"]
        timestamps = self.latest_image_msg.get("timestamps", {})
        source_by_feature = {
            DEPTH_CAMERA_RGB_FEATURE: self._depth_camera_rgb_image_key,
            DEPTH_CAMERA_FEATURE: self._depth_camera_image_key,
            WEBCAM_FEATURE: self._webcam_image_key,
            HEAD_DEPTH_CAMERA_FEATURE: self._head_depth_camera_image_key,
        }
        for feature_name, feature_info in self.data_exporter.features.items():
            if feature_info.get("dtype") not in ("image", "video"):
                continue
            if feature_name == EXTERNAL_VIEW_CAMERA_FEATURE:
                continue
            image_key = source_by_feature.get(feature_name, feature_name.split(".")[-1])
            if image_key not in images:
                raise ValueError(
                    f"Required image '{image_key}' for feature '{feature_name}' not found. "
                    f"Available: {list(images.keys())}"
                )
            image = images[image_key]
            if feature_name in (DEPTH_CAMERA_FEATURE, HEAD_DEPTH_CAMERA_FEATURE):
                stream_name = (
                    "ego_view" if feature_name == DEPTH_CAMERA_FEATURE else "head"
                )
                depth_mm = self._depth_to_mm(image)
                self._pending_depth_frames[stream_name] = (
                    depth_mm,
                    timestamps.get(image_key),
                )
                image = self._depth_to_rgb(depth_mm)
            frame_data[feature_name] = image

        if self._external_view_camera is not None:
            frame_data[EXTERNAL_VIEW_CAMERA_FEATURE] = (
                self._external_view_camera.latest_frame()
            )

        # Raw depth can be enabled without adding the visualization video to
        # the dataset schema, so prepare it separately in that configuration.
        if self._record_raw_depth:
            depth_sources = {
                "ego_view": self._depth_camera_image_key,
                "head": self._head_depth_camera_image_key,
            }
            for stream_name in self._enabled_camera_streams:
                image_key = depth_sources[stream_name]
                if stream_name in self._pending_depth_frames:
                    continue
                if image_key not in images:
                    raise ValueError(
                        f"Required raw depth image '{image_key}' not found. "
                        f"Available: {list(images.keys())}"
                    )
                self._pending_depth_frames[stream_name] = (
                    self._depth_to_mm(images[image_key]),
                    timestamps.get(image_key),
                )

    def _finalize_frame(self, t_start: float) -> bool:
        skipped_empty_episode = (
            self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE
            and self.data_exporter.episode_buffer.get("size", 0) == 0
        )
        result = super()._finalize_frame(t_start)
        if skipped_empty_episode:
            self.data_exporter.clear_episode_event_timestamps()
            self.data_exporter.clear_episode_raw_telemetry()
        return result

    def _add_data_frame(self):
        self._telemetry_receiver.poll()
        raw_samples = self._telemetry_receiver.drain_raw()
        if self._record_raw_telemetry and (
            self._episode_state.get_state() == self._episode_state.RECORDING
        ):
            # A very short episode can start between publisher ticks. Preserve
            # the same fresh sample used for the aligned frame in that case.
            if not raw_samples and self.data_exporter.raw_telemetry_sample_count() == 0:
                latest = self._telemetry_receiver.latest(self._telemetry_max_age)
                raw_samples = self._telemetry_receiver.drain_raw()
                if not raw_samples and latest is not None:
                    raw_samples = [latest]
            if raw_samples:
                self.data_exporter.add_raw_telemetry_samples(raw_samples)
        return super()._add_data_frame()

    def _add_g1_telemetry_features(self, frame_data: dict) -> bool:
        sample = self._telemetry_receiver.latest(self._telemetry_max_age)
        if sample is None:
            return False
        for field in G1_DATASET_ACTION_FIELDS:
            frame_data[f"action.upper_body.{field}"] = sample[field].copy()
        for field in G1_DATASET_OBSERVATION_FIELDS:
            frame_data[f"observation.upper_body.{field}"] = sample[field].copy()
        frame_data["observation.upper_body.source_age_sec"] = np.asarray(
            [sample["state_age_sec"], sample["command_age_sec"]], dtype=np.float32
        )
        frame_data["observation.upper_body.telemetry_timestamps_ns"] = np.asarray(
            [sample["source_timestamp_ns"], sample["host_receive_timestamp_ns"]],
            dtype=np.int64,
        )
        frame_data["observation.upper_body.telemetry_sequence_id"] = np.asarray(
            [sample["sequence_id"]], dtype=np.int64
        )
        return True

    def _add_data_frame_sonic(self, t_start: float) -> bool:
        assert self.latest_proprio_msg is not None
        proprio = self.latest_proprio_msg
        snapshot = self._brainco_subscriber.snapshot()

        left_state = self._fresh(snapshot, "left_state")
        right_state = self._fresh(snapshot, "right_state")
        left_tau_est = self._fresh(snapshot, "left_tau_est")
        right_tau_est = self._fresh(snapshot, "right_tau_est")
        if any(
            value is None
            for value in (left_state, right_state, left_tau_est, right_tau_est)
        ):
            self._print_and_say(
                "Waiting for fresh BrainCo left/right q and tau_est", say=False
            )
            return False

        g1_frame_data = {}
        if not self._add_g1_telemetry_features(g1_frame_data):
            self._print_and_say("Waiting for fresh G1 upper-body telemetry", say=False)
            return False

        # Commands are sparse events, not periodic state.  Hold the latest target
        # indefinitely; before the first command for a hand, use its measured
        # position so an idle hand neither blocks collection nor gets a bogus
        # zero-valued action.
        left_action = snapshot.get("left_command")
        right_action = snapshot.get("right_command")
        if left_action is None:
            left_action = left_state
        if right_action is None:
            right_action = right_state
        left_action = np.asarray(left_action, dtype=np.float32)
        right_action = np.asarray(right_action, dtype=np.float32)

        whole_q = self._brainco_schema.assemble(proprio["body_q"], left_state, right_state)
        whole_action_wbc = self._brainco_schema.assemble(
            proprio["last_action"], left_action, right_action
        )

        fk_q = self._brainco_schema.body_configuration_for_fk(proprio["body_q"])
        self.robot_model.cache_forward_kinematics(fk_q)
        eef_parts = []
        for side in ("left", "right"):
            placement = self.robot_model.frame_placement(
                self.robot_model.supplemental_info.hand_frame_names[side]
            )
            quat = R.from_matrix(placement.rotation).as_quat(scalar_first=True)
            eef_parts.append(np.concatenate([placement.translation[:3], quat]))

        frame_data = {
            "observation.state": whole_q,
            "observation.eef_state": np.concatenate(eef_parts),
            "observation.left_hand.tau_est": left_tau_est.copy(),
            "observation.right_hand.tau_est": right_tau_est.copy(),
            "action.wbc": whole_action_wbc,
            **g1_frame_data,
        }
        self._add_cpp_state_features(frame_data, proprio)
        sonic_latency_ms = self._add_sonic_pose_features(frame_data)

        # Teleop hand targets are the actual DDS commands sent to the service.
        frame_data["teleop.left_hand_joints"] = left_action.copy()
        frame_data["teleop.right_hand_joints"] = right_action.copy()

        self._add_images_to_frame_data(frame_data)
        self._log_latency_periodic(sonic_latency_ms)
        self.data_exporter.add_frame(frame_data)
        if self._record_raw_depth:
            for stream_name in self._enabled_camera_streams:
                pending = self._pending_depth_frames.get(stream_name)
                if pending is None:
                    raise RuntimeError(
                        f"Depth frame for {stream_name} was not prepared for raw PNG16 storage"
                    )
                depth_mm, camera_timestamp = pending
                self.data_exporter.add_raw_depth_frame(
                    depth_mm,
                    camera_timestamp,
                    stream_name=stream_name,
                )
        return self._finalize_frame(t_start)

    def save_and_cleanup(self):
        try:
            super().save_and_cleanup()
        finally:
            try:
                self._brainco_subscriber.close()
            finally:
                try:
                    self._telemetry_receiver.close()
                finally:
                    if self._external_view_camera is not None:
                        self._external_view_camera.close()
