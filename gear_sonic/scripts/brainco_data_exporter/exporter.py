"""Episode persistence with controller events and raw depth."""

import json
from pathlib import Path

import cv2
import numpy as np

from gear_sonic.data.exporter import Gr00tDataExporter

from .constants import (
    DEPTH_CAMERA_RGB_FEATURE,
    G1_RAW_TELEMETRY_ARRAY_FIELDS,
    WEBCAM_FEATURE,
)
from gear_sonic.g1_upper_body_telemetry.protocol import (
    JOINT_INDICES as TELEMETRY_JOINT_INDICES,
    JOINT_NAMES as TELEMETRY_JOINT_NAMES,
    SCHEMA_VERSION as TELEMETRY_SCHEMA_VERSION,
)

class BraincoDataExporter(Gr00tDataExporter):
    """Exporter that adds controller-event timestamps to ``episodes.jsonl``."""

    GRIP_TIMESTAMPS_KEY = "grip_timestamps"
    INTERACTION_METADATA_REL_PATH = Path("meta/interaction_metadata.jsonl")
    EMPTY_PHASE_TIMESTAMPS = {
        "approach_start": 0.0,
        "contact_active_start": 0.0,
        "release_start": 0.0,
        "idle_start": 0.0,
    }
    TRIGGER_TIMESTAMP_KEYS = {
        ("left", "press"): "left_trigger_press_timestamps",
        ("left", "release"): "left_trigger_release_timestamps",
        ("right", "press"): "right_trigger_press_timestamps",
        ("right", "release"): "right_trigger_release_timestamps",
    }

    def add_grip_timestamp(self, timestamp_sec: float) -> None:
        timestamps = getattr(self, "_grip_timestamps", None)
        if timestamps is None:
            timestamps = []
            self._grip_timestamps = timestamps
        timestamps.append(round(max(0.0, float(timestamp_sec)), 6))

    def add_trigger_timestamp(self, side: str, event: str, timestamp_sec: float) -> None:
        key = self.TRIGGER_TIMESTAMP_KEYS.get((side, event))
        if key is None:
            raise ValueError(f"Invalid trigger event: side={side!r}, event={event!r}")
        timestamps = getattr(self, "_trigger_timestamps", None)
        if timestamps is None:
            timestamps = {name: [] for name in self.TRIGGER_TIMESTAMP_KEYS.values()}
            self._trigger_timestamps = timestamps
        timestamps[key].append(round(max(0.0, float(timestamp_sec)), 6))

    def clear_episode_event_timestamps(self) -> None:
        self._grip_timestamps = []
        self._trigger_timestamps = {
            name: [] for name in self.TRIGGER_TIMESTAMP_KEYS.values()
        }

    def add_raw_telemetry_samples(self, samples: list[dict]) -> None:
        if not getattr(self, "record_raw_telemetry", True):
            return
        stored = getattr(self, "_raw_telemetry_samples", None)
        if stored is None:
            stored = []
            self._raw_telemetry_samples = stored
        stored.extend(samples)

    def clear_episode_raw_telemetry(self) -> None:
        self._raw_telemetry_samples = []

    def raw_telemetry_sample_count(self) -> int:
        return len(getattr(self, "_raw_telemetry_samples", []))

    def _write_raw_telemetry(self, episode_index: int, samples: list[dict]) -> str:
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        relative_path = (
            Path("raw-telemetry")
            / f"chunk-{episode_chunk:03d}"
            / f"episode_{episode_index:06d}.npz"
        )
        output_path = Path(self.root) / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")

        arrays = {
            field: np.stack([sample[field] for sample in samples]).astype(np.float32)
            for field in G1_RAW_TELEMETRY_ARRAY_FIELDS
        }
        for field in (
            "sequence_id", "source_timestamp_ns", "source_monotonic_ns",
            "host_receive_timestamp_ns", "host_receive_monotonic_ns",
        ):
            arrays[field] = np.asarray([sample[field] for sample in samples], dtype=np.int64)
        for field in ("state_age_sec", "command_age_sec", "publish_hz"):
            arrays[field] = np.asarray([sample[field] for sample in samples], dtype=np.float64)
        arrays["schema_version"] = np.asarray([TELEMETRY_SCHEMA_VERSION], dtype=np.int64)
        arrays["joint_indices"] = np.asarray(TELEMETRY_JOINT_INDICES, dtype=np.int64)
        arrays["joint_names"] = np.asarray(TELEMETRY_JOINT_NAMES)
        with temporary_path.open("wb") as dst:
            np.savez_compressed(dst, **arrays)
        temporary_path.replace(output_path)
        return str(relative_path)

    def add_raw_depth_frame(
        self,
        depth_mm: np.ndarray,
        camera_timestamp: float | None,
        *,
        stream_name: str = "ego_view",
    ) -> None:
        depth_mm = np.asarray(depth_mm)
        if depth_mm.ndim != 2 or depth_mm.dtype != np.uint16:
            raise ValueError(
                f"Raw depth must be an HxW uint16 millimetre image, got "
                f"shape={depth_mm.shape}, dtype={depth_mm.dtype}"
            )

        episode_index = int(self.episode_buffer["episode_index"])
        frame_index = int(self.episode_buffer["size"]) - 1
        if frame_index < 0:
            raise RuntimeError("Raw depth must be written after the corresponding dataset frame")
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        depth_root = "depth" if stream_name == "ego_view" else f"depth_{stream_name}"
        relative_dir = (
            Path(depth_root)
            / f"chunk-{episode_chunk:03d}"
            / f"episode_{episode_index:06d}"
        )
        output_dir = Path(self.root) / relative_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"frame_{frame_index:06d}.png"
        temporary_path = output_dir / f".frame_{frame_index:06d}.tmp.png"
        if not cv2.imwrite(str(temporary_path), depth_mm):
            raise RuntimeError(f"Failed to write raw depth PNG: {output_path}")
        temporary_path.replace(output_path)

        frames_by_stream = getattr(self, "_raw_depth_frames", None)
        if frames_by_stream is None:
            frames_by_stream = {}
            self._raw_depth_frames = frames_by_stream
        frames_by_stream.setdefault(stream_name, []).append(
            {
                "frame_index": frame_index,
                "camera_timestamp": (
                    None if camera_timestamp is None else float(camera_timestamp)
                ),
                "path": str(relative_dir / output_path.name),
            }
        )

    def _write_episode_metadata(self, episode_index: int, metadata: dict) -> None:
        episodes_path = Path(self.root) / "meta" / "episodes.jsonl"
        records = []
        found = False
        with episodes_path.open(encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record.get("episode_index", -1)) == int(episode_index):
                    record.update(metadata)
                    found = True
                records.append(record)

        if not found:
            raise RuntimeError(
                f"Episode {episode_index} was saved but is missing from {episodes_path}"
            )

        temporary_path = episodes_path.with_name(f".{episodes_path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as dst:
            for record in records:
                dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary_path.replace(episodes_path)

        # Keep the in-memory metadata consistent with the JSONL file.
        if episode_index in self.meta.episodes:
            self.meta.episodes[episode_index].update(metadata)

    def _write_interaction_metadata(self, episode_index: int) -> None:
        profile = getattr(self, "session_profile", None)
        if profile is None:
            raise RuntimeError("Session profile was not configured")

        record = {
            "episode_index": int(episode_index),
            "interaction_class": profile["interaction_class"],
            "sex": profile["sex"],
            "age": profile["age"],
            "height_cm": profile["height_cm"],
            "weight_kg": profile["weight_kg"],
            "phase_timestamps": dict(self.EMPTY_PHASE_TIMESTAMPS),
        }
        metadata_path = Path(self.root) / self.INTERACTION_METADATA_REL_PATH
        records = []
        replaced = False
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    existing = json.loads(line)
                    if int(existing.get("episode_index", -1)) == int(episode_index):
                        records.append(record)
                        replaced = True
                    else:
                        records.append(existing)
        if not replaced:
            records.append(record)
        records.sort(key=lambda item: int(item["episode_index"]))

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as dst:
            for item in records:
                dst.write(
                    json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        temporary_path.replace(metadata_path)

    def _write_depth_timestamps(
        self, episode_index: int, frames: list[dict], stream_name: str = "ego_view"
    ) -> tuple[str, str]:
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        depth_root = "depth" if stream_name == "ego_view" else f"depth_{stream_name}"
        relative_dir = (
            Path(depth_root)
            / f"chunk-{episode_chunk:03d}"
            / f"episode_{episode_index:06d}"
        )
        output_dir = Path(self.root) / relative_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamps_path = output_dir / "timestamps.jsonl"
        temporary_path = output_dir / ".timestamps.jsonl.tmp"
        with temporary_path.open("w", encoding="utf-8") as dst:
            for frame in frames:
                dst.write(json.dumps(frame, separators=(",", ":")) + "\n")
        temporary_path.replace(timestamps_path)
        return str(relative_dir), str(relative_dir / timestamps_path.name)

    def save_episode(self, episode_data: dict | None = None) -> None:
        episode_buffer = self.episode_buffer if not episode_data else episode_data
        episode_index = int(episode_buffer["episode_index"])
        episode_length = int(episode_buffer["size"])
        record_raw_depth = getattr(self, "record_raw_depth", True)
        grip_timestamps = list(getattr(self, "_grip_timestamps", []))
        stored_trigger_timestamps = getattr(self, "_trigger_timestamps", {})
        trigger_timestamps = {
            key: list(stored_trigger_timestamps.get(key, []))
            for key in self.TRIGGER_TIMESTAMP_KEYS.values()
        }
        depth_frames_by_stream = dict(getattr(self, "_raw_depth_frames", {}))
        record_raw_telemetry = getattr(self, "record_raw_telemetry", True)
        raw_telemetry_samples = list(getattr(self, "_raw_telemetry_samples", []))
        enabled_camera_streams = tuple(
            getattr(self, "enabled_camera_streams", ("ego_view", "head"))
        )
        if record_raw_depth:
            for stream_name in enabled_camera_streams:
                frame_count = len(depth_frames_by_stream.get(stream_name, []))
                if frame_count != episode_length:
                    raise RuntimeError(
                        f"Raw depth frame count for {stream_name} ({frame_count}) "
                        f"does not match episode length ({episode_length})"
                    )
        if record_raw_telemetry and not raw_telemetry_samples:
            raise RuntimeError("No raw G1 telemetry samples were collected for the episode")
        telemetry_metadata = None
        if record_raw_telemetry:
            telemetry_path = self._write_raw_telemetry(
                episode_index, raw_telemetry_samples
            )
            source_times = np.asarray(
                [sample["source_monotonic_ns"] for sample in raw_telemetry_samples],
                dtype=np.int64,
            )
            sequences = np.asarray(
                [sample["sequence_id"] for sample in raw_telemetry_samples],
                dtype=np.int64,
            )
            duration_sec = (
                0.0
                if len(source_times) < 2
                else max(0.0, (source_times[-1] - source_times[0]) * 1e-9)
            )
            sequence_diffs = np.diff(sequences)
            telemetry_metadata = {
                "path": telemetry_path,
                "encoding": "numpy_npz_compressed",
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "array_fields": list(G1_RAW_TELEMETRY_ARRAY_FIELDS),
                "sample_count": len(raw_telemetry_samples),
                "target_hz": float(raw_telemetry_samples[0]["publish_hz"]),
                "duration_sec": duration_sec,
                "measured_hz": (
                    0.0
                    if duration_sec <= 0
                    else (len(raw_telemetry_samples) - 1) / duration_sec
                ),
                "first_sequence_id": int(sequences[0]),
                "last_sequence_id": int(sequences[-1]),
                "missing_packet_count": int(
                    np.maximum(sequence_diffs - 1, 0).sum()
                ),
                "publisher_restart_count": int((sequence_diffs <= 0).sum()),
            }

        super().save_episode(episode_data)
        metadata = {
            self.GRIP_TIMESTAMPS_KEY: grip_timestamps,
            **trigger_timestamps,
            "raw_depth_recorded": record_raw_depth,
            "raw_telemetry_recorded": record_raw_telemetry,
        }
        if telemetry_metadata is not None:
            metadata["raw_telemetry"] = telemetry_metadata
        if record_raw_depth:
            raw_depth_streams = {}
            aligned_features = {
                "ego_view": DEPTH_CAMERA_RGB_FEATURE,
                "head": WEBCAM_FEATURE,
            }
            for stream_name in enabled_camera_streams:
                aligned_to = aligned_features[stream_name]
                depth_frames = depth_frames_by_stream[stream_name]
                depth_dir, depth_timestamps_path = self._write_depth_timestamps(
                    episode_index, depth_frames, stream_name
                )
                raw_depth_streams[stream_name] = {
                    "depth_png_dir": depth_dir,
                    "depth_timestamps_path": depth_timestamps_path,
                    "depth_aligned_to": aligned_to,
                    "depth_frame_count": len(depth_frames),
                }
            metadata.update(
                {
                    "depth_encoding": "uint16_mm_png",
                    "depth_unit_meters": 0.001,
                    "depth_invalid_value": 0,
                    "raw_depth_streams": raw_depth_streams,
                }
            )
            # Keep the original single-camera metadata aliases when ego-view exists.
            if "ego_view" in raw_depth_streams:
                metadata.update(raw_depth_streams["ego_view"])
        self._write_episode_metadata(
            episode_index,
            metadata,
        )
        self._write_interaction_metadata(episode_index)
        self.clear_episode_event_timestamps()
        self._raw_depth_frames = {}
        self.clear_episode_raw_telemetry()
