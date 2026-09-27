"""Exact low-latency G1 SONIC encoder input and inference contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import numbers
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gear_sonic.data.unitree_conversion import provenance
from gear_sonic.data.unitree_conversion.contracts import ArtifactSpec, ResampledEpisode
from gear_sonic.data.unitree_conversion.joint_mapping import (
    G1_ISAACLAB_NAMES,
    reorder_by_name,
)
from gear_sonic.data.unitree_conversion.quaternion import validate_wxyz

_INPUT_NAME = "obs_dict"
_INPUT_SHAPE = (1, 1247)
_OUTPUT_NAME = "encoded_tokens"
_OUTPUT_SHAPE = (1, 64)
_TENSOR_TYPE = "tensor(float)"
_WINDOW_LENGTH = 10

_ENCODER_OBSERVATIONS = (
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step1", 290),
    ("motion_joint_velocities_10frame_step1", 290),
    ("motion_anchor_orientation_10frame_step1", 60),
    ("motion_anchor_orientation", 6),
    ("motion_joint_positions_lowerbody_10frame_step1", 120),
    ("motion_joint_velocities_lowerbody_10frame_step1", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_4frame_step1", 288),
    ("smpl_anchor_orientation_4frame_step1", 24),
    ("motion_joint_positions_wrists_4frame_step1", 24),
)
_G1_REQUIRED_OBSERVATIONS = tuple(name for name, _ in _ENCODER_OBSERVATIONS[:4])


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable YAML mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate YAML mapping key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a YAML mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must use string keys")
    return value


def _sequence(value: object, *, field_name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a YAML sequence")
    return value


def _parse_observation_config(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("observation config must be UTF-8") from error
    loader = _UniqueKeySafeLoader(text)
    try:
        document = loader.get_single_data()
    except yaml.YAMLError as error:
        raise ValueError(f"malformed observation config YAML: {error}") from error
    finally:
        loader.dispose()

    root = _mapping(document, field_name="observation config")
    if set(root) != {"observations", "encoder"}:
        raise ValueError("observation config must contain exactly observations and encoder")
    encoder = _mapping(root["encoder"], field_name="encoder")
    if set(encoder) != {"dimension", "use_fp16", "encoder_observations", "encoder_modes"}:
        raise ValueError("encoder config fields do not match the pinned low-latency config")
    if type(encoder["dimension"]) is not int or encoder["dimension"] != 64:
        raise ValueError("encoder dimension must be exactly integer 64")
    if encoder["use_fp16"] is not False:
        raise ValueError("encoder use_fp16 must be false for the CPU acceptance reference")

    configured_observations = _sequence(
        encoder["encoder_observations"],
        field_name="encoder_observations",
    )
    observation_names: list[str] = []
    for index, value in enumerate(configured_observations):
        observation = _mapping(value, field_name=f"encoder_observations[{index}]")
        if set(observation) != {"name", "enabled"} or observation["enabled"] is not True:
            raise ValueError("encoder_observations must use exact enabled name entries")
        if not isinstance(observation["name"], str):
            raise ValueError("encoder_observations names must be strings")
        observation_names.append(observation["name"])

    expected_names = tuple(name for name, _ in _ENCODER_OBSERVATIONS)
    if tuple(observation_names) != expected_names:
        raise ValueError("encoder_observations order must match the pinned 1,247D layout")
    if sum(size for _, size in _ENCODER_OBSERVATIONS) != _INPUT_SHAPE[1]:
        raise RuntimeError("internal pinned encoder observation dimensions are inconsistent")

    modes = _sequence(encoder["encoder_modes"], field_name="encoder_modes")
    g1_modes = []
    for index, value in enumerate(modes):
        mode = _mapping(value, field_name=f"encoder_modes[{index}]")
        if mode.get("name") == "g1":
            g1_modes.append(mode)
    if len(g1_modes) != 1:
        raise ValueError("encoder_modes must contain exactly one G1 mode")
    g1_mode = g1_modes[0]
    if set(g1_mode) != {"name", "mode_id", "required_observations"}:
        raise ValueError("G1 encoder mode fields do not match the pinned config")
    if type(g1_mode["mode_id"]) is not int or g1_mode["mode_id"] != 0:
        raise ValueError("G1 mode_id must be exactly integer 0")
    required = _sequence(g1_mode["required_observations"], field_name="G1 required_observations")
    if tuple(required) != _G1_REQUIRED_OBSERVATIONS:
        raise ValueError("G1 required_observations must match the exact four-observation order")


def _window(value: object, *, field_name: str, shape: tuple[int, int]) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a real numeric array") from error
    if source.dtype.kind not in "iuf":
        raise ValueError(f"{field_name} must be a real numeric array")
    if source.shape != shape:
        raise ValueError(f"{field_name} must have shape [{shape[0]},{shape[1]}]; got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError(f"{field_name} must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        owned = np.array(source, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(owned).all():
        raise ValueError(f"{field_name} float32 conversion must remain finite")
    return owned


def build_g1_encoder_input(
    positions: object,
    velocities: object,
    orientations: object,
) -> np.ndarray:
    """Pack the active G1 observations and exact inactive zeros into ``[1,1247]``."""
    position_window = _window(positions, field_name="positions", shape=(10, 29))
    velocity_window = _window(velocities, field_name="velocities", shape=(10, 29))
    orientation_window = _window(orientations, field_name="orientations", shape=(10, 6))

    tensor = np.zeros(_INPUT_SHAPE, dtype=np.float32, order="C")
    tensor[0, 4:294] = position_window.reshape(-1)
    tensor[0, 294:584] = velocity_window.reshape(-1)
    tensor[0, 584:644] = orientation_window.reshape(-1)
    return tensor


def _unit_internal(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("internal heading quaternion is not normalizable")
    return np.array([w / norm, x / norm, y / norm, z / norm], dtype=np.float64)


def _heading_normalized(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    rotated_x = ((2.0 * w * w - 1.0) + 0.0) + x * x * 2.0
    rotated_y = (0.0 + z * w * 2.0) + y * x * 2.0
    half_heading = math.atan2(rotated_y, rotated_x) / 2.0
    return _unit_internal(
        np.array(
            [math.cos(half_heading), 0.0, 0.0, math.sin(half_heading)],
            dtype=np.float64,
        )
    )


def _conjugate_normalized(quaternion: np.ndarray) -> np.ndarray:
    return np.array(
        [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
        dtype=np.float64,
    )


def _multiply_normalized(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    return _unit_internal(
        np.array(
            [
                qq - ww + (z1 - y1) * (y2 - z2),
                qq - xx + (x1 + w1) * (x2 + w2),
                qq - yy + (w1 - x1) * (y2 + z2),
                qq - zz + (z1 + y1) * (w2 - x2),
            ],
            dtype=np.float64,
        )
    )


def _rot6d_normalized(quaternion: np.ndarray) -> np.ndarray:
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
    )


def _build_encoder_orientation_window_normalized(
    initial_observed: np.ndarray,
    current_observed: np.ndarray,
    initial_reference: np.ndarray,
    future_reference: np.ndarray,
) -> np.ndarray:
    q_apply = _multiply_normalized(
        _heading_normalized(initial_observed),
        _conjugate_normalized(_heading_normalized(initial_reference)),
    )
    current_inverse = _conjugate_normalized(current_observed)
    orientations = np.empty((_WINDOW_LENGTH, 6), dtype=np.float64, order="C")
    for index in range(_WINDOW_LENGTH):
        aligned = _multiply_normalized(q_apply, future_reference[index])
        relative = _multiply_normalized(current_inverse, aligned)
        orientations[index] = _rot6d_normalized(relative)
    return np.array(orientations, dtype=np.float64, order="C", copy=True)


def build_encoder_orientation_window(
    initial_observed_root_wxyz: object,
    current_observed_root_wxyz: object,
    initial_reference_root_wxyz: object,
    future_reference_root_wxyz: object,
) -> np.ndarray:
    """Build ten deployed heading-aligned, current-relative row-wise Rot6D values."""
    initial_observed, _ = validate_wxyz(initial_observed_root_wxyz)
    current_observed, _ = validate_wxyz(current_observed_root_wxyz)
    initial_reference, _ = validate_wxyz(initial_reference_root_wxyz)

    try:
        future_source = np.asarray(future_reference_root_wxyz)
    except (TypeError, ValueError) as error:
        raise ValueError("future_reference_root_wxyz must be a numeric array with shape [10,4]") from error
    if future_source.shape != (_WINDOW_LENGTH, 4):
        raise ValueError(f"future_reference_root_wxyz must have shape [10,4]; got {future_source.shape}")
    if future_source.dtype.kind not in "iuf":
        raise ValueError("future_reference_root_wxyz must be a numeric array with shape [10,4]")
    future = np.empty((_WINDOW_LENGTH, 4), dtype=np.float64, order="C")
    for index, quaternion in enumerate(future_source):
        future[index], _ = validate_wxyz(quaternion)

    return _build_encoder_orientation_window_normalized(
        initial_observed,
        current_observed,
        initial_reference,
        future,
    )


def build_frame_encoder_input(episode: ResampledEpisode, frame_index: object) -> np.ndarray:
    """Build one G1 encoder tensor using an episode-local clamped future window."""
    if not isinstance(episode, ResampledEpisode):
        raise ValueError("episode must be a ResampledEpisode")
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, numbers.Integral)
        or not 0 <= int(frame_index) < episode.frame_count
    ):
        raise ValueError(f"frame_index must be an integer in [0,{episode.frame_count})")
    index = int(frame_index)
    future_indices = np.minimum(
        index + np.arange(_WINDOW_LENGTH, dtype=np.int64),
        episode.frame_count - 1,
    )
    positions = reorder_by_name(
        episode.desired_body_q[future_indices],
        episode.body_joint_names,
        G1_ISAACLAB_NAMES,
    )
    velocities = reorder_by_name(
        episode.desired_body_velocity[future_indices],
        episode.body_joint_names,
        G1_ISAACLAB_NAMES,
    )
    orientations = _build_encoder_orientation_window_normalized(
        episode.observed_root_wxyz[0],
        episode.observed_root_wxyz[index],
        episode.reference_root_wxyz[0],
        episode.reference_root_wxyz[future_indices],
    )
    return build_g1_encoder_input(positions, velocities, orientations)


class SonicEncoder:
    """Validated ONNX Runtime wrapper for the immutable 64D SONIC encoder."""

    def __init__(
        self,
        session: object,
        *,
        encoder_path: Path | None = None,
        observation_config_path: Path | None = None,
        provider: str | None = None,
        runtime_version: str | None = None,
    ) -> None:
        self._session = session
        self.encoder_path = encoder_path
        self.observation_config_path = observation_config_path
        self.provider = provider
        self.runtime_version = runtime_version

    @staticmethod
    def _validate_tensor_metadata(
        tensors: object,
        *,
        kind: str,
        expected_name: str,
        expected_shape: tuple[int, int],
    ) -> None:
        if not isinstance(tensors, Sequence) or isinstance(tensors, (str, bytes)) or len(tensors) != 1:
            raise ValueError(f"encoder session must expose exactly one {kind}")
        tensor = tensors[0]
        if getattr(tensor, "name", None) != expected_name:
            raise ValueError(f"encoder {kind} name must be {expected_name!r}")
        shape = getattr(tensor, "shape", None)
        if (
            not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes))
            or tuple(shape) != expected_shape
            or any(type(dimension) is not int for dimension in shape)
        ):
            raise ValueError(f"encoder {kind} shape must be {list(expected_shape)}")
        if getattr(tensor, "type", None) != _TENSOR_TYPE:
            raise ValueError(f"encoder {kind} type must be {_TENSOR_TYPE!r}")

    @classmethod
    def from_session(
        cls,
        session: object,
        *,
        encoder_path: Path | None = None,
        observation_config_path: Path | None = None,
        provider: str | None = None,
        runtime_version: str | None = None,
    ) -> SonicEncoder:
        """Validate the exact ONNX tensor metadata before accepting a session."""
        try:
            inputs = session.get_inputs()  # type: ignore[attr-defined]
            outputs = session.get_outputs()  # type: ignore[attr-defined]
        except AttributeError as error:
            raise ValueError("session must expose ONNX Runtime input and output metadata") from error
        cls._validate_tensor_metadata(
            inputs,
            kind="input",
            expected_name=_INPUT_NAME,
            expected_shape=_INPUT_SHAPE,
        )
        cls._validate_tensor_metadata(
            outputs,
            kind="output",
            expected_name=_OUTPUT_NAME,
            expected_shape=_OUTPUT_SHAPE,
        )
        return cls(
            session,
            encoder_path=encoder_path,
            observation_config_path=observation_config_path,
            provider=provider,
            runtime_version=runtime_version,
        )

    @classmethod
    def from_artifacts(cls, lock: object, cache_dir: str | Path | None = None) -> SonicEncoder:
        """Materialize pinned artifacts, validate config/model contracts, and use CPU ORT."""
        encoder_spec = getattr(lock, "encoder", None)
        config_spec = getattr(lock, "observation_config", None)
        if not isinstance(encoder_spec, ArtifactSpec) or not isinstance(config_spec, ArtifactSpec):
            raise ValueError("lock must expose encoder and observation_config ArtifactSpec values")

        cache_path: Path | None = None
        if cache_dir is not None:
            try:
                cache_path = Path(cache_dir)
            except TypeError as error:
                raise ValueError("cache_dir must be path-like or None") from error

        from huggingface_hub import hf_hub_download

        def downloader(**kwargs: object) -> str | Path:
            if cache_path is not None:
                kwargs["cache_dir"] = str(cache_path)
            return hf_hub_download(**kwargs)

        encoder_path = provenance.materialize_artifact(encoder_spec, downloader=downloader)
        observation_config_path = provenance.materialize_artifact(config_spec, downloader=downloader)
        _parse_observation_config(observation_config_path)

        import onnxruntime

        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        provider = "CPUExecutionProvider"
        session = onnxruntime.InferenceSession(
            str(encoder_path),
            providers=[provider],
            sess_options=session_options,
        )
        return cls.from_session(
            session,
            encoder_path=encoder_path,
            observation_config_path=observation_config_path,
            provider=provider,
            runtime_version=str(onnxruntime.__version__),
        )

    def encode(self, tensor: object) -> np.ndarray:
        """Run exact-rank float32 inference and return an owned ``[1,64]`` token."""
        if not isinstance(tensor, np.ndarray) or tensor.shape != _INPUT_SHAPE:
            shape = getattr(tensor, "shape", None)
            raise ValueError(f"encoder input must have shape [1,1247]; got {shape}")
        if tensor.dtype != np.dtype(np.float32):
            raise ValueError("encoder input dtype must be float32")
        if not tensor.flags.c_contiguous:
            raise ValueError("encoder input must be C-contiguous")
        if not tensor.flags.owndata:
            raise ValueError("encoder input must own its memory")
        if not np.isfinite(tensor).all():
            raise ValueError("encoder input must contain only finite values")

        outputs = self._session.run([_OUTPUT_NAME], {_INPUT_NAME: tensor})
        if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)) or len(outputs) != 1:
            raise ValueError("encoder runtime must return exactly one output")
        output = outputs[0]
        if not isinstance(output, np.ndarray) or output.shape != _OUTPUT_SHAPE:
            shape = getattr(output, "shape", None)
            raise ValueError(f"encoder output must have shape [1,64]; got {shape}")
        if output.dtype != np.dtype(np.float32):
            raise ValueError("encoder output dtype must be float32")
        if not np.isfinite(output).all():
            raise ValueError("encoder output must contain only finite values")
        return np.array(output, dtype=np.float32, order="C", copy=True)
