from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import warnings

import numpy as np
import pytest

from gear_sonic.data.unitree_conversion import sonic_encoder as sonic_encoder_module
from gear_sonic.data.unitree_conversion.contracts import ArtifactSpec, ResampledEpisode
from gear_sonic.data.unitree_conversion.joint_mapping import (
    G1_ISAACLAB_NAMES,
    G1_MUJOCO_NAMES,
    reorder_by_name,
)
from gear_sonic.data.unitree_conversion.provenance import load_source_lock, verify_file
from gear_sonic.data.unitree_conversion.quaternion import validate_wxyz
from gear_sonic.data.unitree_conversion.sonic_encoder import (
    SonicEncoder,
    build_encoder_orientation_window,
    build_frame_encoder_input,
    build_g1_encoder_input,
)

PINNED_CONFIG = b"""\
observations:
  - name: token_state
    enabled: true
encoder:
  dimension: 64
  use_fp16: false
  encoder_observations:
    - {name: encoder_mode_4, enabled: true}
    - {name: motion_joint_positions_10frame_step1, enabled: true}
    - {name: motion_joint_velocities_10frame_step1, enabled: true}
    - {name: motion_anchor_orientation_10frame_step1, enabled: true}
    - {name: motion_anchor_orientation, enabled: true}
    - {name: motion_joint_positions_lowerbody_10frame_step1, enabled: true}
    - {name: motion_joint_velocities_lowerbody_10frame_step1, enabled: true}
    - {name: vr_3point_local_target, enabled: true}
    - {name: vr_3point_local_orn_target, enabled: true}
    - {name: smpl_joints_4frame_step1, enabled: true}
    - {name: smpl_anchor_orientation_4frame_step1, enabled: true}
    - {name: motion_joint_positions_wrists_4frame_step1, enabled: true}
  encoder_modes:
    - name: g1
      mode_id: 0
      required_observations:
        - encoder_mode_4
        - motion_joint_positions_10frame_step1
        - motion_joint_velocities_10frame_step1
        - motion_anchor_orientation_10frame_step1
    - name: teleop
      mode_id: 1
      required_observations: []
"""

GOLDEN_DIR = Path("gear_sonic/tests/data/unitree_conversion")
GOLDEN_ENCODER_CASE_SHA256 = "3696b203ab720431ae50c693a1c27e72105de76a2e4048bc38b8fc01a48d9509"
GOLDEN_ENCODER_TOKEN_SHA256 = "d94d989f7d8d960653f76a78766b8a4ca8210f2b8e97010792d8189897b7ca2b"


def _yaw(angle: float) -> np.ndarray:
    return np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)], dtype=np.float64)


def _episode(frame_count: int = 12) -> ResampledEpisode:
    observed_roots = np.stack([_yaw(0.1 + index * 0.01) for index in range(frame_count)])
    reference_roots = np.stack([_yaw(-0.2 + index * 0.02) for index in range(frame_count)])
    body = np.arange(frame_count * 29, dtype=np.float64).reshape(frame_count, 29)
    hands = np.arange(frame_count * 7, dtype=np.float64).reshape(frame_count, 7)
    return ResampledEpisode(
        source_repo_id="synthetic/dex3",
        source_revision="a" * 40,
        source_episode_id=7,
        body_joint_names=G1_MUJOCO_NAMES,
        task_indices=np.arange(frame_count, dtype=np.int64) % 2,
        task_texts=tuple("pour" if index % 2 == 0 else "place" for index in range(frame_count)),
        observed_root_wxyz=observed_roots,
        reference_root_wxyz=reference_roots,
        observed_body_q=body,
        desired_body_q=body + 1000.0,
        desired_body_velocity=body + 2000.0,
        observed_left_hand=hands,
        observed_right_hand=hands + 100.0,
        desired_left_hand=hands + 200.0,
        desired_right_hand=hands + 300.0,
    )


def test_resampled_episode_owns_consistent_finite_normalized_50hz_arrays() -> None:
    source = _episode()
    inputs = {
        name: getattr(source, name)
        for name in (
            "task_indices",
            "observed_root_wxyz",
            "reference_root_wxyz",
            "observed_body_q",
            "desired_body_q",
            "desired_body_velocity",
            "observed_left_hand",
            "observed_right_hand",
            "desired_left_hand",
            "desired_right_hand",
        )
    }

    episode = ResampledEpisode(
        source_repo_id=source.source_repo_id,
        source_revision=source.source_revision,
        source_episode_id=source.source_episode_id,
        body_joint_names=source.body_joint_names,
        task_indices=inputs["task_indices"][:],
        task_texts=source.task_texts,
        observed_root_wxyz=inputs["observed_root_wxyz"] * 1.000005,
        reference_root_wxyz=inputs["reference_root_wxyz"] * 1.000005,
        observed_body_q=inputs["observed_body_q"][:, :],
        desired_body_q=inputs["desired_body_q"][:, :],
        desired_body_velocity=inputs["desired_body_velocity"][:, :],
        observed_left_hand=inputs["observed_left_hand"][:, :],
        observed_right_hand=inputs["observed_right_hand"][:, :],
        desired_left_hand=inputs["desired_left_hand"][:, :],
        desired_right_hand=inputs["desired_right_hand"][:, :],
    )

    assert episode.frame_count == 12
    assert episode.target_fps == 50
    assert episode.body_joint_names == G1_MUJOCO_NAMES
    for name in inputs:
        array = getattr(episode, name)
        assert array.flags.owndata
        assert array.flags.c_contiguous
        assert not np.shares_memory(array, inputs[name])
    np.testing.assert_allclose(np.linalg.norm(episode.observed_root_wxyz, axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(episode.reference_root_wxyz, axis=1), 1.0, atol=1e-15)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("desired_body_q", np.zeros((11, 29)), "desired_body_q.*12, 29"),
        ("desired_body_velocity", np.zeros((12, 28)), "desired_body_velocity.*12, 29"),
        ("desired_left_hand", np.zeros((12, 6)), "desired_left_hand.*12, 7"),
        ("reference_root_wxyz", np.zeros((12, 4)), "degenerate"),
        ("observed_body_q", np.full((12, 29), np.nan), "finite"),
    ],
)
def test_resampled_episode_rejects_inconsistent_or_invalid_arrays(
    field_name: str,
    value: np.ndarray,
    message: str,
) -> None:
    kwargs = _episode().__dict__.copy()
    kwargs.pop("target_fps", None)
    kwargs.pop("frame_count", None)
    kwargs[field_name] = value
    with pytest.raises(ValueError, match=message):
        ResampledEpisode(**kwargs)


def test_resampled_episode_requires_explicit_unique_body_order_and_frame_metadata() -> None:
    kwargs = _episode().__dict__.copy()
    kwargs["body_joint_names"] = (*G1_MUJOCO_NAMES[:-1], G1_MUJOCO_NAMES[0])
    with pytest.raises(ValueError, match="body_joint_names.*unique"):
        ResampledEpisode(**kwargs)

    kwargs = _episode().__dict__.copy()
    kwargs["task_texts"] = kwargs["task_texts"][:-1]
    with pytest.raises(ValueError, match="task_texts.*12"):
        ResampledEpisode(**kwargs)


def test_resampled_episode_rejects_unhashable_body_name_with_contract_error() -> None:
    kwargs = _episode().__dict__.copy()
    kwargs["body_joint_names"] = (*G1_MUJOCO_NAMES[:-1], ["not-a-name"])

    with pytest.raises(ValueError, match="body_joint_names.*unique semantic names"):
        ResampledEpisode(**kwargs)


def test_g1_encoder_layout_is_exactly_1247d() -> None:
    positions = np.arange(290, dtype=np.float32).reshape(10, 29)
    velocities = positions + 1000
    orientations = np.arange(60, dtype=np.float32).reshape(10, 6)
    tensor = build_g1_encoder_input(positions, velocities, orientations)
    assert tensor.shape == (1, 1247)
    assert tensor.dtype == np.float32
    assert tensor.flags.owndata
    assert tensor.flags.c_contiguous
    np.testing.assert_array_equal(tensor[0, :4], [0, 0, 0, 0])
    np.testing.assert_array_equal(tensor[0, 4:294], positions.reshape(-1))
    np.testing.assert_array_equal(tensor[0, 294:584], velocities.reshape(-1))
    np.testing.assert_array_equal(tensor[0, 584:644], orientations.reshape(-1))
    np.testing.assert_array_equal(tensor[0, 644:], 0)


@pytest.mark.parametrize(
    ("field_name", "positions", "velocities", "orientations", "message"),
    [
        (
            "positions",
            np.zeros((9, 29), dtype=np.float32),
            np.zeros((10, 29), dtype=np.float32),
            np.zeros((10, 6), dtype=np.float32),
            r"positions.*\[10,29\]",
        ),
        (
            "velocities",
            np.zeros((10, 29), dtype=np.float32),
            np.zeros((10, 28), dtype=np.float32),
            np.zeros((10, 6), dtype=np.float32),
            r"velocities.*\[10,29\]",
        ),
        (
            "orientations",
            np.zeros((10, 29), dtype=np.float32),
            np.zeros((10, 29), dtype=np.float32),
            np.zeros((9, 6), dtype=np.float32),
            r"orientations.*\[10,6\]",
        ),
    ],
)
def test_builder_rejects_wrong_window_shape(
    field_name: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    orientations: np.ndarray,
    message: str,
) -> None:
    del field_name
    with pytest.raises(ValueError, match=message):
        build_g1_encoder_input(positions, velocities, orientations)


def test_builder_rejects_non_numeric_or_nonfinite_window_values() -> None:
    valid_positions = np.zeros((10, 29), dtype=np.float32)
    valid_orientations = np.zeros((10, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="positions.*numeric"):
        build_g1_encoder_input(
            np.full((10, 29), "x"),
            valid_positions,
            valid_orientations,
        )
    invalid = valid_positions.copy()
    invalid[2, 3] = np.inf
    with pytest.raises(ValueError, match="velocities.*finite"):
        build_g1_encoder_input(valid_positions, invalid, valid_orientations)


@pytest.mark.parametrize("field_name", ["positions", "velocities", "orientations"])
def test_builder_rejects_float32_overflow_without_runtime_warning(field_name: str) -> None:
    windows = {
        "positions": np.zeros((10, 29), dtype=np.float64),
        "velocities": np.zeros((10, 29), dtype=np.float64),
        "orientations": np.zeros((10, 6), dtype=np.float64),
    }
    windows[field_name].flat[0] = np.finfo(np.float64).max

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match=rf"{field_name}.*float32.*finite"):
            build_g1_encoder_input(**windows)


def test_orientation_window_applies_initial_heading_and_current_full_root() -> None:
    future_yaws = np.linspace(-0.2, 0.7, 10)
    result = build_encoder_orientation_window(
        _yaw(0.1),
        _yaw(0.3),
        _yaw(-0.2),
        np.stack([_yaw(angle) for angle in future_yaws]),
    )

    expected_relative_yaws = future_yaws
    expected = np.array(
        [
            [math.cos(angle), -math.sin(angle), math.sin(angle), math.cos(angle), 0.0, 0.0]
            for angle in expected_relative_yaws
        ],
        dtype=np.float64,
    )
    assert result.shape == (10, 6)
    assert result.dtype == np.float64
    assert result.flags.owndata
    assert result.flags.c_contiguous
    np.testing.assert_allclose(result, expected, atol=1e-15, rtol=0.0)


def test_orientation_window_normalizes_each_external_root_exactly_once() -> None:
    def unit(quaternion: np.ndarray) -> np.ndarray:
        norm = math.sqrt(sum(float(value) * float(value) for value in quaternion))
        return np.array([float(value) / norm for value in quaternion], dtype=np.float64)

    def heading(quaternion: np.ndarray) -> np.ndarray:
        w, x, y, z = quaternion
        rotated_x = ((2.0 * w * w - 1.0) + 0.0) + x * x * 2.0
        rotated_y = (0.0 + z * w * 2.0) + y * x * 2.0
        half = math.atan2(rotated_y, rotated_x) / 2.0
        return unit(np.array([math.cos(half), 0.0, 0.0, math.sin(half)]))

    def conjugate(quaternion: np.ndarray) -> np.ndarray:
        return np.array(
            [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
            dtype=np.float64,
        )

    def multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = left
        w2, x2, y2, z2 = right
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        return unit(
            np.array(
                [
                    qq - ww + (z1 - y1) * (y2 - z2),
                    qq - xx + (x1 + w1) * (x2 + w2),
                    qq - yy + (w1 - x1) * (y2 + z2),
                    qq - zz + (z1 + y1) * (w2 - x2),
                ]
            )
        )

    def rot6d(quaternion: np.ndarray) -> np.ndarray:
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

    roots = np.random.default_rng(19).normal(size=(13, 4))
    roots /= np.linalg.norm(roots, axis=1)[:, None]
    roots *= (1.0 + np.linspace(-9e-6, 9e-6, 13))[:, None]
    normalized = np.stack([validate_wxyz(quaternion)[0] for quaternion in roots])
    q_apply = multiply(heading(normalized[0]), conjugate(heading(normalized[2])))
    current_inverse = conjugate(normalized[1])
    expected = np.stack(
        [rot6d(multiply(current_inverse, multiply(q_apply, quaternion))) for quaternion in normalized[3:]]
    )

    result = build_encoder_orientation_window(roots[0], roots[1], roots[2], roots[3:])

    np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))


def test_frame_builder_clamps_inside_one_episode_and_reorders_by_joint_name() -> None:
    episode = _episode(frame_count=3)

    tensor = build_frame_encoder_input(episode, frame_index=2)

    expected_positions = reorder_by_name(
        episode.desired_body_q[[2] * 10],
        G1_MUJOCO_NAMES,
        G1_ISAACLAB_NAMES,
    ).astype(np.float32)
    expected_velocities = reorder_by_name(
        episode.desired_body_velocity[[2] * 10],
        G1_MUJOCO_NAMES,
        G1_ISAACLAB_NAMES,
    ).astype(np.float32)
    np.testing.assert_array_equal(tensor[0, 4:294], expected_positions.reshape(-1))
    np.testing.assert_array_equal(tensor[0, 294:584], expected_velocities.reshape(-1))
    np.testing.assert_allclose(
        tensor[0, 584:644],
        build_encoder_orientation_window(
            episode.observed_root_wxyz[0],
            episode.observed_root_wxyz[2],
            episode.reference_root_wxyz[0],
            episode.reference_root_wxyz[[2] * 10],
        )
        .astype(np.float32)
        .reshape(-1),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_array_equal(tensor[0, 644:], 0)


def test_frame_builder_constructs_only_one_constant_size_future_index_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_full_table(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("full [T50,10] index table was constructed")

    monkeypatch.setattr(
        sonic_encoder_module,
        "clamped_future_indices",
        reject_full_table,
        raising=False,
    )

    tensor = build_frame_encoder_input(_episode(frame_count=12), frame_index=4)

    assert tensor.shape == (1, 1247)


@pytest.mark.parametrize("frame_index", [-1, 3, True, 1.0])
def test_frame_builder_rejects_invalid_frame_index(frame_index: object) -> None:
    with pytest.raises(ValueError, match="frame_index"):
        build_frame_encoder_input(_episode(frame_count=3), frame_index=frame_index)


class FakeSession:
    def __init__(
        self,
        *,
        input_name: str = "obs_dict",
        input_shape: list[object] | None = None,
        input_type: str = "tensor(float)",
        output_name: str = "encoded_tokens",
        output_shape: list[object] | None = None,
        output_type: str = "tensor(float)",
        result: np.ndarray | None = None,
    ) -> None:
        self.input_name = input_name
        self.input_shape = [1, 1247] if input_shape is None else input_shape
        self.input_type = input_type
        self.output_name = output_name
        self.output_shape = [1, 64] if output_shape is None else output_shape
        self.output_type = output_type
        self.result = np.zeros((1, 64), dtype=np.float32) if result is None else result
        self.run_calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=self.input_name, shape=self.input_shape, type=self.input_type)]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=self.output_name, shape=self.output_shape, type=self.output_type)]

    def run(self, outputs: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.run_calls.append((outputs, inputs))
        return [self.result]


@pytest.mark.parametrize(
    ("session", "message"),
    [
        (FakeSession(input_name="wrong"), "obs_dict"),
        (FakeSession(input_shape=[None, 1247]), r"\[1, 1247\]"),
        (FakeSession(input_type="tensor(double)"), "tensor.float"),
        (FakeSession(output_name="wrong"), "encoded_tokens"),
        (FakeSession(output_shape=[64]), r"\[1, 64\]"),
        (FakeSession(output_type="tensor(double)"), "tensor.float"),
    ],
)
def test_session_rejects_wrong_tensor_contract(session: FakeSession, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SonicEncoder.from_session(session)


def test_session_requires_exactly_one_input_and_output() -> None:
    session = FakeSession()
    session.get_inputs = lambda: []
    with pytest.raises(ValueError, match="exactly one input"):
        SonicEncoder.from_session(session)

    session = FakeSession()
    session.get_outputs = lambda: [*FakeSession().get_outputs(), *FakeSession().get_outputs()]
    with pytest.raises(ValueError, match="exactly one output"):
        SonicEncoder.from_session(session)


def test_encode_invokes_exact_names_and_preserves_rank_dtype_and_ownership() -> None:
    session = FakeSession(result=np.arange(64, dtype=np.float32).reshape(1, 64)[:, ::-1])
    encoder = SonicEncoder.from_session(session)
    tensor = np.zeros((1, 1247), dtype=np.float32)

    result = encoder.encode(tensor)

    assert session.run_calls == [(["encoded_tokens"], {"obs_dict": tensor})]
    assert result.shape == (1, 64)
    assert result.dtype == np.float32
    assert result.flags.owndata
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, session.result)


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        (np.zeros(1247, dtype=np.float32), r"\[1,1247\]"),
        (np.zeros((1, 1247), dtype=np.float64), "float32"),
        (np.zeros((1, 2494), dtype=np.float32)[:, ::2], "C-contiguous"),
        (np.zeros((2, 1247), dtype=np.float32)[0:1], "own its memory"),
    ],
)
def test_encode_rejects_noncontract_input(tensor: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SonicEncoder.from_session(FakeSession()).encode(tensor)


def test_encode_rejects_nonfinite_input_or_invalid_output() -> None:
    tensor = np.zeros((1, 1247), dtype=np.float32)
    tensor[0, 3] = np.nan
    with pytest.raises(ValueError, match="input.*finite"):
        SonicEncoder.from_session(FakeSession()).encode(tensor)

    for result, message in (
        (np.zeros((64,), dtype=np.float32), r"output.*\[1,64\]"),
        (np.zeros((1, 64), dtype=np.float64), "output.*float32"),
        (np.full((1, 64), np.inf, dtype=np.float32), "output.*finite"),
    ):
        with pytest.raises(ValueError, match=message):
            SonicEncoder.from_session(FakeSession(result=result)).encode(np.zeros((1, 1247), np.float32))


def _artifact(path: Path, *, repo_id: str = "nvidia/GEAR-SONIC") -> ArtifactSpec:
    payload = path.read_bytes()
    return ArtifactSpec(
        repo_id=repo_id,
        revision="9" * 40,
        filename=path.name,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_model: Path,
) -> tuple[ModuleType, list[tuple[str, list[str], object]]]:
    calls: list[tuple[str, list[str], object]] = []
    runtime = ModuleType("onnxruntime")
    runtime.__version__ = "fake-1.0"
    runtime.GraphOptimizationLevel = SimpleNamespace(ORT_DISABLE_ALL="disabled")

    class SessionOptions:
        graph_optimization_level: object | None = None

    def inference_session(
        model_path: str,
        *,
        providers: list[str],
        sess_options: object,
    ) -> FakeSession:
        assert Path(model_path) == expected_model.resolve()
        calls.append((model_path, providers, sess_options))
        return FakeSession()

    runtime.SessionOptions = SessionOptions
    runtime.InferenceSession = inference_session
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", runtime)
    return runtime, calls


def test_from_artifacts_verifies_config_materializes_with_cache_and_uses_cpu_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model_encoder.onnx"
    model_path.write_bytes(b"fake onnx")
    config_path = tmp_path / "observation_config.yaml"
    config_path.write_bytes(PINNED_CONFIG)
    lock = SimpleNamespace(encoder=_artifact(model_path), observation_config=_artifact(config_path))
    download_calls: list[dict[str, object]] = []
    hub = ModuleType("huggingface_hub")

    def download(**kwargs: object) -> str:
        download_calls.append(dict(kwargs))
        return str(model_path if kwargs["filename"] == model_path.name else config_path)

    hub.hf_hub_download = download
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", hub)
    runtime, session_calls = _install_fake_runtime(monkeypatch, expected_model=model_path)

    encoder = SonicEncoder.from_artifacts(lock, cache_dir=tmp_path / "hf-cache")

    assert [call["filename"] for call in download_calls] == [model_path.name, config_path.name]
    assert all(call["revision"] == "9" * 40 for call in download_calls)
    assert all(call["cache_dir"] == str(tmp_path / "hf-cache") for call in download_calls)
    assert len(session_calls) == 1
    assert session_calls[0][1] == ["CPUExecutionProvider"]
    assert session_calls[0][2].graph_optimization_level == runtime.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert encoder.encoder_path == model_path.resolve()
    assert encoder.observation_config_path == config_path.resolve()
    assert encoder.provider == "CPUExecutionProvider"
    assert encoder.runtime_version == "fake-1.0"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            PINNED_CONFIG.replace(b"mode_id: 0", b"mode_id: 3"),
            "G1 mode_id.*0",
        ),
        (
            PINNED_CONFIG.replace(
                b"        - motion_joint_positions_10frame_step1\n        - motion_joint_velocities",
                b"        - motion_joint_velocities_10frame_step1\n        - motion_joint_positions",
            ),
            "required_observations",
        ),
        (
            PINNED_CONFIG.replace(
                b"    - {name: encoder_mode_4, enabled: true}\n",
                b"    - {name: encoder_mode_4, enabled: true}\n    - {name: encoder_mode_4, enabled: true}\n",
            ),
            "encoder_observations",
        ),
        (
            PINNED_CONFIG.replace(b"  dimension: 64\n", b"  dimension: 64\n  dimension: 64\n"),
            "duplicate YAML mapping key",
        ),
    ],
)
def test_from_artifacts_rejects_semantically_ambiguous_config_before_runtime(
    config: bytes,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model_encoder.onnx"
    model_path.write_bytes(b"fake onnx")
    config_path = tmp_path / "observation_config.yaml"
    config_path.write_bytes(config)
    lock = SimpleNamespace(encoder=_artifact(model_path), observation_config=_artifact(config_path))
    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kwargs: str(
        model_path if kwargs["filename"] == model_path.name else config_path
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", hub)
    runtime = ModuleType("onnxruntime")
    runtime.InferenceSession = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("runtime constructed before config validation")
    )
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", runtime)

    with pytest.raises(ValueError, match=message):
        SonicEncoder.from_artifacts(lock, cache_dir=tmp_path / "cache")


def test_local_pinned_encoder_integration() -> None:
    model_path = Path("gear_sonic_deploy/policy/low_latency/model_encoder.onnx")
    config_path = Path("gear_sonic_deploy/policy/low_latency/observation_config.yaml")
    if not model_path.is_file() or not config_path.is_file():
        pytest.skip("pinned low-latency encoder is not materialized")
    ort = pytest.importorskip("onnxruntime")
    lock = load_source_lock("gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml")
    verified_model = verify_file(model_path, lock.encoder.size, lock.encoder.sha256)
    verify_file(config_path, lock.observation_config.size, lock.observation_config.sha256)
    session = ort.InferenceSession(str(verified_model), providers=["CPUExecutionProvider"])
    encoder = SonicEncoder.from_session(session)
    token = encoder.encode(np.zeros((1, 1247), dtype=np.float32))
    assert token.shape == (1, 64)
    assert np.isfinite(token).all()


@pytest.fixture(scope="module")
def pinned_encoder() -> SonicEncoder:
    override = os.environ.get("SONIC_PARITY_MODEL")
    model_path = (
        Path(override) if override is not None else Path("gear_sonic_deploy/policy/low_latency/model_encoder.onnx")
    )
    if not model_path.is_file():
        pytest.skip("pinned low-latency encoder is not materialized")
    ort = pytest.importorskip("onnxruntime")
    lock = load_source_lock("gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml")
    verified_model = verify_file(model_path, lock.encoder.size, lock.encoder.sha256)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(verified_model),
        providers=["CPUExecutionProvider"],
        sess_options=options,
    )
    return SonicEncoder.from_session(session)


def test_golden_encoder_fixtures_are_immutable_numeric_arrays() -> None:
    case_path = GOLDEN_DIR / "golden_encoder_case.npz"
    token_path = GOLDEN_DIR / "golden_encoder_token.npy"
    assert hashlib.sha256(case_path.read_bytes()).hexdigest() == GOLDEN_ENCODER_CASE_SHA256
    assert hashlib.sha256(token_path.read_bytes()).hexdigest() == GOLDEN_ENCODER_TOKEN_SHA256

    with np.load(case_path, allow_pickle=False) as case:
        assert set(case.files) == {
            "current_observed_root_wxyz",
            "encoder_input",
            "future_reference_root_wxyz",
            "initial_observed_root_wxyz",
            "initial_reference_root_wxyz",
            "orientations",
            "positions",
            "velocities",
        }
        assert all(case[name].dtype != np.dtype(object) for name in case.files)
    token = np.load(token_path, allow_pickle=False)
    assert token.shape == (1, 64)
    assert token.dtype == np.dtype(np.float32)


def test_golden_encoder_case_rebuilds_orientation_and_input_without_model() -> None:
    case = np.load(GOLDEN_DIR / "golden_encoder_case.npz")
    orientations = build_encoder_orientation_window(
        case["initial_observed_root_wxyz"],
        case["current_observed_root_wxyz"],
        case["initial_reference_root_wxyz"],
        case["future_reference_root_wxyz"],
    )
    np.testing.assert_allclose(orientations, case["orientations"], atol=1e-12, rtol=0)
    tensor = build_g1_encoder_input(case["positions"], case["velocities"], orientations)
    np.testing.assert_array_equal(tensor, case["encoder_input"])


def test_golden_encoder_case_matches_checked_token(pinned_encoder: SonicEncoder) -> None:
    case = np.load(GOLDEN_DIR / "golden_encoder_case.npz")
    token = pinned_encoder.encode(case["encoder_input"])
    np.testing.assert_allclose(
        token,
        np.load(GOLDEN_DIR / "golden_encoder_token.npy"),
        atol=1e-5,
        rtol=1e-5,
    )


def test_large_semantic_perturbation_changes_token(pinned_encoder: SonicEncoder) -> None:
    case = np.load(GOLDEN_DIR / "golden_encoder_case.npz")
    baseline = pinned_encoder.encode(case["encoder_input"])
    perturbed_positions = case["positions"].copy()
    perturbed_positions[:, 11] += np.float32(0.5)
    perturbed = build_g1_encoder_input(
        perturbed_positions,
        case["velocities"],
        case["orientations"],
    )
    assert np.count_nonzero(pinned_encoder.encode(perturbed) != baseline) >= 1
