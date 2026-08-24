from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils import g1_23dof_native124_21204_adapter as adapter
from gear_sonic.utils.g1_23dof_native124_policy import (
    PUBLIC_POLICY_SHA256,
    SELECTED_GANTRY_POLICY_SHA256,
    Native124Policy,
    native_to_hardware_compact,
)
from gear_sonic.utils.g1_23dof_xr24_soma_stream import (
    ROLLING_SOMA_KIND,
    CausalHistorySemanticProducer,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def binding() -> adapter.Native124Checkpoint21204Binding:
    return adapter.load_checkpoint21204_binding(REPOSITORY_ROOT)


@pytest.fixture(scope="module")
def policy(
    binding: adapter.Native124Checkpoint21204Binding,
) -> adapter.Native124Checkpoint21204Policy:
    return adapter.Native124Checkpoint21204Policy(binding)


@pytest.fixture(scope="module")
def tracker() -> adapter.Native124CausalTracker21204:
    return adapter.Native124CausalTracker21204(REPOSITORY_ROOT)


def _rolling_sample(index: int) -> dict[str, object]:
    reference_ns = 1_000_000_000 + index * 20_000_000
    return {
        "kind": ROLLING_SOMA_KIND,
        "promotion_eligible": False,
        "source_frame_index": index,
        "reference_monotonic_ns": reference_ns,
        "capture_monotonic_ns": reference_ns + 1_000_000,
        "compute_finished_monotonic_ns": reference_ns + 2_000_000,
        "producer_sha256": "d" * 64,
        "joint_pos_il29": [0.01 * joint + index * (joint + 1) * 1.0e-4 for joint in range(29)],
        "body_term": {
            "source_frame_index": index,
            "reference_monotonic_ns": reference_ns,
            "capture_monotonic_ns": reference_ns + 1_000_000,
            "vr_3point_local_target": [0.1] * 9,
            "vr_3point_local_orn_target": [0.0, 0.0, 0.0, 1.0] * 3,
            "reference_anchor_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }


def _causal_packet() -> dict[str, object]:
    producer = CausalHistorySemanticProducer(source_session_id="adapter-test")
    packet = None
    for index in range(11):
        packet = producer.push(_rolling_sample(index))
    assert packet is not None
    return packet


def _robot_state(packet: dict[str, object]) -> adapter.CausalNative124RobotState:
    return adapter.CausalNative124RobotState(
        robot_torso_quaternion_wxyz_q9=(1.0, 0.0, 0.0, 0.0),
        robot_torso_monotonic_ns_q9=int(packet["anchor_reference_monotonic_ns"]),
        control_source_frame_index_q10=int(packet["proof_source_frame_index"]),
        robot_state_monotonic_ns_q10=int(packet["proof_reference_monotonic_ns"]),
        q_measured_hardware_q10=adapter.HOME_Q_HARDWARE + np.arange(23, dtype=np.float32) * 0.01,
        qd_measured_hardware_q10=np.arange(23, dtype=np.float32) * -0.02,
        base_angular_velocity_q10=(0.25, -0.5, 0.75),
        previous_applied_target_hardware=(
            adapter.HOME_Q_HARDWARE + adapter.ACTION_SCALE_HARDWARE * (np.arange(23, dtype=np.float32) * 0.03)
        ),
    )


def test_selected_manifest_and_all_bound_artifacts_are_hash_locked(
    binding: adapter.Native124Checkpoint21204Binding,
) -> None:
    assert binding.iteration == 21204
    assert binding.manifest_sha256 == adapter.MANIFEST_SHA256
    assert binding.selection_sha256 == adapter.SELECTION_SHA256
    assert binding.export_report_sha256 == adapter.EXPORT_REPORT_SHA256
    assert binding.checkpoint_sha256 == adapter.CHECKPOINT_SHA256
    assert binding.actor_state_sha256 == adapter.ACTOR_STATE_SHA256
    assert binding.onnx_sha256 == adapter.ONNX_SHA256
    assert adapter.sha256_file(binding.manifest_path) == adapter.MANIFEST_SHA256
    assert adapter.sha256_file(binding.selection_path) == adapter.SELECTION_SHA256
    assert adapter.sha256_file(binding.export_report_path) == adapter.EXPORT_REPORT_SHA256
    assert adapter.sha256_file(binding.checkpoint_path) == adapter.CHECKPOINT_SHA256
    assert adapter.sha256_file(binding.onnx_path) == adapter.ONNX_SHA256
    assert adapter.sha256_file(binding.resolved_env_evidence_path) == adapter.RESOLVED_ENV_EVIDENCE_SHA256


def test_tampered_manifest_rejected_before_artifact_loading(tmp_path: Path) -> None:
    path = tmp_path / adapter.MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracker manifest SHA256 mismatch"):
        adapter.load_checkpoint21204_binding(tmp_path)


def test_selected_policy_is_exact_single_input_cpu_and_deterministic(
    policy: adapter.Native124Checkpoint21204Policy,
) -> None:
    assert policy._session.get_providers() == ["CPUExecutionProvider"]  # noqa: SLF001
    assert [
        (item.name, item.shape, item.type)
        for item in policy._session.get_inputs()  # noqa: SLF001
    ] == [("obs", [1, 124], "tensor(float)")]
    assert [
        (item.name, item.shape, item.type)
        for item in policy._session.get_outputs()  # noqa: SLF001
    ] == [("actions", [1, 23], "tensor(float)")]
    probes = (
        np.zeros((1, 124), dtype=np.float32),
        np.linspace(-1.0, 1.0, 124, dtype=np.float32)[None, :],
        np.eye(1, 124, 63, dtype=np.float32),
    )
    for observation in probes:
        first = policy.run(observation)
        second = policy.run(observation.copy())
        assert first.shape == (23,) and first.dtype == np.float32
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "observation",
    (
        [0.0] * 124,
        np.zeros(124, dtype=np.float32),
        np.zeros((1, 124), dtype=np.float64),
        np.zeros((2, 124), dtype=np.float32),
        np.full((1, 124), np.nan, dtype=np.float32),
        np.full((1, 124), np.inf, dtype=np.float32),
    ),
)
def test_selected_policy_rejects_bad_observation(
    policy: adapter.Native124Checkpoint21204Policy,
    observation: object,
) -> None:
    with pytest.raises(ValueError, match="finite float32"):
        policy.run(observation)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("inputs", "outputs", "providers", "match"),
    (
        (
            [("obs", [1, 124], "tensor(float)"), ("time_step", [1, 1], "tensor(float)")],
            [("actions", [1, 23], "tensor(float)")],
            ["CPUExecutionProvider"],
            "input contract",
        ),
        (
            [("input", [1, 124], "tensor(float)")],
            [("actions", [1, 23], "tensor(float)")],
            ["CPUExecutionProvider"],
            "input contract",
        ),
        (
            [("obs", [None, 124], "tensor(float)")],
            [("actions", [1, 23], "tensor(float)")],
            ["CPUExecutionProvider"],
            "input contract",
        ),
        (
            [("obs", [1, 124], "tensor(float)")],
            [("action", [1, 23], "tensor(float)")],
            ["CPUExecutionProvider"],
            "output contract",
        ),
        (
            [("obs", [1, 124], "tensor(float)")],
            [("actions", [1, 23], "tensor(float)")],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "provider contract",
        ),
    ),
)
def test_selected_policy_rejects_abi_or_provider_drift(
    monkeypatch: pytest.MonkeyPatch,
    binding: adapter.Native124Checkpoint21204Binding,
    inputs: list[tuple[str, list[object], str]],
    outputs: list[tuple[str, list[object], str]],
    providers: list[str],
    match: str,
) -> None:
    class FakeSession:
        def get_providers(self) -> list[str]:
            return providers

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name=name, shape=shape, type=dtype) for name, shape, dtype in inputs]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name=name, shape=shape, type=dtype) for name, shape, dtype in outputs]

    monkeypatch.setattr(adapter, "_create_cpu_session", lambda _path: FakeSession())
    with pytest.raises(ValueError, match=match):
        adapter.Native124Checkpoint21204Policy(binding)


@pytest.mark.parametrize(
    "bad_output",
    (
        np.zeros((23,), dtype=np.float32),
        np.zeros((1, 23), dtype=np.float64),
        np.full((1, 23), np.nan, dtype=np.float32),
        np.full((1, 23), np.inf, dtype=np.float32),
    ),
)
def test_selected_policy_rejects_bad_onnx_output(
    monkeypatch: pytest.MonkeyPatch,
    binding: adapter.Native124Checkpoint21204Binding,
    bad_output: np.ndarray,
) -> None:
    class FakeSession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="obs", shape=[1, 124], type="tensor(float)")]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="actions", shape=[1, 23], type="tensor(float)")]

        def run(self, _outputs: object, _inputs: object) -> list[np.ndarray]:
            return [bad_output]

    monkeypatch.setattr(adapter, "_create_cpu_session", lambda _path: FakeSession())
    with pytest.raises(RuntimeError, match="finite float32"):
        adapter.Native124Checkpoint21204Policy(binding)


def test_candidate_home_scale_and_legacy_contract_remain_distinct(
    binding: adapter.Native124Checkpoint21204Binding,
) -> None:
    observation = adapter.build_checkpoint21204_observation(
        q_ref_hardware=np.zeros(23),
        qd_ref_hardware=np.zeros(23),
        torso_motion_anchor_ori_b=(1, 0, 0, 1, 0, 0),
        base_angular_velocity=np.zeros(3),
        q_measured_hardware=adapter.HOME_Q_HARDWARE,
        qd_measured_hardware=np.zeros(23),
        previous_applied_raw_action_hardware=np.zeros(23),
    )
    np.testing.assert_array_equal(observation[0, 55:78], np.zeros(23, dtype=np.float32))
    np.testing.assert_array_equal(
        adapter.checkpoint21204_raw_action_to_hardware_targets(np.zeros(23)),
        adapter.HOME_Q_HARDWARE,
    )
    np.testing.assert_allclose(
        adapter.checkpoint21204_raw_action_to_hardware_targets(np.ones(23)),
        adapter.HOME_Q_HARDWARE + adapter.ACTION_SCALE_HARDWARE,
        rtol=0.0,
        atol=0.0,
    )
    raw = np.linspace(-2.0, 2.0, 23, dtype=np.float32)
    np.testing.assert_allclose(
        adapter.hardware_targets_to_checkpoint21204_raw_action(
            adapter.checkpoint21204_raw_action_to_hardware_targets(raw)
        ),
        raw,
        rtol=0.0,
        atol=2.0e-7,
    )
    assert PUBLIC_POLICY_SHA256 == "632c62569348a5cfd0769ea21a71bef2b66078c1c1543f06f91efb3b1163033e"
    assert SELECTED_GANTRY_POLICY_SHA256 == "cc644839807b6ef522e47b3bcb69845843aa345b4fb895847c76642830b5d2b9"
    with pytest.raises(ValueError, match="input contract mismatch"):
        Native124Policy(binding.onnx_path, expected_sha256=adapter.ONNX_SHA256)


def test_causal_tracker_joins_full_q9_q10_layout_without_mutating_inputs(
    tracker: adapter.Native124CausalTracker21204,
) -> None:
    packet = _causal_packet()
    original_packet = copy.deepcopy(packet)
    state = _robot_state(packet)
    result = tracker.evaluate(semantic_packet=packet, robot_state=state)
    second = tracker.evaluate(semantic_packet=packet, robot_state=state)
    assert packet == original_packet
    observation = np.asarray(result["observation_124"], dtype=np.float32)
    expected_q_ref = native_to_hardware_compact(packet["q_ref23_native"])
    expected_qd_ref = native_to_hardware_compact(packet["qd_ref23_native"])
    expected_previous = adapter.hardware_targets_to_checkpoint21204_raw_action(
        state.previous_applied_target_hardware
    )
    np.testing.assert_array_equal(observation[0:23], expected_q_ref)
    np.testing.assert_array_equal(observation[23:46], expected_qd_ref)
    np.testing.assert_array_equal(observation[46:52], np.asarray((1, 0, 0, 1, 0, 0), dtype=np.float32))
    np.testing.assert_array_equal(
        observation[52:55], np.asarray(state.base_angular_velocity_q10, dtype=np.float32)
    )
    np.testing.assert_allclose(
        observation[55:78],
        np.arange(23, dtype=np.float32) * 0.01,
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(
        observation[78:101], np.asarray(state.qd_measured_hardware_q10, dtype=np.float32)
    )
    np.testing.assert_array_equal(observation[101:124], expected_previous)
    assert len(result["encoder_input_267"]) == 267
    assert len(result["raw_tracker_action_hardware"]) == 23
    assert len(result["candidate_target_hardware"]) == 23
    assert result["raw_tracker_action_hardware"] == second["raw_tracker_action_hardware"]
    assert result["observation_sha256"] == second["observation_sha256"]
    assert result["diagnostic_only"] is True
    assert result["robot_commands_performed"] is False
    assert result["actuation_permitted"] is False
    assert result["teacher_label_admitted"] is False
    assert result["promotion_eligible"] is False
    assert result["causal_observation_parity_claimed"] is False
    assert result["causal_distribution_shift_requires_support_gate"] is True


@pytest.mark.parametrize(
    ("field", "delta", "match"),
    (
        ("robot_torso_monotonic_ns_q9", 1, "q9 torso time"),
        ("robot_state_monotonic_ns_q10", 1, "q10 state time"),
        ("control_source_frame_index_q10", 1, "q10 control frame"),
    ),
)
def test_causal_tracker_rejects_misaligned_join(
    tracker: adapter.Native124CausalTracker21204,
    field: str,
    delta: int,
    match: str,
) -> None:
    packet = _causal_packet()
    state = _robot_state(packet)
    bad = replace(state, **{field: getattr(state, field) + delta})
    with pytest.raises(ValueError, match=match):
        tracker.evaluate(semantic_packet=packet, robot_state=bad)


def test_adapter_has_no_transport_or_command_import_surface() -> None:
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    forbidden = ("unitree", "dds", "zmq", "socket", "subprocess")
    assert not any(token in name.lower() for name in imports for token in forbidden)
