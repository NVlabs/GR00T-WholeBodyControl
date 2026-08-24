from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_contract import (
    EXCLUDED_HARDWARE_JOINT_IDS,
    HARDWARE_JOINT_IDS,
)
from gear_sonic.utils.g1_23dof_native124_policy import (
    ACTION_DIM,
    DEFAULT_Q_NATIVE,
    PUBLIC_POLICY_SHA256,
    Native124Policy,
    build_observation,
    hardware_compact_to_native,
    hardware_targets_to_raw_action,
    native_to_hardware_compact,
    raw_action_to_hardware_targets,
    scatter_hardware_targets,
    sha256_file,
)

MODEL = (
    Path(__file__).parents[2]
    / "artifacts"
    / "unitree23_public"
    / "OldTownRoad_v1.onnx"
)


def test_native_hardware_permutations_are_exact_inverse() -> None:
    values = np.arange(ACTION_DIM, dtype=np.float32)
    assert np.array_equal(
        native_to_hardware_compact(hardware_compact_to_native(values)),
        values,
    )


def test_observation_layout_is_exact_124() -> None:
    observation = build_observation(
        q_ref_native=np.arange(23),
        qd_ref_native=np.arange(23) + 100,
        motion_anchor_ori_b=[1, 0, 0, 1, 0, 0],
        base_ang_vel=[7, 8, 9],
        q_measured_native=DEFAULT_Q_NATIVE + 0.25,
        qd_measured_native=np.arange(23) + 200,
        previous_raw_action_native=np.arange(23) + 300,
    )
    assert observation.shape == (1, 124)
    assert observation.dtype == np.float32
    assert observation[0, :23].tolist() == pytest.approx(range(23))
    assert observation[0, 23:46].tolist() == pytest.approx(
        np.arange(23) + 100
    )
    assert observation[0, 52:55].tolist() == pytest.approx([7, 8, 9])
    assert observation[0, 55:78].tolist() == pytest.approx([0.25] * 23)
    assert observation[0, 101:].tolist() == pytest.approx(
        np.arange(23) + 300
    )


def test_scatter_never_changes_six_excluded_slots() -> None:
    untouched = np.arange(29, dtype=np.float32) + 1000
    result = scatter_hardware_targets(
        np.arange(23, dtype=np.float32), untouched_slots=untouched
    )
    assert result[list(HARDWARE_JOINT_IDS)].tolist() == pytest.approx(range(23))
    assert np.array_equal(
        result[list(EXCLUDED_HARDWARE_JOINT_IDS)],
        untouched[list(EXCLUDED_HARDWARE_JOINT_IDS)],
    )


def test_applied_target_action_round_trip() -> None:
    raw = np.linspace(-0.75, 0.75, 23, dtype=np.float32)
    targets = raw_action_to_hardware_targets(raw)
    recovered = hardware_targets_to_raw_action(targets)
    assert np.allclose(recovered, raw, rtol=0.0, atol=2.0e-7)


@pytest.mark.skipif(not MODEL.is_file(), reason="pinned public ONNX not downloaded")
def test_pinned_policy_hash_inference_and_time_independence() -> None:
    assert sha256_file(MODEL) == PUBLIC_POLICY_SHA256
    policy = Native124Policy(MODEL)
    observation = build_observation(
        q_ref_native=DEFAULT_Q_NATIVE,
        qd_ref_native=np.zeros(23),
        motion_anchor_ori_b=[1, 0, 0, 1, 0, 0],
        base_ang_vel=np.zeros(3),
        q_measured_native=DEFAULT_Q_NATIVE,
        qd_measured_native=np.zeros(23),
        previous_raw_action_native=np.zeros(23),
    )
    action = policy.run(observation)
    assert action.shape == (23,)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action)) < 1.0
    targets = raw_action_to_hardware_targets(action)
    assert targets.shape == (23,)
    assert np.isfinite(targets).all()
