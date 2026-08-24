"""Native 23-DoF whole-body reference-policy contract.

This module is deliberately independent of DDS and Unitree command APIs.  It
assembles the exact 124-value observation used by public G1 23-DoF
BeyondMimic/UniLab actors, validates the pinned ONNX, and converts its native
23-value normalized action into compact hardware targets.  Calling it can
never actuate a robot.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES,
    HARDWARE_JOINT_IDS,
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_JOINT_NAMES,
)

OBSERVATION_DIM = 124
ACTION_DIM = 23
OBSERVATION_LAYOUT = (
    ("q_ref_native", 23),
    ("qd_ref_native", 23),
    ("motion_anchor_ori_b", 6),
    ("base_ang_vel", 3),
    ("joint_pos_rel_native", 23),
    ("joint_vel_native", 23),
    ("previous_raw_action_native", 23),
)

PUBLIC_POLICY_REPO = "Kennyp-Chen/robojudo_sar-assets"
PUBLIC_POLICY_REVISION = "80f33ae12a5ce0862e1a3d6d36990e43f94154a0"
PUBLIC_POLICY_PATH = (
    "assets/models/g1/beyondmimic/23dof_50fps/OldTownRoad_v1.onnx"
)
PUBLIC_POLICY_SHA256 = (
    "632c62569348a5cfd0769ea21a71bef2b66078c1c1543f06f91efb3b1163033e"
)

# Selected from the same pinned native-23 release after neutral and external
# full-body reference MuJoCo qualification.  This is the gantry candidate;
# the OldTownRoad constants above remain the reproducible baseline fixture.
SELECTED_GANTRY_POLICY_PATH = (
    "assets/models/g1/beyondmimic/23dof_50fps/"
    "fightAndSports1_subject1.onnx"
)
SELECTED_GANTRY_POLICY_SHA256 = (
    "cc644839807b6ef522e47b3bcb69845843aa345b4fb895847c76642830b5d2b9"
)

# Values embedded in pinned model metadata, native IsaacLab breadth-first order.
DEFAULT_Q_NATIVE = np.asarray(
    [
        -0.312, -0.312, 0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0,
        0.2, -0.2, 0.669, 0.669, 0.0, 0.0, -0.363, -0.363,
        0.6, 0.6, 0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)
ACTION_SCALE_NATIVE = np.asarray(
    [
        0.548, 0.548, 0.439, 0.351, 0.351, 0.439, 0.439, 0.548,
        0.548, 0.439, 0.439, 0.351, 0.351, 0.439, 0.439, 0.439,
        0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439,
    ],
    dtype=np.float32,
)


def _vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def hardware_compact_to_native(values: Sequence[float]) -> np.ndarray:
    """Convert compact Unitree/MuJoCo-23 order to policy-native order."""

    hardware = _vector(values, ACTION_DIM, "hardware_compact")
    return hardware[np.asarray(MUJOCO_TO_ISAACLAB_DOF, dtype=np.int64)]


def native_to_hardware_compact(values: Sequence[float]) -> np.ndarray:
    """Convert policy-native order to compact Unitree/MuJoCo-23 order."""

    native = _vector(values, ACTION_DIM, "native")
    return native[np.asarray(ISAACLAB_TO_MUJOCO_DOF, dtype=np.int64)]


def sdk_slots_to_native(values: Sequence[float]) -> np.ndarray:
    """Select the 23 real joints from a 29-slot Unitree SDK array."""

    slots = _vector(values, 29, "sdk_slots")
    compact = slots[np.asarray(HARDWARE_JOINT_IDS, dtype=np.int64)]
    return hardware_compact_to_native(compact)


def build_observation(
    *,
    q_ref_native: Sequence[float],
    qd_ref_native: Sequence[float],
    motion_anchor_ori_b: Sequence[float],
    base_ang_vel: Sequence[float],
    q_measured_native: Sequence[float],
    qd_measured_native: Sequence[float],
    previous_raw_action_native: Sequence[float],
) -> np.ndarray:
    """Build exact float32 ``[1,124]`` actor input."""

    result = np.concatenate(
        (
            _vector(q_ref_native, 23, "q_ref_native"),
            _vector(qd_ref_native, 23, "qd_ref_native"),
            _vector(motion_anchor_ori_b, 6, "motion_anchor_ori_b"),
            _vector(base_ang_vel, 3, "base_ang_vel"),
            _vector(q_measured_native, 23, "q_measured_native")
            - DEFAULT_Q_NATIVE,
            _vector(qd_measured_native, 23, "qd_measured_native"),
            _vector(
                previous_raw_action_native,
                23,
                "previous_raw_action_native",
            ),
        )
    ).astype(np.float32, copy=False)
    if result.shape != (OBSERVATION_DIM,) or not np.isfinite(result).all():
        raise RuntimeError("native124 observation layout drift")
    return result[None, :]


def raw_action_to_hardware_targets(raw_action_native: Sequence[float]) -> np.ndarray:
    """Apply model metadata scale/default and return compact hardware targets."""

    raw = _vector(raw_action_native, ACTION_DIM, "raw_action_native")
    target_native = DEFAULT_Q_NATIVE + ACTION_SCALE_NATIVE * raw
    return native_to_hardware_compact(target_native)


def hardware_targets_to_raw_action(hardware_targets: Sequence[float]) -> np.ndarray:
    """Encode the target actually applied as the actor's last-action input."""

    target_native = hardware_compact_to_native(hardware_targets)
    return ((target_native - DEFAULT_Q_NATIVE) / ACTION_SCALE_NATIVE).astype(
        np.float32,
        copy=False,
    )


def scatter_hardware_targets(
    compact_targets: Sequence[float],
    *,
    untouched_slots: Sequence[float],
) -> np.ndarray:
    """Scatter into only 23 enabled SDK slots; preserve six absent slots."""

    compact = _vector(compact_targets, ACTION_DIM, "compact_targets")
    result = _vector(untouched_slots, 29, "untouched_slots").copy()
    result[np.asarray(HARDWARE_JOINT_IDS, dtype=np.int64)] = compact
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Native124Policy:
    """Pinned ONNX inference wrapper; has no robot/network mutation surface."""

    path: Path
    expected_sha256: str = PUBLIC_POLICY_SHA256

    def __post_init__(self) -> None:
        self.path = Path(self.path).resolve()
        if (
            len(self.expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.expected_sha256)
        ):
            raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")
        actual_hash = sha256_file(self.path)
        if actual_hash != self.expected_sha256:
            raise ValueError(
                "native23 ONNX SHA256 mismatch: "
                f"expected {self.expected_sha256}, got {actual_hash}"
            )
        self.sha256 = actual_hash
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - environment-specific
            raise RuntimeError("onnxruntime is required for native23 inference") from error
        self._session = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )
        inputs = [(item.name, item.shape, item.type) for item in self._session.get_inputs()]
        outputs = [(item.name, item.shape, item.type) for item in self._session.get_outputs()]
        if inputs != [
            ("obs", [1, OBSERVATION_DIM], "tensor(float)"),
            ("time_step", [1, 1], "tensor(float)"),
        ]:
            raise ValueError(f"native23 ONNX input contract mismatch: {inputs}")
        action_outputs = [item for item in outputs if item[0] == "actions"]
        if action_outputs != [("actions", [1, ACTION_DIM], "tensor(float)")]:
            raise ValueError("native23 ONNX actions output contract mismatch")

    def run(self, observation: Sequence[float] | np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape == (OBSERVATION_DIM,):
            obs = obs[None, :]
        if obs.shape != (1, OBSERVATION_DIM) or not np.isfinite(obs).all():
            raise ValueError("observation must be finite float32 [1,124]")
        action = self._session.run(
            ["actions"],
            {"obs": obs, "time_step": np.zeros((1, 1), dtype=np.float32)},
        )[0]
        if action.shape != (1, ACTION_DIM) or not np.isfinite(action).all():
            raise RuntimeError("native23 ONNX returned invalid action")
        return action[0].astype(np.float32, copy=False)

    def embedded_reference(self, time_step: int) -> dict[str, np.ndarray]:
        """Read clip reference outputs for reproducible MuJoCo validation only."""

        if isinstance(time_step, bool) or not isinstance(time_step, int) or time_step < 0:
            raise ValueError("time_step must be a non-negative integer")
        names = [
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
        ]
        values = self._session.run(
            names,
            {
                "obs": np.zeros((1, OBSERVATION_DIM), dtype=np.float32),
                "time_step": np.asarray([[time_step]], dtype=np.float32),
            },
        )
        result = {
            name: np.asarray(value[0], dtype=np.float32)
            for name, value in zip(names, values, strict=True)
        }
        expected = {
            "joint_pos": (23,),
            "joint_vel": (23,),
            "body_pos_w": (14, 3),
            "body_quat_w": (14, 4),
        }
        if any(
            result[name].shape != shape or not np.isfinite(result[name]).all()
            for name, shape in expected.items()
        ):
            raise RuntimeError("native23 embedded reference output contract drift")
        return result


assert len(NATIVE_IL23_JOINT_NAMES) == len(HARDWARE_23_JOINT_NAMES) == 23
assert sum(size for _, size in OBSERVATION_LAYOUT) == OBSERVATION_DIM
assert math.isfinite(float(ACTION_SCALE_NATIVE.max()))
