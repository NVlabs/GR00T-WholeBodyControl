"""Hash-locked offline adapter for the selected iteration-21204 native124 actor.

This module is intentionally separate from :mod:`g1_23dof_native124_policy`.
That legacy public-policy wrapper has a different two-input ABI, joint ordering,
default pose, and action scale.  Nothing here opens a robot, DDS, ZMQ, or command
surface.  Returned actions are unadmitted simulator/offline tracker candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gear_sonic.utils.g1_23dof_native124_policy import (
    native_to_hardware_compact,
)
from gear_sonic.utils.g1_23dof_xr24_soma_stream import (
    CAUSAL_HISTORY_ROBOT_ANCHOR_CONTRACT,
    complete_causal_history_encoder_packet,
    validate_causal_history_packet,
)

OBSERVATION_DIM = 124
ACTION_DIM = 23
CHECKPOINT_ITERATION = 21204

MANIFEST_RELATIVE_PATH = Path("gear_sonic/config/sim_validation/g1_true23_native124_21204_tracker_v1.json")
MANIFEST_SHA256 = "03435a9c86aff7f53ca12082dbfcf44e8579a9f16cbb7e44ddde21bf8f189826"
SELECTION_SHA256 = "8a0c2658ad0b366c210189ebf87c7c9487c3531ea36b12654ca4d4dc8ffa04cb"
EXPORT_REPORT_SHA256 = "21faa01eacd13b66f7ccc4c7a40c67f6a12777d0b4d6e94390cc50807d54c66b"
CHECKPOINT_SHA256 = "9cb0a06db441b8ceb51404b45ba25a81bd4120114aa6b97d6f660cac3f742f81"
ACTOR_STATE_SHA256 = "17302f7076cb480fe4ffc253e7b8228fcbaa033ccb3bf7aac1ed34940b8648ec"
ONNX_SHA256 = "321504108e677fb4b70d1398ff9a20e168def2231eb574e6d8fc1f39385d7b9b"
RESOLVED_ENV_EVIDENCE_SHA256 = "40fdcadb3096842414d6e5307d50ef69bec42f696d0866a40cc487f1ae7103b8"

# Exact stock MJLab G1-23DoF resolved environment values used by this lineage,
# in Unitree/MJLab compact hardware order.  These are deliberately not the
# knees-bent public-policy constants in g1_23dof_native124_policy.py.
HOME_Q_HARDWARE = np.asarray(
    (
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        0.0,
        0.35,
        0.18,
        0.0,
        0.87,
        0.0,
        0.35,
        -0.18,
        0.0,
        0.87,
        0.0,
    ),
    dtype=np.float32,
)

_HIP_YAW_PITCH_SCALE = 0.5475464629911068
_HIP_ROLL_KNEE_SCALE = 0.35066146637882434
_ANKLE_ARM_SCALE = 0.43857731392336724
ACTION_SCALE_HARDWARE = np.asarray(
    (
        _HIP_YAW_PITCH_SCALE,
        _HIP_ROLL_KNEE_SCALE,
        _HIP_YAW_PITCH_SCALE,
        _HIP_ROLL_KNEE_SCALE,
        _ANKLE_ARM_SCALE,
        _ANKLE_ARM_SCALE,
        _HIP_YAW_PITCH_SCALE,
        _HIP_ROLL_KNEE_SCALE,
        _HIP_YAW_PITCH_SCALE,
        _HIP_ROLL_KNEE_SCALE,
        _ANKLE_ARM_SCALE,
        _ANKLE_ARM_SCALE,
        _HIP_YAW_PITCH_SCALE,
        _ANKLE_ARM_SCALE,
        _ANKLE_ARM_SCALE,
        _ANKLE_ARM_SCALE,
        _ANKLE_ARM_SCALE,
        _ANKLE_ARM_SCALE,
        _ANKLE_ARM_SCALE,
        _ANKLE_ARM_SCALE,
        _ANKLE_ARM_SCALE,
        _ANKLE_ARM_SCALE,
        _ANKLE_ARM_SCALE,
    ),
    dtype=np.float32,
)

OBSERVATION_LAYOUT = (
    ("q_ref_hardware_q9", 23),
    ("qd_ref_hardware_q9_to_q10", 23),
    ("torso_motion_anchor_ori_b_q9", 6),
    ("base_angular_velocity_q10", 3),
    ("joint_position_hardware_q10_minus_home", 23),
    ("joint_velocity_hardware_q10", 23),
    ("previous_applied_raw_action_hardware", 23),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be lowercase SHA256")
    return value


def _require_regular_file(path: Path, expected_sha256: str, context: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{context} SHA256 mismatch: expected {expected_sha256}, got {actual}")


def _repo_file(repository_root: Path, relative: object, context: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{context} must be a non-empty repository-relative path")
    root = repository_root.resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"{context} escapes repository root")
    return result


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _load_json(path: Path, expected_sha256: str, context: str) -> Mapping[str, Any]:
    _require_regular_file(path, expected_sha256, context)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error
    return _mapping(value, context)


@dataclass(frozen=True)
class Native124Checkpoint21204Binding:
    repository_root: Path
    manifest_path: Path
    selection_path: Path
    export_report_path: Path
    checkpoint_path: Path
    onnx_path: Path
    resolved_env_evidence_path: Path
    manifest_sha256: str = MANIFEST_SHA256
    selection_sha256: str = SELECTION_SHA256
    export_report_sha256: str = EXPORT_REPORT_SHA256
    checkpoint_sha256: str = CHECKPOINT_SHA256
    actor_state_sha256: str = ACTOR_STATE_SHA256
    onnx_sha256: str = ONNX_SHA256
    resolved_env_evidence_sha256: str = RESOLVED_ENV_EVIDENCE_SHA256
    iteration: int = CHECKPOINT_ITERATION


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "g1_true23_native124_21204_tracker_manifest_v1"
        or manifest.get("role") != "offline_simulator_causal_tracker_teacher_candidate_only"
    ):
        raise ValueError("iteration-21204 tracker manifest identity mismatch")
    actor = _mapping(manifest.get("actor_contract"), "manifest.actor_contract")
    if actor != {
        "input": {"name": "obs", "dtype": "float32", "shape": [1, 124]},
        "output": {
            "name": "actions",
            "dtype": "float32",
            "shape": [1, 23],
            "joint_order": "unitree_mjlab_hardware_compact_23",
        },
        "activation": "ELU",
        "hidden_dims": [512, 256, 128],
        "normalization_epsilon": 0.01,
    }:
        raise ValueError("iteration-21204 tracker actor contract mismatch")
    observation = _mapping(manifest.get("observation_contract"), "manifest.observation_contract")
    if (
        observation.get("control_hz") != 50
        or observation.get("layout") != [list(item) for item in OBSERVATION_LAYOUT]
        or not np.array_equal(
            np.asarray(observation.get("home_q_hardware"), dtype=np.float32),
            HOME_Q_HARDWARE,
        )
        or not np.array_equal(
            np.asarray(observation.get("action_scale_hardware"), dtype=np.float32),
            ACTION_SCALE_HARDWARE,
        )
        or observation.get("external_reference_joint_order") != "native_il23"
        or observation.get("robot_anchor") != "torso_quaternion_at_q9"
        or observation.get("simulator_robot_anchor_source")
        != "robot.data.body_link_quat_w[torso_link]_snapshotted_before_q9_to_q10_transition"
        or observation.get("live_robot_anchor_source") != "torso_imu_quaternion_interpolated_at_pico_q9"
        or observation.get("robot_state") != "q10_current"
        or observation.get("q_measured_source") != "robot.data.joint_pos_biased_q10"
        or observation.get("joint_position_encoder_bias_semantics")
        != "match_stock_mjlab_actor_joint_pos_rel_biased_true"
        or observation.get("previous_action_source")
        != "physical_hardware_target_at_transition_q9_to_q10_inverted_through_checkpoint21204_home_and_scale"
    ):
        raise ValueError("iteration-21204 tracker observation contract mismatch")
    boundaries = _mapping(manifest.get("boundaries"), "manifest.boundaries")
    required_true = {
        "cpu_only",
        "diagnostic_only",
        "causal_distribution_shift_requires_support_gate",
    }
    required_false = {
        "actuation_permitted",
        "deployment_ready",
        "teacher_label_admitted",
        "promotion_eligible",
        "legacy_two_input_abi_may_change",
        "causal_observation_parity_claimed",
    }
    if any(boundaries.get(name) is not True for name in required_true) or any(
        boundaries.get(name) is not False for name in required_false
    ):
        raise ValueError("iteration-21204 tracker safety boundary mismatch")


def load_checkpoint21204_binding(
    repository_root: str | Path | None = None,
) -> Native124Checkpoint21204Binding:
    """Validate frozen manifest plus every bound artifact before inference."""

    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[2]
    manifest_path = _repo_file(root, str(MANIFEST_RELATIVE_PATH), "manifest path")
    manifest = _load_json(manifest_path, MANIFEST_SHA256, "tracker manifest")
    _validate_manifest_contract(manifest)
    identity = _mapping(manifest.get("identity"), "manifest.identity")
    if identity.get("iteration") != CHECKPOINT_ITERATION:
        raise ValueError("tracker manifest checkpoint iteration mismatch")

    expected = {
        "selection": SELECTION_SHA256,
        "export_report": EXPORT_REPORT_SHA256,
        "checkpoint": CHECKPOINT_SHA256,
        "onnx": ONNX_SHA256,
        "resolved_env_evidence": RESOLVED_ENV_EVIDENCE_SHA256,
    }
    paths: dict[str, Path] = {}
    for name, expected_sha in expected.items():
        reference = _mapping(identity.get(name), f"manifest.identity.{name}")
        if _require_sha256(reference.get("sha256"), f"manifest.identity.{name}.sha256") != expected_sha:
            raise ValueError(f"tracker manifest {name} SHA256 binding mismatch")
        path = _repo_file(root, reference.get("path"), f"manifest.identity.{name}.path")
        _require_regular_file(path, expected_sha, f"tracker {name}")
        paths[name] = path
    checkpoint_ref = _mapping(identity.get("checkpoint"), "manifest.identity.checkpoint")
    if checkpoint_ref.get("actor_state_sha256") != ACTOR_STATE_SHA256:
        raise ValueError("tracker manifest actor-state binding mismatch")

    selection = _load_json(paths["selection"], SELECTION_SHA256, "selected policy selection")
    policy_contract = _mapping(selection.get("policy_contract"), "selection.policy_contract")
    checkpoint = _mapping(selection.get("checkpoint"), "selection.checkpoint")
    onnx = _mapping(selection.get("onnx"), "selection.onnx")
    all61 = _mapping(selection.get("all61_gate"), "selection.all61_gate")
    if (
        selection.get("schema") != "g1_true23_native124_all61_selection_v1"
        or selection.get("selected") is not True
        or policy_contract
        != {
            "observation_dim": 124,
            "action_dim": 23,
            "native_true23": True,
            "uses_29dof_action_mask": False,
        }
        or checkpoint.get("path") != "../interpolated/model21204_alpha25.pt"
        or checkpoint.get("sha256") != CHECKPOINT_SHA256
        or onnx.get("path") != "policy.onnx"
        or onnx.get("sha256") != ONNX_SHA256
        or onnx.get("parity_report") != "export_report.json"
        or onnx.get("parity_cases") != 82
        or onnx.get("passed") is not True
        or float(onnx.get("max_absolute_error", math.inf)) > 1.0e-5
        or all61.get("clips_retained") != 61
        or all61.get("clips_rejected") != 0
        or all61.get("rollouts_per_clip") != 10
        or all61.get("steps_per_rollout") != 500
        or selection.get("qualification_scope")
        != "MuJoCo/MJLab clip tracking only; live teleoperation is not yet qualified"
    ):
        raise ValueError("selected policy selection contract mismatch")
    if (paths["selection"].parent / str(checkpoint["path"])).resolve() != paths["checkpoint"] or (
        paths["selection"].parent / str(onnx["path"])
    ).resolve() != paths["onnx"]:
        raise ValueError("selected policy relative path binding mismatch")

    report = _load_json(paths["export_report"], EXPORT_REPORT_SHA256, "selected export report")
    report_contract = _mapping(report.get("actor_contract"), "export.actor_contract")
    lineage = _mapping(report.get("checkpoint_lineage"), "export.checkpoint_lineage")
    actor_state = _mapping(lineage.get("actor_state"), "export.actor_state")
    report_checkpoint = _mapping(lineage.get("checkpoint"), "export.checkpoint")
    export = _mapping(report.get("export"), "export.export")
    parity = _mapping(report.get("parity"), "export.parity")
    onnx_parity = _mapping(
        parity.get("exported_onnx_vs_checkpoint"),
        "export.parity.exported_onnx_vs_checkpoint",
    )
    comparison = _mapping(onnx_parity.get("comparison"), "export.parity.comparison")
    probe = _mapping(parity.get("probe_suite"), "export.parity.probe_suite")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "unitree_g1_23dof_native124_actor_export"
        or report_contract.get("observation_dim") != 124
        or report_contract.get("action_dim") != 23
        or report_contract.get("activation") != "ELU"
        or report_contract.get("hidden_dims") != [512, 256, 128]
        or report_contract.get("normalization_epsilon") != 0.01
        or report_contract.get("input") != {"dtype": "float32", "name": "obs", "shape": [1, 124]}
        or report_contract.get("output") != {"dtype": "float32", "name": "actions", "shape": [1, 23]}
        or actor_state.get("sha256") != ACTOR_STATE_SHA256
        or report_checkpoint.get("iteration") != CHECKPOINT_ITERATION
        or report_checkpoint.get("sha256") != CHECKPOINT_SHA256
        or export.get("output_sha256") != ONNX_SHA256
        or probe.get("case_count") != 82
        or comparison.get("case_count") != 82
        or comparison.get("passed") is not True
        or float(comparison.get("max_absolute_error", math.inf)) > 1.0e-5
    ):
        raise ValueError("selected policy export report contract mismatch")

    return Native124Checkpoint21204Binding(
        repository_root=root,
        manifest_path=manifest_path,
        selection_path=paths["selection"],
        export_report_path=paths["export_report"],
        checkpoint_path=paths["checkpoint"],
        onnx_path=paths["onnx"],
        resolved_env_evidence_path=paths["resolved_env_evidence"],
    )


def _create_cpu_session(path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("onnxruntime is required for selected native124 inference") from error
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class Native124Checkpoint21204Policy:
    """Strict one-input CPU ONNX runtime for exactly the selected artifact."""

    def __init__(self, binding: Native124Checkpoint21204Binding) -> None:
        self.binding = binding
        _require_regular_file(binding.onnx_path, ONNX_SHA256, "selected ONNX before load")
        self._session = _create_cpu_session(binding.onnx_path)
        providers = self._session.get_providers()
        inputs = [(item.name, item.shape, item.type) for item in self._session.get_inputs()]
        outputs = [(item.name, item.shape, item.type) for item in self._session.get_outputs()]
        if providers != ["CPUExecutionProvider"]:
            raise ValueError(f"selected ONNX provider contract mismatch: {providers}")
        if inputs != [("obs", [1, OBSERVATION_DIM], "tensor(float)")]:
            raise ValueError(f"selected ONNX input contract mismatch: {inputs}")
        if outputs != [("actions", [1, ACTION_DIM], "tensor(float)")]:
            raise ValueError(f"selected ONNX output contract mismatch: {outputs}")
        self.run(np.zeros((1, OBSERVATION_DIM), dtype=np.float32))
        _require_regular_file(binding.onnx_path, ONNX_SHA256, "selected ONNX after load")

    def run(self, observation: np.ndarray) -> np.ndarray:
        if (
            not isinstance(observation, np.ndarray)
            or observation.dtype != np.float32
            or observation.shape != (1, OBSERVATION_DIM)
            or not np.isfinite(observation).all()
        ):
            raise ValueError("selected ONNX observation must be finite float32 [1,124]")
        values = self._session.run(["actions"], {"obs": observation})
        if len(values) != 1:
            raise RuntimeError("selected ONNX returned an unexpected output count")
        action = values[0]
        if (
            not isinstance(action, np.ndarray)
            or action.dtype != np.float32
            or action.shape != (1, ACTION_DIM)
            or not np.isfinite(action).all()
        ):
            raise RuntimeError("selected ONNX action must be finite float32 [1,23]")
        return action[0].copy()


def _vector(value: object, size: int, context: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{context} must be finite shape ({size},)")
    return result


def build_checkpoint21204_observation(
    *,
    q_ref_hardware: Sequence[float],
    qd_ref_hardware: Sequence[float],
    torso_motion_anchor_ori_b: Sequence[float],
    base_angular_velocity: Sequence[float],
    q_measured_hardware: Sequence[float],
    qd_measured_hardware: Sequence[float],
    previous_applied_raw_action_hardware: Sequence[float],
) -> np.ndarray:
    """Build selected actor input in its exact stock-MJLab hardware order."""

    observation = np.concatenate(
        (
            _vector(q_ref_hardware, 23, "q_ref_hardware"),
            _vector(qd_ref_hardware, 23, "qd_ref_hardware"),
            _vector(torso_motion_anchor_ori_b, 6, "torso_motion_anchor_ori_b"),
            _vector(base_angular_velocity, 3, "base_angular_velocity"),
            _vector(q_measured_hardware, 23, "q_measured_hardware") - HOME_Q_HARDWARE,
            _vector(qd_measured_hardware, 23, "qd_measured_hardware"),
            _vector(
                previous_applied_raw_action_hardware,
                23,
                "previous_applied_raw_action_hardware",
            ),
        )
    ).astype(np.float32, copy=False)
    if observation.shape != (OBSERVATION_DIM,) or not np.isfinite(observation).all():
        raise RuntimeError("iteration-21204 observation layout drift")
    return observation[None, :]


def checkpoint21204_raw_action_to_hardware_targets(
    raw_action_hardware: Sequence[float],
) -> np.ndarray:
    """Pure diagnostic conversion using candidate-specific HOME and scale."""

    raw = _vector(raw_action_hardware, ACTION_DIM, "raw_action_hardware")
    return (HOME_Q_HARDWARE + ACTION_SCALE_HARDWARE * raw).astype(np.float32, copy=False)


def hardware_targets_to_checkpoint21204_raw_action(
    target_hardware: Sequence[float],
) -> np.ndarray:
    """Express an applied target in the selected actor's raw coordinates.

    A SONIC raw action cannot merely be reordered into this field: SONIC and
    the selected native124 actor use different default poses and scales.  The
    physical joint target is the semantics-preserving boundary.
    """

    target = _vector(target_hardware, ACTION_DIM, "target_hardware")
    return ((target - HOME_Q_HARDWARE) / ACTION_SCALE_HARDWARE).astype(
        np.float32,
        copy=False,
    )


@dataclass(frozen=True)
class CausalNative124RobotState:
    """Robot-side half of one exact q9-reference/q10-control offline join."""

    robot_torso_quaternion_wxyz_q9: Sequence[float]
    robot_torso_monotonic_ns_q9: int
    control_source_frame_index_q10: int
    robot_state_monotonic_ns_q10: int
    q_measured_hardware_q10: Sequence[float]
    qd_measured_hardware_q10: Sequence[float]
    base_angular_velocity_q10: Sequence[float]
    previous_applied_target_hardware: Sequence[float]


class Native124CausalTracker21204:
    """Stateless, non-actuating selected-policy evaluator for causal packets."""

    def __init__(self, repository_root: str | Path | None = None) -> None:
        self.binding = load_checkpoint21204_binding(repository_root)
        self.policy = Native124Checkpoint21204Policy(self.binding)

    def evaluate(
        self,
        *,
        semantic_packet: Mapping[str, Any],
        robot_state: CausalNative124RobotState,
    ) -> dict[str, Any]:
        """Return one unadmitted tracker record; never mutate caller or simulator."""

        validation = validate_causal_history_packet(semantic_packet)
        anchor_ns = int(semantic_packet["anchor_reference_monotonic_ns"])
        control_ns = int(semantic_packet["proof_reference_monotonic_ns"])
        control_index = int(semantic_packet["proof_source_frame_index"])
        for value, expected, context in (
            (robot_state.robot_torso_monotonic_ns_q9, anchor_ns, "robot q9 torso time"),
            (robot_state.robot_state_monotonic_ns_q10, control_ns, "robot q10 state time"),
            (
                robot_state.control_source_frame_index_q10,
                control_index,
                "robot q10 control frame",
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                raise ValueError(f"{context} must exactly match causal packet")
        completed = complete_causal_history_encoder_packet(
            semantic_packet=semantic_packet,
            robot_anchor_quaternion_wxyz=robot_state.robot_torso_quaternion_wxyz_q9,
            robot_anchor_monotonic_ns=robot_state.robot_torso_monotonic_ns_q9,
            robot_anchor_source_contract=CAUSAL_HISTORY_ROBOT_ANCHOR_CONTRACT,
        )
        q_ref_hardware = native_to_hardware_compact(semantic_packet["q_ref23_native"])
        qd_ref_hardware = native_to_hardware_compact(semantic_packet["qd_ref23_native"])
        previous_hardware = hardware_targets_to_checkpoint21204_raw_action(
            robot_state.previous_applied_target_hardware
        )
        motion_anchor = completed["encoder_terms"]["motion_anchor_ori_b"]
        observation = build_checkpoint21204_observation(
            q_ref_hardware=q_ref_hardware,
            qd_ref_hardware=qd_ref_hardware,
            torso_motion_anchor_ori_b=motion_anchor,
            base_angular_velocity=robot_state.base_angular_velocity_q10,
            q_measured_hardware=robot_state.q_measured_hardware_q10,
            qd_measured_hardware=robot_state.qd_measured_hardware_q10,
            previous_applied_raw_action_hardware=previous_hardware,
        )
        raw_action = self.policy.run(observation)
        target = checkpoint21204_raw_action_to_hardware_targets(raw_action)
        observation_sha256 = hashlib.sha256(observation.tobytes(order="C")).hexdigest()
        return {
            "schema_version": 1,
            "kind": "g1_true23_native124_21204_causal_tracker_record_v1",
            "diagnostic_only": True,
            "cpu_only": True,
            "robot_commands_performed": False,
            "actuation_permitted": False,
            "deployment_ready": False,
            "teacher_label_admitted": False,
            "promotion_eligible": False,
            "causal_observation_parity_claimed": False,
            "causal_distribution_shift_requires_support_gate": True,
            "artifact": {
                "iteration": self.binding.iteration,
                "manifest_sha256": self.binding.manifest_sha256,
                "selection_sha256": self.binding.selection_sha256,
                "export_report_sha256": self.binding.export_report_sha256,
                "checkpoint_sha256": self.binding.checkpoint_sha256,
                "actor_state_sha256": self.binding.actor_state_sha256,
                "onnx_sha256": self.binding.onnx_sha256,
            },
            "source_session_id": str(semantic_packet["source_session_id"]),
            "pico_anchor_source_frame_index": int(semantic_packet["anchor_source_frame_index"]),
            "pico_anchor_monotonic_ns": anchor_ns,
            "control_source_frame_index": control_index,
            "control_monotonic_ns": control_ns,
            "reference_packet_sha256": completed["semantic_packet_sha256"],
            "reference_validation_sha256": validation["sha256"],
            "encoder_input_267": list(completed["encoder_input"]),
            "encoder_input_sha256": completed["encoder_input_sha256"],
            "observation_124": observation[0].tolist(),
            "observation_sha256": observation_sha256,
            "raw_tracker_action_hardware": raw_action.tolist(),
            "candidate_target_hardware": target.tolist(),
            "previous_applied_action_source": (
                "physical_hardware_target_q9_to_q10_inverted_through_checkpoint21204_home_and_scale"
            ),
        }


assert sum(size for _, size in OBSERVATION_LAYOUT) == OBSERVATION_DIM
assert HOME_Q_HARDWARE.shape == ACTION_SCALE_HARDWARE.shape == (ACTION_DIM,)
