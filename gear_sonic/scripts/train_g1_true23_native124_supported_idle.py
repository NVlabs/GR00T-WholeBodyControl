"""Authorization-gated native124 supported-idle fine-tuning.

No arguments prints a deterministic static A/B plan. Planning is read-only,
never imports MJLab, and never touches a GPU/run directory. This file stops at
strict CPU preflight and has no run subcommand. Self-attested diagnostics never
unlock later phases; verified replay/raw-trace validation remains required.

Corpus ``diagnostic_only=true`` and ``training_authorized=false`` flags are
immutable evidence.  Explicit authorization lives in a separate file and does
not rewrite or reinterpret those fields.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

PLAN_KIND = "g1_true23_native124_supported_idle_finetune_plan_v1"
AUTHORIZATION_KIND = "g1_true23_native124_supported_idle_training_authorization_v1"
GATE_KIND = "g1_true23_native124_supported_idle_self_attested_curriculum_diagnostic_gate_v1"
CHECKPOINT_INFO_KIND = "g1_true23_native124_supported_idle_checkpoint_v1"
DIAGNOSTIC_EVIDENCE_CLASS = "self_attested_curriculum_diagnostic_only"
CONTACT_EVIDENCE_SCOPE = "aggregate_curriculum_only_not_final_step1b_stance_qualification"
FUTURE_EXECUTION_BLOCKER = "verified diagnostic replay/raw-trace validator not implemented"
SCHEMA_VERSION = 1
CONTROL_HZ = 50
EPISODE_STEPS = 500
NUM_STEPS_PER_ENV = 24
DEFAULT_NUM_ENVS = 512
SESSION_UPDATES = 250
TRAINING_SEED = 20260806
EXPECTED_CORPUS_SHA256 = "97cd8d8acf06a396ba041a7c5742a3eadc442e4235f0932f4e0b75a2cef85687"
EXPECTED_SIDECAR_SHA256 = "acca78155ea075c65f6af0a66369dafc2a2dca88a9982ba8b26f20a206ceca06"

CHANGE_IDLE = "idle__220713__change_idle_left_a021"
HANDS_ON_BACK = "idle__220721__hands_on_back_loop_a036m"
QUALIFICATION_CLIPS = (CHANGE_IDLE, HANDS_ON_BACK)

DEFAULT_SIDECAR = (
    REPO_ROOT
    / "artifacts"
    / "g1_true23_step1c_supported_idle_native_corpus_v1"
    / "supported_idle_native_corpus_v1.spans.json"
)
DEFAULT_SOURCE_RUN = (
    REPO_ROOT
    / "external_dependencies"
    / "unitree_rl_mjlab"
    / "logs"
    / "rsl_rl"
    / "g1_23dof_tracking"
    / "2026-08-04_21-55-39_dad_dance_full_resume_2500_to30000_512"
)
SEED_CHECKPOINTS: Mapping[str, tuple[Path, str, int]] = {
    "model3500": (
        DEFAULT_SOURCE_RUN / "model_3500.pt",
        "41488ab06ee876fb6ddf77ac18d22231e67f06f9c09c45531b21ed2abd709596",
        3500,
    ),
    "model11500": (
        DEFAULT_SOURCE_RUN / "model_11500.pt",
        "1302ed2d7128c5f129611c29a34181d1ac7e27d2c15f551e49453e41ee81ec4a",
        11500,
    ),
}
PHASES: Mapping[str, Mapping[str, Any]] = {
    "static": {
        "phase": "static",
        "start_mode": "start_only",
        "planned_updates": 2000,
        "required_gate_phase": None,
        "qualification_mode": "static_frame0_zero_velocity",
    },
    "trajectory_start": {
        "phase": "trajectory",
        "start_mode": "start_only",
        "planned_updates": 5000,
        "required_gate_phase": "static",
        "qualification_mode": "full_terminal_hold_zero_velocity",
    },
    "trajectory_uniform": {
        "phase": "trajectory",
        "start_mode": "uniform_window",
        "planned_updates": 3000,
        "required_gate_phase": "trajectory_start",
        "qualification_mode": "full_terminal_hold_zero_velocity",
    },
}
RUNTIME_FILES = (
    Path(__file__).resolve(),
    REPO_ROOT / "gear_sonic/envs/mjlab/native124_supported_idle.py",
    REPO_ROOT / "gear_sonic/trl/mjlab/supported_idle_checkpoint.py",
    REPO_ROOT / "gear_sonic/trl/mjlab/supported_idle_runner.py",
    REPO_ROOT / "gear_sonic/utils/g1_23dof_native124_actor_export.py",
    REPO_ROOT / "gear_sonic/utils/g1_true23_step1c_onnx_diagnostic.py",
    REPO_ROOT / "gear_sonic/utils/g1_true23_step1b_mujoco.py",
    REPO_ROOT / "gear_sonic/utils/g1_23dof_contract.py",
    REPO_ROOT / "gear_sonic/utils/g1_23dof_native124_policy.py",
    REPO_ROOT / "gear_sonic/scripts/run_g1_true23_step1c_onnx_diagnostic.py",
    REPO_ROOT / "gear_sonic/data/robots/g1/g1_29dof.xml",
    REPO_ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
    REPO_ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
    REPO_ROOT / "gear_sonic/config/sim_validation/g1_true23_idle_step1b_qualification_v1.json",
    REPO_ROOT / "external_dependencies/mjlab/src/mjlab/rl/runner.py",
    REPO_ROOT / "external_dependencies/mjlab/src/mjlab/tasks/tracking/mdp/commands.py",
    REPO_ROOT / "external_dependencies/mjlab/src/mjlab/managers/command_manager.py",
    REPO_ROOT / "external_dependencies/mjlab/src/mjlab/envs/manager_based_rl_env.py",
    REPO_ROOT / "external_dependencies/unitree_rl_mjlab/src/tasks/tracking/rl/runner.py",
    REPO_ROOT / "external_dependencies/unitree_rl_mjlab/src/tasks/tracking/config/g1_23dof/rl_cfg.py",
    REPO_ROOT / "external_dependencies/unitree_rl_mjlab/src/tasks/tracking/config/g1_23dof/env_cfgs.py",
    REPO_ROOT / "external_dependencies/unitree_rl_mjlab/src/tasks/tracking/tracking_env_cfg.py",
    REPO_ROOT / "external_dependencies/unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py",
    REPO_ROOT / "external_dependencies/unitree_rl_mjlab/src/assets/robots/unitree_g1/g1_23dof_constants.py",
    REPO_ROOT / "external_dependencies/unitree_rl_mjlab/scripts/train.py",
    REPO_ROOT
    / "external_dependencies/unitree_rl_mjlab/deploy/robots/g1_23dof/config/policy/mimic"
    / "B_DadDance/params/deploy.yaml",
)

DEPLOY_PARAMS_PATH = (
    REPO_ROOT
    / "external_dependencies/unitree_rl_mjlab/deploy/robots/g1_23dof/config/policy/mimic"
    / "B_DadDance/params/deploy.yaml"
)
DEPLOY_PARAMS_SHA256 = "40fdcadb3096842414d6e5307d50ef69bec42f696d0866a40cc487f1ae7103b8"
HARDWARE_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)
OBSERVATION_LAYOUT = (
    ("qref_unitree_mjlab_hardware_order", 23),
    ("qdref_unitree_mjlab_hardware_order", 23),
    ("torso_relative_rotation_6d", 6),
    ("base_angular_velocity", 3),
    ("joint_position_minus_home_unitree_mjlab_hardware_order", 23),
    ("joint_velocity_unitree_mjlab_hardware_order", 23),
    ("previous_raw_action_unitree_mjlab_hardware_order", 23),
)

PINNED_PACKAGE_VERSIONS = {
    "mjlab": "1.2.0",
    "mujoco": "3.5.0",
    "numpy": "2.3.4",
    "onnx": "1.22.0",
    "onnxruntime": "1.23.2",
    "rsl-rl-lib": "5.0.1",
    "torch": "2.9.0+cu128",
}
ACTOR_TENSOR_SCHEMA = {
    "obs_normalizer._mean": ((1, 124), "torch.float32"),
    "obs_normalizer._var": ((1, 124), "torch.float32"),
    "obs_normalizer._std": ((1, 124), "torch.float32"),
    "obs_normalizer.count": ((), "torch.int64"),
    "distribution.std_param": ((23,), "torch.float32"),
    "mlp.0.weight": ((512, 124), "torch.float32"),
    "mlp.0.bias": ((512,), "torch.float32"),
    "mlp.2.weight": ((256, 512), "torch.float32"),
    "mlp.2.bias": ((256,), "torch.float32"),
    "mlp.4.weight": ((128, 256), "torch.float32"),
    "mlp.4.bias": ((128,), "torch.float32"),
    "mlp.6.weight": ((23, 128), "torch.float32"),
    "mlp.6.bias": ((23,), "torch.float32"),
}
CRITIC_TENSOR_SCHEMA = {
    "obs_normalizer._mean": ((1, 256), "torch.float32"),
    "obs_normalizer._var": ((1, 256), "torch.float32"),
    "obs_normalizer._std": ((1, 256), "torch.float32"),
    "obs_normalizer.count": ((), "torch.int64"),
    "mlp.0.weight": ((512, 256), "torch.float32"),
    "mlp.0.bias": ((512,), "torch.float32"),
    "mlp.2.weight": ((256, 512), "torch.float32"),
    "mlp.2.bias": ((256,), "torch.float32"),
    "mlp.4.weight": ((128, 256), "torch.float32"),
    "mlp.4.bias": ((128,), "torch.float32"),
    "mlp.6.weight": ((1, 128), "torch.float32"),
    "mlp.6.bias": ((1,), "torch.float32"),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def observed_package_versions() -> dict[str, str | None]:
    """Read installed distribution metadata without importing runtime packages."""

    result: dict[str, str | None] = {}
    for distribution in PINNED_PACKAGE_VERSIONS:
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def read_json_snapshot(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str, int, Path]:
    """Parse and hash one immutable byte snapshot, closing parse/hash TOCTOU."""

    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"JSON input must not be symlink: {unresolved}")
    candidate = unresolved.resolve(strict=True)
    if not candidate.is_file():
        raise ValueError(f"JSON input must be a regular file: {candidate}")
    raw = candidate.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != _require_sha256(expected_sha256, "JSON SHA256"):
        raise ValueError("JSON file SHA256 mismatch")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_json_no_duplicates,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {candidate}")
    return value, digest, len(raw), candidate


def read_json(path: Path) -> dict[str, Any]:
    return read_json_snapshot(path)[0]


def read_bound_json(value: Any, name: str) -> tuple[dict[str, Any], Path]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{name} file binding keys mismatch")
    path = _resolve_repo_relative(value["path"], f"{name}.path")
    parsed, digest, size, resolved = read_json_snapshot(
        path,
        expected_sha256=_require_sha256(value["sha256"], f"{name}.sha256"),
    )
    if type(value["size_bytes"]) is not int or value["size_bytes"] != size:
        raise ValueError(f"{name} size binding changed")
    if digest != value["sha256"]:
        raise ValueError(f"{name} SHA256 binding changed")
    return parsed, resolved


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA256")
    return value


def _require_exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be integer >= {minimum}, not bool/coerced numeric")
    return value


def _require_mapping_keys(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} keys mismatch")
    return value


def _repo_relative(path: Path) -> str:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"file must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"file must remain under repository root: {resolved}") from error
    return PurePosixPath(relative).as_posix()


def _resolve_repo_relative(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} escapes repository root")
    unresolved = REPO_ROOT / Path(*relative.parts)
    if unresolved.is_symlink():
        raise ValueError(f"{name} must not be symlink")
    try:
        candidate = unresolved.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} bound file is missing") from error
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"{name} escapes repository root") from error
    if not candidate.is_file():
        raise ValueError(f"{name} must bind a regular file")
    return candidate


def file_binding(path: Path) -> dict[str, Any]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"bound input must not be symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"bound input must be regular non-symlink file: {resolved}")
    return {
        "path": _repo_relative(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def verify_file_binding(value: Any, name: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{name} file binding keys mismatch")
    path = _resolve_repo_relative(value["path"], f"{name}.path")
    expected_hash = _require_sha256(value["sha256"], f"{name}.sha256")
    if type(value["size_bytes"]) is not int or value["size_bytes"] < 1:
        raise ValueError(f"{name}.size_bytes must be positive integer")
    if path.stat().st_size != value["size_bytes"] or sha256_file(path) != expected_hash:
        raise ValueError(f"{name} file binding changed")
    return path


def read_bound_bytes(value: Any, name: str) -> tuple[bytes, Path]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{name} file binding keys mismatch")
    path = _resolve_repo_relative(value["path"], f"{name}.path")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _require_sha256(value["sha256"], f"{name}.sha256"):
        raise ValueError(f"{name} SHA256 binding changed")
    if type(value["size_bytes"]) is not int or value["size_bytes"] != len(raw):
        raise ValueError(f"{name} size binding changed")
    return raw, path


def validate_actor_onnx(value: Any) -> str:
    """Independently validate actual actor bytes; exporter JSON is not trusted."""

    raw, _ = read_bound_bytes(value, "gate.actor_onnx")
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper

    try:
        model = onnx.load_model_from_string(raw)
        onnx.checker.check_model(model, full_check=True)
    except Exception as error:
        raise ValueError("gate actor ONNX failed full checker") from error
    if [(entry.domain, entry.version) for entry in model.opset_import] != [("", 18)]:
        raise ValueError("gate actor ONNX must use only default opset 18")
    if model.graph.sparse_initializer:
        raise ValueError("gate actor ONNX sparse initializers are forbidden")

    def _tensor_contract(value_info: Any, name: str, shape: tuple[int, int]) -> None:
        tensor_type = value_info.type.tensor_type
        dimensions = tensor_type.shape.dim
        if value_info.name != name or tensor_type.elem_type != TensorProto.FLOAT:
            raise ValueError("gate actor ONNX input/output identity mismatch")
        if len(dimensions) != len(shape):
            raise ValueError("gate actor ONNX input/output rank mismatch")
        for dimension, expected in zip(dimensions, shape, strict=True):
            if dimension.dim_param or dimension.dim_value != expected:
                raise ValueError("gate actor ONNX input/output must have static exact shape")

    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("gate actor ONNX must have one input and one output")
    _tensor_contract(model.graph.input[0], "obs", (1, 124))
    _tensor_contract(model.graph.output[0], "actions", (1, 23))
    expected_initializers = {
        "obs_normalizer._mean": (1, 124),
        "mlp.0.weight": (512, 124),
        "mlp.0.bias": (512,),
        "mlp.2.weight": (256, 512),
        "mlp.2.bias": (256,),
        "mlp.4.weight": (128, 256),
        "mlp.4.bias": (128,),
        "mlp.6.weight": (23, 128),
        "mlp.6.bias": (23,),
    }
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    if len(initializers) != 10 or not set(expected_initializers) < set(initializers):
        raise ValueError("gate actor ONNX initializer names/count mismatch")
    divisor_names = set(initializers) - set(expected_initializers)
    if len(divisor_names) != 1:
        raise ValueError("gate actor ONNX normalization divisor initializer mismatch")
    expected_initializers[next(iter(divisor_names))] = (1, 124)
    for name, shape in expected_initializers.items():
        initializer = initializers[name]
        if initializer.data_type != TensorProto.FLOAT or tuple(initializer.dims) != shape:
            raise ValueError("gate actor ONNX initializer shape/dtype mismatch")
        if initializer.data_location != TensorProto.DEFAULT or initializer.external_data:
            raise ValueError("gate actor ONNX external initializers are forbidden")
        array = numpy_helper.to_array(initializer)
        if array.dtype != np.float32 or not bool(np.isfinite(array).all()):
            raise ValueError("gate actor ONNX initializer contains non-finite data")
    expected_ops = ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]
    if [node.op_type for node in model.graph.node] != expected_ops:
        raise ValueError("gate actor ONNX operation topology mismatch")
    return hashlib.sha256(raw).hexdigest()


def _validate_sidecar(path: Path) -> tuple[dict[str, Any], Path]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ValueError("supported-idle sidecar must not be symlink")
    resolved = unresolved.resolve(strict=True)
    if sha256_file(resolved) != EXPECTED_SIDECAR_SHA256:
        raise ValueError("supported-idle sidecar differs from pinned immutable artifact")
    sidecar = read_json(resolved)
    required = {
        "schema_version": SCHEMA_VERSION,
        "kind": "g1_true23_supported_idle_native_corpus_v1",
        "episode_frames": EPISODE_STEPS,
        "fps": float(CONTROL_HZ),
        "diagnostic_only": True,
        "training_authorized": False,
    }
    for key, expected in required.items():
        if sidecar.get(key) != expected or type(sidecar.get(key)) is not type(expected):
            raise ValueError(f"supported-idle sidecar {key!r} drifted")
    corpus = sidecar.get("corpus")
    if not isinstance(corpus, Mapping) or corpus.get("sha256") != EXPECTED_CORPUS_SHA256:
        raise ValueError("supported-idle corpus SHA256 drifted")
    name = corpus.get("path")
    if not isinstance(name, str) or PurePosixPath(name).name != name:
        raise ValueError("supported-idle corpus path must be one basename")
    unresolved_corpus = resolved.parent / name
    if unresolved_corpus.is_symlink():
        raise ValueError("supported-idle corpus must not be symlink")
    corpus_path = unresolved_corpus.resolve(strict=True)
    if not corpus_path.is_file():
        raise ValueError("supported-idle corpus must be regular non-symlink file")
    if sha256_file(corpus_path) != EXPECTED_CORPUS_SHA256:
        raise ValueError("supported-idle corpus bytes differ from pinned artifact")
    return sidecar, corpus_path


def _validate_export_report(
    binding: Any,
    *,
    checkpoint_sha256: str,
    actor_onnx_sha256: str,
) -> None:
    report, _ = read_bound_json(binding, "gate.export_report")
    _require_mapping_keys(
        report,
        {"actor_contract", "checkpoint_lineage", "export", "kind", "parity", "runtime", "schema_version"},
        "qualification export report",
    )
    if _require_exact_int(report["schema_version"], "export.schema_version") != SCHEMA_VERSION:
        raise ValueError("qualification export report schema version mismatch")
    if report["kind"] != "unitree_g1_23dof_native124_actor_export":
        raise ValueError("qualification export report kind mismatch")

    contract = _require_mapping_keys(
        report["actor_contract"],
        {"action_dim", "activation", "hidden_dims", "input", "normalization_epsilon", "observation_dim", "output"},
        "export.actor_contract",
    )
    expected_contract = {
        "action_dim": 23,
        "activation": "ELU",
        "hidden_dims": [512, 256, 128],
        "input": {"dtype": "float32", "name": "obs", "shape": [1, 124]},
        "normalization_epsilon": 0.01,
        "observation_dim": 124,
        "output": {"dtype": "float32", "name": "actions", "shape": [1, 23]},
    }
    if dict(contract) != expected_contract:
        raise ValueError("qualification export actor contract mismatch")

    lineage = _require_mapping_keys(
        report["checkpoint_lineage"],
        {"actor_state", "checkpoint", "safe_load"},
        "export.checkpoint_lineage",
    )
    checkpoint = _require_mapping_keys(
        lineage["checkpoint"],
        {"iteration", "path", "root_keys", "sha256"},
        "export.checkpoint_lineage.checkpoint",
    )
    _require_exact_int(checkpoint["iteration"], "export.checkpoint.iteration")
    if type(checkpoint["path"]) is not str or not checkpoint["path"]:
        raise ValueError("qualification export checkpoint path missing")
    if checkpoint["root_keys"] != [
        "actor_state_dict",
        "critic_state_dict",
        "infos",
        "iter",
        "optimizer_state_dict",
    ]:
        raise ValueError("qualification export checkpoint root keys mismatch")
    if checkpoint["sha256"] != checkpoint_sha256:
        raise ValueError("qualification export report checkpoint SHA256 mismatch")
    if lineage["safe_load"] != {"map_location": "cpu", "torch_weights_only": True}:
        raise ValueError("qualification export safe-load proof mismatch")
    actor_state = _require_mapping_keys(
        lineage["actor_state"],
        {"parameter_count", "sha256", "tensor_count", "tensors"},
        "export.checkpoint_lineage.actor_state",
    )
    if actor_state["parameter_count"] != 231587 or actor_state["tensor_count"] != 13:
        raise ValueError("qualification export actor-state count mismatch")
    _require_sha256(actor_state["sha256"], "export.actor_state.sha256")
    tensor_rows = actor_state["tensors"]
    if not isinstance(tensor_rows, list) or len(tensor_rows) != len(ACTOR_TENSOR_SCHEMA):
        raise ValueError("qualification export actor-state tensor rows mismatch")
    rows_by_key: dict[str, Mapping[str, Any]] = {}
    for row in tensor_rows:
        row = _require_mapping_keys(
            row,
            {"dtype", "key", "parameter_count", "shape"},
            "export.actor_state.tensor",
        )
        key = row["key"]
        if type(key) is not str or key in rows_by_key:
            raise ValueError("qualification export actor-state tensor key invalid")
        rows_by_key[key] = row
    if set(rows_by_key) != set(ACTOR_TENSOR_SCHEMA):
        raise ValueError("qualification export actor-state tensor key set mismatch")
    for key, (shape, dtype) in ACTOR_TENSOR_SCHEMA.items():
        row = rows_by_key[key]
        parameter_count = 1
        for dimension in shape:
            parameter_count *= dimension
        if row["shape"] != list(shape) or row["dtype"] != dtype or row["parameter_count"] != parameter_count:
            raise ValueError("qualification export actor-state tensor schema mismatch")

    export = _require_mapping_keys(
        report["export"],
        {"no_overwrite", "onnx_opset", "output_path", "output_sha256"},
        "export.export",
    )
    if export["no_overwrite"] is not True or export["onnx_opset"] != 18:
        raise ValueError("qualification export publication contract mismatch")
    if type(export["output_path"]) is not str or not export["output_path"]:
        raise ValueError("qualification export output path missing")
    if export["output_sha256"] != actor_onnx_sha256:
        raise ValueError("qualification export actual ONNX SHA256 mismatch")

    parity_root = _require_mapping_keys(
        report["parity"],
        {
            "exported_onnx_vs_checkpoint",
            "exported_onnx_vs_reference",
            "probe_suite",
            "reference_onnx_vs_checkpoint",
        },
        "export.parity",
    )
    if (
        parity_root["exported_onnx_vs_reference"] is not None
        or parity_root["reference_onnx_vs_checkpoint"] is not None
    ):
        raise ValueError("qualification export unexpected reference parity")
    proof = _require_mapping_keys(
        parity_root["exported_onnx_vs_checkpoint"],
        {"comparison", "onnx"},
        "export.parity.exported_onnx_vs_checkpoint",
    )
    comparison = _require_mapping_keys(
        proof["comparison"],
        {
            "actual",
            "atol",
            "case_count",
            "expected",
            "max_absolute_error",
            "max_relative_error",
            "passed",
            "rtol",
            "worst_absolute_case",
        },
        "export.parity.comparison",
    )
    fixed_comparison = {
        "actual": "onnxruntime_cpu",
        "atol": 1.0e-5,
        "case_count": 82,
        "expected": "checkpoint_torch_cpu",
        "passed": True,
        "rtol": 1.0e-5,
    }
    for key, expected in fixed_comparison.items():
        if comparison[key] != expected or type(comparison[key]) is not type(expected):
            raise ValueError(f"qualification export parity {key!r} mismatch")
    for key, maximum in (("max_absolute_error", 5.0e-5), ("max_relative_error", 1.0e-3)):
        error = comparison[key]
        if type(error) not in (int, float) or not math.isfinite(error) or not 0.0 <= error <= maximum:
            raise ValueError(f"qualification export parity {key!r} invalid")
    if type(comparison["worst_absolute_case"]) is not str or not comparison["worst_absolute_case"]:
        raise ValueError("qualification export parity worst case missing")
    onnx_proof = _require_mapping_keys(
        proof["onnx"],
        {
            "checker_full_check",
            "checkpoint_initializers_exact",
            "input",
            "normalizer_divisor_exact",
            "onnx_opset",
            "operation_types",
            "output",
            "path",
            "sha256",
            "shape_inference_strict",
        },
        "export.parity.onnx",
    )
    if (
        onnx_proof
        != {
            "checker_full_check": True,
            "checkpoint_initializers_exact": True,
            "input": {"dtype": "float32", "name": "obs", "shape": [1, 124]},
            "normalizer_divisor_exact": True,
            "onnx_opset": 18,
            "operation_types": ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"],
            "output": {"dtype": "float32", "name": "actions", "shape": [1, 23]},
            "path": onnx_proof.get("path"),
            "sha256": actor_onnx_sha256,
            "shape_inference_strict": True,
        }
        or type(onnx_proof["path"]) is not str
        or not onnx_proof["path"]
    ):
        raise ValueError("qualification export ONNX proof mismatch")
    probe_suite = _require_mapping_keys(
        parity_root["probe_suite"],
        {"adversarial_case_count", "case_count", "random_case_count", "seed"},
        "export.parity.probe_suite",
    )
    if (
        probe_suite["adversarial_case_count"] != 34
        or probe_suite["case_count"] != 82
        or probe_suite["random_case_count"] != 48
    ):
        raise ValueError("qualification export probe suite count mismatch")
    _require_exact_int(probe_suite["seed"], "export.parity.probe_suite.seed")
    runtime = _require_mapping_keys(
        report["runtime"],
        {"device", "environment_constructed", "mjlab_imported", "numpy_version", "torch_version"},
        "export.runtime",
    )
    if (
        runtime["device"] != "cpu"
        or runtime["environment_constructed"] is not False
        or runtime["mjlab_imported"] is not False
    ):
        raise ValueError("qualification export runtime boundary mismatch")
    for key in ("numpy_version", "torch_version"):
        if type(runtime[key]) is not str or not runtime[key]:
            raise ValueError("qualification export runtime version missing")


def _validate_diagnostic_report(binding: Any, *, policy_sha256: str, required_phase: str) -> None:
    report, _ = read_bound_json(binding, "gate.diagnostic_report")
    _require_mapping_keys(
        report,
        {
            "actor_joint_order",
            "control_hz",
            "deployment_ready",
            "diagnostic_only",
            "evidence_class",
            "env_params",
            "joint_names_hardware",
            "kind",
            "observation_layout",
            "policy",
            "robot_commands_performed",
            "runs",
            "schema_version",
            "stop_on_first_termination",
            "training_authorized",
        },
        "curriculum diagnostic report",
    )
    expected_top = {
        "schema_version": SCHEMA_VERSION,
        "kind": "g1_true23_step1c_external_reference_onnx_diagnostic_v1",
        "evidence_class": DIAGNOSTIC_EVIDENCE_CLASS,
        "control_hz": CONTROL_HZ,
        "diagnostic_only": True,
        "training_authorized": False,
        "deployment_ready": False,
        "robot_commands_performed": False,
    }
    for key, expected in expected_top.items():
        if report.get(key) != expected or type(report.get(key)) is not type(expected):
            raise ValueError(f"curriculum diagnostic report {key!r} mismatch")
    if report["actor_joint_order"] != "unitree_mjlab_hardware_order":
        raise ValueError("curriculum diagnostic actor joint order mismatch")
    if report["joint_names_hardware"] != list(HARDWARE_JOINT_NAMES):
        raise ValueError("curriculum diagnostic hardware joint list mismatch")
    if report["observation_layout"] != [list(row) for row in OBSERVATION_LAYOUT]:
        raise ValueError("curriculum diagnostic observation layout mismatch")
    if type(report["stop_on_first_termination"]) is not bool:
        raise ValueError("curriculum diagnostic stop flag must be boolean")
    env_params = _require_mapping_keys(
        report["env_params"],
        {"action_scale_hardware", "home_hardware", "path", "sha256"},
        "diagnostic.env_params",
    )
    if sha256_file(DEPLOY_PARAMS_PATH) != DEPLOY_PARAMS_SHA256 or env_params["sha256"] != DEPLOY_PARAMS_SHA256:
        raise ValueError("curriculum diagnostic deploy-parameter SHA256 mismatch")
    if type(env_params["path"]) is not str or not env_params["path"].replace("\\", "/").endswith(
        "/deploy/robots/g1_23dof/config/policy/mimic/B_DadDance/params/deploy.yaml"
    ):
        raise ValueError("curriculum diagnostic deploy-parameter path mismatch")
    for key, positive in (("action_scale_hardware", True), ("home_hardware", False)):
        values = env_params[key]
        if not isinstance(values, list) or len(values) != 23:
            raise ValueError(f"curriculum diagnostic {key} shape mismatch")
        if any(type(item) not in (int, float) or not math.isfinite(item) for item in values):
            raise ValueError(f"curriculum diagnostic {key} contains invalid numeric")
        if positive and any(item <= 0 for item in values):
            raise ValueError("curriculum diagnostic action scale must be positive")
    policy = _require_mapping_keys(
        report["policy"],
        {"input_shape", "output_shape", "path", "sha256"},
        "diagnostic.policy",
    )
    if policy["sha256"] != policy_sha256:
        raise ValueError("curriculum diagnostic policy SHA256 mismatch")
    if policy["input_shape"] != [1, 124] or policy["output_shape"] != [1, 23]:
        raise ValueError("curriculum diagnostic policy shape mismatch")
    if type(policy["path"]) is not str or not policy["path"]:
        raise ValueError("curriculum diagnostic policy path missing")
    mode = "static_frame0_zero_velocity" if required_phase == "static" else "full_terminal_hold_zero_velocity"
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError("curriculum diagnostic runs must be list")
    run_keys = {
        "clip_id",
        "contact_steps",
        "first_shipped_termination",
        "horizon_steps",
        "joint_hard_limit_violation_steps",
        "max_abs_joint_tracking_error_rad",
        "mode",
        "peak_abs_torque_nm",
        "seed",
        "steps_executed",
        "torque_saturation_control_steps",
    }
    for run in runs:
        run = _require_mapping_keys(run, run_keys, "diagnostic.run")
        if run["clip_id"] not in QUALIFICATION_CLIPS:
            raise ValueError("curriculum diagnostic run clip ID mismatch")
        if run["mode"] not in {
            "static_frame0_zero_velocity",
            "full_terminal_hold_zero_velocity",
        }:
            raise ValueError("curriculum diagnostic run mode mismatch")
        expected_seed = 2729 if run["clip_id"] == CHANGE_IDLE else 1729
        if _require_exact_int(run["seed"], "diagnostic.run.seed") != expected_seed:
            raise ValueError("curriculum diagnostic run seed mismatch")
        for key in (
            "horizon_steps",
            "steps_executed",
            "joint_hard_limit_violation_steps",
            "torque_saturation_control_steps",
        ):
            _require_exact_int(run[key], f"diagnostic.run.{key}")
        for key in ("max_abs_joint_tracking_error_rad", "peak_abs_torque_nm"):
            number = run[key]
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError(f"curriculum diagnostic run {key} invalid")
        contacts = _require_mapping_keys(
            run["contact_steps"], {"left_foot", "right_foot"}, "diagnostic.run.contact_steps"
        )
        for key in ("left_foot", "right_foot"):
            count = _require_exact_int(contacts[key], f"diagnostic.run.contact_steps.{key}")
            if count > EPISODE_STEPS:
                raise ValueError("curriculum diagnostic contact count exceeds horizon")
    selected = [run for run in runs if run["mode"] == mode]
    if len(selected) != len(QUALIFICATION_CLIPS):
        raise ValueError("curriculum diagnostic lacks exact two mode runs")
    if sorted(run["clip_id"] for run in selected) != sorted(QUALIFICATION_CLIPS):
        raise ValueError("curriculum diagnostic clip set mismatch")
    for run in selected:
        if run["horizon_steps"] != EPISODE_STEPS or run["steps_executed"] != EPISODE_STEPS:
            raise ValueError("curriculum diagnostic did not execute full 500-step horizon")
        if run["first_shipped_termination"] is not None:
            raise ValueError("curriculum diagnostic contains shipped termination")
        if run["joint_hard_limit_violation_steps"] != 0:
            raise ValueError("curriculum diagnostic contains joint hard-limit violation")
        if any(run["contact_steps"][foot] != EPISODE_STEPS for foot in ("left_foot", "right_foot")):
            raise ValueError("curriculum diagnostic has stance contact mismatch")


def _validate_gate(
    path: Path,
    *,
    parent_checkpoint_path: Path,
    parent_sha256: str,
    required_phase: str,
    expected_gate_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate self-attested curriculum diagnostics; never unlock training."""

    gate, gate_sha256, gate_size, resolved_gate = read_json_snapshot(
        path,
        expected_sha256=expected_gate_sha256,
    )
    gate_binding = {
        "path": _repo_relative(resolved_gate),
        "sha256": gate_sha256,
        "size_bytes": gate_size,
    }
    required_keys = {
        "schema_version",
        "kind",
        "qualified_phase",
        "checkpoint_sha256",
        "contact_evidence_scope",
        "corpus_sha256",
        "evidence_class",
        "horizon_steps",
        "clip_ids",
        "passed",
        "no_terminations",
        "no_hard_limit_violations",
        "actor_onnx",
        "export_report",
        "diagnostic_report",
    }
    if set(gate) != required_keys:
        raise ValueError("qualification gate keys mismatch")
    if (
        _require_exact_int(gate["schema_version"], "qualification gate schema_version") != SCHEMA_VERSION
        or gate["kind"] != GATE_KIND
    ):
        raise ValueError("qualification gate schema mismatch")
    parent_sha256 = _require_sha256(parent_sha256, "qualification gate parent SHA256")
    checks = {
        "qualified_phase": required_phase,
        "checkpoint_sha256": parent_sha256,
        "contact_evidence_scope": CONTACT_EVIDENCE_SCOPE,
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "evidence_class": DIAGNOSTIC_EVIDENCE_CLASS,
        "horizon_steps": EPISODE_STEPS,
        "clip_ids": list(QUALIFICATION_CLIPS),
        "passed": True,
        "no_terminations": True,
        "no_hard_limit_violations": True,
    }
    for key, expected in checks.items():
        if gate.get(key) != expected or type(gate.get(key)) is not type(expected):
            raise ValueError(f"qualification gate {key!r} mismatch")

    actor_onnx_sha256 = validate_actor_onnx(gate["actor_onnx"])
    actor_onnx_path = _resolve_repo_relative(gate["actor_onnx"]["path"], "gate.actor_onnx.path")
    from gear_sonic.utils.g1_23dof_native124_actor_export import (
        load_native124_actor,
        verify_actor_onnx_parity,
    )

    actor, lineage = load_native124_actor(parent_checkpoint_path)
    checkpoint_lineage = lineage.get("checkpoint")
    if not isinstance(checkpoint_lineage, Mapping) or checkpoint_lineage.get("sha256") != parent_sha256:
        raise ValueError("qualification actor checkpoint lineage mismatch")
    independent_proof = verify_actor_onnx_parity(
        actor,
        actor_onnx_path,
        expected_sha256=actor_onnx_sha256,
    )
    comparison = _require_mapping_keys(
        independent_proof.get("comparison"),
        {
            "actual",
            "atol",
            "case_count",
            "expected",
            "max_absolute_error",
            "max_relative_error",
            "passed",
            "rtol",
            "worst_absolute_case",
        },
        "independent actor parity comparison",
    )
    fixed_comparison = {
        "actual": "onnxruntime_cpu",
        "atol": 1.0e-5,
        "case_count": 82,
        "expected": "checkpoint_torch_cpu",
        "passed": True,
        "rtol": 1.0e-5,
    }
    for key, expected in fixed_comparison.items():
        if comparison[key] != expected or type(comparison[key]) is not type(expected):
            raise ValueError(f"independent actor parity {key!r} mismatch")
    for key in ("max_absolute_error", "max_relative_error"):
        error = comparison[key]
        if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
            raise ValueError(f"independent actor parity {key!r} invalid")
    if type(comparison["worst_absolute_case"]) is not str or not comparison["worst_absolute_case"]:
        raise ValueError("independent actor parity worst case missing")
    independent_onnx = _require_mapping_keys(
        independent_proof.get("onnx"),
        {
            "checker_full_check",
            "checkpoint_initializers_exact",
            "input",
            "normalizer_divisor_exact",
            "onnx_opset",
            "operation_types",
            "output",
            "path",
            "sha256",
            "shape_inference_strict",
        },
        "independent actor parity ONNX proof",
    )
    expected_onnx_proof = {
        "checker_full_check": True,
        "checkpoint_initializers_exact": True,
        "input": {"dtype": "float32", "name": "obs", "shape": [1, 124]},
        "normalizer_divisor_exact": True,
        "onnx_opset": 18,
        "operation_types": ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"],
        "output": {"dtype": "float32", "name": "actions", "shape": [1, 23]},
        "path": str(actor_onnx_path),
        "sha256": actor_onnx_sha256,
        "shape_inference_strict": True,
    }
    if dict(independent_onnx) != expected_onnx_proof:
        raise ValueError("independent actor parity ONNX proof mismatch")
    if sha256_file(actor_onnx_path) != actor_onnx_sha256:
        raise RuntimeError("qualification actor ONNX changed during independent parity")

    _validate_export_report(
        gate["export_report"],
        checkpoint_sha256=parent_sha256,
        actor_onnx_sha256=actor_onnx_sha256,
    )
    _validate_diagnostic_report(
        gate["diagnostic_report"],
        policy_sha256=actor_onnx_sha256,
        required_phase=required_phase,
    )
    return gate, gate_binding


def _wrap_plan(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "payload_sha256": payload_sha256(payload),
        "payload": payload,
    }


def validate_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "kind",
        "payload_sha256",
        "payload",
    }:
        raise ValueError("plan wrapper keys mismatch")
    if (
        _require_exact_int(value["schema_version"], "plan.schema_version") != SCHEMA_VERSION
        or value["kind"] != PLAN_KIND
    ):
        raise ValueError("plan wrapper schema mismatch")
    payload = value["payload"]
    if not isinstance(payload, dict):
        raise ValueError("plan payload must be object")
    payload_keys = {
        "schema_version",
        "phase_name",
        "control_hz",
        "episode_steps",
        "num_envs",
        "num_steps_per_env",
        "training_seed",
        "optimizer",
        "qualification",
        "corpus",
        "sidecar",
        "corpus_flags",
        "runtime_files",
        "pinned_package_versions",
        "observed_package_versions",
        "jobs",
        "safety",
        "execution",
    }
    if set(payload) != payload_keys:
        raise ValueError("plan payload keys mismatch")
    expected = _require_sha256(value["payload_sha256"], "plan.payload_sha256")
    if payload_sha256(payload) != expected:
        raise ValueError("plan payload hash mismatch")
    phase_name = payload.get("phase_name")
    if phase_name not in PHASES:
        raise ValueError("plan phase is unsupported")
    if phase_name != "static":
        raise ValueError(FUTURE_EXECUTION_BLOCKER)
    phase = PHASES[phase_name]
    if payload.get("corpus_flags") != {
        "diagnostic_only": True,
        "training_authorized": False,
    }:
        raise ValueError("plan must preserve immutable corpus flags")
    safety = payload.get("safety")
    if safety != {
        "actor_observation_corruption": False,
        "command_joint_position_randomization": False,
        "command_pose_randomization": False,
        "command_velocity_randomization": False,
        "domain_randomization_events": False,
        "episode_length_randomization": False,
        "hardware_commands": False,
        "simulator_only": True,
    }:
        raise ValueError("plan safety contract mismatch")
    fixed_scalars = {
        "schema_version": SCHEMA_VERSION,
        "control_hz": CONTROL_HZ,
        "episode_steps": EPISODE_STEPS,
        "num_envs": DEFAULT_NUM_ENVS,
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "training_seed": TRAINING_SEED,
    }
    for key, fixed_value in fixed_scalars.items():
        if payload.get(key) != fixed_value or type(payload.get(key)) is not type(fixed_value):
            raise ValueError(f"plan fixed field {key!r} mismatch")
    expected_optimizer = {
        "seed_optimizer_loaded": False,
        "phase_parent_optimizer_loaded": False,
        "learning_rate": 1.0e-4,
        "entropy_coef": 0.002,
        "max_grad_norm": 0.5,
    }
    if payload.get("optimizer") != expected_optimizer:
        raise ValueError("plan optimizer contract mismatch")
    expected_execution = {
        "available": False,
        "default": "dry_run",
        "preflight_only": True,
        "blocker": FUTURE_EXECUTION_BLOCKER,
        "future_execution_blockers": [
            FUTURE_EXECUTION_BLOCKER,
            "installed rsl_rl runner source is outside repository and not byte-bound",
            "production path/SHA loader and full MJLab environment-state adapter not implemented",
        ],
    }
    if payload.get("execution") != expected_execution:
        raise ValueError("plan execution boundary mismatch")
    if payload.get("pinned_package_versions") != PINNED_PACKAGE_VERSIONS:
        raise ValueError("plan pinned package versions mismatch")
    observed = payload.get("observed_package_versions")
    if not isinstance(observed, Mapping) or set(observed) != set(PINNED_PACKAGE_VERSIONS):
        raise ValueError("plan observed package version keys mismatch")
    if dict(observed) != observed_package_versions():
        raise ValueError("planner package metadata changed")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("plan requires jobs")
    ids = [job.get("job_id") for job in jobs if isinstance(job, Mapping)]
    if (
        len(ids) != len(jobs)
        or any(type(job_id) is not str or not job_id for job_id in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("plan job IDs must be unique strings")
    corpus = payload.get("corpus")
    sidecar = payload.get("sidecar")
    _, expected_corpus_path = _validate_sidecar(DEFAULT_SIDECAR)
    if not isinstance(corpus, Mapping) or corpus != file_binding(expected_corpus_path):
        raise ValueError("plan corpus binding is not pinned")
    if not isinstance(sidecar, Mapping) or sidecar != file_binding(DEFAULT_SIDECAR):
        raise ValueError("plan sidecar binding is not pinned")
    if payload.get("runtime_files") != [file_binding(path) for path in RUNTIME_FILES]:
        raise ValueError("plan runtime file list/bindings mismatch")
    qualification = payload.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ValueError("plan qualification must be object")
    expected_qualification = {
        "clip_ids": list(QUALIFICATION_CLIPS),
        "contact_evidence_scope": CONTACT_EVIDENCE_SCOPE,
        "evidence_class": DIAGNOSTIC_EVIDENCE_CLASS,
        "gate_scope": "self_attested_curriculum_diagnostic_never_unlocks_training",
        "horizon_steps": EPISODE_STEPS,
        "mode": phase["qualification_mode"],
        "required_gate_phase": phase["required_gate_phase"],
        "gate": None,
        "unlocks_later_phases": False,
    }
    if payload.get("qualification") != expected_qualification:
        raise ValueError("plan qualification contract mismatch")
    job_keys = {
        "job_id",
        "phase_name",
        "phase",
        "start_mode",
        "planned_updates",
        "session_updates",
        "source_checkpoint",
        "source_checkpoint_expected_iter",
        "source_kind",
        "qualification_gate",
    }
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ValueError("plan job must be object")
        if set(job) != job_keys:
            raise ValueError("plan job keys mismatch")
        expected_job_values = {
            "phase_name": phase_name,
            "phase": phase["phase"],
            "start_mode": phase["start_mode"],
            "planned_updates": phase["planned_updates"],
            "session_updates": SESSION_UPDATES,
        }
        for key, expected_value in expected_job_values.items():
            if job.get(key) != expected_value or type(job.get(key)) is not type(expected_value):
                raise ValueError(f"plan job {key!r} mismatch")
        verify_file_binding(job.get("source_checkpoint"), "plan.job.source_checkpoint")
        if job.get("qualification_gate") is not None:
            verify_file_binding(job["qualification_gate"], "plan.job.qualification_gate")
    if ids != ["static_model3500", "static_model11500"]:
        raise ValueError("static A/B job IDs/order mismatch")
    for job, (seed_id, (path, checkpoint_sha256, checkpoint_iter)) in zip(
        jobs, SEED_CHECKPOINTS.items(), strict=True
    ):
        if job["job_id"] != f"static_{seed_id}":
            raise ValueError("static A/B seed/job mismatch")
        if job["source_checkpoint"] != file_binding(path):
            raise ValueError("static A/B source checkpoint binding mismatch")
        if job["source_checkpoint"]["sha256"] != checkpoint_sha256:
            raise ValueError("static A/B source checkpoint SHA256 mismatch")
        if job["source_checkpoint_expected_iter"] != checkpoint_iter:
            raise ValueError("static A/B source checkpoint iter mismatch")
        if job["source_kind"] != "dad_dance_seed" or job["qualification_gate"] is not None:
            raise ValueError("static A/B source/gate contract mismatch")
    return dict(value)


def build_plan(
    *,
    phase_name: str = "static",
    sidecar_path: Path = DEFAULT_SIDECAR,
    parent_checkpoint: Path | None = None,
    qualification_gate: Path | None = None,
    num_envs: int = DEFAULT_NUM_ENVS,
) -> dict[str, Any]:
    if phase_name not in PHASES:
        raise ValueError(f"unknown phase: {phase_name}")
    if phase_name != "static":
        raise ValueError(FUTURE_EXECUTION_BLOCKER)
    if type(num_envs) is not int or num_envs != DEFAULT_NUM_ENVS:
        raise ValueError(f"num_envs must be fixed at {DEFAULT_NUM_ENVS}")
    phase = dict(PHASES[phase_name])
    sidecar, corpus_path = _validate_sidecar(sidecar_path)
    gate_binding: dict[str, Any] | None = None
    jobs: list[dict[str, Any]] = []
    if parent_checkpoint is not None or qualification_gate is not None:
        raise ValueError("static plan does not accept parent checkpoint or gate")
    for seed_id, (path, expected_hash, expected_iter) in SEED_CHECKPOINTS.items():
        binding = file_binding(path)
        if binding["sha256"] != expected_hash:
            raise ValueError(f"{seed_id} checkpoint differs from exact pinned bytes")
        jobs.append(
            {
                "job_id": f"static_{seed_id}",
                "phase_name": phase_name,
                "phase": phase["phase"],
                "start_mode": phase["start_mode"],
                "planned_updates": phase["planned_updates"],
                "session_updates": SESSION_UPDATES,
                "source_checkpoint": binding,
                "source_checkpoint_expected_iter": expected_iter,
                "source_kind": "dad_dance_seed",
                "qualification_gate": None,
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase_name": phase_name,
        "control_hz": CONTROL_HZ,
        "episode_steps": EPISODE_STEPS,
        "num_envs": num_envs,
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "training_seed": TRAINING_SEED,
        "optimizer": {
            "seed_optimizer_loaded": False,
            "phase_parent_optimizer_loaded": False,
            "learning_rate": 1.0e-4,
            "entropy_coef": 0.002,
            "max_grad_norm": 0.5,
        },
        "qualification": {
            "clip_ids": list(QUALIFICATION_CLIPS),
            "contact_evidence_scope": CONTACT_EVIDENCE_SCOPE,
            "evidence_class": DIAGNOSTIC_EVIDENCE_CLASS,
            "gate_scope": "self_attested_curriculum_diagnostic_never_unlocks_training",
            "horizon_steps": EPISODE_STEPS,
            "mode": phase["qualification_mode"],
            "required_gate_phase": phase["required_gate_phase"],
            "gate": gate_binding,
            "unlocks_later_phases": False,
        },
        "corpus": file_binding(corpus_path),
        "sidecar": file_binding(sidecar_path),
        "corpus_flags": {
            "diagnostic_only": sidecar["diagnostic_only"],
            "training_authorized": sidecar["training_authorized"],
        },
        "runtime_files": [file_binding(path) for path in RUNTIME_FILES],
        "pinned_package_versions": dict(PINNED_PACKAGE_VERSIONS),
        "observed_package_versions": observed_package_versions(),
        "jobs": jobs,
        "safety": {
            "actor_observation_corruption": False,
            "command_joint_position_randomization": False,
            "command_pose_randomization": False,
            "command_velocity_randomization": False,
            "domain_randomization_events": False,
            "episode_length_randomization": False,
            "hardware_commands": False,
            "simulator_only": True,
        },
        "execution": {
            "available": False,
            "default": "dry_run",
            "preflight_only": True,
            "blocker": FUTURE_EXECUTION_BLOCKER,
            "future_execution_blockers": [
                FUTURE_EXECUTION_BLOCKER,
                "installed rsl_rl runner source is outside repository and not byte-bound",
                "production path/SHA loader and full MJLab environment-state adapter not implemented",
            ],
        },
    }
    return validate_plan(_wrap_plan(payload))


def _find_job(plan: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    jobs = plan["payload"]["jobs"]
    matches = [dict(job) for job in jobs if job.get("job_id") == job_id]
    if len(matches) != 1:
        raise ValueError(f"plan contains no unique job {job_id!r}")
    return matches[0]


def validate_authorization(
    path: Path,
    *,
    expected_file_sha256: str,
    plan: Mapping[str, Any],
    job_id: str,
) -> dict[str, Any]:
    expected_file_sha256 = _require_sha256(expected_file_sha256, "authorization_file_sha256")
    authorization, _, _, _ = read_json_snapshot(
        path,
        expected_sha256=expected_file_sha256,
    )
    required_keys = {
        "schema_version",
        "kind",
        "authorized",
        "plan_payload_sha256",
        "job_id",
        "corpus_sha256",
        "simulator_only",
        "hardware_commands_authorized",
        "corpus_diagnostic_flag_acknowledged",
    }
    if set(authorization) != required_keys:
        raise ValueError("authorization keys mismatch")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "kind": AUTHORIZATION_KIND,
        "authorized": True,
        "plan_payload_sha256": plan["payload_sha256"],
        "job_id": job_id,
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "simulator_only": True,
        "hardware_commands_authorized": False,
        "corpus_diagnostic_flag_acknowledged": True,
    }
    for key, value in expected.items():
        if authorization.get(key) != value or type(authorization.get(key)) is not type(value):
            raise ValueError(f"authorization {key!r} mismatch")
    return authorization


def safe_load_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_iter: int | None,
) -> dict[str, Any]:
    """Load only tensor/container checkpoint types; never allow pickle globals."""

    expected_sha256 = _require_sha256(expected_sha256, "checkpoint SHA256")
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ValueError("checkpoint must not be symlink")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise ValueError("checkpoint bytes differ from exact plan binding")
    import torch

    try:
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("checkpoint is not safe weights-only data") from error
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
    }:
        raise ValueError("checkpoint root schema mismatch")
    if type(checkpoint["iter"]) is not int or checkpoint["iter"] < 0:
        raise ValueError("checkpoint iter must be non-negative integer")
    if expected_iter is not None and checkpoint["iter"] != expected_iter:
        raise ValueError("checkpoint iter differs from exact seed contract")
    for group_name, expected_schema in (
        ("actor_state_dict", ACTOR_TENSOR_SCHEMA),
        ("critic_state_dict", CRITIC_TENSOR_SCHEMA),
    ):
        state = checkpoint[group_name]
        if not isinstance(state, Mapping) or set(state) != set(expected_schema):
            raise ValueError(f"checkpoint {group_name} keys mismatch")
        for key, (expected_shape, expected_dtype) in expected_schema.items():
            tensor = state[key]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"checkpoint {group_name}.{key} is not tensor")
            if tuple(tensor.shape) != expected_shape or str(tensor.dtype) != expected_dtype:
                raise ValueError(f"checkpoint {group_name}.{key} shape/dtype mismatch")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"checkpoint {group_name}.{key} contains non-finite values")
        if int(state["obs_normalizer.count"].item()) < 0:
            raise ValueError(f"checkpoint {group_name} normalizer count is negative")
    if sha256_file(resolved) != expected_sha256:
        raise RuntimeError("checkpoint changed while being safely loaded")
    return checkpoint


def completed_updates_from_checkpoint(checkpoint: Mapping[str, Any], *, resume: bool) -> int:
    """Return next RSL-RL iteration index without repeating saved iteration.

    RSL-RL saves ``iter`` after that update, then naively resumes at the same
    value.  Our own checkpoints persist the unambiguous completed-update count.
    Seed/phase initialization deliberately resets to zero.
    """

    if not resume:
        return 0
    infos = checkpoint.get("infos")
    metadata = infos.get("supported_idle") if isinstance(infos, Mapping) else None
    metadata = _require_mapping_keys(
        metadata,
        {
            "completed_updates",
            "corpus_sha256",
            "job_id",
            "kind",
            "plan_payload_sha256",
            "schema_version",
            "source_checkpoint_sha256",
        },
        "resume checkpoint supported_idle metadata",
    )
    if (
        metadata["kind"] != CHECKPOINT_INFO_KIND
        or _require_exact_int(metadata["schema_version"], "resume metadata schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("resume checkpoint lacks supported-idle update metadata")
    completed = _require_exact_int(
        metadata["completed_updates"],
        "resume completed_updates",
        minimum=1,
    )
    if type(metadata["job_id"]) is not str or not metadata["job_id"]:
        raise ValueError("resume metadata job_id must be non-empty string")
    _require_sha256(metadata["plan_payload_sha256"], "resume metadata plan_payload_sha256")
    _require_sha256(metadata["source_checkpoint_sha256"], "resume metadata source_checkpoint_sha256")
    if metadata["corpus_sha256"] != EXPECTED_CORPUS_SHA256:
        raise ValueError("resume metadata corpus SHA256 mismatch")
    if checkpoint.get("iter") != completed - 1:
        raise ValueError("resume iter/completed_updates off-by-one contract mismatch")
    return completed


def apply_unperturbed_env_contract(env_cfg: Any, *, num_envs: int, seed: int) -> Any:
    """Apply fail-closed no-perturbation settings without importing MJLab.

    Duck typing keeps this contract CPU-testable. Runtime separately verifies
    that motion command is ``SupportedIdleMotionCommandCfg``.
    """

    if type(num_envs) is not int or num_envs < 1 or type(seed) is not int:
        raise ValueError("invalid deterministic environment dimensions/seed")
    try:
        actor_observations = env_cfg.observations["actor"]
        motion_cfg = env_cfg.commands["motion"]
        events = env_cfg.events
        scene = env_cfg.scene
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("environment config lacks native tracking contract") from error
    if not hasattr(events, "clear"):
        raise ValueError("environment events collection cannot be cleared")
    env_cfg.seed = seed
    scene.num_envs = num_envs
    actor_observations.enable_corruption = False
    events.clear()
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)
    motion_cfg.debug_vis = False
    return env_cfg


def preflight_run(
    *,
    plan_path: Path,
    plan_file_sha256: str,
    authorization_path: Path,
    authorization_file_sha256: str,
    job_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_file_sha256 = _require_sha256(plan_file_sha256, "plan_file_sha256")
    plan_snapshot, _, _, _ = read_json_snapshot(
        plan_path,
        expected_sha256=plan_file_sha256,
    )
    plan = validate_plan(plan_snapshot)
    job = _find_job(plan, job_id)
    authorization = validate_authorization(
        authorization_path,
        expected_file_sha256=authorization_file_sha256,
        plan=plan,
        job_id=job_id,
    )
    payload = plan["payload"]
    verify_file_binding(payload["sidecar"], "plan.sidecar")
    verify_file_binding(payload["corpus"], "plan.corpus")
    for index, binding in enumerate(payload["runtime_files"]):
        verify_file_binding(binding, f"plan.runtime_files[{index}]")
    source = verify_file_binding(job["source_checkpoint"], "job.source_checkpoint")
    source_checkpoint = safe_load_checkpoint(
        source,
        expected_sha256=job["source_checkpoint"]["sha256"],
        expected_iter=job["source_checkpoint_expected_iter"],
    )
    if job["source_kind"] == "qualified_phase_parent":
        completed_updates_from_checkpoint(source_checkpoint, resume=True)
        parent_metadata = source_checkpoint["infos"]["supported_idle"]
        required_parent_phase = PHASES[job["phase_name"]]["required_gate_phase"]
        assert isinstance(required_parent_phase, str)
        parent_job_id = parent_metadata.get("job_id")
        if required_parent_phase == "static":
            static_parent_sources = {
                f"static_{seed_id}": checkpoint[1] for seed_id, checkpoint in SEED_CHECKPOINTS.items()
            }
            if parent_job_id not in static_parent_sources:
                raise ValueError("trajectory-start parent is not a static A/B checkpoint")
            if parent_metadata["source_checkpoint_sha256"] != static_parent_sources[parent_job_id]:
                raise ValueError("static parent source checkpoint lineage mismatch")
        elif parent_job_id != required_parent_phase:
            raise ValueError("phase parent checkpoint lineage mismatch")
        gate_binding = job["qualification_gate"]
        gate_path = _resolve_repo_relative(gate_binding["path"], "job.qualification_gate.path")
        _, verified_gate_binding = _validate_gate(
            gate_path,
            parent_checkpoint_path=source,
            parent_sha256=job["source_checkpoint"]["sha256"],
            required_phase=required_parent_phase,
            expected_gate_sha256=gate_binding["sha256"],
        )
        if verified_gate_binding != gate_binding:
            raise ValueError("job qualification gate binding changed")
    return plan, job, authorization


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    plan_parser = subparsers.add_parser("plan", help="Print read-only deterministic plan.")
    plan_parser.add_argument("--phase", choices=tuple(PHASES), default="static")
    plan_parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    plan_parser.add_argument("--parent-checkpoint", type=Path)
    plan_parser.add_argument("--qualification-gate", type=Path)
    plan_parser.add_argument("--num-envs", type=_positive_int, default=DEFAULT_NUM_ENVS)

    preflight_parser = subparsers.add_parser(
        "preflight", help="Verify exact static plan, authorization, and seed checkpoint on CPU."
    )
    _add_plan_authorization_arguments(preflight_parser)

    return parser


def _add_plan_authorization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-file-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-file-sha256", required=True)
    parser.add_argument("--job-id", required=True)


def _print_plan(args: argparse.Namespace) -> int:
    plan = build_plan(
        phase_name=args.phase,
        sidecar_path=args.sidecar,
        parent_checkpoint=args.parent_checkpoint,
        qualification_gate=args.qualification_gate,
        num_envs=args.num_envs,
    )
    output = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    print(output, end="")  # noqa: T201
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    values = list(argv) if argv is not None else sys.argv[1:]
    if not values:
        values = ["plan"]
    args = parser.parse_args(values)
    if args.command == "plan":
        return _print_plan(args)
    if args.command == "preflight":
        plan, job, authorization = preflight_run(
            plan_path=args.plan,
            plan_file_sha256=args.plan_file_sha256,
            authorization_path=args.authorization,
            authorization_file_sha256=args.authorization_file_sha256,
            job_id=args.job_id,
        )
        report = {
            "preflight_valid": True,
            "ready_for_execution": False,
            "simulator_only": True,
            "hardware_commands": False,
            "plan_payload_sha256": plan["payload_sha256"],
            "job_id": job["job_id"],
            "authorization_kind": authorization["kind"],
            "execution_available": False,
            "execution_blocker": FUTURE_EXECUTION_BLOCKER,
        }
        print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201
        return 0
    parser.error("command required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
