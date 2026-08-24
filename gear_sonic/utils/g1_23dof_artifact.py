"""Fail-closed offline export for trained G1 23-DoF decoder artifacts.

This module never imports Unitree DDS or creates a robot command publisher.  It
binds an exact trained checkpoint to explicit simulation evidence, reconstructs
the decoder-only MLP, exports static float32 ONNX, and publishes the model only
after ONNX checker, shape inference, and ONNX Runtime parity all pass.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any
import warnings
import xml.etree.ElementTree as ET

from gear_sonic.utils.g1_23dof_checkpoint_io import (
    extract_global_step,
    load_safe_true23_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import (
    APPROVED_WARM_START_RELEASES,
    ARTIFACT_SCHEMA_VERSION,
    CANONICAL_COMPACT_IL23_JOINT_NAMES,
    DECODER_OUTPUT_LAYOUT,
    DEFAULT_REFERENCE_PROFILE,
    DEPLOYMENT_DECODER_INPUT_DIM,
    DEPLOYMENT_HISTORY_LENGTH,
    EXCLUDED_HARDWARE_JOINT_IDS,
    HARDWARE_23_ACTION_SCALE,
    HARDWARE_23_JOINT_NAMES,
    HARDWARE_JOINT_IDS,
    HISTORY_ORDER,
    ISAACLAB_TO_MUJOCO_DOF,
    MINIMUM_TRAINING_UPDATES,
    MISSING_OBSERVATION_FILL,
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_ACTION_SCALE,
    NATIVE_IL23_JOINT_NAMES,
    NATIVE_IL23_TO_CANONICAL_IL29,
    OBS_LAYOUT_PADDED_IL29,
    OBSERVATION_TERM_ORDER,
    REFERENCE_PROFILE_LOW_LATENCY,
    REFERENCE_PROFILE_NORMAL,
    REFERENCE_PROFILES,
    REQUIRED_MODE_MACHINE,
    REQUIRED_SIM_SCENARIOS,
    ROBOT_MODEL,
    SIM_VALIDATION_SCHEMA_VERSION,
    SOURCE_IL29_EXCLUDED_INDICES,
    SOURCE_IL29_JOINT_NAMES,
    SOURCE_IL29_KEEP_INDICES,
    SOURCE_MJ29_TO_TARGET_IL23,
    TARGET_DOF,
    TELEOP_ENCODER_INPUT_DIM,
    TELEOP_ENCODER_INPUT_TERM_DIMS,
    TELEOP_ENCODER_INPUT_TERM_ORDER,
    TELEOP_FSQ_LEVEL,
    TELEOP_TOKEN_COUNT,
    TELEOP_TOKEN_WIDTH,
    TOKEN_DIM,
    decoder_shape,
    make_artifact_metadata,
    reference_profile_contract,
    validate_artifact_contract,
)

SIM_REPORT_KIND = "g1_23dof_sim_validation"
SIM_TRACE_KIND = "g1_23dof_raw_isaaclab_trace"
SIM_TRACE_SCHEMA_VERSION = 2
MATERIAL_PROVENANCE_SCHEMA_VERSION = 1
FILE_MANIFEST_SCHEMA_VERSION = 1
MOTION_DATASET_SCHEMA_VERSION = 1
TRAINING_EVIDENCE_SCHEMA_VERSION = 3
TRAINING_MATERIAL_SCHEMA_VERSION = 1
ARTIFACT_KIND = "g1_23dof_validated_teleop_encoder_decoder_onnx_pair"
TRAINING_EVIDENCE_KIND = "g1_23dof_training_checkpoint"
TRAINING_EVIDENCE_PRODUCER = "gear_sonic.trl.callbacks.ModelSaveCallback"
APPROVED_TRAINING_MATERIAL_CONFIG_SHA256_BY_REFERENCE_PROFILE = {
    REFERENCE_PROFILE_NORMAL: (
        "fa5f17f649466434b6503272081dedeb2b407bf8bea2c16d39898271a19d7cdd"
    ),
    REFERENCE_PROFILE_LOW_LATENCY: (
        "fd0eabc623bf4709c0c3163fc2e67a46126ed2fd4e9f9c3456347c30dcb894d2"
    ),
}
ONNX_METADATA_KEY = "g1_23dof_artifact"
ONNX_OPSET_VERSION = 13
ONNX_INPUT_NAME = "obs_dict"
ONNX_OUTPUT_NAME = "action"
ENCODER_ONNX_INPUT_NAME = "teleop_obs"
ENCODER_ONNX_OUTPUT_NAME = "token"
PARITY_ATOL = 1.0e-5
PARITY_RTOL = 1.0e-5
MAX_PHANTOM_OBSERVATION_ABS = 1.0e-8
MAX_RECOVERY_TIME_S = 2.0
MAX_ACTION_SATURATION_FRACTION = 0.10
MAX_MPJPE_M = 0.15

_DECODER_PREFIX = "actor_module.decoders.g1_dyn.module."
_ENCODER_PREFIX = "actor_module.encoders.teleop.module."
_DECODER_TENSOR_RE = re.compile(r"^actor_module\.decoders\.g1_dyn\.module\.(?P<index>\d+)\.(?P<kind>weight|bias)$")
_ENCODER_TENSOR_RE = re.compile(r"^actor_module\.encoders\.teleop\.module\.(?P<index>\d+)\.(?P<kind>weight|bias)$")
_NORMAL_DECODER_LAYER_DIMS = (
    DEPLOYMENT_DECODER_INPUT_DIM,
    2048,
    2048,
    1024,
    1024,
    512,
    512,
    TARGET_DOF,
)
_LOW_LATENCY_DECODER_LAYER_DIMS = (
    DEPLOYMENT_DECODER_INPUT_DIM,
    4096,
    4096,
    2048,
    2048,
    1024,
    1024,
    512,
    512,
    TARGET_DOF,
)
_DECODER_LAYER_DIMS_BY_PROFILE = {
    REFERENCE_PROFILE_NORMAL: _NORMAL_DECODER_LAYER_DIMS,
    REFERENCE_PROFILE_LOW_LATENCY: _LOW_LATENCY_DECODER_LAYER_DIMS,
}
_ENCODER_LAYER_DIMS = (
    TELEOP_ENCODER_INPUT_DIM,
    2048,
    1024,
    512,
    512,
    TOKEN_DIM,
)
_ENCODER_LINEAR_INDICES = tuple(range(0, 2 * (len(_ENCODER_LAYER_DIMS) - 1), 2))
_MLP_ACTIVATION = "SiLU"
_FSQ_FORMULA = "fsq_tanh_round_ste_even_levels_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_ROBOT_ASSET_PATH = _PACKAGE_ROOT / "data/robots/g1/g1_23dof_rev_1_0.urdf"
_ROBOT_CONFIG_PATH = _PACKAGE_ROOT / "envs/manager_env/robots/g1_23dof.py"
_DECODER_CONFIG_PATH = _PACKAGE_ROOT / "config/actor_critic/decoders/g1_dyn_mlp.yaml"
_LOW_LATENCY_DECODER_CONFIG_PATH = (
    _PACKAGE_ROOT
    / "config/actor_critic/decoders/g1_dyn_mlp_low_latency.yaml"
)
_ENCODER_CONFIG_PATH = _PACKAGE_ROOT / "config/actor_critic/encoders/teleop_mlp.yaml"
_POLICY_CONFIG_PATH = _PACKAGE_ROOT / "config/actor_critic/universal_token/teleop_mlp_v1.yaml"
_LOW_LATENCY_POLICY_CONFIG_PATH = (
    _PACKAGE_ROOT
    / "config/actor_critic/universal_token/teleop_mlp_v1_low_latency.yaml"
)
_SIM_VALIDATION_CONFIG_PATH = _PACKAGE_ROOT / "config/sim_validation/g1_23dof_rev_1_0.json"
_SIM_VALIDATION_RUNNER_PATH = _PACKAGE_ROOT / "scripts/run_g1_23dof_sim_validation.py"
_REPOSITORY_ROOT = _PACKAGE_ROOT.parent

TRUE23_ROBOT_ASSET_RELPATHS = (
    "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.urdf",
    "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
    "gear_sonic/data/robots/g1/meshes/head_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_ankle_pitch_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_ankle_roll_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_elbow_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_hip_pitch_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_hip_roll_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_hip_yaw_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_knee_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_shoulder_pitch_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_shoulder_roll_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_shoulder_yaw_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_wrist_roll_rubber_hand.STL",
    "gear_sonic/data/robots/g1/meshes/logo_link.STL",
    "gear_sonic/data/robots/g1/meshes/pelvis.STL",
    "gear_sonic/data/robots/g1/meshes/pelvis_contour_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_ankle_pitch_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_ankle_roll_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_elbow_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_hip_pitch_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_hip_roll_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_hip_yaw_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_knee_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_shoulder_pitch_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_shoulder_roll_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_shoulder_yaw_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_wrist_roll_rubber_hand.STL",
    "gear_sonic/data/robots/g1/meshes/torso_link_23dof_rev_1_0.STL",
)

RUNTIME_SOURCE_RELPATHS = (
    (
        "external_dependencies/"
        "XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/"
        "bindings/py_bindings.cpp"
    ),
    "gear_sonic/__init__.py",
    "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.urdf",
    "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
    "gear_sonic/data/robots/g1/meshes/head_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_ankle_pitch_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_ankle_roll_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_elbow_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_hip_pitch_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_hip_roll_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_hip_yaw_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_knee_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_shoulder_pitch_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_shoulder_roll_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_shoulder_yaw_link.STL",
    "gear_sonic/data/robots/g1/meshes/left_wrist_roll_rubber_hand.STL",
    "gear_sonic/data/robots/g1/meshes/logo_link.STL",
    "gear_sonic/data/robots/g1/meshes/pelvis.STL",
    "gear_sonic/data/robots/g1/meshes/pelvis_contour_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_ankle_pitch_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_ankle_roll_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_elbow_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_hip_pitch_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_hip_roll_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_hip_yaw_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_knee_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_shoulder_pitch_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_shoulder_roll_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_shoulder_yaw_link.STL",
    "gear_sonic/data/robots/g1/meshes/right_wrist_roll_rubber_hand.STL",
    "gear_sonic/data/robots/g1/meshes/torso_link_23dof_rev_1_0.STL",
    "gear_sonic/envs/__init__.py",
    "gear_sonic/envs/env_utils/__init__.py",
    "gear_sonic/envs/env_utils/joint_utils.py",
    "gear_sonic/envs/manager_env/__init__.py",
    "gear_sonic/envs/manager_env/mdp/__init__.py",
    "gear_sonic/envs/manager_env/mdp/actions.py",
    "gear_sonic/envs/manager_env/mdp/actuators.py",
    "gear_sonic/envs/manager_env/mdp/commands.py",
    "gear_sonic/envs/manager_env/mdp/curriculum.py",
    "gear_sonic/envs/manager_env/mdp/events.py",
    "gear_sonic/envs/manager_env/mdp/observations.py",
    "gear_sonic/envs/manager_env/mdp/recorders.py",
    "gear_sonic/envs/manager_env/mdp/rewards.py",
    "gear_sonic/envs/manager_env/mdp/terminations.py",
    "gear_sonic/envs/manager_env/mdp/terrain.py",
    "gear_sonic/envs/manager_env/mdp/utils.py",
    "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py",
    "gear_sonic/envs/manager_env/robots/__init__.py",
    "gear_sonic/envs/manager_env/robots/g1.py",
    "gear_sonic/envs/manager_env/robots/g1_23dof.py",
    "gear_sonic/envs/manager_env/robots/h2.py",
    "gear_sonic/envs/wrapper/__init__.py",
    "gear_sonic/envs/wrapper/manager_env_wrapper.py",
    "gear_sonic/eval_agent_trl.py",
    "gear_sonic/isaac_utils/__init__.py",
    "gear_sonic/isaac_utils/maths.py",
    "gear_sonic/isaac_utils/rotations.py",
    "gear_sonic/scripts/preflight_g1_23dof_training.py",
    "gear_sonic/scripts/run_g1_23dof_sim_validation.py",
    "gear_sonic/train_agent_trl.py",
    "gear_sonic/trl/__init__.py",
    "gear_sonic/trl/callbacks/__init__.py",
    "gear_sonic/trl/callbacks/hv_callback_handler.py",
    "gear_sonic/trl/callbacks/im_eval_callback.py",
    "gear_sonic/trl/callbacks/im_resample_callback.py",
    "gear_sonic/trl/callbacks/model_save_callback.py",
    "gear_sonic/trl/callbacks/read_eval_callback.py",
    "gear_sonic/trl/callbacks/wandb_callback.py",
    "gear_sonic/trl/modules/__init__.py",
    "gear_sonic/trl/modules/actor_critic_modules.py",
    "gear_sonic/trl/modules/base_module.py",
    "gear_sonic/trl/modules/data_utils.py",
    "gear_sonic/trl/modules/universal_token_modules.py",
    "gear_sonic/trl/trainer/__init__.py",
    "gear_sonic/trl/trainer/ppo_trainer.py",
    "gear_sonic/trl/trainer/ppo_trainer_aux_loss.py",
    "gear_sonic/trl/utils/__init__.py",
    "gear_sonic/trl/utils/common.py",
    "gear_sonic/trl/utils/g1_23dof_checkpoint.py",
    "gear_sonic/trl/utils/kornia_transform.py",
    "gear_sonic/trl/utils/math.py",
    "gear_sonic/trl/utils/order_converter.py",
    "gear_sonic/trl/utils/rl.py",
    "gear_sonic/trl/utils/scheduler.py",
    "gear_sonic/trl/utils/torch_transform.py",
    "gear_sonic/utils/__init__.py",
    "gear_sonic/utils/average_meters.py",
    "gear_sonic/utils/batch_normalizer.py",
    "gear_sonic/utils/common.py",
    "gear_sonic/utils/config_utils.py",
    "gear_sonic/utils/g1_23dof_artifact.py",
    "gear_sonic/utils/g1_23dof_checkpoint_io.py",
    "gear_sonic/utils/g1_23dof_contract.py",
    "gear_sonic/utils/inference_helpers.py",
    "gear_sonic/utils/logging.py",
    "gear_sonic/utils/motion_lib/__init__.py",
    "gear_sonic/utils/motion_lib/motion_lib_base.py",
    "gear_sonic/utils/motion_lib/motion_lib_robot.py",
    "gear_sonic/utils/motion_lib/skeleton.py",
    "gear_sonic/utils/motion_lib/torch_humanoid_batch.py",
    "gear_sonic/utils/obs_utils.py",
    "gear_sonic/utils/running_mean_std.py",
    "gear_sonic/utils/teleop/input_readers.py",
    "gear_sonic/utils/teleop/isaac_teleop_client.py",
)

MOTION_DATASET_SOURCE_ARCHIVE = {
    "repository": "bones-studio/seed",
    "revision": "2f59b2077b9da34dd4e43618e705c7cb962c9a66",
    "relpath": "g1.tar.gz",
    "size_bytes": 23_499_973_647,
    "sha256": "52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a",
}
MOTION_DATASET_PROCESSED_ROOT_RELPATH = (
    "data/motion_lib_bones_seed/robot_filtered"
)


def _decoder_layer_dims(reference_profile: str) -> tuple[int, ...]:
    try:
        return _DECODER_LAYER_DIMS_BY_PROFILE[reference_profile]
    except KeyError as exc:
        raise ValueError(
            f"unsupported decoder reference profile: {reference_profile}"
        ) from exc


def decoder_layer_dims_for_profile(
    reference_profile: str,
) -> tuple[int, ...]:
    """Return exact released decoder topology for one temporal profile."""

    return _decoder_layer_dims(reference_profile)


def _decoder_config_path(reference_profile: str) -> Path:
    _decoder_layer_dims(reference_profile)
    return (
        _LOW_LATENCY_DECODER_CONFIG_PATH
        if reference_profile == REFERENCE_PROFILE_LOW_LATENCY
        else _DECODER_CONFIG_PATH
    )


def _policy_config_path(reference_profile: str) -> Path:
    _decoder_layer_dims(reference_profile)
    return (
        _LOW_LATENCY_POLICY_CONFIG_PATH
        if reference_profile == REFERENCE_PROFILE_LOW_LATENCY
        else _POLICY_CONFIG_PATH
    )


def _checkpoint_reference_profile(checkpoint: Mapping[str, Any]) -> str:
    metadata = checkpoint.get("g1_23dof_metadata")
    if metadata is None:
        # Compatibility for isolated normal-topology unit fixtures only.
        return DEFAULT_REFERENCE_PROFILE
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint g1_23dof_metadata must be a mapping")
    profile = metadata.get("reference_profile")
    if not isinstance(profile, str):
        raise ValueError("checkpoint metadata lacks reference_profile")
    reference_profile_contract(profile)
    return profile


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON; reject NaN and implementation objects."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_path(value: Any, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context} must be a non-empty POSIX relative path")
    relpath = PurePosixPath(value)
    if (
        relpath.is_absolute()
        or ".." in relpath.parts
        or relpath.as_posix() != value
    ):
        raise ValueError(f"{context} must be normalized and repository-relative")
    return relpath


def _reject_symlink_components(
    root: Path,
    relpath: PurePosixPath,
    context: str,
) -> Path:
    if root.is_symlink():
        raise ValueError(f"{context} root may not be a symlink")
    candidate = root
    for part in relpath.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"{context} may not traverse symlinks")
    return candidate


def canonical_runtime_source_manifest(
    *,
    repo_root: Path = _REPOSITORY_ROOT,
    relpaths: Sequence[str] = RUNTIME_SOURCE_RELPATHS,
) -> dict[str, Any]:
    """Hash exact simulation/runtime sources in canonical relative-path order."""
    if (
        isinstance(relpaths, (str, bytes))
        or tuple(relpaths) != tuple(sorted(set(relpaths)))
    ):
        raise ValueError(
            "runtime source relpaths must be a sorted unique sequence"
        )
    unresolved_root = Path(repo_root)
    root = unresolved_root.resolve()
    records: list[dict[str, Any]] = []
    for index, value in enumerate(relpaths):
        relpath = _repo_relative_path(
            value,
            f"runtime source relpaths[{index}]",
        )
        unresolved = _reject_symlink_components(
            unresolved_root,
            relpath,
            f"runtime source {value}",
        )
        resolved = unresolved.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"runtime source {value!r} escapes repository root"
            ) from exc
        if not resolved.is_file():
            raise ValueError(f"runtime source is missing: {value}")
        records.append(
            {
                "relpath": relpath.as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return {
        "schema_version": FILE_MANIFEST_SCHEMA_VERSION,
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "manifest_sha256": sha256_bytes(canonical_json_bytes(records)),
        "files": records,
    }


def _local_python_import_relpaths(
    source_relpath: str,
    source: str,
    *,
    repo_root: Path,
) -> set[str]:
    """Resolve statically imported in-repository ``gear_sonic`` modules."""

    tree = ast.parse(source, filename=source_relpath)
    source_parts = PurePosixPath(source_relpath).with_suffix("").parts
    package_parts = (
        source_parts[:-1]
        if source_parts[-1] != "__init__"
        else source_parts[:-1]
    )
    discovered: set[str] = set()

    def add_module(module: str) -> None:
        if not module.startswith("gear_sonic"):
            return
        module_path = module.replace(".", "/")
        for candidate in (
            f"{module_path}.py",
            f"{module_path}/__init__.py",
        ):
            if (repo_root / candidate).is_file():
                discovered.add(candidate)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - (node.level - 1)
                base_parts = package_parts[: max(0, keep)]
                if node.module:
                    base_parts += tuple(node.module.split("."))
                module = ".".join(base_parts)
            else:
                module = node.module or ""
            add_module(module)
            for alias in node.names:
                if alias.name != "*":
                    add_module(f"{module}.{alias.name}")
    return discovered


def validate_runtime_source_completeness(
    *,
    repo_root: Path = _REPOSITORY_ROOT,
    relpaths: Sequence[str] = RUNTIME_SOURCE_RELPATHS,
) -> None:
    """Reject omitted local imports and omitted true23 robot asset bytes."""

    approved = set(relpaths)
    missing: set[str] = set()
    for relpath in relpaths:
        if not relpath.endswith(".py"):
            continue
        path = Path(repo_root) / relpath
        missing.update(
            _local_python_import_relpaths(
                relpath,
                path.read_text(encoding="utf-8"),
                repo_root=Path(repo_root),
            )
            - approved
        )
    if missing:
        raise ValueError(
            "runtime source allowlist omits local imports: "
            f"{sorted(missing)}"
        )
    if Path(repo_root).resolve() == _REPOSITORY_ROOT.resolve():
        omitted_assets = set(TRUE23_ROBOT_ASSET_RELPATHS) - approved
        if omitted_assets:
            raise ValueError(
                "runtime source allowlist omits true23 robot assets: "
                f"{sorted(omitted_assets)}"
            )


def validate_file_manifest(
    value: Any,
    *,
    expected_relpaths: Sequence[str],
    context: str,
) -> dict[str, Any]:
    """Validate an embedded canonical file manifest without reading disk."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "file_count",
            "total_bytes",
            "manifest_sha256",
            "files",
        },
        context,
    )
    if value["schema_version"] != FILE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"{context}.schema_version is unsupported")
    files = value["files"]
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise ValueError(f"{context}.files must be an array")
    records: list[dict[str, Any]] = []
    previous = ""
    total_bytes = 0
    for index, record in enumerate(files):
        record_context = f"{context}.files[{index}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{record_context} must be an object")
        _require_exact_keys(
            record,
            {"relpath", "size_bytes", "sha256"},
            record_context,
        )
        relpath = _repo_relative_path(record["relpath"], f"{record_context}.relpath")
        relpath_text = relpath.as_posix()
        if previous and relpath_text <= previous:
            raise ValueError(f"{context}.files must use sorted unique relpaths")
        previous = relpath_text
        size_bytes = _require_int(
            record["size_bytes"],
            f"{record_context}.size_bytes",
        )
        sha256 = _require_sha256(record["sha256"], f"{record_context}.sha256")
        total_bytes += size_bytes
        records.append(
            {
                "relpath": relpath_text,
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )
    if tuple(record["relpath"] for record in records) != tuple(expected_relpaths):
        raise ValueError(f"{context}.files differ from required relpaths")
    if value["file_count"] != len(records):
        raise ValueError(f"{context}.file_count differs from files")
    if value["total_bytes"] != total_bytes:
        raise ValueError(f"{context}.total_bytes differs from files")
    expected_sha256 = sha256_bytes(canonical_json_bytes(records))
    if value["manifest_sha256"] != expected_sha256:
        raise ValueError(f"{context}.manifest_sha256 differs from files")
    return {
        "schema_version": FILE_MANIFEST_SCHEMA_VERSION,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "manifest_sha256": expected_sha256,
        "files": records,
    }


def canonical_true23_robot_asset_manifest(
    *,
    repo_root: Path = _REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Hash URDF, MJCF and every mesh referenced by either model."""

    root = Path(repo_root)
    urdf_relpath, mjcf_relpath = TRUE23_ROBOT_ASSET_RELPATHS[:2]
    urdf_root = ET.parse(root / urdf_relpath).getroot()
    mjcf_root = ET.parse(root / mjcf_relpath).getroot()
    urdf_dir = PurePosixPath(urdf_relpath).parent
    mjcf_dir = PurePosixPath(mjcf_relpath).parent
    meshdir = PurePosixPath(
        mjcf_root.find("compiler").attrib.get("meshdir", "")
    )
    referenced = {
        (urdf_dir / mesh.attrib["filename"]).as_posix()
        for mesh in urdf_root.findall(".//mesh")
    }
    referenced.update(
        (mjcf_dir / meshdir / mesh.attrib["file"]).as_posix()
        for mesh in mjcf_root.findall("./asset/mesh")
    )
    expected_meshes = set(TRUE23_ROBOT_ASSET_RELPATHS[2:])
    if referenced != expected_meshes:
        raise ValueError(
            "true23 URDF/MJCF mesh references differ from asset allowlist; "
            f"missing={sorted(referenced - expected_meshes)}, "
            f"unreferenced={sorted(expected_meshes - referenced)}"
        )
    return canonical_runtime_source_manifest(
        repo_root=root,
        relpaths=TRUE23_ROBOT_ASSET_RELPATHS,
    )


def canonical_processed_motion_manifest(
    *,
    repo_root: Path = _REPOSITORY_ROOT,
    root_relpath: str = MOTION_DATASET_PROCESSED_ROOT_RELPATH,
) -> dict[str, Any]:
    """Stream/hash every processed motion file in deterministic path order."""
    relative_root = _repo_relative_path(
        root_relpath,
        "processed motion root_relpath",
    )
    unresolved_repo = Path(repo_root)
    repository_root = unresolved_repo.resolve()
    unresolved_motion_root = _reject_symlink_components(
        unresolved_repo,
        relative_root,
        "processed motion dataset",
    )
    motion_root = unresolved_motion_root.resolve()
    try:
        motion_root.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("processed motion dataset escapes repository root") from exc
    if not motion_root.is_dir():
        raise ValueError(
            f"processed motion dataset is missing: {relative_root.as_posix()}"
        )

    relative_files: list[PurePosixPath] = []
    for directory, dirnames, filenames in os.walk(
        unresolved_motion_root,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        dirnames.sort()
        filenames.sort()
        for dirname in dirnames:
            if (directory_path / dirname).is_symlink():
                raise ValueError(
                    "processed motion dataset may not contain symlink directories"
                )
        for filename in filenames:
            unresolved = directory_path / filename
            if unresolved.is_symlink():
                raise ValueError(
                    "processed motion dataset may not contain symlink files"
                )
            resolved = unresolved.resolve()
            try:
                relative = resolved.relative_to(motion_root)
                resolved.relative_to(repository_root)
            except ValueError as exc:
                raise ValueError(
                    "processed motion dataset file escapes approved root"
                ) from exc
            if not resolved.is_file():
                raise ValueError(
                    "processed motion dataset contains a non-regular file"
                )
            relative_files.append(PurePosixPath(relative.as_posix()))
    relative_files.sort(key=PurePosixPath.as_posix)
    if not relative_files:
        raise ValueError("processed motion dataset contains no files")
    if len(set(relative_files)) != len(relative_files):
        raise ValueError("processed motion dataset has duplicate relative paths")

    digest = hashlib.sha256()
    digest.update(b"[")
    total_bytes = 0
    for index, relpath in enumerate(relative_files):
        path = motion_root.joinpath(*relpath.parts)
        size_bytes = path.stat().st_size
        total_bytes += size_bytes
        record = {
            "relpath": relpath.as_posix(),
            "size_bytes": size_bytes,
            "sha256": sha256_file(path),
        }
        if index:
            digest.update(b",")
        digest.update(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
    digest.update(b"]\n")
    return {
        "root_relpath": relative_root.as_posix(),
        "file_count": len(relative_files),
        "total_bytes": total_bytes,
        "manifest_sha256": digest.hexdigest(),
    }


def _validate_motion_dataset_contract(
    value: Any,
    *,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "source_archive",
            "processed",
            "promotion_enabled",
        },
        context,
    )
    if value["schema_version"] != MOTION_DATASET_SCHEMA_VERSION:
        raise ValueError(f"{context}.schema_version is unsupported")
    source = value["source_archive"]
    if not isinstance(source, Mapping):
        raise ValueError(f"{context}.source_archive must be an object")
    _require_exact_keys(
        source,
        {"repository", "revision", "relpath", "size_bytes", "sha256"},
        f"{context}.source_archive",
    )
    if dict(source) != MOTION_DATASET_SOURCE_ARCHIVE:
        raise ValueError(
            f"{context}.source_archive is not the approved BONES-SEED archive"
        )
    processed = value["processed"]
    if not isinstance(processed, Mapping):
        raise ValueError(f"{context}.processed must be an object")
    _require_exact_keys(
        processed,
        {
            "root_relpath",
            "file_count",
            "total_bytes",
            "manifest_sha256",
        },
        f"{context}.processed",
    )
    if (
        processed["root_relpath"]
        != MOTION_DATASET_PROCESSED_ROOT_RELPATH
    ):
        raise ValueError(f"{context}.processed root is unsupported")
    promotion_enabled = value["promotion_enabled"]
    if not isinstance(promotion_enabled, bool):
        raise ValueError(f"{context}.promotion_enabled must be boolean")
    pinned_values = (
        processed["file_count"],
        processed["total_bytes"],
        processed["manifest_sha256"],
    )
    if promotion_enabled:
        _require_int(
            processed["file_count"],
            f"{context}.processed.file_count",
            minimum=1,
        )
        _require_int(
            processed["total_bytes"],
            f"{context}.processed.total_bytes",
            minimum=1,
        )
        _require_sha256(
            processed["manifest_sha256"],
            f"{context}.processed.manifest_sha256",
        )
    elif pinned_values != (None, None, None):
        raise ValueError(
            f"{context} unapproved processed manifest must remain null"
        )
    return value


def validate_motion_dataset_evidence(
    value: Any,
    *,
    context: str = "motion_dataset",
) -> dict[str, Any]:
    """Validate exact non-null dataset lineage stored in trained artifacts."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(
        value,
        {"schema_version", "source_archive", "processed"},
        context,
    )
    contract = {
        **dict(value),
        "promotion_enabled": True,
    }
    _validate_motion_dataset_contract(contract, context=context)
    return {
        "schema_version": MOTION_DATASET_SCHEMA_VERSION,
        "source_archive": dict(value["source_archive"]),
        "processed": dict(value["processed"]),
    }


def approved_motion_dataset_provenance(
    runtime_config: Mapping[str, Any],
    *,
    validation_config: Mapping[str, Any] | None = None,
    repo_root: Path = _REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Recompute and verify approved BONES-SEED processed content."""
    validation_config = (
        _sim_validation_config() if validation_config is None else validation_config
    )
    contract = _validate_motion_dataset_contract(
        validation_config.get("motion_dataset"),
        context="simulation motion_dataset contract",
    )
    if contract["promotion_enabled"] is not True:
        raise ValueError(
            "BONES-SEED processed motion manifest is not approved; "
            "training/simulation promotion remains blocked"
        )
    source_relpath = _repo_relative_path(
        contract["source_archive"]["relpath"],
        "simulation motion_dataset source archive relpath",
    )
    unresolved_source = _reject_symlink_components(
        Path(repo_root),
        source_relpath,
        "BONES-SEED source archive",
    )
    source_path = unresolved_source.resolve()
    repository_root = Path(repo_root).resolve()
    try:
        source_path.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("BONES-SEED source archive escapes repository root") from exc
    if not source_path.is_file():
        raise ValueError(
            "approved BONES-SEED source archive is missing: "
            f"{source_relpath.as_posix()}"
        )
    if source_path.stat().st_size != contract["source_archive"]["size_bytes"]:
        raise ValueError("BONES-SEED source archive size differs from approved contract")
    if sha256_file(source_path) != contract["source_archive"]["sha256"]:
        raise ValueError("BONES-SEED source archive hash differs from approved contract")
    motion_paths = (
        (
            "manager_env",
            "commands",
            "motion",
            "motion_lib_cfg",
            "motion_file",
        ),
        ("commands", "motion", "motion_lib_cfg", "motion_file"),
        (
            "manager_env",
            "config",
            "commands",
            "motion",
            "motion_lib_cfg",
            "motion_file",
        ),
        ("config", "commands", "motion", "motion_lib_cfg", "motion_file"),
    )
    configured_motion_roots = [
        value
        for path in motion_paths
        if (value := _nested_value(runtime_config, path)) is not None
    ]
    if not configured_motion_roots:
        raise ValueError("runtime config lacks motion_file")
    if (
        any(not isinstance(value, str) for value in configured_motion_roots)
        or len(set(configured_motion_roots)) != 1
    ):
        raise ValueError("runtime config has divergent motion_file copies")
    if configured_motion_roots[0] != MOTION_DATASET_PROCESSED_ROOT_RELPATH:
        raise ValueError(
            "runtime motion_file differs from approved processed dataset root"
        )
    recomputed = canonical_processed_motion_manifest(
        repo_root=repo_root,
        root_relpath=contract["processed"]["root_relpath"],
    )
    if recomputed != dict(contract["processed"]):
        raise ValueError(
            "processed motion dataset manifest differs from approved contract"
        )
    return validate_motion_dataset_evidence(
        {
            "schema_version": MOTION_DATASET_SCHEMA_VERSION,
            "source_archive": dict(contract["source_archive"]),
            "processed": recomputed,
        }
    )


def approved_runtime_source_manifest(
    *,
    validation_config: Mapping[str, Any] | None = None,
    repo_root: Path = _REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Recompute all approved simulation source files and fail on drift."""
    validation_config = (
        _sim_validation_config() if validation_config is None else validation_config
    )
    contract = validation_config.get("runtime_sources")
    if not isinstance(contract, Mapping):
        raise ValueError("simulation config lacks runtime_sources")
    _require_exact_keys(
        contract,
        {
            "schema_version",
            "relpaths",
            "file_count",
            "manifest_sha256",
        },
        "simulation runtime_sources",
    )
    if contract["schema_version"] != FILE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("simulation runtime_sources schema is unsupported")
    relpaths = contract["relpaths"]
    if (
        not isinstance(relpaths, Sequence)
        or isinstance(relpaths, (str, bytes))
        or tuple(relpaths) != RUNTIME_SOURCE_RELPATHS
    ):
        raise ValueError(
            "simulation runtime_sources relpaths differ from transitive allowlist"
        )
    validate_runtime_source_completeness(
        repo_root=repo_root,
        relpaths=tuple(relpaths),
    )
    recomputed = canonical_runtime_source_manifest(
        repo_root=repo_root,
        relpaths=tuple(relpaths),
    )
    if (
        contract["file_count"] != recomputed["file_count"]
        or _require_sha256(
            contract["manifest_sha256"],
            "simulation runtime_sources.manifest_sha256",
        )
        != recomputed["manifest_sha256"]
    ):
        raise ValueError(
            "simulation runtime source manifest differs from approved contract"
        )
    return recomputed


def simulation_material_provenance(
    runtime_config: Mapping[str, Any],
    *,
    checkpoint_motion_dataset: Mapping[str, Any],
    validation_config: Mapping[str, Any] | None = None,
    repo_root: Path = _REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Bind runtime sources and exact checkpoint dataset to local bytes."""
    validation_config = (
        _sim_validation_config() if validation_config is None else validation_config
    )
    expected_dataset = validate_motion_dataset_evidence(
        checkpoint_motion_dataset,
        context="checkpoint motion_dataset",
    )
    current_dataset = approved_motion_dataset_provenance(
        runtime_config,
        validation_config=validation_config,
        repo_root=repo_root,
    )
    if current_dataset != expected_dataset:
        raise ValueError(
            "simulation motion_dataset differs from trained checkpoint"
        )
    return {
        "schema_version": MATERIAL_PROVENANCE_SCHEMA_VERSION,
        "runtime_source": approved_runtime_source_manifest(
            validation_config=validation_config,
            repo_root=repo_root,
        ),
        "motion_dataset": current_dataset,
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> Mapping[str, Any]:
    """Load JSON while rejecting duplicate keys and non-finite constants."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON evidence file {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("JSON evidence root must be an object")
    return value


def _sim_validation_config() -> Mapping[str, Any]:
    config = load_strict_json(_SIM_VALIDATION_CONFIG_PATH)
    runtime_sources = config.get("runtime_sources")
    motion_dataset = config.get("motion_dataset")
    expected = {
        "schema_version": 3,
        "robot_model": ROBOT_MODEL,
        "control_hz": 50,
        "producer": {
            "kind": "gear_sonic_true23_isaaclab_disturbance_validation",
            "version": 1,
            "runner_sha256": (
                sha256_file(_SIM_VALIDATION_RUNNER_PATH)
                if _SIM_VALIDATION_RUNNER_PATH.is_file()
                else None
            ),
            "promotion_enabled": False,
        },
        "minimum_coverage": {
            "seeds_per_scenario": 3,
            "episodes_per_scenario": 64,
            "episodes_per_seed": 22,
            "seconds_per_episode": 5.0,
            "steps_per_episode": 250,
        },
        "deterministic_seeds": [1729, 2718, 3141],
        "disturbance_schedule": {
            "apply_step": 50,
            "recovery_baseline_steps": 10,
            "recovery_stable_steps": 5,
            "recovery_margin": 0.1,
        },
        "runtime_contract": {
            "material_config_sha256_by_reference_profile": {
                "true23_step5_0p1s": (
                    "8c151c78360a26a1e4e62dea9cacc490"
                    "2c4c7909f7caab632c4222f099497a3d"
                ),
                "released_low_latency_step1_0p02s": (
                    "52bac7e50170d9f2e99d118468c031779"
                    "888350eecf51d1bad669cab0cfc72d0"
                ),
            },
        },
        "runtime_sources": runtime_sources,
        "motion_dataset": motion_dataset,
        "disturbance_envelope": {
            "linear_velocity_delta_mps": {
                "x": [-0.5, 0.5],
                "y": [-0.5, 0.5],
                "z": [-0.2, 0.2],
            },
            "angular_velocity_delta_radps": {
                "roll": [-0.52, 0.52],
                "pitch": [-0.52, 0.52],
                "yaw": [-0.78, 0.78],
            },
        },
        "scenarios": {
            "nominal": {"disturbance_scale": 0.0},
            "disturbance_50": {"disturbance_scale": 0.5},
            "disturbance_100": {"disturbance_scale": 1.0},
        },
    }
    if config != expected:
        raise ValueError("checked-in true23 simulation validation config changed unexpectedly")
    _validate_motion_dataset_contract(
        motion_dataset,
        context="checked-in simulation motion_dataset",
    )
    approved_runtime_source_manifest(
        validation_config=config,
        repo_root=_REPOSITORY_ROOT,
    )
    return config


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


def _require_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be integer >= {minimum}")
    return value


def _require_float(
    value: Any,
    context: str,
    *,
    minimum: float | None = 0.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{context} must be {qualifier}")
    return result


def _nested_value(value: Any, path: Sequence[str]) -> Any:
    for key in path:
        if isinstance(value, Mapping):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
        if value is None:
            return None
    return value


def is_true23_training_config(config: Any) -> bool:
    """Recognize manager-only and full-Hydra wrapper config shapes."""
    candidate_paths = (
        ("robot", "type"),
        ("robot", "embodiment", "model"),
        ("manager_env", "config", "robot", "type"),
        ("manager_env", "config", "robot", "embodiment", "model"),
        ("config", "robot", "type"),
        ("config", "robot", "embodiment", "model"),
    )
    return any(_nested_value(config, path) == ROBOT_MODEL for path in candidate_paths)


def true23_reference_profile_from_config(config: Any) -> str:
    """Return one exact configured profile; reject absent or divergent copies."""

    embodiment_paths = (
        ("robot", "embodiment"),
        ("manager_env", "config", "robot", "embodiment"),
        ("config", "robot", "embodiment"),
    )
    profiles: list[str] = []
    for path in embodiment_paths:
        embodiment = _nested_value(config, path)
        if embodiment is None:
            continue
        profile = _nested_value(embodiment, ("reference_profile",))
        contract = _nested_value(embodiment, ("reference_contract",))
        if profile is None and contract is None:
            continue
        if not isinstance(profile, str):
            raise ValueError("true23 embodiment.reference_profile must be a string")
        expected = reference_profile_contract(profile)
        if contract != expected:
            raise ValueError(
                "true23 embodiment.reference_contract does not match reference_profile"
            )
        profiles.append(profile)
    if not profiles:
        raise ValueError("true23 training config lacks immutable reference profile")
    if len(set(profiles)) != 1:
        raise ValueError(f"true23 training config has divergent reference profiles: {profiles}")
    return profiles[0]


def true23_warm_start_provenance_from_config(
    config: Any,
) -> dict[str, Any]:
    """Validate and normalize exact released-checkpoint lineage."""

    warm_start = _nested_value(config, ("true23_warm_start",))
    expected_keys = {
        "source_family",
        "source_hf_revision",
        "source_checkpoint_sha256",
        "initialization_only",
    }
    if not isinstance(warm_start, Mapping) or set(warm_start) != expected_keys:
        raise ValueError("true23 config lacks exact warm-start provenance")
    profile = true23_reference_profile_from_config(config)
    source_sha256 = warm_start["source_checkpoint_sha256"]
    release = APPROVED_WARM_START_RELEASES.get(source_sha256)
    if (
        release is None
        or release["source_family"] != warm_start["source_family"]
        or release["source_revision"] != warm_start["source_hf_revision"]
        or release["reference_profile"] != profile
        or warm_start["initialization_only"] is not True
    ):
        raise ValueError("true23 warm-start provenance is not an approved release")
    return {
        "source_family": release["source_family"],
        "source_revision": release["source_revision"],
        "source_checkpoint_sha256": source_sha256,
        "reference_profile": profile,
    }


def runtime_material_config_descriptor(config: Mapping[str, Any]) -> dict[str, Any]:
    """Select all simulation/policy fields that can materially change evidence."""
    if not isinstance(config, Mapping):
        raise ValueError("resolved runtime config must be an object")
    manager_env = config.get("manager_env")
    algo = config.get("algo")
    if not isinstance(manager_env, Mapping) or not isinstance(algo, Mapping):
        raise ValueError("resolved runtime config lacks manager_env/algo")
    manager_config = manager_env.get("config")
    algo_config = algo.get("config")
    if not isinstance(manager_config, Mapping) or not isinstance(algo_config, Mapping):
        raise ValueError("resolved runtime config lacks manager/algo config objects")
    required_manager_sections = (
        "commands",
        "actions",
        "observations",
        "rewards",
        "terminations",
        "events",
        "curriculum",
        "recorders",
    )
    missing_sections = [key for key in required_manager_sections if key not in manager_env]
    if missing_sections:
        raise ValueError(f"resolved runtime manager config lacks {missing_sections}")
    if "actor" not in algo_config:
        raise ValueError("resolved runtime config lacks actor policy config")
    trainer = config.get("trainer")
    if not isinstance(trainer, Mapping):
        raise ValueError("resolved runtime config lacks trainer config")
    schedule_dict = trainer.get("schedule_dict") or {}
    if not isinstance(schedule_dict, Mapping) or dict(schedule_dict) != {}:
        raise ValueError("true23 evidence forbids scheduled runtime mutations")
    normalized_manager_config = {
        key: value
        for key, value in manager_config.items()
        if key not in {"save_rendering_dir", "experiment_dir", "obs"}
    }
    robot = normalized_manager_config.get("robot")
    if not isinstance(robot, Mapping):
        raise ValueError("resolved runtime config lacks manager robot config")
    normalized_manager_config["robot"] = {
        key: value for key, value in robot.items() if key != "algo_obs_dim_dict"
    }
    return {
        "use_manager_env": config.get("use_manager_env"),
        "headless": config.get("headless"),
        "num_envs": config.get("num_envs"),
        "sim_type": config.get("sim_type"),
        "force_flat_terrain": config.get("force_flat_terrain"),
        "multi_gpu": config.get("multi_gpu"),
        "num_gpus": config.get("num_gpus"),
        "seed": config.get("seed"),
        "global_rank": config.get("global_rank"),
        "actor_prop_history_length": config.get("actor_prop_history_length"),
        "actor_actions_history_length": config.get("actor_actions_history_length"),
        "manager_env": {
            "_target_": manager_env.get("_target_"),
            "config": normalized_manager_config,
            **{key: manager_env[key] for key in required_manager_sections},
        },
        "actor": algo_config["actor"],
        "module_dim": algo_config.get("module_dim", {}),
        "distill_only": algo_config.get("distill_only", False),
        "trainer_schedule_dict": {},
    }


def resolved_training_config_snapshot(config: Any) -> dict[str, Any]:
    """Convert Hydra/OmegaConf input to one fully resolved JSON-safe snapshot."""

    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_config(config):
        resolved = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, Mapping):
        resolved = dict(config)
    else:
        raise ValueError("training config must be a mapping")
    if not isinstance(resolved, Mapping):
        raise ValueError("resolved training config must be an object")
    try:
        normalized = json.loads(canonical_json_bytes(resolved))
    except (TypeError, ValueError) as exc:
        raise ValueError("resolved training config is not canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise ValueError("resolved training config must be an object")
    return normalized


def training_material_config_descriptor(
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Select every policy, environment and optimizer field affecting training."""

    runtime = runtime_material_config_descriptor(resolved_config)
    algo = resolved_config.get("algo")
    trainer = resolved_config.get("trainer")
    callbacks = resolved_config.get("callbacks")
    warm_start = resolved_config.get("true23_warm_start")
    if not isinstance(algo, Mapping):
        raise ValueError("resolved training config lacks algo")
    if not isinstance(trainer, Mapping):
        raise ValueError("resolved training config lacks trainer")
    if not isinstance(callbacks, Mapping):
        raise ValueError("resolved training config lacks callbacks")
    if not isinstance(warm_start, Mapping):
        raise ValueError("resolved training config lacks true23_warm_start")
    algo_config = algo.get("config")
    algo_trl = algo.get("trl")
    if not isinstance(algo_config, Mapping) or not isinstance(algo_trl, Mapping):
        raise ValueError("resolved training config lacks algo config/TRL settings")
    normalized_callbacks: dict[str, Any] = {}
    for name, callback in callbacks.items():
        if not isinstance(callback, Mapping):
            raise ValueError(f"resolved training callback {name!r} must be an object")
        normalized_callbacks[name] = {
            key: value
            for key, value in callback.items()
            if key not in {"save_dir", "eval_dir"}
        }
    return {
        "codebase_version": resolved_config.get("codebase_version"),
        "runtime": runtime,
        "algo": {
            "config": dict(algo_config),
            "trl": {
                key: value
                for key, value in algo_trl.items()
                if key not in {"output_dir", "report_to"}
            },
            "_recursive_": algo.get("_recursive_"),
        },
        "trainer": dict(trainer),
        "callbacks": normalized_callbacks,
        "true23_warm_start": dict(warm_start),
        "critic_prop_history_length": resolved_config.get(
            "critic_prop_history_length"
        ),
        "critic_actions_history_length": resolved_config.get(
            "critic_actions_history_length"
        ),
    }


def validate_training_dynamic_target_completeness(
    resolved_config: Mapping[str, Any],
    *,
    repo_root: Path = _REPOSITORY_ROOT,
    relpaths: Sequence[str] = RUNTIME_SOURCE_RELPATHS,
) -> None:
    """Reject omitted local modules selected through Hydra ``_target_``."""

    target_modules: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            target = value.get("_target_")
            if isinstance(target, str) and target.startswith("gear_sonic."):
                target_modules.add(target.rsplit(".", 1)[0])
            for child in value.values():
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                visit(child)

    visit(resolved_config)
    approved = set(relpaths)
    missing: set[str] = set()
    for module in target_modules:
        module_path = module.replace(".", "/")
        candidates = (
            f"{module_path}.py",
            f"{module_path}/__init__.py",
        )
        existing = [
            candidate
            for candidate in candidates
            if (Path(repo_root) / candidate).is_file()
        ]
        if not existing:
            raise ValueError(f"Hydra target module is missing: {module}")
        missing.update(set(existing) - approved)
    if missing:
        raise ValueError(
            "runtime source allowlist omits Hydra target modules: "
            f"{sorted(missing)}"
        )


def build_training_material_evidence(
    config: Any,
    *,
    repo_root: Path = _REPOSITORY_ROOT,
    require_approved_material: bool = True,
) -> dict[str, Any]:
    """Bind exact resolved config and all source/model bytes before training."""

    resolved = resolved_training_config_snapshot(config)
    material = training_material_config_descriptor(resolved)
    validate_runtime_source_completeness(repo_root=repo_root)
    validate_training_dynamic_target_completeness(
        resolved,
        repo_root=repo_root,
    )
    evidence = {
        "schema_version": TRAINING_MATERIAL_SCHEMA_VERSION,
        "resolved_config": resolved,
        "resolved_config_sha256": sha256_bytes(canonical_json_bytes(resolved)),
        "material_config": material,
        "material_config_sha256": sha256_bytes(canonical_json_bytes(material)),
        "runtime_source": canonical_runtime_source_manifest(repo_root=repo_root),
        "robot_assets": canonical_true23_robot_asset_manifest(repo_root=repo_root),
    }
    return validate_training_material_evidence(
        evidence,
        require_approved_material=require_approved_material,
    )


def validate_training_material_evidence(
    value: Any,
    *,
    context: str = "training_material",
    require_approved_material: bool = True,
) -> dict[str, Any]:
    """Validate embedded config/source/asset evidence without trusting digests."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "resolved_config",
            "resolved_config_sha256",
            "material_config",
            "material_config_sha256",
            "runtime_source",
            "robot_assets",
        },
        context,
    )
    if value["schema_version"] != TRAINING_MATERIAL_SCHEMA_VERSION:
        raise ValueError(f"{context}.schema_version is unsupported")
    resolved = value["resolved_config"]
    material = value["material_config"]
    if not isinstance(resolved, Mapping):
        raise ValueError(f"{context}.resolved_config must be an object")
    if not isinstance(material, Mapping):
        raise ValueError(f"{context}.material_config must be an object")
    expected_resolved_sha = sha256_bytes(canonical_json_bytes(resolved))
    if (
        _require_sha256(
            value["resolved_config_sha256"],
            f"{context}.resolved_config_sha256",
        )
        != expected_resolved_sha
    ):
        raise ValueError(f"{context}.resolved_config_sha256 differs from snapshot")
    expected_material = training_material_config_descriptor(resolved)
    if dict(material) != expected_material:
        raise ValueError(f"{context}.material_config differs from resolved config")
    expected_material_sha = sha256_bytes(canonical_json_bytes(expected_material))
    if (
        _require_sha256(
            value["material_config_sha256"],
            f"{context}.material_config_sha256",
        )
        != expected_material_sha
    ):
        raise ValueError(f"{context}.material_config_sha256 differs from config")
    profile = true23_reference_profile_from_config(resolved)
    approved_material_sha = (
        APPROVED_TRAINING_MATERIAL_CONFIG_SHA256_BY_REFERENCE_PROFILE.get(
            profile
        )
    )
    if (
        require_approved_material
        and expected_material_sha != approved_material_sha
    ):
        raise ValueError(
            f"{context}.material_config differs from approved true23 "
            f"{profile} training contract"
        )
    runtime_source = validate_file_manifest(
        value["runtime_source"],
        expected_relpaths=RUNTIME_SOURCE_RELPATHS,
        context=f"{context}.runtime_source",
    )
    robot_assets = validate_file_manifest(
        value["robot_assets"],
        expected_relpaths=TRUE23_ROBOT_ASSET_RELPATHS,
        context=f"{context}.robot_assets",
    )
    return {
        "schema_version": TRAINING_MATERIAL_SCHEMA_VERSION,
        "resolved_config": dict(resolved),
        "resolved_config_sha256": expected_resolved_sha,
        "material_config": expected_material,
        "material_config_sha256": expected_material_sha,
        "runtime_source": runtime_source,
        "robot_assets": robot_assets,
    }


def validate_runtime_config_snapshot(
    config: Mapping[str, Any],
    *,
    validation_config: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Bind the full resolved Hydra config and enforce its approved material hash."""
    validation_config = (
        _sim_validation_config() if validation_config is None else validation_config
    )
    resolved_sha256 = sha256_bytes(canonical_json_bytes(config))
    material = runtime_material_config_descriptor(config)
    material_sha256 = sha256_bytes(canonical_json_bytes(material))
    profile = true23_reference_profile_from_config(config)
    expected_by_profile = validation_config["runtime_contract"][
        "material_config_sha256_by_reference_profile"
    ]
    if set(expected_by_profile) != set(REFERENCE_PROFILES):
        raise ValueError("simulation runtime contract lacks an approved reference profile")
    expected = expected_by_profile[profile]
    if material_sha256 != expected:
        raise ValueError(
            "resolved Hydra simulation/termination/observation/policy config "
            "does not match approved true23 runtime contract"
        )
    return {
        "resolved_config_sha256": resolved_sha256,
        "material_config_sha256": material_sha256,
    }


def _deterministic_disturbance_vector(
    *,
    seed: int,
    episode_index: int,
    scale: float,
    config: Mapping[str, Any],
) -> list[float]:
    envelope = config["disturbance_envelope"]
    bounds = (
        envelope["linear_velocity_delta_mps"]["x"],
        envelope["linear_velocity_delta_mps"]["y"],
        envelope["linear_velocity_delta_mps"]["z"],
        envelope["angular_velocity_delta_radps"]["roll"],
        envelope["angular_velocity_delta_radps"]["pitch"],
        envelope["angular_velocity_delta_radps"]["yaw"],
    )
    result: list[float] = []
    for axis, (lower, upper) in enumerate(bounds):
        digest = hashlib.sha256(
            f"{ROBOT_MODEL}:{seed}:{episode_index}:{axis}".encode("ascii")
        ).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        value = (float(lower) + unit * (float(upper) - float(lower))) * scale
        result.append(round(value, 12))
    return result


def _trace_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    episodes: int,
    disturbance_scale: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    schedule = config["disturbance_schedule"]
    apply_step = int(schedule["apply_step"])
    baseline_steps = int(schedule["recovery_baseline_steps"])
    stable_steps = int(schedule["recovery_stable_steps"])
    recovery_margin = float(schedule["recovery_margin"])
    control_hz = int(config["control_hz"])
    termination_count = 0
    nonfinite_count = 0
    soft_limit_violation_count = 0
    phantom_max = 0.0
    saturated_total = 0
    action_total = 0
    mpjpe_total = 0.0
    mpjpe_count = 0
    recovery_by_episode = [0.0] * episodes
    for record in records:
        termination_count += sum(
            bool(terminated) or bool(timed_out)
            for terminated, timed_out in zip(
                record["terminated"],
                record["timed_out"],
                strict=True,
            )
        )
        nonfinite_count += sum(bool(value) for value in record["nonfinite"])
        soft_limit_violation_count += sum(
            bool(value) for value in record["soft_limit_violation"]
        )
        phantom_max = max(
            phantom_max,
            max(float(value) for value in record["phantom_observation_max_abs"]),
        )
        saturated_total += sum(int(value) for value in record["action_saturated_count"])
        action_total += sum(int(value) for value in record["action_count"])
        mpjpe_total += sum(float(value) for value in record["mpjpe_m"])
        mpjpe_count += episodes
    if disturbance_scale > 0.0:
        for episode in range(episodes):
            baseline_values = [
                float(records[step]["recovery_metric"][episode])
                for step in range(apply_step - baseline_steps, apply_step)
            ]
            threshold = sum(baseline_values) / len(baseline_values) + recovery_margin
            consecutive = 0
            recovered_at: int | None = None
            for step in range(apply_step, len(records)):
                if float(records[step]["recovery_metric"][episode]) <= threshold:
                    consecutive += 1
                    if consecutive >= stable_steps:
                        recovered_at = step - stable_steps + 1
                        break
                else:
                    consecutive = 0
            recovery_by_episode[episode] = (
                len(records) / control_hz
                if recovered_at is None
                else max(0.0, (recovered_at - apply_step) / control_hz)
            )
    return {
        "termination_count": termination_count,
        "nonfinite_count": nonfinite_count,
        "soft_limit_violation_count": soft_limit_violation_count,
        "phantom_observation_max_abs": round(phantom_max, 12),
        "max_recovery_time_s": round(max(recovery_by_episode, default=0.0), 12),
        "action_saturation_fraction": round(
            saturated_total / action_total if action_total else 1.0,
            12,
        ),
        "mpjpe_m": round(
            mpjpe_total / mpjpe_count if mpjpe_count else 1.0e30,
            12,
        ),
    }


def _validate_trace_array(
    record: Mapping[str, Any],
    key: str,
    episodes: int,
    context: str,
) -> Sequence[Any]:
    value = record[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context}.{key} must be an array")
    if len(value) != episodes:
        raise ValueError(f"{context}.{key} must contain {episodes} values")
    return value


def _validate_raw_trace(
    trace_record: Mapping[str, Any],
    *,
    report_path: Path,
    scenario: str,
    seed: int,
    episodes: int,
    disturbance_scale: float,
    checkpoint_sha256: str,
    producer: Mapping[str, Any],
    simulator: Mapping[str, Any],
    material_provenance: Mapping[str, Any],
    config: Mapping[str, Any],
    context: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_exact_keys(
        trace_record,
        {"file", "sha256", "payload_sha256", "record_count"},
        f"{context}.trace",
    )
    expected_relative = (
        f"{report_path.stem}.traces/{scenario}-seed-{seed}.json"
    )
    if trace_record["file"] != expected_relative:
        raise ValueError(f"{context}.trace.file must be {expected_relative!r}")
    trace_path = (report_path.parent / expected_relative).resolve()
    report_parent = report_path.parent.resolve()
    try:
        trace_path.relative_to(report_parent)
    except ValueError as exc:
        raise ValueError(f"{context}.trace.file escapes report directory") from exc
    if not trace_path.is_file():
        raise ValueError(f"{context}.trace file is missing")
    trace_bytes = trace_path.read_bytes()
    if _require_sha256(trace_record["sha256"], f"{context}.trace.sha256") != sha256_bytes(
        trace_bytes
    ):
        raise ValueError(f"{context}.trace SHA-256 mismatch")
    trace = load_strict_json(trace_path)
    payload_sha256 = sha256_bytes(canonical_json_bytes(trace))
    if (
        _require_sha256(
            trace_record["payload_sha256"],
            f"{context}.trace.payload_sha256",
        )
        != payload_sha256
    ):
        raise ValueError(f"{context}.trace payload SHA-256 mismatch")
    _require_exact_keys(
        trace,
        {
            "schema_version",
            "kind",
            "checkpoint_sha256",
            "producer",
            "simulator",
            "material_provenance",
            "scenario",
            "seed",
            "disturbance_scale",
            "control_hz",
            "episodes",
            "steps_per_episode",
            "records",
        },
        f"{context}.trace payload",
    )
    if trace["schema_version"] != SIM_TRACE_SCHEMA_VERSION:
        raise ValueError(f"{context}.trace schema_version is unsupported")
    if trace["kind"] != SIM_TRACE_KIND:
        raise ValueError(f"{context}.trace kind is unsupported")
    expected_header = {
        "checkpoint_sha256": checkpoint_sha256,
        "producer": producer,
        "simulator": simulator,
        "material_provenance": material_provenance,
        "scenario": scenario,
        "seed": seed,
        "disturbance_scale": disturbance_scale,
        "control_hz": config["control_hz"],
        "episodes": episodes,
        "steps_per_episode": config["minimum_coverage"]["steps_per_episode"],
    }
    for key, expected_value in expected_header.items():
        if trace[key] != expected_value:
            raise ValueError(f"{context}.trace {key} does not match report contract")
    records = trace["records"]
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError(f"{context}.trace records must be an array")
    expected_record_count = int(config["minimum_coverage"]["steps_per_episode"])
    if (
        _require_int(
            trace_record["record_count"],
            f"{context}.trace.record_count",
            minimum=1,
        )
        != expected_record_count
        or len(records) != expected_record_count
    ):
        raise ValueError(f"{context}.trace record count mismatch")

    bool_keys = (
        "terminated",
        "timed_out",
        "nonfinite",
        "soft_limit_violation",
    )
    float_keys = (
        "phantom_observation_max_abs",
        "recovery_metric",
        "mpjpe_m",
    )
    int_keys = ("action_saturated_count", "action_count")
    apply_step = int(config["disturbance_schedule"]["apply_step"])
    normalized_records: list[Mapping[str, Any]] = []
    for step, record in enumerate(records):
        record_context = f"{context}.trace.records[{step}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{record_context} must be an object")
        _require_exact_keys(
            record,
            {
                "step",
                "disturbance_delta",
                *bool_keys,
                *float_keys,
                *int_keys,
            },
            record_context,
        )
        if _require_int(record["step"], f"{record_context}.step") != step:
            raise ValueError(f"{record_context}.step is not contiguous")
        disturbance = _validate_trace_array(
            record,
            "disturbance_delta",
            episodes,
            record_context,
        )
        for episode, vector in enumerate(disturbance):
            if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
                raise ValueError(
                    f"{record_context}.disturbance_delta[{episode}] must be an array"
                )
            if len(vector) != 6:
                raise ValueError(
                    f"{record_context}.disturbance_delta[{episode}] must have six axes"
                )
            actual = [
                _require_float(
                    value,
                    f"{record_context}.disturbance_delta[{episode}][{axis}]",
                    minimum=None,
                )
                for axis, value in enumerate(vector)
            ]
            expected = (
                _deterministic_disturbance_vector(
                    seed=seed,
                    episode_index=episode,
                    scale=disturbance_scale,
                    config=config,
                )
                if step == apply_step
                else [0.0] * 6
            )
            if actual != expected:
                raise ValueError(
                    f"{record_context}.disturbance_delta[{episode}] is not "
                    "the deterministic configured velocity delta"
                )
        for key in bool_keys:
            values = _validate_trace_array(record, key, episodes, record_context)
            if any(not isinstance(value, bool) for value in values):
                raise ValueError(f"{record_context}.{key} must contain booleans")
        for key in float_keys:
            values = _validate_trace_array(record, key, episodes, record_context)
            for episode, value in enumerate(values):
                _require_float(value, f"{record_context}.{key}[{episode}]")
        saturated_values = _validate_trace_array(
            record,
            "action_saturated_count",
            episodes,
            record_context,
        )
        action_count_values = _validate_trace_array(
            record,
            "action_count",
            episodes,
            record_context,
        )
        for episode, (saturated, action_count) in enumerate(
            zip(saturated_values, action_count_values, strict=True)
        ):
            saturated = _require_int(
                saturated,
                f"{record_context}.action_saturated_count[{episode}]",
            )
            action_count = _require_int(
                action_count,
                f"{record_context}.action_count[{episode}]",
                minimum=1,
            )
            if action_count != TARGET_DOF or saturated > action_count:
                raise ValueError(f"{record_context} action counts violate true23 contract")
        normalized_records.append(record)

    metrics = _trace_metrics(
        normalized_records,
        episodes=episodes,
        disturbance_scale=disturbance_scale,
        config=config,
    )
    manifest_record = {
        "scenario": scenario,
        "seed": seed,
        "file": trace_record["file"],
        "sha256": trace_record["sha256"],
        "payload_sha256": trace_record["payload_sha256"],
        "record_count": trace_record["record_count"],
    }
    return metrics, manifest_record


def validate_simulation_report(
    report: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    report_sha256: str,
    report_payload_sha256: str,
    report_path: Path,
    reference_profile: str = DEFAULT_REFERENCE_PROFILE,
    checkpoint_motion_dataset: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate explicit nominal and disturbance evidence, then summarize it.

    No top-level ``passed`` boolean is consumed.  Pass status is recomputed from
    per-run hard metrics and exact checkpoint/asset bindings.
    """
    _require_exact_keys(
        report,
        {
            "schema_version",
            "kind",
            "robot_model",
            "checkpoint_sha256",
            "producer",
            "reference_profile",
            "reference_contract",
            "observation_layout",
            "history_length",
            "decoder_input_dim",
            "decoder_output_dim",
            "decoder_output_layout",
            "runtime_config",
            "simulator",
            "material_provenance",
            "trace_manifest_sha256",
            "runs",
        },
        "simulation report",
    )
    report_path = report_path.resolve()
    if not report_path.is_file():
        raise ValueError("simulation report path must name the evidence file")
    if report["schema_version"] != SIM_VALIDATION_SCHEMA_VERSION:
        raise ValueError("unsupported simulation report schema_version")
    if report["kind"] != SIM_REPORT_KIND:
        raise ValueError(f"simulation report kind must be {SIM_REPORT_KIND!r}")
    if report["robot_model"] != ROBOT_MODEL:
        raise ValueError(f"simulation report robot_model must be {ROBOT_MODEL!r}")
    if report["reference_profile"] != reference_profile:
        raise ValueError("simulation report reference_profile differs from checkpoint")
    if report["reference_contract"] != reference_profile_contract(reference_profile):
        raise ValueError("simulation report reference_contract mismatch")
    if _require_sha256(report["checkpoint_sha256"], "checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("simulation report is bound to a different checkpoint")
    validation_config = _sim_validation_config()
    producer_contract = validation_config["producer"]
    if producer_contract["promotion_enabled"] is not True or producer_contract["runner_sha256"] is None:
        raise ValueError(
            "true23 deployment promotion unavailable: no checked-in IsaacLab "
            "disturbance validation producer has been approved"
        )
    if not _SIM_VALIDATION_RUNNER_PATH.is_file():
        raise ValueError("approved true23 simulation validation runner is missing")
    runner_sha256 = sha256_file(_SIM_VALIDATION_RUNNER_PATH)
    if runner_sha256 != producer_contract["runner_sha256"]:
        raise ValueError("true23 simulation validation runner hash changed")
    producer = report["producer"]
    if not isinstance(producer, Mapping):
        raise ValueError("simulation report producer must be an object")
    _require_exact_keys(
        producer,
        {"kind", "version", "runner_sha256"},
        "simulation report producer",
    )
    expected_producer = {
        "kind": producer_contract["kind"],
        "version": producer_contract["version"],
        "runner_sha256": runner_sha256,
    }
    if dict(producer) != expected_producer:
        raise ValueError("simulation report producer does not match approved runner")
    runtime_config = report["runtime_config"]
    if not isinstance(runtime_config, Mapping):
        raise ValueError("simulation report runtime_config must be an object")
    _require_exact_keys(
        runtime_config,
        {
            "resolved",
            "resolved_config_sha256",
            "material_config_sha256",
        },
        "simulation report runtime_config",
    )
    resolved_runtime_config = runtime_config["resolved"]
    if not isinstance(resolved_runtime_config, Mapping):
        raise ValueError("simulation report runtime_config.resolved must be an object")
    runtime_hashes = validate_runtime_config_snapshot(
        resolved_runtime_config,
        validation_config=validation_config,
    )
    if dict(runtime_config) != {"resolved": resolved_runtime_config, **runtime_hashes}:
        raise ValueError("simulation report resolved runtime config hashes do not match")
    material_provenance = report["material_provenance"]
    if not isinstance(material_provenance, Mapping):
        raise ValueError("simulation report material_provenance must be an object")
    expected_material_provenance = simulation_material_provenance(
        resolved_runtime_config,
        checkpoint_motion_dataset=checkpoint_motion_dataset,
        validation_config=validation_config,
        repo_root=_REPOSITORY_ROOT,
    )
    if dict(material_provenance) != expected_material_provenance:
        raise ValueError(
            "simulation report material_provenance differs from local runtime "
            "sources or trained motion dataset"
        )
    if report["observation_layout"] != OBS_LAYOUT_PADDED_IL29:
        raise ValueError("simulation report observation layout is not canonical padded IL29")
    if report["history_length"] != DEPLOYMENT_HISTORY_LENGTH:
        raise ValueError("simulation report history_length must be exactly 10")
    if report["decoder_input_dim"] != DEPLOYMENT_DECODER_INPUT_DIM:
        raise ValueError("simulation report decoder_input_dim must be exactly 994")
    if report["decoder_output_dim"] != TARGET_DOF:
        raise ValueError("simulation report decoder_output_dim must be exactly 23")
    if report["decoder_output_layout"] != DECODER_OUTPUT_LAYOUT:
        raise ValueError(f"simulation report decoder_output_layout must be {DECODER_OUTPUT_LAYOUT!r}")

    simulator = report["simulator"]
    if not isinstance(simulator, Mapping):
        raise ValueError("simulation report simulator must be an object")
    _require_exact_keys(
        simulator,
        {
            "name",
            "version",
            "asset_sha256",
            "robot_config_sha256",
            "config_sha256",
            "runtime_config_sha256",
        },
        "simulation report simulator",
    )
    for key in ("name", "version"):
        if not isinstance(simulator[key], str) or not simulator[key].strip():
            raise ValueError(f"simulation report simulator.{key} must be non-empty")
    if simulator["name"] != "IsaacLab":
        raise ValueError("simulation report simulator.name must be exactly 'IsaacLab'")
    local_asset_sha256 = sha256_file(_ROBOT_ASSET_PATH)
    if _require_sha256(simulator["asset_sha256"], "simulator.asset_sha256") != local_asset_sha256:
        raise ValueError("simulation report asset hash does not match vendored true23 URDF")
    local_robot_config_sha256 = sha256_file(_ROBOT_CONFIG_PATH)
    if (
        _require_sha256(
            simulator["robot_config_sha256"],
            "simulator.robot_config_sha256",
        )
        != local_robot_config_sha256
    ):
        raise ValueError("simulation report robot config hash does not match true23 config")
    simulator_config_sha256 = _require_sha256(simulator["config_sha256"], "simulator.config_sha256")
    local_simulator_config_sha256 = sha256_file(_SIM_VALIDATION_CONFIG_PATH)
    if simulator_config_sha256 != local_simulator_config_sha256:
        raise ValueError("simulation report config hash does not match checked-in validation envelope")
    if (
        _require_sha256(
            simulator["runtime_config_sha256"],
            "simulator.runtime_config_sha256",
        )
        != runtime_hashes["resolved_config_sha256"]
    ):
        raise ValueError("simulation report simulator runtime config hash mismatch")

    runs = report["runs"]
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)) or not runs:
        raise ValueError("simulation report runs must be a non-empty array")

    expected_scales = {
        scenario: values["disturbance_scale"] for scenario, values in validation_config["scenarios"].items()
    }
    minimum_coverage = validation_config["minimum_coverage"]
    deterministic_seeds = tuple(validation_config["deterministic_seeds"])
    expected_identities = [
        (scenario, seed)
        for scenario in validation_config["scenarios"]
        for seed in deterministic_seeds
    ]
    if len(runs) != len(expected_identities):
        raise ValueError(
            f"simulation report must contain exactly {len(expected_identities)} "
            "configured scenario/seed runs"
        )
    seen_scenarios: set[str] = set()
    seen_scenario_seeds: set[tuple[str, int]] = set()
    scenario_seeds = {scenario: set() for scenario in REQUIRED_SIM_SCENARIOS}
    scenario_episodes = {scenario: 0 for scenario in REQUIRED_SIM_SCENARIOS}
    scenario_steps = {scenario: 0 for scenario in REQUIRED_SIM_SCENARIOS}
    max_metrics = {
        "phantom_observation_max_abs": 0.0,
        "max_recovery_time_s": 0.0,
        "action_saturation_fraction": 0.0,
        "mpjpe_m": 0.0,
    }
    total_episodes = 0
    total_steps = 0
    trace_manifest: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        context = f"simulation report runs[{index}]"
        if not isinstance(run, Mapping):
            raise ValueError(f"{context} must be an object")
        _require_exact_keys(
            run,
            {
                "scenario",
                "seed",
                "episodes",
                "steps",
                "disturbance_scale",
                "termination_count",
                "nonfinite_count",
                "soft_limit_violation_count",
                "phantom_observation_max_abs",
                "max_recovery_time_s",
                "action_saturation_fraction",
                "mpjpe_m",
                "trace",
            },
            context,
        )
        scenario = run["scenario"]
        if scenario not in expected_scales:
            raise ValueError(f"{context}.scenario is unsupported: {scenario!r}")
        seed = _require_int(run["seed"], f"{context}.seed")
        identity = (str(scenario), seed)
        if identity != expected_identities[index]:
            raise ValueError(
                f"{context} identity must be {expected_identities[index]!r}, "
                f"got {identity!r}"
            )
        if identity in seen_scenario_seeds:
            raise ValueError(f"{context} duplicates scenario/seed {identity!r}")
        seen_scenario_seeds.add(identity)
        seen_scenarios.add(str(scenario))
        scenario_seeds[str(scenario)].add(seed)

        episodes = _require_int(run["episodes"], f"{context}.episodes", minimum=1)
        steps = _require_int(run["steps"], f"{context}.steps", minimum=1)
        if episodes != int(minimum_coverage["episodes_per_seed"]):
            raise ValueError(
                f"{context}.episodes must be exactly "
                f"{minimum_coverage['episodes_per_seed']}"
            )
        required_steps = episodes * int(minimum_coverage["steps_per_episode"])
        if steps != required_steps:
            raise ValueError(
                f"{context}.steps must cover exactly "
                f"{minimum_coverage['seconds_per_episode']}s per episode at "
                f"{validation_config['control_hz']}Hz"
            )
        total_episodes += episodes
        total_steps += steps
        scenario_episodes[str(scenario)] += episodes
        scenario_steps[str(scenario)] += steps
        disturbance_scale = _require_float(run["disturbance_scale"], f"{context}.disturbance_scale")
        if disturbance_scale != expected_scales[str(scenario)]:
            raise ValueError(f"{context}.disturbance_scale must be {expected_scales[str(scenario)]}")
        trace_record = run["trace"]
        if not isinstance(trace_record, Mapping):
            raise ValueError(f"{context}.trace must be an object")
        recomputed, manifest_record = _validate_raw_trace(
            trace_record,
            report_path=report_path,
            scenario=str(scenario),
            seed=seed,
            episodes=episodes,
            disturbance_scale=disturbance_scale,
            checkpoint_sha256=checkpoint_sha256,
            producer=expected_producer,
            simulator=simulator,
            material_provenance=expected_material_provenance,
            config=validation_config,
            context=context,
        )
        trace_manifest.append(manifest_record)
        for count_name in (
            "termination_count",
            "nonfinite_count",
            "soft_limit_violation_count",
        ):
            actual_count = _require_int(run[count_name], f"{context}.{count_name}")
            if actual_count != recomputed[count_name]:
                raise ValueError(f"{context}.{count_name} does not match raw trace")
            if actual_count != 0:
                raise ValueError(f"{context}.{count_name} must be zero")

        phantom = _require_float(
            run["phantom_observation_max_abs"],
            f"{context}.phantom_observation_max_abs",
        )
        recovery = _require_float(run["max_recovery_time_s"], f"{context}.max_recovery_time_s")
        saturation = _require_float(
            run["action_saturation_fraction"],
            f"{context}.action_saturation_fraction",
        )
        mpjpe = _require_float(run["mpjpe_m"], f"{context}.mpjpe_m")
        for metric_name, value in (
            ("phantom_observation_max_abs", phantom),
            ("max_recovery_time_s", recovery),
            ("action_saturation_fraction", saturation),
            ("mpjpe_m", mpjpe),
        ):
            if value != recomputed[metric_name]:
                raise ValueError(f"{context}.{metric_name} does not match raw trace")
        if phantom > MAX_PHANTOM_OBSERVATION_ABS:
            raise ValueError(f"{context} phantom observation slots are not fixed")
        if recovery > MAX_RECOVERY_TIME_S:
            raise ValueError(f"{context} recovery exceeded {MAX_RECOVERY_TIME_S}s")
        if saturation > MAX_ACTION_SATURATION_FRACTION:
            raise ValueError(f"{context} action saturation exceeded safety threshold")
        if mpjpe > MAX_MPJPE_M:
            raise ValueError(f"{context} MPJPE exceeded safety threshold")
        max_metrics["phantom_observation_max_abs"] = max(max_metrics["phantom_observation_max_abs"], phantom)
        max_metrics["max_recovery_time_s"] = max(max_metrics["max_recovery_time_s"], recovery)
        max_metrics["action_saturation_fraction"] = max(max_metrics["action_saturation_fraction"], saturation)
        max_metrics["mpjpe_m"] = max(max_metrics["mpjpe_m"], mpjpe)

    required_scenarios = set(REQUIRED_SIM_SCENARIOS)
    if seen_scenarios != required_scenarios:
        missing = sorted(required_scenarios - seen_scenarios)
        extra = sorted(seen_scenarios - required_scenarios)
        raise ValueError(f"simulation scenario coverage mismatch; missing={missing}, extra={extra}")
    for scenario in REQUIRED_SIM_SCENARIOS:
        if scenario_seeds[scenario] != set(deterministic_seeds):
            raise ValueError(
                f"simulation scenario {scenario} must use exact deterministic seeds"
            )
        if scenario_episodes[scenario] < minimum_coverage["episodes_per_scenario"]:
            raise ValueError(
                f"simulation scenario {scenario} requires at least "
                f"{minimum_coverage['episodes_per_scenario']} episodes"
            )
    manifest_sha256 = sha256_bytes(canonical_json_bytes(trace_manifest))
    if (
        _require_sha256(
            report["trace_manifest_sha256"],
            "simulation report trace_manifest_sha256",
        )
        != manifest_sha256
    ):
        raise ValueError("simulation report trace manifest hash mismatch")

    return {
        "schema_version": SIM_VALIDATION_SCHEMA_VERSION,
        "computed_pass": True,
        "report_sha256": _require_sha256(report_sha256, "report_sha256"),
        "report_payload_sha256": _require_sha256(report_payload_sha256, "report_payload_sha256"),
        "checkpoint_sha256": checkpoint_sha256,
        "producer": expected_producer,
        "runtime_config": runtime_hashes,
        "material_provenance": expected_material_provenance,
        "trace_manifest_sha256": manifest_sha256,
        "trace_count": len(trace_manifest),
        "required_scenarios": list(REQUIRED_SIM_SCENARIOS),
        "run_count": len(runs),
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "scenario_coverage": {
            scenario: {
                "seed_count": len(scenario_seeds[scenario]),
                "episodes": scenario_episodes[scenario],
                "steps": scenario_steps[scenario],
            }
            for scenario in REQUIRED_SIM_SCENARIOS
        },
        "simulator": {
            "name": simulator["name"],
            "version": simulator["version"],
            "asset_sha256": local_asset_sha256,
            "robot_config_sha256": local_robot_config_sha256,
            "config_sha256": simulator_config_sha256,
            "runtime_config_sha256": runtime_hashes["resolved_config_sha256"],
        },
        "max_metrics": {key: round(value, 12) for key, value in max_metrics.items()},
    }


def _policy_state_sha256(policy_state: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for key in sorted(policy_state):
        tensor = policy_state[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"policy_state_dict[{key!r}] is not a tensor")
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(contiguous.shape)))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_training_checkpoint_records(
    *,
    global_step: int,
    history_length: int,
    observation_layout: str,
    policy_state_sha256: str,
    reference_profile: str = DEFAULT_REFERENCE_PROFILE,
    source_family: str,
    source_revision: str | None,
    source_checkpoint_sha256: str,
    initial_policy_state_sha256: str,
    motion_dataset: Mapping[str, Any],
    training_material: Mapping[str, Any],
    training_start_global_step: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build records written by training checkpoint save, never by initializer."""
    _require_int(global_step, "global_step", minimum=1)
    _require_sha256(policy_state_sha256, "policy_state_sha256")
    _require_sha256(
        source_checkpoint_sha256,
        "source_checkpoint_sha256",
    )
    _require_sha256(
        initial_policy_state_sha256,
        "initial_policy_state_sha256",
    )
    _require_int(
        training_start_global_step,
        "training_start_global_step",
    )
    release = APPROVED_WARM_START_RELEASES.get(source_checkpoint_sha256)
    if (
        release is None
        or release["source_family"] != source_family
        or release["source_revision"] != source_revision
        or release["reference_profile"] != reference_profile
        or release["initial_policy_state_sha256"]
        != initial_policy_state_sha256
    ):
        raise ValueError("training lineage is not an approved warm-start release")
    training_updates = global_step - training_start_global_step
    if training_updates < MINIMUM_TRAINING_UPDATES:
        raise ValueError(
            "trained checkpoint has fewer than "
            f"{MINIMUM_TRAINING_UPDATES} policy updates"
        )
    if policy_state_sha256 == initial_policy_state_sha256:
        raise ValueError("trained policy weights are unchanged from initialization")
    normalized_motion_dataset = validate_motion_dataset_evidence(
        motion_dataset,
        context="training motion_dataset",
    )
    normalized_training_material = validate_training_material_evidence(
        training_material,
        context="training material",
    )
    metadata = make_artifact_metadata(
        history_length=history_length,
        observation_layout=observation_layout,
        checkpoint_stage="trained",
        reference_profile=reference_profile,
    )
    evidence = {
        "schema_version": TRAINING_EVIDENCE_SCHEMA_VERSION,
        "kind": TRAINING_EVIDENCE_KIND,
        "producer": TRAINING_EVIDENCE_PRODUCER,
        "robot_model": ROBOT_MODEL,
        "global_step": global_step,
        "history_length": history_length,
        "observation_layout": observation_layout,
        "decoder_input_dim": metadata["decoder_input_dim"],
        "decoder_output_dim": TARGET_DOF,
        "reference_profile": reference_profile,
        "reference_contract": reference_profile_contract(reference_profile),
        "source_family": source_family,
        "source_revision": source_revision,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "initial_policy_state_sha256": initial_policy_state_sha256,
        "training_start_global_step": training_start_global_step,
        "training_updates": training_updates,
        "minimum_training_updates": MINIMUM_TRAINING_UPDATES,
        "policy_state_sha256": policy_state_sha256,
        "motion_dataset": normalized_motion_dataset,
        "training_material": normalized_training_material,
        "weights_only_initialization": False,
    }
    return metadata, evidence


def validate_training_checkpoint_records(
    checkpoint: Mapping[str, Any],
    *,
    global_step: int,
    policy_state_sha256: str,
) -> None:
    """Reject initialization-only or unmarked checkpoints before export."""
    metadata = checkpoint.get("g1_23dof_metadata")
    evidence = checkpoint.get("g1_23dof_training_evidence")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint lacks training-produced g1_23dof_metadata")
    validate_artifact_contract(
        metadata,
        decoder_input_dim=DEPLOYMENT_DECODER_INPUT_DIM,
        decoder_output_dim=TARGET_DOF,
        require_deployment_ready=False,
    )
    if metadata.get("checkpoint_stage") != "trained":
        raise ValueError("checkpoint_stage must be 'trained', not initialization-only")
    reference_profile = metadata.get("reference_profile")
    if not isinstance(reference_profile, str):
        raise ValueError("checkpoint metadata lacks reference_profile")
    expected_metadata = make_artifact_metadata(
        history_length=DEPLOYMENT_HISTORY_LENGTH,
        observation_layout=OBS_LAYOUT_PADDED_IL29,
        checkpoint_stage="trained",
        reference_profile=reference_profile,
    )
    if dict(metadata) != expected_metadata:
        raise ValueError("checkpoint metadata does not match current native true23 contract")
    if not isinstance(evidence, Mapping):
        raise ValueError("checkpoint lacks g1_23dof_training_evidence")
    expected_keys = {
        "schema_version": TRAINING_EVIDENCE_SCHEMA_VERSION,
        "kind": None,
        "producer": None,
        "robot_model": None,
        "global_step": None,
        "history_length": None,
        "observation_layout": None,
        "decoder_input_dim": None,
        "decoder_output_dim": None,
        "reference_profile": None,
        "reference_contract": None,
        "source_family": None,
        "source_revision": None,
        "source_checkpoint_sha256": None,
        "initial_policy_state_sha256": None,
        "training_start_global_step": None,
        "training_updates": None,
        "minimum_training_updates": None,
        "policy_state_sha256": None,
        "motion_dataset": None,
        "training_material": None,
        "weights_only_initialization": None,
    }
    if set(evidence) != set(expected_keys):
        raise ValueError("g1_23dof_training_evidence has unexpected keys")
    expected_values = {
        "schema_version": TRAINING_EVIDENCE_SCHEMA_VERSION,
        "kind": TRAINING_EVIDENCE_KIND,
        "producer": TRAINING_EVIDENCE_PRODUCER,
        "robot_model": ROBOT_MODEL,
        "global_step": global_step,
        "history_length": DEPLOYMENT_HISTORY_LENGTH,
        "observation_layout": OBS_LAYOUT_PADDED_IL29,
        "decoder_input_dim": DEPLOYMENT_DECODER_INPUT_DIM,
        "decoder_output_dim": TARGET_DOF,
        "reference_profile": reference_profile,
        "reference_contract": reference_profile_contract(reference_profile),
        "minimum_training_updates": MINIMUM_TRAINING_UPDATES,
        "policy_state_sha256": policy_state_sha256,
        "weights_only_initialization": False,
    }
    if any(evidence.get(key) != value for key, value in expected_values.items()):
        raise ValueError("g1_23dof_training_evidence does not match trained H10 contract")
    validate_motion_dataset_evidence(
        evidence.get("motion_dataset"),
        context="g1_23dof_training_evidence.motion_dataset",
    )
    validate_training_material_evidence(
        evidence.get("training_material"),
        context="g1_23dof_training_evidence.training_material",
    )
    source_sha256 = evidence.get("source_checkpoint_sha256")
    release = APPROVED_WARM_START_RELEASES.get(source_sha256)
    if (
        release is None
        or evidence.get("source_family") != release["source_family"]
        or evidence.get("source_revision") != release["source_revision"]
        or reference_profile != release["reference_profile"]
        or evidence.get("initial_policy_state_sha256")
        != release["initial_policy_state_sha256"]
    ):
        raise ValueError("training evidence source lineage is not approved")
    initial_sha256 = evidence.get("initial_policy_state_sha256")
    _require_sha256(initial_sha256, "initial_policy_state_sha256")
    if initial_sha256 == policy_state_sha256:
        raise ValueError("trained policy weights are unchanged from initialization")
    start_step = _require_int(
        evidence.get("training_start_global_step"),
        "training_start_global_step",
    )
    updates = _require_int(evidence.get("training_updates"), "training_updates")
    if (
        updates != global_step - start_step
        or updates < MINIMUM_TRAINING_UPDATES
    ):
        raise ValueError("training evidence does not prove minimum policy updates")


def _validated_decoder_tensors(
    checkpoint: Mapping[str, Any],
    *,
    reference_profile: str | None = None,
) -> tuple[Mapping[str, Any], list[tuple[Any, Any]], list[tuple[Any, Any]], str]:
    import torch

    reference_profile = (
        _checkpoint_reference_profile(checkpoint)
        if reference_profile is None
        else reference_profile
    )
    decoder_dims = _decoder_layer_dims(reference_profile)
    decoder_indices = tuple(
        range(0, 2 * (len(decoder_dims) - 1), 2)
    )
    policy_state = checkpoint.get("policy_state_dict")
    if not isinstance(policy_state, Mapping):
        raise ValueError("checkpoint has no policy_state_dict mapping")

    unexpected_families = sorted(
        key
        for key in policy_state
        if (key.startswith("actor_module.encoders.") and not key.startswith(_ENCODER_PREFIX))
        or (key.startswith("actor_module.decoders.") and not key.startswith(_DECODER_PREFIX))
    )
    if unexpected_families:
        raise ValueError(
            f"true23 teleop checkpoint contains unexpected encoder/decoder family: {unexpected_families[0]}"
        )

    def validate_family(
        *,
        prefix: str,
        tensor_pattern: re.Pattern[str],
        expected_indices: tuple[int, ...],
        expected_dims: tuple[int, ...],
        family_name: str,
    ) -> list[tuple[Any, Any]]:
        related = {key: value for key, value in policy_state.items() if key.startswith(prefix)}
        parsed: dict[int, dict[str, Any]] = {}
        for key, tensor in related.items():
            match = tensor_pattern.fullmatch(key)
            if match is None:
                raise ValueError(f"unsupported {family_name} tensor: {key}")
            parsed.setdefault(int(match.group("index")), {})[match.group("kind")] = tensor
        if not parsed:
            raise ValueError(f"checkpoint has no {family_name} tensors")
        layer_indices = tuple(sorted(parsed))
        if layer_indices != expected_indices:
            raise ValueError(
                f"unexpected {family_name} linear layer indices: {list(layer_indices)}; "
                f"expected {list(expected_indices)}"
            )

        affine_tensors = []
        for position, layer_index in enumerate(layer_indices):
            tensors = parsed[layer_index]
            if set(tensors) != {"weight", "bias"}:
                raise ValueError(f"{family_name} layer {layer_index} must contain weight and bias")
            weight = tensors["weight"]
            bias = tensors["bias"]
            if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
                raise ValueError(f"{family_name} layer {layer_index} tensors must be torch.Tensor")
            if weight.dtype != torch.float32 or bias.dtype != torch.float32:
                raise ValueError(f"{family_name} layer {layer_index} must use float32 weights")
            expected_shape = (expected_dims[position + 1], expected_dims[position])
            if tuple(weight.shape) != expected_shape or tuple(bias.shape) != (expected_shape[0],):
                raise ValueError(
                    f"{family_name} layer {layer_index} must be "
                    f"{expected_shape[1]}->{expected_shape[0]}, "
                    f"got weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
                )
            affine_tensors.append((weight, bias))
        return affine_tensors

    encoder_affine_tensors = validate_family(
        prefix=_ENCODER_PREFIX,
        tensor_pattern=_ENCODER_TENSOR_RE,
        expected_indices=_ENCODER_LINEAR_INDICES,
        expected_dims=_ENCODER_LAYER_DIMS,
        family_name="teleop encoder",
    )
    decoder_affine_tensors = validate_family(
        prefix=_DECODER_PREFIX,
        tensor_pattern=_DECODER_TENSOR_RE,
        expected_indices=decoder_indices,
        expected_dims=decoder_dims,
        family_name="g1_dyn decoder",
    )

    noise_keys = [
        key
        for key, value in policy_state.items()
        if key.rsplit(".", 1)[-1] in {"std", "log_std"} and isinstance(value, torch.Tensor) and value.ndim == 1
    ]
    if len(noise_keys) != 1 or policy_state[noise_keys[0]].shape[0] != TARGET_DOF:
        raise ValueError("trained checkpoint requires exactly one 23-D std or log_std tensor")
    return (
        policy_state,
        encoder_affine_tensors,
        decoder_affine_tensors,
        _policy_state_sha256(policy_state),
    )


def inspect_true23_policy_state(
    checkpoint: Mapping[str, Any],
    *,
    reference_profile: str | None = None,
) -> str:
    """Validate policy shapes and hash weights without duplicating the large decoder."""
    _, _, _, policy_state_sha256 = _validated_decoder_tensors(
        checkpoint,
        reference_profile=reference_profile,
    )
    return policy_state_sha256


def _build_silu_mlp(affine_tensors: Sequence[tuple[Any, Any]]):
    import torch
    from torch import nn

    layers = nn.ModuleList()
    for weight, bias in affine_tensors:
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
        with torch.no_grad():
            linear.weight.copy_(weight.detach().cpu())
            linear.bias.copy_(bias.detach().cpu())
        linear.requires_grad_(False)
        layers.append(linear)

    class SiluMlp(nn.Module):
        def __init__(self, affine_layers: nn.ModuleList):
            super().__init__()
            self.layers = affine_layers

        def forward(self, obs_dict):
            value = obs_dict
            for index, layer in enumerate(self.layers):
                value = layer(value)
                if index + 1 != len(self.layers):
                    value = torch.nn.functional.silu(value)
            return value

    return SiluMlp(layers).eval()


def build_true23_policy_pair(checkpoint: Mapping[str, Any]):
    """Reconstruct exact teleop encoder+FSQ and true23 decoder from checkpoint."""
    import torch
    from torch import nn

    _, encoder_tensors, decoder_tensors, policy_state_sha256 = _validated_decoder_tensors(checkpoint)
    raw_encoder = _build_silu_mlp(encoder_tensors)
    decoder = _build_silu_mlp(decoder_tensors)

    class TeleopEncoder(nn.Module):
        def __init__(self, mlp: nn.Module):
            super().__init__()
            self.mlp = mlp

        def forward(self, teleop_obs):
            latent = self.mlp(teleop_obs).reshape(
                teleop_obs.shape[0],
                TELEOP_TOKEN_COUNT,
                TELEOP_TOKEN_WIDTH,
            )
            # Exact vector-quantize-pytorch FSQ defaults for 32 even levels:
            # tanh bound, round-STE forward value, one codebook, no projections.
            half_l = (TELEOP_FSQ_LEVEL - 1) * (1.0 + 1.0e-3) / 2.0
            offset = 0.5
            shift = math.atanh(offset / half_l)
            bounded = torch.tanh(latent + shift) * half_l - offset
            quantized = torch.round(bounded) / (TELEOP_FSQ_LEVEL // 2)
            return quantized.reshape(teleop_obs.shape[0], TOKEN_DIM)

    return TeleopEncoder(raw_encoder).eval(), decoder.eval(), policy_state_sha256


def build_true23_decoder(checkpoint: Mapping[str, Any]):
    """Compatibility helper returning the decoder from the validated pair."""
    _, decoder, policy_state_sha256 = build_true23_policy_pair(checkpoint)
    return decoder, policy_state_sha256


def validate_true23_policy_module(
    policy: Any,
    *,
    reference_profile: str = DEFAULT_REFERENCE_PROFILE,
) -> None:
    """Validate live training module topology/activation/FSQ before marking a save."""
    import torch
    from torch import nn

    from gear_sonic.trl.modules.base_module import BaseModule

    actor_module = getattr(policy, "actor_module", None)
    encoders = getattr(actor_module, "encoders", None)
    decoders = getattr(actor_module, "decoders", None)
    if encoders is None or tuple(encoders.keys()) != ("teleop",):
        raise ValueError("true23 policy must contain only the teleop encoder")
    if decoders is None or tuple(decoders.keys()) != ("g1_dyn",):
        raise ValueError("true23 policy must contain only the g1_dyn decoder")

    def validate_sequential(
        module: Any,
        expected_dims: tuple[int, ...],
        family_name: str,
        *,
        num_output_temporal_dims: int | None,
    ) -> None:
        if type(module) is not BaseModule:
            raise ValueError(f"{family_name} must be the exact BaseModule class")
        if (
            module.input_dim != expected_dims[0]
            or module.output_dim != expected_dims[-1]
            or module.num_input_temporal_dims is not None
            or module.num_output_temporal_dims != num_output_temporal_dims
        ):
            raise ValueError(f"{family_name} BaseModule dimensions/reshape mismatch")
        sequence = getattr(module, "module", None)
        expected_count = 2 * (len(expected_dims) - 2) + 1
        if not isinstance(sequence, nn.Sequential) or len(sequence) != expected_count:
            raise ValueError(f"{family_name} is not the exact released SiLU MLP")
        for index, layer in enumerate(sequence):
            if index % 2:
                if not isinstance(layer, nn.SiLU):
                    raise ValueError(f"{family_name} activation {index} must be SiLU")
                continue
            linear_position = index // 2
            if (
                not isinstance(layer, nn.Linear)
                or layer.in_features != expected_dims[linear_position]
                or layer.out_features != expected_dims[linear_position + 1]
            ):
                raise ValueError(f"{family_name} linear layer {index} shape mismatch")

    validate_sequential(
        encoders["teleop"],
        _ENCODER_LAYER_DIMS,
        "teleop encoder",
        num_output_temporal_dims=TELEOP_TOKEN_COUNT,
    )
    validate_sequential(
        decoders["g1_dyn"],
        _decoder_layer_dims(reference_profile),
        "g1_dyn decoder",
        num_output_temporal_dims=None,
    )
    quantizer = getattr(actor_module, "quantizer", None)
    levels = getattr(quantizer, "_levels", None)
    if (
        not isinstance(levels, torch.Tensor)
        or tuple(levels.tolist()) != (TELEOP_FSQ_LEVEL,) * TELEOP_TOKEN_WIDTH
        or getattr(quantizer, "num_codebooks", None) != 1
        or getattr(quantizer, "codebook_dim", None) != TELEOP_TOKEN_WIDTH
        or getattr(quantizer, "effective_codebook_dim", None) != TELEOP_TOKEN_WIDTH
        or getattr(quantizer, "dim", None) != TELEOP_TOKEN_WIDTH
        or getattr(quantizer, "has_projections", None) is not False
        or getattr(quantizer, "scale", None) is not None
        or getattr(quantizer, "channel_first", None) is not False
        or getattr(quantizer, "preserve_symmetry", None) is not False
        or getattr(quantizer, "noise_dropout", None) != 0.0
        or getattr(quantizer, "bound_hard_clamp", None) is not False
        or getattr(quantizer, "orthogonal_rotation", None) is not False
        or getattr(quantizer, "keep_num_codebooks_dim", None) is not False
        or getattr(quantizer, "return_indices", None) is not True
        or getattr(quantizer, "force_quantization_f32", None) is not True
        or getattr(actor_module, "max_num_tokens", None) != TELEOP_TOKEN_COUNT
    ):
        raise ValueError("true23 teleop quantizer does not match exact FSQ contract")

    probes = torch.linspace(
        -2.0,
        2.0,
        TELEOP_TOKEN_COUNT * TELEOP_TOKEN_WIDTH,
        device=levels.device,
        dtype=torch.float32,
    ).reshape(1, TELEOP_TOKEN_COUNT, TELEOP_TOKEN_WIDTH)
    half_l = (TELEOP_FSQ_LEVEL - 1) * (1.0 + 1.0e-3) / 2.0
    shift = math.atanh(0.5 / half_l)
    expected = torch.round(torch.tanh(probes + shift) * half_l - 0.5) / (TELEOP_FSQ_LEVEL // 2)
    with torch.no_grad():
        actual, _ = quantizer(probes)
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        raise ValueError("live true23 FSQ behavior does not match export formula")


def _contract_descriptor(reference_profile: str) -> dict[str, Any]:
    shape = decoder_shape(DEPLOYMENT_HISTORY_LENGTH, OBS_LAYOUT_PADDED_IL29, TOKEN_DIM)
    simulation_config = _sim_validation_config()
    decoder_dims = _decoder_layer_dims(reference_profile)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "robot_model": ROBOT_MODEL,
        "required_mode_machine": REQUIRED_MODE_MACHINE,
        "observation_layout": OBS_LAYOUT_PADDED_IL29,
        "history_length": DEPLOYMENT_HISTORY_LENGTH,
        "history_order": HISTORY_ORDER,
        "term_order": list(OBSERVATION_TERM_ORDER),
        "token_dim": TOKEN_DIM,
        "proprioception_dim": shape.proprioception_dim,
        "decoder_input_dim": shape.input_dim,
        "decoder_output_dim": TARGET_DOF,
        "decoder_output_layout": DECODER_OUTPUT_LAYOUT,
        "reference_profile": reference_profile,
        "reference_contract": reference_profile_contract(reference_profile),
        "source_il29_keep_indices": list(SOURCE_IL29_KEEP_INDICES),
        "source_il29_excluded_indices": list(SOURCE_IL29_EXCLUDED_INDICES),
        "source_il29_joint_names": list(SOURCE_IL29_JOINT_NAMES),
        "native_il23_to_canonical_il29": list(NATIVE_IL23_TO_CANONICAL_IL29),
        "source_mj29_to_native_il23": list(SOURCE_MJ29_TO_TARGET_IL23),
        "missing_fill": dict(MISSING_OBSERVATION_FILL),
        "hardware_joint_ids": list(HARDWARE_JOINT_IDS),
        "excluded_hardware_joint_ids": list(EXCLUDED_HARDWARE_JOINT_IDS),
        "native_il23_joint_names": list(NATIVE_IL23_JOINT_NAMES),
        "canonical_compact_il23_joint_names": list(CANONICAL_COMPACT_IL23_JOINT_NAMES),
        "hardware_joint_names": list(HARDWARE_23_JOINT_NAMES),
        "hardware_action_scale": list(HARDWARE_23_ACTION_SCALE),
        "native_il23_action_scale": list(NATIVE_IL23_ACTION_SCALE),
        "isaaclab_to_mujoco_dof": list(ISAACLAB_TO_MUJOCO_DOF),
        "mujoco_to_isaaclab_dof": list(MUJOCO_TO_ISAACLAB_DOF),
        "teleop_encoder": {
            "input_dim": TELEOP_ENCODER_INPUT_DIM,
            "input_term_order": list(TELEOP_ENCODER_INPUT_TERM_ORDER),
            "input_term_dims": list(TELEOP_ENCODER_INPUT_TERM_DIMS),
            "layer_dims": list(_ENCODER_LAYER_DIMS),
            "linear_indices": list(_ENCODER_LINEAR_INDICES),
            "activation": _MLP_ACTIVATION,
            "token_count": TELEOP_TOKEN_COUNT,
            "token_width": TELEOP_TOKEN_WIDTH,
            "output_dim": TOKEN_DIM,
            "fsq_level": TELEOP_FSQ_LEVEL,
            "fsq_formula": _FSQ_FORMULA,
            "config_sha256": sha256_file(_ENCODER_CONFIG_PATH),
        },
        "decoder": {
            "layer_dims": list(decoder_dims),
            "linear_indices": list(
                range(0, 2 * (len(decoder_dims) - 1), 2)
            ),
            "activation": _MLP_ACTIVATION,
            "config_sha256": sha256_file(
                _decoder_config_path(reference_profile)
            ),
        },
        "policy_config_sha256": sha256_file(
            _policy_config_path(reference_profile)
        ),
        "sim_validation": {
            "config_sha256": sha256_file(_SIM_VALIDATION_CONFIG_PATH),
            "control_hz": simulation_config["control_hz"],
            "producer": dict(simulation_config["producer"]),
            "minimum_coverage": dict(simulation_config["minimum_coverage"]),
            "disturbance_envelope": dict(simulation_config["disturbance_envelope"]),
            "scenarios": dict(simulation_config["scenarios"]),
        },
        "encoder_onnx": {
            "opset": ONNX_OPSET_VERSION,
            "input_name": ENCODER_ONNX_INPUT_NAME,
            "input_shape": [1, TELEOP_ENCODER_INPUT_DIM],
            "input_dtype": "float32",
            "output_name": ENCODER_ONNX_OUTPUT_NAME,
            "output_shape": [1, TOKEN_DIM],
            "output_dtype": "float32",
            "dynamic_axes": False,
        },
        "decoder_onnx": {
            "opset": ONNX_OPSET_VERSION,
            "input_name": ONNX_INPUT_NAME,
            "input_shape": [1, DEPLOYMENT_DECODER_INPUT_DIM],
            "input_dtype": "float32",
            "output_name": ONNX_OUTPUT_NAME,
            "output_shape": [1, TARGET_DOF],
            "output_dtype": "float32",
            "dynamic_axes": False,
        },
    }


def _tensor_shape(value_info: Any) -> tuple[int, ...]:
    dimensions = value_info.type.tensor_type.shape.dim
    result = []
    for dimension in dimensions:
        if not dimension.HasField("dim_value") or dimension.dim_param:
            raise ValueError(f"dynamic or unknown ONNX dimension: {dimension}")
        result.append(int(dimension.dim_value))
    return tuple(result)


def _validate_static_onnx_structure(
    model: Any,
    *,
    input_name: str,
    input_shape: tuple[int, ...],
    output_name: str,
    output_shape: tuple[int, ...],
    artifact_role: str,
) -> None:
    import onnx

    onnx.checker.check_model(model, full_check=True)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError(f"{artifact_role} ONNX must have exactly one input and one output")
    model_input = model.graph.input[0]
    model_output = model.graph.output[0]
    if model_input.name != input_name or model_output.name != output_name:
        raise ValueError(f"{artifact_role} ONNX tensor names must be {input_name} and {output_name}")
    if model_input.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError(f"{artifact_role} ONNX input must be float32")
    if model_output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError(f"{artifact_role} ONNX output must be float32")
    if _tensor_shape(model_input) != input_shape:
        raise ValueError(f"{artifact_role} ONNX input shape must be static {list(input_shape)}")
    if _tensor_shape(model_output) != output_shape:
        raise ValueError(f"{artifact_role} ONNX output shape must be static {list(output_shape)}")
    default_opsets = [entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")]
    if default_opsets != [ONNX_OPSET_VERSION]:
        raise ValueError(f"{artifact_role} ONNX opset must be exactly {ONNX_OPSET_VERSION}")


def validate_onnx_structure(model: Any) -> None:
    """Enforce one static float32 decoder [1,994] -> [1,23] graph."""
    _validate_static_onnx_structure(
        model,
        input_name=ONNX_INPUT_NAME,
        input_shape=(1, DEPLOYMENT_DECODER_INPUT_DIM),
        output_name=ONNX_OUTPUT_NAME,
        output_shape=(1, TARGET_DOF),
        artifact_role="decoder",
    )


def validate_encoder_onnx_structure(model: Any) -> None:
    """Enforce one static float32 teleop encoder [1,267] -> [1,64] graph."""
    _validate_static_onnx_structure(
        model,
        input_name=ENCODER_ONNX_INPUT_NAME,
        input_shape=(1, TELEOP_ENCODER_INPUT_DIM),
        output_name=ENCODER_ONNX_OUTPUT_NAME,
        output_shape=(1, TOKEN_DIM),
        artifact_role="encoder",
    )


def _parity_vectors(input_dim: int):
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260730)
    return (
        torch.zeros(1, input_dim, dtype=torch.float32),
        torch.linspace(
            -1.0,
            1.0,
            input_dim,
            dtype=torch.float32,
        ).reshape(1, -1),
        torch.randn(
            1,
            input_dim,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.1,
    )


def validate_ort_parity(
    module: Any,
    onnx_path: Path,
    *,
    input_name: str = ONNX_INPUT_NAME,
    input_dim: int = DEPLOYMENT_DECODER_INPUT_DIM,
    output_name: str = ONNX_OUTPUT_NAME,
    output_dim: int = TARGET_DOF,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    max_abs_error = 0.0
    max_rel_error = 0.0
    inputs_digest = hashlib.sha256()
    outputs_digest = hashlib.sha256()
    for vector in _parity_vectors(input_dim):
        input_array = vector.numpy()
        inputs_digest.update(input_array.tobytes())
        with torch.no_grad():
            torch_output = module(vector).detach().cpu().numpy()
        ort_output = session.run([output_name], {input_name: input_array})[0]
        if torch_output.shape != (1, output_dim) or ort_output.shape != (1, output_dim):
            raise ValueError(f"PyTorch/ORT parity output shape is not [1,{output_dim}]")
        if not np.isfinite(torch_output).all() or not np.isfinite(ort_output).all():
            raise ValueError("PyTorch/ORT parity produced non-finite output")
        difference = np.abs(torch_output - ort_output)
        relative = difference / np.maximum(np.abs(torch_output), 1.0e-6)
        max_abs_error = max(max_abs_error, float(difference.max(initial=0.0)))
        max_rel_error = max(max_rel_error, float(relative.max(initial=0.0)))
        if not np.allclose(torch_output, ort_output, rtol=PARITY_RTOL, atol=PARITY_ATOL):
            raise ValueError(f"ONNX Runtime parity failed: max_abs={max_abs_error}, max_rel={max_rel_error}")
        outputs_digest.update(ort_output.tobytes())

    return {
        "onnx_checker_full_check": True,
        "shape_inference": True,
        "ort_provider": "CPUExecutionProvider",
        "parity_case_count": 3,
        "parity_atol": PARITY_ATOL,
        "parity_rtol": PARITY_RTOL,
        "parity_max_abs_error": round(max_abs_error, 12),
        "parity_max_rel_error": round(max_rel_error, 12),
        "parity_inputs_sha256": inputs_digest.hexdigest(),
        "parity_outputs_sha256": outputs_digest.hexdigest(),
    }


def _embedded_metadata(
    *,
    artifact_role: str,
    checkpoint_sha256: str,
    policy_state_sha256: str,
    encoder_state_sha256: str,
    decoder_state_sha256: str,
    sim_report_sha256: str,
    sim_report_payload_sha256: str,
    global_step: int,
    reference_profile: str,
    training_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_role": artifact_role,
        "contract": _contract_descriptor(reference_profile),
        "checkpoint_stage": "trained",
        "training_global_step": global_step,
        "training_evidence": dict(training_evidence),
        "checkpoint_sha256": checkpoint_sha256,
        "policy_state_sha256": policy_state_sha256,
        "encoder_state_sha256": encoder_state_sha256,
        "decoder_state_sha256": decoder_state_sha256,
        "sim_validation_computed": True,
        "sim_report_sha256": sim_report_sha256,
        "sim_report_payload_sha256": sim_report_payload_sha256,
    }


def _set_onnx_metadata(model: Any, embedded: Mapping[str, Any]) -> None:
    import onnx

    onnx.helper.set_model_props(model, {ONNX_METADATA_KEY: canonical_json_bytes(embedded).decode()})


def _get_onnx_metadata(model: Any) -> Mapping[str, Any]:
    if len(model.metadata_props) != 1:
        raise ValueError("ONNX metadata must contain exactly one property")
    values = {entry.key: entry.value for entry in model.metadata_props}
    if set(values) != {ONNX_METADATA_KEY}:
        raise ValueError("ONNX metadata must contain exactly the true23 artifact record")
    try:
        embedded = json.loads(
            values[ONNX_METADATA_KEY],
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite ONNX metadata constant: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid embedded ONNX metadata: {exc}") from exc
    if not isinstance(embedded, Mapping):
        raise ValueError("embedded ONNX metadata must be an object")
    return embedded


def _derive_artifact_paths(
    output: Path,
    metadata_path: Path | None,
) -> tuple[Path, Path, Path]:
    output = output.resolve()
    if output.suffix.lower() == ".onnx":
        base = output.with_suffix("")
    elif output.suffix:
        raise ValueError("output must be an extensionless prefix or end in .onnx")
    else:
        base = output
    encoder_path = base.with_name(f"{base.name}_encoder.onnx")
    decoder_path = base.with_name(f"{base.name}_decoder.onnx")
    if metadata_path is None:
        metadata_path = base.with_name(f"{base.name}.metadata.json")
    metadata_path = metadata_path.resolve()
    resolved = {encoder_path.resolve(), decoder_path.resolve(), metadata_path}
    if len(resolved) != 3:
        raise ValueError("encoder, decoder, and metadata outputs must be distinct paths")
    return encoder_path.resolve(), decoder_path.resolve(), metadata_path


def _require_training_sim_material_match(
    training_material: Mapping[str, Any],
    simulation_evidence: Mapping[str, Any],
) -> None:
    """Require simulation to use exact source and robot bytes seen by training."""

    normalized = validate_training_material_evidence(training_material)
    provenance = simulation_evidence.get("material_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("simulation evidence lacks material_provenance")
    runtime_source = provenance.get("runtime_source")
    if runtime_source != normalized["runtime_source"]:
        raise ValueError(
            "simulation runtime source differs from trained checkpoint material"
        )
    runtime_files = {
        record["relpath"]: record
        for record in normalized["runtime_source"]["files"]
    }
    asset_records = [
        runtime_files[relpath]
        for relpath in TRUE23_ROBOT_ASSET_RELPATHS
    ]
    derived_asset_manifest = {
        "schema_version": FILE_MANIFEST_SCHEMA_VERSION,
        "file_count": len(asset_records),
        "total_bytes": sum(record["size_bytes"] for record in asset_records),
        "manifest_sha256": sha256_bytes(canonical_json_bytes(asset_records)),
        "files": asset_records,
    }
    if derived_asset_manifest != normalized["robot_assets"]:
        raise ValueError(
            "simulation robot assets differ from trained checkpoint material"
        )


def export_validated_true23_artifact(
    checkpoint_path: Path,
    simulation_report_path: Path,
    output: Path,
    *,
    metadata_path: Path | None = None,
) -> tuple[Path, Path, Path, Mapping[str, Any]]:
    """Atomically publish a validated teleop encoder + true23 decoder pair."""
    import onnx
    from onnx import shape_inference
    import torch

    checkpoint_path = checkpoint_path.resolve()
    simulation_report_path = simulation_report_path.resolve()
    encoder_path, decoder_path, metadata_path = _derive_artifact_paths(
        output,
        metadata_path,
    )
    final_paths = (encoder_path, decoder_path, metadata_path)
    if any(path.exists() for path in final_paths):
        raise FileExistsError("refusing to overwrite encoder, decoder, or metadata")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = load_safe_true23_checkpoint(checkpoint_path, map_location="cpu")
    global_step = extract_global_step(checkpoint)
    encoder, decoder, policy_state_sha256 = build_true23_policy_pair(checkpoint)
    validate_training_checkpoint_records(
        checkpoint,
        global_step=global_step,
        policy_state_sha256=policy_state_sha256,
    )
    checkpoint_metadata = checkpoint["g1_23dof_metadata"]
    reference_profile = str(checkpoint_metadata["reference_profile"])
    checkpoint_training_evidence = checkpoint[
        "g1_23dof_training_evidence"
    ]
    policy_state = checkpoint["policy_state_dict"]
    encoder_state_sha256 = _policy_state_sha256(
        {key: value for key, value in policy_state.items() if key.startswith(_ENCODER_PREFIX)}
    )
    decoder_state_sha256 = _policy_state_sha256(
        {key: value for key, value in policy_state.items() if key.startswith(_DECODER_PREFIX)}
    )

    report_bytes = simulation_report_path.read_bytes()
    report = load_strict_json(simulation_report_path)
    report_sha256 = sha256_bytes(report_bytes)
    report_payload_sha256 = sha256_bytes(canonical_json_bytes(report))
    simulation_evidence = validate_simulation_report(
        report,
        checkpoint_sha256=checkpoint_sha256,
        report_sha256=report_sha256,
        report_payload_sha256=report_payload_sha256,
        report_path=simulation_report_path,
        reference_profile=reference_profile,
        checkpoint_motion_dataset=checkpoint_training_evidence[
            "motion_dataset"
        ],
    )
    _require_training_sim_material_match(
        checkpoint_training_evidence["training_material"],
        simulation_evidence,
    )

    embedded_by_role = {
        role: _embedded_metadata(
            artifact_role=role,
            checkpoint_sha256=checkpoint_sha256,
            policy_state_sha256=policy_state_sha256,
            encoder_state_sha256=encoder_state_sha256,
            decoder_state_sha256=decoder_state_sha256,
            sim_report_sha256=report_sha256,
            sim_report_payload_sha256=report_payload_sha256,
            global_step=global_step,
            reference_profile=reference_profile,
            training_evidence=checkpoint_training_evidence,
        )
        for role in ("teleop_encoder", "true23_decoder")
    }
    contract_sha256 = sha256_bytes(
        canonical_json_bytes(_contract_descriptor(reference_profile))
    )
    temporary_paths: list[Path] = []
    temporary_metadata: Path | None = None
    published_paths: list[Path] = []

    def allocate_temporary(path: Path, suffix: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=suffix,
        )
        os.close(descriptor)
        result = Path(name)
        temporary_paths.append(result)
        return result

    def export_one(
        *,
        module: Any,
        temporary_path: Path,
        input_name: str,
        input_dim: int,
        output_name: str,
        embedded: Mapping[str, Any],
        structure_validator: Any,
        output_dim: int,
    ) -> dict[str, Any]:
        example = torch.zeros(1, input_dim, dtype=torch.float32)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(
                module,
                example,
                temporary_path,
                export_params=True,
                input_names=[input_name],
                output_names=[output_name],
                opset_version=ONNX_OPSET_VERSION,
                do_constant_folding=True,
                dynamic_axes=None,
                dynamo=False,
            )
        model = onnx.load(temporary_path, load_external_data=False)
        model = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        _set_onnx_metadata(model, embedded)
        onnx.save_model(model, temporary_path, save_as_external_data=False)
        model = onnx.load(temporary_path, load_external_data=False)
        structure_validator(model)
        if _get_onnx_metadata(model) != embedded:
            raise ValueError("embedded ONNX metadata changed during serialization")
        return validate_ort_parity(
            module,
            temporary_path,
            input_name=input_name,
            input_dim=input_dim,
            output_name=output_name,
            output_dim=output_dim,
        )

    try:
        temporary_encoder = allocate_temporary(encoder_path, ".tmp.onnx")
        temporary_decoder = allocate_temporary(decoder_path, ".tmp.onnx")
        encoder_validation = export_one(
            module=encoder,
            temporary_path=temporary_encoder,
            input_name=ENCODER_ONNX_INPUT_NAME,
            input_dim=TELEOP_ENCODER_INPUT_DIM,
            output_name=ENCODER_ONNX_OUTPUT_NAME,
            embedded=embedded_by_role["teleop_encoder"],
            structure_validator=validate_encoder_onnx_structure,
            output_dim=TOKEN_DIM,
        )
        decoder_validation = export_one(
            module=decoder,
            temporary_path=temporary_decoder,
            input_name=ONNX_INPUT_NAME,
            input_dim=DEPLOYMENT_DECODER_INPUT_DIM,
            output_name=ONNX_OUTPUT_NAME,
            embedded=embedded_by_role["true23_decoder"],
            structure_validator=validate_onnx_structure,
            output_dim=TARGET_DOF,
        )

        hashes = {
            "checkpoint_sha256": checkpoint_sha256,
            "policy_state_sha256": policy_state_sha256,
            "encoder_state_sha256": encoder_state_sha256,
            "decoder_state_sha256": decoder_state_sha256,
            "encoder_onnx_sha256": sha256_file(temporary_encoder),
            "decoder_onnx_sha256": sha256_file(temporary_decoder),
            "sim_report_sha256": report_sha256,
            "sim_report_payload_sha256": report_payload_sha256,
            "contract_sha256": contract_sha256,
            "robot_asset_sha256": sha256_file(_ROBOT_ASSET_PATH),
            "robot_config_sha256": sha256_file(_ROBOT_CONFIG_PATH),
            "sim_config_sha256": sha256_file(_SIM_VALIDATION_CONFIG_PATH),
            "encoder_config_sha256": sha256_file(_ENCODER_CONFIG_PATH),
            "decoder_config_sha256": sha256_file(
                _decoder_config_path(reference_profile)
            ),
            "policy_config_sha256": sha256_file(
                _policy_config_path(reference_profile)
            ),
            "encoder_embedded_metadata_sha256": sha256_bytes(
                canonical_json_bytes(embedded_by_role["teleop_encoder"])
            ),
            "decoder_embedded_metadata_sha256": sha256_bytes(
                canonical_json_bytes(embedded_by_role["true23_decoder"])
            ),
        }
        metadata = make_artifact_metadata(
            history_length=DEPLOYMENT_HISTORY_LENGTH,
            observation_layout=OBS_LAYOUT_PADDED_IL29,
            checkpoint_stage="trained",
            reference_profile=reference_profile,
            deployment_ready=True,
            sim_validation_passed=True,
            simulation_evidence=simulation_evidence,
            artifact_hashes=hashes,
        )
        metadata.update(
            {
                "artifact_kind": ARTIFACT_KIND,
                "encoder_onnx_filename": encoder_path.name,
                "decoder_onnx_filename": decoder_path.name,
                "metadata_filename": metadata_path.name,
                "onnx_opset": ONNX_OPSET_VERSION,
                "training_evidence": dict(checkpoint_training_evidence),
                "validation": {
                    "teleop_encoder": encoder_validation,
                    "true23_decoder": decoder_validation,
                    "pair_dry_run": True,
                },
            }
        )
        validate_artifact_contract(
            metadata,
            decoder_input_dim=DEPLOYMENT_DECODER_INPUT_DIM,
            decoder_output_dim=TARGET_DOF,
        )
        metadata["metadata_payload_sha256"] = sha256_bytes(canonical_json_bytes(metadata))
        metadata_bytes = canonical_json_bytes(metadata)

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=metadata_path.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
        )
        temporary_metadata = Path(temporary_name)
        temporary_paths.append(temporary_metadata)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(metadata_bytes)
            stream.flush()
            os.fsync(stream.fileno())

        _verify_validated_true23_artifact(
            temporary_encoder,
            temporary_decoder,
            temporary_metadata,
            expected_filenames=(
                encoder_path.name,
                decoder_path.name,
                metadata_path.name,
            ),
            checkpoint_path=checkpoint_path,
            simulation_report_path=simulation_report_path,
        )
        for temporary, final in (
            (temporary_encoder, encoder_path),
            (temporary_decoder, decoder_path),
            (temporary_metadata, metadata_path),
        ):
            os.replace(temporary, final)
            published_paths.append(final)
    except Exception:
        for path in published_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)

    return encoder_path, decoder_path, metadata_path, metadata


def _verify_validated_true23_artifact(
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
    *,
    expected_filenames: tuple[str, str, str],
    checkpoint_path: Path | None = None,
    simulation_report_path: Path | None = None,
) -> Mapping[str, Any]:
    """Re-verify both ONNX roles, bindings, and a finite chained dry-run."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    encoder_path = encoder_path.resolve()
    decoder_path = decoder_path.resolve()
    metadata_path = metadata_path.resolve()
    if len({encoder_path, decoder_path, metadata_path}) != 3:
        raise ValueError("encoder, decoder, and metadata must be distinct paths")
    metadata = load_strict_json(metadata_path)
    if metadata.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("metadata artifact_kind is not the validated true23 pair")
    actual_filenames = (
        metadata.get("encoder_onnx_filename"),
        metadata.get("decoder_onnx_filename"),
        metadata.get("metadata_filename"),
    )
    if actual_filenames != expected_filenames:
        raise ValueError("sidecar encoder/decoder/metadata filenames do not match artifact paths")

    payload_hash = metadata.get("metadata_payload_sha256")
    if not isinstance(payload_hash, str):
        raise ValueError("metadata_payload_sha256 is missing")
    unhashed_metadata = dict(metadata)
    unhashed_metadata.pop("metadata_payload_sha256")
    if sha256_bytes(canonical_json_bytes(unhashed_metadata)) != payload_hash:
        raise ValueError("metadata sidecar payload hash mismatch")

    validate_artifact_contract(
        metadata,
        decoder_input_dim=DEPLOYMENT_DECODER_INPUT_DIM,
        decoder_output_dim=TARGET_DOF,
    )
    reference_profile = str(metadata["reference_profile"])
    hashes = metadata["hashes"]
    simulation_evidence = metadata["simulation_evidence"]
    _require_training_sim_material_match(
        metadata["training_evidence"]["training_material"],
        simulation_evidence,
    )
    if simulation_evidence["checkpoint_sha256"] != hashes["checkpoint_sha256"]:
        raise ValueError("simulation evidence checkpoint hash disagrees with hashes")
    if simulation_evidence["report_sha256"] != hashes["sim_report_sha256"]:
        raise ValueError("simulation evidence report hash disagrees with hashes")
    if simulation_evidence["report_payload_sha256"] != hashes["sim_report_payload_sha256"]:
        raise ValueError("simulation evidence payload hash disagrees with hashes")
    if sha256_file(encoder_path) != hashes["encoder_onnx_sha256"]:
        raise ValueError("encoder ONNX hash does not match sidecar")
    if sha256_file(decoder_path) != hashes["decoder_onnx_sha256"]:
        raise ValueError("decoder ONNX hash does not match sidecar")
    if checkpoint_path is not None and sha256_file(checkpoint_path.resolve()) != hashes["checkpoint_sha256"]:
        raise ValueError("checkpoint file hash does not match sidecar")
    if simulation_report_path is not None:
        report_path = simulation_report_path.resolve()
        if sha256_file(report_path) != hashes["sim_report_sha256"]:
            raise ValueError("simulation report file hash does not match sidecar")
        report = load_strict_json(report_path)
        if sha256_bytes(canonical_json_bytes(report)) != hashes["sim_report_payload_sha256"]:
            raise ValueError("simulation report payload hash does not match sidecar")
        recomputed_simulation_evidence = validate_simulation_report(
            report,
            checkpoint_sha256=hashes["checkpoint_sha256"],
            report_sha256=hashes["sim_report_sha256"],
            report_payload_sha256=hashes["sim_report_payload_sha256"],
            report_path=report_path,
            reference_profile=reference_profile,
            checkpoint_motion_dataset=metadata["training_evidence"][
                "motion_dataset"
            ],
        )
        if recomputed_simulation_evidence != simulation_evidence:
            raise ValueError("simulation evidence summary does not match raw traces")

    local_hashes = {
        "robot_asset_sha256": sha256_file(_ROBOT_ASSET_PATH),
        "robot_config_sha256": sha256_file(_ROBOT_CONFIG_PATH),
        "sim_config_sha256": sha256_file(_SIM_VALIDATION_CONFIG_PATH),
        "encoder_config_sha256": sha256_file(_ENCODER_CONFIG_PATH),
        "decoder_config_sha256": sha256_file(
            _decoder_config_path(reference_profile)
        ),
        "policy_config_sha256": sha256_file(
            _policy_config_path(reference_profile)
        ),
        "contract_sha256": sha256_bytes(
            canonical_json_bytes(_contract_descriptor(reference_profile))
        ),
    }
    for key, expected in local_hashes.items():
        if hashes.get(key) != expected:
            raise ValueError(f"local true23 contract binding changed: {key}")

    training_evidence = metadata.get("training_evidence")
    if (
        not isinstance(training_evidence, Mapping)
        or isinstance(training_evidence.get("global_step"), bool)
        or not isinstance(training_evidence.get("global_step"), int)
        or training_evidence["global_step"] <= 0
        or training_evidence.get("reference_profile") != reference_profile
    ):
        raise ValueError(
            "sidecar training_evidence must bind positive global_step and reference_profile"
        )
    validate_training_checkpoint_records(
        {
            "g1_23dof_metadata": make_artifact_metadata(
                history_length=DEPLOYMENT_HISTORY_LENGTH,
                observation_layout=OBS_LAYOUT_PADDED_IL29,
                checkpoint_stage="trained",
                reference_profile=reference_profile,
            ),
            "g1_23dof_training_evidence": training_evidence,
        },
        global_step=training_evidence["global_step"],
        policy_state_sha256=hashes["policy_state_sha256"],
    )

    role_specs = (
        (
            "teleop_encoder",
            encoder_path,
            validate_encoder_onnx_structure,
            "encoder_embedded_metadata_sha256",
        ),
        (
            "true23_decoder",
            decoder_path,
            validate_onnx_structure,
            "decoder_embedded_metadata_sha256",
        ),
    )
    for role, model_path, validator, embedded_hash_key in role_specs:
        model = onnx.load(model_path, load_external_data=False)
        validator(model)
        embedded = _get_onnx_metadata(model)
        embedded_sha256 = sha256_bytes(canonical_json_bytes(embedded))
        if embedded_sha256 != hashes[embedded_hash_key]:
            raise ValueError(f"{role} embedded metadata hash mismatch")
        expected_embedded = _embedded_metadata(
            artifact_role=role,
            checkpoint_sha256=hashes["checkpoint_sha256"],
            policy_state_sha256=hashes["policy_state_sha256"],
            encoder_state_sha256=hashes["encoder_state_sha256"],
            decoder_state_sha256=hashes["decoder_state_sha256"],
            sim_report_sha256=hashes["sim_report_sha256"],
            sim_report_payload_sha256=hashes["sim_report_payload_sha256"],
            global_step=training_evidence["global_step"],
            reference_profile=reference_profile,
            training_evidence=training_evidence,
        )
        if embedded != expected_embedded:
            raise ValueError(f"{role} embedded metadata does not match pair contract")

    encoder_session = ort.InferenceSession(
        str(encoder_path),
        providers=["CPUExecutionProvider"],
    )
    decoder_session = ort.InferenceSession(
        str(decoder_path),
        providers=["CPUExecutionProvider"],
    )
    token = encoder_session.run(
        [ENCODER_ONNX_OUTPUT_NAME],
        {
            ENCODER_ONNX_INPUT_NAME: np.zeros(
                (1, TELEOP_ENCODER_INPUT_DIM),
                dtype=np.float32,
            )
        },
    )[0]
    decoder_input = np.zeros((1, DEPLOYMENT_DECODER_INPUT_DIM), dtype=np.float32)
    if token.shape != (1, TOKEN_DIM) or not np.isfinite(token).all():
        raise ValueError("verified encoder dry-run did not produce finite [1,64]")
    decoder_input[:, :TOKEN_DIM] = token
    action = decoder_session.run(
        [ONNX_OUTPUT_NAME],
        {ONNX_INPUT_NAME: decoder_input},
    )[0]
    if action.shape != (1, TARGET_DOF) or not np.isfinite(action).all():
        raise ValueError("verified pair dry-run did not produce finite [1,23]")
    return metadata


def verify_validated_true23_artifact(
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
    *,
    checkpoint_path: Path | None = None,
    simulation_report_path: Path | None = None,
) -> Mapping[str, Any]:
    """Public verifier: filenames must match the three supplied artifact paths."""
    encoder_path = encoder_path.resolve()
    decoder_path = decoder_path.resolve()
    metadata_path = metadata_path.resolve()
    return _verify_validated_true23_artifact(
        encoder_path,
        decoder_path,
        metadata_path,
        expected_filenames=(
            encoder_path.name,
            decoder_path.name,
            metadata_path.name,
        ),
        checkpoint_path=checkpoint_path,
        simulation_report_path=simulation_report_path,
    )
