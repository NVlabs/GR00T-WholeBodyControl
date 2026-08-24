#!/usr/bin/env python3
"""Fail-fast, offline checks for native G1 rev-1.0 23-DoF training."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from gear_sonic.utils.g1_23dof_artifact import (
    approved_motion_dataset_provenance,
    build_training_material_evidence,
    decoder_layer_dims_for_profile,
    inspect_true23_policy_state,
    true23_reference_profile_from_config,
    validate_training_checkpoint_records,
    validate_training_material_evidence,
)
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    checkpoint_stage,
    load_safe_true23_checkpoint,
    promotion_checkpoint_path,
)
from gear_sonic.utils.g1_23dof_contract import (
    APPROVED_WARM_START_RELEASES,
    DECODER_OUTPUT_LAYOUT,
    OBS_LAYOUT_PADDED_IL29,
    REQUIRED_MODE_MACHINE,
    ROBOT_MODEL,
    SOURCE_IL29_EXCLUDED_INDICES,
    TARGET_DOF,
    reference_profile_contract,
    validate_artifact_contract,
)

EXPECTED_ISAACLAB_VERSION = "2.3.2"
EXPECTED_PYTHON = (3, 11)
FULL_TRAINING_MIN_VRAM_GIB = 16.0
EXPERIMENT = "manager/universal_token/all_modes/sonic_g1_23dof_rev_1_0_warm_start"
LOW_LATENCY_EXPERIMENT = (
    "manager/universal_token/all_modes/"
    "sonic_g1_23dof_rev_1_0_low_latency_warm_start"
)
EXPECTED_POLICY_FUNCTIONS = {
    "joint_pos": "gear_sonic.envs.manager_env.mdp:g1_23dof_padded_joint_pos_rel",
    "joint_vel": "gear_sonic.envs.manager_env.mdp:g1_23dof_padded_joint_vel_rel",
    "actions": "gear_sonic.envs.manager_env.mdp:g1_23dof_padded_last_action",
}
REQUIRED_TRAINING_MODULES = (
    "accelerate",
    "easydict",
    "filelock",
    "hydra",
    "joblib",
    "lxml",
    "loguru",
    "open3d",
    "rich",
    "tensordict",
    "transformers",
    "trl",
    "vector_quantize_pytorch",
    "wandb",
    "yaml",
)


@dataclass(frozen=True)
class TrainingPreflightReport:
    """Machine-readable result; this check never starts Isaac Sim."""

    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    details: Mapping[str, Any]


def _nested(config: Any, *keys: str) -> Any:
    value = config
    for key in keys:
        if isinstance(value, Mapping):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
        if value is None:
            return None
    return value


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _policy_term_order(repo_root: Path) -> tuple[str, ...]:
    """Read configclass declaration order without importing Isaac Lab."""
    source_path = repo_root / "gear_sonic/envs/manager_env/mdp/observations.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg":
            return tuple(
                target.id
                for statement in node.body
                if isinstance(statement, ast.Assign)
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
    raise ValueError("PolicyCfg declaration missing")


def _audit_runtime(
    errors: list[str],
    warnings: list[str],
    details: dict[str, Any],
    *,
    smoke: bool,
) -> None:
    details["python_version"] = ".".join(map(str, sys.version_info[:3]))
    details["expected_python"] = ".".join(map(str, EXPECTED_PYTHON))
    if tuple(sys.version_info[:2]) != EXPECTED_PYTHON:
        errors.append(
            "Isaac Lab 2.3.2 + Isaac Sim 5.1 training requires Python "
            f"{EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}; got "
            f"{sys.version_info[0]}.{sys.version_info[1]}"
        )
    spec = importlib.util.find_spec("isaaclab")
    if spec is None:
        errors.append("Isaac Lab is not importable")
    else:
        import isaaclab

        version = str(getattr(isaaclab, "__version__", "unknown"))
        details["isaaclab_version"] = version
        if version != EXPECTED_ISAACLAB_VERSION:
            errors.append(
                f"Isaac Lab must be exactly {EXPECTED_ISAACLAB_VERSION}; got {version}"
            )

    missing_modules = [
        name for name in REQUIRED_TRAINING_MODULES if importlib.util.find_spec(name) is None
    ]
    details["missing_training_modules"] = missing_modules
    if missing_modules:
        errors.append(f"training Python modules missing: {missing_modules}")

    try:
        import torch
    except ImportError:
        errors.append("PyTorch is not importable")
        return
    details["torch_version"] = str(torch.__version__)
    details["cuda_available"] = bool(torch.cuda.is_available())
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable to PyTorch")
        return
    properties = torch.cuda.get_device_properties(0)
    vram_gib = properties.total_memory / 1024**3
    details["cuda_device"] = properties.name
    details["cuda_vram_gib"] = round(vram_gib, 2)
    if vram_gib < FULL_TRAINING_MIN_VRAM_GIB:
        message = (
            f"GPU has {vram_gib:.1f} GiB VRAM; full true23 fine-tuning/validation "
            f"requires at least {FULL_TRAINING_MIN_VRAM_GIB:.0f} GiB"
        )
        if smoke:
            warnings.append(message + "; smoke only, promotion forbidden")
        else:
            errors.append(message)


def _audit_config(
    config: Any,
    repo_root: Path,
    errors: list[str],
    warnings: list[str],
    details: dict[str, Any],
    *,
    smoke: bool,
) -> None:
    robot = _nested(config, "manager_env", "config", "robot")
    expected_robot = {
        "type": ROBOT_MODEL,
        "embodiment.model": ROBOT_MODEL,
        "embodiment.dof": TARGET_DOF,
        "embodiment.required_mode_machine": REQUIRED_MODE_MACHINE,
        "embodiment.observation_layout": OBS_LAYOUT_PADDED_IL29,
        "embodiment.decoder_output_layout": DECODER_OUTPUT_LAYOUT,
        "embodiment.deployment_ready": False,
    }
    for dotted_key, expected in expected_robot.items():
        actual = _nested(robot, *dotted_key.split("."))
        if actual != expected:
            errors.append(f"robot.{dotted_key}: expected {expected!r}, got {actual!r}")

    try:
        reference_profile = true23_reference_profile_from_config(config)
        reference_contract = reference_profile_contract(reference_profile)
        details["reference_profile"] = reference_profile
        details["reference_contract"] = reference_contract
        motion = _nested(config, "manager_env", "commands", "motion")
        if _nested(motion, "num_future_frames") != reference_contract["future_frame_count"]:
            errors.append("motion.num_future_frames differs from reference contract")
        expected_dt = (
            reference_contract["future_frame_step"]
            / reference_contract["source_sample_rate_hz"]
        )
        if _nested(motion, "dt_future_ref_frames") != expected_dt:
            errors.append(
                "motion.dt_future_ref_frames differs from reference contract"
            )
        configured_hidden_dims = _nested(
            config,
            "algo",
            "config",
            "actor",
            "backbone",
            "decoders",
            "g1_dyn",
            "params",
            "module_config_dict",
            "layer_config",
            "hidden_dims",
        )
        expected_hidden_dims = list(
            decoder_layer_dims_for_profile(reference_profile)[1:-1]
        )
        if list(configured_hidden_dims or []) != expected_hidden_dims:
            errors.append(
                "g1_dyn hidden_dims differ from released reference profile"
            )
    except ValueError as exc:
        errors.append(f"reference profile contract failed: {exc}")
        reference_profile = None

    warm_start = _nested(config, "true23_warm_start")
    expected_warm_start_keys = {
        "source_family",
        "source_hf_revision",
        "source_checkpoint_sha256",
        "initialization_only",
    }
    if (
        not isinstance(warm_start, Mapping)
        or set(warm_start) != expected_warm_start_keys
    ):
        errors.append(
            "true23_warm_start must contain exact pinned source provenance"
        )
    else:
        source_sha256 = warm_start["source_checkpoint_sha256"]
        release = APPROVED_WARM_START_RELEASES.get(source_sha256)
        if (
            release is None
            or release["source_family"] != warm_start["source_family"]
            or release["reference_profile"] != reference_profile
            or release["source_revision"] != warm_start["source_hf_revision"]
            or warm_start["initialization_only"] is not True
        ):
            errors.append(
                "true23_warm_start source hash/revision/profile is not an exact "
                "approved release binding"
            )
        else:
            details["warm_start_source_family"] = warm_start["source_family"]
            details["warm_start_source_sha256"] = source_sha256
            details["warm_start_source_revision"] = warm_start[
                "source_hf_revision"
            ]

    for key in (
        "actor_prop_history_length",
        "actor_actions_history_length",
        "critic_prop_history_length",
        "critic_actions_history_length",
    ):
        if _nested(config, key) != 10:
            errors.append(f"{key} must be 10")

    encoders = _nested(config, "algo", "config", "actor", "backbone", "encoders")
    decoders = _nested(config, "algo", "config", "actor", "backbone", "decoders")
    if encoders is None or tuple(encoders) != ("teleop",):
        errors.append("actor must contain only teleop encoder")
    if decoders is None or tuple(decoders) != ("g1_dyn",):
        errors.append("actor must contain only g1_dyn decoder")

    for term, expected_func in EXPECTED_POLICY_FUNCTIONS.items():
        actual = _nested(
            config, "manager_env", "observations", "policy", term, "func"
        )
        if actual != expected_func:
            errors.append(f"policy {term} must use {expected_func}; got {actual!r}")
        noise = _nested(
            config, "manager_env", "observations", "policy", term, "noise"
        )
        if noise is not None:
            errors.append(f"policy {term} may not corrupt fixed/padded joint slots")

    try:
        order = _policy_term_order(repo_root)
        expected_order = (
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
            "gravity_dir",
        )
        actual_order = tuple(name for name in order if name in expected_order)
        if actual_order != expected_order:
            errors.append(
                f"PolicyCfg proprioception order mismatch: {actual_order!r}"
            )
        details["policy_proprioception_order"] = list(actual_order)
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(f"cannot verify PolicyCfg order: {exc}")

    num_envs = _nested(config, "num_envs")
    details["configured_num_envs"] = num_envs
    if smoke and num_envs != 1:
        errors.append(f"smoke mode requires num_envs=1; got {num_envs!r}")
    elif isinstance(num_envs, int) and num_envs > 512:
        warnings.append(
            f"configured num_envs={num_envs}; first run should override num_envs=1"
        )
    if smoke:
        smoke_constraints = {
            "use_wandb": (_nested(config, "use_wandb"), False),
            "manager_env.config.terrain_type": (
                _nested(config, "manager_env", "config", "terrain_type"),
                "plane",
            ),
            "algo.config.num_learning_iterations": (
                _nested(config, "algo", "config", "num_learning_iterations"),
                1,
            ),
        }
        for key, (actual, expected) in smoke_constraints.items():
            if actual != expected:
                errors.append(
                    f"smoke mode requires {key}={expected!r}; got {actual!r}"
                )


def _audit_assets(
    config: Any,
    repo_root: Path,
    errors: list[str],
    details: dict[str, Any],
    *,
    require_motion_data: bool,
) -> None:
    asset_root_value = _nested(
        config, "manager_env", "commands", "motion", "motion_lib_cfg", "asset", "assetRoot"
    )
    asset_file = _nested(
        config,
        "manager_env",
        "commands",
        "motion",
        "motion_lib_cfg",
        "asset",
        "assetFileName",
    )
    urdf_file = _nested(
        config,
        "manager_env",
        "commands",
        "motion",
        "motion_lib_cfg",
        "asset",
        "urdfFileName",
    )
    if not all(isinstance(value, str) and value for value in (asset_root_value, asset_file, urdf_file)):
        errors.append("motion assetRoot/assetFileName/urdfFileName must be set")
        return

    asset_root = _repo_path(repo_root, asset_root_value)
    mjcf_path = asset_root / asset_file
    urdf_path = asset_root / urdf_file
    details["mjcf_path"] = str(mjcf_path)
    details["urdf_path"] = str(urdf_path)
    for label, path in (("MJCF", mjcf_path), ("URDF", urdf_path)):
        if not path.is_file():
            errors.append(f"{label} asset missing: {path}")

    if urdf_path.is_file():
        urdf = ET.parse(urdf_path).getroot()
        revolute = [
            joint
            for joint in urdf.findall("joint")
            if joint.attrib.get("type") in {"revolute", "continuous", "prismatic"}
        ]
        if len(revolute) != TARGET_DOF:
            errors.append(f"URDF must contain 23 movable joints; got {len(revolute)}")
        missing_meshes = [
            mesh.attrib["filename"]
            for mesh in urdf.findall(".//mesh")
            if not (urdf_path.parent / mesh.attrib["filename"]).is_file()
        ]
        if missing_meshes:
            errors.append(f"URDF mesh files missing: {missing_meshes}")

    if mjcf_path.is_file():
        mjcf = ET.parse(mjcf_path).getroot()
        motors = mjcf.findall("./actuator/motor")
        if len(motors) != TARGET_DOF:
            errors.append(f"MJCF must contain 23 motors; got {len(motors)}")

    motion_value = _nested(
        config, "manager_env", "commands", "motion", "motion_lib_cfg", "motion_file"
    )
    if not isinstance(motion_value, str) or not motion_value:
        errors.append("motion_file must be configured")
    else:
        motion_path = _repo_path(repo_root, motion_value)
        details["motion_path"] = str(motion_path)
        if require_motion_data:
            if not motion_path.exists():
                errors.append(f"training motion data missing: {motion_path}")
            elif motion_path.is_dir() and next(motion_path.rglob("*.pkl"), None) is None:
                errors.append(f"training motion directory contains no PKL files: {motion_path}")
            else:
                try:
                    details["motion_dataset"] = (
                        approved_motion_dataset_provenance(
                            config,
                            repo_root=repo_root,
                        )
                    )
                except (OSError, ValueError) as exc:
                    errors.append(
                        "training motion dataset provenance failed: "
                        f"{exc}"
                    )


def _audit_checkpoint(
    config: Any,
    repo_root: Path,
    errors: list[str],
    details: dict[str, Any],
) -> None:
    checkpoint_value = _nested(config, "checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        errors.append("true23 warm-start requires +checkpoint=<initialized true23 .pt>")
        return
    checkpoint_path = _repo_path(repo_root, checkpoint_value)
    details["checkpoint_path"] = str(checkpoint_path)
    if not checkpoint_path.is_file():
        errors.append(f"checkpoint missing: {checkpoint_path}")
        return

    resume_requested = bool(_nested(config, "resume"))
    checkpoint_audit_path = checkpoint_path
    if resume_requested:
        if checkpoint_path.name.endswith(".promotion.pt"):
            errors.append(
                "resume=true requires the trusted full trainer checkpoint, "
                "not its weights-only promotion sidecar"
            )
            return
        checkpoint_audit_path = promotion_checkpoint_path(checkpoint_path)
        details["checkpoint_audit_path"] = str(checkpoint_audit_path)
        if not checkpoint_audit_path.is_file():
            errors.append(
                "trusted full resume requires its safe promotion sidecar for "
                f"preflight audit: {checkpoint_audit_path}"
            )
            return

    try:
        checkpoint = load_safe_true23_checkpoint(
            checkpoint_audit_path,
            map_location="cpu",
            allow_legacy_initialization=True,
        )
        metadata = checkpoint.get("g1_23dof_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("checkpoint lacks g1_23dof_metadata")
        validate_artifact_contract(
            metadata,
            decoder_input_dim=994,
            decoder_output_dim=TARGET_DOF,
            require_deployment_ready=False,
        )
        config_profile = true23_reference_profile_from_config(config)
        if metadata.get("reference_profile") != config_profile:
            raise ValueError(
                "checkpoint reference_profile differs from training config"
            )
        policy_sha256 = inspect_true23_policy_state(checkpoint)
        stage = checkpoint_stage(checkpoint)
        if stage == "checkpoint_initialization":
            initialization_report = checkpoint.get(
                "g1_23dof_initialization_report"
            )
            if not isinstance(initialization_report, Mapping):
                raise ValueError("initialization checkpoint lacks provenance report")
            expected_source_sha256 = _nested(
                config,
                "true23_warm_start",
                "source_checkpoint_sha256",
            )
            expected_source_revision = _nested(
                config,
                "true23_warm_start",
                "source_hf_revision",
            )
            release = APPROVED_WARM_START_RELEASES.get(
                expected_source_sha256
            )
            if (
                release is None
                or initialization_report.get("source_checkpoint_sha256")
                != expected_source_sha256
                or initialization_report.get("source_family")
                != _nested(
                    config,
                    "true23_warm_start",
                    "source_family",
                )
                or initialization_report.get("source_revision")
                != expected_source_revision
                or initialization_report.get("reference_profile")
                != config_profile
                or initialization_report.get(
                    "initial_policy_state_sha256"
                )
                != policy_sha256
                or policy_sha256
                != release["initial_policy_state_sha256"]
            ):
                raise ValueError(
                    "initialization checkpoint source provenance differs from "
                    "training config"
                )
        if resume_requested and stage != "trained":
            raise ValueError("resume promotion sidecar must be a trained checkpoint")
        if stage == "trained":
            evidence = checkpoint.get("g1_23dof_training_evidence")
            checkpoint_state = checkpoint.get("state")
            checkpoint_step = (
                checkpoint_state.get("global_step")
                if isinstance(checkpoint_state, Mapping)
                else getattr(checkpoint_state, "global_step", None)
            )
            if checkpoint_step is None and isinstance(evidence, Mapping):
                checkpoint_step = evidence.get("global_step")
            validate_training_checkpoint_records(
                checkpoint,
                global_step=int(checkpoint_step),
                policy_state_sha256=policy_sha256,
            )
            current_material = details.get("training_material")
            trained_material = validate_training_material_evidence(
                evidence.get("training_material")
                if isinstance(evidence, Mapping)
                else None,
                context="checkpoint training material",
            )
            if isinstance(current_material, Mapping):
                for material_key in (
                    "material_config_sha256",
                    "runtime_source",
                    "robot_assets",
                ):
                    if trained_material[material_key] != current_material[
                        material_key
                    ]:
                        raise ValueError(
                            "checkpoint training material differs from "
                            f"current preflight: {material_key}"
                        )
        details["checkpoint_stage"] = stage
        details["policy_state_sha256"] = policy_sha256
        details["checkpoint_output_dof"] = TARGET_DOF
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"checkpoint contract failed: {exc}")


def audit_true23_training(
    config: Any,
    *,
    repo_root: Path,
    require_runtime: bool = True,
    require_motion_data: bool = True,
    smoke: bool = False,
) -> TrainingPreflightReport:
    """Audit config/assets/checkpoint without launching simulator or robot."""
    repo_root = repo_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {
        "repo_root": str(repo_root),
        "expected_isaaclab_version": EXPECTED_ISAACLAB_VERSION,
        "action_dof": TARGET_DOF,
        "missing_observation_slots": list(SOURCE_IL29_EXCLUDED_INDICES),
        "runtime_started": False,
        "robot_contacted": False,
        "training_mode": "smoke" if smoke else "full",
        "promotion_allowed": False,
    }
    if require_runtime:
        _audit_runtime(errors, warnings, details, smoke=smoke)
    _audit_config(config, repo_root, errors, warnings, details, smoke=smoke)
    try:
        details["training_material"] = build_training_material_evidence(
            config,
            repo_root=repo_root,
            require_approved_material=not smoke,
        )
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"training material provenance failed: {exc}")
    _audit_assets(
        config,
        repo_root,
        errors,
        details,
        require_motion_data=require_motion_data,
    )
    _audit_checkpoint(config, repo_root, errors, details)
    return TrainingPreflightReport(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        details=details,
    )


def require_true23_training_ready(
    config: Any,
    *,
    repo_root: Path,
    require_runtime: bool = True,
    require_motion_data: bool = True,
    smoke: bool = False,
) -> TrainingPreflightReport:
    """Raise one actionable error before AppLauncher allocates simulator state."""
    report = audit_true23_training(
        config,
        repo_root=repo_root,
        require_runtime=require_runtime,
        require_motion_data=require_motion_data,
        smoke=smoke,
    )
    if not report.ok:
        raise RuntimeError("G1 23-DoF training preflight failed:\n- " + "\n- ".join(report.errors))
    return report


def _compose_config(
    repo_root: Path,
    checkpoint: str,
    *,
    smoke: bool = False,
    motion_file: str | None = None,
    experiment: str = EXPERIMENT,
):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    config_dir = str((repo_root / "gear_sonic/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        overrides = [f"+exp={experiment}", f"+checkpoint={checkpoint}"]
        if smoke:
            overrides.extend(
                [
                    "+true23_smoke=true",
                    "num_envs=1",
                    "use_wandb=false",
                    "manager_env.config.terrain_type=plane",
                    "algo.config.num_learning_iterations=1",
                ]
            )
        if motion_file is not None:
            overrides.append(
                "++manager_env.commands.motion.motion_lib_cfg.motion_file="
                + motion_file
            )
        config = compose(
            config_name="base",
            overrides=overrides,
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(config)
    return OmegaConf.masked_copy(
        config,
        [key for key in config if key != "hydra"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate true23 training inputs without launching Isaac Sim"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--skip-motion-data", action="store_true")
    parser.add_argument(
        "--low-latency",
        action="store_true",
        help="compose genuine 0.00..0.18 s low-latency true23 profile",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "validate one-env smoke only; defaults motion input to "
            "sample_data/robot_filtered and never permits promotion"
        ),
    )
    parser.add_argument(
        "--motion-file",
        help="override motion PKL file/directory (smoke default: sample_data/robot_filtered)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    motion_file = args.motion_file
    if args.smoke and motion_file is None:
        motion_file = "sample_data/robot_filtered"
    checkpoint = args.checkpoint or (
        "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"
        if args.low_latency
        else "sonic_release/g1_23dof_rev_1_0_init.pt"
    )
    config = _compose_config(
        args.repo_root,
        checkpoint,
        smoke=args.smoke,
        motion_file=motion_file,
        experiment=LOW_LATENCY_EXPERIMENT if args.low_latency else EXPERIMENT,
    )
    report = audit_true23_training(
        config,
        repo_root=args.repo_root,
        require_runtime=not args.skip_runtime,
        require_motion_data=not args.skip_motion_data,
        smoke=args.smoke,
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))  # noqa: T201
    else:
        print("PASS" if report.ok else "FAIL")  # noqa: T201
        for warning in report.warnings:
            print(f"WARN: {warning}")  # noqa: T201
        for error in report.errors:
            print(f"ERROR: {error}")  # noqa: T201
    raise SystemExit(0 if report.ok else 2)


if __name__ == "__main__":
    main()
