"""Run bounded, simulator-only Stage-1 selected-21204 ankle adaptation.

This launcher intentionally exposes no learning-rate, seed, resume, row-selection,
upload, or hardware controls.  It constructs the exact causal DadDance task, lets
RSL-RL build a fresh 256-input critic, then delegates the only checkpoint load and
optimizer replacement to ``configure_selected_v2_ankle_rsl_runner``.  Checkpoints
are warm restarts only: adapted weights survive, Adam state and iteration do not.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import sys
from typing import Any, Literal, Mapping, Sequence

from gear_sonic.envs.mjlab.native124_selected_v2_ankle_task import (
    audit_native124_selected_v2_ankle_task_env_cfg,
    native124_selected_v2_ankle_task_contract,
)
from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
    DAD_DANCE_RELATIVE_PATH,
    DAD_DANCE_SHA256,
    validate_dad_dance_motion_file,
)
from gear_sonic.trl.mjlab.native124_selected_v2_ankle_runner import (
    ANKLE_HARDWARE_ROWS,
    SELECTED_ACTOR_STATE_SHA256,
    SELECTED_CHECKPOINT_SHA256,
)
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes, sha256_file
from gear_sonic.utils.g1_23dof_mjlab_training import build_file_manifest
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    CHECKPOINT_ITERATION,
    ONNX_SHA256,
    load_checkpoint21204_binding,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
UNITREE_ROOT = REPO_ROOT / "external_dependencies" / "unitree_rl_mjlab"
MJLAB_ROOT = REPO_ROOT / "external_dependencies" / "mjlab"

FIXED_SEED = 20260805
FIXED_GPU = 0
FIXED_LEARNING_RATE = 1.0e-5
ACTOR_OBSERVATION_DIM = 124
CRITIC_OBSERVATION_DIM = 256
ACTION_DIM = 23
HIDDEN_DIMS = (512, 256, 128)

SMOKE_NUM_ENVS = 4
SMOKE_ITERATIONS = 2
SMOKE_STEPS_PER_ENV = 8
TRAIN_DEFAULT_NUM_ENVS = 64
TRAIN_MAX_NUM_ENVS = 128
TRAIN_DEFAULT_ITERATIONS = 100
TRAIN_MAX_ITERATIONS = 500
TRAIN_STEPS_PER_ENV = 16
TRAIN_CHECKPOINT_INTERVAL = 5

RSL_DISTRIBUTION_NAME = "rsl-rl-lib"
RSL_EXPECTED_VERSION = "5.0.1"
RSL_EXPECTED_DIST_INFO_DIRNAME = "rsl_rl_lib-5.0.1.dist-info"
RSL_EXPECTED_TREE_FILE_COUNT = 30
RSL_EXPECTED_TREE_TOTAL_BYTES = 185_057
RSL_EXPECTED_TREE_MANIFEST_SHA256 = "088055d971470ad218cf8d2d2d4b1b0a9c3d6a91c9b3410df689cb13bc588737"
RSL_RUNTIME_MODULE_FILES = {
    "rsl_rl": "__init__.py",
    "rsl_rl.runners.on_policy_runner": "runners/on_policy_runner.py",
    "rsl_rl.algorithms.ppo": "algorithms/ppo.py",
    "rsl_rl.models.mlp_model": "models/mlp_model.py",
    "rsl_rl.modules.distribution": "modules/distribution.py",
    "rsl_rl.modules.normalization": "modules/normalization.py",
    "rsl_rl.storage.rollout_storage": "storage/rollout_storage.py",
    "rsl_rl.utils.logger": "utils/logger.py",
}
RSL_EXPECTED_FILE_SHA256 = {
    "rsl_rl/__init__.py": "9b4c420fef3ae39683544833b6d2efbb26aecfc0583ec59be7f4eeb748fecf66",
    "rsl_rl/algorithms/__init__.py": ("e370271e689bada5cdf2b4623e0d16d75e8439864a2ae913b22215f7b2985843"),
    "rsl_rl/algorithms/distillation.py": ("c9a326d1c9d0f57a2c14fbea23b8b0f92d3b96bb7a38224e090e07f13e1fb7e0"),
    "rsl_rl/algorithms/ppo.py": "a2d35e7ad7b884c80b7434e7d2ce785a6da1e18d93c96f1179fbd3a208669f8c",
    "rsl_rl/env/__init__.py": "33bfc9562580377c2ce6388c37b0df2b8e992c22ce46923443681cbbf1960d60",
    "rsl_rl/env/vec_env.py": "4a72df2fbf9b3e162e0e4b10e0b473511f02ad8c3ebaab0b7993ddfd63a9fa43",
    "rsl_rl/extensions/__init__.py": ("595fd79b13fa6c64cce1f20adb7440660a4c5e5b78749db4794554a782a1ccfe"),
    "rsl_rl/extensions/rnd.py": "e8f25627c04f4cf478cbbf1ec4c362276e80566a1222c75ba55f7b8e0f8471af",
    "rsl_rl/extensions/symmetry.py": ("eb83c3f77e4cc3f3b3b18f7a592429761b7152c8ac1160c32e02562dc0987481"),
    "rsl_rl/models/__init__.py": "6444fc1df8bc94d40ab1911f0e8e6b3f777459986e7f7944d25b7e5ca9b3e206",
    "rsl_rl/models/cnn_model.py": ("818422d1d7a93cd8174a3c34fec8c425f2adf9d81c9623f2b335c2934a60276e"),
    "rsl_rl/models/mlp_model.py": "6219ebb3ed4df036dae7ff1c30d0fbb169eaaa659e44a659dd757a292124476b",
    "rsl_rl/models/rnn_model.py": ("80c1d6452ef021e4d7b174231ed7abac1ed0db8de9604561f8060148d4333ccc"),
    "rsl_rl/modules/__init__.py": "56df7ca2ff95c53f17451fd98720c48871a1679e9d6e3ab0c0961e4339908054",
    "rsl_rl/modules/cnn.py": "4f5bab5f192c42180125f32a4af350108dd1c247e1e84c81f6cf4ea560037ccf",
    "rsl_rl/modules/distribution.py": ("4631eae1939dcd6b065d79c3c0128a88ba2c6126d08f46944c8795682a208a1b"),
    "rsl_rl/modules/mlp.py": "bad934b364b26cd47b6d1612e00ace107a425ae4d272c0cd0c82cbfc57bbcdbc",
    "rsl_rl/modules/normalization.py": ("539e1193cebcc472585cd3df100946fb8d788c4aa0f9f5cbcb3387a64d312926"),
    "rsl_rl/modules/rnn.py": "07f0eec06c7f2638edbdbc3ba25f56122a7941cb0543138f4a6b04675b2a97e6",
    "rsl_rl/runners/__init__.py": "26f4130a403414e13041b62a848d04da3fb0166757603950ea758b151d4b5aaa",
    "rsl_rl/runners/distillation_runner.py": ("7445cbaf3e1fa241d742447ab17d941a659c92b6f8365ef22c5d818923d9e275"),
    "rsl_rl/runners/on_policy_runner.py": ("39358a0a60219fdb360df7acddf80f6a382837aa58e2ec0ccf875df4afee4825"),
    "rsl_rl/storage/__init__.py": "7c9d96df3cad10aa27f34e77f7105a29f74e2d7dd10467b2919f99867a69298e",
    "rsl_rl/storage/rollout_storage.py": ("ececf28e7412d24867066829cc443acae811f034f8e63975b90343947779918d"),
    "rsl_rl/utils/__init__.py": "129856dea5f7ccff56aee74c8590b6cd2201183131bc836c666d7d1a4880e5c4",
    "rsl_rl/utils/logger.py": "d9822d8317724507a7c817e85399db0db9e56a4eae959435138e75cf92e44b8d",
    "rsl_rl/utils/neptune_utils.py": ("a3446faf8d655b75feb215ff4df8598bc91ff5aab446a6349e9ced31bf3cc495"),
    "rsl_rl/utils/utils.py": "4d094c17e54ab09a0d37b14711157f638f0bd929d0f1b49a3f89b4c8fc67caaa",
    "rsl_rl/utils/wandb_utils.py": ("0fdc97e968bf9c47fed3c1beeacaab7d65769c5342b221f9a20ea2e64b2d407d"),
    "distribution/METADATA": "ed8870a13e6f7caf92dd13115919a1896c6411d0cca0b7833b1e60cfa6684f41",
}

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "g1_true23" / "native124_selected_v2_ankle_stage1"
DEFAULT_SMOKE_RUN = DEFAULT_OUTPUT_ROOT / f"seed{FIXED_SEED}_smoke"
DEFAULT_TRAIN_RUN = DEFAULT_OUTPUT_ROOT / f"seed{FIXED_SEED}_train"
DEFAULT_MOTION = REPO_ROOT / DAD_DANCE_RELATIVE_PATH

_LOCAL_SOURCE_PATHS = (
    "gear_sonic/envs/mjlab/native124_selected_v2_ankle_adaptation.py",
    "gear_sonic/envs/mjlab/native124_selected_v2_ankle_task.py",
    "gear_sonic/envs/mjlab/native124_selected_v2_causal_adaptation.py",
    "gear_sonic/envs/mjlab/sonic_true23.py",
    "gear_sonic/envs/mjlab/sonic_true23_causal_history.py",
    "gear_sonic/envs/mjlab/sonic_true23_causal_history_safe_target_v11.py",
    "gear_sonic/envs/mjlab/sonic_true23_low_latency_recovery.py",
    "gear_sonic/trl/mjlab/native124_selected_v2_ankle_rsl.py",
    "gear_sonic/trl/mjlab/native124_selected_v2_ankle_runner.py",
    "gear_sonic/utils/g1_23dof_contract.py",
    "gear_sonic/utils/g1_23dof_native124_21204_adapter.py",
    "gear_sonic/utils/g1_23dof_safe_target_transform.py",
    "gear_sonic/scripts/train_g1_true23_native124_selected_v2_ankle.py",
)
_UNITREE_SOURCE_PATHS = (
    "src/assets/robots/unitree_g1/g1_23dof_constants.py",
    "src/tasks/tracking/config/g1_23dof/env_cfgs.py",
    "src/tasks/tracking/tracking_env_cfg.py",
    "src/tasks/tracking/mdp/commands.py",
    "src/tasks/tracking/mdp/observations.py",
    "src/tasks/tracking/mdp/rewards.py",
    "src/tasks/tracking/mdp/terminations.py",
)
_MJLAB_SOURCE_PATHS = (
    "src/mjlab/rl/config.py",
    "src/mjlab/rl/runner.py",
    "src/mjlab/rl/vecenv_wrapper.py",
)


@dataclass(frozen=True)
class Stage1LaunchPlan:
    """Fully resolved finite launch request."""

    mode: Literal["smoke", "train"]
    motion_file: Path
    run_dir: Path
    num_envs: int
    iterations: int

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "train"}:
            raise ValueError("Stage-1 mode must be smoke or train")
        if not isinstance(self.motion_file, Path) or not isinstance(self.run_dir, Path):
            raise TypeError("motion_file and run_dir must be Path values")
        if type(self.num_envs) is not int or type(self.iterations) is not int:
            raise TypeError("num_envs and iterations must be exact integers")
        if self.mode == "smoke":
            if self.num_envs != SMOKE_NUM_ENVS or self.iterations != SMOKE_ITERATIONS:
                raise ValueError("smoke plan dimensions are fixed")
        elif not (1 <= self.num_envs <= TRAIN_MAX_NUM_ENVS):
            raise ValueError(f"train num_envs must be in [1,{TRAIN_MAX_NUM_ENVS}]")
        elif not (TRAIN_CHECKPOINT_INTERVAL <= self.iterations <= TRAIN_MAX_ITERATIONS):
            raise ValueError(f"train iterations must be in [{TRAIN_CHECKPOINT_INTERVAL},{TRAIN_MAX_ITERATIONS}]")
        elif self.iterations % TRAIN_CHECKPOINT_INTERVAL != 0:
            raise ValueError(f"train iterations must be a multiple of {TRAIN_CHECKPOINT_INTERVAL}")

    @property
    def steps_per_env(self) -> int:
        return SMOKE_STEPS_PER_ENV if self.mode == "smoke" else TRAIN_STEPS_PER_ENV

    @property
    def checkpoint_interval(self) -> int:
        return self.iterations if self.mode == "smoke" else TRAIN_CHECKPOINT_INTERVAL


def expected_stage1_checkpoint_filenames(plan: Stage1LaunchPlan) -> tuple[str, ...]:
    """Return exact ordered warm-checkpoint publication contract."""

    if type(plan) is not Stage1LaunchPlan:
        raise TypeError("plan must be exact Stage1LaunchPlan")
    periodic = tuple(f"model_{iteration}.pt" for iteration in range(0, plan.iterations, plan.checkpoint_interval))
    final = f"model_{plan.iterations - 1}.pt"
    if final in periodic:
        raise RuntimeError("Stage-1 final checkpoint collides with periodic cadence")
    return ("initial_model.pt", *periodic, final)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _with_content_hash(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if field_name in value:
        raise ValueError(f"hash field already exists: {field_name}")
    result = copy.deepcopy(dict(value))
    result[field_name] = _canonical_sha256(result)
    return result


def _validate_content_hash(value: Mapping[str, Any], field_name: str) -> str:
    copied = copy.deepcopy(dict(value))
    claimed = copied.pop(field_name, None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(copied):
        raise ValueError(f"{field_name} mismatch")
    return claimed


def _validate_rsl_runtime_paths(
    *,
    version: str,
    package_root: Path,
    distribution_metadata_root: Path,
    module_files: Mapping[str, Path],
) -> dict[str, Any]:
    """Validate exact installed RSL source files against frozen 5.0.1 bytes."""

    if version != RSL_EXPECTED_VERSION:
        raise ValueError(f"RSL version mismatch: expected {RSL_EXPECTED_VERSION}, got {version}")
    root = package_root.resolve(strict=True)
    if not root.is_dir() or root.name != "rsl_rl":
        raise ValueError("RSL package root must be exact rsl_rl directory")
    metadata_root = distribution_metadata_root.resolve(strict=True)
    if (
        not metadata_root.is_dir()
        or metadata_root.name != RSL_EXPECTED_DIST_INFO_DIRNAME
        or metadata_root.parent != root.parent
    ):
        raise ValueError("RSL distribution metadata path drift")
    if set(module_files) != set(RSL_RUNTIME_MODULE_FILES):
        raise ValueError("RSL executed-module set drift")

    for module_name, relative in RSL_RUNTIME_MODULE_FILES.items():
        expected = (root / relative).resolve(strict=True)
        actual = Path(module_files[module_name]).resolve(strict=True)
        if actual != expected or actual.suffix != ".py" or "__pycache__" in actual.parts:
            raise ValueError(f"RSL module path drift: {module_name}")
    tree = sorted(path for path in root.rglob("*.py") if path.is_file() and "__pycache__" not in path.parts)
    files = {f"rsl_rl/{path.relative_to(root).as_posix()}": path.resolve(strict=True) for path in tree}
    metadata_file = (metadata_root / "METADATA").resolve(strict=True)
    files["distribution/METADATA"] = metadata_file

    actual_hashes = {logical: sha256_file(path) for logical, path in files.items()}
    if actual_hashes != RSL_EXPECTED_FILE_SHA256:
        changed = sorted(
            logical
            for logical in set(actual_hashes) | set(RSL_EXPECTED_FILE_SHA256)
            if actual_hashes.get(logical) != RSL_EXPECTED_FILE_SHA256.get(logical)
        )
        raise ValueError(f"RSL installed source SHA drift: {changed}")
    manifest = build_file_manifest(files, kind="source_files")
    if (
        manifest["file_count"] != RSL_EXPECTED_TREE_FILE_COUNT
        or manifest["total_bytes"] != RSL_EXPECTED_TREE_TOTAL_BYTES
        or manifest["manifest_sha256"] != RSL_EXPECTED_TREE_MANIFEST_SHA256
    ):
        raise ValueError("RSL full-tree manifest drift")
    value = {
        "schema": "g1_true23_native124_selected_v2_rsl_runtime_v1",
        "distribution_name": RSL_DISTRIBUTION_NAME,
        "version": version,
        "package_root": str(root),
        "distribution_metadata_root": str(metadata_root),
        "executed_source_manifest": manifest,
        "expected_file_sha256": dict(RSL_EXPECTED_FILE_SHA256),
    }
    return _with_content_hash(value, "runtime_binding_sha256")


def resolve_rsl_runtime_binding() -> dict[str, Any]:
    """Resolve imports and package metadata used by actual RSL training."""

    distribution = importlib_metadata.distribution(RSL_DISTRIBUTION_NAME)
    version = distribution.version
    metadata_root_value = getattr(distribution, "_path", None)
    if metadata_root_value is None:
        raise RuntimeError("RSL distribution metadata root is unavailable")
    modules = {name: importlib.import_module(name) for name in RSL_RUNTIME_MODULE_FILES}
    package_file = getattr(modules["rsl_rl"], "__file__", None)
    if package_file is None:
        raise RuntimeError("RSL package lacks a concrete __file__")
    module_files: dict[str, Path] = {}
    for name, module in modules.items():
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError(f"RSL module lacks concrete __file__: {name}")
        module_files[name] = Path(module_file)
    return _validate_rsl_runtime_paths(
        version=version,
        package_root=Path(package_file).resolve().parent,
        distribution_metadata_root=Path(metadata_root_value),
        module_files=module_files,
    )


def _validate_supplied_rsl_runtime_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(dict(value))
    _validate_content_hash(copied, "runtime_binding_sha256")
    manifest = copied.get("executed_source_manifest")
    if (
        copied.get("schema") != "g1_true23_native124_selected_v2_rsl_runtime_v1"
        or copied.get("distribution_name") != RSL_DISTRIBUTION_NAME
        or copied.get("version") != RSL_EXPECTED_VERSION
        or copied.get("expected_file_sha256") != RSL_EXPECTED_FILE_SHA256
        or not isinstance(manifest, Mapping)
        or manifest.get("file_count") != RSL_EXPECTED_TREE_FILE_COUNT
        or manifest.get("total_bytes") != RSL_EXPECTED_TREE_TOTAL_BYTES
        or manifest.get("manifest_sha256") != RSL_EXPECTED_TREE_MANIFEST_SHA256
    ):
        raise ValueError("RSL supplied runtime binding contract drift")
    return copied


def stage1_agent_config(plan: Stage1LaunchPlan) -> dict[str, Any]:
    """Return exact RSL config consumed before mandatory ankle configuration."""

    if type(plan) is not Stage1LaunchPlan:
        raise TypeError("plan must be exact Stage1LaunchPlan")
    if plan.mode == "smoke":
        epochs = 1
        mini_batches = 1
    else:
        epochs = 3
        mini_batches = 4
    return {
        "seed": FIXED_SEED,
        "num_steps_per_env": plan.steps_per_env,
        "max_iterations": plan.iterations,
        # RSL preserves configured container type.  Exact lists are required by
        # MLPModel.obs_groups and the hash-locked integration validator.
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        # Smoke retains its original model_0/final-only behavior.  Train emits
        # five-iteration selection checkpoints plus the distinct final model.
        "save_interval": plan.checkpoint_interval,
        "experiment_name": "g1_true23_native124_selected_v2_ankle_stage1",
        "run_name": f"{plan.mode}_seed{FIXED_SEED}",
        "logger": "tensorboard",
        "wandb_project": "disabled",
        "wandb_tags": (),
        "resume": False,
        "load_run": "forbidden",
        "load_checkpoint": "forbidden",
        "clip_actions": None,
        "upload_model": False,
        "check_for_nan": True,
        "class_name": "OnPolicyRunner",
        "actor": {
            "hidden_dims": HIDDEN_DIMS,
            "activation": "elu",
            "obs_normalization": True,
            "cnn_cfg": None,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
            "class_name": "MLPModel",
        },
        "critic": {
            "hidden_dims": HIDDEN_DIMS,
            "activation": "elu",
            "obs_normalization": True,
            "cnn_cfg": None,
            "distribution_cfg": None,
            "class_name": "MLPModel",
        },
        "algorithm": {
            "num_learning_epochs": epochs,
            "num_mini_batches": mini_batches,
            "learning_rate": FIXED_LEARNING_RATE,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.0,
            "desired_kl": None,
            "max_grad_norm": 0.5,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "normalize_advantage_per_mini_batch": False,
            "optimizer": "adam",
            "share_cnn_encoders": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "class_name": "PPO",
        },
    }


def expected_stage1_optimizer_steps(plan: Stage1LaunchPlan) -> int:
    cfg = stage1_agent_config(plan)
    algorithm = cfg["algorithm"]
    return plan.iterations * int(algorithm["num_learning_epochs"]) * int(algorithm["num_mini_batches"])


def _optimizer_step_audit(plan: Stage1LaunchPlan, integration: Any) -> dict[str, int]:
    expected = expected_stage1_optimizer_steps(plan)
    prior = getattr(integration, "prior_completed_optimizer_steps", None)
    session = getattr(integration, "optimizer_steps", None)
    total = getattr(integration, "completed_optimizer_steps_total", None)
    if prior != 0 or session != expected or total != expected:
        raise RuntimeError(
            "Stage-1 optimizer-step count drift: "
            f"expected={expected}, prior={prior}, session={session}, total={total}"
        )
    return {
        "expected_session_optimizer_steps": expected,
        "prior_completed_optimizer_steps": prior,
        "actual_session_optimizer_steps": session,
        "actual_completed_optimizer_steps_total": total,
    }


def _best_effort_stop_logger(runner: Any) -> None:
    logger = getattr(runner, "logger", None)
    if logger is None or getattr(logger, "writer", None) is None:
        return
    stop = getattr(logger, "stop_logging_writer", None)
    if not callable(stop):
        return
    try:
        stop()
    except Exception:
        # Never mask original rollout/update failure.
        pass


def resolved_training_config(
    plan: Stage1LaunchPlan,
    rsl_runtime_binding: Mapping[str, Any],
) -> dict[str, Any]:
    agent = stage1_agent_config(plan)
    task_contract = native124_selected_v2_ankle_task_contract()
    runtime = _validate_supplied_rsl_runtime_binding(rsl_runtime_binding)
    value = {
        "schema": "g1_true23_native124_selected_v2_ankle_stage1_training_v1",
        "mode": plan.mode,
        "seed": FIXED_SEED,
        "gpu": FIXED_GPU,
        "motion_file": str(plan.motion_file.resolve(strict=False)),
        "motion_sha256": DAD_DANCE_SHA256,
        "run_dir": str(plan.run_dir.resolve(strict=False)),
        "num_envs": plan.num_envs,
        "planned_outer_ppo_iterations": plan.iterations,
        "random_initial_episode_length": False,
        "agent": agent,
        "rsl_runtime": runtime,
        "model": {
            "actor_observation_dim": ACTOR_OBSERVATION_DIM,
            "critic_observation_dim": CRITIC_OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dims": list(HIDDEN_DIMS),
            "activation": "elu",
            "source_checkpoint_iteration": CHECKPOINT_ITERATION,
            "source_checkpoint_sha256": SELECTED_CHECKPOINT_SHA256,
            "source_actor_state_sha256": SELECTED_ACTOR_STATE_SHA256,
            "source_onnx_sha256": ONNX_SHA256,
            "selected_load": {
                "actor": True,
                "critic": False,
                "iteration": False,
                "optimizer": False,
                "rnd": False,
            },
            "trainable_actor_hardware_rows": list(ANKLE_HARDWARE_ROWS),
            "actor_normalizer_trainable": False,
            "actor_trunk_trainable": False,
            "actor_std_trainable": False,
            "frozen_output_rows_stochastic": False,
            "critic_initialization": "fresh_privileged_256",
            "optimizer_initialization": "fresh_adam_iteration_0",
            "expected_optimizer_steps": expected_stage1_optimizer_steps(plan),
        },
        "task": task_contract,
        "checkpointing": {
            "kind": "weights_only_warm_restart",
            "optimizer_saved": False,
            "iteration_saved_for_resume": False,
            "stock_load_forbidden": True,
            "overwrite": False,
            "save_interval_outer_ppo_iterations": plan.checkpoint_interval,
            "periodic_checkpoint_iterations": list(range(0, plan.iterations, plan.checkpoint_interval)),
            "expected_checkpoint_files": list(expected_stage1_checkpoint_filenames(plan)),
            "initial_checkpoint": "initial_model.pt",
            "first_updated_checkpoint": "model_0.pt",
            "final_checkpoint": f"model_{plan.iterations - 1}.pt",
        },
        "safety": {
            "simulator_only": True,
            "hardware_authorized": False,
            "robot_network_commands": False,
            "external_network_calls": False,
            "upload_model": False,
            "domain_randomization": False,
            "push_disturbances": False,
            "deployment_ready": False,
        },
    }
    return _with_content_hash(value, "resolved_config_sha256")


def _source_files() -> dict[str, Path]:
    files = {path: REPO_ROOT / path for path in _LOCAL_SOURCE_PATHS}
    files.update({f"unitree_rl_mjlab/{path}": UNITREE_ROOT / path for path in _UNITREE_SOURCE_PATHS})
    files.update({f"mjlab/{path}": MJLAB_ROOT / path for path in _MJLAB_SOURCE_PATHS})
    return files


def _robot_asset_files() -> dict[str, Path]:
    asset_root = UNITREE_ROOT / "src" / "assets" / "robots" / "unitree_g1"
    return {
        f"unitree_g1/{path.relative_to(asset_root).as_posix()}": path
        for path in sorted(asset_root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    }


def build_stage1_material_manifest(
    plan: Stage1LaunchPlan,
    resolved: Mapping[str, Any],
    rsl_runtime_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash source, robot assets, motion, and frozen selected artifacts."""

    resolved_sha256 = _validate_content_hash(resolved, "resolved_config_sha256")
    runtime = _validate_supplied_rsl_runtime_binding(rsl_runtime_binding)
    if resolved.get("rsl_runtime") != runtime:
        raise ValueError("resolved RSL runtime differs from material runtime")
    binding = load_checkpoint21204_binding(REPO_ROOT)
    source = build_file_manifest(_source_files(), kind="source_files")
    robot_assets = build_file_manifest(_robot_asset_files(), kind="robot_assets")
    inputs = build_file_manifest(
        {
            "motions/B_DadDance.npz": plan.motion_file,
            "selected/tracker_manifest.json": binding.manifest_path,
            "selected/selection.json": binding.selection_path,
            "selected/export_report.json": binding.export_report_path,
            "selected/model21204_alpha25.pt": binding.checkpoint_path,
            "selected/policy.onnx": binding.onnx_path,
            "selected/resolved_env_evidence.json": binding.resolved_env_evidence_path,
        },
        kind="motion_dataset",
    )
    value = {
        "schema": "g1_true23_native124_selected_v2_ankle_stage1_materials_v1",
        "resolved_config_sha256": resolved_sha256,
        "source_files": source,
        "robot_assets": robot_assets,
        "bound_inputs": inputs,
        "rsl_runtime": runtime,
    }
    return _with_content_hash(value, "material_manifest_sha256")


def _require_unchanged_materials(
    plan: Stage1LaunchPlan,
    resolved: Mapping[str, Any],
    expected: Mapping[str, Any],
    context: str,
) -> None:
    runtime = resolve_rsl_runtime_binding()
    if resolved_training_config(plan, runtime) != resolved:
        raise RuntimeError(f"Stage-1 resolved runtime changed {context}")
    if build_stage1_material_manifest(plan, resolved, runtime) != expected:
        raise RuntimeError(f"Stage-1 materials changed {context}")


def preflight(
    plan: Stage1LaunchPlan,
    *,
    rsl_runtime_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate all immutable inputs without constructing or stepping a simulator."""

    problems: list[str] = []
    motion_path = plan.motion_file.resolve(strict=False)
    run_dir = plan.run_dir.resolve(strict=False)
    resolved: dict[str, Any] | None = None
    materials: dict[str, Any] | None = None
    binding_report: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    try:
        validated_motion = validate_dad_dance_motion_file(motion_path)
        if sha256_file(validated_motion) != DAD_DANCE_SHA256:
            raise RuntimeError("DadDance hash changed after validation")
    except Exception as error:
        problems.append(f"motion binding failed: {error}")
    try:
        binding = load_checkpoint21204_binding(REPO_ROOT)
        binding_report = {
            "checkpoint_path": str(binding.checkpoint_path),
            "checkpoint_sha256": binding.checkpoint_sha256,
            "actor_state_sha256": binding.actor_state_sha256,
            "iteration": binding.iteration,
            "onnx_path": str(binding.onnx_path),
            "onnx_sha256": binding.onnx_sha256,
        }
    except Exception as error:
        problems.append(f"selected source binding failed: {error}")
    if os.path.lexists(run_dir):
        problems.append(f"run directory already exists; overwrite forbidden: {run_dir}")
    try:
        runtime = (
            resolve_rsl_runtime_binding()
            if rsl_runtime_binding is None
            else _validate_supplied_rsl_runtime_binding(rsl_runtime_binding)
        )
        resolved = resolved_training_config(plan, runtime)
        materials = build_stage1_material_manifest(plan, resolved, runtime)
    except Exception as error:
        problems.append(f"resolved material binding failed: {error}")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        problems.append("distributed training is unsupported; WORLD_SIZE must equal 1")
    return {
        "schema": "g1_true23_native124_selected_v2_ankle_stage1_preflight_v1",
        "ready": not problems,
        "problems": problems,
        "source": binding_report,
        "motion": {
            "path": str(motion_path),
            "sha256": DAD_DANCE_SHA256,
        },
        "run_dir": str(run_dir),
        "resolved_training": resolved,
        "material_manifest": materials,
        "rsl_runtime": runtime,
        "safety": {
            "simulator_constructed": False,
            "simulator_steps": 0,
            "training_updates": 0,
            "hardware_authorized": False,
            "network_used": False,
        },
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            # Partial exclusive output remains evidence of failed run; never
            # remove or overwrite it automatically.
            pass
        raise


def _create_run_dir_exclusive(path: Path) -> Path:
    output = path.expanduser().resolve(strict=False)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite Stage-1 run directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _validated_stage1_checkpoint_paths(
    run_dir: Path,
    plan: Stage1LaunchPlan,
) -> tuple[Path, ...]:
    """Require every planned checkpoint and reject unplanned model files."""

    expected_names = expected_stage1_checkpoint_filenames(plan)
    expected_paths = tuple(run_dir / name for name in expected_names)
    missing = [path.name for path in expected_paths if not path.is_file()]
    expected_model_names = set(expected_names[1:])
    actual_model_names = {path.name for path in run_dir.glob("model_*.pt")}
    unexpected = sorted(actual_model_names - expected_model_names)
    omitted = sorted(expected_model_names - actual_model_names)
    if missing or omitted or unexpected:
        raise RuntimeError(
            "Stage-1 checkpoint publication drift: "
            f"missing={sorted(set(missing) | set(omitted))}, unexpected={unexpected}"
        )
    return expected_paths


def _bind_local_runtime_sources() -> dict[str, str]:
    """Bind task and MJLab imports to source trees included in manifest."""

    unitree_root = UNITREE_ROOT.resolve(strict=True)
    unitree_init = (unitree_root / "src" / "__init__.py").resolve(strict=True)
    root_text = str(unitree_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    existing_src = sys.modules.get("src")
    if existing_src is not None:
        existing_file = getattr(existing_src, "__file__", None)
        if existing_file is None or Path(existing_file).resolve() != unitree_init:
            raise RuntimeError("unbound src package was imported before Stage-1 runtime binding")
    src_module = importlib.import_module("src")
    if getattr(src_module, "__file__", None) is None or Path(src_module.__file__).resolve() != unitree_init:
        raise RuntimeError("Stage-1 task package source binding mismatch")

    mjlab_module = importlib.import_module("mjlab")
    expected_mjlab = (MJLAB_ROOT / "src" / "mjlab" / "__init__.py").resolve(strict=True)
    if Path(mjlab_module.__file__).resolve() != expected_mjlab:
        raise RuntimeError("Stage-1 MJLab source binding mismatch")
    return {
        "unitree_task_init": str(unitree_init),
        "mjlab_init": str(expected_mjlab),
    }


def run_training(plan: Stage1LaunchPlan) -> Path:
    """Construct simulator and execute one finite Stage-1 PPO session."""

    audit = preflight(plan)
    if not audit["ready"]:
        raise RuntimeError("Stage-1 preflight failed:\n" + json.dumps(audit, indent=2, sort_keys=True))
    if audit["resolved_training"] is None or audit["material_manifest"] is None:
        raise RuntimeError("Stage-1 preflight omitted bound materials")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("Stage-1 launcher forbids distributed training")

    # Fixed local-only runtime.  Set before first CUDA operation.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FIXED_GPU)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"

    runtime_sources = _bind_local_runtime_sources()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.utils.torch import configure_torch_backends
    import torch

    from gear_sonic.envs.mjlab.native124_selected_v2_ankle_task import (
        make_native124_selected_v2_ankle_task_env_cfg,
    )
    from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
        Native124SelectedV2CausalAdaptationWrapper,
        prime_native124_selected_v2_causal_adaptation_environment,
    )
    from gear_sonic.trl.mjlab.native124_selected_v2_ankle_rsl import (
        configure_selected_v2_ankle_rsl_runner,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Stage-1 launcher requires exactly fixed visible CUDA device 0")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    device = "cuda:0"

    run_dir = _create_run_dir_exclusive(plan.run_dir)
    resolved = audit["resolved_training"]
    materials = audit["material_manifest"]
    _write_json_exclusive(run_dir / "resolved_training.json", resolved)
    _write_json_exclusive(run_dir / "material_manifest.json", materials)

    env_cfg = make_native124_selected_v2_ankle_task_env_cfg(
        motion_file=str(plan.motion_file.resolve(strict=True)),
        num_envs=plan.num_envs,
        play=False,
    )
    env_cfg.seed = FIXED_SEED
    task_audit = audit_native124_selected_v2_ankle_task_env_cfg(env_cfg)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    runner: Any | None = None
    try:
        wrapped = Native124SelectedV2CausalAdaptationWrapper(env, clip_actions=None)
        prime = prime_native124_selected_v2_causal_adaptation_environment(wrapped)
        _write_json_exclusive(run_dir / "environment_prime.json", prime)
        agent_cfg = stage1_agent_config(plan)
        # Decouple fresh critic initialization from simulator reset RNG draws.
        torch.manual_seed(FIXED_SEED)
        torch.cuda.manual_seed_all(FIXED_SEED)
        runner = MjlabOnPolicyRunner(wrapped, copy.deepcopy(agent_cfg), str(run_dir), device)
        integration = configure_selected_v2_ankle_rsl_runner(
            runner,
            repository_root=REPO_ROOT,
        )
        integration.assert_frozen_invariants()
        if (
            runner.current_learning_iteration != 0
            or integration.prior_completed_optimizer_steps != 0
            or integration.optimizer_steps != 0
            or integration.completed_optimizer_steps_total != 0
        ):
            raise RuntimeError("Stage-1 runner did not begin at fresh iteration/optimizer step zero")
        initial_publication = integration.save_warm_restart(run_dir / "initial_model.pt")
        initial = {
            "schema": "g1_true23_native124_selected_v2_ankle_stage1_initialization_v1",
            "task_audit": task_audit,
            "source": integration.lineage.metadata(),
            "parity": integration.parity,
            "fresh_critic_state_sha256": integration.fresh_critic_state_sha256,
            "fresh_adam_state_entries": len(runner.alg.optimizer.state),
            "current_learning_iteration": runner.current_learning_iteration,
            "initial_warm_checkpoint": {
                "filename": initial_publication.path.name,
                "sha256": initial_publication.sha256,
            },
            "trainable_hardware_rows": list(integration.config.trainable_hardware_rows),
            "clip_actions": wrapped.clip_actions,
            "fixed_learning_rate": runner.alg.learning_rate,
            "fixed_schedule": runner.alg.schedule,
            "runtime_sources": runtime_sources,
            "hardware_authorized": False,
        }
        _write_json_exclusive(run_dir / "initialization.json", initial)
        _require_unchanged_materials(plan, resolved, materials, "before first PPO rollout")
        runner.learn(num_learning_iterations=plan.iterations, init_at_random_ep_len=False)
        integration.assert_frozen_invariants()
        _require_unchanged_materials(plan, resolved, materials, "during PPO session")
        expected_iteration = plan.iterations - 1
        if runner.current_learning_iteration != expected_iteration:
            raise RuntimeError("Stage-1 runner outer iteration counter drift")
        optimizer_steps = _optimizer_step_audit(plan, integration)
        checkpoints = _validated_stage1_checkpoint_paths(run_dir, plan)
        final = {
            "schema": "g1_true23_native124_selected_v2_ankle_stage1_completion_v1",
            "completed_outer_ppo_iterations": plan.iterations,
            "optimizer_steps": optimizer_steps,
            "checkpoint_files": [{"filename": path.name, "sha256": sha256_file(path)} for path in checkpoints],
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        _write_json_exclusive(run_dir / "completion.json", final)
    except BaseException:
        _best_effort_stop_logger(runner)
        raise
    finally:
        env.close()
    return run_dir


def _bounded_train_num_envs(value: str) -> int:
    result = int(value)
    if not 1 <= result <= TRAIN_MAX_NUM_ENVS:
        raise argparse.ArgumentTypeError(f"must be in [1,{TRAIN_MAX_NUM_ENVS}]")
    return result


def _bounded_train_iterations(value: str) -> int:
    result = int(value)
    if not TRAIN_CHECKPOINT_INTERVAL <= result <= TRAIN_MAX_ITERATIONS:
        raise argparse.ArgumentTypeError(f"must be in [{TRAIN_CHECKPOINT_INTERVAL},{TRAIN_MAX_ITERATIONS}]")
    if result % TRAIN_CHECKPOINT_INTERVAL != 0:
        raise argparse.ArgumentTypeError(f"must be a multiple of {TRAIN_CHECKPOINT_INTERVAL}")
    return result


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--run-dir", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    _add_paths(preflight_parser)
    preflight_parser.add_argument("--profile", choices=("smoke", "train"), default="smoke")
    smoke_parser = subparsers.add_parser("smoke")
    _add_paths(smoke_parser)
    train_parser = subparsers.add_parser("train")
    _add_paths(train_parser)
    train_parser.add_argument("--num-envs", type=_bounded_train_num_envs, default=TRAIN_DEFAULT_NUM_ENVS)
    train_parser.add_argument(
        "--iterations",
        type=_bounded_train_iterations,
        default=TRAIN_DEFAULT_ITERATIONS,
    )
    return parser


def _plan_from_args(args: argparse.Namespace) -> Stage1LaunchPlan:
    mode = args.profile if args.command == "preflight" else args.command
    if mode == "smoke":
        num_envs = SMOKE_NUM_ENVS
        iterations = SMOKE_ITERATIONS
        default_run = DEFAULT_SMOKE_RUN
    elif mode == "train":
        num_envs = getattr(args, "num_envs", TRAIN_DEFAULT_NUM_ENVS)
        iterations = getattr(args, "iterations", TRAIN_DEFAULT_ITERATIONS)
        default_run = DEFAULT_TRAIN_RUN
    else:  # pragma: no cover - argparse and Stage1LaunchPlan both guard this.
        raise ValueError(f"unsupported Stage-1 mode: {mode}")
    return Stage1LaunchPlan(
        mode=mode,
        motion_file=args.motion_file.expanduser(),
        run_dir=(default_run if args.run_dir is None else args.run_dir.expanduser()),
        num_envs=num_envs,
        iterations=iterations,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _plan_from_args(args)
    if args.command == "preflight":
        report = preflight(plan)
        print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201
        return 0 if report["ready"] else 2
    output = run_training(plan)
    print(output)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
