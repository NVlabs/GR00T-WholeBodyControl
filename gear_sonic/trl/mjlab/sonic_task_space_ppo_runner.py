"""Bounded genuine-SONIC task-space PPO pilot.

This module is additive and simulator-only.  It overlays the immutable v2
final affine on the verified causal model-250 actor, ignores every source
critic/optimizer/counter, freezes the encoder and early decoder, and trains
only decoder modules 14 and 16 with a fresh critic and Adam optimizer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import MethodType
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.native124_selected_v2_ankle_task import (
    ALIVE_WEIGHT,
    EE_TERMINATION_BODY_NAMES,
    NON_TIMEOUT_TERMINATION_WEIGHT,
)
from gear_sonic.envs.mjlab.sonic_true23 import (
    native_actions_to_hardware_targets,
)
from gear_sonic.envs.mjlab.sonic_true23_student_qualification import (
    audit_sonic_true23_student_qualification_env_cfg,
    capture_student_action_chain,
    make_sonic_true23_student_qualification_env_cfg,
)
from gear_sonic.trl.mjlab.causal_history_runner import (
    CausalHistoryMjlabOnPolicyRunner,
)
from gear_sonic.utils.g1_23dof_artifact import (
    inspect_true23_policy_state,
    sha256_file,
)
from gear_sonic.utils.g1_23dof_mjlab_training import (
    validate_mjlab_training_checkpoint,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
    safe_target_transform_torch,
)

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_sonic_task_space_ppo_contract_v1"
CHECKPOINT_KIND = "g1_true23_sonic_task_space_ppo_checkpoint_v1"
CONTRACT_RELATIVE_PATH = Path("gear_sonic/config/sim_validation/g1_true23_sonic_task_space_ppo_v1.json")
CONTRACT_SHA256 = "a42d661e68e583e9f7e750e7fd8211ba36747172b8ff9be031414b2fe0f03044"

REFERENCE_PROFILE = "released_low_latency_step1_0p02s"
ACTION_DIM = 23
ENCODER_LAYER_COUNT = 5
DECODER_LAYER_COUNT = 9
EXPLORATION_STD = 0.10
NUM_ENVS = 128
NUM_STEPS_PER_ENV = 16
MAXIMUM_UPDATES = 25
MAXIMUM_TRAINING_TRANSITIONS = 51_200
TRANSITIONS_PER_UPDATE = NUM_ENVS * NUM_STEPS_PER_ENV
CHECKPOINT_UPDATES = (0, 5, 10, 25)
EVALUATION_UPDATES = CHECKPOINT_UPDATES
FIXED_LEARNING_RATE = 1.0e-5
PPO_LEARNING_EPOCHS = 2
PPO_MINI_BATCHES = 4
OPTIMIZER_STEPS_PER_UPDATE = PPO_LEARNING_EPOCHS * PPO_MINI_BATCHES
MAXIMUM_OPTIMIZER_STEPS = MAXIMUM_UPDATES * OPTIMIZER_STEPS_PER_UPDATE

TRAINABLE_ACTOR_PARAMETERS = (
    "core.actor_module.decoders.g1_dyn.module.14.bias",
    "core.actor_module.decoders.g1_dyn.module.14.weight",
    "core.actor_module.decoders.g1_dyn.module.16.bias",
    "core.actor_module.decoders.g1_dyn.module.16.weight",
)
STD_PARAMETER = "distribution.std_param"
DECODER_PARAMETER_PREFIX = "core.actor_module.decoders.g1_dyn.module."
ENCODER_PARAMETER_PREFIX = "core.actor_module.encoders.teleop.module."
EXPECTED_TRAINABLE_PARAMETER_COUNT = 274_455
EXPECTED_CRITIC_PARAMETER_TENSOR_COUNT = 8
EXPECTED_OPTIMIZER_PARAMETER_TENSOR_COUNT = (
    len(TRAINABLE_ACTOR_PARAMETERS) + EXPECTED_CRITIC_PARAMETER_TENSOR_COUNT
)
OPTIMIZER_PARAMETER_SHAPES = (
    (512,),
    (512, 512),
    (23,),
    (23, 512),
    (512, 256),
    (512,),
    (256, 512),
    (256,),
    (128, 256),
    (128,),
    (1, 128),
    (1,),
)
INITIAL_FROZEN_ACTOR_STATE_SHA256 = "14ed9fb20dc61e0847b6ef71e2bed3a8ce52ac8c2da5eeb876da8fd4c0896fae"
INITIAL_TRAINABLE_ACTOR_STATE_SHA256 = "000d5ea094fe13a184aecafa8fc00868164591ba3de7ad5af36333656e415c67"

WORST_EE_WEIGHT = -20.0
WORST_EE_NORMALIZATION_M = 0.25
RIGHT_WRIST_BODY_NAME = "right_wrist_roll_rubber_hand"
RIGHT_WRIST_BARRIER_WEIGHT = -25.0
RIGHT_WRIST_BARRIER_ONSET_M = 0.15
RIGHT_WRIST_TERMINATION_M = 0.25
RIGHT_WRIST_BARRIER_DENOMINATOR_FLOOR = 0.05
GATE_PASSING_TERMINATION_NAMES = ("time_out", "anchor_pos", "anchor_ori", "ee_body_pos")

PILOT_RUNTIME_FILENAME = "sonic_task_space_ppo_runtime.json"


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_sha256(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if type(name) is not str or not isinstance(value, torch.Tensor):
            raise TypeError("tensor state must map strings to tensors")
        tensor = value.detach().cpu().contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"tensor state contains nonfinite value: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _value_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _policy_state_subsets(
    policy_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split deployment-namespace policy tensors at the exact trainable seam."""

    trainable_names = {name.removeprefix("core.") for name in TRAINABLE_ACTOR_PARAMETERS}
    if any(name.startswith("core.") for name in trainable_names):
        raise RuntimeError("task-space trainable namespace conversion failed")
    if set(trainable_names) - set(policy_state):
        raise ValueError("task-space policy lacks a trainable deployment tensor")
    if set(policy_state) != {
        "std",
        *(
            f"actor_module.encoders.teleop.module.{module}.{suffix}"
            for module in (0, 2, 4, 6, 8)
            for suffix in ("weight", "bias")
        ),
        *(
            f"actor_module.decoders.g1_dyn.module.{module}.{suffix}"
            for module in (0, 2, 4, 6, 8, 10, 12, 14, 16)
            for suffix in ("weight", "bias")
        ),
    }:
        raise ValueError("task-space policy deployment namespace mismatch")
    trainable = {name: policy_state[name].detach().cpu().contiguous() for name in sorted(trainable_names)}
    frozen = {
        name: value.detach().cpu().contiguous()
        for name, value in policy_state.items()
        if name not in trainable_names
    }
    return frozen, trainable


def _validate_critic_state_schema(state: Mapping[str, torch.Tensor]) -> None:
    specs = {
        "obs_normalizer._mean": ((1, 256), torch.float32),
        "obs_normalizer._var": ((1, 256), torch.float32),
        "obs_normalizer._std": ((1, 256), torch.float32),
        "obs_normalizer.count": ((), torch.int64),
        "mlp.0.weight": ((512, 256), torch.float32),
        "mlp.0.bias": ((512,), torch.float32),
        "mlp.2.weight": ((256, 512), torch.float32),
        "mlp.2.bias": ((256,), torch.float32),
        "mlp.4.weight": ((128, 256), torch.float32),
        "mlp.4.bias": ((128,), torch.float32),
        "mlp.6.weight": ((1, 128), torch.float32),
        "mlp.6.bias": ((1,), torch.float32),
    }
    if set(state) != set(specs):
        raise ValueError("task-space critic state namespace mismatch")
    for name, (shape, dtype) in specs.items():
        value = state[name]
        if (
            type(value) is not torch.Tensor
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
        ):
            raise ValueError(f"task-space critic tensor schema mismatch: {name}")


def _strict_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"nonfinite JSON token {token}")),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} must be strict UTF-8 JSON") from error
    return _mapping(value, context)


def _regular_repository_file(
    root: Path,
    relative: Any,
    expected_sha256: Any,
    context: str,
) -> Path:
    expected = _require_sha256(expected_sha256, f"{context} SHA256")
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise ValueError(f"{context} path must be repository-relative")
    lexical = root / relative
    if lexical.is_symlink():
        raise ValueError(f"{context} may not be a symlink")
    try:
        path = lexical.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{context} is missing") from error
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError(f"{context} must be a repository file")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{context} SHA256 mismatch: expected {expected}, got {actual}")
    return path


def load_task_space_ppo_contract(
    repository_root: str | Path | None = None,
) -> Mapping[str, Any]:
    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    path = root / CONTRACT_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise ValueError("task-space PPO contract is missing or symlinked")
    actual = sha256_file(path)
    if actual != CONTRACT_SHA256:
        raise ValueError(f"task-space PPO contract SHA256 mismatch: expected {CONTRACT_SHA256}, got {actual}")
    contract = _strict_json(path, "task-space PPO contract")
    validate_task_space_ppo_contract(contract)
    return contract


def validate_task_space_ppo_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("role") != "offline_mjlab_genuine_sonic_task_space_ppo_pilot_only"
        or contract.get("seed") != 20260805
    ):
        raise ValueError("task-space PPO contract identity mismatch")

    actor = _mapping(contract.get("actor_initialization"), "actor_initialization")
    expected_actor = {
        "source_checkpoint_sha256": "85bd6de646905a44190dbf32c79737082bb604ab007a90a62e4fd2fdeeee6bd9",
        "source_checkpoint_update_count": 250,
        "source_policy_state_sha256": "c3bfcb5c42929293b62425f155b59ccb731f57c98e8852c7f1e97094525684af",
        "source_lineage_sha256": "08bbd03d0df751328e449d3624d79167587b461d6835fa1f4a8742aad9ffa82a",
        "v2_decoder_sha256": "011740f86483323fc0f1c39ab25b784cf9411b401e56fee8b7a716664e921ee1",
        "encoder_onnx_sha256": "733353148bef1eb8dd83a96416b7a89f0b5c3530ceb9e0cec9c25fdb04f56ff2",
        "overlay_policy_state_sha256": "358310ececeff0177386ae28f60b513a94902465b7e99ac480d40ba21578af61",
    }
    if any(actor.get(key) != value for key, value in expected_actor.items()):
        raise ValueError("task-space actor initialization identity mismatch")
    if any(
        actor.get(name) is not False
        for name in (
            "source_critic_reused",
            "source_optimizer_reused",
            "source_counters_reused",
        )
    ):
        raise ValueError("task-space source may supply actor weights only")

    policy = _mapping(contract.get("policy"), "policy")
    if (
        policy.get("reference_profile") != REFERENCE_PROFILE
        or policy.get("exploration_std") != EXPLORATION_STD
        or policy.get("exploration_std_trainable") is not False
        or policy.get("trainable_actor_parameters") != list(TRAINABLE_ACTOR_PARAMETERS)
        or policy.get("trainable_actor_parameter_count") != EXPECTED_TRAINABLE_PARAMETER_COUNT
        or policy.get("initial_frozen_actor_state_sha256") != INITIAL_FROZEN_ACTOR_STATE_SHA256
        or policy.get("initial_trainable_actor_state_sha256") != INITIAL_TRAINABLE_ACTOR_STATE_SHA256
        or policy.get("all_23_output_rows_trainable") is not True
    ):
        raise ValueError("task-space policy boundary mismatch")

    environment = _mapping(contract.get("environment"), "environment")
    if (
        environment.get("anchor_q9") != 9
        or environment.get("last_action_q9") != 518
        or environment.get("transitions") != 510
        or environment.get("action_input") != "plain_sonic_raw_native23"
        or environment.get("wrapper_action_clip") is not None
        or environment.get("safe_target_transform") != "v2_exactly_once"
        or environment.get("teacher_action_or_label_present") is not False
        or environment.get("domain_randomization") is not False
    ):
        raise ValueError("task-space environment contract mismatch")

    pilot = _mapping(contract.get("ppo_pilot"), "ppo_pilot")
    expected_pilot = {
        "num_envs": NUM_ENVS,
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_training_transitions": MAXIMUM_TRAINING_TRANSITIONS,
        "learning_rate": FIXED_LEARNING_RATE,
        "schedule": "fixed",
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "evaluation_updates": list(EVALUATION_UPDATES),
        "random_episode_length_initialization": False,
        "training_start_update_count": 0,
        "num_learning_epochs": PPO_LEARNING_EPOCHS,
        "num_mini_batches": PPO_MINI_BATCHES,
    }
    if any(pilot.get(key) != value for key, value in expected_pilot.items()):
        raise ValueError("task-space PPO pilot schedule mismatch")
    gates = _mapping(contract.get("pilot_gates"), "pilot_gates")
    if (
        gates.get("baseline_update") != 0
        or gates.get("baseline_expected_completed_transitions") != 155
        or gates.get("baseline_expected_terminal_q9") != 163
        or gates.get("baseline_expected_termination_names") != ["ee_body_pos"]
        or gates.get("update5_episode_length_regression_is_diagnostic_only") is not True
        or gates.get("reward_divergence_relative_tolerance") != 0.2
        or gates.get("reward_divergence_absolute_tolerance") != 50.0
        or gates.get("single_pilot_pass_is_not_support_or_deployment_evidence") is not True
        or gates.get("q163_improvement_required_by_update") != 10
        or gates.get("minimum_completed_transitions_for_q163_improvement") != 156
        or gates.get("stop_before_update25_if_update10_gate_fails") is not True
        or "no_episode_length_regression_from_baseline" in gates
        or gates.get("required_zero_counts")
        != [
            "nonfinite_count",
            "raw_clip_required_count",
            "action_semantics_mismatch_count",
            "q9_discontinuity_count",
        ]
    ):
        raise ValueError("task-space pilot gate contract mismatch")

    rewards = _mapping(contract.get("rewards"), "rewards")
    worst = _mapping(rewards.get("worst_ee_z_normalized_squared"), "worst EE reward")
    wrist = _mapping(rewards.get("right_wrist_prethreshold_barrier"), "wrist barrier")
    if (
        rewards.get("alive_weight") != ALIVE_WEIGHT
        or rewards.get("non_timeout_termination_weight") != NON_TIMEOUT_TERMINATION_WEIGHT
        or tuple(worst.get("body_names", ())) != EE_TERMINATION_BODY_NAMES
        or worst.get("normalization_m") != WORST_EE_NORMALIZATION_M
        or worst.get("weight") != WORST_EE_WEIGHT
        or wrist.get("body_name") != RIGHT_WRIST_BODY_NAME
        or wrist.get("onset_m") != RIGHT_WRIST_BARRIER_ONSET_M
        or wrist.get("termination_m") != RIGHT_WRIST_TERMINATION_M
        or wrist.get("denominator_floor") != RIGHT_WRIST_BARRIER_DENOMINATOR_FLOOR
        or wrist.get("weight") != RIGHT_WRIST_BARRIER_WEIGHT
    ):
        raise ValueError("task-space death-proof reward contract mismatch")

    boundaries = _mapping(contract.get("boundaries"), "boundaries")
    if boundaries.get("simulator_training_only") is not True or any(
        boundaries.get(name) is not False
        for name in (
            "teacher_labels_used",
            "dagger_data",
            "support_qualified",
            "promotion_eligible",
            "deployment_ready",
            "hardware_authorized",
            "robot_or_network_commands_permitted",
        )
    ):
        raise ValueError("task-space permanent boundary mismatch")


def _onnx_initializers(model: Any) -> dict[str, np.ndarray]:
    from onnx import numpy_helper

    values: dict[str, np.ndarray] = {}
    for initializer in model.graph.initializer:
        if initializer.name in values:
            raise ValueError(f"duplicate ONNX initializer: {initializer.name}")
        array = numpy_helper.to_array(initializer)
        if array.dtype != np.float32 or not bool(np.isfinite(array).all()):
            raise ValueError(f"ONNX initializer must be finite float32: {initializer.name}")
        values[initializer.name] = np.asarray(array).copy()
    return values


def build_overlay_policy_state(
    source_checkpoint: Mapping[str, Any],
    encoder_model: Any,
    decoder_model: Any,
    contract: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return model250 actor with v2 module16 and fixed std; reuse nothing else."""

    validate_task_space_ppo_contract(contract)
    actor_contract = _mapping(contract["actor_initialization"], "actor_initialization")
    validated = validate_mjlab_training_checkpoint(source_checkpoint)
    if (
        validated.get("update_count") != actor_contract["source_checkpoint_update_count"]
        or validated.get("policy_state_sha256") != actor_contract["source_policy_state_sha256"]
        or validated.get("lineage_sha256") != actor_contract["source_lineage_sha256"]
    ):
        raise ValueError("model250 checkpoint internal identity mismatch")
    source_state = _mapping(validated.get("policy_state_dict"), "source policy state")
    state = {
        name: value.detach().cpu().to(torch.float32).contiguous().clone()
        for name, value in source_state.items()
        if isinstance(value, torch.Tensor)
    }
    if set(state) != set(source_state) or "std" not in state or len(state) != 29:
        raise ValueError("model250 policy tensor namespace mismatch")

    encoder = _onnx_initializers(encoder_model)
    decoder = _onnx_initializers(decoder_model)
    expected_encoder_names = {
        f"mlp.layers.{layer}.{suffix}" for layer in range(ENCODER_LAYER_COUNT) for suffix in ("weight", "bias")
    }
    expected_decoder_names = {
        f"layers.{layer}.{suffix}" for layer in range(DECODER_LAYER_COUNT) for suffix in ("weight", "bias")
    }
    if set(encoder) != expected_encoder_names or set(decoder) != expected_decoder_names:
        raise ValueError("bound ONNX initializer namespace mismatch")

    for layer in range(ENCODER_LAYER_COUNT):
        module = layer * 2
        for suffix in ("weight", "bias"):
            source_name = f"actor_module.encoders.teleop.module.{module}.{suffix}"
            onnx_name = f"mlp.layers.{layer}.{suffix}"
            if source_name not in state or not np.array_equal(state[source_name].numpy(), encoder[onnx_name]):
                raise ValueError(f"model250 encoder differs from bound ONNX: {onnx_name}")

    for layer in range(DECODER_LAYER_COUNT - 1):
        module = layer * 2
        for suffix in ("weight", "bias"):
            source_name = f"actor_module.decoders.g1_dyn.module.{module}.{suffix}"
            onnx_name = f"layers.{layer}.{suffix}"
            if source_name not in state or not np.array_equal(state[source_name].numpy(), decoder[onnx_name]):
                raise ValueError(f"model250 decoder trunk differs from v2: {onnx_name}")

    final_weight = decoder["layers.8.weight"]
    final_bias = decoder["layers.8.bias"]
    if (
        final_weight.shape != (ACTION_DIM, 512)
        or final_bias.shape != (ACTION_DIM,)
        or _value_sha256(final_weight) != actor_contract["v2_final_weight_value_sha256"]
        or _value_sha256(final_bias) != actor_contract["v2_final_bias_value_sha256"]
    ):
        raise ValueError("v2 final affine value identity mismatch")
    state["actor_module.decoders.g1_dyn.module.16.weight"] = torch.from_numpy(final_weight.copy())
    state["actor_module.decoders.g1_dyn.module.16.bias"] = torch.from_numpy(final_bias.copy())
    state["std"] = torch.full((ACTION_DIM,), EXPLORATION_STD, dtype=torch.float32)

    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": state},
        reference_profile=REFERENCE_PROFILE,
    )
    if policy_hash != actor_contract["overlay_policy_state_sha256"]:
        raise ValueError(
            "model250+v2+std overlay policy hash mismatch: "
            f"expected {actor_contract['overlay_policy_state_sha256']}, got {policy_hash}"
        )
    return state, {
        "source_update_count": validated["update_count"],
        "source_policy_state_sha256": validated["policy_state_sha256"],
        "source_lineage_sha256": validated["lineage_sha256"],
        "overlay_policy_state_sha256": policy_hash,
        "encoder_tensor_match_count": ENCODER_LAYER_COUNT * 2,
        "decoder_trunk_tensor_match_count": (DECODER_LAYER_COUNT - 1) * 2,
        "v2_overlay_tensor_count": 2,
        "exploration_std": EXPLORATION_STD,
        "source_critic_reused": False,
        "source_optimizer_reused": False,
        "source_counters_reused": False,
    }


def load_overlay_policy_state(
    *,
    source_checkpoint_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    contract: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    import onnx

    actor = _mapping(contract["actor_initialization"], "actor_initialization")
    expected_checkpoint_sha = _require_sha256(actor["source_checkpoint_sha256"], "source checkpoint SHA256")
    if source_checkpoint_path.is_symlink() or not source_checkpoint_path.is_file():
        raise ValueError("source checkpoint must be regular non-symlink file")
    actual_checkpoint_sha = sha256_file(source_checkpoint_path)
    if actual_checkpoint_sha != expected_checkpoint_sha:
        raise ValueError("source checkpoint byte SHA256 mismatch")
    try:
        checkpoint = torch.load(
            source_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError("source checkpoint is not weights-only safe") from error
    encoder_model = onnx.load(encoder_path, load_external_data=True)
    decoder_model = onnx.load(decoder_path, load_external_data=True)
    state, report = build_overlay_policy_state(
        checkpoint,
        encoder_model,
        decoder_model,
        contract,
    )
    return state, {
        **report,
        "source_checkpoint_sha256": actual_checkpoint_sha,
        "encoder_onnx_sha256": sha256_file(encoder_path),
        "v2_decoder_sha256": sha256_file(decoder_path),
    }


def preflight_task_space_ppo(
    repository_root: str | Path | None = None,
    *,
    source_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    contract = load_task_space_ppo_contract(root)
    actor = _mapping(contract["actor_initialization"], "actor_initialization")
    prerequisite = _mapping(contract["prerequisite_failure"], "prerequisite_failure")
    environment = _mapping(contract["environment"], "environment")
    paths = {
        "campaign_report": _regular_repository_file(
            root,
            prerequisite["campaign_report_relative_path"],
            prerequisite["campaign_report_sha256"],
            "campaign report",
        ),
        "topology_checkpoint": _regular_repository_file(
            root,
            actor["topology_checkpoint_relative_path"],
            actor["topology_checkpoint_sha256"],
            "topology checkpoint",
        ),
        "encoder": _regular_repository_file(
            root,
            actor["encoder_onnx_relative_path"],
            actor["encoder_onnx_sha256"],
            "encoder ONNX",
        ),
        "v2_manifest": _regular_repository_file(
            root,
            actor["v2_manifest_relative_path"],
            actor["v2_manifest_sha256"],
            "v2 manifest",
        ),
        "v2_decoder": _regular_repository_file(
            root,
            actor["v2_decoder_relative_path"],
            actor["v2_decoder_sha256"],
            "v2 decoder",
        ),
        "motion": _regular_repository_file(
            root,
            environment["motion_relative_path"],
            environment["motion_sha256"],
            "DadDance motion",
        ),
    }
    campaign = _strict_json(paths["campaign_report"], "campaign report")
    runs = campaign.get("runs")
    failed = None
    if isinstance(runs, list):
        failed = next(
            (
                item
                for item in runs
                if isinstance(item, Mapping) and item.get("run_id") == prerequisite["first_failed_run_id"]
            ),
            None,
        )
    if (
        campaign.get("verdict") != prerequisite["required_verdict"]
        or campaign.get("campaign_qualified") is not False
        or campaign.get("published_teacher_label_count") != prerequisite["published_teacher_label_count"]
        or campaign.get("published_training_row_count") != prerequisite["published_training_row_count"]
        or not isinstance(failed, Mapping)
        or failed.get("terminal_transition") != prerequisite["terminal_transition"]
        or failed.get("terminal_q9") != prerequisite["terminal_q9"]
        or failed.get("hard_safety_violation_count") != prerequisite["hard_safety_violation_count"]
        or failed.get("soft_safety_warning_count") != prerequisite["soft_safety_warning_count"]
    ):
        raise ValueError("campaign failure prerequisite identity mismatch")

    source = (
        Path(source_checkpoint_path).expanduser().resolve()
        if source_checkpoint_path is not None
        else Path(actor["source_checkpoint_linux_path"])
    )
    overlay, overlay_report = load_overlay_policy_state(
        source_checkpoint_path=source,
        encoder_path=paths["encoder"],
        decoder_path=paths["v2_decoder"],
        contract=contract,
    )
    del overlay
    return {
        "schema_version": 1,
        "kind": "g1_true23_sonic_task_space_ppo_preflight_v1",
        "ready": True,
        "contract_sha256": CONTRACT_SHA256,
        "campaign_report_sha256": prerequisite["campaign_report_sha256"],
        "source_checkpoint_path": str(source),
        "overlay": overlay_report,
        "pilot": dict(contract["ppo_pilot"]),
        "boundaries": dict(contract["boundaries"]),
    }


def _body_z_errors(
    env: Any,
    command_name: str,
    body_names: Sequence[str],
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    configured = tuple(command.cfg.body_names)
    requested = tuple(body_names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("EE body reward names must be unique and non-empty")
    if any(name not in configured for name in requested):
        raise ValueError("EE body reward references unknown command body")
    indices = [configured.index(name) for name in requested]
    reference = command.body_pos_relative_w[:, indices, -1]
    measured = command.robot_body_pos_w[:, indices, -1]
    if (
        reference.shape != (env.num_envs, len(indices))
        or measured.shape != reference.shape
        or not bool(torch.isfinite(reference).all())
        or not bool(torch.isfinite(measured).all())
    ):
        raise ValueError("EE body z-error tensors are invalid")
    return torch.abs(reference - measured)


def worst_ee_z_normalized_squared(
    env: Any,
    command_name: str = "motion",
    body_names: Sequence[str] = EE_TERMINATION_BODY_NAMES,
    normalization_m: float = WORST_EE_NORMALIZATION_M,
) -> torch.Tensor:
    if not math.isfinite(normalization_m) or normalization_m <= 0.0:
        raise ValueError("worst EE normalization must be finite and positive")
    worst = torch.amax(_body_z_errors(env, command_name, body_names), dim=1)
    return torch.square(worst / normalization_m)


def right_wrist_prethreshold_barrier(
    env: Any,
    command_name: str = "motion",
    body_name: str = RIGHT_WRIST_BODY_NAME,
    onset_m: float = RIGHT_WRIST_BARRIER_ONSET_M,
    termination_m: float = RIGHT_WRIST_TERMINATION_M,
    denominator_floor: float = RIGHT_WRIST_BARRIER_DENOMINATOR_FLOOR,
) -> torch.Tensor:
    if (
        not math.isfinite(onset_m)
        or not math.isfinite(termination_m)
        or not math.isfinite(denominator_floor)
        or not 0.0 < onset_m < termination_m
        or not 0.0 < denominator_floor < 1.0
    ):
        raise ValueError("right-wrist barrier parameters are invalid")
    error = _body_z_errors(env, command_name, (body_name,))[:, 0]
    x = torch.relu((error - onset_m) / (termination_m - onset_m))
    return torch.square(x) / torch.clamp(1.0 - x, min=denominator_floor)


def task_space_reward_contract() -> dict[str, Any]:
    return {
        "alive_weight": ALIVE_WEIGHT,
        "non_timeout_termination_weight": NON_TIMEOUT_TERMINATION_WEIGHT,
        "worst_ee_z_normalized_squared": {
            "function": f"{__name__}:worst_ee_z_normalized_squared",
            "body_names": list(EE_TERMINATION_BODY_NAMES),
            "normalization_m": WORST_EE_NORMALIZATION_M,
            "weight": WORST_EE_WEIGHT,
        },
        "right_wrist_prethreshold_barrier": {
            "function": f"{__name__}:right_wrist_prethreshold_barrier",
            "body_name": RIGHT_WRIST_BODY_NAME,
            "onset_m": RIGHT_WRIST_BARRIER_ONSET_M,
            "termination_m": RIGHT_WRIST_TERMINATION_M,
            "denominator_floor": RIGHT_WRIST_BARRIER_DENOMINATOR_FLOOR,
            "weight": RIGHT_WRIST_BARRIER_WEIGHT,
        },
        "joint_target_imitation": False,
        "teacher_labels_used": False,
    }


def make_task_space_ppo_env_cfg(
    *,
    motion_file: str,
    num_envs: int,
) -> Any:
    from mjlab.managers.reward_manager import RewardTermCfg

    cfg = make_sonic_true23_student_qualification_env_cfg(
        motion_file=motion_file,
        num_envs=num_envs,
        anchor_q9=9,
        transitions=510,
    )
    cfg.rewards["worst_ee_z_normalized_squared"] = RewardTermCfg(
        func=worst_ee_z_normalized_squared,
        weight=WORST_EE_WEIGHT,
        params={
            "command_name": "motion",
            "body_names": EE_TERMINATION_BODY_NAMES,
            "normalization_m": WORST_EE_NORMALIZATION_M,
        },
    )
    cfg.rewards["right_wrist_prethreshold_barrier"] = RewardTermCfg(
        func=right_wrist_prethreshold_barrier,
        weight=RIGHT_WRIST_BARRIER_WEIGHT,
        params={
            "command_name": "motion",
            "body_name": RIGHT_WRIST_BODY_NAME,
            "onset_m": RIGHT_WRIST_BARRIER_ONSET_M,
            "termination_m": RIGHT_WRIST_TERMINATION_M,
            "denominator_floor": RIGHT_WRIST_BARRIER_DENOMINATOR_FLOOR,
        },
    )
    audit_task_space_ppo_env_cfg(cfg, expected_num_envs=num_envs)
    return cfg


def audit_task_space_ppo_env_cfg(
    cfg: Any,
    *,
    expected_num_envs: int,
) -> dict[str, Any]:
    if isinstance(expected_num_envs, bool) or not isinstance(expected_num_envs, int) or expected_num_envs <= 0:
        raise ValueError("expected_num_envs must be positive integer")
    student = audit_sonic_true23_student_qualification_env_cfg(
        cfg,
        expected_anchor_q9=9,
        expected_transitions=510,
    )
    if int(cfg.scene.num_envs) != expected_num_envs:
        raise ValueError("task-space environment count drift")
    worst = cfg.rewards.get("worst_ee_z_normalized_squared")
    wrist = cfg.rewards.get("right_wrist_prethreshold_barrier")
    if (
        worst is None
        or worst.func is not worst_ee_z_normalized_squared
        or float(worst.weight) != WORST_EE_WEIGHT
        or tuple(worst.params.get("body_names", ())) != EE_TERMINATION_BODY_NAMES
        or float(worst.params.get("normalization_m")) != WORST_EE_NORMALIZATION_M
        or wrist is None
        or wrist.func is not right_wrist_prethreshold_barrier
        or float(wrist.weight) != RIGHT_WRIST_BARRIER_WEIGHT
        or wrist.params.get("body_name") != RIGHT_WRIST_BODY_NAME
        or float(wrist.params.get("onset_m")) != RIGHT_WRIST_BARRIER_ONSET_M
        or float(wrist.params.get("termination_m")) != RIGHT_WRIST_TERMINATION_M
        or float(wrist.params.get("denominator_floor")) != RIGHT_WRIST_BARRIER_DENOMINATOR_FLOOR
        or float(cfg.rewards["alive"].weight) != ALIVE_WEIGHT
        or float(cfg.rewards["non_timeout_termination"].weight) != NON_TIMEOUT_TERMINATION_WEIGHT
    ):
        raise ValueError("task-space reward execution contract drift")
    if "action_target_reference_l2" in cfg.rewards:
        raise ValueError("task-space PPO may not use joint-target imitation")
    return {
        "schema": "g1_true23_sonic_task_space_ppo_env_v1",
        "student_environment": student,
        "num_envs": expected_num_envs,
        "rewards": task_space_reward_contract(),
        "teacher_action_or_label_present": False,
        "safe_target_transform_application_count": 1,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite runtime material: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _checkpoint_header() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "training_state_evidence_included": True,
        "pilot_resume_permitted": False,
        "weights_only_load_validation_required": True,
        "deployment_ready": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
    }


def validate_task_space_checkpoint(
    checkpoint: Any,
    *,
    expected_contract_sha256: str = CONTRACT_SHA256,
    expected_initial_critic_sha256: str | None = None,
    expected_run_materials_sha256: str | None = None,
) -> Mapping[str, Any]:
    body = _mapping(checkpoint, "task-space checkpoint")
    expected_keys = {
        "g1_true23_sonic_task_space_ppo_checkpoint",
        "policy_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "update_count",
        "trainer_state",
        "initial_critic_state_sha256",
        "optimizer_parameter_tensor_count",
        "optimizer_step_count",
        "executed_training_transitions",
        "policy_state_sha256",
        "critic_state_sha256",
        "frozen_actor_state_sha256",
        "trainable_actor_state_sha256",
        "initial_overlay_policy_state_sha256",
        "contract_sha256",
        "run_materials_sha256",
        "source_actor",
        "training_boundary",
    }
    if set(body) != expected_keys or body.get("g1_true23_sonic_task_space_ppo_checkpoint") != _checkpoint_header():
        raise ValueError("task-space checkpoint root/header mismatch")
    if body.get("contract_sha256") != _require_sha256(expected_contract_sha256, "expected contract SHA256"):
        raise ValueError("task-space checkpoint contract mismatch")
    run_materials_sha256 = _require_sha256(
        body.get("run_materials_sha256"),
        "checkpoint run materials SHA256",
    )
    if expected_run_materials_sha256 is not None and run_materials_sha256 != _require_sha256(
        expected_run_materials_sha256,
        "expected run materials SHA256",
    ):
        raise ValueError("task-space checkpoint run materials mismatch")
    update = body.get("update_count")
    if isinstance(update, bool) or not isinstance(update, int) or update not in CHECKPOINT_UPDATES:
        raise ValueError("task-space checkpoint update is not allowed")
    policy = _mapping(body.get("policy_state_dict"), "checkpoint policy")
    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy},
        reference_profile=REFERENCE_PROFILE,
    )
    if body.get("policy_state_sha256") != policy_hash:
        raise ValueError("task-space checkpoint policy hash mismatch")
    if body.get("initial_overlay_policy_state_sha256") != (
        "358310ececeff0177386ae28f60b513a94902465b7e99ac480d40ba21578af61"
    ):
        raise ValueError("task-space checkpoint initial overlay identity mismatch")
    if update == 0 and policy_hash != body["initial_overlay_policy_state_sha256"]:
        raise ValueError("task-space update0 policy is not the exact overlay")
    frozen, trainable = _policy_state_subsets(policy)
    if (
        body.get("frozen_actor_state_sha256") != _state_sha256(frozen)
        or body.get("frozen_actor_state_sha256") != INITIAL_FROZEN_ACTOR_STATE_SHA256
        or body.get("trainable_actor_state_sha256") != _state_sha256(trainable)
        or (update == 0 and body.get("trainable_actor_state_sha256") != INITIAL_TRAINABLE_ACTOR_STATE_SHA256)
    ):
        raise ValueError("task-space checkpoint actor subset identity mismatch")
    critic = _mapping(body.get("critic_state_dict"), "checkpoint critic")
    _validate_critic_state_schema(critic)
    if body.get("critic_state_sha256") != _state_sha256(critic):
        raise ValueError("task-space checkpoint critic hash mismatch")
    _require_sha256(
        body.get("initial_critic_state_sha256"),
        "checkpoint initial critic state SHA256",
    )
    if expected_initial_critic_sha256 is not None and body.get("initial_critic_state_sha256") != _require_sha256(
        expected_initial_critic_sha256,
        "expected initial critic state SHA256",
    ):
        raise ValueError("task-space checkpoint initial critic binding mismatch")
    if update == 0 and body.get("critic_state_sha256") != body.get("initial_critic_state_sha256"):
        raise ValueError("task-space update0 critic differs from fresh critic")
    optimizer = _mapping(body.get("optimizer_state_dict"), "checkpoint optimizer")
    if set(optimizer) != {"state", "param_groups"}:
        raise ValueError("task-space checkpoint optimizer schema mismatch")
    optimizer_state = _mapping(optimizer.get("state"), "checkpoint Adam state")
    param_groups = optimizer.get("param_groups")
    if not isinstance(param_groups, list) or len(param_groups) != 1:
        raise ValueError("task-space checkpoint must have one Adam parameter group")
    param_group = _mapping(param_groups[0], "checkpoint Adam parameter group")
    parameter_ids = param_group.get("params")
    parameter_tensor_count = body.get("optimizer_parameter_tensor_count")
    if (
        not isinstance(parameter_ids, list)
        or len(parameter_ids) != len(set(parameter_ids))
        or parameter_ids != list(range(EXPECTED_OPTIMIZER_PARAMETER_TENSOR_COUNT))
        or isinstance(parameter_tensor_count, bool)
        or not isinstance(parameter_tensor_count, int)
        or parameter_tensor_count != len(parameter_ids)
        or parameter_tensor_count != EXPECTED_OPTIMIZER_PARAMETER_TENSOR_COUNT
        or set(param_group)
        != {
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
            "decoupled_weight_decay",
            "params",
        }
        or float(param_group.get("lr", math.nan)) != FIXED_LEARNING_RATE
        or tuple(param_group.get("betas", ())) != (0.9, 0.999)
        or float(param_group.get("eps", math.nan)) != 1.0e-8
        or float(param_group.get("weight_decay", math.nan)) != 0.0
        or param_group.get("amsgrad") is not False
        or param_group.get("maximize") is not False
        or param_group.get("foreach") is not None
        or param_group.get("capturable") is not False
        or param_group.get("differentiable") is not False
        or param_group.get("fused") is not None
        or param_group.get("decoupled_weight_decay") is not False
    ):
        raise ValueError("task-space checkpoint Adam parameter group mismatch")
    optimizer_steps = body.get("optimizer_step_count")
    expected_optimizer_steps = update * OPTIMIZER_STEPS_PER_UPDATE
    transitions = body.get("executed_training_transitions")
    if (
        optimizer_steps != expected_optimizer_steps
        or transitions != update * TRANSITIONS_PER_UPDATE
        or transitions > MAXIMUM_TRAINING_TRANSITIONS
        or optimizer_steps > MAXIMUM_OPTIMIZER_STEPS
    ):
        raise ValueError("task-space checkpoint execution counters mismatch")
    if update == 0:
        if optimizer_state:
            raise ValueError("task-space update0 Adam state must be empty")
    else:
        if set(optimizer_state) != set(parameter_ids):
            raise ValueError("task-space Adam state parameter coverage mismatch")
        for parameter_id, entry_value in optimizer_state.items():
            entry = _mapping(entry_value, f"Adam state {parameter_id}")
            if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("task-space Adam state entry schema mismatch")
            step = entry.get("step")
            if isinstance(step, torch.Tensor):
                if step.numel() != 1 or not step.is_floating_point() or not bool(torch.isfinite(step).all()):
                    raise ValueError("task-space Adam step tensor must be scalar")
                step = step.detach().cpu().item()
            if (
                isinstance(step, bool)
                or not isinstance(step, (int, float))
                or not math.isfinite(float(step))
                or float(step) != float(optimizer_steps)
            ):
                raise ValueError("task-space Adam internal step counter mismatch")
            expected_shape = OPTIMIZER_PARAMETER_SHAPES[int(parameter_id)]
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = entry[moment_name]
                if (
                    type(moment) is not torch.Tensor
                    or tuple(moment.shape) != expected_shape
                    or moment.dtype != torch.float32
                    or not bool(torch.isfinite(moment).all())
                ):
                    raise ValueError(f"task-space Adam {moment_name} tensor mismatch")
    trainer = _mapping(body.get("trainer_state"), "checkpoint trainer state")
    if (
        set(trainer)
        != {
            "completed_update_count",
            "current_learning_iteration",
            "env_common_step_counter",
            "algorithm_learning_rate",
        }
        or trainer.get("completed_update_count") != update
        or trainer.get("current_learning_iteration") != update
        or trainer.get("env_common_step_counter") != update * NUM_STEPS_PER_ENV
        or trainer.get("algorithm_learning_rate") != FIXED_LEARNING_RATE
    ):
        raise ValueError("task-space checkpoint counters mismatch")
    source = _mapping(body.get("source_actor"), "checkpoint source actor")
    if (
        source.get("checkpoint_sha256") != "85bd6de646905a44190dbf32c79737082bb604ab007a90a62e4fd2fdeeee6bd9"
        or source.get("policy_state_sha256") != "c3bfcb5c42929293b62425f155b59ccb731f57c98e8852c7f1e97094525684af"
        or source.get("lineage_sha256") != "08bbd03d0df751328e449d3624d79167587b461d6835fa1f4a8742aad9ffa82a"
        or source.get("checkpoint_update_count") != 250
        or source.get("v2_decoder_sha256") != "011740f86483323fc0f1c39ab25b784cf9411b401e56fee8b7a716664e921ee1"
        or source.get("critic_reused") is not False
        or source.get("optimizer_reused") is not False
        or source.get("counters_reused") is not False
    ):
        raise ValueError("task-space checkpoint source boundary mismatch")
    boundary = _mapping(body.get("training_boundary"), "training boundary")
    if (
        boundary.get("trainable_actor_parameters") != list(TRAINABLE_ACTOR_PARAMETERS)
        or boundary.get("trainable_actor_parameter_count") != EXPECTED_TRAINABLE_PARAMETER_COUNT
        or boundary.get("exploration_std") != EXPLORATION_STD
        or boundary.get("exploration_std_trainable") is not False
        or boundary.get("teacher_labels_used") is not False
        or boundary.get("hardware_authorized") is not False
        or boundary.get("deployment_ready") is not False
    ):
        raise ValueError("task-space checkpoint training boundary mismatch")
    return body


class SonicTaskSpacePpoRunner(CausalHistoryMjlabOnPolicyRunner):
    """Actor-only model250/v2 overlay with fresh value/Adam/counters."""

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
        *,
        source_actor_checkpoint_path: str | Path,
        overlay_encoder_path: str | Path,
        overlay_decoder_path: str | Path,
        task_space_contract: Mapping[str, Any],
        run_materials_sha256: str,
        **kwargs: Any,
    ) -> None:
        validate_task_space_ppo_contract(task_space_contract)
        algorithm_cfg = _mapping(train_cfg.get("algorithm"), "PPO algorithm config")
        actor_cfg = _mapping(train_cfg.get("actor"), "PPO actor config")
        critic_cfg = _mapping(train_cfg.get("critic"), "PPO critic config")
        obs_groups = _mapping(train_cfg.get("obs_groups"), "PPO observation groups")
        distribution_cfg = _mapping(
            actor_cfg.get("distribution_cfg"),
            "PPO actor distribution config",
        )
        if (
            int(getattr(env, "num_envs", -1)) != NUM_ENVS
            or train_cfg.get("seed") != 20260805
            or train_cfg.get("num_steps_per_env") != NUM_STEPS_PER_ENV
            or train_cfg.get("max_iterations") != MAXIMUM_UPDATES
            or train_cfg.get("save_interval") != 5
            or train_cfg.get("clip_actions") is not None
            or train_cfg.get("upload_model") is not False
            or tuple(obs_groups.get("actor", ())) != ("tokenizer", "policy")
            or tuple(obs_groups.get("critic", ())) != ("critic",)
            or actor_cfg.get("class_name") != "gear_sonic.trl.mjlab.true23_actor:True23SonicActorModel"
            or actor_cfg.get("obs_normalization") is not False
            or actor_cfg.get("tokenizer_obs_group") != "tokenizer"
            or actor_cfg.get("proprioception_obs_group") != "policy"
            or distribution_cfg
            != {
                "class_name": "GaussianDistribution",
                "init_std": 0.1,
                "std_type": "scalar",
            }
            or tuple(critic_cfg.get("hidden_dims", ())) != (512, 256, 128)
            or critic_cfg.get("activation") != "elu"
            or critic_cfg.get("obs_normalization") is not True
            or algorithm_cfg.get("num_learning_epochs") != PPO_LEARNING_EPOCHS
            or algorithm_cfg.get("num_mini_batches") != PPO_MINI_BATCHES
            or algorithm_cfg.get("learning_rate") != FIXED_LEARNING_RATE
            or algorithm_cfg.get("schedule") != "fixed"
            or algorithm_cfg.get("desired_kl") is not None
            or algorithm_cfg.get("clip_param") != 0.1
            or algorithm_cfg.get("entropy_coef") != 0.0
            or algorithm_cfg.get("gamma") != 0.99
            or algorithm_cfg.get("lam") != 0.95
            or algorithm_cfg.get("max_grad_norm") != 0.25
        ):
            raise ValueError("task-space executed PPO configuration drift")
        if log_dir is None:
            raise ValueError("task-space runner requires log directory")
        self._task_space_contract = copy.deepcopy(dict(task_space_contract))
        self._run_materials_sha256 = _require_sha256(
            run_materials_sha256,
            "task-space run materials SHA256",
        )
        self._source_actor_checkpoint_path = Path(source_actor_checkpoint_path).expanduser().resolve()
        self._overlay_encoder_path = Path(overlay_encoder_path).expanduser().resolve()
        self._overlay_decoder_path = Path(overlay_decoder_path).expanduser().resolve()
        super().__init__(
            env,
            train_cfg,
            log_dir,
            device,
            registry_name,
            **kwargs,
        )

        state, overlay = load_overlay_policy_state(
            source_checkpoint_path=self._source_actor_checkpoint_path,
            encoder_path=self._overlay_encoder_path,
            decoder_path=self._overlay_decoder_path,
            contract=self._task_space_contract,
        )
        self._policy_state_adapter.load_state_dict(state, strict=True)
        actor = self.alg.get_policy()
        named_actor = dict(actor.named_parameters())
        if set(TRAINABLE_ACTOR_PARAMETERS) - set(named_actor) or STD_PARAMETER not in named_actor:
            raise ValueError("task-space actor parameter namespace mismatch")
        for name, parameter in named_actor.items():
            parameter.requires_grad_(name in TRAINABLE_ACTOR_PARAMETERS)
        trainable_actor = [named_actor[name] for name in TRAINABLE_ACTOR_PARAMETERS]
        if sum(parameter.numel() for parameter in trainable_actor) != EXPECTED_TRAINABLE_PARAMETER_COUNT:
            raise ValueError("task-space trainable actor parameter count mismatch")
        if not torch.all(named_actor[STD_PARAMETER].detach() == EXPLORATION_STD):
            raise ValueError("task-space exploration std overlay mismatch")

        self._frozen_actor_initial = {
            name: parameter.detach().clone()
            for name, parameter in named_actor.items()
            if name not in TRAINABLE_ACTOR_PARAMETERS
        }
        self._initial_overlay_policy_state_sha256 = inspect_true23_policy_state(
            {"policy_state_dict": self._policy_state_adapter.state_dict()},
            reference_profile=REFERENCE_PROFILE,
        )
        if self._initial_overlay_policy_state_sha256 != overlay["overlay_policy_state_sha256"]:
            raise RuntimeError("loaded task-space overlay hash drifted")
        frozen_export, trainable_export = _policy_state_subsets(self._policy_state_adapter.state_dict())
        if (
            _state_sha256(frozen_export) != INITIAL_FROZEN_ACTOR_STATE_SHA256
            or _state_sha256(trainable_export) != INITIAL_TRAINABLE_ACTOR_STATE_SHA256
        ):
            raise RuntimeError("task-space overlay actor subset hash drifted")

        trainable_critic = list(self.alg.critic.parameters())
        _validate_critic_state_schema(self.alg.critic.state_dict())
        if not trainable_critic or any(not parameter.requires_grad for parameter in trainable_critic):
            raise ValueError("task-space critic must be fresh and fully trainable")
        if len(trainable_critic) != EXPECTED_CRITIC_PARAMETER_TENSOR_COUNT:
            raise ValueError("task-space critic parameter tensor count mismatch")
        self.alg.optimizer = torch.optim.Adam(
            [*trainable_actor, *trainable_critic],
            lr=FIXED_LEARNING_RATE,
        )
        self.alg.learning_rate = FIXED_LEARNING_RATE
        if self.alg.optimizer.state:
            raise RuntimeError("fresh task-space Adam unexpectedly has state")
        self._initial_critic_state_sha256 = _state_sha256(self.alg.critic.state_dict())
        self._optimizer_parameter_tensor_count = len([*trainable_actor, *trainable_critic])
        if self._optimizer_parameter_tensor_count != EXPECTED_OPTIMIZER_PARAMETER_TENSOR_COUNT:
            raise RuntimeError("task-space optimizer tensor count drifted")
        self._optimizer_step_count = 0
        self._executed_training_transitions = 0
        self._optimizer_parameter_ids = {id(parameter) for parameter in (*trainable_actor, *trainable_critic)}
        self._frozen_parameter_ids = {
            id(parameter) for name, parameter in named_actor.items() if name not in TRAINABLE_ACTOR_PARAMETERS
        }
        if self.completed_update_count != 0 or self.current_learning_iteration != 0:
            raise RuntimeError("task-space runner counters must start at zero")
        self._assert_training_boundary()

        runtime = {
            "schema_version": 1,
            "kind": "g1_true23_sonic_task_space_ppo_runtime_v1",
            "contract_sha256": CONTRACT_SHA256,
            "run_materials_sha256": self._run_materials_sha256,
            "source_actor": overlay,
            "initial_overlay_policy_state_sha256": (self._initial_overlay_policy_state_sha256),
            "fresh_critic_state_sha256": _state_sha256(self.alg.critic.state_dict()),
            "fresh_optimizer": "torch.optim.Adam",
            "fresh_optimizer_state_entry_count": len(self.alg.optimizer.state),
            "fresh_optimizer_parameter_tensor_count": (self._optimizer_parameter_tensor_count),
            "optimizer_step_count": self._optimizer_step_count,
            "executed_training_transitions": (self._executed_training_transitions),
            "fresh_counters": {
                "completed_update_count": self.completed_update_count,
                "current_learning_iteration": self.current_learning_iteration,
            },
            "trainable_actor_parameters": list(TRAINABLE_ACTOR_PARAMETERS),
            "trainable_actor_parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
            "frozen_actor_parameter_names": sorted(self._frozen_actor_initial),
            "exploration_std": EXPLORATION_STD,
            "checkpoint_updates": list(CHECKPOINT_UPDATES),
            "evaluation_updates": list(EVALUATION_UPDATES),
            "teacher_labels_used": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }
        runtime_dir = Path(log_dir).expanduser().resolve()
        _write_json_new(runtime_dir / PILOT_RUNTIME_FILENAME, runtime)

    def _assert_training_boundary(self) -> None:
        actor = self.alg.get_policy()
        named_actor = dict(actor.named_parameters())
        for name, initial in self._frozen_actor_initial.items():
            if name not in named_actor or not torch.equal(named_actor[name].detach(), initial):
                raise RuntimeError(f"task-space frozen actor changed: {name}")
        if not torch.all(named_actor[STD_PARAMETER].detach() == EXPLORATION_STD):
            raise RuntimeError("task-space fixed std changed")
        optimizer_ids = {
            id(parameter) for group in self.alg.optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != self._optimizer_parameter_ids:
            raise RuntimeError("task-space optimizer parameter boundary changed")
        if optimizer_ids & self._frozen_parameter_ids:
            raise RuntimeError("task-space optimizer contains frozen actor")
        if type(self.alg.optimizer) is not torch.optim.Adam:
            raise RuntimeError("task-space optimizer is not fresh exact Adam")
        if float(self.alg.learning_rate) != FIXED_LEARNING_RATE or any(
            float(group["lr"]) != FIXED_LEARNING_RATE for group in self.alg.optimizer.param_groups
        ):
            raise RuntimeError("task-space fixed learning rate changed")
        update = self._require_counter_coherence()
        if (
            self._optimizer_step_count != update * OPTIMIZER_STEPS_PER_UPDATE
            or self._executed_training_transitions != update * TRANSITIONS_PER_UPDATE
            or self._optimizer_step_count > MAXIMUM_OPTIMIZER_STEPS
            or self._executed_training_transitions > MAXIMUM_TRAINING_TRANSITIONS
        ):
            raise RuntimeError("task-space execution counters diverged")

    def _numbered_checkpoint_path(self, update_count: int) -> Path:
        if isinstance(update_count, bool) or not isinstance(update_count, int) or update_count < 0:
            raise ValueError("task-space checkpoint update must be nonnegative integer")
        return self.checkpoint_dir / f"sonic_task_space_model_{update_count}.pt"

    def _checkpoint_body(self) -> dict[str, Any]:
        self._assert_training_boundary()
        update = self._require_counter_coherence()
        if update not in CHECKPOINT_UPDATES:
            raise ValueError("task-space checkpoint update is not predeclared")
        policy = self._policy_state_adapter.state_dict()
        critic = {
            name: value.detach().cpu().contiguous().clone() for name, value in self.alg.critic.state_dict().items()
        }
        frozen, trainable = _policy_state_subsets(policy)
        actor_contract = self._task_space_contract["actor_initialization"]
        return {
            "g1_true23_sonic_task_space_ppo_checkpoint": _checkpoint_header(),
            "policy_state_dict": policy,
            "critic_state_dict": critic,
            "optimizer_state_dict": copy.deepcopy(self.alg.optimizer.state_dict()),
            "update_count": update,
            "trainer_state": self._trainer_state(),
            "initial_critic_state_sha256": self._initial_critic_state_sha256,
            "optimizer_parameter_tensor_count": (self._optimizer_parameter_tensor_count),
            "optimizer_step_count": self._optimizer_step_count,
            "executed_training_transitions": (self._executed_training_transitions),
            "policy_state_sha256": inspect_true23_policy_state(
                {"policy_state_dict": policy},
                reference_profile=REFERENCE_PROFILE,
            ),
            "critic_state_sha256": _state_sha256(critic),
            "frozen_actor_state_sha256": _state_sha256(frozen),
            "trainable_actor_state_sha256": _state_sha256(trainable),
            "initial_overlay_policy_state_sha256": (self._initial_overlay_policy_state_sha256),
            "contract_sha256": CONTRACT_SHA256,
            "run_materials_sha256": self._run_materials_sha256,
            "source_actor": {
                "checkpoint_sha256": actor_contract["source_checkpoint_sha256"],
                "policy_state_sha256": actor_contract["source_policy_state_sha256"],
                "lineage_sha256": actor_contract["source_lineage_sha256"],
                "checkpoint_update_count": actor_contract["source_checkpoint_update_count"],
                "v2_decoder_sha256": actor_contract["v2_decoder_sha256"],
                "critic_reused": False,
                "optimizer_reused": False,
                "counters_reused": False,
            },
            "training_boundary": {
                "trainable_actor_parameters": list(TRAINABLE_ACTOR_PARAMETERS),
                "trainable_actor_parameter_count": (EXPECTED_TRAINABLE_PARAMETER_COUNT),
                "exploration_std": EXPLORATION_STD,
                "exploration_std_trainable": False,
                "teacher_labels_used": False,
                "hardware_authorized": False,
                "deployment_ready": False,
            },
        }

    def save(self, path: str, infos: dict | None = None) -> None:
        if infos is not None:
            raise ValueError("task-space checkpoint forbids stock infos")
        output = Path(path).expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite task-space checkpoint: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        body = self._checkpoint_body()
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            torch.save(body, temporary)
            loaded = torch.load(temporary, map_location="cpu", weights_only=True)
            validate_task_space_checkpoint(
                loaded,
                expected_initial_critic_sha256=(self._initial_critic_state_sha256),
                expected_run_materials_sha256=self._run_materials_sha256,
            )
            os.link(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self._last_checkpoint_path = output
        self._last_checkpoint_update_count = body["update_count"]

    def load(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("task-space 25-update pilot does not permit resume")

    def save_current_checkpoint(self) -> Path:
        self._save_numbered_checkpoint()
        if self._last_checkpoint_path is None:
            raise RuntimeError("task-space checkpoint publication failed")
        return self._last_checkpoint_path

    def _save_numbered_checkpoint(self) -> None:
        self._assert_training_boundary()
        update = self._require_counter_coherence()
        if update not in CHECKPOINT_UPDATES:
            return
        super()._save_numbered_checkpoint()

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        if init_at_random_ep_len is not False:
            raise ValueError("task-space pilot requires exact q9=9 episode starts")
        if self.completed_update_count + num_learning_iterations > MAXIMUM_UPDATES:
            raise ValueError("task-space pilot exceeds maximum 25 updates")
        original_update = self.alg.update
        original_optimizer_step = self.alg.optimizer.step
        original_env_step = self.env.step

        def bounded_optimizer_step(*args: Any, **kwargs: Any) -> Any:
            if self._optimizer_step_count >= MAXIMUM_OPTIMIZER_STEPS:
                raise RuntimeError("task-space optimizer-step cap reached")
            result = original_optimizer_step(*args, **kwargs)
            self._optimizer_step_count += 1
            return result

        def bounded_env_step(actions: torch.Tensor) -> Any:
            proposed = self._executed_training_transitions + NUM_ENVS
            if proposed > MAXIMUM_TRAINING_TRANSITIONS:
                raise RuntimeError("task-space training-transition cap reached")
            result = original_env_step(actions)
            self._executed_training_transitions = proposed
            return result

        def bounded_update() -> Any:
            steps_before = self._optimizer_step_count
            result = original_update()
            if self._optimizer_step_count - steps_before != OPTIMIZER_STEPS_PER_UPDATE:
                raise RuntimeError("task-space PPO optimizer-step cadence drift")
            next_update = self.completed_update_count + 1
            if self._executed_training_transitions != (next_update * TRANSITIONS_PER_UPDATE):
                raise RuntimeError("task-space PPO rollout transition cadence drift")
            # Base runner increments outer counters immediately after return.
            self.completed_update_count = next_update
            self.current_learning_iteration = next_update
            self._assert_training_boundary()
            self.completed_update_count = next_update - 1
            self.current_learning_iteration = next_update - 1
            return result

        self.alg.update = bounded_update
        self.alg.optimizer.step = bounded_optimizer_step
        self.env.step = bounded_env_step
        try:
            super().learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=False,
            )
        finally:
            self.alg.update = original_update
            self.alg.optimizer.step = original_optimizer_step
            self.env.step = original_env_step
        self._assert_training_boundary()


def _evaluation_q9(command: Any) -> int:
    value = getattr(command, "time_steps", None)
    if type(value) is not torch.Tensor or value.shape != (1,) or value.dtype != torch.long:
        raise ValueError("task-space evaluation q9 must be int64 [1]")
    return int(value.detach().cpu().item())


def _evaluation_termination_names(raw_env: Any) -> list[str]:
    names: list[str] = []
    for name, values in raw_env.termination_manager.get_active_iterable_terms(0):
        if len(values) != 1 or float(values[0]) not in {0.0, 1.0}:
            raise ValueError("task-space evaluation termination value drift")
        if float(values[0]) == 1.0:
            names.append(name)
    return sorted(names)


class _TaskSpaceEvaluationTerminalRecorder:
    """Capture the diagnostic action chain before one-env autoreset clears it."""

    def __init__(self, raw_env: Any) -> None:
        self.raw_env = raw_env
        self._original_reset = raw_env._reset_idx
        self._armed = False
        self.captured: dict[str, Any] | None = None

        def observed_reset(_env: Any, env_ids: torch.Tensor | None = None) -> None:
            if self._armed and int(_env.common_step_counter) > 0:
                if (
                    self.captured is not None
                    or env_ids is None
                    or int(env_ids.numel()) != 1
                    or int(env_ids.detach().cpu().item()) != 0
                ):
                    raise RuntimeError("task-space terminal capture seam drift")
                self.captured = {
                    "action_chain": capture_student_action_chain(_env),
                    "termination_names": _evaluation_termination_names(_env),
                    "episode_length_pre_reset": int(_env.episode_length_buf[0].detach().cpu().item()),
                }
            self._original_reset(env_ids)

        raw_env._reset_idx = MethodType(observed_reset, raw_env)

    def arm(self) -> None:
        if self._armed or self.captured is not None:
            raise RuntimeError("task-space terminal recorder was not consumed")
        self._armed = True

    def finish(self, *, done: bool) -> dict[str, Any] | None:
        captured = self.captured
        self._armed = False
        self.captured = None
        if done != (captured is not None):
            raise RuntimeError("task-space done/autoreset capture mismatch")
        return captured

    def restore(self) -> None:
        self.raw_env._reset_idx = self._original_reset


def _finite_observations(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_finite_observations(item) for item in value.values())
    return True


def _action_chain_mismatch_count(
    raw_action: torch.Tensor,
    chain: Mapping[str, torch.Tensor],
) -> tuple[int, int]:
    raw = raw_action.detach().to(torch.float32)
    expected_safe, expected_target = safe_target_transform_torch(raw)
    expected_candidate = native_actions_to_hardware_targets(
        raw,
        SAFE_TARGET_DEFAULT_Q_HARDWARE,
    )
    expected_mask = torch.abs(raw) >= SAFE_TARGET_RAW_ACTION_CLIP
    values = {
        "raw_native": raw,
        "candidate_target_hardware": expected_candidate,
        "safe_native": expected_safe,
        "final_target_hardware": expected_target,
    }
    mismatch = 0
    for name, expected in values.items():
        actual = chain.get(name)
        if (
            type(actual) is not torch.Tensor
            or actual.shape != expected.shape
            or not bool(torch.isfinite(actual).all())
            or not torch.allclose(actual, expected, atol=1.0e-6, rtol=0.0)
        ):
            mismatch = 1
    actual_mask = chain.get("raw_clip_mask_native")
    if (
        type(actual_mask) is not torch.Tensor
        or actual_mask.dtype != torch.bool
        or actual_mask.shape != expected_mask.shape
        or not torch.equal(actual_mask, expected_mask)
    ):
        mismatch = 1
    return mismatch, int(torch.count_nonzero(expected_mask).detach().cpu().item())


def evaluate_task_space_policy(
    *,
    policy: Any,
    wrapped_env: Any,
    update_count: int,
    evaluation_seed: int,
) -> dict[str, Any]:
    """Run one deterministic mean-action episode in the exact one-env task."""

    if update_count not in EVALUATION_UPDATES:
        raise ValueError("task-space evaluation update is not predeclared")
    if evaluation_seed != 20260805:
        raise ValueError("task-space evaluation seed is fixed")
    raw_env = getattr(wrapped_env, "unwrapped", None)
    if (
        raw_env is None
        or int(getattr(raw_env, "num_envs", -1)) != 1
        or int(getattr(getattr(raw_env, "cfg", None), "seed", -1)) != evaluation_seed
        or getattr(wrapped_env, "clip_actions", "missing") is not None
        or int(getattr(wrapped_env, "max_episode_length", -1)) != 510
    ):
        raise ValueError("task-space evaluation wrapper/environment drift")
    if (
        int(raw_env.common_step_counter) != 0
        or int(raw_env._sim_step_counter) != 0
        or int(raw_env.episode_length_buf[0].detach().cpu().item()) != 0
    ):
        raise ValueError("task-space evaluation environment was not freshly primed")
    command = raw_env.command_manager.get_term("motion")
    if _evaluation_q9(command) != 9:
        raise ValueError("task-space evaluation did not start at q9=9")

    policy_state = policy.export_true23_policy_state()
    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy_state},
        reference_profile=REFERENCE_PROFILE,
    )
    if update_count == 0 and policy_hash != ("358310ececeff0177386ae28f60b513a94902465b7e99ac480d40ba21578af61"):
        raise ValueError("task-space update0 evaluation policy identity mismatch")

    observations = wrapped_env.get_observations()
    was_training = bool(policy.training)
    policy.eval()
    recorder = _TaskSpaceEvaluationTerminalRecorder(raw_env)
    completed = 0
    episode_return = 0.0
    terminal_q9: int | None = None
    termination_names: list[str] = []
    nonfinite_count = 0
    raw_clip_required_count = 0
    action_semantics_mismatch_count = 0
    q9_discontinuity_count = 0
    try:
        with torch.inference_mode():
            for transition in range(510):
                q9 = _evaluation_q9(command)
                if q9 != 9 + transition:
                    q9_discontinuity_count += 1
                    break
                if not _finite_observations(observations):
                    nonfinite_count += 1
                    break
                raw_action = policy(observations, stochastic_output=False)
                if (
                    type(raw_action) is not torch.Tensor
                    or raw_action.shape != (1, ACTION_DIM)
                    or not bool(torch.isfinite(raw_action).all())
                ):
                    nonfinite_count += 1
                    break
                recorder.arm()
                observations, rewards, dones, _extras = wrapped_env.step(raw_action.to(wrapped_env.device))
                if (
                    type(rewards) is not torch.Tensor
                    or rewards.shape != (1,)
                    or not bool(torch.isfinite(rewards).all())
                    or type(dones) is not torch.Tensor
                    or dones.shape != (1,)
                ):
                    nonfinite_count += 1
                    raise RuntimeError("task-space evaluation step tensors drifted")
                done = bool(int(dones[0].detach().cpu().item()))
                terminal = recorder.finish(done=done)
                chain = terminal["action_chain"] if terminal is not None else capture_student_action_chain(raw_env)
                mismatch, clip_count = _action_chain_mismatch_count(
                    raw_action,
                    chain,
                )
                action_semantics_mismatch_count += mismatch
                raw_clip_required_count += clip_count
                episode_return += float(rewards[0].detach().cpu().item())
                completed += 1
                if done:
                    terminal_q9 = q9
                    termination_names = list(terminal["termination_names"])
                    if terminal["episode_length_pre_reset"] != completed:
                        raise RuntimeError("task-space terminal episode length drift")
                    break
    finally:
        recorder.restore()
        policy.train(was_training)

    return {
        "schema_version": 1,
        "kind": "g1_true23_sonic_task_space_ppo_evaluation_v1",
        "update_count": update_count,
        "evaluation_seed": evaluation_seed,
        "controller": "deterministic_actor_mean",
        "policy_state_sha256": policy_hash,
        "completed_transitions": completed,
        "terminal_q9": terminal_q9,
        "termination_names": termination_names,
        "episode_return": episode_return,
        "nonfinite_count": nonfinite_count,
        "raw_clip_required_count": raw_clip_required_count,
        "action_semantics_mismatch_count": action_semantics_mismatch_count,
        "q9_discontinuity_count": q9_discontinuity_count,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def assess_pilot_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply predeclared episode-length/reward/q163 stop gates."""

    if not evaluations:
        raise ValueError("pilot evaluations may not be empty")
    by_update: dict[int, Mapping[str, Any]] = {}
    observed_updates: list[int] = []
    for record in evaluations:
        update = record.get("update_count")
        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or update not in EVALUATION_UPDATES
            or update in by_update
        ):
            raise ValueError("pilot evaluation update coverage invalid")
        if record.get("teacher_labels_used") is not False:
            raise ValueError("pilot evaluation may not use teacher labels")
        by_update[update] = record
        observed_updates.append(update)
    if 0 not in by_update:
        raise ValueError("pilot evaluation lacks update0 baseline")
    if tuple(observed_updates) not in (
        (0,),
        (0, 5),
        (0, 5, 10),
        (0, 5, 10, 25),
    ):
        raise ValueError("pilot evaluations are not an exact sequential prefix")
    ordered_updates = tuple(sorted(by_update))
    allowed_prefixes = (
        (0,),
        (0, 5),
        (0, 5, 10),
        (0, 5, 10, 25),
    )
    if ordered_updates not in allowed_prefixes:
        raise ValueError("pilot evaluations are not an exact sequential prefix")
    baseline = by_update[0]
    baseline_transitions = baseline.get("completed_transitions")
    baseline_return = baseline.get("episode_return")
    if (
        baseline_transitions != 155
        or baseline.get("terminal_q9") != 163
        or baseline.get("termination_names") != ["ee_body_pos"]
        or isinstance(baseline_return, bool)
        or not isinstance(baseline_return, (int, float))
        or not math.isfinite(float(baseline_return))
    ):
        return {
            "passed_so_far": False,
            "stop": True,
            "stop_reason": "update0_baseline_identity_failed",
            "highest_evaluated_update": 0,
            "q163_improved": False,
            "q163_update10_gate_reached": False,
            "q163_update10_gate_passed": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }
    reward_floor = float(baseline_return) - max(
        0.2 * abs(float(baseline_return)),
        50.0,
    )
    minimum_completed_transitions_for_q163_improvement = 156
    q163_improved = False
    q163_update10_gate_reached = False
    q163_update10_gate_passed = False
    update5_episode_length_regression_is_diagnostic_only = False
    stop_reason: str | None = None
    for update in observed_updates:
        record = by_update[update]
        safety_names = (
            "nonfinite_count",
            "raw_clip_required_count",
            "action_semantics_mismatch_count",
            "q9_discontinuity_count",
        )
        safety_counts_valid = all(
            not isinstance(record.get(name), bool) and isinstance(record.get(name), int) and record.get(name) == 0
            for name in safety_names
        )
        completed = record.get("completed_transitions")
        terminal_q9 = record.get("terminal_q9")
        termination_names = record.get("termination_names")
        episode_return = record.get("episode_return")
        if (
            not safety_counts_valid
            or isinstance(completed, bool)
            or not isinstance(completed, int)
            or not 1 <= completed <= 510
            or terminal_q9 != 9 + completed - 1
            or not isinstance(termination_names, list)
            or not termination_names
            or any(type(name) is not str for name in termination_names)
            or any(name not in GATE_PASSING_TERMINATION_NAMES for name in termination_names)
            or len(termination_names) != len(set(termination_names))
            or (completed == 510 and termination_names != ["time_out"])
            or (completed < 510 and "time_out" in termination_names)
        ):
            stop_reason = f"update{update}_episode_length_or_safety_divergence"
            break
        if update == 5:
            if completed < int(baseline_transitions):
                update5_episode_length_regression_is_diagnostic_only = True
        elif completed < int(baseline_transitions):
            stop_reason = f"update{update}_episode_length_or_safety_divergence"
            break
        if (
            isinstance(episode_return, bool)
            or not isinstance(episode_return, (int, float))
            or not math.isfinite(float(episode_return))
            or float(episode_return) < reward_floor
        ):
            stop_reason = f"update{update}_reward_divergence"
            break
        if update == 10:
            q163_update10_gate_reached = True
            q163_update10_gate_passed = completed >= int(minimum_completed_transitions_for_q163_improvement)
            q163_improved = q163_update10_gate_passed
            if not q163_update10_gate_passed:
                stop_reason = "update10_no_q163_improvement"
                break
        if update == 25 and completed < int(minimum_completed_transitions_for_q163_improvement):
            q163_improved = False
            stop_reason = "update25_q163_regression"
            break
        if update == 25 and not q163_update10_gate_passed:
            q163_improved = False
            stop_reason = "update25_without_update10_q163_gate"
            break
    highest = max(by_update)
    stop = stop_reason is not None
    return {
        "passed_so_far": not stop,
        "stop": stop,
        "stop_reason": stop_reason,
        "highest_evaluated_update": highest,
        "q163_improved": q163_improved,
        "q163_update10_gate_reached": q163_update10_gate_reached,
        "q163_update10_gate_passed": q163_update10_gate_passed,
        "update5_episode_length_regression_is_diagnostic_only": (
            update5_episode_length_regression_is_diagnostic_only
        ),
        "update25_permitted": (10 in by_update and not stop and q163_improved),
        "pilot_complete": (ordered_updates == EVALUATION_UPDATES and not stop and q163_improved),
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def execute_bounded_pilot_schedule(
    runner: SonicTaskSpacePpoRunner,
    evaluator: Callable[[SonicTaskSpacePpoRunner, int], Mapping[str, Any]],
    *,
    phase_boundary: Callable[[str], None] | None = None,
    evaluation_publisher: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute only 0→5→10→25, evaluating before every continuation."""

    if type(runner) is not SonicTaskSpacePpoRunner:
        raise TypeError("bounded pilot requires exact SonicTaskSpacePpoRunner")
    if not callable(evaluator):
        raise TypeError("bounded pilot evaluator must be callable")
    boundary = phase_boundary or (lambda _phase: None)
    evaluations: list[dict[str, Any]] = []
    checkpoints: list[str] = []

    def checkpoint_and_evaluate(update: int) -> dict[str, Any]:
        if runner.completed_update_count != update:
            raise RuntimeError("bounded pilot reached wrong evaluation update")
        checkpoint = runner.save_current_checkpoint()
        checkpoints.append(str(checkpoint))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        validated_checkpoint = validate_task_space_checkpoint(
            saved,
            expected_initial_critic_sha256=(runner._initial_critic_state_sha256),
            expected_run_materials_sha256=runner._run_materials_sha256,
        )
        if validated_checkpoint["update_count"] != update:
            raise RuntimeError("published checkpoint update identity drift")
        critic_before = _state_sha256(runner.alg.critic.state_dict())
        optimizer_before = copy.deepcopy(runner.alg.optimizer.state_dict())
        counters_before = (
            runner.completed_update_count,
            runner.current_learning_iteration,
            runner._optimizer_step_count,
            runner._executed_training_transitions,
        )
        boundary(f"before_evaluation_update_{update}")
        record = dict(evaluator(runner, update))
        boundary(f"after_evaluation_update_{update}")
        if record.get("update_count") != update:
            raise ValueError("pilot evaluator returned wrong update identity")
        if record.get("policy_state_sha256") != validated_checkpoint.get("policy_state_sha256"):
            raise ValueError("evaluated policy differs from published checkpoint")
        live_policy_hash = inspect_true23_policy_state(
            {"policy_state_dict": runner._policy_state_adapter.state_dict()},
            reference_profile=REFERENCE_PROFILE,
        )
        if (
            live_policy_hash != validated_checkpoint["policy_state_sha256"]
            or _state_sha256(runner.alg.critic.state_dict()) != critic_before
            or not _nested_state_equal(
                runner.alg.optimizer.state_dict(),
                optimizer_before,
            )
            or counters_before
            != (
                runner.completed_update_count,
                runner.current_learning_iteration,
                runner._optimizer_step_count,
                runner._executed_training_transitions,
            )
        ):
            raise RuntimeError("task-space evaluation mutated live training state")
        runner._assert_training_boundary()
        if evaluation_publisher is not None:
            evaluation_publisher(record)
        evaluations.append(record)
        return assess_pilot_evaluations(evaluations)

    assessment = checkpoint_and_evaluate(0)
    if assessment["stop"]:
        return _pilot_schedule_report(runner, evaluations, checkpoints, assessment)

    for target in (5, 10):
        boundary(f"before_training_to_update_{target}")
        runner.learn(target - runner.completed_update_count, False)
        boundary(f"after_training_to_update_{target}")
        assessment = checkpoint_and_evaluate(target)
        if assessment["stop"]:
            return _pilot_schedule_report(
                runner,
                evaluations,
                checkpoints,
                assessment,
            )

    if assessment.get("update25_permitted") is not True:
        raise RuntimeError("update10 gate did not explicitly permit update25")
    boundary("before_training_to_update_25")
    runner.learn(25 - runner.completed_update_count, False)
    boundary("after_training_to_update_25")
    assessment = checkpoint_and_evaluate(25)
    return _pilot_schedule_report(runner, evaluations, checkpoints, assessment)


def _nested_state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_nested_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_nested_state_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _pilot_schedule_report(
    runner: SonicTaskSpacePpoRunner,
    evaluations: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[str],
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    update = runner._require_counter_coherence()
    expected_transitions = update * TRANSITIONS_PER_UPDATE
    expected_optimizer_steps = update * OPTIMIZER_STEPS_PER_UPDATE
    if (
        runner._executed_training_transitions != expected_transitions
        or runner._optimizer_step_count != expected_optimizer_steps
    ):
        raise RuntimeError("bounded pilot final execution counters drifted")
    if len(evaluations) != len(checkpoints):
        raise RuntimeError("bounded pilot checkpoint/evaluation coverage drifted")

    selected = _select_pilot_candidate(evaluations, checkpoints)
    return {
        "schema_version": 1,
        "kind": "g1_true23_sonic_task_space_ppo_pilot_result_v1",
        "completed_update_count": update,
        "executed_training_transitions": expected_transitions,
        "optimizer_step_count": expected_optimizer_steps,
        "checkpoint_paths": list(checkpoints),
        **selected,
        "evaluations": [dict(item) for item in evaluations],
        "assessment": dict(assessment),
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def _select_pilot_candidate(
    evaluations: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[str],
) -> dict[str, Any]:
    if len(evaluations) != len(checkpoints):
        raise ValueError("pilot candidate checkpoint/evaluation coverage drifted")
    if not evaluations:
        raise ValueError("pilot candidate requires at least one evaluation")
    by_update: dict[int, tuple[Mapping[str, Any], str]] = {}
    observed_updates: list[int] = []
    seen_paths: set[str] = set()
    for evaluation, checkpoint in zip(evaluations, checkpoints):
        if not isinstance(evaluation, Mapping):
            raise ValueError("pilot candidate record must be a mapping")
        if not isinstance(checkpoint, str) or not checkpoint or checkpoint in seen_paths:
            raise ValueError("pilot candidate checkpoint path must be unique and non-empty")
        seen_paths.add(checkpoint)
        update = evaluation.get("update_count")
        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or update not in EVALUATION_UPDATES
            or update in by_update
        ):
            raise ValueError("pilot candidate update coverage invalid")
        by_update[int(update)] = (evaluation, checkpoint)
        observed_updates.append(int(update))
    ordered = tuple(sorted(by_update))
    if tuple(observed_updates) not in ((0,), (0, 5), (0, 5, 10), (0, 5, 10, 25)):
        raise ValueError("pilot candidate update coverage invalid")
    if ordered not in ((0,), (0, 5), (0, 5, 10), (0, 5, 10, 25)):
        raise ValueError("pilot candidate update coverage invalid")
    eligible: list[tuple[Mapping[str, Any], str]] = []
    if ordered == (0, 5, 10) or ordered == (0, 5, 10, 25):
        update10_prefix = [by_update[update][0] for update in (0, 5, 10)]
        update10_assessment = assess_pilot_evaluations(update10_prefix)
        if update10_assessment["stop"] is False and update10_assessment["q163_update10_gate_passed"] is True:
            eligible.append(by_update[10])
            if ordered == (0, 5, 10, 25):
                full = [by_update[update][0] for update in (0, 5, 10, 25)]
                if assess_pilot_evaluations(full)["stop"] is False:
                    eligible.append(by_update[25])
    if not eligible:
        return {
            "selected_candidate_update": None,
            "selected_candidate_checkpoint_path": None,
            "selected_candidate_rule": (
                "passed_update10_or_update25_only_then_maximum_completed_transitions_"
                "then_maximum_episode_return_then_earliest_update"
            ),
        }

    def candidate_rank(item: tuple[Mapping[str, Any], str]) -> tuple[int, float, int]:
        evaluation, _path = item
        completed = evaluation.get("completed_transitions")
        episode_return = evaluation.get("episode_return")
        update_count = evaluation.get("update_count")
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or isinstance(episode_return, bool)
            or not isinstance(episode_return, (int, float))
            or not math.isfinite(float(episode_return))
            or isinstance(update_count, bool)
            or not isinstance(update_count, int)
        ):
            raise ValueError("pilot candidate record malformed")
        return (
            completed if type(completed) is int else -1,
            (
                float(episode_return)
                if isinstance(episode_return, (int, float))
                and not isinstance(episode_return, bool)
                and math.isfinite(float(episode_return))
                else -math.inf
            ),
            -(update_count if type(update_count) is int else MAXIMUM_UPDATES + 1),
        )

    selected_evaluation, selected_path = max(eligible, key=candidate_rank)
    return {
        "selected_candidate_update": selected_evaluation["update_count"],
        "selected_candidate_checkpoint_path": selected_path,
        "selected_candidate_rule": (
            "passed_update10_or_update25_only_then_maximum_completed_transitions_"
            "then_maximum_episode_return_then_earliest_update"
        ),
    }


__all__ = [
    "CHECKPOINT_UPDATES",
    "CONTRACT_SHA256",
    "EVALUATION_UPDATES",
    "MAXIMUM_OPTIMIZER_STEPS",
    "MAXIMUM_TRAINING_TRANSITIONS",
    "OPTIMIZER_STEPS_PER_UPDATE",
    "SonicTaskSpacePpoRunner",
    "TRAINABLE_ACTOR_PARAMETERS",
    "assess_pilot_evaluations",
    "audit_task_space_ppo_env_cfg",
    "build_overlay_policy_state",
    "evaluate_task_space_policy",
    "execute_bounded_pilot_schedule",
    "load_overlay_policy_state",
    "load_task_space_ppo_contract",
    "make_task_space_ppo_env_cfg",
    "preflight_task_space_ppo",
    "right_wrist_prethreshold_barrier",
    "task_space_reward_contract",
    "validate_task_space_checkpoint",
    "validate_task_space_ppo_contract",
    "worst_ee_z_normalized_squared",
]
