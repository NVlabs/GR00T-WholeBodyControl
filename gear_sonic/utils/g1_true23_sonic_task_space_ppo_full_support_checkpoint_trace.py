"""Deterministic paired trace of immutable full-support PPO models 0 and 1.

This diagnostic imports the sealed v3 clone-only reward seam. It never creates
an optimizer, trains, exports, promotes, or communicates with robot hardware.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    inspect_true23_policy_state,
    sha256_file,
)
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES,
    NATIVE_IL23_JOINT_NAMES,
)
from gear_sonic.utils.g1_true23_sonic_task_space_checkpoint_trace import (
    CAPTURE_CONTRACT,
    CONTACT_SITE_NAMES,
    EE_BODY_NAMES,
    _bound_input_paths,
    _logical_parent_path,
    _manifest_record_map,
    _q9,
    _regular_directory,
    _regular_file,
    _reward_layout,
    _RewardComputeTraceRecorder,
    _seed_everything,
    _verify_bound_inputs,
    _verify_record,
    _verify_robot_assets,
    frames_to_series,
)

SCHEMA_VERSION = 1
TRACE_KIND = "g1_true23_sonic_task_space_ppo_full_support_model0_vs_model1_trace_v1"
PREFLIGHT_KIND = f"{TRACE_KIND}_preflight"
FAILURE_KIND = f"{TRACE_KIND}_failure"
TRACE_CONTRACT_KIND = "g1_true23_sonic_task_space_ppo_full_support_model0_vs_model1_trace_contract_v1"
TRACE_CONTRACT_SHA256 = "55aa08ee099f5ca853043523d32b73cca9d97ae4551a8623acfe170e10c6989a"
FULL_SUPPORT_CONTRACT_SHA256 = "7454d22296f2f1415531fd02d01ea3a18f8907270687a96434ea84be51051f3d"
RUN_MATERIALS_SHA256 = "47720f2bb39522b186099b0bd8a726834c148cd9eae8a1ca5af13f95fc30b2dd"
TRACE_V3_SHA256 = "608e126d61d14225149706d90a876445489982c7d88a49a02088d76f453fb22d"
ROLLOUT_EVIDENCE_SHA256 = "66b72261e160df34cd9defca27eb229068b0efce0b1b27f7183669f2f81e9bc6"
FIXED_SEED = 20260805
DEVICE = "cuda:0"
ACTION_DIM = 23
EXPECTED_UPDATES = (0, 1)
OUTPUT_FILENAME = "sonic_task_space_ppo_full_support_model0_vs_model1_trace_v1.json"

TRACE_CONTRACT_RELATIVE_PATH = (
    "gear_sonic/config/sim_validation/g1_true23_sonic_task_space_ppo_full_support_model0_vs_model1_trace_v1.json"
)
TRACE_SOURCE_RELATIVE_PATHS = (
    TRACE_CONTRACT_RELATIVE_PATH,
    "gear_sonic/utils/g1_true23_sonic_task_space_ppo_full_support_checkpoint_trace.py",
    "gear_sonic/scripts/trace_g1_true23_sonic_task_space_ppo_full_support_checkpoints.py",
    "gear_sonic/tests/test_g1_true23_sonic_task_space_ppo_full_support_checkpoint_trace.py",
    "gear_sonic/utils/g1_true23_sonic_task_space_checkpoint_trace.py",
    "gear_sonic/trl/mjlab/sonic_task_space_ppo_full_support_runner.py",
    "gear_sonic/trl/mjlab/sonic_task_space_ppo_runner.py",
    "gear_sonic/trl/mjlab/true23_actor.py",
    "gear_sonic/trl/mjlab/causal_history_runner.py",
)


@dataclass(frozen=True)
class TraceInputs:
    repository_root: Path
    run_dir: Path
    run_files: Mapping[str, Path]
    parent_json: Mapping[str, Mapping[str, Any]]
    material: Mapping[str, Any]
    contract: Mapping[str, Any]
    provenance: Mapping[str, Any]


class TraceReproductionError(ValueError):
    """Structural replay mismatch with publication-safe scalar evidence."""

    def __init__(self, message: str, partial_evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        _assert_scalar_evidence(partial_evidence)
        self.partial_evidence = dict(partial_evidence)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _sha256(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be lowercase SHA256")
    return value


def _canonical_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json(path: Path, context: str) -> Mapping[str, Any]:
    value = json.loads(
        _regular_file(path, context).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{context} contains non-finite JSON token {token}")
        ),
    )
    return _mapping(value, context)


def _relative_file_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    }


def _without_key(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(key, None)
    return result


def load_trace_contract(repository_root: str | Path | None = None) -> Mapping[str, Any]:
    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    path = _regular_file(root / TRACE_CONTRACT_RELATIVE_PATH, "full-support trace contract")
    if sha256_file(path) != TRACE_CONTRACT_SHA256:
        raise ValueError("full-support trace contract SHA256 mismatch")
    contract = _strict_json(path, "full-support trace contract")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != TRACE_CONTRACT_KIND
        or contract.get("purpose") != "deterministic_diagnostic_only_localize_update1_nominal_survival_regression"
    ):
        raise ValueError("full-support trace contract identity mismatch")
    parent = _mapping(contract.get("parent_run"), "trace parent run")
    execution = _mapping(contract.get("execution"), "trace execution")
    comparison = _mapping(contract.get("comparison"), "trace comparison")
    boundaries = _mapping(contract.get("boundaries"), "trace boundaries")
    output = _mapping(contract.get("output"), "trace output")
    if (
        parent.get("directory_name") != "sonic_task_space_ppo_full_support_v1_seed20260805"
        or parent.get("full_support_contract_sha256") != FULL_SUPPORT_CONTRACT_SHA256
        or parent.get("run_materials_sha256") != RUN_MATERIALS_SHA256
        or parent.get("rollout_evidence_sha256") != ROLLOUT_EVIDENCE_SHA256
        or parent.get("diagnostic_trace_v3_sha256") != TRACE_V3_SHA256
        or execution
        != {
            "updates": [0, 1],
            "evaluation_seed": FIXED_SEED,
            "num_envs": 1,
            "device": DEVICE,
            "controller": "deterministic_actor_mean",
            "max_transitions": 510,
            "initial_q9": 9,
            "expected_max_episode_length": 510,
            "wrapper_clip_actions": None,
            "capture_contract": CAPTURE_CONTRACT,
            "fresh_environment_per_update": True,
            "parent_absolute_episode_returns_are_diagnostic_only": True,
        }
        or comparison.get("common_q9_first") != 9
        or comparison.get("common_q9_last") != 159
        or comparison.get("preterminal_common_q9_last") != 158
        or comparison.get("model0_only_suffix_q9_first") != 160
        or comparison.get("model0_only_suffix_q9_last") != 163
        or output.get("filename") != OUTPUT_FILENAME
        or output.get("exclusive_create") is not True
        or boundaries
        != {
            "diagnostic_only": True,
            "failed_model5_loaded": False,
            "failed_model5_resumed": False,
            "optimizer_steps": 0,
            "training_transitions": 0,
            "checkpoints_written": 0,
            "candidate_selected": False,
            "teacher_labels_used": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "hardware_authorized": False,
            "deployment_ready": False,
            "robot_or_network_commands_permitted": False,
        }
    ):
        raise ValueError("full-support trace contract semantics mismatch")
    reproductions = _mapping(contract.get("expected_reproductions"), "expected reproductions")
    if set(reproductions) != {"0", "1"}:
        raise ValueError("full-support trace reproduction pair mismatch")
    return contract


def _callable_identity(value: Any) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if type(module) is str and type(qualname) is str:
        return f"{module}:{qualname}"
    cls = type(value)
    return f"{cls.__module__}:{cls.__qualname__}"


def reward_internal_identity(raw_env: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Bind public names to internal config objects, callables, and columns."""

    manager = raw_env.reward_manager
    names = list(manager.active_terms)
    cfgs = getattr(manager, "_term_cfgs", None)
    rates = getattr(manager, "_step_reward", None)
    expected = contract.get("expected_reward_terms")
    if (
        not isinstance(cfgs, list)
        or len(cfgs) != len(names)
        or type(rates) is not torch.Tensor
        or tuple(rates.shape) != (1, len(names))
        or not isinstance(expected, list)
        or len(expected) != len(names)
    ):
        raise ValueError("reward manager internal layout drift")
    terms: list[dict[str, Any]] = []
    for index, (name, cfg, expected_row) in enumerate(zip(names, cfgs, expected, strict=True)):
        if manager.get_term_cfg(name) is not cfg:
            raise ValueError("reward public/internal config identity mismatch")
        identity = _callable_identity(cfg.func)
        observed = [name, float(cfg.weight), identity]
        if observed != expected_row:
            raise ValueError(f"reward callable identity mismatch at {name}")
        terms.append(
            {
                "column_index": index,
                "name": name,
                "weight": float(cfg.weight),
                "callable_identity": identity,
                "parameter_names": sorted(cfg.params),
                "public_internal_cfg_object_identical": True,
            }
        )
    return {
        "manager_class": _callable_identity(type(manager)),
        "step_reward_shape": [1, len(names)],
        "column_order_verified": True,
        "terms": terms,
    }


def _trace_source_binding(root: Path) -> dict[str, Any]:
    files = {logical: root / logical for logical in TRACE_SOURCE_RELATIVE_PATHS}
    env_tree = root / "gear_sonic" / "envs" / "mjlab"
    for path in sorted(env_tree.rglob("*.py")):
        if "__pycache__" not in path.parts:
            files[f"gear_sonic/envs/mjlab/{path.relative_to(env_tree).as_posix()}"] = path
    records = []
    for logical in sorted(files):
        resolved = _regular_file(files[logical], f"trace source {logical}")
        records.append(
            {
                "logical_path": logical,
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return {
        "schema_version": 1,
        "kind": "g1_true23_full_support_pair_trace_executed_sources_v1",
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "manifest_sha256": _canonical_sha256(records),
        "files": records,
    }


def _verify_material_execution_inputs(root: Path, material: Mapping[str, Any]) -> dict[str, Any]:
    base = _mapping(material.get("base_task_space_materials"), "base task-space materials")
    source_records = _manifest_record_map(base["source_files"], "source_files")
    verified_sources = [
        _verify_record(
            _logical_parent_path(root, logical),
            record,
            f"full-support material source {logical}",
        )
        for logical, record in sorted(source_records.items())
    ]
    robot_records = _verify_robot_assets(root, base["robot_assets"])
    input_records = _verify_bound_inputs(root, base["bound_inputs"])
    from gear_sonic.scripts.train_g1_true23_native124_selected_v2_ankle import (
        resolve_rsl_runtime_binding,
    )

    rsl = resolve_rsl_runtime_binding()
    if rsl != base["rsl_runtime"]:
        raise ValueError("full-support trace RSL runtime differs from parent")
    return {
        "source_file_count": len(verified_sources),
        "source_manifest_sha256": _canonical_sha256(verified_sources),
        "robot_file_count": len(robot_records),
        "robot_manifest_sha256": _canonical_sha256(robot_records),
        "bound_input_file_count": len(input_records),
        "bound_input_manifest_sha256": _canonical_sha256(input_records),
        "rsl_runtime_binding_sha256": rsl["runtime_binding_sha256"],
    }


def _validate_parent_semantics(parent: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any]) -> None:
    expected = _mapping(contract["expected_reproductions"], "expected reproductions")
    result = parent["full_support_result.json"]
    rollout = parent["full_support_rollout_evidence.json"]
    if (
        result.get("contract_sha256") != FULL_SUPPORT_CONTRACT_SHA256
        or result.get("run_materials_sha256") != RUN_MATERIALS_SHA256
        or result.get("rollout_evidence_sha256") != ROLLOUT_EVIDENCE_SHA256
        or result.get("candidate") is not None
        or _mapping(result.get("assessment"), "parent result assessment").get("stop_reason")
        != "update1_no_q164_or_reward_improvement"
        or result.get("completed_update_count") != 1
        or result.get("executed_training_transitions") != 20_480
        or result.get("optimizer_step_count") != 8
        or result.get("failed_model5_loaded") is not False
        or result.get("failed_model5_resumed") is not False
        or rollout.get("contract_sha256") != FULL_SUPPORT_CONTRACT_SHA256
        or rollout.get("run_materials_sha256") != RUN_MATERIALS_SHA256
        or rollout.get("total_inserted_transitions") != 20_480
        or _mapping(rollout.get("coverage_assessment"), "rollout coverage").get("gate_passed") is not True
        or rollout.get("optimizer_steps_at_publication") != 0
        or rollout.get("training_updates_at_publication") != 0
    ):
        raise ValueError("full-support result/rollout semantics mismatch")
    evaluations = []
    for update in EXPECTED_UPDATES:
        evaluation = parent[f"evaluations/evaluation_update_{update}.json"]
        reproduction = _mapping(expected[str(update)], f"expected update{update}")
        if (
            evaluation.get("update_count") != update
            or evaluation.get("evaluation_seed") != FIXED_SEED
            or evaluation.get("controller") != "deterministic_actor_mean"
            or evaluation.get("policy_state_sha256") != reproduction.get("policy_state_sha256")
            or evaluation.get("completed_transitions") != reproduction.get("completed_transitions")
            or evaluation.get("terminal_q9") != reproduction.get("terminal_q9")
            or evaluation.get("termination_names") != reproduction.get("termination_names")
            or evaluation.get("episode_return") != reproduction.get("historical_episode_return")
            or any(
                evaluation.get(name) != 0
                for name in (
                    "nonfinite_count",
                    "raw_clip_required_count",
                    "action_semantics_mismatch_count",
                    "q9_discontinuity_count",
                )
            )
        ):
            raise ValueError(f"full-support update{update} evaluation semantics mismatch")
        evaluations.append(evaluation)
    if result.get("evaluations") != evaluations:
        raise ValueError("full-support result embedded evaluations mismatch")


def _load_and_validate_checkpoints(inputs: TraceInputs) -> dict[int, Mapping[str, Any]]:
    from gear_sonic.trl.mjlab.sonic_task_space_ppo_full_support_runner import (
        validate_full_support_checkpoint,
    )

    expected = inputs.contract["expected_reproductions"]
    checkpoints: dict[int, Mapping[str, Any]] = {}
    for update in EXPECTED_UPDATES:
        path = inputs.run_files[f"checkpoints/sonic_task_space_full_support_model_{update}.pt"]
        body = validate_full_support_checkpoint(
            torch.load(path, map_location="cpu", weights_only=True),
            expected_run_materials_sha256=RUN_MATERIALS_SHA256,
        )
        if (
            body.get("update_count") != update
            or body.get("policy_state_sha256") != expected[str(update)]["policy_state_sha256"]
            or body.get("rollout_evidence_sha256") != (None if update == 0 else ROLLOUT_EVIDENCE_SHA256)
        ):
            raise ValueError(f"full-support update{update} checkpoint binding mismatch")
        checkpoints[update] = body
    return checkpoints


def _current_provenance_snapshot(
    inputs: TraceInputs,
    *,
    verified_material_execution: Mapping[str, Any] | None = None,
) -> str:
    contract_parent = inputs.contract["parent_run"]
    artifacts = {
        logical: sha256_file(_regular_file(inputs.run_dir / logical, f"parent {logical}"))
        for logical in sorted(contract_parent["artifact_sha256"])
    }
    prior = _regular_file(Path(contract_parent["diagnostic_trace_v3_linux_path"]), "bound trace v3")
    sealed = {
        logical: sha256_file(_regular_file(inputs.repository_root / logical, f"sealed {logical}"))
        for logical in sorted(inputs.contract["sealed_sources"])
    }
    motion_inputs = _bound_input_paths(inputs.repository_root)
    execution_inputs = {
        name: sha256_file(_regular_file(motion_inputs[name], f"execution input {name}"))
        for name in ("actor/topology.pt", "motion/B_DadDance.npz")
    }
    material_execution = (
        _verify_material_execution_inputs(inputs.repository_root, inputs.material)
        if verified_material_execution is None
        else dict(verified_material_execution)
    )
    return _canonical_sha256(
        {
            "parent_artifact_sha256": artifacts,
            "trace_v3_sha256": sha256_file(prior),
            "sealed_source_sha256": sealed,
            "executed_trace_sources": _trace_source_binding(inputs.repository_root),
            "execution_input_sha256": execution_inputs,
            "verified_material_execution": material_execution,
        }
    )


def resolve_trace_inputs(repository_root: Path, parent_run_dir: Path) -> TraceInputs:
    root = repository_root.expanduser().resolve(strict=True)
    contract = load_trace_contract(root)
    expected_run = Path(contract["parent_run"]["linux_path"]).expanduser().resolve(strict=True)
    run_dir = parent_run_dir.expanduser().resolve(strict=True)
    if run_dir != expected_run or run_dir.name != contract["parent_run"]["directory_name"] or run_dir.is_symlink():
        raise ValueError("full-support trace parent run identity mismatch")
    artifact_hashes = _mapping(contract["parent_run"]["artifact_sha256"], "artifact hashes")
    if _relative_file_set(run_dir) != set(artifact_hashes):
        raise ValueError("full-support parent run file set mismatch")
    run_files = {
        logical: _regular_file(run_dir / logical, f"parent artifact {logical}") for logical in artifact_hashes
    }
    observed = {logical: sha256_file(path) for logical, path in run_files.items()}
    if observed != artifact_hashes:
        raise ValueError("full-support parent artifact SHA256 mismatch")
    trace_v3 = _regular_file(Path(contract["parent_run"]["diagnostic_trace_v3_linux_path"]), "bound trace v3")
    if sha256_file(trace_v3) != TRACE_V3_SHA256:
        raise ValueError("bound trace v3 SHA256 mismatch")
    sealed = contract["sealed_sources"]
    if {
        logical: sha256_file(_regular_file(root / logical, f"sealed source {logical}")) for logical in sealed
    } != sealed:
        raise ValueError("sealed full-support/v3 source SHA256 mismatch")
    parent = {
        logical: _strict_json(path, f"parent {logical}")
        for logical, path in run_files.items()
        if path.suffix == ".json"
    }
    _validate_parent_semantics(parent, contract)
    material = parent["material_manifest.json"]
    if (
        material.get("contract_sha256") != FULL_SUPPORT_CONTRACT_SHA256
        or material.get("material_manifest_sha256") != RUN_MATERIALS_SHA256
        or _canonical_sha256(_without_key(material, "material_manifest_sha256")) != RUN_MATERIALS_SHA256
        or parent["preflight.json"].get("material_manifest") != material
    ):
        raise ValueError("full-support material manifest identity mismatch")
    physical = _verify_material_execution_inputs(root, material)
    provisional = TraceInputs(
        repository_root=root,
        run_dir=run_dir,
        run_files=run_files,
        parent_json=parent,
        material=material,
        contract=contract,
        provenance={},
    )
    _load_and_validate_checkpoints(provisional)
    provenance = {
        "trace_contract_sha256": TRACE_CONTRACT_SHA256,
        "parent_run_dir": str(run_dir),
        "parent_artifact_sha256": observed,
        "parent_material_manifest_sha256": RUN_MATERIALS_SHA256,
        "bound_trace_v3_sha256": TRACE_V3_SHA256,
        "verified_material_execution": physical,
        "executed_trace_sources": _trace_source_binding(root),
    }
    inputs = TraceInputs(
        repository_root=root,
        run_dir=run_dir,
        run_files=run_files,
        parent_json=parent,
        material=material,
        contract=contract,
        provenance=provenance,
    )
    provenance["snapshot_sha256"] = _current_provenance_snapshot(inputs, verified_material_execution=physical)
    return inputs


def trace_preflight(repository_root: Path, parent_run_dir: Path) -> dict[str, Any]:
    try:
        inputs = resolve_trace_inputs(repository_root, parent_run_dir)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": PREFLIGHT_KIND,
            "ready": True,
            "parent_run_dir": str(inputs.run_dir),
            "provenance": inputs.provenance,
            **_boundary_receipt(simulator_constructed=False),
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": PREFLIGHT_KIND,
            "ready": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            **_boundary_receipt(simulator_constructed=False),
        }


def _load_actor_policy(actor: Any, checkpoint: Mapping[str, Any], update: int, expected_hash: str) -> None:
    policy = _mapping(checkpoint["policy_state_dict"], f"update{update} policy")
    network = {key: value for key, value in policy.items() if key != "std"}
    missing, unexpected = actor.core.load_state_dict(network, strict=True)
    if missing or unexpected:
        raise ValueError(f"update{update} actor load mismatch")
    std = policy.get("std")
    if type(std) is not torch.Tensor or tuple(std.shape) != (ACTION_DIM,):
        raise ValueError(f"update{update} actor std mismatch")
    with torch.no_grad():
        actor.distribution.std_param.copy_(std.to(actor.distribution.std_param.device))
    observed = inspect_true23_policy_state(
        {"policy_state_dict": actor.export_true23_policy_state()},
        reference_profile="released_low_latency_step1_0p02s",
    )
    if observed != expected_hash:
        raise ValueError(f"update{update} live actor identity mismatch")


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_finite(item) for item in value)
    return False


def validate_trace_series(
    trace: Mapping[str, Any], layout: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    update = trace.get("update_count")
    if (
        type(update) is not int
        or update not in EXPECTED_UPDATES
        or trace.get("evaluation_seed") != FIXED_SEED
        or trace.get("controller") != "deterministic_actor_mean"
    ):
        raise ValueError("trace update is not model0/model1 pair")
    required_layout = {
        "reward_terms",
        "reward_internal_identity",
        "control_dt_s",
        "ee_body_names",
        "contact_site_names",
        "native_joint_names",
        "hardware_joint_names",
        "action_orders",
        "ee_body_pos_threshold_m",
    }
    if (
        set(layout) != required_layout
        or layout["control_dt_s"] != 0.02
        or tuple(layout["ee_body_names"]) != EE_BODY_NAMES
        or tuple(layout["contact_site_names"]) != CONTACT_SITE_NAMES
        or tuple(layout["native_joint_names"]) != tuple(NATIVE_IL23_JOINT_NAMES)
        or tuple(layout["hardware_joint_names"]) != tuple(HARDWARE_23_JOINT_NAMES)
        or layout["ee_body_pos_threshold_m"] != 0.25
    ):
        raise ValueError("full-support trace layout mismatch")
    expected_terms = contract["expected_reward_terms"]
    terms = layout["reward_terms"]
    if [[term["name"], term["weight"], term["callable_identity"]] for term in terms] != expected_terms:
        raise ValueError("full-support trace reward layout mismatch")
    series = _mapping(trace.get("series"), "trace series")
    required = {
        "q9",
        "reward_total",
        "reward_raw",
        "reward_weighted",
        "ee_z_error_m",
        "raw_native_action",
        "safe_native_action",
        "final_target_hardware",
        "contact_found",
        "contact_force_magnitude_n",
        "landing_force_mean_n",
        "termination_names",
    }
    if set(series) != required:
        raise ValueError("full-support trace series schema mismatch")
    completed = trace.get("completed_transitions")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed <= 0
        or series["q9"] != list(range(9, 9 + completed))
    ):
        raise ValueError("full-support trace q9 discontinuity")
    for name in required - {"q9"}:
        if not isinstance(series[name], list) or len(series[name]) != completed:
            raise ValueError(f"full-support trace {name} length mismatch")
    widths = {
        "reward_raw": len(terms),
        "reward_weighted": len(terms),
        "ee_z_error_m": len(EE_BODY_NAMES),
        "raw_native_action": ACTION_DIM,
        "safe_native_action": ACTION_DIM,
        "final_target_hardware": ACTION_DIM,
        "contact_found": len(CONTACT_SITE_NAMES),
        "contact_force_magnitude_n": len(CONTACT_SITE_NAMES),
    }
    for name, width in widths.items():
        if any(not isinstance(row, list) or len(row) != width or not _all_finite(row) for row in series[name]):
            raise ValueError(f"full-support trace {name} row mismatch")
    if any(any(type(value) is not bool for value in row) for row in series["contact_found"]):
        raise ValueError("full-support trace contacts must be booleans")
    if not _all_finite(series["reward_total"]) or not _all_finite(series["landing_force_mean_n"]):
        raise ValueError("full-support trace scalar series nonfinite")
    weights = [term["weight"] for term in terms]
    for index in range(completed):
        if not math.isclose(
            sum(series["reward_weighted"][index]),
            series["reward_total"][index],
            rel_tol=0.0,
            abs_tol=2.0e-5,
        ):
            raise ValueError("full-support trace reward sum mismatch")
        for raw, weighted, weight in zip(
            series["reward_raw"][index],
            series["reward_weighted"][index],
            weights,
            strict=True,
        ):
            if not math.isclose(raw * weight * 0.02, weighted, rel_tol=0.0, abs_tol=2.0e-5):
                raise ValueError("full-support trace raw/weighted reward mismatch")
    terminal = [i for i, names in enumerate(series["termination_names"]) if names]
    if terminal != [completed - 1]:
        raise ValueError("full-support trace must have one final terminal frame")
    if (
        any(max(row) > 0.25 for row in series["ee_z_error_m"][:-1])
        or max(series["ee_z_error_m"][-1]) <= 0.25
        or series["termination_names"][-1] != ["ee_body_pos"]
    ):
        raise ValueError("full-support trace terminal EE threshold semantics mismatch")
    if (
        trace.get("terminal_q9") != series["q9"][-1]
        or trace.get("termination_names") != series["termination_names"][-1]
        or not math.isclose(sum(series["reward_total"]), trace.get("episode_return"), rel_tol=0.0, abs_tol=2e-5)
    ):
        raise ValueError("full-support trace terminal/return summary mismatch")


def _scalar_reproduction_evidence(
    trace: Mapping[str, Any], expected: Mapping[str, Any], update: int
) -> dict[str, Any]:
    evidence = {
        "stage": f"update{update}_structural_reproduction",
        "update_count": update,
        "observed_completed_transitions": trace.get("completed_transitions"),
        "expected_completed_transitions": expected.get("completed_transitions"),
        "observed_terminal_q9": trace.get("terminal_q9"),
        "expected_terminal_q9": expected.get("terminal_q9"),
        "observed_termination_names": ",".join(trace.get("termination_names", [])),
        "expected_termination_names": ",".join(expected.get("termination_names", [])),
        "observed_policy_state_sha256": trace.get("policy_state_sha256"),
        "expected_policy_state_sha256": expected.get("policy_state_sha256"),
    }
    _assert_scalar_evidence(evidence)
    return evidence


def validate_structural_reproduction(
    trace: Mapping[str, Any], expected: Mapping[str, Any], update: int
) -> dict[str, Any]:
    mismatch = (
        trace.get("completed_transitions") != expected.get("completed_transitions")
        or trace.get("terminal_q9") != expected.get("terminal_q9")
        or trace.get("termination_names") != expected.get("termination_names")
        or trace.get("policy_state_sha256") != expected.get("policy_state_sha256")
    )
    if mismatch:
        raise TraceReproductionError(
            f"update{update} trace structural reproduction failed",
            _scalar_reproduction_evidence(trace, expected, update),
        )
    observed_return = float(trace["episode_return"])
    historical_return = float(expected["historical_episode_return"])
    return {
        "observed_episode_return": observed_return,
        "historical_episode_return": historical_return,
        "delta_observed_minus_historical": observed_return - historical_return,
        "exact_match": observed_return == historical_return,
        "gate_applied": False,
        "reason": "absolute_replay_return_is_diagnostic_only",
    }


def run_policy_trace(
    *,
    policy: Any,
    wrapped_env: Any,
    update_count: int,
    expected: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    raw_env = getattr(wrapped_env, "unwrapped", None)
    if (
        raw_env is None
        or int(getattr(raw_env, "num_envs", -1)) != 1
        or int(getattr(raw_env.cfg, "seed", -1)) != FIXED_SEED
        or getattr(wrapped_env, "clip_actions", "missing") is not None
        or int(getattr(wrapped_env, "max_episode_length", -1)) != 510
        or int(raw_env.common_step_counter) != 0
        or int(raw_env._sim_step_counter) != 0
    ):
        raise ValueError("full-support trace environment is not fresh exact one-env task")
    command = raw_env.command_manager.get_term("motion")
    if _q9(command) != 9:
        raise ValueError("full-support trace did not start at q9=9")
    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy.export_true23_policy_state()},
        reference_profile="released_low_latency_step1_0p02s",
    )
    if policy_hash != expected["policy_state_sha256"]:
        raise ValueError("full-support trace live policy hash mismatch")
    reward = _reward_layout(raw_env)
    internal = reward_internal_identity(raw_env, contract)
    terms = [
        {
            "name": term["name"],
            "weight": term["weight"],
            "callable_identity": term["callable_identity"],
        }
        for term in internal["terms"]
    ]
    layout = {
        "reward_terms": terms,
        "reward_internal_identity": internal,
        "control_dt_s": reward["control_dt_s"],
        "ee_body_names": list(EE_BODY_NAMES),
        "contact_site_names": list(CONTACT_SITE_NAMES),
        "native_joint_names": list(NATIVE_IL23_JOINT_NAMES),
        "hardware_joint_names": list(HARDWARE_23_JOINT_NAMES),
        "action_orders": {
            "raw_native_action": "native_isaaclab_23",
            "safe_native_action": "native_isaaclab_23",
            "final_target_hardware": "hardware_mujoco_23",
        },
        "ee_body_pos_threshold_m": 0.25,
    }
    observations = wrapped_env.get_observations()
    recorder = _RewardComputeTraceRecorder(raw_env, reward)
    was_training = bool(policy.training)
    policy.eval()
    frames: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for transition in range(510):
                q9 = _q9(command)
                if q9 != 9 + transition:
                    raise ValueError("full-support trace q9 discontinuity")
                actor_action = policy(observations, stochastic_output=False)
                if (
                    type(actor_action) is not torch.Tensor
                    or tuple(actor_action.shape) != (1, ACTION_DIM)
                    or not actor_action.is_floating_point()
                    or not bool(torch.isfinite(actor_action).all())
                ):
                    raise ValueError("full-support trace actor action metadata drift")
                recorder.arm(q9, actor_action)
                observations, rewards, dones, _extras = wrapped_env.step(actor_action.to(wrapped_env.device))
                frame = recorder.finish()
                if (
                    type(rewards) is not torch.Tensor
                    or tuple(rewards.shape) != (1,)
                    or type(dones) is not torch.Tensor
                    or tuple(dones.shape) != (1,)
                ):
                    raise ValueError("full-support trace wrapped step schema drift")
                wrapped_reward = float(rewards[0].detach().cpu().item())
                if not math.isfinite(wrapped_reward) or not math.isclose(
                    frame["reward_total"], wrapped_reward, rel_tol=0.0, abs_tol=1.0e-7
                ):
                    raise ValueError("full-support trace wrapped reward identity mismatch")
                done = bool(int(dones[0].detach().cpu().item()))
                if done != bool(frame["termination_names"]):
                    raise ValueError("full-support trace done/termination mismatch")
                frames.append(frame)
                if done:
                    if frame["episode_length_pre_reset"] != len(frames):
                        raise ValueError("full-support trace terminal episode length mismatch")
                    break
    finally:
        recorder.restore()
        policy.train(was_training)
    series = frames_to_series(frames)
    trace = {
        "update_count": update_count,
        "evaluation_seed": FIXED_SEED,
        "controller": "deterministic_actor_mean",
        "policy_state_sha256": policy_hash,
        "completed_transitions": len(frames),
        "terminal_q9": frames[-1]["q9"],
        "termination_names": frames[-1]["termination_names"],
        "episode_return": sum(frame["reward_total"] for frame in frames),
        "series": series,
    }
    validate_trace_series(trace, layout, contract)
    diagnostic = validate_structural_reproduction(trace, expected, update_count)
    return {"layout": layout, "trace": trace, "episode_return_diagnostic": diagnostic}


def _threshold_key(value: float) -> str:
    return format(value, ".9g")


def _row_delta(left: Sequence[float], right: Sequence[float]) -> tuple[float, int]:
    values = [abs(a - b) for a, b in zip(left, right, strict=True)]
    index = max(range(len(values)), key=values.__getitem__)
    return values[index], index


def _threshold_ladder(
    rows: Sequence[Sequence[float]],
    q9s: Sequence[int],
    names: Sequence[str],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    deltas = [_row_delta(left, right) for left, right in rows]
    maximum_index = max(range(len(deltas)), key=lambda index: deltas[index][0])
    first: dict[str, Any] = {}
    for threshold in thresholds:
        match = next(
            ((q9, delta, index) for q9, (delta, index) in zip(q9s, deltas, strict=True) if delta >= threshold),
            None,
        )
        first[_threshold_key(float(threshold))] = (
            None if match is None else {"q9": match[0], "delta": match[1], "culprit_name": names[match[2]]}
        )
    maximum = deltas[maximum_index]
    return {
        "first_at_or_above": first,
        "maximum": {
            "q9": q9s[maximum_index],
            "delta": maximum[0],
            "culprit_name": names[maximum[1]],
        },
    }


def _scalar_divergence(
    left: Sequence[float],
    right: Sequence[float],
    q9s: Sequence[int],
    threshold: float,
) -> dict[str, Any]:
    deltas = [abs(a - b) for a, b in zip(left, right, strict=True)]
    maximum_index = max(range(len(deltas)), key=deltas.__getitem__)
    first_index = next((index for index, delta in enumerate(deltas) if delta > threshold), None)
    return {
        "threshold_strictly_greater_than": threshold,
        "first_q9": None if first_index is None else q9s[first_index],
        "first_absolute_delta": None if first_index is None else deltas[first_index],
        "first_model0": None if first_index is None else left[first_index],
        "first_model1": None if first_index is None else right[first_index],
        "first_model1_minus_model0": (None if first_index is None else right[first_index] - left[first_index]),
        "maximum_q9": q9s[maximum_index],
        "maximum_absolute_delta": deltas[maximum_index],
        "maximum_model0": left[maximum_index],
        "maximum_model1": right[maximum_index],
        "maximum_model1_minus_model0": right[maximum_index] - left[maximum_index],
    }


def _terminal_ee_culprit(trace: Mapping[str, Any]) -> dict[str, Any]:
    errors = trace["series"]["ee_z_error_m"][-1]
    index = max(range(len(errors)), key=errors.__getitem__)
    crossing = [name for name, value in zip(EE_BODY_NAMES, errors, strict=True) if value > 0.25]
    if trace["termination_names"] != ["ee_body_pos"] or not crossing:
        raise ValueError("full-support trace terminal EE culprit mismatch")
    return {
        "q9": trace["terminal_q9"],
        "body_name": EE_BODY_NAMES[index],
        "z_error_m": errors[index],
        "threshold_m": 0.25,
        "threshold_crossing_bodies": crossing,
        "all_four_ee_z_error_m": {name: value for name, value in zip(EE_BODY_NAMES, errors, strict=True)},
    }


def _reward_sums(series: Mapping[str, Any], start: int, stop: int) -> list[float]:
    width = len(series["reward_weighted"][0])
    return [sum(row[index] for row in series["reward_weighted"][start:stop]) for index in range(width)]


def compare_trace_pair(
    model0: Mapping[str, Any],
    model1: Mapping[str, Any],
    layout: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    validate_trace_series(model0, layout, contract)
    validate_trace_series(model1, layout, contract)
    if model0["update_count"] != 0 or model1["update_count"] != 1:
        raise ValueError("full-support trace comparison order mismatch")
    left = model0["series"]
    right = model1["series"]
    common_q9s = sorted(set(left["q9"]) & set(right["q9"]))
    if common_q9s != list(range(9, 160)):
        raise ValueError("full-support trace common q9 identity mismatch")
    left_index = {q9: index for index, q9 in enumerate(left["q9"])}
    right_index = {q9: index for index, q9 in enumerate(right["q9"])}

    def paired(column: str) -> list[list[Sequence[float]]]:
        return [[left[column][left_index[q9]], right[column][right_index[q9]]] for q9 in common_q9s]

    thresholds = contract["comparison"]["action_linf_thresholds"]
    action = {}
    for column, names in (
        ("raw_native_action", NATIVE_IL23_JOINT_NAMES),
        ("safe_native_action", NATIVE_IL23_JOINT_NAMES),
        ("final_target_hardware", HARDWARE_23_JOINT_NAMES),
    ):
        action[column] = _threshold_ladder(paired(column), common_q9s, names, thresholds)
    ee = _threshold_ladder(
        paired("ee_z_error_m"),
        common_q9s,
        EE_BODY_NAMES,
        contract["comparison"]["ee_error_delta_thresholds_m"],
    )
    contact_state = {}
    for site_index, site in enumerate(CONTACT_SITE_NAMES):
        contact_state[site] = next(
            (
                q9
                for q9 in common_q9s
                if left["contact_found"][left_index[q9]][site_index]
                != right["contact_found"][right_index[q9]][site_index]
            ),
            None,
        )
    contact_force = _threshold_ladder(
        paired("contact_force_magnitude_n"),
        common_q9s,
        CONTACT_SITE_NAMES,
        [contract["comparison"]["contact_force_delta_atol_n"]],
    )
    reward_names = [term["name"] for term in layout["reward_terms"]]
    common_count = len(common_q9s)
    preterminal_count = common_count - 1
    common0 = _reward_sums(left, 0, common_count)
    common1 = _reward_sums(right, 0, common_count)
    pre0 = _reward_sums(left, 0, preterminal_count)
    pre1 = _reward_sums(right, 0, preterminal_count)
    suffix_start = left_index[160]
    suffix0 = _reward_sums(left, suffix_start, len(left["q9"]))
    reward_atol = contract["comparison"]["reward_delta_atol"]
    reward_total_divergence = _scalar_divergence(
        [left["reward_total"][left_index[q9]] for q9 in common_q9s],
        [right["reward_total"][right_index[q9]] for q9 in common_q9s],
        common_q9s,
        reward_atol,
    )
    reward_term_divergence = {}
    for term_index, term_name in enumerate(reward_names):
        reward_term_divergence[term_name] = _scalar_divergence(
            [left["reward_weighted"][left_index[q9]][term_index] for q9 in common_q9s],
            [right["reward_weighted"][right_index[q9]][term_index] for q9 in common_q9s],
            common_q9s,
            reward_atol,
        )
    landing_force_divergence = _scalar_divergence(
        [left["landing_force_mean_n"][left_index[q9]] for q9 in common_q9s],
        [right["landing_force_mean_n"][right_index[q9]] for q9 in common_q9s],
        common_q9s,
        contract["comparison"]["landing_force_delta_atol_n"],
    )
    term_evidence = {}
    for term_name in (
        contract["comparison"]["barrier_term_name"],
        contract["comparison"]["worst_ee_term_name"],
    ):
        term_index = reward_names.index(term_name)
        term_evidence[term_name] = {
            "model0_first_positive_raw_q9": next(
                (q9 for q9, row in zip(left["q9"], left["reward_raw"], strict=True) if row[term_index] > 0.0),
                None,
            ),
            "model1_first_positive_raw_q9": next(
                (q9 for q9, row in zip(right["q9"], right["reward_raw"], strict=True) if row[term_index] > 0.0),
                None,
            ),
            "common_weighted_sum_model0": common0[term_index],
            "common_weighted_sum_model1": common1[term_index],
            "common_weighted_delta_model1_minus_model0": common1[term_index] - common0[term_index],
            "model0_only_suffix_weighted_sum": suffix0[term_index],
        }
    model0_suffix = {
        name: left[name][suffix_start:]
        for name in (
            "q9",
            "reward_total",
            "reward_raw",
            "reward_weighted",
            "ee_z_error_m",
            "raw_native_action",
            "safe_native_action",
            "final_target_hardware",
            "contact_found",
            "contact_force_magnitude_n",
            "landing_force_mean_n",
            "termination_names",
        )
    }
    return {
        "common_transition_count": common_count,
        "common_q9_first": common_q9s[0],
        "common_q9_last": common_q9s[-1],
        "preterminal_common_transition_count": preterminal_count,
        "preterminal_common_q9_last": common_q9s[-2],
        "model0_only_suffix_q9_first": model0_suffix["q9"][0],
        "model0_only_suffix_q9_last": model0_suffix["q9"][-1],
        "action_divergence": action,
        "ee_error_divergence": ee,
        "contact_state_first_divergence_q9": contact_state,
        "contact_force_divergence": contact_force,
        "landing_force_divergence": landing_force_divergence,
        "reward_total_divergence": reward_total_divergence,
        "reward_term_divergence": reward_term_divergence,
        "reward_by_term": {
            "names": reward_names,
            "common_through_model1_terminal": {
                "model0": common0,
                "model1": common1,
                "model1_minus_model0": [b - a for a, b in zip(common0, common1, strict=True)],
            },
            "preterminal_common_prefix": {
                "model0": pre0,
                "model1": pre1,
                "model1_minus_model0": [b - a for a, b in zip(pre0, pre1, strict=True)],
            },
            "model0_only_suffix": suffix0,
        },
        "target_reward_terms": term_evidence,
        "terminal_ee_culprit": {
            "model0": _terminal_ee_culprit(model0),
            "model1": _terminal_ee_culprit(model1),
        },
        "model0_only_suffix": model0_suffix,
    }


def _boundary_receipt(*, simulator_constructed: bool | None) -> dict[str, Any]:
    receipt = {
        "diagnostic_only": True,
        "failed_model5_loaded": False,
        "failed_model5_resumed": False,
        "optimizer_steps": 0,
        "training_transitions": 0,
        "checkpoints_written": 0,
        "candidate_selected": False,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
        "robot_or_network_commands_permitted": False,
    }
    if simulator_constructed is not None:
        receipt["simulator_constructed"] = simulator_constructed
    return receipt


def execute_checkpoint_trace(repository_root: Path, parent_run_dir: Path) -> dict[str, Any]:
    """Execute two fresh deterministic evaluations; never optimize or train."""

    inputs = resolve_trace_inputs(repository_root, parent_run_dir)
    starting_snapshot = inputs.provenance["snapshot_sha256"]
    checkpoints = _load_and_validate_checkpoints(inputs)
    expected = inputs.contract["expected_reproductions"]
    from gear_sonic.trl.mjlab.true23_actor import True23SonicActorModel

    topology = _bound_input_paths(inputs.repository_root)["actor/topology.pt"]
    actor = True23SonicActorModel(
        {
            "tokenizer": torch.zeros((1, 268), dtype=torch.float32),
            "policy": torch.zeros((1, 930), dtype=torch.float32),
        },
        {"actor": ["tokenizer", "policy"]},
        "actor",
        ACTION_DIM,
        warm_start_path=str(topology),
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.1,
            "std_type": "scalar",
        },
        hidden_dims=(),
        activation="silu",
        obs_normalization=False,
    ).to(DEVICE)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment
    from gear_sonic.trl.mjlab.sonic_task_space_ppo_runner import (
        audit_task_space_ppo_env_cfg,
        make_task_space_ppo_env_cfg,
    )

    motion = _bound_input_paths(inputs.repository_root)["motion/B_DadDance.npz"]
    traces: dict[int, Mapping[str, Any]] = {}
    return_diagnostics: dict[int, Mapping[str, Any]] = {}
    common_layout: Mapping[str, Any] | None = None
    common_audit: Mapping[str, Any] | None = None
    common_prime: Mapping[str, Any] | None = None
    for update in EXPECTED_UPDATES:
        if _current_provenance_snapshot(inputs) != starting_snapshot:
            raise RuntimeError(f"trace provenance changed before model{update}")
        _load_actor_policy(actor, checkpoints[update], update, expected[str(update)]["policy_state_sha256"])
        _seed_everything()
        cfg = make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=1)
        cfg.seed = FIXED_SEED
        audit = audit_task_space_ppo_env_cfg(cfg, expected_num_envs=1)
        env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
            prime = prime_sonic_true23_training_environment(wrapped)
            parent_evaluation = inputs.parent_json[f"evaluations/evaluation_update_{update}.json"]
            if prime != parent_evaluation["prime"] or audit != parent_evaluation["task_audit"]:
                raise ValueError(f"model{update} env/prime differs from parent evaluation")
            executed = run_policy_trace(
                policy=actor,
                wrapped_env=wrapped,
                update_count=update,
                expected=expected[str(update)],
                contract=inputs.contract,
            )
        finally:
            env.close()
        if common_layout is None:
            common_layout, common_audit, common_prime = executed["layout"], audit, prime
        elif executed["layout"] != common_layout or audit != common_audit or prime != common_prime:
            raise ValueError("model0/model1 trace environment/layout mismatch")
        traces[update] = executed["trace"]
        return_diagnostics[update] = executed["episode_return_diagnostic"]
        if _current_provenance_snapshot(inputs) != starting_snapshot:
            raise RuntimeError(f"trace provenance changed after model{update}")
    assert common_layout is not None and common_audit is not None and common_prime is not None
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRACE_KIND,
        "purpose": inputs.contract["purpose"],
        "evaluation_seed": FIXED_SEED,
        "parent_run_dir": str(inputs.run_dir),
        "provenance": inputs.provenance,
        "capture_contract": CAPTURE_CONTRACT,
        "layout": common_layout,
        "task_audit": common_audit,
        "prime": common_prime,
        "traces": {"model0": traces[0], "model1": traces[1]},
        "parent_episode_return_diagnostics": {
            "model0": return_diagnostics[0],
            "model1": return_diagnostics[1],
        },
        "comparison": compare_trace_pair(traces[0], traces[1], common_layout, inputs.contract),
        **_boundary_receipt(simulator_constructed=True),
    }
    assert_publication_boundary(report)
    if _current_provenance_snapshot(inputs) != starting_snapshot:
        raise RuntimeError("trace provenance changed before publication")
    return report


def validate_publication_provenance(repository_root: Path, parent_run_dir: Path, report: Mapping[str, Any]) -> str:
    """Rebind immutable inputs/sources immediately around publication."""

    expected = _sha256(
        _mapping(report.get("provenance"), "trace report provenance").get("snapshot_sha256"),
        "trace report provenance snapshot",
    )
    inputs = resolve_trace_inputs(repository_root, parent_run_dir)
    actual = _current_provenance_snapshot(inputs)
    if actual != expected or inputs.provenance["snapshot_sha256"] != expected:
        raise RuntimeError("trace publication provenance changed")
    return actual


def _assert_scalar_evidence(value: Any, path: str = "partial_evidence") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{path} key mismatch")
            _assert_scalar_evidence(child, f"{path}.{key}")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{path} must contain scalar JSON evidence only")


def assert_publication_boundary(value: Any) -> None:
    forbidden_false = {
        "failed_model5_loaded",
        "failed_model5_resumed",
        "candidate_selected",
        "teacher_labels_used",
        "support_qualified",
        "promotion_eligible",
        "hardware_authorized",
        "deployment_ready",
        "robot_or_network_commands_permitted",
    }

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError(f"trace publication key mismatch at {path}")
                if key in forbidden_false and child is not False:
                    raise ValueError(f"trace boundary violated at {path}.{key}")
                if key in {"optimizer_steps", "training_transitions", "checkpoints_written"} and child != 0:
                    raise ValueError(f"trace mutation boundary violated at {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif item is None or isinstance(item, (str, bool, int)):
            return
        elif isinstance(item, float) and math.isfinite(item):
            return
        else:
            raise ValueError(f"trace publication contains unsupported value at {path}")

    visit(value, "trace")


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    assert_publication_boundary(value)
    output = path.expanduser().absolute()
    _regular_directory(output.parent, "trace output parent")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite full-support trace output: {output}")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def failure_report(
    error: Exception,
    *,
    failure_stage: str = "unattributed_failure",
    parent_run_dir: str | None = None,
    provenance_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    if type(failure_stage) is not str or not failure_stage:
        raise ValueError("failure stage must be non-empty string")
    if provenance_snapshot_sha256 is not None:
        _sha256(provenance_snapshot_sha256, "failure provenance snapshot")
    if parent_run_dir is not None and (type(parent_run_dir) is not str or not parent_run_dir):
        raise ValueError("failure parent run must be non-empty string")
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": FAILURE_KIND,
        "trace_contract_sha256": TRACE_CONTRACT_SHA256,
        "failure_stage": failure_stage,
        "parent_run_dir": parent_run_dir,
        "preexecution_provenance_snapshot_sha256": provenance_snapshot_sha256,
        "error": {"type": type(error).__name__, "message": str(error)},
        "simulator_trace_complete": False,
        **_boundary_receipt(simulator_constructed=None),
    }
    partial = getattr(error, "partial_evidence", None)
    if partial is not None:
        _assert_scalar_evidence(partial)
        report["partial_scalar_evidence"] = dict(partial)
    assert_publication_boundary(report)
    return report


__all__ = [
    "OUTPUT_FILENAME",
    "TRACE_CONTRACT_SHA256",
    "TraceReproductionError",
    "assert_publication_boundary",
    "compare_trace_pair",
    "execute_checkpoint_trace",
    "failure_report",
    "load_trace_contract",
    "resolve_trace_inputs",
    "reward_internal_identity",
    "run_policy_trace",
    "trace_preflight",
    "validate_publication_provenance",
    "validate_structural_reproduction",
    "validate_trace_series",
    "write_json_exclusive",
]
