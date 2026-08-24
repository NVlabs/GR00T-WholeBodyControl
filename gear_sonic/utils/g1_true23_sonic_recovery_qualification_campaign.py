"""Fail-closed cutoff-50 recovery qualification campaign.

This additive runner reuses the sealed recovery-v3 simulator diagnostic.  It
changes no recovery-v1/v2/v3 source file.  A scoped wrapper interposition adds
one exact world-frame root-link velocity delta on disturbance runs after both
policies infer and immediately before the global-transition-100 environment
step.  All twenty runs are qualification-only: this module publishes compact
scalar evidence and hashes, never observations, actions, labels, or NPZ data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any

import numpy as np

from gear_sonic.utils import (
    g1_true23_sonic_student_closed_loop_qualification as student,
    g1_true23_sonic_student_teacher_recovery as recovery,
)

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_sonic_recovery_qualification_campaign_contract_v1"
REPORT_KIND = "g1_true23_sonic_recovery_qualification_campaign_report_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_recovery_qualification_campaign_v1.json"
)
CONTRACT_SHA256 = "2e24e3acf1d9ef7e1cf664de64935773a99ad225e8046f5b23f413932e21fad3"

TOTAL_TRANSITIONS = 510
RECOVERY_ROLE = "offline_mjlab_cutoff50_mixed_controller_qualification_only"
RECOVERY_MODE = "cutoff50"
STUDENT_TRANSITIONS = 50
TEACHER_TRANSITIONS = 460
STUDENT_FIRST_Q9 = 9
STUDENT_LAST_Q9 = 58
TEACHER_FIRST_Q9 = 59
TEACHER_LAST_Q9 = 518
GLOBAL_IMPULSE_TRANSITION = 100
IMPULSE_Q9 = 109
FIRST_DISTURBED_OBSERVATION_TRANSITION = 101
FIRST_DISTURBED_OBSERVATION_Q9 = 110
BASELINE_TRANSITIONS = tuple(range(90, 100))
RECOVERY_SCAN_FIRST = 100
RECOVERY_STABLE_STEPS = 5
CONTROL_DT_S = 0.02
PREREQUISITE_REPORT_NAME = "cutoff50_pass_report"
PREREQUISITE_REPORT_PATH_KEY = "cutoff50_pass_report_relative_path"
PREREQUISITE_REPORT_SHA_KEY = "cutoff50_pass_report_sha256"
PREREQUISITE_REPORT_OUTPUT_KEY = "cutoff50_prerequisite_report_sha256"

EXECUTED_SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_sonic_recovery_qualification_campaign.py"),
    Path("gear_sonic/scripts/qualify_g1_true23_sonic_recovery_campaign.py"),
)

EXPECTED_V3_GATE_KEYS = frozenset(
    {
        "complete_window",
        "student_inferred_every_transition",
        "teacher_inferred_every_transition",
        "teacher_pt_onnx_parity",
        "controller_partition_exact",
        "only_final_timeout",
        "history_exact",
        "actual_v11_action_chain_exact",
        "teacher_composite_chain_exact",
        "selected_external_state_exact",
        "nominal_safety_gate",
        "frozen_inputs_unchanged",
        "sources_assets_contracts_unchanged",
        "no_partial_failure",
    }
)

RUN_RECORD_KEYS = frozenset(
    {
        "scenario",
        "rollout_index",
        "seed",
        "run_id",
        "passed",
        "issue_count",
        "issues_sha256",
        "first_issue",
        "failure_detail_sha256",
        "v3_report_sha256",
        "v3_verdict",
        "step_calls_started",
        "transitions_completed",
        "attempted_transitions",
        "student_controller_count",
        "teacher_controller_count",
        "student_inference_count",
        "teacher_query_count",
        "terminal_transition",
        "terminal_q9",
        "final_timeout_only",
        "history_check_count",
        "history_shift_count",
        "v3_gate_count",
        "v3_false_gate_count",
        "hard_safety_violation_count",
        "soft_safety_warning_count",
        "rollout_required_zero_total",
        "support_required_zero_total",
        "minimum_base_height_m",
        "maximum_base_tilt_rad",
        "maximum_joint_velocity_ratio",
        "maximum_tracking_rmse_rad",
        "maximum_actuator_force_ratio",
        "maximum_plain_sonic_raw_native_abs",
        "teacher_parity_max_absolute_error",
        "action_chain_max_absolute_error",
        "effective_seed",
        "effective_seed_sha256",
        "disturbance_performed",
        "impulse_count",
        "impulse_apply_transition",
        "impulse_apply_q9",
        "planned_impulse_sha256",
        "target_root_qvel_sha256",
        "realized_impulse_sha256",
        "impulse_qvel_write_exact",
        "impulse_readback_max_absolute_error",
        "metric_sample_count",
        "metric_trajectory_sha256",
        "baseline_mean",
        "recovery_threshold",
        "first_threshold_crossing_transition",
        "stable_streak_start_transition",
        "stable_streak_end_transition",
        "recovery_time_s",
        "recovered",
        "published_teacher_label_count",
        "published_training_row_count",
    }
)

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "campaign_qualified",
        "verdict",
        "campaign_contract_sha256",
        PREREQUISITE_REPORT_OUTPUT_KEY,
        "candidate_manifest_sha256",
        "teacher_onnx_sha256",
        "motion_sha256",
        "planned_run_count",
        "completed_run_count",
        "passed_run_count",
        "nominal_passed_count",
        "disturbance_passed_count",
        "disturbance_recovered_count",
        "remaining_run_count",
        "run_order_sha256",
        "effective_seed_set_sha256",
        "all_effective_seeds_unique",
        "first_failed_run_id",
        "first_failure",
        "failure_detail_sha256",
        "preflight_ready",
        "preflight_issue_count",
        "executed_source_binding_before_sha256",
        "executed_source_binding_after_sha256",
        "source_rehash_succeeded",
        "source_rehash_error_sha256",
        "sources_unchanged",
        "runs",
        "published_teacher_label_count",
        "published_training_row_count",
        "fresh_disjoint_suffix_collection_authorized",
        "teacher_support_qualified",
        "dagger_data",
        "training_performed",
        "promotion_or_deployment",
        "hardware_authorized",
    }
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be mapping")
    return value


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{context} must be finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite number")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be integer")
    return value


def _bounded_public_failure(value: Any, context: str) -> str:
    allowed = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _:-.")
    if type(value) is not str or not 1 <= len(value) <= 160 or any(char not in allowed for char in value):
        raise ValueError(f"{context} must be bounded scalar failure code")
    return value


def _quat_apply_wxyz(quat: Any, vector: Any, *, inverse: bool) -> Any:
    """Match MJLab's wxyz quaternion vector rotation without importing MJLab."""

    shape = vector.shape
    quat = quat.reshape(-1, 4)
    vector = vector.reshape(-1, 3)
    xyz = quat[:, 1:]
    cross = xyz.cross(vector, dim=-1) * 2
    signed = -quat[:, 0:1] * cross if inverse else quat[:, 0:1] * cross
    return (vector + signed + xyz.cross(cross, dim=-1)).view(shape)


def _strict_json(path: Path, context: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{context} contains nonfinite constant: {value}")

    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {context}: {error}") from error
    return _mapping(result, context)


def _regular_repo_file(root: Path, relative: str, context: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{context} path must be repository-relative")
    candidate = root / value
    resolved = candidate.resolve(strict=True)
    if (
        not resolved.is_relative_to(root)
        or candidate.is_symlink()
        or resolved.is_symlink()
        or not resolved.is_file()
    ):
        raise ValueError(f"{context} must be regular repository file")
    return resolved


@dataclass(frozen=True)
class CampaignRunSpec:
    scenario: str
    rollout_index: int
    seed: int
    impulse: tuple[float, ...] | None

    def __post_init__(self) -> None:
        if self.scenario not in {"nominal", "disturbance"}:
            raise ValueError("campaign scenario must be nominal or disturbance")
        if not 0 <= self.rollout_index < 10:
            raise ValueError("campaign rollout index outside 0..9")
        if not 0 <= self.seed < 2**31:
            raise ValueError("campaign seed must be uint31")
        if self.scenario == "nominal" and self.impulse is not None:
            raise ValueError("nominal campaign run cannot carry impulse")
        if self.scenario == "disturbance":
            if self.impulse is None or len(self.impulse) != 6:
                raise ValueError("disturbance campaign run requires six-axis impulse")
            if not all(math.isfinite(value) for value in self.impulse):
                raise ValueError("campaign impulse must be finite")

    @property
    def run_id(self) -> str:
        return f"{self.scenario}_{self.rollout_index:02d}_seed{self.seed}"

    def descriptor(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "rollout_index": self.rollout_index,
            "seed": self.seed,
            "run_id": self.run_id,
            "planned_impulse_sha256": (
                None if self.impulse is None else _sha256_bytes(_canonical_json_bytes(list(self.impulse)))
            ),
        }


@dataclass(frozen=True)
class CampaignRequest:
    repository_root: Path
    output: Path

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root, Path) or not isinstance(self.output, Path):
            raise TypeError("campaign request paths must be pathlib.Path")

    @property
    def root(self) -> Path:
        value = self.repository_root.expanduser().resolve(strict=True)
        if value.is_symlink() or not value.is_dir():
            raise ValueError("campaign repository root must be regular directory")
        return value

    @property
    def output_path(self) -> Path:
        raw = self.output.expanduser()
        candidate = raw if raw.is_absolute() else self.root / raw
        value = candidate.resolve(strict=False)
        if not value.is_relative_to(self.root) or value.suffix.lower() != ".json":
            raise ValueError("campaign output must be repository-contained .json")
        if candidate.is_symlink() or value.is_symlink():
            raise ValueError("campaign output cannot be symlink")
        parent = value.parent.resolve(strict=True)
        if parent.is_symlink() or not parent.is_dir() or not parent.is_relative_to(self.root):
            raise ValueError("campaign output parent must be regular repository directory")
        return value


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("role") != RECOVERY_ROLE
    ):
        raise ValueError("campaign contract identity mismatch")
    window = _mapping(contract.get("window"), "campaign window")
    expected_window = {
        "mode": RECOVERY_MODE,
        "anchor_q9": 9,
        "last_action_q9": 518,
        "total_transitions": TOTAL_TRANSITIONS,
        "student_transition_count": STUDENT_TRANSITIONS,
        "student_first_q9": STUDENT_FIRST_Q9,
        "student_last_q9": STUDENT_LAST_Q9,
        "teacher_transition_count": TEACHER_TRANSITIONS,
        "teacher_first_q9": TEACHER_FIRST_Q9,
        "teacher_last_q9": TEACHER_LAST_Q9,
        "control_hz": 50.0,
        "final_timeout_only": True,
    }
    if dict(window) != expected_window:
        raise ValueError("campaign recovery window mismatch")
    cohort = _mapping(contract.get("cohort"), "campaign cohort")
    nominal = cohort.get("nominal_seeds")
    disturbance = cohort.get("disturbance_seeds")
    if (
        cohort.get("scenario_order") != ["nominal", "disturbance"]
        or cohort.get("execution_order") != "alternating_index_order_nominal_i_then_disturbance_i"
        or cohort.get("rollouts_per_scenario") != 10
        or cohort.get("total_rollouts") != 20
        or cohort.get("environment_rebuilt_per_rollout") is not True
        or cohort.get("preflight_once_before_first_rollout") is not True
        or cohort.get("fail_fast") is not True
        or not isinstance(nominal, list)
        or not isinstance(disturbance, list)
        or len(nominal) != 10
        or len(disturbance) != 10
        or len(set(nominal + disturbance)) != 20
        or cohort.get("support_v1_reuse_scope")
        != "seeds_only_not_its_500_step_support_qualification_or_tranche_admission"
        or cohort.get("effective_seed_must_equal_requested_seed") is not True
        or cohort.get("effective_seed_digests_must_be_unique") is not True
    ):
        raise ValueError("campaign cohort/seed contract mismatch")
    rng = cohort.get("per_rollout_rng_application")
    if rng != [
        "random.seed(run_seed)",
        "numpy.random.seed(run_seed_mod_2_pow_32)",
        "torch.manual_seed(run_seed)",
        "torch.cuda.manual_seed_all(run_seed)",
        "environment_cfg.seed=run_seed",
    ]:
        raise ValueError("campaign RNG application contract mismatch")
    disturbance_cfg = _mapping(contract.get("disturbance"), "campaign disturbance")
    metric = _mapping(disturbance_cfg.get("recovery_metric"), "campaign recovery metric")
    derivation = _mapping(
        disturbance_cfg.get("vector_derivation"),
        "campaign disturbance vector derivation",
    )
    if (
        disturbance_cfg.get("scope") != "teacher_controlled_suffix"
        or disturbance_cfg.get("teacher_local_apply_transition") != 50
        or disturbance_cfg.get("global_apply_transition") != GLOBAL_IMPULSE_TRANSITION
        or disturbance_cfg.get("apply_q9") != IMPULSE_Q9
        or disturbance_cfg.get("ordering")
        != "after_student_and_teacher_inference_before_action_processing_and_physics"
        or disturbance_cfg.get("first_disturbed_observation_global_transition")
        != FIRST_DISTURBED_OBSERVATION_TRANSITION
        or disturbance_cfg.get("first_disturbed_observation_q9") != FIRST_DISTURBED_OBSERVATION_Q9
        or disturbance_cfg.get("baseline_global_transitions") != list(BASELINE_TRANSITIONS)
        or disturbance_cfg.get("baseline_step_count") != 10
        or disturbance_cfg.get("stable_recovery_steps") != RECOVERY_STABLE_STEPS
        or disturbance_cfg.get("recovery_margin") != 0.1
        or disturbance_cfg.get("maximum_recovery_time_s") != 2.0
        or disturbance_cfg.get("readback_max_absolute_error") != 1.0e-6
    ):
        raise ValueError("campaign disturbance schedule mismatch")
    formula = (
        "base_tilt_rad+abs(reference_anchor_z_m-robot_anchor_z_m)+"
        "sqrt(mean((applied_target_hardware_rad-"
        "measured_joint_position_hardware_rad)^2))"
    )
    if (
        metric.get("callable")
        != ("gear_sonic.envs.mjlab.sonic_true23_causal_history_safe_target_v11:evaluator_aligned_recovery_metric")
        or metric.get("formula") != formula
        or metric.get("formula_sha256") != _sha256_bytes(formula.encode("utf-8"))
        or metric.get("reward_term_name") != "evaluator_aligned_recovery"
        or metric.get("reward_weight") != -25.0
        or metric.get("extraction") != "weighted_reward_rate_divided_by_negative_25"
        or metric.get("required_finite_sample_count") != TOTAL_TRANSITIONS
        or metric.get("threshold_formula")
        != f"mean(metric[global{BASELINE_TRANSITIONS[0]}:global{BASELINE_TRANSITIONS[-1] + 1}])+0.1"
        or metric.get("scan_first_global_transition") != RECOVERY_SCAN_FIRST
        or metric.get("recovery_time_formula")
        != f"(stable_streak_end_global_transition-{GLOBAL_IMPULSE_TRANSITION}+1)*0.02"
    ):
        raise ValueError("campaign recovery metric contract mismatch")
    if (
        derivation.get("algorithm") != "sha256_first_u64_uniform_v1"
        or derivation.get("episode_index") != 0
        or derivation.get("axis_order") != ["x", "y", "z", "roll", "pitch", "yaw"]
        or derivation.get("linear_velocity_abs_max_m_s") != [0.5, 0.5, 0.2]
        or derivation.get("angular_velocity_abs_max_rad_s") != [0.52, 0.52, 0.78]
        or derivation.get("world_frame") is not True
        or derivation.get("root_link_body_origin") is not True
        or ".digest()[:8]" not in str(derivation.get("formula"))
    ):
        raise ValueError("campaign disturbance derivation mismatch")
    report = _mapping(contract.get("report"), "campaign report contract")
    boundaries = _mapping(contract.get("boundaries"), "campaign boundaries")
    if (
        report.get("compact_aggregates_and_hashes_only") is not True
        or report.get("npz_output_permitted") is not False
        or report.get("published_teacher_label_count") != 0
        or report.get("published_training_row_count") != 0
        or report.get("strict_report_schema_allowlist_required") is not True
        or report.get("arbitrary_json_lists_or_mappings_permitted") is not False
        or report.get("disturbance_vectors_in_report_permitted") is not False
        or boundaries.get("qualification_only") is not True
        or any(
            boundaries.get(name) is not False
            for name in (
                "teacher_labels_admitted",
                "teacher_support_qualified",
                "dagger_data",
                "training_performed",
                "training_export_permitted",
                "promotion_eligible",
                "deployment_ready",
                "hardware_authorized",
                "robot_or_network_commands_permitted",
            )
        )
    ):
        raise ValueError("campaign report/safety boundary mismatch")


def load_campaign_contract(repository_root: str | Path | None = None) -> Mapping[str, Any]:
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).expanduser().resolve(strict=True)
    )
    path = _regular_repo_file(root, CONTRACT_RELATIVE_PATH.as_posix(), "campaign contract")
    if _sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("campaign contract SHA256 mismatch")
    contract = _strict_json(path, "campaign contract")
    _validate_contract(contract)
    return contract


def _deterministic_vector(seed: int, contract: Mapping[str, Any]) -> tuple[float, ...]:
    disturbance = _mapping(contract["disturbance"], "campaign disturbance")
    derivation = _mapping(disturbance["vector_derivation"], "vector derivation")
    linear = derivation["linear_velocity_abs_max_m_s"]
    angular = derivation["angular_velocity_abs_max_rad_s"]
    maxima = [*linear, *angular]
    result: list[float] = []
    for axis, maximum in enumerate(maxima):
        digest = hashlib.sha256(f"g1_23dof_rev_1_0:{seed}:0:{axis}".encode("ascii")).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        result.append(round(-float(maximum) + unit * (2.0 * float(maximum)), 12))
    exact = _mapping(disturbance["exact_vectors_by_seed"], "exact disturbance vectors")
    expected = exact.get(str(seed))
    if expected is None or result != expected:
        raise ValueError(f"campaign disturbance vector mismatch for seed {seed}")
    return tuple(result)


def campaign_run_specs(
    repository_root: str | Path | None = None,
) -> tuple[CampaignRunSpec, ...]:
    contract = load_campaign_contract(repository_root)
    cohort = _mapping(contract["cohort"], "campaign cohort")
    nominal = [_integer(value, "nominal seed") for value in cohort["nominal_seeds"]]
    disturbed = [_integer(value, "disturbance seed") for value in cohort["disturbance_seeds"]]
    result: list[CampaignRunSpec] = []
    for index in range(10):
        result.append(CampaignRunSpec("nominal", index, nominal[index], None))
        result.append(
            CampaignRunSpec(
                "disturbance",
                index,
                disturbed[index],
                _deterministic_vector(disturbed[index], contract),
            )
        )
    if len({spec.seed for spec in result}) != 20:
        raise RuntimeError("campaign run seeds are not globally unique")
    return tuple(result)


def executed_campaign_source_binding(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative in EXECUTED_SOURCE_RELATIVE_PATHS:
        path = _regular_repo_file(root, relative.as_posix(), "campaign executed source")
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "binding_sha256": _sha256_bytes(_canonical_json_bytes(records)),
        "files": records,
    }


def _verify_prerequisites(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    prerequisites = _mapping(contract["prerequisites"], "campaign prerequisites")
    pairs = (
        ("recovery_contract", "recovery_contract_relative_path", "recovery_contract_sha256"),
        (
            PREREQUISITE_REPORT_NAME,
            PREREQUISITE_REPORT_PATH_KEY,
            PREREQUISITE_REPORT_SHA_KEY,
        ),
        ("support_contract", "support_contract_relative_path", "support_contract_sha256"),
        (
            "candidate_manifest",
            "candidate_manifest_relative_path",
            "candidate_manifest_sha256",
        ),
        ("motion", "motion_relative_path", "motion_sha256"),
    )
    files: dict[str, dict[str, Any]] = {}
    for name, path_key, hash_key in pairs:
        expected = _require_sha256(prerequisites[hash_key], f"{name} expected SHA256")
        path = _regular_repo_file(root, prerequisites[path_key], name)
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"campaign prerequisite SHA256 mismatch: {name}")
        files[name] = {"path": path, "sha256": actual}
    metric = _mapping(contract["disturbance"]["recovery_metric"], "recovery metric")
    metric_path = _regular_repo_file(root, metric["source_relative_path"], "recovery metric source")
    if _sha256_file(metric_path) != metric["source_sha256"]:
        raise ValueError("campaign recovery metric source SHA256 mismatch")
    prerequisite_report = _strict_json(
        files[PREREQUISITE_REPORT_NAME]["path"],
        "recovery prerequisite",
    )
    support_contract = _strict_json(files["support_contract"]["path"], "support-v1 prerequisite")
    support_qualification = _mapping(
        support_contract.get("qualification"),
        "support-v1 qualification",
    )
    support_disturbance = _mapping(
        support_qualification.get("disturbance"),
        "support-v1 disturbance",
    )
    cohort = _mapping(contract["cohort"], "campaign cohort")
    if (
        support_contract.get("kind") != "g1_true23_native124_21204_support_contract_v1"
        or support_qualification.get("nominal_seeds") != cohort["nominal_seeds"]
        or support_qualification.get("disturbance_seeds") != cohort["disturbance_seeds"]
        or support_qualification.get("steps_per_rollout") != 500
        or support_disturbance.get("apply_step") != 50
        or support_disturbance.get("baseline_steps") != 10
        or support_disturbance.get("stable_recovery_steps") != RECOVERY_STABLE_STEPS
        or support_disturbance.get("recovery_margin") != 0.1
        or support_qualification.get("maximum_recovery_time_s") != 2.0
        or support_disturbance.get("linear_velocity_abs_max_m_s") != [0.5, 0.5, 0.2]
        or support_disturbance.get("angular_velocity_abs_max_rad_s") != [0.52, 0.52, 0.78]
    ):
        raise ValueError("support-v1 seed/disturbance provenance mismatch")
    rollout = _mapping(prerequisite_report.get("rollout"), "recovery prerequisite rollout")
    claims = _mapping(prerequisite_report.get("claims"), "recovery prerequisite claims")
    if (
        prerequisite_report.get("schema_version") != recovery.RECOVERY_SCHEMA_VERSION
        or prerequisite_report.get("kind") != recovery.RECOVERY_KIND
        or prerequisite_report.get("verdict") != "mixed_controller_recovery_passed"
        or prerequisite_report.get("recovered_to_original_q9_518_boundary") is not True
        or claims.get("mode") != RECOVERY_MODE
        or rollout.get("hard_safety_violation_count") != 0
        or rollout.get("soft_safety_warning_count") != 0
    ):
        raise ValueError("recovery prerequisite is not exact safe PASS")
    return {
        "files": files,
        "prerequisite_report": prerequisite_report,
        # Backward-compatible alias for the historical cutoff-50 campaign tests.
        "cutoff": prerequisite_report,
    }


def _base_recovery_request(request: CampaignRequest) -> recovery.RecoveryRequest:
    root = request.root
    return recovery.RecoveryRequest(
        repository_root=root,
        candidate_manifest=root / recovery.CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH,
        expected_candidate_manifest_sha256=recovery.CURRENT_CANDIDATE_MANIFEST_SHA256,
        output=root / "artifacts/g1_true23/.campaign-inner-report-unused.json",
        mode=RECOVERY_MODE,
    )


def _preflight_internal(request: CampaignRequest) -> dict[str, Any]:
    if type(request) is not CampaignRequest:
        raise TypeError("request must be exact CampaignRequest")
    root = request.root
    contract = load_campaign_contract(root)
    prerequisites = _verify_prerequisites(root, contract)
    sources = executed_campaign_source_binding(root)
    base_request = _base_recovery_request(request)
    recovery_preflight = recovery._preflight_internal(base_request)  # noqa: SLF001
    issues = list(recovery_preflight.get("issues", ()))
    ready = recovery_preflight.get("ready") is True and not issues
    if not ready and not issues:
        issues.append("sealed recovery preflight not ready without diagnostic issue")
    return {
        "ready": ready,
        "issues": issues,
        "root": root,
        "contract": contract,
        "prerequisites": prerequisites,
        "sources": sources,
        "base_request": base_request,
        "recovery_preflight": recovery_preflight,
    }


def preflight_campaign(request: CampaignRequest) -> dict[str, Any]:
    value = _preflight_internal(request)
    prerequisites = _mapping(value["contract"]["prerequisites"], "prerequisites")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "g1_true23_sonic_recovery_qualification_campaign_preflight_v1",
        "ready": value["ready"],
        "issue_count": len(value["issues"]),
        "campaign_contract_sha256": CONTRACT_SHA256,
        PREREQUISITE_REPORT_OUTPUT_KEY: prerequisites[PREREQUISITE_REPORT_SHA_KEY],
        "planned_run_count": 20,
        "executed_source_binding_sha256": value["sources"]["binding_sha256"],
        "simulator_constructed": False,
        "published_teacher_label_count": 0,
        "published_training_row_count": 0,
        "hardware_authorized": False,
    }


class _StepProbe:
    """Capture scalar seed, impulse, and exact reward-term recovery evidence."""

    def __init__(self, spec: CampaignRunSpec, contract: Mapping[str, Any]) -> None:
        self.spec = spec
        self.contract = contract
        self.step_calls_started = 0
        self.transitions_completed = 0
        self.effective_seed: int | None = None
        self.impulse_count = 0
        self.impulse_apply_transition: int | None = None
        self.impulse_apply_q9: int | None = None
        self.target_root_qvel: np.ndarray | None = None
        self.realized_root_qvel: np.ndarray | None = None
        self.qvel_write_exact: bool | None = None
        self.readback_error: float | None = None
        self.metric_digest = hashlib.sha256()
        self.metric_values: list[float] = []
        self.reward_metric_bound = False

    def on_construct(self, wrapper: Any) -> None:
        value = getattr(wrapper.unwrapped.cfg, "seed", None)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError("campaign environment effective seed is not integer")
        self.effective_seed = value
        reward_cfg = wrapper.unwrapped.reward_manager.get_term_cfg("evaluator_aligned_recovery")
        callable_name = f"{reward_cfg.func.__module__}:{reward_cfg.func.__qualname__}"
        expected = self.contract["disturbance"]["recovery_metric"]
        if callable_name != expected["callable"] or float(reward_cfg.weight) != -25.0:
            raise RuntimeError("campaign recovery reward term binding drift")
        self.reward_metric_bound = True

    def before_step(self, wrapper: Any) -> None:
        if self.spec.scenario != "disturbance" or self.transitions_completed != GLOBAL_IMPULSE_TRANSITION:
            return
        if self.impulse_count:
            raise RuntimeError("campaign disturbance applied more than once")
        import torch

        try:
            from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
        except ModuleNotFoundError:  # Unit-test fallback; simulator runtime has MJLab.
            quat_apply = lambda quat, vector: _quat_apply_wxyz(  # noqa: E731
                quat, vector, inverse=False
            )
            quat_apply_inverse = lambda quat, vector: _quat_apply_wxyz(  # noqa: E731
                quat, vector, inverse=True
            )

        robot = wrapper.unwrapped.scene["robot"]
        before = robot.data.root_link_vel_w
        if (
            type(before) is not torch.Tensor
            or before.shape != (1, 6)
            or before.dtype != torch.float32
            or not bool(torch.isfinite(before).all())
        ):
            raise RuntimeError("campaign root velocity must be finite [1,6]")
        before = before.detach().clone()
        delta = torch.as_tensor(
            self.spec.impulse,
            dtype=before.dtype,
            device=before.device,
        ).reshape(1, 6)
        target = before + delta
        env_ids = torch.zeros(1, dtype=torch.long, device=before.device)
        entity_data = robot.data
        qpos = entity_data.data.qpos
        qvel = entity_data.data.qvel
        q_indices = entity_data.indexing.free_joint_q_adr
        v_indices = entity_data.indexing.free_joint_v_adr
        env_rows = env_ids[:, None]
        quat_w = qpos[env_rows, q_indices[3:7]].detach().clone()
        if quat_w.shape != (1, 4) or quat_w.dtype != torch.float32 or not bool(torch.isfinite(quat_w).all()):
            raise RuntimeError("campaign root quaternion must be finite float32 [1,4]")
        target_qvel = torch.cat(
            (target[:, :3], quat_apply_inverse(quat_w, target[:, 3:])),
            dim=-1,
        )
        robot.write_root_link_velocity_to_sim(target, env_ids=env_ids)
        after_qvel = qvel[env_rows, v_indices].detach().clone()
        if (
            after_qvel.shape != (1, 6)
            or after_qvel.dtype != torch.float32
            or not bool(torch.isfinite(after_qvel).all())
        ):
            raise RuntimeError("campaign root qvel readback must be finite [1,6]")
        after_world = torch.cat(
            (after_qvel[:, :3], quat_apply(quat_w, after_qvel[:, 3:])),
            dim=-1,
        )
        observed_delta = after_world.to(dtype=torch.float64) - before.to(dtype=torch.float64)
        expected_delta = delta.to(dtype=torch.float64)
        self.readback_error = float(torch.max(torch.abs(observed_delta - expected_delta)).item())
        self.target_root_qvel = (
            target_qvel.to(device="cpu")
            .contiguous()
            .numpy()[0]
            .astype(
                np.float32,
                copy=False,
            )
            .copy()
        )
        self.realized_root_qvel = (
            after_qvel.to(device="cpu").contiguous().numpy()[0].astype(np.float32, copy=False).copy()
        )
        self.qvel_write_exact = bool(torch.equal(after_qvel, target_qvel))
        command = wrapper.unwrapped.command_manager.get_term("motion")
        self.impulse_apply_transition = self.transitions_completed
        self.impulse_apply_q9 = recovery.evaluation._q9(command)  # noqa: SLF001
        self.impulse_count += 1

    def capture_evidence(self, evidence: Any) -> None:
        """Capture exact evidence object later consumed by v3 rollout accumulator."""

        if type(evidence) is not recovery.evaluation.StepEvidence:
            raise RuntimeError("campaign metric capture requires exact StepEvidence")
        weighted = _finite_float(
            evidence.reward_rates.get("evaluator_aligned_recovery"),
            "campaign weighted recovery rate",
        )
        metric = weighted / -25.0
        if metric < 0.0:
            raise RuntimeError("campaign recovery metric became negative")
        transition = len(self.metric_values)
        self.metric_values.append(metric)
        self.metric_digest.update(struct.pack("<qd", transition, metric))

    def before_super_step(self) -> None:
        self.step_calls_started += 1

    def after_step(self) -> None:
        self.transitions_completed += 1

    def report(self) -> dict[str, Any]:
        planned_sha = (
            None if self.spec.impulse is None else _sha256_bytes(_canonical_json_bytes(list(self.spec.impulse)))
        )
        target_qvel_sha = (
            None
            if self.target_root_qvel is None
            else _sha256_bytes(np.ascontiguousarray(self.target_root_qvel).tobytes(order="C"))
        )
        realized_sha = (
            None
            if self.realized_root_qvel is None
            else _sha256_bytes(np.ascontiguousarray(self.realized_root_qvel).tobytes(order="C"))
        )
        seed_sha = (
            None
            if self.effective_seed is None
            else _sha256_bytes(
                _canonical_json_bytes(
                    {
                        "requested_seed": self.spec.seed,
                        "effective_seed": self.effective_seed,
                    }
                )
            )
        )
        common = {
            "effective_seed": self.effective_seed,
            "effective_seed_sha256": seed_sha,
            "metric_sample_count": len(self.metric_values),
            "metric_trajectory_sha256": self.metric_digest.hexdigest(),
            "planned_impulse_sha256": planned_sha,
            "target_root_qvel_sha256": target_qvel_sha,
            "realized_impulse_sha256": realized_sha,
            "impulse_qvel_write_exact": self.qvel_write_exact,
            "impulse_count": self.impulse_count,
            "impulse_apply_transition": self.impulse_apply_transition,
            "impulse_apply_q9": self.impulse_apply_q9,
            "impulse_readback_max_absolute_error": self.readback_error,
            "reward_metric_bound": self.reward_metric_bound,
            "step_calls_started": self.step_calls_started,
            "transitions_completed": self.transitions_completed,
            "metric_sum": sum(self.metric_values),
            "metric_minimum": min(self.metric_values, default=None),
            "metric_maximum": max(self.metric_values, default=None),
        }
        if self.spec.scenario == "nominal":
            return {
                **common,
                "disturbance_performed": False,
                "baseline_mean": None,
                "recovery_threshold": None,
                "first_threshold_crossing_transition": None,
                "stable_streak_start_transition": None,
                "stable_streak_end_transition": None,
                "recovery_time_s": None,
                "recovered": None,
            }
        baseline_mean: float | None = None
        threshold: float | None = None
        first_crossing: int | None = None
        streak_start: int | None = None
        streak_end: int | None = None
        recovery_time: float | None = None
        recovered = False
        if len(self.metric_values) == TOTAL_TRANSITIONS:
            baseline = [self.metric_values[index] for index in BASELINE_TRANSITIONS]
            baseline_mean = sum(baseline) / len(baseline)
            threshold = baseline_mean + float(self.contract["disturbance"]["recovery_margin"])
            consecutive = 0
            candidate_start: int | None = None
            for transition in range(RECOVERY_SCAN_FIRST, TOTAL_TRANSITIONS):
                if self.metric_values[transition] <= threshold:
                    if first_crossing is None:
                        first_crossing = transition
                    if consecutive == 0:
                        candidate_start = transition
                    consecutive += 1
                    if consecutive == RECOVERY_STABLE_STEPS:
                        streak_start = candidate_start
                        streak_end = transition
                        break
                else:
                    consecutive = 0
                    candidate_start = None
            if streak_end is not None:
                recovery_time = (streak_end - GLOBAL_IMPULSE_TRANSITION + 1) * CONTROL_DT_S
                recovered = recovery_time <= float(self.contract["disturbance"]["maximum_recovery_time_s"])
        return {
            **common,
            "disturbance_performed": True,
            "baseline_mean": baseline_mean,
            "recovery_threshold": threshold,
            "first_threshold_crossing_transition": first_crossing,
            "stable_streak_start_transition": streak_start,
            "stable_streak_end_transition": streak_end,
            "recovery_time_s": recovery_time,
            "recovered": recovered,
        }


@contextmanager
def _scoped_runtime_interposition(
    *,
    spec: CampaignRunSpec,
    probe: _StepProbe,
    cached_preflight: Mapping[str, Any],
):
    """Patch only dynamic seed, cached preflight, and wrapper step; restore all."""

    import mjlab.rl as mjlab_rl

    original_wrapper = mjlab_rl.RslRlVecEnvWrapper
    original_seed = student.FIXED_SEED
    original_preflight = recovery._preflight_internal  # noqa: SLF001
    original_step_evidence = recovery.evaluation._step_evidence  # noqa: SLF001

    class InstrumentedWrapper(original_wrapper):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            probe.on_construct(self)

        def step(self, actions: Any) -> Any:
            probe.before_step(self)
            probe.before_super_step()
            result = super().step(actions)
            probe.after_step()
            return result

    def captured_step_evidence(raw_env: Any, velocity_limits: np.ndarray) -> Any:
        evidence = original_step_evidence(raw_env, velocity_limits)
        probe.capture_evidence(evidence)
        return evidence

    def cached(_request: recovery.RecoveryRequest) -> Mapping[str, Any]:
        return copy.deepcopy(dict(cached_preflight))

    mjlab_rl.RslRlVecEnvWrapper = InstrumentedWrapper
    student.FIXED_SEED = spec.seed
    recovery._preflight_internal = cached  # type: ignore[assignment]  # noqa: SLF001
    recovery.evaluation._step_evidence = captured_step_evidence  # type: ignore[assignment]  # noqa: SLF001
    try:
        yield
    finally:
        recovery.evaluation._step_evidence = original_step_evidence  # type: ignore[assignment]  # noqa: SLF001
        recovery._preflight_internal = original_preflight  # type: ignore[assignment]  # noqa: SLF001
        student.FIXED_SEED = original_seed
        mjlab_rl.RslRlVecEnvWrapper = original_wrapper


def _nested(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    candidate = value.get(name)
    return candidate if isinstance(candidate, Mapping) else {}


def _scalar_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return _finite_float(value, "compact scalar")
    except ValueError:
        return None


def _assess_run(
    *,
    spec: CampaignRunSpec,
    full_report: Mapping[str, Any],
    probe: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            issues.append(message)

    qualification = _nested(full_report, "qualification")
    v3_gates = _nested(qualification, "gates")
    rollout = _nested(full_report, "rollout")
    rollout_totals = _nested(rollout, "safety_count_totals")
    rollout_maxima = _nested(rollout, "safety_count_maximum_per_transition")
    rollout_reward_rates = _nested(rollout, "reward_rate_by_term")
    support = _nested(full_report, "support_summary")
    parity = _nested(full_report, "teacher_parity")
    action = _nested(full_report, "action_semantics")
    teacher_chain = _nested(full_report, "teacher_composite_action_chain")
    selected = _nested(full_report, "selected124_external_state")
    history = _nested(full_report, "history")
    history_autoreset = _nested(history, "autoreset")
    terminal = _nested(full_report, "first_done")
    safety = _nested(full_report, "safety")
    controller = _nested(full_report, "controller_counts")
    claims = _nested(full_report, "claims")
    frozen = _nested(full_report, "frozen_inputs")
    gate = _mapping(contract["gates"], "campaign gates")

    require(full_report.get("schema_version") == recovery.RECOVERY_SCHEMA_VERSION, "v3 schema")
    require(full_report.get("kind") == recovery.RECOVERY_KIND, "v3 kind")
    require(full_report.get("verdict") == "mixed_controller_recovery_passed", "v3 verdict")
    require(
        full_report.get("recovered_to_original_q9_518_boundary") is True,
        "v3 recovery boundary",
    )
    require(set(v3_gates) == EXPECTED_V3_GATE_KEYS, "v3 gate key set")
    require(bool(v3_gates) and all(value is True for value in v3_gates.values()), "v3 gates")
    require(full_report.get("partial_failure") is None, "v3 partial failure")
    require(full_report.get("attempted_transitions") == TOTAL_TRANSITIONS, "attempted transitions")
    require(full_report.get("simulator_step_calls_started") == TOTAL_TRANSITIONS, "step calls")
    require(probe.get("step_calls_started") == TOTAL_TRANSITIONS, "probe step calls")
    require(
        probe.get("transitions_completed") == TOTAL_TRANSITIONS,
        "probe completed transitions",
    )
    require(controller == {"student": STUDENT_TRANSITIONS, "teacher": TEACHER_TRANSITIONS}, "controller partition")
    require(full_report.get("student_inference_count") == TOTAL_TRANSITIONS, "student queries")
    require(parity.get("query_count") == TOTAL_TRANSITIONS, "teacher queries")
    require(parity.get("passed") is True, "teacher parity pass")
    require(parity.get("violation_count") == 0, "teacher parity violations")
    require(
        parity.get("threshold") == float(gate["teacher_pt_onnx_parity_max_absolute_error"]),
        "teacher parity threshold",
    )
    expected_actor_sha = contract["prerequisites"]["teacher_actor_state_sha256"]
    require(
        parity.get("actor_state_sha256_before") == expected_actor_sha
        and parity.get("actor_state_sha256_after") == expected_actor_sha
        and parity.get("actor_state_unchanged") is True,
        "teacher actor frozen",
    )
    require(
        _scalar_or_none(parity.get("maximum_absolute_error")) is not None
        and float(parity["maximum_absolute_error"]) <= float(gate["teacher_pt_onnx_parity_max_absolute_error"]),
        "teacher parity ceiling",
    )
    require(rollout.get("transition_count") == TOTAL_TRANSITIONS, "rollout transitions")
    require(rollout.get("all_values_finite") is True, "rollout finite")
    require(rollout.get("hard_safety_violation_count") == 0, "hard safety count")
    require(rollout.get("soft_safety_warning_count") == 0, "soft safety count")
    for name in gate["required_rollout_zero_counts"]:
        require(rollout_totals.get(name) == 0, f"rollout zero count:{name}")
        require(rollout_maxima.get(name) == 0, f"rollout zero maximum:{name}")
    for name in gate["required_v3_zero_counts"]:
        require(support.get(name) == 0, f"support zero count:{name}")
    scalar_gates = (
        ("minimum_base_height_m", ">=", "minimum_base_height_m"),
        ("maximum_base_tilt_rad", "<=", "maximum_base_tilt_rad"),
        ("maximum_joint_velocity_ratio", "<=", "maximum_joint_velocity_ratio"),
        ("maximum_tracking_rmse_rad", "<=", "maximum_tracking_rmse_rad"),
        (
            "maximum_actuator_force_ratio_evidence_only",
            "<=",
            "maximum_actuator_force_ratio",
        ),
    )
    for actual_name, operator, expected_name in scalar_gates:
        actual = _scalar_or_none(support.get(actual_name))
        expected = float(gate[expected_name])
        require(
            actual is not None and (actual >= expected if operator == ">=" else actual <= expected),
            f"support scalar gate:{actual_name}",
        )
    raw_max = _scalar_or_none(support.get("maximum_plain_sonic_raw_native_abs"))
    require(
        raw_max is not None and raw_max < float(gate["plain_sonic_raw_native_strict_abs_max"]),
        "plain raw strict ceiling",
    )
    action_errors = _nested(action, "maximum_absolute_error_by_link")
    action_threshold = _scalar_or_none(action.get("threshold"))
    require(
        action.get("passed") is True
        and action.get("check_count") == TOTAL_TRANSITIONS
        and action.get("required_check_count") == TOTAL_TRANSITIONS
        and action.get("mismatch_count") == 0
        and action.get("raw_clip_coordinate_count") == 0
        and action_threshold == recovery.ACTION_LINK_ATOL
        and bool(action_errors)
        and all(
            _scalar_or_none(value) is not None and float(value) <= recovery.ACTION_LINK_ATOL
            for value in action_errors.values()
        ),
        "actual action chain",
    )
    teacher_errors = _nested(teacher_chain, "maximum_absolute_error_by_link")
    teacher_threshold = _scalar_or_none(teacher_chain.get("threshold"))
    require(
        teacher_chain.get("passed") is True
        and teacher_chain.get("check_count") == TEACHER_TRANSITIONS
        and teacher_chain.get("required_check_count") == TEACHER_TRANSITIONS
        and teacher_chain.get("mismatch_count") == 0
        and teacher_chain.get("teacher_action_arrays_published") is False
        and teacher_threshold == recovery.ACTION_LINK_ATOL
        and bool(teacher_errors)
        and all(
            _scalar_or_none(value) is not None and float(value) <= recovery.ACTION_LINK_ATOL
            for value in teacher_errors.values()
        ),
        "teacher composite chain",
    )
    require(selected.get("mismatch_count") == 0, "selected state mismatch")
    require(
        selected.get("build_count") == TOTAL_TRANSITIONS
        and selected.get("nonterminal_update_count") == TOTAL_TRANSITIONS - 1
        and selected.get("autoreset_synchronization_count") == 1
        and selected.get("reset_previous_selected_raw_is_exact_zero") is True
        and selected.get("autoreset_previous_selected_raw_is_exact_zero") is True
        and selected.get("handoff_observed") is True
        and selected.get("student_teacher_handoff_resets_state") is False
        and _scalar_or_none(selected.get("maximum_target_round_trip_absolute_error")) is not None
        and float(selected["maximum_target_round_trip_absolute_error"]) <= recovery.ACTION_LINK_ATOL,
        "selected handoff",
    )
    require(history.get("check_count") == TOTAL_TRANSITIONS, "history checks")
    require(history.get("single_append_shift_check_count") == TOTAL_TRANSITIONS - 1, "history shifts")
    require(
        history_autoreset.get("local_transition") == 0
        and history_autoreset.get("actual_history_depth") == 1
        and history_autoreset.get("reset_padding_count") == 9
        and history_autoreset.get("previous_action_slice_zero") is True
        and history_autoreset.get("term_major_policy930_exact") is True,
        "history autoreset",
    )
    require(
        terminal.get("transition") == TOTAL_TRANSITIONS - 1
        and terminal.get("q9_before") == 518
        and terminal.get("is_timeout") is True
        and terminal.get("is_terminated") is False
        and terminal.get("termination_names") == ["time_out"],
        "final timeout only",
    )
    require(full_report.get("bindings_unchanged") is True, "runtime bindings")
    require(frozen.get("all_preflight_bound_inputs_unchanged") is True, "frozen inputs")
    require(claims.get("mode") == RECOVERY_MODE, "recovery claim mode")
    require(safety.get("published_teacher_label_count") == 0, "published labels")
    require(safety.get("training_updates") == 0, "training updates")
    require(safety.get("training_performed") is False, "training performed")

    effective_seed = probe.get("effective_seed")
    require(effective_seed == spec.seed, "effective seed")
    require(probe.get("reward_metric_bound") is True, "recovery metric binding")
    require(probe.get("metric_sample_count") == TOTAL_TRANSITIONS, "metric sample count")
    metric_digest = probe.get("metric_trajectory_sha256")
    require(
        isinstance(metric_digest, str)
        and len(metric_digest) == 64
        and all(character in "0123456789abcdef" for character in metric_digest),
        "metric trajectory digest",
    )
    seed_digest = probe.get("effective_seed_sha256")
    require(
        isinstance(seed_digest, str)
        and seed_digest
        == _sha256_bytes(_canonical_json_bytes({"requested_seed": spec.seed, "effective_seed": spec.seed})),
        "effective seed digest",
    )
    reward_metric_stats = _nested(
        rollout_reward_rates,
        "evaluator_aligned_recovery",
    )
    require(reward_metric_stats.get("count") == TOTAL_TRANSITIONS, "rollout metric count")
    metric_sum = _scalar_or_none(probe.get("metric_sum"))
    weighted_sum = _scalar_or_none(reward_metric_stats.get("sum"))
    weighted_minimum = _scalar_or_none(reward_metric_stats.get("minimum"))
    weighted_maximum = _scalar_or_none(reward_metric_stats.get("maximum"))
    require(
        metric_sum is not None
        and weighted_sum is not None
        and math.isclose(weighted_sum, metric_sum * -25.0, rel_tol=1.0e-12, abs_tol=1.0e-9),
        "rollout metric sum",
    )
    require(
        _scalar_or_none(probe.get("metric_minimum")) is not None
        and weighted_maximum is not None
        and math.isclose(
            weighted_maximum,
            float(probe["metric_minimum"]) * -25.0,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        ),
        "rollout metric minimum",
    )
    require(
        _scalar_or_none(probe.get("metric_maximum")) is not None
        and weighted_minimum is not None
        and math.isclose(
            weighted_minimum,
            float(probe["metric_maximum"]) * -25.0,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        ),
        "rollout metric maximum",
    )
    if spec.scenario == "nominal":
        require(probe.get("disturbance_performed") is False, "nominal disturbance flag")
        require(probe.get("impulse_count") == 0, "nominal impulse count")
        require(probe.get("impulse_apply_transition") is None, "nominal impulse transition")
        require(probe.get("impulse_apply_q9") is None, "nominal impulse q9")
        require(probe.get("planned_impulse_sha256") is None, "nominal planned impulse")
        require(probe.get("target_root_qvel_sha256") is None, "nominal target qvel")
        require(probe.get("realized_impulse_sha256") is None, "nominal realized impulse")
        require(probe.get("impulse_qvel_write_exact") is None, "nominal qvel write")
        require(probe.get("baseline_mean") is None, "nominal recovery baseline")
        require(probe.get("recovery_threshold") is None, "nominal recovery threshold")
        require(
            probe.get("first_threshold_crossing_transition") is None,
            "nominal recovery crossing",
        )
        require(
            probe.get("stable_streak_start_transition") is None
            and probe.get("stable_streak_end_transition") is None,
            "nominal recovery streak",
        )
        require(probe.get("recovery_time_s") is None, "nominal recovery time")
        require(probe.get("recovered") is None, "nominal recovery not performed")
    else:
        require(probe.get("disturbance_performed") is True, "disturbance flag")
        require(probe.get("impulse_count") == 1, "disturbance impulse count")
        require(
            probe.get("impulse_apply_transition") == GLOBAL_IMPULSE_TRANSITION,
            "disturbance impulse transition",
        )
        require(probe.get("impulse_apply_q9") == IMPULSE_Q9, "disturbance impulse q9")
        require(
            probe.get("planned_impulse_sha256") == _sha256_bytes(_canonical_json_bytes(list(spec.impulse or ()))),
            "planned impulse digest",
        )
        realized_digest = probe.get("realized_impulse_sha256")
        target_qvel_digest = probe.get("target_root_qvel_sha256")
        require(
            isinstance(realized_digest, str)
            and len(realized_digest) == 64
            and all(character in "0123456789abcdef" for character in realized_digest),
            "realized impulse digest",
        )
        require(
            isinstance(target_qvel_digest, str) and target_qvel_digest == realized_digest,
            "exact root qvel write digest",
        )
        require(probe.get("impulse_qvel_write_exact") is True, "exact root qvel write")
        readback = _scalar_or_none(probe.get("impulse_readback_max_absolute_error"))
        require(
            readback is not None and readback <= float(contract["disturbance"]["readback_max_absolute_error"]),
            "impulse readback",
        )
        require(probe.get("stable_streak_end_transition") is not None, "stable recovery streak")
        require(probe.get("recovered") is True, "disturbance recovery")
        baseline_mean = _scalar_or_none(probe.get("baseline_mean"))
        threshold = _scalar_or_none(probe.get("recovery_threshold"))
        first_crossing = probe.get("first_threshold_crossing_transition")
        streak_start = probe.get("stable_streak_start_transition")
        streak_end = probe.get("stable_streak_end_transition")
        recovery_time = _scalar_or_none(probe.get("recovery_time_s"))
        require(
            baseline_mean is not None
            and threshold is not None
            and math.isclose(threshold, baseline_mean + 0.1, rel_tol=0.0, abs_tol=1.0e-12),
            "disturbance recovery threshold",
        )
        require(
            isinstance(first_crossing, int)
            and not isinstance(first_crossing, bool)
            and RECOVERY_SCAN_FIRST <= first_crossing < TOTAL_TRANSITIONS,
            "disturbance first threshold crossing",
        )
        require(
            isinstance(streak_start, int)
            and not isinstance(streak_start, bool)
            and isinstance(streak_end, int)
            and not isinstance(streak_end, bool)
            and streak_start >= RECOVERY_SCAN_FIRST
            and streak_end - streak_start + 1 == RECOVERY_STABLE_STEPS,
            "disturbance stable recovery indices",
        )
        expected_recovery_time = (
            None
            if not isinstance(streak_end, int) or isinstance(streak_end, bool)
            else (streak_end - GLOBAL_IMPULSE_TRANSITION + 1) * CONTROL_DT_S
        )
        require(
            recovery_time is not None
            and expected_recovery_time is not None
            and math.isclose(recovery_time, expected_recovery_time, rel_tol=0.0, abs_tol=1.0e-12)
            and recovery_time <= float(contract["disturbance"]["maximum_recovery_time_s"]),
            "disturbance recovery time",
        )

    try:
        report_sha = _sha256_bytes(_canonical_json_bytes(full_report))
    except (TypeError, ValueError):
        report_sha = None
        issues.append("v3 report is not strict finite JSON")
    issue_sha = _sha256_bytes(_canonical_json_bytes(issues))
    action_max = max(
        (
            float(value)
            for value in action_errors.values()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
        ),
        default=math.inf,
    )
    record = {
        "scenario": spec.scenario,
        "rollout_index": spec.rollout_index,
        "seed": spec.seed,
        "run_id": spec.run_id,
        "passed": not issues,
        "issue_count": len(issues),
        "issues_sha256": issue_sha,
        "first_issue": None if not issues else issues[0],
        "failure_detail_sha256": None if not issues else issue_sha,
        "v3_report_sha256": report_sha,
        "v3_verdict": full_report.get("verdict"),
        "step_calls_started": probe.get("step_calls_started"),
        "transitions_completed": probe.get("transitions_completed"),
        "attempted_transitions": full_report.get("attempted_transitions"),
        "student_controller_count": controller.get("student"),
        "teacher_controller_count": controller.get("teacher"),
        "student_inference_count": full_report.get("student_inference_count"),
        "teacher_query_count": parity.get("query_count"),
        "terminal_transition": terminal.get("transition"),
        "terminal_q9": terminal.get("q9_before"),
        "final_timeout_only": "final timeout only" not in issues,
        "history_check_count": history.get("check_count"),
        "history_shift_count": history.get("single_append_shift_check_count"),
        "v3_gate_count": len(v3_gates),
        "v3_false_gate_count": sum(value is not True for value in v3_gates.values()),
        "hard_safety_violation_count": rollout.get("hard_safety_violation_count"),
        "soft_safety_warning_count": rollout.get("soft_safety_warning_count"),
        "rollout_required_zero_total": sum(
            int(rollout_totals.get(name, 1)) for name in gate["required_rollout_zero_counts"]
        ),
        "support_required_zero_total": sum(int(support.get(name, 1)) for name in gate["required_v3_zero_counts"]),
        "minimum_base_height_m": _scalar_or_none(support.get("minimum_base_height_m")),
        "maximum_base_tilt_rad": _scalar_or_none(support.get("maximum_base_tilt_rad")),
        "maximum_joint_velocity_ratio": _scalar_or_none(support.get("maximum_joint_velocity_ratio")),
        "maximum_tracking_rmse_rad": _scalar_or_none(support.get("maximum_tracking_rmse_rad")),
        "maximum_actuator_force_ratio": _scalar_or_none(support.get("maximum_actuator_force_ratio_evidence_only")),
        "maximum_plain_sonic_raw_native_abs": raw_max,
        "teacher_parity_max_absolute_error": _scalar_or_none(parity.get("maximum_absolute_error")),
        "action_chain_max_absolute_error": action_max if math.isfinite(action_max) else None,
        "effective_seed": effective_seed,
        "effective_seed_sha256": probe.get("effective_seed_sha256"),
        "disturbance_performed": probe.get("disturbance_performed"),
        "impulse_count": probe.get("impulse_count"),
        "impulse_apply_transition": probe.get("impulse_apply_transition"),
        "impulse_apply_q9": probe.get("impulse_apply_q9"),
        "planned_impulse_sha256": probe.get("planned_impulse_sha256"),
        "target_root_qvel_sha256": probe.get("target_root_qvel_sha256"),
        "realized_impulse_sha256": probe.get("realized_impulse_sha256"),
        "impulse_qvel_write_exact": probe.get("impulse_qvel_write_exact"),
        "impulse_readback_max_absolute_error": _scalar_or_none(probe.get("impulse_readback_max_absolute_error")),
        "metric_sample_count": probe.get("metric_sample_count"),
        "metric_trajectory_sha256": probe.get("metric_trajectory_sha256"),
        "baseline_mean": _scalar_or_none(probe.get("baseline_mean")),
        "recovery_threshold": _scalar_or_none(probe.get("recovery_threshold")),
        "first_threshold_crossing_transition": probe.get("first_threshold_crossing_transition"),
        "stable_streak_start_transition": probe.get("stable_streak_start_transition"),
        "stable_streak_end_transition": probe.get("stable_streak_end_transition"),
        "recovery_time_s": _scalar_or_none(probe.get("recovery_time_s")),
        "recovered": probe.get("recovered"),
        "published_teacher_label_count": 0,
        "published_training_row_count": 0,
    }
    _validate_run_record(record)
    return record


def _failed_run_record(
    spec: CampaignRunSpec,
    error: BaseException,
    *,
    probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    issue = f"runtime:{type(error).__name__}"
    failure_detail = _sha256_bytes(
        _canonical_json_bytes(
            {
                "stage": "campaign_run_runtime",
                "exception_type": type(error).__name__,
                "exception_message_sha256": _sha256_bytes(str(error).encode("utf-8")),
            }
        )
    )
    result = {key: None for key in RUN_RECORD_KEYS}
    result.update(
        {
            "scenario": spec.scenario,
            "rollout_index": spec.rollout_index,
            "seed": spec.seed,
            "run_id": spec.run_id,
            "passed": False,
            "issue_count": 1,
            "issues_sha256": failure_detail,
            "first_issue": issue,
            "failure_detail_sha256": failure_detail,
            "step_calls_started": 0 if probe is None else probe.get("step_calls_started", 0),
            "transitions_completed": (0 if probe is None else probe.get("transitions_completed", 0)),
            "attempted_transitions": (0 if probe is None else probe.get("transitions_completed", 0)),
            "student_controller_count": None,
            "teacher_controller_count": None,
            "student_inference_count": None,
            "teacher_query_count": None,
            "v3_gate_count": 0,
            "v3_false_gate_count": len(EXPECTED_V3_GATE_KEYS),
            "hard_safety_violation_count": None,
            "soft_safety_warning_count": None,
            "rollout_required_zero_total": None,
            "support_required_zero_total": None,
            "disturbance_performed": spec.scenario == "disturbance",
            "impulse_count": 0 if probe is None else probe.get("impulse_count"),
            "impulse_apply_transition": None if probe is None else probe.get("impulse_apply_transition"),
            "impulse_apply_q9": None if probe is None else probe.get("impulse_apply_q9"),
            "effective_seed": None if probe is None else probe.get("effective_seed"),
            "effective_seed_sha256": None if probe is None else probe.get("effective_seed_sha256"),
            "planned_impulse_sha256": None if probe is None else probe.get("planned_impulse_sha256"),
            "target_root_qvel_sha256": None if probe is None else probe.get("target_root_qvel_sha256"),
            "realized_impulse_sha256": None if probe is None else probe.get("realized_impulse_sha256"),
            "impulse_qvel_write_exact": None if probe is None else probe.get("impulse_qvel_write_exact"),
            "impulse_readback_max_absolute_error": None
            if probe is None
            else probe.get("impulse_readback_max_absolute_error"),
            "metric_sample_count": 0 if probe is None else probe.get("metric_sample_count", 0),
            "metric_trajectory_sha256": None if probe is None else probe.get("metric_trajectory_sha256"),
            "recovered": False if spec.scenario == "disturbance" else None,
            "published_teacher_label_count": 0,
            "published_training_row_count": 0,
        }
    )
    _validate_run_record(result)
    return result


def _validate_run_record(record: Mapping[str, Any]) -> None:
    if set(record) != RUN_RECORD_KEYS:
        raise ValueError("campaign run record schema mismatch")
    for key, value in record.items():
        if isinstance(value, (Mapping, list, tuple, np.ndarray)):
            raise ValueError(f"campaign run record contains non-scalar: {key}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"campaign run record contains nonfinite scalar: {key}")
    if record["scenario"] not in {"nominal", "disturbance"}:
        raise ValueError("campaign run record scenario invalid")
    if type(record.get("passed")) is not bool:
        raise ValueError("campaign run record passed must be boolean")
    if (
        not isinstance(record.get("issue_count"), int)
        or isinstance(record.get("issue_count"), bool)
        or record["issue_count"] < 0
    ):
        raise ValueError("campaign run record issue count invalid")
    if record["passed"] is True:
        if record["issue_count"] != 0 or record.get("first_issue") is not None:
            raise ValueError("passing campaign run contains issue")
        if record.get("failure_detail_sha256") is not None:
            raise ValueError("passing campaign run contains failure detail")
    elif record["issue_count"] < 1:
        raise ValueError("failed campaign run lacks issue")
    elif record.get("failure_detail_sha256") is None:
        raise ValueError("failed campaign run lacks detail digest")
    if record["passed"] is False:
        _bounded_public_failure(record.get("first_issue"), "campaign run first issue")
    _require_sha256(record["issues_sha256"], "campaign issue digest")
    for key in (
        "failure_detail_sha256",
        "v3_report_sha256",
        "effective_seed_sha256",
        "planned_impulse_sha256",
        "target_root_qvel_sha256",
        "realized_impulse_sha256",
        "metric_trajectory_sha256",
    ):
        if record[key] is not None:
            _require_sha256(record[key], f"campaign run {key}")
    if record["published_teacher_label_count"] != 0 or record["published_training_row_count"] != 0:
        raise ValueError("campaign run record attempts data publication")
    if record.get("passed") is True:
        required_equal = {
            "issue_count": 0,
            "first_issue": None,
            "attempted_transitions": TOTAL_TRANSITIONS,
            "step_calls_started": TOTAL_TRANSITIONS,
            "transitions_completed": TOTAL_TRANSITIONS,
            "student_controller_count": STUDENT_TRANSITIONS,
            "teacher_controller_count": TEACHER_TRANSITIONS,
            "student_inference_count": TOTAL_TRANSITIONS,
            "teacher_query_count": TOTAL_TRANSITIONS,
            "terminal_transition": TOTAL_TRANSITIONS - 1,
            "terminal_q9": 518,
            "final_timeout_only": True,
            "history_check_count": TOTAL_TRANSITIONS,
            "history_shift_count": TOTAL_TRANSITIONS - 1,
            "v3_gate_count": len(EXPECTED_V3_GATE_KEYS),
            "v3_false_gate_count": 0,
            "hard_safety_violation_count": 0,
            "soft_safety_warning_count": 0,
            "rollout_required_zero_total": 0,
            "support_required_zero_total": 0,
            "metric_sample_count": TOTAL_TRANSITIONS,
        }
        for key, expected in required_equal.items():
            if record.get(key) != expected:
                raise ValueError(f"passing campaign run has invalid {key}")
        if record.get("effective_seed") != record.get("seed"):
            raise ValueError("passing campaign run effective seed mismatch")
        expected_seed_digest = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "requested_seed": record["seed"],
                    "effective_seed": record["effective_seed"],
                }
            )
        )
        if record.get("effective_seed_sha256") != expected_seed_digest:
            raise ValueError("passing campaign run seed digest mismatch")
        for key in ("v3_report_sha256", "metric_trajectory_sha256"):
            _require_sha256(record.get(key), f"passing campaign run {key}")
        if record["scenario"] == "nominal":
            if any(
                (
                    record.get("disturbance_performed") is not False,
                    record.get("impulse_count") != 0,
                    record.get("impulse_apply_transition") is not None,
                    record.get("impulse_apply_q9") is not None,
                    record.get("planned_impulse_sha256") is not None,
                    record.get("target_root_qvel_sha256") is not None,
                    record.get("realized_impulse_sha256") is not None,
                    record.get("impulse_qvel_write_exact") is not None,
                    record.get("recovery_time_s") is not None,
                    record.get("recovered") is not None,
                )
            ):
                raise ValueError("passing nominal run claims recovery operation")
        elif any(
            (
                record.get("disturbance_performed") is not True,
                record.get("impulse_count") != 1,
                record.get("impulse_apply_transition") != GLOBAL_IMPULSE_TRANSITION,
                record.get("impulse_apply_q9") != IMPULSE_Q9,
                record.get("impulse_qvel_write_exact") is not True,
                record.get("target_root_qvel_sha256") != record.get("realized_impulse_sha256"),
                record.get("recovered") is not True,
            )
        ):
            raise ValueError("passing disturbance run lacks exact recovery evidence")
        if record["scenario"] == "disturbance":
            for key in (
                "planned_impulse_sha256",
                "target_root_qvel_sha256",
                "realized_impulse_sha256",
            ):
                _require_sha256(record.get(key), f"passing disturbance {key}")


def _execute_one(
    *,
    spec: CampaignRunSpec,
    request: recovery.RecoveryRequest,
    cached_preflight: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    probe = _StepProbe(spec, contract)
    try:
        with _scoped_runtime_interposition(
            spec=spec,
            probe=probe,
            cached_preflight=cached_preflight,
        ):
            full_report = recovery.run_recovery_diagnostic(request)
        if not isinstance(full_report, Mapping):
            raise RuntimeError("sealed recovery runner returned non-mapping")
        return _assess_run(
            spec=spec,
            full_report=full_report,
            probe=probe.report(),
            contract=contract,
        )
    except Exception as error:
        return _failed_run_record(spec, error, probe=probe.report())


def _campaign_report(
    *,
    preflight: Mapping[str, Any],
    specs: Sequence[CampaignRunSpec],
    records: Sequence[Mapping[str, Any]],
    sources_after: Mapping[str, Any],
    runtime_failure: str | None,
) -> dict[str, Any]:
    contract = _mapping(preflight["contract"], "campaign contract")
    prerequisites = _mapping(contract["prerequisites"], "campaign prerequisites")
    source_before = _mapping(preflight["sources"], "campaign sources before")
    if len(records) > len(specs):
        raise ValueError("campaign has more run records than planned")
    for spec, record in zip(specs, records, strict=False):
        _validate_run_record(record)
        if (
            record.get("scenario") != spec.scenario
            or record.get("rollout_index") != spec.rollout_index
            or record.get("seed") != spec.seed
            or record.get("run_id") != spec.run_id
        ):
            raise ValueError("campaign run record order/spec mismatch")
    completed = len(records)
    passed = sum(record.get("passed") is True for record in records)
    nominal_passed = sum(
        record.get("passed") is True and record.get("scenario") == "nominal" for record in records
    )
    disturbance_passed = sum(
        record.get("passed") is True and record.get("scenario") == "disturbance" for record in records
    )
    disturbance_recovered = sum(
        record.get("passed") is True
        and record.get("scenario") == "disturbance"
        and record.get("recovered") is True
        for record in records
    )
    source_rehash_succeeded = sources_after.get("available", True) is True
    source_rehash_error_sha256 = sources_after.get("error_sha256")
    sources_unchanged = source_rehash_succeeded and source_before == sources_after
    seed_digests = [record.get("effective_seed_sha256") for record in records]
    valid_seed_digests = [value for value in seed_digests if isinstance(value, str)]
    unique_seed_digests = bool(
        completed > 0 and len(valid_seed_digests) == len(set(valid_seed_digests)) == completed
    )
    qualified = bool(
        completed == len(specs) == 20
        and passed == 20
        and nominal_passed == 10
        and disturbance_passed == 10
        and disturbance_recovered == 10
        and unique_seed_digests
        and sources_unchanged
        and runtime_failure is None
    )
    first_failed = next((record for record in records if record.get("passed") is not True), None)
    first_failure = runtime_failure or (None if first_failed is None else str(first_failed.get("first_issue")))
    if qualified:
        failure_detail_sha256 = None
    elif first_failed is not None:
        failure_detail_sha256 = first_failed.get("failure_detail_sha256")
    elif source_rehash_error_sha256 is not None:
        failure_detail_sha256 = source_rehash_error_sha256
    elif not sources_unchanged:
        failure_detail_sha256 = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "stage": "post_run_sources_changed",
                    "before": source_before["binding_sha256"],
                    "after": sources_after["binding_sha256"],
                }
            )
        )
    else:
        failure_detail_sha256 = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "stage": runtime_failure or "campaign_preflight",
                    "preflight_issue_sha256": [
                        _sha256_bytes(str(issue).encode("utf-8")) for issue in preflight.get("issues", ())
                    ],
                }
            )
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "campaign_qualified": qualified,
        "verdict": "qualification_campaign_passed" if qualified else "qualification_campaign_failed",
        "campaign_contract_sha256": CONTRACT_SHA256,
        PREREQUISITE_REPORT_OUTPUT_KEY: prerequisites[PREREQUISITE_REPORT_SHA_KEY],
        "candidate_manifest_sha256": prerequisites["candidate_manifest_sha256"],
        "teacher_onnx_sha256": prerequisites["teacher_onnx_sha256"],
        "motion_sha256": prerequisites["motion_sha256"],
        "planned_run_count": len(specs),
        "completed_run_count": completed,
        "passed_run_count": passed,
        "nominal_passed_count": nominal_passed,
        "disturbance_passed_count": disturbance_passed,
        "disturbance_recovered_count": disturbance_recovered,
        "remaining_run_count": len(specs) - completed,
        "run_order_sha256": _sha256_bytes(_canonical_json_bytes([spec.descriptor() for spec in specs])),
        "effective_seed_set_sha256": _sha256_bytes(_canonical_json_bytes(valid_seed_digests)),
        "all_effective_seeds_unique": unique_seed_digests,
        "first_failed_run_id": None if first_failed is None else first_failed.get("run_id"),
        "first_failure": first_failure,
        "failure_detail_sha256": failure_detail_sha256,
        "preflight_ready": preflight.get("ready") is True,
        "preflight_issue_count": len(preflight.get("issues", ())),
        "executed_source_binding_before_sha256": source_before["binding_sha256"],
        "executed_source_binding_after_sha256": sources_after["binding_sha256"],
        "source_rehash_succeeded": source_rehash_succeeded,
        "source_rehash_error_sha256": source_rehash_error_sha256,
        "sources_unchanged": sources_unchanged,
        "runs": [dict(record) for record in records],
        "published_teacher_label_count": 0,
        "published_training_row_count": 0,
        "fresh_disjoint_suffix_collection_authorized": qualified,
        "teacher_support_qualified": False,
        "dagger_data": False,
        "training_performed": False,
        "promotion_or_deployment": False,
        "hardware_authorized": False,
    }
    _validate_public_report(report, specs=specs)
    return report


def run_campaign(
    request: CampaignRequest,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run exact 20-rollout campaign; stop first failure; never publish data."""

    preflight = _preflight_internal(request)
    specs = campaign_run_specs(request.root)
    records: list[dict[str, Any]] = []
    runtime_failure: str | None = None
    if preflight.get("ready") is True:
        base_request = preflight["base_request"]
        cached = preflight["recovery_preflight"]
        contract = preflight["contract"]
        for spec in specs:
            try:
                record = _execute_one(
                    spec=spec,
                    request=base_request,
                    cached_preflight=cached,
                    contract=contract,
                )
            except Exception as error:  # compact fail-closed evidence
                record = _failed_run_record(spec, error)
                runtime_failure = record["first_issue"]
            records.append(record)
            if progress is not None:
                progress(
                    {
                        "scenario": record["scenario"],
                        "rollout_index": record["rollout_index"],
                        "seed": record["seed"],
                        "passed": record["passed"],
                        "attempted_transitions": record["attempted_transitions"],
                        "recovery_time_s": record["recovery_time_s"],
                        "recovered": record["recovered"],
                    }
                )
            if record["passed"] is not True:
                break
    else:
        runtime_failure = "campaign preflight not ready"
    try:
        sources_after = executed_campaign_source_binding(request.root)
    except Exception as error:
        source_error = f"post_run_source_rehash:{type(error).__name__}"
        source_error_sha = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "stage": "post_run_source_rehash",
                    "exception_type": type(error).__name__,
                    "exception_message_sha256": _sha256_bytes(str(error).encode("utf-8")),
                }
            )
        )
        sources_after = {
            "available": False,
            "binding_sha256": source_error_sha,
            "error_sha256": source_error_sha,
        }
        if runtime_failure is None and all(record.get("passed") is True for record in records):
            runtime_failure = source_error
    else:
        if (
            sources_after != preflight["sources"]
            and runtime_failure is None
            and all(record.get("passed") is True for record in records)
        ):
            runtime_failure = "post_run_sources_changed"
    return _campaign_report(
        preflight=preflight,
        specs=specs,
        records=records,
        sources_after=sources_after,
        runtime_failure=runtime_failure,
    )


def failure_report(error: BaseException, request: CampaignRequest) -> dict[str, Any]:
    """Return strict compact zero-row failure when preflight/orchestration raises."""

    root = request.root
    contract = load_campaign_contract(root)
    sources = executed_campaign_source_binding(root)
    prerequisites = _mapping(contract["prerequisites"], "campaign prerequisites")
    message = f"orchestration:{type(error).__name__}"
    failure_detail = _sha256_bytes(
        _canonical_json_bytes(
            {
                "stage": "campaign_orchestration",
                "exception_type": type(error).__name__,
                "exception_message_sha256": _sha256_bytes(str(error).encode("utf-8")),
            }
        )
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "campaign_qualified": False,
        "verdict": "qualification_campaign_failed",
        "campaign_contract_sha256": CONTRACT_SHA256,
        PREREQUISITE_REPORT_OUTPUT_KEY: prerequisites[PREREQUISITE_REPORT_SHA_KEY],
        "candidate_manifest_sha256": prerequisites["candidate_manifest_sha256"],
        "teacher_onnx_sha256": prerequisites["teacher_onnx_sha256"],
        "motion_sha256": prerequisites["motion_sha256"],
        "planned_run_count": 20,
        "completed_run_count": 0,
        "passed_run_count": 0,
        "nominal_passed_count": 0,
        "disturbance_passed_count": 0,
        "disturbance_recovered_count": 0,
        "remaining_run_count": 20,
        "run_order_sha256": _sha256_bytes(
            _canonical_json_bytes([spec.descriptor() for spec in campaign_run_specs(root)])
        ),
        "effective_seed_set_sha256": _sha256_bytes(_canonical_json_bytes([])),
        "all_effective_seeds_unique": False,
        "first_failed_run_id": None,
        "first_failure": message,
        "failure_detail_sha256": failure_detail,
        "preflight_ready": False,
        "preflight_issue_count": 1,
        "executed_source_binding_before_sha256": sources["binding_sha256"],
        "executed_source_binding_after_sha256": sources["binding_sha256"],
        "source_rehash_succeeded": True,
        "source_rehash_error_sha256": None,
        "sources_unchanged": True,
        "runs": [],
        "published_teacher_label_count": 0,
        "published_training_row_count": 0,
        "fresh_disjoint_suffix_collection_authorized": False,
        "teacher_support_qualified": False,
        "dagger_data": False,
        "training_performed": False,
        "promotion_or_deployment": False,
        "hardware_authorized": False,
    }
    _validate_public_report(report, specs=campaign_run_specs(root))
    return report


def _validate_public_report(
    report: Mapping[str, Any],
    *,
    specs: Sequence[CampaignRunSpec] | None = None,
) -> None:
    if set(report) != TOP_LEVEL_KEYS:
        raise ValueError("campaign public report schema mismatch")
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) > 20:
        raise ValueError("campaign report runs must be fixed list up to twenty")
    for record in runs:
        if not isinstance(record, Mapping):
            raise ValueError("campaign report run must be mapping")
        _validate_run_record(record)
    if report.get("schema_version") != SCHEMA_VERSION or report.get("kind") != REPORT_KIND:
        raise ValueError("campaign public report identity mismatch")
    if report.get("campaign_contract_sha256") != CONTRACT_SHA256:
        raise ValueError("campaign public report contract digest mismatch")
    for key, value in report.items():
        if key == "runs":
            continue
        if isinstance(value, (Mapping, list, tuple, np.ndarray)):
            raise ValueError(f"campaign top-level field is not scalar: {key}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"campaign top-level field is nonfinite: {key}")
    for key in (
        "campaign_contract_sha256",
        PREREQUISITE_REPORT_OUTPUT_KEY,
        "candidate_manifest_sha256",
        "teacher_onnx_sha256",
        "motion_sha256",
        "run_order_sha256",
        "effective_seed_set_sha256",
        "executed_source_binding_before_sha256",
        "executed_source_binding_after_sha256",
    ):
        _require_sha256(report[key], f"campaign report {key}")
    if report.get("failure_detail_sha256") is not None:
        _require_sha256(
            report["failure_detail_sha256"],
            "campaign report failure_detail_sha256",
        )
    if report.get("source_rehash_error_sha256") is not None:
        _require_sha256(
            report["source_rehash_error_sha256"],
            "campaign report source_rehash_error_sha256",
        )
    if report.get("source_rehash_succeeded") is not True:
        if report.get("sources_unchanged") is not False:
            raise ValueError("failed source rehash cannot claim unchanged sources")
        if report.get("source_rehash_error_sha256") is None:
            raise ValueError("failed source rehash lacks error digest")
    elif report.get("source_rehash_error_sha256") is not None:
        raise ValueError("successful source rehash contains error digest")
    before_digest = report.get("executed_source_binding_before_sha256")
    after_digest = report.get("executed_source_binding_after_sha256")
    if report.get("sources_unchanged") is True:
        if report.get("source_rehash_succeeded") is not True or before_digest != after_digest:
            raise ValueError("unchanged source claim contradicts binding digests")
    elif before_digest == after_digest and report.get("source_rehash_succeeded") is True:
        raise ValueError("changed source claim contradicts equal binding digests")
    if (
        report.get("published_teacher_label_count") != 0
        or report.get("published_training_row_count") != 0
        or report.get("teacher_support_qualified") is not False
        or report.get("dagger_data") is not False
        or report.get("training_performed") is not False
        or report.get("promotion_or_deployment") is not False
        or report.get("hardware_authorized") is not False
    ):
        raise ValueError("campaign public report crosses safety boundary")
    qualified = report.get("campaign_qualified") is True
    if report.get("fresh_disjoint_suffix_collection_authorized") is not qualified:
        raise ValueError("campaign collector authorization differs from qualification")

    completed = len(runs)
    passed = sum(record.get("passed") is True for record in runs)
    nominal_passed = sum(record.get("passed") is True and record.get("scenario") == "nominal" for record in runs)
    disturbance_passed = sum(
        record.get("passed") is True and record.get("scenario") == "disturbance" for record in runs
    )
    disturbance_recovered = sum(
        record.get("passed") is True
        and record.get("scenario") == "disturbance"
        and record.get("recovered") is True
        for record in runs
    )
    expected_counts = {
        "planned_run_count": 20,
        "completed_run_count": completed,
        "passed_run_count": passed,
        "nominal_passed_count": nominal_passed,
        "disturbance_passed_count": disturbance_passed,
        "disturbance_recovered_count": disturbance_recovered,
        "remaining_run_count": 20 - completed,
    }
    for key, expected in expected_counts.items():
        if report.get(key) != expected:
            raise ValueError(f"campaign public report counter mismatch: {key}")
    seed_digests = [record.get("effective_seed_sha256") for record in runs]
    valid_seed_digests = [value for value in seed_digests if isinstance(value, str)]
    unique_seeds = bool(completed > 0 and len(valid_seed_digests) == len(set(valid_seed_digests)) == completed)
    if report.get("all_effective_seeds_unique") is not unique_seeds:
        raise ValueError("campaign effective seed uniqueness claim mismatch")
    expected_seed_set_digest = _sha256_bytes(_canonical_json_bytes(valid_seed_digests))
    if report.get("effective_seed_set_sha256") != expected_seed_set_digest:
        raise ValueError("campaign effective seed set digest mismatch")
    first_failed = next((record for record in runs if record.get("passed") is not True), None)
    expected_failed_id = None if first_failed is None else first_failed.get("run_id")
    if report.get("first_failed_run_id") != expected_failed_id:
        raise ValueError("campaign first failed run mismatch")
    if qualified:
        if report.get("first_failure") is not None:
            raise ValueError("qualified campaign contains failure")
        if report.get("failure_detail_sha256") is not None:
            raise ValueError("qualified campaign contains failure detail")
    else:
        _bounded_public_failure(report.get("first_failure"), "campaign first failure")
    if not qualified and report.get("failure_detail_sha256") is None:
        raise ValueError("failed campaign lacks failure detail digest")
    if report.get("preflight_ready") is True:
        if report.get("preflight_issue_count") != 0:
            raise ValueError("ready campaign preflight contains issues")
    elif not isinstance(report.get("preflight_issue_count"), int) or report["preflight_issue_count"] < 1:
        raise ValueError("failed campaign preflight lacks issue count")
    expected_qualified = bool(
        report.get("preflight_ready") is True
        and completed == 20
        and passed == 20
        and nominal_passed == 10
        and disturbance_passed == 10
        and disturbance_recovered == 10
        and unique_seeds
        and report.get("source_rehash_succeeded") is True
        and report.get("sources_unchanged") is True
        and report.get("first_failure") is None
    )
    if qualified is not expected_qualified:
        raise ValueError("campaign qualification claim not derived from evidence")
    expected_verdict = "qualification_campaign_passed" if expected_qualified else "qualification_campaign_failed"
    if report.get("verdict") != expected_verdict:
        raise ValueError("campaign verdict contradicts evidence")
    if specs is not None:
        if len(specs) != 20:
            raise ValueError("campaign validation requires exact twenty specs")
        expected_order_digest = _sha256_bytes(_canonical_json_bytes([spec.descriptor() for spec in specs]))
        if report.get("run_order_sha256") != expected_order_digest:
            raise ValueError("campaign run order digest mismatch")
        for spec, record in zip(specs, runs, strict=False):
            if (
                record.get("scenario") != spec.scenario
                or record.get("rollout_index") != spec.rollout_index
                or record.get("seed") != spec.seed
                or record.get("run_id") != spec.run_id
            ):
                raise ValueError("campaign report run order/spec mismatch")


def _temporary_json(parent: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=".recovery-qualification-campaign-",
        suffix=".json.tmp",
        dir=parent,
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def write_campaign_report_new(
    request: CampaignRequest,
    report: Mapping[str, Any],
) -> Path:
    """Publish one exclusive compact JSON report; never overwrite."""

    _validate_public_report(report, specs=campaign_run_specs(request.root))
    output = request.output_path
    if os.path.lexists(output):
        raise FileExistsError(f"campaign output already exists: {output}")
    payload = (
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = _temporary_json(output.parent, payload)
    try:
        os.link(temporary, output)
    except FileExistsError:
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "CampaignRequest",
    "CampaignRunSpec",
    "CONTRACT_SHA256",
    "REPORT_KIND",
    "campaign_run_specs",
    "failure_report",
    "load_campaign_contract",
    "preflight_campaign",
    "run_campaign",
    "write_campaign_report_new",
]
