"""Fail-closed 20-run qualification campaign for q250 scale-2 recovery."""

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
import random
import struct
from typing import Any

import numpy as np
import torch

from gear_sonic.utils import (
    g1_true23_sonic_recovery_qualification_campaign as base_campaign,
    g1_true23_sonic_scale2_teacher_recovery as recovery,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_sonic_scale2_recovery_qualification_campaign_contract_v1"
REPORT_KIND = "g1_true23_sonic_scale2_recovery_qualification_campaign_report_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_scale2_recovery_qualification_campaign_v1.json"
)
CONTRACT_SHA256 = "e7862e0160937ba0cab16af26d06fb91bf202bd8a0c0e41d8216bfed4d051117"
EXECUTED_SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_sonic_scale2_recovery_qualification_campaign.py"),
    Path("gear_sonic/scripts/qualify_g1_true23_sonic_scale2_recovery_campaign.py"),
    Path("gear_sonic/utils/g1_true23_sonic_scale2_teacher_recovery.py"),
    Path("gear_sonic/utils/g1_true23_sonic_recovery_qualification_campaign.py"),
)

TOTAL_TRANSITIONS = 510
STUDENT_TRANSITIONS = 241
TEACHER_TRANSITIONS = 269
GLOBAL_IMPULSE_TRANSITION = 291
IMPULSE_Q9 = 300
FIRST_DISTURBED_Q9 = 301
BASELINE_TRANSITIONS = tuple(range(281, 291))
RECOVERY_STABLE_STEPS = 5
CONTROL_DT_S = 0.02


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be mapping")
    return value


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{context} must be finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite number")
    return result


def _strict_json(path: Path, context: str) -> Mapping[str, Any]:
    return base_campaign._strict_json(path, context)  # noqa: SLF001


@dataclass(frozen=True)
class CampaignRequest:
    repository_root: Path
    output: Path

    @property
    def root(self) -> Path:
        value = self.repository_root.expanduser().resolve(strict=True)
        if value.is_symlink() or not value.is_dir():
            raise ValueError("scale2 campaign repository root invalid")
        return value

    @property
    def output_path(self) -> Path:
        candidate = self.output.expanduser()
        candidate = candidate if candidate.is_absolute() else self.root / candidate
        value = candidate.resolve(strict=False)
        if not value.is_relative_to(self.root) or value.suffix.lower() != ".json":
            raise ValueError("scale2 campaign output must be repository-contained JSON")
        if candidate.is_symlink() or value.is_symlink():
            raise ValueError("scale2 campaign output cannot be symlink")
        parent = value.parent.resolve(strict=True)
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("scale2 campaign output parent invalid")
        return value


def _source_binding(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in EXECUTED_SOURCE_RELATIVE_PATHS:
        path = (root / relative).resolve(strict=True)
        if path.is_symlink() or not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"scale2 campaign source invalid: {relative.as_posix()}")
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"files": files, "binding_sha256": _sha256_bytes(_canonical_bytes(files))}


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("scale2 campaign contract mismatch")
    contract = _strict_json(path, "scale2 campaign contract")
    window = _mapping(contract.get("window"), "scale2 campaign window")
    cohort = _mapping(contract.get("cohort"), "scale2 campaign cohort")
    disturbance = _mapping(contract.get("disturbance"), "scale2 campaign disturbance")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("role") != "offline_mjlab_q250_mixed_controller_qualification_only"
        or window.get("mode") != "q250"
        or window.get("total_transitions") != TOTAL_TRANSITIONS
        or window.get("student_transition_count") != STUDENT_TRANSITIONS
        or window.get("teacher_transition_count") != TEACHER_TRANSITIONS
        or window.get("teacher_first_q9") != 250
        or window.get("last_action_q9") != 518
        or cohort.get("total_rollouts") != 20
        or cohort.get("rollouts_per_scenario") != 10
        or cohort.get("fail_fast") is not True
        or disturbance.get("global_apply_transition") != GLOBAL_IMPULSE_TRANSITION
        or disturbance.get("apply_q9") != IMPULSE_Q9
        or disturbance.get("first_disturbed_observation_q9") != FIRST_DISTURBED_Q9
        or disturbance.get("baseline_global_transitions") != list(BASELINE_TRANSITIONS)
        or disturbance.get("stable_recovery_steps") != RECOVERY_STABLE_STEPS
        or disturbance.get("maximum_recovery_time_s") != 2.0
    ):
        raise ValueError("scale2 campaign contract semantics mismatch")
    return contract


def _verify_prerequisites(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    prerequisite = _mapping(contract["prerequisites"], "scale2 campaign prerequisites")
    base_path = (root / str(prerequisite["base_campaign_contract_relative_path"])).resolve(strict=True)
    q250_path = (root / str(prerequisite["q250_pass_report_relative_path"])).resolve(strict=True)
    if (
        base_path.is_symlink()
        or q250_path.is_symlink()
        or sha256_file(base_path) != prerequisite["base_campaign_contract_sha256"]
        or sha256_file(q250_path) != prerequisite["q250_pass_report_sha256"]
    ):
        raise ValueError("scale2 campaign prerequisite bytes mismatch")
    base_contract = base_campaign.load_campaign_contract(root)
    q250 = _strict_json(q250_path, "scale2 q250 prerequisite")
    qualification = _mapping(q250.get("qualification"), "scale2 q250 qualification")
    if (
        q250.get("kind") != "g1_true23_sonic_scale2_teacher_recovery_diagnostic_v1"
        or q250.get("mode") != "q250"
        or qualification.get("passed") is not True
        or q250.get("completed_transitions") != TOTAL_TRANSITIONS
        or q250.get("controller_counts") != {"student": STUDENT_TRANSITIONS, "teacher": TEACHER_TRANSITIONS}
        or q250.get("teacher_labels_admitted") != 0
        or q250.get("training_performed") is not False
    ):
        raise ValueError("scale2 q250 prerequisite is not exact safe pass")
    return {"base_contract": base_contract, "q250": q250}


def _base_request(root: Path, seed: int | None = None) -> recovery.Scale2RecoveryRequest:
    return recovery.Scale2RecoveryRequest(
        root,
        root / "artifacts/g1_true23/.scale2-campaign-inner-unused.json",
        "q250",
        runtime_seed=seed,
    )


def _preflight_internal(request: CampaignRequest) -> dict[str, Any]:
    root = request.root
    contract = load_contract(root)
    prerequisites = _verify_prerequisites(root, contract)
    sources = _source_binding(root)
    recovery_preflight = recovery._preflight_internal(_base_request(root))  # noqa: SLF001
    if recovery_preflight.get("ready") is not True:
        raise RuntimeError("scale2 recovery preflight not ready")
    specs = base_campaign.campaign_run_specs(root)
    if len(specs) != 20:
        raise RuntimeError("scale2 campaign spec count mismatch")
    return {
        "ready": True,
        "root": root,
        "contract": contract,
        "prerequisites": prerequisites,
        "sources": sources,
        "recovery_preflight": recovery_preflight,
        "specs": specs,
    }


def preflight(request: CampaignRequest) -> dict[str, Any]:
    try:
        value = _preflight_internal(request)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "g1_true23_sonic_scale2_recovery_qualification_campaign_preflight_v1",
            "ready": True,
            "planned_run_count": 20,
            "campaign_contract_sha256": CONTRACT_SHA256,
            "q250_prerequisite_report_sha256": value["contract"]["prerequisites"]["q250_pass_report_sha256"],
            "executed_source_binding_sha256": value["sources"]["binding_sha256"],
            "simulator_constructed": False,
            "simulator_steps": 0,
            "published_teacher_label_count": 0,
            "published_training_row_count": 0,
            "training_performed": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "g1_true23_sonic_scale2_recovery_qualification_campaign_preflight_v1",
            "ready": False,
            "error": {"type": type(error).__name__, "detail_sha256": _sha256_bytes(str(error).encode())},
            "simulator_constructed": False,
            "simulator_steps": 0,
            "published_teacher_label_count": 0,
            "published_training_row_count": 0,
            "training_performed": False,
            "hardware_authorized": False,
        }


class _StepProbe:
    def __init__(self, spec: Any, contract: Mapping[str, Any]) -> None:
        self.spec = spec
        self.contract = contract
        self.step_calls_started = 0
        self.transitions_completed = 0
        self.effective_seed: int | None = None
        self.impulse_count = 0
        self.impulse_apply_transition: int | None = None
        self.impulse_apply_q9: int | None = None
        self.readback_error: float | None = None
        self.qvel_write_exact: bool | None = None
        self.target_qvel_sha256: str | None = None
        self.realized_qvel_sha256: str | None = None
        self.metric_values: list[float] = []
        self.metric_digest = hashlib.sha256()

    def on_construct(self, wrapper: Any) -> None:
        seed = getattr(wrapper.unwrapped.cfg, "seed", None)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RuntimeError("scale2 campaign effective seed invalid")
        self.effective_seed = seed

    def before_step(self, wrapper: Any) -> None:
        if self.spec.scenario != "disturbance" or self.transitions_completed != GLOBAL_IMPULSE_TRANSITION:
            return
        if self.impulse_count:
            raise RuntimeError("scale2 campaign impulse repeated")
        from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

        robot = wrapper.unwrapped.scene["robot"]
        before = robot.data.root_link_vel_w.detach().clone()
        if before.shape != (1, 6) or before.dtype != torch.float32 or not bool(torch.isfinite(before).all()):
            raise RuntimeError("scale2 campaign root velocity invalid")
        delta = torch.as_tensor(self.spec.impulse, dtype=torch.float32, device=before.device).reshape(1, 6)
        target = before + delta
        env_ids = torch.zeros(1, dtype=torch.long, device=before.device)
        qpos = robot.data.data.qpos
        qvel = robot.data.data.qvel
        q_indices = robot.data.indexing.free_joint_q_adr
        v_indices = robot.data.indexing.free_joint_v_adr
        rows = env_ids[:, None]
        quat = qpos[rows, q_indices[3:7]].detach().clone()
        target_qvel = torch.cat((target[:, :3], quat_apply_inverse(quat, target[:, 3:])), dim=-1)
        robot.write_root_link_velocity_to_sim(target, env_ids=env_ids)
        observed_qvel = qvel[rows, v_indices].detach().clone()
        observed_world = torch.cat((observed_qvel[:, :3], quat_apply(quat, observed_qvel[:, 3:])), dim=-1)
        self.readback_error = float(
            torch.max(torch.abs((observed_world - before).double() - delta.double())).item()
        )
        self.qvel_write_exact = bool(torch.equal(observed_qvel, target_qvel))
        target_np = target_qvel.cpu().contiguous().numpy().astype(np.float32, copy=False)
        observed_np = observed_qvel.cpu().contiguous().numpy().astype(np.float32, copy=False)
        self.target_qvel_sha256 = _sha256_bytes(target_np.tobytes(order="C"))
        self.realized_qvel_sha256 = _sha256_bytes(observed_np.tobytes(order="C"))
        command = wrapper.unwrapped.command_manager.get_term("motion")
        self.impulse_apply_transition = self.transitions_completed
        self.impulse_apply_q9 = recovery.evaluation._q9(command)  # noqa: SLF001
        self.impulse_count += 1

    def capture(self, evidence: Any) -> None:
        if type(evidence) is not recovery.evaluation.StepEvidence:
            raise RuntimeError("scale2 campaign evidence type mismatch")
        weighted = _finite(evidence.reward_rates["evaluator_aligned_recovery"], "recovery rate")
        metric = weighted / -25.0
        if metric < 0.0:
            raise RuntimeError("scale2 campaign recovery metric negative")
        index = len(self.metric_values)
        self.metric_values.append(metric)
        self.metric_digest.update(struct.pack("<qd", index, metric))

    def report(self) -> dict[str, Any]:
        baseline_mean: float | None = None
        threshold: float | None = None
        streak_end: int | None = None
        recovery_time: float | None = None
        recovered: bool | None = None
        if self.spec.scenario == "disturbance":
            recovered = False
            if len(self.metric_values) == TOTAL_TRANSITIONS:
                baseline_mean = sum(self.metric_values[index] for index in BASELINE_TRANSITIONS) / len(
                    BASELINE_TRANSITIONS
                )
                threshold = baseline_mean + float(self.contract["disturbance"]["recovery_margin"])
                consecutive = 0
                for index in range(GLOBAL_IMPULSE_TRANSITION, TOTAL_TRANSITIONS):
                    consecutive = consecutive + 1 if self.metric_values[index] <= threshold else 0
                    if consecutive == RECOVERY_STABLE_STEPS:
                        streak_end = index
                        break
                if streak_end is not None:
                    recovery_time = (streak_end - GLOBAL_IMPULSE_TRANSITION + 1) * CONTROL_DT_S
                    recovered = recovery_time <= float(self.contract["disturbance"]["maximum_recovery_time_s"])
        return {
            "effective_seed": self.effective_seed,
            "step_calls_started": self.step_calls_started,
            "transitions_completed": self.transitions_completed,
            "metric_sample_count": len(self.metric_values),
            "metric_trajectory_sha256": self.metric_digest.hexdigest(),
            "impulse_count": self.impulse_count,
            "impulse_apply_transition": self.impulse_apply_transition,
            "impulse_apply_q9": self.impulse_apply_q9,
            "impulse_qvel_write_exact": self.qvel_write_exact,
            "impulse_readback_max_absolute_error": self.readback_error,
            "target_root_qvel_sha256": self.target_qvel_sha256,
            "realized_root_qvel_sha256": self.realized_qvel_sha256,
            "baseline_mean": baseline_mean,
            "recovery_threshold": threshold,
            "stable_streak_end_transition": streak_end,
            "recovery_time_s": recovery_time,
            "recovered": recovered,
        }


@contextmanager
def _instrumented_run(spec: Any, probe: _StepProbe, cached_preflight: Mapping[str, Any]):
    import mjlab.rl as mjlab_rl

    original_wrapper = mjlab_rl.RslRlVecEnvWrapper
    original_preflight = recovery._preflight_internal  # noqa: SLF001
    original_step_evidence = recovery.evaluation._step_evidence  # noqa: SLF001

    class InstrumentedWrapper(original_wrapper):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            probe.on_construct(self)

        def step(self, actions: Any) -> Any:
            probe.before_step(self)
            probe.step_calls_started += 1
            result = super().step(actions)
            probe.transitions_completed += 1
            return result

    def captured_step_evidence(raw_env: Any, velocity_limits: np.ndarray) -> Any:
        evidence = original_step_evidence(raw_env, velocity_limits)
        probe.capture(evidence)
        return evidence

    def cached(_: recovery.Scale2RecoveryRequest) -> Mapping[str, Any]:
        return copy.deepcopy(dict(cached_preflight))

    mjlab_rl.RslRlVecEnvWrapper = InstrumentedWrapper
    recovery._preflight_internal = cached  # type: ignore[assignment]  # noqa: SLF001
    recovery.evaluation._step_evidence = captured_step_evidence  # type: ignore[assignment]  # noqa: SLF001
    try:
        yield
    finally:
        recovery.evaluation._step_evidence = original_step_evidence  # type: ignore[assignment]  # noqa: SLF001
        recovery._preflight_internal = original_preflight  # type: ignore[assignment]  # noqa: SLF001
        mjlab_rl.RslRlVecEnvWrapper = original_wrapper


def _assess_run(
    spec: Any, report: Mapping[str, Any], probe: Mapping[str, Any], base_contract: Mapping[str, Any]
) -> dict[str, Any]:
    issues: list[str] = []

    def require(condition: bool, issue: str) -> None:
        if not condition:
            issues.append(issue)

    qualification = _mapping(report.get("qualification"), "scale2 run qualification")
    rollout = _mapping(report.get("rollout"), "scale2 run rollout")
    support = _mapping(report.get("support_summary"), "scale2 run support")
    parity = _mapping(report.get("teacher", {}).get("parity"), "scale2 teacher parity")
    action = _mapping(report.get("action_semantics"), "scale2 action semantics")
    composite = _mapping(report.get("teacher_composite_action_chain"), "scale2 teacher composite")
    selected = _mapping(report.get("selected124_external_state"), "scale2 selected state")
    terminal = _mapping(report.get("first_done"), "scale2 terminal")
    totals = _mapping(rollout.get("safety_count_totals"), "scale2 safety totals")
    gates = _mapping(base_contract.get("gates"), "base campaign gates")
    require(report.get("effective_seed") == spec.seed, "effective_seed")
    require(qualification.get("passed") is True, "qualification")
    require(report.get("completed_transitions") == TOTAL_TRANSITIONS, "completed_transitions")
    require(report.get("simulator_step_calls_started") == TOTAL_TRANSITIONS, "step_calls")
    require(
        report.get("controller_counts") == {"student": STUDENT_TRANSITIONS, "teacher": TEACHER_TRANSITIONS},
        "partition",
    )
    require(report.get("partial_failure") is None, "partial_failure")
    require(report.get("bindings_unchanged") is True, "bindings")
    require(terminal.get("transition") == 509 and terminal.get("q9_before") == 518, "terminal_index")
    require(terminal.get("termination_names") == ["time_out"], "terminal_reason")
    require(rollout.get("hard_safety_violation_count") == 0, "hard_safety")
    require(rollout.get("soft_safety_warning_count") == 0, "soft_safety")
    for name in gates["required_rollout_zero_counts"]:
        require(totals.get(name) == 0, f"rollout_zero:{name}")
    for name in (
        "termination_count",
        "q9_discontinuity_count",
        "nonfinite_count",
        "raw_clip_required_count",
        "action_semantics_mismatch_count",
        "target_soft_limit_violation_count",
        "actuator_target_soft_limit_violation_count",
        "measured_soft_limit_violation_count",
        "joint_velocity_limit_violation_count",
    ):
        require(support.get(name) == 0, f"support_zero:{name}")
    require(parity.get("passed") is True and parity.get("violation_count") == 0, "teacher_parity")
    require(action.get("passed") is True and action.get("mismatch_count") == 0, "action_chain")
    require(composite.get("passed") is True and composite.get("mismatch_count") == 0, "teacher_chain")
    require(selected.get("mismatch_count") == 0 and selected.get("handoff_observed") is True, "selected_state")
    require(
        float(support.get("minimum_base_height_m", -math.inf)) >= float(gates["minimum_base_height_m"]), "height"
    )
    require(float(support.get("maximum_base_tilt_rad", math.inf)) <= float(gates["maximum_base_tilt_rad"]), "tilt")
    require(
        float(support.get("maximum_joint_velocity_ratio", math.inf))
        <= float(gates["maximum_joint_velocity_ratio"]),
        "velocity",
    )
    require(
        float(support.get("maximum_tracking_rmse_rad", math.inf)) <= float(gates["maximum_tracking_rmse_rad"]),
        "tracking",
    )
    require(float(support.get("maximum_plain_sonic_raw_native_abs", math.inf)) < 10.0, "raw_abs")
    require(probe.get("effective_seed") == spec.seed, "probe_seed")
    require(probe.get("step_calls_started") == TOTAL_TRANSITIONS, "probe_steps")
    require(probe.get("transitions_completed") == TOTAL_TRANSITIONS, "probe_completed")
    require(probe.get("metric_sample_count") == TOTAL_TRANSITIONS, "probe_metric_count")
    if spec.scenario == "nominal":
        require(probe.get("impulse_count") == 0, "nominal_impulse")
    else:
        require(probe.get("impulse_count") == 1, "impulse_count")
        require(probe.get("impulse_apply_transition") == GLOBAL_IMPULSE_TRANSITION, "impulse_transition")
        require(probe.get("impulse_apply_q9") == IMPULSE_Q9, "impulse_q9")
        require(probe.get("impulse_qvel_write_exact") is True, "impulse_write")
        require(
            _finite(probe.get("impulse_readback_max_absolute_error"), "impulse readback") <= 1e-6,
            "impulse_readback",
        )
        require(probe.get("recovered") is True, "recovered")
    passed = not issues
    detail = {
        "scenario": spec.scenario,
        "index": spec.rollout_index,
        "seed": spec.seed,
        "report_sha256": _sha256_bytes(_canonical_bytes(report)),
        "issues": issues,
    }
    return {
        "run_id": spec.run_id,
        "scenario": spec.scenario,
        "rollout_index": spec.rollout_index,
        "seed": spec.seed,
        "passed": passed,
        "issue_count": len(issues),
        "first_issue": None if passed else issues[0],
        "detail_sha256": _sha256_bytes(_canonical_bytes(detail)),
        "effective_seed": probe.get("effective_seed"),
        "completed_transitions": report.get("completed_transitions"),
        "student_controller_count": report.get("controller_counts", {}).get("student"),
        "teacher_controller_count": report.get("controller_counts", {}).get("teacher"),
        "teacher_query_count": report.get("teacher", {}).get("query_count"),
        "terminal_q9": terminal.get("q9_before"),
        "hard_safety_violation_count": rollout.get("hard_safety_violation_count"),
        "soft_safety_warning_count": rollout.get("soft_safety_warning_count"),
        "minimum_base_height_m": support.get("minimum_base_height_m"),
        "maximum_base_tilt_rad": support.get("maximum_base_tilt_rad"),
        "maximum_joint_velocity_ratio": support.get("maximum_joint_velocity_ratio"),
        "maximum_tracking_rmse_rad": support.get("maximum_tracking_rmse_rad"),
        "teacher_parity_max_absolute_error": parity.get("maximum_absolute_error"),
        "impulse_count": probe.get("impulse_count"),
        "impulse_apply_transition": probe.get("impulse_apply_transition"),
        "impulse_apply_q9": probe.get("impulse_apply_q9"),
        "impulse_readback_max_absolute_error": probe.get("impulse_readback_max_absolute_error"),
        "metric_trajectory_sha256": probe.get("metric_trajectory_sha256"),
        "recovery_time_s": probe.get("recovery_time_s"),
        "recovered": probe.get("recovered"),
        "published_teacher_label_count": 0,
        "published_training_row_count": 0,
    }


def _failure_record(spec: Any, error: Exception, probe: Mapping[str, Any]) -> dict[str, Any]:
    detail = f"{type(error).__name__}:{error}"
    return {
        "run_id": spec.run_id,
        "scenario": spec.scenario,
        "rollout_index": spec.rollout_index,
        "seed": spec.seed,
        "passed": False,
        "issue_count": 1,
        "first_issue": type(error).__name__,
        "detail_sha256": _sha256_bytes(detail.encode()),
        "effective_seed": probe.get("effective_seed"),
        "completed_transitions": probe.get("transitions_completed", 0),
        "student_controller_count": None,
        "teacher_controller_count": None,
        "teacher_query_count": None,
        "terminal_q9": None,
        "hard_safety_violation_count": None,
        "soft_safety_warning_count": None,
        "minimum_base_height_m": None,
        "maximum_base_tilt_rad": None,
        "maximum_joint_velocity_ratio": None,
        "maximum_tracking_rmse_rad": None,
        "teacher_parity_max_absolute_error": None,
        "impulse_count": probe.get("impulse_count", 0),
        "impulse_apply_transition": probe.get("impulse_apply_transition"),
        "impulse_apply_q9": probe.get("impulse_apply_q9"),
        "impulse_readback_max_absolute_error": probe.get("impulse_readback_max_absolute_error"),
        "metric_trajectory_sha256": probe.get("metric_trajectory_sha256"),
        "recovery_time_s": probe.get("recovery_time_s"),
        "recovered": probe.get("recovered"),
        "published_teacher_label_count": 0,
        "published_training_row_count": 0,
    }


def run_campaign(
    request: CampaignRequest, progress: Callable[[Mapping[str, Any]], None] | None = None
) -> dict[str, Any]:
    preflight_value = _preflight_internal(request)
    root = request.root
    contract = preflight_value["contract"]
    base_contract = preflight_value["prerequisites"]["base_contract"]
    cached = preflight_value["recovery_preflight"]
    specs: Sequence[Any] = preflight_value["specs"]
    records: list[dict[str, Any]] = []
    for spec in specs:
        probe = _StepProbe(spec, contract)
        try:
            random.seed(spec.seed)
            np.random.seed(spec.seed % (2**32))
            torch.manual_seed(spec.seed)
            torch.cuda.manual_seed_all(spec.seed)
            run_request = _base_request(root, spec.seed)
            with _instrumented_run(spec, probe, cached):
                full_report = recovery.run(run_request)
            record = _assess_run(spec, full_report, probe.report(), base_contract)
        except Exception as error:
            record = _failure_record(spec, error, probe.report())
        records.append(record)
        if progress is not None:
            progress(record)
        if record["passed"] is not True:
            break
    sources_after = _source_binding(root)
    passed_count = sum(record["passed"] is True for record in records)
    nominal_passed = sum(record["passed"] is True and record["scenario"] == "nominal" for record in records)
    disturbance_passed = sum(
        record["passed"] is True and record["scenario"] == "disturbance" for record in records
    )
    recovered_count = sum(
        record["passed"] is True and record["scenario"] == "disturbance" and record["recovered"] is True
        for record in records
    )
    seeds = [record["effective_seed"] for record in records]
    qualified = bool(
        len(records) == 20
        and passed_count == nominal_passed + disturbance_passed == 20
        and nominal_passed == disturbance_passed == recovered_count == 10
        and len(set(seeds)) == 20
        and sources_after == preflight_value["sources"]
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "campaign_qualified": qualified,
        "verdict": "scale2_q250_recovery_campaign_passed" if qualified else "scale2_q250_recovery_campaign_failed",
        "campaign_contract_sha256": CONTRACT_SHA256,
        "q250_prerequisite_report_sha256": contract["prerequisites"]["q250_pass_report_sha256"],
        "planned_run_count": 20,
        "completed_run_count": len(records),
        "passed_run_count": passed_count,
        "nominal_passed_count": nominal_passed,
        "disturbance_passed_count": disturbance_passed,
        "disturbance_recovered_count": recovered_count,
        "first_failed_run_id": next((record["run_id"] for record in records if not record["passed"]), None),
        "executed_source_binding_before_sha256": preflight_value["sources"]["binding_sha256"],
        "executed_source_binding_after_sha256": sources_after["binding_sha256"],
        "sources_unchanged": sources_after == preflight_value["sources"],
        "runs": records,
        "fresh_disjoint_suffix_collection_authorized": qualified,
        "teacher_support_qualified": qualified,
        "published_teacher_label_count": 0,
        "published_training_row_count": 0,
        "dagger_data": False,
        "training_performed": False,
        "promotion_or_deployment": False,
        "hardware_authorized": False,
    }
    if any(isinstance(value, (np.ndarray, torch.Tensor)) for value in report.values()):
        raise RuntimeError("scale2 campaign publication contains forbidden array")
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("kind") != REPORT_KIND:
        raise ValueError("scale2 campaign report identity mismatch")
    records = report.get("runs")
    if not isinstance(records, list) or len(records) != report.get("completed_run_count"):
        raise ValueError("scale2 campaign run list mismatch")
    qualified = report.get("campaign_qualified") is True
    recomputed = bool(
        len(records) == 20
        and all(isinstance(record, Mapping) and record.get("passed") is True for record in records)
        and sum(record.get("scenario") == "nominal" for record in records) == 10
        and sum(record.get("scenario") == "disturbance" and record.get("recovered") is True for record in records)
        == 10
        and report.get("sources_unchanged") is True
    )
    if qualified != recomputed:
        raise ValueError("scale2 campaign qualification recomputation mismatch")
    if (
        report.get("fresh_disjoint_suffix_collection_authorized") is not qualified
        or report.get("teacher_support_qualified") is not qualified
        or report.get("published_teacher_label_count") != 0
        or report.get("published_training_row_count") != 0
        or report.get("training_performed") is not False
        or report.get("hardware_authorized") is not False
    ):
        raise ValueError("scale2 campaign claim boundary mismatch")


def write_json_exclusive(path: Path, report: Mapping[str, Any]) -> None:
    validate_report(report)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise
