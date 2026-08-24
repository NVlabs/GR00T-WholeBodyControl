"""Paired failed-seed probe for noisy versus clean SONIC observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import struct
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment
from gear_sonic.envs.mjlab.sonic_true23_student_qualification import (
    audit_sonic_true23_student_qualification_env_cfg,
    make_sonic_true23_student_qualification_env_cfg,
)
from gear_sonic.scripts import train_g1_true23_sonic_survival_length_score as scale2
from gear_sonic.trl.mjlab import (
    sonic_task_space_ppo_full_support_runner as fs,
    sonic_task_space_ppo_runner as task,
)
from gear_sonic.utils import (
    g1_true23_sonic_scale2_teacher_recovery as recovery,
    g1_true23_sonic_student_closed_loop_qualification as student,
    g1_true23_sonic_student_teacher_recovery as legacy,
)
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import load_composite_mjlab_contract

SCHEMA_VERSION = 1
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_scale2_observation_corruption_probe_v1.json"
)
CONTRACT_SHA256 = "0a9ba55b5e0f1f8fc9f446de4ec3e647dfeab2cc8feacab5f07788bff1e1f7d9"
REPORT_KIND = "g1_true23_sonic_scale2_observation_corruption_probe_report_v1"
SEED = 611723381
TRANSITIONS = 510
ACTION_DIM = 23
EE_BODY_NAMES = task.EE_TERMINATION_BODY_NAMES
EXECUTED_SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_sonic_scale2_observation_corruption_probe.py"),
    Path("gear_sonic/scripts/probe_g1_true23_sonic_scale2_observation_corruption.py"),
    Path("gear_sonic/utils/g1_true23_sonic_scale2_teacher_recovery.py"),
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json(path: Path, context: str) -> Mapping[str, Any]:
    return fs._strict_json(path, context)  # noqa: SLF001


def _source_binding(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in EXECUTED_SOURCE_RELATIVE_PATHS:
        path = (root / relative).resolve(strict=True)
        if path.is_symlink() or not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"corruption probe source invalid: {relative.as_posix()}")
        files.append({"path": relative.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"files": files, "binding_sha256": _sha256_bytes(_canonical_bytes(files))}


@dataclass(frozen=True)
class ProbeRequest:
    repository_root: Path
    output: Path

    @property
    def root(self) -> Path:
        value = self.repository_root.expanduser().resolve(strict=True)
        if value.is_symlink() or not value.is_dir():
            raise ValueError("corruption probe repository root invalid")
        return value

    @property
    def output_path(self) -> Path:
        candidate = self.output.expanduser()
        candidate = candidate if candidate.is_absolute() else self.root / candidate
        value = candidate.resolve(strict=False)
        if not value.is_relative_to(self.root) or value.suffix.lower() != ".json":
            raise ValueError("corruption probe output must be repository-contained JSON")
        if candidate.is_symlink() or value.is_symlink() or value.parent.resolve(strict=True).is_symlink():
            raise ValueError("corruption probe output path invalid")
        return value


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("corruption probe contract mismatch")
    contract = _strict_json(path, "corruption probe contract")
    modes = contract.get("modes")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != "g1_true23_sonic_scale2_observation_corruption_probe_contract_v1"
        or contract.get("seed") != SEED
        or contract.get("source", {}).get("checkpoint_sha256") != scale2.SOURCE_CHECKPOINT_SHA256
        or contract.get("source", {}).get("policy_state_sha256") != scale2.SOURCE_POLICY_SHA256
        or not isinstance(modes, list)
        or [mode.get("name") for mode in modes if isinstance(mode, Mapping)] != ["noisy", "clean"]
        or modes[0].get("required_completed_transitions") != 206
        or modes[0].get("required_terminal_q9") != 214
        or contract.get("episode", {}).get("teacher_present") is not False
    ):
        raise ValueError("corruption probe contract semantics mismatch")
    return contract


def _verify_prerequisites(root: Path, contract: Mapping[str, Any]) -> None:
    prerequisites = contract["prerequisites"]
    for prefix in ("campaign_failure", "q250_pass"):
        path = (root / prerequisites[f"{prefix}_relative_path"]).resolve(strict=True)
        if path.is_symlink() or sha256_file(path) != prerequisites[f"{prefix}_sha256"]:
            raise ValueError(f"corruption probe {prefix} bytes mismatch")
    campaign = _strict_json(
        (root / prerequisites["campaign_failure_relative_path"]).resolve(strict=True),
        "corruption probe campaign failure",
    )
    run = campaign.get("runs", [None])[0]
    if (
        campaign.get("campaign_qualified") is not False
        or campaign.get("published_teacher_label_count") != 0
        or not isinstance(run, Mapping)
        or run.get("run_id") != prerequisites["campaign_failed_run_id"]
        or run.get("completed_transitions") != prerequisites["campaign_observed_completed_transitions"]
        or run.get("terminal_q9") != prerequisites["campaign_observed_terminal_q9"]
    ):
        raise ValueError("corruption probe campaign prerequisite semantics mismatch")


def _preflight_internal(request: ProbeRequest) -> dict[str, Any]:
    root = request.root
    contract = load_contract(root)
    _verify_prerequisites(root, contract)
    sources = _source_binding(root)
    recovery_request = recovery.Scale2RecoveryRequest(
        root,
        root / "artifacts/g1_true23/.corruption-probe-unused.json",
        "q250",
        runtime_seed=SEED,
    )
    base = recovery._preflight_internal(recovery_request)  # noqa: SLF001
    if base.get("ready") is not True:
        raise RuntimeError("corruption probe scale2 preflight not ready")
    return {"root": root, "contract": contract, "sources": sources, "base": base}


def preflight(request: ProbeRequest) -> dict[str, Any]:
    try:
        value = _preflight_internal(request)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "g1_true23_sonic_scale2_observation_corruption_probe_preflight_v1",
            "ready": True,
            "seed": SEED,
            "contract_sha256": CONTRACT_SHA256,
            "executed_source_binding_sha256": value["sources"]["binding_sha256"],
            "simulator_constructed": False,
            "simulator_steps": 0,
            "teacher_queries": 0,
            "training_performed": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "g1_true23_sonic_scale2_observation_corruption_probe_preflight_v1",
            "ready": False,
            "error": {"type": type(error).__name__, "detail_sha256": _sha256_bytes(str(error).encode())},
            "simulator_constructed": False,
            "simulator_steps": 0,
            "teacher_queries": 0,
            "training_performed": False,
            "hardware_authorized": False,
        }


def _terminal_ee_values(terminal: Mapping[str, Any]) -> np.ndarray:
    values = terminal.get("ee_body_position_errors")
    if not isinstance(values, list) or [item.get("name") for item in values] != list(EE_BODY_NAMES):
        raise RuntimeError("corruption probe terminal EE order mismatch")
    result = np.asarray([item["absolute_z_error_m"] for item in values], dtype=np.float32)
    if result.shape != (len(EE_BODY_NAMES),) or not np.isfinite(result).all():
        raise RuntimeError("corruption probe terminal EE values invalid")
    return result


def _run_one(*, internal: Mapping[str, Any], corruption_enabled: bool, runtime_seed: int = SEED) -> dict[str, Any]:
    root = internal["root"]
    base = internal["base"]
    student_base = base["base"]["student"]
    motion = Path(student_base["fixed_inputs"]["motion_path"])
    topology_contract = fs.load_task_space_ppo_contract(root)
    topology = (root / topology_contract["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(
        strict=True
    )
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(student.FIXED_GPU),
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
        }
    )
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("corruption probe requires fixed CUDA device 0")
    if isinstance(runtime_seed, bool) or not isinstance(runtime_seed, int) or not 0 <= runtime_seed < 2**31:
        raise ValueError("corruption probe runtime seed must be uint31")
    random.seed(runtime_seed)
    np.random.seed(runtime_seed % (2**32))
    torch.manual_seed(runtime_seed)
    torch.cuda.manual_seed_all(runtime_seed)
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    actor = scale2._actor(base["policy_state"], topology)  # noqa: SLF001
    actor.eval()
    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": actor.export_true23_policy_state()}, reference_profile=fs.REFERENCE_PROFILE
    )
    if policy_hash != scale2.SOURCE_POLICY_SHA256:
        raise RuntimeError("corruption probe policy identity mismatch")
    velocity_contract = load_composite_mjlab_contract(root)
    velocity_limits = np.asarray(
        velocity_contract["nominal_gate"]["velocity_limit_hardware_radps"], dtype=np.float32
    )
    cfg = make_sonic_true23_student_qualification_env_cfg(
        motion_file=str(motion), num_envs=1, anchor_q9=9, transitions=TRANSITIONS
    )
    cfg.seed = runtime_seed
    noisy_audit = audit_sonic_true23_student_qualification_env_cfg(
        cfg, expected_anchor_q9=9, expected_transitions=TRANSITIONS
    )
    if not corruption_enabled:
        cfg.observations["tokenizer"].enable_corruption = False
        cfg.observations["policy"].enable_corruption = False
    observation_semantics = {
        "tokenizer_corruption_enabled": cfg.observations["tokenizer"].enable_corruption,
        "policy_corruption_enabled": cfg.observations["policy"].enable_corruption,
        "noise_term_definitions_preserved": True,
        "base_noisy_audit_sha256": _sha256_bytes(_canonical_bytes(noisy_audit)),
    }
    env = ManagerBasedRlEnv(cfg=cfg, device=student.DEVICE)
    recorder: legacy._RecoveryTerminalRecorder | None = None  # noqa: SLF001
    actions: list[np.ndarray] = []
    ee_values: list[np.ndarray] = []
    action_digest = hashlib.sha256()
    ee_digest = hashlib.sha256()
    episode_return = 0.0
    completed = 0
    terminal_q9: int | None = None
    termination_names: list[str] = []
    nonfinite_count = 0
    q9_discontinuity_count = 0
    raw_clip_required_count = 0
    action_semantics_mismatch_count = 0
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        if wrapped.max_episode_length != TRANSITIONS or wrapped.clip_actions is not None:
            raise RuntimeError("corruption probe wrapper mismatch")
        prime = prime_sonic_true23_training_environment(wrapped)
        observations = wrapped.get_observations()
        initial_encoder, initial_policy = student._observation_arrays(observations)  # noqa: SLF001
        command = env.command_manager.get_term("motion")
        recorder = legacy._RecoveryTerminalRecorder(env, velocity_limits)  # noqa: SLF001
        with torch.inference_mode():
            for transition in range(TRANSITIONS):
                q9 = recovery.evaluation._q9(command)  # noqa: SLF001
                if q9 != 9 + transition:
                    q9_discontinuity_count += 1
                    break
                student._observation_arrays(observations)  # noqa: SLF001
                raw = actor(observations, stochastic_output=False)
                if raw.shape != (1, ACTION_DIM) or not bool(torch.isfinite(raw).all()):
                    nonfinite_count += 1
                    break
                raw_np = raw.detach().cpu().contiguous().numpy()[0].astype(np.float32, copy=True)
                actions.append(raw_np)
                action_digest.update(struct.pack("<q", q9))
                action_digest.update(raw_np.tobytes(order="C"))
                recorder.arm(transition=transition, q9_before=q9, controller="student", behavior_raw=raw_np)
                observations, rewards, dones, extras = wrapped.step(raw)
                if not bool(torch.isfinite(rewards).all()):
                    nonfinite_count += 1
                    break
                episode_return += float(rewards[0].detach().cpu().item())
                completed += 1
                done = bool(int(dones[0].detach().cpu().item()))
                terminal = recorder.finish(done=done)
                if done:
                    if terminal is None or terminal.get("capture_errors"):
                        raise RuntimeError("corruption probe terminal capture failed")
                    semantics = terminal["action_semantics"]
                    ee = _terminal_ee_values(terminal)
                    terminal_q9 = q9
                    termination_names = list(terminal["termination_names"])
                else:
                    if terminal is not None or recovery.evaluation._extra_termination_names(extras):  # noqa: SLF001
                        raise RuntimeError("corruption probe nonterminal termination drift")
                    semantics = student._action_semantics(env, raw_np)  # noqa: SLF001
                    ee = (
                        task._body_z_errors(env, "motion", EE_BODY_NAMES)  # noqa: SLF001
                        .detach()
                        .cpu()
                        .contiguous()
                        .numpy()[0]
                        .astype(np.float32, copy=True)
                    )
                action_semantics_mismatch_count += int(semantics.get("passed") is not True)
                raw_clip_required_count += int(semantics.get("raw_clip_coordinate_count", 1))
                ee_values.append(ee)
                ee_digest.update(struct.pack("<q", q9))
                ee_digest.update(ee.tobytes(order="C"))
                if done:
                    break
        return {
            "mode": "noisy" if corruption_enabled else "clean",
            "seed": runtime_seed,
            "observation_semantics": observation_semantics,
            "initial_encoder267_sha256": _sha256_bytes(initial_encoder.tobytes(order="C")),
            "initial_policy930_sha256": _sha256_bytes(initial_policy.tobytes(order="C")),
            "prime": prime,
            "policy_state_sha256": policy_hash,
            "completed_transitions": completed,
            "terminal_q9": terminal_q9,
            "termination_names": termination_names,
            "episode_return_diagnostic_only": episode_return,
            "nonfinite_count": nonfinite_count,
            "q9_discontinuity_count": q9_discontinuity_count,
            "raw_clip_required_count": raw_clip_required_count,
            "action_semantics_mismatch_count": action_semantics_mismatch_count,
            "action_series_sha256": action_digest.hexdigest(),
            "ee_z_series_sha256": ee_digest.hexdigest(),
            "maximum_absolute_raw_action": max((float(np.max(np.abs(item))) for item in actions), default=None),
            "maximum_ee_z_error_m": max((float(np.max(item)) for item in ee_values), default=None),
            "_actions": actions,
            "_ee": ee_values,
        }
    finally:
        if recorder is not None:
            recorder.restore()
        env.close()


def _first_threshold_q9(differences: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(differences >= threshold)
    return None if not len(indices) else 9 + int(indices[0])


def _compare(noisy: Mapping[str, Any], clean: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    count = min(len(noisy["_actions"]), len(clean["_actions"]))
    noisy_actions = np.asarray(noisy["_actions"][:count], dtype=np.float32)
    clean_actions = np.asarray(clean["_actions"][:count], dtype=np.float32)
    noisy_ee = np.asarray(noisy["_ee"][:count], dtype=np.float32)
    clean_ee = np.asarray(clean["_ee"][:count], dtype=np.float32)
    action_diff = np.max(np.abs(noisy_actions - clean_actions), axis=1)
    ee_diff = np.max(np.abs(noisy_ee - clean_ee), axis=1)
    comparison = contract["comparison"]
    return {
        "common_transition_count": count,
        "common_first_q9": 9 if count else None,
        "common_last_q9": 8 + count if count else None,
        "maximum_raw_action_linf_difference": float(np.max(action_diff)) if count else None,
        "maximum_ee_z_linf_difference_m": float(np.max(ee_diff)) if count else None,
        "first_raw_action_linf_threshold_q9": {
            str(threshold): _first_threshold_q9(action_diff, float(threshold))
            for threshold in comparison["raw_action_linf_thresholds"]
        },
        "first_ee_z_linf_threshold_q9": {
            str(threshold): _first_threshold_q9(ee_diff, float(threshold))
            for threshold in comparison["ee_z_linf_thresholds_m"]
        },
    }


def run(request: ProbeRequest) -> dict[str, Any]:
    internal = _preflight_internal(request)
    noisy = _run_one(internal=internal, corruption_enabled=True)
    noisy_reproduced = bool(
        noisy["completed_transitions"] == 206
        and noisy["terminal_q9"] == 214
        and noisy["nonfinite_count"] == 0
        and noisy["q9_discontinuity_count"] == 0
        and noisy["raw_clip_required_count"] == 0
        and noisy["action_semantics_mismatch_count"] == 0
    )
    clean: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    if noisy_reproduced:
        clean = _run_one(internal=internal, corruption_enabled=False)
        comparison = _compare(noisy, clean, internal["contract"])
    sources_after = _source_binding(internal["root"])
    sources_unchanged = sources_after == internal["sources"]
    clean_passed = bool(
        clean is not None
        and clean["completed_transitions"] == 510
        and clean["terminal_q9"] == 518
        and clean["termination_names"] == ["time_out"]
        and all(
            clean[name] == 0
            for name in (
                "nonfinite_count",
                "q9_discontinuity_count",
                "raw_clip_required_count",
                "action_semantics_mismatch_count",
            )
        )
    )
    verdict = (
        "observation_corruption_is_decisive_failed_seed_cause"
        if noisy_reproduced and clean_passed and sources_unchanged
        else "observation_corruption_not_sufficiently_proven"
    )
    for result in (noisy, clean):
        if result is not None:
            result.pop("_actions")
            result.pop("_ee")
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "contract_sha256": CONTRACT_SHA256,
        "seed": SEED,
        "noisy_reproduced_campaign_failure": noisy_reproduced,
        "clean_completed_full_episode": clean_passed,
        "verdict": verdict,
        "noisy": noisy,
        "clean": clean,
        "comparison": comparison,
        "executed_source_binding_before_sha256": internal["sources"]["binding_sha256"],
        "executed_source_binding_after_sha256": sources_after["binding_sha256"],
        "sources_unchanged": sources_unchanged,
        "training_performed": False,
        "optimizer_steps": 0,
        "teacher_queries": 0,
        "teacher_labels": 0,
        "support_qualified": False,
        "promotion_or_deployment": False,
        "hardware_authorized": False,
        "network_or_external_actuation": False,
    }
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("kind") != REPORT_KIND:
        raise ValueError("corruption probe report identity mismatch")
    noisy = report.get("noisy")
    if not isinstance(noisy, Mapping):
        raise ValueError("corruption probe missing noisy result")
    reproduced = bool(
        noisy.get("completed_transitions") == 206
        and noisy.get("terminal_q9") == 214
        and all(
            noisy.get(name) == 0
            for name in (
                "nonfinite_count",
                "q9_discontinuity_count",
                "raw_clip_required_count",
                "action_semantics_mismatch_count",
            )
        )
    )
    if report.get("noisy_reproduced_campaign_failure") is not reproduced:
        raise ValueError("corruption probe noisy reproduction mismatch")
    clean = report.get("clean")
    clean_pass = bool(
        isinstance(clean, Mapping)
        and clean.get("completed_transitions") == 510
        and clean.get("terminal_q9") == 518
        and clean.get("termination_names") == ["time_out"]
        and all(
            clean.get(name) == 0
            for name in (
                "nonfinite_count",
                "q9_discontinuity_count",
                "raw_clip_required_count",
                "action_semantics_mismatch_count",
            )
        )
    )
    if report.get("clean_completed_full_episode") is not clean_pass:
        raise ValueError("corruption probe clean result mismatch")
    if clean is not None and not reproduced:
        raise ValueError("corruption probe ran clean mode without noisy reproduction")
    expected_verdict = (
        "observation_corruption_is_decisive_failed_seed_cause"
        if reproduced and clean_pass and report.get("sources_unchanged") is True
        else "observation_corruption_not_sufficiently_proven"
    )
    if report.get("verdict") != expected_verdict:
        raise ValueError("corruption probe verdict mismatch")
    if any(
        report.get(key) != expected
        for key, expected in {
            "training_performed": False,
            "optimizer_steps": 0,
            "teacher_queries": 0,
            "teacher_labels": 0,
            "support_qualified": False,
            "promotion_or_deployment": False,
            "hardware_authorized": False,
            "network_or_external_actuation": False,
        }.items()
    ):
        raise ValueError("corruption probe boundary mismatch")


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
        path.unlink(missing_ok=True)
        raise
