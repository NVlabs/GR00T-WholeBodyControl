"""Truthful promotion evidence for causal true23 MJLab checkpoints.

This is deliberately separate from the legacy ``*.promotion.pt`` bridge.
The causal training checkpoint keeps its original filename and role.  A new
JSON promotion is created only after re-verifying the checkpoint, schema-2
ONNX pair, and every record in a complete MuJoCo campaign.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    CAUSAL_HISTORY_PROFILE,
    causal_history_profile_contract,
)
from gear_sonic.utils import (
    g1_23dof_mjlab_diagnostic_mujoco as diagnostic,
    g1_23dof_mujoco_sim2sim as sim2sim,
)
from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from gear_sonic.utils.g1_23dof_contract import ROBOT_MODEL
from gear_sonic.utils.g1_23dof_mjlab_diagnostic_onnx import (
    verify_mjlab_diagnostic_onnx,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    safe_target_transform_contract,
)

CAUSAL_PROMOTION_SCHEMA_VERSION = 2
CAUSAL_PROMOTION_KIND = "g1_true23_causal_mujoco_promotion"
CAUSAL_PROMOTION_STAGE = "causal_mujoco_full"
APPLIED_SAFE_NATIVE_ACTION = "applied_safe_native_action"
_REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_SEEDS = (1729, 2718, 3141)
FULL_SCENARIOS = ("nominal", "push_50", "push_100", "domain_push_100")
FULL_RUN_COUNT = 12
FULL_EPISODES_PER_SCENARIO = 66
FULL_TOTAL_EPISODES = 264
FULL_TOTAL_RECORDS = 66_000
MAXIMUM_RECOVERY_TIME_S = 2.0

_PROMOTION_KEYS = {
    "schema_version",
    "kind",
    "robot_model",
    "promotion_stage",
    "deployment_bytes_authorized",
    "active_motor_control_authorized",
    "gantry_or_rated_support_required",
    "free_standing_authorized",
    "required_mode_machine",
    "decoder_output_semantics",
    "previous_action_semantics",
    "external_safe_target_transform_allowed",
    "safe_target_transform_sha256",
    "source_artifact",
    "full_campaign_evidence",
    "promotion_payload_sha256",
}
_SOURCE_KEYS = {
    "checkpoint_filename",
    "checkpoint_sha256",
    "checkpoint_update_count",
    "lineage_sha256",
    "policy_state_sha256",
    "reference_profile",
    "causal_reference_contract_sha256",
    "encoder_onnx_filename",
    "encoder_onnx_sha256",
    "decoder_onnx_filename",
    "decoder_onnx_sha256",
    "metadata_filename",
    "metadata_sha256",
    "metadata_payload_sha256",
}
_CAMPAIGN_KEYS = {
    "campaign_layout",
    "aggregate_report_filename",
    "aggregate_report_sha256",
    "aggregate_report_payload_sha256",
    "trace_manifest_sha256",
    "shard_manifest_sha256",
    "shards",
    "deterministic_seeds",
    "scenarios",
    "run_count",
    "episodes_per_scenario",
    "total_episodes",
    "total_records",
    "all_strict_gates_pass",
    "summary_metrics",
    "provenance",
}
_PROVENANCE_KEYS = {
    "base_sim2sim_config_relpath",
    "base_sim2sim_config_sha256",
    "mjcf_relpath",
    "mjcf_sha256",
    "domain_source_relpath",
    "domain_source_sha256",
    "mjlab_env_relpath",
    "mjlab_env_sha256",
    "executed_producer_archive_manifest_relpath",
    "executed_producer_archive_manifest_sha256",
    "executed_runtime_filename",
    "executed_runtime_sha256",
    "executed_runner_filename",
    "executed_runner_sha256",
}
_SUMMARY_KEYS = {
    "run_count",
    "episode_count",
    "record_count",
    "termination_count",
    "fall_count",
    "policy_output_nonfinite_count",
    "joint_limit_violation_count",
    "joint_velocity_bound_violation_count",
    "effort_bound_violation_count",
    "minimum_recovery_fraction",
    "maximum_recovery_time_s",
    "max_action_saturation_fraction",
    "max_joint_velocity_ratio",
    "max_effort_ratio",
    "max_abs_native_action_raw",
    "minimum_base_height_m",
    "maximum_tilt_rad",
}


def _strict_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_strict_json(path: Path, context: str) -> Mapping[str, Any]:
    path = _regular_file(path, context)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _lexical_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _reject_symlink_components(path: Path, context: str) -> None:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink():
            raise ValueError(f"{context} must not traverse symlinks")


def _regular_file(path: Path, context: str) -> Path:
    lexical = _lexical_absolute(path)
    _reject_symlink_components(lexical, context)
    if not lexical.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return lexical.resolve()


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} keys differ")


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} is not lowercase SHA-256")
    return value


def _safe_transform_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(safe_target_transform_contract()))


def _resolve_trace(report_path: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("trace file is invalid")
    root = report_path.parent.resolve()
    lexical_trace = root / relative
    _reject_symlink_components(lexical_trace, "trace")
    trace = lexical_trace.resolve()
    try:
        trace.relative_to(root)
    except ValueError as exc:
        raise ValueError("trace escapes report directory") from exc
    if not trace.is_file():
        raise ValueError("trace must be a regular non-symlink file")
    return trace


def _load_trace(
    report_path: Path,
    descriptor: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    _exact_keys(
        descriptor,
        {"file", "sha256", "payload_sha256", "record_count"},
        "trace descriptor",
    )
    trace_path = _resolve_trace(report_path, descriptor["file"])
    if sha256_file(trace_path) != _require_sha256(
        descriptor["sha256"], "trace sha256"
    ):
        raise ValueError("trace byte hash mismatch")
    records: list[Mapping[str, Any]] = []
    payload = hashlib.sha256()
    payload.update(b"[")
    with trace_path.open("rb") as stream:
        for index, line in enumerate(stream):
            if not line.endswith(b"\n"):
                raise ValueError("trace JSONL line lacks newline")
            try:
                record = json.loads(
                    line,
                    object_pairs_hook=_strict_object_pairs,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(f"non-finite trace token: {token}")
                    ),
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"trace line {index} is invalid") from exc
            if not isinstance(record, Mapping):
                raise ValueError("trace record must be an object")
            if index:
                payload.update(b",")
            payload.update(canonical_json_bytes(record).removesuffix(b"\n"))
            records.append(record)
    payload.update(b"]\n")
    if len(records) != descriptor["record_count"]:
        raise ValueError("trace record count mismatch")
    if payload.hexdigest() != _require_sha256(
        descriptor["payload_sha256"], "trace payload_sha256"
    ):
        raise ValueError("trace canonical payload hash mismatch")
    return records, {
        "file": descriptor["file"],
        "sha256": descriptor["sha256"],
        "payload_sha256": descriptor["payload_sha256"],
        "record_count": descriptor["record_count"],
    }


def _validate_source(
    report: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
) -> None:
    source = report.get("source_artifact")
    if not isinstance(source, Mapping):
        raise ValueError("full report source_artifact is missing")
    hashes = metadata["hashes"]
    expected = {
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": hashes["checkpoint_sha256"],
        "checkpoint_update_count": metadata["source"]["checkpoint_update_count"],
        "lineage_sha256": hashes["lineage_sha256"],
        "policy_state_sha256": hashes["policy_state_sha256"],
        "encoder_onnx_sha256": sha256_file(encoder_path),
        "decoder_onnx_sha256": sha256_file(decoder_path),
        "diagnostic_metadata_sha256": sha256_file(metadata_path),
        "diagnostic_metadata_payload_sha256": metadata["metadata_payload_sha256"],
        "reference_profile": CAUSAL_HISTORY_PROFILE,
        "decoder_output_semantics": APPLIED_SAFE_NATIVE_ACTION,
        "external_safe_target_transform_applied": False,
        "safe_target_transform": safe_target_transform_contract(),
    }
    for key, expected_value in expected.items():
        if source.get(key) != expected_value:
            raise ValueError(f"full report source binding mismatch: {key}")


def _validate_provenance(
    report_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    configuration = report.get("configuration")
    producer = report.get("producer")
    if not isinstance(configuration, Mapping) or not isinstance(producer, Mapping):
        raise ValueError("full report provenance is missing")
    _exact_keys(
        producer,
        {"kind", "version", "runtime_sha256", "runner_sha256"},
        "full report producer",
    )
    if (
        producer["kind"] != diagnostic.PRODUCER_KIND
        or producer["version"] != diagnostic.PRODUCER_VERSION
    ):
        raise ValueError("full report producer identity mismatch")

    expected_files = {
        "base_sim2sim_config": _REPO_ROOT / sim2sim.DEFAULT_CONFIG_RELPATH,
        "mjcf": _REPO_ROOT / sim2sim.DEFAULT_MJCF_RELPATH,
        "domain_source": _REPO_ROOT / diagnostic.DEFAULT_DOMAIN_SOURCE_RELPATH,
        "mjlab_env": _REPO_ROOT / diagnostic.DEFAULT_MJLAB_ENV_RELPATH,
    }
    resolved_files = {
        role: _regular_file(path, f"approved {role}")
        for role, path in expected_files.items()
    }
    report_paths = {
        "base_sim2sim_config": configuration.get("base_sim2sim_config_path"),
        "mjcf": configuration.get("mjcf_path"),
    }
    for role, reported in report_paths.items():
        if not isinstance(reported, str):
            raise ValueError(f"full report {role} path is invalid")
        if _regular_file(Path(reported), f"reported {role}") != resolved_files[role]:
            raise ValueError(f"full report {role} is not the approved repository file")
    expected_hash_fields = {
        "base_sim2sim_config": "base_sim2sim_config_sha256",
        "mjcf": "mjcf_sha256",
        "domain_source": "domain_source_sha256",
        "mjlab_env": "sonic_true23_env_sha256",
    }
    for role, field in expected_hash_fields.items():
        digest = sha256_file(resolved_files[role])
        if configuration.get(field) != digest:
            raise ValueError(f"full report {role} hash mismatch")

    archive_root = report_path.parent / "pre_shard_producer_source"
    manifest_path = _regular_file(
        archive_root / "manifest.json", "executed producer archive manifest"
    )
    runtime_name = Path(diagnostic.DEFAULT_RUNTIME_RELPATH).name
    runner_name = Path(diagnostic.DEFAULT_RUNNER_RELPATH).name
    runtime_path = _regular_file(
        archive_root / runtime_name, "archived executed runtime"
    )
    runner_path = _regular_file(
        archive_root / runner_name, "archived executed runner"
    )
    if any(path.stat().st_mode & 0o222 for path in (manifest_path, runtime_path, runner_path)):
        raise ValueError("executed producer archive must be read-only")
    manifest = _read_strict_json(manifest_path, "executed producer archive manifest")
    _exact_keys(
        manifest,
        {
            "frozen_report",
            "kind",
            "reconstruction",
            "report_rewritten",
            "runner_sha256",
            "runtime_sha256",
        },
        "executed producer archive manifest",
    )
    runtime_sha256 = sha256_file(runtime_path)
    runner_sha256 = sha256_file(runner_path)
    frozen_report = manifest.get("frozen_report")
    if (
        manifest["kind"] != "g1_true23_executed_producer_source_archive"
        or manifest["reconstruction"]
        != "exact_reverse_of_tested_seed_shard_patch_from_unmodified_running_source"
        or manifest["report_rewritten"] is not False
        or not isinstance(frozen_report, str)
        or Path(frozen_report).expanduser().resolve() != report_path
        or manifest["runtime_sha256"] != runtime_sha256
        or manifest["runner_sha256"] != runner_sha256
        or producer["runtime_sha256"] != runtime_sha256
        or producer["runner_sha256"] != runner_sha256
    ):
        raise ValueError("executed producer archive does not bind the full report")
    provenance = {
        "base_sim2sim_config_relpath": sim2sim.DEFAULT_CONFIG_RELPATH,
        "base_sim2sim_config_sha256": sha256_file(
            resolved_files["base_sim2sim_config"]
        ),
        "mjcf_relpath": sim2sim.DEFAULT_MJCF_RELPATH,
        "mjcf_sha256": sha256_file(resolved_files["mjcf"]),
        "domain_source_relpath": diagnostic.DEFAULT_DOMAIN_SOURCE_RELPATH,
        "domain_source_sha256": sha256_file(resolved_files["domain_source"]),
        "mjlab_env_relpath": diagnostic.DEFAULT_MJLAB_ENV_RELPATH,
        "mjlab_env_sha256": sha256_file(resolved_files["mjlab_env"]),
        "executed_producer_archive_manifest_relpath": str(
            manifest_path.relative_to(report_path.parent)
        ),
        "executed_producer_archive_manifest_sha256": sha256_file(manifest_path),
        "executed_runtime_filename": runtime_name,
        "executed_runtime_sha256": runtime_sha256,
        "executed_runner_filename": runner_name,
        "executed_runner_sha256": runner_sha256,
    }
    _exact_keys(provenance, _PROVENANCE_KEYS, "promotion provenance")
    return provenance


def _validate_full_report(
    report_path: Path,
    metadata: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    public_verification = diagnostic.verify_full_mjlab_diagnostic_report(report_path)
    report = _read_strict_json(report_path, "full MuJoCo report")
    if (
        report != public_verification["report"]
        or sha256_file(report_path) != public_verification["report_sha256"]
    ):
        raise ValueError("full report changed during public verification")
    required_root = {
        "schema_version",
        "kind",
        "active_motor_control_authorized",
        "deployment_ready",
        "diagnostic_only",
        "promotion_eligible",
        "robot_or_network_commands_performed",
        "allowed_uses",
        "forbidden_uses",
        "profile",
        "computed_pass",
        "promotion_assessment",
        "error",
        "source_artifact",
        "producer",
        "configuration",
        "simulator",
        "runs",
        "summary",
    }
    if "campaign_shard" in report:
        required_root.add("campaign_shard")
    _exact_keys(report, required_root, "full MuJoCo report")
    if (
        report["schema_version"] != 1
        or report["kind"] != diagnostic.REPORT_KIND
        or report["profile"] != "full"
        or report["computed_pass"] is not True
        or report["error"] is not None
        or report["diagnostic_only"] is not True
        or report["deployment_ready"] is not False
        or report["promotion_eligible"] is not False
        or report["active_motor_control_authorized"] is not False
        or report["robot_or_network_commands_performed"] is not False
    ):
        raise ValueError("full report flags do not describe passing offline evidence")
    if "campaign_shard" in report:
        shard = report["campaign_shard"]
        if (
            not isinstance(shard, Mapping)
            or shard.get("is_shard") is not False
            or shard.get("full_campaign_complete") is not True
            or shard.get("selected_seeds") != list(FULL_SEEDS)
        ):
            raise ValueError("monolithic full campaign shard declaration mismatch")
    assessment = report["promotion_assessment"]
    if (
        not isinstance(assessment, Mapping)
        or assessment.get("computed_pass") is not True
        or assessment.get("full_campaign_required") is not True
        or assessment.get("authorization_created") is not False
        or assessment.get("required_reference_profile") != CAUSAL_HISTORY_PROFILE
        or assessment.get("required_semantic_contract_sha256")
        != causal_history_profile_contract()["contract_sha256"]
        or assessment.get("required_scenarios") != list(FULL_SCENARIOS)
    ):
        raise ValueError("full report promotion assessment mismatch")
    _validate_source(
        report,
        metadata,
        checkpoint_path=checkpoint_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        metadata_path=metadata_path,
    )
    provenance = _validate_provenance(report_path, report)
    configuration = report["configuration"]
    if (
        not isinstance(configuration, Mapping)
        or configuration.get("profile_contract")
        != dict(diagnostic._PROFILE_CONTRACT["full"])  # noqa: SLF001
        or configuration.get("required_reference_profile")
        != CAUSAL_HISTORY_PROFILE
        or configuration.get("scenarios") != diagnostic._SCENARIOS  # noqa: SLF001
        or configuration.get("domain_randomization")
        != diagnostic._DOMAIN_CONTRACT  # noqa: SLF001
    ):
        raise ValueError("full report configuration mismatch")
    base_config_path = Path(configuration["base_sim2sim_config_path"])
    if (
        not base_config_path.is_file()
        or sha256_file(base_config_path)
        != configuration["base_sim2sim_config_sha256"]
    ):
        raise ValueError("full report base configuration changed")
    campaign_config = diagnostic._campaign_config(  # noqa: SLF001
        sim2sim.load_sim2sim_config(base_config_path), "full"
    )
    runs = report["runs"]
    if not isinstance(runs, list) or len(runs) != FULL_RUN_COUNT:
        raise ValueError("full report must contain exactly 12 runs")
    expected_coverage = {
        (scenario, seed) for scenario in FULL_SCENARIOS for seed in FULL_SEEDS
    }
    actual_coverage: set[tuple[str, int]] = set()
    trace_manifest: list[dict[str, Any]] = []
    maximum_recovery = 0.0
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise ValueError(f"full run {index} must be an object")
        scenario = run.get("scenario")
        seed = run.get("seed")
        if not isinstance(scenario, str) or not isinstance(seed, int):
            raise ValueError("full run identity is invalid")
        identity = (scenario, seed)
        if identity not in expected_coverage or identity in actual_coverage:
            raise ValueError("full run coverage is duplicated or unexpected")
        actual_coverage.add(identity)
        if (
            run.get("computed_pass") is not True
            or run.get("episodes") != 22
            or run.get("steps_per_episode") != 250
            or run.get("policy_output_nonfinite_count") != 0
        ):
            raise ValueError("full run gate failed")
        records, trace = _load_trace(report_path, run["trace"])
        if len(records) != 5_500:
            raise ValueError("full run trace must contain 5500 records")
        metrics = sim2sim.recompute_metrics(
            records,
            config=campaign_config,
            disturbance_scale=float(run["disturbance_scale"]),
        )
        bounds = diagnostic._bound_metrics(records, campaign_config)  # noqa: SLF001
        if metrics != run.get("metrics") or bounds != run.get("bounds"):
            raise ValueError("full run metrics differ from raw trace")
        if (
            not sim2sim.metrics_pass(metrics, campaign_config)
            or metrics["termination_count"] != 0
            or metrics["nonfinite_count"] != 0
            or metrics["joint_limit_violation_count"] != 0
            or metrics["recovery_fraction"] != 1.0
            or metrics["max_recovery_time_s"] > MAXIMUM_RECOVERY_TIME_S
            or bounds["joint_velocity_bound_violation_count"] != 0
            or bounds["effort_bound_violation_count"] != 0
            or bounds["fall_count"] != 0
        ):
            raise ValueError("full run strict safety/recovery gate failed")
        maximum_recovery = max(maximum_recovery, metrics["max_recovery_time_s"])
        trace_manifest.append({"scenario": scenario, "seed": seed, **trace})
    if actual_coverage != expected_coverage:
        raise ValueError("full run coverage is incomplete")
    summary = report["summary"]
    if not isinstance(summary, Mapping):
        raise ValueError("full report summary is missing")
    expected_summary_fields = {
        "computed_pass": True,
        "full_campaign_complete": True,
        "run_count": FULL_RUN_COUNT,
        "episode_count": FULL_TOTAL_EPISODES,
        "record_count": FULL_TOTAL_RECORDS,
        "termination_count": 0,
        "fall_count": 0,
        "policy_output_nonfinite_count": 0,
        "joint_limit_violation_count": 0,
        "joint_velocity_bound_violation_count": 0,
        "effort_bound_violation_count": 0,
        "minimum_recovery_fraction": 1.0,
        "max_action_saturation_fraction": 0.0,
        "diagnostic_only": True,
        "deployment_ready": False,
        "promotion_eligible": False,
        "active_motor_control_authorized": False,
    }
    for key, expected in expected_summary_fields.items():
        if summary.get(key) != expected:
            raise ValueError(f"full report summary mismatch: {key}")
    trace_manifest.sort(key=lambda item: (item["seed"], item["scenario"]))
    summary_metrics = {
        "run_count": FULL_RUN_COUNT,
        "episode_count": FULL_TOTAL_EPISODES,
        "record_count": FULL_TOTAL_RECORDS,
        "termination_count": 0,
        "fall_count": 0,
        "policy_output_nonfinite_count": 0,
        "joint_limit_violation_count": 0,
        "joint_velocity_bound_violation_count": 0,
        "effort_bound_violation_count": 0,
        "minimum_recovery_fraction": 1.0,
        "maximum_recovery_time_s": maximum_recovery,
        "max_action_saturation_fraction": summary["max_action_saturation_fraction"],
        "max_joint_velocity_ratio": summary["max_joint_velocity_ratio"],
        "max_effort_ratio": summary["max_effort_ratio"],
        "max_abs_native_action_raw": summary["max_abs_native_action_raw"],
        "minimum_base_height_m": summary["minimum_base_height_m"],
        "maximum_tilt_rad": summary["maximum_tilt_rad"],
    }
    _exact_keys(summary_metrics, _SUMMARY_KEYS, "promotion summary metrics")
    if any(
        not isinstance(summary_metrics[key], (int, float))
        or not math.isfinite(float(summary_metrics[key]))
        for key in _SUMMARY_KEYS
    ):
        raise ValueError("promotion summary contains non-finite values")
    final_public_verification = diagnostic.verify_full_mjlab_diagnostic_report(
        report_path
    )
    report_sha256 = sha256_file(report_path)
    if (
        final_public_verification != public_verification
        or report_sha256 != public_verification["report_sha256"]
    ):
        raise ValueError("full report or traces changed while verifying evidence")
    return {
        "campaign_layout": "monolithic",
        "aggregate_report_filename": report_path.name,
        "aggregate_report_sha256": report_sha256,
        "aggregate_report_payload_sha256": sha256_bytes(
            canonical_json_bytes(report)
        ),
        "trace_manifest_sha256": sha256_bytes(
            canonical_json_bytes(trace_manifest)
        ),
        "shard_manifest_sha256": sha256_bytes(canonical_json_bytes([])),
        "shards": [],
        "deterministic_seeds": list(FULL_SEEDS),
        "scenarios": list(FULL_SCENARIOS),
        "run_count": FULL_RUN_COUNT,
        "episodes_per_scenario": FULL_EPISODES_PER_SCENARIO,
        "total_episodes": FULL_TOTAL_EPISODES,
        "total_records": FULL_TOTAL_RECORDS,
        "all_strict_gates_pass": True,
        "summary_metrics": summary_metrics,
        "provenance": provenance,
    }


def causal_promotion_body(
    *,
    checkpoint_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
    full_report_path: Path,
) -> dict[str, Any]:
    """Re-verify all source bytes/traces and return a promotion body."""

    paths = [
        checkpoint_path,
        encoder_path,
        decoder_path,
        metadata_path,
        full_report_path,
    ]
    resolved = [
        _regular_file(path, f"causal promotion input {index}")
        for index, path in enumerate(paths)
    ]
    if len(set(resolved)) != len(resolved):
        raise ValueError("causal promotion input paths must be distinct")
    initial_hashes = tuple(sha256_file(path) for path in resolved)
    checkpoint, encoder, decoder, metadata_file, report = resolved
    metadata = verify_mjlab_diagnostic_onnx(
        encoder,
        decoder,
        metadata_file,
        checkpoint_path=checkpoint,
        expected_reference_profile=CAUSAL_HISTORY_PROFILE,
    )
    transform = metadata["contract"].get("safe_target_transform")
    transform_hash = _safe_transform_sha256()
    if (
        metadata["schema_version"] != 2
        or metadata["contract"].get("decoder_output_semantics")
        != APPLIED_SAFE_NATIVE_ACTION
        or metadata["contract"].get("previous_action_semantics")
        != APPLIED_SAFE_NATIVE_ACTION
        or metadata["contract"].get("external_safe_target_transform_allowed")
        is not False
        or transform != safe_target_transform_contract()
        or metadata["hashes"].get("safe_target_transform_sha256")
        != transform_hash
    ):
        raise ValueError("causal promotion requires exact schema-2 safe decoder")
    campaign = _validate_full_report(
        report,
        metadata,
        checkpoint_path=checkpoint,
        encoder_path=encoder,
        decoder_path=decoder,
        metadata_path=metadata_file,
    )
    hashes = metadata["hashes"]
    profile_contract = causal_history_profile_contract()
    source = {
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": hashes["checkpoint_sha256"],
        "checkpoint_update_count": metadata["source"]["checkpoint_update_count"],
        "lineage_sha256": hashes["lineage_sha256"],
        "policy_state_sha256": hashes["policy_state_sha256"],
        "reference_profile": CAUSAL_HISTORY_PROFILE,
        "causal_reference_contract_sha256": profile_contract["contract_sha256"],
        "encoder_onnx_filename": encoder.name,
        "encoder_onnx_sha256": hashes["encoder_onnx_sha256"],
        "decoder_onnx_filename": decoder.name,
        "decoder_onnx_sha256": hashes["decoder_onnx_sha256"],
        "metadata_filename": metadata_file.name,
        "metadata_sha256": sha256_file(metadata_file),
        "metadata_payload_sha256": metadata["metadata_payload_sha256"],
    }
    _exact_keys(source, _SOURCE_KEYS, "causal promotion source")
    body = {
        "schema_version": CAUSAL_PROMOTION_SCHEMA_VERSION,
        "kind": CAUSAL_PROMOTION_KIND,
        "robot_model": ROBOT_MODEL,
        "promotion_stage": CAUSAL_PROMOTION_STAGE,
        "deployment_bytes_authorized": True,
        "active_motor_control_authorized": False,
        "gantry_or_rated_support_required": True,
        "free_standing_authorized": False,
        "required_mode_machine": 4,
        "decoder_output_semantics": APPLIED_SAFE_NATIVE_ACTION,
        "previous_action_semantics": APPLIED_SAFE_NATIVE_ACTION,
        "external_safe_target_transform_allowed": False,
        "safe_target_transform_sha256": transform_hash,
        "source_artifact": source,
        "full_campaign_evidence": campaign,
    }
    final_campaign_verification = diagnostic.verify_full_mjlab_diagnostic_report(
        report
    )
    if (
        final_campaign_verification["report_sha256"]
        != campaign["aggregate_report_sha256"]
        or _validate_provenance(report, final_campaign_verification["report"])
        != campaign["provenance"]
    ):
        raise ValueError("full campaign provenance changed after metric verification")
    if tuple(sha256_file(path) for path in resolved) != initial_hashes:
        raise ValueError("causal promotion input bytes changed during verification")
    body["promotion_payload_sha256"] = sha256_bytes(canonical_json_bytes(body))
    _exact_keys(body, _PROMOTION_KEYS, "causal promotion")
    return body


def verify_causal_promotion(
    promotion_path: Path,
    *,
    checkpoint_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
    full_report_path: Path,
) -> Mapping[str, Any]:
    promotion = _regular_file(promotion_path, "causal promotion")
    actual = _read_strict_json(promotion, "causal promotion")
    expected = causal_promotion_body(
        checkpoint_path=checkpoint_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        metadata_path=metadata_path,
        full_report_path=full_report_path,
    )
    if actual != expected:
        raise ValueError("causal promotion differs from re-verified source evidence")
    return actual


def write_new_json(path: Path, value: Mapping[str, Any]) -> Path:
    lexical = _lexical_absolute(path)
    _reject_symlink_components(lexical, "promotion output")
    if lexical.suffix.lower() != ".json" or "promotion" not in lexical.stem.lower():
        raise ValueError("promotion output must be a new promotion .json")
    if lexical.exists():
        raise FileExistsError("refusing to overwrite causal promotion output")
    lexical.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(lexical, "promotion output")
    output = lexical.resolve()
    with output.open("xb") as stream:
        stream.write(canonical_json_bytes(dict(value)))
        stream.flush()
    return output
