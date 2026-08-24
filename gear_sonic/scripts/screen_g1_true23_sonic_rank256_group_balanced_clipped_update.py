"""Fresh-process paired screen for one group-balanced rank256 clipped update."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from gear_sonic.scripts import (
    train_g1_true23_sonic_rank256_disturbance_survival_score as disturbance,
    train_g1_true23_sonic_rank256_group_balanced_clipped_update as update,
    train_g1_true23_sonic_task_space_ppo_full_support as full_support,
)
from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as fs
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_true23_sonic_survival_score_line_search import STATE_PARAMETER_NAMES

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_group_balanced_clipped_screen_v1.json"
)
CONTRACT_SHA256 = "a51c2f4798a6a580533e9583d724be4d5c622f35db38dae115c9ac2d4ea97ddb"
SCALES = (0.0, 0.25, 0.5, 1.0, 2.0)
SCENARIOS = ("nominal", "disturbance")
RESULT_FILENAME = "rank256_group_balanced_clipped_screen_result.json"
CANDIDATE_FILENAME = "rank256_group_balanced_clipped_screen_candidate.pt"
SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/scripts/screen_g1_true23_sonic_rank256_group_balanced_clipped_update.py"),
    Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_group_balanced_clipped_update.py"),
)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("group-balanced clipped screen contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    parents = body.get("parents", {})
    screen = body.get("screen", {})
    boundaries = body.get("boundaries", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_group_balanced_clipped_screen_contract_v1"
        or screen.get("scales") != list(SCALES)
        or screen.get("scenarios") != list(SCENARIOS)
        or screen.get("fresh_process_per_policy_scenario") is not True
        or screen.get("candidate_minimum_nominal_completed_transitions") != 486
        or screen.get("candidate_minimum_disturbance_completed_transitions") != 288
        or set(screen.get("reconstructed_policy_state_sha256_by_scale", {})) != {str(scale) for scale in SCALES}
        or screen.get("float32_delta_subtract_add_not_byte_identical_to_direct_updated_state") is not True
        or boundaries.get("additional_training_transitions") != 0
        or boundaries.get("optimizer_steps") != 0
        or boundaries.get("critic_updates") != 0
        or boundaries.get("hardware_authorized") is not False
        or boundaries.get("robot_or_network_commands_permitted") is not False
    ):
        raise ValueError("group-balanced clipped screen contract semantic mismatch")
    checks = (
        (root / parents["update_contract_relative_path"], parents["update_contract_sha256"]),
        (Path(parents["update_result_path"]), parents["update_result_sha256"]),
        (Path(parents["update_delta_path"]), parents["update_delta_sha256"]),
    )
    for raw, expected in checks:
        resolved = raw.expanduser().resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file() or sha256_file(resolved) != expected:
            raise ValueError("group-balanced clipped screen parent mismatch")
    update._load_contract(root)  # noqa: SLF001
    result = json.loads(Path(parents["update_result_path"]).read_text(encoding="utf-8"))
    optimizer = result.get("optimizer_evidence", {})
    if (
        result.get("kind") != "g1_true23_sonic_rank256_group_balanced_clipped_update_result_v1"
        or result.get("candidate") is not None
        or optimizer.get("initial_policy_state_sha256") != screen.get("baseline_policy_state_sha256")
        or optimizer.get("updated_policy_state_sha256")
        != screen.get("optimizer_reported_updated_policy_state_sha256")
        or optimizer.get("delta_state_sha256") != screen.get("delta_state_sha256")
        or result.get("optimizer_steps") != 4
        or result.get("critic_updates") != 0
    ):
        raise ValueError("group-balanced clipped update result mismatch")
    return body


def _inputs(
    root: Path,
) -> tuple[Mapping[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], Path, Path]:
    contract = _load_contract(root)
    _update_contract, baseline, _overlay, topology, motion = update._inputs(root)  # noqa: SLF001
    payload = torch.load(Path(contract["parents"]["update_delta_path"]), map_location="cpu", weights_only=True)
    delta = payload.get("delta_state_dict")
    if (
        payload.get("kind") != "g1_true23_sonic_rank256_group_balanced_clipped_update_delta_v1"
        or payload.get("contract_sha256") != update.CONTRACT_SHA256
        or payload.get("initial_policy_state_sha256") != contract["screen"]["baseline_policy_state_sha256"]
        or payload.get("updated_policy_state_sha256")
        != contract["screen"]["optimizer_reported_updated_policy_state_sha256"]
        or payload.get("delta_state_sha256") != contract["screen"]["delta_state_sha256"]
        or not isinstance(delta, Mapping)
        or set(delta) != set(STATE_PARAMETER_NAMES)
    ):
        raise ValueError("group-balanced clipped delta artifact mismatch")
    normalized = {
        name: value.detach().cpu().to(torch.float32).contiguous().clone() for name, value in delta.items()
    }
    if fs._state_sha256(normalized) != contract["screen"]["delta_state_sha256"]:  # noqa: SLF001
        raise ValueError("group-balanced clipped delta state hash mismatch")
    return contract, baseline, normalized, topology, motion


def _state(
    baseline: Mapping[str, torch.Tensor], delta: Mapping[str, torch.Tensor], scale: float
) -> dict[str, torch.Tensor]:
    result = {name: value.detach().cpu().contiguous().clone() for name, value in baseline.items()}
    for name in STATE_PARAMETER_NAMES:
        result[name] = torch.add(result[name], delta[name], alpha=scale).contiguous()
    return result


def preflight(repository_root: Path) -> dict[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract, baseline, delta, topology, motion = _inputs(root)
        states = {str(scale): _state(baseline, delta, scale) for scale in SCALES}
        identities = {
            scale: inspect_true23_policy_state(
                {"policy_state_dict": state}, reference_profile=fs.REFERENCE_PROFILE
            )
            for scale, state in states.items()
        }
        if identities != contract["screen"]["reconstructed_policy_state_sha256_by_scale"] or len(
            set(identities.values())
        ) != len(SCALES):
            raise ValueError("group-balanced clipped screen policy identity mismatch")
        sources = {
            relative.as_posix(): sha256_file((root / relative).resolve(strict=True))
            for relative in SOURCE_RELATIVE_PATHS
        }
        material = {
            "contract_sha256": CONTRACT_SHA256,
            "update_result_sha256": contract["parents"]["update_result_sha256"],
            "update_delta_sha256": contract["parents"]["update_delta_sha256"],
            "policy_state_sha256_by_scale": identities,
            "topology_sha256": sha256_file(topology),
            "motion_sha256": sha256_file(motion),
            "sources": sources,
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_group_balanced_clipped_screen_preflight_v1",
            "ready": True,
            "contract": contract,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha(material),
            "simulator_constructed": False,
            "evaluation_runs": 0,
            "optimizer_steps": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_group_balanced_clipped_screen_preflight_v1",
            "ready": False,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "simulator_constructed": False,
            "evaluation_runs": 0,
            "optimizer_steps": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }


def _name(scale: float, scenario: str) -> str:
    return f"scale_{format(scale, 'g').replace('.', 'p')}_{scenario}.json"


def evaluate_one(
    repository_root: Path,
    run_dir: Path,
    scale: float,
    scenario: str,
) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    if scale not in SCALES or scenario not in SCENARIOS:
        raise ValueError("group-balanced clipped evaluation request mismatch")
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("group-balanced clipped screen preflight failed")
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError("group-balanced clipped screen run directory invalid")
    output = run_dir / "evaluations" / _name(scale, scenario)
    if os.path.lexists(output):
        raise FileExistsError("group-balanced clipped evaluation exists")
    _contract, baseline, delta, topology, motion = _inputs(root)
    state = _state(baseline, delta, scale)
    record = disturbance._evaluate(  # noqa: SLF001
        state=state,
        scale=scale,
        scenario=scenario,
        topology=topology,
        motion=motion,
        material_sha=audit["material_manifest_sha256"],
    )
    record["contract_sha256"] = CONTRACT_SHA256
    record["screen_scale"] = scale
    record["fresh_process_evaluation"] = True
    record["screen_material_manifest_sha256"] = audit["material_manifest_sha256"]
    full_support._write_json_exclusive(output, record)  # noqa: SLF001
    return record


def _clean(record: Mapping[str, Any]) -> bool:
    return disturbance._clean(record)  # noqa: SLF001


def _assess(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(float(row["screen_scale"]), str(row["scenario"])): row for row in records}
    if set(by_key) != {(scale, scenario) for scale in SCALES for scenario in SCENARIOS}:
        raise ValueError("group-balanced clipped screen matrix mismatch")
    eligible: list[tuple[float, Mapping[str, Any], Mapping[str, Any]]] = []
    for scale in SCALES:
        if scale == 0.0:
            continue
        nominal = by_key[(scale, "nominal")]
        disturbance_record = by_key[(scale, "disturbance")]
        if (
            _clean(nominal)
            and _clean(disturbance_record)
            and int(nominal["completed_transitions"]) >= 486
            and int(disturbance_record["completed_transitions"]) >= 288
            and nominal["policy_state_sha256"] == disturbance_record["policy_state_sha256"]
        ):
            eligible.append((scale, nominal, disturbance_record))
    selected = max(
        eligible,
        key=lambda item: (
            int(item[2]["completed_transitions"]),
            int(item[1]["completed_transitions"]),
            -abs(item[0]),
        ),
        default=None,
    )
    return {
        "candidate_selected": selected is not None,
        "selected_scale": None if selected is None else selected[0],
        "selected_policy_state_sha256": None if selected is None else selected[1]["policy_state_sha256"],
        "selected_nominal_completed_transitions": (
            None if selected is None else selected[1]["completed_transitions"]
        ),
        "selected_disturbance_completed_transitions": (
            None if selected is None else selected[2]["completed_transitions"]
        ),
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def finalize(repository_root: Path, run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("group-balanced clipped final preflight failed")
    contract, baseline, delta, _topology, _motion = _inputs(root)
    records = []
    for scale in SCALES:
        for scenario in SCENARIOS:
            path = run_dir / "evaluations" / _name(scale, scenario)
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                float(record.get("screen_scale", float("nan"))) != scale
                or record.get("scenario") != scenario
                or record.get("fresh_process_evaluation") is not True
                or record.get("screen_material_manifest_sha256") != audit["material_manifest_sha256"]
            ):
                raise ValueError("group-balanced clipped evaluation mismatch")
            records.append(record)
    assessment = _assess(records)
    candidate = None
    if assessment["candidate_selected"] is True:
        scale = float(assessment["selected_scale"])
        state = _state(baseline, delta, scale)
        path = run_dir / "checkpoints" / CANDIDATE_FILENAME
        disturbance._write_checkpoint(  # noqa: SLF001
            path,
            {
                "schema_version": 1,
                "kind": "g1_true23_sonic_rank256_group_balanced_clipped_screen_candidate_v1",
                "contract_sha256": CONTRACT_SHA256,
                "source_update_contract_sha256": update.CONTRACT_SHA256,
                "policy_state_dict": state,
                "policy_state_sha256": assessment["selected_policy_state_sha256"],
                "selected_scale": scale,
                "additional_training_transitions": 0,
                "optimizer_steps": 0,
                "critic_updates": 0,
                "support_qualified": False,
                "deployment_ready": False,
                "hardware_authorized": False,
            },
        )
        candidate = {"path": str(path), "sha256": sha256_file(path)}
    result = {
        "schema_version": 1,
        "kind": "g1_true23_sonic_rank256_group_balanced_clipped_screen_result_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "parent_update_result_sha256": contract["parents"]["update_result_sha256"],
        "evaluations": records,
        "assessment": assessment,
        "candidate": candidate,
        "additional_training_transitions": 0,
        "evaluation_runs": len(records),
        "optimizer_steps": 0,
        "critic_updates": 0,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    full_support._write_json_exclusive(run_dir / RESULT_FILENAME, result)  # noqa: SLF001
    return result


def run(repository_root: Path, requested_run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("group-balanced clipped screen preflight failed")
    run_dir = full_support._create_run_dir_exclusive(requested_run_dir)  # noqa: SLF001
    full_support._write_json_exclusive(run_dir / "preflight.json", audit)  # noqa: SLF001
    for scale in SCALES:
        for scenario in SCENARIOS:
            command = (
                sys.executable,
                "-m",
                "gear_sonic.scripts.screen_g1_true23_sonic_rank256_group_balanced_clipped_update",
                "evaluate-one",
                "--repository-root",
                str(root),
                "--run-dir",
                str(run_dir),
                "--scale",
                str(scale),
                "--scenario",
                scenario,
            )
            if subprocess.run(command, check=False).returncode != 0:  # noqa: S603
                raise RuntimeError(f"group-balanced clipped child failed: {scale} {scenario}")
    return finalize(root, run_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=ROOT)
    one = sub.add_parser("evaluate-one")
    one.add_argument("--repository-root", type=Path, default=ROOT)
    one.add_argument("--run-dir", type=Path, required=True)
    one.add_argument("--scale", type=float, required=True)
    one.add_argument("--scenario", choices=SCENARIOS, required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--repository-root", type=Path, default=ROOT)
    final.add_argument("--run-dir", type=Path, required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("--repository-root", type=Path, default=ROOT)
    screen.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight(args.repository_root)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    if args.command == "evaluate-one":
        record = evaluate_one(args.repository_root, args.run_dir, args.scale, args.scenario)
        print(  # noqa: T201
            f"scale={args.scale:g} {args.scenario} steps={record['completed_transitions']} "
            f"q9={record['terminal_q9']}",
            flush=True,
        )
        return 0
    if args.command == "finalize":
        result = finalize(args.repository_root, args.run_dir)
    else:
        result = run(args.repository_root, args.run_dir)
    print(json.dumps(result["assessment"], indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if result["candidate"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
