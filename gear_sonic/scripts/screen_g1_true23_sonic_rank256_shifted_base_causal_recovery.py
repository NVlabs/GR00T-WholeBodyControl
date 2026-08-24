"""Fresh-process paired screen for shifted-base post-push causal direction."""

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
    collect_g1_true23_sonic_rank256_shifted_base_causal_recovery_score as collector,
    train_g1_true23_sonic_rank256_disturbance_survival_score as disturbance,
    train_g1_true23_sonic_task_space_ppo_full_support as full_support,
)
from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as fs
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_true23_sonic_survival_score_line_search import STATE_PARAMETER_NAMES

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_shifted_base_causal_recovery_screen_v1.json"
)
CONTRACT_SHA256 = "65a0378cf2aab2257ad48eeb9fd5e1c3e5ed1417bdd64ab8821c9be1f5d13d1f"
SCALES = (-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
SCENARIOS = ("nominal", "disturbance")
RESULT_FILENAME = "rank256_shifted_base_causal_recovery_screen_result.json"
CANDIDATE_FILENAME = "rank256_shifted_base_causal_recovery_screen_candidate.pt"
SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/scripts/screen_g1_true23_sonic_rank256_shifted_base_causal_recovery.py"),
    Path("gear_sonic/scripts/collect_g1_true23_sonic_rank256_shifted_base_causal_recovery_score.py"),
)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("shifted-base causal screen contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    parents = body.get("parents", {})
    screen = body.get("screen", {})
    boundaries = body.get("boundaries", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_shifted_base_causal_recovery_screen_contract_v1"
        or screen.get("scales") != list(SCALES)
        or screen.get("scenarios") != list(SCENARIOS)
        or screen.get("one_fresh_process_per_evaluation") is not True
        or screen.get("baseline_policy_state_sha256")
        != "6acada92dd6e2700499c64255e077c7e844aa0f96b0700e2e2c5640606ae0650"
        or screen.get("candidate_minimum_nominal_completed_transitions") != 486
        or screen.get("candidate_minimum_disturbance_completed_transitions") != 288
        or boundaries.get("additional_training_transitions") != 0
        or boundaries.get("optimizer_steps") != 0
        or boundaries.get("critic_updates") != 0
        or boundaries.get("hardware_authorized") is not False
        or boundaries.get("robot_or_network_commands_permitted") is not False
    ):
        raise ValueError("shifted-base causal screen contract semantic mismatch")
    checks = (
        (root / parents["collection_contract_relative_path"], parents["collection_contract_sha256"]),
        (Path(parents["collection_result_path"]), parents["collection_result_sha256"]),
        (Path(parents["direction_path"]), parents["direction_sha256"]),
    )
    for raw, expected in checks:
        resolved = raw.expanduser().resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file() or sha256_file(resolved) != expected:
            raise ValueError("shifted-base causal screen parent mismatch")
    collector._load_contract(root)  # noqa: SLF001
    result = json.loads(Path(parents["collection_result_path"]).read_text(encoding="utf-8"))
    if (
        result.get("kind") != "g1_true23_sonic_rank256_shifted_base_causal_recovery_score_result_v1"
        or result.get("gradient_evidence", {}).get("direction_state_sha256") != parents["direction_state_sha256"]
        or result.get("candidate") is not None
        or result.get("optimizer_steps") != 0
    ):
        raise ValueError("shifted-base causal collection result mismatch")
    return body


def _direction(contract: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(contract["parents"]["direction_path"]), map_location="cpu", weights_only=True)
    state = payload.get("direction_state_dict")
    if (
        payload.get("kind") != "g1_true23_sonic_rank256_shifted_base_causal_recovery_direction_v1"
        or payload.get("contract_sha256") != collector.CONTRACT_SHA256
        or payload.get("initial_policy_state_sha256") != contract["screen"]["baseline_policy_state_sha256"]
        or payload.get("direction_state_sha256") != contract["parents"]["direction_state_sha256"]
        or not isinstance(state, Mapping)
        or tuple(state) != STATE_PARAMETER_NAMES
    ):
        raise ValueError("shifted-base causal direction artifact mismatch")
    result = {name: value.detach().cpu().to(torch.float32).contiguous().clone() for name, value in state.items()}
    if fs._state_sha256(result) != contract["parents"]["direction_state_sha256"]:  # noqa: SLF001
        raise ValueError("shifted-base causal direction state mismatch")
    return result


def _inputs(
    root: Path,
) -> tuple[Mapping[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], Mapping[str, Any], Path, Path]:
    contract = _load_contract(root)
    _collection_contract, baseline, overlay, topology, motion = collector._inputs(root)  # noqa: SLF001
    direction = _direction(contract)
    return contract, baseline, direction, overlay, topology, motion


def _state(
    baseline: Mapping[str, torch.Tensor], direction: Mapping[str, torch.Tensor], scale: float
) -> dict[str, torch.Tensor]:
    result = {name: value.detach().cpu().contiguous().clone() for name, value in baseline.items()}
    for name in STATE_PARAMETER_NAMES:
        result[name] = torch.add(result[name], direction[name], alpha=scale).contiguous()
    return result


def preflight(repository_root: Path) -> dict[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract, baseline, direction, overlay, topology, motion = _inputs(root)
        identities = {
            str(scale): inspect_true23_policy_state(
                {"policy_state_dict": _state(baseline, direction, scale)},
                reference_profile=fs.REFERENCE_PROFILE,
            )
            for scale in SCALES
        }
        if identities["0.0"] != contract["screen"]["baseline_policy_state_sha256"] or len(
            set(identities.values())
        ) != len(SCALES):
            raise ValueError("shifted-base causal screen state identity mismatch")
        sources = {
            relative.as_posix(): sha256_file((root / relative).resolve(strict=True))
            for relative in SOURCE_RELATIVE_PATHS
        }
        material = {
            "contract_sha256": CONTRACT_SHA256,
            "collection_result_sha256": contract["parents"]["collection_result_sha256"],
            "direction_sha256": contract["parents"]["direction_sha256"],
            "direction_state_sha256": contract["parents"]["direction_state_sha256"],
            "policy_state_sha256_by_scale": identities,
            "source_overlay": overlay,
            "topology_sha256": sha256_file(topology),
            "motion_sha256": sha256_file(motion),
            "sources": sources,
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_shifted_base_causal_recovery_screen_preflight_v1",
            "ready": True,
            "contract": contract,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha(material),
            "simulator_constructed": False,
            "evaluation_runs": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_shifted_base_causal_recovery_screen_preflight_v1",
            "ready": False,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "simulator_constructed": False,
            "evaluation_runs": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }


def _name(scale: float, scenario: str) -> str:
    token = format(scale, "g").replace("-", "m").replace(".", "p")
    return f"scale_{token}_{scenario}.json"


def evaluate_one(repository_root: Path, run_dir: Path, scale: float, scenario: str) -> Mapping[str, Any]:
    if scale not in SCALES or scenario not in SCENARIOS:
        raise ValueError("shifted-base causal evaluation request mismatch")
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("shifted-base causal screen preflight failed")
    _contract, baseline, direction, _overlay, topology, motion = _inputs(root)
    output = run_dir / "evaluations" / _name(scale, scenario)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or os.path.lexists(output):
        raise FileExistsError("shifted-base causal screen output exists")
    record = disturbance._evaluate(  # noqa: SLF001
        state=_state(baseline, direction, scale),
        scale=scale,
        scenario=scenario,
        topology=topology,
        motion=motion,
        material_sha=audit["material_manifest_sha256"],
    )
    record["contract_sha256"] = CONTRACT_SHA256
    record["fresh_process_evaluation"] = True
    record["screen_material_manifest_sha256"] = audit["material_manifest_sha256"]
    full_support._write_json_exclusive(output, record)  # noqa: SLF001
    return record


def _assess(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(float(row["scale"]), str(row["scenario"])): row for row in records}
    if set(by_key) != {(scale, scenario) for scale in SCALES for scenario in SCENARIOS}:
        raise ValueError("shifted-base causal screen matrix mismatch")
    baseline_nominal = by_key[(0.0, "nominal")]
    baseline_push = by_key[(0.0, "disturbance")]
    baseline_ok = (
        disturbance._clean(baseline_nominal)  # noqa: SLF001
        and disturbance._clean(baseline_push)  # noqa: SLF001
        and int(baseline_nominal["completed_transitions"]) >= 486
        and int(baseline_push["completed_transitions"]) == 287
    )
    eligible: list[tuple[float, Mapping[str, Any], Mapping[str, Any]]] = []
    if baseline_ok:
        for scale in SCALES:
            if scale == 0.0:
                continue
            nominal = by_key[(scale, "nominal")]
            push = by_key[(scale, "disturbance")]
            if (
                disturbance._clean(nominal)  # noqa: SLF001
                and disturbance._clean(push)  # noqa: SLF001
                and int(nominal["completed_transitions"]) >= 486
                and int(push["completed_transitions"]) >= 288
                and nominal["policy_state_sha256"] == push["policy_state_sha256"]
            ):
                eligible.append((scale, nominal, push))
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
        "baseline_passed": baseline_ok,
        "baseline_nominal_completed_transitions": baseline_nominal["completed_transitions"],
        "baseline_disturbance_completed_transitions": baseline_push["completed_transitions"],
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
        raise RuntimeError("shifted-base causal final preflight failed")
    _contract, baseline, direction, overlay, _topology, _motion = _inputs(root)
    records = []
    for scale in SCALES:
        for scenario in SCENARIOS:
            record = json.loads((run_dir / "evaluations" / _name(scale, scenario)).read_text(encoding="utf-8"))
            if (
                float(record.get("scale", float("nan"))) != scale
                or record.get("scenario") != scenario
                or record.get("fresh_process_evaluation") is not True
                or record.get("screen_material_manifest_sha256") != audit["material_manifest_sha256"]
            ):
                raise ValueError("shifted-base causal evaluation mismatch")
            records.append(record)
    assessment = _assess(records)
    candidate = None
    if assessment["candidate_selected"] is True:
        scale = float(assessment["selected_scale"])
        state = _state(baseline, direction, scale)
        path = run_dir / "checkpoints" / CANDIDATE_FILENAME
        disturbance._write_checkpoint(  # noqa: SLF001
            path,
            {
                "schema_version": 1,
                "kind": "g1_true23_sonic_rank256_shifted_base_causal_recovery_screen_candidate_v1",
                "contract_sha256": CONTRACT_SHA256,
                "source_overlay": overlay,
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
        "kind": "g1_true23_sonic_rank256_shifted_base_causal_recovery_screen_result_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
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
        raise RuntimeError("shifted-base causal screen preflight failed")
    run_dir = full_support._create_run_dir_exclusive(requested_run_dir)  # noqa: SLF001
    full_support._write_json_exclusive(run_dir / "preflight.json", audit)  # noqa: SLF001
    for scale in SCALES:
        for scenario in SCENARIOS:
            command = (
                sys.executable,
                "-m",
                "gear_sonic.scripts.screen_g1_true23_sonic_rank256_shifted_base_causal_recovery",
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
                raise RuntimeError(f"shifted-base causal child failed: {scale} {scenario}")
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
