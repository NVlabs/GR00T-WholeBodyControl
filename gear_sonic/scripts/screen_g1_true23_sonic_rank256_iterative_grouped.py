"""Finish shifted-base grouped-gradient screen using one fresh process per evaluation."""

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
    train_g1_true23_sonic_rank256_iterative_grouped_survival_score as iterative,
    train_g1_true23_sonic_task_space_ppo_full_support as full_support,
)
from gear_sonic.trl.mjlab import sonic_task_space_ppo_runner as task_space
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_iterative_grouped_screen_v3.json"
)
CONTRACT_SHA256 = "767427f0a8b331202cbc0ddb04dbba5060c3338dcb2e2d5946616a5bd138612e"
RESULT_FILENAME = "rank256_iterative_grouped_screen_result.json"
CANDIDATE_FILENAME = "rank256_iterative_grouped_screen_candidate.pt"
SCALES = iterative.SCALES
SCENARIOS = ("nominal", "disturbance")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("iterative grouped screen contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    parents = body.get("parents", {})
    screen = body.get("screen", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_iterative_grouped_screen_contract_v3"
        or screen.get("scales") != list(SCALES)
        or screen.get("scenarios") != list(SCENARIOS)
        or screen.get("one_fresh_process_per_evaluation") is not True
        or body.get("boundaries", {}).get("reuses_completed_gradient_without_recomputation") is not True
        or body.get("boundaries", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("iterative grouped screen contract semantic mismatch")
    partial = Path(parents["partial_run_path"]).resolve(strict=True)
    checks = (
        (root / parents["iteration_contract_relative_path"], parents["iteration_contract_sha256"]),
        (partial / "preflight.json", parents["partial_preflight_sha256"]),
        (partial / "material_manifest.json", parents["partial_material_manifest_sha256"]),
        (partial / "gradient_evidence.json", parents["gradient_evidence_sha256"]),
        (
            partial / "checkpoints/rank256_disturbance_survival_score_direction.pt",
            parents["gradient_direction_sha256"],
        ),
    )
    for raw, expected in checks:
        resolved = raw.resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file() or sha256_file(resolved) != expected:
            raise ValueError("iterative grouped screen parent mismatch")
    return body


def _direction(contract: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    path = (
        Path(contract["parents"]["partial_run_path"])
        / "checkpoints/rank256_disturbance_survival_score_direction.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("direction_state_dict")
    if not isinstance(state, Mapping) or tuple(state) != iterative.DIRECTION_KEYS:
        raise ValueError("iterative grouped screen direction namespace mismatch")
    result = {name: value.detach().cpu().to(torch.float32).contiguous().clone() for name, value in state.items()}
    if iterative.fs._state_sha256(result) != contract["parents"]["gradient_direction_state_sha256"]:  # noqa: SLF001
        raise ValueError("iterative grouped screen direction state mismatch")
    return result


def _inputs(
    root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, torch.Tensor], dict[str, Any], Path, Path]:
    contract = _load_contract(root)
    with iterative._scope(root):  # noqa: SLF001
        audit = iterative.grouped.parent.preflight(root)
        if audit.get("ready") is not True:
            raise RuntimeError("iterative grouped screen base preflight failed")
        baseline, overlay = iterative.grouped.parent._rank256_state(root, audit["contract"])  # noqa: SLF001
    _direction(contract)
    base_contract = task_space.load_task_space_ppo_contract(root)
    topology = (root / base_contract["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(
        strict=True
    )
    motion = (root / base_contract["environment"]["motion_relative_path"]).resolve(strict=True)
    return contract, audit, baseline, overlay, topology, motion


def preflight(repository_root: Path) -> Mapping[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract, audit, baseline, overlay, topology, motion = _inputs(root)
        sources = {
            CONTRACT_RELATIVE_PATH.as_posix(): CONTRACT_SHA256,
            "gear_sonic/scripts/screen_g1_true23_sonic_rank256_iterative_grouped.py": sha256_file(
                root / "gear_sonic/scripts/screen_g1_true23_sonic_rank256_iterative_grouped.py"
            ),
        }
        material = {
            "base_material_manifest_sha256": audit["material_manifest_sha256"],
            "parent_artifacts": {
                name: value for name, value in contract["parents"].items() if name.endswith("sha256")
            },
            "sources": sources,
            "topology_sha256": sha256_file(topology),
            "motion_sha256": sha256_file(motion),
            "shifted_policy_state_sha256": iterative.fs._state_sha256(baseline),  # noqa: SLF001
            "source_overlay": overlay,
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_iterative_grouped_screen_preflight_v3",
            "ready": True,
            "contract": contract,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha(material),
            "simulator_constructed": False,
            "gradient_recomputed": False,
            "additional_training_transitions": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_iterative_grouped_screen_preflight_v3",
            "ready": False,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "simulator_constructed": False,
            "gradient_recomputed": False,
            "additional_training_transitions": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }


def _state(
    baseline: Mapping[str, torch.Tensor], direction: Mapping[str, torch.Tensor], scale: float
) -> dict[str, torch.Tensor]:
    result = {name: value.detach().cpu().contiguous().clone() for name, value in baseline.items()}
    for name in iterative.DIRECTION_KEYS:
        result[name] = torch.add(result[name], direction[name], alpha=scale).contiguous()
    return result


def _name(scale: float, scenario: str) -> str:
    scale_token = f"{scale:g}".replace("-", "m").replace(".", "p")
    return f"scale_{scale_token}_{scenario}.json"


def evaluate_one(repository_root: Path, run_dir: Path, scale: float, scenario: str) -> Mapping[str, Any]:
    if scale not in SCALES or scenario not in SCENARIOS:
        raise ValueError("iterative grouped screen evaluation identity mismatch")
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("iterative grouped screen preflight failed")
    _contract, _base_audit, baseline, _overlay, topology, motion = _inputs(root)
    state = _state(baseline, _direction(audit["contract"]), scale)
    output = run_dir / "evaluations" / _name(scale, scenario)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or os.path.lexists(output):
        raise FileExistsError("iterative grouped screen evaluation output exists")
    record = iterative.grouped.parent._evaluate(  # noqa: SLF001
        state=state,
        scale=scale,
        scenario=scenario,
        topology=topology,
        motion=motion,
        material_sha=audit["material_manifest_sha256"],
    )
    record["screen_material_manifest_sha256"] = audit["material_manifest_sha256"]
    record["fresh_process_evaluation"] = True
    full_support._write_json_exclusive(output, record)  # noqa: SLF001
    return record


def _finalize(repository_root: Path, run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("iterative grouped screen final preflight failed")
    contract, _base_audit, baseline, overlay, _topology, _motion = _inputs(root)
    direction = _direction(contract)
    records: list[Mapping[str, Any]] = []
    for scale in SCALES:
        for scenario in SCENARIOS:
            path = (run_dir / "evaluations" / _name(scale, scenario)).resolve(strict=True)
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                float(record.get("scale", float("nan"))) != scale
                or record.get("scenario") != scenario
                or record.get("fresh_process_evaluation") is not True
                or record.get("screen_material_manifest_sha256") != audit["material_manifest_sha256"]
            ):
                raise ValueError("iterative grouped screen evaluation mismatch")
            records.append(record)
    assessment = iterative._assess(records)  # noqa: SLF001
    candidate = None
    if assessment["candidate_selected"] is True:
        scale = float(assessment["selected_scale"])
        state = _state(baseline, direction, scale)
        path = run_dir / "checkpoints" / CANDIDATE_FILENAME
        iterative.grouped.parent._write_checkpoint(  # noqa: SLF001
            path,
            {
                "schema_version": 1,
                "kind": "g1_true23_sonic_rank256_iterative_grouped_screen_candidate_v3",
                "contract_sha256": CONTRACT_SHA256,
                "source_overlay": overlay,
                "policy_state_dict": state,
                "policy_state_sha256": assessment["selected_policy_state_sha256"],
                "selected_scale": scale,
                "gradient_recomputed": False,
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
        "kind": "g1_true23_sonic_rank256_iterative_grouped_screen_result_v3",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "evaluations": records,
        "assessment": assessment,
        "candidate": candidate,
        "gradient_recomputed": False,
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


def run(repository_root: Path, run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("iterative grouped screen preflight failed")
    if os.path.lexists(run_dir):
        raise FileExistsError("iterative grouped screen run directory exists")
    run_dir.mkdir(parents=True)
    full_support._write_json_exclusive(run_dir / "preflight.json", audit)  # noqa: SLF001
    for scale in SCALES:
        for scenario in SCENARIOS:
            command = (
                sys.executable,
                "-m",
                "gear_sonic.scripts.screen_g1_true23_sonic_rank256_iterative_grouped",
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
            completed = subprocess.run(command, check=False)  # noqa: S603
            if completed.returncode != 0:
                raise RuntimeError(f"iterative grouped child evaluation failed: {scale} {scenario}")
    return _finalize(root, run_dir)


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
            f"scale={args.scale:g} scenario={args.scenario} "
            f"steps={record['completed_transitions']} q9={record['terminal_q9']}",
            flush=True,
        )
        return 0
    result = run(args.repository_root, args.run_dir)
    print(json.dumps(result["assessment"], indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if result["candidate"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
