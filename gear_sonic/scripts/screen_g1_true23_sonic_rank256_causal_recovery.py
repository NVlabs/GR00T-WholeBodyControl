"""Screen causal recovery direction with one fresh process per evaluation."""

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
    collect_g1_true23_sonic_rank256_causal_recovery_score_v2 as collector,
    screen_g1_true23_sonic_rank256_iterative_grouped as fresh,
    train_g1_true23_sonic_task_space_ppo_full_support as full_support,
)
from gear_sonic.trl.mjlab import sonic_task_space_ppo_runner as task_space
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_causal_recovery_screen_v1.json"
)
CONTRACT_SHA256 = "9523fd9505fa1b19ccb15a957ae5cbb7cad9005a8ea9b26b572f02c9082e1149"
SCALES = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SCENARIOS = ("nominal", "disturbance")
DIRECTION_KEYS = fresh.iterative.DIRECTION_KEYS
RESULT_FILENAME = "rank256_causal_recovery_screen_result.json"
CANDIDATE_FILENAME = "rank256_causal_recovery_screen_candidate.pt"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("causal recovery screen contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    parents = body.get("parents", {})
    screen = body.get("screen", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_causal_recovery_screen_contract_v1"
        or screen.get("scales") != list(SCALES)
        or screen.get("scenarios") != list(SCENARIOS)
        or screen.get("one_fresh_process_per_evaluation") is not True
        or screen.get("original_policy_nominal_minimum_transitions") != 486
        or screen.get("original_policy_disturbance_strictly_above_transitions") != 287
        or body.get("boundaries", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("causal recovery screen contract semantic mismatch")
    for prefix in ("collection_result", "gradient_evidence", "direction"):
        artifact = Path(parents[f"{prefix}_path"]).resolve(strict=True)
        if artifact.is_symlink() or sha256_file(artifact) != parents[f"{prefix}_sha256"]:
            raise ValueError(f"causal recovery screen parent mismatch: {prefix}")
    return body


def _direction(contract: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(contract["parents"]["direction_path"]), map_location="cpu", weights_only=True)
    state = payload.get("direction_state_dict")
    if not isinstance(state, Mapping) or tuple(state) != DIRECTION_KEYS:
        raise ValueError("causal recovery screen direction namespace mismatch")
    result = {name: value.detach().cpu().to(torch.float32).contiguous().clone() for name, value in state.items()}
    if (
        collector.v1.grouped.parent.fs._state_sha256(result)
        != contract["parents"][  # noqa: SLF001
            "direction_state_sha256"
        ]
    ):
        raise ValueError("causal recovery screen direction state mismatch")
    return result


def _inputs(
    root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, torch.Tensor], dict[str, Any], Path, Path]:
    contract = _load_contract(root)
    with collector._scope(root):  # noqa: SLF001
        audit = collector.v1.grouped.parent.preflight(root)
        if audit.get("ready") is not True:
            raise RuntimeError("causal recovery screen base preflight failed")
        baseline, overlay = collector.v1.grouped.parent._rank256_state(root, audit["contract"])  # noqa: SLF001
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
        material = {
            "base_material_manifest_sha256": audit["material_manifest_sha256"],
            "parents": {name: value for name, value in contract["parents"].items() if name.endswith("sha256")},
            "sources": {
                CONTRACT_RELATIVE_PATH.as_posix(): CONTRACT_SHA256,
                "gear_sonic/scripts/screen_g1_true23_sonic_rank256_causal_recovery.py": sha256_file(
                    root / "gear_sonic/scripts/screen_g1_true23_sonic_rank256_causal_recovery.py"
                ),
            },
            "baseline_policy_state_sha256": collector.v1.grouped.parent.fs._state_sha256(baseline),  # noqa: SLF001
            "source_overlay": overlay,
            "topology_sha256": sha256_file(topology),
            "motion_sha256": sha256_file(motion),
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_causal_recovery_screen_preflight_v1",
            "ready": True,
            "contract": contract,
            "material_manifest": material,
            "material_manifest_sha256": _sha(material),
            "simulator_constructed": False,
            "additional_training_transitions": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_causal_recovery_screen_preflight_v1",
            "ready": False,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "simulator_constructed": False,
            "additional_training_transitions": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }


def _name(scale: float, scenario: str) -> str:
    token = f"{scale:g}".replace("-", "m").replace(".", "p")
    return f"scale_{token}_{scenario}.json"


def _state(
    baseline: Mapping[str, torch.Tensor], direction: Mapping[str, torch.Tensor], scale: float
) -> dict[str, torch.Tensor]:
    result = {name: value.detach().cpu().contiguous().clone() for name, value in baseline.items()}
    for name in DIRECTION_KEYS:
        result[name] = torch.add(result[name], direction[name], alpha=scale).contiguous()
    return result


def evaluate_one(repository_root: Path, run_dir: Path, scale: float, scenario: str) -> Mapping[str, Any]:
    if scale not in SCALES or scenario not in SCENARIOS:
        raise ValueError("causal recovery evaluation identity mismatch")
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("causal recovery screen preflight failed")
    contract, _base, baseline, _overlay, topology, motion = _inputs(root)
    output = run_dir / "evaluations" / _name(scale, scenario)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or os.path.lexists(output):
        raise FileExistsError("causal recovery screen output exists")
    record = collector.v1.grouped.parent._evaluate(  # noqa: SLF001
        state=_state(baseline, _direction(contract), scale),
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


def _clean(record: Mapping[str, Any]) -> bool:
    return collector.v1.grouped.parent._clean(record)  # noqa: SLF001


def _assess(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    by_key = {(float(record["scale"]), str(record["scenario"])): record for record in records}
    eligible = []
    for scale in SCALES:
        nominal, disturbance = by_key[(scale, "nominal")], by_key[(scale, "disturbance")]
        if (
            _clean(nominal)
            and _clean(disturbance)
            and int(nominal["completed_transitions"]) >= 486
            and int(disturbance["completed_transitions"]) > 287
        ):
            eligible.append((scale, nominal, disturbance))
    selected = max(
        eligible,
        key=lambda item: (
            min(int(item[1]["completed_transitions"]), int(item[2]["completed_transitions"])),
            int(item[1]["completed_transitions"]) + int(item[2]["completed_transitions"]),
            -abs(float(item[0])),
        ),
        default=None,
    )
    return {
        "baseline_nominal_completed_transitions": by_key[(0.0, "nominal")]["completed_transitions"],
        "baseline_disturbance_completed_transitions": by_key[(0.0, "disturbance")]["completed_transitions"],
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


def _finalize(repository_root: Path, run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("causal recovery final preflight failed")
    contract, _base, baseline, overlay, _topology, _motion = _inputs(root)
    direction = _direction(contract)
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
                raise ValueError("causal recovery evaluation mismatch")
            records.append(record)
    assessment = _assess(records)
    candidate = None
    if assessment["candidate_selected"] is True:
        scale = float(assessment["selected_scale"])
        state = _state(baseline, direction, scale)
        path = run_dir / "checkpoints" / CANDIDATE_FILENAME
        collector.v1.grouped.parent._write_checkpoint(  # noqa: SLF001
            path,
            {
                "schema_version": 1,
                "kind": "g1_true23_sonic_rank256_causal_recovery_screen_candidate_v1",
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
        "kind": "g1_true23_sonic_rank256_causal_recovery_screen_result_v1",
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


def run(repository_root: Path, run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("causal recovery screen preflight failed")
    if os.path.lexists(run_dir):
        raise FileExistsError("causal recovery screen run exists")
    run_dir.mkdir(parents=True)
    full_support._write_json_exclusive(run_dir / "preflight.json", audit)  # noqa: SLF001
    for scale in SCALES:
        for scenario in SCENARIOS:
            command = (
                sys.executable,
                "-m",
                "gear_sonic.scripts.screen_g1_true23_sonic_rank256_causal_recovery",
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
                raise RuntimeError(f"causal recovery child failed: {scale} {scenario}")
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
            f"scale={args.scale:g} {args.scenario} steps={record['completed_transitions']} "
            f"q9={record['terminal_q9']}",
            flush=True,
        )
        return 0
    result = run(args.repository_root, args.run_dir)
    print(json.dumps(result["assessment"], indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if result["candidate"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
