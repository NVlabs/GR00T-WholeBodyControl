"""Screen constrained reward/grouped/causal direction mixes in fresh processes."""

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
    screen_g1_true23_sonic_rank256_causal_recovery as causal,
    train_g1_true23_sonic_task_space_ppo_full_support as full_support,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_constrained_recovery_mix_v1.json"
)
CONTRACT_SHA256 = "9496fc59a288fd6b74ad09e33ad3a7cb096e3795a7b806cee6c7ac27055902d7"
REWARD_COEFFICIENT = 1.0
GROUPED_COEFFICIENTS = (0.5, 1.0, 2.0)
CAUSAL_COEFFICIENTS = (0.0, 4.0, 8.0)
SCENARIOS = ("nominal", "disturbance")
DIRECTION_KEYS = causal.DIRECTION_KEYS
RESULT_FILENAME = "rank256_constrained_recovery_mix_result.json"
CANDIDATE_FILENAME = "rank256_constrained_recovery_mix_candidate.pt"


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("constrained recovery mix contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    screen = body.get("screen", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_constrained_recovery_mix_contract_v1"
        or screen.get("reward_coefficient") != REWARD_COEFFICIENT
        or screen.get("grouped_coefficients") != list(GROUPED_COEFFICIENTS)
        or screen.get("causal_coefficients") != list(CAUSAL_COEFFICIENTS)
        or screen.get("scenario_order") != list(SCENARIOS)
        or screen.get("one_fresh_process_per_evaluation") is not True
        or screen.get("original_policy_nominal_minimum_transitions") != 486
        or screen.get("original_policy_disturbance_strictly_above_transitions") != 287
        or body.get("boundaries", {}).get("hardware_authorized") is not False
        or body.get("boundaries", {}).get("robot_or_network_commands_permitted") is not False
    ):
        raise ValueError("constrained recovery mix contract semantic mismatch")
    parents = body["parents"]
    for prefix in (
        "reward_direction",
        "grouped_direction",
        "causal_direction",
        "iterative_screen",
        "causal_screen",
    ):
        artifact = Path(parents[f"{prefix}_path"]).resolve(strict=True)
        if artifact.is_symlink() or not artifact.is_file() or sha256_file(artifact) != parents[f"{prefix}_sha256"]:
            raise ValueError(f"constrained recovery mix parent mismatch: {prefix}")
    return body


def _direction(contract: Mapping[str, Any], prefix: str) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(contract["parents"][f"{prefix}_path"]), map_location="cpu", weights_only=True)
    state = payload.get("direction_state_dict")
    if not isinstance(state, Mapping) or tuple(state) != DIRECTION_KEYS:
        raise ValueError(f"constrained recovery direction namespace mismatch: {prefix}")
    result = {name: value.detach().cpu().to(torch.float32).contiguous().clone() for name, value in state.items()}
    expected = contract["parents"][f"{prefix}_state_sha256"]
    if causal.collector.v1.grouped.parent.fs._state_sha256(result) != expected:  # noqa: SLF001
        raise ValueError(f"constrained recovery direction state mismatch: {prefix}")
    return result


def _inputs(
    root: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    dict[str, torch.Tensor],
    dict[str, Any],
    dict[str, dict[str, torch.Tensor]],
    Path,
    Path,
]:
    contract = _load_contract(root)
    _causal_contract, audit, baseline, overlay, topology, motion = causal._inputs(root)  # noqa: SLF001
    directions = {
        "reward_direction": _direction(contract, "reward_direction"),
        "grouped_direction": _direction(contract, "grouped_direction"),
        "causal_direction": _direction(contract, "causal_direction"),
    }
    return contract, audit, baseline, overlay, directions, topology, motion


def preflight(repository_root: Path) -> Mapping[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract, audit, baseline, overlay, directions, topology, motion = _inputs(root)
        cosine = {}
        names = tuple(directions)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                dot = sum(
                    float(torch.sum(directions[left][name] * directions[right][name])) for name in DIRECTION_KEYS
                )
                left_norm = sum(float(torch.sum(value * value)) for value in directions[left].values()) ** 0.5
                right_norm = sum(float(torch.sum(value * value)) for value in directions[right].values()) ** 0.5
                cosine[f"{left}_vs_{right}"] = dot / (left_norm * right_norm)
        material = {
            "base_material_manifest_sha256": audit["material_manifest_sha256"],
            "parents": {name: value for name, value in contract["parents"].items() if name.endswith("sha256")},
            "sources": {
                CONTRACT_RELATIVE_PATH.as_posix(): CONTRACT_SHA256,
                "gear_sonic/scripts/screen_g1_true23_sonic_rank256_constrained_recovery_mix.py": sha256_file(
                    root / "gear_sonic/scripts/screen_g1_true23_sonic_rank256_constrained_recovery_mix.py"
                ),
            },
            "baseline_policy_state_sha256": causal.collector.v1.grouped.parent.fs._state_sha256(  # noqa: SLF001
                baseline
            ),
            "source_overlay": overlay,
            "direction_cosines": cosine,
            "topology_sha256": sha256_file(topology),
            "motion_sha256": sha256_file(motion),
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_constrained_recovery_mix_preflight_v1",
            "ready": True,
            "contract": contract,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha(material),
            "simulator_constructed": False,
            "additional_training_transitions": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_constrained_recovery_mix_preflight_v1",
            "ready": False,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "simulator_constructed": False,
            "additional_training_transitions": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }


def _state(
    baseline: Mapping[str, torch.Tensor],
    directions: Mapping[str, Mapping[str, torch.Tensor]],
    grouped_coefficient: float,
    causal_coefficient: float,
) -> dict[str, torch.Tensor]:
    result = {name: value.detach().cpu().contiguous().clone() for name, value in baseline.items()}
    for name in DIRECTION_KEYS:
        result[name] = torch.add(result[name], directions["reward_direction"][name], alpha=1.0)
        result[name] = torch.add(result[name], directions["grouped_direction"][name], alpha=grouped_coefficient)
        result[name] = torch.add(
            result[name], directions["causal_direction"][name], alpha=causal_coefficient
        ).contiguous()
    return result


def _name(grouped_coefficient: float, causal_coefficient: float, scenario: str) -> str:
    grouped_token = f"{grouped_coefficient:g}".replace(".", "p")
    causal_token = f"{causal_coefficient:g}".replace(".", "p")
    return f"grouped_{grouped_token}_causal_{causal_token}_{scenario}.json"


def evaluate_one(
    repository_root: Path,
    run_dir: Path,
    grouped_coefficient: float,
    causal_coefficient: float,
    scenario: str,
) -> Mapping[str, Any]:
    if (
        grouped_coefficient not in GROUPED_COEFFICIENTS
        or causal_coefficient not in CAUSAL_COEFFICIENTS
        or scenario not in SCENARIOS
    ):
        raise ValueError("constrained recovery evaluation identity mismatch")
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("constrained recovery mix preflight failed")
    _contract, _base, baseline, _overlay, directions, topology, motion = _inputs(root)
    output = run_dir / "evaluations" / _name(grouped_coefficient, causal_coefficient, scenario)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or os.path.lexists(output):
        raise FileExistsError("constrained recovery evaluation exists")
    record = causal.collector.v1.grouped.parent._evaluate(  # noqa: SLF001
        state=_state(baseline, directions, grouped_coefficient, causal_coefficient),
        scale=0.0,
        scenario=scenario,
        topology=topology,
        motion=motion,
        material_sha=audit["material_manifest_sha256"],
    )
    record["reward_coefficient"] = REWARD_COEFFICIENT
    record["grouped_coefficient"] = grouped_coefficient
    record["causal_coefficient"] = causal_coefficient
    record["screen_material_manifest_sha256"] = audit["material_manifest_sha256"]
    record["fresh_process_evaluation"] = True
    full_support._write_json_exclusive(output, record)  # noqa: SLF001
    return record


def _assess(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = []
    for grouped_coefficient in GROUPED_COEFFICIENTS:
        for causal_coefficient in CAUSAL_COEFFICIENTS:
            selected = [
                record
                for record in records
                if float(record["grouped_coefficient"]) == grouped_coefficient
                and float(record["causal_coefficient"]) == causal_coefficient
            ]
            by_scenario = {str(record["scenario"]): record for record in selected}
            nominal, disturbance = by_scenario["nominal"], by_scenario["disturbance"]
            if (
                causal._clean(nominal)  # noqa: SLF001
                and causal._clean(disturbance)  # noqa: SLF001
                and int(nominal["completed_transitions"]) >= 486
                and int(disturbance["completed_transitions"]) > 287
            ):
                eligible.append((grouped_coefficient, causal_coefficient, nominal, disturbance))
    selected = max(
        eligible,
        key=lambda item: (
            min(int(item[2]["completed_transitions"]), int(item[3]["completed_transitions"])),
            int(item[2]["completed_transitions"]) + int(item[3]["completed_transitions"]),
            -(item[0] ** 2 + item[1] ** 2),
        ),
        default=None,
    )
    return {
        "candidate_selected": selected is not None,
        "selected_reward_coefficient": None if selected is None else REWARD_COEFFICIENT,
        "selected_grouped_coefficient": None if selected is None else selected[0],
        "selected_causal_coefficient": None if selected is None else selected[1],
        "selected_policy_state_sha256": None if selected is None else selected[2]["policy_state_sha256"],
        "selected_nominal_completed_transitions": (
            None if selected is None else selected[2]["completed_transitions"]
        ),
        "selected_disturbance_completed_transitions": (
            None if selected is None else selected[3]["completed_transitions"]
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
        raise RuntimeError("constrained recovery final preflight failed")
    _contract, _base, baseline, overlay, directions, _topology, _motion = _inputs(root)
    records = []
    for grouped_coefficient in GROUPED_COEFFICIENTS:
        for causal_coefficient in CAUSAL_COEFFICIENTS:
            for scenario in SCENARIOS:
                path = run_dir / "evaluations" / _name(grouped_coefficient, causal_coefficient, scenario)
                record = json.loads(path.read_text(encoding="utf-8"))
                if (
                    float(record.get("grouped_coefficient", float("nan"))) != grouped_coefficient
                    or float(record.get("causal_coefficient", float("nan"))) != causal_coefficient
                    or record.get("scenario") != scenario
                    or record.get("fresh_process_evaluation") is not True
                    or record.get("screen_material_manifest_sha256") != audit["material_manifest_sha256"]
                ):
                    raise ValueError("constrained recovery evaluation mismatch")
                records.append(record)
    assessment = _assess(records)
    candidate = None
    if assessment["candidate_selected"] is True:
        grouped_coefficient = float(assessment["selected_grouped_coefficient"])
        causal_coefficient = float(assessment["selected_causal_coefficient"])
        state = _state(baseline, directions, grouped_coefficient, causal_coefficient)
        path = run_dir / "checkpoints" / CANDIDATE_FILENAME
        causal.collector.v1.grouped.parent._write_checkpoint(  # noqa: SLF001
            path,
            {
                "schema_version": 1,
                "kind": "g1_true23_sonic_rank256_constrained_recovery_mix_candidate_v1",
                "contract_sha256": CONTRACT_SHA256,
                "source_overlay": overlay,
                "policy_state_dict": state,
                "policy_state_sha256": assessment["selected_policy_state_sha256"],
                "selected_reward_coefficient": REWARD_COEFFICIENT,
                "selected_grouped_coefficient": grouped_coefficient,
                "selected_causal_coefficient": causal_coefficient,
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
        "kind": "g1_true23_sonic_rank256_constrained_recovery_mix_result_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "direction_cosines": audit["material_manifest"]["direction_cosines"],
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
        raise RuntimeError("constrained recovery mix preflight failed")
    if os.path.lexists(run_dir):
        raise FileExistsError("constrained recovery mix run exists")
    run_dir.mkdir(parents=True)
    full_support._write_json_exclusive(run_dir / "preflight.json", audit)  # noqa: SLF001
    for grouped_coefficient in GROUPED_COEFFICIENTS:
        for causal_coefficient in CAUSAL_COEFFICIENTS:
            for scenario in SCENARIOS:
                command = (
                    sys.executable,
                    "-m",
                    "gear_sonic.scripts.screen_g1_true23_sonic_rank256_constrained_recovery_mix",
                    "evaluate-one",
                    "--repository-root",
                    str(root),
                    "--run-dir",
                    str(run_dir),
                    "--grouped-coefficient",
                    str(grouped_coefficient),
                    "--causal-coefficient",
                    str(causal_coefficient),
                    "--scenario",
                    scenario,
                )
                if subprocess.run(command, check=False).returncode != 0:  # noqa: S603
                    raise RuntimeError(
                        f"constrained recovery child failed: {grouped_coefficient} {causal_coefficient} {scenario}"
                    )
    return _finalize(root, run_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=ROOT)
    one = sub.add_parser("evaluate-one")
    one.add_argument("--repository-root", type=Path, default=ROOT)
    one.add_argument("--run-dir", type=Path, required=True)
    one.add_argument("--grouped-coefficient", type=float, required=True)
    one.add_argument("--causal-coefficient", type=float, required=True)
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
        record = evaluate_one(
            args.repository_root,
            args.run_dir,
            args.grouped_coefficient,
            args.causal_coefficient,
            args.scenario,
        )
        print(  # noqa: T201
            f"grouped={args.grouped_coefficient:g} causal={args.causal_coefficient:g} "
            f"{args.scenario} steps={record['completed_transitions']} q9={record['terminal_q9']}",
            flush=True,
        )
        return 0
    result = run(args.repository_root, args.run_dir)
    print(json.dumps(result["assessment"], indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if result["candidate"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
