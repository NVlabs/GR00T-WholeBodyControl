"""Screen bounded mixtures of fresh survival and post-push reward directions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from gear_sonic.scripts import (
    train_g1_true23_sonic_rank256_disturbance_survival_score as engine,
    train_g1_true23_sonic_rank256_exact_impulse_survival_score as exact,
    train_g1_true23_sonic_task_space_ppo_full_support as full_support,
)
from gear_sonic.trl.mjlab import sonic_task_space_ppo_runner as task_space
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_two_direction_screen_v1.json"
)
CONTRACT_SHA256 = "f564798c6d5a0ef8d83dbb244710bddf5d6589d58d1477ceb02c645deb0e775e"
RESULT_FILENAME = "rank256_two_direction_screen_result.json"
CANDIDATE_FILENAME = "rank256_two_direction_screen_candidate.pt"
COEFFICIENTS = (
    (0.0, 0.0),
    (1.0, 0.0),
    (2.0, 0.0),
    (4.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (2.0, 1.0),
    (4.0, 1.0),
    (2.0, 2.0),
    (4.0, 2.0),
)
DIRECTION_KEYS = (
    "actor_module.decoders.g1_dyn.module.14.bias",
    "actor_module.decoders.g1_dyn.module.14.weight",
    "actor_module.decoders.g1_dyn.module.16.bias",
    "actor_module.decoders.g1_dyn.module.16.weight",
)
EXTRA_SOURCE_RELATIVE_PATHS: tuple[Path, ...] = ()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("two-direction screen contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    screen = body.get("screen", {})
    boundaries = body.get("boundaries", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_two_direction_screen_contract_v1"
        or screen.get("coefficients") != [list(value) for value in COEFFICIENTS]
        or screen.get("scenario_order") != ["nominal", "disturbance"]
        or screen.get("candidate_requires_nominal_not_below_baseline") is not True
        or screen.get("candidate_requires_disturbance_above_baseline") is not True
        or boundaries.get("training_transitions") != 0
        or boundaries.get("robot_or_network_commands_permitted") is not False
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("two-direction screen contract semantic mismatch")
    for name, entry in body["inputs"].items():
        artifact = Path(entry["path"]).resolve(strict=True)
        if artifact.is_symlink() or sha256_file(artifact) != entry["sha256"]:
            raise ValueError(f"two-direction screen input mismatch: {name}")
    return body


def _direction(entry: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(entry["path"]), map_location="cpu", weights_only=True)
    state = payload.get("direction_state_dict")
    if not isinstance(state, Mapping) or tuple(state) != DIRECTION_KEYS:
        raise ValueError("two-direction state namespace/order mismatch")
    result = {name: value.detach().cpu().to(torch.float32).contiguous().clone() for name, value in state.items()}
    from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as fs

    if fs._state_sha256(result) != entry["state_sha256"]:  # noqa: SLF001
        raise ValueError("two-direction state hash mismatch")
    return result


def _preflight(root: Path) -> Mapping[str, Any]:
    try:
        contract = _load_contract(root)
        with exact._scope(root):  # noqa: SLF001
            base = engine.preflight(root)
        if base.get("ready") is not True:
            raise RuntimeError("two-direction base preflight failed")
        survival = _direction(contract["inputs"]["survival_direction"])
        reward = _direction(contract["inputs"]["reward_direction"])
        dot = sum(float(torch.sum(survival[name] * reward[name])) for name in DIRECTION_KEYS)
        survival_l2 = math.sqrt(sum(float(torch.sum(value * value)) for value in survival.values()))
        reward_l2 = math.sqrt(sum(float(torch.sum(value * value)) for value in reward.values()))
        sources = {
            CONTRACT_RELATIVE_PATH.as_posix(): CONTRACT_SHA256,
            "gear_sonic/scripts/screen_g1_true23_sonic_rank256_two_directions.py": sha256_file(
                root / "gear_sonic/scripts/screen_g1_true23_sonic_rank256_two_directions.py"
            ),
        }
        sources.update(
            {
                relative.as_posix(): sha256_file((root / relative).resolve(strict=True))
                for relative in EXTRA_SOURCE_RELATIVE_PATHS
            }
        )
        material = {
            "base_material_manifest_sha256": base["material_manifest_sha256"],
            "inputs": {name: entry["sha256"] for name, entry in contract["inputs"].items()},
            "sources": sources,
            "direction_cosine": dot / (survival_l2 * reward_l2),
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_two_direction_screen_preflight_v1",
            "ready": True,
            "contract": contract,
            "base": base,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha(material),
            "simulator_constructed": False,
            "training_transitions": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_two_direction_screen_preflight_v1",
            "ready": False,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "simulator_constructed": False,
            "training_transitions": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }


def _unchanged(root: Path, expected: str, phase: str) -> None:
    current = _preflight(root)
    if current.get("ready") is not True or current.get("material_manifest_sha256") != expected:
        raise RuntimeError(f"two-direction screen materials changed: {phase}")


def _state(
    baseline: Mapping[str, torch.Tensor],
    survival: Mapping[str, torch.Tensor],
    reward: Mapping[str, torch.Tensor],
    coefficients: tuple[float, float],
) -> dict[str, torch.Tensor]:
    result = {name: value.detach().cpu().contiguous().clone() for name, value in baseline.items()}
    for name in DIRECTION_KEYS:
        result[name] = torch.add(
            torch.add(result[name], survival[name], alpha=coefficients[0]),
            reward[name],
            alpha=coefficients[1],
        ).contiguous()
    return result


def _assess(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    by_key = {
        ((float(record["survival_coefficient"]), float(record["reward_coefficient"])), record["scenario"]): record
        for record in records
    }
    base_nominal = by_key[((0.0, 0.0), "nominal")]
    base_disturbance = by_key[((0.0, 0.0), "disturbance")]
    eligible = []
    for coefficients in COEFFICIENTS[1:]:
        nominal = by_key[(coefficients, "nominal")]
        disturbance = by_key[(coefficients, "disturbance")]
        if (
            engine._clean(nominal)  # noqa: SLF001
            and engine._clean(disturbance)  # noqa: SLF001
            and int(nominal["completed_transitions"]) >= int(base_nominal["completed_transitions"])
            and int(disturbance["completed_transitions"]) > int(base_disturbance["completed_transitions"])
        ):
            eligible.append((coefficients, nominal, disturbance))
    selected = max(
        eligible,
        key=lambda item: (
            min(int(item[1]["completed_transitions"]), int(item[2]["completed_transitions"])),
            int(item[1]["completed_transitions"]) + int(item[2]["completed_transitions"]),
            -(item[0][0] ** 2 + item[0][1] ** 2),
        ),
        default=None,
    )
    return {
        "baseline_nominal_completed_transitions": base_nominal["completed_transitions"],
        "baseline_disturbance_completed_transitions": base_disturbance["completed_transitions"],
        "candidate_selected": selected is not None,
        "selected_coefficients": None if selected is None else list(selected[0]),
        "selected_policy_state_sha256": None if selected is None else selected[1]["policy_state_sha256"],
        "selected_nominal_completed_transitions": None
        if selected is None
        else selected[1]["completed_transitions"],
        "selected_disturbance_completed_transitions": (
            None if selected is None else selected[2]["completed_transitions"]
        ),
        "support_qualified": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def run(repository_root: Path, requested_run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = _preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("two-direction screen preflight failed")
    if os.path.lexists(requested_run_dir):
        raise FileExistsError("two-direction screen run directory exists")
    run_dir = full_support._create_run_dir_exclusive(requested_run_dir)  # noqa: SLF001
    full_support._write_json_exclusive(run_dir / "preflight.json", audit)  # noqa: SLF001
    contract = audit["contract"]
    survival = _direction(contract["inputs"]["survival_direction"])
    reward = _direction(contract["inputs"]["reward_direction"])
    baseline, overlay = engine._rank256_state(root, audit["base"]["contract"])  # noqa: SLF001
    base_contract = task_space.load_task_space_ppo_contract(root)
    topology = (root / base_contract["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(
        strict=True
    )
    motion = (root / base_contract["environment"]["motion_relative_path"]).resolve(strict=True)
    states = {coefficients: _state(baseline, survival, reward, coefficients) for coefficients in COEFFICIENTS}
    records = []
    evaluations = run_dir / "evaluations"
    if not evaluations.is_dir() or evaluations.is_symlink() or any(evaluations.iterdir()):
        raise RuntimeError("two-direction screen evaluations directory ABI mismatch")
    for coefficients in COEFFICIENTS:
        for scenario in ("nominal", "disturbance"):
            _unchanged(root, audit["material_manifest_sha256"], f"before_{coefficients}_{scenario}")
            record = engine._evaluate(  # noqa: SLF001
                state=states[coefficients],
                scale=0.0,
                scenario=scenario,
                topology=topology,
                motion=motion,
                material_sha=audit["material_manifest_sha256"],
            )
            record["survival_coefficient"] = coefficients[0]
            record["reward_coefficient"] = coefficients[1]
            name = f"s{coefficients[0]:g}_r{coefficients[1]:g}_{scenario}.json".replace("-", "m")
            full_support._write_json_exclusive(evaluations / name, record)  # noqa: SLF001
            print(  # noqa: T201
                f"s={coefficients[0]:g} r={coefficients[1]:g} {scenario} "
                f"steps={record['completed_transitions']} q9={record['terminal_q9']}",
                flush=True,
            )
            records.append(record)
    assessment = _assess(records)
    candidate = None
    if assessment["candidate_selected"] is True:
        coefficients = tuple(float(value) for value in assessment["selected_coefficients"])
        path = run_dir / "checkpoints" / CANDIDATE_FILENAME
        engine._write_checkpoint(  # noqa: SLF001
            path,
            {
                "schema_version": 1,
                "kind": "g1_true23_sonic_rank256_two_direction_screen_candidate_v1",
                "contract_sha256": CONTRACT_SHA256,
                "source_overlay": overlay,
                "policy_state_dict": states[coefficients],
                "policy_state_sha256": assessment["selected_policy_state_sha256"],
                "selected_coefficients": list(coefficients),
                "training_transitions": 0,
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
        "kind": "g1_true23_sonic_rank256_two_direction_screen_result_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "source_overlay": overlay,
        "direction_cosine": audit["material_manifest"]["direction_cosine"],
        "evaluations": records,
        "assessment": assessment,
        "candidate": candidate,
        "training_transitions": 0,
        "evaluation_runs": len(records),
        "optimizer_steps": 0,
        "critic_updates": 0,
        "support_qualified": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    _unchanged(root, audit["material_manifest_sha256"], "before_result")
    full_support._write_json_exclusive(run_dir / RESULT_FILENAME, result)  # noqa: SLF001
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--repository-root", type=Path, default=ROOT)
    screen = sub.add_parser("screen")
    screen.add_argument("--repository-root", type=Path, default=ROOT)
    screen.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        report = _preflight(args.repository_root.expanduser().resolve(strict=True))
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    result = run(args.repository_root, args.run_dir)
    print(json.dumps(result["assessment"], indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if result["candidate"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
