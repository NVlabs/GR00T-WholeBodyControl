"""Run one grouped survival-score iteration from reward-direction scale 1."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from typing import Any

import torch

from gear_sonic.scripts import train_g1_true23_sonic_rank256_grouped_multiobjective_survival_score as grouped
from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as fs
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_iterative_grouped_survival_score_v2.json"
)
CONTRACT_SHA256 = "bc77be53fd840a7a1c68cdc7a1ec09577f6b16c221fb52d05a7191f260e0f845"
SCALES = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
DIRECTION_KEYS = (
    "actor_module.decoders.g1_dyn.module.14.bias",
    "actor_module.decoders.g1_dyn.module.14.weight",
    "actor_module.decoders.g1_dyn.module.16.bias",
    "actor_module.decoders.g1_dyn.module.16.weight",
)
RESULT_FILENAME = "rank256_iterative_grouped_survival_score_result.json"
FAILURE_FILENAME = "rank256_iterative_grouped_survival_score_failure.json"
CHECKPOINT_FILENAME = "rank256_iterative_grouped_survival_score_candidate.pt"
ORIGINAL_NOMINAL_MINIMUM = 486
ORIGINAL_DISTURBANCE_STRICTLY_ABOVE = 287


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("iterative grouped contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    parents = body.get("parents", {})
    gradient = body.get("gradient", {})
    evaluation = body.get("evaluation", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_iterative_grouped_survival_score_contract_v2"
        or gradient.get("scales") != list(SCALES)
        or gradient.get("one_iteration_only") is not True
        or body.get("shifted_initialization", {}).get("reward_direction_scale") != 1.0
        or evaluation.get("original_policy_nominal_minimum_transitions") != ORIGINAL_NOMINAL_MINIMUM
        or evaluation.get("original_policy_disturbance_strictly_above_transitions")
        != ORIGINAL_DISTURBANCE_STRICTLY_ABOVE
        or body.get("boundaries", {}).get("hardware_authorized") is not False
        or body.get("boundaries", {}).get("robot_or_network_commands_permitted") is not False
    ):
        raise ValueError("iterative grouped contract semantic mismatch")
    paths = (
        (root / parents["grouped_v1_contract_relative_path"], parents["grouped_v1_contract_sha256"]),
        (Path(parents["reward_result_path"]), parents["reward_result_sha256"]),
        (Path(parents["reward_direction_path"]), parents["reward_direction_sha256"]),
    )
    for raw, expected in paths:
        resolved = raw.resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file() or sha256_file(resolved) != expected:
            raise ValueError("iterative grouped parent mismatch")
    return body


def _reward_direction(contract: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(contract["parents"]["reward_direction_path"]), map_location="cpu", weights_only=True)
    state = payload.get("direction_state_dict")
    if not isinstance(state, Mapping) or tuple(state) != DIRECTION_KEYS:
        raise ValueError("iterative grouped reward direction namespace mismatch")
    result = {name: value.detach().cpu().to(torch.float32).contiguous().clone() for name, value in state.items()}
    if fs._state_sha256(result) != contract["parents"]["reward_direction_state_sha256"]:  # noqa: SLF001
        raise ValueError("iterative grouped reward direction state mismatch")
    return result


def _shifted_state(
    baseline: Mapping[str, torch.Tensor], reward: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    result = {name: value.detach().cpu().contiguous().clone() for name, value in baseline.items()}
    for name in DIRECTION_KEYS:
        result[name] = torch.add(result[name], reward[name]).contiguous()
    return result


def _assess(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(float(record["scale"]), str(record["scenario"])): record for record in records}
    eligible: list[Mapping[str, Any]] = []
    for scale in SCALES:
        nominal = by_key[(scale, "nominal")]
        disturbance = by_key[(scale, "disturbance")]
        if (
            grouped.parent._clean(nominal)  # noqa: SLF001
            and grouped.parent._clean(disturbance)  # noqa: SLF001
            and int(nominal.get("completed_transitions", -1)) >= ORIGINAL_NOMINAL_MINIMUM
            and int(disturbance.get("completed_transitions", -1)) > ORIGINAL_DISTURBANCE_STRICTLY_ABOVE
        ):
            eligible.append({"scale": scale, "nominal": nominal, "disturbance": disturbance})
    selected = max(
        eligible,
        key=lambda item: (
            min(
                int(item["nominal"]["completed_transitions"]),
                int(item["disturbance"]["completed_transitions"]),
            ),
            int(item["nominal"]["completed_transitions"]) + int(item["disturbance"]["completed_transitions"]),
            -abs(float(item["scale"])),
        ),
        default=None,
    )
    shifted_nominal = by_key[(0.0, "nominal")]
    shifted_disturbance = by_key[(0.0, "disturbance")]
    return {
        "shifted_baseline_nominal_completed_transitions": shifted_nominal["completed_transitions"],
        "shifted_baseline_disturbance_completed_transitions": shifted_disturbance["completed_transitions"],
        "admission_nominal_minimum_transitions": ORIGINAL_NOMINAL_MINIMUM,
        "admission_disturbance_strictly_above_transitions": ORIGINAL_DISTURBANCE_STRICTLY_ABOVE,
        "candidate_selected": selected is not None,
        "selected_scale": None if selected is None else selected["scale"],
        "selected_policy_state_sha256": (None if selected is None else selected["nominal"]["policy_state_sha256"]),
        "selected_nominal_completed_transitions": (
            None if selected is None else selected["nominal"]["completed_transitions"]
        ),
        "selected_disturbance_completed_transitions": (
            None if selected is None else selected["disturbance"]["completed_transitions"]
        ),
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


@contextmanager
def _scope(root: Path) -> Iterator[None]:
    contract = _load_contract(root)
    reward = _reward_direction(contract)
    with grouped._scope(root):  # noqa: SLF001
        parent = grouped.parent
        names = (
            "CONTRACT_RELATIVE_PATH",
            "CONTRACT_SHA256",
            "KIND",
            "SCALES",
            "RESULT_FILENAME",
            "FAILURE_FILENAME",
            "CHECKPOINT_FILENAME",
            "EXTRA_SOURCE_RELATIVE_PATHS",
            "_load_contract",
            "_rank256_state",
            "_assess",
        )
        saved = {name: getattr(parent, name) for name in names}
        merged = copy.deepcopy(saved["_load_contract"](root))
        for section in ("collection", "gradient", "evaluation", "boundaries"):
            merged[section] = {**dict(merged[section]), **dict(contract[section])}
        merged["kind"] = contract["kind"]
        merged["role"] = contract["role"]
        merged["parents"] = copy.deepcopy(contract["parents"])
        original_rank256_state = saved["_rank256_state"]

        def load_current(_root: Path) -> Mapping[str, Any]:
            return copy.deepcopy(merged)

        def shifted_rank256_state(
            current_root: Path, current_contract: Mapping[str, Any]
        ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
            baseline, overlay = original_rank256_state(current_root, current_contract)
            shifted = _shifted_state(baseline, reward)
            overlay = {
                **overlay,
                "shifted_from_reward_direction_scale": 1.0,
                "reward_direction_state_sha256": contract["parents"]["reward_direction_state_sha256"],
                "shifted_policy_state_sha256": fs._state_sha256(shifted),  # noqa: SLF001
            }
            return shifted, overlay

        try:
            parent.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
            parent.CONTRACT_SHA256 = CONTRACT_SHA256
            parent.KIND = "g1_true23_sonic_rank256_iterative_grouped_survival_score_result_v2"
            parent.SCALES = SCALES
            parent.RESULT_FILENAME = RESULT_FILENAME
            parent.FAILURE_FILENAME = FAILURE_FILENAME
            parent.CHECKPOINT_FILENAME = CHECKPOINT_FILENAME
            parent.EXTRA_SOURCE_RELATIVE_PATHS = (
                Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_grouped_multiobjective_survival_score.py"),
                Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_iterative_grouped_survival_score.py"),
            )
            parent._load_contract = load_current
            parent._rank256_state = shifted_rank256_state
            parent._assess = _assess
            yield
        finally:
            for name, value in saved.items():
                setattr(parent, name, value)


def preflight(repository_root: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        report = dict(grouped.parent.preflight(root))
    report["iterative_shift"] = {
        "reward_direction_scale": 1.0,
        "one_gradient_iteration": True,
        "absolute_original_policy_admission_gate": True,
    }
    return report


def run(repository_root: Path, run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        return grouped.parent.run(root, run_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=ROOT)
    train = sub.add_parser("train")
    train.add_argument("--repository-root", type=Path, default=ROOT)
    train.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight(args.repository_root)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    try:
        result = run(args.repository_root, args.run_dir)
    except Exception as error:
        print(f"iterative grouped score failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result["assessment"], indent=2, sort_keys=True))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
