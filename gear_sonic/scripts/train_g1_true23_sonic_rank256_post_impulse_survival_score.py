"""Fit one survival-score direction from post-impulse observations only."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gear_sonic.scripts import (
    train_g1_true23_sonic_rank256_disturbance_survival_score as parent,
    train_g1_true23_sonic_rank256_grouped_multiobjective_survival_score as grouped,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_post_impulse_survival_score_v1.json"
)
CONTRACT_SHA256 = "39541cb29ff183f4797a9b504564a39f67137d9bada489cd6614f455c4c365a6"
KIND = "g1_true23_sonic_rank256_post_impulse_survival_score_result_v1"
SCALES = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
FIRST_RECOVERY_TRANSITION = 242
MIN_FIRST_RECOVERY_SURVIVORS = 32
MIN_ACTIVE = 16
MIN_STD = 0.01
MIN_STD_FRAMES = 5.1
MIN_VALID_TRANSITIONS = 16
RESULT_FILENAME = "rank256_post_impulse_survival_score_result.json"
FAILURE_FILENAME = "rank256_post_impulse_survival_score_failure.json"
CHECKPOINT_FILENAME = "rank256_post_impulse_survival_score_candidate.pt"
DIRECTION_FILENAME = "rank256_post_impulse_survival_score_direction.pt"
POST_EXTRA_SOURCE_RELATIVE_PATHS: tuple[Path, ...] = ()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("post-impulse survival contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    collection = body.get("collection", {})
    gradient = body.get("gradient", {})
    parent_entry = body.get("parent", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_post_impulse_survival_score_contract_v1"
        or collection.get("all_envs_receive_exact_vector") is not True
        or collection.get("first_recovery_observation_transition") != FIRST_RECOVERY_TRANSITION
        or gradient.get("credit_assignment") != "post_impulse_observations_only"
        or gradient.get("scales") != list(SCALES)
        or gradient.get("minimum_first_recovery_survivors") != MIN_FIRST_RECOVERY_SURVIVORS
        or gradient.get("minimum_active_per_transition") != MIN_ACTIVE
        or gradient.get("minimum_population_std_frames") != MIN_STD_FRAMES
        or gradient.get("minimum_valid_transitions") != MIN_VALID_TRANSITIONS
        or body.get("boundaries", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("post-impulse survival contract semantic mismatch")
    result = Path(parent_entry["grouped_result_path"]).resolve(strict=True)
    if result.is_symlink() or sha256_file(result) != parent_entry["grouped_result_sha256"]:
        raise ValueError("post-impulse survival parent mismatch")
    return body


def _post_impulse_weights(
    lengths: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if lengths.shape != (parent.NUM_ENVS,) or active_mask.shape != (
        parent.COLLECTION_STEPS,
        parent.NUM_ENVS,
    ):
        raise ValueError("post-impulse outcome tensor shape mismatch")
    first_survivors = int(torch.count_nonzero(active_mask[FIRST_RECOVERY_TRANSITION]).detach().cpu())
    if first_survivors < MIN_FIRST_RECOVERY_SURVIVORS:
        raise RuntimeError("post-impulse first recovery survivor gate failed")
    outcomes = lengths.to(torch.float32) / float(parent.COLLECTION_STEPS)
    standardized: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for transition in range(FIRST_RECOVERY_TRANSITION, parent.COLLECTION_STEPS):
        mask = active_mask[transition]
        count = int(torch.count_nonzero(mask).detach().cpu())
        if count < MIN_ACTIVE:
            continue
        values = outcomes[mask]
        std = values.std(unbiased=False)
        if not bool(torch.isfinite(std)) or float(std.detach().cpu()) < MIN_STD:
            continue
        standardized.append((transition, mask, (values - values.mean()) / std))
    if len(standardized) < MIN_VALID_TRANSITIONS:
        raise RuntimeError("post-impulse valid transition gate failed")
    weights = torch.zeros_like(active_mask, dtype=torch.float32)
    total_samples = sum(int(torch.count_nonzero(mask).detach().cpu()) for _, mask, _ in standardized)
    for transition, mask, centered in standardized:
        count = int(torch.count_nonzero(mask).detach().cpu())
        assigned = centered * (total_samples / (len(standardized) * count))
        assigned = assigned - assigned.mean()
        weights[transition, mask] = assigned
    first_valid = standardized[0][0]
    last_valid = standardized[-1][0]
    evidence = [
        {
            "name": "post_impulse_recovery",
            "first_recovery_transition": FIRST_RECOVERY_TRANSITION,
            "first_recovery_q9": 9 + FIRST_RECOVERY_TRANSITION,
            "first_recovery_survivor_count": first_survivors,
            "first_valid_transition": first_valid,
            "last_valid_transition": last_valid,
            "valid_transition_count": len(standardized),
            "weighted_active_sample_count": total_samples,
            "pre_impulse_nonzero_weight_count": int(
                torch.count_nonzero(weights[:FIRST_RECOVERY_TRANSITION]).detach().cpu()
            ),
        }
    ]
    return weights, evidence


@contextmanager
def _scope(root: Path) -> Iterator[None]:
    contract = _load_contract(root)
    parent_names = (
        "CONTRACT_RELATIVE_PATH",
        "CONTRACT_SHA256",
        "KIND",
        "SCALES",
        "RESULT_FILENAME",
        "FAILURE_FILENAME",
        "CHECKPOINT_FILENAME",
        "DIRECTION_FILENAME",
        "COLLECT_AND_SCORE",
        "EXTRA_SOURCE_RELATIVE_PATHS",
        "_load_contract",
        "_impulse_vectors",
    )
    grouped_names = ("WEIGHT_BUILDER", "NORMALIZATION_NAME", "CONTRIBUTION_NAME")
    saved_parent = {name: getattr(parent, name) for name in parent_names}
    saved_grouped = {name: getattr(grouped, name) for name in grouped_names}
    base_contract = copy.deepcopy(saved_parent["_load_contract"](root))
    for section in ("collection", "gradient", "evaluation", "boundaries"):
        base_contract[section] = {**dict(base_contract[section]), **dict(contract[section])}
    base_contract["kind"] = contract["kind"]
    base_contract["role"] = contract["role"]
    if "parent" in contract:
        base_contract["parent"] = copy.deepcopy(contract["parent"])
    elif "parents" in contract:
        base_contract["parents"] = copy.deepcopy(contract["parents"])
    else:
        raise ValueError("post-impulse contract parent provenance missing")

    def load_current(_root: Path) -> Mapping[str, Any]:
        return copy.deepcopy(base_contract)

    def exact_vectors(num_envs: int, device: Any) -> torch.Tensor:
        if num_envs != 128:
            raise ValueError("post-impulse env count mismatch")
        vector = np.asarray(contract["collection"]["exact_vector"], dtype=np.float32)
        return torch.from_numpy(np.repeat(vector.reshape(1, 6), num_envs, axis=0)).to(device=device)

    try:
        parent.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
        parent.CONTRACT_SHA256 = CONTRACT_SHA256
        parent.KIND = KIND
        parent.SCALES = SCALES
        parent.RESULT_FILENAME = RESULT_FILENAME
        parent.FAILURE_FILENAME = FAILURE_FILENAME
        parent.CHECKPOINT_FILENAME = CHECKPOINT_FILENAME
        parent.DIRECTION_FILENAME = DIRECTION_FILENAME
        parent.COLLECT_AND_SCORE = grouped._collect_and_score  # noqa: SLF001
        parent.EXTRA_SOURCE_RELATIVE_PATHS = (
            Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_grouped_multiobjective_survival_score.py"),
            Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_post_impulse_survival_score.py"),
            *POST_EXTRA_SOURCE_RELATIVE_PATHS,
        )
        parent._load_contract = load_current
        parent._impulse_vectors = exact_vectors
        grouped.WEIGHT_BUILDER = _post_impulse_weights
        grouped.NORMALIZATION_NAME = "per_transition_active_population_z_score"
        grouped.CONTRIBUTION_NAME = "equal_valid_transition_normalized"
        yield
    finally:
        for name, value in saved_parent.items():
            setattr(parent, name, value)
        for name, value in saved_grouped.items():
            setattr(grouped, name, value)


def preflight(repository_root: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        report = dict(parent.preflight(root))
    report["post_impulse_credit_assignment"] = {
        "first_recovery_transition": FIRST_RECOVERY_TRANSITION,
        "first_recovery_q9": 9 + FIRST_RECOVERY_TRANSITION,
        "pre_impulse_actions_weighted": False,
        "per_transition_baseline": True,
        "equal_valid_transition_contribution": True,
    }
    return report


def run(repository_root: Path, run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        return parent.run(root, run_dir)


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
        print(f"post-impulse survival score failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result["assessment"], indent=2, sort_keys=True))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
