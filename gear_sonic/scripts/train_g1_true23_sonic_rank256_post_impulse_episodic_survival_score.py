"""Fit final bounded score direction using episodic post-impulse credit only."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from typing import Any

import torch

from gear_sonic.scripts import train_g1_true23_sonic_rank256_post_impulse_survival_score as parent
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_post_impulse_episodic_survival_score_v2.json"
)
CONTRACT_SHA256 = "a727f20f8cfbb162ee9ae8792a2b93521062bba8a18269533a0995b5d3424fff"
KIND = "g1_true23_sonic_rank256_post_impulse_episodic_survival_score_result_v2"
RESULT_FILENAME = "rank256_post_impulse_episodic_survival_score_result.json"
FAILURE_FILENAME = "rank256_post_impulse_episodic_survival_score_failure.json"
CHECKPOINT_FILENAME = "rank256_post_impulse_episodic_survival_score_candidate.pt"
DIRECTION_FILENAME = "rank256_post_impulse_episodic_survival_score_direction.pt"


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("post-impulse episodic contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    collection = body.get("collection", {})
    gradient = body.get("gradient", {})
    parents = body.get("parents", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_post_impulse_episodic_survival_score_contract_v2"
        or collection.get("all_envs_receive_exact_vector") is not True
        or collection.get("first_recovery_observation_transition") != parent.FIRST_RECOVERY_TRANSITION
        or gradient.get("credit_assignment") != "post_impulse_observations_only"
        or gradient.get("normalization") != "first_recovery_survivor_population_z_score"
        or gradient.get("scales") != list(parent.SCALES)
        or gradient.get("minimum_first_recovery_survivors") != parent.MIN_FIRST_RECOVERY_SURVIVORS
        or gradient.get("minimum_population_std") != parent.MIN_STD
        or body.get("boundaries", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("post-impulse episodic contract semantic mismatch")
    for name in ("grouped_result", "failed_v1_preflight", "failed_v1_material"):
        artifact = Path(parents[f"{name}_path"]).resolve(strict=True)
        if artifact.is_symlink() or sha256_file(artifact) != parents[f"{name}_sha256"]:
            raise ValueError(f"post-impulse episodic parent mismatch: {name}")
    if parents.get("failed_v1_policy_mutations") != 0:
        raise ValueError("failed v1 mutation boundary mismatch")
    return body


def _episodic_post_impulse_weights(
    lengths: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if lengths.shape != (parent.parent.NUM_ENVS,) or active_mask.shape != (
        parent.parent.COLLECTION_STEPS,
        parent.parent.NUM_ENVS,
    ):
        raise ValueError("post-impulse episodic tensor shape mismatch")
    first_mask = active_mask[parent.FIRST_RECOVERY_TRANSITION]
    survivor_count = int(torch.count_nonzero(first_mask).detach().cpu())
    if survivor_count < parent.MIN_FIRST_RECOVERY_SURVIVORS:
        raise RuntimeError("post-impulse episodic survivor gate failed")
    outcomes = lengths[first_mask].to(torch.float32) / float(parent.parent.COLLECTION_STEPS)
    mean = outcomes.mean()
    std = outcomes.std(unbiased=False)
    if not bool(torch.isfinite(std)) or float(std.detach().cpu()) < parent.MIN_STD:
        raise RuntimeError("post-impulse episodic variance gate failed")
    centered = (outcomes - mean) / std
    weights = torch.zeros_like(active_mask, dtype=torch.float32)
    post_mask = active_mask[parent.FIRST_RECOVERY_TRANSITION :, first_mask]
    weights[parent.FIRST_RECOVERY_TRANSITION :, first_mask] = post_mask.to(torch.float32) * centered.unsqueeze(0)
    post_count = int(torch.count_nonzero(post_mask).detach().cpu())
    evidence = [
        {
            "name": "post_impulse_episodic_recovery",
            "first_recovery_transition": parent.FIRST_RECOVERY_TRANSITION,
            "first_recovery_q9": 9 + parent.FIRST_RECOVERY_TRANSITION,
            "first_recovery_survivor_count": survivor_count,
            "post_impulse_weighted_action_count": post_count,
            "pre_impulse_nonzero_weight_count": int(
                torch.count_nonzero(weights[: parent.FIRST_RECOVERY_TRANSITION]).detach().cpu()
            ),
            "survivor_outcome_mean": float(mean.detach().cpu()),
            "survivor_outcome_population_std": float(std.detach().cpu()),
            "minimum_survivor_length": int(lengths[first_mask].min().detach().cpu()),
            "maximum_survivor_length": int(lengths[first_mask].max().detach().cpu()),
        }
    ]
    return weights, evidence


@contextmanager
def _scope(root: Path) -> Iterator[None]:
    contract = _load_contract(root)
    names = (
        "CONTRACT_RELATIVE_PATH",
        "CONTRACT_SHA256",
        "KIND",
        "RESULT_FILENAME",
        "FAILURE_FILENAME",
        "CHECKPOINT_FILENAME",
        "DIRECTION_FILENAME",
        "POST_EXTRA_SOURCE_RELATIVE_PATHS",
        "_load_contract",
        "_post_impulse_weights",
    )
    saved = {name: getattr(parent, name) for name in names}

    def load_current(_root: Path) -> Mapping[str, Any]:
        return copy.deepcopy(contract)

    try:
        parent.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
        parent.CONTRACT_SHA256 = CONTRACT_SHA256
        parent.KIND = KIND
        parent.RESULT_FILENAME = RESULT_FILENAME
        parent.FAILURE_FILENAME = FAILURE_FILENAME
        parent.CHECKPOINT_FILENAME = CHECKPOINT_FILENAME
        parent.DIRECTION_FILENAME = DIRECTION_FILENAME
        parent.POST_EXTRA_SOURCE_RELATIVE_PATHS = (
            Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_post_impulse_episodic_survival_score.py"),
        )
        parent._load_contract = load_current
        parent._post_impulse_weights = _episodic_post_impulse_weights
        yield
    finally:
        for name, value in saved.items():
            setattr(parent, name, value)


def preflight(repository_root: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        report = dict(parent.preflight(root))
    report["episodic_post_impulse_credit"] = {
        "first_recovery_q9": 9 + parent.FIRST_RECOVERY_TRANSITION,
        "pre_impulse_actions_weighted": False,
        "population": "alive_at_first_recovery_observation",
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
        print(f"post-impulse episodic score failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result["assessment"], indent=2, sort_keys=True))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
