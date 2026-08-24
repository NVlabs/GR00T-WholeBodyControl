"""Fit one exact-push recovery direction from post-impulse reward-to-go."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gear_sonic.scripts import train_g1_true23_sonic_rank256_disturbance_survival_score as parent
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_post_impulse_reward_to_go_score_v1.json"
)
CONTRACT_SHA256 = "ece1d898e0590272e88306063d9fd1913d4e334e37701503f55b63e5bc4b0efc"
KIND = "g1_true23_sonic_rank256_post_impulse_reward_to_go_score_result_v1"
SCALES = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
FIRST_CREDITED_TRANSITION = 242
MIN_SURVIVORS = 32
MIN_ACTIVE = 16
MIN_STD = 0.001
MIN_VALID_TRANSITIONS = 16
RESULT_FILENAME = "rank256_post_impulse_reward_to_go_score_result.json"
FAILURE_FILENAME = "rank256_post_impulse_reward_to_go_score_failure.json"
CHECKPOINT_FILENAME = "rank256_post_impulse_reward_to_go_score_candidate.pt"
DIRECTION_FILENAME = "rank256_post_impulse_reward_to_go_score_direction.pt"


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("reward-to-go contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    collection = body.get("collection", {})
    gradient = body.get("gradient", {})
    parents = body.get("parents", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_post_impulse_reward_to_go_score_contract_v1"
        or collection.get("all_envs_receive_exact_vector") is not True
        or collection.get("strict_tracking_terminations_retained") is not True
        or collection.get("first_credited_transition") != FIRST_CREDITED_TRANSITION
        or gradient.get("credit_assignment") != "reward_to_go_on_post_impulse_observations_only"
        or gradient.get("scales") != list(SCALES)
        or gradient.get("minimum_first_credited_survivors") != MIN_SURVIVORS
        or gradient.get("minimum_active_per_transition") != MIN_ACTIVE
        or gradient.get("minimum_population_std") != MIN_STD
        or gradient.get("minimum_valid_transitions") != MIN_VALID_TRANSITIONS
        or body.get("boundaries", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("reward-to-go contract semantic mismatch")
    for name in ("relaxed_diagnostic", "failed_episodic_preflight", "failed_episodic_material"):
        raw = Path(parents[f"{name}_path"])
        artifact = (root / raw).resolve(strict=True) if not raw.is_absolute() else raw.resolve(strict=True)
        if artifact.is_symlink() or sha256_file(artifact) != parents[f"{name}_sha256"]:
            raise ValueError(f"reward-to-go parent mismatch: {name}")
    if parents.get("failed_episodic_policy_mutations") != 0:
        raise ValueError("failed episodic mutation boundary mismatch")
    return body


def _reward_to_go_weights(
    rewards: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if rewards.shape != active_mask.shape or rewards.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS):
        raise ValueError("reward-to-go tensor shape mismatch")
    first_survivors = int(torch.count_nonzero(active_mask[FIRST_CREDITED_TRANSITION]).detach().cpu())
    if first_survivors < MIN_SURVIVORS:
        raise RuntimeError("reward-to-go first credited survivor gate failed")
    returns = torch.zeros_like(rewards, dtype=torch.float32)
    running = torch.zeros(parent.NUM_ENVS, dtype=torch.float32, device=rewards.device)
    for transition in range(parent.COLLECTION_STEPS - 1, -1, -1):
        running = rewards[transition].to(torch.float32) + running
        returns[transition] = running
        running = running * active_mask[transition].to(torch.float32)
    weights = torch.zeros_like(rewards, dtype=torch.float32)
    valid: list[dict[str, Any]] = []
    for transition in range(FIRST_CREDITED_TRANSITION, parent.COLLECTION_STEPS):
        mask = active_mask[transition]
        count = int(torch.count_nonzero(mask).detach().cpu())
        if count < MIN_ACTIVE:
            continue
        values = returns[transition, mask]
        std = values.std(unbiased=False)
        if not bool(torch.isfinite(std)) or float(std.detach().cpu()) < MIN_STD:
            continue
        standardized = (values - values.mean()) / std
        standardized = standardized - standardized.mean()
        weights[transition, mask] = standardized
        valid.append(
            {
                "transition": transition,
                "q9": 9 + transition,
                "active_count": count,
                "return_mean": float(values.mean().detach().cpu()),
                "return_population_std": float(std.detach().cpu()),
            }
        )
    if len(valid) < MIN_VALID_TRANSITIONS:
        raise RuntimeError("reward-to-go valid transition gate failed")
    return weights, {
        "first_credited_transition": FIRST_CREDITED_TRANSITION,
        "first_credited_q9": 9 + FIRST_CREDITED_TRANSITION,
        "first_credited_survivor_count": first_survivors,
        "valid_transition_count": len(valid),
        "first_valid": valid[0],
        "last_valid": valid[-1],
        "pre_impulse_nonzero_weight_count": int(
            torch.count_nonzero(weights[:FIRST_CREDITED_TRANSITION]).detach().cpu()
        ),
        "weight_state_sha256": hashlib.sha256(weights.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
    }


def _collect_and_score(*, actor: Any, wrapped: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    base = parent.parent
    fs = parent.fs
    raw_env = wrapped.unwrapped
    observations = wrapped.get_observations().to(base.DEVICE)
    active = torch.ones(parent.NUM_ENVS, dtype=torch.bool, device=base.DEVICE)
    tokenizer_rows: list[torch.Tensor] = []
    policy_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    reward_rows: list[torch.Tensor] = []
    active_rows: list[torch.Tensor] = []
    q9_rows: list[torch.Tensor] = []
    recorder = fs._FullSupportRewardEvidenceRecorder(raw_env)  # noqa: SLF001
    raw_env.extras["log"] = {}
    try:
        with torch.no_grad():
            for _step in range(parent.COLLECTION_STEPS):
                q9 = fs._vector_q9(raw_env)  # noqa: SLF001
                tokenizer_rows.append(observations["tokenizer"].detach().clone())
                policy_rows.append(observations["policy"].detach().clone())
                active_rows.append(active.detach().clone())
                q9_rows.append(q9.detach().clone())
                actions = actor(observations, stochastic_output=True)
                action_rows.append(actions.detach().clone())
                recorder.arm(q9, active, actions)
                observations, rewards, dones, extras = wrapped.step(actions)
                recorder.finish(dones)
                reward_rows.append(rewards.detach().to(torch.float32).clone())
                extras["log"] = {}
                observations = observations.to(base.DEVICE)
                active = active & ~dones.to(dtype=torch.bool, device=base.DEVICE)
    finally:
        recorder.restore()
    reward_evidence = base._materialize_360_reward_evidence(recorder)  # noqa: SLF001
    tokenizer = torch.stack(tokenizer_rows)
    policy = torch.stack(policy_rows)
    actions = torch.stack(action_rows)
    rewards = torch.stack(reward_rows)
    active_mask = torch.stack(active_rows)
    q9 = torch.stack(q9_rows)
    if (
        tokenizer.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS, 268)
        or policy.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS, 930)
        or actions.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS, fs.ACTION_DIM)
        or rewards.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS)
        or active_mask.shape != rewards.shape
        or not all(bool(torch.isfinite(value).all()) for value in (tokenizer, policy, actions, rewards))
    ):
        raise RuntimeError("reward-to-go collected tensor mismatch")
    expected_q9 = torch.arange(9, 9 + parent.COLLECTION_STEPS, device=base.DEVICE).unsqueeze(1)
    if int(torch.count_nonzero(active_mask & (q9 != expected_q9)).detach().cpu()) or any(
        reward_evidence.get(name) != 0
        for name in ("nonfinite_count", "action_semantics_mismatch_count", "raw_clip_required_count")
    ):
        raise RuntimeError("reward-to-go collection safety gate failed")
    weights, credit = _reward_to_go_weights(rewards, active_mask)
    active_count = int(torch.count_nonzero(active_mask).detach().cpu())
    named = dict(actor.named_parameters())
    for parameter in actor.parameters():
        parameter.grad = None
    flat_tokenizer = tokenizer.flatten(0, 1)
    flat_policy = policy.flatten(0, 1)
    flat_actions = actions.flatten(0, 1)
    flat_weights = weights.flatten()
    total_loss = 0.0
    for start in range(0, parent.TOTAL_TRANSITIONS, base.GRADIENT_BATCH_SIZE):
        stop = start + base.GRADIENT_BATCH_SIZE
        actor(
            {"tokenizer": flat_tokenizer[start:stop], "policy": flat_policy[start:stop]},
            stochastic_output=True,
        )
        log_prob = actor.get_output_log_prob(flat_actions[start:stop]).reshape(-1)
        chunk = -(log_prob * flat_weights[start:stop]).sum() / float(active_count)
        if not bool(torch.isfinite(chunk)):
            raise RuntimeError("reward-to-go gradient loss nonfinite")
        chunk.backward()
        total_loss += float(chunk.detach().cpu())
    gradients = {
        state_name: named[parameter_name].grad.detach().cpu().float().contiguous().clone()
        for parameter_name, state_name in zip(
            fs.TRAINABLE_ACTOR_PARAMETERS, base.STATE_PARAMETER_NAMES, strict=True
        )
    }
    direction, normalization = base.normalized_negative_gradient_direction(gradients)
    lengths = active_mask.sum(dim=0).detach().cpu()
    return direction, {
        "collection_steps": parent.COLLECTION_STEPS,
        "num_envs": parent.NUM_ENVS,
        "total_transitions": parent.TOTAL_TRANSITIONS,
        "first_episode_active_transition_count": active_count,
        "autoreset_transition_count_excluded": parent.TOTAL_TRANSITIONS - active_count,
        "minimum_first_episode_length": int(lengths.min()),
        "median_first_episode_length": float(lengths.median()),
        "maximum_first_episode_length": int(lengths.max()),
        "credit_assignment": credit,
        "loss": total_loss,
        "gradient_state_sha256": fs._state_sha256(gradients),  # noqa: SLF001
        "direction_state_sha256": fs._state_sha256(direction),  # noqa: SLF001
        "reward_evidence": reward_evidence,
        **normalization,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "teacher_labels_used": False,
    }


@contextmanager
def _scope(root: Path) -> Iterator[None]:
    contract = _load_contract(root)
    names = (
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
    saved = {name: getattr(parent, name) for name in names}
    base_contract = copy.deepcopy(saved["_load_contract"](root))
    for section in ("collection", "gradient", "evaluation", "boundaries"):
        base_contract[section] = {**dict(base_contract[section]), **dict(contract[section])}
    base_contract["kind"] = contract["kind"]
    base_contract["role"] = contract["role"]
    base_contract["parents"] = copy.deepcopy(contract["parents"])

    def load_current(_root: Path) -> Mapping[str, Any]:
        return copy.deepcopy(base_contract)

    def exact_vectors(num_envs: int, device: Any) -> torch.Tensor:
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
        parent.COLLECT_AND_SCORE = _collect_and_score
        parent.EXTRA_SOURCE_RELATIVE_PATHS = (
            Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_post_impulse_reward_to_go_score.py"),
        )
        parent._load_contract = load_current
        parent._impulse_vectors = exact_vectors
        yield
    finally:
        for name, value in saved.items():
            setattr(parent, name, value)


def preflight(repository_root: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        report = dict(parent.preflight(root))
    report["reward_to_go_credit"] = {
        "first_credited_q9": 9 + FIRST_CREDITED_TRANSITION,
        "pre_impulse_actions_weighted": False,
        "strict_tracking_terminations_retained": True,
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
        print(f"reward-to-go score failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result["assessment"], indent=2, sort_keys=True))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
