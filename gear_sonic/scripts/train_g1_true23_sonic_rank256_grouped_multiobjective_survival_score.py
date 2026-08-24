"""Fit one equal-weight grouped nominal/exact-impulse survival-score direction."""

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

from gear_sonic.scripts import train_g1_true23_sonic_rank256_disturbance_survival_score as parent
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_grouped_multiobjective_survival_score_v1.json"
)
CONTRACT_SHA256 = "1ee5a68d567be06020358acce3f6166903c0695c9e96a4ee68ab241f1f4c59dd"
SCALES = (-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
GROUPS = (("nominal", 0, 64), ("exact_impulse", 64, 128))
RESULT_FILENAME = "rank256_grouped_multiobjective_survival_score_result.json"
FAILURE_FILENAME = "rank256_grouped_multiobjective_survival_score_failure.json"
CHECKPOINT_FILENAME = "rank256_grouped_multiobjective_survival_score_candidate.pt"
WEIGHT_BUILDER = None
NORMALIZATION_NAME = "within_group_population_z_score"
CONTRIBUTION_NAME = "equal_active_transition_normalized"


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("grouped multiobjective contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    groups = body.get("collection", {}).get("groups")
    parents = body.get("parents", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_grouped_multiobjective_survival_score_contract_v1"
        or groups
        != [
            {"name": "nominal", "start": 0, "stop": 64, "impulse": False},
            {"name": "exact_impulse", "start": 64, "stop": 128, "impulse": True},
        ]
        or body.get("gradient", {}).get("scales") != list(SCALES)
        or body.get("gradient", {}).get("normalization") != "within_group_population_z_score"
        or body.get("boundaries", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("grouped multiobjective contract semantic mismatch")
    for prefix in ("mixed", "exact_impulse"):
        result = Path(parents[f"{prefix}_result_path"]).resolve(strict=True)
        if result.is_symlink() or sha256_file(result) != parents[f"{prefix}_result_sha256"]:
            raise ValueError(f"grouped multiobjective parent mismatch: {prefix}")
    return body


def _group_standardized_weights(
    lengths: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if lengths.shape != (parent.NUM_ENVS,) or active_mask.shape != (
        parent.COLLECTION_STEPS,
        parent.NUM_ENVS,
    ):
        raise ValueError("grouped outcome tensor shape mismatch")
    weights = torch.zeros_like(active_mask, dtype=torch.float32)
    evidence: list[dict[str, Any]] = []
    group_active_counts: list[int] = []
    standardized: list[tuple[int, int, torch.Tensor]] = []
    for name, start, stop in GROUPS:
        outcomes = lengths[start:stop].to(torch.float32) / float(parent.COLLECTION_STEPS)
        mean = outcomes.mean()
        std = outcomes.std(unbiased=False)
        if not bool(torch.isfinite(std)) or float(std.detach().cpu()) < 0.01:
            raise RuntimeError(f"grouped outcome variance insufficient: {name}")
        centered = (outcomes - mean) / std
        active_count = int(torch.count_nonzero(active_mask[:, start:stop]).detach().cpu())
        if active_count <= 0:
            raise RuntimeError(f"grouped active count invalid: {name}")
        standardized.append((start, stop, centered))
        group_active_counts.append(active_count)
        group_lengths = lengths[start:stop].detach().cpu()
        evidence.append(
            {
                "name": name,
                "environment_start": start,
                "environment_stop": stop,
                "active_transition_count": active_count,
                "minimum_first_episode_length": int(group_lengths.min()),
                "median_first_episode_length": float(group_lengths.median()),
                "maximum_first_episode_length": int(group_lengths.max()),
                "outcome_mean": float(mean.detach().cpu()),
                "outcome_population_std": float(std.detach().cpu()),
            }
        )
    total_active = sum(group_active_counts)
    for (start, stop, centered), active_count in zip(standardized, group_active_counts, strict=True):
        equal_group_scale = total_active / (len(GROUPS) * active_count)
        weights[:, start:stop] = (
            active_mask[:, start:stop].to(torch.float32) * centered.unsqueeze(0) * equal_group_scale
        )
    return weights, evidence


def _collect_and_score(*, actor: Any, wrapped: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    base = parent.parent
    fs = parent.fs
    raw_env = wrapped.unwrapped
    if int(raw_env.num_envs) != parent.NUM_ENVS:
        raise ValueError("grouped collection env count mismatch")
    observations = wrapped.get_observations().to(base.DEVICE)
    active = torch.ones(parent.NUM_ENVS, dtype=torch.bool, device=base.DEVICE)
    tokenizer_rows: list[torch.Tensor] = []
    policy_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    done_rows: list[torch.Tensor] = []
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
                observations, _rewards, dones, extras = wrapped.step(actions)
                recorder.finish(dones)
                done_rows.append(dones.detach().to(dtype=torch.bool).clone())
                extras["log"] = {}
                observations = observations.to(base.DEVICE)
                active = active & ~dones.to(dtype=torch.bool, device=base.DEVICE)
    finally:
        recorder.restore()
    reward_evidence = base._materialize_360_reward_evidence(recorder)  # noqa: SLF001
    tokenizer = torch.stack(tokenizer_rows)
    policy = torch.stack(policy_rows)
    actions = torch.stack(action_rows)
    dones = torch.stack(done_rows)
    active_mask = torch.stack(active_rows)
    q9 = torch.stack(q9_rows)
    if (
        tokenizer.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS, 268)
        or policy.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS, 930)
        or actions.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS, fs.ACTION_DIM)
        or dones.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS)
        or active_mask.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS)
        or not bool(torch.isfinite(tokenizer).all())
        or not bool(torch.isfinite(policy).all())
        or not bool(torch.isfinite(actions).all())
    ):
        raise RuntimeError("grouped collected tensor mismatch")
    expected_q9 = torch.arange(9, 9 + parent.COLLECTION_STEPS, device=base.DEVICE).unsqueeze(1)
    q9_mismatch = int(torch.count_nonzero(active_mask & (q9 != expected_q9)).detach().cpu())
    if q9_mismatch or any(
        reward_evidence.get(name) != 0
        for name in ("nonfinite_count", "action_semantics_mismatch_count", "raw_clip_required_count")
    ):
        raise RuntimeError("grouped collection safety gate failed")
    lengths = active_mask.sum(dim=0).to(torch.float32)
    weight_builder = _group_standardized_weights if WEIGHT_BUILDER is None else WEIGHT_BUILDER
    weights, group_evidence = weight_builder(lengths, active_mask)
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
            raise RuntimeError("grouped gradient loss nonfinite")
        chunk.backward()
        total_loss += float(chunk.detach().cpu())
    gradients = {
        state_name: named[parameter_name].grad.detach().cpu().float().contiguous().clone()
        for parameter_name, state_name in zip(
            fs.TRAINABLE_ACTOR_PARAMETERS, base.STATE_PARAMETER_NAMES, strict=True
        )
    }
    direction, normalization = base.normalized_negative_gradient_direction(gradients)
    length_cpu = lengths.detach().cpu()
    evidence = {
        "collection_steps": parent.COLLECTION_STEPS,
        "num_envs": parent.NUM_ENVS,
        "total_transitions": parent.TOTAL_TRANSITIONS,
        "first_episode_active_transition_count": active_count,
        "autoreset_transition_count_excluded": parent.TOTAL_TRANSITIONS - active_count,
        "minimum_first_episode_length": int(length_cpu.min()),
        "median_first_episode_length": float(length_cpu.median()),
        "maximum_first_episode_length": int(length_cpu.max()),
        "group_normalization": NORMALIZATION_NAME,
        "group_contribution": CONTRIBUTION_NAME,
        "groups": group_evidence,
        "loss": total_loss,
        "gradient_state_sha256": fs._state_sha256(gradients),  # noqa: SLF001
        "direction_state_sha256": fs._state_sha256(direction),  # noqa: SLF001
        "reward_evidence": reward_evidence,
        **normalization,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "teacher_labels_used": False,
    }
    return direction, evidence


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

    def grouped_vectors(num_envs: int, device: Any) -> torch.Tensor:
        if num_envs != 128:
            raise ValueError("grouped impulse env count mismatch")
        values = np.zeros((num_envs, 6), dtype=np.float32)
        values[64:] = np.asarray(contract["collection"]["exact_vector"], dtype=np.float32)
        return torch.from_numpy(values).to(device=device)

    try:
        parent.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
        parent.CONTRACT_SHA256 = CONTRACT_SHA256
        parent.KIND = "g1_true23_sonic_rank256_grouped_multiobjective_survival_score_result_v1"
        parent.SCALES = SCALES
        parent.RESULT_FILENAME = RESULT_FILENAME
        parent.FAILURE_FILENAME = FAILURE_FILENAME
        parent.CHECKPOINT_FILENAME = CHECKPOINT_FILENAME
        parent.COLLECT_AND_SCORE = _collect_and_score
        parent.EXTRA_SOURCE_RELATIVE_PATHS = (
            Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_grouped_multiobjective_survival_score.py"),
        )
        parent._load_contract = load_current
        parent._impulse_vectors = grouped_vectors
        yield
    finally:
        for name, value in saved.items():
            setattr(parent, name, value)


def preflight(repository_root: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        report = dict(parent.preflight(root))
    report["grouped_multiobjective"] = {
        "groups": [name for name, _start, _stop in GROUPS],
        "group_sizes": [stop - start for _name, start, stop in GROUPS],
        "within_group_normalization": True,
        "equal_group_contribution": True,
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
        print(f"grouped survival score failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result["assessment"], indent=2, sort_keys=True))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
