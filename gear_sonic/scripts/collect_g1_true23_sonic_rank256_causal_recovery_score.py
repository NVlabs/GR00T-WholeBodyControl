"""Collect one causal recovery-score direction; screen occurs in fresh processes."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from gear_sonic.scripts import train_g1_true23_sonic_rank256_grouped_multiobjective_survival_score as grouped
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_causal_recovery_score_v1.json"
)
CONTRACT_SHA256 = "c964aa32d7603981711d43849fc6937f3b8417cd0d38cb6dc9e53df07b579fc5"
SCALES = (0.0,)
FIRST_CREDITED_TRANSITION = 242
MIN_ACTIVE = 16
MIN_VALID = 100
MIN_VALID_BY_GROUP = {"nominal": MIN_VALID, "exact_impulse": MIN_VALID}
RESULT_FILENAME = "rank256_causal_recovery_score_collection_result.json"
FAILURE_FILENAME = "rank256_causal_recovery_score_collection_failure.json"
CHECKPOINT_FILENAME = "rank256_causal_recovery_score_unused_candidate.pt"
DIRECTION_FILENAME = "rank256_causal_recovery_score_direction.pt"


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("causal recovery-score contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    parents = body.get("parents", {})
    collection = body.get("collection", {})
    objective = body.get("objective", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_causal_recovery_score_contract_v1"
        or collection.get("first_credited_global_transition") != FIRST_CREDITED_TRANSITION
        or collection.get("first_credited_q9") != 251
        or objective.get("minimum_active_per_group_transition") != MIN_ACTIVE
        or objective.get("minimum_valid_transitions_per_group") != MIN_VALID
        or objective.get("return_to_go") is not True
        or objective.get("equal_group_contribution") is not True
        or body.get("gradient", {}).get("scales") != list(SCALES)
        or body.get("gradient", {}).get("direction_only_collection") is not True
        or body.get("boundaries", {}).get("hardware_authorized") is not False
        or body.get("boundaries", {}).get("robot_or_network_commands_permitted") is not False
    ):
        raise ValueError("causal recovery-score contract semantic mismatch")
    checks = (
        (root / parents["grouped_contract_relative_path"], parents["grouped_contract_sha256"]),
        (Path(parents["causal_trace_path"]), parents["causal_trace_sha256"]),
        (Path(parents["iterative_screen_path"]), parents["iterative_screen_sha256"]),
    )
    for raw, expected in checks:
        resolved = raw.resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file() or sha256_file(resolved) != expected:
            raise ValueError("causal recovery-score parent mismatch")
    return body


def _causal_score(
    *,
    anchor_pos_error: torch.Tensor,
    anchor_ori_error: torch.Tensor,
    anchor_lin_vel_error: torch.Tensor,
    anchor_ang_vel_error: torch.Tensor,
    non_timeout_terminal: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if (
        anchor_pos_error.ndim != 1
        or anchor_ori_error.shape != anchor_pos_error.shape
        or anchor_lin_vel_error.shape != anchor_pos_error.shape
        or anchor_ang_vel_error.shape != anchor_pos_error.shape
        or non_timeout_terminal.shape != anchor_pos_error.shape
        or non_timeout_terminal.dtype != torch.bool
    ):
        raise ValueError("causal score tensor ABI mismatch")
    terms = {
        "anchor_position": 2.0 * torch.square(anchor_pos_error / 0.25),
        "anchor_orientation": 0.5 * torch.square(anchor_ori_error / 1.0),
        "anchor_linear_velocity": 0.25 * torch.square(anchor_lin_vel_error / 2.0),
        "anchor_angular_velocity": 0.25 * torch.square(anchor_ang_vel_error / 4.0),
        "non_timeout_terminal": 50.0 * non_timeout_terminal.to(torch.float32),
    }
    score = -sum(terms.values())
    if not bool(torch.isfinite(score).all()):
        raise RuntimeError("causal score nonfinite")
    return score, terms


class _CausalMetricRecorder(grouped.parent.fs._FullSupportRewardEvidenceRecorder):  # noqa: SLF001
    def _capture(self, reward: torch.Tensor, dt: float) -> None:
        super()._capture(reward, dt)
        snapshot = self._pending_snapshot
        if snapshot is None:
            raise RuntimeError("causal metric base snapshot missing")
        command = self.raw_env.command_manager.get_term("motion")
        pos = torch.linalg.vector_norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1)
        quat_dot = torch.sum(command.anchor_quat_w * command.robot_anchor_quat_w, dim=-1).abs().clamp(0.0, 1.0)
        ori = 2.0 * torch.acos(quat_dot)
        lin = torch.linalg.vector_norm(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w, dim=-1)
        ang = torch.linalg.vector_norm(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w, dim=-1)
        terminations = dict(zip(self.termination_names, snapshot["terminations"], strict=True))
        non_timeout = torch.zeros_like(pos, dtype=torch.bool)
        for name, value in terminations.items():
            if name != "time_out":
                non_timeout |= value
        score, terms = _causal_score(
            anchor_pos_error=pos,
            anchor_ori_error=ori,
            anchor_lin_vel_error=lin,
            anchor_ang_vel_error=ang,
            non_timeout_terminal=non_timeout,
        )
        snapshot["causal_score"] = score.detach().clone()
        snapshot["causal_terms"] = {name: value.detach().clone() for name, value in terms.items()}


def _return_weights(scores: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    steps = grouped.parent.COLLECTION_STEPS
    envs = grouped.parent.NUM_ENVS
    if scores.shape != (steps, envs) or active.shape != scores.shape or active.dtype != torch.bool:
        raise ValueError("causal return tensor shape mismatch")
    returns = torch.zeros_like(scores, dtype=torch.float32)
    running = torch.zeros(envs, device=scores.device, dtype=torch.float32)
    for transition in range(steps - 1, -1, -1):
        running = scores[transition].to(torch.float32) + running
        returns[transition] = running
        running = running * active[transition].to(torch.float32)
    weights = torch.zeros_like(returns)
    evidence: list[dict[str, Any]] = []
    nonzero_counts: list[int] = []
    staged: list[tuple[int, int, torch.Tensor]] = []
    for name, start, stop in grouped.GROUPS:
        group_weights = torch.zeros((steps, stop - start), device=scores.device, dtype=torch.float32)
        valid = 0
        for transition in range(FIRST_CREDITED_TRANSITION, steps):
            mask = active[transition, start:stop]
            count = int(torch.count_nonzero(mask).detach().cpu())
            if count < MIN_ACTIVE:
                continue
            values = returns[transition, start:stop][mask]
            std = values.std(unbiased=False)
            if not bool(torch.isfinite(std)) or float(std.detach().cpu()) < 1.0e-3:
                continue
            standardized = (values - values.mean()) / std
            standardized -= standardized.mean()
            group_weights[transition, mask] = standardized
            valid += 1
        required_valid = MIN_VALID_BY_GROUP[name]
        if valid < required_valid:
            raise RuntimeError(f"causal return valid transition gate failed: {name}")
        nonzero = int(torch.count_nonzero(group_weights).detach().cpu())
        if nonzero <= 0:
            raise RuntimeError(f"causal return nonzero weight gate failed: {name}")
        staged.append((start, stop, group_weights))
        nonzero_counts.append(nonzero)
        evidence.append(
            {
                "name": name,
                "required_valid_transition_count": required_valid,
                "valid_transition_count": valid,
                "nonzero_weight_count": nonzero,
                "first_episode_active_transition_count": int(
                    torch.count_nonzero(active[:, start:stop]).detach().cpu()
                ),
            }
        )
    total_nonzero = sum(nonzero_counts)
    for (start, stop, group_weights), nonzero in zip(staged, nonzero_counts, strict=True):
        weights[:, start:stop] = group_weights * (total_nonzero / (len(staged) * nonzero))
    return weights, {
        "first_credited_transition": FIRST_CREDITED_TRANSITION,
        "first_credited_q9": 9 + FIRST_CREDITED_TRANSITION,
        "groups": evidence,
        "precredited_nonzero_weight_count": int(
            torch.count_nonzero(weights[:FIRST_CREDITED_TRANSITION]).detach().cpu()
        ),
        "weight_state_sha256": hashlib.sha256(weights.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
    }


def _collect_and_score(*, actor: Any, wrapped: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    parent = grouped.parent
    base = parent.parent
    fs = parent.fs
    raw_env = wrapped.unwrapped
    observations = wrapped.get_observations().to(base.DEVICE)
    active = torch.ones(parent.NUM_ENVS, dtype=torch.bool, device=base.DEVICE)
    tokenizer_rows: list[torch.Tensor] = []
    policy_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    active_rows: list[torch.Tensor] = []
    q9_rows: list[torch.Tensor] = []
    recorder = _CausalMetricRecorder(raw_env)
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
                extras["log"] = {}
                observations = observations.to(base.DEVICE)
                active = active & ~dones.to(dtype=torch.bool, device=base.DEVICE)
    finally:
        recorder.restore()
    reward_evidence = base._materialize_360_reward_evidence(recorder)  # noqa: SLF001
    tokenizer = torch.stack(tokenizer_rows)
    policy = torch.stack(policy_rows)
    actions = torch.stack(action_rows)
    active_mask = torch.stack(active_rows)
    q9 = torch.stack(q9_rows)
    scores = torch.stack([snapshot["causal_score"] for snapshot in recorder._snapshots])  # noqa: SLF001
    if (
        tokenizer.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS, 268)
        or policy.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS, 930)
        or actions.shape != (parent.COLLECTION_STEPS, parent.NUM_ENVS, fs.ACTION_DIM)
        or scores.shape != active_mask.shape
        or not all(bool(torch.isfinite(value).all()) for value in (tokenizer, policy, actions, scores))
    ):
        raise RuntimeError("causal recovery collected tensor mismatch")
    expected_q9 = torch.arange(9, 9 + parent.COLLECTION_STEPS, device=base.DEVICE).unsqueeze(1)
    if int(torch.count_nonzero(active_mask & (q9 != expected_q9)).detach().cpu()) or any(
        reward_evidence.get(name) != 0
        for name in ("nonfinite_count", "action_semantics_mismatch_count", "raw_clip_required_count")
    ):
        raise RuntimeError("causal recovery collection safety gate failed")
    weights, credit = _return_weights(scores, active_mask)
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
            raise RuntimeError("causal recovery gradient loss nonfinite")
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
    score_active = scores[active_mask]
    term_sums: dict[str, float] = {}
    for name in recorder._snapshots[0]["causal_terms"]:  # noqa: SLF001
        values = torch.stack(  # noqa: SLF001
            [snapshot["causal_terms"][name] for snapshot in recorder._snapshots]
        )
        term_sums[name] = float(values[active_mask].sum().detach().cpu())
    return direction, {
        "collection_steps": parent.COLLECTION_STEPS,
        "num_envs": parent.NUM_ENVS,
        "total_transitions": parent.TOTAL_TRANSITIONS,
        "first_episode_active_transition_count": active_count,
        "autoreset_transition_count_excluded": parent.TOTAL_TRANSITIONS - active_count,
        "minimum_first_episode_length": int(lengths.min()),
        "median_first_episode_length": float(lengths.median()),
        "maximum_first_episode_length": int(lengths.max()),
        "causal_score_mean_active": float(score_active.mean().detach().cpu()),
        "causal_score_minimum_active": float(score_active.min().detach().cpu()),
        "causal_cost_term_sums_active": term_sums,
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
            "DIRECTION_FILENAME",
            "COLLECT_AND_SCORE",
            "EXTRA_SOURCE_RELATIVE_PATHS",
            "_load_contract",
        )
        saved = {name: getattr(parent, name) for name in names}
        merged = copy.deepcopy(saved["_load_contract"](root))
        for section in ("collection", "gradient", "evaluation", "boundaries"):
            merged[section] = {**dict(merged[section]), **dict(contract[section])}
        merged["kind"] = contract["kind"]
        merged["role"] = contract["role"]
        merged["parents"] = copy.deepcopy(contract["parents"])
        merged["objective"] = copy.deepcopy(contract["objective"])

        def load_current(_root: Path) -> Mapping[str, Any]:
            return copy.deepcopy(merged)

        try:
            parent.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
            parent.CONTRACT_SHA256 = CONTRACT_SHA256
            parent.KIND = "g1_true23_sonic_rank256_causal_recovery_score_collection_result_v1"
            parent.SCALES = SCALES
            parent.RESULT_FILENAME = RESULT_FILENAME
            parent.FAILURE_FILENAME = FAILURE_FILENAME
            parent.CHECKPOINT_FILENAME = CHECKPOINT_FILENAME
            parent.DIRECTION_FILENAME = DIRECTION_FILENAME
            parent.COLLECT_AND_SCORE = _collect_and_score
            parent.EXTRA_SOURCE_RELATIVE_PATHS = (
                Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_grouped_multiobjective_survival_score.py"),
                Path("gear_sonic/scripts/collect_g1_true23_sonic_rank256_causal_recovery_score.py"),
            )
            parent._load_contract = load_current
            yield
        finally:
            for name, value in saved.items():
                setattr(parent, name, value)


def preflight(repository_root: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        report = dict(grouped.parent.preflight(root))
    report["causal_recovery_objective"] = {
        "first_credited_q9": 251,
        "reference_relative": True,
        "nominal_and_exact_push_equal_contribution": True,
        "direction_only": True,
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
    collect = sub.add_parser("collect")
    collect.add_argument("--repository-root", type=Path, default=ROOT)
    collect.add_argument("--run-dir", type=Path, required=True)
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
        print(f"causal recovery score failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result["assessment"], indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
