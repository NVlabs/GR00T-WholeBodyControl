"""Collect post-push causal recovery direction at nominal-strong shifted rank256 base."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from gear_sonic.scripts import (
    collect_g1_true23_sonic_rank256_causal_recovery_score as causal,
    collect_g1_true23_sonic_rank256_causal_recovery_score_v2 as causal_v2,
    screen_g1_true23_sonic_rank256_group_balanced_clipped_update as clipped_screen,
    train_g1_true23_sonic_rank256_disturbance_survival_score as disturbance,
    train_g1_true23_sonic_survival_length_score as survival,
    train_g1_true23_sonic_task_space_ppo_full_support as full_support,
)
from gear_sonic.trl.mjlab import (
    sonic_task_space_ppo_full_support_runner as fs,
    sonic_task_space_ppo_runner as task_space,
)
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_true23_sonic_survival_score_line_search import STATE_PARAMETER_NAMES

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_shifted_base_causal_recovery_score_v1.json"
)
CONTRACT_SHA256 = "9a6ebac05c49463d579f9166fcbe69ac5e34efbbc49e7ba49ec5ae8480fe7583"
NUM_ENVS = 128
GROUP_SIZE = 64
COLLECTION_STEPS = 510
TOTAL_TRANSITIONS = NUM_ENVS * COLLECTION_STEPS
IMPULSE_TRANSITION = 241
FIRST_CREDITED_TRANSITION = 242
MIN_ACTIVE = 16
MIN_VALID = 30
GROUPS = (("exact_impulse_a", 0, GROUP_SIZE), ("exact_impulse_b", GROUP_SIZE, NUM_ENVS))
RESULT_FILENAME = "rank256_shifted_base_causal_recovery_score_result.json"
FAILURE_FILENAME = "rank256_shifted_base_causal_recovery_score_failure.json"
DIRECTION_FILENAME = "rank256_shifted_base_causal_recovery_score_direction.pt"
SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/scripts/collect_g1_true23_sonic_rank256_shifted_base_causal_recovery_score.py"),
    Path("gear_sonic/scripts/collect_g1_true23_sonic_rank256_causal_recovery_score.py"),
    Path("gear_sonic/scripts/collect_g1_true23_sonic_rank256_causal_recovery_score_v2.py"),
    Path("gear_sonic/scripts/screen_g1_true23_sonic_rank256_group_balanced_clipped_update.py"),
)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("shifted-base causal contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    parents = body.get("parents", {})
    collection = body.get("collection", {})
    objective = body.get("objective", {})
    initial = body.get("initial_policy", {})
    gradient = body.get("gradient", {})
    boundaries = body.get("boundaries", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_shifted_base_causal_recovery_score_contract_v1"
        or initial.get("policy_state_sha256") != "6acada92dd6e2700499c64255e077c7e844aa0f96b0700e2e2c5640606ae0650"
        or [initial.get(name) for name in ("reward_coefficient", "grouped_coefficient", "causal_coefficient")]
        != [1.0, 2.0, 0.0]
        or collection.get("num_envs") != NUM_ENVS
        or collection.get("steps") != COLLECTION_STEPS
        or collection.get("total_transitions") != TOTAL_TRANSITIONS
        or collection.get("all_environments_receive_exact_impulse") is not True
        or collection.get("impulse_global_transition") != IMPULSE_TRANSITION
        or collection.get("first_credited_transition") != FIRST_CREDITED_TRANSITION
        or collection.get("minimum_active_per_group_transition") != MIN_ACTIVE
        or collection.get("minimum_valid_transitions_per_group") != MIN_VALID
        or objective.get("return_to_go") is not True
        or objective.get("within_transition_replicate_group_standardization") is not True
        or objective.get("equal_replicate_group_contribution") is not True
        or objective.get("precredited_nonzero_weights_required") != 0
        or gradient.get("parameters") != list(fs.TRAINABLE_ACTOR_PARAMETERS)
        or gradient.get("optimizer_steps") != 0
        or gradient.get("critic_updates") != 0
        or gradient.get("direction_only") is not True
        or boundaries.get("hardware_authorized") is not False
        or boundaries.get("robot_or_network_commands_permitted") is not False
    ):
        raise ValueError("shifted-base causal contract semantic mismatch")
    checks = (
        (
            root / parents["clipped_screen_contract_relative_path"],
            parents["clipped_screen_contract_sha256"],
        ),
        (Path(parents["clipped_screen_result_path"]), parents["clipped_screen_result_sha256"]),
        (root / parents["causal_v2_contract_relative_path"], parents["causal_v2_contract_sha256"]),
    )
    for raw, expected in checks:
        resolved = raw.expanduser().resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file() or sha256_file(resolved) != expected:
            raise ValueError("shifted-base causal parent mismatch")
    result = json.loads(Path(parents["clipped_screen_result_path"]).read_text(encoding="utf-8"))
    base_rows = [row for row in result.get("evaluations", []) if row.get("screen_scale") == 0.0]
    if (
        result.get("assessment", {}).get("candidate_selected") is not False
        or {row.get("scenario"): row.get("completed_transitions") for row in base_rows}
        != {"nominal": 507, "disturbance": 287}
        or {row.get("policy_state_sha256") for row in base_rows} != {initial["policy_state_sha256"]}
    ):
        raise ValueError("shifted-base causal baseline evidence mismatch")
    causal_v2._load_contract(root)  # noqa: SLF001
    return body


def _inputs(
    root: Path,
) -> tuple[Mapping[str, Any], dict[str, torch.Tensor], Mapping[str, Any], Path, Path]:
    contract = _load_contract(root)
    _screen_contract, baseline, _delta, topology, motion = clipped_screen._inputs(root)  # noqa: SLF001
    observed = inspect_true23_policy_state({"policy_state_dict": baseline}, reference_profile=fs.REFERENCE_PROFILE)
    if observed != contract["initial_policy"]["policy_state_sha256"]:
        raise ValueError("shifted-base causal reconstructed policy mismatch")
    _update_contract, _initial, overlay, _topology, _motion = clipped_screen.update._inputs(root)  # noqa: SLF001
    return contract, baseline, overlay, topology, motion


def preflight(repository_root: Path) -> dict[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract, baseline, overlay, topology, motion = _inputs(root)
        sources = {
            relative.as_posix(): sha256_file((root / relative).resolve(strict=True))
            for relative in SOURCE_RELATIVE_PATHS
        }
        material = {
            "contract_sha256": CONTRACT_SHA256,
            "clipped_screen_result_sha256": contract["parents"]["clipped_screen_result_sha256"],
            "initial_policy_state_sha256": contract["initial_policy"]["policy_state_sha256"],
            "source_overlay": overlay,
            "topology_sha256": sha256_file(topology),
            "motion_sha256": sha256_file(motion),
            "sources": sources,
        }
        del baseline
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_shifted_base_causal_recovery_score_preflight_v1",
            "ready": True,
            "contract": contract,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha(material),
            "simulator_constructed": False,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_shifted_base_causal_recovery_score_preflight_v1",
            "ready": False,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "simulator_constructed": False,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }


def _materials_unchanged(root: Path, audit: Mapping[str, Any], phase: str) -> None:
    current = preflight(root)
    if current.get("ready") is not True or current.get("material_manifest_sha256") != audit.get(
        "material_manifest_sha256"
    ):
        raise RuntimeError(f"shifted-base causal materials changed: {phase}")


def _return_weights(scores: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    if scores.shape != (COLLECTION_STEPS, NUM_ENVS) or active.shape != scores.shape:
        raise ValueError("shifted-base causal score tensor shape mismatch")
    returns = torch.zeros_like(scores, dtype=torch.float32)
    running = torch.zeros(NUM_ENVS, device=scores.device, dtype=torch.float32)
    for transition in range(COLLECTION_STEPS - 1, -1, -1):
        running = scores[transition].to(torch.float32) + running
        returns[transition] = running
        running = running * active[transition].to(torch.float32)
    weights = torch.zeros_like(returns)
    group_evidence: list[dict[str, Any]] = []
    staged: list[tuple[int, int, torch.Tensor, int]] = []
    for name, start, stop in GROUPS:
        group = torch.zeros((COLLECTION_STEPS, stop - start), device=scores.device)
        valid = 0
        for transition in range(FIRST_CREDITED_TRANSITION, COLLECTION_STEPS):
            mask = active[transition, start:stop]
            count = int(torch.count_nonzero(mask).detach().cpu())
            if count < MIN_ACTIVE:
                continue
            values = returns[transition, start:stop][mask]
            std = values.std(unbiased=False)
            if not bool(torch.isfinite(std)) or float(std.detach().cpu()) < 1.0e-3:
                continue
            normalized = (values - values.mean()) / std
            normalized -= normalized.mean()
            group[transition, mask] = normalized
            valid += 1
        nonzero = int(torch.count_nonzero(group).detach().cpu())
        if valid < MIN_VALID or nonzero <= 0:
            raise RuntimeError(f"shifted-base causal replicate gate failed: {name}")
        staged.append((start, stop, group, nonzero))
        group_evidence.append(
            {
                "name": name,
                "valid_transition_count": valid,
                "required_valid_transition_count": MIN_VALID,
                "nonzero_weight_count": nonzero,
                "first_episode_active_transition_count": int(
                    torch.count_nonzero(active[:, start:stop]).detach().cpu()
                ),
            }
        )
    total_nonzero = sum(item[3] for item in staged)
    for start, stop, group, nonzero in staged:
        weights[:, start:stop] = group * (total_nonzero / (len(staged) * nonzero))
    precredited = int(torch.count_nonzero(weights[:FIRST_CREDITED_TRANSITION]).detach().cpu())
    if precredited != 0:
        raise RuntimeError("shifted-base causal precredited weights nonzero")
    return weights, {
        "first_credited_transition": FIRST_CREDITED_TRANSITION,
        "first_credited_q9": 9 + FIRST_CREDITED_TRANSITION,
        "replicate_groups": group_evidence,
        "precredited_nonzero_weight_count": precredited,
        "weight_state_sha256": hashlib.sha256(weights.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
    }


def _collect_and_score(actor: Any, wrapped: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw_env = wrapped.unwrapped
    observations = wrapped.get_observations().to(survival.DEVICE)
    active = torch.ones(NUM_ENVS, dtype=torch.bool, device=survival.DEVICE)
    tokenizer_rows: list[torch.Tensor] = []
    policy_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    active_rows: list[torch.Tensor] = []
    q9_rows: list[torch.Tensor] = []
    recorder = causal._CausalMetricRecorder(raw_env)  # noqa: SLF001
    exact = torch.from_numpy(disturbance.FAILED_VECTOR.copy()).to(device=survival.DEVICE)
    vectors = exact.expand(NUM_ENVS, -1).contiguous()
    proxy = disturbance._ImpulseProxy(wrapped, vectors)  # noqa: SLF001
    raw_env.extras["log"] = {}
    try:
        with torch.no_grad():
            for _step in range(COLLECTION_STEPS):
                q9 = fs._vector_q9(raw_env)  # noqa: SLF001
                tokenizer_rows.append(observations["tokenizer"].detach().clone())
                policy_rows.append(observations["policy"].detach().clone())
                active_rows.append(active.detach().clone())
                q9_rows.append(q9.detach().clone())
                actions = actor(observations, stochastic_output=True)
                action_rows.append(actions.detach().clone())
                recorder.arm(q9, active, actions)
                observations, _rewards, dones, extras = proxy.step(actions)
                recorder.finish(dones)
                extras["log"] = {}
                observations = observations.to(survival.DEVICE)
                active = active & ~dones.to(dtype=torch.bool, device=survival.DEVICE)
    finally:
        recorder.restore()
    with disturbance._collection_scope():  # noqa: SLF001
        reward_evidence = survival._materialize_360_reward_evidence(recorder)  # noqa: SLF001
    tokenizer = torch.stack(tokenizer_rows)
    policy = torch.stack(policy_rows)
    actions = torch.stack(action_rows)
    active_mask = torch.stack(active_rows)
    q9 = torch.stack(q9_rows)
    scores = torch.stack([snapshot["causal_score"] for snapshot in recorder._snapshots])  # noqa: SLF001
    if (
        tokenizer.shape != (COLLECTION_STEPS, NUM_ENVS, 268)
        or policy.shape != (COLLECTION_STEPS, NUM_ENVS, 930)
        or actions.shape != (COLLECTION_STEPS, NUM_ENVS, fs.ACTION_DIM)
        or scores.shape != active_mask.shape
        or not all(bool(torch.isfinite(value).all()) for value in (tokenizer, policy, actions, scores))
    ):
        raise RuntimeError("shifted-base causal collected tensor mismatch")
    expected_q9 = torch.arange(9, 9 + COLLECTION_STEPS, device=survival.DEVICE).unsqueeze(1)
    q9_mismatch = int(torch.count_nonzero(active_mask & (q9 != expected_q9)).detach().cpu())
    if q9_mismatch or any(
        reward_evidence.get(name) != 0
        for name in ("nonfinite_count", "action_semantics_mismatch_count", "raw_clip_required_count")
    ):
        raise RuntimeError("shifted-base causal collection safety gate failed")
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
    for start in range(0, TOTAL_TRANSITIONS, 512):
        stop = start + 512
        actor(
            {"tokenizer": flat_tokenizer[start:stop], "policy": flat_policy[start:stop]},
            stochastic_output=True,
        )
        log_prob = actor.get_output_log_prob(flat_actions[start:stop]).reshape(-1)
        chunk = -(log_prob * flat_weights[start:stop]).sum() / float(active_count)
        if not bool(torch.isfinite(chunk)):
            raise RuntimeError("shifted-base causal gradient loss nonfinite")
        chunk.backward()
        total_loss += float(chunk.detach().cpu())
    gradients = {
        state_name: named[parameter_name].grad.detach().cpu().float().contiguous().clone()
        for parameter_name, state_name in zip(fs.TRAINABLE_ACTOR_PARAMETERS, STATE_PARAMETER_NAMES, strict=True)
    }
    direction, normalization = survival.normalized_negative_gradient_direction(gradients)
    lengths = active_mask.sum(dim=0).detach().cpu()
    term_sums: dict[str, float] = {}
    for name in recorder._snapshots[0]["causal_terms"]:  # noqa: SLF001
        values = torch.stack(  # noqa: SLF001
            [snapshot["causal_terms"][name] for snapshot in recorder._snapshots]
        )
        term_sums[name] = float(values[active_mask].sum().detach().cpu())
    return direction, {
        "collection_steps": COLLECTION_STEPS,
        "num_envs": NUM_ENVS,
        "total_transitions": TOTAL_TRANSITIONS,
        "first_episode_active_transition_count": active_count,
        "autoreset_transition_count_excluded": TOTAL_TRANSITIONS - active_count,
        "minimum_first_episode_length": int(lengths.min()),
        "median_first_episode_length": float(lengths.median()),
        "maximum_first_episode_length": int(lengths.max()),
        "replicate_a_survivors_at_q295": int(torch.count_nonzero(active_mask[286, :64]).cpu()),
        "replicate_b_survivors_at_q295": int(torch.count_nonzero(active_mask[286, 64:]).cpu()),
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


def run(repository_root: Path, requested_run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("shifted-base causal preflight failed")
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
        }
    )
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment

    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    random.seed(fs.FIXED_SEED)
    np.random.seed(fs.FIXED_SEED % (2**32))
    torch.manual_seed(fs.FIXED_SEED)
    torch.cuda.manual_seed_all(fs.FIXED_SEED)
    contract, baseline, overlay, topology, motion = _inputs(root)
    run_dir = full_support._create_run_dir_exclusive(requested_run_dir)  # noqa: SLF001
    full_support._write_json_exclusive(run_dir / "preflight.json", audit)  # noqa: SLF001
    full_support._write_json_exclusive(  # noqa: SLF001
        run_dir / "material_manifest.json", audit["material_manifest"]
    )
    cfg = task_space.make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=NUM_ENVS)
    cfg.seed = fs.FIXED_SEED
    task_audit = task_space.audit_task_space_ppo_env_cfg(cfg, expected_num_envs=NUM_ENVS)
    env = ManagerBasedRlEnv(cfg=cfg, device=survival.DEVICE)
    actor = disturbance._actor(baseline, topology)  # noqa: SLF001
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        prime = prime_sonic_true23_training_environment(wrapped)
        _materials_unchanged(root, audit, "before_collection")
        direction, evidence = _collect_and_score(actor, wrapped)
    finally:
        env.close()
    path = run_dir / "checkpoints" / DIRECTION_FILENAME
    disturbance._write_checkpoint(  # noqa: SLF001
        path,
        {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_shifted_base_causal_recovery_direction_v1",
            "contract_sha256": CONTRACT_SHA256,
            "source_overlay": overlay,
            "initial_policy_state_sha256": contract["initial_policy"]["policy_state_sha256"],
            "direction_state_dict": direction,
            "direction_state_sha256": evidence["direction_state_sha256"],
            "direction_l2": evidence["target_direction_l2"],
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        },
    )
    result = {
        "schema_version": 1,
        "kind": "g1_true23_sonic_rank256_shifted_base_causal_recovery_score_result_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "task_audit": task_audit,
        "prime": prime,
        "source_overlay": overlay,
        "gradient_evidence": evidence,
        "direction_artifact": {"path": str(path), "sha256": sha256_file(path)},
        "training_transitions": TOTAL_TRANSITIONS,
        "gradient_computations": 1,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "candidate": None,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    _materials_unchanged(root, audit, "before_result")
    full_support._write_json_exclusive(run_dir / RESULT_FILENAME, result)  # noqa: SLF001
    return result


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
        run_dir = args.run_dir.expanduser().resolve(strict=False)
        if run_dir.is_dir() and not run_dir.is_symlink() and not (run_dir / FAILURE_FILENAME).exists():
            full_support._write_json_exclusive(  # noqa: SLF001
                run_dir / FAILURE_FILENAME,
                {
                    "schema_version": 1,
                    "kind": "g1_true23_sonic_rank256_shifted_base_causal_recovery_score_failure_v1",
                    "contract_sha256": CONTRACT_SHA256,
                    "error_type": type(error).__name__,
                    "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                    "candidate": None,
                    "support_qualified": False,
                    "deployment_ready": False,
                    "hardware_authorized": False,
                },
            )
        print(f"shifted-base causal recovery failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result["gradient_evidence"], indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
