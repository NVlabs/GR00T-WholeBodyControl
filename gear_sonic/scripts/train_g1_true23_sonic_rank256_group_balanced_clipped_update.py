"""Collect one mixed rank256 rollout and apply one bounded clipped policy update."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from gear_sonic.scripts import (
    screen_g1_true23_sonic_rank256_constrained_recovery_mix as constrained,
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
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_group_balanced_clipped_update_v1.json"
)
CONTRACT_SHA256 = "9c5071e4f3348f783994ed02e3c8584ebcdf7df8ea27a9ac7234c45e1a19937e"
NUM_ENVS = 128
GROUP_SIZE = 64
COLLECTION_STEPS = 510
TOTAL_TRANSITIONS = NUM_ENVS * COLLECTION_STEPS
IMPULSE_TRANSITION = 241
GAMMA = 0.99
LEARNING_RATE = 1.0e-5
MINI_BATCHES = 4
CLIP_PARAM = 0.1
MAX_GRAD_NORM = 0.25
OPTIMIZER_STEPS = 4
MIN_PUSH_AT_Q250 = 48
MIN_PUSH_AT_Q295 = 16
SHUFFLE_SEED = 20260813
RESULT_FILENAME = "rank256_group_balanced_clipped_update_result.json"
FAILURE_FILENAME = "rank256_group_balanced_clipped_update_failure.json"
DELTA_FILENAME = "rank256_group_balanced_clipped_update_delta.pt"
SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_group_balanced_clipped_update.py"),
    Path("gear_sonic/scripts/screen_g1_true23_sonic_rank256_constrained_recovery_mix.py"),
    Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_disturbance_survival_score.py"),
    Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_grouped_multiobjective_survival_score.py"),
    Path("gear_sonic/scripts/collect_g1_true23_sonic_rank256_causal_recovery_score.py"),
    Path("gear_sonic/scripts/collect_g1_true23_sonic_rank256_causal_recovery_score_v2.py"),
)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("group-balanced clipped-update contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    collection = body.get("collection", {})
    optimizer = body.get("optimizer", {})
    initial = body.get("initial_policy", {})
    parents = body.get("parents", {})
    boundaries = body.get("boundaries", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_group_balanced_clipped_update_contract_v1"
        or initial.get("policy_state_sha256") != "6acada92dd6e2700499c64255e077c7e844aa0f96b0700e2e2c5640606ae0650"
        or [initial.get(name) for name in ("reward_coefficient", "grouped_coefficient", "causal_coefficient")]
        != [1.0, 2.0, 0.0]
        or collection.get("num_envs") != NUM_ENVS
        or collection.get("steps") != COLLECTION_STEPS
        or collection.get("total_transitions") != TOTAL_TRANSITIONS
        or collection.get("impulse_global_transition") != IMPULSE_TRANSITION
        or collection.get("minimum_exact_impulse_survivors_at_q250") != MIN_PUSH_AT_Q250
        or collection.get("minimum_exact_impulse_survivors_at_q295") != MIN_PUSH_AT_Q295
        or optimizer.get("gamma") != GAMMA
        or optimizer.get("learning_rate") != LEARNING_RATE
        or optimizer.get("epochs") != 1
        or optimizer.get("mini_batches") != MINI_BATCHES
        or optimizer.get("optimizer_steps") != OPTIMIZER_STEPS
        or optimizer.get("clip_param") != CLIP_PARAM
        or optimizer.get("max_grad_norm") != MAX_GRAD_NORM
        or optimizer.get("critic_used") is not False
        or optimizer.get("updated_parameters") != list(fs.TRAINABLE_ACTOR_PARAMETERS)
        or boundaries.get("single_optimizer_iteration") is not True
        or boundaries.get("hardware_authorized") is not False
        or boundaries.get("robot_or_network_commands_permitted") is not False
    ):
        raise ValueError("group-balanced clipped-update contract semantic mismatch")
    checks = (
        (
            root / parents["constrained_contract_relative_path"],
            parents["constrained_contract_sha256"],
        ),
        (Path(parents["constrained_result_path"]), parents["constrained_result_sha256"]),
    )
    for raw, expected in checks:
        resolved = raw.expanduser().resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file() or sha256_file(resolved) != expected:
            raise ValueError("group-balanced clipped-update parent mismatch")
    parent_result = json.loads(Path(parents["constrained_result_path"]).read_text(encoding="utf-8"))
    if parent_result.get("assessment", {}).get("candidate_selected") is not False:
        raise ValueError("constrained parent must be negative")
    matching = [
        row
        for row in parent_result.get("evaluations", [])
        if row.get("grouped_coefficient") == 2.0 and row.get("causal_coefficient") == 0.0
    ]
    if (
        len(matching) != 2
        or {row.get("scenario") for row in matching} != {"nominal", "disturbance"}
        or {row.get("policy_state_sha256") for row in matching} != {initial["policy_state_sha256"]}
        or {row.get("scenario"): row.get("completed_transitions") for row in matching}
        != {"nominal": 505, "disturbance": 287}
    ):
        raise ValueError("group-balanced initial policy parent evidence mismatch")
    constrained._load_contract(root)  # noqa: SLF001
    return body


def _inputs(
    root: Path,
) -> tuple[Mapping[str, Any], dict[str, torch.Tensor], Mapping[str, Any], Path, Path]:
    contract = _load_contract(root)
    (
        _constrained_contract,
        _base,
        baseline,
        overlay,
        directions,
        topology,
        motion,
    ) = constrained._inputs(root)  # noqa: SLF001
    initial = constrained._state(baseline, directions, 2.0, 0.0)  # noqa: SLF001
    observed = inspect_true23_policy_state({"policy_state_dict": initial}, reference_profile=fs.REFERENCE_PROFILE)
    if observed != contract["initial_policy"]["policy_state_sha256"]:
        raise ValueError("group-balanced initial policy reconstruction mismatch")
    return contract, initial, overlay, topology, motion


def preflight(repository_root: Path) -> dict[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract, initial, overlay, topology, motion = _inputs(root)
        sources = {
            relative.as_posix(): sha256_file((root / relative).resolve(strict=True))
            for relative in SOURCE_RELATIVE_PATHS
        }
        material = {
            "contract_sha256": CONTRACT_SHA256,
            "parent_result_sha256": contract["parents"]["constrained_result_sha256"],
            "initial_policy_state_sha256": contract["initial_policy"]["policy_state_sha256"],
            "topology_sha256": sha256_file(topology),
            "motion_sha256": sha256_file(motion),
            "source_overlay": overlay,
            "sources": sources,
        }
        del initial
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_group_balanced_clipped_update_preflight_v1",
            "ready": True,
            "contract": contract,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha(material),
            "simulator_constructed": False,
            "training_transitions": 0,
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
            "kind": "g1_true23_sonic_rank256_group_balanced_clipped_update_preflight_v1",
            "ready": False,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "simulator_constructed": False,
            "training_transitions": 0,
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
        raise RuntimeError(f"group-balanced materials changed: {phase}")


def _mixed_impulse_vectors(device: torch.device) -> torch.Tensor:
    vectors = torch.zeros((NUM_ENVS, 6), dtype=torch.float32, device=device)
    exact = torch.from_numpy(disturbance.FAILED_VECTOR.copy()).to(device=device)
    vectors[GROUP_SIZE:] = exact
    return vectors


def _reward_to_go(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    if rewards.shape != dones.shape or rewards.shape != active.shape:
        raise ValueError("reward-to-go tensor shape mismatch")
    result = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1], dtype=rewards.dtype, device=rewards.device)
    for step in range(rewards.shape[0] - 1, -1, -1):
        live = active[step]
        continuation = (~dones[step]).to(dtype=rewards.dtype)
        running = torch.where(live, rewards[step] + GAMMA * running * continuation, 0.0)
        result[step] = running
    return result


def _standardized_group_advantages(
    returns: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    advantages = torch.zeros_like(returns)
    evidence: list[dict[str, Any]] = []
    for name, start, stop in (("nominal", 0, GROUP_SIZE), ("exact_impulse", GROUP_SIZE, NUM_ENVS)):
        mask = active[:, start:stop]
        values = returns[:, start:stop][mask]
        mean = values.mean()
        std = values.std(unbiased=False)
        if not bool(torch.isfinite(mean)) or not bool(torch.isfinite(std)) or float(std) <= 0.01:
            raise RuntimeError(f"group advantage variance insufficient: {name}")
        group = advantages[:, start:stop]
        group[mask] = (values - mean) / std
        evidence.append(
            {
                "name": name,
                "active_transition_count": int(values.numel()),
                "return_mean": float(mean.detach().cpu()),
                "return_population_std": float(std.detach().cpu()),
                "return_minimum": float(values.min().detach().cpu()),
                "return_maximum": float(values.max().detach().cpu()),
            }
        )
    return advantages, evidence


def _survivor_count(active: torch.Tensor, q9: int, start: int, stop: int) -> int:
    index = q9 - 9
    if not 0 <= index < active.shape[0]:
        raise ValueError("survivor q9 outside collection")
    return int(torch.count_nonzero(active[index, start:stop]).detach().cpu())


def _collect(
    actor: Any,
    wrapped: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw_env = wrapped.unwrapped
    observations = wrapped.get_observations().to(survival.DEVICE)
    active = torch.ones(NUM_ENVS, dtype=torch.bool, device=survival.DEVICE)
    tokenizer_rows: list[torch.Tensor] = []
    policy_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    log_prob_rows: list[torch.Tensor] = []
    reward_rows: list[torch.Tensor] = []
    done_rows: list[torch.Tensor] = []
    active_rows: list[torch.Tensor] = []
    q9_rows: list[torch.Tensor] = []
    recorder = fs._FullSupportRewardEvidenceRecorder(raw_env)  # noqa: SLF001
    vectors = _mixed_impulse_vectors(torch.device(survival.DEVICE))
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
                old_log_prob = actor.get_output_log_prob(actions).reshape(-1)
                if old_log_prob.shape != (NUM_ENVS,):
                    raise RuntimeError("group-balanced behavior log-prob shape mismatch")
                action_rows.append(actions.detach().clone())
                log_prob_rows.append(old_log_prob.detach().clone())
                recorder.arm(q9, active, actions)
                observations, rewards, dones, extras = proxy.step(actions)
                recorder.finish(dones)
                reward_rows.append(rewards.detach().clone())
                done_rows.append(dones.detach().to(dtype=torch.bool).clone())
                extras["log"] = {}
                observations = observations.to(survival.DEVICE)
                active = active & ~dones.to(dtype=torch.bool, device=survival.DEVICE)
    finally:
        recorder.restore()
    with disturbance._collection_scope():  # noqa: SLF001
        reward_evidence = survival._materialize_360_reward_evidence(recorder)  # noqa: SLF001
    tensors = {
        "tokenizer": torch.stack(tokenizer_rows),
        "policy": torch.stack(policy_rows),
        "actions": torch.stack(action_rows),
        "old_log_prob": torch.stack(log_prob_rows),
        "rewards": torch.stack(reward_rows),
        "dones": torch.stack(done_rows),
        "active": torch.stack(active_rows),
        "q9": torch.stack(q9_rows),
    }
    expected_shapes = {
        "tokenizer": (COLLECTION_STEPS, NUM_ENVS, 268),
        "policy": (COLLECTION_STEPS, NUM_ENVS, 930),
        "actions": (COLLECTION_STEPS, NUM_ENVS, fs.ACTION_DIM),
        "old_log_prob": (COLLECTION_STEPS, NUM_ENVS),
        "rewards": (COLLECTION_STEPS, NUM_ENVS),
        "dones": (COLLECTION_STEPS, NUM_ENVS),
        "active": (COLLECTION_STEPS, NUM_ENVS),
        "q9": (COLLECTION_STEPS, NUM_ENVS),
    }
    if any(tensors[name].shape != shape for name, shape in expected_shapes.items()):
        raise RuntimeError("group-balanced collected tensor mismatch")
    floating = ("tokenizer", "policy", "actions", "old_log_prob", "rewards")
    if any(not bool(torch.isfinite(tensors[name]).all()) for name in floating):
        raise RuntimeError("group-balanced collected tensor nonfinite")
    expected_q9 = torch.arange(9, 9 + COLLECTION_STEPS, device=survival.DEVICE).unsqueeze(1)
    q9_mismatch = int(torch.count_nonzero(tensors["active"] & (tensors["q9"] != expected_q9)).cpu())
    push_q250 = _survivor_count(tensors["active"], 250, GROUP_SIZE, NUM_ENVS)
    push_q295 = _survivor_count(tensors["active"], 295, GROUP_SIZE, NUM_ENVS)
    if (
        q9_mismatch
        or push_q250 < MIN_PUSH_AT_Q250
        or push_q295 < MIN_PUSH_AT_Q295
        or any(
            reward_evidence.get(name) != 0
            for name in ("nonfinite_count", "action_semantics_mismatch_count", "raw_clip_required_count")
        )
    ):
        raise RuntimeError("group-balanced collection gate failed")
    evidence = {
        "collection_steps": COLLECTION_STEPS,
        "num_envs": NUM_ENVS,
        "total_transitions": TOTAL_TRANSITIONS,
        "first_episode_active_transition_count": int(torch.count_nonzero(tensors["active"]).cpu()),
        "autoreset_transition_count_excluded": TOTAL_TRANSITIONS
        - int(torch.count_nonzero(tensors["active"]).cpu()),
        "q9_mismatch_count": q9_mismatch,
        "nominal_survivors_at_q486": _survivor_count(tensors["active"], 486, 0, GROUP_SIZE),
        "exact_impulse_survivors_at_q250": push_q250,
        "exact_impulse_survivors_at_q295": push_q295,
        "exact_impulse_survivors_at_q518": _survivor_count(tensors["active"], 518, GROUP_SIZE, NUM_ENVS),
        "reward_evidence": reward_evidence,
    }
    return tensors, evidence


def _optimize(
    actor: Any,
    tensors: Mapping[str, torch.Tensor],
    initial_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    returns = _reward_to_go(tensors["rewards"], tensors["dones"], tensors["active"])
    advantages, group_evidence = _standardized_group_advantages(returns, tensors["active"])
    flat = {
        name: value.flatten(0, 1)
        for name, value in tensors.items()
        if name in {"tokenizer", "policy", "actions", "old_log_prob", "active"}
    }
    flat_advantages = advantages.flatten()
    env_ids = torch.arange(NUM_ENVS, device=survival.DEVICE).repeat(COLLECTION_STEPS)
    nominal_indices = torch.nonzero(flat["active"] & (env_ids < GROUP_SIZE), as_tuple=False).reshape(-1)
    push_indices = torch.nonzero(flat["active"] & (env_ids >= GROUP_SIZE), as_tuple=False).reshape(-1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SHUFFLE_SEED)
    nominal_permutation = nominal_indices.detach().cpu()[
        torch.randperm(nominal_indices.numel(), generator=generator)
    ]
    push_permutation = push_indices.detach().cpu()[torch.randperm(push_indices.numel(), generator=generator)]
    nominal_chunks = torch.tensor_split(nominal_permutation, MINI_BATCHES)
    push_chunks = torch.tensor_split(push_permutation, MINI_BATCHES)
    named = dict(actor.named_parameters())
    trainable = [named[name] for name in fs.TRAINABLE_ACTOR_PARAMETERS]
    if any(not parameter.requires_grad for parameter in trainable):
        raise RuntimeError("group-balanced trainable parameter freeze mismatch")
    optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE)
    steps: list[dict[str, Any]] = []
    for index, (nominal_cpu, push_cpu) in enumerate(zip(nominal_chunks, push_chunks, strict=True), start=1):
        nominal = nominal_cpu.to(device=survival.DEVICE)
        push = push_cpu.to(device=survival.DEVICE)
        batch = torch.cat((nominal, push))
        actor(
            {"tokenizer": flat["tokenizer"][batch], "policy": flat["policy"][batch]},
            stochastic_output=True,
        )
        new_log_prob = actor.get_output_log_prob(flat["actions"][batch]).reshape(-1)
        old_log_prob = flat["old_log_prob"][batch]
        ratio = torch.exp(new_log_prob - old_log_prob)
        advantage = flat_advantages[batch]
        count_nominal = nominal.numel()
        ratio_nominal, ratio_push = ratio[:count_nominal], ratio[count_nominal:]
        advantage_nominal = advantage[:count_nominal]
        advantage_push = advantage[count_nominal:]

        def surrogate(group_ratio: torch.Tensor, group_advantage: torch.Tensor) -> torch.Tensor:
            unclipped = group_ratio * group_advantage
            clipped = torch.clamp(group_ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * group_advantage
            return torch.minimum(unclipped, clipped).mean()

        objective_nominal = surrogate(ratio_nominal, advantage_nominal)
        objective_push = surrogate(ratio_push, advantage_push)
        loss = -0.5 * (objective_nominal + objective_push)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("group-balanced clipped loss nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("group-balanced gradient norm nonfinite")
        optimizer.step()
        steps.append(
            {
                "step": index,
                "nominal_sample_count": int(nominal.numel()),
                "exact_impulse_sample_count": int(push.numel()),
                "loss": float(loss.detach().cpu()),
                "nominal_objective": float(objective_nominal.detach().cpu()),
                "exact_impulse_objective": float(objective_push.detach().cpu()),
                "approximate_kl": float((old_log_prob - new_log_prob).mean().detach().cpu()),
                "ratio_minimum": float(ratio.min().detach().cpu()),
                "ratio_maximum": float(ratio.max().detach().cpu()),
                "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
            }
        )
    if len(steps) != OPTIMIZER_STEPS or len(optimizer.state) != len(trainable):
        raise RuntimeError("group-balanced optimizer step accounting mismatch")
    updated = {name: value.detach().cpu().contiguous().clone() for name, value in initial_state.items()}
    for parameter_name, state_name in zip(fs.TRAINABLE_ACTOR_PARAMETERS, STATE_PARAMETER_NAMES, strict=True):
        updated[state_name] = named[parameter_name].detach().cpu().to(torch.float32).contiguous().clone()
    delta = {
        name: torch.sub(updated[name], initial_state[name]).to(torch.float32).contiguous()
        for name in STATE_PARAMETER_NAMES
    }
    for name in set(initial_state) - set(STATE_PARAMETER_NAMES):
        if not torch.equal(initial_state[name], updated[name]):
            raise RuntimeError(f"group-balanced frozen policy tensor changed: {name}")
    delta_l2 = math.sqrt(sum(float(torch.sum(value.double().square())) for value in delta.values()))
    delta_max = max(float(value.abs().max()) for value in delta.values())
    if not math.isfinite(delta_l2) or not 0.0 < delta_l2 < 0.1 or not math.isfinite(delta_max):
        raise RuntimeError("group-balanced update delta bound failed")
    initial_sha = inspect_true23_policy_state(
        {"policy_state_dict": initial_state}, reference_profile=fs.REFERENCE_PROFILE
    )
    updated_sha = inspect_true23_policy_state(
        {"policy_state_dict": updated}, reference_profile=fs.REFERENCE_PROFILE
    )
    if initial_sha == updated_sha:
        raise RuntimeError("group-balanced optimizer did not change actor")
    evidence = {
        "objective": "equal_group_clipped_importance_weighted_standardized_reward_to_go",
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "clip_param": CLIP_PARAM,
        "max_grad_norm": MAX_GRAD_NORM,
        "epochs": 1,
        "mini_batches": MINI_BATCHES,
        "optimizer_steps": OPTIMIZER_STEPS,
        "critic_used": False,
        "group_advantages": group_evidence,
        "steps": steps,
        "initial_policy_state_sha256": initial_sha,
        "updated_policy_state_sha256": updated_sha,
        "delta_state_sha256": fs._state_sha256(delta),  # noqa: SLF001
        "delta_l2": delta_l2,
        "delta_max_abs": delta_max,
        "updated_parameter_names": list(fs.TRAINABLE_ACTOR_PARAMETERS),
        "frozen_policy_tensor_count": len(initial_state) - len(STATE_PARAMETER_NAMES),
        "teacher_labels_used": False,
    }
    return updated, delta, evidence


def run(repository_root: Path, requested_run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("group-balanced clipped-update preflight failed")
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
    contract, initial, overlay, topology, motion = _inputs(root)
    run_dir = full_support._create_run_dir_exclusive(requested_run_dir)  # noqa: SLF001
    full_support._write_json_exclusive(run_dir / "preflight.json", audit)  # noqa: SLF001
    full_support._write_json_exclusive(  # noqa: SLF001
        run_dir / "material_manifest.json", audit["material_manifest"]
    )
    cfg = task_space.make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=NUM_ENVS)
    cfg.seed = fs.FIXED_SEED
    task_audit = task_space.audit_task_space_ppo_env_cfg(cfg, expected_num_envs=NUM_ENVS)
    env = ManagerBasedRlEnv(cfg=cfg, device=survival.DEVICE)
    actor = disturbance._actor(initial, topology)  # noqa: SLF001
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        prime = prime_sonic_true23_training_environment(wrapped)
        _materials_unchanged(root, audit, "before_collection")
        tensors, collection_evidence = _collect(actor, wrapped)
    finally:
        env.close()
    updated, delta, optimizer_evidence = _optimize(actor, tensors, initial)
    del tensors
    artifact_path = run_dir / "checkpoints" / DELTA_FILENAME
    disturbance._write_checkpoint(  # noqa: SLF001
        artifact_path,
        {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_group_balanced_clipped_update_delta_v1",
            "contract_sha256": CONTRACT_SHA256,
            "source_overlay": overlay,
            "initial_policy_state_sha256": optimizer_evidence["initial_policy_state_sha256"],
            "updated_policy_state_sha256": optimizer_evidence["updated_policy_state_sha256"],
            "delta_state_dict": delta,
            "delta_state_sha256": optimizer_evidence["delta_state_sha256"],
            "optimizer_steps": OPTIMIZER_STEPS,
            "critic_updates": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        },
    )
    artifact = {"path": str(artifact_path), "sha256": sha256_file(artifact_path)}
    result = {
        "schema_version": 1,
        "kind": "g1_true23_sonic_rank256_group_balanced_clipped_update_result_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "task_audit": task_audit,
        "prime": prime,
        "collection_evidence": collection_evidence,
        "optimizer_evidence": optimizer_evidence,
        "delta_artifact": artifact,
        "training_transitions": TOTAL_TRANSITIONS,
        "optimizer_steps": OPTIMIZER_STEPS,
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
    del updated
    return result


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
        run_dir = args.run_dir.expanduser().resolve(strict=False)
        if run_dir.is_dir() and not run_dir.is_symlink() and not (run_dir / FAILURE_FILENAME).exists():
            full_support._write_json_exclusive(  # noqa: SLF001
                run_dir / FAILURE_FILENAME,
                {
                    "schema_version": 1,
                    "kind": "g1_true23_sonic_rank256_group_balanced_clipped_update_failure_v1",
                    "contract_sha256": CONTRACT_SHA256,
                    "error_type": type(error).__name__,
                    "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                    "candidate": None,
                    "support_qualified": False,
                    "deployment_ready": False,
                    "hardware_authorized": False,
                },
            )
        print(f"group-balanced clipped update failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result["optimizer_evidence"], indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
