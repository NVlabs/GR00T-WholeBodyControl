"""Collect 360-step on-policy trajectories and optimize survival length directly."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment
from gear_sonic.scripts import train_g1_true23_sonic_task_space_ppo_full_support as parent
from gear_sonic.scripts.train_g1_true23_sonic_survival_score_line_search import (
    _evaluate_state as _evaluate_parent_state,
)
from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as fs
from gear_sonic.trl.mjlab.sonic_task_space_ppo_runner import (
    audit_task_space_ppo_env_cfg,
    make_task_space_ppo_env_cfg,
)
from gear_sonic.utils import g1_true23_sonic_task_space_ppo_full_support_delta_line_search as delta_trace
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_true23_sonic_survival_score_line_search import (
    STATE_PARAMETER_NAMES,
    TARGET_DIRECTION_L2,
    ZERO_COUNTS,
    normalized_negative_gradient_direction,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path("gear_sonic/config/sim_validation/g1_true23_sonic_survival_length_score_v1.json")
CONTRACT_SHA256 = "ea866d6376cb393646e94d5b18c87f7524c416e7171d83fa80d3238d7a709d38"
SOURCE_CHECKPOINT_PATH = Path(
    "/root/g1_true23_runs/sonic_survival_scale_extension_v1/checkpoints/sonic_survival_scale_extension_candidate.pt"
)
SOURCE_CHECKPOINT_SHA256 = "7c49c951b0a11536d12a68ab463f61111c5f8f483dc5fe820d6ec22137830613"
SOURCE_RESULT_PATH = Path(
    "/root/g1_true23_runs/sonic_survival_scale_extension_v1/survival_scale_extension_result.json"
)
SOURCE_RESULT_SHA256 = "8a3b47d417d3129144284c1d2377edca5e1b86f8a824c6abf038931c63edaeac"
SOURCE_POLICY_SHA256 = "a808df03d0badadf61d0a144aa016d2ee27a15c3c87661ade0c6465be74f3a9f"
DEVICE = "cuda:0"
NUM_ENVS = 128
COLLECTION_STEPS = 360
TOTAL_TRANSITIONS = NUM_ENVS * COLLECTION_STEPS
GRADIENT_BATCH_SIZE = 512
SCALES = (-0.5, 0.0, 0.25, 0.5, 1.0, 2.0)
RESULT_FILENAME = "survival_length_result.json"
FAILURE_FILENAME = "survival_length_failure.json"
CHECKPOINT_FILENAME = "sonic_survival_length_candidate.pt"


def _evaluate_state(**kwargs: Any) -> dict[str, Any]:
    """Evaluate every late-region probe as a changed actor, including local scale zero."""

    return _evaluate_parent_state(**kwargs, evaluation_update_count=1)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _strict_file(path: Path, expected: str, context: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise ValueError(f"{context} must be regular file")
    resolved = lexical.resolve(strict=True)
    if sha256_file(resolved) != expected:
        raise ValueError(f"{context} SHA256 mismatch")
    return resolved


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("survival-length contract mismatch")
    contract = fs._strict_json(path, "survival-length contract")
    if (
        contract.get("kind") != "g1_true23_sonic_survival_length_score_contract_v1"
        or contract.get("seed") != fs.FIXED_SEED
        or contract.get("collection", {}).get("num_steps") != COLLECTION_STEPS
        or contract.get("collection", {}).get("total_transitions") != TOTAL_TRANSITIONS
        or contract.get("gradient", {}).get("parameters") != list(fs.TRAINABLE_ACTOR_PARAMETERS)
        or contract.get("gradient", {}).get("batch_size") != GRADIENT_BATCH_SIZE
        or contract.get("gradient", {}).get("batches") != TOTAL_TRANSITIONS // GRADIENT_BATCH_SIZE
        or contract.get("gradient", {}).get("direction_l2_target") != TARGET_DIRECTION_L2
        or contract.get("gradient", {}).get("scales") != list(SCALES)
        or contract.get("evaluation_gate", {}).get("required_zero_counts") != list(ZERO_COUNTS)
    ):
        raise ValueError("survival-length contract semantic mismatch")
    return contract


def _sources(root: Path) -> dict[str, str]:
    relatives = (
        CONTRACT_RELATIVE_PATH,
        Path("gear_sonic/scripts/train_g1_true23_sonic_survival_length_score.py"),
    )
    return {relative.as_posix(): sha256_file((root / relative).resolve(strict=True)) for relative in relatives}


def preflight(repository_root: Path) -> dict[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract = _load_contract(root)
        parent_audit = parent.preflight(root)
        if parent_audit.get("ready") is not True:
            raise RuntimeError("survival-length parent preflight not ready")
        checkpoint = _strict_file(
            SOURCE_CHECKPOINT_PATH, SOURCE_CHECKPOINT_SHA256, "survival-length source checkpoint"
        )
        result = _strict_file(SOURCE_RESULT_PATH, SOURCE_RESULT_SHA256, "survival-length source result")
        material = {
            "parent_material_manifest_sha256": parent_audit["material_manifest"]["material_manifest_sha256"],
            "source_checkpoint": {"path": str(checkpoint), "sha256": SOURCE_CHECKPOINT_SHA256},
            "source_result": {"path": str(result), "sha256": SOURCE_RESULT_SHA256},
            "sources": _sources(root),
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_survival_length_score_preflight_v1",
            "ready": True,
            "contract": contract,
            "contract_sha256": CONTRACT_SHA256,
            "parent_preflight": parent_audit,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha256(material),
            "simulator_constructed": False,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_survival_length_score_preflight_v1",
            "ready": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "simulator_constructed": False,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }


def _materials_unchanged(root: Path, audit: Mapping[str, Any], phase: str) -> None:
    current = preflight(root)
    if current.get("ready") is not True or current.get("material_manifest_sha256") != audit.get(
        "material_manifest_sha256"
    ):
        raise RuntimeError(f"survival-length materials changed at {phase}")


def _load_source_policy() -> dict[str, torch.Tensor]:
    checkpoint = torch.load(SOURCE_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    state = checkpoint.get("policy_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("survival-length source policy missing")
    policy = {name: value.detach().cpu().contiguous().clone() for name, value in state.items()}
    observed = inspect_true23_policy_state({"policy_state_dict": policy}, reference_profile=fs.REFERENCE_PROFILE)
    if observed != SOURCE_POLICY_SHA256 or checkpoint.get("policy_state_sha256") != SOURCE_POLICY_SHA256:
        raise ValueError("survival-length source policy identity mismatch")
    return policy


def _actor(state: Mapping[str, torch.Tensor], topology: Path) -> Any:
    from gear_sonic.trl.mjlab.true23_actor import True23SonicActorModel

    actor = True23SonicActorModel(
        {
            "tokenizer": torch.zeros((1, 268), dtype=torch.float32),
            "policy": torch.zeros((1, 930), dtype=torch.float32),
        },
        {"actor": ["tokenizer", "policy"]},
        "actor",
        fs.ACTION_DIM,
        warm_start_path=str(topology),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"},
        hidden_dims=(),
        activation="silu",
        obs_normalization=False,
    ).to(DEVICE)
    if delta_trace._load_live_policy(actor, state) != SOURCE_POLICY_SHA256:
        raise RuntimeError("survival-length live source actor mismatch")
    named = dict(actor.named_parameters())
    for name, parameter in named.items():
        parameter.requires_grad_(name in fs.TRAINABLE_ACTOR_PARAMETERS)
    return actor


def _materialize_360_reward_evidence(recorder: Any) -> dict[str, Any]:
    snapshots = recorder._snapshots
    if recorder._armed is not None or recorder._pending_snapshot is not None or len(snapshots) != COLLECTION_STEPS:
        raise RuntimeError("survival-length reward recorder incomplete")
    termination_counts = {name: 0 for name in recorder.termination_names}
    barrier_active_count = 0
    barrier_raw_sum = 0.0
    barrier_weighted_sum = 0.0
    worst_raw_sum = 0.0
    nonfinite_count = 0
    action_semantics_mismatch_count = 0
    raw_clip_required_count = 0
    for snapshot in snapshots:
        active = snapshot["active"]
        rates = snapshot["rates"]
        dones = snapshot["dones"].to(dtype=torch.bool)
        terminations = snapshot["terminations"]
        tensors = (
            snapshot["action"],
            snapshot["reward"],
            rates,
            *snapshot["action_chain"].values(),
        )
        if any(tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()) for tensor in tensors):
            nonfinite_count += 1
        if not torch.equal(dones, torch.stack(terminations, dim=0).any(dim=0)):
            raise RuntimeError("survival-length done/termination evidence mismatch")
        for name, values in zip(recorder.termination_names, terminations, strict=True):
            termination_counts[name] += int(torch.count_nonzero(values & active).detach().cpu().item())
        barrier_rate = rates[:, recorder.barrier_index]
        worst_rate = rates[:, recorder.worst_index]
        barrier_active_count += int(torch.count_nonzero(active & (barrier_rate < 0.0)).detach().cpu().item())
        barrier_raw_sum += float((barrier_rate[active] / recorder.barrier_weight).sum().detach().cpu().item())
        barrier_weighted_sum += float((barrier_rate[active] * recorder.dt).sum().detach().cpu().item())
        worst_raw_sum += float((worst_rate[active] / recorder.worst_weight).sum().detach().cpu().item())
        mismatch, clip_count = fs._action_chain_mismatch_count(snapshot["action"], snapshot["action_chain"])
        action_semantics_mismatch_count += mismatch
        raw_clip_required_count += clip_count
    return {
        "right_wrist_barrier_active_count": barrier_active_count,
        "right_wrist_barrier_raw_sum": barrier_raw_sum,
        "right_wrist_barrier_weighted_sum": barrier_weighted_sum,
        "worst_ee_raw_sum": worst_raw_sum,
        "ee_body_pos_terminal_count": termination_counts.get("ee_body_pos", 0),
        "termination_counts": termination_counts,
        "nonfinite_count": nonfinite_count,
        "action_semantics_mismatch_count": action_semantics_mismatch_count,
        "raw_clip_required_count": raw_clip_required_count,
    }


def collect_and_score(*, actor: Any, wrapped: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw_env = wrapped.unwrapped
    if int(raw_env.num_envs) != NUM_ENVS:
        raise ValueError("survival-length collection env count mismatch")
    observations = wrapped.get_observations().to(DEVICE)
    active = torch.ones(NUM_ENVS, dtype=torch.bool, device=DEVICE)
    tokenizer_rows: list[torch.Tensor] = []
    policy_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    done_rows: list[torch.Tensor] = []
    active_rows: list[torch.Tensor] = []
    q9_rows: list[torch.Tensor] = []
    recorder = fs._FullSupportRewardEvidenceRecorder(raw_env)
    raw_env.extras["log"] = {}
    try:
        with torch.no_grad():
            for _step in range(COLLECTION_STEPS):
                q9 = fs._vector_q9(raw_env)
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
                observations = observations.to(DEVICE)
                active = active & ~dones.to(dtype=torch.bool, device=DEVICE)
    finally:
        recorder.restore()
    reward_evidence = _materialize_360_reward_evidence(recorder)
    tokenizer = torch.stack(tokenizer_rows)
    policy = torch.stack(policy_rows)
    actions = torch.stack(action_rows)
    dones = torch.stack(done_rows)
    active_mask = torch.stack(active_rows)
    q9 = torch.stack(q9_rows)
    if (
        tokenizer.shape != (COLLECTION_STEPS, NUM_ENVS, 268)
        or policy.shape != (COLLECTION_STEPS, NUM_ENVS, 930)
        or actions.shape != (COLLECTION_STEPS, NUM_ENVS, fs.ACTION_DIM)
        or dones.shape != (COLLECTION_STEPS, NUM_ENVS)
        or active_mask.shape != (COLLECTION_STEPS, NUM_ENVS)
        or not bool(torch.isfinite(tokenizer).all())
        or not bool(torch.isfinite(policy).all())
        or not bool(torch.isfinite(actions).all())
    ):
        raise RuntimeError("survival-length collected tensor mismatch")
    expected_q9 = torch.arange(9, 9 + COLLECTION_STEPS, device=DEVICE).unsqueeze(1)
    q9_mismatch = int(torch.count_nonzero(active_mask & (q9 != expected_q9)).detach().cpu().item())
    if q9_mismatch or any(
        reward_evidence.get(name) != 0
        for name in ("nonfinite_count", "action_semantics_mismatch_count", "raw_clip_required_count")
    ):
        raise RuntimeError("survival-length collection safety gate failed")
    lengths = active_mask.sum(dim=0).to(dtype=torch.float32)
    outcomes = lengths / float(COLLECTION_STEPS)
    outcome_mean = outcomes.mean()
    outcome_std = outcomes.std(unbiased=False)
    if not bool(torch.isfinite(outcome_std)) or float(outcome_std.detach().cpu().item()) < 0.01:
        raise RuntimeError("survival-length outcome variance is insufficient")
    centered_scaled = (outcomes - outcome_mean) / outcome_std
    weights = active_mask.to(dtype=torch.float32) * centered_scaled.unsqueeze(0)
    active_count = int(torch.count_nonzero(active_mask).detach().cpu().item())
    named = dict(actor.named_parameters())
    for parameter in actor.parameters():
        parameter.grad = None
    flat_tokenizer = tokenizer.flatten(0, 1)
    flat_policy = policy.flatten(0, 1)
    flat_actions = actions.flatten(0, 1)
    flat_weights = weights.flatten()
    total_loss = 0.0
    for start in range(0, TOTAL_TRANSITIONS, GRADIENT_BATCH_SIZE):
        stop = start + GRADIENT_BATCH_SIZE
        actor(
            {"tokenizer": flat_tokenizer[start:stop], "policy": flat_policy[start:stop]},
            stochastic_output=True,
        )
        log_prob = actor.get_output_log_prob(flat_actions[start:stop]).reshape(-1)
        chunk = -(log_prob * flat_weights[start:stop]).sum() / float(active_count)
        if not bool(torch.isfinite(chunk)):
            raise RuntimeError("survival-length gradient loss nonfinite")
        chunk.backward()
        total_loss += float(chunk.detach().cpu().item())
    gradients = {
        state_name: named[parameter_name].grad.detach().cpu().float().contiguous().clone()
        for parameter_name, state_name in zip(fs.TRAINABLE_ACTOR_PARAMETERS, STATE_PARAMETER_NAMES, strict=True)
    }
    direction, normalization = normalized_negative_gradient_direction(gradients)
    length_cpu = lengths.detach().cpu()
    evidence = {
        "collection_steps": COLLECTION_STEPS,
        "num_envs": NUM_ENVS,
        "total_transitions": TOTAL_TRANSITIONS,
        "first_episode_active_transition_count": active_count,
        "autoreset_transition_count_excluded": TOTAL_TRANSITIONS - active_count,
        "minimum_first_episode_length": int(length_cpu.min().item()),
        "median_first_episode_length": float(length_cpu.median().item()),
        "maximum_first_episode_length": int(length_cpu.max().item()),
        "survived_all_360_count": int(torch.count_nonzero(length_cpu == COLLECTION_STEPS).item()),
        "outcome_mean": float(outcome_mean.detach().cpu().item()),
        "outcome_population_std": float(outcome_std.detach().cpu().item()),
        "loss": total_loss,
        "gradient_state_sha256": fs._state_sha256(gradients),
        "direction_state_sha256": fs._state_sha256(direction),
        "reward_evidence": reward_evidence,
        **normalization,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "teacher_labels_used": False,
    }
    return direction, evidence


def construct_states(
    baseline: Mapping[str, torch.Tensor], direction: Mapping[str, torch.Tensor]
) -> dict[float, dict[str, torch.Tensor]]:
    states: dict[float, dict[str, torch.Tensor]] = {}
    for scale in SCALES:
        state = {name: value.detach().cpu().contiguous().clone() for name, value in baseline.items()}
        for name in STATE_PARAMETER_NAMES:
            state[name] = torch.add(state[name], direction[name], alpha=scale).contiguous()
        states[scale] = state
    return states


def assess(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if [record.get("scale") for record in records] != list(SCALES):
        raise ValueError("survival-length evaluation order mismatch")
    baseline = records[list(SCALES).index(0.0)]
    baseline_ok = (
        baseline.get("completed_transitions") == 358
        and baseline.get("terminal_q9") == 366
        and baseline.get("termination_names") == ["anchor_pos"]
        and all(baseline.get(name) == 0 for name in ZERO_COUNTS)
    )
    eligible = [
        record
        for record in records
        if record.get("scale") != 0.0
        and record.get("completed_transitions", -1) >= 359
        and record.get("terminal_q9", -1) >= 367
        and record.get("termination_names") in (["ee_body_pos"], ["anchor_pos"], ["anchor_ori"], ["time_out"])
        and all(record.get(name) == 0 for name in ZERO_COUNTS)
        and not isinstance(record.get("episode_return"), bool)
        and isinstance(record.get("episode_return"), (int, float))
        and math.isfinite(float(record["episode_return"]))
    ]
    selected = (
        max(
            eligible,
            key=lambda record: (
                int(record["completed_transitions"]),
                float(record["episode_return"]),
                -abs(float(record["scale"])),
            ),
            default=None,
        )
        if baseline_ok
        else None
    )
    return {
        "baseline_passed": baseline_ok,
        "candidate_selected": selected is not None,
        "selected_scale": selected.get("scale") if selected is not None else None,
        "selected_policy_state_sha256": selected.get("policy_state_sha256") if selected is not None else None,
        "selected_completed_transitions": selected.get("completed_transitions") if selected is not None else None,
        "selected_terminal_q9": selected.get("terminal_q9") if selected is not None else None,
        "selected_termination_names": selected.get("termination_names") if selected is not None else None,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def _write_torch(path: Path, payload: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"survival-length checkpoint exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(dict(payload), temporary)
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run(repository_root: Path, requested_run_dir: Path) -> dict[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("survival-length preflight failed")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    random.seed(fs.FIXED_SEED)
    np.random.seed(fs.FIXED_SEED % (2**32))
    torch.manual_seed(fs.FIXED_SEED)
    torch.cuda.manual_seed_all(fs.FIXED_SEED)
    base_contract = fs.load_task_space_ppo_contract(root)
    topology = (root / base_contract["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(
        strict=True
    )
    motion = (root / base_contract["environment"]["motion_relative_path"]).resolve(strict=True)
    baseline = _load_source_policy()
    actor = _actor(baseline, topology)
    run_dir = parent._create_run_dir_exclusive(requested_run_dir)
    parent._write_json_exclusive(run_dir / "preflight.json", audit)
    parent._write_json_exclusive(run_dir / "material_manifest.json", audit["material_manifest"])
    cfg = make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=NUM_ENVS)
    cfg.seed = fs.FIXED_SEED
    task_audit = audit_task_space_ppo_env_cfg(cfg, expected_num_envs=NUM_ENVS)
    env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        prime = prime_sonic_true23_training_environment(wrapped)
        _materials_unchanged(root, audit, "before_collection")
        direction, gradient_evidence = collect_and_score(actor=actor, wrapped=wrapped)
    finally:
        env.close()
    parent._write_json_exclusive(run_dir / "gradient_evidence.json", gradient_evidence)
    del actor
    torch.cuda.empty_cache()
    states = construct_states(baseline, direction)
    records: list[dict[str, Any]] = []
    for index, scale in enumerate(SCALES):
        _materials_unchanged(root, audit, f"before_evaluation_{index}")
        record = _evaluate_state(
            state=states[scale],
            scale=scale,
            topology=topology,
            motion=motion,
            material_manifest_sha256=audit["material_manifest_sha256"],
        )
        record["contract_sha256"] = CONTRACT_SHA256
        parent._write_json_exclusive(run_dir / "evaluations" / f"evaluation_{index}.json", record)
        records.append(record)
    assessment = assess(records)
    candidate: dict[str, str] | None = None
    if assessment["candidate_selected"] is True:
        scale = float(assessment["selected_scale"])
        checkpoint = {
            "schema_version": 1,
            "kind": "g1_true23_sonic_survival_length_candidate_v1",
            "contract_sha256": CONTRACT_SHA256,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "policy_state_dict": states[scale],
            "policy_state_sha256": assessment["selected_policy_state_sha256"],
            "selected_scale": scale,
            "gradient_evidence": gradient_evidence,
            "training_transitions": TOTAL_TRANSITIONS,
            "gradient_computations": 1,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        path = run_dir / "checkpoints" / CHECKPOINT_FILENAME
        _write_torch(path, checkpoint)
        candidate = {"path": str(path), "sha256": sha256_file(path)}
    result = {
        "schema_version": 1,
        "kind": "g1_true23_sonic_survival_length_score_result_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "task_audit": task_audit,
        "prime": prime,
        "gradient_evidence": gradient_evidence,
        "evaluations": records,
        "assessment": assessment,
        "candidate": candidate,
        "training_transitions": TOTAL_TRANSITIONS,
        "gradient_computations": 1,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    _materials_unchanged(root, audit, "before_result")
    parent._write_json_exclusive(run_dir / RESULT_FILENAME, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    train = sub.add_parser("train")
    train.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
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
        if run_dir.is_dir() and not run_dir.is_symlink():
            path = run_dir / FAILURE_FILENAME
            if not os.path.lexists(path):
                parent._write_json_exclusive(
                    path,
                    {
                        "schema_version": 1,
                        "kind": "g1_true23_sonic_survival_length_score_failure_v1",
                        "contract_sha256": CONTRACT_SHA256,
                        "error": {"type": type(error).__name__, "message": str(error)},
                        "candidate": None,
                        "support_qualified": False,
                        "promotion_eligible": False,
                        "deployment_ready": False,
                        "hardware_authorized": False,
                    },
                )
        print(f"survival-length experiment failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
