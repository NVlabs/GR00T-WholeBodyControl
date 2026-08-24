"""Collect one admitted rollout and line-search its direct survival-score gradient."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment
from gear_sonic.scripts import train_g1_true23_sonic_task_space_ppo_full_support as parent
from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as fs
from gear_sonic.trl.mjlab.sonic_task_space_ppo_runner import (
    audit_task_space_ppo_env_cfg,
    load_task_space_ppo_contract,
    make_task_space_ppo_env_cfg,
)
from gear_sonic.utils import g1_true23_sonic_task_space_ppo_full_support_delta_line_search as delta_trace
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_true23_sonic_survival_score_line_search import (
    CONTRACT_RELATIVE_PATH,
    CONTRACT_SHA256,
    GRADIENT_BATCH_SIZE,
    PPO_LINE_SEARCH_SHA256,
    QUARTER_RESULT_SHA256,
    SCALES,
    STATE_PARAMETER_NAMES,
    assess_survival_evaluations,
    candidate_checkpoint,
    construct_scaled_policy_state,
    first_episode_mask_and_success,
    load_survival_contract,
    normalized_negative_gradient_direction,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda:0"
PPO_LINE_SEARCH_PATH = Path(
    "/root/g1_true23_runs/sonic_task_space_ppo_full_support_delta_line_search_v1_retry1.json"
)
QUARTER_RESULT_PATH = Path(
    "/root/g1_true23_runs/sonic_task_space_ppo_quarter_step_v1_seed20260805_retry1/quarter_step_result.json"
)
RESULT_FILENAME = "survival_score_result.json"
FAILURE_FILENAME = "survival_score_failure.json"
CHECKPOINT_FILENAME = "sonic_survival_score_candidate.pt"


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


def _supplemental_sources(root: Path) -> dict[str, str]:
    relatives = (
        CONTRACT_RELATIVE_PATH,
        Path("gear_sonic/utils/g1_true23_sonic_survival_score_line_search.py"),
        Path("gear_sonic/scripts/train_g1_true23_sonic_survival_score_line_search.py"),
    )
    result: dict[str, str] = {}
    for relative in relatives:
        path = (root / relative).resolve(strict=True)
        if path.is_symlink() or path.parent != (root / relative.parent).resolve(strict=True):
            raise ValueError(f"survival-score source path drift: {relative}")
        result[relative.as_posix()] = sha256_file(path)
    return result


def preflight(repository_root: Path) -> dict[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract = load_survival_contract(root)
        parent_audit = parent.preflight(root)
        if parent_audit.get("ready") is not True:
            raise RuntimeError("survival-score parent preflight not ready")
        ppo = _strict_file(PPO_LINE_SEARCH_PATH, PPO_LINE_SEARCH_SHA256, "survival-score PPO trace")
        quarter = _strict_file(QUARTER_RESULT_PATH, QUARTER_RESULT_SHA256, "survival-score quarter result")
        material = {
            "parent_material_manifest_sha256": parent_audit["material_manifest"]["material_manifest_sha256"],
            "negative_evidence": {
                "ppo_line_search": {"path": str(ppo), "sha256": PPO_LINE_SEARCH_SHA256},
                "quarter_result": {"path": str(quarter), "sha256": QUARTER_RESULT_SHA256},
            },
            "supplemental_sources": _supplemental_sources(root),
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_survival_score_line_search_preflight_v1",
            "ready": True,
            "contract_sha256": CONTRACT_SHA256,
            "contract": contract,
            "parent_preflight": parent_audit,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha256(material),
            "simulator_constructed": False,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_survival_score_line_search_preflight_v1",
            "ready": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "simulator_constructed": False,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
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
        raise RuntimeError(f"survival-score materials changed at {phase}")


def compute_survival_score_direction(
    runner: fs.SonicTaskSpacePpoFullSupportRunner,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if runner._phase != "rollout_ready" or runner._optimizer_step_count != 0 or runner.alg.optimizer.state:
        raise RuntimeError("survival-score gradient requires admitted pre-Adam rollout")
    storage = runner.alg.storage
    active, success = first_episode_mask_and_success(storage.dones)
    success_float = success.to(dtype=torch.float32)
    centered = success_float - success_float.mean()
    weights = active.to(dtype=torch.float32) * centered.unsqueeze(0)
    active_count = int(torch.count_nonzero(active).detach().cpu().item())
    success_count = int(torch.count_nonzero(success).detach().cpu().item())
    if active_count != runner._rollout_evidence["first_episode_transition_count"]:
        raise RuntimeError("survival-score active transition count differs from admitted evidence")
    if success_count < 16 or success_count >= fs.NUM_ENVS:
        raise RuntimeError("survival-score success cohort is degenerate")

    actor = runner.alg.get_policy()
    named = dict(actor.named_parameters())
    if set(fs.TRAINABLE_ACTOR_PARAMETERS) - set(named):
        raise RuntimeError("survival-score actor trainable namespace mismatch")
    for parameter in actor.parameters():
        parameter.grad = None
    observations = storage.observations.flatten(0, 1)
    actions = storage.actions.flatten(0, 1)
    flat_weights = weights.flatten()
    total = actions.shape[0]
    if total != fs.TRAINING_TRANSITIONS or total % GRADIENT_BATCH_SIZE:
        raise RuntimeError("survival-score flat storage size mismatch")
    total_loss = 0.0
    for start in range(0, total, GRADIENT_BATCH_SIZE):
        stop = start + GRADIENT_BATCH_SIZE
        actor(observations[start:stop], stochastic_output=True)
        log_prob = actor.get_output_log_prob(actions[start:stop]).reshape(-1)
        if log_prob.shape != (GRADIENT_BATCH_SIZE,) or not bool(torch.isfinite(log_prob).all()):
            raise RuntimeError("survival-score action log probability mismatch")
        chunk = -(log_prob * flat_weights[start:stop]).sum() / float(active_count)
        if not bool(torch.isfinite(chunk)):
            raise RuntimeError("survival-score loss is nonfinite")
        chunk.backward()
        total_loss += float(chunk.detach().cpu().item())
    gradients: dict[str, torch.Tensor] = {}
    for parameter_name, state_name in zip(fs.TRAINABLE_ACTOR_PARAMETERS, STATE_PARAMETER_NAMES, strict=True):
        gradient = named[parameter_name].grad
        if gradient is None:
            raise RuntimeError(f"survival-score gradient missing: {parameter_name}")
        gradients[state_name] = gradient.detach().cpu().float().contiguous().clone()
    direction, normalization = normalized_negative_gradient_direction(gradients)
    baseline_sha = inspect_true23_policy_state(
        {"policy_state_dict": runner._policy_state_adapter.state_dict()}, reference_profile=fs.REFERENCE_PROFILE
    )
    if baseline_sha != fs.INITIAL_OVERLAY_POLICY_STATE_SHA256:
        raise RuntimeError("survival-score gradient mutated baseline policy")
    evidence = {
        "outcome": "no_first_episode_done_through_all_160_actions",
        "num_envs": fs.NUM_ENVS,
        "success_count": success_count,
        "failure_count": fs.NUM_ENVS - success_count,
        "success_fraction": success_count / fs.NUM_ENVS,
        "first_episode_active_transition_count": active_count,
        "autoreset_transition_count_excluded": fs.TRAINING_TRANSITIONS - active_count,
        "centered_outcome_mean": float(centered.mean().detach().cpu().item()),
        "loss": total_loss,
        "gradient_state_sha256": fs._state_sha256(gradients),
        "direction_state_sha256": fs._state_sha256(direction),
        "gradient_batches": fs.TRAINING_TRANSITIONS // GRADIENT_BATCH_SIZE,
        "gradient_batch_size": GRADIENT_BATCH_SIZE,
        **normalization,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "teacher_labels_used": False,
    }
    return direction, evidence


def _evaluate_state(
    *,
    state: Mapping[str, torch.Tensor],
    scale: float,
    topology: Path,
    motion: Path,
    material_manifest_sha256: str,
    evaluation_update_count: int | None = None,
) -> dict[str, Any]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

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
    policy_sha = delta_trace._load_live_policy(actor, state)
    with parent._preserved_rng_for_evaluation():
        cfg = make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=1)
        cfg.seed = fs.FIXED_SEED
        task_audit = audit_task_space_ppo_env_cfg(cfg, expected_num_envs=1)
        env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
            prime = prime_sonic_true23_training_environment(wrapped)
            update_count = (
                (0 if scale == 0.0 else 1) if evaluation_update_count is None else evaluation_update_count
            )
            if update_count not in fs.EVALUATION_UPDATES:
                raise ValueError("survival-score evaluation update count mismatch")
            record = fs.evaluate_full_support_policy(
                policy=actor,
                wrapped_env=wrapped,
                update_count=update_count,
                evaluation_seed=fs.FIXED_SEED,
            )
        finally:
            env.close()
    after = inspect_true23_policy_state(
        {"policy_state_dict": actor.export_true23_policy_state()}, reference_profile=fs.REFERENCE_PROFILE
    )
    if after != policy_sha:
        raise RuntimeError("survival-score policy changed during evaluation")
    return {
        **record,
        "scale": scale,
        "policy_state_sha256": policy_sha,
        "task_audit": task_audit,
        "prime": prime,
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": material_manifest_sha256,
    }


def _write_torch_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"survival-score checkpoint exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(dict(payload), temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        if not isinstance(loaded, Mapping) or loaded.get("contract_sha256") != CONTRACT_SHA256:
            raise ValueError("survival-score checkpoint round-trip mismatch")
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_experiment(repository_root: Path, requested_run_dir: Path) -> dict[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("survival-score preflight failed:\n" + json.dumps(audit, indent=2, sort_keys=True))
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("survival-score experiment requires WORLD_SIZE=1")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("survival-score experiment requires one CUDA device")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    random.seed(fs.FIXED_SEED)
    np.random.seed(fs.FIXED_SEED % (2**32))
    torch.manual_seed(fs.FIXED_SEED)
    torch.cuda.manual_seed_all(fs.FIXED_SEED)

    base_contract = load_task_space_ppo_contract(root)
    full_contract = fs.load_full_support_contract(root)
    actor_contract = base_contract["actor_initialization"]
    environment_contract = base_contract["environment"]
    source = Path(actor_contract["source_checkpoint_linux_path"]).resolve(strict=True)
    topology = (root / actor_contract["topology_checkpoint_relative_path"]).resolve(strict=True)
    encoder = (root / actor_contract["encoder_onnx_relative_path"]).resolve(strict=True)
    decoder = (root / actor_contract["v2_decoder_relative_path"]).resolve(strict=True)
    motion = (root / environment_contract["motion_relative_path"]).resolve(strict=True)
    run_dir = parent._create_run_dir_exclusive(requested_run_dir)
    parent._write_json_exclusive(run_dir / "preflight.json", audit)
    parent._write_json_exclusive(run_dir / "material_manifest.json", audit["material_manifest"])
    parent_materials = audit["parent_preflight"]["material_manifest"]
    agent_cfg = parent._agent_config(topology)
    engine_resolved = parent._resolved_config(agent_cfg, parent_materials)
    parent._write_json_exclusive(run_dir / "resolved_collection.json", engine_resolved)
    env_cfg = make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=fs.NUM_ENVS)
    env_cfg.seed = fs.FIXED_SEED
    task_audit = audit_task_space_ppo_env_cfg(env_cfg, expected_num_envs=fs.NUM_ENVS)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        prime = prime_sonic_true23_training_environment(wrapped)
        parent._write_json_exclusive(run_dir / "environment_prime.json", prime)
        torch.manual_seed(fs.FIXED_SEED)
        torch.cuda.manual_seed_all(fs.FIXED_SEED)
        engine_dir = run_dir / "engine"
        engine_dir.mkdir()
        (engine_dir / "checkpoints").mkdir()
        runner = fs.SonicTaskSpacePpoFullSupportRunner(
            wrapped,
            asdict(agent_cfg),
            str(engine_dir),
            DEVICE,
            warm_start_checkpoint_path=topology,
            resolved_config=engine_resolved,
            source_manifest=parent_materials["base_task_space_materials"]["source_files"],
            asset_manifest=parent_materials["base_task_space_materials"]["robot_assets"],
            dataset_manifest=parent_materials["base_task_space_materials"]["bound_inputs"],
            checkpoint_dir=engine_dir / "checkpoints",
            source_actor_checkpoint_path=source,
            overlay_encoder_path=encoder,
            overlay_decoder_path=decoder,
            base_task_space_contract=base_contract,
            full_support_contract=full_contract,
            run_materials_sha256=audit["material_manifest_sha256"],
        )
        _materials_unchanged(root, audit, "before_rollout")
        rollout = runner.collect_full_support_rollout()
        if rollout["coverage_assessment"]["gate_passed"] is not True:
            raise RuntimeError("survival-score pre-Adam full-support gate failed")
        _materials_unchanged(root, audit, "before_gradient")
        direction, gradient_evidence = compute_survival_score_direction(runner)
        parent._write_json_exclusive(run_dir / "gradient_evidence.json", gradient_evidence)
        baseline_state = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in runner._policy_state_adapter.state_dict().items()
        }
        states = {scale: construct_scaled_policy_state(baseline_state, direction, scale) for scale in SCALES}
        evaluations: list[dict[str, Any]] = []
        for index, scale in enumerate(SCALES):
            _materials_unchanged(root, audit, f"before_evaluation_{index}")
            record = _evaluate_state(
                state=states[scale],
                scale=scale,
                topology=topology,
                motion=motion,
                material_manifest_sha256=audit["material_manifest_sha256"],
            )
            parent._write_json_exclusive(run_dir / "evaluations" / f"evaluation_scale_{index}.json", record)
            evaluations.append(record)
        assessment = assess_survival_evaluations(evaluations)
        candidate: dict[str, str] | None = None
        if assessment["candidate_selected"] is True:
            selected_scale = float(assessment["selected_scale"])
            checkpoint = candidate_checkpoint(
                state=states[selected_scale],
                assessment=assessment,
                gradient_evidence=gradient_evidence,
                rollout_evidence_sha256=runner._rollout_evidence_sha256,
                material_manifest_sha256=audit["material_manifest_sha256"],
            )
            checkpoint_path = run_dir / "checkpoints" / CHECKPOINT_FILENAME
            _write_torch_exclusive(checkpoint_path, checkpoint)
            candidate = {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)}
        result = {
            "schema_version": 1,
            "kind": "g1_true23_sonic_survival_score_line_search_result_v1",
            "contract_sha256": CONTRACT_SHA256,
            "material_manifest_sha256": audit["material_manifest_sha256"],
            "run_dir": str(run_dir),
            "task_audit": task_audit,
            "prime": prime,
            "rollout_evidence_sha256": runner._rollout_evidence_sha256,
            "gradient_evidence": gradient_evidence,
            "evaluations": evaluations,
            "assessment": assessment,
            "candidate": candidate,
            "training_transitions": fs.TRAINING_TRANSITIONS,
            "gradient_computations": 1,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "teacher_labels_used": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        _materials_unchanged(root, audit, "before_result_publication")
        parent._write_json_exclusive(run_dir / RESULT_FILENAME, result)
        return result
    finally:
        env.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    exp = sub.add_parser("experiment")
    exp.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    exp.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight(args.repository_root)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    try:
        result = run_experiment(args.repository_root, args.run_dir)
    except Exception as error:
        run_dir = args.run_dir.expanduser().resolve(strict=False)
        if run_dir.is_dir() and not run_dir.is_symlink():
            path = run_dir / FAILURE_FILENAME
            if not os.path.lexists(path):
                parent._write_json_exclusive(
                    path,
                    {
                        "schema_version": 1,
                        "kind": "g1_true23_sonic_survival_score_line_search_failure_v1",
                        "contract_sha256": CONTRACT_SHA256,
                        "error": {"type": type(error).__name__, "message": str(error)},
                        "candidate": None,
                        "support_qualified": False,
                        "promotion_eligible": False,
                        "deployment_ready": False,
                        "hardware_authorized": False,
                    },
                )
        print(f"survival-score experiment failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
