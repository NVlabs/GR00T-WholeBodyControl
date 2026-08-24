#!/usr/bin/env python3
"""Run one hash-bound 128x160 SONIC task-space PPO update in MJLab."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23 import (
    prime_sonic_true23_training_environment,
)
from gear_sonic.scripts.train_g1_true23_sonic_task_space_ppo import (
    _bind_local_runtime_sources,
    _material_manifests,
)
from gear_sonic.trl.mjlab.sonic_task_space_ppo_full_support_runner import (
    CONTRACT_SHA256,
    FIXED_SEED,
    MAXIMUM_UPDATES,
    NUM_ENVS,
    NUM_STEPS_PER_ENV,
    SonicTaskSpacePpoFullSupportRunner,
    audit_task_space_ppo_env_cfg,
    evaluate_full_support_policy,
    execute_full_support_schedule,
    load_full_support_contract,
    load_task_space_ppo_contract,
    make_task_space_ppo_env_cfg,
    validate_bound_full_support_evidence,
)
from gear_sonic.trl.mjlab.sonic_task_space_ppo_runner import (
    FIXED_LEARNING_RATE,
    PPO_LEARNING_EPOCHS,
    PPO_MINI_BATCHES,
    preflight_task_space_ppo,
)
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes, sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXED_GPU = 0
DEVICE = "cuda:0"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _create_run_dir_exclusive(path: Path) -> Path:
    output = path.expanduser().resolve(strict=False)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite full-support run: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    (output / "evaluations").mkdir()
    return output


def _bound_evidence_files(contract: Mapping[str, Any]) -> dict[str, Path]:
    diagnostic = contract["diagnostic_evidence"]
    parent = contract["failed_parent_pilot"]
    parent_dir = Path(parent["run_dir_linux_path"]).expanduser().resolve(strict=True)
    files = {"diagnostic/trace_v3.json": Path(diagnostic["trace_linux_path"]).expanduser().resolve(strict=True)}
    for relative in parent["artifact_sha256"]:
        files[f"failed_parent/{relative}"] = (parent_dir / relative).resolve(strict=True)
    return files


def _diagnostic_evidence_manifest(files: Mapping[str, Path]) -> dict[str, Any]:
    """Hash non-MJLab diagnostic inputs under stable logical names."""

    records = []
    for logical_path in sorted(files):
        path = files[logical_path]
        if not path.is_file():
            raise ValueError(f"diagnostic evidence is not a regular file: {path}")
        records.append(
            {
                "logical_path": logical_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise ValueError("diagnostic evidence manifest must not be empty")
    value = {
        "schema": "g1_true23_sonic_task_space_ppo_diagnostic_evidence_manifest_v1",
        "kind": "diagnostic_evidence",
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }
    value["manifest_sha256"] = _canonical_sha256(value)
    return value


def _full_support_materials(
    root: Path,
    base_contract: Mapping[str, Any],
    full_contract: Mapping[str, Any],
    source_checkpoint: Path,
) -> dict[str, Any]:
    validate_bound_full_support_evidence(full_contract)
    base = _material_manifests(root, base_contract, source_checkpoint)
    value = {
        "schema": "g1_true23_sonic_task_space_ppo_full_support_materials_v1",
        "contract_sha256": CONTRACT_SHA256,
        "base_task_space_materials": base,
        "diagnostic_evidence": _diagnostic_evidence_manifest(_bound_evidence_files(full_contract)),
    }
    value["material_manifest_sha256"] = _canonical_sha256(value)
    return value


def _require_materials_unchanged(
    root: Path,
    base_contract: Mapping[str, Any],
    full_contract: Mapping[str, Any],
    source_checkpoint: Path,
    expected: Mapping[str, Any],
    phase: str,
) -> None:
    actual = _full_support_materials(root, base_contract, full_contract, source_checkpoint)
    if actual != expected:
        raise RuntimeError(f"full-support materials changed at {phase}")


def _agent_config(topology_checkpoint: Path) -> Any:
    from gear_sonic.trl.mjlab.config import true23_mjlab_ppo_runner_cfg

    cfg = true23_mjlab_ppo_runner_cfg()
    cfg.actor.warm_start_path = str(topology_checkpoint)
    cfg.actor.distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": 0.1,
        "std_type": "scalar",
    }
    cfg.seed = FIXED_SEED
    cfg.num_steps_per_env = NUM_STEPS_PER_ENV
    cfg.max_iterations = MAXIMUM_UPDATES
    cfg.save_interval = 1
    cfg.clip_actions = None
    cfg.upload_model = False
    cfg.algorithm.learning_rate = FIXED_LEARNING_RATE
    cfg.algorithm.schedule = "fixed"
    cfg.algorithm.clip_param = 0.1
    cfg.algorithm.entropy_coef = 0.0
    cfg.algorithm.num_learning_epochs = PPO_LEARNING_EPOCHS
    cfg.algorithm.num_mini_batches = PPO_MINI_BATCHES
    cfg.algorithm.gamma = 0.99
    cfg.algorithm.lam = 0.95
    cfg.algorithm.desired_kl = None
    cfg.algorithm.max_grad_norm = 0.25
    return cfg


def _resolved_config(agent_cfg: Any, materials: Mapping[str, Any]) -> dict[str, Any]:
    agent = asdict(agent_cfg)
    for key in ("resume", "load_run", "load_checkpoint"):
        agent.pop(key, None)
    return {
        "schema": "g1_true23_sonic_task_space_ppo_full_support_resolved_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": materials["material_manifest_sha256"],
        "seed": FIXED_SEED,
        "gpu": FIXED_GPU,
        "num_envs": NUM_ENVS,
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "maximum_updates": MAXIMUM_UPDATES,
        "random_episode_length_initialization": False,
        "agent": agent,
        "failed_model5_loaded": False,
        "failed_model5_resumed": False,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def preflight(repository_root: Path) -> dict[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    try:
        base_contract = load_task_space_ppo_contract(root)
        full_contract = load_full_support_contract(root)
        base = preflight_task_space_ppo(root)
        evidence = validate_bound_full_support_evidence(full_contract)
        actor = base_contract["actor_initialization"]
        source = Path(actor["source_checkpoint_linux_path"]).resolve(strict=True)
        materials = _full_support_materials(root, base_contract, full_contract, source)
        return {
            **base,
            "base_task_space_parent_pilot": base["pilot"],
            "pilot": dict(full_contract["single_update"]),
            "schema_version": 1,
            "kind": "g1_true23_sonic_task_space_ppo_full_support_preflight_v1",
            "contract_sha256": CONTRACT_SHA256,
            "ready": True,
            "bound_diagnostic_evidence": evidence,
            "material_manifest": materials,
            "simulator_constructed": False,
            "training_updates": 0,
            "optimizer_steps": 0,
            "failed_model5_loaded": False,
            "failed_model5_resumed": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_task_space_ppo_full_support_preflight_v1",
            "contract_sha256": CONTRACT_SHA256,
            "ready": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "simulator_constructed": False,
            "training_updates": 0,
            "optimizer_steps": 0,
            "failed_model5_loaded": False,
            "failed_model5_resumed": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }


@contextmanager
def _preserved_rng_for_evaluation() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED % (2**32))
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_states)


def _evaluation_callback(*, motion: Path) -> Any:
    def evaluate(runner: SonicTaskSpacePpoFullSupportRunner, update: int) -> Mapping[str, Any]:
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper

        with _preserved_rng_for_evaluation():
            cfg = make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=1)
            cfg.seed = FIXED_SEED
            task_audit = audit_task_space_ppo_env_cfg(cfg, expected_num_envs=1)
            env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
            try:
                wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
                prime = prime_sonic_true23_training_environment(wrapped)
                record = evaluate_full_support_policy(
                    policy=runner.alg.get_policy(),
                    wrapped_env=wrapped,
                    update_count=update,
                    evaluation_seed=FIXED_SEED,
                )
            finally:
                env.close()
        return {
            **record,
            "task_audit": task_audit,
            "prime": prime,
            "contract_sha256": CONTRACT_SHA256,
            "material_manifest_sha256": runner._run_materials_sha256,
        }

    return evaluate


def run_experiment(repository_root: Path, requested_run_dir: Path) -> dict[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("full-support preflight failed:\n" + json.dumps(audit, indent=2, sort_keys=True))
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("full-support experiment requires WORLD_SIZE=1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FIXED_GPU)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    runtime_sources = _bind_local_runtime_sources(root)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("full-support experiment requires fixed visible CUDA device 0")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED % (2**32))
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)

    base_contract = load_task_space_ppo_contract(root)
    full_contract = load_full_support_contract(root)
    actor = base_contract["actor_initialization"]
    environment = base_contract["environment"]
    source = Path(actor["source_checkpoint_linux_path"]).resolve(strict=True)
    topology = (root / actor["topology_checkpoint_relative_path"]).resolve(strict=True)
    encoder = (root / actor["encoder_onnx_relative_path"]).resolve(strict=True)
    decoder = (root / actor["v2_decoder_relative_path"]).resolve(strict=True)
    motion = (root / environment["motion_relative_path"]).resolve(strict=True)
    materials = audit["material_manifest"]
    material_sha = materials["material_manifest_sha256"]
    run_dir = _create_run_dir_exclusive(requested_run_dir)
    _write_json_exclusive(run_dir / "preflight.json", audit)
    _write_json_exclusive(run_dir / "material_manifest.json", materials)

    agent_cfg = _agent_config(topology)
    resolved = _resolved_config(agent_cfg, materials)
    _write_json_exclusive(run_dir / "resolved_training.json", resolved)
    env_cfg = make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=NUM_ENVS)
    env_cfg.seed = FIXED_SEED
    task_audit = audit_task_space_ppo_env_cfg(env_cfg, expected_num_envs=NUM_ENVS)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        prime = prime_sonic_true23_training_environment(wrapped)
        _write_json_exclusive(run_dir / "environment_prime.json", prime)
        # Exact parent order: reset only Torch before actor/critic construction.
        torch.manual_seed(FIXED_SEED)
        torch.cuda.manual_seed_all(FIXED_SEED)
        train_cfg = asdict(agent_cfg)
        runner = SonicTaskSpacePpoFullSupportRunner(
            wrapped,
            train_cfg,
            str(run_dir),
            DEVICE,
            warm_start_checkpoint_path=topology,
            resolved_config=resolved,
            source_manifest=materials["base_task_space_materials"]["source_files"],
            asset_manifest=materials["base_task_space_materials"]["robot_assets"],
            dataset_manifest=materials["base_task_space_materials"]["bound_inputs"],
            checkpoint_dir=run_dir / "checkpoints",
            source_actor_checkpoint_path=source,
            overlay_encoder_path=encoder,
            overlay_decoder_path=decoder,
            base_task_space_contract=base_contract,
            full_support_contract=full_contract,
            run_materials_sha256=material_sha,
        )
        _write_json_exclusive(
            run_dir / "initialization.json",
            {
                "schema": "g1_true23_sonic_task_space_ppo_full_support_initialization_v1",
                "contract_sha256": CONTRACT_SHA256,
                "material_manifest_sha256": material_sha,
                "task_audit": task_audit,
                "runtime_sources": runtime_sources,
                "fresh_critic_state_sha256": runner._initial_critic_state_sha256,
                "optimizer_state_entry_count": len(runner.alg.optimizer.state),
                "completed_update_count": runner.completed_update_count,
                "optimizer_step_count": runner._optimizer_step_count,
                "executed_training_transitions": runner._executed_training_transitions,
                "failed_model5_loaded": False,
                "failed_model5_resumed": False,
                "teacher_labels_used": False,
                "support_qualified": False,
                "promotion_eligible": False,
                "hardware_authorized": False,
                "deployment_ready": False,
            },
        )

        def phase_boundary(phase: str) -> None:
            _require_materials_unchanged(
                root,
                base_contract,
                full_contract,
                source,
                materials,
                phase,
            )

        result = execute_full_support_schedule(
            runner,
            _evaluation_callback(motion=motion),
            phase_boundary=phase_boundary,
            evaluation_publisher=lambda record: _write_json_exclusive(
                run_dir / "evaluations" / f"evaluation_update_{record['update_count']}.json",
                record,
            ),
        )
        phase_boundary("before_full_support_result_publication")
        result = {
            **result,
            "material_manifest_sha256": material_sha,
            "run_dir": str(run_dir),
        }
        _write_json_exclusive(run_dir / "full_support_result.json", result)
    finally:
        env.close()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "experiment"):
        child = subparsers.add_parser(name)
        child.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
        if name == "experiment":
            child.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight(args.repository_root)
        print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    try:
        result = run_experiment(args.repository_root, args.run_dir)
    except Exception as error:
        partial = args.run_dir.expanduser().resolve(strict=False)
        if partial.is_dir() and not partial.is_symlink():
            failure_path = partial / "full_support_failure.json"
            try:
                if not os.path.lexists(failure_path):
                    _write_json_exclusive(
                        failure_path,
                        {
                            "schema_version": 1,
                            "kind": "g1_true23_sonic_task_space_ppo_full_support_failure_v1",
                            "contract_sha256": CONTRACT_SHA256,
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                            "candidate": None,
                            "teacher_labels_used": False,
                            "support_qualified": False,
                            "promotion_eligible": False,
                            "hardware_authorized": False,
                            "deployment_ready": False,
                        },
                    )
            except Exception:
                pass
        print(f"full-support experiment failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
