"""Run one fresh full-support SONIC PPO update at quarter learning rate."""

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
from gear_sonic.trl.mjlab.sonic_task_space_ppo_full_support_runner import (
    FIXED_SEED,
    NUM_ENVS,
    evaluate_full_support_policy,
    load_full_support_contract,
)
from gear_sonic.trl.mjlab.sonic_task_space_ppo_quarter_step_runner import (
    CONTRACT_RELATIVE_PATH,
    CONTRACT_SHA256,
    LINE_SEARCH_REPORT_SHA256,
    QUARTER_LEARNING_RATE,
    SonicTaskSpacePpoQuarterStepRunner,
    assess_quarter_step_evaluations,
    load_quarter_step_contract,
)
from gear_sonic.trl.mjlab.sonic_task_space_ppo_runner import (
    audit_task_space_ppo_env_cfg,
    load_task_space_ppo_contract,
    make_task_space_ppo_env_cfg,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda:0"
FIXED_GPU = 0
RESULT_FILENAME = "quarter_step_result.json"
FAILURE_FILENAME = "quarter_step_failure.json"
CHECKPOINT_FILENAME = "sonic_task_space_quarter_step_model_1.pt"
LINE_SEARCH_PATH = Path("/root/g1_true23_runs/sonic_task_space_ppo_full_support_delta_line_search_v1_retry1.json")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _strict_file(path: Path, expected_sha256: str, context: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise ValueError(f"{context} must be regular file")
    resolved = lexical.resolve(strict=True)
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{context} SHA256 mismatch")
    return resolved


def _supplemental_sources(root: Path) -> dict[str, str]:
    relative = (
        CONTRACT_RELATIVE_PATH,
        Path("gear_sonic/trl/mjlab/sonic_task_space_ppo_quarter_step_runner.py"),
        Path("gear_sonic/scripts/train_g1_true23_sonic_task_space_ppo_quarter_step.py"),
    )
    result: dict[str, str] = {}
    for item in relative:
        path = (root / item).resolve(strict=True)
        if path.is_symlink() or path.parent != (root / item.parent).resolve(strict=True):
            raise ValueError(f"quarter-step source path drift: {item}")
        result[item.as_posix()] = sha256_file(path)
    return result


def preflight(repository_root: Path) -> dict[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract = load_quarter_step_contract(root)
        base = parent.preflight(root)
        if base.get("ready") is not True:
            raise RuntimeError("parent full-support preflight not ready")
        line = _strict_file(LINE_SEARCH_PATH, LINE_SEARCH_REPORT_SHA256, "quarter-step line search")
        sources = _supplemental_sources(root)
        material = {
            "parent_material_manifest_sha256": base["material_manifest"]["material_manifest_sha256"],
            "line_search_report": {"path": str(line), "sha256": LINE_SEARCH_REPORT_SHA256},
            "supplemental_sources": sources,
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_task_space_ppo_quarter_step_preflight_v1",
            "ready": True,
            "contract_sha256": CONTRACT_SHA256,
            "contract": contract,
            "parent_preflight": base,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha256(material),
            "simulator_constructed": False,
            "training_transitions": 0,
            "training_updates": 0,
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
            "kind": "g1_true23_sonic_task_space_ppo_quarter_step_preflight_v1",
            "ready": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "simulator_constructed": False,
            "training_transitions": 0,
            "training_updates": 0,
            "optimizer_steps": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }


def _materials_unchanged(root: Path, audit: Mapping[str, Any], phase: str) -> None:
    current = preflight(root)
    if current.get("ready") is not True:
        raise RuntimeError(f"quarter-step material recheck failed at {phase}")
    if current.get("material_manifest_sha256") != audit.get("material_manifest_sha256"):
        raise RuntimeError(f"quarter-step materials changed at {phase}")


def _write_torch_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"quarter-step checkpoint exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as h:
            temporary = Path(h.name)
        torch.save(dict(payload), temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        if not isinstance(loaded, Mapping) or loaded.get("contract_sha256") != CONTRACT_SHA256:
            raise ValueError("quarter-step checkpoint round-trip failed")
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _evaluate(runner: SonicTaskSpacePpoQuarterStepRunner, motion: Path, update: int) -> dict[str, Any]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    with parent._preserved_rng_for_evaluation():
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
        "parent_engine_contract_sha256": parent.CONTRACT_SHA256,
        "material_manifest_sha256": runner._run_materials_sha256,
    }


def run_experiment(repository_root: Path, requested_run_dir: Path) -> dict[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("quarter-step preflight failed:\n" + json.dumps(audit, indent=2, sort_keys=True))
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("quarter-step experiment requires WORLD_SIZE=1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FIXED_GPU)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("quarter-step experiment requires fixed visible CUDA device 0")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED % (2**32))
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)

    base_contract = load_task_space_ppo_contract(root)
    full_contract = load_full_support_contract(root)
    quarter_contract = load_quarter_step_contract(root)
    actor = base_contract["actor_initialization"]
    environment = base_contract["environment"]
    source = Path(actor["source_checkpoint_linux_path"]).resolve(strict=True)
    topology = (root / actor["topology_checkpoint_relative_path"]).resolve(strict=True)
    encoder = (root / actor["encoder_onnx_relative_path"]).resolve(strict=True)
    decoder = (root / actor["v2_decoder_relative_path"]).resolve(strict=True)
    motion = (root / environment["motion_relative_path"]).resolve(strict=True)
    run_dir = parent._create_run_dir_exclusive(requested_run_dir)
    parent._write_json_exclusive(run_dir / "preflight.json", audit)
    parent._write_json_exclusive(run_dir / "material_manifest.json", audit["material_manifest"])

    agent_cfg = parent._agent_config(topology)
    parent_materials = audit["parent_preflight"]["material_manifest"]
    engine_resolved = parent._resolved_config(agent_cfg, parent_materials)
    resolved_report = {
        **engine_resolved,
        "executed_learning_rate": QUARTER_LEARNING_RATE,
        "parent_constructor_learning_rate": parent.FIXED_LEARNING_RATE,
    }
    parent._write_json_exclusive(run_dir / "resolved_training.json", resolved_report)
    env_cfg = make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=NUM_ENVS)
    env_cfg.seed = FIXED_SEED
    task_audit = audit_task_space_ppo_env_cfg(env_cfg, expected_num_envs=NUM_ENVS)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        prime = prime_sonic_true23_training_environment(wrapped)
        parent._write_json_exclusive(run_dir / "environment_prime.json", prime)
        torch.manual_seed(FIXED_SEED)
        torch.cuda.manual_seed_all(FIXED_SEED)
        engine_dir = run_dir / "engine"
        engine_dir.mkdir()
        (engine_dir / "checkpoints").mkdir()
        runner = SonicTaskSpacePpoQuarterStepRunner(
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
            quarter_step_contract=quarter_contract,
            run_materials_sha256=audit["material_manifest_sha256"],
        )
        parent._write_json_exclusive(
            run_dir / "initialization.json",
            {
                "schema_version": 1,
                "kind": "g1_true23_sonic_task_space_ppo_quarter_step_initialization_v1",
                "contract_sha256": CONTRACT_SHA256,
                "task_audit": task_audit,
                "initial_policy_state_sha256": runner._initial_overlay_policy_state_sha256,
                "initial_critic_state_sha256": runner._initial_critic_state_sha256,
                "executed_learning_rate": runner.alg.learning_rate,
                "optimizer_state_entry_count": len(runner.alg.optimizer.state),
                "training_updates": 0,
                "optimizer_steps": 0,
                "training_transitions": 0,
                "candidate_selected": False,
                "support_qualified": False,
                "promotion_eligible": False,
                "deployment_ready": False,
                "hardware_authorized": False,
            },
        )

        _materials_unchanged(root, audit, "before_update0_evaluation")
        evaluation0 = _evaluate(runner, motion, 0)
        parent._write_json_exclusive(run_dir / "evaluations" / "evaluation_update_0.json", evaluation0)
        baseline_assessment = assess_quarter_step_evaluations([evaluation0, {**evaluation0, "update_count": 1}])
        if baseline_assessment["baseline_passed"] is not True:
            raise RuntimeError("quarter-step update0 structural baseline mismatch")

        _materials_unchanged(root, audit, "before_rollout")
        rollout = runner.collect_full_support_rollout()
        if rollout["coverage_assessment"]["gate_passed"] is not True:
            raise RuntimeError("quarter-step pre-Adam coverage gate failed")
        loss = runner.optimize_collected_rollout()
        checkpoint = runner.quarter_step_checkpoint()
        checkpoint_path = run_dir / "checkpoints" / CHECKPOINT_FILENAME
        _write_torch_exclusive(checkpoint_path, checkpoint)
        checkpoint_sha = sha256_file(checkpoint_path)

        _materials_unchanged(root, audit, "before_update1_evaluation")
        evaluation1 = _evaluate(runner, motion, 1)
        parent._write_json_exclusive(run_dir / "evaluations" / "evaluation_update_1.json", evaluation1)
        assessment = assess_quarter_step_evaluations([evaluation0, evaluation1])
        result = {
            "schema_version": 1,
            "kind": "g1_true23_sonic_task_space_ppo_quarter_step_result_v1",
            "contract_sha256": CONTRACT_SHA256,
            "material_manifest_sha256": audit["material_manifest_sha256"],
            "run_dir": str(run_dir),
            "learning_rate": QUARTER_LEARNING_RATE,
            "rollout_evidence_sha256": runner._rollout_evidence_sha256,
            "loss": loss,
            "evaluations": [evaluation0, evaluation1],
            "assessment": assessment,
            "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
            "candidate": (
                {"path": str(checkpoint_path), "sha256": checkpoint_sha}
                if assessment["candidate_selected"] is True
                else None
            ),
            "training_transitions": runner._executed_training_transitions,
            "training_updates": runner.completed_update_count,
            "optimizer_steps": runner._optimizer_step_count,
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
            failure = run_dir / FAILURE_FILENAME
            if not os.path.lexists(failure):
                parent._write_json_exclusive(
                    failure,
                    {
                        "schema_version": 1,
                        "kind": "g1_true23_sonic_task_space_ppo_quarter_step_failure_v1",
                        "contract_sha256": CONTRACT_SHA256,
                        "error": {"type": type(error).__name__, "message": str(error)},
                        "candidate": None,
                        "support_qualified": False,
                        "promotion_eligible": False,
                        "deployment_ready": False,
                        "hardware_authorized": False,
                    },
                )
        print(f"quarter-step experiment failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
