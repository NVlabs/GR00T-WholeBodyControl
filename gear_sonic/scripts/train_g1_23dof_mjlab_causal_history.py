"""Train the exact true23 network under a new causal-history contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Sequence

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    CAUSAL_HISTORY_PROFILE,
    causal_history_profile_contract,
)
from gear_sonic.scripts.train_g1_23dof_mjlab import (
    UNITREE_ROOT,
    _manifest_files,
    _resolved_training_config,
    _source_files,
)
from gear_sonic.scripts.train_g1_23dof_mjlab_low_latency_recovery import (
    DEFAULT_METADATA,
    DEFAULT_MOTION,
    DEFAULT_WARM_START,
    _read_metadata,
    _sha256,
    _write_or_verify_json,
    preflight as low_latency_material_preflight,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = Path(
    "/root/g1_true23_runs/causal_history_recovery_stand_transition_dance_v1"
)
CAUSAL_SOURCE_FILES = (
    REPO_ROOT
    / "gear_sonic"
    / "envs"
    / "mjlab"
    / "sonic_true23_causal_history.py",
    REPO_ROOT
    / "gear_sonic"
    / "envs"
    / "mjlab"
    / "sonic_true23_low_latency_recovery.py",
    REPO_ROOT
    / "gear_sonic"
    / "trl"
    / "mjlab"
    / "causal_history_runner.py",
    REPO_ROOT
    / "gear_sonic"
    / "scripts"
    / "build_g1_23dof_low_latency_recovery_motion.py",
    REPO_ROOT
    / "gear_sonic"
    / "scripts"
    / "train_g1_23dof_mjlab_low_latency_recovery.py",
    Path(__file__).resolve(),
)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    base = low_latency_material_preflight(args)
    contract = causal_history_profile_contract()
    problems = list(base["problems"])
    if contract["future_samples_relative_to_emission"] is not False:
        problems.append("causal profile unexpectedly requires future samples")
    if contract["released_profile_relabel_permitted"] is not False:
        problems.append("causal profile permits released-profile relabeling")
    metadata = None
    if base["motion"]["ok"]:
        try:
            metadata = _read_metadata(
                args.motion_metadata.expanduser().resolve(),
                args.motion_file.expanduser().resolve(),
            )
        except Exception as exc:
            problems.append(f"causal curriculum binding failed: {exc}")
    return {
        "schema": "g1_true23_causal_history_recovery_preflight_v1",
        "ready": not problems,
        "problems": problems,
        "runtime": base["runtime"],
        "architecture_initialization": base["warm_start"],
        "motion": base["motion"],
        "motion_metadata": metadata,
        "semantic_profile": contract,
        "safety": {
            "simulator_only": True,
            "robot_network_commands": False,
            "released_future_profile_relabelled": False,
            "existing_future_profile_exporter_compatible": False,
            "deployment_ready": False,
        },
    }


def run_training(args: argparse.Namespace) -> Path:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    import torch

    from gear_sonic.envs.mjlab.sonic_true23 import (
        prime_sonic_true23_training_environment,
    )
    from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
        make_causal_history_recovery_env_cfg,
    )
    from gear_sonic.envs.mjlab.sonic_true23_low_latency_recovery import (
        recovery_reward_contract,
    )
    from gear_sonic.trl.mjlab.causal_history_runner import (
        CausalHistoryMjlabOnPolicyRunner,
    )
    from gear_sonic.trl.mjlab.config import true23_mjlab_ppo_runner_cfg
    from gear_sonic.utils.g1_23dof_mjlab_training import build_file_manifest

    audit = preflight(args)
    if not audit["ready"]:
        raise RuntimeError(
            "causal-history recovery preflight failed:\n"
            + json.dumps(audit, indent=2, sort_keys=True)
        )
    warm_start = args.warm_start.expanduser().resolve()
    motion_path = args.motion_file.expanduser().resolve()
    metadata_path = args.motion_metadata.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and args.resume is None:
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg = true23_mjlab_ppo_runner_cfg()
    agent_cfg.actor.warm_start_path = str(warm_start)
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    agent_cfg.save_interval = args.save_interval
    agent_cfg.algorithm.learning_rate = args.learning_rate
    agent_cfg.algorithm.entropy_coef = 0.002
    agent_cfg.algorithm.max_grad_norm = 0.5
    if args.mode == "smoke":
        agent_cfg.num_steps_per_env = min(agent_cfg.num_steps_per_env, 8)
        agent_cfg.algorithm.num_learning_epochs = min(
            agent_cfg.algorithm.num_learning_epochs, 2
        )
        agent_cfg.algorithm.num_mini_batches = min(
            agent_cfg.algorithm.num_mini_batches, 2
        )

    semantic_contract = causal_history_profile_contract()
    resolved = _resolved_training_config(
        agent_cfg=agent_cfg,
        mode=args.mode,
        motion_path=motion_path,
        num_envs=args.num_envs,
        seed=args.seed,
        planned_updates=args.iterations,
        reference_profile=CAUSAL_HISTORY_PROFILE,
    )
    resolved["schema"] = "g1_true23_mjlab_causal_history_recovery_v1"
    resolved["task"] = "Unitree-G1-23Dof-SONIC-CausalHistory-Recovery"
    resolved["semantic_profile"] = semantic_contract
    resolved["architecture_initialization"] = {
        "checkpoint_filename": warm_start.name,
        "checkpoint_sha256": _sha256(warm_start),
        "source_profile": "released_low_latency_step1_0p02s",
        "weights_reused_as_training_initialization": True,
        "source_future_semantics_inherited": False,
        "checkpoint_relabelled": False,
        "retraining_required": True,
        "normal_profile_model1000_transferred": False,
    }
    resolved["recovery"] = {
        "curriculum_metadata_filename": metadata_path.name,
        "curriculum_metadata_sha256": _sha256(metadata_path),
        "reward_contract": recovery_reward_contract(),
        "critic_reused": False,
        "optimizer_reused": False,
        "checkpoint_filename_pattern": "causal_model_N.pt",
        "released_future_profile_exporter_must_reject": True,
    }

    source_files = _source_files()
    for path in CAUSAL_SOURCE_FILES:
        source_files[f"causal_recovery/{path.name}"] = path
    source_manifest = build_file_manifest(source_files, kind="source_files")
    asset_root = UNITREE_ROOT / "src" / "assets" / "robots" / "unitree_g1"
    asset_manifest = build_file_manifest(
        _manifest_files(asset_root, logical_prefix="unitree_g1"),
        kind="robot_assets",
    )
    dataset_manifest = build_file_manifest(
        {
            f"motions/{motion_path.name}": motion_path,
            f"motions/{metadata_path.name}": metadata_path,
        },
        kind="motion_dataset",
    )

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    configure_torch_backends()
    torch.cuda.set_device(0)
    device = "cuda:0"
    env_cfg = make_causal_history_recovery_env_cfg(
        motion_file=str(motion_path),
        num_envs=args.num_envs,
        play=False,
    )
    env_cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        prime = prime_sonic_true23_training_environment(wrapped)
        prime_name = (
            "environment_prime.json"
            if args.resume is None
            else f"environment_prime_{args.resume.stem}.json"
        )
        _write_or_verify_json(run_dir / prime_name, prime)
        runner = CausalHistoryMjlabOnPolicyRunner(
            wrapped,
            asdict(agent_cfg),
            str(run_dir),
            device,
            warm_start_checkpoint_path=warm_start,
            resolved_config=resolved,
            source_manifest=source_manifest,
            asset_manifest=asset_manifest,
            dataset_manifest=dataset_manifest,
            checkpoint_dir=run_dir / "checkpoints",
        )
        if args.resume is not None:
            runner.load(str(args.resume.expanduser().resolve()))
        _write_or_verify_json(run_dir / "resolved_training.json", resolved)
        _write_or_verify_json(run_dir / "lineage.json", runner.training_lineage)
        _write_or_verify_json(run_dir / "causal_semantic_profile.json", semantic_contract)
        remaining = args.iterations - runner.completed_update_count
        if remaining <= 0:
            raise ValueError("resume checkpoint already reached planned updates")
        session_updates = min(args.session_updates or remaining, remaining)
        runner.learn(
            num_learning_iterations=session_updates,
            init_at_random_ep_len=args.mode == "train",
        )
    finally:
        env.close()
    return run_dir


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not 0.0 < result <= 1.0e-3:
        raise argparse.ArgumentTypeError("must be in (0,1e-3]")
    return result


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--warm-start", type=Path, default=DEFAULT_WARM_START)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--motion-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--session-updates", type=_positive_int)
    parser.add_argument("--num-envs", type=_positive_int, default=128)
    parser.add_argument("--iterations", type=_positive_int, default=5001)
    parser.add_argument("--save-interval", type=_positive_int, default=250)
    parser.add_argument("--learning-rate", type=_positive_float, default=5.0e-5)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--gpu", type=int, default=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    _common(preflight_parser)
    preflight_parser.add_argument("--json-output", type=Path)
    smoke_parser = subparsers.add_parser("smoke")
    _common(smoke_parser)
    smoke_parser.set_defaults(num_envs=4, iterations=2, save_interval=1)
    train_parser = subparsers.add_parser("train")
    _common(train_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "preflight":
        report = preflight(args)
        output = json.dumps(report, indent=2, sort_keys=True)
        print(output)  # noqa: T201
        if args.json_output is not None:
            destination = args.json_output.expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(output + "\n", encoding="utf-8")
        return 0 if report["ready"] else 2
    run_dir = run_training(args)
    print(run_dir)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
