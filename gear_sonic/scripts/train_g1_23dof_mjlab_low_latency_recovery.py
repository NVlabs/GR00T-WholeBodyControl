"""Train exact released-low-latency true23 policy on recovery material.

This launcher is isolated from the existing normal-profile run.  It starts
from the approved low-latency initialization checkpoint and records the new
FK-built curriculum plus reward source in immutable simulator-only lineage.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from gear_sonic.scripts.train_g1_23dof_mjlab import (
    _manifest_files,
    _resolved_training_config,
    _source_files,
    motion_report,
    runtime_report,
)
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    checkpoint_stage,
    load_safe_true23_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import REFERENCE_PROFILE_LOW_LATENCY

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WARM_START = (
    REPO_ROOT / "sonic_release" / "g1_23dof_rev_1_0_low_latency_init.pt"
)
DEFAULT_MOTION = Path(
    "/root/.cache/g1_true23_mjlab/recovery/"
    "g1_true23_low_latency_stand_transition_dance_v2.npz"
)
DEFAULT_METADATA = DEFAULT_MOTION.with_suffix(".json")
DEFAULT_RUN = Path(
    "/root/g1_true23_runs/"
    "low_latency_recovery_stand_transition_dance_v2"
)
RECOVERY_SOURCE_FILES = (
    REPO_ROOT
    / "gear_sonic"
    / "envs"
    / "mjlab"
    / "sonic_true23_low_latency_recovery.py",
    REPO_ROOT
    / "gear_sonic"
    / "scripts"
    / "build_g1_23dof_low_latency_recovery_motion.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_metadata(path: Path, motion_path: Path) -> dict[str, Any]:
    metadata = path.expanduser().resolve()
    if not metadata.is_file():
        raise FileNotFoundError(f"recovery metadata missing: {metadata}")
    value = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("recovery metadata must be an object")
    if value.get("schema") != "g1_true23_low_latency_recovery_motion_v1":
        raise ValueError("recovery metadata schema mismatch")
    output = value.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("recovery metadata lacks output binding")
    if output.get("filename") != motion_path.name:
        raise ValueError("recovery metadata filename differs from motion")
    if output.get("sha256") != _sha256(motion_path):
        raise ValueError("recovery metadata SHA-256 differs from motion")
    if value.get("deployment_ready") is not False:
        raise ValueError("recovery motion must remain simulator-only")
    return value


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"existing run material differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    warm_start = args.warm_start.expanduser().resolve()
    motion = args.motion_file.expanduser().resolve()
    metadata = args.motion_metadata.expanduser().resolve()
    problems: list[str] = []
    warm: dict[str, Any] = {"path": str(warm_start), "present": warm_start.is_file()}
    if warm_start.is_file():
        try:
            checkpoint = load_safe_true23_checkpoint(warm_start)
            profile = checkpoint["g1_23dof_metadata"].get("reference_profile")
            warm.update(
                {
                    "checkpoint_stage": checkpoint_stage(checkpoint),
                    "reference_profile": profile,
                    "sha256": _sha256(warm_start),
                }
            )
            if profile != REFERENCE_PROFILE_LOW_LATENCY:
                problems.append("warm start is not released low-latency profile")
        except Exception as exc:
            problems.append(f"warm-start validation failed: {exc}")
    else:
        problems.append("warm start missing")
    motion_audit = motion_report(motion)
    if not motion_audit["ok"]:
        problems.append("recovery motion audit failed")
    recovery_metadata: dict[str, Any] | None = None
    if motion_audit["ok"]:
        try:
            recovery_metadata = _read_metadata(metadata, motion)
        except Exception as exc:
            problems.append(f"recovery metadata validation failed: {exc}")
    runtime = runtime_report()
    if not runtime["ready"]:
        problems.append("MJLab runtime not ready")
    return {
        "schema": "g1_true23_low_latency_recovery_preflight_v1",
        "ready": not problems,
        "problems": problems,
        "runtime": runtime,
        "warm_start": warm,
        "motion": motion_audit,
        "motion_metadata": recovery_metadata,
        "safety": {
            "simulator_only": True,
            "robot_network_commands": False,
            "deployment_ready": False,
        },
    }


def run_training(args: argparse.Namespace) -> Path:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    from gear_sonic.envs.mjlab.sonic_true23 import (
        prime_sonic_true23_training_environment,
    )
    from gear_sonic.envs.mjlab.sonic_true23_low_latency_recovery import (
        make_low_latency_recovery_env_cfg,
        recovery_reward_contract,
    )
    from gear_sonic.scripts.train_g1_23dof_mjlab import UNITREE_ROOT
    from gear_sonic.trl.mjlab.config import true23_mjlab_ppo_runner_cfg
    from gear_sonic.trl.mjlab.runner import True23MjlabOnPolicyRunner
    from gear_sonic.utils.g1_23dof_mjlab_training import build_file_manifest

    report = preflight(args)
    if not report["ready"]:
        raise RuntimeError(
            "low-latency recovery preflight failed:\n"
            + json.dumps(report, indent=2, sort_keys=True)
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

    resolved = _resolved_training_config(
        agent_cfg=agent_cfg,
        mode=args.mode,
        motion_path=motion_path,
        num_envs=args.num_envs,
        seed=args.seed,
        planned_updates=args.iterations,
        reference_profile=REFERENCE_PROFILE_LOW_LATENCY,
    )
    resolved["schema"] = "g1_true23_mjlab_low_latency_recovery_v1"
    resolved["task"] = "Unitree-G1-23Dof-SONIC-LowLatency-Recovery"
    resolved["recovery"] = {
        "approved_low_latency_initialization_only": True,
        "normal_profile_model1000_transferred": False,
        "normal_profile_transfer_rejected_reason": (
            "decoder topology and encoder temporal semantics differ"
        ),
        "curriculum_metadata_filename": metadata_path.name,
        "curriculum_metadata_sha256": _sha256(metadata_path),
        "reward_contract": recovery_reward_contract(),
        "critic_reused": False,
        "optimizer_reused": False,
    }

    source_files = _source_files()
    for path in RECOVERY_SOURCE_FILES:
        source_files[f"recovery/{path.name}"] = path
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
    env_cfg = make_low_latency_recovery_env_cfg(
        motion_file=str(motion_path),
        num_envs=args.num_envs,
        play=False,
    )
    env_cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        prime = prime_sonic_true23_training_environment(wrapped)
        _write_or_verify_json(run_dir / "environment_prime.json", prime)
        runner = True23MjlabOnPolicyRunner(
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
        raise argparse.ArgumentTypeError("must be in (0, 1e-3]")
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
    output = run_training(args)
    print(output)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
