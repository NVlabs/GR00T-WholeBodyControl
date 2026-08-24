"""WSL/MJLab launcher for direct exact-policy SONIC true-23 training.

This launcher never talks to DDS, XRoboToolkit, Pico, or robot hardware.  It
trains the exact 267 -> FSQ64 -> [64 + H10-930] -> 23 policy topology inside
the official Unitree G1 23-DoF MJLab task.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
UNITREE_ROOT = REPO_ROOT / "external_dependencies" / "unitree_rl_mjlab"
MJLAB_ROOT = REPO_ROOT / "external_dependencies" / "mjlab"
DEFAULT_WARM_START = REPO_ROOT / "sonic_release" / "g1_23dof_rev_1_0_init.pt"
DEFAULT_SAMPLE_CSV = UNITREE_ROOT / "src" / "assets" / "motions" / "g1_23dof" / "dance1_subject2.csv"
DEFAULT_SAMPLE_NPZ = Path.home() / ".cache" / "g1_true23_mjlab" / "dance1_subject2_23dof.npz"
DEFAULT_RUN_ROOT = Path.home() / "g1_true23_runs"

UNITREE_COMMIT = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
MJLAB_COMMIT = "5af32e378dcb93c9e881ace83cc5a3f5d373fe60"
MUJOCO_WARP_COMMIT = "5a86ec28aa07741eb2e000d158f4ca4068ec146e"
MINIMUM_MOTION_FRAMES = 47
NORMAL_REFERENCE_PROFILE = "true23_step5_0p1s"
LOW_LATENCY_REFERENCE_PROFILE = "released_low_latency_step1_0p02s"


def _git_output(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _checkout_report(
    checkout: Path,
    expected_commit: str,
) -> dict[str, Any]:
    if not (checkout / ".git").is_dir():
        return {
            "path": str(checkout),
            "expected_commit": expected_commit,
            "present": False,
            "ok": False,
        }
    commit = _git_output(checkout, "rev-parse", "HEAD")
    return {
        "path": str(checkout),
        "expected_commit": expected_commit,
        "commit": commit,
        "present": True,
        # Relevant working files are content-hashed into every training
        # lineage. Avoid cross-OS false dirtiness from CRLF conversion here.
        "working_files_bound_by_lineage": True,
        "ok": commit == expected_commit,
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def _direct_vcs_commit(distribution_name: str) -> str | None:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    text = distribution.read_text("direct_url.json")
    if text is None:
        return None
    value = json.loads(text)
    commit = value.get("vcs_info", {}).get("commit_id")
    return commit if isinstance(commit, str) else None


def runtime_report(*, exercise_cuda: bool = True) -> dict[str, Any]:
    """Audit the exact low-VRAM runtime without constructing an environment."""

    import mujoco
    import torch

    packages = {
        name: _package_version(name)
        for name in (
            "mjlab",
            "mujoco",
            "mujoco-warp",
            "numpy",
            "rsl-rl-lib",
            "scipy",
            "tensordict",
            "torch",
            "unitree-rl-mjlab",
            "warp-lang",
        )
    }
    unitree = _checkout_report(UNITREE_ROOT, UNITREE_COMMIT)
    mjlab = _checkout_report(MJLAB_ROOT, MJLAB_COMMIT)
    mw_commit = _direct_vcs_commit("mujoco-warp")
    cuda_available = torch.cuda.is_available()
    gpu: dict[str, Any] | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        if exercise_cuda:
            value = torch.arange(
                4096,
                dtype=torch.float32,
                device="cuda:0",
            )
            if float((value * value).sum().cpu()) <= 0.0:
                raise RuntimeError("CUDA arithmetic smoke failed")
            torch.cuda.synchronize()
        gpu = {
            "name": properties.name,
            "total_vram_bytes": properties.total_memory,
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
        }

    checks = {
        "python_3_11": sys.version_info[:2] == (3, 11),
        "torch_2_9_cu128": packages["torch"] == "2.9.0+cu128",
        "cuda_available": cuda_available,
        "mujoco_stable_3_5_0": mujoco.__version__ == "3.5.0",
        "mujoco_warp_exact_commit": mw_commit == MUJOCO_WARP_COMMIT,
        "mjlab_1_2_0": packages["mjlab"] == "1.2.0",
        "rsl_rl_5_0_1": packages["rsl-rl-lib"] == "5.0.1",
        "scipy_1_16_2": packages["scipy"] == "1.16.2",
        "tensordict_0_10_0": packages["tensordict"] == "0.10.0",
        "warp_1_12_0": packages["warp-lang"] == "1.12.0",
        "unitree_checkout": unitree["ok"],
        "mjlab_checkout": mjlab["ok"],
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "packages": packages,
        "python": sys.version.split()[0],
        "mujoco_warp_commit": mw_commit,
        "gpu": gpu,
        "unitree_checkout": unitree,
        "mjlab_checkout": mjlab,
        "declared_compatibility_substitution": {
            "reason": (
                "MJLab 1.2 uv.lock development MuJoCo wheel is no longer available from the upstream index"
            ),
            "replacement": "mujoco==3.5.0",
            "mujoco_warp_commit_unchanged": True,
        },
    }


def motion_report(path: Path) -> dict[str, Any]:
    import numpy as np

    motion_path = path.expanduser().resolve()
    if not motion_path.is_file():
        return {
            "path": str(motion_path),
            "present": False,
            "ok": False,
        }
    required = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    }
    problems: list[str] = []
    shapes: dict[str, list[int]] = {}
    with np.load(motion_path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            problems.append(f"missing arrays: {missing}")
        for key in sorted(required & set(data.files)):
            value = data[key]
            shapes[key] = list(value.shape)
            if not np.isfinite(value).all():
                problems.append(f"{key} contains NaN or Inf")
        frames = (
            int(data["joint_pos"].shape[0]) if "joint_pos" in data.files and data["joint_pos"].ndim == 2 else 0
        )
        if frames < MINIMUM_MOTION_FRAMES:
            problems.append(f"motion needs at least {MINIMUM_MOTION_FRAMES} frames")
        if "joint_pos" in data.files and tuple(data["joint_pos"].shape[1:]) != (23,):
            problems.append("joint_pos must have shape [frames, 23]")
        if "joint_vel" in data.files and tuple(data["joint_vel"].shape) != (frames, 23):
            problems.append("joint_vel must have shape [frames, 23]")
        for key in required - {"fps", "joint_pos", "joint_vel"}:
            if key in data.files and data[key].shape[0] != frames:
                problems.append(f"{key} frame count differs from joint_pos")
        fps = float(data["fps"].reshape(-1)[0]) if "fps" in data.files and data["fps"].size == 1 else 0.0
        if fps != 50.0:
            problems.append("motion fps must be exactly 50")
    return {
        "path": str(motion_path),
        "present": True,
        "frames": frames,
        "fps": fps,
        "shapes": shapes,
        "problems": problems,
        "ok": not problems,
    }


def convert_sample_motion(
    *,
    input_csv: Path,
    output_npz: Path,
    device: str,
    overwrite: bool,
) -> Path:
    source = input_csv.expanduser().resolve()
    output = output_npz.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"sample CSV not found: {source}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite motion NPZ: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    converter = UNITREE_ROOT / "scripts" / "csv_to_npz.py"
    command = [
        sys.executable,
        str(converter),
        "--robot",
        "g1_23dof",
        "--input-file",
        str(source),
        "--output-name",
        str(output),
        "--input-fps",
        "30",
        "--output-fps",
        "50",
        "--device",
        device,
    ]
    subprocess.run(command, cwd=UNITREE_ROOT, check=True)
    report = motion_report(output)
    if not report["ok"]:
        raise RuntimeError(f"converted motion failed audit: {report}")
    return output


def _manifest_files(
    root: Path,
    *,
    logical_prefix: str,
) -> dict[str, Path]:
    return {
        f"{logical_prefix}/{path.relative_to(root).as_posix()}": path
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    }


def _source_files() -> dict[str, Path]:
    local_paths = (
        "gear_sonic/envs/mjlab/sonic_true23.py",
        "gear_sonic/trl/mjlab/config.py",
        "gear_sonic/trl/mjlab/runner.py",
        "gear_sonic/trl/mjlab/true23_actor.py",
        "gear_sonic/utils/g1_23dof_artifact.py",
        "gear_sonic/utils/g1_23dof_checkpoint_io.py",
        "gear_sonic/utils/g1_23dof_contract.py",
        "gear_sonic/utils/g1_23dof_mjlab_training.py",
        "gear_sonic/scripts/train_g1_23dof_mjlab.py",
    )
    files = {logical_path: REPO_ROOT / logical_path for logical_path in local_paths}
    official_paths = (
        "src/assets/robots/unitree_g1/g1_23dof_constants.py",
        "src/tasks/tracking/config/g1_23dof/env_cfgs.py",
        "src/tasks/tracking/tracking_env_cfg.py",
        "src/tasks/tracking/mdp/commands.py",
    )
    files.update(
        {f"unitree_rl_mjlab/{logical_path}": (UNITREE_ROOT / logical_path) for logical_path in official_paths}
    )
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"training source files missing: {missing}")
    return files


def _resolved_training_config(
    *,
    agent_cfg: Any,
    mode: str,
    motion_path: Path,
    num_envs: int,
    seed: int,
    planned_updates: int,
    reference_profile: str,
) -> dict[str, Any]:
    packages = {
        name: _package_version(name)
        for name in (
            "mjlab",
            "mujoco",
            "mujoco-warp",
            "numpy",
            "rsl-rl-lib",
            "scipy",
            "tensordict",
            "torch",
            "warp-lang",
        )
    }
    agent = asdict(agent_cfg)
    # Session path and logger duration are not policy semantics. Keep the
    # intended total update budget explicit so interrupted runs match.
    agent.pop("resume", None)
    agent.pop("load_run", None)
    agent.pop("load_checkpoint", None)
    agent["max_iterations"] = planned_updates
    return {
        "schema": "g1_true23_mjlab_resolved_training_v1",
        "mode": mode,
        "task": "Unitree-G1-23Dof-SONIC-True23-Tracking",
        "robot": "g1_23dof_rev_1_0",
        "action_count": 23,
        "reference_profile": reference_profile,
        "motion_filename": motion_path.name,
        "num_envs": num_envs,
        "seed": seed,
        "planned_updates": planned_updates,
        "history_length": 10,
        "tokenizer_dim": 268,
        "policy_dim": 930,
        "initial_simulation_prime_steps": 0,
        "initial_target_refresh_max_attempts": 32,
        "initial_full_batch_reset_rejection": True,
        "prime_after_rsl_wrapper_reset": True,
        "randomize_initial_episode_lengths": mode == "train",
        "push_disturbances_enabled": True,
        "domain_randomization": [
            "base_com",
            "encoder_bias",
            "foot_friction",
        ],
        "agent": agent,
        "runtime_packages": packages,
        "runtime_source_commits": {
            "unitree_rl_mjlab": UNITREE_COMMIT,
            "mjlab": MJLAB_COMMIT,
            "mujoco_warp": MUJOCO_WARP_COMMIT,
        },
    }


def _build_material_manifests(
    *,
    motion_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from gear_sonic.utils.g1_23dof_mjlab_training import build_file_manifest

    asset_root = UNITREE_ROOT / "src" / "assets" / "robots" / "unitree_g1"
    source_manifest = build_file_manifest(
        _source_files(),
        kind="source_files",
    )
    asset_manifest = build_file_manifest(
        _manifest_files(
            asset_root,
            logical_prefix="unitree_g1",
        ),
        kind="robot_assets",
    )
    dataset_manifest = build_file_manifest(
        {f"motions/{motion_path.name}": motion_path},
        kind="motion_dataset",
    )
    return source_manifest, asset_manifest, dataset_manifest


def _new_run_dir(mode: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(os.environ.get("G1_MJLAB_RUN_ROOT", str(DEFAULT_RUN_ROOT)))
    return run_root.expanduser().resolve() / f"{mode}_{timestamp}"


def run_training(args: argparse.Namespace) -> Path:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    import torch

    from gear_sonic.envs.mjlab.sonic_true23 import (
        make_sonic_true23_tracking_env_cfg,
        prime_sonic_true23_training_environment,
    )
    from gear_sonic.trl.mjlab.config import true23_mjlab_ppo_runner_cfg
    from gear_sonic.trl.mjlab.runner import True23MjlabOnPolicyRunner

    runtime = runtime_report()
    if not runtime["ready"]:
        raise RuntimeError("MJLab runtime preflight failed:\n" + json.dumps(runtime, indent=2, sort_keys=True))
    warm_start = args.warm_start.expanduser().resolve()
    motion_path = args.motion_file.expanduser().resolve()
    if not warm_start.is_file():
        raise FileNotFoundError(f"warm start missing: {warm_start}")
    motion = motion_report(motion_path)
    if not motion["ok"]:
        raise ValueError("motion audit failed:\n" + json.dumps(motion, indent=2, sort_keys=True))
    if args.num_envs <= 0 or args.iterations <= 0:
        raise ValueError("num-envs and iterations must be positive")
    if args.mode == "train" and args.iterations < 50:
        raise ValueError(
            "production training requires at least 50 outer PPO iterations"
        )

    run_dir = args.run_dir.expanduser().resolve() if args.run_dir is not None else _new_run_dir(args.mode)
    if run_dir.exists() and any(run_dir.iterdir()) and args.resume is None:
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg = true23_mjlab_ppo_runner_cfg()
    agent_cfg.actor.warm_start_path = str(warm_start)
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    agent_cfg.save_interval = args.save_interval
    if args.mode == "smoke":
        agent_cfg.num_steps_per_env = min(
            agent_cfg.num_steps_per_env,
            8,
        )
        agent_cfg.algorithm.num_learning_epochs = min(
            agent_cfg.algorithm.num_learning_epochs,
            2,
        )
        agent_cfg.algorithm.num_mini_batches = min(
            agent_cfg.algorithm.num_mini_batches,
            2,
        )

    resolved_config = _resolved_training_config(
        agent_cfg=agent_cfg,
        mode=args.mode,
        motion_path=motion_path,
        num_envs=args.num_envs,
        seed=args.seed,
        planned_updates=args.iterations,
        reference_profile=args.reference_profile,
    )
    source_manifest, asset_manifest, dataset_manifest = _build_material_manifests(
        motion_path=motion_path,
    )

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    configure_torch_backends()
    torch.cuda.set_device(0)
    device = "cuda:0"

    env_cfg = make_sonic_true23_tracking_env_cfg(
        motion_file=str(motion_path),
        reference_profile=args.reference_profile,
        num_envs=args.num_envs,
        play=False,
    )
    env_cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
        )
        # RslRlVecEnvWrapper performs its own explicit reset. Refresh cached
        # motion targets only after that reset, without a fake simulation step.
        prime_report = prime_sonic_true23_training_environment(wrapped)
        (run_dir / "environment_prime.json").write_text(
            json.dumps(prime_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        env.close()
        raise
    try:
        runner = True23MjlabOnPolicyRunner(
            wrapped,
            asdict(agent_cfg),
            str(run_dir),
            device,
            warm_start_checkpoint_path=warm_start,
            resolved_config=resolved_config,
            source_manifest=source_manifest,
            asset_manifest=asset_manifest,
            dataset_manifest=dataset_manifest,
            checkpoint_dir=run_dir / "checkpoints",
        )
        if args.resume is not None:
            runner.load(str(args.resume.expanduser().resolve()))
        (run_dir / "resolved_training.json").write_text(
            json.dumps(resolved_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_dir / "lineage.json").write_text(
            json.dumps(runner.training_lineage, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        remaining = args.iterations - runner.completed_update_count
        if remaining <= 0:
            raise ValueError("resume checkpoint already reached planned updates")
        session_updates = (
            remaining
            if args.session_updates is None
            else min(args.session_updates, remaining)
        )
        runner.learn(
            num_learning_iterations=session_updates,
            # Keep tiny smoke diagnostics free of artificial near-timeouts.
            # Production training retains randomized initial episode lengths.
            init_at_random_ep_len=args.mode == "train",
        )
    finally:
        env.close()
    return run_dir


def _preflight(args: argparse.Namespace) -> int:
    report = {
        "runtime": runtime_report(),
        "warm_start": {
            "path": str(args.warm_start.expanduser().resolve()),
            "present": args.warm_start.expanduser().resolve().is_file(),
        },
        "motion": motion_report(args.motion_file),
        "safety": {
            "simulator_only": True,
            "robot_network_commands": False,
            "deployment_ready": False,
        },
    }
    report["ready"] = report["runtime"]["ready"] and report["warm_start"]["present"] and report["motion"]["ok"]
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.json_output is not None:
        destination = args.json_output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output + "\n", encoding="utf-8")
    return 0 if report["ready"] else 2


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _add_common_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=DEFAULT_WARM_START,
    )
    parser.add_argument(
        "--motion-file",
        type=Path,
        default=DEFAULT_SAMPLE_NPZ,
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--session-updates",
        type=_positive_int,
        help=(
            "Run only this many outer PPO iterations now; omitted means all "
            "remaining planned iterations. This does not change immutable "
            "lineage."
        ),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-profile",
        default=NORMAL_REFERENCE_PROFILE,
        choices=(
            NORMAL_REFERENCE_PROFILE,
            LOW_LATENCY_REFERENCE_PROFILE,
        ),
    )
    parser.add_argument(
        "--save-interval",
        type=_positive_int,
        default=10,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Exact 23-DoF SONIC training in MJLab; simulator only."))
    subparsers = parser.add_subparsers(dest="mode", required=True)

    convert = subparsers.add_parser(
        "convert-sample",
        help="Convert Unitree's included G1-23 CSV into a 50-Hz NPZ.",
    )
    convert.add_argument("--input-csv", type=Path, default=DEFAULT_SAMPLE_CSV)
    convert.add_argument("--output-npz", type=Path, default=DEFAULT_SAMPLE_NPZ)
    convert.add_argument("--device", default="cuda:0")
    convert.add_argument("--overwrite", action="store_true")

    preflight = subparsers.add_parser(
        "preflight",
        help="Audit CUDA, source commits, checkpoint, and motion data.",
    )
    preflight.add_argument(
        "--warm-start",
        type=Path,
        default=DEFAULT_WARM_START,
    )
    preflight.add_argument(
        "--motion-file",
        type=Path,
        default=DEFAULT_SAMPLE_NPZ,
    )
    preflight.add_argument("--json-output", type=Path)

    smoke = subparsers.add_parser(
        "smoke",
        help="Run a short real CUDA/MJLab PPO update test.",
    )
    _add_common_training_arguments(smoke)
    smoke.add_argument("--num-envs", type=_positive_int, default=4)
    smoke.add_argument("--iterations", type=_positive_int, default=2)

    train = subparsers.add_parser(
        "train",
        help="Run low-VRAM exact-policy training with disturbances.",
    )
    _add_common_training_arguments(train)
    train.add_argument("--num-envs", type=_positive_int, default=128)
    train.add_argument("--iterations", type=_positive_int, default=10_001)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.mode == "convert-sample":
        output = convert_sample_motion(
            input_csv=args.input_csv,
            output_npz=args.output_npz,
            device=args.device,
            overwrite=args.overwrite,
        )
        print(output)
        return 0
    if args.mode == "preflight":
        return _preflight(args)
    run_dir = run_training(args)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
