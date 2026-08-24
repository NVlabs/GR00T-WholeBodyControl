"""Trace immutable task-space PPO v2 checkpoints 0 and 5 in exclusive v3 output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_true23_sonic_task_space_checkpoint_trace import (
    RETRY_TRACE_FILENAME,
    execute_checkpoint_trace,
    failure_report,
    trace_preflight,
    write_json_exclusive,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT_RUN = Path("/root/g1_true23_runs/sonic_task_space_ppo_seed20260805_v2")


def _bind_local_runtime_sources(root: Path) -> dict[str, str]:
    unitree = (root / "external_dependencies" / "unitree_rl_mjlab").resolve(strict=True)
    mjlab = (root / "external_dependencies" / "mjlab").resolve(strict=True)
    unitree_text = str(unitree)
    if unitree_text not in sys.path:
        sys.path.insert(0, unitree_text)
    import importlib

    src = importlib.import_module("src")
    mjlab_module = importlib.import_module("mjlab")
    expected_src = (unitree / "src" / "__init__.py").resolve(strict=True)
    expected_mjlab = (mjlab / "src" / "mjlab" / "__init__.py").resolve(strict=True)
    if Path(src.__file__).resolve() != expected_src or Path(mjlab_module.__file__).resolve() != expected_mjlab:
        raise RuntimeError("checkpoint trace local runtime import binding mismatch")
    return {"unitree_task_init": str(expected_src), "mjlab_init": str(expected_mjlab)}


def _configure_simulator_runtime(root: Path) -> dict[str, str]:
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("checkpoint trace requires WORLD_SIZE=1")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    sources = _bind_local_runtime_sources(root)
    from mjlab.utils.torch import configure_torch_backends

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("checkpoint trace requires fixed visible CUDA device 0")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    return sources


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "trace"):
        child = subparsers.add_parser(name)
        child.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
        child.add_argument("--parent-run-dir", type=Path, default=DEFAULT_PARENT_RUN)
        if name == "trace":
            child.add_argument("--output-json", type=Path, required=True)
    return parser


def _success_receipt(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "trace_complete": True,
        "output_json": str(resolved),
        "report_file_sha256": sha256_file(resolved),
        "training_updates": 0,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def _validate_retry_output(parent_run_dir: Path, output: Path) -> Path:
    parent_run = parent_run_dir.expanduser().resolve(strict=True)
    requested = output.expanduser().absolute()
    if requested.name != RETRY_TRACE_FILENAME or requested.parent.resolve(strict=True) != parent_run.parent:
        raise ValueError(
            f"checkpoint trace retry output must be immutable sibling {parent_run.parent / RETRY_TRACE_FILENAME}"
        )
    return requested


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        report = trace_preflight(args.repository_root, args.parent_run_dir)
        print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    try:
        output = _validate_retry_output(args.parent_run_dir, args.output_json)
    except Exception as error:
        print(f"checkpoint trace output rejected: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    if os.path.lexists(output):
        print(f"checkpoint trace refused overwrite: {output}")  # noqa: T201
        return 1
    try:
        root = args.repository_root.expanduser().resolve(strict=True)
        _configure_simulator_runtime(root)
        report = execute_checkpoint_trace(root, args.parent_run_dir)
        write_json_exclusive(output, report)
    except Exception as error:
        try:
            if output.parent.is_dir() and not output.parent.is_symlink() and not os.path.lexists(output):
                write_json_exclusive(output, failure_report(error))
        except Exception as persistence_error:
            print(  # noqa: T201
                "checkpoint trace failure evidence persistence failed: "
                f"{type(persistence_error).__name__}: {persistence_error}"
            )
        print(f"checkpoint trace failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(_success_receipt(output), indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
