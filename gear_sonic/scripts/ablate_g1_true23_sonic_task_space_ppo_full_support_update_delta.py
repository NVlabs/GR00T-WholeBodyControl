"""Run exact in-memory module14/module16 full-support update-delta ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_true23_sonic_task_space_ppo_full_support_update_delta_ablation import (
    OUTPUT_FILENAME,
    ablation_preflight,
    execute_update_delta_ablation,
    failure_report,
    load_ablation_contract,
    resolve_ablation_inputs,
    validate_publication_provenance,
    write_json_exclusive,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT_RUN = Path("/root/g1_true23_runs/sonic_task_space_ppo_full_support_v1_seed20260805")


def _bind_local_runtime_sources(root: Path) -> dict[str, str]:
    unitree = (root / "external_dependencies" / "unitree_rl_mjlab").resolve(strict=True)
    mjlab = (root / "external_dependencies" / "mjlab").resolve(strict=True)
    if str(unitree) not in sys.path:
        sys.path.insert(0, str(unitree))
    import importlib

    src = importlib.import_module("src")
    mjlab_module = importlib.import_module("mjlab")
    expected_src = (unitree / "src" / "__init__.py").resolve(strict=True)
    expected_mjlab = (mjlab / "src" / "mjlab" / "__init__.py").resolve(strict=True)
    if Path(src.__file__).resolve() != expected_src or Path(mjlab_module.__file__).resolve() != expected_mjlab:
        raise RuntimeError("update-delta ablation local runtime binding mismatch")
    return {"unitree_task_init": str(expected_src), "mjlab_init": str(expected_mjlab)}


def _configure_simulator_runtime(root: Path) -> dict[str, str]:
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("update-delta ablation requires WORLD_SIZE=1")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    sources = _bind_local_runtime_sources(root)
    from mjlab.utils.torch import configure_torch_backends

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("update-delta ablation requires fixed visible CUDA device 0")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    return sources


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "ablate"):
        child = subparsers.add_parser(name)
        child.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
        child.add_argument("--parent-run-dir", type=Path, default=DEFAULT_PARENT_RUN)
        if name == "ablate":
            child.add_argument("--output-json", type=Path, required=True)
    return parser


def _expected_output(repository_root: Path) -> Path:
    contract = load_ablation_contract(repository_root)
    return Path(contract["output"]["linux_path"]).expanduser().absolute()


def _validate_output(repository_root: Path, parent_run_dir: Path, output: Path) -> Path:
    parent = parent_run_dir.expanduser().resolve(strict=True)
    requested = output.expanduser().absolute()
    expected = _expected_output(repository_root)
    if (
        requested != expected
        or requested.name != OUTPUT_FILENAME
        or requested.parent.resolve(strict=True) != parent.parent
    ):
        raise ValueError(f"update-delta ablation output must be exact sibling {expected}")
    return requested


def _success_receipt(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "ablation_complete": True,
        "output_json": str(resolved),
        "report_file_sha256": sha256_file(resolved),
        "optimizer_steps": 0,
        "training_transitions": 0,
        "checkpoints_written": 0,
        "candidate_selected": False,
        "support_qualified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def _strict_json(path: Path, context: str) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{context} contains non-finite JSON token {token}")
        ),
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be a JSON object")
    return payload


def _validate_success_receipt(path: Path, repository_root: Path, parent_run_dir: Path) -> str:
    report = _strict_json(path, "ablation receipt")
    return validate_publication_provenance(repository_root, parent_run_dir, report)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.expanduser().resolve(strict=True)
    if args.command == "preflight":
        report = ablation_preflight(root, args.parent_run_dir)
        print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    try:
        output = _validate_output(root, args.parent_run_dir, args.output_json)
    except Exception as error:
        print(f"update-delta ablation output rejected: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    if os.path.lexists(output):
        print(f"update-delta ablation refused overwrite: {output}")  # noqa: T201
        return 1
    failure_stage = "input_resolution"
    provenance_snapshot_sha256: str | None = None
    try:
        inputs = resolve_ablation_inputs(root, args.parent_run_dir)
        provenance_snapshot_sha256 = str(inputs.provenance["snapshot_sha256"])
        failure_stage = "runtime_configuration"
        _configure_simulator_runtime(root)
        failure_stage = "simulator_execution"
        report = execute_update_delta_ablation(root, args.parent_run_dir)
        failure_stage = "prepublication_provenance"
        validate_publication_provenance(root, args.parent_run_dir, report)
        failure_stage = "exclusive_publication"
        write_json_exclusive(
            output,
            report,
            publication_guard=lambda: validate_publication_provenance(
                root,
                args.parent_run_dir,
                report,
            ),
            publication_receipt_guard=lambda: _validate_success_receipt(
                output,
                root,
                args.parent_run_dir,
            ),
        )
    except Exception as error:
        try:
            if output.parent.is_dir() and not output.parent.is_symlink() and not os.path.lexists(output):
                write_json_exclusive(
                    output,
                    failure_report(
                        error,
                        failure_stage=failure_stage,
                        parent_run_dir=str(args.parent_run_dir.expanduser().absolute()),
                        provenance_snapshot_sha256=provenance_snapshot_sha256,
                    ),
                )
        except Exception as persistence_error:
            print(  # noqa: T201
                "update-delta ablation failure persistence failed: "
                f"{type(persistence_error).__name__}: {persistence_error}"
            )
        print(f"update-delta ablation failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(_success_receipt(output), indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
