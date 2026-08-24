"""Run exact in-memory module14/module16 full-support line-search trace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

from gear_sonic.utils import (
    g1_true23_sonic_task_space_ppo_full_support_delta_line_search as line_search_utils,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file

OUTPUT_FILENAME = line_search_utils.OUTPUT_FILENAME
load_line_search_contract = line_search_utils.load_line_search_contract
line_search_preflight = line_search_utils.line_search_preflight
resolve_line_search_inputs = line_search_utils.resolve_line_search_inputs
validate_publication_provenance = line_search_utils.validate_publication_provenance
SOURCE_ABLATION_REPORT_SHA256 = line_search_utils.SOURCE_ABLATION_REPORT_SHA256
SOURCE_ABLATION_REPORT_PROVENANCE_SHA256 = line_search_utils.SOURCE_ABLATION_REPORT_PROVENANCE_SHA256
failure_report = line_search_utils.failure_report
write_json_exclusive = line_search_utils.write_json_exclusive
execute_line_search = line_search_utils.execute_line_search

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
        raise RuntimeError("line-search trace local runtime binding mismatch")
    return {"unitree_task_init": str(expected_src), "mjlab_init": str(expected_mjlab)}


def _configure_simulator_runtime(root: Path) -> dict[str, str]:
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("line-search trace requires WORLD_SIZE=1")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    sources = _bind_local_runtime_sources(root)
    from mjlab.utils.torch import configure_torch_backends

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("line-search trace requires fixed visible CUDA device 0")
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


def _expected_output(repository_root: Path) -> Path:
    contract = load_line_search_contract(repository_root)
    return Path(contract["output"]["linux_path"]).expanduser().absolute()


def _strict_directory(path: Path, *, context: str) -> Path:
    expanded = path.expanduser().absolute()
    if not expanded.exists():
        raise ValueError(f"{context} missing: {expanded}")
    if not expanded.is_dir():
        raise ValueError(f"{context} must be a directory: {expanded}")
    resolved = expanded.resolve(strict=True)
    if expanded != resolved or expanded.is_symlink():
        raise ValueError(f"{context} must be a regular directory without symlink drift: {expanded}")
    return resolved


def _validate_output(repository_root: Path, parent_run_dir: Path, output: Path) -> Path:
    contract = load_line_search_contract(repository_root)
    expected = _expected_output(repository_root)
    if expected.name != OUTPUT_FILENAME:
        raise ValueError(f"line-search contract output filename mismatch: {expected}")
    expected_parent = _strict_directory(
        Path(contract["parent_run"]["linux_path"]), context="line-search parent run"
    )
    parent_requested = _strict_directory(parent_run_dir, context="line-search requested parent run")
    if parent_requested != expected_parent:
        raise ValueError(f"line-search parent run mismatch: {parent_requested} != {expected_parent}")

    requested = output.expanduser().absolute()
    if requested != expected or requested.name != OUTPUT_FILENAME or requested.parent != expected.parent:
        raise ValueError(f"line-search output must be exact sibling {expected}")
    if requested.parent != parent_requested.parent:
        raise ValueError(f"line-search output parent mismatch: {requested.parent} != {parent_requested.parent}")
    if requested.parent.is_symlink() or requested.parent != _strict_directory(
        requested.parent, context="line-search output parent"
    ):
        raise ValueError(f"line-search output parent must be regular directory: {requested.parent}")
    if os.path.lexists(requested):
        raise ValueError(f"line-search output already exists: {requested}")
    return requested


def _success_receipt(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "line_search_complete": True,
        "diagnostic_only": True,
        "output": str(resolved),
        "output_json": str(resolved),
        "hash": sha256_file(resolved),
        "report_file_sha256": sha256_file(resolved),
        "optimizer_steps": 0,
        "training_transitions": 0,
        "training_updates": 0,
        "checkpoints_written": 0,
        "candidate_selected": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
        "teacher_labels_used": False,
        "teacher_queries": 0,
        "failed_model5_loaded": False,
        "failed_model5_resumed": False,
        "robot_or_network_commands_permitted": False,
        "robot_commands_issued": 0,
        "network_commands_issued": 0,
        "hardware_actions": 0,
    }


def _strict_json(path: Path, context: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            payload[key] = value
        return payload

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{context} contains non-finite JSON token {token}")
        ),
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be a JSON object")
    return payload


def _validate_success_receipt(path: Path, repository_root: Path, parent_run_dir: Path) -> str:
    report = _strict_json(path, "line-search receipt")
    return validate_publication_provenance(repository_root, parent_run_dir, report)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = args.repository_root.expanduser().resolve(strict=True)
    except Exception as error:
        print(f"line-search repository root resolution failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    if args.command == "preflight":
        try:
            report = line_search_preflight(root, args.parent_run_dir)
        except Exception as error:
            print(f"line-search preflight failed: {type(error).__name__}: {error}")  # noqa: T201
            return 1
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    try:
        output = _validate_output(root, args.parent_run_dir, args.output_json)
    except Exception as error:
        print(f"line-search output rejected: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    failure_stage = "input_resolution"
    provenance_snapshot_sha256: str | None = None
    try:
        inputs = resolve_line_search_inputs(root, args.parent_run_dir)
        provenance_snapshot_sha256 = str(inputs.provenance["snapshot_sha256"])
        failure_stage = "runtime_configuration"
        _configure_simulator_runtime(root)
        failure_stage = "simulator_execution"
        report = execute_line_search(root, args.parent_run_dir)
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
                        source_delta_ablation_report_sha256=SOURCE_ABLATION_REPORT_SHA256,
                        source_delta_ablation_report_provenance_snapshot_sha256=SOURCE_ABLATION_REPORT_PROVENANCE_SHA256,
                    ),
                )
        except Exception as persistence_error:
            print(  # noqa: T201
                f"line-search failure persistence failed: {type(persistence_error).__name__}: {persistence_error}"
            )
        print(f"line-search failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(_success_receipt(output), indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
