"""Train a separate true23 causal acquisition lineage after collapse audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from gear_sonic.envs.mjlab import sonic_true23_causal_history as causal_task
from gear_sonic.envs.mjlab.sonic_true23_causal_history_acquisition import (
    causal_acquisition_contract,
    make_causal_history_acquisition_env_cfg,
)
from gear_sonic.scripts import train_g1_23dof_mjlab_causal_history as base

DEFAULT_RUN = Path("/root/g1_true23_runs/causal_history_acquisition_v2")
_THIS_FILE = Path(__file__).resolve()
_TASK_FILE = (
    _THIS_FILE.parents[1]
    / "envs"
    / "mjlab"
    / "sonic_true23_causal_history_acquisition.py"
)


def _install_isolated_v2_hooks() -> None:
    """Route the base audited trainer through the source-bound v2 task."""

    causal_task.make_causal_history_recovery_env_cfg = (
        make_causal_history_acquisition_env_cfg
    )
    source_files = list(base.CAUSAL_SOURCE_FILES)
    for path in (_TASK_FILE, _THIS_FILE):
        if path not in source_files:
            source_files.append(path)
    base.CAUSAL_SOURCE_FILES = tuple(source_files)

    original = base._resolved_training_config

    def resolved_with_acquisition(*args: Any, **kwargs: Any) -> dict[str, Any]:
        resolved = original(*args, **kwargs)
        resolved["causal_acquisition_v2"] = causal_acquisition_contract()
        return resolved

    base._resolved_training_config = resolved_with_acquisition


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    report = base.preflight(args)
    return {
        **report,
        "schema": "g1_true23_causal_history_acquisition_preflight_v2",
        "causal_acquisition_v2": causal_acquisition_contract(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = base._parser()
    for action in parser._subparsers._group_actions:  # noqa: SLF001
        for subparser in action.choices.values():
            subparser.set_defaults(run_dir=DEFAULT_RUN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _install_isolated_v2_hooks()
    if args.mode == "preflight":
        report = preflight(args)
        output = json.dumps(report, indent=2, sort_keys=True)
        print(output)  # noqa: T201
        if args.json_output is not None:
            destination = args.json_output.expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(output + "\n", encoding="utf-8")
        return 0 if report["ready"] else 2
    run_dir = base.run_training(args)
    print(run_dir)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
