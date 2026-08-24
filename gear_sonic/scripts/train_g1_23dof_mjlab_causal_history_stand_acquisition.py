"""Train the separate neutral-stand causal acquisition v3 lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from gear_sonic.envs.mjlab import sonic_true23_causal_history as causal_task
from gear_sonic.envs.mjlab.sonic_true23_causal_history_stand_acquisition import (
    causal_stand_acquisition_contract,
    make_causal_history_stand_acquisition_env_cfg,
)
from gear_sonic.scripts import (
    train_g1_23dof_mjlab_causal_history as base,
    train_g1_23dof_mjlab_causal_history_acquisition as acquisition,
)
from gear_sonic.scripts.build_g1_true23_causal_neutral_acquisition_motion import (
    DEFAULT_OUTPUT as DEFAULT_MOTION,
)

DEFAULT_METADATA = DEFAULT_MOTION.with_suffix(".json")
DEFAULT_RUN = Path("/root/g1_true23_runs/causal_history_stand_acquisition_v3")
_THIS_FILE = Path(__file__).resolve()
_PACKAGE_ROOT = _THIS_FILE.parents[1]
_SOURCE_FILES = (
    _PACKAGE_ROOT
    / "envs"
    / "mjlab"
    / "sonic_true23_causal_history_stand_acquisition.py",
    _THIS_FILE,
    _THIS_FILE.parent / "build_g1_true23_causal_neutral_acquisition_motion.py",
)


def _install_isolated_v3_hooks() -> None:
    acquisition._install_isolated_v2_hooks()
    causal_task.make_causal_history_recovery_env_cfg = (
        make_causal_history_stand_acquisition_env_cfg
    )
    source_files = list(base.CAUSAL_SOURCE_FILES)
    for path in _SOURCE_FILES:
        if path not in source_files:
            source_files.append(path)
    base.CAUSAL_SOURCE_FILES = tuple(source_files)

    original = base._resolved_training_config

    def resolved_with_stand(*args: Any, **kwargs: Any) -> dict[str, Any]:
        resolved = original(*args, **kwargs)
        resolved["causal_stand_acquisition_v3"] = (
            causal_stand_acquisition_contract()
        )
        return resolved

    base._resolved_training_config = resolved_with_stand


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    report = base.preflight(args)
    return {
        **report,
        "schema": "g1_true23_causal_stand_acquisition_preflight_v3",
        "causal_stand_acquisition_v3": causal_stand_acquisition_contract(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = base._parser()
    for action in parser._subparsers._group_actions:  # noqa: SLF001
        for subparser in action.choices.values():
            subparser.set_defaults(
                run_dir=DEFAULT_RUN,
                motion_file=DEFAULT_MOTION,
                motion_metadata=DEFAULT_METADATA,
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _install_isolated_v3_hooks()
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
