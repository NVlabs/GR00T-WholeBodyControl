"""Train low-exploration nominal causal stand acquisition v6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from gear_sonic.scripts import (
    train_g1_23dof_mjlab_causal_history as base,
    train_g1_23dof_mjlab_causal_history_stand_acquisition_v5 as v5,
)
from gear_sonic.scripts.build_g1_true23_causal_neutral_acquisition_motion import (
    DEFAULT_OUTPUT as DEFAULT_MOTION,
)
from gear_sonic.trl.mjlab import causal_history_runner as runner_module
from gear_sonic.trl.mjlab.causal_stand_acquisition_runner_v6 import (
    ACQUISITION_ENTROPY_COEFFICIENT,
    ACQUISITION_EXPLORATION_STD,
    CausalStandAcquisitionRunnerV6,
)

DEFAULT_METADATA = DEFAULT_MOTION.with_suffix(".json")
DEFAULT_RUN = Path("/root/g1_true23_runs/causal_history_stand_acquisition_v6")
_THIS_FILE = Path(__file__).resolve()
_RUNNER_FILE = (
    _THIS_FILE.parents[1]
    / "trl"
    / "mjlab"
    / "causal_stand_acquisition_runner_v6.py"
)


def causal_stand_acquisition_v6_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_causal_stand_acquisition_v6",
        "exploration_std_before_first_rollout": ACQUISITION_EXPLORATION_STD,
        "exploration_std_applies_to": "training_gaussian_only",
        "network_weights_changed_before_first_rollout": False,
        "exploration_state_change_declared_in_resolved_lineage": True,
        "entropy_coefficient": ACQUISITION_ENTROPY_COEFFICIENT,
        "strict_ee_threshold_m": 0.25,
        "actual_joint_limit_penalty_weight": -20.0,
        "action_target_reference_penalty_weight": -2.0,
        "action_target_soft_limit_barrier_weight": -10.0,
        "target_barrier_maximum_normalized_penalty": 0.25,
        "fixed_start_no_wrap_guard_preserved": True,
        "rsi_noise_disabled": True,
        "nominal_acquisition_only": True,
        "deployment_ready": False,
    }


def _install_isolated_v6_hooks() -> None:
    v5._install_isolated_v5_hooks()
    source_files = list(base.CAUSAL_SOURCE_FILES)
    for path in (_RUNNER_FILE, _THIS_FILE):
        if path not in source_files:
            source_files.append(path)
    base.CAUSAL_SOURCE_FILES = tuple(source_files)
    runner_module.CausalHistoryMjlabOnPolicyRunner = CausalStandAcquisitionRunnerV6

    original = base._resolved_training_config

    def resolved_with_v6(*args: Any, **kwargs: Any) -> dict[str, Any]:
        agent_cfg = kwargs.get("agent_cfg")
        if agent_cfg is None:
            raise ValueError("v6 resolved config requires agent_cfg")
        agent_cfg.algorithm.entropy_coef = ACQUISITION_ENTROPY_COEFFICIENT
        resolved = original(*args, **kwargs)
        resolved["causal_stand_acquisition_v6"] = (
            causal_stand_acquisition_v6_contract()
        )
        return resolved

    base._resolved_training_config = resolved_with_v6


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    report = base.preflight(args)
    return {
        **report,
        "schema": "g1_true23_causal_stand_acquisition_preflight_v6",
        "causal_stand_acquisition_v6": causal_stand_acquisition_v6_contract(),
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
    _install_isolated_v6_hooks()
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
