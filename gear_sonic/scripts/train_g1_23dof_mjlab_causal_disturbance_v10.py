"""Run 100-update bounded disturbance causal fine-tuning v10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from gear_sonic.scripts import (
    train_g1_23dof_mjlab_causal_disturbance_v9 as v9,
    train_g1_23dof_mjlab_causal_history as base,
)
from gear_sonic.scripts.build_g1_true23_causal_neutral_acquisition_motion import (
    DEFAULT_OUTPUT as DEFAULT_MOTION,
)
from gear_sonic.trl.mjlab import causal_history_runner as runner_module
from gear_sonic.trl.mjlab.causal_final_affine_projected_runner_v10 import (
    V10_ALLOWED_CHECKPOINT_UPDATES,
    V10_FIXED_LEARNING_RATE,
    CausalFinalAffineProjectedRunnerV10,
)

DEFAULT_METADATA = DEFAULT_MOTION.with_suffix(".json")
DEFAULT_RUN = Path("/root/g1_true23_runs/causal_disturbance_v10")
_THIS_FILE = Path(__file__).resolve()
_RUNNER_FILE = (
    _THIS_FILE.parents[1]
    / "trl"
    / "mjlab"
    / "causal_final_affine_projected_runner_v10.py"
)


def causal_final_affine_projected_v10_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_causal_final_affine_projected_v10",
        "restart_from_model0": True,
        "trainable_actor_parameters": [
            "core.actor_module.decoders.g1_dyn.module.16.bias",
            "core.actor_module.decoders.g1_dyn.module.16.weight",
        ],
        "all_other_actor_parameters_and_std_frozen": True,
        "learning_rate": V10_FIXED_LEARNING_RATE,
        "schedule": "fixed",
        "ppo_clip": 0.05,
        "learning_epochs": 1,
        "entropy_coefficient": 0.0,
        "calibration_kl_limit": 2.0e-3,
        "projection_target_kl": 1.8e-3,
        "projection_math_unchanged_from_v8": True,
        "planned_accepted_updates": 100,
        "allowed_checkpoint_updates": sorted(V10_ALLOWED_CHECKPOINT_UPDATES),
        "v9_executed_task_contract_unchanged": True,
        "deployment_ready": False,
    }


def _install_isolated_v10_hooks() -> None:
    v9._install_isolated_v9_hooks()
    runner_module.CausalHistoryMjlabOnPolicyRunner = (
        CausalFinalAffineProjectedRunnerV10
    )
    source_files = list(base.CAUSAL_SOURCE_FILES)
    for path in (_RUNNER_FILE, _THIS_FILE):
        if path not in source_files:
            source_files.append(path)
    base.CAUSAL_SOURCE_FILES = tuple(source_files)
    original = base._resolved_training_config

    def resolved_with_v10(*args: Any, **kwargs: Any) -> dict[str, Any]:
        resolved = original(*args, **kwargs)
        agent_cfg = kwargs.get("agent_cfg")
        if agent_cfg is None:
            raise ValueError("v10 resolved config requires agent_cfg")
        agent_cfg.algorithm.learning_rate = V10_FIXED_LEARNING_RATE
        resolved["agent"]["algorithm"]["learning_rate"] = (
            V10_FIXED_LEARNING_RATE
        )
        resolved.pop("causal_final_affine_projected_v8", None)
        resolved["causal_final_affine_projected_v10"] = (
            causal_final_affine_projected_v10_contract()
        )
        return resolved

    base._resolved_training_config = resolved_with_v10


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    report = v9.preflight(args)
    return {
        **report,
        "schema": "g1_true23_causal_disturbance_preflight_v10",
        "causal_final_affine_projected_v10": (
            causal_final_affine_projected_v10_contract()
        ),
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
    _install_isolated_v10_hooks()
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
