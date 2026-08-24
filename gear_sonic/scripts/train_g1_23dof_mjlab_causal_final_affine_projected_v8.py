"""Run bounded final-affine causal stand fine-tuning v8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from gear_sonic.scripts import (
    train_g1_23dof_mjlab_causal_history as base,
    train_g1_23dof_mjlab_causal_history_stand_acquisition_v6 as v6,
)
from gear_sonic.scripts.build_g1_true23_causal_neutral_acquisition_motion import (
    DEFAULT_OUTPUT as DEFAULT_MOTION,
)
from gear_sonic.trl.mjlab import causal_history_runner as runner_module
from gear_sonic.trl.mjlab.causal_final_affine_projected_runner_v8 import (
    ALLOWED_CHECKPOINT_UPDATES,
    CALIBRATION_KL_LIMIT,
    FIXED_LEARNING_RATE,
    PPO_CLIP,
    PROJECTION_TARGET_KL,
    CausalFinalAffineProjectedRunnerV8,
)

DEFAULT_METADATA = DEFAULT_MOTION.with_suffix(".json")
DEFAULT_RUN = Path("/root/g1_true23_runs/causal_final_affine_projected_v8")
_THIS_FILE = Path(__file__).resolve()
_RUNNER_FILE = (
    _THIS_FILE.parents[1]
    / "trl"
    / "mjlab"
    / "causal_final_affine_projected_runner_v8.py"
)


def causal_final_affine_projected_v8_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_causal_final_affine_projected_v8",
        "trainable_actor_parameters": [
            "core.actor_module.decoders.g1_dyn.module.16.bias",
            "core.actor_module.decoders.g1_dyn.module.16.weight",
        ],
        "encoder_and_fsq_frozen": True,
        "decoder_hidden_layers_frozen": True,
        "exploration_std_frozen": True,
        "optimizer_excludes_all_other_actor_parameters": True,
        "learning_rate": FIXED_LEARNING_RATE,
        "schedule": "fixed",
        "ppo_clip": PPO_CLIP,
        "learning_epochs": 1,
        "entropy_coefficient": 0.0,
        "calibration_kl_limit": CALIBRATION_KL_LIMIT,
        "projection_target_kl": PROJECTION_TARGET_KL,
        "projection_basis": "cumulative_final_affine_delta_from_model0",
        "projection_math": "alpha=sqrt(target_kl/candidate_kl)",
        "fail_closed_boundary_frequency_updates": 1,
        "allowed_checkpoint_updates": sorted(ALLOWED_CHECKPOINT_UPDATES),
        "v6_task_and_physical_safety_unchanged": True,
        "deployment_ready": False,
    }


def _install_isolated_v8_hooks() -> None:
    v6._install_isolated_v6_hooks()
    source_files = list(base.CAUSAL_SOURCE_FILES)
    for path in (_RUNNER_FILE, _THIS_FILE):
        if path not in source_files:
            source_files.append(path)
    base.CAUSAL_SOURCE_FILES = tuple(source_files)
    runner_module.CausalHistoryMjlabOnPolicyRunner = (
        CausalFinalAffineProjectedRunnerV8
    )
    original = base._resolved_training_config

    def resolved_with_v8(*args: Any, **kwargs: Any) -> dict[str, Any]:
        agent_cfg = kwargs.get("agent_cfg")
        if agent_cfg is None:
            raise ValueError("v8 resolved config requires agent_cfg")
        algorithm = agent_cfg.algorithm
        algorithm.learning_rate = FIXED_LEARNING_RATE
        algorithm.schedule = "fixed"
        algorithm.clip_param = PPO_CLIP
        algorithm.num_learning_epochs = 1
        algorithm.entropy_coef = 0.0
        algorithm.max_grad_norm = 0.1
        agent_cfg.save_interval = 1
        resolved = original(*args, **kwargs)
        resolved["causal_final_affine_projected_v8"] = (
            causal_final_affine_projected_v8_contract()
        )
        return resolved

    base._resolved_training_config = resolved_with_v8


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    report = base.preflight(args)
    return {
        **report,
        "schema": "g1_true23_causal_final_affine_projected_preflight_v8",
        "causal_final_affine_projected_v8": (
            causal_final_affine_projected_v8_contract()
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
    _install_isolated_v8_hooks()
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
