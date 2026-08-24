"""Run bounded encoder-only causal stand fine-tuning v7."""

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
from gear_sonic.trl.mjlab.causal_encoder_conservative_runner_v7 import (
    ALLOWED_CHECKPOINT_UPDATES,
    CALIBRATION_KL_LIMIT,
    ENCODER_RELATIVE_L2_LIMIT,
    FIXED_LEARNING_RATE,
    PPO_CLIP,
    CausalEncoderConservativeRunnerV7,
)

DEFAULT_METADATA = DEFAULT_MOTION.with_suffix(".json")
DEFAULT_RUN = Path("/root/g1_true23_runs/causal_encoder_conservative_v7")
_THIS_FILE = Path(__file__).resolve()
_RUNNER_FILE = (
    _THIS_FILE.parents[1]
    / "trl"
    / "mjlab"
    / "causal_encoder_conservative_runner_v7.py"
)


def causal_encoder_conservative_v7_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_causal_encoder_conservative_v7",
        "actor_trainable_prefix": "core.actor_module.encoders.teleop.",
        "decoder_frozen": True,
        "exploration_std_frozen": True,
        "optimizer_excludes_decoder_and_std": True,
        "learning_rate": FIXED_LEARNING_RATE,
        "schedule": "fixed",
        "ppo_clip": PPO_CLIP,
        "learning_epochs": 1,
        "entropy_coefficient": 0.0,
        "calibration_kl_limit": CALIBRATION_KL_LIMIT,
        "encoder_relative_l2_limit": ENCODER_RELATIVE_L2_LIMIT,
        "fail_closed_boundary_frequency_updates": 1,
        "allowed_checkpoint_updates": sorted(ALLOWED_CHECKPOINT_UPDATES),
        "rejected_update_behavior": "poison_process_keep_last_exact_checkpoint",
        "v6_task_and_physical_safety_unchanged": True,
        "deployment_ready": False,
    }


def _install_isolated_v7_hooks() -> None:
    v6._install_isolated_v6_hooks()
    source_files = list(base.CAUSAL_SOURCE_FILES)
    for path in (_RUNNER_FILE, _THIS_FILE):
        if path not in source_files:
            source_files.append(path)
    base.CAUSAL_SOURCE_FILES = tuple(source_files)
    runner_module.CausalHistoryMjlabOnPolicyRunner = (
        CausalEncoderConservativeRunnerV7
    )
    original = base._resolved_training_config

    def resolved_with_v7(*args: Any, **kwargs: Any) -> dict[str, Any]:
        agent_cfg = kwargs.get("agent_cfg")
        if agent_cfg is None:
            raise ValueError("v7 resolved config requires agent_cfg")
        algorithm = agent_cfg.algorithm
        algorithm.learning_rate = FIXED_LEARNING_RATE
        algorithm.schedule = "fixed"
        algorithm.clip_param = PPO_CLIP
        algorithm.num_learning_epochs = 1
        algorithm.entropy_coef = 0.0
        algorithm.max_grad_norm = 0.1
        agent_cfg.save_interval = 1
        resolved = original(*args, **kwargs)
        resolved["causal_encoder_conservative_v7"] = (
            causal_encoder_conservative_v7_contract()
        )
        return resolved

    base._resolved_training_config = resolved_with_v7


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    report = base.preflight(args)
    return {
        **report,
        "schema": "g1_true23_causal_encoder_conservative_preflight_v7",
        "causal_encoder_conservative_v7": (
            causal_encoder_conservative_v7_contract()
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
    _install_isolated_v7_hooks()
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
