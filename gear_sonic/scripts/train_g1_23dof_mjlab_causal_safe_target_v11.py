"""Train true23 causal V11 with the exact deploy target transform in-loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from gear_sonic.envs.mjlab import sonic_true23_causal_history as causal_task
from gear_sonic.envs.mjlab.sonic_true23_causal_history_safe_target_v11 import (
    audit_causal_history_safe_target_v11_env_cfg,
    causal_history_safe_target_v11_contract,
    make_causal_history_safe_target_v11_env_cfg,
)
from gear_sonic.scripts import (
    train_g1_23dof_mjlab_causal_disturbance_v9 as v9,
    train_g1_23dof_mjlab_causal_history as base,
)
from gear_sonic.scripts.build_g1_true23_causal_neutral_acquisition_motion import (
    DEFAULT_OUTPUT as DEFAULT_MOTION,
)
from gear_sonic.trl.mjlab.causal_final_affine_projected_runner_v8 import (
    CALIBRATION_KL_LIMIT,
    FIXED_LEARNING_RATE,
    PPO_CLIP,
    PROJECTION_TARGET_KL,
)

DEFAULT_METADATA = DEFAULT_MOTION.with_suffix(".json")
DEFAULT_RUN = Path("/root/g1_true23_runs/causal_safe_target_v11")
_THIS_FILE = Path(__file__).resolve()
_ENV_FILE = (
    _THIS_FILE.parents[1]
    / "envs"
    / "mjlab"
    / "sonic_true23_causal_history_safe_target_v11.py"
)
_TRANSFORM_FILE = (
    _THIS_FILE.parents[1]
    / "utils"
    / "g1_23dof_safe_target_transform.py"
)


def causal_safe_target_v11_training_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_causal_safe_target_training_v11",
        "restart_from_model0": True,
        "trainable_actor_parameters": [
            "core.actor_module.decoders.g1_dyn.module.16.bias",
            "core.actor_module.decoders.g1_dyn.module.16.weight",
        ],
        "all_other_actor_parameters_and_std_frozen": True,
        "learning_rate": FIXED_LEARNING_RATE,
        "schedule": "fixed",
        "ppo_clip": PPO_CLIP,
        "learning_epochs": 1,
        "entropy_coefficient": 0.0,
        "calibration_kl_limit": CALIBRATION_KL_LIMIT,
        "projection_target_kl": PROJECTION_TARGET_KL,
        "planned_accepted_updates": 50,
        "v9_push_domain_randomization_and_physical_gates_unchanged": True,
        "safe_target_transform_in_training_loop": True,
        "deployment_ready": False,
    }


def _executed_env_contract(
    *,
    motion_path: Path,
    num_envs: int,
) -> dict[str, object]:
    cfg = make_causal_history_safe_target_v11_env_cfg(
        motion_file=str(motion_path),
        num_envs=num_envs,
        play=False,
    )
    return audit_causal_history_safe_target_v11_env_cfg(cfg)


def _install_isolated_v11_hooks() -> None:
    v9._install_isolated_v9_hooks()
    causal_task.make_causal_history_recovery_env_cfg = (
        make_causal_history_safe_target_v11_env_cfg
    )
    source_files = list(base.CAUSAL_SOURCE_FILES)
    for path in (_ENV_FILE, _TRANSFORM_FILE, _THIS_FILE):
        if path not in source_files:
            source_files.append(path)
    base.CAUSAL_SOURCE_FILES = tuple(source_files)
    original = base._resolved_training_config

    def resolved_with_v11(*args: Any, **kwargs: Any) -> dict[str, Any]:
        resolved = original(*args, **kwargs)
        motion_path = kwargs.get("motion_path")
        num_envs = kwargs.get("num_envs")
        if not isinstance(motion_path, Path) or not isinstance(num_envs, int):
            raise ValueError("v11 resolved config requires motion path and env count")
        resolved["causal_history_safe_target_v11"] = {
            **causal_history_safe_target_v11_contract(),
            "executed_environment": _executed_env_contract(
                motion_path=motion_path,
                num_envs=num_envs,
            ),
        }
        resolved["causal_safe_target_training_v11"] = (
            causal_safe_target_v11_training_contract()
        )
        return resolved

    base._resolved_training_config = resolved_with_v11


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    report = base.preflight(args)
    return {
        **report,
        "schema": "g1_true23_causal_safe_target_preflight_v11",
        "causal_history_safe_target_v11": {
            **causal_history_safe_target_v11_contract(),
            "executed_environment": _executed_env_contract(
                motion_path=args.motion_file.expanduser().resolve(),
                num_envs=args.num_envs,
            ),
        },
        "causal_safe_target_training_v11": (
            causal_safe_target_v11_training_contract()
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
    _install_isolated_v11_hooks()
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
