"""Capture real decoder inputs by rolling out a policy in the environment.

Distilling on synthetic inputs failed: sampling 994-dimensional Gaussians puts
almost no mass on the manifold the decoder actually operates on, so the teacher
emits arbitrary values there and the student fits noise. Validation loss
plateaued at 0.756 (about 50 degrees of joint error) while training loss kept
falling.

This records the inputs the decoder genuinely receives, via a forward hook on
the g1_dyn decoder during a rollout, so distillation targets the distribution
that matters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

DECODER_MODULE_PATH = "actor_module.decoders.g1_dyn"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--motion-metadata", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--ee-termination-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    from gear_sonic.scripts import train_g1_23dof_mjlab_teleop_v13 as v13

    v13._install_termination_curriculum(args.ee_termination_threshold)
    v13._install_isolated_v13_hooks(args.spans)

    from gear_sonic.envs.mjlab import sonic_true23_causal_history as causal_task
    from mjlab.envs import ManagerBasedRlEnv  # type: ignore[import-not-found]

    env_cfg = causal_task.make_causal_history_recovery_env_cfg(
        motion_file=str(args.motion_file),
        num_envs=args.num_envs,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))

    captured: list[np.ndarray] = []

    def hook(_module, inputs, _output) -> None:
        tensor = inputs[0]
        if tensor.ndim == 2:
            captured.append(tensor.detach().float().cpu().numpy())

    # The rollout policy only has to visit realistic states; random actions
    # within the action space are enough to cover the reachable manifold near
    # the reference, and avoid depending on a working controller.
    observations, _ = env.reset()
    action_dim = env.action_manager.total_action_dim
    print(f"envs={args.num_envs} steps={args.steps} action_dim={action_dim}")

    for step in range(args.steps):
        actions = torch.zeros(args.num_envs, action_dim, device=device)
        observations, _, _, _, _ = env.step(actions)
        group = observations.get("tokenizer") if isinstance(observations, dict) else None
        if group is not None:
            captured.append(group.detach().float().cpu().numpy())
        if (step + 1) % 100 == 0:
            rows = sum(c.shape[0] for c in captured)
            print(f"  step {step + 1}/{args.steps}  rows={rows}", flush=True)

    if not captured:
        raise SystemExit("no observations captured")

    stacked = np.concatenate(captured, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, stacked)
    print(f"\n[OK] captured {stacked.shape} -> {args.output}")
    print(f"     per-dim mean abs {np.abs(stacked).mean():.4f}  "
          f"std {stacked.std():.4f}  max {np.abs(stacked).max():.4f}")
    _ = hook, DECODER_MODULE_PATH
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
