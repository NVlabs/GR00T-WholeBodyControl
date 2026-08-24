"""Incrementally fine-tune the proven native124 DadDance actor on a motion corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ee-threshold", type=float, default=0.25)
    parser.add_argument("--anchor-threshold", type=float, default=0.25)
    parser.add_argument("--orientation-threshold", type=float, default=0.8)
    parser.add_argument("--ee-reward-weight", type=float, default=0.0)
    parser.add_argument("--ee-reward-std", type=float, default=0.15)
    parser.add_argument("--root-reward-weight", type=float, default=0.5)
    parser.add_argument("--root-reward-std", type=float, default=0.3)
    parser.add_argument("--command-lead-frames", type=int, default=0)
    parser.add_argument("--only-clip", type=str)
    parser.add_argument("--reference-action-blend", type=float, default=0.0)
    parser.add_argument("--reference-leg-blend", type=float, default=0.0)
    parser.add_argument("--disable-pushes", action="store_true")
    parser.add_argument("--hidden-dims", type=int, nargs=3)
    parser.add_argument("--skip-optimizer-load", action="store_true")
    parser.add_argument("--freeze-actor-core-dims", type=int, nargs=3)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.save_interval) <= 0 or args.learning_rate <= 0:
        raise ValueError("training arguments must be positive")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from src.tasks.tracking.config.g1_23dof.rl_cfg import unitree_g1_23dof_tracking_ppo_runner_cfg
    import torch

    from gear_sonic.envs.mjlab.native124_multi_motion import (
        blend_reference_actions,
        blend_reference_leg_actions,
        make_native124_multi_motion_env_cfg,
    )
    from gear_sonic.utils.g1_23dof_incremental_corpus import load_catalog

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    spans = args.spans.expanduser().resolve(strict=True)
    catalog = load_catalog(spans)
    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    configure_torch_backends()
    torch.cuda.set_device(0)
    env_cfg = make_native124_multi_motion_env_cfg(
        spans,
        ee_termination_threshold=args.ee_threshold,
        anchor_termination_threshold=args.anchor_threshold,
        orientation_termination_threshold=args.orientation_threshold,
        enable_pushes=not args.disable_pushes,
        ee_reward_weight=args.ee_reward_weight,
        ee_reward_std=args.ee_reward_std,
        root_reward_weight=args.root_reward_weight,
        root_reward_std=args.root_reward_std,
        command_lead_frames=args.command_lead_frames,
        only_clip=args.only_clip,
    )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg = unitree_g1_23dof_tracking_ppo_runner_cfg()
    if args.hidden_dims is not None:
        agent_cfg.actor.hidden_dims = tuple(args.hidden_dims)
        agent_cfg.critic.hidden_dims = tuple(args.hidden_dims)
    agent_cfg.seed = args.seed
    agent_cfg.save_interval = args.save_interval
    agent_cfg.max_iterations = args.iterations
    agent_cfg.algorithm.learning_rate = args.learning_rate
    agent_cfg.algorithm.schedule = "fixed"
    agent_cfg.logger = "tensorboard"
    agent_cfg.run_name = run_dir.name

    resolved = {
        "schema": "g1_true23_native124_incremental_training_v1",
        "checkpoint": str(checkpoint),
        "corpus": str(catalog.corpus_path),
        "corpus_sha256": catalog.corpus_sha256,
        "spans": [asdict(span) for span in catalog.spans],
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "save_interval": args.save_interval,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "ee_termination_threshold": args.ee_threshold,
        "anchor_termination_threshold": args.anchor_threshold,
        "orientation_termination_threshold": args.orientation_threshold,
        "pushes_enabled": not args.disable_pushes,
        "ee_reward_weight": args.ee_reward_weight,
        "ee_reward_std": args.ee_reward_std,
        "root_reward_weight": args.root_reward_weight,
        "root_reward_std": args.root_reward_std,
        "command_lead_frames": args.command_lead_frames,
        "only_clip": args.only_clip,
        "reference_action_blend": args.reference_action_blend,
        "reference_leg_blend": args.reference_leg_blend,
        "observation_dim": 124,
        "action_dim": 23,
        "hidden_dims": list(agent_cfg.actor.hidden_dims),
    }
    (run_dir / "resolved_training.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    try:
        class ReferenceBlendWrapper(RslRlVecEnvWrapper):
            def step(self, actions: torch.Tensor):
                action_term = self.unwrapped.action_manager.get_term("joint_pos")
                command = self.unwrapped.command_manager.get_term("motion")
                blended = blend_reference_actions(
                    actions,
                    command.joint_pos,
                    action_term.scale,
                    action_term.offset,
                    args.reference_action_blend,
                )
                blended = blend_reference_leg_actions(
                    blended,
                    command.joint_pos,
                    action_term.scale,
                    action_term.offset,
                    args.reference_leg_blend,
                )
                return super().step(blended)

        wrapped = ReferenceBlendWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), str(run_dir), "cuda:0")
        load_cfg = None
        if args.skip_optimizer_load:
            load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": True}
        runner.load(str(checkpoint), load_cfg=load_cfg, map_location="cuda:0")
        if args.freeze_actor_core_dims is not None:
            core_dims = tuple(args.freeze_actor_core_dims)
            actor = runner.alg.get_policy()
            parameter_by_name = dict(actor.named_parameters())

            def mask_rows(name: str, core: int) -> None:
                parameter = parameter_by_name[name]
                mask = torch.ones_like(parameter)
                mask[:core] = 0.0
                parameter.register_hook(lambda gradient, mask=mask: gradient * mask)

            for layer, core in zip((0, 2, 4), core_dims, strict=True):
                mask_rows(f"mlp.{layer}.weight", core)
                mask_rows(f"mlp.{layer}.bias", core)
            output_weight = parameter_by_name["mlp.6.weight"]
            output_mask = torch.ones_like(output_weight)
            output_mask[:, : core_dims[-1]] = 0.0
            output_weight.register_hook(lambda gradient: gradient * output_mask)
            parameter_by_name["mlp.6.bias"].requires_grad_(False)
        runner.alg.learning_rate = args.learning_rate
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = args.learning_rate
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        runner.export_policy_to_onnx(str(run_dir), "policy.onnx")
    finally:
        env.close()
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
