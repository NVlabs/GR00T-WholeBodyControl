"""Strict deterministic native124 checkpoint evaluation on one corpus clip."""

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
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--pushes", action="store_true")
    parser.add_argument("--ee-threshold", type=float, default=0.25)
    parser.add_argument("--anchor-threshold", type=float, default=0.25)
    parser.add_argument("--orientation-threshold", type=float, default=0.8)
    parser.add_argument("--command-lead-frames", type=int, default=0)
    parser.add_argument("--reference-action-blend", type=float, default=0.0)
    parser.add_argument("--reference-leg-blend", type=float, default=0.0)
    args = parser.parse_args()
    if args.rollouts <= 0:
        raise ValueError("rollouts must be positive")

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

    configure_torch_backends()
    torch.cuda.set_device(0)
    cfg = make_native124_multi_motion_env_cfg(
        args.spans.resolve(strict=True),
        enable_pushes=args.pushes,
        only_clip=args.clip,
        ee_termination_threshold=args.ee_threshold,
        anchor_termination_threshold=args.anchor_threshold,
        orientation_termination_threshold=args.orientation_threshold,
        command_lead_frames=args.command_lead_frames,
    )
    cfg.scene.num_envs = args.rollouts
    cfg.seed = args.seed
    cfg.observations["actor"].enable_corruption = args.pushes
    if not args.pushes:
        for event in ("base_com", "encoder_bias", "foot_friction"):
            cfg.events.pop(event, None)
        cfg.commands["motion"].pose_range = {}
        cfg.commands["motion"].velocity_range = {}
        cfg.commands["motion"].joint_position_range = (0.0, 0.0)

    agent = unitree_g1_23dof_tracking_ppo_runner_cfg()
    agent.seed = args.seed
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    failures = torch.zeros(args.rollouts, dtype=torch.bool, device="cuda:0")
    first_failure_step = torch.full((args.rollouts,), -1, dtype=torch.long, device="cuda:0")
    first_failure_reason = [None] * args.rollouts
    failure_counts: dict[str, int] = {}
    max_errors = {
        name: 0.0
        for name in (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_body_pos",
            "error_joint_pos",
        )
    }
    max_ee_z_errors: dict[str, float] = {}
    first_failure_body = [None] * args.rollouts
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device="cuda:0")
        runner.load(
            str(args.checkpoint.resolve(strict=True)),
            load_cfg={"actor": True},
            strict=True,
            map_location="cuda:0",
        )
        policy = runner.get_inference_policy(device="cuda:0")
        # The stock environment updates the motion-relative body targets after each
        # automatic reset, but its wrapper's one initial reset happens before any
        # command update.  Prime that derived state so the first transition is not
        # falsely terminated against the command term's zero-filled buffers.
        env.command_manager.compute(dt=0.0)
        initial_reference_steps = env.command_manager.get_term("motion").time_steps.clone().cpu().tolist()
        observations = wrapped.get_observations()
        with torch.inference_mode():
            for step in range(500):
                actions = policy(observations)
                action_term = env.action_manager.get_term("joint_pos")
                command = env.command_manager.get_term("motion")
                actions = blend_reference_actions(
                    actions,
                    command.joint_pos,
                    action_term.scale,
                    action_term.offset,
                    args.reference_action_blend,
                )
                actions = blend_reference_leg_actions(
                    actions,
                    command.joint_pos,
                    action_term.scale,
                    action_term.offset,
                    args.reference_leg_blend,
                )
                observations, _, dones, extras = wrapped.step(actions)
                dones = dones.bool()
                timeouts = extras.get("time_outs", env.termination_manager.time_outs).bool()
                failed_now = dones & ~timeouts & ~failures
                first_failure_step[failed_now] = step
                for term_name in env.termination_manager.active_terms:
                    term_cfg = env.termination_manager.get_term_cfg(term_name)
                    if term_cfg.time_out:
                        continue
                    term_failed = env.termination_manager.get_term(term_name) & failed_now
                    failure_counts[term_name] = failure_counts.get(term_name, 0) + int(term_failed.sum().item())
                    for env_id in term_failed.nonzero(as_tuple=False).flatten().cpu().tolist():
                        previous = first_failure_reason[env_id]
                        first_failure_reason[env_id] = term_name if previous is None else f"{previous}+{term_name}"
                failures |= failed_now
                command = env.command_manager.get_term("motion")
                ee_names = cfg.terminations["ee_body_pos"].params["body_names"]
                ee_indexes = [command.cfg.body_names.index(name) for name in ee_names]
                ee_z_errors = torch.abs(
                    command.body_pos_relative_w[:, ee_indexes, -1]
                    - command.robot_body_pos_w[:, ee_indexes, -1]
                )
                for body_index, body_name in enumerate(ee_names):
                    max_ee_z_errors[body_name] = max(
                        max_ee_z_errors.get(body_name, 0.0),
                        float(ee_z_errors[:, body_index].max().item()),
                    )
                for env_id in failed_now.nonzero(as_tuple=False).flatten().cpu().tolist():
                    first_failure_body[env_id] = ee_names[int(ee_z_errors[env_id].argmax().item())]
                for name in max_errors:
                    max_errors[name] = max(max_errors[name], float(command.metrics[name].max().item()))
    finally:
        env.close()

    report = {
        "schema": "g1_true23_native124_incremental_eval_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "clip": args.clip,
        "rollouts": args.rollouts,
        "seed": args.seed,
        "pushes": args.pushes,
        "termination_thresholds": {
            "ee_body_pos": args.ee_threshold,
            "anchor_pos": args.anchor_threshold,
            "anchor_ori": args.orientation_threshold,
        },
        "command_lead_frames": args.command_lead_frames,
        "reference_action_blend": args.reference_action_blend,
        "reference_leg_blend": args.reference_leg_blend,
        "completed_rollouts": int((~failures).sum().item()),
        "completion_rate": float((~failures).float().mean().item()),
        "first_failure_steps": first_failure_step.cpu().tolist(),
        "initial_reference_steps": initial_reference_steps,
        "first_failure_reasons": first_failure_reason,
        "first_failure_bodies": first_failure_body,
        "failure_counts": failure_counts,
        "max_errors": max_errors,
        "max_ee_z_errors": max_ee_z_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not bool(failures.any()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
