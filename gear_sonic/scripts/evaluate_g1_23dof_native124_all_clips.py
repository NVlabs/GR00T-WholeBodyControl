"""Evaluate one fixed GPU environment per corpus clip in a single strict run."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--parallel-rollouts", action="store_true")
    parser.add_argument("--command-lead-frames", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs=3)
    parser.add_argument("--dynamic-phase-schedule", action="store_true")
    args = parser.parse_args()
    if args.rollouts <= 0 or args.command_lead_frames < 0:
        raise ValueError("rollouts must be positive and command lead non-negative")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from src.tasks.tracking.config.g1_23dof.rl_cfg import unitree_g1_23dof_tracking_ppo_runner_cfg
    import torch

    from gear_sonic.envs.mjlab.native124_multi_motion import make_native124_multi_motion_env_cfg

    spans_path = args.spans.resolve(strict=True)
    catalog = json.loads(spans_path.read_text(encoding="utf-8"))
    names = [str(span["name"]) for span in catalog["spans"]]
    configure_torch_backends()
    torch.cuda.set_device(0)
    cfg = make_native124_multi_motion_env_cfg(
        spans_path,
        enable_pushes=False,
        clip_by_env=True,
        deterministic_start_seed=None if args.dynamic_phase_schedule else args.seed,
        command_lead_frames=args.command_lead_frames,
    )
    env_count = len(names) * args.rollouts if args.parallel_rollouts else len(names)
    target_episodes = 1 if args.parallel_rollouts else args.rollouts
    cfg.scene.num_envs = env_count
    cfg.seed = args.seed
    cfg.observations["actor"].enable_corruption = False
    for event in ("base_com", "encoder_bias", "foot_friction"):
        cfg.events.pop(event, None)
    cfg.commands["motion"].pose_range = {}
    cfg.commands["motion"].velocity_range = {}
    cfg.commands["motion"].joint_position_range = (0.0, 0.0)

    agent = unitree_g1_23dof_tracking_ppo_runner_cfg()
    if args.hidden_dims is not None:
        agent.actor.hidden_dims = tuple(args.hidden_dims)
        agent.critic.hidden_dims = tuple(args.hidden_dims)
    agent.seed = args.seed
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    episode_counts = torch.zeros(env_count, dtype=torch.long, device="cuda:0")
    episode_steps = torch.zeros_like(episode_counts)
    completed = torch.zeros_like(episode_counts)
    failure_steps: list[list[int]] = [[] for _ in range(env_count)]
    start_reference_steps: list[list[int]] = [[] for _ in range(env_count)]
    terminal_reference_steps: list[list[int]] = [[] for _ in range(env_count)]
    current_starts = torch.full_like(episode_counts, -1)
    failure_counts = {
        term: torch.zeros(env_count, dtype=torch.long, device="cuda:0")
        for term in ("anchor_pos", "anchor_ori", "ee_body_pos")
    }
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
        env.command_manager.compute(dt=0.0)
        command = env.command_manager.get_term("motion")
        observations = wrapped.get_observations()
        with torch.inference_mode():
            for _ in range(target_episodes * 500):
                starting = episode_steps == 0
                current_starts[starting] = command.time_steps[starting]
                reference_steps = command.time_steps.clone()
                observations, _, dones, extras = wrapped.step(policy(observations))
                episode_steps += 1
                active = episode_counts < target_episodes
                dones = dones.bool() & active
                timeouts = extras.get("time_outs", env.termination_manager.time_outs).bool()
                failed = dones & ~timeouts
                succeeded = dones & timeouts
                completed += succeeded.long()
                for term_name, counts in failure_counts.items():
                    counts += (env.termination_manager.get_term(term_name) & failed).long()
                for env_id in dones.nonzero(as_tuple=False).flatten().cpu().tolist():
                    failure_steps[env_id].append(-1 if bool(succeeded[env_id]) else int(episode_steps[env_id] - 1))
                    start_reference_steps[env_id].append(int(current_starts[env_id]))
                    terminal_reference_steps[env_id].append(int(reference_steps[env_id]))
                episode_counts += dones.long()
                episode_steps[dones] = 0
        if bool((episode_counts != target_episodes).any()):
            raise RuntimeError(f"incomplete rollout counts: {episode_counts.cpu().tolist()}")
    finally:
        env.close()

    args.output.mkdir(parents=True, exist_ok=True)
    perfect = 0
    for index, name in enumerate(names):
        env_ids = (
            list(range(index, env_count, len(names))) if args.parallel_rollouts else [index]
        )
        span_start = int(catalog["spans"][index]["start"])
        complete = int(completed[env_ids].sum().item())
        perfect += complete == args.rollouts
        report = {
            "schema": "g1_true23_native124_incremental_eval_v1",
            "checkpoint": str(args.checkpoint.resolve()),
            "clip": name,
            "rollouts": args.rollouts,
            "seed": args.seed,
            "phase_schedule": (
                "dynamic_legacy"
                if args.dynamic_phase_schedule
                else (
                    "per_environment_stateless_v1_parallel"
                    if args.parallel_rollouts
                    else "per_environment_stateless_v1"
                )
            ),
            "pushes": False,
            "command_lead_frames": args.command_lead_frames,
            "completed_rollouts": complete,
            "completion_rate": complete / args.rollouts,
            "first_failure_steps": [step for env_id in env_ids for step in failure_steps[env_id]],
            "start_reference_steps": [
                step - span_start for env_id in env_ids for step in start_reference_steps[env_id]
            ],
            "terminal_reference_steps": [
                step - span_start
                for env_id in env_ids
                for step in terminal_reference_steps[env_id]
            ],
            "failure_counts": {
                term: int(counts[env_ids].sum().item()) for term, counts in failure_counts.items()
            },
        }
        (args.output / f"{name}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"{name}: {complete}/{args.rollouts}")
    print(f"perfect: {perfect}/{len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
