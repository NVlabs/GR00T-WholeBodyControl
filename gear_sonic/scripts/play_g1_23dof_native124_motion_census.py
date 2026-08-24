"""Open one MuJoCo playlist environment per native124 census motion."""

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
    parser.add_argument("--legend", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.viewer import NativeMujocoViewer
    from src.tasks.tracking.config.g1_23dof.rl_cfg import unitree_g1_23dof_tracking_ppo_runner_cfg
    import torch

    from gear_sonic.envs.mjlab.native124_multi_motion import make_native124_multi_motion_env_cfg

    spans_path = args.spans.resolve(strict=True)
    catalog = json.loads(spans_path.read_text(encoding="utf-8"))
    names = [str(span["name"]) for span in catalog["spans"]]
    legend = args.legend or spans_path.with_name("viewer_legend.json")
    legend_data = {
        "controls": {",": "previous", ".": "next", "A": "show all"},
        "environments": {str(index + 1): name for index, name in enumerate(names)},
    }
    legend.write_text(
        json.dumps(legend_data, indent=2),
        encoding="utf-8",
    )

    configure_torch_backends()
    torch.cuda.set_device(0)
    cfg = make_native124_multi_motion_env_cfg(
        spans_path,
        enable_pushes=False,
        clip_by_env=True,
    )
    cfg.scene.num_envs = len(names)
    cfg.observations["actor"].enable_corruption = False
    for event in ("base_com", "encoder_bias", "foot_friction"):
        cfg.events.pop(event, None)
    cfg.commands["motion"].pose_range = {}
    cfg.commands["motion"].velocity_range = {}
    cfg.commands["motion"].joint_position_range = (0.0, 0.0)

    agent = unitree_g1_23dof_tracking_ppo_runner_cfg()
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device="cuda:0")
    runner.load(
        str(args.checkpoint.resolve(strict=True)),
        load_cfg={"actor": True},
        strict=True,
        map_location="cuda:0",
    )
    policy = runner.get_inference_policy(device="cuda:0")
    print(f"61-motion viewer ready. ',' previous, '.' next, 'A' all. Legend: {legend}")
    NativeMujocoViewer(wrapped, policy, frame_rate=50.0).run()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
