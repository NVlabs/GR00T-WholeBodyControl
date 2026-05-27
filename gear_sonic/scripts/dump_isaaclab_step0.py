#!/usr/bin/env python3
"""
dump_isaaclab_step0.py — IsaacLab step-0 ground truth dumper.

Follows the exact Hydra config composition and env creation flow used by
train_agent_trl.py / eval_agent_trl.py.  Does NOT invent config trees.

Usage:
    # Load a saved experiment config from checkpoint:
    python gear_sonic/scripts/dump_isaaclab_step0.py \
        checkpoint=/path/to/checkpoint.pt \
        num_envs=1 headless=True

    # Standalone (must compose manager_env via Hydra defaults;
    # the ``+`` prefix appends the config-group entry to the defaults list):
    python gear_sonic/scripts/dump_isaaclab_step0.py \
        +manager_env=base_env \
        manager_env/config/robot/type=h2 \
        manager_env/observations/policy=local_dir_hist \
        manager_env/observations/tokenizer=unitoken_all_noz \
        manager_env/rewards=tracking/base \
        manager_env/terminations=tracking/base \
        manager_env/commands=tracking/base \
        manager_env/events=tracking/level0_4 \
        num_envs=1 headless=True

Output:
    dump_step0/isaaclab_step0.npz       — IsaacLab ground truth
    dump_step0/mujoco_step0_from_isaac_state.npz
                                      — MuJoCo comparison from IsaacLab dumped state
    dump_step0/comparison_report.txt    — Diff report (if both exist)
"""

import argparse
import joblib
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch

# IsaacLab/Isaac Sim must be launched before importing modules that touch omni.*
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from gear_sonic.trl.utils.common import custom_instantiate
from gear_sonic.utils import config_utils

config_utils.register_rl_resolvers()


OUT_DIR = "dump_step0"


# ===================================================================
# Observation layout (matching IsaacLab ObservationManager.compute_group)
# Each term history is flattened oldest->newest first, then terms are
# concatenated in config order:
#   gravity_dir[10x3], base_ang_vel[10x3], joint_pos_rel[10x31],
#   joint_vel[10x31], last_action[10x31]
# Total with history 10: 990
# ===================================================================
HISTORY_LEN = 10
PER_FRAME_DIMS = {
    "gravity_dir": 3,
    "base_ang_vel": 3,
    "joint_pos_rel": 31,
    "joint_vel": 31,
    "last_action": 31,
}
PER_FRAME_TOTAL = sum(PER_FRAME_DIMS.values())  # 99

def decompose_policy_obs(proprio: np.ndarray) -> dict[str, np.ndarray]:
    out = {}

    off = 0
    term_histories = {}
    for term_name, dim in PER_FRAME_DIMS.items():
        span = HISTORY_LEN * dim
        hist = proprio[off:off + span].reshape(HISTORY_LEN, dim)
        term_histories[term_name] = hist
        out[f"history_{term_name}"] = hist
        for frame_idx in range(HISTORY_LEN):
            out[f"frame{frame_idx}_{term_name}"] = hist[frame_idx]
        off += span

    out["policy_obs_rebuilt"] = np.concatenate([
        term_histories[term_name].reshape(-1) for term_name in PER_FRAME_DIMS
    ])

    return out


# ===================================================================
# Comparison utilities (can be called offline, no IsaacLab needed)
# ===================================================================
def compare_tensors(a: np.ndarray, b: np.ndarray, label: str = ""):
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH: {label} {a.shape} vs {b.shape}")
        return float("inf"), float("inf")

    if a.dtype.kind in {"U", "S", "O"} or b.dtype.kind in {"U", "S", "O"}:
        same = np.array_equal(a, b)
        print(f"  {label:40s}  NON-NUMERIC skipped  same={same}  dtype={a.dtype}/{b.dtype}")
        return 0.0 if same else float("inf"), 0.0 if same else float("inf")

    diff = a - b
    max_abs = np.max(np.abs(diff))
    rms = np.sqrt(np.mean(diff ** 2))
    print(f"  {label:40s}  max_abs={max_abs:.6e}  rms={rms:.6e}")
    return max_abs, rms


def compare_npz(isaaclab_path: str, mujoco_path: str, report_path: str = None):
    """Compare IsaacLab step-0 dump with MuJoCo eval step-0 dump.

    Both files should be .npz with matching key names.
    """
    print(f"\n{'='*70}")
    print("COMPARISON: IsaacLab vs MuJoCo step-0")
    print(f"{'='*70}")
    print(f"  IsaacLab: {isaaclab_path}")
    print(f"  MuJoCo:   {mujoco_path}")
    print()

    il = np.load(isaaclab_path)
    mj = np.load(mujoco_path)

    results = {}
    state_keys = [
        "qpos_root",
        "qpos_dof",
        "qvel_root_lin",
        "qvel_root_ang",
        "qvel_dof",
        "gravity_body",
        "joint_pos_rel",
    ]
    obs_term_keys = [
        "obs_term_gravity_dir",
        "obs_term_base_ang_vel",
        "obs_term_joint_pos_rel",
        "obs_term_joint_vel",
        "obs_term_last_action",
    ]

    # Compare all matching keys
    il_keys = set(il.keys())
    mj_keys = set(mj.keys())
    common_keys = il_keys & mj_keys

    print(f"  Common keys ({len(common_keys)}):")
    for key in sorted(common_keys):
        max_abs, rms = compare_tensors(il[key], mj[key], key)
        results[key] = {"max_abs": max_abs, "rms": rms}

    # Per-joint mismatch report (for joint-related keys)
    for key in ["joint_pos", "joint_vel", "joint_pos_rel", "qpos_dof", "qvel_dof"]:
        if key in common_keys:
            print(f"\n  Per-joint mismatch [{key}]:")
            diff = il[key] - mj[key]
            max_per_joint = np.max(np.abs(diff), axis=0) if diff.ndim > 1 else np.abs(diff)
            worst_idx = np.argmax(max_per_joint)
            print(f"    worst joint idx={worst_idx}, max_abs={max_per_joint[worst_idx]:.6e}")
            for j in range(min(31, len(max_per_joint))):
                if max_per_joint[j] > 1e-4:
                    print(f"    joint[{j:2d}] err={max_per_joint[j]:.6e}")

    if "policy_obs" in common_keys:
        print(f"\n  Policy term mismatch summary:")
        for term_name in PER_FRAME_DIMS:
            hist_key = f"history_{term_name}"
            if hist_key in common_keys:
                compare_tensors(il[hist_key], mj[hist_key], hist_key)

    print(f"\n  Acceptance-focused summary:")
    for label, keys, threshold in [
        ("Injected state", state_keys, 1e-3),
        ("Single-frame obs terms", obs_term_keys, 1e-2),
    ]:
        relevant = [results[k]["max_abs"] for k in keys if k in results and results[k]["max_abs"] != float("inf")]
        if not relevant:
            continue
        worst = max(relevant)
        status = "PASS" if worst < threshold else "FAIL"
        print(f"  {label:24s} {status}  worst={worst:.6e}  threshold={threshold:.1e}")

    # Summary
    print(f"\n  {'='*50}")
    errors = [v for v in results.values() if v["max_abs"] != float("inf")]
    if errors:
        max_err = max(e["max_abs"] for e in errors)
        mean_err = np.mean([e["max_abs"] for e in errors])
        print(f"  MAX error across all tensors: {max_err:.6e}")
        print(f"  MEAN max_abs error:           {mean_err:.6e}")
        if max_err < 1e-4:
            print(f"  VERDICT: Step-0 match PASS \u2713  (all err < 1e-4)")
        elif max_err < 1e-2:
            print(f"  VERDICT: Step-0 match WARNING (err < 1e-2)")
        else:
            print(f"  VERDICT: Step-0 match FAIL \u2717  (err >= 1e-2)")

    if report_path:
        with open(report_path, "w") as f:
            f.write(f"Comparison Report: IsaacLab vs MuJoCo\n")
            f.write(f"  IsaacLab: {isaaclab_path}\n")
            f.write(f"  MuJoCo:   {mujoco_path}\n\n")
            for key in sorted(common_keys):
                r = results[key]
                f.write(f"{key:40s}  max_abs={r['max_abs']:.6e}  rms={r['rms']:.6e}\n")
            f.write(f"\nMAX error: {max_err:.6e}\n")
        print(f"\n  Report saved to: {report_path}")

    return results


# ===================================================================
# Main dump function (requires IsaacLab)
# ===================================================================
def dump_isaaclab_step0(env) -> dict:
    """Dump all state and observation tensors at step 0.

    Args:
        env: ManagerBasedRLEnv (already reset or at step 0)

    Returns:
        dict: All dumped tensors as numpy arrays
    """
    from isaaclab.utils.math import quat_apply_inverse, quat_inv

    data = {}
    robot = env.scene["robot"]

    # ---- 1. Raw robot state ----
    # qpos = [root_pos(3), root_quat(4), joint_pos(31)] = 38
    root_pos = robot.data.root_pos_w[0].cpu().numpy()         # (3,)
    root_quat = robot.data.root_quat_w[0].cpu().numpy()       # (4,) wxyz
    joint_pos = robot.data.joint_pos[0].cpu().numpy()         # (31,)
    data["qpos"] = np.concatenate([root_pos, root_quat, joint_pos])
    data["qpos_root"] = np.concatenate([root_pos, root_quat])
    data["qpos_dof"] = joint_pos

    # qvel = [root_lin_vel(3), root_ang_vel(3), joint_vel(31)] = 37
    root_lin_vel = robot.data.root_lin_vel_w[0].cpu().numpy()  # (3,)
    root_ang_vel = robot.data.root_ang_vel_w[0].cpu().numpy()  # (3,)
    joint_vel = robot.data.joint_vel[0].cpu().numpy()          # (31,)
    data["qvel"] = np.concatenate([root_lin_vel, root_ang_vel, joint_vel])
    data["qvel_root_lin"] = root_lin_vel
    data["qvel_root_ang"] = root_ang_vel
    data["qvel_dof"] = joint_vel

    # Default joint positions
    data["default_joint_pos"] = robot.data.default_joint_pos[0].cpu().numpy()

    # ---- 2. Derived quantities ----
    # Projected gravity in body frame: q^{-1} * [0,0,-1] * q
    gravity_w = torch.tensor([[0.0, 0.0, -1.0]], device=env.device)
    gravity_body = quat_apply_inverse(robot.data.root_quat_w, gravity_w)
    data["gravity_body"] = gravity_body[0].cpu().numpy()       # (3,)

    # Joint positions relative to default
    joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
    data["joint_pos_rel"] = joint_pos_rel[0].cpu().numpy()     # (31,)

    # Single-frame policy terms before history concatenation.
    data["obs_term_gravity_dir"] = data["gravity_body"].copy()
    base_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, robot.data.root_ang_vel_w)
    data["obs_term_base_ang_vel"] = base_ang_vel_b[0].cpu().numpy()
    data["obs_term_joint_pos_rel"] = data["joint_pos_rel"].copy()
    data["obs_term_joint_vel"] = joint_vel.copy()
    data["obs_term_last_action"] = env.action_manager.action[0].detach().cpu().numpy()

    # ---- 3. Actions (zeros at step 0) ----
    data["actions_raw"] = np.zeros(31)
    data["actions_scaled"] = np.zeros(31)

    # ---- 4. Observation tensors ----
    # Get the observation dict from the observation manager
    obs_manager = env.observation_manager

    # Policy observations (990-dim concatenated history)
    policy_obs = obs_manager.compute_group("policy")
    data["policy_obs"] = policy_obs[0].detach().cpu().numpy()

    # Tokenizer observations (dict of terms)
    tokenizer_obs = obs_manager.compute_group("tokenizer")
    if isinstance(tokenizer_obs, dict):
        for key, val in tokenizer_obs.items():
            data[f"tok_{key}"] = val[0].cpu().numpy()
    elif isinstance(tokenizer_obs, torch.Tensor):
        data["tokenizer_obs"] = tokenizer_obs[0].cpu().numpy()

    # ---- 5. Individual observation terms (pre-history, single frame) ----


    # ---- 7. Motion library / command state (if available) ----
    try:
        cmd = env.command_manager.get_term("motion")
        if hasattr(cmd, "motion_lib") and cmd.motion_lib is not None:
            ml = cmd.motion_lib
            motion_id = int(cmd.motion_ids[0].item()) if hasattr(cmd, "motion_ids") else -1
            start_step = int(cmd.motion_start_time_steps[0].item()) if hasattr(cmd, "motion_start_time_steps") else -1
            delta_step = int(cmd.time_steps[0].item()) if hasattr(cmd, "time_steps") else -1
            current_step = start_step + delta_step if start_step >= 0 and delta_step >= 0 else -1
            motion_key = ""
            if hasattr(ml, "_motion_data_keys") and motion_id >= 0:
                motion_key = str(ml._motion_data_keys[motion_id])

            data["motion_motion_id"] = np.array(motion_id, dtype=np.int64)
            data["motion_start_frame_idx"] = np.array(start_step, dtype=np.int64)
            data["motion_delta_frame_idx"] = np.array(delta_step, dtype=np.int64)
            data["motion_frame_idx"] = np.array(current_step, dtype=np.int64)
            data["motion_key"] = np.array(motion_key)
            future_frame_idxs = cmd.future_time_steps.view(env.num_envs, -1)[0].detach().cpu().numpy().copy()
            data["motion_future_frame_idxs"] = future_frame_idxs

            raw_entry = ml._motion_data_load[motion_key] if motion_key in ml._motion_data_load else None
            if raw_entry is not None and "path" in raw_entry:
                loaded = joblib.load(raw_entry["path"])
                raw_entry = next(iter(loaded.values())) if isinstance(loaded, dict) and "root_trans_offset" not in loaded else loaded

            if raw_entry is not None:
                raw_fps = float(raw_entry.get("fps", 30.0))
                raw_joint_pos = np.asarray(raw_entry.get("dof", raw_entry.get("joint_pos")), dtype=np.float32)
                raw_root_pos = np.asarray(raw_entry.get("root_trans_offset", raw_entry.get("root_pos")), dtype=np.float32)
                raw_root_rot = np.asarray(raw_entry.get("root_rot"), dtype=np.float32)
                raw_joint_vel = np.asarray(raw_entry.get("joint_vel"), dtype=np.float32) if "joint_vel" in raw_entry else None
                if raw_joint_vel is None:
                    raw_joint_vel = np.zeros_like(raw_joint_pos)
                    raw_joint_vel[1:] = (raw_joint_pos[1:] - raw_joint_pos[:-1]) * raw_fps
                    raw_joint_vel[0] = raw_joint_vel[1]

                raw_root_vel_lin = np.asarray(raw_entry.get("root_vel_lin"), dtype=np.float32) if "root_vel_lin" in raw_entry else None
                if raw_root_vel_lin is None:
                    raw_root_vel_lin = np.zeros_like(raw_root_pos)
                    raw_root_vel_lin[1:] = (raw_root_pos[1:] - raw_root_pos[:-1]) * raw_fps
                    raw_root_vel_lin[0] = raw_root_vel_lin[1]

                raw_num_frames = raw_joint_pos.shape[0]
                safe_current_step = int(np.clip(current_step, 0, raw_num_frames - 1))
                safe_future_frame_idxs = np.clip(future_frame_idxs, 0, raw_num_frames - 1)
                
                data["motion_raw_fps"] = np.array(raw_fps, dtype=np.float32)
                data["motion_raw_root_pos"] = raw_root_pos[safe_current_step].copy()
                data["motion_raw_root_rot_xyzw"] = raw_root_rot[safe_current_step].copy()
                data["motion_raw_joint_pos"] = raw_joint_pos[safe_current_step].copy()
                data["motion_raw_joint_vel"] = raw_joint_vel[safe_current_step].copy()
                data["motion_raw_root_vel_lin"] = raw_root_vel_lin[safe_current_step].copy()

                data["motion_raw_root_pos_future"] = raw_root_pos[safe_future_frame_idxs].copy()
                data["motion_raw_root_rot_future_xyzw"] = raw_root_rot[safe_future_frame_idxs].copy()
                data["motion_raw_joint_pos_future"] = raw_joint_pos[safe_future_frame_idxs].copy()
                data["motion_raw_joint_vel_future"] = raw_joint_vel[safe_future_frame_idxs].copy()

            cur_motion_ids = cmd.motion_ids[:1]
            cur_motion_steps = (cmd.motion_start_time_steps + cmd.time_steps)[:1]
            data["motion_ref_root_pos_w"] = ml.get_root_pos_w(cur_motion_ids, cur_motion_steps)[0].detach().cpu().numpy()
            data["motion_ref_root_quat_w"] = ml.get_root_quat_w(cur_motion_ids, cur_motion_steps)[0].detach().cpu().numpy()
            data["motion_ref_joint_pos"] = ml.get_dof_pos(cur_motion_ids, cur_motion_steps)[0].detach().cpu().numpy()
            data["motion_ref_joint_vel"] = ml.get_dof_vel(cur_motion_ids, cur_motion_steps)[0].detach().cpu().numpy()

            fut_motion_ids = cmd.future_motion_ids.view(env.num_envs, -1)[:1]
            fut_steps = cmd.future_time_steps[:1]
            data["motion_ref_root_pos_future_w"] = ml.get_root_pos_w(
                fut_motion_ids.reshape(-1), fut_steps.reshape(-1)
            ).view(1, -1, 3)[0].detach().cpu().numpy()
            data["motion_ref_root_quat_future_w"] = ml.get_root_quat_w(
                fut_motion_ids.reshape(-1), fut_steps.reshape(-1)
            ).view(1, -1, 4)[0].detach().cpu().numpy()
            data["motion_ref_joint_pos_future"] = ml.get_dof_pos(
                fut_motion_ids.reshape(-1), fut_steps.reshape(-1)
            ).view(1, -1, joint_pos.shape[0])[0].detach().cpu().numpy()
            data["motion_ref_joint_vel_future"] = ml.get_dof_vel(
                fut_motion_ids.reshape(-1), fut_steps.reshape(-1)
            ).view(1, -1, joint_pos.shape[0])[0].detach().cpu().numpy()

    except Exception as e:
        print(f"\n[WARNING] Motion library dump failed: {e}")
        import traceback
        traceback.print_exc() # 에러 원인을 터미널에 상세히 출력

    # ---- 8. Proprioception breakdown using actual ObservationGroup layout ----
    data.update(decompose_policy_obs(data["policy_obs"]))

    return data


def save_dump(data: dict, path: str):
    """Save all tensors to .npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **data)
    print(f"\n  Saved {len(data)} arrays to {path}")
    for key, val in data.items():
        shape = val.shape if hasattr(val, "shape") else "scalar"
        dtype = val.dtype if hasattr(val, "dtype") else type(val)
        print(f"    {key:35s}  shape={str(shape):20s}  dtype={dtype}")


def print_step0_summary(data: dict):
    """Print a human-readable summary of step-0 dump."""
    print(f"\n{'='*70}")
    print("STEP-0 GROUND TRUTH SUMMARY")
    print(f"{'='*70}")
    print(f"  qpos:         {data['qpos'].shape}  "
          f"root_z={data['qpos'][2]:.4f}  "
          f"quat=({data['qpos'][3]:.4f}, {data['qpos'][4]:.4f}, "
          f"{data['qpos'][5]:.4f}, {data['qpos'][6]:.4f})")
    print(f"  qpos_dof:     {data['qpos_dof'].shape}  "
          f"min={data['qpos_dof'].min():.4f}  max={data['qpos_dof'].max():.4f}")
    print(f"  qvel:         {data['qvel'].shape}  "
          f"lin=({data['qvel'][0]:.4f}, {data['qvel'][1]:.4f}, {data['qvel'][2]:.4f})  "
          f"ang=({data['qvel'][3]:.4f}, {data['qvel'][4]:.4f}, {data['qvel'][5]:.4f})")
    print(f"  gravity_body: {data['gravity_body']}")
    print(f"  policy_obs:   {data['policy_obs'].shape}  "
          f"min={data['policy_obs'].min():.4f}  max={data['policy_obs'].max():.4f}")
    if "motion_key" in data:
        print(
            f"  motion:       key={data['motion_key']}  "
            f"motion_id={data.get('motion_motion_id', 'n/a')}  "
            f"frame_idx={data.get('motion_frame_idx', 'n/a')}"
        )

    # First frame breakdown
    print(f"\n  Term-major history breakdown (frame 0):")
    for term in PER_FRAME_DIMS:
        val = data[f"frame0_{term}"]
        print(f"    {term:20s}  {val.shape}  {val}")

    # Check if history buffers are all identical (broadcast-fill at reset)
    print(f"\n  History buffer check (all frames identical after reset?):")
    for term in PER_FRAME_DIMS:
        hist = data[f"history_{term}"]
        all_close = np.allclose(hist, hist[0:1])
        print(f"    {term:20s}  {all_close}")


# ===================================================================
# CLI
# ===================================================================
@hydra.main(config_path="../config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    """Hydra entry point: load config, create env, dump step 0 / trajectory.

    Config must contain a fully composed ``manager_env`` subtree (with
    ``_target_``).  This is satisfied either by loading a saved checkpoint
    config or by providing ``manager_env=base_env`` on the CLI.
    """
    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument("--dump-frames", type=int, default=None,
                         help="Number of frames to dump (1 = step0 only, >1 = trajectory with env stepping)")
    _cli_args, _remaining = _parser.parse_known_args()
    if _remaining:
        import sys
        sys.argv = [sys.argv[0]] + _remaining

    os.chdir(hydra.utils.get_original_cwd())

    config.num_envs = 1

    # ---- Load saved training config from checkpoint (like eval_agent_trl.py) ----
    if config.get("checkpoint") is not None:
        checkpoint_path = Path(config.checkpoint)
        config_path = checkpoint_path.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint_path.parent.parent / "config.yaml"
        if config_path.exists():
            print(f"Loading saved training config from {config_path}")
            raw = config_path.read_text()
            from omegaconf import OmegaConf
            import io

            if isinstance(raw, str) and "\n" in raw:
                # YAML text 자체인 경우
                saved_cfg = OmegaConf.create(raw)
            else:
                # 파일 경로인 경우
                saved_cfg = OmegaConf.load(raw)
            # Merge: CLI overrides take precedence over saved config
            config = OmegaConf.merge(saved_cfg, config)
        else:
            print(f"Warning: no saved config found at {config_path}")

    # ---- Validate the config tree ----
    if "manager_env" not in config or "_target_" not in config.manager_env:
        raise ValueError(
            "manager_env config subtree is not fully composed. "
            "Either:\n"
            "  1. Provide checkpoint=/path/to/checkpoint.pt to load a saved config\n"
            "  2. Provide +manager_env=base_env on CLI (with the ``+`` prefix to\n"
            "     append to the defaults list, since manager_env is not in\n"
            "     config/base.yaml's defaults list)\n"
            "     (plus required overrides for observations, rewards, etc.)"
        )

    print(f"\n{'='*70}")
    print("IsaacLab Step-0 Dumper")
    print(f"{'='*70}")
    robot_type = config.manager_env.config.get("robot", {}).get("type", "unknown")
    print(f"  Robot type:    {robot_type}")
    obs_policy = config.manager_env.observations.get("policy", "?")
    obs_token = config.manager_env.observations.get("tokenizer", "?")
    print(f"  Policy obs:    {obs_policy}")
    print(f"  Tokenizer:     {obs_token}")
    
    dump_frames = (
    _cli_args.dump_frames
    if _cli_args.dump_frames is not None
    else int(config.get("dump_frames", 1))
)

    print(f"  Dump frames:   {dump_frames}")

    # ---- Create env (same flow as train_agent_trl.py::create_manager_env) ----
    env_instance_cfg = custom_instantiate(config.manager_env)
    env_instance_cfg.seed = config.seed
    env_instance_cfg.sim.device = config.get("device", "cuda:0")
    env_instance_cfg.config["headless"] = config.get("headless", True)

    from isaaclab.envs import ManagerBasedRLEnv

    env = ManagerBasedRLEnv(
        cfg=env_instance_cfg,
        render_mode=None,
    )

    # ---- Reset -> step 0 ----
    obs, info = env.reset()
    
    #if not config.get("headless", True):
    #    robot_pos = env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()

    #    env.sim.set_camera_view(
    #        eye=(robot_pos[0] + 2.5, robot_pos[1] - 4.0, robot_pos[2] + 1.5),
    #        target=(robot_pos[0], robot_pos[1], robot_pos[2] + 0.7),
    #    )
    env.render()

    print(f"\n  Env created. Num envs: {env.num_envs}")
    print(f"  Policy obs dim: {obs['policy'].shape[-1]}")
    if isinstance(obs.get("tokenizer"), dict):
        for k, v in obs["tokenizer"].items():
            print(f"  Tokenizer [{k}]: {v.shape[-1]}")
    elif obs.get("tokenizer") is not None:
        print(f"  Tokenizer obs dim: {obs['tokenizer'].shape[-1]}")

    # ---- Collect processed motion reference trajectory, no env.step rollout ----
    all_frames = []
    motion_keys_recorded = set()

    cmd = env.command_manager.get_term("motion")
    ml = cmd.motion_lib
    motion_ids = cmd.motion_ids[:1]

    for frame_idx in range(dump_frames):
        with torch.no_grad():
            steps = torch.tensor([frame_idx], device=env.device, dtype=torch.long)

            root_pos = ml.get_root_pos_w(motion_ids, steps)
            root_quat = ml.get_root_quat_w(motion_ids, steps)
            joint_pos = ml.get_dof_pos(motion_ids, steps)
            joint_vel = ml.get_dof_vel(motion_ids, steps)

            root_pose = torch.cat([root_pos, root_quat], dim=-1)
            root_vel = torch.zeros((1, 6), device=env.device, dtype=torch.float32)

            robot = env.scene["robot"]
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_vel)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            env.sim.forward()
            env.render()

        frame_data = dump_isaaclab_step0(env)

        # force stored ref index to match replay frame
        frame_data["motion_frame_idx"] = np.array(frame_idx, dtype=np.int64)
        frame_data["motion_delta_frame_idx"] = np.array(frame_idx, dtype=np.int64)

        for key in list(frame_data.keys()):
            motion_keys_recorded.add(key)

        all_frames.append(frame_data)

        if (frame_idx + 1) % 10 == 0:
            print(f"  Dumped reference frame {frame_idx + 1}/{dump_frames}")

        # Store scalar-per-frame keys as time-indexed arrays
        for key in list(frame_data.keys()):
            val = frame_data[key]
            if isinstance(val, np.ndarray) and val.ndim == 1 and val.shape[0] > 1:
                # Already time-series (e.g. joint_pos_future, tokenizer arrays)
                pass
            # Mark all keys present
            motion_keys_recorded.add(key)

        if (frame_idx + 1) % 10 == 0:
            print(f"  Dumped frame {frame_idx + 1}/{dump_frames}")

    # ---- Convert list-of-dicts to dict-of-arrays ----
    trajectory = {}
    scalar_keys = ["motion_frame_idx", "motion_motion_id", "motion_start_frame_idx",
                   "motion_delta_frame_idx", "motion_key", "motion_raw_fps"]
    time_series_keys = [
        "qpos", "qpos_root", "qpos_dof", "qvel", "qvel_root_lin", "qvel_root_ang", "qvel_dof",
        "gravity_body", "joint_pos_rel", "default_joint_pos",
        "obs_term_gravity_dir", "obs_term_base_ang_vel", "obs_term_joint_pos_rel",
        "obs_term_joint_vel", "obs_term_last_action",
        "actions_raw", "actions_scaled", "policy_obs",
        "motion_raw_root_pos", "motion_raw_root_rot_xyzw",
        "motion_raw_joint_pos", "motion_raw_joint_vel", "motion_raw_root_vel_lin",
        "motion_ref_root_pos_w", "motion_ref_root_quat_w",
        "motion_ref_joint_pos", "motion_ref_joint_vel",
        "motion_ref_root_pos_future_w", "motion_ref_root_quat_future_w",
        "motion_ref_joint_pos_future", "motion_ref_joint_vel_future",
        # Reconstructed history terms from policy_obs decomposition
    ]

    for key in time_series_keys:
        if all(key in f for f in all_frames):
            stacked = np.stack([f[key] for f in all_frames], axis=0)
            trajectory[key] = stacked
            motion_keys_recorded.discard(key)
        else:
            missing = [i for i, f in enumerate(all_frames) if key not in f]
            print(f"  Skip key {key}: missing in {len(missing)} frames, first missing frame={missing[0]}")

    # Handle scalar keys (first frame only, copied per frame if needed)
    for key in scalar_keys:
        if all(key in f for f in all_frames):
            trajectory[key] = np.stack([f[key] for f in all_frames], axis=0)

    tok_keys = sorted(set().union(*(f.keys() for f in all_frames)))
    tok_keys = [k for k in tok_keys if k.startswith("tok_")]
    for key in tok_keys:
        if all(key in f for f in all_frames):
            trajectory[key] = np.stack([f[key] for f in all_frames], axis=0)
        else:
            missing = [i for i, f in enumerate(all_frames) if key not in f]
            print(f"  Skip key {key}: missing in {len(missing)} frames, first missing frame={missing[0]}")

    hist_keys = sorted(set().union(*(f.keys() for f in all_frames)))
    hist_keys = [k for k in hist_keys if k.startswith("history_") or k.startswith("frame")]
    for key in hist_keys:
        if all(key in f for f in all_frames):
            trajectory[key] = np.stack([f[key] for f in all_frames], axis=0)
        else:
            missing = [i for i, f in enumerate(all_frames) if key not in f]
            print(f"  Skip key {key}: missing in {len(missing)} frames, first missing frame={missing[0]}")

    # Motion library full arrays (motion_joint_pos, motion_root_pos, etc.) - same across frames
    for key in ["motion_joint_pos", "motion_joint_vel", "motion_root_pos", "motion_root_rot",
                 "motion_root_vel_lin", "motion_root_vel_ang"]:
        if key in all_frames[0]:
            trajectory[key] = all_frames[0][key]  # same across frames

    # Future frame idxs (same across frames within one motion)
    if "motion_future_frame_idxs" in all_frames[0]:
        trajectory["motion_future_frame_idxs"] = all_frames[0]["motion_future_frame_idxs"]

    # ---- Save ----
    traj_path = os.path.join(OUT_DIR, f"isaaclab_trajectory_{dump_frames}frames.npz")
    save_dump(trajectory, traj_path)
    print(f"\nTrajectory: {dump_frames} frames, {len(trajectory)} keys")

    # [추가] 최종 궤적 저장 상태 확인 로그
    print(f"\n{'='*70}")
    print(" MOTION REFERENCE TRAJECTORY VERIFICATION")
    print(f"{'='*70}")
    check_keys = [
        'motion_ref_root_pos_w',
        'motion_ref_root_quat_w',
        'motion_ref_joint_pos',
        'motion_ref_joint_vel'
    ]
    for k in check_keys:
        if k in trajectory:
            print(f"  {k:30s} -> SHAPE: {trajectory[k].shape}  [PASS \u2713]")
        else:
            print(f"  {k:30s} -> MISSING  [FAIL \u2717]")
    print(f"{'='*70}\n")

    env.close()

    # ---- Compare with MuJoCo dump if available ----
    mj_dump_path = os.path.join(OUT_DIR, "mujoco_step0_from_isaac_state.npz")
    il_step0_path = os.path.join(OUT_DIR, "isaaclab_step0.npz")
    if os.path.exists(mj_dump_path) and os.path.exists(il_step0_path):
        compare_npz(
            il_step0_path,
            mj_dump_path,
            os.path.join(OUT_DIR, "comparison_report.txt"),
        )
    else:
        print("\n  Skip step0 comparison: missing isaaclab_step0.npz or mujoco_step0_from_isaac_state.npz")
        print(f"\n  No MuJoCo dump found at {mj_dump_path}")
        print("  Run dump_h2_mujoco_step0_only.py with --load-isaaclab-state to generate it.")
        print("  Then re-run this script for comparison.")


# ===================================================================
# Standalone comparison entry point (runs outside IsaacLab)
# ===================================================================
def run_comparison():
    """Standalone comparison: just compare existing .npz files."""
    parser = argparse.ArgumentParser(description="Compare IsaacLab vs MuJoCo step-0 dumps")
    parser.add_argument("--isaaclab", default=os.path.join(OUT_DIR, "isaaclab_step0.npz"))
    parser.add_argument("--mujoco", default=os.path.join(OUT_DIR, "mujoco_step0_from_isaac_state.npz"))
    parser.add_argument("--report", default=os.path.join(OUT_DIR, "comparison_report.txt"))
    args = parser.parse_args()

    if not os.path.exists(args.isaaclab):
        print(f"IsaacLab dump not found: {args.isaaclab}")
        print("Run the main script in IsaacLab first.")
        return
    if not os.path.exists(args.mujoco):
        print(f"MuJoCo dump not found: {args.mujoco}")
        print("Run eval_h2_mujoco.py with --dump-step0 first.")
        return

    compare_npz(args.isaaclab, args.mujoco, args.report)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare-only":
        run_comparison()
    else:
        main()