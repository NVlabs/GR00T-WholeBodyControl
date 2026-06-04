import argparse

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Compare IsaacLab vs MuJoCo motion-based step0 dumps")
    parser.add_argument(
        "--isaaclab",
        type=str,
        default="dump_step0/isaaclab_step0.npz",
    )
    parser.add_argument(
        "--mujoco",
        type=str,
        default="dump_step0/mujoco_step0_from_motion.npz",
    )
    return parser.parse_args()


def compare_array(name: str, a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        print(f"{name:40s} SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    d = a - b
    ma = float(np.max(np.abs(d)))
    rms = float(np.sqrt(np.mean(d**2)))
    print(f"{name:40s} max_abs={ma:.6e} rms={rms:.6e}")


def compare_scalar_like(name: str, a, b):
    print(f"{name:40s} A={a!r} B={b!r} match={a == b}")


def xyzw_to_wxyz(q):
    q = np.asarray(q)
    return q[..., [3, 0, 1, 2]]


def quat_angle_deg_wxyz(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        print(f"{'quat_angle_deg':40s} SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    dots = np.sum(a * b, axis=-1)
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    ang = 2.0 * np.arccos(dots) * 180.0 / np.pi
    print(f"{'quat_angle_deg':40s} max={float(np.max(ang)):.6e} mean={float(np.mean(ang)):.6e}")


def norm_key(v):
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, np.ndarray) and v.shape == ():
        v = v.item()
    if isinstance(v, str) and v.endswith('.pkl'):
        v = v[:-4]
    return v


args = get_args()

il = np.load(args.isaaclab, allow_pickle=True)
mj = np.load(args.mujoco, allow_pickle=True)

print("MOTION STEP0 COMPARISON")
print(f"IsaacLab: {args.isaaclab}")
print(f"MuJoCo:   {args.mujoco}")
print()

print("[Metadata]")
for key in ["motion_key", "motion_motion_id", "motion_start_frame_idx", "motion_delta_frame_idx", "motion_frame_idx"]:
    if key in il and key in mj:
        compare_scalar_like(key, norm_key(il[key]), norm_key(mj[key]))
print()

print("[Axis 1: MuJoCo raw vs IsaacLab raw]")
raw_pairs = [
    ("root_pos", "motion_raw_root_pos", "motion_ref_root_pos_w"),
    ("joint_pos", "motion_raw_joint_pos", "motion_ref_joint_pos"),
    ("joint_vel", "motion_raw_joint_vel", "motion_ref_joint_vel"),
    ("future_frame_idxs", "motion_future_frame_idxs", "motion_future_frame_idxs"),
    ("root_pos_future", "motion_raw_root_pos_future", "motion_ref_root_pos_future_w"),
    ("joint_pos_future", "motion_raw_joint_pos_future", "motion_ref_joint_pos_future"),
    ("joint_vel_future", "motion_raw_joint_vel_future", "motion_ref_joint_vel_future"),
]
for label, il_key, mj_key in raw_pairs:
    if il_key in il and mj_key in mj:
        compare_array(label, il[il_key], mj[mj_key])
if "motion_raw_root_rot_xyzw" in il and "motion_ref_root_quat_xyzw" in mj:
    compare_array("root_rot_xyzw", il["motion_raw_root_rot_xyzw"], mj["motion_ref_root_quat_xyzw"])
if "motion_raw_root_rot_xyzw" in il and "motion_ref_root_quat_w" in mj:
    quat_angle_deg_wxyz(xyzw_to_wxyz(il["motion_raw_root_rot_xyzw"]), mj["motion_ref_root_quat_w"])
print()

print("[Axis 2: IsaacLab raw vs IsaacLab processed]")
if "motion_raw_root_rot_xyzw" in il and "motion_ref_root_quat_w" in il:
    quat_angle_deg_wxyz(xyzw_to_wxyz(il["motion_raw_root_rot_xyzw"]), il["motion_ref_root_quat_w"])
for raw_key, ref_key in [
    ("motion_raw_root_pos", "motion_ref_root_pos_w"),
    ("motion_raw_joint_pos", "motion_ref_joint_pos"),
    ("motion_raw_joint_vel", "motion_ref_joint_vel"),
    ("motion_raw_root_pos_future", "motion_ref_root_pos_future_w"),
    ("motion_raw_joint_pos_future", "motion_ref_joint_pos_future"),
    ("motion_raw_joint_vel_future", "motion_ref_joint_vel_future"),
]:
    if raw_key in il and ref_key in il:
        compare_array(f"{raw_key} -> {ref_key}", il[raw_key], il[ref_key])
print()

print("[Axis 3: MuJoCo processed/eval input vs IsaacLab processed]")
if "motion_ref_root_quat_w" in il and "motion_ref_root_quat_w" in mj:
    quat_angle_deg_wxyz(il["motion_ref_root_quat_w"], mj["motion_ref_root_quat_w"])
for key in [
    "motion_ref_root_pos_w",
    "motion_ref_root_quat_w",
    "motion_ref_joint_pos",
    "motion_ref_joint_vel",
    "motion_future_frame_idxs",
    "motion_ref_root_pos_future_w",
    "motion_ref_root_quat_future_w",
    "motion_ref_joint_pos_future",
    "motion_ref_joint_vel_future",
    "tok_encoder_index",
    "tok_command_multi_future_nonflat",
    "tok_motion_anchor_ori_b_mf_nonflat",
]:
    if key in il and key in mj:
        compare_array(key, il[key], mj[key])
print()

print("[Current sim state vs IsaacLab step0 state]")
for key in [
    "qpos_root",
    "qpos_dof",
    "qvel_root_lin",
    "qvel_root_ang",
    "qvel_dof",
    "obs_term_gravity_dir",
    "obs_term_base_ang_vel",
    "obs_term_joint_pos_rel",
    "obs_term_joint_vel",
]:
    if key in il and key in mj:
        compare_array(key, il[key], mj[key])
