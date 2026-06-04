import argparse
import os

import joblib
import mujoco
import numpy as np
import torch

from eval_h2_mujoco import (
    H2_DEFAULT_JOINT_POS,
    H2_MJCF_PATH,
    H2_NUM_DOF,
    HISTORY_LEN,
    IL_TO_MJ_DOF,
    MJ_TO_IL_DOF,
    PER_FRAME_DIMS,
    quat_inv,
    quat_mul,
    quat_apply,
    quat_apply_inverse,
)

MJ_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_roll_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_pitch_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

def get_h2_joint_qpos_qvel_addrs(model):
    qpos_addrs = []
    qvel_addrs = []
    for name in MJ_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Missing MuJoCo joint: {name}")
        qpos_addrs.append(model.jnt_qposadr[jid])
        qvel_addrs.append(model.jnt_dofadr[jid])
    return np.array(qpos_addrs), np.array(qvel_addrs)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--motion")
    p.add_argument("--dump-step0", required=True)
    p.add_argument("--init-frame", type=int, default=0)
    p.add_argument("--load-isaaclab-state")
    p.add_argument("--processed-motion-reference")
    p.add_argument("--rollout-compare", action="store_true")
    p.add_argument("--max-frames", type=int, default=-1)
    args = p.parse_args()
    if args.load_isaaclab_state is None and args.motion is None:
        p.error("Either --motion or --load-isaaclab-state is required")
    return args


def load_motion(path):
    obj = joblib.load(path)

    if isinstance(obj, dict) and "joint_pos" not in obj:
        key = next(iter(obj.keys()))
        print(f"[MuJoCo dump] using motion key: {key}")
        obj = obj[key]

    if "joint_pos" not in obj and "dof" in obj:
        fps = obj.get("fps", 30)
        dt = 1.0 / float(fps)

        joint_pos = obj["dof"]
        root_pos = obj["root_trans_offset"]
        root_rot = obj["root_rot"]

        joint_vel = np.zeros_like(joint_pos)
        joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) / dt
        joint_vel[0] = joint_vel[1]

        root_vel_lin = np.zeros_like(root_pos)
        root_vel_lin[1:] = (root_pos[1:] - root_pos[:-1]) / dt
        root_vel_lin[0] = root_vel_lin[1]

        root_vel_ang = np.zeros_like(root_pos)

        obj = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "root_vel_lin": root_vel_lin,
            "root_vel_ang": root_vel_ang,
        }

    print("[MuJoCo dump] normalized keys:", obj.keys())
    return obj


def motion_quat_xyzw_to_wxyz(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    return quat_xyzw[[3, 0, 1, 2]]


def decompose_policy_obs(proprio: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    off = 0

    for term_name, dim in PER_FRAME_DIMS.items():
        span = HISTORY_LEN * dim
        hist = proprio[off:off + span].reshape(HISTORY_LEN, dim)
        out[f"history_{term_name}"] = hist
        for frame_idx in range(HISTORY_LEN):
            out[f"frame{frame_idx}_{term_name}"] = hist[frame_idx]
        off += span

    return out

def build_policy_obs(gravity_body, base_ang_vel_body, joint_pos_rel, joint_vel_il, last_action_il):
    terms = [
        gravity_body,
        base_ang_vel_body,
        joint_pos_rel,
        joint_vel_il,
        last_action_il,
    ]
    term_histories = []
    for term in terms:
        hist = np.repeat(term[None, :], HISTORY_LEN, axis=0)
        term_histories.append(hist.reshape(-1))
    return np.concatenate(term_histories, axis=0).astype(np.float32)


def quat_to_6d(q: torch.Tensor) -> torch.Tensor:
    assert q.shape[-1] == 4, f"quat_to_6d expects (..., 4), got {tuple(q.shape)}"
    q = q / torch.norm(q)
    x2 = q[1] * q[1]
    y2 = q[2] * q[2]
    z2 = q[3] * q[3]
    xy = q[1] * q[2]
    xz = q[1] * q[3]
    yz = q[2] * q[3]
    wx = q[0] * q[1]
    wy = q[0] * q[2]
    wz = q[0] * q[3]
    r00 = 1 - 2 * (y2 + z2)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (x2 + z2)
    r12 = 2 * (yz - wx)
    return torch.tensor([r00, r01, r02, r10, r11, r12], dtype=torch.float32)


def build_tokenizer_terms(motion_lib, current_frame: int, base_quat_wxyz: torch.Tensor) -> dict[str, np.ndarray]:
    num_future = 10
    step = 5
    cmd_frames = []
    anchor_frames = []
    encoder_index = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    future_frame_idxs = []

    for f in range(num_future):
        idx = min(current_frame + f * step, motion_lib["joint_pos"].shape[0] - 1)
        future_frame_idxs.append(idx)
        joint_pos_il = torch.from_numpy(motion_lib["joint_pos"][idx]).float()
        joint_vel_il = torch.from_numpy(motion_lib["joint_vel"][idx]).float()
        ref_root_quat = torch.from_numpy(motion_quat_xyzw_to_wxyz(motion_lib["root_rot"][idx])).float()
        cmd_frames.append(torch.cat([joint_pos_il, joint_vel_il]).numpy())
        ref_ori_b = quat_mul(quat_inv(base_quat_wxyz), ref_root_quat)
        ref_ori_b = ref_ori_b / torch.norm(ref_ori_b)
        anchor_frames.append(quat_to_6d(ref_ori_b).numpy())

    return {
        "tok_encoder_index": encoder_index,
        "tok_command_multi_future_nonflat": np.stack(cmd_frames, axis=0),
        "tok_motion_anchor_ori_b_mf_nonflat": np.stack(anchor_frames, axis=0),
        "motion_future_frame_idxs": np.asarray(future_frame_idxs, dtype=np.int64),
    }


def initialize_from_motion(mj_data, motion_lib, idx, qpos_addrs, qvel_addrs):
    root_pos = motion_lib["root_pos"][idx]
    root_quat = motion_quat_xyzw_to_wxyz(motion_lib["root_rot"][idx])
    root_vel_lin = motion_lib["root_vel_lin"][idx]
    root_vel_ang_world = motion_lib["root_vel_ang"][idx]

    joint_pos_il = torch.from_numpy(motion_lib["joint_pos"][idx]).float()
    joint_vel_il = torch.from_numpy(motion_lib["joint_vel"][idx]).float()

    mj_data.qpos[:3] = root_pos
    mj_data.qpos[3:7] = root_quat
    joint_pos_mj = joint_pos_il.numpy()[MJ_TO_IL_DOF.numpy()]
    joint_vel_mj = joint_vel_il.numpy()[MJ_TO_IL_DOF.numpy()]
    mj_data.qpos[qpos_addrs] = joint_pos_mj

    mj_data.qvel[:3] = root_vel_lin
    mj_data.qvel[3:6] = quat_apply_inverse(
        torch.from_numpy(root_quat).float(),
        torch.from_numpy(root_vel_ang_world).float(),
    ).numpy()
    mj_data.qvel[qvel_addrs] = joint_vel_mj
    return "motion"


def initialize_from_isaaclab_dump(mj_data, isaaclab_path, qpos_addrs, qvel_addrs):
    il = dict(np.load(isaaclab_path))

    # trajectory npz: do not inject here
    if (
        "qpos_dof" in il
        and isinstance(il["qpos_dof"], np.ndarray)
        and il["qpos_dof"].ndim == 2
    ):
        return "isaaclab_trajectory", il

    for k in il.keys():
        if isinstance(il[k], np.ndarray) and il[k].ndim > 0 and il[k].shape[0] == 1:
            il[k] = il[k][0]

    root_pos = il["qpos_root"][:3]
    root_quat = il["qpos_root"][3:7]

    joint_pos_il = torch.from_numpy(il["qpos_dof"]).float()
    joint_vel_il = torch.from_numpy(il["qvel_dof"]).float()
    root_vel_lin = il["qvel_root_lin"]
    root_vel_ang_world = torch.from_numpy(il["qvel_root_ang"]).float()

    joint_pos_mj = joint_pos_il.numpy()[MJ_TO_IL_DOF.numpy()]
    joint_vel_mj = joint_vel_il.numpy()[MJ_TO_IL_DOF.numpy()]
    mj_data.qpos[:3] = root_pos
    mj_data.qpos[3:7] = root_quat
    mj_data.qpos[qpos_addrs] = joint_pos_mj

    mj_data.qvel[:3] = root_vel_lin
    mj_data.qvel[3:6] = quat_apply_inverse(torch.from_numpy(root_quat).float(), root_vel_ang_world).numpy()
    mj_data.qvel[qvel_addrs] = joint_vel_mj
    return "isaaclab_dump", il


def initialize_from_processed_motion_reference(mj_data, processed_path, qpos_addrs, qvel_addrs):
    ref = np.load(processed_path, allow_pickle=True)
    root_pos = ref["motion_ref_root_pos_w"]
    root_quat = ref["motion_ref_root_quat_w"]
    joint_pos_il = torch.from_numpy(ref["motion_ref_joint_pos"]).float()
    joint_vel_il = torch.from_numpy(ref["motion_ref_joint_vel"]).float()
    root_vel_lin = np.zeros(3, dtype=np.float32)
    root_vel_ang_world = torch.zeros(3)

    joint_pos_mj = joint_pos_il.numpy()[MJ_TO_IL_DOF.numpy()]
    joint_vel_mj = joint_vel_il.numpy()[MJ_TO_IL_DOF.numpy()]
    mj_data.qpos[:3] = root_pos
    mj_data.qpos[3:7] = root_quat
    mj_data.qpos[qpos_addrs] = joint_pos_mj

    mj_data.qvel[:3] = root_vel_lin
    mj_data.qvel[3:6] = quat_apply_inverse(torch.from_numpy(root_quat).float(), root_vel_ang_world).numpy()
    mj_data.qvel[qvel_addrs] = joint_vel_mj
    return "processed_motion_reference", ref


def main():
    args = get_args()
    
    mj_model = mujoco.MjModel.from_xml_path(H2_MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    qpos_addrs, qvel_addrs = get_h2_joint_qpos_qvel_addrs(mj_model)


    if args.load_isaaclab_state is not None:
        mode, il_dump = initialize_from_isaaclab_dump(
            mj_data, args.load_isaaclab_state, qpos_addrs, qvel_addrs
        )
        motion_lib = None
        idx = -1

    elif args.processed_motion_reference is not None:
        mode, il_dump = initialize_from_processed_motion_reference(
            mj_data, args.processed_motion_reference, qpos_addrs, qvel_addrs
        )
        motion_lib = None
        idx = int(il_dump["motion_frame_idx"].item()) if "motion_frame_idx" in il_dump else 0

    else:
        motion_lib = load_motion(args.motion)
        num_frames = motion_lib["joint_pos"].shape[0]
        idx = min(args.init_frame, num_frames - 1)
        mode = initialize_from_motion(
            mj_data, motion_lib, idx, qpos_addrs, qvel_addrs
        )
        il_dump = None

    if il_dump is not None and "default_joint_pos" in il_dump:
        if il_dump["default_joint_pos"].ndim > 1:
            default_joint_pos = torch.from_numpy(il_dump["default_joint_pos"][0]).float()
        else:
            default_joint_pos = torch.from_numpy(il_dump["default_joint_pos"]).float()
    else:
        default_joint_pos = H2_DEFAULT_JOINT_POS

    # 궤적 모드인지 확인
    is_traj = (
        il_dump is not None and
        "qpos" in il_dump and
        isinstance(il_dump["qpos"], np.ndarray) and
        il_dump["qpos"].ndim == 2
    )

    if args.rollout_compare and is_traj:
        num_frames = il_dump["qpos"].shape[0]
        if args.max_frames > 0:
            num_frames = min(num_frames, args.max_frames)

        rollout_dump = {}
        print(f"\n[Rollout Compare] Starting frame injection for {num_frames} frames...")

        for t in range(num_frames):
            # -------------------------
            # 1. inject IsaacLab frame t
            # -------------------------
            mj_data.qpos[:7] = il_dump["qpos_root"][t]
            joint_pos_il = torch.from_numpy(il_dump["qpos_dof"][t]).float()
            joint_pos_mj = np.empty(H2_NUM_DOF, dtype=np.float32)
            joint_pos_mj[IL_TO_MJ_DOF.numpy()] = joint_pos_il.numpy()
            mj_data.qpos[qpos_addrs] = joint_pos_mj

            mj_data.qvel[:3] = il_dump["qvel_root_lin"][t]
            root_quat = torch.from_numpy(il_dump["qpos_root"][t, 3:7]).float()
            root_ang_world = torch.from_numpy(il_dump["qvel_root_ang"][t]).float()
            mj_data.qvel[3:6] = quat_apply_inverse(root_quat.unsqueeze(0), root_ang_world.unsqueeze(0))[0].numpy()
            
            joint_vel_il = torch.from_numpy(il_dump["qvel_dof"][t]).float()
            joint_vel_mj = np.empty(H2_NUM_DOF, dtype=np.float32)
            joint_vel_mj[IL_TO_MJ_DOF.numpy()] = joint_vel_il.numpy()
            mj_data.qvel[qvel_addrs] = joint_vel_mj

            # Forward kinematics for frame t
            mujoco.mj_forward(mj_model, mj_data)

            # -------------------------
            # 2. recompute MuJoCo obs
            # -------------------------
            base_quat = torch.from_numpy(mj_data.qpos[3:7].copy()).float()
            base_ang_vel_body = torch.from_numpy(mj_data.qvel[3:6].copy()).float()
            gravity_body = quat_apply_inverse(base_quat.unsqueeze(0), torch.tensor([[0.0, 0.0, -1.0]]))[0]

            joint_pos_mj_tensor = torch.from_numpy(mj_data.qpos[qpos_addrs].copy()).float()
            joint_vel_mj_tensor = torch.from_numpy(mj_data.qvel[qvel_addrs].copy()).float()

            # 다시 Policy 입력을 위해 IL order로 복원 (정확한 매핑 적용)
            joint_pos_il_re = joint_pos_mj_tensor[IL_TO_MJ_DOF]
            joint_vel_il_re = joint_vel_mj_tensor[IL_TO_MJ_DOF]

            joint_pos_rel = joint_pos_il_re - default_joint_pos

            # -------------------------
            # 3. save framewise tensors
            # -------------------------
            rollout_dump[f"frame{t}_gravity_body"] = gravity_body.numpy()
            rollout_dump[f"frame{t}_joint_pos_rel"] = joint_pos_rel.numpy()
            rollout_dump[f"frame{t}_joint_vel"] = joint_vel_il_re.numpy()
            
            if "policy_obs" in il_dump:
                rollout_dump[f"frame{t}_policy_obs_isaaclab"] = il_dump["policy_obs"][t].copy()

        np.savez(args.dump_step0, **rollout_dump)
        print(f"[Rollout Compare] Successfully saved {num_frames} frames to {args.dump_step0}")
        return
    # =========================================================================
    mujoco.mj_forward(mj_model, mj_data)

    base_quat = torch.from_numpy(mj_data.qpos[3:7].copy()).float()
    base_ang_vel_body = torch.from_numpy(mj_data.qvel[3:6].copy()).float()
    gravity_body = quat_apply_inverse(base_quat, torch.tensor([0.0, 0.0, -1.0]))
    root_ang_vel_world = quat_apply(base_quat, base_ang_vel_body)

    joint_pos_mj = torch.from_numpy(mj_data.qpos[qpos_addrs].copy()).float()
    joint_vel_mj = torch.from_numpy(mj_data.qvel[qvel_addrs].copy()).float()
    joint_pos_il = joint_pos_mj[IL_TO_MJ_DOF]
    joint_vel_il = joint_vel_mj[IL_TO_MJ_DOF]

    # 여기에 넣어야 함
    if il_dump is not None and "default_joint_pos" in il_dump:
        default_joint_pos = torch.from_numpy(np.asarray(il_dump["default_joint_pos"]).reshape(-1)).float()
    else:
        default_joint_pos = H2_DEFAULT_JOINT_POS.float()

    joint_pos_rel = joint_pos_il - default_joint_pos

    last_action_il = np.zeros(H2_NUM_DOF, dtype=np.float32)

    if il_dump is not None and "policy_obs" in il_dump:
        policy_obs = il_dump["policy_obs"].copy()
        policy_obs_generated = policy_obs.copy()
    else:
        policy_obs_generated = build_policy_obs(
            gravity_body.numpy(),
            base_ang_vel_body.numpy(),
            joint_pos_rel.numpy(),
            joint_vel_il.numpy(),
            last_action_il,
        )
        policy_obs = policy_obs_generated

    if il_dump is not None and "policy_obs" in il_dump:
        policy_obs = policy_obs_generated
    else:
        policy_obs = policy_obs_generated

    dump = {
        "qpos": mj_data.qpos.copy(),
        "qvel": mj_data.qvel.copy(),
        "qpos_root": mj_data.qpos[:7].copy(),
        "qpos_dof": joint_pos_il.numpy(),
        "qvel_root_lin": mj_data.qvel[:3].copy(),
        "qvel_root_ang": root_ang_vel_world.numpy(),
        "qvel_dof": joint_vel_il.numpy(),
        "gravity_body": gravity_body.numpy(),
        "joint_pos": joint_pos_il.numpy(),
        "joint_vel": joint_vel_il.numpy(),
        "joint_pos_rel": joint_pos_rel.numpy(),
        "obs_term_gravity_dir": gravity_body.numpy(),
        "obs_term_base_ang_vel": base_ang_vel_body.numpy(),
        "obs_term_joint_pos_rel": joint_pos_rel.numpy(),
        "obs_term_joint_vel": joint_vel_il.numpy(),
        "obs_term_last_action": last_action_il.copy(),
        "default_joint_pos": default_joint_pos.numpy(),
        "actions_raw": np.zeros(H2_NUM_DOF),
        "actions_scaled": np.zeros(H2_NUM_DOF),
        "policy_obs": policy_obs,
        "policy_obs_mujoco_generated": policy_obs_generated,
    }

    if motion_lib is not None:
        dump.update({
            "motion_frame_idx": np.array(idx, dtype=np.int64),
            "motion_start_frame_idx": np.array(idx, dtype=np.int64),
            "motion_delta_frame_idx": np.array(0, dtype=np.int64),
            "motion_key": np.array(os.path.basename(args.motion)),
            "motion_ref_root_pos_w": np.asarray(motion_lib["root_pos"][idx], dtype=np.float32),
            "motion_ref_root_quat_xyzw": np.asarray(motion_lib["root_rot"][idx], dtype=np.float32),
            "motion_ref_root_quat_w": motion_quat_xyzw_to_wxyz(motion_lib["root_rot"][idx]),
            "motion_ref_joint_pos": np.asarray(motion_lib["joint_pos"][idx], dtype=np.float32),
            "motion_ref_joint_vel": np.asarray(motion_lib["joint_vel"][idx], dtype=np.float32),
            "motion_ref_root_pos_future_w": np.asarray([
                motion_lib["root_pos"][min(idx + f * 5, motion_lib["joint_pos"].shape[0] - 1)]
                for f in range(10)
            ], dtype=np.float32),
            "motion_ref_root_quat_future_xyzw": np.asarray([
                motion_lib["root_rot"][min(idx + f * 5, motion_lib["joint_pos"].shape[0] - 1)]
                for f in range(10)
            ], dtype=np.float32),
            "motion_ref_root_quat_future_w": np.asarray([
                motion_quat_xyzw_to_wxyz(motion_lib["root_rot"][min(idx + f * 5, motion_lib["joint_pos"].shape[0] - 1)])
                for f in range(10)
            ], dtype=np.float32),
            "motion_ref_joint_pos_future": np.asarray([
                motion_lib["joint_pos"][min(idx + f * 5, motion_lib["joint_pos"].shape[0] - 1)]
                for f in range(10)
            ], dtype=np.float32),
            "motion_ref_joint_vel_future": np.asarray([
                motion_lib["joint_vel"][min(idx + f * 5, motion_lib["joint_pos"].shape[0] - 1)]
                for f in range(10)
            ], dtype=np.float32),
        })
        dump.update(build_tokenizer_terms(motion_lib, idx, base_quat))
    
    if il_dump is not None and "policy_obs" in il_dump:
        dump["policy_obs_isaaclab"] = il_dump["policy_obs"].copy()

        # In replay-comparison mode, preserve IsaacLab history exactly.
        for key in [
            "motion_motion_id",
            "motion_start_frame_idx",
            "motion_delta_frame_idx",
            "motion_frame_idx",
            "motion_key",
            "motion_future_frame_idxs",
            "motion_ref_root_pos_w",
            "motion_ref_root_quat_w",
            "motion_ref_joint_pos",
            "motion_ref_joint_vel",
            "motion_ref_root_pos_future_w",
            "motion_ref_root_quat_future_w",
            "motion_ref_joint_pos_future",
            "motion_ref_joint_vel_future",
            "tok_encoder_index",
            "tok_command_multi_future_nonflat",
            "tok_motion_anchor_ori_b_mf_nonflat",
        ]:
            if key in il_dump:
                dump[key] = il_dump[key].copy()

        for key, value in decompose_policy_obs(policy_obs).items():
            dump[key] = value

        generated_decomp = decompose_policy_obs(policy_obs_generated)
        for key, value in generated_decomp.items():
            dump[f"{key}_mujoco_generated"] = value
    else:
        dump.update(decompose_policy_obs(policy_obs))

    os.makedirs(os.path.dirname(args.dump_step0), exist_ok=True)
    np.savez(args.dump_step0, **dump)
    print(f"[MuJoCo dump] mode={mode}")
    print(f"[MuJoCo dump] saved to {args.dump_step0}")
    print(f"[MuJoCo dump] qpos_root z={dump['qpos_root'][2]:.6f}")
    print(f"[MuJoCo dump] gravity_body={dump['gravity_body']}")
    print(f"[MuJoCo dump] policy_obs dim={dump['policy_obs'].shape[0]}")


if __name__ == "__main__":
    main()