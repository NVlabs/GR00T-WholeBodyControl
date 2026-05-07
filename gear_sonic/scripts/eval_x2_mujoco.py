#!/usr/bin/env python3
"""Evaluate a trained SONIC checkpoint for the X2 Ultra in MuJoCo.

Loads a .pt checkpoint, reconstructs the universal-token actor (g1 encoder
+ g1_dyn decoder), reads the motion-lib PKL for reference commands, and
runs a PD-control loop in MuJoCo with on-screen rendering.

The robot is reset to the motion's frame-``--init-frame`` state using
Reference State Initialization (RSI) — the same teleport-into-the-motion
trick that IsaacLab does at every episode reset. RSI sets joint pos+vel
and root pos+quat+lin_vel+ang_vel from the motion, then physics takes
over and the policy must keep tracking.

Whenever the robot falls (pelvis drops below ``--fall-height``) or tips
over (gravity_body z > ``--fall-tilt-cos``) the episode is auto-reset to
the same RSI state — same as IsaacLab's termination/reset cycle. This
lets the policy keep getting fresh in-distribution starts during a long
viewer session.

Usage:
    python gear_sonic/scripts/eval_x2_mujoco.py \\
        --checkpoint logs_rl/.../model_step_006000.pt \\
        --motion gear_sonic/data/motions/x2_ultra_walk_forward.pkl

    Optional:
      --init-frame N         RSI motion frame (default 0)
      --fall-height 0.4      Pelvis z below this (m) triggers a reset
      --fall-tilt-cos -0.3   gravity_body[z] above this triggers a reset
                             (i.e. body tilted >~70° off upright)
      --max-episode 10.0     Force-reset after this many seconds (s)

Controls:
    SPACE - Pause / resume
    R     - Manually reset
    V     - Toggle camera tracking / free camera
"""

import argparse
import collections
import math
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as Rot

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------- X2 Ultra constants ----------
NUM_DOFS = 31
HISTORY_LEN = 10
CONTROL_DT = 0.02
SIM_DT = 0.005
DECIMATION = 4
NUM_FUTURE_FRAMES = 10
DT_FUTURE_REF = 0.1

MJCF_PATH = str(GEAR_SONIC_ROOT / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml")

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
]

JOINT_TO_ACTUATOR = [
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12, 13, 14,
    17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30,
    15, 16,
]

IL_TO_MJ_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 29, 15, 22, 4, 10,
    30, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
MJ_TO_IL_DOF = [
    0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 12,
    17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30, 11, 16,
]

NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0

# Default joint positions from x2_ultra.py InitialStateCfg (training reset pose)
DEFAULT_JOINT_POS = {
    "hip_pitch": -0.312,
    "knee": 0.669,
    "ankle_pitch": -0.363,
    "elbow": -0.6,
    "left_shoulder_roll": 0.2,
    "left_shoulder_pitch": 0.2,
    "right_shoulder_roll": -0.2,
    "right_shoulder_pitch": 0.2,
}

# Effort limits from x2_ultra.py actuator config
EFFORT_LIMITS = {
    "hip_yaw": 120.0, "hip_roll": 120.0, "hip_pitch": 120.0, "knee": 120.0,
    "ankle_pitch": 36.0, "ankle_roll": 24.0,
    "waist_yaw": 120.0, "waist_pitch": 48.0, "waist_roll": 48.0,
    "shoulder_pitch": 36.0, "shoulder_roll": 36.0, "shoulder_yaw": 24.0, "elbow": 24.0,
    "wrist_yaw": 24.0, "wrist_pitch": 4.8, "wrist_roll": 4.8,
    "head_yaw": 2.6, "head_pitch": 0.6,
}

ARMATURES = {
    "hip": 0.025101925, "knee": 0.025101925,
    "waist_yaw": 0.010177520, "waist_pitch": 0.003609725, "waist_roll": 0.003609725,
    "ankle": 0.003609725,
    "shoulder": 0.003609725, "elbow": 0.003609725,
    "wrist_yaw": 0.003609725, "wrist_pitch": 0.00425, "wrist_roll": 0.00425,
    "head": 0.00425,
}


def _compute_gains_and_scales():
    kp = np.zeros(NUM_DOFS, dtype=np.float64)
    kd = np.zeros(NUM_DOFS, dtype=np.float64)
    action_scale = np.ones(NUM_DOFS, dtype=np.float64)
    default_pos = np.zeros(NUM_DOFS, dtype=np.float64)

    for i, jname in enumerate(MUJOCO_JOINT_NAMES):
        # PD gains from armature
        for key, arm in ARMATURES.items():
            if key in jname:
                kp[i] = arm * NATURAL_FREQ**2
                kd[i] = 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ
                break

        # Action scale = 0.25 * effort / stiffness
        short = jname.replace("_joint", "").replace("left_", "").replace("right_", "")
        for ekey, effort in EFFORT_LIMITS.items():
            if ekey in jname.replace("_joint", ""):
                action_scale[i] = 0.25 * effort / kp[i]
                break

        # Default positions
        for dkey, dval in DEFAULT_JOINT_POS.items():
            if dkey == "left_shoulder_roll" and jname == "left_shoulder_roll_joint":
                default_pos[i] = dval; break
            elif dkey == "left_shoulder_pitch" and jname == "left_shoulder_pitch_joint":
                default_pos[i] = dval; break
            elif dkey == "right_shoulder_roll" and jname == "right_shoulder_roll_joint":
                default_pos[i] = dval; break
            elif dkey == "right_shoulder_pitch" and jname == "right_shoulder_pitch_joint":
                default_pos[i] = dval; break
            elif dkey in jname.replace("_joint", "") and "shoulder" not in dkey:
                default_pos[i] = dval; break

    return kp, kd, action_scale, default_pos


KP, KD, ACTION_SCALE, DEFAULT_DOF = _compute_gains_and_scales()


# ---------- Actor ----------
class SimpleMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.SiLU())
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


def fsq_quantize(z, levels=32):
    """Replicate vector_quantize_pytorch.FSQ with uniform `levels`-per-dim.

    Matches FSQ.bound() exactly:
        half_l = (L-1) * (1+eps) / 2
        offset = 0.5 if L%2==0 else 0
        shift  = atanh(offset / half_l)
        bounded = tanh(z + shift) * half_l - offset
        return round(bounded) / (L // 2)

    The training config uses num_fsq_levels=32, fsq_level_list=32, so dim==
    effective_codebook_dim and FSQ.project_in/out are nn.Identity (no params).
    """
    L = float(levels)
    eps = 1e-3
    half_l = (L - 1.0) * (1.0 + eps) / 2.0
    offset = 0.5 if int(levels) % 2 == 0 else 0.0
    shift = math.atanh(offset / half_l) if offset != 0.0 else 0.0
    bounded = torch.tanh(z + shift) * half_l - offset
    return torch.round(bounded) / (int(levels) // 2)


class UniversalTokenActor(nn.Module):
    """Reproduces gear_sonic.trl.modules.universal_token_modules.UniversalTokenModule
    inference path for a single g1 encoder + g1_dyn decoder + FSQ quantizer.

    Pipeline (matches universal_token_modules.encode + decode for g1/g1_dyn):
        tokenizer_obs(680) ── encoder MLP ──► 64
                          ── reshape(B, 2, 32) ──► FSQ(quantize) ──► (B, 2, 32)
                          ── flatten ──► token_flattened (B, 64)
        cat([token_flattened, proprioception(990)]) ──► decoder MLP ──► action(31)

    FSQ is a fixed deterministic bound+round (32 levels per dim) — no params.
    """

    MAX_NUM_TOKENS = 2
    TOKEN_DIM = 32
    FSQ_LEVELS = 32

    def __init__(self):
        super().__init__()
        self.encoder = SimpleMLP([680, 2048, 1024, 512, 512, 64])
        self.decoder = SimpleMLP([1054, 2048, 2048, 1024, 1024, 512, 512, 31])
        self.std = nn.Parameter(torch.zeros(31))

    def forward(self, proprioception, tokenizer_obs):
        # Encoder produces 64-dim continuous latent
        latent = self.encoder(tokenizer_obs)
        # Reshape to (B, num_tokens, token_dim) = (B, 2, 32)
        latent = latent.view(*latent.shape[:-1], self.MAX_NUM_TOKENS, self.TOKEN_DIM)
        # FSQ quantize (bound + round) — same shape
        quantized = fsq_quantize(latent, levels=self.FSQ_LEVELS)
        # Flatten back to token_flattened (B, 64)
        token_flat = quantized.view(*quantized.shape[:-2], -1)
        decoder_input = torch.cat([token_flat, proprioception], dim=-1)
        return self.decoder(decoder_input)


def load_actor_from_checkpoint(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("policy_state_dict") or ckpt.get("actor_model_state_dict")
    if sd is None:
        raise KeyError("Cannot find policy state dict in checkpoint")
    actor = UniversalTokenActor()
    new_sd = {}
    for k, v in sd.items():
        if k == "std":
            new_sd["std"] = v
        elif k.startswith("actor_module.encoders.g1.module."):
            new_sd[k.replace("actor_module.encoders.g1.module.", "encoder.module.")] = v
        elif k.startswith("actor_module.decoders.g1_dyn.module."):
            new_sd[k.replace("actor_module.decoders.g1_dyn.module.", "decoder.module.")] = v
    actor.load_state_dict(new_sd)
    actor.eval()
    return actor.to(device)


# ---------- Motion helpers ----------
def load_motion_data(path):
    return joblib.load(path)


def _m(data):
    return data[list(data.keys())[0]]


def get_total_frames(data):
    return _m(data)["dof"].shape[0]


def get_motion_fps(data):
    return float(_m(data)["fps"])


def compute_motion_state(motion_data, frame, fps):
    """Reconstruct the full robot reset state from a motion frame.

    Mirrors IsaacLab's Reference State Initialization (RSI) in
    ``TrackingCommand`` (see ``gear_sonic/envs/manager_env/mdp/commands.py``
    ``write_joint_state_to_sim`` / ``write_root_state_to_sim``): at reset,
    the simulator teleports the robot to the motion's frame ``f`` state —
    joint pos+vel, root pos+quat+lin_vel+ang_vel — bypassing physics. We
    reproduce that here so MuJoCo starts in the same distribution the
    policy was trained on.

    The PKL stores joint DOFs in **MuJoCo order**, ``root_trans_offset`` in
    world frame, and ``root_rot`` as a scipy-style **xyzw** quaternion.
    Velocities are not stored in the PKL, so we reconstruct them with a
    one-step forward finite difference (matches IsaacLab's motion-lib
    ``dof_vels`` and ``global_root_velocity`` to ~1e-2 precision).

    Returns a dict with:
        joint_pos_mj      (31,) MuJoCo joint order
        joint_vel_mj      (31,) MuJoCo joint order
        root_pos_w        (3,)  world-frame xyz
        root_quat_w_wxyz  (4,)  MuJoCo wxyz quaternion
        root_lin_vel_w    (3,)  world-frame linear vel
        root_ang_vel_w    (3,)  world-frame angular vel (axis-angle / dt)
    """
    m = _m(motion_data)
    n_frames = m["dof"].shape[0]
    f = int(frame) % n_frames
    f_next = min(f + 1, n_frames - 1)
    dt = 1.0 / float(fps)

    joint_pos_mj = np.asarray(m["dof"][f], dtype=np.float64)
    if f_next != f:
        joint_vel_mj = (np.asarray(m["dof"][f_next], dtype=np.float64) - joint_pos_mj) / dt
    else:
        joint_vel_mj = np.zeros(NUM_DOFS, dtype=np.float64)

    root_pos_w = np.asarray(m["root_trans_offset"][f], dtype=np.float64).copy()

    # PKL root_rot is xyzw (scipy). MuJoCo qpos[3:7] is wxyz.
    quat_xyzw = np.asarray(m["root_rot"][f], dtype=np.float64)
    root_quat_w_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )

    if f_next != f:
        root_lin_vel_w = (
            np.asarray(m["root_trans_offset"][f_next], dtype=np.float64) - root_pos_w
        ) / dt
        q_next_xyzw = np.asarray(m["root_rot"][f_next], dtype=np.float64)
        rel = Rot.from_quat(q_next_xyzw) * Rot.from_quat(quat_xyzw).inv()
        root_ang_vel_w = rel.as_rotvec() / dt
    else:
        root_lin_vel_w = np.zeros(3, dtype=np.float64)
        root_ang_vel_w = np.zeros(3, dtype=np.float64)

    return {
        "joint_pos_mj": joint_pos_mj,
        "joint_vel_mj": joint_vel_mj,
        "root_pos_w": root_pos_w,
        "root_quat_w_wxyz": root_quat_w_wxyz,
        "root_lin_vel_w": root_lin_vel_w,
        "root_ang_vel_w": root_ang_vel_w,
    }


# ---------- Observation construction ----------
# IsaacLab observation layout for policy group (concatenate_terms=True).
# CRITICAL: term order follows the PolicyCfg dataclass ATTRIBUTE order
# (gear_sonic/envs/manager_env/mdp/observations.py lines 107-128), NOT the
# YAML ordering. The PolicyAtmCfg comment confirms it explicitly:
#   "Order matches PolicyCfg: base_ang_vel, joint_pos, joint_vel, actions, gravity_dir"
# Each term: history_length=10, oldest at index 0, newest at index max_length-1
# (CircularBuffer.buffer property; broadcast-fills all slots with first sample
# at reset). Layout (oldest..newest within each term):
#   base_ang_vel  (3)  x 10 -> 30
#   joint_pos_rel (31) x 10 -> 310
#   joint_vel     (31) x 10 -> 310
#   last_action   (31) x 10 -> 310
#   gravity_dir   (3)  x 10 -> 30
# Total proprioception: 990
# Verified via /tmp/x2_step0_isaaclab.pt dump (slicing each block matched the
# corresponding env_state quantity within the configured noise tolerance).

def quat_rotate_inverse(q_wxyz, v):
    """Rotate vector v by the INVERSE of quaternion q (wxyz convention).

    Matches IsaacLab's quat_apply_inverse: v - w*t + cross(u, t).
    """
    w, x, y, z = q_wxyz
    u = np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return v - w * t + np.cross(u, t)


class ProprioceptionBuffer:
    """Maintains per-term history buffers matching IsaacLab's layout.

    Mirrors IsaacLab's ``CircularBuffer`` semantics: at reset the buffer is
    empty, and the first ``append`` after reset broadcast-fills all
    ``HISTORY_LEN`` slots with the first observation (see IsaacLab
    ``circular_buffer.py``: ``buffer`` returns the full history with the
    first sample replicated until the buffer fills naturally).

    Without this priming we would inject ``HISTORY_LEN-1`` zeroed frames
    into the proprioception, which is OOD for any policy trained with
    history.
    """

    def __init__(self):
        self.gravity_hist = collections.deque(maxlen=HISTORY_LEN)
        self.angvel_hist = collections.deque(maxlen=HISTORY_LEN)
        self.jpos_hist = collections.deque(maxlen=HISTORY_LEN)
        self.jvel_hist = collections.deque(maxlen=HISTORY_LEN)
        self.action_hist = collections.deque(maxlen=HISTORY_LEN)
        self._primed = False

    def reset(self):
        self.gravity_hist.clear()
        self.angvel_hist.clear()
        self.jpos_hist.clear()
        self.jvel_hist.clear()
        self.action_hist.clear()
        self._primed = False

    def append(self, gravity, angvel, jpos_rel, jvel, action):
        g = gravity.astype(np.float32)
        a = angvel.astype(np.float32)
        jp = jpos_rel.astype(np.float32)
        jv = jvel.astype(np.float32)
        ac = action.astype(np.float32)
        if not self._primed:
            for _ in range(HISTORY_LEN):
                self.gravity_hist.append(g)
                self.angvel_hist.append(a)
                self.jpos_hist.append(jp)
                self.jvel_hist.append(jv)
                self.action_hist.append(ac)
            self._primed = True
        else:
            self.gravity_hist.append(g)
            self.angvel_hist.append(a)
            self.jpos_hist.append(jp)
            self.jvel_hist.append(jv)
            self.action_hist.append(ac)

    def get_flat(self) -> np.ndarray:
        """Return 990-dim proprioception in IsaacLab term-by-term layout.

        Term order MUST match ``PolicyCfg`` dataclass attribute order:
            base_ang_vel, joint_pos, joint_vel, actions, gravity_dir.
        Within each term, frames are oldest-first (CircularBuffer convention).
        """
        parts = []
        for hist in [self.angvel_hist, self.jpos_hist, self.jvel_hist,
                     self.action_hist, self.gravity_hist]:
            for frame in hist:
                parts.append(frame)
        return np.concatenate(parts).astype(np.float32)


def build_tokenizer_obs(motion_data, current_time, base_quat_wxyz, motion_fps):
    """Build 680-dim tokenizer input matching IsaacLab's exact layout.

    Training layout (from command_multi_future + motion_anchor_ori_b_mf):
      command_flat = cat([jpos_flat(10*31=310), jvel_flat(10*31=310)]) = 620
      command_nonflat = command_flat.reshape(10, 62)
      ori_nonflat = 6D_rot_diff per frame, shape (10, 6)
      encoder_input = cat(command_nonflat, ori_nonflat, dim=-1) = (10, 68)
      flattened to 680
    """
    m = _m(motion_data)
    total_frames = m["dof"].shape[0]
    dt = 1.0 / motion_fps

    cur_rot = Rot.from_quat([base_quat_wxyz[1], base_quat_wxyz[2],
                              base_quat_wxyz[3], base_quat_wxyz[0]])

    # Collect 10 future frames of jpos and jvel (in IsaacLab DOF order)
    jpos_frames = []  # each (31,) in IL order
    jvel_frames = []  # each (31,) in IL order
    ori_frames = []   # each (6,) 6D rotation diff

    for f in range(NUM_FUTURE_FRAMES):
        future_time = current_time + (f + 1) * DT_FUTURE_REF
        fi = min(int(future_time / dt), total_frames - 1)

        jpos_il = m["dof"][fi][IL_TO_MJ_DOF]
        jpos_frames.append(jpos_il.astype(np.float32))

        prev_fi = max(0, fi - 1)
        jvel_mj = (m["dof"][fi] - m["dof"][prev_fi]) * motion_fps
        jvel_il = jvel_mj[IL_TO_MJ_DOF]
        jvel_frames.append(jvel_il.astype(np.float32))

        # root_rot is xyzw in the PKL (scipy convention)
        fq = m["root_rot"][fi]
        future_rot = Rot.from_quat(fq)
        relative = cur_rot.inv() * future_rot
        rot_mat = relative.as_matrix()
        ori_6d = np.concatenate([rot_mat[:, 0], rot_mat[:, 1]]).astype(np.float32)
        ori_frames.append(ori_6d)

    # Replicate IsaacLab's layout: cat([all_jpos_flat, all_jvel_flat]).reshape(10, 62)
    jpos_flat = np.concatenate(jpos_frames)  # (310,)
    jvel_flat = np.concatenate(jvel_frames)  # (310,)
    command_flat = np.concatenate([jpos_flat, jvel_flat])  # (620,)
    command_nonflat = command_flat.reshape(NUM_FUTURE_FRAMES, -1)  # (10, 62)

    ori_nonflat = np.stack(ori_frames)  # (10, 6)

    encoder_input = np.concatenate([command_nonflat, ori_nonflat], axis=-1)  # (10, 68)
    return encoder_input.reshape(-1).astype(np.float32)  # (680,)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--init-frame",
        type=int,
        default=0,
        help="Motion frame to RSI-initialize the robot at (default 0).",
    )
    parser.add_argument(
        "--fall-height",
        type=float,
        default=0.4,
        help="If pelvis world z drops below this (m), auto-reset (default 0.4).",
    )
    parser.add_argument(
        "--fall-tilt-cos",
        type=float,
        default=-0.3,
        help="If gravity in body frame's z component goes above this, auto-reset. "
        "-1.0 = perfectly upright, 0.0 = horizontal. Default -0.3 ~ 72° tilt.",
    )
    parser.add_argument(
        "--max-episode",
        type=float,
        default=0.0,
        help="If > 0, force a reset after this many simulated seconds (default 0 = no limit).",
    )
    args = parser.parse_args()

    print(f"Loading actor from {args.checkpoint} ...", flush=True)
    actor = load_actor_from_checkpoint(args.checkpoint, args.device)
    print("  Actor loaded.", flush=True)

    print(f"Loading motion from {args.motion} ...", flush=True)
    motion_data = load_motion_data(args.motion)
    total_frames = get_total_frames(motion_data)
    motion_fps = get_motion_fps(motion_data)
    print(f"  {total_frames} frames @ {motion_fps} fps = {total_frames / motion_fps:.1f}s",
          flush=True)

    print("Loading MuJoCo model ...", flush=True)
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = SIM_DT

    pelvis_id = mj_model.body("pelvis").id

    # ---- RSI: pull full robot state from motion frame `init_frame` ----
    init_frame = int(args.init_frame)
    init_motion_state = compute_motion_state(motion_data, init_frame, motion_fps)
    init_root_z = float(init_motion_state["root_pos_w"][2])
    print(f"  [RSI] Initializing from motion frame {init_frame} "
          f"(t={init_frame / motion_fps:.3f}s):", flush=True)
    print(f"    root_pos_w     = {init_motion_state['root_pos_w']}", flush=True)
    print(f"    root_quat_wxyz = {init_motion_state['root_quat_w_wxyz']}", flush=True)
    print(f"    root_lin_vel_w = {init_motion_state['root_lin_vel_w']}", flush=True)
    print(f"    root_ang_vel_w = {init_motion_state['root_ang_vel_w']}", flush=True)
    print(f"    joint_pos_mj[:6] = {init_motion_state['joint_pos_mj'][:6]}", flush=True)
    print(f"    joint_vel_mj[:6] = {init_motion_state['joint_vel_mj'][:6]}", flush=True)

    print(f"  Default DOF (MJ order): {DEFAULT_DOF}", flush=True)
    print(f"  Action scale (MJ order): {ACTION_SCALE}", flush=True)

    prop_buf = ProprioceptionBuffer()
    last_action_mj = np.zeros(NUM_DOFS, dtype=np.float32)
    sim_time = float(init_frame) / motion_fps
    step_count = 0
    episode_count = 0
    episode_start_step = 0
    paused = False

    def _apply_init_state():
        """Teleport the robot to the RSI motion frame state.

        Mirrors IsaacLab's ``write_joint_state_to_sim`` /
        ``write_root_state_to_sim`` at episode reset.
        """
        s = init_motion_state
        mj_data.qpos[0] = 0.0  # discard motion world XY offset
        mj_data.qpos[1] = 0.0
        mj_data.qpos[2] = float(s["root_pos_w"][2])
        mj_data.qpos[3:7] = s["root_quat_w_wxyz"]
        mj_data.qpos[7:7 + NUM_DOFS] = s["joint_pos_mj"]
        # MuJoCo free-joint qvel convention (verified empirically; see commit msg):
        #   qvel[0:3]  -> WORLD-frame linear velocity
        #   qvel[3:6]  -> BODY-LOCAL-frame angular velocity (NOT world!)
        # Motion lib stores both in world frame, so the angular component must
        # be rotated into the body frame before writing.
        mj_data.qvel[0:3] = s["root_lin_vel_w"]
        mj_data.qvel[3:6] = quat_rotate_inverse(
            s["root_quat_w_wxyz"], s["root_ang_vel_w"]
        )
        mj_data.qvel[6:6 + NUM_DOFS] = s["joint_vel_mj"]
        mj_data.xfrc_applied[:] = 0
        mujoco.mj_forward(mj_model, mj_data)

    _apply_init_state()

    def reset_state(reason=""):
        nonlocal sim_time, last_action_mj, episode_count, episode_start_step
        sim_time = float(init_frame) / motion_fps
        last_action_mj[:] = 0
        prop_buf.reset()
        _apply_init_state()
        episode_count += 1
        episode_start_step = step_count
        tag = f" ({reason})" if reason else ""
        print(f"\n[reset]{tag} starting episode {episode_count}", flush=True)

    def key_callback(keycode):
        nonlocal paused
        import glfw
        if keycode == glfw.KEY_SPACE:
            paused = not paused
            print("Paused" if paused else "Resumed", flush=True)
        elif keycode == glfw.KEY_R:
            reset_state("manual")
        elif keycode == glfw.KEY_V:
            if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = pelvis_id

    print("\n=== X2 MuJoCo Eval ===", flush=True)
    print(f"Robot RSI-initialized from motion frame {init_frame}.", flush=True)
    print(f"Auto-reset triggers: pelvis_z < {args.fall_height:.2f} m, "
          f"or gravity_body[z] > {args.fall_tilt_cos:.2f} (~tilt > "
          f"{int(np.rad2deg(np.arccos(-args.fall_tilt_cos)))}°).", flush=True)
    if args.max_episode > 0:
        print(f"Max episode length: {args.max_episode:.1f} s.", flush=True)
    print("Press SPACE pause, R reset, V toggle camera.\n", flush=True)

    with mujoco.viewer.launch_passive(
        mj_model, mj_data,
        key_callback=key_callback,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0.0, 0.0, init_root_z]
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id

        wall_start = time.time()
        # Motion-tracking sim_time is anchored to wall_start. Each reset
        # rebases wall_start so real-time pacing stays smooth across episodes.
        wall_start -= sim_time

        while viewer.is_running():
            if paused:
                viewer.sync()
                time.sleep(0.02)
                continue

            # -- Motion phase --
            motion_time = sim_time * args.speed
            motion_frame = int(motion_time * motion_fps) % total_frames
            motion_time = motion_frame / motion_fps

            # -- Read MuJoCo state --
            qpos_j = mj_data.qpos[7:7 + NUM_DOFS].copy()
            qvel_j = mj_data.qvel[6:6 + NUM_DOFS].copy()
            base_quat = mj_data.qpos[3:7].copy()  # wxyz
            # MuJoCo free-joint qvel[3:6] is in BODY-LOCAL frame (verified
            # empirically via quaternion integration). IsaacLab's
            # ``base_ang_vel`` proprioception term is also body-local, so we
            # consume qvel[3:6] directly — no rotation. Previously we
            # quat_rotate_inverse'd it under the wrong assumption that it was
            # world-frame, which caused a double rotation and corrupted the
            # angular-velocity history every step.
            base_angvel = mj_data.qvel[3:6].copy()

            # Convert MJ→IL using IL_TO_MJ_DOF as gather index:
            #   result[il_pos] = source[IL_TO_MJ_DOF[il_pos]]
            dof_pos_il = qpos_j[IL_TO_MJ_DOF]
            dof_vel_il = qvel_j[IL_TO_MJ_DOF]
            action_il = last_action_mj[IL_TO_MJ_DOF]

            gravity = quat_rotate_inverse(base_quat, np.array([0., 0., -1.]))
            dof_pos_rel_il = dof_pos_il - DEFAULT_DOF[IL_TO_MJ_DOF]

            # -- Build observation + run policy --
            prop_buf.append(gravity, base_angvel, dof_pos_rel_il, dof_vel_il, action_il)
            proprioception = prop_buf.get_flat()
            tokenizer_obs = build_tokenizer_obs(
                motion_data, motion_time, base_quat, motion_fps)

            with torch.no_grad():
                prop_t = torch.from_numpy(proprioception).unsqueeze(0).to(args.device)
                tok_t = torch.from_numpy(tokenizer_obs).unsqueeze(0).to(args.device)
                action_il_t = actor(prop_t, tok_t).squeeze(0).cpu().numpy()

            # Convert IL→MJ using MJ_TO_IL_DOF as gather index:
            #   result[mj_pos] = source[MJ_TO_IL_DOF[mj_pos]]
            action_mj = action_il_t[MJ_TO_IL_DOF]
            last_action_mj = action_mj.copy()
            target_pos = DEFAULT_DOF + action_mj * ACTION_SCALE

            # -- PD control + step --
            for _ in range(DECIMATION):
                torque = KP * (target_pos - mj_data.qpos[7:7 + NUM_DOFS]) \
                       - KD * mj_data.qvel[6:6 + NUM_DOFS]
                for j in range(NUM_DOFS):
                    mj_data.ctrl[JOINT_TO_ACTUATOR[j]] = torque[j]
                mujoco.mj_step(mj_model, mj_data)

            sim_time += CONTROL_DT
            step_count += 1
            viewer.sync()

            wall_elapsed = time.time() - wall_start
            if sim_time > wall_elapsed:
                time.sleep(sim_time - wall_elapsed)

            # -- Auto-reset on fall (IsaacLab-style termination) --
            pelvis_z = float(mj_data.qpos[2])
            grav_z = float(gravity[2])
            episode_seconds = (step_count - episode_start_step) * CONTROL_DT
            reset_reason = None
            if pelvis_z < args.fall_height:
                reset_reason = f"pelvis_z={pelvis_z:.3f} < {args.fall_height:.2f}"
            elif grav_z > args.fall_tilt_cos:
                reset_reason = (f"gravity_body[z]={grav_z:+.2f} > {args.fall_tilt_cos:.2f} "
                                f"(tilt {int(np.rad2deg(np.arccos(np.clip(-grav_z,-1,1))))}°)")
            elif args.max_episode > 0 and episode_seconds >= args.max_episode:
                reset_reason = f"reached --max-episode={args.max_episode:.1f}s"
            if reset_reason is not None:
                print(f"  [fall] ep={episode_count} ran {episode_seconds:.2f}s, "
                      f"reason: {reset_reason}", flush=True)
                reset_state(reset_reason)
                wall_start = time.time() - sim_time
                continue

            if step_count <= 5 or step_count % 250 == 0:
                h = mj_data.qpos[2]
                print(f"\n[ep {episode_count}] step={step_count}  sim={sim_time:.2f}s  "
                      f"frame={motion_frame}/{total_frames}  height={h:.3f}m",
                      flush=True)
                print(f"  gravity_body:  {gravity}", flush=True)
                print(f"  angvel_body:   {base_angvel}", flush=True)
                print(f"  base_quat:     {base_quat}", flush=True)
                print(f"  dof_pos_il[:6]: {dof_pos_il[:6]}", flush=True)
                print(f"  dof_vel_il[:6]: {dof_vel_il[:6]}", flush=True)
                print(f"  action_il[:6]:  {action_il_t[:6]}", flush=True)
                print(f"  action_mj[:6]:  {action_mj[:6]}", flush=True)
                print(f"  target_pos[:6]: {target_pos[:6]}", flush=True)
                print(f"  action |min|={np.abs(action_il_t).min():.4f} "
                      f"|max|={np.abs(action_il_t).max():.4f} "
                      f"|mean|={np.abs(action_il_t).mean():.4f}", flush=True)
                print(f"  prop shape={proprioception.shape} "
                      f"|min|={proprioception.min():.4f} |max|={proprioception.max():.4f}",
                      flush=True)
                print(f"  tok  shape={tokenizer_obs.shape} "
                      f"|min|={tokenizer_obs.min():.4f} |max|={tokenizer_obs.max():.4f}",
                      flush=True)

    print("Viewer closed.")


if __name__ == "__main__":
    main()
