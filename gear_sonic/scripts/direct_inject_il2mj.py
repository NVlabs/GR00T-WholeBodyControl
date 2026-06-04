"""
Sim-to-Sim deployment for Unitree H2 in MuJoCo.

Loads a trained SONIC checkpoint, runs closed-loop control in MuJoCo,
reproducing the IsaacLab observation pipeline exactly.

Observation pipeline (matching IsaacLab ObservationManager with history_length=10):
  Term-major layout:
    [gravity_dir_hist(10x3), base_ang_vel_hist(10x3), joint_pos_rel_hist(10x31),
     joint_vel_hist(10x31), last_action_hist(10x31)]
  Each term history is flattened oldest->newest before concatenation.
  Total: 10 * (3 + 3 + 31 + 31 + 31) = 990

Tokenizer pipeline (matching unitoken_all_noz.yaml):
  command_multi_future_nonflat: 10 frames * 68 = 680
  motion_anchor_ori_b_mf_nonflat: 10 frames * 6 = 60

Usage:
    python gear_sonic/scripts/eval_h2_mujoco.py \
        --checkpoint logs_rl/<run>/model_step_NNNNNN.pt \
        --motion gear_sonic/data/motions/h2_<motion>.pkl
"""

import argparse
import faulthandler
import joblib
import math
import os
import re
import time
from pathlib import Path

os.environ["GEAR_DISABLE_JIT"] = "1"

import mujoco
import numpy as np
from omegaconf import OmegaConf
import torch
import yaml
import scipy.spatial.transform

try:
    from mujoco import viewer as mujoco_viewer
except ImportError:
    mujoco_viewer = None

from gear_sonic.trl.utils.common import custom_instantiate


faulthandler.enable()

H2_NUM_DOF = 31
HISTORY_LEN = 10
CONTROL_DT = 0.02  # 50 Hz
MAIN_QPOS_DOF_SLICE = slice(7, 7 + H2_NUM_DOF)
MAIN_QVEL_DOF_SLICE = slice(6, 6 + H2_NUM_DOF)
GRAVITY = np.array([0.0, 0.0, -9.81])

# Per-term history layout (matching IsaacLab ObservationManager)
PER_FRAME_DIMS = {
    "gravity_dir": 3,
    "base_ang_vel": 3,
    "joint_pos_rel": 31,
    "joint_vel": 31,
    "last_action": 31,
}
PER_FRAME_TOTAL = sum(PER_FRAME_DIMS.values())  # 99

H2_MJCF_PATH = "gear_sonic/data/assets/robot_description/mjcf/h2_shadow.xml"


def motion_quat_xyzw_to_wxyz(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    return quat_xyzw[[3, 0, 1, 2]]


def quat_wxyz_to_xyzw(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    return quat_wxyz[[1, 2, 3, 0]]


def load_motion(path: str):
    obj = joblib.load(path)

    if isinstance(obj, dict) and "joint_pos" not in obj:
        key = next(iter(obj.keys()))
        print(f"[H2 Eval] using motion key: {key}")
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

    print(f"[H2 Eval] normalized motion keys: {list(obj.keys())}")
    if "root_rot" in obj:
        first_quat = np.asarray(obj["root_rot"][0], dtype=np.float32)
        print(f"[H2 Eval] motion root_rot convention: xyzw sample={first_quat}")
    return obj


def load_processed_motion_reference(path: str):
    # 읽기 전용 npz를 수정 가능한 딕셔너리로 변환
    ref = dict(np.load(path, allow_pickle=True))
    
    # 1프레임(단일 스텝) 데이터일 경우 인덱싱 에러를 방지하기 위해 
    # (1, N) 형태의 2D 배열로 시간 차원을 다시 감싸줍니다. (단, 토크나이저 데이터 제외)
    is_1d = False
    if "motion_ref_joint_pos" in ref and ref["motion_ref_joint_pos"].ndim == 1:
        is_1d = True
    elif "qpos_dof" in ref and ref["qpos_dof"].ndim == 1:
        is_1d = True
        
    if is_1d:
        for k, v in ref.items():
            if isinstance(v, np.ndarray) and not k.startswith("tok_"):
                ref[k] = v[None, ...]  # 차원 추가
                
    print(f"[H2 Eval] loaded processed motion reference: {path}")
    return ref


def _normalize_yaml_obj(obj):
    if isinstance(obj, dict):
        return {k: _normalize_yaml_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_yaml_obj(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _strip_python_path_tags(raw_text: str) -> str:
    pattern = re.compile(
        r"(?m)^(?P<indent>\s*)(?P<key>[A-Za-z0-9_]+):\s*!!python/object/apply:pathlib\.PosixPath\s*\n"
        r"(?P<items>(?:^(?P=indent)\s*- .*\n)+)"
    )

    def repl(match: re.Match) -> str:
        indent = match.group("indent")
        key = match.group("key")
        items_block = match.group("items")
        parts = []
        for line in items_block.splitlines():
            stripped = line.strip()
            if not stripped.startswith("-"):
                continue
            value = stripped[1:].strip()
            if value.startswith(('"', "'")) and value.endswith(('"', "'")):
                value = value[1:-1]
            parts.append(value)
        path_value = "/".join(part.strip("/") for part in parts if part not in {"/", ""})
        if items_block.lstrip().startswith("- /"):
            path_value = "/" + path_value
        return f'{indent}{key}: "{path_value}"\n'

    return pattern.sub(repl, raw_text)


def load_saved_model_configs(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    model_config_path = checkpoint_path.parent / "model_config.yaml"
    if model_config_path.exists():
        print(f"[H2 Eval] Loading model config: {model_config_path}")
        raw_text = model_config_path.read_text(encoding="utf-8")
        raw_text = _strip_python_path_tags(raw_text)
        raw_cfg = yaml.safe_load(raw_text)
        raw_cfg = _normalize_yaml_obj(raw_cfg)
        resolved_cfg = OmegaConf.create(raw_cfg)
        return resolved_cfg.env_config, resolved_cfg.algo_config

    config_path = checkpoint_path.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint_path.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing resolved model config: {model_config_path}. "
            f"Also could not find training config at {checkpoint_path.parent / 'config.yaml'} "
            f"or {checkpoint_path.parent.parent / 'config.yaml'}."
        )

    print(f"[H2 Eval] model_config.yaml missing; falling back to training config: {config_path}")
    raw_text = config_path.read_text(encoding="utf-8")
    raw_text = _strip_python_path_tags(raw_text)
    raw_cfg = yaml.safe_load(raw_text)
    raw_cfg = _normalize_yaml_obj(raw_cfg)
    train_cfg = OmegaConf.create(raw_cfg)

    env_config = train_cfg.manager_env.config
    algo_config = train_cfg.algo.config
    return env_config, algo_config


def ensure_inference_env_metadata(env_config, algo_config):
    if "obs" not in env_config:
        env_config.obs = OmegaConf.create({})
    if "group_obs_dims" not in env_config.obs:
        env_config.obs.group_obs_dims = OmegaConf.create({})
    if "group_obs_names" not in env_config.obs:
        env_config.obs.group_obs_names = OmegaConf.create({})
    if "obs_dims" not in env_config.obs:
        env_config.obs.obs_dims = OmegaConf.create({})
    if "robot" not in env_config:
        env_config.robot = OmegaConf.create({})
    if "algo_obs_dim_dict" not in env_config.robot:
        env_config.robot.algo_obs_dim_dict = OmegaConf.create({})

    if "tokenizer" not in env_config.obs.group_obs_dims:
        env_config.obs.group_obs_dims.tokenizer = OmegaConf.create({
            "encoder_index": [3],
            "command_multi_future_nonflat": [10, 62],
            "motion_anchor_ori_b_mf_nonflat": [10, 6],
        })
    if "tokenizer" not in env_config.obs.group_obs_names:
        env_config.obs.group_obs_names.tokenizer = [
            "encoder_index",
            "command_multi_future_nonflat",
            "motion_anchor_ori_b_mf_nonflat",
        ]

    env_config.obs.obs_dims.actor_obs = 990
    tokenizer_total_dim = int(sum(np.prod(v) for v in env_config.obs.group_obs_dims.tokenizer.values()))
    env_config.obs.obs_dims.tokenizer = tokenizer_total_dim
    env_config.robot.algo_obs_dim_dict.actor_obs = 990
    env_config.robot.algo_obs_dim_dict.tokenizer = tokenizer_total_dim
    if "actions_dim" not in env_config.robot:
        env_config.robot.actions_dim = H2_NUM_DOF

    return env_config, algo_config


def build_actor_from_checkpoint(checkpoint_path: str, device: torch.device):
    print("[H2 Eval] Resolving saved env/algo config")
    env_config, algo_config = load_saved_model_configs(checkpoint_path)
    env_config, algo_config = ensure_inference_env_metadata(env_config, algo_config)
    algo_config = OmegaConf.create(OmegaConf.to_container(algo_config, resolve=False))

    # Inference path does not use training aux losses. Disabling them avoids
    # importing loss modules that transitively trigger TorchScript at import time.
    if "backbone" in algo_config.actor:
        algo_config.actor.backbone.aux_loss_func = {}
        algo_config.actor.backbone.aux_loss_coef = {}
    algo_config.actor.has_aux_loss = False

    backbone_kwargs = {"active_encoders": ["g1"], "active_decoders": ["g1_dyn"]}

    print("[H2 Eval] Instantiating actor from saved config")
    actor = custom_instantiate(
        algo_config.actor,
        env_config=env_config,
        algo_config=algo_config,
        module_dim_dict={},
        backbone_kwargs=backbone_kwargs,
        _resolve=False,
    ).to(device)

    print(f"[H2 Eval] Loading checkpoint tensor state on {device}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    elif "actor_model_state_dict" in checkpoint:
        state_dict = checkpoint["actor_model_state_dict"]
    elif "policy" in checkpoint:
        state_dict = checkpoint["policy"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise KeyError(
            "Checkpoint does not contain a recognized policy state dict key. "
            f"Available keys: {sorted(checkpoint.keys())}"
        )

    model_uses_std = "std" in actor.state_dict()
    checkpoint_has_std = "std" in state_dict
    checkpoint_has_log_std = "log_std" in state_dict
    print(
        f"[H2 Eval] Actor noise param: {'std' if model_uses_std else 'log_std'} | "
        f"checkpoint: {'std' if checkpoint_has_std else 'log_std' if checkpoint_has_log_std else 'unknown'}"
    )
    if model_uses_std and checkpoint_has_log_std and not checkpoint_has_std:
        state_dict = dict(state_dict)
        state_dict["std"] = torch.exp(state_dict.pop("log_std"))
    elif not model_uses_std and checkpoint_has_std and not checkpoint_has_log_std:
        state_dict = dict(state_dict)
        state_dict["log_std"] = torch.log(state_dict.pop("std"))

    print("[H2 Eval] Applying policy state dict")
    missing, unexpected = actor.load_state_dict(state_dict, strict=False)
    actor.to(device)
    print(f"[H2 Eval] Loaded policy_state_dict from {checkpoint_path}")
    print(f"[H2 Eval] Missing keys ({len(missing)}): {missing}")
    print(f"[H2 Eval] Unexpected keys ({len(unexpected)}): {unexpected}")

    actor.eval()
    return actor, checkpoint, env_config, algo_config


# ===================================================================
# Joint name list: IsaacLab order (from h2py H2_ISAACLAB_JOINTS, skip pelvis)
# ===================================================================
IL_JOINT_NAMES = [
    "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link", "head_pitch_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_roll_link", "right_ankle_roll_link", "head_yaw_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]

# MuJoCo DFS joint order (from h2.xml, skip floating_base_joint)
MJ_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_roll_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_pitch_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Build IsaacLab ↔ MuJoCo DOF mapping by name
_IL_LINK_TO_MJ_JOINT = {
    n: n.rsplit("_link", 1)[0] + "_joint"
    for n in IL_JOINT_NAMES
}
_IL_LINK_TO_MJ_JOINT["torso_link"] = "waist_pitch_joint"

_mj_joint_to_dof = {
    n: i for i, n in enumerate(MJ_JOINT_NAMES)
}

_il_joint_to_dof = {
    _IL_LINK_TO_MJ_JOINT[n]: i
    for i, n in enumerate(IL_JOINT_NAMES)
}

# IsaacLab order index i -> MuJoCo order index
IL_TO_MJ_DOF = torch.tensor([
    _mj_joint_to_dof[_IL_LINK_TO_MJ_JOINT[il_name]]
    for il_name in IL_JOINT_NAMES
], dtype=torch.long)

# MuJoCo order index j -> IsaacLab order index
MJ_TO_IL_DOF = torch.tensor([
    _il_joint_to_dof[mj_name]
    for mj_name in MJ_JOINT_NAMES
], dtype=torch.long)


# ===================================================================
# Default joint positions (IsaacLab order, from h2.py init_state.joint_pos)
# ===================================================================
H2_DEFAULT_JOINT_POS = torch.tensor([
    -0.312,   # left_hip_pitch
    -0.312,   # right_hip_pitch
     0.000,   # waist_yaw
     0.000,   # left_hip_roll
     0.000,   # right_hip_roll
     0.000,   # waist_roll
     0.000,   # left_hip_yaw
     0.000,   # right_hip_yaw
     0.000,   # torso (waist_pitch)
     0.669,   # left_knee
     0.669,   # right_knee
     0.000,   # head_pitch
     0.200,   # left_shoulder_pitch
     0.200,   # right_shoulder_pitch
     0.000,   # left_ankle_roll
     0.000,   # right_ankle_roll
     0.000,   # head_yaw
     0.200,   # left_shoulder_roll
    -0.200,   # right_shoulder_roll
    -0.363,   # left_ankle_pitch
    -0.363,   # right_ankle_pitch
     0.000,   # left_shoulder_yaw
     0.000,   # right_shoulder_yaw
     0.600,   # left_elbow
     0.600,   # right_elbow
     0.000,   # left_wrist_roll
     0.000,   # right_wrist_roll
     0.000,   # left_wrist_pitch
     0.000,   # right_wrist_pitch
     0.000,   # left_wrist_yaw
     0.000,   # right_wrist_yaw
])

LEG_IDXS = torch.tensor([0, 1, 3, 4, 6, 7, 9, 10])      # hip + knee
FEET_IDXS = torch.tensor([14, 15, 19, 20])              # ankle roll/pitch
WAIST_IDXS = torch.tensor([2, 5, 8])                    # waist yaw/roll/pitch

# ===================================================================
# Action scale in IsaacLab order, computed from h2.py formula:
#   scale = 0.25 * effort_limit / stiffness
#   stiffness = armature * w_n^2 (optionally scaled by actuator group factor)
# ===================================================================
_ARMATURE_VALS = {
    "5020":    0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010":    0.00425,
}
_W_N = 10.0 * 2.0 * 3.1415926535  # must match h2.py NATURAL_FREQ

# (actuator_group, armature_key, stiffness_scale, effort)
_JOINT_PARAMS = [
    ("legs",     "7520_22", 1.0, 417.0),   # 0  left_hip_pitch
    ("legs",     "7520_22", 1.0, 417.0),   # 1  right_hip_pitch
    ("waist_yaw","7520_14", 1.0, 264.0),   # 2  waist_yaw
    ("legs",     "7520_22", 1.0, 417.0),   # 3  left_hip_roll
    ("legs",     "7520_22", 1.0, 417.0),   # 4  right_hip_roll
    ("waist",    "5020",    2.0, 150.0),   # 5  waist_roll
    ("legs",     "7520_14", 1.0, 264.0),   # 6  left_hip_yaw
    ("legs",     "7520_14", 1.0, 264.0),   # 7  right_hip_yaw
    ("waist",    "5020",    2.0, 150.0),   # 8  torso (waist_pitch)
    ("legs",     "7520_22", 1.0, 417.0),   # 9  left_knee
    ("legs",     "7520_22", 1.0, 417.0),   # 10 right_knee
    ("head",     "5020",    2.0, 150.0),   # 11 head_pitch
    ("arms",     "5020",    1.0, 75.0),    # 12 left_shoulder_pitch
    ("arms",     "5020",    1.0, 75.0),    # 13 right_shoulder_pitch
    ("feet",     "5020",    2.0, 150.0),   # 14 left_ankle_roll
    ("feet",     "5020",    2.0, 150.0),   # 15 right_ankle_roll
    ("head",     "5020",    2.0, 150.0),   # 16 head_yaw
    ("arms",     "5020",    1.0, 75.0),    # 17 left_shoulder_roll
    ("arms",     "5020",    1.0, 75.0),    # 18 right_shoulder_roll
    ("feet",     "5020",    2.0, 150.0),   # 19 left_ankle_pitch
    ("feet",     "5020",    2.0, 150.0),   # 20 right_ankle_pitch
    ("arms",     "5020",    1.0, 75.0),    # 21 left_shoulder_yaw
    ("arms",     "5020",    1.0, 75.0),    # 22 right_shoulder_yaw
    ("arms",     "5020",    1.0, 75.0),    # 23 left_elbow
    ("arms",     "5020",    1.0, 75.0),    # 24 right_elbow
    ("arms",     "5020",    1.0, 75.0),    # 25 left_wrist_roll
    ("arms",     "5020",    1.0, 75.0),    # 26 right_wrist_roll
    ("arms",     "4010",    1.0, 15.0),    # 27 left_wrist_pitch
    ("arms",     "4010",    1.0, 15.0),    # 28 right_wrist_pitch
    ("arms",     "4010",    1.0, 15.0),    # 29 left_wrist_yaw
    ("arms",     "4010",    1.0, 15.0),    # 30 right_wrist_yaw
]
H2_ACTION_SCALE = torch.tensor([
    0.25 * effort / (_ARMATURE_VALS[arm_key] * _W_N ** 2 * stiff_scale)
    for _, arm_key, stiff_scale, effort in _JOINT_PARAMS
])

# ===================================================================
# PD gains: computed from IsaacLab's implicit actuator armature
# stiffness = armature * w_n^2, damping = 2 * zeta * armature * w_n
# ===================================================================
W_N = 10.0 * 2.0 * math.pi
ZETA = 2.0

_ARMATURE_MAP = {
    "5020":    (0.003609725, 1.0),
    "7520_14": (0.010177520, 1.0),
    "7520_22": (0.025101925, 1.0),
    "4010":    (0.00425,     1.0),
}

def _compute_pd(armature_key, scale=1.0):
    arm = _ARMATURE_MAP[armature_key][0]
    return scale * arm * W_N ** 2, 2.0 * ZETA * scale * arm * W_N

# PD gains in IsaacLab joint order (see h2.py actuator groups)
_H2_KP, _H2_KD = zip(*[
    _compute_pd("7520_22"),   # left_hip_pitch
    _compute_pd("7520_22"),   # right_hip_pitch
    _compute_pd("7520_14"),   # waist_yaw
    _compute_pd("7520_22"),   # left_hip_roll
    _compute_pd("7520_22"),   # right_hip_roll
    _compute_pd("5020", 2.0), # waist_roll
    _compute_pd("7520_14"),   # left_hip_yaw
    _compute_pd("7520_14"),   # right_hip_yaw
    _compute_pd("5020", 2.0), # torso (waist_pitch)
    _compute_pd("7520_22"),   # left_knee
    _compute_pd("7520_22"),   # right_knee
    _compute_pd("5020", 2.0), # head_pitch
    _compute_pd("5020"),      # left_shoulder_pitch
    _compute_pd("5020"),      # right_shoulder_pitch
    _compute_pd("5020", 2.0), # left_ankle_roll
    _compute_pd("5020", 2.0), # right_ankle_roll
    _compute_pd("5020", 2.0), # head_yaw
    _compute_pd("5020"),      # left_shoulder_roll
    _compute_pd("5020"),      # right_shoulder_roll
    _compute_pd("5020", 2.0), # left_ankle_pitch
    _compute_pd("5020", 2.0), # right_ankle_pitch
    _compute_pd("5020"),      # left_shoulder_yaw
    _compute_pd("5020"),      # right_shoulder_yaw
    _compute_pd("5020"),      # left_elbow
    _compute_pd("5020"),      # right_elbow
    _compute_pd("5020"),      # left_wrist_roll
    _compute_pd("5020"),      # right_wrist_roll
    _compute_pd("4010"),      # left_wrist_pitch
    _compute_pd("4010"),      # right_wrist_pitch
    _compute_pd("4010"),      # left_wrist_yaw
    _compute_pd("4010"),      # right_wrist_yaw
])
_H2_KP = torch.tensor(_H2_KP)
_H2_KD = torch.tensor(_H2_KD)
# MuJoCo-order PD gains (fix: scatter IL-order values into MJ positions)
_H2_KP_MJ = _H2_KP[MJ_TO_IL_DOF]
_H2_KD_MJ = _H2_KD[MJ_TO_IL_DOF]

# Pre-compute: action_scale in MuJoCo order
H2_ACTION_SCALE_MJ = H2_ACTION_SCALE[MJ_TO_IL_DOF]

# ===================================================================
# IsaacLab armature and joint damping (matched actuator semantics)
# armature = _ARMATURE_MAP[key][0] * scale
# damping = 2 * zeta * armature * omega_n
# ===================================================================
H2_ARMATURE_IL = torch.tensor([
    _ARMATURE_MAP[arm_key][0] * stiff_scale
    for _, arm_key, stiff_scale, _ in _JOINT_PARAMS
])
H2_DAMPING_IL = 2.0 * ZETA * H2_ARMATURE_IL * W_N
# MuJoCo-order armature and damping
H2_ARMATURE_MJ = H2_ARMATURE_IL[MJ_TO_IL_DOF]
H2_DAMPING_MJ = H2_DAMPING_IL[MJ_TO_IL_DOF]


# ===================================================================
# Quaternion utilities (wxyz scalar-first, matching IsaacLab)
# ===================================================================
def quat_mul(a, b):
    return torch.stack([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ])

def quat_conjugate(q):
    return torch.tensor([q[0], -q[1], -q[2], -q[3]])

def quat_inv(q):
    return quat_conjugate(q)  # unit quaternion only

def quat_apply(q, v):
    """Rotate vector v by quaternion q (q * [0, v] * q^{-1})."""
    qv = torch.tensor([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[1:]

def quat_apply_inverse(q, v):
    return quat_apply(quat_conjugate(q), v)


# ===================================================================
# Circular buffer — broadcast-fill on reset, oldest-first on get
# ===================================================================
class CircularBuffer:
    """History buffer mimicking IsaacLab's observation manager.

    On reset: broadcast-fills entire buffer with init value.
    On append: rolls left, places newest at end.
    On get: returns flattened [oldest, ..., newest].
    """
    def __init__(self, history_len: int, dim: int):
        self.history_len = history_len
        self.dim = dim
        self.buf = torch.zeros(history_len, dim)
        self._primed = False

    def reset(self, init_val: torch.Tensor):
        self.buf[:] = init_val
        self._primed = True

    def append(self, obs: torch.Tensor):
        if not self._primed:
            self.buf[:] = obs
            self._primed = True
        else:
            self.buf = torch.roll(self.buf, -1, dims=0)
            self.buf[-1] = obs

    def get(self):
        return self.buf.flatten()


# ===================================================================
# FSQ quantization (from sim2sim gotcha G8)
# ===================================================================
def fsq_quantize(z: torch.Tensor, levels: int = 32):
    bound = (levels - 1) * 0.5
    z_bounded = bound * torch.tanh(z)
    z_quant = torch.round(z_bounded) / bound
    return z_quant


# ===================================================================
# Argument parser
# ===================================================================
def get_args():
    parser = argparse.ArgumentParser(description="H2 MuJoCo sim2sim evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--motion", type=str, required=True)
    parser.add_argument(
        "--processed-motion-reference",
        type=str,
        default=None,
        help="Optional IsaacLab step0 dump to use processed motion_ref/tok_* for initialization and step0 tokenizer parity",
    )
    parser.add_argument("--init-frame", type=int, default=0)
    parser.add_argument("--fall-height", type=float, default=0.4)
    parser.add_argument("--fall-tilt-cos", type=float, default=-0.3)
    parser.add_argument("--max-episode", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--no-viewer", action="store_true", help="Disable MuJoCo viewer (headless)")
    parser.add_argument(
        "--snap-to-ground",
        action="store_true",
        help="Lower root z after RSI so the lowest foot/robot geom touches ground",
    )
    parser.add_argument("--freeze-root", action="store_true", help="Freeze root body pose (pelvis) for joint-only dynamics debug")
    parser.add_argument("--debug-actuator-limits", action="store_true", help="Print detailed actuator ctrl/force/limit comparison at each log step")
    parser.add_argument("--remove-joint-force-limits", action="store_true", help="DANGEROUS: Remove joint-level actuatorfrcrange limits (set to ±10000). Can cause dynamics explosion.")
    parser.add_argument("--match-isaaclab-effort-limits", action="store_true", help="Set joint actuatorfrcrange to match IsaacLab effort limits (e.g. ankle_roll: ±19→±150)")
    parser.add_argument("--match-isaaclab-armature", action="store_true", help="Set joint armature to match IsaacLab per-motor values (adds rotor inertia to mass matrix)")
    parser.add_argument("--match-isaaclab-joint-damping", action="store_true", help="Set joint damping to match IsaacLab computed damping (2*zeta*armature*omega)")
    parser.add_argument("--debug-contact", action="store_true", help="Show contact debug in viewer/logs")
    parser.add_argument("--debug-contact-force", action="store_true", help="Show detailed contact force/normal/distance per contact")
    parser.add_argument("--disable-self-collision", action="store_true", help="Disable self-collision for non-foot geoms (set contype=0 on robot geoms except feet)")
    parser.add_argument("--debug-foot-transform", action="store_true", help="Print detailed foot/ankle body and collision geom world transforms at each log step")
    parser.add_argument("--debug-foot-kinematics", action="store_true", help="Print comprehensive foot kinematics: local transforms, body frames, FK comparison, rotation-aware OBB lowest point")
    parser.add_argument("--debug-joint-mapping", action="store_true", help="Print full joint mapping table at startup: IL→MJ target, qpos, axis, range, error, and sign-flip detection")
    parser.add_argument("--dump-foot-trajectory", type=str, default=None, help="Path to save per-step left/right foot world trajectory (.npz)")
    parser.add_argument("--zero-pd", action="store_true", help="Set PD torque to zero (kp_scale=0, kd_scale=0) to check baseline foot poses without active control")
    parser.add_argument("--debug-com", action="store_true", help="Show COM debug in viewer when available")
    parser.add_argument("--show-floor", action="store_true", help="Print ground geom info and keep floor visible")
    parser.add_argument("--dump-step0", type=str, default=None,
                        help="Path to save step-0 .npz dump (e.g. dump_step0/mujoco_step0.npz)")
    parser.add_argument(
        "--replay-reference",
        action="store_true",
        help="Use pure processed reference replay mode (skip policy, direct joint position target)",
    )
    parser.add_argument("--action-scale-mult", type=float, default=1.0)
    parser.add_argument("--score-out", type=str, default=None)
    parser.add_argument("--action-scale-legs", type=float, default=1.0)
    parser.add_argument("--action-scale-feet", type=float, default=1.0)
    parser.add_argument("--action-scale-waist", type=float, default=1.0)

    parser.add_argument("--kp-scale-legs", type=float, default=1.0)
    parser.add_argument("--kp-scale-feet", type=float, default=1.0)
    parser.add_argument("--kp-scale-waist", type=float, default=1.0)

    parser.add_argument("--kd-scale-legs", type=float, default=1.0)
    parser.add_argument("--kd-scale-feet", type=float, default=1.0)
    parser.add_argument("--kd-scale-waist", type=float, default=1.0)
    parser.add_argument("--root-z-offset", type=float, default=0.0)
    return parser.parse_args()


# ===================================================================
# Main deployment loop
# ===================================================================
def main():
    args = get_args()
    device = torch.device(args.device)

    if args.zero_pd:
        print("[ZERO_PD] Zeroing PD: setting kp_scale=0, kd_scale=0 — all PD torque disabled")
        args.kp_scale = 0.0
        args.kd_scale = 0.0

    # ------------------------------------------------------------------
    # 1. Load policy checkpoint and resolved model config
    # ------------------------------------------------------------------
    print(f"[H2 Eval] Loading checkpoint: {args.checkpoint}")
    actor, ckpt, saved_env_config, saved_algo_config = build_actor_from_checkpoint(
        args.checkpoint, device
    )
    fsq_levels = int(getattr(saved_algo_config.actor.backbone, "num_fsq_levels", 32))
    print(f"[H2 Eval] FSQ levels: {fsq_levels}")
    
    PROPRIO_DIM = int(saved_env_config.obs.obs_dims.actor_obs)
    TOKENIZER_DIM = int(saved_env_config.obs.obs_dims.tokenizer)
    print(f"[H2 Eval] Actor loaded. Proprio dim={PROPRIO_DIM}, tokenizer dim={TOKENIZER_DIM}")

    # ------------------------------------------------------------------
    # 2. Load motion library
    # ------------------------------------------------------------------
    print(f"[H2 Eval] Loading motion: {args.motion}")
    motion_lib = load_motion(args.motion)
    processed_motion_ref = load_processed_motion_reference(args.processed_motion_reference) if args.processed_motion_reference else None
    if args.replay_reference:
        print("[REPLAY] pure reference replay enabled (policy bypassed)")
    # Detect multi-frame trajectory vs single-frame step0 dump
    processed_ref_num_frames = 0
    if processed_motion_ref is not None:
        if "motion_ref_joint_pos" in processed_motion_ref:
            _jps = processed_motion_ref["motion_ref_joint_pos"]
        elif "qpos_dof" in processed_motion_ref:
            _jps = processed_motion_ref["qpos_dof"]
        else:
            _jps = None

        if _jps is not None:
            if _jps.ndim == 2:
                processed_ref_num_frames = _jps.shape[0]
                print(f"[PROCESSED_REF] Multi-frame trajectory: {processed_ref_num_frames} frames")
            elif _jps.ndim == 1:
                processed_ref_num_frames = 1
                print(f"[PROCESSED_REF] Single-frame reference")
    num_frames = motion_lib["joint_pos"].shape[0] if isinstance(motion_lib, dict) else 0
    print(f"[H2 Eval] Motion frames: {num_frames}, keys: {list(motion_lib.keys())}")

    # ------------------------------------------------------------------
    # 3. Load MuJoCo model
    # ------------------------------------------------------------------
    print(f"[H2 Eval] Loading MuJoCo model: {H2_MJCF_PATH}")
    mj_model = mujoco.MjModel.from_xml_path(H2_MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    mj_dof_pos = mj_data.qpos[MAIN_QPOS_DOF_SLICE]
    assert len(mj_dof_pos) == H2_NUM_DOF, f"MJ DOF mismatch: {len(mj_dof_pos)} vs {H2_NUM_DOF}"
    print(f"[H2 Eval] MuJoCo model loaded: {mj_model.nq} qpos, {mj_model.nv} qvel, {mj_model.nu} ctrl")

    # ------------------------------------------------------------------
    # Get MuJoCo joint names for verification
    # ------------------------------------------------------------------
    mj_joint_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                      for i in range(mj_model.njnt)]
    mj_actuator_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                         for i in range(mj_model.nu)]
    mj_geom_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                     for i in range(mj_model.ngeom)]
    print(f"[H2 Eval] MuJoCo joints ({len(mj_joint_names)}): {mj_joint_names}")
    print(f"[H2 Eval] MuJoCo actuators ({len(mj_actuator_names)}): {mj_actuator_names}")

    # Joint-level force limits (from h2.xml <joint actuatorfrcrange="...">)
    mj_joint_frcrange = [
        (mj_model.jnt_actfrcrange[i, :].copy() if mj_model.jnt_actfrcrange is not None
         else np.array([-1e10, 1e10], dtype=np.float64))
        for i in range(mj_model.njnt)
    ]
    # Map joint index to joint name
    joint_frcrange_by_name = {
        mj_joint_names[i]: mj_joint_frcrange[i]
        for i in range(mj_model.njnt) if mj_joint_names[i] != "floating_base_joint"
    }

    # IsaacLab effort limits (matching actuator groups in h2.py)
    IL_EFFORT_DICT = {
        "left_hip_pitch_joint": 417.0, "right_hip_pitch_joint": 417.0,
        "left_hip_roll_joint": 417.0,  "right_hip_roll_joint": 417.0,
        "left_knee_joint": 417.0,      "right_knee_joint": 417.0,
        "left_hip_yaw_joint": 264.0,   "right_hip_yaw_joint": 264.0,
        "waist_yaw_joint": 264.0,
        "left_ankle_roll_joint": 150.0,   "right_ankle_roll_joint": 150.0,
        "left_ankle_pitch_joint": 150.0,  "right_ankle_pitch_joint": 150.0,
        "waist_roll_joint": 150.0, "waist_pitch_joint": 150.0,
        "head_pitch_joint": 150.0, "head_yaw_joint": 150.0,
        "left_shoulder_pitch_joint": 75.0, "right_shoulder_pitch_joint": 75.0,
        "left_shoulder_roll_joint": 75.0,  "right_shoulder_roll_joint": 75.0,
        "left_shoulder_yaw_joint": 75.0,   "right_shoulder_yaw_joint": 75.0,
        "left_elbow_joint": 75.0,          "right_elbow_joint": 75.0,
        "left_wrist_roll_joint": 75.0,     "right_wrist_roll_joint": 75.0,
        "left_wrist_pitch_joint": 15.0,    "right_wrist_pitch_joint": 15.0,
        "left_wrist_yaw_joint": 15.0,      "right_wrist_yaw_joint": 15.0,
    }

    print("[JOINT_FRC_LIMITS]")
    limiting_joints = []
    for jname in MJ_JOINT_NAMES:
        xml_lo, xml_hi = joint_frcrange_by_name.get(jname, [-1e10, 1e10])
        il_limit = IL_EFFORT_DICT.get(jname)
        xml_limit = float(xml_hi) if xml_hi < 1e9 else float('inf')
        if il_limit is not None and xml_limit < il_limit:
            limiting_joints.append((jname, xml_limit, il_limit))
        il_str = f"IL={il_limit:7.1f}" if il_limit else "IL=N/A"
        print(f"  {jname:35s} XML actuatorfrcrange=[{xml_lo:8.2f}, {xml_hi:8.2f}]  {il_str}")
    if limiting_joints:
        print(f"[LIMIT_WARN] XML joint force limit < IsaacLab effort for {len(limiting_joints)} joints:")
        for jname, xml_lim, il_lim in limiting_joints:
            print(f"  {jname:35s} XML={xml_lim:.1f}  IL={il_lim:.1f}  ({(il_lim - xml_lim):+.1f} deficit)")
    else:
        print("[LIMIT_WARN] All XML joint force limits >= IsaacLab effort limits")

    # ------------------------------------------------------------------
    # Joint-level dynamics override section
    # ------------------------------------------------------------------
    # Build reverse lookup: MJ joint name → index in mj_model joint list
    mj_joint_name_to_idx = {name: i for i, name in enumerate(mj_joint_names)}

    # Apply --match-isaaclab-effort-limits
    if args.match_isaaclab_effort_limits:
        if not hasattr(mj_model, "jnt_actfrcrange"):
            print("[WARN] jnt_actfrcrange not available on MjModel; cannot match limits")
        else:
            match_count = 0
            for jname in MJ_JOINT_NAMES:
                il_effort = IL_EFFORT_DICT.get(jname)
                if il_effort is not None:
                    j_idx = mj_joint_name_to_idx[jname]
                    mj_model.jnt_actfrcrange[j_idx, 0] = -il_effort
                    mj_model.jnt_actfrcrange[j_idx, 1] = il_effort
                    match_count += 1
            print(f"[LIMIT_MATCH] Set {match_count} joints to IsaacLab effort limits")
            if match_count < len(MJ_JOINT_NAMES):
                print(f"[LIMIT_MATCH]  (skipped {len(MJ_JOINT_NAMES) - match_count} joints not in IL_EFFORT_DICT)")

    # Apply --remove-joint-force-limits (dangerous: complete removal, no upper bound)
    if args.remove_joint_force_limits:
        if not hasattr(mj_model, "jnt_actfrcrange"):
            print("[WARN] jnt_actfrcrange not available on MjModel; cannot remove limits")
        else:
            if args.match_isaaclab_effort_limits:
                print("[LIMIT_WARN] --remove-joint-force-limits overrides --match-isaaclab-effort-limits")
            for i in range(mj_model.njnt):
                if mj_joint_names[i] != "floating_base_joint":
                    mj_model.jnt_actfrcrange[i, 0] = -10000.0
                    mj_model.jnt_actfrcrange[i, 1] = 10000.0
            print("[LIMIT_OVERRIDE] All joint actuatorfrcrange set to [-10000, 10000] (limits removed — WARNING: dynamics instability expected)")

    # Apply --match-isaaclab-armature
    if args.match_isaaclab_armature:
        if not hasattr(mj_model, "dof_armature"):
            print("[WARN] dof_armature not available on MjModel; cannot match armature")
        else:
            for jname in MJ_JOINT_NAMES:
                mj_idx = _mj_joint_to_dof[jname]
                j_idx = mj_joint_name_to_idx[jname]
                dof_adr = mj_model.jnt_dofadr[j_idx]
                mj_model.dof_armature[dof_adr] = H2_ARMATURE_MJ[mj_idx].item()
            print(f"[ARMATURE_MATCH] Set {len(MJ_JOINT_NAMES)} joint armatures to IsaacLab values")

    # Apply --match-isaaclab-joint-damping
    if args.match_isaaclab_joint_damping:
        if not hasattr(mj_model, "dof_damping"):
            print("[WARN] dof_damping not available on MjModel; cannot match damping")
        else:
            for jname in MJ_JOINT_NAMES:
                mj_idx = _mj_joint_to_dof[jname]
                j_idx = mj_joint_name_to_idx[jname]
                dof_adr = mj_model.jnt_dofadr[j_idx]
                mj_model.dof_damping[dof_adr] = H2_DAMPING_MJ[mj_idx].item()
            print(f"[DAMPING_MATCH] Set {len(MJ_JOINT_NAMES)} joint dampings to IsaacLab computed values (2*zeta*armature*omega)")

    # ------------------------------------------------------------------
    # Startup diagnostic: per-joint dynamics parameter table
    # ------------------------------------------------------------------
    # =========================================================
    print("\n[IL_TO_MJ_DOF] MAPPING VERIFICATION")
    for i in range(H2_NUM_DOF):
        mj_idx = IL_TO_MJ_DOF[i].item()
        print(f"IL {i:02d} -> MJ {mj_idx:02d} : {MJ_JOINT_NAMES[mj_idx]}")
    print("="*60)
    # =========================================================
    print("[JOINT_DYNAMICS] per-joint parameters (IL order):")
    print(f"  {'joint_name':30s} {'armature':>10s} {'damping':>10s} {'frictionloss':>12s} {'effort':>8s}")
    for i, jname in enumerate(IL_JOINT_NAMES):
        arm = H2_ARMATURE_IL[i].item()
        damp = H2_DAMPING_IL[i].item()
        friction = 0.0  # not set in IsaacLab
        effort = _JOINT_PARAMS[i][3]
        print(f"  {jname:30s} {arm:10.6f} {damp:10.6f} {friction:12.6f} {effort:8.1f}")
    mj_arm = mj_model.dof_armature.copy() if hasattr(mj_model, "dof_armature") else None
    mj_damp = mj_model.dof_damping.copy() if hasattr(mj_model, "dof_damping") else None
    if mj_arm is not None:
        print(f"[MJ_DOF_ARMATURE] range=[{mj_arm.min():.6f}, {mj_arm.max():.6f}] mean={mj_arm.mean():.6f}  "
              f"(6 base DOFs + 31 joint DOFs, non-zero only after --match-isaaclab-armature)")
    if mj_damp is not None:
        print(f"[MJ_DOF_DAMPING]  range=[{mj_damp.min():.6f}, {mj_damp.max():.6f}] mean={mj_damp.mean():.6f}  "
              f"(non-zero only after --match-isaaclab-joint-damping)")
    print(f"[MJ_DOF_FRICTIONLOSS] not set (MuJoCo default=0)")
    print(f"[JOINT_FRC_LIMITS] see above for XML vs IsaacLab comparison")

    kp_il = _H2_KP.clone() * args.kp_scale
    kd_il = _H2_KD.clone() * args.kd_scale

    kp_il[LEG_IDXS] *= args.kp_scale_legs
    kp_il[FEET_IDXS] *= args.kp_scale_feet
    kp_il[WAIST_IDXS] *= args.kp_scale_waist

    kd_il[LEG_IDXS] *= args.kd_scale_legs
    kd_il[FEET_IDXS] *= args.kd_scale_feet
    kd_il[WAIST_IDXS] *= args.kd_scale_waist

    kp = kp_il[MJ_TO_IL_DOF]
    kd = kd_il[MJ_TO_IL_DOF]

    ground_name_tokens = ("ground", "floor", "plane", "terrain")
    support_name_tokens = ("sole", "foot", "ankle_pitch", "ankle")
    robot_geom_ids = [
        i for i, name in enumerate(mj_geom_names)
        if not any(tok in name.lower() for tok in ground_name_tokens)
    ]
    foot_geom_ids = [
        i for i in robot_geom_ids
        if any(tok in mj_geom_names[i].lower() for tok in support_name_tokens)
    ]
    ground_geom_ids = [
        i for i, name in enumerate(mj_geom_names)
        if any(tok in name.lower() for tok in ground_name_tokens)
    ]

    print(f"[GROUND] ngeom={mj_model.ngeom} contacts_enabled={not mj_model.opt.disableflags}")
    if ground_geom_ids:
        for geom_id in ground_geom_ids:
            print(
                f"[GROUND] plane geom name={mj_geom_names[geom_id]} contype={int(mj_model.geom_contype[geom_id])} "
                f"conaffinity={int(mj_model.geom_conaffinity[geom_id])} rgba={mj_model.geom_rgba[geom_id].tolist()}"
            )
    else:
        print("[GROUND] no plane/floor geom matched by name tokens")
    print(
        "[GROUND] foot geom names="
        + str([
            {
                "name": mj_geom_names[i],
                "contype": int(mj_model.geom_contype[i]),
                "conaffinity": int(mj_model.geom_conaffinity[i]),
            }
            for i in foot_geom_ids
        ])
    )

    # ------------------------------------------------------------------
    # Foot collision geometry audit
    # ------------------------------------------------------------------
    print("[FOOT_COLLISION_AUDIT]")
    foot_collision_geoms = [i for i, n in enumerate(mj_geom_names) if "foot" in n.lower()]
    if foot_collision_geoms:
        for gid in foot_collision_geoms:
            body_id = mj_model.geom_bodyid[gid]
            body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            geom_type = mj_model.geom_type[gid]
            pos = mj_model.geom_pos[gid].tolist()
            size = mj_model.geom_size[gid].tolist()
            # For boxes: size = [half_x, half_y, half_z]; for planes: [half_x, half_y, thickness]
            print(f"  geom={mj_geom_names[gid]:30s} body={body_name:30s} type={geom_type}"
                  f" pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f})"
                  f" size={size}"
                  f" contype={int(mj_model.geom_contype[gid])} conaffinity={int(mj_model.geom_conaffinity[gid])}"
                  f" friction={mj_model.geom_friction[gid].tolist()}")
            # solref, solimp, margin, condim
            if hasattr(mj_model, "geom_solref") and mj_model.geom_solref is not None:
                print(f"    solref={mj_model.geom_solref[gid].tolist()} solimp={mj_model.geom_solimp[gid].tolist()}"
                      f" margin={mj_model.geom_margin[gid]:.4f} condim={int(mj_model.geom_condim[gid])}")
    else:
        print("  [WARN] no geoms with 'foot' in name found")

    # List ALL collision-enabled robot geoms (contype>0) for self-collision audit
    collision_robot = [i for i in robot_geom_ids if mj_model.geom_contype[i] > 0]
    if collision_robot:
        print(f"  collision-enabled robot geoms ({len(collision_robot)}):")
        for gid in collision_robot:
            print(f"    geom={mj_geom_names[gid]:30s} contype={int(mj_model.geom_contype[gid])} "
                  f"conaffinity={int(mj_model.geom_conaffinity[gid])}")

    # ------------------------------------------------------------------
    # Apply --disable-self-collision (set contype=0 on all non-foot robot geoms)
    # ------------------------------------------------------------------
    if args.disable_self_collision:
        disabled_count = 0
        for gid in robot_geom_ids:
            if gid not in foot_geom_ids and mj_model.geom_contype[gid] > 0:
                mj_model.geom_contype[gid] = 0
                mj_model.geom_conaffinity[gid] = 0
                disabled_count += 1
        print(f"[SELFCOLL_OFF] Disabled self-collision on {disabled_count} non-foot robot geoms (only foot_collision geoms remain active)")

    # ------------------------------------------------------------------
    # IsaacLab foot body IDs (left_ankle_roll_link / right_ankle_roll_link)
    # IsaacLab treats the ankle_roll link itself as the foot for contact sensing.
    # ------------------------------------------------------------------
    il_foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    il_foot_body_ids = []
    for fname in il_foot_body_names:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, fname)
        il_foot_body_ids.append(bid if bid >= 0 else None)
        if bid >= 0:
            print(f"[FOOT_BODY] IsaacLab foot body '{fname}' → MuJoCo body_id={bid}")
        else:
            print(f"[FOOT_BODY] WARNING: IsaacLab foot body '{fname}' not found in MuJoCo model")

    # ------------------------------------------------------------------
    # Motion reference foot contact heuristic (body height < 0.05m + low velocity)
    # Uses body_pos_w from motion lib if available, otherwise estimates from root+joints.
    # ------------------------------------------------------------------
    ref_foot_contact_l = None
    ref_foot_contact_r = None
    if "body_pos_w" in motion_lib and motion_lib["body_pos_w"].shape[-1] == 3:
        # retargeted format with full body kinematics
        body_pos = motion_lib["body_pos_w"]  # (T, 32, 3)
        left_foot_body_idx = 6   # default from MotionLibBase for H2
        right_foot_body_idx = 12
        BODY_NAMES = ["pelvis", "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
                      "left_ankle_roll", "left_ankle_pitch", "right_hip_pitch", "right_hip_roll",
                      "right_hip_yaw", "right_knee", "right_ankle_roll", "right_ankle_pitch",
                      "waist_yaw", "waist_roll", "torso", "head_pitch", "head_yaw",
                      "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
                      "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
                      "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
                      "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
        foot_body_z_l = body_pos[:, left_foot_body_idx, 2]
        foot_body_z_r = body_pos[:, right_foot_body_idx, 2]
        ref_foot_contact_l = foot_body_z_l < 0.05
        ref_foot_contact_r = foot_body_z_r < 0.05
        n_contact_l = int(ref_foot_contact_l.sum())
        n_contact_r = int(ref_foot_contact_r.sum())
        print(f"[FOOT_REF] motion reference foot contact (body_z<0.05m): left={n_contact_l}/{len(ref_foot_contact_l)} frames  right={n_contact_r}/{len(ref_foot_contact_r)} frames")
        # Print first/last contact frames for gait phase understanding
        l_contact_frames = np.where(ref_foot_contact_l.cpu().numpy() if hasattr(ref_foot_contact_l, 'cpu') else ref_foot_contact_l)[0]
        r_contact_frames = np.where(ref_foot_contact_r.cpu().numpy() if hasattr(ref_foot_contact_r, 'cpu') else ref_foot_contact_r)[0]
        if len(l_contact_frames) > 0:
            print(f"[FOOT_REF]   left foot contact frame range: {l_contact_frames[0]}–{l_contact_frames[-1]}")
        if len(r_contact_frames) > 0:
            print(f"[FOOT_REF]   right foot contact frame range: {r_contact_frames[0]}–{r_contact_frames[-1]}")
        # Current frame status
        def get_foot_contact_at(frame):
            if frame < len(foot_body_z_l):
                return bool(foot_body_z_l[frame] < 0.05), bool(foot_body_z_r[frame] < 0.05)
            return None, None
    elif "body_pos" in motion_lib and motion_lib["body_pos"].ndim == 3:
        # SOMA format
        body_pos = motion_lib["body_pos"]
        # try indices 6 and 12 for ankle_roll
        for idx, side in [(6, "left"), (12, "right")]:
            if idx < body_pos.shape[1]:
                foot_z = body_pos[:, idx, 2]
                print(f"[FOOT_REF] {side} ankle body[z] min={foot_z.min():.4f} max={foot_z.max():.4f} mean={foot_z.mean():.4f}")
    else:
        print("[FOOT_REF] body_pos_w not available in motion lib; cannot estimate reference foot contact")

    actuator_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}" for i in range(mj_model.nu)]
    print(f"[PD] kp_scale={args.kp_scale} kd_scale={args.kd_scale}")
    print(f"[PD] kp range=({kp.min().item():.4f}, {kp.max().item():.4f})")
    print(f"[PD] kd range=({kd.min().item():.4f}, {kd.max().item():.4f})")
    print(
        "[PD] actuator ctrlrange="
        + str([
            {
                "name": actuator_names[i],
                "ctrllimited": bool(mj_model.actuator_ctrllimited[i]),
                "ctrlrange": mj_model.actuator_ctrlrange[i].tolist(),
                "forcerange": mj_model.actuator_forcerange[i].tolist(),
            }
            for i in range(mj_model.nu)
        ])
    )

    def compute_min_geom_z(preferred_geom_ids=None):
        geom_ids = preferred_geom_ids if preferred_geom_ids else robot_geom_ids
        if not geom_ids:
            return None, []
        z_values = mj_data.geom_xpos[geom_ids, 2]
        min_idx = int(np.argmin(z_values))
        min_geom_id = geom_ids[min_idx]
        return float(z_values[min_idx]), [(min_geom_id, mj_geom_names[min_geom_id])]

    def compute_foot_obb_lowest_z(geom_id):
        """Compute the lowest world-Z of a box geom (OBB, rotation-aware).
        Returns min_z and the corresponding corner index."""
        if mj_model.geom_type[geom_id] != 2:  # 2 = box type
            center = mj_data.geom_xpos[geom_id].copy()
            return float(center[2]), -1, center
        center = mj_data.geom_xpos[geom_id].copy()
        R = mj_data.geom_xmat[geom_id].copy().reshape(3, 3)
        hx, hy, hz = mj_model.geom_size[geom_id]
        corners_local = np.array([
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
            [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
        ])
        corners_world = center + corners_local @ R.T
        min_z_idx = int(np.argmin(corners_world[:, 2]))
        return float(corners_world[min_z_idx, 2]), min_z_idx, corners_world[min_z_idx]

    def collect_contact_pairs():
        pairs = []
        foot_pairs = []
        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            name1 = mj_geom_names[g1]
            name2 = mj_geom_names[g2]
            pair = f"{name1}<->{name2}"
            pairs.append(pair)
            if (g1 in foot_geom_ids or g2 in foot_geom_ids) and (g1 in ground_geom_ids or g2 in ground_geom_ids):
                foot_pairs.append(pair)
        return foot_pairs, pairs

    def compute_tracking_metrics(frame_idx: int):
        if processed_motion_ref is not None and processed_ref_num_frames > 0:
            ref_idx = min(frame_idx, processed_ref_num_frames - 1)

            if "motion_ref_root_pos_w" in processed_motion_ref:
                ref_root_pos = np.asarray(processed_motion_ref["motion_ref_root_pos_w"][ref_idx], dtype=np.float32)
                ref_root_quat_wxyz = np.asarray(processed_motion_ref["motion_ref_root_quat_w"][ref_idx], dtype=np.float32)
                ref_joint_pos_il = np.asarray(processed_motion_ref["motion_ref_joint_pos"][ref_idx], dtype=np.float32)
                ref_joint_vel_il = np.asarray(processed_motion_ref["motion_ref_joint_vel"][ref_idx], dtype=np.float32)
            else:
                ref_root_pos = np.asarray(processed_motion_ref["qpos_root"][ref_idx, :3], dtype=np.float32)
                ref_root_quat_wxyz = np.asarray(processed_motion_ref["qpos_root"][ref_idx, 3:7], dtype=np.float32)
                ref_joint_pos_il = np.asarray(processed_motion_ref["qpos_dof"][ref_idx], dtype=np.float32)
                ref_joint_vel_il = np.asarray(processed_motion_ref["qvel_dof"][ref_idx], dtype=np.float32)

            ref_root_quat_xyzw = quat_wxyz_to_xyzw(ref_root_quat_wxyz)

        else:
            idx = min(frame_idx, num_frames - 1)
            ref_root_pos = np.asarray(motion_lib["root_pos"][idx], dtype=np.float32)
            ref_root_quat_xyzw = np.asarray(motion_lib["root_rot"][idx], dtype=np.float32)
            ref_joint_pos_mj = np.asarray(motion_lib["joint_pos"][idx], dtype=np.float32)
            ref_joint_vel_mj = np.asarray(motion_lib["joint_vel"][idx], dtype=np.float32)

            ref_joint_pos_il = ref_joint_pos_mj[MJ_TO_IL_DOF.numpy()]
            ref_joint_vel_il = ref_joint_vel_mj[MJ_TO_IL_DOF.numpy()]

        sim_root_pos = mj_data.qpos[:3].copy()
        root_pos_err = sim_root_pos - ref_root_pos

        sim_root_quat_wxyz = mj_data.qpos[3:7].copy()
        rel_rot = scipy.spatial.transform.Rotation.from_quat(
            quat_wxyz_to_xyzw(sim_root_quat_wxyz)
        ) * scipy.spatial.transform.Rotation.from_quat(
            ref_root_quat_xyzw
        ).inv()

        root_ori_err_deg = float(np.linalg.norm(rel_rot.as_rotvec()) * 180.0 / np.pi)

        sim_joint_pos_il = mj_data.qpos[MAIN_QPOS_DOF_SLICE].copy()[MJ_TO_IL_DOF.numpy()]
        sim_joint_vel_il = mj_data.qvel[MAIN_QVEL_DOF_SLICE].copy()[MJ_TO_IL_DOF.numpy()]

        joint_pos_err = sim_joint_pos_il - ref_joint_pos_il
        joint_vel_err = sim_joint_vel_il - ref_joint_vel_il
        worst_pos_idx = int(np.argmax(np.abs(joint_pos_err)))
        worst_vel_idx = int(np.argmax(np.abs(joint_vel_err)))

        min_foot_z, min_foot_geoms = compute_min_geom_z(
            foot_geom_ids if foot_geom_ids else robot_geom_ids
        )
        foot_contact_pairs, all_contact_pairs = collect_contact_pairs()

        return {
            "sim_root_pos": sim_root_pos,
            "ref_root_pos": ref_root_pos,
            "root_pos_err": root_pos_err,
            "root_ori_err_deg": root_ori_err_deg,
            "ref_joint_pos_il": ref_joint_pos_il,
            "ref_joint_vel_il": ref_joint_vel_il,
            "sim_joint_pos_il": sim_joint_pos_il,
            "sim_joint_vel_il": sim_joint_vel_il,
            "joint_pos_err": joint_pos_err,
            "joint_vel_err": joint_vel_err,
            "joint_pos_rms": float(np.sqrt(np.mean(joint_pos_err ** 2))),
            "joint_vel_rms": float(np.sqrt(np.mean(joint_vel_err ** 2))),
            "joint_pos_max": float(np.max(np.abs(joint_pos_err))),
            "joint_vel_max": float(np.max(np.abs(joint_vel_err))),
            "worst_pos_joint": IL_JOINT_NAMES[worst_pos_idx],
            "worst_vel_joint": IL_JOINT_NAMES[worst_vel_idx],
            "min_foot_z": min_foot_z,
            "min_foot_geoms": min_foot_geoms,
            "foot_contact_pairs": foot_contact_pairs,
            "all_contact_pairs": all_contact_pairs,
        }

    def snap_root_to_ground_if_needed():
        candidate_ids = foot_geom_ids if foot_geom_ids else robot_geom_ids
        min_geom_z_before, min_geoms_before = compute_min_geom_z(candidate_ids)
        if min_geom_z_before is None:
            return
        root_z_before = float(mj_data.qpos[2])
        if min_geom_z_before > 0.0:
            mj_data.qpos[2] -= min_geom_z_before
            mujoco.mj_forward(mj_model, mj_data)
        min_geom_z_after, min_geoms_after = compute_min_geom_z(candidate_ids)
        root_z_after = float(mj_data.qpos[2])
        print(
            f"[H2 Eval] snap_to_ground root_z_before={root_z_before:.6f} "
            f"min_geom_z_before={min_geom_z_before:.6f} via={min_geoms_before} "
            f"root_z_after={root_z_after:.6f} min_geom_z_after={min_geom_z_after:.6f} via={min_geoms_after}"
        )

    # ------------------------------------------------------------------
    # 4. RSI — teleport robot to motion frame
    # ------------------------------------------------------------------
    def apply_rsi(frame_idx: int):
        idx = min(frame_idx, num_frames - 1)
        if processed_motion_ref is not None:
            _ref_idx = min(frame_idx, processed_ref_num_frames - 1) if processed_ref_num_frames > 1 else 0

            if "motion_ref_root_pos_w" in processed_motion_ref:
                root_pos = processed_motion_ref["motion_ref_root_pos_w"][_ref_idx]
                root_quat = processed_motion_ref["motion_ref_root_quat_w"][_ref_idx]
                joint_pos_il = torch.from_numpy(processed_motion_ref["motion_ref_joint_pos"][_ref_idx]).float()
                joint_vel_il = torch.from_numpy(processed_motion_ref["motion_ref_joint_vel"][_ref_idx]).float()
            else:
                root_pos = processed_motion_ref["qpos_root"][_ref_idx, :3]
                root_quat = processed_motion_ref["qpos_root"][_ref_idx, 3:7]
                joint_pos_il = torch.from_numpy(processed_motion_ref["qpos_dof"][_ref_idx]).float()
                joint_vel_il = torch.from_numpy(processed_motion_ref["qvel_dof"][_ref_idx]).float()

            root_quat_xyzw = quat_wxyz_to_xyzw(root_quat)
            root_vel_lin = processed_motion_ref["qvel_root_lin"][_ref_idx] if "qvel_root_lin" in processed_motion_ref else np.zeros(3, dtype=np.float32)
            root_vel_ang = processed_motion_ref["qvel_root_ang"][_ref_idx] if "qvel_root_ang" in processed_motion_ref else np.zeros(3, dtype=np.float32)
        else:
            root_pos = motion_lib["root_pos"][idx]
            root_quat_xyzw = motion_lib["root_rot"][idx]
            root_quat = motion_quat_xyzw_to_wxyz(root_quat_xyzw)
            root_vel_lin = motion_lib["root_vel_lin"][idx]
            root_vel_ang = motion_lib["root_vel_ang"][idx]
            joint_pos_il = torch.from_numpy(motion_lib["joint_pos"][idx]).float()
            joint_vel_il = torch.from_numpy(motion_lib["joint_vel"][idx]).float()

        joint_pos_mj = np.empty(H2_NUM_DOF, dtype=np.float32)
        joint_vel_mj = np.empty(H2_NUM_DOF, dtype=np.float32)
        joint_pos_mj[:] = joint_pos_il.numpy()  # motion pkl dof is MuJoCo order
        joint_vel_mj[:] = joint_vel_il.numpy()  # motion pkl dof is MuJoCo order

        mj_data.qpos[:3] = root_pos
        mj_data.qpos[3:7] = root_quat
        mj_data.qpos[MAIN_QPOS_DOF_SLICE] = joint_pos_mj
        mj_data.qpos[2] += args.root_z_offset

        # qvel: linear in world, angular in body frame (MuJoCo convention)
        mj_data.qvel[:3] = root_vel_lin
        mj_data.qvel[3:6] = quat_apply_inverse(
            torch.from_numpy(root_quat), torch.from_numpy(root_vel_ang)
        ).numpy()
        mj_data.qvel[MAIN_QVEL_DOF_SLICE] = joint_vel_mj

        mujoco.mj_forward(mj_model, mj_data)

        if args.snap_to_ground:
            snap_root_to_ground_if_needed()

        gravity_body = quat_apply_inverse(
            torch.from_numpy(root_quat).float(),
            torch.tensor([0.0, 0.0, -1.0]),
        ).numpy()
        print(
            f"[H2 Eval] apply_rsi frame={idx} root_quat_xyzw={root_quat_xyzw} "
            f"-> wxyz={root_quat} gravity_body={gravity_body} root_z={mj_data.qpos[2]:.6f}"
        )


    # ------------------------------------------------------------------
    # Shadow reference robot update
    # ------------------------------------------------------------------
    def update_shadow_ref(frame_idx: int):
        ref_idx = min(frame_idx, num_frames - 1)

        root_pos = np.asarray(motion_lib["root_pos"][ref_idx], dtype=np.float32)
        root_quat_xyzw = np.asarray(motion_lib["root_rot"][ref_idx], dtype=np.float32)
        root_quat = motion_quat_xyzw_to_wxyz(root_quat_xyzw)
        joint_pos_il = np.asarray(motion_lib["joint_pos"][ref_idx], dtype=np.float32)

        joint_pos_mj = np.empty(H2_NUM_DOF, dtype=np.float32)
        joint_pos_mj[:] = joint_pos_il

        ref_root_jid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ref_floating_base_joint"
        )
        if ref_root_jid < 0:
            return

        ref_qadr = mj_model.jnt_qposadr[ref_root_jid]
        ref_dadr = mj_model.jnt_dofadr[ref_root_jid]

        mj_data.qpos[ref_qadr:ref_qadr + 3] = root_pos
        mj_data.qpos[ref_qadr + 3:ref_qadr + 7] = root_quat
        mj_data.qvel[ref_dadr:ref_dadr + 6] = 0.0

        for i, jname in enumerate(MJ_JOINT_NAMES):
            jid = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"ref_{jname}"
            )
            if jid < 0:
                continue
            qadr = mj_model.jnt_qposadr[jid]
            dadr = mj_model.jnt_dofadr[jid]
            mj_data.qpos[qadr] = joint_pos_mj[i]
            mj_data.qvel[dadr] = 0.0

    apply_rsi(args.init_frame)
    frozen_root_qpos = mj_data.qpos[:7].copy()
    frozen_root_qvel = mj_data.qvel[:6].copy()

    # ------------------------------------------------------------------
    # Joint mapping debug (RSI 후 IL→MJ mapping 검증)
    # ------------------------------------------------------------------
    if args.debug_joint_mapping:
        # Get reference joint positions in IL order
        if processed_motion_ref is not None:
            _ref_idx = min(args.init_frame, processed_ref_num_frames - 1) if processed_ref_num_frames > 1 else 0
            if processed_ref_num_frames > 1:
                ref_il = torch.from_numpy(processed_motion_ref["motion_ref_joint_pos"][_ref_idx]).float()
                ref_vel_il = torch.from_numpy(processed_motion_ref["motion_ref_joint_vel"][_ref_idx]).float()
            else:
                ref_il = torch.from_numpy(processed_motion_ref["motion_ref_joint_pos"]).float()
                ref_vel_il = torch.from_numpy(processed_motion_ref["motion_ref_joint_vel"]).float()
        else:
            ref_il = torch.from_numpy(motion_lib["joint_pos"][args.init_frame]).float()
            ref_vel_il = torch.from_numpy(motion_lib["joint_vel"][args.init_frame]).float()
        ref_mj = torch.empty(H2_NUM_DOF)
        ref_mj[IL_TO_MJ_DOF] = ref_il
        qpos_mj = torch.from_numpy(mj_data.qpos[MAIN_QPOS_DOF_SLICE].copy())
        qvel_mj = torch.from_numpy(mj_data.qvel[MAIN_QVEL_DOF_SLICE].copy())
        qpos_il = qpos_mj[IL_TO_MJ_DOF]

        print("\n[JOINT_MAPPING] RSI initial state — IL→MJ mapping table:")
        header = (f"{'IL_idx':>6s} {'MJ_idx':>6s} {'IL_joint_name':30s} {'MJ_joint_name':30s} "
                  f"{'axis':>20s} {'range':>14s} "
                  f"{'ref_IL':>8s} {'ref_MJ':>8s} {'qpos_MJ':>8s} {'err_MJ':>8s} "
                  f"{'ref_IL_vel':>10s} {'qvel_MJ':>10s}")
        print(header)
        print("-" * len(header))

        suspect_joints = ["left_hip_roll", "right_hip_roll", "left_ankle_roll", "right_ankle_roll",
                          "left_ankle_pitch", "right_ankle_pitch", "left_hip_yaw", "right_hip_yaw"]

        # Build axis string for each MJ joint
        mj_axis_strs = {}
        for mj_idx, jname in enumerate(MJ_JOINT_NAMES):
            # Find the joint index in the model
            for ji in range(mj_model.njnt):
                if mj_joint_names[ji] == jname:
                    ax = mj_model.jnt_axis[ji]
                    mj_axis_strs[mj_idx] = f"({ax[0]:.2f},{ax[1]:.2f},{ax[2]:.2f})"
                    break

        for il_i, il_name in enumerate(IL_JOINT_NAMES):
            mj_i = IL_TO_MJ_DOF[il_i].item()
            mj_name = MJ_JOINT_NAMES[mj_i]
            ax_str = mj_axis_strs.get(mj_i, "N/A")
            # joint range from MuJoCo
            jrange_lo, jrange_hi = mj_model.jnt_range[0], mj_model.jnt_range[1]  # placeholder
            # Get actual range from the joint
            for ji in range(mj_model.njnt):
                if mj_joint_names[ji] == mj_name:
                    jrange_lo = mj_model.jnt_range[ji, 0]
                    jrange_hi = mj_model.jnt_range[ji, 1]
                    break
            rng_str = f"[{jrange_lo:.3f},{jrange_hi:.3f}]"
            ref_il_v = ref_il[il_i].item()
            ref_mj_v = ref_mj[mj_i].item()
            qpos_v = qpos_mj[mj_i].item()
            err_v = qpos_v - ref_mj_v
            ref_vel_v = ref_vel_il[il_i].item()
            qvel_v = qvel_mj[mj_i].item()

            # Check if any IL name contains suspect joints
            marker = " <===" if any(s in il_name for s in suspect_joints) else ""

            print(f"{il_i:6d} {mj_i:6d} {il_name:30s} {mj_name:30s} "
                  f"{ax_str:>20s} {rng_str:>14s} "
                  f"{ref_il_v:8.4f} {ref_mj_v:8.4f} {qpos_v:8.4f} {err_v:8.4f} "
                  f"{ref_vel_v:10.4f} {qvel_v:10.4f}{marker}")

        # Sign flip detection: compare |target - qpos| vs |-target - qpos|
        print("\n[JOINT_MAPPING] Sign-flip candidates (|ref - qpos| vs |-ref - qpos|):")
        flip_candidates = []
        for il_i, il_name in enumerate(IL_JOINT_NAMES):
            mj_i = IL_TO_MJ_DOF[il_i].item()
            ref_mj_v = ref_mj[mj_i].item()
            qpos_v = qpos_mj[mj_i].item()
            err_orig = abs(qpos_v - ref_mj_v)
            err_flip = abs(qpos_v + ref_mj_v)
            if err_flip < err_orig * 0.8 and err_orig > 0.01:
                improvement = (err_orig - err_flip) / err_orig * 100
                flip_candidates.append((il_i, mj_i, il_name, MJ_JOINT_NAMES[mj_i],
                                        ref_mj_v, qpos_v, err_orig, err_flip, improvement))
        if flip_candidates:
            print(f"  {'IL_idx':>6s} {'MJ_idx':>6s} {'IL_joint':30s} {'MJ_joint':30s} "
                  f"{'ref_MJ':>8s} {'qpos':>8s} {'err_orig':>9s} {'err_flip':>9s} {'improv%':>7s}")
            for c in flip_candidates:
                print(f"  {c[0]:6d} {c[1]:6d} {c[2]:30s} {c[3]:30s} "
                      f"{c[4]:8.4f} {c[5]:8.4f} {c[6]:9.4f} {c[7]:9.4f} {c[8]:6.1f}%")
        else:
            print("  (no significant sign-flip candidates detected)")

        # Focus joints detailed output
        print("\n[JOINT_MAPPING] Focus joints (suspected sign/axis issues):")
        focus_names = [("left_hip_roll_link", "left_hip_roll_joint"),
                       ("right_hip_roll_link", "right_hip_roll_joint"),
                       ("left_ankle_roll_link", "left_ankle_roll_joint"),
                       ("right_ankle_roll_link", "right_ankle_roll_joint"),
                       ("left_ankle_pitch_link", "left_ankle_pitch_joint"),
                       ("right_ankle_pitch_link", "right_ankle_pitch_joint"),
                       ("left_hip_yaw_link", "left_hip_yaw_joint"),
                       ("right_hip_yaw_link", "right_hip_yaw_joint")]
        for il_link, mj_joint_name in focus_names:
            il_idx = IL_JOINT_NAMES.index(il_link) if il_link in IL_JOINT_NAMES else -1
            if il_idx < 0:
                continue
            mj_idx = MJ_JOINT_NAMES.index(mj_joint_name) if mj_joint_name in MJ_JOINT_NAMES else -1
            if mj_idx < 0:
                continue
            # Find model joint index
            for ji in range(mj_model.njnt):
                if mj_joint_names[ji] == mj_joint_name:
                    ax = mj_model.jnt_axis[ji]
                    jr = mj_model.jnt_range[ji]
                    break
            ref_il_v = ref_il[il_idx].item()
            ref_mj_v = ref_mj[mj_idx].item()
            qpos_v = qpos_mj[mj_idx].item()
            # Body world pose for this joint's child body
            body_name = il_link  # the child body of the joint
            bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid >= 0:
                xpos = mj_data.xpos[bid]
                xquat = mj_data.xquat[bid]
                body_str = f"body_pos=({xpos[0]:.4f},{xpos[1]:.4f},{xpos[2]:.4f})"
            else:
                body_str = "body=N/A"
            print(f"  {il_link:30s} → {mj_joint_name:30s}  "
                  f"axis=({ax[0]:.4f},{ax[1]:.4f},{ax[2]:.4f})  range=[{jr[0]:.3f},{jr[1]:.3f}]  "
                  f"ref_IL={ref_il_v:8.4f}  ref_MJ={ref_mj_v:8.4f}  qpos={qpos_v:8.4f}  err={qpos_v - ref_mj_v:8.4f}  "
                  f"{body_str}")

        # World pose summary
        print("\n[JOINT_MAPPING] World pose after RSI:")
        pelvis_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_bid >= 0:
            pp = mj_data.xpos[pelvis_bid]
            pq = mj_data.xquat[pelvis_bid]
            print(f"  pelvis  pos=({pp[0]:.4f},{pp[1]:.4f},{pp[2]:.4f})  quat=({pq[0]:.4f},{pq[1]:.4f},{pq[2]:.4f},{pq[3]:.4f})")
        for side in ["left", "right"]:
            for body_name in [f"{side}_ankle_roll_link", f"{side}_ankle_pitch_link"]:
                bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if bid >= 0:
                    bp = mj_data.xpos[bid]
                    bq = mj_data.xquat[bid]
                    print(f"  {body_name:30s}  pos=({bp[0]:.4f},{bp[1]:.4f},{bp[2]:.4f})  "
                          f"quat=({bq[0]:.4f},{bq[1]:.4f},{bq[2]:.4f},{bq[3]:.4f})")
        print()

    # ------------------------------------------------------------------
    # Foot kinematics diagnostic (local transforms, body frames, FK comparison)
    # ------------------------------------------------------------------
    if args.debug_foot_kinematics:
        print("\n[FOOT_KINEMATICS] Left/Right foot collision geom local transform comparison:")
        for side in ["left", "right"]:
            cg_name = f"{side}_foot_collision"
            cgids = [i for i, n in enumerate(mj_geom_names) if n == cg_name]
            if not cgids:
                print(f"  {side}: no collision geom")
                continue
            cgid = cgids[0]
            body_id = mj_model.geom_bodyid[cgid]
            body_name = mj_joint_names[body_id] if body_id < len(mj_joint_names) else f"body_{body_id}"
            body_name_str = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            geom_pos_local = mj_model.geom_pos[cgid]
            geom_quat_local = mj_model.geom_quat[cgid]
            geom_type = mj_model.geom_type[cgid]
            # World transform
            geom_xpos = mj_data.geom_xpos[cgid]
            geom_xmat = mj_data.geom_xmat[cgid].copy().reshape(3, 3)
            # OBB lowest point
            obb_z, obb_corner_idx, obb_corner_pos = compute_foot_obb_lowest_z(cgid)
            print(f"  {side:5s} collision_geom={cg_name}")
            print(f"         parent_body={body_name_str:30s} body_id={body_id}")
            print(f"         local_pos=({geom_pos_local[0]:.6f},{geom_pos_local[1]:.6f},{geom_pos_local[2]:.6f})  "
                  f"local_quat=({geom_quat_local[0]:.6f},{geom_quat_local[1]:.6f},{geom_quat_local[2]:.6f},{geom_quat_local[3]:.6f})")
            print(f"         type={geom_type} size=({mj_model.geom_size[cgid][0]:.4f},{mj_model.geom_size[cgid][1]:.4f},{mj_model.geom_size[cgid][2]:.4f})")
            print(f"         world_pos=({geom_xpos[0]:.6f},{geom_xpos[1]:.6f},{geom_xpos[2]:.6f})")
            print(f"         world_R=[[{geom_xmat[0,0]:.4f},{geom_xmat[0,1]:.4f},{geom_xmat[0,2]:.4f}],"
                  f"[{geom_xmat[1,0]:.4f},{geom_xmat[1,1]:.4f},{geom_xmat[1,2]:.4f}],"
                  f"[{geom_xmat[2,0]:.4f},{geom_xmat[2,1]:.4f},{geom_xmat[2,2]:.4f}]]")
            print(f"         OBB_lowest_z={obb_z:.6f}  (corner {obb_corner_idx}: "
                  f"({obb_corner_pos[0]:.4f},{obb_corner_pos[1]:.4f},{obb_corner_pos[2]:.4f}))")

        print("[FOOT_KINEMATICS] Left/Right ankle body frame comparison:")
        for side in ["left", "right"]:
            for body_name in [f"{side}_ankle_roll_link", f"{side}_ankle_pitch_link"]:
                bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if bid < 0:
                    continue
                bp = mj_data.xpos[bid]
                bq = mj_data.xquat[bid]
                # Parent body
                parent_id = mj_model.body_parentid[bid]
                parent_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or f"body_{parent_id}"
                pp = mj_data.xpos[parent_id]
                rel_pos = bp - pp
                print(f"  {body_name:30s}  world_pos=({bp[0]:.6f},{bp[1]:.6f},{bp[2]:.6f})  "
                      f"world_quat=({bq[0]:.6f},{bq[1]:.6f},{bq[2]:.6f},{bq[3]:.6f})  "
                      f"parent={parent_name:25s}  parent_pos=({pp[0]:.4f},{pp[1]:.4f},{pp[2]:.4f})  "
                      f"rel2parent=({rel_pos[0]:.4f},{rel_pos[1]:.4f},{rel_pos[2]:.4f})")

        # FK recomputation: run mj_forward with reference joint positions on a copy
        print("[FOOT_KINEMATICS] FK recomputation from IsaacLab reference joint positions:")
        _saved_qpos = mj_data.qpos.copy()
        _saved_qvel = mj_data.qvel.copy()
        # Get reference in MJ order
        if processed_motion_ref is not None:
            if processed_ref_num_frames > 1:
                _ref_il = torch.from_numpy(processed_motion_ref["motion_ref_joint_pos"][args.init_frame]).float()
            else:
                _ref_il = torch.from_numpy(processed_motion_ref["motion_ref_joint_pos"]).float()
        else:
            _ref_il = torch.from_numpy(motion_lib["joint_pos"][args.init_frame]).float()
        _ref_mj = np.empty(H2_NUM_DOF, dtype=np.float32)
        _ref_mj[IL_TO_MJ_DOF.numpy()] = _ref_il.numpy()
        mj_data.qpos[MAIN_QPOS_DOF_SLICE] = _ref_mj
        mujoco.mj_forward(mj_model, mj_data)
        for side in ["left", "right"]:
            for body_name in [f"{side}_ankle_roll_link", f"{side}_ankle_pitch_link"]:
                bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if bid >= 0:
                    bp = mj_data.xpos[bid]
                    print(f"  FK({body_name:30s})  pos=({bp[0]:.6f},{bp[1]:.6f},{bp[2]:.6f})")
            cg_name = f"{side}_foot_collision"
            cgids = [i for i, n in enumerate(mj_geom_names) if n == cg_name]
            if cgids:
                cgid = cgids[0]
                cgp = mj_data.geom_xpos[cgid]
                obb_z_fk, _, _ = compute_foot_obb_lowest_z(cgid)
                print(f"  FK({cg_name:30s})  pos=({cgp[0]:.6f},{cgp[1]:.6f},{cgp[2]:.6f})  OBB_lowest_z={obb_z_fk:.6f}")
        # Restore
        mj_data.qpos[:] = _saved_qpos
        mj_data.qvel[:] = _saved_qvel
        mujoco.mj_forward(mj_model, mj_data)
        print()

    if args.freeze_root:
        # Verification: save foot body positions before/after re-applying frozen root
        _pre_foot_pos = {}
        for _side_name in ["left", "right"]:
            _bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{_side_name}_ankle_roll_link")
            if _bid >= 0:
                _pre_foot_pos[_side_name] = mj_data.xpos[_bid].copy()
        # Simulate freeze-root: restore root and run mj_forward
        _saved_jpos = mj_data.qpos[MAIN_QPOS_DOF_SLICE].copy()
        mj_data.qpos[:7] = frozen_root_qpos
        mj_data.qvel[:6] = frozen_root_qvel
        mujoco.mj_forward(mj_model, mj_data)
        # Restore joint positions
        mj_data.qpos[MAIN_QPOS_DOF_SLICE] = _saved_jpos
        mujoco.mj_forward(mj_model, mj_data)
        # Compare foot positions
        for _side_name in ["left", "right"]:
            _bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{_side_name}_ankle_roll_link")
            if _bid >= 0 and _side_name in _pre_foot_pos:
                _post = mj_data.xpos[_bid].copy()
                _pre = _pre_foot_pos[_side_name]
                _delta = _post - _pre
                print(f"[FREEZE_VERIFY] {_side_name}_ankle_roll_link body after re-freeze: "
                      f"pre=({_pre[0]:.4f},{_pre[1]:.4f},{_pre[2]:.4f})  "
                      f"post=({_post[0]:.4f},{_post[1]:.4f},{_post[2]:.4f})  "
                      f"delta=({_delta[0]:.6f},{_delta[1]:.6f},{_delta[2]:.6f})")
        print("[FREEZE] root body pose frozen to initial RSI state (verified: kinematics preserved)")

    # ------------------------------------------------------------------
    # 4b. Step-0 dump (before any control steps)
    # ------------------------------------------------------------------
    if args.dump_step0:
        os.makedirs(os.path.dirname(args.dump_step0), exist_ok=True)
        base_quat = torch.from_numpy(mj_data.qpos[3:7].copy())
        gravity_w = torch.tensor([0.0, 0.0, -1.0])
        gravity_body = quat_apply_inverse(base_quat, gravity_w)

        joint_pos_mj = torch.from_numpy(mj_data.qpos[MAIN_QPOS_DOF_SLICE].copy())
        joint_vel_mj = torch.from_numpy(mj_data.qvel[MAIN_QVEL_DOF_SLICE].copy())
        joint_pos_il = joint_pos_mj[IL_TO_MJ_DOF]
        joint_vel_il = joint_vel_mj[IL_TO_MJ_DOF]

        dump = {
            "qpos": mj_data.qpos.copy(),
            "qvel": mj_data.qvel.copy(),
            "qpos_root": mj_data.qpos[:7].copy(),
            "qpos_dof": mj_data.qpos[MAIN_QPOS_DOF_SLICE].copy(),
            "qvel_root_lin": mj_data.qvel[:3].copy(),
            "qvel_root_ang": mj_data.qvel[3:6].copy(),
            "qvel_dof": mj_data.qvel[MAIN_QVEL_DOF_SLICE].copy(),
            "gravity_body": gravity_body.numpy(),
            "joint_pos": joint_pos_il.numpy(),
            "joint_vel": joint_vel_il.numpy(),
            "joint_pos_rel": (joint_pos_il - H2_DEFAULT_JOINT_POS).numpy(),
            "default_joint_pos": H2_DEFAULT_JOINT_POS.numpy(),
            "actions_raw": np.zeros(H2_NUM_DOF),
            "actions_scaled": np.zeros(H2_NUM_DOF),
        }

        # Proprioception follows IsaacLab's term-major history flattening.
        term_values = [
            gravity_body.numpy(),
            mj_data.qvel[3:6].copy(),
            (joint_pos_il - H2_DEFAULT_JOINT_POS).numpy(),
            joint_vel_il.numpy(),
            np.zeros(H2_NUM_DOF),
        ]
        proprio = np.concatenate([np.tile(term, HISTORY_LEN) for term in term_values])
        dump["policy_obs"] = proprio

        # Per-frame breakdown reconstructed from term-major layout
        term_histories = {}
        off = 0
        for term_name, dim in PER_FRAME_DIMS.items():
            span = HISTORY_LEN * dim
            hist = proprio[off:off + span].reshape(HISTORY_LEN, dim)
            dump[f"history_{term_name}"] = hist
            term_histories[term_name] = hist
            off += span

        for f in range(HISTORY_LEN):
            for term_name, dim in PER_FRAME_DIMS.items():
                dump[f"frame{f}_{term_name}"] = term_histories[term_name][f]

        np.savez(args.dump_step0, **dump)
        n_keys = len(dump)
        print(f"[H2 Eval] Step-0 dump saved to {args.dump_step0} ({n_keys} arrays)")

    # ------------------------------------------------------------------
    # 5. Initialize observation buffers
    # ------------------------------------------------------------------
    # Per IsaacLab ObservationManager:
    # each term keeps its own history buffer, each history is flattened, then
    # the term blocks are concatenated in config order.
    buf_gravity = CircularBuffer(HISTORY_LEN, 3)
    buf_ang_vel = CircularBuffer(HISTORY_LEN, 3)
    buf_joint_pos = CircularBuffer(HISTORY_LEN, H2_NUM_DOF)
    buf_joint_vel = CircularBuffer(HISTORY_LEN, H2_NUM_DOF)
    buf_actions = CircularBuffer(HISTORY_LEN, H2_NUM_DOF)

    last_action_il = torch.zeros(H2_NUM_DOF)

    def read_obs():
        """Read current MuJoCo state and return IsaacLab-frame observation terms."""
        base_quat = torch.from_numpy(mj_data.qpos[3:7].copy())
        base_ang_vel = torch.from_numpy(mj_data.qvel[3:6].copy())

        joint_pos_mj = torch.from_numpy(mj_data.qpos[MAIN_QPOS_DOF_SLICE].copy())
        joint_vel_mj = torch.from_numpy(mj_data.qvel[MAIN_QVEL_DOF_SLICE].copy())
        joint_pos_il = joint_pos_mj[IL_TO_MJ_DOF]
        joint_vel_il = joint_vel_mj[IL_TO_MJ_DOF]
        joint_pos_rel = joint_pos_il - H2_DEFAULT_JOINT_POS

        gravity_w = torch.tensor([0.0, 0.0, -1.0])
        gravity_body = quat_apply_inverse(base_quat, gravity_w)

        return {
            "gravity_dir": gravity_body,
            "base_ang_vel": base_ang_vel,
            "joint_pos_rel": joint_pos_rel,
            "joint_vel": joint_vel_il,
            "base_quat": base_quat,
        }

    # Prime buffers with initial observation
    obs0 = read_obs()
    buf_gravity.reset(obs0["gravity_dir"])
    buf_ang_vel.reset(obs0["base_ang_vel"])
    buf_joint_pos.reset(obs0["joint_pos_rel"])
    buf_joint_vel.reset(obs0["joint_vel"])
    buf_actions.reset(last_action_il)

    def build_proprio():
        return torch.cat([
            buf_gravity.get(),
            buf_ang_vel.get(),
            buf_joint_pos.get(),
            buf_joint_vel.get(),
            buf_actions.get(),
        ])

    # ------------------------------------------------------------------
    # 6. Tokenizer observation builder
    # ------------------------------------------------------------------
    tokenizer_group_dims = saved_env_config.obs.group_obs_dims.tokenizer
    tokenizer_group_names = list(saved_env_config.obs.group_obs_names.tokenizer)
    tokenizer_encoder_names = list(actor.actor_module.encoder_sample_probs.keys())

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

    def flatten_tokenizer_terms(term_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        flat_terms = []
        for term_name in tokenizer_group_names:
            dims = tuple(tokenizer_group_dims[term_name])
            if term_name in term_dict:
                value = term_dict[term_name].reshape(dims)
            else:
                value = torch.zeros(dims, dtype=torch.float32)
            flat_terms.append(value.reshape(-1))
        return torch.cat(flat_terms, dim=0)

    def build_tokenizer_obs(current_frame: int):
        """Build a flat tokenizer observation matching saved model_config.yaml."""

        if processed_motion_ref is not None and "isaac_tokenizer" in processed_motion_ref:
            print("[DEBUG] using isaac_tokenizer direct injection", flush=True)
            tokenizer_flat = torch.from_numpy(
                np.asarray(processed_motion_ref["isaac_tokenizer"])
            ).float().reshape(-1)

            return {
                "tokenizer": tokenizer_flat,
            }

        if processed_motion_ref is not None:

            if processed_ref_num_frames > 1:
                _tok_idx = min(current_frame, processed_ref_num_frames - 1)

                term_dict = {
                    "encoder_index": processed_motion_ref["tok_encoder_index"][_tok_idx],
                    "command_multi_future_nonflat": processed_motion_ref["tok_command_multi_future_nonflat"][_tok_idx],
                    "motion_anchor_ori_b_mf_nonflat": processed_motion_ref["tok_motion_anchor_ori_b_mf_nonflat"][_tok_idx],
                }

            else:
                if current_frame != args.init_frame and args.init_frame != current_frame:
                    pass

                term_dict = {
                    "encoder_index": processed_motion_ref["tok_encoder_index"],
                    "command_multi_future_nonflat": processed_motion_ref["tok_command_multi_future_nonflat"],
                    "motion_anchor_ori_b_mf_nonflat": processed_motion_ref["tok_motion_anchor_ori_b_mf_nonflat"],
                }

            tokenizer_flat = flatten_tokenizer_terms({
                k: torch.from_numpy(v).float() for k, v in term_dict.items()
            })

            return {
                "tokenizer": tokenizer_flat,
                **{k: torch.from_numpy(v).float() for k, v in term_dict.items()},
            }

        # ------------------------------------------------------------
        # Original online tokenizer generation
        # ------------------------------------------------------------
        num_future = int(saved_algo_config.actor.backbone.num_future_frames)
        step = 5  # dt_future_ref_frames / CONTROL_DT = 0.1 / 0.02 = 5
        base_quat = torch.from_numpy(mj_data.qpos[3:7].copy()).float()

        cmd_frames = []
        cmd_z_frames = []
        anchor_frames = []

        for f in range(num_future):
            idx = min(current_frame + f * step, num_frames - 1)
            joint_pos_il = torch.from_numpy(motion_lib["joint_pos"][idx]).float()
            joint_vel_il = torch.from_numpy(motion_lib["joint_vel"][idx]).float()
            ref_root_pos = torch.from_numpy(motion_lib["root_pos"][idx]).float()
            ref_root_quat = torch.from_numpy(
                motion_quat_xyzw_to_wxyz(motion_lib["root_rot"][idx])
            ).float()

            cmd_frames.append(torch.cat([joint_pos_il, joint_vel_il]))
            cmd_z_frames.append(ref_root_pos[2:3])
            ref_ori_b = quat_mul(quat_inv(base_quat), ref_root_quat)
            ref_ori_b = ref_ori_b / torch.norm(ref_ori_b)
            anchor_frames.append(quat_to_6d(ref_ori_b))

        encoder_index = torch.zeros(len(tokenizer_encoder_names), dtype=torch.float32)
        if "g1" in tokenizer_encoder_names:
            encoder_index[tokenizer_encoder_names.index("g1")] = 1.0

        term_dict = {
            "encoder_index": encoder_index,
            "command_multi_future_nonflat": torch.stack(cmd_frames, dim=0),
            "command_z_multi_future_nonflat": torch.stack(cmd_z_frames, dim=0),
            "command_z": cmd_z_frames[0],
            "motion_anchor_ori_b": anchor_frames[0],
            "motion_anchor_ori_b_mf_nonflat": torch.stack(anchor_frames, dim=0),
        }
        tokenizer_flat = flatten_tokenizer_terms(term_dict)
        return {
            "tokenizer": tokenizer_flat,
            **term_dict,
        }

    # ------------------------------------------------------------------
    # 7. Logging helper
    # ------------------------------------------------------------------
    def log_step(step, total_steps, episode_time, mj_data, obs_dict, tok_obs,
                 action_il, target_il, fall_count, pelvis_z, gravity_body, current_frame):
        base_quat = mj_data.qpos[3:7]
        actor_obs_flat = obs_dict["actor_obs"].reshape(-1)
        tracking = compute_tracking_metrics(current_frame)
        target_err = (target_il - torch.from_numpy(tracking["sim_joint_pos_il"]).float()).numpy()
        ctrl = mj_data.ctrl.copy()
        ctrl_ranges = mj_model.actuator_ctrlrange.copy()
        ctrl_limited = mj_model.actuator_ctrllimited.astype(bool)
        saturated = []
        for i, limited in enumerate(ctrl_limited):
            if not limited:
                continue
            lo, hi = ctrl_ranges[i]
            if ctrl[i] <= lo + 1e-6 or ctrl[i] >= hi - 1e-6:
                saturated.append(actuator_names[i])

        abs_target_err = np.abs(target_err)
        worst_target_idx = np.argsort(abs_target_err)[-5:][::-1]
        print(f"\n=== Step {total_steps} (episode t={episode_time:.3f}s) ===")
        print(f"  pelvis_z={pelvis_z:.4f}  gravity_body=({gravity_body[0]:.4f}, {gravity_body[1]:.4f}, {gravity_body[2]:.4f})")
        print(f"  base_quat (wxyz)=({base_quat[0]:.4f}, {base_quat[1]:.4f}, {base_quat[2]:.4f}, {base_quat[3]:.4f})")
        print(f"  qpos[:7] (root)=({mj_data.qpos[0]:.4f}, {mj_data.qpos[1]:.4f}, {mj_data.qpos[2]:.4f}, "
              f"{mj_data.qpos[3]:.4f}, {mj_data.qpos[4]:.4f}, {mj_data.qpos[5]:.4f}, {mj_data.qpos[6]:.4f})")
        print(f"  qpos[7:] (dof) min={mj_data.qpos[MAIN_QPOS_DOF_SLICE].min():.4f} max={mj_data.qpos[MAIN_QPOS_DOF_SLICE].max():.4f} "
              f"mean={mj_data.qpos[MAIN_QPOS_DOF_SLICE].mean():.4f}")
        print(f"  qvel[:6] (root) lin=({mj_data.qvel[0]:.4f}, {mj_data.qvel[1]:.4f}, {mj_data.qvel[2]:.4f}) "
              f"ang=({mj_data.qvel[3]:.4f}, {mj_data.qvel[4]:.4f}, {mj_data.qvel[5]:.4f})")
        print(f"  qvel[6:] (dof) min={mj_data.qvel[MAIN_QVEL_DOF_SLICE].min():.4f} max={mj_data.qvel[MAIN_QVEL_DOF_SLICE].max():.4f} "
              f"mean={mj_data.qvel[MAIN_QVEL_DOF_SLICE].mean():.4f}")
        print(f"  action min/max/mean: {action_il.min():.4f}, {action_il.max():.4f}, {action_il.mean():.4f}")
        print(f"  target min/max/mean: {target_il.min():.4f}, {target_il.max():.4f}, {target_il.mean():.4f}")
        print(f"  ctrl min/max/mean: {mj_data.ctrl.min():.4f}, {mj_data.ctrl.max():.4f}, {mj_data.ctrl.mean():.4f}")
        print(f"  target-current err rms/max: {np.sqrt(np.mean(target_err ** 2)):.4f}, {np.max(np.abs(target_err)):.4f}")
        print(f"  action_il (first 6): {action_il[:6].tolist()}")
        print(f"  target_il (first 6): {target_il[:6].tolist()}")
        print(f"  ctrl (first 6): {mj_data.ctrl[:6].tolist()}")
        print(f"  actor_obs shape: {tuple(obs_dict['actor_obs'].shape)}")
        print(f"  proprio[:30] gravity_hist: {actor_obs_flat[:30].tolist()}")
        print(f"  proprio[30:60] ang_vel_hist: {actor_obs_flat[30:60].tolist()}")
        print(f"  proprio[60:70] joint_pos_hist first 10: {actor_obs_flat[60:70].tolist()}")
        if "command_multi_future_nonflat" in tok_obs:
            print(f"  tok_cmd_mf[0,:10]: {tok_obs['command_multi_future_nonflat'][0, :10].tolist()}")
        else:
            print("  tok_cmd_mf: [Direct injection mode - details unavailable]")
            
        if "motion_anchor_ori_b_mf_nonflat" in tok_obs:
            print(f"  tok_anchor_mf[0,:6]: {tok_obs['motion_anchor_ori_b_mf_nonflat'][0, :6].tolist()}")
        print("[TRACK]")
        print(
            f"  root_pos sim={tracking['sim_root_pos'].tolist()} ref={tracking['ref_root_pos'].tolist()} "
            f"err={tracking['root_pos_err'].tolist()}"
        )
        print(
            f"  root_z sim={tracking['sim_root_pos'][2]:.4f} ref={tracking['ref_root_pos'][2]:.4f} "
            f"err={tracking['root_pos_err'][2]:.4f}"
        )
        print(f"  root_ori_err_deg={tracking['root_ori_err_deg']:.4f}")
        print(
            f"  joint_pos_err rms={tracking['joint_pos_rms']:.4f} max={tracking['joint_pos_max']:.4f} "
            f"worst={tracking['worst_pos_joint']}"
        )
        print(
            f"  joint_vel_err rms={tracking['joint_vel_rms']:.4f} max={tracking['joint_vel_max']:.4f} "
            f"worst={tracking['worst_vel_joint']}"
        )
        print(f"[PD]")
        print(f"  saturated_actuators count={len(saturated)} names={saturated}")
        print(
            "  worst_target_tracking="
            + str([
                {
                    "joint": IL_JOINT_NAMES[int(i)],
                    "target": float(target_il[int(i)]),
                    "current": float(tracking['sim_joint_pos_il'][int(i)]),
                    "abs_err": float(abs_target_err[int(i)]),
                }
                for i in worst_target_idx
            ])
        )
        min_foot_z = tracking['min_foot_z']
        min_foot_z_str = f"{min_foot_z:.4f}" if min_foot_z is not None else "None"
        print(f"  contacts={mj_data.ncon} min_foot_z={min_foot_z_str} support_geoms={tracking['min_foot_geoms']}")
        print(f"  foot_contact_pairs={tracking['foot_contact_pairs']}")
        print(f"  all_contact_pairs={tracking['all_contact_pairs']}")
        if fall_count > 0:
            print(f"  falls={fall_count}")

        # --- Joint-level force diagnostics ---
        actuator_force = mj_data.actuator_force.copy()
        qfrc_actuator = mj_data.qfrc_actuator.copy()
        qfrc_passive = mj_data.qfrc_passive.copy()
        qfrc_constraint = mj_data.qfrc_constraint.copy()

        pd_torque_mj = mj_data.ctrl.copy()
        target_mj_np = np.empty(H2_NUM_DOF, dtype=np.float32)
        target_il_np = target_il.detach().cpu().numpy() if hasattr(target_il, 'device') else np.asarray(target_il)
        target_mj_np[IL_TO_MJ_DOF.numpy()] = target_il_np
        current_mj_np = mj_data.qpos[MAIN_QPOS_DOF_SLICE].copy()
        vel_mj_np = mj_data.qvel[MAIN_QVEL_DOF_SLICE].copy()
        print("[JOINT_DIAG]")
        print(f"  actuator_force min/max/mean: {actuator_force.min():.4f}, {actuator_force.max():.4f}, {actuator_force.mean():.4f}")
        print(f"  qfrc_actuator min/max/mean: {qfrc_actuator.min():.4f}, {qfrc_actuator.max():.4f}, {qfrc_actuator.mean():.4f}")
        print(f"  qfrc_passive min/max/mean: {qfrc_passive.min():.4f}, {qfrc_passive.max():.4f}, {qfrc_passive.mean():.4f}")
        print(f"  qfrc_constraint min/max/mean: {qfrc_constraint.min():.4f}, {qfrc_constraint.max():.4f}, {qfrc_constraint.mean():.4f}")
        dof_abs_err = np.abs(target_mj_np - current_mj_np)
        worst_dof_idx = np.argsort(dof_abs_err)[-5:][::-1]
        print("  worst5_target_current_ctrl:")
        for i in worst_dof_idx:
            print(f"    {MJ_JOINT_NAMES[i]:30s} target={target_mj_np[i]:7.4f}  "
                  f"cur={current_mj_np[i]:7.4f}  err={dof_abs_err[i]:7.4f}  "
                  f"vel={vel_mj_np[i]:7.4f}  pd_torque={pd_torque_mj[i]:8.3f}  "
                  f"act_force={actuator_force[i]:8.3f}  "
                  f"qfrc_act={qfrc_actuator[6+i]:8.3f}  "
                  f"qfrc_pass={qfrc_passive[6+i]:8.3f}  "
                  f"qfrc_con={qfrc_constraint[6+i]:8.3f}")
        # =========================================================
        print("\n[MJ_ORDER_TRACKING_CHECK] ALL JOINTS:")
        for i, jname in enumerate(MJ_JOINT_NAMES):
            err = target_mj_np[i] - current_mj_np[i]
            print(
                f"{i:02d} {jname:28s} "
                f"target={target_mj_np[i]: 8.4f} "
                f"current={current_mj_np[i]: 8.4f} "
                f"err={err: 8.4f} "
                f"vel={vel_mj_np[i]: 8.4f} "
                f"ctrl={pd_torque_mj[i]: 8.4f}"
            )
        # =========================================================
        # --- Contact force diagnostics ---
        if args.debug_contact_force:
            print("[CONTACT_DIAG]")
            for i in range(min(mj_data.ncon, 8)):
                c = mj_data.contact[i]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                contact_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(mj_model, mj_data, i, contact_force)
                print(f"  contact[{i}]: {mj_geom_names[g1]:20s} <-> {mj_geom_names[g2]:20s}  "
                      f"dist={c.dist:.6f}  "
                      f"pos=({c.pos[0]:.4f},{c.pos[1]:.4f},{c.pos[2]:.4f})  "
                      f"normal=({c.frame[0]:.4f},{c.frame[1]:.4f},{c.frame[2]:.4f})  "
                      f"force_xy=({contact_force[0]:.1f},{contact_force[1]:.1f})  "
                       f"force_z={contact_force[2]:.1f}  "
                       f"torque=({contact_force[3]:.2f},{contact_force[4]:.2f},{contact_force[5]:.2f})")

        # --- Foot contact summary ---
        print("[FOOT_CONTACT]")
        # IsaacLab foot body world positions (ankle_roll_link = foot in IsaacLab)
        for side_idx, side_name in enumerate(["left", "right"]):
            bid = il_foot_body_ids[side_idx]
            if bid is not None:
                body_world_pos = mj_data.xpos[bid].copy()
                # Also get the foot collision geom for this side
                collision_geom_name = f"{side_name}_foot_collision"
                cgids = [i for i, n in enumerate(mj_geom_names) if n == collision_geom_name]
                cg_contacts = 0
                cg_normal_force = 0.0
                cg_friction_force = 0.0
                if cgids:
                    cgid = cgids[0]
                    cg_world_pos = mj_data.geom_xpos[cgid].copy()
                    cg_obb_z, _, _ = compute_foot_obb_lowest_z(cgid)
                    cg_low_z = cg_obb_z
                    for ci in range(mj_data.ncon):
                        c = mj_data.contact[ci]
                        if int(c.geom1) == cgid or int(c.geom2) == cgid:
                            contact_force = np.zeros(6, dtype=np.float64)
                            mujoco.mj_contactForce(mj_model, mj_data, ci, contact_force)
                            cg_contacts += 1
                            cg_normal_force += abs(contact_force[2])
                            cg_friction_force += np.sqrt(contact_force[0]**2 + contact_force[1]**2)
                    cg_str = (f"collision_pos=({cg_world_pos[0]:.4f},{cg_world_pos[1]:.4f},{cg_world_pos[2]:.4f}) "
                              f"lowest_z={cg_low_z:.4f} ")
                else:
                    cg_str = "collision_geom=N/A "
                # Reference foot contact estimate
                ref_l, ref_r = None, None
                if ref_foot_contact_l is not None and current_frame < len(ref_foot_contact_l):
                    ref_l = bool(ref_foot_contact_l[current_frame])
                    ref_r = bool(ref_foot_contact_r[current_frame])
                ref_contact = ""
                if ref_l is not None:
                    ref_contact = f"ref_L={ref_l} ref_R={ref_r}"
                print(f"  {side_name:5s} IL_foot_body({side_name}_ankle_roll_link) "
                      f"world_pos=({body_world_pos[0]:.4f},{body_world_pos[1]:.4f},{body_world_pos[2]:.4f})  "
                      f"{cg_str}"
                      f"contacts={cg_contacts} F_norm={cg_normal_force:.1f} F_fric={cg_friction_force:.1f}  "
                      f"{ref_contact}")

        # --- Foot transform debug ---
        if args.debug_foot_transform:
            print("[FOOT_TRANSFORM]")
            for side_idx, side_name in enumerate(["left", "right"]):
                roll_body_id = il_foot_body_ids[side_idx]
                if roll_body_id is None:
                    continue
                # Find pitch body (child of roll)
                pitch_body_name = f"{side_name}_ankle_pitch_link"
                pitch_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, pitch_body_name)
                # Collision geom
                collision_geom_name = f"{side_name}_foot_collision"
                cgids = [i for i, n in enumerate(mj_geom_names) if n == collision_geom_name]
                # Joint angles for this leg chain
                roll_joint_name = f"{side_name}_ankle_roll_joint"
                pitch_joint_name = f"{side_name}_ankle_pitch_joint"
                roll_jid = next((i for i, n in enumerate(MJ_JOINT_NAMES) if n == roll_joint_name), None)
                pitch_jid = next((i for i, n in enumerate(MJ_JOINT_NAMES) if n == pitch_joint_name), None)
                roll_q = mj_data.qpos[7 + roll_jid].item() if roll_jid is not None else 0.0
                pitch_q = mj_data.qpos[7 + pitch_jid].item() if pitch_jid is not None else 0.0
                # World positions
                roll_pos = mj_data.xpos[roll_body_id].copy()
                pitch_pos = mj_data.xpos[pitch_body_id].copy() if pitch_body_id >= 0 else np.zeros(3)
                print(f"  {side_name:5s} ankle_roll pos=({roll_pos[0]:.4f},{roll_pos[1]:.4f},{roll_pos[2]:.4f})  "
                      f"quat=({mj_data.xquat[roll_body_id][0]:.4f},{mj_data.xquat[roll_body_id][1]:.4f},{mj_data.xquat[roll_body_id][2]:.4f},{mj_data.xquat[roll_body_id][3]:.4f})")
                if pitch_body_id >= 0:
                    print(f"         ankle_pitch pos=({pitch_pos[0]:.4f},{pitch_pos[1]:.4f},{pitch_pos[2]:.4f})  "
                          f"q_roll={roll_q:.4f} q_pitch={pitch_q:.4f}")
                if cgids:
                    cgid = cgids[0]
                    cg_pos = mj_data.geom_xpos[cgid].copy()
                    cg_xmat = mj_data.geom_xmat[cgid].copy().reshape(3, 3)
                    _obb_z_col, _, _ = compute_foot_obb_lowest_z(cgid)
                    print(f"         collision_geom pos=({cg_pos[0]:.4f},{cg_pos[1]:.4f},{cg_pos[2]:.4f})  "
                          f"OBB_lowest_z={_obb_z_col:.4f}  "
                          f"rot_mat=[[{cg_xmat[0,0]:.4f},{cg_xmat[0,1]:.4f},{cg_xmat[0,2]:.4f}],"
                          f"[{cg_xmat[1,0]:.4f},{cg_xmat[1,1]:.4f},{cg_xmat[1,2]:.4f}],"
                          f"[{cg_xmat[2,0]:.4f},{cg_xmat[2,1]:.4f},{cg_xmat[2,2]:.4f}]]")
                # Root-to-foot distance
                root_pos = mj_data.qpos[:3]
                leg_vec = roll_pos[:2] - root_pos[:2]
                print(f"         root_to_ankle_roll_xy_dist={np.linalg.norm(leg_vec):.4f}  "
                      f"ankle_roll_z_wrt_root={roll_pos[2] - root_pos[2]:.4f}")

        # --- Actuator limit diagnostics ---
        if args.debug_actuator_limits:
            print("[ACTUATOR_LIMITS]")
            for i in range(mj_model.nu):
                joint_id = mj_model.actuator_trnid[i, 0]
                joint_name = mj_joint_names[joint_id] if joint_id < len(mj_joint_names) else f"jnt_{joint_id}"
                frc_lo, frc_hi = mj_model.jnt_actfrcrange[joint_id]
                ctrl_lo, ctrl_hi = mj_model.actuator_ctrlrange[i]
                forcer_lo, forcer_hi = mj_model.actuator_forcerange[i]
                gear = mj_model.actuator_gear[i] if mj_model.actuator_gear is not None else 1.0
                dyn_type = mj_model.actuator_dyntype[i]; gain_type = mj_model.actuator_gaintype[i]; bias_type = mj_model.actuator_biastype[i]
                il_effort = IL_EFFORT_DICT.get(joint_name, None)
                print(f"  {actuator_names[i]:25s} joint={joint_name:25s}"
                      f" gear={gear[0]:.1f}  dyn={dyn_type} gain={gain_type} bias={bias_type}"
                      f" ctrlrange=[{ctrl_lo:8.2f},{ctrl_hi:8.2f}]"
                      f" forcerange=[{forcer_lo:8.2f},{forcer_hi:8.2f}]"
                      f" joint_frc=[{frc_lo:8.2f},{frc_hi:8.2f}]"
                      f" IL_effort={il_effort if il_effort is not None else 'N/A':>8}"
                      f"  ctrl={mj_data.ctrl[i]:8.3f}"
                      f" act_f={mj_data.actuator_force[i]:8.3f}"
                      f" qfrc={mj_data.qfrc_actuator[6 + i]:8.3f}")

    # ------------------------------------------------------------------
    # 8. Main control loop
    # ------------------------------------------------------------------
    print("\n[H2 Eval] Starting control loop.")
    if not args.no_viewer:
        print("Controls: SPACE=pause, R=reset, V=camera\n")

    current_frame = args.init_frame
    frame_accum = float(args.init_frame)
    action_il = torch.zeros(H2_NUM_DOF)
    episode_time = 0.0
    total_steps = 0
    fall_count = 0

    track_errs = []
    target_errs = []
    fall_penalty = 0.0
    # --- Foot trajectory buffer ---
    foot_trajectory = {
        "step": [], "time": [], "frame": [],
        "left_ankle_roll_pos": [], "right_ankle_roll_pos": [],
        "left_ankle_pitch_pos": [], "right_ankle_pitch_pos": [],
        "left_foot_collision_pos": [], "right_foot_collision_pos": [],
        "left_foot_obb_lowest_z": [], "right_foot_obb_lowest_z": [],
        "left_foot_contacts": [], "right_foot_contacts": [],
        "pelvis_pos": [],
    }

    viewer = None
    if not args.no_viewer:
        if mujoco_viewer is None:
            print("[H2 Eval] mujoco.viewer is unavailable; running headless.")
        else:
            viewer = mujoco_viewer.launch_passive(mj_model, mj_data)
            viewer.cam.lookat[:] = mj_data.qpos[:3]
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -25
            if args.debug_contact or args.debug_foot_transform:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                viewer.opt.frame = 1
            if args.debug_com:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
            if args.show_floor and ground_geom_ids:
                for geom_id in ground_geom_ids:
                    mj_model.geom_rgba[geom_id, 3] = max(mj_model.geom_rgba[geom_id, 3], 1.0)
    start_time = time.time()

    if args.score_out is not None and processed_ref_num_frames > 0:
        max_frames = processed_ref_num_frames
    elif num_frames > 0:
        max_frames = num_frames
    else:
        max_frames = 200

    print(f"[ROLLOUT] max_frames={max_frames}")

    try:
        while current_frame < max_frames and (viewer is None or viewer.is_running()):
            # --- Reset detection ---
            should_reset = False
            pelvis_z = mj_data.qpos[2]
            base_quat = torch.from_numpy(mj_data.qpos[3:7].copy())
            gravity_body = quat_apply_inverse(base_quat, torch.tensor([0.0, 0.0, -1.0]))
            grav_z = gravity_body[2].item()

            if pelvis_z < args.fall_height or grav_z > args.fall_tilt_cos:
                should_reset = True
                fall_count += 1
                fall_penalty += 100.0 
                print(f"[H2 Eval] Fall detected (z={pelvis_z:.3f}, gz={grav_z:.3f}). "
                      f"Resetting. Falls: {fall_count}")

            if args.max_episode > 0 and episode_time >= args.max_episode:
                should_reset = True
                print(f"[H2 Eval] Episode timeout ({episode_time:.1f}s). Resetting.")

            if should_reset:
                # GA 튜닝 중(score_out) 넘어지면 페널티만 남기고 즉시 종료
                if args.score_out:
                    print("[ROLLOUT_END] Fall detected during GA evaluation. Breaking early.")
                    break
                
                # 일반 뷰어 모드일 때는 기존처럼 리셋
                current_frame = args.init_frame
                frame_accum = float(args.init_frame)
                apply_rsi(current_frame)
                frozen_root_qpos = mj_data.qpos[:7].copy()
                frozen_root_qvel = mj_data.qvel[:6].copy()
                obs0 = read_obs()
                buf_gravity.reset(obs0["gravity_dir"])
                buf_ang_vel.reset(obs0["base_ang_vel"])
                buf_joint_pos.reset(obs0["joint_pos_rel"])
                buf_joint_vel.reset(obs0["joint_vel"])
                buf_actions.reset(last_action_il)
                action_il = torch.zeros(H2_NUM_DOF)
                last_action_il = torch.zeros(H2_NUM_DOF)
                episode_time = 0.0
                if viewer:
                    viewer.sync()
                continue

            # --- Advance motion (accumulator-based, supports fractional speed) ---
            _max_frame = processed_ref_num_frames - 1 if args.replay_reference and processed_ref_num_frames > 1 else num_frames - 1
            # --- Advance motion ---
            if args.speed > 0:
                frame_accum += args.speed
                current_frame = int(frame_accum)

            if current_frame >= max_frames:
                print(f"[ROLLOUT_END] reached max_frames={max_frames}")
                break
            if args.replay_reference and total_steps % max(args.log_every, 1) == 0:
                print(f"[REPLAY_FRAME] step={total_steps} frame_accum={frame_accum:.2f} current_frame={current_frame} "
                      f"max_frame={_max_frame} speed={args.speed}")

            # --------------------------------------------------
            # Ghost reference robot update
            # --------------------------------------------------

            ref_idx = min(current_frame, num_frames - 1)

            root_pos = motion_lib["root_pos"][ref_idx]
            root_quat_xyzw = motion_lib["root_rot"][ref_idx]
            root_quat = motion_quat_xyzw_to_wxyz(root_quat_xyzw)

            joint_pos_il = motion_lib["joint_pos"][ref_idx]

            joint_pos_mj = np.empty(H2_NUM_DOF, dtype=np.float32)
            joint_pos_mj[:] = joint_pos_il

            ref_root_bid = mujoco.mj_name2id(
                mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                "ref_pelvis"
            )

            ref_root_jid = mj_model.body_jntadr[ref_root_bid]
            ref_root_jnt_type = mj_model.jnt_type[ref_root_jid]

            if ref_root_jnt_type != mujoco.mjtJoint.mjJNT_FREE:
                raise RuntimeError(
                    f"ref_pelvis root joint is not freejoint. "
                    f"jid={ref_root_jid}, type={ref_root_jnt_type}, "
                    f"name={mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, ref_root_jid)}"
                )

            ref_qadr = mj_model.jnt_qposadr[ref_root_jid]

            # floating base
            mj_data.qpos[ref_qadr:ref_qadr+3] = root_pos
            mj_data.qpos[ref_qadr+3:ref_qadr+7] = root_quat

            # joints
            for i, jname in enumerate(MJ_JOINT_NAMES):

                jid = mujoco.mj_name2id(
                    mj_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"ref_{jname}"
                )

                qadr = mj_model.jnt_qposadr[jid]

                mj_data.qpos[qadr] = joint_pos_mj[i]
            mujoco.mj_forward(mj_model, mj_data)

            # --- Build observations ---
            obs_current = read_obs()

            actor_obs = build_proprio()
            tokenizer_dict = build_tokenizer_obs(current_frame)

            if processed_motion_ref is not None and "isaac_actor_obs" in processed_motion_ref:
                print("[DEBUG] using isaac_actor_obs direct injection", flush=True)
                actor_obs = torch.from_numpy(
                    np.asarray(processed_motion_ref["isaac_actor_obs"])
                ).float().reshape(-1)

            obs_dict = {
                "actor_obs": actor_obs.view(1, 1, -1).to(device),
                "tokenizer": tokenizer_dict["tokenizer"].view(1, 1, -1).to(device),
            }

            # --- Policy inference ---
            with torch.no_grad():
                action_il = actor(obs_dict).squeeze(0).squeeze(0).cpu()
            if total_steps == 0:
                torch.save({
                    "action_il": action_il.detach().cpu(),
                    "actor_obs": obs_dict["actor_obs"].detach().cpu(),
                    "tokenizer": obs_dict["tokenizer"].detach().cpu(),
                }, "dump_step0/h2_policy_step0_mujoco.pt")
            # --- FSQ quantization ---
            action_il = fsq_quantize(action_il, fsq_levels)

            # --- Compute joint targets ---
            if args.replay_reference:
                if processed_motion_ref is not None:
                    ref_idx = min(current_frame, processed_ref_num_frames - 1)

                    if "motion_ref_joint_pos" in processed_motion_ref:
                        target_joint_pos = processed_motion_ref["motion_ref_joint_pos"][ref_idx]
                    elif "qpos_dof" in processed_motion_ref:
                        target_joint_pos = processed_motion_ref["qpos_dof"][ref_idx]
                    elif "motion_ref_joint_pos_future" in processed_motion_ref:
                        target_joint_pos = processed_motion_ref["motion_ref_joint_pos_future"][ref_idx, 0]
                    else:
                        raise RuntimeError(
                            "processed motion reference has no motion_ref_joint_pos, qpos_dof, or motion_ref_joint_pos_future"
                        )

                    target_il = torch.from_numpy(
                        np.asarray(target_joint_pos, dtype=np.float32)
                    )
                    target_order = "il"

                else:
                    # pkl motion replay: this pkl dof is already MuJoCo joint order
                    ref_idx = min(current_frame, num_frames - 1)
                    target_mj_from_pkl = torch.from_numpy(
                        np.asarray(motion_lib["joint_pos"][ref_idx], dtype=np.float32)
                    )
                    target_il = target_mj_from_pkl
                    target_order = "mj"

            else:
                action_scale_il = H2_ACTION_SCALE.clone() * args.action_scale_mult

                action_scale_il[LEG_IDXS] *= args.action_scale_legs
                action_scale_il[FEET_IDXS] *= args.action_scale_feet
                action_scale_il[WAIST_IDXS] *= args.action_scale_waist

                target_il = H2_DEFAULT_JOINT_POS + action_il * action_scale_il
                target_order = "il"

            # --- GA score metrics ---
            if args.score_out is not None:
                if processed_motion_ref is not None and processed_ref_num_frames > 1:
                    ref_idx = min(current_frame, processed_ref_num_frames - 1)

                    if "motion_ref_joint_pos" in processed_motion_ref:
                        ref_joint_il = torch.from_numpy(
                            np.asarray(processed_motion_ref["motion_ref_joint_pos"][ref_idx], dtype=np.float32)
                        )
                        q_il_now = torch.from_numpy(
                            mj_data.qpos[MAIN_QPOS_DOF_SLICE].copy()
                        ).float()[MJ_TO_IL_DOF]

                        if target_order == "mj":
                            target_il_for_score = target_il.cpu()[MJ_TO_IL_DOF]
                        else:
                            target_il_for_score = target_il.cpu()

                        track_errs.append(torch.mean((q_il_now - ref_joint_il) ** 2).item())
                        target_errs.append(torch.mean((target_il_for_score - ref_joint_il) ** 2).item())

                else:
                    # pkl motion is MuJoCo joint order
                    ref_idx = min(current_frame, num_frames - 1)
                    ref_joint_mj = torch.from_numpy(
                        np.asarray(motion_lib["joint_pos"][ref_idx], dtype=np.float32)
                    )

                    q_mj_now = torch.from_numpy(
                        mj_data.qpos[MAIN_QPOS_DOF_SLICE].copy()
                    ).float()

                    if target_order == "mj":
                        target_mj_for_score = target_il.cpu()
                    else:
                        target_mj_for_score = torch.empty(H2_NUM_DOF)
                        target_mj_for_score[IL_TO_MJ_DOF] = target_il.cpu()

                    track_errs.append(torch.mean((q_mj_now - ref_joint_mj) ** 2).item())
                    target_errs.append(torch.mean((target_mj_for_score - ref_joint_mj) ** 2).item())

            # --- PD control in MuJoCo ---
            if target_order == "mj":
                target_mj = target_il.clone()
            else:
                target_mj = torch.empty(H2_NUM_DOF, device=target_il.device)
                target_mj[IL_TO_MJ_DOF] = target_il
            current_mj = torch.from_numpy(mj_data.qpos[MAIN_QPOS_DOF_SLICE].copy())
            vel_mj = torch.from_numpy(mj_data.qvel[MAIN_QVEL_DOF_SLICE].copy())
            mj_data.ctrl[:] = (
                kp * (target_mj - current_mj) - kd * vel_mj
            ).numpy()

            # --- Step physics ---
            mjdt = mj_model.opt.timestep
            physics_steps = max(1, int(round(CONTROL_DT / mjdt)))
            for _ in range(physics_steps):
                mujoco.mj_step(mj_model, mj_data)

            # --- Freeze root if requested ---
            if args.freeze_root:
                mj_data.qpos[:7] = frozen_root_qpos
                mj_data.qvel[:6] = frozen_root_qvel
                mujoco.mj_forward(mj_model, mj_data)

            # --- Update shadow reference robot ---
            update_shadow_ref(current_frame)
            mujoco.mj_forward(mj_model, mj_data)

            # --- Record foot trajectory ---
            if args.dump_foot_trajectory:
                _l_roll = il_foot_body_ids[0]
                _r_roll = il_foot_body_ids[1]
                _l_pitch = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_pitch_link")
                _r_pitch = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_pitch_link")
                _l_cg = next((i for i, n in enumerate(mj_geom_names) if n == "left_foot_collision"), None)
                _r_cg = next((i for i, n in enumerate(mj_geom_names) if n == "right_foot_collision"), None)
                foot_trajectory["step"].append(total_steps)
                foot_trajectory["time"].append(episode_time)
                foot_trajectory["frame"].append(current_frame)
                foot_trajectory["pelvis_pos"].append(mj_data.qpos[:3].copy())
                if _l_roll is not None:
                    foot_trajectory["left_ankle_roll_pos"].append(mj_data.xpos[_l_roll].copy())
                if _r_roll is not None:
                    foot_trajectory["right_ankle_roll_pos"].append(mj_data.xpos[_r_roll].copy())
                if _l_pitch >= 0:
                    foot_trajectory["left_ankle_pitch_pos"].append(mj_data.xpos[_l_pitch].copy())
                if _r_pitch >= 0:
                    foot_trajectory["right_ankle_pitch_pos"].append(mj_data.xpos[_r_pitch].copy())
                if _l_cg is not None:
                    foot_trajectory["left_foot_collision_pos"].append(mj_data.geom_xpos[_l_cg].copy())
                    foot_trajectory["left_foot_obb_lowest_z"].append(compute_foot_obb_lowest_z(_l_cg)[0])
                    _l_contact_count = sum(1 for ci in range(mj_data.ncon) if int(mj_data.contact[ci].geom1) == _l_cg or int(mj_data.contact[ci].geom2) == _l_cg)
                    foot_trajectory["left_foot_contacts"].append(_l_contact_count)
                if _r_cg is not None:
                    foot_trajectory["right_foot_collision_pos"].append(mj_data.geom_xpos[_r_cg].copy())
                    foot_trajectory["right_foot_obb_lowest_z"].append(compute_foot_obb_lowest_z(_r_cg)[0])
                    _r_contact_count = sum(1 for ci in range(mj_data.ncon) if int(mj_data.contact[ci].geom1) == _r_cg or int(mj_data.contact[ci].geom2) == _r_cg)
                    foot_trajectory["right_foot_contacts"].append(_r_contact_count)

            # --- Update buffers ---
            buf_gravity.append(obs_current["gravity_dir"])
            buf_ang_vel.append(obs_current["base_ang_vel"])
            buf_joint_pos.append(obs_current["joint_pos_rel"])
            buf_joint_vel.append(obs_current["joint_vel"])
            buf_actions.append(action_il)
            last_action_il = action_il.clone()

            episode_time += CONTROL_DT
            total_steps += 1

            # --- Logging ---
            if total_steps % args.log_every == 0:
                log_step(
                    0, total_steps, episode_time, mj_data, obs_dict, tokenizer_dict,
                    action_il, target_il, fall_count, pelvis_z, gravity_body, current_frame,
                )

            if viewer:
                viewer.sync()

    except KeyboardInterrupt:
        print("\n[H2 Eval] Interrupted by user.")
    finally:
        if viewer:
            viewer.close()

    # --- Dump foot trajectory ---
    if args.dump_foot_trajectory and len(foot_trajectory["step"]) > 0:
        dump_out = {}
        for k, v in foot_trajectory.items():
            if v and isinstance(v[0], np.ndarray):
                dump_out[k] = np.stack(v, axis=0)
            elif v:
                dump_out[k] = np.asarray(v)
        os.makedirs(os.path.dirname(args.dump_foot_trajectory) or ".", exist_ok=True)
        np.savez(args.dump_foot_trajectory, **dump_out)
        print(f"[FOOT_TRAJ] Saved foot trajectory ({len(dump_out['step'])} steps) to {args.dump_foot_trajectory}")
    # ----------------------------------------------------------
    # GA score
    # ----------------------------------------------------------
    if args.score_out is not None:

        mean_track_err = (
            float(np.mean(track_errs))
            if len(track_errs) > 0 else 1e6
        )

        mean_target_err = (
            float(np.mean(target_errs))
            if len(target_errs) > 0 else 1e6
        )

        # ---------------------------------------------
        # survival ratio
        # ---------------------------------------------
        survival_ratio = float(total_steps) / float(max(1, max_frames))
        survival_ratio = np.clip(survival_ratio, 0.0, 1.0)

        # ---------------------------------------------
        # early fall penalty
        #
        # 빨리 죽을수록 penalty 큼
        # 오래 버티면 penalty 감소
        # ---------------------------------------------
        if fall_count > 0:
            fall_penalty = 50.0 * (1.0 - survival_ratio)
        else:
            fall_penalty = 0.0

        # ---------------------------------------------
        # final score (MINIMIZE)
        # ---------------------------------------------
        score = (
            1.0 * mean_track_err
            + 0.2 * mean_target_err
            + fall_penalty
        )

        os.makedirs(os.path.dirname(args.score_out) or ".", exist_ok=True)

        np.savez(
            args.score_out,
            score=score,
            mean_track_err=mean_track_err,
            mean_target_err=mean_target_err,
            fall_penalty=fall_penalty,
            fall_count=fall_count,
            survival_ratio=survival_ratio,
            total_steps=total_steps,
            max_frames=max_frames,
        )

        print(
            f"[GA_SCORE] "
            f"score={score:.6f} "
            f"track={mean_track_err:.6f} "
            f"target={mean_target_err:.6f} "
            f"survival={survival_ratio:.3f} "
            f"fall_penalty={fall_penalty:.3f} "
            f"falls={fall_count} "
            f"steps={total_steps}/{max_frames}"
        )


if __name__ == "__main__":
    main()
