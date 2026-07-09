# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOMA -> SMPL conversion + SONIC Protocol-v3 publishing (live-webcam path).

GEM-X estimates SOMA (77-joint) body params. SONIC's deployed policy has an
``smpl`` encoder mode (Protocol v3) whose encoder observations are:
  - smpl_joints (24x3, root-local)
  - smpl_anchor_orientation (derived by the deploy from the streamed body_quat)
  - motion_joint_positions_wrists (6 G1 wrist joints)

This module converts a per-frame SOMA decode into the v3 stream fields, matching
SONIC's own convention in
``gear_sonic/scripts/pico_manager_thread_server.py:process_smpl_joints`` exactly:
the root quaternion goes aa -> quat -> smpl_root_ytoz_up (+90 deg about X) ->
remove_smpl_base_rot; smpl_joints are the SMPL joints with that root orientation
removed (root-local), in Z-up. Optional temporal smoothing steadies the live
signal.

Note: this file imports ``gear_sonic`` (native when run from inside the SONIC
repo). The SOMA body-model layer is passed in by the caller, so GEM-X is not
imported here directly.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

# Canonical SMPL 24 joint names (SMPL body kinematic tree order).
SMPL_24_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

# Aliases mapping canonical SMPL joints -> SOMA rig joint names (Maya-style).
# SOMA rig convention: "Foot" = ankle, "ToeBase" = foot/ball; "Arm" = shoulder,
# "ForeArm" = elbow, "Hand" = wrist, "Shoulder" = collar/clavicle.
_ALIASES = {
    "pelvis": ["hips", "pelvis", "root"],
    "left_hip": ["leftleg", "left_hip", "l_hip", "left_upleg"],
    "right_hip": ["rightleg", "right_hip", "r_hip", "right_upleg"],
    "spine1": ["spine1", "spine_1", "spine"],
    "left_knee": ["leftshin", "left_knee", "l_knee"],
    "right_knee": ["rightshin", "right_knee", "r_knee"],
    "spine2": ["spine2", "spine_2"],
    "left_ankle": ["leftfoot", "left_ankle", "l_ankle"],
    "right_ankle": ["rightfoot", "right_ankle", "r_ankle"],
    "spine3": ["chest", "spine3", "spine_3"],
    "left_foot": ["lefttoebase", "left_foot", "l_foot", "left_toe"],
    "right_foot": ["righttoebase", "right_foot", "r_foot", "right_toe"],
    "neck": ["neck1", "neck"],
    "left_collar": ["leftshoulder", "left_collar", "l_collar", "left_clavicle"],
    "right_collar": ["rightshoulder", "right_collar", "r_collar", "right_clavicle"],
    "head": ["head"],
    "left_shoulder": ["leftarm", "left_shoulder", "l_shoulder", "left_upperarm"],
    "right_shoulder": ["rightarm", "right_shoulder", "r_shoulder", "right_upperarm"],
    "left_elbow": ["leftforearm", "left_elbow", "l_elbow", "left_lowerarm"],
    "right_elbow": ["rightforearm", "right_elbow", "r_elbow"],
    "left_wrist": ["lefthand", "left_wrist", "l_wrist"],
    "right_wrist": ["righthand", "right_wrist", "r_wrist"],
    "left_hand": ["lefthandmiddle1", "left_hand", "l_hand", "left_middle1"],
    "right_hand": ["righthandmiddle1", "right_hand", "r_hand", "right_middle1"],
}


def _norm(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def build_soma_to_smpl_index(soma_joint_names: list[str]) -> list[int]:
    """Map each of the 24 SMPL joints to a SOMA joint index, by name alias."""
    norm_names = [_norm(n) for n in soma_joint_names]
    name_to_idx = {n: i for i, n in enumerate(norm_names)}
    idx, unmatched = [], []
    for smpl_name in SMPL_24_NAMES:
        found = None
        for alias in _ALIASES[smpl_name]:
            if _norm(alias) in name_to_idx:
                found = name_to_idx[_norm(alias)]
                break
        if found is None:
            unmatched.append(smpl_name)
            found = 0
        idx.append(found)
    if unmatched:
        raise ValueError(
            f"Could not map SMPL joints {unmatched} to SOMA joints. "
            f"Available SOMA joint names: {soma_joint_names}"
        )
    return idx


# Y-up -> Z-up rotation (+90 deg about X): same as smpl_root_ytoz_up on points.
_YUP_TO_ZUP = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
)


class SomaToSmpl:
    """Convert per-frame SOMA decode -> SONIC v3 SMPL fields."""

    def __init__(self, soma_layer, device="cuda", y_to_z_up=True, smooth=0.0, sonic_root=None):
        self.soma = soma_layer
        self.device = device
        self.y_to_z_up = y_to_z_up
        # Temporal smoothing weight on history (0 = off; 0.6-0.85 = progressively smoother).
        self.smooth = float(smooth)
        self._ema_joints = None
        self._ema_quat = None

        names = self._resolve_joint_names(soma_layer)
        if len(names) == 78 and _norm(names[0]) == "root":
            names = names[1:]  # align to the 77-joint forward() output
        self.soma_joint_names = names
        self.smpl_idx = build_soma_to_smpl_index(names)
        self._R_up = _YUP_TO_ZUP.to(device)

        # SONIC's EXACT SMPL convention helpers (native import inside the repo).
        import sys
        if sonic_root and sonic_root not in sys.path:
            sys.path.insert(0, sonic_root)
        from gear_sonic.trl.utils.torch_transform import (
            angle_axis_to_quaternion, quat_apply, quat_inv,
        )
        from gear_sonic.isaac_utils.rotations import (
            remove_smpl_base_rot, smpl_root_ytoz_up,
        )
        self._aa2quat = angle_axis_to_quaternion
        self._quat_apply = quat_apply
        self._quat_inv = quat_inv
        self._remove_base_rot = remove_smpl_base_rot
        self._ytoz = smpl_root_ytoz_up

    @staticmethod
    def _resolve_joint_names(soma_layer):
        inner = getattr(soma_layer, "soma", soma_layer)
        rig = getattr(inner, "rig_data", None)
        if rig is not None and hasattr(rig, "files") and "joint_names" in rig.files:
            return [str(x) for x in rig["joint_names"]]
        if isinstance(rig, dict) and "joint_names" in rig:
            return [str(x) for x in rig["joint_names"]]
        names = getattr(inner, "joint_names", None) or getattr(soma_layer, "joint_names", None)
        if names is not None:
            return [str(x) for x in names]
        raise AttributeError(
            "Could not find SOMA joint_names (checked .soma.rig_data['joint_names'])."
        )

    @torch.no_grad()
    def convert(self, soma_params: dict) -> dict:
        """soma_params: per-frame tensors (body_pose, global_orient,
        identity_coeffs, scale_params[, transl]). Returns v3 stream fields."""
        def _t(x):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            return x.unsqueeze(0) if x.dim() == 1 else x

        body_pose = _t(soma_params["body_pose"])
        global_orient = _t(soma_params["global_orient"])
        identity = _t(soma_params["identity_coeffs"])
        scale = _t(soma_params["scale_params"])
        transl = _t(soma_params.get("transl", torch.zeros(3, device=self.device)))

        out = self.soma(
            body_pose=body_pose, global_orient=global_orient, transl=transl,
            identity_coeffs=identity, scale_params=scale,
        )
        joints77 = out["joints"][0]                       # (77,3) GEM y-up, global applied
        joints24 = joints77[self.smpl_idx]                # (24,3)

        # Mirror SONIC's process_smpl_joints convention exactly.
        g_quat = self._aa2quat(global_orient)             # (1,4) wxyz, y-up
        g_quat_z = self._ytoz(g_quat)                     # (1,4) z-up
        g_quat_nobase = self._remove_base_rot(g_quat_z, w_last=False)  # (1,4)

        joints0 = joints24 - joints24[0:1]                # root at origin (y-up)
        joints_z = joints0 @ self._R_up.T                 # (24,3) z-up
        inv = self._quat_inv(g_quat_nobase).repeat(joints_z.shape[0], 1)  # (24,4)
        smpl_joints_local = self._quat_apply(inv, joints_z)               # (24,3)

        smpl_joints = smpl_joints_local.unsqueeze(0).cpu().numpy().astype(np.float32)
        body_quat = g_quat_nobase.reshape(1, 4).cpu().numpy().astype(np.float32)

        # Temporal smoothing (reduces live jitter -> stepping/wobble).
        if self.smooth > 0.0:
            w = self.smooth
            if self._ema_joints is None:
                self._ema_joints = smpl_joints.copy()
                self._ema_quat = body_quat.copy()
            else:
                self._ema_joints = w * self._ema_joints + (1.0 - w) * smpl_joints
                q_prev, q_new = self._ema_quat, body_quat.copy()
                if float((q_prev * q_new).sum()) < 0.0:   # sign-align before lerp
                    q_new = -q_new
                q = w * q_prev + (1.0 - w) * q_new
                self._ema_quat = q / (np.linalg.norm(q) + 1e-8)
            smpl_joints = self._ema_joints.astype(np.float32)
            body_quat = self._ema_quat.astype(np.float32)

        smpl_pose = np.zeros((1, 21, 3), dtype=np.float32)
        wrists = np.zeros((1, 6), dtype=np.float32)
        return {
            "smpl_joints": smpl_joints,
            "body_quat": body_quat,
            "smpl_pose": smpl_pose,
            "wrists": wrists,
        }


HEADER_SIZE = 1280  # SONIC ZMQ header size (see gear_sonic zmq_planner_sender.py)


def _fallback_pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 3) -> bytes:
    """Byte-identical fallback for SONIC's pack_pose_message.

    Layout: [topic_bytes][1280-byte JSON header][concatenated little-endian binary fields].
    """
    dtype_map = {
        np.dtype(np.float32): "f32", np.dtype(np.float64): "f64",
        np.dtype(np.int32): "i32", np.dtype(np.int64): "i64", np.dtype(bool): "bool",
    }
    fields, binary = [], []
    for key, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue
        dtype_str = dtype_map.get(value.dtype, "f32")
        if dtype_str == "f32" and value.dtype != np.float32:
            value = value.astype(np.float32)
        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})
        binary.append(value.tobytes())
    header = {"v": version, "endian": "le", "count": 1, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return topic.encode("utf-8") + header_json.ljust(HEADER_SIZE, b"\x00") + b"".join(binary)


def _get_packer(sonic_root: str | None = None):
    """Return SONIC's pack_pose_message if importable, else the local fallback.

    Native import works when run from inside the SONIC repo; pass ``sonic_root``
    only if running from outside.
    """
    if sonic_root and sonic_root not in sys.path:
        sys.path.insert(0, sonic_root)
    try:
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
        print("[bridge] Using SONIC's pack_pose_message (exact wire format).")
        return pack_pose_message
    except Exception as exc:
        print(f"[bridge] gear_sonic not importable ({exc}); using built-in fallback packer.")
        return _fallback_pack_pose_message


class SonicV3Publisher:
    """ZMQ PUB publisher for SONIC Protocol v3 (SMPL) using SONIC's exact packer."""

    def __init__(self, port=5556, topic="pose", sonic_root=None):
        import zmq

        self.pack = _get_packer(sonic_root)
        self.topic = topic
        self.frame_index = 0
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{port}")

    def publish(self, fields: dict):
        joint_pos = np.zeros((1, 29), dtype=np.float32)
        joint_pos[:, 23:29] = fields["wrists"]            # only wrists meaningful in v3
        pose_data = {
            "smpl_joints": fields["smpl_joints"].reshape(1, 24, 3),
            "smpl_pose": fields["smpl_pose"].reshape(1, 21, 3),
            "joint_pos": joint_pos,
            "joint_vel": np.zeros((1, 29), dtype=np.float32),
            "body_quat": fields["body_quat"].reshape(1, 4),  # wxyz, SONIC convention
            "frame_index": np.array([self.frame_index], dtype=np.int64),
        }
        self.sock.send(self.pack(pose_data, topic=self.topic, version=3))
        self.frame_index += 1

    def close(self):
        self.sock.close(0)
        self.ctx.term()
