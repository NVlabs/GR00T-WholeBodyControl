"""Lightweight SMPL body models (SmplLite, SmplxLiteJ24) for efficient FK and skinning."""

import pickle
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from smplx.utils import Struct, to_np, to_tensor

from .rotation_conversions import axis_angle_to_matrix

from .smplx_lite import batch_rigid_transform_v2


class SmplLite(nn.Module):
    def __init__(
        self,
        model_path="inputs/checkpoints/body_models/smpl",
        gender="neutral",
        num_betas=10,
    ):
        super().__init__()

        # Load the model
        model_path = Path(model_path)
        if model_path.is_dir():
            smpl_path = Path(model_path) / f"SMPL_{gender.upper()}.pkl"
        else:
            smpl_path = model_path
        assert smpl_path.exists()
        with open(smpl_path, "rb") as smpl_file:
            data_struct = Struct(**pickle.load(smpl_file, encoding="latin1"))
        self.faces = data_struct.f  # (F, 3)

        self.register_smpl_buffers(data_struct, num_betas)
        self.register_fast_skeleton_computing_buffers()

    def register_smpl_buffers(self, data_struct, num_betas):
        # shapedirs, (V, 3, N_betas), V=10475 for SMPLX
        shapedirs = to_tensor(to_np(data_struct.shapedirs[:, :, :num_betas])).float()
        self.register_buffer("shapedirs", shapedirs, False)

        # v_template, (V, 3)
        v_template = to_tensor(to_np(data_struct.v_template)).float()
        self.register_buffer("v_template", v_template, False)

        # J_regressor, (J, V), J=55 for SMPLX
        J_regressor = to_tensor(to_np(data_struct.J_regressor)).float()
        self.register_buffer("J_regressor", J_regressor, False)

        # posedirs, (54*9, V, 3), note that the first global_orient is not included
        posedirs = to_tensor(to_np(data_struct.posedirs)).float()  # (V, 3, 54*9)
        posedirs = rearrange(posedirs, "v c n -> n v c")
        self.register_buffer("posedirs", posedirs, False)

        # lbs_weights, (V, J), J=55
        lbs_weights = to_tensor(to_np(data_struct.weights)).float()
        self.register_buffer("lbs_weights", lbs_weights, False)

        # parents, (J), long
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents, False)

    def register_fast_skeleton_computing_buffers(self):
        # For fast computing of skeleton under beta
        J_template = self.J_regressor @ self.v_template  # (J, 3)
        J_shapedirs = torch.einsum("jv, vcd -> jcd", self.J_regressor, self.shapedirs)  # (J, 3, 10)
        self.register_buffer("J_template", J_template, False)
        self.register_buffer("J_shapedirs", J_shapedirs, False)

    def get_skeleton(self, betas):
        return self.J_template + einsum(betas, self.J_shapedirs, "... k, j c k -> ... j c")

    def forward(
        self,
        body_pose,
        betas,
        global_orient,
        transl,
    ):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
            transl: (B, L, 3)
        Returns:
            vertices: (B, L, V, 3)
        """
        # 1. Convert [global_orient, body_pose] to rot_mats
        full_pose = torch.cat([global_orient, body_pose], dim=-1)
        rot_mats = axis_angle_to_matrix(
            full_pose.reshape(*full_pose.shape[:-1], full_pose.shape[-1] // 3, 3)
        )

        # 2. Forward Kinematics
        J = self.get_skeleton(betas)  # (*, 55, 3)
        A = batch_rigid_transform_v2(rot_mats, J, self.parents)[1]

        # 3. Canonical v_posed = v_template + shaped_offsets + pose_offsets
        pose_feature = rot_mats[..., 1:, :, :] - rot_mats.new([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pose_feature = pose_feature.view(*pose_feature.shape[:-3], -1)  # (*, 55*3*3)
        v_posed = (
            self.v_template
            + einsum(betas, self.shapedirs, "... k, v c k -> ... v c")
            + einsum(pose_feature, self.posedirs, "... k, k v c -> ... v c")
        )
        del pose_feature, rot_mats, full_pose

        # 4. Skinning
        T = einsum(self.lbs_weights, A, "v j, ... j c d -> ... v c d")
        verts = einsum(T[..., :3, :3], v_posed, "... v c d, ... v d -> ... v c") + T[..., :3, 3]

        # 5. Translation
        verts = verts + transl[..., None, :]
        return verts


class SmplxLiteJ24(SmplLite):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Compute mapping
        smpl2j24 = self.J_regressor  # (24, 6890)

        jids, smplx_vids = torch.where(smpl2j24 != 0)
        interestd = torch.zeros([len(smplx_vids), 24])
        for idx, (jid, smplx_vid) in enumerate(zip(jids, smplx_vids)):
            interestd[idx, jid] = smpl2j24[jid, smplx_vid]
        self.register_buffer("interestd", interestd, False)  # (236, 24)

        # Update to vertices of interest
        self.v_template = self.v_template[smplx_vids].clone()  # (V', 3)
        self.shapedirs = self.shapedirs[smplx_vids].clone()  # (V', 3, K)
        self.posedirs = self.posedirs[:, smplx_vids].clone()  # (K, V', 3)
        self.lbs_weights = self.lbs_weights[smplx_vids].clone()  # (V', J)

    def forward(self, body_pose, betas, global_orient, transl):
        """Returns: joints (*, J, 3). (B, L) or  (B,) are both supported."""
        # Use super class's forward to get verts
        verts = super().forward(body_pose, betas, global_orient, transl)  # (*, 236, 3)
        joints = einsum(self.interestd, verts, "v j, ... v c -> ... j c")
        return joints
