"""Minimal Linear Blend Skinning (LBS) for sparse sensor-point vertices on SMPLH."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from smplx.utils import Struct, to_np, to_tensor

from hmr4d.utils.smplx_utils import forward_kinematics_motion
from motiondiff.models.mdm.rotation_conversions import axis_angle_to_matrix


class MinimalLBS(nn.Module):
    def __init__(self, sp_ids, bm_dir="models/smplh", num_betas=16, model_type="smplh", **kwargs):
        super().__init__()
        self.num_betas = num_betas
        self.sensor_point_vid = torch.tensor(sp_ids)

        # load struct data on predefined sensor-point
        self.load_struct_on_sp(f"{bm_dir}/male/model.npz", prefix="male")
        self.load_struct_on_sp(f"{bm_dir}/female/model.npz", prefix="female")

    def load_struct_on_sp(self, bm_path, prefix="m"):
        """
        Load 4 weights from body-model-struct.
        Keep the sensor points only. Use prefix to label different bm.
        """
        num_betas = self.num_betas
        sp_vid = self.sensor_point_vid
        # load data
        data_struct = Struct(**np.load(bm_path, encoding="latin1"))

        # v-template
        v_template = to_tensor(to_np(data_struct.v_template))  # (V, 3)
        v_template_sp = v_template[sp_vid]  # (N, 3)
        self.register_buffer(f"{prefix}_v_template_sp", v_template_sp, False)

        # shapedirs
        shapedirs = to_tensor(to_np(data_struct.shapedirs[:, :, :num_betas]))  # (V, 3, NB)
        shapedirs_sp = shapedirs[sp_vid]
        self.register_buffer(f"{prefix}_shapedirs_sp", shapedirs_sp, False)

        # posedirs
        posedirs = to_tensor(to_np(data_struct.posedirs))  # (V, 3, 51*9)
        posedirs_sp = posedirs[sp_vid]
        posedirs_sp = posedirs_sp.reshape(len(sp_vid) * 3, -1).T  # (51*9, N*3)
        self.register_buffer(f"{prefix}_posedirs_sp", posedirs_sp, False)

        # lbs_weights
        lbs_weights = to_tensor(to_np(data_struct.weights))  # (V, J+1)
        lbs_weights_sp = lbs_weights[sp_vid]
        self.register_buffer(f"{prefix}_lbs_weights_sp", lbs_weights_sp, False)

    def forward(
        self,
        root_orient=None,
        pose_body=None,
        trans=None,
        betas=None,
        A=None,
        recompute_A=False,
        genders=None,
        joints_zero=None,
    ):
        """
        Args:
            root_orient, Optional: (B, T, 3)
            pose_body: (B, T, J*3)
            trans: (B, T, 3)
            betas: (B, T, 16)
            A, Optional: (B, T, J+1, 4, 4)
            recompute_A: if True, root_orient should be given, otherwise use A
            genders, List: ['male', 'female', ...]
            joints_zero: (B, J+1, 3), required when recompute_A is True
        Returns:
            sensor_verts: (B, T, N, 3)
        """
        B, T = pose_body.shape[:2]

        v_template = torch.stack(
            [getattr(self, f"{g}_v_template_sp") for g in genders]
        )  # (B, N, 3)
        shapedirs = torch.stack(
            [getattr(self, f"{g}_shapedirs_sp") for g in genders]
        )  # (B, N, 3, NB)
        posedirs = torch.stack(
            [getattr(self, f"{g}_posedirs_sp") for g in genders]
        )  # (B, 51*9, N*3)
        lbs_weights = torch.stack(
            [getattr(self, f"{g}_lbs_weights_sp") for g in genders]
        )  # (B, N, J+1)

        # ===== LBS, handle T ===== #
        # 2. Add shape contribution
        if betas.shape[1] == 1:
            betas = betas.expand(-1, T, -1)
        blend_shape = torch.einsum("btl,bmkl->btmk", [betas, shapedirs])
        v_shaped = v_template[:, None] + blend_shape

        # 3. Add pose blend shapes
        ident = torch.eye(3).to(pose_body)
        aa = pose_body.reshape(B, T, -1, 3)
        R = axis_angle_to_matrix(aa)
        pose_feature = (R - ident).view(B, T, -1)
        dim_pf = pose_feature.shape[-1]
        # (B, T, P) @ (B, P, N*3) -> (B, T, N, 3)
        pose_offsets = torch.matmul(pose_feature, posedirs[:, :dim_pf]).view(B, T, -1, 3)
        v_posed = pose_offsets + v_shaped

        # 4. Compute A
        if recompute_A:
            _, _, A = forward_kinematics_motion(root_orient, pose_body, trans, joints_zero)

        # 5. Skinning
        W = lbs_weights
        # (B, 1, N, J+1)) @ (B, T, J+1, 16)
        num_joints = A.shape[-3]  # 22
        Ts = torch.matmul(W[:, None, :, :num_joints], A.view(B, T, num_joints, 16))
        Ts = Ts.view(B, T, -1, 4, 4)  # (B, T, N, 4, 4)
        v_posed_homo = F.pad(v_posed, (0, 1), value=1)  # (B, T, N, 4)
        v_homo = torch.matmul(Ts, torch.unsqueeze(v_posed_homo, dim=-1))

        # 6. translate
        sensor_verts = v_homo[:, :, :, :3, 0] + trans[:, :, None]

        return sensor_verts
