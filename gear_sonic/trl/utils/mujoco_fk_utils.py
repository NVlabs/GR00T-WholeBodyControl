"""
MuJoCo FK utilities: qpos <-> global transforms.
qpos format: [root_trans(3), root_quat_wxyz(4), dof_angles(N)]
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

from groot.rl.trl.utils.rotation_conversion import quaternion_to_matrix, matrix_to_quaternion


def dof_to_rotation_matrices(dof_angles: torch.Tensor, dof_axis: torch.Tensor) -> torch.Tensor:
    """Convert DOF angles [..., N] to rotation matrices [..., N, 3, 3]."""
    half_angles = dof_angles / 2
    cos_half, sin_half = torch.cos(half_angles), torch.sin(half_angles)

    axis = dof_axis.to(dof_angles.device)
    for _ in range(dof_angles.dim() - 1):
        axis = axis.unsqueeze(0)
    axis = axis.expand(*dof_angles.shape, 3)

    quaternion = torch.cat([cos_half.unsqueeze(-1), sin_half.unsqueeze(-1) * axis], dim=-1)
    return quaternion_to_matrix(quaternion)


def rotation_matrices_to_dof(
    rotation_matrices: torch.Tensor, dof_axis: torch.Tensor
) -> torch.Tensor:
    """Extract DOF angles [..., N] from rotation matrices [..., N, 3, 3]."""
    R = rotation_matrices
    x_angle = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    y_angle = torch.atan2(R[..., 0, 2], R[..., 0, 0])
    z_angle = torch.atan2(R[..., 1, 0], R[..., 1, 1])
    xyz_angles = torch.stack([x_angle, y_angle, z_angle], dim=-1)

    axis = dof_axis.to(rotation_matrices.device)
    for _ in range(xyz_angles.dim() - 2):
        axis = axis.unsqueeze(0)
    axis = axis.expand(*xyz_angles.shape[:-1], 3)

    return (xyz_angles * axis).sum(dim=-1)


def qpos_to_root_and_dof(qpos: torch.Tensor, num_dof: int, root_quat_wxyz: bool = True):
    """Parse qpos into (root_trans, root_quat_wxyz, dof_angles)."""
    root_trans = qpos[..., :3]
    root_quat = qpos[..., 3:7]
    dof_angles = qpos[..., 7 : 7 + num_dof]
    if not root_quat_wxyz:
        root_quat = root_quat[..., [3, 0, 1, 2]]
    return root_trans, root_quat, dof_angles


def root_and_dof_to_qpos(root_trans, root_quat, dof_angles, root_quat_wxyz: bool = True):
    """Assemble qpos from (root_trans, root_quat_wxyz, dof_angles)."""
    if not root_quat_wxyz:
        root_quat = root_quat[..., [1, 2, 3, 0]]
    return torch.cat([root_trans, root_quat, dof_angles], dim=-1)


class MuJoCoFKHelper(nn.Module):
    """FK helper wrapping Humanoid_Batch. Supports extended bodies (head, toes)."""

    # G1 29-DOF joint order mappings (IsaacLab <-> MuJoCo)
    # MuJoCo groups by limb, IsaacLab interleaves left/right
    ISAACLAB_TO_MUJOCO_DOF = [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ]
    MUJOCO_TO_ISAACLAB_DOF = [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ]
    ROOT_DOF_OFFSET = 7

    def __init__(self, humanoid_batch):
        super().__init__()
        self.register_buffer("dof_axis", humanoid_batch.dof_axis.float())
        self.register_buffer("_offsets", humanoid_batch._offsets.float())
        self.register_buffer("_local_rotation_mat", humanoid_batch._local_rotation_mat.float())

        self._parents = humanoid_batch._parents
        self.num_dof = humanoid_batch.num_dof
        self.num_bodies = humanoid_batch.num_bodies
        self.num_bodies_augment = humanoid_batch.num_bodies_augment
        self.body_names = humanoid_batch.body_names
        self.body_names_augment = humanoid_batch.body_names_augment

    def qpos_to_global_transforms(
        self,
        qpos: torch.Tensor,
        from_isaaclab_order: bool,
        root_quat_wxyz: bool = True,
        include_extended: bool = False,
    ):
        """Convert qpos [B, T, D] to global positions [B, T, J, 3] and rotations [B, T, J, 3, 3]."""
        if from_isaaclab_order:
            qpos = torch.cat(
                [
                    qpos[..., : self.ROOT_DOF_OFFSET],
                    qpos[..., self.ROOT_DOF_OFFSET :][..., self.ISAACLAB_TO_MUJOCO_DOF],
                ],
                dim=-1,
            )

        squeeze_time = qpos.dim() == 2
        if squeeze_time:
            qpos = qpos.unsqueeze(1)

        B, T = qpos.shape[:2]
        root_trans, root_quat, dof_angles = qpos_to_root_and_dof(qpos, self.num_dof, root_quat_wxyz)

        root_rot_mat = quaternion_to_matrix(root_quat).unsqueeze(2)
        joint_rot_mat = dof_to_rotation_matrices(dof_angles, self.dof_axis)

        global_pos, global_rot = self._forward_kinematics(joint_rot_mat, root_rot_mat, root_trans)

        if not include_extended:
            global_pos = global_pos[..., : self.num_bodies, :]
            global_rot = global_rot[..., : self.num_bodies, :, :]

        if squeeze_time:
            global_pos, global_rot = global_pos.squeeze(1), global_rot.squeeze(1)
        return global_pos, global_rot

    def _forward_kinematics(self, joint_rotations, root_rotations, root_positions):
        """FK for all bodies including extended."""
        device, dtype = root_rotations.device, root_rotations.dtype
        B, T = joint_rotations.shape[:2]
        J = self._offsets.shape[1]

        offsets = self._offsets[:, None].expand(B, T, J, 3).to(device, dtype)
        local_rot_mat = self._local_rotation_mat.to(device, dtype)
        eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3).expand(B, T, 1, 3, 3)

        positions, rotations = [], []
        for i in range(J):
            if self._parents[i] == -1:
                positions.append(root_positions)
                rotations.append(root_rotations)
            else:
                parent_rot, parent_pos = rotations[self._parents[i]], positions[self._parents[i]]
                jpos = (
                    torch.matmul(parent_rot[:, :, 0], offsets[:, :, i, :, None]).squeeze(-1)
                    + parent_pos
                )

                joint_rot = joint_rotations[:, :, i - 1 : i] if i < self.num_bodies else eye
                rot_mat = torch.matmul(
                    parent_rot, torch.matmul(local_rot_mat[:, i : i + 1], joint_rot)
                )

                positions.append(jpos)
                rotations.append(rot_mat)

        return torch.stack(positions, dim=2), torch.cat(rotations, dim=2)

    def global_to_local_rotations(self, global_rotations):
        """Convert global rotations to local rotations."""
        local_rotations = torch.zeros_like(global_rotations)
        for i in range(global_rotations.shape[-3]):
            if self._parents[i] == -1:
                local_rotations[..., i, :, :] = global_rotations[..., i, :, :]
            else:
                parent_rot = global_rotations[..., self._parents[i], :, :]
                local_rotations[..., i, :, :] = torch.matmul(
                    parent_rot.transpose(-1, -2), global_rotations[..., i, :, :]
                )
        return local_rotations

    def global_transforms_to_qpos(
        self,
        global_rotations,
        global_positions,
        to_isaaclab_order: bool,
        root_quat_wxyz: bool = True,
    ):
        """Convert global transforms back to qpos."""
        squeeze_time = global_rotations.dim() == 4
        if squeeze_time:
            global_rotations = global_rotations.unsqueeze(1)
            global_positions = global_positions.unsqueeze(1)

        root_trans = global_positions[..., 0, :]
        local_rotations = self.global_to_local_rotations(global_rotations)

        root_quat = matrix_to_quaternion(local_rotations[..., 0, :, :])
        local_rot_mat = self._local_rotation_mat.to(local_rotations.device)
        joint_rot_mat = torch.matmul(
            local_rot_mat[:, 1 : self.num_bodies].transpose(-1, -2),
            local_rotations[..., 1 : self.num_bodies, :, :],
        )
        dof_angles = rotation_matrices_to_dof(joint_rot_mat, self.dof_axis)

        qpos = root_and_dof_to_qpos(root_trans, root_quat, dof_angles, root_quat_wxyz)

        if to_isaaclab_order:
            qpos = torch.cat(
                [
                    qpos[..., : self.ROOT_DOF_OFFSET],
                    qpos[..., self.ROOT_DOF_OFFSET :][..., self.MUJOCO_TO_ISAACLAB_DOF],
                ],
                dim=-1,
            )
        return qpos.squeeze(1) if squeeze_time else qpos

    @property
    def device(self):
        return self.dof_axis.device


def load_qpos_from_csv(csv_path: str) -> torch.Tensor:
    """Load qpos [T, D] from CSV."""
    import pandas as pd

    return torch.from_numpy(pd.read_csv(csv_path).values.astype(np.float32))


def save_qpos_to_csv(qpos: torch.Tensor, csv_path: str):
    """Save qpos to CSV."""
    import pandas as pd

    data = qpos[0].cpu().numpy() if qpos.dim() == 3 else qpos.cpu().numpy()
    pd.DataFrame(data).to_csv(csv_path, index=False)


if __name__ == "__main__":
    from pathlib import Path
    from omegaconf import OmegaConf
    from groot.rl.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

    groot_root = Path(__file__).parent.parent.parent.parent
    motion_yaml = (
        groot_root
        / "rl"
        / "config"
        / "manager_env"
        / "commands"
        / "terms"
        / "motion_g1_extended_toe.yaml"
    )

    cfg = OmegaConf.load(motion_yaml).motion.motion_lib_cfg
    fk_helper = MuJoCoFKHelper(Humanoid_Batch(cfg, device=torch.device("cpu")))
    print(
        f"Loaded: {fk_helper.num_bodies} + {fk_helper.num_bodies_augment - fk_helper.num_bodies} extended bodies"
    )

    # Load from CSV or generate random
    csv_path = groot_root / ".." / "data" / "example_csv_g1_navigation.csv"
    if csv_path.exists():
        qpos = load_qpos_from_csv(str(csv_path)).unsqueeze(0)
    else:
        T = 30
        qpos = torch.cat(
            [
                torch.tensor([[0.0, 0.0, 1.0]]).expand(T, 3),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(T, 4),
                torch.randn(T, fk_helper.num_dof) * 0.3,
            ],
            dim=-1,
        ).unsqueeze(0)

    # FK round-trip test
    global_pos, global_rot = fk_helper.qpos_to_global_transforms(qpos, False)
    qpos_out = fk_helper.global_transforms_to_qpos(global_rot, global_pos, False)
    print(f"Round-trip error: {(qpos[..., 7:] - qpos_out[..., 7:]).abs().max():.2e}")
