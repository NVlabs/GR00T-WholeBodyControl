from typing import Optional

import torch

from motionbricks.motionlib.core.utils.rotations import quat_apply, quat_mul


def apply_base_rot(
    joints_pos: Optional[torch.Tensor] = None, joints_rot: Optional[torch.Tensor] = None
):
    """Flips output joint positions and/or rotations that is y-up to be z-up.

    Args:
        joints_pos(Optional[torch.Tensor]): [B, T, J, 3] joint positions
        joints_rot(Optional[torch.Tensor]): [B, T, J, 4] joint quaternions
    """
    base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    if joints_pos is not None:
        # rotate positions
        joints_pos = quat_apply(
            base_rot.to(joints_pos).expand(joints_pos.shape[:-1] + (4,)), joints_pos
        )
    if joints_rot is not None:
        # for rotations, apply base_rot to just the root
        root_rot_quat = joints_rot[:, :, 0:1]
        root_rot_quat = quat_mul(
            base_rot[:, None, None].to(joints_rot).expand_as(root_rot_quat),
            root_rot_quat,
        )
        joints_rot = torch.cat([root_rot_quat, joints_rot[:, :, 1:]], dim=2)

    return joints_pos, joints_rot
