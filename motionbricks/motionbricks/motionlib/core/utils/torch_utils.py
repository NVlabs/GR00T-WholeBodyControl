from typing import List, Optional, Union

import torch
import torch.nn.functional as F


def normalize_vec(x: torch.Tensor, dim: int = -1, eps: float = 1e-9):
    return x / x.norm(p=2, dim=dim).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def transform_mat(R, t):
    """Creates a batch of transformation matrices.

    Args:
        - R: Bx3x3 array of a batch of rotation matrices
        - t: Bx3x1 array of a batch of translation vectors
    Returns:
        - T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1.0)], dim=2)


def compute_idx_levels(parents):
    idx_levs = [[]]
    lev_dicts = {0: -1}
    for i in range(1, parents.shape[0]):
        assert int(parents[i]) in lev_dicts
        lev = lev_dicts[int(parents[i])] + 1
        if lev + 1 > len(idx_levs):
            idx_levs.append([])
        idx_levs[lev].append(int(i))
        lev_dicts[int(i)] = lev
    idx_levs = [torch.tensor(x).long() for x in idx_levs]
    return idx_levs


def batch_rigid_transform(rot_mats, joints, parents, root_idx):
    """Perform batch rigid transformation on a skeletal structure.

    Args:
        rot_mats: Local rotation matrices for each joint: (B, J, 3, 3)
        joints: Initial joint positions: (B, J, 3)
        parents: Tensor indicating the parent of each joint: (J,)
        root_idx (int): index of the root

    Returns:
        Transformed joint positions after applying forward kinematics.
    """

    # Compute the hierarchical levels of joints based on their parent relationships
    idx_levs = compute_idx_levels(parents)

    # Apply forward kinematics to transform the joints
    return forward_kinematics(rot_mats, joints, parents, idx_levs, root_idx)


@torch.jit.script
def forward_kinematics(
    rot_mats,
    joints,
    parents: torch.Tensor,
    idx_levs: List[torch.Tensor],
    root_idx: int,
):
    """Perform forward kinematics to compute posed joints and global rotation matrices.

    Args:
        rot_mats: Local rotation matrices for each joint: (B, J, 3, 3)
        joints: Initial joint positions: (B, J, 3)
        parents: Tensor indicating the parent of each joint: (J,)
        idx_levs: List of tensors containing indices for each level in the kinematic tree
        root_idx (int): index of the root
    Returns:
        Posed joints: (B, J, 3)
        Global rotation matrices: (B, J, 3, 3)
    """

    # Add an extra dimension to joints
    joints = torch.unsqueeze(joints, dim=-1)

    # Compute relative joint positions
    rel_joints = joints.clone()

    mask_no_root = torch.ones(joints.shape[1], dtype=torch.bool)
    mask_no_root[root_idx] = False
    rel_joints[:, mask_no_root] -= joints[:, parents[mask_no_root]].clone()

    # Compute initial transformation matrices
    # (B, J + 1, 4, 4)
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)
    ).reshape(-1, joints.shape[1], 4, 4)

    # Initialize the root transformation matrices
    transforms = torch.zeros_like(transforms_mat)
    transforms[:, root_idx] = transforms_mat[:, root_idx]

    # Compute global transformations level by level
    for indices in idx_levs:
        curr_res = torch.matmul(
            transforms[:, parents[indices]], transforms_mat[:, indices]
        )
        transforms[:, indices] = curr_res

    # Extract posed joint positions from the transformation matrices
    posed_joints = transforms[:, :, :3, 3]

    # Extract global rotation matrices from the transformation matrices
    global_rot_mat = transforms[:, :, :3, :3]

    return posed_joints, global_rot_mat


def length_to_mask(
    length: Union[torch.Tensor, List],
    max_len: Optional[int] = None,
    device=None,
) -> torch.Tensor:
    if isinstance(length, list):
        if device is None:
            device = "cpu"
        length = torch.tensor(length, device=device)

    if device is not None:
        assert device == length.device
    device = length.device

    if max_len is None:
        max_len = max(length)

    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask
