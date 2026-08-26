import math
import os
from typing import Optional

import einops
import torch

from motionbricks.motionlib.core.skeletons import (
    G1Skeleton,
    G1Skeleton32,
    G1Skeleton34,
    SkeletonBase,
)
from motionbricks.motionlib.core.utils.rotations import exp_map_to_matrix, quaternion_to_matrix
from motionbricks.motionlib.core.utils.torch_utils import batch_rigid_transform


def _local_rot_offset_from_old_neutral_to_new_T(
    n_bones: int, name_to_idx: dict
) -> torch.Tensor:
    # local offsets to make the weird skeleton a custom t-pose

    const_exp_map = torch.zeros(n_bones, 3)

    # set (pelvis, x) to -pi/2
    const_exp_map[name_to_idx["Hips"], 0] = -math.pi / 2
    # set (right_shoulder, y) to pi
    const_exp_map[name_to_idx["RightShoulder"], 1] = math.pi
    # set (right up leg, y) to pi/2
    const_exp_map[name_to_idx["RightUpLeg"], 1] = math.pi / 2
    # set (left up leg, y) to pi/2
    const_exp_map[name_to_idx["LeftUpLeg"], 1] = math.pi / 2
    # set (right foot, z) to -pi/2
    const_exp_map[name_to_idx["RightFoot"], 2] = -math.pi / 2
    # set (left foot, z) to -pi/2
    const_exp_map[name_to_idx["LeftFoot"], 2] = -math.pi / 2

    const_rot_mats = exp_map_to_matrix(const_exp_map)  # (N, 3, 3)

    return const_rot_mats


def get_global_offset(
    t_pose: str,
    skeleton: SkeletonBase,
    neutral_joints: Optional[torch.Tensor] = None,
    return_neutral_joints=False,
    base_path: str = "./",  # this is helpful when using the package as a submodule in other projects
):
    """Loads the t-pose that we want to convert to and returns the joint positions along with global
    joint rotations."""
    if neutral_joints is None:
        neutral_joints = skeleton.neutral_joints

    device = neutral_joints.device
    dtype = neutral_joints.dtype

    if t_pose == "capture":
        # identity: no changes
        local_offset = torch.eye(3).repeat(skeleton.nbjoints, 1, 1)
    elif t_pose == "custom":
        n_bones = skeleton.nbjoints
        name_to_idx = skeleton.bone_index
        # rotation offsets: (N, 3, 3)
        local_offset = _local_rot_offset_from_old_neutral_to_new_T(n_bones, name_to_idx)
    elif t_pose == "standard":
        t_pose_path = None
        native_skel = None
        if isinstance(skeleton, G1Skeleton32):
            t_pose_path = os.path.join(
                base_path,
                "assets/skeletons/g1skel32/standard_t_pose_g1skel32_joint_quat.p",
            )
            native_skel = G1Skeleton32()
        elif isinstance(skeleton, G1Skeleton34):
            t_pose_path = os.path.join(
                base_path,
                "assets/skeletons/g1skel34/standard_t_pose_g1skel34_joint_quat.p",
            )
            native_skel = G1Skeleton34()
        else:
            raise NotImplementedError(
                f"This skeleton is not supported for t-pose conversion: {skeleton}"
            )
        joints_orients_with_hands = torch.load(t_pose_path)

        skel_slice = skeleton.get_skel_slice(native_skel)
        joints_orients = joints_orients_with_hands[skel_slice]
        local_offset = quaternion_to_matrix(joints_orients)
    else:
        raise NotImplementedError(f"This t-pose is not recognized: {t_pose}")

    # run FK to compute new neutral joint positions, and global rot offsets for next step of transforming the motion rot mats
    new_neutral_joints, global_rot_offsets = batch_rigid_transform(
        local_offset[None].to(device=device, dtype=dtype),
        neutral_joints[None],
        skeleton.joint_parents,
        skeleton.root_idx,
    )
    new_neutral_joints = new_neutral_joints[0]  # (N, 3)
    global_rot_offsets = global_rot_offsets[0]  # (N, 3, 3)

    if return_neutral_joints:
        return global_rot_offsets, new_neutral_joints
    return global_rot_offsets


def change_t_pose_global_mats(
    global_mats: torch.Tensor,
    t_pose_to: str,
    skeleton: SkeletonBase,
    t_pose_from: Optional[str] = None,
):
    if t_pose_from is None:
        t_pose_from = skeleton.t_pose

    # no changes
    if t_pose_from == t_pose_to:
        return global_mats

    assert t_pose_from in ["capture", "custom", "standard"]
    assert t_pose_to in ["capture", "custom", "standard"]

    dtype = global_mats.dtype
    device = global_mats.device

    global_offset_from = get_global_offset(t_pose_from, skeleton)
    global_offset_to = get_global_offset(t_pose_to, skeleton)

    new_global_mats = torch.einsum(
        "... N m n, N n o, N p o -> ... N m p",
        global_mats,
        global_offset_from.to(device=device, dtype=dtype),
        global_offset_to.to(device=device, dtype=dtype),
    )
    return new_global_mats


def change_t_pose_local_mats(
    local_mats: torch.Tensor,
    t_pose_to: str,
    skeleton: SkeletonBase,
    return_global_rots=False,
    t_pose_from: Optional[str] = None,
):
    if t_pose_from is None:
        t_pose_from = skeleton.t_pose

    dtype = local_mats.dtype
    device = local_mats.device

    orig_shape = local_mats.shape
    local_mats = local_mats.reshape(-1, skeleton.nbjoints, 3, 3)

    nbpose = local_mats.shape[0]
    neutral_joints = skeleton.neutral_joints
    batched_neutral_joints = einops.repeat(neutral_joints, "j k -> b j k", b=nbpose).to(
        dtype=dtype, device=device
    )

    _, global_mats = batch_rigid_transform(
        local_mats,
        batched_neutral_joints,
        skeleton.joint_parents,
        skeleton.root_idx,
    )  # (T, N, 3, 3)

    new_global_mats = change_t_pose_global_mats(
        global_mats,
        t_pose_to,
        skeleton,
        t_pose_from=t_pose_from,
    )
    new_local_mats = global_mats_to_local_mats(new_global_mats, skeleton)

    if return_global_rots:
        return new_local_mats.reshape(orig_shape), new_global_mats.reshape(orig_shape)
    return new_local_mats.reshape(orig_shape)


def global_mats_to_local_mats(
    global_rot_mats: torch.Tensor, skeleton: SkeletonBase
):
    # obtain back the local rotations from the global rotations
    parent_rot_mats = global_rot_mats[..., skeleton.joint_parents, :, :]
    parent_rot_mats[..., skeleton.root_idx, :, :] = torch.eye(3)  # the root joint
    local_rot_mats = torch.einsum(
        "... N n m, ... N n o -> ... N m o",
        parent_rot_mats,  # taken as the inverse/transpose in einsum
        global_rot_mats,
    )
    return local_rot_mats
