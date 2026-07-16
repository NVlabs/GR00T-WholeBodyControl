from typing import Dict, List, Optional

import einops
import torch

from motionbricks.motionlib.core.motion_reps.tools.changing_t_pose import (
    change_t_pose_global_mats,
    change_t_pose_local_mats,
    global_mats_to_local_mats,
)
from motionbricks.motionlib.core.skeletons import SkeletonBase
from motionbricks.motionlib.core.utils.rotations import (
    cont6d_to_matrix,
    diff_angles,
    matrix_to_quaternion,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_unit,
    quaternion_to_cont6d,
    quaternion_to_matrix,
)
from motionbricks.motionlib.core.utils.torch_utils import batch_rigid_transform

from .feature_info import tensor_needed
from .feet import foot_detect_from_pos_and_vel
from .heading import calc_heading


def compute_vel_angle(
    root_rot_angles: torch.Tensor,
    fps: float,
    lengths: Optional[torch.Tensor] = None,
):
    """Compute the local root rotation velocity: dtheta/dt.

    Args:
        root_rot_angles (torch.Tensor): [..., T] rotation angle (in radian)
        fps (float): frame per seconds
        lengths (Optional[torch.Tensor]): [...] size of each input batched. If not provided, root_rot_angles should not be batched

    Returns:
        local_root_rot_vel (torch.Tensor): [..., T] local root rotation velocity (in radian/s)
    """
    device = root_rot_angles.device
    # If the lengths is not provided, we do not assume full length
    # the input should be a single sequence
    if lengths is None:
        # make sure it is a unique sequence input
        assert len(root_rot_angles.shape) == 1
        lengths = torch.tensor([len(root_rot_angles)], device=device)

    root_rot_angles, ps = einops.pack([root_rot_angles], "* nbframes")
    lengths, _ = einops.pack([lengths], "*")

    # useful for indexing
    range_len = torch.arange(len(lengths))

    local_root_rot_vel = diff_angles(root_rot_angles, fps)
    pad_rot_vel_angles = torch.zeros_like(root_rot_angles[:, 0])
    local_root_rot_vel, _ = einops.pack(
        [local_root_rot_vel, pad_rot_vel_angles],
        "batch *",
    )
    # repeat the last rotation angle
    # with special care for different lengths with batches
    local_root_rot_vel[(range_len, lengths - 1)] = local_root_rot_vel[
        (range_len, lengths - 2)
    ]

    [local_root_rot_vel] = einops.unpack(local_root_rot_vel, ps, "* nbframes")
    return local_root_rot_vel


def compute_vel_xyz(
    positions: torch.Tensor,
    fps: float,
    lengths: Optional[torch.Tensor] = None,
):
    """Compute the velocities from positions: dx/dt. Works with batches. The last velocity is duplicated to keep the same size.

    Args:
        positions (torch.Tensor): [..., T, J, 3] xyz positions of a human skeleton
        fps (float): frame per seconds
        lengths (Optional[torch.Tensor]): [...] size of each input batched. If not provided, positions should not be batched

    Returns:
        velocity (torch.Tensor): [..., T, J, 3] velocities computed from the positions
    """
    device = positions.device

    # If the lengths is not provided, we do not assume full length
    # the input should be a single sequence
    if lengths is None:
        # make sure it is a unique sequence input
        assert len(positions.shape) == 3
        lengths = torch.tensor([len(positions)], device=device)

    positions, ps = einops.pack([positions], "* nbframes nbjoints xyz")
    lengths, _ = einops.pack([lengths], "*")

    # useful for indexing
    range_len = torch.arange(len(lengths))

    # compute velocities with fps
    velocity = fps * (positions[:, 1:] - positions[:, :-1])
    # pading the velocity vector
    vel_pad = torch.zeros_like(velocity[:, 0])
    velocity, _ = einops.pack([velocity, vel_pad], "batch * nbjoints dim")

    # repeat the last velocities
    # with special care for different lengths with batches
    velocity[(range_len, lengths - 1)] = velocity[(range_len, lengths - 2)]
    [velocity] = einops.unpack(velocity, ps, "* nbframes nbjoints xyz")
    return velocity


def compute_heading_info(
    heading_quat_raw: torch.Tensor,
    nbjoints: int,
    root_quat: Optional[torch.Tensor],
    global_joint_quat: Optional[torch.Tensor],
    **kwargs,  # throw away unecessary args
):
    """Compute heading info. From the raw output of calc_heading, compute a canonicalized heading
    quaternion (first frame is 0 rotation), save the previous initial direction, and remove the
    heading from the global root rotation if provided.

    Args:
        heading_quat_raw: (torch.Tensor): [B, T, 4]  heading direction in quaternion, direct output of calc_heading
        nbjoints (int): number of joints of the skeleton
        root_quat (Optional[torch.Tensor]): [B, T, 4] global rotation of the root
        global_joint_quat (torch.Tensor): [B, T, J, 4] global joint quaternions of the input motion
        **kwargs (Dict): unused arguments for easy function call

    Returns:
        info (Dict): dictionnary which encapsulate all the output: heading_quat, init_heading_quat_inv, etc
    """

    nbframes = heading_quat_raw.shape[1]
    init_heading_quat_raw = heading_quat_raw[:, 0]

    root_quat_wo_heading = None
    root_quat_wo_first_heading = None
    if root_quat is not None:
        # root rotation without heading
        root_quat_wo_heading = quat_mul(
            quat_conjugate(heading_quat_raw), root_quat
        )  # [X, T, 4]

    # root heading rotation on all joints
    heading_quat = einops.repeat(
        heading_quat_raw,
        "batch nbframes quat -> batch nbframes nbjoints quat",
        nbjoints=nbjoints,
    )  # [X, T, J, 4]

    global_joint_quat_wo_heading = None
    if global_joint_quat is not None:
        # remove heading direction to all global rotations
        global_joint_quat_wo_heading = quat_mul(
            quat_conjugate(heading_quat), global_joint_quat
        )  # [X, T, J, 4]

    # first frame root heading rotation on all joints
    init_heading_quat_inv = quat_conjugate(heading_quat[:, 0])  # [X, D, 4]
    init_heading_quat_inv = einops.repeat(
        init_heading_quat_inv,
        "batch nbjoints quat -> batch nbframes nbjoints quat",
        nbframes=nbframes,
    )  # [X, T, J, 4]

    root_quat_wo_first_heading = None
    if root_quat is not None:
        # root rotation without first heading
        root_quat_wo_first_heading = quat_mul(
            init_heading_quat_inv[:, :, 0], root_quat
        )  # [X, T, 4]

    global_joint_quat_wo_first_heading = None
    if global_joint_quat is not None:
        global_joint_quat_wo_first_heading = quat_mul(
            init_heading_quat_inv, global_joint_quat
        )  # [X, T, J, 4]

    # root heading rotation on all joints, with first frame normalized to be 0
    heading_quat = quat_mul(heading_quat, init_heading_quat_inv)  # [X, T, J, 4]
    # fix some numerical issues: it is only a Y rotation axis rotation
    heading_quat[..., 1] = 0
    heading_quat[..., 3] = 0
    # make it a unit quaternion agains
    heading_quat = quat_unit(heading_quat)
    heading_quat_inv = quat_conjugate(heading_quat)  # [X, T, J, 4]

    info = {
        "heading_quat": heading_quat,
        "heading_quat_inv": heading_quat_inv,
        "init_heading_quat_inv": init_heading_quat_inv,
        "root_quat_wo_heading": root_quat_wo_heading,
        "root_quat_wo_first_heading": root_quat_wo_first_heading,
        "global_joint_quat_wo_heading": global_joint_quat_wo_heading,
        "global_joint_quat_wo_first_heading": global_joint_quat_wo_first_heading,
        "init_heading_quat_raw": init_heading_quat_raw,
    }
    return info


def compute_position_features(
    posed_joints: torch.Tensor,
    skeleton: SkeletonBase,
    heading_quat_inv: torch.Tensor,
    init_heading_quat_inv: torch.Tensor,
    foot_contacts: Optional[torch.Tensor],
    lengths: torch.Tensor,
    fps: float,
    local_root_vel_with_y: bool,
    local_vel_without_root: bool,
    removing_heading: bool = True,
    **kwargs,  # throw away unecessary args
):
    """Compute local position features from the original joints position, and the heading direction.
    Canonicalize each frame so that they all face in the same direction.

    Args:
        posed_joints (torch.Tensor): [B, T, J, 3] joint positions of the input motion
        skeleton (SkeletonBase): the skeleton corresponding to the human
        heading_quat_inv (torch.Tensor): the inverse quaternion of the heading direction
        init_heading_quat_inv (torch.Tensor): the first inverse direction (for canonicalization)
        lengths (torch.Tensor): lengths of each motion for batch computation
        fps (float): frame per seconds
        local_root_vel_with_y (bool): if True: put Y (gravity axis) in the local_root_vel
        local_vel_without_root (bool): if True: remove the root_idx from the local velocities
        **kwargs (Dict): unused arguments for easy function call

    Returns:
        info (Dict): dictionnary which encapsulate all the output: ric_data, local velocities, foot contacts etc.
    """

    root_idx = skeleton.root_idx

    global_positions = posed_joints.clone()
    root_pos_init_xz = global_positions[:, 0, root_idx, [0, 2]].clone()
    global_positions[..., [0, 2]] -= root_pos_init_xz[:, None, None]

    # all initially face Z+
    global_positions = quat_apply(init_heading_quat_inv, global_positions)

    # get root rotation/translation representation
    # root linear velovity on xz plane (T, 2)

    velocity = compute_vel_xyz(global_positions, fps, lengths=lengths)

    if removing_heading:
        # Rotate the velocity vector only if we remove heading
        local_vel = quat_apply(heading_quat_inv, velocity)
    else:
        local_vel = velocity

    if local_root_vel_with_y:
        root_local_vel = local_vel[:, :, root_idx].clone()
    else:
        root_local_vel = local_vel[:, :, root_idx, [0, 2]].clone()

    if local_vel_without_root:
        # remove the root from the local velocities
        local_vel, _ = einops.pack(
            [
                local_vel[:, :, :root_idx],
                local_vel[:, :, root_idx + 1 :],
            ],
            "batch time * dim",
        )

    # regroup data
    local_vel = einops.rearrange(
        local_vel,
        "batch time joints dim -> batch time (joints dim)",
    )

    # root height (T, 1)
    global_root_y = global_positions[:, :, root_idx, 1]
    global_root_pos = global_positions[:, :, root_idx]  # noqa

    # global_positions
    # get joint position represention (T, (J-1)x3)
    positions = global_positions.clone()

    # Root at the reference
    # avoid "-=" for good results
    positions[..., 0] = positions[..., 0] - positions[..., [root_idx], 0]
    positions[..., 2] = positions[..., 2] - positions[..., [root_idx], 2]

    # all pose face Z+ if we remove the heading
    if removing_heading:
        positions = quat_apply(heading_quat_inv, positions)

    # remove the root index as it is all zeros (for x and z), and y is already saved
    ric_data, _ = einops.pack(
        [positions[:, :, :root_idx], positions[:, :, root_idx + 1 :]],
        "batch time * dim",
    )

    # regroup data
    ric_data = einops.rearrange(
        ric_data, "batch time joints dim -> batch time (joints dim)"
    )

    if foot_contacts is None:
        # compute them with the positions/velocities

        # get foot contact representation (T, 4)
        # velocity is already padded correctly, with factor 1/dt
        feet_l, feet_r = foot_detect_from_pos_and_vel(
            global_positions, velocity, skeleton, 0.15, 0.10
        )
        foot_contacts = torch.cat((feet_l, feet_r), axis=-1)

    info = {
        "ric_data": ric_data,
        "local_root_vel": root_local_vel,
        "local_vel": local_vel,
        "global_root_y": global_root_y,
        "global_root_pos": global_root_pos,
        "root_pos_init_xz": root_pos_init_xz,
        "foot_contacts": foot_contacts,
    }
    return info


def compute_position_features_with_smooth_root(
    posed_joints: torch.Tensor,
    smooth_translations: torch.Tensor,
    skeleton: SkeletonBase,
    heading_quat_inv: torch.Tensor,
    init_heading_quat_inv: torch.Tensor,
    foot_contacts: Optional[torch.Tensor],
    lengths: torch.Tensor,
    fps: float,
    local_root_vel_with_y: bool,
    local_vel_without_root: bool,
    removing_heading: bool = True,
    **kwargs,  # throw away unecessary args
):
    """Compute local position features from the original joints position, and the heading direction.
    Canonicalize each frame so that they all face in the same direction.

    Args:
        posed_joints (torch.Tensor): [B, T, J, 3] joint positions of the input motion
        smooth_translations (torch.Tensor): [B, T, 3] smooth transltations of the input motion
        skeleton (SkeletonBase): the skeleton corresponding to the human
        heading_quat_inv (torch.Tensor): the inverse quaternion of the heading direction
        init_heading_quat_inv (torch.Tensor): the first inverse direction (for canonicalization)
        lengths (torch.Tensor): lengths of each motion for batch computation
        fps (float): frame per seconds
        local_root_vel_with_y (bool): if True: put Y (gravity axis) in the local_root_vel
        local_vel_without_root (bool): if True: remove the root_idx from the local velocities
        **kwargs (Dict): unused arguments for easy function call

    Returns:
        info (Dict): dictionnary which encapsulate all the output: ric_data, local velocities, foot contacts etc.
    """

    root_idx = skeleton.root_idx

    global_positions = posed_joints.clone()

    # use the smooth root instead
    # root_pos_init_xz = global_positions[:, 0, root_idx, [0, 2]].clone()
    root_pos_init_xz = smooth_translations[:, 0, [0, 2]].clone()
    global_positions[..., [0, 2]] -= root_pos_init_xz[:, None, None]

    # all initially face Z+
    global_positions = quat_apply(init_heading_quat_inv, global_positions)

    # also put the smooth translations at 0 and turn to face Z+
    smooth_translations[..., [0, 2]] -= root_pos_init_xz[:, None]
    smooth_translations = quat_apply(
        init_heading_quat_inv[:, :, root_idx], smooth_translations
    )

    # get root rotation/translation representation
    # root linear velovity on xz plane (T, 2)

    velocity = compute_vel_xyz(global_positions, fps, lengths=lengths)

    if removing_heading:
        # Rotate the velocity vector only if we remove heading
        local_vel = quat_apply(heading_quat_inv, velocity)
    else:
        local_vel = velocity

    if local_root_vel_with_y:
        root_local_vel = local_vel[:, :, root_idx].clone()
    else:
        root_local_vel = local_vel[:, :, root_idx, [0, 2]].clone()

    if local_vel_without_root:
        # remove the root from the local velocities
        local_vel, _ = einops.pack(
            [
                local_vel[:, :, :root_idx],
                local_vel[:, :, root_idx + 1 :],
            ],
            "batch time * dim",
        )

    # regroup data
    local_vel = einops.rearrange(
        local_vel,
        "batch time joints dim -> batch time (joints dim)",
    )

    # root height (T, 1)
    #
    # same results for now
    # global_root_y = global_positions[:, :, root_idx, 1]
    global_root_y = smooth_translations[..., 1]
    global_root_pos = smooth_translations

    # global_positions
    # get joint position represention (T, (J-1)x3)
    positions = global_positions.clone()

    # Root at the reference
    # avoid "-=" for good results
    positions[..., 0] = positions[..., 0] - smooth_translations[..., [0]]
    positions[..., 2] = positions[..., 2] - smooth_translations[..., [2]]

    # all pose face Z+ if we remove the heading
    if removing_heading:
        positions = quat_apply(heading_quat_inv, positions)

    # does not remove the root index
    # it is not all zeros anymore (for x and z)
    # ric_data, _ = einops.pack(
    #     [positions[:, :, :root_idx], positions[:, :, root_idx + 1 :]],
    #     "batch time * dim",
    # )
    ric_data = positions

    # regroup data
    ric_data = einops.rearrange(
        ric_data, "batch time joints dim -> batch time (joints dim)"
    )

    if foot_contacts is None:
        # compute them with the positions/velocities

        # get foot contact representation (T, 4)
        # velocity is already padded correctly, with factor 1/dt
        feet_l, feet_r = foot_detect_from_pos_and_vel(
            global_positions, velocity, skeleton, 0.15, 0.10
        )
        foot_contacts = torch.cat((feet_l, feet_r), axis=-1)

    info = {
        "ric_data": ric_data,
        "local_root_vel": root_local_vel,
        "local_vel": local_vel,
        "global_root_y": global_root_y,
        "global_root_pos": global_root_pos,
        "root_pos_init_xz": root_pos_init_xz,
        "foot_contacts": foot_contacts,
    }
    return info


def compute_heading_features(
    heading_quat: torch.Tensor,
    root_idx: int,
    fps: float,
    lengths: Optional[torch.Tensor] = None,
    **kwargs,  # throw away unecessary args
):
    """Compute heading features, local and global.

    Args:
        heading_quat: (torch.Tensor): [B, T, 4]  heading direction in quaternion
        root_idx (int): index of the root in the skeleton
        fps (float): frame per seconds
        lengths (Optional[torch.Tensor]): [...] size of each input batched. If not provided, positions should not be batched
        **kwargs (Dict): unused arguments for easy function call

    Returns:
        info (Dict): dictionnary which encapsulate all the output: global_root_heading, local_root_rot_vel, root_rot
    """
    # root rotation velocity along y-axis (T, 1)
    root_rot = heading_quat[:, :, root_idx]
    root_rot_angles = torch.arctan2(root_rot[..., 2], root_rot[..., 0]) * 2

    local_root_rot_vel = compute_vel_angle(root_rot_angles, fps, lengths=lengths)
    global_root_heading = torch.stack(
        [torch.cos(root_rot_angles), torch.sin(root_rot_angles)], dim=-1
    )
    info = {
        "global_root_heading": global_root_heading,
        "local_root_rot_vel": local_root_rot_vel,
        "root_rot": root_rot,
    }
    return info


def compute_local_rotation_features_wo_heading(
    local_joint_quat: torch.Tensor,
    root_quat_wo_heading: torch.Tensor,
    lengths: torch.Tensor,
    root_idx: int,
    fps: int,
    **kwargs,  # throw away unecessary args
):
    """Compute local rotational features from the original rotations, and the heading direction.

    Args:
        local_joint_quat (torch.Tensor): [B, T, J, 4] local joint quaternions of the input motion
        root_quat_wo_heading (torch.Tensor): the quaternion of the heading direction (after canonicalizing the first frame)
        root_idx (int): index of the root
        fps (float): frame per seconds
        **kwargs (Dict): unused arguments for easy function call

    Returns:
        info (Dict): dictionnary which encapsulate the output: rot_data
    """

    # get joint rotation representation using cont6d (T, Jx6)
    # replace the root quat with the heading invariant one
    rot_joint_quat, _ = einops.pack(
        [
            local_joint_quat[:, :, :root_idx],
            root_quat_wo_heading,
            local_joint_quat[:, :, root_idx + 1 :],
        ],
        "batch time * dim",
    )

    cont_6d_params = quaternion_to_cont6d(rot_joint_quat)
    rot_data = einops.rearrange(
        cont_6d_params, "batch time joints dim -> batch time (joints dim)"
    )
    info = {
        "rot_data": rot_data,
    }
    return info


def compute_local_rotation_features(
    local_joint_quat: torch.Tensor,
    root_quat_wo_first_heading: torch.Tensor,
    lengths: torch.Tensor,
    root_idx: int,
    fps: int,
    **kwargs,  # throw away unecessary args
):
    """Compute local rotational features from the original rotations, and the heading direction.

    Args:
        local_joint_quat (torch.Tensor): [B, T, J, 4] local joint quaternions of the input motion
        root_quat_wo_heading (torch.Tensor): the quaternion of the heading direction (after canonicalizing the first frame)
        root_idx (int): index of the root
        fps (float): frame per seconds
        **kwargs (Dict): unused arguments for easy function call

    Returns:
        info (Dict): dictionnary which encapsulate the output: rot_data
    """

    # get joint rotation representation using cont6d (T, Jx6)
    # replace the root quat with the heading invariant one
    rot_joint_quat, _ = einops.pack(
        [
            local_joint_quat[:, :, :root_idx],
            root_quat_wo_first_heading,
            local_joint_quat[:, :, root_idx + 1 :],
        ],
        "batch time * dim",
    )

    cont_6d_params = quaternion_to_cont6d(rot_joint_quat)
    rot_data = einops.rearrange(
        cont_6d_params, "batch time joints dim -> batch time (joints dim)"
    )
    info = {
        "rot_data": rot_data,
    }
    return info


def compute_global_rotation_features_wo_heading(
    global_joint_quat_wo_heading: torch.Tensor,
    lengths: torch.Tensor,
    root_idx: int,
    fps: int,
    **kwargs,  # throw away unecessary args
):
    """Compute local rotational features from the original rotations, and the heading direction.

    Args:
        global_joint_quat_wo_heading (torch.Tensor): [B, T, J, 4] global joint quaternions of the input motion without heading direction
        root_idx (int): index of the root
        fps (float): frame per seconds
        **kwargs (Dict): unused arguments for easy function call

    Returns:
        info (Dict): dictionnary which encapsulate the output: rot_data
    """

    # get joint rotation representation using cont6d [B, T, Jx6]
    cont_6d_params = quaternion_to_cont6d(global_joint_quat_wo_heading)
    global_rot_data = einops.rearrange(
        cont_6d_params, "batch time joints dim -> batch time (joints dim)"
    )
    info = {
        "global_rot_data": global_rot_data,
    }
    return info


def compute_global_rotation_features(
    global_joint_quat_wo_first_heading: torch.Tensor,
    lengths: torch.Tensor,
    root_idx: int,
    fps: int,
    **kwargs,  # throw away unecessary args
):
    """Compute local rotational features from the original rotations, and the heading direction.

    Args:
        global_joint_quat_wo_first_heading (torch.Tensor): [B, T, J, 4] global joint quaternions of the input motion without heading direction for the first frame only
        root_idx (int): index of the root
        fps (float): frame per seconds
        **kwargs (Dict): unused arguments for easy function call

    Returns:
        info (Dict): dictionnary which encapsulate the output: rot_data
    """

    # get joint rotation representation using cont6d [B, T, Jx6]
    cont_6d_params = quaternion_to_cont6d(global_joint_quat_wo_first_heading)
    global_rot_data = einops.rearrange(
        cont_6d_params, "batch time joints dim -> batch time (joints dim)"
    )
    info = {
        "global_rot_data": global_rot_data,
    }
    return info


def compute_motion_features(
    input_tensor_dict: Dict[str, torch.Tensor],
    motion_rep,
    *,
    # keywords mandatory arguments
    local_vel_without_root: bool,
    local_root_vel_with_y: bool,
    compute_heading_method: str,
    # keywords optional arguments
    lengths: Optional[torch.Tensor] = None,
    input_quat: Optional[bool] = False,
    only_pos: Optional[bool] = False,
    t_pose_from: Optional[str] = None,
    return_init_heading_info: bool = False,
    removing_heading: bool = True,
    using_smooth_root: bool = False,
    extra_skel: bool = False,
) -> torch.Tensor:
    """Generate feature vector for one motion given joint rotation matrices and positions.

    Args:
        input_tensor_dict (Dict[torch.Tensor]): contain all the input data (some can be optional)
            posed_joints (torch.Tensor): [..., T, J, 3] joint positions of the input motion
            local_joint_rots (torch.Tensor): [..., T, J, 3, 3] local joint rotation matrices of the input motion or [..., T, J, 4] quaternions
            global_joint_rots (torch.Tensor): [..., T, J, 3, 3] global joint rotation matrices of the input motion or [..., T, J, 4] quaternions
            foot_contacts (torch.Tensor): [..., T, 4] foot contacts (or None if we computed them from the positions)
        lengths (torch.Tensor): [...] lengths of the motions
        keys (List[str]): list of feature names in order, we want to concatenate in the motion rep
        compute_heading_method (str): "quat" / "refdir" / "refdir_inter"
        fps (float): frame per seconds
        lengths (torch.Tensor): lengths of each motion if batched
        local_root_vel_with_y (bool): if True: put Y (gravity axis) in the local_root_vel
        local_vel_without_root (bool): if True: remove the root_idx from the local velocities
        input_quat (bool): if True, consider the input to be quaternion, else matrices
        only_pos (bool): if True, do not compute any rotation features (rot_data)
        return_init_heading_info (bool): if True, return the canonicalization info
        removing_heading (bool): if True, return the canonicalization info
        using_smooth_root (bool): if True, compute a smooth trajectory for the root, and keep the hips joint in the positions data
        extra_skel (bool): whether we use the extra position representation or not. This is not use in this script yet but necessary to avoid using **kwargs
    """
    skeleton = motion_rep.skeleton
    fps = motion_rep.fps

    # compute on the fly missing necessary elements if possible

    local_joint_rots = input_tensor_dict.get("local_joint_rots")
    posed_joints = input_tensor_dict.get("posed_joints")
    global_joint_rots = input_tensor_dict.get("global_joint_rots")
    translation = input_tensor_dict.get("translation")
    foot_contacts = input_tensor_dict.get("foot_contacts")

    # quick einpack for batch_rigid, extend joints
    needs = tensor_needed(motion_rep)

    # Creating local joints rotations if it is not provided
    if "local_joint_rots" in needs and local_joint_rots is None:
        if global_joint_rots is None:
            raise ValueError(
                "Cannot create local joint rots, which is necessary for this motion rep."
            )
        if global_joint_rots.shape[-1] == 4:
            global_joint_rots = quaternion_to_matrix(global_joint_rots)
        # compute the locals from the globals
        local_joint_rots = global_mats_to_local_mats(global_joint_rots, skeleton)

    # Changing the local rotations / t_pose:
    # do this Before FK, so that our skeleton is compatible with the input
    if (
        local_joint_rots is not None
        and t_pose_from is not None
        and t_pose_from != motion_rep.skeleton.t_pose
    ):
        if local_joint_rots.shape[-1] == 4:
            local_joint_rots = quaternion_to_matrix(local_joint_rots)

        # do this after FK, so that we can keep the old neutral joints
        local_joint_rots, global_joint_rots = change_t_pose_local_mats(
            local_joint_rots,
            skeleton.t_pose,
            skeleton,
            t_pose_from=t_pose_from,
            return_global_rots=True,
        )

    # Creating global joints rotations or posed joints if it is not provided
    if ("global_joint_rots" in needs and global_joint_rots is None) or (
        "posed_joints" in needs and posed_joints is None
    ):
        if local_joint_rots is None:
            raise ValueError(
                "Cannot create global joint rots, which is necessary for this motion rep."
            )
        if local_joint_rots.shape[-1] == 4:
            local_joint_rots = quaternion_to_matrix(local_joint_rots)

        # one big chunk
        local_joint_rots, ps = einops.pack([local_joint_rots], "* nbjoints dim1 dim2")
        # run FK to compute global joints positions
        _joints = einops.repeat(
            skeleton.neutral_joints.to(dtype=local_joint_rots.dtype),
            "j k -> b j k",
            b=len(local_joint_rots),
        )

        _posed_joints, _global_joint_rots = batch_rigid_transform(
            local_joint_rots, _joints, skeleton.joint_parents, skeleton.root_idx
        )
        [local_joint_rots] = einops.unpack(local_joint_rots, ps, "* nbjoints dim1 dim2")
        [_global_joint_rots] = einops.unpack(
            _global_joint_rots, ps, "* nbjoints dim1 dim2"
        )

        if global_joint_rots is None and "global_joint_rots" in needs:
            global_joint_rots = _global_joint_rots

        if posed_joints is None and "posed_joints" in needs:
            if translation is None:
                raise ValueError(
                    "You should provide at least translation if posed_joints are missing."
                )
            [_posed_joints] = einops.unpack(_posed_joints, ps, "* nbjoints dim")
            # add the translation to the posed joints
            _posed_joints += translation[..., None, :]
            posed_joints = _posed_joints

    # Converting the local rotations into quaternions
    if local_joint_rots is not None and local_joint_rots.shape[-1] == 3:
        local_joint_quat = matrix_to_quaternion(local_joint_rots)
    else:
        local_joint_quat = local_joint_rots

    # Converting the global rotations into quaternions
    if global_joint_rots is not None and global_joint_rots.shape[-1] == 3:
        global_joint_quat = matrix_to_quaternion(global_joint_rots)
    else:
        global_joint_quat = global_joint_rots

    if only_pos:
        local_joint_quat = None
        global_joint_quat = None

    if local_joint_quat is not None:
        device = local_joint_quat.device
    else:
        assert posed_joints is not None
        device = posed_joints.device

    # Store important info used for subfunction
    info = {
        "fps": fps,
        "local_vel_without_root": local_vel_without_root,
        "local_root_vel_with_y": local_root_vel_with_y,
        "compute_heading_method": compute_heading_method,
        "skeleton": skeleton,
        "root_idx": skeleton.root_idx,
        "device": device,
    }

    # make it all [B, T, ...]
    input_dict = {
        "local_joint_quat": local_joint_quat,
        "global_joint_quat": global_joint_quat,
        "posed_joints": posed_joints,
        "foot_contacts": foot_contacts,
    }
    universal_dict, lengths, original_ps = make_universal_input(
        input_dict, lengths=lengths
    )
    local_joint_quat = universal_dict["local_joint_quat"]
    global_joint_quat = universal_dict["global_joint_quat"]
    posed_joints = universal_dict["posed_joints"]
    foot_contacts = universal_dict["foot_contacts"]

    if local_joint_quat is not None:
        nbatch, nbframes, nbjoints = local_joint_quat.shape[:3]
        root_quat = local_joint_quat[:, :, skeleton.root_idx]
    else:
        nbatch, nbframes, nbjoints = posed_joints.shape[:3]
        root_quat = None

    if global_joint_quat is not None:
        if local_joint_quat is None:
            # take the info from the global one
            # (it is the same as the local, as it is for the root)
            nbatch, nbframes, nbjoints = global_joint_quat.shape[:3]
            root_quat = global_joint_quat[:, :, skeleton.root_idx]
        else:
            # verify that the info is the same
            assert local_joint_quat.shape[:3] == global_joint_quat.shape[:3]
            assert (
                local_joint_quat[:, :, skeleton.root_idx]
                == global_joint_quat[:, :, skeleton.root_idx]
            ).all()

    info.update(
        {
            "local_joint_quat": local_joint_quat,
            "global_joint_quat": global_joint_quat,
            "root_quat": root_quat,
            "posed_joints": posed_joints,
            "foot_contacts": foot_contacts,
            "lengths": lengths,
            "nbatch": nbatch,
            "nbframes": nbframes,
            "nbjoints": nbjoints,
            "removing_heading": removing_heading,
        }
    )

    # compute raw root heading rotation # [B, T, 4]
    info["heading_quat_raw"] = calc_heading(return_quat=True, **info)

    # + canonicalize it, compute the inverse, compute the global root without heading etc
    info.update(compute_heading_info(**info))

    # compute features from the heading: global root / local root
    info.update(compute_heading_features(**info))

    if posed_joints is not None:
        # compute local positions features based on the heading

        if using_smooth_root:
            from .smooth_root import get_smooth_root_pos

            # using the smooth root
            # and store the hips pos in ric_data
            hip_translations = posed_joints[:, :, skeleton.root_idx]
            smooth_translations = get_smooth_root_pos(hip_translations)
            info["smooth_translations"] = smooth_translations
            info.update(compute_position_features_with_smooth_root(**info))
        else:
            # using the hips pos as root, removing the hip from ric_data
            info.update(compute_position_features(**info))

    if local_joint_quat is not None:
        # compute local rotation features based on heading
        if removing_heading:
            info.update(compute_local_rotation_features_wo_heading(**info))
        else:
            info.update(compute_local_rotation_features(**info))

    if global_joint_quat is not None:
        # compute global rotation features based on heading
        if removing_heading:
            info.update(compute_global_rotation_features_wo_heading(**info))
        else:
            info.update(compute_global_rotation_features(**info))

    # verify all the dimensions
    for key in motion_rep.keys:
        dim_lst = motion_rep.keys_dim[key]
        feat = info[key]

        if len(dim_lst) not in [0, 1]:
            raise ValueError(
                "In the key_dim dictionary, the lists should contain zero or one element."
            )

        # squeezed tensor
        if not dim_lst and len(feat.shape) == 2:
            continue

        # check the last dim
        if len(feat.shape) == 3 and feat.shape[-1] == dim_lst[0]:
            continue

        raise ValueError(
            f"For the key {key}, the shape of the sub feature is {feat.shape[-1]} where it should be {dim_lst[0]}."
        )

    features, feats_ps = einops.pack(
        [info[key] for key in motion_rep.keys],
        "batch time *",
    )
    # put back the original shape
    # https://einops.rocks/4-pack-and-unpack/
    [features] = einops.unpack(features, original_ps, "* nbframes dim")

    # return extra info relative to canonicalization
    if return_init_heading_info:
        [init_heading_quat] = einops.unpack(
            info["init_heading_quat_raw"], original_ps, "* dim"
        )
        [root_pos_init_xz] = einops.unpack(
            info["root_pos_init_xz"], original_ps, "* dim"
        )
        init_heading_info = {
            "init_heading_quat": init_heading_quat,
            "root_pos_init_xz": root_pos_init_xz,
        }
        return features, init_heading_info

    return features


def make_universal_input(
    input_tensor_dict: Dict[str, torch.Tensor],
    lengths: Optional[torch.Tensor] = None,
):
    """Make the input universal [B, T, J, X] from tensors of shape [..., T, J, X]

    Args:
        input_tensor_dict (Dict[torch.Tensor]): contain all the input data (some can be optional)
        lengths (torch.Tensor): lengths of each motion if batched
    Return:
        output_tensor_dict (Dict[torch.Tensor]): contain all the input data but with the same shape [B, T, J, X]
        lengths (torch.Tensor): [B]
        ps (Tuple): Save the indices for getting back the original shape
    """
    output_tensor_dict = {}
    keys = []
    for key, val in input_tensor_dict.items():
        if val is None:
            output_tensor_dict[key] = None
        else:
            keys.append(key)

    # should not be empty
    if not keys:
        raise ValueError("At least one tensor should not be None.")

    no_joints_dim_keys = ["foot_contacts", "translation"]
    candidate_first_keys = [key for key in keys if key not in no_joints_dim_keys]

    if not candidate_first_keys:
        raise ValueError("At least one tensor should have the nbjoints dim.")

    first_key = candidate_first_keys[0]
    first_el = input_tensor_dict[first_key]
    device = first_el.device

    # If the lengths is not provided, we do not assume full length
    # the input should be a single sequence
    if lengths is None:
        if len(first_el.shape) > 3:
            raise ValueError("You should provide the lengths tensor using batching.")
        elif len(first_el.shape) < 3:
            raise ValueError("The tensor is not recognized")
        # len(first_el.shape) == 3
        lengths = torch.tensor([len(first_el)], device=device)

    for key in keys:
        val = input_tensor_dict[key]
        if key in no_joints_dim_keys:
            # make is universally: [X, T, Y]
            val, _ = einops.pack([val], "* nbframes dim")
        else:
            # make is universally: [X, T, J, Y]
            val, original_ps = einops.pack([val], "* nbframes nbjoints dim")
        output_tensor_dict[key] = val

    # make is universally: [X]
    lengths, _ = einops.pack([lengths], "*")
    return output_tensor_dict, lengths, original_ps


def reconstruct_joint_rot_mats_from_ric_rots(
    joints_ric_rot6d: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot_quat: torch.Tensor,
    skeleton: SkeletonBase,
    removed_heading: bool,
    init_heading_quat: Optional[torch.Tensor] = None,
):
    """Recovers the local rotation matrices from the separated root heading quat and the other
    rotations.

    Args:
        joints_ric_rot6d (torch.Tensor): [..., T, nbjoints * 6] local 6D joint rotations
        root_pos (torch.Tensor): [..., T, 3] global root position
        root_quat (torch.Tensor): [..., T, 4] global root rot quaternion
        removed_heading (bool): if True, the heading have been removed from the rotations
        init_heading_quat (torch.Tensor): the first direction in case the heading is not removed.
                                          Will just rotate by that, not overrided
    Returns:
        joint_rot_mats (torch.Tensor): [..., T, nbjoints, 3, 3] joint rotation matrices
    """

    root_idx = skeleton.root_idx

    joints_ric_rot6d, ps = einops.pack([joints_ric_rot6d], "* time dim")
    root_pos, _ = einops.pack([root_pos], "* time dim")

    rot_6d = einops.rearrange(
        joints_ric_rot6d,
        "batch time (nbjoints six) -> batch time nbjoints six",
        six=6,
    )

    rotmat = cont6d_to_matrix(rot_6d)  # [B, T, 29, 3, 3]

    # deouble check
    if removed_heading:
        root_rot_quat, _ = einops.pack([root_rot_quat], "* time dim")
        heading_rot_mat = quaternion_to_matrix(root_rot_quat)
        # get global root rot by combining heading with local root rot
        # do matrix product here
        root_rot = torch.einsum(
            "btik,btkj->btij",
            heading_rot_mat,
            rotmat[:, :, root_idx],
        )
        # replace the root_rot by the full global one
        joint_rot_mats, _ = einops.pack(
            [
                rotmat[:, :, :root_idx],
                root_rot,
                rotmat[:, :, root_idx + 1 :],
            ],
            "batch time * dim1 dim2",
        )
    elif init_heading_quat is not None:
        init_heading_quat, _ = einops.pack([init_heading_quat], "* dim")  # [B, 4]

        heading_rot_mat = quaternion_to_matrix(init_heading_quat)  # [B, 3, 3]
        root_rot = torch.einsum(
            "bik,btkj->btij",
            heading_rot_mat,
            rotmat[:, :, root_idx],
        )
        # replace the root_rot by the full global one
        joint_rot_mats, _ = einops.pack(
            [
                rotmat[:, :, :root_idx],
                root_rot,
                rotmat[:, :, root_idx + 1 :],
            ],
            "batch time * dim1 dim2",
        )
    else:
        # do not put back heading since it was not removed
        joint_rot_mats = rotmat

    [joint_rot_mats] = einops.unpack(joint_rot_mats, ps, "* time nbjoints dim1 dim2")
    return joint_rot_mats


def reconstruct_joint_rot_mats_from_ric_global_rots(
    global_joints_ric_rots: torch.Tensor,
    root_rot_quat: torch.Tensor,
    skeleton: SkeletonBase,
    removed_heading: bool,
    init_heading_quat: Optional[torch.Tensor] = None,
):
    """Recovers the local rotation matrices from the separated root heading quat and the other
    global rotations without heading.

    Args:
        global_joints_ric_rots (torch.Tensor): [..., T, 3, 3] or [..., T, 6] global 6D joint rotations
        root_rot_quat (torch.Tensor): [..., T, 4] global root rot quaternion
        removed_heading (bool): if True, the heading have been removed from the rotations
        init_heading_quat (torch.Tensor): [..., 4] the first direction in case the heading is not removed.
                                          Will just rotate by that, not overrided
    Returns:
        joint_rot_mats (torch.Tensor): [..., T, nbjoints, 3, 3] joint rotation matrices
    """

    # saving for shapes
    _saved_root_quat = root_rot_quat

    # put batch and time together in a big batch
    global_joints_ric_rots, ps = einops.pack([global_joints_ric_rots], "* dim")

    root_rot_quat, _ = einops.pack([root_rot_quat], "* dim")

    is_6d = global_joints_ric_rots.shape[-1] == skeleton.nbjoints * 6
    if is_6d:
        global_rot = einops.rearrange(
            global_joints_ric_rots,
            "batch (nbjoints six) -> batch nbjoints six",
            six=6,
        )
        global_rot = cont6d_to_matrix(global_rot)  # [B, J, 3, 3]
    else:
        global_rot = einops.rearrange(
            global_joints_ric_rots,
            "batch (nbjoints dim1 dim2) -> batch nbjoints dim1 dim2",
            dim1=3,
            dim2=3,
        )

    nbjoints = global_rot.shape[1]

    if removed_heading:
        global_rotmat_wo_heading = global_rot

        heading_rot_mat = quaternion_to_matrix(root_rot_quat)  # [B, 3, 3]

        # put back heading direction to all global rotations
        global_rot_mats = torch.einsum(
            "bik,bdkj->bdij",
            heading_rot_mat,
            global_rotmat_wo_heading,
        )
    elif init_heading_quat is not None:
        # repeat by time
        init_heading_quat, _ = einops.pack([init_heading_quat], "* dim")  # [B, 4]

        _saved_root_quat, _ = einops.pack([_saved_root_quat], "* time dim")
        time = _saved_root_quat.shape[1]

        init_heading_quat = einops.repeat(
            init_heading_quat,
            "batch quat -> batch time quat",
            time=time,
        )

        # joint batch and time
        init_heading_quat, _ = einops.pack([init_heading_quat], "* dim")
        # first init rotation
        heading_rot_mat = quaternion_to_matrix(init_heading_quat)  # [B, 3, 3]

        # put back heading direction to all global rotations
        global_rot_mats = torch.einsum(
            "bik,bdkj->bdij",
            heading_rot_mat,
            global_rot,
        )
    else:
        global_rot_mats = global_rot

    # obtain back the local rotations from the new global rotations
    parent_rot_mats = global_rot_mats[:, skeleton.joint_parents]
    parent_rot_mats[:, skeleton.root_idx] = torch.eye(3)  # the root joint
    parent_rot_mats_inv = parent_rot_mats.transpose(2, 3)
    local_rot_mats = torch.einsum(
        "T N m n, T N n o -> T N m o", parent_rot_mats_inv, global_rot_mats
    )

    # add dummy rotations if it is more than the number of joints
    local_rot_mats[:, nbjoints:] = torch.eye(3)

    [local_rot_mats] = einops.unpack(local_rot_mats, ps, "* nbjoints dim1 dim2")
    return local_rot_mats, global_rot_mats


def recover_joints_with_FK(
    joint_rot_mats: torch.Tensor,
    root_pos: torch.Tensor,
    skeleton: SkeletonBase,
    neutral_joints: Optional[torch.Tensor] = None,
    return_global_rots: Optional[bool] = False,
):
    """Recovers global joint positions given the global root motion and local joint rotations.

    Args:
        joint_rot_mats (torch.Tensor): [..., T, J, 3, 3] joint rotation matrices
        root_pos (torch.Tensor): [..., T, 3] positions xyz
        skeleton (SkeletonBase)
        return_global_rots (Optional[bool]=False): whether to return the global rotations in addition to positions
    Returns:
        torch.Tensor: [B, T, J, 3] global joint positions
    """

    original_shape = joint_rot_mats.shape
    # big batch size for batch rigid transform
    big_bs = torch.tensor(original_shape[:-3]).prod().item()

    device = joint_rot_mats.device
    dtype = joint_rot_mats.dtype

    if neutral_joints is None:
        neutral_joints = skeleton.neutral_joints.to(device=device, dtype=dtype)
        joints = einops.repeat(
            neutral_joints,
            "nbjoints xyz -> big_batch nbjoints xyz",
            big_batch=big_bs,
        )
    else:
        joints = neutral_joints.to(device=device, dtype=dtype)
        # make it [B, J, 3]
        joints, _ = einops.pack([joints], "* nbjoints dim1")
        # make it [B, 1, J, 3]
        joints = joints[:, None]

        joints = einops.repeat(
            joints,
            "bs 1 nbjoints xyz -> bs time nbjoints xyz",
            time=big_bs // len(joints),
        )
        joints = einops.rearrange(
            joints, "bs time nbjoints xyz -> (bs time) nbjoints xyz"
        )
        assert len(joints) == big_bs

    parents = skeleton.joint_parents.to(device)
    root_idx = skeleton.root_idx
    joint_rot_mats, ps = einops.pack([joint_rot_mats], "* nbjoints dim1 dim2")

    batch_size, nbframes = joint_rot_mats.shape[:2]

    # perform FK
    positions, global_rots = batch_rigid_transform(
        joint_rot_mats, joints, parents, root_idx
    )
    [positions] = einops.unpack(positions, ps, "* nbjoints xyz")
    # apply global root pos
    positions = positions + root_pos[..., None, :]

    if return_global_rots:
        [global_rots] = einops.unpack(global_rots, ps, "* nbjoints dim1 dim2")
        return positions, global_rots
    else:
        return positions


def recover_joints_from_ric_pos(
    joints_ric_pos: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    skeleton,
    removed_heading: bool,
    init_heading_quat: Optional[torch.Tensor],
    using_smooth_root: bool,
):
    """Recovers global joint positions from global root motion and heading-invariant joint
    positions.

    Args:
        joints_ric_pos (torch.Tensor): [B, T, (J-1)*3] local joint positions, excluding root
        root_pos (torch.Tensor): [B, T, 3] global root position
        root_quat (torch.Tensor): [B, T, 4] global root rot quaternion
        removed_heading (bool): if True, the heading have been removed from the rotations
        init_heading_quat (torch.Tensor): the first direction in case the heading is not removed.
                                          Will just rotate by that, not overrided

    Returns:
        torch.Tensor: [B, T, J, 3] global joint positions
    """

    root_idx = skeleton.root_idx

    positions = einops.rearrange(
        joints_ric_pos,
        "batch time (nbjoints_minus_one xyz) -> batch time nbjoints_minus_one xyz",
        xyz=3,
    )  # [B, T, J-1, 3]

    if using_smooth_root:
        # removing the hips joints
        hips_positions = positions[:, :, root_idx].clone()

        positions, _ = einops.pack(
            [
                positions[:, :, :root_idx],
                positions[:, :, root_idx + 1 :],
            ],
            "batch time * dim",
        )
        # removing the hips positions to all the positions
        positions[..., [0, 2]] -= hips_positions[..., None, [0, 2]]

    time = positions.shape[1]

    if removed_heading:
        root_quat = einops.repeat(
            root_quat,
            "batch time quat -> batch time nbjoints_minus_one quat",
            nbjoints_minus_one=positions.shape[2],
        )
        # apply root heading to the positions
        positions = quat_apply(
            root_quat,
            positions,
        )
    elif init_heading_quat is not None:
        init_heading_all = einops.repeat(
            init_heading_quat,
            "batch quat -> batch time nbjoints_minus_one quat",
            nbjoints_minus_one=positions.shape[2],
            time=time,
        )
        # apply first heading to the positions
        positions = quat_apply(
            init_heading_all,
            positions,
        )
    else:
        pass

    # Concat root and joints
    # add back the root joint (initialized to 0)
    dummy_root = 0 * positions[:, :, 0]
    positions, _ = einops.pack(
        [
            positions[:, :, :root_idx],
            dummy_root,
            positions[:, :, root_idx:],
        ],
        "batch time * dim",
    )
    # add the XZ to all the joints
    positions[..., [0, 2]] += root_pos[..., None, [0, 2]]

    # put root_y
    positions[:, :, root_idx, 1] += root_pos[..., 1]
    return positions
