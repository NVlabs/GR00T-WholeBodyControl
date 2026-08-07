from typing import Optional

import einops
import torch

from motionbricks.motionlib.core.skeletons import SkeletonBase
from motionbricks.motionlib.core.utils.rotations import diff_angles, diff_between_two_angles, quat_apply
from motionbricks.motionlib.core.utils.torch_utils import normalize_vec


def calc_heading(compute_heading_method: str, **kwargs):
    """Compute heading direction from various ways.

    Args:
        input_tensor (torch.Tensor): [..., T, 4] quaternions or joints_pos [..., T, X, 3]
        compute_heading_method (str): which function to use
    Returns:
        heading (torch.Tensor): [...] heading angle, or [..., 4] heading direction
    """

    if compute_heading_method == "hips_pos":
        heading = calc_heading_from_joints_pos(**kwargs)
    else:
        if compute_heading_method == "quat":
            heading = get_y_heading(**kwargs)
        elif compute_heading_method == "refdir":
            heading = calc_heading_refdir(**kwargs)
        elif compute_heading_method == "refdir_inter":
            heading = calc_heading_refdir(**kwargs)
        else:
            raise NotImplementedError
    return heading


def calc_heading_refdir(
    root_quat: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    return_quat: Optional[bool] = False,
    inverse: Optional[bool] = False,
    fix_rot: Optional[bool] = False,
    **kwargs,
):
    """Compute the heading direction from the root quaternion, by computing the changes from a
    reference direction.

    Args:
        root_quat (torch.Tensor): [..., T, 4] global root rot quaternion
        lengths (Optional[torch.Tensor]): [...] lengths of each motions (used only for the fix)
        return_quat (bool): return quaternions or not
        inverse (bool): return the inverse quaternion
        fix_rot (bool): use the rotation interpolation fix
        **kwargs (Dict): unecessary arguments
    Returns:
        heading (torch.Tensor): [...] heading angle, or [..., 4] heading quaternion
    """

    # type: (Tensor) -> Tensor
    # calculate heading direction from quaternion
    # the heading is the direction on the xz plane
    # root_quat must be normalized

    assert root_quat.shape[-1] == 4

    # make it [X, 4]
    rquat, ps = einops.pack([root_quat], "* quat")
    ref_dir = torch.zeros_like(rquat[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_apply(rquat, ref_dir)
    heading = -torch.atan2(rot_dir[..., 2], rot_dir[..., 0])
    # negative value because of xz

    # get back original shape
    [heading] = einops.unpack(heading, ps, "*")

    if fix_rot:
        # ON GOING
        __import__("ipdb").set_trace()
        [rot_dir] = einops.unpack(rot_dir, ps, "* dim")
        heading = fix_discountinuity_interpolation(heading, rot_dir, lengths)

    if inverse:
        heading = -heading

    if return_quat:
        heading_quat = torch.zeros_like(root_quat)
        heading_quat[..., 0] = torch.cos(heading / 2)
        heading_quat[..., 2] = torch.sin(heading / 2)
        return heading_quat

    return heading


# moved from previous
# convert_joint_pos_to_rep function of global_root_local_joints_root_rot.py
def calc_heading_from_joints_pos(
    posed_joints: torch.Tensor,
    skeleton: SkeletonBase,
    return_quat: Optional[bool] = False,
    inverse: Optional[bool] = False,
    **kargs,
):
    """Compute the heading direction from the joint positions, by looking at the hip vector.

    Args:
        posed_joints (torch.Tensor): [..., T, J, 3] global positions
        skeleton (SkeletonBase): skeleton of the human, used to find location of hips
        return_quat (bool): return quaternions or not
        inverse (bool): return the inverse quaternion
        **kwargs (Dict): unecessary arguments
    Returns:
        heading (torch.Tensor): [...] heading angle, or [..., 4] heading quaternion
    """
    assert posed_joints.shape[-1] == 3

    device = posed_joints.device
    dtype = posed_joints.dtype

    # compute root heading for the sequence from hip positions
    r_hip, l_hip = skeleton.hip_joint_idx

    skel2d = 1 * posed_joints
    skel2d[..., 1] *= 0  # only need 2D (x,z)

    across = skel2d[:, :, r_hip] - skel2d[:, :, l_hip]
    across = across / (torch.linalg.norm(across, axis=-1, keepdim=True) + 1e-6)

    root_heading = torch.cross(
        torch.tensor([[[0.0, 1.0, 0.0]]]).to(across), across, dim=-1
    )
    root_heading = root_heading / torch.linalg.norm(root_heading, axis=-1, keepdim=True)

    # compute (inverse) quaternion from heading
    root_heading = torch.atan2(
        root_heading[..., 0],
        root_heading[..., 2],
    )  # z is the forward facing direction of the motion, so it is the second argument for atan2
    if inverse:
        root_heading = -root_heading

    if return_quat:
        root_quat = torch.zeros(
            (*root_heading.shape, 4),
            device=device,
            dtype=dtype,
        )

        # NOTE: cos and sin here can crash with a floating point exception
        #   due to weird mkl issue on certain CPUs
        root_quat[..., 0] = torch.cos(root_heading / 2)
        root_quat[..., 2] = torch.sin(root_heading / 2)
        return root_quat

    return root_heading


def get_y_heading(
    root_quat: torch.Tensor,
    return_quat: Optional[bool] = False,
    inverse: Optional[bool] = False,
    **kargs,
) -> torch.Tensor:
    """Compute the heading direction from the root quaternion, by computing the changes from a
    reference direction.

    Args:
        root_quat (torch.Tensor): [..., T, 4] global root rot quaternion
        return_quat (bool): return quaternions or not
        inverse (bool): return the inverse quaternion
        **kwargs (Dict): unecessary arguments
    Returns:
        heading (torch.Tensor): [...] heading angle, or [..., 4] heading quaternion
    """

    assert root_quat.shape[-1] == 4

    root_quat = root_quat.clone()
    root_quat[..., 1] = 0
    root_quat[..., 3] = 0

    if inverse:
        # reverse the sin in -sin
        root_quat[..., 2] = -root_quat[..., 2]

    root_quat = normalize_vec(root_quat, dim=-1)

    if return_quat:
        return root_quat

    root_heading = 2 * torch.atan2(root_quat[2], root_quat[0])
    return root_heading


def fix_discountinuity_interpolation(
    heading: torch.Tensor,
    rot_dir: torch.Tensor,
    lengths: torch.Tensor,
    threshold=0.5,
):
    """Fix discountinuity in rotation
    Args:
        heading (torch.Tensor): [..., T]
        rot_dir (torch.Tensor): [..., T, 3]
    Returns:
        heading (torch.Tensor): [..., T] fixed
    """

    # make it [X, T]
    heading, ps = einops.pack([heading], "* nbframes")
    # make it [X, T, 3]
    rot_dir, _ = einops.pack([rot_dir], "* nbframes dim")

    # difference of angles
    dangle = diff_angles(heading)

    unreliable_mask = rot_dir[..., 1].abs() > 0.5

    # Find the indices where the mask changes value (True -> False or False -> True)
    change_indices = torch.diff(unreliable_mask.to(int))

    # Start of True intervals
    starts = torch.where(change_indices == 1)[0] + 1

    # End of True intervals
    ends = torch.where(change_indices == -1)[0]

    # If the mask starts with True, include the start
    if unreliable_mask[0]:
        starts = torch.cat([torch.tensor([0]), starts])

    # If the mask ends with True, include the end
    if unreliable_mask[-1]:
        ends = torch.cat([ends, torch.tensor([len(unreliable_mask) - 1])])

    # Combine starts and ends into intervals
    intervals = torch.column_stack((starts, ends))
    # close intervals

    # maximim index value of the array
    N = len(unreliable_mask) - 1
    for a, b in intervals:
        if a == 0:
            # start (only info we have)
            hstart = heading[a]
        else:
            # a >= 1
            # copy the last good heading
            hstart = heading[a - 1]
            if a >= 2:
                # apply the last good velocity
                # to the last heading
                # (that's why "-2")
                hstart += dangle[a - 2]
        if b == N:
            # end (only info we can take)
            hend = heading[b]
        else:
            # b <= N-1
            # copy the first following good heading
            hend = heading[b + 1]

            if b <= N - 2:
                # apply the first good velocity
                # to the first heading in reverse
                # (that's why "+1", and "-")
                hend -= dangle[b + 1]

        total_diff = diff_between_two_angles(hend, hstart)
        total_elements = b - a + 1
        if total_elements > 1:
            indexes = torch.arange(0, total_elements)
            test_fill = indexes * total_diff / (total_elements - 1) + hstart
            heading[a : b + 1] = test_fill
        else:
            # total_elements == 1
            # a = b
            heading[a] = (hstart + hend) / 2

    dangle = diff_angles(heading)

    # negative value because of xz
    return -heading
