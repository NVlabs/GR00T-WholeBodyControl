from typing import Optional, Tuple

import einops
import numpy as np
import torch

from motionbricks.motionlib.core.skeletons import SkeletonBase
from motionbricks.motionlib.core.utils.rotations import (
    angle_to_Y_rotation_matrix,
)
from motionbricks.motionlib.core.utils.stats import Stats

from .seperate_root_local_body import SeparatedRootLocalBody


class GlobalRootLocalBody(SeparatedRootLocalBody):
    """Representation with global root."""

    dual_class = None

    def __init__(
        self,
        fps: float,
        skeleton: SkeletonBase,
        name: str,
        stats: Optional[Stats] = None,
    ):
        # Subclasses should define
        # get_body_keys_dim
        # compute_kwargs

        self.root_keys_dim = {
            "global_root_pos": [3],  # xyz
            "global_root_heading": [2],  # cos / sin
        }
        self.body_keys_dim = self.get_body_keys_dim(skeleton.nbjoints)

        # If we got the stats from the dual representation
        if self.dual_class is not None and self.dual_class._name_ in name:
            super().__init__(fps, skeleton, name, stats=None)
            # full stats for dual rep
            self.dual_rep = self.dual_class(fps, skeleton, name, stats=stats)
            self.dual_rep_mode = "global"
            # load the subset stats
            self.stats = self.dual_rep.global_motion_rep.stats
        else:
            self.dual_rep = None
            super().__init__(fps, skeleton, name, stats)

        # additional indices
        self.indices["global_root_pos_2d"] = self.indices["global_root_pos"][[0, 2]]
        self.root_mode = "global"

    def compute_root_pos_and_rot(
        self,
        motion: torch.Tensor,
        return_quat: bool = True,
        return_angle: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute root position and rotation from the representation.

        Args:
            motion (torch.Tensor): Motion tensor of shape [..., T, D] where T is number of frames, and D is feature dimension

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Global root position tensor of shape [B, T, 3]
                - Global root rotation quaternion tensor of shape [B, T, 4] (heading only)
        """

        motion, ps = einops.pack([motion], "* nbframes dim")
        root_motion = self.extract_root(motion)

        r_pos, rot_cos, rot_sin = einops.unpack(
            root_motion,
            [[3], [], []],
            "batch time *",
        )

        r_rot_ang = torch.atan2(rot_sin, rot_cos)
        r_rot_quat = torch.stack(
            [
                torch.cos(r_rot_ang / 2),
                torch.zeros_like(rot_cos),
                torch.sin(r_rot_ang / 2),
                torch.zeros_like(rot_cos),
            ],
            dim=-1,
        )

        [r_pos] = einops.unpack(r_pos, ps, "* nbframes xyz")
        [r_rot_quat] = einops.unpack(r_rot_quat, ps, "* nbframes quat")
        [r_rot_ang] = einops.unpack(r_rot_ang, ps, "* nbframes")
        if return_quat and not return_angle:
            return r_pos, r_rot_quat
        if not return_quat and return_angle:
            return r_pos, r_rot_ang
        if return_quat and return_angle:
            return r_pos, r_rot_quat, r_rot_ang
        if not return_quat and not return_angle:
            return r_pos

    def compute_root_rep_from_root_pos_and_rot(
        self,
        r_pos: torch.Tensor,
        r_rot_quat: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute root representation from root position and rotation.

        Args:
            r_pos (torch.Tensor): Global root position tensor of shape [..., T, 3]
            r_rot_quat (torch.Tensor): Global root rotation quaternion tensor of shape [..., T, 4]
            lengths (Optional[torch.Tensor]): Sequence lengths, defaults to None (necessary for batched sequences)

        Returns:
            torch.Tensor: Root motion representation tensor of shape [..., T, D]
        """

        root_rot_angles = torch.arctan2(r_rot_quat[..., 2], r_rot_quat[..., 0]) * 2
        root_motion = torch.cat(
            [
                r_pos,
                torch.cos(root_rot_angles)[..., None],
                torch.sin(root_rot_angles)[..., None],
            ],
            axis=-1,
        )
        return root_motion

    def change_first_heading(
        self,
        motion: torch.Tensor | np.ndarray,
        first_heading_angle: float | torch.Tensor,
        is_normalized: bool,
        to_normalize: bool,
        return_numpy: bool = False,
    ) -> torch.Tensor | np.ndarray:
        """Canonicalize motion by aligning the first frame to face +Z direction and moving to
        origin.

        Args:
            motion (Union[torch.Tensor, np.ndarray]): Input motion tensor of shape [..., T, D]
            is_normalized (bool): Whether input motion is normalized
            to_normalize (bool): Whether to normalize output motion
            return_numpy (bool): Whether to return numpy array. Defaults to False.

        Returns:
            Union[torch.Tensor, np.ndarray]: Canonicalized motion of same shape as input
        """
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)

        if is_normalized:
            motion = self.unnormalize(motion)

        # make is universally: [X, T, D]
        motion, ps = einops.pack([motion], "* nbframes dim")

        root_pos, root_rot_angle = self.compute_root_pos_and_rot(
            motion,
            return_angle=True,
            return_quat=False,
        )
        first_heading_angle = (
            first_heading_angle.reshape(root_rot_angle[..., 0].shape)
            if isinstance(first_heading_angle, torch.Tensor)
            else first_heading_angle
        )
        corrective_angle = first_heading_angle - root_rot_angle[..., 0]  # [Batch]
        new_angles = root_rot_angle + corrective_angle[..., None]  # [Batch, T]
        corrective_mat = angle_to_Y_rotation_matrix(corrective_angle)

        new_heading = torch.stack(
            [torch.cos(new_angles), torch.sin(new_angles)],
            dim=-1,
        )

        # move to origin and rotate
        first_pos = 1 * root_pos[:, [0]]
        first_pos[..., 1] = 0  # don't canonicalize height

        new_root_pos = torch.einsum(
            "bik,btk->bti",
            corrective_mat,
            root_pos - first_pos,
        )
        # create the new feature vector
        new_global_root, _ = einops.pack(
            [new_root_pos, new_heading], "batch nbframes *"
        )

        # be carefull here, could need to rotate part of the body motion
        new_body_motion = self.change_first_heading_body(motion, corrective_mat)
        new_motion, _ = einops.pack(
            [new_global_root, new_body_motion],
            "batch nbframes *",
        )
        [new_motion] = einops.unpack(new_motion, ps, "* nbframes dim")

        if to_normalize:
            new_motion = self.normalize(new_motion)

        if return_numpy:
            new_motion = new_motion.cpu().numpy()
        return new_motion
