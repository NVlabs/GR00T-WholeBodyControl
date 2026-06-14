from typing import Optional, Tuple

import einops
import numpy as np
import torch

from motionbricks.motionlib.core.motion_reps.tools.motion_features import (
    compute_vel_angle,
    compute_vel_xyz,
)
from motionbricks.motionlib.core.skeletons import SkeletonBase
from motionbricks.motionlib.core.utils.rotations import (
    angle_to_Y_rotation_matrix,
    quat_apply,
    quat_conjugate,
)
from motionbricks.motionlib.core.utils.stats import Stats

from .seperate_root_local_body import SeparatedRootLocalBody


class LocalRootLocalBody(SeparatedRootLocalBody):
    """Representation with local root and local body."""

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

        self.nfeats_vel = 3 if self.compute_kwargs["local_root_vel_with_y"] else 2
        self.root_keys_dim = {
            "local_root_rot_vel": [],  # vel theta
            "local_root_vel": [self.nfeats_vel],  # vel xyz or xz
            "global_root_y": [],  # gravity axis global
        }

        self.body_keys_dim = self.get_body_keys_dim(skeleton.nbjoints)

        # If we got the stats from the dual representation
        if self.dual_class is not None and self.dual_class._name_ in name:
            super().__init__(fps, skeleton, name, stats=None)
            # full stats for dual rep
            self.dual_rep = self.dual_class(fps, skeleton, name, stats=stats)
            self.dual_rep_mode = "local"
            # load the subset stats
            self.stats = self.dual_rep.local_motion_rep.stats
        else:
            self.dual_rep = None
            super().__init__(fps, skeleton, name, stats)

        self.root_mode = "local"

    def compute_root_pos_and_rot(
        self, motion: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute root position and rotation from a local root representation.

        Args:
            motion (torch.Tensor): [..., T, 4] (unnormalized) root motion where 4 is [angvel, vel_x, vel_z, height]
            or full motion representation [.., T, D]

        Returns:
            torch.Tensor: [..., T, 3] global root position
            torch.Tensor: [..., T, 4] global root rot quaternion
        """
        root_motion = self.extract_root(motion)

        device = root_motion.device
        batch, nbframes = root_motion.shape[:2]

        root_rot_vel_angles, root_local_vel, root_y_pos = einops.unpack(
            root_motion,
            [[], [self.nfeats_vel], []],
            "batch time *",
        )

        # multiply by dt (= div by fps) to recover true differences
        # dv = dv/dt * dt
        root_rot_vel_angles = root_rot_vel_angles / self.fps
        root_local_vel = root_local_vel / self.fps

        # Get Y-axis rotation from rotation velocity
        r_rot_ang = torch.zeros_like(root_rot_vel_angles, device=device)
        r_rot_ang[..., 1:] = root_rot_vel_angles[..., :-1]
        # don't use the dummy last one
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        # Create the quaternion from the angle
        r_rot_quat = torch.zeros((batch, nbframes, 4), device=device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang / 2)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang / 2)

        # Integrate position from linear velocity (for xz only)
        r_pos = torch.zeros((batch, nbframes, 3), device=device)
        r_pos[..., 1:, [0, 2]] = root_local_vel[..., :-1, :]
        # don't use the dummy last one

        # Add Y-axis rotation to velocities
        # shift one frame is needed to align root_quat with local_root_motion,
        # to match convert_root_global_to_local
        removed_heading = self.compute_kwargs.get("removing_heading", True)
        if removed_heading:
            r_pos[..., 1:, :] = quat_apply(r_rot_quat[..., :-1, :], r_pos[..., 1:, :])

        r_pos = torch.cumsum(r_pos, dim=-2)

        # Set height
        r_pos[..., 1] = root_y_pos
        return r_pos, r_rot_quat

    def compute_root_rep_from_root_pos_and_rot(
        self,
        r_pos,
        r_rot_quat,
        lengths: Optional[torch.Tensor] = None,
    ):
        """Compute root representation from root position and rotation.

        Args:
            r_pos (torch.Tensor): [..., T, 3] global root position
            r_rot_quat (torch.Tensor): [..., T, 4] global root rot quaternion (heading only)
        Return:
            root_motion (torch.Tensor): [..., T, 4]
        """

        root_rot_angles = torch.arctan2(r_rot_quat[..., 2], r_rot_quat[..., 0]) * 2
        local_root_rot_vel = compute_vel_angle(
            root_rot_angles, self.fps, lengths=lengths
        )
        root_vel = compute_vel_xyz(
            r_pos[..., None, :],
            self.fps,
            lengths=lengths,
        )[..., 0, :]

        removed_heading = self.compute_kwargs.get("removing_heading", True)
        if removed_heading:
            # rotate back
            local_root_vel = quat_apply(quat_conjugate(r_rot_quat), root_vel)[
                ..., [0, 2]
            ]
        else:
            local_root_vel = root_vel[..., [0, 2]]

        global_root_y = r_pos[..., 1]
        root_motion = torch.cat(
            [
                local_root_rot_vel[..., None],
                local_root_vel,
                global_root_y[..., None],
            ],
            axis=-1,
        )
        return root_motion

    def change_first_heading(
        self,
        motion: torch.Tensor,
        first_heading_angle: float,
        is_normalized: bool,
        to_normalize: bool,
        return_numpy: bool = False,
    ) -> torch.Tensor:
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)

        if is_normalized:
            motion = self.unnormalize(motion)

        # make is universally: [X, T, D]
        motion, ps = einops.pack([motion], "* nbframes dim")
        root_motion = self.extract_root(motion)

        device = root_motion.device
        batch, nbframes = root_motion.shape[:2]

        root_rot_vel_angles, root_local_vel, root_y_pos = einops.unpack(
            root_motion,
            [[], [self.nfeats_vel], []],
            "batch time *",
        )

        if not self.removed_heading:
            # need to find the original heading from the body features
            # only done with hips_pos at the moment
            assert self.compute_kwargs["compute_heading_method"] == "hips_pos"
            from motionbricks.motionlib.core.motion_reps.tools.heading import calc_heading_from_joints_pos

            root_idx = self.skeleton.root_idx

            # do the smooth root here
            if self.using_smooth_root:
                # extract the first position
                first_position = einops.rearrange(
                    self.slice(motion, "ric_data")[:, 0],
                    "batch (nbjoints xyz) -> batch nbjoints xyz",
                    xyz=3,
                )
                assert first_position.shape[-2] == self.skeleton.nbjoints
            else:
                # extract the first position
                first_position = einops.rearrange(
                    self.slice(motion, "ric_data")[:, 0],
                    "batch (nbjoints_minus_one xyz) -> batch nbjoints_minus_one xyz",
                    xyz=3,
                )
                assert first_position.shape[-2] == (self.skeleton.nbjoints - 1)

                # add back the dummy (to get good indices for finding the heading)
                dummy_root = 0 * first_position[:, 0]
                first_position, _ = einops.pack(
                    [
                        first_position[:, :root_idx],
                        dummy_root,
                        first_position[:, root_idx:],
                    ],
                    "batch * dim",
                )

            # compute the first heading angle
            prev_heading_angle = calc_heading_from_joints_pos(
                first_position[:, None],
                skeleton=self.skeleton,
                return_quat=False,
                inverse=False,
            )[:, 0]

            corrective_angle = first_heading_angle - prev_heading_angle
            corrective_mat = angle_to_Y_rotation_matrix(corrective_angle)

            if root_local_vel.shape[-1] == 2:
                # rotate the 2D velocities
                new_root_local_vel = torch.einsum(
                    "bik,btk->bti",
                    corrective_mat[..., [0, 2]][..., [0, 2], :],
                    root_local_vel,
                )
            else:
                # rotate the 3D velocities
                new_root_local_vel = torch.einsum(
                    "bik,btk->bti",
                    corrective_mat,
                    root_local_vel,
                )

            # create the new local root
            new_local_root, _ = einops.pack(
                [root_rot_vel_angles, new_root_local_vel, root_y_pos],
                "batch time *",
            )

            new_body_motion = self.change_first_heading_body(motion, corrective_mat)
            new_motion, _ = einops.pack(
                [new_local_root, new_body_motion],
                "batch nbframes *",
            )
        else:
            # changing first heading does not matter
            new_motion = motion

        [new_motion] = einops.unpack(new_motion, ps, "* nbframes dim")

        if to_normalize:
            new_motion = self.normalize(new_motion)

        if return_numpy:
            new_motion = new_motion.cpu().numpy()
        return new_motion
