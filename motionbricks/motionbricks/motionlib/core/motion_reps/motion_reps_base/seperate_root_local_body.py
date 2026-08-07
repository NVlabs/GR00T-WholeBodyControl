from typing import Dict, Optional

import einops
import numpy as np
import torch

from motionbricks.motionlib.core.motion_reps.tools.changing_t_pose import (
    change_t_pose_global_mats,
    global_mats_to_local_mats,
)
from motionbricks.motionlib.core.motion_reps.tools.motion_features import (
    compute_motion_features,
    reconstruct_joint_rot_mats_from_ric_global_rots,
    reconstruct_joint_rot_mats_from_ric_rots,
    recover_joints_from_ric_pos,
    recover_joints_with_FK,
)
from motionbricks.motionlib.core.skeletons import SkeletonBase
from motionbricks.motionlib.core.utils.rotations import (
    cont6d_to_matrix,
    matrix_to_cont6d,
    matrix_to_quaternion,
    quat_apply,
    quat_mul,
)
from motionbricks.motionlib.core.utils.stats import Stats

from .motion_rep_base import MotionRepBase


class SeparatedRootLocalBody(MotionRepBase):
    """Motion representation that separates root and local body motion.

    The representation concatenates [root_motion, local_body_motion] features.

    Subclasses must define:

    Attributes:
        body_keys_dim (Dict): Mapping of body feature keys to their dimensions
        root_keys_dim (Dict): Mapping of root feature keys to their dimensions
        compute_kwargs (Dict): Extra parameters for computing features
        default_joint_positions_from (str): Default source for joint positions, one of:
            - "ric_data": Relative joint positions
            - "rot_data": Local joint rotations
            - "global_rot_data": Global joint rotations

    Methods:
        compute_root_pos_and_rot(): Extracts root position and rotation from features
        compute_root_rep_from_root_pos_and_rot(): Converts root pos/rot to root feature representation
        change_first_heading(): Rotate the motion rep to the given heading
    """

    def __init__(
        self,
        fps: float,
        skeleton: SkeletonBase,
        name: str,
        stats: Optional[Stats] = None,
    ):
        super().__init__(fps, skeleton, name, stats)

        # set the name with the setter defined below
        self.name = name

        # define usefull info for the class
        # from "root_keys_dim" and "body_keys_dim"
        [*self.root_keys], [*self.root_ps] = zip(*self.root_keys_dim.items())
        [*self.body_keys], [*self.body_ps] = zip(*self.body_keys_dim.items())

        # all info
        self.keys_dim = self.root_keys_dim | self.body_keys_dim
        self.keys = self.root_keys + self.body_keys
        self.ps = self.root_ps + self.body_ps

        # compute dim
        self.motion_rep_dim = sum((sum(x) if x else 1 for x in self.ps))
        self.motion_root_dim = sum((sum(x) if x else 1 for x in self.root_ps))
        self.motion_body_dim = sum((sum(x) if x else 1 for x in self.body_ps))

        if stats is not None and stats.is_loaded():
            assert self.motion_rep_dim == stats.get_dim()

        # no subset, entire motion rep
        self.motion_rep_subset_dim = self.motion_rep_dim

        # compute indices, for easy access
        self.indices = {}
        idx = 0
        for key, x in zip(self.keys, self.ps):
            dim = sum(x) if x else 1
            self.indices[key] = np.arange(idx, idx + dim)
            idx += dim

        self.indices["all"] = np.arange(0, self.motion_rep_dim)
        self.indices["root"] = np.arange(0, self.motion_root_dim)
        self.indices["body"] = np.arange(
            self.motion_root_dim, self.motion_root_dim + self.motion_body_dim
        )

        self.removed_heading = self.compute_kwargs.get("removing_heading", True)
        self.using_smooth_root = self.compute_kwargs.get("using_smooth_root", False)

        self.extra_skel = self.compute_kwargs.get("extra_skel", False)
        if self.extra_skel:
            # verify that the skeleton is extra
            assert (
                repr(self.skeleton.__class__).split("'")[1].endswith("Extra")
            ), "The skeleton should be an extra skeleton"

    def canonicalize_to_first_frame(
        self,
        motion: torch.Tensor,
        is_normalized: bool,
        to_normalize: bool,
        return_numpy: bool = False,
    ) -> torch.Tensor:
        new_motion = self.change_first_heading(
            motion,
            first_heading_angle=0.0,
            is_normalized=is_normalized,
            to_normalize=to_normalize,
            return_numpy=return_numpy,
        )
        return new_motion

    def randomize_first_heading(
        self,
        motion: torch.Tensor,
        is_normalized: bool,
        to_normalize: bool,
        return_numpy: bool = False,
    ) -> torch.Tensor:
        first_heading_angle = np.random.rand() * 2 * np.pi
        new_motion = self.change_first_heading(
            motion,
            first_heading_angle=first_heading_angle,
            is_normalized=is_normalized,
            to_normalize=to_normalize,
            return_numpy=return_numpy,
        )
        return new_motion

    def slice(self, motion, key: str):
        """Extracts a specific feature subset from the motion representation.

        Args:
            motion (torch.Tensor): Motion features of shape [..., D]
            key (str): Feature key to extract, must be in self.indices

        Returns:
            torch.Tensor: Extracted feature subset
        """

        assert key in self.indices
        assert motion.shape[-1] == self.motion_rep_dim
        return motion[..., self.indices[key]]

    def detect_all_body_or_root(self, motion: torch.Tensor):
        """Detects whether input contains full, root-only, or body-only features.

        Args:
            motion (torch.Tensor): Motion features of shape [..., D]

        Returns:
            str: One of "all", "root", or "body" indicating feature type

        Raises:
            ValueError: If input dimension doesn't match any known feature subset
        """
        dim = motion.shape[-1]
        if dim == self.motion_rep_dim:
            return "all"
        elif dim == self.motion_root_dim:
            return "root"
        elif dim == self.motion_body_dim:
            return "body"
        else:
            raise ValueError(f"This input dim is not recognized: {dim}")

    def normalize(self, motion: torch.Tensor, index=None) -> torch.Tensor:
        """Normalizes motion features using stored statistics.

        Args:
            motion (torch.Tensor): Motion features of shape [..., D] to normalize
            index (Optional[np.ndarray]): If motion is a subset of full representation,
                the indices within the full representation that the input contains

        Returns:
            torch.Tensor: Normalized motion features
        """
        if index is None:
            index = self.indices[self.detect_all_body_or_root(motion)]
        return self.stats.normalize(motion, index=index)

    def unnormalize(self, motion: torch.Tensor, index=None) -> torch.Tensor:
        """Unnormalizes motion features using stored statistics.

        Args:
            motion (torch.Tensor): Normalized motion features of shape [..., D]
            index (Optional[np.ndarray]): If motion is a subset of full representation,
                the indices within the full representation that the input contains

        Returns:
            torch.Tensor: Unnormalized motion features
        """
        if index is None:
            index = self.indices[self.detect_all_body_or_root(motion)]
        return self.stats.unnormalize(motion, index=index)

    def extract_root(self, motion: torch.Tensor) -> torch.Tensor:
        """Extracts root motion features from input.

        Args:
            motion (torch.Tensor): Motion features of shape [..., D]

        Returns:
            torch.Tensor: Root motion features

        Raises:
            ValueError: If input doesn't contain root features
        """
        type = self.detect_all_body_or_root(motion)

        if type == "root":
            return motion
        elif type == "all":
            return self.slice(motion, "root")
        else:
            raise ValueError("Cannot compute root info without root")

    def extract_body(self, motion: torch.Tensor) -> torch.Tensor:
        """Extracts body motion features from input.

        Args:
            motion (torch.Tensor): Motion features of shape [..., D]

        Returns:
            torch.Tensor: Body motion features

        Raises:
            ValueError: If input doesn't contain body features
        """
        type = self.detect_all_body_or_root(motion)
        if type == "body":
            return motion
        elif type == "all":
            return self.slice(motion, "body")
        else:
            raise ValueError("Cannot compute body info without body")

    def extract_foot_contacts(
        self,
        motion: torch.Tensor,
        is_normalized: bool,
        contact_thresh: Optional[float] = 0.5,
    ) -> torch.Tensor:
        """Extracts foot contact states from motion features.

        Args:
            motion (torch.Tensor): Motion features of shape [..., D]
            is_normalized (bool): Whether input features are normalized
            contact_thresh (Optional[float]): Threshold for binary contact classification.
                If None, returns raw contact values.

        Returns:
            torch.Tensor: Foot contact states, binary if threshold provided
        """
        assert "foot_contacts" in self.indices

        foot_contacts = motion[..., self.indices["foot_contacts"]]

        if is_normalized:
            foot_contacts = self.unnormalize(
                foot_contacts, index=self.indices["foot_contacts"]
            )

        if contact_thresh is not None:
            contacts = foot_contacts > contact_thresh
        else:
            contacts = foot_contacts
        return contacts

    def compute_root_pos_and_rot(
        self,
        motion: torch.Tensor,
        type: Optional[str] = None,
    ):
        """Compute root position and rotation from the representation.

        Args:
            motion (torch.Tensor): Motion features of shape [..., D] (can be the full representation or the root subset)
            type (Optional[str]): Type of motion features, if None will be auto-detected

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - torch.Tensor: [..., T, 3] global root position
                - torch.Tensor: [..., T, 4] global root rotation quaternion (heading only)
        """

        root_motion = self.extract_root(motion)  # noqa
        ...
        raise NotImplementedError
        return root_pos, root_rot_quat  # noqa

    def compute_root_rep_from_root_pos_and_rot(
        self, root_pos: torch.Tensor, root_rot_quat: torch.Tensor
    ):
        """Compute root representation from root position and rotation.

        Args:
            root_pos (torch.Tensor): [..., T, 3] global root position
            root_rot_quat (torch.Tensor): [..., T, 4] global root rotation quaternion

        Returns:
            torch.Tensor: [..., T, D] Root motion features
        """
        raise NotImplementedError
        return root_motion  # noqa

    def __call__(
        self,
        input_tensor_dict,
        to_normalize: bool,
        original_skeleton: Optional[SkeletonBase] = None,
        lengths: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        t_pose_from: Optional[str] = None,
        return_init_heading_info: bool = False,
    ) -> torch.Tensor:
        """Converts input motion data to feature representation.

        Args:
            input_tensor_dict (Dict): Input motion data containing joint rotations/positions
            to_normalize (bool): Whether to normalize the output features
            original_skeleton (Optional[SkeletonBase]): Source skeleton if different from self.skeleton
            lengths (Optional[torch.Tensor]): Sequence lengths for batched data
            return_numpy (bool): Whether to return numpy array instead of torch tensor
            t_pose_from (Optional[str]): Source t-pose if converting between differnet t-poses
            return_init_heading_info (bool): Whether to return initial heading information

        Returns:
            torch.Tensor: Motion features of shape [..., D]
            Optional[Dict]: Initial heading info if return_init_heading_info=True
        """
        if original_skeleton is None:
            # take the current one
            # assume the skeleton is the same
            original_skeleton = self.skeleton

        skel_slice = self.skeleton.get_skel_slice(original_skeleton)

        new_in_tensor_dict = dict()
        for key, val in input_tensor_dict.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)

            if not isinstance(val, torch.Tensor):
                # it is not a tensor
                new_in_tensor_dict[key] = val
                continue

            # no slice for those
            if key in ["translation", "foot_contacts", "root_pos"]:
                new_in_tensor_dict[key] = val
                continue

            # rotations matrices
            if val.shape[-1] == 3 and val.shape[-2] == 3:
                # verify the dimensions
                if original_skeleton.nbjoints != val.shape[-3]:
                    base_skel = getattr(original_skeleton, "base_skel", None)
                    if base_skel is not None and base_skel.nbjoints == val.shape[-3]:
                        pass
                    else:
                        raise ValueError(
                            "The data is not compatible with the provided skeleton."
                        )
                val = val[..., skel_slice, :, :]
            else:
                # verify the dimensions
                if original_skeleton.nbjoints != val.shape[-2]:
                    base_skel = getattr(original_skeleton, "base_skel", None)
                    if base_skel is not None and base_skel.nbjoints == val.shape[-2]:
                        pass
                    else:
                        raise ValueError(
                            "The data is not compatible with the provided skeleton."
                        )
                val = val[..., skel_slice, :]

            new_in_tensor_dict[key] = val

        features = compute_motion_features(
            new_in_tensor_dict,
            lengths=lengths,
            motion_rep=self,
            t_pose_from=t_pose_from,
            return_init_heading_info=return_init_heading_info,
            **self.compute_kwargs,
        )
        if return_init_heading_info:
            features, init_heading_info = features

        assert features.shape[-1] == self.motion_rep_dim

        if to_normalize:
            features = self.normalize(features)

        if return_numpy:
            features = features.cpu().numpy()

        if return_init_heading_info:
            return features, init_heading_info
        return features

    def inverse(
        self,
        features: torch.Tensor,
        is_normalized: bool,
        # Neutral joints (take the one from the skeleton by default)
        neutral_joints: Optional[torch.Tensor] = None,
        # Canonicalization info
        init_heading_info: Optional[Dict] = None,
        # Default options which depends on the motion rep
        joint_positions_from: Optional[str] = None,
        # Default options
        return_quat: bool = False,
        return_all: bool = False,
        run_fk: bool = True,
        return_numpy: bool = False,
        extra_skel_process: bool = True,
        # Optional to change t-pose
        t_pose_to: str = None,
    ) -> torch.Tensor:
        """Converts motion features back to joint rotations and positions.

        Args:
            features (torch.Tensor): Motion features of shape [..., D]
            is_normalized (bool): Whether input features are normalized
            neutral_joints (Optional[torch.Tensor]): Custom neutral joints to use for FK. Otherwise uses default from skeleton.
            init_heading_info (Optional[Dict]): Initial heading/position for motion
            joint_positions_from (Optional[str]): Source for computing default joint positions:
                "rot_data", "ric_data", or "global_rot_data"
            return_quat (bool): Return quaternions instead of rotation matrices
            return_all (bool): Return joints positions from all the possible sources
            run_fk (bool): Run forward kinematics to get joint positions
            return_numpy (bool): Return numpy arrays instead of torch tensors
            t_pose_to (Optional[str]): Target t-pose if converting between different t-poses

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - posed_joints: Joint positions
                - local_joint_rots: Local joint rotations
                - global_joint_rots: Global joint rotations
                - foot_contacts: Foot contact states
                - all_posed_joints: Joint positions from all methods if return_all=True
                - all_local_rots: Local rotations from all methods if return_all=True
                - all_global_rots: Global rotations from all methods if return_all=True
        """

        # Changing the t_pose for the rotations
        changing_t_pose = False
        if t_pose_to is not None:
            if self.skeleton.t_pose is None:
                raise ValueError(
                    "Cannot change the t_pose if we don't know the origin t_pose"
                )
            # True only if it is changing
            changing_t_pose = self.skeleton.t_pose != t_pose_to

        # storage of the outputs
        output_tensor_dict = {
            "all_posed_joints": {},
            "all_local_rots": {},
            "all_global_rots": {},
        }

        if joint_positions_from is None:
            joint_positions_from = self.default_joint_positions_from

        # make sure the default input can be taken
        assert (
            joint_positions_from in self.indices
            or joint_positions_from == "global_rot_data"
            and self.extra_skel
        )
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        # make is universally: [X, T, D]
        features, ps = einops.pack([features], "* nbframes dim")
        nbframes = features.shape[1]

        if init_heading_info is not None:
            # universal shapes [X, D]
            init_heading_info["init_heading_quat"] = einops.pack(
                [init_heading_info["init_heading_quat"]], "* dim"
            )[0].to(dtype=features.dtype, device=features.device)
            init_heading_info["root_pos_init_xz"] = einops.pack(
                [init_heading_info["root_pos_init_xz"]], "* dim"
            )[0].to(dtype=features.dtype, device=features.device)

        if is_normalized:
            features = self.unnormalize(features)

        root_pos, root_rot_quat = self.compute_root_pos_and_rot(features)
        output_tensor_dict["root_pos"] = root_pos.clone()

        # add the hips offset to the root pos if using smooth root
        if self.using_smooth_root:
            assert (
                "ric_data" in self.indices
            ), "ric_data should be in the motion rep if using smooth root. This is useful to get back the root offset"
            joints_ric_pos = self.slice(features, "ric_data")

            hips_indices = np.array(
                [
                    self.skeleton.root_idx,
                    self.skeleton.root_idx + 1,
                    self.skeleton.root_idx + 2,
                ]
            )
            hips_positions = joints_ric_pos[..., hips_indices]

            # add back the hips offset to the root position
            root_pos[..., [0, 2]] += hips_positions[..., [0, 2]]
            root_pos[..., 1] = hips_positions[..., 1]

        # remove the rotation of the root position
        # add back the init positions (already rotated by the rotations)
        if init_heading_info is not None:
            init_heading_quat = init_heading_info["init_heading_quat"]
            root_pos_init_xz = init_heading_info["root_pos_init_xz"]

            # add time dimension
            init_heading_quat_time = einops.repeat(
                init_heading_quat,
                "batch quat -> batch time quat",
                time=nbframes,
            )

            # then rotate, to get back the first Z rotation
            root_pos = quat_apply(init_heading_quat_time, root_pos)

            # and put the original root position + the offset
            root_pos[:, :, [0, 2]] += root_pos_init_xz[:, None]

            # add extra init rotation to all root rotations
            root_rot_quat = quat_mul(init_heading_quat_time, root_rot_quat)
        else:
            init_heading_quat = None

        # Do ric_data first, to get extrapos joints first, to get the global rotations
        posed_joints_extra_skel = None
        global_rot_mats_from_extra_skel = None
        if "ric_data" in self.indices and (
            return_all or joint_positions_from == "ric_data" or self.extra_skel
        ):
            joints_ric_pos = self.slice(features, "ric_data")

            # recover global joint positions from positions
            posed_joints_from_pos = recover_joints_from_ric_pos(
                joints_ric_pos,
                root_pos,
                root_rot_quat,
                skeleton=self.skeleton,
                removed_heading=self.removed_heading,
                init_heading_quat=init_heading_quat,  # in case the heading is not removed
                using_smooth_root=self.using_smooth_root,
            )

            if extra_skel_process and self.extra_skel:
                # crop the output
                skel_slice = self.skeleton.base_skel.get_skel_slice(self.skeleton)
                posed_joints_extra_skel = posed_joints_from_pos

                # compute the global rotations from extrapos
                global_rot_mats_from_extra_skel = compute_rotations_from_extrapos(
                    posed_joints_extra_skel, self.skeleton
                )
                posed_joints_from_pos = posed_joints_extra_skel[..., skel_slice, :]

            [posed_joints_from_pos] = einops.unpack(
                posed_joints_from_pos, ps, "* nbframes nbjoints xyz"
            )
            output_tensor_dict["all_posed_joints"]["ric_data"] = posed_joints_from_pos
            # the one by default
            if joint_positions_from == "ric_data":
                output_tensor_dict["posed_joints"] = posed_joints_from_pos

        if extra_skel_process and self.extra_skel:
            skeleton = self.skeleton.base_skel
        else:
            skeleton = self.skeleton

        if "rot_data" in self.indices and (
            return_all or joint_positions_from == "rot_data"
        ):
            joints_ric_rot6d = self.slice(features, "rot_data")
            # Get back the full rotation matrix
            local_rot_mats = reconstruct_joint_rot_mats_from_ric_rots(
                joints_ric_rot6d,
                root_pos,
                root_rot_quat,
                skeleton,
                removed_heading=self.removed_heading,
                init_heading_quat=init_heading_quat,
            )

            # fk is necessary for changing t-pose
            if run_fk or changing_t_pose:
                # recover global joint positions from rotations
                posed_joints_from_local_rots, global_rot_mats = recover_joints_with_FK(
                    local_rot_mats,
                    root_pos,
                    skeleton,
                    neutral_joints=neutral_joints,
                    return_global_rots=True,
                )

                # save the joints positions
                [posed_joints_from_local_rots] = einops.unpack(
                    posed_joints_from_local_rots, ps, "* nbframes nbjoints xyz"
                )
                output_tensor_dict["all_posed_joints"]["rot_data"] = (
                    posed_joints_from_local_rots
                )
                # the one by default
                if joint_positions_from == "rot_data":
                    output_tensor_dict["posed_joints"] = posed_joints_from_local_rots

            if changing_t_pose:
                # do this after FK, so that we can keep the old neutral joints
                # it will not be compatible with our skeleton anymore
                global_rot_mats = change_t_pose_global_mats(
                    global_rot_mats,
                    t_pose_to,
                    skeleton,
                )
                local_rot_mats = global_mats_to_local_mats(global_rot_mats, skeleton)

            # save the local rots from local
            [local_rot_mats] = einops.unpack(
                local_rot_mats, ps, "* nbframes nbjoints dim1 dim2"
            )
            if return_quat:
                local_rots = matrix_to_quaternion(local_rot_mats)
            else:
                local_rots = local_rot_mats

            output_tensor_dict["all_local_rots"]["rot_data"] = local_rots
            # the one by default
            if joint_positions_from == "rot_data":
                output_tensor_dict["local_joint_rots"] = local_rots

            # save the global rots from local
            [global_rot_mats] = einops.unpack(
                global_rot_mats, ps, "* nbframes nbjoints dim1 dim2"
            )

            if return_quat:
                global_rots = matrix_to_quaternion(global_rot_mats)
            else:
                global_rots = global_rot_mats

            output_tensor_dict["all_global_rots"]["rot_data"] = global_rots
            # the one by default
            if joint_positions_from == "rot_data":
                output_tensor_dict["global_joint_rots"] = global_rots

        if (
            "global_rot_data" in self.indices
            or global_rot_mats_from_extra_skel is not None
        ) and (return_all or joint_positions_from == "global_rot_data"):
            if global_rot_mats_from_extra_skel is not None:
                # use the global rotations from extrapos
                global_rot_mats_from_global = (
                    global_rot_mats_from_extra_skel  # [B, T, J, 3, 3]
                )

                # obtain back the local rotations from the new global rotations
                parent_rot_mats = global_rot_mats_from_global[
                    :, :, skeleton.joint_parents
                ]
                # root joint
                parent_rot_mats[:, :, skeleton.root_idx] = torch.eye(3)
                local_rot_mats_from_global = torch.einsum(
                    "B T N n m , B T N n o -> B T N m o",
                    parent_rot_mats,
                    global_rot_mats_from_global,
                )
                # add extra identity rots: should not be needed
                local_rot_mats_from_global[:, :, skeleton.nbjoints :] = torch.eye(3)
            else:
                global_joints_ric_rot = self.slice(features, "global_rot_data")

                # Get back the local rotation matrix
                local_rot_mats_from_global, global_rot_mats_from_global = (
                    reconstruct_joint_rot_mats_from_ric_global_rots(
                        global_joints_ric_rot,
                        root_rot_quat,
                        skeleton,
                        removed_heading=self.removed_heading,
                        init_heading_quat=init_heading_quat,
                    )
                )

            if run_fk:
                # recover global joint positions from rotations
                posed_joints_from_global_rots, global_rot_mats_from_global = (
                    recover_joints_with_FK(
                        local_rot_mats_from_global,
                        root_pos,
                        skeleton,
                        neutral_joints=neutral_joints,
                        return_global_rots=True,
                    )
                )

                # save the joints positions
                [posed_joints_from_global_rots] = einops.unpack(
                    posed_joints_from_global_rots, ps, "* nbframes nbjoints xyz"
                )
                output_tensor_dict["all_posed_joints"]["global_rot_data"] = (
                    posed_joints_from_global_rots
                )
                # the one by default
                if joint_positions_from == "global_rot_data":
                    output_tensor_dict["posed_joints"] = posed_joints_from_global_rots

            # fk was not necessary for changing t-pose since we already have the global
            if changing_t_pose:
                # do this after FK, so that we can keep the old neutral joints
                # it will not be compatible with our skeleton anymore
                global_rot_mats_from_global = change_t_pose_global_mats(
                    global_rot_mats_from_global,
                    t_pose_to,
                    skeleton,
                )
                local_rot_mats_from_global = global_mats_to_local_mats(
                    global_rot_mats_from_global,
                    skeleton,
                )

            # save the local rots from global
            [local_rot_mats_from_global] = einops.unpack(
                local_rot_mats_from_global, ps, "* nbframes nbjoints dim1 dim2"
            )
            if return_quat:
                local_rot_from_global = matrix_to_quaternion(local_rot_mats_from_global)
            else:
                local_rot_from_global = local_rot_mats_from_global

            output_tensor_dict["all_local_rots"]["global_rot_data"] = (
                local_rot_from_global
            )
            # the one by default
            if joint_positions_from == "global_rot_data":
                output_tensor_dict["local_joint_rots"] = local_rot_from_global

            # save the global rots from global
            [global_rot_mats_from_global] = einops.unpack(
                global_rot_mats_from_global, ps, "* nbframes nbjoints dim1 dim2"
            )

            if return_quat:
                global_rots_from_global = matrix_to_quaternion(
                    global_rot_mats_from_global
                )
            else:
                global_rots_from_global = global_rot_mats_from_global

            output_tensor_dict["all_global_rots"]["global_rot_data"] = (
                global_rots_from_global
            )
            # the one by default
            if joint_positions_from == "global_rot_data":
                output_tensor_dict["global_joint_rots"] = global_rots_from_global

        # foot contacts
        foot_contacts = self.extract_foot_contacts(
            features,
            is_normalized=False,  # already unnormalized
        )
        [output_tensor_dict["foot_contacts"]] = einops.unpack(
            foot_contacts, ps, "* nbframes dim"
        )

        if return_numpy:
            for key, val in output_tensor_dict.items():
                output_tensor_dict[key] = val.cpu().numpy()
        return output_tensor_dict

    def concat_root_body(self, root_data: torch.Tensor, body_data: torch.Tensor):
        """Concatenates root and body features into full motion representation.

        Args:
            root_data (torch.Tensor): Root motion features of shape [..., root_dim]
            body_data (torch.Tensor): Body motion features of shape [..., body_dim]

        Returns:
            torch.Tensor: Combined motion features of shape [..., D]
        """
        assert root_data.shape[-1] == self.motion_root_dim
        assert body_data.shape[-1] == self.motion_body_dim
        motion_data = torch.cat([root_data, body_data], axis=-1)
        assert motion_data.shape[-1] == self.motion_rep_dim
        return motion_data

    def get_feature_subset(self, motion: torch.Tensor, mode: str):
        """Extracts a feature subset based on specified mode.

        Args:
            motion (torch.Tensor): Motion features of shape [..., D]
            mode (str): Subset selection mode

        Returns:
            torch.Tensor: Feature subset
        """
        return motion

    def get_motion_rep_subset(self, mode: str):
        """Gets a subset motion representation.

        Args:
            mode (str): Subset selection mode

        Returns:
            SeparatedRootLocalBody: Motion representation for subset
        """
        return self

    def get_root_index_subset(self, mode: str):
        """Gets indices for root features subset.

        Args:
            mode (str): Subset selection mode

        Returns:
            np.ndarray: Indices for root features
        """
        return self.indices["root"]

    def get_body_index_subset(self, mode: str):
        """Gets indices for body features subset.

        Args:
            mode (str): Subset selection mode

        Returns:
            np.ndarray: Indices for body features
        """
        return self.indices["body"]

    def change_first_heading_body(
        self,
        motion: torch.Tensor,
        corrective_mat: torch.Tensor,
    ):
        root_idx = self.skeleton.root_idx
        if not self.removed_heading:
            all_body_feats = {}
            for key in self.body_keys:
                feats = self.slice(motion, key)
                if key == "ric_data":
                    positions = einops.rearrange(
                        feats,
                        "batch time (nbjoints_minus_one xyz) -> batch time nbjoints_minus_one xyz",
                        xyz=3,
                    )

                    if self.using_smooth_root:
                        assert positions.shape[-2] == self.skeleton.nbjoints
                    else:
                        assert positions.shape[-2] == (self.skeleton.nbjoints - 1)

                    # rotate the positions
                    positions = torch.einsum(
                        "bik,btdk->btdi",
                        corrective_mat,
                        positions,
                    )  # (AX)i = sum_k A_ik X_k
                    # put back into features
                    new_feats = einops.rearrange(
                        positions,
                        "batch time nbjoints_minus_one xyz -> batch time (nbjoints_minus_one xyz)",
                    )
                elif key == "local_vel":
                    local_vel = einops.rearrange(
                        feats,
                        "batch time (joints dim) -> batch time joints dim",
                        dim=3,
                    )
                    # rotate the velocities
                    new_local_vel = torch.einsum(
                        "bik,btdk->btdi",
                        corrective_mat,
                        local_vel,
                    )
                    new_feats = einops.rearrange(
                        new_local_vel,
                        "batch time joints dim -> batch time (joints dim)",
                    )
                elif key == "rot_data":
                    local_rot_data = einops.rearrange(
                        feats,
                        "batch time (joints dim) -> batch time joints dim",
                        dim=6,
                    )
                    # extract only the root rotation
                    local_root_6d = local_rot_data[..., root_idx, :]
                    local_root_mat = cont6d_to_matrix(local_root_6d)

                    new_local_root_mat = torch.einsum(
                        "bik,btkj->btij",
                        corrective_mat,
                        local_root_mat,
                    )
                    new_local_root_6d = matrix_to_cont6d(new_local_root_mat)
                    new_local_rot_data, _ = einops.pack(
                        [
                            local_rot_data[:, :, :root_idx],
                            new_local_root_6d,
                            local_rot_data[:, :, root_idx + 1 :],
                        ],
                        "batch time * dim",
                    )
                    new_feats = einops.rearrange(
                        new_local_rot_data,
                        "batch time joints dim -> batch time (joints dim)",
                    )
                elif key == "global_rot_data":
                    global_rot_data = einops.rearrange(
                        feats,
                        "batch time (joints dim) -> batch time joints dim",
                        dim=6,
                    )
                    global_rot_mats = cont6d_to_matrix(global_rot_data)
                    global_rot_mats = torch.einsum(
                        "bik,btdkj->btdij",
                        corrective_mat,
                        global_rot_mats,
                    )
                    # (AB)ij = sum_k A_ik B_jk
                    new_global_rot_data = matrix_to_cont6d(global_rot_mats)
                    new_feats = einops.rearrange(
                        new_global_rot_data,
                        "batch time joints dim -> batch time (joints dim)",
                    )
                elif key == "foot_contacts":
                    new_feats = feats
                else:
                    raise ValueError(
                        "This body feature is not recognised. Needs to verify the rotation for non heading motion reps."
                    )
                all_body_feats[key] = new_feats
            new_body_motion, _ = einops.pack(
                [all_body_feats[key] for key in self.body_keys], "batch time *"
            )
        else:
            # the body motion is already canonical for each frame
            new_body_motion = self.extract_body(motion)
        return new_body_motion
