import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch

from motionbricks.motionlib.core.skeletons import SkeletonBase
from motionbricks.motionlib.core.utils.stats import Stats

from .global_root_local_body import GlobalRootLocalBody
from .local_root_local_body import LocalRootLocalBody
from .seperate_root_local_body import SeparatedRootLocalBody

log = logging.getLogger(__name__)


class DualRootLocalBody(SeparatedRootLocalBody):
    """Representation with global root and local root."""

    def __init__(
        self,
        fps: float,
        skeleton: SkeletonBase,
        name: str,
        *,
        # keywords only args
        global_class: GlobalRootLocalBody,
        local_class: LocalRootLocalBody,
        stats: Optional[Stats] = None,
    ):
        global_motion_rep = global_class(fps=fps, skeleton=skeleton, name="global")
        local_motion_rep = local_class(fps=fps, skeleton=skeleton, name="local")

        assert (
            global_motion_rep.default_joint_positions_from
            == local_motion_rep.default_joint_positions_from
        )
        self.default_joint_positions_from = (
            global_motion_rep.default_joint_positions_from
        )

        self.root_keys_dim = (
            global_motion_rep.root_keys_dim | local_motion_rep.root_keys_dim
        )
        # double check there is no overlap between them
        assert len(self.root_keys_dim) == len(global_motion_rep.root_keys_dim) + len(
            local_motion_rep.root_keys_dim
        )

        # make sure they use the same body motion representation
        assert global_motion_rep.body_keys_dim == local_motion_rep.body_keys_dim
        self.body_keys_dim = global_motion_rep.body_keys_dim

        super().__init__(fps, skeleton, name, stats)

        # assign after module.init
        self.global_motion_rep = global_motion_rep
        self.local_motion_rep = local_motion_rep

        # useful indices
        self.indices["global_root"] = np.arange(0, global_motion_rep.motion_root_dim)
        self.indices["local_root"] = np.arange(
            global_motion_rep.motion_root_dim,
            self.motion_root_dim,
        )
        self.indices["global_rep"] = np.concatenate(
            (self.indices["global_root"], self.indices["body"])
        )
        self.indices["local_rep"] = np.concatenate(
            (self.indices["local_root"], self.indices["body"])
        )

        # make sure it uses the same args for compute local motion features
        assert global_motion_rep.compute_kwargs == local_motion_rep.compute_kwargs
        self.compute_kwargs = global_motion_rep.compute_kwargs

        # subset: it is global by default
        self.motion_rep_subset_dim = self.global_motion_rep.motion_rep_dim

        # no dual rep as it is the rep itself
        self.dual_rep = None

        # extract the stats for the subset motion rep
        if stats is not None and stats.is_loaded():
            # global stats
            global_mean = stats.mean[self.indices["global_rep"]]
            global_std = stats.std[self.indices["global_rep"]]

            self.global_motion_rep.stats = Stats(eps=stats.eps, legacy=stats.legacy)
            self.global_motion_rep.stats.register_from_tensors(global_mean, global_std)

            # local stats
            local_mean = stats.mean[self.indices["local_rep"]]
            local_std = stats.std[self.indices["local_rep"]]
            self.local_motion_rep.stats = Stats(eps=stats.eps, legacy=stats.legacy)
            self.local_motion_rep.stats.register_from_tensors(local_mean, local_std)

    def get_feature_subset(
        self, motion: torch.Tensor, mode: str = "global"
    ) -> torch.Tensor:
        """Extract either global or local features from the motion tensor.

        Args:
            motion: Motion tensor of shape [..., feature_dim]
            mode: Either "global" or "local"

        Returns:
            Subset of features corresponding to the specified mode
        """
        assert mode in ["global", "local"]
        return motion[..., self.indices[f"{mode}_rep"]]

    def get_motion_rep_subset(
        self, mode: str = "global"
    ) -> Union[GlobalRootLocalBody, LocalRootLocalBody]:
        """Get the motion representation object for either global or local features.

        Args:
            mode: Either "global" or "local"

        Returns:
            The corresponding motion representation object

        Raises:
            ValueError: If mode is not "global" or "local"
        """
        if mode == "global":
            return self.global_motion_rep
        elif mode == "local":
            return self.local_motion_rep
        else:
            raise ValueError("The mode should be global or local only.")

    def get_root_index_subset(self, mode: str = "global") -> np.ndarray:
        """Get indices for either global or local root features.

        Args:
            mode: Either "global" or "local"

        Returns:
            Array of indices for the specified root features

        Raises:
            ValueError: If mode is not "global" or "local"
        """
        if mode == "global":
            return self.indices["global_root"]
        elif mode == "local":
            return self.indices["local_root"]
        else:
            raise ValueError("The mode should be global or local only.")

    def get_body_index_subset(self, mode: str) -> np.ndarray:
        """Get indices for body features.

        Args:
            mode: Unused parameter kept for API consistency

        Returns:
            Array of indices for body features
        """
        return self.indices["body"]

    def _one_subset_to_the_other(
        self,
        features: torch.Tensor,
        is_normalized: bool,
        to_normalize: bool,
        mode: str,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert features between global and local representations.

        Args:
            features: Input features to convert
            is_normalized: Whether input features are normalized
            to_normalize: Whether to normalize output features
            mode: Either "local_to_global" or "global_to_local"
            lengths: Optional sequence lengths for batched data

        Returns:
            Converted features in the target representation
        """
        if mode == "local_to_global":
            motion_rep_from = self.local_motion_rep
            motion_rep_to = self.global_motion_rep
        elif mode == "global_to_local":
            motion_rep_from = self.global_motion_rep
            motion_rep_to = self.local_motion_rep
        else:
            raise ValueError("Mode was not recognized.")

        from_rep, to_rep = mode.split("_to_")

        data_type = motion_rep_from.detect_all_body_or_root(features)
        if data_type == "all":
            # also take care of the body
            body_motion = motion_rep_from.extract_body(features)
        else:
            assert data_type == "root"

        root_motion = motion_rep_from.extract_root(features)

        if is_normalized:
            root_motion = self.unnormalize(
                root_motion, index=self.indices[f"{from_rep}_root"]
            )
            # unnormalize the body only if needed
            if not to_normalize and data_type == "all":
                body_motion = self.unnormalize(body_motion, index=self.indices["body"])

        r_pos, r_rot_quat = motion_rep_from.compute_root_pos_and_rot(root_motion)

        # compute new root representation
        new_root_motion = motion_rep_to.compute_root_rep_from_root_pos_and_rot(
            r_pos, r_rot_quat, lengths
        )

        if to_normalize:
            new_root_motion = self.normalize(
                new_root_motion, index=self.indices[f"{to_rep}_root"]
            )
            if not is_normalized and data_type == "all":
                # normalize the body only if needed
                body_motion = self.normalize(body_motion, index=self.indices["body"])

        if data_type == "root":
            assert new_root_motion.shape[-1] == motion_rep_to.motion_root_dim
            return new_root_motion

        # otherwise recreate the all feature vector
        new_features = motion_rep_to.concat_root_body(new_root_motion, body_motion)
        assert new_features.shape[-1] == motion_rep_to.motion_rep_dim
        return new_features

    def global_to_local(
        self,
        features: torch.Tensor,
        is_normalized: bool,
        to_normalize: bool,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert features from global to local representation.

        Args:
            features: Input features in global representation
            is_normalized: Whether input features are normalized
            to_normalize: Whether to normalize output features
            lengths: Optional sequence lengths for batched data

        Returns:
            Features in local representation
        """
        return self._one_subset_to_the_other(
            features,
            is_normalized=is_normalized,
            to_normalize=to_normalize,
            mode="global_to_local",
            lengths=lengths,
        )

    def local_to_global(
        self,
        features: torch.Tensor,
        is_normalized: bool,
        to_normalize: bool,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert features from local to global representation.

        Args:
            features: Input features in local representation
            is_normalized: Whether input features are normalized
            to_normalize: Whether to normalize output features
            lengths: Optional sequence lengths for batched data

        Returns:
            Features in global representation
        """
        return self._one_subset_to_the_other(
            features,
            is_normalized=is_normalized,
            to_normalize=to_normalize,
            mode="local_to_global",
            lengths=lengths,
        )

    def cat_global_and_local(
        self,
        global_motion: torch.Tensor,
        local_motion: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate global and local motion features.

        Args:
            global_motion: Motion features in global representation
            local_motion: Motion features in local representation

        Returns:
            Combined motion features containing both roots and body (the dual form)
        """
        # extract global info
        global_root = self.global_motion_rep.extract_root(global_motion)
        global_body = self.global_motion_rep.extract_body(global_motion)

        # extract local info
        local_root = self.local_motion_rep.extract_root(local_motion)
        local_body = self.local_motion_rep.extract_body(local_motion)

        # global_body and local body should be the same

        assert (local_body == global_body).all()
        motion = torch.cat([global_root, local_root, local_body], axis=-1)
        return motion

    def change_first_heading(
        self,
        motion: torch.Tensor,
        first_heading_angle: float,
        is_normalized: bool,
        to_normalize: bool,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Transform motion to be relative to the first frame.

        Args:
            motion: Input motion features
            is_normalized: Whether input features are normalized
            to_normalize: Whether to normalize output features
            return_numpy: Whether to return a numpy array instead of tensor

        Returns:
            Canonicalized motion features
        """
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)

        if is_normalized:
            motion = self.unnormalize(motion)

        new_global_rep_motion = self.global_motion_rep.change_first_heading(  # noqa
            self.get_feature_subset(motion, mode="global"),
            first_heading_angle,
            is_normalized=False,
            to_normalize=False,
        )

        new_local_rep_motion = self.local_motion_rep.change_first_heading(  # noqa
            self.get_feature_subset(motion, mode="local"),
            first_heading_angle,
            is_normalized=False,
            to_normalize=False,
        )
        new_motion = self.cat_global_and_local(
            new_global_rep_motion, new_local_rep_motion
        )

        if to_normalize:
            new_motion = self.normalize(new_motion)

        if return_numpy:
            new_motion = new_motion.cpu().numpy()
        return new_motion

    def compute_root_pos_and_rot(
        self, motion: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute root position and rotation from the representation.

        Returns:
            torch.Tensor: [..., T, 3] global root position
            torch.Tensor: [..., T, 4] global root rot quaternion (heading only)
        """
        global_features = self.get_feature_subset(motion, mode="global")
        r_pos, r_rot_quat = self.global_motion_rep.compute_root_pos_and_rot(
            global_features
        )
        return r_pos, r_rot_quat
