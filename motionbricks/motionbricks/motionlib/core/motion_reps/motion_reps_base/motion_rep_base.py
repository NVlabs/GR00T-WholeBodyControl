from typing import Optional

import torch

from motionbricks.motionlib.core.skeletons import SkeletonBase
from motionbricks.motionlib.core.utils.stats import Stats


class MotionRepBase(torch.nn.Module):
    """Base class for a motion representation."""

    def __init__(
        self,
        fps: float,
        skeleton: SkeletonBase,
        name: str,
        stats: Optional[Stats] = None,
    ):
        super().__init__()
        self.stats = stats
        # native resolution of the motion
        self.fps = fps
        # skeleton holds joint names and other info like root and feet indices
        #      if need to do FK, also holds parents and neutral pose
        self.skeleton = skeleton

        # name of the motion rep
        self.name = name

        # number of joints in the skeleton
        self.num_joints = self.nbjoints = skeleton.nbjoints

    def normalize(self, motion: torch.Tensor, index=None) -> torch.Tensor:
        """Normalize a feature vector or a index of it.

        Args:
            motion (torch.Tensor): [..., D] the motion to normalize
            index (Optional[index]) the index to crop the motion
        """
        return self.stats.normalize(motion, index=index)

    def unnormalize(self, motion: torch.Tensor, index=None) -> torch.Tensor:
        """Unnormalize a normalized motion. Only one of root_only, nonroot_only, and contacts_only
        should be true.

        Args:
            motion (torch.Tensor): [..., D] the motion to unnormalize
            index (Optional[index]) the index to crop the motion
        """
        return self.stats.unnormalize(motion, index=index)
