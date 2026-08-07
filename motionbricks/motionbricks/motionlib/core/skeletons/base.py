import os.path as osp
from typing import Optional

import torch


class SkeletonBase(torch.nn.Module):
    "Utility class to hold info about a skeleton"

    # these should be defined in the subclass
    name = None
    bone_order_names_with_parents = None
    bone_order_names_no_root = None
    root_idx = None
    foot_joint_names = None
    foot_joint_idx = None
    hip_joint_names = None  # in order [right, left]
    hip_joint_idx = None  # in order [right, left]

    def __init__(
        self,
        folder: Optional[str] = None,
        name: Optional[str] = None,
        load: bool = True,
        t_pose: Optional[str] = None,
        **kwargs,  # to catch addition args in configs
    ):
        super().__init__()

        if name is not None:
            assert self.name in name

        self.folder = folder
        self.name = name
        self.t_pose = t_pose

        self.dim = len(self.bone_order_names_with_parents)

        if load and folder is not None:
            neutral_joints = torch.load(osp.join(folder, "joints.p")).squeeze()
            self.register_buffer("neutral_joints", neutral_joints, persistent=False)

            joint_parents = torch.load(osp.join(folder, "parents.p"))
            self.register_buffer("joint_parents", joint_parents, persistent=False)

        self.bone_order_names = [x for x, y in self.bone_order_names_with_parents]

        self.bone_parents = dict(self.bone_order_names_with_parents)
        self.bone_index = {x: idx for idx, x in enumerate(self.bone_order_names)}
        self.bone_order_names_index = self.bone_index

        # create the parents tensor on the fly
        joint_parents = torch.tensor(
            [
                -1 if (y := self.bone_parents[x]) is None else self.bone_index[y]
                for x in self.bone_order_names
            ]
        )

        if "joint_parents" not in self.__dict__:
            self.register_buffer("joint_parents", joint_parents, persistent=False)
        else:
            # check the saved one is coherent with the class
            assert (self.joint_parents == joint_parents).all()

        self.nbjoints = len(self.bone_order_names)

        # check lengths
        assert self.nbjoints == len(self.joint_parents)
        if "neutral_joints" in self.__dict__:
            assert self.nbjoints == len(self.neutral_joints)

        root_indices = torch.where(joint_parents == -1)[0]
        assert len(root_indices) == 1  # should be one root only
        self.root_idx = root_indices[0].item()

        if "neutral_joints" in self.__dict__:
            assert (self.neutral_joints[0] == 0).all()

        # remove the root
        self.bone_order_names_no_root = (
            self.bone_order_names[: self.root_idx]
            + self.bone_order_names[self.root_idx + 1 :]
        )

        self.foot_joint_names = self.left_foot_joint_names + self.right_foot_joint_names
        self.foot_joint_names_index = {
            x: idx for idx, x in enumerate(self.foot_joint_names)
        }

        self.left_foot_joint_idx = [
            self.bone_order_names.index(foot_joint)
            for foot_joint in self.left_foot_joint_names
        ]

        self.right_foot_joint_idx = [
            self.bone_order_names.index(foot_joint)
            for foot_joint in self.right_foot_joint_names
        ]

        self.foot_joint_idx = self.left_foot_joint_idx + self.right_foot_joint_idx

        self.hip_joint_idx = [
            self.bone_order_names.index(hip_joint) for hip_joint in self.hip_joint_names
        ]

    def __repr__(self):
        if self.folder is None:
            return f"{self.__class__.__name__}()"
        return f'{self.__class__.__name__}(folder="{self.folder}")'


