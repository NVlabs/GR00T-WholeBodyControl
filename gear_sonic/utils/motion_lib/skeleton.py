# from abc import ABCMeta, abstractmethod, classmethod
from abc import abstractmethod
from collections import OrderedDict
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import scipy.ndimage.filters as filters
import torch

from gear_sonic.isaac_utils.rotations import (
    quat_identity_like,
    quat_inverse,
    quat_mul_norm,
    transform_from_rotation_translation,
    transform_mul,
    transform_rotation,
    transform_translation,
)


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            np.int_
            | np.intc
            | np.intp
            | np.int8
            | np.int16
            | np.int32
            | np.int64
            | np.uint8
            | np.uint16
            | np.uint32
            | np.uint64,
        ):
            return int(obj)
        elif isinstance(obj, np.float_ | np.float16 | np.float32 | np.float64):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype), "shape": obj.shape}
        return json.JSONEncoder.default(self, obj)


def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and "__ndarray__" in dct:
        data = np.asarray(dct["__ndarray__"], dtype=dct["dtype"])
        return data.reshape(dct["shape"])
    return dct


class Serializable:
    """Implementation to read/write to file.
    All class the is inherited from this class needs to implement to_dict() and
    from_dict()
    """

    @classmethod
    def from_dict(cls, dict_repr, *args, **kwargs):
        """Read the object from an ordered dictionary

        :param dict_repr: the ordered dictionary that is used to construct the object
        :type dict_repr: OrderedDict
        :param args, kwargs: the arguments that need to be passed into from_dict()
        :type args, kwargs: additional arguments
        """
        pass

    @abstractmethod
    def to_dict(self):
        """Construct an ordered dictionary from the object

        :rtype: OrderedDict
        """
        pass

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        """Read the object from a file (either .npy or .json)

        :param path: path of the file
        :type path: string
        :param args, kwargs: the arguments that need to be passed into from_dict()
        :type args, kwargs: additional arguments
        """
        if path.endswith(".json"):
            with open(path) as f:
                d = json.load(f, object_hook=json_numpy_obj_hook)
        elif path.endswith(".npy"):
            d = np.load(path, allow_pickle=True).item()
        else:
            assert False, f"failed to load {cls.__name__} from {path}"
        assert d["__name__"] == cls.__name__, "the file belongs to {}, not {}".format(
            d["__name__"], cls.__name__
        )
        return cls.from_dict(d, *args, **kwargs)

    def to_file(self, path: str) -> None:
        """Write the object to a file (either .npy or .json)

        :param path: path of the file
        :type path: string
        """
        if os.path.dirname(path) != "" and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        d = self.to_dict()
        d["__name__"] = self.__class__.__name__
        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(d, f, cls=NumpyEncoder, indent=4)
        elif path.endswith(".npy"):
            np.save(path, d)


class TensorUtils(Serializable):
    @classmethod
    def from_dict(cls, dict_repr, *args, **kwargs):
        """Read the object from an ordered dictionary

        :param dict_repr: the ordered dictionary that is used to construct the object
        :type dict_repr: OrderedDict
        :param kwargs: the arguments that need to be passed into from_dict()
        :type kwargs: additional arguments
        """
        return torch.from_numpy(dict_repr["arr"].astype(dict_repr["context"]["dtype"]))

    def to_dict(self):
        """Construct an ordered dictionary from the object

        :rtype: OrderedDict
        """
        return NotImplemented


def tensor_to_dict(x):
    """Construct an ordered dictionary from the object

    :rtype: OrderedDict
    """
    x_np = x.numpy()
    return {"arr": x_np, "context": {"dtype": x_np.dtype.name}}


class SkeletonTree(Serializable):
    """
    A skeleton tree gives a complete description of a rigid skeleton. It describes a tree structure
    over a list of nodes with their names indicated by strings. Each edge in the tree has a local
    translation associated with it which describes the distance between the two nodes that it
    connects.

    Basic Usage:
        >>> t = SkeletonTree.from_mjcf(SkeletonTree.__example_mjcf_path__)
        >>> t
        SkeletonTree(
            node_names=['torso', 'front_left_leg', 'aux_1', 'front_left_foot', 'front_right_leg', 'aux_2', 'front_right_foot', 'left_back_leg', 'aux_3', 'left_back_foot', 'right_back_leg', 'aux_4', 'right_back_foot'],
            parent_indices=tensor([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  0, 10, 11]),
            local_translation=tensor([[ 0.0000,  0.0000,  0.7500],
                    [ 0.0000,  0.0000,  0.0000],
                    [ 0.2000,  0.2000,  0.0000],
                    [ 0.2000,  0.2000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000],
                    [-0.2000,  0.2000,  0.0000],
                    [-0.2000,  0.2000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000],
                    [-0.2000, -0.2000,  0.0000],
                    [-0.2000, -0.2000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000],
                    [ 0.2000, -0.2000,  0.0000],
                    [ 0.2000, -0.2000,  0.0000]])
        )
        >>> t.node_names
        ['torso', 'front_left_leg', 'aux_1', 'front_left_foot', 'front_right_leg', 'aux_2', 'front_right_foot', 'left_back_leg', 'aux_3', 'left_back_foot', 'right_back_leg', 'aux_4', 'right_back_foot']
        >>> t.parent_indices
        tensor([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  0, 10, 11])
        >>> t.local_translation
        tensor([[ 0.0000,  0.0000,  0.7500],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.2000,  0.2000,  0.0000],
                [ 0.2000,  0.2000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [-0.2000,  0.2000,  0.0000],
                [-0.2000,  0.2000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [-0.2000, -0.2000,  0.0000],
                [-0.2000, -0.2000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.2000, -0.2000,  0.0000],
                [ 0.2000, -0.2000,  0.0000]])
        >>> t.parent_of('front_left_leg')
        'torso'
        >>> t.index('front_right_foot')
        6
        >>> t[2]
        'aux_1'
    """

    __example_mjcf_path__ = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "tests/ant.xml"
    )

    def __init__(self, node_names, parent_indices, local_translation):
        """
        :param node_names: a list of names for each tree node
        :type node_names: list[str]
        :param parent_indices: an int32-typed tensor that represents the edge to its parent.\
        -1 represents the root node
        :type parent_indices: Tensor
        :param local_translation: a 3d vector that gives local translation information
        :type local_translation: Tensor
        """
        ln, lp, ll = len(node_names), len(parent_indices), len(local_translation)
        assert len({ln, lp, ll}) == 1
        self._node_names = node_names
        self._parent_indices = parent_indices.long()
        self._local_translation = local_translation
        self._node_indices = {self.node_names[i]: i for i in range(len(self))}

    def __len__(self):
        """number of nodes in the skeleton tree"""
        return len(self.node_names)

    def __iter__(self):
        """iterator that iterate through the name of each node"""
        yield from self.node_names

    def __getitem__(self, item):
        """get the name of the node given the index"""
        return self.node_names[item]

    def __repr__(self):
        return (
            f"SkeletonTree(\n    node_names={self._indent(repr(self.node_names))},\n    parent_indices={self._indent(repr(self.parent_indices))},"
            f"\n    local_translation={self._indent(repr(self.local_translation))}\n)"
        )

    def _indent(self, s):
        return "\n    ".join(s.split("\n"))

    @property
    def node_names(self):
        return self._node_names

    @property
    def parent_indices(self):
        return self._parent_indices

    @property
    def local_translation(self):
        return self._local_translation

    @property
    def num_joints(self):
        """number of nodes in the skeleton tree"""
        return len(self)

    @classmethod
    def from_dict(cls, dict_repr, *args, **kwargs):
        return cls(
            list(map(str, dict_repr["node_names"])),
            TensorUtils.from_dict(dict_repr["parent_indices"], *args, **kwargs),
            TensorUtils.from_dict(dict_repr["local_translation"], *args, **kwargs),
        )

    def to_dict(self):
        return OrderedDict(
            [
                ("node_names", self.node_names),
                ("parent_indices", tensor_to_dict(self.parent_indices)),
                ("local_translation", tensor_to_dict(self.local_translation)),
            ]
        )

    @classmethod
    def from_mjcf(cls, path: str) -> "SkeletonTree":
        """
        Parses a mujoco xml scene description file and returns a Skeleton Tree.
        We use the model attribute at the root as the name of the tree.

        :param path:
        :type path: string
        :return: The skeleton tree constructed from the mjcf file
        :rtype: SkeletonTree
        """
        tree = ET.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")

        node_names = []
        parent_indices = []
        local_translation = []

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            curr_index = node_index
            node_index += 1
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
            return node_index

        _add_xml_node(xml_body_root, -1, 0)

        return cls(
            node_names,
            torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            torch.from_numpy(np.array(local_translation, dtype=np.float32)),
        )

    def parent_of(self, node_name):
        """get the name of the parent of the given node

        :param node_name: the name of the node
        :type node_name: string
        :rtype: string
        """
        return self[int(self.parent_indices[self.index(node_name)].item())]

    def index(self, node_name):
        """get the index of the node

        :param node_name: the name of the node
        :type node_name: string
        :rtype: int
        """
        return self._node_indices[node_name]

    def drop_nodes_by_names(
        self, node_names: list[str], pairwise_translation=None
    ) -> "SkeletonTree":
        new_length = len(self) - len(node_names)
        new_node_names = []
        new_local_translation = torch.zeros(new_length, 3, dtype=self.local_translation.dtype)
        new_parent_indices = torch.zeros(new_length, dtype=self.parent_indices.dtype)
        parent_indices = self.parent_indices.numpy()
        new_node_indices: dict = {}
        new_node_index = 0
        for node_index in range(len(self)):
            if self[node_index] in node_names:
                continue
            tb_node_index = parent_indices[node_index]
            if tb_node_index != -1:
                local_translation = self.local_translation[node_index, :]
                while tb_node_index != -1 and self[tb_node_index] in node_names:
                    local_translation += self.local_translation[tb_node_index, :]
                    tb_node_index = parent_indices[tb_node_index]
                assert tb_node_index != -1, "the root node cannot be dropped"

                if pairwise_translation is not None:
                    local_translation = pairwise_translation[tb_node_index, node_index, :]
            else:
                local_translation = self.local_translation[node_index, :]

            new_node_names.append(self[node_index])
            new_local_translation[new_node_index, :] = local_translation
            if tb_node_index == -1:
                new_parent_indices[new_node_index] = -1
            else:
                new_parent_indices[new_node_index] = new_node_indices[self[tb_node_index]]
            new_node_indices[self[node_index]] = new_node_index
            new_node_index += 1

        return SkeletonTree(new_node_names, new_parent_indices, new_local_translation)

    def keep_nodes_by_names(
        self, node_names: list[str], pairwise_translation=None
    ) -> "SkeletonTree":
        nodes_to_drop = list(filter(lambda x: x not in node_names, self))
        return self.drop_nodes_by_names(nodes_to_drop, pairwise_translation)


class SkeletonState(Serializable):
    """
    A skeleton state contains all the information needed to describe a static state of a skeleton.
    It requires a skeleton tree, local/global rotation at each joint and the root translation.

    Example:
        >>> t = SkeletonTree.from_mjcf(SkeletonTree.__example_mjcf_path__)
        >>> zero_pose = SkeletonState.zero_pose(t)
        >>> plot_skeleton_state(zero_pose)  # can be imported from `.visualization.base_interface`
        [plot of the ant at zero pose
        >>> local_rotation = zero_pose.local_rotation.clone()
        >>> local_rotation[2] = torch.tensor([0, 0, 1, 0])
        >>> new_pose = SkeletonState.from_rotation_and_root_translation(
        ...             skeleton_tree=t,
        ...             r=local_rotation,
        ...             t=zero_pose.root_translation,
        ...             is_local=True
        ...         )
        >>> new_pose.local_rotation
        tensor([[0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 1., 0., 0.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.]])
        >>> plot_skeleton_state(new_pose)  # you should be able to see one of ant's leg is bent
        [plot of the ant with the new pose
        >>> new_pose.global_rotation  # the local rotation is propagated to the global rotation at joint #3
        tensor([[0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.]])

    Global/Local Representation (cont. from the previous example)
        >>> new_pose.is_local
        True
        >>> new_pose.tensor  # this will return the local rotation followed by the root translation
        tensor([0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
                0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
                0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
                0.])
        >>> new_pose.tensor.shape  # 4 * 13 (joint rotation) + 3 (root translatio
        torch.Size([55])
        >>> new_pose.global_repr().is_local
        False
        >>> new_pose.global_repr().tensor  # this will return the global rotation followed by the root translation instead
        tensor([0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
                0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
                0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
                0.])
        >>> new_pose.global_repr().tensor.shape  # 4 * 13 (joint rotation) + 3 (root translation
        torch.Size([55])
    """

    def __init__(self, tensor_backend, skeleton_tree, is_local):
        self._skeleton_tree = skeleton_tree
        self._is_local = is_local
        self.tensor = tensor_backend.clone()

    def __len__(self):
        return self.tensor.shape[0]

    @property
    def rotation(self):
        if not hasattr(self, "_rotation"):
            self._rotation = self.tensor[..., : self.num_joints * 4].reshape(
                *(self.tensor.shape[:-1] + (self.num_joints, 4))
            )
        return self._rotation

    @property
    def _local_rotation(self):
        if self._is_local:
            return self.rotation
        else:
            return None

    @property
    def _global_rotation(self):
        if not self._is_local:
            return self.rotation
        else:
            return None

    @property
    def is_local(self):
        """is the rotation represented in local frame?

        :rtype: bool
        """
        return self._is_local

    @property
    def invariant_property(self):
        return {"skeleton_tree": self.skeleton_tree, "is_local": self.is_local}

    @property
    def num_joints(self):
        """number of joints in the skeleton tree

        :rtype: int
        """
        return self.skeleton_tree.num_joints

    @property
    def skeleton_tree(self):
        """skeleton tree

        :rtype: SkeletonTree
        """
        return self._skeleton_tree

    @property
    def root_translation(self):
        """root translation

        :rtype: Tensor
        """
        if not hasattr(self, "_root_translation"):
            self._root_translation = self.tensor[..., self.num_joints * 4 : self.num_joints * 4 + 3]
        return self._root_translation

    @property
    def global_transformation(self):
        """global transformation of each joint (transform from joint frame to global frame)"""
        # Forward Kinematics
        if not hasattr(self, "_global_transformation"):
            local_transformation = self.local_transformation
            global_transformation = []
            parent_indices = self.skeleton_tree.parent_indices.numpy()
            # global_transformation = local_transformation.identity_like()
            for node_index in range(len(self.skeleton_tree)):
                parent_index = parent_indices[node_index]
                if parent_index == -1:
                    global_transformation.append(local_transformation[..., node_index, :])
                else:
                    global_transformation.append(
                        transform_mul(
                            global_transformation[parent_index],
                            local_transformation[..., node_index, :],
                        )
                    )
            self._global_transformation = torch.stack(global_transformation, axis=-2)
        return self._global_transformation

    @property
    def global_rotation(self):
        """global rotation of each joint (rotation matrix to rotate from joint's F.O.R to global
        F.O.R)"""
        if self._global_rotation is None:
            if not hasattr(self, "_comp_global_rotation"):
                self._comp_global_rotation = transform_rotation(self.global_transformation)
            return self._comp_global_rotation
        else:
            return self._global_rotation

    @property
    def global_translation(self):
        """global translation of each joint"""
        if not hasattr(self, "_global_translation"):
            self._global_translation = transform_translation(self.global_transformation)
        return self._global_translation

    @property
    def global_translation_xy(self):
        """global translation in xy"""
        trans_xy_data = self.global_translation.zeros_like()
        trans_xy_data[..., 0:2] = self.global_translation[..., 0:2]
        return trans_xy_data

    @property
    def global_translation_xz(self):
        """global translation in xz"""
        trans_xz_data = self.global_translation.zeros_like()
        trans_xz_data[..., 0:1] = self.global_translation[..., 0:1]
        trans_xz_data[..., 2:3] = self.global_translation[..., 2:3]
        return trans_xz_data

    @property
    def local_rotation(self):
        """the rotation from child frame to parent frame given in the order of child nodes appeared
        in `.skeleton_tree.node_names`"""
        if self._local_rotation is None:
            if not hasattr(self, "_comp_local_rotation"):
                local_rotation = quat_identity_like(self.global_rotation)
                for node_index in range(len(self.skeleton_tree)):
                    parent_index = self.skeleton_tree.parent_indices[node_index]
                    if parent_index == -1:
                        local_rotation[..., node_index, :] = self.global_rotation[
                            ..., node_index, :
                        ]
                    else:
                        local_rotation[..., node_index, :] = quat_mul_norm(
                            quat_inverse(self.global_rotation[..., parent_index, :]),
                            self.global_rotation[..., node_index, :],
                        )
                self._comp_local_rotation = local_rotation
            return self._comp_local_rotation
        else:
            return self._local_rotation

    @property
    def local_transformation(self):
        """local translation + local rotation. It describes the transformation from child frame to
        parent frame given in the order of child nodes appeared in `.skeleton_tree.node_names`
        """
        if not hasattr(self, "_local_transformation"):
            self._local_transformation = transform_from_rotation_translation(
                r=self.local_rotation, t=self.local_translation
            )
        return self._local_transformation

    @property
    def local_translation(self):
        """local translation of the skeleton state. It is identical to the local translation in
        `.skeleton_tree.local_translation` except the root translation. The root translation is
        identical to `.root_translation`"""
        if not hasattr(self, "_local_translation"):
            broadcast_shape = (
                tuple(self.tensor.shape[:-1])
                + (len(self.skeleton_tree),)
                + tuple(self.skeleton_tree.local_translation.shape[-1:])
            )
            local_translation = self.skeleton_tree.local_translation.broadcast_to(
                *broadcast_shape
            ).clone()
            local_translation[..., 0, :] = self.root_translation
            self._local_translation = local_translation
        return self._local_translation

    # Root Properties
    @property
    def root_translation_xy(self):
        """root translation on xy"""
        if not hasattr(self, "_root_translation_xy"):
            self._root_translation_xy = self.global_translation_xy[..., 0, :]
        return self._root_translation_xy

    @property
    def global_root_rotation(self):
        """root rotation"""
        if not hasattr(self, "_global_root_rotation"):
            self._global_root_rotation = self.global_rotation[..., 0, :]
        return self._global_root_rotation

    @property
    def global_root_yaw_rotation(self):
        """root yaw rotation"""
        if not hasattr(self, "_global_root_yaw_rotation"):
            self._global_root_yaw_rotation = self.global_root_rotation.yaw_rotation()
        return self._global_root_yaw_rotation

    # Properties relative to root
    @property
    def local_translation_to_root(self):
        """The 3D translation from joint frame to the root frame."""
        if not hasattr(self, "_local_translation_to_root"):
            self._local_translation_to_root = (
                self.global_translation - self.root_translation.unsqueeze(-1)
            )
        return self._local_translation_to_root

    @property
    def local_rotation_to_root(self):
        """The 3D rotation from joint frame to the root frame. It is equivalent to
        The root_R_world * world_R_node"""
        return quat_inverse(self.global_root_rotation).unsqueeze(-1) * self.global_rotation

    def compute_forward_vector(
        self,
        left_shoulder_index,
        right_shoulder_index,
        left_hip_index,
        right_hip_index,
        gaussian_filter_width=20,
    ):
        """Computes forward vector based on cross product of the up vector with
        average of the right->left shoulder and hip vectors"""
        global_positions = self.global_translation
        # Perpendicular to the forward direction.
        # Uses the shoulders and hips to find this.
        side_direction = (
            global_positions[:, left_shoulder_index].numpy()
            - global_positions[:, right_shoulder_index].numpy()
            + global_positions[:, left_hip_index].numpy()
            - global_positions[:, right_hip_index].numpy()
        )
        side_direction = side_direction / np.sqrt((side_direction**2).sum(axis=-1))[..., np.newaxis]

        # Forward direction obtained by crossing with the up direction.
        forward_direction = np.cross(side_direction, np.array([[0, 1, 0]]))

        # Smooth the forward direction with a Gaussian.
        # Axis 0 is the time/frame axis.
        forward_direction = filters.gaussian_filter1d(
            forward_direction, gaussian_filter_width, axis=0, mode="nearest"
        )
        forward_direction = (
            forward_direction / np.sqrt((forward_direction**2).sum(axis=-1))[..., np.newaxis]
        )

        return torch.from_numpy(forward_direction)

    @staticmethod
    def _to_state_vector(rot, rt):
        state_shape = rot.shape[:-2]
        vr = rot.reshape(*(state_shape + (-1,)))
        vt = rt.broadcast_to(*state_shape + rt.shape[-1:]).reshape(*(state_shape + (-1,)))
        v = torch.cat([vr, vt], axis=-1)
        return v

    @classmethod
    def from_dict(
        cls: type["SkeletonState"], dict_repr: OrderedDict, *args, **kwargs
    ) -> "SkeletonState":
        rot = TensorUtils.from_dict(dict_repr["rotation"], *args, **kwargs)
        rt = TensorUtils.from_dict(dict_repr["root_translation"], *args, **kwargs)
        return cls(
            SkeletonState._to_state_vector(rot, rt),
            SkeletonTree.from_dict(dict_repr["skeleton_tree"], *args, **kwargs),
            dict_repr["is_local"],
        )

    def to_dict(self) -> OrderedDict:
        return OrderedDict(
            [
                ("rotation", tensor_to_dict(self.rotation)),
                ("root_translation", tensor_to_dict(self.root_translation)),
                ("skeleton_tree", self.skeleton_tree.to_dict()),
                ("is_local", self.is_local),
            ]
        )

    @classmethod
    def from_rotation_and_root_translation(cls, skeleton_tree, r, t, is_local=True):
        """
        Construct a skeleton state from rotation and root translation

        :param skeleton_tree: the skeleton tree
        :type skeleton_tree: SkeletonTree
        :param r: rotation (either global or local)
        :type r: Tensor
        :param t: root translation
        :type t: Tensor
        :param is_local: to indicate that whether the rotation is local or global
        :type is_local: bool, optional, default=True
        """
        assert r.dim() > 0, f"the rotation needs to have at least 1 dimension (dim = {r.dim})"
        state_vec = SkeletonState._to_state_vector(r, t)

        return cls(
            state_vec,
            skeleton_tree=skeleton_tree,
            is_local=is_local,
        )

    @classmethod
    def zero_pose(cls, skeleton_tree):
        """
        Construct a zero-pose skeleton state from the skeleton tree by assuming that all the local
        rotation is 0 and root translation is also 0.

        :param skeleton_tree: the skeleton tree as the rigid body
        :type skeleton_tree: SkeletonTree
        """
        return cls.from_rotation_and_root_translation(
            skeleton_tree=skeleton_tree,
            r=quat_identity([skeleton_tree.num_joints]),
            t=torch.zeros(3, dtype=skeleton_tree.local_translation.dtype),
            is_local=True,
        )

    def local_repr(self):
        """
        Convert the skeleton state into local representation. This will only affects the values of
        .tensor. If the skeleton state already has `is_local=True`. This method will do nothing.

        :rtype: SkeletonState
        """
        if self.is_local:
            return self
        return SkeletonState.from_rotation_and_root_translation(
            self.skeleton_tree,
            r=self.local_rotation,
            t=self.root_translation,
            is_local=True,
        )

    def global_repr(self):
        """
        Convert the skeleton state into global representation. This will only affects the values of
        .tensor. If the skeleton state already has `is_local=False`. This method will do nothing.

        :rtype: SkeletonState
        """
        if not self.is_local:
            return self
        return SkeletonState.from_rotation_and_root_translation(
            self.skeleton_tree,
            r=self.global_rotation,
            t=self.root_translation,
            is_local=False,
        )

    def _get_pairwise_average_translation(self):
        global_transform_inv = transform_inverse(self.global_transformation)
        p1 = global_transform_inv.unsqueeze(-2)
        p2 = self.global_transformation.unsqueeze(-3)

        pairwise_translation = (
            transform_translation(transform_mul(p1, p2))
            .reshape(-1, len(self.skeleton_tree), len(self.skeleton_tree), 3)
            .mean(axis=0)
        )
        return pairwise_translation

    def _transfer_to(self, new_skeleton_tree: SkeletonTree):
        old_indices = list(map(self.skeleton_tree.index, new_skeleton_tree))
        return SkeletonState.from_rotation_and_root_translation(
            new_skeleton_tree,
            r=self.global_rotation[..., old_indices, :],
            t=self.root_translation,
            is_local=False,
        )

    def drop_nodes_by_names(
        self, node_names: list[str], estimate_local_translation_from_states: bool = True
    ) -> "SkeletonState":
        """
        Drop a list of nodes from the skeleton and re-compute the local rotation to match the
        original joint position as much as possible.

        :param node_names: a list node names that specifies the nodes need to be dropped
        :type node_names: List of strings
        :param estimate_local_translation_from_states: the boolean indicator that specifies whether\
        or not to re-estimate the local translation from the states (avg.)
        :type estimate_local_translation_from_states: boolean
        :rtype: SkeletonState
        """
        if estimate_local_translation_from_states:
            pairwise_translation = self._get_pairwise_average_translation()
        else:
            pairwise_translation = None
        new_skeleton_tree = self.skeleton_tree.drop_nodes_by_names(node_names, pairwise_translation)
        return self._transfer_to(new_skeleton_tree)

    def keep_nodes_by_names(
        self, node_names: list[str], estimate_local_translation_from_states: bool = True
    ) -> "SkeletonState":
        """
        Keep a list of nodes and drop all other nodes from the skeleton and re-compute the local
        rotation to match the original joint position as much as possible.

        :param node_names: a list node names that specifies the nodes need to be dropped
        :type node_names: List of strings
        :param estimate_local_translation_from_states: the boolean indicator that specifies whether\
        or not to re-estimate the local translation from the states (avg.)
        :type estimate_local_translation_from_states: boolean
        :rtype: SkeletonState
        """
        return self.drop_nodes_by_names(
            list(filter(lambda x: (x not in node_names), self)),
            estimate_local_translation_from_states,
        )

    def _remapped_to(self, joint_mapping: dict[str, str], target_skeleton_tree: SkeletonTree):
        joint_mapping_inv = {target: source for source, target in joint_mapping.items()}
        reduced_target_skeleton_tree = target_skeleton_tree.keep_nodes_by_names(
            list(joint_mapping_inv)
        )
        n_joints = (
            len(joint_mapping),
            len(self.skeleton_tree),
            len(reduced_target_skeleton_tree),
        )
        assert (
            len(set(n_joints)) == 1
        ), "the joint mapping is not consistent with the skeleton trees"
        source_indices = [
            self.skeleton_tree.index(joint_mapping_inv[x]) for x in reduced_target_skeleton_tree
        ]
        target_local_rotation = self.local_rotation[..., source_indices, :]
        return SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=reduced_target_skeleton_tree,
            r=target_local_rotation,
            t=self.root_translation,
            is_local=True,
        )

    def retarget_to(
        self,
        joint_mapping: dict[str, str],
        source_tpose_local_rotation,
        source_tpose_root_translation: np.ndarray,
        target_skeleton_tree: SkeletonTree,
        target_tpose_local_rotation,
        target_tpose_root_translation: np.ndarray,
        rotation_to_target_skeleton,
        scale_to_target_skeleton: float,
        z_up: bool = True,
    ) -> "SkeletonState":
        """
        Retarget the skeleton state to a target skeleton tree. This is a naive retarget
        implementation with rough approximations. The function follows the procedures below.

        Steps:
            1. Drop the joints from the source (self) that do not belong to the joint mapping\
            with an implementation that is similar to "keep_nodes_by_names()" - take a\
            look at the function doc for more details (same for source_tpose)

            2. Rotate the source state and the source tpose by "rotation_to_target_skeleton"\
            to align the source with the target orientation

            3. Extract the root translation and normalize it to match the scale of the target\
            skeleton

            4. Extract the global rotation from source state relative to source tpose and\
            re-apply the relative rotation to the target tpose to construct the global\
            rotation after retargetting

            5. Combine the computed global rotation and the root translation from 3 and 4 to\
            complete the retargeting.

            6. Make feet on the ground (global translation z)

        :param joint_mapping: a dictionary of that maps the joint node from the source skeleton to \
        the target skeleton
        :type joint_mapping: dict[str, str]

        :param source_tpose_local_rotation: the local rotation of the source skeleton
        :type source_tpose_local_rotation: Tensor

        :param source_tpose_root_translation: the root translation of the source tpose
        :type source_tpose_root_translation: np.ndarray

        :param target_skeleton_tree: the target skeleton tree
        :type target_skeleton_tree: SkeletonTree

        :param target_tpose_local_rotation: the local rotation of the target skeleton
        :type target_tpose_local_rotation: Tensor

        :param target_tpose_root_translation: the root translation of the target tpose
        :type target_tpose_root_translation: Tensor

        :param rotation_to_target_skeleton: the rotation that needs to be applied to the source\
        skeleton to align with the target skeleton. Essentially the rotation is t_R_s, where t is\
        the frame of reference of the target skeleton and s is the frame of reference of the source\
        skeleton
        :type rotation_to_target_skeleton: Tensor
        :param scale_to_target_skeleton: the factor that needs to be multiplied from source\
        skeleton to target skeleton (unit in distance). For example, to go from `cm` to `m`, the \
        factor needs to be 0.01.
        :type scale_to_target_skeleton: float
        :rtype: SkeletonState
        """

        # STEP 0: Preprocess
        source_tpose = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=self.skeleton_tree,
            r=source_tpose_local_rotation,
            t=source_tpose_root_translation,
            is_local=True,
        )
        target_tpose = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=target_skeleton_tree,
            r=target_tpose_local_rotation,
            t=target_tpose_root_translation,
            is_local=True,
        )

        # STEP 1: Drop the irrelevant joints
        pairwise_translation = self._get_pairwise_average_translation()
        node_names = list(joint_mapping)
        new_skeleton_tree = self.skeleton_tree.keep_nodes_by_names(node_names, pairwise_translation)

        # TODO: combine the following steps before STEP 3
        source_tpose = source_tpose._transfer_to(new_skeleton_tree)
        source_state = self._transfer_to(new_skeleton_tree)

        source_tpose = source_tpose._remapped_to(joint_mapping, target_skeleton_tree)
        source_state = source_state._remapped_to(joint_mapping, target_skeleton_tree)

        # STEP 2: Rotate the source to align with the target
        new_local_rotation = source_tpose.local_rotation.clone()
        new_local_rotation[..., 0, :] = quat_mul_norm(
            rotation_to_target_skeleton, source_tpose.local_rotation[..., 0, :]
        )

        source_tpose = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=source_tpose.skeleton_tree,
            r=new_local_rotation,
            t=quat_rotate(rotation_to_target_skeleton, source_tpose.root_translation),
            is_local=True,
        )

        new_local_rotation = source_state.local_rotation.clone()
        new_local_rotation[..., 0, :] = quat_mul_norm(
            rotation_to_target_skeleton, source_state.local_rotation[..., 0, :]
        )
        source_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=source_state.skeleton_tree,
            r=new_local_rotation,
            t=quat_rotate(rotation_to_target_skeleton, source_state.root_translation),
            is_local=True,
        )

        # STEP 3: Normalize to match the target scale
        root_translation_diff = (
            source_state.root_translation - source_tpose.root_translation
        ) * scale_to_target_skeleton

        # STEP 4: the global rotation from source state relative to source tpose and
        # re-apply to the target
        current_skeleton_tree = source_state.skeleton_tree
        target_tpose_global_rotation = source_state.global_rotation[0, :].clone()
        for current_index, name in enumerate(current_skeleton_tree):
            if name in target_tpose.skeleton_tree:
                target_tpose_global_rotation[current_index, :] = target_tpose.global_rotation[
                    target_tpose.skeleton_tree.index(name), :
                ]

        global_rotation_diff = quat_mul_norm(
            source_state.global_rotation, quat_inverse(source_tpose.global_rotation)
        )
        new_global_rotation = quat_mul_norm(global_rotation_diff, target_tpose_global_rotation)

        # STEP 5: Putting 3 and 4 together
        current_skeleton_tree = source_state.skeleton_tree
        shape = source_state.global_rotation.shape[:-1]
        shape = shape[:-1] + target_tpose.global_rotation.shape[-2:-1]
        new_global_rotation_output = quat_identity(shape)
        for current_index, name in enumerate(target_skeleton_tree):
            while name not in current_skeleton_tree:
                name = target_skeleton_tree.parent_of(name)
            parent_index = current_skeleton_tree.index(name)
            new_global_rotation_output[:, current_index, :] = new_global_rotation[
                :, parent_index, :
            ]

        source_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=target_skeleton_tree,
            r=new_global_rotation_output,
            t=target_tpose.root_translation + root_translation_diff,
            is_local=False,
        ).local_repr()

        return source_state

    def retarget_to_by_tpose(
        self,
        joint_mapping: dict[str, str],
        source_tpose: "SkeletonState",
        target_tpose: "SkeletonState",
        rotation_to_target_skeleton,
        scale_to_target_skeleton: float,
    ) -> "SkeletonState":
        """
        Retarget the skeleton state to a target skeleton tree. This is a naive retarget
        implementation with rough approximations. See the method `retarget_to()` for more information

        :param joint_mapping: a dictionary of that maps the joint node from the source skeleton to \
        the target skeleton
        :type joint_mapping: dict[str, str]

        :param source_tpose: t-pose of the source skeleton
        :type source_tpose: SkeletonState

        :param target_tpose: t-pose of the target skeleton
        :type target_tpose: SkeletonState

        :param rotation_to_target_skeleton: the rotation that needs to be applied to the source\
        skeleton to align with the target skeleton. Essentially the rotation is t_R_s, where t is\
        the frame of reference of the target skeleton and s is the frame of reference of the source\
        skeleton
        :type rotation_to_target_skeleton: Tensor
        :param scale_to_target_skeleton: the factor that needs to be multiplied from source\
        skeleton to target skeleton (unit in distance). For example, to go from `cm` to `m`, the \
        factor needs to be 0.01.
        :type scale_to_target_skeleton: float
        :rtype: SkeletonState
        """
        assert (
            len(source_tpose.shape) == 0 and len(target_tpose.shape) == 0
        ), "the retargeting script currently doesn't support vectorized operations"
        return self.retarget_to(
            joint_mapping,
            source_tpose.local_rotation,
            source_tpose.root_translation,
            target_tpose.skeleton_tree,
            target_tpose.local_rotation,
            target_tpose.root_translation,
            rotation_to_target_skeleton,
            scale_to_target_skeleton,
        )


class SkeletonMotion(SkeletonState):

    def __init__(self, tensor_backend, skeleton_tree, is_local, fps, *args, **kwargs):
        self._fps = fps
        super().__init__(tensor_backend, skeleton_tree, is_local, *args, **kwargs)

    def clone(self):
        return SkeletonMotion(self.tensor.clone(), self.skeleton_tree, self._is_local, self._fps)

    @property
    def invariant_property(self):
        return {
            "skeleton_tree": self.skeleton_tree,
            "is_local": self.is_local,
            "fps": self.fps,
        }

    @property
    def global_velocity(self):
        """global velocity"""
        curr_index = self.num_joints * 4 + 3
        return self.tensor[..., curr_index : curr_index + self.num_joints * 3].reshape(
            *(self.tensor.shape[:-1] + (self.num_joints, 3))
        )

    @property
    def global_angular_velocity(self):
        """global angular velocity"""
        curr_index = self.num_joints * 7 + 3
        return self.tensor[..., curr_index : curr_index + self.num_joints * 3].reshape(
            *(self.tensor.shape[:-1] + (self.num_joints, 3))
        )

    @property
    def fps(self):
        """number of frames per second"""
        return self._fps

    @property
    def time_delta(self):
        """time between two adjacent frames"""
        return 1.0 / self.fps

    @property
    def global_root_velocity(self):
        """global root velocity"""
        return self.global_velocity[..., 0, :]

    @property
    def global_root_angular_velocity(self):
        """global root angular velocity"""
        return self.global_angular_velocity[..., 0, :]

    @classmethod
    def from_state_vector_and_velocity(
        cls,
        skeleton_tree,
        state_vector,
        global_velocity,
        global_angular_velocity,
        is_local,
        fps,
    ):
        """
        Construct a skeleton motion from a skeleton state vector, global velocity and angular
        velocity at each joint.

        :param skeleton_tree: the skeleton tree that the motion is based on
        :type skeleton_tree: SkeletonTree
        :param state_vector: the state vector from the skeleton state by `.tensor`
        :type state_vector: Tensor
        :param global_velocity: the global velocity at each joint
        :type global_velocity: Tensor
        :param global_angular_velocity: the global angular velocity at each joint
        :type global_angular_velocity: Tensor
        :param is_local: if the rotation ins the state vector is given in local frame
        :type is_local: boolean
        :param fps: number of frames per second
        :type fps: int

        :rtype: SkeletonMotion
        """
        state_shape = state_vector.shape[:-1]
        v = global_velocity.reshape(*(state_shape + (-1,)))
        av = global_angular_velocity.reshape(*(state_shape + (-1,)))
        new_state_vector = torch.cat([state_vector, v, av], axis=-1)
        return cls(
            new_state_vector,
            skeleton_tree=skeleton_tree,
            is_local=is_local,
            fps=fps,
        )

    @classmethod
    def from_skeleton_state(cls: type["SkeletonMotion"], skeleton_state: SkeletonState, fps: int):
        """
        Construct a skeleton motion from a skeleton state. The velocities are estimated using second
        order guassian filter along the last axis. The skeleton state must have at least .dim >= 1

        :param skeleton_state: the skeleton state that the motion is based on
        :type skeleton_state: SkeletonState
        :param fps: number of frames per second
        :type fps: int

        :rtype: SkeletonMotion
        """
        assert (
            type(skeleton_state) == SkeletonState
        ), f"expected type of {SkeletonState}, got {type(skeleton_state)}"
        global_velocity = SkeletonMotion._compute_velocity(
            p=skeleton_state.global_translation, time_delta=1 / fps
        )
        global_angular_velocity = SkeletonMotion._compute_angular_velocity(
            r=skeleton_state.global_rotation, time_delta=1 / fps
        )
        return cls.from_state_vector_and_velocity(
            skeleton_tree=skeleton_state.skeleton_tree,
            state_vector=skeleton_state.tensor,
            global_velocity=global_velocity,
            global_angular_velocity=global_angular_velocity,
            is_local=skeleton_state.is_local,
            fps=fps,
        )

    @staticmethod
    def _to_state_vector(rot, rt, vel, avel):
        state_shape = rot.shape[:-2]
        skeleton_state_v = SkeletonState._to_state_vector(rot, rt)
        v = vel.reshape(*(state_shape + (-1,)))
        av = avel.reshape(*(state_shape + (-1,)))
        skeleton_motion_v = torch.cat([skeleton_state_v, v, av], axis=-1)
        return skeleton_motion_v

    @classmethod
    def from_dict(
        cls: type["SkeletonMotion"], dict_repr: OrderedDict, *args, **kwargs
    ) -> "SkeletonMotion":
        rot = TensorUtils.from_dict(dict_repr["rotation"], *args, **kwargs)
        rt = TensorUtils.from_dict(dict_repr["root_translation"], *args, **kwargs)
        vel = TensorUtils.from_dict(dict_repr["global_velocity"], *args, **kwargs)
        avel = TensorUtils.from_dict(dict_repr["global_angular_velocity"], *args, **kwargs)
        return cls(
            SkeletonMotion._to_state_vector(rot, rt, vel, avel),
            skeleton_tree=SkeletonTree.from_dict(dict_repr["skeleton_tree"], *args, **kwargs),
            is_local=dict_repr["is_local"],
            fps=dict_repr["fps"],
        )

    def to_dict(self) -> OrderedDict:
        return OrderedDict(
            [
                ("rotation", tensor_to_dict(self.rotation)),
                ("root_translation", tensor_to_dict(self.root_translation)),
                ("global_velocity", tensor_to_dict(self.global_velocity)),
                (
                    "global_angular_velocity",
                    tensor_to_dict(self.global_angular_velocity),
                ),
                ("skeleton_tree", self.skeleton_tree.to_dict()),
                ("is_local", self.is_local),
                ("fps", self.fps),
            ]
        )

    # @classmethod
    # def from_fbx(
    #     cls: type["SkeletonMotion"],
    #     fbx_file_path,
    #     fbx_configs,
    #     skeleton_tree=None,
    #     is_local=True,
    #     fps=120,
    #     root_joint="",
    #     root_trans_index=0,
    #     *args,
    #     **kwargs,
    # ) -> "SkeletonMotion":
    #     """
    #     Construct a skeleton motion from a fbx file (TODO - generalize this). If the skeleton tree
    #     is not given, it will use the first frame of the mocap to construct the skeleton tree.

    #     :param fbx_file_path: the path of the fbx file
    #     :type fbx_file_path: string
    #     :param fbx_configs: the configuration in terms of {"tmp_path": ..., "fbx_py27_path": ...}
    #     :type fbx_configs: dict
    #     :param skeleton_tree: the optional skeleton tree that the rotation will be applied to
    #     :type skeleton_tree: SkeletonTree, optional
    #     :param is_local: the state vector uses local or global rotation as the representation
    #     :type is_local: bool, optional, default=True
    #     :rtype: SkeletonMotion
    #     """
    #     joint_names, joint_parents, transforms, fps = fbx_to_array(fbx_file_path, fbx_configs, root_joint, fps)
    #     # swap the last two axis to match the convention
    #     local_transform = euclidean_to_transform(transformation_matrix=torch.from_numpy(np.swapaxes(np.array(transforms), -1, -2),).float())
    #     local_rotation = transform_rotation(local_transform)
    #     root_translation = transform_translation(local_transform)[..., root_trans_index, :]
    #     joint_parents = torch.from_numpy(np.array(joint_parents)).int()

    #     if skeleton_tree is None:
    #         local_translation = transform_translation(local_transform).reshape(-1, len(joint_parents), 3)[0]
    #         skeleton_tree = SkeletonTree(joint_names, joint_parents, local_translation)
    #     skeleton_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, r=local_rotation, t=root_translation, is_local=True)
    #     if not is_local:
    #         skeleton_state = skeleton_state.global_repr()
    #     return cls.from_skeleton_state(skeleton_state=skeleton_state, fps=fps)

    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        if guassian_filter:
            velocity = torch.from_numpy(
                filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")
            ).to(p)
        else:
            velocity = torch.from_numpy(velocity).to(p)

        return velocity

    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        # assume the second last dimension is the time axis
        diff_quat_data = quat_identity_like(r).to(r)
        diff_quat_data[..., :-1, :, :] = quat_mul_norm(
            r[..., 1:, :, :], quat_inverse(r[..., :-1, :, :])
        )
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = torch.from_numpy(
                filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"),
            )
        return angular_velocity

    def crop(self, start: int, end: int, fps: int | None = None):
        """
        Crop the motion along its last axis. This is equivalent to performing a slicing on the
        object with [..., start: end: skip_every] where skip_every = old_fps / fps. Note that the
        new fps provided must be a factor of the original fps.

        :param start: the beginning frame index
        :type start: int
        :param end: the ending frame index
        :type end: int
        :param fps: number of frames per second in the output (if not given the original fps will be used)
        :type fps: int, optional
        :rtype: SkeletonMotion
        """
        if fps is None:
            new_fps = int(self.fps)
            old_fps = int(self.fps)
        else:
            new_fps = int(fps)
            old_fps = int(self.fps)
            assert old_fps % fps == 0, (
                "the resampling doesn't support fps with non-integer division "
                f"from the original fps: {old_fps} => {fps}"
            )
        skip_every = old_fps // new_fps
        s = slice(start, end, skip_every)
        z = self[..., s]

        rot = z.local_rotation if z.is_local else z.global_rotation
        rt = z.root_translation
        vel = z.global_velocity
        avel = z.global_angular_velocity
        return SkeletonMotion(
            SkeletonMotion._to_state_vector(rot, rt, vel, avel),
            skeleton_tree=z.skeleton_tree,
            is_local=z.is_local,
            fps=new_fps,
        )

    def retarget_to(
        self,
        joint_mapping: dict[str, str],
        source_tpose_local_rotation,
        source_tpose_root_translation: np.ndarray,
        target_skeleton_tree: "SkeletonTree",
        target_tpose_local_rotation,
        target_tpose_root_translation: np.ndarray,
        rotation_to_target_skeleton,
        scale_to_target_skeleton: float,
        z_up: bool = True,
    ) -> "SkeletonMotion":
        """
        Same as the one in :class:`SkeletonState`. This method discards all velocity information before
        retargeting and re-estimate the velocity after the retargeting. The same fps is used in the
        new retargetted motion.

        :param joint_mapping: a dictionary of that maps the joint node from the source skeleton to \
        the target skeleton
        :type joint_mapping: dict[str, str]

        :param source_tpose_local_rotation: the local rotation of the source skeleton
        :type source_tpose_local_rotation: Tensor

        :param source_tpose_root_translation: the root translation of the source tpose
        :type source_tpose_root_translation: np.ndarray

        :param target_skeleton_tree: the target skeleton tree
        :type target_skeleton_tree: SkeletonTree

        :param target_tpose_local_rotation: the local rotation of the target skeleton
        :type target_tpose_local_rotation: Tensor

        :param target_tpose_root_translation: the root translation of the target tpose
        :type target_tpose_root_translation: Tensor

        :param rotation_to_target_skeleton: the rotation that needs to be applied to the source\
        skeleton to align with the target skeleton. Essentially the rotation is t_R_s, where t is\
        the frame of reference of the target skeleton and s is the frame of reference of the source\
        skeleton
        :type rotation_to_target_skeleton: Tensor
        :param scale_to_target_skeleton: the factor that needs to be multiplied from source\
        skeleton to target skeleton (unit in distance). For example, to go from `cm` to `m`, the \
        factor needs to be 0.01.
        :type scale_to_target_skeleton: float
        :rtype: SkeletonMotion
        """
        return SkeletonMotion.from_skeleton_state(
            super().retarget_to(
                joint_mapping,
                source_tpose_local_rotation,
                source_tpose_root_translation,
                target_skeleton_tree,
                target_tpose_local_rotation,
                target_tpose_root_translation,
                rotation_to_target_skeleton,
                scale_to_target_skeleton,
                z_up,
            ),
            self.fps,
        )

    def retarget_to_by_tpose(
        self,
        joint_mapping: dict[str, str],
        source_tpose: "SkeletonState",
        target_tpose: "SkeletonState",
        rotation_to_target_skeleton,
        scale_to_target_skeleton: float,
        z_up: bool = True,
    ) -> "SkeletonMotion":
        """
        Same as the one in :class:`SkeletonState`. This method discards all velocity information before
        retargeting and re-estimate the velocity after the retargeting. The same fps is used in the
        new retargetted motion.

        :param joint_mapping: a dictionary of that maps the joint node from the source skeleton to \
        the target skeleton
        :type joint_mapping: dict[str, str]

        :param source_tpose: t-pose of the source skeleton
        :type source_tpose: SkeletonState

        :param target_tpose: t-pose of the target skeleton
        :type target_tpose: SkeletonState

        :param rotation_to_target_skeleton: the rotation that needs to be applied to the source\
        skeleton to align with the target skeleton. Essentially the rotation is t_R_s, where t is\
        the frame of reference of the target skeleton and s is the frame of reference of the source\
        skeleton
        :type rotation_to_target_skeleton: Tensor
        :param scale_to_target_skeleton: the factor that needs to be multiplied from source\
        skeleton to target skeleton (unit in distance). For example, to go from `cm` to `m`, the \
        factor needs to be 0.01.
        :type scale_to_target_skeleton: float
        :rtype: SkeletonMotion
        """
        return self.retarget_to(
            joint_mapping,
            source_tpose.local_rotation,
            source_tpose.root_translation,
            target_tpose.skeleton_tree,
            target_tpose.local_rotation,
            target_tpose.root_translation,
            rotation_to_target_skeleton,
            scale_to_target_skeleton,
            z_up,
        )
