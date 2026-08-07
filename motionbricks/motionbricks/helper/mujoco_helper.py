import numpy as np
import torch as t
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET
import os
import torch.nn.functional as F
from typing import List
from motionbricks.motionlib.core.motion_reps import MotionRepBase
from motionbricks.motionlib.core.motion_reps.tools.changing_t_pose import get_global_offset
from motionbricks.motionlib.core.utils.rotations import cont6d_to_matrix, quaternion_to_matrix
from motionbricks.motionlib.core.utils.torch_utils import compute_idx_levels
from motionbricks.motionlib.core.skeletons import SkeletonBase


# using this matrix_to_quaternion instead of the one in motionbricks.motionlib.core.utils.rotations
# to avoid some tensorrt issues
from motionbricks.geometry.quaternions import matrix_to_quaternion

# redefining this function in motionbricks.motionlib.core.motion_rep.tools.changing_t_pose as
# tensorrt complains about capital letters in t.einsum()
def global_mats_to_local_mats(global_rot_mats: t.Tensor, skeleton: SkeletonBase):
    # obtain back the local rotations from the global rotations
    parent_rot_mats = global_rot_mats[..., skeleton.joint_parents, :, :]
    parent_rot_mats[..., skeleton.root_idx, :, :] = t.eye(3)  # the root joint
    local_rot_mats = t.einsum(
        "... x n m, ... x n o -> ... x m o",
        parent_rot_mats,  # taken as the inverse/transpose in einsum
        global_rot_mats,
    )
    return local_rot_mats

# this is the same as motionbricks.motionlib.core.utils.torch_utils.transform_mat, but without the __jit_traced__ decorator
# which is not supported by onnx / tensorrt export
def transform_mat(R: t.Tensor, t: t.Tensor):
    """Creates a batch of transformation matrices.

    Args:
        - R: Bx3x3 array of a batch of rotation matrices
        - t: Bx3x1 array of a batch of translation vectors
    Returns:
        - T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1.0)], dim=2)

# this is the same as motionbricks.motionlib.core.utils.torch_utils.forward_kinematics, but without the __jit_traced__ decorator
# which is not supported by onnx / tensorrt export
def forward_kinematics(
    rot_mats,
    joints,
    parents: t.Tensor,
    idx_levs: List[t.Tensor],
    root_idx: int,
):
    """Perform forward kinematics to compute posed joints and global rotation matrices.

    Args:
        rot_mats: Local rotation matrices for each joint: (B, J, 3, 3)
        joints: Initial joint positions: (B, J, 3)
        parents: Tensor indicating the parent of each joint: (J,)
        idx_levs: List of tensors containing indices for each level in the kinematic tree
        root_idx (int): index of the root
    Returns:
        Posed joints: (B, J, 3)
        Global rotation matrices: (B, J, 3, 3)
    """

    # Add an extra dimension to joints
    joints = torch.unsqueeze(joints, dim=-1)

    # Compute relative joint positions
    rel_joints = joints.clone()

    mask_no_root = torch.ones(joints.shape[1], dtype=torch.bool)
    mask_no_root[root_idx] = False
    mask_no_root_inds = mask_no_root.nonzero().squeeze().tolist()
    rel_joints[:, mask_no_root_inds] -= joints[:, parents[mask_no_root_inds]].clone()

    # Compute initial transformation matrices
    # (B, J + 1, 4, 4)
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)
    ).reshape(-1, joints.shape[1], 4, 4)

    # Initialize the root transformation matrices
    transforms = torch.zeros_like(transforms_mat)
    transforms[:, root_idx] = transforms_mat[:, root_idx]

    # Compute global transformations level by level
    for indices in idx_levs:
        curr_res = torch.matmul(
            transforms[:, parents[indices]], transforms_mat[:, indices]
        )
        transforms[:, indices] = curr_res

    # Extract posed joint positions from the transformation matrices
    posed_joints = transforms[:, :, :3, 3]

    # Extract global rotation matrices from the transformation matrices
    global_rot_mat = transforms[:, :, :3, :3]

    return posed_joints, global_rot_mat


_global_converter = None

def get_mujoco_converter(motion_rep: MotionRepBase, xml_path: str = "assets/skeletons/g1/g1.xml"):
    """Get a cached mujoco converter instance for efficient reuse."""
    global _global_converter
    if _global_converter is None:
        _global_converter = mujoco_qpos_converter(motion_rep, xml_path)
    return _global_converter


def motion_feature_to_mujoco_qpos(motion_features: t.Tensor, motion_rep: MotionRepBase,
                               is_normalized: bool = False, use_fast_converter: bool = True):
    """Convert motion features to mujoco qpos format.

    Args:
        motion_features: Motion features
        motion_rep: MotionRepBase object
        is_normalized: Whether input features are normalized
        use_fast_converter: Whether to use the optimized converter (recommended)
    """
    assert use_fast_converter, "only fast converter is supported for now"
    converter = get_mujoco_converter(motion_rep)
    qpos = converter.convert_motion_features_to_mujoco_qpos(motion_features, motion_rep,
                                                         is_normalized, root_quat_w_first=True)
    return qpos
class mujoco_qpos_converter(nn.Module):
    """Fast batch converter from motion features to mujoco qpos with precomputed transforms.

    In mujoco, the coordination is z up and x forward, right handed

    features (30 joints):
        root (pelvis, 7 = translation + rotation) + 29 dof joints (29)

    The motion feature coordinate system is y up and z forward, right handed
    features (34 joints):
        root (pelvis) + (34 - 1) joints; among these joints, 4 are end-effector joints.
    """

    def __init__(self, motion_rep: MotionRepBase, xml_path: str = "assets/skeletons/g1/g1.xml",
                 dead_joint_rotation_scheme: str = "dummy"):
        """Initialize converter with precomputed transforms.

        Args:
            xml_path: Path to the mujoco XML file containing joint definitions
            dead_joint_rotation_scheme: Scheme for handling dead joints (end-effectors joints);
            if "dummy", the dead joints's global rotations are set to identity matrix;
            if "parent", the dead joints's global rotations are set to the parent's rotation.
        """
        super(mujoco_qpos_converter, self).__init__()
        self.xml_path = xml_path
        self.motion_rep = motion_rep
        self._prepare_transforms()
        self._subtree_joints = {}
        self._dead_joint_rotation_scheme = dead_joint_rotation_scheme

    def _prepare_transforms(self):
        """Precompute all necessary transforms for efficient batch processing."""
        # Define coordinate transformations between mujoco and motion space
        # 1) R_zup_to_yup: rotation around x-axis by -90 degrees
        # 2) x_forward_to_y_forward: rotation around z-axis by -90 degrees
        # Combined transformation matrix: mujoco_to_motion = R_zup_to_yup * x_forward_to_y_forward
        self.mujoco_to_motion_matrix = t.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], dtype=t.float32)
        self.motion_to_mujoco_matrix = self.mujoco_to_motion_matrix.T  # Inverse transformation: motion_to_mujoco

        # Parse XML once and extract joint information
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        xml_classes = [x for x in tree.findall('.//default') if "class" in x.attrib]
        joint_axes = dict()
        for xml_class in xml_classes:
            j = xml_class.findall("joint")
            if j:
                joint_axes[xml_class.get("class")] = j[0].get("axis")

        mujoco_hinge_joints = root.find("worldbody").findall(".//joint")  # skip the base joint
        self._mujoco_joint_axis_values_motion_space = \
            t.zeros((len(mujoco_hinge_joints), 3), dtype=t.float32)  # mujoco order but motion space
        self._mujoco_joint_axis_values_mujoco_space = \
            t.zeros((len(mujoco_hinge_joints), 3), dtype=t.float32)  # mujoco order but mujoco space

        # for the below indices, mujoco_indices_to_motion_indices does not include mujoco root (30 - 1 = 29 elements),
        # while motion_indices_to_mujoco_indices includes the motion root (32 elements).
        self._mujoco_indices_to_motion_indices = t.zeros((len(mujoco_hinge_joints),), dtype=t.int32)
        self._motion_indices_to_mujoco_indices = \
            t.ones((self.motion_rep.skeleton.nbjoints,), dtype=t.int32) * -1  # -1 means not in the mujoco skeleton

        self._nb_joints_mujoco = len(mujoco_hinge_joints) + 1
        self._nb_joints_motion = self.motion_rep.skeleton.nbjoints
        self._mujoco_joint_including_root_parent_list = t.full((len(mujoco_hinge_joints) + 1,), -1, dtype=t.int32)
        self._mujoco_joint_including_root_list = ['pelvis_skel']

        for joint_id_in_csv, joint in enumerate(mujoco_hinge_joints):
            joint_name_in_skeleton = joint.get("name").replace("_joint", "_skel")
            joint_parent_name_in_skeleton = self.motion_rep.skeleton.bone_parents[joint_name_in_skeleton]

            self._mujoco_joint_including_root_list.append(joint_name_in_skeleton)
            self._mujoco_joint_including_root_parent_list[joint_id_in_csv + 1] = \
                self._mujoco_joint_including_root_list.index(joint_parent_name_in_skeleton)

            joint_idx_in_skeleton = self.motion_rep.skeleton.bone_order_names.index(joint_name_in_skeleton)
            axis_values = [
                float(x) for x in
                (
                    joint.get("axis") or
                    joint_axes[joint.get("class")]
                ).split(" ")
            ]

            # the mapped axis in motion skeleton space is calculated as motion_axis = mujoco_to_motion.apply(axis_values)
            # [1, 0, 0] -> [0, 0, 1]; [0, 1, 0] -> [1, 0, 0]; [0, 0, 1] -> [0, 1, 0]
            mujoco_joint_axis_mapping_motion_space = \
                [t.tensor([0, 0, 1]), t.tensor([1, 0, 0]), t.tensor([0, 1, 0])][np.argmax(axis_values)]

            self._mujoco_joint_axis_values_motion_space[joint_id_in_csv] = mujoco_joint_axis_mapping_motion_space
            self._mujoco_joint_axis_values_mujoco_space[joint_id_in_csv] = t.tensor(axis_values)

            self._mujoco_indices_to_motion_indices[joint_id_in_csv] = joint_idx_in_skeleton
            self._motion_indices_to_mujoco_indices[joint_idx_in_skeleton] = joint_id_in_csv + 1  # +1 for the root
        self._motion_indices_to_mujoco_indices[0] = 0  # the root joint mapping

        # load the offset matrices from the xml
        from scipy.spatial.transform import Rotation

        R_zup_to_yup = Rotation.from_euler("x", -90, degrees=True)
        x_forward_to_y_forward = Rotation.from_euler("z", -90, degrees=True)
        mujoco_to_motion = R_zup_to_yup * x_forward_to_y_forward


        self._rot_offsets_q2t = t.zeros(len(self._motion_indices_to_mujoco_indices), 3, 3, dtype=t.float32)
        self._rot_offsets_q2t[...] = t.eye(3)[None]

        self._rot_offsets_f2q = t.zeros(len(self._motion_indices_to_mujoco_indices), 3, 3, dtype=t.float32)
        self._rot_offsets_f2q[...] = t.eye(3)[None]
        parent_map = {child: parent for parent in root.iter() for child in parent}
        for i, joint in enumerate(mujoco_hinge_joints):
            body = parent_map[joint]
            if "quat" in body.attrib:
                rot = Rotation.from_quat(
                    [float(x) for x in body.get("quat").strip().split(" ")],
                    scalar_first=True
                )
                idx = self._mujoco_indices_to_motion_indices[i]
                self._rot_offsets_q2t[idx] = torch.from_numpy(rot.as_matrix())
                rot = mujoco_to_motion * rot * mujoco_to_motion.inv()
                self._rot_offsets_f2q[idx] = torch.from_numpy(rot.as_matrix().T)

        self._capture_neutral_joints_mujoco = t.zeros(len(self._mujoco_indices_to_motion_indices) + 1, 3, dtype=t.float32)
        for i, joint in enumerate(mujoco_hinge_joints):
            body = parent_map[joint]
            pos = body.get("pos")
            if pos is not None:
                self._capture_neutral_joints_mujoco[i+1] = t.tensor([float(x) for x in pos.strip().split(" ")])

        self._mujoco_joint_idx_levs = compute_idx_levels(self._mujoco_joint_including_root_parent_list)
        for indices in self._mujoco_joint_idx_levs:
            self._capture_neutral_joints_mujoco[indices] += self._capture_neutral_joints_mujoco[self._mujoco_joint_including_root_parent_list[indices]]


    def convert_motion_features_to_mujoco_qpos(self, motion_features: t.Tensor, motion_rep: MotionRepBase,
                                            is_normalized: bool = True, root_quat_w_first: bool = False) -> t.Tensor:
        """Fast batch conversion from motion features to mujoco qpos format.

        Args:
            motion_features: [batch, numFrames, motion_dim] Motion features
            motion_rep: MotionRepBase object for the motion representation
            is_normalized: Whether the input features are normalized

        Returns:
            torch.Tensor of shape [batch, numFrames, 36] containing mujoco qpos data:
            - root_trans (3) + root_quat (4) + joint_dofs (29) = 36 columns
        """
        # Get joint output from motion representation
        batch_size, num_frames, nb_joints = motion_features.shape[0], motion_features.shape[1], motion_rep.skeleton.nbjoints
        motion_rep = motion_rep.to(motion_features.device)
        if is_normalized:
            motion_features = motion_rep.unnormalize(motion_features)
        root_translation, root_rot_quat = motion_rep.compute_root_pos_and_rot(motion_features)

        global_joints_ric_rot = motion_rep.slice(motion_features, "global_rot_data")
        global_rot = cont6d_to_matrix(global_joints_ric_rot.view([batch_size, num_frames, nb_joints, 6]))

        local_joint_rot = global_mats_to_local_mats(global_rot, motion_rep.skeleton)
        local_joint_rot = t.matmul(self._rot_offsets_f2q.to(motion_features.device), local_joint_rot)

        batch_size, num_frames = root_translation.shape[0], root_translation.shape[1]
        device, dtype = root_translation.device, root_translation.dtype

        # Move precomputed matrices to the same device/dtype
        motion_to_mujoco_matrix = self.motion_to_mujoco_matrix.to(device=device, dtype=dtype)

        # Initialize output tensor: [batch, numFrames, 36]
        qpos = t.zeros((batch_size, num_frames, 36), dtype=dtype, device=device)

        # Convert root translation: apply coordinate transformation
        root_translation_mujoco = t.matmul(motion_to_mujoco_matrix[None, None, ...],
                                           root_translation[..., None])
        qpos[:, :, :3] = root_translation_mujoco.view(batch_size, num_frames, 3)

        # Convert root rotation: apply coordinate transformation to rotation matrix
        root_rot = local_joint_rot[:, :, 0, :]  # [batch, numFrames, 3, 3]

        # Apply coordinate transformation: R_mujoco = motion_to_mujoco * R_motion * motion_to_mujoco^T
        mujoco_to_motion_matrix = motion_to_mujoco_matrix.T
        root_rot_mujoco = t.matmul(t.matmul(motion_to_mujoco_matrix[None, None, ...], root_rot),
                                   mujoco_to_motion_matrix[None, None, ...])
        root_rot_quat = matrix_to_quaternion(root_rot_mujoco)  # [w, x, y, z]
        if root_quat_w_first:
            qpos[:, :, 3: 7] = root_rot_quat[:, :, [0, 1, 2, 3]]  # [w, x, y, z]
        else:
            qpos[:, :, 3: 7] = root_rot_quat[:, :, [1, 2, 3, 0]]  # [w, x, y, z] -> [x, y, z, w]

        # Convert joint DOFs using precomputed mappings
        joint_rot_mujoco = \
            local_joint_rot[:, :, self._mujoco_indices_to_motion_indices, :]  # mujoco joint order but motion feature space
        x_joint_dof = t.atan2(joint_rot_mujoco[..., 2, 1], joint_rot_mujoco[..., 2, 2])
        y_joint_dof = t.atan2(joint_rot_mujoco[..., 0, 2], joint_rot_mujoco[..., 0, 0])
        z_joint_dof = t.atan2(joint_rot_mujoco[..., 1, 0], joint_rot_mujoco[..., 1, 1])
        xyz_joint_dofs = t.stack([x_joint_dof, y_joint_dof, z_joint_dof], dim=-1)
        joint_dofs = \
            (xyz_joint_dofs * self._mujoco_joint_axis_values_motion_space[None, None, :, :].to(device)).sum(dim=-1)
        qpos[:, :, 7:] = joint_dofs

        return qpos

    def convert_mujoco_qpos_to_mujoco_transforms(self, mujoco_qpos: t.Tensor) -> t.Tensor:
        """ @brief: the inverse process of convert_motion_features_to_mujoco_qpos """
        raise NotImplementedError("Not implemented yet")

    def convert_mujoco_qpos_to_motion_transforms(self, mujoco_qpos: t.Tensor) -> t.Tensor:
        """ @brief: the inverse process of convert_motion_features_to_mujoco_qpos """
        batch_size, num_frames = mujoco_qpos.shape[:2]
        device, dtype = mujoco_qpos.device, mujoco_qpos.dtype

        # the obtain the root (pelvis) information
        root_translation_mujoco = mujoco_qpos[:, :, :3]
        root_quat_mujoco = mujoco_qpos[:, :, 3: 7]  # [w, x, y, z]
        root_rotation_mujoco = quaternion_to_matrix(root_quat_mujoco)

        # the joint rotations from dof and rotation axis
        dof = mujoco_qpos[:, :, 7:]  # batch_size, num_frames=4, joints=30 - 1 (pelvis) = 29
        quaternion_if_x_axis = t.stack([t.cos(dof / 2), t.sin(dof / 2), t.zeros_like(dof), t.zeros_like(dof)], dim=-1)
        quaternion_if_y_axis = t.stack([t.cos(dof / 2), t.zeros_like(dof), t.sin(dof / 2), t.zeros_like(dof)], dim=-1)
        quaternion_if_z_axis = t.stack([t.cos(dof / 2), t.zeros_like(dof), t.zeros_like(dof), t.sin(dof / 2)], dim=-1)
        quaternion_from_xyz_axis = t.stack([quaternion_if_x_axis, quaternion_if_y_axis, quaternion_if_z_axis],
                                           dim=-1)  # [batch_size, num_frames, joints, 4 quat, 3 axis]
        joint_quaternion = (
            quaternion_from_xyz_axis *
            self._mujoco_joint_axis_values_mujoco_space[None, None, :, None, :].to(device)
        ).sum(dim=-1)
        joint_rotation_matrix = quaternion_to_matrix(joint_quaternion)  # [batch_size, num_frames, joints, 3, 3]
        joint_rotation_matrix = t.matmul(self._rot_offsets_q2t.to(device)[self._mujoco_indices_to_motion_indices],
                                         joint_rotation_matrix)

        # run FK to compute joint positions
        rot_matrices = t.concat([root_rotation_mujoco[:, :, None, :, :], joint_rotation_matrix], dim=2)
        rot_matrices = rot_matrices.view(batch_size * num_frames, self._nb_joints_mujoco, 3, 3)
        global_joint_positions, global_joint_rotations = forward_kinematics(
            rot_matrices,
            self._capture_neutral_joints_mujoco.to(device).repeat(batch_size * num_frames, 1, 1),
            self._mujoco_joint_including_root_parent_list, self._mujoco_joint_idx_levs, 0  # root index = 0
        )
        global_joint_positions = global_joint_positions.view(batch_size, num_frames, self._nb_joints_mujoco, 3)
        global_joint_rotations = global_joint_rotations.view(batch_size, num_frames, self._nb_joints_mujoco, 3, 3)

        global_joint_positions = global_joint_positions + root_translation_mujoco[:, :, None, :]

        # convert to the motion joint transforms
        return self.convert_mujoco_transforms_to_motion_transforms(global_joint_positions, global_joint_rotations)

    def convert_mujoco_transforms_to_motion_transforms(self, global_joint_positions_mujoco: t.Tensor,
                                                    global_joint_rotations_mujoco: t.Tensor):
        """ @brief:
            convert the mujoco transforms to motion transforms:
            1. coordinate system conversion, 2. t-pose conversion, 3. populate dead end-effector joints
        """
        batch_size, num_frames = global_joint_rotations_mujoco.shape[:2]
        device, dtype = global_joint_rotations_mujoco.device, global_joint_rotations_mujoco.dtype
        mujoco_to_motion_matrix = self.mujoco_to_motion_matrix.to(device=device, dtype=dtype)
        motion_to_mujoco_matrix = self.motion_to_mujoco_matrix.to(device=device, dtype=dtype)

        # coordinate system conversion
        global_joint_positions_mujoco = t.matmul(mujoco_to_motion_matrix[None, None, None, ...],
                                                 global_joint_positions_mujoco[..., None])[..., 0]  # [B, F, J, 3]
        global_joint_rotations_mujoco = t.matmul(t.matmul(mujoco_to_motion_matrix[None, None, None, ...],
                                                          global_joint_rotations_mujoco),
                                                 motion_to_mujoco_matrix[None, None, None, ...])  # [B, F, J, 3, 3]

        # swap the order of the joints
        global_joint_rotations_motion = global_joint_rotations_mujoco[:, :, self._motion_indices_to_mujoco_indices, ...]
        global_joint_positions_motion = global_joint_positions_mujoco[:, :, self._motion_indices_to_mujoco_indices, ...]

        # populate the dead joints' rotations (`right_hand_roll_skel` and `left_hand_roll_skel` not in the mujoco dof)
        # this is assuming the dead joints has 0 relative rotation, which is true for `capture` tpose
        is_dead_joints = self._motion_indices_to_mujoco_indices == -1
        is_dead_joints_inds = is_dead_joints.nonzero().squeeze().tolist()

        dead_joints_parent_joints = self.motion_rep.skeleton.joint_parents[t.where(is_dead_joints)[0]]
        dead_joints_parent_joints_inds = dead_joints_parent_joints.squeeze().tolist()

        if self._dead_joint_rotation_scheme == "dummy":
            # Explicitly expand identity matrix to match target shape for ONNX compatibility
            batch_size, num_frames = global_joint_rotations_motion.shape[:2]
            num_dead_joints = len(is_dead_joints_inds)
            identity_expanded = t.eye(3, device=device, dtype=dtype).expand(batch_size, num_frames, num_dead_joints, 3, 3)
            global_joint_rotations_motion[:, :, is_dead_joints_inds, ...] = identity_expanded
        elif self._dead_joint_rotation_scheme == "parent":
            global_joint_rotations_motion[:, :, is_dead_joints_inds, ...] = \
                global_joint_rotations_motion[:, :, dead_joints_parent_joints_inds, ...]
        else:
            raise ValueError(f"Invalid dead joint rotation scheme: {self._dead_joint_rotation_scheme}")

        # populate the dead joints's global positions
        dead_joint_neutral_positions = \
            self.motion_rep.skeleton.neutral_joints[is_dead_joints_inds][None, None, :, :, None].to(device, dtype) - \
            self.motion_rep.skeleton.neutral_joints[dead_joints_parent_joints_inds][None, None, :, :, None].to(device, dtype)
        global_joint_positions_motion[:, :, is_dead_joints_inds, ...] = \
            global_joint_positions_motion[:, :, dead_joints_parent_joints_inds, ...] + \
            t.matmul(global_joint_rotations_motion[:, :, dead_joints_parent_joints_inds, ...],
                     dead_joint_neutral_positions)[..., 0]

        return global_joint_positions_motion, global_joint_rotations_motion

    @property
    def t_pose_translations(self):
        return {'standard': self._global_offset_standard, 'capture': self._global_offset_capture}

    @property
    def joint_indice_mapping(self):
        return {'motion_to_mujoco': self._motion_indices_to_mujoco_indices, 'mujoco_to_motion': self._mujoco_indices_to_motion_indices}

    def get_subtree_joints(self, sub_tree_start_joint: str):
        """ @brief: get the indices of the joints starting from the @sub_tree_start_joint """

        if sub_tree_start_joint in self._subtree_joints:
            return self._subtree_joints[sub_tree_start_joint]

        assert sub_tree_start_joint in self.motion_rep.skeleton.bone_order_names, \
            f"sub_tree_start_joint {sub_tree_start_joint} not in the skeleton"
        tree_idx = self.motion_rep.skeleton.bone_order_names.index(sub_tree_start_joint)
        parents, all_children = self.motion_rep.skeleton.joint_parents, [tree_idx]
        while True:
            new_children = [i for i, p in enumerate(parents) if p in all_children and i not in all_children]
            all_children.extend(new_children)
            if not new_children:
                break
        all_children_mujoco = \
            np.array([i for i in self.joint_indice_mapping['motion_to_mujoco'][all_children] if i != -1])
        self._subtree_joints[sub_tree_start_joint] = all_children_mujoco

        return all_children_mujoco
