from typing import List, Optional, Set, Union

import numpy as np
import pinocchio as pin

from decoupled_wbc.control.robot_model.supplemental_info import RobotSupplementalInfo


class RobotModel:
    def __init__(
        self,
        urdf_path,
        asset_path,
        set_floating_base=False,
        supplemental_info: Optional[RobotSupplementalInfo] = None,
    ):
        self.pinocchio_wrapper = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=[asset_path],
            root_joint=pin.JointModelFreeFlyer() if set_floating_base else None,
        )
        self.is_floating_base_model = set_floating_base

        self.joint_to_dof_index = {}
        # Assume we only have single-dof joints
        # First two names correspond to universe and floating base joints
        names = (
            self.pinocchio_wrapper.model.names[2:]
            if set_floating_base
            else self.pinocchio_wrapper.model.names[1:]
        )
        for name in names:
            j_id = self.pinocchio_wrapper.model.getJointId(name)
            jmodel = self.pinocchio_wrapper.model.joints[j_id]
            self.joint_to_dof_index[name] = jmodel.idx_q

        # Store joint limits only for actual joints (excluding floating base)
        # if set floating base is true and the robot can move in the world
        # then we don't want to impose joint limits for the 7 dofs corresponding
        # to the floating base dofs.
        root_nq = 7 if set_floating_base else 0
        self.upper_joint_limits = self.pinocchio_wrapper.model.upperPositionLimit[root_nq:].copy()
        self.lower_joint_limits = self.pinocchio_wrapper.model.lowerPositionLimit[root_nq:].copy()

        # Set up supplemental info if provided
        self.supplemental_info = supplemental_info
        if self.supplemental_info is not None:
            # Cache indices for body and hand actuated joints separately
            self._body_actuated_joint_indices = [
                self.dof_index(name) for name in self.supplemental_info.body_actuated_joints
            ]
            self._left_hand_actuated_joint_indices = [
                self.dof_index(name) for name in self.supplemental_info.left_hand_actuated_joints
            ]
            self._right_hand_actuated_joint_indices = [
                self.dof_index(name) for name in self.supplemental_info.right_hand_actuated_joints
            ]
            self._hand_actuated_joint_indices = (
                self._left_hand_actuated_joint_indices + self._right_hand_actuated_joint_indices
            )

            # Cache indices for joint groups, handling nested groups
            self._joint_group_indices = {}
            for group_name, group_info in self.supplemental_info.joint_groups.items():
                indices = []
                # Add indices for direct joints
                indices.extend([self.dof_index(name) for name in group_info["joints"]])
                # Add indices from subgroups
                for subgroup_name in group_info["groups"]:
                    indices.extend(self.get_joint_group_indices(subgroup_name))
                self._joint_group_indices[group_name] = sorted(set(indices))

            # Update joint limits from supplemental info if available
            if (
                hasattr(self.supplemental_info, "joint_limits")
                and self.supplemental_info.joint_limits
            ):
                for joint_name, limits in self.supplemental_info.joint_limits.items():
                    if joint_name in self.joint_to_dof_index:
                        idx = self.joint_to_dof_index[joint_name] - root_nq
                        self.lower_joint_limits[idx] = limits[0]
                        self.upper_joint_limits[idx] = limits[1]

        # Initialize default body pose
        self.default_body_pose = self.q_zero.copy()

        # Update with supplemental info if available
        if self.supplemental_info is not None:
            default_joint_q = self.supplemental_info.default_joint_q
            for joint, joint_values in default_joint_q.items():
                # Get the joint name mapping for this type
                joint_mapping = self.supplemental_info.joint_name_mapping[joint]

                # Handle both single joint names and left/right mappings
                if isinstance(joint_mapping, str):
                    # Single joint (e.g., waist joints)
                    if joint_mapping in self.joint_to_dof_index:
                        joint_idx = self.dof_index(joint_mapping)
                        self.default_body_pose[joint_idx] = (
                            joint_values  # joint_values is the value for single joints
                        )
                else:
                    # Left/right mapping (e.g., arm joints)
                    for side, value in joint_values.items():
                        if side in joint_mapping and joint_mapping[side] in self.joint_to_dof_index:
                            joint_idx = self.dof_index(joint_mapping[side])
                            self.default_body_pose[joint_idx] = value

        # Initialize initial body pose
        self.initial_body_pose = self.default_body_pose.copy()

    @property
    def num_dofs(self) -> int:
        """Get the number of degrees of freedom of the robot (floating base pose + joints)."""
        return self.pinocchio_wrapper.model.nq

    @property
    def q_zero(self) -> np.ndarray:
        """Get the zero pose of the robot."""
        return self.pinocchio_wrapper.q0

    @property
    def joint_names(self) -> List[str]:
        """Get the names of the joints of the robot."""
        return list(self.joint_to_dof_index.keys())

    @property
    def num_joints(self) -> int:
        """Get the number of joints of the robot."""
        return len(self.joint_to_dof_index)

    def dof_index(self, joint_name: str) -> int:
        """
        Get the index in the degrees of freedom vector corresponding
        to the single-DoF joint with name `joint_name`.
        """
        if joint_name not in self.joint_to_dof_index:
            raise ValueError(
                f"Unknown joint name: '{joint_name}'. "
                f"Available joints: {list(self.joint_to_dof_index.keys())}"
            )
        return self.joint_to_dof_index[joint_name]

    def get_body_actuated_joint_indices(self) -> List[int]:
        """
        Get the indices of body actuated joints in the full configuration.
        Ordering is that of the actuated joints as defined in the supplemental info.
        Requires supplemental_info to be provided.
        """
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")
        return self._body_actuated_joint_indices

    def get_hand_actuated_joint_indices(self, side: str = "both") -> List[int]:
        """
        Get the indices of hand actuated joints in the full configuration.
        Ordering is that of the actuated joints as defined in the supplemental info.
        Requires supplemental_info to be provided.

        Args:
            side: String specifying which hand to get indices for ('left', 'right', or 'both')
        """
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")

        if side.lower() == "both":
            return self._hand_actuated_joint_indices
        elif side.lower() == "left":
            return self._left_hand_actuated_joint_indices
        elif side.lower() == "right":
            return self._right_hand_actuated_joint_indices
        else:
            raise ValueError("side must be 'left', 'right', or 'both'")

    def get_joint_group_indices(self, group_names: Union[str, Set[str]]) -> List[int]:
        """
        Get the indices of joints in one or more groups in the full configuration.
        Requires supplemental_info to be provided.
        The returned indices are sorted in ascending order, so that the joint ordering
        of the full model is preserved.

        Args:
            group_names: Either a single group name (str) or a set of group names (Set[str])

        Returns:
            List of joint indices in sorted order with no duplicates
        """
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")

        # Convert single string to set for uniform handling
        if isinstance(group_names, str):
            group_names = {group_names}

        # Collect indices from all groups
        all_indices = set()
        for group_name in group_names:
            if group_name not in self._joint_group_indices:
                raise ValueError(f"Unknown joint group: {group_name}")
            all_indices.update(self._joint_group_indices[group_name])

        return sorted(all_indices)

    def cache_forward_kinematics(self, q: np.ndarray, auto_clip=True) -> None:
        """
        Perform forward kinematics to update the pose of every joint and frame
        in the Pinocchio data structures for the given configuration `q`.

        :param q: A numpy array of shape (num_dofs,) representing the robot configuration.
        """
        if q.shape[0] != self.num_dofs:
            raise ValueError(f"Expected q of length {self.num_dofs}, got {q.shape[0]} instead.")

        # Apply auto-clip if enabled
        if auto_clip:
            q = self.clip_configuration(q)

        pin.framesForwardKinematics(self.pinocchio_wrapper.model, self.pinocchio_wrapper.data, q)

    def compute_gravity_compensation_torques(
        self, q: np.ndarray, joint_groups: Union[str, List[str], Set[str]] = None, auto_clip=True
    ) -> np.ndarray:
        """
        Compute gravity compensation torques for specified joint groups using pinocchio.

        :param q: Robot configuration (joint positions)
        :param joint_groups: Joint groups to compensate (e.g., "arms", ["left_arm", "waist"],
             {"left_arm", "waist"}). If None, compensates all joints
        :param auto_clip: Whether to automatically clip joint values to limits
        :return: Array of gravity compensation torques for all DOFs (zero for non-compensated joints)
        """
        if q.shape[0] != self.num_dofs:
            raise ValueError(f"Expected q of length {self.num_dofs}, got {q.shape[0]} instead.")

        # Apply auto-clip if enabled
        if auto_clip:
            q = self.clip_configuration(q)

        try:
            # Cache forward kinematics for the current configuration
            self.cache_forward_kinematics(q, auto_clip=False)  # Already clipped if needed

            # Compute gravity vector using RNEA with zero velocity and acceleration
            v = np.zeros(self.num_dofs)
            a = np.zeros(self.num_dofs)

            gravity_torques_full = pin.rnea(
                self.pinocchio_wrapper.model, self.pinocchio_wrapper.data, q, v, a
            )

            # If no joint groups specified, return full gravity torques
            if joint_groups is None:
                return gravity_torques_full

            # Convert list to set for get_joint_group_indices compatibility
            if isinstance(joint_groups, list):
                joint_groups = set(joint_groups)

            # Get joint indices for specified groups - get_joint_group_indices handles str and Set[str]
            try:
                compensated_joint_indices = self.get_joint_group_indices(joint_groups)
            except ValueError as e:
                raise ValueError(f"Error resolving joint groups {joint_groups}: {e}")

            # Create mask for joints that should receive gravity compensation
            compensation_mask = np.zeros(self.num_dofs, dtype=bool)
            for joint_idx in compensated_joint_indices:
                if 0 <= joint_idx < len(compensation_mask):
                    compensation_mask[joint_idx] = True

            # Apply mask to only compensate specified joints
            compensated_torques = np.zeros_like(gravity_torques_full)
            compensated_torques[compensation_mask] = gravity_torques_full[compensation_mask]

            return compensated_torques

        except Exception as e:
            raise RuntimeError(f"Error computing gravity compensation: {e}")

    def clip_configuration(self, q: np.ndarray, margin: float = 1e-6) -> np.ndarray:
        """
        Clip the configuration to stay within joint limits with a small tolerance.

        :param q: Configuration to clip
        :param margin: Tolerance to keep away from joint limits
        :return: Clipped configuration
        """
        q_clipped = q.copy()

        # Only clip joint positions, not floating base
        root_nq = 7 if self.is_floating_base_model else 0
        q_clipped[root_nq:] = np.clip(
            q[root_nq:], self.lower_joint_limits + margin, self.upper_joint_limits - margin
        )

        return q_clipped

    def frame_placement(self, frame_name: str) -> pin.SE3:
        """
        Returns the SE3 transform of the specified frame in the world coordinate system.
        Note: make sure cache_forward_kinematics() has been previously called.

        :param frame_name: Name of the frame, e.g. "link_elbow_frame", "hand_imu_frame", etc.
        :return: A pin.SE3 object representing the pose of the frame.
        """
        model = self.pinocchio_wrapper.model
        data = self.pinocchio_wrapper.data

        frame_id = model.getFrameId(frame_name)
        if frame_id < 0 or frame_id >= len(model.frames):
            valid_frames = [f.name for f in model.frames]
            raise ValueError(f"Unknown frame '{frame_name}'. Valid frames: {valid_frames}")

        # Pinocchio's data.oMf[frame_id] is a pin.SE3.
        return data.oMf[frame_id].copy()

    def get_body_actuated_joints(self, q: np.ndarray) -> np.ndarray:
        """
        Get the configuration of body actuated joints from a full configuration.

        :param q: Configuration in full space
        :return: Configuration of body actuated joints
        """
        indices = self.get_body_actuated_joint_indices()

        return q[indices]

    def get_hand_actuated_joints(self, q: np.ndarray, side: str = "both") -> np.ndarray:
        """
        Get the configuration of hand actuated joints from a full configuration.

        Args:
            q: Configuration in full space
            side: String specifying which hand to get joints for ('left', 'right', or 'both')
        """
        indices = self.get_hand_actuated_joint_indices(side)
        return q[indices]

    def get_configuration_from_actuated_joints(
        self,
        body_actuated_joint_values: np.ndarray,
        hand_actuated_joint_values: Optional[np.ndarray] = None,
        left_hand_actuated_joint_values: Optional[np.ndarray] = None,
        right_hand_actuated_joint_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get the full configuration from the body and hand actuated joint configurations.
        Can specify either both hands together or left and right hands separately.

        Args:
            body_actuated_joint_values: Configuration of body actuated joints
            hand_actuated_joint_values: Configuration of both hands' actuated joints (optional)
            left_hand_actuated_joint_values: Configuration of left hand actuated joints (optional)
            right_hand_actuated_joint_values: Configuration of right hand actuated joints (optional)

        Returns:
            Full configuration including body and hand joints
        """
        q = self.pinocchio_wrapper.q0.copy()
        q[self.get_body_actuated_joint_indices()] = body_actuated_joint_values

        # Handle hand configurations
        if hand_actuated_joint_values is not None:
            # Use combined hand configuration
            q[self.get_hand_actuated_joint_indices("both")] = hand_actuated_joint_values
        else:
            # Use separate hand configurations
            if left_hand_actuated_joint_values is not None:
                q[self.get_hand_actuated_joint_indices("left")] = left_hand_actuated_joint_values
            if right_hand_actuated_joint_values is not None:
                q[self.get_hand_actuated_joint_indices("right")] = right_hand_actuated_joint_values

        return q

    def reset_forward_kinematics(self) -> None:
        """
        Reset the forward kinematics to the initial configuration.
        """
        self.cache_forward_kinematics(self.q_zero)

    def get_initial_upper_body_pose(self) -> np.ndarray:
        """
        Get the initial upper body pose of the robot.
        """
        return self.initial_body_pose[self.get_joint_group_indices("upper_body")]

    def get_default_body_pose(self) -> np.ndarray:
        """
        Get the default body pose of the robot.
        """
        return self.default_body_pose

    def set_initial_body_pose(self, q: np.ndarray, q_idx=None) -> None:
        """
        Set the initial body pose of the robot.
        """
        if q_idx is None:
            self.initial_body_pose = q
        else:
            self.initial_body_pose[q_idx] = q


class ReducedRobotModel(RobotModel):
    """
    A class that creates a reduced order robot model by fixing certain joints.
    This class maintains a mapping between the reduced state space and the full state space.
    """

    def __init__(
        self,
        full_robot_model: RobotModel,
        fixed_joints: List[str],
        fixed_values: Optional[List[float]] = None,
    ):
        """
        Create a reduced order robot model by fixing specified joints.

        :param full_robot_model: The original robot model
        :param fixed_joints: List of joint names to fix
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        """
        self.full_robot = full_robot_model
        self.supplemental_info = full_robot_model.supplemental_info

        # If fixed_values is None, use q0 from the full robot model
        if fixed_values is None:
            fixed_values = []
            for joint_name in fixed_joints:
                full_idx = full_robot_model.dof_index(joint_name)
                fixed_values.append(full_robot_model.pinocchio_wrapper.q0[full_idx])
        elif len(fixed_joints) != len(fixed_values):
            raise ValueError("fixed_joints and fixed_values must have the same length")

        # Store fixed joints and their values
        self.fixed_joints = fixed_joints
        self.fixed_values = fixed_values

        # Create mapping between reduced and full state spaces
        self.reduced_to_full = []
        self.full_to_reduced = {}

        # Initialize with floating base indices if present
        if full_robot_model.is_floating_base_model:
            self.reduced_to_full.extend(range(7))  # Floating base indices
            for i in range(7):
                self.full_to_reduced[i] = i

        # Add active joint indices
        for joint_name in full_robot_model.joint_names:
            if joint_name not in fixed_joints:
                full_idx = full_robot_model.dof_index(joint_name)
                reduced_idx = len(self.reduced_to_full)
                self.reduced_to_full.append(full_idx)
                self.full_to_reduced[full_idx] = reduced_idx

        # Create a reduced Pinocchio model using buildReducedModel
        # First, get the list of joint IDs to lock
        locked_joint_ids = []
        for joint_name in fixed_joints:
            joint_id = full_robot_model.pinocchio_wrapper.model.getJointId(joint_name)
            if (full_robot_model.is_floating_base_model and joint_id > 1) or (
                not full_robot_model.is_floating_base_model and joint_id > 0
            ):
                locked_joint_ids.append(joint_id)

        # First build the reduced kinematic model
        reduced_model = pin.buildReducedModel(
            full_robot_model.pinocchio_wrapper.model,
            locked_joint_ids,
            full_robot_model.pinocchio_wrapper.q0,
        )

        # Then build the reduced geometry models using the reduced kinematic model
        self.pinocchio_wrapper = pin.RobotWrapper(
            model=reduced_model,
        )

        # Create joint to dof index mapping
        self.joint_to_dof_index = {}
        # Assume we only have single-dof joints
        # First two names correspond to universe and floating base joints
        names = (
            self.pinocchio_wrapper.model.names[2:]
            if self.full_robot.is_floating_base_model
            else self.pinocchio_wrapper.model.names[1:]
        )
        for name in names:
            j_id = self.pinocchio_wrapper.model.getJointId(name)
            jmodel = self.pinocchio_wrapper.model.joints[j_id]
            self.joint_to_dof_index[name] = jmodel.idx_q

        # Initialize joint limits
        root_nq = 7 if self.full_robot.is_floating_base_model else 0
        self.lower_joint_limits = self.pinocchio_wrapper.model.lowerPositionLimit[root_nq:].copy()
        self.upper_joint_limits = self.pinocchio_wrapper.model.upperPositionLimit[root_nq:].copy()

        # Update joint limits from supplemental info if available
        if self.supplemental_info is not None:
            if (
                hasattr(self.supplemental_info, "joint_limits")
                and self.supplemental_info.joint_limits
            ):
                for joint_name, limits in self.supplemental_info.joint_limits.items():
                    if joint_name in self.joint_to_dof_index:
                        idx = self.joint_to_dof_index[joint_name] - root_nq
                        self.lower_joint_limits[idx] = limits[0]
                        self.upper_joint_limits[idx] = limits[1]

            # Get full indices for body and hand actuated joints
            full_body_indices = full_robot_model.get_body_actuated_joint_indices()
            full_hand_indices = full_robot_model.get_hand_actuated_joint_indices("both")
            full_left_hand_indices = full_robot_model.get_hand_actuated_joint_indices("left")
            full_right_hand_indices = full_robot_model.get_hand_actuated_joint_indices("right")

            # Map to reduced indices
            self._body_actuated_joint_indices = []
            for idx in full_body_indices:
                if idx in self.full_to_reduced:
                    self._body_actuated_joint_indices.append(self.full_to_reduced[idx])

            self._hand_actuated_joint_indices = []
            for idx in full_hand_indices:
                if idx in self.full_to_reduced:
                    self._hand_actuated_joint_indices.append(self.full_to_reduced[idx])

            self._left_hand_actuated_joint_indices = []
            for idx in full_left_hand_indices:
                if idx in self.full_to_reduced:
                    self._left_hand_actuated_joint_indices.append(self.full_to_reduced[idx])

            self._right_hand_actuated_joint_indices = []
            for idx in full_right_hand_indices:
                if idx in self.full_to_reduced:
                    self._right_hand_actuated_joint_indices.append(self.full_to_reduced[idx])

            # Cache indices for joint groups in reduced space
            self._joint_group_indices = {}
            for group_name in self.supplemental_info.joint_groups:
                full_indices = full_robot_model.get_joint_group_indices(group_name)
                reduced_indices = []
                for idx in full_indices:
                    if idx in self.full_to_reduced:
                        reduced_indices.append(self.full_to_reduced[idx])
                self._joint_group_indices[group_name] = sorted(set(reduced_indices))

        # Initialize default body pose in reduced space
        self.default_body_pose = self.full_to_reduced_configuration(
            full_robot_model.default_body_pose
        )

        # Initialize initial body pose in reduced space
        self.initial_body_pose = self.full_to_reduced_configuration(
            full_robot_model.initial_body_pose
        )

    @property
    def num_joints(self) -> int:
        """Get the number of active joints in the reduced model."""
        return len(self.joint_names)

    @property
    def joint_names(self) -> List[str]:
        """Get the names of the active joints in the reduced model."""
        return [name for name in self.full_robot.joint_names if name not in self.fixed_joints]

    @classmethod
    def from_fixed_groups(
        cls,
        full_robot_model: RobotModel,
        fixed_group_names: List[str],
        fixed_values: Optional[List[float]] = None,
    ) -> "ReducedRobotModel":
        """
        Create a reduced order robot model by fixing all joints in specified groups.

        :param full_robot_model: The original robot model
        :param fixed_group_names: List of joint group names to fix
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        :return: A ReducedRobotModel instance
        """
        if full_robot_model.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")

        # Get all joints in the groups, including those from subgroups
        fixed_joints = set()  # Use a set to avoid duplicates

        for group_name in fixed_group_names:
            if group_name not in full_robot_model.supplemental_info.joint_groups:
                raise ValueError(f"Unknown joint group: {group_name}")

            group_info = full_robot_model.supplemental_info.joint_groups[group_name]

            # Add direct joints
            fixed_joints.update(group_info["joints"])

            # Add joints from subgroups
            for subgroup_name in group_info["groups"]:
                subgroup_joints = full_robot_model.get_joint_group_indices(subgroup_name)
                fixed_joints.update([full_robot_model.joint_names[idx] for idx in subgroup_joints])

        # Convert set back to list for compatibility with the original constructor
        return cls(full_robot_model, list(fixed_joints), fixed_values)

    @classmethod
    def from_fixed_group(
        cls,
        full_robot_model: RobotModel,
        fixed_group_name: str,
        fixed_values: Optional[List[float]] = None,
    ) -> "ReducedRobotModel":
        """
        Create a reduced order robot model by fixing all joints in a specified group.
        This is a convenience method that calls from_fixed_groups with a single group.

        :param full_robot_model: The original robot model
        :param fixed_group_name: Name of the joint group to fix
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        :return: A ReducedRobotModel instance
        """
        return cls.from_fixed_groups(full_robot_model, [fixed_group_name], fixed_values)

    @classmethod
    def from_active_group(
        cls,
        full_robot_model: RobotModel,
        active_group_name: str,
        fixed_values: Optional[List[float]] = None,
    ) -> "ReducedRobotModel":
        """
        Create a reduced order robot model by fixing all joints EXCEPT those in the specified group.
        This is a convenience method that calls from_active_groups with a single group.

        :param full_robot_model: The original robot model
        :param active_group_name: Name of the joint group to keep active (all other joints will be fixed)
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        :return: A ReducedRobotModel instance
        """
        return cls.from_active_groups(full_robot_model, [active_group_name], fixed_values)

    @classmethod
    def from_active_groups(
        cls,
        full_robot_model: RobotModel,
        active_group_names: List[str],
        fixed_values: Optional[List[float]] = None,
    ) -> "ReducedRobotModel":
        """
        Create a reduced order robot model by fixing all joints EXCEPT those in the specified groups.
        This is useful when you want to keep multiple groups active and fix everything else.

        :param full_robot_model: The original robot model
        :param active_group_names: List of joint group names to keep active (all other joints will be fixed)
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        :return: A ReducedRobotModel instance
        """
        if full_robot_model.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")

        # Get all joints in the active groups, including those from subgroups
        active_joints = set()

        def add_group_joints(group_name: str):
            if group_name not in full_robot_model.supplemental_info.joint_groups:
                raise ValueError(f"Unknown joint group: {group_name}")

            group_info = full_robot_model.supplemental_info.joint_groups[group_name]

            # Add direct joints
            if "joints" in group_info:
                active_joints.update(group_info["joints"])

            # Add joints from subgroups
            if "groups" in group_info:
                for subgroup_name in group_info["groups"]:
                    add_group_joints(subgroup_name)

        for group_name in active_group_names:
            add_group_joints(group_name)

        # Get all joints from the model
        all_joints = set(full_robot_model.joint_names)

        # The fixed joints are all joints minus the active joints
        fixed_joints = list(all_joints - active_joints)

        return cls(full_robot_model, fixed_joints, fixed_values)

    def reduced_to_full_configuration(self, q_reduced: np.ndarray) -> np.ndarray:
        """
        Convert a reduced configuration to the full configuration space.

        :param q_reduced: Configuration in reduced space
        :return: Configuration in full space with fixed joints set to their fixed values
        """
        if q_reduced.shape[0] != self.num_dofs:
            raise ValueError(
                f"Expected q_reduced of length {self.num_dofs}, got {q_reduced.shape[0]} instead"
            )

        q_full = np.zeros(self.full_robot.num_dofs)

        # Set active joints
        for reduced_idx, full_idx in enumerate(self.reduced_to_full):
            q_full[full_idx] = q_reduced[reduced_idx]

        # Set fixed joints
        for joint_name, value in zip(self.fixed_joints, self.fixed_values):
            full_idx = self.full_robot.dof_index(joint_name)
            q_full[full_idx] = value

        return q_full

    def full_to_reduced_configuration(self, q_full: np.ndarray) -> np.ndarray:
        """
        Convert a full configuration to the reduced configuration space.

        :param q_full: Configuration in full space
        :return: Configuration in reduced space
        """
        if q_full.shape[0] != self.full_robot.num_dofs:
            raise ValueError(
                f"Expected q_full of length {self.full_robot.num_dofs}, got {q_full.shape[0]} instead"
            )

        q_reduced = np.zeros(self.num_dofs)

        # Copy active joints
        for reduced_idx, full_idx in enumerate(self.reduced_to_full):
            q_reduced[reduced_idx] = q_full[full_idx]

        return q_reduced

    def cache_forward_kinematics(self, q_reduced: np.ndarray, auto_clip=True) -> None:
        """
        Perform forward kinematics using the reduced configuration.

        :param q_reduced: Configuration in reduced space
        """
        # First update the full robot's forward kinematics
        q_full = self.reduced_to_full_configuration(q_reduced)
        self.full_robot.cache_forward_kinematics(q_full, auto_clip)

        # Then update the reduced model's forward kinematics
        pin.framesForwardKinematics(
            self.pinocchio_wrapper.model, self.pinocchio_wrapper.data, q_reduced
        )

    def clip_configuration(self, q_reduced: np.ndarray, margin: float = 1e-6) -> np.ndarray:
        """
        Clip the reduced configuration to stay within joint limits with a small tolerance.

        :param q_reduced: Configuration to clip
        :param margin: Tolerance to keep away from joint limits
        :return: Clipped configuration
        """
        q_full = self.reduced_to_full_configuration(q_reduced)
        q_full_clipped = self.full_robot.clip_configuration(q_full, margin)
        return self.full_to_reduced_configuration(q_full_clipped)

    def reset_forward_kinematics(self):
        """
        Reset the forward kinematics to the initial configuration.
        """
        # Reset full robot's forward kinematics
        self.full_robot.reset_forward_kinematics()
        # Reset reduced model's forward kinematics
        self.cache_forward_kinematics(self.q_zero)
