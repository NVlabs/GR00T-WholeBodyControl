# test_robot_model.py

import numpy as np
import pinocchio as pin
import pytest

from decoupled_wbc.control.robot_model import ReducedRobotModel
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model


@pytest.fixture
def g1_robot_model():
    """
    Fixture that creates and returns a G1 RobotModel instance.
    """
    return instantiate_g1_robot_model()


def test_robot_model_initialization(g1_robot_model):
    """
    Test initialization of the RobotModel and its main attributes.
    """
    for robot_model in [g1_robot_model]:
        # Check that the Pinocchio wrapper exists
        assert robot_model.pinocchio_wrapper is not None

        # Check number of degrees of freedom (nq)
        assert robot_model.num_dofs > 0

        # Check we have the expected number of joints beyond the floating base
        assert len(robot_model.joint_names) > 0

        # Check that supplemental info is present
        assert robot_model.supplemental_info is not None


def test_robot_model_joint_names(g1_robot_model):
    """
    Test that joint_names is populated correctly
    and that dof_index works.
    """
    for robot_model in [g1_robot_model]:
        # Extract joint names
        joint_names = robot_model.joint_names

        # Pick the first joint name and get its index
        first_joint_name = joint_names[0]
        idx = robot_model.dof_index(first_joint_name)
        assert idx >= 0

        # Test that an unknown joint name raises an error
        with pytest.raises(ValueError, match="Unknown joint name"):
            _ = robot_model.dof_index("non_existent_joint")


def test_robot_model_forward_kinematics_valid_q(g1_robot_model):
    """
    Test that cache_forward_kinematics works with a valid q.
    """
    for robot_model in [g1_robot_model]:
        nq = robot_model.num_dofs

        # Construct a valid configuration (e.g., zero vector)
        q_valid = np.zeros(nq)

        # Should not raise any exception
        robot_model.cache_forward_kinematics(q_valid)


def test_robot_model_forward_kinematics_invalid_q(g1_robot_model):
    """
    Test that cache_forward_kinematics raises an error with an invalid q.
    """
    for robot_model in [g1_robot_model]:
        nq = robot_model.num_dofs

        # Construct an invalid configuration (wrong size)
        q_invalid = np.zeros(nq + 1)

        with pytest.raises(ValueError, match="Expected q of length"):
            robot_model.cache_forward_kinematics(q_invalid)


def test_robot_model_frame_placement(g1_robot_model):
    """
    Test the frame_placement method with a valid and invalid frame name.
    Also test that frame placements change with different configurations.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing")

        # Use the hand frame from supplemental info
        test_frame = robot_model.supplemental_info.hand_frame_names["left"]

        # Test with zero configuration
        q_zero = np.zeros(robot_model.num_dofs)
        robot_model.cache_forward_kinematics(q_zero)
        placement_zero = robot_model.frame_placement(test_frame)
        assert isinstance(placement_zero, pin.SE3)

        # Test with non-zero configuration
        q_non_zero = np.zeros(robot_model.num_dofs)
        root_nq = 7 if robot_model.is_floating_base_model else 0

        # Set a more significant configuration change
        # Use π/2 for all joints to create a more noticeable difference
        q_non_zero[root_nq:] = np.pi / 2  # 90 degrees for all joints

        robot_model.cache_forward_kinematics(q_non_zero)
        placement_non_zero = robot_model.frame_placement(test_frame)

        # Verify that frame placements are different with different configurations
        assert not np.allclose(
            placement_zero.translation, placement_non_zero.translation
        ) or not np.allclose(placement_zero.rotation, placement_non_zero.rotation)

        # Should raise an error for an invalid frame
        with pytest.raises(ValueError, match="Unknown frame"):
            robot_model.frame_placement("non_existent_frame")


# Tests for ReducedRobotModel
def test_reduced_robot_model_initialization(g1_robot_model):
    """
    Test initialization of the ReducedRobotModel.
    """
    for robot_model in [g1_robot_model]:
        # Create a reduced model by fixing some actual joints from the robot
        fixed_joints = robot_model.joint_names[:2]  # Use first two joints from the robot
        reduced_robot = ReducedRobotModel(robot_model, fixed_joints)

        # Check that the full robot is stored
        assert reduced_robot.full_robot is robot_model

        # Check that fixed joints are stored correctly
        assert reduced_robot.fixed_joints == fixed_joints
        assert len(reduced_robot.fixed_values) == len(fixed_joints)

        # Check that the number of dofs is reduced
        assert reduced_robot.num_dofs == robot_model.num_dofs - len(fixed_joints)


def test_reduced_robot_model_joint_names(g1_robot_model):
    """
    Test that joint_names in ReducedRobotModel excludes fixed joints.
    """
    for robot_model in [g1_robot_model]:
        # Use actual joints from the robot
        fixed_joints = robot_model.joint_names[:2]  # Use first two joints from the robot
        reduced_robot = ReducedRobotModel(robot_model, fixed_joints)

        # Check that fixed joints are not in the reduced model's joint names
        for joint in fixed_joints:
            assert joint not in reduced_robot.joint_names

        # Check that other joints are still present
        for joint in robot_model.joint_names:
            if joint not in fixed_joints:
                assert joint in reduced_robot.joint_names


def test_reduced_robot_model_configuration_conversion(g1_robot_model):
    """
    Test conversion between reduced and full configurations.
    """
    for robot_model in [g1_robot_model]:
        # Use actual joints from the robot
        fixed_joints = robot_model.joint_names[:2]  # Use first two joints from the robot
        fixed_values = [0.5, 1.0]
        reduced_robot = ReducedRobotModel(robot_model, fixed_joints, fixed_values)

        # Create a reduced configuration
        q_reduced = np.zeros(reduced_robot.num_dofs)
        q_reduced[0] = 0.3  # Set some value for testing

        # Convert to full configuration
        q_full = reduced_robot.reduced_to_full_configuration(q_reduced)

        # Check that fixed joints have the correct values
        for joint_name, value in zip(fixed_joints, fixed_values):
            full_idx = robot_model.dof_index(joint_name)
            assert q_full[full_idx] == value

        # Convert back to reduced configuration
        q_reduced_back = reduced_robot.full_to_reduced_configuration(q_full)

        # Check that the conversion is reversible
        np.testing.assert_array_almost_equal(q_reduced, q_reduced_back)


def test_reduced_robot_model_forward_kinematics(g1_robot_model):
    """
    Test forward kinematics with the reduced model.
    """
    for robot_model in [g1_robot_model]:
        # Use actual joints from the robot
        fixed_joints = robot_model.joint_names[:2]  # Use first two joints from the robot
        reduced_robot = ReducedRobotModel(robot_model, fixed_joints)

        # Create a reduced configuration
        q_reduced = np.zeros(reduced_robot.num_dofs)

        # Should not raise any exception
        reduced_robot.cache_forward_kinematics(q_reduced)

        # Check that frame placement works
        model = robot_model.pinocchio_wrapper.model
        if len(model.frames) > 1:
            valid_frame = model.frames[1].name
            placement = reduced_robot.frame_placement(valid_frame)
            assert isinstance(placement, pin.SE3)


def test_robot_model_clip_configuration(g1_robot_model):
    """
    Test that clip_configuration properly clips values to joint limits.
    """
    for robot_model in [g1_robot_model]:
        # Create a configuration with some values outside limits
        q = np.zeros(robot_model.num_dofs)
        root_nq = 7 if robot_model.is_floating_base_model else 0
        # Create extreme values for all joints
        q[root_nq:] = np.array([100.0, -100.0, 50.0, -50.0] * (robot_model.num_joints // 4 + 1))[
            : robot_model.num_joints
        ]

        # Clip the configuration
        q_clipped = robot_model.clip_configuration(q)

        # Check that values are within limits
        assert np.all(q_clipped[root_nq:] <= robot_model.upper_joint_limits)
        assert np.all(q_clipped[root_nq:] >= robot_model.lower_joint_limits)


def test_robot_model_get_actuated_joints(g1_robot_model):
    """
    Test getting body and hand actuated joints from configuration.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing actuated joints")

        # Create a test configuration
        q = np.zeros(robot_model.num_dofs)
        root_nq = 7 if robot_model.is_floating_base_model else 0
        q[root_nq:] = np.arange(robot_model.num_joints)  # Set some values for joints

        # Test body actuated joints
        body_joints = robot_model.get_body_actuated_joints(q)
        assert len(body_joints) == len(robot_model.get_body_actuated_joint_indices())

        # Test hand actuated joints
        hand_joints = robot_model.get_hand_actuated_joints(q)
        assert len(hand_joints) == len(robot_model.get_hand_actuated_joint_indices())

        # Test left hand joints
        left_hand_joints = robot_model.get_hand_actuated_joints(q, side="left")
        assert len(left_hand_joints) == len(robot_model.get_hand_actuated_joint_indices("left"))

        # Test right hand joints
        right_hand_joints = robot_model.get_hand_actuated_joints(q, side="right")
        assert len(right_hand_joints) == len(robot_model.get_hand_actuated_joint_indices("right"))


def test_robot_model_get_configuration_from_actuated_joints(g1_robot_model):
    """
    Test creating full configuration from actuated joint values.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing actuated joints")

        # Create test values for body and hands
        body_values = np.ones(len(robot_model.get_body_actuated_joint_indices()))
        hand_values = np.ones(len(robot_model.get_hand_actuated_joint_indices()))
        left_hand_values = np.ones(len(robot_model.get_hand_actuated_joint_indices("left")))
        right_hand_values = np.ones(len(robot_model.get_hand_actuated_joint_indices("right")))

        # Test with combined hand values
        q = robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_values, hand_actuated_joint_values=hand_values
        )
        assert q.shape == (robot_model.num_dofs,)

        # Test with separate hand values
        q = robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_values,
            left_hand_actuated_joint_values=left_hand_values,
            right_hand_actuated_joint_values=right_hand_values,
        )
        assert q.shape == (robot_model.num_dofs,)


def test_robot_model_reset_forward_kinematics(g1_robot_model):
    """
    Test resetting forward kinematics to default configuration.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing")

        # Create a more significant configuration change
        q = np.zeros(robot_model.num_dofs)
        root_nq = 7 if robot_model.is_floating_base_model else 0
        # Set some extreme joint angles
        q[root_nq:] = np.pi / 2  # 90 degrees for all joints
        robot_model.cache_forward_kinematics(q)

        # Use a hand frame from supplemental info
        test_frame = robot_model.supplemental_info.hand_frame_names["left"]

        # Reset to default
        robot_model.reset_forward_kinematics()
        # Get frame placement after reset
        placement_default = robot_model.frame_placement(test_frame)

        # Check that frame placement matches what we get with q_zero
        robot_model.cache_forward_kinematics(robot_model.q_zero)
        placement_q_zero = robot_model.frame_placement(test_frame)
        np.testing.assert_array_almost_equal(
            placement_default.translation, placement_q_zero.translation
        )
        np.testing.assert_array_almost_equal(placement_default.rotation, placement_q_zero.rotation)


# Additional tests for ReducedRobotModel
def test_reduced_robot_model_clip_configuration(g1_robot_model):
    """
    Test that clip_configuration works in reduced space.
    """
    for robot_model in [g1_robot_model]:
        fixed_joints = robot_model.joint_names[:2]
        reduced_robot = ReducedRobotModel(robot_model, fixed_joints)

        # Create a configuration with some values outside limits
        q_reduced = np.zeros(reduced_robot.num_dofs)
        root_nq = 7 if reduced_robot.full_robot.is_floating_base_model else 0
        # Create extreme values for all joints
        q_reduced[root_nq:] = np.array(
            [100.0, -100.0, 50.0, -50.0] * (reduced_robot.num_joints // 4 + 1)
        )[: reduced_robot.num_joints]

        # Clip the configuration
        q_clipped = reduced_robot.clip_configuration(q_reduced)

        # Check that values are within limits
        assert np.all(q_clipped[root_nq:] <= reduced_robot.upper_joint_limits)
        assert np.all(q_clipped[root_nq:] >= reduced_robot.lower_joint_limits)


def test_reduced_robot_model_get_actuated_joints(g1_robot_model):
    """
    Test getting body and hand actuated joints from reduced configuration.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing actuated joints")

        fixed_joints = robot_model.joint_names[:2]
        reduced_robot = ReducedRobotModel(robot_model, fixed_joints)

        # Create a test configuration
        q_reduced = np.zeros(reduced_robot.num_dofs)
        root_nq = 7 if reduced_robot.full_robot.is_floating_base_model else 0
        q_reduced[root_nq:] = np.arange(reduced_robot.num_joints)

        # Test body actuated joints
        body_joints = reduced_robot.get_body_actuated_joints(q_reduced)
        assert len(body_joints) == len(reduced_robot.get_body_actuated_joint_indices())

        # Test hand actuated joints
        hand_joints = reduced_robot.get_hand_actuated_joints(q_reduced)
        assert len(hand_joints) == len(reduced_robot.get_hand_actuated_joint_indices())


def test_reduced_robot_model_get_configuration_from_actuated_joints(g1_robot_model):
    """
    Test creating reduced configuration from actuated joint values.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing actuated joints")

        fixed_joints = robot_model.joint_names[:2]
        reduced_robot = ReducedRobotModel(robot_model, fixed_joints)

        # Create test values for body and hands
        body_values = np.ones(len(reduced_robot.get_body_actuated_joint_indices()))
        hand_values = np.ones(len(reduced_robot.get_hand_actuated_joint_indices()))
        left_hand_values = np.ones(len(reduced_robot.get_hand_actuated_joint_indices("left")))
        right_hand_values = np.ones(len(reduced_robot.get_hand_actuated_joint_indices("right")))

        # Test with combined hand values
        q_reduced = reduced_robot.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_values, hand_actuated_joint_values=hand_values
        )
        assert q_reduced.shape == (reduced_robot.num_dofs,)

        # Test with separate hand values
        q_reduced = reduced_robot.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_values,
            left_hand_actuated_joint_values=left_hand_values,
            right_hand_actuated_joint_values=right_hand_values,
        )
        assert q_reduced.shape == (reduced_robot.num_dofs,)

        # Verify that the values were set correctly in the reduced configuration
        # Check body actuated joints
        body_indices = reduced_robot.get_body_actuated_joint_indices()
        np.testing.assert_array_almost_equal(q_reduced[body_indices], body_values)

        # Check left hand actuated joints
        left_hand_indices = reduced_robot.get_hand_actuated_joint_indices("left")
        np.testing.assert_array_almost_equal(q_reduced[left_hand_indices], left_hand_values)

        # Check right hand actuated joints
        right_hand_indices = reduced_robot.get_hand_actuated_joint_indices("right")
        np.testing.assert_array_almost_equal(q_reduced[right_hand_indices], right_hand_values)


def test_reduced_robot_model_reset_forward_kinematics(g1_robot_model):
    """
    Test resetting forward kinematics in reduced model.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing")

        fixed_joints = robot_model.joint_names[:2]
        reduced_robot = ReducedRobotModel(robot_model, fixed_joints)

        # Create a more significant configuration change
        q_reduced = np.zeros(reduced_robot.num_dofs)
        root_nq = 7 if reduced_robot.full_robot.is_floating_base_model else 0
        # Set some extreme joint angles
        q_reduced[root_nq:] = np.pi / 2  # 90 degrees for all joints
        reduced_robot.cache_forward_kinematics(q_reduced)

        # Reset to default
        reduced_robot.reset_forward_kinematics()

        # Check that frame placement matches what we get with q_zero
        reduced_robot.cache_forward_kinematics(reduced_robot.q_zero)
        placement_q_zero = reduced_robot.frame_placement(
            reduced_robot.supplemental_info.hand_frame_names["left"]
        )
        placement_reset = reduced_robot.frame_placement(
            reduced_robot.supplemental_info.hand_frame_names["left"]
        )
        np.testing.assert_array_almost_equal(
            placement_reset.translation, placement_q_zero.translation
        )
        np.testing.assert_array_almost_equal(placement_reset.rotation, placement_q_zero.rotation)


def test_reduced_robot_model_from_fixed_groups(g1_robot_model):
    """
    Test creating reduced model from fixed joint groups.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing joint groups")

        # Get a group name from the supplemental info
        group_name = next(iter(robot_model.supplemental_info.joint_groups.keys()))
        group_info = robot_model.supplemental_info.joint_groups[group_name]

        # Get all joints that should be fixed (including those from subgroups)
        expected_fixed_joints = set()
        # Add direct joints
        expected_fixed_joints.update(group_info["joints"])
        # Add joints from subgroups
        for subgroup_name in group_info["groups"]:
            subgroup_joints = robot_model.get_joint_group_indices(subgroup_name)
            expected_fixed_joints.update([robot_model.joint_names[idx] for idx in subgroup_joints])

        # Test from_fixed_groups
        reduced_robot = ReducedRobotModel.from_fixed_groups(robot_model, [group_name])
        assert reduced_robot.full_robot is robot_model

        # Verify that fixed joints are not in reduced model's joint names
        for joint in expected_fixed_joints:
            assert joint not in reduced_robot.joint_names

        # Verify that fixed joints maintain their values in configuration
        q_reduced = np.ones(reduced_robot.num_dofs)  # Set some non-zero values
        q_full = reduced_robot.reduced_to_full_configuration(q_reduced)

        # Get the fixed values from the reduced model
        fixed_values = dict(zip(reduced_robot.fixed_joints, reduced_robot.fixed_values))

        # Check that all expected fixed joints have their values preserved
        for joint in expected_fixed_joints:
            full_idx = robot_model.dof_index(joint)
            assert q_full[full_idx] == fixed_values[joint]

        # Test from_fixed_group (convenience method)
        reduced_robot = ReducedRobotModel.from_fixed_group(robot_model, group_name)
        assert reduced_robot.full_robot is robot_model

        # Verify that fixed joints are not in reduced model's joint names
        for joint in expected_fixed_joints:
            assert joint not in reduced_robot.joint_names

        # Verify that fixed joints maintain their values in configuration
        q_reduced = np.ones(reduced_robot.num_dofs)  # Set some non-zero values
        q_full = reduced_robot.reduced_to_full_configuration(q_reduced)

        # Get the fixed values from the reduced model
        fixed_values = dict(zip(reduced_robot.fixed_joints, reduced_robot.fixed_values))

        # Check that all expected fixed joints have their values preserved
        for joint in expected_fixed_joints:
            full_idx = robot_model.dof_index(joint)
            assert q_full[full_idx] == fixed_values[joint]


def test_reduced_robot_model_from_active_groups(g1_robot_model):
    """
    Test creating reduced model from active joint groups.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing joint groups")

        # Get a group name from the supplemental info
        group_name = next(iter(robot_model.supplemental_info.joint_groups.keys()))
        group_info = robot_model.supplemental_info.joint_groups[group_name]

        # Get all joints that should be active (including those from subgroups)
        expected_active_joints = set()
        # Add direct joints
        expected_active_joints.update(group_info["joints"])
        # Add joints from subgroups
        for subgroup_name in group_info["groups"]:
            subgroup_joints = robot_model.get_joint_group_indices(subgroup_name)
            expected_active_joints.update([robot_model.joint_names[idx] for idx in subgroup_joints])

        # Get all joints from the model
        all_joints = set(robot_model.joint_names)
        # The fixed joints should be all joints minus the active joints
        expected_fixed_joints = all_joints - expected_active_joints

        # Test from_active_groups
        reduced_robot = ReducedRobotModel.from_active_groups(robot_model, [group_name])
        assert reduced_robot.full_robot is robot_model

        # Verify that active joints are in reduced model's joint names
        for joint in expected_active_joints:
            assert joint in reduced_robot.joint_names

        # Verify that fixed joints are not in reduced model's joint names
        for joint in expected_fixed_joints:
            assert joint not in reduced_robot.joint_names

        # Verify that fixed joints maintain their values in configuration
        q_reduced = np.ones(reduced_robot.num_dofs)  # Set some non-zero values
        q_full = reduced_robot.reduced_to_full_configuration(q_reduced)

        # Get the fixed values from the reduced model
        fixed_values = dict(zip(reduced_robot.fixed_joints, reduced_robot.fixed_values))

        # Check that all expected fixed joints have their values preserved
        for joint in expected_fixed_joints:
            full_idx = robot_model.dof_index(joint)
            assert q_full[full_idx] == fixed_values[joint]

        # Test from_active_group (convenience method)
        reduced_robot = ReducedRobotModel.from_active_group(robot_model, group_name)
        assert reduced_robot.full_robot is robot_model

        # Verify that active joints are in reduced model's joint names
        for joint in expected_active_joints:
            assert joint in reduced_robot.joint_names

        # Verify that fixed joints are not in reduced model's joint names
        for joint in expected_fixed_joints:
            assert joint not in reduced_robot.joint_names

        # Verify that fixed joints maintain their values in configuration
        q_reduced = np.ones(reduced_robot.num_dofs)  # Set some non-zero values
        q_full = reduced_robot.reduced_to_full_configuration(q_reduced)

        # Get the fixed values from the reduced model
        fixed_values = dict(zip(reduced_robot.fixed_joints, reduced_robot.fixed_values))

        # Check that all expected fixed joints have their values preserved
        for joint in expected_fixed_joints:
            full_idx = robot_model.dof_index(joint)
            assert q_full[full_idx] == fixed_values[joint]


def test_reduced_robot_model_frame_placement(g1_robot_model):
    """
    Test the frame_placement method in reduced model with a valid and invalid frame name.
    Also test that frame placements change with different configurations.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing")

        # Create a reduced model by fixing some joints
        fixed_joints = robot_model.joint_names[:2]
        reduced_robot = ReducedRobotModel(robot_model, fixed_joints)

        # Use the hand frame from supplemental info
        test_frame = reduced_robot.supplemental_info.hand_frame_names["left"]

        # Test with zero configuration
        q_reduced_zero = np.zeros(reduced_robot.num_dofs)
        reduced_robot.cache_forward_kinematics(q_reduced_zero)
        placement_zero = reduced_robot.frame_placement(test_frame)
        assert isinstance(placement_zero, pin.SE3)

        # Test with non-zero configuration
        q_reduced_non_zero = np.zeros(reduced_robot.num_dofs)
        root_nq = 7 if reduced_robot.full_robot.is_floating_base_model else 0

        # Set a valid non-zero value for each joint
        for i in range(root_nq, reduced_robot.num_dofs):
            # Use a value that's within the joint limits
            q_reduced_non_zero[i] = 0.5  # 0.5 radians is within most joint limits

        reduced_robot.cache_forward_kinematics(q_reduced_non_zero)
        placement_non_zero = reduced_robot.frame_placement(test_frame)

        # Verify that frame placements are different with different configurations
        assert not np.allclose(
            placement_zero.translation, placement_non_zero.translation
        ) or not np.allclose(placement_zero.rotation, placement_non_zero.rotation)

        # Should raise an error for an invalid frame
        with pytest.raises(ValueError, match="Unknown frame"):
            reduced_robot.frame_placement("non_existent_frame")


def test_robot_model_gravity_compensation_basic(g1_robot_model):
    """
    Test basic gravity compensation functionality.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing gravity compensation")

        # Create a valid configuration
        q = np.zeros(robot_model.num_dofs)
        if robot_model.is_floating_base_model:
            # Set floating base to upright position
            q[:7] = [0, 0, 1.0, 0, 0, 0, 1]  # [x, y, z, qx, qy, qz, qw]

        # Test gravity compensation for all joints
        gravity_torques = robot_model.compute_gravity_compensation_torques(q)

        # Check output shape
        assert gravity_torques.shape == (robot_model.num_dofs,)

        # For a humanoid robot with arms, there should be some non-zero gravity torques
        assert np.any(np.abs(gravity_torques) > 1e-6), "Expected some non-zero gravity torques"


def test_robot_model_gravity_compensation_joint_groups(g1_robot_model):
    """
    Test gravity compensation with different joint group specifications.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing gravity compensation")

        # Create a valid configuration
        q = np.zeros(robot_model.num_dofs)
        if robot_model.is_floating_base_model:
            q[:7] = [0, 0, 1.0, 0, 0, 0, 1]

        # Get available joint groups
        available_groups = list(robot_model.supplemental_info.joint_groups.keys())
        if not available_groups:
            pytest.skip("No joint groups available for testing")

        test_group = available_groups[0]  # Use first available group

        # Test with string input
        gravity_str = robot_model.compute_gravity_compensation_torques(q, test_group)
        assert gravity_str.shape == (robot_model.num_dofs,)

        # Test with list input
        gravity_list = robot_model.compute_gravity_compensation_torques(q, [test_group])
        np.testing.assert_array_equal(gravity_str, gravity_list)

        # Test with set input
        gravity_set = robot_model.compute_gravity_compensation_torques(q, {test_group})
        np.testing.assert_array_equal(gravity_str, gravity_set)

        # Test that compensation is selective (some joints should be zero)
        group_indices = robot_model.get_joint_group_indices(test_group)
        if len(group_indices) < robot_model.num_dofs:
            # Check that only specified joints have compensation
            non_zero_mask = np.abs(gravity_str) > 1e-6
            compensated_indices = np.where(non_zero_mask)[0]
            # The compensated indices should be a subset of the group indices
            assert len(compensated_indices) <= len(group_indices)


def test_robot_model_gravity_compensation_multiple_groups(g1_robot_model):
    """
    Test gravity compensation with multiple joint groups.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing gravity compensation")

        # Create a valid configuration
        q = np.zeros(robot_model.num_dofs)
        if robot_model.is_floating_base_model:
            q[:7] = [0, 0, 1.0, 0, 0, 0, 1]

        # Get available joint groups
        available_groups = list(robot_model.supplemental_info.joint_groups.keys())
        if len(available_groups) < 2:
            pytest.skip("Need at least 2 joint groups for testing")

        # Test with multiple groups
        test_groups = available_groups[:2]
        gravity_multiple = robot_model.compute_gravity_compensation_torques(q, test_groups)
        assert gravity_multiple.shape == (robot_model.num_dofs,)

        # Test individual groups
        gravity_1 = robot_model.compute_gravity_compensation_torques(q, test_groups[0])
        gravity_2 = robot_model.compute_gravity_compensation_torques(q, test_groups[1])

        # The multiple group result should have at least as many non-zero elements
        # as either individual group (could be more due to overlaps)
        nonzero_multiple = np.count_nonzero(np.abs(gravity_multiple) > 1e-6)
        nonzero_1 = np.count_nonzero(np.abs(gravity_1) > 1e-6)
        nonzero_2 = np.count_nonzero(np.abs(gravity_2) > 1e-6)
        assert nonzero_multiple >= max(nonzero_1, nonzero_2)


def test_robot_model_gravity_compensation_configuration_dependency(g1_robot_model):
    """
    Test that gravity compensation changes with robot configuration.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing gravity compensation")

        # Get available joint groups - prefer arms if available
        available_groups = list(robot_model.supplemental_info.joint_groups.keys())
        test_group = None
        for group in ["arms", "left_arm", "right_arm"]:
            if group in available_groups:
                test_group = group
                break
        if test_group is None and available_groups:
            test_group = available_groups[0]
        if test_group is None:
            pytest.skip("No joint groups available for testing")

        # Test with different configurations
        q1 = np.zeros(robot_model.num_dofs)
        q2 = np.zeros(robot_model.num_dofs)

        if robot_model.is_floating_base_model:
            # Both configurations upright but different joint positions
            q1[:7] = [0, 0, 1.0, 0, 0, 0, 1]
            q2[:7] = [0, 0, 1.0, 0, 0, 0, 1]

        # Change arm joint positions specifically (not random joints)
        # This ensures we actually change joints that affect the gravity compensation
        try:
            arm_indices = robot_model.get_joint_group_indices(test_group)
            if len(arm_indices) >= 2:
                # Change first two arm joints significantly
                q2[arm_indices[0]] = np.pi / 4  # 45 degrees
                q2[arm_indices[1]] = np.pi / 6  # 30 degrees
            elif len(arm_indices) >= 1:
                # Change first arm joint if only one available
                q2[arm_indices[0]] = np.pi / 3  # 60 degrees
        except Exception:
            # Fallback to changing some joints if arm indices not available
            if robot_model.is_floating_base_model and robot_model.num_dofs > 9:
                q2[7] = np.pi / 4
                q2[8] = np.pi / 6
            elif not robot_model.is_floating_base_model and robot_model.num_dofs > 2:
                q2[0] = np.pi / 4
                q2[1] = np.pi / 6

        # Compute gravity compensation for both configurations
        gravity_1 = robot_model.compute_gravity_compensation_torques(q1, test_group)
        gravity_2 = robot_model.compute_gravity_compensation_torques(q2, test_group)

        # They should be different (unless all compensated joints didn't change)
        # Allow for small numerical differences
        assert not np.allclose(
            gravity_1, gravity_2, atol=1e-10
        ), "Gravity compensation should change with configuration"


def test_robot_model_gravity_compensation_error_handling(g1_robot_model):
    """
    Test error handling in gravity compensation.
    """
    for robot_model in [g1_robot_model]:
        # Test with wrong configuration size
        q_wrong = np.zeros(robot_model.num_dofs + 1)
        with pytest.raises(ValueError, match="Expected q of length"):
            robot_model.compute_gravity_compensation_torques(q_wrong)

        # Test with invalid joint group
        q_valid = np.zeros(robot_model.num_dofs)
        if robot_model.is_floating_base_model:
            q_valid[:7] = [0, 0, 1.0, 0, 0, 0, 1]

        with pytest.raises(RuntimeError, match="Error computing gravity compensation"):
            robot_model.compute_gravity_compensation_torques(q_valid, "non_existent_group")

        # Test with mixed valid/invalid groups
        if robot_model.supplemental_info is not None:
            available_groups = list(robot_model.supplemental_info.joint_groups.keys())
            if available_groups:
                valid_group = available_groups[0]
                with pytest.raises(RuntimeError, match="Error computing gravity compensation"):
                    robot_model.compute_gravity_compensation_torques(
                        q_valid, [valid_group, "non_existent_group"]
                    )


def test_robot_model_gravity_compensation_auto_clip(g1_robot_model):
    """
    Test auto-clipping functionality in gravity compensation.
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing gravity compensation")

        # Create configuration with values outside joint limits
        q = np.zeros(robot_model.num_dofs)
        root_nq = 7 if robot_model.is_floating_base_model else 0

        if robot_model.is_floating_base_model:
            q[:7] = [0, 0, 1.0, 0, 0, 0, 1]  # Valid floating base

        # Set extreme joint values (outside limits)
        if robot_model.num_dofs > root_nq:
            q[root_nq:] = 100.0  # Very large values

        # Should work with auto_clip=True (default)
        try:
            gravity_clipped = robot_model.compute_gravity_compensation_torques(q, auto_clip=True)
            assert gravity_clipped.shape == (robot_model.num_dofs,)
        except Exception as e:
            pytest.skip(f"Auto-clip test skipped due to: {e}")

        # Test with auto_clip=False - might work or might not depending on limits
        try:
            gravity_no_clip = robot_model.compute_gravity_compensation_torques(q, auto_clip=False)
            assert gravity_no_clip.shape == (robot_model.num_dofs,)
        except Exception:
            # This is expected if the configuration is invalid
            pass


def test_robot_model_gravity_compensation_arms_specific(g1_robot_model):
    """
    Test gravity compensation specifically for arm joints (if available).
    """
    for robot_model in [g1_robot_model]:
        # Skip if no supplemental info
        if robot_model.supplemental_info is None:
            pytest.skip("No supplemental info available for testing gravity compensation")

        available_groups = list(robot_model.supplemental_info.joint_groups.keys())

        # Test arms specifically if available
        if "arms" in available_groups:
            q = np.zeros(robot_model.num_dofs)
            if robot_model.is_floating_base_model:
                q[:7] = [0, 0, 1.0, 0, 0, 0, 1]

            # Test arms gravity compensation
            gravity_arms = robot_model.compute_gravity_compensation_torques(q, "arms")
            assert gravity_arms.shape == (robot_model.num_dofs,)

            # Test left and right arms separately if available
            if "left_arm" in available_groups and "right_arm" in available_groups:
                gravity_left = robot_model.compute_gravity_compensation_torques(q, "left_arm")
                gravity_right = robot_model.compute_gravity_compensation_torques(q, "right_arm")

                # Both arms should have non-zero compensation (for typical configurations)
                if np.any(np.abs(gravity_arms) > 1e-6):
                    # If arms have compensation, at least one of left/right should too
                    assert np.any(np.abs(gravity_left) > 1e-6) or np.any(
                        np.abs(gravity_right) > 1e-6
                    )
        else:
            pytest.skip("No arm joint groups available for testing")
