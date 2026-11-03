import numpy as np
import pytest

from decoupled_wbc.control.policy.interpolation_policy import (
    InterpolationPolicy,
)


def test_trajectory_interpolation():
    """
    Test that the InterpolationPolicy correctly interpolates between waypoints.

    Initial pose is at all zeros.
    At t=4sec, the index 27 position (right_shoulder_yaw_joint) should be -1.5.
    We run at 100Hz to see all intermediate waypoints.

    Notes:
        - The trajectory data is at 'trajectory_data.npy' in the current directory
        - The visualization is at 'trajectory.png' in the current directory
    """
    # Create a pose with 32 joints (all zeros initially)
    num_joints = 32
    initial_pose = np.zeros(num_joints)

    # Initial time (use a fixed value for reproducibility)
    initial_time = 0.0

    # Create the wrapper with initial pose
    interpolator = InterpolationPolicy(
        init_time=initial_time,
        init_values={"target_pose": initial_pose},
        max_change_rate=np.inf,
    )

    # Target pose: all zeros except index 27 which should be -1.5
    target_pose = np.zeros(num_joints)
    target_pose[27] = -1.5  # right_shoulder_yaw_joint
    target_time = 4.0  # 4 seconds from now

    # Set the planner command to schedule the waypoint
    interpolator.set_goal(
        {
            "target_pose": target_pose,
            "target_time": target_time,
            "interpolation_garbage_collection_time": initial_time,
        }
    )

    # Sample the trajectory at 100Hz
    frequency = 100
    dt = 1.0 / frequency
    sample_times = np.arange(initial_time, target_time + dt, dt)

    # Collect the interpolated poses
    sampled_poses = []
    for t in sample_times:
        action = interpolator.get_action(t)
        sampled_poses.append(action["target_pose"])

    # Convert to numpy array for easier analysis
    sampled_poses = np.array(sampled_poses)

    # Check specific requirements
    # Verify we actually moved from 0 to -1.5
    joint_27_positions = sampled_poses[:, 27]
    assert joint_27_positions[0] == pytest.approx(0.0)
    assert joint_27_positions[-1] == pytest.approx(-1.5)

    # Calculate the absolute changes between each step
    changes = np.abs(np.diff(joint_27_positions))
    assert np.all(changes < 0.004), "Joint 27 position should change by less than 0.004"

    # Print some statistics about the trajectory
    print(f"Total time steps: {len(sample_times)}")
    print(f"Joint 27 trajectory start: {joint_27_positions[0]}")
    print(f"Joint 27 trajectory end: {joint_27_positions[-1]}")
    print(f"Joint 27 max velocity: {np.max(np.abs(np.diff(joint_27_positions) / dt))}")
    print(f"Max velocity timestep: {np.argmax(np.abs(np.diff(joint_27_positions) / dt))}")
