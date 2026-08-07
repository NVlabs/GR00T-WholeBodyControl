from decoupled_wbc.control.main.teleop.configs.configs import SyncSimDataCollectionConfig
from decoupled_wbc.control.main.teleop.run_sync_sim_data_collection import (
    main as data_collection_main,
)


def test_sim_data_collection_unit(robot_name="G1", task_name="GroundOnly"):
    """
    Fast CI unit test for simulation data collection (50 steps, no tracking checks).

    This test validates that:
    1. Data collection completes successfully
    2. Upper body joints are moving (velocity check)

    Note: This test runs for only 50 steps and does not perform end effector tracking validation
    for faster CI execution.
    """
    config = SyncSimDataCollectionConfig()
    config.robot = robot_name
    config.task_name = task_name
    config.enable_visualization = False
    config.enable_real_device = False
    config.enable_onscreen = False
    config.save_img_obs = True
    config.ci_test = True
    config.ci_test_mode = "unit"
    config.replay_data_path = "decoupled_wbc/tests/replay_data/all_joints_raw_data_replay.pkl"
    config.remove_existing_dir = True
    config.enable_gravity_compensation = True
    res = data_collection_main(config)
    assert res, "Data collection did not pass for all datasets"


def test_sim_data_collection_pre_merge(robot_name="G1", task_name="GroundOnly"):
    """
    Pre-merge test for simulation data collection with end effector tracking validation (500 steps).

    This test validates that:
    1. Data collection completes successfully
    2. Upper body joints are moving (velocity check)
    3. End effector tracking error is within thresholds:
    - G1 robots:
        Max position error < 7cm (0.07m), Max rotation error < 17°,
        Average position error < 5cm (0.05m), Average rotation error < 12°
    """
    config = SyncSimDataCollectionConfig()
    config.robot = robot_name
    config.task_name = task_name
    config.enable_visualization = False
    config.enable_real_device = False
    config.enable_onscreen = False
    config.save_img_obs = True
    config.ci_test = True
    config.ci_test_mode = "pre_merge"
    config.replay_data_path = "decoupled_wbc/tests/replay_data/all_joints_raw_data_replay.pkl"
    config.remove_existing_dir = True
    config.enable_gravity_compensation = True
    res = data_collection_main(config)
    assert res, "Data collection did not pass for all datasets"


if __name__ == "__main__":
    # Run unit tests for fast CI
    test_sim_data_collection_unit("G1", "GroundOnly")
