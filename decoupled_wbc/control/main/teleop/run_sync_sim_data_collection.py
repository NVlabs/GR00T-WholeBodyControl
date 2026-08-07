from pathlib import Path
import time

import tyro

from decoupled_wbc.control.main.teleop.configs.configs import SyncSimDataCollectionConfig
from decoupled_wbc.control.robot_model.instantiation import get_robot_type_and_model
from decoupled_wbc.control.utils.keyboard_dispatcher import (
    KeyboardDispatcher,
    KeyboardListener,
)
from decoupled_wbc.control.utils.ros_utils import ROSManager
from decoupled_wbc.control.utils.sync_sim_utils import (
    COLLECTION_KEY,
    SKIP_KEY,
    CITestManager,
    EpisodeManager,
    generate_frame,
    get_data_exporter,
    get_env,
    get_policies,
)
from decoupled_wbc.control.utils.telemetry import Telemetry

CONTROL_NODE_NAME = "ControlPolicy"


CONTROL_CMD_TOPIC = CONTROL_NODE_NAME + "/q_target"
ENV_NODE_NAME = "SyncEnv"
ENV_OBS_TOPIC = ENV_NODE_NAME + "/obs"


def display_controls(config: SyncSimDataCollectionConfig):
    """
    Method to pretty print controls.
    """

    def print_command(char, info):
        char += " " * (30 - len(char))
        print("{}\t{}".format(char, info))

    print("")
    print_command("Keys", "Command")
    if config.manual_control:
        print_command(COLLECTION_KEY, "start/stop data collection")
    print_command(SKIP_KEY, "skip and collect new episodes")
    print_command("w-s-a-d", "move horizontally in x-y plane (press '=' first to enable)")
    print_command("q", "rotate (counter-clockwise)")
    print_command("e", "rotate (clockwise)")
    print_command("space", "reset all velocity to zero")
    print("")


def main(config: SyncSimDataCollectionConfig):
    ros_manager = ROSManager(node_name=CONTROL_NODE_NAME)
    node = ros_manager.node

    # Initialize telemetry
    telemetry = Telemetry(window_size=100)

    # Initialize robot model
    robot_type, robot_model = get_robot_type_and_model(
        config.robot,
        enable_waist_ik=config.enable_waist,
    )

    # Initialize sim env
    env = get_env(config, onscreen=config.enable_onscreen, offscreen=config.save_img_obs)
    seed = int(time.time())
    env.reset(seed)
    env.render()
    obs = env.observe()
    robot_model.set_initial_body_pose(obs["q"])

    # Initialize data exporter
    exporter = get_data_exporter(
        config,
        obs,
        robot_model,
        save_path=Path("./outputs/ci_test/") if config.ci_test else None,
    )

    # Display control signals
    display_controls(config)

    # Initialize policies
    wbc_policy, teleop_policy = get_policies(config, robot_type, robot_model)

    dispatcher = KeyboardDispatcher()
    keyboard_listener = KeyboardListener()  # for data collection keys
    dispatcher.register(keyboard_listener)
    dispatcher.register(wbc_policy)
    dispatcher.register(teleop_policy)

    dispatcher.start()

    rate = node.create_rate(config.control_frequency)

    # Initialize episode manager to handle state transitions and data collection
    episode_manager = EpisodeManager(config)

    # Initialize CI test manager
    ci_test_manager = CITestManager(config) if config.ci_test else None

    try:
        while ros_manager.ok():

            need_reset = False
            keyboard_input = keyboard_listener.pop_key()

            with telemetry.timer("total_loop"):
                max_mujoco_state_len, mujoco_state_len, mujoco_state = env.get_mujoco_state_info()

                # Measure observation time
                with telemetry.timer("observe"):
                    obs = env.observe()
                    wbc_policy.set_observation(obs)

                # Measure policy setup time
                with telemetry.timer("policy_setup"):
                    teleop_cmd = teleop_policy.get_action()

                    wbc_goal = {}

                    # Note that wbc_goal["navigation_cmd'] could be overwritten by teleop_cmd
                    if teleop_cmd:
                        for key, value in teleop_cmd.items():
                            wbc_goal[key] = value
                        # Draw IK indicators
                        if config.ik_indicator:
                            env.set_ik_indicator(teleop_cmd)
                    if wbc_goal:
                        wbc_policy.set_goal(wbc_goal)

                # Measure policy action calculation time
                with telemetry.timer("policy_action"):
                    wbc_action = wbc_policy.get_action()

                if config.ci_test:
                    ci_test_manager.check_upper_body_motion(robot_model, wbc_action, config)

                # Measure action queue time
                with telemetry.timer("step"):
                    obs, _, _, _, step_info = env.step(wbc_action)
                    env.render()
                    episode_manager.increment_step()

                if config.ci_test and config.ci_test_mode == "pre_merge":
                    ci_test_manager.check_end_effector_tracking(
                        teleop_cmd, obs, config, episode_manager.get_step_count()
                    )

                # Handle data collection trigger
                episode_manager.handle_collection_trigger(wbc_goal, keyboard_input, step_info)

                # Collect data frame
                if episode_manager.should_collect_data():
                    frame = generate_frame(
                        obs,
                        wbc_action,
                        seed,
                        mujoco_state,
                        mujoco_state_len,
                        max_mujoco_state_len,
                        teleop_cmd,
                        wbc_goal,
                        config.save_img_obs,
                    )
                    # exporting data
                    exporter.add_frame(frame)

                    # if done and task_completion_hold_count is 0, save the episode
                    need_reset = episode_manager.check_export_and_completion(exporter)

                # check data abort
                if episode_manager.handle_skip(wbc_goal, keyboard_input, exporter):
                    need_reset = True

                if need_reset:
                    if config.ci_test:
                        print("CI test: Completed...")
                        raise KeyboardInterrupt

                    seed = int(time.time())
                    env.reset(seed)
                    env.render()

                    print("Sleeping for 3 seconds before resetting teleop policy...")
                    for j in range(3, 0, -1):
                        print(f"Starting in {j}...")
                        time.sleep(1)

                    wbc_policy, teleop_policy = get_policies(config, robot_type, robot_model)
                    episode_manager.reset_step_count()

            rate.sleep()

    except ros_manager.exceptions() as e:
        print(f"ROSManager interrupted by user: {e}")
    finally:
        # Cleanup resources
        teleop_policy.close()
        dispatcher.stop()
        ros_manager.shutdown()
        env.close()
        print("Sync sim data collection loop terminated.")

    return True


if __name__ == "__main__":
    config = tyro.cli(SyncSimDataCollectionConfig)
    main(config)
