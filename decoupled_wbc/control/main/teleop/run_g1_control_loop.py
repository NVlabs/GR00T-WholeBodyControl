from copy import deepcopy
import time

import tyro

from decoupled_wbc.control.envs.g1.g1_env import G1Env
from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    DEFAULT_WRIST_POSE,
    JOINT_SAFETY_STATUS_TOPIC,
    LOWER_BODY_POLICY_STATUS_TOPIC,
    ROBOT_CONFIG_TOPIC,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from decoupled_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from decoupled_wbc.control.utils.keyboard_dispatcher import (
    KeyboardDispatcher,
    KeyboardEStop,
    KeyboardListenerPublisher,
    ROSKeyboardDispatcher,
)
from decoupled_wbc.control.utils.ros_utils import (
    ROSManager,
    ROSMsgPublisher,
    ROSMsgSubscriber,
    ROSServiceServer,
)
from decoupled_wbc.control.utils.telemetry import Telemetry

CONTROL_NODE_NAME = "ControlPolicy"


def main(config: ControlLoopConfig):
    ros_manager = ROSManager(node_name=CONTROL_NODE_NAME)
    node = ros_manager.node

    # start the robot config server
    ROSServiceServer(ROBOT_CONFIG_TOPIC, config.to_dict())

    wbc_config = config.load_wbc_yaml()

    data_exp_pub = ROSMsgPublisher(STATE_TOPIC_NAME)
    lower_body_policy_status_pub = ROSMsgPublisher(LOWER_BODY_POLICY_STATUS_TOPIC)
    joint_safety_status_pub = ROSMsgPublisher(JOINT_SAFETY_STATUS_TOPIC)

    # Initialize telemetry
    telemetry = Telemetry(window_size=100)

    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
    )

    env = G1Env(
        env_name=config.env_name,
        robot_model=robot_model,
        config=wbc_config,
        wbc_version=config.wbc_version,
    )
    if env.sim and not config.sim_sync_mode:
        env.start_simulator()

    wbc_policy = get_wbc_policy("g1", robot_model, wbc_config, config.upper_body_joint_speed)

    keyboard_listener_pub = KeyboardListenerPublisher()
    keyboard_estop = KeyboardEStop()
    if config.keyboard_dispatcher_type == "raw":
        dispatcher = KeyboardDispatcher()
    elif config.keyboard_dispatcher_type == "ros":
        dispatcher = ROSKeyboardDispatcher()
    else:
        raise ValueError(
            f"Invalid keyboard dispatcher: {config.keyboard_dispatcher_type}, please use 'raw' or 'ros'"
        )
    dispatcher.register(env)
    dispatcher.register(wbc_policy)
    dispatcher.register(keyboard_listener_pub)
    dispatcher.register(keyboard_estop)
    dispatcher.start()

    rate = node.create_rate(config.control_frequency)

    upper_body_policy_subscriber = ROSMsgSubscriber(CONTROL_GOAL_TOPIC)

    last_teleop_cmd = None
    try:
        while ros_manager.ok():
            t_start = time.monotonic()
            with telemetry.timer("total_loop"):
                # Step simulator if in sync mode
                with telemetry.timer("step_simulator"):
                    if env.sim and config.sim_sync_mode:
                        env.step_simulator()

                # Measure observation time
                with telemetry.timer("observe"):
                    obs = env.observe()
                    wbc_policy.set_observation(obs)

                # Measure policy setup time
                with telemetry.timer("policy_setup"):
                    upper_body_cmd = upper_body_policy_subscriber.get_msg()

                    t_now = time.monotonic()

                    wbc_goal = {}
                    if upper_body_cmd:
                        wbc_goal = upper_body_cmd.copy()
                        last_teleop_cmd = upper_body_cmd.copy()
                        if config.ik_indicator:
                            env.set_ik_indicator(upper_body_cmd)
                    # Send goal to policy
                    if wbc_goal:
                        wbc_goal["interpolation_garbage_collection_time"] = t_now - 2 * (
                            1 / config.control_frequency
                        )
                        wbc_policy.set_goal(wbc_goal)

                # Measure policy action calculation time
                with telemetry.timer("policy_action"):
                    wbc_action = wbc_policy.get_action(time=t_now)

                # Measure action queue time
                with telemetry.timer("queue_action"):
                    env.queue_action(wbc_action)

                # Publish status information for InteractiveModeController
                with telemetry.timer("publish_status"):
                    # Get policy status - check if the lower body policy has use_policy_action enabled
                    policy_use_action = False
                    try:
                        # Access the lower body policy through the decoupled whole body policy
                        if hasattr(wbc_policy, "lower_body_policy"):
                            policy_use_action = getattr(
                                wbc_policy.lower_body_policy, "use_policy_action", False
                            )
                    except (AttributeError, TypeError):
                        policy_use_action = False

                    policy_status_msg = {"use_policy_action": policy_use_action, "timestamp": t_now}
                    lower_body_policy_status_pub.publish(policy_status_msg)

                    # Get joint safety status from G1Env (which already runs the safety monitor)
                    joint_safety_ok = env.get_joint_safety_status()

                    joint_safety_status_msg = {
                        "joint_safety_ok": joint_safety_ok,
                        "timestamp": t_now,
                    }
                    joint_safety_status_pub.publish(joint_safety_status_msg)

                # Start or Stop data collection
                if wbc_goal.get("toggle_data_collection", False):
                    dispatcher.handle_key("c")

                # Abort the current episode
                if wbc_goal.get("toggle_data_abort", False):
                    dispatcher.handle_key("x")

                if env.use_sim and wbc_goal.get("reset_env_and_policy", False):
                    print("Resetting sim environment and policy")
                    # Reset teleop policy & sim env
                    dispatcher.handle_key("k")

                    # Clear upper body commands
                    upper_body_policy_subscriber._msg = None
                    upper_body_cmd = {
                        "target_upper_body_pose": obs["q"][
                            robot_model.get_joint_group_indices("upper_body")
                        ],
                        "wrist_pose": DEFAULT_WRIST_POSE,
                        "base_height_command": DEFAULT_BASE_HEIGHT,
                        "navigate_cmd": DEFAULT_NAV_CMD,
                    }
                    last_teleop_cmd = upper_body_cmd.copy()

                    time.sleep(0.5)

                msg = deepcopy(obs)
                for key in obs.keys():
                    if key.endswith("_image"):
                        del msg[key]

                # exporting data
                if last_teleop_cmd:
                    msg.update(
                        {
                            "action": wbc_action["q"],
                            "action.eef": last_teleop_cmd.get("wrist_pose", DEFAULT_WRIST_POSE),
                            "base_height_command": last_teleop_cmd.get(
                                "base_height_command", DEFAULT_BASE_HEIGHT
                            ),
                            "navigate_command": last_teleop_cmd.get(
                                "navigate_cmd", DEFAULT_NAV_CMD
                            ),
                            "timestamps": {
                                "main_loop": time.time(),
                                "proprio": time.time(),
                            },
                        }
                    )
                data_exp_pub.publish(msg)
                end_time = time.monotonic()

            if env.sim and (not env.sim.sim_thread or not env.sim.sim_thread.is_alive()):
                raise RuntimeError("Simulator thread is not alive")

            rate.sleep()

            # Log timing information every 100 iterations (roughly every 2 seconds at 50Hz)
            if config.verbose_timing:
                # When verbose timing is enabled, always show timing
                telemetry.log_timing_info(context="G1 Control Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.control_frequency) and not config.sim_sync_mode:
                # Only show timing when loop is slow and verbose_timing is disabled
                telemetry.log_timing_info(context="G1 Control Loop Missed", threshold=0.001)

    except ros_manager.exceptions() as e:
        print(f"ROSManager interrupted by user: {e}")
    finally:
        print("Cleaning up...")
        # the order of the following is important
        dispatcher.stop()
        ros_manager.shutdown()
        env.close()


if __name__ == "__main__":
    config = tyro.cli(ControlLoopConfig)
    main(config)
