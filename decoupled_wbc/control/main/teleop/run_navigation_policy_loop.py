import threading
import time

import rclpy

from decoupled_wbc.control.main.constants import NAV_CMD_TOPIC
from decoupled_wbc.control.policy.keyboard_navigation_policy import KeyboardNavigationPolicy
from decoupled_wbc.control.utils.keyboard_dispatcher import KeyboardListenerSubscriber
from decoupled_wbc.control.utils.ros_utils import ROSMsgPublisher

FREQUENCY = 10
NAV_NODE_NAME = "NavigationPolicy"


def main():
    rclpy.init(args=None)
    node = rclpy.create_node(NAV_NODE_NAME)

    # Start ROS spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    time.sleep(0.5)

    dict_publisher = ROSMsgPublisher(NAV_CMD_TOPIC)
    keyboard_listener = KeyboardListenerSubscriber()

    # Initialize navigation policy
    navigation_policy = KeyboardNavigationPolicy()

    # Create rate controller
    rate = node.create_rate(FREQUENCY)

    try:
        while rclpy.ok():
            t_now = time.monotonic()
            # get keyboard input

            navigation_policy.handle_keyboard_button(keyboard_listener.read_msg())
            # Get action from navigation policy
            action = navigation_policy.get_action(time=t_now)

            # Add timestamp to the data
            action["timestamp"] = t_now

            # Create and publish ByteMultiArray message
            dict_publisher.publish(action)

            # Print status periodically (optional)
            if int(t_now * 10) % 10 == 0:
                nav_cmd = action["navigate_cmd"]
                node.get_logger().info(
                    f"Nav cmd: linear=({nav_cmd[0]:.2f}, {nav_cmd[1]:.2f}), "
                    f"angular={nav_cmd[2]:.2f}"
                )

            rate.sleep()

    except KeyboardInterrupt:
        print("Navigation control loop terminated by user")

    finally:
        # Clean shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
