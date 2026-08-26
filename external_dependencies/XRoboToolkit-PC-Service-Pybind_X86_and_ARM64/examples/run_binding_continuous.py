import sys
import time

import xrobotoolkit_sdk as xrt


def run_tests():
    print("Starting Python binding test...")

    try:
        print("Initializing SDK...")
        xrt.init()
        print("SDK Initialized successfully.")

        print("\n--- Testing all functions for 10 iterations ---")
        for i in range(100):
            print(f"\n--- Iteration {i+1} ---")

            # Poses
            left_pose = xrt.get_left_controller_pose()
            right_pose = xrt.get_right_controller_pose()
            headset_pose = xrt.get_headset_pose()
            print(f"Left Controller Pose: {left_pose}")
            print(f"Right Controller Pose: {right_pose}")
            print(f"Headset Pose: {headset_pose}")

            # Triggers
            left_trigger = xrt.get_left_trigger()
            right_trigger = xrt.get_right_trigger()
            print(f"Left Trigger: {left_trigger}")
            print(f"Right Trigger: {right_trigger}")

            # Grips
            left_grip = xrt.get_left_grip()
            right_grip = xrt.get_right_grip()
            print(f"Left Grip: {left_grip}")
            print(f"Right Grip: {right_grip}")

            # Menu Buttons
            left_menu = xrt.get_left_menu_button()
            right_menu = xrt.get_right_menu_button()
            print(f"Left Menu Button: {left_menu}")
            print(f"Right Menu Button: {right_menu}")

            # Axis Clicks
            left_axis_click = xrt.get_left_axis_click()
            right_axis_click = xrt.get_right_axis_click()
            print(f"Left Axis Click: {left_axis_click}")
            print(f"Right Axis Click: {right_axis_click}")

            # Axes
            left_axis = xrt.get_left_axis()
            right_axis = xrt.get_right_axis()
            print(f"Left Axis (X, Y): {left_axis}")
            print(f"Right Axis (X, Y): {right_axis}")

            # Primary Buttons (X, A)
            x_button = xrt.get_X_button()  # Left Primary
            a_button = xrt.get_A_button()  # Right Primary
            print(f"X Button (Left Primary): {x_button}")
            print(f"A Button (Right Primary): {a_button}")

            # Secondary Buttons (Y, B)
            y_button = xrt.get_Y_button()  # Left Secondary
            b_button = xrt.get_B_button()  # Right Secondary
            print(f"Y Button (Left Secondary): {y_button}")
            print(f"B Button (Right Secondary): {b_button}")

            # Left Hand State
            left_hand_state = xrt.get_left_hand_tracking_state()
            print(f"Left Hand State: {left_hand_state}")
            # Right Hand State
            right_hand_state = xrt.get_right_hand_tracking_state()
            print(f"Right Hand State: {right_hand_state}")

            # Timestamp
            timestamp = xrt.get_time_stamp_ns()
            print(f"Timestamp (ns): {timestamp}")

            num_motion_data = xrt.num_motion_data_available()
            print(f"Number of Motion Trackers: {num_motion_data}")
            if num_motion_data > 0:
                motion_tracker_pose = xrt.get_motion_tracker_pose()
                motion_tracker_velocity = xrt.get_motion_tracker_velocity()
                motion_tracker_acceleration = xrt.get_motion_tracker_acceleration()
                motion_tracker_serial_numbers = xrt.get_motion_tracker_serial_numbers()
                motion_timestamp_ns = xrt.get_motion_timestamp_ns()

                print(f"Motion Tracker Pose: {motion_tracker_pose}")
                print(f"Motion Tracker Velocity: {motion_tracker_velocity}")
                print(f"Motion Tracker Acceleration: {motion_tracker_acceleration}")
                print(f"Motion Tracker Serial Numbers: {motion_tracker_serial_numbers}")
                print(f"Motion Timestamp (ns): {motion_timestamp_ns}")

            time.sleep(0.5)  # Wait for 0.5 seconds before the next iteration

        print("\nAll iterations complete.")

    except RuntimeError as e:
        print(f"Runtime Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        print("\nClosing SDK...")
        xrt.close()
        print("SDK closed.")
        print("Test finished.")


if __name__ == "__main__":
    run_tests()
