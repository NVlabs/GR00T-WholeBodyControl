import xrobotoolkit_sdk as xrt

xrt.init()
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
