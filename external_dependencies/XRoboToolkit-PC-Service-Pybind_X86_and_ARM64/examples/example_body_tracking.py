#!/usr/bin/env python3
"""
Example script demonstrating whole body motion tracking with XRoboToolkit SDK.

This script shows how to:
1. Check if body tracking data is available
2. Get body joint poses (position and rotation)
3. Get body joint velocities and accelerations
4. Get IMU timestamps for each joint
5. Get body data timestamp
6. Save data to structured format (pickle/json)

Body Joint Indices (24 joints total):
0: Pelvis, 1: Left Hip, 2: Right Hip, 3: Spine1, 4: Left Knee, 5: Right Knee,
6: Spine2, 7: Left Ankle, 8: Right Ankle, 9: Spine3, 10: Left Foot, 11: Right Foot,
12: Neck, 13: Left Collar, 14: Right Collar, 15: Head, 16: Left Shoulder, 17: Right Shoulder,
18: Left Elbow, 19: Right Elbow, 20: Left Wrist, 21: Right Wrist, 22: Left Hand, 23: Right Hand
"""

import xrobotoolkit_sdk as xrt
import time
import argparse
import csv
import os
import json
import pickle
from datetime import datetime


def main():

    xrt.init()
    
    while not xrt.is_body_data_available():
        time.sleep(0.01)
        
    if xrt.is_body_data_available():
        print("Body tracking data is available!")
        
        # Joint names for reference
        joint_names = [
            "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
            "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
            "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
            "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand"
        ]
        
        
        body_poses = xrt.get_body_joints_pose() # list of [x, y, z, qx, qy, qz, qw]
        body_velocities = xrt.get_body_joints_velocity() # vx, vy, vz, wx, wy, wz
        body_accelerations = xrt.get_body_joints_acceleration() # ax, ay, az, wax, way, waz
        imu_timestamps = xrt.get_body_joints_timestamp() # list of [timestamp]
        body_timestamp = xrt.get_body_timestamp_ns() # timestamp in ns
        
        saved_data = []
        length = 500
        step_idx = 0
        while len(saved_data) < length:
            
            # Sample data at specified rate
            if xrt.is_body_data_available():
                # Get all body tracking data
                body_poses = xrt.get_body_joints_pose()
                body_velocities = xrt.get_body_joints_velocity()
                body_accelerations = xrt.get_body_joints_acceleration()
                imu_timestamps = xrt.get_body_joints_timestamp()
                body_timestamp = xrt.get_body_timestamp_ns()


                body_pose_dict = {}
                for i, joint_name in enumerate(joint_names):
                    pos = [body_poses[i][0], body_poses[i][1], body_poses[i][2]]
                    rot = [body_poses[i][6], body_poses[i][3], body_poses[i][4], body_poses[i][5]] # scalar first
                    body_pose_dict[joint_name] = [pos, rot]

                saved_data.append(body_pose_dict)
                step_idx += 1
            time.sleep(1/50) 
            print(f"Step {step_idx} of {length}")
        
        with open('body_tracking_data.pkl', 'wb') as f:
            pickle.dump(saved_data, f)
        

if __name__ == "__main__":
    main() 