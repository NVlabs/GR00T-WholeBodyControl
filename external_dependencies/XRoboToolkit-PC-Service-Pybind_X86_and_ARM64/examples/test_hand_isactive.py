#!/usr/bin/env python3
"""
Test script for hand tracking isActive functionality.

This script demonstrates how to use the new isActive functions to check
hand tracking quality for both left and right hands.
"""

import xrobotoolkit_sdk as xrt
import time

def main():
    try:
        # Initialize the SDK
        print("Initializing XRoboToolkit SDK...")
        xrt.init()
        
        print("Testing hand tracking isActive functionality...")
        print("isActive values: 0 = low quality, 1 = high quality")
        print("Press Ctrl+C to stop\n")
        
        while True:
            # Get hand tracking states
            left_hand_state = xrt.get_left_hand_tracking_state()
            right_hand_state = xrt.get_right_hand_tracking_state()
            
            # Get hand tracking quality (isActive)
            left_hand_active = xrt.get_left_hand_is_active()
            right_hand_active = xrt.get_right_hand_is_active()
            
            
            print(f"Left Hand:  isActive={left_hand_active}")
            print(f"Right Hand: isActive={right_hand_active}")
            
            # Show hand tracking quality status
            left_quality = "HIGH" if left_hand_active == 1 else "LOW"
            right_quality = "HIGH" if right_hand_active == 1 else "LOW"
            
            print(f"Left Hand Quality: {left_quality}")
            print(f"Right Hand Quality: {right_quality}")
            
            # Example of first joint position for reference
            if len(left_hand_state) > 0:
                left_wrist_pos = left_hand_state[0][:3]  # x, y, z
                print(f"Left Wrist Position: ({left_wrist_pos[0]:.3f}, {left_wrist_pos[1]:.3f}, {left_wrist_pos[2]:.3f})")
            
            if len(right_hand_state) > 0:
                right_wrist_pos = right_hand_state[0][:3]  # x, y, z
                print(f"Right Wrist Position: ({right_wrist_pos[0]:.3f}, {right_wrist_pos[1]:.3f}, {right_wrist_pos[2]:.3f})")
            
            print("-" * 50)
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping hand tracking test...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing SDK...")
        xrt.close()

if __name__ == "__main__":
    main()