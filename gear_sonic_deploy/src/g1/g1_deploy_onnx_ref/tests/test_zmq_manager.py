#!/usr/bin/env python3
"""
Test script for ZMQManager

This script publishes test messages to the command, planner, and pose topics
to demonstrate how to use ZMQManager.

Usage:
    python3 test_zmq_manager.py [--port PORT] [--host HOST]
"""

import zmq
import numpy as np
import struct
import json
import time
import argparse

HEADER_SIZE = 1280  # Must match ZMQPackedMessageSubscriber::HEADER_SIZE

class ZMQPublisher:
    """Publisher for ZMQManager topics"""
    
    def __init__(self, host="*", port=5556, verbose=True):
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.endpoint = f"tcp://{host}:{port}"
        self.publisher.bind(self.endpoint)
        self.verbose = verbose
        
        # Give subscribers time to connect
        time.sleep(0.5)
        
        if self.verbose:
            print(f"[Publisher] Bound to {self.endpoint}")
            print(f"[Publisher] Publishing on topics: command, planner, pose")
    
    def send_command(self, start, stop, planner, delta_heading=None):
        """
        Send a command message
        start: bool (True=start control)
        stop: bool (True=stop control)
        planner: bool (True=planner mode, False=streamed motion mode)
        delta_heading: optional float (absolute delta heading in radians)
        """
        topic = b"command"
        
        # Create header
        fields = [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]}
        ]
        
        # Optional delta_heading field
        if delta_heading is not None:
            fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        
        header = {
            "v": 1,
            "endian": "le",
            "count": 1,
            "fields": fields
        }
        
        # Serialize header (pad to HEADER_SIZE bytes to match C++ ZMQPackedMessageSubscriber)
        header_json = json.dumps(header).encode('utf-8')
        header_bytes = header_json + b'\x00' * (HEADER_SIZE - len(header_json))
        
        # Serialize data
        data = b''
        data += struct.pack('B', 1 if start else 0)   # u8
        data += struct.pack('B', 1 if stop else 0)    # u8
        data += struct.pack('B', 1 if planner else 0) # u8
        if delta_heading is not None:
            data += struct.pack('<f', delta_heading)  # f32
        
        # Send packed message
        message = topic + header_bytes + data
        self.publisher.send(message)
        
        if self.verbose:
            if delta_heading is not None:
                print(f"[Command] Sent: start={start}, stop={stop}, planner={planner}, delta_heading={delta_heading:.3f} rad")
            else:
                print(f"[Command] Sent: start={start}, stop={stop}, planner={planner}")
    
    def send_planner(
        self,
        mode,
        movement,
        facing,
        speed=-1.0,
        height=-1.0,
        upper_body_position=None,
        upper_body_velocity=None,
        left_hand_joints=None,
        right_hand_joints=None,
    ):
        """
        Send a planner/movement command
        mode: int (LocomotionMode enum value)
        movement: [x, y, z] direction vector
        facing: [x, y, z] facing direction vector
        speed: float (-1 for default)
        height: float (-1 for default)
        upper_body_position: optional [17] upper body joint positions (f32)
        upper_body_velocity: optional [17] upper body joint velocities (f32)
        left_hand_joints: optional [7] left hand joint positions (f32)
        right_hand_joints: optional [7] right hand joint positions (f32)
        """
        topic = b"planner"
        
        # Create header
        fields = [
            {"name": "mode", "dtype": "i32", "shape": [1]},
            {"name": "movement", "dtype": "f32", "shape": [3]},
            {"name": "facing", "dtype": "f32", "shape": [3]},
            {"name": "speed", "dtype": "f32", "shape": [1]},
            {"name": "height", "dtype": "f32", "shape": [1]},
        ]

        # Optional upper body fields (17-DOF, f32)
        if upper_body_position is not None:
            fields.append(
                {"name": "upper_body_position", "dtype": "f32", "shape": [17]}
            )
        if upper_body_velocity is not None:
            fields.append(
                {"name": "upper_body_velocity", "dtype": "f32", "shape": [17]}
            )
        
        # Optional hand joint fields (7-DOF, f32)
        if left_hand_joints is not None:
            fields.append(
                {"name": "left_hand_joints", "dtype": "f32", "shape": [7]}
            )
        if right_hand_joints is not None:
            fields.append(
                {"name": "right_hand_joints", "dtype": "f32", "shape": [7]}
            )

        header = {
            "v": 1,
            "endian": "le",
            "count": 1,
            "fields": fields,
        }
        
        # Serialize header (pad to HEADER_SIZE bytes)
        header_json = json.dumps(header).encode('utf-8')
        header_bytes = header_json + b'\x00' * (HEADER_SIZE - len(header_json))
        
        # Serialize data (little-endian)
        data = b''
        data += struct.pack('<i', mode)  # i32
        data += struct.pack('<fff', *movement)  # f32[3]
        data += struct.pack('<fff', *facing)  # f32[3]
        data += struct.pack('<f', speed)  # f32
        data += struct.pack('<f', height)  # f32
        if upper_body_position is not None:
            ub_pos = np.asarray(upper_body_position, dtype=np.float32).reshape(17)
            data += ub_pos.tobytes()
        if upper_body_velocity is not None:
            ub_vel = np.asarray(upper_body_velocity, dtype=np.float32).reshape(17)
            data += ub_vel.tobytes()
        if left_hand_joints is not None:
            lh_joints = np.asarray(left_hand_joints, dtype=np.float32).reshape(7)
            data += lh_joints.tobytes()
        if right_hand_joints is not None:
            rh_joints = np.asarray(right_hand_joints, dtype=np.float32).reshape(7)
            data += rh_joints.tobytes()
        
        # Send packed message
        message = topic + header_bytes + data
        self.publisher.send(message)
        
        if self.verbose:
            print(f"[Planner] mode={mode}, movement={movement}, facing={facing}, speed={speed:.2f}")
    
    def send_pose(self, joint_pos, joint_vel, body_quat, frame_indices, catch_up=True):
        """
        Send pose/motion data (Protocol Version 1: joint-based)
        joint_pos: [N, num_joints] joint positions
        joint_vel: [N, num_joints] joint velocities
        body_quat: [N, 4] body quaternion (w,x,y,z)
        frame_indices: [N] frame indices
        catch_up: bool (True=allow catch up, False=real-time streaming)
        """
        topic = b"pose"
        
        N, num_joints = joint_pos.shape
        
        # Create header (Protocol Version 1)
        header = {
            "v": 1,
            "endian": "le",
            "count": N,
            "fields": [
                {"name": "joint_pos", "dtype": "f32", "shape": [N, num_joints]},
                {"name": "joint_vel", "dtype": "f32", "shape": [N, num_joints]},
                {"name": "body_quat_w", "dtype": "f32", "shape": [N, 4]},
                {"name": "frame_index", "dtype": "i64", "shape": [N]},
                {"name": "catch_up", "dtype": "u8", "shape": [1]}
            ]
        }
        
        # Serialize header (pad to HEADER_SIZE bytes)
        header_json = json.dumps(header).encode('utf-8')
        header_bytes = header_json + b'\x00' * (HEADER_SIZE - len(header_json))
        
        # Serialize data (little-endian, row-major)
        data = b''
        data += joint_pos.astype(np.float32).tobytes()
        data += joint_vel.astype(np.float32).tobytes()
        data += body_quat.astype(np.float32).tobytes()
        data += frame_indices.astype(np.int64).tobytes()
        data += struct.pack('B', 1 if catch_up else 0)  # catch_up flag
        
        # Send packed message
        message = topic + header_bytes + data
        self.publisher.send(message)
        
        if self.verbose:
            print(f"[Pose] {N} frames, {num_joints} joints, frame_index={frame_indices[0]}..{frame_indices[-1]}, catch_up={catch_up}")
    
    def close(self):
        self.publisher.close()
        self.context.term()


def generate_test_pose_data(num_frames=10, num_joints=29, start_frame=0):
    """Generate synthetic pose data for testing"""
    # Generate sinusoidal joint trajectories
    t = np.linspace(0, 2*np.pi, num_frames)
    joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
    joint_vel = np.zeros((num_frames, num_joints), dtype=np.float32)
    
    for j in range(num_joints):
        joint_pos[:, j] = 0.1 * np.sin(t + j * 0.1)
        joint_vel[:, j] = 0.1 * np.cos(t + j * 0.1)
    
    # Generate body quaternion (identity quaternion)
    body_quat = np.zeros((num_frames, 4), dtype=np.float32)
    body_quat[:, 0] = 1.0  # w component
    
    # Generate frame indices
    frame_indices = np.arange(start_frame, start_frame + num_frames, dtype=np.int64)
    
    return joint_pos, joint_vel, body_quat, frame_indices


def generate_shoulder_pitch_waving(num_frames=10, num_joints=29, start_frame=0, wave_amplitude=0.5, wave_frequency=1.0):
    """
    Generate simple shoulder pitch waving motion for testing streamed motion at 50 Hz
    
    Parameters:
        num_frames: Number of frames to generate
        num_joints: Number of joints (29 for G1)
        start_frame: Starting frame index
        wave_amplitude: Amplitude of shoulder wave (radians)
        wave_frequency: Frequency of wave (Hz)
    """
    # Joint indices for G1 robot (from policy_parameters.hpp)
    # Pose data is sent in MuJoCo order
    left_shoulder_pitch = 11   # left_shoulder_pitch in isaaclab order
    right_shoulder_pitch = 12  # right_shoulder_pitch in isaaclab order
    
    # Robot runs at 50 Hz
    dt = 0.02  # 50 Hz timestep
    
    # Time array for wave motion (at 50 Hz)
    # Use start_frame to ensure phase-continuous streaming across chunks.
    t = (start_frame + np.arange(num_frames)) * dt
    
    # Initialize joint positions and velocities
    joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
    joint_vel = np.zeros((num_frames, num_joints), dtype=np.float32)
    
    # Generate shoulder pitch waving motion with proper velocities
    omega = 2 * np.pi * wave_frequency  # Angular frequency
    
    for i in range(num_frames):
        time_val = t[i]
        
        # Left shoulder pitch wave (up and down)
        joint_pos[i, left_shoulder_pitch] = wave_amplitude * np.sin(omega * time_val)
        joint_vel[i, left_shoulder_pitch] = wave_amplitude * omega * np.cos(omega * time_val)
        
        # Right shoulder pitch wave (opposite phase)
        joint_pos[i, right_shoulder_pitch] = wave_amplitude * np.sin(omega * time_val + np.pi)
        joint_vel[i, right_shoulder_pitch] = wave_amplitude * omega * np.cos(omega * time_val + np.pi)
    
    # Other joints stay at zero (no motion)
    # Already initialized to zeros
    
    # Generate body quaternion (identity - no body motion)
    body_quat = np.zeros((num_frames, 4), dtype=np.float32)
    body_quat[:, 0] = 1.0  # w component
    
    # Generate frame indices
    frame_indices = np.arange(start_frame, start_frame + num_frames, dtype=np.int64)
    
    # Debug: Print motion range
    if num_frames > 0:
        left_min = np.min(joint_pos[:, left_shoulder_pitch])
        left_max = np.max(joint_pos[:, left_shoulder_pitch])
        right_min = np.min(joint_pos[:, right_shoulder_pitch])
        right_max = np.max(joint_pos[:, right_shoulder_pitch])
        print(f"    [Motion] Left shoulder pitch (idx {left_shoulder_pitch}): [{left_min:.3f}, {left_max:.3f}] rad ({np.degrees(left_min):.1f}°, {np.degrees(left_max):.1f}°)")
        print(f"    [Motion] Right shoulder pitch (idx {right_shoulder_pitch}): [{right_min:.3f}, {right_max:.3f}] rad ({np.degrees(right_min):.1f}°, {np.degrees(right_max):.1f}°)")
    
    return joint_pos, joint_vel, body_quat, frame_indices


def test_command_sequence(publisher):
    """Test command topic with a sequence of commands"""
    print("\n=== Testing Command Topic ===")
    
    # Test planner mode
    print("Starting with planner mode...")
    publisher.send_command(start=True, stop=False, planner=True)
    time.sleep(1.0)
    
    # Stop control
    print("Stopping control...")
    publisher.send_command(start=False, stop=True, planner=True)
    time.sleep(1.0)
    
    # Test streamed motion mode
    print("Starting with streamed motion mode...")
    publisher.send_command(start=True, stop=False, planner=False)
    time.sleep(1.0)
    
    # Stop control
    print("Stopping control...")
    publisher.send_command(start=False, stop=True, planner=False)
    time.sleep(1.0)


def test_planner_sequence(publisher):
    """Test planner topic with a sequence of movements"""
    print("\n=== Testing Planner Topic ===")
    
    # Walk forward
    print("Walking forward...")
    for i in range(10):
        publisher.send_planner(
            mode=2,  # WALK mode
            movement=[1.0, 0.0, 0.0],  # Forward
            facing=[1.0, 0.0, 0.0],
            speed=-1.0,
            height=-1.0
        )
        time.sleep(0.1)
    
    # Walk backward
    print("Walking backward...")
    for i in range(10):
        publisher.send_planner(
            mode=2,  # WALK mode
            movement=[-1.0, 0.0, 0.0],  # Backward
            facing=[1.0, 0.0, 0.0],
            speed=-1.0,
            height=-1.0
        )
        time.sleep(0.1)
    
    # Walk in circle
    print("Walking in circle...")
    for i in range(36):  # 3.6 seconds, 10 degrees per step
        angle = np.radians(i * 10)
        move_x = np.cos(angle)
        move_y = np.sin(angle)
        face_x = np.cos(angle)
        face_y = np.sin(angle)
        
        publisher.send_planner(
            mode=2,  # WALK mode
            movement=[move_x, move_y, 0.0],
            facing=[face_x, face_y, 0.0],
            speed=-1.0,
            height=-1.0
        )
        time.sleep(0.1)
    
    # Idle
    print("Idle...")
    publisher.send_planner(
        mode=0,  # IDLE mode
        movement=[0.0, 0.0, 0.0],
        facing=[1.0, 0.0, 0.0],
        speed=-1.0,
        height=-1.0
    )
    time.sleep(1.0)


def test_pose_sequence(publisher):
    """Test pose topic with shoulder pitch waving motion data"""
    print("\n=== Testing Pose Topic (Shoulder Waving) ===")
    
    # Send shoulder waving motion data with real-time streaming (catch_up=False)
    frame_idx = 0
    chunk_frames = 100  # 2 seconds at 50 Hz
    
    print("Sending 2s of motion...")
    joint_pos, joint_vel, body_quat, frame_indices = generate_shoulder_pitch_waving(
        num_frames=chunk_frames, num_joints=29, start_frame=frame_idx,
        wave_amplitude=1.5,  # Large motion (±1.5 rad = ±86°)
        wave_frequency=0.5
    )
    publisher.send_pose(joint_pos, joint_vel, body_quat, frame_indices, catch_up=False)
    frame_idx += chunk_frames
    
    print("Waiting 2s for playback...")
    time.sleep(2.0)
    
    print("Sending another 2s of motion (continuous)...")
    joint_pos, joint_vel, body_quat, frame_indices = generate_shoulder_pitch_waving(
        num_frames=chunk_frames, num_joints=29, start_frame=frame_idx,
        wave_amplitude=1.5,  # Large motion
        wave_frequency=0.5
    )
    publisher.send_pose(joint_pos, joint_vel, body_quat, frame_indices, catch_up=False)
    frame_idx += chunk_frames
    
    print("Waiting 2s for playback...")
    time.sleep(2.0)
    
    print("Shoulder waving complete!")


def _run_planner_and_streamed_steps(publisher):
    """Steps 2-13: planner walking, streamed motion, upper body control."""
    # 2. Send initial pose data
    input("\n[Press ENTER to begin streaming]")
    print("Step 2: Sending initial pose data...")
    frame_idx = 0
    for i in range(3):
        joint_pos, joint_vel, body_quat, frame_indices = generate_shoulder_pitch_waving(
            num_frames=10, num_joints=29, start_frame=frame_idx,
            wave_amplitude=0.5, wave_frequency=1.0
        )
        publisher.send_pose(joint_pos, joint_vel, body_quat, frame_indices)
        frame_idx += 10
        time.sleep(0.05)

    # 3. Walk forward for 3 seconds
    print("Step 3: Walking forward (3 seconds)...")
    for i in range(30):
        publisher.send_planner(
            mode=2, movement=[1.0, 0.0, 0.0], facing=[1.0, 0.0, 0.0],
            speed=-1.0, height=-1.0
        )
        joint_pos, joint_vel, body_quat, frame_indices = generate_shoulder_pitch_waving(
            num_frames=5, num_joints=29, start_frame=frame_idx,
            wave_amplitude=0.5, wave_frequency=1.0
        )
        publisher.send_pose(joint_pos, joint_vel, body_quat, frame_indices)
        frame_idx += 5
        time.sleep(0.1)

    # 4. Turn around (rotate 180 degrees)
    print("Step 4: Turning around...")
    for i in range(20):
        angle = np.pi * (i / 20.0)
        publisher.send_planner(
            mode=2, movement=[0.0, 0.0, 0.0],
            facing=[np.cos(angle), np.sin(angle), 0.0],
            speed=-1.0, height=-1.0
        )
        joint_pos, joint_vel, body_quat, frame_indices = generate_shoulder_pitch_waving(
            num_frames=5, num_joints=29, start_frame=frame_idx,
            wave_amplitude=0.5, wave_frequency=1.0
        )
        publisher.send_pose(joint_pos, joint_vel, body_quat, frame_indices)
        frame_idx += 5
        time.sleep(0.1)

    # 5. Walk forward again (now facing backward)
    print("Step 5: Walking forward (after turn around) for 3 seconds...")
    for i in range(30):
        publisher.send_planner(
            mode=2, movement=[-1.0, 0.0, 0.0], facing=[-1.0, 0.0, 0.0],
            speed=-1.0, height=-1.0
        )
        joint_pos, joint_vel, body_quat, frame_indices = generate_shoulder_pitch_waving(
            num_frames=5, num_joints=29, start_frame=frame_idx,
            wave_amplitude=0.5, wave_frequency=1.0
        )
        publisher.send_pose(joint_pos, joint_vel, body_quat, frame_indices)
        frame_idx += 5
        time.sleep(0.1)

    # 6. Mode switch test
    print("\nStep 6 - 1: Switching to streamed motion mode...")
    publisher.send_command(start=True, stop=False, planner=False)
    time.sleep(1.0)
    print("\nStep 6 - 2: Switching to planner motion mode...")
    publisher.send_command(start=True, stop=False, planner=True)
    time.sleep(1.0)
    print("\nStep 6 - 3: Switching to streamed motion mode...")
    publisher.send_command(start=True, stop=False, planner=False)
    time.sleep(1.0)

    # 7. Shoulder waving with rotation (streamed motion)
    print("Step 7: Waving shoulders with rotation (streamed motion)...")
    chunk_duration = 2.0
    frames_per_chunk = int(50 * chunk_duration)

    print(f"  Chunk 1: Sending {frames_per_chunk} frames (2 seconds)...")
    joint_pos_1, joint_vel_1, body_quat_1, frame_indices_1 = generate_shoulder_pitch_waving(
        num_frames=frames_per_chunk, num_joints=29, start_frame=frame_idx,
        wave_amplitude=1.5, wave_frequency=0.5
    )
    publisher.send_pose(joint_pos_1, joint_vel_1, body_quat_1, frame_indices_1, catch_up=False)
    frame_idx += frames_per_chunk

    print(f"  Waiting 2s for robot to play chunk 1 (with rotation)...")
    for i in range(20):
        delta_heading = (np.pi / 2.0) * (i / 20.0)
        publisher.send_command(start=False, stop=False, planner=False, delta_heading=delta_heading)
        time.sleep(0.1)

    print(f"  Chunk 2: Sending {frames_per_chunk} frames (2 seconds, continuous from frame {frame_idx})...")
    joint_pos_2, joint_vel_2, body_quat_2, frame_indices_2 = generate_shoulder_pitch_waving(
        num_frames=frames_per_chunk, num_joints=29, start_frame=frame_idx,
        wave_amplitude=1.5, wave_frequency=0.5
    )
    publisher.send_pose(joint_pos_2, joint_vel_2, body_quat_2, frame_indices_2, catch_up=False)
    frame_idx += frames_per_chunk

    print(f"  Waiting 2s for robot to play chunk 2 (continue rotation)...")
    for i in range(20):
        delta_heading = (np.pi / 2.0) + (np.pi / 2.0) * (i / 20.0)
        publisher.send_command(start=False, stop=False, planner=False, delta_heading=delta_heading)
        time.sleep(0.1)

    # 8. Wait
    print("Step 8: Additional wait (2 seconds)...")
    time.sleep(2.0)

    # 9. Switch back to planner mode
    print("\nStep 9: Switching back to planner mode...")
    publisher.send_command(start=True, stop=False, planner=True)
    time.sleep(1.0)

    # 10. Walk forward again (3 seconds)
    print("Step 10: Walking forward again (3 seconds)...")
    for i in range(30):
        publisher.send_planner(
            mode=2, movement=[1.0, 0.0, 0.0], facing=[1.0, 0.0, 0.0],
            speed=-1.0, height=-1.0
        )
        joint_pos, joint_vel, body_quat, frame_indices = generate_shoulder_pitch_waving(
            num_frames=5, num_joints=29, start_frame=frame_idx,
            wave_amplitude=0.5, wave_frequency=1.0
        )
        publisher.send_pose(joint_pos, joint_vel, body_quat, frame_indices)
        frame_idx += 5
        time.sleep(0.1)

    # 11. Turn around again (180 degrees)
    print("Step 11: Turning around again...")
    for i in range(20):
        angle = np.pi * (i / 20.0)
        publisher.send_planner(
            mode=2, movement=[0.0, 0.0, 0.0],
            facing=[np.cos(angle), np.sin(angle), 0.0],
            speed=-1.0, height=-1.0
        )
        joint_pos, joint_vel, body_quat, frame_indices = generate_shoulder_pitch_waving(
            num_frames=5, num_joints=29, start_frame=frame_idx,
            wave_amplitude=0.5, wave_frequency=1.0
        )
        publisher.send_pose(joint_pos, joint_vel, body_quat, frame_indices)
        frame_idx += 5
        time.sleep(0.1)

    # 12. Walk forward with upper body + hand control (10 seconds)
    print("Step 12: Walking forward again (after second turn, 10 seconds)...")
    print("         with upper body shoulder waving + hand open/close...")
    for i in range(100):
        t = i / 100.0
        pitch_amp = 0.5
        omega = 2 * np.pi
        pitch = pitch_amp * np.sin(omega * t)
        pitch_dot = pitch_amp * omega * np.cos(omega * t)

        upper_body_position = [0.0] * 17
        upper_body_velocity = [0.0] * 17
        upper_body_position[3] = pitch
        upper_body_position[4] = -pitch
        upper_body_velocity[3] = pitch_dot
        upper_body_velocity[4] = -pitch_dot

        time_seconds = i * 0.1
        hand_open_factor = 1.0 if (time_seconds % 2.0) >= 1.0 else 0.0
        left_hand_base = [0.0, 0.0, 1.75, -1.57, -1.75, -1.57, -1.75]
        left_hand_position = [b * (1.0 - hand_open_factor) for b in left_hand_base]
        right_hand_base = [0.0, 0.0, -1.75, 1.57, 1.75, 1.57, 1.75]
        right_hand_position = [b * (1.0 - hand_open_factor) for b in right_hand_base]

        publisher.send_planner(
            mode=2, movement=[-1.0, 0.0, 0.0], facing=[-1.0, 0.0, 0.0],
            speed=-1.0, height=-1.0,
            upper_body_position=upper_body_position,
            upper_body_velocity=upper_body_velocity,
            left_hand_joints=left_hand_position,
            right_hand_joints=right_hand_position,
        )
        joint_pos, joint_vel, body_quat, frame_indices = generate_shoulder_pitch_waving(
            num_frames=5, num_joints=29, start_frame=frame_idx,
            wave_amplitude=0.5, wave_frequency=1.0
        )
        publisher.send_pose(joint_pos, joint_vel, body_quat, frame_indices)
        frame_idx += 5
        time.sleep(0.1)

    # 13. Return to idle
    print("Step 13: Returning to idle...")
    for i in range(10):
        publisher.send_planner(
            mode=0, movement=[0.0, 0.0, 0.0], facing=[-1.0, 0.0, 0.0],
            speed=-1.0, height=-1.0
        )
        time.sleep(0.1)


def test_combined_sequence(publisher):
    """Test all topics together in a realistic scenario"""
    print("\n=== Testing Full Combined Sequence ===")
    
    # 1. Start control with planner mode
    input("\n[Press ENTER to send START command]")
    print("Step 1: Starting control with planner mode...")
    publisher.send_command(start=True, stop=False, planner=True)
    time.sleep(0.5)

    _run_planner_and_streamed_steps(publisher)

    # 14. Stop control
    print("\nStep 14: Stopping control...")
    publisher.send_command(start=False, stop=True, planner=False)
    time.sleep(1.0)

    print("\n=== Full sequence complete! ===")
    print("  Phase 1 - Planner mode (steps 1-5): walk, turn, walk")
    print("  Phase 2 - Streamed motion v1 (steps 6-8): shoulder waving + delta heading")
    print("  Phase 3 - Planner + upper body (steps 9-13): walk with hands")


def main():
    parser = argparse.ArgumentParser(description="Test ZMQManager")
    parser.add_argument("--host", type=str, default="*", help="Host to bind to (default: *)")
    parser.add_argument("--port", type=int, default=5556, help="Port to bind to (default: 5556)")
    parser.add_argument("--test", type=str, default="combined", 
                       choices=["command", "planner", "pose", "combined", "all"],
                       help="Test to run (default: combined)")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ZMQManager Test Publisher")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Test: {args.test}")
    print("=" * 60)
    
    # Create publisher
    publisher = ZMQPublisher(host=args.host, port=args.port, verbose=not args.quiet)
    
    try:
        if args.test == "command":
            test_command_sequence(publisher)
        elif args.test == "planner":
            test_planner_sequence(publisher)
        elif args.test == "pose":
            test_pose_sequence(publisher)
        elif args.test == "combined":
            test_combined_sequence(publisher)
        elif args.test == "all":
            test_command_sequence(publisher)
            test_planner_sequence(publisher)
            test_pose_sequence(publisher)
            test_combined_sequence(publisher)
        
        print("\n" + "=" * 60)
        print("Test complete!")
        print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        publisher.close()


if __name__ == "__main__":
    main()

