#!/usr/bin/env python3
"""
Standalone Pose Estimation Server

This server runs the WebcamPoseEstimator on a separate computer and streams
pose estimation results over the network to robot control systems.

Usage:
    python pose_estimation_server.py --config server_config.yaml
"""

import argparse
import json
import os
import struct
import time
import numpy as np
from typing import Dict, Optional

import sys

import logging
Log = logging.getLogger()
import zmq

sys.path.append("./")



class PoseEstimationServer:
    """
    Standalone server for pose estimation that can run on a separate computer.

    Features:
    - Runs WebcamPoseEstimator in a dedicated process
    - Streams pose data over ZMQ network protocol
    - Supports multiple concurrent clients
    - Handles camera input and video files
    - Provides real-time pose estimation with low latency
    """

    def __init__(
        self,
        port: int = 5558,
        webcam_config: Optional[Dict] = None,
        max_clients: int = 5,
        fps: float = 30,
        enable_display: bool = True,
        verbose: bool = False,
        fixed_fps: bool = False,
        optimize_onnx_graph: bool = False,
    ):
        """
        Initialize the pose estimation server.

        Args:
            port: ZMQ server port
            webcam_config: Configuration for WebcamPoseEstimator
            max_clients: Maximum number of concurrent clients
            fps: target FPS for pose estimation
            enable_display: Whether to show local display window
            verbose: Enable verbose logging
            fixed_fps: Use fixed FPS timing instead of adaptive timing
        """
        self.port = port
        self.max_clients = max_clients
        self.fps = fps
        self.enable_display = enable_display
        self.verbose = verbose
        self.fixed_fps = fixed_fps

        # Default webcam configuration
        self.webcam_config = {
            "camera_id": 0,
            "video_path": None,
            "target_fps": fps,
            "context_frames": 121,
            "yolo_period": 1,
            "yolo_imgsz": 480,
            "yolo_conf": 0.4,
            "yolo_fp16": True,
            "use_torch": False,
            "output_root": "outputs/pose_server",
            "verbose": verbose,
            "display": enable_display,
            "save": False,
            "disable_cv2_window": not enable_display,
            "pose_est_freq": 1,
            "fixed_fps": fixed_fps,
            "optimize_onnx_graph": optimize_onnx_graph,
        }

        # Update with provided config
        if webcam_config:
            self.webcam_config.update(webcam_config)

        # Initialize ZMQ for publishing
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        self.socket.setsockopt(
            zmq.SNDHWM, 3
        )  # Send high water mark - only keep 3 messages in queue
        self.socket.setsockopt(zmq.CONFLATE, 1)  # Receive high water mark - small queue
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait for unsent messages on close
        self.socket.setsockopt(zmq.IMMEDIATE, 1)  # Don't queue messages if no subscribers

        # Give socket time to establish
        time.sleep(0.1)

        # Initialize pose estimator
        self.pose_estimator = None
        self.is_running = False
        self.client_count = 0

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_report = time.time()

        # Remove client tracking since we're broadcasting
        self.client_count = 0  # Keep for compatibility but not used

        Log.info(f"[PoseEstimationServer] Initialized on port {self.port}")

    def initialize_pose_estimator(self):
        """Initialize the WebcamPoseEstimator."""
        # self.pose_estimator = WebcamPoseEstimator(**self.webcam_config)
        Log.info("[PoseEstimationServer] WebcamPoseEstimator initialized successfully")
        return True

    def process_pose_estimation(self) -> Optional[Dict]:
        """
        Process a single frame and return pose estimation results.

        Returns:
            Dictionary containing pose estimation results or None if not ready
        """

        try:
            
            pose_data = {
                "joint_pos": np.ones((201, 29), dtype=np.float32),
                "joint_vel": np.ones((201, 29), dtype=np.float32) * 2,
                "body_quat_w": np.ones((201, 4), dtype=np.float32) * 3,
                "frame_index": np.arange(201, dtype=np.int32),
            }

            return pose_data

        except Exception as e:
            Log.error(f"[PoseEstimationServer] Error processing frame: {e}")
            return {"status": "error", "ready": False, "error": str(e)}

    def _pack_pose_message(self, pose_data: Dict, topic: str = "pose") -> bytes:
        """
        Pack pose data into single-frame format:
        [topic_prefix][1024-byte JSON header][concatenated binary fields]
        
        Args:
            pose_data: Dictionary containing numpy arrays to send
            topic: Topic prefix string
            
        Returns:
            Packed message as bytes
        """
        HEADER_SIZE = 1024
        
        # Build fields list from pose_data
        fields = []
        binary_data = []
        
        for key, value in pose_data.items():
            if isinstance(value, np.ndarray):
                # Determine dtype string
                if value.dtype == np.float32:
                    dtype_str = "f32"
                elif value.dtype == np.float64:
                    dtype_str = "f64"
                elif value.dtype == np.int32:
                    dtype_str = "i32"
                elif value.dtype == np.int64:
                    dtype_str = "i64"
                else:
                    # Default to f32, cast if needed
                    dtype_str = "f32"
                    value = value.astype(np.float32)
                
                fields.append({
                    "name": key,
                    "dtype": dtype_str,
                    "shape": list(value.shape)
                })
                
                # Ensure contiguous and little-endian
                if not value.flags['C_CONTIGUOUS']:
                    value = np.ascontiguousarray(value)
                if value.dtype.byteorder == '>':
                    value = value.astype(value.dtype.newbyteorder('<'))
                    
                binary_data.append(value.tobytes())
        
        # Build JSON header
        header_dict = {
            "v": 1,
            "endian": "le",
            "fields": fields
        }
        header_json = json.dumps(header_dict, separators=(',', ':'))
        
        # Ensure header fits in HEADER_SIZE
        if len(header_json) >= HEADER_SIZE:
            raise ValueError(f"JSON header too large: {len(header_json)} >= {HEADER_SIZE}")
        
        # Pack message: [topic][1024-byte header][binary data]
        topic_bytes = topic.encode('utf-8')
        header_bytes = header_json.encode('utf-8').ljust(HEADER_SIZE, b'\x00')
        data_bytes = b''.join(binary_data)
        
        packed_message = topic_bytes + header_bytes + data_bytes
        return packed_message

    def broadcast_pose_data(self):
        """
        Broadcast pose estimation data to all subscribers.
        """
        pose_data = self.process_pose_estimation()
        if pose_data is not None:
            try:
                # Broadcast pose data with topic "pose"
                # print(f"\nsend frame_index:", pose_data["frame_index"])
                
                # Pack and send as single frame
                packed_message = self._pack_pose_message(pose_data, topic="pose")
                self.socket.send(packed_message)
                
                self.frame_count += 1

                if self.verbose and self.frame_count % 100 == 0:
                    Log.info(f"[PoseEstimationServer] Broadcasted frame {self.frame_count}")

            except Exception as e:
                Log.error(f"[PoseEstimationServer] Error broadcasting pose data: {e}")

    def report_fps(self):
        """Report FPS periodically."""
        current_time = time.time()
        if current_time - self.last_fps_report >= 5.0:  # Report every 5 seconds
            last_frame_count = (
                self.frame_count if not hasattr(self, "last_frame_count") else self.last_frame_count
            )
            elapsed = current_time - self.last_fps_report
            avg_fps = (self.frame_count - last_frame_count) / elapsed if elapsed > 0 else 0
            Log.info(
                f"frame_count: {self.frame_count}, last_frame_count: {last_frame_count}, "
                f"[PoseEstimationServer] Broadcasted {self.frame_count - last_frame_count} frames, "
                f"Average FPS: {avg_fps:.2f}"
            )
            self.last_fps_report = current_time
            self.last_frame_count = self.frame_count

    def run(self):
        """Run the pose estimation server."""
        Log.info(f"[PoseEstimationServer] Starting broadcast server on port {self.port}")

        # Initialize pose estimator
        if not self.initialize_pose_estimator():
            Log.error("[PoseEstimationServer] Failed to initialize pose estimator. Exiting.")
            return

        self.is_running = True
        self.start_time = time.time()

        Log.info("[PoseEstimationServer] Server ready and broadcasting pose data...")

        try:

            while self.is_running:
                try:
                    # Broadcast pose data continuously
                    self.broadcast_pose_data()

                    # Report FPS periodically
                    self.report_fps()

                except zmq.ZMQError as e:
                    if e.errno == zmq.ETERM:
                        break
                    Log.error(f"[PoseEstimationServer] ZMQ Error: {e}")

                except Exception as e:
                    Log.error(f"[PoseEstimationServer] Unexpected error: {e}")

        except KeyboardInterrupt:
            Log.info("[PoseEstimationServer] Received interrupt signal")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        Log.info("[PoseEstimationServer] Shutting down...")

        self.is_running = False

        if self.pose_estimator:
            try:
                self.pose_estimator.release()
            except Exception:
                pass

        self.socket.close()
        self.context.term()

        Log.info("[PoseEstimationServer] Shutdown complete")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    import yaml

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        Log.error(f"Failed to load config from {config_path}: {e}")
        return {}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pose Estimation Server")

    parser.add_argument("--port", type=int, default=5558, help="Server port (default: 5558)")
    parser.add_argument("--config", type=str, default=None, help="Configuration file path")
    parser.add_argument("--camera", type=int, default=-1, help="Camera device ID (default: 0)")
    parser.add_argument(
        "--video",
        type=str,
        default="external_dependencies/genmo/inputs/videos/zen2_540p.mp4",
        help="Video file path (default: None for webcam)",
    )
    parser.add_argument("--fps", type=float, default=30, help="Target FPS (default: None)")
    parser.add_argument(
        "--pose_est_freq",
        type=int,
        default=1,
        help="Run pose estimation every N frames (default: 1)",
    )
    parser.add_argument("--no_display", action="store_true", help="Disable display window")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--fixed_fps", action="store_true", help="Use fixed FPS timing instead of adaptive timing"
    )
    parser.add_argument("-opt", "--optimize_onnx_graph", action="store_true", help="Optimize ONNX graph for inference")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)

    # Override with command line arguments
    server_config = config.get("server", {})
    webcam_config = config.get("webcam", {})

    # Command line overrides
    if args.camera is not None:
        webcam_config["camera_id"] = args.camera
    if args.video is not None:
        webcam_config["video_path"] = args.video
    if hasattr(args, "pose_est_freq") and args.pose_est_freq is not None:
        webcam_config["pose_est_freq"] = args.pose_est_freq
    # Create and run server
    server = PoseEstimationServer(
        port=server_config.get("port", args.port),
        webcam_config=webcam_config,
        max_clients=server_config.get("max_clients", 5),
        fps=args.fps,
        enable_display=not args.no_display,
        verbose=args.verbose,
        fixed_fps=args.fixed_fps,
        optimize_onnx_graph=args.optimize_onnx_graph,
    )

    server.run()


if __name__ == "__main__":
    main()
