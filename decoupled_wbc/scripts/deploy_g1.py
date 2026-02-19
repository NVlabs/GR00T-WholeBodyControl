from pathlib import Path
import signal
import subprocess
import sys
import time

import tyro

from decoupled_wbc.control.main.teleop.configs.configs import DeploymentConfig
from decoupled_wbc.control.utils.run_real_checklist import show_deployment_checklist


class G1Deployment:
    """
    Unified deployment manager for G1 robot with one-click operation.
    Handles camera setup, control loop, teleoperation, and data collection.
    Uses tmux for process management and I/O handling.
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config

        # Process directories
        self.project_root = Path(__file__).resolve().parent.parent

        # Tmux session name
        self.session_name = "g1_deployment"

        # Create tmux session if it doesn't exist
        self._create_tmux_session()

    def _create_tmux_session(self):
        """Create a new tmux session if it doesn't exist"""
        # Check if session exists
        result = subprocess.run(
            ["tmux", "has-session", "-t", self.session_name], capture_output=True, text=True
        )

        if result.returncode != 0:
            # Create new session
            subprocess.run(["tmux", "new-session", "-d", "-s", self.session_name])
            print(f"Created new tmux session: {self.session_name}")

            # Set up the default window for control, data collection, and teleop
            # First rename the default window (which is 0) to our desired name
            subprocess.run(
                ["tmux", "rename-window", "-t", f"{self.session_name}:0", "control_data_teleop"]
            )
            # Split the window horizontally (left and right)
            subprocess.run(["tmux", "split-window", "-t", f"{self.session_name}:0", "-h"])
            # Split the right pane vertically (top and bottom)
            subprocess.run(["tmux", "split-window", "-t", f"{self.session_name}:0.1", "-v"])
            # Select the left pane (control)
            subprocess.run(["tmux", "select-pane", "-t", f"{self.session_name}:0.0"])

    def _run_in_tmux(self, name, cmd, wait_time=2, pane_index=None):
        """Run a command in a new tmux window or pane"""
        if pane_index is not None:
            # Run in existing window's pane
            target = f"{self.session_name}:0.{pane_index}"
        else:
            # Create new window
            subprocess.run(["tmux", "new-window", "-t", self.session_name, "-n", name])
            target = f"{self.session_name}:{name}"

        # Set up trap for Ctrl+\ in the window
        trap_cmd = f"trap 'tmux kill-session -t {self.session_name}' QUIT"

        # Set environment variable for the tmux session name
        env_cmd = f"export DECOUPLED_WBC_TMUX_SESSION={self.session_name}"

        # Construct the command with proper escaping and trap
        cmd_str = " ".join(str(x) for x in cmd)
        full_cmd = f"{trap_cmd}; {env_cmd}; {cmd_str}"

        # Send command to tmux window/pane
        subprocess.run(["tmux", "send-keys", "-t", target, full_cmd, "C-m"])

        # Wait for process to start
        time.sleep(wait_time)

        # Check if process is still running
        result = subprocess.run(
            ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
            capture_output=True,
            text=True,
        )

        if result.stdout.strip() == "1":
            print(f"ERROR: {name} failed to start!")
            return False

        return True

    def start_camera_sensor(self):
        """Start the camera sensor in local mode if we are using replay video"""
        if self.config.egoview_replay_dummy is None and self.config.head_replay_dummy is None:
            return

        print("Starting camera sensor in local mode...")
        cmd = [
            sys.executable,
            str(self.project_root / "control/sensor/composed_camera.py"),
            "--egoview_camera",
            self.config.egoview_replay_dummy,
            "--head_camera",
            self.config.head_replay_dummy,
            "--port",
            str(self.config.camera_port),
            "--host",
            "localhost",
        ]

        if not self._run_in_tmux("camera_sensor", cmd):
            print("ERROR: Camera sensor failed to start!")
            print("Continuing without camera sensor...")
        else:
            print("Camera sensor started successfully.")

    def start_camera_viewer(self):
        """Start the ROS rqt camera viewer"""
        if not self.config.view_camera:
            return

        print("Starting camera viewer...")
        # Use rqt directly instead of ros2 run
        cmd = [
            sys.executable,
            str(self.project_root / "control/main/teleop/run_camera_viewer.py"),
            "--camera_host",
            self.config.camera_host,
            "--camera_port",
            str(self.config.camera_port),
            "--fps",
            str(self.config.fps),
        ]

        if not self._run_in_tmux("camera_viewer", cmd):
            print("ERROR: Camera viewer failed to start!")
            print("Continuing without camera viewer...")
        else:
            print("Camera viewer started successfully.")

    def start_sim_loop(self):
        """Start the simulation loop in a separate process"""
        print("Starting simulation loop...")
        cmd = [
            sys.executable,
            str(self.project_root / "control/main/teleop/run_sim_loop.py"),
            "--wbc_version",
            self.config.wbc_version,
            "--interface",
            self.config.interface,
            "--simulator",
            self.config.simulator,
            "--sim_frequency",
            str(self.config.sim_frequency),
            "--env_name",
            self.config.env_name,
            "--camera_port",
            str(self.config.camera_port),
        ]

        # Handle boolean flags
        if self.config.enable_waist:
            cmd.append("--enable_waist")
        else:
            cmd.append("--no-enable_waist")

        if self.config.with_hands:
            cmd.append("--with_hands")
        else:
            cmd.append("--no-with_hands")

        if self.config.image_publish:
            cmd.append("--enable_image_publish")
            cmd.append("--enable_offscreen")
        else:
            cmd.append("--no-enable_image_publish")

        if self.config.enable_onscreen:
            cmd.append("--enable_onscreen")
        else:
            cmd.append("--no-enable_onscreen")

        if not self._run_in_tmux("sim_loop", cmd, wait_time=5):
            print("ERROR: Simulation loop failed to start!")
            self.cleanup()
            sys.exit(1)

        print("Simulation loop started successfully. Waiting for warmup for 10 seconds...")
        time.sleep(10)  # Wait for sim loop to warm up

    def start_control_loop(self):
        """Start the G1 control loop"""
        print("Starting G1 control loop...")
        cmd = [
            sys.executable,
            str(self.project_root / "control/main/teleop/run_g1_control_loop.py"),
            "--wbc_version",
            self.config.wbc_version,
            "--wbc_model_path",
            self.config.wbc_model_path,
            "--wbc_policy_class",
            self.config.wbc_policy_class,
            "--interface",
            self.config.interface,
            "--simulator",
            "None" if self.config.sim_in_single_process else self.config.simulator,
            "--control_frequency",
            str(self.config.control_frequency),
        ]

        # Handle boolean flag using presence/absence pattern
        if self.config.enable_waist:
            cmd.append("--enable_waist")
        else:
            cmd.append("--no-enable_waist")

        if self.config.with_hands:
            cmd.append("--with_hands")
        else:
            cmd.append("--no-with_hands")

        if self.config.high_elbow_pose:
            cmd.append("--high_elbow_pose")
        else:
            cmd.append("--no-high_elbow_pose")

        # Gravity compensation configuration
        # Note: This is where gravity compensation is actually applied since the control loop
        # contains the G1Body that interfaces directly with the robot motors
        if self.config.enable_gravity_compensation:
            cmd.append("--enable_gravity_compensation")
            # Add joint groups if specified
            if self.config.gravity_compensation_joints:
                cmd.extend(
                    ["--gravity_compensation_joints"] + self.config.gravity_compensation_joints
                )
        else:
            cmd.append("--no-enable_gravity_compensation")

        if not self._run_in_tmux("control", cmd, wait_time=3, pane_index=0):
            print("ERROR: Control loop failed to start!")
            self.cleanup()
            sys.exit(1)

        print("Control loop started successfully.")
        print("Controls: 'i' for initial pose, ']' to activate locomotion")

    def start_policy(self):
        """Start either teleop or inference policy based on configuration"""
        if not self.config.enable_upper_body_operation:
            print("Upper body operation disabled in config.")
            return

        self.start_teleop()

    def start_teleop(self):
        """Start the teleoperation policy"""
        print("Starting teleoperation policy...")
        cmd = [
            sys.executable,
            str(self.project_root / "control/main/teleop/run_teleop_policy_loop.py"),
            "--body_control_device",
            self.config.body_control_device,
            "--hand_control_device",
            self.config.hand_control_device,
            "--body_streamer_ip",
            self.config.body_streamer_ip,
            "--body_streamer_keyword",
            self.config.body_streamer_keyword,
        ]

        # Handle boolean flags using tyro syntax
        if self.config.enable_waist:
            cmd.append("--enable_waist")
        else:
            cmd.append("--no-enable_waist")

        if self.config.high_elbow_pose:
            cmd.append("--high_elbow_pose")
        else:
            cmd.append("--no-high_elbow_pose")

        if self.config.enable_visualization:
            cmd.append("--enable_visualization")
        else:
            cmd.append("--no-enable_visualization")

        if self.config.enable_real_device:
            cmd.append("--enable_real_device")
        else:
            cmd.append("--no-enable_real_device")

        if not self._run_in_tmux("teleop", cmd, pane_index=2):
            print("ERROR: Teleoperation policy failed to start!")
            print("Continuing without teleoperation...")
        else:
            print("Teleoperation policy started successfully.")
            print("Press 'l' in the control loop terminal to start teleoperation.")

    def start_data_collection(self):
        """Start the data collection process"""
        if not self.config.data_collection:
            print("Data collection disabled in config.")
            return

        print("Starting data collection...")
        cmd = [
            sys.executable,
            str(self.project_root / "control/main/teleop/run_g1_data_exporter.py"),
            "--data_collection_frequency",
            str(self.config.data_collection_frequency),
            "--root_output_dir",
            self.config.root_output_dir,
            "--lower_body_policy",
            self.config.wbc_version,
            "--wbc_model_path",
            self.config.wbc_model_path,
            "--camera_host",
            self.config.camera_host,
            "--camera_port",
            str(self.config.camera_port),
        ]

        if not self._run_in_tmux("data", cmd, pane_index=1):
            print("ERROR: Data collection failed to start!")
            print("Continuing without data collection...")
        else:
            print("Data collection started successfully.")
            print("Press 'c' in the control loop terminal to start/stop recording data.")

    def start_webcam_recording(self):
        """Start webcam recording for real robot deployment monitoring"""
        if not self.config.enable_webcam_recording or self.config.env_type != "real":
            return

        print("Starting webcam recording for deployment monitoring...")
        cmd = [
            sys.executable,
            str(self.project_root / "scripts/run_webcam_recorder.py"),
            "--output_dir",
            self.config.webcam_output_dir,
        ]

        if not self._run_in_tmux("webcam", cmd):
            print("ERROR: Webcam recording failed to start!")
            print("Continuing without webcam recording...")
        else:
            print("Webcam recording started successfully.")
            print("External camera recording deployment activities to logs_experiment/")

    def deploy(self):
        """
        Run the complete deployment process
        """
        print("Starting G1 deployment with config:")
        print(f"  Robot IP: {self.config.robot_ip}")
        print(f"  WBC Version: {self.config.wbc_version}")
        print(f"  Interface: {self.config.interface}")
        print(f"  Policy Mode: {self.config.upper_body_operation_mode}")
        print(f"  With Hands: {self.config.with_hands}")
        print(f"  View Camera: {self.config.view_camera}")
        print(f"  Enable Waist: {self.config.enable_waist}")
        print(f"  High Elbow Pose: {self.config.high_elbow_pose}")
        print(f"  Gravity Compensation: {self.config.enable_gravity_compensation}")
        if self.config.enable_gravity_compensation:
            print(f"  Gravity Comp Joints: {self.config.gravity_compensation_joints}")
        print(
            f"  Webcam Recording: {self.config.enable_webcam_recording and self.config.env_type == 'real'}"
        )
        print(f"  Sim in Single Process: {self.config.sim_in_single_process}")
        if self.config.sim_in_single_process:
            print(f"  Image Publish: {self.config.image_publish}")

        # Check if this is a real robot deployment and run safety checklist
        if self.config.env_type == "real":
            if not show_deployment_checklist():
                sys.exit(1)

        # Register signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

        # Start components in sequence
        # Start sim loop first if sim_in_single_process is enabled
        if self.config.sim_in_single_process:
            self.start_sim_loop()

        self.start_control_loop()
        self.start_camera_viewer()
        self.start_policy()  # This will start either teleop or inference policy
        self.start_data_collection()
        self.start_webcam_recording()  # Only runs for real robot deployment

        print("\n--- G1 DEPLOYMENT COMPLETE ---")
        print("All systems running in tmux session:", self.session_name)
        print("Press Ctrl+b then d to detach from the session")
        print("Press Ctrl+\\ in any window to shutdown all components.")

        try:
            # Automatically attach to the tmux session and switch to control window
            subprocess.run(
                [
                    "tmux",
                    "attach",
                    "-t",
                    self.session_name,
                    ";",
                    "select-window",
                    "-t",
                    "control_data_teleop",
                ]
            )
        except KeyboardInterrupt:
            print("\nShutdown requested...")
            self.cleanup()
            sys.exit(0)

        # Keep main thread alive to handle signals
        try:
            while True:
                # Check if tmux session still exists
                result = subprocess.run(
                    ["tmux", "has-session", "-t", self.session_name], capture_output=True, text=True
                )

                if result.returncode != 0:
                    print("Tmux session terminated. Exiting.")
                    break

                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up tmux session"""
        print("Cleaning up tmux session...")
        try:
            # Kill the tmux session
            subprocess.run(["tmux", "kill-session", "-t", self.session_name], timeout=5)
            print("Tmux session terminated successfully.")
        except subprocess.TimeoutExpired:
            print("Warning: Tmux session termination timed out, forcing kill...")
            subprocess.run(["tmux", "kill-session", "-t", self.session_name, "-9"])
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
        print("Cleanup complete.")

    def signal_handler(self, sig, frame):
        """Handle SIGINT (Ctrl+C) gracefully"""
        print("\nShutdown signal received...")
        self.cleanup()
        sys.exit(0)


def main():
    """Main entry point with automatic CLI generation from G1Config dataclass"""
    # This single line automatically generates a complete CLI from the dataclass!
    config = tyro.cli(DeploymentConfig)

    # Run deployment with the configured settings
    deployment = G1Deployment(config)
    deployment.deploy()


if __name__ == "__main__":
    # Edited outside docker

    main()
