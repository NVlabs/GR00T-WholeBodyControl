#!/usr/bin/env python3
"""
Simple webcam recorder with optional preview window.
Usage:
    webcam_recorder.py [--output-dir DIR] [--no-preview]
    webcam_recorder.py --help
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class WebcamRecorder:
    def __init__(self, debug=False):
        self.process = None
        self.debug = debug

    def _debug_print(self, message):
        """Print debug message only if debug mode is enabled."""
        if self.debug:
            print(f"[webcam_recorder] {message}", file=sys.stderr)

    def _info_print(self, message):
        """Print info message always."""
        print(f"[webcam_recorder] {message}", file=sys.stderr)

    def find_webcam(self):
        """Find available webcam device."""
        # Check container status only in debug mode
        if self.debug and (os.path.exists("/.dockerenv") or os.environ.get("CONTAINER")):
            self._debug_print("Running in Docker container")
            # Check if we have video group permissions
            try:
                import grp

                groups = os.getgroups()
                video_gid = grp.getgrnam("video").gr_gid
                if video_gid in groups:
                    self._debug_print("Has video group permissions ✓")
                else:
                    self._debug_print("Warning: Not in video group")
            except Exception:
                pass

        # Check /dev/v4l/by-id for Logitech webcam first
        by_id_path = Path("/dev/v4l/by-id")
        if by_id_path.exists():
            self._debug_print("Checking /dev/v4l/by-id for devices...")
            for device in by_id_path.iterdir():
                if device.is_symlink():
                    device_name = device.name
                    if "Logitech" in device_name or "046d" in device_name:
                        resolved_device = str(device.resolve())
                        self._debug_print(
                            f"Found Logitech device: {device_name} -> {resolved_device}"
                        )
                        if self._is_video_capture_device(resolved_device):
                            return resolved_device

            # No Logitech found, try any webcam
            for device in by_id_path.iterdir():
                if device.is_symlink() and "metadata" not in device.name:
                    resolved_device = str(device.resolve())
                    self._debug_print(f"Checking device: {device.name} -> {resolved_device}")
                    if self._is_video_capture_device(resolved_device):
                        return resolved_device

        # Fallback to /dev/video* - prioritize external cameras
        self._debug_print("Scanning /dev/video* devices...")
        external_devices = []
        integrated_devices = []

        for i in range(20):
            device = f"/dev/video{i}"
            if os.path.exists(device):
                # Check if we can access the device
                if not os.access(device, os.R_OK):
                    self._debug_print(f"Found {device} but no read access")
                    continue

                # Check if it's a video capture device (not metadata)
                if self._is_video_capture_device(device):
                    # Check device name to prioritize external cameras
                    device_name = self._get_device_name(device)
                    if device_name:
                        if any(
                            keyword in device_name.lower()
                            for keyword in ["logitech", "brio", "c920", "c930", "c925"]
                        ):
                            self._info_print(f"Found external camera: {device} ({device_name})")
                            external_devices.append(device)
                        elif (
                            "integrated" in device_name.lower() or "internal" in device_name.lower()
                        ):
                            self._debug_print(f"Found integrated camera: {device} ({device_name})")
                            integrated_devices.append(device)
                        else:
                            self._debug_print(f"Found unknown camera: {device} ({device_name})")
                            external_devices.append(device)  # Assume external if unknown
                    else:
                        self._debug_print(f"Found camera: {device} (no name info)")
                        external_devices.append(device)

        # Return external camera first, then integrated as fallback
        if external_devices:
            self._info_print(f"Using external camera: {external_devices[0]}")
            return external_devices[0]
        elif integrated_devices:
            self._info_print(
                "No external camera found - integrated camera available but not preferred"
            )
            self._info_print(
                "Please connect an external camera (Logitech, Brio, etc.) for recording"
            )
            return None

        return None

    def _get_device_name(self, device):
        """Get the friendly name of a video device."""
        try:
            # Extract device number from /dev/videoX
            device_num = device.split("video")[-1]
            name_path = f"/sys/class/video4linux/video{device_num}/name"
            if os.path.exists(name_path):
                with open(name_path, "r") as f:
                    return f.read().strip()
        except Exception:
            pass
        return None

    def _is_video_capture_device(self, device):
        """Check if a device is a video capture device (not metadata)."""
        try:
            # Try v4l2-ctl first if available
            if subprocess.run(["which", "v4l2-ctl"], capture_output=True).returncode == 0:
                result = subprocess.run(
                    ["v4l2-ctl", "--device=" + device, "--info"], capture_output=True, timeout=2
                )
                if result.returncode == 0:
                    output = result.stdout.decode()
                    # Look for "Video Capture" capability in Device Caps, not just general Capabilities
                    lines = output.split("\n")
                    in_device_caps = False
                    for line in lines:
                        if "Device Caps" in line:
                            in_device_caps = True
                            continue
                        if in_device_caps:
                            if line.strip().startswith("Video Capture"):
                                # Double-check with ffmpeg
                                ffmpeg_result = subprocess.run(
                                    [
                                        "ffmpeg",
                                        "-f",
                                        "v4l2",
                                        "-i",
                                        device,
                                        "-frames:v",
                                        "1",
                                        "-f",
                                        "null",
                                        "-",
                                    ],
                                    capture_output=True,
                                    timeout=2,
                                )
                                return ffmpeg_result.returncode == 0
                            # Stop checking if we hit another section
                            elif line.strip() and not line.startswith("\t"):
                                break

            # Fallback: try ffmpeg directly if v4l2-ctl not available
            self._debug_print(f"Testing {device} with ffmpeg...")
            ffmpeg_result = subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "v4l2",
                    "-i",
                    device,
                    "-frames:v",
                    "1",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                timeout=5,
            )
            if ffmpeg_result.returncode == 0:
                self._debug_print(f"{device} is a working video capture device")
                return True
            else:
                stderr_output = ffmpeg_result.stderr.decode()
                # Check if it's just a metadata device
                if (
                    "metadata" in stderr_output.lower()
                    or "not a capture device" in stderr_output.lower()
                ):
                    self._debug_print(f"{device} is metadata device, skipping")
                    return False
                else:
                    self._debug_print(f"{device} failed ffmpeg test: {stderr_output[:100]}")
                    return False
        except Exception as e:
            self._debug_print(f"Error testing {device}: {e}")
            pass
        return False

    def test_camera_format(self, device):
        """Test camera and determine best format."""
        # Try default format
        result = subprocess.run(
            ["ffmpeg", "-f", "v4l2", "-i", device, "-frames:v", "1", "-f", "null", "-"],
            capture_output=True,
        )
        if result.returncode == 0:
            return []  # Default format works

        # Try MJPEG format
        result = subprocess.run(
            [
                "ffmpeg",
                "-f",
                "v4l2",
                "-input_format",
                "mjpeg",
                "-i",
                device,
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            return ["-input_format", "mjpeg"]

        raise RuntimeError(f"Cannot access camera at {device}")

    def record(self, output_dir="./logs_experiment", preview=True):
        """Start recording from webcam."""
        # Find webcam
        device = self.find_webcam()
        if not device:
            self._info_print("No webcam found")
            return False

        self._info_print(f"Recording from: {device}")

        # Test camera format
        try:
            format_args = self.test_camera_format(device)
        except RuntimeError as e:
            self._info_print(str(e))
            return False

        # Create output directory with date subfolder and filename
        now = datetime.now()
        date_folder = now.strftime("%Y_%m_%d")
        output_path = Path(output_dir) / date_folder
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        output_file = output_path / f"robot_video_{timestamp}.mp4"

        self._info_print(f"Saving to: {output_file}")
        if self.debug:
            self._debug_print(f"Absolute path: {output_file.absolute()}")

        # Check if preview is possible
        if preview and not os.environ.get("DISPLAY"):
            self._debug_print("No DISPLAY found, disabling preview")
            preview = False

        # Build ffmpeg command
        cmd = (
            ["ffmpeg", "-loglevel", "error", "-f", "v4l2"]
            + format_args
            + ["-i", device, "-c:v", "libx264", "-preset", "ultrafast"]
        )

        if preview:
            # Try with preview first
            preview_cmd = cmd + ["-f", "tee", "-map", "0:v", f"[f=mp4]{output_file}|[f=nut]pipe:"]

            try:
                ffmpeg_proc = subprocess.Popen(
                    preview_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                ffplay_proc = subprocess.Popen(
                    [
                        "ffplay",
                        "-loglevel",
                        "quiet",
                        "-f",
                        "nut",
                        "-i",
                        "pipe:",
                        "-window_title",
                        "Webcam Preview",
                    ],
                    stdin=ffmpeg_proc.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Wait a bit to see if it starts successfully
                time.sleep(0.5)
                if ffmpeg_proc.poll() is None and ffplay_proc.poll() is None:
                    self.process = ffplay_proc  # Kill ffplay to stop both
                    self._info_print("Recording with preview started")
                    return True
                else:
                    self._debug_print("Preview failed, falling back to no-preview mode")
                    # Clean up failed processes
                    try:
                        ffmpeg_proc.kill()
                        ffplay_proc.kill()
                    except Exception:
                        pass
            except Exception as e:
                self._debug_print(f"Preview error: {e}, using no-preview mode")

        # Record without preview
        cmd.append(str(output_file))
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        # Verify it's running
        time.sleep(0.5)
        if self.process.poll() is None:
            self._info_print("Recording started (no preview)")
            return True
        else:
            stderr = self.process.stderr.read().decode() if self.process.stderr else ""
            self._info_print(f"Failed to start recording: {stderr}")
            return False

    def stop(self):
        """Stop recording."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self._info_print("Recording stopped")
            self.process = None


def main():
    parser = argparse.ArgumentParser(description="Simple webcam recorder")
    parser.add_argument(
        "--output-dir", default="./logs_experiment", help="Output directory for recordings"
    )
    parser.add_argument("--no-preview", action="store_true", help="Disable preview window")
    parser.add_argument("--test", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    recorder = WebcamRecorder(debug=args.test)

    # Set up signal handlers for clean shutdown
    def signal_handler(signum, frame):
        print("\nStopping recording...", file=sys.stderr)
        recorder.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start recording
    if recorder.record(args.output_dir, preview=not args.no_preview):
        print("Press Ctrl+C to stop recording", file=sys.stderr)
        # Keep running until interrupted
        try:
            signal.pause()
        except KeyboardInterrupt:
            pass
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
