"""
Camera viewer with manual recording support.

This script provides a camera viewer that can display multiple camera streams
and record them to video files with manual start/stop controls.

Features:
- Onscreen mode: Display camera feeds with optional recording
- Offscreen mode: No display, recording only when triggered
- Manual recording control with keyboard (R key to start/stop)

Usage Examples:

1. Basic onscreen viewing (with recording capability):
   python run_camera_viewer.py --camera-host localhost --camera-port 5555

2. Offscreen mode (no display, recording only):
   python run_camera_viewer.py --offscreen --camera-host localhost --camera-port 5555

3. Custom output directory:
   python run_camera_viewer.py --output-path ./my_recordings --camera-host localhost

Controls:
- R key: Start/Stop recording
- Q key: Quit application

Output Structure:
camera_output_20241211_143052/
├── rec_143205/
│   ├── ego_view_color_image.mp4
│   ├── head_left_color_image.mp4
│   └── head_right_color_image.mp4
└── rec_143410/
    ├── ego_view_color_image.mp4
    └── head_left_color_image.mp4
"""

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Optional

import cv2
import rclpy
from sshkeyboard import listen_keyboard, stop_listening
import tyro

from decoupled_wbc.control.main.teleop.configs.configs import ComposedCameraClientConfig
from decoupled_wbc.control.sensor.composed_camera import ComposedCameraClientSensor
from decoupled_wbc.control.utils.img_viewer import ImageViewer


@dataclass
class CameraViewerConfig(ComposedCameraClientConfig):
    """Config for running the camera viewer with recording support."""

    offscreen: bool = False
    """Run in offscreen mode (no display, manual recording with R key)."""

    output_path: Optional[str] = None
    """Output path for saving videos. If None, auto-generates path."""

    codec: str = "mp4v"
    """Video codec to use for saving (e.g., 'mp4v', 'XVID')."""


ArgsConfig = CameraViewerConfig


def _get_camera_titles(image_data: dict[str, Any]) -> list[str]:
    """
    Detect all the individual camera streams from the image data.

    schema format:
    {
        "timestamps": {"ego_view": 123.45, "ego_view_left_mono": 123.46},
        "images": {"ego_view": np.ndarray, "ego_view_left_mono": np.ndarray}
    }

    Returns list of camera keys (e.g., ["ego_view", "ego_view_left_mono", "ego_view_right_mono"])
    """
    # Extract all camera keys from the images dictionary
    camera_titles = list(image_data.get("images", {}).keys())
    return camera_titles


def main(config: ArgsConfig):
    """Main function to run the camera viewer."""
    # Initialize ROS
    rclpy.init(args=None)
    node = rclpy.create_node("camera_viewer")

    # Start ROS spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    image_sub = ComposedCameraClientSensor(server_ip=config.camera_host, port=config.camera_port)

    # pre-fetch a sample image to get the number of camera angles
    retry_count = 0
    while True:
        _sample_image = image_sub.read()
        if _sample_image:
            break
        retry_count += 1
        time.sleep(0.1)
        if retry_count > 10:
            raise Exception("Failed to get sample image")

    camera_titles = _get_camera_titles(_sample_image)

    # Setup output directory
    if config.output_path is None:
        output_dir = Path("camera_recordings")
    else:
        output_dir = Path(config.output_path)

    # Recording state
    is_recording = False
    video_writers = {}
    frame_count = 0
    recording_start_time = None
    should_quit = False

    def on_press(key):
        nonlocal is_recording, video_writers, frame_count, recording_start_time, should_quit

        if key == "r":
            if not is_recording:
                # Start recording
                recording_dir = output_dir / f"rec_{time.strftime('%Y%m%d_%H%M%S')}"
                recording_dir.mkdir(parents=True, exist_ok=True)

                # Create video writers
                fourcc = cv2.VideoWriter_fourcc(*config.codec)
                video_writers = {}

                for title in camera_titles:
                    img = _sample_image["images"].get(title)
                    if img is not None:
                        height, width = img.shape[:2]
                        video_path = recording_dir / f"{title}.mp4"
                        writer = cv2.VideoWriter(
                            str(video_path), fourcc, config.fps, (width, height)
                        )
                        video_writers[title] = writer

                is_recording = True
                recording_start_time = time.time()
                frame_count = 0
                print(f"🔴 Recording started: {recording_dir}")
            else:
                # Stop recording
                is_recording = False
                for title, writer in video_writers.items():
                    writer.release()
                video_writers = {}

                duration = time.time() - recording_start_time if recording_start_time else 0
                print(f"⏹️  Recording stopped - {duration:.1f}s, {frame_count} frames")
        elif key == "q":
            should_quit = True
            stop_listening()

    # Setup keyboard listener in a separate thread
    keyboard_thread = threading.Thread(
        target=lambda: listen_keyboard(on_press=on_press), daemon=True
    )
    keyboard_thread.start()

    # Setup viewer for onscreen mode
    viewer = None
    if not config.offscreen:
        viewer = ImageViewer(
            title="Camera Viewer",
            figsize=(10, 8),
            num_images=len(camera_titles),
            image_titles=camera_titles,
        )

    # Print instructions
    mode = "Offscreen" if config.offscreen else "Onscreen"
    print(f"{mode} mode - Target FPS: {config.fps}")
    print(f"Videos will be saved to: {output_dir}")
    print("Controls: R key to start/stop recording, Q key to quit, Ctrl+C to exit")

    # Create ROS rate controller
    rate = node.create_rate(config.fps)

    try:
        while rclpy.ok() and not should_quit:
            # Get images from all subscribers
            images = []
            image_data = image_sub.read()
            if image_data:
                for title in camera_titles:
                    img = image_data["images"].get(title)
                    images.append(img)

                    # Save frame if recording
                    if is_recording and img is not None and title in video_writers:
                        # Convert from RGB to BGR for OpenCV
                        if len(img.shape) == 3 and img.shape[2] == 3:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        else:
                            img_bgr = img
                        video_writers[title].write(img_bgr)

            # Display images if not offscreen
            if not config.offscreen and viewer and any(img is not None for img in images):
                status = "🔴 REC" if is_recording else "⏸️ Ready"
                viewer._fig.suptitle(f"Camera Viewer - {status}")
                viewer.show_multiple(images)

            # Progress feedback
            if is_recording:
                frame_count += 1
                if frame_count % 100 == 0:
                    duration = time.time() - recording_start_time
                    print(f"Recording: {frame_count} frames ({duration:.1f}s)")

            rate.sleep()

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Cleanup
        try:
            stop_listening()
        except Exception:
            pass

        if video_writers:
            for title, writer in video_writers.items():
                writer.release()
            if is_recording:
                duration = time.time() - recording_start_time
                print(f"Final: {duration:.1f}s, {frame_count} frames")

        if viewer:
            viewer.close()

        rclpy.shutdown()


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
