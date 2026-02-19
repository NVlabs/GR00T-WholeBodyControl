"""
MuJoCo Visualizer Class

A standalone visualizer for MuJoCo simulations that supports both interactive viewing
and offline rendering for video recording. Extracted and refactored from the
MetricNeuralRetarget callback.

Features:
- Interactive viewer with keyboard controls
- Offline rendering for video recording
- SMPL joints visualization as 3D spheres
- Side-by-side comparison of ground truth and predicted poses
- Headless rendering support (EGL/OSMesa)
- Configurable camera settings and rendering parameters
"""

import logging
import os
import tempfile
import threading
import time
from typing import Dict, List, Optional, Union
import xml.etree.ElementTree as ET

import numpy as np
import torch


# Configure Mesa for headless rendering before any MuJoCo imports
def _configure_headless_rendering():
    """Configure environment for headless MuJoCo rendering"""
    # Set MuJoCo to use EGL for hardware-accelerated offscreen rendering
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    # Set PyOpenGL platform for EGL
    if "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    # Fallback to OSMesa if EGL is not available
    if os.environ.get("MUJOCO_GL") == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"


# Configure headless rendering before importing MuJoCo
_configure_headless_rendering()

# MuJoCo imports for visualization
try:
    import imageio
    import mujoco
    import mujoco.viewer

    MUJOCO_AVAILABLE = True
    logging.info(
        f"MuJoCo available with rendering backend: {os.environ.get('MUJOCO_GL', 'default')}"
    )
except ImportError as e:
    MUJOCO_AVAILABLE = False
    logging.warning(f"MuJoCo not available, visualization will be disabled: {e}")
except Exception as e:
    MUJOCO_AVAILABLE = False
    logging.warning(f"MuJoCo import failed, visualization will be disabled: {e}")


class MuJoCoVisualizer:
    """
    Standalone MuJoCo visualizer supporting interactive viewing and offline rendering.

    Features:
    - Interactive viewer with keyboard controls:
      - R: Reset to first frame
      - Space: Pause/unpause animation
      - N/P: Next/previous frame
      - G: Toggle ground truth robot visibility
      - T: Toggle predicted robot visibility
      - S: Toggle SMPL joints visibility
    - Offline rendering for video recording
    - SMPL joints visualization as 3D spheres
    - Side-by-side comparison support
    """

    def __init__(
        self,
        xml_path: str,
        enable_interactive: bool = True,
        enable_video_recording: bool = False,
        video_output_dir: str = "./videos",
        video_width: int = 1280,
        video_height: int = 720,
        video_fps: int = 30,
        smpl_sphere_radius: float = 0.02,
        fps: int = 30,
        realtime_mode: bool = False,
    ):
        """
        Initialize MuJoCo visualizer.

        Args:
            xml_path: Path to MuJoCo XML model file
            enable_interactive: Enable interactive viewer
            enable_video_recording: Enable video recording
            video_output_dir: Directory for video output
            video_width: Video width in pixels
            video_height: Video height in pixels
            video_fps: Video frame rate
            smpl_sphere_radius: Radius of SMPL joint spheres
            fps: Simulation/animation frame rate
            realtime_mode: If True, only visualize latest frame without buffering (default: False)
        """
        self.xml_path = xml_path
        self.enable_interactive = enable_interactive and MUJOCO_AVAILABLE
        self.enable_video_recording = enable_video_recording and MUJOCO_AVAILABLE
        self.realtime_mode = realtime_mode

        # Video recording parameters
        self.video_output_dir = video_output_dir
        self.video_width = video_width
        self.video_height = video_height
        self.video_fps = video_fps
        self.video_writer = None
        self.offscreen_renderer = None
        self.camera = None

        # MuJoCo visualization state
        self.mj_model = None
        self.mj_data = None
        self.viewer = None
        self.viewer_thread = None

        # Animation data buffers
        if self.realtime_mode:
            # Real-time mode: only store latest frame
            self.latest_qpos_gt = None
            self.latest_qpos_pred = None
            self.latest_smpl_joints_gt = None
            self.latest_smpl_joints_pred = None
            logging.info("MuJoCo visualizer initialized in REAL-TIME mode (latest frame only)")
        else:
            # Buffered mode: store full trajectory
            self.qpos_gt_buffer = (
                []
            )  # Ground truth qpos (translation + quaternion + joint positions)
            self.qpos_pred_buffer = (
                []
            )  # Predicted qpos (translation + quaternion + joint positions)
            self.smpl_joints_gt_buffer = []  # Ground truth SMPL joints (B x J x 3)
            self.smpl_joints_pred_buffer = []  # Predicted SMPL joints (B x J x 3)
            logging.info("MuJoCo visualizer initialized in BUFFER mode (full trajectory)")

        # Animation control
        self.current_frame = 0
        self.paused = False
        self.fps = fps
        self.dt = 1.0 / self.fps

        # Visibility toggles
        self.show_gt = True  # Show ground truth robot
        self.show_pred = True  # Show predicted robot
        self.show_smpl_joints = True  # Show SMPL joints as spheres

        # SMPL visualization
        self.sphere_radius = smpl_sphere_radius
        self.smpl_sphere_sites = []  # List to store SMPL joint sphere site IDs

        # Initialize MuJoCo model
        self._init_mujoco_model()

    def _create_xml_with_smpl_sites(self) -> str:
        """Create a modified XML file with SMPL joint sites"""
        # Read the original XML file
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # Fix include paths to be absolute
        xml_dir = os.path.dirname(os.path.abspath(self.xml_path))
        for include_elem in root.findall("include"):
            file_attr = include_elem.get("file")
            if file_attr and not os.path.isabs(file_attr):
                # Convert relative path to absolute path
                abs_path = os.path.join(xml_dir, file_attr)
                include_elem.set("file", abs_path)

        # Find the worldbody element
        worldbody = root.find("worldbody")
        if worldbody is None:
            return self.xml_path  # Return original if no worldbody found

        # Add SMPL joint sites for ground truth (blue spheres)
        for j in range(24):
            site = ET.SubElement(worldbody, "site")
            site.set("name", f"smpl_gt_joint_{j}")
            site.set("pos", "0 0 0")  # Will be updated dynamically
            site.set("size", str(self.sphere_radius))
            site.set("rgba", "0 0 1 0.8")  # Blue for GT
            site.set("type", "sphere")

        # Add SMPL joint sites for predictions (red spheres)
        for j in range(24):
            site = ET.SubElement(worldbody, "site")
            site.set("name", f"smpl_pred_joint_{j}")
            site.set("pos", "0 0 0")  # Will be updated dynamically
            site.set("size", str(self.sphere_radius))
            site.set("rgba", "1 0 0 0.8")  # Red for predictions
            site.set("type", "sphere")

        # Save the modified XML to a temporary file in the same directory as the original
        temp_xml = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, dir=xml_dir)
        tree.write(temp_xml.name, encoding="unicode", xml_declaration=True)
        temp_xml.close()

        return temp_xml.name

    def _init_mujoco_model(self):
        """Initialize MuJoCo model and data"""
        if not self.enable_interactive and not self.enable_video_recording:
            return

        # Log current rendering configuration
        current_backend = os.environ.get("MUJOCO_GL", "default")
        logging.info(f"Initializing MuJoCo model with rendering backend: {current_backend}")

        try:
            # Create XML with SMPL sites
            xml_path = self._create_xml_with_smpl_sites()
            logging.info(f"Created modified XML with SMPL sites: {xml_path}")

            self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
            self.mj_data = mujoco.MjData(self.mj_model)
            self.mj_model.opt.timestep = self.dt
            logging.info("MuJoCo model loaded successfully with SMPL joint sites")

            # Clean up temporary XML file if it's different from original
            if xml_path != self.xml_path:
                try:
                    os.unlink(xml_path)
                    logging.info(f"Cleaned up temporary XML file: {xml_path}")
                except Exception as cleanup_e:
                    logging.warning(
                        f"Failed to clean up temporary XML file {xml_path}: {cleanup_e}"
                    )

            # Initialize offline renderer for video recording
            if self.enable_video_recording:
                self._init_offscreen_renderer()

        except Exception as e:
            logging.error(f"Failed to load MuJoCo model with {current_backend}: {e}")
            logging.info(f"Falling back to original XML file: {self.xml_path}")

            # Try to load the original XML file as fallback
            try:
                self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
                self.mj_data = mujoco.MjData(self.mj_model)
                self.mj_model.opt.timestep = self.dt
                logging.info("Successfully loaded original MuJoCo model (without SMPL sites)")

                # Disable SMPL joints visualization since sites weren't added
                self.show_smpl_joints = False

                if self.enable_video_recording:
                    self._init_offscreen_renderer()

            except Exception as fallback_e:
                logging.error(f"Failed to load original MuJoCo model as fallback: {fallback_e}")

                # If we're in a headless environment, disable interactive visualization but keep video recording
                if current_backend in ["egl", "osmesa"]:
                    logging.warning(
                        (
                            "Headless environment detected, disabling interactive "
                            "visualization but keeping video recording"
                        )
                    )
                    self.enable_interactive = False
                    # Try to keep video recording enabled if possible
                    if self.enable_video_recording:
                        try:
                            self._init_offscreen_renderer()
                        except Exception as video_e:
                            logging.error(f"Video recording also failed: {video_e}")
                            self.enable_video_recording = False
                else:
                    self.enable_interactive = False
                    self.enable_video_recording = False

    def _init_offscreen_renderer(self):
        """Initialize MuJoCo offscreen renderer for video recording"""
        if not self.enable_video_recording or self.mj_model is None:
            return

        try:
            # Ensure headless rendering is configured
            current_backend = os.environ.get("MUJOCO_GL", "default")
            logging.info(f"Initializing offscreen renderer with backend: {current_backend}")

            # Create offscreen rendering context
            self.offscreen_renderer = mujoco.Renderer(
                self.mj_model, height=self.video_height, width=self.video_width
            )

            # Create camera for rendering
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.camera)

            # Set camera parameters for side-by-side view
            self.camera.distance = 3.5
            self.camera.azimuth = 180.0
            self.camera.elevation = -0.0
            self.camera.lookat[:] = [0.0, 0.0, 0.5]  # Look at center between robots

            logging.info(
                f"Offscreen renderer initialized successfully - "
                f"Resolution: {self.video_width}x{self.video_height} @ "
                f"{self.video_fps} FPS"
            )

        except Exception as e:
            logging.error(f"Failed to initialize offscreen renderer with {current_backend}: {e}")

            # Try fallback to OSMesa if EGL failed
            if current_backend == "egl":
                logging.info("Attempting fallback to OSMesa for software rendering...")
                try:
                    os.environ["MUJOCO_GL"] = "osmesa"
                    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
                    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

                    # Recreate renderer with OSMesa
                    self.offscreen_renderer = mujoco.Renderer(
                        self.mj_model, height=self.video_height, width=self.video_width
                    )

                    # Create camera for rendering
                    self.camera = mujoco.MjvCamera()
                    mujoco.mjv_defaultCamera(self.camera)

                    # Set camera parameters for side-by-side view
                    self.camera.distance = 3.5
                    self.camera.azimuth = 90.0
                    self.camera.elevation = -0.0
                    self.camera.lookat[:] = [0.0, 0.0, 0.5]

                    logging.info(
                        f"OSMesa fallback successful - Resolution: "
                        f"{self.video_width}x{self.video_height} @ "
                        f"{self.video_fps} FPS"
                    )

                except Exception as fallback_e:
                    logging.error(f"OSMesa fallback also failed: {fallback_e}")
                    self.enable_video_recording = False
            else:
                self.enable_video_recording = False

    def _key_callback(self, keycode):
        """Keyboard callback for MuJoCo viewer"""
        if chr(keycode) == "R":
            print("Reset")
            self.current_frame = 0
        elif chr(keycode) == " ":
            print("Paused")
            self.paused = not self.paused
        elif chr(keycode) == "N":
            print("Next frame")
            max_frames = max(len(self.qpos_gt_buffer), len(self.qpos_pred_buffer))
            if self.current_frame < max_frames - 1:
                self.current_frame += 1
        elif chr(keycode) == "P":
            print("Previous frame")
            if self.current_frame > 0:
                self.current_frame -= 1
        elif chr(keycode) == "G":
            self.show_gt = not self.show_gt
            print(f"Ground truth robot: {'ON' if self.show_gt else 'OFF'}")
        elif chr(keycode) == "T":
            self.show_pred = not self.show_pred
            print(f"Predicted robot: {'ON' if self.show_pred else 'OFF'}")
        elif chr(keycode) == "S":
            self.show_smpl_joints = not self.show_smpl_joints
            print(f"SMPL joints: {'ON' if self.show_smpl_joints else 'OFF'}")
        else:
            print(
                (
                    "Controls: R=Reset, Space=Pause, N=Next frame, P=Previous frame, "
                    "G=Toggle GT robot, T=Toggle predicted robot, S=Toggle SMPL joints"
                )
            )

    def _update_smpl_joints(self, frame_idx):
        """Update SMPL joint positions for current frame"""
        if not self.show_smpl_joints or self.mj_data is None:
            return

        # Get SMPL joints for current frame
        gt_joints = None
        pred_joints = None

        if self.realtime_mode:
            # Real-time mode: use latest joints
            gt_joints = self.latest_smpl_joints_gt
            pred_joints = self.latest_smpl_joints_pred
        else:
            # Buffered mode: use frame index
            if frame_idx < len(self.smpl_joints_gt_buffer):
                gt_joints = self.smpl_joints_gt_buffer[frame_idx]
            if frame_idx < len(self.smpl_joints_pred_buffer):
                pred_joints = self.smpl_joints_pred_buffer[frame_idx]

        # Update site positions for SMPL joints
        try:
            # Update GT SMPL joint sites (blue spheres)
            if gt_joints is not None and self.show_gt:
                for j in range(min(gt_joints.shape[0], 24)):  # Ensure we don't exceed 24 joints
                    site_name = f"smpl_gt_joint_{j}"
                    site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                    if site_id >= 0:
                        pos = gt_joints[j].copy()
                        # Adjust position to match GT robot position (left side)
                        pos[0] -= 1.0  # Move to left side like GT robot
                        pos[2] += 0.793  # Height adjustment
                        self.mj_data.site_xpos[site_id] = pos

            # Update predicted SMPL joint sites (red spheres)
            if pred_joints is not None and self.show_pred:
                for j in range(min(pred_joints.shape[0], 24)):  # Ensure we don't exceed 24 joints
                    site_name = f"smpl_pred_joint_{j}"
                    site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                    if site_id >= 0:
                        pos = pred_joints[j].copy()
                        # Adjust position to match predicted robot position (right side)
                        pos[0] += 1.0  # Move to right side like predicted robot
                        pos[2] += 0.793  # Height adjustment
                        self.mj_data.site_xpos[site_id] = pos

        except Exception:
            # Silently handle any rendering errors to avoid crashing the viewer
            pass

    def _update_robot_poses(self, frame_idx):
        """Update robot poses for current frame"""
        if self.mj_data is None:
            return

        if self.realtime_mode:
            # Real-time mode: use latest frames
            # Ground truth robot (left side) - first robot in the model
            if self.show_gt and self.latest_qpos_gt is not None:
                qpos_gt = self.latest_qpos_gt

                # Set GT robot full qpos (translation + quaternion + joint positions)
                if qpos_gt.shape[0] >= 36:  # Full qpos: 3 (trans) + 4 (quat) + 29 (joints)
                    self.mj_data.qpos[0:36] = qpos_gt[:36]  # GT robot full qpos
                else:
                    self.mj_data.qpos[0 : qpos_gt.shape[0]] = qpos_gt

                # Adjust GT robot position for side-by-side visualization
                self.mj_data.qpos[0] = -1.0  # Move GT robot to left side
            elif not self.show_gt:
                # Hide GT robot by moving it far away
                self.mj_data.qpos[0:3] = [-100, 0, -10]

            # Predicted robot (right side) - second robot in the model
            if self.show_pred and self.latest_qpos_pred is not None:
                qpos_pred = self.latest_qpos_pred

                # Set predicted robot full qpos (second robot in dual robot scene)
                pred_start_idx = 36  # After GT robot's full qpos (36 DOFs)
                if qpos_pred.shape[0] >= 36:  # Full qpos: 3 (trans) + 4 (quat) + 29 (joints)
                    self.mj_data.qpos[pred_start_idx : pred_start_idx + 36] = qpos_pred[:36]
                else:
                    self.mj_data.qpos[pred_start_idx : pred_start_idx + qpos_pred.shape[0]] = (
                        qpos_pred
                    )

                # Adjust predicted robot position for side-by-side visualization
                self.mj_data.qpos[pred_start_idx + 0] = 1.0  # Move predicted robot to right side
            elif not self.show_pred:
                # Hide predicted robot by moving it far away
                pred_start_idx = 36
                self.mj_data.qpos[pred_start_idx : pred_start_idx + 3] = [100, 0, -10]
        else:
            # Buffered mode: use frame index
            max_frames = max(len(self.qpos_gt_buffer), len(self.qpos_pred_buffer))
            if frame_idx >= max_frames:
                return

            # Ground truth robot (left side) - first robot in the model
            if self.show_gt and frame_idx < len(self.qpos_gt_buffer):
                qpos_gt = self.qpos_gt_buffer[frame_idx]

                # Set GT robot full qpos (translation + quaternion + joint positions)
                if qpos_gt.shape[0] >= 36:  # Full qpos: 3 (trans) + 4 (quat) + 29 (joints)
                    self.mj_data.qpos[0:36] = qpos_gt[:36]  # GT robot full qpos
                else:
                    self.mj_data.qpos[0 : qpos_gt.shape[0]] = qpos_gt

                # Adjust GT robot position for side-by-side visualization
                self.mj_data.qpos[0] = -1.0  # Move GT robot to left side
            elif not self.show_gt:
                # Hide GT robot by moving it far away
                self.mj_data.qpos[0:3] = [-100, 0, -10]

            # Predicted robot (right side) - second robot in the model
            if self.show_pred and frame_idx < len(self.qpos_pred_buffer):
                qpos_pred = self.qpos_pred_buffer[frame_idx]

                # Set predicted robot full qpos (second robot in dual robot scene)
                pred_start_idx = 36  # After GT robot's full qpos (36 DOFs)
                if qpos_pred.shape[0] >= 36:  # Full qpos: 3 (trans) + 4 (quat) + 29 (joints)
                    self.mj_data.qpos[pred_start_idx : pred_start_idx + 36] = qpos_pred[:36]
                else:
                    self.mj_data.qpos[pred_start_idx : pred_start_idx + qpos_pred.shape[0]] = (
                        qpos_pred
                    )

                # Adjust predicted robot position for side-by-side visualization
                self.mj_data.qpos[pred_start_idx + 0] = 1.0  # Move predicted robot to right side
            elif not self.show_pred:
                # Hide predicted robot by moving it far away
                pred_start_idx = 36
                self.mj_data.qpos[pred_start_idx : pred_start_idx + 3] = [100, 0, -10]

    def _run_interactive_viewer(self):
        """Run MuJoCo viewer in a separate thread"""
        if not self.enable_interactive or self.mj_model is None:
            return

        try:
            with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=self._key_callback
            ) as viewer:
                self.viewer = viewer
                # Set camera position
                viewer.cam.distance = 15.0
                viewer.cam.azimuth = 90.0
                viewer.cam.elevation = -20.0

                while viewer.is_running():
                    step_start = time.time()

                    if self.realtime_mode:
                        # Real-time mode: always show latest frame
                        if self.latest_qpos_gt is not None or self.latest_qpos_pred is not None:
                            # Update robot poses
                            self._update_robot_poses(0)  # frame_idx not used in realtime mode

                            # Forward simulation to update visualization
                            mujoco.mj_forward(self.mj_model, self.mj_data)

                            # Update SMPL joints
                            self._update_smpl_joints(0)  # frame_idx not used in realtime mode

                            viewer.sync()
                    else:
                        # Buffered mode: iterate through frames
                        if len(self.qpos_gt_buffer) > 0 or len(self.qpos_pred_buffer) > 0:

                            # Update robot poses
                            self._update_robot_poses(self.current_frame)

                            # Forward simulation to update visualization
                            mujoco.mj_forward(self.mj_model, self.mj_data)

                            # Update SMPL joints
                            self._update_smpl_joints(self.current_frame)

                            # Auto-advance frames if not paused
                            max_frames = max(len(self.qpos_gt_buffer), len(self.qpos_pred_buffer))
                            if not self.paused and max_frames > 1:
                                self.current_frame = (self.current_frame + 1) % max_frames

                            viewer.sync()

                    # Control frame rate
                    time_until_next_step = self.dt - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

        except Exception as e:
            logging.error(f"Failed to launch MuJoCo viewer: {e}")
            logging.info("Disabling interactive visualization, keeping video recording if enabled")
            self.enable_interactive = False

    def _render_offline_frame(self, frame_idx):
        """Render a single frame using offline renderer for video recording"""
        if not self.enable_video_recording or self.offscreen_renderer is None:
            return None

        try:
            # Update robot poses
            self._update_robot_poses(frame_idx)

            # Forward simulation to update visualization
            mujoco.mj_forward(self.mj_model, self.mj_data)

            # Update SMPL joint positions for offline rendering
            self._update_smpl_joints(frame_idx)

            # Update scene and render frame
            self.offscreen_renderer.update_scene(self.mj_data, camera=self.camera)
            frame = self.offscreen_renderer.render()

            return frame

        except Exception as e:
            logging.error(f"Error rendering offline frame {frame_idx}: {e}")
            return None

    def add_trajectory_data(
        self,
        qpos_gt: Optional[Union[np.ndarray, torch.Tensor]] = None,
        qpos_pred: Optional[Union[np.ndarray, torch.Tensor]] = None,
        smpl_joints_gt: Optional[Union[np.ndarray, torch.Tensor]] = None,
        smpl_joints_pred: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Add trajectory data to visualization buffers or update latest frame (realtime mode).

        Args:
            qpos_gt: Ground truth joint positions (B, T, DOF) or (T, DOF) or (DOF,)
            qpos_pred: Predicted joint positions (B, T, DOF) or (T, DOF) or (DOF,)
            smpl_joints_gt: Ground truth SMPL joints (B, T, 24, 3) or (T, 24, 3) or (24, 3)
            smpl_joints_pred: Predicted SMPL joints (B, T, 24, 3) or (T, 24, 3) or (24, 3)
        """
        # Convert tensors to numpy arrays
        if qpos_gt is not None:
            if torch.is_tensor(qpos_gt):
                qpos_gt = qpos_gt.detach().cpu().numpy()

            if self.realtime_mode:
                # Real-time mode: just store the latest frame
                if qpos_gt.ndim > 1:
                    self.latest_qpos_gt = (
                        qpos_gt[-1]
                        if qpos_gt.ndim == 2
                        else qpos_gt.reshape(-1, qpos_gt.shape[-1])[-1]
                    )
                else:
                    self.latest_qpos_gt = qpos_gt
            else:
                # Buffered mode: add to buffer
                self._add_qpos_data(qpos_gt, self.qpos_gt_buffer)

        if qpos_pred is not None:
            if torch.is_tensor(qpos_pred):
                qpos_pred = qpos_pred.detach().cpu().numpy()

            if self.realtime_mode:
                # Real-time mode: just store the latest frame
                if qpos_pred.ndim > 1:
                    self.latest_qpos_pred = (
                        qpos_pred[-1]
                        if qpos_pred.ndim == 2
                        else qpos_pred.reshape(-1, qpos_pred.shape[-1])[-1]
                    )
                else:
                    self.latest_qpos_pred = qpos_pred
            else:
                # Buffered mode: add to buffer
                self._add_qpos_data(qpos_pred, self.qpos_pred_buffer)

        if smpl_joints_gt is not None:
            if torch.is_tensor(smpl_joints_gt):
                smpl_joints_gt = smpl_joints_gt.detach().cpu().numpy()

            if self.realtime_mode:
                # Real-time mode: just store the latest frame
                if smpl_joints_gt.ndim == 2:  # (24, 3)
                    self.latest_smpl_joints_gt = smpl_joints_gt
                elif smpl_joints_gt.ndim == 3:  # (T, 24, 3) or (B, 24, 3)
                    self.latest_smpl_joints_gt = smpl_joints_gt[-1]
                elif smpl_joints_gt.ndim == 4:  # (B, T, 24, 3)
                    self.latest_smpl_joints_gt = smpl_joints_gt.reshape(-1, 24, 3)[-1]
            else:
                # Buffered mode: add to buffer
                self._add_smpl_data(smpl_joints_gt, self.smpl_joints_gt_buffer)

        if smpl_joints_pred is not None:
            if torch.is_tensor(smpl_joints_pred):
                smpl_joints_pred = smpl_joints_pred.detach().cpu().numpy()

            if self.realtime_mode:
                # Real-time mode: just store the latest frame
                if smpl_joints_pred.ndim == 2:  # (24, 3)
                    self.latest_smpl_joints_pred = smpl_joints_pred
                elif smpl_joints_pred.ndim == 3:  # (T, 24, 3) or (B, 24, 3)
                    self.latest_smpl_joints_pred = smpl_joints_pred[-1]
                elif smpl_joints_pred.ndim == 4:  # (B, T, 24, 3)
                    self.latest_smpl_joints_pred = smpl_joints_pred.reshape(-1, 24, 3)[-1]
            else:
                # Buffered mode: add to buffer
                self._add_smpl_data(smpl_joints_pred, self.smpl_joints_pred_buffer)

    def _add_qpos_data(self, qpos_data: np.ndarray, buffer: List):
        """Add qpos data to buffer, handling different dimensions"""
        if qpos_data.ndim == 3:  # (batch, seq, qpos)
            for b in range(qpos_data.shape[0]):
                for t in range(qpos_data.shape[1]):
                    buffer.append(qpos_data[b, t])
        elif qpos_data.ndim == 2:  # (seq, qpos) or (batch, qpos)
            if qpos_data.shape[1] > 50:  # Assume (seq, qpos) if many DOFs
                for t in range(qpos_data.shape[0]):
                    buffer.append(qpos_data[t])
            else:  # Assume (batch, qpos)
                for b in range(qpos_data.shape[0]):
                    buffer.append(qpos_data[b])
        else:  # Single frame
            buffer.append(qpos_data)

    def _add_smpl_data(self, smpl_data: np.ndarray, buffer: List):
        """Add SMPL joints data to buffer, handling different dimensions"""
        # Reshape to ensure proper format and center joints
        if smpl_data.ndim == 4:  # (batch, seq, joints, 3)
            smpl_data = smpl_data.reshape(-1, 24, 3)
        elif smpl_data.ndim == 3:  # (seq, joints, 3) or (batch, joints, 3)
            if smpl_data.shape[1] == 24:  # (seq, 24, 3) or (batch, 24, 3)
                smpl_data = smpl_data.reshape(-1, 24, 3)
        elif smpl_data.ndim == 2:  # (joints, 3)
            smpl_data = smpl_data.reshape(1, 24, 3)

        # Center joints relative to root joint (joint 0)
        smpl_data = smpl_data - smpl_data[:, [0], :]

        # Add to buffer
        for i in range(smpl_data.shape[0]):
            buffer.append(smpl_data[i])

    def start_interactive_viewer(self):
        """Start interactive viewer in a separate thread"""
        if self.enable_interactive and (
            self.viewer_thread is None or not self.viewer_thread.is_alive()
        ):
            self.viewer_thread = threading.Thread(target=self._run_interactive_viewer, daemon=True)
            self.viewer_thread.start()
            logging.info("Started MuJoCo interactive visualization thread")

    def create_video(self, output_path: str, clear_buffers: bool = True) -> bool:
        """
        Create video from stored trajectory data using offline rendering.

        Args:
            output_path: Path for output video file
            clear_buffers: Whether to clear buffers after creating video

        Returns:
            True if video was created successfully, False otherwise
        """
        if not self.enable_video_recording or (
            len(self.qpos_gt_buffer) == 0 and len(self.qpos_pred_buffer) == 0
        ):
            logging.warning("Video recording not enabled or no data available")
            return False

        logging.info(f"Creating video: {output_path}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Initialize video writer
        try:
            video_writer = imageio.get_writer(
                output_path, fps=self.video_fps, codec="libx264", quality=4, pixelformat="yuv420p"
            )
        except Exception as e:
            logging.error(f"Failed to create video writer: {e}")
            return False

        max_frames = max(len(self.qpos_gt_buffer), len(self.qpos_pred_buffer))

        try:
            for frame_idx in range(max_frames):
                frame = self._render_offline_frame(frame_idx)
                if frame is not None:
                    video_writer.append_data(frame)

                # Log progress every 10% of frames
                if frame_idx % max(1, max_frames // 10) == 0:
                    progress = (frame_idx + 1) / max_frames * 100
                    logging.info(
                        f"Video rendering progress: {progress:.1f}% ({frame_idx + 1}/{max_frames})"
                    )

            video_writer.close()
            logging.info(f"Video saved successfully: {output_path}")

            if clear_buffers:
                self.clear_buffers()

            return True

        except Exception as e:
            logging.error(f"Error creating video: {e}")
            video_writer.close()
            return False

    def clear_buffers(self):
        """Clear all trajectory data buffers"""
        if self.realtime_mode:
            self.latest_qpos_gt = None
            self.latest_qpos_pred = None
            self.latest_smpl_joints_gt = None
            self.latest_smpl_joints_pred = None
            logging.info("Cleared latest frame data (realtime mode)")
        else:
            self.qpos_gt_buffer.clear()
            self.qpos_pred_buffer.clear()
            self.smpl_joints_gt_buffer.clear()
            self.smpl_joints_pred_buffer.clear()
            self.current_frame = 0
            logging.info("Cleared all trajectory buffers")

    def set_camera_params(
        self,
        distance: float = 3.5,
        azimuth: float = 90.0,
        elevation: float = 0.0,
        lookat: List[float] = [0.0, 0.0, 0.5],
    ):
        """Set camera parameters for offline rendering"""
        if self.camera is not None:
            self.camera.distance = distance
            self.camera.azimuth = azimuth
            self.camera.elevation = elevation
            self.camera.lookat[:] = lookat
            logging.info(
                f"Camera parameters updated: "
                f"distance={distance}, "
                f"azimuth={azimuth}, "
                f"elevation={elevation}, "
                f"lookat={lookat}"
            )

    def get_status(self) -> Dict:
        """Get current status of the visualizer"""
        status = {
            "mujoco_available": MUJOCO_AVAILABLE,
            "interactive_enabled": self.enable_interactive,
            "video_recording_enabled": self.enable_video_recording,
            "realtime_mode": self.realtime_mode,
            "model_loaded": self.mj_model is not None,
            "paused": self.paused,
            "show_gt": self.show_gt,
            "show_pred": self.show_pred,
            "show_smpl_joints": self.show_smpl_joints,
            "viewer_running": self.viewer_thread is not None and self.viewer_thread.is_alive(),
        }

        if self.realtime_mode:
            status.update(
                {
                    "has_gt_data": self.latest_qpos_gt is not None,
                    "has_pred_data": self.latest_qpos_pred is not None,
                    "has_smpl_gt_data": self.latest_smpl_joints_gt is not None,
                    "has_smpl_pred_data": self.latest_smpl_joints_pred is not None,
                }
            )
        else:
            status.update(
                {
                    "gt_frames": len(self.qpos_gt_buffer),
                    "pred_frames": len(self.qpos_pred_buffer),
                    "smpl_gt_frames": len(self.smpl_joints_gt_buffer),
                    "smpl_pred_frames": len(self.smpl_joints_pred_buffer),
                    "current_frame": self.current_frame,
                }
            )

        return status

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, "video_writer") and self.video_writer is not None:
            self.video_writer.close()
