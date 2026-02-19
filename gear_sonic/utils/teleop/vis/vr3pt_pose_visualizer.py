"""
VR 3-Point Pose Visualizer

A standalone PyVista-based visualizer for VR 3-point pose data (Head, Left Wrist, Right Wrist).
Can be used by any process that provides pose data as numpy arrays.

Coordinate convention:
- X: forward (RED axis)
- Y: left (GREEN axis)
- Z: up (BLUE axis)

Quaternion format: [qw, qx, qy, qz] (scalar-first)

Usage:
    # Basic static visualization with reference frames
    visualizer = VR3PtPoseVisualizer()
    visualizer.show_static()

    # Visualize with pose data
    vr_3pt_pose = np.array([...])  # Shape (3, 7): [x, y, z, qw, qx, qy, qz] for each point
    visualizer.show_with_vr_pose(vr_3pt_pose)

    # Real-time visualization (requires update callback)
    visualizer.create_realtime_plotter()
    # In your loop:
    visualizer.update_vr_poses(vr_3pt_pose)
    visualizer.render()

    # Visualization with G1 robot model
    visualizer = VR3PtPoseVisualizer(with_g1_robot=True)
    visualizer.show_with_vr_pose(vr_3pt_pose)  # G1 robot will be shown at origin
"""

import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as sRot

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    pv = None
    PYVISTA_AVAILABLE = False

try:
    import vtk

    VTK_AVAILABLE = True
except ImportError:
    vtk = None
    VTK_AVAILABLE = False

try:
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except ImportError:
    pin = None
    PINOCCHIO_AVAILABLE = False


# =============================================================================
# Shared FK constants and function (display-independent, only needs Pinocchio)
# =============================================================================

# Key frame names for FK pose extraction
G1_LEFT_WRIST_FRAME = "left_wrist_yaw_link"
G1_RIGHT_WRIST_FRAME = "right_wrist_yaw_link"
G1_TORSO_FRAME = "torso_link"

# Key frame offsets applied in the local frame of each link
# (from gear_sonic/config/manager_env/commands/terms/force.yaml)
G1_KEY_FRAME_OFFSETS = {
    "left_wrist": np.array([0.18, -0.025, 0.0]),
    "right_wrist": np.array([0.18, 0.025, 0.0]),
    "torso": np.array([0.0, 0.0, 0.35]),
}

G1_FRAME_MAPPING = {
    "left_wrist": G1_LEFT_WRIST_FRAME,
    "right_wrist": G1_RIGHT_WRIST_FRAME,
    "torso": G1_TORSO_FRAME,
}


def get_g1_key_frame_poses(
    robot_model,
    q: np.ndarray = None,
    root_position: np.ndarray = None,
    apply_offset: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Get poses (position + orientation) of G1 key frames using Pinocchio FK.

    This is a **display-independent** function — it only needs a Pinocchio robot
    model, no PyVista/VTK/display. It can be used by both the visualizer and
    headless calibration code.

    Args:
        robot_model: Pinocchio-based robot model with cache_forward_kinematics()
                     and frame_placement() methods.
        q: Joint configuration. If None, uses robot_model.default_body_pose.
        root_position: Position of robot root. Default is origin [0, 0, 0].
        apply_offset: Whether to apply the local frame offsets. Default True.

    Returns:
        Dict with keys 'left_wrist', 'right_wrist', 'torso', each containing:
            - 'position': np.ndarray [x, y, z] (with offset applied in local frame)
            - 'orientation_xyzw': np.ndarray [qx, qy, qz, qw] (scipy/ROS convention)
            - 'orientation_wxyz': np.ndarray [qw, qx, qy, qz] (scalar-first convention)
    """
    if q is None:
        q = robot_model.default_body_pose
    if root_position is None:
        root_position = np.array([0.0, 0.0, 0.0])

    # Update forward kinematics
    robot_model.cache_forward_kinematics(q, auto_clip=False)

    result = {}
    for key, frame_name in G1_FRAME_MAPPING.items():
        # Get frame placement from Pinocchio — if the frame doesn't exist,
        # this is a fatal configuration error (wrong URDF or frame name).
        try:
            placement = robot_model.frame_placement(frame_name)
        except ValueError as e:
            raise RuntimeError(
                f"Cannot find frame '{frame_name}' (key='{key}') in robot model. "
                f"Ensure the URDF contains this frame. Original error: {e}"
            ) from e

        rotation_matrix = placement.rotation

        # Apply offset in local frame, then transform to world frame
        if apply_offset and key in G1_KEY_FRAME_OFFSETS:
            local_offset = G1_KEY_FRAME_OFFSETS[key]
            world_offset = rotation_matrix @ local_offset
            position = placement.translation + world_offset + root_position
        else:
            position = placement.translation + root_position

        # Convert rotation matrix to quaternion using scipy
        rot = sRot.from_matrix(rotation_matrix)
        quat_xyzw = rot.as_quat()  # scipy returns [qx, qy, qz, qw]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        result[key] = {
            "position": position.copy(),
            "orientation_xyzw": quat_xyzw.copy(),
            "orientation_wxyz": quat_wxyz.copy(),
        }

    return result


class G1RobotVisualizer:
    """
    PyVista-based G1 robot visualizer that loads STL meshes and transforms them
    based on joint configurations using Pinocchio forward kinematics.

    The robot is placed at the origin (pelvis at [0, 0, 0]).
    """

    # Default robot color (dark gray for main body)
    ROBOT_COLOR = "#404040"
    ROBOT_OPACITY = 0.1  # Semi-transparent to see key points better

    # Key frame names and offsets — reference the shared module-level constants
    LEFT_WRIST_FRAME = G1_LEFT_WRIST_FRAME
    RIGHT_WRIST_FRAME = G1_RIGHT_WRIST_FRAME
    TORSO_FRAME = G1_TORSO_FRAME
    KEY_FRAME_OFFSETS = G1_KEY_FRAME_OFFSETS

    # Key point visualization colors
    KEY_POINT_COLORS = {
        "left_wrist": "lightgreen",
        "right_wrist": "lightblue",
        "torso": "yellow",
    }

    # Key point labels (with offset indicator)
    KEY_POINT_LABELS = {
        "left_wrist": "L-Wrist (with offset)",
        "right_wrist": "R-Wrist (with offset)",
        "torso": "Torso (with offset)",
    }

    # Waist joint names in the G1 robot (order: yaw, roll, pitch)
    WAIST_JOINT_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

    def __init__(self, robot_model=None):
        """
        Initialize the G1 robot visualizer.

        Args:
            robot_model: Optional pre-instantiated RobotModel. If None, will create one.
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required. Install with: pip install pyvista")
        if not PINOCCHIO_AVAILABLE:
            raise ImportError("Pinocchio is required. Install with: pip install pin")

        # Import here to avoid circular imports and make the dependency optional
        from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model

        self.robot_model = robot_model if robot_model is not None else instantiate_g1_robot_model()

        # Get paths for mesh files
        gear_sonic_root = Path(__file__).resolve().parent.parent.parent.parent
        self.mesh_dir = gear_sonic_root / "data" / "robot_model" / "model_data" / "g1" / "meshes"

        # Load mesh data from Pinocchio visual model
        self._load_visual_geometries()

        # Store actors for real-time updates
        self.mesh_actors: Dict[str, any] = {}
        self.key_point_actors: Dict[str, any] = {}
        self._initialized = False
        self._key_points_initialized = False

        # Cache waist joint indices for efficient updates
        self._waist_joint_indices: Optional[List[int]] = None
        try:
            self._waist_joint_indices = self.robot_model.get_joint_group_indices("waist")
        except (ValueError, AttributeError) as e:
            raise RuntimeError(
                f"Could not get waist joint indices from robot model. "
                f"Ensure the robot model supplemental info defines a 'waist' joint group. "
                f"Original error: {e}"
            ) from e

    def compute_waist_joints_from_orientation(
        self,
        neck_quat_wxyz: np.ndarray,
        scale_factor: float = 1.0,
    ) -> Optional[np.ndarray]:
        """
        Compute waist joint angles from VR neck orientation.

        The neck orientation from VR represents the upper body tilt. We decompose it
        into yaw, roll, pitch Euler angles and map them to the waist joints.

        Args:
            neck_quat_wxyz: Quaternion [qw, qx, qy, qz] representing neck orientation
                           (relative to root, already calibrated)
            scale_factor: Scale factor for joint angles (0.0-1.0), useful for limiting range

        Returns:
            np.ndarray of shape (3,) containing [waist_yaw, waist_roll, waist_pitch]
            or None if waist control is not available
        """
        if self._waist_joint_indices is None:
            return None

        # Convert quaternion to Euler angles (ZYX = yaw, pitch, roll in extrinsic)
        # G1 waist joints order: yaw, roll, pitch
        quat_xyzw = np.array(
            [neck_quat_wxyz[1], neck_quat_wxyz[2], neck_quat_wxyz[3], neck_quat_wxyz[0]]
        )
        rot = sRot.from_quat(quat_xyzw)

        # Use ZYX Euler convention: rotation about Z (yaw), then Y (pitch), then X (roll)
        # Output order is [z_angle, y_angle, x_angle] = [yaw, pitch, roll]
        euler_zyx = rot.as_euler("ZYX", degrees=False)

        # Map to waist joints: [waist_yaw, waist_roll, waist_pitch]
        # euler_zyx = [yaw, pitch, roll]
        waist_yaw = euler_zyx[0] * scale_factor
        waist_roll = euler_zyx[2] * scale_factor  # X rotation
        waist_pitch = euler_zyx[1] * scale_factor  # Y rotation

        return np.array([waist_yaw, waist_roll, waist_pitch])

    def apply_waist_joints_to_config(
        self,
        q: np.ndarray,
        waist_joints: np.ndarray,
    ) -> np.ndarray:
        """
        Apply waist joint angles to a robot configuration.

        Args:
            q: Full robot joint configuration array
            waist_joints: Array of shape (3,) containing [waist_yaw, waist_roll, waist_pitch]

        Returns:
            Updated robot configuration with waist joints set
        """
        if self._waist_joint_indices is None or waist_joints is None:
            return q

        q_new = q.copy()
        for i, idx in enumerate(self._waist_joint_indices):
            if i < len(waist_joints):
                q_new[idx] = waist_joints[i]
        return q_new

    def _load_visual_geometries(self):
        """Load visual geometry info from Pinocchio's visual model."""
        self.visual_geometries: List[Dict] = []

        visual_model = self.robot_model.pinocchio_wrapper.visual_model
        model = self.robot_model.pinocchio_wrapper.model

        if len(visual_model.geometryObjects) == 0:
            raise RuntimeError(
                "No visual geometries found in Pinocchio visual model. "
                "Check that the URDF file contains visual geometry elements."
            )

        for geom_id, geom in enumerate(visual_model.geometryObjects):
            # Get the mesh file path
            mesh_path = geom.meshPath
            if not mesh_path:
                print(
                    f"Warning: Visual geometry {geom_id} ({geom.name}) has no mesh path, skipping."
                )
                continue

            # Get the frame this geometry is attached to
            frame_id = geom.parentFrame
            frame_name = model.frames[frame_id].name if frame_id < len(model.frames) else None

            # Get the local placement (geometry relative to parent frame)
            local_placement = geom.placement

            self.visual_geometries.append(
                {
                    "geom_id": geom_id,
                    "mesh_path": str(mesh_path),
                    "frame_id": frame_id,
                    "frame_name": frame_name,
                    "local_placement": local_placement,
                    "mesh": None,  # Will be loaded later
                }
            )

    def _load_mesh(self, mesh_path: str) -> "pv.PolyData":
        """Load a mesh file and return PyVista PolyData.

        Raises:
            FileNotFoundError: If the mesh file cannot be found.
            RuntimeError: If the mesh file exists but cannot be loaded.
        """
        original_path = mesh_path
        if not os.path.exists(mesh_path):
            # Try relative to mesh_dir
            mesh_name = os.path.basename(mesh_path)
            mesh_path = str(self.mesh_dir / mesh_name)

        if not os.path.exists(mesh_path):
            raise FileNotFoundError(
                f"Robot mesh file not found: '{original_path}'\n"
                f"  Also tried: '{mesh_path}'\n"
                f"  Mesh directory: '{self.mesh_dir}'\n"
                f"  Please ensure robot mesh files (STL) are present."
            )

        try:
            return pv.read(mesh_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load robot mesh file '{mesh_path}': {e}"
            ) from e

    def _get_geometry_world_transform(
        self, geom_info: Dict, q: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the world transform (position, rotation matrix) for a visual geometry.

        Args:
            geom_info: Geometry info dict from visual_geometries
            q: Joint configuration. If None, uses q_zero.

        Returns:
            Tuple of (position [3], rotation_matrix [3x3])
        """
        if q is None:
            q = self.robot_model.q_zero

        # Update forward kinematics
        self.robot_model.cache_forward_kinematics(q, auto_clip=False)

        # Get the frame's world placement
        frame_placement = self.robot_model.pinocchio_wrapper.data.oMf[geom_info["frame_id"]]

        # Compose with local placement
        world_placement = frame_placement * geom_info["local_placement"]

        # Extract position and rotation
        position = world_placement.translation
        rotation = world_placement.rotation

        return position, rotation

    def add_to_plotter(
        self,
        plotter: "pv.Plotter",
        q: np.ndarray = None,
        color: str = None,
        opacity: float = None,
        root_position: np.ndarray = None,
    ) -> Dict[str, any]:
        """
        Add G1 robot meshes to a PyVista plotter.

        Args:
            plotter: PyVista plotter to add meshes to
            q: Joint configuration. If None, uses default body pose.
            color: Override mesh color
            opacity: Override mesh opacity
            root_position: Position of the robot root (pelvis). Default is origin [0, 0, 0].

        Returns:
            Dict mapping geometry names to actors for later updates
        """
        if q is None:
            q = self.robot_model.default_body_pose
        if color is None:
            color = self.ROBOT_COLOR
        if opacity is None:
            opacity = self.ROBOT_OPACITY
        if root_position is None:
            root_position = np.array([0.0, 0.0, 0.0])

        actors = {}

        for geom_info in self.visual_geometries:
            # Load mesh if not already loaded
            if geom_info["mesh"] is None:
                geom_info["mesh"] = self._load_mesh(geom_info["mesh_path"])

            # Get world transform
            position, rotation = self._get_geometry_world_transform(geom_info, q)

            # Apply root offset
            position = position + root_position

            # Create a copy of the mesh and transform it
            mesh = geom_info["mesh"].copy()

            # Build 4x4 transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = position

            mesh.transform(transform)

            # Add to plotter
            actor = plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                smooth_shading=True,
                name=geom_info["frame_name"],
            )
            actors[geom_info["frame_name"]] = {
                "actor": actor,
                "geom_info": geom_info,
            }

        self.mesh_actors = actors
        self._initialized = True
        return actors

    def add_to_plotter_realtime(
        self,
        plotter: "pv.Plotter",
        q: np.ndarray = None,
        color: str = None,
        opacity: float = None,
        root_position: np.ndarray = None,
    ) -> Dict[str, any]:
        """
        Add G1 robot meshes to a PyVista plotter for real-time updates.
        Uses VTK transforms for efficient updates without recreating meshes.

        Args:
            plotter: PyVista plotter to add meshes to
            q: Initial joint configuration. If None, uses default body pose.
            color: Override mesh color
            opacity: Override mesh opacity
            root_position: Position of the robot root (pelvis). Default is origin [0, 0, 0].

        Returns:
            Dict mapping geometry names to actors for later updates
        """
        if not VTK_AVAILABLE:
            raise ImportError("VTK is required for real-time mode")

        if q is None:
            q = self.robot_model.default_body_pose
        if color is None:
            color = self.ROBOT_COLOR
        if opacity is None:
            opacity = self.ROBOT_OPACITY
        if root_position is None:
            root_position = np.array([0.0, 0.0, 0.0])

        actors = {}

        for geom_info in self.visual_geometries:
            # Load mesh if not already loaded
            if geom_info["mesh"] is None:
                geom_info["mesh"] = self._load_mesh(geom_info["mesh_path"])

            # Add mesh without initial transform (we'll set it via VTK transform)
            mesh = geom_info["mesh"].copy()
            actor = plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                smooth_shading=True,
                name=geom_info["frame_name"],
            )

            actors[geom_info["frame_name"]] = {
                "actor": actor,
                "geom_info": geom_info,
            }

        self.mesh_actors = actors
        self._root_position = root_position
        self._initialized = True

        # Set initial pose
        self.update_pose(q, root_position)

        return actors

    def add_key_points_to_plotter(
        self,
        plotter: "pv.Plotter",
        q: np.ndarray = None,
        root_position: np.ndarray = None,
        axis_length: float = 0.08,
        ball_radius: float = 0.02,
        show_axes: bool = True,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Add key point visualizations (left wrist, right wrist, torso) to the plotter.

        Args:
            plotter: PyVista plotter
            q: Joint configuration. If None, uses default body pose.
            root_position: Position of robot root. Default is origin [0, 0, 0].
            axis_length: Length of coordinate frame axes
            ball_radius: Radius of position marker balls
            show_axes: Whether to show coordinate frame axes

        Returns:
            Dict with poses of each key point (same format as get_g1_key_frame_poses)
        """
        if q is None:
            q = self.robot_model.default_body_pose
        if root_position is None:
            root_position = np.array([0.0, 0.0, 0.0])

        # Get key frame poses
        poses = get_g1_key_frame_poses(self.robot_model, q=q, root_position=root_position)

        # Axis colors (RGB for XYZ)
        axis_colors = ["red", "green", "blue"]
        axis_dirs = [
            np.array([1, 0, 0]),  # X
            np.array([0, 1, 0]),  # Y
            np.array([0, 0, 1]),  # Z
        ]

        for key, pose in poses.items():
            position = pose["position"]
            quat_xyzw = pose["orientation_xyzw"]

            # Convert quaternion to rotation matrix
            rot = sRot.from_quat(quat_xyzw)
            rot_matrix = rot.as_matrix()

            if show_axes:
                # Add coordinate frame arrows
                for i, (color, local_dir) in enumerate(zip(axis_colors, axis_dirs)):
                    world_dir = rot_matrix @ local_dir
                    arrow = pv.Arrow(
                        start=position,
                        direction=world_dir,
                        scale=axis_length,
                        tip_length=0.3,
                        tip_radius=0.15,
                        shaft_radius=0.05,
                    )
                    plotter.add_mesh(arrow, color=color, smooth_shading=True)

            # Add colored ball at position
            ball = pv.Sphere(radius=ball_radius, center=position)
            plotter.add_mesh(
                ball, color=self.KEY_POINT_COLORS[key], smooth_shading=True, name=f"keypoint_{key}"
            )

            # Add label (with offset indicator)
            plotter.add_point_labels(
                [position + np.array([0, 0, ball_radius * 2])],
                [self.KEY_POINT_LABELS[key]],
                font_size=10,
                point_color=self.KEY_POINT_COLORS[key],
                text_color="white",
                always_visible=True,
                shape_opacity=0.7,
            )

        return poses

    def add_key_points_realtime(
        self,
        plotter: "pv.Plotter",
        q: np.ndarray = None,
        root_position: np.ndarray = None,
        axis_length: float = 0.08,
        ball_radius: float = 0.02,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Add key point visualizations for real-time updates.

        Args:
            plotter: PyVista plotter
            q: Initial joint configuration. If None, uses default body pose.
            root_position: Position of robot root. Default is origin [0, 0, 0].
            axis_length: Length of coordinate frame axes
            ball_radius: Radius of position marker balls

        Returns:
            Dict with initial poses of each key point
        """
        if not VTK_AVAILABLE:
            raise ImportError("VTK is required for real-time mode")

        if q is None:
            q = self.robot_model.default_body_pose
        if root_position is None:
            root_position = np.array([0.0, 0.0, 0.0])

        self.key_point_actors = {}

        # Axis colors (RGB for XYZ)
        axis_colors = ["red", "green", "blue"]

        for key in ["left_wrist", "right_wrist", "torso"]:
            actors = {"arrows": [], "ball": None}

            # Create arrows for each axis
            for color in axis_colors:
                arrow = pv.Arrow(
                    start=(0, 0, 0),
                    direction=(1, 0, 0),
                    scale=axis_length,
                    tip_length=0.3,
                    tip_radius=0.15,
                    shaft_radius=0.05,
                )
                actor = plotter.add_mesh(arrow, color=color, smooth_shading=True)
                actors["arrows"].append(actor)

            # Create ball
            ball = pv.Sphere(radius=ball_radius, center=(0, 0, 0))
            actors["ball"] = plotter.add_mesh(
                ball, color=self.KEY_POINT_COLORS[key], smooth_shading=True
            )

            self.key_point_actors[key] = actors

        self._key_points_initialized = True
        self._key_point_axis_length = axis_length

        # Set initial poses
        poses = self.update_key_points(q, root_position)
        return poses

    def update_key_points(
        self, q: np.ndarray, root_position: np.ndarray = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Update key point visualizations for real-time mode.

        Args:
            q: Joint configuration
            root_position: Optional new root position

        Returns:
            Dict with updated poses of each key point
        """
        if not self._key_points_initialized or not VTK_AVAILABLE:
            return {}

        if root_position is None:
            root_position = getattr(self, "_root_position", np.array([0.0, 0.0, 0.0]))

        # Get key frame poses
        poses = get_g1_key_frame_poses(self.robot_model, q=q, root_position=root_position)

        axis_dirs = [
            np.array([1, 0, 0]),  # X
            np.array([0, 1, 0]),  # Y
            np.array([0, 0, 1]),  # Z
        ]

        for key, pose in poses.items():
            if key not in self.key_point_actors:
                continue

            position = pose["position"]
            quat_xyzw = pose["orientation_xyzw"]

            # Convert quaternion to rotation matrix
            rot = sRot.from_quat(quat_xyzw)
            rot_matrix = rot.as_matrix()

            actors = self.key_point_actors[key]

            # Update each arrow's transform
            for j, local_dir in enumerate(axis_dirs):
                world_dir = rot_matrix @ local_dir

                # Compute rotation to align arrow (which points along X) to world_dir
                x_axis = np.array([1.0, 0.0, 0.0])
                v = np.cross(x_axis, world_dir)
                c = np.dot(x_axis, world_dir)

                if np.linalg.norm(v) > 1e-6:
                    s = np.linalg.norm(v)
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    arrow_rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-9))
                elif c < 0:
                    arrow_rot = np.diag([-1.0, 1.0, -1.0])
                else:
                    arrow_rot = np.eye(3)

                # Create VTK transform
                transform = vtk.vtkTransform()
                mat = vtk.vtkMatrix4x4()

                for ri in range(3):
                    for ci in range(3):
                        mat.SetElement(ri, ci, arrow_rot[ri, ci])
                mat.SetElement(0, 3, position[0])
                mat.SetElement(1, 3, position[1])
                mat.SetElement(2, 3, position[2])

                transform.SetMatrix(mat)
                actors["arrows"][j].SetUserTransform(transform)

            # Update ball position
            ball_transform = vtk.vtkTransform()
            ball_transform.Translate(position[0], position[1], position[2])
            actors["ball"].SetUserTransform(ball_transform)

        return poses

    def update_pose(
        self, q: np.ndarray, root_position: np.ndarray = None
    ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """
        Update robot pose for real-time visualization.

        Args:
            q: Joint configuration
            root_position: Optional new root position

        Returns:
            Dict with updated key frame poses if key points are initialized, else None
        """
        if not self._initialized or not VTK_AVAILABLE:
            return None

        if root_position is None:
            root_position = getattr(self, "_root_position", np.array([0.0, 0.0, 0.0]))
        else:
            self._root_position = root_position

        # Update forward kinematics once
        self.robot_model.cache_forward_kinematics(q, auto_clip=False)

        for name, actor_info in self.mesh_actors.items():
            geom_info = actor_info["geom_info"]
            actor = actor_info["actor"]

            # Get the frame's world placement
            frame_placement = self.robot_model.pinocchio_wrapper.data.oMf[geom_info["frame_id"]]
            world_placement = frame_placement * geom_info["local_placement"]

            position = world_placement.translation + root_position
            rotation = world_placement.rotation

            # Create VTK transform
            transform = vtk.vtkTransform()
            mat = vtk.vtkMatrix4x4()

            for i in range(3):
                for j in range(3):
                    mat.SetElement(i, j, rotation[i, j])
            mat.SetElement(0, 3, position[0])
            mat.SetElement(1, 3, position[1])
            mat.SetElement(2, 3, position[2])

            transform.SetMatrix(mat)
            actor.SetUserTransform(transform)

        # Also update key points if they are initialized
        if self._key_points_initialized:
            return self.update_key_points(q, root_position)
        return None


class VR3PtPoseVisualizer:
    """
    PyVista-based visualizer for VR 3-point pose debugging.

    Coordinate convention:
    - X: forward (RED axis)
    - Y: left (GREEN axis)
    - Z: up (BLUE axis)

    Quaternion format: [qw, qx, qy, qz] (scalar-first)

    Reference frame:
    - World frame at origin (0, 0, 0) - WHITE ball

    Head kinematic chain visualization:
    - Origin (root) → torso_link (+0.05m along Z)
    - torso_link → head (+0.35m along head's local Z axis)

    G1 Robot visualization (optional):
    - Loads G1 robot meshes using Pinocchio
    - Robot root (pelvis) placed at origin
    - Key points at left wrist, right wrist, and torso
    """

    # Ball color for world reference frame
    WORLD_BALL_COLOR = "white"

    # VR pose ball colors - order: [0]=L-Wrist, [1]=R-Wrist, [2]=Head
    VR_BALL_COLORS = ["lightgreen", "lightblue", "orange"]
    VR_POSE_LABELS = ["L-Wrist", "R-Wrist", "Head"]

    # Axis colors (RGB for XYZ)
    AXIS_COLORS = ["red", "green", "blue"]

    # Head kinematic chain constants (must match pico_manager_thread_server.py)
    TORSO_LINK_OFFSET_Z = 0.05  # meters from root to torso_link
    HEAD_LINK_LENGTH = 0.35  # meters from torso_link to head along head's local Z
    TORSO_LINK_COLOR = "purple"
    HEAD_LINK_COLOR = "orange"

    # SMPL body joint visualization constants
    SMPL_NUM_JOINTS = 24
    SMPL_LOWER_BODY_INDICES = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    SMPL_JOINT_RADIUS = 0.02

    # SMPL kinematic tree parent indices (standard 24-joint model)
    # Joint i connects to SMPL_PARENT_INDICES[i]; root (joint 0) has parent -1
    SMPL_PARENT_INDICES = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,  # 0-11: pelvis, hips, spine, knees, ankles, feet
        9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,  # 12-23: neck, collars, head, shoulders..hands
    ]
    SMPL_BONE_COLOR = "white"
    SMPL_BONE_WIDTH = 2.0
    SMPL_BONE_OPACITY = 0.5

    # Timing report interval (seconds)
    TIMING_REPORT_INTERVAL = 5.0

    def __init__(
        self,
        axis_length: float = 0.1,
        axis_radius: float = 0.005,
        ball_radius: float = 0.02,
        with_g1_robot: bool = False,
        robot_model=None,
        robot_opacity: float = 0.4,
        enable_waist_tracking: bool = False,
        enable_smpl_vis: bool = False,
        smpl_root_position: Optional[np.ndarray] = None,
    ):
        """
        Initialize the VR 3-point pose visualizer.

        Args:
            axis_length: Length of each axis arrow
            axis_radius: Radius of the axis cylinders (for arrows)
            ball_radius: Radius of the position marker balls
            with_g1_robot: If True, load and display G1 robot at origin
            robot_model: Optional pre-instantiated RobotModel for G1 visualization
            robot_opacity: Opacity of G1 robot (0.0 = invisible, 1.0 = opaque, default 0.4)
            enable_waist_tracking: If True, G1 robot waist follows VR head orientation
            enable_smpl_vis: If True, show SMPL body joint spheres (24 joints)
            smpl_root_position: Where to anchor the SMPL skeleton root (joint 0) on
                                the first frame. Default [-0.3, 0.0, 0.0] (0.3m behind
                                the G1 robot at origin, for side-by-side comparison).
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required. Install with: pip install pyvista")

        self.axis_length = axis_length
        self.axis_radius = axis_radius
        self.ball_radius = ball_radius
        self.plotter = None
        self.vr_actors = []  # For real-time mode: list of dicts with 'arrows' and 'ball'
        self._initialized = False

        # SMPL body joint visualization
        self.enable_smpl_vis = enable_smpl_vis
        self.smpl_joint_actors: List = []

        # SMPL root anchoring: on the first frame, capture the root (joint 0) position
        # and offset all subsequent frames so the skeleton starts at smpl_root_position.
        self._smpl_initial_root: Optional[np.ndarray] = None  # captured on first frame
        self.smpl_root_position: np.ndarray = (
            np.array(smpl_root_position, dtype=np.float64)
            if smpl_root_position is not None
            else np.array([-0.3, 0.0, 0.0])
        )

        # Pre-allocated VTK caches (populated in create_realtime_plotter)
        self._vr_arrow_transforms: List = []   # 3×3 vtkTransform for arrows
        self._vr_arrow_matrices: List = []     # 3×3 vtkMatrix4x4 for arrows
        self._vr_ball_transforms: List = []    # 3 vtkTransform for balls
        self._smpl_transforms: List = []       # 24 vtkTransform for joints
        self._smpl_bone_actor = None           # Single actor for all 23 bones
        self._smpl_bone_cells: Optional[np.ndarray] = None  # Line connectivity

        # Pre-allocated numpy arrays for arrow rotation computation
        self._x_axis = np.array([1.0, 0.0, 0.0])
        self._axis_dirs = [
            np.array([1, 0, 0], dtype=np.float64),
            np.array([0, 1, 0], dtype=np.float64),
            np.array([0, 0, 1], dtype=np.float64),
        ]
        self._diag_flip = np.diag([-1.0, 1.0, -1.0])

        # Timing instrumentation (deques of per-frame durations in seconds)
        self._vis_times_vr3pt: deque = deque(maxlen=200)
        self._vis_times_smpl: deque = deque(maxlen=200)
        self._vis_times_render: deque = deque(maxlen=200)
        self._last_timing_report: float = 0.0

        # G1 robot visualization
        self.with_g1_robot = with_g1_robot
        self.robot_opacity = robot_opacity
        self.enable_waist_tracking = enable_waist_tracking
        self.g1_visualizer: Optional[G1RobotVisualizer] = None
        self._robot_q: Optional[np.ndarray] = None  # Current robot joint configuration
        self._last_key_frame_poses: Optional[Dict[str, Dict[str, np.ndarray]]] = None

        if with_g1_robot:
            if not PINOCCHIO_AVAILABLE:
                raise ImportError(
                    "Pinocchio is required for G1 robot visualization. "
                    "Install with: pip install pin"
                )
            self.g1_visualizer = G1RobotVisualizer(robot_model=robot_model)
            self._robot_q = self.g1_visualizer.robot_model.default_body_pose.copy()

    def _add_coordinate_frame(
        self,
        plotter,
        position: np.ndarray,
        quat_wxyz: np.ndarray,
        ball_color: str,
        axis_length: float = None,
        ball_radius: float = None,
        label: str = "",
    ):
        """
        Add a coordinate frame (3 RGB arrows + colored ball) to the plotter.

        Args:
            plotter: PyVista plotter
            position: [x, y, z] position
            quat_wxyz: [qw, qx, qy, qz] quaternion (scalar-first)
            ball_color: Color of the position marker ball
            axis_length: Override axis length (uses default if None)
            ball_radius: Override ball radius (uses default if None)
            label: Optional label for the frame
        """
        if axis_length is None:
            axis_length = self.axis_length
        if ball_radius is None:
            ball_radius = self.ball_radius

        # Convert quaternion to rotation matrix
        # quat_wxyz is [qw, qx, qy, qz], scipy uses [qx, qy, qz, qw]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rot = sRot.from_quat(quat_xyzw)
        rot_matrix = rot.as_matrix()

        # RGB colors for XYZ axes
        axis_dirs = [
            np.array([1, 0, 0]),  # X
            np.array([0, 1, 0]),  # Y
            np.array([0, 0, 1]),  # Z
        ]

        # Add arrows for each axis
        for i, (color, local_dir) in enumerate(zip(self.AXIS_COLORS, axis_dirs)):
            # Rotate local direction by the frame's orientation
            world_dir = rot_matrix @ local_dir

            # Create arrow from position in world_dir direction
            arrow = pv.Arrow(
                start=position,
                direction=world_dir,
                scale=axis_length,
                tip_length=0.3,
                tip_radius=0.15,
                shaft_radius=0.05,
            )
            plotter.add_mesh(arrow, color=color, smooth_shading=True)

        # Add colored ball at position
        ball = pv.Sphere(radius=ball_radius, center=position)
        plotter.add_mesh(ball, color=ball_color, smooth_shading=True)

        # Add label if provided
        if label:
            plotter.add_point_labels(
                [position + np.array([0, 0, ball_radius * 2])],
                [label],
                font_size=12,
                point_color=ball_color,
                text_color="white",
                always_visible=True,
                shape_opacity=0.7,
            )

    def _add_head_kinematic_chain(
        self,
        plotter,
        head_position: np.ndarray,
        origin: np.ndarray = None,
        line_width: float = 3.0,
        torso_ball_radius: float = None,
    ):
        """
        Add visualization of head kinematic chain: origin → torso_link → head.

        Args:
            plotter: PyVista plotter
            head_position: [x, y, z] head position (already computed via kinematic chain)
            origin: [x, y, z] origin/root position (default: [0, 0, 0])
            line_width: Width of the link lines
            torso_ball_radius: Radius of torso_link marker ball (default: half of self.ball_radius)
        """
        if origin is None:
            origin = np.array([0.0, 0.0, 0.0])
        if torso_ball_radius is None:
            torso_ball_radius = self.ball_radius * 0.5

        # Torso link position (fixed offset from origin along Z)
        torso_link_pos = origin + np.array([0.0, 0.0, self.TORSO_LINK_OFFSET_Z])

        # Link 1: Origin → torso_link
        line1 = pv.Line(origin, torso_link_pos)
        plotter.add_mesh(line1, color=self.TORSO_LINK_COLOR, line_width=line_width)

        # Link 2: torso_link → head
        line2 = pv.Line(torso_link_pos, head_position)
        plotter.add_mesh(line2, color=self.HEAD_LINK_COLOR, line_width=line_width)

        # Small ball at torso_link
        torso_ball = pv.Sphere(radius=torso_ball_radius, center=torso_link_pos)
        plotter.add_mesh(torso_ball, color=self.TORSO_LINK_COLOR, smooth_shading=True)

    def _add_reference_frames(self, plotter):
        """Add the world reference frame to the plotter."""

        # World frame at origin (0, 0, 0) - identity rotation, WHITE ball
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [qw, qx, qy, qz]
        self._add_coordinate_frame(
            plotter,
            position=np.array([0.0, 0.0, 0.0]),
            quat_wxyz=identity_quat,
            ball_color=self.WORLD_BALL_COLOR,
            axis_length=self.axis_length * 1.5,  # Larger for world frame
            ball_radius=self.ball_radius * 1.5,
            label="World (origin)",
        )

    def _add_ground_and_grid(self, plotter):
        """Add ground plane and grid for spatial reference."""
        # Ground plane
        ground = pv.Plane(center=(0, 0, -0.005), direction=(0, 0, 1), i_size=1.5, j_size=1.5)
        plotter.add_mesh(ground, color="gray", opacity=0.3)

        # Grid lines
        for i in range(-7, 8):
            val = i * 0.1
            # X-direction lines
            line_x = pv.Line((-0.7, val, 0.001), (0.7, val, 0.001))
            plotter.add_mesh(line_x, color="darkgray", line_width=1)
            # Y-direction lines
            line_y = pv.Line((val, -0.7, 0.001), (val, 0.7, 0.001))
            plotter.add_mesh(line_y, color="darkgray", line_width=1)

    def _add_legend(
        self,
        plotter,
        include_vr_poses: bool = True,
        live: bool = False,
        include_g1: bool = False,
        include_smpl: bool = False,
    ):
        """Add legend text to the plotter."""
        legend_text = (
            "VR 3-Point Pose Debugger\n"
            "─────────────────────────\n"
            "Axes: RED=X  GREEN=Y  BLUE=Z\n"
            "─────────────────────────\n"
            "WHITE: World origin"
        )

        if include_vr_poses:
            live_str = " (live)" if live else ""
            legend_text += (
                "\n─────────────────────────\n"
                f"VR Pose{live_str}:\n"
                "  LIGHTGREEN: L-Wrist\n"
                "  LIGHTBLUE: R-Wrist\n"
                "  ORANGE: Head\n"
                "─────────────────────────\n"
                "Head Kinematic Chain:\n"
                f"  PURPLE: torso_link (+{self.TORSO_LINK_OFFSET_Z}m Z)\n"
                f"  ORANGE line: head link ({self.HEAD_LINK_LENGTH}m)"
            )

        if include_smpl:
            legend_text += (
                "\n─────────────────────────\n"
                "SMPL Body (24 joints):\n"
                "  BLUE gradient: Lower body\n"
                "  RED: Upper body\n"
                "  WHITE lines: Skeleton"
            )

        if include_g1:
            legend_text += (
                "\n─────────────────────────\n"
                "G1 Robot: Root at origin\n"
                "Key Points (with offset):\n"
                "  LIGHTGREEN: L-Wrist\n"
                "  LIGHTBLUE: R-Wrist\n"
                "  YELLOW: Torso"
            )

        plotter.add_text(legend_text, position="upper_left", font_size=9, color="white")

    # =========================================================================
    # SMPL body joint visualization
    # =========================================================================

    @staticmethod
    def _get_smpl_joint_color(joint_idx: int) -> List[float]:
        """Get color for SMPL joint (blue gradient for lower body, red for upper body).

        Lower body joints are colored with a blue gradient from light to dark.
        Upper body joints are red.
        """
        lower_indices = VR3PtPoseVisualizer.SMPL_LOWER_BODY_INDICES
        if joint_idx in lower_indices:
            idx_in_list = lower_indices.index(joint_idx)
            t = idx_in_list / max(1, len(lower_indices) - 1)
            r = 0.6 * (1 - t) + 0.0 * t
            g = 0.8 * (1 - t) + 0.1 * t
            b = 1.0 * (1 - t) + 0.5 * t
            return [r, g, b]
        return [1.0, 0.0, 0.0]  # Red for upper body

    def _create_smpl_joint_actors(self):
        """Pre-create 24 SMPL joint spheres + bone PolyData for real-time updates.

        Optimizations applied:
        - Low-resolution spheres (8×8 instead of default 30×30) — ~14x fewer triangles
        - Pre-allocated vtkTransform per joint — no per-frame Python object creation
        - Single PolyData with 23 line segments for all bones — one draw call
        """
        self.smpl_joint_actors = []
        self._smpl_transforms = []

        for joint_idx in range(self.SMPL_NUM_JOINTS):
            # Low-res sphere: 8×8 is plenty for r=0.02 joints
            sphere = pv.Sphere(
                radius=self.SMPL_JOINT_RADIUS,
                center=(0, 0, 0),
                theta_resolution=8,
                phi_resolution=8,
            )
            color = self._get_smpl_joint_color(joint_idx)
            actor = self.plotter.add_mesh(
                sphere, color=color, smooth_shading=True, opacity=0.8
            )

            # Pre-allocate VTK transform and bind to actor once
            t = vtk.vtkTransform()
            actor.SetUserTransform(t)

            self.smpl_joint_actors.append(actor)
            self._smpl_transforms.append(t)

        # Build bone connectivity (23 line segments: each child → parent)
        cells = []
        for child_idx in range(1, self.SMPL_NUM_JOINTS):
            parent_idx = self.SMPL_PARENT_INDICES[child_idx]
            cells.extend([2, parent_idx, child_idx])
        self._smpl_bone_cells = np.array(cells, dtype=np.int64)

        # Create initial bone PolyData and add as single actor
        bone_points = np.zeros((self.SMPL_NUM_JOINTS, 3), dtype=np.float64)
        bone_poly = pv.PolyData(bone_points)
        bone_poly.lines = self._smpl_bone_cells
        self._smpl_bone_actor = self.plotter.add_mesh(
            bone_poly,
            color=self.SMPL_BONE_COLOR,
            line_width=self.SMPL_BONE_WIDTH,
            opacity=self.SMPL_BONE_OPACITY,
        )

    def update_smpl_joints(self, joints_np: np.ndarray):
        """Update SMPL joint sphere positions and bone lines for real-time visualization.

        On the first call, captures the root (joint 0) position and anchors
        all subsequent frames so the skeleton is placed at ``smpl_root_position``.

        Uses pre-allocated vtkTransform objects (no per-frame allocation) and
        updates the bone PolyData via mapper input swap.

        Args:
            joints_np: Shape (24, 3) array of joint positions in local (root-relative) frame.
        """
        if not self.enable_smpl_vis or len(self.smpl_joint_actors) == 0:
            return

        t0 = time.perf_counter()
        n = min(len(self._smpl_transforms), len(joints_np))

        # First-frame anchoring: capture the initial root and compute the offset
        if self._smpl_initial_root is None:
            self._smpl_initial_root = joints_np[0].copy()
            print(
                f"[VR3PtVis] SMPL root anchored: initial_root="
                f"[{self._smpl_initial_root[0]:.4f}, {self._smpl_initial_root[1]:.4f}, "
                f"{self._smpl_initial_root[2]:.4f}] → anchor="
                f"[{self.smpl_root_position[0]:.4f}, {self.smpl_root_position[1]:.4f}, "
                f"{self.smpl_root_position[2]:.4f}]"
            )

        # Shift all joints: subtract initial root, add desired anchor position
        offset = self.smpl_root_position - self._smpl_initial_root
        joints_shifted = joints_np[:n] + offset

        # Update joint sphere transforms (reuse pre-allocated vtkTransform)
        for i in range(n):
            t = self._smpl_transforms[i]
            t.Identity()
            t.Translate(
                float(joints_shifted[i, 0]),
                float(joints_shifted[i, 1]),
                float(joints_shifted[i, 2]),
            )

        # Update bone PolyData (single draw call for all 23 bones)
        if self._smpl_bone_actor is not None and self._smpl_bone_cells is not None:
            bone_poly = pv.PolyData(joints_shifted.astype(np.float64))
            bone_poly.lines = self._smpl_bone_cells
            self._smpl_bone_actor.GetMapper().SetInputData(bone_poly)

        self._vis_times_smpl.append(time.perf_counter() - t0)

    def reset_smpl_anchor(self):
        """Reset the SMPL root anchor so it is re-captured on the next frame.

        Call this after recalibration or when the operator's pose has changed
        significantly and the skeleton should be re-anchored.
        """
        self._smpl_initial_root = None
        print("[VR3PtVis] SMPL root anchor reset — will re-capture on next frame")

    # =========================================================================
    # Timing instrumentation
    # =========================================================================

    def _maybe_report_timing(self):
        """Periodically log average timing breakdown for vis_vr3pt vs vis_both."""
        now = time.time()
        if now - self._last_timing_report < self.TIMING_REPORT_INTERVAL:
            return
        self._last_timing_report = now

        def _avg_ms(dq: deque) -> float:
            return (sum(dq) / len(dq) * 1000.0) if dq else 0.0

        vr3pt_ms = _avg_ms(self._vis_times_vr3pt)
        smpl_ms = _avg_ms(self._vis_times_smpl)
        render_ms = _avg_ms(self._vis_times_render)
        vr3pt_only_ms = vr3pt_ms + render_ms
        both_ms = vr3pt_ms + smpl_ms + render_ms

        parts = [f"vr3pt: {vr3pt_ms:.2f}ms"]
        if self.enable_smpl_vis:
            parts.append(f"smpl: {smpl_ms:.2f}ms")
        parts.append(f"render: {render_ms:.2f}ms")
        parts.append(f"vr3pt_only: {vr3pt_only_ms:.2f}ms")
        if self.enable_smpl_vis:
            parts.append(f"both(vr3pt+smpl): {both_ms:.2f}ms")

        print(f"[Vis Timing] {' | '.join(parts)}")

    def show_static(self, robot_q: np.ndarray = None):
        """
        Show static visualization with reference frames only (blocking).

        Args:
            robot_q: Optional joint configuration for G1 robot (if with_g1_robot=True).
                    If None, uses default body pose.
        """
        pv.set_plot_theme("dark")
        plotter = pv.Plotter(window_size=(1400, 900))
        plotter.set_background("black")

        # Add ground and grid
        self._add_ground_and_grid(plotter)

        # Add reference frames
        self._add_reference_frames(plotter)

        # Add G1 robot if enabled
        if self.with_g1_robot and self.g1_visualizer is not None:
            q = robot_q if robot_q is not None else self._robot_q
            self.g1_visualizer.add_to_plotter(
                plotter, q=q, root_position=np.array([0.0, 0.0, 0.0]), opacity=self.robot_opacity
            )
            # Add key point markers and get their poses
            self._last_key_frame_poses = self.g1_visualizer.add_key_points_to_plotter(
                plotter, q=q, root_position=np.array([0.0, 0.0, 0.0])
            )

        # Set camera — zoomed out for global view
        plotter.camera_position = [(1.5, -1.2, 1.2), (0.0, 0.0, 0.2), (0, 0, 1)]

        # Add legend
        self._add_legend(plotter, include_vr_poses=False, include_g1=self.with_g1_robot)

        print("\n[VR3PtPoseVisualizer] Reference frame displayed:")
        print("  - WHITE ball at (0, 0, 0): World frame, identity rotation")
        if self.with_g1_robot:
            print("  - G1 Robot: Displayed at origin")
            if self._last_key_frame_poses:
                print("\n  Key Frame Poses (position + orientation_xyzw):")
                for key, pose in self._last_key_frame_poses.items():
                    pos = pose["position"]
                    quat = pose["orientation_xyzw"]
                    print(
                        f"    {key}: pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], "
                        f"quat_xyzw=[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"
                    )
        print("\nClose the window to exit.")

        plotter.show()

    def show_with_vr_pose(self, vr_3pt_pose: np.ndarray, robot_q: np.ndarray = None):
        """
        Show visualization with both reference frames and actual VR pose data.

        Args:
            vr_3pt_pose: Shape (3, 7) array where each row is [x, y, z, qw, qx, qy, qz]
                        Row 0: L-Wrist, Row 1: R-Wrist, Row 2: Head
            robot_q: Optional joint configuration for G1 robot (if with_g1_robot=True).
                    If None, uses default body pose.
        """
        pv.set_plot_theme("dark")
        plotter = pv.Plotter(window_size=(1400, 900))
        plotter.set_background("black")

        # Add ground and grid
        self._add_ground_and_grid(plotter)

        # Add reference frames
        self._add_reference_frames(plotter)

        # Add G1 robot if enabled
        if self.with_g1_robot and self.g1_visualizer is not None:
            q = robot_q if robot_q is not None else self._robot_q
            self.g1_visualizer.add_to_plotter(
                plotter, q=q, root_position=np.array([0.0, 0.0, 0.0]), opacity=self.robot_opacity
            )
            # Add key point markers and get their poses
            self._last_key_frame_poses = self.g1_visualizer.add_key_points_to_plotter(
                plotter, q=q, root_position=np.array([0.0, 0.0, 0.0])
            )

        # Add VR pose frames
        for i in range(vr_3pt_pose.shape[0]):
            position = vr_3pt_pose[i, :3]
            quat_wxyz = vr_3pt_pose[i, 3:7]

            self._add_coordinate_frame(
                plotter,
                position=position,
                quat_wxyz=quat_wxyz,
                ball_color=self.VR_BALL_COLORS[i],
                label=f"VR {self.VR_POSE_LABELS[i]}",
            )

        # Add head kinematic chain visualization (origin → torso_link → head)
        # Head is at index 2 in vr_3pt_pose
        head_position = vr_3pt_pose[2, :3]
        self._add_head_kinematic_chain(plotter, head_position)

        # Set camera — zoomed out for global view
        plotter.camera_position = [(1.5, -1.2, 1.2), (0.0, 0.0, 0.2), (0, 0, 1)]

        # Add legend
        self._add_legend(plotter, include_vr_poses=True, include_g1=self.with_g1_robot)

        plotter.show()

    def create_realtime_plotter(
        self,
        interactive: bool = True,
        window_size: tuple = (1400, 900),
        with_reference_frames: bool = True,
        robot_q: np.ndarray = None,
    ):
        """
        Create a plotter for real-time visualization with pre-created actors.

        This method initializes a plotter that can be updated efficiently without
        recreating actors each frame.

        Args:
            interactive: If True, enables interactive updates (non-blocking)
            window_size: Window size as (width, height)
            with_reference_frames: Whether to add static reference frames
            robot_q: Optional initial joint configuration for G1 robot.

        Returns:
            The PyVista plotter object
        """
        if not VTK_AVAILABLE:
            raise ImportError("VTK is required for real-time mode. Install with: pip install vtk")

        pv.set_plot_theme("dark")
        self.plotter = pv.Plotter(window_size=window_size)
        self.plotter.set_background("black")

        # Add ground and grid (static)
        self._add_ground_and_grid(self.plotter)

        # Add reference frames (static)
        if with_reference_frames:
            self._add_reference_frames(self.plotter)

        # Add G1 robot if enabled (for real-time updates)
        if self.with_g1_robot and self.g1_visualizer is not None:
            q = robot_q if robot_q is not None else self._robot_q
            self.g1_visualizer.add_to_plotter_realtime(
                self.plotter,
                q=q,
                root_position=np.array([0.0, 0.0, 0.0]),
                opacity=self.robot_opacity,
            )
            # Add key point markers for real-time updates
            self._last_key_frame_poses = self.g1_visualizer.add_key_points_realtime(
                self.plotter, q=q, root_position=np.array([0.0, 0.0, 0.0])
            )

        # Set camera — zoomed out for global view of SMPL + G1/VR3pt
        self.plotter.camera_position = [(1.5, -1.2, 1.2), (0.0, 0.0, 0.2), (0, 0, 1)]

        # Add legend
        self._add_legend(
            self.plotter,
            include_vr_poses=True,
            live=True,
            include_g1=self.with_g1_robot,
            include_smpl=self.enable_smpl_vis,
        )

        # Pre-create VR pose actors with pre-allocated VTK transforms
        # (3 poses × (3 arrows + 1 ball) = 12 actors, 9 transforms + 9 matrices + 3 ball transforms)
        self.vr_actors = []
        self._vr_arrow_transforms = []
        self._vr_arrow_matrices = []
        self._vr_ball_transforms = []

        for i in range(3):
            pose_actors = {"arrows": [], "ball": None}
            arrow_transforms_i = []
            arrow_matrices_i = []

            # Create low-overhead arrows for each axis
            for j, color in enumerate(self.AXIS_COLORS):
                arrow = pv.Arrow(
                    start=(0, 0, 0),
                    direction=(1, 0, 0),
                    scale=0.08,
                    tip_length=0.3,
                    tip_radius=0.15,
                    shaft_radius=0.05,
                    tip_resolution=6,
                    shaft_resolution=6,
                )
                actor = self.plotter.add_mesh(arrow, color=color, smooth_shading=True)

                # Pre-allocate transform + matrix and bind once
                t = vtk.vtkTransform()
                m = vtk.vtkMatrix4x4()
                actor.SetUserTransform(t)

                pose_actors["arrows"].append(actor)
                arrow_transforms_i.append(t)
                arrow_matrices_i.append(m)

            # Low-res ball
            ball = pv.Sphere(
                radius=0.015, center=(0, 0, 0), theta_resolution=8, phi_resolution=8
            )
            ball_actor = self.plotter.add_mesh(
                ball, color=self.VR_BALL_COLORS[i], smooth_shading=True
            )
            bt = vtk.vtkTransform()
            ball_actor.SetUserTransform(bt)
            pose_actors["ball"] = ball_actor

            self.vr_actors.append(pose_actors)
            self._vr_arrow_transforms.append(arrow_transforms_i)
            self._vr_arrow_matrices.append(arrow_matrices_i)
            self._vr_ball_transforms.append(bt)

        # Pre-create SMPL body joint spheres + bone PolyData (if enabled)
        if self.enable_smpl_vis:
            self._create_smpl_joint_actors()

        # Pre-create head kinematic chain actors (origin → torso_link → head)
        # These will be updated dynamically as head position changes
        origin = np.array([0.0, 0.0, 0.0])
        torso_link_pos = np.array([0.0, 0.0, self.TORSO_LINK_OFFSET_Z])
        initial_head_pos = np.array([0.0, 0.0, self.TORSO_LINK_OFFSET_Z + self.HEAD_LINK_LENGTH])

        # Link 1: origin → torso_link (static, doesn't change)
        line1 = pv.Line(origin, torso_link_pos)
        self.plotter.add_mesh(line1, color=self.TORSO_LINK_COLOR, line_width=3.0)

        # Torso link ball (static)
        torso_ball = pv.Sphere(radius=self.ball_radius * 0.5, center=torso_link_pos)
        self.plotter.add_mesh(torso_ball, color=self.TORSO_LINK_COLOR, smooth_shading=True)

        # Link 2: torso_link → head (dynamic, needs updating)
        line2 = pv.Line(torso_link_pos, initial_head_pos)
        self.head_link_actor = self.plotter.add_mesh(
            line2, color=self.HEAD_LINK_COLOR, line_width=3.0
        )
        # Store torso_link position for updating head link
        self._torso_link_pos = torso_link_pos

        self._initialized = True

        if interactive:
            self.plotter.show(interactive_update=True)

        return self.plotter

    def update_vr_poses(self, vr_3pt_pose: np.ndarray):
        """
        Update VR pose actors with new pose data (for real-time mode).

        Uses pre-allocated vtkTransform and vtkMatrix4x4 objects — no per-frame
        Python object creation. Transforms are bound to actors once at init time;
        mutating them in-place triggers VTK re-render via MTime.

        Args:
            vr_3pt_pose: Shape (3, 7) array where each row is [x, y, z, qw, qx, qy, qz]
                        Row 0: L-Wrist, Row 1: R-Wrist, Row 2: Head
        """
        if not self._initialized or len(self.vr_actors) != 3:
            return

        for i in range(min(vr_3pt_pose.shape[0], 3)):
            position = vr_3pt_pose[i, :3]
            quat_wxyz = vr_3pt_pose[i, 3:7]

            # Convert quaternion to rotation matrix
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            rot_matrix = sRot.from_quat(quat_xyzw).as_matrix()

            # Update each arrow's transform (reuse pre-allocated objects)
            for j, local_dir in enumerate(self._axis_dirs):
                world_dir = rot_matrix @ local_dir

                # Rodrigues' rotation: align X-axis arrow to world_dir
                v = np.cross(self._x_axis, world_dir)
                c = float(np.dot(self._x_axis, world_dir))
                v_norm = float(np.linalg.norm(v))

                if v_norm > 1e-6:
                    s = v_norm
                    vx = np.array(
                        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
                    )
                    arrow_rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-9))
                elif c < 0:
                    arrow_rot = self._diag_flip
                else:
                    arrow_rot = np.eye(3)

                # Reuse pre-allocated matrix and transform
                mat = self._vr_arrow_matrices[i][j]
                mat.Identity()
                for ri in range(3):
                    for ci in range(3):
                        mat.SetElement(ri, ci, arrow_rot[ri, ci])
                mat.SetElement(0, 3, float(position[0]))
                mat.SetElement(1, 3, float(position[1]))
                mat.SetElement(2, 3, float(position[2]))
                self._vr_arrow_transforms[i][j].SetMatrix(mat)

            # Reuse pre-allocated ball transform
            bt = self._vr_ball_transforms[i]
            bt.Identity()
            bt.Translate(float(position[0]), float(position[1]), float(position[2]))

        # Update head kinematic chain link (torso_link → head)
        if hasattr(self, "head_link_actor") and self.head_link_actor is not None:
            head_position = vr_3pt_pose[2, :3]
            new_line = pv.Line(self._torso_link_pos, head_position)
            self.head_link_actor.GetMapper().SetInputData(new_line)

    def update_robot_pose(self, robot_q: np.ndarray) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """
        Update G1 robot pose for real-time visualization.

        Args:
            robot_q: Joint configuration for the robot

        Returns:
            Dict with updated key frame poses (left_wrist, right_wrist, torso),
            each containing 'position' and 'orientation_xyzw'. Returns None if
            G1 robot is not enabled.
        """
        if self.with_g1_robot and self.g1_visualizer is not None:
            self._robot_q = robot_q.copy()
            self._last_key_frame_poses = self.g1_visualizer.update_pose(robot_q)
            return self._last_key_frame_poses
        return None

    def update_from_vr_pose(
        self,
        vr_3pt_pose: np.ndarray,
        waist_scale: float = 1.0,
    ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """
        Update both VR pose visualization and optionally G1 robot waist from VR pose data.

        This method:
        1. Updates the VR pose markers (L-Wrist, R-Wrist, Neck)
        2. If enable_waist_tracking is True: computes waist joint angles from VR neck
           orientation and updates the G1 robot visualization

        Timing: The entire method is tracked as "vr3pt" time for the delay comparison
        between vis_vr3pt vs vis_both(vr3pt+smpl).

        Args:
            vr_3pt_pose: Shape (3, 7) array where each row is [x, y, z, qw, qx, qy, qz]
                        Row 0: L-Wrist, Row 1: R-Wrist, Row 2: Neck
            waist_scale: Scale factor for waist joint angles (0.0-1.0)

        Returns:
            Dict with updated key frame poses, or None if G1 robot is not enabled
            or waist tracking is disabled
        """
        t0 = time.perf_counter()

        # Update VR pose markers
        self.update_vr_poses(vr_3pt_pose)

        result = None

        # Update G1 robot waist from VR neck orientation (only if waist tracking enabled)
        if self.enable_waist_tracking and self.with_g1_robot and self.g1_visualizer is not None:
            # Extract neck orientation (row 2, columns 3-7 are qw,qx,qy,qz)
            neck_quat_wxyz = vr_3pt_pose[2, 3:]

            # Compute waist joints from neck orientation
            waist_joints = self.g1_visualizer.compute_waist_joints_from_orientation(
                neck_quat_wxyz, scale_factor=waist_scale
            )

            if waist_joints is not None and self._robot_q is not None:
                # Apply waist joints to current robot configuration
                new_q = self.g1_visualizer.apply_waist_joints_to_config(self._robot_q, waist_joints)
                # Update robot visualization
                self._robot_q = new_q
                self._last_key_frame_poses = self.g1_visualizer.update_pose(new_q)
                result = self._last_key_frame_poses

        self._vis_times_vr3pt.append(time.perf_counter() - t0)
        return result

    @property
    def last_key_frame_poses(self) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """Get the last computed key frame poses."""
        return self._last_key_frame_poses

    @property
    def robot_model(self):
        """Get the robot model (if G1 visualization is enabled)."""
        if self.g1_visualizer is not None:
            return self.g1_visualizer.robot_model
        return None

    def render(self):
        """Render the current frame (for real-time mode). Tracks render time and reports timing."""
        if self.plotter is not None:
            t0 = time.perf_counter()
            self.plotter.update()
            self._vis_times_render.append(time.perf_counter() - t0)
            self._maybe_report_timing()

    def close(self):
        """Close the plotter window."""
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None
            self._initialized = False

    @property
    def is_open(self) -> bool:
        """Check if the plotter window is still open."""
        if self.plotter is None:
            return False
        try:
            # Check if plotter is still active
            return self.plotter.ren_win is not None and not self.plotter._closed
        except (AttributeError, RuntimeError):
            return False


def run_vr3pt_visualizer_test():
    """
    Standalone test for VR 3-point pose visualizer using PyVista.
    Run this to verify the reference frames are displayed correctly.
    """
    print("=" * 60)
    print("VR 3-Point Pose Visualizer Test (PyVista)")
    print("=" * 60)
    print("\nWorld reference frame (RGB axes for XYZ):")
    print("  WHITE ball at origin (0, 0, 0) - World frame")
    print("\nClose the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015)
    visualizer.show_static()


def run_vr3pt_demo_with_fake_data():
    """
    Demo visualization with fake VR pose data for testing without hardware.
    """
    print("=" * 60)
    print("VR 3-Point Pose Demo (Fake Data)")
    print("=" * 60)

    # Create fake VR 3-point pose data
    # Format: [x, y, z, qw, qx, qy, qz] for each of [L-Wrist, R-Wrist, Head]
    fake_pose = np.array(
        [
            [0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0],  # L-Wrist at (0.2, 0.3, 0.4), identity
            [0.2, -0.3, 0.4, 1.0, 0.0, 0.0, 0.0],  # R-Wrist at (0.2, -0.3, 0.4), identity
            [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],  # Head at (0, 0, 0.5), identity
        ],
        dtype=np.float32,
    )

    print(f"\nFake pose data shape: {fake_pose.shape}")
    print(f"  L-Wrist: pos={fake_pose[0, :3]}, quat_wxyz={fake_pose[0, 3:]}")
    print(f"  R-Wrist: pos={fake_pose[1, :3]}, quat_wxyz={fake_pose[1, 3:]}")
    print(f"  Head:    pos={fake_pose[2, :3]}, quat_wxyz={fake_pose[2, 3:]}")
    print("\nClose the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015)
    visualizer.show_with_vr_pose(fake_pose)


def run_realtime_demo_with_fake_data(duration: float = 10.0, update_hz: int = 30):
    """
    Real-time demo with animated fake VR pose data.

    Args:
        duration: How long to run the demo in seconds
        update_hz: Update rate in Hz
    """
    import time

    print("=" * 60)
    print("VR 3-Point Pose Real-time Demo (Animated Fake Data)")
    print("=" * 60)
    print(f"Duration: {duration}s, Update rate: {update_hz} Hz")
    print("Close the window to exit early.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015)
    visualizer.create_realtime_plotter()

    start_time = time.time()

    while time.time() - start_time < duration:
        if not visualizer.is_open:
            break

        t = time.time() - start_time

        # Animate the fake pose
        fake_pose = np.array(
            [
                # L-Wrist: circular motion
                [
                    0.2 + 0.1 * np.sin(t * 2),
                    0.3 + 0.1 * np.cos(t * 2),
                    0.4,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                # R-Wrist: circular motion (opposite phase)
                [
                    0.2 + 0.1 * np.sin(t * 2 + np.pi),
                    -0.3 + 0.1 * np.cos(t * 2 + np.pi),
                    0.4,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                # Head: slight bobbing
                [0.0, 0.0, 0.5 + 0.05 * np.sin(t * 3), 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        visualizer.update_vr_poses(fake_pose)
        visualizer.render()

        time.sleep(1.0 / update_hz)

    visualizer.close()
    print("Demo finished.")


def run_g1_robot_demo():
    """
    Demo visualization showing G1 robot at origin with default pose.
    """
    print("=" * 60)
    print("G1 Robot Visualization Demo")
    print("=" * 60)
    print("Loading G1 robot model...")

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)

    print(f"Robot DOFs: {visualizer.robot_model.num_dofs}")

    # Get and print key frame poses
    key_poses = get_g1_key_frame_poses(visualizer.robot_model)
    if key_poses:
        print("\nKey Frame Poses (at default body pose):")
        print("-" * 50)
        for key, pose in key_poses.items():
            pos = pose["position"]
            quat_xyzw = pose["orientation_xyzw"]
            print(f"  {key}:")
            print(f"    position:         [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            print(
                f"    orientation_xyzw: [{quat_xyzw[0]:.4f}, {quat_xyzw[1]:.4f}, "
                f"{quat_xyzw[2]:.4f}, {quat_xyzw[3]:.4f}]"
            )
        print("-" * 50)

    print("\nG1 robot displayed at origin with key point markers.")
    print("Close the window to exit.")
    print("=" * 60)

    visualizer.show_static()


def run_g1_robot_with_vr_demo():
    """
    Demo visualization showing G1 robot with fake VR pose data.
    """
    print("=" * 60)
    print("G1 Robot + VR Pose Demo")
    print("=" * 60)
    print("Loading G1 robot model...")

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)

    # Create fake VR 3-point pose data
    fake_pose = np.array(
        [
            [0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0],  # L-Wrist
            [0.2, -0.3, 0.4, 1.0, 0.0, 0.0, 0.0],  # R-Wrist
            [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],  # Head
        ],
        dtype=np.float32,
    )

    print(f"Robot DOFs: {visualizer.robot_model.num_dofs}")
    print("G1 robot displayed at origin with VR pose overlay.")
    print("Close the window to exit.")
    print("=" * 60)

    visualizer.show_with_vr_pose(fake_pose)


def run_g1_realtime_demo(duration: float = 10.0, update_hz: int = 30):
    """
    Real-time demo with G1 robot and animated joint movements.

    Args:
        duration: How long to run the demo in seconds
        update_hz: Update rate in Hz
    """
    import time

    print("=" * 60)
    print("G1 Robot Real-time Demo (Animated Joints)")
    print("=" * 60)
    print("Loading G1 robot model...")

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)

    print(f"Robot DOFs: {visualizer.robot_model.num_dofs}")
    print(f"Duration: {duration}s, Update rate: {update_hz} Hz")
    print("Key frame poses will be printed every second.")
    print("Close the window to exit early.")
    print("=" * 60)

    visualizer.create_realtime_plotter()

    start_time = time.time()
    last_print_time = 0
    base_q = visualizer.robot_model.default_body_pose.copy()

    # Get joint indices for arm joints
    try:
        left_arm_indices = visualizer.robot_model.get_joint_group_indices("left_arm")
        right_arm_indices = visualizer.robot_model.get_joint_group_indices("right_arm")
    except (ValueError, AttributeError) as e:
        raise RuntimeError(
            f"Could not get arm joint indices from robot model for the demo. "
            f"Ensure the robot model defines 'left_arm' and 'right_arm' joint groups. "
            f"Original error: {e}"
        ) from e

    while time.time() - start_time < duration:
        if not visualizer.is_open:
            break

        t = time.time() - start_time

        # Animate joint positions
        q = base_q.copy()

        # Animate arm joints if available
        if left_arm_indices:
            for i, idx in enumerate(left_arm_indices[:3]):  # First 3 joints
                q[idx] = base_q[idx] + 0.3 * np.sin(t * 2 + i * 0.5)

        if right_arm_indices:
            for i, idx in enumerate(right_arm_indices[:3]):  # First 3 joints
                q[idx] = base_q[idx] + 0.3 * np.sin(t * 2 + i * 0.5 + np.pi)

        # Update robot pose and get key frame poses
        key_poses = visualizer.update_robot_pose(q)

        # Print key frame poses every second
        if t - last_print_time >= 1.0 and key_poses:
            last_print_time = t
            print(f"\n[t={t:.1f}s] Key Frame Poses:")
            for key, pose in key_poses.items():
                pos = pose["position"]
                quat = pose["orientation_xyzw"]
                print(
                    f"  {key}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
                    f"quat_xyzw=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]"
                )

        # Also animate VR poses
        fake_pose = np.array(
            [
                [0.2 + 0.1 * np.sin(t * 2), 0.3 + 0.1 * np.cos(t * 2), 0.4, 1.0, 0.0, 0.0, 0.0],
                [
                    0.2 + 0.1 * np.sin(t * 2 + np.pi),
                    -0.3 + 0.1 * np.cos(t * 2 + np.pi),
                    0.4,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.5 + 0.05 * np.sin(t * 3), 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        visualizer.update_vr_poses(fake_pose)

        visualizer.render()
        time.sleep(1.0 / update_hz)

    visualizer.close()
    print("\nDemo finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VR 3-Point Pose Visualizer with G1 Robot")
    parser.add_argument(
        "--mode",
        choices=["static", "demo", "realtime", "g1", "g1_vr", "g1_realtime"],
        default="g1_vr",
        help=(
            "Visualization mode: "
            "static (reference frames only), "
            "demo (fake VR pose), "
            "realtime (animated VR demo), "
            "g1 (G1 robot at origin), "
            "g1_vr (G1 + VR pose), "
            "g1_realtime (G1 with animated joints)"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration for realtime demo in seconds",
    )
    parser.add_argument("--hz", type=int, default=30, help="Update rate for realtime demo")

    args = parser.parse_args()

    if args.mode == "static":
        run_vr3pt_visualizer_test()
    elif args.mode == "demo":
        run_vr3pt_demo_with_fake_data()
    elif args.mode == "realtime":
        run_realtime_demo_with_fake_data(duration=args.duration, update_hz=args.hz)
    elif args.mode == "g1":
        run_g1_robot_demo()
    elif args.mode == "g1_vr":
        run_g1_robot_with_vr_demo()
    elif args.mode == "g1_realtime":
        run_g1_realtime_demo(duration=args.duration, update_hz=args.hz)
