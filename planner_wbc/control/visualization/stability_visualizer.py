import time

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from planner_wbc.control.teleop.planner_constraints import G1Constraints


class G1StabilityVisualizer:
    """Draw Pinocchio feet, support polygon, and COM projection in MuJoCo."""

    def __init__(self, robot_model, sim_env, update_frequency=10.0):
        self.constraints = G1Constraints(robot_model)
        self.sim_env = sim_env
        self.update_period = 1.0 / update_frequency
        self.last_update_time = 0.0
        self.last_log_time = 0.0
        self.was_enabled = False

    @property
    def viewer(self):
        return getattr(self.sim_env, "viewer", None)

    def clear(self):
        viewer = self.viewer
        if viewer is None:
            return
        with viewer.lock():
            viewer.user_scn.ngeom = 0

    def update(self, q, floating_base_pose, enabled):
        if not enabled:
            if self.was_enabled:
                self.clear()
                print("Stability visualization disabled")
            self.was_enabled = False
            return

        if not self.was_enabled:
            print("Stability visualization enabled")
            self.was_enabled = True

        now = time.monotonic()
        if now - self.last_update_time < self.update_period:
            return
        self.last_update_time = now

        contact_points = self.constraints.compute_foot_contact_points(q)
        left_foot = self.constraints.robot_model.frame_placement(
            self.constraints.LEFT_FOOT_FRAME
        ).translation.copy()
        right_foot = self.constraints.robot_model.frame_placement(
            self.constraints.RIGHT_FOOT_FRAME
        ).translation.copy()
        feet = np.vstack((left_foot, right_foot))
        com = self.constraints.compute_center_of_mass(q)

        contact_points = self._to_world(contact_points, floating_base_pose)
        feet = self._to_world(feet, floating_base_pose)
        com = self._to_world(com[None, :], floating_base_pose)[0]

        hull = ConvexHull(contact_points[:, :2])
        polygon = contact_points[hull.vertices].copy()
        polygon[:, 2] = np.mean(contact_points[:, 2])
        projection = com.copy()
        projection[2] = np.mean(contact_points[:, 2])

        normals = hull.equations[:, :2]
        offsets = hull.equations[:, 2]
        margins = -(normals @ com[:2] + offsets) / np.linalg.norm(normals, axis=1)
        stability_margin = float(np.min(margins))

        self._draw(
            feet=feet,
            contact_points=contact_points,
            support_polygon=polygon,
            com=com,
            projection=projection,
            stable=stability_margin >= 0.05,
        )

        if now - self.last_log_time >= 1.0:
            print(
                f"Stability margin: {stability_margin:.4f} m | "
                f"stable: {stability_margin >= 0.05}"
            )
            self.last_log_time = now

    @staticmethod
    def _to_world(points, floating_base_pose):
        base_position = np.asarray(floating_base_pose[:3], dtype=np.float64)
        base_quaternion = np.asarray(floating_base_pose[3:7], dtype=np.float64)
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, base_quaternion)
        rotation = rotation.reshape(3, 3)
        return np.asarray(points, dtype=np.float64) @ rotation.T + base_position

    def _draw(self, feet, contact_points, support_polygon, com, projection, stable):
        viewer = self.viewer
        if viewer is None:
            return

        with viewer.lock():
            scene = viewer.user_scn
            scene.ngeom = 0

            for point in contact_points:
                self._add_sphere(scene, point, 0.012, [1.0, 0.65, 0.0, 1.0])

            for point in feet:
                self._add_sphere(scene, point, 0.018, [0.1, 0.4, 1.0, 1.0])

            for index, start in enumerate(support_polygon):
                end = support_polygon[(index + 1) % len(support_polygon)]
                self._add_line(scene, start, end, 4.0, [0.0, 0.9, 0.9, 1.0])

            projection_color = [0.0, 1.0, 0.0, 1.0] if stable else [1.0, 0.0, 0.0, 1.0]
            self._add_sphere(scene, projection, 0.025, projection_color)
            self._add_sphere(scene, com, 0.02, [0.8, 0.1, 1.0, 1.0])
            self._add_line(scene, com, projection, 3.0, projection_color)

    @staticmethod
    def _next_geom(scene):
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        return geom

    @classmethod
    def _add_sphere(cls, scene, position, radius, color):
        geom = cls._next_geom(scene)
        if geom is None:
            return
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius], dtype=np.float64),
            np.asarray(position, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(color, dtype=np.float32),
        )

    @classmethod
    def _add_line(cls, scene, start, end, width, color):
        geom = cls._next_geom(scene)
        if geom is None:
            return
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        geom.rgba[:] = np.asarray(color, dtype=np.float32)
