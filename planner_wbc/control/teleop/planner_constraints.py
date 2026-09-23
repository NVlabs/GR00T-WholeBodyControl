import numpy as np
import pinocchio as pin

from scipy.spatial import ConvexHull

class G1Constraints:
    LEFT_FOOT_FRAME = "left_ankle_roll_link"
    RIGHT_FOOT_FRAME = "right_ankle_roll_link"

    def __init__(self, robot_model):
        self.robot_model = robot_model

    def compute_feet_poses(self, q):
        """Return both foot poses in the Pinocchio world frame."""
        self.robot_model.cache_forward_kinematics(q, auto_clip=False)

        left_foot_pose = self.robot_model.frame_placement(
            self.LEFT_FOOT_FRAME
        )
        right_foot_pose = self.robot_model.frame_placement(
            self.RIGHT_FOOT_FRAME
        )

        return left_foot_pose, right_foot_pose

    def compute_feet_positions(self, q):
        """Return both foot positions in the Pinocchio world frame."""
        left_pose, right_pose = self.compute_feet_poses(q)

        return (
            left_pose.translation.copy(),
            right_pose.translation.copy(),
        )

    def compute_center_of_mass(self, q):
        """Return the center of mass in the Pinocchio world frame."""
        model = self.robot_model.pinocchio_wrapper.model
        data = self.robot_model.pinocchio_wrapper.data

        return pin.centerOfMass(model, data, q).copy()
    

    def compute_foot_contact_points(self, q):
        """Return collision-derived foot contact points in world coordinates."""
        self.compute_feet_poses(q)
        wrapper = self.robot_model.pinocchio_wrapper
        model = wrapper.model
        collision_model = wrapper.collision_model
        collision_data = wrapper.collision_data
        pin.updateGeometryPlacements(
            model, wrapper.data, collision_model, collision_data
        )

        def contact_points(frame_name):
            frame_id = model.getFrameId(frame_name)
            return np.array(
                [
                    collision_data.oMg[index].translation.copy()
                    for index, geometry in enumerate(collision_model.geometryObjects)
                    if geometry.parentFrame == frame_id
                ]
            )

        left_points = contact_points(self.LEFT_FOOT_FRAME)
        right_points = contact_points(self.RIGHT_FOOT_FRAME)

        return np.vstack((left_points, right_points))

    def compute_support_polygon(self, q):
        """Return the support polygon vertices in the XY plane."""
        contact_points_xy = self.compute_foot_contact_points(q)[:, :2]
        hull = ConvexHull(contact_points_xy)
        return contact_points_xy[hull.vertices]
    
    def com_projection(self, q):
        """Project the center of mass onto the ground plane."""
        com = self.compute_center_of_mass(q)
        return np.array([com[0], com[1], 0.0])
    
    def compute_stability(self, q):
        """Return the signed COM distance to the nearest support-polygon edge.

        The result is positive inside the polygon, zero on its boundary, and
        negative outside.
        """
        contact_points_xy = self.compute_foot_contact_points(q)[:, :2]

        hull = ConvexHull(contact_points_xy)

        com_xy = self.compute_center_of_mass(q)[:2]

        margins = []

        # Hull interiors satisfy a*x + b*y + c <= 0 for every edge.
        for equation in hull.equations:
            normal = equation[:2]
            offset = equation[2]

            margin = -(
                normal @ com_xy + offset
            ) / np.linalg.norm(normal)

            margins.append(margin)

        return float(min(margins))


    def is_stable(self, q, margin=0.05):
        """Return whether the COM is at least `margin` inside the support polygon."""

        return self.compute_stability(q) >= margin
