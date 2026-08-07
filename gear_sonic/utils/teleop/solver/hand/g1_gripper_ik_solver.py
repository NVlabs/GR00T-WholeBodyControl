"""IK solver that maps fingertip distances to G1 gripper joint targets.

Computes a 7-DOF hand joint vector from thumb-to-finger distances,
selecting a gesture (index / middle / ring / pinky) based on which
finger has the largest grip value and interpolating toward a preset
closed pose.
"""

import numpy as np

from gear_sonic.utils.teleop.solver.solver import Solver
class G1GripperInverseKinematicsSolver(Solver):
    def __init__(self, side) -> None:
        self.side = "L" if side.lower() == "left" else "R"

    def register_robot(self, robot):
        pass

    def __call__(self, finger_data):
        # manus data
        fingertips = finger_data["position"]

        # Extract X, Y, Z positions of fingertips from the transformation matrices
        positions = np.array([finger[:3, 3] for finger in fingertips])

        # Ensure the positions are 2D arrays (N, 3)
        positions = np.reshape(positions, (-1, 3))  # Ensure 2D array with shape (N, 3)

        # Fingertip positions: each finger has 5 joints, tip is at base_index + 4
        # thumb=4, index=9, middle=14, ring=19, pinky=24
        thumb_pos = positions[4, :]
        index_pos = positions[4 + 5, :]
        middle_pos = positions[4 + 10, :]
        ring_pos = positions[4 + 15, :]
        pinky_pos = positions[4 + 20, :]

        # Calculate distances for continuous grip control
        # When thumb at (1,0,0) and finger at (grip_value,0,0): dist = 1.0 - grip_value
        index_dist = np.linalg.norm(thumb_pos - index_pos)
        middle_dist = np.linalg.norm(thumb_pos - middle_pos)
        ring_dist = np.linalg.norm(thumb_pos - ring_pos)
        pinky_dist = np.linalg.norm(thumb_pos - pinky_pos)

        # Dead zone threshold - ignore very small grip values, snap to full at high values
        dist_threshold = 0.05

        # Convert distance to grip amount (0.0 = open, 1.0 = closed)
        index_grip = np.clip(1.0 - index_dist, 0.0, 1.0)
        middle_grip = np.clip(1.0 - middle_dist, 0.0, 1.0)
        ring_grip = np.clip(1.0 - ring_dist, 0.0, 1.0)
        pinky_grip = np.clip(1.0 - pinky_dist, 0.0, 1.0)

        # Apply dead zone: ignore small values, snap to full near 1.0
        def apply_dead_zone(grip, threshold):
            if grip < threshold:
                return 0.0
            return grip

        index_grip = apply_dead_zone(index_grip, dist_threshold)
        middle_grip = apply_dead_zone(middle_grip, dist_threshold)
        ring_grip = apply_dead_zone(ring_grip, dist_threshold)
        pinky_grip = apply_dead_zone(pinky_grip, dist_threshold)

        # Choose the active gesture based on which finger has the highest grip value
        # Each gesture type maps to a different close pose
        q_open = np.zeros(7)

        # Find the finger with the highest grip value
        grips = [index_grip, middle_grip, ring_grip, pinky_grip]
        max_grip = max(grips)

        if max_grip == 0:
            # No grip - fully open
            q_desired = q_open
        elif index_grip == max_grip:
            # Index gesture (trigger only): interpolate to index close pose
            q_closed = self._get_index_close_q_desired()
            q_desired = q_open + index_grip * (q_closed - q_open)
        elif middle_grip == max_grip:
            # Middle gesture (both pressed): interpolate to middle close pose
            q_closed = self._get_middle_close_q_desired()
            q_desired = q_open + middle_grip * (q_closed - q_open)
        elif ring_grip == max_grip:
            # Ring gesture (grip only): interpolate to ring close pose
            q_closed = self._get_ring_close_q_desired()
            q_desired = q_open + ring_grip * (q_closed - q_open)
        else:
            # Pinky gesture: interpolate to pinky close pose
            q_closed = self._get_pinky_close_q_desired()
            q_desired = q_open + pinky_grip * (q_closed - q_open)

        return q_desired

    def _get_index_close_q_desired(self):
        q_desired = np.zeros(7)

        amp0 = 0.5
        if self.side == "L":
            q_desired[0] -= amp0
        else:
            q_desired[0] += amp0

        amp = 0.7

        q_desired[1] += amp
        q_desired[2] += amp

        ampA1 = 1.5
        ampB1 = 1.5
        ampA2 = 0.6
        ampB2 = 1.5

        q_desired[3] -= ampA1
        q_desired[4] -= ampB1
        q_desired[5] -= ampA2
        q_desired[6] -= ampB2

        # Right hand has mirrored joint convention, so negate all targets
        return q_desired if self.side == "L" else -q_desired

    def _get_middle_close_q_desired(self):
        q_desired = np.zeros(7)

        amp0 = 0.0
        if self.side == "L":
            q_desired[0] -= amp0
        else:
            q_desired[0] += amp0

        amp = 0.7

        q_desired[1] += amp
        q_desired[2] += amp

        ampA1 = 1.0
        ampB1 = 1.5
        ampA2 = 1.0
        ampB2 = 1.5

        q_desired[3] -= ampA1
        q_desired[4] -= ampB1
        q_desired[5] -= ampA2
        q_desired[6] -= ampB2

        return q_desired if self.side == "L" else -q_desired

    def _get_ring_close_q_desired(self):
        q_desired = np.zeros(7)

        amp0 = -0.5
        if self.side == "L":
            q_desired[0] -= amp0
        else:
            q_desired[0] += amp0

        amp = 0.7

        q_desired[1] += amp
        q_desired[2] += amp

        ampA1 = 0.6
        ampB1 = 1.5
        ampA2 = 1.5
        ampB2 = 1.5

        q_desired[3] -= ampA1
        q_desired[4] -= ampB1
        q_desired[5] -= ampA2
        q_desired[6] -= ampB2

        return q_desired if self.side == "L" else -q_desired

    def _get_pinky_close_q_desired(self):
        q_desired = np.zeros(7)

        return q_desired if self.side == "L" else -q_desired
