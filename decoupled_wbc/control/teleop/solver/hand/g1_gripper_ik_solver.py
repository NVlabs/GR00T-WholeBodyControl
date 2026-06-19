import numpy as np

from decoupled_wbc.control.envs.g1.utils import inspire_hand_spec
from decoupled_wbc.control.teleop.solver.solver import Solver


def _pinch_target(finger_data, dist_threshold: float = 0.05):
    """Detect which fingertip (if any) is pinching the thumb from Manus glove data.

    Returns one of "index", "middle", "ring", "pinky", or None. Shared by the
    dex3 and inspire gripper solvers so both interpret the glove identically.
    """
    fingertips = finger_data["position"]
    positions = np.array([finger[:3, 3] for finger in fingertips])
    positions = np.reshape(positions, (-1, 3))

    thumb_pos = positions[4, :]
    index_pos = positions[4 + 5, :]
    middle_pos = positions[4 + 10, :]
    ring_pos = positions[4 + 15, :]
    pinky_pos = positions[4 + 20, :]

    if np.linalg.norm(thumb_pos - index_pos) < dist_threshold:
        return "index"
    if np.linalg.norm(thumb_pos - middle_pos) < dist_threshold:
        return "middle"
    if np.linalg.norm(thumb_pos - ring_pos) < dist_threshold:
        return "ring"
    if np.linalg.norm(thumb_pos - pinky_pos) < dist_threshold:
        return "pinky"
    return None


######################################
### Define your solver here
######################################
class G1GripperInverseKinematicsSolver(Solver):
    def __init__(self, side) -> None:
        self.side = "L" if side.lower() == "left" else "R"

    def register_robot(self, robot):
        pass

    def __call__(self, finger_data):
        target = _pinch_target(finger_data)

        if target == "index":
            return self._get_index_close_q_desired()
        elif target == "middle":
            return self._get_middle_close_q_desired()
        elif target == "ring":
            return self._get_ring_close_q_desired()
        elif target == "pinky":
            return self._get_pinky_close_q_desired()

        return np.zeros(7)

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


class G1InspireGripperInverseKinematicsSolver(Solver):
    """Heuristic pinch -> joint solver for the Inspire RH56 6-DOF hand.

    Uses the same Manus pinch detection as the dex3 gripper solver, but outputs
    joint angles for the 6 Inspire drives in the actuator order defined by
    ``inspire_hand_spec.JOINT_ORDER`` (pinky, ring, middle, index, thumb_bend,
    thumb_rot). Angles are in radians where 0 is fully open and positive is
    flexion; values stay within the URDF joint limits. The pose deepens as the
    pinch moves from index toward pinky (index -> ... -> full grasp). These
    amplitudes are intentionally simple and meant to be tuned on hardware.
    """

    # Flexion targets (radians), ~80% of the inspire joint travel from the manual
    # (fingers [0, 2.72], thumb_bend [0, 1.45], thumb_rot [0, 1.31]).
    FINGER_FLEX = 2.2
    THUMB_BEND = 1.1
    THUMB_ROT = 1.0

    # Index of each named joint within the 6-vector, from the shared spec.
    _IDX = {name: i for i, name in enumerate(inspire_hand_spec.JOINT_ORDER)}

    # Which fingers close for each detected pinch (progressively more fingers).
    _PINCH_TO_FINGERS = {
        "index": ("index",),
        "middle": ("index", "middle"),
        "ring": ("index", "middle", "ring"),
        "pinky": ("index", "middle", "ring", "pinky"),
    }

    def __init__(self, side) -> None:
        self.side = "L" if side.lower() == "left" else "R"

    def register_robot(self, robot):
        pass

    def __call__(self, finger_data):
        q_desired = np.zeros(inspire_hand_spec.NUM_INSPIRE_DOF)

        target = _pinch_target(finger_data)
        if target is None:
            return q_desired

        for finger in self._PINCH_TO_FINGERS[target]:
            q_desired[self._IDX[finger]] = self.FINGER_FLEX

        # Oppose with the thumb whenever any pinch is active.
        q_desired[self._IDX["thumb_bend"]] = self.THUMB_BEND
        q_desired[self._IDX["thumb_rot"]] = self.THUMB_ROT

        return q_desired
