from dataclasses import dataclass


@dataclass
class BodyIKSolverSettings:
    def __init__(self):
        self.dt = 0.05
        self.num_step_per_frame = 3
        self.amplify_factor = 1.0
        self.posture_cost = 0.01
        self.posture_lm_damping = 1.0
        self.link_costs = {
            "hand": {
                "orientation_cost": 2.0,
                "position_cost": 8.0,
                "lm_damping": 3.0,
            }
        }
        self.posture_weight = {
            "waist_pitch": 10.0,
            "waist_yaw": 2.0,
            "waist_roll": 10.0,
            "shoulder_pitch": 4.0,
            "shoulder_roll": 3.0,
            "shoulder_yaw": 0.1,
            "elbow_pitch": 3.0,
            "wrist_pitch": 1.0,
            "wrist_yaw": 0.1,
        }
        # These joint limits override the joint limits in the robot model for the IK solver
        self.ik_joint_limits = {
            # Increase waist pitch limit since lower body policy can use hip joints to pitch
            "waist_pitch": [-0.52, 0.9],
            "elbow_pitch": [-1.0472, 1.4],
        }
