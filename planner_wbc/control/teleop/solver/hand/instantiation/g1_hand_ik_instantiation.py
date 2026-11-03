from decoupled_wbc.control.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)


# initialize hand ik solvers for g1 robot
def instantiate_g1_hand_ik_solver():
    left_hand_ik_solver = G1GripperInverseKinematicsSolver(side="left")
    right_hand_ik_solver = G1GripperInverseKinematicsSolver(side="right")
    return left_hand_ik_solver, right_hand_ik_solver
