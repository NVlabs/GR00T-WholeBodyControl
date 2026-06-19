from decoupled_wbc.control.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
    G1InspireGripperInverseKinematicsSolver,
)


# initialize hand ik solvers for g1 robot
def instantiate_g1_hand_ik_solver(hand_type: str = "dex3"):
    """Instantiate left/right hand IK solvers matching the robot's hand type.

    Args:
        hand_type: "dex3" for the Unitree three-finger hand (7 DOF) or "inspire"
                   for the Inspire RH56 hand (6 DOF). Must match the hand_type used
                   to build the robot model.
    """
    if hand_type == "dex3":
        solver_cls = G1GripperInverseKinematicsSolver
    elif hand_type == "inspire":
        solver_cls = G1InspireGripperInverseKinematicsSolver
    else:
        raise ValueError(f"Unsupported hand_type: {hand_type}. Must be 'dex3' or 'inspire'.")

    left_hand_ik_solver = solver_cls(side="left")
    right_hand_ik_solver = solver_cls(side="right")
    return left_hand_ik_solver, right_hand_ik_solver
