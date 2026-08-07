import os
from pathlib import Path
from typing import Literal

from decoupled_wbc.control.robot_model.robot_model import RobotModel
from decoupled_wbc.control.robot_model.supplemental_info.g1.g1_supplemental_info import (
    ElbowPose,
    G1SupplementalInfo,
    WaistLocation,
)


def instantiate_g1_robot_model(
    waist_location: Literal["lower_body", "upper_body", "lower_and_upper_body"] = "lower_body",
    high_elbow_pose: bool = False,
):
    """
    Instantiate a G1 robot model with configurable waist location and pose.

    Args:
        waist_location: Whether to put waist in "lower_body" (default G1 behavior),
                        "upper_body" (waist controlled with arms/manipulation via IK),
                        or "lower_and_upper_body" (waist reference from arms/manipulation
                        via IK then passed to lower body policy)
        high_elbow_pose: Whether to use high elbow pose configuration for default joint positions

    Returns:
        RobotModel: Configured G1 robot model
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    robot_model_config = {
        "asset_path": os.path.join(project_root, "decoupled_wbc/control/robot_model/model_data/g1"),
        "urdf_path": os.path.join(
            project_root, "decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
        ),
    }
    assert waist_location in [
        "lower_body",
        "upper_body",
        "lower_and_upper_body",
    ], f"Invalid waist_location: {waist_location}. Must be 'lower_body' or 'upper_body' or 'lower_and_upper_body'"

    # Map string values to enums
    waist_location_enum = {
        "lower_body": WaistLocation.LOWER_BODY,
        "upper_body": WaistLocation.UPPER_BODY,
        "lower_and_upper_body": WaistLocation.LOWER_AND_UPPER_BODY,
    }[waist_location]

    elbow_pose_enum = ElbowPose.HIGH if high_elbow_pose else ElbowPose.LOW

    # Create single configurable supplemental info instance
    robot_model_supplemental_info = G1SupplementalInfo(
        waist_location=waist_location_enum, elbow_pose=elbow_pose_enum
    )

    robot_model = RobotModel(
        robot_model_config["urdf_path"],
        robot_model_config["asset_path"],
        supplemental_info=robot_model_supplemental_info,
    )
    return robot_model
