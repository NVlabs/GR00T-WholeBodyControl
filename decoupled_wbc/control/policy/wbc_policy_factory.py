import os
from pathlib import Path
import time

import numpy as np

import decoupled_wbc
from decoupled_wbc.control.main.constants import DEFAULT_BASE_HEIGHT, DEFAULT_NAV_CMD
from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy
from decoupled_wbc.control.policy.identity_policy import IdentityPolicy
from decoupled_wbc.control.policy.interpolation_policy import InterpolationPolicy

from .g1_decoupled_whole_body_policy import G1DecoupledWholeBodyPolicy

WBC_VERSIONS = ["gear_wbc"]


def get_wbc_policy(
    robot_type,
    robot_model,
    wbc_config,
    init_time=time.monotonic(),
):
    current_upper_body_pose = robot_model.get_initial_upper_body_pose()

    if robot_type == "g1":
        upper_body_policy_type = wbc_config.get("upper_body_policy_type", "interpolation")
        if upper_body_policy_type == "identity":
            upper_body_policy = IdentityPolicy()
        else:
            upper_body_policy = InterpolationPolicy(
                init_time=init_time,
                init_values={
                    "target_upper_body_pose": current_upper_body_pose,
                    "base_height_command": np.array([DEFAULT_BASE_HEIGHT]),
                    "navigate_cmd": np.array([DEFAULT_NAV_CMD]),
                },
                max_change_rate=wbc_config["upper_body_max_joint_speed"],
            )

        lower_body_policy_type = wbc_config.get("VERSION", "gear_wbc")
        if lower_body_policy_type not in ["gear_wbc"]:
            raise ValueError(
                f"Invalid lower body policy version: {lower_body_policy_type}. "
                f"Only 'gear_wbc' is supported."
            )

        # Get the base path to decoupled_wbc and convert to Path object
        package_path = Path(os.path.dirname(decoupled_wbc.__file__))
        gear_wbc_config = str(package_path / ".." / wbc_config["GEAR_WBC_CONFIG"])
        if lower_body_policy_type == "gear_wbc":
            lower_body_policy = G1GearWbcPolicy(
                robot_model=robot_model,
                config=gear_wbc_config,
                model_path=wbc_config["model_path"],
            )

        wbc_policy = G1DecoupledWholeBodyPolicy(
            robot_model=robot_model,
            upper_body_policy=upper_body_policy,
            lower_body_policy=lower_body_policy,
        )
    else:
        raise ValueError(f"Invalid robot type: {robot_type}")
    return wbc_policy
