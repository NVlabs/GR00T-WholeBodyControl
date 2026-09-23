import copy
import os
import pathlib
import robosuite
from robosuite.controllers import load_composite_controller_config


def is_stale_controller_config(config: dict):
    """
    Checks if a controller config is in the format from robosuite versions <= 1.4.1.
    Does not check for config validity, only the format type.
    Args:
        config (dict): Controller configuration
    Returns:
        bool: True if the config is in the old format, False otherwise
    """

    OLD_CONTROLLER_TYPES = [
        "JOINT_VELOCITY",
        "JOINT_TORQUE",
        "JOINT_POSITION",
        "OSC_POSITION",
        "OSC_POSE",
        "IK_POSE",
    ]
    if (
        "body_parts" not in config or "body_parts_controller_configs" not in config
    ) and "type" in config:
        return config["type"] in OLD_CONTROLLER_TYPES
    return False


def refactor_composite_controller_config(controller_config, robot_type, arms):
    """
    Checks if a controller config is in the format from robosuite versions <= 1.4.1.
    If this is the case, converts the controller config to the new composite controller
    config format in robosuite versions >= 1.5. If the robot has a default
    controller config use that and override the arms with the old controller config.
    If not just use the old controller config for arms.
    Args:
        old_controller_config (dict): Old controller config
    Returns:
        dict: New controller config
    """
    if not is_stale_controller_config(controller_config):
        return controller_config

    config_dir = pathlib.Path(robosuite.__file__).parent / "controllers/config/robots/"
    name = robot_type.lower()
    configs = os.listdir(config_dir)
    if f"default_{name}.json" in configs:
        new_controller_config = load_composite_controller_config(robot=name)
    else:
        new_controller_config = {}
        new_controller_config["type"] = "BASIC"
        new_controller_config["body_parts"] = {}

    for arm in arms:
        new_arm_config = copy.deepcopy(controller_config)
        if "gripper" not in new_arm_config:
            new_arm_config["gripper"] = {"type": "GRIP"}
        new_controller_config["body_parts"][arm] = new_arm_config
    return new_controller_config
