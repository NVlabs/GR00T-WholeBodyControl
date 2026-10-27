"""Configure the shared G1 + Inspire scene using named actuator limits."""

from pathlib import Path
import xml.etree.ElementTree as ET

from gear_sonic.data.robot_model.instantiation.g1_inspire_assets import (
    PACKAGE_PARENT,
    load_asset_paths,
)
from gear_sonic.utils.hand_control.inspire.config import DEFAULT_MAPPING_PATH, load_mapping
from gear_sonic.utils.mujoco_sim.inspire.controller import SimControlConfig


def _actuated_joint_names(path: Path) -> list[str]:
    actuator = ET.parse(path).getroot().find("actuator")
    if actuator is None or any(element.get("joint") is None for element in actuator):
        raise ValueError(f"MJCF must have named joint actuators: {path}")
    return [element.get("joint") for element in actuator]


def configure_simulation(config, config_path=DEFAULT_MAPPING_PATH):
    """Preserve body settings, remapping effort limits by name for passive fingers."""
    config = dict(config)
    if config.get("FREE_BASE"):
        raise ValueError("Inspire scene uses a floating joint, not actuated root coordinates")
    assets = load_asset_paths(config_path)
    mapping = load_mapping(config_path)
    official = PACKAGE_PARENT / "gear_sonic/data/robots/g1/g1_29dof_with_hand.xml"
    names = _actuated_joint_names(official)
    limits = config["motor_effort_limit_list"]
    if len(names) != len(limits):
        raise ValueError("body effort limits must match the original actuator layout")
    body_limits = {name: value for name, value in zip(names, limits) if "_hand_" not in name}
    hand_names = {joint.mujoco_joint for joint in (*mapping.left, *mapping.right)}
    target_names = _actuated_joint_names(assets.model)
    if len(body_limits) != 29 or set(target_names) != set(body_limits) | hand_names:
        raise ValueError("Inspire scene must contain body29 + hands12 actuators")
    hand_limit = SimControlConfig().effort_limit
    config["motor_effort_limit_list"] = [
        hand_limit if name in hand_names else body_limits[name] for name in target_names
    ]
    config.update(
        ROBOT_SCENE=str(assets.scene), NUM_HAND_MOTORS=6, NUM_HAND_JOINTS=6, HAND_TYPE="inspire"
    )
    config["HAND_JOINT_NAMES"] = {
        side: [joint.mujoco_joint for joint in getattr(mapping, side)] for side in ("left", "right")
    }
    config["HAND_ACTUATOR_NAMES"] = {
        side: [joint.mujoco_actuator for joint in getattr(mapping, side)]
        for side in ("left", "right")
    }
    return config
