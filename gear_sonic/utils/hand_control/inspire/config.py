"""Configuration loading and validation for the Inspire extension."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from gear_sonic.utils.hand_control.inspire.contract import (
    CANONICAL_LOWER_RAD,
    CANONICAL_UPPER_RAD,
    CANONICAL_VELOCITY_LIMIT_RAD_S,
    HAND_DOF,
    JOINT_ORDER,
    ContractError,
    hand_vector,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MAPPING_PATH = PACKAGE_ROOT.parents[2] / "data/robots/g1/inspire/hand_config.yaml"


@dataclass(frozen=True)
class JointCalibration:
    name: str
    domain_index: int
    mujoco_joint: str
    mujoco_actuator: str
    urdf_joint: str
    direction: float
    scale: float
    offset: float
    lower: float
    upper: float
    velocity_limit: float

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "JointCalibration":
        lower, upper = value["position_limit"]
        calibration = cls(
            name=name,
            domain_index=int(value["domain_index"]),
            mujoco_joint=str(value["mujoco_joint"]),
            mujoco_actuator=str(value["mujoco_actuator"]),
            urdf_joint=str(value["urdf_joint"]),
            direction=float(value.get("direction", 1.0)),
            scale=float(value.get("scale", 1.0)),
            offset=float(value.get("offset", 0.0)),
            lower=float(lower),
            upper=float(upper),
            velocity_limit=float(value["velocity_limit"]),
        )
        if calibration.direction not in (-1.0, 1.0):
            raise ContractError(f"{name}.direction must be -1 or 1")
        numeric = np.asarray(
            [
                calibration.scale,
                calibration.offset,
                calibration.lower,
                calibration.upper,
                calibration.velocity_limit,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(numeric)):
            raise ContractError(f"{name} calibration values must be finite")
        if calibration.scale <= 0:
            raise ContractError(f"{name}.scale must be positive")
        if not calibration.lower < calibration.upper:
            raise ContractError(f"{name}.position_limit must be increasing")
        if calibration.velocity_limit <= 0:
            raise ContractError(f"{name}.velocity_limit must be positive")
        return calibration


@dataclass(frozen=True)
class InspireMapping:
    left: tuple[JointCalibration, ...]
    right: tuple[JointCalibration, ...]
    source_path: Path

    def validate_limits(self, value: Any, *, field_name: str, side: str = "left") -> np.ndarray:
        vector = hand_vector(value, field_name=field_name)
        if side not in ("left", "right"):
            raise ContractError(f"unknown hand side: {side!r}")
        joints = self.left if side == "left" else self.right
        lower = np.asarray([joint.lower for joint in joints], dtype=np.float32)
        upper = np.asarray([joint.upper for joint in joints], dtype=np.float32)
        invalid = np.flatnonzero((vector < lower) | (vector > upper))
        if invalid.size:
            index = int(invalid[0])
            raise ContractError(
                f"{field_name}[{index}]={vector[index]:.6g} outside [{lower[index]:.6g}, {upper[index]:.6g}]"
            )
        return vector


def load_mapping(path: str | Path = DEFAULT_MAPPING_PATH) -> InspireMapping:
    source_path = Path(path).expanduser().resolve()
    with source_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if document.get("schema") != "inspire.rh56dfx.mapping.v1":
        raise ContractError("mapping schema must be 'inspire.rh56dfx.mapping.v1'")
    if tuple(document.get("joint_order", ())) != JOINT_ORDER:
        raise ContractError(f"mapping joint_order must be {list(JOINT_ORDER)}")
    sides: dict[str, tuple[JointCalibration, ...]] = {}
    for side in ("left", "right"):
        raw_joints = document.get("joints", {}).get(side, {})
        if tuple(raw_joints) != JOINT_ORDER:
            raise ContractError(f"mapping joints.{side} must follow canonical joint order")
        joints = tuple(
            JointCalibration.from_mapping(name, raw_joints[name]) for name in JOINT_ORDER
        )
        if tuple(joint.domain_index for joint in joints) != tuple(range(HAND_DOF)):
            raise ContractError(f"mapping joints.{side} domain indices must be 0..5")
        sides[side] = joints

    for field in ("mujoco_joint", "mujoco_actuator", "urdf_joint"):
        names = [getattr(joint, field) for side in sides.values() for joint in side]
        if any(not name for name in names) or len(set(names)) != HAND_DOF * 2:
            raise ContractError(f"mapping {field} values must be non-empty and unique")
    for side, joints in sides.items():
        lower = np.asarray([joint.lower for joint in joints], dtype=np.float32)
        upper = np.asarray([joint.upper for joint in joints], dtype=np.float32)
        velocity = np.asarray([joint.velocity_limit for joint in joints], dtype=np.float32)
        if not np.array_equal(lower, CANONICAL_LOWER_RAD):
            raise ContractError(f"mapping joints.{side} lower limits differ from canonical native6")
        if not np.array_equal(upper, CANONICAL_UPPER_RAD):
            raise ContractError(f"mapping joints.{side} upper limits differ from canonical native6")
        if not np.array_equal(velocity, CANONICAL_VELOCITY_LIMIT_RAD_S):
            raise ContractError(
                f"mapping joints.{side} velocity limits differ from canonical native6"
            )
    return InspireMapping(left=sides["left"], right=sides["right"], source_path=source_path)


def hardware_settings(config_path=DEFAULT_MAPPING_PATH):
    with Path(config_path).expanduser().open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)["hardware"]
    rate = float(settings["command_rate_hz"])
    if not np.isfinite(rate) or not 1 <= rate <= 100:
        raise ValueError("command_rate_hz must be finite and in [1, 100]")
    for field in ("state_timeout_s", "operation_timeout_s"):
        if not np.isfinite(float(settings[field])) or float(settings[field]) <= 0:
            raise ValueError(f"{field} must be finite and positive")
    return settings


def driver_command(binary, config_path=DEFAULT_MAPPING_PATH):
    """Build argv without opening devices; consumed by the explicit launcher."""
    cfg = hardware_settings(config_path)
    for key in ("left_device", "right_device", "lock_path"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ValueError(f"configure hardware.{key} before launching the driver")
    if cfg["left_device"] == cfg["right_device"]:
        raise ValueError("left and right devices must differ")
    if cfg["baudrate"] not in (9600, 19200, 38400, 57600, 115200):
        raise ValueError("unsupported baudrate")
    for key in ("left_id", "right_id"):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= 254:
            raise ValueError(f"{key} must be an integer in [1, 254]")
    if type(cfg["runtime_speed"]) is not int or not 1 <= cfg["runtime_speed"] <= 1000:
        raise ValueError("runtime_speed must be in [1, 1000]")
    return [
        str(Path(binary).expanduser().resolve()),
        cfg["right_device"],
        cfg["left_device"],
        cfg["command_endpoint"],
        cfg["state_endpoint"],
        str(cfg["runtime_speed"]),
        str(cfg["command_rate_hz"]),
        cfg["lock_path"],
        str(cfg["baudrate"]),
        str(cfg["right_id"]),
        str(cfg["left_id"]),
    ]
