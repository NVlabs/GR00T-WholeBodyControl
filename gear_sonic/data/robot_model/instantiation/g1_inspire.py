"""Factory for the true 41-DoF G1 + RH56DFX data/FK model."""

from __future__ import annotations

from pathlib import Path


def instantiate_g1_rh56dfx_robot_model(
    *,
    waist_location: str = "lower_and_upper_body",
    high_elbow_pose: bool = False,
    config_path: str | Path | None = None,
):
    """Create the 41-joint model from the selected hand configuration.

    Inspire-specific dependencies are loaded only when this factory is called.
    """
    from gear_sonic.data.robot_model.instantiation.g1_inspire_assets import load_asset_paths
    from gear_sonic.data.robot_model.robot_model import RobotModel
    from gear_sonic.data.robot_model.supplemental_info.g1.g1_inspire_supplemental_info import (
        G1Rh56dfxSupplementalInfo,
    )
    from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import (
        ElbowPose,
        WaistLocation,
    )

    waist = WaistLocation(waist_location)
    elbow = ElbowPose.HIGH if high_elbow_pose else ElbowPose.LOW
    supplemental = G1Rh56dfxSupplementalInfo(waist_location=waist, elbow_pose=elbow)
    assets = load_asset_paths() if config_path is None else load_asset_paths(config_path)
    model = RobotModel(
        str(assets.urdf),
        str(assets.urdf.parent),
        supplemental_info=supplemental,
    )
    if model.num_joints != 41:
        raise ValueError(f"RH56DFX RobotModel must contain 41 joints, got {model.num_joints}")
    return model
