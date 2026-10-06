"""Supplemental RobotModel metadata for G1 with native RH56DFX hands."""

from __future__ import annotations

from copy import deepcopy

from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import (
    ElbowPose,
    G1SupplementalInfo,
    WaistLocation,
)
from gear_sonic.data.robot_model.supplemental_info.robot_supplemental_info import (
    RobotSupplementalInfo,
)
from gear_sonic.utils.hand_control.inspire.config import InspireMapping, load_mapping
from gear_sonic.utils.hand_control.inspire.contract import JOINT_ORDER


class G1Rh56dfxSupplementalInfo(RobotSupplementalInfo):
    """Build the 29-body + 6-left + 6-right native hand description."""

    def __init__(
        self,
        waist_location: WaistLocation = WaistLocation.LOWER_AND_UPPER_BODY,
        elbow_pose: ElbowPose = ElbowPose.LOW,
        mapping: InspireMapping | None = None,
    ) -> None:
        base = G1SupplementalInfo(waist_location=waist_location, elbow_pose=elbow_pose)
        mapping = mapping or load_mapping()
        left_joints = [joint.urdf_joint for joint in mapping.left]
        right_joints = [joint.urdf_joint for joint in mapping.right]

        groups = deepcopy(base.joint_groups)
        groups["left_hand"] = {"joints": left_joints, "groups": []}
        groups["right_hand"] = {"joints": right_joints, "groups": []}
        groups["hands"] = {"joints": [], "groups": ["left_hand", "right_hand"]}
        groups["upper_body"] = {
            "joints": [],
            "groups": ["upper_body_no_hands", "hands"],
        }

        limits = {
            name: value
            for name, value in base.joint_limits.items()
            if not name.startswith(("left_hand_", "right_hand_"))
        }
        for joint in mapping.left + mapping.right:
            limits[joint.urdf_joint] = [joint.lower, joint.upper]

        super().__init__(
            name="G1_RH56DFX_41DOF",
            body_actuated_joints=list(base.body_actuated_joints),
            left_hand_actuated_joints=left_joints,
            right_hand_actuated_joints=right_joints,
            joint_groups=groups,
            root_frame_name=base.root_frame_name,
            hand_frame_names={"left": "left_inspire_base", "right": "right_inspire_base"},
            joint_limits=limits,
            calibration_joint_q=deepcopy(base.calibration_joint_q),
            joint_name_mapping=deepcopy(base.joint_name_mapping),
            default_joint_q=deepcopy(base.default_joint_q),
            hand_rotation_correction=base.hand_rotation_correction.copy(),
            teleop_upper_body_motion_scale=base.teleop_upper_body_motion_scale,
        )

        if len(self.body_actuated_joints) != 29:
            raise ValueError("official G1 body contract changed; expected 29 actuated joints")
        expected = [f"left_inspire_{name}_joint" for name in JOINT_ORDER]
        if self.left_hand_actuated_joints != expected:
            raise ValueError("left RH56DFX joint order does not match the domain contract")
