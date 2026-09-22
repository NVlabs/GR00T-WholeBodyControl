"""BrainCo-aware 41-DOF dataset and camera schemas."""

from copy import deepcopy

import numpy as np

from gear_sonic.data.features_sonic_vla import (
    get_features_sonic_vla, get_modality_config_sonic_vla,
)

from .config import BraincoDataExporterConfig
from .constants import (
    BRAINCO_MOTOR_NAMES, BRAINCO_NUM_MOTORS, DEPTH_CAMERA_FEATURE,
    DEPTH_CAMERA_RGB_FEATURE, EXTERNAL_VIEW_CAMERA_FEATURE,
    HEAD_DEPTH_CAMERA_FEATURE, WEBCAM_FEATURE,
    get_g1_dds_features,
)

class BraincoSchema:
    """Build 41-DOF arrays while retaining the standard G1 body ordering."""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.left_indices = sorted(robot_model.get_joint_group_indices("left_hand"))
        self.right_indices = sorted(robot_model.get_joint_group_indices("right_hand"))
        self.hand_indices = set(self.left_indices + self.right_indices)
        self.left_insert_at = self.left_indices[0]
        self.right_insert_at = self.right_indices[0]

        self.joint_names: list[str] = []
        self.old_to_new: dict[int, int] = {}
        self.left_slice: slice | None = None
        self.right_slice: slice | None = None
        output_index = 0
        for index, name in enumerate(robot_model.joint_names):
            if index == self.left_insert_at:
                start = output_index
                self.joint_names.extend(f"left_brainco_{name}" for name in BRAINCO_MOTOR_NAMES)
                output_index += BRAINCO_NUM_MOTORS
                self.left_slice = slice(start, output_index)
            if index == self.right_insert_at:
                start = output_index
                self.joint_names.extend(f"right_brainco_{name}" for name in BRAINCO_MOTOR_NAMES)
                output_index += BRAINCO_NUM_MOTORS
                self.right_slice = slice(start, output_index)
            if index not in self.hand_indices:
                self.old_to_new[index] = output_index
                self.joint_names.append(name)
                output_index += 1

        if self.left_slice is None or self.right_slice is None:
            raise RuntimeError("Could not locate hand groups in the G1 robot model")
        if len(self.joint_names) != 41:
            raise RuntimeError(f"Expected a 41-DOF BrainCo schema, got {len(self.joint_names)}")

    def assemble(
        self, body_values: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray
    ) -> np.ndarray:
        full = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=np.asarray(body_values),
        )
        result: list[float] = []
        for index, value in enumerate(full):
            if index == self.left_insert_at:
                result.extend(np.asarray(left_hand, dtype=np.float64).reshape(6))
            if index == self.right_insert_at:
                result.extend(np.asarray(right_hand, dtype=np.float64).reshape(6))
            if index not in self.hand_indices:
                result.append(float(value))
        return np.asarray(result, dtype=np.float64)

    def body_configuration_for_fk(self, body_values: np.ndarray) -> np.ndarray:
        return self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=np.asarray(body_values)
        )


def get_brainco_features(robot_model, schema: BraincoSchema) -> dict:
    features = deepcopy(get_features_sonic_vla(robot_model))
    features.update(get_g1_dds_features())
    for side in ("left", "right"):
        features[f"observation.{side}_hand.tau_est"] = {
            "dtype": "float32",
            "shape": (BRAINCO_NUM_MOTORS,),
            "names": list(BRAINCO_MOTOR_NAMES),
        }
    for key in ("observation.state", "action.wbc"):
        features[key]["shape"] = (len(schema.joint_names),)
        features[key]["names"] = schema.joint_names
    for key in ("teleop.left_hand_joints", "teleop.right_hand_joints"):
        features[key]["shape"] = (BRAINCO_NUM_MOTORS,)
        features[key]["names"] = list(BRAINCO_MOTOR_NAMES)
    return features


def get_brainco_modality_config(robot_model, schema: BraincoSchema) -> dict:
    config = deepcopy(get_modality_config_sonic_vla(robot_model))
    for group in ("left_leg", "right_leg", "waist", "left_arm", "right_arm"):
        mapped = [
            schema.old_to_new[index]
            for index in robot_model.get_joint_group_indices(group)
        ]
        config["state"][group] = {"start": min(mapped), "end": max(mapped) + 1}
    config["state"]["left_hand"] = {
        "start": schema.left_slice.start,
        "end": schema.left_slice.stop,
    }
    config["state"]["right_hand"] = {
        "start": schema.right_slice.start,
        "end": schema.right_slice.stop,
    }
    config["action"]["left_hand_joints"]["end"] = BRAINCO_NUM_MOTORS
    config["action"]["right_hand_joints"]["end"] = BRAINCO_NUM_MOTORS
    for side in ("left", "right"):
        config["state"][f"{side}_hand_tau_est"] = {
            "start": 0,
            "end": BRAINCO_NUM_MOTORS,
            "original_key": f"observation.{side}_hand.tau_est",
        }
    for key, feature in get_g1_dds_features().items():
        if key.startswith("action."):
            group = "action"
            modality_name = key.removeprefix("action.").replace(".", "_")
        else:
            group = "state"
            modality_name = key.removeprefix("observation.").replace(".", "_")
        config[group][modality_name] = {
            "start": 0,
            "end": feature["shape"][0],
            "original_key": key,
        }
    return config


def configure_camera_schema(
    features: dict, modality_config: dict, config: BraincoDataExporterConfig
) -> None:
    """Configure enabled RGB streams and their optional depth previews."""
    features.pop("observation.images.ego_view", None)
    features.pop(DEPTH_CAMERA_RGB_FEATURE, None)
    features.pop(DEPTH_CAMERA_FEATURE, None)
    features.pop(WEBCAM_FEATURE, None)
    features.pop(HEAD_DEPTH_CAMERA_FEATURE, None)
    features.pop(EXTERNAL_VIEW_CAMERA_FEATURE, None)

    video_config = {}
    if not config.ignore_ego_view:
        features[DEPTH_CAMERA_RGB_FEATURE] = {
            "dtype": "video",
            "shape": [config.depth_camera_height, config.depth_camera_width, 3],
            "names": ["height", "width", "channel"],
        }
        video_config["depth_camera_rgb"] = {
            "original_key": DEPTH_CAMERA_RGB_FEATURE
        }
        if config.record_depth_video:
            features[DEPTH_CAMERA_FEATURE] = {
                "dtype": "video",
                "shape": [config.depth_stream_height, config.depth_stream_width, 3],
                "names": ["height", "width", "channel"],
            }
            video_config["depth_camera"] = {
                "original_key": DEPTH_CAMERA_FEATURE
            }

    if not config.ignore_head:
        features[WEBCAM_FEATURE] = {
            "dtype": "video",
            "shape": [config.webcam_height, config.webcam_width, 3],
            "names": ["height", "width", "channel"],
        }
        video_config["webcam"] = {"original_key": WEBCAM_FEATURE}
        if config.record_depth_video:
            features[HEAD_DEPTH_CAMERA_FEATURE] = {
                "dtype": "video",
                "shape": [
                    config.head_depth_stream_height,
                    config.head_depth_stream_width,
                    3,
                ],
                "names": ["height", "width", "channel"],
            }
            video_config["head_depth_camera"] = {
                "original_key": HEAD_DEPTH_CAMERA_FEATURE
            }

    if config.external_view_camera_host is not None:
        features[EXTERNAL_VIEW_CAMERA_FEATURE] = {
            "dtype": "video",
            "shape": [
                config.external_view_camera_height,
                config.external_view_camera_width,
                3,
            ],
            "names": ["height", "width", "channel"],
        }
        # Keep the user-facing dataset modality name requested for this view;
        # the feature key itself follows the existing underscore convention.
        video_config["external-view-camera"] = {
            "original_key": EXTERNAL_VIEW_CAMERA_FEATURE
        }
    modality_config["video"] = video_config
