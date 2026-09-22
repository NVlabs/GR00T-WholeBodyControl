"""Dataset feature names, joint metadata, and control gains."""

import numpy as np

from gear_sonic.g1_upper_body_telemetry.protocol import JOINT_NAMES


BRAINCO_MOTOR_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
BRAINCO_NUM_MOTORS = len(BRAINCO_MOTOR_NAMES)
BRAINCO_MAX_AGE_SEC = 0.25
DEPTH_CAMERA_RGB_FEATURE = "observation.images.depth_camera_rgb"
DEPTH_CAMERA_FEATURE = "observation.images.depth_camera"
WEBCAM_FEATURE = "observation.images.webcam"
HEAD_DEPTH_CAMERA_FEATURE = "observation.images.head_depth_camera"
EXTERNAL_VIEW_CAMERA_FEATURE = "observation.images.external_view_camera"

G1_UPPER_BODY_JOINT_NAMES = JOINT_NAMES
G1_UPPER_BODY_SIZE = len(G1_UPPER_BODY_JOINT_NAMES)

# Only these telemetry arrays are persisted in new datasets.  The robot-host
# wire protocol intentionally remains richer for backward compatibility and
# diagnostics.
G1_DATASET_ACTION_FIELDS = ("q_cmd", "kp", "kd")
G1_DATASET_OBSERVATION_FIELDS = (
    "q_est",
    "dq_est",
    "tau_est",
    "q_residual",
)
G1_RAW_TELEMETRY_ARRAY_FIELDS = (
    *G1_DATASET_ACTION_FIELDS,
    *G1_DATASET_OBSERVATION_FIELDS,
)

# Same gains used by gear_sonic_deploy/.../policy_parameters.hpp.
_NATURAL_FREQ = 10.0 * 2.0 * np.pi
_DAMPING_RATIO = 2.0
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_4010 = 0.00425


def _stiffness(armature: float) -> float:
    return armature * _NATURAL_FREQ**2


def _damping(armature: float) -> float:
    return 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ


G1_UPPER_BODY_STIFFNESS_KP = (
    _stiffness(_ARMATURE_7520_14),
    2.0 * _stiffness(_ARMATURE_5020),
    2.0 * _stiffness(_ARMATURE_5020),
    *([_stiffness(_ARMATURE_5020)] * 5),
    *([_stiffness(_ARMATURE_4010)] * 2),
    *([_stiffness(_ARMATURE_5020)] * 5),
    *([_stiffness(_ARMATURE_4010)] * 2),
)
G1_UPPER_BODY_DAMPING_KD = (
    _damping(_ARMATURE_7520_14),
    2.0 * _damping(_ARMATURE_5020),
    2.0 * _damping(_ARMATURE_5020),
    *([_damping(_ARMATURE_5020)] * 5),
    *([_damping(_ARMATURE_4010)] * 2),
    *([_damping(_ARMATURE_5020)] * 5),
    *([_damping(_ARMATURE_4010)] * 2),
)

def get_g1_dds_features() -> dict:
    joints = list(G1_UPPER_BODY_JOINT_NAMES)
    result = {}
    for field in G1_DATASET_ACTION_FIELDS:
        result[f"action.upper_body.{field}"] = {
            "dtype": "float32", "shape": (G1_UPPER_BODY_SIZE,), "names": joints,
        }
    for field in G1_DATASET_OBSERVATION_FIELDS:
        result[f"observation.upper_body.{field}"] = {
            "dtype": "float32", "shape": (G1_UPPER_BODY_SIZE,), "names": joints,
        }
    result.update(
        {
            "observation.upper_body.source_age_sec": {
                "dtype": "float32", "shape": (2,), "names": ["lowstate", "lowcmd"],
            },
            "observation.upper_body.telemetry_timestamps_ns": {
                "dtype": "int64", "shape": (2,),
                "names": ["robot_publish_wall", "host_receive_wall"],
            },
            "observation.upper_body.telemetry_sequence_id": {
                "dtype": "int64", "shape": (1,), "names": ["sequence_id"],
            },
        }
    )
    return result
