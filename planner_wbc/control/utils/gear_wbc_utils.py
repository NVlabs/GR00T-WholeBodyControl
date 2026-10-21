import os

import numpy as np
import yaml


def load_config(config_path):
    """Load and process the YAML configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set the path to the LEGGED_GYM_ROOT_DIR using relative path
    current_file_dir = os.path.dirname(os.path.abspath(config_path))
    LEGGED_GYM_ROOT_DIR = os.path.join(current_file_dir, "..", "GearWbcRL", "legged_gym")
    LEGGED_GYM_ROOT_DIR = os.path.abspath(LEGGED_GYM_ROOT_DIR)

    # Process paths with LEGGED_GYM_ROOT_DIR
    for path_key in ["policy_path", "xml_path", "onnx_policy_path"]:
        if path_key in config:
            config[path_key] = config[path_key].format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

    # Convert lists to numpy arrays where needed
    array_keys = ["kps", "kds", "default_angles", "cmd_scale", "cmd_init"]
    for key in array_keys:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)

    return config, LEGGED_GYM_ROOT_DIR


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    q_conj = np.array([w, -x, -y, -z])

    return np.array(
        [
            v[0] * (q_conj[0] ** 2 + q_conj[1] ** 2 - q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3])
            + v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
            v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3])
            + v[1] * (q_conj[0] ** 2 - q_conj[1] ** 2 + q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
            v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2])
            + v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1])
            + v[2] * (q_conj[0] ** 2 - q_conj[1] ** 2 - q_conj[2] ** 2 + q_conj[3] ** 2),
        ]
    )


def get_gravity_orientation(quat):
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)


def compute_observation(d, config, action, cmd, height_cmd, n_joints):
    """Compute the observation vector from current state"""
    # Get state from MuJoCo
    qj = d.qpos[7 : 7 + n_joints].copy()
    dqj = d.qvel[6 : 6 + n_joints].copy()
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()

    # Handle default angles padding
    if len(config["default_angles"]) < n_joints:
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[: len(config["default_angles"])] = config["default_angles"]
    else:
        padded_defaults = config["default_angles"][:n_joints]

    # Scale the values
    qj_scaled = (qj - padded_defaults) * config["dof_pos_scale"]
    dqj_scaled = dqj * config["dof_vel_scale"]
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * config["ang_vel_scale"]

    # Calculate single observation dimension
    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + 12

    # Create single observation
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd[:3] * config["cmd_scale"]
    single_obs[3:4] = np.array([height_cmd])
    single_obs[4:7] = omega_scaled
    single_obs[7:10] = gravity_orientation
    single_obs[10 : 10 + n_joints] = qj_scaled
    single_obs[10 + n_joints : 10 + 2 * n_joints] = dqj_scaled
    single_obs[10 + 2 * n_joints : 10 + 2 * n_joints + 12] = action

    return single_obs, single_obs_dim
