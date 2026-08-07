import collections
import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
from pynput import keyboard as pkb
import torch
import yaml


class GearWbcController:
    def __init__(self, config_path):
        self.CONFIG_PATH = config_path
        self.cmd_lock = threading.Lock()
        self.config = self.load_config(os.path.join(self.CONFIG_PATH, "g1_gear_wbc.yaml"))

        self.control_dict = {
            "loco_cmd": self.config["cmd_init"],
            "height_cmd": self.config["height_cmd"],
            "rpy_cmd": self.config.get("rpy_cmd", [0.0, 0.0, 0.0]),
            "freq_cmd": self.config.get("freq_cmd", 1.5),
        }

        self.model = mujoco.MjModel.from_xml_path(self.config["xml_path"])
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.config["simulation_dt"]
        self.n_joints = self.data.qpos.shape[0] - 7
        self.torso_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.base_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.action = np.zeros(self.config["num_actions"], dtype=np.float32)
        self.target_dof_pos = self.config["default_angles"].copy()
        self.policy = self.load_onnx_policy(self.config["policy_path"])
        self.walk_policy = self.load_onnx_policy(self.config["walk_policy_path"])
        self.gait_indices = torch.zeros((1), dtype=torch.float32)
        self.counter = 0
        self.just_started = 0.0
        self.walking_mask = False
        self.frozen_FL = False
        self.frozen_FR = False
        self.single_obs, self.single_obs_dim = self.compute_observation(
            self.data, self.config, self.action, self.control_dict, self.n_joints
        )
        self.obs_history = collections.deque(
            [np.zeros(self.single_obs_dim, dtype=np.float32)] * self.config["obs_history_len"],
            maxlen=self.config["obs_history_len"],
        )
        self.obs = np.zeros(self.config["num_obs"], dtype=np.float32)
        self.keyboard_listener(self.control_dict, self.config)

    def keyboard_listener(self, control_dict, config):
        """Listen to key press events and update cmd and height_cmd"""

        def on_press(key):
            try:
                k = key.char
            except AttributeError:
                return  # Special keys ignored

            with self.cmd_lock:
                if k == "w":
                    control_dict["loco_cmd"][0] += 0.1
                elif k == "s":
                    control_dict["loco_cmd"][0] -= 0.1
                elif k == "a":
                    control_dict["loco_cmd"][1] += 0.1
                elif k == "d":
                    control_dict["loco_cmd"][1] -= 0.1
                elif k == "q":
                    control_dict["loco_cmd"][2] += 0.1
                elif k == "e":
                    control_dict["loco_cmd"][2] -= 0.1
                elif k == "z":
                    control_dict["loco_cmd"][:] = config["cmd_init"]
                    control_dict["height_cmd"] = config["height_cmd"]
                    control_dict["rpy_cmd"][:] = config["rpy_cmd"]
                    control_dict["freq_cmd"] = config["freq_cmd"]
                elif k == "1":
                    control_dict["height_cmd"] += 0.05
                elif k == "2":
                    control_dict["height_cmd"] -= 0.05
                elif k == "3":
                    control_dict["rpy_cmd"][0] += 0.2
                elif k == "4":
                    control_dict["rpy_cmd"][0] -= 0.2
                elif k == "5":
                    control_dict["rpy_cmd"][1] += 0.2
                elif k == "6":
                    control_dict["rpy_cmd"][1] -= 0.2
                elif k == "7":
                    control_dict["rpy_cmd"][2] += 0.2
                elif k == "8":
                    control_dict["rpy_cmd"][2] -= 0.2
                elif k == "m":
                    control_dict["freq_cmd"] += 0.1
                elif k == "n":
                    control_dict["freq_cmd"] -= 0.1

                print(
                    f"Current Commands: loco_cmd = {control_dict['loco_cmd']}, height_cmd = {control_dict['height_cmd']}, rpy_cmd = {control_dict['rpy_cmd']}, freq_cmd = {control_dict['freq_cmd']}"
                )

        listener = pkb.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        for path_key in ["policy_path", "xml_path", "walk_policy_path"]:
            config[path_key] = os.path.join(CONFIG_PATH, config[path_key])

        array_keys = ["kps", "kds", "default_angles", "cmd_scale", "cmd_init"]
        for key in array_keys:
            config[key] = np.array(config[key], dtype=np.float32)

        return config

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        return (target_q - q) * kp + (target_dq - dq) * kd

    def quat_rotate_inverse(self, q, v):
        w, x, y, z = q
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

    def get_gravity_orientation(self, quat):
        gravity_vec = np.array([0.0, 0.0, -1.0])
        return self.quat_rotate_inverse(quat, gravity_vec)

    def compute_observation(self, d, config, action, control_dict, n_joints):
        command = np.zeros(7, dtype=np.float32)
        command[:3] = control_dict["loco_cmd"][:3] * config["cmd_scale"]
        command[3] = control_dict["height_cmd"]
        # command[4] = control_dict['freq_cmd']
        command[4:7] = control_dict["rpy_cmd"]

        # # gait indice
        # is_static = np.linalg.norm(command[:3]) < 0.1
        # just_entered_walk = (not is_static) and (not self.walking_mask)
        # self.walking_mask = not is_static

        # if just_entered_walk:
        #     self.just_started = 0.0
        #     self.gait_indices = torch.tensor([-0.25])
        # if not is_static:
        #     self.just_started += 0.02
        # else:
        #     self.just_started = 0.0

        # if not is_static:
        #     self.frozen_FL = False
        #     self.frozen_FR = False

        # self.gait_indices = torch.remainder(self.gait_indices + 0.02 * command[4], 1.0)

        # # Parameters
        # duration = 0.5
        # phase = 0.5

        # # Gait indices
        # gait_FR = self.gait_indices.clone()
        # gait_FL = torch.remainder(gait_FR + phase, 1.0)

        # if self.just_started < (0.5 / command[4]):
        #     gait_FR = torch.tensor([0.25])
        # gait_pair = [gait_FL.clone(), gait_FR.clone()]

        # for i, fi in enumerate(gait_pair):
        #     if fi.item() < duration:
        #         gait_pair[i] = fi * (0.5 / duration)
        #     else:
        #         gait_pair[i] = 0.5 + (fi - duration) * (0.5 / (1 - duration))

        # # Clock signal
        # clock = [torch.sin(2 * np.pi * fi) for fi in gait_pair]

        # for i, (clk, frozen_mask_attr) in enumerate(
        #     zip(clock, ['frozen_FL', 'frozen_FR'])
        # ):
        #     frozen_mask = getattr(self, frozen_mask_attr)
        #     # Freeze condition: static and at sin peak
        #     if is_static and (not frozen_mask) and clk.item() > 0.98:
        #         setattr(self, frozen_mask_attr, True)
        #         clk = torch.tensor([1.0])
        #     if getattr(self, frozen_mask_attr):
        #         clk = torch.tensor([1.0])
        #     clock[i] = clk

        # self.clock_inputs = torch.stack(clock).unsqueeze(0)
        qj = d.qpos[7 : 7 + n_joints].copy()
        dqj = d.qvel[6 : 6 + n_joints].copy()
        quat = d.qpos[3:7].copy()
        omega = d.qvel[3:6].copy()
        # omega = self.data.xmat[self.base_index].reshape(3, 3).T @ self.data.cvel[self.base_index][3:6]
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        L = min(len(config["default_angles"]), n_joints)
        padded_defaults[:L] = config["default_angles"][:L]

        qj_scaled = (qj - padded_defaults) * config["dof_pos_scale"]
        dqj_scaled = dqj * config["dof_vel_scale"]
        gravity_orientation = self.get_gravity_orientation(quat)
        omega_scaled = omega * config["ang_vel_scale"]

        torso_quat = self.data.xquat[self.torso_index]
        torso_omega = (
            self.data.xmat[self.torso_index].reshape(3, 3).T @ self.data.cvel[self.torso_index][3:6]
        )
        torso_omega_scaled = torso_omega * config["ang_vel_scale"]
        torso_gravity_orientation = self.get_gravity_orientation(torso_quat)

        single_obs_dim = 86
        single_obs = np.zeros(single_obs_dim, dtype=np.float32)
        single_obs[0:7] = command[:7]
        single_obs[7:10] = omega_scaled
        single_obs[10:13] = gravity_orientation
        # single_obs[14:17] = 0.#torso_omega_scaled
        # single_obs[17:20] = 0.#torso_gravity_orientation
        single_obs[13 : 13 + n_joints] = qj_scaled
        single_obs[13 + n_joints : 13 + 2 * n_joints] = dqj_scaled
        single_obs[13 + 2 * n_joints : 13 + 2 * n_joints + 15] = action
        # single_obs[20+2*n_joints+15:20+2*n_joints+15+2] = self.clock_inputs.cpu().numpy().reshape(2)

        return single_obs, single_obs_dim

    def load_onnx_policy(self, path):
        model = ort.InferenceSession(path)

        def run_inference(input_tensor):
            ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
            ort_outs = model.run(None, ort_inputs)
            return torch.tensor(ort_outs[0], device="cuda:0")

        return run_inference

    def run(self):

        self.counter = 0

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < self.config["simulation_duration"]:
                step_start = time.time()

                leg_tau = self.pd_control(
                    self.target_dof_pos,
                    self.data.qpos[7 : 7 + self.config["num_actions"]],
                    self.config["kps"],
                    np.zeros_like(self.config["kps"]),
                    self.data.qvel[6 : 6 + self.config["num_actions"]],
                    self.config["kds"],
                )
                self.data.ctrl[: self.config["num_actions"]] = leg_tau

                if self.n_joints > self.config["num_actions"]:
                    arm_tau = self.pd_control(
                        np.zeros(self.n_joints - self.config["num_actions"], dtype=np.float32),
                        self.data.qpos[7 + self.config["num_actions"] : 7 + self.n_joints],
                        np.full(self.n_joints - self.config["num_actions"], 100.0),
                        np.zeros(self.n_joints - self.config["num_actions"]),
                        self.data.qvel[6 + self.config["num_actions"] : 6 + self.n_joints],
                        np.full(self.n_joints - self.config["num_actions"], 0.5),
                    )
                    self.data.ctrl[self.config["num_actions"] :] = arm_tau

                mujoco.mj_step(self.model, self.data)

                self.counter += 1
                if self.counter % self.config["control_decimation"] == 0:
                    with self.cmd_lock:
                        current_cmd = self.control_dict

                    single_obs, _ = self.compute_observation(
                        self.data, self.config, self.action, current_cmd, self.n_joints
                    )
                    self.obs_history.append(single_obs)

                    for i, hist_obs in enumerate(self.obs_history):
                        self.obs[i * self.single_obs_dim : (i + 1) * self.single_obs_dim] = hist_obs

                    obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
                    if (np.linalg.norm(np.array(current_cmd["loco_cmd"]))) <= 0.05:
                        self.action = self.policy(obs_tensor).cpu().detach().numpy().squeeze()
                    else:
                        self.action = self.walk_policy(obs_tensor).cpu().detach().numpy().squeeze()
                    self.target_dof_pos = (
                        self.action * self.config["action_scale"] + self.config["default_angles"]
                    )

                viewer.sync()
                # time.sleep(max(0, self.model.opt.timestep - (time.time() - step_start)))


if __name__ == "__main__":
    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "robots", "g1"
    )
    controller = GearWbcController(CONFIG_PATH)
    controller.run()
