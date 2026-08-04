from abc import abstractmethod
import threading
import time
from typing import Any, Dict, Tuple

import mujoco
import numpy as np
import rclpy

from decoupled_wbc.control.envs.g1.sim.image_publish_utils import ImagePublishProcess
from decoupled_wbc.control.envs.robocasa.utils.robocasa_env import (
    Gr00tLocomanipRoboCasaEnv,
)  # noqa: F401
from decoupled_wbc.control.robot_model.robot_model import RobotModel
from decoupled_wbc.control.utils.keyboard_dispatcher import KeyboardListenerSubscriber


class RoboCasaEnvServer:
    """
    This class is responsible for running the simulation environment loop in a separate thread.
    It communicates with the main thread via the `publish_obs` and `get_action` methods through `channel_bridge`.
    It will also handle the viewer sync when `onscreen` is True.
    """

    def __init__(
        self,
        env_name: str,
        robot_name: str,
        robot_model: RobotModel,
        env_kwargs: Dict[str, Any],
        **kwargs,
    ):
        # initialize environment
        if env_kwargs.get("onscreen", False):
            env_kwargs["onscreen"] = False
            self.onscreen = True  # onscreen render in the main thread
            self.render_camera = env_kwargs.get("render_camera", None)
        else:
            self.onscreen = False
        self.env_name = env_name
        self.env = Gr00tLocomanipRoboCasaEnv(env_name, robot_name, robot_model, **env_kwargs)
        self.init_caches()
        self.cache_lock = threading.Lock()

        # initialize channel
        self.init_channel()

        # initialize ROS2 node
        if not rclpy.ok():
            rclpy.init()
            self.node = rclpy.create_node("sim_robocasa")
            self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
            self.thread.start()
        else:
            self.thread = None
            executor = rclpy.get_global_executor()
            self.node = executor.get_nodes()[0]  # will only take the first node

        self.control_freq = env_kwargs.get("control_freq", 1 / 0.02)
        self.sim_freq = kwargs.get("sim_freq", 1 / 0.005)
        self.control_rate = self.node.create_rate(self.control_freq)

        self.running = False
        self.sim_thread = None
        self.sync_lock = threading.Lock()

        self.sync_mode = kwargs.get("sync_mode", False)
        self.steps_per_action = kwargs.get("steps_per_action", 1)

        self.image_dt = kwargs.get("image_dt", 0.04)
        self.image_publish_process = None
        self.viewer_freq = kwargs.get("viewer_freq", 1 / 0.02)
        self.viewer = None

        self.verbose = kwargs.get("verbose", True)

        # Initialize keyboard listener for env reset
        self.keyboard_listener = KeyboardListenerSubscriber()

        self.reset()

    @property
    def base_env(self):
        return self.env.env

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        """Initialize image publishing subprocess if cameras are configured"""
        if len(self.env.camera_names) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return

        # Build camera configs from env camera settings
        camera_configs = {}
        for env_cam_name in self.env.camera_names:
            camera_config = self.env.camera_key_mapper.get_camera_config(env_cam_name)
            mapped_cam_name, cam_width, cam_height = camera_config
            camera_configs[mapped_cam_name] = {"height": cam_height, "width": cam_width}

        self.image_publish_process = ImagePublishProcess(
            camera_configs=camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.verbose,
        )

        self.image_publish_process.start_process()

    def update_render_caches(self, obs: Dict[str, Any]):
        """Update render cache and shared memory for subprocess"""
        if self.image_publish_process is None:
            return

        # Extract image observations from obs dict
        render_caches = {
            k: v for k, v in obs.items() if k.endswith("_image") and isinstance(v, np.ndarray)
        }

        # Update shared memory if image publishing process is available
        if render_caches:
            self.image_publish_process.update_shared_memory(render_caches)

    def init_caches(self):
        self.caches = {
            "obs": None,
            "reward": None,
            "terminated": None,
            "truncated": None,
            "info": None,
        }

    def reset(self, **kwargs):
        if self.viewer is not None:
            self.viewer.close()

        obs, info = self.env.reset(**kwargs)
        self.caches["obs"] = obs
        self.caches["reward"] = 0
        self.caches["terminated"] = False
        self.caches["truncated"] = False
        self.caches["info"] = info

        # initialize viewer
        if self.onscreen:
            self.viewer = mujoco.viewer.launch_passive(
                self.base_env.sim.model._model,
                self.base_env.sim.data._data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.opt.geomgroup[0] = 0  # disable collision visualization
            if self.render_camera is not None:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = self.base_env.sim.model._model.cam(
                    self.render_camera
                ).id

        # self.episode_state.reset_state()
        return obs, info

    @abstractmethod
    def init_channel(self):
        raise NotImplementedError("init_channel must be implemented by the subclass")

    @abstractmethod
    def publish_obs(self):
        raise NotImplementedError("publish_obs must be implemented by the subclass")

    @abstractmethod
    def get_action(self) -> Tuple[Dict[str, Any], bool, bool]:
        raise NotImplementedError("get_action must be implemented by the subclass")

    def start_as_thread(self):
        """Start the simulation thread"""
        if self.sim_thread is not None and self.sim_thread.is_alive():
            return

        self.sim_thread = threading.Thread(target=self.start)
        self.sim_thread.daemon = True
        self.sim_thread.start()

    def set_sync_mode(self, sync_mode: bool, steps_per_action: int = 4):
        """Set the sync mode of the environment server"""
        with self.sync_lock:
            self.sync_mode = sync_mode
            self.steps_per_action = steps_per_action

    def _check_keyboard_input(self):
        """Check for keyboard input and handle state transitions"""
        key = self.keyboard_listener.read_msg()
        if key == "k":
            print("\033[1;32m[Sim env]\033[0m Resetting sim environment")
            self.reset()

    def start(self):
        """Function executed by the simulation thread"""
        iter_idx = 0
        steps_per_cur_action = 0
        t_start = time.monotonic()

        self.running = True

        while self.running:
            # Check keyboard input for state transitions
            self._check_keyboard_input()

            # Publish observations and get new action
            self.publish_obs()
            action, ready, is_new_action = self.get_action()
            # ready is True if the action is received from the control loop
            # is_new_action is True if the action is new (not the same as the previous action)
            with self.sync_lock:
                sync_mode = self.sync_mode
                max_steps_per_action = self.steps_per_action

            # Process action if ready and within step limits
            action_should_apply = ready and (
                (not sync_mode) or steps_per_cur_action < max_steps_per_action
            )
            if action_should_apply:
                obs, reward, terminated, truncated, info = self.env.step(action)
                with self.cache_lock:
                    self.caches["obs"] = obs
                    self.caches["reward"] = reward
                    self.caches["terminated"] = terminated
                    self.caches["truncated"] = truncated
                    self.caches["info"] = info

                if reward == 1.0 and iter_idx % 50 == 0:
                    print("\033[92mTask successful. Can save data now.\033[0m")

                iter_idx += 1
                steps_per_cur_action += 1
                if self.verbose and sync_mode:
                    print("steps_per_cur_action: ", steps_per_cur_action)

            # Update render caches at image publishing rate
            if action_should_apply and iter_idx % int(self.image_dt * self.control_freq) == 0:
                with self.cache_lock:
                    obs_copy = self.caches["obs"].copy()
                self.update_render_caches(obs_copy)

            # Reset step counter for new actions
            if is_new_action:
                steps_per_cur_action = 0

            # Update viewer at specified frequency
            if self.onscreen and iter_idx % (self.control_freq / self.viewer_freq) == 0:
                self.viewer.sync()

            # Check if we're meeting the desired control frequency
            if iter_idx % 100 == 0:
                end_time = time.monotonic()
                if self.verbose:
                    print(
                        f"sim FPS: {100.0 / (end_time - t_start) * (self.sim_freq / self.control_freq)}"
                    )
                if (end_time - t_start) > ((110.0 / self.control_freq)):  # for tolerance
                    print(
                        f"Warning: Sim runs at "
                        "{100.0/(end_time - t_start) * (self.sim_freq / self.control_freq):.1f}Hz, "
                        f"but should run at {self.sim_freq:.1f}Hz"
                    )
                t_start = end_time

            # reset obj pos every 200 steps
            if iter_idx % 200 == 0:
                if hasattr(self.base_env, "reset_obj_pos"):
                    self.base_env.reset_obj_pos()

            self.control_rate.sleep()

    def get_privileged_obs(self):
        """Get privileged observation. Should be implemented by subclasses."""
        obs = {}
        with self.cache_lock:
            if hasattr(self.base_env, "get_privileged_obs_keys"):
                for key in self.base_env.get_privileged_obs_keys():
                    obs[key] = self.caches["obs"][key]

            for key in self.caches["obs"].keys():
                if key.endswith("_image"):
                    obs[key] = self.caches["obs"][key]

        return obs

    def stop(self):
        """Stop the simulation thread"""
        self.running = False
        if self.sim_thread is not None:
            self.sim_thread.join(timeout=1.0)  # Wait for thread to finish with timeout
            self.sim_thread = None

    def close(self):
        self.stop()
        if self.image_publish_process is not None:
            self.image_publish_process.stop()
        if self.onscreen:
            self.viewer.close()
        self.env.close()

    def get_reward(self):
        return self.base_env.reward()
