from typing import Any, Dict, Tuple

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from decoupled_wbc.control.envs.g1.sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from decoupled_wbc.control.envs.robocasa.async_env_server import RoboCasaEnvServer
from decoupled_wbc.control.robot_model.instantiation import get_robot_type_and_model


class RoboCasaG1EnvServer(RoboCasaEnvServer):
    def __init__(
        self, env_name: str, wbc_config: Dict[str, Any], env_kwargs: Dict[str, Any], **kwargs
    ):
        if UnitreeSdk2Bridge is None:
            raise ImportError("UnitreeSdk2Bridge is required for RoboCasaG1EnvServer")
        self.wbc_config = wbc_config
        _, robot_model = get_robot_type_and_model(
            "G1",
            enable_waist_ik=wbc_config["enable_waist"],
        )
        if env_kwargs.get("camera_names", None) is None:
            env_kwargs["camera_names"] = [
                "robot0_oak_egoview",
                "robot0_oak_left_monoview",
                "robot0_oak_right_monoview",
                "robot0_rs_tppview",
            ]
        if env_kwargs.get("render_camera", None) is None:
            if env_kwargs.get("renderer", "mjviewer") == "mjviewer":
                env_kwargs["render_camera"] = "robot0_oak_egoview"
            else:
                env_kwargs["render_camera"] = [
                    "robot0_oak_egoview",
                    "robot0_rs_tppview",
                ]

        super().__init__(env_name, "G1", robot_model, env_kwargs=env_kwargs, **kwargs)

    def init_channel(self):

        try:
            if self.wbc_config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.wbc_config["DOMAIN_ID"], self.wbc_config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.wbc_config["DOMAIN_ID"])
        except Exception:
            # If it fails because it's already initialized, that's okay
            pass

        self.channel_bridge = UnitreeSdk2Bridge(config=self.wbc_config)

    def publish_obs(self):
        # with self.cache_lock:
        obs = self.caches["obs"]
        self.channel_bridge.PublishLowState(obs)

    def get_action(self) -> Tuple[Dict[str, Any], bool, bool]:
        q, ready, is_new_action = self.channel_bridge.GetAction()
        return {"q": q}, ready, is_new_action

    def reset(self):
        super().reset()
        self.channel_bridge.reset()
