"""Factory for creating and launching MuJoCo simulators with Unitree SDK channel setup."""

import time
from threading import Lock
from typing import Any, Dict

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator

_CHANNEL_INIT_LOCK = Lock()
_CHANNEL_INITIALIZED = False


def init_channel(config: Dict[str, Any] | None = None) -> None:
    """
    Initialize the communication channel for simulator/robot communication.

    This is intentionally guarded so the same process does not reinitialize
    CycloneDDS repeatedly when the simulator and the environment both try to
    create a channel.

    Args:
        config: Configuration dictionary containing DOMAIN_ID and optionally INTERFACE.
            If not provided, fall back to the default loopback-safe config.
    """
    global _CHANNEL_INITIALIZED

    if _CHANNEL_INITIALIZED:
        return

    with _CHANNEL_INIT_LOCK:
        if _CHANNEL_INITIALIZED:
            return

        domain_id = 0
        interface = None

        if config is not None:
            if isinstance(config, dict):
                domain_id = config.get("DOMAIN_ID", 0)
                interface = config.get("INTERFACE", None)
            else:
                domain_id = getattr(config, "DOMAIN_ID", 0)
                interface = getattr(config, "INTERFACE", None)

        if interface:
            ChannelFactoryInitialize(domain_id, interface)
        else:
            ChannelFactoryInitialize(domain_id)

        _CHANNEL_INITIALIZED = True


class SimulatorFactory:
    """Factory class for creating different types of simulators."""

    @staticmethod
    def create_simulator(config: Dict[str, Any], env_name: str = "default", **kwargs):
        """
        Create a simulator based on the configuration.

        Args:
            config: Configuration dictionary containing SIMULATOR type
            env_name: Environment name
            **kwargs: Additional keyword arguments for specific simulators
        """
        simulator_type = config.get("SIMULATOR", "mujoco")
        if simulator_type == "mujoco":
            return SimulatorFactory._create_mujoco_simulator(config, env_name, **kwargs)
        else:
            print(
                f"Warning: Invalid simulator type: {simulator_type}. "
                "If you are using run_sim_loop, please ignore this warning."
            )
            return None

    @staticmethod
    def _create_mujoco_simulator(config: Dict[str, Any], env_name: str = "default", **kwargs):
        """Create a MuJoCo simulator instance."""
        return BaseSimulator(
            onscreen=kwargs.pop("onscreen", True),
            offscreen=kwargs.pop("offscreen", False),
            enable_image_publish=kwargs.get("enable_image_publish", False),
            config=config,
            env_name=env_name,
            redis_client=kwargs.get("redis_client", None),
        )

    @staticmethod
    def start_simulator(
        simulator,
        as_thread: bool = True,
        enable_image_publish: bool = False,
        mp_start_method: str = "spawn",
        camera_port: int = 5555,
    ):
        """
        Start the simulator either as a thread or as a separate process.

        Args:
            simulator: The simulator instance to start
            as_thread: If True, start as thread; if False, start as subprocess
            enable_image_publish: If True and not as_thread, start image publishing
            mp_start_method: Multiprocessing start method
            camera_port: Camera port for image publishing
        """

        if as_thread:
            simulator.start_as_thread()
        else:
            try:
                if enable_image_publish:
                    simulator.start_image_publish_subprocess(
                        start_method=mp_start_method,
                        camera_port=camera_port,
                    )
                    time.sleep(1)
                simulator.start()
            except KeyboardInterrupt:
                print("+++++Simulator interrupted by user.")
            except Exception as e:
                print(f"++++error in simulator: {e} ++++")
            finally:
                print("++++closing simulator ++++")
                simulator.close()

        # Allow simulator to initialize
        time.sleep(1)
