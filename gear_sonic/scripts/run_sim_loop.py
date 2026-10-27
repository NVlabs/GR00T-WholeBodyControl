"""Entry point for running a MuJoCo simulation loop with the G1 robot model.

Parses a YAML-based WBC config via tyro CLI, instantiates the G1 robot model,
and launches the simulator (optionally with offscreen image publishing).
"""

from typing import Dict

import tyro

from gear_sonic.data.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from gear_sonic.data.robot_model.robot_model import RobotModel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel

ArgsConfig = SimLoopConfig


class SimWrapper:
    def __init__(self, robot_model: RobotModel, env_name: str, config: Dict[str, any], **kwargs):
        self.robot_model = robot_model
        self.config = config

        # BaseSimulator owns DDS initialization for Inspire, as in its direct
        # launch path. Initializing here too creates the same domain twice.
        # Keep the existing Dex3 launch path unchanged.
        if self.config.get("HAND_TYPE", "dex3") != "inspire":
            init_channel(config=self.config)

        # Create simulator using factory
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name=env_name,
            **kwargs,
        )


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    # NOTE: we will override the interface to local if it is not specified
    wbc_config["ENV_NAME"] = config.env_name

    if config.enable_image_publish:
        assert config.enable_offscreen, (
            "enable_offscreen must be True when enable_image_publish is True"
        )

    hand_options = {}
    if config.hand == "inspire":
        from gear_sonic.data.robot_model.instantiation.g1_inspire import (
            instantiate_g1_rh56dfx_robot_model,
        )
        from gear_sonic.utils.hand_control.inspire.config import DEFAULT_MAPPING_PATH
        from gear_sonic.utils.mujoco_sim.inspire.environment import configure_simulation
        from gear_sonic.utils.mujoco_sim.inspire.gateway import InspireSimGateway

        hand_config = config.hand_config or DEFAULT_MAPPING_PATH
        wbc_config = configure_simulation(wbc_config, hand_config)
        robot_model = instantiate_g1_rh56dfx_robot_model(config_path=hand_config)
        hand_options["hand_controller_factory"] = lambda model, data: InspireSimGateway(
            model, data, hand_config
        )
    else:
        robot_model = instantiate_g1_robot_model()

    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
        **hand_options,
    )
    # Start simulator as independent process
    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
