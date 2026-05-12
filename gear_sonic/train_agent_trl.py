#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fix sys.path: when running as `python gear_sonic/train_agent_trl.py`, Python adds
# gear_sonic/ to sys.path[0], causing `from trl import ...` to resolve to our local
# gear_sonic/trl/ instead of the HuggingFace trl package. Replace with repo root.
import sys
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import isaaclab  # noqa: F401
except ImportError:
    print(
        "\n"
        "ERROR: Isaac Lab is required for training but not installed.\n"
        "\n"
        "Isaac Lab is not a pip dependency — it must be installed separately.\n"
        "Follow the official guide:\n"
        "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html\n"
        "\n"
        "After installing, activate the Isaac Lab conda/venv environment\n"
        "before running this script.\n"
    )
    sys.exit(1)

import glob
import logging
import os
from pathlib import Path
import re
import sys

from filelock import FileLock
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import wandb
import yaml

from gear_sonic.trl.utils.common import (
    custom_instantiate,
    get_filtered_state_dict,
    materialize_lazy_params,
    wandb_run_exists,
)
from gear_sonic.utils.common import seeding
from gear_sonic.utils.config_utils import register_rl_resolvers
from gear_sonic.utils.obs_utils import get_group_term_obs_shape

register_rl_resolvers()


def resume_training(config):
    if config.get("checkpoint", None) is not None:
        last_existing_checkpoint = config.checkpoint
    elif config.get("experiment_dir", None) is not None:
        last_existing_checkpoint = os.path.join(config.experiment_dir, "last.pt")
    else:
        # Use experiment_dir to find the checkpoint, rather than reconstructing
        # from config.project_name which can differ from the actual filesystem path.
        experiment_dir_base = re.sub(r"-\d{8}_\d{6}$", "", config.experiment_dir)
        checkpoints = sorted(glob.glob(os.path.join(f"{experiment_dir_base}-*", "last.pt")))
        if not checkpoints:
            print(f"No checkpoint found matching {experiment_dir_base}-*/last.pt, starting fresh")
            return
        last_existing_checkpoint = checkpoints[-1]
    experiment_dir = os.path.dirname(last_existing_checkpoint)
    config.experiment_dir = experiment_dir
    config.checkpoint = last_existing_checkpoint
    print(f"Resuming training from {last_existing_checkpoint}")


def resume_checkpoint(config):
    config.checkpoint = config.checkpoint


def create_manager_env(config, device, args_cli):

    # import wandb

    from isaaclab.envs import (
        ManagerBasedRLEnv,
    )

    from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper

    env_instance_cfg = custom_instantiate(config.manager_env)

    # Iteratively check the difference in attribute of env_instance_cfg1 and env_instance_cfg, print out the difference
    def compare_attrs(obj1, obj2, prefix=""):
        # Only compare attributes that do not start with '__' and are not methods
        attrs1 = set(dir(obj1))
        attrs2 = set(dir(obj2))
        common_attrs = attrs1 & attrs2
        for attr in sorted(common_attrs):
            if (
                attr.startswith("__")
                or callable(getattr(obj1, attr))
                or callable(getattr(obj2, attr))
            ):
                continue
            try:
                val1 = getattr(obj1, attr)
                val2 = getattr(obj2, attr)
            except Exception:
                continue
            # Recursively compare if both are objects with __dict__ or are dicts
            if isinstance(val1, dict | DictConfig) and isinstance(val2, dict | DictConfig):
                compare_attrs(val1, val2, prefix + attr + ".")
            elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
                compare_attrs(val1, val2, prefix + attr + ".")
            else:
                if isinstance(val1, list):
                    val1 = tuple(val1)
                if isinstance(val2, list):
                    val2 = tuple(val2)
                if val1 != val2:
                    print(
                        f"\nDifference found at '{prefix}{attr}':\n"
                        f"  - env_instance_cfg1: {val1!r}\n"
                        f"  - env_instance_cfg : {val2!r}\n"
                    )

    env_instance_cfg.seed = config.seed
    env_instance_cfg.sim.device = device
    env_instance_cfg.config["headless"] = args_cli.headless
    env = ManagerBasedRLEnv(
        cfg=env_instance_cfg, render_mode="rgb_array" if not args_cli.headless else None
    )

    env = ManagerEnvWrapper(env, env_instance_cfg.config)
    return env


@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    simulator_type = "IsaacSim"
    env_config = config.manager_env
    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    # Setup model components
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))

    if config.get("resume", False):
        resume_training(config)
    elif config.get("checkpoint", None) is not None:
        resume_checkpoint(config)

    config.algo.trl.output_dir = str(Path(config.experiment_dir))

    script_args, training_args, model_args = parser.parse_dict(config.algo.trl)

    # Add exp_name from main config to training_args
    training_args.exp_name = config.experiment_name

    from datetime import timedelta

    from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
    import torch  # noqa: E402

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=6000))
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs, kwargs],
    )

    device = str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    seeding(config.seed)

    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path))
        config.wandb.wandb_id = meta["wandb_run"]
        print(f"resume wandb from run: {config.wandb.wandb_id}")

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    if config.use_wandb and accelerator.is_main_process:
        project_name = f"{config.project_name}"
        run_name = config.experiment_dir.replace(f"{config.base_dir}/{project_name}/", "")
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        wandb_group = None if config.wandb.wandb_id is not None else config.wandb.wandb_group
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb.init(
            project=project_name,
            entity=config.wandb.wandb_entity,
            name=run_name,
            sync_tensorboard=True,
            config=unresolved_conf,
            dir=wandb_dir,
            id=config.wandb.wandb_id,
            group=wandb_group,
            resume="allow",
        )

    # Setup simulator similar to train_agent.py

    if simulator_type == "IsaacSim":
        try:
            with open("./rl/simulator/isaacsim/.isaacsim_version", encoding="utf-8") as f:
                DEFAULT_ISAACSIM_VERSION = f.read().strip()
        except FileNotFoundError:
            DEFAULT_ISAACSIM_VERSION = "4.5"

        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.app import AppLauncher
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
            from omni.isaac.lab.app import AppLauncher  # 4.2

            # from isaaclab.app import AppLauncher # not working
            # from omni.isaac.lab.app import AppLauncher

        import argparse

        parser = argparse.ArgumentParser(description="Train an RL agent with TRL.")
        AppLauncher.add_app_launcher_args(parser)

        ######################################################### ZL: fix isaacsim 4.5 rendering #########################################################
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = env_config.config.env_spacing  # config.env_spacing
        args_cli.output_dir = config.output_dir
        # Enable cameras if enable_cameras, render_results, render_ego, or overview_camera is True
        args_cli.enable_cameras = (
            env_config.config.get("enable_cameras", False)
            or env_config.config.get("render_results", False)
            or env_config.config.get("render_ego", False)
            or env_config.config.get("overview_camera", False)
        )
        args_cli.headless = config.headless
        args_cli.multi_gpu = config.multi_gpu
        args_cli.distributed = config.multi_gpu
        args_cli.device = device

        # Base kit args (quiet logs)
        args_cli.kit_args = (
            "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
        )

        # AppLauncher can't handle multiple processes creating it at the same time so we need a lock
        _lock_path = "/tmp/isaaclab_app_launcher.lock"
        _local_rank = int(os.environ.get("LOCAL_RANK", 0))
        with FileLock(_lock_path):
            app_launcher = AppLauncher(args_cli)

        simulation_app = app_launcher.app

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    from gear_sonic.utils.logging import HydraLoggerBridge

    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    # Setup wandb if enabled
    os.chdir(hydra.utils.get_original_cwd())

    # Save config and meta BEFORE env creation so eval jobs can postprocess
    # checkpoint configs even if training crashes during env init.
    experiment_save_dir = Path(config.experiment_dir)
    if accelerator.is_main_process:
        experiment_save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving config file to {experiment_save_dir}")
        with open(experiment_save_dir / "config.yaml", "w") as file:
            OmegaConf.save(unresolved_conf, file)
        meta = {"wandb_run": wandb.run.id if wandb_run_exists() else None}
        meta["max_train_steps"] = config.algo.config.num_learning_iterations
        yaml.safe_dump(meta, open(meta_path, "w"))
        print("saved meta:", meta)

    # Initialize environment
    env_config.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    env_config.config.experiment_dir = str(Path(config.experiment_dir))

    env = create_manager_env(config, device, args_cli)
    if config.get("replay", False):
        _save_video_path = config.get("replay_save_video", None)
        env.run_replay(
            start_time_step=-1,
            loop=config.get("replay_loop_num", True),
            save_video_path=_save_video_path,
            grid_spacing=config.get("replay_grid_spacing", 2.0),
        )
        os._exit(0)
    if config.get("vplanner_replay", False):
        vplanner_checkpoint = config.get("vplanner_checkpoint", None)
        if vplanner_checkpoint is None:
            raise ValueError("vplanner_checkpoint must be specified for vplanner_replay")
        env.run_vplanner_replay(
            checkpoint_path=vplanner_checkpoint,
            max_frames=config.get("vplanner_max_frames", 500),
            replan_interval=config.get("vplanner_replan_interval", 0),
            speed=config.get("vplanner_speed", 1.0),
            loop=config.get("vplanner_loop", True),
            save_images=config.get("vplanner_save_images", False),
            output_dir=config.get("vplanner_output_dir", None),
            dof_noise=config.get("vplanner_dof_noise", 0.0),
            dof_vel_noise=config.get("vplanner_dof_vel_noise", 0.0),
            quat_noise=config.get("vplanner_quat_noise", 0.0),
        )
        os._exit(0)

    ref_model = None
    value_model = None
    disc_model = None
    # import ipdb; ipdb.set_trace()

    if config.algo.config.get("use_new_actor_critic", False):
        module_dim_dict = getattr(config.algo.config, "module_dim", {})
        policy_backbone_kwargs = {}
        critic_backbone_kwargs = {}
        env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
        env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
        env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
            "policy"
        ].shape[-1]
        env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
            "critic"
        ].shape[-1]
        example_obs = env.reset(flatten_dict_obs=False)
        for key in env.env.observation_space:
            if key not in ["policy", "critic"]:
                group_obs_dims, group_obs_names, group_obs_total_dim = get_group_term_obs_shape(
                    example_obs, key
                )
                env.config["obs"]["group_obs_dims"][key] = group_obs_dims
                env.config["obs"]["group_obs_names"][key] = group_obs_names
                env.config["obs"]["obs_dims"][key] = group_obs_total_dim
                env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim
        if config.manager_env.config.get("meta_action_dim", None) is not None:
            env.config["robot"]["actions_dim"] = config.manager_env.config.meta_action_dim
        else:
            env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

        policy = custom_instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            backbone_kwargs=policy_backbone_kwargs,
            _resolve=False,
        ).to(device)

        if getattr(config.algo.config, "use_dagger", False):
            # Get teacher input key from config or default to "teacher"
            teacher_input_key = config.algo.config.get("teacher_input_key", "teacher")
            ref_model = custom_instantiate(
                config.algo.config.teacher_actor,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _resolve=False,
                input_key=teacher_input_key,
            ).to(device)
        if not getattr(config.algo.config, "distill_only", False):
            value_model = custom_instantiate(
                config.algo.config.critic,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                backbone_kwargs=critic_backbone_kwargs,
                _resolve=False,
            ).to(device)
        if config.algo.config.get("use_amp", False):
            disc_model = custom_instantiate(
                config.algo.config.disc,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _resolve=False,
            ).to(device)
    else:
        raise ValueError("No longer supported")

    materialize_lazy_params(policy, env)

    if config.algo.config.get("pretrained_model", None) is not None:
        pretrained_cfg = config.algo.config.pretrained_model
        sd_key = pretrained_cfg.get("state_dict_key", "state_dict")
        strict = pretrained_cfg.get("strict", True)
        state_dict = torch.load(pretrained_cfg.path, map_location=device, weights_only=False)[
            sd_key
        ]
        for (
            module_name,
            state_dict_key,
        ) in pretrained_cfg.module_mapping.items():
            module = eval(module_name)
            filtered_state_dict = get_filtered_state_dict(state_dict, state_dict_key)
            missing, unexpected = module.load_state_dict(filtered_state_dict, strict=strict)
            if missing:
                logger.info(f"Pretrained loading '{module_name}': missing keys: {missing}")
            if unexpected:
                logger.info(f"Pretrained loading '{module_name}': unexpected keys: {unexpected}")

    accelerator.wait_for_everyone()

    callbacks = []
    for callback in config.callbacks.values():
        callbacks.append(instantiate(callback))

    ################
    # Training
    ################
    trainer = custom_instantiate(
        config.trainer,
        args=training_args,
        config=config.algo.config,
        env=env,
        model=policy,
        disc_model=disc_model,
        value_model=value_model,
        ref_model=ref_model,
        use_ref_model=getattr(config.algo.config, "use_dagger", False),
        train_dataset=None,
        eval_dataset=None,
        callbacks=callbacks,
        checkpoint=config.checkpoint,
        resume=config.get("resume", False),
        local_seed=config.seed,
        log_dir=experiment_save_dir,
        accelerator=accelerator,
        _resolve=False,
    )

    # Training loop
    trainer.train()

    if simulator_type == "IsaacSim":
        os._exit(0)


if __name__ == "__main__":

    main()
