"""PPO trainer adapted from HuggingFace TRL for humanoid whole-body control."""

import gc  # noqa: F401
import math
import os

import accelerate
from accelerate import utils as accelerate_utils
import numpy as np
import torch
from torch import nn
from transformers import utils as transformers_utils
from transformers.trainer import *  # noqa: F403
from trl import models
from trl.experimental.ppo import ppo_trainer
from trl.models import utils as models_utils
from trl.trainer import utils as trainer_utils
from trl.trainer.ppo_trainer import *  # noqa: F403
import wandb

# Conditional peft imports
if accelerate_utils.is_peft_available():
    import peft

# Compatibility shim for loading old checkpoints saved with TRL < 0.28.0
# These were moved in TRL 0.28.0, but old checkpoints reference the old paths
import sys

import trl.trainer.utils

trl.trainer.utils.OnlineTrainerState = ppo_trainer.OnlineTrainerState
trl.trainer.utils.exact_div = ppo_trainer.exact_div
sys.modules["trl.trainer.utils"].OnlineTrainerState = ppo_trainer.OnlineTrainerState
sys.modules["trl.trainer.utils"].exact_div = ppo_trainer.exact_div

# Constants and utilities that may not be exported from trl
INVALID_LOGPROB = 1.0  # Invalid log probability marker


def masked_mean(values, mask, axis=None):
    """Compute mean of values where mask is True."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis).clamp(min=1)
    return (values * mask).sum() / mask.sum().clamp(min=1)


from collections import deque  # noqa: E402

import pandas as pd  # noqa: E402, F401
from rich import (
    console,
    live,
    panel,
)
from tqdm import tqdm  # noqa: E402, F401

from gear_sonic.trl.callbacks import hv_callback_handler  # noqa: E402
from gear_sonic.trl.modules import data_utils  # noqa: E402
from gear_sonic.trl.utils import (
    common,
    rl,
    scheduler,
)
from gear_sonic.utils import average_meters  # noqa: E402

console_ = console.Console()
import time  # noqa: E402


class PolicyAndValueWrapper(nn.Module):
    """Wrap policy, value, and optional discriminator models into a single nn.Module.

    This wrapper enables a single ``forward`` call to dispatch to multiple model
    components (policy, value, discriminator) in one DDP-safe pass, which is
    required because calling ``forward`` on a DDP module more than once per
    backward triggers gradient synchronization errors.
    """

    def __init__(self, policy, value_model, **kwargs) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        if "disc_model" in kwargs:
            self.disc_model = kwargs["disc_model"]

    @property
    def is_gradient_checkpointing(self):
        """
        Whether gradient checkpointing is enabled for this model.
        """  # noqa: D200, D212
        if hasattr(self.policy, "is_gradient_checkpointing"):
            return self.policy.is_gradient_checkpointing
        return False

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.policy, "gradient_checkpointing_enable"):
            self.policy.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        if hasattr(self.policy, "gradient_checkpointing_disable"):
            self.policy.gradient_checkpointing_disable(**kwargs)

    def set_mode(self, mode):
        """Set the operating mode on the policy (e.g. ``"train"``, ``"eval"``, ``"train_rollout"``)."""
        if hasattr(self.policy, "mode"):
            self.policy.mode = mode

    def transform_train(self):
        """Enable training-time transforms (e.g. image augmentation) on the policy."""
        if hasattr(self.policy, "transform_train"):
            self.policy.transform_train()

    def transform_eval(self):
        """Disable training-time transforms on the policy for clean rollouts."""
        if hasattr(self.policy, "transform_eval"):
            self.policy.transform_eval()

    def forward(self, modes, input_kwargs):
        """Run forward passes for the requested component modes in a single call.

        Args:
            modes: List of mode strings to evaluate (e.g. ``["policy", "value"]``).
            input_kwargs: Mapping from mode name to keyword arguments for that
                component's forward pass.

        Returns:
            Dict mapping each mode name to its forward-pass output.
        """
        results = {}
        for mode in modes:
            results[mode] = self.forward_component(mode, **input_kwargs[mode])
        return results

    def forward_component(self, mode, actions=None, **kwargs):
        """Dispatch a forward pass to a single component identified by *mode*.

        Supported modes:
            - ``"policy"`` -- standard policy forward with log-prob computation.
            - ``"policy_distill"`` -- policy forward for distillation (no log-probs).
            - ``"policy_distill_ppo"`` -- distillation forward that also computes
              log-probs for PPO training.
            - ``"policy_w_and_wo_imgaug"`` -- two policy forwards (with and without
              image augmentation) for image-augmentation BC loss.
            - ``"policy_deterministic"`` -- deterministic (mean) action only.
            - ``"vae_policy_deterministic"`` -- VAE policy with prior evaluation.
            - ``"value"`` -- value model evaluation.
            - ``"eval_disc"`` / ``"train_disc"`` -- discriminator evaluation / training.

        Args:
            mode: Component mode string.
            actions: Actions tensor for log-prob computation, ``(num_envs, num_steps, act_dim)``.
            **kwargs: Forwarded to the underlying model component.

        Returns:
            Dict of component outputs (keys vary by mode).
        """
        if mode == "policy":
            self.policy.act(**kwargs)
            log_probs = self.policy.get_actions_log_prob(actions=actions)
            results = {
                "logprobs": log_probs,
                "action_mean": self.policy.action_mean,
                "action_std": self.policy.action_std,
                "entropy": self.policy.entropy,
            }
            if self.policy.has_aux_loss:
                results["aux_losses"] = self.policy.aux_losses
                results["aux_loss_coef"] = self.policy.aux_loss_coef
        elif mode == "policy_distill":
            results = self.policy.act(**kwargs)
        elif mode == "policy_distill_ppo":
            policy_state_dict = self.policy.act(**kwargs)
            log_probs = self.policy.get_actions_log_prob(actions=actions)
            results = {
                "actions": policy_state_dict["actions"],
                "logprobs": log_probs,
                "action_mean": policy_state_dict["action_mean"],
                "action_std": policy_state_dict["action_sigma"],
                "entropy": self.policy.entropy,
            }
            if "normalized_actions" in policy_state_dict:
                results["normalized_actions"] = policy_state_dict["normalized_actions"]
        elif mode == "policy_w_and_wo_imgaug":
            # The first forward is without image augmentation
            self.policy.transform_eval()
            policy_state_dict = self.policy.act(**kwargs)
            # Use the distribution without image augmentation to get the log_probs
            log_probs = self.policy.get_actions_log_prob(actions=actions)
            results = {
                "actions": policy_state_dict["actions"],
                "logprobs": log_probs,
                "action_mean": policy_state_dict["action_mean"],
                "action_std": policy_state_dict["action_sigma"],
                "entropy": self.policy.entropy,
            }

            # The second forward is with image augmentation
            self.policy.transform_train()
            # The second time doesn't need deepcopy
            policy_state_dict_w_imgaug = self.policy.act(**kwargs)
            results["action_mean_w_imgaug"] = policy_state_dict_w_imgaug["action_mean"]
            results["actions_w_imgaug"] = policy_state_dict_w_imgaug["actions"]
            if "normalized_actions" in policy_state_dict_w_imgaug:
                results["normalized_actions_w_imgaug"] = policy_state_dict_w_imgaug[
                    "normalized_actions"
                ]
        elif mode == "policy_deterministic":
            self.policy.act(**kwargs)
            results = {
                "action_mean": self.policy.action_mean,
            }
        elif mode == "vae_policy_deterministic":
            self.policy.act(**kwargs)
            prior_mu, prior_log_var = self.policy.eval_prior(**kwargs)
            results = {
                "action_mean": self.policy.action_mean,
                "vae_mu": self.policy.z_mu,
                "vae_log_var": self.policy.z_log_sigma,
                "prior_mu": prior_mu,
                "prior_log_var": prior_log_var,
            }
        elif mode == "value":
            results = self.value_model.evaluate(**kwargs)
        elif mode == "eval_disc":
            results = self.disc_model.eval_disc(**kwargs)
        elif mode == "train_disc":
            results = self.disc_model.evaluate(**kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return results


class PrinterHVCallback(TrainerCallback):  # noqa: F405
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """  # noqa: D200, D212

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ARG002
        _ = logs.pop("total_flos", None)
        if state.is_world_process_zero:
            width = 80
            pad = 35
            print_str = f" \033[1m Learning iteration {state.global_step}  \033[0m "

            log_string = (
                f"""{print_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {logs['fps']:.0f} steps/s (Collection: {logs['collection_time']:.3f}s, Learning {logs['learn_time']:.3f}s)\n"""  # noqa: E501
                f"""{'Mean action noise std:':>{pad}} {logs['Policy/mean_noise_std']:.2f}\n"""
            )

            for k, v in logs.items():
                if k.startswith("objective/"):
                    # Keep the original logic
                    if k.startswith("objective/kin_"):
                        log_string += f"""{f'{k}:':>{pad}} {v:.5f}\n"""
                    else:
                        new_key = k.replace("objective/", "")
                        log_string += f"""{f'Mean {new_key}:':>{pad}} {v:.5f}\n"""

            env_log_string = ""
            ep_string = ""
            for k, v in logs.items():
                if k.startswith("Env/"):
                    entry = f"{f'{k}:':>{pad}} {v:.4f}"
                    env_log_string += f"{entry}\n"
                if k.startswith("Disc/"):
                    entry = f"{f'{k}:':>{pad}} {v:.4f}"
                    env_log_string += f"{entry}\n"
                if k.startswith("Episode/"):
                    new_key = k.replace("Episode/", "")
                    ep_string += f"""{f'Mean episode {new_key}:':>{pad}} {v:.4f}\n"""

            log_string += env_log_string
            log_string += ep_string
            log_string += (
                f"""{'-' * width}\n"""
                f"""{'Total episodes:':>{pad}} {logs['episode']}\n"""
                f"""{'Total timesteps:':>{pad}} {logs['tot_timesteps']}\n"""
                f"""{'Iteration time:':>{pad}} {logs['collection_time'] + logs['learn_time']:.2f}s\n"""
                f"""{'Total time:':>{pad}} {logs['tot_time']:.2f}s\n"""
                f"""{'ETA:':>{pad}} {logs['tot_time'] / logs['batch_idx'] * (logs['num_total_batches'] - logs['batch_idx']):.1f}s\n"""  # noqa: E501
            )

            log_string += f"Logging Directory: {logs['experiment_save_dir']}"
            with live.Live(
                panel.Panel(log_string, title="Training Log"),
                refresh_per_second=4,
                console=console_,
            ):
                # Your training loop or other operations
                pass


def process_ep_infos(ep_infos, device):
    """Aggregate per-episode info dicts into a single dict of per-key means.

    Args:
        ep_infos: List of episode info dicts, each mapping metric names to
            scalars or tensors.
        device: Torch device to place intermediate tensors on.

    Returns:
        Dict mapping each metric name to its mean value across all episodes.
    """
    infos = {}
    for key in ep_infos[0]:
        infotensor = torch.tensor([], device=device)
        for ep_info in ep_infos:
            # handle scalar and zero dimensional tensor infos
            if not isinstance(ep_info[key], torch.Tensor):
                ep_info[key] = torch.Tensor([ep_info[key]])
            if len(ep_info[key].shape) == 0:
                ep_info[key] = ep_info[key].unsqueeze(0)
            infotensor = torch.cat((infotensor, ep_info[key].to(device)))
        value = torch.mean(infotensor)
        infos[key] = value
    return infos


class TRLPPOTrainer(PPOTrainer):  # noqa: F405
    """PPO trainer adapted from HuggingFace TRL for humanoid whole-body control.

    Extends TRL's ``PPOTrainer`` to support IsaacLab-based vectorized
    environments, multi-critic advantage estimation, symmetry augmentation,
    adaptive KL-based learning rate scheduling, and gradient-checkpointed
    policy/value models.

    The training loop follows a standard on-policy PPO cycle:
        1. Collect rollouts with the current policy (``_rollout_step``).
        2. Compute GAE returns and advantages (``_compute_returns``).
        3. Run multiple PPO epochs of mini-/micro-batch updates (``train``).
        4. Synchronize running statistics across processes.
        5. Log metrics and invoke callbacks.
    """

    _tag_names = ["trl", "humanoid_ppo"]  # noqa: RUF012

    def __init__(
        self,
        args,
        config,
        env,
        model,
        ref_model=None,
        reward_model=None,
        processing_class=None,
        value_model=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        log_dir=None,
        # less commonly used
        optimizers=(None, None),
        callbacks=None,
        peft_config=None,
        use_ref_model=False,
        checkpoint=None,
        resume=False,
        local_seed=None,
        schedule_dict=None,
        accelerator=None,
        **kwargs,
    ) -> None:
        """Initialize the PPO trainer, models, optimizer, storage, and optionally load a checkpoint.

        Args:
            args: Training arguments (learning rate, batch sizes, clipping, etc.).
            config: Algorithm config dict (PPO hyperparameters, num_steps_per_env, etc.).
            env: Vectorized IsaacLab environment instance.
            model: Policy (actor) model.
            ref_model: Optional frozen reference policy for KL regularization.
            reward_model: Optional learned reward model.
            processing_class: HuggingFace processing class (unused, kept for API compat).
            value_model: Critic (value) model.
            data_collator: Optional data collator for the dataloader.
            train_dataset: Optional training dataset (defaults to env-driven rollouts).
            eval_dataset: Optional evaluation dataset.
            log_dir: Directory for saving training logs.
            optimizers: Tuple of ``(optimizer, lr_scheduler)``; created automatically if
                ``(None, None)``.
            callbacks: Additional ``TrainerCallback`` instances.
            peft_config: Optional PEFT configuration for parameter-efficient fine-tuning.
            use_ref_model: Whether to create/use a reference model for KL penalty.
            checkpoint: Path to a checkpoint file to load on init.
            resume: If True and *checkpoint* is provided, also restore optimizer,
                scheduler, and trainer state for full training resumption.
            local_seed: Per-process random seed for reproducibility under DDP.
            schedule_dict: Dict defining parameter schedules over training steps.
            accelerator: HuggingFace ``Accelerator`` instance for distributed training.
            **kwargs: Extra keyword arguments forwarded to ``_init_trl`` (e.g.
                ``disc_model`` for discriminator-based training).
        """
        self.accelerator = accelerator
        self._init_trl(
            args,
            config,
            env,
            processing_class,
            model,
            ref_model,
            reward_model,
            train_dataset,
            value_model,
            data_collator,
            eval_dataset,
            optimizers,
            callbacks,
            peft_config,
            use_ref_model,
            local_seed,
            log_dir,
            schedule_dict=schedule_dict,
            **kwargs,
        )
        self._init_config()
        self._setup_storage()

        if checkpoint is not None:
            self.load_checkpoint(checkpoint, resume=resume)

    def _init_trl(
        self,
        args,
        config,
        env,
        processing_class,
        model,
        ref_model,
        reward_model,
        train_dataset,
        value_model,
        data_collator,
        eval_dataset,
        optimizers,
        callbacks,
        peft_config,
        use_ref_model,
        local_seed,
        log_dir,
        schedule_dict=None,
        **kwargs,
    ):
        """Initialize TRL internals: models, optimizer, accelerator, dataloaders, and callbacks.

        NOTE: This replicates much of TRL's ``PPOTrainer.__init__`` because the
        upstream implementation assumes language-model rollouts, not vectorized
        environment rollouts. Batch-size calculations, PEFT handling, DeepSpeed
        preparation, and callback wiring are all customized here.
        """
        self.args = args
        self.config = config
        self.env = env
        self.processing_class = processing_class
        self.policy_model = model
        self.learn_normalized_actions = model.has_normalized_actions
        self.episode_env_tensors = average_meters.TensorAverageMeterDict()
        self.ep_infos = []
        self.eval_callbacks = []
        self.log_dir = log_dir
        self.schedule_dict = schedule_dict
        self.scheduled_params_dict = {}

        # peft support
        if not accelerate_utils.is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"  # noqa: E501
            )
        elif accelerate_utils.is_peft_available() and peft_config is not None:  # noqa: RET506
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            if isinstance(self.policy_model, peft.PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()

            # get peft model with the given config
            self.policy_model = peft.get_peft_model(self.policy_model, peft_config)
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                models_utils.peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = accelerate_utils.is_peft_available() and isinstance(
            self.policy_model, peft.PeftModel
        )
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if use_ref_model:
            if ref_model:
                self.ref_model = ref_model
            elif self.is_peft_model:
                self.ref_model = None
            else:
                self.ref_model = models.create_reference_model(self.policy_model)
        else:
            self.ref_model = None

        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = (
            len(train_dataset) if train_dataset is not None else self.env.config.num_envs
        )
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset

        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########

        accelerator = self.accelerator

        self.device = accelerator.device
        if "use_symmetry" in self.env.config and self.env.config.use_symmetry is not None:
            self.use_symmetry = self.env.use_symmetry
        else:
            self.use_symmetry = False
        args.global_rank = accelerator.process_index
        args.world_size = accelerator.num_processes
        args.is_main_process = accelerator.is_main_process
        args.local_batch_size = self.env.config.num_envs
        args.batch_size = int(args.local_batch_size * args.world_size)
        try:
            args.mini_batch_size = ppo_trainer.exact_div(
                args.batch_size,
                args.num_mini_batches,
                "`batch_size` must be a multiple of `num_mini_batches`",
            )
            args.local_mini_batch_size = ppo_trainer.exact_div(
                args.local_batch_size,
                args.num_mini_batches,
                "`local_batch_size` must be a multiple of `num_mini_batches`",
            )
        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")  # noqa: T201
            args.mini_batch_size = 1
            args.local_mini_batch_size = 1

        if args.per_device_train_batch_size is None:
            args.per_device_train_batch_size = (
                args.local_mini_batch_size
            )  # same as mini-batch size, which implies no micro-batching (num_micro_batches = 1)
        args.num_micro_batches = args.local_mini_batch_size // args.per_device_train_batch_size
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        if args.total_episodes is None:
            assert args.num_total_batches is not None
            args.total_episodes = args.num_total_batches * args.batch_size
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = accelerate_utils.broadcast(
            time_tensor, 0
        ).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = local_seed
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(
                1, args.num_total_batches // args.num_sample_generations
            )
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        if self.config.get("disable_dropout", True):
            for module in [self.policy_model, self.ref_model, self.value_model, self.reward_model]:
                if module is not None:
                    trainer_utils.disable_dropout_in_model(module)
        addition_models = {}
        if "reward_model" in kwargs:
            addition_models["reward_model"] = kwargs["reward_model"]
        if "disc_model" in kwargs:
            addition_models["disc_model"] = kwargs["disc_model"]

        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model, **addition_models)
        # self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(  # noqa: F405
            self.args.report_to
        )
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = hv_callback_handler.HVCallbackHandler(
            self.callbacks,
            self.model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
            self.env,
            self.accelerator,
        )
        self.add_callback(
            PrinterHVCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK  # noqa: F405
        )
        self.control = TrainerControl()  # noqa: F405
        self.state = ppo_trainer.OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]  # noqa: RUF005
                if isinstance(cb, ExportableState)  # noqa: F405
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = (
            getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        )
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        if self.train_dataset is not None:
            self.dataloader = DataLoader(  # noqa: F405
                self.train_dataset,
                batch_size=self.local_dataloader_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                drop_last=False,  # needed; otherwise the last batch will be of ragged shape
            )
        else:
            self.dataloader = None
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)

        self.model, self.optimizer, self.dataloader = accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
        self.unwrapped_model = unwrap_model(self.model)  # noqa: F405
        torch.manual_seed(self.local_seed)  # reset the local seed again

        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(  # noqa: F405
                self.eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                drop_last=False,
            )  # no need to shuffle eval dataset
            self.eval_dataloader = accelerator.prepare(self.eval_dataloader)
        else:
            self.eval_dataloader = None

        if self.is_deepspeed_enabled:
            if self.reward_model is not None:
                self.reward_model = trainer_utils.prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )

            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = trainer_utils.prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is None:
                # if not self.is_peft_model:
                #     raise ValueError("No reference model and model is not a Peft model.")
                pass
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)
        self.use_apex = False

        self.train_with_evaluating_env = self.config.get("train_with_evaluating_env", False)

        # Camera resolution
        if "vision_obs" in self.env.config.obs.obs_dict:
            if self.env.config.obs.obs_dict.vision_obs[0] in ["depth_image", "height_map"]:
                num_channels = 1
            elif self.env.config.obs.obs_dict.vision_obs[0] in ["rgb_image"]:
                num_channels = 3
            else:
                raise ValueError(
                    f"Invalid vision observation type: {self.env.config.obs.obs_dict.vision_obs[0]}"
                )

            if self.env.config.obs.obs_dict.vision_obs[0] == "height_map":
                heightmap_resolution = self.env.config.simulator.config.heightmap.resolution
                self.camera_resolution = [
                    heightmap_resolution,
                    heightmap_resolution,
                ] + [  # noqa: RUF005
                    num_channels
                ]
            else:
                self.camera_resolution = [
                    *self.env.config.simulator.config.cameras.camera_resolutions,
                    num_channels,
                ]
        elif "camera_rgb" in self.env.config.robot.algo_obs_dim_dict:
            # For manager_env with camera_rgb observation group
            camera_res = self.env.config.get("camera_resolution", [144, 256])
            self.camera_resolution = (camera_res[0], camera_res[1], 3)
        else:
            self.camera_resolution = None

        self.num_critics = self.env.config.rewards.get("num_critics", 1)

    def _init_config(self):
        """Extract PPO hyperparameters and environment dimensions from config."""
        # Env related Config
        self.num_envs: int = self.env.config.num_envs
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_act = self.policy_model.num_actions

        self.num_steps_per_env = self.config.num_steps_per_env
        self.use_padding_mask = self.config.get("use_padding_mask", False)
        self.ppo_shuffle_every_epoch = self.config.get("ppo_shuffle_every_epoch", True)
        self.empty_cache_every_n_ppo_epoch = self.config.get("empty_cache_every_n_ppo_epoch", -1)

        self.entropy_coef = self.config.entropy_coef
        self.desired_kl = self.config.desired_kl
        self.gamma = self.args.gamma
        self.lam = self.args.lam
        self.adaptive_lr_min = self.config.get("adaptive_lr_min", 1e-5)
        self.adaptive_lr_max = self.config.get("adaptive_lr_max", 1e-2)
        self.sync_advantage_normalization = self.config.get("sync_advantage_normalization", True)
        self.multi_critic_advantage_weights = self.config.get(
            "multi_critic_advantage_weights", None
        )

        self.compute_imgaug_bc_loss = self.config.get("compute_imgaug_bc_loss", False)
        self.imgaug_bc_loss_coef = self.config.get("imgaug_bc_loss_coef", 1.0)
        self.imgaug_bc_loss_fn = torch.nn.MSELoss()

    def _setup_storage(self):
        """Allocate rollout storage buffers and episode tracking accumulators.

        Registers observation, action, reward, done, value, return, and advantage
        buffers in a ``RolloutStorage`` instance sized for
        ``(num_envs, num_steps_per_env)``.
        """
        self.storage = data_utils.RolloutStorage(
            self.env.num_envs, self.num_steps_per_env, device=self.accelerator.device
        )
        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            obs_shape = (obs_dim,) if isinstance(obs_dim, int) else tuple(obs_dim)
            if obs_key in ["vision_obs", "camera_rgb"]:
                # Vision observations are stored as [H, W, C] image, not flattened
                self.storage.register_key(
                    obs_key, shape=tuple(self.camera_resolution), dtype=torch.float
                )
            else:
                self.storage.register_key(obs_key, shape=obs_shape, dtype=torch.float)
            if obs_key == "critic_obs" and self.use_symmetry:
                self.storage.register_key("next_" + obs_key, shape=obs_shape, dtype=torch.float)
        ## Register others
        reward_dim = self.num_critics
        self.storage.register_key("actions", shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key("rewards", shape=(reward_dim,), dtype=torch.float)
        self.storage.register_key("dones", shape=(1,), dtype=torch.bool)
        self.storage.register_key("time_outs", shape=(1,), dtype=torch.bool)
        self.storage.register_key("values", shape=(reward_dim,), dtype=torch.float)
        self.storage.register_key("returns", shape=(reward_dim,), dtype=torch.float)
        self.storage.register_key("advantages", shape=(reward_dim,), dtype=torch.float)
        self.storage.register_key("actions_log_prob", shape=(1,), dtype=torch.float)
        self.storage.register_key("action_mean", shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key("action_sigma", shape=(self.num_act,), dtype=torch.float)

        if self.learn_normalized_actions:
            self.storage.register_key(
                "normalized_actions", shape=(self.num_act,), dtype=torch.float
            )

        self.state.rewbuffer = deque(maxlen=100)
        self.state.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, self.num_critics, dtype=torch.float, device=self.accelerator.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.accelerator.device
        )
        self.state.cur_reward_sum = self.cur_reward_sum
        self.state.cur_episode_length = self.cur_episode_length
        self.ep_infos = []
        self.state.tot_timesteps = 0
        self.state.tot_time = 0
        self.state.eval_step = 0
        self.state.eval_render_step = 0

    def policy_step(self, policy_model, obs_dict, cur_dones=None):
        """Run the policy model for one rollout step, returning actions and log-probs.

        Constructs an episode attention mask from the done history in storage
        (for transformer-based policies), then calls ``policy_model.rollout``.

        Args:
            policy_model: The actor model to query.
            obs_dict: Current observations, each value ``(num_envs, obs_dim)``.
            cur_dones: Current done flags ``(num_envs,)``.  If None, the mask
                is built from the full storage done history.

        Returns:
            Dict containing ``"actions"`` ``(num_envs, act_dim)``,
            ``"actions_log_prob"`` ``(num_envs, 1)``, ``"action_mean"``,
            ``"action_sigma"``, and any additional policy outputs.
        """
        actor_obs_dict = obs_dict.copy()

        if cur_dones is None:
            dones = (
                self.storage.query_key("dones")
                .to(self.accelerator.device)[: self.storage.step + 1]
                .squeeze(-1)
                .transpose(0, 1)
            )
            episode_attnmask = rl.compute_episode_attnmask(dones)
        else:
            episode_attnmask = None

        policy_state_dict = policy_model.rollout(
            obs_dict=actor_obs_dict, episode_attnmask=episode_attnmask, cur_dones=cur_dones
        )
        actions = policy_state_dict["actions"]
        actions_log_prob = policy_model.get_actions_log_prob(actions).unsqueeze(1)
        policy_state_dict["actions_log_prob"] = actions_log_prob

        # assert len(actions.shape) == 2, f"{actions.shape=}"
        # assert len(actions_log_prob.shape) == 2, f"{actions_log_prob.shape=}"
        # assert len(action_mean.shape) == 2, f"{action_mean.shape=}"
        # assert len(action_sigma.shape) == 2, f"{action_sigma.shape=}"

        return policy_state_dict

    def _chunked_value_evaluate(self, value_model, obs_dict, episode_attnmask, chunk_size=1024):
        """Evaluate the value model in chunks to limit peak GPU memory.

        Args:
            value_model: Critic model with an ``evaluate`` method.
            obs_dict: Observation dict, each value ``(batch, seq, obs_dim)``.
            episode_attnmask: Attention mask ``(batch, seq, seq)`` or None.
            chunk_size: Maximum batch dimension per chunk.

        Returns:
            Value predictions ``(batch, seq, num_critics)``.
        """
        batch_size = list(obs_dict.values())[0].shape[0]  # noqa: RUF015
        if batch_size <= chunk_size:
            return value_model.evaluate(obs_dict=obs_dict, episode_attnmask=episode_attnmask)

        obs_chunks = {}
        for key, value in obs_dict.items():
            obs_chunks[key] = torch.split(value, chunk_size, dim=0)
        if episode_attnmask is not None:
            attnmask_chunks = torch.split(episode_attnmask, chunk_size, dim=0)
        else:
            attnmask_chunks = [None] * len(obs_chunks[list(obs_chunks.keys())[0]])  # noqa: RUF015

        value_chunks = []
        for i in range(len(attnmask_chunks)):
            chunk_obs_dict = {key: obs_chunks[key][i] for key in obs_chunks}
            chunk_values = value_model.evaluate(
                obs_dict=chunk_obs_dict, episode_attnmask=attnmask_chunks[i]
            )
            value_chunks.append(chunk_values)
        return torch.cat(value_chunks, dim=0)

    def _rollout_step(self, model, obs_dict):
        """Collect a full rollout of ``num_steps_per_env`` transitions and compute returns.

        Performs the environment interaction loop under ``torch.no_grad()``,
        stores transitions in ``self.storage``, then runs the value model over
        the full trajectory to compute GAE returns and advantages.

        Args:
            model: The ``PolicyAndValueWrapper`` model (unwrapped from DDP).
            obs_dict: Initial observations from the environment, each value
                ``(num_envs, obs_dim)``.

        Returns:
            The final observation dict after the last environment step,
            to be used as the starting point for the next rollout.
        """
        self._train_rollout_mode()
        device = self.accelerator.device
        policy_model = model.policy
        value_model = model.value_model
        policy_model.init_rollout()
        self.storage.clear()

        dones = torch.zeros(self.env.num_envs, device=device)
        with torch.no_grad():
            for i in range(self.num_steps_per_env):  # noqa: B007
                # Compute the actions and values
                # TODO: 1: unsqueeze to [B, 1, ...]  # noqa: TD002, TD003
                policy_state_dict = self.policy_step(policy_model, obs_dict, cur_dones=dones)

                # Append states to storage
                for key, value in obs_dict.items():
                    if key == "height_map":
                        if getattr(self.storage, key).ndim != 5:
                            # re-register height_map
                            delattr(self.storage, key)
                            self.storage.register_key(key, shape=value.shape[1:])
                    elif key in ["vision_obs", "camera_rgb"]:
                        # Vision observations have shape [B, H, W, C], verify storage matches
                        if getattr(self.storage, key, None) is None:
                            self.storage.register_key(key, shape=value.shape[1:], dtype=torch.float)
                        elif getattr(self.storage, key).shape[2:] != value.shape[1:]:
                            # re-register with correct shape
                            delattr(self.storage, key)
                            self.storage.register_key(key, shape=value.shape[1:], dtype=torch.float)
                    self.storage.update_key(key, value)
                for key, value in policy_state_dict.items():
                    if key == "obs_dict":
                        continue
                    self.storage.update_key(key, value)

                # Step the environment
                if self.use_symmetry:
                    obs_dict, rewards, dones, infos, termination_ids, termination_observations = (
                        self.env.step(policy_state_dict)
                    )
                else:
                    obs_dict, rewards, dones, infos = self.env.step(policy_state_dict)
                for obs_key in obs_dict.keys():  # noqa: SIM118
                    obs_dict[obs_key] = obs_dict[obs_key].to(device)
                    if obs_key == "critic_obs" and self.use_symmetry:
                        next_critic_obs = obs_dict[obs_key].clone()
                        next_critic_obs[termination_ids.to(device)] = termination_observations.to(
                            device
                        )
                        self.storage.update_key("next_" + obs_key, next_critic_obs)
                rewards, dones = rewards.to(device), dones.to(device)
                rewards_stored = rewards.clone()
                if rewards.dim() == 1:
                    rewards_stored = rewards_stored.unsqueeze(1)

                assert len(rewards_stored.shape) == 2

                self.ep_infos.append(infos["episode"])
                self.storage.update_key("rewards", rewards_stored)
                self.storage.update_key("dones", dones.unsqueeze(1))
                self.storage.update_key("time_outs", infos["time_outs"].unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)
                self.cur_reward_sum += rewards_stored
                self.cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                self.state.rewbuffer.extend(self.cur_reward_sum[new_ids].cpu().numpy().tolist())
                self.state.lenbuffer.extend(self.cur_episode_length[new_ids].cpu().numpy().tolist())
                self.cur_reward_sum[new_ids] = 0
                self.cur_episode_length[new_ids] = 0

            policy_model.clear_rollout()
            # gc.collect()
            # torch.cuda.empty_cache()

            if self.value_model is not None:
                dones = self.storage.query_key("dones").to(device).squeeze(-1).transpose(0, 1)
                dones = torch.cat([dones, torch.zeros_like(dones[:, :1])], dim=1)
                episode_attnmask = rl.compute_episode_attnmask(dones)
                all_obs_dict = {}
                for key in obs_dict.keys():  # noqa: SIM118
                    if key not in ["actor_obs"]:  # actor_obs not required by value model
                        obs_value = self.storage.query_key(key).to(device)
                        obs_value = torch.cat([obs_value, obs_dict[key].unsqueeze(0)], dim=0)
                        all_obs_dict[key] = obs_value.transpose(0, 1)
                all_values = self._chunked_value_evaluate(
                    value_model, all_obs_dict, episode_attnmask
                ).transpose(0, 1)
                values, last_values = all_values[:-1], all_values[-1]

                rewards = self.storage.query_key("rewards")

                new_rewards = (
                    rewards.to(device)
                    + self.gamma * self.storage.query_key("time_outs").to(device) * values
                )
                self.storage.batch_update_data("rewards", new_rewards)

                returns, advantages = self._compute_returns(
                    values=values,
                    last_values=last_values,
                    policy_state_dict={
                        "dones": self.storage.query_key("dones"),
                        "rewards": self.storage.query_key("rewards"),
                    },
                )
                self.storage.batch_update_data("values", values)
                self.storage.batch_update_data("returns", returns)
                self.storage.batch_update_data("advantages", advantages)

        return obs_dict

    def _flip_obs(self, obs, key):
        """Mirror observations left-right for symmetry augmentation.

        Args:
            obs: Observation tensor ``(batch, seq, obs_dim)``.
            key: Observation key (``"actor_obs"`` or ``"critic_obs"``), which
                determines the history length and flip index mapping.

        Returns:
            Flipped observation tensor with the same shape as *obs*.
        """
        if key == "actor_obs":
            proprioceptive_obs = obs.clone().view(
                obs.shape[0], obs.shape[1], self.env.actor_history_length + 1, -1
            )
            flipper_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
            flipper_proprioceptive_obs[:, :, :, :] = (
                proprioceptive_obs[:, :, :, self.env.flip_actor_obs_info[:, 0]]
                * self.env.flip_actor_obs_info[:, 1]
            )
        elif key == "critic_obs":
            proprioceptive_obs = obs.clone().view(
                obs.shape[0], obs.shape[1], self.env.critic_history_length + 1, -1
            )
            flipper_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
            flipper_proprioceptive_obs[:, :, :, :] = (
                proprioceptive_obs[:, :, :, self.env.flip_critic_obs_info[:, 0]]
                * self.env.flip_critic_obs_info[:, 1]
            )
        else:
            raise NotImplementedError
        return flipper_proprioceptive_obs.view(obs.shape[0], obs.shape[1], -1)

    def _flip_actions(self, actions):
        """Mirror actions left-right for symmetry augmentation.

        Args:
            actions: Action tensor ``(batch, seq, act_dim)``.

        Returns:
            Flipped action tensor with the same shape.
        """
        flipped_actions = (
            actions[:, :, self.env.flip_action_info[:, 0]] * self.env.flip_action_info[:, 1]
        )
        return flipped_actions

    def _process_env_step(self, rewards, dones, infos):  # noqa: ARG002
        """Handle post-step bookkeeping: reset models on done envs and log metrics.

        Args:
            rewards: Reward tensor ``(num_envs,)`` or ``(num_envs, num_critics)``.
            dones: Done flags ``(num_envs,)``.
            infos: Info dict from the environment step, containing ``"to_log"``
                entries for metric tracking.
        """
        self.policy_model.reset(dones)
        if self.value_model is not None:
            self.value_model.reset(dones)
        self.episode_env_tensors.add(infos["to_log"])

    def _register_stats_buffer(self):
        """Allocate per-epoch/mini-batch/micro-batch statistic tensors for logging.

        Tensors have shape ``(num_ppo_epochs, num_mini_batches, num_micro_batches)``
        and are overwritten each training iteration.
        """
        args = self.args
        device = self.accelerator.device

        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.num_micro_batches)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        weighted_ppo_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        advantage_mean_stats = torch.zeros(stats_shape, device=device)
        advantage_std_stats = torch.zeros(stats_shape, device=device)
        if self.compute_imgaug_bc_loss:
            imgaug_bc_loss_stats = torch.zeros(stats_shape, device=device)
            weighted_imgaug_bc_loss_stats = torch.zeros(stats_shape, device=device)

        self.approxkl_stats = approxkl_stats
        self.pg_clipfrac_stats = pg_clipfrac_stats
        self.pg_loss_stats = pg_loss_stats
        self.vf_loss_stats = vf_loss_stats
        self.entropy_stats = entropy_stats
        self.weighted_ppo_loss_stats = weighted_ppo_loss_stats
        self.vf_clipfrac_stats = vf_clipfrac_stats
        self.ratio_stats = ratio_stats
        self.advantage_mean_stats = advantage_mean_stats
        self.advantage_std_stats = advantage_std_stats
        if self.use_symmetry:
            estimation_loss_stats = torch.zeros(stats_shape, device=device)
            swap_loss_stats = torch.zeros(stats_shape, device=device)
            actor_sym_loss_stats = torch.zeros(stats_shape, device=device)
            critic_sym_loss_stats = torch.zeros(stats_shape, device=device)
            self.estimation_loss_stats = estimation_loss_stats
            self.swap_loss_stats = swap_loss_stats
            self.actor_sym_loss_stats = actor_sym_loss_stats
            self.critic_sym_loss_stats = critic_sym_loss_stats

        if self.compute_imgaug_bc_loss:
            self.imgaug_bc_loss_stats = imgaug_bc_loss_stats
            self.weighted_imgaug_bc_loss_stats = weighted_imgaug_bc_loss_stats

    def _get_rollout_data(self, obs_keys):
        """Transpose storage from ``(steps, envs, ...)`` to ``(envs, steps, ...)`` and apply augmentations.

        Retrieves all rollout tensors from ``self.storage``, optionally doubles
        the batch via left-right symmetry flipping, and builds padding masks
        for episodes that terminated mid-rollout.

        Args:
            obs_keys: Observation keys to extract from storage.

        Returns:
            Dict containing ``"all_obs_dict"``, ``"actions"``, ``"logprobs"``,
            ``"values"``, ``"rewards"``, ``"dones"``, ``"old_mu_batch"``,
            ``"old_sigma_batch"``, ``"returns"``, ``"advantages"``, and
            padding masks.
        """
        device = self.accelerator.device

        all_obs_dict = {
            key: self.storage.query_key(key).transpose(0, 1).to(device) for key in obs_keys
        }
        actions = self.storage.actions.transpose(0, 1).to(device)
        logprobs = self.storage.actions_log_prob.transpose(0, 1).squeeze(-1).to(device)
        values = self.storage.values.transpose(0, 1).to(device)  # noqa: PD011
        rewards = self.storage.rewards.transpose(0, 1).to(device)
        dones = self.storage.dones.transpose(0, 1).squeeze(-1).to(device)
        old_mu_batch = self.storage.action_mean.transpose(0, 1).to(device)
        old_sigma_batch = self.storage.action_sigma.transpose(0, 1).to(device)
        returns = self.storage.returns.transpose(0, 1).to(device)
        advantages = self.storage.advantages.transpose(0, 1).to(device)

        if self.use_symmetry:
            next_critic_obs = self.storage.next_critic_obs.transpose(0, 1).to(device)
            all_obs_dict = {
                key: torch.cat((all_obs_dict[key], self._flip_obs(all_obs_dict[key], key)), dim=0)
                for key in all_obs_dict.keys()  # noqa: SIM118
            }
            next_critic_obs = torch.cat(
                (next_critic_obs, self._flip_obs(next_critic_obs, "critic_obs")), dim=0
            )
            actions = torch.cat((actions, self._flip_actions(actions)), dim=0)
            logprobs = logprobs.repeat(2, 1)
            values = values.repeat(2, 1, 1)
            rewards = rewards.repeat(2, 1, 1)
            dones = dones.repeat(2, 1)
            old_mu_batch = old_mu_batch.repeat(2, 1, 1)
            old_sigma_batch = old_sigma_batch.repeat(2, 1, 1)
            returns = returns.repeat(2, 1, 1)
            advantages = advantages.repeat(2, 1, 1)

        if self.use_padding_mask:
            padding_mask = dones.clone()
            padding_mask_p1 = padding_mask.clone()
            for i in range(padding_mask.shape[0]):
                true_indices = torch.where(padding_mask[i])[0]
                if len(true_indices) > 0:
                    padding_mask[i, true_indices[0]] = False
                    padding_mask_p1[
                        i, true_indices[0] : min(true_indices[0] + 2, padding_mask_p1.shape[1])
                    ] = False
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            values = torch.masked_fill(values, padding_mask_p1, 0)
        else:
            padding_mask = torch.zeros_like(dones)
            padding_mask_p1 = torch.zeros_like(dones)

        rollout_data = {
            "all_obs_dict": all_obs_dict,
            "actions": actions,
            "logprobs": logprobs,
            "values": values,
            "rewards": rewards,
            "dones": dones,
            "old_mu_batch": old_mu_batch,
            "old_sigma_batch": old_sigma_batch,
            "returns": returns,
            "advantages": advantages,
            "padding_mask": padding_mask,
            "padding_mask_p1": padding_mask_p1,
        }
        if self.use_symmetry:
            rollout_data["next_critic_obs"] = next_critic_obs
        return rollout_data

    def _get_mb_rollout_data(self, rollout_data, micro_batch_inds):
        """Slice a micro-batch from the full rollout data and build its attention mask.

        Args:
            rollout_data: Full rollout dict from ``_get_rollout_data``.
            micro_batch_inds: 1-D index tensor selecting environments for this
                micro-batch.

        Returns:
            Dict of micro-batch tensors (prefixed ``"mb_"``) plus the
            ``"episode_attnmask"`` for this subset.
        """
        mb_obs_dict = {
            key: rollout_data["all_obs_dict"][key][micro_batch_inds]
            for key in rollout_data["all_obs_dict"].keys()  # noqa: SIM118
        }
        mb_advantage = rollout_data["advantages"][micro_batch_inds]
        mb_logprobs = rollout_data["logprobs"][micro_batch_inds]
        mb_return = rollout_data["returns"][micro_batch_inds]
        mb_values = rollout_data["values"][micro_batch_inds]
        mb_dones = rollout_data["dones"][micro_batch_inds]
        mb_actions = rollout_data["actions"][micro_batch_inds]
        mb_old_mu = rollout_data["old_mu_batch"][micro_batch_inds]
        mb_old_sigma = rollout_data["old_sigma_batch"][micro_batch_inds]
        mb_padding_mask = rollout_data["padding_mask"][micro_batch_inds]
        mb_padding_mask_p1 = rollout_data["padding_mask_p1"][micro_batch_inds]

        episode_attnmask = rl.compute_episode_attnmask(mb_dones)

        mb_rollout_data = {
            "micro_batch_inds": micro_batch_inds,
            "mb_obs_dict": mb_obs_dict,
            "mb_advantage": mb_advantage,
            "mb_logprobs": mb_logprobs,
            "mb_return": mb_return,
            "mb_values": mb_values,
            "mb_dones": mb_dones,
            "mb_actions": mb_actions,
            "mb_old_mu": mb_old_mu,
            "mb_old_sigma": mb_old_sigma,
            "mb_padding_mask": mb_padding_mask,
            "mb_padding_mask_p1": mb_padding_mask_p1,
            "episode_attnmask": episode_attnmask,
        }
        if self.use_symmetry:
            mb_next_critic_obs = rollout_data["next_critic_obs"][micro_batch_inds]
            mb_rollout_data["mb_next_critic_obs"] = mb_next_critic_obs
        return mb_rollout_data

    def _forward_model(self, model, mb_rollout_data):
        """Run a single combined policy + value forward pass on a micro-batch.

        NOTE: Policy and value are forwarded together in one ``model.forward``
        call so DDP only synchronizes gradients once per backward pass.

        Args:
            model: The ``PolicyAndValueWrapper`` (possibly DDP-wrapped).
            mb_rollout_data: Micro-batch dict from ``_get_mb_rollout_data``.

        Returns:
            Dict with ``"policy_results"`` and ``"value_results"`` sub-dicts.
        """
        mb_obs_dict = mb_rollout_data["mb_obs_dict"]
        mb_actions = mb_rollout_data["mb_actions"]
        episode_attnmask = mb_rollout_data["episode_attnmask"]

        # We should only do one forward pass for especially DDP model
        if self.compute_imgaug_bc_loss:
            results = model.forward(
                modes=["policy_w_and_wo_imgaug", "value"],
                input_kwargs={
                    "policy_w_and_wo_imgaug": {
                        "obs_dict": mb_obs_dict,
                        "actions": mb_actions,
                        "episode_attnmask": episode_attnmask,
                    },
                    "value": {"obs_dict": mb_obs_dict, "episode_attnmask": episode_attnmask},
                },
            )
            policy_results = results["policy_w_and_wo_imgaug"]
        else:
            with common.Timer("wrapper_forward_model"):
                results = model.forward(
                    modes=["policy", "value"],
                    input_kwargs={
                        "policy": {
                            "obs_dict": mb_obs_dict,
                            "actions": mb_actions,
                            "episode_attnmask": episode_attnmask,
                        },
                        "value": {"obs_dict": mb_obs_dict, "episode_attnmask": episode_attnmask},
                    },
                )
                policy_results = results["policy"]
        return {
            "policy_results": policy_results,
            "value_results": results["value"],
        }

    def _compute_loss(self, forward_results, mb_rollout_data):
        """Compute the total loss as a weighted sum of PPO and optional auxiliary losses.

        Args:
            forward_results: Output of ``_forward_model``.
            mb_rollout_data: Micro-batch dict from ``_get_mb_rollout_data``.

        Returns:
            Dict containing ``"loss"`` (scalar to backprop), ``"ppo_loss_dict"``,
            and optionally ``"imgaug_bc_loss_dict"``.
        """
        ppo_loss_dict = self._compute_ppo_loss(forward_results, mb_rollout_data)

        loss = ppo_loss_dict["ppo_loss"] * self.config.get("ppo_loss_coef", 1.0)

        ret_dict = {
            "ppo_loss_dict": ppo_loss_dict,
        }

        if self.compute_imgaug_bc_loss:
            imgaug_bc_loss_dict = self._compute_imgaug_bc_loss(forward_results, mb_rollout_data)
            loss += imgaug_bc_loss_dict["imgaug_bc_loss"] * self.config.imgaug_bc_loss_coef
            ret_dict["imgaug_bc_loss_dict"] = imgaug_bc_loss_dict

        ret_dict["loss"] = loss

        return ret_dict

    def _compute_ppo_loss(self, forward_results, mb_rollout_data):
        """Compute the clipped PPO surrogate loss, value loss, and entropy bonus.

        Implements standard clipped PPO with:
        - Clipped surrogate policy gradient loss.
        - Clipped value function loss.
        - Entropy regularization.
        - Adaptive learning rate adjustment based on KL divergence.
        - Optional left-right symmetry consistency losses for actor and critic.

        Args:
            forward_results: Output of ``_forward_model``.
            mb_rollout_data: Micro-batch dict from ``_get_mb_rollout_data``.

        Returns:
            Dict with ``"ppo_loss"`` (combined scalar), plus individual loss
            components and diagnostic metrics (KL, clip fractions, ratios).
        """
        args = self.args
        optimizer = self.optimizer

        policy_results = forward_results["policy_results"]
        value_results = forward_results["value_results"]

        mb_obs_dict = mb_rollout_data["mb_obs_dict"]
        mb_old_mu = mb_rollout_data["mb_old_mu"]
        mb_old_sigma = mb_rollout_data["mb_old_sigma"]
        mb_values = mb_rollout_data["mb_values"]
        mb_return = mb_rollout_data["mb_return"]
        mb_logprobs = mb_rollout_data["mb_logprobs"]
        mb_advantage = mb_rollout_data["mb_advantage"]
        padding_mask = mb_rollout_data["mb_padding_mask"]
        padding_mask_p1 = mb_rollout_data["mb_padding_mask_p1"]
        micro_batch_inds = mb_rollout_data["micro_batch_inds"]  # noqa: F841

        new_logprobs = policy_results["logprobs"]
        sigma_batch = policy_results["action_std"]
        mu_batch = policy_results["action_mean"]
        entropy_batch = policy_results["entropy"]
        with torch.no_grad():
            kl = torch.sum(
                torch.log(sigma_batch / mb_old_sigma + 1.0e-5)
                + (torch.square(mb_old_sigma) + torch.square(mb_old_mu - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                axis=-1,
            )
            local_kl_mean = torch.mean(kl)
            kl_mean = self.accelerator.gather(local_kl_mean).mean()
            self._adjust_learning_rate_based_on_kl(kl_mean, optimizer)

        # Forward a DDP model twice will cause the error: "one of the variables needed for gradient computation has been modified by an inplace operation"  # noqa: E501
        vpred = value_results
        vpredclipped = torch.clamp(
            vpred,
            mb_values - args.cliprange_value,
            mb_values + args.cliprange_value,
        )
        vf_losses1 = torch.square(vpred - mb_return)
        vf_losses2 = torch.square(vpredclipped - mb_return)
        vf_loss_max = torch.max(vf_losses1, vf_losses2).mean(dim=-1)
        # vf_loss_max[vf_loss_max.isnan()] = 0.0
        vf_loss = masked_mean(vf_loss_max, ~padding_mask_p1)
        vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), ~padding_mask_p1.unsqueeze(-1))
        logprobs_diff = new_logprobs - mb_logprobs
        ratio = torch.exp(logprobs_diff).unsqueeze(-1)
        if self.multi_critic_advantage_weights is not None:
            mb_advantage = (
                mb_advantage
                * torch.tensor(self.multi_critic_advantage_weights).to(mb_advantage)[None, None, :]
            )
        pg_losses = -mb_advantage * ratio
        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
        pg_loss_max = torch.max(pg_losses, pg_losses2).sum(dim=-1)
        # pg_loss_max[pg_loss_max.isnan()] = 0.0
        pg_loss = masked_mean(pg_loss_max, ~padding_mask)

        # entropy_batch[entropy_batch.isnan()] = 0.0
        entropy_loss = -masked_mean(entropy_batch, ~padding_mask)
        if self.use_symmetry:
            actor_sym_loss = torch.mean(
                torch.sum(
                    torch.square(
                        self._flip_actions(self.policy_model(mb_obs_dict))
                        - self.policy_model(
                            {"actor_obs": self._flip_obs(mb_obs_dict["actor_obs"], "actor_obs")}
                        )
                    ),
                    dim=-1,
                )
            )
            critic_sym_loss = torch.mean(
                torch.sum(
                    torch.square(
                        self.value_model.critic(mb_obs_dict["critic_obs"])
                        - self.value_model.critic(
                            {"critic_obs": self._flip_obs(mb_obs_dict["critic_obs"], "critic_obs")}
                        )
                    ),
                    dim=-1,
                )
            )
            loss = (
                pg_loss
                + args.vf_coef * vf_loss
                + self.entropy_coef * entropy_loss
                + actor_sym_loss
                + critic_sym_loss
            )
        else:
            loss = pg_loss + args.vf_coef * vf_loss + self.entropy_coef * entropy_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss detected: {loss}")  # noqa: T201
            print(
                f"Ratio stats: min={ratio.min()}, max={ratio.max()}, mean={ratio.mean()}"
            )  # noqa: T201
            print(
                f"Advantage stats: min={mb_advantage.min()}, max={mb_advantage.max()}"
            )  # noqa: T201
            # Skip this update or use previous valid parameters

        loss_dict = {
            "ppo_loss": loss,
            # logging metrics
            "local_kl_mean": local_kl_mean,
            "pg_losses": pg_losses,
            "pg_losses2": pg_losses2,
            "pg_loss": pg_loss,
            "vf_loss": vf_loss,
            "entropy_loss": entropy_loss,
            "ratio": ratio,
            "vf_clipfrac": vf_clipfrac,
        }
        if self.use_symmetry:
            loss_dict["actor_sym_loss"] = actor_sym_loss
            loss_dict["critic_sym_loss"] = critic_sym_loss
        return loss_dict

    def _compute_imgaug_bc_loss(self, forward_results, mb_rollout_data):  # noqa: ARG002
        """Compute behavior cloning loss between augmented and non-augmented action means.

        Encourages the policy to produce similar actions regardless of image
        augmentation, improving sim-to-real visual transfer.

        Args:
            forward_results: Output of ``_forward_model`` (must use
                ``"policy_w_and_wo_imgaug"`` mode).
            mb_rollout_data: Micro-batch dict (unused directly, kept for API
                consistency).

        Returns:
            Dict with ``"imgaug_bc_loss"`` scalar.
        """
        policy_results = forward_results["policy_results"]
        mu_batch = policy_results["action_mean"]

        action_mean_w_imgaug = policy_results["action_mean_w_imgaug"]
        imgaug_bc_loss = self.imgaug_bc_loss_fn(action_mean_w_imgaug, mu_batch.detach())

        return {
            "imgaug_bc_loss": imgaug_bc_loss,
        }

    def _update_stats_buffer(
        self,
        ppo_epoch_idx,
        minibatch_idx,
        microbatch_idx,
        loss_dict,
        forward_results,  # noqa: ARG002
        mb_rollout_data,
    ):
        """Record per-update diagnostic statistics into the pre-allocated stat buffers.

        Args:
            ppo_epoch_idx: Current PPO epoch index.
            minibatch_idx: Current mini-batch index within the epoch.
            microbatch_idx: Current micro-batch index within the mini-batch.
            loss_dict: Output of ``_compute_loss``.
            forward_results: Output of ``_forward_model``.
            mb_rollout_data: Micro-batch dict from ``_get_mb_rollout_data``.
        """
        local_kl_mean = loss_dict["ppo_loss_dict"]["local_kl_mean"]
        pg_losses = loss_dict["ppo_loss_dict"]["pg_losses"].mean(dim=-1)
        pg_losses2 = loss_dict["ppo_loss_dict"]["pg_losses2"].mean(dim=-1)
        pg_loss = loss_dict["ppo_loss_dict"]["pg_loss"]
        vf_loss = loss_dict["ppo_loss_dict"]["vf_loss"]
        entropy_loss = loss_dict["ppo_loss_dict"]["entropy_loss"]
        weighted_ppo_loss = loss_dict["ppo_loss_dict"]["ppo_loss"] * self.config.get(
            "ppo_loss_coef", 1.0
        )
        ratio = loss_dict["ppo_loss_dict"]["ratio"]
        vf_clipfrac = loss_dict["ppo_loss_dict"]["vf_clipfrac"]

        padding_mask = mb_rollout_data["mb_padding_mask"]
        micro_batch_inds = mb_rollout_data["micro_batch_inds"]  # noqa: F841

        self.approxkl_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = local_kl_mean
        pg_clipfrac = masked_mean((pg_losses2 > pg_losses).float(), ~padding_mask)
        self.pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = pg_clipfrac
        self.pg_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = pg_loss
        self.vf_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = vf_loss
        if self.compute_imgaug_bc_loss:
            imgaug_bc_loss = loss_dict["imgaug_bc_loss_dict"]["imgaug_bc_loss"]
            self.imgaug_bc_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = imgaug_bc_loss
            self.weighted_imgaug_bc_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = (
                self.config.imgaug_bc_loss_coef * imgaug_bc_loss
            )
        if self.use_symmetry:
            self.actor_sym_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = loss_dict[
                "ppo_loss_dict"
            ]["actor_sym_loss"]
            self.critic_sym_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = loss_dict[
                "ppo_loss_dict"
            ]["critic_sym_loss"]
            self.estimation_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = loss_dict[
                "ppo_loss_dict"
            ]["estimation_loss"]
            self.swap_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = loss_dict[
                "ppo_loss_dict"
            ]["swap_loss"]
        self.entropy_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = -entropy_loss
        self.weighted_ppo_loss_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = (
            weighted_ppo_loss
        )
        self.vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = vf_clipfrac
        self.ratio_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = ratio.mean()
        self.advantage_mean_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = mb_rollout_data[
            "mb_advantage"
        ].mean()
        self.advantage_std_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = mb_rollout_data[
            "mb_advantage"
        ].std()

    def _get_train_metrics(self):
        """Gather and aggregate training statistics from all processes.

        Returns:
            Dict of scalar metrics (approx KL, clip fractions, losses, entropy,
            ratios, advantage stats) averaged across all GPUs and update steps.
        """
        metrics = {}

        approxkl_avg = self.accelerator.gather_for_metrics(self.approxkl_stats).mean().item()

        metrics["policy/approxkl_avg"] = approxkl_avg
        metrics["policy/clipfrac_avg"] = (
            self.accelerator.gather_for_metrics(self.pg_clipfrac_stats).mean().item()
        )
        metrics["loss/policy_avg"] = (
            self.accelerator.gather_for_metrics(self.pg_loss_stats).mean().item()
        )
        if self.compute_imgaug_bc_loss:
            metrics["loss/imgaug_bc_avg"] = (
                self.accelerator.gather_for_metrics(self.imgaug_bc_loss_stats).mean().item()
            )
            metrics["loss/weighted_imgaug_bc_avg"] = (
                self.accelerator.gather_for_metrics(self.weighted_imgaug_bc_loss_stats)
                .mean()
                .item()
            )
        if self.use_symmetry:
            metrics["loss/actor_sym"] = (
                self.accelerator.gather_for_metrics(self.actor_sym_loss_stats).mean().item()
            )
            metrics["loss/critic_sym"] = (
                self.accelerator.gather_for_metrics(self.critic_sym_loss_stats).mean().item()
            )
            metrics["loss/estimation"] = (
                self.accelerator.gather_for_metrics(self.estimation_loss_stats).mean().item()
            )
            metrics["loss/swap"] = (
                self.accelerator.gather_for_metrics(self.swap_loss_stats).mean().item()
            )
        metrics["loss/value_avg"] = (
            self.accelerator.gather_for_metrics(self.vf_loss_stats).mean().item()
        )
        metrics["loss/entropy_avg"] = (
            self.accelerator.gather_for_metrics(self.entropy_stats).mean().item()
        )
        metrics["loss/weighted_ppo_loss_avg"] = (
            self.accelerator.gather_for_metrics(self.weighted_ppo_loss_stats).mean().item()
        )
        metrics["val/clipfrac_avg"] = (
            self.accelerator.gather_for_metrics(self.vf_clipfrac_stats).mean().item()
        )
        metrics["val/ratio"] = self.accelerator.gather_for_metrics(self.ratio_stats).mean().item()
        metrics["val/ratio_var"] = (
            self.accelerator.gather_for_metrics(self.ratio_stats).var().item()
        )
        metrics["val/advantage_mean"] = (
            self.accelerator.gather_for_metrics(self.advantage_mean_stats).mean().item()
        )
        metrics["val/advantage_std"] = (
            self.accelerator.gather_for_metrics(self.advantage_std_stats).mean().item()
        )
        metrics["objective/entropy"] = metrics["loss/entropy_avg"]

        return metrics

    def train(self):
        """Run the full PPO training loop until ``num_total_batches`` iterations.

        Each iteration:
            1. Collect ``num_steps_per_env`` transitions via ``_rollout_step``.
            2. Compute GAE returns and advantages.
            3. Run ``num_ppo_epochs`` of mini-batch gradient updates.
            4. Synchronize running mean/std and adaptive sampling across GPUs.
            5. Log metrics and invoke ``on_step_end`` callbacks (which handle
               checkpointing, evaluation, and early stopping).
        """
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                if dataloader is not None:
                    yield from dataloader
                else:
                    yield None

        iter_dataloader = iter(repeat_generator())

        accelerator.print("===training policy===")
        start_time = time.time()
        self._register_stats_buffer()
        model.train()

        # trainer state initialization
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        # env
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():  # noqa: SIM118
            obs_dict[obs_key] = obs_dict[obs_key].to(device)

        for batch_idx in range(1, args.num_total_batches + 1):
            batch_start_time = time.time()
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)  # noqa: F841

            # update scheduled params
            if self.schedule_dict is not None:
                self.scheduled_params_dict = scheduler.update_scheduled_params(
                    self, self.schedule_dict, self.state.global_step
                )

            reinit_dr_freq = self.env.config.get("reinit_dr_freq", 0)
            if reinit_dr_freq > 0 and self.state.global_step % reinit_dr_freq == 0:
                self.env.reinit_dr()
                if self.env.config.get("reset_on_reinit_dr", False):
                    obs_dict = self.env.reset_all()
                    for obs_key in obs_dict.keys():  # noqa: SIM118
                        obs_dict[obs_key] = obs_dict[obs_key].to(device)

            with torch.no_grad():
                with models_utils.unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as model:
                    obs_dict = self._rollout_step(model, obs_dict)

                end_collection_time = time.time()
                collection_time = end_collection_time - batch_start_time

            with common.Timer("get_rollout_data"):
                rollout_data = self._get_rollout_data(obs_keys=obs_dict.keys())

                model = self.model
                self._train_mode()
            with common.Timer("ppo_training"):
                for ppo_epoch_idx in range(args.num_ppo_epochs):
                    minibatch_idx = 0
                    if self.ppo_shuffle_every_epoch or ppo_epoch_idx == 0:
                        b_inds = torch.randperm(args.local_batch_size, device=device)
                    for mini_batch_start in range(
                        0, args.local_batch_size, args.local_mini_batch_size
                    ):
                        mini_batch_end = mini_batch_start + args.local_mini_batch_size
                        mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                        microbatch_idx = 0
                        for micro_batch_start in range(
                            0, args.local_mini_batch_size, args.per_device_train_batch_size
                        ):
                            with common.Timer(  # noqa: SIM117
                                f"ppo_microbatch_{ppo_epoch_idx}_{minibatch_idx}_{microbatch_idx}"
                            ):
                                with accelerator.accumulate(model):
                                    with common.Timer("get_mb_rollout_data"):
                                        micro_batch_end = (
                                            micro_batch_start + args.per_device_train_batch_size
                                        )
                                        micro_batch_inds = mini_batch_inds[
                                            micro_batch_start:micro_batch_end
                                        ]
                                        mb_rollout_data = self._get_mb_rollout_data(
                                            rollout_data, micro_batch_inds
                                        )

                                    if self.use_symmetry:
                                        estimation_loss, swap_loss = (
                                            self.policy_model.update_estimator(
                                                mb_rollout_data["mb_obs_dict"]["actor_obs"],
                                                mb_rollout_data["mb_next_critic_obs"],
                                                self.args.learning_rate,
                                            )
                                        )

                                    with common.Timer("forward_model"):
                                        forward_results = self._forward_model(
                                            model, mb_rollout_data
                                        )

                                    with common.Timer("compute_loss"):
                                        loss_dict = self._compute_loss(
                                            forward_results, mb_rollout_data
                                        )

                                    with common.Timer("backward"):
                                        accelerator.backward(loss_dict["loss"])

                                    with common.Timer("gradient_clipping"):
                                        grad_norm = self._gradient_clipping()

                                    if grad_norm is not None:
                                        with common.Timer("optimizer_step"):
                                            optimizer.step()
                                    else:
                                        print("NaN in gradient! Skipped!!!!")  # noqa: T201

                                    optimizer.zero_grad()
                                    with torch.no_grad():
                                        if self.use_symmetry:
                                            loss_dict["ppo_loss_dict"][
                                                "estimation_loss"
                                            ] = estimation_loss
                                            loss_dict["ppo_loss_dict"]["swap_loss"] = swap_loss
                                        with common.Timer("update_stats_buffer"):
                                            self._update_stats_buffer(
                                                ppo_epoch_idx,
                                                minibatch_idx,
                                                microbatch_idx,
                                                loss_dict,
                                                forward_results,
                                                mb_rollout_data,
                                            )
                                    del loss_dict, forward_results, mb_rollout_data
                                    microbatch_idx += 1
                        minibatch_idx += 1  # noqa: SIM113
                        # del everything and empty cache
                        # fmt: off
                    # if self.empty_cache_every_n_ppo_epoch > 0 and (ppo_epoch_idx + 1) % self.empty_cache_every_n_ppo_epoch == 0:  # noqa: E501
                    #     # print(f"Empty cache at ppo_epoch_idx {ppo_epoch_idx}")
                    #     gc.collect()
                    #     torch.cuda.empty_cache()
            ######################################################### Sync Running Mean Std #########################################################  # noqa: E501
            with common.Timer("sync_running_mean_std"):
                self.sync_running_mean_std()

            with common.Timer("sync_adaptive_sampling"):
                self.sync_adaptive_sampling()

            # print(self.accelerator.process_index, self.model.module.policy.running_mean_std.running_mean.mean(), self.model.module.policy.running_mean_std.running_var.mean(), self.model.module.policy.running_mean_std.count)  # noqa: E501
            # print(self.accelerator.process_index, self.policy_model.running_mean_std.running_mean)
            # print('--------------------------------')
            # print('**********', self.policy_model.running_mean_std.running_mean)

            # print(self.accelerator.process_index, self.policy_model.running_mean_std._normalizer.state_dict())
            # print('--------------------------------')
            # if self.state.global_step == 3:
            #     if self.accelerator.is_main_process :
            #         import ipdb; ipdb.set_trace()
            #     else:
            #         time.sleep(100000000)

            ######################################################### Sync Running Mean Std #########################################################  # noqa: E501

            with torch.no_grad():
                learn_time = time.time() - end_collection_time
                eps = int(self.state.episode / (time.time() - start_time))

                metrics = {}
                train_metrics = self._get_train_metrics()
                metrics.update(train_metrics)
                metrics["eps"] = eps
                metrics["objective/rewards"] = (
                    self.accelerator.gather_for_metrics(
                        torch.tensor(np.mean(np.array(self.state.rewbuffer).sum(axis=-1))).to(
                            device
                        )
                    )
                    .mean()
                    .item()
                )
                metrics["objective/length"] = (
                    self.accelerator.gather_for_metrics(
                        torch.tensor(np.mean(self.state.lenbuffer)).to(device)
                    )
                    .mean()
                    .item()
                )
                metrics["lr"] = self.args.learning_rate
                metrics["episode"] = self.state.episode
                env_log_dict = self.episode_env_tensors.mean_and_clear()

                ep_infos = process_ep_infos(self.ep_infos, device)
                self.state.tot_timesteps += (
                    self.num_steps_per_env * self.env.num_envs * accelerator.num_processes
                )
                self.state.tot_time += collection_time + learn_time
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                log_dict = {
                    "collection_time": collection_time,
                    "learn_time": learn_time,
                    "tot_timesteps": self.state.tot_timesteps,
                    "tot_time": self.state.tot_time,
                    "it": self.state.global_step,
                    "fps": int(
                        self.num_steps_per_env
                        * self.env.num_envs
                        * accelerator.num_processes
                        / (collection_time + learn_time)
                    ),
                    "experiment_save_dir": self.args.output_dir,
                    "batch_idx": batch_idx,
                    "num_total_batches": args.num_total_batches,
                }

                for key, value in ep_infos.items():
                    log_dict[f"Episode/{key}"] = value

                # Add scheduled parameters to metrics
                for param_name, param_value in self.scheduled_params_dict.items():
                    log_dict[f"scheduled_params/{param_name}"] = param_value

                if hasattr(self.policy_model, "std"):
                    metrics["Policy/mean_noise_std"] = self.policy_model.std.mean().item()
                else:
                    metrics["Policy/mean_noise_std"] = 0.0
                self.append_to_log_dict(log_dict)
                metrics.update({f"Env/{k}": v for k, v in env_log_dict.items()})
                metrics.update(env_log_dict)
                metrics.update(log_dict)

                self.log(metrics)
                self.ep_infos.clear()

            self.lr_scheduler.step()

            del metrics, rollout_data
            gc.collect()
            torch.cuda.empty_cache()

            self.control = self.callback_handler.on_step_end(args, self.state, self.control)

            if self.control.should_training_stop:
                break

        if self.control.should_training_stop:
            return

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        if common.wandb_run_exists():
            wandb.finish()

    def sync_running_mean_std(self):
        """Synchronize observation running mean/std normalizers across all GPU processes.

        Syncs every step for the first 200 iterations (warm-up), then at
        ``sync_running_mean_std_freq`` intervals.
        """
        sync_running_mean_std_freq = self.env.config.get("sync_running_mean_std_freq", 1)
        if self.state.global_step < 200 or (
            sync_running_mean_std_freq > 0
            and (self.state.global_step + 1) % sync_running_mean_std_freq == 0
        ):
            if (
                hasattr(self.policy_model, "use_running_mean_std")
                and self.policy_model.use_running_mean_std
            ):
                # print(f"Syncing policy running mean std at global step {self.state.global_step}")
                self.accelerator.wait_for_everyone()
                self.policy_model.running_mean_std.sync_across_gpus(self.accelerator)
            if (
                hasattr(self.value_model, "use_running_mean_std")
                and self.value_model.use_running_mean_std
            ):
                self.accelerator.wait_for_everyone()
                self.value_model.running_mean_std.sync_across_gpus(self.accelerator)

    def sync_adaptive_sampling(self):
        """Synchronize adaptive motion sampling weights across GPU processes."""
        sync_adaptive_sampling_all_gpus_freq = self.env.config.get(
            "sync_adaptive_sampling_all_gpus_freq", 200
        )
        sync_across_gpus = (
            sync_adaptive_sampling_all_gpus_freq > 0
            and (self.state.global_step + 1) % sync_adaptive_sampling_all_gpus_freq == 0
        )
        if hasattr(self.env, "sync_and_compute_adaptive_sampling"):
            self.env.sync_and_compute_adaptive_sampling(
                self.accelerator, sync_across_gpus=sync_across_gpus
            )

    def append_to_log_dict(self, log_dict):
        """Hook for subclasses to inject additional entries into the per-iteration log dict."""
        pass

    def _eval_mode(self):
        """Switch models and environment to evaluation mode."""
        self.model.eval()
        model = self.accelerator.unwrap_model(self.model)
        model.set_mode("eval")
        model.transform_eval()
        self.env.set_is_evaluating(is_evaluating=True, log_info=False)

    def _train_rollout_mode(self):
        """Switch to rollout collection mode: model in eval, transforms off, env in train."""
        self.model.eval()
        model = self.accelerator.unwrap_model(self.model)
        model.set_mode("train_rollout")
        model.transform_eval()
        if self.train_with_evaluating_env:
            self.env.set_is_evaluating(is_evaluating=True, log_info=False)
        else:
            self.env.set_is_evaluating(is_evaluating=False, log_info=False)

    def _train_mode(self):
        """Switch to gradient-update mode: model in train, transforms on."""
        self.model.train()
        model = self.accelerator.unwrap_model(self.model)
        model.set_mode("train")
        model.transform_train()
        if self.train_with_evaluating_env:
            self.env.set_is_evaluating(is_evaluating=True, log_info=False)
        else:
            self.env.set_is_evaluating(is_evaluating=False, log_info=False)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """  # noqa: D212
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics(  # noqa: F405
                    "train", start_time, num_tokens=self.state.num_input_tokens_seen
                )

        output = {**logs, **{"step": self.state.global_step}}  # noqa: PIE800
        self.state.log_history.append(output)

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _gradient_clipping(self):
        """Clip gradients and detect NaN/Inf, skipping the update if found.

        Returns:
            The global gradient norm after clipping, or None if NaN/Inf
            gradients were detected (signaling the caller to skip the
            optimizer step).
        """
        args = self.args
        model = self.model

        # Check for NaN/Inf in gradients
        for name, param in model.named_parameters():
            if param.grad is not None and (
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            ):
                print(  # noqa: T201
                    f"[Rank {self.accelerator.process_index}] NaN/Inf grad in {name}, norm={param.grad.norm():.3e}"
                )
                self.optimizer.zero_grad()
                return None
        grad_norm = None
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            # deepspeed does its own clipping

            if is_sagemaker_mp_enabled() and args.fp16:  # noqa: F405
                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
            elif self.use_apex:
                # Revert to normal clipping otherwise, handling Apex or full precision
                _grad_norm = nn.utils.clip_grad_norm_(
                    amp.master_params(self.optimizer),  # noqa: F405
                    args.max_grad_norm,
                )
            else:
                _grad_norm = self.accelerator.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )

            if (
                transformers_utils.is_accelerate_available()
                and self.accelerator.distributed_type == accelerate.DistributedType.DEEPSPEED
            ):
                grad_norm = model.get_global_grad_norm()
                # In some cases the grad norm may not return a float
                if hasattr(grad_norm, "item"):
                    grad_norm = grad_norm.item()
            else:
                grad_norm = _grad_norm

        return grad_norm

    def _compute_returns(self, values, last_values, policy_state_dict):
        """Compute the returns and advantages for the given policy state.
        This function calculates the returns and advantages for each step in the
        environment based on the provided observations and policy state. It uses
        Generalized Advantage Estimation (GAE) to compute the advantages, which
        helps in reducing the variance of the policy gradient estimates.
        Args:
            values (torch.Tensor): The values for each step.
            last_values (torch.Tensor): The last values for the last step.
            policy_state_dict (dict): A dictionary containing the policy state
                          information, including 'values', 'dones',
                          and 'rewards'.
        Returns:
            tuple: A tuple containing:
            - returns (torch.Tensor): The computed returns for each step.
            - advantages (torch.Tensor): The normalized advantages for each step.
        """  # noqa: D205, D410, D411
        device = self.accelerator.device
        advantage = 0

        dones = policy_state_dict["dones"]
        rewards = policy_state_dict["rewards"]

        dones = dones.to(device)
        rewards = rewards.to(device)

        returns = torch.zeros_like(values)

        num_steps = returns.shape[0]

        for step in reversed(range(num_steps)):
            if step == num_steps - 1:  # noqa: SIM108
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]

        # Compute and normalize the advantages
        advantages = returns - values
        if self.sync_advantage_normalization:
            # gather advantages from all processes before normalization
            advantages = self.accelerator.gather(advantages)
            advantages = (advantages - advantages.mean(dim=(0, 1), keepdim=True)) / (
                advantages.std(dim=(0, 1), keepdim=True) + 1e-8
            )
            # ungather advantages
            advantages = advantages.reshape(
                self.accelerator.num_processes, -1, *advantages.shape[1:]
            )[self.accelerator.process_index].to(device)
        else:
            advantages = (advantages - advantages.mean(dim=(0, 1), keepdim=True)) / (
                advantages.std(dim=(0, 1), keepdim=True) + 1e-8
            )
        return returns, advantages

    def _adjust_learning_rate_based_on_kl(self, kl_mean, optimizer):
        """Adjust the learning rate based on the KL divergence.

        This function implements a learning rate schedule that adjusts the learning rate
        based on the KL divergence between the current policy and the old policy.
        If the KL divergence is too high, the learning rate is decreased.
        If the KL divergence is too low, the learning rate is increased.

        Args:
            kl_mean (float): The mean KL divergence across all processes.
            optimizer (torch.optim.Optimizer): The optimizer to update.
        """
        if self.desired_kl is None:
            return

        if kl_mean > self.desired_kl * 2.0:
            new_lr = max(self.adaptive_lr_min, self.args.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            new_lr = min(self.adaptive_lr_max, self.args.learning_rate * 1.5)
        else:
            new_lr = self.args.learning_rate
        self.args.learning_rate = new_lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.args.learning_rate

    def load_checkpoint(self, checkpoint_path, resume=False):  # noqa: D417
        """Load a checkpoint to restore model weights and optionally full training state.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint file.
            resume: If True, also restore optimizer state, LR scheduler,
                environment state, and trainer counters for seamless resumption.

        Returns:
            The loaded checkpoint dict.
        """
        print(f"Loading checkpoint from {checkpoint_path}")  # noqa: T201
        checkpoint = torch.load(
            checkpoint_path, map_location=self.accelerator.device, weights_only=False
        )

        # Load model state
        model = self.accelerator.unwrap_model(self.model)
        if "actor_model_state_dict" in checkpoint:
            model.policy.load_state_dict(checkpoint["actor_model_state_dict"])
        elif "policy_state_dict" in checkpoint:
            model.policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
        if "value_state_dict" in checkpoint and model.value_model is not None:
            model.value_model.load_state_dict(checkpoint["value_state_dict"])

        if resume:
            # Load optimizer state
            if (
                "optimizer_state_dict" in checkpoint
                and checkpoint["optimizer_state_dict"] is not None
            ):
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # Update learning rate if available
                if "args" in checkpoint and hasattr(checkpoint["args"], "learning_rate"):
                    self.args.learning_rate = checkpoint["args"].learning_rate
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.args.learning_rate

            # Load learning rate scheduler state
            if (
                "lr_scheduler_state_dict" in checkpoint
                and checkpoint["lr_scheduler_state_dict"] is not None
            ):
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

            if "env_state_dict" in checkpoint:
                self.env.load_env_state_dict(checkpoint["env_state_dict"])

            if "state" in checkpoint:
                for key, value in checkpoint["state"].__dict__.items():
                    if key in ["cur_reward_sum", "cur_episode_length"]:
                        cur_value = getattr(self, key)
                        if cur_value.shape != value.shape:
                            continue
                        setattr(self, key, value)
                    if key not in [
                        "stateful_callbacks",
                        "is_local_process_zero",
                        "is_world_process_zero",
                        "log_history",
                    ]:
                        setattr(self.state, key, value)

        print(f"Loaded checkpoint from step {checkpoint['state'].global_step}")  # noqa: T201
        return checkpoint

    def eval(self):
        """Run an infinite deterministic evaluation loop with the current policy.

        Resets the environment, then repeatedly queries the policy for mean
        actions (no sampling) and steps the environment.  Intended for
        interactive visualization; exits only when interrupted externally.
        """
        self._eval_mode()
        self.env.set_is_evaluating()
        self.model.policy.eval_mode()
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():  # noqa: SIM118
            obs_dict[obs_key] = obs_dict[obs_key].to(self.accelerator.device)

        self.callback_handler.on_step_end(self.args, self.state, self.control)

        with torch.no_grad():  # noqa: SIM117
            with models_utils.unwrap_model_for_generation(
                self.model,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as model:
                while True:
                    device = self.accelerator.device
                    policy_model = model.policy
                    value_model = model.value_model  # noqa: F841
                    policy_model.init_rollout()

                    policy_state_dict = {}  # noqa: F841
                    actor_state = {}
                    actions = policy_model.rollout(obs_dict=obs_dict)  # noqa: F841
                    action_mean = policy_model.action_mean.detach()

                    actor_state["actions"] = action_mean
                    results = self.env.step(actor_state)
                    obs_dict, rewards, dones, infos = (
                        results[0],
                        results[1],
                        results[2],
                        results[3],
                    )  # noqa: F841

                    for obs_key in obs_dict.keys():  # noqa: SIM118
                        obs_dict[obs_key] = obs_dict[obs_key].to(device)

    @torch.no_grad()
    def get_example_obs(self):
        """Reset the environment and return a CPU observation dict for inspection or ONNX tracing.

        Returns:
            Dict mapping observation keys to CPU tensors ``(num_envs, obs_dim)``.
        """
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():  # noqa: SIM118
            print(obs_key, sorted(self.env.config.obs.obs_dict[obs_key]))  # noqa: T201
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    @property
    def inference_model(self):
        return {"actor": self.model.policy, "critic": self.model.value_model}
