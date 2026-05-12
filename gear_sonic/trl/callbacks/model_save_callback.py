import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys

from loguru import logger
import torch
from transformers import TrainerCallback
import wandb

from gear_sonic.trl.utils.common import wandb_run_exists


class ModelSaveCallback(TrainerCallback):
    """Callback to save model state_dict during training."""

    def __init__(self, save_dir, save_frequency=1000, save_last_frequency=50, max_disk_usage=None):
        """
        Args:
            save_dir (str): Directory to save model checkpoints
            save_frequency (int): Save model every N steps
            max_disk_usage (float): Maximum disk usage in TB
        """
        self.save_dir = Path(save_dir)
        self.save_frequency = save_frequency
        self.save_last_frequency = save_last_frequency
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_last_only = False
        self.max_disk_usage = max_disk_usage

    def check_disk_usage(self):
        """Check if current working directory has more than 0.9TB of usage."""
        if self.save_last_only or self.max_disk_usage is None:
            return

        try:
            result = subprocess.run(["du", "-sb", "."], capture_output=True, text=True, check=True)
            size_bytes = int(result.stdout.split()[0])
            size_tb = size_bytes / (1024**4)
            if size_tb > self.max_disk_usage:
                self.save_last_only = True
                logger.info(
                    f"Directory size {size_tb:.2f}TB > {self.max_disk_usage}TB, setting save_last_only=True"
                )
                if wandb_run_exists():
                    wandb.alert(
                        title="Disk usage warning",
                        text=f"Directory size {size_tb:.2f}TB > {self.max_disk_usage}TB, setting save_last_only=True",
                        level="WARN",
                    )
        except Exception as e:
            logger.error(f"Error checking disk usage: {e}. Skip!")

    def on_step_end(self, args, state, control, **kwargs):
        """Save model state_dict at the end of each step if frequency matches."""
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        lr_scheduler = kwargs.get("lr_scheduler")
        env = kwargs.get("env")

        if state.is_world_process_zero and not env.is_evaluating:
            # Only save regular checkpoints if save_last_only is False
            if not self.save_last_only and state.global_step % self.save_frequency == 0:
                env_state_dict = env.get_env_state_dict()
                ModelSaveCallback.save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    state,
                    env_state_dict,
                    args,
                    f"{self.save_dir}/model_step_{state.global_step:06d}.pt",
                )

            # Always save last checkpoint every 50 steps
            if state.global_step % self.save_last_frequency == 0:
                env_state_dict = env.get_env_state_dict()
                ModelSaveCallback.save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    state,
                    env_state_dict,
                    args,
                    f"{self.save_dir}/last.pt",
                )
                # self.export_policy_to_onnx(env, state, model)

    def get_example_obs(self, env):
        obs_dict = copy.deepcopy(env.obs_buf_dict)
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()[0:1]
        return obs_dict

    def export_policy_to_onnx(self, env, state, model):
        checkpoint_path = os.path.join(env.config.experiment_dir, "last.pt")
        cmd = [
            sys.executable,
            "gear_sonic/eval_agent_trl.py",
            f"+checkpoint={checkpoint_path}",
            "+num_envs=1",
            "+headless=true",
            "+export_onnx_only=true",
        ]
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=os.getcwd())
        onnx_last_path = os.path.join(env.config.experiment_dir, "exported", "last.onnx")
        onnx_step_path = os.path.join(
            env.config.experiment_dir, "exported", f"model_step_{state.global_step:06d}.onnx"
        )
        shutil.copy(onnx_last_path, onnx_step_path)

    @classmethod
    def save_checkpoint(
        cls, model, optimizer, lr_scheduler, state, env_state_dict, args, save_path
    ):
        if model is not None:
            # Save model, optimizer, scheduler and training state
            _state = copy.deepcopy(state)
            _state.__dict__.pop("log_history")
            checkpoint = {
                "policy_state_dict": model.policy.state_dict(),
                "value_state_dict": (
                    model.value_model.state_dict() if model.value_model is not None else None
                ),  # Value model is not always preset (e.g. for distillation)
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "lr_scheduler_state_dict": (
                    lr_scheduler.state_dict() if lr_scheduler is not None else None
                ),
                "state": _state,
                "args": args,
                "env_state_dict": env_state_dict,
            }

            if hasattr(model, "disc_model") and model.disc_model is not None:
                checkpoint["disc_state_dict"] = model.disc_model.state_dict()

            import tempfile
            import time

            save_dir = os.path.dirname(save_path)

            for attempt in range(5):
                try:
                    # Save to a temp file first, then atomically rename
                    # This prevents corrupted partial checkpoints on filesystem failures
                    with tempfile.NamedTemporaryFile(
                        dir=save_dir, delete=False, suffix=".pt.tmp"
                    ) as tmp_file:
                        tmp_path = tmp_file.name

                    torch.save(checkpoint, tmp_path)

                    # Atomic rename (os.replace is atomic on POSIX systems)
                    os.replace(tmp_path, save_path)
                    print(f"Saved model checkpoint to {save_path}")
                    break
                except Exception as e:
                    # Clean up temp file if it exists
                    if "tmp_path" in locals() and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except:
                            pass

                    if attempt == 4:  # Last attempt
                        print(f"Failed to save checkpoint after 5 attempts. Error: {e}")
                    print(f"Attempt {attempt + 1} failed to save checkpoint. Retrying...")
                    time.sleep(
                        5
                    )  # Wait a bit before retrying (helps with transient filesystem issues)
