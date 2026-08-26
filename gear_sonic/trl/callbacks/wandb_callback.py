from transformers import TrainerCallback
import wandb

from gear_sonic.trl.utils.common import wandb_run_exists


class WandbCallback(TrainerCallback):
    """Callback to save model state_dict during training."""

    def __init__(
        self,
    ):
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):

        if state.is_world_process_zero and wandb_run_exists():
            logs["global_step"] = state.global_step
            wandb.log(logs, step=state.global_step)
