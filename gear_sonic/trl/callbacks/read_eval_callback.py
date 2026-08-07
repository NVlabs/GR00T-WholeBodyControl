import json
from pathlib import Path

from transformers import TrainerCallback
import wandb

from gear_sonic.trl.callbacks.im_eval_callback import create_html_table
from gear_sonic.trl.utils.common import wandb_run_exists


class ReadEvalCallback(TrainerCallback):
    """Callback to read evaluation metrics and log them to wandb."""

    def __init__(self, eval_dir: str, check_interval: int = 1):
        """
        Initialize the callback.

        Args:
            experiment_dir: Path to the experiment directory where eval results are stored
            check_interval: How often to check for new evaluation results (in log steps)
        """
        super().__init__()
        self.eval_dir = Path(eval_dir)
        self.check_interval = check_interval
        self.last_check_step = 0
        self.state = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, logs
        return  # Disable this callback for now. We will let eval script to handle the wandb logging.

    def find_next_unread_checkpoint(self, current_eval_step: int = 0, mode: str = "metrics"):
        """
        Find the next unread checkpoint folder that has finished evaluation.

        Args:
            current_eval_step: Current evaluation step to start searching from

        Returns:
            tuple: (eval_step, eval_step_dir) or (None, None) if not found
        """
        if not self.eval_dir.exists():
            return []

        # Find all evaluation directories with 6-digit zero-padded names
        eval_step_dirs = []
        for d in self.eval_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                eval_step = int(d.name)
                if eval_step > current_eval_step:
                    eval_step_dirs.append((eval_step, d))

        if not eval_step_dirs:
            return []

        # Sort by step number to get the next one in sequence
        eval_step_dirs.sort(key=lambda x: x[0])

        # Find the first one that has finished evaluation
        for eval_step, eval_step_dir in eval_step_dirs:
            metrics_finish_file = eval_step_dir / "all_eval_finish.txt"
            if metrics_finish_file.exists():
                eval_dir_res = []
                for f in eval_step_dir.iterdir():
                    if f.is_dir():
                        mode_finish_file = f / f"{mode}_finish.txt"
                        if mode_finish_file.exists():
                            eval_dir_res.append((eval_step, f))
                return eval_dir_res

        return []

    def _check_and_log_eval_results(self, eval_step, eval_step_dir):
        """Check for new evaluation results and log them to wandb."""

        metrics_file = eval_step_dir / "metrics_eval.json"
        # Read and log the metrics
        with open(metrics_file) as f:
            try:
                metrics_eval = json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading metrics_eval.json for step {eval_step}")
                return

        if "log_keys" in metrics_eval:
            self.log_keys = metrics_eval["log_keys"]
        else:
            self.log_keys = None

        # Add eval_step if not already present
        if "eval_step" not in metrics_eval:
            metrics_eval["eval_step"] = eval_step

        metrics_eval["eval/all_metrics_dict"][
            "sampling_prob"
        ] = self.env._motion_lib._sampling_prob.cpu().numpy()
        metrics_eval["eval/failed_metrics_dict"][
            "sampling_prob"
        ] = self.env._motion_lib._sampling_prob.cpu().numpy()[metrics_eval["failed_idxes"]]
        metrics_eval["eval/all_metrics_dict"] = create_html_table(
            metrics_eval["eval/all_metrics_dict"]
        )
        metrics_eval["eval/failed_metrics_dict"] = create_html_table(
            metrics_eval["eval/failed_metrics_dict"]
        )

        # Log to wandb
        if self.accelerator.is_main_process and wandb_run_exists():
            if self.log_keys is not None:
                metrics_eval = {f"{self.log_keys}/{k}": v for k, v in metrics_eval.items()}
            wandb.log(metrics_eval)

        for key in ["failed_keys", "failed_idxes"]:
            # ZL: why do we need to do this?
            if key in metrics_eval:
                del metrics_eval[key]

        print(f"Logged evaluation metrics for step {eval_step} {eval_step_dir}")

    def _check_and_log_eval_render_results(self, eval_step, eval_step_dir):
        """Check for new evaluation results and log them to wandb."""

        video_dir = eval_step_dir / "render_results"
        metrics_file = eval_step_dir / "metrics_eval.json"

        if not video_dir.exists():
            print(f"No render_results directory for step {eval_step}, skipping render logging")
            return

        # Read and log the metrics
        with open(metrics_file) as f:
            try:
                metrics_eval = json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading metrics_eval.json for step {eval_step}")
                return

        if "log_keys" in metrics_eval:
            self.log_keys = metrics_eval["log_keys"]
        else:
            self.log_keys = None

        video_files = []
        for i, video_file in enumerate(sorted(video_dir.iterdir())):
            if video_file.is_file() and video_file.name.endswith(".mp4"):
                video_files.append((i, video_file))
        if self.log_keys is not None:
            wandb_videos = {
                f"videos_hard_{self.log_keys}/{i: 04d}": wandb.Video(str(video_file), format="mp4")
                for i, video_file in reversed(video_files)
            }
        else:
            wandb_videos = {
                f"videos_hard/{i: 04d}": wandb.Video(str(video_file), format="mp4")
                for i, video_file in reversed(video_files)
            }
        wandb_videos["eval_step"] = eval_step

        # Log to wandb
        if self.accelerator.is_main_process and wandb_run_exists():
            wandb.log(wandb_videos)

        print(f"Logged rendered video for step {eval_step}")
