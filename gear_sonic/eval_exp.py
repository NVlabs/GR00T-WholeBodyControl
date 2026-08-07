#!/usr/bin/env python3  # noqa: EXE001
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

import glob
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import hydra
from loguru import logger
import omegaconf
import wandb
import yaml

from gear_sonic.trl.callbacks import im_eval_callback
from gear_sonic.utils import config_utils

config_utils.register_rl_resolvers()


class CheckpointEvaluator:
    """Continuously monitors an experiment directory for new checkpoints and evaluates them sequentially."""

    def __init__(self, config):
        self.config = config
        self.experiment_dir = Path(config.experiment_dir)
        self.evaluated_checkpoints: set[str] = set()
        self.shutdown_flag = False
        self.last_evaluation_time = time.time()
        self.evaluation_timeout = config.get("evaluation_timeout", 24 * 3600)
        self.eval_frequency = config.get("eval_frequency", None)
        self.eval_last_n = config.get("eval_last_n", None)

        if not self.experiment_dir.exists():
            raise ValueError(f"Experiment directory does not exist: {self.experiment_dir}")

        self.find_evaluated_checkpoints()

        self.wandb_run_id = None
        self.wandb_project = None
        self.wandb_entity = None
        self._load_wandb_config()

        logger.info(f"Monitoring experiment directory: {self.experiment_dir}")
        logger.info(f"Scan interval: {config.scan_interval} seconds")
        logger.info(f"Evaluation timeout: {self.evaluation_timeout / 3600:.1f} hours")
        if self.wandb_run_id:
            logger.info(f"Wandb logging enabled: run_id={self.wandb_run_id}")

        self._backfill_wandb()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):  # noqa: ARG002
        logger.info("Received shutdown signal. Stopping checkpoint monitoring...")
        self.shutdown_flag = True

    def _load_wandb_config(self):
        meta_path = self.experiment_dir / "meta.yaml"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            self.wandb_run_id = meta.get("wandb_run")

        config_path = self.experiment_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            config_path = self.experiment_dir / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    train_config = yaml.safe_load(f)
                wandb_cfg = train_config.get("wandb", {})
                self.wandb_project = train_config.get("project_name", "TRL_G1_Track")
                self.wandb_entity = wandb_cfg.get("wandb_entity", None)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not load training config for wandb: {e}")

        if self.wandb_project is None:
            self.wandb_project = "TRL_G1_Track"
        if self.wandb_entity is None:
            self.wandb_entity = None  # uses wandb default entity

    def _get_wandb_logged_steps(self) -> set[int]:
        logged_steps = set()
        if not self.wandb_run_id:
            return logged_steps

        try:
            api = wandb.Api(timeout=30)
            run = api.run(f"{self.wandb_entity}/{self.wandb_project}/{self.wandb_run_id}")
            hist = run.scan_history(
                keys=["eval/success/success_rate", "eval_step"],
                min_step=0,
                page_size=10000,
            )
            for row in hist:
                if row.get("eval/success/success_rate") is not None:
                    step = row.get("eval_step")
                    if step is not None:
                        logged_steps.add(int(step))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not query wandb for logged eval steps: {e}")

        return logged_steps

    def _backfill_wandb(self):
        if not self.wandb_run_id:
            self._load_wandb_config()
            if not self.wandb_run_id:
                logger.info("No wandb run ID available, skipping backfill")
                return

        eval_dir = self.experiment_dir / "eval"
        if not eval_dir.exists():
            return

        completed_steps = []
        for eval_subdir in sorted(eval_dir.iterdir()):
            if not eval_subdir.is_dir():
                continue
            try:
                step_num = int(eval_subdir.name)
            except ValueError:
                continue
            if (eval_subdir / "all_eval_finish.txt").exists():
                completed_steps.append((step_num, str(eval_subdir)))

        if not completed_steps:
            logger.info("No completed eval steps found on disk, nothing to backfill")
            return

        logged_steps = self._get_wandb_logged_steps()
        missing = [(step, path) for step, path in completed_steps if step not in logged_steps]

        if not missing:
            logger.info(
                f"All {len(completed_steps)} eval steps already logged to wandb, no backfill needed"
            )
            return

        logger.info(f"Backfilling {len(missing)}/{len(completed_steps)} eval steps to wandb")
        for eval_step, checkpoint_work_dir in missing:
            self._log_eval_to_wandb(eval_step, checkpoint_work_dir)

        logger.info("Backfill complete")

    def _log_eval_to_wandb(self, eval_step: int, checkpoint_work_dir: str):
        if not self.wandb_run_id:
            self._load_wandb_config()
            if not self.wandb_run_id:
                logger.warning("No wandb run ID found, skipping wandb logging")
                return

        eval_dir = Path(checkpoint_work_dir)
        if not eval_dir.exists():
            return

        try:
            wandb.init(
                id=self.wandb_run_id,
                project=self.wandb_project,
                entity=self.wandb_entity,
                resume="allow",
            )

            wandb.define_metric("eval_step")
            wandb.define_metric("eval/*", step_metric="eval_step")
            wandb.define_metric("videos_hard*", step_metric="eval_step")
            for subdir in sorted(eval_dir.iterdir()):
                if subdir.is_dir() and subdir.name != "train":
                    wandb.define_metric(f"{subdir.name}/*", step_metric="eval_step")

            all_metrics = {"eval_step": eval_step}

            for subdir in sorted(eval_dir.iterdir()):
                if not subdir.is_dir():
                    continue

                try:
                    metrics_file = subdir / "metrics_eval.json"
                    metrics_finish = subdir / "metrics_finish.txt"
                    if metrics_finish.exists() and metrics_file.exists():
                        self._log_metrics(eval_step, metrics_file)

                    render_finish = subdir / "render_finish.txt"
                    video_dir = subdir / "render_results"
                    if render_finish.exists() and video_dir.exists():
                        self._log_videos(eval_step, metrics_file, video_dir)
                except Exception as subdir_e:  # noqa: BLE001
                    logger.error(
                        f"Failed to log subdir {subdir.name} for step {eval_step}: {subdir_e}"
                    )

            wandb.log(all_metrics)
            wandb.finish()
            logger.info(f"Logged eval results to wandb for step {eval_step}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to log to wandb for step {eval_step}: {e}")
            try:
                wandb.finish()
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error finishing wandb: {e}")

    def _load_metrics(self, eval_step: int, metrics_file: Path) -> dict | None:
        try:
            with open(metrics_file) as f:
                metrics_eval = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error loading {metrics_file}")
            return None

        log_keys = metrics_eval.pop("log_keys", None)

        file_size_mb = metrics_file.stat().st_size / 1024 / 1024
        if file_size_mb > 20:
            metrics_eval.pop("eval/all_metrics_dict", None)
            metrics_eval.pop("eval/failed_metrics_dict", None)
            logger.info(
                f"Skipping per-motion dicts for {metrics_file.parent.name} ({file_size_mb:.0f} MB > 20 MB)"
            )
        else:
            if "eval/all_metrics_dict" in metrics_eval:
                metrics_eval["eval/all_metrics_dict"] = im_eval_callback.create_html_table(
                    metrics_eval["eval/all_metrics_dict"]
                )
            if "eval/failed_metrics_dict" in metrics_eval:
                metrics_eval["eval/failed_metrics_dict"] = im_eval_callback.create_html_table(
                    metrics_eval["eval/failed_metrics_dict"]
                )

        for key in ["failed_keys", "failed_idxes"]:
            metrics_eval.pop(key, None)

        metrics_eval["eval_step"] = eval_step

        if log_keys is not None:
            metrics_eval = {f"{log_keys}/{k}": v for k, v in metrics_eval.items()}
            metrics_eval["eval_step"] = eval_step

        return metrics_eval

    def _log_metrics(self, eval_step: int, metrics_file: Path):
        metrics = self._load_metrics(eval_step, metrics_file)
        if metrics:
            wandb.log(metrics)

    def _log_videos(self, eval_step: int, metrics_file: Path, video_dir: Path):
        if not video_dir.exists():
            return

        log_keys = None
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    log_keys = metrics.get("log_keys")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error getting log_keys from metrics file: {e}")

        video_files = sorted(
            [
                (i, f)
                for i, f in enumerate(sorted(video_dir.iterdir()))
                if f.is_file() and f.name.endswith(".mp4")
            ]
        )

        if not video_files:
            return

        prefix = f"videos_hard_{log_keys}" if log_keys else "videos_hard"
        wandb_videos = {
            f"{prefix}/{i:04d}": wandb.Video(str(video_file), format="mp4")
            for i, video_file in reversed(video_files)
        }
        wandb_videos["eval_step"] = eval_step
        wandb.log(wandb_videos)

    def find_evaluated_checkpoints(self):
        """Find all checkpoints that have been successfully evaluated."""
        eval_dir = self.experiment_dir / "eval"

        if not eval_dir.exists():
            logger.info("No eval directory found")
            return

        for eval_subdir in sorted(eval_dir.iterdir()):
            if eval_subdir.is_dir():
                metrics_finish_file = eval_subdir / "metrics_finish.txt"
                metrics_file = eval_subdir / "metrics_eval.json"
                render_finish_file = eval_subdir / "render_finish.txt"
                if (
                    metrics_finish_file.exists()
                    and metrics_file.exists()
                    and render_finish_file.exists()
                ):
                    try:
                        step_num = int(eval_subdir.name)
                        checkpoint_path = (
                            self.experiment_dir / f"model_step_{step_num:06d}.pt"
                        )
                        if checkpoint_path.exists():
                            self.evaluated_checkpoints.add(str(checkpoint_path))
                    except ValueError:
                        pass

        logger.info(f"Found {len(self.evaluated_checkpoints)} already evaluated checkpoints")

    def find_checkpoints(self) -> list[Path]:
        """Find all checkpoint files in the experiment directory."""
        checkpoint_pattern = str(self.experiment_dir / "model_step_*.pt")
        checkpoints = sorted(
            [Path(p) for p in glob.glob(checkpoint_pattern)],
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        return checkpoints

    def is_checkpoint_ready(self, checkpoint_path: Path) -> bool:
        """Check if a checkpoint is ready for evaluation (not being written)."""
        checkpoint_ready_delay = self.config.get("checkpoint_ready_delay", 60)
        mtime = checkpoint_path.stat().st_mtime
        age = time.time() - mtime
        return age > checkpoint_ready_delay

    def evaluate_checkpoint(
        self,
        checkpoint_path: Path,
        mode: str = "metrics",
        work_dir: str = None,
        num_render_videos: int = None,
        eval_step: int = None,  # noqa: ARG002
        eval_dataset: str = None,
        eval_mode: str = None,
    ):
        """Evaluate a single checkpoint using eval_agent_trl.py."""
        checkpoint_str = str(checkpoint_path)
        success = False

        mode_finish_file = os.path.join(work_dir, f"{mode}_finish.txt")
        metrics_file = os.path.join(work_dir, "metrics_eval.json")
        skip = os.path.exists(mode_finish_file)
        if skip and mode == "metrics" and not os.path.exists(metrics_file):
            logger.info(f"[{mode}] Not skipping since metrics file not found: {metrics_file}")
            skip = False
        if skip:
            logger.info(
                f"[{mode}] Skipping evaluation for checkpoint: {checkpoint_path} because it has already been evaluated"  # noqa: E501
            )
            return True

        try:
            logger.info(f"[{mode}] Starting evaluation for checkpoint: {checkpoint_path}")

            eval_callbacks = self.config.get("eval_callbacks", "im_eval")

            if mode == "metrics":
                cmd = f"accelerate launch gear_sonic/eval_agent_trl.py +checkpoint={checkpoint_str} +headless=True ++eval_callbacks={eval_callbacks} ++run_eval_loop=False"  # noqa: E501
                cmd += f" ++num_envs={self.config.num_eval_envs}"
                cmd += f" ++eval_output_dir={work_dir}"
                if eval_mode is not None:
                    cmd += f" ++use_encoder={eval_mode}"
                cmd += " ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False"
                cmd += " +manager_env/terminations=tracking/eval"
                if eval_dataset is not None:
                    cmd += (
                        f" +manager_env.commands.motion.motion_lib_cfg.motion_file={eval_dataset}"
                    )
                    cmd += f" +log_keys={Path(eval_dataset).name}_{eval_mode if eval_mode is not None else 'all'}"

            elif mode == "render":
                cmd = f"python -u gear_sonic/eval_agent_trl.py +checkpoint={checkpoint_str} +headless=True ++eval_callbacks={eval_callbacks} ++run_eval_loop=False"  # noqa: E501
                cmd += f" ++num_envs={num_render_videos}"
                cmd += f" ++metrics_file={metrics_file}"
                render_sort_by = self.config.get("render_sort_by", None)
                if render_sort_by is not None:
                    cmd += f" ++render_sort_by={render_sort_by}"
                cmd += f" ++manager_env.config.save_rendering_dir={work_dir}/render_results"
                cmd += " ++manager_env.config.render_results=True"
                cmd += " ++manager_env.config.env_spacing=10.0"
                cmd += " +manager_env/recorders=render"
                cmd += " ++manager_env.commands.motion.motion_lib_cfg.multi_thread=False"
                if eval_mode is not None:
                    cmd += f" ++use_encoder={eval_mode}"

                if eval_dataset is not None:
                    cmd += (
                        f" +manager_env.commands.motion.motion_lib_cfg.motion_file={eval_dataset}"
                    )

            extra_overrides = self.config.get("extra_overrides", [])
            for override in extra_overrides:
                cmd += f" {override}"

            logger.info(f"Running command: {cmd}")
            capture_output = self.config.get("capture_output", True)
            timeout_seconds = self.config.get("render_timeout", 3600) if mode == "render" else 21600
            proc = subprocess.Popen(
                cmd,
                shell=True,
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                text=True,
            )
            try:
                stdout_data, stderr_data = proc.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.communicate()
                logger.error(f"Evaluation timeout for checkpoint: {checkpoint_path}")
                return False
            result_returncode = proc.returncode
            result_stdout = stdout_data or ""
            result_stderr = stderr_data or ""

            success = result_returncode == 0 and os.path.exists(metrics_file)

            if mode == "render":
                found_videos = len(glob.glob(os.path.join(work_dir, "render_results", "*.mp4")))
                expected_videos = num_render_videos or self.config.get("num_render_videos", 64)
                if result_returncode == 0 and found_videos < expected_videos:
                    logger.warning(
                        f"[{mode}] Fewer videos than requested: found {found_videos}/{expected_videos} "
                        f"(OK if dataset has fewer unique motions)"
                    )
                elif found_videos == 0:
                    logger.error(f"[{mode}] No videos produced")
                    success = False

            if success:
                logger.info(f"[{mode}] Successfully evaluated checkpoint: {checkpoint_path}")
                self.last_evaluation_time = time.time()
            else:
                logger.error(f"[{mode}] Evaluation failed for checkpoint {checkpoint_path}")
                logger.error("=" * 20 + " stdout " + "=" * 20)
                logger.error(result_stdout)
                logger.error("=" * 20 + " stderr " + "=" * 20)
                logger.error(result_stderr)
                logger.error("=" * 20 + " end " + "=" * 20 + "\n")
                if not os.path.exists(metrics_file):
                    logger.error(f"[{mode}] Metrics file not found: {metrics_file}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error evaluating checkpoint {checkpoint_path}: {e}")
            return False

        return success

    def run(self):
        """Main monitoring loop."""
        single_pass = self.config.get("single_pass", False)
        if single_pass:
            logger.info("Running in single-pass mode...")
        else:
            logger.info("Starting checkpoint monitoring loop...")
        eval_datasets = self.config.get("eval_datasets", None)
        eval_modes = self.config.get("eval_modes", [None])
        num_render_videos = self.config.get("num_render_videos", 64)
        num_test_render_videos = self.config.get("num_test_render_videos", 32)
        while not self.shutdown_flag:
            try:
                checkpoints = self.find_checkpoints()

                new_checkpoints = []
                for cp in checkpoints:
                    cp_str = str(cp)
                    if (
                        self.eval_frequency is not None
                        and int(cp.stem.split("_")[-1]) % self.eval_frequency != 0
                    ):
                        continue
                    if cp_str not in self.evaluated_checkpoints and self.is_checkpoint_ready(cp):
                        new_checkpoints.append(cp)

                if self.eval_last_n is not None and len(new_checkpoints) > self.eval_last_n:
                    skipped = len(new_checkpoints) - self.eval_last_n
                    new_checkpoints = new_checkpoints[-self.eval_last_n :]
                    logger.info(
                        f"eval_last_n={self.eval_last_n}: skipping {skipped} earlier checkpoints"
                    )

                if single_pass and not new_checkpoints:
                    logger.info("Single-pass mode: no new checkpoints to evaluate, exiting")
                    break

                evaluation_success_count = 0
                for checkpoint in new_checkpoints:
                    if self.shutdown_flag:
                        break

                    eval_step = int(checkpoint.stem.split("_")[-1])
                    checkpoint_work_dir = os.path.join(
                        self.experiment_dir, "eval", f"{eval_step:06d}"
                    )
                    os.makedirs(checkpoint_work_dir, exist_ok=True)
                    logger.info(f"Found new checkpoint: {checkpoint}")

                    success = True
                    metrics_success = True

                    for mode in ["metrics", "render"]:
                        mode_work_dir = checkpoint_work_dir + "/train"
                        mode_success = self.evaluate_checkpoint(
                            checkpoint,
                            mode=mode,
                            work_dir=mode_work_dir,
                            eval_step=eval_step,
                            num_render_videos=num_render_videos,
                        )
                        if mode_success:
                            with open(os.path.join(mode_work_dir, f"{mode}_finish.txt"), "w") as f:
                                f.write(f"{mode}_finish")
                        success = success and mode_success
                        if mode == "metrics":
                            metrics_success = metrics_success and mode_success

                    if eval_datasets is not None:
                        for eval_dataset, eval_mode in itertools.product(eval_datasets, eval_modes):
                            for mode in ["metrics", "render"]:
                                mode_work_dir = (
                                    checkpoint_work_dir
                                    + f"/{Path(eval_dataset).name}_{eval_mode if eval_mode is not None else 'all'}"
                                )
                                mode_success = self.evaluate_checkpoint(
                                    checkpoint,
                                    mode=mode,
                                    work_dir=mode_work_dir,
                                    eval_step=eval_step,
                                    eval_dataset=eval_dataset,
                                    num_render_videos=num_test_render_videos,
                                    eval_mode=eval_mode,
                                )
                                if mode_success:
                                    with open(
                                        os.path.join(mode_work_dir, f"{mode}_finish.txt"), "w"
                                    ) as f:
                                        f.write(f"{mode}_finish")
                                success = success and mode_success
                                if mode == "metrics":
                                    metrics_success = metrics_success and mode_success

                    if success:
                        with open(
                            os.path.join(checkpoint_work_dir, "all_eval_finish.txt"), "w"
                        ) as f:
                            f.write("all_eval_finish")
                        self._log_eval_to_wandb(eval_step, checkpoint_work_dir)
                    elif metrics_success:
                        logger.warning(
                            f"Render failed for step {eval_step}, logging metrics-only to W&B"
                        )
                        self._log_eval_to_wandb(eval_step, checkpoint_work_dir)

                    if success:
                        self.evaluated_checkpoints.add(str(checkpoint))
                        evaluation_success_count += 1
                        if eval_step >= self.config.max_train_steps:
                            logger.info(
                                f"Reached max train steps: {eval_step} >= {self.config.max_train_steps}. Shutting down..."  # noqa: E501
                            )
                            self.shutdown_flag = True
                            break

                if new_checkpoints:
                    logger.info(f"Evaluated {evaluation_success_count} new checkpoints")
                    logger.info(f"Total evaluated checkpoints: {len(self.evaluated_checkpoints)}")

                if single_pass:
                    logger.info(
                        f"Single-pass mode: evaluated {evaluation_success_count} checkpoint(s), exiting"
                    )
                    break

                time_since_last_eval = time.time() - self.last_evaluation_time
                if time_since_last_eval > self.evaluation_timeout:
                    logger.info(
                        f"No checkpoints evaluated in {time_since_last_eval / 3600:.1f} hours. Shutting down..."
                    )
                    self.shutdown_flag = True
                    break

                time.sleep(self.config.scan_interval)

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt. Shutting down...")
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.scan_interval)

        logger.info("Checkpoint monitoring stopped.")


@hydra.main(config_path="config", config_name="eval_exp", version_base="1.1")
def main(config: omegaconf.OmegaConf) -> None:
    """Main function to start checkpoint monitoring and evaluation."""
    os.chdir(hydra.utils.get_original_cwd())

    single_pass = config.get("single_pass", False)

    experiment_dir = Path(config.experiment_dir)
    if not experiment_dir.exists():
        parent_dir = experiment_dir.parent
        prefix = experiment_dir.name
        logger.info(
            f"Experiment directory doesn't exist, looking for prefix match: {prefix}* in {parent_dir}"
        )

        while True:
            if parent_dir.exists():
                matches = sorted(
                    [d for d in parent_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
                )[::-1]
                if matches:
                    experiment_dir = None
                    for match in matches:
                        if (match / "meta.yaml").exists():
                            experiment_dir = match
                            logger.info(
                                f"Found matching directory with meta.yaml: {experiment_dir}"
                            )
                            break
                    if experiment_dir is None:
                        experiment_dir = matches[-1]
                        logger.info(
                            f"Found matching directory (no meta.yaml yet): {experiment_dir}"
                        )
                    config.experiment_dir = str(experiment_dir)
                    break

            if single_pass:
                logger.info("Single-pass mode: no matching directory found, exiting")
                return

            logger.info("No match found yet, waiting...")
            time.sleep(5)

    meta_file = os.path.join(config.experiment_dir, "meta.yaml")
    logger.info(f"Waiting for meta.yaml to exist: {meta_file}")

    if single_pass and not os.path.exists(meta_file):
        logger.info("Single-pass mode: meta.yaml not found, exiting")
        return
    while not os.path.exists(meta_file):
        time.sleep(1)
    meta = yaml.safe_load(open(meta_file))  # noqa: SIM115
    config.max_train_steps = meta["max_train_steps"]
    logger.info(f"Loaded meta: {meta}")

    hydra_log_path = os.path.join(config.experiment_dir, "eval_exp.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    evaluator = CheckpointEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
