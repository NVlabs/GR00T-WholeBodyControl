"""Teleoperation training: multi-clip corpus, full decoder, frozen encoder.

Follows the same isolation pattern as the v10/v12 variants — install hooks that
swap in this variant's runner and environment behaviour, then delegate to the
shared causal trainer.

Two departures from the v12 lineage.

The motion source is a corpus rather than one clip. The stock command samples
episode starts uniformly across the whole timeline and only resamples on running
off the end of the motion, which on a concatenated corpus walks an episode from
the end of one clip straight into the next. The command installed here keeps
every episode inside the clip it started in, using the span sidecar written by
``build_g1_23dof_motion_corpus``.

The trainable set is the whole ``g1_dyn`` decoder at a learning rate that can
move it, instead of the final affine at 5e-7. See ``causal_teleop_runner_v13``.

Training authorizes nothing. Promotion still requires the MuJoCo campaign,
paired ONNX export, a live shadow, and gantry authorization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from gear_sonic.scripts import train_g1_23dof_mjlab_causal_history as base
from gear_sonic.trl.mjlab import causal_history_runner as runner_module
from gear_sonic.trl.mjlab.causal_teleop_runner_v13 import CausalTeleopRunnerV13

_THIS_FILE = Path(__file__).resolve()
_RUNNER_FILE = (
    _THIS_FILE.parents[1] / "trl" / "mjlab" / "causal_teleop_runner_v13.py"
)
_CORPUS_FILE = _THIS_FILE.parents[1] / "utils" / "g1_23dof_multi_motion.py"

# Set by main() before the environment is built.
_SPAN_SIDECAR: Path | None = None

# The causal history window spans ten source frames back from the anchor, so an
# episode starts this far into a clip and the whole window is drawn from that
# same clip rather than from whatever precedes it in the corpus.
_CAUSAL_HISTORY_LEAD_IN_FRAMES = 12

# The profile also needs a q10 forward-difference proof frame one step *after*
# the anchor. Episodes therefore have to stop short of the clip's last frame,
# or that proof frame is read from the following clip.
_CAUSAL_HISTORY_TRAIL_FRAMES = 2


def _update_corpus_causal_command(command: Any) -> None:
    """Advance one contained clip while preserving exact causal anchor state."""

    current_pos = command.robot_anchor_pos_w.clone()
    current_quat = command.robot_anchor_quat_w.clone()
    resampled = command._causal_resampled.clone()  # noqa: SLF001
    advancing = ~resampled
    command.time_steps[advancing] += 1
    exhausted = advancing & (command.time_steps >= command._env_clip_stop)  # noqa: SLF001
    exhausted_ids = exhausted.nonzero(as_tuple=False).flatten()
    if len(exhausted_ids) > 0:
        command._resample_command(exhausted_ids)  # noqa: SLF001
        resampled[exhausted_ids] = True
        advancing[exhausted_ids] = False

    if torch.any(advancing):
        command._causal_robot_anchor_pos_w[advancing] = (  # noqa: SLF001
            command._causal_last_current_anchor_pos_w[advancing]  # noqa: SLF001
        )
        command._causal_robot_anchor_quat_w[advancing] = (  # noqa: SLF001
            command._causal_last_current_anchor_quat_w[advancing]  # noqa: SLF001
        )
    resampled_ids = resampled.nonzero(as_tuple=False).flatten()
    if len(resampled_ids) > 0:
        buffered_pos, buffered_quat = command._virtual_anchor_at_q9(resampled_ids)  # noqa: SLF001
        command._causal_robot_anchor_pos_w[resampled_ids] = buffered_pos  # noqa: SLF001
        command._causal_robot_anchor_quat_w[resampled_ids] = buffered_quat  # noqa: SLF001

    command._refresh_targets_from_causal_anchor()  # noqa: SLF001
    command._causal_last_current_anchor_pos_w.copy_(current_pos)  # noqa: SLF001
    command._causal_last_current_anchor_quat_w.copy_(current_quat)  # noqa: SLF001
    command._causal_resampled.zero_()  # noqa: SLF001
    if command.cfg.sampling_mode == "adaptive":
        command.bin_failed_count = (
            command.cfg.adaptive_alpha * command._current_bin_failed  # noqa: SLF001
            + (1 - command.cfg.adaptive_alpha) * command.bin_failed_count
        )
        command._current_bin_failed.zero_()  # noqa: SLF001


def teleop_v13_training_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_causal_teleop_training_v13",
        "trainable_actor_scope": "decoders.g1_dyn.* plus exploration std",
        "frozen_actor_scope": "encoders.* (SONIC low-latency weights preserved)",
        "learning_rate": CausalTeleopRunnerV13.learning_rate_override,
        "motion_source": "multi_clip_corpus",
        "episode_clip_containment": True,
        "deployment_ready": False,
    }


def _install_corpus_command(span_sidecar: Path) -> None:
    """Constrain episode sampling so no episode crosses a clip boundary."""
    from gear_sonic.envs.mjlab import sonic_true23_causal_history as causal_task

    payload = json.loads(span_sidecar.read_text())
    spans = payload["spans"]
    starts_cpu = torch.tensor([int(s["start"]) for s in spans], dtype=torch.long)
    stops_cpu = torch.tensor(
        [int(s["start"]) + int(s["length"]) for s in spans], dtype=torch.long
    )
    base_command = causal_task.CausalHistoryMotionCommand

    class CorpusMotionCommand(base_command):  # type: ignore[misc,valid-type]
        def __init__(self, cfg: Any, env: Any) -> None:
            super().__init__(cfg, env)
            device = self.device
            self._clip_starts = starts_cpu.to(device)
            self._clip_stops = stops_cpu.to(device)
            total = int(self.motion.time_step_total)
            if int(self._clip_stops.max()) > total:
                raise ValueError(
                    "span sidecar exceeds the loaded motion length; "
                    "corpus and sidecar are mismatched"
                )
            owner = torch.zeros(total, dtype=torch.long, device=device)
            for index in range(self._clip_starts.numel()):
                owner[int(self._clip_starts[index]) : int(self._clip_stops[index])] = (
                    index
                )
            self._frame_owner = owner
            self._env_clip_stop = torch.full(
                (self.num_envs,), total, dtype=torch.long, device=device
            )
            self._assign_within_clips(
                torch.arange(self.num_envs, device=device)
            )

        def _assign_within_clips(self, env_ids: torch.Tensor) -> None:
            device = self.device
            count = int(env_ids.numel())
            if count == 0:
                return
            # The causal profile reads a window of past frames back to q0, so an
            # episode cannot start at the very first frame of a clip: there is no
            # history behind it, and the frames physically before it belong to a
            # different clip. Start far enough in that the whole window is inside
            # this clip.
            lead_in = _CAUSAL_HISTORY_LEAD_IN_FRAMES
            trail = _CAUSAL_HISTORY_TRAIL_FRAMES
            lengths = self._clip_stops - self._clip_starts
            eligible = torch.where(lengths > lead_in + trail + 1)[0]
            if eligible.numel() == 0:
                raise ValueError(
                    f"no clip has room for a {lead_in}-frame history lead-in "
                    f"plus a {trail}-frame proof margin; rebuild the corpus "
                    "with longer clips"
                )
            choice = eligible[torch.randint(eligible.numel(), (count,), device=device)]
            lo = self._clip_starts[choice] + lead_in
            # Stop short of the clip end so the q10 proof frame stays inside it.
            hi = self._clip_stops[choice] - trail
            room = (hi - lo - 1).clamp(min=0)
            offset = (torch.rand(count, device=device) * (room + 1)).long()
            self.time_steps[env_ids] = lo + torch.minimum(offset, room)
            self._env_clip_stop[env_ids] = hi

        def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
            self._assign_within_clips(env_ids)
            self.metrics["sampling_entropy"][:] = 1.0
            self.metrics["sampling_top1_prob"][:] = 1.0 / max(self.bin_count, 1)
            self.metrics["sampling_top1_bin"][:] = 0.5

        def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
            # Adaptive bins assume one contiguous motion; on a corpus they
            # straddle clips, so use clip-constrained uniform instead.
            self._uniform_sampling(env_ids)

        # _resample_command is deliberately NOT overridden. Besides dispatching
        # to the sampling helpers below, it writes the robot's root and joint
        # state onto the reference pose for the newly chosen frame. Replacing it
        # would leave the robot in its previous pose while the reference jumped
        # to another clip, which reads as an enormous tracking error and
        # terminates the episode before the first step.

        def _update_command(self) -> None:
            _update_corpus_causal_command(self)

    causal_task.CausalHistoryMotionCommand = CorpusMotionCommand

    original_build = causal_task.CausalHistoryMotionCommandCfg.build

    def build_corpus(self: Any, env: Any) -> Any:  # noqa: ANN401
        return CorpusMotionCommand(self, env)

    causal_task.CausalHistoryMotionCommandCfg.build = build_corpus
    _ = original_build


def _install_termination_curriculum(threshold_m: float) -> None:
    """Relax the end-effector termination threshold for corpus training.

    The shipped 0.25 m threshold is tuned for neutral-stand recovery clips,
    where the policy starts already matching the reference. On arbitrary
    BONES-SEED motion a freshly reshaped 23-output decoder cannot hold that
    from the first step, so every episode terminates within a few steps and the
    policy optimizes early termination instead of tracking.

    Relaxing it buys episodes long enough to carry a learning signal. It is a
    training-time curriculum only and must be tightened back before any
    checkpoint is considered for validation.
    """
    from gear_sonic.envs.mjlab import sonic_true23_causal_history as causal_task

    original = causal_task.make_causal_history_recovery_env_cfg

    def with_relaxed_termination(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        cfg = original(*args, **kwargs)
        # Relax every distance gate together. Loosening only one moves the
        # binding constraint to the next: raising ee_body_pos alone made
        # anchor_pos terminations grow fivefold and become dominant.
        scale = float(threshold_m) / 0.25
        for name in ("ee_body_pos", "anchor_pos"):
            term = cfg.terminations.get(name)
            if term is None:
                continue
            shipped = float(term.params["threshold"])
            term.params["threshold"] = float(threshold_m)
            print(
                f"[v13] {name} threshold {shipped:.3f} m -> {threshold_m:.3f} m",
                flush=True,
            )
        orientation = cfg.terminations.get("anchor_ori")
        if orientation is not None:
            shipped_ori = float(orientation.params["threshold"])
            orientation.params["threshold"] = shipped_ori * scale
            print(
                f"[v13] anchor_ori threshold {shipped_ori:.3f} -> "
                f"{shipped_ori * scale:.3f}",
                flush=True,
            )
        print(
            "[v13] TRAINING CURRICULUM ONLY - tighten back to shipped values "
            "before any validation or promotion",
            flush=True,
        )
        return cfg

    causal_task.make_causal_history_recovery_env_cfg = with_relaxed_termination


def _install_reward_rebalance(barrier_scale: float) -> None:
    """Scale down the soft-limit barrier so tracking can drive the objective.

    Measured on the stage-1 run, penalties outweighed every tracking term
    combined by 12.8x, dominated by action_target_soft_limit_barrier starting
    near -17.5 per episode against a maximum total tracking reward of +5.0.
    PPO correctly optimized penalty avoidance and abandoned tracking: reward
    climbed while body-position error fell by 60% and episodes shortened.

    The barrier is a v11/v12 calibration scaffold, sized to keep a two-tensor
    probe at 5e-7 from moving. It is not a safety property of the deployed
    policy - joint limits are enforced by the termination terms, the MuJoCo
    validation campaign, and the promotion gates. Scaling it for training does
    not weaken any deployment check.
    """
    from gear_sonic.envs.mjlab import sonic_true23_causal_history as causal_task

    original = causal_task.make_causal_history_recovery_env_cfg

    def with_rebalanced_rewards(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        cfg = original(*args, **kwargs)
        for name in (
            "action_target_soft_limit_barrier",
            "action_target_soft_limit_barrier_hw10",
        ):
            term = cfg.rewards.get(name)
            if term is None:
                continue
            before = float(term.weight)
            term.weight = before * barrier_scale
            print(
                f"[v13] reward {name} weight {before:.3f} -> {term.weight:.3f}",
                flush=True,
            )
        tracking = sum(
            float(cfg.rewards[k].weight)
            for k in cfg.rewards
            if k.startswith("motion_") and float(cfg.rewards[k].weight) > 0
        )
        penalty = sum(
            abs(float(cfg.rewards[k].weight))
            for k in cfg.rewards
            if float(cfg.rewards[k].weight) < 0
        )
        print(
            f"[v13] reward balance: tracking +{tracking:.2f} vs penalties "
            f"-{penalty:.2f} (ratio {penalty / max(tracking, 1e-9):.2f}x)",
            flush=True,
        )
        return cfg

    causal_task.make_causal_history_recovery_env_cfg = with_rebalanced_rewards


def _install_isolated_v13_hooks(span_sidecar: Path) -> None:
    runner_module.CausalHistoryMjlabOnPolicyRunner = CausalTeleopRunnerV13
    base.CausalHistoryMjlabOnPolicyRunner = CausalTeleopRunnerV13
    _install_corpus_command(span_sidecar)

    source_files = list(base.CAUSAL_SOURCE_FILES)
    for path in (_RUNNER_FILE, _CORPUS_FILE, _THIS_FILE):
        if path not in source_files:
            source_files.append(path)
    base.CAUSAL_SOURCE_FILES = tuple(source_files)

    original = base._resolved_training_config

    def resolved_with_v13(*args: Any, **kwargs: Any) -> dict[str, Any]:
        resolved = original(*args, **kwargs)
        resolved["causal_teleop_training_v13"] = teleop_v13_training_contract()
        return resolved

    base._resolved_training_config = resolved_with_v13


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    sidecar: Path | None = None
    ee_threshold: float | None = None
    barrier_scale: float | None = None
    filtered: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--spans":
            sidecar = Path(argv[index + 1]).expanduser().resolve()
            index += 2
            continue
        if argv[index] == "--ee-termination-threshold":
            ee_threshold = float(argv[index + 1])
            index += 2
            continue
        if argv[index] == "--barrier-weight-scale":
            barrier_scale = float(argv[index + 1])
            index += 2
            continue
        filtered.append(argv[index])
        index += 1

    if sidecar is None:
        for index, token in enumerate(filtered):
            if token == "--motion-file":
                sidecar = Path(filtered[index + 1]).with_suffix(".spans.json")
                break
    if sidecar is None or not sidecar.is_file():
        raise SystemExit(
            "v13 needs the corpus span sidecar; pass --spans <file> "
            "(or place it beside the corpus as <corpus>.spans.json)"
        )

    payload = json.loads(sidecar.read_text())
    print(
        f"[v13] corpus {payload['clip_count']} clips, "
        f"{payload['total_frames']} frames "
        f"({payload['total_frames'] / payload['fps'] / 3600:.2f} h)",
        flush=True,
    )
    if ee_threshold is not None:
        _install_termination_curriculum(ee_threshold)
    if barrier_scale is not None:
        _install_reward_rebalance(barrier_scale)
    _install_isolated_v13_hooks(sidecar)
    return base.main(filtered)


if __name__ == "__main__":
    raise SystemExit(main())
