from __future__ import annotations

from types import SimpleNamespace

import torch

from gear_sonic.scripts.train_g1_23dof_mjlab_teleop_v13 import (
    _update_corpus_causal_command,
)


class _FakeCorpusCommand:
    def __init__(self) -> None:
        self.time_steps = torch.tensor([18, 19, 18], dtype=torch.long)
        self._env_clip_stop = torch.tensor([20, 20, 20], dtype=torch.long)
        self._causal_resampled = torch.zeros(3, dtype=torch.bool)
        self.robot_anchor_pos_w = torch.tensor([[1.0], [2.0], [3.0]])
        self.robot_anchor_quat_w = torch.tensor([[4.0], [5.0], [6.0]])
        self._causal_last_current_anchor_pos_w = torch.tensor([[10.0], [20.0], [30.0]])
        self._causal_last_current_anchor_quat_w = torch.tensor([[40.0], [50.0], [60.0]])
        self._causal_robot_anchor_pos_w = torch.zeros((3, 1))
        self._causal_robot_anchor_quat_w = torch.zeros((3, 1))
        self.cfg = SimpleNamespace(sampling_mode="uniform", adaptive_alpha=0.5)
        self.bin_failed_count = torch.zeros(1)
        self._current_bin_failed = torch.zeros(1)
        self.resampled_ids: list[int] = []
        self.refreshed = 0

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.resampled_ids = env_ids.tolist()
        self.time_steps[env_ids] = 12
        self._causal_resampled[env_ids] = True

    def _virtual_anchor_at_q9(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.full((len(env_ids), 1), 70.0),
            torch.full((len(env_ids), 1), 80.0),
        )

    def _refresh_targets_from_causal_anchor(self) -> None:
        self.refreshed += 1


def test_corpus_update_preserves_causal_anchor_refresh_and_clip_containment() -> None:
    command = _FakeCorpusCommand()

    _update_corpus_causal_command(command)

    assert command.time_steps.tolist() == [19, 12, 19]
    assert command.resampled_ids == [1]
    assert command._causal_robot_anchor_pos_w[:, 0].tolist() == [10.0, 70.0, 30.0]
    assert command._causal_robot_anchor_quat_w[:, 0].tolist() == [40.0, 80.0, 60.0]
    assert command._causal_last_current_anchor_pos_w[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert command._causal_last_current_anchor_quat_w[:, 0].tolist() == [4.0, 5.0, 6.0]
    assert command.refreshed == 1
    assert not bool(command._causal_resampled.any())
