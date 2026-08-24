"""Causal-history runner boundary.

Training still uses the exact released low-latency network topology, but
checkpoints get a distinct filename namespace.  Existing future-profile ONNX
exporters accept only ``model_N.pt`` and therefore fail closed on these files.
"""

from __future__ import annotations

from pathlib import Path

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    CAUSAL_HISTORY_PROFILE,
)
from gear_sonic.trl.mjlab.runner import True23MjlabOnPolicyRunner


class CausalHistoryMjlabOnPolicyRunner(True23MjlabOnPolicyRunner):
    """Exact-policy PPO runner with non-relabelable causal checkpoints."""

    def _numbered_checkpoint_path(self, update_count: int) -> Path:
        if isinstance(update_count, bool) or not isinstance(update_count, int):
            raise ValueError("causal checkpoint update_count must be integer")
        if update_count < 0:
            raise ValueError("causal checkpoint update_count must be non-negative")
        return self.checkpoint_dir / f"causal_model_{update_count}.pt"

    @property
    def semantic_profile(self) -> str:
        return CAUSAL_HISTORY_PROFILE
