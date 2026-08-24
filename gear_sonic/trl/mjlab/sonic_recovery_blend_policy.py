"""Hash-bound genuine SONIC recovery/late-policy blend.

This composes two compatible causal H10 SONIC actors without resetting history
or substituting actions.  It is simulator-qualified only; export/deployment is
intentionally unavailable until a single-policy distillation is qualified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_23dof_contract import TARGET_DOF

RECOVERY_CHECKPOINT_PATH = Path(
    "/root/g1_true23_runs/causal_history_stand_acquisition_v3_smoke100/checkpoints/causal_model_100.pt"
)
RECOVERY_CHECKPOINT_SHA256 = "d13f47eff7348a7fce1277233a1d1795a2bafe12cd8000b2e101351c73c63bcc"
RECOVERY_POLICY_SHA256 = "cc2a31a5b3a23e27c8c109d8c4d14d8b2b5d61f12d558e817e71f8c96831cfef"
LATE_CHECKPOINT_PATH = Path("/root/g1_true23_runs/low_latency_recovery_smoke_v2_20260803/checkpoints/model_2.pt")
LATE_CHECKPOINT_SHA256 = "c5cf5f2325549dcf839e576784051dfb5ea13095ba4c12a66020b3e86e517621"
LATE_POLICY_SHA256 = "6223dcd1386aa88cfef41970f1648cf92dfe1c515549ddf244c4fff9aa49b61e"
BLEND_START_Q9 = 360
BLEND_DURATION_STEPS = 10


def smooth_blend_alpha(
    q9: torch.Tensor,
    *,
    start_q9: int = BLEND_START_Q9,
    duration_steps: int = BLEND_DURATION_STEPS,
) -> torch.Tensor:
    """Return smoothstep blend weight, shape ``[batch, 1]``."""

    if type(q9) is not torch.Tensor or q9.ndim != 1 or q9.dtype != torch.long:
        raise ValueError("recovery blend q9 must be int64 [batch]")
    if isinstance(start_q9, bool) or not isinstance(start_q9, int) or start_q9 < 0:
        raise ValueError("recovery blend start must be nonnegative integer")
    if isinstance(duration_steps, bool) or not isinstance(duration_steps, int) or duration_steps <= 0:
        raise ValueError("recovery blend duration must be positive integer")
    alpha = ((q9.to(torch.float32) - float(start_q9)) / float(duration_steps)).clamp(0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return alpha.unsqueeze(1)


def load_hash_bound_policy_state(
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_policy_sha256: str,
) -> dict[str, torch.Tensor]:
    """Load one exact weights-only genuine SONIC policy state."""

    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    if path.is_symlink():
        raise ValueError("recovery blend checkpoint may not be a symlink")
    if sha256_file(path) != expected_checkpoint_sha256:
        raise ValueError("recovery blend checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("policy_state_dict"), Mapping):
        raise ValueError("recovery blend checkpoint schema mismatch")
    state = {
        str(name): value.detach().cpu().to(torch.float32).contiguous().clone()
        for name, value in checkpoint["policy_state_dict"].items()
        if type(name) is str and type(value) is torch.Tensor
    }
    if len(state) != len(checkpoint["policy_state_dict"]):
        raise ValueError("recovery blend policy tensor schema mismatch")
    policy_sha = inspect_true23_policy_state(
        {"policy_state_dict": state},
        reference_profile="released_low_latency_step1_0p02s",
    )
    if policy_sha != expected_policy_sha256:
        raise ValueError("recovery blend policy-state hash mismatch")
    return state


def load_hash_bound_actor(
    checkpoint_path: str | Path,
    *,
    topology_checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_policy_sha256: str,
    device: str | torch.device,
) -> nn.Module:
    """Construct one exact SONIC actor from a bound checkpoint."""

    from gear_sonic.trl.mjlab.true23_actor import True23SonicActorModel

    state = load_hash_bound_policy_state(
        checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_policy_sha256=expected_policy_sha256,
    )
    actor = True23SonicActorModel(
        {
            "tokenizer": torch.zeros((1, 268), dtype=torch.float32),
            "policy": torch.zeros((1, 930), dtype=torch.float32),
        },
        {"actor": ["tokenizer", "policy"]},
        "actor",
        TARGET_DOF,
        warm_start_path=str(Path(topology_checkpoint_path).expanduser().resolve(strict=True)),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"},
        hidden_dims=(),
        activation="silu",
        obs_normalization=False,
    )
    std_names = [name for name in state if name.rsplit(".", 1)[-1] in {"std", "log_std"}]
    if len(std_names) != 1:
        raise ValueError("recovery blend actor requires one std tensor")
    std_name = std_names[0]
    network = {name: value for name, value in state.items() if name != std_name}
    actor.core.load_state_dict(network, strict=True)
    std = state[std_name].exp() if std_name.endswith("log_std") else state[std_name]
    with torch.no_grad():
        actor.distribution.std_param.copy_(std)
    actor = actor.to(device)
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    exported = actor.export_true23_policy_state()
    if (
        inspect_true23_policy_state(
            {"policy_state_dict": exported},
            reference_profile="released_low_latency_step1_0p02s",
        )
        != expected_policy_sha256
    ):
        raise RuntimeError("live recovery blend actor hash mismatch")
    actor.eval()
    return actor


class SonicRecoveryBlendPolicy(nn.Module):
    """Recovery actor before q360, smooth cross-fade, late actor after q370."""

    def __init__(
        self,
        recovery_actor: nn.Module,
        late_actor: nn.Module,
        raw_env: Any,
        *,
        start_q9: int = BLEND_START_Q9,
        duration_steps: int = BLEND_DURATION_STEPS,
    ) -> None:
        super().__init__()
        self.recovery_actor = recovery_actor
        self.late_actor = late_actor
        self.raw_env = raw_env
        self.start_q9 = start_q9
        self.duration_steps = duration_steps

    def forward(self, observations: Any, stochastic_output: bool = False) -> torch.Tensor:
        if stochastic_output is not False:
            raise ValueError("qualified recovery blend is deterministic only")
        q9 = self.raw_env.command_manager.get_term("motion").time_steps
        alpha = smooth_blend_alpha(q9, start_q9=self.start_q9, duration_steps=self.duration_steps)
        recovery = self.recovery_actor(observations, stochastic_output=False)
        late = self.late_actor(observations, stochastic_output=False)
        if recovery.shape != late.shape or recovery.ndim != 2 or recovery.shape[1] != TARGET_DOF:
            raise RuntimeError("recovery blend actor action ABI mismatch")
        result = recovery * (1.0 - alpha) + late * alpha
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("recovery blend emitted non-finite action")
        return result

    def export_true23_policy_state(self) -> Mapping[str, torch.Tensor]:
        raise RuntimeError("two-actor recovery blend requires distillation before deployment export")


__all__ = [
    "BLEND_DURATION_STEPS",
    "BLEND_START_Q9",
    "LATE_CHECKPOINT_PATH",
    "LATE_CHECKPOINT_SHA256",
    "LATE_POLICY_SHA256",
    "RECOVERY_CHECKPOINT_PATH",
    "RECOVERY_CHECKPOINT_SHA256",
    "RECOVERY_POLICY_SHA256",
    "SonicRecoveryBlendPolicy",
    "load_hash_bound_actor",
    "load_hash_bound_policy_state",
    "smooth_blend_alpha",
]
