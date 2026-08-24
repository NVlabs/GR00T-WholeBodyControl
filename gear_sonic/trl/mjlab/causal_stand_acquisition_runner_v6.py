"""Explicit low-exploration runner for nominal causal stand acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from gear_sonic.trl.mjlab.causal_history_runner import (
    CausalHistoryMjlabOnPolicyRunner,
)

ACQUISITION_EXPLORATION_STD = 0.10
ACQUISITION_ENTROPY_COEFFICIENT = 0.0


class CausalStandAcquisitionRunnerV6(CausalHistoryMjlabOnPolicyRunner):
    """Apply one source-bound exploration-only initialization override."""

    def __init__(
        self,
        *args: Any,
        resolved_config: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        contract = resolved_config.get("causal_stand_acquisition_v6")
        if not isinstance(contract, Mapping):
            raise ValueError("v6 runner requires resolved acquisition contract")
        if (
            contract.get("exploration_std_before_first_rollout")
            != ACQUISITION_EXPLORATION_STD
            or contract.get("entropy_coefficient")
            != ACQUISITION_ENTROPY_COEFFICIENT
        ):
            raise ValueError("v6 runner exploration contract mismatch")
        agent = resolved_config.get("agent")
        algorithm = agent.get("algorithm") if isinstance(agent, Mapping) else None
        if (
            not isinstance(algorithm, Mapping)
            or algorithm.get("entropy_coef")
            != ACQUISITION_ENTROPY_COEFFICIENT
        ):
            raise ValueError("v6 executed agent entropy differs from contract")
        super().__init__(*args, resolved_config=resolved_config, **kwargs)
        actor = self.alg.get_policy()
        distribution = getattr(actor, "distribution", None)
        std_param = getattr(distribution, "std_param", None)
        if not isinstance(std_param, torch.Tensor) or tuple(std_param.shape) != (23,):
            raise ValueError("v6 actor lacks exact direct [23] exploration std")
        with torch.no_grad():
            std_param.fill_(ACQUISITION_EXPLORATION_STD)
        if not torch.all(std_param == ACQUISITION_EXPLORATION_STD):
            raise RuntimeError("v6 exploration std initialization failed")
