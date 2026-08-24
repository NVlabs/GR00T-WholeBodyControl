"""Higher-throughput final-affine trust-region runner v10."""

from gear_sonic.trl.mjlab.causal_final_affine_projected_runner_v8 import (
    CausalFinalAffineProjectedRunnerV8,
)

V10_FIXED_LEARNING_RATE = 5.0e-7
V10_ALLOWED_CHECKPOINT_UPDATES = frozenset({0, 10, 25, 50, 100})


class CausalFinalAffineProjectedRunnerV10(CausalFinalAffineProjectedRunnerV8):
    """Retain exact v8 projection while running 100 stronger updates."""

    required_contract_key = "causal_final_affine_projected_v10"
    fixed_learning_rate = V10_FIXED_LEARNING_RATE
    allowed_checkpoint_updates = V10_ALLOWED_CHECKPOINT_UPDATES
    runtime_schema = "g1_true23_causal_final_affine_projected_runtime_v10"
    runtime_filename = "v10_runtime_contract.json"
    telemetry_prefix = "V10"
