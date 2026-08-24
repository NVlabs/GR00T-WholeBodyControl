from __future__ import annotations

from gear_sonic.scripts import collect_g1_true23_sonic_rank256_causal_recovery_score_v2 as v2


def test_retry_thresholds_preserve_nominal_and_fit_observed_push_support() -> None:
    assert v2.MIN_VALID_BY_GROUP == {"nominal": 100, "exact_impulse": 30}
    assert v2.MIN_VALID_BY_GROUP["nominal"] > v2.MIN_VALID_BY_GROUP["exact_impulse"]
