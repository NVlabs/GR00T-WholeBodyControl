from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gear_sonic.scripts import screen_g1_true23_sonic_rank256_two_directions as screen


def test_contract_hash_coefficients_and_boundaries() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / screen.CONTRACT_RELATIVE_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == screen.CONTRACT_SHA256
    body = json.loads(path.read_text())
    assert body["screen"]["coefficients"] == [list(value) for value in screen.COEFFICIENTS]
    assert body["boundaries"]["training_transitions"] == 0
    assert body["boundaries"]["hardware_authorized"] is False


def test_assessment_requires_joint_improvement() -> None:
    records = []
    steps = {
        (0.0, 0.0): (486, 287),
        (1.0, 0.0): (500, 286),
        (2.0, 0.0): (490, 288),
        **{value: (480, 280) for value in screen.COEFFICIENTS[3:]},
    }
    for coefficients in screen.COEFFICIENTS:
        nominal, disturbance = steps[coefficients]
        for scenario, completed in (("nominal", nominal), ("disturbance", disturbance)):
            records.append(
                {
                    "survival_coefficient": coefficients[0],
                    "reward_coefficient": coefficients[1],
                    "scenario": scenario,
                    "completed_transitions": completed,
                    "policy_state_sha256": "a" * 64,
                    **{name: 0 for name in screen.engine.ZERO_COUNTS},
                }
            )
    assessment = screen._assess(records)  # noqa: SLF001
    assert assessment["candidate_selected"] is True
    assert assessment["selected_coefficients"] == [2.0, 0.0]


def test_run_helper_owns_evaluation_directory_creation() -> None:
    source = Path(screen.__file__).read_text()
    assert 'evaluations = run_dir / "evaluations"' in source
    assert "evaluations.mkdir()" not in source
    assert "any(evaluations.iterdir())" in source
