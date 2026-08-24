from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from gear_sonic.envs.mjlab import native124_selected_v2_causal_adaptation as causal_env
from gear_sonic.utils import g1_true23_native124_selected_v2_causal_parity as parity

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _baseline_termination(baseline: dict) -> dict:
    failure = baseline["failure"]
    return {
        "control_transition": 47,
        "q9_before": 58,
        "q9_after": 9,
        "done": 1,
        "names": ["ee_body_pos"],
        "composite_action": failure["composite_action"],
        "actor_observation_124": failure["candidate_shadow_record"]["selected_observation_124"],
    }


def test_durable_baseline_is_hash_locked_to_expected_failure_boundary() -> None:
    baseline = parity.load_durable_baseline(REPOSITORY_ROOT)

    assert len(baseline["records"]) == 47
    assert baseline["summary"]["q9_first"] == 11
    assert baseline["summary"]["q9_last_after"] == 58
    assert baseline["failure"]["step"] == 47
    assert baseline["failure"]["q9_before"] == 58
    assert baseline["failure"]["q9_after"] == 9


def test_selected_warmup_matches_old_plain_native_zero_transform() -> None:
    proof = parity.prove_warmup_action_equivalence(device="cpu")

    assert proof["passed"] is True
    assert proof["candidate_default_linf_rad"] <= parity.WARMUP_TARGET_ATOL
    assert proof["plain_zero_linf"] <= parity.WARMUP_TARGET_ATOL
    assert proof["safe_action_linf"] <= parity.WARMUP_TARGET_ATOL
    assert proof["final_target_linf_rad"] <= parity.WARMUP_TARGET_ATOL


def test_stable_comparator_accepts_baseline_and_rejects_action_drift() -> None:
    baseline = parity.load_durable_baseline(REPOSITORY_ROOT)
    termination = _baseline_termination(baseline)
    passing = parity.compare_replay_to_baseline(
        replay_records=baseline["records"],
        termination=termination,
        baseline=baseline,
    )
    assert passing["passed"] is True
    assert passing["mismatch_count"] == 0
    assert passing["semantic_boundary"]["exact_match"] is True
    assert passing["parity_characterization"] == ("bounded CUDA/Warp numerical parity; not bitwise determinism")
    assert passing["tolerance_calibration"]["fresh_exact_implementation_replays"] == 2

    drifted = copy.deepcopy(baseline["records"])
    drifted[0]["composite_action"]["teacher_composite_target_hardware"][4] += 1.0e-3
    failing = parity.compare_replay_to_baseline(
        replay_records=drifted,
        termination=termination,
        baseline=baseline,
    )
    assert failing["passed"] is False
    assert any(
        item["path"] == "records[0].composite_action.teacher_composite_target_hardware"
        for item in failing["mismatches"]
    )


def test_report_writer_is_scoped_and_never_overwrites(tmp_path: Path) -> None:
    evidence = tmp_path / "artifacts/g1_true23"
    evidence.mkdir(parents=True)
    output = evidence / "parity.json"
    report = {"kind": parity.PARITY_KIND, "passed": False}

    written = parity.write_report_new(output, report, repository_root=tmp_path)

    assert written == output.resolve()
    assert output.read_bytes() == parity.canonical_json_bytes(report)
    with pytest.raises(FileExistsError):
        parity.write_report_new(output, report, repository_root=tmp_path)
    with pytest.raises(ValueError, match="must stay under"):
        parity.write_report_new(tmp_path / "outside.json", report, repository_root=tmp_path)


def test_real_wsl_cuda_no_update_replay_matches_durable_failure() -> None:
    if causal_env._MJLAB_IMPORT_ERROR is not None or causal_env._WRAPPER_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab RSL runtime unavailable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA MJLab runtime unavailable")

    report = parity.run_no_update_causal_parity(repository_root=REPOSITORY_ROOT)

    assert report["passed"] is True, report["comparison"]["mismatches"][:3]
    assert report["verdict"] == "bounded_cuda_warp_numerical_parity"
    assert report["parity_characterization"] == ("bounded CUDA/Warp numerical parity; not bitwise determinism")
    assert report["comparison"]["semantic_boundary"]["exact_match"] is True
    assert report["comparison"]["mismatch_count"] == 0
    assert report["warmup_equivalence"]["candidate_default_linf_rad"] == 0.0
    assert report["warmup_equivalence"]["plain_zero_linf"] == 0.0
    assert report["warmup_equivalence"]["safe_action_linf"] == 0.0
    assert report["warmup_equivalence"]["final_target_linf_rad"] == 0.0
    assert report["policy"]["updates_performed"] == 0
    assert report["training_performed"] is False
    assert report["hardware_or_network_commands_performed"] is False
    assert report["summary"]["record_count"] == 47
    assert report["termination"]["control_transition"] == 47
    assert report["termination"]["q9_before"] == 58
    assert report["termination"]["q9_after"] == 9
    assert report["termination"]["names"] == ["ee_body_pos"]
