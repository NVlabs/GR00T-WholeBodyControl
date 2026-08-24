from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.scripts.diagnose_g1_true23_native124_selected_v2_reset_seam import (
    _parser,
)
import gear_sonic.utils.g1_true23_native124_selected_v2_reset_seam_diagnostic as diagnostic

REPO_ROOT = Path(__file__).resolve().parents[2]


def _request(tmp_path: Path) -> diagnostic.TrainingResetSeamDiagnosticRequest:
    evidence = tmp_path / "artifacts" / "g1_true23"
    evidence.mkdir(parents=True)
    checkpoint = tmp_path / "warm.pt"
    checkpoint.write_bytes(b"warm")
    return diagnostic.TrainingResetSeamDiagnosticRequest(
        repository_root=tmp_path,
        warm_checkpoint=checkpoint,
        expected_warm_sha256="0" * 64,
        output=Path("artifacts/g1_true23/reset_seam.json"),
    )


def test_request_and_writer_are_artifact_scoped_exclusive_and_nonqualifying(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    scope = diagnostic.diagnostic_scope()
    assert scope == {
        "classification": "training_reset_seam_diagnostic",
        "qualification_performed": False,
        "candidate_decision_emitted": False,
        "promotion_decision_emitted": False,
        "deployment_decision_emitted": False,
    }
    report = diagnostic.failure_report(RuntimeError("boom"), request)

    output = diagnostic.write_training_reset_seam_diagnostic_new(request, report)

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["kind"] == diagnostic.DIAGNOSTIC_KIND
    assert written["completed"] is False
    assert written["diagnostic_scope"] == scope
    assert "qualification" not in written
    assert "passed" not in written
    with pytest.raises(FileExistsError):
        diagnostic.write_training_reset_seam_diagnostic_new(request, report)

    outside = diagnostic.TrainingResetSeamDiagnosticRequest(
        repository_root=tmp_path,
        warm_checkpoint=request.warm_checkpoint,
        expected_warm_sha256="0" * 64,
        output=Path("outside.json"),
    )
    with pytest.raises(ValueError, match="must stay under"):
        _ = outside.output_path


def test_exact_ee_error_capture_uses_named_z_termination_contract() -> None:
    body_names = ("torso", "left_ankle", "right_wrist")
    reference = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 2.0, 0.5], [2.0, 3.0, 1.5]]],
        dtype=torch.float32,
    )
    measured = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.1, 1.8, 0.7], [2.1, 3.2, 1.1]]],
        dtype=torch.float32,
    )
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=body_names),
        body_pos_relative_w=reference,
        robot_body_pos_w=measured,
    )
    raw_env = SimpleNamespace(
        cfg=SimpleNamespace(
            terminations={
                "ee_body_pos": SimpleNamespace(
                    params={
                        "command_name": "motion",
                        "threshold": 0.25,
                        "body_names": ("left_ankle", "right_wrist"),
                    }
                )
            }
        ),
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )

    result = diagnostic.capture_exact_ee_position_errors(raw_env)

    assert [item["name"] for item in result] == ["left_ankle", "right_wrist"]
    assert result[0]["command_body_index"] == 1
    assert result[0]["absolute_z_error_m"] == pytest.approx(0.2)
    assert result[0]["z_termination_breached"] is False
    assert result[1]["command_body_index"] == 2
    assert result[1]["error_measured_minus_reference_m"] == pytest.approx([0.1, 0.2, -0.4])
    assert result[1]["absolute_z_error_m"] == pytest.approx(0.4)
    assert result[1]["z_termination_breached"] is True


def test_reset_actor_proof_requires_virtual_q9_and_zero_previous_action() -> None:
    actor = torch.randn(1, 124, dtype=torch.float32)
    actor[:, -23:] = 0.0
    diagnostics = SimpleNamespace(
        reset_virtual_torso_mask=torch.ones(1, dtype=torch.bool),
        previous_effective_selected_raw_hardware=torch.zeros(
            1,
            23,
            dtype=torch.float32,
        ),
    )

    proof = diagnostic.prove_reset_actor_observation(
        {"actor": actor},
        diagnostics,
    )

    assert proof["reset_virtual_torso_mask"] is True
    assert proof["actor_previous_action_slice_is_zero"] is True
    actor[:, -1] = 1.0
    with pytest.raises(RuntimeError, match="zero-prior"):
        diagnostic.prove_reset_actor_observation({"actor": actor}, diagnostics)


def test_preflight_rejects_wrong_hash_before_simulator(tmp_path: Path) -> None:
    checkpoint = tmp_path / "wrong.pt"
    checkpoint.write_bytes(b"wrong")
    request = diagnostic.TrainingResetSeamDiagnosticRequest(
        repository_root=REPO_ROOT,
        warm_checkpoint=checkpoint,
        expected_warm_sha256="0" * 64,
        output=Path(f"artifacts/g1_true23/{tmp_path.name}_reset_seam_never_written.json"),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        diagnostic.preflight_training_reset_seam_diagnostic(request)


def test_cli_exposes_only_hash_bound_paths_and_no_warmup_controls() -> None:
    parser = _parser()
    destinations = {
        action.dest
        for action in parser._actions  # noqa: SLF001
        if action.dest not in {"help"}
    }
    assert destinations == {
        "repository_root",
        "warm_checkpoint",
        "expected_sha256",
        "output",
    }
    args = parser.parse_args(
        [
            "--warm-checkpoint",
            "warm.pt",
            "--expected-sha256",
            "0" * 64,
            "--output",
            "artifacts/g1_true23/reset.json",
        ]
    )
    assert args.warm_checkpoint == Path("warm.pt")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--warm-checkpoint",
                "warm.pt",
                "--expected-sha256",
                "0" * 64,
                "--output",
                "artifacts/g1_true23/reset.json",
                "--warmup-steps",
                "2",
            ]
        )


def test_runtime_source_contains_no_qualification_warmup_action_path() -> None:
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")
    assert "prove_warmup_action_equivalence" not in source
    assert "wrapped.step(action)" in source
    assert '"fixed_warmup_steps": 0' in source
