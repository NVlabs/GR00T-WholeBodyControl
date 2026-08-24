"""Fail-closed tests for the diagnostic MJLab/RSL-RL checkpoint boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gear_sonic.scripts import inspect_g1_23dof_mjlab_checkpoint as cli
from gear_sonic.tests.test_g1_23dof_artifact import (
    _write_trained_checkpoint,
)
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    load_safe_true23_checkpoint,
)
from gear_sonic.utils.g1_23dof_mjlab_bridge import (
    audit_mjlab_rsl_rl_checkpoint,
    write_mjlab_audit_report,
)


class _UnsafeResumeObject:
    pass


def _stock_actor() -> dict[str, torch.Tensor]:
    return {
        "obs_normalizer._mean": torch.zeros(100),
        "obs_normalizer._var": torch.ones(100),
        "mlp.0.weight": torch.zeros(512, 100),
        "mlp.0.bias": torch.zeros(512),
        "mlp.2.weight": torch.zeros(256, 512),
        "mlp.2.bias": torch.zeros(256),
        "mlp.4.weight": torch.zeros(128, 256),
        "mlp.4.bias": torch.zeros(128),
        "mlp.6.weight": torch.zeros(23, 128),
        "mlp.6.bias": torch.zeros(23),
        "distribution.std_param": torch.ones(1),
    }


def test_stock_mjlab_checkpoint_is_teacher_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_100.pt"
    torch.save(
        {
            "actor_state_dict": _stock_actor(),
            "critic_state_dict": {"mlp.0.weight": torch.zeros(1, 1)},
            "optimizer_state_dict": {},
            "iter": 100,
            "infos": {"env_state": {"common_step_counter": 2400}},
        },
        checkpoint,
    )
    report = audit_mjlab_rsl_rl_checkpoint(checkpoint)
    assert report["promotion_eligible"] is False
    assert report["conversion_permitted"] is False
    assert report["promotion_pt_written"] is False
    assert report["source_checkpoint"]["safe_weights_only_load"] is True
    assert report["source_checkpoint"]["format"] == (
        "rsl_rl_v5_actor_state_dict"
    )
    assert report["compatibility"]["architecture_compatible"] is False
    assert report["compatibility"]["iteration_threshold_met"] is True
    assert report["truthful_bridge_boundary"]["stock_checkpoint_role"] == (
        "teacher_or_diagnostic_only"
    )
    assert any(
        "monolithic MLP" in finding for finding in report["findings"]
    )


def test_exact_tensor_shapes_still_cannot_fabricate_lineage(
    tmp_path: Path,
) -> None:
    safe_checkpoint = tmp_path / "trained.promotion.pt"
    _write_trained_checkpoint(safe_checkpoint)
    policy_state = load_safe_true23_checkpoint(safe_checkpoint)[
        "policy_state_dict"
    ]
    rsl_checkpoint = tmp_path / "custom_model_100.pt"
    torch.save(
        {
            "actor_state_dict": policy_state,
            "critic_state_dict": {"weight": torch.zeros(1)},
            "optimizer_state_dict": {},
            "iter": 100,
            "infos": None,
        },
        rsl_checkpoint,
    )
    report = audit_mjlab_rsl_rl_checkpoint(rsl_checkpoint)
    assert report["compatibility"]["architecture_compatible"] is True
    assert report["compatibility"]["target_policy_state_sha256"]
    assert report["compatibility"]["training_lineage_verified"] is False
    assert report["conversion_permitted"] is False
    assert report["promotion_pt_written"] is False


def test_legacy_rsl_actor_is_detected_without_critic_copy(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": {
                "actor.0.weight": torch.zeros(4, 3),
                "actor.0.bias": torch.zeros(4),
                "critic.0.weight": torch.zeros(5, 3),
                "std": torch.ones(23),
            },
            "iter": 75,
        },
        checkpoint,
    )
    report = audit_mjlab_rsl_rl_checkpoint(checkpoint)
    assert report["source_checkpoint"]["format"] == (
        "rsl_rl_legacy_model_state_dict"
    )
    keys = {
        item["key"] for item in report["actor_state"]["tensors"]
    }
    assert keys == {"0.weight", "0.bias", "std"}
    assert not any(key.startswith("critic") for key in keys)


def test_scalar_int64_actor_tensor_is_hashed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "scalar.pt"
    torch.save(
        {
            "actor_state_dict": {
                "step": torch.tensor(7, dtype=torch.int64),
            },
            "iter": 7,
        },
        checkpoint,
    )
    report = audit_mjlab_rsl_rl_checkpoint(checkpoint)
    assert report["actor_state"]["parameter_count"] == 1
    assert report["actor_state"]["tensors"] == [
        {
            "key": "step",
            "dtype": "torch.int64",
            "shape": [],
            "parameter_count": 1,
        }
    ]
    assert len(report["actor_state"]["sha256"]) == 64


def test_unsafe_pickle_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unsafe.pt"
    torch.save(
        {
            "actor_state_dict": _stock_actor(),
            "resume_object": _UnsafeResumeObject(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="refuses arbitrary pickle"):
        audit_mjlab_rsl_rl_checkpoint(checkpoint)


def test_audit_report_refuses_overwrite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"actor_state_dict": _stock_actor(), "iter": 1},
        checkpoint,
    )
    report = audit_mjlab_rsl_rl_checkpoint(checkpoint)
    output = tmp_path / "audit.json"
    assert write_mjlab_audit_report(report, output) == output.resolve()
    assert b'"promotion_pt_written":false' in output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_mjlab_audit_report(report, output)


def test_cli_reports_no_promotion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "audit.json"
    report = {
        "compatibility": {"architecture_compatible": False},
        "truthful_bridge_boundary": {
            "stock_checkpoint_role": "teacher_or_diagnostic_only"
        },
    }
    calls: list[tuple] = []
    monkeypatch.setattr(
        cli,
        "audit_mjlab_rsl_rl_checkpoint",
        lambda *args, **kwargs: calls.append((args, kwargs)) or report,
    )
    monkeypatch.setattr(
        cli,
        "write_mjlab_audit_report",
        lambda value, path: calls.append((value, path)) or path,
    )
    assert (
        cli.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert "promotion eligible: false" in stdout
    assert "promotion checkpoint written: false" in stdout
    assert calls[0][0] == (checkpoint,)
