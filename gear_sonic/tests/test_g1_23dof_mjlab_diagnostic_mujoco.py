from __future__ import annotations

import json
from pathlib import Path

import pytest

from gear_sonic.utils import g1_23dof_mjlab_diagnostic_mujoco as diagnostic


def test_bundle_paths_are_exact_and_extensionless(tmp_path: Path) -> None:
    prefix = tmp_path / "model_250"
    assert diagnostic.diagnostic_bundle_paths(prefix) == (
        (tmp_path / "model_250.encoder.onnx").resolve(),
        (tmp_path / "model_250.decoder.onnx").resolve(),
        (tmp_path / "model_250.diagnostic.json").resolve(),
    )
    with pytest.raises(ValueError, match="extensionless"):
        diagnostic.diagnostic_bundle_paths(tmp_path / "model_250.pt")


def test_domain_sample_is_deterministic_and_inside_training_ranges() -> None:
    first = diagnostic.deterministic_domain_sample(
        seed=1729, episode=3, enabled=True
    )
    second = diagnostic.deterministic_domain_sample(
        seed=1729, episode=3, enabled=True
    )
    assert first == second
    assert first["enabled"] is True
    assert len(first["base_com_offset_m"]) == 3
    assert all(-0.05 <= value <= 0.05 for value in first["base_com_offset_m"])
    assert len(first["encoder_bias_hardware_rad"]) == 23
    assert all(
        -0.01 <= value <= 0.01
        for value in first["encoder_bias_hardware_rad"]
    )
    assert 0.3 <= first["foot_tangential_friction"] <= 1.2
    assert first != diagnostic.deterministic_domain_sample(
        seed=1729, episode=4, enabled=True
    )


def test_disabled_domain_sample_is_exact_nominal() -> None:
    sample = diagnostic.deterministic_domain_sample(
        seed=1729, episode=0, enabled=False
    )
    assert sample == {
        "enabled": False,
        "base_com_offset_m": [0.0, 0.0, 0.0],
        "encoder_bias_hardware_rad": [0.0] * 23,
        "foot_tangential_friction": None,
    }


def test_profiles_bind_short_smoke_and_full_approved_coverage() -> None:
    assert diagnostic._PROFILE_CONTRACT["smoke"] == {  # noqa: SLF001
        "deterministic_seeds": [1729],
        "episodes_per_seed": 1,
        "steps_per_episode": 100,
        "seconds_per_episode": 2.0,
    }
    assert diagnostic._PROFILE_CONTRACT["full"] == {  # noqa: SLF001
        "deterministic_seeds": [1729, 2718, 3141],
        "episodes_per_seed": 22,
        "steps_per_episode": 250,
        "seconds_per_episode": 5.0,
    }
    assert set(diagnostic._SCENARIOS) == {  # noqa: SLF001
        "nominal",
        "push_50",
        "push_100",
        "domain_push_100",
    }


def test_seed_shards_are_ordered_validated_and_never_full() -> None:
    assert diagnostic._resolve_campaign_seeds(  # noqa: SLF001
        "full", [3141]
    ) == ([3141], False)
    assert diagnostic._resolve_campaign_seeds(  # noqa: SLF001
        "full", [3141, 1729, 2718]
    ) == ([1729, 2718, 3141], True)
    with pytest.raises(ValueError, match="non-empty and unique"):
        diagnostic._resolve_campaign_seeds("full", [1729, 1729])  # noqa: SLF001
    with pytest.raises(ValueError, match="outside"):
        diagnostic._resolve_campaign_seeds("full", [1])  # noqa: SLF001


def test_validation_failure_still_writes_fail_closed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_config(_path: Path) -> None:
        raise ValueError("deliberate config failure")

    monkeypatch.setattr(
        diagnostic.sim2sim,
        "load_sim2sim_config",
        fail_config,
    )
    output = tmp_path / "model_250.mujoco_smoke_diagnostic.json"
    report = diagnostic.run_mjlab_diagnostic_mujoco(
        checkpoint_path=tmp_path / "model_250.pt",
        encoder_path=tmp_path / "model_250.encoder.onnx",
        decoder_path=tmp_path / "model_250.decoder.onnx",
        metadata_path=tmp_path / "model_250.diagnostic.json",
        output_path=output,
        profile="smoke",
        config_path=tmp_path / "config.json",
        mjcf_path=tmp_path / "robot.xml",
    )
    assert report["computed_pass"] is False
    assert report["diagnostic_only"] is True
    assert report["deployment_ready"] is False
    assert report["promotion_eligible"] is False
    assert report["active_motor_control_authorized"] is False
    assert report["robot_or_network_commands_performed"] is False
    assert report["error"] == "ValueError: deliberate config failure"
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_report_and_trace_paths_are_new_only(tmp_path: Path) -> None:
    existing = tmp_path / "existing_diagnostic.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostic.run_mjlab_diagnostic_mujoco(
            checkpoint_path=tmp_path / "model.pt",
            encoder_path=tmp_path / "encoder.onnx",
            decoder_path=tmp_path / "decoder.onnx",
            metadata_path=tmp_path / "metadata.json",
            output_path=existing,
            profile="smoke",
        )


def test_output_name_cannot_claim_promotion_or_deployment(tmp_path: Path) -> None:
    for name in (
        "candidate_promotion_diagnostic.json",
        "deployment_diagnostic.json",
    ):
        with pytest.raises(ValueError, match="may not claim"):
            diagnostic.run_mjlab_diagnostic_mujoco(
                checkpoint_path=tmp_path / "model.pt",
                encoder_path=tmp_path / "encoder.onnx",
                decoder_path=tmp_path / "decoder.onnx",
                metadata_path=tmp_path / "metadata.json",
                output_path=tmp_path / name,
                profile="smoke",
            )


def test_runtime_source_has_no_robot_or_network_imports() -> None:
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "unitree_sdk",
        "import socket",
        "import zmq",
        "cyclonedds",
        "g1_23dof_mujoco_promotion",
    ):
        assert forbidden not in source


def test_full_report_verifier_rejects_partial_report(tmp_path: Path) -> None:
    path = tmp_path / "partial_full_diagnostic.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": diagnostic.REPORT_SCHEMA_VERSION,
                "kind": diagnostic.REPORT_KIND,
                "profile": "full",
                "computed_pass": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a passing complete"):
        diagnostic.verify_full_mjlab_diagnostic_report(path)
