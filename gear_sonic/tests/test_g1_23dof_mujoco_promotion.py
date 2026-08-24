"""Adversarial gates for non-deployable true23 MuJoCo candidates."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.scripts import promote_g1_23dof_mujoco_candidate as cli
from gear_sonic.tests.test_g1_23dof_artifact import (
    _write_trained_checkpoint,
)
from gear_sonic.utils import (
    g1_23dof_mujoco_promotion as promotion,
    g1_23dof_mujoco_sim2sim as sim2sim,
)
from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from gear_sonic.utils.g1_23dof_mujoco_promotion import (
    CANDIDATE_KIND,
    export_true23_mujoco_candidate,
    promote_true23_mujoco_candidate,
    validate_true23_mujoco_report,
    verify_true23_mujoco_candidate,
    verify_true23_mujoco_promotion,
)


@pytest.fixture(scope="module")
def candidate_files(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path, Path]:
    root = tmp_path_factory.mktemp("true23-mujoco-candidate")
    checkpoint = root / "trained.promotion.pt"
    encoder = root / "candidate.encoder.onnx"
    decoder = root / "candidate.decoder.onnx"
    metadata = root / "candidate.metadata.json"
    _write_trained_checkpoint(checkpoint)
    export_true23_mujoco_candidate(
        checkpoint,
        encoder,
        decoder,
        metadata,
    )
    return checkpoint, encoder, decoder, metadata


def test_initialization_checkpoint_cannot_be_candidate(tmp_path: Path) -> None:
    initialization = (
        Path(__file__).resolve().parents[2]
        / "sonic_release/g1_23dof_rev_1_0_init.pt"
    )
    with pytest.raises(ValueError, match=r"trained weights-only \*\.promotion\.pt"):
        export_true23_mujoco_candidate(
            initialization,
            tmp_path / "encoder.onnx",
            tmp_path / "decoder.onnx",
            tmp_path / "metadata.json",
        )
    assert not list(tmp_path.iterdir())


def test_candidate_is_exact_trained_pair_but_never_deployable(
    candidate_files: tuple[Path, Path, Path, Path],
) -> None:
    checkpoint, encoder, decoder, metadata_path = candidate_files
    metadata = verify_true23_mujoco_candidate(
        encoder,
        decoder,
        metadata_path,
        checkpoint_path=checkpoint,
    )
    assert metadata["artifact_kind"] == CANDIDATE_KIND
    assert metadata["checkpoint_stage"] == "trained"
    assert metadata["decoder_output_dim"] == 23
    assert metadata["deployment_authorized"] is False
    assert metadata["deployment_ready"] is False
    assert metadata["sim_validation_passed"] is False
    assert "simulation_evidence" not in metadata
    assert metadata["asset_provenance"]["verified"] is True


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("deployment_authorized",), True, "falsely claims"),
        (("deployment_ready",), True, "falsely claims"),
        (("checkpoint_stage",), "checkpoint_initialization", "contract"),
        (("decoder_output_dim",), 29, "artifact contract"),
        (
            ("asset_provenance", "revision"),
            "0" * 40,
            "asset provenance",
        ),
        (
            ("training_evidence", "global_step"),
            1,
            "training evidence",
        ),
    ),
)
def test_candidate_metadata_tamper_fails_closed(
    candidate_files: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    checkpoint, encoder, decoder, metadata_path = candidate_files
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target = metadata
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    metadata.pop("metadata_payload_sha256")

    metadata["metadata_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(metadata)
    )
    tampered = tmp_path / "candidate.metadata.json"
    tampered.write_bytes(canonical_json_bytes(metadata))
    with pytest.raises(ValueError, match=message):
        verify_true23_mujoco_candidate(
            encoder,
            decoder,
            tampered,
            checkpoint_path=checkpoint,
            expected_filenames=(
                encoder.name,
                decoder.name,
                metadata_path.name,
            ),
        )


def test_candidate_onnx_byte_tamper_fails_hash_binding(
    candidate_files: tuple[Path, Path, Path, Path],
    tmp_path: Path,
) -> None:
    checkpoint, encoder, decoder, metadata_path = candidate_files
    tampered_encoder = tmp_path / encoder.name
    tampered_encoder.write_bytes(encoder.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="encoder_onnx_sha256"):
        verify_true23_mujoco_candidate(
            tampered_encoder,
            decoder,
            metadata_path,
            checkpoint_path=checkpoint,
        )


def test_candidate_refuses_overwrite(
    candidate_files: tuple[Path, Path, Path, Path],
) -> None:
    checkpoint, encoder, decoder, metadata = candidate_files
    with pytest.raises(FileExistsError, match="overwrite"):
        export_true23_mujoco_candidate(
            checkpoint,
            encoder,
            decoder,
            metadata,
        )


def test_checked_in_approval_pins_current_mujoco_material() -> None:
    approval, config, module, model, physics = (
        promotion._approved_sim2sim_material()  # noqa: SLF001
    )
    assert approval["promotion_enabled"] is True
    assert module.__version__ == "3.2.3"
    assert model.nu == 23
    assert config["coverage"]["deterministic_seeds"] == [1729, 2718, 3141]
    assert config["coverage"]["episodes_per_seed"] == 22
    assert physics["payload_sha256"]


def test_approval_rejects_promotion_verifier_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = tmp_path / "g1_23dof_mujoco_promotion.py"
    drifted.write_bytes(
        promotion._PROMOTION_SOURCE_PATH.read_bytes() + b"\n# drift\n"  # noqa: SLF001
    )
    monkeypatch.setattr(promotion, "_PROMOTION_SOURCE_PATH", drifted)
    with pytest.raises(
        ValueError,
        match="approved MuJoCo material changed: promotion_source_sha256",
    ):
        promotion._approved_sim2sim_material()  # noqa: SLF001


class _ZeroRuntime:
    def __init__(self, **_kwargs):
        pass

    def infer(self, encoder_input, proprio_history):
        assert len(encoder_input) == 267
        assert len(proprio_history) == 930
        return np.zeros(64), np.zeros(23)


def _summary_metrics(metrics: dict) -> dict:
    return {
        "run_count": 1,
        "episode_count": metrics["episode_count"],
        "record_count": metrics["record_count"],
        "termination_count": metrics["termination_count"],
        "nonfinite_count": metrics["nonfinite_count"],
        "joint_limit_violation_count": metrics[
            "joint_limit_violation_count"
        ],
        "min_base_height_m": metrics["min_base_height_m"],
        "max_tilt_rad": metrics["max_tilt_rad"],
        "max_tracking_rmse_rad": metrics["max_tracking_rmse_rad"],
        "max_abs_joint_velocity_radps": metrics[
            "max_abs_joint_velocity_radps"
        ],
        "max_abs_applied_torque_nm": metrics[
            "max_abs_applied_torque_nm"
        ],
        "max_abs_native_action": metrics["max_abs_native_action"],
        "max_abs_native_action_raw": metrics[
            "max_abs_native_action_raw"
        ],
        "max_action_saturation_fraction": metrics[
            "action_saturation_fraction"
        ],
        "minimum_recovery_fraction": metrics["recovery_fraction"],
        "max_recovery_time_s": metrics["max_recovery_time_s"],
    }


@pytest.fixture
def synthetic_promotion_report(
    candidate_files: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, tuple[Path, Path, Path, Path], dict]:
    checkpoint, encoder, decoder, metadata_path = candidate_files
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_path = (
        Path(__file__).resolve().parents[2]
        / sim2sim.DEFAULT_CONFIG_RELPATH
    )
    config = deepcopy(sim2sim.load_sim2sim_config(config_path))
    config["coverage"] = {
        "deterministic_seeds": [1729],
        "episodes_per_seed": 1,
        "seconds_per_episode": 0.06,
        "steps_per_episode": 3,
    }
    config["disturbance_schedule"] = {
        "apply_step": 1,
        "recovery_baseline_steps": 1,
        "recovery_stable_steps": 1,
        "recovery_margin": 0.1,
    }
    config["scenarios"] = {"nominal": {"disturbance_scale": 0.0}}
    config["termination"] = {
        "minimum_base_height_m": 0.1,
        "maximum_tilt_rad": 3.0,
        "maximum_joint_velocity_ratio": 100.0,
    }
    config["promotion_thresholds"] = {
        "termination_count": 0,
        "nonfinite_count": 0,
        "joint_limit_violation_count": 0,
        "minimum_recovery_fraction": 1.0,
        "maximum_recovery_time_s": 2.0,
        "maximum_tracking_rmse_rad": 5.0,
        "maximum_native_action_abs": 20.0,
        "maximum_action_saturation_fraction": 0.1,
    }
    mjcf_path = (
        Path(__file__).resolve().parents[2]
        / sim2sim.DEFAULT_MJCF_RELPATH
    )
    module, model, physics = sim2sim.prepare_mujoco_model(
        mjcf_path=mjcf_path,
        config=config,
    )
    reference = sim2sim.NeutralReference(
        model,
        module,
        config["initial_state"],
    )
    records = sim2sim.run_episode(
        module=module,
        model=model,
        runtime=_ZeroRuntime(),
        reference=reference,
        config=config,
        scenario="nominal",
        seed=1729,
        episode=0,
        disturbance_scale=0.0,
    )
    report_path = tmp_path / "report.json"
    trace = sim2sim._write_trace(  # noqa: SLF001
        output_path=report_path,
        scenario="nominal",
        seed=1729,
        records=records,
    )
    metrics = sim2sim.recompute_metrics(
        records,
        config=config,
        disturbance_scale=0.0,
    )
    assert sim2sim.metrics_pass(metrics, config)
    source = promotion._candidate_source_artifact(  # noqa: SLF001
        metadata=metadata,
        checkpoint_path=checkpoint,
        encoder_path=encoder,
        decoder_path=decoder,
        metadata_path=metadata_path,
    )
    approval = {
        "mujoco_version": module.__version__,
        "runner_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "mjcf_sha256": sha256_file(mjcf_path),
        "asset_manifest_sha256": metadata["asset_provenance"][
            "manifest_sha256"
        ],
    }
    run = {
        "scenario": "nominal",
        "seed": 1729,
        "episodes": 1,
        "steps_per_episode": 3,
        "disturbance_scale": 0.0,
        "computed_pass": True,
        "metrics": metrics,
        "trace": trace,
    }
    trace_manifest = [{"scenario": "nominal", "seed": 1729, **trace}]
    report = {
        "schema_version": sim2sim.REPORT_SCHEMA_VERSION,
        "kind": sim2sim.REPORT_KIND,
        "robot_model": "g1_23dof_rev_1_0",
        "checkpoint_stage": "trained",
        "diagnostic_only": False,
        "promotion_eligible": True,
        "computed_pass": True,
        "source_artifact": source,
        "producer": {
            "kind": sim2sim.PRODUCER_KIND,
            "version": sim2sim.PRODUCER_VERSION,
            "runner_sha256": approval["runner_sha256"],
            "runtime_sha256": approval["runtime_sha256"],
        },
        "simulator": {
            "name": "MuJoCo",
            "version": module.__version__,
            "mjcf_sha256": approval["mjcf_sha256"],
            "config_sha256": approval["config_sha256"],
            "approved_offline_inputs": True,
            "host": {
                "system": "test",
                "release": "test",
                "machine": "test",
                "processor": "test",
                "python": "3.10",
            },
            "compiled_model": sim2sim._compiled_model_contract(  # noqa: SLF001
                module,
                model,
            ),
            "physics_contract": deepcopy(physics),
            "asset_provenance": sim2sim._asset_provenance(),  # noqa: SLF001
        },
        "contract": sim2sim._contract_descriptor(),  # noqa: SLF001
        "reference_command": reference.descriptor(),
        "trace_manifest_sha256": sha256_bytes(
            canonical_json_bytes(trace_manifest)
        ),
        "runs": [run],
        "summary": {
            "computed_pass": True,
            "promotion_eligible": True,
            "thresholds": config["promotion_thresholds"],
            "metrics": _summary_metrics(metrics),
        },
    }
    report_path.write_bytes(canonical_json_bytes(report))
    monkeypatch.setattr(
        promotion,
        "verify_true23_mujoco_candidate",
        lambda *_args, **_kwargs: metadata,
    )
    monkeypatch.setattr(
        promotion,
        "_approved_sim2sim_material",
        lambda: (approval, config, module, model, physics),
    )
    monkeypatch.setattr(sim2sim, "True23PolicyRuntime", _ZeroRuntime)
    return report_path, candidate_files, report


def test_report_replays_raw_trace_before_promotion(
    synthetic_promotion_report,
) -> None:
    report_path, files, _report = synthetic_promotion_report
    checkpoint, encoder, decoder, metadata = files
    evidence = validate_true23_mujoco_report(
        report_path,
        checkpoint_path=checkpoint,
        encoder_path=encoder,
        decoder_path=decoder,
        metadata_path=metadata,
    )
    assert evidence["computed_pass"] is True
    assert evidence["trace_count"] == 1
    assert evidence["total_episodes"] == 1
    assert evidence["source_artifact"]["encoder_onnx_sha256"] == sha256_file(
        encoder
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("diagnostic_only",), True, "diagnostic"),
        (("promotion_eligible",), False, "diagnostic"),
        (
            ("source_artifact", "encoder_onnx_sha256"),
            "0" * 64,
            "different candidate",
        ),
        (
            ("simulator", "physics_contract", "control_hz"),
            49,
            "physics_contract",
        ),
        (("runs", 0, "metrics", "termination_count"), 1, "raw trace"),
    ),
)
def test_report_claim_tamper_fails_closed(
    synthetic_promotion_report,
    path: tuple[object, ...],
    value: object,
    message: str,
) -> None:
    report_path, files, report = synthetic_promotion_report
    checkpoint, encoder, decoder, metadata = files
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    report_path.write_bytes(canonical_json_bytes(report))
    with pytest.raises(ValueError, match=message):
        validate_true23_mujoco_report(
            report_path,
            checkpoint_path=checkpoint,
            encoder_path=encoder,
            decoder_path=decoder,
            metadata_path=metadata,
        )


def test_trace_semantic_tamper_rejected_even_after_rehash(
    synthetic_promotion_report,
) -> None:
    report_path, files, report = synthetic_promotion_report
    checkpoint, encoder, decoder, metadata = files
    trace_path = report_path.parent / report["runs"][0]["trace"]["file"]
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["base_height_m"] += 0.1
    payload = b"".join(canonical_json_bytes(record) for record in records)
    trace_path.write_bytes(payload)
    descriptor = report["runs"][0]["trace"]
    descriptor["sha256"] = sha256_bytes(payload)
    descriptor["payload_sha256"] = sha256_bytes(
        canonical_json_bytes(records)
    )
    manifest = [{"scenario": "nominal", "seed": 1729, **descriptor}]
    report["trace_manifest_sha256"] = sha256_bytes(
        canonical_json_bytes(manifest)
    )
    report_path.write_bytes(canonical_json_bytes(report))
    with pytest.raises(ValueError, match="base_height_m"):
        validate_true23_mujoco_report(
            report_path,
            checkpoint_path=checkpoint,
            encoder_path=encoder,
            decoder_path=decoder,
            metadata_path=metadata,
        )


def test_self_consistent_hashes_cannot_replace_deterministic_replay(
    synthetic_promotion_report,
) -> None:
    report_path, files, report = synthetic_promotion_report
    checkpoint, encoder, decoder, metadata = files
    trace_path = report_path.parent / report["runs"][0]["trace"]["file"]
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    # This field is finite, correctly shaped, and not used by aggregate metrics.
    # Rehashing every envelope used to make such a fabricated trace self-consistent.
    records[0]["base_linear_velocity_mps"][0] += 0.01
    payload = b"".join(canonical_json_bytes(record) for record in records)
    trace_path.write_bytes(payload)
    descriptor = report["runs"][0]["trace"]
    descriptor["sha256"] = sha256_bytes(payload)
    descriptor["payload_sha256"] = sha256_bytes(
        canonical_json_bytes(records)
    )
    report["trace_manifest_sha256"] = sha256_bytes(
        canonical_json_bytes(
            [{"scenario": "nominal", "seed": 1729, **descriptor}]
        )
    )
    report_path.write_bytes(canonical_json_bytes(report))
    with pytest.raises(ValueError, match="deterministic ONNX\\+MuJoCo replay"):
        validate_true23_mujoco_report(
            report_path,
            checkpoint_path=checkpoint,
            encoder_path=encoder,
            decoder_path=decoder,
            metadata_path=metadata,
        )


def test_promotion_sidecar_authorizes_bytes_not_active_motors(
    synthetic_promotion_report,
    tmp_path: Path,
) -> None:
    report_path, files, _report = synthetic_promotion_report
    checkpoint, encoder, decoder, metadata = files
    sidecar = tmp_path / "promotion.json"
    result = promote_true23_mujoco_candidate(
        sidecar,
        checkpoint_path=checkpoint,
        encoder_path=encoder,
        decoder_path=decoder,
        metadata_path=metadata,
        report_path=report_path,
    )
    assert result["deployment_authorized"] is True
    assert result["active_motor_control_authorized"] is False
    assert (
        result["deployment_conditions"][
            "free_standing_first_actuation_authorized"
        ]
        is False
    )
    assert verify_true23_mujoco_promotion(
        sidecar,
        checkpoint_path=checkpoint,
        encoder_path=encoder,
        decoder_path=decoder,
        metadata_path=metadata,
        report_path=report_path,
    ) == result

    tampered = json.loads(sidecar.read_text(encoding="utf-8"))
    tampered["active_motor_control_authorized"] = True
    body = dict(tampered)
    body.pop("promotion_payload_sha256")
    tampered["promotion_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(body)
    )
    sidecar.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="recomputed raw evidence"):
        verify_true23_mujoco_promotion(
            sidecar,
            checkpoint_path=checkpoint,
            encoder_path=encoder,
            decoder_path=decoder,
            metadata_path=metadata,
            report_path=report_path,
        )


def test_promotion_cli_create_and_verify_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = {
        "sidecar": tmp_path / "promotion.json",
        "checkpoint": tmp_path / "trained.promotion.pt",
        "encoder": tmp_path / "candidate.encoder.onnx",
        "decoder": tmp_path / "candidate.decoder.onnx",
        "metadata": tmp_path / "candidate.metadata.json",
        "report": tmp_path / "mujoco.report.json",
    }
    calls: list[tuple[str, Path, dict]] = []
    result = {"promotion_payload_sha256": "a" * 64}

    def record_create(sidecar: Path, **kwargs) -> dict:
        calls.append(("create", sidecar, kwargs))
        return result

    def record_verify(sidecar: Path, **kwargs) -> dict:
        calls.append(("verify", sidecar, kwargs))
        return result

    monkeypatch.setattr(cli, "promote_true23_mujoco_candidate", record_create)
    monkeypatch.setattr(cli, "verify_true23_mujoco_promotion", record_verify)
    args = [
        "--sidecar",
        str(paths["sidecar"]),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--encoder-onnx",
        str(paths["encoder"]),
        "--decoder-onnx",
        str(paths["decoder"]),
        "--metadata",
        str(paths["metadata"]),
        "--report",
        str(paths["report"]),
    ]
    assert cli.main(args) == 0
    assert cli.main([*args, "--verify-only"]) == 0
    assert [call[0] for call in calls] == ["create", "verify"]
    assert all(call[1] == paths["sidecar"] for call in calls)
    assert calls[0][2] == {
        "checkpoint_path": paths["checkpoint"],
        "encoder_path": paths["encoder"],
        "decoder_path": paths["decoder"],
        "metadata_path": paths["metadata"],
        "report_path": paths["report"],
    }
    output = capsys.readouterr().out
    assert "active motor control authorized: false" in output
