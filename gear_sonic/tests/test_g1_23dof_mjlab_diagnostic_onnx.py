"""Focused fail-closed tests for MJLab diagnostic-only ONNX export."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    CAUSAL_HISTORY_PROFILE,
    causal_history_profile_contract,
)
from gear_sonic.utils import (
    g1_23dof_artifact as artifact,
    g1_23dof_mjlab_diagnostic_onnx as diagnostic,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    safe_target_transform_contract,
)


class _Flat(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.uniform_(self.linear.weight, -0.01, 0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


@pytest.fixture
def fake_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str]:
    checkpoint_path = tmp_path / "model_250.pt"
    checkpoint_path.write_bytes(b"weights-only-test-checkpoint")
    lineage_hash = "1" * 64
    policy_hash = "2" * 64
    encoder_weight = torch.zeros(64, 267, dtype=torch.float32)
    decoder_weight = torch.zeros(23, 994, dtype=torch.float32)
    checkpoint = {
        "policy_state_dict": {
            "actor_module.encoders.teleop.module.0.weight": encoder_weight,
            "actor_module.decoders.g1_dyn.module.0.weight": decoder_weight,
        },
        "policy_state_sha256": policy_hash,
        "lineage_sha256": lineage_hash,
        "lineage": {
            "warm_start": {"reference_profile": "true23_step5_0p1s"}
        },
        "update_count": 250,
        "training_gate": {
            "simulation_candidate_review_allowed": True,
            "deployment_ready": False,
            "promotion_eligible": False,
        },
    }
    encoder = _Flat(267, 64).eval()
    decoder = _Flat(994, 23).eval()

    def load_stub(
        path: Path,
        *,
        expected_lineage_sha256: str,
        map_location: str,
    ) -> dict:
        assert Path(path).resolve() == checkpoint_path.resolve()
        assert expected_lineage_sha256 == lineage_hash
        assert map_location == "cpu"
        return checkpoint

    monkeypatch.setattr(diagnostic, "load_mjlab_training_checkpoint", load_stub)
    monkeypatch.setattr(
        artifact,
        "build_true23_policy_pair",
        lambda value: (encoder, decoder, policy_hash),
    )
    return checkpoint_path, lineage_hash


def test_output_names_reject_promotion_deployment_and_overwrite(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="promotion or deployment"):
        diagnostic.diagnostic_output_paths(tmp_path / "model.promotion")
    with pytest.raises(ValueError, match="promotion or deployment"):
        diagnostic.diagnostic_output_paths(tmp_path / "robot_deployment")
    encoder, _, _ = diagnostic.diagnostic_output_paths(tmp_path / "candidate")
    encoder.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="overwrite"):
        diagnostic.diagnostic_output_paths(tmp_path / "candidate")


def test_gate_must_explicitly_allow_simulation_review(
    fake_checkpoint: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path, lineage_hash = fake_checkpoint
    original = diagnostic.load_mjlab_training_checkpoint

    def blocked(*args, **kwargs):
        value = copy.deepcopy(original(*args, **kwargs))
        value["training_gate"]["simulation_candidate_review_allowed"] = False
        return value

    monkeypatch.setattr(diagnostic, "load_mjlab_training_checkpoint", blocked)
    with pytest.raises(ValueError, match="simulation_candidate_review_allowed=true"):
        diagnostic.export_mjlab_diagnostic_onnx(
            checkpoint_path,
            checkpoint_path.parent / "blocked",
            expected_lineage_sha256=lineage_hash,
        )


def test_export_is_static_hash_bound_and_diagnostic_only(
    fake_checkpoint: tuple[Path, str],
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint_path, lineage_hash = fake_checkpoint
    encoder, decoder, metadata_path, metadata = (
        diagnostic.export_mjlab_diagnostic_onnx(
            checkpoint_path,
            tmp_path / "model_250_shadow",
            expected_lineage_sha256=lineage_hash,
        )
    )
    assert metadata["diagnostic_only"] is True
    assert metadata["deployment_ready"] is False
    assert metadata["promotion_eligible"] is False
    assert metadata["active_motor_control_authorized"] is False
    assert metadata["source"]["simulation_candidate_review_allowed"] is True
    assert metadata["contract"]["encoder"]["input_shape"] == [1, 267]
    assert metadata["contract"]["encoder"]["output_shape"] == [1, 64]
    assert metadata["contract"]["decoder"]["input_shape"] == [1, 994]
    assert metadata["contract"]["decoder"]["output_shape"] == [1, 23]
    assert metadata["contract"]["onnx_opset"] == 13
    assert metadata["hashes"]["lineage_sha256"] == lineage_hash
    assert metadata["validation"]["paired_inference"]["case_count"] == 3
    verified = diagnostic.verify_mjlab_diagnostic_onnx(
        encoder,
        decoder,
        metadata_path,
        checkpoint_path=checkpoint_path,
    )
    assert verified == metadata
    with pytest.raises(FileExistsError, match="overwrite"):
        diagnostic.export_mjlab_diagnostic_onnx(
            checkpoint_path,
            tmp_path / "model_250_shadow",
            expected_lineage_sha256=lineage_hash,
        )


def test_onnx_byte_tamper_breaks_bundle_hash(
    fake_checkpoint: tuple[Path, str],
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint_path, lineage_hash = fake_checkpoint
    encoder, decoder, metadata_path, _ = diagnostic.export_mjlab_diagnostic_onnx(
        checkpoint_path,
        tmp_path / "tamper_probe",
        expected_lineage_sha256=lineage_hash,
    )
    encoder.write_bytes(encoder.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="encoder_onnx_sha256"):
        diagnostic.verify_mjlab_diagnostic_onnx(
            encoder,
            decoder,
            metadata_path,
        )


def _causal_fake_checkpoint() -> dict:
    policy_hash = "4" * 64
    return {
        "policy_state_dict": {
            "actor_module.encoders.teleop.module.0.weight": torch.zeros(
                64, 267, dtype=torch.float32
            ),
            "actor_module.decoders.g1_dyn.module.0.weight": torch.zeros(
                23, 994, dtype=torch.float32
            ),
        },
        "policy_state_sha256": policy_hash,
        "lineage_sha256": "3" * 64,
        "lineage": {
            "warm_start": {
                "reference_profile": "released_low_latency_step1_0p02s"
            },
            "materials": {
                "resolved_config": {
                    "payload": {
                        "reference_profile": CAUSAL_HISTORY_PROFILE,
                        "semantic_profile": causal_history_profile_contract(),
                        "architecture_initialization": {
                            "source_profile": (
                                "released_low_latency_step1_0p02s"
                            ),
                            "source_future_semantics_inherited": False,
                            "checkpoint_relabelled": False,
                            "retraining_required": True,
                        },
                        "recovery": {
                            "released_future_profile_exporter_must_reject": True,
                            "checkpoint_filename_pattern": "causal_model_N.pt",
                        },
                    }
                }
            },
        },
        "update_count": 250,
        "training_gate": {
            "simulation_candidate_review_allowed": True,
            "deployment_ready": False,
            "promotion_eligible": False,
        },
    }


def test_causal_export_binds_profile_and_rejects_future_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint_path = tmp_path / "causal_model_250.pt"
    checkpoint_path.write_bytes(b"causal-weights-only-test-checkpoint")
    checkpoint = _causal_fake_checkpoint()
    encoder = _Flat(267, 64).eval()
    decoder = _Flat(994, 23).eval()

    monkeypatch.setattr(
        diagnostic,
        "load_mjlab_training_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        artifact,
        "build_true23_policy_pair",
        lambda value: (
            encoder,
            decoder,
            checkpoint["policy_state_sha256"],
        ),
    )
    with pytest.raises(ValueError, match="semantic namespace"):
        diagnostic.export_mjlab_diagnostic_onnx(
            checkpoint_path,
            tmp_path / "future_path_must_reject",
            expected_lineage_sha256=checkpoint["lineage_sha256"],
        )

    encoder_path, decoder_path, metadata_path, metadata = (
        diagnostic.export_mjlab_diagnostic_onnx(
            checkpoint_path,
            tmp_path / "causal_model_250_diagnostic",
            expected_lineage_sha256=checkpoint["lineage_sha256"],
            expected_reference_profile=CAUSAL_HISTORY_PROFILE,
        )
    )
    assert metadata["source"]["reference_profile"] == CAUSAL_HISTORY_PROFILE
    assert diagnostic.verify_mjlab_diagnostic_onnx(
        encoder_path,
        decoder_path,
        metadata_path,
        checkpoint_path=checkpoint_path,
        expected_reference_profile=CAUSAL_HISTORY_PROFILE,
    ) == metadata
    with pytest.raises(ValueError, match="required profile"):
        diagnostic.verify_mjlab_diagnostic_onnx(
            encoder_path,
            decoder_path,
            metadata_path,
            expected_reference_profile="released_low_latency_step1_0p02s",
        )


def test_causal_export_rejects_future_semantic_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "causal_model_250.pt"
    checkpoint_path.write_bytes(b"causal-future-semantics-test-checkpoint")
    checkpoint = _causal_fake_checkpoint()
    checkpoint["lineage"]["materials"]["resolved_config"]["payload"][
        "semantic_profile"
    ]["future_samples_relative_to_emission"] = True
    monkeypatch.setattr(
        diagnostic,
        "load_mjlab_training_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )
    with pytest.raises(ValueError, match="semantic contract/hash mismatch"):
        diagnostic.export_mjlab_diagnostic_onnx(
            checkpoint_path,
            tmp_path / "must_reject_future_semantics",
            expected_lineage_sha256=checkpoint["lineage_sha256"],
            expected_reference_profile=CAUSAL_HISTORY_PROFILE,
        )


def test_v11_export_embeds_safe_transform_and_forbids_external_squash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint_path = tmp_path / "causal_model_250.pt"
    checkpoint_path.write_bytes(b"causal-v11-safe-target-test-checkpoint")
    checkpoint = _causal_fake_checkpoint()
    transform = safe_target_transform_contract()
    resolved = checkpoint["lineage"]["materials"]["resolved_config"][
        "payload"
    ]
    resolved["causal_history_safe_target_v11"] = {
        "schema": "g1_true23_causal_history_safe_target_v11",
        "restart_from_model0": True,
        "v9_push_domain_randomization_and_physical_gates_unchanged": True,
        "safe_target_transform": transform,
        "target_transform_trained_in_loop": True,
        "previous_action_is_applied_safe_native_action": True,
        "evaluator_aligned_rapid_recovery_weight": -25.0,
        "post_hoc_clamp_relabel": False,
        "deployment_ready": False,
        "executed_environment": {
            "target_transform": transform,
            "history_uses_applied_safe_native_action": True,
            "encoder_bias_applied_after_unbiased_target": True,
            "evaluator_aligned_recovery": {
                "function": (
                    "gear_sonic.envs.mjlab."
                    "sonic_true23_causal_history_safe_target_v11:"
                    "evaluator_aligned_recovery_metric"
                ),
                "weight": -25.0,
                "metric": (
                    "tilt+abs(pelvis_height-reference)+"
                    "target_tracking_rmse"
                ),
                "continuous_under_interval_pushes": True,
            },
        },
    }
    resolved["causal_safe_target_training_v11"] = {
        "schema": "g1_true23_causal_safe_target_training_v11",
        "restart_from_model0": True,
        "v9_push_domain_randomization_and_physical_gates_unchanged": True,
        "safe_target_transform_in_training_loop": True,
        "deployment_ready": False,
    }
    encoder = _Flat(267, 64).eval()
    decoder = _Flat(994, 23).eval()
    monkeypatch.setattr(
        diagnostic,
        "load_mjlab_training_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        artifact,
        "build_true23_policy_pair",
        lambda value: (
            encoder,
            decoder,
            checkpoint["policy_state_sha256"],
        ),
    )
    encoder_path, decoder_path, metadata_path, metadata = (
        diagnostic.export_mjlab_diagnostic_onnx(
            checkpoint_path,
            tmp_path / "causal_model_250_v11_diagnostic",
            expected_lineage_sha256=checkpoint["lineage_sha256"],
            expected_reference_profile=CAUSAL_HISTORY_PROFILE,
        )
    )
    assert metadata["schema_version"] == 2
    assert (
        metadata["contract"]["decoder_output_semantics"]
        == "applied_safe_native_action"
    )
    assert metadata["contract"]["safe_target_transform"] == transform
    assert (
        metadata["contract"]["external_safe_target_transform_allowed"]
        is False
    )
    assert diagnostic.verify_mjlab_diagnostic_onnx(
        encoder_path,
        decoder_path,
        metadata_path,
        checkpoint_path=checkpoint_path,
        expected_reference_profile=CAUSAL_HISTORY_PROFILE,
    ) == metadata
    relabelled = copy.deepcopy(metadata)
    relabelled["contract"]["external_safe_target_transform_allowed"] = True
    with pytest.raises(ValueError, match="ONNX/robot contract mismatch"):
        diagnostic._validate_metadata(relabelled)  # noqa: SLF001
