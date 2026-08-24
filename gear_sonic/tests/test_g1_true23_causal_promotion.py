from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    CAUSAL_HISTORY_PROFILE,
)
from gear_sonic.utils import g1_true23_causal_promotion as promotion
from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    safe_target_transform_contract,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _paths(tmp_path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for role in ("checkpoint", "encoder", "decoder", "metadata", "report"):
        path = tmp_path / role
        path.write_bytes(role.encode())
        result[f"{role}_path" if role != "report" else "full_report_path"] = path
    return result


def _metadata(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "contract": {
            "decoder_output_semantics": promotion.APPLIED_SAFE_NATIVE_ACTION,
            "previous_action_semantics": promotion.APPLIED_SAFE_NATIVE_ACTION,
            "external_safe_target_transform_allowed": False,
            "safe_target_transform": safe_target_transform_contract(),
        },
        "hashes": {
            "checkpoint_sha256": sha256_file(paths["checkpoint_path"]),
            "lineage_sha256": _sha("lineage"),
            "policy_state_sha256": _sha("policy"),
            "encoder_onnx_sha256": sha256_file(paths["encoder_path"]),
            "decoder_onnx_sha256": sha256_file(paths["decoder_path"]),
            "safe_target_transform_sha256": promotion._safe_transform_sha256(),
        },
        "source": {"checkpoint_update_count": 50},
        "metadata_payload_sha256": _sha("metadata-payload"),
    }


def _campaign(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "campaign_layout": "monolithic",
        "aggregate_report_filename": paths["full_report_path"].name,
        "aggregate_report_sha256": sha256_file(paths["full_report_path"]),
        "provenance": {"test": True},
    }


def _patch_verifiers(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
) -> None:
    metadata = _metadata(paths)
    campaign = _campaign(paths)
    monkeypatch.setattr(
        promotion,
        "verify_mjlab_diagnostic_onnx",
        lambda *unused_args, **unused_kwargs: metadata,
    )
    monkeypatch.setattr(
        promotion,
        "_validate_full_report",
        lambda *unused_args, **unused_kwargs: campaign,
    )
    monkeypatch.setattr(
        promotion.diagnostic,
        "verify_full_mjlab_diagnostic_report",
        lambda unused_path: {
            "report_sha256": campaign["aggregate_report_sha256"],
            "report": {},
        },
    )
    monkeypatch.setattr(
        promotion,
        "_validate_provenance",
        lambda *unused_args, **unused_kwargs: campaign["provenance"],
    )


def test_causal_promotion_is_bytes_only_and_binds_safe_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    _patch_verifiers(monkeypatch, paths)

    body = promotion.causal_promotion_body(**paths)

    assert set(body) == promotion._PROMOTION_KEYS
    assert body["deployment_bytes_authorized"] is True
    assert "deployment_authorized" not in body
    assert body["active_motor_control_authorized"] is False
    assert body["free_standing_authorized"] is False
    assert body["decoder_output_semantics"] == "applied_safe_native_action"
    assert body["previous_action_semantics"] == "applied_safe_native_action"
    assert body["external_safe_target_transform_allowed"] is False
    assert body["source_artifact"]["reference_profile"] == CAUSAL_HISTORY_PROFILE


def test_causal_promotion_rejects_input_symlink_before_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    target = paths["checkpoint_path"]
    link = tmp_path / "checkpoint-link"
    link.symlink_to(target)
    paths["checkpoint_path"] = link
    verifier_called = False

    def verifier(*unused_args: object, **unused_kwargs: object) -> object:
        nonlocal verifier_called
        verifier_called = True
        raise AssertionError("symlink must be rejected first")

    monkeypatch.setattr(promotion, "verify_mjlab_diagnostic_onnx", verifier)
    with pytest.raises(ValueError, match="must not traverse symlinks"):
        promotion.causal_promotion_body(**paths)
    assert verifier_called is False


def test_causal_promotion_detects_source_change_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    metadata = _metadata(paths)
    campaign = _campaign(paths)
    monkeypatch.setattr(
        promotion,
        "verify_mjlab_diagnostic_onnx",
        lambda *unused_args, **unused_kwargs: metadata,
    )

    def mutate_source(*unused_args: object, **unused_kwargs: object) -> dict[str, Any]:
        paths["checkpoint_path"].write_bytes(b"changed")
        return campaign

    monkeypatch.setattr(promotion, "_validate_full_report", mutate_source)
    monkeypatch.setattr(
        promotion.diagnostic,
        "verify_full_mjlab_diagnostic_report",
        lambda unused_path: {
            "report_sha256": campaign["aggregate_report_sha256"],
            "report": {},
        },
    )
    monkeypatch.setattr(
        promotion,
        "_validate_provenance",
        lambda *unused_args, **unused_kwargs: campaign["provenance"],
    )
    with pytest.raises(ValueError, match="input bytes changed"):
        promotion.causal_promotion_body(**paths)


def test_write_new_json_rejects_symlink_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "causal-promotion.json"
    output.symlink_to(tmp_path / "missing-target.json")
    with pytest.raises(ValueError, match="must not traverse symlinks"):
        promotion.write_new_json(output, {"safe": True})
