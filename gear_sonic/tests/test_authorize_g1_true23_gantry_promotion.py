from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from gear_sonic.scripts import authorize_g1_true23_gantry_promotion as authorize
from gear_sonic.utils.g1_23dof_artifact import sha256_file


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for role in (
        "promotion",
        "checkpoint",
        "encoder_onnx",
        "decoder_onnx",
        "metadata",
        "full_report",
    ):
        path = tmp_path / role
        path.write_bytes(role.encode())
        result[role] = path
    return result


def _records(paths: dict[str, Path], *, action_frames: int = 100) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {
            "schema_version": 1,
            "kind": "g1_true23_integrated_live_shadow_evidence",
            "event": "session_start",
            "started_monotonic_ns": 900_000_000,
            "reference_profile": "true23_causal_step1_history_0p02s_v1",
            "reference_contract_sha256": (
                "bd046467325fe7f7f585fd692f01223ed7a3b2742c51ed414072c98fe12806f7"
            ),
            "artifact_class": "promoted_shadow",
            "decoder_output_semantics": "applied_safe_native_action",
            "external_safe_target_transform_applied": False,
            "encoder_sha256": sha256_file(paths["encoder_onnx"]),
            "decoder_sha256": sha256_file(paths["decoder_onnx"]),
            "metadata_sha256": sha256_file(paths["metadata"]),
            "promotion_sha256": sha256_file(paths["promotion"]),
            "network": "eth0",
            "pico_endpoint": "tcp://127.0.0.1:5557",
            "requested_action_frames": action_frames,
            "robot_mutation_authorized": False,
        },
        {
            "schema_version": 1,
            "kind": "g1_true23_integrated_live_shadow_evidence",
            "event": "lowstate_gate_open",
            "mode_machine": 4,
            "crc_rejects": 0,
            "history_warmup_span_ns": 40_000_000,
        },
    ]
    first_frame = 100
    first_time = 1_000_000_000
    for index in range(9):
        records.append(
            {
                "schema_version": 1,
                "kind": "g1_true23_integrated_live_shadow_evidence",
                "event": "causal_warmup_frame",
                "control_source_frame_index": first_frame + index,
                "control_monotonic_ns": first_time + index * 20_000_000,
                "history_samples": index + 1,
                "packet_age_ns": 1_000_000,
                "sdk_derivatives_consumed": False,
            }
        )
    for index in range(action_frames):
        frame = first_frame + 9 + index
        control_ns = first_time + (9 + index) * 20_000_000
        received_ns = control_ns + 1_000_000
        produced_ns = received_ns + 1_000_000
        records.append(
            {
                "schema_version": 1,
                "kind": "g1_true23_integrated_live_shadow_evidence",
                "event": "action_frame",
                "action_frame_index": index,
                "control_source_frame_index": frame,
                "pico_anchor_source_frame_index": frame - 1,
                "pico_anchor_monotonic_ns": control_ns - 20_000_000,
                "control_monotonic_ns": control_ns,
                "received_monotonic_ns": received_ns,
                "produced_monotonic_ns": produced_ns,
                "packet_age_ns": 1_000_000,
                "end_to_end_age_ns": 2_000_000,
                "lowstate_age_ns": 1_000_000,
                "inference_ns": 1_000_000,
                "native_action": [0.0] * 23,
                "decoder_output_semantics": "applied_safe_native_action",
                "external_safe_target_transform_applied": False,
                "normalized_max_abs": 0.0,
                "target_position_min_margin_rad": 0.01,
                "target_limit_violations": 0,
                "slew_checked": index > 0,
                "target_slew_ratio_max": 0.0,
                "target_slew_violations": 0,
                "sdk_derivatives_consumed": False,
                "accepted": True,
            }
        )
    records.append(
        {
            "schema_version": 1,
            "kind": "g1_true23_integrated_live_shadow_evidence",
            "event": "session_complete",
            "passed": True,
            "action_frames": action_frames,
            "causal_warmup_frames": 9,
            "maximum_normalized_abs": 0.0,
            "minimum_target_position_margin_rad": 0.01,
            "maximum_target_slew_ratio": 0.0,
            "crc_rejects": 0,
            "robot_mutation_authorized": False,
        }
    )
    return records


def _write_evidence(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_bytes(
        b"".join(
            (
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            for record in records
        )
    )


def _args(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, Path]]:
    paths = _artifacts(tmp_path)
    evidence = tmp_path / "shadow.jsonl"
    _write_evidence(evidence, _records(paths))
    return (
        argparse.Namespace(
            output=tmp_path / "causal-active-promotion.json",
            live_shadow_evidence=evidence,
            authorization_id="operator-review-20260803",
            gantry_authorize=authorize.AUTHORIZATION_PHRASE,
            verify_only=False,
            **paths,
        ),
        paths,
    )


def _base_promotion(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "deployment_bytes_authorized": True,
        "active_motor_control_authorized": False,
        "gantry_or_rated_support_required": True,
        "free_standing_authorized": False,
        "source_artifact": {
            "checkpoint_sha256": sha256_file(paths["checkpoint"]),
            "lineage_sha256": "1" * 64,
            "policy_state_sha256": "2" * 64,
            "encoder_onnx_sha256": sha256_file(paths["encoder_onnx"]),
            "decoder_onnx_sha256": sha256_file(paths["decoder_onnx"]),
            "metadata_sha256": sha256_file(paths["metadata"]),
        },
        "full_campaign_evidence": {
            "aggregate_report_sha256": sha256_file(paths["full_report"]),
            "shard_manifest_sha256": "3" * 64,
        },
    }


def test_schema2_authorizer_accepts_exact_promoted_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, paths = _args(tmp_path)
    monkeypatch.setattr(
        "gear_sonic.scripts.authorize_g1_true23_causal_gantry.verify_causal_promotion",
        lambda *unused_args, **unused_kwargs: _base_promotion(paths),
    )

    body = authorize._body(args)

    assert len(body) == 25
    assert body["schema_version"] == 2
    assert body["kind"] == "g1_true23_causal_gantry_active_promotion"
    assert body["active_motor_control_authorized"] is True
    assert body["gantry_authorized"] is True
    assert body["free_standing_authorized"] is False
    assert body["live_shadow_evidence_sha256"] == sha256_file(
        args.live_shadow_evidence
    )


def test_wrong_phrase_rejected_before_any_artifact_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, unused_paths = _args(tmp_path)
    args.gantry_authorize = "WRONG"
    called = False

    def verifier(*unused_args: object, **unused_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("must not run")

    monkeypatch.setattr(
        "gear_sonic.scripts.authorize_g1_true23_causal_gantry.verify_causal_promotion",
        verifier,
    )
    with pytest.raises(ValueError, match="exact explicit gantry"):
        authorize._body(args)
    assert called is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda records: records[-1].update(passed=False), "terminal PASS"),
        (lambda records: records[11].update(accepted=False), "action safety"),
        (
            lambda records: records[12].update(action_frame_index=9),
            "indices are not consecutive",
        ),
        (
            lambda records: records[12].update(
                control_monotonic_ns=records[12]["control_monotonic_ns"] + 1
            ),
            "timestamps are not exact",
        ),
    ],
)
def test_shadow_mutations_fail_closed(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    paths = _artifacts(tmp_path)
    records = _records(paths)
    mutation(records)
    evidence = tmp_path / "shadow.jsonl"
    _write_evidence(evidence, records)
    with pytest.raises(ValueError, match=message):
        authorize.validate_causal_live_shadow_evidence(
            evidence,
            promotion_path=paths["promotion"],
            encoder_path=paths["encoder_onnx"],
            decoder_path=paths["decoder_onnx"],
            metadata_path=paths["metadata"],
        )

def test_shadow_rejects_failure_or_trailing_record(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    records = _records(paths)
    records.append(
        {
            "schema_version": 1,
            "kind": "g1_true23_integrated_live_shadow_evidence",
            "event": "session_failed",
        }
    )
    evidence = tmp_path / "shadow.jsonl"
    _write_evidence(evidence, records)
    with pytest.raises(ValueError, match="event count is not exact"):
        authorize.validate_causal_live_shadow_evidence(
            evidence,
            promotion_path=paths["promotion"],
            encoder_path=paths["encoder_onnx"],
            decoder_path=paths["decoder_onnx"],
            metadata_path=paths["metadata"],
        )


def test_shadow_rejects_duplicate_json_key(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    evidence = tmp_path / "shadow.jsonl"
    _write_evidence(evidence, _records(paths))
    lines = evidence.read_bytes().splitlines(keepends=True)
    lines[0] = lines[0].replace(b'{"artifact_class":', b'{"event":"fake","artifact_class":')
    evidence.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="duplicate key"):
        authorize.validate_causal_live_shadow_evidence(
            evidence,
            promotion_path=paths["promotion"],
            encoder_path=paths["encoder_onnx"],
            decoder_path=paths["decoder_onnx"],
            metadata_path=paths["metadata"],
        )
