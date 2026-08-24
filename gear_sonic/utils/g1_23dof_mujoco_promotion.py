"""Candidate and promotion evidence for true23 MuJoCo Sim2Sim.

Candidate export is deliberately separate from deployment authorization:

* only a safe, genuinely trained ``*.promotion.pt`` checkpoint is accepted;
* candidate ONNX contains exactly 23 outputs and is marked non-deployable;
* no simulation result is claimed during candidate export;
* later MuJoCo evidence must bind the unchanged candidate bytes.

This module imports no Unitree SDK code and exposes no robot command surface.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import warnings

from gear_sonic.utils import g1_23dof_artifact as artifact
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    TRAINED_STAGE,
    checkpoint_stage,
    extract_global_step,
    load_safe_true23_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import (
    ARTIFACT_SCHEMA_VERSION,
    DEPLOYMENT_DECODER_INPUT_DIM,
    DEPLOYMENT_HISTORY_LENGTH,
    OBS_LAYOUT_PADDED_IL29,
    ROBOT_MODEL,
    TARGET_DOF,
    TELEOP_ENCODER_INPUT_DIM,
    TOKEN_DIM,
    make_artifact_metadata,
    reference_profile_contract,
    validate_artifact_contract,
)

CANDIDATE_SCHEMA_VERSION = 1
CANDIDATE_KIND = "g1_true23_mujoco_candidate_onnx_pair"
CANDIDATE_EMBEDDED_KIND = "g1_true23_mujoco_candidate_onnx"
CANDIDATE_STAGE = "mujoco_candidate"
PINNED_ASSET_SOURCE = {
    "repository": "https://github.com/unitreerobotics/unitree_ros.git",
    "revision": "f3772ce54c56ef2d34c6aee8100bc768896c7d19",
    "root_relpath": "robots/g1_description",
    "text_normalization": "crlf_to_lf_and_ensure_final_newline_v1",
    "file_count": 29,
    "total_bytes": 25_211_170,
    "manifest_sha256": (
        "a98562b34a591fd26a2f4024d84454aa0d3f40ca9067e4d17d900d00f18a492b"
    ),
}

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_URDF_PATH = _PACKAGE_ROOT / "data/robots/g1/g1_23dof_rev_1_0.urdf"
_MJCF_PATH = _PACKAGE_ROOT / "data/robots/g1/g1_23dof_rev_1_0.xml"
_ROBOT_CONFIG_PATH = _PACKAGE_ROOT / "envs/manager_env/robots/g1_23dof.py"
_ENCODER_PREFIX = "actor_module.encoders.teleop.module."
_DECODER_PREFIX = "actor_module.decoders.g1_dyn.module."
_SHA256_RE = artifact._SHA256_RE  # noqa: SLF001

_BASE_METADATA_KEYS = {
    "schema_version",
    "robot_model",
    "mode_machine",
    "action_dof",
    "hardware_joint_ids",
    "excluded_hardware_joint_ids",
    "decoder_output_layout",
    "observation_layout",
    "history_length",
    "decoder_input_dim",
    "decoder_output_dim",
    "reference_profile",
    "reference_contract",
    "checkpoint_stage",
    "deployment_ready",
    "sim_validation_passed",
    "naive_output_masking",
    "observation_contract",
    "action_contract",
}
_CANDIDATE_METADATA_KEYS = _BASE_METADATA_KEYS | {
    "artifact_kind",
    "promotion_stage",
    "deployment_authorized",
    "encoder_onnx_filename",
    "decoder_onnx_filename",
    "metadata_filename",
    "onnx_opset",
    "asset_provenance",
    "training_evidence",
    "hashes",
    "validation",
    "metadata_payload_sha256",
}
_CANDIDATE_HASH_KEYS = {
    "checkpoint_sha256",
    "policy_state_sha256",
    "encoder_state_sha256",
    "decoder_state_sha256",
    "encoder_onnx_sha256",
    "decoder_onnx_sha256",
    "training_evidence_sha256",
    "contract_sha256",
    "urdf_sha256",
    "mjcf_sha256",
    "robot_config_sha256",
    "asset_manifest_sha256",
    "encoder_embedded_metadata_sha256",
    "decoder_embedded_metadata_sha256",
}


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


def _atomic_write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite candidate output: {path}"
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _set_embedded_metadata(model: Any, value: Mapping[str, Any]) -> None:
    del model.metadata_props[:]
    entry = model.metadata_props.add()
    entry.key = artifact.ONNX_METADATA_KEY
    entry.value = artifact.canonical_json_bytes(value).decode("utf-8").rstrip(
        "\n"
    )


def _get_embedded_metadata(model: Any) -> Mapping[str, Any]:
    entries = [
        item.value
        for item in model.metadata_props
        if item.key == artifact.ONNX_METADATA_KEY
    ]
    if len(entries) != 1:
        raise ValueError("candidate ONNX must contain one contract metadata entry")
    try:
        value = json.loads(
            entries[0],
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"candidate ONNX metadata contains {token}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError("candidate ONNX metadata is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("candidate ONNX metadata must be an object")
    return value


def _asset_provenance() -> dict[str, Any]:
    manifest = artifact.canonical_true23_robot_asset_manifest()
    expected = {
        "schema_version": 1,
        "file_count": PINNED_ASSET_SOURCE["file_count"],
        "total_bytes": PINNED_ASSET_SOURCE["total_bytes"],
        "manifest_sha256": PINNED_ASSET_SOURCE["manifest_sha256"],
    }
    actual = {key: manifest[key] for key in expected}
    if actual != expected:
        raise ValueError("local true23 asset manifest differs from pinned Unitree source")
    return {
        **PINNED_ASSET_SOURCE,
        "urdf_sha256": artifact.sha256_file(_URDF_PATH),
        "mjcf_sha256": artifact.sha256_file(_MJCF_PATH),
        "verified": True,
    }


def _base_contract(reference_profile: str) -> dict[str, Any]:
    return make_artifact_metadata(
        history_length=DEPLOYMENT_HISTORY_LENGTH,
        observation_layout=OBS_LAYOUT_PADDED_IL29,
        checkpoint_stage=TRAINED_STAGE,
        reference_profile=reference_profile,
        deployment_ready=False,
        sim_validation_passed=False,
    )


def _candidate_embedded_metadata(
    *,
    role: str,
    checkpoint_sha256: str,
    policy_state_sha256: str,
    encoder_state_sha256: str,
    decoder_state_sha256: str,
    training_evidence_sha256: str,
    contract_sha256: str,
    global_step: int,
    reference_profile: str,
) -> dict[str, Any]:
    if role not in {"teleop_encoder", "true23_decoder"}:
        raise ValueError(f"unsupported candidate ONNX role: {role}")
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "kind": CANDIDATE_EMBEDDED_KIND,
        "artifact_role": role,
        "promotion_stage": CANDIDATE_STAGE,
        "deployment_authorized": False,
        "robot_model": ROBOT_MODEL,
        "checkpoint_stage": TRAINED_STAGE,
        "checkpoint_sha256": checkpoint_sha256,
        "policy_state_sha256": policy_state_sha256,
        "encoder_state_sha256": encoder_state_sha256,
        "decoder_state_sha256": decoder_state_sha256,
        "training_evidence_sha256": training_evidence_sha256,
        "contract_sha256": contract_sha256,
        "global_step": global_step,
        "reference_profile": reference_profile,
        "reference_contract": reference_profile_contract(reference_profile),
        "encoder_input_dim": TELEOP_ENCODER_INPUT_DIM,
        "token_dim": TOKEN_DIM,
        "decoder_input_dim": DEPLOYMENT_DECODER_INPUT_DIM,
        "decoder_output_dim": TARGET_DOF,
        "naive_output_masking": False,
    }


def _checkpoint_material(
    checkpoint_path: Path,
) -> tuple[
    Mapping[str, Any],
    Any,
    Any,
    dict[str, Any],
]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.name.endswith(".promotion.pt"):
        raise ValueError(
            "MuJoCo candidate requires trained weights-only *.promotion.pt"
        )
    checkpoint = load_safe_true23_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    if checkpoint_stage(checkpoint) != TRAINED_STAGE:
        raise ValueError("initialization checkpoint cannot become MuJoCo candidate")
    global_step = extract_global_step(checkpoint)
    encoder, decoder, policy_state_sha256 = artifact.build_true23_policy_pair(
        checkpoint
    )
    artifact.validate_training_checkpoint_records(
        checkpoint,
        global_step=global_step,
        policy_state_sha256=policy_state_sha256,
    )
    training_evidence = checkpoint["g1_23dof_training_evidence"]
    reference_profile = str(
        checkpoint["g1_23dof_metadata"]["reference_profile"]
    )
    policy_state = checkpoint["policy_state_dict"]
    encoder_state_sha256 = artifact._policy_state_sha256(  # noqa: SLF001
        {
            key: value
            for key, value in policy_state.items()
            if key.startswith(_ENCODER_PREFIX)
        }
    )
    decoder_state_sha256 = artifact._policy_state_sha256(  # noqa: SLF001
        {
            key: value
            for key, value in policy_state.items()
            if key.startswith(_DECODER_PREFIX)
        }
    )
    contract = _base_contract(reference_profile)
    material = {
        "global_step": global_step,
        "reference_profile": reference_profile,
        "policy_state_sha256": policy_state_sha256,
        "encoder_state_sha256": encoder_state_sha256,
        "decoder_state_sha256": decoder_state_sha256,
        "training_evidence": dict(training_evidence),
        "training_evidence_sha256": artifact.sha256_bytes(
            artifact.canonical_json_bytes(training_evidence)
        ),
        "contract": contract,
        "contract_sha256": artifact.sha256_bytes(
            artifact.canonical_json_bytes(contract)
        ),
    }
    return checkpoint, encoder, decoder, material


def _export_one(
    *,
    module: Any,
    output_path: Path,
    input_name: str,
    input_dim: int,
    output_name: str,
    output_dim: int,
    embedded: Mapping[str, Any],
    structure_validator: Any,
) -> dict[str, Any]:
    import onnx
    from onnx import shape_inference
    import torch

    example = torch.zeros(1, input_dim, dtype=torch.float32)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            module,
            example,
            output_path,
            export_params=True,
            input_names=[input_name],
            output_names=[output_name],
            opset_version=artifact.ONNX_OPSET_VERSION,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
    model = onnx.load(output_path, load_external_data=False)
    model = shape_inference.infer_shapes(
        model,
        strict_mode=True,
        data_prop=True,
    )
    _set_embedded_metadata(model, embedded)
    onnx.save_model(model, output_path, save_as_external_data=False)
    model = onnx.load(output_path, load_external_data=False)
    structure_validator(model)
    if _get_embedded_metadata(model) != embedded:
        raise ValueError("candidate embedded metadata changed during serialization")
    return artifact.validate_ort_parity(
        module,
        output_path,
        input_name=input_name,
        input_dim=input_dim,
        output_name=output_name,
        output_dim=output_dim,
    )


def _pair_dry_run(encoder_path: Path, decoder_path: Path) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    encoder_session = ort.InferenceSession(
        str(encoder_path),
        providers=["CPUExecutionProvider"],
    )
    decoder_session = ort.InferenceSession(
        str(decoder_path),
        providers=["CPUExecutionProvider"],
    )
    token = encoder_session.run(
        [artifact.ENCODER_ONNX_OUTPUT_NAME],
        {
            artifact.ENCODER_ONNX_INPUT_NAME: np.zeros(
                (1, TELEOP_ENCODER_INPUT_DIM),
                dtype=np.float32,
            )
        },
    )[0]
    decoder_input = np.zeros(
        (1, DEPLOYMENT_DECODER_INPUT_DIM),
        dtype=np.float32,
    )
    if token.shape != (1, TOKEN_DIM) or not np.isfinite(token).all():
        raise ValueError("candidate encoder dry-run is not finite [1,64]")
    decoder_input[:, :TOKEN_DIM] = token
    action = decoder_session.run(
        [artifact.ONNX_OUTPUT_NAME],
        {artifact.ONNX_INPUT_NAME: decoder_input},
    )[0]
    if action.shape != (1, TARGET_DOF) or not np.isfinite(action).all():
        raise ValueError("candidate pair dry-run is not finite [1,23]")
    return {
        "performed": True,
        "device": "cpu",
        "dtype": "float32",
        "token_sha256": hashlib.sha256(token.tobytes()).hexdigest(),
        "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
    }


def _candidate_metadata_without_payload_hash(
    *,
    encoder_filename: str,
    decoder_filename: str,
    metadata_filename: str,
    checkpoint_sha256: str,
    material: Mapping[str, Any],
    encoder_onnx_sha256: str,
    decoder_onnx_sha256: str,
    embedded_by_role: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(material["contract"])
    result.update(
        {
            "artifact_kind": CANDIDATE_KIND,
            "promotion_stage": CANDIDATE_STAGE,
            "deployment_authorized": False,
            "encoder_onnx_filename": encoder_filename,
            "decoder_onnx_filename": decoder_filename,
            "metadata_filename": metadata_filename,
            "onnx_opset": artifact.ONNX_OPSET_VERSION,
            "asset_provenance": _asset_provenance(),
            "training_evidence": material["training_evidence"],
            "hashes": {
                "checkpoint_sha256": checkpoint_sha256,
                "policy_state_sha256": material["policy_state_sha256"],
                "encoder_state_sha256": material["encoder_state_sha256"],
                "decoder_state_sha256": material["decoder_state_sha256"],
                "encoder_onnx_sha256": encoder_onnx_sha256,
                "decoder_onnx_sha256": decoder_onnx_sha256,
                "training_evidence_sha256": material[
                    "training_evidence_sha256"
                ],
                "contract_sha256": material["contract_sha256"],
                "urdf_sha256": artifact.sha256_file(_URDF_PATH),
                "mjcf_sha256": artifact.sha256_file(_MJCF_PATH),
                "robot_config_sha256": artifact.sha256_file(
                    _ROBOT_CONFIG_PATH
                ),
                "asset_manifest_sha256": PINNED_ASSET_SOURCE[
                    "manifest_sha256"
                ],
                "encoder_embedded_metadata_sha256": artifact.sha256_bytes(
                    artifact.canonical_json_bytes(
                        embedded_by_role["teleop_encoder"]
                    )
                ),
                "decoder_embedded_metadata_sha256": artifact.sha256_bytes(
                    artifact.canonical_json_bytes(
                        embedded_by_role["true23_decoder"]
                    )
                ),
            },
            "validation": dict(validation),
        }
    )
    return result


def export_true23_mujoco_candidate(
    checkpoint_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
) -> tuple[Path, Path, Path, Mapping[str, Any]]:
    """Export immutable non-deployable ONNX from a trained promotion checkpoint."""

    checkpoint_path = checkpoint_path.resolve()
    encoder_path = encoder_path.resolve()
    decoder_path = decoder_path.resolve()
    metadata_path = metadata_path.resolve()
    outputs = (encoder_path, decoder_path, metadata_path)
    if len(set(outputs)) != 3:
        raise ValueError("candidate encoder, decoder, and metadata paths must differ")
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite MuJoCo candidate outputs")

    _checkpoint, encoder, decoder, material = _checkpoint_material(
        checkpoint_path
    )
    checkpoint_sha256 = artifact.sha256_file(checkpoint_path)
    embedded_by_role = {
        role: _candidate_embedded_metadata(
            role=role,
            checkpoint_sha256=checkpoint_sha256,
            policy_state_sha256=material["policy_state_sha256"],
            encoder_state_sha256=material["encoder_state_sha256"],
            decoder_state_sha256=material["decoder_state_sha256"],
            training_evidence_sha256=material["training_evidence_sha256"],
            contract_sha256=material["contract_sha256"],
            global_step=material["global_step"],
            reference_profile=material["reference_profile"],
        )
        for role in ("teleop_encoder", "true23_decoder")
    }

    temporary_paths: list[Path] = []
    published: list[Path] = []

    def temporary_for(path: Path, suffix: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=suffix,
        )
        os.close(descriptor)
        result = Path(name)
        temporary_paths.append(result)
        return result

    try:
        temporary_encoder = temporary_for(encoder_path, ".tmp.onnx")
        temporary_decoder = temporary_for(decoder_path, ".tmp.onnx")
        encoder_validation = _export_one(
            module=encoder,
            output_path=temporary_encoder,
            input_name=artifact.ENCODER_ONNX_INPUT_NAME,
            input_dim=TELEOP_ENCODER_INPUT_DIM,
            output_name=artifact.ENCODER_ONNX_OUTPUT_NAME,
            output_dim=TOKEN_DIM,
            embedded=embedded_by_role["teleop_encoder"],
            structure_validator=artifact.validate_encoder_onnx_structure,
        )
        decoder_validation = _export_one(
            module=decoder,
            output_path=temporary_decoder,
            input_name=artifact.ONNX_INPUT_NAME,
            input_dim=DEPLOYMENT_DECODER_INPUT_DIM,
            output_name=artifact.ONNX_OUTPUT_NAME,
            output_dim=TARGET_DOF,
            embedded=embedded_by_role["true23_decoder"],
            structure_validator=artifact.validate_onnx_structure,
        )
        validation = {
            "teleop_encoder": encoder_validation,
            "true23_decoder": decoder_validation,
            "pair_dry_run": _pair_dry_run(
                temporary_encoder,
                temporary_decoder,
            ),
        }
        metadata = _candidate_metadata_without_payload_hash(
            encoder_filename=encoder_path.name,
            decoder_filename=decoder_path.name,
            metadata_filename=metadata_path.name,
            checkpoint_sha256=checkpoint_sha256,
            material=material,
            encoder_onnx_sha256=artifact.sha256_file(temporary_encoder),
            decoder_onnx_sha256=artifact.sha256_file(temporary_decoder),
            embedded_by_role=embedded_by_role,
            validation=validation,
        )
        metadata["metadata_payload_sha256"] = artifact.sha256_bytes(
            artifact.canonical_json_bytes(metadata)
        )
        temporary_metadata = temporary_for(metadata_path, ".tmp.json")
        temporary_metadata.write_bytes(artifact.canonical_json_bytes(metadata))
        verify_true23_mujoco_candidate(
            temporary_encoder,
            temporary_decoder,
            temporary_metadata,
            checkpoint_path=checkpoint_path,
            expected_filenames=(
                encoder_path.name,
                decoder_path.name,
                metadata_path.name,
            ),
        )
        for temporary, final in (
            (temporary_encoder, encoder_path),
            (temporary_decoder, decoder_path),
            (temporary_metadata, metadata_path),
        ):
            if final.exists():
                raise FileExistsError(
                    f"refusing to overwrite MuJoCo candidate output: {final}"
                )
            os.replace(temporary, final)
            published.append(final)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    return encoder_path, decoder_path, metadata_path, metadata


def verify_true23_mujoco_candidate(
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
    *,
    checkpoint_path: Path,
    expected_filenames: tuple[str, str, str] | None = None,
) -> Mapping[str, Any]:
    """Rebuild training identity and verify candidate files and CPU parity."""

    import onnx

    encoder_path = encoder_path.resolve()
    decoder_path = decoder_path.resolve()
    metadata_path = metadata_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    if len({encoder_path, decoder_path, metadata_path, checkpoint_path}) != 4:
        raise ValueError("candidate and checkpoint paths must be distinct")
    for path in (encoder_path, decoder_path, metadata_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = artifact.load_strict_json(metadata_path)
    _require_exact_keys(
        metadata,
        _CANDIDATE_METADATA_KEYS,
        "candidate metadata",
    )
    filenames = expected_filenames or (
        encoder_path.name,
        decoder_path.name,
        metadata_path.name,
    )
    if (
        metadata["encoder_onnx_filename"],
        metadata["decoder_onnx_filename"],
        metadata["metadata_filename"],
    ) != filenames:
        raise ValueError("candidate metadata filenames do not match supplied files")
    if (
        metadata["schema_version"] != ARTIFACT_SCHEMA_VERSION
        or metadata["artifact_kind"] != CANDIDATE_KIND
        or metadata["promotion_stage"] != CANDIDATE_STAGE
        or metadata["deployment_authorized"] is not False
        or metadata["deployment_ready"] is not False
        or metadata["sim_validation_passed"] is not False
        or "simulation_evidence" in metadata
    ):
        raise ValueError("candidate metadata falsely claims deployment/simulation")
    validate_artifact_contract(
        metadata,
        decoder_input_dim=DEPLOYMENT_DECODER_INPUT_DIM,
        decoder_output_dim=TARGET_DOF,
        require_deployment_ready=False,
    )
    unhashed = dict(metadata)
    claimed_payload = _require_sha256(
        unhashed.pop("metadata_payload_sha256"),
        "candidate metadata_payload_sha256",
    )
    if artifact.sha256_bytes(artifact.canonical_json_bytes(unhashed)) != claimed_payload:
        raise ValueError("candidate metadata payload hash mismatch")

    _checkpoint, encoder, decoder, material = _checkpoint_material(
        checkpoint_path
    )
    expected_contract = material["contract"]
    for key, value in expected_contract.items():
        if metadata.get(key) != value:
            raise ValueError(f"candidate contract changed: {key}")
    if metadata["training_evidence"] != material["training_evidence"]:
        raise ValueError("candidate training evidence differs from checkpoint")
    if metadata["asset_provenance"] != _asset_provenance():
        raise ValueError("candidate asset provenance changed")

    hashes = metadata["hashes"]
    if not isinstance(hashes, Mapping):
        raise ValueError("candidate hashes must be an object")
    _require_exact_keys(hashes, _CANDIDATE_HASH_KEYS, "candidate hashes")
    expected_hashes = {
        "checkpoint_sha256": artifact.sha256_file(checkpoint_path),
        "policy_state_sha256": material["policy_state_sha256"],
        "encoder_state_sha256": material["encoder_state_sha256"],
        "decoder_state_sha256": material["decoder_state_sha256"],
        "encoder_onnx_sha256": artifact.sha256_file(encoder_path),
        "decoder_onnx_sha256": artifact.sha256_file(decoder_path),
        "training_evidence_sha256": material["training_evidence_sha256"],
        "contract_sha256": material["contract_sha256"],
        "urdf_sha256": artifact.sha256_file(_URDF_PATH),
        "mjcf_sha256": artifact.sha256_file(_MJCF_PATH),
        "robot_config_sha256": artifact.sha256_file(_ROBOT_CONFIG_PATH),
        "asset_manifest_sha256": PINNED_ASSET_SOURCE["manifest_sha256"],
    }
    for key, value in expected_hashes.items():
        if hashes.get(key) != value:
            raise ValueError(f"candidate hash mismatch: {key}")

    embedded_by_role = {
        role: _candidate_embedded_metadata(
            role=role,
            checkpoint_sha256=expected_hashes["checkpoint_sha256"],
            policy_state_sha256=material["policy_state_sha256"],
            encoder_state_sha256=material["encoder_state_sha256"],
            decoder_state_sha256=material["decoder_state_sha256"],
            training_evidence_sha256=material["training_evidence_sha256"],
            contract_sha256=material["contract_sha256"],
            global_step=material["global_step"],
            reference_profile=material["reference_profile"],
        )
        for role in ("teleop_encoder", "true23_decoder")
    }
    for role, path, structure_validator, embedded_hash_key in (
        (
            "teleop_encoder",
            encoder_path,
            artifact.validate_encoder_onnx_structure,
            "encoder_embedded_metadata_sha256",
        ),
        (
            "true23_decoder",
            decoder_path,
            artifact.validate_onnx_structure,
            "decoder_embedded_metadata_sha256",
        ),
    ):
        model = onnx.load(path, load_external_data=False)
        structure_validator(model)
        embedded = _get_embedded_metadata(model)
        if embedded != embedded_by_role[role]:
            raise ValueError(f"{role} candidate embedded metadata mismatch")
        embedded_hash = artifact.sha256_bytes(
            artifact.canonical_json_bytes(embedded)
        )
        if hashes.get(embedded_hash_key) != embedded_hash:
            raise ValueError(f"{role} candidate embedded hash mismatch")

    artifact.validate_ort_parity(
        encoder,
        encoder_path,
        input_name=artifact.ENCODER_ONNX_INPUT_NAME,
        input_dim=TELEOP_ENCODER_INPUT_DIM,
        output_name=artifact.ENCODER_ONNX_OUTPUT_NAME,
        output_dim=TOKEN_DIM,
    )
    artifact.validate_ort_parity(
        decoder,
        decoder_path,
        input_name=artifact.ONNX_INPUT_NAME,
        input_dim=DEPLOYMENT_DECODER_INPUT_DIM,
        output_name=artifact.ONNX_OUTPUT_NAME,
        output_dim=TARGET_DOF,
    )
    _pair_dry_run(encoder_path, decoder_path)
    return metadata


PROMOTION_SCHEMA_VERSION = 1
PROMOTION_KIND = "g1_true23_mujoco_promoted_onnx_pair"
PROMOTED_STAGE = "mujoco_sim2sim_promoted"
APPROVAL_RELPATH = (
    "gear_sonic/config/sim_validation/"
    "g1_23dof_mujoco_sim2sim_approval.json"
)
_REPOSITORY_ROOT = _PACKAGE_ROOT.parent
_APPROVAL_PATH = _REPOSITORY_ROOT / APPROVAL_RELPATH
_SIM2SIM_CONFIG_PATH = (
    _PACKAGE_ROOT
    / "config/sim_validation/g1_23dof_mujoco_sim2sim.json"
)
_SIM2SIM_RUNNER_PATH = (
    _PACKAGE_ROOT / "scripts/run_g1_23dof_mujoco_sim2sim.py"
)
_SIM2SIM_RUNTIME_PATH = (
    _PACKAGE_ROOT / "utils/g1_23dof_mujoco_sim2sim.py"
)
_PROMOTION_SOURCE_PATH = Path(__file__).resolve()

_APPROVAL_KEYS = {
    "schema_version",
    "kind",
    "promotion_enabled",
    "robot_model",
    "mujoco_version",
    "runner_sha256",
    "runtime_sha256",
    "promotion_source_sha256",
    "config_sha256",
    "mjcf_sha256",
    "urdf_sha256",
    "asset_manifest_sha256",
}
_REPORT_KEYS = {
    "schema_version",
    "kind",
    "robot_model",
    "checkpoint_stage",
    "diagnostic_only",
    "promotion_eligible",
    "computed_pass",
    "source_artifact",
    "producer",
    "simulator",
    "contract",
    "reference_command",
    "trace_manifest_sha256",
    "runs",
    "summary",
}
_SOURCE_ARTIFACT_KEYS = {
    "artifact_kind",
    "checkpoint_sha256",
    "policy_state_sha256",
    "encoder_onnx_sha256",
    "decoder_onnx_sha256",
    "candidate_manifest_sha256",
    "candidate_manifest_payload_sha256",
    "candidate_claimed_payload_sha256",
    "inference_runtime",
    "inference_threads",
}
_RUN_KEYS = {
    "scenario",
    "seed",
    "episodes",
    "steps_per_episode",
    "disturbance_scale",
    "computed_pass",
    "metrics",
    "trace",
}
_TRACE_DESCRIPTOR_KEYS = {
    "file",
    "sha256",
    "payload_sha256",
    "record_count",
}
_METRIC_KEYS = {
    "episode_count",
    "record_count",
    "termination_count",
    "nonfinite_count",
    "joint_limit_violation_count",
    "min_base_height_m",
    "max_tilt_rad",
    "max_tracking_rmse_rad",
    "mean_tracking_rmse_rad",
    "max_abs_joint_velocity_radps",
    "max_abs_applied_torque_nm",
    "max_abs_native_action",
    "max_abs_native_action_raw",
    "action_saturation_fraction",
    "recovery_fraction",
    "max_recovery_time_s",
}
_SUMMARY_METRIC_KEYS = {
    "run_count",
    "episode_count",
    "record_count",
    "termination_count",
    "nonfinite_count",
    "joint_limit_violation_count",
    "min_base_height_m",
    "max_tilt_rad",
    "max_tracking_rmse_rad",
    "max_abs_joint_velocity_radps",
    "max_abs_applied_torque_nm",
    "max_abs_native_action",
    "max_abs_native_action_raw",
    "max_action_saturation_fraction",
    "minimum_recovery_fraction",
    "max_recovery_time_s",
}
_PROMOTION_SIDECAR_KEYS = {
    "schema_version",
    "kind",
    "robot_model",
    "promotion_stage",
    "deployment_authorized",
    "active_motor_control_authorized",
    "checkpoint_stage",
    "source_candidate",
    "mujoco_evidence",
    "deployment_conditions",
    "promotion_payload_sha256",
}


def _approved_sim2sim_material() -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Any,
    Any,
    Mapping[str, Any],
]:
    """Load exact approved producer/config/model and reject local drift."""

    from gear_sonic.utils import g1_23dof_mujoco_sim2sim as sim2sim

    approval = artifact.load_strict_json(_APPROVAL_PATH)
    _require_exact_keys(approval, _APPROVAL_KEYS, "MuJoCo approval")
    if (
        approval["schema_version"] != 1
        or approval["kind"] != "g1_true23_mujoco_sim2sim_approval"
        or approval["promotion_enabled"] is not True
        or approval["robot_model"] != ROBOT_MODEL
    ):
        raise ValueError("MuJoCo Sim2Sim producer is not approved for promotion")
    expected_hashes = {
        "runner_sha256": artifact.sha256_file(_SIM2SIM_RUNNER_PATH),
        "runtime_sha256": artifact.sha256_file(_SIM2SIM_RUNTIME_PATH),
        "promotion_source_sha256": artifact.sha256_file(
            _PROMOTION_SOURCE_PATH
        ),
        "config_sha256": artifact.sha256_file(_SIM2SIM_CONFIG_PATH),
        "mjcf_sha256": artifact.sha256_file(_MJCF_PATH),
        "urdf_sha256": artifact.sha256_file(_URDF_PATH),
        "asset_manifest_sha256": PINNED_ASSET_SOURCE["manifest_sha256"],
    }
    for key, value in expected_hashes.items():
        if _require_sha256(approval.get(key), f"approval.{key}") != value:
            raise ValueError(f"approved MuJoCo material changed: {key}")
    if approval["mujoco_version"] != "3.2.3":
        raise ValueError("approved MuJoCo version must be exactly 3.2.3")
    config = sim2sim.load_sim2sim_config(_SIM2SIM_CONFIG_PATH)
    if (
        config["coverage"]["deterministic_seeds"] != [1729, 2718, 3141]
        or config["coverage"]["episodes_per_seed"] != 22
        or config["coverage"]["steps_per_episode"] != 250
        or config["coverage"]["seconds_per_episode"] != 5.0
    ):
        raise ValueError("approved MuJoCo coverage must be 66 episodes/scenario")
    module, model, physics_contract = sim2sim.prepare_mujoco_model(
        mjcf_path=_MJCF_PATH,
        config=config,
    )
    if module.__version__ != approval["mujoco_version"]:
        raise ValueError("local MuJoCo version differs from approved evidence engine")
    return approval, config, module, model, physics_contract


def _candidate_source_artifact(
    *,
    metadata: Mapping[str, Any],
    checkpoint_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    return {
        "artifact_kind": "paired_onnx_trained_candidate",
        "checkpoint_sha256": artifact.sha256_file(checkpoint_path),
        "policy_state_sha256": metadata["hashes"]["policy_state_sha256"],
        "encoder_onnx_sha256": artifact.sha256_file(encoder_path),
        "decoder_onnx_sha256": artifact.sha256_file(decoder_path),
        "candidate_manifest_sha256": artifact.sha256_file(metadata_path),
        "candidate_manifest_payload_sha256": artifact.sha256_bytes(
            artifact.canonical_json_bytes(metadata)
        ),
        "candidate_claimed_payload_sha256": metadata[
            "metadata_payload_sha256"
        ],
        "inference_runtime": "onnxruntime_cpu",
        "inference_threads": 1,
    }


def _strict_json_line(line: bytes, context: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            line.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{context} contains non-finite number {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _trace_path(
    *,
    report_path: Path,
    relative: str,
    expected_relative: str,
    context: str,
) -> Path:
    if relative != expected_relative:
        raise ValueError(f"{context}.file must be {expected_relative!r}")
    unresolved = report_path.parent / relative
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{context}.file must stay below report directory")
    candidate = report_path.parent
    for part in relative_path.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"{context}.file may not traverse symlinks")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(report_path.parent.resolve())
    except ValueError as exc:
        raise ValueError(f"{context}.file escapes report directory") from exc
    if not resolved.is_file():
        raise ValueError(f"{context}.file is missing")
    return resolved


def _finite_array(value: Any, size: int, context: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not __import__("math").isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"{context} must contain {size} finite numbers")
    return [float(item) for item in value]


def _close(actual: float, expected: float, context: str) -> None:
    if abs(float(actual) - float(expected)) > 1.0e-8:
        raise ValueError(f"{context} is inconsistent with raw state")


def _validate_trace_record_semantics(
    record: Mapping[str, Any],
    *,
    scenario: str,
    seed: int,
    episode: int,
    step: int,
    disturbance_scale: float,
    config: Mapping[str, Any],
    model: Any,
    neutral_height: float,
) -> None:
    """Cross-check redundant record fields, mappings, limits, and disturbance."""

    import math

    import numpy as np

    from gear_sonic.utils import g1_23dof_mujoco_sim2sim as sim2sim
    from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_ACTION_SCALE
    from gear_sonic.utils.g1_23dof_live_shadow import (
        HARDWARE_DEFAULT_Q,
        native_to_hardware,
    )

    _require_exact_keys(
        record,
        sim2sim._TRACE_RECORD_KEYS,  # noqa: SLF001
        "MuJoCo trace record",
    )
    expected_identity = {
        "schema_version": sim2sim.TRACE_SCHEMA_VERSION,
        "kind": sim2sim.TRACE_KIND,
        "scenario": scenario,
        "seed": seed,
        "episode": episode,
        "step": step,
    }
    for key, expected in expected_identity.items():
        if record.get(key) != expected:
            raise ValueError(f"MuJoCo trace record identity mismatch: {key}")
    _close(
        float(record["time_s"]),
        (step + 1) / float(config["control_hz"]),
        "trace time_s",
    )
    vectors = {
        "disturbance_delta": 6,
        "base_position_m": 3,
        "base_quaternion_wxyz": 4,
        "base_linear_velocity_mps": 3,
        "base_angular_velocity_radps": 3,
        "projected_gravity": 3,
        "joint_position_hardware_rad": TARGET_DOF,
        "joint_velocity_hardware_radps": TARGET_DOF,
        "action_native_raw": TARGET_DOF,
        "action_native": TARGET_DOF,
        "target_position_hardware_rad": TARGET_DOF,
        "applied_torque_hardware_nm": TARGET_DOF,
    }
    normalized = {
        key: _finite_array(record[key], size, f"trace.{key}")
        for key, size in vectors.items()
    }
    expected_disturbance = (
        sim2sim.deterministic_disturbance(
            config=config,
            seed=seed,
            episode=episode,
            scale=disturbance_scale,
        )
        if step == config["disturbance_schedule"]["apply_step"]
        and disturbance_scale > 0
        else [0.0] * 6
    )
    if normalized["disturbance_delta"] != expected_disturbance:
        raise ValueError("trace disturbance does not match deterministic schedule")

    quaternion = np.asarray(
        normalized["base_quaternion_wxyz"],
        dtype=np.float64,
    )
    if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1.0e-6:
        raise ValueError("trace base quaternion is not normalized")
    expected_gravity = sim2sim._projected_gravity(quaternion)  # noqa: SLF001
    if not np.allclose(
        normalized["projected_gravity"],
        expected_gravity,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError("trace projected gravity disagrees with quaternion")
    expected_tilt = math.acos(
        float(np.clip(-expected_gravity[2], -1.0, 1.0))
    )
    _close(float(record["tilt_rad"]), expected_tilt, "trace tilt_rad")
    _close(
        float(record["base_height_m"]),
        normalized["base_position_m"][2],
        "trace base_height_m",
    )

    clip = float(config["physics"]["action_clip_value"])
    raw_action = np.asarray(normalized["action_native_raw"])
    clipped_action = np.clip(raw_action, -clip, clip)
    if not np.allclose(
        normalized["action_native"],
        clipped_action,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("trace action_native is not exact configured clip")
    saturated = int(np.count_nonzero(clipped_action != raw_action))
    if (
        isinstance(record["action_saturated_count"], bool)
        or record["action_saturated_count"] != saturated
    ):
        raise ValueError("trace action saturation count mismatch")
    hardware_action = np.asarray(
        native_to_hardware(clipped_action.tolist()),
        dtype=np.float64,
    )
    expected_target = np.asarray(HARDWARE_DEFAULT_Q) + hardware_action * np.asarray(
        HARDWARE_23_ACTION_SCALE
    )
    if not np.allclose(
        normalized["target_position_hardware_rad"],
        expected_target,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError("trace target does not match native-to-hardware action")
    q = np.asarray(normalized["joint_position_hardware_rad"])
    expected_tracking = float(np.sqrt(np.mean((expected_target - q) ** 2)))
    _close(
        float(record["tracking_rmse_rad"]),
        expected_tracking,
        "trace tracking_rmse_rad",
    )
    expected_recovery = (
        expected_tilt
        + abs(normalized["base_position_m"][2] - neutral_height)
        + expected_tracking
    )
    _close(
        float(record["recovery_metric"]),
        expected_recovery,
        "trace recovery_metric",
    )
    effort = np.asarray(config["physics"]["effort_limit_hardware_nm"])
    if np.any(
        np.abs(normalized["applied_torque_hardware_nm"]) > effort + 1.0e-8
    ):
        raise ValueError("trace applied torque exceeds configured effort limit")

    hard_ranges = np.asarray(model.jnt_range[1:], dtype=np.float64)
    midpoint = np.mean(hard_ranges, axis=1)
    half_range = (
        (hard_ranges[:, 1] - hard_ranges[:, 0])
        * float(config["physics"]["soft_joint_pos_limit_factor"])
        * 0.5
    )
    soft_ranges = np.column_stack((midpoint - half_range, midpoint + half_range))
    expected_violation = bool(
        np.any(q < soft_ranges[:, 0])
        or np.any(q > soft_ranges[:, 1])
        or np.any(expected_target < soft_ranges[:, 0])
        or np.any(expected_target > soft_ranges[:, 1])
    )
    if record["joint_limit_violation"] is not expected_violation:
        raise ValueError("trace joint_limit_violation flag mismatch")
    for key in ("nonfinite", "terminated", "joint_limit_violation"):
        if not isinstance(record[key], bool):
            raise ValueError(f"trace {key} must be boolean")
    if not isinstance(record["termination_reason"], str):
        raise ValueError("trace termination_reason must be string")


def _load_and_validate_trace(
    descriptor: Mapping[str, Any],
    *,
    report_path: Path,
    scenario: str,
    seed: int,
    episodes: int,
    disturbance_scale: float,
    config: Mapping[str, Any],
    model: Any,
    neutral_height: float,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    _require_exact_keys(
        descriptor,
        _TRACE_DESCRIPTOR_KEYS,
        "MuJoCo trace descriptor",
    )
    expected_relative = (
        f"{report_path.stem}.traces/{scenario}-seed-{seed}.jsonl"
    )
    trace_path = _trace_path(
        report_path=report_path,
        relative=descriptor["file"],
        expected_relative=expected_relative,
        context="MuJoCo trace",
    )
    payload = trace_path.read_bytes()
    if _require_sha256(descriptor["sha256"], "trace.sha256") != (
        artifact.sha256_bytes(payload)
    ):
        raise ValueError("MuJoCo trace byte hash mismatch")
    lines = payload.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("MuJoCo JSONL trace contains blank lines")
    records = [
        _strict_json_line(line, f"MuJoCo trace line {index + 1}")
        for index, line in enumerate(lines)
    ]
    expected_count = episodes * int(config["coverage"]["steps_per_episode"])
    if (
        isinstance(descriptor["record_count"], bool)
        or descriptor["record_count"] != expected_count
        or len(records) != expected_count
    ):
        raise ValueError("MuJoCo trace record_count mismatch")
    canonical_payload_hash = artifact.sha256_bytes(
        artifact.canonical_json_bytes(records)
    )
    if (
        _require_sha256(
            descriptor["payload_sha256"],
            "trace.payload_sha256",
        )
        != canonical_payload_hash
    ):
        raise ValueError("MuJoCo trace canonical payload hash mismatch")

    steps = int(config["coverage"]["steps_per_episode"])
    for index, record in enumerate(records):
        episode, step = divmod(index, steps)
        _validate_trace_record_semantics(
            record,
            scenario=scenario,
            seed=seed,
            episode=episode,
            step=step,
            disturbance_scale=disturbance_scale,
            config=config,
            model=model,
            neutral_height=neutral_height,
        )
    manifest = {
        "scenario": scenario,
        "seed": seed,
        **dict(descriptor),
    }
    return records, manifest


def validate_true23_mujoco_report(
    report_path: Path,
    *,
    checkpoint_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
) -> Mapping[str, Any]:
    """Replay every raw trace and return hash-bound promotion evidence."""

    from gear_sonic.utils import g1_23dof_mujoco_sim2sim as sim2sim

    report_path = report_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    encoder_path = encoder_path.resolve()
    decoder_path = decoder_path.resolve()
    metadata_path = metadata_path.resolve()
    metadata = verify_true23_mujoco_candidate(
        encoder_path,
        decoder_path,
        metadata_path,
        checkpoint_path=checkpoint_path,
    )
    approval, config, module, model, physics_contract = (
        _approved_sim2sim_material()
    )
    report_bytes = report_path.read_bytes()
    report = artifact.load_strict_json(report_path)
    _require_exact_keys(report, _REPORT_KEYS, "MuJoCo report")
    if (
        report["schema_version"] != sim2sim.REPORT_SCHEMA_VERSION
        or report["kind"] != sim2sim.REPORT_KIND
        or report["robot_model"] != ROBOT_MODEL
        or report["checkpoint_stage"] != TRAINED_STAGE
        or report["diagnostic_only"] is not False
        or report["promotion_eligible"] is not True
        or report["computed_pass"] is not True
    ):
        raise ValueError("MuJoCo report is diagnostic, failed, or wrong embodiment")

    source_artifact = report["source_artifact"]
    if not isinstance(source_artifact, Mapping):
        raise ValueError("MuJoCo report source_artifact must be object")
    _require_exact_keys(
        source_artifact,
        _SOURCE_ARTIFACT_KEYS,
        "MuJoCo report source_artifact",
    )
    expected_source = _candidate_source_artifact(
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        metadata_path=metadata_path,
    )
    if dict(source_artifact) != expected_source:
        raise ValueError("MuJoCo report is bound to different candidate bytes")

    expected_producer = {
        "kind": sim2sim.PRODUCER_KIND,
        "version": sim2sim.PRODUCER_VERSION,
        "runner_sha256": approval["runner_sha256"],
        "runtime_sha256": approval["runtime_sha256"],
    }
    if report["producer"] != expected_producer:
        raise ValueError("MuJoCo report producer hash/identity mismatch")
    simulator = report["simulator"]
    if not isinstance(simulator, Mapping):
        raise ValueError("MuJoCo report simulator must be object")
    _require_exact_keys(
        simulator,
        {
            "name",
            "version",
            "mjcf_sha256",
            "config_sha256",
            "approved_offline_inputs",
            "host",
            "compiled_model",
            "physics_contract",
            "asset_provenance",
        },
        "MuJoCo report simulator",
    )
    expected_simulator_values = {
        "name": "MuJoCo",
        "version": approval["mujoco_version"],
        "mjcf_sha256": approval["mjcf_sha256"],
        "config_sha256": approval["config_sha256"],
        "approved_offline_inputs": True,
        "compiled_model": sim2sim._compiled_model_contract(  # noqa: SLF001
            module,
            model,
        ),
        "physics_contract": physics_contract,
        "asset_provenance": sim2sim._asset_provenance(),  # noqa: SLF001
    }
    for key, value in expected_simulator_values.items():
        if simulator.get(key) != value:
            raise ValueError(f"MuJoCo simulator binding mismatch: {key}")
    host = simulator["host"]
    if (
        not isinstance(host, Mapping)
        or set(host) != {"system", "release", "machine", "processor", "python"}
        or any(not isinstance(value, str) for value in host.values())
    ):
        raise ValueError("MuJoCo report host descriptor is malformed")
    if report["contract"] != sim2sim._contract_descriptor():  # noqa: SLF001
        raise ValueError("MuJoCo report true23 contract mismatch")
    reference = sim2sim.NeutralReference(
        model,
        module,
        config["initial_state"],
    )
    if report["reference_command"] != reference.descriptor():
        raise ValueError("MuJoCo report reference command mismatch")
    replay_runtime = sim2sim.True23PolicyRuntime(
        checkpoint_path=checkpoint_path,
        encoder_onnx_path=encoder_path,
        decoder_onnx_path=decoder_path,
        metadata_path=metadata_path,
    )

    runs = report["runs"]
    if not isinstance(runs, list):
        raise ValueError("MuJoCo report runs must be array")
    expected_identities = [
        (scenario, seed)
        for scenario in config["scenarios"]
        for seed in config["coverage"]["deterministic_seeds"]
    ]
    if len(runs) != len(expected_identities):
        raise ValueError("MuJoCo report run coverage mismatch")
    trace_manifest: list[dict[str, Any]] = []
    recomputed_runs: list[dict[str, Any]] = []
    for index, (run, identity) in enumerate(
        zip(runs, expected_identities, strict=True)
    ):
        context = f"MuJoCo report runs[{index}]"
        if not isinstance(run, Mapping):
            raise ValueError(f"{context} must be object")
        _require_exact_keys(run, _RUN_KEYS, context)
        scenario, seed = identity
        scale = float(config["scenarios"][scenario]["disturbance_scale"])
        expected_header = {
            "scenario": scenario,
            "seed": seed,
            "episodes": config["coverage"]["episodes_per_seed"],
            "steps_per_episode": config["coverage"]["steps_per_episode"],
            "disturbance_scale": scale,
        }
        for key, value in expected_header.items():
            if run.get(key) != value:
                raise ValueError(f"{context}.{key} coverage mismatch")
        if not isinstance(run["trace"], Mapping):
            raise ValueError(f"{context}.trace must be object")
        records, manifest = _load_and_validate_trace(
            run["trace"],
            report_path=report_path,
            scenario=scenario,
            seed=seed,
            episodes=int(run["episodes"]),
            disturbance_scale=scale,
            config=config,
            model=model,
            neutral_height=float(reference.root_position[2]),
        )
        replayed_records: list[Mapping[str, Any]] = []
        for episode in range(int(run["episodes"])):
            replayed_records.extend(
                sim2sim.run_episode(
                    module=module,
                    model=model,
                    runtime=replay_runtime,
                    reference=reference,
                    config=config,
                    scenario=scenario,
                    seed=seed,
                    episode=episode,
                    disturbance_scale=scale,
                )
            )
        if artifact.canonical_json_bytes(records) != (
            artifact.canonical_json_bytes(replayed_records)
        ):
            raise ValueError(
                f"{context} raw trace differs from deterministic ONNX+MuJoCo replay"
            )
        trace_manifest.append(manifest)
        metrics = sim2sim.recompute_metrics(
            records,
            config=config,
            disturbance_scale=scale,
        )
        if not isinstance(run["metrics"], Mapping):
            raise ValueError(f"{context}.metrics must be object")
        _require_exact_keys(run["metrics"], _METRIC_KEYS, f"{context}.metrics")
        if dict(run["metrics"]) != metrics:
            raise ValueError(f"{context}.metrics differ from raw trace")
        computed_pass = sim2sim.metrics_pass(metrics, config)
        if run["computed_pass"] is not computed_pass or not computed_pass:
            raise ValueError(f"{context} fails promotion thresholds")
        recomputed_runs.append(metrics)

    manifest_hash = artifact.sha256_bytes(
        artifact.canonical_json_bytes(trace_manifest)
    )
    if (
        _require_sha256(
            report["trace_manifest_sha256"],
            "report.trace_manifest_sha256",
        )
        != manifest_hash
    ):
        raise ValueError("MuJoCo trace manifest hash mismatch")
    summary_metrics = {
        "run_count": len(recomputed_runs),
        "episode_count": sum(item["episode_count"] for item in recomputed_runs),
        "record_count": sum(item["record_count"] for item in recomputed_runs),
        "termination_count": sum(
            item["termination_count"] for item in recomputed_runs
        ),
        "nonfinite_count": sum(
            item["nonfinite_count"] for item in recomputed_runs
        ),
        "joint_limit_violation_count": sum(
            item["joint_limit_violation_count"] for item in recomputed_runs
        ),
        "min_base_height_m": min(
            item["min_base_height_m"] for item in recomputed_runs
        ),
        "max_tilt_rad": max(item["max_tilt_rad"] for item in recomputed_runs),
        "max_tracking_rmse_rad": max(
            item["max_tracking_rmse_rad"] for item in recomputed_runs
        ),
        "max_abs_joint_velocity_radps": max(
            item["max_abs_joint_velocity_radps"] for item in recomputed_runs
        ),
        "max_abs_applied_torque_nm": max(
            item["max_abs_applied_torque_nm"] for item in recomputed_runs
        ),
        "max_abs_native_action": max(
            item["max_abs_native_action"] for item in recomputed_runs
        ),
        "max_abs_native_action_raw": max(
            item["max_abs_native_action_raw"] for item in recomputed_runs
        ),
        "max_action_saturation_fraction": max(
            item["action_saturation_fraction"] for item in recomputed_runs
        ),
        "minimum_recovery_fraction": min(
            item["recovery_fraction"] for item in recomputed_runs
        ),
        "max_recovery_time_s": max(
            item["max_recovery_time_s"] for item in recomputed_runs
        ),
    }
    summary = report["summary"]
    if not isinstance(summary, Mapping):
        raise ValueError("MuJoCo report summary must be object")
    _require_exact_keys(
        summary,
        {"computed_pass", "promotion_eligible", "thresholds", "metrics"},
        "MuJoCo report summary",
    )
    if (
        summary["computed_pass"] is not True
        or summary["promotion_eligible"] is not True
        or summary["thresholds"] != config["promotion_thresholds"]
        or not isinstance(summary["metrics"], Mapping)
    ):
        raise ValueError("MuJoCo report summary claims do not match config")
    _require_exact_keys(
        summary["metrics"],
        _SUMMARY_METRIC_KEYS,
        "MuJoCo report summary.metrics",
    )
    if dict(summary["metrics"]) != summary_metrics:
        raise ValueError("MuJoCo report summary metrics differ from raw traces")

    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "computed_pass": True,
        "report_sha256": artifact.sha256_bytes(report_bytes),
        "report_payload_sha256": artifact.sha256_bytes(
            artifact.canonical_json_bytes(report)
        ),
        "source_artifact": expected_source,
        "producer": expected_producer,
        "simulator": {
            "name": "MuJoCo",
            "version": approval["mujoco_version"],
            "mjcf_sha256": approval["mjcf_sha256"],
            "config_sha256": approval["config_sha256"],
            "physics_contract_sha256": physics_contract["payload_sha256"],
            "asset_manifest_sha256": approval["asset_manifest_sha256"],
        },
        "trace_manifest_sha256": manifest_hash,
        "trace_count": len(trace_manifest),
        "scenario_count": len(config["scenarios"]),
        "run_count": len(recomputed_runs),
        "episodes_per_scenario": (
            len(config["coverage"]["deterministic_seeds"])
            * config["coverage"]["episodes_per_seed"]
        ),
        "total_episodes": summary_metrics["episode_count"],
        "total_records": summary_metrics["record_count"],
        "deterministic_onnx_mujoco_replay_verified": True,
        "summary_metrics": summary_metrics,
    }


def _promotion_body(
    *,
    checkpoint_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    metadata = verify_true23_mujoco_candidate(
        encoder_path,
        decoder_path,
        metadata_path,
        checkpoint_path=checkpoint_path,
    )
    evidence = validate_true23_mujoco_report(
        report_path,
        checkpoint_path=checkpoint_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        metadata_path=metadata_path,
    )
    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "kind": PROMOTION_KIND,
        "robot_model": ROBOT_MODEL,
        "promotion_stage": PROMOTED_STAGE,
        "deployment_authorized": True,
        "active_motor_control_authorized": False,
        "checkpoint_stage": TRAINED_STAGE,
        "source_candidate": {
            "checkpoint_filename": checkpoint_path.name,
            "encoder_onnx_filename": encoder_path.name,
            "decoder_onnx_filename": decoder_path.name,
            "metadata_filename": metadata_path.name,
            **_candidate_source_artifact(
                metadata=metadata,
                checkpoint_path=checkpoint_path,
                encoder_path=encoder_path,
                decoder_path=decoder_path,
                metadata_path=metadata_path,
            ),
        },
        "mujoco_evidence": evidence,
        "deployment_conditions": {
            "mode_machine": 4,
            "paired_onnx_bytes_must_remain_unchanged": True,
            "live_shadow_required": True,
            "gantry_or_rated_support_required_for_first_actuation": True,
            "free_standing_first_actuation_authorized": False,
        },
    }


def promote_true23_mujoco_candidate(
    sidecar_path: Path,
    *,
    checkpoint_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
    report_path: Path,
) -> Mapping[str, Any]:
    """Emit self-hashed authorization sidecar; never mutate ONNX bytes."""

    sidecar_path = sidecar_path.resolve()
    paths = tuple(
        path.resolve()
        for path in (
            checkpoint_path,
            encoder_path,
            decoder_path,
            metadata_path,
            report_path,
        )
    )
    body = _promotion_body(
        checkpoint_path=paths[0],
        encoder_path=paths[1],
        decoder_path=paths[2],
        metadata_path=paths[3],
        report_path=paths[4],
    )
    result = dict(body)
    result["promotion_payload_sha256"] = artifact.sha256_bytes(
        artifact.canonical_json_bytes(body)
    )
    encoded = artifact.canonical_json_bytes(result)
    _atomic_write_new(sidecar_path, encoded)
    if sidecar_path.read_bytes() != encoded:
        raise RuntimeError("persisted MuJoCo promotion sidecar differs from result")
    return result


def verify_true23_mujoco_promotion(
    sidecar_path: Path,
    *,
    checkpoint_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
    report_path: Path,
) -> Mapping[str, Any]:
    """Recompute candidate and trace evidence before accepting authorization."""

    sidecar = artifact.load_strict_json(sidecar_path.resolve())
    _require_exact_keys(
        sidecar,
        _PROMOTION_SIDECAR_KEYS,
        "MuJoCo promotion sidecar",
    )
    body = dict(sidecar)
    claimed = _require_sha256(
        body.pop("promotion_payload_sha256"),
        "promotion_payload_sha256",
    )
    if artifact.sha256_bytes(artifact.canonical_json_bytes(body)) != claimed:
        raise ValueError("MuJoCo promotion sidecar payload hash mismatch")
    expected = _promotion_body(
        checkpoint_path=checkpoint_path.resolve(),
        encoder_path=encoder_path.resolve(),
        decoder_path=decoder_path.resolve(),
        metadata_path=metadata_path.resolve(),
        report_path=report_path.resolve(),
    )
    if body != expected:
        raise ValueError(
            "MuJoCo promotion sidecar differs from recomputed raw evidence"
        )
    return sidecar
