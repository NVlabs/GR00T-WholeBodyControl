"""Build the immutable SONIC encoder parity fixture from an audited seed."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from gear_sonic.data.unitree_conversion.joint_mapping import (
    G1_ISAACLAB_NAMES,
    G1_MUJOCO_NAMES,
    NOMINAL_G1_MUJOCO,
    reorder_by_name,
)
from gear_sonic.data.unitree_conversion.provenance import load_source_lock, verify_file
from gear_sonic.data.unitree_conversion.resampling import resample_positions, resample_quaternions
from gear_sonic.data.unitree_conversion.sonic_encoder import (
    SonicEncoder,
    build_encoder_orientation_window,
    build_g1_encoder_input,
)

SEED = 20260810
GOLDEN_CASE_SHA256 = "3696b203ab720431ae50c693a1c27e72105de76a2e4048bc38b8fc01a48d9509"
GOLDEN_TOKEN_SHA256 = "d94d989f7d8d960653f76a78766b8a4ca8210f2b8e97010792d8189897b7ca2b"
_SOURCE_LOCK = Path("gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml")

_CASE_SHAPES = {
    "current_observed_root_wxyz": (4,),
    "encoder_input": (1, 1247),
    "future_reference_root_wxyz": (10, 4),
    "initial_observed_root_wxyz": (4,),
    "initial_reference_root_wxyz": (4,),
    "orientations": (10, 6),
    "positions": (10, 29),
    "velocities": (10, 29),
}
_JSON_FIELDS = (
    "current_observed_root_wxyz",
    "future_reference_root_wxyz",
    "initial_observed_root_wxyz",
    "initial_reference_root_wxyz",
    "positions",
    "velocities",
)


@dataclass(frozen=True)
class GoldenCaseArtifacts:
    case_npz: Path
    case_json: Path
    encoder_input_raw: Path


@dataclass(frozen=True)
class GoldenFixtureArtifacts(GoldenCaseArtifacts):
    token_npy: Path
    ort_token_raw: Path


def _unit(quaternion: np.ndarray) -> np.ndarray:
    return quaternion / np.linalg.norm(quaternion)


def _from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return _unit(
        np.array(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            dtype=np.float64,
        )
    )


def build_canonical_case() -> dict[str, np.ndarray]:
    """Return the seed-20260810 canonical ten-frame G1 encoder case."""
    rng = np.random.default_rng(SEED)
    nominal = reorder_by_name(
        NOMINAL_G1_MUJOCO,
        G1_MUJOCO_NAMES,
        G1_ISAACLAB_NAMES,
    )
    source_positions = np.tile(nominal, (6, 1))
    arm_indices = np.array(
        [
            index
            for index, name in enumerate(G1_ISAACLAB_NAMES)
            if "shoulder" in name or "elbow" in name or "wrist" in name
        ],
        dtype=np.int64,
    )
    increments = rng.uniform(-0.035, 0.035, size=(5, arm_indices.size))
    offsets = np.vstack((np.zeros((1, arm_indices.size)), np.cumsum(increments, axis=0)))
    source_positions[:, arm_indices] += np.clip(offsets, -0.14, 0.14)
    positions, velocities = resample_positions(source_positions)

    source_observed = np.stack(
        [
            _from_rpy(
                0.08 + 0.01 * index,
                -0.06 + 0.006 * index,
                0.31 + 0.025 * index,
            )
            for index in range(6)
        ]
    )
    source_reference = np.stack(
        [
            _from_rpy(
                -0.05 + 0.008 * index,
                0.07 - 0.005 * index,
                -0.43 + 0.035 * index,
            )
            for index in range(6)
        ]
    )
    observed = resample_quaternions(source_observed)
    future_reference = resample_quaternions(source_reference)
    initial_observed = observed[0]
    current_observed = observed[4]
    initial_reference = future_reference[0]
    orientations = build_encoder_orientation_window(
        initial_observed,
        current_observed,
        initial_reference,
        future_reference,
    )
    encoder_input = build_g1_encoder_input(positions, velocities, orientations)
    return {
        "current_observed_root_wxyz": current_observed,
        "encoder_input": encoder_input,
        "future_reference_root_wxyz": future_reference,
        "initial_observed_root_wxyz": initial_observed,
        "initial_reference_root_wxyz": initial_reference,
        "orientations": orientations,
        "positions": positions,
        "velocities": velocities,
    }


def _validated_case(case: Mapping[str, object]) -> dict[str, np.ndarray]:
    if set(case) != set(_CASE_SHAPES):
        raise ValueError("case fields must match the exact canonical contract")
    result: dict[str, np.ndarray] = {}
    for name in sorted(_CASE_SHAPES):
        value = np.asarray(case[name])
        if value.dtype == np.dtype(object):
            raise ValueError("canonical case must not contain object arrays")
        if value.dtype.kind not in "iuf" or value.shape != _CASE_SHAPES[name]:
            raise ValueError(f"{name} must be a numeric array with shape {_CASE_SHAPES[name]}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        result[name] = np.array(value, order="C", copy=True)
    if result["encoder_input"].dtype != np.dtype(np.float32):
        raise ValueError("encoder_input must be float32")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_case_artifacts(
    output_dir: str | Path,
    case: Mapping[str, object],
) -> GoldenCaseArtifacts:
    """Write deterministic NPZ, canonical sorted-key JSON, and raw float32 input."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = _validated_case(case)

    case_npz = destination / "golden_encoder_case.npz"
    np.savez(case_npz, **dict(sorted(arrays.items())))
    if _sha256(case_npz) != GOLDEN_CASE_SHA256:
        raise RuntimeError("canonical NPZ does not match the pinned golden case hash")

    case_json = destination / "sonic_encoder_case.json"
    json_payload = {name: arrays[name].tolist() for name in _JSON_FIELDS}
    case_json.write_text(
        json.dumps(json_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    encoder_input_raw = destination / "sonic_encoder_input.f32"
    encoder_input_raw.write_bytes(arrays["encoder_input"].tobytes(order="C"))
    if encoder_input_raw.stat().st_size != 4_988:
        raise RuntimeError("raw encoder input must contain exactly 1,247 float32 values")
    return GoldenCaseArtifacts(
        case_npz=case_npz,
        case_json=case_json,
        encoder_input_raw=encoder_input_raw,
    )


def encode_pinned_model(model_path: str | Path, encoder_input: np.ndarray) -> np.ndarray:
    lock = load_source_lock(_SOURCE_LOCK)
    verified_model = verify_file(model_path, lock.encoder.size, lock.encoder.sha256)
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required to freeze the golden token") from error

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(verified_model),
        providers=["CPUExecutionProvider"],
        sess_options=options,
    )
    return SonicEncoder.from_session(session).encode(encoder_input)


def write_golden_fixture(
    output_dir: str | Path,
    model_path: str | Path,
) -> GoldenFixtureArtifacts:
    """Build the canonical case and freeze its CPU ORT token from the pinned ONNX."""
    case = build_canonical_case()
    case_artifacts = write_case_artifacts(output_dir, case)
    token = encode_pinned_model(model_path, case["encoder_input"])

    token_npy = Path(output_dir) / "golden_encoder_token.npy"
    np.save(token_npy, token, allow_pickle=False)
    if _sha256(token_npy) != GOLDEN_TOKEN_SHA256:
        raise RuntimeError("CPU ORT token does not match the pinned golden token hash")
    ort_token_raw = Path(output_dir) / "sonic_encoder_ort_token.f32"
    ort_token_raw.write_bytes(token.tobytes(order="C"))
    if ort_token_raw.stat().st_size != 256:
        raise RuntimeError("raw ORT token must contain exactly 64 float32 values")
    return GoldenFixtureArtifacts(
        case_npz=case_artifacts.case_npz,
        case_json=case_artifacts.case_json,
        encoder_input_raw=case_artifacts.encoder_input_raw,
        token_npy=token_npy,
        ort_token_raw=ort_token_raw,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Pinned model_encoder.onnx")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    artifacts = write_golden_fixture(arguments.output_dir, arguments.model)
    print(
        json.dumps(
            {
                "case_npz": str(artifacts.case_npz),
                "case_sha256": _sha256(artifacts.case_npz),
                "token_npy": str(artifacts.token_npy),
                "token_sha256": _sha256(artifacts.token_npy),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
