"""Run and verify pinned CPU-ORT versus deployment-TensorRT SONIC parity."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from gear_sonic.data.unitree_conversion.provenance import load_source_lock, verify_file
from gear_sonic.scripts.build_unitree_sonic_encoder_golden import (
    GOLDEN_CASE_SHA256,
    GOLDEN_TOKEN_SHA256,
    encode_pinned_model,
    write_golden_fixture,
)

_SOURCE_LOCK = Path("gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml")
_GTEST_FILTER = "--gtest_filter=SonicEncoderParity.EncodesPinnedFloat32Input"


@dataclass(frozen=True)
class RawOutputPaths:
    input_path: Path
    token_path: Path


@dataclass(frozen=True)
class RawParityReport:
    input_bytes: int
    token_bytes: int
    input_max_abs: float
    token_max_abs: float


@dataclass(frozen=True)
class DeploymentParityReport(RawParityReport):
    model_path: str
    case_sha256: str
    token_sha256: str
    gtest_stdout: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_output_paths_from_environment(
    environ: Mapping[str, str] | None = None,
) -> RawOutputPaths | None:
    """Resolve the opt-in raw pair, rejecting one-sided/stale test configuration."""
    source = os.environ if environ is None else environ
    input_value = source.get("SONIC_PARITY_INPUT_OUT", "")
    token_value = source.get("SONIC_PARITY_TOKEN_OUT", "")
    if bool(input_value) != bool(token_value):
        raise ValueError("SONIC_PARITY_INPUT_OUT and SONIC_PARITY_TOKEN_OUT must be set together")
    if not input_value:
        return None
    return RawOutputPaths(Path(input_value), Path(token_value))


def launch_parity_gtest(
    *,
    run_tests_path: str | Path,
    case_json_path: str | Path,
    model_path: str | Path,
    input_path: str | Path,
    token_path: str | Path,
    runner: Callable[..., Any] = subprocess.run,
) -> str:
    """Launch exactly one parity GTest and require fresh, exact-sized raw outputs."""
    raw_input = Path(input_path).resolve()
    raw_token = Path(token_path).resolve()
    raw_input.parent.mkdir(parents=True, exist_ok=True)
    raw_token.parent.mkdir(parents=True, exist_ok=True)
    raw_input.unlink(missing_ok=True)
    raw_token.unlink(missing_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "SONIC_PARITY_CASE_JSON": str(Path(case_json_path).resolve()),
            "SONIC_PARITY_MODEL": str(Path(model_path).resolve()),
            "SONIC_PARITY_INPUT_OUT": str(raw_input),
            "SONIC_PARITY_TOKEN_OUT": str(raw_token),
        }
    )
    command = [str(Path(run_tests_path).resolve()), _GTEST_FILTER]
    result = runner(
        command,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = str(getattr(result, "stdout", ""))
    stderr = str(getattr(result, "stderr", ""))
    returncode = int(getattr(result, "returncode", -1))
    combined = stdout + "\n" + stderr
    if returncode != 0:
        raise RuntimeError(f"parity GTest failed with exit code {returncode}: {combined}")
    if "SKIPPED" in combined.upper():
        raise RuntimeError(f"parity GTest skipped instead of running: {combined}")
    if "[  PASSED  ] 1 test." not in combined:
        raise RuntimeError(f"parity harness did not observe one passing GTest result: {combined}")
    for path, byte_count, label in (
        (raw_input, 4_988, "input"),
        (raw_token, 256, "token"),
    ):
        if not path.is_file():
            raise RuntimeError(f"parity GTest did not produce the {label} output")
        if path.stat().st_size != byte_count:
            raise RuntimeError(f"parity {label} output must contain exactly {byte_count} bytes")
    return stdout


def _allclose(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
    label: str,
) -> None:
    try:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise ValueError(f"{label} parity check failed: {error}") from error


def verify_raw_outputs_against_pinned(
    *,
    model_path: str | Path,
    input_path: str | Path,
    token_path: str | Path,
    case_path: str | Path,
    golden_token_path: str | Path,
) -> RawParityReport:
    """Bind raw C++ outputs to the pinned model, CPU ORT, and immutable fixtures."""
    case_file = Path(case_path)
    golden_file = Path(golden_token_path)
    if _sha256(case_file) != GOLDEN_CASE_SHA256:
        raise ValueError("golden encoder case hash does not match the pinned fixture")
    if _sha256(golden_file) != GOLDEN_TOKEN_SHA256:
        raise ValueError("golden encoder token hash does not match the pinned fixture")

    raw_input_bytes = Path(input_path).read_bytes()
    raw_token_bytes = Path(token_path).read_bytes()
    if len(raw_input_bytes) != 4_988:
        raise ValueError("TensorRT input output must contain exactly 4,988 bytes")
    if len(raw_token_bytes) != 256:
        raise ValueError("TensorRT token output must contain exactly 256 bytes")
    cpp_input = np.frombuffer(raw_input_bytes, dtype=np.float32).reshape(1, 1247)
    trt_token = np.frombuffer(raw_token_bytes, dtype=np.float32).reshape(1, 64)
    if not np.isfinite(cpp_input).all() or not np.isfinite(trt_token).all():
        raise ValueError("TensorRT parity outputs must contain only finite float32 values")

    with np.load(case_file, allow_pickle=False) as case:
        encoder_input = np.array(case["encoder_input"], dtype=np.float32, order="C", copy=True)
    golden_token = np.load(golden_file, allow_pickle=False)
    if encoder_input.shape != (1, 1247) or encoder_input.dtype != np.dtype(np.float32):
        raise ValueError("golden encoder input must have exact shape [1,1247] and dtype float32")
    if golden_token.shape != (1, 64) or golden_token.dtype != np.dtype(np.float32):
        raise ValueError("golden token must have exact shape [1,64] and dtype float32")

    cpu_token = encode_pinned_model(model_path, encoder_input)
    _allclose(cpp_input, encoder_input, atol=1e-7, rtol=0.0, label="C++ packed input")
    _allclose(cpu_token, golden_token, atol=1e-5, rtol=1e-5, label="CPU ORT versus golden token")
    _allclose(trt_token, cpu_token, atol=1e-5, rtol=1e-5, label="TensorRT versus CPU ORT token")
    _allclose(trt_token, golden_token, atol=1e-5, rtol=1e-5, label="TensorRT versus golden token")
    return RawParityReport(
        input_bytes=len(raw_input_bytes),
        token_bytes=len(raw_token_bytes),
        input_max_abs=float(np.max(np.abs(cpp_input - encoder_input))),
        token_max_abs=float(
            max(
                np.max(np.abs(trt_token - cpu_token)),
                np.max(np.abs(trt_token - golden_token)),
            )
        ),
    )


def run_deployment_parity(
    *,
    model_path: str | Path,
    run_tests_path: str | Path,
    output_dir: str | Path,
    golden_dir: str | Path = Path("gear_sonic/tests/data/unitree_conversion"),
    runner: Callable[..., Any] = subprocess.run,
) -> DeploymentParityReport:
    """Generate, run, and compare the complete pinned parity chain in one call."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    lock = load_source_lock(_SOURCE_LOCK)
    verified_model = verify_file(model_path, lock.encoder.size, lock.encoder.sha256)
    generated = write_golden_fixture(destination / "generated", verified_model)
    tracked_dir = Path(golden_dir)
    tracked_case = tracked_dir / "golden_encoder_case.npz"
    tracked_token = tracked_dir / "golden_encoder_token.npy"
    if generated.case_npz.read_bytes() != tracked_case.read_bytes():
        raise RuntimeError("generated encoder case differs from the tracked golden fixture")
    if generated.token_npy.read_bytes() != tracked_token.read_bytes():
        raise RuntimeError("generated CPU token differs from the tracked golden fixture")

    input_path = destination / "tensor_rt_encoder_input.f32"
    token_path = destination / "tensor_rt_encoder_token.f32"
    stdout = launch_parity_gtest(
        run_tests_path=run_tests_path,
        case_json_path=generated.case_json,
        model_path=verified_model,
        input_path=input_path,
        token_path=token_path,
        runner=runner,
    )
    raw_report = verify_raw_outputs_against_pinned(
        model_path=verified_model,
        input_path=input_path,
        token_path=token_path,
        case_path=tracked_case,
        golden_token_path=tracked_token,
    )
    return DeploymentParityReport(
        **asdict(raw_report),
        model_path=str(verified_model),
        case_sha256=GOLDEN_CASE_SHA256,
        token_sha256=GOLDEN_TOKEN_SHA256,
        gtest_stdout=stdout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Pinned model_encoder.onnx")
    parser.add_argument("--run-tests", type=Path, required=True, help="Deployment run_tests binary")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path("gear_sonic/tests/data/unitree_conversion"),
    )
    arguments = parser.parse_args()
    report = run_deployment_parity(
        model_path=arguments.model,
        run_tests_path=arguments.run_tests,
        output_dir=arguments.output_dir,
        golden_dir=arguments.golden_dir,
    )
    print(json.dumps(asdict(report), sort_keys=True))


if __name__ == "__main__":
    main()
