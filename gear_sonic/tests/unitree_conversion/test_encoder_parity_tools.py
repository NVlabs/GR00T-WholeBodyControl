from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.data.unitree_conversion.provenance import load_source_lock, verify_file
from gear_sonic.scripts.build_unitree_sonic_encoder_golden import (
    GOLDEN_CASE_SHA256,
    GOLDEN_TOKEN_SHA256,
    build_canonical_case,
    write_case_artifacts,
    write_golden_fixture,
)
from gear_sonic.scripts.verify_unitree_sonic_encoder_parity import (
    launch_parity_gtest,
    raw_output_paths_from_environment,
    verify_raw_outputs_against_pinned,
)

GOLDEN_DIR = Path("gear_sonic/tests/data/unitree_conversion")


def _pinned_model_path() -> Path:
    override = os.environ.get("SONIC_PARITY_MODEL")
    model_path = (
        Path(override) if override is not None else Path("gear_sonic_deploy/policy/low_latency/model_encoder.onnx")
    )
    if not model_path.is_file():
        pytest.skip("pinned low-latency encoder is not materialized")
    lock = load_source_lock("gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml")
    return verify_file(model_path, lock.encoder.size, lock.encoder.sha256)


def test_seeded_case_builder_reproduces_checked_npz_and_sorted_json(tmp_path: Path) -> None:
    case = build_canonical_case()
    assert set(case) == {
        "current_observed_root_wxyz",
        "encoder_input",
        "future_reference_root_wxyz",
        "initial_observed_root_wxyz",
        "initial_reference_root_wxyz",
        "orientations",
        "positions",
        "velocities",
    }
    assert all(value.dtype != np.dtype(object) for value in case.values())

    first = write_case_artifacts(tmp_path / "first", case)
    second = write_case_artifacts(tmp_path / "second", build_canonical_case())

    assert first.case_npz.read_bytes() == second.case_npz.read_bytes()
    assert first.case_json.read_bytes() == second.case_json.read_bytes()
    assert first.encoder_input_raw.read_bytes() == second.encoder_input_raw.read_bytes()
    assert hashlib.sha256(first.case_npz.read_bytes()).hexdigest() == GOLDEN_CASE_SHA256
    assert first.case_npz.read_bytes() == (GOLDEN_DIR / "golden_encoder_case.npz").read_bytes()
    assert first.encoder_input_raw.stat().st_size == 4_988
    json_text = first.case_json.read_text(encoding="utf-8")
    assert json_text.endswith("\n")
    assert json_text == json.dumps(json.loads(json_text), sort_keys=True, separators=(",", ":")) + "\n"


def test_case_writer_rejects_object_arrays(tmp_path: Path) -> None:
    case = build_canonical_case()
    case["positions"] = np.array(case["positions"], dtype=object)

    with pytest.raises(ValueError, match="object arrays"):
        write_case_artifacts(tmp_path, case)


def test_pinned_builder_reproduces_checked_token(tmp_path: Path) -> None:
    artifacts = write_golden_fixture(tmp_path, _pinned_model_path())

    assert hashlib.sha256(artifacts.token_npy.read_bytes()).hexdigest() == GOLDEN_TOKEN_SHA256
    assert artifacts.token_npy.read_bytes() == (GOLDEN_DIR / "golden_encoder_token.npy").read_bytes()
    assert artifacts.ort_token_raw.stat().st_size == 256


def test_raw_output_environment_rejects_partial_pair() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        raw_output_paths_from_environment(
            {
                "SONIC_PARITY_INPUT_OUT": "/tmp/input.f32",
                "SONIC_PARITY_TOKEN_OUT": "",
            }
        )


def test_optional_deployment_outputs_match_python_and_golden() -> None:
    paths = raw_output_paths_from_environment(os.environ)
    if paths is None:
        pytest.skip("TensorRT parity raw outputs were not supplied")

    report = verify_raw_outputs_against_pinned(
        model_path=_pinned_model_path(),
        input_path=paths.input_path,
        token_path=paths.token_path,
        case_path=GOLDEN_DIR / "golden_encoder_case.npz",
        golden_token_path=GOLDEN_DIR / "golden_encoder_token.npy",
    )

    assert report.input_bytes == 4_988
    assert report.token_bytes == 256
    assert report.input_max_abs <= 1e-7
    assert report.token_max_abs <= 1e-5


def test_gtest_launcher_owns_exact_environment_and_output_sizes(tmp_path: Path) -> None:
    case_json = tmp_path / "case.json"
    case_json.write_text("{}", encoding="utf-8")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    input_out = tmp_path / "input.f32"
    token_out = tmp_path / "token.f32"
    calls: list[tuple[list[str], dict[str, str]]] = []

    def runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        calls.append((command, environment))
        Path(environment["SONIC_PARITY_INPUT_OUT"]).write_bytes(bytes(4_988))
        Path(environment["SONIC_PARITY_TOKEN_OUT"]).write_bytes(bytes(256))
        return SimpleNamespace(returncode=0, stdout="[  PASSED  ] 1 test.\n", stderr="")

    launch_parity_gtest(
        run_tests_path=tmp_path / "run_tests",
        case_json_path=case_json,
        model_path=model,
        input_path=input_out,
        token_path=token_out,
        runner=runner,
    )

    assert calls[0][0][-1] == "--gtest_filter=SonicEncoderParity.EncodesPinnedFloat32Input"
    assert calls[0][1]["SONIC_PARITY_CASE_JSON"] == str(case_json.resolve())
    assert calls[0][1]["SONIC_PARITY_MODEL"] == str(model.resolve())
    assert calls[0][1]["SONIC_PARITY_INPUT_OUT"] == str(input_out.resolve())
    assert calls[0][1]["SONIC_PARITY_TOKEN_OUT"] == str(token_out.resolve())


@pytest.mark.parametrize(
    ("returncode", "stdout", "message"),
    [
        (1, "", "failed with exit code 1"),
        (0, "[  SKIPPED ] 1 test.", "skipped"),
        (0, "", "passing GTest result"),
    ],
)
def test_gtest_launcher_rejects_failure_skip_or_missing_pass_marker(
    returncode: int,
    stdout: str,
    message: str,
    tmp_path: Path,
) -> None:
    def runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        del command, kwargs
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    with pytest.raises(RuntimeError, match=message):
        launch_parity_gtest(
            run_tests_path=tmp_path / "run_tests",
            case_json_path=tmp_path / "case.json",
            model_path=tmp_path / "model.onnx",
            input_path=tmp_path / "input.f32",
            token_path=tmp_path / "token.f32",
            runner=runner,
        )
