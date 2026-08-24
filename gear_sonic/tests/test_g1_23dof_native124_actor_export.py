"""Focused tests for the standalone native124 checkpoint actor exporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from gear_sonic.utils import g1_23dof_native124_actor_export as exporter
from gear_sonic.utils.g1_23dof_native124_actor_export import (
    ACTION_DIM,
    NORMALIZATION_EPSILON,
    OBSERVATION_DIM,
    export_native124_actor,
    load_native124_actor,
    run_native124_actor,
    sha256_file,
    verify_actor_onnx_parity,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_11500 = (
    REPOSITORY_ROOT
    / "external_dependencies"
    / "unitree_rl_mjlab"
    / "logs"
    / "rsl_rl"
    / "g1_23dof_tracking"
    / "2026-08-04_21-55-39_dad_dance_full_resume_2500_to30000_512"
    / "model_11500.pt"
)
MODEL_11500_SHA256 = "1302ed2d7128c5f129611c29a34181d1ac7e27d2c15f551e49453e41ee81ec4a"
OFFICIAL_ACTOR = REPOSITORY_ROOT / "artifacts" / "g1_true23_step1c_native_policy_model_11500" / "actor.onnx"
OFFICIAL_ACTOR_SHA256 = "b0476b3e5d281d2f3bb47efd5dfe8fdf2fd93a3daf2d6c752d977a9ba7e05b02"


def _actor_state(seed: int = 7) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    variance = torch.linspace(0.04, 4.0, OBSERVATION_DIM).reshape(1, -1)

    def random_tensor(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.02

    return {
        "obs_normalizer._mean": random_tensor(1, OBSERVATION_DIM),
        "obs_normalizer._var": variance,
        "obs_normalizer._std": torch.sqrt(variance),
        "obs_normalizer.count": torch.tensor(12345, dtype=torch.int64),
        "distribution.std_param": torch.full((ACTION_DIM,), 0.25),
        "mlp.0.weight": random_tensor(512, OBSERVATION_DIM),
        "mlp.0.bias": random_tensor(512),
        "mlp.2.weight": random_tensor(256, 512),
        "mlp.2.bias": random_tensor(256),
        "mlp.4.weight": random_tensor(128, 256),
        "mlp.4.bias": random_tensor(128),
        "mlp.6.weight": random_tensor(ACTION_DIM, 128),
        "mlp.6.bias": random_tensor(ACTION_DIM),
    }


def _write_checkpoint(
    path: Path,
    state: dict[str, Any] | None = None,
    *,
    iteration: int = 42,
) -> None:
    torch.save(
        {
            "actor_state_dict": state if state is not None else _actor_state(),
            "critic_state_dict": {"unused": torch.ones(1)},
            "infos": {"env_state": {"common_step_counter": 10}},
            "iter": iteration,
            "optimizer_state_dict": {},
        },
        path,
    )


def test_safe_loader_uses_weights_only_and_reconstructs_exact_actor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"stable fake checkpoint bytes")
    state = _actor_state()
    calls: list[tuple[Path, dict[str, Any]]] = []

    def fake_load(path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append((path, kwargs))
        return {"actor_state_dict": state, "iter": 42}

    monkeypatch.setattr(exporter.torch, "load", fake_load)
    actor, lineage = load_native124_actor(checkpoint)

    assert calls == [(checkpoint.resolve(), {"map_location": "cpu", "weights_only": True})]
    assert lineage["safe_load"]["torch_weights_only"] is True
    assert lineage["checkpoint"]["iteration"] == 42
    assert set(actor.state_dict()) == set(state)
    assert all(torch.equal(actor.state_dict()[key], value) for key, value in state.items())

    observation = np.linspace(-1.0, 1.0, OBSERVATION_DIM, dtype=np.float32)[None, :]
    latent = (torch.from_numpy(observation) - state["obs_normalizer._mean"]) / (
        state["obs_normalizer._std"] + NORMALIZATION_EPSILON
    )
    for index in (0, 2, 4):
        latent = F.elu(F.linear(latent, state[f"mlp.{index}.weight"], state[f"mlp.{index}.bias"]))
    expected = F.linear(latent, state["mlp.6.weight"], state["mlp.6.bias"])
    assert np.array_equal(run_native124_actor(actor, observation), expected.numpy())


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("unexpected_key", "key mismatch"),
        ("missing_key", "key mismatch"),
        ("wrong_shape", "shape mismatch"),
        ("wrong_dtype", "dtype mismatch"),
        ("wrong_type", "exact torch.Tensor"),
        ("nonfinite", "non-finite"),
        ("inconsistent_std", "inconsistent"),
    ),
)
def test_loader_rejects_actor_schema_drift(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    state: dict[str, Any] = _actor_state()
    if corruption == "unexpected_key":
        state["unexpected"] = torch.zeros(1)
    elif corruption == "missing_key":
        del state["mlp.6.bias"]
    elif corruption == "wrong_shape":
        state["mlp.6.bias"] = torch.zeros(ACTION_DIM + 1)
    elif corruption == "wrong_dtype":
        state["mlp.6.bias"] = torch.zeros(ACTION_DIM, dtype=torch.float64)
    elif corruption == "wrong_type":
        state["mlp.6.bias"] = [0.0] * ACTION_DIM
    elif corruption == "nonfinite":
        state["mlp.6.bias"][0] = torch.nan
    elif corruption == "inconsistent_std":
        state["obs_normalizer._std"] += 1.0
    checkpoint = tmp_path / f"{corruption}.pt"
    _write_checkpoint(checkpoint, state)
    with pytest.raises(ValueError, match=message):
        load_native124_actor(checkpoint)


def test_runtime_rejects_wrong_input_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint)
    actor, _ = load_native124_actor(checkpoint)
    with pytest.raises(TypeError, match="numpy.ndarray"):
        run_native124_actor(actor, [0.0] * OBSERVATION_DIM)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dtype"):
        run_native124_actor(actor, np.zeros((1, OBSERVATION_DIM), dtype=np.float64))
    with pytest.raises(ValueError, match="shape"):
        run_native124_actor(actor, np.zeros(OBSERVATION_DIM, dtype=np.float32))
    invalid = np.zeros((1, OBSERVATION_DIM), dtype=np.float32)
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        run_native124_actor(actor, invalid)


def test_export_refuses_to_overwrite_before_loading_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "actor.onnx"
    output.write_bytes(b"sentinel")
    with pytest.raises(FileExistsError, match="overwrite"):
        export_native124_actor(
            tmp_path / "missing.pt",
            output,
            tmp_path / "lineage.json",
        )
    assert output.read_bytes() == b"sentinel"


def test_export_static_onnx_and_hash_bound_lineage(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "actor.onnx"
    report_path = tmp_path / "lineage.json"
    _write_checkpoint(checkpoint, iteration=99)

    report = export_native124_actor(checkpoint, output, report_path)

    assert output.is_file() and report_path.is_file()
    assert report["checkpoint_lineage"]["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert report["export"]["output_sha256"] == sha256_file(output)
    assert report["export"]["no_overwrite"] is True
    assert report["runtime"]["environment_constructed"] is False
    assert report["parity"]["exported_onnx_vs_checkpoint"]["comparison"]["passed"] is True
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="overwrite"):
        export_native124_actor(checkpoint, output, report_path)


@pytest.mark.skipif(
    not MODEL_11500.is_file() or not OFFICIAL_ACTOR.is_file(),
    reason="local immutable model11500 fixtures are unavailable",
)
def test_model_11500_matches_immutable_official_actor_and_fresh_export(
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    assert sha256_file(MODEL_11500) == MODEL_11500_SHA256
    assert sha256_file(OFFICIAL_ACTOR) == OFFICIAL_ACTOR_SHA256

    actor, lineage = load_native124_actor(MODEL_11500)
    official_parity = verify_actor_onnx_parity(
        actor,
        OFFICIAL_ACTOR,
        expected_sha256=OFFICIAL_ACTOR_SHA256,
    )
    assert lineage["checkpoint"]["iteration"] == 11500
    assert official_parity["comparison"]["passed"] is True
    assert official_parity["comparison"]["max_absolute_error"] < 1.0e-5

    output = tmp_path / "actor.onnx"
    report_path = tmp_path / "lineage.json"
    report = export_native124_actor(
        MODEL_11500,
        output,
        report_path,
        reference_onnx_path=OFFICIAL_ACTOR,
        expected_reference_sha256=OFFICIAL_ACTOR_SHA256,
    )
    assert report["parity"]["reference_onnx_vs_checkpoint"]["comparison"]["passed"] is True
    assert report["parity"]["exported_onnx_vs_checkpoint"]["comparison"]["passed"] is True
    assert report["parity"]["exported_onnx_vs_reference"]["comparison"]["passed"] is True
