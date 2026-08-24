from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from gear_sonic.utils.g1_23dof_contract import (
    LOW_LATENCY_RELEASE_HF_REVISION,
    LOW_LATENCY_RELEASE_SHA256,
    REFERENCE_PROFILE_LOW_LATENCY,
)
import gear_sonic.utils.g1_29dof_low_latency_teacher as teacher_module
from gear_sonic.utils.g1_29dof_low_latency_teacher import (
    ExactLowLatencyTeacher,
    exact_fsq32,
)


def _toy_policy() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(23)
    result: dict[str, torch.Tensor] = {}
    for prefix, dims in (
        (teacher_module._ENCODER_PREFIX, (267, 7, 64)),  # noqa: SLF001
        (teacher_module._DECODER_PREFIX, (994, 11, 29)),  # noqa: SLF001
    ):
        for layer_index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            module_index = 2 * layer_index
            result[f"{prefix}{module_index}.weight"] = (
                torch.randn(output_dim, input_dim, generator=generator) * 0.01
            )
            result[f"{prefix}{module_index}.bias"] = torch.randn(output_dim, generator=generator) * 0.01
    return result


@pytest.fixture
def toy_teacher(monkeypatch: pytest.MonkeyPatch) -> ExactLowLatencyTeacher:
    release = {
        "source_revision": LOW_LATENCY_RELEASE_HF_REVISION,
        "reference_profile": REFERENCE_PROFILE_LOW_LATENCY,
    }
    checkpoint = {"policy_state_dict": _toy_policy()}
    monkeypatch.setattr(teacher_module, "ENCODER_DIMS", (267, 7, 64))
    monkeypatch.setattr(teacher_module, "DECODER_DIMS", (994, 11, 29))
    monkeypatch.setattr(
        teacher_module,
        "_load_pinned_legacy_release",
        lambda _path: (checkpoint, LOW_LATENCY_RELEASE_SHA256, release),
    )
    return ExactLowLatencyTeacher(Path("synthetic-low-latency.pt"))


def test_exact_fsq32_forward_values_and_shape() -> None:
    latent = torch.stack(
        (
            torch.zeros(64, dtype=torch.float32),
            torch.full((64,), 100.0, dtype=torch.float32),
            torch.full((64,), -100.0, dtype=torch.float32),
        )
    )
    quantized = exact_fsq32(latent)
    assert quantized.shape == (3, 64)
    assert quantized.dtype == torch.float32
    assert torch.equal(quantized[0], torch.zeros(64))
    assert torch.equal(quantized[1], torch.full((64,), 15.0 / 16.0))
    assert torch.equal(quantized[2], torch.full((64,), -1.0))


def test_synthetic_teacher_forward_is_finite_float32_and_deterministic(
    toy_teacher: ExactLowLatencyTeacher,
) -> None:
    rng = np.random.default_rng(29)
    semantic = rng.standard_normal((3, 267), dtype=np.float32)
    proprio = rng.standard_normal((3, 930), dtype=np.float32)
    first = toy_teacher.infer_batch(semantic, proprio)
    second = toy_teacher.infer_batch(semantic, proprio)
    assert first.shape == (3, 29)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert toy_teacher.infer(semantic[0], proprio[0]).shape == (29,)


@pytest.mark.parametrize(
    ("semantic", "proprio", "match"),
    [
        (np.zeros(266, dtype=np.float32), np.zeros(930, dtype=np.float32), "semantic267"),
        (np.zeros(267, dtype=np.float64), np.zeros(930, dtype=np.float32), "float32"),
        (np.zeros(267, dtype=np.float32), np.zeros(929, dtype=np.float32), "proprio930"),
        (
            np.zeros((2, 267), dtype=np.float32),
            np.zeros((1, 930), dtype=np.float32),
            "batch sizes",
        ),
    ],
)
def test_teacher_rejects_invalid_inputs(
    toy_teacher: ExactLowLatencyTeacher,
    semantic: np.ndarray,
    proprio: np.ndarray,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        toy_teacher.infer_batch(semantic, proprio)


def test_checkpoint_component_keys_are_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _toy_policy()
    policy[f"{teacher_module._ENCODER_PREFIX}unexpected"] = torch.zeros(1)  # noqa: SLF001
    release = {
        "source_revision": LOW_LATENCY_RELEASE_HF_REVISION,
        "reference_profile": REFERENCE_PROFILE_LOW_LATENCY,
    }
    monkeypatch.setattr(teacher_module, "ENCODER_DIMS", (267, 7, 64))
    monkeypatch.setattr(teacher_module, "DECODER_DIMS", (994, 11, 29))
    monkeypatch.setattr(
        teacher_module,
        "_load_pinned_legacy_release",
        lambda _path: ({"policy_state_dict": policy}, LOW_LATENCY_RELEASE_SHA256, release),
    )
    with pytest.raises(ValueError, match="keys differ"):
        ExactLowLatencyTeacher(Path("synthetic-low-latency.pt"))


def test_checkpoint_must_be_exact_low_latency_release(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        teacher_module,
        "_load_pinned_legacy_release",
        lambda _path: ({"policy_state_dict": _toy_policy()}, "0" * 64, {}),
    )
    with pytest.raises(ValueError, match="not pinned low_latency"):
        ExactLowLatencyTeacher(Path("wrong.pt"))


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REAL_CHECKPOINT = _REPOSITORY_ROOT / "low_latency" / "last.pt"
_RUN_REAL = os.environ.get("G1_RUN_REAL_LOW_LATENCY_CHECKPOINT_TEST") == "1"


@pytest.mark.skipif(
    not (_RUN_REAL and _REAL_CHECKPOINT.is_file()),
    reason="set G1_RUN_REAL_LOW_LATENCY_CHECKPOINT_TEST=1 for 1.1 GB checkpoint test",
)
def test_real_checkpoint_forward_is_finite_and_deterministic() -> None:
    teacher = ExactLowLatencyTeacher(_REAL_CHECKPOINT, device="cpu")
    semantic = np.zeros(267, dtype=np.float32)
    proprio = np.zeros(930, dtype=np.float32)
    first = teacher.infer(semantic, proprio)
    second = teacher.infer(semantic, proprio)
    assert first.shape == (29,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert teacher.checkpoint_sha256 == LOW_LATENCY_RELEASE_SHA256
