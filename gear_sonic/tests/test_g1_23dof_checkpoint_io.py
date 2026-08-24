from pathlib import Path

import pytest
import torch

from gear_sonic.scripts import init_g1_23dof_checkpoint as initializer
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    INITIALIZATION_STAGE,
    SAFE_CHECKPOINT_HEADER,
    TRAINED_STAGE,
    build_safe_initialization_checkpoint,
    build_safe_promotion_checkpoint,
    checkpoint_stage,
    extract_global_step,
    load_safe_true23_checkpoint,
    load_true23_trainer_checkpoint,
    promotion_checkpoint_path,
)


def _mark_executed(path: str) -> int:
    Path(path).write_text("executed", encoding="utf-8")
    return 0


class _PickleExploit:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return _mark_executed, (str(self.marker),)


def _policy() -> dict[str, torch.Tensor]:
    return {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}


def _metadata(stage: str) -> dict[str, object]:
    return {
        "checkpoint_stage": stage,
        "nested": {"enabled": True, "indices": [0, 12, 15, 26]},
    }


def test_safe_promotion_round_trip_uses_mapping_global_step(tmp_path):
    checkpoint = build_safe_promotion_checkpoint(
        policy_state_dict=_policy(),
        global_step=1200,
        metadata=_metadata(TRAINED_STAGE),
        training_evidence={"global_step": 1200, "producer": "test"},
    )
    path = tmp_path / "last.promotion.pt"
    torch.save(checkpoint, path)

    loaded = load_safe_true23_checkpoint(path)

    assert checkpoint_stage(loaded) == TRAINED_STAGE
    assert extract_global_step(loaded) == 1200
    assert loaded[SAFE_CHECKPOINT_HEADER]["resume_state_included"] is False
    assert torch.equal(loaded["policy_state_dict"]["weight"], _policy()["weight"])


def test_safe_initialization_round_trip_has_no_resume_state(tmp_path):
    checkpoint = build_safe_initialization_checkpoint(
        policy_state_dict=_policy(),
        metadata=_metadata(INITIALIZATION_STAGE),
        initialization_report={"initialization_only": True},
    )
    path = tmp_path / "init.pt"
    torch.save(checkpoint, path)

    loaded = load_safe_true23_checkpoint(path)

    assert checkpoint_stage(loaded) == INITIALIZATION_STAGE
    assert "state" not in loaded


def test_safe_builder_rejects_nonprimitive_metadata():
    with pytest.raises(ValueError, match="unsafe value type"):
        build_safe_initialization_checkpoint(
            policy_state_dict=_policy(),
            metadata={
                "checkpoint_stage": INITIALIZATION_STAGE,
                "unsafe": object(),
            },
            initialization_report={"initialization_only": True},
        )


def test_weights_only_loader_rejects_pickle_global_without_execution(tmp_path):
    marker = tmp_path / "executed.txt"
    malicious = tmp_path / "malicious.pt"
    torch.save({"payload": _PickleExploit(marker)}, malicious)

    with pytest.raises(ValueError, match="safe weights-only true23 artifact"):
        load_safe_true23_checkpoint(malicious)

    assert not marker.exists()


def test_true23_trainer_initialization_rejects_pickle_global_without_execution(
    tmp_path,
):
    marker = tmp_path / "trainer-executed.txt"
    malicious = tmp_path / "malicious-init.pt"
    torch.save({"payload": _PickleExploit(marker)}, malicious)

    with pytest.raises(ValueError, match="safe weights-only true23 artifact"):
        load_true23_trainer_checkpoint(
            malicious,
            resume=False,
            map_location="cpu",
        )

    assert not marker.exists()


def test_schema_rejects_resume_or_unknown_root_state(tmp_path):
    checkpoint = build_safe_promotion_checkpoint(
        policy_state_dict=_policy(),
        global_step=5,
        metadata=_metadata(TRAINED_STAGE),
        training_evidence={"global_step": 5},
    )
    checkpoint["optimizer_state_dict"] = {"unsafe_resume_state": 1}
    path = tmp_path / "not-promotion.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="unexpected root keys"):
        load_safe_true23_checkpoint(path)

    with pytest.raises(ValueError, match="contain only global_step"):
        extract_global_step({"state": {"global_step": 5, "epoch": 1}})


def test_legacy_initialization_requires_explicit_compatibility_flag(tmp_path):
    legacy = {
        "policy_state_dict": _policy(),
        "g1_23dof_metadata": _metadata(INITIALIZATION_STAGE),
        "g1_23dof_initialization_report": {"initialization_only": True},
    }
    path = tmp_path / "legacy-init.pt"
    torch.save(legacy, path)

    with pytest.raises(ValueError, match="lacks the safe true23 schema header"):
        load_safe_true23_checkpoint(path)

    loaded = load_safe_true23_checkpoint(
        path,
        allow_legacy_initialization=True,
    )
    assert checkpoint_stage(loaded) == INITIALIZATION_STAGE


def test_promotion_checkpoint_path_is_separate():
    assert promotion_checkpoint_path("checkpoints/last.pt") == Path(
        "checkpoints/last.promotion.pt"
    )
    assert promotion_checkpoint_path("checkpoints/model") == Path(
        "checkpoints/model.promotion.pt"
    )


def test_true23_trainer_full_resume_is_explicit_trusted_boundary(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "full-resume.pt"
    path.write_bytes(b"fixture")
    state = object()
    calls = []

    def fake_load(checkpoint_path, *, map_location, weights_only):
        calls.append((checkpoint_path, map_location, weights_only))
        return {"state": state, "optimizer_state_dict": {"step": 1}}

    monkeypatch.setattr(
        "gear_sonic.utils.g1_23dof_checkpoint_io.torch.load",
        fake_load,
    )

    loaded = load_true23_trainer_checkpoint(
        path,
        resume=True,
        map_location="cuda:0",
    )

    assert loaded["state"] is state
    assert calls == [(path.resolve(), "cuda:0", False)]


def test_true23_trainer_rejects_truthy_nonboolean_resume_before_load(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "full-resume.pt"
    path.write_bytes(b"fixture")
    called = False

    def fake_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("ambiguous resume value must fail before load")

    monkeypatch.setattr(
        "gear_sonic.utils.g1_23dof_checkpoint_io.torch.load",
        fake_load,
    )

    with pytest.raises(TypeError, match="explicit bool"):
        load_true23_trainer_checkpoint(
            path,
            resume="false",
            map_location="cpu",
        )
    assert called is False


def test_safe_promotion_cannot_be_used_as_full_resume(tmp_path):
    checkpoint = build_safe_promotion_checkpoint(
        policy_state_dict=_policy(),
        global_step=12,
        metadata=_metadata(TRAINED_STAGE),
        training_evidence={"global_step": 12},
    )
    path = tmp_path / "promotion.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="cannot be used with resume=True"):
        load_true23_trainer_checkpoint(
            path,
            resume=True,
            map_location="cpu",
        )


def test_converter_rejects_unpinned_file_before_unsafe_load(tmp_path, monkeypatch):
    source = tmp_path / "unknown.pt"
    source.write_bytes(b"not an approved release")
    called = False

    def fake_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run before hash approval")

    monkeypatch.setattr(initializer.torch, "load", fake_load)

    with pytest.raises(ValueError, match="no exact pinned release match"):
        initializer.load_pinned_legacy_release(source)
    assert called is False


@pytest.mark.parametrize(
    "release_sha256",
    [
        initializer.LEGACY_SONIC_RELEASE_SHA256,
        initializer.LOW_LATENCY_SONIC_RELEASE_SHA256,
    ],
)
def test_converter_unsafe_load_is_limited_to_exact_pins(
    tmp_path,
    monkeypatch,
    release_sha256,
):
    source = tmp_path / "approved.pt"
    source.write_bytes(b"test fixture")
    calls = []
    monkeypatch.setattr(initializer, "_sha256", lambda path: release_sha256)

    def fake_load(path, *, map_location, weights_only):
        calls.append((path, map_location, weights_only))
        return {"policy_state_dict": _policy()}

    monkeypatch.setattr(initializer.torch, "load", fake_load)

    loaded = initializer.load_pinned_legacy_release(source)

    assert "policy_state_dict" in loaded
    assert calls == [(source.resolve(), "cpu", False)]
