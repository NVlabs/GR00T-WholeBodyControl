from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import pytest
import torch

from gear_sonic.trl.mjlab import supported_idle_checkpoint as checkpoint_module
from gear_sonic.trl.mjlab.supported_idle_checkpoint import (
    ACTOR_STATE_SCHEMA,
    CRITIC_STATE_SCHEMA,
    OPTIMIZER_PARAMETER_SPECS,
    TensorSpec,
    build_checkpoint,
    capture_rng_state,
    load_checkpoint,
    save_checkpoint_exclusive,
    validate_checkpoint,
)

HASH = "a" * 64
COMMAND_SCHEMA = {
    "time_steps": TensorSpec((2,), torch.int64),
    "span_ids": TensorSpec((2,), torch.int64),
}


def _lineage() -> dict[str, object]:
    return {
        "plan_payload_sha256": HASH,
        "plan_file_sha256": "b" * 64,
        "authorization_file_sha256": "c" * 64,
        "job_id": "static_model3500",
        "source_checkpoint_sha256": "d" * 64,
        "source_kind": "dad_dance_seed",
        "corpus_sha256": "e" * 64,
        "sidecar_sha256": "f" * 64,
        "runtime_file_sha256": {"runner.py": "1" * 64},
        "package_sha256": {"torch==2.9.0+cu128": "2" * 64},
    }


def _adam_state() -> dict[str, object]:
    parameters = [
        torch.nn.Parameter(torch.ones(spec.shape, dtype=spec.dtype)) for spec in OPTIMIZER_PARAMETER_SPECS
    ]
    optimizer = torch.optim.Adam(parameters, lr=1.0e-4)
    sum(parameter.sum() for parameter in parameters).backward()
    optimizer.step()
    return optimizer.state_dict()


def _checkpoint() -> dict[str, object]:
    return build_checkpoint(
        actor_state_dict={
            key: torch.ones(spec.shape, dtype=spec.dtype) for key, spec in ACTOR_STATE_SCHEMA.items()
        },
        critic_state_dict={
            key: torch.ones(spec.shape, dtype=spec.dtype) for key, spec in CRITIC_STATE_SCHEMA.items()
        },
        optimizer_state_dict=_adam_state(),
        completed_updates=4,
        env_common_step_counter=96,
        lineage=_lineage(),
        optimizer_semantics={
            "initialization": "seed_actor_critic_only",
            "source_optimizer_loaded": False,
            "optimizer_reset": True,
            "resume_loads_optimizer": True,
        },
        rng_state=capture_rng_state(),
        command_state={"time_steps": torch.tensor([1, 2]), "span_ids": torch.tensor([3, 4])},
        command_schema=COMMAND_SCHEMA,
    )


def test_round_trip_is_weights_only_and_has_unambiguous_counter(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    if os.name == "nt":
        with pytest.raises(OSError, match="directory fsync"):
            save_checkpoint_exclusive(tmp_path / "model_4.pt", checkpoint, command_schema=COMMAND_SCHEMA)
        return
    published = save_checkpoint_exclusive(tmp_path / "model_4.pt", checkpoint, command_schema=COMMAND_SCHEMA)
    restored = load_checkpoint(
        published.path,
        expected_sha256=published.sha256,
        expected_lineage=_lineage(),
        expected_rng_execution_mode="cpu_only",
        expected_cuda_devices=(),
        command_schema=COMMAND_SCHEMA,
    )
    assert restored["completed_updates"] == 4
    assert restored["iter"] == 3
    assert torch.equal(restored["command_state"]["time_steps"], torch.tensor([1, 2]))


def test_exclusive_save_never_overwrites(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows lacks standard-library directory fsync")
    path = tmp_path / "model_4.pt"
    save_checkpoint_exclusive(path, _checkpoint(), command_schema=COMMAND_SCHEMA)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_checkpoint_exclusive(path, _checkpoint(), command_schema=COMMAND_SCHEMA)


def test_exclusive_publication_cannot_overwrite_racing_creator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("Windows lacks standard-library directory fsync")
    output = tmp_path / "model_4.pt"
    original_link = checkpoint_module.os.link

    def racing_link(source: object, destination: object, *, follow_symlinks: bool) -> None:
        Path(destination).write_bytes(b"racer")
        original_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(checkpoint_module.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_checkpoint_exclusive(output, _checkpoint(), command_schema=COMMAND_SCHEMA)
    assert output.read_bytes() == b"racer"


def test_publication_detects_post_link_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.name == "nt":
        pytest.skip("Windows lacks standard-library directory fsync")
    output = tmp_path / "model_4.pt"
    original_link = checkpoint_module.os.link

    def mutating_link(source: object, destination: object, *, follow_symlinks: bool) -> None:
        original_link(source, destination, follow_symlinks=follow_symlinks)
        os.chmod(destination, 0o644)
        Path(destination).write_bytes(b"tampered")

    monkeypatch.setattr(checkpoint_module.os, "link", mutating_link)
    with pytest.raises(RuntimeError, match="changed during publication"):
        save_checkpoint_exclusive(output, _checkpoint(), command_schema=COMMAND_SCHEMA)
    assert not output.exists()


def test_temporary_path_swap_never_touches_victim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.name == "nt":
        pytest.skip("Windows lacks standard-library directory fsync")
    output = tmp_path / "model_4.pt"
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"victim")
    original_link = checkpoint_module.os.link

    def swapping_link(source: object, destination: object, *, follow_symlinks: bool) -> None:
        source_path = Path(source)
        source_path.unlink()
        source_path.symlink_to(victim)
        original_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(checkpoint_module.os, "link", swapping_link)
    with pytest.raises(RuntimeError, match="inode differs"):
        save_checkpoint_exclusive(output, _checkpoint(), command_schema=COMMAND_SCHEMA)
    assert victim.read_bytes() == b"victim"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.update(iter=4), "iter"),
        (lambda value: value["lineage"].update(unexpected=HASH), "lineage keys"),
        (
            lambda value: value["command_state"].update(extra=torch.ones(2, dtype=torch.int64)),
            "command_state keys",
        ),
        (
            lambda value: value["command_state"].update(time_steps=torch.ones(2, dtype=torch.float32)),
            "dtype/shape",
        ),
        (
            lambda value: value["actor_state_dict"].update(extra=torch.ones(1)),
            "actor_state_dict keys",
        ),
        (
            lambda value: value["actor_state_dict"].pop("mlp.0.bias"),
            "actor_state_dict keys",
        ),
        (
            lambda value: value["actor_state_dict"].update(
                **{"mlp.0.weight": torch.ones((512, 124), dtype=torch.float64)}
            ),
            "dtype/shape",
        ),
        (
            lambda value: value["actor_state_dict"].update(
                **{"mlp.0.weight": torch.full((512, 124), float("nan"))}
            ),
            "non-finite",
        ),
        (
            lambda value: (
                value["rng_state"]["torch_cpu"].update() if False else value["rng_state"].update(unexpected=1)
            ),
            "rng_state keys",
        ),
    ],
)
def test_tampering_fails_closed(mutate: object, match: str) -> None:
    checkpoint = deepcopy(_checkpoint())
    mutate(checkpoint)
    with pytest.raises(ValueError, match=match):
        validate_checkpoint(checkpoint, command_schema=COMMAND_SCHEMA)


def test_optimizer_semantics_rejects_seed_optimizer_import() -> None:
    checkpoint = _checkpoint()
    checkpoint["optimizer_semantics"]["source_optimizer_loaded"] = True
    with pytest.raises(ValueError, match="contradicts"):
        build_checkpoint(
            actor_state_dict=checkpoint["actor_state_dict"],
            critic_state_dict=checkpoint["critic_state_dict"],
            optimizer_state_dict=checkpoint["optimizer_state_dict"],
            completed_updates=checkpoint["completed_updates"],
            env_common_step_counter=checkpoint["trainer_state"]["env_common_step_counter"],
            lineage=checkpoint["lineage"],
            optimizer_semantics=checkpoint["optimizer_semantics"],
            rng_state=checkpoint["rng_state"],
            command_state=checkpoint["command_state"],
            command_schema=COMMAND_SCHEMA,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["optimizer_state_dict"].update(unexpected=True),
            "optimizer_state_dict keys",
        ),
        (
            lambda value: value["lineage"].update(source_kind="same_job_resume"),
            "source_kind unsupported",
        ),
        (
            lambda value: value["rng_state"].update(execution_mode="cuda_full"),
            "requires CUDA",
        ),
        (
            lambda value: value[checkpoint_module.HEADER_KEY].update(schema_version=True),
            "header mismatch",
        ),
    ],
)
def test_metadata_contracts_fail_closed(mutate: object, match: str) -> None:
    checkpoint = deepcopy(_checkpoint())
    mutate(checkpoint)
    with pytest.raises(ValueError, match=match):
        validate_checkpoint(checkpoint, command_schema=COMMAND_SCHEMA)


def test_completed_checkpoint_rejects_empty_or_partial_optimizer_state() -> None:
    empty = _checkpoint()
    empty["optimizer_state_dict"]["state"] = {}
    with pytest.raises(ValueError, match="cover native PPO"):
        validate_checkpoint(empty, command_schema=COMMAND_SCHEMA)
    partial = _checkpoint()
    parameter_id = next(iter(partial["optimizer_state_dict"]["state"]))
    partial["optimizer_state_dict"]["state"].pop(parameter_id)
    with pytest.raises(ValueError, match="cover native PPO"):
        validate_checkpoint(partial, command_schema=COMMAND_SCHEMA)


def test_completed_checkpoint_rejects_one_parameter_optimizer() -> None:
    checkpoint = _checkpoint()
    checkpoint["optimizer_state_dict"]["param_groups"][0]["params"] = [0]
    checkpoint["optimizer_state_dict"]["state"] = {0: checkpoint["optimizer_state_dict"]["state"][0]}
    with pytest.raises(ValueError, match="param_groups"):
        validate_checkpoint(checkpoint, command_schema=COMMAND_SCHEMA)


def test_fresh_seed_checkpoint_allows_only_empty_optimizer_state() -> None:
    checkpoint = _checkpoint()
    checkpoint["completed_updates"] = 0
    checkpoint["iter"] = -1
    checkpoint["trainer_state"]["completed_updates"] = 0
    checkpoint["trainer_state"]["current_learning_iteration"] = 0
    checkpoint["optimizer_state_dict"]["state"] = {}
    assert validate_checkpoint(checkpoint, command_schema=COMMAND_SCHEMA)["completed_updates"] == 0


def test_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.pt"
    target.write_bytes(b"x")
    link = tmp_path / "link.pt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink unavailable")
    with pytest.raises(ValueError, match="symlink"):
        load_checkpoint(
            link,
            expected_sha256=HASH,
            expected_lineage=_lineage(),
            expected_rng_execution_mode="cpu_only",
            expected_cuda_devices=(),
            command_schema=COMMAND_SCHEMA,
        )


def test_load_binds_exact_published_bytes(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows lacks standard-library directory fsync")
    published = save_checkpoint_exclusive(tmp_path / "model_4.pt", _checkpoint(), command_schema=COMMAND_SCHEMA)
    published.path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_checkpoint(
            published.path,
            expected_sha256=published.sha256,
            expected_lineage=_lineage(),
            expected_rng_execution_mode="cpu_only",
            expected_cuda_devices=(),
            command_schema=COMMAND_SCHEMA,
        )


def test_load_requires_exact_expected_lineage(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows lacks standard-library directory fsync")
    published = save_checkpoint_exclusive(tmp_path / "model_4.pt", _checkpoint(), command_schema=COMMAND_SCHEMA)
    expected = _lineage()
    expected["job_id"] = "static_model11500"
    with pytest.raises(ValueError, match="lineage differs"):
        load_checkpoint(
            published.path,
            expected_sha256=published.sha256,
            expected_lineage=expected,
            expected_rng_execution_mode="cpu_only",
            expected_cuda_devices=(),
            command_schema=COMMAND_SCHEMA,
        )
