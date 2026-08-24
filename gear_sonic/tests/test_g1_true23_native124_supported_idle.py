"""CPU-only tests for the native124 supported-idle span command contract."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab.native124_supported_idle import (
    EPISODE_FRAMES,
    SIDECAR_KIND,
    SupportedIdleMotionCommandCfg,
    advance_span_time_steps,
    balanced_span_indices,
    load_supported_idle_catalog,
    make_supported_idle_native_env_cfg,
    span_window_start_bounds,
)
from gear_sonic.utils.g1_true23_supported_idle_corpus import (
    CHANGE_IDLE,
    HANDS_ON_BACK,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_catalog(tmp_path: Path) -> tuple[Path, dict]:
    frame_count = 2047
    arrays = {
        "fps": np.asarray([50.0], dtype=np.float64),
        "joint_pos": np.zeros((frame_count, 23), dtype=np.float32),
        "joint_vel": np.zeros((frame_count, 23), dtype=np.float32),
        "body_pos_w": np.zeros((frame_count, 24, 3), dtype=np.float32),
        "body_quat_w": np.zeros((frame_count, 24, 4), dtype=np.float32),
        "body_lin_vel_w": np.zeros((frame_count, 24, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((frame_count, 24, 3), dtype=np.float32),
    }
    arrays["body_quat_w"][..., 0] = 1.0
    corpus = tmp_path / "supported_idle.npz"
    np.savez(corpus, **arrays)
    sources = [
        {
            "clip_id": CHANGE_IDLE.clip_id,
            "path": "/source/alpha.npz",
            "sha256": CHANGE_IDLE.sha256,
            "source_frame_count": CHANGE_IDLE.frame_count,
        },
        {
            "clip_id": HANDS_ON_BACK.clip_id,
            "path": "/source/beta.npz",
            "sha256": HANDS_ON_BACK.sha256,
            "source_frame_count": HANDS_ON_BACK.frame_count,
        },
    ]
    spans = [
        {
            "id": f"{CHANGE_IDLE.clip_id}::static_frame0",
            "source_clip_id": CHANGE_IDLE.clip_id,
            "kind": "static_frame0",
            "start": 0,
            "stop": 500,
            "stored_length": 500,
            "original_length": 1,
            "source_frame_count": CHANGE_IDLE.frame_count,
            "terminal_repeat_count": 499,
            "transform": "static_frame0_repeat_zero_velocity_v1",
        },
        {
            "id": f"{HANDS_ON_BACK.clip_id}::static_frame0",
            "source_clip_id": HANDS_ON_BACK.clip_id,
            "kind": "static_frame0",
            "start": 500,
            "stop": 1000,
            "stored_length": 500,
            "original_length": 1,
            "source_frame_count": HANDS_ON_BACK.frame_count,
            "terminal_repeat_count": 499,
            "transform": "static_frame0_repeat_zero_velocity_v1",
        },
        {
            "id": f"{CHANGE_IDLE.clip_id}::trajectory",
            "source_clip_id": CHANGE_IDLE.clip_id,
            "kind": "trajectory",
            "start": 1000,
            "stop": 1500,
            "stored_length": 500,
            "original_length": CHANGE_IDLE.frame_count,
            "source_frame_count": CHANGE_IDLE.frame_count,
            "terminal_repeat_count": 138,
            "transform": "trajectory_terminal_hold_zero_velocity_v1",
        },
        {
            "id": f"{HANDS_ON_BACK.clip_id}::trajectory",
            "source_clip_id": HANDS_ON_BACK.clip_id,
            "kind": "trajectory",
            "start": 1500,
            "stop": 2047,
            "stored_length": 547,
            "original_length": HANDS_ON_BACK.frame_count,
            "source_frame_count": HANDS_ON_BACK.frame_count,
            "terminal_repeat_count": 0,
            "transform": "trajectory_terminal_hold_zero_velocity_v1",
        },
    ]
    payload = {
        "schema_version": 1,
        "kind": SIDECAR_KIND,
        "episode_frames": 500,
        "fps": 50.0,
        "diagnostic_only": True,
        "training_authorized": False,
        "corpus": {
            "path": corpus.name,
            "sha256": _sha256(corpus),
            "frame_count": frame_count,
            "arrays": {
                name: {"shape": list(value.shape), "dtype": value.dtype.name} for name, value in arrays.items()
            },
        },
        "sources": sources,
        "spans": spans,
    }
    sidecar = tmp_path / "supported_idle.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar, payload


def test_catalog_validates_hash_arrays_sources_and_exact_four_spans(tmp_path: Path) -> None:
    sidecar, payload = _write_catalog(tmp_path)
    catalog = load_supported_idle_catalog(
        sidecar,
        expected_corpus_sha256=payload["corpus"]["sha256"],
    )
    assert catalog.corpus_sha256 == payload["corpus"]["sha256"]
    assert catalog.frame_count == 2047
    assert tuple(catalog.arrays) == tuple(payload["corpus"]["arrays"])
    assert tuple(source.clip_id for source in catalog.sources) == (
        CHANGE_IDLE.clip_id,
        HANDS_ON_BACK.clip_id,
    )
    assert len(catalog.spans) == 4
    assert catalog.spans_for_phase("static") == (catalog.spans[0], catalog.spans[1])
    assert catalog.spans_for_phase("trajectory") == (catalog.spans[2], catalog.spans[3])


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("extra_top", "keys mismatch"),
        ("schema", "schema_version"),
        ("authorization", "diagnostic_only"),
        ("hash", "caller-pinned"),
        ("array_dtype", "joint_pos mismatch"),
        ("source_identity", "pinned clip"),
        ("unknown_source", "unknown source"),
        ("source_count", "differs from source"),
        ("span_gap", "non-contiguous"),
        ("span_transform", "transform lineage"),
    ),
)
def test_catalog_rejects_schema_and_lineage_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    sidecar, original = _write_catalog(tmp_path)
    expected_corpus_sha256 = original["corpus"]["sha256"]
    payload = deepcopy(original)
    if mutation == "extra_top":
        payload["extra"] = True
    elif mutation == "schema":
        payload["schema_version"] = True
    elif mutation == "authorization":
        payload["training_authorized"] = True
    elif mutation == "hash":
        payload["corpus"]["sha256"] = "0" * 64
    elif mutation == "array_dtype":
        payload["corpus"]["arrays"]["joint_pos"]["dtype"] = "float64"
    elif mutation == "source_identity":
        payload["sources"][0]["clip_id"] = "arbitrary"
        payload["sources"][0]["sha256"] = "c" * 64
        payload["spans"][0]["source_clip_id"] = "arbitrary"
        payload["spans"][0]["id"] = "arbitrary::static_frame0"
        payload["spans"][2]["source_clip_id"] = "arbitrary"
        payload["spans"][2]["id"] = "arbitrary::trajectory"
    elif mutation == "unknown_source":
        payload["spans"][0]["source_clip_id"] = "unknown"
    elif mutation == "source_count":
        payload["spans"][0]["source_frame_count"] = 363
    elif mutation == "span_gap":
        payload["spans"][1]["start"] = 501
    elif mutation == "span_transform":
        payload["spans"][0]["transform"] = "repeat-ish"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_supported_idle_catalog(
            sidecar,
            expected_corpus_sha256=expected_corpus_sha256,
        )


def test_catalog_rejects_non_unit_quaternion_even_with_updated_hash(tmp_path: Path) -> None:
    sidecar, payload = _write_catalog(tmp_path)
    corpus = tmp_path / payload["corpus"]["path"]
    with np.load(corpus, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["body_quat_w"][100, 0, 0] = 2.0
    np.savez(corpus, **arrays)
    payload["corpus"]["sha256"] = _sha256(corpus)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-unit quaternions"):
        load_supported_idle_catalog(
            sidecar,
            expected_corpus_sha256=payload["corpus"]["sha256"],
        )


def test_external_hash_pin_rejects_tamper_with_refreshed_sidecar(tmp_path: Path) -> None:
    sidecar, payload = _write_catalog(tmp_path)
    original_expected_hash = payload["corpus"]["sha256"]
    corpus = tmp_path / payload["corpus"]["path"]
    with np.load(corpus, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["joint_pos"][1010, 0] = 0.125
    np.savez(corpus, **arrays)
    payload["corpus"]["sha256"] = _sha256(corpus)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="caller-pinned"):
        load_supported_idle_catalog(
            sidecar,
            expected_corpus_sha256=original_expected_hash,
        )


def test_balanced_selection_is_deterministic_and_phase_isolated(tmp_path: Path) -> None:
    sidecar, payload = _write_catalog(tmp_path)
    catalog = load_supported_idle_catalog(
        sidecar,
        expected_corpus_sha256=payload["corpus"]["sha256"],
    )
    for phase, expected_kind in (("static", "static_frame0"), ("trajectory", "trajectory")):
        first = balanced_span_indices(catalog, phase, 48)
        assert first == balanced_span_indices(catalog, phase, 48)
        selected = [catalog.spans[index] for index in first]
        assert {span.kind for span in selected} == {expected_kind}
        assert Counter(span.source_clip_id for span in selected) == {
            CHANGE_IDLE.clip_id: 24,
            HANDS_ON_BACK.clip_id: 24,
        }
        shifted = balanced_span_indices(catalog, phase, 2, selection_offset=1)
        assert [catalog.spans[index].source_clip_id for index in shifted] == [
            HANDS_ON_BACK.clip_id,
            CHANGE_IDLE.clip_id,
        ]


def test_uniform_window_bounds_cover_offsets_zero_through_47(tmp_path: Path) -> None:
    sidecar, payload = _write_catalog(tmp_path)
    catalog = load_supported_idle_catalog(
        sidecar,
        expected_corpus_sha256=payload["corpus"]["sha256"],
    )
    span = catalog.spans[3]
    lower, upper = span_window_start_bounds(span)
    assert (lower, upper) == (1500, 1547)
    assert [start - span.start for start in range(lower, upper + 1)] == list(range(48))
    static_lower, static_upper = span_window_start_bounds(catalog.spans[0])
    assert static_lower == static_upper == 0


def test_frame_zero_499_and_reset_boundaries_never_cross_or_wrap() -> None:
    current = torch.tensor([1000], dtype=torch.long)
    stop = torch.tensor([1500], dtype=torch.long)
    current = advance_span_time_steps(current, stop, torch.tensor([0], dtype=torch.long))
    assert current.item() == 1000
    for episode_step in range(1, EPISODE_FRAMES):
        current = advance_span_time_steps(
            current,
            stop,
            torch.tensor([episode_step], dtype=torch.long),
        )
    assert current.item() == 1499
    with pytest.raises(RuntimeError, match="cross"):
        advance_span_time_steps(current, stop, torch.tensor([500], dtype=torch.long))
    resampled = torch.tensor([1500], dtype=torch.long)
    held = advance_span_time_steps(
        resampled,
        torch.tensor([2000], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
    )
    assert held.item() == 1500


def test_config_builder_replaces_only_stock_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mjlab")
    from mjlab.tasks.tracking.mdp import MotionCommandCfg
    from src.tasks.tracking.config.g1_23dof import env_cfgs

    sidecar, payload = _write_catalog(tmp_path)
    stock_factory = env_cfgs.unitree_g1_23dof_flat_tracking_env_cfg
    untouched_stock = stock_factory(has_state_estimation=False)
    supplied_stock = stock_factory(has_state_estimation=False)
    original_command = supplied_stock.commands["motion"]
    original_command_values = {
        field.name: getattr(original_command, field.name) for field in fields(original_command)
    }
    non_command_objects = {
        field.name: getattr(supplied_stock, field.name)
        for field in fields(supplied_stock)
        if field.name != "commands"
    }
    monkeypatch.setattr(
        env_cfgs,
        "unitree_g1_23dof_flat_tracking_env_cfg",
        lambda *, has_state_estimation: supplied_stock,
    )
    cfg = make_supported_idle_native_env_cfg(
        sidecar_path=sidecar,
        expected_corpus_sha256=payload["corpus"]["sha256"],
        phase="trajectory",
        start_mode="uniform_window",
    )

    assert cfg is supplied_stock
    command = cfg.commands["motion"]
    assert isinstance(command, SupportedIdleMotionCommandCfg)
    assert command.phase == "trajectory"
    assert command.start_mode == "uniform_window"
    assert command.episode_frames == 500
    assert command.sampling_mode == "uniform"
    assert Path(command.motion_file).resolve() == (tmp_path / "supported_idle.npz").resolve()
    assert cfg.episode_length_s == 10.0
    assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)
    assert tuple(cfg.observations["actor"].terms) == (
        "command",
        "motion_anchor_ori_b",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    )
    assert tuple(cfg.actions) == ("joint_pos",)
    assert all(getattr(cfg, name) is value for name, value in non_command_objects.items())
    assert all(getattr(original_command, name) == value for name, value in original_command_values.items())
    assert isinstance(untouched_stock.commands["motion"], MotionCommandCfg)
    assert not isinstance(untouched_stock.commands["motion"], SupportedIdleMotionCommandCfg)
    assert untouched_stock.commands["motion"].motion_file == ""
