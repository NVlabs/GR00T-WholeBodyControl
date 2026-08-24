"""Tests for the exact four-span Step1A supported-idle corpus."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from gear_sonic.scripts.build_g1_true23_step1a_idle_native_corpus import main
from gear_sonic.utils.g1_true23_supported_idle_corpus import (
    CHANGE_IDLE,
    EPISODE_FRAMES,
    EXPECTED_SPANS,
    HANDS_ON_BACK,
    KIND,
    MOTION_ARRAY_NAMES,
    NPZ_ARRAY_NAMES,
    PINNED_CLIPS,
    POSE_ARRAY_NAMES,
    SCHEMA_VERSION,
    STATIC_KIND,
    STATIC_TRANSFORM,
    TOTAL_FRAMES,
    TRAJECTORY_KIND,
    TRAJECTORY_TRANSFORM,
    VELOCITY_ARRAY_NAMES,
    build_supported_idle_corpus,
    sha256_file,
    validate_supported_idle_corpus,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "artifacts" / "g1_true23_step1b_idle_nominal_20260806_v1" / "inputs" / "schema6"
CHANGE_MOTION = SOURCE_ROOT / CHANGE_IDLE.clip_id / "motion.npz"
HANDS_MOTION = SOURCE_ROOT / HANDS_ON_BACK.clip_id / "motion.npz"
SOURCE_PATHS = {
    CHANGE_IDLE.clip_id: CHANGE_MOTION,
    HANDS_ON_BACK.clip_id: HANDS_MOTION,
}
pytestmark = pytest.mark.skipif(
    not CHANGE_MOTION.is_file() or not HANDS_MOTION.is_file(),
    reason="local immutable Step1B schema-v6 motion fixtures are unavailable",
)


@pytest.fixture(scope="module")
def built_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    directory = tmp_path_factory.mktemp("supported_idle_corpus")
    corpus = directory / "supported_idle.npz"
    sidecar = directory / "supported_idle.spans.json"
    report = build_supported_idle_corpus(
        change_motion=CHANGE_MOTION,
        hands_motion=HANDS_MOTION,
        corpus_path=corpus,
        sidecar_path=sidecar,
    )
    return corpus, sidecar, report


def _read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in NPZ_ARRAY_NAMES}


def _read_source(path: Path) -> dict[str, np.ndarray]:
    return _read_arrays(path)


def _span_slice(arrays: dict[str, np.ndarray], span: dict, name: str) -> np.ndarray:
    return arrays[name][span["start"] : span["stop"]]


def _copy_bundle(tmp_path: Path, built_bundle: tuple[Path, Path, dict]) -> tuple[Path, Path, dict]:
    source_corpus, source_sidecar, _ = built_bundle
    corpus = tmp_path / "copy.npz"
    sidecar = tmp_path / "copy.spans.json"
    shutil.copy2(source_corpus, corpus)
    payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
    payload["corpus"]["path"] = corpus.name
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return corpus, sidecar, payload


def _rewrite_corpus(corpus: Path, sidecar: Path, arrays: dict[str, np.ndarray]) -> None:
    with corpus.open("wb") as stream:
        np.savez(stream, **arrays)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["corpus"]["sha256"] = sha256_file(corpus)
    payload["corpus"]["arrays"] = {
        name: {"shape": list(arrays[name].shape), "dtype": arrays[name].dtype.name} for name in NPZ_ARRAY_NAMES
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_builds_exact_four_span_stock_motion_npz(built_bundle) -> None:
    corpus, sidecar, report = built_bundle
    assert (
        validate_supported_idle_corpus(
            corpus,
            sidecar,
            source_paths=SOURCE_PATHS,
        )
        == report
    )
    assert set(report) == {
        "schema_version",
        "kind",
        "episode_frames",
        "fps",
        "diagnostic_only",
        "training_authorized",
        "corpus",
        "sources",
        "spans",
    }
    assert report["schema_version"] == SCHEMA_VERSION == 1
    assert report["kind"] == KIND == "g1_true23_supported_idle_native_corpus_v1"
    assert report["episode_frames"] == EPISODE_FRAMES == 500
    assert report["diagnostic_only"] is True
    assert report["training_authorized"] is False
    assert report["corpus"]["path"] == corpus.name
    assert report["corpus"]["sha256"] == sha256_file(corpus)
    assert report["corpus"]["frame_count"] == TOTAL_FRAMES == 2047
    assert report["spans"] == list(EXPECTED_SPANS)
    assert [source["clip_id"] for source in report["sources"]] == [clip.clip_id for clip in PINNED_CLIPS]
    assert [source["sha256"] for source in report["sources"]] == [clip.sha256 for clip in PINNED_CLIPS]

    assert report["spans"] == [
        {
            "id": f"{CHANGE_IDLE.clip_id}::{STATIC_KIND}",
            "source_clip_id": CHANGE_IDLE.clip_id,
            "kind": STATIC_KIND,
            "start": 0,
            "stop": 500,
            "stored_length": 500,
            "original_length": 1,
            "source_frame_count": 362,
            "terminal_repeat_count": 499,
            "transform": STATIC_TRANSFORM,
        },
        {
            "id": f"{HANDS_ON_BACK.clip_id}::{STATIC_KIND}",
            "source_clip_id": HANDS_ON_BACK.clip_id,
            "kind": STATIC_KIND,
            "start": 500,
            "stop": 1000,
            "stored_length": 500,
            "original_length": 1,
            "source_frame_count": 547,
            "terminal_repeat_count": 499,
            "transform": STATIC_TRANSFORM,
        },
        {
            "id": f"{CHANGE_IDLE.clip_id}::{TRAJECTORY_KIND}",
            "source_clip_id": CHANGE_IDLE.clip_id,
            "kind": TRAJECTORY_KIND,
            "start": 1000,
            "stop": 1500,
            "stored_length": 500,
            "original_length": 362,
            "source_frame_count": 362,
            "terminal_repeat_count": 138,
            "transform": TRAJECTORY_TRANSFORM,
        },
        {
            "id": f"{HANDS_ON_BACK.clip_id}::{TRAJECTORY_KIND}",
            "source_clip_id": HANDS_ON_BACK.clip_id,
            "kind": TRAJECTORY_KIND,
            "start": 1500,
            "stop": 2047,
            "stored_length": 547,
            "original_length": 547,
            "source_frame_count": 547,
            "terminal_repeat_count": 0,
            "transform": TRAJECTORY_TRANSFORM,
        },
    ]


def test_materialized_values_are_exact(built_bundle) -> None:
    corpus, _, report = built_bundle
    arrays = _read_arrays(corpus)
    sources = {
        CHANGE_IDLE.clip_id: _read_source(CHANGE_MOTION),
        HANDS_ON_BACK.clip_id: _read_source(HANDS_MOTION),
    }
    spans = {span["id"]: span for span in report["spans"]}

    for clip in PINNED_CLIPS:
        source = sources[clip.clip_id]
        static = spans[f"{clip.clip_id}::{STATIC_KIND}"]
        trajectory = spans[f"{clip.clip_id}::{TRAJECTORY_KIND}"]
        for name in POSE_ARRAY_NAMES:
            actual = _span_slice(arrays, static, name)
            expected = np.repeat(source[name][0:1], EPISODE_FRAMES, axis=0)
            assert np.array_equal(actual, expected)
        for name in VELOCITY_ARRAY_NAMES:
            assert np.count_nonzero(_span_slice(arrays, static, name)) == 0
        for name in MOTION_ARRAY_NAMES:
            original = _span_slice(arrays, trajectory, name)[: clip.frame_count]
            assert np.array_equal(original, source[name])

    change_trajectory = spans[f"{CHANGE_IDLE.clip_id}::{TRAJECTORY_KIND}"]
    for name in POSE_ARRAY_NAMES:
        actual = _span_slice(arrays, change_trajectory, name)
        expected = np.repeat(actual[361:362], 138, axis=0)
        assert np.array_equal(actual[362:], expected)
    for name in VELOCITY_ARRAY_NAMES:
        assert np.count_nonzero(_span_slice(arrays, change_trajectory, name)[362:]) == 0

    assert arrays["fps"].shape == (1,)
    assert arrays["fps"].dtype == np.float64
    for name in MOTION_ARRAY_NAMES:
        assert arrays[name].shape[0] == TOTAL_FRAMES
        assert arrays[name].dtype == np.float32
        assert np.isfinite(arrays[name]).all()
    assert np.allclose(
        np.linalg.norm(arrays["body_quat_w"].astype(np.float64), axis=-1),
        1.0,
        atol=1.0e-5,
        rtol=0.0,
    )


def test_stock_motion_loader_accepts_materialized_corpus(built_bundle) -> None:
    commands = pytest.importorskip("mjlab.tasks.tracking.mdp.commands")
    torch = pytest.importorskip("torch")
    corpus, _, _ = built_bundle

    loader = commands.MotionLoader(
        str(corpus),
        torch.arange(24, dtype=torch.long),
        device="cpu",
    )

    assert loader.time_step_total == TOTAL_FRAMES
    assert tuple(loader.joint_pos.shape) == (TOTAL_FRAMES, 23)
    assert tuple(loader.body_pos_w.shape) == (TOTAL_FRAMES, 24, 3)


def test_builder_refuses_any_overwrite(built_bundle) -> None:
    corpus, sidecar, _ = built_bundle
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_supported_idle_corpus(
            change_motion=CHANGE_MOTION,
            hands_motion=HANDS_MOTION,
            corpus_path=corpus,
            sidecar_path=sidecar,
        )


def test_builder_rejects_non_pinned_source_bytes(tmp_path) -> None:
    changed = tmp_path / "changed.npz"
    shutil.copy2(CHANGE_MOTION, changed)
    with changed.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_supported_idle_corpus(
            change_motion=changed,
            hands_motion=HANDS_MOTION,
            corpus_path=tmp_path / "corpus.npz",
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (lambda payload: payload.__setitem__("schema_version", 2), "schema_version"),
        (lambda payload: payload.__setitem__("kind", "wrong"), "schema or kind"),
        (lambda payload: payload.__setitem__("training_authorized", True), "unauthorized"),
        (lambda payload: payload["sources"].pop(), "exactly two"),
        (
            lambda payload: payload["sources"][0].__setitem__("sha256", "0" * 64),
            "source identity",
        ),
        (lambda payload: payload["spans"][0].__setitem__("stop", 501), "spans differ"),
        (
            lambda payload: payload["corpus"]["arrays"]["joint_pos"].__setitem__("dtype", "float64"),
            "shapes or dtypes",
        ),
    ),
)
def test_validator_rejects_sidecar_contract_tampering(tmp_path, built_bundle, mutation, error) -> None:
    corpus, sidecar, payload = _copy_bundle(tmp_path, built_bundle)
    mutation(payload)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        validate_supported_idle_corpus(corpus, sidecar)


def test_validator_rejects_corpus_hash_tampering(tmp_path, built_bundle) -> None:
    corpus, sidecar, _ = _copy_bundle(tmp_path, built_bundle)
    with corpus.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="corpus SHA-256 mismatch"):
        validate_supported_idle_corpus(corpus, sidecar)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (
            lambda arrays: arrays["joint_vel"].__setitem__((0, 0), np.float32(1.0)),
            "must be exactly zero",
        ),
        (
            lambda arrays: arrays["joint_pos"].__setitem__((0, 0), np.float32(np.nan)),
            "NaN or Inf",
        ),
        (
            lambda arrays: arrays["body_quat_w"].__setitem__((0, 0), [2.0, 0.0, 0.0, 0.0]),
            "non-unit quaternions",
        ),
        (
            lambda arrays: arrays.__setitem__("joint_pos", arrays["joint_pos"][:-1]),
            "shape mismatch",
        ),
    ),
)
def test_validator_rejects_array_and_semantic_tampering(tmp_path, built_bundle, mutation, error) -> None:
    corpus, sidecar, _ = _copy_bundle(tmp_path, built_bundle)
    arrays = _read_arrays(corpus)
    mutation(arrays)
    _rewrite_corpus(corpus, sidecar, arrays)
    with pytest.raises(ValueError, match=error):
        validate_supported_idle_corpus(corpus, sidecar)


def test_cli_builds_default_pinned_sources(tmp_path, capsys) -> None:
    corpus = tmp_path / "cli.npz"
    assert main(["--output", str(corpus)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["kind"] == KIND
    assert report["corpus"]["path"] == corpus.name
    assert validate_supported_idle_corpus(corpus) == report
