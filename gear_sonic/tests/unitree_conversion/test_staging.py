from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import re
import shutil
import threading

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import (
    G1SupplementalInfo,
)
from gear_sonic.data.unitree_conversion import staging as staging_module
from gear_sonic.data.unitree_conversion.staging import (
    MergeVerification,
    StageIdentity,
    StagePayload,
    can_resume,
    dataset_manifest_digest,
    load_stage_rows,
    merge_stages,
    stage_path,
    write_stage,
)
from gear_sonic.data.unitree_conversion.validation import (
    TARGET_FRAME_SCHEMA,
    regenerate_eef_state,
    target_joint_limits,
)


def _write_video(
    path: Path,
    *,
    frame_count: int = 2,
    fps: int = 50,
    size: tuple[int, int] = (640, 480),
    value_offset: int = 0,
    codec_name: str | None = None,
    pixel_format: str = "yuv420p",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in (codec_name,) if codec_name is not None else ("libx264", "libopenh264", "h264"):
        try:
            av.codec.Codec(codec, "w")
        except (av.error.FFmpegError, ValueError):
            continue
        break
    else:
        pytest.skip("PyAV has no usable H.264 encoder")
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = pixel_format
        for index in range(frame_count):
            rgb = np.full((size[1], size[0], 3), index * 50 + value_offset, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _row(index: int, task: str) -> dict[str, object]:
    row: dict[str, object] = {}
    for key, spec in TARGET_FRAME_SCHEMA.items():
        if key == "timestamp":
            row[key] = np.float32(index / 50.0)
        elif key == "task":
            row[key] = task
        else:
            value = np.zeros(spec.shape, dtype=spec.dtype)
            if key == "teleop.smpl_frame_index":
                value[0] = index
            row[key] = value
    row["observation.root_orientation"][:] = [1.0, 0.0, 0.0, 0.0]
    row["observation.cpp_rotation_offset"][:] = [1.0, 0.0, 0.0, 0.0]
    row["observation.init_base_quat"][:] = [1.0, 0.0, 0.0, 0.0]
    row["observation.projected_gravity"][:] = [0.0, 0.0, -1.0]
    row["observation.eef_state"] = regenerate_eef_state(row["observation.state"])
    row["teleop.body_quat_w"][:] = [1.0, 0.0, 0.0, 0.0]
    row["teleop.target_body_orientation"][:] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    row["teleop.planner_facing"][:] = [1.0, 0.0, 0.0]
    row["teleop.planner_speed"][:] = [-1.0]
    row["teleop.planner_height"][:] = [-1.0]
    row["observation.state"][[16, 30]] = [0.2, -0.2]
    row["action.wbc"][[16, 30]] = [0.2, -0.2]
    row["observation.eef_state"] = regenerate_eef_state(row["observation.state"])
    return row


def _identity(episode_id: int = 7, *, repo: str = "unitree/source") -> StageIdentity:
    return StageIdentity(
        source_repo_id=repo,
        source_revision="a" * 40,
        source_episode_id=episode_id,
        source_file_sha256={f"data/episode-{episode_id}.parquet": "b" * 64},
        conversion_config_sha256="c" * 64,
        converter_version="unitree-sonic-v1",
        encoder_sha256="d" * 64,
        encoder_config_sha256="e" * 64,
        source_lock_sha256="f" * 64,
    )


def _payload(tmp_path: Path, episode_id: int = 7, tasks: tuple[str, str] = ("pour", "place")) -> StagePayload:
    video = tmp_path / f"source-{episode_id}.mp4"
    _write_video(video)
    return StagePayload(
        rows=tuple(_row(index, task) for index, task in enumerate(tasks)),
        videos={"observation.images.ego_view": video},
    )


def test_stage_identity_is_deeply_immutable_and_path_is_safe(tmp_path: Path) -> None:
    identity = _identity()

    with pytest.raises(FrozenInstanceError):
        identity.source_episode_id = 8  # type: ignore[misc]
    with pytest.raises(TypeError):
        identity.source_file_sha256["new"] = "0" * 64  # type: ignore[index]

    assert stage_path(tmp_path, identity) == (
        tmp_path / ".staging" / "unitree%2Fsource" / ("a" * 40) / "episode-7" / ("c" * 64)
    )
    with pytest.raises(ValueError, match="source_repo_id"):
        replace(identity, source_repo_id="../escape")
    with pytest.raises(ValueError, match="source file"):
        replace(identity, source_file_sha256={"../escape": "b" * 64})


def test_stage_root_rejects_dotdot_alias_before_writing(tmp_path: Path) -> None:
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    aliased_root = decoy / ".." / "real-root"

    with pytest.raises(ValueError, match=r"stage output.*canonical|\.\."):
        stage_path(aliased_root, _identity())
    with pytest.raises(ValueError, match=r"stage output.*canonical|\.\."):
        write_stage(aliased_root, _identity(), _payload(tmp_path))

    assert not (tmp_path / "real-root").exists()


def test_write_stage_preserves_exact_arrays_and_resumes_matching_identity(tmp_path: Path) -> None:
    identity = _identity()
    result = write_stage(tmp_path, identity, _payload(tmp_path))

    assert result.path == stage_path(tmp_path, identity)
    assert result.reused is False
    assert can_resume(result.path, identity) is True
    assert can_resume(result.path, replace(identity, encoder_sha256="0" * 64)) is False
    resumed = write_stage(tmp_path, identity, _payload(tmp_path, episode_id=8))
    assert resumed.path == result.path
    assert resumed.reused is True

    rows = load_stage_rows(result.path)
    assert tuple(rows[0]) == tuple(TARGET_FRAME_SCHEMA)
    for key, spec in TARGET_FRAME_SCHEMA.items():
        value = rows[1][key]
        if key == "task":
            assert value == "place"
        elif key == "timestamp":
            assert isinstance(value, np.float32)
            assert value.tobytes() == np.float32(1 / 50).tobytes()
        else:
            assert isinstance(value, np.ndarray)
            assert value.dtype == spec.dtype
            assert value.shape == spec.shape


def test_path_video_stage_never_uses_whole_file_secure_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    payload = _payload(tmp_path)
    real_secure_read = staging_module._secure_read

    def reject_mp4(path: Path) -> bytes:
        if path.suffix == ".mp4":
            raise AssertionError("whole MP4 read")
        return real_secure_read(path)

    monkeypatch.setattr(staging_module, "_secure_read", reject_mp4)
    result = write_stage(tmp_path, identity, payload)
    assert can_resume(result.path, identity)
    episode = staging_module._stage_input(result)
    artifact = next(iter(episode.videos.values()))
    assert (
        staging_module._inspect_bound_video(
            artifact.path,
            expected_sha256=artifact.sha256,
            field_name="staged video",
        )[0]
        == 2
    )


@pytest.mark.parametrize(
    "artifact",
    (
        "manifest.json",
        "validation.json",
        "frame-data.parquet",
        "videos/observation.images.ego_view.mp4",
        "checksums.sha256",
    ),
)
def test_resume_rejects_corruption_in_every_stage_artifact(tmp_path: Path, artifact: str) -> None:
    identity = _identity()
    result = write_stage(tmp_path, identity, _payload(tmp_path))
    target = result.path / artifact
    target.write_bytes(target.read_bytes() + b"corruption")

    assert can_resume(result.path, identity) is False
    with pytest.raises(FileExistsError, match="immutable stage"):
        write_stage(tmp_path, identity, _payload(tmp_path, episode_id=8))


def test_resume_rejects_unexpected_artifact_and_symlink(tmp_path: Path) -> None:
    identity = _identity()
    result = write_stage(tmp_path, identity, _payload(tmp_path))
    (result.path / "unexpected").write_text("no")
    assert can_resume(result.path, identity) is False

    result2 = write_stage(tmp_path, replace(identity, source_episode_id=8), _payload(tmp_path, episode_id=8))
    (result2.path / "videos" / "observation.images.ego_view.mp4").unlink()
    (result2.path / "videos" / "observation.images.ego_view.mp4").symlink_to(tmp_path / "source-8.mp4")
    assert can_resume(result2.path, replace(identity, source_episode_id=8)) is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda rows: rows[:1], "video frame count"),
        (lambda rows: [{**rows[0], "task": ""}, rows[1]], "task"),
        (
            lambda rows: [rows[0], {**rows[1], "timestamp": np.float32(9.0)}],
            "timestamp",
        ),
        (
            lambda rows: [
                rows[0],
                {**rows[1], "action.wbc": np.zeros((42,), dtype=np.float64)},
            ],
            "action.wbc",
        ),
        (
            lambda rows: [
                rows[0],
                {**rows[1], "action.motion_token": np.full((64,), np.nan, dtype=np.float64)},
            ],
            "finite",
        ),
    ),
)
def test_write_stage_rejects_invalid_rows_or_video_counts_and_cleans_temp(
    tmp_path: Path, mutator, message: str
) -> None:
    payload = _payload(tmp_path)

    with pytest.raises(ValueError, match=message):
        invalid = StagePayload(rows=tuple(mutator(list(payload.rows))), videos=payload.videos)
        write_stage(tmp_path, _identity(), invalid)

    final = stage_path(tmp_path, _identity())
    assert not final.exists()
    assert not list(final.parent.glob(f".{final.name}.*"))


def test_stage_rejects_object_values_pickle_files_and_unsafe_video_targets(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    bad_row = dict(payload.rows[0])
    bad_row["action.wbc"] = np.array([object()] * 43, dtype=object)
    with pytest.raises(ValueError, match="action.wbc"):
        StagePayload(rows=(bad_row, payload.rows[1]), videos=payload.videos)
    with pytest.raises(ValueError, match="video target"):
        StagePayload(rows=payload.rows, videos={"../evil.pkl": b"pickle"})
    with pytest.raises(ValueError, match="unsupported"):
        StagePayload(rows=payload.rows, videos={"observation.images.unknown": b"video"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("teleop.planner_speed", np.array([0.0], dtype=np.float32), "neutral"),
        (
            "observation.root_orientation",
            np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "unit WXYZ",
        ),
        (
            "observation.projected_gravity",
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "inverse root rotation",
        ),
    ),
)
def test_stage_payload_rejects_corrupt_neutral_or_root_semantics(
    tmp_path: Path, field: str, value: np.ndarray, message: str
) -> None:
    rows = [_row(0, "task"), _row(1, "task")]
    rows[1][field] = value
    with pytest.raises(ValueError, match=message):
        StagePayload(
            rows=tuple(rows),
            videos={"observation.images.ego_view": _payload(tmp_path).videos["observation.images.ego_view"]},
        )


def test_asset_free_target_kinematics_exposes_exact_43_joint_limits_and_fk() -> None:
    lower, upper = target_joint_limits()
    assert lower.shape == upper.shape == (43,)
    assert lower.dtype == upper.dtype == np.dtype(np.float64)
    assert np.all(lower <= upper)
    eef = regenerate_eef_state(np.zeros(43, dtype=np.float64))
    assert eef.shape == (14,)
    assert np.isfinite(eef).all()
    assert np.linalg.norm(eef[3:7]) == pytest.approx(1.0)
    assert np.linalg.norm(eef[10:14]) == pytest.approx(1.0)


def test_target_limits_use_robot_model_body_and_asymmetric_raw_dex3_domains() -> None:
    supplemental = G1SupplementalInfo()
    expected_lower = np.array(
        [supplemental.joint_limits[name][0] for name in staging_module.TARGET_JOINT_NAMES],
        dtype=np.float64,
    )
    expected_upper = np.array(
        [supplemental.joint_limits[name][1] for name in staging_module.TARGET_JOINT_NAMES],
        dtype=np.float64,
    )
    expected_lower[staging_module.TARGET_JOINT_NAMES.index("left_shoulder_roll_joint")] = -1.5882
    expected_upper[staging_module.TARGET_JOINT_NAMES.index("right_shoulder_roll_joint")] = 1.5882
    expected_hand_limits = {
        "left_hand_index_0_joint": (-1.832595705986023, 0.19198620319366455),
        "left_hand_index_1_joint": (-2.094395160675049, 0.0),
        "left_hand_middle_0_joint": (-1.832595705986023, 0.19198620319366455),
        "left_hand_middle_1_joint": (-2.094395160675049, 0.0),
        "left_hand_thumb_0_joint": (-1.0471975803375244, 1.0471975803375244),
        "left_hand_thumb_1_joint": (-1.0471975803375244, 1.0471975803375244),
        "left_hand_thumb_2_joint": (0.0, 1.7453292608261108),
        "right_hand_index_0_joint": (-0.19198620319366455, 1.832595705986023),
        "right_hand_index_1_joint": (0.0, 2.094395160675049),
        "right_hand_middle_0_joint": (-0.19198620319366455, 1.832595705986023),
        "right_hand_middle_1_joint": (0.0, 2.094395160675049),
        "right_hand_thumb_0_joint": (-1.0471975803375244, 1.0471975803375244),
        "right_hand_thumb_1_joint": (-1.0471975803375244, 1.0471975803375244),
        "right_hand_thumb_2_joint": (-1.7453292608261108, 0.0),
    }
    for name, (lower_bound, upper_bound) in expected_hand_limits.items():
        index = staging_module.TARGET_JOINT_NAMES.index(name)
        expected_lower[index] = lower_bound
        expected_upper[index] = upper_bound

    lower, upper = target_joint_limits()

    assert np.array_equal(lower, expected_lower)
    assert np.array_equal(upper, expected_upper)
    assert lower[16] == np.float64(-1.5882)
    assert upper[30] == np.float64(1.5882)


def test_stage_accepts_asymmetric_raw_dex3_values_outside_symmetric_model_limits(
    tmp_path: Path,
) -> None:
    rows = [_row(0, "task"), _row(1, "task")]
    right_index_1 = staging_module.TARGET_JOINT_NAMES.index("right_hand_index_1_joint")
    right_thumb_2 = staging_module.TARGET_JOINT_NAMES.index("right_hand_thumb_2_joint")
    left_index_0 = staging_module.TARGET_JOINT_NAMES.index("left_hand_index_0_joint")
    for row in rows:
        row["observation.state"] = row["observation.state"].copy()
        row["action.wbc"] = row["action.wbc"].copy()
        row["observation.state"][right_index_1] = 2.085
        row["observation.state"][right_thumb_2] = -1.56
        row["observation.state"][left_index_0] = 0.19
        row["action.wbc"][right_index_1] = np.float64(2.094395160675049)
        row["action.wbc"][right_thumb_2] = -1.56
        row["action.wbc"][left_index_0] = np.float64(0.19198620319366455)
        row["observation.eef_state"] = regenerate_eef_state(row["observation.state"])

    payload = StagePayload(rows=tuple(rows), videos=_payload(tmp_path).videos)

    assert len(payload.rows) == 2


def test_stage_allows_small_dex3_measurement_overshoot_but_not_commands(tmp_path: Path) -> None:
    rows = [_row(0, "task"), _row(1, "task")]
    left_thumb_1 = staging_module.TARGET_JOINT_NAMES.index("left_hand_thumb_1_joint")
    rows[1]["observation.state"] = rows[1]["observation.state"].copy()
    rows[1]["observation.state"][left_thumb_1] = 1.0506
    rows[1]["observation.eef_state"] = regenerate_eef_state(rows[1]["observation.state"])
    StagePayload(rows=tuple(rows), videos=_payload(tmp_path).videos)

    rows[1]["action.wbc"] = rows[1]["action.wbc"].copy()
    rows[1]["action.wbc"][left_thumb_1] = 1.0506
    with pytest.raises(ValueError, match="action.wbc.*Dex3 DDS command limits"):
        StagePayload(rows=tuple(rows), videos=_payload(tmp_path).videos)


@pytest.mark.parametrize("field", ("observation.state", "action.wbc"))
def test_stage_accepts_raw_urdf_shoulder_roll_values_used_by_deployed_feedback(tmp_path: Path, field: str) -> None:
    rows = [_row(0, "task"), _row(1, "task")]
    rows[1][field] = rows[1][field].copy()
    rows[1][field][16] = 0.0
    rows[1][field][30] = 0.0
    if field == "observation.state":
        rows[1]["observation.eef_state"] = regenerate_eef_state(rows[1][field])

    payload = StagePayload(rows=tuple(rows), videos=_payload(tmp_path).videos)

    assert len(payload.rows) == 2


@pytest.mark.parametrize("field", ("observation.state", "action.wbc"))
def test_stage_payload_rejects_finite_joint_target_outside_urdf_limits(tmp_path: Path, field: str) -> None:
    rows = [_row(0, "task"), _row(1, "task")]
    rows[1][field] = rows[1][field].copy()
    rows[1][field][0] = 1e100
    with pytest.raises(ValueError, match=f"{re.escape(field)}.*joint limits"):
        StagePayload(rows=tuple(rows), videos=_payload(tmp_path).videos)


def test_stage_payload_rejects_finite_but_fk_incorrect_eef_translation(tmp_path: Path) -> None:
    rows = [_row(0, "task"), _row(1, "task")]
    rows[1]["observation.eef_state"] = rows[1]["observation.eef_state"].copy()
    rows[1]["observation.eef_state"][0] += 0.01
    with pytest.raises(ValueError, match="observation.eef_state.*forward kinematics"):
        StagePayload(rows=tuple(rows), videos=_payload(tmp_path).videos)


@pytest.mark.parametrize(
    ("fps", "size", "message"),
    ((30, (640, 480), "fps 50"), (50, (320, 240), "frame size")),
)
def test_stage_rejects_wrong_target_video_fps_or_dimensions(
    tmp_path: Path, fps: int, size: tuple[int, int], message: str
) -> None:
    video = tmp_path / "wrong.mp4"
    _write_video(video, fps=fps, size=size)
    payload = StagePayload(
        rows=(_row(0, "task"), _row(1, "task")),
        videos={"observation.images.ego_view": video},
    )
    with pytest.raises(ValueError, match=message):
        write_stage(tmp_path, _identity(), payload)


@pytest.mark.parametrize(
    ("codec_name", "pixel_format", "message"),
    (("mpeg4", "yuv420p", "H264"), ("libx264", "yuv444p", "yuv420p")),
)
def test_stage_rejects_incompatible_video_codec_or_pixel_format(
    tmp_path: Path,
    codec_name: str,
    pixel_format: str,
    message: str,
) -> None:
    video = tmp_path / f"{codec_name}-{pixel_format}.mp4"
    _write_video(video, codec_name=codec_name, pixel_format=pixel_format)
    payload = StagePayload(
        rows=(_row(0, "task"), _row(1, "task")),
        videos={"observation.images.ego_view": video},
    )

    with pytest.raises(ValueError, match=message):
        write_stage(tmp_path, _identity(), payload)


def test_stage_write_failure_cleans_unique_sibling_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _identity()
    payload = _payload(tmp_path)
    real_write = staging_module._write_file
    calls = 0

    def fail_second(path: Path, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated write failure")
        real_write(path, data)

    monkeypatch.setattr(staging_module, "_write_file", fail_second)
    with pytest.raises(OSError, match="simulated"):
        write_stage(tmp_path, identity, payload)
    final = stage_path(tmp_path, identity)
    assert not final.exists()
    assert not list(final.parent.glob(f".{final.name}.*"))


def test_atomic_publication_fails_closed_when_rename_noreplace_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()

    def unavailable(source: Path, destination: Path) -> None:
        raise OSError("RENAME_NOREPLACE unavailable")

    monkeypatch.setattr(staging_module, "_rename_noreplace", unavailable)
    with pytest.raises(OSError, match="unavailable"):
        write_stage(tmp_path, identity, _payload(tmp_path))
    final = stage_path(tmp_path, identity)
    assert not final.exists()
    assert not list(final.parent.glob(f".{final.name}.*"))


class RecordingMergeBackend:
    def __init__(self) -> None:
        self.orders: list[tuple[int, ...]] = []
        self.task_catalogs: list[tuple[str, ...]] = []

    def __call__(self, output: Path, episodes, task_catalog: tuple[str, ...]) -> None:
        episode_ids = tuple(episode.identity.source_episode_id for episode in episodes)
        self.orders.append(episode_ids)
        self.task_catalogs.append(task_catalog)
        for episode in episodes:
            assert not hasattr(episode, "rows")
            assert len(tuple(staging_module.iter_stage_rows(episode, batch_size=1))) == episode.row_count
        staging_module._default_merge_backend(output, episodes, task_catalog)

    def validate(self, output, episodes, task_catalog) -> MergeVerification:
        return _exact_verification(episodes, task_catalog)


def _exact_verification(episodes, task_catalog: tuple[str, ...]) -> MergeVerification:
    return staging_module._expected_merge_verification(episodes, task_catalog)


def _two_stages(tmp_path: Path):
    first = write_stage(
        tmp_path,
        _identity(10),
        _payload(tmp_path, 10, tasks=("zeta", "alpha")),
    )
    second = write_stage(
        tmp_path,
        _identity(2),
        _payload(tmp_path, 2, tasks=("beta", "alpha")),
    )
    return first, second


def _unpublished_real_output(tmp_path: Path, *, episode_id: int = 3):
    stage = write_stage(tmp_path, _identity(episode_id), _payload(tmp_path, episode_id))
    episodes = staging_module._validate_merge_inputs((stage,))
    tasks = tuple(sorted({row["task"] for row in staging_module.iter_stage_rows(episodes[0])}))
    output = tmp_path / f"candidate-{episode_id}"
    merge_stages((stage,), output)
    return stage, episodes, tasks, output


def test_merge_is_deterministic_and_preregisters_lexicographic_tasks(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)
    backend = RecordingMergeBackend()
    first = merge_stages(
        reversed(stages), tmp_path / "first", merge_backend=backend, final_validator=backend.validate
    )
    second = merge_stages(stages, tmp_path / "second", merge_backend=backend, final_validator=backend.validate)

    assert backend.orders == [(2, 10), (2, 10)]
    assert backend.task_catalogs == [("alpha", "beta", "zeta")] * 2
    assert dataset_manifest_digest(first) == dataset_manifest_digest(second)
    manifest = json.loads((first / "source-manifest.json").read_text())
    assert manifest["source_episode_ids"] == [2, 10]
    assert manifest["task_catalog"] == ["alpha", "beta", "zeta"]


def test_default_merge_uses_pre_encoded_video_mode_without_video_writer_or_thread_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gear_sonic.data import exporter as exporter_module

    stage = write_stage(tmp_path, _identity(31), _payload(tmp_path, 31))
    episodes = staging_module._validate_merge_inputs((stage,))
    tasks = tuple(sorted({row["task"] for episode in episodes for row in staging_module.iter_stage_rows(episode)}))
    calls = 0

    class ForbiddenVideoWriter:
        def __init__(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("conversion merge must not construct VideoWriter")

    monkeypatch.setattr(exporter_module, "VideoWriter", ForbiddenVideoWriter)
    before_threads = {thread.ident for thread in threading.enumerate()}
    output = tmp_path / "pre-encoded"
    staging_module._default_merge_backend(output, episodes, tasks)

    from gear_sonic.data.exporter import Gr00tDatasetMetadata

    key = "observation.images.ego_view"
    exported = output / Gr00tDatasetMetadata(repo_id="tmp/tmp_dataset", root=output).get_video_file_path(0, key)
    assert calls == 0
    new_threads = [thread for thread in threading.enumerate() if thread.ident not in before_threads]
    assert all(hasattr(thread, "tqdm_cls") for thread in new_threads)
    assert staging_module._hash_secure_file(exported) == episodes[0].videos[key].sha256


def test_default_backend_preserves_exact_immutable_staged_video_bytes(tmp_path: Path) -> None:
    _, episodes, _, output = _unpublished_real_output(tmp_path)
    episode = episodes[0]
    key = "observation.images.ego_view"
    from gear_sonic.data.exporter import Gr00tDatasetMetadata

    meta = Gr00tDatasetMetadata(repo_id="tmp/tmp_dataset", root=output)
    exported = output / meta.get_video_file_path(0, key)

    assert staging_module._hash_secure_file(exported) == episode.videos[key].sha256


def test_independent_validator_rejects_unrelated_same_structure_video(tmp_path: Path) -> None:
    _, episodes, tasks, output = _unpublished_real_output(tmp_path)
    from gear_sonic.data.exporter import Gr00tDatasetMetadata

    key = "observation.images.ego_view"
    meta = Gr00tDatasetMetadata(repo_id="tmp/tmp_dataset", root=output)
    exported = output / meta.get_video_file_path(0, key)
    unrelated = tmp_path / "unrelated.mp4"
    _write_video(unrelated, value_offset=17)
    shutil.copyfile(unrelated, exported)

    with pytest.raises(ValueError, match="video content differs"):
        staging_module._validate_gr00t_output(output, episodes, tasks)


def test_stage_rejects_non_h264_media_before_final_publication(tmp_path: Path) -> None:
    video = tmp_path / "mpeg4.mp4"
    _write_video(video, codec_name="mpeg4")
    with pytest.raises(ValueError, match="H264"):
        write_stage(
            tmp_path,
            _identity(33),
            StagePayload(
                rows=(_row(0, "task"), _row(1, "task")),
                videos={"observation.images.ego_view": video},
            ),
        )


def test_real_gr00t_exporter_adapts_timestamp_vector_to_exact_scalar_parquet(tmp_path: Path) -> None:
    output = tmp_path / "real-exporter"
    exporter = Gr00tDataExporter.create(
        save_root=output,
        fps=50,
        features={
            "observation.state": {
                "dtype": "float64",
                "shape": (2,),
                "names": ["a", "b"],
            }
        },
        modality_config={"state": {}, "action": {}, "video": {}, "annotation": {}},
        task="task",
        script_config={},
    )
    for index in range(2):
        exporter.add_frame(
            {
                "observation.state": np.array([index, index + 1], dtype=np.float64),
                "timestamp": np.array([np.float32(index / 50.0)], dtype=np.float32),
                "task": "task",
            }
        )
        assert isinstance(exporter.episode_buffer["timestamp"][-1], np.float32)

    exporter.save_episode()

    table = pq.read_table(output / exporter.meta.get_data_file_path(0))
    timestamp = table.column("timestamp").combine_chunks()
    assert timestamp.type == pa.float32()
    assert timestamp.null_count == 0
    assert timestamp.to_numpy().tobytes() == np.array([0.0, 0.02], dtype=np.float32).tobytes()


def test_merge_rejects_mixed_repositories_duplicate_ids_and_conversion_locks(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)
    mixed = write_stage(
        tmp_path,
        _identity(20, repo="different/source"),
        _payload(tmp_path, 20),
    )
    backend = RecordingMergeBackend()
    with pytest.raises(ValueError, match="source repository"):
        merge_stages(
            (stages[0], mixed),
            tmp_path / "mixed",
            merge_backend=backend,
            final_validator=backend.validate,
        )
    with pytest.raises(ValueError, match="duplicate source episode"):
        merge_stages(
            (stages[0], stages[0]),
            tmp_path / "duplicate",
            merge_backend=backend,
            final_validator=backend.validate,
        )

    changed_lock = write_stage(
        tmp_path,
        replace(_identity(21), encoder_sha256="1" * 64),
        _payload(tmp_path, 21),
    )
    with pytest.raises(ValueError, match="conversion identity"):
        merge_stages(
            (stages[0], changed_lock),
            tmp_path / "changed",
            merge_backend=backend,
            final_validator=backend.validate,
        )


def test_merge_rejects_output_inside_stage_or_staging_namespace_without_mutating_inputs(
    tmp_path: Path,
) -> None:
    stages = _two_stages(tmp_path)
    backend = RecordingMergeBackend()
    with pytest.raises(ValueError, match="input staging ancestry"):
        merge_stages(
            stages,
            stages[0].path / "nested-output",
            merge_backend=backend,
            final_validator=backend.validate,
        )
    with pytest.raises(ValueError, match="input staging ancestry"):
        merge_stages(
            stages,
            tmp_path / ".staging" / "unrelated-output",
            merge_backend=backend,
            final_validator=backend.validate,
        )
    assert all(can_resume(stage.path, stage.identity) for stage in stages)


def test_merge_rejects_dotdot_output_alias_before_overlap_checks(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)
    aliased_output = stages[0].path / "nested" / ".." / "dataset"

    with pytest.raises(ValueError, match=r"canonical.*output|\.\."):
        merge_stages(stages, aliased_output)

    assert all(can_resume(stage.path, stage.identity) for stage in stages)


def test_merge_rejects_legacy_dotdot_stage_alias_before_resolved_ancestry_checks(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-root"
    stage = write_stage(real_root, _identity(), _payload(tmp_path))
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    aliased_stage_path = decoy / ".." / "real-root" / stage.path.relative_to(real_root)
    legacy_aliased_stage = replace(stage, path=aliased_stage_path)

    with pytest.raises(ValueError, match=r"stage.*canonical|\.\."):
        merge_stages(
            (legacy_aliased_stage,),
            real_root / ".staging" / "merged-dataset",
        )

    assert can_resume(stage.path, stage.identity)


def test_merge_revalidates_stages_and_never_overwrites_existing_output(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)
    backend = RecordingMergeBackend()
    output = merge_stages(stages, tmp_path / "dataset", merge_backend=backend, final_validator=backend.validate)
    assert (
        merge_stages(
            stages,
            output,
            merge_backend=backend,
            final_validator=backend.validate,
            resume_existing=True,
        )
        == output
    )

    (output / "unexpected.bin").write_bytes(b"corrupt")
    with pytest.raises(FileExistsError, match="existing output"):
        merge_stages(
            stages,
            output,
            merge_backend=backend,
            final_validator=backend.validate,
            resume_existing=True,
        )
    with pytest.raises(FileExistsError, match="existing output"):
        merge_stages(stages, output, merge_backend=backend, final_validator=backend.validate)

    (stages[0].path / "frame-data.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="stage is not complete"):
        merge_stages(
            stages,
            tmp_path / "bad-stage",
            merge_backend=backend,
            final_validator=backend.validate,
        )


def test_resume_revalidates_semantics_even_when_corruption_checksums_are_regenerated(
    tmp_path: Path,
) -> None:
    stages = _two_stages(tmp_path)
    output = merge_stages(stages, tmp_path / "dataset")
    from gear_sonic.data.exporter import Gr00tDatasetMetadata

    meta = Gr00tDatasetMetadata(repo_id="tmp/tmp_dataset", root=output)
    parquet_path = output / meta.get_data_file_path(0)
    table = pq.read_table(parquet_path)
    column = table.column("observation.state").combine_chunks()
    values = column.values.to_numpy(zero_copy_only=False).copy()
    values[0] += 1e-6
    changed = pa.FixedSizeListArray.from_arrays(pa.array(values, type=pa.float64()), 43)
    table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state", changed)
    pq.write_table(table, parquet_path)
    artifacts = tuple(
        path for path in staging_module._artifact_paths(output) if path != "dataset-checksums.sha256"
    )
    (output / "dataset-checksums.sha256").write_bytes(staging_module._checksums_bytes(output, artifacts))

    with pytest.raises(FileExistsError, match="existing output"):
        merge_stages(stages, output, resume_existing=True)


def test_merge_failure_cleans_sibling_temp_and_leaves_output_absent(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)

    def fail(output, episodes, tasks):
        output.mkdir()
        (output / "partial").write_text("partial")
        raise RuntimeError("boom")

    final = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="boom"):
        merge_stages(
            stages,
            final,
            merge_backend=fail,
            final_validator=lambda output, episodes, tasks: _exact_verification(episodes, tasks),
        )
    assert not final.exists()
    assert not list(tmp_path.glob(".dataset.*"))


def test_merge_rejects_independent_final_validation_mismatch_and_cleans_temp(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)

    def backend(output, episodes, tasks):
        staging_module._default_merge_backend(output, episodes, tasks)

    def wrong_validator(output, episodes, tasks):
        report = _exact_verification(episodes, tasks)
        return replace(report, frame_content_sha256="0" * 64)

    final = tmp_path / "wrong-verification"
    with pytest.raises(ValueError, match="validation hook differs"):
        merge_stages(
            stages,
            final,
            merge_backend=backend,
            final_validator=wrong_validator,
        )
    assert not final.exists()
    assert not list(tmp_path.glob(".wrong-verification.*"))


def test_custom_final_hook_cannot_replace_mandatory_builtin_output_validation(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)

    def backend_json_only(output, episodes, tasks):
        output.mkdir()
        (output / "backend.json").write_text("{}")

    with pytest.raises(ValueError):
        merge_stages(
            stages,
            tmp_path / "backend-json-only",
            merge_backend=backend_json_only,
            final_validator=lambda output, episodes, tasks: _exact_verification(episodes, tasks),
        )


def test_mutating_final_hook_runs_after_fsync_but_before_mandatory_builtin_validation(
    tmp_path: Path,
) -> None:
    stage = write_stage(tmp_path, _identity(3), _payload(tmp_path, 3))

    def mutating_hook(output, episodes, tasks):
        assert (output / "source-manifest.json").is_file()
        assert (output / "merge-validation.json").is_file()
        assert (output / "dataset-checksums.sha256").is_file()
        from gear_sonic.data.exporter import Gr00tDatasetMetadata

        meta = Gr00tDatasetMetadata(repo_id="tmp/tmp_dataset", root=output)
        parquet_path = output / meta.get_data_file_path(0)
        table = pq.read_table(parquet_path)
        column = table.column("observation.state").combine_chunks()
        values = column.values.to_numpy(zero_copy_only=False).copy()
        values[0] += 1e-6
        changed = pa.FixedSizeListArray.from_arrays(pa.array(values, type=pa.float64()), 43)
        pq.write_table(
            table.set_column(
                table.schema.get_field_index("observation.state"),
                "observation.state",
                changed,
            ),
            parquet_path,
        )
        artifacts = tuple(
            path for path in staging_module._artifact_paths(output) if path != "dataset-checksums.sha256"
        )
        (output / "dataset-checksums.sha256").write_bytes(staging_module._checksums_bytes(output, artifacts))
        return _exact_verification(episodes, tasks)

    final = tmp_path / "mutated-by-hook"
    with pytest.raises(ValueError, match="nonvideo value differs"):
        merge_stages((stage,), final, final_validator=mutating_hook)

    assert not final.exists()


def test_final_metadata_contract_is_exact_including_full_video_info(tmp_path: Path) -> None:
    _, episodes, tasks, output = _unpublished_real_output(tmp_path)
    info_path = output / "meta" / "info.json"
    modality_path = output / "meta" / "modality.json"
    original_info = json.loads(info_path.read_text())
    original_modality = json.loads(modality_path.read_text())
    mutations = (
        ("fps", lambda info, modality: info.__setitem__("fps", 49)),
        ("robot_type", lambda info, modality: info.__setitem__("robot_type", "not-g1")),
        (
            "script_config",
            lambda info, modality: info.__setitem__("script_config", {"unexpected": True}),
        ),
        (
            "data_path",
            lambda info, modality: info.__setitem__("data_path", "data/episode_{episode_index}.parquet"),
        ),
        (
            "video_path",
            lambda info, modality: info.__setitem__("video_path", "video/{video_key}.mp4"),
        ),
        (
            "video feature info",
            lambda info, modality: info["features"]["observation.images.ego_view"]["info"].__setitem__(
                "video.fps", 49
            ),
        ),
        (
            "modality",
            lambda info, modality: modality["video"].__setitem__("ego_view", {"original_key": "wrong"}),
        ),
    )

    for _, mutate in mutations:
        info = deepcopy(original_info)
        modality = deepcopy(original_modality)
        mutate(info, modality)
        info_path.write_text(json.dumps(info))
        modality_path.write_text(json.dumps(modality))
        with pytest.raises(ValueError, match="metadata|modality|path|feature"):
            staging_module._validate_gr00t_output(output, episodes, tasks)


@pytest.mark.parametrize("artifact_kind", ("data", "video"))
def test_final_metadata_path_traversal_is_rejected_even_for_valid_external_artifact(
    tmp_path: Path, artifact_kind: str
) -> None:
    _, episodes, tasks, output = _unpublished_real_output(tmp_path)
    from gear_sonic.data.exporter import Gr00tDatasetMetadata

    meta = Gr00tDatasetMetadata(repo_id="tmp/tmp_dataset", root=output)
    info_path = output / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if artifact_kind == "data":
        source = output / meta.get_data_file_path(0)
        external = output.parent / "external.parquet"
        info["data_path"] = "../external.parquet"
    else:
        source = output / meta.get_video_file_path(0, "observation.images.ego_view")
        external = output.parent / "external.mp4"
        info["video_path"] = "../external.mp4"
    shutil.copyfile(source, external)
    info_path.write_text(json.dumps(info))

    with pytest.raises(ValueError, match="canonical|contain|metadata"):
        staging_module._validate_gr00t_output(output, episodes, tasks)


def test_final_artifact_resolved_path_must_remain_inside_output(tmp_path: Path) -> None:
    _, episodes, tasks, output = _unpublished_real_output(tmp_path)
    data_chunk = output / "data" / "chunk-000"
    external = output.parent / "external-data"
    data_chunk.rename(external)
    data_chunk.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="contain"):
        staging_module._validate_gr00t_output(output, episodes, tasks)


def test_statistics_are_memmap_bounded_match_long_float32_and_clean_up() -> None:
    from lerobot.common.datasets.lerobot_dataset import compute_episode_stats

    features = {"x": {"dtype": "float32", "shape": (1,), "names": None}}
    values = np.linspace(-1000.0, 1000.0, 5000, dtype=np.float32)
    with staging_module._StreamingStats(features, expected_count=len(values)) as verifier:
        temporary_root = verifier.temporary_root
        assert isinstance(verifier.values["x"], np.memmap)
        for value in values:
            verifier.update({"x": np.array([value], dtype=np.float32)})
        expected = compute_episode_stats({"x": values}, features)

        verifier.verify(expected)
        corrupted = deepcopy(expected)
        corrupted["x"]["mean"] = corrupted["x"]["mean"] + np.float32(1e-6)
        with pytest.raises(ValueError, match="x.mean"):
            verifier.verify(corrupted)

    assert not temporary_root.exists()


def test_injected_backend_without_hook_still_runs_builtin_final_validation(tmp_path: Path) -> None:
    stages = _two_stages(tmp_path)
    output = merge_stages(stages, tmp_path / "safe", merge_backend=RecordingMergeBackend())
    assert output.is_dir()


def test_production_merge_reopens_and_validates_real_gr00t_output(tmp_path: Path) -> None:
    stage = write_stage(tmp_path, _identity(3), _payload(tmp_path, 3))

    output = merge_stages((stage,), tmp_path / "real-dataset")

    assert output.is_dir()
    assert len(dataset_manifest_digest(output)) == 64
    validation = json.loads((output / "merge-validation.json").read_text())
    assert validation["total_frames"] == 2
    assert validation["source_episode_ids"] == [3]


def test_independent_validator_rejects_corrupt_nonvideo_despite_consistent_backend_report(
    tmp_path: Path,
) -> None:
    stage = write_stage(tmp_path, _identity(3), _payload(tmp_path, 3))

    def corrupting_backend(output, episodes, tasks):
        staging_module._default_merge_backend(output, episodes, tasks)
        from gear_sonic.data.exporter import Gr00tDatasetMetadata

        meta = Gr00tDatasetMetadata(repo_id="tmp/tmp_dataset", root=output)
        parquet_path = output / meta.get_data_file_path(0)
        table = pq.read_table(parquet_path)
        column = table.column("observation.state").combine_chunks()
        values = column.values.to_numpy(zero_copy_only=False).copy()
        values[0] += 1e-6
        changed = pa.FixedSizeListArray.from_arrays(pa.array(values, type=pa.float64()), 43)
        table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state", changed)
        pq.write_table(table, parquet_path)
        return _exact_verification(episodes, tasks)

    final = tmp_path / "corrupt-dataset"
    with pytest.raises(ValueError, match="nonvideo value differs"):
        merge_stages(
            (stage,),
            final,
            merge_backend=corrupting_backend,
            final_validator=staging_module._validate_gr00t_output,
        )
    assert not final.exists()


def test_independent_validator_rejects_extra_global_task_catalog_entry(tmp_path: Path) -> None:
    stage = write_stage(tmp_path, _identity(3), _payload(tmp_path, 3))

    def extra_task_backend(output, episodes, tasks):
        staging_module._default_merge_backend(output, episodes, tasks)
        from gear_sonic.data.exporter import Gr00tDatasetMetadata

        meta = Gr00tDatasetMetadata(repo_id="tmp/tmp_dataset", root=output)
        meta.add_task("unexpected extra task")

    with pytest.raises(ValueError, match="global task catalog"):
        merge_stages(
            (stage,),
            tmp_path / "extra-task",
            merge_backend=extra_task_backend,
            final_validator=staging_module._validate_gr00t_output,
        )


def test_resume_rejects_duplicate_valid_task_record_even_with_regenerated_checksum(
    tmp_path: Path,
) -> None:
    stage = write_stage(tmp_path, _identity(3), _payload(tmp_path, 3))
    output = merge_stages((stage,), tmp_path / "duplicate-task-record")
    tasks_path = output / "meta" / "tasks.jsonl"
    first_line = tasks_path.read_text().splitlines()[0]
    with tasks_path.open("a") as stream:
        stream.write(first_line + "\n")
    artifacts = tuple(
        path for path in staging_module._artifact_paths(output) if path != "dataset-checksums.sha256"
    )
    (output / "dataset-checksums.sha256").write_bytes(staging_module._checksums_bytes(output, artifacts))

    episodes = staging_module._validate_merge_inputs((stage,))
    with pytest.raises(ValueError, match=r"task catalog.*exact record count|duplicate task_index"):
        staging_module._validate_gr00t_output(output, episodes, ("place", "pour"))
    with pytest.raises(FileExistsError, match="existing output"):
        merge_stages((stage,), output, resume_existing=True)


@pytest.mark.parametrize("mutation", ("unexpected-field", "wrong-order", "wrong-value"))
def test_independent_validator_rejects_noncanonical_raw_task_catalog(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, episodes, tasks, output = _unpublished_real_output(tmp_path)
    tasks_path = output / "meta" / "tasks.jsonl"
    records = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    if mutation == "unexpected-field":
        records[0]["unexpected"] = True
    elif mutation == "wrong-order":
        records.reverse()
    else:
        records[0]["task"] = "unexpected task value"
    tasks_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError, match=r"task catalog.*exact|task_index.*contiguous|lexicographic"):
        staging_module._validate_gr00t_output(output, episodes, tasks)


@pytest.mark.parametrize("metadata_name", ("episodes.jsonl", "episodes_stats.jsonl"))
def test_independent_validator_rejects_ghost_episode_metadata_keys(
    tmp_path: Path,
    metadata_name: str,
) -> None:
    stage = write_stage(tmp_path, _identity(3), _payload(tmp_path, 3))
    output = merge_stages((stage,), tmp_path / f"ghost-{metadata_name}")
    metadata_path = output / "meta" / metadata_name
    first = json.loads(metadata_path.read_text().splitlines()[0])
    first["episode_index"] = 999
    with metadata_path.open("a") as stream:
        stream.write(json.dumps(first) + "\n")
    episodes = staging_module._validate_merge_inputs((stage,))

    with pytest.raises(ValueError, match="episode metadata.*exact|statistics.*exact"):
        staging_module._validate_gr00t_output(output, episodes, ("place", "pour"))


@pytest.mark.parametrize("metadata_name", ("episodes.jsonl", "episodes_stats.jsonl"))
def test_independent_validator_rejects_duplicate_valid_episode_metadata_line(
    tmp_path: Path,
    metadata_name: str,
) -> None:
    _, episodes, tasks, output = _unpublished_real_output(tmp_path)
    metadata_path = output / "meta" / metadata_name
    line = metadata_path.read_text().splitlines()[0]
    with metadata_path.open("a") as stream:
        stream.write(line + "\n")

    with pytest.raises(ValueError, match=r"exact record count|duplicate episode_index"):
        staging_module._validate_gr00t_output(output, episodes, tasks)


@pytest.mark.parametrize("metadata_name", ("episodes.jsonl", "episodes_stats.jsonl"))
def test_independent_validator_rejects_unexpected_episode_metadata_field(
    tmp_path: Path,
    metadata_name: str,
) -> None:
    _, episodes, tasks, output = _unpublished_real_output(tmp_path)
    metadata_path = output / "meta" / metadata_name
    record = json.loads(metadata_path.read_text())
    record["unexpected"] = True
    metadata_path.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match=r"exact fields|unexpected"):
        staging_module._validate_gr00t_output(output, episodes, tasks)


def test_independent_validator_rejects_unexpected_regular_artifact(tmp_path: Path) -> None:
    stage = write_stage(tmp_path, _identity(3), _payload(tmp_path, 3))
    output = merge_stages((stage,), tmp_path / "unexpected-artifact")
    (output / "ghost.bin").write_bytes(b"ghost")
    episodes = staging_module._validate_merge_inputs((stage,))

    with pytest.raises(ValueError, match="unexpected.*artifact|canonical.*tree"):
        staging_module._validate_gr00t_output(output, episodes, ("place", "pour"))


def test_empty_merge_and_output_symlink_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        merge_stages((), tmp_path / "empty", merge_backend=RecordingMergeBackend())
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="existing output"):
        merge_stages(_two_stages(tmp_path), link, merge_backend=RecordingMergeBackend())
