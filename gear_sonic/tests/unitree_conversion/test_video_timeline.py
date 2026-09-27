from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
from types import MappingProxyType

import av
import numpy as np
import pytest

from gear_sonic.data.unitree_conversion.contracts import SourceVideoSegment
from gear_sonic.data.unitree_conversion.video_timeline import (
    CameraStreamReport,
    EpisodeCameraReport,
    ResampledVideoFrame,
    TargetCameraSchema,
    VideoTimeline,
    choose_target_camera_schema,
    inspect_video,
    iter_resampled_video,
)

_SIZE = (640, 480)
_CAMERA_MAP = {
    "observation.images.cam_left_high": "observation.images.ego_view",
    "observation.images.cam_left_wrist": "observation.images.left_wrist",
    "observation.images.cam_right_wrist": "observation.images.right_wrist",
}
_PRIMARY = "observation.images.cam_left_high"


def _h264_encoder_name() -> str:
    for name in ("libx264", "libopenh264", "h264"):
        try:
            av.codec.Codec(name, "w")
        except (av.error.FFmpegError, ValueError):
            continue
        return name
    pytest.skip("PyAV has no usable H.264 encoder")


def _write_rgb_video(
    path: Path,
    *,
    colors: tuple[tuple[int, int, int], ...] = ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
    size: tuple[int, int] = _SIZE,
    fps: int = 30,
) -> None:
    width, height = size
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(_h264_encoder_name(), rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for color in colors:
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            rgb[...] = color
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def rgb_video(tmp_path: Path) -> Path:
    path = tmp_path / "three_frames.mp4"
    _write_rgb_video(path)
    return path


def test_streams_exact_target_count_with_nearest_indices(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)

    frames = list(iter_resampled_video(timeline, target_fps=50))

    assert timeline.decoded_frames == 3
    assert timeline.source_fps == 30
    assert timeline.rgb_size == _SIZE
    assert len(frames) == 5
    assert [frame.target_index for frame in frames] == [0, 1, 2, 3, 4]
    assert [frame.source_index for frame in frames] == [0, 1, 1, 2, 2]
    assert all(frame.rgb.shape == (480, 640, 3) for frame in frames)
    assert all(frame.rgb.dtype == np.uint8 for frame in frames)
    assert np.mean(frames[0].rgb, axis=(0, 1)).argmax() == 0
    assert np.mean(frames[1].rgb, axis=(0, 1)).argmax() == 1
    assert np.mean(frames[-1].rgb, axis=(0, 1)).argmax() == 2


def test_decodes_nonzero_episode_interval_from_shared_h264_shard(tmp_path: Path) -> None:
    path = tmp_path / "two_episodes.mp4"
    _write_rgb_video(
        path,
        colors=((255, 0, 0),) * 3 + ((0, 0, 255),) * 3,
    )
    segment = SourceVideoSegment(
        source_key="observation.images.cam_left_high",
        path=path,
        from_timestamp=3 / 30,
        to_timestamp=6 / 30,
        start_frame=3,
        end_frame=6,
        frame_count=3,
    )

    timeline = inspect_video(
        path,
        expected_frames=3,
        expected_size=_SIZE,
        segment=segment,
    )
    frames = list(iter_resampled_video(timeline))

    assert timeline.source_key == segment.source_key
    assert timeline.from_timestamp == 3 / 30
    assert timeline.to_timestamp == 6 / 30
    assert timeline.start_frame == 3
    assert timeline.end_frame == 6
    assert [frame.source_index for frame in frames] == [0, 1, 1, 2, 2]
    assert all(np.mean(frame.rgb, axis=(0, 1)).argmax() == 2 for frame in frames)


def test_inspection_rejects_segment_path_or_count_mismatch(rgb_video: Path, tmp_path: Path) -> None:
    other_path = tmp_path / "other.mp4"
    other_path.write_bytes(rgb_video.read_bytes())
    segment = SourceVideoSegment(
        source_key="camera",
        path=other_path,
        from_timestamp=0.0,
        to_timestamp=3 / 30,
        start_frame=0,
        end_frame=3,
        frame_count=3,
    )

    with pytest.raises(ValueError, match="segment path must equal video path"):
        inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE, segment=segment)
    with pytest.raises(ValueError, match="segment frame_count must equal expected_frames"):
        inspect_video(other_path, expected_frames=4, expected_size=_SIZE, segment=segment)


def test_resampled_frames_are_independent_deeply_immutable_rgb_copies(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)

    frames = list(iter_resampled_video(timeline))

    assert frames[1].source_index == frames[2].source_index == 1
    assert not np.shares_memory(frames[1].rgb, frames[2].rgb)
    assert all(frame.rgb.flags.c_contiguous and not frame.rgb.flags.writeable for frame in frames)
    with pytest.raises(ValueError, match="read-only"):
        frames[0].rgb[0, 0, 0] = 0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        frames[0].rgb.setflags(write=True)


def test_primary_count_mismatch_is_not_repaired(rgb_video: Path) -> None:
    with pytest.raises(ValueError, match="decoded frame count 3 != data frame count 4"):
        inspect_video(rgb_video, expected_frames=4, expected_size=_SIZE)


def test_inspection_rejects_wrong_rgb_size_without_repair(rgb_video: Path) -> None:
    with pytest.raises(ValueError, match=r"RGB shape \(480, 640, 3\) != expected \(240, 320, 3\)"):
        inspect_video(rgb_video, expected_frames=3, expected_size=(320, 240))


def test_inspection_requires_exact_nominal_thirty_fps(tmp_path: Path) -> None:
    path = tmp_path / "wrong_fps.mp4"
    _write_rgb_video(path, fps=25)

    with pytest.raises(ValueError, match="nominal video fps 25 != required 30"):
        inspect_video(path, expected_frames=3, expected_size=_SIZE)


@pytest.mark.parametrize("path", ["missing.mp4", Path("missing.mp4")])
def test_inspection_requires_a_regular_video_file(tmp_path: Path, path: object) -> None:
    with pytest.raises(ValueError, match="video path must be an existing regular file"):
        inspect_video(tmp_path / path, expected_frames=3, expected_size=_SIZE)


@pytest.mark.parametrize("expected_frames", [True, 0, 1, -1, 3.0, "3"])
def test_inspection_requires_integer_episode_frame_count_at_least_two(
    rgb_video: Path,
    expected_frames: object,
) -> None:
    with pytest.raises(ValueError, match="expected_frames must be a non-boolean integer of at least two"):
        inspect_video(rgb_video, expected_frames=expected_frames, expected_size=_SIZE)


@pytest.mark.parametrize(
    "expected_size",
    [None, (), (640,), (640, 480, 3), (True, 480), (640.0, 480), (0, 480), (640, -1)],
)
def test_inspection_requires_positive_integer_width_height(
    rgb_video: Path,
    expected_size: object,
) -> None:
    with pytest.raises(ValueError, match="expected_size must be exactly two positive integer values"):
        inspect_video(rgb_video, expected_frames=3, expected_size=expected_size)


def test_second_pass_rejects_a_changed_file(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    original_stat = rgb_video.stat()
    with rgb_video.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        final_byte = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([final_byte[0] ^ 0x01]))
    os.utime(
        rgb_video,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert rgb_video.stat().st_size == timeline.file_size
    assert rgb_video.stat().st_mtime_ns == timeline.mtime_ns

    with pytest.raises(ValueError, match="video file changed after inspection"):
        list(iter_resampled_video(timeline))


def test_second_pass_rejects_a_truncated_file(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    with rgb_video.open("r+b") as stream:
        stream.truncate(max(1, timeline.file_size // 2))

    with pytest.raises(ValueError, match="video file changed after inspection"):
        list(iter_resampled_video(timeline))


def test_second_pass_rejects_same_identity_mutation_after_first_yield(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    frames = iter_resampled_video(timeline)
    first = next(frames)
    original_stat = rgb_video.stat()
    with rgb_video.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        final_byte = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([final_byte[0] ^ 0x01]))
    os.utime(
        rgb_video,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert first.target_index == 0
    assert rgb_video.stat().st_size == timeline.file_size
    assert rgb_video.stat().st_mtime_ns == timeline.mtime_ns

    delivered = [first]
    with pytest.raises(ValueError, match="video file changed after inspection"):
        while True:
            delivered.append(next(frames))
    assert len(delivered) == 4


def test_receiving_exact_target_count_proves_iterator_finalized(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    frames = iter_resampled_video(timeline)

    delivered = [next(frames) for _ in range(5)]

    assert [frame.target_index for frame in delivered] == list(range(5))
    with pytest.raises(StopIteration):
        next(frames)


@pytest.mark.parametrize("target_fps", [True, 49, 50.0, "50", None])
def test_resampling_accepts_only_the_exact_integer_target_fps(
    rgb_video: Path,
    target_fps: object,
) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)

    with pytest.raises(ValueError, match="target_fps must be integer 50"):
        list(iter_resampled_video(timeline, target_fps=target_fps))


def test_video_timeline_is_frozen_and_uses_resolved_identity(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)

    assert timeline.path == rgb_video.resolve()
    assert timeline.path.is_absolute()
    assert timeline.expected_frames == timeline.decoded_frames == 3
    assert len(timeline.sha256) == 64
    assert timeline.file_size == rgb_video.stat().st_size
    with pytest.raises(FrozenInstanceError):
        timeline.decoded_frames = 4


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("decoded_frames", True, "decoded_frames"),
        ("decoded_frames", 1, "decoded_frames"),
        ("source_fps", 29, "source_fps"),
        ("rgb_size", (0, 480), "rgb_size"),
        ("sha256", "0" * 63, "sha256"),
        ("file_size", 0, "file_size"),
        ("mtime_ns", -1, "mtime_ns"),
    ],
)
def test_video_timeline_rejects_invalid_direct_construction(
    rgb_video: Path,
    field_name: str,
    value: object,
    message: str,
) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)

    with pytest.raises(ValueError, match=message):
        replace(timeline, **{field_name: value})


def test_resampled_frame_validates_and_detaches_input() -> None:
    source = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    frame = ResampledVideoFrame(target_index=2, source_index=1, rgb=source)
    source[...] = 0

    assert frame.target_index == 2
    assert frame.source_index == 1
    assert frame.rgb[0, 0, 1] == 1
    assert not frame.rgb.flags.writeable
    with pytest.raises(FrozenInstanceError):
        frame.source_index = 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_index": True}, "target_index"),
        ({"target_index": -1}, "target_index"),
        ({"source_index": 1.0}, "source_index"),
        ({"source_index": -1}, "source_index"),
        ({"rgb": np.zeros((2, 3, 3), dtype=np.float32)}, "RGB uint8"),
        ({"rgb": np.zeros((2, 3), dtype=np.uint8)}, "RGB uint8"),
        ({"rgb": np.zeros((2, 3, 4), dtype=np.uint8)}, "RGB uint8"),
    ],
)
def test_resampled_frame_rejects_invalid_direct_construction(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values = {
        "target_index": 0,
        "source_index": 0,
        "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        ResampledVideoFrame(**values)


def _exact_stream(timeline: VideoTimeline) -> CameraStreamReport:
    return CameraStreamReport(status="exact", timeline=timeline)


def _episode_report(
    timeline: VideoTimeline,
    *,
    episode_id: int,
    streams: dict[str, CameraStreamReport] | None = None,
    primary_camera: str = _PRIMARY,
) -> EpisodeCameraReport:
    exact_streams = {source: _exact_stream(timeline) for source in _CAMERA_MAP} if streams is None else streams
    return EpisodeCameraReport(
        source_repo_id="unitreerobotics/G1_Dex3_Pouring_Dataset",
        source_episode_id=episode_id,
        source_frame_count=3,
        primary_camera=primary_camera,
        streams=exact_streams,
    )


def test_camera_schema_includes_primary_and_only_cohort_complete_optionals(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    reports = [_episode_report(timeline, episode_id=78), _episode_report(timeline, episode_id=0)]

    schema = choose_target_camera_schema(reports, _CAMERA_MAP)

    assert isinstance(schema, TargetCameraSchema)
    assert schema.primary_source == _PRIMARY
    assert schema.primary_target == "observation.images.ego_view"
    assert dict(schema.source_to_target) == _CAMERA_MAP
    assert schema.target_features == (
        "observation.images.ego_view",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )
    assert dict(schema.omission_reasons) == {}
    assert isinstance(schema.source_to_target, MappingProxyType)
    with pytest.raises(TypeError):
        schema.source_to_target[_PRIMARY] = "changed"
    with pytest.raises(FrozenInstanceError):
        schema.primary_source = "changed"


def test_camera_schema_omits_optional_camera_from_every_episode_when_one_is_missing(
    rgb_video: Path,
) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    missing_key = "observation.images.cam_left_wrist"
    first_streams = {source: _exact_stream(timeline) for source in _CAMERA_MAP}
    second_streams = {source: _exact_stream(timeline) for source in _CAMERA_MAP}
    second_streams[missing_key] = CameraStreamReport(
        status="missing",
        reason="source stream absent",
    )
    reports = [
        _episode_report(timeline, episode_id=78, streams=first_streams),
        _episode_report(timeline, episode_id=0, streams=second_streams),
    ]

    schema = choose_target_camera_schema(reports, _CAMERA_MAP)

    assert missing_key not in schema.source_to_target
    assert "observation.images.left_wrist" not in schema.target_features
    assert schema.omission_reasons == {
        missing_key: ("unitreerobotics/G1_Dex3_Pouring_Dataset episode 0: missing: source stream absent",)
    }
    assert all(
        missing_key not in report.streams or report.streams[missing_key].status != "exact"
        for report in reports[1:]
    )


def test_camera_schema_records_all_optional_failures_in_identity_order(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    optional = "observation.images.cam_right_wrist"
    invalid = CameraStreamReport(status="invalid", reason="decoded frame count mismatch")
    missing = CameraStreamReport(status="missing", reason="source stream absent")
    reports = [
        _episode_report(
            timeline,
            episode_id=78,
            streams={_PRIMARY: _exact_stream(timeline), optional: invalid},
        ),
        _episode_report(
            timeline,
            episode_id=0,
            streams={_PRIMARY: _exact_stream(timeline), optional: missing},
        ),
    ]

    schema = choose_target_camera_schema(
        reports,
        {_PRIMARY: _CAMERA_MAP[_PRIMARY], optional: _CAMERA_MAP[optional]},
    )

    assert schema.omission_reasons[optional] == (
        "unitreerobotics/G1_Dex3_Pouring_Dataset episode 0: missing: source stream absent",
        "unitreerobotics/G1_Dex3_Pouring_Dataset episode 78: invalid: decoded frame count mismatch",
    )


@pytest.mark.parametrize("primary_status", ["missing", "invalid"])
def test_camera_schema_rejects_nonexact_primary_for_any_episode(
    rgb_video: Path,
    primary_status: str,
) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    report = _episode_report(
        timeline,
        episode_id=0,
        streams={
            _PRIMARY: CameraStreamReport(status=primary_status, reason="primary failure"),
        },
    )

    with pytest.raises(ValueError, match=r"required primary .* episode 0 .*primary failure"):
        choose_target_camera_schema([report], {_PRIMARY: _CAMERA_MAP[_PRIMARY]})


def test_camera_schema_treats_an_absent_primary_report_as_missing(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    report = _episode_report(timeline, episode_id=0, streams={})

    with pytest.raises(ValueError, match=r"required primary .* episode 0 .*missing"):
        choose_target_camera_schema([report], {_PRIMARY: _CAMERA_MAP[_PRIMARY]})


def test_camera_schema_rejects_episode_dependent_primary_configuration(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    reports = [
        _episode_report(timeline, episode_id=0),
        _episode_report(timeline, episode_id=1, primary_camera="observation.images.other"),
    ]

    with pytest.raises(ValueError, match="every episode report must use the same primary_camera"):
        choose_target_camera_schema(reports, _CAMERA_MAP)


@pytest.mark.parametrize(
    ("camera_map", "message"),
    [
        ([], "camera_map must be a mapping"),
        ({}, "camera_map must not be empty"),
        ({_PRIMARY: ""}, "camera_map keys and values"),
        ({"": "observation.images.ego_view"}, "camera_map keys and values"),
        (
            {_PRIMARY: "observation.images.ego_view", "optional": "observation.images.ego_view"},
            "target camera keys must be unique",
        ),
        ({"optional": "observation.images.left_wrist"}, "primary_camera must be present in camera_map"),
    ],
)
def test_camera_schema_rejects_invalid_or_ambiguous_mapping(
    rgb_video: Path,
    camera_map: object,
    message: str,
) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    report = _episode_report(timeline, episode_id=0)

    with pytest.raises(ValueError, match=message):
        choose_target_camera_schema([report], camera_map)


def test_camera_schema_requires_nonempty_unique_episode_reports(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    report = _episode_report(timeline, episode_id=0)

    with pytest.raises(ValueError, match="episode_reports must not be empty"):
        choose_target_camera_schema([], _CAMERA_MAP)
    with pytest.raises(ValueError, match="episode report identities must be unique"):
        choose_target_camera_schema([report, report], _CAMERA_MAP)


def test_episode_camera_report_is_deeply_immutable_and_requires_exact_count(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    streams = {_PRIMARY: _exact_stream(timeline)}
    report = _episode_report(timeline, episode_id=0, streams=streams)
    streams.clear()

    assert tuple(report.streams) == (_PRIMARY,)
    with pytest.raises(TypeError):
        report.streams[_PRIMARY] = CameraStreamReport(status="missing", reason="changed")
    with pytest.raises(ValueError, match="exact timeline decoded_frames must equal source_frame_count"):
        replace(report, source_frame_count=4)


def test_episode_camera_report_rejects_mixed_nonstr_stream_keys_before_sorting(rgb_video: Path) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)
    exact = _exact_stream(timeline)

    with pytest.raises(ValueError, match="stream camera key must be a nonempty string"):
        _episode_report(
            timeline,
            episode_id=0,
            streams={_PRIMARY: exact, ("not", "a", "string"): exact},
        )


@pytest.mark.parametrize(
    ("status", "timeline_present", "reason", "message"),
    [
        ("unknown", False, "reason", "status"),
        ("exact", False, None, "exact camera report requires a timeline"),
        ("exact", True, "reason", "exact camera report must not contain a reason"),
        ("missing", True, "reason", "nonexact camera report must not contain a timeline"),
        ("missing", False, None, "nonexact camera report requires a reason"),
        ("invalid", False, "", "nonexact camera report requires a reason"),
    ],
)
def test_camera_stream_report_validates_status_payload(
    rgb_video: Path,
    status: str,
    timeline_present: bool,
    reason: str | None,
    message: str,
) -> None:
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=_SIZE)

    with pytest.raises(ValueError, match=message):
        CameraStreamReport(
            status=status,
            timeline=timeline if timeline_present else None,
            reason=reason,
        )


@pytest.mark.parametrize("status", [None, True, 1, [], {}])
def test_camera_stream_report_rejects_nonstr_status_as_value_error(status: object) -> None:
    with pytest.raises(ValueError, match="camera report status must be a string"):
        CameraStreamReport(status=status, reason="invalid status")
