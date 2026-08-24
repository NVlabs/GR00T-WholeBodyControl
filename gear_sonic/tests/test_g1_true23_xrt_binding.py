from __future__ import annotations

import hashlib
from pathlib import Path
import queue
import signal
import threading
import time
from typing import Any

import pytest

from gear_sonic.scripts import (
    stream_g1_23dof_pico_causal_zmq as publisher,
    stream_g1_23dof_pico_raw_worker as worker,
)
from gear_sonic.utils.g1_23dof_xr24_soma_stream import (
    CausalHistorySemanticProducer,
    ROLLING_SOMA_KIND,
    validate_causal_history_packet,
)


class _ControlledCaptureWorker:
    def __init__(self) -> None:
        self.frames: queue.Queue[dict[str, Any]] = queue.Queue()

    def next_frame(self) -> dict[str, Any]:
        return self.frames.get(timeout=1.0)


def _capture_frame(
    index: int,
    *,
    capture_ns: int | None = None,
    body_timestamp_ns: int | None = None,
    body_sequence: int | None = None,
    body_poses: Any = None,
) -> dict[str, Any]:
    measured_ns = (
        time.monotonic_ns() - 1_000_000 + index
        if capture_ns is None
        else capture_ns
    )
    return {
        "frame_index": index,
        "capture_monotonic_ns": measured_ns,
        "body_sample_timestamp_ns": (
            measured_ns if body_timestamp_ns is None else body_timestamp_ns
        ),
        "body_sample_sequence": (
            index + 100 if body_sequence is None else body_sequence
        ),
        "body_poses": (
            {"measured_pose_marker": index}
            if body_poses is None
            else body_poses
        ),
    }


def _delivery(
    index: int,
    capture_ns: int,
    *,
    body_timestamp_ns: int,
    body_sequence: int,
    body_poses: Any = None,
) -> dict[str, Any]:
    return {
        "frame": _capture_frame(
            index,
            capture_ns=capture_ns,
            body_timestamp_ns=body_timestamp_ns,
            body_sequence=body_sequence,
            body_poses=body_poses,
        ),
        "request_started_monotonic_ns": capture_ns - 2,
        "received_monotonic_ns": capture_ns + 2,
        "capture_wait_duration_ns": 4,
        "capture_ipc_duration_ns": 2,
        "capture_frame_index_delta": 1,
        "body_sample_sequence_delta": 1,
        "body_source_sequence_gap_count": 0,
        "queue_depth_before_enqueue": 0,
        "queue_depth_after_dequeue": 0,
        "mailbox_batch_size": 1,
        "mailbox_coalesced_prior_frames": 0,
    }


def _xr24_pose_frame(value: float) -> list[list[float]]:
    return [
        [value + role * 0.001, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        for role in range(24)
    ]


def _write_binding(directory: Path, payload: bytes) -> Path:
    directory.mkdir(parents=True)
    binding = directory / "xrobotoolkit_sdk.cpython-310-x86_64-linux-gnu.so"
    binding.write_bytes(payload)
    return binding.resolve()


def test_neutral_prime_retries_only_measured_rejected_frames() -> None:
    class Rolling:
        def prime(self, body_poses: dict[str, int]) -> dict[str, bool]:
            if body_poses["measured_pose_marker"] < 3:
                raise ValueError(
                    "XR24 acquisition pose is not neutral standing: "
                    "pelvis_height_m"
                )
            return {"primed": True}

    class Worker:
        def __init__(self) -> None:
            self.frames = iter([_capture_frame(1), _capture_frame(2), _capture_frame(3)])

        def next_frame(self) -> dict[str, Any]:
            return next(self.frames)

    rejections: list[tuple[int, int, str]] = []
    frame, report, count = publisher._prime_until_neutral_standing(  # noqa: SLF001
        Rolling(),
        Worker(),  # type: ignore[arg-type]
        initial_frame=_capture_frame(0),
        deadline_ns=time.monotonic_ns() + 1_000_000_000,
        on_rejection=lambda n, f, reason: rejections.append(
            (n, f["frame_index"], reason)
        ),
    )

    assert frame["frame_index"] == 3
    assert report == {"primed": True}
    assert count == 3
    assert rejections == [
        (
            1,
            0,
            "XR24 acquisition pose is not neutral standing: pelvis_height_m",
        )
    ]


def test_neutral_prime_does_not_swallow_other_value_errors() -> None:
    class Rolling:
        def prime(self, _body_poses: Any) -> dict[str, bool]:
            raise ValueError("solver input is malformed")

    with pytest.raises(ValueError, match="solver input is malformed"):
        publisher._prime_until_neutral_standing(  # noqa: SLF001
            Rolling(),
            object(),  # type: ignore[arg-type]
            initial_frame=_capture_frame(0),
            deadline_ns=time.monotonic_ns() + 1_000_000_000,
        )


def test_default_xrt_binding_is_anchored_to_selected_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "mounted-current-repo"
    current = _write_binding(
        workspace / publisher._XRT_MODULE_RELATIVE_DIR,  # noqa: SLF001
        b"current-binding-with-source-coherence",
    )
    stale = _write_binding(
        tmp_path
        / "root-stale-clone"
        / "external_dependencies"
        / "XRoboToolkit-PC-Service-Pybind_X86_and_ARM64",
        b"stale-binding",
    )

    selected_dir = publisher._default_xrt_module_dir(workspace)  # noqa: SLF001
    selected = publisher._resolve_xrt_module_binary(selected_dir)  # noqa: SLF001

    assert selected == current
    assert selected != stale
    assert "root-stale-clone" not in str(selected)


def test_worker_rejects_stale_clone_even_with_same_binding_bytes(
    tmp_path: Path,
) -> None:
    expected = _write_binding(tmp_path / "current", b"same-extension-bytes")
    stale = _write_binding(tmp_path / "root-stale", b"same-extension-bytes")
    digest = hashlib.sha256(expected.read_bytes()).hexdigest()

    with pytest.raises(RuntimeError, match="imported XRT module path mismatch"):
        worker._verify_imported_xrt_binding(  # noqa: SLF001
            stale,
            expected_path=expected,
            expected_sha256=digest,
        )


def test_worker_rejects_changed_binding_hash(tmp_path: Path) -> None:
    expected = _write_binding(tmp_path / "current", b"current-extension")
    stale_digest = hashlib.sha256(b"stale-extension").hexdigest()

    with pytest.raises(RuntimeError, match="imported XRT module hash mismatch"):
        worker._verify_imported_xrt_binding(  # noqa: SLF001
            expected,
            expected_path=expected,
            expected_sha256=stale_digest,
        )


def test_worker_reports_exact_verified_binding(tmp_path: Path) -> None:
    expected = _write_binding(tmp_path / "current", b"current-extension")
    digest = hashlib.sha256(expected.read_bytes()).hexdigest()

    assert worker._verify_imported_xrt_binding(  # noqa: SLF001
        expected,
        expected_path=expected,
        expected_sha256=digest,
    ) == {"path": str(expected), "sha256": digest}


def test_background_capture_preserves_fifo_without_drops() -> None:
    controlled = _ControlledCaptureWorker()
    capture = publisher.BackgroundRawCapture(
        controlled,  # type: ignore[arg-type]
        deadline_ns=time.monotonic_ns() + 5_000_000_000,
        capacity=4,
    )
    capture.start()
    try:
        received = []
        for index in range(3):
            controlled.frames.put(_capture_frame(index))
            delivery = capture.next_frame(
                deadline_ns=time.monotonic_ns() + 1_000_000_000
            )
            received.append(delivery["frame"]["frame_index"])
            assert delivery["capture_wait_duration_ns"] >= 0
            assert delivery["capture_ipc_duration_ns"] >= 0
        assert received == [0, 1, 2]
        assert capture.stats()["frames_silently_dropped"] == 0
    finally:
        capture.request_stop()
        controlled.frames.put(_capture_frame(3))
        capture.join()


def test_background_capture_queue_overflow_fails_closed() -> None:
    controlled = _ControlledCaptureWorker()
    capture = publisher.BackgroundRawCapture(
        controlled,  # type: ignore[arg-type]
        deadline_ns=time.monotonic_ns() + 5_000_000_000,
        capacity=1,
    )
    capture.start()
    controlled.frames.put(_capture_frame(0))
    controlled.frames.put(_capture_frame(1))
    deadline = time.monotonic() + 1.0
    while capture.stats()["failure"] is None and time.monotonic() < deadline:
        time.sleep(0.001)
    try:
        with pytest.raises(RuntimeError, match="queue overflow"):
            capture.next_frame(
                deadline_ns=time.monotonic_ns() + 1_000_000_000
            )
        stats = capture.stats()
        assert stats["queue_overflowed"] is True
        assert stats["frames_silently_dropped"] == 0
    finally:
        capture.request_stop()
        capture.join()


def test_publisher_age_budget_accepts_only_fresh_reference() -> None:
    control_ns = 1_000_000_000
    assert publisher._require_publisher_age_budget(  # noqa: SLF001
        control_monotonic_ns=control_ns,
        observed_monotonic_ns=(
            control_ns + publisher._PUBLISHER_MAX_AGE_NS  # noqa: SLF001
        ),
    ) == publisher._PUBLISHER_MAX_AGE_NS  # noqa: SLF001


def test_publisher_age_budget_rejects_before_send() -> None:
    control_ns = 1_000_000_000
    with pytest.raises(RuntimeError, match="refusing stale.*before ZMQ send"):
        publisher._require_publisher_age_budget(  # noqa: SLF001
            control_monotonic_ns=control_ns,
            observed_monotonic_ns=(
                control_ns
                + publisher._PUBLISHER_MAX_AGE_NS  # noqa: SLF001
                + 1
            ),
        )


def test_read_only_diagnostic_age_budget_accepts_100ms_only_explicitly() -> None:
    control_ns = 1_000_000_000
    observed_ns = control_ns + 100_000_000
    with pytest.raises(RuntimeError, match="refusing stale"):
        publisher._require_publisher_age_budget(  # noqa: SLF001
            control_monotonic_ns=control_ns,
            observed_monotonic_ns=observed_ns,
        )
    assert publisher._require_publisher_age_budget(  # noqa: SLF001
        control_monotonic_ns=control_ns,
        observed_monotonic_ns=observed_ns,
        maximum_age_ns=100_000_000,
    ) == 100_000_000


def test_publisher_stop_state_preserves_first_signal_and_terminal_contract() -> None:
    stop = publisher.PublisherStopState()
    stop.handle(signal.SIGTERM, None)
    stop.handle(signal.SIGINT, None)

    assert stop.evidence() == {
        "stop_requested": True,
        "stop_signal_number": int(signal.SIGTERM),
        "stop_signal_name": "SIGTERM",
    }
    assert publisher._terminal_outcome(  # noqa: SLF001
        published_packets=17,
        requested_packets=100,
        stop_requested=True,
    ) == ("session_stopped", "signal_requested")
    assert publisher._terminal_outcome(  # noqa: SLF001
        published_packets=100,
        requested_packets=100,
        stop_requested=True,
    ) == ("session_complete", "packet_target_reached")


def test_publisher_stop_interrupts_empty_capture_wait() -> None:
    controlled = _ControlledCaptureWorker()
    capture = publisher.BackgroundRawCapture(
        controlled,  # type: ignore[arg-type]
        deadline_ns=time.monotonic_ns() + 5_000_000_000,
        capacity=4,
    )
    stop_event = threading.Event()
    stop_event.set()

    with pytest.raises(publisher.PublisherStopRequested):
        capture.next_frame(
            deadline_ns=time.monotonic_ns() + 1_000_000_000,
            stop_event=stop_event,
        )


def test_measured_selector_uses_unique_real_frames_without_interpolation() -> None:
    pose0 = {"pose": 0}
    pose1 = {"pose": 1}
    pose2 = {"pose": 2}
    pose3 = {"pose": 3}
    pose4 = {"pose": 4}
    origin = _delivery(
        0,
        100_000_000,
        body_timestamp_ns=1_000,
        body_sequence=10,
        body_poses=pose0,
    )
    selector = publisher.MeasuredFrameTickSelector(origin)

    sample0 = selector.take_origin()
    samples1 = selector.push(
        [
            _delivery(
                1,
                111_000_000,
                body_timestamp_ns=1_100,
                body_sequence=11,
                body_poses=pose1,
            ),
            _delivery(
                2,
                122_000_000,
                body_timestamp_ns=1_200,
                body_sequence=12,
                body_poses=pose2,
            ),
        ]
    )
    samples2 = selector.push(
        [
            _delivery(
                3,
                133_000_000,
                body_timestamp_ns=1_300,
                body_sequence=13,
                body_poses=pose3,
            ),
            _delivery(
                4,
                144_000_000,
                body_timestamp_ns=1_400,
                body_sequence=14,
                body_poses=pose4,
            ),
        ]
    )

    assert len(samples1) == 1
    assert len(samples2) == 1
    sample1 = samples1[0]
    sample2 = samples2[0]
    assert [
        sample0["reference_monotonic_ns"],
        sample1["reference_monotonic_ns"],
        sample2["reference_monotonic_ns"],
    ] == [100_000_000, 120_000_000, 140_000_000]
    assert [
        sample0["selected_raw_frame_index"],
        sample1["selected_raw_frame_index"],
        sample2["selected_raw_frame_index"],
    ] == [0, 1, 3]
    assert sample0["body_poses"] is pose0
    assert sample1["body_poses"] is pose1
    assert sample2["body_poses"] is pose3
    assert sample2["raw_bracket_indices"] == [3, 3]
    assert sample2["raw_interpolation_alpha"] == 0.0
    assert sample2["selected_capture_delta_ns"] == 22_000_000
    assert sample2["selected_body_source_timestamp_delta_ns"] == 200
    assert sample2["selected_source_delta_ns"] == 200
    assert sample2["selected_body_sequence_delta_from_previous_selected"] == 2
    assert sample2["selected_capture_age_at_tick_ns"] == 7_000_000
    assert sample2["selected_source_age_clock"] == "local_capture_monotonic"
    assert sample2["superseded_raw_frame_indices"] == [2]
    assert sample2["positions_repeated_or_synthesized"] is False
    assert sample2["source_pose_timestamp_relabelled"] is False
    stats = selector.stats()
    assert stats["selected_raw_frame_indices"] == [0, 1, 3]
    assert stats["superseded_raw_frame_indices"] == [2]
    assert stats["pending_raw_frame_indices"] == [4]


def test_measured_selector_brackets_one_jittered_tick_without_repeat() -> None:
    selector = publisher.MeasuredFrameTickSelector(
        _delivery(
            0,
            100_000_000,
            body_timestamp_ns=1_000,
            body_sequence=10,
            body_poses=_xr24_pose_frame(0.0),
        )
    )
    selector.take_origin()

    samples = selector.push(
        [
            _delivery(
                1,
                125_000_000,
                body_timestamp_ns=26_000_000,
                body_sequence=11,
                body_poses=_xr24_pose_frame(1.0),
            )
        ]
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample["reference_monotonic_ns"] == 120_000_000
    assert sample["capture_monotonic_ns"] == 125_000_000
    assert sample["raw_bracket_indices"] == [0, 1]
    assert sample["raw_interpolation_alpha"] == pytest.approx(0.8)
    assert sample["body_poses"][0][0] == pytest.approx(0.8)
    assert sample["selected_raw_frame_index"] is None
    assert sample["positions_interpolated_from_measured_xr24"] is True
    assert sample["positions_repeated_or_synthesized"] is False
    assert sample["interpolation_ready_delay_ns"] == 5_000_000
    assert selector.stats()["pending_raw_frame_indices"] == [1]


def test_measured_selector_rejects_unbounded_capture_dropout() -> None:
    selector = publisher.MeasuredFrameTickSelector(
        _delivery(
            0,
            100_000_000,
            body_timestamp_ns=1_000,
            body_sequence=10,
            body_poses=_xr24_pose_frame(0.0),
        )
    )
    selector.take_origin()

    with pytest.raises(RuntimeError, match="capture bracket exceeds bounded"):
        selector.push(
            [
                _delivery(
                    1,
                    180_000_001,
                    body_timestamp_ns=30_001_000,
                    body_sequence=11,
                    body_poses=_xr24_pose_frame(1.0),
                )
            ]
        )


def test_jittered_measured_selector_builds_100_causal_packets() -> None:
    origin_capture_ns = 1_000_000_000
    origin_body_ns = 2_000_000_000
    selector = publisher.MeasuredFrameTickSelector(
        _delivery(
            0,
            origin_capture_ns,
            body_timestamp_ns=origin_body_ns,
            body_sequence=100,
            body_poses=_xr24_pose_frame(0.0),
        )
    )
    samples = [selector.take_origin()]
    capture_ns = origin_capture_ns
    body_ns = origin_body_ns
    for raw_index in range(1, 111):
        capture_ns += 25_000_000 if raw_index % 2 else 15_000_000
        body_ns += 20_000_000
        samples.extend(
            selector.push(
                [
                    _delivery(
                        raw_index,
                        capture_ns,
                        body_timestamp_ns=body_ns,
                        body_sequence=100 + raw_index,
                        body_poses=_xr24_pose_frame(float(raw_index)),
                    )
                ]
            )
        )
    samples = samples[:110]
    assert len(samples) == 110
    assert any(
        sample["positions_interpolated_from_measured_xr24"]
        for sample in samples
    )
    assert all(
        sample["positions_repeated_or_synthesized"] is False
        for sample in samples
    )
    assert [sample["reference_monotonic_ns"] for sample in samples] == [
        origin_capture_ns + index * 20_000_000 for index in range(110)
    ]

    producer = CausalHistorySemanticProducer(
        source_session_id="jittered-100-packet-read-only-test"
    )
    packets = []
    for index, sample in enumerate(samples):
        reference_ns = int(sample["reference_monotonic_ns"])
        capture_ns = int(sample["capture_monotonic_ns"])
        rolling = {
            "kind": ROLLING_SOMA_KIND,
            "promotion_eligible": False,
            "source_frame_index": index,
            "reference_monotonic_ns": reference_ns,
            "capture_monotonic_ns": capture_ns,
            "compute_finished_monotonic_ns": max(reference_ns, capture_ns) + 1,
            "producer_sha256": "d" * 64,
            "joint_pos_il29": [
                0.1 + index * (joint + 1) * 1.0e-4
                for joint in range(29)
            ],
            "body_term": {
                "source_frame_index": index,
                "reference_monotonic_ns": reference_ns,
                "capture_monotonic_ns": capture_ns,
                "vr_3point_local_target": [0.1] * 9,
                "vr_3point_local_orn_target": [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
                * 3,
                "reference_anchor_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
        packet = producer.push(rolling)
        if packet is not None:
            validate_causal_history_packet(packet)
            packets.append(packet)

    assert len(packets) == 100
    assert all(
        packet["positions_repeated_or_synthesized"] is False
        for packet in packets
    )
    assert publisher._AUTHORIZATION["robot_channel_opened"] is False  # noqa: SLF001
    assert publisher._AUTHORIZATION["actuation_authorized"] is False  # noqa: SLF001


def test_measured_selector_rejects_large_body_source_timestamp_jump() -> None:
    selector = publisher.MeasuredFrameTickSelector(
        _delivery(
            0,
            100_000_000,
            body_timestamp_ns=100_000_000,
            body_sequence=10,
        )
    )
    selector.take_origin()

    with pytest.raises(RuntimeError, match="body source timestamp delta exceeds"):
        selector.push(
            [
                _delivery(
                    1,
                    111_000_000,
                    body_timestamp_ns=180_000_001,
                    body_sequence=11,
                ),
                _delivery(
                    2,
                    122_000_000,
                    body_timestamp_ns=170_000_000,
                    body_sequence=12,
                ),
            ]
        )


def test_background_capture_counts_body_source_sequence_gaps() -> None:
    controlled = _ControlledCaptureWorker()
    capture = publisher.BackgroundRawCapture(
        controlled,  # type: ignore[arg-type]
        deadline_ns=time.monotonic_ns() + 5_000_000_000,
        capacity=4,
    )
    capture.start()
    try:
        controlled.frames.put(_capture_frame(0, body_sequence=10))
        capture.next_frame(deadline_ns=time.monotonic_ns() + 1_000_000_000)
        controlled.frames.put(_capture_frame(1, body_sequence=13))
        second = capture.next_frame(
            deadline_ns=time.monotonic_ns() + 1_000_000_000
        )
        assert second["body_sample_sequence_delta"] == 3
        assert second["body_source_sequence_gap_count"] == 2
        assert capture.stats()["source_sequence_gap_count"] == 2
    finally:
        capture.request_stop()
        controlled.frames.put(_capture_frame(2, body_sequence=14))
        capture.join()
