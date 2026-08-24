"""Publish real causal XR24/SOMA reference terms for read-only C++ shadow.

This process contains no Unitree or DDS imports.  It receives hardened PICO
frames through the CPython-3.10 capture worker, runs the pinned exact IK24 SOMA
solver in its Python-3.12 environment, and publishes only robot-independent
reference observations.  Timestamps are measured locally and never relabelled.
"""

from __future__ import annotations

import argparse
from collections import deque
import gc
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

import zmq

from gear_sonic.utils.g1_23dof_semantic_reference import SOURCE_SAMPLE_PERIOD_NS
from gear_sonic.utils.g1_23dof_xr24_soma_adapter import (
    _interpolate_pose as _interpolate_xr24_pose,
)
from gear_sonic.utils.g1_23dof_xr24_soma_stream import (
    CausalHistorySemanticProducer,
    ExperimentalSomaRollingRetargeter,
    PinnedSomaRollingRetargeter,
    causal_history_reference_terms,
    validate_causal_history_packet,
)

_WORKER_PREFIX = "G1_TRUE23_XR24\t"
_PUBLISHER_PERFORMANCE_TARGET_NS = 40_000_000
_PUBLISHER_MAX_AGE_NS = 80_000_000
_DOWNSTREAM_HIGH_LEVEL_FRESHNESS_NS = 100_000_000
_MAX_SELECTED_BODY_SOURCE_TIMESTAMP_DELTA_NS = 80_000_000
_MAX_INTERPOLATION_CAPTURE_SPAN_NS = 80_000_000
_MAX_INTERPOLATION_LEFT_BRACKET_AGE_NS = 80_000_000
_MEASURED_INTERPOLATION_CONTRACT = "linear_xyz_slerp_quaternion_between_consecutive_measured_xr24_v1"
_NEUTRAL_STANDING_REJECTION_PREFIX = "XR24 acquisition pose is not neutral standing:"
_XRT_MODULE_RELATIVE_DIR = Path("external_dependencies") / "XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
_AUTHORIZATION = {
    "read_only": True,
    "dds_opened": False,
    "robot_channel_opened": False,
    "actuation_authorized": False,
    "robot_commands_published": False,
}


class PublisherStopRequested(InterruptedError):
    """Internal control-flow exception for graceful publisher termination."""


class PublisherStopState:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.signal_number: int | None = None

    def handle(self, signum: int, _frame: Any) -> None:
        if not self.event.is_set():
            self.signal_number = int(signum)
            self.event.set()

    def evidence(self) -> dict[str, Any]:
        number = self.signal_number
        return {
            "stop_requested": self.event.is_set(),
            "stop_signal_number": number,
            "stop_signal_name": (None if number is None else signal.Signals(number).name),
        }


def _canonical_bytes(value: Any, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + suffix).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_xrt_module_dir(workspace: Path) -> Path:
    """Resolve the XRT binding from the selected workspace, never another clone."""

    return (workspace.resolve() / _XRT_MODULE_RELATIVE_DIR).resolve()


def _resolve_xrt_module_binary(xrt_module_dir: Path) -> Path:
    module_dir = xrt_module_dir.resolve()
    if not module_dir.is_dir():
        raise FileNotFoundError(f"XRT module directory missing: {module_dir}")
    candidates = sorted(path.resolve() for path in module_dir.glob("xrobotoolkit_sdk*.so") if path.is_file())
    if len(candidates) != 1:
        raise RuntimeError(
            "XRT module directory must contain exactly one "
            f"xrobotoolkit_sdk*.so; found {len(candidates)} in {module_dir}"
        )
    binary = candidates[0]
    if binary.parent != module_dir:
        raise RuntimeError(f"XRT module binary resolves outside requested directory: {binary}")
    return binary


class EvidenceLog:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._stream = os.fdopen(descriptor, "wb")

    def write(self, value: dict[str, Any], *, durable: bool = True) -> None:
        self._stream.write(_canonical_bytes(value))
        if durable:
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        self._stream.close()


class RawCaptureWorker:
    def __init__(
        self,
        *,
        python: Path,
        script: Path,
        workspace: Path,
        xrt_module_dir: Path,
        expected_xrt_module_path: Path,
        expected_xrt_module_sha256: str,
        service_binary: Path,
        pico_client_apk_sha256: str,
        frame_timeout_s: float,
        session_id: str,
        request_driven: bool = True,
    ):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(workspace),
                str(xrt_module_dir),
                environment.get("PYTHONPATH", ""),
            )
        )
        command = [
            str(python),
            str(script),
            "--service-binary",
            str(service_binary),
            "--expected-xrt-module-path",
            str(expected_xrt_module_path),
            "--expected-xrt-module-sha256",
            expected_xrt_module_sha256,
            "--pico-client-apk-sha256",
            pico_client_apk_sha256,
            "--frame-timeout-s",
            str(frame_timeout_s),
            "--session-id",
            session_id,
        ]
        if request_driven:
            command.append("--request-driven")
        self._request_driven = request_driven
        self._process = subprocess.Popen(
            command,
            cwd=workspace,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self.identity: dict[str, Any] | None = None
        self.xrt_binding: dict[str, str] | None = None
        self._expected_xrt_binding = {
            "path": str(expected_xrt_module_path.resolve()),
            "sha256": expected_xrt_module_sha256,
        }

    def next_frame(self) -> dict[str, Any]:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("capture worker pipes are unavailable")
        if self._process.poll() is not None:
            raise RuntimeError(f"capture worker exited with {self._process.returncode}")
        if self._request_driven:
            self._process.stdin.write("NEXT\n")
            self._process.stdin.flush()
        while True:
            line = self._process.stdout.readline()
            if line == "":
                raise RuntimeError(f"capture worker closed stdout with {self._process.poll()}")
            if not line.startswith(_WORKER_PREFIX):
                print(f"[XRT] {line.rstrip()}", file=sys.stderr)
                continue
            event = json.loads(line[len(_WORKER_PREFIX) :])
            kind = event.get("event")
            if kind == "xrt_binding_verified":
                binding = event.get("binding")
                if binding != self._expected_xrt_binding:
                    raise RuntimeError("capture worker reported an unexpected XRT binding")
                if self.xrt_binding is not None and binding != self.xrt_binding:
                    raise RuntimeError("capture worker XRT binding changed live")
                self.xrt_binding = dict(binding)
                print(
                    f"[XRT] verified imported binding {binding['path']} sha256={binding['sha256']}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if kind == "identity":
                if self.xrt_binding is None:
                    raise RuntimeError("capture worker reported identity before XRT binding")
                identity = event.get("identity")
                if not isinstance(identity, dict):
                    raise RuntimeError("capture worker identity is invalid")
                self.identity = identity
                continue
            if kind == "frame":
                if self.xrt_binding is None:
                    raise RuntimeError("capture worker reported frame before XRT binding")
                frame = event.get("frame")
                if not isinstance(frame, dict):
                    raise RuntimeError("capture worker frame is invalid")
                return frame
            if kind in {"failure", "timeout"}:
                raise RuntimeError(f"capture worker {kind}: {event.get('failure', 'unknown')}")
            raise RuntimeError(f"unknown capture worker event: {kind!r}")

    def start_continuous(self) -> None:
        if not self._request_driven:
            raise RuntimeError("capture worker is already continuous")
        if self._process.stdin is None or self._process.poll() is not None:
            raise RuntimeError("capture worker is unavailable")
        self._process.stdin.write("STREAM\n")
        self._process.stdin.flush()
        self._request_driven = False

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                if self._request_driven and self._process.stdin is not None:
                    self._process.stdin.write("STOP\n")
                    self._process.stdin.flush()
                else:
                    self._process.terminate()
                self._process.wait(timeout=3.0)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self._process.terminate()
                try:
                    self._process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=3.0)
        if self._process.stdin is not None:
            self._process.stdin.close()
        if self._process.stdout is not None:
            self._process.stdout.close()


def _parser() -> argparse.ArgumentParser:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Real PICO -> pinned SOMA -> causal read-only ZMQ shadow")
    parser.add_argument("--bind", default="tcp://127.0.0.1:5557")
    parser.add_argument("--packets", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--subscriber-warmup-s", type=float, default=2.0)
    parser.add_argument("--frame-timeout-s", type=float, default=2.0)
    parser.add_argument("--capture-queue-capacity", type=int, default=64)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--pico-client-apk-sha256", required=True)
    parser.add_argument(
        "--soma-source-root",
        type=Path,
        default=Path("/root/.cache/g1_true23_soma/source"),
    )
    parser.add_argument(
        "--experimental-solver-iterations",
        type=int,
        choices=(12, 16),
        help=(
            "Read-only non-promotable SOMA timing experiment. Changes only "
            "IK iteration count; never authorizes robot actuation."
        ),
    )
    parser.add_argument(
        "--read-only-diagnostic-100ms-max-age",
        action="store_true",
        help=(
            "Allow a 100 ms high-level reference age only for read-only "
            "diagnostics. This never authorizes robot actuation."
        ),
    )
    parser.add_argument("--capture-python", type=Path, default=Path("/usr/bin/python3"))
    parser.add_argument(
        "--xrt-module-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the CPython-3.10 xrobotoolkit_sdk binding. "
            "Defaults to external_dependencies in --workspace."
        ),
    )
    parser.add_argument(
        "--service-binary",
        type=Path,
        default=Path("/opt/apps/roboticsservice/RoboticsServiceProcess"),
    )
    parser.add_argument("--workspace", type=Path, default=workspace)
    return parser


def _new_worker(
    args: argparse.Namespace,
    session_id: str,
    *,
    expected_xrt_module_path: Path,
    expected_xrt_module_sha256: str,
) -> RawCaptureWorker:
    return RawCaptureWorker(
        python=args.capture_python.resolve(),
        script=(args.workspace.resolve() / "gear_sonic/scripts/stream_g1_23dof_pico_raw_worker.py"),
        workspace=args.workspace.resolve(),
        xrt_module_dir=args.xrt_module_dir.resolve(),
        expected_xrt_module_path=expected_xrt_module_path,
        expected_xrt_module_sha256=expected_xrt_module_sha256,
        service_binary=args.service_binary.resolve(),
        pico_client_apk_sha256=args.pico_client_apk_sha256,
        frame_timeout_s=args.frame_timeout_s,
        session_id=session_id,
    )


def _next_frame_until(worker: RawCaptureWorker, *, deadline_ns: int) -> dict[str, Any]:
    warned = False
    while time.monotonic_ns() < deadline_ns:
        try:
            return worker.next_frame()
        except RuntimeError as exc:
            if not str(exc).startswith("capture worker timeout:"):
                raise
            if not warned:
                print("[WAIT] PICO tracking frame unavailable; retrying", flush=True)
                warned = True
    raise TimeoutError("PICO tracking did not resume before session deadline")


def _prime_until_neutral_standing(
    rolling: Any,
    worker: RawCaptureWorker,
    *,
    initial_frame: dict[str, Any],
    deadline_ns: int,
    on_rejection: Any = None,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Wait for a measured upright XR24 frame without weakening the gate."""

    frame = initial_frame
    neutral_hold: list[list[list[float]]] = []
    rejection_count = 0
    while True:
        neutral_hold.append(frame["body_poses"])
        if len(neutral_hold) < 10:
            frame = _next_frame_until(worker, deadline_ns=deadline_ns)
            continue
        try:
            report = rolling.prime(neutral_hold)
        except ValueError as exc:
            reason = str(exc)
            if not reason.startswith(_NEUTRAL_STANDING_REJECTION_PREFIX):
                if not reason.startswith("rolling SOMA prime requires") and not reason.startswith(
                    "XR24 SOMA calibration"
                ):
                    raise
            rejection_count += 1
            if on_rejection is not None and (rejection_count == 1 or rejection_count % 50 == 0):
                on_rejection(rejection_count, frame, reason)
            neutral_hold = []
            frame = _next_frame_until(worker, deadline_ns=deadline_ns)
            continue
        return frame, report, rejection_count


class BackgroundRawCapture:
    """Capture every advancing XR24 frame while exact SOMA runs.

    The queue is FIFO and bounded.  Frames are never overwritten or dropped;
    overflow is a terminal failure so a slow solver cannot silently change the
    measured source stream.
    """

    def __init__(
        self,
        worker: RawCaptureWorker,
        *,
        deadline_ns: int,
        capacity: int,
        initial_previous_frame: dict[str, Any] | None = None,
    ):
        if capacity <= 0:
            raise ValueError("capture queue capacity must be positive")
        self._worker = worker
        self._deadline_ns = deadline_ns
        self._capacity = capacity
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._failure: BaseException | None = None
        self._captured_frames = 0
        self._max_queue_depth = 0
        self._capture_wait_total_ns = 0
        self._capture_wait_max_ns = 0
        self._capture_ipc_total_ns = 0
        self._capture_ipc_max_ns = 0
        self._source_sequence_gap_count = 0
        self._max_source_sequence_delta = 0
        self._mailbox_batch_count = 0
        self._mailbox_coalesced_prior_frames = 0
        self._last_frame_index: int | None = None
        self._last_capture_ns: int | None = None
        self._last_body_timestamp_ns: int | None = None
        self._last_body_sequence: int | None = None
        if initial_previous_frame is not None:
            self._set_progress_origin(initial_previous_frame)
        self._thread = threading.Thread(
            target=self._run,
            name="g1-true23-xr24-capture",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _set_failure(self, failure: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = failure

    def _get_failure(self) -> BaseException | None:
        with self._lock:
            return self._failure

    def _set_progress_origin(self, frame: dict[str, Any]) -> None:
        self._last_frame_index = int(frame["frame_index"])
        self._last_capture_ns = int(frame["capture_monotonic_ns"])
        self._last_body_timestamp_ns = int(frame["body_sample_timestamp_ns"])
        self._last_body_sequence = int(frame["body_sample_sequence"])

    def _validate_progress(self, frame: dict[str, Any]) -> tuple[int, int]:
        frame_index = int(frame["frame_index"])
        capture_ns = int(frame["capture_monotonic_ns"])
        body_timestamp_ns = int(frame["body_sample_timestamp_ns"])
        body_sequence = int(frame["body_sample_sequence"])
        if self._last_frame_index is None:
            frame_index_delta = 1
            body_sequence_delta = 1
        else:
            frame_index_delta = frame_index - self._last_frame_index
            body_sequence_delta = body_sequence - int(self._last_body_sequence)
            if frame_index_delta != 1:
                raise RuntimeError(
                    f"background XR24 capture frame index is not contiguous: delta={frame_index_delta}"
                )
            if (
                capture_ns <= int(self._last_capture_ns)
                or body_timestamp_ns <= int(self._last_body_timestamp_ns)
                or body_sequence_delta <= 0
            ):
                raise RuntimeError("background XR24 capture source time/sequence regressed")
        self._set_progress_origin(frame)
        return frame_index_delta, body_sequence_delta

    def _run(self) -> None:
        while not self._stop.is_set():
            request_started_ns = time.monotonic_ns()
            if request_started_ns >= self._deadline_ns:
                self._set_failure(TimeoutError("background XR24 capture reached session deadline"))
                return
            try:
                frame = _next_frame_until(
                    self._worker,
                    deadline_ns=self._deadline_ns,
                )
            except BaseException as exc:
                if not self._stop.is_set():
                    self._set_failure(exc)
                return
            received_ns = time.monotonic_ns()
            capture_ns = int(frame["capture_monotonic_ns"])
            wait_ns = received_ns - request_started_ns
            ipc_ns = received_ns - capture_ns
            if wait_ns < 0 or ipc_ns < 0:
                self._set_failure(RuntimeError("background XR24 capture timing is non-causal"))
                return
            try:
                frame_index_delta, body_sequence_delta = self._validate_progress(frame)
            except BaseException as exc:
                self._set_failure(exc)
                return
            delivery = {
                "frame": frame,
                "request_started_monotonic_ns": request_started_ns,
                "received_monotonic_ns": received_ns,
                "capture_wait_duration_ns": wait_ns,
                "capture_ipc_duration_ns": ipc_ns,
                "capture_frame_index_delta": frame_index_delta,
                "body_sample_sequence_delta": body_sequence_delta,
                "body_source_sequence_gap_count": max(0, body_sequence_delta - 1),
                "queue_depth_before_enqueue": self._queue.qsize(),
            }
            try:
                self._queue.put_nowait(delivery)
            except queue.Full:
                self._set_failure(
                    RuntimeError(
                        "background XR24 capture queue overflow; "
                        f"capacity={self._capacity} frame_index={frame.get('frame_index')}"
                    )
                )
                return
            depth = self._queue.qsize()
            with self._lock:
                self._captured_frames += 1
                self._max_queue_depth = max(self._max_queue_depth, depth)
                self._capture_wait_total_ns += wait_ns
                self._capture_wait_max_ns = max(self._capture_wait_max_ns, wait_ns)
                self._capture_ipc_total_ns += ipc_ns
                self._capture_ipc_max_ns = max(self._capture_ipc_max_ns, ipc_ns)
                self._source_sequence_gap_count += max(0, body_sequence_delta - 1)
                self._max_source_sequence_delta = max(self._max_source_sequence_delta, body_sequence_delta)

    def next_frame(
        self,
        *,
        deadline_ns: int,
        stop_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        while True:
            if stop_event is not None and stop_event.is_set():
                raise PublisherStopRequested("publisher stop requested")
            failure = self._get_failure()
            if failure is not None:
                raise RuntimeError(f"background XR24 capture failed: {failure}") from failure
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                raise TimeoutError("causal PICO publisher session timed out")
            try:
                delivery = self._queue.get(timeout=min(0.05, remaining_ns / 1e9))
            except queue.Empty:
                continue
            failure = self._get_failure()
            if failure is not None:
                raise RuntimeError(f"background XR24 capture failed: {failure}") from failure
            delivery["queue_depth_after_dequeue"] = self._queue.qsize()
            return delivery

    def next_latest_batch(
        self,
        *,
        deadline_ns: int,
        stop_event: threading.Event | None = None,
    ) -> list[dict[str, Any]]:
        """Drain the FIFO once and expose the newest causal raw frame.

        Every drained frame remains available to the caller for ordered source
        validation/selection.  Older frames are only coalesced for compute
        scheduling; none are silently overwritten.
        """

        batch = [
            self.next_frame(
                deadline_ns=deadline_ns,
                stop_event=stop_event,
            )
        ]
        while True:
            self.raise_if_failed()
            try:
                delivery = self._queue.get_nowait()
            except queue.Empty:
                break
            delivery["queue_depth_after_dequeue"] = self._queue.qsize()
            batch.append(delivery)
        coalesced = len(batch) - 1
        for delivery in batch:
            delivery["mailbox_batch_size"] = len(batch)
            delivery["mailbox_coalesced_prior_frames"] = coalesced
        with self._lock:
            self._mailbox_batch_count += 1
            self._mailbox_coalesced_prior_frames += coalesced
        return batch

    def raise_if_failed(self) -> None:
        failure = self._get_failure()
        if failure is not None:
            raise RuntimeError(f"background XR24 capture failed: {failure}") from failure

    def request_stop(self) -> None:
        self._stop.set()

    def join(self, timeout_s: float = 4.0) -> None:
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("background XR24 capture thread did not stop")

    def stats(self) -> dict[str, Any]:
        with self._lock:
            count = self._captured_frames
            with self._queue.mutex:
                queued_raw_frame_indices = [
                    int(delivery["frame"]["frame_index"]) for delivery in self._queue.queue
                ]
            return {
                "queue_capacity": self._capacity,
                "captured_frames": count,
                "max_queue_depth": self._max_queue_depth,
                "queue_depth_at_snapshot": self._queue.qsize(),
                "queued_raw_frame_indices": queued_raw_frame_indices,
                "capture_wait_mean_ns": (self._capture_wait_total_ns / count if count else None),
                "capture_wait_max_ns": (self._capture_wait_max_ns if count else None),
                "capture_ipc_mean_ns": (self._capture_ipc_total_ns / count if count else None),
                "capture_ipc_max_ns": (self._capture_ipc_max_ns if count else None),
                "source_sequence_gap_count": self._source_sequence_gap_count,
                "max_source_sequence_delta": self._max_source_sequence_delta,
                "mailbox_batch_count": self._mailbox_batch_count,
                "mailbox_coalesced_prior_frames": (self._mailbox_coalesced_prior_frames),
                "queue_overflowed": (
                    isinstance(self._failure, RuntimeError) and "queue overflow" in str(self._failure)
                ),
                "failure": None if self._failure is None else str(self._failure),
                "frames_silently_dropped": 0,
            }


class MeasuredFrameTickSelector:
    """Select measured XR24 poses on an exact 20 ms control clock.

    Source capture time and control tick time remain separate.  A selected pose
    is normally copied from one real raw frame.  When ordinary source jitter
    leaves no unused frame at/before one tick, exactly that tick may be derived
    from the consecutive real frames bracketing it.  All 24 measured roles are
    linearly interpolated in position and SLERPed in orientation.  Bracket
    spans are bounded and explicit; frames are never repeated or fabricated.
    """

    _POSE_SELECTION_CONTRACT = "unique_or_bounded_bracketed_measured_xr24_at_tick_v1"

    def __init__(self, origin_delivery: dict[str, Any]):
        self._pending: deque[dict[str, Any]] = deque()
        self._control_index = 0
        self._next_tick_ns: int | None = None
        self._last_selected_raw_index: int | None = None
        self._last_selected_capture_ns: int | None = None
        self._last_selected_body_timestamp_ns: int | None = None
        self._last_selected_body_sequence: int | None = None
        self._last_selected_delivery: dict[str, Any] | None = None
        self._last_ingested_raw_index: int | None = None
        self._last_ingested_capture_ns: int | None = None
        self._selected_frames = 0
        self._interpolated_frames = 0
        self._superseded_frames = 0
        self._selected_raw_indices: list[int] = []
        self._superseded_raw_indices: list[int] = []
        self._selected_capture_deltas_ns: list[int] = []
        self._selected_body_source_timestamp_deltas_ns: list[int] = []
        self._interpolation_capture_spans_ns: list[int] = []
        self._interpolation_body_source_spans_ns: list[int] = []
        self._interpolation_ready_delays_ns: list[int] = []
        self._interpolation_raw_brackets: list[list[int]] = []
        self._max_capture_age_at_tick_ns = 0
        self._origin_sample: dict[str, Any] | None = self._select_origin(origin_delivery)

    def _ingest(self, delivery: dict[str, Any]) -> None:
        frame = delivery["frame"]
        raw_index = int(frame["frame_index"])
        capture_ns = int(frame["capture_monotonic_ns"])
        if self._last_ingested_raw_index is not None and (
            raw_index != self._last_ingested_raw_index + 1 or capture_ns <= int(self._last_ingested_capture_ns)
        ):
            raise RuntimeError("measured XR24 mailbox input regressed or skipped")
        self._last_ingested_raw_index = raw_index
        self._last_ingested_capture_ns = capture_ns
        self._pending.append(delivery)

    def _sample(
        self,
        delivery: dict[str, Any],
        *,
        tick_ns: int,
        superseded: list[dict[str, Any]],
    ) -> dict[str, Any]:
        frame = delivery["frame"]
        raw_index = int(frame["frame_index"])
        capture_ns = int(frame["capture_monotonic_ns"])
        body_timestamp_ns = int(frame["body_sample_timestamp_ns"])
        body_sequence = int(frame["body_sample_sequence"])
        if capture_ns > tick_ns:
            raise RuntimeError("selected XR24 pose is newer than its control tick")
        if self._last_selected_raw_index is not None and (raw_index <= self._last_selected_raw_index):
            raise RuntimeError("measured XR24 pose was repeated or reordered")
        capture_delta_ns = (
            None if self._last_selected_capture_ns is None else capture_ns - self._last_selected_capture_ns
        )
        body_source_timestamp_delta_ns = (
            None
            if self._last_selected_body_timestamp_ns is None
            else body_timestamp_ns - self._last_selected_body_timestamp_ns
        )
        body_sequence_delta = (
            None
            if self._last_selected_body_sequence is None
            else body_sequence - self._last_selected_body_sequence
        )
        if capture_delta_ns is not None and capture_delta_ns <= 0:
            raise RuntimeError("selected XR24 capture timestamp did not advance")
        if body_source_timestamp_delta_ns is not None and body_source_timestamp_delta_ns <= 0:
            raise RuntimeError("selected XR24 body source timestamp did not advance")
        if (
            body_source_timestamp_delta_ns is not None
            and body_source_timestamp_delta_ns > _MAX_SELECTED_BODY_SOURCE_TIMESTAMP_DELTA_NS
        ):
            raise RuntimeError(
                "selected XR24 body source timestamp delta exceeds limit: "
                f"delta_ns={body_source_timestamp_delta_ns} "
                "limit_ns="
                f"{_MAX_SELECTED_BODY_SOURCE_TIMESTAMP_DELTA_NS}"
            )
        if body_sequence_delta is not None and body_sequence_delta <= 0:
            raise RuntimeError("selected XR24 body source sequence did not advance")
        capture_age_ns = tick_ns - capture_ns
        if capture_age_ns > SOURCE_SAMPLE_PERIOD_NS:
            raise RuntimeError(
                "selected XR24 pose exceeds one control-period source-age limit: "
                f"age_ns={capture_age_ns} limit_ns={SOURCE_SAMPLE_PERIOD_NS}"
            )
        superseded_indices = [int(item["frame"]["frame_index"]) for item in superseded]
        self._last_selected_raw_index = raw_index
        self._last_selected_capture_ns = capture_ns
        self._last_selected_body_timestamp_ns = body_timestamp_ns
        self._last_selected_body_sequence = body_sequence
        self._last_selected_delivery = delivery
        self._selected_frames += 1
        self._superseded_frames += len(superseded_indices)
        self._selected_raw_indices.append(raw_index)
        self._superseded_raw_indices.extend(superseded_indices)
        self._max_capture_age_at_tick_ns = max(self._max_capture_age_at_tick_ns, capture_age_ns)
        if capture_delta_ns is not None:
            self._selected_capture_deltas_ns.append(capture_delta_ns)
        if body_source_timestamp_delta_ns is not None:
            self._selected_body_source_timestamp_deltas_ns.append(body_source_timestamp_delta_ns)
        sample = {
            "source_frame_index": self._control_index,
            "reference_monotonic_ns": tick_ns,
            "capture_monotonic_ns": capture_ns,
            "raw_bracket_indices": [raw_index, raw_index],
            "raw_interpolation_alpha": 0.0,
            "body_poses": frame["body_poses"],
            "pose_selection_contract": self._POSE_SELECTION_CONTRACT,
            "selected_raw_frame_index": raw_index,
            "selected_body_sample_sequence": body_sequence,
            "selected_body_sample_timestamp_ns": body_timestamp_ns,
            "selected_capture_delta_ns": capture_delta_ns,
            "selected_body_source_timestamp_delta_ns": (body_source_timestamp_delta_ns),
            "selected_source_delta_ns": body_source_timestamp_delta_ns,
            "selected_body_sequence_delta_from_previous_selected": (body_sequence_delta),
            "selected_capture_age_at_tick_ns": capture_age_ns,
            "selected_source_age_at_tick_ns": capture_age_ns,
            "selected_source_age_clock": "local_capture_monotonic",
            "selected_capture_wait_duration_ns": int(delivery["capture_wait_duration_ns"]),
            "selected_capture_ipc_duration_ns": int(delivery["capture_ipc_duration_ns"]),
            "selected_capture_queue_depth_before_enqueue": int(delivery["queue_depth_before_enqueue"]),
            "selected_capture_queue_depth_after_dequeue": int(delivery["queue_depth_after_dequeue"]),
            "selected_capture_frame_index_delta": int(delivery["capture_frame_index_delta"]),
            "selected_raw_body_sample_sequence_delta": int(delivery["body_sample_sequence_delta"]),
            "selected_body_source_sequence_gap_count": int(delivery["body_source_sequence_gap_count"]),
            "selected_mailbox_batch_size": int(delivery.get("mailbox_batch_size", 1)),
            "selected_mailbox_coalesced_prior_frames": int(delivery.get("mailbox_coalesced_prior_frames", 0)),
            "superseded_raw_frame_count": len(superseded_indices),
            "superseded_raw_frame_indices": superseded_indices,
            "control_derivative_contract": ("soma_il29_q_50hz_forward_difference_dq_v1"),
            "control_derivative_period_ns": SOURCE_SAMPLE_PERIOD_NS,
            "source_pose_timestamp_relabelled": False,
            "positions_repeated_or_synthesized": False,
            "positions_interpolated_from_measured_xr24": False,
            "measured_interpolation_contract": None,
            "interpolation_capture_span_ns": 0,
            "interpolation_body_source_span_ns": 0,
            "interpolation_ready_delay_ns": 0,
        }
        self._control_index += 1
        return sample

    def _sample_interpolated(
        self,
        right_delivery: dict[str, Any],
        *,
        tick_ns: int,
    ) -> dict[str, Any]:
        left_delivery = self._last_selected_delivery
        if left_delivery is None:
            raise RuntimeError("bounded XR24 interpolation has no measured left frame")
        left = left_delivery["frame"]
        right = right_delivery["frame"]
        left_raw_index = int(left["frame_index"])
        right_raw_index = int(right["frame_index"])
        left_capture_ns = int(left["capture_monotonic_ns"])
        right_capture_ns = int(right["capture_monotonic_ns"])
        left_body_ns = int(left["body_sample_timestamp_ns"])
        right_body_ns = int(right["body_sample_timestamp_ns"])
        capture_span_ns = right_capture_ns - left_capture_ns
        body_span_ns = right_body_ns - left_body_ns
        if right_raw_index != left_raw_index + 1 or not left_capture_ns < tick_ns < right_capture_ns:
            raise RuntimeError("bounded XR24 interpolation does not have consecutive causal brackets")
        if capture_span_ns > _MAX_INTERPOLATION_CAPTURE_SPAN_NS:
            raise RuntimeError(
                "XR24 interpolation capture bracket exceeds bounded dropout limit: "
                f"span_ns={capture_span_ns} "
                f"limit_ns={_MAX_INTERPOLATION_CAPTURE_SPAN_NS}"
            )
        if body_span_ns <= 0 or body_span_ns > _MAX_SELECTED_BODY_SOURCE_TIMESTAMP_DELTA_NS:
            raise RuntimeError(
                "XR24 interpolation body-source bracket exceeds bounded dropout "
                f"limit: span_ns={body_span_ns} "
                f"limit_ns={_MAX_SELECTED_BODY_SOURCE_TIMESTAMP_DELTA_NS}"
            )
        alpha = (tick_ns - left_capture_ns) / capture_span_ns
        left_poses = left["body_poses"]
        right_poses = right["body_poses"]
        if len(left_poses) != 24 or len(right_poses) != 24:
            raise RuntimeError("bounded interpolation requires 24 measured XR roles")
        body_poses = [
            _interpolate_xr24_pose(left_pose, right_pose, alpha)
            for left_pose, right_pose in zip(left_poses, right_poses, strict=True)
        ]
        interpolated_body_ns = round(left_body_ns + alpha * body_span_ns)
        ready_delay_ns = right_capture_ns - tick_ns
        source_age_ns = tick_ns - left_capture_ns
        if source_age_ns > _MAX_INTERPOLATION_LEFT_BRACKET_AGE_NS:
            raise RuntimeError(
                "XR24 interpolation left bracket exceeds bounded age limit: "
                f"age_ns={source_age_ns} "
                f"limit_ns={_MAX_INTERPOLATION_LEFT_BRACKET_AGE_NS}"
            )
        self._interpolated_frames += 1
        self._interpolation_capture_spans_ns.append(capture_span_ns)
        self._interpolation_body_source_spans_ns.append(body_span_ns)
        self._interpolation_ready_delays_ns.append(ready_delay_ns)
        self._interpolation_raw_brackets.append([left_raw_index, right_raw_index])
        self._max_capture_age_at_tick_ns = max(self._max_capture_age_at_tick_ns, source_age_ns)
        sample = {
            "source_frame_index": self._control_index,
            "reference_monotonic_ns": tick_ns,
            # The derived pose cannot exist until its right measured bracket.
            "capture_monotonic_ns": right_capture_ns,
            "raw_bracket_indices": [left_raw_index, right_raw_index],
            "raw_interpolation_alpha": alpha,
            "body_poses": body_poses,
            "pose_selection_contract": self._POSE_SELECTION_CONTRACT,
            "selected_raw_frame_index": None,
            "selected_body_sample_sequence": None,
            "selected_body_sample_timestamp_ns": interpolated_body_ns,
            "selected_capture_delta_ns": None,
            "selected_body_source_timestamp_delta_ns": (interpolated_body_ns - left_body_ns),
            "selected_source_delta_ns": interpolated_body_ns - left_body_ns,
            "selected_body_sequence_delta_from_previous_selected": None,
            "selected_capture_age_at_tick_ns": source_age_ns,
            "selected_source_age_at_tick_ns": source_age_ns,
            "selected_source_age_clock": "local_capture_monotonic_left_bracket",
            "selected_capture_wait_duration_ns": int(right_delivery["capture_wait_duration_ns"]),
            "selected_capture_ipc_duration_ns": int(right_delivery["capture_ipc_duration_ns"]),
            "selected_capture_queue_depth_before_enqueue": int(right_delivery["queue_depth_before_enqueue"]),
            "selected_capture_queue_depth_after_dequeue": int(right_delivery["queue_depth_after_dequeue"]),
            "selected_capture_frame_index_delta": None,
            "selected_raw_body_sample_sequence_delta": None,
            "selected_body_source_sequence_gap_count": int(right_delivery["body_source_sequence_gap_count"]),
            "selected_mailbox_batch_size": int(right_delivery.get("mailbox_batch_size", 1)),
            "selected_mailbox_coalesced_prior_frames": int(
                right_delivery.get("mailbox_coalesced_prior_frames", 0)
            ),
            "superseded_raw_frame_count": 0,
            "superseded_raw_frame_indices": [],
            "control_derivative_contract": ("soma_il29_q_50hz_forward_difference_dq_v1"),
            "control_derivative_period_ns": SOURCE_SAMPLE_PERIOD_NS,
            "source_pose_timestamp_relabelled": False,
            "positions_repeated_or_synthesized": False,
            "positions_interpolated_from_measured_xr24": True,
            "measured_interpolation_contract": (_MEASURED_INTERPOLATION_CONTRACT),
            "interpolation_capture_span_ns": capture_span_ns,
            "interpolation_body_source_span_ns": body_span_ns,
            "interpolation_ready_delay_ns": ready_delay_ns,
        }
        self._control_index += 1
        return sample

    def _select_origin(self, delivery: dict[str, Any]) -> dict[str, Any]:
        self._ingest(delivery)
        selected = self._pending.popleft()
        capture_ns = int(selected["frame"]["capture_monotonic_ns"])
        self._next_tick_ns = capture_ns + SOURCE_SAMPLE_PERIOD_NS
        return self._sample(selected, tick_ns=capture_ns, superseded=[])

    def take_origin(self) -> dict[str, Any]:
        origin = self._origin_sample
        self._origin_sample = None
        if origin is None:
            raise RuntimeError("measured XR24 stream origin already consumed")
        return origin

    def push(self, deliveries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for delivery in deliveries:
            self._ingest(delivery)
        emitted: list[dict[str, Any]] = []
        assert self._next_tick_ns is not None
        while self._pending and int(self._pending[-1]["frame"]["capture_monotonic_ns"]) >= self._next_tick_ns:
            eligible: list[dict[str, Any]] = []
            while self._pending and int(self._pending[0]["frame"]["capture_monotonic_ns"]) <= self._next_tick_ns:
                eligible.append(self._pending.popleft())
            if not eligible:
                emitted.append(self._sample_interpolated(self._pending[0], tick_ns=self._next_tick_ns))
            else:
                selected = eligible[-1]
                emitted.append(
                    self._sample(
                        selected,
                        tick_ns=self._next_tick_ns,
                        superseded=eligible[:-1],
                    )
                )
            self._next_tick_ns += SOURCE_SAMPLE_PERIOD_NS
        return emitted

    def stats(self) -> dict[str, Any]:
        return {
            "pose_selection_contract": self._POSE_SELECTION_CONTRACT,
            "selected_frames": self._selected_frames,
            "interpolated_control_frames": self._interpolated_frames,
            "total_control_frames": (self._selected_frames + self._interpolated_frames),
            "selected_raw_frame_indices": list(self._selected_raw_indices),
            "superseded_raw_frames": self._superseded_frames,
            "superseded_raw_frame_indices": list(self._superseded_raw_indices),
            "selected_capture_delta_timing": _duration_summary_ns(self._selected_capture_deltas_ns),
            "selected_body_source_timestamp_delta_timing": (
                _duration_summary_ns(self._selected_body_source_timestamp_deltas_ns)
            ),
            "max_capture_age_at_tick_ns": self._max_capture_age_at_tick_ns,
            "maximum_allowed_capture_age_at_tick_ns": (_MAX_INTERPOLATION_LEFT_BRACKET_AGE_NS),
            "positions_repeated_or_synthesized": False,
            "positions_interpolated_from_measured_xr24": (self._interpolated_frames > 0),
            "measured_interpolation_contract": (_MEASURED_INTERPOLATION_CONTRACT),
            "maximum_interpolation_capture_span_ns": (_MAX_INTERPOLATION_CAPTURE_SPAN_NS),
            "maximum_interpolation_left_bracket_age_ns": (_MAX_INTERPOLATION_LEFT_BRACKET_AGE_NS),
            "maximum_interpolation_body_source_span_ns": (_MAX_SELECTED_BODY_SOURCE_TIMESTAMP_DELTA_NS),
            "interpolation_capture_span_timing": _duration_summary_ns(self._interpolation_capture_spans_ns),
            "interpolation_body_source_span_timing": _duration_summary_ns(
                self._interpolation_body_source_spans_ns
            ),
            "interpolation_ready_delay_timing": _duration_summary_ns(self._interpolation_ready_delays_ns),
            "interpolation_raw_brackets": list(self._interpolation_raw_brackets),
            "pending_raw_frames": len(self._pending),
            "pending_raw_frame_indices": [int(delivery["frame"]["frame_index"]) for delivery in self._pending],
        }


def _duration_summary_ns(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_ns": None, "max_ns": None}
    return {
        "count": len(values),
        "mean_ns": sum(values) / len(values),
        "max_ns": max(values),
    }


def _terminal_outcome(
    *,
    published_packets: int,
    requested_packets: int,
    stop_requested: bool,
) -> tuple[str, str]:
    if published_packets >= requested_packets:
        return "session_complete", "packet_target_reached"
    if stop_requested:
        return "session_stopped", "signal_requested"
    raise RuntimeError("publisher ended before requested packet target")


def _require_publisher_age_budget(
    *,
    control_monotonic_ns: int,
    observed_monotonic_ns: int,
    maximum_age_ns: int = _PUBLISHER_MAX_AGE_NS,
) -> int:
    age_ns = observed_monotonic_ns - control_monotonic_ns
    if age_ns < 0:
        raise RuntimeError("causal reference timestamp is in future")
    if age_ns > maximum_age_ns:
        raise RuntimeError(
            "refusing stale causal reference before ZMQ send: "
            f"age_ns={age_ns} "
            f"publisher_hard_stale_limit_ns={maximum_age_ns}"
        )
    return age_ns


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    publisher_max_age_ns = (
        _DOWNSTREAM_HIGH_LEVEL_FRESHNESS_NS if args.read_only_diagnostic_100ms_max_age else _PUBLISHER_MAX_AGE_NS
    )
    if args.packets <= 0:
        raise ValueError("packets must be positive")
    if args.timeout_seconds <= 0.0 or args.subscriber_warmup_s < 0.0:
        raise ValueError("timeouts must be positive")
    if args.capture_queue_capacity <= 0:
        raise ValueError("capture queue capacity must be positive")
    args.workspace = args.workspace.resolve()
    args.xrt_module_dir = (
        _default_xrt_module_dir(args.workspace) if args.xrt_module_dir is None else args.xrt_module_dir.resolve()
    )
    xrt_module_path = _resolve_xrt_module_binary(args.xrt_module_dir)
    xrt_module_sha256 = _sha256_file(xrt_module_path)
    for path, label in (
        (args.capture_python.resolve(), "capture Python"),
        (args.xrt_module_dir.resolve(), "XRT module directory"),
        (args.service_binary.resolve(), "XR service"),
        (args.soma_source_root.resolve(), "SOMA source"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} missing: {path}")
    pico_client_apk_sha256 = str(args.pico_client_apk_sha256).strip().lower()
    if len(pico_client_apk_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in pico_client_apk_sha256
    ):
        raise ValueError("pico-client-apk-sha256 must be 64 lowercase hex chars")
    args.pico_client_apk_sha256 = pico_client_apk_sha256
    publisher_path = Path(__file__).resolve()
    capture_worker_path = args.workspace.resolve() / "gear_sonic/scripts/stream_g1_23dof_pico_raw_worker.py"
    publisher_sha256 = _sha256_file(publisher_path)
    capture_worker_sha256 = _sha256_file(capture_worker_path)

    evidence = EvidenceLog(args.evidence)
    session_id = f"pico-causal-live-{uuid.uuid4()}"
    context: zmq.Context[Any] | None = None
    socket: zmq.Socket[Any] | None = None
    worker: RawCaptureWorker | None = None
    background_capture: BackgroundRawCapture | None = None
    selector: MeasuredFrameTickSelector | None = None
    startup_raw_frame_indices: list[int] = []
    startup_superseded_raw_frame_indices: list[int] = []
    stream_origin_raw_frame_index: int | None = None
    stop_state = PublisherStopState()
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, stop_state.handle)
    signal.signal(signal.SIGTERM, stop_state.handle)
    gc_was_enabled: bool | None = None
    gc_collected_objects: int | None = None
    gc_pause_active = False
    previous_thread_switch_interval_s = sys.getswitchinterval()
    capture_thread_switch_interval_active = False
    capture_wait_durations_ns: list[int] = []
    capture_ipc_durations_ns: list[int] = []
    queue_depths_after_dequeue: list[int] = []
    rolling_durations_ns: list[int] = []
    target_durations_ns: list[int] = []
    solver_durations_ns: list[int] = []
    body_term_durations_ns: list[int] = []
    evidence_write_durations_ns: list[int] = []
    started_ns = time.monotonic_ns()
    deadline_ns = started_ns + int(args.timeout_seconds * 1e9)
    evidence.write(
        {
            "schema_version": 1,
            "kind": "g1_true23_pico_causal_zmq_evidence",
            "event": "session_start",
            "session_id": session_id,
            "started_monotonic_ns": started_ns,
            "started_unix_ns": time.time_ns(),
            "completed_unix_ns_contract": "terminal_record_v1",
            "bind": args.bind,
            "pinned_soma_source_root": str(args.soma_source_root.resolve()),
            "publisher_sha256": publisher_sha256,
            "requested_xrt_module_path": str(xrt_module_path),
            "requested_xrt_module_sha256": xrt_module_sha256,
            "capture_worker_sha256": capture_worker_sha256,
            "pico_client_apk_sha256": pico_client_apk_sha256,
            "capture_queue_capacity": args.capture_queue_capacity,
            "requested_packets": args.packets,
            "termination_contract": ("finite_session_complete_or_signal_session_stopped_v1"),
            "authorization": dict(_AUTHORIZATION),
        }
    )
    try:
        if args.experimental_solver_iterations is None:
            rolling = PinnedSomaRollingRetargeter(soma_source_root=args.soma_source_root.resolve())
        else:
            rolling = ExperimentalSomaRollingRetargeter(
                soma_source_root=args.soma_source_root.resolve(),
                solver_iterations=args.experimental_solver_iterations,
            )
        worker = _new_worker(
            args,
            session_id,
            expected_xrt_module_path=xrt_module_path,
            expected_xrt_module_sha256=xrt_module_sha256,
        )
        prime_frame = _next_frame_until(worker, deadline_ns=deadline_ns)
        if worker.identity is None:
            raise RuntimeError("capture worker did not bind source identity")
        if worker.xrt_binding is None:
            raise RuntimeError("capture worker did not verify imported XRT binding")
        xrt_binding = dict(worker.xrt_binding)
        if _sha256_file(xrt_module_path) != xrt_module_sha256:
            raise RuntimeError("requested XRT module changed during session")
        evidence.write(
            {
                "schema_version": 1,
                "kind": "g1_true23_pico_causal_zmq_evidence",
                "event": "xrt_binding_verified",
                "session_id": session_id,
                "imported_xrt_module_path": xrt_binding["path"],
                "imported_xrt_module_sha256": xrt_binding["sha256"],
                "matches_requested_binding": True,
                "authorization": dict(_AUTHORIZATION),
            }
        )
        prime_identity = worker.identity

        def _record_neutral_rejection(
            rejection_count: int,
            rejected_frame: dict[str, Any],
            reason: str,
        ) -> None:
            evidence.write(
                {
                    "schema_version": 1,
                    "kind": "g1_true23_pico_causal_zmq_evidence",
                    "event": "neutral_standing_frame_rejected",
                    "session_id": session_id,
                    "neutral_standing_rejection_count": rejection_count,
                    "rejected_raw_frame_index": rejected_frame["frame_index"],
                    "rejected_body_sample_sequence": rejected_frame["body_sample_sequence"],
                    "failure": reason,
                    "semantic_sample_emitted": False,
                    "authorization": dict(_AUTHORIZATION),
                }
            )

        prime_frame, prime_report, neutral_rejection_count = _prime_until_neutral_standing(
            rolling,
            worker,
            initial_frame=prime_frame,
            deadline_ns=deadline_ns,
            on_rejection=_record_neutral_rejection,
        )
        evidence.write(
            {
                "schema_version": 1,
                "kind": "g1_true23_pico_causal_zmq_evidence",
                "event": "solver_primed_before_stream_clock",
                "session_id": session_id,
                "prime_report": prime_report,
                "prime_body_sample_sequence": prime_frame["body_sample_sequence"],
                "neutral_standing_rejected_frame_count": (neutral_rejection_count),
                "semantic_sample_emitted": False,
            }
        )

        context = zmq.Context(io_threads=1)
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 1000)
        socket.bind(args.bind)
        if args.subscriber_warmup_s:
            time.sleep(args.subscriber_warmup_s)

        worker.start_continuous()
        first_request_started_ns = time.monotonic_ns()
        first_frame = _next_frame_until(worker, deadline_ns=deadline_ns)
        first_received_ns = time.monotonic_ns()
        first_capture_ipc_ns = first_received_ns - int(first_frame["capture_monotonic_ns"])
        if first_capture_ipc_ns < 0:
            raise RuntimeError("initial XR24 capture timing is non-causal")
        first_delivery = {
            "frame": first_frame,
            "request_started_monotonic_ns": first_request_started_ns,
            "received_monotonic_ns": first_received_ns,
            "capture_wait_duration_ns": (first_received_ns - first_request_started_ns),
            "capture_ipc_duration_ns": first_capture_ipc_ns,
            "capture_frame_index_delta": (int(first_frame["frame_index"]) - int(prime_frame["frame_index"])),
            "body_sample_sequence_delta": (
                int(first_frame["body_sample_sequence"]) - int(prime_frame["body_sample_sequence"])
            ),
            "body_source_sequence_gap_count": max(
                0,
                int(first_frame["body_sample_sequence"]) - int(prime_frame["body_sample_sequence"]) - 1,
            ),
            "queue_depth_before_enqueue": 0,
            "queue_depth_after_dequeue": 0,
        }
        if worker.xrt_binding != xrt_binding:
            raise RuntimeError("XRT binding changed in single worker lifecycle")
        if worker.identity != prime_identity:
            raise RuntimeError("PICO source identity changed in single worker lifecycle")
        identity = worker.identity
        assert identity is not None
        if identity.get("source", {}).get("xrobotoolkit_sdk_sha256") != xrt_module_sha256:
            raise RuntimeError("PICO source identity XRT hash mismatch")
        background_capture = BackgroundRawCapture(
            worker,
            deadline_ns=deadline_ns,
            capacity=args.capture_queue_capacity,
            initial_previous_frame=first_frame,
        )
        # The pinned Warp/Newton call spends most of each 20 ms control period
        # in Python-owned native calls.  Use a 1 ms interpreter handoff while
        # streaming so the lossless pipe-drain thread can account for measured
        # XR24 frames before the next solver burst.  This changes scheduling
        # only: solver iterations, samples, timestamps, and freshness limits
        # remain identical.
        sys.setswitchinterval(0.001)
        capture_thread_switch_interval_active = True
        background_capture.start()
        producer = CausalHistorySemanticProducer(source_session_id=session_id)
        gc_was_enabled = gc.isenabled()
        gc_collected_objects = gc.collect()
        if gc_was_enabled:
            gc.disable()
        gc_pause_active = True
        evidence.write(
            {
                "schema_version": 1,
                "kind": "g1_true23_pico_causal_zmq_evidence",
                "event": "stream_runtime_configured",
                "session_id": session_id,
                "capture_mode": ("continuous_worker_fifo_bounded_measured_tick_selector_v3"),
                "pose_selection_contract": (MeasuredFrameTickSelector._POSE_SELECTION_CONTRACT),
                "xrt_worker_lifecycle": ("single_process_request_prime_then_continuous_stream_v2"),
                "capture_queue_capacity": args.capture_queue_capacity,
                "capture_queue_overflow_policy": "fail_closed",
                "raw_capture_accounting": ("every_raw_frame_ordered; selected_or_explicitly_superseded"),
                "startup_origin_selection": ("newest_validated_raw_frame_after_runtime_setup_v1"),
                "maximum_selected_source_age_ns": SOURCE_SAMPLE_PERIOD_NS,
                "selected_source_age_clock": "local_capture_monotonic",
                "measured_interpolation_enabled": True,
                "measured_interpolation_contract": (_MEASURED_INTERPOLATION_CONTRACT),
                "maximum_interpolation_capture_span_ns": (_MAX_INTERPOLATION_CAPTURE_SPAN_NS),
                "maximum_interpolation_left_bracket_age_ns": (_MAX_INTERPOLATION_LEFT_BRACKET_AGE_NS),
                "maximum_interpolation_body_source_span_ns": (_MAX_SELECTED_BODY_SOURCE_TIMESTAMP_DELTA_NS),
                "maximum_selected_body_source_timestamp_delta_ns": (_MAX_SELECTED_BODY_SOURCE_TIMESTAMP_DELTA_NS),
                "control_derivative_contract": ("soma_il29_q_50hz_forward_difference_dq_v1"),
                "control_derivative_period_ns": SOURCE_SAMPLE_PERIOD_NS,
                "source_pose_timestamp_relabelled": False,
                "termination_contract": ("finite_session_complete_or_signal_session_stopped_v1"),
                "post_start_exact_contiguous_20ms_required": True,
                "publisher_max_age_ns": publisher_max_age_ns,
                "read_only_diagnostic_100ms_max_age": (args.read_only_diagnostic_100ms_max_age),
                "publisher_performance_target_ns": (_PUBLISHER_PERFORMANCE_TARGET_NS),
                "publisher_performance_target_is_safety_gate": False,
                "downstream_total_freshness_limit_ns": (_DOWNSTREAM_HIGH_LEVEL_FRESHNESS_NS),
                "reserved_transport_and_inference_margin_ns": (
                    _DOWNSTREAM_HIGH_LEVEL_FRESHNESS_NS - publisher_max_age_ns
                ),
                "cyclic_gc_was_enabled": gc_was_enabled,
                "cyclic_gc_collected_objects_before_stream": gc_collected_objects,
                "cyclic_gc_enabled_during_stream": gc.isenabled(),
                "python_thread_switch_interval_s": sys.getswitchinterval(),
                "authorization": dict(_AUTHORIZATION),
            }
        )
        startup_background_deliveries = background_capture.next_latest_batch(
            deadline_ns=deadline_ns,
            stop_event=stop_state.event,
        )
        startup_deliveries = [first_delivery, *startup_background_deliveries]
        startup_raw_frame_indices = [int(delivery["frame"]["frame_index"]) for delivery in startup_deliveries]
        startup_superseded_deliveries = startup_deliveries[:-1]
        startup_superseded_raw_frame_indices = startup_raw_frame_indices[:-1]
        stream_origin_delivery = startup_deliveries[-1]
        stream_origin_raw_frame_index = startup_raw_frame_indices[-1]
        selector = MeasuredFrameTickSelector(stream_origin_delivery)
        origin_sample = selector.take_origin()
        evidence.write(
            {
                "schema_version": 1,
                "kind": "g1_true23_pico_causal_zmq_evidence",
                "event": "stream_origin_selected",
                "session_id": session_id,
                "pose_selection_contract": origin_sample["pose_selection_contract"],
                "startup_raw_frame_count": len(startup_deliveries),
                "startup_raw_frame_indices": startup_raw_frame_indices,
                "startup_superseded_raw_frame_count": len(startup_superseded_raw_frame_indices),
                "startup_superseded_raw_frame_indices": (startup_superseded_raw_frame_indices),
                "startup_superseded_body_source_sequence_gap_count": sum(
                    int(delivery["body_source_sequence_gap_count"]) for delivery in startup_superseded_deliveries
                ),
                "selected_raw_frame_index": origin_sample["selected_raw_frame_index"],
                "selected_capture_monotonic_ns": origin_sample["capture_monotonic_ns"],
                "selected_body_sample_sequence": origin_sample["selected_body_sample_sequence"],
                "selected_body_sample_timestamp_ns": origin_sample["selected_body_sample_timestamp_ns"],
                "selected_capture_delta_ns": origin_sample["selected_capture_delta_ns"],
                "selected_body_source_timestamp_delta_ns": origin_sample[
                    "selected_body_source_timestamp_delta_ns"
                ],
                "selected_capture_age_at_tick_ns": origin_sample["selected_capture_age_at_tick_ns"],
                "selected_source_age_clock": origin_sample["selected_source_age_clock"],
                "selected_body_source_sequence_gap_count": origin_sample[
                    "selected_body_source_sequence_gap_count"
                ],
                "control_source_frame_index": origin_sample["source_frame_index"],
                "control_monotonic_ns": origin_sample["reference_monotonic_ns"],
                "raw_bracket_indices": origin_sample["raw_bracket_indices"],
                "raw_interpolation_alpha": origin_sample["raw_interpolation_alpha"],
                "source_pose_timestamp_relabelled": False,
                "positions_repeated_or_synthesized": False,
                "semantic_sample_emitted": False,
                "authorization": dict(_AUTHORIZATION),
            },
            durable=False,
        )
        raw_frames = len(startup_deliveries)
        retargeted_samples = 0
        published_packets = 0
        fresh_packets = 0
        performance_target_packets = 0
        fresh_within_60ms_packets = 0
        last_published_control_index: int | None = None
        last_published_control_ns: int | None = None
        pending_samples: deque[dict[str, Any]] = deque([origin_sample])
        capture_wait_durations_ns.extend(
            int(delivery["capture_wait_duration_ns"]) for delivery in startup_deliveries
        )
        capture_ipc_durations_ns.extend(
            int(delivery["capture_ipc_duration_ns"]) for delivery in startup_deliveries
        )
        queue_depths_after_dequeue.extend(
            int(delivery["queue_depth_after_dequeue"]) for delivery in startup_deliveries
        )
        while published_packets < args.packets:
            if stop_state.event.is_set():
                break
            if time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("causal PICO publisher session timed out")
            if not pending_samples:
                try:
                    deliveries = background_capture.next_latest_batch(
                        deadline_ns=deadline_ns,
                        stop_event=stop_state.event,
                    )
                except PublisherStopRequested:
                    break
                raw_frames += len(deliveries)
                capture_wait_durations_ns.extend(
                    int(delivery["capture_wait_duration_ns"]) for delivery in deliveries
                )
                capture_ipc_durations_ns.extend(
                    int(delivery["capture_ipc_duration_ns"]) for delivery in deliveries
                )
                queue_depths_after_dequeue.extend(
                    int(delivery["queue_depth_after_dequeue"]) for delivery in deliveries
                )
                pending_samples.extend(selector.push(deliveries))
                if not pending_samples:
                    continue
            sample = pending_samples.popleft()
            if stop_state.event.is_set():
                pending_samples.appendleft(sample)
                break
            rolling_row = rolling.push(sample)
            retargeted_samples += 1
            rolling_durations_ns.append(int(rolling_row["compute_duration_ns"]))
            target_durations_ns.append(int(rolling_row["target_build_duration_ns"]))
            solver_durations_ns.append(int(rolling_row["solver_duration_ns"]))
            body_term_durations_ns.append(int(rolling_row["body_term_duration_ns"]))
            semantic = producer.push(rolling_row)
            if stop_state.event.is_set():
                break
            if semantic is None:
                continue
            if sample["positions_repeated_or_synthesized"] is not False:
                raise RuntimeError("measured selector synthesized a pose")
            if (
                int(semantic["proof_source_frame_index"]) != int(sample["source_frame_index"])
                or int(semantic["proof_reference_monotonic_ns"]) != int(sample["reference_monotonic_ns"])
                or int(semantic["proof_capture_monotonic_ns"]) != int(sample["capture_monotonic_ns"])
            ):
                raise RuntimeError("published semantic proof is not aligned to selected XR24 pose")
            if semantic["control_derivative_contract"] != sample["control_derivative_contract"]:
                raise RuntimeError("published derivative contract changed from exact 50 Hz")
            validation_started_ns = time.monotonic_ns()
            validation = validate_causal_history_packet(semantic)
            validation_finished_ns = time.monotonic_ns()
            reference = causal_history_reference_terms(semantic)
            control_index = int(reference["control_source_frame_index"])
            control_ns = int(reference["control_monotonic_ns"])
            if last_published_control_index is not None and (
                control_index != last_published_control_index + 1
                or control_ns != last_published_control_ns + 20_000_000
            ):
                raise RuntimeError("published causal control stream lost exact contiguous 20 ms")
            wire_started_ns = time.monotonic_ns()
            wire = _canonical_bytes(reference, newline=False)
            wire_finished_ns = time.monotonic_ns()
            background_capture.raise_if_failed()
            if stop_state.event.is_set():
                break
            send_started_ns = time.monotonic_ns()
            pre_send_age_ns = _require_publisher_age_budget(
                control_monotonic_ns=control_ns,
                observed_monotonic_ns=send_started_ns,
                maximum_age_ns=publisher_max_age_ns,
            )
            socket.send(wire, flags=zmq.NOBLOCK)
            published_ns = time.monotonic_ns()
            age_ns = published_ns - control_ns
            if age_ns > publisher_max_age_ns:
                raise RuntimeError(
                    "ZMQ send completed outside publisher freshness budget: "
                    f"age_ns={age_ns} "
                    f"publisher_budget_ns={publisher_max_age_ns}"
                )
            is_fresh = age_ns <= publisher_max_age_ns
            if not is_fresh:
                raise RuntimeError("stale causal reference was not publishable")
            fresh_packets += int(is_fresh)
            within_performance_target = age_ns <= _PUBLISHER_PERFORMANCE_TARGET_NS
            performance_target_packets += int(within_performance_target)
            within_60ms = age_ns <= 60_000_000
            fresh_within_60ms_packets += int(within_60ms)
            previous_evidence_write_duration_ns = (
                evidence_write_durations_ns[-1] if evidence_write_durations_ns else None
            )
            evidence_started_ns = time.monotonic_ns()
            evidence.write(
                {
                    "schema_version": 1,
                    "kind": "g1_true23_pico_causal_zmq_evidence",
                    "event": "reference_packet_published",
                    "session_id": session_id,
                    "packet_index": published_packets,
                    "control_source_frame_index": control_index,
                    "control_monotonic_ns": control_ns,
                    "published_monotonic_ns": published_ns,
                    "publisher_age_ns": age_ns,
                    "fresh_within_40ms": within_performance_target,
                    "fresh_within_60ms": within_60ms,
                    "publisher_performance_target_ns": (_PUBLISHER_PERFORMANCE_TARGET_NS),
                    "publisher_performance_target_met": (within_performance_target),
                    "publisher_performance_target_is_safety_gate": False,
                    "publisher_max_age_ns": publisher_max_age_ns,
                    "read_only_diagnostic_100ms_max_age": (args.read_only_diagnostic_100ms_max_age),
                    "within_publisher_age_budget": True,
                    "pre_send_age_ns": pre_send_age_ns,
                    "pose_selection_contract": sample["pose_selection_contract"],
                    "selected_raw_frame_index": sample["selected_raw_frame_index"],
                    "selected_body_sample_sequence": sample["selected_body_sample_sequence"],
                    "selected_body_sample_timestamp_ns": sample["selected_body_sample_timestamp_ns"],
                    "selected_capture_monotonic_ns": sample["capture_monotonic_ns"],
                    "selected_source_delta_ns": sample["selected_source_delta_ns"],
                    "selected_capture_delta_ns": sample["selected_capture_delta_ns"],
                    "selected_body_source_timestamp_delta_ns": sample["selected_body_source_timestamp_delta_ns"],
                    "maximum_selected_body_source_timestamp_delta_ns": (
                        _MAX_SELECTED_BODY_SOURCE_TIMESTAMP_DELTA_NS
                    ),
                    "selected_body_sequence_delta_from_previous_selected": sample[
                        "selected_body_sequence_delta_from_previous_selected"
                    ],
                    "selected_capture_age_at_tick_ns": sample["selected_capture_age_at_tick_ns"],
                    "selected_source_age_at_tick_ns": sample["selected_source_age_at_tick_ns"],
                    "selected_source_age_clock": sample["selected_source_age_clock"],
                    "maximum_selected_source_age_ns": (
                        _MAX_INTERPOLATION_LEFT_BRACKET_AGE_NS
                        if sample["positions_interpolated_from_measured_xr24"]
                        else SOURCE_SAMPLE_PERIOD_NS
                    ),
                    "positions_interpolated_from_measured_xr24": sample[
                        "positions_interpolated_from_measured_xr24"
                    ],
                    "measured_interpolation_contract": sample["measured_interpolation_contract"],
                    "interpolation_capture_span_ns": sample["interpolation_capture_span_ns"],
                    "interpolation_body_source_span_ns": sample["interpolation_body_source_span_ns"],
                    "interpolation_ready_delay_ns": sample["interpolation_ready_delay_ns"],
                    "selected_capture_wait_duration_ns": sample["selected_capture_wait_duration_ns"],
                    "selected_capture_ipc_duration_ns": sample["selected_capture_ipc_duration_ns"],
                    "selected_capture_queue_depth_before_enqueue": sample[
                        "selected_capture_queue_depth_before_enqueue"
                    ],
                    "selected_capture_queue_depth_after_dequeue": sample[
                        "selected_capture_queue_depth_after_dequeue"
                    ],
                    "selected_capture_frame_index_delta": sample["selected_capture_frame_index_delta"],
                    "selected_raw_body_sample_sequence_delta": sample["selected_raw_body_sample_sequence_delta"],
                    "selected_body_source_sequence_gap_count": sample["selected_body_source_sequence_gap_count"],
                    "selected_mailbox_batch_size": sample["selected_mailbox_batch_size"],
                    "selected_mailbox_coalesced_prior_frames": sample["selected_mailbox_coalesced_prior_frames"],
                    "superseded_raw_frame_count": sample["superseded_raw_frame_count"],
                    "superseded_raw_frame_indices": sample["superseded_raw_frame_indices"],
                    "raw_bracket_indices": sample["raw_bracket_indices"],
                    "raw_interpolation_alpha": sample["raw_interpolation_alpha"],
                    "control_tick_period_ns": SOURCE_SAMPLE_PERIOD_NS,
                    "control_derivative_contract": sample["control_derivative_contract"],
                    "control_derivative_period_ns": sample["control_derivative_period_ns"],
                    "source_pose_timestamp_relabelled": sample["source_pose_timestamp_relabelled"],
                    "rolling_compute_duration_ns": rolling_row["compute_duration_ns"],
                    "target_build_duration_ns": rolling_row["target_build_duration_ns"],
                    "solver_duration_ns": rolling_row["solver_duration_ns"],
                    "body_term_duration_ns": rolling_row["body_term_duration_ns"],
                    "packet_validation_duration_ns": (validation_finished_ns - validation_started_ns),
                    "wire_encode_duration_ns": (wire_finished_ns - wire_started_ns),
                    "zmq_send_duration_ns": published_ns - send_started_ns,
                    "previous_evidence_write_duration_ns": (previous_evidence_write_duration_ns),
                    "cyclic_gc_enabled": gc.isenabled(),
                    "post_start_contiguous_20ms": True,
                    "packet_validation": validation,
                    "wire_sha256": hashlib.sha256(wire).hexdigest(),
                    "sdk_derivatives_consumed": False,
                    "positions_repeated_or_synthesized": sample["positions_repeated_or_synthesized"],
                    "authorization": dict(_AUTHORIZATION),
                },
                durable=False,
            )
            evidence_write_durations_ns.append(time.monotonic_ns() - evidence_started_ns)
            last_published_control_index = control_index
            last_published_control_ns = control_ns
            published_packets += 1
            if published_packets == 1 or published_packets % 10 == 0:
                print(
                    f"[PUBLISH] {published_packets}/{args.packets} "
                    f"age_ms={age_ns / 1e6:.1f} "
                    f"target40={within_performance_target} safe={is_fresh}",
                    flush=True,
                )

        terminal_event, completion_reason = _terminal_outcome(
            published_packets=published_packets,
            requested_packets=args.packets,
            stop_requested=stop_state.event.is_set(),
        )
        stopped_before_packet_target = terminal_event == "session_stopped"
        if fresh_packets != published_packets:
            raise RuntimeError("terminal success requires every packet fresh")
        background_capture.raise_if_failed()
        background_capture.request_stop()
        worker.close()
        worker = None
        background_capture.join()
        background_capture_stats = background_capture.stats()
        background_capture = None
        selector_stats = selector.stats()
        terminal_unprocessed_selected_raw_frame_indices = [
            int(sample["selected_raw_frame_index"])
            for sample in pending_samples
            if sample["selected_raw_frame_index"] is not None
        ]
        captured_raw_frame_accounting = {
            "startup_superseded_raw_frame_indices": list(startup_superseded_raw_frame_indices),
            "selector_selected_raw_frame_indices": list(selector_stats["selected_raw_frame_indices"]),
            "selector_superseded_raw_frame_indices": list(selector_stats["superseded_raw_frame_indices"]),
            "selector_pending_raw_frame_indices": list(selector_stats["pending_raw_frame_indices"]),
            "terminal_capture_queue_raw_frame_indices": list(background_capture_stats["queued_raw_frame_indices"]),
        }
        accounted_raw_frame_indices = [
            int(raw_index) for indices in captured_raw_frame_accounting.values() for raw_index in indices
        ]
        captured_raw_frames = 1 + int(background_capture_stats["captured_frames"])
        if (
            len(accounted_raw_frame_indices) != captured_raw_frames
            or len(set(accounted_raw_frame_indices)) != captured_raw_frames
        ):
            raise RuntimeError("terminal raw capture accounting is not exact and disjoint")
        if accounted_raw_frame_indices and sorted(accounted_raw_frame_indices) != list(
            range(
                min(accounted_raw_frame_indices),
                max(accounted_raw_frame_indices) + 1,
            )
        ):
            raise RuntimeError("terminal raw capture accounting is not contiguous")
        if raw_frames != captured_raw_frames - int(background_capture_stats["queue_depth_at_snapshot"]):
            raise RuntimeError("terminal drained raw capture count is inconsistent")
        if int(background_capture_stats["captured_frames"]) != (
            raw_frames - 1 + int(background_capture_stats["queue_depth_at_snapshot"])
        ):
            raise RuntimeError("terminal background capture equation failed")
        if raw_frames != (
            len(startup_superseded_raw_frame_indices)
            + int(selector_stats["selected_frames"])
            + int(selector_stats["superseded_raw_frames"])
            + int(selector_stats["pending_raw_frames"])
        ):
            raise RuntimeError("terminal drained selector accounting equation failed")
        if int(selector_stats["total_control_frames"]) != (retargeted_samples + len(pending_samples)):
            raise RuntimeError("terminal control/retargeted accounting equation failed")
        if int(selector_stats["max_capture_age_at_tick_ns"]) > _MAX_INTERPOLATION_LEFT_BRACKET_AGE_NS:
            raise RuntimeError("terminal selected capture-age bound was violated")
        selected_body_delta_max = selector_stats["selected_body_source_timestamp_delta_timing"]["max_ns"]
        if (
            selected_body_delta_max is not None
            and int(selected_body_delta_max) > _MAX_SELECTED_BODY_SOURCE_TIMESTAMP_DELTA_NS
        ):
            raise RuntimeError("terminal body source timestamp-delta bound violated")
        if _sha256_file(publisher_path) != publisher_sha256:
            raise RuntimeError("publisher source changed during session")
        if _sha256_file(capture_worker_path) != capture_worker_sha256:
            raise RuntimeError("capture worker source changed during session")
        if _sha256_file(xrt_module_path) != xrt_module_sha256:
            raise RuntimeError("requested XRT module changed during session")
        evidence.write(
            {
                "schema_version": 1,
                "kind": "g1_true23_pico_causal_zmq_evidence",
                "event": terminal_event,
                "session_id": session_id,
                "requested_packets": args.packets,
                "completion_reason": completion_reason,
                **stop_state.evidence(),
                "completed_unix_ns_contract": "terminal_record_v1",
                "completed_unix_ns": time.time_ns(),
                "raw_frames": captured_raw_frames,
                "raw_frames_drained": raw_frames,
                "retargeted_samples": retargeted_samples,
                "published_packets": published_packets,
                "fresh_packets": fresh_packets,
                "all_packets_fresh_within_60ms": (fresh_within_60ms_packets == published_packets),
                "all_packets_fresh_within_40ms": (performance_target_packets == published_packets),
                "packets_meeting_40ms_performance_target": (performance_target_packets),
                "publisher_performance_target_ns": (_PUBLISHER_PERFORMANCE_TARGET_NS),
                "publisher_performance_target_is_safety_gate": False,
                "all_packets_within_publisher_age_budget": True,
                "publisher_max_age_ns": publisher_max_age_ns,
                "read_only_diagnostic_100ms_max_age": (args.read_only_diagnostic_100ms_max_age),
                "capture_mode": ("continuous_worker_fifo_bounded_measured_tick_selector_v3"),
                "pose_selection": selector_stats,
                "background_capture": background_capture_stats,
                "stream_origin_raw_frame_index": stream_origin_raw_frame_index,
                "startup_raw_frame_indices": startup_raw_frame_indices,
                "startup_superseded_raw_frame_count": len(startup_superseded_raw_frame_indices),
                "startup_superseded_raw_frame_indices": (startup_superseded_raw_frame_indices),
                "captured_raw_frame_accounting": (captured_raw_frame_accounting),
                "captured_raw_frame_accounting_count": len(accounted_raw_frame_indices),
                "all_captured_raw_frames_explicitly_accounted": True,
                "all_raw_accounting_categories_disjoint": True,
                "total_body_source_sequence_gap_count": (
                    int(first_delivery["body_source_sequence_gap_count"])
                    + int(background_capture_stats["source_sequence_gap_count"])
                ),
                "xrt_worker_lifecycle": ("single_process_request_prime_then_continuous_stream_v2"),
                "capture_wait_timing": _duration_summary_ns(capture_wait_durations_ns),
                "capture_ipc_timing": _duration_summary_ns(capture_ipc_durations_ns),
                "capture_queue_depth_after_dequeue_max": (
                    max(queue_depths_after_dequeue) if queue_depths_after_dequeue else None
                ),
                "rolling_compute_timing": _duration_summary_ns(rolling_durations_ns),
                "target_build_timing": _duration_summary_ns(target_durations_ns),
                "solver_timing": _duration_summary_ns(solver_durations_ns),
                "body_term_timing": _duration_summary_ns(body_term_durations_ns),
                "evidence_write_timing": _duration_summary_ns(evidence_write_durations_ns),
                "startup_frames_dropped_silently": 0,
                "semantic_frames_skipped_between_published_packets": 0,
                "raw_frames_silently_dropped": 0,
                "terminal_unprocessed_selected_samples": len(pending_samples),
                "terminal_unprocessed_selected_raw_frame_indices": (
                    terminal_unprocessed_selected_raw_frame_indices
                ),
                "terminal_unconsumed_capture_queue_depth": (background_capture_stats["queue_depth_at_snapshot"]),
                "post_start_exact_contiguous_20ms": True,
                "cyclic_gc_was_enabled": gc_was_enabled,
                "cyclic_gc_collected_objects_before_stream": gc_collected_objects,
                "cyclic_gc_enabled_during_stream": gc.isenabled(),
                "cyclic_gc_restore_in_finally": True,
                "imported_xrt_module_path": xrt_binding["path"],
                "imported_xrt_module_sha256": xrt_binding["sha256"],
                "publisher_sha256": publisher_sha256,
                "capture_worker_sha256": capture_worker_sha256,
                "pico_client_apk_sha256": pico_client_apk_sha256,
                "authorization": dict(_AUTHORIZATION),
            }
        )
        terminal_label = "STOPPED" if stopped_before_packet_target else "PASS"
        print(
            f"[{terminal_label}] published {published_packets} real causal "
            f"PICO packets; fresh={fresh_packets}/{published_packets}; "
            f"target40={performance_target_packets}/{published_packets}; "
            "no robot channel opened",
            flush=True,
        )
        return 0
    except BaseException as exc:
        evidence.write(
            {
                "schema_version": 1,
                "kind": "g1_true23_pico_causal_zmq_evidence",
                "event": "session_failed",
                "session_id": session_id,
                "completed_unix_ns_contract": "terminal_record_v1",
                "completed_unix_ns": time.time_ns(),
                "failed_monotonic_ns": time.monotonic_ns(),
                "failure": str(exc),
                **stop_state.evidence(),
                "background_capture": (background_capture.stats() if background_capture is not None else None),
                "pose_selection": (selector.stats() if selector is not None else None),
                "capture_wait_timing": _duration_summary_ns(capture_wait_durations_ns),
                "rolling_compute_timing": _duration_summary_ns(rolling_durations_ns),
                "target_build_timing": _duration_summary_ns(target_durations_ns),
                "solver_timing": _duration_summary_ns(solver_durations_ns),
                "body_term_timing": _duration_summary_ns(body_term_durations_ns),
                "cyclic_gc_enabled_at_failure": gc.isenabled(),
                "publisher_sha256": publisher_sha256,
                "capture_worker_sha256": capture_worker_sha256,
                "pico_client_apk_sha256": pico_client_apk_sha256,
                "authorization": dict(_AUTHORIZATION),
            }
        )
        raise
    finally:
        if background_capture is not None:
            background_capture.request_stop()
        if worker is not None:
            worker.close()
        if background_capture is not None:
            background_capture.join()
        if socket is not None:
            socket.close(linger=0)
        if context is not None:
            context.term()
        if gc_pause_active:
            if gc_was_enabled:
                gc.enable()
            elif gc.isenabled():
                gc.disable()
        if capture_thread_switch_interval_active:
            sys.setswitchinterval(previous_thread_switch_interval_s)
        signal.signal(signal.SIGINT, previous_sigint_handler)
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        evidence.close()


if __name__ == "__main__":
    raise SystemExit(main())
