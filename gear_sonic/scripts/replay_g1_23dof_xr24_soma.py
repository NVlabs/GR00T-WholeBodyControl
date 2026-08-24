"""Capture/replay hardened PICO XR24 through pinned SOMA without robot I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any
import uuid

from gear_sonic.utils.g1_23dof_contract import (
    DEFAULT_REFERENCE_PROFILE,
    REFERENCE_PROFILES,
    reference_profile_contract,
)
from gear_sonic.utils.g1_23dof_live_shadow import build_encoder_observation
from gear_sonic.utils.g1_23dof_semantic_reference import (
    build_stream_reference_window,
    true23_command_from_window,
)
from gear_sonic.utils.g1_23dof_xr24_soma_adapter import (
    complete_encoder_terms,
    replay_capture,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, value: Any) -> None:
    payload = _canonical_json_bytes(value)
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        resolved,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        resolved.unlink(missing_ok=True)
        raise


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hardened XR24 capture -> pinned NVIDIA SOMA -> exact 50 Hz "
            "MJ29/IL29 semantic replay. Opens no Unitree, DDS, or ZMQ channel."
        )
    )
    parser.add_argument(
        "--soma-source-root",
        type=Path,
        default=Path("/root/.cache/g1_true23_soma/source"),
    )
    parser.add_argument(
        "--profile",
        choices=sorted(REFERENCE_PROFILES),
        default=DEFAULT_REFERENCE_PROFILE,
    )
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--capture-live", action="store_true")
    parser.add_argument("--pico-client-apk-sha256")
    parser.add_argument(
        "--service-binary",
        type=Path,
        default=Path("/opt/apps/roboticsservice/RoboticsServiceProcess"),
    )
    parser.add_argument("--frames", type=int, default=70)
    parser.add_argument("--frame-timeout-s", type=float, default=5.0)
    parser.add_argument("--session-id")
    parser.add_argument(
        "--capture-python",
        type=Path,
        default=Path("/usr/bin/python3"),
        help="Python 3.10 executable matching the XRoboToolkit extension ABI.",
    )
    parser.add_argument(
        "--xrt-binding-dir",
        type=Path,
        default=Path(
            "/root/GR00T-WholeBodyControl/external_dependencies/"
            "XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
        ),
    )
    parser.add_argument(
        "--robot-anchor-quat-wxyz",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        help=(
            "Measured robot pelvis/IMU quaternion aligned to playback time. "
            "Required to materialize complete 267-value encoder terms."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _capture_live(args: argparse.Namespace) -> dict[str, Any]:
    if not args.pico_client_apk_sha256:
        raise ValueError("--capture-live requires --pico-client-apk-sha256")
    service_binary = args.service_binary.resolve()
    if not service_binary.is_file():
        raise FileNotFoundError(
            f"XRoboToolkit service binary missing: {service_binary}"
        )
    capture_python = args.capture_python.resolve()
    binding_dir = args.xrt_binding_dir.resolve()
    if not capture_python.is_file():
        raise FileNotFoundError(f"capture Python missing: {capture_python}")
    if not binding_dir.is_dir():
        raise FileNotFoundError(f"XRoboToolkit binding dir missing: {binding_dir}")
    repository_root = Path(__file__).resolve().parents[2]
    capture_script = (
        repository_root / "gear_sonic" / "scripts" / "capture_g1_23dof_pico_raw.py"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository_root), str(binding_dir))
    )
    with tempfile.TemporaryDirectory(prefix="g1_true23_xr24_") as temp_dir:
        capture_path = Path(temp_dir) / "capture.json"
        command = [
            str(capture_python),
            str(capture_script),
            "--output",
            str(capture_path),
            "--pico-client-apk-sha256",
            args.pico_client_apk_sha256,
            "--service-binary",
            str(service_binary),
            "--frames",
            str(args.frames),
            "--frame-timeout-s",
            str(args.frame_timeout_s),
            "--session-id",
            args.session_id or f"pico-xr24-{uuid.uuid4()}",
        ]
        subprocess.run(
            command,
            check=True,
            env=environment,
            timeout=max(30.0, args.frames * args.frame_timeout_s + 10.0),
        )
        return _load_json(capture_path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.capture is None) == (not args.capture_live):
        raise ValueError("choose exactly one of --capture or --capture-live")
    output_dir = args.output_dir.resolve()
    outputs = {
        "capture": output_dir / "xr24_capture.json",
        "trace": output_dir / "soma_mj29_il29_trace.json",
        "semantic_frames": output_dir / "semantic_frames.json",
        "encoder_reference": output_dir / "encoder_reference_terms.json",
        "report": output_dir / "replay_report.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))

    capture = _capture_live(args) if args.capture_live else _load_json(args.capture)
    replay_started_ns = time.monotonic_ns()
    trace, frames, body_terms, report = replay_capture(
        capture,
        soma_source_root=args.soma_source_root.resolve(),
        profile=args.profile,
    )
    replay_finished_ns = time.monotonic_ns()
    emitted_ns = max(frame["capture_monotonic_ns"] for frame in frames)
    window = build_stream_reference_window(
        frames,
        profile=args.profile,
        emitted_monotonic_ns=emitted_ns,
    )
    playback_index = window["playback"]["frame_index"]
    body_term = next(
        term
        for term in body_terms
        if term["source_frame_index"] == playback_index
    )
    partial_encoder_reference = {
        "schema_version": 1,
        "kind": "g1_true23_xr24_soma_encoder_reference_terms",
        "source_session_id": capture["session_id"],
        "source_frame_index": window["playback"]["frame_index"],
        "source_monotonic_ns": window["playback"]["frame_monotonic_ns"],
        "emitted_monotonic_ns": emitted_ns,
        "measured_delay_ns": (
            emitted_ns - window["playback"]["frame_monotonic_ns"]
        ),
        "future_frame_offsets_s": window["future_frame_offsets_s"],
        "command_multi_future_lower_body": true23_command_from_window(window),
        "vr_3point_local_target": body_term["vr_3point_local_target"],
        "vr_3point_local_orn_target": body_term[
            "vr_3point_local_orn_target"
        ],
        "reference_anchor_quaternion_xyzw": body_term[
            "reference_anchor_quaternion_xyzw"
        ],
        "semantic_window_sha256": _sha256(window),
        "complete_encoder_terms": False,
        "missing_terms": ["motion_anchor_ori_b_relative_to_live_robot"],
        "authorization": {
            "read_only": True,
            "dds_opened": False,
            "robot_channel_opened": False,
            "actuation_authorized": False,
            "robot_commands_published": False,
        },
    }
    encoder_reference = (
        complete_encoder_terms(
            reference_window=window,
            body_term=body_term,
            robot_anchor_quaternion_wxyz=args.robot_anchor_quat_wxyz,
        )
        if args.robot_anchor_quat_wxyz is not None
        else partial_encoder_reference
    )
    encoder_input = (
        build_encoder_observation(
            encoder_reference,
            expected_reference_contract=reference_profile_contract(args.profile),
        )
        if args.robot_anchor_quat_wxyz is not None
        else None
    )
    report = dict(report)
    report.update(
        {
            "semantic_window_sha256": _sha256(window),
            "encoder_reference_sha256": _sha256(encoder_reference),
            "semantic_measured_delay_ns": partial_encoder_reference[
                "measured_delay_ns"
            ],
            "batch_compute_duration_ns": replay_finished_ns - replay_started_ns,
            "batch_compute_lag_ns": (
                replay_finished_ns - emitted_ns if args.capture_live else None
            ),
            "complete_encoder_terms": args.robot_anchor_quat_wxyz is not None,
            "encoder_input_dim": (
                len(encoder_input) if encoder_input is not None else None
            ),
            "encoder_input_sha256": (
                _sha256(encoder_input) if encoder_input is not None else None
            ),
            "continuous_50hz_live_proven": False,
            "continuous_live_blocker": (
                "pinned SOMA exposes an offline batch pipeline; rolling 50 Hz "
                "deadline and cross-batch continuity remain unproven"
            ),
        }
    )

    _write_exclusive(outputs["capture"], capture)
    _write_exclusive(outputs["trace"], trace)
    _write_exclusive(outputs["semantic_frames"], frames)
    _write_exclusive(outputs["encoder_reference"], encoder_reference)
    _write_exclusive(outputs["report"], report)
    print(
        _canonical_json_bytes(
            {
                "status": "exact_read_only_replay_complete_nonpromotable",
                "outputs": {key: str(path) for key, path in outputs.items()},
                "report": report,
            }
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
