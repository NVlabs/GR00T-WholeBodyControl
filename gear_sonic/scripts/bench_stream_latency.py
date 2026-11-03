#!/usr/bin/env python3
"""Benchmark ingest->decode latency through MediaMTX (unversioned test tool).

A producer encodes time.time() (us, 64 bits) as a grid of black/white blocks in
each frame and pipes rawvideo to ffmpeg, which pushes to MediaMTX via RTSP or
SRT. A reader opens rtsp://localhost:8554/<path> with OpenCV (same consumer as
gem_webcam_zmq_publisher) and decodes the timestamp to measure end-to-end
latency (encode + server + decode) on this machine.

Usage (GENMO venv):
    python gear_sonic/scripts/bench_stream_latency.py --protocol rtsp
    python gear_sonic/scripts/bench_stream_latency.py --protocol srt
"""

import argparse
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    "|probesize;32768|analyzeduration;0|max_delay;0|reorder_queue_size;0",
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

W, H, FPS = 640, 480, 30
BLOCK = 20  # px per bit; 32 bits per row, 2 rows = 64 bits
BITS = 64


def encode_ts(frame: np.ndarray, ts_us: int) -> None:
    for i in range(BITS):
        row, col = divmod(i, 32)
        val = 255 if (ts_us >> (BITS - 1 - i)) & 1 else 0
        y0, x0 = row * BLOCK, col * BLOCK
        frame[y0 : y0 + BLOCK, x0 : x0 + BLOCK] = val


def decode_ts(frame: np.ndarray) -> int:
    ts = 0
    for i in range(BITS):
        row, col = divmod(i, 32)
        y0, x0 = row * BLOCK, col * BLOCK
        block = frame[y0 + 4 : y0 + BLOCK - 4, x0 + 4 : x0 + BLOCK - 4]
        bit = 1 if block.mean() > 127 else 0
        ts = (ts << 1) | bit
    return ts


def producer(url: str, fmt_args: list, stop: threading.Event) -> None:
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-g", str(FPS), "-pix_fmt", "yuv420p",
        "-muxdelay", "0", "-muxpreload", "0", "-flush_packets", "1",
        *fmt_args, url,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:] = 60
    period = 1.0 / FPS
    next_t = time.monotonic()
    try:
        while not stop.is_set():
            encode_ts(frame, int(time.time() * 1e6))
            proc.stdin.write(frame.tobytes())
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except BrokenPipeError:
        pass
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.terminate()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["rtsp", "srt"], default="rtsp")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--drain", action="store_true", help="LatestFrameCapture-style reader")
    args = ap.parse_args()

    path = f"bench{args.protocol}"
    if args.protocol == "rtsp":
        push_url, fmt = f"rtsp://localhost:8554/{path}", ["-f", "rtsp"]
    else:
        push_url = f"srt://localhost:8890?streamid=publish:{path}"
        fmt = ["-f", "mpegts"]

    stop = threading.Event()
    t = threading.Thread(target=producer, args=(push_url, fmt, stop), daemon=True)
    t.start()
    time.sleep(2.0)  # let the stream register

    read_url = f"rtsp://localhost:8554/{path}"
    cap = cv2.VideoCapture(read_url, cv2.CAP_FFMPEG)
    deadline = time.time() + 10
    while not cap.isOpened() and time.time() < deadline:
        time.sleep(0.5)
        cap = cv2.VideoCapture(read_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"FAIL: could not open {read_url}")
        stop.set()
        return 1

    lat_ms, t_end = [], time.time() + args.seconds
    if args.drain:
        # LatestFrameCapture-style: grab continuously, sample newest at ~10 Hz.
        latest = {"frame": None}
        run = threading.Event()
        run.set()

        def _drain():
            while run.is_set():
                ok = cap.grab()
                if not ok:
                    break
                ok, f = cap.retrieve()
                if ok:
                    latest["frame"] = f

        dt = threading.Thread(target=_drain, daemon=True)
        dt.start()
        time.sleep(1.0)
        while time.time() < t_end:
            time.sleep(0.1)
            f = latest["frame"]
            if f is None:
                continue
            age_ms = (time.time() * 1e6 - decode_ts(f)) / 1e3
            if 0 < age_ms < 10_000:
                lat_ms.append(age_ms)
        run.clear()
    else:
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                break
            now_us = time.time() * 1e6
            ts = decode_ts(frame)
            age_ms = (now_us - ts) / 1e3
            if 0 < age_ms < 10_000:
                lat_ms.append(age_ms)
    cap.release()
    stop.set()

    if not lat_ms:
        print("FAIL: no decodable frames")
        return 1
    arr = np.array(lat_ms)
    print(
        f"[{args.protocol}] n={len(arr)} frames | "
        f"latency ms: mean={arr.mean():.0f} p50={np.percentile(arr, 50):.0f} "
        f"p90={np.percentile(arr, 90):.0f} p99={np.percentile(arr, 99):.0f} "
        f"min={arr.min():.0f} max={arr.max():.0f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
