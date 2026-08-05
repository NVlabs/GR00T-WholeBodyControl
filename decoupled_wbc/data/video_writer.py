import os
import queue
import sys
import threading
import time

import av
import numpy as np


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "h264",
        buffer_size: int = 50,
    ):
        self.output_path = output_path
        self._first_frame = True  # Track first frame to suppress x264 info output

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.queue = queue.Queue(maxsize=buffer_size)
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self._closed = False
        self._thread = threading.Thread(target=self._writer_worker, daemon=True)
        self._thread.start()

    def _assert_dimensions(self, frame: np.ndarray) -> None:
        assert (
            frame.shape[1] == self.stream.width and frame.shape[0] == self.stream.height
        ), f"""Incorrect frame dimensions. Input dimensions: {frame.shape[1]}x{frame.shape[0]}. 
            Expected dimensions: {self.stream.width}x{self.stream.height}"""

    def add_frame(self, frame: np.ndarray) -> None:
        if self._closed:
            return
        self._assert_dimensions(frame)
        self.queue.put(frame)

    def _writer_worker(self) -> None:
        while True:
            frame = self.queue.get()
            if frame is None:
                break
            self._assert_dimensions(frame)
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

            # Suppress stderr for first frame encoding (x264 prints info then)
            if self._first_frame:
                stderr_fd = sys.stderr.fileno()
                old_stderr = os.dup(stderr_fd)
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, stderr_fd)
                try:
                    packets = self.stream.encode(frame)
                    for packet in packets:
                        self.container.mux(packet)
                finally:
                    os.dup2(old_stderr, stderr_fd)
                    os.close(old_stderr)
                    os.close(devnull)
                    self._first_frame = False
            else:
                packets = self.stream.encode(frame)
                for packet in packets:
                    self.container.mux(packet)

    def _drain_queue(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def _flush_stream(self) -> None:
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

    def stop(self) -> str:
        """
        Blocking call. Waits until all the frames in the queue have been written to the file
        and the video writer has been closed.
        """
        if self._closed:
            return self.output_path

        timeout_s = float(os.getenv("VIDEO_WRITER_STOP_TIMEOUT_S", "5.0"))
        start_ts = time.monotonic()
        if not self.queue.empty():
            print("Waiting for video writer queue to empty...")
            while not self.queue.empty():
                if timeout_s > 0 and (time.monotonic() - start_ts) >= timeout_s:
                    print("Video writer stop timeout reached, dropping remaining queued frames")
                    self._drain_queue()
                    break
                time.sleep(0.05)

        self.queue.put(None)
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            print("Video writer worker did not exit in time; forcing container close")

        print("Video writer queue is empty, flushing stream...")
        try:
            self._flush_stream()
        finally:
            self.container.close()
            self._closed = True
        return self.output_path

    def cancel(self) -> None:
        """Immediately stops writing and deletes the output file"""
        if self._closed:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
            return

        self._drain_queue()
        self.queue.put(None)
        self._thread.join(timeout=0.5)
        try:
            self.container.close()
        finally:
            self._closed = True
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def __del__(self) -> None:
        if hasattr(self, "container") and not getattr(self, "_closed", True):
            try:
                self.container.close()
            except Exception:
                pass
