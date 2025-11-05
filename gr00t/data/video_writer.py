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
        thread = threading.Thread(target=self._writer_worker, daemon=True)
        thread.start()

    def _assert_dimensions(self, frame: np.ndarray) -> None:
        assert (
            frame.shape[1] == self.stream.width and frame.shape[0] == self.stream.height
        ), f"""Incorrect frame dimensions. Input dimensions: {frame.shape[1]}x{frame.shape[0]}. 
            Expected dimensions: {self.stream.width}x{self.stream.height}"""

    def add_frame(self, frame: np.ndarray) -> None:
        self._assert_dimensions(frame)
        self.queue.put(frame)

    def _writer_worker(self) -> None:
        while True:
            frame = self.queue.get()
            if frame is None:
                continue
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

    def _flush_stream(self) -> None:
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

    def stop(self) -> str:
        """
        Blocking call. Waits until all the frames in the queue have been written to the file
        and the video writer has been closed.
        """
        if not self.queue.empty():
            print("Waiting for video writer queue to empty...")
            while not self.queue.empty():
                time.sleep(0.1)

        print("Video writer queue is empty, flushing stream...")
        self._flush_stream()
        self.container.close()
        return self.output_path

    def cancel(self) -> None:
        """Immediately stops writing and deletes the output file"""
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        self.container.close()

    def __del__(self) -> None:
        self.container.close()
