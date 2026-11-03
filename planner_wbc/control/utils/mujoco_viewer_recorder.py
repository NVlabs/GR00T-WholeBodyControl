from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import numpy as np


_STOP_WRITER = object()


class _AsyncVideoWriter:
    def __init__(self, output_path: Path, width: int, height: int, fps: float):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.queue: queue.Queue[np.ndarray | object] = queue.Queue(maxsize=max(30, int(fps * 3)))
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.dropped_frames = 0
        self.thread = threading.Thread(target=self._run, name="sim-video-writer", daemon=True)

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        if not self.ready.wait(timeout=5.0):
            raise RuntimeError("Timed out while opening the simulation video writer")
        if self.error is not None:
            raise RuntimeError(f"Could not open simulation video: {self.error}") from self.error

    def add_frame(self, rgb_frame: np.ndarray) -> None:
        if self.error is not None:
            raise RuntimeError(f"Simulation video writer failed: {self.error}") from self.error
        try:
            self.queue.put_nowait(rgb_frame)
        except queue.Full:
            self.dropped_frames += 1

    def stop(self) -> None:
        if self.thread.is_alive():
            while self.thread.is_alive():
                try:
                    self.queue.put(_STOP_WRITER, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"Simulation video writer failed: {self.error}") from self.error

    def _run(self) -> None:
        writer = None
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(self.output_path), fourcc, self.fps, (self.width, self.height)
            )
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not initialize the MP4 encoder")
            self.ready.set()

            while True:
                item = self.queue.get()
                try:
                    if item is _STOP_WRITER:
                        break
                    writer.write(cv2.cvtColor(item, cv2.COLOR_RGB2BGR))
                finally:
                    self.queue.task_done()
        except Exception as exc:
            self.error = exc
            self.ready.set()
        finally:
            if writer is not None:
                writer.release()


class MujocoViewerRecorder:
    """Toggle video recording from the current interactive MuJoCo viewer camera."""

    def __init__(self, sim_env, output_dir: str, fps: float):
        if fps <= 0:
            raise ValueError("Simulation video fps must be greater than zero")

        self.sim_env = sim_env
        self.viewer = sim_env.viewer
        self.model = sim_env.mj_model
        self.output_dir = Path(output_dir).expanduser()
        self.fps = float(fps)

        self._state_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._recording = False
        self._writer: _AsyncVideoWriter | None = None
        self._renderer: mujoco.Renderer | None = None
        self._renderer_size: tuple[int, int] | None = None
        self._renderer_thread_id: int | None = None
        self._next_frame_time = 0.0
        self._original_update_viewer = None
        self._shutdown_renderer = False
        self._renderer_closed = threading.Event()

    @property
    def is_recording(self) -> bool:
        with self._state_lock:
            return self._recording

    def attach(self) -> None:
        if self._original_update_viewer is not None:
            return

        self._original_update_viewer = self.sim_env.update_viewer

        def update_viewer_and_record() -> None:
            self._original_update_viewer()
            self.capture_frame()

        self.sim_env.update_viewer = update_viewer_and_record

    def handle_keyboard_button(self, key: str) -> None:
        if key != "c":
            return
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> Path | None:
        with self._state_lock:
            if self._recording:
                return None

        width, height = self._recording_size()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = (self.output_dir / f"g1_sim_{timestamp}.mp4").resolve()
        writer = _AsyncVideoWriter(output_path, width, height, self.fps)

        try:
            writer.start()
        except Exception as exc:
            print(f"Could not start simulation video recording: {exc}")
            return None

        with self._state_lock:
            self._writer = writer
            self._recording = True
            self._next_frame_time = 0.0

        print(
            f"Simulation video recording started: {output_path} "
            f"({width}x{height} at {self.fps:g} FPS). Press c again to stop."
        )
        return output_path

    def stop_recording(self) -> Path | None:
        with self._state_lock:
            if not self._recording:
                return None
            self._recording = False
            writer = self._writer
            self._writer = None

        # Wait until an in-progress render has handed its frame to the writer.
        with self._capture_lock:
            pass

        if writer is None:
            return None

        try:
            writer.stop()
        except Exception as exc:
            print(f"Simulation video recording failed: {exc}")
            return None

        dropped = (
            f" ({writer.dropped_frames} frames dropped because the encoder was busy)"
            if writer.dropped_frames
            else ""
        )
        print(f"Simulation video saved: {writer.output_path}{dropped}")
        return writer.output_path

    def capture_frame(self) -> None:
        if self._close_renderer_if_requested():
            return

        now = time.monotonic()
        with self._state_lock:
            if not self._recording or now < self._next_frame_time:
                return
            writer = self._writer
            next_frame_time = self._next_frame_time + 1.0 / self.fps
            self._next_frame_time = (
                now + 1.0 / self.fps if next_frame_time <= now else next_frame_time
            )

        if writer is None:
            return

        with self._capture_lock:
            if not self.is_recording:
                return
            try:
                self._ensure_renderer(writer.width, writer.height)
                camera = self._copy_viewer_camera()
                self._renderer.update_scene(self.sim_env.mj_data, camera=camera)
                writer.add_frame(self._renderer.render().copy())
            except Exception as exc:
                print(f"Simulation video capture failed: {exc}")
                with self._state_lock:
                    self._recording = False
                    if self._writer is writer:
                        self._writer = None
                try:
                    writer.stop()
                except Exception as writer_exc:
                    print(f"Simulation video writer cleanup failed: {writer_exc}")

    def close(self) -> None:
        self.stop_recording()

        if self._renderer is not None:
            if self._renderer_thread_id == threading.get_ident():
                self._close_renderer()
            else:
                self._renderer_closed.clear()
                self._shutdown_renderer = True
                self._renderer_closed.wait(timeout=2.0)

        if self._original_update_viewer is not None:
            self.sim_env.update_viewer = self._original_update_viewer
            self._original_update_viewer = None

    def _recording_size(self) -> tuple[int, int]:
        with self.viewer.lock():
            viewport = self.viewer.viewport
            viewport_width = int(viewport.width) if viewport is not None else 0
            viewport_height = int(viewport.height) if viewport is not None else 0

        max_width = int(self.model.vis.global_.offwidth)
        max_height = int(self.model.vis.global_.offheight)
        if viewport_width <= 0 or viewport_height <= 0:
            viewport_width, viewport_height = max_width, max_height

        scale = min(1.0, max_width / viewport_width, max_height / viewport_height)
        width = max(2, int(viewport_width * scale) // 2 * 2)
        height = max(2, int(viewport_height * scale) // 2 * 2)
        return width, height

    def _ensure_renderer(self, width: int, height: int) -> None:
        size = (width, height)
        if self._renderer is not None and self._renderer_size == size:
            return
        self._close_renderer()
        self._renderer = mujoco.Renderer(self.model, width=width, height=height)
        self._renderer_size = size
        self._renderer_thread_id = threading.get_ident()

    def _copy_viewer_camera(self) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        with self.viewer.lock():
            source = self.viewer.cam
            camera.azimuth = source.azimuth
            camera.distance = source.distance
            camera.elevation = source.elevation
            camera.fixedcamid = source.fixedcamid
            camera.lookat[:] = source.lookat
            camera.orthographic = source.orthographic
            camera.trackbodyid = source.trackbodyid
            camera.type = source.type
        return camera

    def _close_renderer_if_requested(self) -> bool:
        if not self._shutdown_renderer:
            return False
        with self._capture_lock:
            self._close_renderer()
            self._shutdown_renderer = False
            self._renderer_closed.set()
        return True

    def _close_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_size = None
            self._renderer_thread_id = None
