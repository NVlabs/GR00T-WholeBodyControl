"""Offloads rendered MuJoCo camera images to a subprocess via shared memory + ZMQ."""

import multiprocessing as mp
from multiprocessing import shared_memory
import time
from typing import Any, Dict

import numpy as np

from gear_sonic.utils.mujoco_sim.sensor_server import ImageMessageSchema, SensorServer


def get_multiprocessing_info(verbose: bool = True):
    """Get information about multiprocessing start methods"""

    if verbose:
        print(f"Available start methods: {mp.get_all_start_methods()}")
    return mp.get_start_method()


class ImagePublishProcess:
    """Subprocess for publishing images using shared memory and ZMQ"""

    def __init__(
        self,
        camera_configs: Dict[str, Any],
        image_dt: float,
        zmq_port: int = 5555,
        start_method: str = "spawn",
        verbose: bool = False,
    ):
        self.camera_configs = camera_configs
        self.image_dt = image_dt
        self.zmq_port = zmq_port
        self.verbose = verbose
        self.shared_memory_blocks = {}
        self.shared_memory_info = {}
        self.process = None

        self.mp_context = mp.get_context(start_method)
        if self.verbose:
            print(f"Using multiprocessing context: {start_method}")

        self.stop_event = self.mp_context.Event()
        self.data_ready_event = self.mp_context.Event()

        self.stop_event.clear()
        self.data_ready_event.clear()

        for camera_name, camera_config in camera_configs.items():
            height = camera_config["height"]
            width = camera_config["width"]
            size = height * width * 3

            shm = shared_memory.SharedMemory(create=True, size=size)
            self.shared_memory_blocks[camera_name] = shm
            self.shared_memory_info[camera_name] = {
                "name": shm.name,
                "size": size,
                "shape": (height, width, 3),
                "dtype": np.uint8,
            }

    def start_process(self):
        """Start the image publishing subprocess"""
        self.process = self.mp_context.Process(
            target=self._image_publish_worker,
            args=(
                self.shared_memory_info,
                self.image_dt,
                self.zmq_port,
                self.stop_event,
                self.data_ready_event,
                self.verbose,
            ),
        )
        self.process.start()

    def update_shared_memory(self, render_caches: Dict[str, np.ndarray]):
        """Update shared memory with new rendered images"""
        images_updated = 0
        for camera_name in self.camera_configs.keys():
            image_key = f"{camera_name}_image"
            if image_key in render_caches:
                image = render_caches[image_key]

                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)

                shm = self.shared_memory_blocks[camera_name]
                shared_array = np.ndarray(
                    self.shared_memory_info[camera_name]["shape"],
                    dtype=self.shared_memory_info[camera_name]["dtype"],
                    buffer=shm.buf,
                )

                np.copyto(shared_array, image)
                images_updated += 1

        if images_updated > 0:
            self.data_ready_event.set()

    def stop(self):
        """Stop the image publishing subprocess"""
        self.stop_event.set()

        if self.process and self.process.is_alive():
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join()

        for camera_name, shm in self.shared_memory_blocks.items():
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                print(f"Warning: Failed to cleanup shared memory for {camera_name}: {e}")

        self.shared_memory_blocks.clear()

    @staticmethod
    def _image_publish_worker(
        shared_memory_info, image_dt, zmq_port, stop_event, data_ready_event, verbose
    ):
        """Worker function that runs in the subprocess"""
        try:
            sensor_server = SensorServer()
            sensor_server.start_server(port=zmq_port)

            shared_arrays = {}
            shm_blocks = {}
            for camera_name, info in shared_memory_info.items():
                shm = shared_memory.SharedMemory(name=info["name"])
                shm_blocks[camera_name] = shm
                shared_arrays[camera_name] = np.ndarray(
                    info["shape"], dtype=info["dtype"], buffer=shm.buf
                )

            print(
                f"Image publishing subprocess started with {len(shared_arrays)} cameras "
                f"on ZMQ port {zmq_port}"
            )

            loop_count = 0
            last_data_time = time.time()

            while not stop_event.is_set():
                loop_count += 1

                timeout = min(image_dt, 0.1)
                data_available = data_ready_event.wait(timeout=timeout)

                current_time = time.time()

                if data_available:
                    data_ready_event.clear()
                    if loop_count % 50 == 0:
                        print("Image publish frequency: ", 1 / (current_time - last_data_time))
                    last_data_time = current_time

                    try:
                        from gear_sonic.utils.mujoco_sim.sensor_server import ImageUtils

                        image_copies = {name: arr.copy() for name, arr in shared_arrays.items()}

                        message_dict = {
                            "images": image_copies,
                            "timestamps": {name: current_time for name in image_copies.keys()},
                        }

                        image_msg = ImageMessageSchema(
                            timestamps=message_dict.get("timestamps"),
                            images=message_dict.get("images", None),
                        )

                        serialized_data = image_msg.serialize()

                        for camera_name, image_copy in image_copies.items():
                            serialized_data[f"{camera_name}"] = ImageUtils.encode_image(image_copy)

                        sensor_server.send_message(serialized_data)

                    except Exception as e:
                        print(f"Error publishing images: {e}")

                if not data_available:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            print("Image publisher interrupted by user")
        finally:
            try:
                for shm in shm_blocks.values():
                    shm.close()
                sensor_server.stop_server()
            except Exception as e:
                print(f"Error during subprocess cleanup: {e}")
