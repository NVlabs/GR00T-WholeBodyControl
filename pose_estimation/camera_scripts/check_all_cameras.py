#!/usr/bin/python3

# /usr/bin/python3 /home/unitree/image_server/check_all_cameras.py --image
# /usr/bin/python3 /home/unitree/image_server/check_all_cameras.py --video
import argparse
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import cv2
import numpy as np
try:
    import pyrealsense2 as rs
except ImportError as error:
    raise SystemExit(
        "pyrealsense2 is unavailable in this Python environment.\n"
        "Run this script with /usr/bin/python3, for example:\n"
        "  /usr/bin/python3 "
        "/home/unitree/image_server/check_all_cameras.py --image"
    ) from error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "camera_test_output")


class FrameStore:
    def __init__(self):
        self._frames = {}
        self._labels = {}
        self._lock = threading.Lock()

    def register(self, key, label):
        with self._lock:
            self._labels[key] = label
            self._frames[key] = None

    def update(self, key, frame):
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )

        if not success:
            return

        with self._lock:
            self._frames[key] = encoded.tobytes()

    def get(self, key):
        with self._lock:
            return self._frames.get(key)

    def streams(self):
        with self._lock:
            return list(self._labels.items())


def list_realsense_devices():
    context = rs.context()
    cameras = []

    for device in context.query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)

        cameras.append(
            {
                "serial": serial,
                "name": name,
            }
        )

    return cameras


def select_head_camera(cameras, requested_serial=None):
    if requested_serial:
        for camera in cameras:
            if camera["serial"] == requested_serial:
                return camera

        raise RuntimeError(
            f"Head-camera serial {requested_serial} was not detected"
        )

    candidates = [
        camera
        for camera in cameras
        if "D435" in camera["name"].upper()
    ]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise RuntimeError(
            "No D435-series head camera detected; "
            "use --head-serial to select another RealSense camera"
        )

    raise RuntimeError(
        "Multiple D435-series cameras detected; "
        "use --head-serial to select the head camera"
    )


def camera_role(camera, head_serial):
    if camera["serial"] == head_serial:
        return "head"

    return "wrist"


def create_realsense_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(serial)

    config.enable_stream(
        rs.stream.color,
        640,
        480,
        rs.format.bgr8,
        30,
    )

    config.enable_stream(
        rs.stream.depth,
        640,
        480,
        rs.format.z16,
        30,
    )

    pipeline.start(config)

    return pipeline


def make_depth_grayscale(depth_image):
    grayscale = np.zeros(depth_image.shape, dtype=np.uint8)
    valid_depth = depth_image[depth_image > 0]

    if valid_depth.size == 0:
        return grayscale

    near, far = np.percentile(valid_depth, (2, 98))

    if far <= near:
        return grayscale

    scaled = (
        (depth_image.astype(np.float32) - near)
        * (255.0 / (far - near))
    )
    grayscale = np.clip(scaled, 0, 255).astype(np.uint8)
    grayscale[depth_image == 0] = 0

    return grayscale


def capture_realsense_images(cameras, head_serial):
    if not cameras:
        print("No RealSense cameras detected")
        return

    print(f"Found {len(cameras)} RealSense camera(s)")

    for camera in cameras:
        serial = camera["serial"]
        name = camera["name"]
        role = camera_role(camera, head_serial)

        print(f"\nTesting {role} camera: {name}")
        print(f"Serial: {serial}")

        pipeline = None

        try:
            pipeline = create_realsense_pipeline(serial)

            frames = None

            for _ in range(30):
                frames = pipeline.wait_for_frames(5000)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame:
                raise RuntimeError("No color frame received")

            if not depth_frame:
                raise RuntimeError("No depth frame received")

            color_image = np.asanyarray(
                color_frame.get_data()
            )

            depth_image = np.asanyarray(
                depth_frame.get_data()
            )

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(
                    depth_image,
                    alpha=0.03,
                ),
                cv2.COLORMAP_JET,
            )
            depth_grayscale = make_depth_grayscale(
                depth_image
            )

            filename_prefix = f"{role}_camera_{serial}"

            color_filename = os.path.join(
                OUTPUT_DIR,
                f"{filename_prefix}_color.jpg",
            )

            depth_raw_filename = os.path.join(
                OUTPUT_DIR,
                f"{filename_prefix}_depth_raw.png",
            )

            depth_filename = os.path.join(
                OUTPUT_DIR,
                f"{filename_prefix}_depth.jpg",
            )

            depth_gray_filename = os.path.join(
                OUTPUT_DIR,
                f"{filename_prefix}_depth_gray.jpg",
            )

            cv2.imwrite(color_filename, color_image)
            cv2.imwrite(depth_raw_filename, depth_image)
            cv2.imwrite(depth_filename, depth_colormap)
            cv2.imwrite(
                depth_gray_filename,
                depth_grayscale,
            )

            print(f"Saved color: {color_filename}")
            print(f"Saved raw depth: {depth_raw_filename}")
            print(f"Saved visible depth: {depth_filename}")
            print(
                f"Saved grayscale depth: "
                f"{depth_gray_filename}"
            )

        except Exception as error:
            print(
                f"ERROR while capturing RealSense "
                f"{serial}: {error}"
            )

        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass


def realsense_stream_worker(
    serial,
    frame_store,
    stop_event,
):
    pipeline = None
    color_key = f"realsense_{serial}_color"
    depth_key = f"realsense_{serial}_depth"

    try:
        print(f"Starting RealSense {serial}")
        pipeline = create_realsense_pipeline(serial)

        for _ in range(30):
            if stop_event.is_set():
                return

            pipeline.wait_for_frames(5000)

        while not stop_event.is_set():
            frames = pipeline.wait_for_frames(5000)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame:
                color_image = np.asanyarray(
                    color_frame.get_data()
                )
                frame_store.update(
                    color_key,
                    color_image,
                )

            if depth_frame:
                depth_image = np.asanyarray(
                    depth_frame.get_data()
                )

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(
                        depth_image,
                        alpha=0.03,
                    ),
                    cv2.COLORMAP_JET,
                )

                frame_store.update(
                    depth_key,
                    depth_colormap,
                )

    except Exception as error:
        print(
            f"RealSense {serial} streaming error: "
            f"{error}"
        )

    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

        print(f"RealSense {serial} stream stopped")


def make_http_handler(frame_store, stop_event):
    class CameraHTTPHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = unquote(self.path)

            if path == "/":
                self.send_index()
                return

            if path.startswith("/stream/"):
                stream_key = path[len("/stream/"):]
                self.send_stream(stream_key)
                return

            self.send_error(404, "Not found")

        def send_index(self):
            stream_elements = []

            for key, label in frame_store.streams():
                stream_elements.append(
                    f"""
                    <section class="camera">
                        <h2>{label}</h2>
                        <img
                            src="/stream/{key}"
                            alt="{label}"
                        >
                    </section>
                    """
                )

            page = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta
                    name="viewport"
                    content="width=device-width, initial-scale=1"
                >
                <title>G1 Camera Streams</title>
                <style>
                    body {{
                        font-family: sans-serif;
                        margin: 20px;
                        background: #eeeeee;
                    }}

                    h1 {{
                        text-align: center;
                    }}

                    .grid {{
                        display: grid;
                        grid-template-columns:
                            repeat(auto-fit, minmax(500px, 1fr));
                        gap: 20px;
                    }}

                    .camera {{
                        background: white;
                        padding: 12px;
                        border-radius: 8px;
                    }}

                    .camera h2 {{
                        margin-top: 0;
                    }}

                    .camera img {{
                        display: block;
                        width: 100%;
                        height: auto;
                        background: black;
                    }}
                </style>
            </head>
            <body>
                <h1>Unitree G1 Camera Streams</h1>
                <div class="grid">
                    {''.join(stream_elements)}
                </div>
            </body>
            </html>
            """

            encoded_page = page.encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8",
            )
            self.send_header(
                "Content-Length",
                str(len(encoded_page)),
            )
            self.end_headers()
            self.wfile.write(encoded_page)

        def send_stream(self, stream_key):
            known_keys = {
                key
                for key, _ in frame_store.streams()
            }

            if stream_key not in known_keys:
                self.send_error(
                    404,
                    f"Unknown stream: {stream_key}",
                )
                return

            self.send_response(200)
            self.send_header(
                "Cache-Control",
                "no-cache, private",
            )
            self.send_header(
                "Pragma",
                "no-cache",
            )
            self.send_header(
                "Connection",
                "close",
            )
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; "
                "boundary=frame",
            )
            self.end_headers()

            try:
                while not stop_event.is_set():
                    jpeg = frame_store.get(stream_key)

                    if jpeg is None:
                        time.sleep(0.05)
                        continue

                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(
                        b"Content-Type: image/jpeg\r\n"
                    )
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                        .encode("ascii")
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

                    time.sleep(1.0 / 30.0)

            except (
                BrokenPipeError,
                ConnectionResetError,
            ):
                pass

        def log_message(self, format_string, *args):
            return

    return CameraHTTPHandler


def run_video_server(head_serial, port):
    frame_store = FrameStore()
    stop_event = threading.Event()
    threads = []

    cameras = list_realsense_devices()

    print(
        f"Found {len(cameras)} RealSense camera(s)"
    )

    for camera in cameras:
        serial = camera["serial"]
        name = camera["name"]
        role = camera_role(camera, head_serial).title()

        frame_store.register(
            f"realsense_{serial}_color",
            f"{role} camera — {name} {serial} — Color",
        )

        frame_store.register(
            f"realsense_{serial}_depth",
            f"{role} camera — {name} {serial} — Depth",
        )

    for camera in cameras:
        serial = camera["serial"]

        thread = threading.Thread(
            target=realsense_stream_worker,
            args=(
                serial,
                frame_store,
                stop_event,
            ),
            daemon=True,
        )

        threads.append(thread)
        thread.start()

    handler = make_http_handler(
        frame_store,
        stop_event,
    )

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        handler,
    )

    print()
    print(
        f"Open http://192.168.123.164:{port} "
        f"in the host computer's browser."
    )
    print("Press Ctrl+C to stop.")
    print()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nStopping camera streams...")

    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()

        for thread in threads:
            thread.join(timeout=5)

        print("Camera server stopped")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Capture images or continuously stream "
            "the Unitree G1 cameras"
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--image",
        action="store_true",
        help="Capture and save one image from each camera",
    )

    mode.add_argument(
        "--video",
        action="store_true",
        help="Continuously stream all cameras over HTTP",
    )

    parser.add_argument(
        "--head-serial",
        help=(
            "RealSense serial for the head camera "
            "(default: automatically select the D435-series camera)"
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP server port for --video",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    cameras = list_realsense_devices()

    if not cameras:
        raise RuntimeError("No RealSense cameras detected")

    head_camera = select_head_camera(
        cameras,
        requested_serial=args.head_serial,
    )
    head_serial = head_camera["serial"]

    print(
        f"Head camera: {head_camera['name']} "
        f"(serial {head_serial})"
    )

    if args.image:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"Output directory: {OUTPUT_DIR}")
        capture_realsense_images(cameras, head_serial)

        print("\nFinished capturing images")

    elif args.video:
        run_video_server(
            head_serial=head_serial,
            port=args.port,
        )


if __name__ == "__main__":
    main()
