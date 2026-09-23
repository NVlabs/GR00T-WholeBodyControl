import argparse
import logging
import os
from pathlib import Path
import struct
import sys
import warnings

import cv2
import numpy as np
import trimesh
import zmq


SCRIPT_DIR = Path(__file__).resolve().parent

# External FoundationPose repository containing models, weights, and modules.
# Update this one path if the dependency repository is moved in the future.
POSE_ESTIMATION_ROOT = Path("/home/athena/Camera/pose_estimation")
sys.path.insert(1, str(POSE_ESTIMATION_ROOT))

CUDA_HOME = Path("/usr/local/cuda-12.1")
if CUDA_HOME.exists():
    os.environ["CUDA_HOME"] = str(CUDA_HOME)
    os.environ["PATH"] = f"{CUDA_HOME / 'bin'}:{os.environ['PATH']}"
    os.environ["LD_LIBRARY_PATH"] = (
        f"{CUDA_HOME / 'lib64'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

from fastsam import FastSAM, FastSAMPrompt
from groundingdino.util.inference import load_model, predict
from dino_utils import post_process_result, transform_image
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import draw_posed_3d_box, draw_xyz_axis
import nvdiffrast.torch as dr


logging.disable(logging.INFO)
warnings.filterwarnings("ignore")


class ObjectDetector:
    """Continuously estimate one object's pose from the G1 RGB-D stream."""

    def __init__(
        self,
        obj_name,
        text_prompt="",
        endpoint="tcp://192.168.123.164:5555",
        pose_endpoint="tcp://127.0.0.1:5561",
        mesh_obb=False,
    ):
        self.device = "cuda"
        self.obj_name = obj_name
        self.text_prompt = text_prompt or obj_name
        self.endpoint = endpoint
        self.registered = False
        self.last_mask = None

        model_dir = POSE_ESTIMATION_ROOT / "models" / obj_name
        mesh_path = model_dir / "textured.obj"
        if not mesh_path.exists():
            raise ValueError(f"Object model not found: {mesh_path}")

        # Factory intrinsics of G1 head D435I 344522070363 at 640x480.
        self.cam_k = np.array(
            [
                [605.231384, 0.0, 324.900238],
                [0.0, 604.627441, 246.474838],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.fastsam = FastSAM(
            str(POSE_ESTIMATION_ROOT / "weights" / "FastSAM.pt")
        )
        self.dino = load_model(
            str(
                POSE_ESTIMATION_ROOT
                / "groundingdino"
                / "config"
                / "GroundingDINO_SwinT_OGC.py"
            ),
            str(
                POSE_ESTIMATION_ROOT
                / "weights"
                / "groundingdino_swint_ogc.pth"
            ),
        )

        self.mesh = trimesh.load(mesh_path)
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate(self.mesh.dump())
        self.foundation_pose = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            debug_dir=str(POSE_ESTIMATION_ROOT / "debug"),
            debug=0,
            glctx=dr.RasterizeCudaContext(),
        )

        if mesh_obb:
            self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        else:
            mins, maxs = self.mesh.bounds
            extents = maxs - mins
            self.to_origin = np.eye(4)
            self.to_origin[:3, 3] = -0.5 * (mins + maxs)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(
            2, 3
        )

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.connect(endpoint)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.pose_socket = self.context.socket(zmq.PUB)
        self.pose_socket.setsockopt(zmq.SNDHWM, 1)
        self.pose_socket.connect(pose_endpoint)

    def _receive_frame(self, timeout_ms=5000):
        events = dict(self.poller.poll(timeout_ms))
        if self.socket not in events:
            return None, None

        message = self.socket.recv_multipart()
        while True:
            try:
                message = self.socket.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break

        if len(message) != 3:
            raise RuntimeError(
                "Expected metadata, raw BGR, and raw depth from the G1; "
                f"received {len(message)} message parts"
            )

        metadata, color_bytes, depth_bytes = message
        (
            color_height,
            color_width,
            color_channels,
            depth_height,
            depth_width,
        ) = struct.unpack("!IIIII", metadata)

        expected_color_bytes = color_height * color_width * color_channels
        expected_depth_bytes = depth_height * depth_width * 2
        if len(color_bytes) != expected_color_bytes:
            raise RuntimeError("Received color byte count does not match metadata")
        if len(depth_bytes) != expected_depth_bytes:
            raise RuntimeError("Received depth byte count does not match metadata")

        bgr = np.frombuffer(color_bytes, dtype=np.uint8).reshape(
            color_height, color_width, color_channels
        )
        depth_raw = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(
            depth_height, depth_width
        )
        if bgr.shape[:2] != depth_raw.shape:
            raise RuntimeError(
                "FoundationPose requires one head RGB image and matching "
                f"aligned depth; received {bgr.shape[:2]} and {depth_raw.shape}"
            )
        return bgr, depth_raw

    def _segment_object(self, rgb):
        boxes, logits, phrases = predict(
            model=self.dino,
            image=transform_image(rgb),
            caption=self.text_prompt,
            box_threshold=0.35,
            text_threshold=0.25,
            device=self.device,
        )
        xyxy = post_process_result(rgb, boxes, logits, phrases)
        xyxy = xyxy.astype(np.int32).tolist()
        if not xyxy:
            return None

        everything = self.fastsam(
            rgb,
            device=self.device,
            retina_masks=True,
            imgsz=512,
            conf=0.4,
            iou=0.9,
            verbose=False,
        )
        prompt = FastSAMPrompt(rgb, everything, device=self.device)
        masks = prompt.box_prompt(bboxes=xyxy)
        if len(masks) == 0:
            return None
        return masks[0].astype(np.uint8)

    def detect(self):
        """Receive one current frame and register or track the object."""
        bgr, depth_raw = self._receive_frame()
        if bgr is None or depth_raw is None:
            print("No G1 RGB-D frame received")
            return None

        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        depth = depth_raw.astype(np.float32) * 0.001
        depth[(depth < 0.001) | (depth >= 5.0)] = 0.0

        if not self.registered:
            mask = self._segment_object(rgb)
            if mask is None:
                print("No object detected")
                return None
            pose = self.foundation_pose.register(
                K=self.cam_k,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=10,
            )
            self.last_mask = mask
            self.registered = True
            print("Object registered; continuing with FoundationPose tracking")
        else:
            pose = self.foundation_pose.track_one(
                rgb=rgb,
                depth=depth,
                K=self.cam_k,
                iteration=1,  # Previously 2; reduced for lower real-time latency.
            )

        center_pose = pose @ np.linalg.inv(self.to_origin)
        self.pose_socket.send_json(center_pose.tolist())
        result_rgb = draw_posed_3d_box(
            self.cam_k,
            img=rgb.copy(),
            ob_in_cam=center_pose,
            bbox=self.bbox,
        )
        result_rgb = draw_xyz_axis(
            result_rgb,
            ob_in_cam=center_pose,
            scale=0.1,
            K=self.cam_k,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )

        return (
            center_pose,
            self.bbox,
            result_rgb,
            rgb,
            depth,
            self.last_mask,
        )

    def reset(self):
        """Force detection, segmentation, and registration on the next frame."""
        self.registered = False
        self.last_mask = None

    def close(self):
        self.poller.unregister(self.socket)
        self.socket.close()
        self.pose_socket.close()
        self.context.term()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_name", nargs="?", default="trash_truck")
    parser.add_argument(
        "--text-prompt",
        default="green and white toy trash truck",
    )
    parser.add_argument(
        "--endpoint",
        default="tcp://192.168.123.164:5555",
    )
    parser.add_argument(
        "--pose-endpoint",
        default="tcp://127.0.0.1:5561",
    )
    args = parser.parse_args()

    detector = ObjectDetector(
        obj_name=args.obj_name,
        text_prompt=args.text_prompt,
        endpoint=args.endpoint,
        pose_endpoint=args.pose_endpoint,
    )

    try:
        while True:
            result = detector.detect()
            if result is None:
                continue
            pose, _, result_rgb, rgb, depth, mask = result

            depth_visible = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=50.0),
                cv2.COLORMAP_JET,
            )
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pose_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            mask_bgr = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
            matrix_bgr = np.zeros_like(rgb_bgr)
            empty_bgr = np.zeros_like(rgb_bgr)

            panels = {
                "RGB": rgb_bgr,
                "Depth": depth_visible,
                "Mask": mask_bgr,
                "Pose": pose_bgr,
                "Pose Matrix": matrix_bgr,
            }
            for label, panel in panels.items():
                cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
                cv2.putText(
                    panel,
                    label,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            matrix_panel = panels["Pose Matrix"]
            for row_index, row in enumerate(pose):
                matrix_text = " ".join(f"{value: 8.4f}" for value in row)
                cv2.putText(
                    matrix_panel,
                    matrix_text,
                    (24, 100 + row_index * 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            dashboard = np.vstack(
                (
                    np.hstack(
                        (panels["RGB"], panels["Depth"], panels["Pose Matrix"])
                    ),
                    np.hstack((panels["Mask"], panels["Pose"], empty_bgr)),
                )
            )
            cv2.imshow("G1 D435 FoundationPose Dashboard", dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                detector.reset()

            print("Object pose in G1 head-camera frame:")
            print(pose)
    finally:
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
