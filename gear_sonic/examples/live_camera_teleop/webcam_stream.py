# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-time GEM-X pose estimation from a live webcam -> SONIC (live-webcam path).

Captures frames from a webcam (or a video file used as a stand-in), runs the
GEM-X ONNX pipeline (YOLOX detect -> VitPose 2D -> GEM denoiser) over a rolling
sliding window, decodes per-frame SOMA body params, and (with --stream-sonic)
converts each frame SOMA->SMPL and publishes Protocol v3 to SONIC's smpl encoder.

Requires GEM-X (https://github.com/NVlabs/GEM-X) installed separately. Point to
its repo root with --gemx-root or the GEMX_ROOT environment variable.

Examples:
    # camera-only sanity check (no robot), saves an overlay video
    python webcam_stream.py --gemx-root /path/to/GEM-X --source 0 \
        --kp-only --save webcam_test.mp4 --max-frames 60

    # live teleop -> SONIC
    python webcam_stream.py --gemx-root /path/to/GEM-X --source 0 \
        --stream-sonic --window 30 --smooth 0.8
"""

# ruff: noqa: E402
import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")


def _add_gemx_to_path() -> str | None:
    """Resolve the GEM-X repo root (from --gemx-root or $GEMX_ROOT) and add it to
    sys.path so ``gem`` and GEM-X's ``scripts.demo`` modules import."""
    root = os.environ.get("GEMX_ROOT")
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--gemx-root" and i + 1 < len(argv):
            root = argv[i + 1]
        elif a.startswith("--gemx-root="):
            root = a.split("=", 1)[1]
    if root and root not in sys.path:
        sys.path.insert(0, root)
    return root


GEMX_ROOT = _add_gemx_to_path()

import cv2
import numpy as np
import torch

try:
    from scripts.demo.demo_soma_onnx import (
        load_denoiser, load_vitpose, run_denoiser_onnx, run_vitpose_onnx,
    )
    from gem.utils.cam_utils import estimate_K
    from gem.utils.geo_transform import compute_cam_angvel, get_bbx_xys_from_xyxy
    from gem.utils.pylogger import Log
except ModuleNotFoundError as e:
    raise SystemExit(
        "GEM-X not found (%s). Install https://github.com/NVlabs/GEM-X and pass "
        "--gemx-root <path> or set GEMX_ROOT to its repo root." % e
    )

# sibling module (same folder)
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _open_source(source: str):
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


class GemWebcamStreamer:
    """Rolling-window GEM-X inference over a live frame source."""

    def __init__(self, gemx_root, window=120, no_imgfeat=True, device="cuda"):
        self.gemx_root = gemx_root
        self.window = window
        self.no_imgfeat = no_imgfeat
        self.device = device

        from gem.utils.yolox_detector import YOLOXDetector
        self.yolox = YOLOXDetector(device=device)

        self.vitpose_runner, self.vitpose_backend = load_vitpose()
        self.denoiser_runner, self.denoiser_backend = load_denoiser(no_imgfeat=no_imgfeat)
        Log.info(f"[webcam] backends: vitpose={self.vitpose_backend}, denoiser={self.denoiser_backend}")

        self._endecoder = None
        self._get_body_params_w_Rt_v2 = None
        self.buf_kp2d = deque(maxlen=window)
        self.buf_bbx = deque(maxlen=window)
        self.K = None

    def _ensure_decoder(self, ckpt_path=None):
        if self._endecoder is not None:
            return
        import hydra
        from hydra import compose, initialize_config_dir
        from gem.pipeline.gem_pipeline import get_body_params_w_Rt_v2

        cfg_dir = str(Path(self.gemx_root) / "configs")
        with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
            cfg = compose(config_name="demo_soma", overrides=[
                "exp=gem_soma_regression", "video_name=stream", "video_path=stream",
                "use_wandb=false", "task=test",
            ])
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        if ckpt_path is None:
            from gem.utils.hf_utils import download_checkpoint
            ckpt_path = download_checkpoint()
        model.load_pretrained_model(ckpt_path)
        model = model.eval().to(self.device)
        self._endecoder = model.endecoder
        if self._endecoder.obs_indices_dict is None:
            self._endecoder.build_obs_indices_dict()
        self._get_body_params_w_Rt_v2 = get_body_params_w_Rt_v2

    def detect_bbox(self, frame_bgr, W, H):
        from gem.utils.yolox_detector import detect_and_track
        bbx_xyxy_np, _ = detect_and_track(frame_bgr[None], self.yolox)
        bbx_xyxy = torch.from_numpy(bbx_xyxy_np).float()
        bbx_xyxy[:, [0, 2]] = bbx_xyxy[:, [0, 2]].clamp(0, W - 1)
        bbx_xyxy[:, [1, 3]] = bbx_xyxy[:, [1, 3]].clamp(0, H - 1)
        return get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()[0]

    @torch.no_grad()
    def process_frame(self, frame_rgb, W, H, decode=True):
        """Ingest one frame; return (kp2d [77,3], soma_params or None)."""
        if self.K is None:
            self.K = estimate_K(W, H)

        bbx_xys = self.detect_bbox(frame_rgb, W, H)
        vitpose = run_vitpose_onnx(self.vitpose_runner, self.vitpose_backend,
                                   frame_rgb[None], bbx_xys[None])
        kp2d = vitpose[0] if isinstance(vitpose, tuple) else vitpose
        kp2d = torch.as_tensor(kp2d)[0]                   # [77,3]

        self.buf_kp2d.append(kp2d)
        self.buf_bbx.append(bbx_xys)
        if not decode or len(self.buf_kp2d) < 2:
            return kp2d, None

        L = len(self.buf_kp2d)
        obs = torch.stack(list(self.buf_kp2d)).unsqueeze(0)
        bbx = torch.stack(list(self.buf_bbx)).unsqueeze(0)
        K = self.K.repeat(L, 1, 1).unsqueeze(0)
        f_imgseq = torch.zeros(1, L, 1024)                # no-imgfeat
        cam_angvel = compute_cam_angvel(torch.eye(3).repeat(L, 1, 1)).unsqueeze(0)  # static cam

        batch = {"obs": obs, "bbx_xys": bbx, "K_fullimg": K,
                 "f_imgseq": f_imgseq, "f_cam_angvel": cam_angvel}
        pred_x, _pred_cam = run_denoiser_onnx(self.denoiser_runner, self.denoiser_backend, batch)

        self._ensure_decoder()
        decode_dict = self._endecoder.decode(pred_x)

        # WORLD (gravity-aligned) root orientation, like the offline pipeline.
        # decode_dict["global_orient"] is camera-frame; fuse to gravity-view so the
        # robot doesn't pitch/tip to a tilted camera "up".
        world_global_orient = decode_dict["global_orient"][0, -1].cpu()  # fallback
        if "global_orient_gv" in decode_dict and "local_transl_vel" in decode_dict:
            gp = self._get_body_params_w_Rt_v2(
                global_orient_gv=decode_dict["global_orient_gv"],
                local_transl_vel=decode_dict["local_transl_vel"],
                global_orient_c=decode_dict["global_orient"],
                cam_angvel=cam_angvel.to(pred_x.device),
            )
            world_global_orient = gp["global_orient"][0, -1].cpu()

        soma = {"body_pose": decode_dict["body_pose"][0, -1].cpu(),
                "global_orient": world_global_orient}
        for k in ("identity_coeffs", "scale_params"):
            if k in decode_dict:
                soma[k] = decode_dict[k][0, -1].cpu()
        return kp2d, soma


def _draw_overlay(frame_bgr, kp2d, conf_thr=0.4):
    for x, y, c in kp2d.numpy():
        if c > conf_thr:
            cv2.circle(frame_bgr, (int(x), int(y)), 2, (0, 255, 0), -1)
    return frame_bgr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gemx-root", default=GEMX_ROOT, help="GEM-X repo root (or set $GEMX_ROOT)")
    ap.add_argument("--source", default="0", help="Webcam index (e.g. 0) or video file path")
    ap.add_argument("--window", type=int, default=120, help="Rolling window length")
    ap.add_argument("--no-imgfeat", action="store_true", help="Skip SAM3DB image features (faster)")
    ap.add_argument("--kp-only", action="store_true", help="2D keypoints only (skip denoiser/decode)")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = run forever)")
    ap.add_argument("--show", action="store_true", help="Show cv2 preview window (needs a display)")
    ap.add_argument("--save", default=None, help="Optional path to save 2D-overlay preview mp4")
    ap.add_argument("--stream-sonic", action="store_true", help="Publish SMPL v3 stream to SONIC")
    ap.add_argument("--port", type=int, default=5556, help="ZMQ PUB port for SONIC stream")
    ap.add_argument("--sonic-root", default=None, help="SONIC repo root (only if run outside the repo)")
    ap.add_argument("--smooth", type=float, default=0.75,
                    help="Temporal smoothing of streamed SMPL (0=off, 0.6-0.85 smoother)")
    args = ap.parse_args()

    if not args.gemx_root:
        raise SystemExit("Provide --gemx-root <GEM-X repo root> or set GEMX_ROOT.")

    cap = _open_source(args.source)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    Log.info(f"[webcam] source={args.source} {W}x{H} @ ~{src_fps:.1f}fps")

    streamer = GemWebcamStreamer(args.gemx_root, window=args.window,
                                 no_imgfeat=args.no_imgfeat or True)

    converter, publisher = None, None
    if args.stream_sonic:
        from soma_to_smpl import SomaToSmpl, SonicV3Publisher
        from gem.utils.soma_utils.soma_layer import SomaLayer
        soma_layer = SomaLayer(data_root=str(Path(args.gemx_root) / "inputs" / "soma_assets"),
                               low_lod=True, device="cuda",
                               identity_model_type="mhr", mode="warp")
        converter = SomaToSmpl(soma_layer, device="cuda", smooth=args.smooth, sonic_root=args.sonic_root)
        publisher = SonicV3Publisher(port=args.port, sonic_root=args.sonic_root)
        Log.info(f"[webcam] streaming SMPL v3 to SONIC on tcp://*:{args.port}")

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))

    n, t_start, t_last = 0, time.time(), time.time()
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = frame_bgr[..., ::-1].copy()
            kp2d, soma = streamer.process_frame(frame_rgb, W, H, decode=not args.kp_only)

            if publisher is not None and soma is not None:
                publisher.publish(converter.convert(soma))

            n += 1
            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - t_last)
            t_last = now
            print(f"\r[webcam] frame {n} | {inst_fps:5.1f} fps | "
                  f"soma={'yes' if soma else 'warmup/kp-only'}", end="", flush=True)

            if writer is not None or args.show:
                vis = _draw_overlay(frame_bgr, kp2d)
                if writer is not None:
                    writer.write(vis)
                if args.show:
                    cv2.imshow("GEM webcam", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if publisher is not None:
            publisher.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        dur = time.time() - t_start
        print(f"\n[webcam] processed {n} frames in {dur:.1f}s ({n / max(1e-6, dur):.1f} fps avg)")


if __name__ == "__main__":
    main()
