# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline driver: stream a saved GEM-X SOMA result to SONIC as SMPL (Protocol v3).

Replays ``hpe_results.pt`` (the SOMA body params GEM-X already produced for a
video) through the SOMA->SMPL converter and publishes the v3 stream SONIC's
``smpl`` encoder expects. Useful to verify the SMPL path against the SONIC sim
WITHOUT a camera.

Requires GEM-X (https://github.com/NVlabs/GEM-X); pass --gemx-root or set GEMX_ROOT.

Example:
    python soma_pt_to_sonic_v3.py --gemx-root /path/to/GEM-X \
        --pt /path/to/hpe_results.pt --fps 30 --loop
"""

import argparse
import functools
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _add_gemx_to_path() -> str | None:
    root = os.environ.get("GEMX_ROOT")
    for i, a in enumerate(sys.argv):
        if a == "--gemx-root" and i + 1 < len(sys.argv):
            root = sys.argv[i + 1]
        elif a.startswith("--gemx-root="):
            root = a.split("=", 1)[1]
    if root and root not in sys.path:
        sys.path.insert(0, root)
    return root


GEMX_ROOT = _add_gemx_to_path()
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules
torch.load = functools.partial(torch.load, weights_only=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gemx-root", default=GEMX_ROOT, help="GEM-X repo root (or set $GEMX_ROOT)")
    ap.add_argument("--pt", required=True, help="Path to hpe_results.pt")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--sonic-root", default=None, help="SONIC repo root (only if run outside the repo)")
    ap.add_argument("--smooth", type=float, default=0.0, help="Temporal smoothing (0=off)")
    ap.add_argument("--dry-run", action="store_true", help="Convert frame 0 and print, no ZMQ")
    args = ap.parse_args()

    if not args.gemx_root:
        raise SystemExit("Provide --gemx-root <GEM-X repo root> or set GEMX_ROOT.")

    try:
        from gem.utils.soma_utils.soma_layer import SomaLayer
    except ModuleNotFoundError as e:
        raise SystemExit(
            "GEM-X not found (%s). Install https://github.com/NVlabs/GEM-X and pass "
            "--gemx-root <path> or set GEMX_ROOT." % e
        )
    from soma_to_smpl import SomaToSmpl, SonicV3Publisher

    pred = torch.load(args.pt)
    g = pred["body_params_global"]
    T = g["body_pose"].shape[0]
    print(f"[soma->v3] loaded {T} frames from {args.pt}")

    soma = SomaLayer(data_root=str(Path(args.gemx_root) / "inputs" / "soma_assets"),
                     low_lod=True, device="cuda", identity_model_type="mhr", mode="warp")
    conv = SomaToSmpl(soma, device="cuda", smooth=args.smooth, sonic_root=args.sonic_root)

    def frame_at(t):
        return {k: g[k][t] for k in ("body_pose", "global_orient", "identity_coeffs", "scale_params")}

    if args.dry_run:
        out = conv.convert(frame_at(0))
        for k, v in out.items():
            print(k, v.shape)
        return

    pub = SonicV3Publisher(port=args.port, sonic_root=args.sonic_root)
    print(f"[soma->v3] publishing 'pose' v3 on tcp://*:{args.port}; enable streaming (ENTER) on deploy")
    time.sleep(1.0)
    dt = 1.0 / args.fps
    sent = 0
    try:
        while True:
            for t in range(T):
                pub.publish(conv.convert(frame_at(t)))
                sent += 1
                if t % int(max(1, args.fps)) == 0:
                    print(f"\r[soma->v3] frame {t + 1}/{T} (sent {sent})", end="")
                time.sleep(dt)
                if args.max_frames and sent >= args.max_frames:
                    return
            if not args.loop:
                break
            print("\n[soma->v3] loop restart")
    except KeyboardInterrupt:
        print("\n[soma->v3] stopped")
    finally:
        pub.close()
        print("\n[soma->v3] done")


if __name__ == "__main__":
    main()
