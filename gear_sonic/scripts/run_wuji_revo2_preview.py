#!/usr/bin/env python3
"""Read Wuji Glove frames and preview Revo2 targets without hardware writes.

The script intentionally has no Revo2 transport.  It reuses the existing
Revo-Retargeting ``DexRetargeter`` when that repository and its dependencies
are supplied, then prints the bounded, named six-channel target produced by
``WujiRevo2Bridge``.  This is the first live gate before assigning a unique
Revo2 hardware owner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from gear_sonic.utils.teleop.wuji_revo2 import WujiGloveSource, WujiRevo2Bridge


def _build_dex_backend(revo_root: Path, config_path: Path, side: str):
    package_root = revo_root / "src" / "brainco_capabilities" / "manus_revo2_retarget"
    if not package_root.is_dir():
        raise FileNotFoundError(f"Revo-Retargeting package root does not exist: {package_root}")
    sys.path.insert(0, str(package_root))
    from manus_revo2_retarget.retargeters.dex_retargeter import DexRetargeter

    return DexRetargeter(config_path, enabled_sides=(side,))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-sn", required=True, help="Wuji Glove serial number")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument(
        "--revo-root",
        type=Path,
        required=True,
        help="Checkout of Revo-Retargeting; used read-only as the mature retarget backend",
    )
    parser.add_argument(
        "--retarget-config",
        type=Path,
        required=True,
        help="Path to the backend's retarget.yaml",
    )
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--stale-after-sec", type=float, default=0.12)
    parser.add_argument("--print-every-sec", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = _build_dex_backend(
        args.revo_root.expanduser().resolve(),
        args.retarget_config.expanduser().resolve(),
        args.side,
    )
    source = WujiGloveSource.connect(args.glove_sn, side=args.side)
    bridge = WujiRevo2Bridge(backend, stale_after_sec=args.stale_after_sec)
    period = 1.0 / max(args.hz, 1.0)
    next_tick = time.monotonic()
    next_print = next_tick
    try:
        print("[Preview] connected; no Revo2 command is being sent")
        while True:
            source.poll()
            now = time.monotonic()
            target = bridge.step(source.store, now=now)
            if target is not None and now >= next_print:
                values = ", ".join(
                    f"{name}={value:.3f}"
                    for name, value in target.as_named_dict().items()
                )
                print(
                    f"[Preview] seq={target.source_sequence} "
                    f"age_ms={(now - target.received_monotonic) * 1000.0:.1f} {values}"
                )
                next_print = now + max(args.print_every_sec, 0.01)
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        print("\n[Preview] stopped; no Revo2 writer was opened")
    finally:
        bridge.reset()
        source.close()


if __name__ == "__main__":
    main()
