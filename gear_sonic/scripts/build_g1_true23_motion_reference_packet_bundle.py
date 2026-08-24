"""Build saved causal packets for simulator-only PICO transport probing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gear_sonic.utils.g1_true23_motion_reference_packet_bundle import (
    load_motion_reference_packet_bundle,
    write_exclusive_packet_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--transitions", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_motion_reference_packet_bundle(args.motion, transitions=args.transitions)
    write_exclusive_packet_bundle(args.output, bundle)
    raw = args.output.resolve().read_bytes()
    report = {
        "packet_count": len(bundle["robot_independent_reference_packets"]),
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_live_pico_packet": False,
        "live_headset_source_proven": False,
        "hardware_authorized": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
