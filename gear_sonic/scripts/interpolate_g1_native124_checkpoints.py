"""Interpolate aligned native124 actor tensors while preserving a valid broad checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--broad", type=Path, required=True)
    parser.add_argument("--broad-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.broad_weight <= 1.0:
        raise ValueError("broad-weight must be within [0, 1]")
    if args.output.exists():
        raise FileExistsError(args.output)

    pilot = torch.load(args.pilot.resolve(strict=True), map_location="cpu", weights_only=True)
    broad = torch.load(args.broad.resolve(strict=True), map_location="cpu", weights_only=True)
    pilot_actor = pilot["actor_state_dict"]
    broad_actor = broad["actor_state_dict"]
    if pilot_actor.keys() != broad_actor.keys():
        raise ValueError("actor state keys differ")
    actor = {}
    for key in broad_actor:
        left = pilot_actor[key]
        right = broad_actor[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"actor tensor contract differs: {key}")
        actor[key] = torch.lerp(left, right, args.broad_weight) if left.is_floating_point() else right.clone()
    broad["actor_state_dict"] = actor
    broad.setdefault("infos", {})["actor_interpolation"] = {
        "pilot": str(args.pilot.resolve()),
        "broad": str(args.broad.resolve()),
        "broad_weight": args.broad_weight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(broad, args.output)
    print(f"broad_weight={args.broad_weight:.3f} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
