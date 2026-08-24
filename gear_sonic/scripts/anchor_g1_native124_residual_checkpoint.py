"""Restore a frozen base actor inside a trained residual-wide checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-dims", type=int, nargs=3, default=(512, 256, 128))
    parser.add_argument("--residual-scale", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 <= args.residual_scale <= 1.0:
        raise ValueError("residual scale must be within [0, 1]")
    if args.output.exists():
        raise FileExistsError(args.output)
    base = torch.load(args.base.resolve(strict=True), map_location="cpu", weights_only=True)
    trained = torch.load(args.trained.resolve(strict=True), map_location="cpu", weights_only=True)
    source = base["actor_state_dict"]
    actor = trained["actor_state_dict"]
    core_dims = tuple(args.core_dims)

    previous_core = int(source["mlp.0.weight"].shape[1])
    for layer, core in zip((0, 2, 4), core_dims, strict=True):
        actor[f"mlp.{layer}.weight"][:core].zero_()
        actor[f"mlp.{layer}.weight"][:core, :previous_core] = source[f"mlp.{layer}.weight"]
        actor[f"mlp.{layer}.bias"][:core] = source[f"mlp.{layer}.bias"]
        previous_core = core
    actor["mlp.6.weight"][:, : core_dims[-1]] = source["mlp.6.weight"]
    actor["mlp.6.weight"][:, core_dims[-1] :] *= args.residual_scale
    actor["mlp.6.bias"] = source["mlp.6.bias"].clone()
    for key, value in source.items():
        if key.startswith("obs_normalizer.") or key.startswith("distribution."):
            actor[key] = value.clone()

    trained["actor_state_dict"] = actor
    trained.setdefault("infos", {})["residual_anchor"] = {
        "base": str(args.base.resolve()),
        "trained": str(args.trained.resolve()),
        "core_dims": list(core_dims),
        "residual_scale": args.residual_scale,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained, args.output)
    report = {
        "base": str(args.base.resolve()),
        "trained": str(args.trained.resolve()),
        "output": str(args.output.resolve()),
        "core_dims": list(core_dims),
        "residual_scale": args.residual_scale,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
