"""Net2Wider-expand native124 actor and critic while preserving their functions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional


def _mapping(old_width: int, new_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    if new_width < old_width:
        raise ValueError("new hidden widths must not shrink")
    mapping = torch.arange(new_width) % old_width
    counts = torch.bincount(mapping, minlength=old_width)
    return mapping, counts


def _widen_mlp(
    state: dict[str, torch.Tensor], new_dims: tuple[int, int, int]
) -> dict[str, torch.Tensor]:
    old_dims = tuple(int(state[f"mlp.{index}.bias"].shape[0]) for index in (0, 2, 4))
    maps_counts = [_mapping(old, new) for old, new in zip(old_dims, new_dims, strict=True)]
    result = {key: value.clone() for key, value in state.items()}
    previous_map: torch.Tensor | None = None
    previous_counts: torch.Tensor | None = None
    for layer_index, (mapping, _counts) in zip((0, 2, 4), maps_counts, strict=True):
        weight = state[f"mlp.{layer_index}.weight"]
        widened = weight[mapping]
        if previous_map is not None and previous_counts is not None:
            widened = widened[:, previous_map] / previous_counts[previous_map].to(weight.dtype)
        result[f"mlp.{layer_index}.weight"] = widened
        result[f"mlp.{layer_index}.bias"] = state[f"mlp.{layer_index}.bias"][mapping]
        previous_map, previous_counts = mapping, _counts
    assert previous_map is not None and previous_counts is not None
    output_weight = state["mlp.6.weight"]
    result["mlp.6.weight"] = (
        output_weight[:, previous_map] / previous_counts[previous_map].to(output_weight.dtype)
    )
    return result


def _residual_widen_mlp(
    state: dict[str, torch.Tensor], new_dims: tuple[int, int, int]
) -> dict[str, torch.Tensor]:
    old_dims = tuple(int(state[f"mlp.{index}.bias"].shape[0]) for index in (0, 2, 4))
    if any(new < old for old, new in zip(old_dims, new_dims, strict=True)):
        raise ValueError("new hidden widths must not shrink")
    result = {key: value.clone() for key, value in state.items()}
    previous_old = int(state["mlp.0.weight"].shape[1])
    previous_new = previous_old
    for layer_index, old_width, new_width in zip(
        (0, 2, 4), old_dims, new_dims, strict=True
    ):
        old_weight = state[f"mlp.{layer_index}.weight"]
        weight = torch.zeros(new_width, previous_new, dtype=old_weight.dtype)
        weight[:old_width, :previous_old] = old_weight
        bias = torch.empty(new_width, dtype=state[f"mlp.{layer_index}.bias"].dtype)
        bias[:old_width] = state[f"mlp.{layer_index}.bias"]
        if new_width > old_width:
            mapping = torch.arange(new_width - old_width) % old_width
            weight[old_width:, :previous_old] = old_weight[mapping]
            bias[old_width:] = state[f"mlp.{layer_index}.bias"][mapping]
        result[f"mlp.{layer_index}.weight"] = weight
        result[f"mlp.{layer_index}.bias"] = bias
        previous_old, previous_new = old_width, new_width
    old_output = state["mlp.6.weight"]
    output = torch.zeros(old_output.shape[0], previous_new, dtype=old_output.dtype)
    output[:, :previous_old] = old_output
    result["mlp.6.weight"] = output
    return result


def _forward(state: dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    value = inputs
    for index in (0, 2, 4):
        value = functional.elu(functional.linear(value, state[f"mlp.{index}.weight"], state[f"mlp.{index}.bias"]))
    return functional.linear(value, state["mlp.6.weight"], state["mlp.6.bias"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dims", type=int, nargs=3, default=(768, 384, 192))
    parser.add_argument("--seed", type=int, default=7601)
    parser.add_argument("--residual", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    new_dims = tuple(args.hidden_dims)

    checkpoint = torch.load(args.checkpoint.resolve(strict=True), map_location="cpu", weights_only=True)
    actor = checkpoint["actor_state_dict"]
    critic = checkpoint["critic_state_dict"]
    widen = _residual_widen_mlp if args.residual else _widen_mlp
    widened_actor = widen(actor, new_dims)
    widened_critic = widen(critic, new_dims)

    generator = torch.Generator().manual_seed(args.seed)
    actor_inputs = torch.randn(256, actor["mlp.0.weight"].shape[1], generator=generator)
    critic_inputs = torch.randn(256, critic["mlp.0.weight"].shape[1], generator=generator)
    actor_error = float((_forward(actor, actor_inputs) - _forward(widened_actor, actor_inputs)).abs().max())
    critic_error = float(
        (_forward(critic, critic_inputs) - _forward(widened_critic, critic_inputs)).abs().max()
    )
    if max(actor_error, critic_error) > 1.0e-4:
        raise RuntimeError("Net2Wider parity failed")

    checkpoint["actor_state_dict"] = widened_actor
    checkpoint["critic_state_dict"] = widened_critic
    checkpoint["optimizer_state_dict"] = {}
    checkpoint.setdefault("infos", {})["net2wider"] = {
        "source": str(args.checkpoint.resolve()),
        "hidden_dims": list(new_dims),
        "residual": args.residual,
        "actor_max_absolute_error": actor_error,
        "critic_max_absolute_error": critic_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "source": str(args.checkpoint.resolve()),
        "output": str(args.output.resolve()),
        "hidden_dims": list(new_dims),
        "residual": args.residual,
        "actor_max_absolute_error": actor_error,
        "critic_max_absolute_error": critic_error,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
