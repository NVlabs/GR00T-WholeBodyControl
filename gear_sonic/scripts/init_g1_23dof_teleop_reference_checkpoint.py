"""Initialize a short-horizon teleoperation checkpoint from a true23 warm start.

The released reference profiles carry ten future frames. Built only from
measurements, that costs ``horizon + one source sample`` before a frame can be
emitted: 200 ms for the low-latency profile and 920 ms for the step5 profile,
against a 60 ms active-policy freshness budget. Clip replay can pay that,
because the future of a scripted motion is already known. Live teleoperation
cannot: the operator's future does not exist yet, and the producer refuses to
synthesize one.

This script narrows the encoder's leading reference term to the first N future
frames, which puts the intrinsic delay at ``N * 20 ms``. Every column that is
kept keeps its trained weight, so the result is a genuine warm start rather than
a reinitialization. Only the first encoder layer changes shape; every deeper
encoder layer, the token head, and the whole decoder are copied unchanged.

The output is an initialization checkpoint, not a trained policy. It carries no
optimizer or value state and must be trained before it can be exported or
deployed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from gear_sonic.utils.g1_23dof_contract import (
    ALL_REFERENCE_PROFILES,
    REFERENCE_FRAME_TERM_WIDTH,
    REFERENCE_PROFILE_TELEOP_2F,
    TELEOP_ENCODER_INPUT_DIM,
    TELEOP_ENCODER_INPUT_TERM_DIMS,
    reference_profile_contract,
)

REFERENCE_PROFILES = ALL_REFERENCE_PROFILES

_ENCODER_INPUT_WEIGHT_KEY = "actor_module.encoders.teleop.module.0.weight"
_SOURCE_FRAME_COUNT = TELEOP_ENCODER_INPUT_TERM_DIMS[0] // (
    2 * REFERENCE_FRAME_TERM_WIDTH
)


def retained_encoder_columns(frame_count: int) -> list[int]:
    """Encoder input columns kept when the future horizon shrinks.

    The leading term is laid out as all future positions followed by all future
    velocities, so the retained columns are two separate runs, plus the three
    trailing terms which do not depend on the horizon.
    """

    if not 1 <= frame_count <= _SOURCE_FRAME_COUNT:
        raise ValueError(
            f"frame count must be within 1..{_SOURCE_FRAME_COUNT}, "
            f"got {frame_count}"
        )
    width = REFERENCE_FRAME_TERM_WIDTH
    positions_end = frame_count * width
    velocities_start = _SOURCE_FRAME_COUNT * width
    columns = list(range(positions_end))
    columns.extend(
        range(velocities_start, velocities_start + frame_count * width)
    )
    columns.extend(
        range(TELEOP_ENCODER_INPUT_TERM_DIMS[0], TELEOP_ENCODER_INPUT_DIM)
    )
    return columns


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_teleop_checkpoint(
    source: dict[str, Any],
    *,
    profile_name: str,
) -> dict[str, Any]:
    profile = REFERENCE_PROFILES[profile_name]
    policy_state = source.get("policy_state_dict")
    if not isinstance(policy_state, dict):
        raise ValueError("source checkpoint has no policy_state_dict")
    weight = policy_state.get(_ENCODER_INPUT_WEIGHT_KEY)
    if weight is None:
        raise ValueError(f"source checkpoint lacks {_ENCODER_INPUT_WEIGHT_KEY}")
    if weight.ndim != 2 or weight.shape[1] != TELEOP_ENCODER_INPUT_DIM:
        raise ValueError(
            f"{_ENCODER_INPUT_WEIGHT_KEY} must be [*, {TELEOP_ENCODER_INPUT_DIM}], "
            f"got {tuple(weight.shape)}"
        )

    columns = retained_encoder_columns(profile.future_frame_count)
    if len(columns) != profile.encoder_input_dim:
        raise ValueError(
            f"retained {len(columns)} columns but profile expects "
            f"{profile.encoder_input_dim}"
        )

    narrowed = dict(policy_state)
    index = torch.tensor(columns, dtype=torch.long)
    narrowed[_ENCODER_INPUT_WEIGHT_KEY] = (
        weight.detach().index_select(1, index).contiguous().clone()
    )

    report = {
        "source_encoder_input_dim": TELEOP_ENCODER_INPUT_DIM,
        "target_encoder_input_dim": profile.encoder_input_dim,
        "source_future_frame_count": _SOURCE_FRAME_COUNT,
        "target_future_frame_count": profile.future_frame_count,
        "input_weight_key": _ENCODER_INPUT_WEIGHT_KEY,
        "retained_encoder_columns": columns,
        "reference_profile": profile.name,
        "reference_contract": reference_profile_contract(profile.name),
        "optimizer_reused": False,
        "value_model_reused": False,
        "initialization_only": True,
        "source_policy_state_sha256": _state_dict_sha256(policy_state),
        "initial_policy_state_sha256": _state_dict_sha256(narrowed),
    }

    metadata = dict(source.get("g1_23dof_metadata") or {})
    metadata["reference_profile"] = profile.name
    metadata["teleop_encoder_input_dim"] = profile.encoder_input_dim

    return {
        "g1_23dof_safe_checkpoint": {
            "schema_version": 1,
            "kind": "g1_23dof_safe_weights_checkpoint",
            "checkpoint_stage": "checkpoint_initialization",
            "resume_state_included": False,
        },
        "policy_state_dict": narrowed,
        "g1_23dof_metadata": metadata,
        "g1_23dof_initialization_report": report,
        "g1_23dof_teleop_reference_report": report,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--profile",
        default=REFERENCE_PROFILE_TELEOP_2F,
        choices=sorted(REFERENCE_PROFILES),
    )
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = REFERENCE_PROFILES[args.profile]
    if profile.encoder_input_dim == TELEOP_ENCODER_INPUT_DIM:
        raise SystemExit(
            f"profile {args.profile} already uses the released encoder width; "
            "nothing to narrow"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    checkpoint = build_teleop_checkpoint(source, profile_name=args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    report = dict(checkpoint["g1_23dof_teleop_reference_report"])
    report["source_checkpoint"] = str(args.source)
    report["source_checkpoint_sha256"] = _sha256_bytes(args.source.read_bytes())
    report["output_checkpoint"] = str(args.output)
    report["output_checkpoint_sha256"] = _sha256_bytes(args.output.read_bytes())
    if args.report:
        args.report.write_text(json.dumps(report, indent=1, sort_keys=True))

    print(
        f"[OK] {args.source.name} -> {args.output.name}: encoder "
        f"{report['source_encoder_input_dim']} -> "
        f"{report['target_encoder_input_dim']} "
        f"({report['target_future_frame_count']} future frames, profile "
        f"{profile.name})"
    )
    print("[NOTE] Initialization only. Train before export or deployment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
