"""Plan or run the paired G1 source29/true23 MuJoCo Step 1B diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_step1b_mujoco import (
    EPISODES_PER_CLIP,
    HORIZON_STEPS,
    REPOSITORY_ROOT,
    SIM_CONFIG_RELPATH,
    SOURCE_MODEL_RELPATH,
    TARGET_MODEL_RELPATH,
    build_plan,
    frozen_clip_paths,
    run_paired_diagnostic_campaign,
)
from gear_sonic.utils.g1_true23_step1b_qualification import (
    DEFAULT_CONTRACT_PATH,
    FROZEN_CLIP_IDS,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "low_latency" / "last.pt",
        help="exact pinned low_latency/last.pt",
    )
    parser.add_argument("--source-csv-root", type=Path, required=True)
    parser.add_argument(
        "--step1a-root",
        type=Path,
        required=True,
        help="schema-v6 root containing motions/, experts/, and reports/",
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--source-model", type=Path, default=REPOSITORY_ROOT / SOURCE_MODEL_RELPATH)
    parser.add_argument("--target-model", type=Path, default=REPOSITORY_ROOT / TARGET_MODEL_RELPATH)
    parser.add_argument("--sim-config", type=Path, default=REPOSITORY_ROOT / SIM_CONFIG_RELPATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--clip-id",
        action="append",
        choices=FROZEN_CLIP_IDS,
        help="repeat to select clips; smoke defaults to the first, full requires both",
    )
    parser.add_argument("--episodes-per-clip", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--full",
        action="store_true",
        help="select both frozen clips, the contract episode count, and 500 steps",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run MuJoCo; without this flag only immutable inputs/work are planned",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.full:
        if args.clip_id is not None and tuple(args.clip_id) != FROZEN_CLIP_IDS:
            raise ValueError("--full requires both frozen clips in contract order")
        selected = FROZEN_CLIP_IDS
        episodes = EPISODES_PER_CLIP
        horizon = HORIZON_STEPS
    else:
        selected = tuple(args.clip_id) if args.clip_id else (FROZEN_CLIP_IDS[0],)
        episodes = args.episodes_per_clip
        horizon = args.horizon
    paths = frozen_clip_paths(
        contract_path=args.contract,
        source_csv_root=args.source_csv_root,
        step1a_root=args.step1a_root,
        selected_clip_ids=selected,
    )
    if not args.execute:
        plan = build_plan(
            checkpoint_path=args.checkpoint,
            clip_paths=paths,
            contract_path=args.contract,
            device=args.device,
            episodes_per_clip=episodes,
            horizon_steps=horizon,
        )
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.output is None:
        raise ValueError("--execute requires --output")
    manifest = run_paired_diagnostic_campaign(
        output_root=args.output,
        checkpoint_path=args.checkpoint,
        clip_paths=paths,
        contract_path=args.contract,
        source_model_path=args.source_model,
        target_model_path=args.target_model,
        sim_config_path=args.sim_config,
        device=args.device,
        episodes_per_clip=episodes,
        horizon_steps=horizon,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
