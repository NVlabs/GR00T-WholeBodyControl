"""Tune proven ankle-negative residual behind nominal-safe deadband."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from gear_sonic.scripts import screen_g1_true23_sonic_rank256_balance_residual as parent

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_balance_residual_tuning_v2.json"
)
CONTRACT_SHA256 = "511d4f27ccf9faa693e434e2803c83a26fe93cd963d6d21c810961f40c97252f"
RESULT_FILENAME = "rank256_balance_residual_tuning_result.json"
VARIANTS = (
    ("baseline", 0.0, 0.0),
    ("ankle_m1", -1.0, 0.0),
    ("ankle_m1p5", -1.5, 0.0),
    ("ankle_m2", -2.0, 0.0),
    ("ankle_m2p5", -2.5, 0.0),
    ("ankle_m3", -3.0, 0.0),
)


@contextmanager
def _scope() -> Iterator[None]:
    names = (
        "CONTRACT_RELATIVE_PATH",
        "CONTRACT_SHA256",
        "RESULT_FILENAME",
        "VARIANTS",
        "EXTRA_SOURCE_RELATIVE_PATHS",
    )
    saved = {name: getattr(parent, name) for name in names}
    try:
        parent.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
        parent.CONTRACT_SHA256 = CONTRACT_SHA256
        parent.RESULT_FILENAME = RESULT_FILENAME
        parent.VARIANTS = VARIANTS
        parent.EXTRA_SOURCE_RELATIVE_PATHS = (
            Path("gear_sonic/scripts/screen_g1_true23_sonic_rank256_balance_residual_tuning.py"),
        )
        yield
    finally:
        for name, value in saved.items():
            setattr(parent, name, value)


def preflight(repository_root: Path) -> Mapping[str, Any]:
    with _scope():
        return parent.preflight(repository_root)


def run(repository_root: Path, run_dir: Path) -> Mapping[str, Any]:
    with _scope():
        return parent.run(repository_root, run_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=ROOT)
    screen = sub.add_parser("screen")
    screen.add_argument("--repository-root", type=Path, default=ROOT)
    screen.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = (
        preflight(args.repository_root) if args.command == "preflight" else run(args.repository_root, args.run_dir)
    )
    print(
        json.dumps(
            report if args.command == "preflight" else report["assessment"],
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )  # noqa: T201
    return 0 if args.command == "preflight" and report.get("ready") is True or args.command == "screen" else 1


if __name__ == "__main__":
    raise SystemExit(main())
