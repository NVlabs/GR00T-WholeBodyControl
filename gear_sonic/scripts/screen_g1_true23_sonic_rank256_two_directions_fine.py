"""Fine screen around coarse rank256 survival/reward direction tradeoff."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from gear_sonic.scripts import screen_g1_true23_sonic_rank256_two_directions as coarse

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_two_direction_fine_screen_v1.json"
)
CONTRACT_SHA256 = "04f4d3489c7bba6499ba73c18f2f8f6adaf7c5149a6bedd5772eabc16b98d658"
COEFFICIENTS = (
    (0.0, 0.0),
    (0.0, 1.0),
    (0.25, 1.0),
    (0.5, 0.75),
    (0.5, 1.0),
    (0.5, 1.25),
    (0.75, 1.0),
)


@contextmanager
def _scope() -> Iterator[None]:
    names = ("CONTRACT_RELATIVE_PATH", "CONTRACT_SHA256", "COEFFICIENTS", "EXTRA_SOURCE_RELATIVE_PATHS")
    saved = {name: getattr(coarse, name) for name in names}
    try:
        coarse.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
        coarse.CONTRACT_SHA256 = CONTRACT_SHA256
        coarse.COEFFICIENTS = COEFFICIENTS
        coarse.EXTRA_SOURCE_RELATIVE_PATHS = (
            Path("gear_sonic/scripts/screen_g1_true23_sonic_rank256_two_directions_fine.py"),
        )
        yield
    finally:
        for name, value in saved.items():
            setattr(coarse, name, value)


def preflight(repository_root: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope():
        return coarse._preflight(root)  # noqa: SLF001


def run(repository_root: Path, run_dir: Path) -> Mapping[str, Any]:
    with _scope():
        return coarse.run(repository_root, run_dir)


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
    if args.command == "preflight":
        report = preflight(args.repository_root)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    result = run(args.repository_root, args.run_dir)
    print(json.dumps(result["assessment"], indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if result["candidate"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
