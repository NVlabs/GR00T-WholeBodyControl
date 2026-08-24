#!/usr/bin/env python3
"""Run exact selected teacher from q9=9 on heldout seed835."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_selected_teacher_seed835_nominal_diagnostic import run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(repository_root=args.repository_root, output=args.output)
    passed = report.get("qualified_nominal_slice") is True
    print(json.dumps({"passed": passed, "output": str(args.output)}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
