#!/usr/bin/env python3
"""Screen push-trained model21248 on exact failed q200 disturbance."""

from __future__ import annotations

import argparse
from pathlib import Path

from gear_sonic.utils import g1_true23_sonic_rank256_q200_balanced_teacher_screen as screen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = screen.run(repository_root=args.repository_root, output=args.output)
    print(f"verdict={report['verdict']} steps={report['attempted_transitions']} q9={report['terminal_q9']}")
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
