"""Run diagnostic-only external-reference true23 ONNX replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gear_sonic.utils.g1_true23_step1c_onnx_diagnostic import MODES, run_diagnostic, write_report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--env", type=Path, required=True)
    p.add_argument("--source-csv-root", type=Path, required=True)
    p.add_argument("--step1a-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policy-sha256")
    p.add_argument("--env-sha256")
    p.add_argument("--mode", action="append", choices=sorted(MODES))
    p.add_argument("--clip", action="append")
    p.add_argument("--stop-on-first-termination", action="store_true")
    p.add_argument("--horizon", type=int, default=500)
    a = p.parse_args()
    report = run_diagnostic(
        policy_path=a.policy,
        env_path=a.env,
        source_csv_root=a.source_csv_root,
        step1a_root=a.step1a_root,
        modes=a.mode or sorted(MODES),
        horizon_steps=a.horizon,
        selected_clip_ids=a.clip,
        stop_on_first_termination=a.stop_on_first_termination,
        policy_sha256=a.policy_sha256,
        env_sha256=a.env_sha256,
    )
    write_report(a.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
