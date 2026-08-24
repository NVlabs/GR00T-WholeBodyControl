"""Batch the official Unitree MJLab true-23 CSV-to-NPZ conversion."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_converter(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("unitree_csv_to_npz", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-fps", type=int, default=60)
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path("external_dependencies/unitree_rl_mjlab/scripts/csv_to_npz.py"),
    )
    args = parser.parse_args()

    sources = sorted(args.input.resolve().glob("*.csv"))
    if not sources:
        raise FileNotFoundError("no input CSV files found")
    args.output.mkdir(parents=True, exist_ok=True)
    converter = _load_converter(args.converter.resolve(strict=True))
    for index, source in enumerate(sources, start=1):
        destination = (args.output / source.with_suffix(".npz").name).resolve()
        if args.skip_existing and destination.exists() and destination.stat().st_size > 0:
            print(f"[{index}/{len(sources)}] skip {destination.name}", flush=True)
            continue
        print(f"[{index}/{len(sources)}] {source.name} -> {destination.name}", flush=True)
        converter.main(
            input_file=str(source),
            output_name=str(destination),
            input_fps=args.input_fps,
            output_fps=args.output_fps,
            device=args.device,
            render=False,
            robot="g1_23dof",
            line_range=None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
