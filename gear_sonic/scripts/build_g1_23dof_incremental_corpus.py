"""Build a weighted, span-safe corpus from native true-23 motion NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gear_sonic.utils.g1_23dof_incremental_corpus import build_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--manifest", type=Path)
    sources.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spans", type=Path)
    args = parser.parse_args()
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        root = args.manifest.parent
        entries = [
            (str(item["name"]), (root / item["path"]).resolve(), float(item.get("weight", 1.0)))
            for item in manifest["motions"]
        ]
    else:
        assert args.input_dir is not None
        paths = sorted(args.input_dir.resolve().glob("*.npz"))
        if not paths:
            raise FileNotFoundError("no input NPZ files found")
        entries = [(path.stem, path, 1.0) for path in paths]
    sidecar = args.spans or args.output.with_suffix(".spans.json")
    catalog = build_corpus(entries, args.output.resolve(), sidecar.resolve())
    print(f"{len(catalog.spans)} clips, {catalog.frame_count} frames -> {catalog.corpus_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
