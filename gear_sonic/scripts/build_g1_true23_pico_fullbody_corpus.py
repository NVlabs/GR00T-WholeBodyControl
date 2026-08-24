"""Build hash-bound Pico plus SONIC-library native-23 training corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from gear_sonic.utils.g1_23dof_incremental_corpus import build_corpus, sha256_file

MANIFEST_KIND = "g1_true23_pico_fullbody_multimotion_manifest_v1"
SPAN_KIND = "g1_true23_motion_corpus_spans_v1"
RECOVERY_METADATA_SCHEMA = "g1_true23_low_latency_recovery_motion_v1"
MODEL_XML_RELATIVE = Path("external_dependencies/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1_23dof.xml")


def _load_entries(repository_root: Path, manifest_path: Path) -> list[tuple[str, Path, float]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("kind") != MANIFEST_KIND
        or set(payload) != {"schema_version", "kind", "motions"}
        or not isinstance(payload["motions"], list)
        or not payload["motions"]
    ):
        raise ValueError("full-body corpus manifest schema mismatch")
    entries: list[tuple[str, Path, float]] = []
    for index, item in enumerate(payload["motions"]):
        if not isinstance(item, dict) or set(item) != {"name", "path", "weight"}:
            raise ValueError(f"full-body corpus motion {index} schema mismatch")
        name = item["name"]
        path = item["path"]
        weight = item["weight"]
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(path, str)
            or not path
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or float(weight) <= 0.0
        ):
            raise ValueError(f"full-body corpus motion {index} value mismatch")
        entries.append((name, (repository_root / path).resolve(strict=True), float(weight)))
    return entries


def build_fullbody_corpus(
    *,
    repository_root: Path,
    manifest_path: Path,
    output_path: Path,
    catalog_path: Path,
    spans_path: Path,
    recovery_metadata_path: Path,
    episode_frames: int = 500,
) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    manifest = manifest_path.resolve(strict=True)
    entries = _load_entries(root, manifest)
    catalog = build_corpus(
        entries,
        output_path.resolve(),
        catalog_path.resolve(),
        episode_frames=episode_frames,
    )
    payload = {
        "kind": SPAN_KIND,
        "corpus": str(catalog.corpus_path),
        "corpus_sha256": catalog.corpus_sha256,
        "episode_frames": episode_frames,
        "fps": catalog.fps,
        "clip_count": len(catalog.spans),
        "total_frames": catalog.frame_count,
        "manifest": str(manifest),
        "spans": [
            {
                "name": span.name,
                "start": span.start,
                "length": span.stored_length,
                "original_length": span.original_length,
                "weight": span.weight,
            }
            for span in catalog.spans
        ],
        "authorization": {
            "training_input_only": True,
            "deployment_ready": False,
            "hardware_authorized": False,
        },
    }
    spans_path.parent.mkdir(parents=True, exist_ok=True)
    with spans_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    model_xml = (root / MODEL_XML_RELATIVE).resolve(strict=True)
    recovery_metadata = {
        "schema": RECOVERY_METADATA_SCHEMA,
        "simulator_only": True,
        "deployment_ready": False,
        "source": {
            "filename": manifest.name,
            "sha256": sha256_file(manifest),
            "frames": catalog.frame_count,
            "kind": MANIFEST_KIND,
        },
        "model": {
            "filename": model_xml.name,
            "sha256": sha256_file(model_xml),
        },
        "output": {
            "filename": catalog.corpus_path.name,
            "sha256": catalog.corpus_sha256,
            "frames": catalog.frame_count,
            "fps": catalog.fps,
            "duration_s": (catalog.frame_count - 1) / catalog.fps,
        },
        "segments_inclusive": {span.name: [span.start, span.stop - 1] for span in catalog.spans},
        "corpus": {
            "schema": "g1_true23_native124_incremental_corpus_v1",
            "clip_count": len(catalog.spans),
            "episode_frames": episode_frames,
            "span_sidecar_filename": spans_path.name,
            "span_sidecar_sha256": sha256_file(spans_path),
            "source_catalog_filename": catalog_path.name,
            "source_catalog_sha256": sha256_file(catalog_path),
            "full_body_controlled_joint_count": 23,
            "source_29dof_physics_used": False,
        },
        "kinematics": {
            "motion_arrays_preserved_without_retargeting": True,
            "joint_and_body_arrays_present_at_50hz": True,
            "clip_boundaries_are_episode_contained": True,
        },
        "authorization": {
            "training_input_only": True,
            "hardware_authorized": False,
            "robot_network_commands": False,
        },
    }
    recovery_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with recovery_metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(recovery_metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--recovery-metadata", type=Path, required=True)
    parser.add_argument("--episode-frames", type=int, default=500)
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)

    def rooted(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    output = rooted(args.output)
    catalog = rooted(args.catalog)
    spans = rooted(args.spans)
    recovery_metadata = rooted(args.recovery_metadata)
    if any(os.path.lexists(path) for path in (output, catalog, spans, recovery_metadata)):
        raise FileExistsError("full-body corpus output exists")
    report = build_fullbody_corpus(
        repository_root=root,
        manifest_path=rooted(args.manifest),
        output_path=output,
        catalog_path=catalog,
        spans_path=spans,
        recovery_metadata_path=recovery_metadata,
        episode_frames=args.episode_frames,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
