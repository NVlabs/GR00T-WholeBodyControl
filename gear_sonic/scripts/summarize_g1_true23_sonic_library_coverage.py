"""Validate and summarize released SONIC feature coverage on true23 physics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file

FEATURE_REPORTS = {
    "hand_crawling": "artifacts/g1_true23/sonic_library_true23_released_adapter_v9_crawl_dataset/report.json",
    "elbow_crawling": "artifacts/g1_true23/sonic_library_true23_released_adapter_v11_elbow_dataset/report.json",
    "happy_dance": "artifacts/g1_true23/sonic_library_true23_released_adapter_v10_happy_dataset/report.json",
    "zombie_walk": "artifacts/g1_true23/sonic_library_true23_released_adapter_v12_zombie_dataset/report.json",
}
NATIVE_CRAWL_REPORT = "artifacts/g1_true23/sonic_crawl_native_final_v1/report.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    output = args.output if args.output.is_absolute() else root / args.output
    if os.path.lexists(output):
        raise FileExistsError(output)
    features: list[dict[str, Any]] = []
    for expected_name, relative_report in FEATURE_REPORTS.items():
        report_path = (root / relative_report).resolve(strict=True)
        report = _load(report_path)
        records = report.get("records")
        if (
            report.get("passed") is not True
            or report.get("physical_model") != "g1_23dof_rev_1_0"
            or report.get("physical_dof") != 23
            or not isinstance(records, list)
            or len(records) != 1
        ):
            raise ValueError(f"{expected_name} true23 feature report failed validation")
        record = records[0]
        if not isinstance(record, dict):
            raise ValueError(f"{expected_name} record is not a JSON object")
        metrics = record.get("metrics")
        if (
            record.get("name") != expected_name
            or not isinstance(metrics, dict)
            or metrics.get("passed") is not True
            or metrics.get("actuator_count") != 23
            or metrics.get("physical_dof") != 23
            or metrics.get("removed_source_outputs_discarded") != 6
            or metrics.get("teacher_mode") != "live_released_policy_on_true23_state"
        ):
            raise ValueError(f"{expected_name} physical metrics failed validation")
        video = (report_path.parent / record["physical_video"]).resolve(strict=True)
        trajectory = (report_path.parent / record["physical_npz"]).resolve(strict=True)
        if sha256_file(video) != record["physical_video_sha256"]:
            raise ValueError(f"{expected_name} video hash changed")
        if sha256_file(trajectory) != record["physical_npz_sha256"]:
            raise ValueError(f"{expected_name} trajectory hash changed")
        features.append(
            {
                "name": expected_name,
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "video": str(video),
                "video_sha256": sha256_file(video),
                "trajectory": str(trajectory),
                "trajectory_sha256": sha256_file(trajectory),
                "completed_control_steps": metrics["completed_control_steps"],
                "horizontal_displacement_m": metrics["horizontal_displacement_m"],
                "minimum_root_height_m": metrics["minimum_root_height_m"],
                "maximum_root_tilt_rad": metrics["maximum_root_tilt_rad"],
                "passed": True,
            }
        )
    native_path = (root / NATIVE_CRAWL_REPORT).resolve(strict=True)
    native = _load(native_path)
    if (
        native.get("passed") is not True
        or native.get("physical_model") != "g1_23dof_rev_1_0"
        or native.get("physical_dof") != 23
        or native.get("actuator_count") != 23
        or native.get("decoder_output_dof") != 23
        or native.get("source_29dof_physics_used") is not False
        or native.get("completed_transitions") != native.get("requested_transitions")
    ):
        raise ValueError("native true23 crawl report failed validation")
    result = {
        "schema_version": 1,
        "kind": "g1_true23_released_sonic_library_feature_coverage_v1",
        "physical_model": "g1_23dof_rev_1_0",
        "physical_dof": 23,
        "actuator_count": 23,
        "released_library_feature_count": len(features),
        "released_library_features": features,
        "all_released_library_features_true23_physics_passed": True,
        "native_true23_causal_hand_crawl": {
            "report": str(native_path),
            "report_sha256": sha256_file(native_path),
            "decoder_sha256": native["decoder_sha256"],
            "completed_transitions": native["completed_transitions"],
            "passed": True,
        },
        "interface_boundary": {
            "compatibility_teacher_internal_output_dof": 29,
            "discarded_teacher_outputs": 6,
            "physical_state_dof": 23,
            "physical_target_dof": 23,
            "physical_actuator_dof": 23,
            "native_crawl_decoder_output_dof": 23,
        },
        "passed": True,
        "authorization": {
            "simulator_only": True,
            "hardware_authorized": False,
            "robot_commands_published": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
