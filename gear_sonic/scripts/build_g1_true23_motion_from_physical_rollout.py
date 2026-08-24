"""Build exact true23 motion-reference arrays from a 23-actuator physical rollout."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_task_space_retarget import _angular_velocity, _finite_difference
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion

FPS = 50.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--physical-rollout", type=Path, required=True)
    parser.add_argument("--output-motion", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    source = args.physical_rollout if args.physical_rollout.is_absolute() else root / args.physical_rollout
    output = args.output_motion if args.output_motion.is_absolute() else root / args.output_motion
    manifest = args.output_manifest if args.output_manifest.is_absolute() else root / args.output_manifest
    if os.path.lexists(output) or os.path.lexists(manifest):
        raise FileExistsError("true23 physical-reference output exists")
    with np.load(source, allow_pickle=False) as archive:
        pre_qpos = np.asarray(archive["pre_qpos"], dtype=np.float64)
        pre_qvel = np.asarray(archive["pre_qvel"], dtype=np.float64)
        control_dt = float(np.asarray(archive["control_dt"])[0])
    if (
        pre_qpos.ndim != 2
        or pre_qpos.shape[1] != 30
        or pre_qvel.shape != (len(pre_qpos), 29)
        or len(pre_qpos) < 12
        or not np.isfinite(pre_qpos).all()
        or not np.isfinite(pre_qvel).all()
    ):
        raise ValueError("physical rollout state arrays are invalid")
    if not math.isclose(control_dt, 1.0 / FPS, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("physical rollout must use exact 50-Hz control")
    model_path = root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq, model.nv, model.nu, model.nbody - 1) != (30, 29, 23, 24):
        raise ValueError("true23 physical model ABI changed")
    data = mujoco.MjData(model)
    body_pos = np.empty((len(pre_qpos), 24, 3), dtype=np.float64)
    body_quat = np.empty((len(pre_qpos), 24, 4), dtype=np.float64)
    for index, qpos in enumerate(pre_qpos):
        data.qpos[:] = qpos
        data.qvel[:] = pre_qvel[index]
        mujoco.mj_forward(model, data)
        body_pos[index] = data.xpos[1:]
        body_quat[index] = data.xquat[1:]
    body_ang_vel = np.stack(
        [_angular_velocity(body_quat[:, body], control_dt) for body in range(24)],
        axis=1,
    )
    arrays = {
        "fps": np.asarray([FPS], dtype=np.float64),
        "joint_pos": pre_qpos[:, 7:].astype(np.float32),
        "joint_vel": pre_qvel[:, 6:].astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": _finite_difference(body_pos, control_dt).astype(np.float32),
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
    }
    validate_library_motion(arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {
        "schema_version": 1,
        "kind": "g1_true23_physical_rollout_motion_reference_v1",
        "source_path": str(source.resolve(strict=True)),
        "source_sha256": sha256_file(source),
        "physical_model": "g1_23dof_rev_1_0",
        "physical_dof": 23,
        "frame_count": len(pre_qpos),
        "output_motion": str(output),
        "output_sha256": sha256_file(output),
        "passed": True,
        "authorization": {
            "offline_motion_material_only": True,
            "simulator_qualification_complete": False,
            "hardware_authorized": False,
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
