#!/usr/bin/env python3
"""Convert SOMA/Bones-style H2 CSV data to SONIC H2 motion_lib format.

This is an H2-specific variant based on the G1 SOMA converter idea, but with:
- H2 31 DOFs / 32 bodies
- configurable position unit: m or cm
- configurable angle unit: rad or deg
- default assumptions suitable for SOMA retarget CSVs that are already meters/radians
- OOM Protection: memory cleanup and chunked saving

Expected flat CSV columns:
    root_translateX, root_translateY, root_translateZ
    root_rotateX, root_rotateY, root_rotateZ
    31 columns ending with "_dof"

Output motion_lib entry:
    root_trans_offset: (T, 3)
    pose_aa:           (T, 32, 3)
    dof:               (T, 31)
    root_rot:          (T, 4), xyzw
    smpl_joints:       (T, 24, 3), placeholder zeros
    fps:               int
"""

import argparse
import os
import sys
import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import transform


NUM_DOF = 31
NUM_BODIES = 32

H2_ISAACLAB_TO_MUJOCO_DOF = np.array(
    [
        0, 3, 6, 9, 14, 19,
        1, 4, 7, 10, 15, 20,
        2, 5, 8, 11, 16,
        12, 17,
        21, 23, 25, 27, 29,
        13, 18, 22, 24, 26, 28, 30,
    ],
    dtype=np.int32,
)

DOF_AXIS = np.array(
    [
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0],
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0],
        [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ],
    dtype=np.float32,
)


def collect_csvs(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() != ".csv":
            raise ValueError(f"Input file must be CSV: {p}")
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.csv"))
    raise ValueError(f"Input path does not exist: {input_path}")


def infer_joint_cols(df):
    joint_cols = [c for c in df.columns if c.endswith("_dof")]
    if len(joint_cols) != NUM_DOF:
        raise ValueError(f"Expected {NUM_DOF} *_dof columns, got {len(joint_cols)}")
    return joint_cols


def load_h2_flat_csv(csv_path: Path, fps: int, position_unit: str, angle_unit: str, joint_order: str, root_z_offset: float):
    df = pd.read_csv(csv_path)
    required = [
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {missing}")

    root_pos = np.stack(
        [
            df["root_translateX"].to_numpy(),
            df["root_translateY"].to_numpy(),
            df["root_translateZ"].to_numpy(),
        ],
        axis=1,
    ).astype(np.float32)

    if position_unit == "cm":
        root_pos /= 100.0
    elif position_unit != "m":
        raise ValueError(f"Unsupported position_unit: {position_unit}")

    root_pos[:, 2] += float(root_z_offset)

    euler = np.stack(
        [
            df["root_rotateX"].to_numpy(),
            df["root_rotateY"].to_numpy(),
            df["root_rotateZ"].to_numpy(),
        ],
        axis=1,
    ).astype(np.float64)

    degrees = angle_unit == "deg"
    if angle_unit not in {"rad", "deg"}:
        raise ValueError(f"Unsupported angle_unit: {angle_unit}")

    root_quat_xyzw = transform.Rotation.from_euler("xyz", euler, degrees=degrees).as_quat().astype(np.float32)

    joint_cols = infer_joint_cols(df)
    dof = df[joint_cols].to_numpy(dtype=np.float32)

    # [OOM 방어] DataFrame 추출이 끝났으므로 즉시 메모리 해제
    del df
    gc.collect()

    if angle_unit == "deg":
        dof = np.deg2rad(dof).astype(np.float32)

    if joint_order == "il":
        dof = dof[:, H2_ISAACLAB_TO_MUJOCO_DOF]
    elif joint_order != "mj":
        raise ValueError(f"Unsupported joint_order: {joint_order}")

    return root_pos, root_quat_xyzw, dof


def make_motion_entry(root_pos, root_quat_xyzw, dof, fps: int):
    T = dof.shape[0]

    pose_aa = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_quat_xyzw).as_rotvec().astype(np.float32)
    pose_aa[:, 1:, :] = DOF_AXIS[None, :, :] * dof[:, :, None]

    return {
        "root_trans_offset": root_pos.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof.astype(np.float32),
        "root_rot": root_quat_xyzw.astype(np.float32),
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),
        "fps": int(fps),
    }


def print_stats(name: str, entry: dict):
    root = entry["root_trans_offset"]
    dof = entry["dof"]
    pose = entry["pose_aa"]
    print(f"  {name}")
    print(f"    frames: {root.shape[0]}, fps: {entry['fps']}")
    print(f"    root z min/max/mean: {root[:,2].min():.4f} / {root[:,2].max():.4f} / {root[:,2].mean():.4f}")
    print(f"    root std mean: {root.std(axis=0).mean():.6f}")
    print(f"    dof std mean: {dof.std(axis=0).mean():.6f}")
    print(f"    pose_aa std mean: {pose.reshape(pose.shape[0], -1).std(axis=0).mean():.6f}")


def save_chunk(motion_lib, base_output: Path, chunk_idx: int, is_chunked: bool):
    """지정된 경로에 현재까지 모인 모션 라이브러리를 저장합니다."""
    if not motion_lib:
        return
        
    out_path = base_output
    if is_chunked:
        # 분할 저장일 경우 파일명에 _part001, _part002 등을 붙임
        out_path = base_output.with_name(f"{base_output.stem}_part{chunk_idx:03d}{base_output.suffix}")
        
    joblib.dump(motion_lib, out_path, compress=True)
    print(f"--> [Chunk Saved] {out_path} ({len(motion_lib)} motions)")


def main():
    parser = argparse.ArgumentParser(description="Convert H2 SOMA flat CSV(s) to SONIC H2 motion_lib PKL")
    parser.add_argument("--input", required=True, help="CSV file or directory containing CSVs")
    parser.add_argument("--output", required=True, help="Output motion_lib PKL path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--position_unit", choices=["m", "cm"], default="m")
    parser.add_argument("--angle_unit", choices=["rad", "deg"], default="rad")
    parser.add_argument("--joint_order", choices=["mj", "il"], default="mj")
    parser.add_argument("--root_z_offset", type=float, default=0.0)
    # [OOM 방어] 일정 개수마다 잘라서 저장하는 옵션 추가
    parser.add_argument("--chunk_size", type=int, default=0, help="If >0, saves to multiple PKL files to prevent OOM (e.g., 500)")
    parser.add_argument(
    "--individual",
    action="store_true",
    help="Write one PKL per motion into output directory",
    )
    args = parser.parse_args()

    csvs = collect_csvs(args.input)
    if not csvs:
        print("ERROR: no CSVs found")
        sys.exit(1)

    print(f"H2 converter: {NUM_DOF} DOFs, {NUM_BODIES} bodies")
    print(f"input CSVs: {len(csvs)}")
    print(f"position_unit={args.position_unit}, angle_unit={args.angle_unit}, joint_order={args.joint_order}, root_z_offset={args.root_z_offset}")
    if args.chunk_size > 0:
        print(f"chunk_size={args.chunk_size} (Will save in multiple parts)")

    out = Path(args.output)
    
    
    if args.individual:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        converted = 0
        failed = 0

        for csv_path in csvs:
            name = csv_path.stem
            out_path = out_dir / f"{name}.pkl"

            if out_path.exists():
                print(f"SKIP existing: {out_path}")
                converted += 1
                continue

            try:
                root_pos, root_quat_xyzw, dof = load_h2_flat_csv(
                    csv_path=csv_path,
                    fps=args.fps,
                    position_unit=args.position_unit,
                    angle_unit=args.angle_unit,
                    joint_order=args.joint_order,
                    root_z_offset=args.root_z_offset,
                )
                entry = make_motion_entry(root_pos, root_quat_xyzw, dof, args.fps)

                # 공식 motion_lib individual 구조: {motion_name: entry}
                joblib.dump({name: entry}, out_path, compress=True)

                print(f"saved: {out_path}")
                converted += 1

            except Exception as exc:
                print(f"FAILED {csv_path}: {type(exc).__name__}: {exc}")
                failed += 1

        print(f"Done: converted={converted}, failed={failed}, total={len(csvs)}")
        return
    motion_lib = {}
    chunk_idx = 1
    total_processed = 0

    for csv_path in csvs:
        name = csv_path.stem
        try:
            root_pos, root_quat_xyzw, dof = load_h2_flat_csv(
                csv_path=csv_path,
                fps=args.fps,
                position_unit=args.position_unit,
                angle_unit=args.angle_unit,
                joint_order=args.joint_order,
                root_z_offset=args.root_z_offset,
            )
            entry = make_motion_entry(root_pos, root_quat_xyzw, dof, args.fps)
            motion_lib[name] = entry
            print_stats(name, entry)
            total_processed += 1

            # 청크 크기에 도달하면 디스크에 저장하고 메모리 비우기
            if args.chunk_size > 0 and len(motion_lib) >= args.chunk_size:
                save_chunk(motion_lib, out, chunk_idx, True)
                motion_lib.clear()  # 딕셔너리 비우기
                gc.collect()        # 강제 가비지 컬렉션
                chunk_idx += 1

        except Exception as exc:
            print(f"FAILED {csv_path}: {type(exc).__name__}: {exc}")

    # 남아있는 데이터 최종 저장
    if motion_lib:
        is_chunked = (args.chunk_size > 0)
        save_chunk(motion_lib, out, chunk_idx, is_chunked)

    if total_processed == 0:
        print("ERROR: no motions converted")
        sys.exit(1)
    else:
        print(f"Done! Successfully processed {total_processed} motions.")


if __name__ == "__main__":
    main()