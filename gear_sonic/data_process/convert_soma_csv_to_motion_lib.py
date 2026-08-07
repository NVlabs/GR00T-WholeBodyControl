#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Convert SOMA retargeter CSV/PKL data to motion_lib format for SONIC training.

SOMA retargeter outputs G1 29-DOF motion data as CSV files (joint_pos.csv,
body_pos.csv, body_quat.csv) or as a joblib PKL with the same fields. This
script converts that data into the motion_lib PKL format expected by SONIC
training (root_trans_offset, pose_aa, dof, root_rot, fps).

Supports five input modes:
  1. Single motion directory with CSVs (joint_pos.csv, body_pos.csv, body_quat.csv)
  2. Parent directory containing multiple motion subdirectories
  3. Deploy PKL file (joblib dict with joint_pos, body_pos_w, body_quat_w per sequence)
  4. Directory of flat Bones-SEED CSVs (single CSV per motion, degrees+cm)
  5. Parent directory of session dirs containing Bones-SEED CSVs

Usage:
    # Single CSV directory
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input data/soma_retarget/tired_squat_003__A360 \
        --output data/soma_test.pkl --fps 50

    # Batch: parent dir with multiple motion subdirs
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input data/soma_retarget/all_demo_4seqs \
        --output data/soma_demo_4seqs.pkl --fps 50

    # Deploy PKL file
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input data/soma_retarget/bones_test.pkl \
        --output data/soma_bones_test.pkl --fps 50

    # Bones-SEED: directory of flat CSVs (single session)
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input /path/to/bones_SEED/g1/csv/210531 \
        --output data/bones_seed_210531.pkl --fps 50

    # Bones-SEED: all sessions (parent dir)
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input /path/to/bones_SEED/g1/csv \
        --output data/bones_seed_all.pkl --fps 50
"""

import argparse
import os
import sys

import joblib
import numpy as np
from scipy.spatial import transform

# IsaacLab ↔ MuJoCo joint reordering (29 DOFs for G1).
# MJ_TO_IL[mj] = il: for MuJoCo DOF index mj, gives the IsaacLab index il.
# Source: external_dependencies/SONIC_Web/demo_python.py
MJ_TO_IL = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int32,
)

# G1 29-DOF axis definitions (from Humanoid_Batch / g1_29dof_rev_1_0.xml).
# Each DOF rotates around a single axis. Hardcoded to avoid torch dependency.
NUM_DOF = 29
NUM_BODIES = 30  # pelvis + 29 actuated links
DOF_AXIS = np.array(
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],  # left leg
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],  # right leg
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],  # waist
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],  # left arm
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],  # right arm
    ],
    dtype=np.float32,
)


# Joint names in Bones-SEED CSV column order (after Frame + 6 root columns).
# These are in MuJoCo/MJCF actuator order (same as g1_29dof_rev_1_0.xml motors).
BONES_CSV_JOINT_NAMES = [
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]


def load_bones_csv(csv_path: str) -> dict:
    """Load a single Bones-SEED flat CSV motion file.

    Bones-SEED CSV format: Frame, root_translate{X,Y,Z}, root_rotate{X,Y,Z}, 29 joint DOFs.
    All angles in degrees, positions in centimeters.
    """
    import pandas as pd

    data = pd.read_csv(csv_path)
    T = len(data)

    # Root position: cm → meters
    root_pos = (
        np.stack(
            [
                data["root_translateX"].values,  # noqa: PD011
                data["root_translateY"].values,  # noqa: PD011
                data["root_translateZ"].values,  # noqa: PD011
            ],
            axis=1,
        ).astype(np.float32)
        / 100.0
    )  # cm → m

    # Root rotation: Euler xyz (intrinsic) degrees → quaternion (xyzw scipy convention)
    # Reference: gear_sonic/data_process/process_bones_to_motionlib.py uses "xyz" (intrinsic)
    euler_deg = np.stack(
        [
            data["root_rotateX"].values,  # noqa: PD011
            data["root_rotateY"].values,  # noqa: PD011
            data["root_rotateZ"].values,  # noqa: PD011
        ],
        axis=1,
    ).astype(np.float64)
    root_quat_xyzw = (
        transform.Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float32)
    )
    # Convert xyzw → wxyz for body_quat_w format
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    # Joint DOFs: degrees → radians, already in MuJoCo/MJCF actuator order
    joint_cols = [c for c in data.columns if c.endswith("_dof")]
    joint_pos_mj = np.deg2rad(data[joint_cols].values).astype(np.float32)  # (T, 29)

    # Create dummy body_pos_w and body_quat_w (only root body populated, rest zeros)
    # The converter only uses body_pos_w[:,0] for root_trans and body_quat_w[:,0] for root_rot
    body_pos_w = np.zeros((T, 14, 3), dtype=np.float32)
    body_pos_w[:, 0, :] = root_pos
    body_quat_w = np.zeros((T, 14, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0  # identity quaternion wxyz
    body_quat_w[:, 0, :] = root_quat_wxyz

    return {
        "joint_pos": joint_pos_mj,  # (T, 29) MuJoCo order, radians
        "body_pos_w": body_pos_w,  # (T, 14, 3)
        "body_quat_w": body_quat_w,  # (T, 14, 4) wxyz
        "joint_order": "mj",  # already in MuJoCo order, skip IL→MJ reorder
    }


def load_csv_motion(motion_dir: str) -> dict:
    """Load a single motion from a directory of CSV files."""
    joint_pos_f = os.path.join(motion_dir, "joint_pos.csv")
    body_pos_f = os.path.join(motion_dir, "body_pos.csv")
    body_quat_f = os.path.join(motion_dir, "body_quat.csv")

    if not os.path.exists(joint_pos_f):
        return None

    joint_pos = np.loadtxt(joint_pos_f, delimiter=",", skiprows=1, dtype=np.float32)
    body_pos = np.loadtxt(body_pos_f, delimiter=",", skiprows=1, dtype=np.float32)
    body_quat = np.loadtxt(body_quat_f, delimiter=",", skiprows=1, dtype=np.float32)

    # Reshape body data: (T, 14*3) → (T, 14, 3), (T, 14*4) → (T, 14, 4)
    T = joint_pos.shape[0]
    body_pos = body_pos.reshape(T, -1, 3)
    body_quat = body_quat.reshape(T, -1, 4)

    return {
        "joint_pos": joint_pos,  # (T, 29) IsaacLab order
        "body_pos_w": body_pos,  # (T, 14, 3) world frame
        "body_quat_w": body_quat,  # (T, 14, 4) wxyz format
    }


def convert_sequence(seq_data: dict, fps: int, humanoid_fk=None) -> dict:  # noqa: ARG001
    """Convert a single deploy-format sequence to motion_lib format.

    Args:
        seq_data: dict with joint_pos (T, 29), body_pos_w (T, 14, 3),
                  body_quat_w (T, 14, 4 wxyz)
        fps: frame rate of the input data
        humanoid_fk: Optional Humanoid_Batch instance (unused, kept for compat)

    Returns:
        motion_lib entry dict with root_trans_offset, pose_aa, dof, root_rot, fps
    """
    joint_pos = seq_data["joint_pos"]  # (T, 29)
    body_pos_w = seq_data["body_pos_w"]  # (T, 14, 3)
    body_quat_w = seq_data["body_quat_w"]  # (T, 14, 4) wxyz
    joint_order = seq_data.get("joint_order", "il")  # "il" or "mj"

    T = joint_pos.shape[0]

    # 1. Root position: body_0 (pelvis) position
    root_trans_offset = body_pos_w[:, 0, :].copy()  # (T, 3)

    # 2. Root quaternion: body_0 quaternion, convert wxyz → xyzw (scipy convention)
    root_quat_wxyz = body_quat_w[:, 0, :]  # (T, 4) [w, x, y, z]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]  # (T, 4) [x, y, z, w]

    # 3. Reorder DOFs to MuJoCo order if needed
    if joint_order == "il":
        # Input is IsaacLab order → reorder to MuJoCo (MJCF actuator order)
        dof_mj = joint_pos[:, MJ_TO_IL]  # (T, 29)
    else:
        # Input is already in MuJoCo order (e.g., Bones-SEED CSVs)
        dof_mj = joint_pos  # (T, 29)

    # 4. Convert DOF → pose_aa using hardcoded G1 axis definitions
    dof = dof_mj[:, :NUM_DOF]

    # pose_aa[body_idx] = dof_axis * dof_value (axis-angle representation)
    # Body 0 = pelvis (root), bodies 1-29 = actuated joints
    pose_aa = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    # Actuated joints: body idx = dof idx + 1
    pose_aa[:, 1:NUM_BODIES, :] = DOF_AXIS[None, :, :] * dof[:, :, None]

    # Set root rotation as axis-angle
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_quat_xyzw).as_rotvec()

    return {
        "root_trans_offset": root_trans_offset.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof.astype(np.float32),
        "root_rot": root_quat_xyzw.astype(np.float32),  # xyzw (scipy convention)
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),  # placeholder
        "fps": fps,
    }


def downsample_sequence(entry: dict, fps_source: int, fps_target: int) -> dict:
    """Downsample a motion_lib entry using stride-based frame skipping.

    Matches process_bones_to_motionlib.py: jump = int(fps_source / fps_target).
    Best used when fps_source is an exact multiple of fps_target (e.g. 120→30).
    The resulting PKL is stored at fps_target; fk_batch handles the final
    resampling to target_fps at load time using the canonical interploate_pose formula.
    """
    if fps_source == fps_target:
        return entry
    jump = int(fps_source / fps_target)
    if jump <= 1:
        return entry
    return {
        "root_trans_offset": entry["root_trans_offset"][::jump],
        "pose_aa": entry["pose_aa"][::jump],
        "dof": entry["dof"][::jump],
        "root_rot": entry["root_rot"][::jump],
        "smpl_joints": entry["smpl_joints"][::jump],
        "fps": fps_target,
    }


def init_humanoid_fk():
    """Initialize Humanoid_Batch from the G1 MJCF config.

    Only needed for non-Bones-SEED inputs (deploy PKL, SOMA CSV dirs).
    Bones-SEED path uses hardcoded DOF_AXIS constants instead.
    """
    import omegaconf

    motion_cfg = omegaconf.OmegaConf.create(
        {
            "asset": {
                "assetRoot": "gear_sonic/data/assets/robot_description/mjcf/",
                "assetFileName": "g1_29dof_rev_1_0.xml",
                "urdfFileName": "",
            },
            "extend_config": [],
        }
    )
    from gear_sonic.utils.motion_lib import torch_humanoid_batch

    return torch_humanoid_batch.Humanoid_Batch(motion_cfg)


def process_session_csvs(args_tuple):
    """Process all CSVs in a single session directory. Used by multiprocessing."""
    session_dir, session_name, out_dir, fps, fps_source = args_tuple
    import warnings

    warnings.filterwarnings("ignore")

    csv_files = sorted([f for f in os.listdir(session_dir) if f.endswith(".csv")])

    session_out = os.path.join(out_dir, session_name)
    os.makedirs(session_out, exist_ok=True)

    converted = 0
    failed = 0
    for csv_f in csv_files:
        name = os.path.splitext(csv_f)[0]
        out_path = os.path.join(session_out, name + ".pkl")
        if os.path.exists(out_path):
            converted += 1  # skip existing
            continue
        try:
            seq = load_bones_csv(os.path.join(session_dir, csv_f))
            fps_for_convert = fps_source if fps_source else fps
            entry = convert_sequence(seq, fps_for_convert)
            if fps_source and fps_source != fps:
                entry = downsample_sequence(entry, fps_source, fps)
            joblib.dump({name: entry}, out_path, compress=True)
            converted += 1
        except Exception:  # noqa: BLE001
            failed += 1
    return session_name, converted, failed, len(csv_files)


def main():
    parser = argparse.ArgumentParser(description="Convert SOMA CSV/PKL to motion_lib format")
    parser.add_argument(
        "--input", required=True, help="CSV dir, parent dir of CSV dirs, or deploy PKL"
    )
    parser.add_argument(
        "--output", required=True, help="Output path (PKL file or directory for individual PKLs)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target output FPS (default: 30, matches process_bones_to_motionlib)",
    )
    parser.add_argument(
        "--fps_source",
        type=int,
        default=None,
        help="Source data FPS. If set and != --fps, data is downsampled. "
        "Bones-SEED CSVs are typically 120fps.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Write individual PKLs per motion (preserves session dir structure)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for --individual mode",
    )
    args = parser.parse_args()

    print(f"G1 {NUM_DOF} DOFs, {NUM_BODIES} bodies (hardcoded axes)")

    # Individual PKL mode: skip scanning, go straight to parallel per-session processing
    if args.individual:
        if not os.path.isdir(args.input):
            print("ERROR: --individual requires a directory input")
            sys.exit(1)

        # Detect: is input a single session dir (contains CSVs) or parent of sessions?
        has_csvs = any(f.endswith(".csv") for f in os.listdir(args.input))
        subdirs = sorted(
            [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
        )
        has_session_subdirs = (
            any(
                any(f.endswith(".csv") for f in os.listdir(os.path.join(args.input, d)))
                for d in subdirs[:3]
            )
            if subdirs
            else False
        )

        session_dirs = []
        if has_session_subdirs:
            for d in subdirs:
                subdir = os.path.join(args.input, d)
                if any(f.endswith(".csv") for f in os.listdir(subdir)):
                    session_dirs.append((subdir, d, args.output, args.fps, args.fps_source))
        elif has_csvs:
            session_name = os.path.basename(args.input.rstrip("/"))
            session_dirs.append((args.input, session_name, args.output, args.fps, args.fps_source))

        print(f"\nBatch converting {len(session_dirs)} sessions with {args.num_workers} workers")
        print(f"Output: {args.output}")
        os.makedirs(args.output, exist_ok=True)

        import multiprocessing

        total_converted = 0
        total_failed = 0
        total_csvs = 0
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            for session_name, converted, failed, n_csvs in pool.imap_unordered(
                process_session_csvs, session_dirs
            ):
                total_converted += converted
                total_failed += failed
                total_csvs += n_csvs
                print(
                    f"  {session_name}: {converted}/{n_csvs} converted"
                    + (f" ({failed} failed)" if failed else "")
                )

        print(
            f"\nDone: {total_converted} motions converted, {total_failed} failed, {total_csvs} total CSVs"
        )
        return

    # Detect input mode (combined PKL output path)
    sequences = {}

    if args.input.endswith(".pkl"):
        # Mode 3: Deploy PKL file
        print(f"Loading deploy PKL: {args.input}")
        data = joblib.load(args.input)
        for name, seq in data.items():
            sequences[name] = seq
        print(f"  Found {len(sequences)} sequences")

    elif os.path.isfile(os.path.join(args.input, "joint_pos.csv")):
        # Mode 1: Single CSV directory
        name = os.path.basename(args.input)
        print(f"Loading single CSV motion: {name}")
        seq = load_csv_motion(args.input)
        if seq is None:
            print("ERROR: joint_pos.csv not found")
            sys.exit(1)
        sequences[name] = seq
        print(f"  {seq['joint_pos'].shape[0]} frames")

    elif os.path.isdir(args.input):
        # Check if directory contains flat CSVs (Bones-SEED format)
        csv_files = sorted([f for f in os.listdir(args.input) if f.endswith(".csv")])
        subdirs = sorted(
            [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
        )

        if csv_files and not any(
            os.path.exists(os.path.join(args.input, d, "joint_pos.csv"))
            for d in subdirs[:5]  # check first 5 subdirs
        ):
            # Mode 4: Directory of flat Bones-SEED CSVs
            print(f"Scanning directory for Bones-SEED CSVs: {args.input}")
            for csv_f in csv_files:
                csv_path = os.path.join(args.input, csv_f)
                name = os.path.splitext(csv_f)[0]
                try:
                    seq = load_bones_csv(csv_path)
                    sequences[name] = seq
                except Exception as e:  # noqa: BLE001
                    print(f"  WARNING: Failed to load {csv_f}: {e}")
            print(f"  Found {len(sequences)} Bones-SEED CSV motions")
        elif subdirs:
            # Check if subdirs contain flat CSVs (batch of session dirs)
            has_session_csvs = False
            for dname in subdirs[:3]:
                subdir = os.path.join(args.input, dname)
                sub_csvs = [f for f in os.listdir(subdir) if f.endswith(".csv")]
                if sub_csvs and not os.path.exists(os.path.join(subdir, "joint_pos.csv")):
                    has_session_csvs = True
                    break

            if has_session_csvs:
                # Mode 5: Parent dir of session dirs containing Bones-SEED CSVs
                print(f"Scanning session directories for Bones-SEED CSVs: {args.input}")
                for dname in sorted(subdirs):
                    subdir = os.path.join(args.input, dname)
                    sub_csvs = sorted([f for f in os.listdir(subdir) if f.endswith(".csv")])
                    for csv_f in sub_csvs:
                        csv_path = os.path.join(subdir, csv_f)
                        name = os.path.splitext(csv_f)[0]
                        try:
                            seq = load_bones_csv(csv_path)
                            sequences[name] = seq
                        except Exception as e:  # noqa: BLE001
                            print(f"  WARNING: Failed to load {dname}/{csv_f}: {e}")
                    if sub_csvs:
                        print(f"  Session {dname}: {len(sub_csvs)} CSVs")
                print(f"  Found {len(sequences)} total Bones-SEED CSV motions")
            else:
                # Mode 2: Parent directory with SOMA-style subdirectories
                print(f"Scanning directory: {args.input}")
                for dname in sorted(subdirs):
                    subdir = os.path.join(args.input, dname)
                    seq = load_csv_motion(subdir)
                    if seq is not None:
                        sequences[dname] = seq
                print(f"  Found {len(sequences)} motion directories with CSVs")
    else:
        print(f"ERROR: {args.input} is not a valid input")
        sys.exit(1)

    if not sequences:
        print("ERROR: No sequences found")
        sys.exit(1)

    # Convert each sequence (combined PKL mode)
    motion_lib_dict = {}
    for name, seq_data in sequences.items():
        T = seq_data["joint_pos"].shape[0]
        print(f"  Converting {name}: {T} frames @ {args.fps} fps")
        fps_for_convert = args.fps_source if args.fps_source else args.fps
        entry = convert_sequence(seq_data, fps_for_convert)
        if args.fps_source and args.fps_source != args.fps:
            entry = downsample_sequence(entry, args.fps_source, args.fps)
        motion_lib_dict[name] = entry

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"\nSaving motion_lib PKL: {args.output}")
    joblib.dump(motion_lib_dict, args.output, compress=True)
    print(f"Done: {len(motion_lib_dict)} sequences saved")


if __name__ == "__main__":
    main()
