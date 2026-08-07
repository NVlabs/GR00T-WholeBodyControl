#!/usr/bin/env python3
# ruff: noqa: T201, BLE001, DOC
"""Extract SOMA skeleton FK joint positions from BVH files.

Parses NOVA-skeleton BVH files and computes forward kinematics to extract
world-space 3D joint positions for a selected 26-joint subset. Outputs
per-motion PKL files in the same directory structure as the robot PKLs.

The 26 selected joints cover the major body landmarks with symmetric arms
(including Thumb1 + Middle1 per hand for orientation): hips, spine chain,
shoulders, arms, hands+fingers, legs, feet.

Input:  BVH files from bones_update_240924/anims_uniform_novaskel_v1/BVH/
Output: Per-motion PKL files with soma_joints (T, 26, 3) Z-up meters body-local,
        soma_root_quat (T, 4) wxyz Y-up BVH world rotation

Usage:
    # Single session
    python scripts/motion/extract_soma_joints_from_bvh.py \
        --input /path/to/novaskel_v1/BVH/210531 \
        --output /path/to/output/bones_soma_joints/210531 \
        --fps 30

    # All sessions (parent dir)
    python scripts/motion/extract_soma_joints_from_bvh.py \
        --input /path/to/novaskel_v1/BVH \
        --output /path/to/output/bones_soma_joints \
        --fps 30 --num_workers 8
"""

import argparse
import glob
import multiprocessing
import os
import os.path as osp
import re
import sys
import time

import joblib
import numpy as np
from scipy.spatial import transform

# 26-joint subset of the 78-joint NOVA skeleton (Root excluded).
# Covers major body landmarks, excluding most fingers, face details, end sites.
# Arms are fully symmetric with two finger joints per hand (Thumb1 + Middle1)
# to determine hand orientation.
SOMA_JOINTS = [
    "Hips",  # 0  - pelvis
    "Spine1",  # 1  - lower spine
    "Spine2",  # 2  - mid spine
    "Chest",  # 3  - upper spine
    "Neck1",  # 4  - neck
    "Head",  # 5  - head
    "LeftShoulder",  # 6  - left clavicle
    "LeftArm",  # 7  - left upper arm
    "LeftForeArm",  # 8  - left elbow
    "LeftHand",  # 9  - left wrist
    "LeftHandThumb1",  # 10 - left thumb (hand orientation)
    "LeftHandMiddle1",  # 11 - left middle finger (hand orientation)
    "RightShoulder",  # 12 - right clavicle
    "RightArm",  # 13 - right upper arm
    "RightForeArm",  # 14 - right elbow
    "RightHand",  # 15 - right wrist
    "RightHandThumb1",  # 16 - right thumb (hand orientation)
    "RightHandMiddle1",  # 17 - right middle finger (hand orientation)
    "LeftLeg",  # 18 - left hip / upper leg
    "LeftShin",  # 19 - left knee
    "LeftFoot",  # 20 - left ankle
    "LeftToeBase",  # 21 - left toe
    "RightLeg",  # 22 - right hip / upper leg
    "RightShin",  # 23 - right knee
    "RightFoot",  # 24 - right ankle
    "RightToeBase",  # 25 - right toe
]

NUM_SOMA_JOINTS = len(SOMA_JOINTS)


def parse_bvh(filepath):
    """Parse BVH hierarchy and motion data.

    Returns:
        joints: list of dicts with name, offset, channels, parent_idx
        channel_order: list of (joint_idx, channel_name) tuples
        motion_data: (n_frames, n_channels) numpy array
        n_frames: int
        frame_time: float (seconds per frame)
    """
    with open(filepath) as f:
        lines = f.readlines()

    joints = []
    joint_stack = []
    channel_order = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line == "MOTION":
            i += 1
            break

        m = re.match(r"(ROOT|JOINT)\s+(\S+)", line)
        if m:
            name = m.group(2)
            parent_idx = joint_stack[-1] if joint_stack else -1
            joints.append({"name": name, "offset": None, "channels": [], "parent_idx": parent_idx})
            joint_stack.append(len(joints) - 1)
        elif line.startswith("OFFSET") and joint_stack:
            vals = [float(x) for x in line.split()[1:]]
            joints[joint_stack[-1]]["offset"] = np.array(vals)
        elif line.startswith("CHANNELS") and joint_stack:
            parts = line.split()
            n_ch = int(parts[1])
            ch_names = parts[2 : 2 + n_ch]
            joints[joint_stack[-1]]["channels"] = ch_names
            for ch in ch_names:
                channel_order.append((joint_stack[-1], ch))
        elif line == "}":
            if joint_stack:
                joint_stack.pop()
        i += 1

    # Parse MOTION section
    frames_line = lines[i].strip()
    n_frames = int(frames_line.split(":")[1])
    i += 1
    frame_time = float(lines[i].strip().split(":")[1])
    i += 1

    motion_data = np.empty((n_frames, len(channel_order)))
    for f_idx in range(n_frames):
        vals = lines[i].strip().split()
        motion_data[f_idx] = [float(v) for v in vals]
        i += 1

    return joints, channel_order, motion_data, n_frames, frame_time


def compute_fk_selected(joints, channel_order, motion_data, selected_names):
    """Compute FK world positions for selected joints only.

    Uses vectorized rotation computation per joint across all frames.

    Args:
        joints: parsed joint hierarchy
        channel_order: channel mapping
        motion_data: (n_frames, n_channels)
        selected_names: list of joint names to extract

    Returns:
        selected_positions: (n_frames, len(selected_names), 3) in BVH units (cm)
        root_quats: (n_frames, 4) root orientation quaternions (xyzw)
    """
    n_frames = motion_data.shape[0]
    n_joints = len(joints)
    joint_names = [j["name"] for j in joints]

    # Build selected indices
    selected_indices = set()
    for name in selected_names:
        if name in joint_names:
            selected_indices.add(joint_names.index(name))

    # Also include all ancestors needed for FK
    ancestors = set()
    for idx in selected_indices:
        j = idx
        while j >= 0:
            ancestors.add(j)
            j = joints[j]["parent_idx"]
    compute_joints = sorted(ancestors | selected_indices)

    # Pre-compute per-joint channel indices
    joint_channels = {j: [] for j in range(n_joints)}
    for ch_idx, (j_idx, ch_name) in enumerate(channel_order):
        joint_channels[j_idx].append((ch_idx, ch_name))

    # Compute FK for all frames
    world_rots = np.zeros((n_frames, n_joints, 3, 3))
    world_pos = np.zeros((n_frames, n_joints, 3))

    for j_idx in compute_joints:
        joint = joints[j_idx]
        offset = joint["offset"] if joint["offset"] is not None else np.zeros(3)

        # Extract position and rotation channels
        pos_channels = {}
        rot_order = ""
        rot_ch_indices = []
        for ch_idx, ch_name in joint_channels[j_idx]:
            if ch_name.endswith("position"):
                pos_channels[ch_name] = ch_idx
            elif ch_name.endswith("rotation"):
                rot_order += ch_name[0].lower()
                rot_ch_indices.append(ch_idx)

        # Local position (all frames)
        has_pos_channels = bool(pos_channels)
        if has_pos_channels:
            # Joints with position channels: use channels directly (not additive to offset)
            local_pos = np.zeros((n_frames, 3))
            if "Xposition" in pos_channels:
                local_pos[:, 0] = motion_data[:, pos_channels["Xposition"]]
            if "Yposition" in pos_channels:
                local_pos[:, 1] = motion_data[:, pos_channels["Yposition"]]
            if "Zposition" in pos_channels:
                local_pos[:, 2] = motion_data[:, pos_channels["Zposition"]]
        else:
            # Joints with only rotation channels: use static offset
            local_pos = np.tile(offset, (n_frames, 1))

        # Local rotation (all frames)
        # BVH uses extrinsic rotations: uppercase in scipy convention
        if rot_order:
            rot_vals = motion_data[:, rot_ch_indices]  # (n_frames, n_rot_channels)
            local_rot = transform.Rotation.from_euler(
                rot_order.upper(), rot_vals, degrees=True
            ).as_matrix()
        else:
            local_rot = np.tile(np.eye(3), (n_frames, 1, 1))

        if joint["parent_idx"] < 0:
            # Root joint: no parent transform
            world_rots[:, j_idx] = local_rot
            world_pos[:, j_idx] = local_pos
        else:
            p = joint["parent_idx"]
            parent_rot = world_rots[:, p]  # (n_frames, 3, 3)
            parent_pos = world_pos[:, p]  # (n_frames, 3)
            # world_pos = parent_pos + parent_rot @ local_pos
            world_pos[:, j_idx] = parent_pos + np.einsum("fij,fj->fi", parent_rot, local_pos)
            # world_rot = parent_rot @ local_rot
            world_rots[:, j_idx] = np.einsum("fij,fjk->fik", parent_rot, local_rot)

    # Extract selected joints
    sel_indices = [joint_names.index(name) for name in selected_names if name in joint_names]
    selected_positions = world_pos[:, sel_indices, :]  # (n_frames, len(selected_names), 3)

    # Extract root quaternion (Hips joint, index 1)
    hips_idx = joint_names.index("Hips") if "Hips" in joint_names else 0
    root_quats_scipy = transform.Rotation.from_matrix(world_rots[:, hips_idx])
    root_quats = root_quats_scipy.as_quat()  # (n_frames, 4) as xyzw

    return selected_positions, root_quats


def process_single_bvh(args):
    """Process a single BVH file → PKL with soma_joints.

    Returns (motion_name, success, error_msg)
    """
    bvh_path, output_dir, fps_target, skip_existing = args
    motion_name = osp.splitext(osp.basename(bvh_path))[0]
    output_path = osp.join(output_dir, f"{motion_name}.pkl")

    if skip_existing and osp.exists(output_path):
        return motion_name, True, "skipped"

    try:
        joints, channel_order, motion_data, n_frames, frame_time = parse_bvh(bvh_path)
        fps_source = round(1.0 / frame_time)

        positions, root_quats = compute_fk_selected(joints, channel_order, motion_data, SOMA_JOINTS)

        # Convert cm → meters
        positions_m = positions / 100.0

        # Extract hips translation and subtract from all joints to get
        # body-local positions (matching SMPL's compute_human_joints which
        # produces joints without global translation).
        hips_idx = 0  # Hips is joint index 0 in SOMA_JOINTS (Root removed)
        transl = positions_m[:, hips_idx, :].copy()  # (T, 3) Y-up
        positions_m = positions_m - transl[:, None, :]  # body-local

        # Convert Y-up → Z-up: (x, y, z) → (x, -z, y)
        # Same as applying rot90x, matching SMPL's convert_smpl_bones which
        # applies rot90x to global_orient before FK to produce Z-up joints.
        positions_zup = positions_m.copy()
        positions_zup[..., 1] = -positions_m[..., 2]
        positions_zup[..., 2] = positions_m[..., 1]

        # Downsample to target fps using stride-based frame skipping.
        # Matches convert_soma_csv_to_motion_lib.py: jump = int(fps_source / fps_target).
        # For Bones-SEED (120fps BVH → 30fps), this is stride-4 (exact division).
        # Both BVH and CSV have identical source frame counts at 120fps, so
        # stride-based downsampling produces identical frame counts.
        if fps_source != fps_target:
            jump = max(1, int(fps_source / fps_target))
            positions_zup = positions_zup[::jump]
            transl = transl[::jump]
            root_quats = root_quats[::jump]

        # Convert xyzw → wxyz for compatibility with IsaacLab quat pipeline.
        # Root quats stay Y-up — runtime converts via smpl_root_ytoz_up +
        # remove_bvh_base_rot (same pattern as SMPL pose_aa).
        root_quats = root_quats[:, [3, 0, 1, 2]]

        # Store as PKL
        entry = {
            motion_name: {
                "soma_joints": positions_zup.astype(
                    np.float32
                ),  # (T, 26, 3) Z-up meters, body-local
                "soma_root_quat": root_quats.astype(
                    np.float32
                ),  # (T, 4) wxyz, Y-up BVH world rotation
                "soma_transl": transl.astype(np.float32),  # (T, 3) Hips world position, Y-up
                "fps": fps_target,
                "joint_names": SOMA_JOINTS,
            }
        }

        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(entry, output_path)
        return motion_name, True, None

    except Exception as e:
        return motion_name, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Extract SOMA joints from BVH files")
    parser.add_argument("--input", required=True, help="BVH dir (session or parent)")
    parser.add_argument("--output", required=True, help="Output dir for PKL files")
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS (default: 30, matches process_bones_to_motionlib)",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing PKL files")
    args = parser.parse_args()

    # Discover BVH files
    bvh_files = sorted(glob.glob(osp.join(args.input, "*.bvh")))

    if bvh_files:
        # Single session directory
        sessions = {osp.basename(args.input): bvh_files}
    else:
        # Parent directory with session subdirs
        session_dirs = sorted([d for d in glob.glob(osp.join(args.input, "*")) if osp.isdir(d)])
        sessions = {}
        for sd in session_dirs:
            files = sorted(glob.glob(osp.join(sd, "*.bvh")))
            if files:
                sessions[osp.basename(sd)] = files

    if not sessions:
        print(f"No BVH files found in {args.input}")
        sys.exit(1)

    total_bvh = sum(len(v) for v in sessions.values())
    print(f"Found {total_bvh} BVH files across {len(sessions)} sessions")
    print(f"Output: {args.output}, FPS: {args.fps}, Workers: {args.num_workers}")
    total_converted = 0
    total_failed = 0
    t0 = time.time()

    for session_name, bvh_list in sessions.items():
        session_output = osp.join(args.output, session_name)
        tasks = [(bvh_path, session_output, args.fps, args.skip_existing) for bvh_path in bvh_list]

        with multiprocessing.Pool(args.num_workers) as pool:
            results = pool.map(process_single_bvh, tasks)

        converted = sum(1 for _, s, e in results if s and e != "skipped")
        skipped = sum(1 for _, s, e in results if e == "skipped")
        failed = sum(1 for _, s, _ in results if not s)

        if failed > 0:
            for name, success, err in results:
                if not success:
                    print(f"  FAILED: {name}: {err}")

        total_converted += converted + skipped
        total_failed += failed

        elapsed = time.time() - t0
        rate = total_converted / elapsed if elapsed > 0 else 0
        print(
            f"  {session_name}: {converted} converted, {skipped} skipped, "
            f"{failed} failed [{total_converted}/{total_bvh}, {rate:.0f}/s]"
        )

    elapsed = time.time() - t0
    print(f"\nDone: {total_converted} converted, {total_failed} failed, " f"{elapsed:.1f}s elapsed")


if __name__ == "__main__":
    main()
