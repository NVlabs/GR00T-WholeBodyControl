#!/usr/bin/env python3
"""Capture raw body-tracking joints for headset comparison (Pico vs Quest).

Why this exists
---------------
``input_readers._body_data_to_24x7()`` **discards** the per-joint ``is_valid``
flag: invalid joints are skipped, leaving a zero position *and a zero
quaternion* in the output array. That means downstream code cannot tell a
measured joint from an unmeasured one.

That matters for Quest support. The Quest reports body pose via Meta IOBT,
which does not camera-track the legs -- lower body is inferred. The CloudXR
client SDK then squeezes that skeleton into the ByteDance 24-joint layout
before it reaches this machine. If the SDK marks synthesised joints as valid,
every consumer treats fabricated data as measured.

This tool records the raw joints *with* validity preserved so Pico and Quest
can be compared directly.

Usage
-----
Record (run once per headset, same pose/motion both times)::

    python3 gear_sonic/scripts/capture_body_skeleton.py --label pico  --duration 20
    python3 gear_sonic/scripts/capture_body_skeleton.py --label quest --duration 20

Compare two recordings::

    python3 gear_sonic/scripts/capture_body_skeleton.py --compare \
        /tmp/skeleton_pico.npz /tmp/skeleton_quest.npz

Interpreting the report
-----------------------
``valid%``    fraction of samples where the runtime marked the joint valid.
``pos_sd``    std-dev of joint position (mm) about its own mean, summed over
              XYZ. A joint that never moves relative to the body is suspect.
``rot_sd``    std-dev of orientation (degrees) about the mean rotation.
``d_pelvis``  correlation of this joint's motion with the pelvis. A value very
              close to 1.0 means the joint is rigidly following the root rather
              than being independently measured -- the signature of a
              synthesised joint.

A joint that is reported 100% valid but has near-zero ``rot_sd`` and
``d_pelvis`` ~ 1.0 is almost certainly fabricated.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Repo root on sys.path so `gear_sonic.*` imports work when run directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

NUM_JOINTS = 24

# XR_BD_body_tracking joint names, index-aligned with BodyJointPico.
JOINT_NAMES = [
    "PELVIS", "LEFT_HIP", "RIGHT_HIP", "SPINE1",
    "LEFT_KNEE", "RIGHT_KNEE", "SPINE2", "LEFT_ANKLE",
    "RIGHT_ANKLE", "SPINE3", "LEFT_FOOT", "RIGHT_FOOT",
    "NECK", "LEFT_COLLAR", "RIGHT_COLLAR", "HEAD",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HAND", "RIGHT_HAND",
]

# Joints Meta IOBT does not camera-track; inferred from head/hand motion.
# See IsaacTeleop docs/source/device/body_tracking.rst:221-225.
SUSPECT_ON_QUEST = {1, 2, 4, 5, 7, 8, 10, 11}


def _extract_joints(body_data):
    """Return ``(pos[24,3], quat[24,4] xyzw, valid[24] bool)`` or None.

    Mirrors the schema handling in ``input_readers._body_data_to_24x7`` but
    keeps ``is_valid`` instead of collapsing it into zeros.
    """
    if body_data is None:
        return None

    pos = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
    quat = np.zeros((NUM_JOINTS, 4), dtype=np.float64)
    valid = np.zeros(NUM_JOINTS, dtype=bool)

    # Schema B: msgpack wire format (teleop_ros2_ref-compatible).
    positions = getattr(body_data, "joint_positions", None)
    orientations = getattr(body_data, "joint_orientations", None)
    if positions is None and isinstance(body_data, dict):
        positions = body_data.get("joint_positions")
        orientations = body_data.get("joint_orientations")
    if positions is not None and orientations is not None:
        n = min(len(positions), len(orientations), NUM_JOINTS)
        for i in range(n):
            p, o = positions[i], orientations[i]
            pos[i] = [float(p[0]), float(p[1]), float(p[2])]
            quat[i] = [float(o[0]), float(o[1]), float(o[2]), float(o[3])]
            valid[i] = True  # schema B carries no per-joint validity
        return (pos, quat, valid) if n else None

    # Schema A: native FullBodyPosePicoT.
    joints_container = getattr(body_data, "joints", None)
    get_joint = getattr(joints_container, "joints", None) if joints_container else None
    if not callable(get_joint):
        return None

    any_present = False
    for i in range(NUM_JOINTS):
        try:
            joint = get_joint(i)
        except Exception:
            continue
        if joint is None:
            continue
        any_present = True
        valid[i] = bool(getattr(joint, "is_valid", True))
        pose = getattr(joint, "pose", None)
        if pose is None:
            continue
        p = getattr(pose, "position", None)
        o = getattr(pose, "orientation", None)
        if p is not None:
            pos[i] = [float(p.x), float(p.y), float(p.z)]
        if o is not None:
            quat[i] = [float(o.x), float(o.y), float(o.z), float(o.w)]

    return (pos, quat, valid) if any_present else None


def _quat_angle_deg(q, ref):
    """Angle in degrees between quaternion ``q`` and reference ``ref`` (xyzw)."""
    nq = np.linalg.norm(q, axis=-1, keepdims=True)
    nr = np.linalg.norm(ref)
    if nr < 1e-9:
        return np.zeros(len(q))
    q = q / np.where(nq < 1e-9, 1.0, nq)
    ref = ref / nr
    dot = np.abs(np.clip(q @ ref, -1.0, 1.0))
    return np.degrees(2.0 * np.arccos(dot))


def record(args):
    from gear_sonic.utils.teleop.isaac_teleop_client import IsaacTeleopClient

    client = IsaacTeleopClient()
    print("Starting IsaacTeleop streaming (connect the headset if not already)...")
    client.start_streaming()

    print(f"Waiting for first valid body sample (label={args.label!r})...")
    t_wait = time.monotonic()
    while True:
        got = _extract_joints(client.get_full_body_data())
        if got is not None and got[2].any():
            break
        if time.monotonic() - t_wait > args.timeout:
            print(f"ERROR: no body data after {args.timeout}s. Is body tracking active?")
            return 1
        time.sleep(0.2)

    print(f"Recording {args.duration}s at ~{args.poll_hz} Hz. Move naturally "
          "(same motion on both headsets).")
    P, Q, V, T = [], [], [], []
    period = 1.0 / args.poll_hz
    t_end = time.monotonic() + args.duration
    last_print = 0.0
    while time.monotonic() < t_end:
        got = _extract_joints(client.get_full_body_data())
        if got is not None:
            p, q, v = got
            P.append(p); Q.append(q); V.append(v); T.append(time.monotonic())
            now = time.monotonic()
            if now - last_print >= 2.0:
                last_print = now
                print(f"  {len(P):5d} samples   valid joints this frame: {v.sum()}/24"
                      f"   {t_end - now:4.1f}s left")
        time.sleep(period)

    if not P:
        print("ERROR: no samples captured.")
        return 1

    P = np.array(P); Q = np.array(Q); V = np.array(V); T = np.array(T)
    out = args.out or f"/tmp/skeleton_{args.label}.npz"
    np.savez_compressed(out, pos=P, quat=Q, valid=V, t=T, label=args.label)
    print(f"\nSaved {len(P)} samples -> {out}")
    report(P, Q, V, args.label)
    client.close()
    return 0


def report(P, Q, V, label):
    print(f"\n{'=' * 78}\n  PER-JOINT REPORT — {label}   ({len(P)} samples)\n{'=' * 78}")
    print(f"{'idx':>3} {'joint':<16} {'valid%':>7} {'pos_sd(mm)':>11} "
          f"{'rot_sd(deg)':>12} {'d_pelvis':>9}")
    print("-" * 78)

    pelvis = P[:, 0, :]
    pelvis_d = np.linalg.norm(np.diff(pelvis, axis=0), axis=-1)

    flags = []
    for j in range(NUM_JOINTS):
        vpct = 100.0 * V[:, j].mean()
        # position spread relative to pelvis (cancels whole-body translation)
        rel = P[:, j, :] - pelvis
        pos_sd = np.linalg.norm(rel.std(axis=0)) * 1000.0
        mean_q = Q[:, j, :].mean(axis=0)
        rot_sd = float(np.std(_quat_angle_deg(Q[:, j, :], mean_q)))
        jd = np.linalg.norm(np.diff(P[:, j, :], axis=0), axis=-1)
        if jd.std() > 1e-9 and pelvis_d.std() > 1e-9:
            corr = float(np.corrcoef(jd, pelvis_d)[0, 1])
        else:
            corr = float("nan")
        print(f"{j:>3} {JOINT_NAMES[j]:<16} {vpct:>6.1f}% {pos_sd:>11.1f} "
              f"{rot_sd:>12.2f} {corr:>9.3f}")
        if vpct > 99.0 and rot_sd < 0.5:
            flags.append((j, "valid but orientation never changes"))
        elif vpct > 99.0 and pos_sd < 5.0 and j != 0:
            flags.append((j, "valid but no motion relative to pelvis"))

    print("-" * 78)
    print(f"joints always valid : {(V.mean(axis=0) > 0.99).sum()}/24")
    print(f"joints never valid  : {(V.mean(axis=0) < 0.01).sum()}/24")
    if flags:
        print("\nSUSPECT (reported valid but showing no independent signal):")
        for j, why in flags:
            tag = "  <-- Meta IOBT does not track this" if j in SUSPECT_ON_QUEST else ""
            print(f"  [{j:2d}] {JOINT_NAMES[j]:<16} {why}{tag}")
    else:
        print("\nNo joints flagged as static-but-valid.")


def compare(path_a, path_b):
    a, b = np.load(path_a, allow_pickle=True), np.load(path_b, allow_pickle=True)
    la = str(a["label"]) if "label" in a else os.path.basename(path_a)
    lb = str(b["label"]) if "label" in b else os.path.basename(path_b)

    for tag, d in ((la, a), (lb, b)):
        report(d["pos"], d["quat"], d["valid"], tag)

    print(f"\n{'=' * 78}\n  SIDE-BY-SIDE — {la} vs {lb}\n{'=' * 78}")
    print(f"{'idx':>3} {'joint':<16} {'valid% A':>9} {'valid% B':>9} "
          f"{'rot_sd A':>9} {'rot_sd B':>9} {'ratio':>7}")
    print("-" * 78)
    for j in range(NUM_JOINTS):
        va = 100.0 * a["valid"][:, j].mean()
        vb = 100.0 * b["valid"][:, j].mean()
        ra = float(np.std(_quat_angle_deg(a["quat"][:, j, :], a["quat"][:, j, :].mean(axis=0))))
        rb = float(np.std(_quat_angle_deg(b["quat"][:, j, :], b["quat"][:, j, :].mean(axis=0))))
        ratio = (rb / ra) if ra > 1e-6 else float("nan")
        mark = "  <--" if (ra > 1.0 and rb < 0.2 * ra) else ""
        print(f"{j:>3} {JOINT_NAMES[j]:<16} {va:>8.1f}% {vb:>8.1f}% "
              f"{ra:>9.2f} {rb:>9.2f} {ratio:>7.2f}{mark}")
    print("-" * 78)
    print("'<--' marks joints where B has far less orientation variation than A,")
    print("i.e. B is likely synthesising a joint that A measures.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="unknown", help="headset label, e.g. pico / quest")
    ap.add_argument("--duration", type=float, default=20.0, help="record seconds")
    ap.add_argument("--poll-hz", type=float, default=60.0)
    ap.add_argument("--timeout", type=float, default=60.0, help="wait for first sample")
    ap.add_argument("--out", default=None, help="output .npz (default /tmp/skeleton_<label>.npz)")
    ap.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"),
                    help="compare two recordings instead of recording")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return 0
    return record(args)


if __name__ == "__main__":
    sys.exit(main())
