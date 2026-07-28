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


# ---------------------------------------------------------------------------
# Batch pose battery.
#
# Every pose here is chosen to be REPRODUCIBLE minutes apart on a different
# headset -- that is the binding constraint, not coverage. An earlier round of
# this analysis was wasted because "arms down" is not a well-defined pose and
# the two captures did not match. Each entry therefore has an unambiguous
# endpoint (straight, 90 degrees, against a surface) rather than a vague one.
#
# Between them they place every joint group in at least two clearly different
# configurations, which is what separates a constant rest-pose offset from a
# pose-dependent one.
# ---------------------------------------------------------------------------
POSE_BATTERY = [
    ("tpose", "Arms straight out to the sides at shoulder height, palms DOWN.",
     "baseline; matches the existing correction table"),
    ("arms_forward", "Arms straight FORWARD at shoulder height, palms DOWN.",
     "shoulder flexion 90 deg, elbows straight"),
    ("arms_forward_palms_up", "Same as before -- arms straight forward -- but palms UP.",
     "isolates wrist/forearm roll: only the wrists change from the previous pose"),
    ("goalpost", "Upper arms out to the sides horizontal, forearms straight UP, palms forward.",
     "elbow 90 deg with shoulder abducted; separates shoulder from elbow"),
    ("arms_overhead", "Arms straight UP overhead, palms facing each other.",
     "shoulder flexion ~180 deg, the far end of shoulder range"),
    ("bend_forward", "Hinge at the hips ~45 deg, back flat, arms hanging straight down.",
     "spine flexion + hip flexion; never exercised before"),
    ("lean_right", "Stand straight, lean sideways to your RIGHT, arms relaxed at sides.",
     "lateral spine bend; breaks the left/right symmetry of the other poses"),
    ("sitting", "Sit on a chair, feet flat on the floor, hands resting on knees.",
     "hip and knee flexion ~90 deg -- the only pose that bends the legs"),
]


def _beep():
    """Terminal bell. Audible cue matters because the operator is wearing a
    headset and cannot read the terminal."""
    sys.stdout.write("\a")
    sys.stdout.flush()


def _countdown(seconds):
    for k in range(int(seconds), 0, -1):
        print(f"    capturing in {k}...   ", end="\r", flush=True)
        _beep()
        time.sleep(1.0)


def _trigger_pulled(client, thresh=0.7):
    """True if either trigger is past ``thresh``. Never raises."""
    try:
        return (client.get_key_value_by_name("right_trigger") > thresh
                or client.get_key_value_by_name("left_trigger") > thresh)
    except Exception:
        return False


def _triggers_released(client, thresh=0.3):
    try:
        return (client.get_key_value_by_name("right_trigger") < thresh
                and client.get_key_value_by_name("left_trigger") < thresh)
    except Exception:
        return True


def _stdin_ready():
    """Non-blocking check for a pending line on stdin (POSIX only).

    Requires a tty: a closed or redirected stdin selects as readable
    immediately at EOF, which would silently auto-advance every pose.
    """
    try:
        if not sys.stdin.isatty():
            return False
        import select
        return bool(select.select([sys.stdin], [], [], 0.0)[0])
    except Exception:
        return False


def _wait_for_go(client, args):
    """Block until the operator signals ready. Returns False to abort.

    Accepts whichever comes first: a controller trigger pull, or ENTER. The
    trigger path exists because the operator is in a headset holding
    controllers and cannot reach the keyboard while holding a pose.

    In --auto mode nothing is required: the run is fully hands-free and each
    pose simply gets ``--pose-delay`` seconds to assume.
    """
    if args.auto:
        print(f"  AUTO: {args.pose_delay:.0f}s to get into position...")
        for k in range(int(args.pose_delay), 0, -1):
            if k <= 5 or k % 5 == 0:
                print(f"    {k:2d}s to assume pose...   ", end="\r", flush=True)
                if k <= 3:
                    _beep()
            time.sleep(1.0)
        print(" " * 40, end="\r")
        return True

    print("  Get into position, then PULL EITHER TRIGGER (or press ENTER).")
    print("  Ctrl-C to stop.")
    _beep()

    # Require a release first so a trigger still held from the previous pose
    # does not instantly advance this one.
    t0 = time.monotonic()
    while not _triggers_released(client):
        if time.monotonic() - t0 > 5.0:
            break
        time.sleep(0.05)

    while True:
        if _trigger_pulled(client):
            print("    trigger detected.                    ")
            return True
        if _stdin_ready():
            sys.stdin.readline()
            return True
        time.sleep(0.05)


def batch(args):
    """Walk through POSE_BATTERY in one session, saving one file per pose."""
    from gear_sonic.utils.teleop.isaac_teleop_client import IsaacTeleopClient

    dev = args.batch
    print("=" * 72)
    print(f"  POSE BATTERY — device label {dev!r}   ({len(POSE_BATTERY)} poses)")
    print("=" * 72)
    print("Ground rules (these matter more than the poses themselves):")
    print("  * Face the SAME direction for every pose. Pick a spot on the wall and")
    print("    keep facing it. A body turn between captures corrupts the analysis.")
    print("  * Stand on the same spot. Mark the floor if you can.")
    print("  * HOLD STILL during each capture -- it warns you if you did not.")
    print("  * Run this identically on both headsets.")
    if args.auto:
        print(f"\n  AUTO MODE: hands-free. {args.pose_delay:.0f}s to assume each pose,")
        print("  then a 3-beep countdown and the capture starts. No input needed.\n")
    else:
        print("\n  To start each capture: PULL EITHER TRIGGER (or press ENTER).")
        print("  Beeps mark the countdown and the start of recording.\n")

    client = IsaacTeleopClient()
    print("Starting IsaacTeleop streaming (connect the headset if not already)...")
    client.start_streaming()
    print("Waiting for body tracking...")
    t0 = time.monotonic()
    while True:
        got = _extract_joints(client.get_full_body_data())
        if got is not None and got[2].any():
            break
        if time.monotonic() - t0 > args.timeout:
            print(f"ERROR: no body data after {args.timeout}s. Is body tracking active?")
            return 1
        time.sleep(0.2)
    print("Body tracking live.\n")

    written = []
    for n, (name, how, why) in enumerate(POSE_BATTERY, start=1):
        print("-" * 72)
        print(f"  POSE {n}/{len(POSE_BATTERY)}: {name}")
        print(f"  {how}")
        print(f"  (why: {why})")
        print("-" * 72)
        try:
            if not _wait_for_go(client, args):
                print("\nAborted by user.")
                break
        except (KeyboardInterrupt, EOFError):
            print("\nAborted by user.")
            break

        _countdown(args.lead_in)
        print(f"    HOLD STILL — recording {args.duration:.0f}s ...        ")
        _beep()

        P, Q, V, T = [], [], [], []
        period = 1.0 / args.poll_hz
        t_end = time.monotonic() + args.duration
        while time.monotonic() < t_end:
            got = _extract_joints(client.get_full_body_data())
            if got is not None:
                p, q, v = got
                P.append(p); Q.append(q); V.append(v); T.append(time.monotonic())
            time.sleep(period)

        if not P:
            print("    ERROR: no samples captured for this pose; skipping.")
            continue

        P, Q, V = np.array(P), np.array(Q), np.array(V)
        label = f"{dev}_{n:02d}_{name}"
        out = os.path.join(args.out_dir, f"skeleton_{label}.npz")
        np.savez_compressed(out, pos=P, quat=Q, valid=V, t=np.array(T), label=label)

        # immediate staticness feedback so a bad take can be redone on the spot
        sd = np.array([np.std(_quat_angle_deg(Q[:, j, :], Q[:, j, :].mean(axis=0)))
                       for j in range(NUM_JOINTS)])
        verdict = "OK" if sd.max() < 5.0 else "TOO MUCH MOTION — consider redoing this pose"
        print(f"    saved {len(P)} samples -> {out}")
        print(f"    staticness: median {np.median(sd):.2f} deg, max {sd.max():.2f} deg   {verdict}\n")
        written.append(out)

    client.close()
    print("=" * 72)
    print(f"  Done — {len(written)}/{len(POSE_BATTERY)} poses captured for {dev!r}")
    for w in written:
        print(f"    {w}")
    print("\n  Now repeat this identically on the other headset, e.g.:")
    other = "quest" if dev == "pico" else "pico"
    print(f"    --batch {other}")
    print("=" * 72)
    return 0


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
    ap.add_argument("--rest-pose", nargs=2, metavar=("A.npz", "B.npz"),
                    help="rest-pose offset analysis of two STATIC captures")
    ap.add_argument("--batch", metavar="DEVICE",
                    help="guided capture of the full pose battery, e.g. --batch quest")
    ap.add_argument("--out-dir", default="/tmp", help="output directory for --batch")
    ap.add_argument("--auto", action="store_true",
                    help="hands-free: no trigger or ENTER needed, each pose gets "
                         "--pose-delay seconds to assume")
    ap.add_argument("--pose-delay", type=float, default=15.0,
                    help="seconds to assume each pose in --auto mode (default 15)")
    ap.add_argument("--lead-in", type=float, default=3.0,
                    help="countdown seconds after the go signal (default 3)")
    args = ap.parse_args()

    if args.batch:
        return batch(args)
    if args.rest_pose:
        rest_pose(*args.rest_pose)
        return 0
    if args.compare:
        compare(*args.compare)
        return 0
    return record(args)



# ---------------------------------------------------------------------------
# Rest-pose offset analysis (--rest-pose A.npz B.npz)
#
# Valid ONLY for static captures. Verifies staticness first, then reports the
# per-joint mean-orientation offset of B relative to A, and tests whether a
# single rotation explains each anatomical group.
# ---------------------------------------------------------------------------

AXIAL = [0, 3, 6, 9, 12, 15]
LEFT_LIMB = [1, 4, 7, 10, 13, 16, 18, 20, 22]
RIGHT_LIMB = [2, 5, 8, 11, 14, 17, 19, 21, 23]


def _mean_rot(q):
    from scipy.spatial.transform import Rotation as R
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = q * np.sign(q[:, 3:4] + 1e-12)          # hemisphere-align before averaging
    m = q.mean(axis=0)
    return R.from_quat(m / np.linalg.norm(m))


def _fit_group(offsets, idxs):
    from scipy.spatial.transform import Rotation as R
    M = np.mean([offsets[j].as_matrix() for j in idxs], axis=0)
    U, _, Vt = np.linalg.svd(M)                  # nearest proper rotation
    return R.from_matrix(U @ Vt)


def rest_pose(path_a, path_b):
    a, b = np.load(path_a, allow_pickle=True), np.load(path_b, allow_pickle=True)
    la = str(a["label"]) if "label" in a else "A"
    lb = str(b["label"]) if "label" in b else "B"

    print("STATICNESS CHECK (this analysis is only valid for held poses)")
    ok = True
    for tag, d in ((la, a), (lb, b)):
        sd = np.array([np.std(_quat_angle_deg(d["quat"][:, j, :],
                                              d["quat"][:, j, :].mean(axis=0)))
                       for j in range(NUM_JOINTS)])
        flag = "OK" if sd.max() < 5.0 else "TOO MUCH MOTION"
        if sd.max() >= 5.0:
            ok = False
        print(f"  {tag:<14} median {np.median(sd):5.2f} deg  max {sd.max():5.2f} deg   {flag}")
    if not ok:
        print("\nWARNING: a capture is not static. Mean orientations reflect the motion\n"
              "performed, not the rest pose, and the offsets below are not meaningful.")
    print()

    offsets = {}
    print(f"PER-JOINT REST-POSE OFFSET ({lb} relative to {la})")
    print(f"{'idx':>3} {'joint':<16} {'angle':>7}  axis(x,y,z)")
    print("-" * 56)
    for j in range(NUM_JOINTS):
        off = _mean_rot(b["quat"][:, j, :]) * _mean_rot(a["quat"][:, j, :]).inv()
        offsets[j] = off
        rv = off.as_rotvec()
        ang = np.degrees(np.linalg.norm(rv))
        ax = rv / (np.linalg.norm(rv) + 1e-12)
        print(f"{j:>3} {JOINT_NAMES[j]:<16} {ang:>6.1f}  "
              f"({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f})")

    print("-" * 56)
    print("\nGROUP FIT — can ONE rotation explain each anatomical group?")
    for name, idxs in (("axial/spine", AXIAL), ("left limbs", LEFT_LIMB),
                       ("right limbs", RIGHT_LIMB)):
        Rg = _fit_group(offsets, idxs)
        res = [np.degrees(np.linalg.norm((Rg.inv() * offsets[j]).as_rotvec())) for j in idxs]
        verdict = "consistent" if np.median(res) < 15 else "NOT a shared rotation"
        print(f"  {name:<12} fitted {np.degrees(np.linalg.norm(Rg.as_rotvec())):6.1f} deg   "
              f"median residual {np.median(res):5.1f} deg   {verdict}")

    print("\nNOTE: a single static pose cannot distinguish a constant rest-pose offset\n"
          "from a pose-dependent error. Capture a SECOND distinct static pose and\n"
          "re-run: if the per-joint offsets match, they are constant and a fixed\n"
          "correction table will work. If they differ, the mapping is pose-dependent\n"
          "and no constant correction can fix it.")
    return offsets


if __name__ == "__main__":
    sys.exit(main())
