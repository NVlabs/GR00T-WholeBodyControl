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
``rot_spr``   mean angular distance (degrees) of orientation about the mean
              rotation. Not a std-dev: the distance is non-negative, so its
              std-dev reads 0 for a joint oscillating steadily about its mean
              and would call that joint frozen.
``d_pelvis``  correlation of this joint's motion with the pelvis. A value very
              close to 1.0 means the joint is rigidly following the root rather
              than being independently measured -- the signature of a
              synthesised joint.

A joint that is reported 100% valid but has near-zero ``rot_spr`` and
``d_pelvis`` ~ 1.0 is almost certainly fabricated.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

if TYPE_CHECKING:  # import-time cost avoided; these names are annotations only
    # The native body-tracking payload, as isaacteleop itself annotates it (see
    # retargeting_engine/deviceio_source_nodes/full_body_source.py, which types
    # FullBodyPoseTrackedT.data as "FullBodyPoseT | None").
    from isaacteleop.schema import FullBodyPoseT
    from scipy.spatial.transform import Rotation

    from gear_sonic.utils.teleop.isaac_teleop_client import IsaacTeleopClient

# Repo root on sys.path so `gear_sonic.*` imports work when run directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from gear_sonic.utils.teleop.body_joints import (  # noqa: E402
    INFERRED_ON_QUEST,
    NUM_BODY_JOINTS,
    AXIAL_JOINTS,
    BodyJoint,
    LEFT_LIMB_JOINTS,
    RIGHT_LIMB_JOINTS,
)

NUM_JOINTS = NUM_BODY_JOINTS


class JointSample(NamedTuple):
    """One frame of body tracking, as this script needs it.

    A NamedTuple rather than a dataclass so existing positional use keeps
    working -- ``p, q, v = sample`` and ``sample[2]`` are unchanged -- while the
    fields acquire names. ``sample[2].any()`` said nothing about what index 2
    was; ``sample.valid.any()`` does.
    """

    #: ``(24, 3)`` joint positions, metres.
    positions: np.ndarray
    #: ``(24, 4)`` global joint orientations, xyzw.
    orientations: np.ndarray
    #: ``(24,)`` per-joint validity, as reported by the tracker.
    valid: np.ndarray


def _extract_joints(
    body_data: "FullBodyPoseT | Mapping[str, Any] | None",
) -> Optional[JointSample]:
    """Return a :class:`JointSample`, or None when there is no body data.

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
        return JointSample(pos, quat, valid) if n else None

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

    return JointSample(pos, quat, valid) if any_present else None


def _quat_angle_deg(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Angle in degrees between quaternion ``q`` and reference ``ref`` (xyzw)."""
    nq = np.linalg.norm(q, axis=-1, keepdims=True)
    nr = np.linalg.norm(ref)
    if nr < 1e-9:
        return np.zeros(len(q))
    q = q / np.where(nq < 1e-9, 1.0, nq)
    ref = ref / nr
    dot = np.abs(np.clip(q @ ref, -1.0, 1.0))
    return np.degrees(2.0 * np.arccos(dot))


def _rot_spread_deg(quat: np.ndarray) -> float:
    """Mean angular distance (degrees) of ``quat`` ``(N, 4)`` xyzw about its own
    mean rotation.

    The *mean* distance, not its standard deviation, which is what this used to
    measure and which is blind to exactly the motion it exists to catch.
    ``_quat_angle_deg`` returns a distance, so a joint oscillating a steady
    +/-10 degrees about its mean gives 10 for every sample: the std is 0 and the
    joint is reported frozen, while it is in fact moving 20 degrees peak to
    peak. The mean distance reports the 10 degrees that is actually there.

    (The same correction was made to ``skeleton_guards.signature()`` in 4b9932a;
    this is the other half of it.)

    The reference is hemisphere-aligned before averaging, for the reason spelled
    out in :func:`_mean_rot`: a componentwise mean over quaternions straddling
    w = 0 collapses toward zero and the reference stops meaning anything.

    Returns NaN when no mean rotation exists (samples spread over the whole
    sphere), which the callers' thresholds read as "not a held pose".
    """
    quat = np.asarray(quat, dtype=np.float64)
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    unit = quat / np.where(norms < 1e-9, 1.0, norms)
    aligned = unit * np.where(unit @ unit[0] < 0.0, -1.0, 1.0)[:, None]
    mean_q = aligned.mean(axis=0)
    n = np.linalg.norm(mean_q)
    if n < 1e-9:
        return float("nan")
    return float(np.mean(_quat_angle_deg(aligned, mean_q / n)))


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
@dataclass(frozen=True)
class PoseSpec:
    """One pose in a capture battery.

    Frozen, and named, because these are read far from where they are written:
    the batteries are long literals and an entry used to be an anonymous
    3-tuple, so a reader at the capture loop had to come back here to learn
    what the third element meant.
    """

    #: Short identifier; becomes the capture's filename and label.
    name: str
    #: What the operator is told to do, read aloud during capture.
    instruction: str
    #: Why the pose is in the battery -- which joints it exercises, and what it
    #: separates from what. Not shown to the operator; it is the record of the
    #: design, kept beside the pose rather than in a comment that drifts.
    rationale: str


POSE_BATTERY = [
    PoseSpec("tpose",
             "Arms straight out to the sides at shoulder height, palms DOWN.",
             "baseline; matches the existing correction table"),
    PoseSpec("arms_forward",
             "Arms straight FORWARD at shoulder height, palms DOWN.",
             "shoulder flexion 90 deg, elbows straight"),
    PoseSpec("arms_forward_palms_up",
             "Same as before -- arms straight forward -- but palms UP.",
             "isolates wrist/forearm roll: only the wrists change from the previous pose"),
    PoseSpec("goalpost",
             "Upper arms out to the sides horizontal, forearms straight UP, palms forward.",
             "elbow 90 deg with shoulder abducted; separates shoulder from elbow"),
    PoseSpec("arms_overhead",
             "Arms straight UP overhead, palms facing each other.",
             "shoulder flexion ~180 deg, the far end of shoulder range"),
    PoseSpec("bend_forward",
             "Hinge at the hips ~45 deg, back flat, arms hanging straight down.",
             "spine flexion + hip flexion; never exercised before"),
    PoseSpec("lean_right",
             "Stand straight, lean sideways to your RIGHT, arms relaxed at sides.",
             "lateral spine bend; breaks the left/right symmetry of the other poses"),
    PoseSpec("sitting",
             "Sit on a chair, feet flat on the floor, hands resting on knees.",
             "hip and knee flexion ~90 deg -- the only pose that bends the legs"),
]


# ---------------------------------------------------------------------------
# Shoulder battery.
#
# Fitting the main battery left LEFT_SHOULDER and RIGHT_SHOULDER pose-dependent
# (leave-one-out residual 43 and 73 deg) while every other joint corrected well.
# Two candidate explanations were tested against the existing data and both
# failed: the collar+shoulder composite is no more stable than the shoulder
# alone, and the offset does not track arm elevation. So the shoulder offset is
# neither a redistribution artefact nor a simple function of arm pose.
#
# That leaves the question this battery answers: is the shoulder offset
# REPEATABLE? With one capture per pose we cannot tell a structured
# pose-dependent offset (modellable with a richer fit) from an unrepeatable one
# (nothing will fix it; derive arm swing from positions instead).
#
# Poses 1-3 are exact repeats of main-battery poses, captured in a fresh
# session. Comparing the offset for the same pose across two sessions measures
# repeatability directly.
#
# Poses 4-6 hold the upper arm in one place and vary ONLY shoulder axial roll.
# Elevation and azimuth are held fixed, so any offset change between them is
# attributable to roll -- the component positions constrain least and the most
# likely place for the SDK's conversion to go wrong.
# ---------------------------------------------------------------------------
POSE_BATTERY_SHOULDER = [
    PoseSpec("tpose_r2",
             "Arms straight out to the sides at shoulder height, palms DOWN.",
             "exact repeat of main battery pose 1 -- tests repeatability"),
    PoseSpec("arms_forward_r2",
             "Arms straight FORWARD at shoulder height, palms DOWN.",
             "exact repeat of main battery pose 2 -- tests repeatability"),
    PoseSpec("goalpost_r2",
             "Upper arms out to the sides horizontal, forearms straight UP, palms forward.",
             "exact repeat of main battery pose 4 -- tests repeatability"),
    PoseSpec("elbow90_forward",
             "Upper arms hanging at your SIDES, elbows bent 90 deg, forearms pointing "
     "straight FORWARD, palms facing each other.",
             "shoulder roll neutral; upper arm vertical"),
    PoseSpec("elbow90_inward",
             "Same stance -- upper arms at sides, elbows bent 90 -- but rotate the "
     "forearms INWARD across your belly until the hands nearly touch.",
             "same upper-arm placement, shoulder rolled inward (internal rotation)"),
    PoseSpec("elbow90_outward",
             "Same stance again, but rotate the forearms OUTWARD as far as is "
     "comfortable, opening the arms wide.",
             "same upper-arm placement, shoulder rolled outward (external rotation)"),
]

# ---------------------------------------------------------------------------
# Torso + legs battery.
#
# Correcting a joint properly needs a two-sided fit, local_quest = A * local_pico
# * B: A is the parent-frame rotation and decides which axis a motion comes out
# about, B is the child-frame relabelling. Separating them requires the joint to
# rotate about at least TWO independent axes across the captures. A single axis
# leaves A and B degenerate -- the fit still matches every static pose while
# reproducing motion about the wrong axis.
#
# Measured axis diversity from the first two batteries (0 = all motion about one
# axis, 1 = fully independent): SPINE1 0.08, SPINE2 0.15, SPINE3 0.13, PELVIS
# 0.61 over only 21 deg, hips/ankles/feet ~0. The arms scored 0.5-0.95, which is
# why they solve and the torso does not.
#
# Symptoms that traces to: operator waist flexion arriving as robot lateral bend
# (spine axis mapping off by ~86 deg), and a persistent leg crouch.
#
# So this battery deliberately rotates the torso about all three axes, turns the
# whole body to several headings (which also conditions the pelvis, the one
# thing every previous battery held fixed by instruction), and puts the legs
# through real range.
#
# NOTE: unlike the earlier batteries, here you SHOULD change facing -- poses 9-11
# exist precisely to sample different body headings.
# ---------------------------------------------------------------------------
POSE_BATTERY_TORSO = [
    PoseSpec("neutral",
             "Stand upright and relaxed, arms at your sides, facing forward.",
             "reference for every other pose in this battery"),
    PoseSpec("bend_fwd_deep",
             "Hinge forward at the waist as far as is comfortable, arms hanging.",
             "spine flexion, large"),
    PoseSpec("lean_back",
             "Lean BACKWARD from the waist, shoulders behind your hips.",
             "spine extension -- the opposite sign of flexion, never captured before"),
    PoseSpec("lean_left_deep",
             "Bend sideways to your LEFT, right arm reaching down your side.",
             "lateral spine bend, left"),
    PoseSpec("lean_right_deep",
             "Bend sideways to your RIGHT, left arm reaching down your side.",
             "lateral bend right; with the previous pose this gives the second spine axis"),
    PoseSpec("twist_left",
             "Feet planted facing forward, twist your SHOULDERS to the left.",
             "spine axial twist -- the third spine axis, and never captured before"),
    PoseSpec("twist_right",
             "Feet planted facing forward, twist your SHOULDERS to the right.",
             "spine axial twist, opposite sign"),
    PoseSpec("squat",
             "Squat down, knees bent, back reasonably straight, arms forward for balance.",
             "hip and knee flexion together, deeper than sitting"),
    PoseSpec("face_left_90",
             "Turn your whole body 90 degrees to the LEFT and stand neutral.",
             "body heading #2 -- conditions the pelvis, which every earlier battery held fixed"),
    PoseSpec("face_right_90",
             "Turn your whole body 90 degrees to the RIGHT and stand neutral.",
             "body heading #3"),
    PoseSpec("face_back_180",
             "Turn to face AWAY from your starting direction and stand neutral.",
             "body heading #4; four headings pin the pelvis world frame"),
    PoseSpec("stride",
             "Stand with one foot a full stride forward, weight even, upright.",
             "hip flexion/extension with opposite signs left vs right"),
]

# ---------------------------------------------------------------------------
# Wrist battery.
#
# None of the other batteries flexes the wrist. Across all 26 poses captured so
# far, the commanded G1 wrist pitch spans only 17-27 deg on an axis with roughly
# 180 deg of travel -- every pose happens to hold the wrist near neutral, and the
# one wrist-specific pair (palms down / palms up) exercises ROLL, not flexion.
#
# That is enough to measure a bias, since the rest pose is captured, but not a
# gain: fitting a scale factor from data that never leaves the middle of the
# range would be fitting noise. These poses go to the endpoints deliberately.
#
# The forearm is held fixed throughout so the wrist is the only thing moving --
# otherwise elbow motion leaks into the measurement, and the retargeting already
# folds elbow swing into the wrist channels.
# ---------------------------------------------------------------------------
POSE_BATTERY_WRIST = [
    PoseSpec("wrist_neutral",
             "Upper arms at your sides, elbows bent 90, forearms pointing FORWARD, "
     "palms facing each other, wrists straight and relaxed.",
             "the rest pose an operator expects to be neutral; anchors the bias"),
    PoseSpec("wrist_flex_max",
             "Same stance. Bend BOTH wrists forward as far as they go -- palms curling "
     "toward the forearms. Keep the forearms still.",
             "flexion endpoint"),
    PoseSpec("wrist_extend_max",
             "Same stance. Bend BOTH wrists backward as far as they go -- backs of the "
     "hands toward the forearms. Keep the forearms still.",
             "extension endpoint; with the previous pose this gives the full span"),
    PoseSpec("wrist_flex_half",
             "Same stance, wrists bent forward about HALF of maximum.",
             "midpoint on the flexion side; tests whether the mapping is linear"),
    PoseSpec("wrist_extend_half",
             "Same stance, wrists bent backward about HALF of maximum.",
             "midpoint on the extension side"),
    PoseSpec("wrist_deviate",
             "Same stance, wrists straight but angled outward (thumbs away from each "
     "other) as far as is comfortable.",
             "radial/ulnar deviation -- separates the flexion axis from the other two"),
]

BATTERIES = {
    "main": POSE_BATTERY,
    "shoulder": POSE_BATTERY_SHOULDER,
    "torso": POSE_BATTERY_TORSO,
    "wrist": POSE_BATTERY_WRIST,
}


def _beep() -> None:
    """Terminal bell. Audible cue matters because the operator is wearing a
    headset and cannot read the terminal."""
    sys.stdout.write("\a")
    sys.stdout.flush()


def _countdown(seconds: float) -> None:
    for k in range(int(seconds), 0, -1):
        print(f"    capturing in {k}...   ", end="\r", flush=True)
        _beep()
        time.sleep(1.0)


def _trigger_pulled(client: "IsaacTeleopClient", thresh: float = 0.7) -> bool:
    """True if either trigger is past ``thresh``. Never raises."""
    try:
        return (client.get_key_value_by_name("right_trigger") > thresh
                or client.get_key_value_by_name("left_trigger") > thresh)
    except Exception:
        return False


def _triggers_released(client: "IsaacTeleopClient", thresh: float = 0.3) -> bool:
    try:
        return (client.get_key_value_by_name("right_trigger") < thresh
                and client.get_key_value_by_name("left_trigger") < thresh)
    except Exception:
        return True


def _stdin_ready() -> bool:
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


def _wait_for_go(client: "IsaacTeleopClient", args: argparse.Namespace) -> bool:
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
    #
    # This used to give up after 5 seconds and fall through to the pull loop
    # below, where a still-held trigger reads as an immediate pull -- so holding
    # the trigger advanced every pose in the battery without a single deliberate
    # press, which is the exact thing the release check exists to prevent. Wait
    # for the release instead of timing out past it, and say so once, since a
    # silent wait looks like a hang. ENTER stays available as the way out.
    t0 = time.monotonic()
    nagged = False
    while not _triggers_released(client):
        if _stdin_ready():
            sys.stdin.readline()
            return True
        if not nagged and time.monotonic() - t0 > 5.0:
            nagged = True
            print("    trigger still held -- RELEASE it, then pull again (or press ENTER).")
        time.sleep(0.05)

    while True:
        if _trigger_pulled(client):
            print("    trigger detected.                    ")
            return True
        if _stdin_ready():
            sys.stdin.readline()
            return True
        time.sleep(0.05)


def batch(args: argparse.Namespace) -> int:
    """Walk through POSE_BATTERY in one session, saving one file per pose."""
    from gear_sonic.utils.teleop.isaac_teleop_client import IsaacTeleopClient

    dev = args.batch
    poses = BATTERIES[args.battery]
    print("=" * 72)
    print(f"  POSE BATTERY {args.battery!r} — device label {dev!r}   ({len(poses)} poses)")
    print("=" * 72)
    print("Ground rules (these matter more than the poses themselves):")
    if args.battery == "torso":
        print("  * This battery INTENTIONALLY changes facing -- poses 9-11 turn the body")
        print("    on purpose, to condition the pelvis. Return to the same spot each time.")
    else:
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
        if got is not None and got.valid.any():
            break
        if time.monotonic() - t0 > args.timeout:
            print(f"ERROR: no body data after {args.timeout}s. Is body tracking active?")
            return 1
        time.sleep(0.2)
    print("Body tracking live.\n")

    written = []
    for n, pose in enumerate(poses, start=1):
        name, how, why = pose.name, pose.instruction, pose.rationale
        print("-" * 72)
        print(f"  POSE {n}/{len(poses)}: {name}")
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
        sd = np.array([_rot_spread_deg(Q[:, j, :]) for j in range(NUM_JOINTS)])
        # np.all, not sd.max(), for the NaN reason given in rest_pose().
        verdict = ("OK" if np.all(sd < 5.0)
                   else "TOO MUCH MOTION — consider redoing this pose")
        print(f"    saved {len(P)} samples -> {out}")
        print(f"    staticness: median {np.nanmedian(sd):.2f} deg, "
              f"max {np.nanmax(sd):.2f} deg   {verdict}\n")
        written.append(out)

    client.close()
    print("=" * 72)
    print(f"  Done — {len(written)}/{len(poses)} poses captured for {dev!r}")
    for w in written:
        print(f"    {w}")
    print("\n  Now repeat this identically on the other headset, e.g.:")
    other = "quest" if dev == "pico" else "pico"
    print(f"    --batch {other} --battery {args.battery}")
    print("=" * 72)
    return 0


def record(args: argparse.Namespace) -> int:
    from gear_sonic.utils.teleop.isaac_teleop_client import IsaacTeleopClient

    client = IsaacTeleopClient()
    print("Starting IsaacTeleop streaming (connect the headset if not already)...")
    client.start_streaming()

    print(f"Waiting for first valid body sample (label={args.label!r})...")
    t_wait = time.monotonic()
    while True:
        got = _extract_joints(client.get_full_body_data())
        if got is not None and got.valid.any():
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


def report(P: np.ndarray, Q: np.ndarray, V: np.ndarray, label: str) -> None:
    """Print the per-joint report for one recording.

    ``P`` ``(N, 24, 3)`` positions, ``Q`` ``(N, 24, 4)`` orientations xyzw and
    ``V`` ``(N, 24)`` validity are the three arrays of a recording, stacked
    over its ``N`` samples -- i.e. a run of :class:`JointSample` fields.
    """
    print(f"\n{'=' * 78}\n  PER-JOINT REPORT — {label}   ({len(P)} samples)\n{'=' * 78}")
    print(f"{'idx':>3} {'joint':<16} {'valid%':>7} {'pos_sd(mm)':>11} "
          f"{'rot_spr(deg)':>12} {'d_pelvis':>9}")
    print("-" * 78)

    pelvis = P[:, 0, :]
    pelvis_d = np.linalg.norm(np.diff(pelvis, axis=0), axis=-1)

    flags = []
    for j in range(NUM_JOINTS):
        vpct = 100.0 * V[:, j].mean()
        # position spread relative to pelvis (cancels whole-body translation)
        rel = P[:, j, :] - pelvis
        pos_sd = np.linalg.norm(rel.std(axis=0)) * 1000.0
        rot_spread = _rot_spread_deg(Q[:, j, :])
        jd = np.linalg.norm(np.diff(P[:, j, :], axis=0), axis=-1)
        if jd.std() > 1e-9 and pelvis_d.std() > 1e-9:
            corr = float(np.corrcoef(jd, pelvis_d)[0, 1])
        else:
            corr = float("nan")
        print(f"{j:>3} {BodyJoint(j).name:<16} {vpct:>6.1f}% {pos_sd:>11.1f} "
              f"{rot_spread:>12.2f} {corr:>9.3f}")
        if vpct > 99.0 and rot_spread < 0.5:
            flags.append((j, "valid but orientation never changes"))
        elif vpct > 99.0 and pos_sd < 5.0 and j != 0:
            flags.append((j, "valid but no motion relative to pelvis"))

    print("-" * 78)
    print(f"joints always valid : {(V.mean(axis=0) > 0.99).sum()}/24")
    print(f"joints never valid  : {(V.mean(axis=0) < 0.01).sum()}/24")
    if flags:
        print("\nSUSPECT (reported valid but showing no independent signal):")
        for j, why in flags:
            tag = "  <-- Meta IOBT does not track this" if j in INFERRED_ON_QUEST else ""
            print(f"  [{j:2d}] {BodyJoint(j).name:<16} {why}{tag}")
    else:
        print("\nNo joints flagged as static-but-valid.")


def compare(path_a: str, path_b: str) -> None:
    a, b = np.load(path_a, allow_pickle=True), np.load(path_b, allow_pickle=True)
    la = str(a["label"]) if "label" in a else os.path.basename(path_a)
    lb = str(b["label"]) if "label" in b else os.path.basename(path_b)

    for tag, d in ((la, a), (lb, b)):
        report(d["pos"], d["quat"], d["valid"], tag)

    print(f"\n{'=' * 78}\n  SIDE-BY-SIDE — {la} vs {lb}\n{'=' * 78}")
    print(f"{'idx':>3} {'joint':<16} {'valid% A':>9} {'valid% B':>9} "
          f"{'rot_spr A':>9} {'rot_spr B':>9} {'ratio':>7}")
    print("-" * 78)
    for j in range(NUM_JOINTS):
        va = 100.0 * a["valid"][:, j].mean()
        vb = 100.0 * b["valid"][:, j].mean()
        ra = _rot_spread_deg(a["quat"][:, j, :])
        rb = _rot_spread_deg(b["quat"][:, j, :])
        ratio = (rb / ra) if ra > 1e-6 else float("nan")
        mark = "  <--" if (ra > 1.0 and rb < 0.2 * ra) else ""
        print(f"{j:>3} {BodyJoint(j).name:<16} {va:>8.1f}% {vb:>8.1f}% "
              f"{ra:>9.2f} {rb:>9.2f} {ratio:>7.2f}{mark}")
    print("-" * 78)
    print("'<--' marks joints where B has far less orientation variation than A,")
    print("i.e. B is likely synthesising a joint that A measures.")


def main() -> int:
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
    ap.add_argument("--battery", default="main", choices=sorted(BATTERIES),
                    help="which pose battery to run with --batch (default: main)")
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

AXIAL = AXIAL_JOINTS
LEFT_LIMB = LEFT_LIMB_JOINTS
RIGHT_LIMB = RIGHT_LIMB_JOINTS


def _mean_rot(q: np.ndarray) -> "Rotation":
    """Mean of unit quaternions in ``q`` ``(N, 4)`` xyzw, as a ``Rotation``.

    A quaternion and its negation are the same rotation, so the samples must be
    brought into one hemisphere before a componentwise mean means anything.

    Aligning on the sign of w is the obvious way to do that and it breaks at
    exactly the rotation this is most likely to meet in a rest-pose comparison.
    w is cos(theta/2), so it passes through 0 at a 180 degree rotation: samples
    a hair either side of 180 degrees land in opposite hemispheres and average
    toward zero, and at exactly 180 the sign is 0, which would scale the whole
    quaternion away. (The previous version added 1e-12 before taking the sign
    purely to dodge that second case, which left the first one unhandled.)

    Aligning against the first sample instead has no such special point: for any
    tight cluster the dot product with the reference stays near +/-1, nowhere
    near the flip.
    """
    from scipy.spatial.transform import Rotation as R
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = q * np.where(q @ q[0] < 0.0, -1.0, 1.0)[:, None]
    m = q.mean(axis=0)
    n = np.linalg.norm(m)
    if n < 1e-9:
        # Samples spread over the whole sphere; no mean rotation exists. Only
        # reachable on a non-static capture, which rest_pose() checks for first.
        raise ValueError(
            "mean orientation is undefined: samples are antipodal, so this "
            "capture is not a held pose"
        )
    return R.from_quat(m / n)


def _fit_group(offsets: Mapping[int, "Rotation"], idxs: Sequence[int]) -> "Rotation":
    from scipy.spatial.transform import Rotation as R
    M = np.mean([offsets[j].as_matrix() for j in idxs], axis=0)
    U, _, Vt = np.linalg.svd(M)
    # U @ Vt is the nearest *orthogonal* matrix to M, which is not necessarily a
    # rotation: when det is -1 it is a reflection, and a reflection is not a
    # pose any body can hold. Flipping the sign of the least significant
    # singular direction gives the nearest matrix with det +1 -- the Kabsch
    # correction. Without it a group of offsets that averages toward a
    # degenerate mean can yield a mirrored "fit" that then reports small
    # residuals, i.e. a confident answer that is not a rotation at all.
    d = 1.0 if np.linalg.det(U @ Vt) >= 0.0 else -1.0
    return R.from_matrix(U @ np.diag([1.0, 1.0, d]) @ Vt)


def rest_pose(path_a: str, path_b: str) -> dict[int, "Rotation"]:
    a, b = np.load(path_a, allow_pickle=True), np.load(path_b, allow_pickle=True)
    la = str(a["label"]) if "label" in a else "A"
    lb = str(b["label"]) if "label" in b else "B"

    print("STATICNESS CHECK (this analysis is only valid for held poses)")
    ok = True
    for tag, d in ((la, a), (lb, b)):
        sd = np.array([_rot_spread_deg(d["quat"][:, j, :]) for j in range(NUM_JOINTS)])
        # Test "every joint is below the threshold" rather than comparing the
        # max, because _rot_spread_deg returns NaN for a joint whose samples
        # span the sphere, and every comparison against NaN is False: the old
        # `sd.max() < 5.0` read False and printed TOO MUCH MOTION, while
        # `sd.max() >= 5.0` also read False and left ok True, suppressing the
        # warning below. One condition now drives both, and NaN -- which is not
        # a held pose either -- fails it.
        static = bool(np.all(sd < 5.0))
        undefined = int(np.isnan(sd).sum())
        flag = "OK" if static else "TOO MUCH MOTION"
        if undefined:
            flag += f" ({undefined} joint(s) with no mean rotation)"
        if not static:
            ok = False
        # nan-aware, so one undefined joint does not blank out the figures for
        # the rest; the count above says how many were left out.
        print(f"  {tag:<14} median {np.nanmedian(sd):5.2f} deg  "
              f"max {np.nanmax(sd):5.2f} deg   {flag}")
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
        print(f"{j:>3} {BodyJoint(j).name:<16} {ang:>6.1f}  "
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
