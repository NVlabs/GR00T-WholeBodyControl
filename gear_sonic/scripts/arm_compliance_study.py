#!/usr/bin/env python3
"""User-study session runner for the G1 arm compliance profiles.

Runs one participant through all study sets (e.g. the 8 sets in
gear_sonic_deploy/arm_compliance/study_8sets.json). For each set the robot is
switched to that stiffness profile, the participant performs each gesture
(handshake, fist bump, hug) and rates it right after. Answers are saved after
every trial, so a crash or Ctrl+C never loses data, and a session can be resumed.

It publishes the same ZMQ commands as arm_compliance_cli.py (port 5565, topic
"compliance"), so do NOT run both at the same time. Start deploy with the SAME
profile file:
    ./deploy.sh ... --arm-compliance --compliance-profiles arm_compliance/study_8sets.json sim|real

Run (from the repo root):
    python gear_sonic/scripts/arm_compliance_study.py \\
        --profiles gear_sonic_deploy/arm_compliance/study_8sets.json \\
        --control-mode teleop --practice --blind

At ANY prompt you can type a command (the ! keeps them apart from answers):
    !e  ESTOP (latched)            !r  release ESTOP -> current set's profile
    !b  break (robot -> RIGID)     !n  add a note to this trial
    !s  skip this trial            !q  save and quit (resume later with --resume ID)

Design choices (see ARM_COMPLIANCE.md, "User study"):
- Participants are identified by an ID (P01, P02, ...). The name is stored only
  in a separate key file (names_key.csv), never in the data files.
- Set order: balanced Latin square (Williams design) by participant number, so
  each set appears equally often in each position and after each other set.
- Gesture order inside a set: rotates across sets and participants.
- Optional practice with RIGID first (not analysed).
- Ratings 1-7 per gesture: perceived safety and comfort (edit QUESTIONS to add items).
- Unix timestamps at trial start/end, to join with robot telemetry (tau_est, ...).
"""

import argparse
import csv
import datetime as dt
import json
import os
import random
import sys
import threading
import time

import zmq

GESTURES_DEFAULT = ["handshake", "fist_bump", "hug"]

# Rating items asked after EVERY gesture (keep it short: 8 sets x 3 gestures).
# (key, question, low anchor, high anchor) — add items here if needed, e.g.
#   ("naturalness", "How natural did this {gesture} feel?", "not at all natural", "very natural"),
#   ("softness", "How did the robot's arms feel?", "very stiff", "very soft"),  # manipulation check
QUESTIONS = [
    ("safety", "How safe did you feel during this {gesture}?", "not safe at all", "completely safe"),
    ("comfort", "How comfortable was this {gesture}?", "very uncomfortable", "very comfortable"),
]
SCALE_MIN, SCALE_MAX = 1, 7

CSV_FIELDS = [
    "participant_id", "session_id", "control_mode", "trial_kind",
    "set_position", "set_code", "profile", "shoulder", "elbow", "wrist",
    "gesture", "gesture_position", "rep",
    "t_start_unix", "t_end_unix", "duration_s",
] + [q[0] for q in QUESTIONS] + ["valid", "note"]


# --------------------------------------------------------------------------- #
# Counterbalancing
# --------------------------------------------------------------------------- #
def balanced_latin_square(n):
    """Williams design rows for n conditions (n rows if n even, 2n if odd)."""
    first = [0]
    lo, hi = 1, n - 1
    take_lo = True
    while len(first) < n:
        if take_lo:
            first.append(lo)
            lo += 1
        else:
            first.append(hi)
            hi -= 1
        take_lo = not take_lo
    rows = [[(c + r) % n for c in first] for r in range(n)]
    if n % 2 == 1:
        rows += [list(reversed(row)) for row in rows]
    return rows


def participant_number(pid):
    digits = "".join(ch for ch in pid if ch.isdigit())
    return int(digits) if digits else 1


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
def load_study_profiles(path):
    with open(path) as f:
        data = json.load(f)
    profiles = data.get("profiles", data)
    return {k: v for k, v in profiles.items() if not k.startswith("_")}


def group_level(spec, group):
    """'soft' / 'rigid' for a joint group of a profile spec (soft = kp < 1)."""
    entry = spec.get(group, spec.get("default", {"kp": 1.0}))
    return "soft" if float(entry.get("kp", 1.0)) < 0.999 else "rigid"


# --------------------------------------------------------------------------- #
# Robot link (ZMQ heartbeat publisher)
# --------------------------------------------------------------------------- #
class ComplianceLink:
    def __init__(self, port, topic, rate_hz):
        self._sock = zmq.Context.instance().socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://*:{port}")
        self._topic = topic
        self._period = 1.0 / max(rate_hz, 1.0)
        self._cmd = {"profile": "RIGID"}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.estop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _send(self, cmd):
        self._sock.send_string(f"{self._topic} {json.dumps(cmd)}")

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                cmd = dict(self._cmd)
            self._send(cmd)
            self._stop.wait(self._period)

    def set_profile(self, name):
        with self._lock:
            if self.estop:
                return False
            self._cmd = {"profile": name}
            self._send(self._cmd)
        return True

    def trigger_estop(self):
        with self._lock:
            self.estop = True
            self._cmd = {"estop": True}
            self._send(self._cmd)

    def release(self, name):
        with self._lock:
            self.estop = False
            self._send({"release_estop": True, "profile": name})
            self._cmd = {"profile": name}

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


# --------------------------------------------------------------------------- #
# Session state and storage
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self, out_dir, pid):
        self.dir = os.path.join(out_dir, pid)
        os.makedirs(self.dir, exist_ok=True)
        self.meta_path = os.path.join(self.dir, f"{pid}_session.json")
        self.csv_path = os.path.join(self.dir, f"{pid}_ratings.csv")
        self.meta = {}
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as f:
                self.meta = json.load(f)

    def save_meta(self):
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.meta, f, indent=2)
        os.replace(tmp, self.meta_path)

    def append_row(self, row):
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})
            f.flush()
            os.fsync(f.fileno())

    def completed_keys(self):
        done = set()
        if os.path.exists(self.csv_path):
            with open(self.csv_path) as f:
                for r in csv.DictReader(f):
                    done.add((r["trial_kind"], r["set_code"], r["gesture"], r["rep"]))
        return done


# --------------------------------------------------------------------------- #
# Prompts (every prompt understands e / r / b / n / s / q)
# --------------------------------------------------------------------------- #
class Quit(Exception):
    pass


class Skip(Exception):
    pass


class Prompter:
    def __init__(self, link):
        self.link = link
        self.current_profile = "RIGID"
        self.pending_note = []

    def ask(self, prompt, parse=None, allow_empty=False, allow_skip=False):
        while True:
            try:
                raw = input(prompt).strip()
            except EOFError:
                raise Quit()
            low = raw.lower()
            if low.startswith("!") and low not in ("!e", "!r", "!b", "!n", "!s", "!q"):
                print("  commands: !e ESTOP, !r release, !b break, !n note, !s skip, !q quit")
                continue
            if low == "!e":
                self.link.trigger_estop()
                print("  !! ESTOP sent (latched). Type !r to release when safe.")
                continue
            if low == "!r":
                self.link.release(self.current_profile)
                print(f"  ESTOP released -> {self.current_profile}")
                continue
            if low == "!b":
                self.break_pause()
                continue
            if low == "!n":
                self.pending_note.append(input("  note: ").strip())
                continue
            if low == "!s":
                if allow_skip:
                    raise Skip()
                print("  nothing to skip here.")
                continue
            if low == "!q":
                raise Quit()
            if self.link.estop:
                print("  ESTOP is active — type !r to release before continuing.")
                continue
            if raw == "" and allow_empty:
                return ""
            if parse is None:
                return raw
            try:
                return parse(raw)
            except ValueError as err:
                print(f"  {err}")

    def break_pause(self):
        self.link.set_profile("RIGID")
        print("  -- Break. Robot set to RIGID. Press Enter to continue.")
        input()
        self.link.set_profile(self.current_profile)
        print(f"  -- Resuming with the current set.")

    def take_note(self):
        note = " | ".join(n for n in self.pending_note if n)
        self.pending_note = []
        return note


def parse_int(lo, hi):
    def f(raw):
        try:
            v = int(raw)
        except ValueError:
            raise ValueError(f"enter a whole number {lo}-{hi}")
        if not lo <= v <= hi:
            raise ValueError(f"enter a number {lo}-{hi}")
        return v
    return f


def parse_float(lo, hi):
    def f(raw):
        try:
            v = float(raw)
        except ValueError:
            raise ValueError(f"enter a number {lo}-{hi}")
        if not lo <= v <= hi:
            raise ValueError(f"enter a value {lo}-{hi}")
        return v
    return f


def parse_choice(options):
    def f(raw):
        v = raw.lower()
        if v not in options:
            raise ValueError("choose one of: " + ", ".join(options))
        return v
    return f


def countdown(seconds, text):
    if seconds <= 0:
        return
    for t in range(int(round(seconds)), 0, -1):
        print(f"\r  {text} {t} s ", end="", flush=True)
        time.sleep(1.0)
    print("\r" + " " * (len(text) + 12) + "\r", end="")


# --------------------------------------------------------------------------- #
# Session steps
# --------------------------------------------------------------------------- #
def intake(p, session, args):
    m = session.meta
    print("\n=== Participant intake ===")
    consent = p.ask("Signed consent form received? (y/n): ", parse_choice(["y", "n"]))
    if consent != "y":
        print("No consent — the session cannot start.")
        raise Quit()
    name = p.ask("Name (kept only in the separate key file): ")
    m["demographics"] = {
        "age": p.ask("Age (years): ", parse_int(18, 100)),
        "gender": p.ask("Gender (f/m/other/na): ", parse_choice(["f", "m", "other", "na"])),
        "height_cm": p.ask("Height (cm): ", parse_float(100, 230)),
        "weight_kg": p.ask("Weight (kg, Enter to skip): ", parse_float(30, 250), allow_empty=True),
        "dominant_hand": p.ask("Dominant hand (r/l/both): ", parse_choice(["r", "l", "both"])),
        "robot_experience": p.ask("Experience with robots, 1 (none) - 5 (daily): ", parse_int(1, 5)),
        "touched_humanoid_before": p.ask("Physically interacted with a humanoid before? (y/n): ",
                                         parse_choice(["y", "n"])),
    }
    key_path = os.path.join(args.out_dir, "names_key.csv")
    new = not os.path.exists(key_path)
    with open(key_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["participant_id", "name", "date"])
        w.writerow([args.participant, name, dt.date.today().isoformat()])
    session.save_meta()


def plan(session, args, profiles):
    """Set order and gesture orders, fixed at the first run and reused on resume."""
    m = session.meta
    if "set_order" in m:
        return
    names = list(profiles)
    k = participant_number(args.participant)
    rows = balanced_latin_square(len(names))
    order = [names[i] for i in rows[(k - 1) % len(rows)]]
    codes = [chr(ord("A") + i) for i in range(len(names))]  # neutral labels for blinding
    code_of = {n: codes[i] for i, n in enumerate(names)}
    g_rows = balanced_latin_square(len(args.gestures))
    gesture_orders = [[args.gestures[i] for i in g_rows[(k - 1 + s) % len(g_rows)]] for s in range(len(order))]
    m.update({
        "participant_id": args.participant,
        "session_id": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "control_mode": args.control_mode,
        "profiles_file": os.path.abspath(args.profiles),
        "profiles": profiles,
        "set_codes": code_of,
        "set_order": order,
        "gesture_orders": gesture_orders,
        "reps": args.reps,
        "questions": [{"key": q[0], "text": q[1], "low": q[2], "high": q[3]} for q in QUESTIONS],
        "scale": [SCALE_MIN, SCALE_MAX],
    })
    session.save_meta()


def run_trial(p, link, session, profiles, kind, pos, profile, gesture, gpos, rep, args):
    m = session.meta
    spec = profiles.get(profile, {})
    label = m["set_codes"].get(profile, "practice") if kind == "main" else "practice"
    shown = label if args.blind else f"{label} ({profile})"
    print(f"\n--- Set {shown} | gesture {gpos}/{len(args.gestures)}: {gesture.upper()}"
          + (f" | rep {rep}" if args.reps > 1 else ""))
    p.ask(f"  Press Enter when the participant is ready for the {gesture} (!s = skip): ",
          allow_empty=True, allow_skip=True)
    t0 = time.time()
    p.ask("  Press Enter when the gesture is finished: ", allow_empty=True, allow_skip=True)
    t1 = time.time()
    row = {
        "participant_id": m["participant_id"], "session_id": m["session_id"],
        "control_mode": m["control_mode"], "trial_kind": kind,
        "set_position": pos, "set_code": label, "profile": profile,
        "shoulder": group_level(spec, "shoulder"), "elbow": group_level(spec, "elbow"),
        "wrist": group_level(spec, "wrist"),
        "gesture": gesture, "gesture_position": gpos, "rep": rep,
        "t_start_unix": f"{t0:.3f}", "t_end_unix": f"{t1:.3f}", "duration_s": f"{t1 - t0:.2f}",
    }
    print(f"  Ask the participant ({SCALE_MIN} = low, {SCALE_MAX} = high):")
    for key, text, low, high in QUESTIONS:
        row[key] = p.ask(f"    {text.format(gesture=gesture.replace('_', ' '))} "
                         f"[{SCALE_MIN} {low} … {SCALE_MAX} {high}]: ",
                         parse_int(SCALE_MIN, SCALE_MAX))
    valid = p.ask("  Trial valid? (Enter = yes, x = invalid, e.g. robot stumbled / wrong gesture): ",
                  allow_empty=True)
    row["valid"] = 0 if valid.lower() in ("x", "n", "no", "invalid") else 1
    if row["valid"] == 0 and not p.pending_note:
        p.pending_note.append(input("  reason: ").strip())
    row["note"] = p.take_note()
    session.append_row(row)
    print("  saved.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", required=True, help="study profile file (same as deploy's --compliance-profiles)")
    ap.add_argument("--participant", help="participant ID, e.g. P01 (asked if omitted)")
    ap.add_argument("--resume", metavar="ID", help="resume an interrupted session")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/arm_compliance_study_data"),
                    help="data folder, keep it OUTSIDE the git repo (default: ~/arm_compliance_study_data)")
    ap.add_argument("--control-mode", default="teleop", choices=["teleop", "vla", "replay"],
                    help="how the robot's motion is produced in this session")
    ap.add_argument("--gestures", nargs="+", default=GESTURES_DEFAULT)
    ap.add_argument("--reps", type=int, default=1, help="repetitions per gesture per set")
    ap.add_argument("--practice", action="store_true", help="practice round with RIGID first (not analysed)")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds to wait after a profile switch")
    ap.add_argument("--blind", action="store_true", help="show only set letters (A-H), not profile names")
    ap.add_argument("--port", type=int, default=5565)
    ap.add_argument("--topic", default="compliance")
    ap.add_argument("--rate", type=float, default=10.0)
    args = ap.parse_args()

    profiles = load_study_profiles(args.profiles)
    if not profiles:
        sys.exit("no profiles in " + args.profiles)
    os.makedirs(args.out_dir, exist_ok=True)

    link = ComplianceLink(args.port, args.topic, args.rate)
    p = Prompter(link)
    try:
        if args.resume:
            args.participant = args.resume
        elif not args.participant:
            args.participant = p.ask("Participant ID (e.g. P01): ").strip().upper()
        session = Session(args.out_dir, args.participant)
        if args.resume and not session.meta:
            sys.exit(f"nothing to resume for {args.resume} in {args.out_dir}")
        if not args.resume and session.meta:
            sys.exit(f"{args.participant} already has data in {session.dir}; use --resume {args.participant}")

        print(f"\nReminder: deploy must run with --arm-compliance --compliance-profiles {args.profiles}")
        if not args.resume:
            intake(p, session, args)
        plan(session, args, profiles)
        m = session.meta
        done = session.completed_keys()
        print(f"\nSet order for {m['participant_id']}: "
              + " ".join(m["set_codes"][s] for s in m["set_order"])
              + ("" if args.blind else "   (" + ", ".join(m["set_order"]) + ")"))

        if args.practice and not any(k[0] == "practice" for k in done):
            p.current_profile = "RIGID"
            link.set_profile("RIGID")
            print("\n=== Practice (RIGID, not analysed) ===")
            countdown(args.settle, "settling")
            for gpos, g in enumerate(args.gestures, 1):
                try:
                    run_trial(p, link, session, profiles, "practice", 0, "RIGID", g, gpos, 1, args)
                except Skip:
                    print("  skipped.")

        for pos, profile in enumerate(m["set_order"], 1):
            code = m["set_codes"][profile]
            gestures = m["gesture_orders"][pos - 1]
            todo = [(g, r) for g in gestures for r in range(1, args.reps + 1)
                    if ("main", code, g, str(r)) not in done]
            if not todo:
                continue
            p.current_profile = profile
            print(f"\n=== Set {pos}/{len(m['set_order'])}: {code}"
                  + ("" if args.blind else f" ({profile})") + " ===")
            if not link.set_profile(profile):
                print("  ESTOP is active — release it (!r) first.")
                p.ask("  Press Enter when released: ", allow_empty=True)
                link.set_profile(profile)
            countdown(args.settle, "switching stiffness, wait")
            for g, r in todo:
                gpos = gestures.index(g) + 1
                try:
                    run_trial(p, link, session, profiles, "main", pos, profile, g, gpos, r, args)
                except Skip:
                    print("  skipped (not saved; it will be offered again on --resume).")
            m.setdefault("sets_completed", []).append({"set": code, "t_unix": round(time.time(), 3)})
            session.save_meta()

        link.set_profile("RIGID")
        final = p.ask("\nFinal open comment from the participant (Enter to skip): ", allow_empty=True)
        m["final_comment"] = final
        m["finished_unix"] = round(time.time(), 3)
        session.save_meta()
        print(f"\nSession complete. Data in {session.dir}")
    except Quit:
        print("\nSaved. Resume later with --resume " + str(args.participant))
    except KeyboardInterrupt:
        print("\nInterrupted. Everything answered so far is saved. Resume with --resume " + str(args.participant))
    finally:
        link.close()


if __name__ == "__main__":
    main()
