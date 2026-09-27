#!/usr/bin/env python3
"""Record webcam test clips for validating the teleop safety gating.

Run (any env with opencv, e.g. .venv_teleop, from repo root):
    python gear_sonic/scripts/record_test_clips.py [--camera_id 0]

Records the 4 scenarios below (30 s each) into test_clips/.
Press ENTER to start each clip; ESC/q in the preview window aborts a clip.
"""

import argparse
import time
from pathlib import Path

import cv2

SCENARIOS = [
    ("01_full_body", "FULL BODY: stand fully visible (head to feet), move arms/torso normally"),
    ("02_leave_return", "LEAVE/RETURN: start visible, walk OUT of frame, stay out ~5 s, come back (2-3x)"),
    ("03_torso_only", "TORSO ONLY: come close so only your waist-up is visible, move around"),
    ("04_two_people", "TWO PEOPLE: a second person enters, walks around you, both visible"),
]


def record(cap, path: Path, seconds: float) -> bool:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            writer.release()
            return False
        writer.write(frame)
        remaining = int(seconds - (time.monotonic() - t0))
        disp = frame.copy()
        cv2.putText(disp, f"REC {path.stem}  {remaining}s", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("record_test_clips", disp)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            writer.release()
            return False
    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out_dir", default="test_clips")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera_id}")

    for name, instructions in SCENARIOS:
        path = out_dir / f"{name}.mp4"
        if path.exists():
            print(f"[skip] {path} already exists (delete it to re-record)")
            continue
        print(f"\n=== {name} ===\n{instructions}")
        input(f"ENTER to start recording {args.seconds:.0f}s… ")
        if record(cap, path, args.seconds):
            print(f"[ok] saved {path}")
        else:
            print(f"[aborted] {name} — run again to retry")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Clips in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
