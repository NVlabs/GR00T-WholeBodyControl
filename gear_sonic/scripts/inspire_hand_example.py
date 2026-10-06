"""Read hand feedback or send explicitly requested trajectories."""

import argparse
from contextlib import nullcontext
import json
import time
from types import SimpleNamespace

import numpy as np

# Native6 order: pinky, ring, middle, index, thumb bend, thumb rotation.
# Combined thumb closure uses 0.3 rad rotation to keep clear of a closed index.
OPEN_HAND = np.zeros(6, dtype=np.float32)
CLOSED_HAND = np.asarray([1.7, 1.7, 1.7, 1.7, 0.6, 0.3], dtype=np.float32)
VELOCITY_LIMITS = np.asarray([2, 2, 2, 2, 1, 1], dtype=np.float32)
VELOCITY_MARGIN = 0.8


def _transition_seconds(
    state,
    left_target: np.ndarray,
    right_target: np.ndarray,
    *,
    velocity_limits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the supplied starts and a velocity-safe transition duration."""
    left_start = state.left_q.copy()
    right_start = state.right_q.copy()
    maximum_delta = np.maximum(np.abs(left_target - left_start), np.abs(right_target - right_start))
    duration = float(np.max(maximum_delta / (velocity_limits * VELOCITY_MARGIN)))
    return left_start, right_start, duration


def _interpolate(start: np.ndarray, target: np.ndarray, elapsed_s: float, duration_s: float):
    if duration_s <= 0:
        return target.copy()
    fraction = min(max(elapsed_s / duration_s, 0.0), 1.0)
    return start + (target - start) * fraction


def finger_sequence_stages(initial):
    target = np.array(initial, dtype=np.float32, copy=True)
    for side, label in enumerate(("left", "right")):
        for name, indices in [
            ("thumb", [4, 5]),
            ("index", [3]),
            ("middle", [2]),
            ("ring", [1]),
            ("pinky", [0]),
        ]:
            for action, pose in [("close", CLOSED_HAND), ("open", OPEN_HAND)]:
                start = target.copy()
                target[side, indices] = pose[indices]
                _, _, duration = _transition_seconds(
                    SimpleNamespace(left_q=start[0], right_q=start[1]),
                    target[0],
                    target[1],
                    velocity_limits=VELOCITY_LIMITS,
                )
                yield dict(
                    side=side,
                    hand=label,
                    finger=name,
                    indices=indices,
                    action=action,
                    start=start,
                    target=target.copy(),
                    duration_s=duration,
                )


def run_finger_sequence(
    hand, duration=2.0, check=lambda: None, sample=lambda *args: None, log=lambda row: None
):
    state = hand.read_state()
    if state.error or state.left_position is None or state.right_position is None:
        raise RuntimeError(state.error or "measured feedback is required")
    from gear_sonic.utils.hand_control.inspire.contract import (
        CANONICAL_LOWER_RAD,
        CANONICAL_UPPER_RAD,
    )

    initial = np.clip(
        np.array([state.left_position, state.right_position], dtype=np.float32),
        CANONICAL_LOWER_RAD,
        CANONICAL_UPPER_RAD,
    )
    hand.set_target(*initial)
    command_index = 0
    log(
        dict(
            event="seed",
            command_index=command_index,
            submitted_monotonic_ns=time.monotonic_ns(),
            target=initial.tolist(),
            measured=[list(state.left_position), list(state.right_position)],
            feedback_received_monotonic_ns=state.received_monotonic_ns,
        )
    )
    period = 1 / float(getattr(hand, "settings", {}).get("command_rate_hz", 50))
    completed = []
    for stage_index, stage in enumerate(finger_sequence_stages(initial)):
        event = {k: v for k, v in stage.items() if k not in ("start", "target")}
        event["stage_index"] = stage_index
        event["target"] = stage["target"].tolist()
        event["started"] = time.monotonic()
        log(dict(event="stage_start", **event))
        print(f"{stage['hand']} {stage['finger']} {stage['action']}", flush=True)
        begin = time.monotonic()
        deadline = begin
        while True:
            deadline += period
            time.sleep(max(0, deadline - time.monotonic()))
            check()
            now = time.monotonic()
            elapsed = now - begin
            target = _interpolate(stage["start"], stage["target"], elapsed, stage["duration_s"])
            hand.set_target(*target)
            submitted_ns = time.monotonic_ns()
            command_index += 1
            state = hand.read_state()
            if state.error:
                raise RuntimeError(state.error)
            log(
                dict(
                    event="sample",
                    stage_index=stage_index,
                    hand=stage["hand"],
                    finger=stage["finger"],
                    action=stage["action"],
                    command_index=command_index,
                    submitted_monotonic_ns=submitted_ns,
                    target=target.tolist(),
                    measured=[list(state.left_position), list(state.right_position)],
                    feedback_received_monotonic_ns=state.received_monotonic_ns,
                )
            )
            sample(submitted_ns / 1e9, state)
            actual = np.array([state.left_position, state.right_position])
            error = float(
                np.max(
                    np.abs(
                        actual[stage["side"], stage["indices"]]
                        - stage["target"][stage["side"], stage["indices"]]
                    )
                )
            )
            if elapsed >= max(duration, stage["duration_s"]):
                break
            if deadline < time.monotonic() - period:
                deadline = time.monotonic()
        event.update(ended=time.monotonic(), measured=actual.tolist(), max_selected_error_rad=error)
        completed.append(event)
        log(dict(event="stage_complete", **event))
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Inspire hand interface: simulation or direct-serial gateway"
    )
    parser.add_argument("--backend", choices=("sim", "sim-remote", "real"), default="sim")
    parser.add_argument("--config", help="gateway configuration for sim-remote or real")
    parser.add_argument(
        "--sequence",
        action="store_true",
        help=(
            "close/open each finger: left thumb to pinky, then right; explicit hardware motion with --backend real"
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="simulation only: bend left index and rotate right thumb",
    )
    parser.add_argument("--left", type=float, nargs=6, help="explicit left target in radians")
    parser.add_argument("--right", type=float, nargs=6, help="explicit right target in radians")
    parser.add_argument(
        "--duration", type=float, default=2.0, help="trajectory duration in seconds"
    )
    parser.add_argument(
        "--log-file", help="write sequence targets and measured feedback to a new JSONL file"
    )
    args = parser.parse_args()
    if args.log_file and not args.sequence:
        parser.error("--log-file requires --sequence")
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be positive and finite")
    if args.sequence and (args.demo or args.left is not None or args.right is not None):
        parser.error("--sequence cannot be combined with --demo or explicit targets")
    if args.demo and (args.backend == "real" or args.left is not None or args.right is not None):
        parser.error(
            "--demo is simulation-only; supply explicit --left and --right for real commands"
        )
    if (args.left is None) != (args.right is None):
        parser.error("--left and --right must be supplied together")
    if args.backend == "sim":
        if args.config:
            parser.error("--config currently selects real deployment settings only")
        from gear_sonic.utils.mujoco_sim.inspire.backend import create_backend

        hand = create_backend()
        rate = 50.0
    else:
        if args.backend == "sim-remote":
            from gear_sonic.utils.mujoco_sim.inspire.backend import (
                create_remote_backend as create_backend,
            )
        else:
            from gear_sonic.utils.hand_control.inspire.client import create_backend
        hand = create_backend(**({"config_path": args.config} if args.config else {}))
        rate = float(hand.settings["command_rate_hz"])
    try:
        hand.connect()
        initial = hand.read_state()
        print(hand.description)
        print(initial)
        if args.sequence:
            with (
                open(args.log_file, "x", encoding="utf-8")
                if args.log_file
                else nullcontext(None) as output
            ):

                def record(row):
                    if output is not None:
                        output.write(json.dumps(row, allow_nan=False) + "\n")
                        output.flush()

                run_finger_sequence(hand, duration=args.duration, log=record)
        elif args.demo or args.left is not None:
            if initial.error or initial.left_position is None or initial.right_position is None:
                raise RuntimeError(initial.error or "measured feedback is required")
            start = np.maximum(0, np.array([initial.left_position, initial.right_position]))
            target = (
                np.array([args.left, args.right], dtype=float) if not args.demo else start.copy()
            )
            if args.demo:
                target[0, 3] = 0.4
                target[1, 5] = 0.2
            # Keep interpolation below half the per-joint velocity limits.
            velocity = np.array([2, 2, 2, 2, 1, 1])
            if np.any(abs(target - start) / args.duration > velocity * 0.5):
                raise ValueError("increase --duration to respect the hand velocity limits")
            hand.set_target(*start)
            begin = time.monotonic()
            deadline = begin
            while True:
                deadline += 1 / rate
                time.sleep(max(0, deadline - time.monotonic()))
                fraction = min(1.0, (time.monotonic() - begin) / args.duration)
                hand.set_target(*(start + fraction * (target - start)))
                if fraction == 1:
                    break
                if deadline < time.monotonic() - 1 / rate:
                    deadline = time.monotonic()  # Do not burst to catch up.
            print(hand.read_state())
    finally:
        hand.close()


if __name__ == "__main__":
    main()
