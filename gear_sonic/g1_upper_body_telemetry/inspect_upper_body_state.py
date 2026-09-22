"""Print raw G1 upper-body ``rt/lowstate`` and ``rt/lowcmd`` values."""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from .protocol import JOINT_INDICES, JOINT_NAMES

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
except ImportError as exc:  # pragma: no cover - robot-only dependency
    ChannelFactoryInitialize = ChannelSubscriber = LowCmd_ = LowState_ = None
    _SDK_IMPORT_ERROR = exc
else:
    _SDK_IMPORT_ERROR = None


STATE_FIELDS = ("q", "dq", "ddq", "tau_est")
COMMAND_FIELDS = ("q", "dq", "tau", "kp", "kd")


class UpperBodyDDSReader:
    """Keep the latest valid upper-body LowState and LowCmd DDS samples."""

    def __init__(self, state_topic: str, command_topic: str) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, np.ndarray] | None = None
        self._command: dict[str, np.ndarray] | None = None
        self._state_received_monotonic: float | None = None
        self._command_received_monotonic: float | None = None
        self._state_count = 0
        self._command_count = 0
        self._subscribers = (
            ChannelSubscriber(state_topic, LowState_),
            ChannelSubscriber(command_topic, LowCmd_),
        )
        self._subscribers[0].Init(self._state_callback, 10)
        self._subscribers[1].Init(self._command_callback, 10)
        print(f"[UpperBodyState] DDS LowState subscription: {state_topic}")
        print(f"[UpperBodyState] DDS LowCmd subscription: {command_topic}")

    @staticmethod
    def _selected_motors(motors, source: str):
        if len(motors) <= max(JOINT_INDICES):
            raise ValueError(f"{source} has {len(motors)} motors, expected at least 29")
        return [motors[index] for index in JOINT_INDICES]

    @staticmethod
    def _values(motors, fields: tuple[str, ...], source: str) -> dict[str, np.ndarray]:
        sample = {
            field: np.asarray([getattr(motor, field) for motor in motors], dtype=np.float64)
            for field in fields
        }
        if not all(np.isfinite(values).all() for values in sample.values()):
            raise ValueError(f"{source} contains non-finite upper-body values")
        return sample

    def _state_callback(self, msg) -> None:
        try:
            sample = self._values(
                self._selected_motors(msg.motor_state, "LowState"),
                STATE_FIELDS,
                "LowState",
            )
        except Exception as exc:
            print(f"[UpperBodyState] Invalid LowState: {exc}")
            return

        with self._lock:
            self._state = sample
            self._state_received_monotonic = time.monotonic()
            self._state_count += 1

    def _command_callback(self, msg) -> None:
        try:
            sample = self._values(
                self._selected_motors(msg.motor_cmd, "LowCmd"),
                COMMAND_FIELDS,
                "LowCmd",
            )
        except Exception as exc:
            print(f"[UpperBodyState] Invalid LowCmd: {exc}")
            return

        with self._lock:
            self._command = sample
            self._command_received_monotonic = time.monotonic()
            self._command_count += 1

    def latest(self) -> tuple[dict, dict, float, float, int, int] | None:
        with self._lock:
            if (
                self._state is None
                or self._command is None
                or self._state_received_monotonic is None
                or self._command_received_monotonic is None
            ):
                return None
            return (
                {field: values.copy() for field, values in self._state.items()},
                {field: values.copy() for field, values in self._command.items()},
                self._state_received_monotonic,
                self._command_received_monotonic,
                self._state_count,
                self._command_count,
            )

    def close(self) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber.Close()
            except Exception:
                pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect G1 upper-body rt/lowstate and rt/lowcmd through Unitree SDK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dds-domain-id", type=int, default=0)
    parser.add_argument(
        "--network-interface",
        default=None,
        help="DDS interface, for example enp3s0; omit for SDK auto-selection",
    )
    parser.add_argument("--state-topic", default="rt/lowstate")
    parser.add_argument("--command-topic", default="rt/lowcmd")
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="Print first sample and exit")
    args = parser.parse_args(argv)
    if args.print_hz <= 0:
        parser.error("print-hz must be positive")
    return args


def _print_sample(
    state: dict[str, np.ndarray],
    command: dict[str, np.ndarray],
    state_age_sec: float,
    command_age_sec: float,
    state_count: int,
    command_count: int,
) -> None:
    print(
        f"\n[UpperBodyState] LowState messages={state_count}, "
        f"age={state_age_sec * 1000:.1f} ms | LowCmd messages={command_count}, "
        f"age={command_age_sec * 1000:.1f} ms"
    )
    print("LowState")
    print(
        f"{'joint':<24} {'q [rad]':>11} {'dq [rad/s]':>13} "
        f"{'ddq [rad/s²]':>14} {'tau_est [Nm]':>14}"
    )
    for index, name in enumerate(JOINT_NAMES):
        print(
            f"{name:<24} {state['q'][index]:>11.5f} "
            f"{state['dq'][index]:>13.5f} {state['ddq'][index]:>14.5f} "
            f"{state['tau_est'][index]:>14.5f}"
        )
    print("LowCmd")
    print(
        f"{'joint':<24} {'q_cmd [rad]':>13} {'dq_cmd [rad/s]':>16} "
        f"{'tau_ff [Nm]':>13} {'kp':>11} {'kd':>11}"
    )
    for index, name in enumerate(JOINT_NAMES):
        print(
            f"{name:<24} {command['q'][index]:>13.5f} "
            f"{command['dq'][index]:>16.5f} {command['tau'][index]:>13.5f} "
            f"{command['kp'][index]:>11.5f} {command['kd'][index]:>11.5f}"
        )


def main(argv=None) -> None:
    args = parse_args(argv)
    if _SDK_IMPORT_ERROR is not None:
        raise ImportError(
            "unitree_sdk2py with unitree_hg LowState_/LowCmd_ is required"
        ) from _SDK_IMPORT_ERROR

    if args.network_interface:
        ChannelFactoryInitialize(args.dds_domain_id, networkInterface=args.network_interface)
    else:
        ChannelFactoryInitialize(args.dds_domain_id)

    reader = UpperBodyDDSReader(args.state_topic, args.command_topic)
    period = 1.0 / args.print_hz
    last_printed_counts = (-1, -1)
    last_wait_log = 0.0
    try:
        while True:
            latest = reader.latest()
            now = time.monotonic()
            if latest is None:
                if now - last_wait_log >= 1.0:
                    print(
                        "[UpperBodyState] waiting for both "
                        f"{args.state_topic} and {args.command_topic}..."
                    )
                    last_wait_log = now
                time.sleep(min(period, 0.1))
                continue

            state, command, state_time, command_time, state_count, command_count = latest
            counts = (state_count, command_count)
            if counts != last_printed_counts:
                _print_sample(
                    state,
                    command,
                    now - state_time,
                    now - command_time,
                    state_count,
                    command_count,
                )
                last_printed_counts = counts
                if args.once:
                    return
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[UpperBodyState] stopped")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
