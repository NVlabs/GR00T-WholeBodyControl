"""Adaptive direct control of the BrainCo dexterous hands."""

import threading

import numpy as np

from .constants import (
    BRAINCO_CLOSED_Q, BRAINCO_DEFAULT_EXCLUDED_FINGERS,
    BRAINCO_EXCLUDED_FINGER_CHOICES, BRAINCO_LEFT_COMMAND_TOPIC,
    BRAINCO_LEFT_STATE_TOPIC, BRAINCO_MOTOR_NAMES, BRAINCO_NUM_MOTORS,
    BRAINCO_OPEN_Q, BRAINCO_RIGHT_COMMAND_TOPIC, BRAINCO_RIGHT_STATE_TOPIC,
    BRAINCO_TAU_THRESHOLDS, BRAINCO_TRIGGER_RANGE_CHOICES,
    BRAINCO_TRIGGER_THRESHOLD,
)
from .runtime import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber, MotorCmds_,
    MotorStates_, _BRAINCO_DDS_IMPORT_ERROR, unitree_go_msg_dds__MotorCmd_,
)

try:  # Safe stop (SONIC deploy k / voice): open the hands. Optional helper.
    from .safe_stop_hands import get_hand_guard
except Exception as _safe_stop_import_error:  # noqa: BLE001
    get_hand_guard = None
    print(f"[BrainCoHand] safe_stop_hands unavailable ({_safe_stop_import_error}); "
          "hands will not open on a safe stop")

def _normalize_trigger_value(trigger: float, trigger_range: str = "auto") -> float:
    """Return trigger press in [0, 1], accepting both 0..1 and legacy 10..0 ranges."""
    if trigger_range not in BRAINCO_TRIGGER_RANGE_CHOICES:
        raise ValueError(
            f"trigger_range must be one of {BRAINCO_TRIGGER_RANGE_CHOICES}, got {trigger_range}"
        )
    value = float(trigger)
    if trigger_range == "legacy_10_to_0" or (trigger_range == "auto" and value > 1.0):
        value = (10.0 - value) / 10.0
    return float(np.clip(value, 0.0, 1.0))


def compute_brainco_hand_target(
    trigger: float,
    threshold: float = BRAINCO_TRIGGER_THRESHOLD,
    trigger_range: str = "auto",
    grip: float = 0.0,
) -> np.ndarray:
    """Map only the VR trigger to a BrainCo hand command."""
    del grip  # Grip must not keep the hand closed after trigger release.
    trigger_pressed = _normalize_trigger_value(trigger, trigger_range) >= threshold
    return (BRAINCO_CLOSED_Q if trigger_pressed else BRAINCO_OPEN_Q).copy()


def compute_brainco_hand_targets_from_inputs(
    left_trigger: float,
    right_trigger: float,
    threshold: float = BRAINCO_TRIGGER_THRESHOLD,
    trigger_range: str = "auto",
    left_grip: float = 0.0,
    right_grip: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute left/right BrainCo q targets from VR controller inputs."""
    return (
        compute_brainco_hand_target(left_trigger, threshold, trigger_range, left_grip),
        compute_brainco_hand_target(right_trigger, threshold, trigger_range, right_grip),
    )


class BraincoHandCommandPublisher:
    """Publishes direct BrainCo targets and holds each finger at torque contact."""

    def __init__(
        self,
        dds_domain_id: int = 0,
        dds_network_interface: str | None = None,
        trigger_threshold: float = BRAINCO_TRIGGER_THRESHOLD,
        trigger_range: str = "auto",
        target_q: np.ndarray | None = None,
        tau_thresholds: np.ndarray | None = None,
        excluded_fingers: tuple[str, ...] | list[str] = BRAINCO_DEFAULT_EXCLUDED_FINGERS,
    ):
        if _BRAINCO_DDS_IMPORT_ERROR is not None:
            raise ImportError(
                "unitree_sdk2py is required for BrainCo hand control. "
                "Install/source the Unitree SDK or run with --disable_brainco_hand."
            ) from _BRAINCO_DDS_IMPORT_ERROR
        if trigger_range not in BRAINCO_TRIGGER_RANGE_CHOICES:
            raise ValueError(
                f"trigger_range must be one of {BRAINCO_TRIGGER_RANGE_CHOICES}, got {trigger_range}"
            )

        self.trigger_threshold = trigger_threshold
        self.trigger_range = trigger_range
        self.target_q = np.asarray(
            BRAINCO_CLOSED_Q if target_q is None else target_q,
            dtype=np.float32,
        ).reshape(BRAINCO_NUM_MOTORS)
        self.tau_thresholds = np.asarray(
            BRAINCO_TAU_THRESHOLDS if tau_thresholds is None else tau_thresholds,
            dtype=np.float32,
        ).reshape(BRAINCO_NUM_MOTORS)
        if np.any((self.target_q < 0.0) | (self.target_q > 1.0)):
            raise ValueError("BrainCo finger targets must be between 0 and 1")
        if np.any(self.tau_thresholds < 0.0):
            raise ValueError("BrainCo tau thresholds must be non-negative")
        unknown_excluded_fingers = sorted(
            set(excluded_fingers) - set(BRAINCO_EXCLUDED_FINGER_CHOICES)
        )
        if unknown_excluded_fingers:
            raise ValueError(
                "Unknown BrainCo excluded fingers: "
                f"{unknown_excluded_fingers}; choices: {BRAINCO_EXCLUDED_FINGER_CHOICES}"
            )
        self.excluded_fingers = tuple(dict.fromkeys(excluded_fingers))
        self._excluded_motor_indices = {"left": set(), "right": set()}
        for finger in self.excluded_fingers:
            hand, motor_name = finger.split("_", 1)
            self._excluded_motor_indices[hand].add(BRAINCO_MOTOR_NAMES.index(motor_name))
        self._auto_detected_legacy_trigger_range = False
        self._lock = threading.Lock()
        self._closed = False
        self._states = {"left": None, "right": None}
        self._state_version = {"left": 0, "right": 0}
        self._arm_after_state_version = {"left": 0, "right": 0}
        self._q_cmd = {
            "left": BRAINCO_OPEN_Q.copy(),
            "right": BRAINCO_OPEN_Q.copy(),
        }
        self._requested_target = {
            "left": BRAINCO_OPEN_Q.copy(),
            "right": BRAINCO_OPEN_Q.copy(),
        }
        self._stopped = {
            "left": np.zeros(BRAINCO_NUM_MOTORS, dtype=bool),
            "right": np.zeros(BRAINCO_NUM_MOTORS, dtype=bool),
        }
        self._armed = {
            "left": np.zeros(BRAINCO_NUM_MOTORS, dtype=bool),
            "right": np.zeros(BRAINCO_NUM_MOTORS, dtype=bool),
        }
        self._trigger_active = {"left": False, "right": False}
        self._safe_stop_guard = get_hand_guard("rearm") if get_hand_guard else None

        ChannelFactoryInitialize(dds_domain_id, networkInterface=dds_network_interface)

        self.left_publisher = ChannelPublisher(BRAINCO_LEFT_COMMAND_TOPIC, MotorCmds_)
        self.left_publisher.Init()
        self.right_publisher = ChannelPublisher(BRAINCO_RIGHT_COMMAND_TOPIC, MotorCmds_)
        self.right_publisher.Init()
        self.left_subscriber = ChannelSubscriber(BRAINCO_LEFT_STATE_TOPIC, MotorStates_)
        self.left_subscriber.Init(lambda msg: self._state_callback("left", msg), 10)
        self.right_subscriber = ChannelSubscriber(BRAINCO_RIGHT_STATE_TOPIC, MotorStates_)
        self.right_subscriber.Init(lambda msg: self._state_callback("right", msg), 10)

        self.left_msg = self._make_command_message()
        self.right_msg = self._make_command_message()
        print(
            "[BrainCoHand] DDS publishers ready: "
            f"{BRAINCO_LEFT_COMMAND_TOPIC}, {BRAINCO_RIGHT_COMMAND_TOPIC}"
        )
        excluded_summary = ", ".join(self.excluded_fingers) or "none"
        print(f"[BrainCoHand] Excluded fingers (forced q=0): {excluded_summary}")

    def _state_callback(self, hand: str, msg) -> None:
        with self._lock:
            self._states[hand] = msg
            self._state_version[hand] += 1

    def _make_command_message(self):
        msg = MotorCmds_()
        msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(BRAINCO_NUM_MOTORS)]
        for cmd in msg.cmds:
            cmd.q = 0.0
            cmd.dq = 1.0
        return msg

    def publish(
        self, left_q_target: np.ndarray, right_q_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._closed:
            return self._q_cmd["left"].copy(), self._q_cmd["right"].copy()
        left_q_target = self._zero_excluded_fingers("left", left_q_target)
        right_q_target = self._zero_excluded_fingers("right", right_q_target)

        with self._lock:
            left_q_target = self._guarded_target("left", left_q_target)
            right_q_target = self._guarded_target("right", right_q_target)
            # Enforce the mask immediately before DDS Write as a final safety
            # barrier, independent of the adaptive controller state.
            left_q_target = self._zero_excluded_fingers("left", left_q_target)
            right_q_target = self._zero_excluded_fingers("right", right_q_target)
            self._q_cmd["left"][:] = left_q_target
            self._q_cmd["right"][:] = right_q_target
            for idx in range(BRAINCO_NUM_MOTORS):
                self.left_msg.cmds[idx].q = float(left_q_target[idx])
                self.right_msg.cmds[idx].q = float(right_q_target[idx])

            self.left_publisher.Write(self.left_msg)
            self.right_publisher.Write(self.right_msg)
            return left_q_target.copy(), right_q_target.copy()

    def _guarded_target(self, hand: str, target: np.ndarray) -> np.ndarray:
        """Safe stop active: open slowly and ignore the trigger; else the adaptive grasp."""
        if self._safe_stop_guard is not None:
            q, overridden = self._safe_stop_guard.apply(hand, target, self._q_cmd[hand])
            if overridden:
                # Reset the grasp state so the next trigger press starts a fresh close.
                self._stopped[hand][:] = False
                self._armed[hand][:] = False
                self._trigger_active[hand] = False
                self._requested_target[hand] = BRAINCO_OPEN_Q.copy()
                return np.asarray(q, dtype=np.float32)
        return self._adapt_target(hand, target)

    def _zero_excluded_fingers(self, hand: str, target: np.ndarray) -> np.ndarray:
        """Return a command copy with excluded motor positions forced to zero."""
        masked_target = np.asarray(target, dtype=np.float32).reshape(
            BRAINCO_NUM_MOTORS
        ).copy()
        excluded_indices = self._excluded_motor_indices[hand]
        if excluded_indices:
            masked_target[list(excluded_indices)] = 0.0
        return masked_target

    def _adapt_target(self, hand: str, target: np.ndarray) -> np.ndarray:
        """Close at target q and freeze each motor on a fresh raw-tau contact."""
        q_cmd = self._q_cmd[hand]
        previous_target = self._requested_target[hand]
        stopped = self._stopped[hand]
        armed = self._armed[hand]
        state = self._states[hand]
        clipped_target = np.clip(target, 0.0, 1.0).astype(np.float32)
        closing = clipped_target > BRAINCO_OPEN_Q
        trigger_active = bool(np.any(closing))

        # Trigger released: open every motor and fully rearm the next grasp.
        if not trigger_active:
            q_cmd[:] = BRAINCO_OPEN_Q
            stopped[:] = False
            armed[:] = False
            self._trigger_active[hand] = False
            self._requested_target[hand] = BRAINCO_OPEN_Q.copy()
            return q_cmd.copy()

        changed = not np.allclose(clipped_target, previous_target)
        new_press = not self._trigger_active[hand]
        if new_press or changed:
            # First publish sends every allowed motor directly to its final q.
            # Contact detection starts only after a newer DDS state arrives,
            # so stale tau cannot suppress the initial close command.
            q_cmd[:] = clipped_target
            stopped[:] = False
            armed[:] = False
            self._arm_after_state_version[hand] = self._state_version[hand]
            self._trigger_active[hand] = True
            self._requested_target[hand] = clipped_target.copy()
            print(f"[BrainCoHand] TRIGGER {hand}: sent all finger targets")
            return q_cmd.copy()

        if not np.any(armed):
            if self._state_version[hand] <= self._arm_after_state_version[hand]:
                return q_cmd.copy()
            armed[:] = closing

        for idx in range(BRAINCO_NUM_MOTORS):
            target_q = float(clipped_target[idx])
            if not armed[idx]:
                q_cmd[idx] = float(BRAINCO_OPEN_Q[idx])
                continue
            if stopped[idx]:
                continue

            if state is not None and len(state.states) >= BRAINCO_NUM_MOTORS:
                motor = state.states[idx]
                tau = float(motor.tau_est)
                if tau <= float(self.tau_thresholds[idx]):
                    q_cmd[idx] = target_q
                    continue
                q_cmd[idx] = float(np.clip(motor.q, 0.0, 1.0))
                stopped[idx] = True
                print(
                    f"[BrainCoHand] CONTACT {hand} {BRAINCO_MOTOR_NAMES[idx]}: "
                    f"q={q_cmd[idx]:.3f}, tau={tau:.4f}, "
                    f"threshold={self.tau_thresholds[idx]:.4f}"
                )
                continue

            q_cmd[idx] = target_q

        return q_cmd.copy()

    def _resolve_trigger_range(self, left_trigger: float, right_trigger: float) -> str:
        if self.trigger_range != "auto":
            return self.trigger_range
        if float(left_trigger) > 1.0 or float(right_trigger) > 1.0:
            self._auto_detected_legacy_trigger_range = True
        return (
            "legacy_10_to_0"
            if self._auto_detected_legacy_trigger_range
            else "normal_0_to_1"
        )

    def publish_from_controller_inputs(
        self,
        left_trigger: float,
        right_trigger: float,
        left_grip: float = 0.0,
        right_grip: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        trigger_range = self._resolve_trigger_range(left_trigger, right_trigger)
        del left_grip, right_grip
        left_pressed = (
            _normalize_trigger_value(left_trigger, trigger_range)
            >= self.trigger_threshold
        )
        right_pressed = (
            _normalize_trigger_value(right_trigger, trigger_range)
            >= self.trigger_threshold
        )
        left_q_target = self.target_q.copy() if left_pressed else BRAINCO_OPEN_Q.copy()
        right_q_target = self.target_q.copy() if right_pressed else BRAINCO_OPEN_Q.copy()
        left_q_target = self._zero_excluded_fingers("left", left_q_target)
        right_q_target = self._zero_excluded_fingers("right", right_q_target)
        return self.publish(left_q_target, right_q_target)

    def close(self) -> None:
        self._closed = True
        publishers = (
            getattr(self, "left_publisher", None),
            getattr(self, "right_publisher", None),
        )
        for publisher in publishers:
            if publisher is not None:
                try:
                    publisher.Close()
                except Exception as e:
                    print(f"[BrainCoHand] Warning: failed to close DDS publisher: {e}")
        subscribers = (
            getattr(self, "left_subscriber", None),
            getattr(self, "right_subscriber", None),
        )
        for subscriber in subscribers:
            if subscriber is not None:
                try:
                    subscriber.Close()
                except Exception as e:
                    print(f"[BrainCoHand] Warning: failed to close DDS subscriber: {e}")
