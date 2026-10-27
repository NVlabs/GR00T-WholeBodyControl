"""HandBackend for the direct-serial gateway's ZMQ session protocol.

The gateway is launched separately on the machine with the serial devices.
connect reads feedback only. The first valid target acquires a session; stop
requests measured-position hold/disarm and verifies its receipt.
"""

import time
from types import SimpleNamespace
import uuid

import numpy as np

from gear_sonic.utils.hand_control.inspire.config import (
    DEFAULT_MAPPING_PATH,
    hardware_settings,
    load_mapping,
)
from gear_sonic.utils.hand_control.inspire.contract import (
    ContractError,
    InspireHandCommand,
    InspireHandState,
)
from gear_sonic.utils.hand_control.inspire.safety import CommandGuard
from gear_sonic.utils.hand_control.interface import HandBackend, HandDescription, HandState


class InspireGatewayBackend(HandBackend):
    def __init__(self, config_path=DEFAULT_MAPPING_PATH, *, simulation=False):
        self._simulation = simulation
        self.mapping = load_mapping(config_path)
        self.settings = hardware_settings(config_path)
        self._session = None
        self._stopped = False
        self._prepared = False
        self._sequence = 0
        self._guard = CommandGuard(self.mapping)

    @property
    def description(self):
        return HandDescription(
            "inspire_rh56dfx",
            tuple(j.urdf_joint for j in self.mapping.left),
            tuple(j.urdf_joint for j in self.mapping.right),
        )

    def connect(self):
        if self._session is not None:
            if self._stopped:
                raise RuntimeError("close before reconnecting a stopped backend")
            return
        from gear_sonic.utils.hand_control.inspire.session import OperatorSession

        cfg = self.settings
        session = OperatorSession(
            SimpleNamespace(
                hand_command_endpoint=cfg["command_endpoint"],
                hand_state_endpoint=cfg["state_endpoint"],
                stop_timeout=cfg["operation_timeout_s"],
                state_timeout=cfg["state_timeout_s"],
            )
        )
        try:
            rows = session.ready()
            if (rows["hand"].get("simulation") is True) != self._simulation:
                raise RuntimeError("hand gateway mode differs from selected backend")
        except BaseException:
            session.close()
            raise
        self._session = session
        self._stopped = self._prepared = False
        self._sequence = 0
        self._guard = CommandGuard(self.mapping)
        self._guard.arm()

    def _require_connected(self):
        if self._session is None:
            raise RuntimeError("hand backend is not connected")

    def set_target(self, left, right):
        self._require_connected()
        if self._stopped:
            raise RuntimeError("hand backend is stopped")
        try:
            # Validate both hands before acquiring ownership or sending either target.
            left = self.mapping.validate_limits(left, field_name="left")
            right = self.mapping.validate_limits(right, field_name="right", side="right")
            state = self.read_state()
            if state.error:
                raise RuntimeError(state.error)
            if not self._prepared:
                tolerance = np.array([0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
                if np.any(abs(left - state.left_position) > tolerance) or np.any(
                    abs(right - state.right_position) > tolerance
                ):
                    raise ContractError(
                        "first target must be near measured feedback; ramp from feedback"
                    )
                self._session.prepare(uuid.uuid4().hex)
                self._prepared = True
            now = time.monotonic_ns()
            command = InspireHandCommand(self._sequence + 1, now, left, right)
            self._guard.accept(command, now_ns=now)
            payload = command.to_wire_dict()
            payload["operator_session_id"] = self._session.eid
            self._session.io.send("hand", payload)
            self._sequence += 1
        except Exception as exc:
            try:
                self.stop()
            except Exception as stop_error:
                raise RuntimeError(f"{exc}; DISARM was not confirmed: {stop_error}") from exc
            raise

    def read_state(self):
        self._require_connected()
        session = self._session
        # Streaming must not wait another 20 ms for each feedback packet.
        # The last unique packet's local receive time still bounds freshness.
        session._poll(timeout_ms=0)
        row = session.rows.get("hand")
        if row is None:
            return HandState(error="no measured feedback")
        payload, received_s = row
        if time.monotonic() - received_s > self.settings["state_timeout_s"]:
            return HandState(
                received_monotonic_ns=int(received_s * 1e9), error="stale measured feedback"
            )
        native = InspireHandState.from_wire_dict(payload)
        error = None
        if session.faulted or payload["status"] in ("FAULT", "STALE"):
            error = f"gateway {payload['status']} (fault latched={session.faulted})"
        elif self._prepared and (
            payload["status"] != "ARMED"
            or payload["operator_session"]["session_id"] != session.eid
            or not payload["operator_session"]["active"]
        ):
            error = "gateway ownership was lost"
        elif self._stopped:
            error = "STOPPED"
        return HandState(
            tuple(map(float, native.left_q)),
            tuple(map(float, native.right_q)),
            int(received_s * 1e9),
            error,
        )

    def stop(self):
        self._stopped = True  # Stop submissions even if the disarm receipt is lost.
        if self._session is not None and self._session.attempted:
            self._session.finish()
        self._prepared = False

    def close(self):
        if self._session is None:
            return
        try:
            self.stop()
        finally:
            self._session.close()
            self._session = None


# Backward-compatible name for direct-serial users.
InspireRealBackend = InspireGatewayBackend


def create_backend(config_path=DEFAULT_MAPPING_PATH) -> HandBackend:
    return InspireRealBackend(config_path)
