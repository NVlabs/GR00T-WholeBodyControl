"""Nonblocking native6 session endpoint stepped by the full-body simulator."""

import re
import time
import uuid

import msgpack
import numpy as np
import zmq

from gear_sonic.utils.hand_control.inspire.config import (
    DEFAULT_MAPPING_PATH,
    hardware_settings,
    load_mapping,
)
from gear_sonic.utils.hand_control.inspire.contract import (
    ContractError,
    HandStatus,
    InspireHandCommand,
)
from gear_sonic.utils.mujoco_sim.inspire.controller import InspireSimController


class InspireSimGateway:
    """Owns hand transport only; DefaultEnv owns the single physics step.

    Uses the same PREPARE/DISARM receipts as the real gateway. Simulation keeps
    its 200 ms command timeout. A simulator reset revokes the old session.
    """

    def __init__(self, model, data, config_path=DEFAULT_MAPPING_PATH):
        self.controller = InspireSimController(model, data, mapping=load_mapping(config_path))
        self.meta = dict(
            enabled=True,
            instance_id=uuid.uuid4().hex,
            generation=0,
            request_id="",
            session_id="",
            active=False,
            ok=False,
            error="",
            fault_latched=False,
        )
        self.context = zmq.Context()
        self.commands = self.context.socket(zmq.PULL)
        self.states = self.context.socket(zmq.PUB)
        self._next_publish_ns = 0
        settings = hardware_settings(config_path)
        self._period_ns = round(1e9 / float(settings["command_rate_hz"]))
        try:
            self.commands.setsockopt(zmq.LINGER, 0)
            self.commands.setsockopt(zmq.RCVHWM, 64)
            self.commands.setsockopt(zmq.MAXMSGSIZE, 65536)
            self.states.setsockopt(zmq.LINGER, 0)
            self.states.setsockopt(zmq.SNDHWM, 1)
            self.commands.bind(settings["command_endpoint"])
            self.states.bind(settings["state_endpoint"])
        except BaseException:
            self.close()
            raise

    def _operate(self, row):
        meta = self.meta
        for key in ("request_id", "session_id"):
            if not isinstance(row.get(key), str) or not re.fullmatch("[0-9a-f]{32}", row[key]):
                raise ContractError(f"invalid operator {key}")
        if type(row.get("generation")) is not int or row["generation"] < 0:
            raise ContractError("invalid operator generation")
        if row["request_id"] == meta["request_id"]:
            return
        meta.update(request_id=row["request_id"], ok=False, error="")
        operation = row.get("operation")
        guard = self.controller.guard
        if row.get("instance_id") != meta["instance_id"] or (
            operation != "DISARM" and row["generation"] != meta["generation"]
        ):
            meta["error"] = "stale_operator_request"
        elif operation not in ("PREPARE", "DISARM"):
            meta["error"] = "unknown_operator_operation"
        elif operation == "PREPARE" and (
            meta["fault_latched"] or guard.status != HandStatus.DISARMED
        ):
            meta["error"] = "prepare_requires_healthy_disarmed_without_fault"
        elif operation == "PREPARE" and row["session_id"] == meta["session_id"]:
            meta["error"] = "new_episode_identity_required"
        elif operation == "DISARM" and row["session_id"] != meta["session_id"]:
            meta["error"] = "wrong_operator_session"
        if meta["error"]:
            return
        meta["generation"] += 1
        if operation == "PREPARE":
            guard.clear_fault()
            guard.arm()
            meta["session_id"] = row["session_id"]
        else:
            guard.disarm()
        meta.update(ok=True, active=operation == "PREPARE")

    def _receive(self, row, now_ns):
        if not isinstance(row, dict):
            raise ContractError("command must be a map")
        if row.get("schema") == "g1.operator.command.v1":
            self._operate(row)
            return
        if not self.meta["active"] or row.get("operator_session_id") != self.meta["session_id"]:
            return
        command = InspireHandCommand.from_wire_dict(row)
        if self.controller.guard.last_command is None:
            left, right, _, _ = self.controller.layout.read(self.controller.data)
            tolerance = np.array([0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
            if np.any(abs(command.left_q - left) > tolerance) or np.any(
                abs(command.right_q - right) > tolerance
            ):
                raise ContractError("first target must be near measured feedback")
        self.controller.submit(command, now_ns=now_ns)

    def step(self, *, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        guard = self.controller.guard
        # Expire before receiving: a late packet cannot revive a stale session.
        guard.poll(now_ns=now_ns)
        self.meta["fault_latched"] |= guard.status in (HandStatus.FAULT, HandStatus.STALE)
        for _ in range(32):
            try:
                payload = self.commands.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                self._receive(msgpack.unpackb(payload, raw=False), now_ns)
            except (ValueError, TypeError, KeyError, msgpack.ExtraData) as exc:
                guard.status, guard.last_error = HandStatus.FAULT, str(exc)
                self.meta["fault_latched"] = True
        state = self.controller.step(now_ns=now_ns)
        self.meta["fault_latched"] |= state.status in (HandStatus.FAULT, HandStatus.STALE)
        if now_ns >= self._next_publish_ns:
            row = state.to_wire_dict()
            command = guard.last_command
            if command is not None:
                row.update(
                    accepted_command_sequence=command.sequence,
                    left_command_q=command.left_q.tolist(),
                    right_command_q=command.right_q.tolist(),
                )
            row.update(operator_session=dict(self.meta), simulation=True)
            self.states.send(msgpack.packb(row, use_bin_type=True), zmq.NOBLOCK)
            self._next_publish_ns = now_ns + self._period_ns
        return state

    def reset(self):
        self.controller.guard.clear_fault()
        self.meta.update(
            instance_id=uuid.uuid4().hex,
            generation=0,
            request_id="",
            session_id="",
            active=False,
            ok=False,
            error="",
            fault_latched=False,
        )
        self.controller.step()
        self._next_publish_ns = 0

    def close(self):
        self.controller.guard.disarm()
        self.controller.step()
        self.commands.close(linger=0)
        self.states.close(linger=0)
        self.context.term()
