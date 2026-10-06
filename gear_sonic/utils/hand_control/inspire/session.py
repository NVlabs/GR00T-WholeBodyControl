"""Hand control sessions and command receipts."""

from __future__ import annotations

import time
import uuid

import msgpack
import zmq

from gear_sonic.utils.hand_control.inspire.contract import InspireHandState


class OperatorSessionError(RuntimeError):
    pass


class GatewayIO:
    def __init__(self, args):
        self.context = zmq.Context()
        self.commands, self.states = {}, {}
        self.poller = zmq.Poller()
        try:
            for side in ("hand",):
                command = self.context.socket(zmq.PUSH)
                self.commands[side] = command
                command.setsockopt(zmq.LINGER, 0)
                command.setsockopt(zmq.IMMEDIATE, 1)
                command.setsockopt(zmq.SNDTIMEO, 100)
                command.setsockopt(zmq.SNDHWM, 1)
                command.connect(getattr(args, f"{side}_command_endpoint"))
                state = self.context.socket(zmq.SUB)
                self.states[side] = state
                state.setsockopt(zmq.LINGER, 0)
                state.setsockopt(zmq.SUBSCRIBE, b"")
                state.setsockopt(zmq.CONFLATE, 1)
                state.connect(getattr(args, f"{side}_state_endpoint"))
                self.poller.register(state, zmq.POLLIN)
        except BaseException:
            self.close()
            raise

    def poll(self, timeout_ms=20):
        ready = dict(self.poller.poll(timeout_ms))
        return {
            side: msgpack.unpackb(socket.recv(), raw=False)
            for side, socket in self.states.items()
            if socket in ready
        }

    def send(self, side, payload):
        self.commands[side].send(msgpack.packb(payload, use_bin_type=True))

    def close(self):
        for socket in (*self.commands.values(), *self.states.values()):
            socket.close()
        self.context.term()


class OperatorSession:
    def __init__(self, args, *, io=None, clock=time.monotonic):
        self.io = GatewayIO(args) if io is None else io
        self.clock = clock
        self.timeout = args.stop_timeout
        self.maximum_age = args.state_timeout
        self.rows = {}
        self.instances = {}
        self.attempted = set()
        self.eid = None
        self.faulted = False

    def _poll(self, timeout_ms=20):
        for side, payload in self.io.poll(timeout_ms).items():
            InspireHandState.from_wire_dict(payload)
            meta = payload.get("operator_session")
            if not isinstance(meta, dict) or meta.get("enabled") is not True:
                raise OperatorSessionError(
                    f"{side} Gateway needs the updated binary with "
                    "G1_RL_SESSION_CONTROL=1; no automatic fallback"
                )
            if (
                not isinstance(meta.get("instance_id"), str)
                or not meta["instance_id"]
                or type(meta.get("generation")) is not int
                or meta["generation"] < 0
                or any(type(meta.get(key)) is not bool for key in ("ok", "active", "fault_latched"))
                or any(
                    not isinstance(meta.get(key), str)
                    for key in ("request_id", "session_id", "error")
                )
            ):
                raise OperatorSessionError(f"invalid {side} operator receipt")
            previous = self.rows.get(side)
            if self.instances.setdefault(side, meta["instance_id"]) != meta["instance_id"]:
                raise OperatorSessionError(f"{side} Gateway restarted during the session")
            if previous and payload["sequence"] <= previous[0]["sequence"]:
                if payload["sequence"] < previous[0]["sequence"]:
                    raise OperatorSessionError(f"{side} state sequence regressed")
                continue
            self.faulted |= meta["fault_latched"] or payload["status"] in ("FAULT", "STALE")
            self.rows[side] = payload, self.clock()

    def _wait(self, predicate, description, *, required=("hand",)):
        deadline = self.clock() + self.timeout
        while self.clock() < deadline:
            self._poll()
            if all(
                side in self.rows and self.clock() - self.rows[side][1] <= self.maximum_age
                for side in required
            ):
                result = predicate({side: row[0] for side, row in self.rows.items()})
                if result:
                    return result
        raise OperatorSessionError(
            f"timed out waiting for {description}; stop and inspect Gateways"
        )

    def ready(self):
        def check(rows):
            if self.faulted:
                raise OperatorSessionError("Gateway fault observed; manual recovery required")
            if any(
                row["status"] != "DISARMED" or row["operator_session"]["active"]
                for row in rows.values()
            ):
                raise OperatorSessionError(
                    "hand Gateway must be DISARMED before next; "
                    "do not ARM them manually in session mode"
                )
            return rows

        return self._wait(check, "idle hand Gateway")

    def _request(self, side, operation):
        if operation == "DISARM":
            # Always attempt revocation on the reachable side. Do not wait for
            # its peer (or another fresh packet) before sending. The Gateway
            # accepts an old generation only for DISARM of this exact owner.
            if side not in self.rows:
                raise OperatorSessionError(f"no known {side} Gateway instance to revoke")
            meta = self.rows[side][0]["operator_session"]
        else:
            rows = self._wait(lambda rows: rows, "fresh Gateway state")
            meta = rows[side]["operator_session"]
        request_id = uuid.uuid4().hex
        payload = dict(
            schema="g1.operator.command.v1",
            operation=operation,
            request_id=request_id,
            session_id=self.eid,
            instance_id=meta["instance_id"],
            generation=meta["generation"],
        )
        self.io.send(side, payload)

        def acknowledged(rows):
            row = rows[side]
            ack = row["operator_session"]
            if ack["request_id"] != request_id:
                return False
            if not ack["ok"] or ack["session_id"] != self.eid:
                raise OperatorSessionError(f"{side} {operation} rejected: {ack['error']}")
            if (
                ack["generation"] <= meta["generation"]
                or operation == "PREPARE"
                and ack["generation"] != meta["generation"] + 1
            ):
                raise OperatorSessionError(f"{side} operator generation did not advance correctly")
            expected = "ARMED" if operation == "PREPARE" else "DISARMED"
            if row["status"] != expected or ack["active"] != (operation == "PREPARE"):
                raise OperatorSessionError(f"{side} {operation} did not reach {expected}")
            if operation == "PREPARE" and (
                self.faulted or row.get("accepted_command_sequence") is not None
            ):
                raise OperatorSessionError(
                    f"{side} prepare did not create a clean, healthy session"
                )
            return ack

        return self._wait(
            acknowledged,
            f"{side} {operation} acknowledgement",
            required=(side,) if operation == "DISARM" else ("hand",),
        )

    def prepare(self, eid):
        self.ready()
        self.eid = eid
        self.attempted.clear()
        for side in ("hand",):
            self.attempted.add(side)  # Include a request whose acknowledgement is lost.
            self._request(side, "PREPARE")

    def finish(self):
        errors = []
        for side in ("hand",):
            if side in self.attempted:
                try:
                    self._request(side, "DISARM")
                except BaseException as exc:
                    errors.append(f"{side}: {exc}")
        if errors:
            self.faulted = True
            raise OperatorSessionError(
                "DISARM could not be verified; manually stop and inspect. " + "; ".join(errors)
            )
        self.attempted.clear()
        self.ready()

    def close(self):
        self.io.close()
