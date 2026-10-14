"""Lifecycle, hand physics and adapter failure checks without robot devices."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.hand_control.inspire.client import InspireRealBackend
from gear_sonic.utils.hand_control.inspire.contract import ContractError, HandStatus
from gear_sonic.utils.hand_control.inspire.session import OperatorSession, OperatorSessionError
from gear_sonic.utils.mujoco_sim.inspire.backend import InspireSimBackend


def test_sim_moves_reads_feedback_and_preserves_body_controls():
    hand = InspireSimBackend()
    assert hand.model is None
    hand.connect()
    try:
        start = hand.read_state()
        assert start.error is None
        left = np.maximum(0, start.left_position)
        right = np.maximum(0, start.right_position)
        hand.set_target(left, right)
        for i in range(1, 31):
            time.sleep(0.02)
            target = left.copy()
            target[3] += i * 0.008
            hand.set_target(target, right)
        state = hand.read_state()
        assert np.isfinite(state.left_position).all()
        assert state.left_position[3] > start.left_position[3] + 0.015
        # Feedback is an actual qpos sample, not the last command target.
        with hand._lock:
            expected = hand.controller.layout.read(hand.data)[0]
            sampled = hand.data.qpos[hand.controller.layout.left.qpos_addresses].copy()
            np.testing.assert_allclose(expected, sampled)
            body = [
                i
                for i in range(hand.model.nu)
                if i
                not in set(
                    hand.controller.layout.left.actuator_ids.tolist()
                    + hand.controller.layout.right.actuator_ids.tolist()
                )
            ]
            hand.data.ctrl[body] = 0.123
            hand.controller.step(now_ns=time.monotonic_ns())
            np.testing.assert_allclose(hand.data.ctrl[body], 0.123)
            hand.data.ctrl[body] = 0
        time.sleep(0.25)
        assert "stale" in hand.read_state().error
        with hand._lock:
            for side in (hand.controller.layout.left, hand.controller.layout.right):
                assert np.all(hand.data.ctrl[side.actuator_ids] == 0)
        with pytest.raises(ContractError):
            hand.set_target(left, right)
        hand.stop()
        hand.stop()
        with pytest.raises(RuntimeError):
            hand.connect()
    finally:
        hand.close()
        hand.close()
    with pytest.raises(RuntimeError):
        hand.read_state()


@pytest.mark.parametrize("bad", [[0] * 5, [float("nan")] * 6, [-1] * 6, [2] * 6])
def test_sim_rejects_both_hands_before_submission(bad):
    hand = InspireSimBackend()
    hand.connect()
    try:
        with pytest.raises(ValueError):
            hand.set_target([0] * 6, bad)
        assert hand.controller.guard.last_command is None
        with pytest.raises(RuntimeError):
            hand.set_target([0] * 6, [0] * 6)
    finally:
        hand.close()


def test_sim_rejects_first_jump_and_slew():
    hand = InspireSimBackend()
    hand.connect()
    try:
        with pytest.raises(ContractError, match="first target"):
            hand.set_target([0.4] * 6, [0] * 6)
    finally:
        hand.close()
    hand.connect()
    try:
        hand.set_target([0] * 6, [0] * 6)
        with pytest.raises(ContractError, match="slew"):
            hand.set_target([0.5] * 6, [0] * 6)
    finally:
        hand.close()


def test_real_close_cleans_resources_when_disarm_fails():
    hand = InspireRealBackend()
    closed = []

    def fail():
        raise RuntimeError("DISARM timeout")

    hand._session = SimpleNamespace(
        attempted={"hand"}, finish=fail, close=lambda: closed.append(True)
    )
    with pytest.raises(RuntimeError, match="DISARM timeout"):
        hand.close()
    assert closed and hand._session is None and hand._stopped
    hand.close()


def test_real_connect_failure_releases_socket_owner(monkeypatch):
    closed = []

    def fail():
        raise RuntimeError("gateway is not ready")

    monkeypatch.setattr(
        "gear_sonic.utils.hand_control.inspire.session.OperatorSession",
        lambda args: SimpleNamespace(ready=fail, close=lambda: closed.append(True)),
    )
    hand = InspireRealBackend()
    with pytest.raises(RuntimeError, match="not ready"):
        hand.connect()
    assert hand._session is None and closed


class ScriptedIO:
    def __init__(self, rows):
        self.rows = rows

    def poll(self, timeout_ms=20):
        return {"hand": self.rows.pop(0)} if self.rows else {}


def state(sequence=1, instance="a", **meta):
    from gear_sonic.utils.hand_control.inspire.contract import InspireHandState

    row = InspireHandState(
        sequence, 0, 0, HandStatus.DISARMED, np.zeros(6), np.zeros(6)
    ).to_wire_dict()
    row["operator_session"] = dict(
        enabled=True,
        instance_id=instance,
        generation=0,
        request_id="",
        session_id="",
        ok=False,
        active=False,
        fault_latched=False,
        error="",
        **meta,
    )
    return row


def test_session_rejects_restart_regression_and_duplicate_freshness():
    for changed, message in [(state(2, "b"), "restarted"), (state(0), "regressed")]:
        now = [10.0]
        session = OperatorSession(
            SimpleNamespace(stop_timeout=0.1, state_timeout=0.1),
            io=ScriptedIO([state(), state(), changed]),
            clock=lambda: now[0],
        )
        session._poll()
        now[0] = 11
        session._poll()
        assert session.rows["hand"][1] == 10.0
        with pytest.raises(OperatorSessionError, match=message):
            session._poll()


@pytest.mark.parametrize("defect", ["request", "session", "generation", "status"])
def test_operator_prepare_requires_matching_receipt(defect):
    class ReplyIO(ScriptedIO):
        def send(self, side, request):
            reply = state(2)
            reply["status"] = "ARMED"
            reply["operator_session"].update(
                request_id=request["request_id"],
                session_id=request["session_id"],
                generation=1,
                ok=True,
                active=True,
            )
            if defect == "request":
                reply["operator_session"]["request_id"] = "other"
            elif defect == "session":
                reply["operator_session"]["session_id"] = "other"
            elif defect == "generation":
                reply["operator_session"]["generation"] = 0
            else:
                reply["status"] = "DISARMED"
            self.rows.append(reply)

    session = OperatorSession(
        SimpleNamespace(stop_timeout=0.03, state_timeout=0.1), io=ReplyIO([state()])
    )
    session.eid = "a" * 32
    with pytest.raises(OperatorSessionError):
        session._request("hand", "PREPARE")


def test_prepare_failure_revokes_attempted_session_without_sending_target(monkeypatch):
    from gear_sonic.utils.hand_control.interface import HandState

    hand = InspireRealBackend()
    stopped = []

    def failed_prepare(eid):
        hand._session.attempted.add("hand")
        raise RuntimeError("prepare acknowledgement lost")

    hand._session = SimpleNamespace(
        attempted=set(), prepare=failed_prepare, finish=lambda: stopped.append(True)
    )
    monkeypatch.setattr(hand, "read_state", lambda: HandState(tuple([0.0] * 6), tuple([0.0] * 6)))
    with pytest.raises(RuntimeError, match="prepare acknowledgement lost"):
        hand.set_target([0] * 6, [0] * 6)
    assert hand._stopped and stopped


def test_real_stale_feedback_is_not_presented_as_current_measurement():
    hand = InspireRealBackend()
    hand._session = SimpleNamespace(
        _poll=lambda **kw: None, rows={"hand": (state(), time.monotonic() - 1)}
    )
    feedback = hand.read_state()
    assert feedback.left_position is None and feedback.right_position is None
    assert feedback.error == "stale measured feedback"


def test_sequence_records_targets_and_measured_feedback(monkeypatch):
    from types import SimpleNamespace

    from gear_sonic.scripts import inspire_hand_example as example
    from gear_sonic.utils.hand_control.interface import HandState

    clock = [1.0]
    monkeypatch.setattr(example.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(example.time, "monotonic_ns", lambda: round(clock[0] * 1e9))
    monkeypatch.setattr(example.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    targets = []

    def measured():
        q = np.asarray(targets[-1]) * 0.5 if targets else np.zeros((2, 6))
        return HandState(tuple(q[0]), tuple(q[1]), round(clock[0] * 1e9))

    hand = SimpleNamespace(
        settings={"command_rate_hz": 50},
        read_state=measured,
        set_target=lambda left, right: targets.append([list(left), list(right)]),
    )
    events = []
    stages = example.run_finger_sequence(
        hand, check=lambda: None, sample=lambda *args: None, log=events.append
    )
    assert len(stages) == 20
    assert [stage["action"] for stage in stages] == ["close", "open"] * 10
    np.testing.assert_allclose(stages[-1]["target"], np.zeros((2, 6)))
    rows = [row for row in events if row["event"] in ("seed", "sample")]
    assert len(rows) == len(targets)
    assert [row["command_index"] for row in rows] == list(range(len(rows)))
    np.testing.assert_allclose([row["target"] for row in rows], targets)
    # Final open targets are zero; check a nonzero sample to distinguish feedback.
    moving = next(row for row in rows if np.any(row["target"]))
    assert moving["target"] != moving["measured"]  # Never turn desired position into feedback.
    assert [r["stage_index"] for r in events if r["event"] == "stage_complete"] == list(range(20))
    assert all(
        isinstance(r["submitted_monotonic_ns"], int) and r["feedback_received_monotonic_ns"] > 0
        for r in rows
    )


def test_sequence_cli_closes_on_error(monkeypatch):
    import sys
    from types import SimpleNamespace

    from gear_sonic.scripts import inspire_hand_example as example
    from gear_sonic.utils.hand_control.interface import HandState
    from gear_sonic.utils.mujoco_sim.inspire import backend as sim

    closed = []
    hand = SimpleNamespace(
        connect=lambda: None,
        description="test",
        read_state=lambda: HandState((0.0,) * 6, (0.0,) * 6, 1),
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(sim, "create_backend", lambda: hand)

    def fail(*args, **kwargs):
        raise RuntimeError("sequence failure")

    monkeypatch.setattr(example, "run_finger_sequence", fail)
    monkeypatch.setattr(sys, "argv", ["example", "--sequence"])
    with pytest.raises(RuntimeError, match="sequence failure"):
        example.main()
    assert closed == [True]
