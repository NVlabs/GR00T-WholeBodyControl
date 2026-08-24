import threading
from types import SimpleNamespace

import pytest

from gear_sonic.scripts import pico_g1_preflight as preflight


def _validated_args(*arguments: str):
    parser = preflight.build_parser()
    return preflight._validate_cli_args(parser, parser.parse_args(arguments))


def _lowstate_message(
    *,
    tick: int = 1,
    motor_slots: int = preflight.HG_LOWSTATE_MOTOR_SLOTS,
    motor_q: float = 0.0,
    quaternion=(1.0, 0.0, 0.0, 0.0),
    gyroscope=(0.0, 0.0, 0.0),
):
    motor_states = [SimpleNamespace(q=motor_q, dq=0.0, tau_est=0.0) for _ in range(motor_slots)]
    return SimpleNamespace(
        tick=tick,
        mode_machine=5,
        motor_state=motor_states,
        imu_state=SimpleNamespace(
            quaternion=quaternion,
            gyroscope=gyroscope,
        ),
    )


def _sample(arrival_s: float, tick: int, mode_machine: int = 5) -> preflight._LowStateSample:
    return preflight._LowStateSample(
        arrival_s=arrival_s,
        tick=tick,
        mode_machine=mode_machine,
    )


def _valid_pico_health_snapshot(**overrides):
    snapshot = {
        "health_supported": True,
        "health_available": True,
        "health_valid": True,
        "health_schema_version": 1,
        "health_sample_sequence": 11,
        "health_timestamp_ns": 11_000_000,
        "health_calibration_result": 0,
        "health_calibrated": True,
        "health_tracking_mode": 0,
        "health_connect_state_result": 0,
        "health_tracker_count": 2,
        "health_unique_tracker_count": 2,
        "health_body_state_result": 0,
        "health_is_tracking": True,
        "health_tracking_state_code": 0,
        "health_body_state_code": 1,
        "health_body_error_code": 0,
        "health_connected_band_count": 2,
        "health_body_data_result": 0,
        "health_body_role_count": 24,
    }
    snapshot.update(overrides)
    return snapshot


def _valid_pico_controller_health_snapshot(**overrides):
    snapshot = {
        "health_supported": True,
        "health_available": True,
        "health_valid": True,
        "health_schema_version": 1,
        "health_sample_sequence": 11,
        "health_timestamp_ns": 11_000_000,
    }
    for side in ("left", "right"):
        snapshot.update(
            {
                f"health_{side}_device_valid": True,
                f"health_{side}_is_tracked_available": True,
                f"health_{side}_is_tracked": True,
                f"health_{side}_tracking_state_available": True,
                f"health_{side}_tracking_state": 3,
                f"health_{side}_valid": True,
            }
        )
    snapshot.update(overrides)
    return snapshot


def test_assess_pico_tracker_health_accepts_advancing_authoritative_state():
    passed, detail = preflight._assess_pico_tracker_health(
        _valid_pico_health_snapshot(),
        previous_sequence=10,
    )

    assert passed
    assert "BT_VALID" in detail
    assert "2 connected" in detail


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"health_supported": False}, "protocol missing"),
        ({"health_available": False}, "unavailable"),
        ({"health_sample_sequence": 10}, "did not advance"),
        ({"health_calibrated": False}, "not calibrated"),
        ({"health_tracker_count": 1}, "at least two"),
        ({"health_unique_tracker_count": 1}, "at least two"),
        ({"health_is_tracking": False}, "tracking lost"),
        ({"health_body_state_code": 2}, "not BT_VALID"),
        ({"health_connected_band_count": 1}, "only 1 connected"),
        ({"health_body_data_result": 1}, "data query failed"),
        ({"health_body_role_count": 23}, "expected 24"),
    ],
)
def test_assess_pico_tracker_health_fails_closed(overrides, reason_fragment):
    passed, detail = preflight._assess_pico_tracker_health(
        _valid_pico_health_snapshot(**overrides),
        previous_sequence=10,
    )

    assert not passed
    assert reason_fragment in detail


def test_assess_pico_controller_health_accepts_authoritative_tracking_state():
    passed, detail = preflight._assess_pico_controller_tracking_health(
        _valid_pico_controller_health_snapshot(),
        previous_sequence=10,
    )

    assert passed
    assert "position+rotation" in detail


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"health_supported": False}, "protocol missing"),
        ({"health_available": False}, "unavailable"),
        ({"health_sample_sequence": 10}, "did not advance"),
        ({"health_left_device_valid": False}, "device is invalid"),
        ({"health_left_is_tracked": False}, "tracking lost"),
        ({"health_right_tracking_state": 1}, "position+rotation"),
        ({"health_right_valid": False}, "health gate reports invalid"),
    ],
)
def test_assess_pico_controller_health_fails_closed(overrides, reason_fragment):
    passed, detail = preflight._assess_pico_controller_tracking_health(
        _valid_pico_controller_health_snapshot(**overrides),
        previous_sequence=10,
    )

    assert not passed
    assert reason_fragment in detail


def test_wait_for_pico_body_snapshot_tolerates_hardened_client_cold_start(monkeypatch):
    snapshots = iter(
        (
            {"available": False, "health_supported": False},
            {"available": False, "health_supported": False},
            {
                "available": True,
                **_valid_pico_health_snapshot(),
            },
        )
    )
    clock = iter((0.0, 0.1, 0.2, 0.3))

    class _FakeXrt:
        @staticmethod
        def get_body_snapshot():
            return next(snapshots)

    monkeypatch.setattr(preflight.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(preflight.time, "sleep", lambda _seconds: None)

    snapshot = preflight._wait_for_pico_body_snapshot(_FakeXrt(), 1.0)

    assert snapshot["available"] is True
    assert snapshot["health_supported"] is True


def test_validate_lowstate_requires_exact_hg_layout_and_finite_active_state():
    sample = preflight._validate_lowstate_message(_lowstate_message(), arrival_s=10.0)

    assert sample == preflight._LowStateSample(arrival_s=10.0, tick=1, mode_machine=5)

    with pytest.raises(ValueError, match="exact HG 35-slot representation"):
        preflight._validate_lowstate_message(_lowstate_message(motor_slots=29), arrival_s=10.0)
    with pytest.raises(ValueError, match=r"motor_state\[0\]\.q is non-finite"):
        preflight._validate_lowstate_message(_lowstate_message(motor_q=float("nan")), arrival_s=10.0)


def test_validate_lowstate_rejects_implausible_motor_values_and_arrival_time():
    message = _lowstate_message()
    message.motor_state[0].dq = 101.0
    with pytest.raises(ValueError, match=r"motor_state\[0\]\.dq magnitude 101 exceeds 100"):
        preflight._validate_lowstate_message(message, arrival_s=10.0)

    with pytest.raises(ValueError, match="arrival time must be finite and nonnegative"):
        preflight._validate_lowstate_message(_lowstate_message(), arrival_s=float("nan"))


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("quaternion", (float("inf"), 0.0, 0.0, 0.0), "imu_state.quaternion[0] is non-finite"),
        ("quaternion", (0.0, 0.0, 0.0, 0.0), "imu_state.quaternion norm 0 outside"),
        ("gyroscope", (0.0, float("nan"), 0.0), "imu_state.gyroscope[1] is non-finite"),
    ],
)
def test_validate_lowstate_rejects_nonfinite_imu(field, value, expected_error):
    kwargs = {field: value}

    with pytest.raises(ValueError, match=expected_error.replace("[", r"\[").replace("]", r"\]")):
        preflight._validate_lowstate_message(_lowstate_message(**kwargs), arrival_s=10.0)


def test_assess_lowstate_accepts_fresh_advancing_ticks_across_uint32_wrap():
    passed, detail = preflight._assess_lowstate_samples(
        [
            _sample(10.0, 0xFFFFFFFE),
            _sample(10.1, 0xFFFFFFFF),
            _sample(10.2, 0),
            _sample(10.3, 1),
        ],
        invalid_count=0,
        invalid_details=[],
        interface="eth-test",
        observation_start_s=9.8,
        observation_end_s=10.35,
    )

    assert passed
    assert "4 advancing samples from 4 callback(s)" in detail
    assert "rate=10.0Hz" in detail
    assert "duplicates=0" in detail
    assert "35 slots/29 body-index slots finite" in detail
    assert "IMU quaternion=4, gyro=3 finite" in detail


def test_assess_lowstate_accepts_duplicate_callbacks_without_refreshing_source_tick():
    passed, detail = preflight._assess_lowstate_samples(
        [
            _sample(10.0, 10),
            _sample(10.1, 11),
            _sample(10.2, 11),
            _sample(10.3, 12),
            _sample(10.4, 13),
        ],
        invalid_count=0,
        invalid_details=[],
        interface="eth-test",
        observation_start_s=9.9,
        observation_end_s=10.45,
    )

    assert passed
    assert "4 advancing samples from 5 callback(s)" in detail
    assert "duplicates=1" in detail


def test_assess_lowstate_rejects_regressing_tick():
    passed, detail = preflight._assess_lowstate_samples(
        [_sample(10.0 + index * 0.1, tick) for index, tick in enumerate([10, 11, 9, 12])],
        invalid_count=0,
        invalid_details=[],
        interface="eth-test",
        observation_start_s=9.9,
        observation_end_s=10.4,
    )

    assert not passed
    assert "regressing rt/lowstate tick 11->9" in detail


def test_assess_lowstate_rejects_live_23dof_embodiment_even_if_cli_claims_29dof():
    passed, detail = preflight._assess_lowstate_samples(
        [_sample(10.0 + index * 0.1, index, mode_machine=4) for index in range(4)],
        invalid_count=0,
        invalid_details=[],
        interface="eth-test",
        observation_start_s=9.9,
        observation_end_s=10.4,
    )

    assert not passed
    assert "detected 23-DoF rev1 (mode_machine=4)" in detail
    assert "released SONIC requires a full 29-DoF G1" in detail


def test_assess_lowstate_rejects_embodiment_change_during_probe():
    passed, detail = preflight._assess_lowstate_samples(
        [
            _sample(10.0, 1, mode_machine=5),
            _sample(10.1, 2, mode_machine=5),
            _sample(10.2, 3, mode_machine=4),
            _sample(10.3, 4, mode_machine=4),
        ],
        invalid_count=0,
        invalid_details=[],
        interface="eth-test",
        observation_start_s=9.9,
        observation_end_s=10.4,
    )

    assert not passed
    assert "mode_machine changed during probe: 4, 5" in detail


def test_assess_lowstate_duplicates_cannot_hide_stalled_source():
    passed, detail = preflight._assess_lowstate_samples(
        [
            _sample(10.0, 10),
            _sample(10.1, 11),
            _sample(10.2, 12),
            _sample(10.3, 13),
            _sample(10.6, 13),
            _sample(10.9, 13),
        ],
        invalid_count=0,
        invalid_details=[],
        interface="eth-test",
        observation_start_s=9.9,
        observation_end_s=10.95,
    )

    assert not passed
    assert "latest sample older than 0.5s" in detail


def test_assess_lowstate_rejects_invalid_callback_even_with_valid_samples():
    passed, detail = preflight._assess_lowstate_samples(
        [_sample(10.0 + index * 0.1, index) for index in range(4)],
        invalid_count=1,
        invalid_details=["imu_state.gyroscope[1] is non-finite"],
        interface="eth-test",
        observation_start_s=9.9,
        observation_end_s=10.4,
    )

    assert not passed
    assert "1/5 invalid" in detail
    assert "gyroscope" in detail


@pytest.mark.parametrize(
    ("arrival_times", "observation_end_s", "expected_error"),
    [
        ([10.0, 10.3, 10.6, 10.9], 10.95, "rate below 5.0Hz"),
        ([10.0, 10.1, 10.2, 10.3], 11.0, "latest sample older than 0.5s"),
    ],
)
def test_assess_lowstate_rejects_slow_or_stale_stream(
    arrival_times,
    observation_end_s,
    expected_error,
):
    passed, detail = preflight._assess_lowstate_samples(
        [_sample(arrival_s, index) for index, arrival_s in enumerate(arrival_times)],
        invalid_count=0,
        invalid_details=[],
        interface="eth-test",
        observation_start_s=9.9,
        observation_end_s=observation_end_s,
    )

    assert not passed
    assert expected_error in detail


def test_probe_lowstate_closes_subscriber_when_init_raises(monkeypatch):
    closed = False

    class FakeSubscriber:
        def __init__(self, topic, message_type):
            assert topic == "rt/lowstate"
            assert message_type is object

        def Init(self, callback, queue_len):
            assert callable(callback)
            assert queue_len == 10
            raise RuntimeError("subscribe failed")

        def Close(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(
        preflight,
        "_load_lowstate_api",
        lambda: (lambda domain, interface: None, FakeSubscriber, object),
    )
    checks = preflight.Checks()

    preflight._probe_lowstate(checks, "eth-test", 0.01)

    assert closed
    assert len(checks.results) == 1
    assert checks.results[0].status == "FAIL"
    assert "DDS probe failed on eth-test: subscribe failed" in checks.results[0].detail


def test_probe_lowstate_observes_full_window_after_subscriber_init(monkeypatch):
    clock_s = 100.0
    next_tick = 1
    subscriber = None

    class FakeSubscriber:
        def __init__(self, topic, message_type):
            nonlocal subscriber
            assert topic == "rt/lowstate"
            assert message_type is object
            self.callback = None
            self.closed = False
            subscriber = self

        def Init(self, callback, queue_len):
            nonlocal clock_s
            assert queue_len == 10
            clock_s += 2.0
            self.callback = callback

        def Close(self):
            self.closed = True

    def monotonic():
        return clock_s

    def sleep(duration_s):
        nonlocal clock_s, next_tick
        clock_s += duration_s
        subscriber.callback(_lowstate_message(tick=next_tick))
        next_tick += 1

    monkeypatch.setattr(
        preflight,
        "_load_lowstate_api",
        lambda: (lambda domain, interface: None, FakeSubscriber, object),
    )
    monkeypatch.setattr(preflight.time, "monotonic", monotonic)
    monkeypatch.setattr(preflight.time, "sleep", sleep)
    checks = preflight.Checks()

    preflight._probe_lowstate(checks, "eth-test", 0.4)

    assert subscriber.closed
    assert len(checks.results) == 1
    assert checks.results[0].status == "PASS"
    assert "in 0.40s" in checks.results[0].detail


def test_pico_only_implies_live_pico_without_full_live_gates():
    args = _validated_args("--robot-dof", "29", "--pico-only")

    assert args.pico_only
    assert args.live_pico
    assert args.robot_interface is None
    assert not args.probe_lowstate


@pytest.mark.parametrize(
    ("conflicting_args", "expected_flag"),
    [
        (("--offline",), "--offline"),
        (("--robot-interface", "eth0"), "--robot-interface"),
        (("--probe-lowstate",), "--probe-lowstate"),
    ],
)
def test_pico_only_rejects_g1_and_offline_conflicts(
    conflicting_args,
    expected_flag,
    capsys,
):
    with pytest.raises(SystemExit) as exc_info:
        _validated_args("--robot-dof", "29", "--pico-only", *conflicting_args)

    assert exc_info.value.code == 2
    assert f"--pico-only cannot be combined with {expected_flag}" in capsys.readouterr().err


def test_offline_mode_remains_explicit_and_needs_no_live_gates():
    args = _validated_args("--robot-dof", "29", "--offline")

    assert args.offline
    assert not args.pico_only
    assert not args.live_pico
    assert not args.probe_lowstate


def test_default_live_preflight_still_requires_all_live_gates():
    with pytest.raises(SystemExit) as exc_info:
        _validated_args("--robot-dof", "29")

    assert exc_info.value.code == 2


def test_pico_only_routes_only_pico_probe_and_marks_not_robot_ready(monkeypatch):
    calls = []
    args = _validated_args(
        "--robot-dof",
        "29",
        "--pico-only",
        "--pico-timeout",
        "7.5",
    )
    monkeypatch.setattr(
        preflight,
        "_probe_lowstate",
        lambda checks, interface, duration_s: calls.append(("lowstate", interface, duration_s)),
    )
    monkeypatch.setattr(
        preflight,
        "_probe_pico",
        lambda checks, timeout_s: calls.append(("pico", timeout_s)),
    )
    checks = preflight.Checks()

    preflight._run_requested_live_probes(checks, args, interface=None)

    assert calls == [("pico", 7.5)]
    assert checks.results == [
        preflight.Result(
            name="PICO-only scope",
            status="WARN",
            detail=(
                "live PICO full-body diagnostic only; G1 robot interface and Unitree "
                "LowState intentionally skipped; no robot commands sent; result is not "
                "G1/robot-ready"
            ),
        )
    ]


def test_full_live_routes_lowstate_then_pico_without_scope_warning(monkeypatch):
    calls = []
    args = _validated_args(
        "--robot-dof",
        "29",
        "--robot-interface",
        "eth0",
        "--probe-lowstate",
        "--live-pico",
    )
    monkeypatch.setattr(
        preflight,
        "_probe_lowstate",
        lambda checks, interface, duration_s: calls.append(("lowstate", interface, duration_s)),
    )
    monkeypatch.setattr(
        preflight,
        "_probe_pico",
        lambda checks, timeout_s: calls.append(("pico", timeout_s)),
    )
    checks = preflight.Checks()

    preflight._run_requested_live_probes(checks, args, interface="eth0")

    assert calls == [("lowstate", "eth0", 3.0), ("pico", 10.0)]
    assert checks.results == []


def _true23_args(*extra: str):
    return _validated_args(
        "--policy-profile",
        "true23",
        "--robot-dof",
        "23",
        "--checkpoint",
        "trained.pt",
        "--simulation-report",
        "sim.json",
        "--encoder-onnx",
        "encoder.onnx",
        "--decoder-onnx",
        "decoder.onnx",
        "--metadata",
        "pair.metadata.json",
        *extra,
    )


def test_true23_cli_requires_exact_profile_dof_and_all_artifact_paths(capsys):
    with pytest.raises(SystemExit):
        _validated_args(
            "--policy-profile",
            "true23",
            "--robot-dof",
            "29",
            "--offline",
        )
    assert "requires --robot-dof 23" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        _validated_args(
            "--policy-profile",
            "true23",
            "--robot-dof",
            "23",
            "--offline",
        )
    assert "requires --checkpoint" in capsys.readouterr().err

    args = _true23_args("--offline")
    assert args.policy_profile == "true23"
    assert args.robot_dof == 23


def test_true23_lowstate_accepts_only_mode4_and_validates_required_slots():
    samples = [
        _sample(10.0 + index * 0.1, index + 1, mode_machine=4)
        for index in range(4)
    ]
    passed, detail = preflight._assess_lowstate_samples(
        samples,
        invalid_count=0,
        invalid_details=[],
        interface="eth-test",
        observation_start_s=9.9,
        observation_end_s=10.4,
        policy_profile="true23",
    )
    assert passed
    assert "mode_machine=4" in detail
    assert "23 required body-index slots finite" in detail

    wrong_mode = [
        _sample(10.0 + index * 0.1, index + 1, mode_machine=5)
        for index in range(4)
    ]
    passed, detail = preflight._assess_lowstate_samples(
        wrong_mode,
        invalid_count=0,
        invalid_details=[],
        interface="eth-test",
        observation_start_s=9.9,
        observation_end_s=10.4,
        policy_profile="true23",
    )
    assert not passed
    assert "requires 23-DoF rev1 mode_machine=4" in detail

    message = _lowstate_message()
    message.mode_machine = 4
    message.motor_state[13].q = float("nan")
    message.motor_state[14].q = float("nan")
    preflight._validate_lowstate_message(
        message,
        arrival_s=10.0,
        active_motor_ids=preflight.TRUE23_HARDWARE_JOINT_IDS,
    )
    with pytest.raises(ValueError, match=r"motor_state\[13\]\.q is non-finite"):
        preflight._validate_lowstate_message(message, arrival_s=10.0)


def test_true23_artifacts_use_public_verifier_and_remain_integrated_no_go(
    monkeypatch,
):
    from gear_sonic.utils import g1_23dof_artifact

    calls = []

    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return {"training_evidence": {"global_step": 123}}

    monkeypatch.setattr(
        g1_23dof_artifact,
        "verify_validated_true23_artifact",
        verify,
    )
    args = _true23_args("--offline")
    checks = preflight.Checks()
    preflight._check_policy_artifacts(checks, args, preflight.Path("deploy"))

    assert len(calls) == 1
    assert [result.status for result in checks.results] == ["PASS", "FAIL"]
    assert checks.results[0].name == "True23 trained artifact pair"
    assert "global_step=123" in checks.results[0].detail
    assert checks.results[1].name == "True23 integrated live inference"
    assert "robot commands remain prohibited" in checks.results[1].detail


def test_full_live_true23_probes_use_overlapping_windows(monkeypatch):
    barrier = threading.Barrier(2, timeout=1.0)
    calls = []

    def lowstate(checks, interface, duration_s, *, policy_profile):
        barrier.wait()
        calls.append(("lowstate", interface, duration_s, policy_profile))

    def pico(checks, timeout_s):
        barrier.wait()
        calls.append(("pico", timeout_s))

    monkeypatch.setattr(preflight, "_probe_lowstate", lowstate)
    monkeypatch.setattr(preflight, "_probe_pico", pico)
    args = _true23_args(
        "--robot-interface",
        "eth0",
        "--probe-lowstate",
        "--live-pico",
    )
    checks = preflight.Checks()

    preflight._run_requested_live_probes(checks, args, interface="eth0")

    assert {call[0] for call in calls} == {"lowstate", "pico"}
    assert ("lowstate", "eth0", 3.0, "true23") in calls
    assert checks.results == []
