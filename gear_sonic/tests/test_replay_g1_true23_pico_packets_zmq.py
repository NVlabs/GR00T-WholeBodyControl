from __future__ import annotations

import copy
from pathlib import Path
import time

import pytest

from gear_sonic.scripts.replay_g1_true23_pico_packets_zmq import (
    CONTROL_PERIOD_NS,
    load_reference_packets,
    rebase_reference_packet_time,
)
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    run_balanced_upper_body_reference_sequence,
    validate_reference_terms,
)

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / ("artifacts/g1_true23/pico_saved_clip_replay_v1/upright/causal_packets_neutral_calibrated_v1.json")


def test_real_pico_packets_rebase_only_timestamps() -> None:
    packets = load_reference_packets(BUNDLE)
    first = validate_reference_terms(packets[0])
    rebased = rebase_reference_packet_time(
        packets[3],
        first_control_index=first["control_index"],
        first_control_monotonic_ns=10_000_000_000,
    )
    assert rebased["control_monotonic_ns"] == 10_000_000_000 + 3 * CONTROL_PERIOD_NS
    assert rebased["pico_anchor_monotonic_ns"] == rebased["control_monotonic_ns"] - CONTROL_PERIOD_NS
    original_without_time = copy.deepcopy(packets[3])
    rebased_without_time = copy.deepcopy(rebased)
    for key in ("pico_anchor_monotonic_ns", "control_monotonic_ns"):
        original_without_time.pop(key)
        rebased_without_time.pop(key)
    assert rebased_without_time == original_without_time


def test_rebase_rejects_packet_before_origin() -> None:
    packet = load_reference_packets(BUNDLE)[0]
    summary = validate_reference_terms(packet)
    with pytest.raises(ValueError, match="precedes"):
        rebase_reference_packet_time(
            packet,
            first_control_index=summary["control_index"] + 1,
            first_control_monotonic_ns=10_000_000_000,
        )


class _FakeBalancedController:
    arm_blend = 0.4

    def __init__(self) -> None:
        self.completed = 0

    def step_reference(self, _packet: object) -> dict[str, float]:
        self.completed += 1
        return {
            "base_height_m": 0.76,
            "base_tilt_rad": 0.01,
            "arm_tracking_rmse_rad": 0.1,
            "max_abs_torque_nm": 1.0,
        }


def test_live_balanced_consumer_accepts_fresh_packet_and_rejects_stale() -> None:
    packet = load_reference_packets(BUNDLE)[0]
    summary = validate_reference_terms(packet)
    fresh = rebase_reference_packet_time(
        packet,
        first_control_index=summary["control_index"],
        first_control_monotonic_ns=time.monotonic_ns() - 1_000_000,
    )
    report = run_balanced_upper_body_reference_sequence(
        controller=_FakeBalancedController(),  # type: ignore[arg-type]
        packets=[fresh],
        steps=1,
        maximum_age_ns=100_000_000,
    )
    assert report["passed"] is True
    stale = rebase_reference_packet_time(
        packet,
        first_control_index=summary["control_index"],
        first_control_monotonic_ns=time.monotonic_ns() - 200_000_000,
    )
    with pytest.raises(RuntimeError, match="stale"):
        run_balanced_upper_body_reference_sequence(
            controller=_FakeBalancedController(),  # type: ignore[arg-type]
            packets=[stale],
            steps=1,
            maximum_age_ns=100_000_000,
        )


def test_live_balanced_consumer_rejects_noncontiguous_real_packets() -> None:
    packets = load_reference_packets(BUNDLE)
    first = validate_reference_terms(packets[0])
    rebased = [
        rebase_reference_packet_time(
            packet,
            first_control_index=first["control_index"],
            first_control_monotonic_ns=10_000_000_000,
        )
        for packet in (packets[0], packets[2])
    ]
    with pytest.raises(RuntimeError, match="lost contiguous"):
        run_balanced_upper_body_reference_sequence(
            controller=_FakeBalancedController(),  # type: ignore[arg-type]
            packets=rebased,
            steps=2,
        )
