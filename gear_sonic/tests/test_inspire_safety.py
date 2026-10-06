import numpy as np
import pytest

from gear_sonic.utils.hand_control.inspire.config import load_mapping
from gear_sonic.utils.hand_control.inspire.contract import (
    ContractError,
    HandStatus,
    InspireHandCommand,
)
from gear_sonic.utils.hand_control.inspire.safety import CommandGuard, SafetyConfig


def command(sequence, timestamp, value=0.0):
    values = np.full(6, value, dtype=np.float32)
    return InspireHandCommand(sequence, timestamp, values, values)


def test_guard_requires_explicit_arm_and_rejects_replay():
    guard = CommandGuard(load_mapping())
    with pytest.raises(ContractError, match="not armed"):
        guard.accept(command(1, 1_000), now_ns=1_000)
    guard.arm()
    guard.accept(command(1, 1_000), now_ns=1_000)
    with pytest.raises(ContractError, match="sequence must increase"):
        guard.accept(command(1, 2_000), now_ns=2_000)
    assert guard.status is HandStatus.FAULT


def test_guard_rejects_range_slew_and_stale_commands():
    guard = CommandGuard(load_mapping(), SafetyConfig(command_timeout_ns=100))
    guard.arm()
    with pytest.raises(ContractError, match="outside"):
        guard.accept(command(1, 1_000, value=-0.2), now_ns=1_000)

    guard.clear_fault()
    guard.arm()
    guard.accept(command(1, 1_000, value=0.0), now_ns=1_000)
    assert guard.poll(now_ns=1_101) is HandStatus.STALE

    guard.clear_fault()
    guard.arm()
    guard.accept(command(1, 1_000_000_000, value=0.0), now_ns=1_000_000_000)
    with pytest.raises(ContractError, match="slew"):
        guard.accept(command(2, 1_010_000_000, value=0.5), now_ns=1_010_000_000)


def test_disarm_preserves_replay_history_until_explicit_fault_clear():
    guard = CommandGuard(load_mapping())
    guard.arm()
    guard.accept(command(7, 1_000_000_000), now_ns=1_000_000_000)
    guard.disarm()
    assert guard.last_command is not None and guard.last_command.sequence == 7
    guard.arm()
    with pytest.raises(ContractError, match="sequence must increase"):
        guard.accept(command(7, 1_010_000_000), now_ns=1_010_000_000)
